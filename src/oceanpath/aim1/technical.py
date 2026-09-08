"""Label-independent per-slide technical descriptors (Aim1_Setup.md §4.5).

The UMAP island audit (``univ1_umap_separation_audit``) established that the
UNI-v1 slide-mean representation recognizes three technical domains, but it
only covered SurGen and RIH (1,389 slides). Aim 1's development population is
TCGA + SR386 and its externals include CPTAC, so the same descriptors are
recomputed here for **every** feature-backed slide from the frozen artifacts
each slide already has: its feature H5 (coordinate attributes) and its HEST
thumbnail. Nothing in this module reads a KRAS label.

Three groups of variables come out:

``continuous``
    native_mpp, patch_count, tissue_area_mm2, slide_area_mm2,
    tissue_grid_occupancy, thumbnail_hue_median, thumbnail_tissue_pixels

``derived classes``
    mpp_bin              exact native MPP, 4 dp, as a string label
    section_size_class   large_section | small_fragment, from a single frozen
                         PHYSICAL tissue-area threshold (comparable across
                         cohorts because every slide was resampled to a common
                         0.5 um/px before tiling)
    color_class          main_stain | cool_shifted, a within-subcohort robust
                         low-hue tail using one shared cutoff k
    technical_class      the per-subcohort composite of §4.5: TCGA/CPTAC use
                         the MPP bin, SR386 the color class, SR1482 the
                         section-size class, RIH both

``audit``
    the extraction-policy check that makes the cross-cohort biological claim
    admissible: every slide sampled with ``sampling_mode == exact_mpp`` at
    ``target_mpp == 0.5``.

The section-size threshold and the colour cutoff are *fitted* — but only
against technical ground truth (SurGen biopsy wording, the RIH source biopsy
field, and the frozen SR386 island), never against KRAS — and then frozen into
``summary.json`` so downstream phases adopt them verbatim.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from oceanpath.aim1 import paths

# One tile is 256 px at the common target resolution of 0.5 um/px.
TILE_UM = 256 * 0.5
TILE_AREA_MM2 = (TILE_UM / 1000.0) ** 2

# Candidate grid for the frozen physical section-size threshold (mm^2 of
# tissue). The empirical per-cohort optima of the island audit sit at 81.5
# (SR1482) and 59.8 (RIH) mm^2; one shared cutoff is selected on the pooled
# technical ground truth.
SECTION_THRESHOLD_GRID_MM2 = np.arange(20.0, 141.0, 0.5)

# Robust within-subcohort low-hue tail: hue < median - K_HUE * 1.4826 * MAD.
# K_HUE is chosen so SR386 reproduces the frozen 48-slide cool-stain island.
HUE_K_GRID = np.arange(0.5, 6.01, 0.05)
SR386_COOL_ISLAND_N = 48


# ── Raw descriptors ───────────────────────────────────────────────────────────


def _thumbnail_path(slide_id: str) -> Path:
    for directory in paths.THUMBNAIL_DIRS:
        candidate = directory / f"{slide_id}.jpg"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No thumbnail for {slide_id} in {list(paths.THUMBNAIL_DIRS)}")


# Tissue-mask cascade for the thumbnail hue. The first rule is the island
# audit's, kept identical so SurGen/RIH hues stay comparable with the frozen
# assignments; the looser rules only ever fire on the washed-out thumbnails
# some TCGA slides carry, and the rule that fired is recorded per slide.
HUE_MASK_RULES: tuple[tuple[str, float, float], ...] = (
    ("audit", 0.94, 0.06),
    ("relaxed", 0.90, 0.0),
    ("permissive", 0.97, 0.0),
)
HUE_MIN_PIXELS = 50


def thumbnail_hue(path: Path) -> tuple[float, int, str]:
    """Median tissue hue of a slide, from its frozen HEST thumbnail.

    Returns ``(hue, mask_pixels, rule)``. A thumbnail with no usable tissue
    pixels under any rule (a near-blank strip) yields ``nan`` and rule
    ``"unmeasured"`` rather than failing the build — the slide keeps every
    other technical descriptor, and colour simply is not measurable for it.
    """
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((192, 192), resample=Image.Resampling.BICUBIC, reducing_gap=2.0)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = (maximum - minimum) / (maximum + 1e-6)
    for rule, value_max, saturation_min in HUE_MASK_RULES:
        mask = (maximum < value_max) & (saturation > saturation_min)
        if int(mask.sum()) >= HUE_MIN_PIXELS:
            return float(np.median(hsv[:, :, 0][mask])), int(mask.sum()), rule
    return float("nan"), 0, "unmeasured"


def slide_descriptors(slide_ids: Sequence[str], progress_every: int = 250) -> pd.DataFrame:
    """Read H5 coordinate attributes and thumbnail statistics per slide."""
    rows: list[dict[str, Any]] = []
    for index, slide_id in enumerate(map(str, slide_ids), start=1):
        h5_path = paths.PINNED_FEATURE_DIR / f"{slide_id}.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(f"Missing feature H5 for {slide_id}")
        with h5py.File(h5_path, "r") as handle:
            coords = handle["coords"]
            features = handle["features"]
            if len(coords) != len(features):
                raise ValueError(f"Coordinate/feature count mismatch: {slide_id}")
            attrs = coords.attrs
            row = {
                "slide_id": slide_id,
                "patch_count": int(len(features)),
                "level0_width": int(attrs["level0_width"]),
                "level0_height": int(attrs["level0_height"]),
                "native_mpp": float(attrs["level0_mpp"]),
                "patch_size_level0": int(attrs["patch_size_level0"]),
                "sampling_mode": str(attrs["sampling_mode"]),
                "target_mpp": float(attrs["target_mpp"]),
                "effective_target_mpp": float(attrs["effective_target_mpp"]),
            }
        hue, hue_pixels, hue_rule = thumbnail_hue(_thumbnail_path(slide_id))
        row["thumbnail_hue_median"] = hue
        row["thumbnail_tissue_pixels"] = hue_pixels
        row["thumbnail_hue_rule"] = hue_rule
        rows.append(row)
        if progress_every and index % progress_every == 0:
            print(f"  descriptors {index:,}/{len(slide_ids):,}", flush=True)

    frame = pd.DataFrame(rows)
    frame["slide_area_mm2"] = (
        frame["level0_width"]
        * frame["level0_height"]
        * frame["native_mpp"] ** 2
        / 1e6  # um^2 -> mm^2
    )
    # Tissue actually tiled, in physical units. Comparable across scanners
    # because every slide was resampled to the same target resolution.
    frame["tissue_area_mm2"] = frame["patch_count"] * TILE_AREA_MM2
    frame["tissue_grid_occupancy"] = frame["tissue_area_mm2"] / frame["slide_area_mm2"]
    return frame


def audit_extraction_policy(frame: pd.DataFrame) -> dict[str, Any]:
    """The §4.5 physical-resolution check, as a reportable record.

    Raises if any slide was not resampled to the common target resolution —
    without this, a cross-cohort morphology claim would be confounded by
    scanner scale rather than tested against it.
    """
    modes = sorted(frame["sampling_mode"].unique().tolist())
    targets = sorted(frame["target_mpp"].round(6).unique().tolist())
    if modes != ["exact_mpp"]:
        raise ValueError(f"Mixed sampling modes in the feature store: {modes}")
    if targets != [0.5]:
        raise ValueError(f"Mixed target MPP in the feature store: {targets}")
    deviation = (frame["effective_target_mpp"] - 0.5).abs()
    return {
        "sampling_mode": modes,
        "target_mpp": targets,
        "n_slides": int(len(frame)),
        # Integer level-0 patch sizes quantize the achievable resolution; this
        # is the worst-case departure from exactly 0.5 um/px anywhere.
        "max_abs_effective_mpp_deviation": float(deviation.max()),
        "native_mpp_values": {
            f"{value:.4f}": int(count)
            for value, count in frame["native_mpp"].round(4).value_counts().sort_index().items()
        },
    }


# ── Derived technical classes ─────────────────────────────────────────────────


def _balanced_accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    positives = truth == 1
    negatives = ~positives
    if not positives.any() or not negatives.any():
        return float("nan")
    sensitivity = float(predicted[positives].mean())
    specificity = float(1.0 - predicted[negatives].mean())
    return 0.5 * (sensitivity + specificity)


def fit_section_size_threshold(
    frame: pd.DataFrame, truth: pd.Series
) -> tuple[float, dict[str, Any]]:
    """Choose one physical tissue-area cutoff separating fragments from sections.

    ``truth`` is the technical ground truth (1 = biopsy/small fragment) and is
    defined only where a source field records it; slides without it are
    ignored for fitting but still classified afterwards.
    """
    known = truth.notna()
    area = frame.loc[known, "tissue_area_mm2"].to_numpy(dtype=float)
    labels = truth[known].to_numpy(dtype=float)
    scores = [
        _balanced_accuracy(labels, (area < threshold).astype(float))
        for threshold in SECTION_THRESHOLD_GRID_MM2
    ]
    best = int(np.nanargmax(scores))
    threshold = float(SECTION_THRESHOLD_GRID_MM2[best])
    record = {
        "threshold_mm2": threshold,
        "threshold_patches": int(round(threshold / TILE_AREA_MM2)),
        "n_labelled": int(known.sum()),
        "balanced_accuracy": float(scores[best]),
        "sensitivity": float((area[labels == 1] < threshold).mean()),
        "specificity": float((area[labels == 0] >= threshold).mean()),
        "per_subcohort": {},
    }
    # One shared threshold is only defensible if it also works inside each
    # cohort that has ground truth; report that, do not average it away.
    for subcohort, block in frame.loc[known].groupby(frame.loc[known, "subcohort"]):
        block_truth = truth[block.index].to_numpy(dtype=float)
        predicted = (block["tissue_area_mm2"].to_numpy(dtype=float) < threshold).astype(float)
        record["per_subcohort"][str(subcohort)] = {
            "n": int(len(block)),
            "n_positive": int(block_truth.sum()),
            "balanced_accuracy": _balanced_accuracy(block_truth, predicted),
        }
    return threshold, record


def fit_hue_cutoff(frame: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    """Pick the robust-tail multiplier that reproduces SR386's cool island."""
    sr386 = frame[frame["subcohort"] == "SR386"]
    hue = sr386["thumbnail_hue_median"].dropna().to_numpy(dtype=float)
    median = float(np.median(hue))
    mad = float(np.median(np.abs(hue - median)))
    counts = [int((hue < median - k * 1.4826 * mad).sum()) for k in HUE_K_GRID]
    errors = [abs(count - SR386_COOL_ISLAND_N) for count in counts]
    best = int(np.argmin(errors))
    return float(HUE_K_GRID[best]), {
        "k": float(HUE_K_GRID[best]),
        "sr386_flagged": counts[best],
        "sr386_island_target": SR386_COOL_ISLAND_N,
        "sr386_median_hue": median,
        "sr386_mad_hue": mad,
    }


def _robust_low_tail(values: pd.Series, k: float) -> tuple[pd.Series, float]:
    """Low-tail flag and the cutoff, computed on the measurable values only."""
    known = values.dropna()
    if known.empty:
        return pd.Series(False, index=values.index), float("nan")
    median = float(np.median(known))
    mad = float(np.median(np.abs(known - median)))
    cutoff = median - k * 1.4826 * mad
    if mad == 0.0:
        return pd.Series(False, index=values.index), cutoff
    return values.lt(cutoff).fillna(False), cutoff


def technical_ground_truth(frame: pd.DataFrame) -> pd.Series:
    """1 = biopsy/small fragment, 0 = resection/full section, NA = unrecorded.

    SurGen records the procedure in free-text site wording; RIH carries an
    explicit source-biopsy flag in the reconciliation workbook (both reused
    from the frozen island audit table). TCGA and CPTAC record neither, so
    their slides are classified by the fitted threshold without contributing
    to the fit.
    """
    truth = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    if "sr1482_biopsy_text" in frame:
        surgen = frame["subcohort"].isin(["SR386", "SR1482"]) & frame["sr1482_biopsy_text"].notna()
        truth[surgen] = frame.loc[surgen, "sr1482_biopsy_text"].astype(float)
    if "rih_source_biopsy_flag" in frame:
        rih = frame["subcohort"].eq("RIH-Colon") & frame["rih_source_biopsy_flag"].notna()
        truth[rih] = frame.loc[rih, "rih_source_biopsy_flag"].astype(float)
    return truth


# Native-MPP bins. Scanner calibration makes the raw values near-continuous
# (0.2456 ... 0.2533 are all nominal-40x Aperio), so binning by acquisition
# regime rather than by exact float keeps the strata interpretable and large
# enough to balance folds on. Edges are frozen here, not fitted.
MPP_BIN_EDGES: tuple[tuple[float, str], ...] = (
    (0.21, "mpp~0.19"),  # RIH Versa-class fine scans
    (0.24, "mpp~0.23"),  # TCGA calibration cluster
    (0.30, "mpp~0.25"),  # nominal 40x (TCGA/SurGen/CPTAC)
    (0.45, "mpp~0.38"),  # RIH intermediate
    (float("inf"), "mpp~0.50"),  # nominal 20x (RIH Aperio)
)


def mpp_bin(value: float) -> str:
    for upper, label in MPP_BIN_EDGES:
        if value < upper:
            return label
    raise ValueError(f"Unbinnable MPP: {value}")


ISLAND_POSITIVE = {"cool_stain_island", "small_fragment_island"}


def _island_agreement(
    frame: pd.DataFrame, subcohort: str, column: str, positive_value: str
) -> dict[str, Any]:
    """How often a recomputed class matches the frozen UMAP island label."""
    block = frame[frame["subcohort"].eq(subcohort) & frame["umap_island"].notna()]
    if block.empty:
        return {"n": 0}
    frozen = block["umap_island"].isin(ISLAND_POSITIVE)
    derived = block[column].eq(positive_value)
    return {
        "n": int(len(block)),
        "frozen_positive": int(frozen.sum()),
        "derived_positive": int(derived.sum()),
        "agreement": float((frozen == derived).mean()),
    }


def derive_classes(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add mpp_bin / section_size_class / color_class / technical_class."""
    out = frame.copy()
    out["mpp_bin"] = out["native_mpp"].map(mpp_bin)

    truth = technical_ground_truth(out)
    threshold, section_fit = fit_section_size_threshold(out, truth)
    out["section_size_class"] = np.where(
        out["tissue_area_mm2"] < threshold, "small_fragment", "large_section"
    )

    k, hue_fit = fit_hue_cutoff(out)
    cool = pd.Series(False, index=out.index)
    hue_thresholds: dict[str, float] = {}
    for subcohort, block in out.groupby("subcohort"):
        flag, cutoff = _robust_low_tail(block["thumbnail_hue_median"], k)
        cool.loc[block.index] = flag
        hue_thresholds[str(subcohort)] = cutoff
    out["color_class"] = np.where(cool, "cool_shifted", "main_stain")
    out.loc[out["thumbnail_hue_median"].isna(), "color_class"] = "unmeasured"

    # §4.5 composite: each subcohort's technical axis, as pre-registered.
    composite = pd.Series("", index=out.index, dtype=object)
    for subcohort, block in out.groupby("subcohort"):
        if subcohort in {"TCGA-COAD", "TCGA-READ", "CPTAC-COAD"}:
            value = block["mpp_bin"]
        elif subcohort == "SR386":
            value = block["color_class"]
        elif subcohort == "SR1482":
            value = block["section_size_class"]
        elif subcohort == "RIH-Colon":
            value = block["mpp_bin"] + "|" + block["section_size_class"]
        else:
            raise ValueError(f"Unknown subcohort {subcohort!r}")
        composite.loc[block.index] = f"{subcohort}:" + value
    out["technical_class"] = composite

    fit = {
        "section_size": section_fit,
        "color": {
            **hue_fit,
            "per_subcohort_hue_threshold": hue_thresholds,
            # Agreement with the frozen island membership is the check that
            # the recomputed rule still names the same SR386 batch.
            "sr386_vs_frozen_island": _island_agreement(
                out, "SR386", "color_class", "cool_shifted"
            ),
        },
        "section_size_vs_frozen_island": {
            subcohort: _island_agreement(out, subcohort, "section_size_class", "small_fragment")
            for subcohort in ("SR1482", "RIH-Colon")
        },
        "class_counts": {
            column: {str(key): int(value) for key, value in out[column].value_counts().items()}
            for column in (
                "mpp_bin",
                "section_size_class",
                "color_class",
                "technical_class",
                "thumbnail_hue_rule",
            )
        },
    }
    return out, fit


def attach_island_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Join the frozen SurGen/RIH island labels where they exist."""
    if not paths.UMAP_AUDIT_ASSIGNMENTS.is_file():
        return frame.assign(
            umap_island=pd.NA, sr1482_biopsy_text=pd.NA, rih_source_biopsy_flag=pd.NA
        )
    audit = pd.read_parquet(paths.UMAP_AUDIT_ASSIGNMENTS)
    keep = ["slide_id", "visible_island", "sr1482_biopsy_text", "rih_source_biopsy_flag"]
    audit = audit[[column for column in keep if column in audit.columns]].rename(
        columns={"visible_island": "umap_island"}
    )
    return frame.merge(audit, on="slide_id", how="left", validate="one_to_one")


# ── Entry point ───────────────────────────────────────────────────────────────


def build(slide_ids: Iterable[str], subcohorts: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full technical table for the given slides plus the frozen fit record."""
    slide_ids = list(slide_ids)
    frame = slide_descriptors(slide_ids)
    frame = frame.merge(
        pd.DataFrame({"slide_id": slide_ids, "subcohort": list(subcohorts)}),
        on="slide_id",
        validate="one_to_one",
    )
    audit = audit_extraction_policy(frame)
    frame = attach_island_audit(frame)
    frame, fit = derive_classes(frame)
    return frame, {"extraction_policy_audit": audit, "derived_classes": fit}


def load() -> pd.DataFrame:
    """Read the frozen technical table written by aim1_phase1."""
    if not paths.TECHNICAL_TABLE.is_file():
        raise SystemExit(
            f"{paths.TECHNICAL_TABLE} not found — run tools/phase1_technical_audit.py build --apply"
        )
    return pd.read_parquet(paths.TECHNICAL_TABLE)


def load_summary() -> dict[str, Any]:
    return json.loads(paths.TECHNICAL_SUMMARY.read_text())
