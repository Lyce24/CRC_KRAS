#!/usr/bin/env python3
"""Build reviews/v4 — leak-proof, ~30-minute single-reader validation.

SUPERSEDES reviews/v3, whose handoff had a blinding no-go: absolute symlinks
recoverable via readlink, a near-perfect cohort signature in file extensions
(.tiff = SurGen/RIH-fixed only), and raw WSI metadata / label images carrying
identifiers. v4 removes the entire channel by handing the reader RENDERED
JPEG IMAGES only:

    per case:  1 whole-tissue overview  +  6 systematic 1.0 x 1.0 mm fields
               at ~1 um/px (10x-equivalent), rendered fresh through PIL, so
               no path, extension, EXIF, vendor metadata, label or macro
               image exists anywhere in the reader's material.

MEASUREMENT DECLARATION. This is a REGION-SHEET read, not a whole-slide read:
gland formation is judged from the six fields, mucin extent from overview
plus fields. That is a declared limitation — sampled fields can miss focal
mucin — and it biases the primary endpoint toward the null, not away from it.

SCOPE. 60 cases, two ordinal fields each, ~20-30 s/case: about 25-35 minutes.
Fewer cases than v3 (120), because the construct-validation question expects
a LARGE effect if p17 truly is extracellular mucin; 60 balanced cases detect
Kendall tau >= ~0.25 at 80% power, ample for that purpose.

DESIGN FIXES carried over from the v3 review verdict, all implemented here:
  * tie-preserving exposure groups (absent p17=0 / positive-low /
    positive-high at the within-cohort median of positives) replace
    rank-split tertiles that scattered identical zeros;
  * the 18 patients whose tiles appeared in ANY prior montage packet are
    excluded from the sampling frame (reader-exposure hygiene);
  * multi-slide patients are excluded, so no multi-slide scoring rule is
    needed;
  * a frozen ANALYSIS_PLAN.md specifies direction, coding, blocked
    permutation inference, effect measures, tie/missing/multiplicity/
    stopping rules and the bootstrap unit BEFORE reading, and drops the
    v3 "attenuation/mediation" analysis as not estimable at this n;
  * fixed completion target with a disclosed-early-stop rule replaces the
    "any prefix is unbiased" overclaim;
  * reviewer identity, date, software, experience, elapsed time and a
    blinding attestation are collected in reviewer_info.csv;
  * a packet receipt hashes the builder, every frozen input, every
    deliverable and every rendered image.

Read-only over frozen inputs; writes only under reviews/v4.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import paths  # noqa: E402
from tools.build_reviews_v2 import (  # noqa: E402
    CANONICAL_SLIDE_DIRS,
    SLIDE_ROOT,
    load_case_frame,
)

OUTPUT = REPO / "reviews" / "v4"
SELECTION_SEED = 20260821
PER_GROUP = 20  # absent / positive-low / positive-high -> 60 cases
N_FIELDS = 6
FIELD_MM = 1.0
FIELD_PX = 1024
OVERVIEW_MAX = 2200
MONTAGE_KEYS = [
    Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim4_cap8192_corrected_v2_20260820/review_bundles/k32/base/unblinding_key_DO_NOT_SHARE.csv"),
    Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim4_cap8192_corrected_v2_20260820/review_bundles/k32/attention_addendum/unblinding_key_DO_NOT_SHARE.csv"),
    Path("/mnt/wsl/oceanpath-hot/outputs/aim1/e3b/montages/k32/KEY_do_not_open_before_review.csv"),
    Path("/mnt/wsl/oceanpath-hot/outputs/aim1/e3b/montages/k32/KEY_M11_followup_do_not_open_before_review.csv"),
]
FORM_COLUMNS = [
    "case_id",
    "assessable",
    "extracellular_mucin_extent",
    "gland_formation",
    "note_if_unusual",
]
REVIEWER_COLUMNS = [
    "reviewer_id",
    "review_date",
    "viewer_software",
    "years_experience_gi_pathology",
    "elapsed_minutes",
    "blinding_attestation",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── sampling frame ───────────────────────────────────────────────────────────
def previously_exposed_patients() -> set[str]:
    seen: set[str] = set()
    for key in MONTAGE_KEYS:
        seen |= set(pd.read_csv(key)["patient_id"].astype(str))
    return seen


def build_frame() -> pd.DataFrame:
    frame = load_case_frame()
    manifest = pd.read_csv(paths.DEV_MANIFEST, low_memory=False)
    mpp = manifest.drop_duplicates("slide_id")[["slide_id", "native_mpp"]]
    frame = frame[frame["slide_ids"].apply(len) == 1].copy()
    frame["slide_id"] = frame["slide_ids"].str[0]
    frame = frame.merge(mpp, on="slide_id", validate="one_to_one")
    exposed = previously_exposed_patients()
    frame = frame[~frame["patient_id"].astype(str).isin(exposed)].reset_index(drop=True)
    return frame


def assign_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Tie-preserving exposure groups: absent / positive-low / positive-high."""
    work = frame.copy()
    work["p17_group"] = "absent"
    positive = work["p17_abundance"] > 0
    for _cohort, block in work[positive].groupby("cohort"):
        median = float(block["p17_abundance"].median())
        high = block.index[block["p17_abundance"] > median]
        low = block.index[block["p17_abundance"] <= median]
        work.loc[high, "p17_group"] = "positive_high"
        work.loc[low, "p17_group"] = "positive_low"
    return work


def select_cases(frame: pd.DataFrame, per_group: int) -> pd.DataFrame:
    rng = np.random.default_rng(SELECTION_SEED)
    picks: list[int] = []
    audit: list[dict[str, Any]] = []
    for group in ("absent", "positive_low", "positive_high"):
        per_class = per_group // 2
        for kras in ("mutant", "wild_type"):
            block = frame[(frame["p17_group"] == group) & (frame["kras"] == kras)]
            by_cohort = {
                cohort: rng.permutation(sub.sort_values("patient_id").index.to_numpy())
                for cohort, sub in block.groupby("cohort", sort=True)
            }
            chosen: list[int] = []
            position = 0
            while len(chosen) < per_class and any(position < len(v) for v in by_cohort.values()):
                for cohort in sorted(by_cohort):
                    if len(chosen) >= per_class:
                        break
                    if position < len(by_cohort[cohort]):
                        chosen.append(int(by_cohort[cohort][position]))
                position += 1
            picks.extend(chosen)
            audit.append(
                {"p17_group": group, "kras": kras, "available": int(len(block)), "selected": int(len(chosen))}
            )
    sample = frame.loc[picks].copy()
    order = rng.permutation(len(sample))
    sample = sample.iloc[order].reset_index(drop=True)
    sample.insert(0, "case_id", [f"W{i + 1:03d}" for i in range(len(sample))])
    sample.attrs["audit"] = audit
    return sample


# ── rendering ────────────────────────────────────────────────────────────────
def resolve_slide(slide_id: str) -> Path:
    for directory in CANONICAL_SLIDE_DIRS:
        for suffix in (".svs", ".tiff", ".tif", ".scn"):
            candidate = SLIDE_ROOT / directory / f"{slide_id}{suffix}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"{slide_id} not found in canonical directories")


def tissue_mask(thumb: np.ndarray) -> np.ndarray:
    spread = thumb.max(axis=2).astype(int) - thumb.min(axis=2).astype(int)
    value = thumb.max(axis=2)
    return (spread > 18) & (value < 242) & (value > 30)


def field_centers(mask: np.ndarray, window: int, n_fields: int) -> list[tuple[int, int]]:
    """Deterministic: densest tissue window inside each cell of a 3x2 grid."""
    density = mask.astype(np.float32)
    integral = np.pad(density, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def box(r0: int, c0: int) -> float:
        r1, c1 = r0 + window, c0 + window
        return float(integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0])

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows) or not len(cols):
        return []
    r_lo, r_hi = int(rows[0]), int(rows[-1])
    c_lo, c_hi = int(cols[0]), int(cols[-1])
    grid_rows, grid_cols = (3, 2) if (r_hi - r_lo) >= (c_hi - c_lo) else (2, 3)
    centers: list[tuple[int, int, float]] = []
    for gr in range(grid_rows):
        for gc in range(grid_cols):
            a0 = r_lo + (r_hi - r_lo) * gr // grid_rows
            a1 = r_lo + (r_hi - r_lo) * (gr + 1) // grid_rows
            b0 = c_lo + (c_hi - c_lo) * gc // grid_cols
            b1 = c_lo + (c_hi - c_lo) * (gc + 1) // grid_cols
            best, best_score = None, -1.0
            step = max(1, window // 4)
            for r0 in range(a0, max(a0 + 1, a1 - window), step):
                for c0 in range(b0, max(b0 + 1, b1 - window), step):
                    score = box(min(r0, mask.shape[0] - window - 1), min(c0, mask.shape[1] - window - 1))
                    if score > best_score:
                        best_score, best = score, (r0 + window // 2, c0 + window // 2)
            if best is not None and best_score > 0.05 * window * window:
                centers.append((*best, best_score))
    centers.sort(key=lambda t: -t[2])
    return [(r, c) for r, c, _ in centers[:n_fields]]


def render_case(slide_path: Path, native_mpp: float, out_dir: Path, case_id: str) -> list[Path]:
    import openslide
    from PIL import Image

    produced: list[Path] = []
    slide = openslide.OpenSlide(str(slide_path))
    try:
        thumb = slide.get_thumbnail((OVERVIEW_MAX, OVERVIEW_MAX)).convert("RGB")
        overview_path = out_dir / f"{case_id}_overview.jpg"
        thumb.save(overview_path, "JPEG", quality=88)
        produced.append(overview_path)

        mask = tissue_mask(np.asarray(thumb))
        scale = slide.dimensions[0] / thumb.size[0]  # level0 px per thumb px
        field_level0 = FIELD_MM * 1000.0 / native_mpp
        window_thumb = max(4, int(round(field_level0 / scale)))
        centers = field_centers(mask, window_thumb, N_FIELDS)

        target_downsample = max(1.0, (FIELD_MM * 1000.0 / FIELD_PX) / native_mpp)
        level = slide.get_best_level_for_downsample(target_downsample * 1.01)
        level_downsample = slide.level_downsamples[level]
        read_px = int(round(field_level0 / level_downsample))
        for index, (r_thumb, c_thumb) in enumerate(centers, start=1):
            x0 = int(c_thumb * scale - field_level0 / 2)
            y0 = int(r_thumb * scale - field_level0 / 2)
            x0 = int(np.clip(x0, 0, max(0, slide.dimensions[0] - field_level0)))
            y0 = int(np.clip(y0, 0, max(0, slide.dimensions[1] - field_level0)))
            region = slide.read_region((x0, y0), level, (read_px, read_px)).convert("RGB")
            region = region.resize((FIELD_PX, FIELD_PX), Image.LANCZOS)
            field_path = out_dir / f"{case_id}_f{index}.jpg"
            region.save(field_path, "JPEG", quality=90)
            produced.append(field_path)
    finally:
        slide.close()
    return produced


# ── packet writing ───────────────────────────────────────────────────────────
def write_reader_docs(reader: Path, sample: pd.DataFrame) -> None:
    with open(reader / "scoring_form.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORM_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for case_id in sample["case_id"]:
            writer.writerow({c: (case_id if c == "case_id" else "") for c in FORM_COLUMNS})
    with open(reader / "reviewer_info.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEWER_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerow({c: "" for c in REVIEWER_COLUMNS})

    (reader / "INSTRUCTIONS.md").write_text(
        f"""# Case scoring — {len(sample)} cases, about 25–35 minutes

Thank you. Each case is a folder-free set of JPEG images in `images/`:

* `Wxxx_overview.jpg` — the whole tissue section at low power
* `Wxxx_f1.jpg` … `Wxxx_f6.jpg` — six systematically sampled fields,
  each 1.0 × 1.0 mm at roughly 10× (about 1 micrometre per pixel)

Please score from these images only. This is a sampled-field read: judge
what the images show and do not try to infer what was not sampled.

## The two questions per case

**1. `extracellular_mucin_extent`** — the share of TUMOUR area occupied by
extracellular mucin (pools, lakes, stromal mucin — not intracytoplasmic
mucin or goblet cells). Judge from the overview first, confirm on fields.
Answer with exactly one of:

* `none`
* `focal_lt10`
* `moderate_10_50`
* `extensive_gt50`

**2. `gland_formation`** — the share of tumour forming recognisable glands,
judged from the six fields (the usual grading estimate). Answer with exactly
one of:

* `gt95`
* `pct50_95`
* `lt50`

**`assessable`** — `yes`, or `no` when the images are unreadable or show no
tumour. If `no`, leave the two scoring columns blank and move on.

**`note_if_unusual`** — optional; leave blank unless something needs
flagging.

## Ground rules

* The target is all {len(sample)} cases in one sitting. If you must stop
  early, stop between cases and tell the coordinator how far you got; an
  early stop is recorded and disclosed, and partial results remain usable.
* Case identifiers are random and carry no information. Please do not
  attempt to identify a case, institution or patient, and do not ask what
  these features are expected to predict.
* Before starting, fill every field of `reviewer_info.csv`. For
  `blinding_attestation`, enter exactly: `confirmed_no_key_or_source_access`
  after confirming you have received only this folder and the images.
* Record your total time in `reviewer_info.csv` when done.
"""
    )


def write_analysis_plan(root: Path) -> None:
    (root / "ANALYSIS_PLAN.md").write_text(
        """# Frozen analysis plan — reviews/v4 (written before any case is read)

## Units, coding, direction

* Unit of analysis: patient (one slide per patient by design).
* Mucin score: none=0, focal_lt10=1, moderate_10_50=2, extensive_gt50=3.
* Gland score: lt50=1, pct50_95=2, gt95=3.
* Declared directions: higher frozen p17 abundance -> higher mucin score;
  higher frozen p28 abundance -> higher gland score.

## Primary endpoint

Kendall tau-b between the mucin score and CONTINUOUS frozen p17 abundance
(tie-robust; the exposure groups exist for sampling, not analysis).
Inference: permutation test with 20,000 permutations of the pathology score
within cohort x KRAS blocks, one-sided in the declared direction.
Effect presentation: tau-b with a patient-bootstrap 95% CI (2,000 draws,
seed 20260817), plus the absent-versus-positive-high difference in mean
mucin score with a bootstrap 95% CI as the interpretable contrast.

## Secondary endpoint

Identical machinery for gland score versus continuous p28 abundance.

## Rules fixed in advance

* Non-assessable cases (assessable=no) are excluded from both endpoints;
  their count is reported, and if they exceed 10% of read cases the primary
  result is additionally reported on all cases with non-assessable scored
  as the modal category (sensitivity).
* Any blank scoring cell on an assessable case is a query back to the
  reader BEFORE unblinding, never an imputation.
* Multiplicity: one primary and one secondary test, each one-sided at 0.05
  and labeled as such; everything else in the report is descriptive and
  carries no p-values.
* Stopping: fixed target of 60 cases. An early stop is permitted (the
  reader is blind to all outcomes, so stopping cannot depend on results),
  is disclosed with the count completed, and the analysis proceeds on the
  completed cases without reweighting.
* No attenuation or mediation analysis: with 60 cases selected jointly on
  KRAS and p17, such a model is not estimable and is explicitly out of
  scope; it would require the full-cohort design that could not be staffed.
* Sampling is balanced on exposure and KRAS: no prevalence statistic may be
  computed from this set, and effect sizes are not transportable to an
  unselected series.
* Measurement is a sampled-field read (overview + six 1 mm fields), which
  can miss focal mucin; this biases the primary endpoint toward the null.
* Reliability (between- or within-reader) is not estimated and remains a
  stated limitation.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--per-group", type=int, default=PER_GROUP)
    args = parser.parse_args()
    root: Path = args.output
    if root.exists():
        raise SystemExit(f"append-only: {root} already exists")

    frame = assign_groups(build_frame())
    sample = select_cases(frame, args.per_group)

    reader = root / "FOR_PATHOLOGIST"
    images = reader / "images"
    images.mkdir(parents=True)
    write_reader_docs(reader, sample)
    write_analysis_plan(root)

    rendered: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        slide_path = resolve_slide(row["slide_id"])
        produced = render_case(slide_path, float(row["native_mpp"]), images, row["case_id"])
        for path in produced:
            rendered.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
        print(f"rendered {row['case_id']}: {len(produced)} images", flush=True)

    keys = root / "KEYS_DO_NOT_DISTRIBUTE"
    keys.mkdir(parents=True)
    key = sample.copy()
    key["slide_ids"] = key["slide_ids"].apply(lambda v: ";".join(v))
    key["source_path"] = key["slide_id"].apply(lambda s: str(resolve_slide(s)))
    key.to_csv(keys / "case_key.csv", index=False)
    pd.DataFrame(sample.attrs["audit"]).to_csv(keys / "selection_audit.csv", index=False)
    pd.DataFrame(sorted(previously_exposed_patients()), columns=["patient_id"]).to_csv(
        keys / "excluded_previously_exposed_patients.csv", index=False
    )
    image_manifest = pd.DataFrame(rendered)
    image_manifest.to_csv(keys / "image_manifest.csv", index=False)
    (keys / "README.md").write_text(
        "# DO NOT DISTRIBUTE\n\nMaps blinded case ids to patients, cohorts, molecular "
        "status, p17/p28 abundance, held-out WSI logits and source slide paths. Open "
        "only after the completed form is returned.\n"
    )

    created = datetime.now(timezone.utc).isoformat()
    audit = pd.DataFrame(sample.attrs["audit"])
    (root / "README_COORDINATOR.md").write_text(
        f"""# reviews/v4 — leak-proof one-sitting validation ({len(sample)} cases)

Supersedes reviews/v3 (blinding no-go: symlink targets, extension signature,
WSI metadata). The reader receives RENDERED JPEGS ONLY — no slide file, no
path, no metadata, no label or macro image exists in their material, so
there is nothing to leak.

## Hand over exactly one folder

`FOR_PATHOLOGIST/` — instructions, blank `scoring_form.csv`,
`reviewer_info.csv`, and `images/` ({len(image_manifest)} JPEGs:
one overview + up to {N_FIELDS} fields per case).

Never hand over `KEYS_DO_NOT_DISTRIBUTE/`.

## Pre-handoff QC (coordinator, ~5 minutes)

Skim the {len(sample)} overview images for any burned-in text on the glass
(rare, but a scanned edge can include handwriting). If found, delete that
case's images and record the case id in the key as withdrawn; do not
re-render or substitute.

## Sample

{len(sample)} single-slide primaries; {args.per_group} per tie-preserving
p17 exposure group (absent / positive-low / positive-high at the
within-cohort median of positives), KRAS-balanced within group, cohorts
round-robin, seed {SELECTION_SEED}. The 18 patients whose tiles appeared in
any prior montage packet are excluded from the frame, as are multi-slide
patients (so no multi-slide scoring rule is needed).

* by cohort: {sample.groupby('cohort').size().to_dict()}
* by KRAS: {sample.groupby('kras').size().to_dict()}
* by exposure group: {sample.groupby('p17_group').size().to_dict()}

## Analysis

`ANALYSIS_PLAN.md` is frozen as of {created}, before any case is read.
Primary: Kendall tau-b, mucin score versus continuous p17, blocked
permutation inference. Secondary: gland score versus p28. No mediation or
attenuation analysis at this n. No prevalence statistics from this sample.

## Selection audit

{audit.to_string(index=False)}
"""
    )

    produced_docs = [
        Path(__file__),
        reader / "INSTRUCTIONS.md",
        reader / "scoring_form.csv",
        reader / "reviewer_info.csv",
        root / "ANALYSIS_PLAN.md",
        root / "README_COORDINATOR.md",
        keys / "case_key.csv",
        keys / "selection_audit.csv",
        keys / "excluded_previously_exposed_patients.csv",
        keys / "image_manifest.csv",
    ]
    frozen_inputs = [
        Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim4_cap8192_corrected_v2_20260820/profiles/patient_profiles_k32.parquet"),
        paths.DEV_MANIFEST,
        *MONTAGE_KEYS,
    ]
    receipt = {
        "schema_version": 1,
        "created_utc": created,
        "supersedes": "reviews/v3 (blinding no-go)",
        "cases": int(len(sample)),
        "images": int(len(image_manifest)),
        "image_manifest_sha256": sha256(keys / "image_manifest.csv"),
        "deliverables": [{"path": str(p), "sha256": sha256(p)} for p in produced_docs],
        "frozen_inputs": [{"path": str(p), "sha256": sha256(p)} for p in frozen_inputs],
        "blinding": (
            "reader material is rendered JPEG only: no slide file, path, extension "
            "signature, EXIF/vendor metadata, label or macro image is present"
        ),
    }
    (keys / "packet_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    summary = {
        "cases": int(len(sample)),
        "images": int(len(image_manifest)),
        "by_cohort": sample.groupby("cohort").size().to_dict(),
        "by_kras": sample.groupby("kras").size().to_dict(),
        "by_group": sample.groupby("p17_group").size().to_dict(),
    }
    (root / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
