"""The Aim-1 eligible population and every frozen manifest (§3, §4.2, §5.1).

One eligibility rule governs all four cohorts::

    crc_final_v4.csv
      AND specimen_role == "primary"
      AND kras in {mutant, wild_type}                 (unknown is non-random)
      AND a UNI-v1 feature bag exists in the pinned store

No tumour-site restriction: appendiceal and unspecified-site primaries stay in
(13 patients), because Aim 1's question is whether the KRAS phenotype is
molecularly specific, and site is a challenge-set variable (§5.2) rather than
an eligibility criterion.

Applied to the label source this yields exactly 1,486 patients (604 mutant /
882 wild-type) over 1,642 slides, pooled across all four cohorts. The counts
are hard-checked before anything is written, so a silent change in the label
source or the feature store fails the build instead of shifting the study
population.

All four cohorts train together and are read out as pooled out-of-fold
predictions with a per-cohort breakdown. Aim 1 holds nothing out; Aim 2 owns
transportability with its own leave-one-institution-out design.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from oceanpath.aim1 import paths

MANIFEST_COLUMNS = [
    "slide_id",
    "patient_id",
    "target_label",
    "slide_weight",
    "kras",
    "kras_subvariant",
    "cohort",
    "subcohort",
    "specimen_role",
    # clinical / molecular context (challenge sets, §5.1-5.2)
    "msi_dmmr",
    "braf",
    "nras",
    "ras",
    "tumor_site_group",
    "tumor_site_raw",
    "stage_group_major",
    "age_at_diagnosis",
    "sex",
    # technical context (§4.5, §5.3) — never a model input
    "native_mpp",
    "mpp_bin",
    "patch_count",
    "tissue_area_mm2",
    "tissue_grid_occupancy",
    "thumbnail_hue_median",
    "section_size_class",
    "color_class",
    "technical_class",
    "umap_island",
]

# Columns the balancer stratifies on, and the challenge sets slice by.
BALANCE_COLUMNS = [
    "kras",
    "subcohort",
    "site_class",
    "stage_class",
    "mpp_bin",
    "technical_class",
    "n_slides_class",
]


def load_label_source() -> pd.DataFrame:
    frame = pd.read_csv(paths.LABEL_SOURCE, low_memory=False)
    required = {"output_id", "patient_uid", "kras", "specimen_role", "tumor_site_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{paths.LABEL_SOURCE} is missing columns: {sorted(missing)}")
    return frame


def feature_stems() -> set[str]:
    if not paths.PINNED_FEATURE_DIR.is_dir():
        raise SystemExit(f"Pinned feature store not found: {paths.PINNED_FEATURE_DIR}")
    return {os.path.splitext(name)[0] for name in os.listdir(paths.PINNED_FEATURE_DIR)}


def eligible(source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every slide that satisfies the single Aim-1 eligibility rule."""
    frame = load_label_source() if source is None else source
    stems = feature_stems()
    rows = frame[
        frame["specimen_role"].eq("primary")
        & frame["kras"].isin(["mutant", "wild_type"])
        & frame["output_id"].isin(stems)
    ].copy()
    rows = rows.rename(columns={"output_id": "slide_id", "patient_uid": "patient_id"})
    rows["target_label"] = rows["kras"].eq("mutant").astype(int)
    # Patient weight: every patient contributes 1.0 in total, spread evenly
    # over their slides (§4.3), so multi-block patients cannot outvote others.
    rows["slide_weight"] = 1.0 / rows.groupby("patient_id")["slide_id"].transform("size")
    return rows.sort_values("slide_id").reset_index(drop=True)


def arm(rows: pd.DataFrame, group: str) -> pd.DataFrame:
    """Slice the eligible population into one study arm.

    Aim 1 has a single arm: every eligible patient trains, and cohort is a
    stratification and reporting variable rather than a partition.
    """
    if group == "dev":
        return rows[rows["subcohort"].isin(paths.DEV_SUBCOHORTS)].copy()
    if group in paths.EXTERNAL_SUBCOHORT:
        return rows[rows["subcohort"].eq(paths.EXTERNAL_SUBCOHORT[group])].copy()
    raise ValueError(f"Unknown arm {group!r}")


def restricted(rows: pd.DataFrame) -> pd.DataFrame:
    """MSS/pMMR + BRAF-wild-type restriction (§5.1 primary molecular challenge).

    Unknown MSI or unknown BRAF is *not* wild-type: those patients leave the
    restricted population rather than being assumed negative.
    """
    return rows[rows["msi_dmmr"].eq("MSS/pMMR") & rows["braf"].eq("wild_type")].copy()


def patient_labels(rows: pd.DataFrame) -> pd.Series:
    """One KRAS status per patient (mutant if any of their slides is mutant)."""
    return rows.groupby("patient_id")["kras"].agg(
        lambda values: "mutant" if (values == "mutant").any() else "wild_type"
    )


def patient_counts(rows: pd.DataFrame) -> dict[str, int]:
    labels = patient_labels(rows)
    return {
        "n": int(labels.size),
        "mutant": int((labels == "mutant").sum()),
        "wild_type": int((labels == "wild_type").sum()),
    }


def check_counts(rows: pd.DataFrame, expected: dict[str, int], what: str) -> dict[str, int]:
    """Hard-check an arm against its pre-registered size before freezing it."""
    observed = patient_counts(rows)
    if observed != expected:
        raise SystemExit(
            f"{what}: population changed — expected {expected}, observed {observed}. "
            "The label source or the feature store moved; resolve before writing "
            "any manifest (the study population is pre-registered)."
        )
    return observed


# ── Derived grouping columns ──────────────────────────────────────────────────


def add_context_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """Coarse clinical/technical groupings used for balancing and subgroups."""
    out = rows.copy()
    out["site_class"] = out["tumor_site_group"]
    stage = out["stage_group_major"].astype("string")
    out["stage_class"] = stage.map(
        lambda value: (
            "I-II" if value in {"I", "II"} else ("III-IV" if value in {"III", "IV"} else "unknown")
        )
    )
    slides_per_patient = out.groupby("patient_id")["slide_id"].transform("size")
    out["n_slides_class"] = slides_per_patient.map(lambda n: "1" if n == 1 else "2+")
    return out


# Sidedness (§5.2) is read from the EXACT anatomic wording only. TCGA codes
# most cases as a bare "Colon", and inferring a side from that would invent
# data, so those patients stay `unknown` and the sidedness analysis runs on
# the exact-site cohorts (SR386, SR1482, RIH, CPTAC).
# Ordered longest-token-first: "rectosigmoid" is a left-colon term and must be
# matched before the "rect-" rectal terms it contains.
SIDEDNESS_RULES: tuple[tuple[str, str], ...] = (
    ("rectosigmoid", "left"),
    ("recto-sigmoid", "left"),
    ("splenic flexure", "left"),
    ("descending", "left"),
    ("decending", "left"),  # recorded misspelling in the SurGen source
    ("sigmoid", "left"),
    ("left colon", "left"),
    ("left (descending)", "left"),
    ("distal large bowel", "left"),
    ("hepatic flexure", "right"),
    ("hepatix flexure", "right"),  # recorded misspelling in the CPTAC source
    ("ascending", "right"),
    ("caecum", "right"),
    ("cecum", "right"),
    ("caecal", "right"),
    ("ileocaecal", "right"),
    ("ileocecal", "right"),
    # Transverse colon is assigned to the right/proximal side, the usual
    # embryologic split at the splenic flexure.
    ("transverse", "right"),
    ("tranverse", "right"),  # recorded misspelling in the CPTAC source
    ("right colon", "right"),
    ("right (ascending)", "right"),
    ("right hemicolectomy", "right"),
    ("proximal large bowel", "right"),
    ("anorectal", "rectum"),
    ("anal-rectal", "rectum"),
    ("rectal", "rectum"),
    ("rectum", "rectum"),
)


def sidedness(value: object) -> str:
    """right | left | rectum | unknown from free-text site wording.

    A record naming sites on both sides (a tumour spanning the flexure, or a
    resection listing two blocks) resolves to ``unknown`` rather than to
    whichever token happened to match first.
    """
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    text = value.lower()
    matched = {side for token, side in SIDEDNESS_RULES if token in text}
    colon_sides = matched & {"left", "right"}
    if len(colon_sides) == 1:
        return colon_sides.pop()
    if colon_sides:
        return "unknown"  # spans both sides
    if "rectum" in matched:
        return "rectum"
    return "unknown"


def side_coprimary(value: object) -> str:
    """Two-way side used by the co-primary test: distal colon + rectum vs right.

    KRAS prevalence runs ~15-25 points higher on the right in every cohort
    whose site coding is specific enough to classify, and that gap survives
    the MSS/pMMR + BRAF-wild-type restriction — so the molecular challenge set
    does not control it, and a model riding anatomy alone would pass every
    other test in §5. Grouping rectum with the distal colon follows the
    embryologic hindgut split and keeps the arm powered; the three-way
    ``sidedness`` split is reported beside it so the rectal contribution stays
    visible rather than assumed.
    """
    side = sidedness(value)
    return "left_incl_rectum" if side in ("left", "rectum") else side


def attach_technical(rows: pd.DataFrame, technical: pd.DataFrame) -> pd.DataFrame:
    """Merge the frozen technical table onto a manifest."""
    columns = [
        "slide_id",
        "native_mpp",
        "mpp_bin",
        "patch_count",
        "tissue_area_mm2",
        "tissue_grid_occupancy",
        "thumbnail_hue_median",
        "section_size_class",
        "color_class",
        "technical_class",
        "umap_island",
    ]
    available = [column for column in columns if column in technical.columns]
    merged = rows.merge(technical[available], on="slide_id", how="left", validate="one_to_one")
    missing = int(merged["technical_class"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} slide(s) have no technical descriptors; rerun aim1_phase1")
    return merged


def finalize(rows: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    """Project to the frozen manifest column set, in a fixed order."""
    columns = [column for column in MANIFEST_COLUMNS if column in rows.columns]
    for column in extra or []:
        if column in rows.columns and column not in columns:
            columns.append(column)
    out = rows[columns].copy()
    if out["slide_id"].duplicated().any():
        raise ValueError("Duplicate slide_id in manifest")
    if out["target_label"].isna().any():
        raise ValueError("Missing target_label in manifest")
    return out.sort_values("slide_id").reset_index(drop=True)


def summarize(rows: pd.DataFrame) -> dict[str, Any]:
    counts = patient_counts(rows)
    return {
        **counts,
        "slides": int(len(rows)),
        "multi_slide_patients": int(
            (rows.groupby("patient_id")["slide_id"].size() > 1).sum()  # type: ignore[operator]
        ),
    }


# ── Metastatic arms (E2a transport, E3 few-shot) ──────────────────────────────


def eligible_metastatic(source: pd.DataFrame | None = None) -> pd.DataFrame:
    """KRAS-labelled METASTATIC specimens, feature-backed.

    Kept strictly separate from the primary population: a patient's primary and
    metastasis are different specimens with different labels-in-principle, and
    patient scores are never pooled across specimen roles.
    """
    frame = load_label_source() if source is None else source
    stems = feature_stems()
    rows = frame[
        frame["specimen_role"].eq("metastatic")
        & frame["kras"].isin(["mutant", "wild_type"])
        & frame["output_id"].isin(stems)
    ].copy()
    rows = rows.rename(columns={"output_id": "slide_id", "patient_uid": "patient_id"})
    rows["target_label"] = rows["kras"].eq("mutant").astype(int)
    rows["slide_weight"] = 1.0 / rows.groupby("patient_id")["slide_id"].transform("size")
    rows["met_site_class"] = (
        rows["metastatic_site_group"].astype("string").fillna("unknown").str.lower()
    )
    rows["liver_class"] = np.where(rows["met_site_class"].eq("liver"), "liver", "non_liver")
    return rows.sort_values("slide_id").reset_index(drop=True)


def dual_specimen_patients(primary: pd.DataFrame, metastatic: pd.DataFrame) -> set[str]:
    """Patients contributing BOTH a primary and a metastasis.

    E3's leakage rule needs these: a patient's primary must leave the
    local-primary support pool whenever their metastasis is in the test fold.
    """
    return set(primary["patient_id"]) & set(metastatic["patient_id"])


# ── Allele tasks (E5 resolution ladder) ───────────────────────────────────────

CODON12_TOKENS = ("G12",)


def _tokens(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [t.strip() for t in value.split(";") if t.strip()]


def allele_population(primary: pd.DataFrame) -> pd.DataFrame:
    """KRAS-mutant primaries with exactly ONE recorded substitution.

    Multi-substitution patients are excluded rather than resolved by string
    order: assigning "G12V;G12C" to whichever token happens to come first
    would invent an allele label.
    """
    mutant = primary[primary["kras"].eq("mutant")].copy()
    per_patient = mutant.groupby("patient_id")["kras_subvariant"].first()
    single = {patient for patient, value in per_patient.items() if len(_tokens(value)) == 1}
    out = mutant[mutant["patient_id"].isin(single)].copy()
    out["allele"] = out["kras_subvariant"].map(lambda v: _tokens(v)[0])
    return out


def allele_task(primary: pd.DataFrame, task: str) -> pd.DataFrame:
    """One rung of the resolution ladder, labelled in ``target_label``."""
    rows = allele_population(primary).copy()
    if task == "codon12":
        rows["target_label"] = rows["allele"].str.startswith("G12").astype(int)
    elif task in {"g12d", "g12v", "g13d", "g12c"}:
        rows["target_label"] = rows["allele"].str.upper().str.startswith(task.upper()).astype(int)
    else:
        raise ValueError(f"unknown allele task {task!r}")
    return rows
