"""Frozen source paths, protocol constants, and the Aim-1 output layout.

Every path here is an input the study does not own (slides, labels, features)
or a location the study writes exactly once. Keeping them in one module makes
the freeze snapshot (Aim1_Setup.md §8) a single file to read.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Source data (owned upstream; read-only) ───────────────────────────────────

MANIFEST_ROOT = Path("/mnt/d/YC.Liu/manifests/colon")
LABEL_SOURCE = MANIFEST_ROOT / "crc_final_v4.csv"

FEATURE_ROOT = Path("/mnt/wsl/oceanpath-hot/features")
PATCH_PROFILE = "20x_256px_0px_overlap_mpp0.5"
PINNED_ENCODER = "univ1"  # encoder profile name (configs/encoder/univ1.yaml)
FEATURES_SUBDIR = "features_uni_v1"  # TRIDENT's on-disk directory for that encoder
PINNED_FEATURE_DIR = FEATURE_ROOT / "colon_stream" / PATCH_PROFILE / FEATURES_SUBDIR
PACKED_FEATURE_DIR = FEATURE_ROOT / "colon_stream" / PATCH_PROFILE / "packed_uni_v1"

# HEST thumbnails live next to the features; the CPTAC priority run wrote its
# own tree before the cohort was merged into the main store.
THUMBNAIL_DIRS: tuple[Path, ...] = (
    FEATURE_ROOT / "colon_stream" / "thumbnails",
    FEATURE_ROOT / "cptac_priority" / "thumbnails",
)

# The frozen SurGen/RIH island audit — reused verbatim, never recomputed.
UMAP_AUDIT_ASSIGNMENTS = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/univ1_umap_separation_audit/slide_cluster_assignments.parquet"
)

# ── Study outputs ─────────────────────────────────────────────────────────────

OUTPUT_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1")
TECHNICAL_DIR = OUTPUT_ROOT / "technical"
TECHNICAL_TABLE = TECHNICAL_DIR / "slide_technical_variables.parquet"
TECHNICAL_SUMMARY = TECHNICAL_DIR / "summary.json"

TRAIN_ROOT = OUTPUT_ROOT / "train"
EVAL_ROOT = OUTPUT_ROOT / "eval"

# Splits live where Hydra's FoundationPaths looks for them
# (platform.splits_root / data.name / splits.name), so the phase scripts and a
# plain `scripts/train.py` invocation resolve the same directory. The default
# splits_root is worktree-relative, which is what keeps this development
# worktree isolated from the production encoder worktree.
SPLIT_ROOT = Path(os.environ.get("OCEANPATH_STATE_ROOT", "outputs")) / "splits"
SPLIT_NAME = "aim1_balanced5"

# Frozen manifests (written once by aim1_phase2, consumed by everything else).
MANIFEST_DIR = MANIFEST_ROOT
DEV_MANIFEST = MANIFEST_DIR / "aim1_dev.csv"
FOLDS_CSV = MANIFEST_DIR / "aim1_dev_folds.csv"


def external_manifest(group: str) -> Path:
    """One frozen external manifest per evaluation cohort."""
    return MANIFEST_DIR / f"aim1_ext_{group}.csv"


def restricted_manifest(group: str) -> Path:
    """MSS/pMMR + BRAF-wild-type restricted manifest (1C-R training/eval)."""
    return MANIFEST_DIR / f"aim1_restricted_{group}.csv"


# ── Protocol constants (Aim1_Setup.md §4) ─────────────────────────────────────

N_FOLDS = 5
PRIMARY_SEED = 42
STABILITY_SEEDS: tuple[int, ...] = (43, 44)
ES_VAL_RATIO = 0.15

# Locked recipe: user decision 2026-08-17 (phase-4 grid axes, dropout 0.25).
LR = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.25
MAX_EPOCHS = 20
ES_PATIENCE = 5
MIN_EPOCH_BEFORE_STOP = 10
MIN_ES_VAL_POSITIVES = 8

# Standard protocol (2026-08-18): stopping and checkpoint selection both read
# PATIENT-LEVEL validation AUROC. The unweighted patient-level validation loss
# is logged alongside as a calibration diagnostic, not used for selection —
# it diverges while discrimination improves on this cohort.
MONITOR_METRIC = "val/patient_auroc"
MONITOR_MODE = "max"
OVERSIZED_BAG_FALLBACK = 15000

# Patient-clustered percentile bootstrap.
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260817

# Locked triage operating point (§4.6): sensitivity target on development OOF.
TRIAGE_SENSITIVITY = 0.90

# Aim 1 trains on ALL KRAS-labelled primary patients from ALL FOUR COHORTS with
# no tumour-site restriction, and reads out pooled out-of-fold predictions.
# There is no held-out external cohort here by design: Aim 1 asks whether the
# signal is KRAS-SPECIFIC (§5 challenge sets, which gain power from the pooled
# population), and Aim 2 owns transportability through its own
# leave-one-institution-out design.
COHORTS: tuple[str, ...] = ("TCGA", "SurGen", "RIH", "CPTAC")
DEV_SUBCOHORTS: tuple[str, ...] = (
    "TCGA-COAD",
    "TCGA-READ",
    "SR386",
    "SR1482",
    "RIH-Colon",
    "CPTAC-COAD",
)

# Kept for Aim 2, which does hold cohorts out; unused by the Aim-1 pooled design.
EXTERNAL_GROUPS: tuple[str, ...] = ()
EXTERNAL_SUBCOHORT: dict[str, str] = {}

# Eligible-population counts, hard-checked by the phase-2 builder before any
# manifest is written.
EXPECTED_PATIENTS = {"dev": {"n": 1486, "mutant": 604, "wild_type": 882}}
EXPECTED_RESTRICTED_PATIENTS: dict[str, dict[str, int]] = {}

# Per-cohort composition, verified and reported (never a training boundary).
EXPECTED_BY_SUBCOHORT = {
    "CPTAC-COAD": {"patients": 94, "mutant": 33},
    "RIH-Colon": {"patients": 153, "mutant": 70},
    "SR1482": {"patients": 324, "mutant": 147},
    "SR386": {"patients": 413, "mutant": 147},
    "TCGA-COAD": {"patients": 374, "mutant": 160},
    "TCGA-READ": {"patients": 128, "mutant": 47},
}
