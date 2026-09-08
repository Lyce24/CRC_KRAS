"""Shared machinery for the KRAS allele study (paths, CV protocol, statistics).

Composed by the phase CLIs:

    phase3  builds frozen manifests + the shared 5-fold x 3-seed assignment
    phase4  tunes the 8-configuration grid on P1's inner-validation splits
    phase5  trains each experiment (3 seeds x 5-fold OOF CV) and reports the
            three DEV numbers of §6.4
    phase6  scores the frozen 15-model ensembles on EXT-P (once)
    phase7  scores them on EXT-M (once) and computes the T1 transfer deltas

Protocol constants here mirror Experimental_Setup.md and are part of the
freeze snapshot (§6.5).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from oceanpath.eval.external import logit, sigmoid
from oceanpath.kras.registry import EXPERIMENTS, KrasExperiment

logger = logging.getLogger(__name__)

# ── Frozen study constants ────────────────────────────────────────────────────

# One shared patient-level fold assignment: 5 folds x 3 seeds (§6.4). The
# inner 85/15 early-stopping carve-out is drawn inside each (seed, fold) by
# the oof_kfold scheme (splitting.core) with val_ratio ES_VAL_RATIO.
SEEDS: tuple[int, ...] = (42, 43, 44)
N_FOLDS = 5
ES_VAL_RATIO = 0.15

# Early stopping (§6.4): patience 8, max 40 epochs, monitored on
# inner-validation PATIENT AUROC. If an inner-validation split has fewer than
# MIN_ES_VAL_POSITIVES positive patients, early stopping is disabled for that
# run and training uses the fixed epoch budget (P1's median best-epoch).
MAX_EPOCHS = 40
ES_PATIENCE = 8
MIN_ES_VAL_POSITIVES = 8

# Patient-clustered percentile bootstrap (§6.7).
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260814

# The pinned tile encoder (§6.2): selected once, before any DEV training.
PINNED_ENCODER = "univ1"

MANIFEST_ROOT = Path("/mnt/d/YC.Liu/manifests/colon")
KRAS_FINAL_CSV = MANIFEST_ROOT / "kras_final.csv"

FEATURE_ROOT = Path("/mnt/wsl/oceanpath-hot/features/colon_stream")
PINNED_FEATURE_DIR = FEATURE_ROOT / "20x_256px_0px_overlap_mpp0.5" / "features_uni_v1"

DEV_SUBCOHORTS = ("SR386", "TCGA-COAD", "TCGA-READ")

# group key -> (site label, specimen role). Order fixes report row order.
EXTERNAL_GROUPS: dict[str, tuple[str, str]] = {
    "sr1482_primary": ("SR1482", "primary"),
    "rih_primary": ("RIH", "primary"),
    "sr1482_metastatic": ("SR1482", "metastatic"),
    "rih_metastatic": ("RIH", "metastatic"),
}
ARM_GROUPS: dict[str, tuple[str, ...]] = {
    "ext_p": ("sr1482_primary", "rih_primary"),
    "ext_m": ("sr1482_metastatic", "rih_metastatic"),
}

# Manifest columns shared by every study manifest. Sensitivity passthroughs
# (msi_dmmr, tumor_site_group, metastatic_site_group) let phase 8 re-slice
# saved predictions without re-joining the label source.
MANIFEST_COLUMNS = [
    "slide_id",
    "patient_id",
    "target_label",
    "kras",
    "kras_subvariant",
    "kras_class",
    "cohort_group",
    "subcohort",
    "specimen_role",
    "strat_kras_class_cohort",
    "msi_dmmr",
    "tumor_site_group",
    "metastatic_site_group",
]


def split_name(seed: int) -> str:
    """Directory name of one seed's shared fold assignment."""
    return f"oofk{N_FOLDS}_seed{seed}_es15_strat6c"


def master_manifest_path() -> Path:
    return MANIFEST_ROOT / "crc_kras_master_dev.csv"


def dev_manifest_path(exp_id: str) -> Path:
    return MANIFEST_ROOT / f"crc_kras_dev_{exp_id}.csv"


def external_manifest_path(exp_id: str, group: str) -> Path:
    return MANIFEST_ROOT / f"crc_kras_{exp_id}_{group}.csv"


def folds_csv_path() -> Path:
    """The frozen human-readable fold record of §6.4/§6.5."""
    return MANIFEST_ROOT / "crc_kras_dev_folds.csv"


def data_name(exp_id: str) -> str:
    """data.name used for the split/training path layout of one experiment."""
    return f"colon_kras_{exp_id}"


def run_dir(train_root: Path, exp_id: str, seed: int, encoder: str = PINNED_ENCODER) -> Path:
    """Training run directory of one (experiment, seed) 5-fold CV."""
    return Path(train_root) / "kras" / exp_id / encoder / f"seed{seed}"


# ── Manifest invariants ───────────────────────────────────────────────────────


def assert_manifest_invariants(out: pd.DataFrame, key: str) -> None:
    """Hard guarantees every downstream consumer relies on."""
    assert out["slide_id"].is_unique, f"{key}: duplicate slide_ids"
    required = ["slide_id", "patient_id", "target_label"]
    assert out[required].notna().all().all(), f"{key}: NaNs in {required}"
    labels = sorted(out["target_label"].unique())
    n_classes = len(labels)
    assert labels == list(range(n_classes)) and n_classes >= 2, (
        f"{key}: target_label must be contiguous 0..C-1 with C>=2, got {labels}"
    )
    conflicts = out.groupby("patient_id")["target_label"].nunique()
    bad = conflicts[conflicts > 1]
    assert bad.empty, (
        f"{key}: {len(bad)} patient(s) with conflicting labels (labels are "
        f"per-specimen — split the group or drop the patient): {list(bad.index[:5])}"
    )


# ── Run loading ───────────────────────────────────────────────────────────────


def load_run_config(train_dir: Path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(train_dir) / "config.yaml")
    return cfg


def run_is_complete(train_dir: Path) -> bool:
    completion = Path(train_dir) / "training_completion.json"
    if not completion.is_file():
        return False
    try:
        return json.loads(completion.read_text()).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def seed_run_dirs(train_root: Path, exp_id: str, encoder: str = PINNED_ENCODER) -> dict[int, Path]:
    return {seed: run_dir(train_root, exp_id, seed, encoder) for seed in SEEDS}


def fold_metrics(train_dir: Path, n_folds: int = N_FOLDS) -> list[dict]:
    out = []
    for fold_idx in range(n_folds):
        path = Path(train_dir) / f"fold_{fold_idx}" / "fold_metrics.json"
        if not path.is_file():
            raise SystemExit(f"Missing {path} — run phase 5 training first")
        out.append(json.loads(path.read_text()))
    return out


def fold_checkpoints(train_dirs: dict[int, Path]) -> list[tuple[int, int, Path]]:
    """(seed, fold, best-checkpoint path) for the frozen 15-model ensemble."""
    checkpoints = []
    for seed, train_dir in sorted(train_dirs.items()):
        for fold_idx, metrics in enumerate(fold_metrics(train_dir)):
            ckpt = Path(str(metrics["best_checkpoint"]))
            if not ckpt.is_file():
                raise SystemExit(f"Fold checkpoint missing: {ckpt}")
            checkpoints.append((seed, fold_idx, ckpt))
    expected = len(SEEDS) * N_FOLDS
    if len(checkpoints) != expected:
        raise SystemExit(f"Ensemble needs {expected} checkpoints, found {len(checkpoints)}")
    return checkpoints


def median_best_epoch(train_dirs: dict[int, Path]) -> int:
    """P1's fixed epoch budget: median best-epoch across its 15 fold-runs."""
    epochs = [
        int(metrics["best_epoch"])
        for train_dir in train_dirs.values()
        for metrics in fold_metrics(train_dir)
    ]
    budget = int(round(float(np.median(epochs))))
    return max(1, budget)


# ── Patient-level aggregation (§6.6) ──────────────────────────────────────────


def prob_columns(df: pd.DataFrame) -> list[str]:
    cols = sorted(
        (c for c in df.columns if c.startswith("prob_")),
        key=lambda c: int(c.split("_")[1]),
    )
    if not cols:
        raise ValueError("prediction frame has no prob_* columns")
    return cols


def to_patient_logits(slide_df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Slide → patient: MEAN of the patient's slide logits (§6.6, pre-registered).

    Binary frames (prob_1) yield a ``logit`` column; multiclass frames yield
    ``logit_<c>`` per class. Labels must be patient-consistent (asserted at
    manifest build).
    """
    df = slide_df.merge(manifest[["slide_id", "patient_id"]], on="slide_id", validate="one_to_one")
    cols = prob_columns(df)
    if cols == ["prob_1"]:
        df["logit"] = logit(df["prob_1"].to_numpy())
        value_cols = ["logit"]
    else:
        value_cols = []
        for col in cols:
            df[f"logit_{col.split('_')[1]}"] = logit(df[col].to_numpy())
            value_cols.append(f"logit_{col.split('_')[1]}")
    agg = {"label": ("label", "max"), "n_slides": ("slide_id", "count")}
    agg.update({c: (c, "mean") for c in value_cols})
    return df.groupby("patient_id").agg(**agg).reset_index()


def patient_auroc(patients: pd.DataFrame) -> float:
    """Patient-level AUROC: binary on ``logit``; multiclass macro OvR."""
    from sklearn.metrics import roc_auc_score

    y = patients["label"].to_numpy()
    if "logit" in patients.columns:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, patients["logit"].to_numpy()))
    logit_cols = sorted(
        (c for c in patients.columns if c.startswith("logit_")),
        key=lambda c: int(c.split("_")[1]),
    )
    scores = patients[logit_cols].to_numpy()
    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    if len(np.unique(y)) < scores.shape[1]:
        return float("nan")
    return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))


def seed_averaged_patients(
    per_seed_patients: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Average patient logits over seeds (§6.4: 'scores averaged over the 3
    seeds'). Every seed must score exactly the same patient set."""
    frames = []
    reference: set | None = None
    for seed, patients in sorted(per_seed_patients.items()):
        ids = set(patients["patient_id"])
        if reference is None:
            reference = ids
        elif ids != reference:
            raise SystemExit(
                f"Seed {seed} scored a different patient set "
                f"({len(ids ^ reference)} mismatches) — OOF is not a partition"
            )
        frames.append(patients.assign(seed=seed))
    stacked = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in stacked.columns if c == "logit" or c.startswith("logit_")]
    agg = {"label": ("label", "first"), "n_slides": ("n_slides", "first")}
    agg.update({c: (c, "mean") for c in value_cols})
    labels_per_patient = stacked.groupby("patient_id")["label"].nunique()
    assert (labels_per_patient == 1).all(), "label disagreement across seeds"
    return stacked.groupby("patient_id").agg(**agg).reset_index()


# ── Patient-clustered bootstrap (§6.7) ────────────────────────────────────────


def bootstrap_patient_auroc(
    patients: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Percentile CI for patient AUROC, resampling patients with replacement."""
    rng = np.random.default_rng(seed)
    n = len(patients)
    point = patient_auroc(patients)
    samples = []
    for _ in range(n_bootstrap):
        resampled = patients.iloc[rng.integers(0, n, n)]
        value = patient_auroc(resampled)
        if np.isfinite(value):
            samples.append(value)
    lo, hi = (
        (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
        if samples
        else (float("nan"), float("nan"))
    )
    return {
        "auroc": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_patients": int(n),
        "n_positive": int((patients["label"] > 0).sum()),
        "n_resamples_valid": len(samples),
    }


def delta_auroc_bootstrap(
    patients_a: pd.DataFrame,
    patients_b: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Independent two-sample bootstrap for AUROC(b) − AUROC(a) (§6.7).

    a = EXT-P, b = EXT-M: negative delta means metastatic degradation.
    """
    rng = np.random.default_rng(seed)
    point = patient_auroc(patients_b) - patient_auroc(patients_a)
    n_a, n_b = len(patients_a), len(patients_b)
    samples = []
    for _ in range(n_bootstrap):
        auroc_a = patient_auroc(patients_a.iloc[rng.integers(0, n_a, n_a)])
        auroc_b = patient_auroc(patients_b.iloc[rng.integers(0, n_b, n_b)])
        if np.isfinite(auroc_a) and np.isfinite(auroc_b):
            samples.append(auroc_b - auroc_a)
    lo, hi = (
        (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
        if samples
        else (float("nan"), float("nan"))
    )
    return {"delta": float(point), "ci_low": lo, "ci_high": hi, "n_resamples_valid": len(samples)}


# ── DEV reporting: the three numbers of §6.4 ──────────────────────────────────


def dev_report(
    experiment: KrasExperiment,
    train_dirs: dict[int, Path],
    manifest: pd.DataFrame,
) -> dict:
    """Per-fold mean±SD (diagnostic), seed-averaged pooled OOF with CI
    (headline), and per-cohort OOF (shortcut check)."""
    per_fold_aurocs: list[float] = []
    per_seed_patients: dict[int, pd.DataFrame] = {}
    substitutions = []
    for seed, train_dir in sorted(train_dirs.items()):
        oof_path = Path(train_dir) / "oof_predictions.parquet"
        if not oof_path.is_file():
            raise SystemExit(f"Missing {oof_path} — training incomplete")
        oof = pd.read_parquet(oof_path)
        for fold_idx in sorted(oof["fold"].unique()):
            fold_patients = to_patient_logits(oof[oof["fold"] == fold_idx], manifest)
            per_fold_aurocs.append(patient_auroc(fold_patients))
        per_seed_patients[seed] = to_patient_logits(oof.drop(columns=["fold"]), manifest)
        for metrics in fold_metrics(train_dir):
            if metrics.get("early_stopping_disabled"):
                substitutions.append(
                    {
                        "seed": seed,
                        "fold": metrics.get("fold"),
                        "reason": metrics.get("early_stopping_disabled"),
                        "epoch_budget": metrics.get("fixed_epoch_budget"),
                    }
                )

    pooled = seed_averaged_patients(per_seed_patients)
    n_expected = manifest["patient_id"].nunique()
    if len(pooled) != n_expected:
        raise SystemExit(
            f"Pooled OOF covers {len(pooled)} patients, manifest has {n_expected} — "
            "the outer folds do not partition the cohort"
        )
    per_cohort = {}
    cohort_of_patient = manifest.groupby("patient_id")["cohort_group"].first()
    for cohort, ids in cohort_of_patient.groupby(cohort_of_patient).groups.items():
        subset = pooled[pooled["patient_id"].isin(set(ids))]
        per_cohort[str(cohort)] = {
            "auroc": patient_auroc(subset),
            "n_patients": int(len(subset)),
            "n_positive": int((subset["label"] > 0).sum()),
        }

    finite = [a for a in per_fold_aurocs if np.isfinite(a)]
    best_epochs = [
        int(metrics["best_epoch"])
        for train_dir in train_dirs.values()
        for metrics in fold_metrics(train_dir)
    ]
    return {
        "experiment": experiment.exp_id,
        "title": experiment.title,
        "n_fold_runs": len(per_fold_aurocs),
        # Stability diagnostic ONLY — never a confidence interval (§6.4).
        "per_fold_auroc_mean": float(np.mean(finite)) if finite else float("nan"),
        "per_fold_auroc_sd": float(np.std(finite)) if finite else float("nan"),
        "per_fold_aurocs": per_fold_aurocs,
        "pooled_oof": bootstrap_patient_auroc(pooled),
        "per_cohort_oof": per_cohort,
        "median_best_epoch": int(np.median(best_epochs)),
        "early_stopping_substitutions": substitutions,
    }


# ── Frozen-ensemble inference (§6.4: deployed score = mean logit) ─────────────


def score_manifest_with_checkpoints(
    checkpoints: list[tuple[int, int, Path]],
    manifest_csv: Path,
    feature_dir: Path,
    num_classes: int,
    batch_size: int = 1,
    num_workers: int = 4,
    device: str | None = None,
) -> pd.DataFrame:
    """Score every manifest slide with every fold model. FULL bags, fp32/bf16.

    Returns one row per (slide, seed, fold) with logit columns — the caller
    averages logits into the deployed ensemble score, and keeping the
    per-model scores makes every sensitivity re-slice (phase 8) free.
    """
    import torch
    from torch.utils.data import DataLoader

    from oceanpath.datasets.datamodule import SimpleMILCollator, SlideDataset
    from oceanpath.training.lightning import MILTrainModule

    manifest = pd.read_csv(manifest_csv)
    dataset = SlideDataset(
        feature_dir=str(feature_dir),
        slide_ids=manifest["slide_id"].tolist(),
        labels=dict(zip(manifest["slide_id"], manifest["target_label"], strict=True)),
        max_instances=None,  # full bags for the scientific result (§6.3)
        is_train=False,
        force_float32=True,
    )
    missing = set(manifest["slide_id"]) - set(dataset.slide_ids)
    if missing:
        raise SystemExit(f"{len(missing)} manifest slides lack features: {sorted(missing)[:5]}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SimpleMILCollator(max_instances=None),
    )

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = resolved_device == "cuda" and torch.cuda.is_bf16_supported()
    rows: list[dict] = []
    for seed, fold_idx, ckpt in checkpoints:
        try:
            module = MILTrainModule.load_from_checkpoint(
                str(ckpt), map_location=resolved_device, weights_only=False
            )
        except TypeError:
            module = MILTrainModule.load_from_checkpoint(str(ckpt), map_location=resolved_device)
        module.eval().to(resolved_device)
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(resolved_device, non_blocking=True)
                mask = batch["mask"].to(resolved_device) if batch.get("mask") is not None else None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    output = module.model(features, mask=mask)
                logits = output.logits.detach().float().cpu().numpy()
                for i, slide_id in enumerate(batch["slide_ids"]):
                    row = {"slide_id": slide_id, "seed": seed, "fold": fold_idx}
                    values = np.atleast_1d(logits[i]).ravel()
                    if num_classes == 2 and values.size == 1:
                        row["logit"] = float(values[0])
                    else:
                        for c in range(values.size):
                            row[f"logit_{c}"] = float(values[c])
                    rows.append(row)
        del module
        if resolved_device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Scored %s with seed %d fold %d", manifest_csv.name, seed, fold_idx)

    scores = pd.DataFrame(rows)
    n_models = len(checkpoints)
    if len(scores) != n_models * len(dataset):
        raise SystemExit("Scoring produced an unexpected number of rows")
    return scores


def ensemble_slide_predictions(scores: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-model logits → deployed slide score (mean logit over the 15 models),
    emitted as prob_* columns so patient aggregation is shared with OOF."""
    logit_cols = [c for c in scores.columns if c == "logit" or c.startswith("logit_")]
    mean_logits = scores.groupby("slide_id")[logit_cols].mean().reset_index()
    out = mean_logits.merge(
        manifest[["slide_id", "target_label"]], on="slide_id", validate="one_to_one"
    ).rename(columns={"target_label": "label"})
    if logit_cols == ["logit"]:
        out["prob_1"] = sigmoid(out["logit"].to_numpy())
    else:
        raw = out[logit_cols].to_numpy()
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        for index, col in enumerate(logit_cols):
            out[f"prob_{col.split('_')[1]}"] = probs[:, index]
    return out


def experiment_from_id(exp_id: str) -> KrasExperiment:
    if exp_id not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment '{exp_id}'. Known: {', '.join(EXPERIMENTS)}")
    return EXPERIMENTS[exp_id]


# ── One-shot external scoring (freeze protocol, §6.5) ─────────────────────────


def score_group_once(
    exp_id: str,
    group: str,
    eval_root: Path,
    train_root: Path,
    encoder: str = PINNED_ENCODER,
    force: bool = False,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Score one experiment's frozen 15-model ensemble on one external group.

    §6.5: after the freeze tag, EXT-P is scored once, then EXT-M once. Saved
    predictions are therefore authoritative — a completed group is loaded
    from disk, never re-scored (``force`` exists for logged amendments only).
    Persists per-model logits (model_scores.parquet) so every phase-8
    sensitivity re-slice is a free re-read, plus the deployed mean-logit
    slide predictions (slide_predictions.parquet).
    """
    experiment = experiment_from_id(exp_id)
    manifest_csv = external_manifest_path(exp_id, group)
    if not manifest_csv.is_file():
        raise SystemExit(f"Missing {manifest_csv} — run phase3 external-manifests --apply")
    out_dir = Path(eval_root) / exp_id / group
    slide_path = out_dir / "slide_predictions.parquet"
    if slide_path.is_file() and not force:
        logger.info("%s/%s already scored — loading %s", exp_id, group, slide_path)
        return pd.read_parquet(slide_path)

    if force and slide_path.is_file():
        logger.warning(
            "FORCE RE-SCORE of %s/%s — this violates the score-once freeze "
            "protocol unless recorded as an amendment (§6.5)",
            exp_id,
            group,
        )

    train_dirs = seed_run_dirs(train_root, exp_id, encoder)
    incomplete = [seed for seed, d in train_dirs.items() if not run_is_complete(d)]
    if incomplete:
        raise SystemExit(f"{exp_id}: seed-runs {incomplete} incomplete — finish phase 5 first")
    checkpoints = fold_checkpoints(train_dirs)

    manifest = pd.read_csv(manifest_csv)
    scores = score_manifest_with_checkpoints(
        checkpoints,
        manifest_csv,
        PINNED_FEATURE_DIR,
        num_classes=experiment.num_classes,
        num_workers=num_workers,
    )
    slide_preds = ensemble_slide_predictions(scores, manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out_dir / "model_scores.parquet", index=False)
    slide_preds.to_parquet(slide_path, index=False)
    logger.info(
        "%s/%s: scored %d slides x %d models", exp_id, group, len(manifest), len(checkpoints)
    )
    return slide_preds
