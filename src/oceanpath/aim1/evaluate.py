"""Patient-level evaluation, calibration, and triage metrics (§4.6-4.8, §5.4).

Everything here operates on *patients*, never slides: a patient's score is the
mean of their slide logits, and the sigmoid is applied afterwards (§4.3).
Slide-level AUROC is never the reported result.

Calibration follows §4.6 exactly and keeps the two uses separate:

    internal   cross-fitted Platt — the calibrator applied to fold *i* is
               fitted on the OOF predictions of the other four folds, so no
               patient is ever calibrated by a model that saw them
    external   ONE final calibrator fitted on all development OOF, applied
               unchanged to SR1482, RIH, and CPTAC. No external cohort is
               recalibrated in Aim 1.

Two thresholds are locked on development OOF and never re-optimized
externally: a clinical triage point at ~90% sensitivity, and Youden as a
supplementary comparison.

The §5.4 reporting rule is enforced in code rather than left to the writer:
a subgroup with fewer than 10 patients in either class gets no AUROC at all,
10-19 is marked exploratory, and 20+ is a formal estimate.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from oceanpath.aim1 import paths, registry
from oceanpath.eval.external import (
    apply_calibrator,
    calibration_block,
    fit_platt_on_oof,
    operating_point,
    sigmoid,
)

# §5.4 subgroup reporting rule, in patients of the smaller class.
FORMAL_MIN = 20
EXPLORATORY_MIN = 10

TRIAGE_FRACTIONS = (0.10, 0.20, 0.30)


# ── Slide -> patient ──────────────────────────────────────────────────────────


def to_patient_level(slide_predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Mean slide logit per patient, then the sigmoid (§4.3).

    Carries through the manifest's context columns (patient-constant by
    construction) so every challenge set of §5 can slice the same frame.

    ``logit`` is the model's native linear predictor.  Reconstructing it from
    ``prob_1`` is not equivalent for saturated predictions: the probability
    path clips/rounds extreme values and can create artificial ties.  Aim 1/2
    prediction writers persist the native value, so fail closed if it is absent.
    """
    if "logit" not in slide_predictions.columns:
        raise ValueError(
            "slide predictions lack native 'logit'; refusing probability-to-logit "
            "reconstruction"
        )
    native_logits = pd.to_numeric(slide_predictions["logit"], errors="coerce")
    if not np.isfinite(native_logits.to_numpy()).all():
        raise ValueError("slide predictions contain non-finite native logits")
    context = [
        column
        for column in (
            "subcohort",
            "cohort",
            "msi_dmmr",
            "braf",
            "nras",
            "ras",
            "tumor_site_group",
            "tumor_site_raw",
            "stage_class",
            "site_class",
            "age_at_diagnosis",
            "sex",
            "mpp_bin",
            "technical_class",
            "section_size_class",
            "color_class",
            "patch_count",
            "tissue_area_mm2",
            "tissue_grid_occupancy",
            "thumbnail_hue_median",
            "k_fold",
        )
        if column in manifest.columns
    ]
    merged = slide_predictions.merge(
        manifest[["slide_id", "patient_id", *context]], on="slide_id", validate="one_to_one"
    )
    merged["slide_logit"] = pd.to_numeric(merged["logit"], errors="raise")
    aggregation: dict[str, Any] = {
        "label": ("label", "max"),
        "mean_logit": ("slide_logit", "mean"),
        "n_slides": ("slide_id", "count"),
    }
    # Context is patient-constant; "first" after a stable sort is deterministic.
    aggregation.update({column: (column, "first") for column in context})
    patients = merged.sort_values("slide_id").groupby("patient_id").agg(**aggregation).reset_index()
    patients["prob_raw"] = sigmoid(patients["mean_logit"].to_numpy())
    return patients


# ── Calibration (§4.6) ────────────────────────────────────────────────────────


def cross_fitted_platt(oof_patients: pd.DataFrame, fold_column: str = "k_fold") -> pd.DataFrame:
    """Calibrate each fold with a calibrator fitted on the other four.

    Fitting one calibrator on all of the OOF and applying it back to the same
    patients would let each patient's own outcome shape the mapping applied to
    them, which flatters internal calibration metrics. Cross-fitting removes
    that circularity while keeping the estimator identical.
    """
    out = oof_patients.copy()
    out["prob_cal"] = np.nan
    fits: dict[int, dict] = {}
    for fold in sorted(out[fold_column].unique()):
        training = out[out[fold_column] != fold]
        calibrator = fit_platt_on_oof(training)
        fits[int(fold)] = calibrator
        held_out = out[fold_column] == fold
        out.loc[held_out, "prob_cal"] = apply_calibrator(
            out.loc[held_out, "prob_raw"].to_numpy(), calibrator
        )
    if out["prob_cal"].isna().any():
        raise ValueError("cross-fitted calibration left patients uncalibrated")
    out.attrs["calibrators"] = fits
    return out


def lock_thresholds(oof_calibrated: pd.DataFrame) -> dict[str, Any]:
    """The two source-derived operating points, frozen on development OOF."""
    y = oof_calibrated["label"].to_numpy()
    p = oof_calibrated["prob_cal"].to_numpy()
    fpr, tpr, thresholds = roc_curve(y, p)
    specificity = 1.0 - fpr

    reachable = tpr >= paths.TRIAGE_SENSITIVITY
    if not reachable.any():
        raise ValueError(f"development OOF cannot reach sensitivity {paths.TRIAGE_SENSITIVITY}")
    triage_index = int(np.where(reachable)[0][int(np.argmax(specificity[reachable]))])

    youden_index = int(np.argmax(tpr - fpr))
    return {
        "triage": {
            "threshold": float(thresholds[triage_index]),
            "sensitivity_target": paths.TRIAGE_SENSITIVITY,
            "oof_sensitivity": float(tpr[triage_index]),
            "oof_specificity": float(specificity[triage_index]),
        },
        "youden": {
            "threshold": float(thresholds[youden_index]),
            "oof_sensitivity": float(tpr[youden_index]),
            "oof_specificity": float(specificity[youden_index]),
        },
    }


# ── Metrics ───────────────────────────────────────────────────────────────────


def triage_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    """Prioritization view (§4.7): what a ranked worklist would capture.

    Framed as testing *order*, not as replacing molecular testing.
    """
    order = np.argsort(-score, kind="stable")
    ranked = y[order]
    n = len(y)
    total_positive = int(ranked.sum())
    out: dict[str, Any] = {"n": n, "n_positive": total_positive}
    if total_positive == 0:
        return out
    cumulative = np.cumsum(ranked)
    for fraction in TRIAGE_FRACTIONS:
        take = max(1, int(round(fraction * n)))
        out[f"capture_top_{int(fraction * 100)}pct"] = float(cumulative[take - 1] / total_positive)
    # Smallest ranked prefix containing 90% of the mutants.
    needed = int(np.ceil(paths.TRIAGE_SENSITIVITY * total_positive))
    reached = int(np.searchsorted(cumulative, needed) + 1)
    out["fraction_tested_for_90pct_capture"] = float(reached / n)
    return out


def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    two_class = len(np.unique(y)) == 2
    out = {
        "auroc": float(roc_auc_score(y, p)) if two_class else float("nan"),
        "auprc": float(average_precision_score(y, p)) if two_class else float("nan"),
        "auprc_baseline": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
    }
    point = operating_point(y, p, threshold)
    point["balanced_accuracy"] = 0.5 * (point["sensitivity"] + point["specificity"])
    out.update(point)
    return out


def bootstrap_metrics(
    patients: pd.DataFrame,
    threshold: float,
    n_bootstrap: int = paths.N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
    score_column: str = "prob_cal",
) -> dict[str, list[float]]:
    """Patient-resampled percentile CIs. Patients are the resampling unit, so
    a multi-slide patient is drawn or dropped as a whole."""
    rng = np.random.default_rng(seed)
    y_all = patients["label"].to_numpy()
    p_all = patients[score_column].to_numpy()
    n = len(y_all)
    samples: dict[str, list[float]] = {}
    for _ in range(n_bootstrap):
        index = rng.integers(0, n, n)
        y, p = y_all[index], p_all[index]
        if len(np.unique(y)) < 2:
            continue
        for key, value in point_metrics(y, p, threshold).items():
            if np.isfinite(value):
                samples.setdefault(key, []).append(value)
    return {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in samples.items()
    }


def evaluate_patients(
    patients: pd.DataFrame,
    thresholds: dict[str, Any],
    with_ci: bool = True,
    score_column: str = "prob_cal",
) -> dict[str, Any]:
    """The full §4.7 block for one patient population."""
    y = patients["label"].to_numpy()
    p = patients[score_column].to_numpy()
    block: dict[str, Any] = {
        "n": int(len(patients)),
        "n_mutant": int(y.sum()),
        "n_wild_type": int(len(y) - y.sum()),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
    }
    if len(np.unique(y)) < 2:
        block["note"] = "single-class population — discrimination undefined"
        return block
    block["discrimination"] = point_metrics(y, p, thresholds["triage"]["threshold"])
    block["calibration"] = calibration_block(y, p)
    block["operating_points"] = {
        name: {
            **operating_point(y, p, spec["threshold"]),
            "threshold": spec["threshold"],
        }
        for name, spec in thresholds.items()
    }
    for name in block["operating_points"]:
        point = block["operating_points"][name]
        point["balanced_accuracy"] = 0.5 * (point["sensitivity"] + point["specificity"])
    block["triage"] = triage_metrics(y, p)
    if with_ci:
        block["ci95"] = bootstrap_metrics(
            patients, thresholds["triage"]["threshold"], score_column=score_column
        )
    return block


# ── Subgroups (§5.2-5.4) ──────────────────────────────────────────────────────


def support_tier(y: np.ndarray) -> str:
    """§5.4: how much may be claimed from a subgroup of this size."""
    smaller = min(int(y.sum()), int(len(y) - y.sum()))
    if smaller >= FORMAL_MIN:
        return "formal"
    if smaller >= EXPLORATORY_MIN:
        return "exploratory"
    return "insufficient"


def subgroup_report(
    patients: pd.DataFrame,
    column: str,
    thresholds: dict[str, Any],
    score_column: str = "prob_cal",
) -> dict[str, Any]:
    """Evaluate within every level of ``column``, honouring the §5.4 rule.

    Levels below the claim threshold report score distributions instead of an
    unstable AUROC — the number is not computed and then caveated, it is not
    reported at all.
    """
    out: dict[str, Any] = {}
    for level, block in patients.groupby(patients[column].fillna("unknown").astype(str)):
        y = block["label"].to_numpy()
        tier = support_tier(y)
        entry: dict[str, Any] = {
            "n": int(len(block)),
            "n_mutant": int(y.sum()),
            "n_wild_type": int(len(y) - y.sum()),
            "support": tier,
        }
        if tier == "insufficient":
            scores = block[score_column].to_numpy()
            entry["score_distribution"] = {
                "median": float(np.median(scores)) if len(scores) else float("nan"),
                "q1": float(np.percentile(scores, 25)) if len(scores) else float("nan"),
                "q3": float(np.percentile(scores, 75)) if len(scores) else float("nan"),
                "median_mutant": float(np.median(scores[y == 1])) if (y == 1).any() else None,
                "median_wild_type": float(np.median(scores[y == 0])) if (y == 0).any() else None,
            }
        else:
            entry.update(
                evaluate_patients(
                    block, thresholds, with_ci=(tier == "formal"), score_column=score_column
                )
            )
        out[str(level)] = entry
    return out


# ── Development report (§4.7) ─────────────────────────────────────────────────


def load_oof(model_id: str, seed: int = paths.PRIMARY_SEED) -> pd.DataFrame:
    """Patient-level out-of-fold predictions of one trained model."""
    model = registry.MODELS[model_id]
    run_directory = model.run_dir(seed)
    path = run_directory / "oof_predictions.parquet"
    if not path.is_file():
        raise SystemExit(f"{model_id}: {path} not found — train the model first")
    manifest = pd.read_csv(model.manifest_path)
    return to_patient_level(pd.read_parquet(path), manifest)


def fold_best_epochs(model_id: str, seed: int = paths.PRIMARY_SEED) -> list[int]:
    run_directory = registry.MODELS[model_id].run_dir(seed)
    epochs = []
    for fold in range(paths.N_FOLDS):
        metrics_path = run_directory / f"fold_{fold}" / "fold_metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            epochs.append(int(metrics.get("best_epoch", -1)))
    return epochs


def patient_auroc(patients: pd.DataFrame, score_column: str = "prob_raw") -> float:
    y = patients["label"].to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, patients[score_column].to_numpy()))


def bootstrap_auroc(
    patients: pd.DataFrame,
    score_column: str = "prob_raw",
    n_bootstrap: int = paths.N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(patients)
    values = []
    for _ in range(n_bootstrap):
        resampled = patients.iloc[rng.integers(0, n, n)]
        value = patient_auroc(resampled, score_column)
        if np.isfinite(value):
            values.append(value)
    return {
        "auroc": patient_auroc(patients, score_column),
        "ci_low": float(np.percentile(values, 2.5)) if values else float("nan"),
        "ci_high": float(np.percentile(values, 97.5)) if values else float("nan"),
        "n": int(n),
        "n_positive": int(patients["label"].sum()),
    }


def dev_report(model_id: str, seed: int = paths.PRIMARY_SEED) -> dict[str, Any]:
    """The three development numbers of §4.7 for one model."""
    patients = load_oof(model_id, seed)
    per_fold = [
        patient_auroc(patients[patients["k_fold"] == fold]) for fold in range(paths.N_FOLDS)
    ]
    report: dict[str, Any] = {
        "model": model_id,
        "seed": seed,
        "per_fold": per_fold,
        "per_fold_mean": float(np.nanmean(per_fold)),
        "per_fold_sd": float(np.nanstd(per_fold, ddof=1)),
        "pooled_oof": bootstrap_auroc(patients),
        "per_subcohort": {
            str(subcohort): bootstrap_auroc(block)
            for subcohort, block in patients.groupby("subcohort")
        },
        "best_epochs": fold_best_epochs(model_id, seed),
    }
    return report


def compare_auroc_paired(
    left: pd.DataFrame,
    right: pd.DataFrame,
    score_column: str = "prob_cal",
    n_bootstrap: int = paths.N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired patient bootstrap for AUROC(right) - AUROC(left).

    Paired: both models are resampled on the SAME patients in each replicate,
    which is the comparison 1A vs 1C-R/1C-B needs — the two models are scored
    on identical external populations, so an unpaired interval would discard
    the correlation and overstate the uncertainty.
    """
    merged = left[["patient_id", "label", score_column]].merge(
        right[["patient_id", score_column]], on="patient_id", suffixes=("_left", "_right")
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            f"paired comparison needs identical patient sets: left={len(left)}, "
            f"right={len(right)}, shared={len(merged)}"
        )
    y = merged["label"].to_numpy()
    a = merged[f"{score_column}_left"].to_numpy()
    b = merged[f"{score_column}_right"].to_numpy()
    point = float(roc_auc_score(y, b) - roc_auc_score(y, a))

    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, n, n)
        if len(np.unique(y[index])) < 2:
            continue
        deltas.append(roc_auc_score(y[index], b[index]) - roc_auc_score(y[index], a[index]))
    return {
        "delta_auroc": point,
        "ci_low": float(np.percentile(deltas, 2.5)) if deltas else float("nan"),
        "ci_high": float(np.percentile(deltas, 97.5)) if deltas else float("nan"),
        "n_patients": int(n),
        "auroc_left": float(roc_auc_score(y, a)),
        "auroc_right": float(roc_auc_score(y, b)),
    }


def macro_average(blocks: Sequence[dict[str, Any]], key: str = "auroc") -> float:
    """Unweighted mean across cohorts (§4.8 secondary summary).

    Secondary on purpose: one excellent cohort must not be able to carry a
    failed one, so the per-cohort numbers stay the headline.
    """
    values = [
        block["discrimination"][key]
        for block in blocks
        if "discrimination" in block and np.isfinite(block["discrimination"][key])
    ]
    return float(np.mean(values)) if values else float("nan")


# ── E1a: challenge-set contrasts ──────────────────────────────────────────────


def shared_resample_delta(
    full: pd.DataFrame,
    subset_mask: pd.Series,
    score_column: str = "prob_raw",
    n_bootstrap: int = paths.N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """CI for AUROC(A) - AUROC(subset of A), with the right estimator.

    Every challenge set is a RESTRICTION of the full evaluation population and
    the model is unchanged, so "AUROC of A restricted to D's patients" is
    identically AUROC(D): a naive paired bootstrap that resamples the two sets
    independently, or that pairs patient-to-patient, returns exactly zero
    difference and a meaningless interval.

    The valid estimator resamples patients ONCE from the full set A, then
    recomputes both AUROC(A_boot) and AUROC(D and A_boot) on that same
    resample. The two statistics then move together exactly as much as the
    nesting implies, and their difference has an appropriately correlated
    interval.
    """
    rng = np.random.default_rng(seed)
    y_all = full["label"].to_numpy()
    p_all = full[score_column].to_numpy()
    in_subset = subset_mask.to_numpy(dtype=bool)
    n = len(y_all)

    point_full = float(roc_auc_score(y_all, p_all)) if len(np.unique(y_all)) > 1 else float("nan")
    y_sub, p_sub = y_all[in_subset], p_all[in_subset]
    point_subset = float(roc_auc_score(y_sub, p_sub)) if len(np.unique(y_sub)) > 1 else float("nan")

    deltas: list[float] = []
    subset_values: list[float] = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, n, n)
        y_boot, p_boot = y_all[index], p_all[index]
        keep = in_subset[index]
        if len(np.unique(y_boot)) < 2 or keep.sum() < 10:
            continue
        if len(np.unique(y_boot[keep])) < 2:
            continue
        auroc_full = roc_auc_score(y_boot, p_boot)
        auroc_subset = roc_auc_score(y_boot[keep], p_boot[keep])
        deltas.append(auroc_full - auroc_subset)
        subset_values.append(auroc_subset)

    return {
        "auroc_full": point_full,
        "auroc_subset": point_subset,
        "n_full": int(n),
        "n_subset": int(in_subset.sum()),
        "n_subset_mutant": int(y_sub.sum()),
        "delta": point_full - point_subset,
        "delta_ci_low": float(np.percentile(deltas, 2.5)) if deltas else float("nan"),
        "delta_ci_high": float(np.percentile(deltas, 97.5)) if deltas else float("nan"),
        "subset_ci_low": float(np.percentile(subset_values, 2.5))
        if subset_values
        else float("nan"),
        "subset_ci_high": float(np.percentile(subset_values, 97.5))
        if subset_values
        else float("nan"),
        "n_resamples_valid": len(deltas),
        "estimator": "shared-resample bootstrap over the full population",
    }


def auprc_with_ci(
    patients: pd.DataFrame,
    score_column: str = "prob_raw",
    n_bootstrap: int = paths.N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict[str, float]:
    """AUPRC with a patient bootstrap CI and its prevalence baseline."""
    rng = np.random.default_rng(seed)
    y = patients["label"].to_numpy()
    p = patients[score_column].to_numpy()
    n = len(y)
    values = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, n, n)
        if len(np.unique(y[index])) < 2:
            continue
        values.append(average_precision_score(y[index], p[index]))
    return {
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "baseline": float(y.mean()),
        "ci_low": float(np.percentile(values, 2.5)) if values else float("nan"),
        "ci_high": float(np.percentile(values, 97.5)) if values else float("nan"),
    }


def reliability_curve(
    patients: pd.DataFrame, score_column: str = "prob_cal", n_bins: int = 10
) -> dict[str, list[float]]:
    """Equal-count reliability curve (E0's calibration readout)."""
    y = patients["label"].to_numpy()
    p = patients[score_column].to_numpy()
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return {
        "mean_predicted": [float(p[b].mean()) for b in bins if len(b)],
        "observed_rate": [float(y[b].mean()) for b in bins if len(b)],
        "n": [int(len(b)) for b in bins if len(b)],
    }
