"""Shared machinery for locked external-validation experiments.

Study-agnostic statistical pieces used by the frozen-model phases:

  - patient aggregation by mean slide logit → sigmoid;
  - a Platt calibrator fitted ONLY on development patient-level OOF;
  - one operating threshold locked on calibrated OOF;
  - patient-resampled bootstrap CIs for every reported metric.

The KRAS study composes these through ``oceanpath.kras.study`` and the phase
6/7 CLIs; nothing here ever fits on external data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def to_patient_level(slide_df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Patient probability = sigmoid(mean slide logit) (§44). Raw scale."""
    df = slide_df.merge(manifest[["slide_id", "patient_id"]], on="slide_id", validate="one_to_one")
    df["logit"] = logit(df["prob_1"].to_numpy())
    grouped = df.groupby("patient_id").agg(
        label=("label", "max"),  # patient RAS labels are slide-consistent (verified at build)
        mean_logit=("logit", "mean"),
        n_slides=("slide_id", "count"),
    )
    grouped["prob_raw"] = sigmoid(grouped["mean_logit"].to_numpy())
    return grouped.reset_index()[["patient_id", "label", "prob_raw", "n_slides"]]


def fit_platt_on_oof(oof_patients: pd.DataFrame) -> dict:
    """Fit sigmoid(a + b*logit(p)) on patient-level OOF; fall back to
    intercept-only (b=1) if the slope MLE is unstable or non-monotone."""
    from oceanpath.eval.core import compute_calibration_intercept_slope

    y = oof_patients["label"].to_numpy()
    p = oof_patients["prob_raw"].to_numpy()
    fit = compute_calibration_intercept_slope(y, p)
    slope = fit["calibration_slope"]
    if fit.get("slope_converged") and np.isfinite(slope) and slope > 0:
        return {"a": float(fit["slope_model_intercept"]), "b": float(slope), "form": "platt"}
    logger.warning("Platt slope unstable (%s) — falling back to intercept-only", slope)
    return {"a": float(fit["calibration_intercept"]), "b": 1.0, "form": "intercept_only"}


def apply_calibrator(p_raw: np.ndarray, calibrator: dict) -> np.ndarray:
    return sigmoid(calibrator["a"] + calibrator["b"] * logit(p_raw))


def lock_threshold(oof_patients: pd.DataFrame, sens_target: float) -> dict:
    """§17: max specificity s.t. sensitivity >= target, on calibrated OOF."""
    y = oof_patients["label"].to_numpy()
    p = oof_patients["prob_cal"].to_numpy()
    fpr, tpr, thresholds = roc_curve(y, p)
    spec = 1.0 - fpr
    valid = tpr >= sens_target
    if not valid.any():
        raise ValueError(f"OOF cannot reach sensitivity >= {sens_target}")
    idx = np.where(valid)[0][np.argmax(spec[valid])]
    return {
        "threshold": float(thresholds[idx]),
        "sens_target": float(sens_target),
        "oof_sensitivity": float(tpr[idx]),
        "oof_specificity": float(spec[idx]),
        "oof_n_patients": int(len(oof_patients)),
        "oof_auroc_patient": float(roc_auc_score(y, p)),
        "oof_brier_calibrated": float(brier_score_loss(y, p)),
    }


def operating_point(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def _safe(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    return {
        "sensitivity": _safe(tp, tp + fn),
        "specificity": _safe(tn, tn + fp),
        "ppv": _safe(tp, tp + fp),
        "npv": _safe(tn, tn + fn),
        "accuracy": _safe(tp + tn, len(y)),
    }


def point_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    two_class = len(np.unique(y)) == 2
    out = {
        "auroc": float(roc_auc_score(y, p)) if two_class else float("nan"),
        "auprc": float(average_precision_score(y, p)) if two_class else float("nan"),
        "brier": float(brier_score_loss(y, p)),
    }
    out.update(operating_point(y, p, thr))
    return out


def bootstrap_ci(
    y_all: np.ndarray, p_all: np.ndarray, thr: float, n_bootstrap: int, seed: int
) -> dict[str, tuple[float, float]]:
    """Patient-resampled percentile CIs for every point metric."""
    rng = np.random.default_rng(seed)
    n = len(y_all)
    samples: dict[str, list[float]] = {}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y, p = y_all[idx], p_all[idx]
        if len(np.unique(y)) < 2:
            continue
        for key, value in point_metrics(y, p, thr).items():
            samples.setdefault(key, []).append(value)
    return {
        key: (
            float(np.nanpercentile(values, 2.5)),
            float(np.nanpercentile(values, 97.5)),
        )
        for key, values in samples.items()
    }


def contrast_bootstrap(
    primary: pd.DataFrame,
    metastatic: pd.DataFrame,
    thr: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict]:
    """Unpaired bootstrap for role contrasts (§31/§33).

    Primary and metastatic groups contain different patient-role cases, so the
    two groups are resampled independently and the delta metastatic − primary
    is recomputed each round. Returns {metric: {delta, ci}} for auroc, brier
    and sensitivity/specificity at the locked threshold.
    """
    rng = np.random.default_rng(seed)
    keys = ("auroc", "brier", "sensitivity", "specificity")
    yp, pp = primary["label"].to_numpy(), primary["prob_cal"].to_numpy()
    ym, pm = metastatic["label"].to_numpy(), metastatic["prob_cal"].to_numpy()
    point_p = point_metrics(yp, pp, thr)
    point_m = point_metrics(ym, pm, thr)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_bootstrap):
        ip = rng.integers(0, len(yp), len(yp))
        im = rng.integers(0, len(ym), len(ym))
        if len(np.unique(yp[ip])) < 2 or len(np.unique(ym[im])) < 2:
            continue
        mp = point_metrics(yp[ip], pp[ip], thr)
        mm = point_metrics(ym[im], pm[im], thr)
        for k in keys:
            samples[k].append(mm[k] - mp[k])
    return {
        k: {
            "delta": point_m[k] - point_p[k],
            "ci": (
                float(np.nanpercentile(v, 2.5)),
                float(np.nanpercentile(v, 97.5)),
            )
            if v
            else (float("nan"), float("nan")),
        }
        for k, v in samples.items()
    }


def paired_delta_bootstrap(
    base: pd.DataFrame,
    other: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict]:
    """Paired bootstrap for two models evaluated on the SAME patients (§42).

    Both frames need patient_id, label, prob_cal over an identical patient
    set; each round resamples one shared patient index vector and recomputes
    the delta other − base for AUROC and Brier.
    """
    merged = base[["patient_id", "label", "prob_cal"]].merge(
        other[["patient_id", "prob_cal"]],
        on="patient_id",
        suffixes=("_base", "_other"),
        validate="one_to_one",
    )
    if len(merged) != len(base) or len(merged) != len(other):
        raise ValueError("paired bootstrap requires identical patient sets")
    y = merged["label"].to_numpy()
    pb = merged["prob_cal_base"].to_numpy()
    po = merged["prob_cal_other"].to_numpy()

    def _deltas(yy, b, o) -> dict[str, float]:
        return {
            "auroc": float(roc_auc_score(yy, o) - roc_auc_score(yy, b)),
            "brier": float(brier_score_loss(yy, o) - brier_score_loss(yy, b)),
        }

    point = _deltas(y, pb, po)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        for k, v in _deltas(y[idx], pb[idx], po[idx]).items():
            samples[k].append(v)
    return {
        k: {
            "delta": point[k],
            "ci": (
                float(np.nanpercentile(v, 2.5)),
                float(np.nanpercentile(v, 97.5)),
            )
            if v
            else (float("nan"), float("nan")),
        }
        for k, v in samples.items()
    }


def calibration_block(y: np.ndarray, p: np.ndarray) -> dict:
    from oceanpath.eval.core import compute_calibration, compute_calibration_intercept_slope

    return {
        **compute_calibration_intercept_slope(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": compute_calibration(y, p)["ece"],
    }


def evaluate_group(patients: pd.DataFrame, thr: float, n_bootstrap: int, seed: int) -> dict:
    """Full metric block for one (cohort, role, strategy) patient table.

    ``patients`` needs columns label, prob_raw, prob_cal.
    """
    y = patients["label"].to_numpy()
    p_cal = patients["prob_cal"].to_numpy()
    p_raw = patients["prob_raw"].to_numpy()
    return {
        "n_patients": int(len(patients)),
        "n_ras_mutant": int(y.sum()),
        "n_ras_wildtype": int(len(y) - y.sum()),
        "threshold": float(thr),
        **point_metrics(y, p_cal, thr),
        "calibration": calibration_block(y, p_cal),
        "raw": {
            "brier": float(brier_score_loss(y, p_raw)),
            "calibration": calibration_block(y, p_raw),
        },
        "ci95": bootstrap_ci(y, p_cal, thr, n_bootstrap, seed),
    }


def fmt_metric(value: float, ci: tuple[float, float] | None = None, digits: int = 3) -> str:
    if value is None or value != value:  # NaN
        return "unstable"
    text = f"{value:.{digits}f}"
    if ci is not None:
        text += f" ({ci[0]:.{digits}f}–{ci[1]:.{digits}f})"
    return text
