#!/usr/bin/env python3
"""E1d — clinicopathologic baseline (Tier 2, ~0 new fits).

The question: does routine clinicopathologic information already predict KRAS,
and how much does the image add on top of it?

  (i)   clinical   L2 logistic regression on age, sex, site, stage
  (ii)  WSI        the frozen E0 out-of-fold score, unchanged

Evaluated on set G (stage known) so both models see IDENTICAL patients on
IDENTICAL folds. MSI and BRAF are deliberately EXCLUDED: they are molecular
assay results that do not exist at prediction time in the stated use case, so
including them would make the baseline stronger and the paper weaker.

INFERENCE — paired PATIENT BOOTSTRAP, not DeLong, not a likelihood-ratio test.
Three deliberate omissions, each for a stated reason:

* **No likelihood-ratio test.** An LR test is a statement about nested
  likelihoods fitted on the same data. Applied to cross-validated predictions
  it has no valid null distribution — the "parameters" it counts were not
  fitted on the rows being scored. It was reported in an earlier draft of this
  script and is now removed rather than demoted.
* **No net reclassification improvement.** Unstable at these strata counts and
  it adds a threshold choice the design does not need.
* **DeLong is not the primary inference.** DeLong assumes the two score vectors
  are fixed functions evaluated on one sample. Here each score is itself the
  output of a cross-validated fitting procedure, so DeLong's variance omits the
  fold-fitting component and can understate uncertainty. It is still computed
  and reported alongside, clearly labelled, because it is the number a reader
  will look for.

The primary comparison resamples PATIENTS once and recomputes BOTH models on
that same resample, so the two are paired exactly as they are in the data.
Reported for AUROC, AUPRC, Brier, log loss, and calibration.

Both score columns are passed through the SAME cross-fitted Platt calibrator
before any probability-scale metric, so Brier/log-loss/calibration compare the
models and not their differing output conventions.

Usage:
    python aim1_clinical_baseline.py --condition uncapped
    python aim1_clinical_baseline.py --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1.cli import e1a_challenge as e1a  # noqa: E402
from oceanpath.aim1 import baselines, evaluate, paths  # noqa: E402
from oceanpath.eval.external import logit  # noqa: E402

TRAIN = paths.OUTPUT_ROOT / "train"
CONDITIONS = {
    "uncapped": (TRAIN / "1a" / "univ1", "full bags"),
    "cap2048": (TRAIN / "1a_cap2048" / "univ1", "2048 random tiles/epoch"),
    "capped": (TRAIN / "1a_cap4096" / "univ1", "4096 random tiles/epoch"),
    "cap8192": (TRAIN / "1a_cap8192" / "univ1", "8192 random tiles/epoch"),
    "cap16384": (TRAIN / "1a_cap16384" / "univ1", "16384 random tiles/epoch"),
    "pb_cap4096": (TRAIN / "1a_pb_cap4096" / "univ1", "PATIENT-BALANCED, 4096 tiles/epoch"),
    "pb_cap8192": (TRAIN / "1a_pb_cap8192" / "univ1", "PATIENT-BALANCED, 8192 tiles/epoch"),
    "conch": (TRAIN / "1a_conch" / "conch_v15", "CONCH v1.5, full bags"),
    "v2cls": (TRAIN / "1a_pb_cap4096" / "virchow2_cls", "VIRCHOW2 CLS, 4096 tiles/epoch, patient-balanced"),
}
SEEDS = (42, 43, 44)
N_BOOT = 2000

NUMERIC = list(baselines.CLINICAL_NUMERIC)  # age_at_diagnosis
CATEGORICAL = list(baselines.CLINICAL_CATEGORICAL) + [baselines.STAGE_COLUMN]  # sex, site, stage


# ── design-matrix alignment ──────────────────────────────────────────────────
def _align(matrix: np.ndarray, names: list[str], target: list[str]) -> np.ndarray:
    """Reindex a design matrix onto the training fold's column order.

    ``baselines._design`` emits a ``__missing`` indicator only when a fold
    actually has missing values, so a train fold with complete ages and a test
    fold with one missing age produce different widths. Aligning by NAME rather
    than position keeps a coefficient attached to its own variable; columns
    absent from a fold are zero, which is exactly what the indicator means.
    """
    lookup = {n: i for i, n in enumerate(names)}
    cols = [matrix[:, lookup[n]] if n in lookup else np.zeros(len(matrix)) for n in target]
    return np.column_stack(cols)


def cross_fitted_scores(frame: pd.DataFrame, numeric, categorical) -> np.ndarray:
    """Out-of-fold predictions using E0's own k_fold assignment."""
    out = np.full(len(frame), np.nan)
    idx = np.arange(len(frame))
    for fold in sorted(frame["k_fold"].unique()):
        test = frame["k_fold"].to_numpy() == fold
        train = ~test
        xtr, names, levels, medians, scales = baselines._design(
            frame[train], numeric, categorical
        )
        xte, te_names, _, _, _ = baselines._design(
            frame[test], numeric, categorical, levels, medians, scales
        )
        model = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
        model.fit(xtr, frame.loc[train, "label"].to_numpy())
        out[idx[test]] = model.predict_proba(_align(xte, te_names, names))[:, 1]
    assert np.isfinite(out).all(), "every patient must receive an out-of-fold score"
    return out


def platt(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Cross-fitted Platt on E0's folds, applied identically to every model."""
    block = frame[["label", "k_fold"]].copy()
    block["prob_raw"] = np.clip(score, 1e-6, 1 - 1e-6)
    return evaluate.cross_fitted_platt(block)["prob_cal"].to_numpy()


# ── metrics ──────────────────────────────────────────────────────────────────
METRIC_KEYS = ("auroc", "auprc", "brier", "log_loss", "cal_intercept", "cal_slope")
# Metrics where LOWER is better, so a "WSI - clinical" delta should be negative.
LOWER_BETTER = {"brier", "log_loss"}

def calibration_pair(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration intercept and slope, via the study's established estimator.

    Reused from ``oceanpath.eval.core`` rather than re-derived here, so the
    numbers on this sheet mean exactly what the same-named numbers on the E0
    sheet mean.
    """
    from oceanpath.eval.core import compute_calibration_intercept_slope

    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    block = compute_calibration_intercept_slope(y, np.clip(p, 1e-6, 1 - 1e-6))
    return float(block["calibration_intercept"]), float(block["calibration_slope"])


def metrics(y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray) -> dict[str, float]:
    """Rank metrics on RAW scores, probability metrics on CALIBRATED scores.

    The split is deliberate. AUROC and AUPRC are rank statistics and a
    calibrator is meant to be monotone, so calibrating first should not move
    them — except that a CROSS-FITTED Platt fits a different map per fold, and
    stitching five differently-scaled folds back together does reorder patients
    across fold boundaries. Scoring rank metrics on the raw score avoids that
    artefact and keeps this sheet's WSI AUROC identical to E0's, which is the
    same model on the same patients and must not disagree with itself.

    Brier, log loss and calibration are probability-scale and are meaningless
    on an uncalibrated score, so those take the calibrated column.
    """
    if len(np.unique(y)) < 2:
        return dict.fromkeys(METRIC_KEYS, float("nan"))
    intercept, slope = calibration_pair(y, p_cal)
    return {
        "auroc": float(roc_auc_score(y, p_raw)),
        "auprc": float(average_precision_score(y, p_raw)),
        "brier": float(brier_score_loss(y, p_cal)),
        "log_loss": float(log_loss(y, np.clip(p_cal, 1e-6, 1 - 1e-6))),
        "cal_intercept": intercept,
        "cal_slope": slope,
    }


def paired_patient_bootstrap(
    y: np.ndarray,
    wsi: tuple[np.ndarray, np.ndarray],
    clin: tuple[np.ndarray, np.ndarray],
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict:
    """Resample PATIENTS once; recompute both models on that same resample.

    This is the paired estimator the design calls for. Patients are the
    independent unit; the two models are evaluated on identical resampled
    patients, so their difference carries the correlation the data actually has
    rather than the zero correlation an unpaired interval would assume.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    (w_raw, w_cal), (c_raw, c_cal) = wsi, clin
    point = {
        "wsi": metrics(y, w_raw, w_cal),
        "clinical": metrics(y, c_raw, c_cal),
    }
    deltas: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        mw = metrics(yb, w_raw[idx], w_cal[idx])
        mc = metrics(yb, c_raw[idx], c_cal[idx])
        for k in METRIC_KEYS:
            if np.isfinite(mw[k]) and np.isfinite(mc[k]):
                deltas[k].append(mw[k] - mc[k])
    out = {"point": point, "delta": {}}
    for k in METRIC_KEYS:
        v = np.asarray(deltas[k], dtype=float)
        if not len(v):
            out["delta"][k] = dict.fromkeys(("delta", "ci_low", "ci_high", "p_two_sided"), float("nan"))
            continue
        d = point["wsi"][k] - point["clinical"][k]
        # bootstrap two-sided p: how often the resampled difference crosses zero
        frac = float(np.mean(v <= 0)) if d > 0 else float(np.mean(v >= 0))
        out["delta"][k] = {
            "delta": float(d),
            "ci_low": float(np.percentile(v, 2.5)),
            "ci_high": float(np.percentile(v, 97.5)),
            "p_two_sided": float(min(1.0, 2 * max(frac, 1.0 / len(v)))),
            "lower_is_better": k in LOWER_BETTER,
        }
    return out


# ── DeLong, reported alongside but NOT primary ───────────────────────────────
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_test(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict:
    """Secondary only. Its variance omits the cross-validation fitting
    component, so it can understate uncertainty here."""
    m, n = int((y == 1).sum()), int((y == 0).sum())
    scores = np.vstack([s1, s2])
    k = scores.shape[0]
    tx, ty, tz = np.empty([k, m]), np.empty([k, n]), np.empty([k, m + n])
    for r in range(k):
        px, nx = scores[r][y == 1], scores[r][y == 0]
        tx[r], ty[r], tz[r] = _midrank(px), _midrank(nx), _midrank(np.r_[px, nx])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01) / m + np.cov(v10) / n)
    contrast = np.array([[1.0, -1.0]])
    var = float((contrast @ cov @ contrast.T).item())
    delta = float(aucs[0] - aucs[1])
    z = delta / np.sqrt(var) if var > 0 else np.nan
    return {
        "delta": delta,
        "se": float(np.sqrt(var)) if var > 0 else float("nan"),
        "p_value": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan"),
        "caveat": "secondary; variance omits the CV fitting component",
    }


def macro_loco(frame: pd.DataFrame, score: np.ndarray) -> float:
    """Mean of per-cohort AUROCs. Floor is exactly 0.500 by construction."""
    vals = []
    for _c, blk in frame.assign(_s=score).groupby("cohort"):
        y = blk["label"].to_numpy()
        if len(np.unique(y)) > 1:
            vals.append(roc_auc_score(y, blk["_s"].to_numpy()))
    return float(np.mean(vals)) if vals else float("nan")


def load_seed(root: Path, seed: int) -> pd.DataFrame | None:
    path = root / f"seed{seed}" / "oof_predictions.parquet"
    if not path.is_file():
        return None
    manifest = pd.read_csv(paths.DEV_MANIFEST)
    patients = evaluate.to_patient_level(pd.read_parquet(path), manifest)
    stage = manifest.drop_duplicates("patient_id")[["patient_id", "stage_group_major"]]
    return patients.merge(stage, on="patient_id", how="left")


def run_seed(frame: pd.DataFrame, population: str = "G") -> dict:
    """population 'A' = all 1,486 primary patients; 'G' = stage-known (n=1,105).

    The full-population arm is primary: restricting to stage-known throws away
    26% of patients and, worse, conditions the comparison on a variable whose
    MISSINGNESS is itself informative (RIH and SR1482 carry most of it). Stage
    enters the A arm as a category with 'unknown' as an explicit level, which is
    the missingness indicator — so a patient with no recorded stage contributes
    to the model instead of being deleted from it. Age missingness is handled
    the same way by `_design`, which emits a median-imputed column plus an
    explicit `__missing` flag; both the median and the scale come from the
    TRAINING folds only.
    """
    g = (frame if population == "A" else frame[e1a.subset_mask(frame, "G_stage_known")])
    g = g.reset_index(drop=True)
    y = g["label"].to_numpy()

    clin_raw = cross_fitted_scores(g, NUMERIC, CATEGORICAL)
    wsi_raw = g["prob_raw"].to_numpy()
    wsi_cal = platt(g, wsi_raw)
    clin_cal = platt(g, clin_raw)

    boot = paired_patient_bootstrap(y, (wsi_raw, wsi_cal), (clin_raw, clin_cal))

    # The fusion contrasts. "WSI beats clinical" supports only "the image model
    # outperformed the clinical comparator". The claim that IMAGE INFORMATION
    # ADDS BEYOND clinical variables is a different comparison and needs the
    # nested pair: clinical+WSI against clinical alone. And "fusion was no
    # better than WSI" cannot rest on a point comparison either — it needs the
    # paired interval, which is what decides whether the two are separable.

    # Secondary, clearly labelled: the stacked model. Its WSI input is out of
    # fold and its coefficients are fitted off-fold, but the logits it trains on
    # are themselves CV outputs, which is the standard stacking caveat.
    g2 = g.assign(wsi_logit=logit(g["prob_raw"].to_numpy()))
    fusion_raw = cross_fitted_scores(g2, [*NUMERIC, "wsi_logit"], CATEGORICAL)
    fusion_cal = platt(g, fusion_raw)
    boot_fc = paired_patient_bootstrap(y, (fusion_raw, fusion_cal), (clin_raw, clin_cal))
    boot_fw = paired_patient_bootstrap(y, (fusion_raw, fusion_cal), (wsi_raw, wsi_cal))

    return {
        "population": population,
        "n": int(len(g)),
        "n_mutant": int(y.sum()),
        "n_stage_unknown": int((g["stage_class"].astype(str) == "unknown").sum()),
        "n_age_missing": int(pd.to_numeric(g["age_at_diagnosis"], errors="coerce").isna().sum()),
        "wsi": {**boot["point"]["wsi"], "macro_loco": macro_loco(g, wsi_raw)},
        "clinical": {**boot["point"]["clinical"], "macro_loco": macro_loco(g, clin_raw)},
        "delta_wsi_minus_clinical": boot["delta"],
        "delta_fusion_minus_clinical": boot_fc["delta"],
        "delta_fusion_minus_wsi": boot_fw["delta"],
        "delong_secondary": delong_test(y, wsi_raw, clin_raw),
        "fusion": {
            **metrics(y, fusion_raw, fusion_cal),
            "macro_loco": macro_loco(g, fusion_raw),
            "cross_fitted": "coefficients from E0 folds; Platt cross-fitted; no threshold layer",
        },
    }


def fmt(values: list[float], nd: int = 4) -> str:
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return "pending"
    if len(v) == 1:
        return f"{v[0]:.{nd}f} (1 seed)"
    return f"{np.median(v):.{nd}f} [{min(v):.{nd}f}-{max(v):.{nd}f}]"


def report_condition(name: str, population: str = "G") -> dict:
    root, label = CONDITIONS[name]
    frames = {s: f for s in SEEDS if (f := load_seed(root, s)) is not None}
    if not frames:
        print(f"{name}: no completed seeds yet")
        return {}
    per_seed = {s: run_seed(f, population) for s, f in sorted(frames.items())}
    seeds = sorted(per_seed)

    tag = ("ALL PRIMARY (n=1,486)" if population == "A" else "set G · stage-known (n=1,105)")
    print(f"\n{'=' * 86}\nE1d · {name.upper()} ({label}) · {tag} · seeds {seeds}\n{'=' * 86}")
    print(f"  stage unknown: {per_seed[seeds[0]]['n_stage_unknown']}   "
          f"age missing: {per_seed[seeds[0]]['n_age_missing']}   "
          f"(carried as explicit indicator levels, never deleted)")
    print(f"  n = {per_seed[seeds[0]]['n']} ({per_seed[seeds[0]]['n_mutant']} mutant)")
    print(f"\n{'metric':16s} {'(i) clinical':>24s} {'(ii) WSI':>24s}")
    for k in METRIC_KEYS:
        arrow = " (lower better)" if k in LOWER_BETTER else ""
        print(f"  {k + arrow:26s} {fmt([per_seed[s]['clinical'][k] for s in seeds]):>22s} "
              f"{fmt([per_seed[s]['wsi'][k] for s in seeds]):>24s}")
    print(f"  {'macro-LOCO AUROC':26s} {fmt([per_seed[s]['clinical']['macro_loco'] for s in seeds]):>22s} "
          f"{fmt([per_seed[s]['wsi']['macro_loco'] for s in seeds]):>24s}")

    print("\n  PAIRED PATIENT BOOTSTRAP  (ii) - (i), 2000 resamples, patient is the unit:")
    for k in METRIC_KEYS:
        d = [per_seed[s]["delta_wsi_minus_clinical"][k]["delta"] for s in seeds]
        lo = [per_seed[s]["delta_wsi_minus_clinical"][k]["ci_low"] for s in seeds]
        hi = [per_seed[s]["delta_wsi_minus_clinical"][k]["ci_high"] for s in seeds]
        crosses = any(l <= 0 <= h for l, h in zip(lo, hi))
        mark = "" if not crosses else "   (CI crosses 0 in >=1 seed)"
        print(f"    {k:16s} {fmt(d):>24s}   95% CI [{np.median(lo):+.4f}, {np.median(hi):+.4f}]{mark}")
    dl = [per_seed[s]["delong_secondary"]["delta"] for s in seeds]
    print(f"    {'DeLong (secondary)':16s} {fmt(dl):>24s}   variance omits CV fitting")

    for key, title in (("delta_fusion_minus_clinical",
                        "FUSION (clinical+WSI) - CLINICAL  [does the image ADD?]"),
                       ("delta_fusion_minus_wsi",
                        "FUSION (clinical+WSI) - WSI       [do the clinical vars ADD?]")):
        print(f"\n  {title}")
        for m in METRIC_KEYS:
            d = [per_seed[s][key][m]["delta"] for s in seeds]
            lo = np.median([per_seed[s][key][m]["ci_low"] for s in seeds])
            hi = np.median([per_seed[s][key][m]["ci_high"] for s in seeds])
            flag = "  crosses 0" if lo <= 0 <= hi else "  EXCLUDES 0"
            print(f"    {m:16s} {fmt(d):>24s}   95% CI [{lo:+.4f}, {hi:+.4f}]{flag}")
    fu = [per_seed[s]["fusion"]["auroc"] for s in seeds]
    print(f"\n    fusion AUROC {fmt(fu)}")
    return {"condition": name, "description": label, "seeds_complete": seeds, "per_seed": per_seed}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--condition", choices=list(CONDITIONS), default=None)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--population", choices=["A", "G", "both"], default="G",
                   help="A = all 1,486 primary (primary analysis); G = stage-known sensitivity")
    a = p.parse_args()
    names = list(CONDITIONS) if a.compare else [a.condition or "uncapped"]
    pops = ["A", "G"] if a.population == "both" else [a.population]
    results = {}
    for pop in pops:
        for n in names:
            block = report_condition(n, pop)
            if block:
                results[n if pop == "G" else f"{n}__A"] = block
    dest = paths.EVAL_ROOT / "e1d.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # MERGE, never replace. A single-condition run must not silently delete the
    # arms it did not recompute — the workbook reads this file as the record of
    # every arm, and a --condition run would otherwise report finished arms as
    # "pending".
    existing = json.loads(dest.read_text()) if dest.is_file() else {}
    existing.update({k: v for k, v in results.items() if v})
    dest.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
