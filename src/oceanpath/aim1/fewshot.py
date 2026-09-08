"""E3 — few-shot local adaptation: support protocol and adaptation mechanisms.

Three protocol corrections the obvious approach gets wrong, all implemented
here rather than left to the runner:

1. **Rotating outer folds, not support-removal.** Taking support patients out
   of the test set shrinks the denominator at every point on the learning
   curve, so curves at different support sizes are not comparable. Instead the
   target metastatic patients are split into 5 outer folds; support is drawn
   only from the 4 non-test folds; every support size is evaluated on the SAME
   full metastatic population.

2. **Never average predictions across repeats before scoring.** Averaging R
   adapted models while the zero-shot arm is a single deterministic model hands
   adaptation a variance-reduction advantage that is an artefact, not a result.
   The metric is computed WITHIN each repeat and the distribution over repeats
   is reported; inference uses the paired per-repeat difference.

3. **Dual-specimen leakage rule.** A patient's primary leaves the
   local-primary support pool whenever that patient's metastasis is in the
   test fold.

Fold stratification is by KRAS label only. Metastatic-site cells are smaller
than the fold count (RIH peritoneum 3 mut / 4 WT, SurGen lung 2 / 1), so site
stratification is arithmetically impossible; site composition is instead
reported per fold and used only as a tie-break.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

N_OUTER_FOLDS = 5
SUPPORT_SIZES = (2, 4, 8)  # per class; 16/class is cut (support exhausts the pool)
N_REPEATS = 20

# Adaptation mechanisms, by trainable parameter count.
MECHANISMS = {
    "M1_recalibration": 2,  # Platt intercept + slope: local prevalence/scale only
    "M2_head": 1025,  # final linear classifier: local decision boundary
    "M3_attention_head": 1282,  # + attention scoring vector (Tier 3, needed by E4c)
}

SUPPORT_CONDITIONS = {
    "S0_zero_shot": "no target data",
    "S1_local_primary": "target primary — institution + platform + population shift",
    "S2_local_metastatic": "target metastatic — + specimen-role/organ shift",
    "S5_full_target": "all target primary + metastatic, CV — upper bound",
}


def assign_outer_folds(patients: pd.DataFrame, seed: int = 42) -> pd.Series:
    """5 label-stratified, patient-grouped outer folds over the target mets."""
    rng = np.random.default_rng(seed)
    folds = pd.Series(-1, index=patients.index, dtype=int)
    for _, block in patients.groupby("target_label"):
        order = rng.permutation(block.index.to_numpy())
        for position, idx in enumerate(order):
            folds.loc[idx] = position % N_OUTER_FOLDS
    return folds


def draw_support(
    pool: pd.DataFrame,
    per_class: int,
    rng: np.random.Generator,
    exclude_patients: set[str],
) -> pd.DataFrame | None:
    """Balanced support set drawn from the non-test folds.

    Returns None when either class cannot be filled, which is how an
    over-large support size is refused rather than silently unbalanced.
    """
    usable = pool[~pool["patient_id"].isin(exclude_patients)]
    picks = []
    for label in (0, 1):
        block = usable[usable["target_label"] == label]["patient_id"].unique()
        if len(block) < per_class:
            return None
        picks.extend(rng.choice(block, per_class, replace=False).tolist())
    return usable[usable["patient_id"].isin(picks)]


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """M1: two parameters, intercept and slope, on the target logit scale."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(labels)) < 2:
        return 0.0, 1.0
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(logits.reshape(-1, 1), labels)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def apply_platt(logits: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(intercept + slope * logits)))


def paired_repeat_summary(deltas: list[float]) -> dict[str, Any]:
    """Distribution of a per-repeat paired difference (e.g. S1 vs S2)."""
    values = np.asarray([d for d in deltas if np.isfinite(d)], dtype=float)
    if not len(values):
        return {"n_repeats": 0}
    return {
        "n_repeats": int(len(values)),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def tost_equivalence(deltas: list[float], margin: float = 0.075) -> dict[str, Any]:
    """Two one-sided tests for S1 ~ S2 at a PRE-SPECIFIED margin.

    Margin 0.075, not 0.05: at the realised correlation between conditions
    (they share a frozen encoder and differ by ~1k parameters) a 0.05 margin
    has ~38% power and effectively none if correlation drops to 0.7. Declaring
    equivalence at an underpowered margin would be the same error as declaring
    a difference from an underpowered test.
    """
    from scipy import stats

    values = np.asarray([d for d in deltas if np.isfinite(d)], dtype=float)
    if len(values) < 3:
        return {"equivalent": None, "note": "too few repeats"}
    lower = stats.ttest_1samp(values, -margin, alternative="greater")
    upper = stats.ttest_1samp(values, margin, alternative="less")
    p = max(float(lower.pvalue), float(upper.pvalue))
    return {
        "margin": margin,
        "mean_delta": float(values.mean()),
        "p_tost": p,
        "equivalent": bool(p < 0.05),
    }
