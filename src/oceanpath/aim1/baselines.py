"""Non-image baselines, late fusion, and the adjusted association (§5.5-5.6).

Three questions, three models, all patient-level and all fitted on development
only:

``clinical``   age, sex, colon vs rectum (optionally + stage). Does routine
               clinicopathologic information already predict KRAS?
``technical``  native MPP, patch count, tissue fill, section-size class,
               colour class. A **negative control**, never a clinical model:
               if KRAS prevalence is predictable from acquisition metadata
               alone, the imaging result needs that context stated.
``fusion``     the out-of-fold WSI logit plus the clinical variables. Does the
               image add anything beyond the clinical baseline?

Development performance is always cross-fitted on the study's own five folds,
so a baseline is never scored on patients that shaped its coefficients. The
externally applied model is a single fit on all of development, applied
without refitting — the same discipline the WSI model follows.

Missing covariates are median-imputed with an explicit missing indicator
rather than dropped: RIH lacks sex for 48 slides and stage for 38, and SR1482
has no stage at all, so complete-case analysis would silently change which
patients each cohort's number describes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from oceanpath.aim1 import paths
from oceanpath.eval.external import logit

CLINICAL_NUMERIC = ["age_at_diagnosis"]
CLINICAL_CATEGORICAL = ["sex", "site_class"]
STAGE_COLUMN = "stage_class"

TECHNICAL_NUMERIC = [
    "native_mpp",
    "patch_count",
    "tissue_grid_occupancy",
    "tissue_area_mm2",
    "thumbnail_hue_median",
]
TECHNICAL_CATEGORICAL = ["section_size_class", "color_class"]


def _design(
    frame: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
    levels: dict[str, list[str]] | None = None,
    medians: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, list[str]], dict[str, float], dict[str, float]]:
    """Median-imputed numerics with missing indicators + one-hot categoricals.

    ``levels``, ``medians`` and ``scales`` are learned on development and passed
    back in when transforming an external cohort, so a category unseen in
    development cannot silently add a column and shift the coefficient
    alignment, and a numeric column cannot be standardized on the target's own
    spread. Carrying the scale matters as much as carrying the centre: a
    coefficient fitted against the training fold's standard deviation is
    meaningless applied to a column divided by a different fold's standard
    deviation, and the mismatch is silent.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []
    out_levels: dict[str, list[str]] = {}
    out_medians: dict[str, float] = {}
    out_scales: dict[str, float] = {}

    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce")
        median = (
            float(medians[column])
            if medians and column in medians
            else float(values.median(skipna=True))
        )
        missing = values.isna().to_numpy(dtype=float)
        filled = values.fillna(median).to_numpy(dtype=float)
        # Standardize on the development scale so the L2 penalty is not
        # dominated by whichever variable happens to be largest (patch counts
        # run to tens of thousands, MPP is ~0.25). Both the centre AND the
        # scale come from the fitting frame when one was supplied; deriving
        # either from the frame being transformed would leak the target's
        # distribution into a model fitted elsewhere.
        if scales and column in scales:
            scale = float(scales[column]) or 1.0
            centre = float(medians[column]) if medians and column in medians else median
        else:
            scale = float(np.std(filled)) or 1.0
            centre = median if medians else float(np.mean(filled))
        out_scales[column] = scale
        out_medians[column] = centre
        columns.append((filled - centre) / scale)
        names.append(column)
        if missing.any() or (medians and column in medians):
            columns.append(missing)
            names.append(f"{column}__missing")

    for column in categorical:
        values = frame[column].astype("string").fillna("unknown")
        category_levels = (
            levels[column] if levels and column in levels else sorted(values.unique().tolist())
        )
        out_levels[column] = category_levels
        for level in category_levels[1:]:  # drop-first
            columns.append((values == level).to_numpy(dtype=float))
            names.append(f"{column}={level}")

    return np.column_stack(columns), names, out_levels, out_medians, out_scales


class PatientLinearModel:
    """L2 logistic regression over patient-level covariates."""

    def __init__(self, numeric: Sequence[str], categorical: Sequence[str], name: str):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.name = name
        self.model: LogisticRegression | None = None
        self.levels: dict[str, list[str]] = {}
        self.medians: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.feature_names: list[str] = []

    def fit(self, frame: pd.DataFrame, weights: np.ndarray | None = None) -> PatientLinearModel:
        matrix, names, levels, medians, scales = _design(frame, self.numeric, self.categorical)
        self.feature_names, self.levels, self.medians = names, levels, medians
        self.scales = scales
        self.model = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
        self.model.fit(matrix, frame["label"].to_numpy(), sample_weight=weights)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        matrix, names, _, _, _ = _design(
            frame, self.numeric, self.categorical, self.levels, self.medians, self.scales
        )
        if names != self.feature_names:
            matrix = _align(matrix, names, self.feature_names)
        return self.model.predict_proba(matrix)[:, 1]

    def coefficients(self) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return dict(zip(self.feature_names, self.model.coef_[0].tolist(), strict=True))


def _align(matrix: np.ndarray, names: list[str], expected: list[str]) -> np.ndarray:
    """Reorder/pad a transformed matrix onto the development feature layout."""
    index = {name: position for position, name in enumerate(names)}
    columns = [
        matrix[:, index[name]] if name in index else np.zeros(len(matrix)) for name in expected
    ]
    return np.column_stack(columns)


def cross_fitted_predictions(
    dev: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
    name: str,
    fold_column: str = "k_fold",
) -> np.ndarray:
    """Development predictions from models that never saw the patient."""
    out = np.full(len(dev), np.nan)
    for fold in sorted(dev[fold_column].unique()):
        held_out = (dev[fold_column] == fold).to_numpy()
        model = PatientLinearModel(numeric, categorical, name).fit(dev[~held_out])
        out[held_out] = model.predict(dev[held_out])
    if np.isnan(out).any():
        raise RuntimeError(f"{name}: cross-fitting left patients unscored")
    return out


def evaluate_baseline(
    dev: pd.DataFrame,
    externals: dict[str, pd.DataFrame],
    numeric: Sequence[str],
    categorical: Sequence[str],
    name: str,
) -> dict[str, Any]:
    """Cross-fitted development AUROC + external AUROC with no refitting."""
    dev_scores = cross_fitted_predictions(dev, numeric, categorical, name)
    final = PatientLinearModel(numeric, categorical, name).fit(dev)
    block: dict[str, Any] = {
        "name": name,
        "features": list(numeric) + list(categorical),
        "coefficients": final.coefficients(),
        "development_oof": {
            "auroc": float(roc_auc_score(dev["label"], dev_scores)),
            "n": int(len(dev)),
        },
        "cohorts": {},
    }
    for group, frame in externals.items():
        scores = final.predict(frame)
        y = frame["label"].to_numpy()
        block["cohorts"][group] = {
            "auroc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
            "n": int(len(frame)),
            "n_mutant": int(y.sum()),
        }
    return block


def clinical_baseline(
    dev: pd.DataFrame, externals: dict[str, pd.DataFrame], with_stage: bool = False
) -> dict[str, Any]:
    """§5.5 clinical-only model. MSI, BRAF, NRAS, cohort, and technical
    metadata are deliberately excluded — this is the routine-information
    comparator, not a best-possible clinical model."""
    categorical = list(CLINICAL_CATEGORICAL) + ([STAGE_COLUMN] if with_stage else [])
    return evaluate_baseline(
        dev,
        externals,
        CLINICAL_NUMERIC,
        categorical,
        "clinical_with_stage" if with_stage else "clinical",
    )


def technical_baseline(dev: pd.DataFrame, externals: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """§5.5 negative control: is KRAS predictable from acquisition metadata?"""
    return evaluate_baseline(
        dev, externals, TECHNICAL_NUMERIC, TECHNICAL_CATEGORICAL, "technical_only"
    )


def fusion_baseline(
    dev: pd.DataFrame, externals: dict[str, pd.DataFrame], with_stage: bool = False
) -> dict[str, Any]:
    """§5.5 late fusion: WSI logit + clinical variables.

    The fusion model is fitted ONLY on development out-of-fold WSI logits and
    then applied to external WSI logits, so the image model's own training
    data never enters the fusion coefficients.
    """
    categorical = list(CLINICAL_CATEGORICAL) + ([STAGE_COLUMN] if with_stage else [])
    numeric = [*CLINICAL_NUMERIC, "wsi_logit"]

    dev = dev.copy()
    dev["wsi_logit"] = logit(dev["prob_raw"].to_numpy())
    prepared = {}
    for group, frame in externals.items():
        frame = frame.copy()
        frame["wsi_logit"] = logit(frame["prob_raw"].to_numpy())
        prepared[group] = frame
    return evaluate_baseline(
        dev, prepared, numeric, categorical, "wsi_plus_clinical" + ("_stage" if with_stage else "")
    )


# ── Adjusted association (§5.6) ───────────────────────────────────────────────

ADJUSTMENT_CATEGORICAL = ["msi_dmmr", "braf", "site_class", "sex", "subcohort"]


def adjusted_association(
    patients: pd.DataFrame,
    with_stage: bool = False,
    n_bootstrap: int = paths.N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Adjusted OR for KRAS per 1-SD increase in the WSI score.

    The WSI score enters as a standardized logit so the odds ratio is per
    standard deviation, comparable across cohorts with different score
    spreads. CIs come from a patient-clustered bootstrap of the whole fit, not
    from the model's own standard errors, so cohort composition uncertainty is
    included.
    """
    frame = patients.copy()
    score = logit(frame["prob_raw"].to_numpy())
    frame["wsi_z"] = (score - score.mean()) / (score.std() or 1.0)
    categorical = list(ADJUSTMENT_CATEGORICAL) + ([STAGE_COLUMN] if with_stage else [])
    numeric = ["wsi_z", *CLINICAL_NUMERIC]

    def _fit(block: pd.DataFrame) -> float:
        matrix, names, _, _, _ = _design(block, numeric, categorical)
        model = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
        model.fit(matrix, block["label"].to_numpy())
        return float(model.coef_[0][names.index("wsi_z")])

    point = _fit(frame)
    rng = np.random.default_rng(seed)
    n = len(frame)
    samples = []
    for _ in range(min(n_bootstrap, 500)):  # each replicate refits the model
        resampled = frame.iloc[rng.integers(0, n, n)]
        if len(np.unique(resampled["label"])) < 2:
            continue
        try:
            samples.append(_fit(resampled))
        except ValueError:
            continue
    return {
        "with_stage": with_stage,
        "n": int(n),
        "log_odds_per_sd": point,
        "odds_ratio_per_sd": float(np.exp(point)),
        "ci_low": float(np.exp(np.percentile(samples, 2.5))) if samples else float("nan"),
        "ci_high": float(np.exp(np.percentile(samples, 97.5))) if samples else float("nan"),
        "adjusted_for": numeric[1:] + categorical,
        "n_bootstrap": len(samples),
    }


# ── Overlap weighting for 1C-B (§6.2) ─────────────────────────────────────────

BALANCE_CATEGORICAL = [
    "subcohort",
    "site_class",
    "stage_class",
    "sex",
    "msi_dmmr",
    "braf",
    "mpp_bin",
    "section_size_class",
    "color_class",
]
BALANCE_NUMERIC = ["age_at_diagnosis", "patch_count", "tissue_area_mm2"]


def overlap_weights(dev: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Overlap weights from a development-only propensity model for KRAS.

    Overlap weighting (w = 1 - e for mutants, e for wild-types) rather than
    inverse-probability weighting: it is bounded by construction, so a patient
    in a covariate region where one class is nearly absent cannot acquire an
    enormous weight and dominate training. Weight mass concentrates where the
    two classes actually overlap, which is the region where a morphologic
    comparison is meaningful at all.
    """
    matrix, names, _, _, _ = _design(dev, BALANCE_NUMERIC, BALANCE_CATEGORICAL)
    y = dev["label"].to_numpy()
    model = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
    model.fit(matrix, y)
    propensity = model.predict_proba(matrix)[:, 1]
    weights = np.where(y == 1, 1.0 - propensity, propensity)

    effective = float(weights.sum() ** 2 / (weights**2).sum())
    return weights, {
        "propensity_auc": float(roc_auc_score(y, propensity)),
        "covariates": BALANCE_NUMERIC + BALANCE_CATEGORICAL,
        "n_features": len(names),
        "effective_sample_size": effective,
        "effective_fraction": effective / len(weights),
        "weight_quantiles": {
            str(q): float(np.percentile(weights, q)) for q in (0, 5, 25, 50, 75, 95, 100)
        },
        "weight_sum_by_class": {
            "mutant": float(weights[y == 1].sum()),
            "wild_type": float(weights[y == 0].sum()),
        },
    }
