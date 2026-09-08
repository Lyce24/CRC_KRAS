#!/usr/bin/env python3
"""E2d (SurGen component) — SR386 vs SR1482, anatomy of the subcohort gap (0 new fits).

E2d asks whether Aim 2's result is robust to RIH acquisition batch, metastatic
organ, paired specimens and SurGen subcohort. This file is the SurGen-subcohort
component.  The metastatic-site and paired-specimen panels live in e2d1/e2d2
and e2d5; the dependency-restricted repetition lives in e2d3; and the RIH
acquisition/processing-regime sensitivity lives in e2d6.  E2d6 does not claim
an identified scanner effect because repair, resolution, platform and era are
confounded.

The largest single effect in the study is not between institutions: holding
SurGen out, the shared SurGen-naive model scores SR386 at ~0.74 and SR1482 at
~0.63. Both subsets come from Scotland, but the SurGen data paper defines them
as distinct collections with different composition and different MSI/MMR
ascertainment — so this is a SURGEN SUBCOHORT GAP, not established as a
within-institution gap, and must be worded that way.

Both subsets are scored by the SAME model (SurGen held out), so no comparison
here is confounded by which model saw what. Training SR386->SR1482 and
SR1482->SR386 models would add two more confounds (different training sets) to
answer a question this shared-model contrast already answers more cleanly.

STAGE IS A MISSINGNESS-AWARE SENSITIVITY, NOT A COMPLETE-CASE FILTER.  The
development manifest predates the filled-stage derivation and still labels all
SR1482 patients as stage-unknown.  This analysis therefore joins the current
``stage_group_major_filled`` and ``stage_source`` fields from the authoritative
label source, requires patient-constant primary-tumour rows, records the source
hash and coverage, and adds stage as a categorical covariate with an
explicit ``unknown`` level.  It never drops a patient merely because stage is
missing.

Usage:
    python aim2_surgen_subcohort_gap.py --cap 8192
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim2_loco_transport  # noqa: E402
from oceanpath.aim1 import lineage, paths  # noqa: E402

N_BOOT = 10_000
PROPENSITY_CLIP = (0.02, 0.98)
STAGE_COLUMN = "stage_group_major_filled"
STAGE_SOURCE_COLUMN = "stage_source"
EXPECTED_STAGE_LEVELS = {"I", "II", "III", "IV"}


def load(cap: int) -> pd.DataFrame:
    """SurGen target-primary patients plus current, provenance-ready stage.

    ``aim1_dev.csv`` is intentionally frozen for training identity, so its
    legacy ``stage_class`` cannot be used for this post-training sensitivity.
    Filled stage is joined from the current label source on the primary-tumour
    patient row.  The join fails closed on missing/duplicate patients or a
    subcohort disagreement; legitimate missing stage remains an explicit
    ``unknown`` category downstream.
    """
    ens, seeds = aim2_loco_transport.seed_ensemble("SurGen", "primary", cap)
    if set(seeds) != set(aim2_loco_transport.SEEDS):
        raise RuntimeError(
            f"SurGen: incomplete primary seed ensemble {sorted(seeds)}; "
            f"expected {list(aim2_loco_transport.SEEDS)}"
        )
    man = pd.read_csv(paths.DEV_MANIFEST)
    pat = man.drop_duplicates("patient_id")
    extra = pat[["patient_id", "age_at_diagnosis", "tumor_site_group", "msi_dmmr",
                 "braf", "stage_class", "sex"]]
    spp = man.groupby("patient_id").size().rename("n_slides_true")
    tiles = man.groupby("patient_id")["patch_count"].mean().rename("mean_patch_count")
    area = man.groupby("patient_id")["tissue_area_mm2"].mean().rename("mean_tissue_area")
    merged = (
        ens.merge(
            extra,
            on="patient_id",
            how="left",
            suffixes=("", "_m"),
            validate="one_to_one",
        )
        .merge(spp, on="patient_id", validate="one_to_one")
        .merge(tiles, on="patient_id", validate="one_to_one")
        .merge(area, on="patient_id", validate="one_to_one")
    )
    expected_ids = set(ens["patient_id"].astype(str))
    observed_ids = set(merged["patient_id"].astype(str))
    if len(merged) != len(ens) or observed_ids != expected_ids:
        raise RuntimeError(
            "SurGen predictions contain patients absent from the development manifest: "
            f"{sorted(expected_ids - observed_ids)[:5]}"
        )

    source = pd.read_csv(paths.LABEL_SOURCE, low_memory=False)
    required = {
        "patient_uid",
        "specimen_role",
        "subcohort",
        STAGE_COLUMN,
        STAGE_SOURCE_COLUMN,
    }
    missing_columns = sorted(required.difference(source.columns))
    if missing_columns:
        raise RuntimeError(
            f"Filled-stage source is missing required columns: {missing_columns}"
        )
    patient_ids = set(merged["patient_id"].astype(str))
    stage_rows = source[
        source["specimen_role"].eq("primary")
        & source["patient_uid"].astype(str).isin(patient_ids)
    ].copy()
    # The source is slide-level, so multi-slide patients legitimately have more
    # than one primary row.  All stage fields must nevertheless be patient-
    # constant before those rows are collapsed.
    conflicts: list[str] = []
    for patient_id, block in stage_rows.groupby("patient_uid", sort=False):
        if any(
            block[column].nunique(dropna=False) != 1
            for column in ("subcohort", STAGE_COLUMN, STAGE_SOURCE_COLUMN)
        ):
            conflicts.append(str(patient_id))
    if conflicts:
        raise RuntimeError(
            "Filled-stage source is not patient-constant across primary rows: "
            f"{sorted(conflicts)[:5]}"
        )
    stage_rows = stage_rows.drop_duplicates("patient_uid").copy()
    found = set(stage_rows["patient_uid"].astype(str))
    if found != patient_ids:
        raise RuntimeError(
            "Filled-stage join does not cover the evaluated SurGen population: "
            f"missing={sorted(patient_ids - found)[:5]}, "
            f"unexpected={sorted(found - patient_ids)[:5]}"
        )
    unexpected_levels = set(
        stage_rows[STAGE_COLUMN].dropna().astype(str).unique()
    ).difference(EXPECTED_STAGE_LEVELS)
    if unexpected_levels:
        raise RuntimeError(
            f"Unexpected filled-stage levels: {sorted(unexpected_levels)}"
        )
    stage = stage_rows[
        ["patient_uid", "subcohort", STAGE_COLUMN, STAGE_SOURCE_COLUMN]
    ].rename(
        columns={
            "patient_uid": "patient_id",
            "subcohort": "stage_subcohort",
        }
    )
    stage["patient_id"] = stage["patient_id"].astype(str)
    merged["patient_id"] = merged["patient_id"].astype(str)
    merged = merged.merge(stage, on="patient_id", how="left", validate="one_to_one")
    mismatch = merged["stage_subcohort"].ne(merged["subcohort"])
    if mismatch.any():
        rows = merged.loc[
            mismatch, ["patient_id", "subcohort", "stage_subcohort"]
        ].head().to_dict("records")
        raise RuntimeError(f"Filled-stage subcohort mismatch: {rows}")
    merged["stage_known"] = merged[STAGE_COLUMN].notna()
    return merged


def stage_coverage(df: pd.DataFrame) -> dict[str, Any]:
    """JSON-ready coverage audit for the filled-stage join."""

    out: dict[str, Any] = {
        "field": STAGE_COLUMN,
        "missingness_handling": (
            "categorical unknown level plus all observed I/II/III/IV levels; "
            "no complete-case deletion"
        ),
        "overall": {},
        "by_subcohort": {},
    }

    def summarize(block: pd.DataFrame) -> dict[str, Any]:
        known = block[STAGE_COLUMN].notna()
        return {
            "n": int(len(block)),
            "n_known": int(known.sum()),
            "n_unknown": int((~known).sum()),
            "fraction_known": float(known.mean()),
            "stage_levels": {
                str(key): int(value)
                for key, value in block[STAGE_COLUMN]
                .fillna("unknown")
                .value_counts()
                .sort_index()
                .items()
            },
            "stage_source": {
                str(key): int(value)
                for key, value in block[STAGE_SOURCE_COLUMN]
                .fillna("missing_source_value")
                .value_counts()
                .sort_index()
                .items()
            },
        }

    out["overall"] = summarize(df)
    out["by_subcohort"] = {
        str(subcohort): summarize(block)
        for subcohort, block in df.groupby("subcohort", sort=True)
    }
    return out


def auroc(d: pd.DataFrame) -> float:
    y = d["label"].to_numpy()
    return (
        float(roc_auc_score(y, d["mean_logit"].to_numpy()))
        if len(np.unique(y)) > 1
        else float("nan")
    )


def block(d: pd.DataFrame) -> dict:
    y = d["label"].to_numpy()
    p = np.clip(d["prob_raw"].to_numpy(), 1e-6, 1 - 1e-6)
    out = {"n": int(len(d)), "n_mut": int(y.sum()), "prev": float(y.mean()),
           "auroc": auroc(d), "brier": float(brier_score_loss(y, p)),
           "log_loss": float(log_loss(y, p))}
    if len(np.unique(y)) > 1:
        from oceanpath.eval.core import compute_calibration_intercept_slope
        c = compute_calibration_intercept_slope(y, p)
        out["cal_intercept"] = float(c["calibration_intercept"])
        out["cal_slope"] = float(c["calibration_slope"])
    z = d["mean_logit"].to_numpy()
    out["logit_mut_mean"] = float(z[y == 1].mean())
    out["logit_mut_sd"] = float(z[y == 1].std())
    out["logit_wt_mean"] = float(z[y == 0].mean())
    out["logit_wt_sd"] = float(z[y == 0].std())
    out["separation"] = out["logit_mut_mean"] - out["logit_wt_mean"]
    pooled = np.sqrt((z[y == 1].var() + z[y == 0].var()) / 2)
    out["cohens_d"] = float(out["separation"] / pooled) if pooled else float("nan")
    return out


def indep_delta(
    a: pd.DataFrame,
    b: pd.DataFrame,
    seed: int = paths.BOOTSTRAP_SEED,
    n_boot: int = N_BOOT,
) -> dict:
    """AUROC(a) - AUROC(b), INDEPENDENT bootstrap — the subsets are disjoint."""
    rng = np.random.default_rng(seed)
    d = []
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    for _ in range(n_boot):
        ia = rng.integers(0, len(a), len(a))
        ib = rng.integers(0, len(b), len(b))
        va, vb = auroc(a.iloc[ia]), auroc(b.iloc[ib])
        if np.isfinite(va) and np.isfinite(vb):
            d.append(va - vb)
    return {"delta": auroc(a) - auroc(b),
            "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            # A bootstrap tail fraction is a stability diagnostic, not a formal
            # null p-value.  The percentile CI is the inferential readout.
            "bootstrap_two_tail_fraction": float(
                min(np.mean(np.array(d) <= 0), np.mean(np.array(d) >= 0)) * 2
            ),
            "n_bootstrap": int(len(d)), "bootstrap_seed": int(seed),
            "bootstrap_method": "independent patient resampling of disjoint subcohorts"}


def _design_matrix(df: pd.DataFrame, covars: list[str]) -> tuple[np.ndarray, list[str]]:
    """Fit-time encoding with explicit missingness for every selected covariate."""

    missing = sorted(set(covars).difference(df.columns))
    if missing:
        raise ValueError(f"Standardization covariates are missing: {missing}")
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for column in covars:
        if pd.api.types.is_numeric_dtype(df[column]):
            values = pd.to_numeric(df[column], errors="coerce")
            mean = float(values.mean())
            sd = float(values.std(ddof=0))
            if not np.isfinite(sd) or sd == 0:
                sd = 1.0
            arrays.append(
                ((values - mean) / sd).fillna(0.0).to_numpy(dtype=float)[:, None]
            )
            names.append(column)
            if values.isna().any():
                arrays.append(values.isna().to_numpy(dtype=float)[:, None])
                names.append(f"{column}__missing")
        else:
            # Use deterministic reference coding.  ``unknown`` sorts after the
            # observed stage levels, so missing stage remains an explicit column
            # instead of disappearing into the reference category.
            dummies = pd.get_dummies(
                df[column].astype("string").fillna("unknown"),
                prefix=column,
                drop_first=True,
                dtype=float,
            )
            arrays.append(dummies.to_numpy(dtype=float))
            names.extend(map(str, dummies.columns))
    if not arrays:
        raise ValueError("At least one standardization covariate is required")
    matrix = np.hstack(arrays)
    if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise RuntimeError("Propensity design matrix is empty or non-finite")
    return matrix, names


def _weighted_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Tie-correct weighted Mann-Whitney AUROC."""

    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or weights.shape != labels.shape:
        raise ValueError("labels, scores and weights must be aligned one-dimensional arrays")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("weighted AUROC requires both binary classes")
    if not np.isfinite(scores).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("weighted AUROC requires finite scores and positive finite weights")
    positive = labels == 1
    negative = ~positive
    comparisons = (
        (scores[positive, None] > scores[None, negative]).astype(float)
        + 0.5 * (scores[positive, None] == scores[None, negative])
    )
    pair_weights = weights[positive, None] * weights[None, negative]
    denominator = float(pair_weights.sum())
    if denominator <= 0:
        raise ValueError("weighted AUROC has zero pair weight")
    return float((comparisons * pair_weights).sum() / denominator)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(np.asarray(values, dtype=float), [0, 1, 5, 50, 95, 99, 100])
    return {
        key: float(value)
        for key, value in zip(
            ("min", "p01", "p05", "median", "p95", "p99", "max"),
            q,
            strict=True,
        )
    }


def _overlap_diagnostics(
    is386: np.ndarray,
    raw_propensity: np.ndarray,
    clipped_propensity: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    """Report empirical overlap, clipping and SR1482 ATT-weight behavior."""

    masks = {"SR386": is386 == 1, "SR1482": is386 == 0}
    bounds = {
        name: (float(raw_propensity[mask].min()), float(raw_propensity[mask].max()))
        for name, mask in masks.items()
    }
    common_low = max(value[0] for value in bounds.values())
    common_high = min(value[1] for value in bounds.values())
    if common_low >= common_high:
        raise RuntimeError(
            "Propensity distributions have no empirical common support: "
            f"SR386={bounds['SR386']}, SR1482={bounds['SR1482']}"
        )
    clip_low, clip_high = PROPENSITY_CLIP
    per_group: dict[str, Any] = {}
    for name, mask in masks.items():
        raw = raw_propensity[mask]
        per_group[name] = {
            "n": int(mask.sum()),
            "raw_propensity_quantiles": _quantiles(raw),
            "n_clipped_low": int((raw < clip_low).sum()),
            "n_clipped_high": int((raw > clip_high).sum()),
            "fraction_clipped": float(
                ((raw < clip_low) | (raw > clip_high)).mean()
            ),
            "fraction_outside_empirical_common_support": float(
                ((raw < common_low) | (raw > common_high)).mean()
            ),
        }
    control_weights = weights[masks["SR1482"]]
    ess = float(control_weights.sum() ** 2 / np.square(control_weights).sum())
    return {
        "estimand": "ATT on the SR386 covariate distribution",
        "propensity_clip": [float(clip_low), float(clip_high)],
        "empirical_common_support": [common_low, common_high],
        "propensity_by_subcohort": per_group,
        "SR1482_weight_quantiles": _quantiles(control_weights),
        "SR1482_effective_sample_size": ess,
        "SR1482_ess_fraction": float(ess / masks["SR1482"].sum()),
        "SR1482_max_weight": float(control_weights.max()),
        "SR1482_weight_cv": float(
            control_weights.std(ddof=0) / control_weights.mean()
        ),
        "clipped_propensity_quantiles_all": _quantiles(clipped_propensity),
    }


# These thresholds are interpretation guardrails, not a replacement for looking
# at the full propensity and weight distributions.  They are deliberately
# conservative and are recorded in every result so a heavily extrapolated IPW
# estimate cannot silently be presented as an adjusted cohort comparison.
OVERLAP_MAX_CONTROL_CLIPPED = 0.10
OVERLAP_MAX_CONTROL_OUTSIDE_COMMON_SUPPORT = 0.20
OVERLAP_MIN_CONTROL_ESS_FRACTION = 0.25
OVERLAP_MAX_CONTROL_WEIGHT = 10.0


def _overlap_assessment(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable gate for whether an IPW contrast supports inference.

    The gate was added after the cap-8192 audit exposed severe positivity loss
    in the late SurGen adjustment ladder.  It is therefore a reporting safety
    rule, not a pre-registered hypothesis test.  Failure means the estimate is
    retained as a positivity/extrapolation diagnostic but must not drive a
    claim that adjustment did, or did not, explain the observed gap.
    """

    control = diagnostics["propensity_by_subcohort"]["SR1482"]
    observed = {
        "control_fraction_clipped": float(control["fraction_clipped"]),
        "control_fraction_outside_empirical_common_support": float(
            control["fraction_outside_empirical_common_support"]
        ),
        "control_ess_fraction": float(diagnostics["SR1482_ess_fraction"]),
        "control_max_normalized_weight": float(diagnostics["SR1482_max_weight"]),
    }
    thresholds = {
        "max_control_fraction_clipped": OVERLAP_MAX_CONTROL_CLIPPED,
        "max_control_fraction_outside_empirical_common_support": (
            OVERLAP_MAX_CONTROL_OUTSIDE_COMMON_SUPPORT
        ),
        "min_control_ess_fraction": OVERLAP_MIN_CONTROL_ESS_FRACTION,
        "max_control_normalized_weight": OVERLAP_MAX_CONTROL_WEIGHT,
    }
    checks = {
        "clipping": observed["control_fraction_clipped"]
        <= thresholds["max_control_fraction_clipped"],
        "common_support": observed[
            "control_fraction_outside_empirical_common_support"
        ]
        <= thresholds["max_control_fraction_outside_empirical_common_support"],
        "effective_sample_size": observed["control_ess_fraction"]
        >= thresholds["min_control_ess_fraction"],
        "maximum_weight": observed["control_max_normalized_weight"]
        <= thresholds["max_control_normalized_weight"],
    }
    adequate = all(checks.values())
    return {
        "overlap_adequate": adequate,
        "estimable_for_inference": adequate,
        "status": (
            "diagnostically adequate overlap"
            if adequate
            else "limited overlap; positivity diagnostic only"
        ),
        "observed": observed,
        "thresholds": thresholds,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "provenance": (
            "post-audit reporting guardrail; not a pre-registered hypothesis test"
        ),
        "claim_guardrail": (
            "A non-estimable arm cannot support a claim that adjustment explains, "
            "or fails to explain, the subcohort gap."
        ),
    }


def _validate_standardization_frame(df: pd.DataFrame, covars: list[str]) -> None:
    required = {"subcohort", "patient_id", "label", "mean_logit", *covars}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Standardization frame is missing columns: {missing}")
    observed = set(df["subcohort"].dropna().astype(str))
    if observed != {"SR386", "SR1482"}:
        raise ValueError(f"Expected exactly SR386 and SR1482, got {sorted(observed)}")
    if not np.isfinite(pd.to_numeric(df["mean_logit"], errors="coerce")).all():
        raise ValueError("Standardization frame contains non-finite logits")
    for subcohort in ("SR386", "SR1482"):
        labels = set(df.loc[df["subcohort"].eq(subcohort), "label"].unique())
        if labels != {0, 1}:
            raise ValueError(
                f"{subcohort} must contain both binary outcome classes; got {labels}"
            )


def _fit_standardization_once(
    df: pd.DataFrame,
    covars: list[str],
    *,
    diagnostics: bool,
) -> dict[str, Any]:
    """Fit one propensity model and return the standardized AUROC gap."""

    _validate_standardization_frame(df, covars)
    data = df.copy()
    is386 = data["subcohort"].eq("SR386").to_numpy(dtype=int)
    matrix, feature_names = _design_matrix(data, covars)
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=2_000,
        solver="lbfgs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(matrix, is386)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise RuntimeError("Propensity model produced non-finite coefficients")
    raw_propensity = model.predict_proba(matrix)[:, 1]
    clipped_propensity = np.clip(raw_propensity, *PROPENSITY_CLIP)
    weights = np.ones(len(data), dtype=float)
    control = is386 == 0
    weights[control] = clipped_propensity[control] / (1.0 - clipped_propensity[control])
    # A within-SR1482 constant cannot change weighted AUROC or ESS; normalizing
    # to mean one makes max-weight and quantile diagnostics interpretable.
    weights[control] /= weights[control].mean()
    labels = data["label"].to_numpy(dtype=int)
    scores = data["mean_logit"].to_numpy(dtype=float)
    target_auc = _weighted_auroc(labels[~control], scores[~control], weights[~control])
    control_auc = _weighted_auroc(labels[control], scores[control], weights[control])
    result: dict[str, Any] = {
        "auroc_SR386": target_auc,
        "auroc_SR1482_reweighted": control_auc,
        "delta": float(target_auc - control_auc),
        "covariates": list(covars),
        "propensity_model": {
            "model": "L2 logistic regression",
            "C": 1.0,
            "max_iter": 2_000,
            "design_columns": feature_names,
            "n_iter": int(np.max(model.n_iter_)),
        },
    }
    control_weights = weights[control]
    result["ess_SR1482"] = float(
        control_weights.sum() ** 2 / np.square(control_weights).sum()
    )
    if diagnostics:
        overlap = _overlap_diagnostics(
            is386, raw_propensity, clipped_propensity, weights
        )
        result["diagnostics"] = overlap
        result["overlap_assessment"] = _overlap_assessment(overlap)
    return result


def _bootstrap_standardization_frame(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Outcome-stratified patient resample within each disjoint subcohort."""

    pieces = []
    for subcohort in ("SR386", "SR1482"):
        for label in (0, 1):
            cell = df[df["subcohort"].eq(subcohort) & df["label"].eq(label)]
            if cell.empty:
                raise ValueError(
                    f"Cannot bootstrap empty {subcohort}/label={label} cell"
                )
            pieces.append(cell.iloc[rng.integers(0, len(cell), len(cell))])
    return pd.concat(pieces, ignore_index=True)


def standardized_delta(
    df: pd.DataFrame,
    covars: list[str],
    *,
    seed: int = paths.BOOTSTRAP_SEED,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """IPW-standardized gap with full propensity-refit patient bootstrap.

    SR1482 receives inverse-odds weights targeting SR386's case mix. Every
    bootstrap replicate resamples patients within subcohort and outcome, refits
    the propensity model, recomputes weights, and then recomputes both AUROCs.
    Filled stage is categorical with an explicit unknown level when selected.
    """

    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    point = _fit_standardization_once(df, covars, diagnostics=True)
    rng = np.random.default_rng(seed)
    target_draws = np.empty(n_boot, dtype=float)
    control_draws = np.empty(n_boot, dtype=float)
    delta_draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = _bootstrap_standardization_frame(df, rng)
        try:
            draw = _fit_standardization_once(sampled, covars, diagnostics=False)
        except Exception as exc:  # fail the report; never publish a partial IPW panel
            raise RuntimeError(
                f"IPW propensity refit failed at bootstrap replicate {index}"
            ) from exc
        target_draws[index] = draw["auroc_SR386"]
        control_draws[index] = draw["auroc_SR1482_reweighted"]
        delta_draws[index] = draw["delta"]
    if not (
        np.isfinite(target_draws).all()
        and np.isfinite(control_draws).all()
        and np.isfinite(delta_draws).all()
    ):
        raise RuntimeError("IPW bootstrap produced non-finite estimates")
    point.update(
        {
            "auroc_SR386_ci": [
                float(np.percentile(target_draws, 2.5)),
                float(np.percentile(target_draws, 97.5)),
            ],
            "auroc_SR1482_reweighted_ci": [
                float(np.percentile(control_draws, 2.5)),
                float(np.percentile(control_draws, 97.5)),
            ],
            "delta_ci": [
                float(np.percentile(delta_draws, 2.5)),
                float(np.percentile(delta_draws, 97.5)),
            ],
            "bootstrap_two_tail_fraction": float(
                min(
                    1.0,
                    2
                    * min(
                        np.mean(delta_draws <= 0),
                        np.mean(delta_draws >= 0),
                    ),
                )
            ),
            "n_bootstrap": int(n_boot),
            "bootstrap_seed": int(seed),
            "n_propensity_fits": int(n_boot + 1),
            "bootstrap_method": (
                "outcome-stratified patient resampling within SR386 and SR1482; "
                "propensity model, ATT weights and weighted AUROCs refitted in "
                "every replicate"
            ),
        }
    )
    return point


def standardization_specs() -> tuple[tuple[str, list[str]], ...]:
    """Ordered case-mix sensitivity ladder; stage is the final added axis."""

    base = ["age_at_diagnosis", "tumor_site_group"]
    molecular = [*base, "msi_dmmr", "braf"]
    technical = [
        *molecular,
        "n_slides_true",
        "mean_patch_count",
        "mean_tissue_area",
    ]
    demographics = [*technical, "sex"]
    return (
        ("age + site", base),
        ("+ MSI/BRAF", molecular),
        ("+ slide count & size", technical),
        ("+ sex", demographics),
        ("+ filled stage", [*demographics, STAGE_COLUMN]),
    )


def run_standardizations(
    df: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Run every declared IPW model; any failure aborts before artifact write."""

    return {
        f"ipw_{name}": standardized_delta(
            df,
            covariates,
            seed=seed,
            n_boot=n_boot,
        )
        for name, covariates in standardization_specs()
    }


def material_input_identities(cap: int) -> dict[str, Any]:
    """Direct immutable identities for every file read by the E2d4 population."""

    return {
        "code": lineage.artifact_identity(Path(__file__)),
        "development_manifest": lineage.artifact_identity(paths.DEV_MANIFEST),
        "filled_stage_source": lineage.artifact_identity(paths.LABEL_SOURCE),
        "filled_stage_source_selection": (
            "specimen_role == primary; joined by patient_uid to evaluated patient_id"
        ),
        "surgen_primary_score_receipts": {
            str(seed): lineage.artifact_identity(
                aim2_loco_transport.score_receipt_path("SurGen", seed, "primary", cap)
            )
            for seed in aim2_loco_transport.SEEDS
        },
    }


def cmd_report(a: argparse.Namespace) -> None:
    """Run the report with optionally separate upstream and output eval roots."""

    output_eval_root = Path(
        getattr(a, "output_eval_root", None) or lineage.eval_root()
    )
    input_eval_root = Path(
        getattr(a, "input_eval_root", None) or lineage.eval_root()
    )
    final_dest = output_eval_root / f"e2d4_surgen_gap_cap{a.cap}.json"
    lineage.ensure_absent(final_dest)
    inputs_before = material_input_identities(a.cap)
    df = load(a.cap)
    inputs_after = material_input_identities(a.cap)
    if inputs_after != inputs_before:
        raise RuntimeError("E2d4 material inputs changed while the population was loaded")
    upstream = input_eval_root / f"e2b_metastatic_cap{a.cap}.json"
    r386 = df[df.subcohort.eq("SR386")]
    r1482 = df[df.subcohort.eq("SR1482")]
    coverage = stage_coverage(df)
    zero_coverage = [
        subcohort
        for subcohort, values in coverage["by_subcohort"].items()
        if values["n_known"] == 0
    ]
    if zero_coverage:
        raise RuntimeError(
            "Filled stage has zero known patients in required subcohorts: "
            f"{zero_coverage}"
        )
    upstream_identity = lineage.artifact_identity(upstream)
    rep: dict = {
        "schema_version": 2,
        "cap": a.cap,
        "lineage": lineage.lineage_name(),
        "upstream_e2b": upstream_identity,
        "inputs": {
            **inputs_before,
            "upstream_e2b": upstream_identity,
        },
        "filled_stage_coverage": coverage,
        "inference": {
            "n_bootstrap": int(a.n_bootstrap),
            "bootstrap_seed": int(a.bootstrap_seed),
            "sampling_unit": "patient",
            "ipw": (
                "every patient-bootstrap replicate refits the propensity model, "
                "ATT weights and weighted AUROCs"
            ),
        },
    }

    print(f"\n{'=' * 96}\nSURGEN SUBCOHORT GAP · cap {a.cap} · shared SurGen-held-out model\n{'=' * 96}")
    print("\n  STRUCTURAL DIFFERENCES (why 'subcohort', not 'within-institution')")
    print(f"  {'':22s} {'SR386':>18s} {'SR1482':>18s}")
    for lab, fn in (("n", lambda x: f"{len(x)}"),
                    ("KRAS mutant", lambda x: f"{int(x.label.sum())} ({x.label.mean():.1%})"),
                    ("mean age", lambda x: f"{x.age_at_diagnosis.mean():.1f}"),
                    ("multi-slide patients", lambda x: f"{int((x.n_slides_true > 1).sum())}"),
                    ("BRAF unknown", lambda x: f"{int(~x.braf.isin(['mutant','wild_type']).values.sum() if False else (~x.braf.isin(['mutant','wild_type'])).sum())}"),
                    ("MSI unknown", lambda x: f"{int((~x.msi_dmmr.isin(['MSS/pMMR','MSI/dMMR'])).sum())}"),
                    ("filled stage known", lambda x: f"{int(x[STAGE_COLUMN].notna().sum())}"),
                    ("mean patch count", lambda x: f"{x.mean_patch_count.mean():.0f}"),
                    ("mean tissue area", lambda x: f"{x.mean_tissue_area.mean():.1f}")):
        print(f"  {lab:22s} {fn(r386):>18s} {fn(r1482):>18s}")

    print("\n  PERFORMANCE, SEPARATELY")
    print(f"  {'':22s} {'SR386':>12s} {'SR1482':>12s}")
    b386, b1482 = block(r386), block(r1482)
    rep["SR386"], rep["SR1482"] = b386, b1482
    for k in ("n", "prev", "auroc", "brier", "log_loss", "cal_intercept", "cal_slope",
              "logit_mut_mean", "logit_mut_sd", "logit_wt_mean", "logit_wt_sd",
              "separation", "cohens_d"):
        f = lambda v: (f"{v:.4f}" if isinstance(v, float) else str(v))  # noqa: E731
        print(f"  {k:22s} {f(b386.get(k)):>12s} {f(b1482.get(k)):>12s}")

    d = indep_delta(
        r386, r1482, seed=a.bootstrap_seed, n_boot=a.n_bootstrap
    )
    rep["delta_crude"] = d
    print(f"\n  ΔAUROC (SR386 − SR1482) {d['delta']:+.4f}  95% CI [{d['ci_low']:+.4f}, "
          f"{d['ci_high']:+.4f}]  {'EXCLUDES 0' if d['ci_low'] > 0 or d['ci_high'] < 0 else 'crosses 0'}")

    print("\n  WHERE THE GAP COMES FROM (logit scale)")
    print(f"    mutant  mean  SR386 {b386['logit_mut_mean']:+.3f}  SR1482 {b1482['logit_mut_mean']:+.3f}"
          f"   shift {b1482['logit_mut_mean'] - b386['logit_mut_mean']:+.3f}")
    print(f"    WT      mean  SR386 {b386['logit_wt_mean']:+.3f}  SR1482 {b1482['logit_wt_mean']:+.3f}"
          f"   shift {b1482['logit_wt_mean'] - b386['logit_wt_mean']:+.3f}")
    print(f"    separation    SR386 {b386['separation']:.3f}   SR1482 {b1482['separation']:.3f}"
          f"   loss {b1482['separation'] - b386['separation']:+.3f}")
    print(f"    Cohen's d     SR386 {b386['cohens_d']:.3f}   SR1482 {b1482['cohens_d']:.3f}")

    print("\n  COMPLETE-LABEL MSS/pMMR + BRAF-WT ONLY")
    sub = df[df.msi_dmmr.eq("MSS/pMMR") & df.braf.eq("wild_type")]
    s386, s1482 = sub[sub.subcohort.eq("SR386")], sub[sub.subcohort.eq("SR1482")]
    bd = indep_delta(
        s386, s1482, seed=a.bootstrap_seed, n_boot=a.n_bootstrap
    )
    rep["D_subset"] = {"SR386": block(s386), "SR1482": block(s1482), "delta": bd}
    print(f"    SR386 n={len(s386)} AUROC {auroc(s386):.4f} | SR1482 n={len(s1482)} AUROC {auroc(s1482):.4f}")
    print(f"    ΔAUROC {bd['delta']:+.4f}  95% CI [{bd['ci_low']:+.4f}, {bd['ci_high']:+.4f}]")

    print("\n  SR1482 SINGLE-SLIDE vs MULTI-SLIDE PATIENTS")
    for lab, blk in (("single-slide", r1482[r1482.n_slides_true == 1]),
                     ("multi-slide", r1482[r1482.n_slides_true > 1])):
        bb = block(blk)
        rep[f"SR1482_{lab.replace('-', '_')}"] = bb
        print(f"    {lab:14s} n={bb['n']:4d} prev={bb['prev']:.1%} AUROC {bb['auroc']:.4f} "
              f"sep {bb['separation']:+.3f} Brier {bb['brier']:.4f}")

    print("\n  FILLED-STAGE COVERAGE (no complete-case deletion)")
    for subcohort, values in coverage["by_subcohort"].items():
        print(
            f"    {subcohort:7s} {values['n_known']:3d}/{values['n']:3d} known "
            f"({values['fraction_known']:.1%}); "
            f"unknown={values['n_unknown']}"
        )

    print("\n  IPW-STANDARDIZED GAP (SR1482 reweighted to SR386's case mix)")
    standardizations = run_standardizations(
        df,
        n_boot=a.n_bootstrap,
        seed=a.bootstrap_seed,
    )
    rep.update(standardizations)
    for name, _covariates in standardization_specs():
        st = standardizations[f"ipw_{name}"]
        ci = st["delta_ci"]
        diag = st["diagnostics"]
        clip_fraction = diag["propensity_by_subcohort"]["SR1482"][
            "fraction_clipped"
        ]
        print(
            f"    {name:22s} SR386 {st['auroc_SR386']:.4f}  SR1482* "
            f"{st['auroc_SR1482_reweighted']:.4f}  Δ {st['delta']:+.4f} "
            f"[{ci[0]:+.4f},{ci[1]:+.4f}]  ESS {st['ess_SR1482']:.0f} "
            f"max w {diag['SR1482_max_weight']:.2f} clipped {clip_fraction:.1%}"
        )
    print(
        "\n    Filled stage is included in the final missingness-aware model as "
        "I/II/III/IV/unknown. No patient is dropped for missing stage."
    )
    print(
        "    Every IPW interval includes patient resampling plus a fresh propensity fit; "
        "any failed model aborts this command before the immutable report is written."
    )

    dest = final_dest
    if material_input_identities(a.cap) != inputs_before:
        raise RuntimeError("E2d4 material inputs changed during analysis")
    if lineage.artifact_identity(upstream) != upstream_identity:
        raise RuntimeError("Upstream E2b report changed during E2d4 analysis")
    lineage.write_json_once(dest, rep)
    print(f"\nWrote {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=int, default=8192, choices=aim2_loco_transport.E2A_CAPS)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOT)
    ap.add_argument("--bootstrap-seed", type=int, default=paths.BOOTSTRAP_SEED)
    cmd_report(ap.parse_args())


if __name__ == "__main__":
    main()
