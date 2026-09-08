#!/usr/bin/env python3
"""Append-only Aim-4 score-compressibility analysis for the final-v3 add-ons.

This is one deliberately bounded, post-selection explanatory analysis.  It
asks how much of the frozen E0 three-seed native-logit ensemble can be
reconstructed out of cohort from four already selected numeric prototype
features and the exact limited Aim-1 routine-clinical covariates.  It does not
train another KRAS classifier, alter an existing result, or assign morphology
names to prototype IDs.

Outer evaluation leaves one data source (CPTAC, RIH, SurGen, or TCGA) out.
Within each outer source set, ridge alpha is selected by another source-only
leave-one-cohort-out loop.  Every imputation, scale, category level,
coefficient, and hyperparameter is therefore learned without the outer target.

Example::

    uv run python tools/final_v3_aim4_compressibility.py run \
      --output-root /abs/path/to/final_v3_additions_20260820/aim4_compressibility

    uv run python tools/final_v3_aim4_compressibility.py verify \
      --output-root /abs/path/to/final_v3_additions_20260820/aim4_compressibility
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[1]
DEFAULT_AIM4_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim4_cap8192_corrected_v2_20260820"
)
DEFAULT_PROFILES = DEFAULT_AIM4_ROOT / "profiles/patient_profiles_k32.parquet"
DEFAULT_MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/aim1_dev.csv")
DEFAULT_PREDICTION_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/train/1a_pb_cap8192/univ1"
)
SEEDS = (42, 43, 44)
TARGETS = ("CPTAC", "RIH", "SurGen", "TCGA")
PROTOTYPES = (5, 17, 28)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 20260820

EXPECTED_SHA256 = {
    "patient_profiles": "105d26630c6314c91f1424de45b67c67532db1bf7b503ee72b1054e50288c75a",
    "manifest": "d906ed5b61c5d3bbf56da7ec7bad412287307461012deee5e6d98ae97bd1f3d1",
    "seed42_predictions": "fc61f04175817214614f0526677e1baedaead7e0d21ebab8bd66f55a517c3be8",
    "seed43_predictions": "ce725874d940e0e329dc29d928c68aa2b5be4486689667210105cbd716e0a1f6",
    "seed44_predictions": "653b4c0d69bd75a8824bbeae4e14c23b7e37f476ae3871d232cb54a02db84646",
}

CLINICAL_NUMERIC = ("age_at_diagnosis",)
CLINICAL_CATEGORICAL = ("sex", "site_class", "stage_class")
ABUNDANCE = ("p17_abundance", "p28_abundance")
ATTENTION = ("p28_attention_mass", "p5_attention_mass")
MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "clinical": (*CLINICAL_NUMERIC, *CLINICAL_CATEGORICAL),
    "abundance_only": ABUNDANCE,
    "abundance_plus_attention": (*ABUNDANCE, *ATTENTION),
    "combined": (
        *CLINICAL_NUMERIC,
        *CLINICAL_CATEGORICAL,
        *ABUNDANCE,
        *ATTENTION,
    ),
}
OUTPUT_FILES = (
    "design.md",
    "input_hashes.json",
    "results.json",
    "per_target_metrics.csv",
    "patient_predictions.parquet",
    "model_selection.csv",
)


class CompressibilityError(RuntimeError):
    """A frozen-input, design, analysis, or append-only contract failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompressibilityError(message)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file(), f"expected a file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _safe_float(value: float | int | np.floating[Any]) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _prediction_paths(root: Path) -> dict[int, Path]:
    return {seed: root / f"seed{seed}/oof_predictions.parquet" for seed in SEEDS}


def bind_inputs(
    profiles_path: Path,
    manifest_path: Path,
    prediction_root: Path,
    *,
    enforce_authoritative_hashes: bool = True,
) -> dict[str, dict[str, Any]]:
    inputs = {
        "patient_profiles": _identity(profiles_path),
        "manifest": _identity(manifest_path),
    }
    for seed, path in _prediction_paths(prediction_root).items():
        inputs[f"seed{seed}_predictions"] = _identity(path)
    if enforce_authoritative_hashes:
        for name, expected in EXPECTED_SHA256.items():
            observed = inputs[name]["sha256"]
            _require(
                observed == expected,
                f"{name} is not the authoritative frozen input: {observed} != {expected}",
            )
    return inputs


def _patient_seed_logits(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    required_predictions = {"slide_id", "label", "logit"}
    required_manifest = {"slide_id", "patient_id", "target_label"}
    _require(
        required_predictions <= set(predictions),
        f"prediction columns missing: {sorted(required_predictions - set(predictions))}",
    )
    _require(
        required_manifest <= set(manifest),
        f"manifest columns missing: {sorted(required_manifest - set(manifest))}",
    )
    _require(not predictions["slide_id"].duplicated().any(), "prediction slide IDs are duplicated")
    _require(not manifest["slide_id"].duplicated().any(), "manifest slide IDs are duplicated")
    merged = predictions[["slide_id", "label", "logit"]].merge(
        manifest[["slide_id", "patient_id", "target_label"]],
        on="slide_id",
        how="left",
        validate="one_to_one",
    )
    _require(not merged["patient_id"].isna().any(), "a prediction slide is absent from manifest")
    _require(
        np.array_equal(
            pd.to_numeric(merged["label"], errors="raise").to_numpy(dtype=int),
            pd.to_numeric(merged["target_label"], errors="raise").to_numpy(dtype=int),
        ),
        "prediction and manifest labels differ",
    )
    merged["logit"] = pd.to_numeric(merged["logit"], errors="raise")
    _require(np.isfinite(merged["logit"]).all(), "native logits must all be finite")
    return (
        merged.sort_values("slide_id")
        .groupby("patient_id", as_index=False)
        .agg(label=("label", "first"), patient_seed_logit=("logit", "mean"), n_slides=("slide_id", "size"))
    )


def frozen_ensemble_logits(manifest: pd.DataFrame, prediction_root: Path) -> pd.DataFrame:
    """Equal-slide patient mean within seed, then equal-seed native-logit mean."""
    frames: list[pd.DataFrame] = []
    reference: pd.DataFrame | None = None
    for seed, path in _prediction_paths(prediction_root).items():
        patient = _patient_seed_logits(pd.read_parquet(path), manifest).sort_values("patient_id")
        patient = patient.reset_index(drop=True)
        if reference is None:
            reference = patient[["patient_id", "label", "n_slides"]].copy()
        else:
            _require(
                patient["patient_id"].tolist() == reference["patient_id"].tolist(),
                "OOF seeds cover different patient IDs",
            )
            _require(
                np.array_equal(patient["label"].to_numpy(), reference["label"].to_numpy()),
                "OOF seed labels differ",
            )
            _require(
                np.array_equal(patient["n_slides"].to_numpy(), reference["n_slides"].to_numpy()),
                "OOF seed slide counts differ",
            )
        frames.append(patient[["patient_id", "patient_seed_logit"]].rename(
            columns={"patient_seed_logit": f"logit_seed{seed}"}
        ))
    _require(reference is not None, "no OOF seeds were loaded")
    out = reference
    for frame in frames:
        out = out.merge(frame, on="patient_id", validate="one_to_one")
    seed_columns = [f"logit_seed{seed}" for seed in SEEDS]
    out["native_ensemble_logit"] = out[seed_columns].mean(axis=1)
    return out


def _patient_context(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id",
        "cohort",
        "target_label",
        *CLINICAL_NUMERIC,
        *CLINICAL_CATEGORICAL,
    }
    _require(required <= set(manifest), f"manifest columns missing: {sorted(required - set(manifest))}")
    columns = [
        "patient_id",
        "cohort",
        "target_label",
        *CLINICAL_NUMERIC,
        *CLINICAL_CATEGORICAL,
    ]
    for column in columns[1:]:
        counts = manifest.groupby("patient_id")[column].nunique(dropna=False)
        _require(int(counts.max()) == 1, f"{column} is not patient-constant")
    return manifest.sort_values("slide_id").drop_duplicates("patient_id")[columns]


def _prototype_features(profiles: pd.DataFrame) -> pd.DataFrame:
    required = {
        "arm",
        "patient_id",
        "prototype",
        "cohort",
        "label",
        "abundance",
        "attn_mass_mean",
        "n_attention_seeds",
    }
    _require(required <= set(profiles), f"profile columns missing: {sorted(required - set(profiles))}")
    e0 = profiles.loc[
        (profiles["arm"] == "e0") & (profiles["prototype"].isin(PROTOTYPES))
    ].copy()
    e0["prototype"] = pd.to_numeric(e0["prototype"], errors="raise").astype(int)
    _require(
        not e0.duplicated(["patient_id", "prototype"]).any(),
        "E0 patient/prototype rows are duplicated",
    )
    _require(
        set(e0.groupby("patient_id").size().unique()) == {len(PROTOTYPES)},
        "each E0 patient must have p5, p17, and p28",
    )
    _require(
        set(pd.to_numeric(e0["n_attention_seeds"], errors="raise").unique()) == {len(SEEDS)},
        "canonical attention mass must use all three seeds",
    )
    for column in ("abundance", "attn_mass_mean"):
        e0[column] = pd.to_numeric(e0[column], errors="raise")
        _require(np.isfinite(e0[column]).all(), f"{column} contains non-finite values")
        _require(((e0[column] >= 0) & (e0[column] <= 1)).all(), f"{column} is outside [0,1]")

    abundance = e0.loc[e0["prototype"].isin((17, 28))].pivot(
        index="patient_id", columns="prototype", values="abundance"
    )
    attention = e0.loc[e0["prototype"].isin((5, 28))].pivot(
        index="patient_id", columns="prototype", values="attn_mass_mean"
    )
    abundance.columns = [f"p{int(column)}_abundance" for column in abundance.columns]
    attention.columns = [f"p{int(column)}_attention_mass" for column in attention.columns]
    out = abundance.join(attention, how="inner").reset_index()
    return out[["patient_id", *ABUNDANCE, *ATTENTION]]


def build_analysis_frame(
    profiles_path: Path,
    manifest_path: Path,
    prediction_root: Path,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, low_memory=False)
    logits = frozen_ensemble_logits(manifest, prediction_root)
    context = _patient_context(manifest)
    features = _prototype_features(pd.read_parquet(profiles_path))
    frame = logits.merge(context, on="patient_id", validate="one_to_one").merge(
        features, on="patient_id", validate="one_to_one"
    )
    _require(len(frame) == 1_486, f"expected 1,486 E0 patients, observed {len(frame)}")
    _require(set(frame["cohort"]) == set(TARGETS), "unexpected or missing cohort")
    _require(
        np.array_equal(
            frame["label"].to_numpy(dtype=int), frame["target_label"].to_numpy(dtype=int)
        ),
        "ensemble and manifest patient labels differ",
    )
    _require(frame["patient_id"].is_unique, "analysis patient IDs are duplicated")
    return frame.sort_values(["cohort", "patient_id"]).reset_index(drop=True)


def _feature_types(model_name: str) -> tuple[list[str], list[str]]:
    features = MODEL_FEATURES[model_name]
    categorical = [feature for feature in features if feature in CLINICAL_CATEGORICAL]
    numeric = [feature for feature in features if feature not in categorical]
    return numeric, categorical


def make_ridge_pipeline(model_name: str, alpha: float) -> Pipeline:
    numeric, categorical = _feature_types(model_name)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                drop="first",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers, remainder="drop")),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def _fit_frame(frame: pd.DataFrame, model_name: str, alpha: float) -> Pipeline:
    model = make_ridge_pipeline(model_name, alpha)
    model.fit(frame[list(MODEL_FEATURES[model_name])], frame["native_ensemble_logit"])
    return model


def _predict_frame(model: Pipeline, frame: pd.DataFrame, model_name: str) -> np.ndarray:
    values = np.asarray(model.predict(frame[list(MODEL_FEATURES[model_name])]), dtype=float)
    _require(np.isfinite(values).all(), f"{model_name} produced non-finite predictions")
    return values


def select_alpha_source_only(
    source: pd.DataFrame,
    model_name: str,
    alphas: Sequence[float] = ALPHAS,
) -> tuple[float, list[dict[str, Any]]]:
    """Nested source-only LOCO selection by equal-validation-cohort R2."""
    source_cohorts = sorted(source["cohort"].unique())
    _require(len(source_cohorts) >= 3, "alpha selection requires at least three source cohorts")
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        fold_r2: list[float] = []
        fold_mse: list[float] = []
        for validation_cohort in source_cohorts:
            fit = source[source["cohort"] != validation_cohort]
            validation = source[source["cohort"] == validation_cohort]
            model = _fit_frame(fit, model_name, float(alpha))
            prediction = _predict_frame(model, validation, model_name)
            truth = validation["native_ensemble_logit"].to_numpy(dtype=float)
            fold_r2.append(float(r2_score(truth, prediction)))
            fold_mse.append(float(np.mean((truth - prediction) ** 2)))
        rows.append(
            {
                "alpha": float(alpha),
                "nested_macro_r2": float(np.mean(fold_r2)),
                "nested_macro_mse": float(np.mean(fold_mse)),
                "validation_cohorts": source_cohorts,
                "fold_r2": fold_r2,
                "fold_mse": fold_mse,
            }
        )
    # Maximize macro R2.  Exact ties favor stronger shrinkage deterministically.
    selected = max(rows, key=lambda row: (row["nested_macro_r2"], row["alpha"]))
    return float(selected["alpha"]), rows


def _correlation(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    statistic = stats.spearmanr(x, y).statistic if rank else stats.pearsonr(x, y).statistic
    return float(statistic)


def metric_block(truth: np.ndarray, prediction: np.ndarray, label: np.ndarray) -> dict[str, float]:
    residual = truth - prediction
    return {
        "r2": float(r2_score(truth, prediction)),
        "pearson": _correlation(truth, prediction),
        "spearman": _correlation(truth, prediction, rank=True),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "prediction_kras_auroc": float(roc_auc_score(label, prediction)),
        "residual_kras_auroc": float(roc_auc_score(label, residual)),
        "native_logit_kras_auroc": float(roc_auc_score(label, truth)),
    }


def _bootstrap_index_arrays(
    frame: pd.DataFrame, n_bootstrap: int, seed: int
) -> dict[str, np.ndarray]:
    """Indices local to each target, preserving the original interleaved RNG stream."""
    rng = np.random.default_rng(seed)
    target_sizes = {
        cohort: int((frame["cohort"].to_numpy() == cohort).sum()) for cohort in TARGETS
    }
    arrays = {
        cohort: np.empty((n_bootstrap, size), dtype=np.int64)
        for cohort, size in target_sizes.items()
    }
    for draw in range(n_bootstrap):
        for cohort in TARGETS:
            size = target_sizes[cohort]
            arrays[cohort][draw] = rng.integers(0, size, size)
    return arrays


def _row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(left), np.nan, dtype=float),
        where=denominator > 0,
    )


def _row_auc(label: np.ndarray, score: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(score, method="average", axis=1)
    positive = label == 1
    n_positive = positive.sum(axis=1)
    n_negative = label.shape[1] - n_positive
    numerator = np.sum(ranks * positive, axis=1) - n_positive * (n_positive + 1) / 2
    denominator = n_positive * n_negative
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(label), np.nan, dtype=float),
        where=denominator > 0,
    )


def _metric_arrays(
    truth: np.ndarray, prediction: np.ndarray, label: np.ndarray
) -> dict[str, np.ndarray]:
    """Vectorized paired-bootstrap metrics; rows are bootstrap replicates."""
    _require(
        truth.ndim == prediction.ndim == label.ndim == 2,
        "bootstrap metric arrays must be two-dimensional",
    )
    residual = truth - prediction
    denominator = np.sum((truth - truth.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r2 = np.divide(
        np.sum(residual**2, axis=1),
        denominator,
        out=np.full(len(truth), np.nan, dtype=float),
        where=denominator > 0,
    )
    truth_ranks = stats.rankdata(truth, method="average", axis=1)
    prediction_ranks = stats.rankdata(prediction, method="average", axis=1)
    return {
        "r2": 1.0 - r2,
        "pearson": _row_correlation(truth, prediction),
        "spearman": _row_correlation(truth_ranks, prediction_ranks),
        "rmse": np.sqrt(np.mean(residual**2, axis=1)),
        "prediction_kras_auroc": _row_auc(label, prediction),
        "residual_kras_auroc": _row_auc(label, residual),
        "native_logit_kras_auroc": _row_auc(label, truth),
    }


def _ci(values: Sequence[float]) -> list[float | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) == 0:
        return [None, None]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def summarize_predictions(
    patient_predictions: pd.DataFrame,
    *,
    n_bootstrap: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = ("r2", "pearson", "spearman", "rmse", "prediction_kras_auroc", "residual_kras_auroc", "native_logit_kras_auroc")
    result: dict[str, Any] = {"models": {}}
    flat_rows: list[dict[str, Any]] = []
    index_arrays = _bootstrap_index_arrays(patient_predictions, n_bootstrap, seed)
    target_blocks = {
        cohort: patient_predictions[patient_predictions["cohort"] == cohort].reset_index(drop=True)
        for cohort in TARGETS
    }
    bootstrap_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for model_name in MODEL_FEATURES:
        prediction_column = f"prediction_{model_name}"
        target_points: dict[str, dict[str, float]] = {}
        target_draws: dict[str, dict[str, np.ndarray]] = {}
        pooled_truth: list[np.ndarray] = []
        pooled_prediction: list[np.ndarray] = []
        pooled_label: list[np.ndarray] = []
        for cohort in TARGETS:
            block = target_blocks[cohort]
            truth = block["native_ensemble_logit"].to_numpy(dtype=float)
            prediction = block[prediction_column].to_numpy(dtype=float)
            label = block["label"].to_numpy(dtype=int)
            point = metric_block(
                truth,
                prediction,
                label,
            )
            target_points[cohort] = point
            index = index_arrays[cohort]
            target_draws[cohort] = _metric_arrays(
                truth[index], prediction[index], label[index]
            )
            pooled_truth.append(truth[index])
            pooled_prediction.append(prediction[index])
            pooled_label.append(label[index])

        pooled_draws = _metric_arrays(
            np.concatenate(pooled_truth, axis=1),
            np.concatenate(pooled_prediction, axis=1),
            np.concatenate(pooled_label, axis=1),
        )
        macro_draws = {
            metric: np.nanmean(
                np.stack([target_draws[cohort][metric] for cohort in TARGETS]), axis=0
            )
            for metric in metrics
        }
        bootstrap_metrics[model_name] = target_draws

        targets: dict[str, Any] = {}
        for cohort in TARGETS:
            point = target_points[cohort]
            targets[cohort] = {
                "n": int(len(target_blocks[cohort])),
                "n_mutant": int(target_blocks[cohort]["label"].sum()),
                **{
                    metric: {"point": _safe_float(point[metric]), "ci95": _ci(target_draws[cohort][metric])}
                    for metric in metrics
                },
            }
            for metric in metrics:
                flat_rows.append(
                    {
                        "model": model_name,
                        "summary": cohort,
                        "metric": metric,
                        "point": _safe_float(point[metric]),
                        "ci_low": _ci(target_draws[cohort][metric])[0],
                        "ci_high": _ci(target_draws[cohort][metric])[1],
                    }
                )

        pooled_point = metric_block(
            patient_predictions["native_ensemble_logit"].to_numpy(dtype=float),
            patient_predictions[prediction_column].to_numpy(dtype=float),
            patient_predictions["label"].to_numpy(dtype=int),
        )
        macro_point = {
            metric: float(np.nanmean([target_points[cohort][metric] for cohort in TARGETS]))
            for metric in metrics
        }
        summaries: dict[str, Any] = {}
        for summary_name, point, draws in (
            ("equal_target_macro", macro_point, macro_draws),
            ("pooled_loco", pooled_point, pooled_draws),
        ):
            summaries[summary_name] = {
                metric: {"point": _safe_float(point[metric]), "ci95": _ci(draws[metric])}
                for metric in metrics
            }
            for metric in metrics:
                flat_rows.append(
                    {
                        "model": model_name,
                        "summary": summary_name,
                        "metric": metric,
                        "point": _safe_float(point[metric]),
                        "ci_low": _ci(draws[metric])[0],
                        "ci_high": _ci(draws[metric])[1],
                    }
                )
        result["models"][model_name] = {"targets": targets, "summaries": summaries}

    result["comparisons"] = {}
    for left, right, name in (
        ("abundance_only", "abundance_plus_attention", "attention_increment_over_abundance"),
        ("clinical", "combined", "prototype_increment_over_clinical"),
    ):
        comparison: dict[str, Any] = {}
        for metric in ("r2", "pearson", "spearman"):
            point = (
                result["models"][right]["summaries"]["equal_target_macro"][metric]["point"]
                - result["models"][left]["summaries"]["equal_target_macro"][metric]["point"]
            )
            deltas = np.nanmean(
                np.stack(
                    [
                        bootstrap_metrics[right][cohort][metric]
                        - bootstrap_metrics[left][cohort][metric]
                        for cohort in TARGETS
                    ]
                ),
                axis=0,
            )
            comparison[metric] = {"point": _safe_float(point), "ci95": _ci(deltas)}
        result["comparisons"][name] = comparison
    result["bootstrap"] = {
        "unit": "patient within held-out target",
        "stratification": "four outer targets resampled independently at observed target size",
        "n_replicates": int(n_bootstrap),
        "seed": int(seed),
        "refitting": False,
        "interpretation": "conditional on the frozen profiles, logits, outer fits, and source-only alpha selections",
    }
    return result, pd.DataFrame(flat_rows)


def run_loco(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    predictions = frame[
        [
            "patient_id",
            "cohort",
            "label",
            "native_ensemble_logit",
            *[f"logit_seed{seed}" for seed in SEEDS],
            *ABUNDANCE,
            *ATTENTION,
        ]
    ].copy()
    selections: list[dict[str, Any]] = []
    contracts: dict[str, Any] = {}
    for model_name in MODEL_FEATURES:
        predictions[f"prediction_{model_name}"] = np.nan
        contracts[model_name] = {}
        for target in TARGETS:
            source = frame[frame["cohort"] != target]
            held_out = frame[frame["cohort"] == target]
            alpha, candidates = select_alpha_source_only(source, model_name)
            model = _fit_frame(source, model_name, alpha)
            target_prediction = _predict_frame(model, held_out, model_name)
            predictions.loc[held_out.index, f"prediction_{model_name}"] = target_prediction
            for candidate in candidates:
                selections.append(
                    {
                        "model": model_name,
                        "outer_target": target,
                        "alpha": candidate["alpha"],
                        "nested_macro_r2": candidate["nested_macro_r2"],
                        "nested_macro_mse": candidate["nested_macro_mse"],
                        "selected": candidate["alpha"] == alpha,
                    }
                )
            preprocessor = model.named_steps["preprocess"]
            feature_names = preprocessor.get_feature_names_out().tolist()
            coefficients = model.named_steps["ridge"].coef_.tolist()
            contracts[model_name][target] = {
                "outer_source_n": int(len(source)),
                "outer_target_n": int(len(held_out)),
                "outer_source_cohorts": sorted(source["cohort"].unique().tolist()),
                "selected_alpha": float(alpha),
                "transformed_feature_names": feature_names,
                "coefficients": [float(value) for value in coefficients],
                "intercept": float(model.named_steps["ridge"].intercept_),
                "nested_selection": candidates,
            }
    for model_name in MODEL_FEATURES:
        _require(
            np.isfinite(predictions[f"prediction_{model_name}"]).all(),
            f"{model_name} left one or more patients unscored",
        )
        predictions[f"residual_{model_name}"] = (
            predictions["native_ensemble_logit"] - predictions[f"prediction_{model_name}"]
        )
    return predictions.sort_values(["cohort", "patient_id"]), pd.DataFrame(selections), contracts


def _design_markdown() -> str:
    return """# Aim 4 v3 add-on — cohort-held-out score compressibility

## Question

How much of the frozen E0 three-seed native-logit ensemble can be reconstructed
out of cohort using four already selected numeric prototype-profile features?
This is a post-selection explanatory analysis, not an independently validated
KRAS predictor and not a pathology-identification experiment.

## Frozen inputs and outcome

- Canonical corrected k=32 E0 patient profiles only (`arm=e0`).
- Native E0 OOF slide logits for seeds 42, 43 and 44.
- Outcome: equal-slide patient mean within seed, then equal-seed mean native
  logit. No probability-to-logit reconstruction is allowed.
- Numeric prototype IDs only. No pathology name or historical label transfer.

## Four nested models

1. `clinical`: age at diagnosis, sex, site class, stage class.
2. `abundance_only`: p17 abundance and p28 abundance.
3. `abundance_plus_attention`: p17/p28 abundance plus p28/p5 attention mass.
4. `combined`: exact clinical variables plus all four prototype features.

Attention mass is model-derived and is not itself human-readable morphology.
The abundance-only model is therefore the stricter human-readable bridge,
subject to the still-pending blinded pathology identity.

## Estimation

- Four outer leave-one-data-source-out fits: CPTAC, RIH, SurGen, TCGA.
- Ridge regression predicts the frozen native logit, never the KRAS label.
- Candidate alphas: 0.01, 0.1, 1, 10, 100.
- Alpha is selected separately inside each outer fit by equal-cohort mean R2
  over a nested leave-one-source-cohort-out loop; ties favor stronger
  shrinkage.
- Numeric median imputation, missingness indicators, standardization,
  categorical imputation and category levels are fitted on source patients
  only. Unknown target categories are ignored rather than creating columns.

## Readouts

Primary: held-out R2, Pearson correlation and Spearman correlation for each
target, their equal-target macro mean and the pooled LOCO predictions.
Secondary: RMSE, KRAS AUROC of the reconstructed logit, and KRAS AUROC of the
unexplained residual. Secondary AUROCs do not turn this into a KRAS-trained
model: coefficients are fitted only against the frozen WSI logit.

Two descriptive increments are reported: attention over abundance and all
prototype features over clinical features. Patient bootstraps resample within
each target and do not refit; intervals condition on the frozen models and
selected candidates.

## Guardrails

- Candidate IDs were selected using the full development data before this
  analysis; outer cohort holding applies to the explanatory regression, not
  candidate discovery.
- E0 is patient-OOF but not cohort-held-out: an outer target's other patients
  can have contributed to E0 training folds. This is explanation of frozen E0
  behavior, not external deployment validation.
- No binary success gate is imposed post hoc. Negative held-out R2 is retained.
- No morphology name, clinical threshold, clinical utility, causality or model
  sufficiency claim is authorized.
"""


def _output_inventory(
    root: Path,
    *,
    published_root: Path | None = None,
    exclude: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = exclude or set()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if not path.is_file() or relative in excluded:
            continue
        identity = _identity(path)
        if published_root is not None:
            identity["path"] = str((published_root / relative).resolve(strict=False))
        rows.append({"relative_path": relative, **identity})
    return rows


def _publish(stage: Path, output: Path) -> None:
    _require(not output.exists() and not output.is_symlink(), f"output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(stage, output)
    except OSError:
        # Cross-device-safe append-only fallback.  copytree refuses an existing target.
        shutil.copytree(stage, output, copy_function=shutil.copy2)
        shutil.rmtree(stage)


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output_root).expanduser()
    _require(output.is_absolute(), "output root must be absolute")
    _require(not output.exists() and not output.is_symlink(), f"output exists: {output}")
    profiles = Path(args.profiles).expanduser().resolve(strict=True)
    manifest = Path(args.manifest).expanduser().resolve(strict=True)
    prediction_root = Path(args.prediction_root).expanduser().resolve(strict=True)
    inputs = bind_inputs(profiles, manifest, prediction_root)

    stage_parent = output.parent if output.parent.exists() else output.parent.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=stage_parent))
    try:
        frame = build_analysis_frame(profiles, manifest, prediction_root)
        patient_predictions, selections, contracts = run_loco(frame)
        summary, flat_metrics = summarize_predictions(
            patient_predictions,
            n_bootstrap=int(args.bootstrap_reps),
            seed=int(args.bootstrap_seed),
        )
        summary.update(
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "analysis": "AIM4_POST_SELECTION_COHORT_HELD_OUT_SCORE_COMPRESSIBILITY",
                "created_at_utc": _utc_now(),
                "n_patients": int(len(frame)),
                "targets": list(TARGETS),
                "outcome": "frozen E0 equal-slide-within-seed then equal-seed native ensemble logit",
                "models_ordered": list(MODEL_FEATURES),
                "features": {name: list(features) for name, features in MODEL_FEATURES.items()},
                "ridge_alphas": list(ALPHAS),
                "outer_fit_contracts": contracts,
                "interpretation_guardrails": [
                    "post-selection explanatory analysis; candidate selection used all development cohorts",
                    "E0 is patient-OOF but not cohort-held-out",
                    "attention mass is model-derived and not directly human-readable",
                    "no morphology name or historical label transfer",
                    "no independently validated KRAS predictor, clinical threshold, utility, or causal claim",
                    "negative held-out R2 values are retained without a post-hoc success gate",
                ],
            }
        )
        _write_bytes_once(stage / "design.md", _design_markdown().encode())
        _write_bytes_once(stage / "input_hashes.json", _json_bytes(inputs))
        _write_bytes_once(stage / "results.json", _json_bytes(summary))
        flat_metrics.to_csv(stage / "per_target_metrics.csv", index=False)
        patient_predictions.to_parquet(stage / "patient_predictions.parquet", index=False)
        selections.to_csv(stage / "model_selection.csv", index=False)
        receipt = {
            "schema_version": 1,
            "status": "COMPLETE",
            "append_only": True,
            "created_at_utc": _utc_now(),
            "analysis_code": _identity(Path(__file__)),
            "inputs": inputs,
            "artifacts": _output_inventory(stage, published_root=output),
        }
        _write_bytes_once(stage / "completion_receipt.json", _json_bytes(receipt))
        _publish(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    verify_output(output)
    return output


def verify_output(output_root: Path) -> dict[str, Any]:
    root = output_root.expanduser().resolve(strict=True)
    _require(root.is_dir(), f"output root is not a directory: {root}")
    receipt_path = root / "completion_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(receipt.get("status") == "COMPLETE", "completion status is not COMPLETE")
    expected = {row["relative_path"]: row for row in receipt["artifacts"]}
    _require(set(expected) == set(OUTPUT_FILES), "completion artifact set differs from contract")
    for relative, identity in expected.items():
        observed = _identity(root / relative)
        _require(identity["path"] == str(observed["path"]), f"artifact path mismatch: {relative}")
        _require(observed["sha256"] == identity["sha256"], f"artifact hash mismatch: {relative}")
        _require(observed["size_bytes"] == identity["size_bytes"], f"artifact size mismatch: {relative}")
    inputs = json.loads((root / "input_hashes.json").read_text(encoding="utf-8"))
    for name, identity in inputs.items():
        observed = _identity(Path(identity["path"]))
        _require(observed["sha256"] == identity["sha256"], f"input hash mismatch: {name}")
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    _require(results.get("status") == "COMPLETE", "results status is not COMPLETE")
    predictions = pd.read_parquet(root / "patient_predictions.parquet")
    _require(len(predictions) == 1_486, "patient prediction row count differs")
    _require(predictions["patient_id"].is_unique, "patient predictions contain duplicate IDs")
    _require(set(predictions["cohort"]) == set(TARGETS), "patient prediction targets differ")
    metrics = pd.read_csv(root / "per_target_metrics.csv")
    _require(
        set(metrics["model"]) == set(MODEL_FEATURES),
        "per-target metrics omit one or more models",
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "output_root": str(root),
        "artifacts_verified": len(expected),
        "inputs_rehashed": len(inputs),
        "patients_verified": int(len(predictions)),
    }


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--profiles", default=str(DEFAULT_PROFILES))
    run_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    run_parser.add_argument("--prediction-root", default=str(DEFAULT_PREDICTION_ROOT))
    run_parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    run_parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--output-root", required=True)
    return cli


def main() -> None:
    args = parser().parse_args()
    if args.command == "run":
        path = run(args)
        print(json.dumps({"status": "COMPLETE", "output_root": str(path)}, indent=2))
    else:
        print(json.dumps(verify_output(Path(args.output_root)), indent=2))


if __name__ == "__main__":
    main()
