#!/usr/bin/env python3
"""Final-v3 Aim-2 operational add-on: between-slide sampling agreement.

This program is intentionally downstream-only.  It consumes the frozen E2a
held-out slide logits and the frozen E0 OOF slide logits, writes to a new
append-only report directory, and never trains, calibrates, or rewrites an
existing result.  The primary estimand is the SurGen-held-out E2a ensemble in
SR1482 primary tumours, because 144 of the study's 155 multi-slide patients are
in that subgroup.  E0 is retained only as a cohort-exposed OOF sensitivity.

The term "technical reproducibility" is deliberately avoided: the repeated
slides are different tissue sections/specimens without a replicated scanning
or staining experiment.  The supported estimand is between-slide sampling
agreement under the frozen scoring pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO / "reports" / "reruns" / "final_v3_additions_20260820" / "aim2_operational_v3"
)
DEFAULT_E2A_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim2_cap8192_v4_20260819/e2a"
)
DEFAULT_E0_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/train/1a_pb_cap8192/univ1")
DEFAULT_MANIFEST_ROOT = Path("/mnt/d/YC.Liu/manifests/colon")
DEFAULT_FINAL_V2 = REPO / "reports" / "final_v2"

SEEDS = (42, 43, 44)
CAPACITY_FRACTION = 0.30
BOOTSTRAP_SEED = 20260820
SLIDE_DRAW_SEED = 20260821
TECHNICAL_PERMUTATION_SEED = 20260822


@dataclass(frozen=True)
class AnalysisSpec:
    analysis_id: str
    score_source: str
    frame: pd.DataFrame
    primary: bool
    agreement_inferential: bool
    interpretation: str


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def published_identity(staged_path: Path, published_path: Path) -> dict[str, Any]:
    """Hash a staged file while declaring its final post-rename path."""

    item = identity(staged_path)
    item["path"] = str(published_path.resolve(strict=False))
    return item


def validate_published_receipt(receipt_path: Path) -> dict[str, dict[str, Any]]:
    """Require every declared output to exist and match after publication."""

    receipt = json.loads(receipt_path.read_text())
    checked: dict[str, dict[str, Any]] = {}
    for name, expected in receipt.get("outputs", {}).items():
        declared = Path(expected["path"])
        if not declared.is_absolute():
            raise RuntimeError(f"receipt output path is not absolute: {declared}")
        observed = identity(declared)
        if observed["sha256"] != expected["sha256"]:
            raise RuntimeError(f"receipt output hash mismatch: {declared}")
        if observed["size_bytes"] != expected["size_bytes"]:
            raise RuntimeError(f"receipt output size mismatch: {declared}")
        if declared.name != name:
            raise RuntimeError(f"receipt output key/path mismatch: {name} vs {declared}")
        checked[name] = observed
    if not checked:
        raise RuntimeError(f"receipt declares no outputs: {receipt_path}")
    return checked


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def verify_final_v2(final_v2: Path) -> dict[str, Any]:
    receipt_path = final_v2 / "report_bundle_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "PASS" or not receipt.get("append_only"):
        raise RuntimeError("final_v2 receipt is not an append-only PASS bundle")
    checked: dict[str, Any] = {}
    for name, expected in receipt["documents"].items():
        path = final_v2 / name
        observed = identity(path)
        if observed["sha256"] != expected["sha256"]:
            raise RuntimeError(f"final_v2 immutable-document hash mismatch: {path}")
        checked[name] = observed
    checked[receipt_path.name] = identity(receipt_path)
    return checked


def _check_score_table(frame: pd.DataFrame, *, path: Path, seed: int) -> pd.DataFrame:
    required = {"slide_id", "logit"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: missing {sorted(required - set(frame.columns))}")
    if frame["slide_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate slide_id")
    if not np.isfinite(frame["logit"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: non-finite logit")
    if "seed" in frame and set(frame["seed"].astype(int)) != {seed}:
        raise ValueError(f"{path}: seed column does not equal {seed}")
    return frame[["slide_id", "logit"]].rename(columns={"logit": f"logit_seed{seed}"})


def _ensemble_scores(paths: Sequence[tuple[int, Path]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    merged: pd.DataFrame | None = None
    inputs: list[dict[str, Any]] = []
    reference_slides: set[str] | None = None
    for seed, path in paths:
        table = _check_score_table(pd.read_parquet(path), path=path, seed=seed)
        slides = set(table["slide_id"].astype(str))
        if reference_slides is not None and slides != reference_slides:
            raise ValueError(f"seed score slide sets differ at {path}")
        reference_slides = slides
        merged = table if merged is None else merged.merge(table, on="slide_id", validate="one_to_one")
        inputs.append(identity(path))
    assert merged is not None
    seed_columns = [f"logit_seed{seed}" for seed, _ in paths]
    merged["native_logit"] = merged[seed_columns].mean(axis=1)
    return merged, inputs


def _merge_manifest(manifest: pd.DataFrame, scores: pd.DataFrame, *, source: str) -> pd.DataFrame:
    required = {
        "slide_id",
        "patient_id",
        "target_label",
        "subcohort",
        "patch_count",
        "tissue_area_mm2",
    }
    if not required.issubset(manifest.columns):
        raise ValueError(f"{source}: manifest missing {sorted(required - set(manifest.columns))}")
    if manifest["slide_id"].duplicated().any():
        raise ValueError(f"{source}: manifest has duplicate slides")
    joined = manifest.merge(scores, on="slide_id", how="inner", validate="one_to_one")
    if len(joined) != len(manifest) or len(joined) != len(scores):
        raise ValueError(
            f"{source}: score/manifest mismatch ({len(manifest)} manifest, "
            f"{len(scores)} scores, {len(joined)} joined)"
        )
    per_patient_labels = joined.groupby("patient_id")["target_label"].nunique()
    if int(per_patient_labels.max()) != 1:
        raise ValueError(f"{source}: inconsistent patient labels")
    joined["target_label"] = joined["target_label"].astype(int)
    joined["patch_count"] = pd.to_numeric(joined["patch_count"], errors="coerce")
    joined["tissue_area_mm2"] = pd.to_numeric(joined["tissue_area_mm2"], errors="coerce")
    return joined.sort_values(["patient_id", "slide_id"]).reset_index(drop=True)


def load_e2a_target(
    target: str, e2a_root: Path, manifest_root: Path
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    target_key = target.lower()
    score_paths = [
        (
            seed,
            e2a_root / "scores" / f"pb_cap8192_{target_key}_seed{seed}_primary.parquet",
        )
        for seed in SEEDS
    ]
    scores, inputs = _ensemble_scores(score_paths)
    manifest_path = manifest_root / f"aim1_e2a_{target_key}_primary.csv"
    manifest = pd.read_csv(manifest_path)
    inputs.append(identity(manifest_path))
    return _merge_manifest(manifest, scores, source=f"E2a/{target}"), inputs


def load_e0(e0_root: Path, manifest_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    score_paths = [
        (seed, e0_root / f"seed{seed}" / "oof_predictions.parquet") for seed in SEEDS
    ]
    scores, inputs = _ensemble_scores(score_paths)
    manifest_path = manifest_root / "aim1_dev.csv"
    manifest = pd.read_csv(manifest_path)
    inputs.append(identity(manifest_path))
    return _merge_manifest(manifest, scores, source="E0 OOF"), inputs


def canonical_patient_mean(group: pd.DataFrame) -> float:
    """Mean slides within seed, then mean seeds, matching canonical E2a."""

    seed_columns = [f"logit_seed{seed}" for seed in SEEDS]
    if not set(seed_columns).issubset(group.columns):
        return float(group["native_logit"].mean())
    return float(np.mean([group[column].mean() for column in seed_columns]))


def patient_table(slides: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patient_id, group in slides.groupby("patient_id", sort=True):
        values = group["native_logit"].to_numpy(dtype=float)
        labels = group["target_label"].astype(int).unique()
        if len(labels) != 1:
            raise ValueError(f"inconsistent labels for {patient_id}")
        rows.append(
            {
                "patient_id": patient_id,
                "label": int(labels[0]),
                "subcohort": str(group["subcohort"].iloc[0]),
                "n_slides": int(len(group)),
                "mean_logit": canonical_patient_mean(group),
                "lowest_logit": float(np.min(values)),
                "highest_logit": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def grouped_values(slides: pd.DataFrame) -> list[np.ndarray]:
    return [
        group["native_logit"].to_numpy(dtype=float)
        for _, group in slides.groupby("patient_id", sort=True)
        if len(group) > 1
    ]


def icc_oneway(groups: Sequence[np.ndarray]) -> tuple[float, float, float]:
    """Exchangeable one-way random ICC(1,1) and ICC(1,k_eff).

    The unequal-size effective k follows the standard one-way random-effects
    ANOVA correction: (N - sum(k_i^2)/N)/(n - 1).  ICC(1,k_eff) is the
    Spearman-Brown reliability of the mean of k_eff exchangeable slides.
    """

    if len(groups) < 2:
        return math.nan, math.nan, math.nan
    sizes = np.asarray([len(group) for group in groups], dtype=float)
    if np.any(sizes < 2):
        raise ValueError("ICC groups must all contain at least two slides")
    n_groups = len(groups)
    total_n = int(sizes.sum())
    means = np.asarray([np.mean(group) for group in groups], dtype=float)
    grand = float(np.sum(sizes * means) / total_n)
    ss_between = float(np.sum(sizes * np.square(means - grand)))
    ss_within = float(
        sum(np.square(group - np.mean(group)).sum() for group in groups)
    )
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (total_n - n_groups)
    k_eff = float((total_n - np.square(sizes).sum() / total_n) / (n_groups - 1))
    denominator = ms_between + (k_eff - 1.0) * ms_within
    icc_single = (ms_between - ms_within) / denominator if denominator else math.nan
    average_denominator = 1.0 + (k_eff - 1.0) * icc_single
    icc_average = (
        k_eff * icc_single / average_denominator if average_denominator else math.nan
    )
    return float(icc_single), float(icc_average), k_eff


def symmetric_pair_arrays(groups: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    left: list[float] = []
    right: list[float] = []
    for group in groups:
        for first, second in itertools.combinations(group.tolist(), 2):
            left.extend((float(first), float(second)))
            right.extend((float(second), float(first)))
    return np.asarray(left), np.asarray(right)


def correlation(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float:
    if rank:
        x = rankdata(x, method="average")
        y = rankdata(y, method="average")
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def patient_abs_differences(groups: Sequence[np.ndarray]) -> np.ndarray:
    values: list[float] = []
    for group in groups:
        differences = [abs(a - b) for a, b in itertools.combinations(group.tolist(), 2)]
        values.append(float(np.median(differences)))
    return np.asarray(values, dtype=float)


def agreement_metrics(groups: Sequence[np.ndarray]) -> dict[str, float]:
    single, average, k_eff = icc_oneway(groups)
    left, right = symmetric_pair_arrays(groups)
    differences = patient_abs_differences(groups)
    return {
        "icc_1_1": single,
        "icc_1_k_eff": average,
        "k_eff": k_eff,
        "pearson_pair_symmetric": correlation(left, right),
        "spearman_pair_symmetric": correlation(left, right, rank=True),
        "median_abs_logit_difference": float(np.median(differences)),
        "q25_abs_logit_difference": float(np.quantile(differences, 0.25)),
        "q75_abs_logit_difference": float(np.quantile(differences, 0.75)),
        "q90_abs_logit_difference": float(np.quantile(differences, 0.90)),
        "q95_abs_logit_difference": float(np.quantile(differences, 0.95)),
    }


def percentile_interval(values: Iterable[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return math.nan, math.nan
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_group_metrics(
    groups: Sequence[np.ndarray], *, n_bootstrap: int, seed: int
) -> dict[str, tuple[float, float]]:
    """Patient-clustered bootstrap using vectorized sufficient statistics."""

    rng = np.random.default_rng(seed)
    n_groups = len(groups)
    indices = rng.integers(0, n_groups, size=(n_bootstrap, n_groups))

    sizes = np.asarray([len(group) for group in groups], dtype=float)
    sums = np.asarray([np.sum(group) for group in groups], dtype=float)
    sum_squares = np.asarray([np.square(group).sum() for group in groups], dtype=float)
    means = sums / sizes
    within_ss = sum_squares - np.square(sums) / sizes

    selected_sizes = sizes[indices]
    total_n = selected_sizes.sum(axis=1)
    selected_sums = sums[indices]
    grand = selected_sums.sum(axis=1) / total_n
    ss_between = (
        selected_sizes * np.square(means[indices] - grand[:, np.newaxis])
    ).sum(axis=1)
    ss_within = within_ss[indices].sum(axis=1)
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (total_n - n_groups)
    k_eff = (
        total_n - np.square(selected_sizes).sum(axis=1) / total_n
    ) / (n_groups - 1)
    icc_single = (ms_between - ms_within) / (
        ms_between + (k_eff - 1.0) * ms_within
    )
    icc_average = k_eff * icc_single / (1.0 + (k_eff - 1.0) * icc_single)

    # Symmetric pair Pearson correlation has identical marginal distributions
    # on the two axes, so patient-level sufficient statistics are exact.
    pair_n = sizes * (sizes - 1.0)
    pair_sum = (sizes - 1.0) * sums
    pair_sum_square = (sizes - 1.0) * sum_squares
    pair_cross = np.square(sums) - sum_squares
    total_pairs = pair_n[indices].sum(axis=1)
    total_pair_sum = pair_sum[indices].sum(axis=1)
    covariance_numerator = pair_cross[indices].sum(axis=1) - np.square(
        total_pair_sum
    ) / total_pairs
    variance_numerator = pair_sum_square[indices].sum(axis=1) - np.square(
        total_pair_sum
    ) / total_pairs
    pearson = covariance_numerator / variance_numerator

    differences = patient_abs_differences(groups)
    selected_differences = differences[indices]
    quantiles = np.quantile(
        selected_differences, [0.25, 0.50, 0.75, 0.90, 0.95], axis=1
    )

    # Spearman ranks are not reducible to fixed patient-level moments because
    # resampling changes tie multiplicities. Keep that one statistic exact,
    # while avoiding all other per-draw Python work.
    spearman = np.empty(n_bootstrap, dtype=float)
    all_two = all(len(group) == 2 for group in groups)
    if all_two:
        pairs = np.asarray(groups, dtype=float)
        for draw, sampled_indices in enumerate(indices):
            selected = pairs[sampled_indices]
            left = np.concatenate((selected[:, 0], selected[:, 1]))
            right = np.concatenate((selected[:, 1], selected[:, 0]))
            spearman[draw] = correlation(left, right, rank=True)
    else:
        symmetric = [symmetric_pair_arrays([group]) for group in groups]
        for draw, sampled_indices in enumerate(indices):
            left = np.concatenate([symmetric[index][0] for index in sampled_indices])
            right = np.concatenate([symmetric[index][1] for index in sampled_indices])
            spearman[draw] = correlation(left, right, rank=True)

    draws = {
        "icc_1_1": icc_single,
        "icc_1_k_eff": icc_average,
        "k_eff": k_eff,
        "pearson_pair_symmetric": pearson,
        "spearman_pair_symmetric": spearman,
        "median_abs_logit_difference": quantiles[1],
        "q25_abs_logit_difference": quantiles[0],
        "q75_abs_logit_difference": quantiles[2],
        "q90_abs_logit_difference": quantiles[3],
        "q95_abs_logit_difference": quantiles[4],
    }
    return {key: percentile_interval(values) for key, values in draws.items()}


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return math.nan
    ranks = rankdata(scores, method="average")
    mann_whitney = float(ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2)
    return mann_whitney / (n_positive * n_negative)


def rowwise_auc(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """AUROC for every row of a 2-D score matrix, preserving average ties."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError("rowwise_auc score matrix shape mismatch")
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return np.full(scores.shape[0], np.nan)
    ranks = rankdata(scores, method="average", axis=1)
    mann_whitney = ranks[:, labels == 1].sum(axis=1) - n_positive * (
        n_positive + 1
    ) / 2
    return mann_whitney / (n_positive * n_negative)


def stratified_bootstrap_auc(
    patients: pd.DataFrame,
    score_column: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    labels = patients["label"].to_numpy(dtype=int)
    scores = patients[score_column].to_numpy(dtype=float)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    template_labels = np.concatenate(
        (np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int))
    )
    draws = np.empty(n_bootstrap, dtype=float)
    batch_size = 500
    for start in range(0, n_bootstrap, batch_size):
        stop = min(n_bootstrap, start + batch_size)
        count = stop - start
        indices = np.concatenate(
            (
                rng.choice(positive, size=(count, len(positive)), replace=True),
                rng.choice(negative, size=(count, len(negative)), replace=True),
            ),
            axis=1,
        )
        draws[start:stop] = rowwise_auc(template_labels, scores[indices])
    return percentile_interval(draws)


def top_capacity_membership(ids: np.ndarray, scores: np.ndarray, fraction: float) -> set[str]:
    n_priority = max(1, int(math.floor(fraction * len(ids))))
    order = np.lexsort((ids.astype(str), -scores))
    return set(ids[order[:n_priority]].astype(str))


def random_slide_experiment(
    slides: pd.DataFrame,
    *,
    n_draws: int,
    n_bootstrap: int,
    capacity_fraction: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    groups = [group for _, group in slides.groupby("patient_id", sort=True)]
    ids = np.asarray([str(group["patient_id"].iloc[0]) for group in groups])
    labels = np.asarray([int(group["target_label"].iloc[0]) for group in groups])
    values = [group["native_logit"].to_numpy(dtype=float) for group in groups]
    n_slides = np.asarray([len(value) for value in values], dtype=int)
    score_matrix = np.full((len(values), int(n_slides.max())), np.nan, dtype=float)
    for index, value in enumerate(values):
        score_matrix[index, : len(value)] = value
    mean_scores = np.asarray([canonical_patient_mean(group) for group in groups])
    high_scores = np.asarray([np.max(value) for value in values])
    low_scores = np.asarray([np.min(value) for value in values])
    base_priority = top_capacity_membership(ids, mean_scores, capacity_fraction)
    is_multi = n_slides > 1

    rng = np.random.default_rng(seed)
    n_priority = len(base_priority)
    base_mask = np.isin(ids, list(base_priority))
    id_order = np.argsort(ids.astype(str), kind="stable")
    records: list[pd.DataFrame] = []
    batch_size = 500
    for start in range(0, n_draws, batch_size):
        stop = min(n_draws, start + batch_size)
        count = stop - start
        selected_indices = np.floor(
            rng.random((count, len(values))) * n_slides[np.newaxis, :]
        ).astype(int)
        selected = score_matrix[np.arange(len(values))[np.newaxis, :], selected_indices]
        selected_by_id = selected[:, id_order]
        ranked_by_id = np.argsort(-selected_by_id, axis=1, kind="stable")[:, :n_priority]
        selected_patient_indices = id_order[ranked_by_id]
        selected_masks = np.zeros((count, len(values)), dtype=bool)
        selected_masks[
            np.arange(count)[:, np.newaxis], selected_patient_indices
        ] = True
        changed = selected_masks != base_mask[np.newaxis, :]
        intersection = (selected_masks & base_mask[np.newaxis, :]).sum(axis=1)
        union = (selected_masks | base_mask[np.newaxis, :]).sum(axis=1)
        records.append(
            pd.DataFrame(
                {
                    "draw": np.arange(start, stop),
                    "auroc": rowwise_auc(labels, selected),
                    "priority_flip_fraction_all": changed.mean(axis=1),
                    "priority_flip_fraction_multislide": changed[:, is_multi].mean(axis=1),
                    "priority_retention_fraction": intersection / n_priority,
                    "priority_jaccard": intersection / union,
                }
            )
        )
    draw_table = pd.concat(records, ignore_index=True)

    # Nested patient + slide-choice uncertainty: stratify the patient bootstrap
    # by label, then independently redraw one slide for every sampled patient.
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    nested_rng = np.random.default_rng(seed + 1_000_000)
    nested_auc = np.empty(n_bootstrap, dtype=float)
    template_labels = np.concatenate(
        (np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int))
    )
    for start in range(0, n_bootstrap, batch_size):
        stop = min(n_bootstrap, start + batch_size)
        count = stop - start
        indices = np.concatenate(
            (
                nested_rng.choice(positive, size=(count, len(positive)), replace=True),
                nested_rng.choice(negative, size=(count, len(negative)), replace=True),
            ),
            axis=1,
        )
        selected_indices = np.floor(
            nested_rng.random(indices.shape) * n_slides[indices]
        ).astype(int)
        sampled_scores = score_matrix[indices, selected_indices]
        nested_auc[start:stop] = rowwise_auc(template_labels, sampled_scores)

    patients = patient_table(slides)
    summary = {
        "n_patients": int(len(groups)),
        "n_mutant": int(labels.sum()),
        "n_multislide": int(is_multi.sum()),
        "priority_capacity_fraction": capacity_fraction,
        "n_prioritized": int(len(base_priority)),
        "all_slide_mean": {
            "auroc": auc(labels, mean_scores),
            "patient_bootstrap_95_ci": stratified_bootstrap_auc(
                patients, "mean_logit", n_bootstrap=n_bootstrap, seed=seed + 10
            ),
        },
        "highest_slide_sensitivity": {
            "auroc": auc(labels, high_scores),
            "patient_bootstrap_95_ci": stratified_bootstrap_auc(
                patients, "highest_logit", n_bootstrap=n_bootstrap, seed=seed + 11
            ),
        },
        "lowest_slide_sensitivity": {
            "auroc": auc(labels, low_scores),
            "patient_bootstrap_95_ci": stratified_bootstrap_auc(
                patients, "lowest_logit", n_bootstrap=n_bootstrap, seed=seed + 12
            ),
        },
        "random_one_slide": {
            "mean_auroc_over_draws": float(draw_table["auroc"].mean()),
            "slide_choice_95_interval": percentile_interval(draw_table["auroc"]),
            "nested_patient_and_slide_bootstrap_95_ci": percentile_interval(nested_auc),
        },
        "top_capacity_stability": {
            column: {
                "mean": float(draw_table[column].mean()),
                "slide_choice_95_interval": percentile_interval(draw_table[column]),
            }
            for column in (
                "priority_flip_fraction_all",
                "priority_flip_fraction_multislide",
                "priority_retention_fraction",
                "priority_jaccard",
            )
        },
    }
    return summary, draw_table


def bh_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def spearman_permutation_p(
    x: np.ndarray, y: np.ndarray, *, n_permutations: int, seed: int
) -> tuple[float, float]:
    observed = correlation(x, y, rank=True)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        permuted = correlation(x, rng.permutation(y), rank=True)
        if abs(permuted) >= abs(observed):
            extreme += 1
    return observed, float((extreme + 1) / (n_permutations + 1))


def bootstrap_correlation(
    x: np.ndarray, y: np.ndarray, *, n_bootstrap: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(x), len(x))
        draws.append(correlation(x[indices], y[indices], rank=True))
    return percentile_interval(draws)


def technical_associations(
    slides: pd.DataFrame, *, n_bootstrap: int, n_permutations: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_rows: list[dict[str, Any]] = []
    for patient_id, group in slides.groupby("patient_id", sort=True):
        if len(group) < 2:
            continue
        logits = group["native_logit"].to_numpy(dtype=float)
        patch_count = group["patch_count"].to_numpy(dtype=float)
        tissue_area = group["tissue_area_mm2"].to_numpy(dtype=float)
        if (
            not np.isfinite(patch_count).all()
            or not np.isfinite(tissue_area).all()
            or np.min(patch_count) <= 0
            or np.min(tissue_area) <= 0
        ):
            continue
        patient_rows.append(
            {
                "patient_id": str(patient_id),
                "label": int(group["target_label"].iloc[0]),
                "score_range": float(np.max(logits) - np.min(logits)),
                "patch_count_log2_range": float(np.log2(np.max(patch_count) / np.min(patch_count))),
                "tissue_area_log2_range": float(np.log2(np.max(tissue_area) / np.min(tissue_area))),
            }
        )
    patient_frame = pd.DataFrame(patient_rows)
    records: list[dict[str, Any]] = []
    for offset, predictor in enumerate(("patch_count_log2_range", "tissue_area_log2_range")):
        x = patient_frame[predictor].to_numpy(dtype=float)
        y = patient_frame["score_range"].to_numpy(dtype=float)
        rho, p_value = spearman_permutation_p(
            x, y, n_permutations=n_permutations, seed=seed + offset
        )
        low, high = bootstrap_correlation(
            x, y, n_bootstrap=n_bootstrap, seed=seed + 100 + offset
        )
        records.append(
            {
                "predictor": predictor,
                "outcome": "within_patient_native_logit_range",
                "n_patients": int(len(patient_frame)),
                "spearman_rho": rho,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "two_sided_permutation_p": p_value,
            }
        )
    table = pd.DataFrame(records)
    table["bh_q_across_two_technical_screens"] = bh_adjust(
        table["two_sided_permutation_p"].tolist()
    )
    return table, patient_frame


def agreement_table(
    analysis_id: str,
    slides: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    inferential: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for offset, (label_name, label_value) in enumerate(
        (("all", None), ("KRAS_mutant", 1), ("KRAS_wild_type", 0))
    ):
        subset = slides if label_value is None else slides[slides["target_label"].eq(label_value)]
        groups = grouped_values(subset)
        if len(groups) < 2:
            continue
        point = agreement_metrics(groups)
        ci = bootstrap_group_metrics(groups, n_bootstrap=n_bootstrap, seed=seed + offset)
        n_slides = int(sum(len(group) for group in groups))
        for metric, value in point.items():
            low, high = ci[metric]
            records.append(
                {
                    "analysis_id": analysis_id,
                    "KRAS_stratum": label_name,
                    "inferential_status": "inferential" if inferential else "sensitivity",
                    "n_multislide_patients": int(len(groups)),
                    "n_slides": n_slides,
                    "metric": metric,
                    "estimate": value,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "n_bootstrap": n_bootstrap,
                }
            )
    return pd.DataFrame(records)


def census_table(target_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target, frame in target_frames.items():
        for subcohort, group in frame.groupby("subcohort", sort=True):
            patient_counts = group.groupby("patient_id").size()
            patient_labels = group.groupby("patient_id")["target_label"].first()
            records.append(
                {
                    "held_out_target": target,
                    "subcohort": subcohort,
                    "n_patients": int(len(patient_counts)),
                    "n_mutant": int(patient_labels.sum()),
                    "n_slides": int(len(group)),
                    "n_one_slide_patients": int((patient_counts == 1).sum()),
                    "n_two_slide_patients": int((patient_counts == 2).sum()),
                    "n_three_slide_patients": int((patient_counts == 3).sum()),
                    "n_multislide_patients": int((patient_counts > 1).sum()),
                }
            )
    return pd.DataFrame(records)


def deployment_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "deployment_setting": "Primary tumour, development population",
                "ranking_evidence": (
                    "Modest patient-level gene-status ranking: E0 median-seed AUROC 0.664; "
                    "three-seed ensemble AUROC 0.680 is a reference summary."
                ),
                "probability_scale": (
                    "Cross-fitted calibration is descriptive; no clinical operating threshold is validated."
                ),
                "local_adaptation": "Not required for the retrospective development analysis.",
                "recommended_status": "Retrospective worklist-enrichment or QA evaluation only.",
                "claim_boundary": "Not autonomous diagnosis and not a replacement for molecular testing.",
                "canonical_evidence": "final_v2 Results Aim 1 sections 2.1 and 2.6",
            },
            {
                "deployment_setting": "Primary tumour, held-out cohort/data source",
                "ranking_evidence": (
                    "Ranking transported in all four held-out targets: AUROC 0.687-0.738; "
                    "every 95% lower bound exceeded 0.50."
                ),
                "probability_scale": (
                    "Source-only Platt mapping improved loss/Brier, but does not validate a target threshold."
                ),
                "local_adaptation": "No target labels were used in E2a.",
                "recommended_status": "Candidate for temporal/prospective site-level validation.",
                "claim_boundary": (
                    "Cohort-held-out retrospective transport is supported; prospective workflow utility is not."
                ),
                "canonical_evidence": "final_v2 Results section 3.1",
            },
            {
                "deployment_setting": "Metastatic tissue",
                "ranking_evidence": (
                    "The fixed deployment gate failed: equal-cohort macro-AUROC 0.585 "
                    "[0.490, 0.675]."
                ),
                "probability_scale": (
                    "Recalibration cannot repair a ranking gate failure and no metastatic threshold is validated."
                ),
                "local_adaptation": (
                    "Only the tested frozen-embedding residual linear adapter was evaluated; "
                    "2, 4, or 8 labels/class did not establish improvement."
                ),
                "recommended_status": "Not supported for metastatic deployment.",
                "claim_boundary": (
                    "A general primary-to-metastatic decrement remains unresolved: delta -0.099 "
                    "[-0.208, 0.011]."
                ),
                "canonical_evidence": "final_v2 Results sections 3.2 and 3.3",
            },
            {
                "deployment_setting": "Exact KRAS codon/substitution",
                "ranking_evidence": (
                    "Exact allele ranking is not established; codon and G12D contrasts meet empirical "
                    "ceiling criteria, G12V is inconclusive, and G12C has no ceiling consensus."
                ),
                "probability_scale": "Not applicable; no deployable exact-allele score is supported.",
                "local_adaptation": "Not evaluated adequately for exact allele deployment.",
                "recommended_status": "Direct molecular identification is required.",
                "claim_boundary": (
                    "Gene-level scientific detectability is not exact-allele clinical actionability."
                ),
                "canonical_evidence": "final_v2 Results sections 4.1 and 4.2",
            },
        ]
    )


def design_markdown(n_bootstrap: int, n_draws: int, n_permutations: int) -> str:
    return f"""# Aim 2 operational add-on — frozen design

## Estimand and hierarchy

The primary estimand is **between-slide sampling agreement**, not technical
reproducibility. The inferential population is SR1482 primary tumours scored by
the frozen SurGen-held-out E2a models. The three native logits (seeds 42, 43,
44) are averaged per slide before any patient aggregation. No model is trained,
adapted, calibrated, or selected in this add-on.

E0 OOF scores are a sensitivity only because their training folds contain
other patients from the target cohort. CPTAC, RIH, and TCGA have only 4, 2,
and 5 multi-slide patients, respectively, and therefore contribute to the
census but not standalone inferential agreement claims.

## Locked endpoints

1. Exchangeable absolute-agreement one-way random ICC(1,1), plus reliability
   of the k-effective mean ICC(1,k_eff). Unequal slide counts use the standard
   one-way ANOVA effective-k correction.
2. Pearson and Spearman correlation over symmetrized unordered within-patient
   slide pairs. Symmetrization prevents lexicographic slide order from defining
   the correlation orientation; patient bootstrapping handles pair dependence.
3. Per-patient median absolute native-logit difference, with IQR, 90th and 95th
   percentiles.
4. AUROC using the mean of all slides, one randomly chosen slide, the highest
   slide, and the lowest slide. Highest/lowest are sensitivity envelopes, not
   recommended selection policies.
5. Stability of membership in a fixed-capacity top-{CAPACITY_FRACTION:.0%}
   priority tier: whole-population and multi-slide flip fractions, retention,
   and Jaccard agreement.
6. Two exploratory technical screens: Spearman association of within-patient
   score range with within-patient log2 range in patch count or tissue area.
   The two permutation p-values receive Benjamini-Hochberg correction.

## Uncertainty and deterministic choices

- {n_bootstrap:,} patient-clustered percentile bootstrap replicates; AUROC
  bootstrap is stratified by KRAS label.
- {n_draws:,} deterministic random-one-slide draws. Their percentile range is
  slide-choice variability, not a patient-sampling confidence interval.
- A separate nested bootstrap jointly resamples patients and selects one slide.
- {n_permutations:,} deterministic permutations for each technical screen.
- Capacity is `max(1, floor(0.30 * n_patients))` and native-logit ties are broken by
  patient ID only for deterministic queue membership.
- All-slide patient aggregation first averages slides within each seed and then
  averages seeds, exactly matching the canonical E2a report. Single-slide
  selection first averages the three seed logits for that slide.

## Interpretation contract

Population-level AUROC stability does not prove that one slide is sufficient
for every patient. Conversely, individual logit differences do not prove that
multi-slide aggregation improves discrimination. Any operational statement
must jointly consider AUROC, agreement, and priority-tier membership.
"""


def _fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def results_markdown(
    census: pd.DataFrame,
    agreement: pd.DataFrame,
    operational: dict[str, Any],
    technical: pd.DataFrame,
) -> str:
    primary = agreement[
        agreement["analysis_id"].eq("e2a_surgen_heldout__sr1482_primary")
        & agreement["KRAS_stratum"].eq("all")
    ].set_index("metric")

    def metric(name: str) -> tuple[float, float, float]:
        row = primary.loc[name]
        return (
            float(row["estimate"]),
            float(row["bootstrap_ci_low"]),
            float(row["bootstrap_ci_high"]),
        )

    icc = metric("icc_1_1")
    icc_mean = metric("icc_1_k_eff")
    pearson = metric("pearson_pair_symmetric")
    spearman = metric("spearman_pair_symmetric")
    mad = metric("median_abs_logit_difference")
    q25 = float(primary.loc["q25_abs_logit_difference", "estimate"])
    q75 = float(primary.loc["q75_abs_logit_difference", "estimate"])
    q90 = float(primary.loc["q90_abs_logit_difference", "estimate"])
    q95 = float(primary.loc["q95_abs_logit_difference", "estimate"])
    op = operational["e2a_surgen_heldout__sr1482_primary"]

    census_lines = [
        "| Held-out target | Subcohort | Patients | Slides | Multi-slide patients | 2-slide | 3-slide |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in census.itertuples(index=False):
        census_lines.append(
            f"| {row.held_out_target} | {row.subcohort} | {row.n_patients} | {row.n_slides} | "
            f"{row.n_multislide_patients} | {row.n_two_slide_patients} | {row.n_three_slide_patients} |"
        )

    perf_lines = [
        "| Score rule | AUROC | 95% interval | Interval meaning |",
        "| --- | ---: | --- | --- |",
    ]
    for label, key in (
        ("Mean of all slides", "all_slide_mean"),
        ("Highest slide", "highest_slide_sensitivity"),
        ("Lowest slide", "lowest_slide_sensitivity"),
    ):
        block = op[key]
        low, high = block["patient_bootstrap_95_ci"]
        perf_lines.append(
            f"| {label} | {_fmt(block['auroc'])} | [{_fmt(low)}, {_fmt(high)}] | patient bootstrap |"
        )
    random = op["random_one_slide"]
    low, high = random["slide_choice_95_interval"]
    nested_low, nested_high = random["nested_patient_and_slide_bootstrap_95_ci"]
    perf_lines.append(
        f"| One randomly selected slide | {_fmt(random['mean_auroc_over_draws'])} | "
        f"[{_fmt(low)}, {_fmt(high)}] | slide-choice interval; nested patient+slide CI "
        f"[{_fmt(nested_low)}, {_fmt(nested_high)}] |"
    )

    tier = op["top_capacity_stability"]
    technical_lines = [
        "| Technical disparity | Spearman rho [bootstrap 95% CI] | permutation p | BH q |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in technical.itertuples(index=False):
        technical_lines.append(
            f"| {row.predictor} | {_fmt(row.spearman_rho)} "
            f"[{_fmt(row.bootstrap_ci_low)}, {_fmt(row.bootstrap_ci_high)}] | "
            f"{row.two_sided_permutation_p:.4g} | {row.bh_q_across_two_technical_screens:.4g} |"
        )

    return f"""# Aim 2 operational add-on — results

## Population accounting

{os.linesep.join(census_lines)}

Across the six primary subcohorts, 1,331 patients have one slide, 154 have two,
and one TCGA-COAD patient has three: **155 multi-slide patients total**. SR1482
contains 144/155 (92.9%), so the held-out SR1482 analysis is the only
well-powered cohort-specific agreement analysis.

## Primary held-out analysis: SR1482 primary

The analysis contains {op['n_patients']} patients ({op['n_mutant']} mutant),
including {op['n_multislide']} with multiple slides. Between-slide ICC(1,1) is
{_fmt(icc[0])} [{_fmt(icc[1])}, {_fmt(icc[2])}], while ICC(1,k_eff) for the
average of the available slides is {_fmt(icc_mean[0])}
[{_fmt(icc_mean[1])}, {_fmt(icc_mean[2])}]. Symmetrized pair correlations are
Pearson {_fmt(pearson[0])} [{_fmt(pearson[1])}, {_fmt(pearson[2])}] and
Spearman {_fmt(spearman[0])} [{_fmt(spearman[1])}, {_fmt(spearman[2])}].

The median within-patient absolute native-logit difference is {_fmt(mad[0])}
[{_fmt(mad[1])}, {_fmt(mad[2])}]. Its empirical IQR is {_fmt(q25)}-{_fmt(q75)},
with 90th and 95th percentiles {_fmt(q90)} and {_fmt(q95)}.

### Patient-level discrimination under slide selection

{os.linesep.join(perf_lines)}

At the fixed nominal top-30% capacity ({op['n_prioritized']}/{op['n_patients']}
patients; realized {op['n_prioritized'] / op['n_patients']:.1%}), selecting one
random slide changes queue membership for a mean
{_fmt(tier['priority_flip_fraction_all']['mean'])} of all
patients and {_fmt(tier['priority_flip_fraction_multislide']['mean'])} of
multi-slide patients. Mean retention of the all-slide priority set is
{_fmt(tier['priority_retention_fraction']['mean'])}; mean Jaccard agreement is
{_fmt(tier['priority_jaccard']['mean'])}.

## Tissue-amount screens in the 144 multi-slide SR1482 patients

{os.linesep.join(technical_lines)}

These are exploratory associations with disagreement magnitude. They neither
identify a causal tissue-amount effect nor turn the repeated sections into a
technical-replicate experiment.

## Interpretation

The frozen held-out model shows moderate between-slide score agreement. The
population AUROC from one random slide and the all-slide mean should be read
together with priority-tier flips: near-identical aggregate discrimination can
coexist with clinically relevant reordering of individual patients near a
fixed worklist boundary. Highest- and lowest-slide results are sensitivity
bounds, not recommended cherry-picking policies.
"""


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing v3 add-on: {output}")
    final_v2_checks = verify_final_v2(args.final_v2.resolve())

    target_frames: dict[str, pd.DataFrame] = {}
    input_identities: list[dict[str, Any]] = []
    for target in ("CPTAC", "RIH", "SurGen", "TCGA"):
        frame, inputs = load_e2a_target(target, args.e2a_root.resolve(), args.manifest_root.resolve())
        target_frames[target] = frame
        input_identities.extend(inputs)
    e0, e0_inputs = load_e0(args.e0_root.resolve(), args.manifest_root.resolve())
    input_identities.extend(e0_inputs)

    census = census_table(target_frames)
    totals = census[
        [
            "n_patients",
            "n_slides",
            "n_one_slide_patients",
            "n_two_slide_patients",
            "n_three_slide_patients",
            "n_multislide_patients",
        ]
    ].sum()
    expected = {
        "n_patients": 1486,
        "n_slides": 1642,
        "n_one_slide_patients": 1331,
        "n_two_slide_patients": 154,
        "n_three_slide_patients": 1,
        "n_multislide_patients": 155,
    }
    observed = {key: int(totals[key]) for key in expected}
    if observed != expected:
        raise RuntimeError(f"multi-slide census changed: expected {expected}, observed {observed}")

    surgen = target_frames["SurGen"]
    sr1482 = surgen[surgen["subcohort"].eq("SR1482")].copy()
    e0_sr1482 = e0[e0["subcohort"].eq("SR1482")].copy()
    specs = [
        AnalysisSpec(
            "e2a_surgen_heldout__sr1482_primary",
            "E2a frozen SurGen-held-out three-seed native-logit ensemble",
            sr1482,
            True,
            True,
            "Primary inferential analysis; no target-cohort exposure.",
        ),
        AnalysisSpec(
            "e2a_surgen_heldout__all_surgen_primary",
            "E2a frozen SurGen-held-out three-seed native-logit ensemble",
            surgen,
            False,
            False,
            "Operational population sensitivity; all repeated-slide patients are SR1482.",
        ),
        AnalysisSpec(
            "e0_oof__sr1482_primary",
            "E0 three-seed patient-OOF native-logit ensemble",
            e0_sr1482,
            False,
            False,
            "Cohort-exposed OOF sensitivity; not external transport.",
        ),
        AnalysisSpec(
            "e0_oof__all_primary",
            "E0 three-seed patient-OOF native-logit ensemble",
            e0,
            False,
            False,
            "Whole-development OOF sensitivity covering all 155 multi-slide patients.",
        ),
    ]

    agreement_frames: list[pd.DataFrame] = []
    operational: dict[str, Any] = {}
    random_draw_frames: list[pd.DataFrame] = []
    for index, spec in enumerate(specs):
        agreement_frames.append(
            agreement_table(
                spec.analysis_id,
                spec.frame,
                n_bootstrap=args.n_bootstrap,
                seed=BOOTSTRAP_SEED + 1000 * index,
                inferential=spec.agreement_inferential,
            )
        )
        op_summary, draws = random_slide_experiment(
            spec.frame,
            n_draws=args.n_draws,
            n_bootstrap=args.n_bootstrap,
            capacity_fraction=CAPACITY_FRACTION,
            seed=SLIDE_DRAW_SEED + 1000 * index,
        )
        op_summary["score_source"] = spec.score_source
        op_summary["interpretation"] = spec.interpretation
        op_summary["primary"] = spec.primary
        operational[spec.analysis_id] = op_summary
        draws.insert(0, "analysis_id", spec.analysis_id)
        random_draw_frames.append(draws)

    agreement = pd.concat(agreement_frames, ignore_index=True)
    random_draws = pd.concat(random_draw_frames, ignore_index=True)
    technical, technical_patients = technical_associations(
        sr1482,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=TECHNICAL_PERMUTATION_SEED,
    )
    deploy = deployment_matrix()

    # Write into a sibling staging directory, then atomically rename the
    # complete add-on. This cannot touch final_v2 or any model output.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        census.to_csv(staging / "multi_slide_census.csv", index=False)
        agreement.to_csv(staging / "agreement_metrics.csv", index=False)
        random_draws.to_parquet(staging / "random_slide_draws.parquet", index=False)
        technical.to_csv(staging / "technical_associations.csv", index=False)
        technical_patients.to_parquet(staging / "technical_patient_metrics.parquet", index=False)
        deploy.to_csv(staging / "deployment_matrix.csv", index=False)
        (staging / "design.md").write_text(
            design_markdown(args.n_bootstrap, args.n_draws, args.n_permutations)
        )

        results = {
            "schema_version": 1,
            "status": "PASS",
            "analysis": "Aim 2 between-slide sampling agreement and deployment matrix",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "population_census": {
                "totals": observed,
                "by_subcohort": census.to_dict(orient="records"),
            },
            "primary_analysis_id": "e2a_surgen_heldout__sr1482_primary",
            "agreement_metrics": agreement.to_dict(orient="records"),
            "operational_slide_selection": operational,
            "technical_associations": technical.to_dict(orient="records"),
            "deployment_matrix": deploy.to_dict(orient="records"),
            "interpretation_contract": {
                "estimand": "between-slide sampling agreement",
                "forbidden_overclaim": "technical reproducibility",
                "primary_score": "frozen SurGen-held-out E2a native-logit ensemble",
                "e0_role": "cohort-exposed patient-OOF sensitivity only",
                "threshold_status": "no validated clinical threshold",
            },
        }
        write_json(staging / "results.json", results)
        (staging / "Results.md").write_text(
            results_markdown(census, agreement, operational, technical)
        )

        source_files = {
            str(Path(item["path"]).resolve()): item for item in input_identities
        }
        source_files[str((args.final_v2 / "report_bundle_receipt.json").resolve())] = identity(
            args.final_v2 / "report_bundle_receipt.json"
        )
        script_path = Path(__file__).resolve()
        receipt = {
            "schema_version": 1,
            "status": "PASS",
            "append_only": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_code": identity(script_path),
            "python": sys.version,
            "platform": platform.platform(),
            "parameters": {
                "n_bootstrap": args.n_bootstrap,
                "n_random_slide_draws": args.n_draws,
                "n_technical_permutations": args.n_permutations,
                "capacity_fraction": CAPACITY_FRACTION,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "slide_draw_seed": SLIDE_DRAW_SEED,
                "technical_permutation_seed": TECHNICAL_PERMUTATION_SEED,
            },
            "final_v2_preflight_hash_check": final_v2_checks,
            "immutable_inputs": list(source_files.values()),
            "outputs": {
                path.name: published_identity(path, output / path.name)
                for path in sorted(staging.iterdir())
                if path.name != "receipt.json"
            },
            "immutability_note": (
                "The receipt is written last and does not hash itself. Re-running requires a new output path."
            ),
        }
        superseded = output.parent / "aim2_operational_v2" / "receipt.json"
        if superseded.is_file():
            receipt["supersedes"] = {
                "artifact": identity(superseded),
                "reason": (
                    "The v2 receipt retained deleted staging-directory paths after atomic publication. "
                    "This append-only successor records final published paths. Statistical code, "
                    "seeds, repetition counts, estimands, and numeric results are unchanged."
                ),
            }
        write_json(staging / "receipt.json", receipt)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    validate_published_receipt(output / "receipt.json")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--e2a-root", type=Path, default=DEFAULT_E2A_ROOT)
    parser.add_argument("--e0-root", type=Path, default=DEFAULT_E0_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--final-v2", type=Path, default=DEFAULT_FINAL_V2)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--n-draws", type=int, default=10_000)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("n_bootstrap", "n_draws", "n_permutations"):
        if getattr(args, name) < 100:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 100")
    print(run(args))


if __name__ == "__main__":
    main()
