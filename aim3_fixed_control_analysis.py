#!/usr/bin/env python3
"""Append-only, lineage-checked Aim 3 molecular-resolution analysis.

This component consumes the already-trained cap=8192 E3 ladder and E0 gene
reference.  It never trains a model and never changes an upstream artifact.
The correction is statistical and provenance-focused:

* slide-native logits are averaged within patient, then across exactly seeds
  42/43/44; probabilities are never inverted to reconstruct logits;
* every E3 fine/control contrast uses a cohort-stratified, partially paired
  bootstrap: the shared positive patients are drawn jointly and the disjoint
  negative arms independently;
* each task interval is cohort x label stratified;
* a five-rung familywise sensitivity uses one-sided 99% bounds (Bonferroni
  alpha=0.05/5) for each rung's intersection-union ceiling gate;
* a subcohort-standardized sensitivity asks whether the fixed control draw's
  within-cohort composition changes the conclusion.

Inference remains conditional on each pre-existing, fixed control draw.  The
analysis does not regenerate wild-type controls and does not average over
alternative draw seeds.

Commands::

    python aim3_fixed_control_analysis.py verify-inputs \
      --e3-root /abs/path/to/e3a --e0-root /abs/path/to/e0/cap8192

    python aim3_fixed_control_analysis.py run \
      --e3-root /abs/path/to/e3a --e0-root /abs/path/to/e0/cap8192 \
      --output-root /abs/path/to/new/aim3_corrected --n-bootstrap 10000

    python aim3_fixed_control_analysis.py verify-output \
      --e3-root /abs/path/to/e3a --e0-root /abs/path/to/e0/cap8192 \
      --output-root /abs/path/to/new/aim3_corrected
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SEEDS: tuple[int, ...] = (42, 43, 44)
N_FOLDS = 5
CAP = 8192
BOOTSTRAP_SEED = 20260817
DEFAULT_N_BOOTSTRAP = 10_000
CHANCE = 0.50
CEILING_BOUND = 0.60
FAMILYWISE_ALPHA = 0.05

FINE_TASKS: tuple[str, ...] = (
    "codon",
    "g12d_broad",
    "allele1",
    "allele2",
    "g12c",
)
PAIRS: tuple[tuple[str, str], ...] = tuple((task, f"ctrl_{task}") for task in FINE_TASKS)
E3_TASKS: tuple[str, ...] = tuple(item for pair in PAIRS for item in pair)
ALL_TASKS: tuple[str, ...] = ("gene", *E3_TASKS)
CONTROL_DRAW_SEEDS: dict[str, int] = {
    "ctrl_codon": 20260818,
    "ctrl_allele1": 20260819,
    "ctrl_allele2": 20260820,
    "ctrl_g12c": 20260821,
    "ctrl_g12d_broad": 20260822,
}
LEGACY_RENAMES: dict[str, str] = {
    "allele1": "allele",
    "ctrl_allele1": "ctrl_allele",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected file artifact: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _require_absolute_directory(path: Path, *, label: str, must_exist: bool) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an explicit absolute path: {path}")
    resolved = path.resolve(strict=must_exist)
    if must_exist and not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _write_bytes_once_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_once_atomic(path: Path, value: Any) -> None:
    _write_bytes_once_atomic(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n").encode(),
    )


def _relative_artifact(root: Path, evidence: dict[str, Any], *, label: str) -> Path:
    if not isinstance(evidence, dict) or set(evidence) != {"path", "size_bytes", "sha256"}:
        raise RuntimeError(f"{label}: malformed artifact evidence")
    relative = Path(str(evidence["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{label}: unsafe relative artifact path: {relative}")
    resolved_root = root.resolve(strict=True)
    resolved = (resolved_root / relative).resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(f"{label}: artifact escapes its run root: {resolved}") from error
    actual = _artifact_identity(resolved)
    if int(evidence["size_bytes"]) != actual["size_bytes"]:
        raise RuntimeError(f"{label}: size mismatch: {resolved}")
    if str(evidence["sha256"]) != actual["sha256"]:
        raise RuntimeError(f"{label}: SHA256 mismatch: {resolved}")
    return resolved


def _stable_seed(base: int, *parts: object) -> int:
    token = "\0".join([str(base), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (2**32)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise ValueError("AUROC inputs must be aligned one-dimensional arrays")
    if not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("AUROC inputs contain invalid labels or non-finite scores")
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both classes")
    ranks = rankdata(s, method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _interval(values: np.ndarray, low: float = 0.025, high: float = 0.975) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise RuntimeError("Bootstrap distribution is empty or non-finite")
    return [float(np.quantile(array, low)), float(np.quantile(array, high))]


def _stratum_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    counts = frame.groupby(columns, dropna=False, observed=True).size()
    return {"|".join(map(str, key if isinstance(key, tuple) else (key,))): int(value)
            for key, value in counts.items()}


def _assert_patient_constant(manifest: pd.DataFrame, columns: list[str], *, label: str) -> None:
    for column in columns:
        varying = manifest.groupby("patient_id", dropna=False)[column].nunique(dropna=False)
        if (varying != 1).any():
            bad = varying[varying != 1].index.astype(str).tolist()[:5]
            raise RuntimeError(f"{label}: {column} varies within patient: {bad}")


def _manifest_path(
    identity: dict[str, Any], *, expected_task: str, recorded_sha: str
) -> tuple[Path, dict[str, Any]]:
    material = identity["payload"]["material_config"]
    recorded = Path(str(material["data"]["csv_path"]))
    candidates = [recorded]
    legacy = LEGACY_RENAMES.get(expected_task)
    if legacy is not None and recorded.name == f"aim1_e3a_{legacy}.csv":
        candidates.append(recorded.with_name(f"aim1_e3a_{expected_task}.csv"))
    matching = [path.resolve(strict=True) for path in candidates if path.is_file()
                and _sha256_file(path) == recorded_sha]
    if len(matching) != 1:
        raise RuntimeError(
            f"{expected_task}: could not resolve exactly one manifest with recorded SHA; "
            f"recorded={recorded}, candidates={candidates}"
        )
    resolved = matching[0]
    return resolved, {
        "recorded_path": str(recorded),
        "resolved_path": str(resolved),
        "relocated_after_documented_rename": resolved != recorded,
    }


def _validate_prediction_frame(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    required = {"slide_id", "label", "logit", "fold"}
    missing = required - set(predictions)
    if missing:
        raise RuntimeError(f"{label}: prediction columns missing: {sorted(missing)}")
    if predictions["slide_id"].isna().any() or predictions["slide_id"].duplicated().any():
        raise RuntimeError(f"{label}: slide_id is null or duplicated")
    logits = pd.to_numeric(predictions["logit"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(logits).all():
        raise RuntimeError(f"{label}: native logits are not all finite")
    labels = pd.to_numeric(predictions["label"], errors="coerce").to_numpy(dtype=float)
    folds = pd.to_numeric(predictions["fold"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise RuntimeError(f"{label}: prediction labels are not finite binary values")
    if not np.isfinite(folds).all() or not np.isin(folds, range(N_FOLDS)).all():
        raise RuntimeError(f"{label}: prediction folds are invalid")

    context = manifest[
        ["slide_id", "patient_id", "target_label", "k_fold", "cohort", "subcohort"]
    ].copy()
    merged = predictions.merge(context, on="slide_id", how="left", validate="one_to_one")
    if merged["patient_id"].isna().any() or len(merged) != len(manifest):
        raise RuntimeError(f"{label}: OOF slide set does not exactly match its manifest")
    if set(map(str, predictions["slide_id"])) != set(map(str, manifest["slide_id"])):
        raise RuntimeError(f"{label}: OOF and manifest slide identities differ")
    if not np.array_equal(merged["label"].to_numpy(dtype=int), merged["target_label"].to_numpy(dtype=int)):
        raise RuntimeError(f"{label}: OOF labels differ from manifest labels")
    if not np.array_equal(merged["fold"].to_numpy(dtype=int), merged["k_fold"].to_numpy(dtype=int)):
        raise RuntimeError(f"{label}: OOF folds differ from manifest folds")
    merged["native_slide_logit"] = pd.to_numeric(merged["logit"], errors="raise")
    patients = (
        merged.sort_values("slide_id")
        .groupby("patient_id", sort=True, observed=True)
        .agg(
            label=("target_label", "first"),
            mean_logit=("native_slide_logit", "mean"),
            n_slides=("slide_id", "count"),
            cohort=("cohort", "first"),
            subcohort=("subcohort", "first"),
            k_fold=("k_fold", "first"),
        )
        .reset_index()
    )
    if not np.isfinite(patients["mean_logit"].to_numpy()).all():
        raise RuntimeError(f"{label}: patient-native logits are not all finite")
    return patients


def _validate_manifest(manifest: pd.DataFrame, *, label: str) -> None:
    required = {"slide_id", "patient_id", "target_label", "k_fold", "cohort", "subcohort"}
    missing = required - set(manifest)
    if missing:
        raise RuntimeError(f"{label}: manifest columns missing: {sorted(missing)}")
    if manifest["slide_id"].isna().any() or manifest["slide_id"].duplicated().any():
        raise RuntimeError(f"{label}: manifest slide_id is null or duplicated")
    if manifest["patient_id"].isna().any():
        raise RuntimeError(f"{label}: manifest patient_id is null")
    labels = pd.to_numeric(manifest["target_label"], errors="coerce").to_numpy(dtype=float)
    folds = pd.to_numeric(manifest["k_fold"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise RuntimeError(f"{label}: manifest labels are invalid")
    if not np.isfinite(folds).all() or set(map(int, np.unique(folds))) != set(range(N_FOLDS)):
        raise RuntimeError(f"{label}: manifest must contain exactly folds 0..4")
    _assert_patient_constant(
        manifest,
        ["target_label", "k_fold", "cohort", "subcohort"],
        label=label,
    )
    patient = manifest.drop_duplicates("patient_id")
    for fold, block in patient.groupby("k_fold", observed=True):
        if set(block["target_label"].astype(int)) != {0, 1}:
            raise RuntimeError(f"{label}: fold {fold} lacks a class")


@dataclass
class ChainValidation:
    patients: pd.DataFrame
    manifest_path: Path
    manifest_identity: dict[str, Any]
    manifest_resolution: dict[str, Any]
    files: dict[str, dict[str, Any]]
    summary: dict[str, Any]


def _validate_chain(chain: Path, *, task: str, seed: int, is_gene: bool) -> ChainValidation:
    label = f"{task}/seed{seed}"
    if not chain.is_dir():
        raise FileNotFoundError(f"{label}: chain directory missing: {chain}")
    files: dict[str, dict[str, Any]] = {}

    identity_path = chain / "training_identity.json"
    identity = _read_json(identity_path)
    files["training_identity"] = _artifact_identity(identity_path)
    if identity.get("schema_version") != 2 or identity.get("payload", {}).get("schema_version") != 2:
        raise RuntimeError(f"{label}: unsupported training identity schema")
    expected_fingerprint = hashlib.sha256(
        json.dumps(identity["payload"], sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:16]
    if identity.get("fingerprint") != expected_fingerprint:
        raise RuntimeError(f"{label}: training identity fingerprint is invalid")
    material = identity["payload"]["material_config"]
    training = material["training"]
    splits = material["splits"]
    data = material["data"]
    expected_model = "1a" if is_gene else f"e3a_{LEGACY_RENAMES.get(task, task)}"
    checks = {
        "data.aim1_model": data.get("aim1_model") == expected_model,
        "data.label_columns": data.get("label_columns") == ["target_label"],
        "training.seed": int(training.get("seed", -1)) == seed,
        "splits.seed": int(splits.get("seed", -1)) == seed,
        "splits.n_folds": int(splits.get("n_folds", -1)) == N_FOLDS,
        "splits.scheme": splits.get("scheme") == "predefined_oof_kfold",
        "training.dataset_max_instances": int(training.get("dataset_max_instances", -1)) == CAP,
        "training.eval_full_bags": training.get("eval_full_bags") is True,
        "training.train_sampling_strategy": training.get("train_sampling_strategy") == "patient_natural",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{label}: material training contract failed: {failed}")

    recorded_manifest_sha = str(identity["payload"]["input_evidence"]["manifest_sha256"])
    manifest_path, resolution = _manifest_path(
        identity, expected_task=task, recorded_sha=recorded_manifest_sha
    )
    manifest_identity = _artifact_identity(manifest_path)
    files["manifest"] = manifest_identity
    manifest = pd.read_csv(manifest_path)
    _validate_manifest(manifest, label=f"{label}/manifest")

    completion_path = chain / "training_completion.json"
    completion = _read_json(completion_path)
    files["training_completion"] = _artifact_identity(completion_path)
    if (
        completion.get("schema_version") != 1
        or completion.get("status") != "completed"
        or completion.get("skip_finalize") is not False
        or int(completion.get("n_folds", -1)) != N_FOLDS
        or completion.get("training_fingerprint") != expected_fingerprint
    ):
        raise RuntimeError(f"{label}: training completion contract failed")
    top_artifacts = completion.get("artifacts", {})
    if set(top_artifacts) != {"best_fold_checkpoint", "cv_summary"}:
        raise RuntimeError(f"{label}: unexpected chain artifact inventory")
    for role, evidence in top_artifacts.items():
        path = _relative_artifact(chain, evidence, label=f"{label}/{role}")
        files[f"completion_artifact/{role}"] = _artifact_identity(path)

    fold_receipts = completion.get("fold_completions", [])
    if len(fold_receipts) != N_FOLDS or len({item.get("path") for item in fold_receipts}) != N_FOLDS:
        raise RuntimeError(f"{label}: expected five distinct fold receipts")
    fold_predictions: list[pd.DataFrame] = []
    observed_folds: set[int] = set()
    for receipt_evidence in fold_receipts:
        receipt_path = _relative_artifact(
            chain, receipt_evidence, label=f"{label}/fold_completion_receipt"
        )
        fold_receipt = _read_json(receipt_path)
        fold = int(fold_receipt.get("fold", -1))
        observed_folds.add(fold)
        fold_root = receipt_path.parent
        files[f"fold{fold}/completion"] = _artifact_identity(receipt_path)
        if (
            fold_receipt.get("schema_version") != 1
            or fold_receipt.get("status") != "completed"
            or fold_receipt.get("training_fingerprint") != expected_fingerprint
        ):
            raise RuntimeError(f"{label}/fold{fold}: completion contract failed")
        expected_roles = {
            "best_checkpoint",
            "config",
            "metrics",
            "sampling_plan",
            "test_predictions",
            "val_predictions",
        }
        artifacts = fold_receipt.get("artifacts", {})
        if set(artifacts) != expected_roles:
            raise RuntimeError(f"{label}/fold{fold}: unexpected artifact inventory")
        for role, evidence in artifacts.items():
            path = _relative_artifact(fold_root, evidence, label=f"{label}/fold{fold}/{role}")
            files[f"fold{fold}/{role}"] = _artifact_identity(path)
            if role == "test_predictions":
                frame = pd.read_parquet(path)
                if "fold" not in frame:
                    frame = frame.assign(fold=fold)
                if set(pd.to_numeric(frame["fold"], errors="raise").astype(int)) != {fold}:
                    raise RuntimeError(f"{label}/fold{fold}: held-out predictions carry wrong fold")
                fold_predictions.append(frame)
    if observed_folds != set(range(N_FOLDS)):
        raise RuntimeError(f"{label}: fold receipt identities are not exactly 0..4")

    oof_path = chain / "oof_predictions.parquet"
    files["oof_predictions"] = _artifact_identity(oof_path)
    oof = pd.read_parquet(oof_path)
    pooled = pd.concat(fold_predictions, ignore_index=True)
    compare_columns = sorted(set(oof) | set(pooled))
    if set(oof) != set(pooled):
        raise RuntimeError(f"{label}: pooled OOF columns differ from fold predictions")
    left = oof.sort_values("slide_id").reset_index(drop=True)[compare_columns]
    right = pooled.sort_values("slide_id").reset_index(drop=True)[compare_columns]
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=True)
    except AssertionError as error:
        raise RuntimeError(f"{label}: pooled OOF is not the exact union of five folds") from error
    patients = _validate_prediction_frame(oof, manifest, label=f"{label}/OOF")

    cv_summary = _read_json(chain / "cv_summary.json")
    pooled_summary = cv_summary.get("oof_pooled", {})
    if int(pooled_summary.get("n_slides", -1)) != len(oof):
        raise RuntimeError(f"{label}: cv_summary OOF slide count mismatch")
    if int(pooled_summary.get("n_positive", -1)) != int(oof["label"].sum()):
        raise RuntimeError(f"{label}: cv_summary OOF positive count mismatch")
    reported_auc = float(pooled_summary.get("auroc", float("nan")))
    # The trainer's historical summary used its persisted probability column.
    # Validate that receipt on its own terms; corrected analysis below uses the
    # native logit and therefore deliberately retains native-logit ties.
    probabilities = pd.to_numeric(oof.get("prob_1"), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"{label}: persisted probabilities are not finite")
    actual_auc = _auc(oof["label"].to_numpy(), probabilities)
    if not np.isfinite(reported_auc) or abs(reported_auc - actual_auc) > 1e-12:
        raise RuntimeError(f"{label}: cv_summary pooled AUROC mismatch")

    return ChainValidation(
        patients=patients,
        manifest_path=manifest_path,
        manifest_identity=manifest_identity,
        manifest_resolution=resolution,
        files=files,
        summary={
            "task": task,
            "seed": seed,
            "training_fingerprint": expected_fingerprint,
            "n_folds": N_FOLDS,
            "n_slides": int(len(oof)),
            "n_patients": int(len(patients)),
            "n_positive_patients": int(patients["label"].sum()),
            "native_logits_finite": True,
        },
    )


def _ensemble_seed_frames(frames: dict[int, pd.DataFrame], *, task: str) -> pd.DataFrame:
    if set(frames) != set(SEEDS):
        raise RuntimeError(f"{task}: seed inventory must be exactly {SEEDS}")
    ordered = {
        seed: frame.sort_values("patient_id").reset_index(drop=True) for seed, frame in frames.items()
    }
    reference = ordered[SEEDS[0]]
    invariant = ["patient_id", "label", "n_slides", "cohort", "subcohort", "k_fold"]
    for seed in SEEDS[1:]:
        try:
            pd.testing.assert_frame_equal(
                reference[invariant], ordered[seed][invariant], check_exact=True, check_dtype=True
            )
        except AssertionError as error:
            raise RuntimeError(f"{task}: seeds do not cover the identical patient OOF") from error
    result = reference[invariant].copy()
    result["mean_logit"] = np.mean(
        np.vstack([ordered[seed]["mean_logit"].to_numpy(dtype=float) for seed in SEEDS]), axis=0
    )
    if not np.isfinite(result["mean_logit"].to_numpy()).all():
        raise RuntimeError(f"{task}: ensemble native logits are non-finite")
    return result


@dataclass
class InputBundle:
    e3_root: Path
    e0_root: Path
    tasks: dict[str, pd.DataFrame]
    per_seed: dict[str, dict[int, pd.DataFrame]]
    receipt: dict[str, Any]
    validation: dict[str, Any]


def validate_inputs(e3_root: Path, e0_root: Path) -> InputBundle:
    e3 = _require_absolute_directory(e3_root, label="--e3-root", must_exist=True)
    e0 = _require_absolute_directory(e0_root, label="--e0-root", must_exist=True)
    if e3 == e0 or e3 in e0.parents or e0 in e3.parents:
        raise ValueError("E3 and E0 input roots must be distinct and non-nested")

    train_root = e3 / "train"
    actual_tasks = {path.name for path in train_root.iterdir() if path.is_dir()}
    if actual_tasks != set(E3_TASKS):
        raise RuntimeError(
            f"E3 task inventory mismatch: expected {sorted(E3_TASKS)}, got {sorted(actual_tasks)}"
        )

    files: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    resolutions: dict[str, Any] = {}
    per_seed: dict[str, dict[int, pd.DataFrame]] = {}
    manifest_by_task: dict[str, dict[str, Any]] = {}

    for task in E3_TASKS:
        task_root = train_root / task
        actual_seeds = {path.name for path in task_root.iterdir() if path.is_dir() and path.name.startswith("seed")}
        expected_seeds = {f"seed{seed}" for seed in SEEDS}
        if actual_seeds != expected_seeds:
            raise RuntimeError(f"{task}: expected exact seed directories {sorted(expected_seeds)}")
        frames: dict[int, pd.DataFrame] = {}
        manifest_identities: list[dict[str, Any]] = []
        for seed in SEEDS:
            checked = _validate_chain(task_root / f"seed{seed}", task=task, seed=seed, is_gene=False)
            frames[seed] = checked.patients
            summaries.append(checked.summary)
            resolutions[f"{task}/seed{seed}"] = checked.manifest_resolution
            manifest_identities.append(checked.manifest_identity)
            for role, identity in checked.files.items():
                files[f"e3/{task}/seed{seed}/{role}"] = identity
        if len({item["sha256"] for item in manifest_identities}) != 1:
            raise RuntimeError(f"{task}: seeds do not share one manifest identity")
        manifest_by_task[task] = manifest_identities[0]
        per_seed[task] = frames

    actual_e0_seeds = {path.name for path in e0.iterdir() if path.is_dir() and path.name.startswith("seed")}
    expected_e0_seeds = {f"seed{seed}" for seed in SEEDS}
    if actual_e0_seeds != expected_e0_seeds:
        raise RuntimeError(f"E0: expected exact seed directories {sorted(expected_e0_seeds)}")
    gene_frames: dict[int, pd.DataFrame] = {}
    gene_manifests: list[dict[str, Any]] = []
    for seed in SEEDS:
        checked = _validate_chain(e0 / f"seed{seed}", task="gene", seed=seed, is_gene=True)
        gene_frames[seed] = checked.patients
        gene_manifests.append(checked.manifest_identity)
        summaries.append(checked.summary)
        resolutions[f"gene/seed{seed}"] = checked.manifest_resolution
        for role, identity in checked.files.items():
            files[f"e0/gene/seed{seed}/{role}"] = identity
    if len({item["sha256"] for item in gene_manifests}) != 1:
        raise RuntimeError("E0 seeds do not share one manifest identity")
    manifest_by_task["gene"] = gene_manifests[0]
    per_seed["gene"] = gene_frames

    tasks = {task: _ensemble_seed_frames(per_seed[task], task=task) for task in ALL_TASKS}
    pair_checks: dict[str, Any] = {}
    for fine, control in PAIRS:
        fine_frame = tasks[fine]
        control_frame = tasks[control]
        fine_positive = fine_frame[fine_frame["label"].eq(1)].set_index("patient_id")
        control_positive = control_frame[control_frame["label"].eq(1)].set_index("patient_id")
        if set(fine_positive.index.astype(str)) != set(control_positive.index.astype(str)):
            raise RuntimeError(f"{fine}: fine/control positive patient identities differ")
        shared = fine_positive[["cohort", "subcohort"]].join(
            control_positive[["cohort", "subcohort"]], lsuffix="_fine", rsuffix="_control"
        )
        if not (
            shared["cohort_fine"].eq(shared["cohort_control"]).all()
            and shared["subcohort_fine"].eq(shared["subcohort_control"]).all()
        ):
            raise RuntimeError(f"{fine}: shared-positive context differs between arms")
        fine_negative = fine_frame[fine_frame["label"].eq(0)]
        control_negative = control_frame[control_frame["label"].eq(0)]
        if len(fine_frame) != len(control_frame) or len(fine_negative) != len(control_negative):
            raise RuntimeError(f"{fine}: matched control class sizes differ")
        if _stratum_counts(fine_negative, ["cohort"]) != _stratum_counts(control_negative, ["cohort"]):
            raise RuntimeError(f"{fine}: control negative draw is not cohort-matched")
        pair_checks[fine] = {
            "control": control,
            "shared_positive_patients": int(len(fine_positive)),
            "fine_negative_patients": int(len(fine_negative)),
            "control_negative_patients": int(len(control_negative)),
            "cohort_matched_negative_counts": True,
            "control_draw_seed_from_locked_design": CONTROL_DRAW_SEEDS[control],
            "scope": "conditional on this fixed control manifest; alternative draws not sampled",
        }

    validation = {
        "status": "PASS",
        "e3_chains": len(E3_TASKS) * len(SEEDS),
        "e3_folds": len(E3_TASKS) * len(SEEDS) * N_FOLDS,
        "e3_fine_folds": len(FINE_TASKS) * len(SEEDS) * N_FOLDS,
        "e3_control_folds": len(FINE_TASKS) * len(SEEDS) * N_FOLDS,
        "e0_chains": len(SEEDS),
        "e0_folds": len(SEEDS) * N_FOLDS,
        "seeds": list(SEEDS),
        "cap": CAP,
        "native_logits": "finite for every slide, patient, seed, and ensemble",
        "oof_contract": "each pooled OOF is the exact union of five receipted held-out folds",
        "manifest_contract": "exact slide, label, fold, cohort, and patient alignment",
        "pair_checks": pair_checks,
    }
    receipt = {
        "schema_version": 1,
        "component": "aim3_corrected",
        "e3_root": str(e3),
        "e0_root": str(e0),
        "expected_e3_tasks": list(E3_TASKS),
        "expected_seeds": list(SEEDS),
        "files": files,
        "manifests": manifest_by_task,
        "manifest_resolution": resolutions,
        "chains": summaries,
        "validation": validation,
    }
    return InputBundle(e3, e0, tasks, per_seed, receipt, validation)


def _bootstrap_indices(
    frame: pd.DataFrame, columns: list[str], rng: np.random.Generator
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for _key, block in frame.groupby(columns, sort=True, dropna=False, observed=True):
        indices = block.index.to_numpy(dtype=int)
        pieces.append(rng.choice(indices, size=len(indices), replace=True))
    if not pieces:
        raise RuntimeError(f"No strata for {columns}")
    return np.concatenate(pieces)


def task_bootstrap(
    frame: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> np.ndarray:
    """Cohort x label-stratified patient bootstrap for one standalone task."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    work = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    y = work["label"].to_numpy(dtype=int)
    score = work["mean_logit"].to_numpy(dtype=float)
    values = np.empty(n_bootstrap, dtype=float)
    for repetition in range(n_bootstrap):
        index = _bootstrap_indices(work, ["cohort", "label"], rng)
        values[repetition] = _auc(y[index], score[index])
    return values


def partially_paired_bootstrap(
    fine: pd.DataFrame,
    control: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    stratum: str = "cohort",
    target_control_negative_counts: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Joint-positive / independent-negative bootstrap for one rung pair.

    With ``target_control_negative_counts``, control negatives are sampled from
    their fixed subcohort pools to the supplied target counts.  This is direct
    standardization, not a new control draw.
    """
    if stratum not in {"cohort", "subcohort"}:
        raise ValueError("stratum must be cohort or subcohort")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    fine = fine.reset_index(drop=True)
    control = control.reset_index(drop=True)
    fp = fine[fine["label"].eq(1)].set_index("patient_id", drop=False)
    cp = control[control["label"].eq(1)].set_index("patient_id", drop=False)
    if set(fp.index.astype(str)) != set(cp.index.astype(str)):
        raise ValueError("Fine/control positives must be the identical patients")
    fp = fp.sort_index()
    cp = cp.reindex(fp.index)
    if not fp[stratum].eq(cp[stratum]).all():
        raise ValueError("Shared-positive strata differ between arms")
    fn = fine[fine["label"].eq(0)]
    cn = control[control["label"].eq(0)]

    positive_ids = {
        str(key): block.index.to_numpy()
        for key, block in fp.groupby(stratum, sort=True, observed=True)
    }
    fine_negative = {
        str(key): block.index.to_numpy(dtype=int)
        for key, block in fn.groupby(stratum, sort=True, observed=True)
    }
    control_negative = {
        str(key): block.index.to_numpy(dtype=int)
        for key, block in cn.groupby(stratum, sort=True, observed=True)
    }
    if set(fine_negative) != set(control_negative):
        raise ValueError("Fine/control negative strata have different support")
    target = (
        {key: len(value) for key, value in fine_negative.items()}
        if target_control_negative_counts is None
        else {str(key): int(value) for key, value in target_control_negative_counts.items()}
    )
    if set(target) != set(control_negative) or any(value <= 0 for value in target.values()):
        raise ValueError("Control standardization target has invalid or unsupported strata")

    rng = np.random.default_rng(seed)
    fine_auc = np.empty(n_bootstrap, dtype=float)
    control_auc = np.empty(n_bootstrap, dtype=float)
    delta = np.empty(n_bootstrap, dtype=float)
    for repetition in range(n_bootstrap):
        fine_pos_scores: list[np.ndarray] = []
        control_pos_scores: list[np.ndarray] = []
        for ids in positive_ids.values():
            chosen = rng.choice(ids, size=len(ids), replace=True)
            fine_pos_scores.append(fp.loc[chosen, "mean_logit"].to_numpy(dtype=float))
            control_pos_scores.append(cp.loc[chosen, "mean_logit"].to_numpy(dtype=float))
        fine_neg_scores: list[np.ndarray] = []
        control_neg_scores: list[np.ndarray] = []
        for key in sorted(fine_negative):
            fi = rng.choice(fine_negative[key], size=len(fine_negative[key]), replace=True)
            ci = rng.choice(
                control_negative[key], size=target[key], replace=True
            )
            fine_neg_scores.append(fine.loc[fi, "mean_logit"].to_numpy(dtype=float))
            control_neg_scores.append(control.loc[ci, "mean_logit"].to_numpy(dtype=float))
        f_pos = np.concatenate(fine_pos_scores)
        c_pos = np.concatenate(control_pos_scores)
        f_neg = np.concatenate(fine_neg_scores)
        c_neg = np.concatenate(control_neg_scores)
        fine_auc[repetition] = _auc(
            np.r_[np.ones(len(f_pos), dtype=int), np.zeros(len(f_neg), dtype=int)],
            np.r_[f_pos, f_neg],
        )
        control_auc[repetition] = _auc(
            np.r_[np.ones(len(c_pos), dtype=int), np.zeros(len(c_neg), dtype=int)],
            np.r_[c_pos, c_neg],
        )
        delta[repetition] = control_auc[repetition] - fine_auc[repetition]
    return {"fine_auc": fine_auc, "control_auc": control_auc, "delta": delta}


def verdict_for(fine: dict[str, Any], control: dict[str, Any], delta: dict[str, Any]) -> str:
    f_low, f_high = map(float, fine["ci95"])
    c_low = float(control["ci95"][0])
    d_low = float(delta["ci95"][0])
    if c_low <= CHANCE:
        return "UNDERPOWERED"
    ceiling = f_high < CEILING_BOUND and c_low > CHANCE and d_low > 0.0
    above_chance = f_low > CHANCE
    if ceiling and above_chance:
        return "CEILING_WITH_RESIDUAL_SIGNAL"
    if ceiling:
        return "CEILING"
    if above_chance:
        return "FINE_RESOLUTION_EVIDENCE"
    return "INCONCLUSIVE"


def _task_point(frame: pd.DataFrame) -> float:
    return _auc(frame["label"].to_numpy(dtype=int), frame["mean_logit"].to_numpy(dtype=float))


def _per_seed_aurocs(frames: dict[int, pd.DataFrame]) -> dict[str, float]:
    return {str(seed): _task_point(frame) for seed, frame in sorted(frames.items())}


def _subcohort_standardized_point(fine: pd.DataFrame, control: pd.DataFrame) -> float:
    """Control AUROC standardized to fine label x subcohort composition."""
    from sklearn.metrics import roc_auc_score

    fine_counts = _stratum_counts(fine, ["label", "subcohort"])
    control_counts = _stratum_counts(control, ["label", "subcohort"])
    weights = np.empty(len(control), dtype=float)
    for index, row in control.reset_index(drop=True).iterrows():
        key = f"{int(row['label'])}|{row['subcohort']}"
        if key not in fine_counts or key not in control_counts or control_counts[key] <= 0:
            raise RuntimeError(f"Unsupported subcohort-standardization cell: {key}")
        weights[index] = fine_counts[key] / control_counts[key]
    return float(
        roc_auc_score(
            control["label"].to_numpy(dtype=int),
            control["mean_logit"].to_numpy(dtype=float),
            sample_weight=weights,
        )
    )


def build_report(bundle: InputBundle, *, n_bootstrap: int) -> dict[str, Any]:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    report: dict[str, Any] = {
        "schema_version": 1,
        "component": "aim3_corrected_cap8192",
        "inputs": {"e3_root": str(bundle.e3_root), "e0_root": str(bundle.e0_root)},
        "protocol": {
            "seeds": list(SEEDS),
            "cap": CAP,
            "patient_aggregation": "mean native slide logit, then mean across exactly three seeds",
            "task_ci": "patient bootstrap stratified by cohort x label",
            "pair_contrast": (
                "cohort-stratified partially paired patient bootstrap: shared positives jointly, "
                "fine/control negatives independently"
            ),
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "familywise_sensitivity": {
                "rungs": len(FINE_TASKS),
                "familywise_alpha": FAMILYWISE_ALPHA,
                "bonferroni_one_sided_alpha_per_rung": FAMILYWISE_ALPHA / len(FINE_TASKS),
                "bounds": "one-sided 99% bounds",
                "gate": "fine upper <0.60 AND control lower >0.50 AND delta lower >0",
                "intersection_union_note": (
                    "no additional within-rung component correction; all component tests must pass"
                ),
            },
            "subcohort_sensitivity": (
                "control negative arm directly standardized to the fine arm's label x subcohort "
                "composition, using only patients in the fixed control draw"
            ),
            "control_draw_scope": (
                "conditional on each pre-existing fixed control manifest; no alternative wild-type "
                "draws are generated or integrated over"
            ),
        },
        "input_validation": bundle.validation,
        "tasks": {},
        "pairs": {},
        "familywise_summary": {},
    }

    pair_draws: dict[str, dict[str, np.ndarray]] = {}
    for fine, control in PAIRS:
        pair_draws[fine] = partially_paired_bootstrap(
            bundle.tasks[fine],
            bundle.tasks[control],
            n_bootstrap=n_bootstrap,
            seed=_stable_seed(BOOTSTRAP_SEED, "primary_pair", fine),
            stratum="cohort",
        )
    gene_draws = task_bootstrap(
        bundle.tasks["gene"],
        n_bootstrap=n_bootstrap,
        seed=_stable_seed(BOOTSTRAP_SEED, "standalone", "gene"),
    )

    task_draw_map: dict[str, np.ndarray] = {"gene": gene_draws}
    for fine, control in PAIRS:
        task_draw_map[fine] = pair_draws[fine]["fine_auc"]
        task_draw_map[control] = pair_draws[fine]["control_auc"]
    for task in ALL_TASKS:
        frame = bundle.tasks[task]
        per_seed = _per_seed_aurocs(bundle.per_seed[task])
        report["tasks"][task] = {
            "kind": "reference" if task == "gene" else ("control" if task.startswith("ctrl_") else "fine"),
            "n": int(len(frame)),
            "n_positive": int(frame["label"].sum()),
            "n_negative": int((1 - frame["label"]).sum()),
            "auroc": _task_point(frame),
            "ci95": _interval(task_draw_map[task]),
            "bootstrap": {
                "n_completed": n_bootstrap,
                "seed": (
                    _stable_seed(BOOTSTRAP_SEED, "standalone", "gene")
                    if task == "gene"
                    else _stable_seed(
                        BOOTSTRAP_SEED,
                        "primary_pair",
                        task.removeprefix("ctrl_") if task.startswith("ctrl_") else task,
                    )
                ),
                "strata": "cohort x label",
            },
            "cohort_label_counts": _stratum_counts(frame, ["cohort", "label"]),
            "subcohort_label_counts": _stratum_counts(frame, ["subcohort", "label"]),
            "per_seed_auroc": per_seed,
            "seed_range": [min(per_seed.values()), max(per_seed.values())],
        }

    familywise_pass: list[str] = []
    for fine, control in PAIRS:
        draws = pair_draws[fine]
        fine_block = report["tasks"][fine]
        control_block = report["tasks"][control]
        delta_point = control_block["auroc"] - fine_block["auroc"]
        delta_block = {
            "estimate": delta_point,
            "ci95": _interval(draws["delta"]),
        }
        one_sided = {
            "fine_99_upper": float(np.quantile(draws["fine_auc"], 0.99)),
            "control_99_lower": float(np.quantile(draws["control_auc"], 0.01)),
            "delta_99_lower": float(np.quantile(draws["delta"], 0.01)),
        }
        fwer_gate = bool(
            one_sided["fine_99_upper"] < CEILING_BOUND
            and one_sided["control_99_lower"] > CHANCE
            and one_sided["delta_99_lower"] > 0.0
        )
        if fwer_gate:
            familywise_pass.append(fine)

        fine_negative_counts = {
            str(key): int(value)
            for key, value in bundle.tasks[fine][bundle.tasks[fine]["label"].eq(0)]
            .groupby("subcohort", sort=True, observed=True)
            .size()
            .items()
        }
        standardized = partially_paired_bootstrap(
            bundle.tasks[fine],
            bundle.tasks[control],
            n_bootstrap=n_bootstrap,
            seed=_stable_seed(BOOTSTRAP_SEED, "subcohort_standardized", fine),
            stratum="subcohort",
            target_control_negative_counts=fine_negative_counts,
        )
        standardized_control_point = _subcohort_standardized_point(
            bundle.tasks[fine], bundle.tasks[control]
        )
        standardized_delta_point = standardized_control_point - fine_block["auroc"]
        report["pairs"][fine] = {
            "control": control,
            "shared_positive_patients": fine_block["n_positive"],
            "primary_delta_control_minus_fine": delta_block,
            "nominal_95_verdict": verdict_for(fine_block, control_block, delta_block),
            "familywise_ceiling_sensitivity": {
                **one_sided,
                "pass": fwer_gate,
                "status": "CEILING_FWER_PASS" if fwer_gate else "CEILING_FWER_NOT_ESTABLISHED",
            },
            "subcohort_standardized_control_sensitivity": {
                "control_auroc": standardized_control_point,
                "control_ci95": _interval(standardized["control_auc"]),
                "delta_control_minus_fine": standardized_delta_point,
                "delta_ci95": _interval(standardized["delta"]),
                "target_negative_counts": fine_negative_counts,
                "control_pool_counts": {
                    str(key): int(value)
                    for key, value in bundle.tasks[control][bundle.tasks[control]["label"].eq(0)]
                    .groupby("subcohort", sort=True, observed=True)
                    .size()
                    .items()
                },
                "bootstrap_seed": _stable_seed(
                    BOOTSTRAP_SEED, "subcohort_standardized", fine
                ),
            },
            "fixed_control_draw": {
                "design_seed": CONTROL_DRAW_SEEDS[control],
                "scope": "conditional on the frozen observed draw",
            },
        }
    report["familywise_summary"] = {
        "status": "PASS",
        "rungs_passing_ceiling_gate": familywise_pass,
        "n_passing": len(familywise_pass),
        "interpretation": (
            "Only listed rungs establish the three-part ceiling gate under the five-rung "
            "familywise sensitivity. Absence from the list is not evidence of equivalence."
        ),
    }
    return report


def audit_report(report: dict[str, Any], bundle: InputBundle) -> dict[str, Any]:
    if report.get("component") != "aim3_corrected_cap8192":
        raise RuntimeError("Wrong result component")
    if set(report.get("tasks", {})) != set(ALL_TASKS):
        raise RuntimeError("Result task inventory mismatch")
    if set(report.get("pairs", {})) != set(FINE_TASKS):
        raise RuntimeError("Result pair inventory mismatch")
    checks: dict[str, Any] = {
        "input_validation": bundle.validation["status"],
        "task_inventory": "PASS",
        "pair_inventory": "PASS",
        "point_contrasts": "PASS",
        "bootstrap_counts": "PASS",
        "finite_statistics": "PASS",
    }
    n_bootstrap = int(report["protocol"]["n_bootstrap"])
    for task, block in report["tasks"].items():
        if int(block["bootstrap"]["n_completed"]) != n_bootstrap:
            raise RuntimeError(f"{task}: incomplete bootstrap")
        values = [block["auroc"], *block["ci95"], *block["seed_range"]]
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise RuntimeError(f"{task}: non-finite result statistic")
    for fine, block in report["pairs"].items():
        expected = report["tasks"][block["control"]]["auroc"] - report["tasks"][fine]["auroc"]
        if block["primary_delta_control_minus_fine"]["estimate"] != expected:
            raise RuntimeError(f"{fine}: point delta mismatch")
        sensitivity = block["subcohort_standardized_control_sensitivity"]
        if sensitivity["delta_control_minus_fine"] != sensitivity["control_auroc"] - report["tasks"][fine]["auroc"]:
            raise RuntimeError(f"{fine}: standardized point delta mismatch")
    return {
        "schema_version": 1,
        "component": "aim3_corrected",
        "status": "PASS",
        "checks": checks,
        "counts": {
            "e3_chains": 30,
            "e3_folds": 150,
            "e0_chains": 3,
            "e0_folds": 15,
            "tasks": len(ALL_TASKS),
            "rung_pairs": len(PAIRS),
            "bootstrap_repetitions_per_endpoint": n_bootstrap,
        },
        "claim_guardrails": [
            "Nominal 95% rung verdicts are not familywise-confirmed unless the 99% IUT gate passes.",
            "Inference is conditional on each fixed control draw, not all possible wild-type draws.",
            "A failed ceiling gate is inconclusive and is not equivalence or proof of no signal.",
            "The subcohort analysis is a sensitivity analysis, not a replacement primary endpoint.",
        ],
    }


def _source_payload() -> bytes:
    return Path(__file__).read_bytes()


def _validate_recorded_identity(recorded: dict[str, Any], *, expected_path: Path | None = None) -> None:
    if not isinstance(recorded, dict) or set(recorded) != {"path", "size_bytes", "sha256"}:
        raise RuntimeError("Malformed recorded file identity")
    path = Path(str(recorded["path"]))
    if not path.is_absolute():
        raise RuntimeError(f"Recorded identity path is not absolute: {path}")
    if expected_path is not None and path.resolve(strict=True) != expected_path.resolve(strict=True):
        raise RuntimeError(f"Recorded path mismatch: {path} != {expected_path}")
    actual = _artifact_identity(path)
    if actual["size_bytes"] != int(recorded["size_bytes"]) or actual["sha256"] != recorded["sha256"]:
        raise RuntimeError(f"Recorded identity changed: {path}")


def run_analysis(bundle: InputBundle, output_root: Path, *, n_bootstrap: int) -> Path:
    raw = output_root
    if raw.exists() or raw.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output root: {raw}")
    output = _require_absolute_directory(raw, label="--output-root", must_exist=False)
    if output == bundle.e3_root or output == bundle.e0_root:
        raise ValueError("Output root cannot equal an input root")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent must already exist: {output.parent}")
    source_bytes = _source_payload()
    output.mkdir(mode=0o755, exist_ok=False)

    source_snapshot = output / "source_snapshot" / "aim3_fixed_control_analysis.py"
    _write_bytes_once_atomic(source_snapshot, source_bytes)
    source_receipt = {
        "schema_version": 1,
        "component": "aim3_corrected",
        "live_source_at_start": _artifact_identity(Path(__file__)),
        "source_snapshot": _artifact_identity(source_snapshot),
    }
    _write_json_once_atomic(output / "receipts" / "source.json", source_receipt)
    _write_json_once_atomic(output / "receipts" / "inputs.json", bundle.receipt)
    _write_json_once_atomic(
        output / "lineage_start.json",
        {
            "schema_version": 1,
            "component": "aim3_corrected",
            "status": "started",
            "created_at_utc": _utc_now(),
            "output_root": str(output),
            "e3_root": str(bundle.e3_root),
            "e0_root": str(bundle.e0_root),
            "n_bootstrap": n_bootstrap,
        },
    )

    report = build_report(bundle, n_bootstrap=n_bootstrap)
    if _source_payload() != source_bytes:
        raise RuntimeError("Source bytes changed during analysis")
    result_path = output / "analysis" / "aim3_corrected.json"
    _write_json_once_atomic(result_path, report)
    result_receipt = {
        "schema_version": 1,
        "component": "aim3_corrected",
        "result": _artifact_identity(result_path),
        "input_receipt": _artifact_identity(output / "receipts" / "inputs.json"),
    }
    _write_json_once_atomic(output / "receipts" / "result.json", result_receipt)
    audit = audit_report(report, bundle)
    _write_json_once_atomic(output / "receipts" / "audit.json", audit)
    artifacts = {
        "lineage_start": output / "lineage_start.json",
        "source": output / "receipts" / "source.json",
        "inputs": output / "receipts" / "inputs.json",
        "result": result_path,
        "result_receipt": output / "receipts" / "result.json",
        "audit": output / "receipts" / "audit.json",
        "source_snapshot": source_snapshot,
    }
    _write_json_once_atomic(
        output / "lineage_complete.json",
        {
            "schema_version": 1,
            "component": "aim3_corrected",
            "status": "completed",
            "created_at_utc": _utc_now(),
            "artifacts": {name: _artifact_identity(path) for name, path in artifacts.items()},
        },
    )
    return output


def verify_output(bundle: InputBundle, output_root: Path) -> dict[str, Any]:
    output = _require_absolute_directory(output_root, label="--output-root", must_exist=True)
    complete = _read_json(output / "lineage_complete.json")
    if complete.get("status") != "completed" or complete.get("component") != "aim3_corrected":
        raise RuntimeError("Output has no valid completion receipt")
    expected_paths = {
        "lineage_start": output / "lineage_start.json",
        "source": output / "receipts" / "source.json",
        "inputs": output / "receipts" / "inputs.json",
        "result": output / "analysis" / "aim3_corrected.json",
        "result_receipt": output / "receipts" / "result.json",
        "audit": output / "receipts" / "audit.json",
        "source_snapshot": output / "source_snapshot" / "aim3_fixed_control_analysis.py",
    }
    if set(complete.get("artifacts", {})) != set(expected_paths):
        raise RuntimeError("Completion artifact inventory mismatch")
    for name, path in expected_paths.items():
        recorded = complete["artifacts"][name]
        if Path(str(recorded.get("path", ""))).resolve(strict=True) != path.resolve(strict=True):
            raise RuntimeError(f"Completion path mismatch for {name}")
        _validate_recorded_identity(recorded, expected_path=path)
    if (output / "source_snapshot" / "aim3_fixed_control_analysis.py").read_bytes() != _source_payload():
        raise RuntimeError("Current source differs from the immutable run snapshot")
    if _read_json(output / "receipts" / "inputs.json") != bundle.receipt:
        raise RuntimeError("Current upstream identities differ from the input receipt")
    start = _read_json(output / "lineage_start.json")
    if start.get("e3_root") != str(bundle.e3_root) or start.get("e0_root") != str(bundle.e0_root):
        raise RuntimeError("Output roots do not match explicit current inputs")
    recorded_report = _read_json(expected_paths["result"])
    rebuilt = build_report(bundle, n_bootstrap=int(start["n_bootstrap"]))
    if _canonical_json(recorded_report) != _canonical_json(rebuilt):
        raise RuntimeError("Deterministic statistical recomputation differs from result")
    recorded_audit = _read_json(expected_paths["audit"])
    rebuilt_audit = audit_report(rebuilt, bundle)
    if _canonical_json(recorded_audit) != _canonical_json(rebuilt_audit):
        raise RuntimeError("Recomputed audit differs from recorded audit")
    return {
        "schema_version": 1,
        "component": "aim3_corrected",
        "status": "PASS",
        "output_root": str(output),
        "result": _artifact_identity(expected_paths["result"]),
        "checks": {
            "completion_inventory": "PASS",
            "source_snapshot": "PASS",
            "upstream_identities": "PASS",
            "deterministic_statistical_recomputation": "PASS",
            "audit_recomputation": "PASS",
        },
    }


def _add_input_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--e3-root", type=Path, required=True)
    parser.add_argument("--e0-root", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    verify_inputs_parser = commands.add_parser("verify-inputs")
    _add_input_roots(verify_inputs_parser)
    run_parser = commands.add_parser("run")
    _add_input_roots(run_parser)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    verify_parser = commands.add_parser("verify-output")
    _add_input_roots(verify_parser)
    verify_parser.add_argument("--output-root", type=Path, required=True)
    show_parser = commands.add_parser("show")
    show_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "show":
        output = _require_absolute_directory(args.output_root, label="--output-root", must_exist=True)
        print(json.dumps(_read_json(output / "analysis" / "aim3_corrected.json"), indent=2))
        return
    bundle = validate_inputs(args.e3_root, args.e0_root)
    if args.command == "verify-inputs":
        print(json.dumps(bundle.validation, indent=2, sort_keys=True, allow_nan=False))
    elif args.command == "run":
        output = run_analysis(bundle, args.output_root, n_bootstrap=args.n_bootstrap)
        print(f"PASS — wrote immutable Aim 3 corrected lineage: {output}")
    elif args.command == "verify-output":
        print(json.dumps(verify_output(bundle, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
