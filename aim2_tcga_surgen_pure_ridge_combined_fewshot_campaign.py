#!/usr/bin/env python3
"""Lean governed metastatic pure-ridge few-shot campaign.

This immutable sibling answers only the paragraph-completion question.  It
binds the five frozen TCGA+SurGen-primary UNI-v1 p75 source models, seals
label-blind RIH-M and SR1482-M native logits plus 512-dimensional embeddings,
and then evaluates exactly one target-internal adaptation method:

    eta_probe = H @ w + b

The probe minimizes mean binary logistic loss plus
``lambda/2 * (||w||^2 + b^2)`` on combined metastatic support patients.  The
bias is penalized; raw frozen patient-mean 512-D MIL embeddings are used with
no target-wide scaling.  Supports are k=2/4/8/16 patients per class TOTAL,
with exactly k/2 patients from each cohort within each class.  Lambda is
selected separately inside each support set only.  The encoder, MIL model,
source classifier, and native logits remain frozen.

All target-label results are ``TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION``.
CPTAC, Orion, primary tumors, residual adapters, local MIL, full-label fits,
Platt calibration, and target-driven method/budget selection are absent.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import aim2_tcga_surgen_source_anchored_met_adaptation_campaign as broad  # noqa: E402

SCHEMA_VERSION = 1
CAMPAIGN = "aim2_tcga_surgen_pure_ridge_combined_fewshot_univ1_5seed"
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_tcga_surgen_pure_ridge_combined_fewshot_univ1_5seed_v1_20260828"
)
RERUNS_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns")
SOURCE_ROOT = broad.SOURCE_ROOT
MODEL_SEEDS = broad.MODEL_SEEDS
FOLDS = broad.FOLDS
LAYOUT_SEEDS = broad.OUTER_LAYOUT_SEEDS
SUPPORT_PER_CLASS = (2, 4, 8, 16)
DRAWS_PER_LAYOUT = 20
PROCEDURES_PER_BUDGET = len(LAYOUT_SEEDS) * DRAWS_PER_LAYOUT
LAMBDA_GRID = broad.LAMBDA_GRID
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = broad.BOOTSTRAP_SEED
MAX_WORKERS = 6
DEFAULT_NUM_WORKERS = 4
EMBED_DIM = broad.EMBED_DIM
INTERNAL_ROLE = broad.INTERNAL_ROLE
METHOD = "pure_ridge_linear_probe"
EXPECTED_NATIVE = broad.EXPECTED_NATIVE
EXPECTED_NATIVE_POOLED = 0.5926346528228423
EXPECTED_NATIVE_EQUAL_MACRO = 0.6059633497133496
EXPECTED_NATIVE_PER_SEED = {
    "RIH-M": {
        "42": 0.6103603603603605,
        "43": 0.6407657657657657,
        "44": 0.6469594594594595,
        "45": 0.5864301801801802,
        "46": 0.5931869369369369,
    },
    "SurGen-M": {
        "42": 0.5674242424242424,
        "43": 0.5534090909090909,
        "44": 0.5780303030303031,
        "45": 0.5897727272727273,
        "46": 0.5560606060606061,
    },
    "pooled_combined": {
        "42": 0.581440622972096,
        "43": 0.5815217391304348,
        "44": 0.6031797534068787,
        "45": 0.5867131732641142,
        "46": 0.5709766385463985,
    },
    "equal_cohort_macro": {
        "42": 0.5888923013923014,
        "43": 0.5970874283374283,
        "44": 0.6124948812448814,
        "45": 0.5881014537264537,
        "46": 0.5746237714987715,
    },
}
TARGETS = broad.TARGETS
TARGET_ORDER = broad.TARGET_ORDER


class ContractError(RuntimeError):
    """Fail-closed lean-campaign contract violation."""


@dataclass(frozen=True, order=True)
class ProbeJob:
    layout_seed: int
    support_per_class: int

    def __post_init__(self) -> None:
        if self.layout_seed not in LAYOUT_SEEDS or self.support_per_class not in SUPPORT_PER_CLASS:
            raise ValueError(self)

    @property
    def key(self) -> str:
        return f"layout{self.layout_seed}__k{self.support_per_class}"


def probe_jobs() -> list[ProbeJob]:
    return [ProbeJob(layout, support) for layout in LAYOUT_SEEDS for support in SUPPORT_PER_CLASS]


def accounting() -> dict[str, int]:
    return {
        "source_main_model_fits": 0,
        "label_blind_embedding_jobs": 10,
        "unique_support_procedures_by_fold": 2_000,
        "final_probe_head_decisions": 10_000,
        "fit_to_test_cohort_applications": 20_000,
        "inner_plus_final_solver_calls": 1_330_000,
        "residual_adapter_fits": 0,
        "local_mil_fits": 0,
        "full_label_fits": 0,
        "platt_fits": 0,
    }


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _utcnow_precise() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _parse_utc(value: Any, *, context: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ContractError(f"{context}: timestamp is not a string")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ContractError(f"{context}: invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ContractError(f"{context}: timestamp is not UTC-aware")
    return parsed


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_output_root(path: Path, *, must_exist: bool | None = None) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ContractError("--output-root must be absolute")
    lexical = raw.absolute()
    resolved = raw.resolve(strict=False)
    if lexical != resolved:
        raise ContractError("output root must be normalized and not traverse symlinks")
    cursor = raw
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise ContractError(f"output root traverses symlink: {cursor}")
        cursor = cursor.parent
    source = SOURCE_ROOT.resolve(strict=False)
    if resolved == source or _is_relative_to(resolved, source) or _is_relative_to(source, resolved):
        raise ContractError("lean output must never overlap source/main-model root")
    production = DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    if resolved != production and not _is_relative_to(resolved, Path("/tmp").resolve()):
        raise ContractError(f"production root must be exactly {production}; tests may use /tmp")
    if resolved == production and not _is_relative_to(resolved, RERUNS_ROOT.resolve()):
        raise ContractError("production root escaped governed reruns")
    if must_exist is True and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if must_exist is False and (resolved.exists() or resolved.is_symlink()):
        raise FileExistsError(resolved)
    return resolved


def _artifact(path: Path) -> dict[str, Any]:
    try:
        return broad._artifact(path)  # noqa: SLF001
    except broad.ContractError as exc:
        raise ContractError(str(exc)) from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return broad._read_json(path)  # noqa: SLF001
    except broad.ContractError as exc:
        raise ContractError(str(exc)) from exc


def _write_json_once(path: Path, value: Any) -> None:
    broad._write_json_once(path, value)  # noqa: SLF001


def _write_jsonl_once(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    broad._write_jsonl_once(path, rows)  # noqa: SLF001


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return broad._read_jsonl(path)  # noqa: SLF001
    except broad.ContractError as exc:
        raise ContractError(str(exc)) from exc


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    broad._write_parquet_once(path, frame)  # noqa: SLF001


def _write_bytes_once(path: Path, payload: bytes) -> None:
    broad._write_bytes_once(path, payload)  # noqa: SLF001


def contract_path(root: Path) -> Path:
    return root / "contract.json"


def target_manifest_path(root: Path) -> Path:
    return root / "inputs/target_internal/labeled_metastatic.csv"


def target_open_path(root: Path) -> Path:
    return root / "receipts/target_internal_open.json"


def shard_dir(root: Path, job: ProbeJob) -> Path:
    return root / f"adaptation/shards/{job.key}"


def scheduler_path(root: Path) -> Path:
    return root / "receipts/pure_ridge_scheduler.json"


def aggregate_oof_path(root: Path) -> Path:
    return root / "adaptation/pure_ridge_oof.parquet"


def results_path(root: Path) -> Path:
    return root / "analysis/results.json"


def completion_path(root: Path) -> Path:
    return root / "analysis/completion.json"


def _implementation_sources() -> dict[str, Any]:
    paths = {
        "lean_controller": Path(__file__).resolve(),
        "lean_test": REPO
        / "tests/test_aim2_tcga_surgen_pure_ridge_combined_fewshot_campaign.py",
        "audited_source_controller": Path(broad.__file__).resolve(),
        "embedding_runner": REPO / "aim2_confirmatory_transfer.py",
        "ridge_solver": REPO / "aim2_v3_fulllabel_residual_adaptation.py",
    }
    return {name: _artifact(path) for name, path in paths.items()}


def _contract_payload(
    root: Path, source: Mapping[str, Any], *, created_utc: str
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    all_slides: set[str] = set()
    all_patients: set[str] = set()
    for target, spec in TARGETS.items():
        path = broad.blind_path(root, target)
        frame = broad._validate_blind(  # noqa: SLF001
            pd.read_csv(path, low_memory=False), spec, context=target
        )
        slides = set(frame["slide_id"].astype(str))
        patients = set(frame["patient_id"].astype(str))
        if all_slides & slides or all_patients & patients:
            raise ContractError("blind target rosters overlap")
        all_slides |= slides
        all_patients |= patients
        targets[target] = {
            "artifact": _artifact(path),
            "slides": spec.slides,
            "patients": spec.patients,
            "outcome_source_not_opened": {
                "path": str(spec.outcome_source),
                "expected_sha256": spec.outcome_sha256,
            },
            "source_family_exposed": spec.source_family_exposed,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "PREPARED_LABEL_BLIND_LEAN_PURE_RIDGE",
        "created_utc": created_utc,
        "output_root": str(root),
        "source": dict(source),
        "targets": targets,
        "target_census": {"slides": 185, "patients": 159, "mutant_after_open": 67},
        "protocol": {
            "method": METHOD,
            "formula": "eta_probe = H @ w + b",
            "representation": (
                "raw frozen 512-dimensional patient-mean MIL embedding; no target-wide scaling"
            ),
            "objective": "mean binary logistic loss + lambda/2*(||w||^2+b^2)",
            "bias_penalized": True,
            "source_models_frozen": True,
            "source_model_seeds": list(MODEL_SEEDS),
            "layout_seeds": list(LAYOUT_SEEDS),
            "outer_folds": list(FOLDS),
            "support_per_class_total": list(SUPPORT_PER_CLASS),
            "combined_support_balance": (
                "exactly k/2 patients from each cohort within each outcome class"
            ),
            "draws_per_layout": DRAWS_PER_LAYOUT,
            "procedures_per_budget": PROCEDURES_PER_BUDGET,
            "lambda_grid": [broad.adapter.lambda_key(value) for value in LAMBDA_GRID],
            "lambda_selection": (
                "inside support only; leave-one-out when support total <=20 and "
                "stratified five-fold otherwise; same inner layout across source seeds"
            ),
            "test_cohorts_per_head": ["RIH-M", "SurGen-M"],
            "bootstrap_draws": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "maximum_parallel_workers": MAX_WORKERS,
        },
        "accounting": accounting(),
        "firewall": {
            "result_role": INTERNAL_ROLE,
            "external_validation_claim_permitted": False,
            "primary_support_permitted": False,
            "allowed_target_labels": ["RIH-M", "SurGen-M"],
            "combined_support_only": True,
            "all_budgets_reported": True,
            "budget_selected_on_target_results": False,
            "residual_adapter_permitted": False,
            "local_mil_permitted": False,
            "full_label_or_platt_permitted": False,
            "source_or_main_model_mutation": False,
            "surgen_metastatic_source_family_exposed": True,
        },
        "implementation": _implementation_sources(),
    }


def _source_score_command(root: Path, job: broad.ScoreJob, *, num_workers: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_score-one",
        "--output-root",
        str(root),
        "--target",
        job.target,
        "--seed",
        str(job.seed),
        "--num-workers",
        str(num_workers),
    ]


@contextlib.contextmanager
def _broad_context() -> Iterable[None]:
    names = {
        "DEFAULT_OUTPUT_ROOT": DEFAULT_OUTPUT_ROOT,
        "CAMPAIGN": CAMPAIGN,
        "validate_output_root": validate_output_root,
        "_implementation_sources": _implementation_sources,
        "_contract_payload": _contract_payload,
        "_source_score_command": _source_score_command,
    }
    old = {name: getattr(broad, name) for name in names}
    try:
        for name, value in names.items():
            setattr(broad, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(broad, name, value)


def prepare(root: Path, *, apply: bool) -> dict[str, Any]:
    with _broad_context():
        try:
            return broad.prepare(root, apply=apply)
        except broad.ContractError as exc:
            raise ContractError(str(exc)) from exc


def load_contract(root: Path, *, deep_pack: bool) -> dict[str, Any]:
    with _broad_context():
        try:
            return broad.load_contract(root, deep_pack=deep_pack)
        except broad.ContractError as exc:
            raise ContractError(str(exc)) from exc


def preflight(root: Path, *, apply: bool, deep_pack: bool = True) -> dict[str, Any]:
    with _broad_context():
        try:
            return broad.preflight(root, apply=apply, deep_pack=deep_pack)
        except broad.ContractError as exc:
            raise ContractError(str(exc)) from exc


def verify_preflight(root: Path, *, deep_pack: bool) -> dict[str, Any]:
    with _broad_context():
        try:
            return broad.verify_preflight(root, deep_pack=deep_pack)
        except broad.ContractError as exc:
            raise ContractError(str(exc)) from exc


def score_source(
    root: Path, *, apply: bool, max_workers: int, num_workers: int
) -> dict[str, Any]:
    with _broad_context():
        try:
            return broad.score_source(
                root,
                apply=apply,
                max_workers=max_workers,
                num_workers=num_workers,
            )
        except broad.ContractError as exc:
            raise ContractError(str(exc)) from exc


def verify_inference_seal(root: Path) -> dict[str, Any]:
    with _broad_context():
        try:
            return broad.verify_inference_seal(root)
        except broad.ContractError as exc:
            raise ContractError(str(exc)) from exc


def _target_payload(
    root: Path,
    *,
    created_utc: str,
    outcomes: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "TARGET_LABELS_OPENED_FOR_LEAN_INTERNAL_ANALYSIS_ONLY",
        "created_utc": created_utc,
        "result_role": INTERNAL_ROLE,
        "source_inference_seal": _artifact(broad.inference_seal_path(root)),
        "outcome_sources": dict(outcomes),
        "labeled_manifest": _artifact(target_manifest_path(root)),
        "population": {"slides": 185, "patients": 159, "mutant": 67},
        "allowed_target_labels": ["RIH-M", "SurGen-M"],
        "combined_support_only": True,
        "external_validation_claim_permitted": False,
        "surgen_metastatic_source_family_exposed": True,
    }


def _expected_labeled_target(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    with _broad_context():
        try:
            combined, outcomes = broad._open_target_outcomes_after_seal(root)  # noqa: SLF001
        except broad.ContractError as exc:
            raise ContractError(str(exc)) from exc
    source = pd.read_csv(
        SOURCE_ROOT / "inputs/manifests/tcga_surgen_primary.csv", low_memory=False
    )
    if (
        set(combined["slide_id"].astype(str)) & set(source["slide_id"].astype(str))
        or set(combined["patient_id"].astype(str)) & set(source["patient_id"].astype(str))
    ):
        raise ContractError("target patients/slides overlap source development")
    combined = combined.sort_values(
        ["cohort", "patient_id", "slide_id"], kind="mergesort"
    ).reset_index(drop=True)
    return combined, outcomes


def _validate_target_open(root: Path) -> dict[str, Any]:
    verify_inference_seal(root)
    receipt = _read_json(target_open_path(root))
    created = receipt.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("lean target-open receipt lacks created_utc")
    _parse_utc(created, context="target-open")
    expected, outcomes = _expected_labeled_target(root)
    expected_bytes = expected.to_csv(index=False).encode()
    if target_manifest_path(root).read_bytes() != expected_bytes:
        raise ContractError("lean labeled target canonical bytes drifted")
    observed = pd.read_csv(target_manifest_path(root), low_memory=False)
    if list(observed.columns) != list(expected.columns):
        raise ContractError("lean labeled target schema drifted")
    observed = observed.sort_values(
        ["cohort", "patient_id", "slide_id"], kind="mergesort"
    ).reset_index(drop=True)
    canonical_observed = observed.to_csv(index=False).encode()
    if canonical_observed != expected_bytes:
        raise ContractError("lean labeled target semantics do not replay")
    payload = _target_payload(root, created_utc=created, outcomes=outcomes)
    if receipt != payload:
        raise ContractError("lean target-open receipt does not replay")
    return receipt


def open_targets(root: Path, *, apply: bool) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    verify_inference_seal(output)
    if not apply:
        return {
            "status": "DRY_RUN_READY_TO_OPEN_TARGET_LABELS",
            "result_role_after_open": INTERNAL_ROLE,
        }
    if target_open_path(output).exists():
        return _validate_target_open(output)
    if target_manifest_path(output).exists():
        raise ContractError("partial lean target-open namespace exists")
    combined, outcomes = _expected_labeled_target(output)
    _write_bytes_once(target_manifest_path(output), combined.to_csv(index=False).encode())
    _write_json_once(
        target_open_path(output),
        _target_payload(output, created_utc=_utcnow(), outcomes=outcomes),
    )
    return _validate_target_open(output)


def _patient_data(root: Path) -> dict[str, dict[int, pd.DataFrame]]:
    _validate_target_open(root)
    manifest = pd.read_csv(target_manifest_path(root), low_memory=False)
    embedding_columns = [f"e{index}" for index in range(EMBED_DIM)]
    result: dict[str, dict[int, pd.DataFrame]] = {}
    target_for_cohort = {spec.cohort: target for target, spec in TARGETS.items()}
    for cohort in ("RIH", "SurGen"):
        target = target_for_cohort[cohort]
        spec = TARGETS[target]
        target_manifest = manifest.loc[
            manifest["cohort"].astype(str).eq(cohort),
            ["slide_id", "patient_id", "target_label"],
        ]
        per_seed: dict[int, pd.DataFrame] = {}
        for seed in MODEL_SEEDS:
            job = broad.ScoreJob(target, seed)
            with _broad_context():
                scores = broad._validate_cached_embedding(root, job)  # noqa: SLF001
            if scores is None:
                raise ContractError(f"{job.key}: sealed embedding cache is missing")
            joined = scores.merge(target_manifest, on="slide_id", validate="one_to_one")
            patient = joined.groupby("patient_id", sort=True).agg(
                label=("target_label", "first"),
                eta_native=("logit", "mean"),
                **{column: (column, "mean") for column in embedding_columns},
            )
            patient["label"] = patient["label"].astype(int)
            if (
                len(patient) != spec.patients
                or int(patient["label"].sum()) != spec.mutant
                or not np.isfinite(
                    patient[["eta_native", *embedding_columns]].to_numpy(float)
                ).all()
            ):
                raise ContractError(f"{job.key}: patient embedding census/value drifted")
            per_seed[seed] = patient
        reference = per_seed[MODEL_SEEDS[0]]
        for seed in MODEL_SEEDS[1:]:
            if (
                not reference.index.equals(per_seed[seed].index)
                or not np.array_equal(reference["label"], per_seed[seed]["label"])
            ):
                raise ContractError(f"{cohort}: patient/label order differs by source seed")
        native = np.mean(
            np.vstack([per_seed[seed]["eta_native"].to_numpy(float) for seed in MODEL_SEEDS]),
            axis=0,
        )
        point = float(roc_auc_score(reference["label"].to_numpy(int), native))
        if not math.isclose(point, EXPECTED_NATIVE[cohort]["auroc"], abs_tol=1e-15):
            raise ContractError(f"{cohort}: native authority point drifted")
        result[cohort] = per_seed
    if set(result["RIH"][42].index.astype(str)) & set(
        result["SurGen"][42].index.astype(str)
    ):
        raise ContractError("RIH-M and SurGen-M patient IDs overlap")
    return result


def _folds_by_cohort(
    data: Mapping[str, Mapping[int, pd.DataFrame]], layout_seed: int
) -> dict[str, np.ndarray]:
    return {
        cohort: broad.adapter.stratified_folds(
            data[cohort][MODEL_SEEDS[0]]["label"].to_numpy(int), layout_seed
        )
        for cohort in ("RIH", "SurGen")
    }


def _combined_support(
    data: Mapping[str, Mapping[int, pd.DataFrame]],
    *,
    layout_seed: int,
    support_per_class: int,
    draw: int,
    fold: int,
) -> list[tuple[str, str, int]]:
    if support_per_class not in SUPPORT_PER_CLASS or support_per_class % 2:
        raise ContractError("combined support k must be one of the even prespecified budgets")
    folds = _folds_by_cohort(data, layout_seed)
    per_cohort_class = support_per_class // 2
    support: list[tuple[str, str, int]] = []
    for cohort in ("RIH", "SurGen"):
        frame = data[cohort][MODEL_SEEDS[0]]
        labels = frame["label"].to_numpy(int)
        train = np.flatnonzero(folds[cohort] != fold)
        for label in (0, 1):
            candidates = train[labels[train] == label]
            if len(candidates) < per_cohort_class:
                raise ContractError("exact combined cohort-by-class support is infeasible")
            rng = np.random.default_rng(
                broad.adapter.stable_seed(
                    "lean_combined_support",
                    layout_seed,
                    support_per_class,
                    draw,
                    fold,
                    cohort,
                    label,
                )
            )
            picks = rng.choice(candidates, per_cohort_class, replace=False)
            support.extend(
                (cohort, str(frame.index[index]), int(label)) for index in picks
            )
    if (
        len(support) != 2 * support_per_class
        or len({(cohort, patient) for cohort, patient, _ in support}) != len(support)
        or [label for _cohort, _patient, label in support].count(0) != support_per_class
        or [label for _cohort, _patient, label in support].count(1) != support_per_class
    ):
        raise ContractError("combined support total/class balance drifted")
    for cohort in ("RIH", "SurGen"):
        labels = [label for source, _patient, label in support if source == cohort]
        if (
            labels.count(0) != per_cohort_class
            or labels.count(1) != per_cohort_class
        ):
            raise ContractError("combined support is not exactly cohort-by-class balanced")
    return support


def _solver_calls_per_head(support_per_class: int) -> int:
    total = 2 * support_per_class
    inner_splits = total if total <= broad.adapter.LOO_POOL_MAX else len(FOLDS)
    return inner_splits * len(LAMBDA_GRID) + 1


def _expected_shard_solver_calls(job: ProbeJob) -> int:
    heads = DRAWS_PER_LAYOUT * len(FOLDS) * len(MODEL_SEEDS)
    return heads * _solver_calls_per_head(job.support_per_class)


def _run_probe_shard(
    data: Mapping[str, Mapping[int, pd.DataFrame]], job: ProbeJob
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    embedding_columns = [f"e{index}" for index in range(EMBED_DIM)]
    folds = _folds_by_cohort(data, job.layout_seed)
    reference = {cohort: data[cohort][MODEL_SEEDS[0]] for cohort in ("RIH", "SurGen")}
    ledger = broad.adapter.SolverLedger()
    rows: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []
    for draw in range(DRAWS_PER_LAYOUT):
        predictions = {
            cohort: {
                seed: np.full(len(reference[cohort]), np.nan, dtype=float)
                for seed in MODEL_SEEDS
            }
            for cohort in ("RIH", "SurGen")
        }
        for fold in FOLDS:
            support = _combined_support(
                data,
                layout_seed=job.layout_seed,
                support_per_class=job.support_per_class,
                draw=draw,
                fold=fold,
            )
            support_keys = [f"{cohort}::{patient}" for cohort, patient, _ in support]
            support_labels = [label for _cohort, _patient, label in support]
            test_ids = {
                cohort: [
                    str(reference[cohort].index[index])
                    for index in np.flatnonzero(folds[cohort] == fold)
                ]
                for cohort in ("RIH", "SurGen")
            }
            supports.append(
                {
                    "layout_seed": job.layout_seed,
                    "support_per_class": job.support_per_class,
                    "support_total": 2 * job.support_per_class,
                    "draw": draw,
                    "outer_fold": fold,
                    "support_patient_keys": support_keys,
                    "support_labels": support_labels,
                    "test_patient_ids_by_cohort": test_ids,
                    "shared_across_source_seeds": True,
                    "combined_cohort_by_class_balanced": True,
                }
            )
            for source_seed in MODEL_SEEDS:
                features = np.vstack(
                    [
                        data[cohort][source_seed]
                        .loc[patient, embedding_columns]
                        .to_numpy(float)
                        for cohort, patient, _ in support
                    ]
                )
                labels = np.asarray(support_labels, dtype=int)
                zero_offset = np.zeros(len(labels), dtype=float)
                context = {
                    "phase": "lean_pure_ridge_few_shot",
                    "cohort": "COMBINED",
                    "outer_seed": job.layout_seed,
                    "outer_fold": fold,
                    "model_seed": source_seed,
                    "support": 2 * job.support_per_class,
                    "draw": draw,
                    "method": METHOD,
                }
                selected, selection = broad.adapter.select_lambda(
                    features,
                    zero_offset,
                    labels,
                    inner_seed=job.layout_seed + fold,
                    ledger=ledger,
                    context=context,
                )
                weights, bias, diagnostic = broad.adapter.fit_residual(
                    features,
                    zero_offset,
                    labels,
                    selected,
                    ledger=ledger,
                    context={**context, "fit_role": "outer_refit"},
                )
                for cohort in ("RIH", "SurGen"):
                    test_index = np.flatnonzero(folds[cohort] == fold)
                    test_features = (
                        data[cohort][source_seed]
                        .iloc[test_index][embedding_columns]
                        .to_numpy(float)
                    )
                    predictions[cohort][source_seed][test_index] = (
                        test_features @ weights + bias
                    )
                fits.append(
                    {
                        "phase": "lean_pure_ridge_few_shot",
                        "method": METHOD,
                        "anchor_kind": "zero_logit",
                        "infinite_lambda_semantics": "zero-logit no-information pure probe",
                        "coefficient_order": "e0..e511",
                        "layout_seed": job.layout_seed,
                        "support_per_class": job.support_per_class,
                        "support_total": 2 * job.support_per_class,
                        "draw": draw,
                        "outer_fold": fold,
                        "source_model_seed": source_seed,
                        "fit_patient_keys": support_keys,
                        "fit_labels": support_labels,
                        "test_patient_ids_by_cohort": test_ids,
                        "selected_lambda": selection["selected_lambda"],
                        "inner_selection": selection,
                        "coefficients": weights.tolist(),
                        "bias": bias,
                        "solver_diagnostic": diagnostic,
                    }
                )
        for cohort in ("RIH", "SurGen"):
            if any(
                not np.isfinite(predictions[cohort][seed]).all() for seed in MODEL_SEEDS
            ):
                raise ContractError("probe OOF prediction coverage is incomplete")
            native_by_seed = {
                seed: data[cohort][seed]["eta_native"].to_numpy(float)
                for seed in MODEL_SEEDS
            }
            for index, patient_id in enumerate(reference[cohort].index.astype(str)):
                rows.append(
                    {
                        "phase": "lean_pure_ridge_few_shot",
                        "layout_seed": job.layout_seed,
                        "support_per_class": job.support_per_class,
                        "support_total": 2 * job.support_per_class,
                        "draw": draw,
                        "test_cohort": cohort,
                        "patient_id": patient_id,
                        "label": int(reference[cohort].iloc[index]["label"]),
                        "fold": int(folds[cohort][index]),
                        "eta_native": float(
                            np.mean([native_by_seed[seed][index] for seed in MODEL_SEEDS])
                        ),
                        "eta_pure_ridge": float(
                            np.mean(
                                [predictions[cohort][seed][index] for seed in MODEL_SEEDS]
                            )
                        ),
                        **{
                            f"eta_native_seed{seed}": float(native_by_seed[seed][index])
                            for seed in MODEL_SEEDS
                        },
                        **{
                            f"eta_pure_ridge_seed{seed}": float(
                                predictions[cohort][seed][index]
                            )
                            for seed in MODEL_SEEDS
                        },
                    }
                )
    expected_rows = DRAWS_PER_LAYOUT * 159
    expected_heads = DRAWS_PER_LAYOUT * len(FOLDS) * len(MODEL_SEEDS)
    expected_supports = DRAWS_PER_LAYOUT * len(FOLDS)
    solver = {
        **ledger.summary(),
        "infinite_lambda_semantics": "zero-logit no-information pure probe",
    }
    solver_calls = solver["n_finite_calls"] + solver["n_native_exact_calls"]
    if (
        len(rows) != expected_rows
        or len(fits) != expected_heads
        or len(supports) != expected_supports
        or solver_calls != _expected_shard_solver_calls(job)
        or solver.get("n_unaccepted") != 0
    ):
        raise ContractError(f"{job.key}: probe shard census/solver accounting drifted")
    return pd.DataFrame(rows), fits, supports, solver


def _probe_command(root: Path, job: ProbeJob) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_probe-one",
        "--output-root",
        str(root),
        "--layout-seed",
        str(job.layout_seed),
        "--support-per-class",
        str(job.support_per_class),
    ]


def _shard_payload(
    root: Path, job: ProbeJob, *, created_utc: str, execution: Mapping[str, Any]
) -> dict[str, Any]:
    directory = shard_dir(root, job)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_LEAN_PURE_RIDGE_SHARD",
        "created_utc": created_utc,
        "job": {"layout_seed": job.layout_seed, "support_per_class": job.support_per_class},
        "command": _probe_command(root, job),
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "contract": _artifact(contract_path(root)),
        "target_open_receipt": _artifact(target_open_path(root)),
        "oof": _artifact(directory / "oof.parquet"),
        "fit_ledger": _artifact(directory / "fits.jsonl"),
        "support_ledger": _artifact(directory / "supports.jsonl"),
        "solver": _artifact(directory / "solver.json"),
        "execution_record": _artifact(directory / "execution.json"),
        "execution": dict(execution),
    }


def _validate_selection(record: Mapping[str, Any]) -> None:
    selection = record.get("inner_selection")
    keys = [broad.adapter.lambda_key(value) for value in LAMBDA_GRID]
    if not isinstance(selection, Mapping) or set(selection) != {
        "scheme",
        "inner_seed",
        "n_pool",
        "n_splits",
        "losses_by_lambda",
        "mean_loss_by_lambda",
        "solver_summary_by_lambda",
        "selected_lambda",
    }:
        raise ContractError("probe inner-selection schema drifted")
    total = int(record["support_total"])
    expected_scheme = "leave_one_out" if total <= broad.adapter.LOO_POOL_MAX else "five_fold"
    expected_splits = total if expected_scheme == "leave_one_out" else len(FOLDS)
    losses = selection["losses_by_lambda"]
    means = selection["mean_loss_by_lambda"]
    if (
        selection["scheme"] != expected_scheme
        or selection["n_pool"] != total
        or selection["n_splits"] != expected_splits
        or selection["inner_seed"] != record["layout_seed"] + record["outer_fold"]
        or set(losses) != set(keys)
        or set(means) != set(keys)
        or set(selection["solver_summary_by_lambda"]) != set(keys)
        or any(len(losses[key]) != expected_splits for key in keys)
        or any(
            not math.isclose(float(means[key]), float(np.mean(losses[key])), abs_tol=1e-15)
            for key in keys
        )
    ):
        raise ContractError("probe inner-selection grid/split/loss replay failed")
    best = min(float(means[key]) for key in keys)
    selected = next(key for key in keys if float(means[key]) <= best + 1e-12)
    if record["selected_lambda"] != selected or selection["selected_lambda"] != selected:
        raise ContractError("probe selected lambda is not the first grid minimum")


def _expected_oof_columns() -> list[str]:
    return [
        "phase",
        "layout_seed",
        "support_per_class",
        "support_total",
        "draw",
        "test_cohort",
        "patient_id",
        "label",
        "fold",
        "eta_native",
        "eta_pure_ridge",
        *(f"eta_native_seed{seed}" for seed in MODEL_SEEDS),
        *(f"eta_pure_ridge_seed{seed}" for seed in MODEL_SEEDS),
    ]


def _validate_final_head(
    record: Mapping[str, Any],
    *,
    features: np.ndarray,
    labels: np.ndarray,
) -> None:
    coefficients = np.asarray(record["coefficients"], dtype=float)
    bias = float(record["bias"])
    diagnostic = record["solver_diagnostic"]
    if not isinstance(diagnostic, Mapping):
        raise ContractError("probe final diagnostic is absent")
    if record["selected_lambda"] == "infinity":
        expected_keys = {
            "status",
            "lambda",
            "scipy_success",
            "gradient_inf_norm",
            "objective_at_zero",
            "objective_at_fit",
            "objective_decrease",
            "coefficient_l2",
            "bias",
        }
        objective = float(np.mean(np.logaddexp(0.0, 0.0) - labels * 0.0))
        if (
            set(diagnostic) != expected_keys
            or diagnostic["status"] != "native_exact"
            or diagnostic["lambda"] != "infinity"
            or diagnostic["scipy_success"] is not True
            or diagnostic["gradient_inf_norm"] is not None
            or not np.array_equal(coefficients, np.zeros(EMBED_DIM))
            or bias != 0.0
            or float(diagnostic["coefficient_l2"]) != 0.0
            or float(diagnostic["bias"]) != 0.0
            or float(diagnostic["objective_decrease"]) != 0.0
            or not math.isclose(float(diagnostic["objective_at_zero"]), objective, abs_tol=1e-15)
            or not math.isclose(float(diagnostic["objective_at_fit"]), objective, abs_tol=1e-15)
        ):
            raise ContractError("infinite-lambda pure probe is not the exact zero-logit anchor")
        return
    expected_keys = {
        "status",
        "lambda",
        "scipy_success",
        "scipy_status",
        "scipy_message",
        "iterations",
        "function_evaluations",
        "gradient_inf_norm",
        "objective_at_zero",
        "objective_at_fit",
        "objective_decrease",
        "coefficient_l2",
        "bias",
        "accepted_by",
    }
    lam = float(record["selected_lambda"])
    theta = np.asarray([*coefficients, bias], dtype=float)
    zero = np.zeros_like(theta)
    offset = np.zeros(len(labels), dtype=float)
    value, gradient = broad.adapter.residual_objective_and_gradient(
        theta, features, offset, labels, lam
    )
    value0, _ = broad.adapter.residual_objective_and_gradient(
        zero, features, offset, labels, lam
    )
    gradient_inf = float(np.max(np.abs(gradient)))
    if (
        set(diagnostic) != expected_keys
        or diagnostic["status"] != "finite_optimum"
        or not math.isclose(float(diagnostic["lambda"]), lam, abs_tol=0.0)
        or not math.isclose(float(diagnostic["objective_at_fit"]), value, abs_tol=1e-12)
        or not math.isclose(float(diagnostic["objective_at_zero"]), value0, abs_tol=1e-12)
        or not math.isclose(
            float(diagnostic["objective_decrease"]), value0 - value, abs_tol=1e-12
        )
        or not math.isclose(
            float(diagnostic["gradient_inf_norm"]), gradient_inf, abs_tol=1e-10
        )
        or not math.isclose(
            float(diagnostic["coefficient_l2"]),
            float(np.linalg.norm(coefficients)),
            abs_tol=1e-12,
        )
        or not math.isclose(float(diagnostic["bias"]), bias, abs_tol=0.0)
        or (not diagnostic["scipy_success"] and gradient_inf > broad.adapter.SOLVER_GRAD_TOL)
        or value0 - value < -broad.adapter.OBJECTIVE_TOL
    ):
        raise ContractError("finite pure-ridge optimum does not replay")


def _validate_probe_shard(root: Path, job: ProbeJob) -> dict[str, Any] | None:
    directory = shard_dir(root, job)
    if not directory.exists():
        return None
    required = {
        "oof.parquet",
        "fits.jsonl",
        "supports.jsonl",
        "solver.json",
        "execution.json",
        "completion.json",
    }
    if (
        not directory.is_dir()
        or directory.is_symlink()
        or {path.name for path in directory.iterdir()} != required
        or any(
            not (directory / name).is_file() or (directory / name).is_symlink()
            for name in required
        )
    ):
        raise ContractError(f"{job.key}: partial or noncanonical probe shard")
    execution = _read_json(directory / "execution.json")
    execution_keys = {
        "job_id",
        "started_utc",
        "completed_utc",
        "started_unix_ns",
        "completed_unix_ns",
        "returncode",
        "configured_workers",
    }
    if (
        set(execution) != execution_keys
        or execution["job_id"] != f"pure_ridge.{job.key}"
        or execution["returncode"] != 0
        or execution["configured_workers"] != 1
        or not isinstance(execution["started_unix_ns"], int)
        or not isinstance(execution["completed_unix_ns"], int)
        or execution["completed_unix_ns"] <= execution["started_unix_ns"]
        or _parse_utc(execution["completed_utc"], context=f"{job.key}/completed")
        <= _parse_utc(execution["started_utc"], context=f"{job.key}/started")
    ):
        raise ContractError(f"{job.key}: execution receipt drifted")
    stored = _read_json(directory / "completion.json")
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError(f"{job.key}: completion creation time is absent")
    _parse_utc(created, context=f"{job.key}/completion")
    expected_receipt = _shard_payload(
        root, job, created_utc=created, execution=execution
    )
    if stored != expected_receipt:
        raise ContractError(f"{job.key}: completion receipt does not replay")

    frame = pd.read_parquet(directory / "oof.parquet")
    fits = _read_jsonl(directory / "fits.jsonl")
    supports = _read_jsonl(directory / "supports.jsonl")
    solver = _read_json(directory / "solver.json")
    expected_rows = DRAWS_PER_LAYOUT * 159
    expected_heads = DRAWS_PER_LAYOUT * len(FOLDS) * len(MODEL_SEEDS)
    expected_supports = DRAWS_PER_LAYOUT * len(FOLDS)
    numeric = [name for name in frame if name.startswith("eta_")]
    if (
        list(frame.columns) != _expected_oof_columns()
        or len(frame) != expected_rows
        or frame.duplicated(["draw", "test_cohort", "patient_id"]).any()
        or set(frame["phase"].astype(str)) != {"lean_pure_ridge_few_shot"}
        or set(pd.to_numeric(frame["layout_seed"]).astype(int)) != {job.layout_seed}
        or set(pd.to_numeric(frame["support_per_class"]).astype(int))
        != {job.support_per_class}
        or set(pd.to_numeric(frame["support_total"]).astype(int))
        != {2 * job.support_per_class}
        or set(pd.to_numeric(frame["draw"]).astype(int)) != set(range(DRAWS_PER_LAYOUT))
        or set(frame["test_cohort"].astype(str)) != {"RIH", "SurGen"}
        or not np.isfinite(frame[numeric].to_numpy(float)).all()
        or len(fits) != expected_heads
        or len(supports) != expected_supports
    ):
        raise ContractError(f"{job.key}: OOF/fit/support census or schema drifted")

    data = _patient_data(root)
    folds = _folds_by_cohort(data, job.layout_seed)
    support_keys = {
        "layout_seed",
        "support_per_class",
        "support_total",
        "draw",
        "outer_fold",
        "support_patient_keys",
        "support_labels",
        "test_patient_ids_by_cohort",
        "shared_across_source_seeds",
        "combined_cohort_by_class_balanced",
    }
    support_map: dict[tuple[int, int], Mapping[str, Any]] = {}
    for record in supports:
        if set(record) != support_keys:
            raise ContractError(f"{job.key}: support-ledger schema drifted")
        key = (int(record["draw"]), int(record["outer_fold"]))
        if key in support_map:
            raise ContractError(f"{job.key}: duplicate support procedure/fold")
        support = _combined_support(
            data,
            layout_seed=job.layout_seed,
            support_per_class=job.support_per_class,
            draw=key[0],
            fold=key[1],
        )
        wanted_keys = [f"{cohort}::{patient}" for cohort, patient, _ in support]
        wanted_labels = [label for _cohort, _patient, label in support]
        wanted_tests = {
            cohort: [
                str(data[cohort][MODEL_SEEDS[0]].index[index])
                for index in np.flatnonzero(folds[cohort] == key[1])
            ]
            for cohort in ("RIH", "SurGen")
        }
        if (
            record["layout_seed"] != job.layout_seed
            or record["support_per_class"] != job.support_per_class
            or record["support_total"] != 2 * job.support_per_class
            or key[0] not in range(DRAWS_PER_LAYOUT)
            or key[1] not in FOLDS
            or record["support_patient_keys"] != wanted_keys
            or record["support_labels"] != wanted_labels
            or record["test_patient_ids_by_cohort"] != wanted_tests
            or record["shared_across_source_seeds"] is not True
            or record["combined_cohort_by_class_balanced"] is not True
            or any(
                set(wanted_keys)
                & {f"{cohort}::{patient}" for patient in cohort_patients}
                for cohort, cohort_patients in wanted_tests.items()
            )
        ):
            raise ContractError(f"{job.key}: deterministic support/test replay failed at {key}")
        support_map[key] = record
    if set(support_map) != {
        (draw, fold) for draw in range(DRAWS_PER_LAYOUT) for fold in FOLDS
    }:
        raise ContractError(f"{job.key}: support roster drifted")

    fit_keys = {
        "phase",
        "method",
        "anchor_kind",
        "infinite_lambda_semantics",
        "coefficient_order",
        "layout_seed",
        "support_per_class",
        "support_total",
        "draw",
        "outer_fold",
        "source_model_seed",
        "fit_patient_keys",
        "fit_labels",
        "test_patient_ids_by_cohort",
        "selected_lambda",
        "inner_selection",
        "coefficients",
        "bias",
        "solver_diagnostic",
    }
    fit_map: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    embedding_columns = [f"e{index}" for index in range(EMBED_DIM)]
    for record in fits:
        if set(record) != fit_keys:
            raise ContractError(f"{job.key}: fitted-head schema drifted")
        key = (
            int(record["draw"]),
            int(record["outer_fold"]),
            int(record["source_model_seed"]),
        )
        if key in fit_map:
            raise ContractError(f"{job.key}: duplicate fitted head")
        support = support_map[key[:2]]
        if (
            record["phase"] != "lean_pure_ridge_few_shot"
            or record["method"] != METHOD
            or record["anchor_kind"] != "zero_logit"
            or record["infinite_lambda_semantics"]
            != "zero-logit no-information pure probe"
            or record["coefficient_order"] != "e0..e511"
            or record["layout_seed"] != job.layout_seed
            or record["support_per_class"] != job.support_per_class
            or record["support_total"] != 2 * job.support_per_class
            or key[2] not in MODEL_SEEDS
            or record["fit_patient_keys"] != support["support_patient_keys"]
            or record["fit_labels"] != support["support_labels"]
            or record["test_patient_ids_by_cohort"]
            != support["test_patient_ids_by_cohort"]
            or len(record["coefficients"]) != EMBED_DIM
            or not np.isfinite(
                np.asarray([*record["coefficients"], record["bias"]], dtype=float)
            ).all()
        ):
            raise ContractError(f"{job.key}: fitted-head identity/support drifted at {key}")
        _validate_selection(record)
        parsed = [item.split("::", 1) for item in record["fit_patient_keys"]]
        features = np.vstack(
            [
                data[cohort][key[2]].loc[patient, embedding_columns].to_numpy(float)
                for cohort, patient in parsed
            ]
        )
        _validate_final_head(
            record,
            features=features,
            labels=np.asarray(record["fit_labels"], dtype=int),
        )
        fit_map[key] = record
    if set(fit_map) != {
        (draw, fold, seed)
        for draw in range(DRAWS_PER_LAYOUT)
        for fold in FOLDS
        for seed in MODEL_SEEDS
    }:
        raise ContractError(f"{job.key}: fitted-head roster drifted")

    frame_map = frame.assign(patient_id=frame["patient_id"].astype(str)).set_index(
        ["draw", "test_cohort", "patient_id"]
    )
    for draw in range(DRAWS_PER_LAYOUT):
        for cohort in ("RIH", "SurGen"):
            reference = data[cohort][MODEL_SEEDS[0]]
            for position, patient in enumerate(reference.index.astype(str)):
                row = frame_map.loc[(draw, cohort, patient)]
                label = int(reference.iloc[position]["label"])
                fold = int(folds[cohort][position])
                if int(row["label"]) != label or int(row["fold"]) != fold:
                    raise ContractError(f"{job.key}: OOF patient label/fold drifted")
                natives: list[float] = []
                probes: list[float] = []
                for seed in MODEL_SEEDS:
                    seed_frame = data[cohort][seed]
                    native = float(seed_frame.loc[patient, "eta_native"])
                    features = seed_frame.loc[patient, embedding_columns].to_numpy(float)
                    record = fit_map[(draw, fold, seed)]
                    probe = float(
                        features @ np.asarray(record["coefficients"], dtype=float)
                        + float(record["bias"])
                    )
                    natives.append(native)
                    probes.append(probe)
                    if (
                        not math.isclose(
                            float(row[f"eta_native_seed{seed}"]), native, abs_tol=0.0
                        )
                        or not math.isclose(
                            float(row[f"eta_pure_ridge_seed{seed}"]),
                            probe,
                            rel_tol=0,
                            abs_tol=1e-12,
                        )
                    ):
                        raise ContractError(f"{job.key}: per-seed OOF prediction drifted")
                if (
                    not math.isclose(float(row["eta_native"]), float(np.mean(natives)), abs_tol=1e-15)
                    or not math.isclose(
                        float(row["eta_pure_ridge"]),
                        float(np.mean(probes)),
                        abs_tol=1e-15,
                    )
                ):
                    raise ContractError(f"{job.key}: five-seed ensemble drifted")

    calls = int(solver.get("n_finite_calls", -1)) + int(
        solver.get("n_native_exact_calls", -1)
    )
    if (
        set(solver)
        != {
            "acceptance_rule",
            "n_finite_calls",
            "n_native_exact_calls",
            "n_scipy_success",
            "n_accepted_small_gradient",
            "max_gradient_inf_norm",
            "min_objective_decrease",
            "scipy_status_counts",
            "n_unaccepted",
            "infinite_lambda_semantics",
        }
        or calls != _expected_shard_solver_calls(job)
        or solver["n_unaccepted"] != 0
        or solver["infinite_lambda_semantics"]
        != "zero-logit no-information pure probe"
    ):
        raise ContractError(f"{job.key}: exact solver accounting drifted")
    return stored


def _probe_one(root: Path, job: ProbeJob) -> None:
    output = validate_output_root(root, must_exist=True)
    _validate_target_open(output)
    if _validate_probe_shard(output, job) is not None:
        return
    directory = shard_dir(output, job)
    if directory.exists() or directory.is_symlink():
        raise ContractError(f"{job.key}: partial shard cannot be resumed")
    data = _patient_data(output)
    started_unix_ns = time.time_ns()
    started_utc = _utcnow_precise()
    frame, fits, supports, solver = _run_probe_shard(data, job)
    directory.mkdir(parents=True, exist_ok=False)
    _write_parquet_once(directory / "oof.parquet", frame)
    _write_jsonl_once(directory / "fits.jsonl", fits)
    _write_jsonl_once(directory / "supports.jsonl", supports)
    _write_json_once(directory / "solver.json", solver)
    execution = {
        "job_id": f"pure_ridge.{job.key}",
        "started_utc": started_utc,
        "completed_utc": _utcnow_precise(),
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": time.time_ns(),
        "returncode": 0,
        "configured_workers": 1,
    }
    _write_json_once(directory / "execution.json", execution)
    _write_json_once(
        directory / "completion.json",
        _shard_payload(output, job, created_utc=_utcnow(), execution=execution),
    )
    _validate_probe_shard(output, job)


def _write_or_reconcile_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        if path.is_symlink() or not pd.read_parquet(path).equals(frame):
            raise ContractError(f"immutable parquet drifted: {path}")
    else:
        _write_parquet_once(path, frame)


def _collect_probe_shards(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for job in probe_jobs():
        if _validate_probe_shard(root, job) is None:
            raise ContractError(f"{job.key}: required probe shard is missing")
        frames.append(pd.read_parquet(shard_dir(root, job) / "oof.parquet"))
    aggregate = pd.concat(frames, ignore_index=True).sort_values(
        ["support_per_class", "layout_seed", "draw", "test_cohort", "patient_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if (
        len(aggregate) != 63_600
        or aggregate.duplicated(
            ["support_per_class", "layout_seed", "draw", "test_cohort", "patient_id"]
        ).any()
        or set(pd.to_numeric(aggregate["support_per_class"]).astype(int))
        != set(SUPPORT_PER_CLASS)
    ):
        raise ContractError("aggregate pure-ridge OOF census/key roster drifted")
    return aggregate


def _scheduler_payload(root: Path, *, created_utc: str) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for job in probe_jobs():
        receipt = _validate_probe_shard(root, job)
        if receipt is None:
            raise ContractError(f"{job.key}: shard missing from scheduler replay")
        events.append(
            {
                "job_key": job.key,
                "command": receipt["command"],
                **receipt["execution"],
            }
        )
    events.sort(key=lambda event: str(event["job_key"]))
    try:
        peak = broad._parallel_peak(events)  # noqa: SLF001
    except broad.ContractError as exc:
        raise ContractError(str(exc)) from exc
    if peak != MAX_WORKERS:
        raise ContractError(f"observed probe-process peak was {peak}; required exactly 6")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_LEAN_PURE_RIDGE_SCHEDULER",
        "created_utc": created_utc,
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "configured_max_workers": MAX_WORKERS,
        "configured_child_workers": 1,
        "observed_peak_workers": peak,
        "shard_jobs": len(probe_jobs()),
        "events": events,
        "accounting": accounting(),
        "aggregate_oof": _artifact(aggregate_oof_path(root)),
    }


def _validate_scheduler(root: Path) -> dict[str, Any]:
    stored = _read_json(scheduler_path(root))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("lean scheduler lacks created_utc")
    _parse_utc(created, context="lean scheduler")
    expected = _scheduler_payload(root, created_utc=created)
    if stored != expected:
        raise ContractError("lean scheduler receipt does not exactly replay")
    return stored


def run_probes(root: Path, *, apply: bool, max_workers: int) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    _validate_target_open(output)
    if max_workers != MAX_WORKERS:
        raise ContractError("governed lean adaptation requires exactly --max-workers 6")
    commands = [_probe_command(output, job) for job in probe_jobs()]
    if not apply:
        return {
            "status": "DRY_RUN_LEAN_COMBINED_PURE_RIDGE",
            "shards": len(probe_jobs()),
            "configured_max_workers": MAX_WORKERS,
            "accounting": accounting(),
            "commands": commands,
        }

    def run(command: Sequence[str]) -> int:
        environment = dict(os.environ)
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        return int(
            subprocess.run(
                list(command), cwd=REPO, env=environment, check=False
            ).returncode
        )

    pending = [
        command
        for job, command in zip(probe_jobs(), commands, strict=True)
        if _validate_probe_shard(output, job) is None
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run, command) for command in pending]
        returncodes = [future.result() for future in concurrent.futures.as_completed(futures)]
    if any(returncodes):
        raise ContractError("one or more deterministic pure-ridge shards failed")
    aggregate = _collect_probe_shards(output)
    _write_or_reconcile_parquet(aggregate_oof_path(output), aggregate)
    if scheduler_path(output).exists():
        _validate_scheduler(output)
    else:
        _write_json_once(
            scheduler_path(output),
            _scheduler_payload(output, created_utc=_utcnow()),
        )
    return _validate_scheduler(output)


def _auroc_rows(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.asarray(broad.e2c_stats._auroc_rows(labels, scores), dtype=float)  # noqa: SLF001


def _procedure_matrices(
    frame: pd.DataFrame,
) -> tuple[
    list[str],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
]:
    working = frame.copy()
    working["patient_id"] = working["patient_id"].astype(str)
    working["_procedure"] = (
        working[["layout_seed", "draw"]].astype(str).agg(":".join, axis=1)
    )
    procedures = sorted(working["_procedure"].unique())
    if len(procedures) != PROCEDURES_PER_BUDGET:
        raise ContractError("procedure matrix does not contain exactly 100 procedures")
    labels: dict[str, np.ndarray] = {}
    matrices: dict[str, dict[str, np.ndarray]] = {}
    patients_ref: list[str] = []
    for cohort, expected_patients, expected_mutant in (
        ("RIH", 85, 37),
        ("SurGen", 74, 30),
    ):
        block = working.loc[working["test_cohort"].astype(str).eq(cohort)]
        patients = sorted(block["patient_id"].unique())
        if cohort == "RIH":
            patients_ref = patients
        label_table = block[["patient_id", "label"]].drop_duplicates()
        if label_table["patient_id"].duplicated().any():
            raise ContractError("patient label changed across support procedures")
        y = label_table.set_index("patient_id").loc[patients, "label"].to_numpy(int)
        labels[cohort] = y
        matrices[cohort] = {}
        for method, column in {
            "native": "eta_native",
            METHOD: "eta_pure_ridge",
        }.items():
            matrix = (
                block.pivot(index="_procedure", columns="patient_id", values=column)
                .loc[procedures, patients]
                .to_numpy(float)
            )
            if matrix.shape != (PROCEDURES_PER_BUDGET, expected_patients) or not np.isfinite(
                matrix
            ).all():
                raise ContractError("procedure prediction matrix is incomplete/nonfinite")
            matrices[cohort][method] = matrix
        if len(y) != expected_patients or int(y.sum()) != expected_mutant:
            raise ContractError(f"{cohort}: procedure matrix census drifted")
        if not np.allclose(
            matrices[cohort]["native"],
            matrices[cohort]["native"][:1],
            rtol=0,
            atol=0,
        ):
            raise ContractError("frozen native scores changed across procedures")
        native = float(roc_auc_score(y, matrices[cohort]["native"][0]))
        if not math.isclose(native, EXPECTED_NATIVE[cohort]["auroc"], abs_tol=1e-15):
            raise ContractError(f"{cohort}: native point authority drifted")
    if set(patients_ref) & set(
        working.loc[working["test_cohort"].eq("SurGen"), "patient_id"].astype(str)
    ):
        raise ContractError("cohort patient IDs overlap in procedure matrix")
    return procedures, labels, matrices


def _summarize_budget(
    frame: pd.DataFrame,
    *,
    support_per_class: int,
    bootstrap_draws: int | None = None,
) -> dict[str, Any]:
    if support_per_class not in SUPPORT_PER_CLASS:
        raise ContractError("summary requested an unregistered support budget")
    expected_procedures = {
        (layout, draw)
        for layout in LAYOUT_SEEDS
        for draw in range(DRAWS_PER_LAYOUT)
    }
    observed_procedures = set(
        zip(
            pd.to_numeric(frame["layout_seed"]).astype(int),
            pd.to_numeric(frame["draw"]).astype(int),
            strict=True,
        )
    )
    if (
        len(frame) != PROCEDURES_PER_BUDGET * 159
        or set(pd.to_numeric(frame["support_per_class"]).astype(int))
        != {support_per_class}
        or set(pd.to_numeric(frame["support_total"]).astype(int))
        != {2 * support_per_class}
        or observed_procedures != expected_procedures
        or frame.duplicated(["layout_seed", "draw", "test_cohort", "patient_id"]).any()
    ):
        raise ContractError("budget summary input cell is not the exact 100x159 roster")
    n_bootstrap = N_BOOTSTRAP if bootstrap_draws is None else bootstrap_draws
    if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ContractError("bootstrap draw count must be positive")
    procedures, labels, matrices = _procedure_matrices(frame)
    methods = ("native", METHOD)
    per_procedure: dict[str, dict[str, np.ndarray]] = {
        scope: {} for scope in ("RIH-M", "SurGen-M", "pooled_combined", "equal_cohort_macro")
    }
    for method in methods:
        rih = _auroc_rows(labels["RIH"], matrices["RIH"][method])
        surgen = _auroc_rows(labels["SurGen"], matrices["SurGen"][method])
        pooled = _auroc_rows(
            np.concatenate([labels["RIH"], labels["SurGen"]]),
            np.concatenate([matrices["RIH"][method], matrices["SurGen"][method]], axis=1),
        )
        per_procedure["RIH-M"][method] = rih
        per_procedure["SurGen-M"][method] = surgen
        per_procedure["pooled_combined"][method] = pooled
        per_procedure["equal_cohort_macro"][method] = (rih + surgen) / 2.0
    if (
        not math.isclose(
            float(np.mean(per_procedure["pooled_combined"]["native"])),
            EXPECTED_NATIVE_POOLED,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(np.mean(per_procedure["equal_cohort_macro"]["native"])),
            EXPECTED_NATIVE_EQUAL_MACRO,
            abs_tol=1e-15,
        )
    ):
        raise ContractError("pooled/equal-macro frozen native authority drifted")
    scopes = tuple(per_procedure)
    draws: dict[str, dict[str, list[float]]] = {
        scope: {"native": [], METHOD: [], "gain": []} for scope in scopes
    }
    rng = np.random.default_rng(
        broad.adapter.stable_seed("lean_pure_ridge_bootstrap", support_per_class, BOOTSTRAP_SEED)
    )
    for _ in range(n_bootstrap):
        indices = {
            cohort: broad.e2c_stats._bootstrap_patient_indices(y, rng)  # noqa: SLF001
            for cohort, y in labels.items()
        }
        procedure_index = rng.integers(0, len(procedures), len(procedures))
        current: dict[str, dict[str, float]] = {scope: {} for scope in scopes}
        for method in methods:
            cohort_rows: dict[str, np.ndarray] = {}
            for cohort, scope in (("RIH", "RIH-M"), ("SurGen", "SurGen-M")):
                index = indices[cohort]
                cohort_rows[cohort] = _auroc_rows(
                    labels[cohort][index], matrices[cohort][method][:, index]
                )
                current[scope][method] = float(np.mean(cohort_rows[cohort][procedure_index]))
            pooled_y = np.concatenate(
                [labels["RIH"][indices["RIH"]], labels["SurGen"][indices["SurGen"]]]
            )
            pooled_eta = np.concatenate(
                [
                    matrices["RIH"][method][:, indices["RIH"]],
                    matrices["SurGen"][method][:, indices["SurGen"]],
                ],
                axis=1,
            )
            pooled_rows = _auroc_rows(pooled_y, pooled_eta)
            current["pooled_combined"][method] = float(
                np.mean(pooled_rows[procedure_index])
            )
            current["equal_cohort_macro"][method] = float(
                np.mean(
                    (
                        cohort_rows["RIH"][procedure_index]
                        + cohort_rows["SurGen"][procedure_index]
                    )
                    / 2.0
                )
            )
        for scope in scopes:
            for method in methods:
                draws[scope][method].append(current[scope][method])
            draws[scope]["gain"].append(
                current[scope][METHOD] - current[scope]["native"]
            )

    def interval(values: Sequence[float]) -> list[float]:
        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]

    census = {
        "RIH-M": {"patients": 85, "mutant": 37},
        "SurGen-M": {"patients": 74, "mutant": 30},
        "pooled_combined": {"patients": 159, "mutant": 67},
        "equal_cohort_macro": {
            "patients": 159,
            "mutant": 67,
            "weighting": "50% RIH-M + 50% SurGen-M",
        },
    }
    summary_scopes: dict[str, Any] = {}
    for scope in scopes:
        native_point = float(np.mean(per_procedure[scope]["native"]))
        probe_point = float(np.mean(per_procedure[scope][METHOD]))
        summary_scopes[scope] = {
            "census": census[scope],
            "native": {
                "auroc": native_point,
                "ci95": interval(draws[scope]["native"]),
                "procedure_sample_sd": float(
                    np.std(per_procedure[scope]["native"], ddof=1)
                ),
            },
            METHOD: {
                "auroc": probe_point,
                "ci95": interval(draws[scope][METHOD]),
                "procedure_sample_sd": float(
                    np.std(per_procedure[scope][METHOD], ddof=1)
                ),
            },
            f"{METHOD}_minus_native": {
                "auroc_gain": probe_point - native_point,
                "ci95": interval(draws[scope]["gain"]),
            },
        }

    seed_descriptive: dict[str, Any] = {}
    working = frame.copy()
    working["patient_id"] = working["patient_id"].astype(str)
    working["_procedure"] = (
        working[["layout_seed", "draw"]].astype(str).agg(":".join, axis=1)
    )
    for method, template in {
        "native": "eta_native_seed{seed}",
        METHOD: "eta_pure_ridge_seed{seed}",
    }.items():
        by_scope: dict[str, dict[str, float]] = {scope: {} for scope in scopes}
        for seed in MODEL_SEEDS:
            cohort_values: dict[str, np.ndarray] = {}
            for cohort, scope in (("RIH", "RIH-M"), ("SurGen", "SurGen-M")):
                block = working.loc[working["test_cohort"].eq(cohort)]
                patients = sorted(block["patient_id"].unique())
                matrix = (
                    block.pivot(
                        index="_procedure",
                        columns="patient_id",
                        values=template.format(seed=seed),
                    )
                    .loc[procedures, patients]
                    .to_numpy(float)
                )
                cohort_values[cohort] = _auroc_rows(labels[cohort], matrix)
                by_scope[scope][str(seed)] = float(np.mean(cohort_values[cohort]))
            pooled_matrix = np.concatenate(
                [
                    working.loc[working["test_cohort"].eq(cohort)]
                    .pivot(
                        index="_procedure",
                        columns="patient_id",
                        values=template.format(seed=seed),
                    )
                    .loc[procedures, sorted(
                        working.loc[working["test_cohort"].eq(cohort), "patient_id"].unique()
                    )]
                    .to_numpy(float)
                    for cohort in ("RIH", "SurGen")
                ],
                axis=1,
            )
            by_scope["pooled_combined"][str(seed)] = float(
                np.mean(
                    _auroc_rows(
                        np.concatenate([labels["RIH"], labels["SurGen"]]),
                        pooled_matrix,
                    )
                )
            )
            by_scope["equal_cohort_macro"][str(seed)] = float(
                np.mean((cohort_values["RIH"] + cohort_values["SurGen"]) / 2.0)
            )
        seed_descriptive[method] = {
            scope: {
                "by_seed": values,
                "mean": float(np.mean(list(values.values()))),
                "sample_sd": float(np.std(list(values.values()), ddof=1)),
            }
            for scope, values in by_scope.items()
        }
    for scope, expected in EXPECTED_NATIVE_PER_SEED.items():
        observed = seed_descriptive["native"][scope]["by_seed"]
        if set(observed) != set(expected) or any(
            not math.isclose(observed[seed], value, abs_tol=1e-15)
            for seed, value in expected.items()
        ):
            raise ContractError(f"{scope}: per-seed frozen native authority drifted")
    return {
        "support_per_class_total": support_per_class,
        "support_total": 2 * support_per_class,
        "combined_support_balance": "k/2 per cohort within each class",
        "procedures": PROCEDURES_PER_BUDGET,
        "estimand": (
            "mean of 100 per-procedure AUROCs; support-procedure predictions are never "
            "averaged before AUROC"
        ),
        "bootstrap_draws": n_bootstrap,
        "bootstrap_pairing": (
            "patients stratified-resampled within cohort; procedure draw shared across "
            "method, cohorts, pooled, and macro"
        ),
        "scopes": summary_scopes,
        "per_source_seed_expected_auroc": seed_descriptive,
    }


def _build_results(root: Path, *, bootstrap_draws: int | None = None) -> dict[str, Any]:
    _validate_scheduler(root)
    aggregate = pd.read_parquet(aggregate_oof_path(root))
    if len(aggregate) != 63_600 or list(aggregate.columns) != _expected_oof_columns():
        raise ContractError("terminal aggregate OOF schema/census drifted")
    cells: dict[str, Any] = {}
    for support in SUPPORT_PER_CLASS:
        block = aggregate.loc[
            pd.to_numeric(aggregate["support_per_class"]).astype(int).eq(support)
        ].copy()
        if len(block) != PROCEDURES_PER_BUDGET * 159:
            raise ContractError(f"k={support}: aggregate procedure rows drifted")
        cells[str(support)] = _summarize_budget(
            block,
            support_per_class=support,
            bootstrap_draws=bootstrap_draws,
        )
    first = cells[str(SUPPORT_PER_CLASS[0])]
    native_scopes = {
        scope: {
            "census": values["census"],
            "auroc": values["native"]["auroc"],
            "ci95": values["native"]["ci95"],
            "ci_role": (
                "campaign paired patient bootstrap; zero-shot point was sealed before "
                "target outcomes opened"
            ),
        }
        for scope, values in first["scopes"].items()
    }
    if (
        native_scopes["pooled_combined"]["census"]
        != {"patients": 159, "mutant": 67}
        or not math.isclose(
            native_scopes["equal_cohort_macro"]["auroc"],
            float(
                np.mean(
                    [
                        EXPECTED_NATIVE["RIH"]["auroc"],
                        EXPECTED_NATIVE["SurGen"]["auroc"],
                    ]
                )
            ),
            abs_tol=1e-15,
        )
    ):
        raise ContractError("native pooled/macro reporting gate drifted")
    draws = N_BOOTSTRAP if bootstrap_draws is None else bootstrap_draws
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "COMPLETE_LEAN_COMBINED_MET_PURE_RIDGE_FEWSHOT",
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "source_anchor": {
            "development_population": "TCGA+SurGen primaries only",
            "encoder": "UNI-v1",
            "model": "frozen p75 refit",
            "model_seeds": list(MODEL_SEEDS),
            "encoder_mil_source_classifier_and_native_logits_frozen": True,
            "source_or_main_model_fits": 0,
        },
        "native_zero_shot": {
            "role": (
                "frozen zero-shot comparator inside a TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION "
                "analysis; SurGen-M is source-family-exposed"
            ),
            "estimand": "five-seed mean-logit ensemble, one score per target patient",
            "scopes": native_scopes,
        },
        "few_shot": {
            "methods": [METHOD],
            "support_regimes": ["COMBINED"],
            "support_per_class_total": list(SUPPORT_PER_CLASS),
            "all_budgets_reported": True,
            "method_regime_or_budget_selected_on_target_results": False,
            "positive_repair_claim_permitted": False,
            "test_scopes": [
                "RIH-M",
                "SurGen-M",
                "pooled_combined",
                "equal_cohort_macro",
            ],
            "cells": cells,
        },
        "bootstrap": {
            "draws": draws,
            "paired_gain": True,
            "patient_resampling": "stratified within each metastatic cohort",
            "procedure_resampling": "paired across native/probe and all report scopes",
        },
        "fit_and_solver_accounting": accounting(),
        "firewall": {
            "primary_support_used": False,
            "target_labels_used_only_after_native_scores_and_embeddings_sealed": True,
            "allowed_target_labels": ["RIH-M", "SurGen-M"],
            "combined_support_only": True,
            "source_checkpoint_encoder_mil_or_classifier_modified": False,
            "external_validation_claim_permitted": False,
            "surgen_metastatic_source_family_exposed": True,
            "adaptation_feedback_to_source_selection_or_external_claims": False,
            "residual_local_mil_full_label_or_platt_present": False,
        },
    }


def _completion_payload(root: Path, *, created_utc: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_LEAN_COMBINED_MET_PURE_RIDGE_FEWSHOT",
        "created_utc": created_utc,
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "contract": _artifact(contract_path(root)),
        "source_inference_seal": _artifact(broad.inference_seal_path(root)),
        "target_open_receipt": _artifact(target_open_path(root)),
        "scheduler": _artifact(scheduler_path(root)),
        "aggregate_oof": _artifact(aggregate_oof_path(root)),
        "results": _artifact(results_path(root)),
        "accounting": accounting(),
    }


def _validate_completion(root: Path) -> dict[str, Any]:
    _validate_scheduler(root)
    stored_results = _read_json(results_path(root))
    expected_results = _build_results(root)
    if stored_results != expected_results:
        raise ContractError("terminal lean results do not numerically replay")
    stored = _read_json(completion_path(root))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("terminal lean completion lacks created_utc")
    _parse_utc(created, context="terminal lean completion")
    if stored != _completion_payload(root, created_utc=created):
        raise ContractError("terminal lean completion receipt does not replay")
    return stored_results


def analyze_campaign(root: Path, *, apply: bool) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    _validate_scheduler(output)
    if not apply:
        return {
            "status": "DRY_RUN_LEAN_ANALYSIS",
            "bootstrap_draws": N_BOOTSTRAP,
            "budgets": list(SUPPORT_PER_CLASS),
            "all_budgets_reported": True,
        }
    results = _build_results(output)
    if results_path(output).exists():
        if _read_json(results_path(output)) != results:
            raise ContractError("persisted immutable lean results drifted")
    else:
        _write_json_once(results_path(output), results)
    if completion_path(output).exists():
        stored = _read_json(completion_path(output))
        created = stored.get("created_utc")
        if not isinstance(created, str) or stored != _completion_payload(
            output, created_utc=created
        ):
            raise ContractError("persisted lean completion drifted")
    else:
        _write_json_once(
            completion_path(output),
            _completion_payload(output, created_utc=_utcnow()),
        )
    return _validate_completion(output)


def campaign_plan(root: Path) -> dict[str, Any]:
    output = validate_output_root(root)
    return {
        "campaign": CAMPAIGN,
        "output_root": str(output),
        "source_root_read_only": str(SOURCE_ROOT),
        "result_role": INTERNAL_ROLE,
        "production_sequence": [
            "prepare --apply",
            "preflight --apply",
            "score-source --apply --max-workers 6 --num-workers 4",
            "open-targets --apply",
            "adapt --apply --max-workers 6",
            "analyze --apply",
            "verify",
        ],
        "method": METHOD,
        "support_regime": "COMBINED",
        "support_per_class_total": list(SUPPORT_PER_CLASS),
        "shards": len(probe_jobs()),
        "max_workers": MAX_WORKERS,
        "accounting": accounting(),
        "excluded": [
            "residual adapter",
            "local MIL",
            "full-label adapter",
            "Platt calibration",
        ],
        "production_launch_authorized_by_this_command": False,
    }


def campaign_status(root: Path) -> dict[str, Any]:
    output = validate_output_root(root)
    stages = {
        "prepared": contract_path(output),
        "preflight": output / "receipts/deep_preflight.json",
        "source_inference_sealed": broad.inference_seal_path(output),
        "target_internal_open": target_open_path(output),
        "pure_ridge_scheduler_complete": scheduler_path(output),
        "analysis_complete": completion_path(output),
    }
    return {
        "output_root": str(output),
        "exists": output.is_dir(),
        "stages": {
            name: path.is_file() and not path.is_symlink() for name, path in stages.items()
        },
        "production_root_absent": not output.exists(),
    }


def verify_campaign(root: Path) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    load_contract(output, deep_pack=True)
    verify_preflight(output, deep_pack=False)
    verify_inference_seal(output)
    _validate_target_open(output)
    _validate_completion(output)
    return {
        "status": "VERIFIED_COMPLETE_LEAN_PURE_RIDGE",
        "campaign": CAMPAIGN,
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "results": _artifact(results_path(output)),
    }


def _print_json(value: Any) -> None:
    print(json.dumps(broad.adapter.json_ready(value), indent=2, sort_keys=True, allow_nan=False))


def _add_output_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "plan",
        "prepare",
        "preflight",
        "score-source",
        "open-targets",
        "adapt",
        "analyze",
        "verify",
        "status",
    ):
        command = subparsers.add_parser(name)
        _add_output_root(command)
        if name in {
            "prepare",
            "preflight",
            "score-source",
            "open-targets",
            "adapt",
            "analyze",
        }:
            command.add_argument("--apply", action="store_true")
        if name in {"score-source", "adapt"}:
            command.add_argument("--max-workers", type=int, default=MAX_WORKERS)
        if name == "score-source":
            command.add_argument(
                "--num-workers", type=int, default=DEFAULT_NUM_WORKERS
            )
    score_one = subparsers.add_parser("_score-one")
    _add_output_root(score_one)
    score_one.add_argument("--target", choices=TARGET_ORDER, required=True)
    score_one.add_argument("--seed", type=int, choices=MODEL_SEEDS, required=True)
    score_one.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    probe_one = subparsers.add_parser("_probe-one")
    _add_output_root(probe_one)
    probe_one.add_argument("--layout-seed", type=int, choices=LAYOUT_SEEDS, required=True)
    probe_one.add_argument(
        "--support-per-class", type=int, choices=SUPPORT_PER_CLASS, required=True
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.output_root
    if args.command == "plan":
        value = campaign_plan(root)
    elif args.command == "status":
        value = campaign_status(root)
    elif args.command == "prepare":
        value = prepare(root, apply=args.apply)
    elif args.command == "preflight":
        value = preflight(root, apply=args.apply, deep_pack=True)
    elif args.command == "score-source":
        value = score_source(
            root,
            apply=args.apply,
            max_workers=args.max_workers,
            num_workers=args.num_workers,
        )
    elif args.command == "open-targets":
        value = open_targets(root, apply=args.apply)
    elif args.command == "adapt":
        value = run_probes(root, apply=args.apply, max_workers=args.max_workers)
    elif args.command == "analyze":
        value = analyze_campaign(root, apply=args.apply)
    elif args.command == "verify":
        value = verify_campaign(root)
    elif args.command == "_score-one":
        with _broad_context():
            broad._score_one(  # noqa: SLF001
                root,
                broad.ScoreJob(args.target, args.seed),
                num_workers=args.num_workers,
            )
        value = {"status": "COMPLETE", "job": f"{args.target}__seed{args.seed}"}
    elif args.command == "_probe-one":
        job = ProbeJob(args.layout_seed, args.support_per_class)
        _probe_one(root, job)
        value = {"status": "COMPLETE", "job": job.key}
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _print_json(value)


if __name__ == "__main__":
    main()
