#!/usr/bin/env python3
"""Governed combined-met source-anchored residual-head few-shot campaign.

This immutable sibling reuses the already sealed label-blind embeddings,
native logits, and target-open artifacts from the governed pure-ridge
campaign.  It never rescales or refits the UNI-v1 encoder, MIL model, or
source classifier.  The only fitted target-internal object is

    eta_adapt = eta_native + H @ w + b

where ``H`` is the frozen 512-dimensional patient-mean MIL embedding.  The
residual head minimizes mean binary logistic loss plus
``lambda/2 * (||w||^2 + b^2)``; the bias is penalized.  Infinity is the exact
native/no-correction anchor.  Supports contain k=2/4/8/16 patients per class
TOTAL, exactly balanced k/2 per cohort within each class, and are shared
across the five frozen source seeds.  Lambda selection uses support outcomes
only, and all reported predictions are honest outer-fold predictions.

Every target-label result is ``TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION``.
SurGen-M is source-family-exposed.  No target result may feed back to source
model, method, or budget selection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import aim2_tcga_surgen_pure_ridge_combined_fewshot_campaign as pure  # noqa: E402

broad = pure.broad
SCHEMA_VERSION = 1
CAMPAIGN = "aim2_tcga_surgen_residual_combined_fewshot_univ1_5seed"
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_tcga_surgen_residual_combined_fewshot_univ1_5seed_v1_20260828"
)
UPSTREAM_ROOT = pure.DEFAULT_OUTPUT_ROOT
RERUNS_ROOT = pure.RERUNS_ROOT
SOURCE_ROOT = pure.SOURCE_ROOT
MODEL_SEEDS = pure.MODEL_SEEDS
FOLDS = pure.FOLDS
LAYOUT_SEEDS = pure.LAYOUT_SEEDS
SUPPORT_PER_CLASS = pure.SUPPORT_PER_CLASS
DRAWS_PER_LAYOUT = pure.DRAWS_PER_LAYOUT
PROCEDURES_PER_BUDGET = pure.PROCEDURES_PER_BUDGET
LAMBDA_GRID = pure.LAMBDA_GRID
N_BOOTSTRAP = pure.N_BOOTSTRAP
BOOTSTRAP_SEED = pure.BOOTSTRAP_SEED
MAX_WORKERS = 6
EMBED_DIM = pure.EMBED_DIM
INTERNAL_ROLE = pure.INTERNAL_ROLE
METHOD = "source_anchored_residual_ridge"
PHASE = "lean_source_anchored_residual_few_shot"
EXPECTED_NATIVE = pure.EXPECTED_NATIVE
EXPECTED_NATIVE_POOLED = pure.EXPECTED_NATIVE_POOLED
EXPECTED_NATIVE_EQUAL_MACRO = pure.EXPECTED_NATIVE_EQUAL_MACRO
EXPECTED_NATIVE_PER_SEED = pure.EXPECTED_NATIVE_PER_SEED
TARGETS = pure.TARGETS
TARGET_ORDER = pure.TARGET_ORDER


class ContractError(RuntimeError):
    """Fail-closed residual-campaign contract violation."""


@dataclass(frozen=True, order=True)
class ResidualJob:
    layout_seed: int
    support_per_class: int

    def __post_init__(self) -> None:
        if self.layout_seed not in LAYOUT_SEEDS or self.support_per_class not in SUPPORT_PER_CLASS:
            raise ValueError(self)

    @property
    def key(self) -> str:
        return f"layout{self.layout_seed}__k{self.support_per_class}"


def residual_jobs() -> list[ResidualJob]:
    return [
        ResidualJob(layout, support) for layout in LAYOUT_SEEDS for support in SUPPORT_PER_CLASS
    ]


def accounting() -> dict[str, int]:
    return {
        "source_main_model_fits": 0,
        "new_label_blind_embedding_jobs": 0,
        "reused_sealed_embedding_artifacts": 10,
        "unique_support_procedures_by_fold": 2_000,
        "final_residual_head_decisions": 10_000,
        "fit_to_test_cohort_applications": 20_000,
        "inner_plus_final_solver_calls": 1_330_000,
        "residual_adapter_fits": 10_000,
        "pure_probe_fits": 0,
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
    for forbidden, label in (
        (SOURCE_ROOT.resolve(strict=False), "source/main-model"),
        (UPSTREAM_ROOT.resolve(strict=False), "sealed upstream"),
    ):
        if (
            resolved == forbidden
            or _is_relative_to(resolved, forbidden)
            or _is_relative_to(forbidden, resolved)
        ):
            raise ContractError(f"residual output must never overlap {label} root")
    production = DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    if resolved != production and not _is_relative_to(resolved, Path("/tmp").resolve()):
        raise ContractError(f"production root must be exactly {production}; tests may use /tmp")
    if resolved == production and not _is_relative_to(resolved, RERUNS_ROOT.resolve(strict=False)):
        raise ContractError("production root escaped governed reruns")
    if must_exist is True and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if must_exist is False and (resolved.exists() or resolved.is_symlink()):
        raise FileExistsError(resolved)
    return resolved


def _artifact(path: Path) -> dict[str, Any]:
    try:
        return pure._artifact(path)  # noqa: SLF001
    except (pure.ContractError, broad.ContractError) as exc:
        raise ContractError(str(exc)) from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return pure._read_json(path)  # noqa: SLF001
    except (pure.ContractError, broad.ContractError) as exc:
        raise ContractError(str(exc)) from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return pure._read_jsonl(path)  # noqa: SLF001
    except (pure.ContractError, broad.ContractError) as exc:
        raise ContractError(str(exc)) from exc


def _write_json_once(path: Path, value: Any) -> None:
    try:
        pure._write_json_once(path, value)  # noqa: SLF001
    except (pure.ContractError, broad.ContractError) as exc:
        raise ContractError(str(exc)) from exc


def _write_jsonl_once(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    try:
        pure._write_jsonl_once(path, rows)  # noqa: SLF001
    except (pure.ContractError, broad.ContractError) as exc:
        raise ContractError(str(exc)) from exc


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    try:
        pure._write_parquet_once(path, frame)  # noqa: SLF001
    except (pure.ContractError, broad.ContractError) as exc:
        raise ContractError(str(exc)) from exc


def contract_path(root: Path) -> Path:
    return root / "contract.json"


def preflight_path(root: Path) -> Path:
    return root / "receipts/residual_preflight.json"


def shard_dir(root: Path, job: ResidualJob) -> Path:
    return root / f"adaptation/shards/{job.key}"


def scheduler_path(root: Path) -> Path:
    return root / "receipts/residual_scheduler.json"


def aggregate_oof_path(root: Path) -> Path:
    return root / "adaptation/residual_ridge_oof.parquet"


def results_path(root: Path) -> Path:
    return root / "analysis/results.json"


def completion_path(root: Path) -> Path:
    return root / "analysis/completion.json"


def _implementation_sources() -> dict[str, Any]:
    paths = {
        "residual_controller": Path(__file__).resolve(),
        "residual_test": REPO / "tests/test_aim2_tcga_surgen_residual_combined_fewshot_campaign.py",
        "sealed_upstream_controller": Path(pure.__file__).resolve(),
        "sealed_upstream_test": REPO
        / "tests/test_aim2_tcga_surgen_pure_ridge_combined_fewshot_campaign.py",
        "source_campaign_controller": Path(broad.__file__).resolve(),
        "ridge_solver": REPO / "aim2_v3_fulllabel_residual_adaptation.py",
    }
    return {name: _artifact(path) for name, path in paths.items()}


def _validate_upstream_authority() -> None:
    try:
        pure.load_contract(UPSTREAM_ROOT, deep_pack=False)
        pure.verify_preflight(UPSTREAM_ROOT, deep_pack=False)
        pure.verify_inference_seal(UPSTREAM_ROOT)
        pure._validate_target_open(UPSTREAM_ROOT)  # noqa: SLF001
    except (pure.ContractError, broad.ContractError) as exc:
        raise ContractError(f"sealed upstream authority failed: {exc}") from exc


def _upstream_artifacts() -> dict[str, Any]:
    _validate_upstream_authority()
    paths: dict[str, Path] = {
        "contract": pure.contract_path(UPSTREAM_ROOT),
        "deep_preflight": UPSTREAM_ROOT / "receipts/deep_preflight.json",
        "source_embedding_scheduler": broad.source_scheduler_path(UPSTREAM_ROOT),
        "inference_seal": broad.inference_seal_path(UPSTREAM_ROOT),
        "target_open": pure.target_open_path(UPSTREAM_ROOT),
        "labeled_target_manifest": pure.target_manifest_path(UPSTREAM_ROOT),
        "source_environment": UPSTREAM_ROOT / "source_inference/environment.json",
    }
    for target in TARGET_ORDER:
        for seed in MODEL_SEEDS:
            job = broad.ScoreJob(target, seed)
            paths[f"embedding__{job.key}"] = broad.score_path(UPSTREAM_ROOT, job)
            paths[f"embedding_receipt__{job.key}"] = broad.score_receipt_path(UPSTREAM_ROOT, job)
    return {name: _artifact(path) for name, path in paths.items()}


def _contract_payload(
    root: Path,
    *,
    created_utc: str,
    upstream_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "PREPARED_SEALED_UPSTREAM_RESIDUAL_HEAD",
        "created_utc": created_utc,
        "output_root": str(root),
        "upstream": {
            "root": str(UPSTREAM_ROOT),
            "reuse_mode": "direct read-only artifact binding; no rescoring or copying",
            "artifacts": dict(upstream_artifacts),
            "development_population": "TCGA+SurGen primaries only",
            "encoder": "UNI-v1",
            "source_model": "frozen p75 refit",
            "source_model_seeds": list(MODEL_SEEDS),
        },
        "target_census": {"slides": 185, "patients": 159, "mutant": 67},
        "protocol": {
            "method": METHOD,
            "formula": "eta_adapt = eta_native + H @ w + b",
            "representation": (
                "raw frozen 512-dimensional patient-mean MIL embedding; no target-wide scaling"
            ),
            "objective": "mean binary logistic loss + lambda/2*(||w||^2+b^2)",
            "bias_penalized": True,
            "native_offset_frozen": True,
            "infinity_semantics": "exact frozen native logit; zero residual correction",
            "source_models_frozen": True,
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
            "test_scopes": [
                "RIH-M",
                "SurGen-M",
                "pooled_combined",
                "equal_cohort_macro",
            ],
            "bootstrap_draws": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "maximum_parallel_cpu_shards": MAX_WORKERS,
        },
        "accounting": accounting(),
        "firewall": {
            "result_role": INTERNAL_ROLE,
            "external_validation_claim_permitted": False,
            "surgen_metastatic_source_family_exposed": True,
            "primary_support_permitted": False,
            "allowed_target_labels": ["RIH-M", "SurGen-M"],
            "combined_support_only": True,
            "all_budgets_reported": True,
            "method_or_budget_selected_on_target_results": False,
            "target_feedback_to_source_model_permitted": False,
            "pure_probe_permitted": False,
            "residual_head_only": True,
            "local_mil_permitted": False,
            "full_label_or_platt_permitted": False,
        },
        "implementation": _implementation_sources(),
    }


def load_contract(root: Path) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    stored = _read_json(contract_path(output))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("residual contract lacks created_utc")
    _parse_utc(created, context="contract")
    expected = _contract_payload(
        output,
        created_utc=created,
        upstream_artifacts=_upstream_artifacts(),
    )
    if stored != expected:
        raise ContractError("residual contract does not exactly replay")
    return stored


def _reconcile_prepare_staging(output: Path) -> None:
    prefix = f".{output.name}.prepare."
    for candidate in output.parent.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        if not candidate.is_dir() or candidate.is_symlink():
            raise ContractError(f"noncanonical prepare staging path: {candidate}")
        names = {path.name for path in candidate.iterdir()}
        if not names <= {"contract.json"} or any(path.is_symlink() for path in candidate.iterdir()):
            raise ContractError(f"unsafe prepare staging contents: {candidate}")
        shutil.rmtree(candidate)


def prepare(root: Path, *, apply: bool) -> dict[str, Any]:
    output = validate_output_root(root)
    upstream = _upstream_artifacts()
    if not apply:
        return {
            "status": "DRY_RUN_PREPARE_RESIDUAL_HEAD",
            "output_root": str(output),
            "upstream_root": str(UPSTREAM_ROOT),
            "bound_upstream_artifacts": len(upstream),
            "new_embedding_jobs": 0,
        }
    if output.exists() or output.is_symlink():
        return load_contract(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reconcile_prepare_staging(output)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare.", dir=output.parent))
    try:
        payload = _contract_payload(
            output,
            created_utc=_utcnow(),
            upstream_artifacts=upstream,
        )
        _write_json_once(stage / "contract.json", payload)
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return load_contract(output)


def _load_upstream_patient_data() -> dict[str, dict[int, pd.DataFrame]]:
    try:
        return pure._patient_data(UPSTREAM_ROOT)  # noqa: SLF001
    except (pure.ContractError, broad.ContractError) as exc:
        raise ContractError(f"sealed upstream patient data failed: {exc}") from exc


def _folds_by_cohort(
    data: Mapping[str, Mapping[int, pd.DataFrame]], layout_seed: int
) -> dict[str, np.ndarray]:
    try:
        return pure._folds_by_cohort(data, layout_seed)  # noqa: SLF001
    except (pure.ContractError, broad.ContractError) as exc:
        raise ContractError(str(exc)) from exc


def _combined_support(
    data: Mapping[str, Mapping[int, pd.DataFrame]],
    *,
    layout_seed: int,
    support_per_class: int,
    draw: int,
    fold: int,
) -> list[tuple[str, str, int]]:
    try:
        return pure._combined_support(  # noqa: SLF001
            data,
            layout_seed=layout_seed,
            support_per_class=support_per_class,
            draw=draw,
            fold=fold,
        )
    except (pure.ContractError, broad.ContractError) as exc:
        raise ContractError(str(exc)) from exc


def _support_roster_digest(
    data: Mapping[str, Mapping[int, pd.DataFrame]],
) -> str:
    rows: list[dict[str, Any]] = []
    for job in residual_jobs():
        for draw in range(DRAWS_PER_LAYOUT):
            for fold in FOLDS:
                support = _combined_support(
                    data,
                    layout_seed=job.layout_seed,
                    support_per_class=job.support_per_class,
                    draw=draw,
                    fold=fold,
                )
                rows.append(
                    {
                        "layout_seed": job.layout_seed,
                        "support_per_class": job.support_per_class,
                        "draw": draw,
                        "fold": fold,
                        "patients": [
                            f"{cohort}::{patient}::{label}" for cohort, patient, label in support
                        ],
                    }
                )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _preflight_payload(
    root: Path,
    *,
    created_utc: str,
    data: Mapping[str, Mapping[int, pd.DataFrame]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_SEALED_UPSTREAM_RESIDUAL_PREFLIGHT",
        "created_utc": created_utc,
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "contract": _artifact(contract_path(root)),
        "upstream_root": str(UPSTREAM_ROOT),
        "upstream_artifacts": _upstream_artifacts(),
        "patient_census": {
            "RIH-M": {
                "patients": len(data["RIH"][MODEL_SEEDS[0]]),
                "mutant": int(data["RIH"][MODEL_SEEDS[0]]["label"].sum()),
            },
            "SurGen-M": {
                "patients": len(data["SurGen"][MODEL_SEEDS[0]]),
                "mutant": int(data["SurGen"][MODEL_SEEDS[0]]["label"].sum()),
            },
        },
        "support_roster_sha256": _support_roster_digest(data),
        "support_procedures_by_fold": 2_000,
        "source_models_embeddings_and_native_logits_frozen": True,
        "new_embedding_jobs": 0,
    }


def _validate_preflight(root: Path, *, deep: bool) -> dict[str, Any]:
    load_contract(root)
    stored = _read_json(preflight_path(root))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("residual preflight lacks created_utc")
    _parse_utc(created, context="preflight")
    if deep:
        expected = _preflight_payload(
            root,
            created_utc=created,
            data=_load_upstream_patient_data(),
        )
        if stored != expected:
            raise ContractError("residual preflight does not exactly replay")
    else:
        if (
            stored.get("status") != "PASS_SEALED_UPSTREAM_RESIDUAL_PREFLIGHT"
            or stored.get("result_role") != INTERNAL_ROLE
            or stored.get("external_validation_claim_permitted") is not False
            or stored.get("contract") != _artifact(contract_path(root))
            or stored.get("new_embedding_jobs") != 0
        ):
            raise ContractError("residual preflight shallow gate failed")
    return stored


def preflight(root: Path, *, apply: bool) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    load_contract(output)
    if not apply:
        return {
            "status": "DRY_RUN_RESIDUAL_PREFLIGHT",
            "upstream_root": str(UPSTREAM_ROOT),
            "new_embedding_jobs": 0,
            "support_procedures_by_fold": 2_000,
        }
    if preflight_path(output).exists():
        return _validate_preflight(output, deep=True)
    data = _load_upstream_patient_data()
    _write_json_once(
        preflight_path(output),
        _preflight_payload(output, created_utc=_utcnow(), data=data),
    )
    return _validate_preflight(output, deep=True)


def _patient_data(root: Path) -> dict[str, dict[int, pd.DataFrame]]:
    _validate_preflight(root, deep=False)
    return _load_upstream_patient_data()


def _solver_calls_per_head(support_per_class: int) -> int:
    total = 2 * support_per_class
    inner_splits = total if total <= broad.adapter.LOO_POOL_MAX else len(FOLDS)
    return inner_splits * len(LAMBDA_GRID) + 1


def _expected_shard_solver_calls(job: ResidualJob) -> int:
    heads = DRAWS_PER_LAYOUT * len(FOLDS) * len(MODEL_SEEDS)
    return heads * _solver_calls_per_head(job.support_per_class)


def _run_residual_shard(
    data: Mapping[str, Mapping[int, pd.DataFrame]], job: ResidualJob
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
                seed: np.full(len(reference[cohort]), np.nan, dtype=float) for seed in MODEL_SEEDS
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
                        data[cohort][source_seed].loc[patient, embedding_columns].to_numpy(float)
                        for cohort, patient, _ in support
                    ]
                )
                offsets = np.asarray(
                    [
                        float(data[cohort][source_seed].loc[patient, "eta_native"])
                        for cohort, patient, _ in support
                    ],
                    dtype=float,
                )
                labels = np.asarray(support_labels, dtype=int)
                context = {
                    "phase": PHASE,
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
                    offsets,
                    labels,
                    inner_seed=job.layout_seed + fold,
                    ledger=ledger,
                    context=context,
                )
                weights, bias, diagnostic = broad.adapter.fit_residual(
                    features,
                    offsets,
                    labels,
                    selected,
                    ledger=ledger,
                    context={**context, "fit_role": "outer_refit"},
                )
                for cohort in ("RIH", "SurGen"):
                    test_index = np.flatnonzero(folds[cohort] == fold)
                    test_frame = data[cohort][source_seed].iloc[test_index]
                    test_features = test_frame[embedding_columns].to_numpy(float)
                    test_native = test_frame["eta_native"].to_numpy(float)
                    predictions[cohort][source_seed][test_index] = (
                        test_native + test_features @ weights + bias
                    )
                fits.append(
                    {
                        "phase": PHASE,
                        "method": METHOD,
                        "anchor_kind": "frozen_native_logit",
                        "infinite_lambda_semantics": (
                            "exact frozen native logit; zero residual correction"
                        ),
                        "coefficient_order": "e0..e511",
                        "layout_seed": job.layout_seed,
                        "support_per_class": job.support_per_class,
                        "support_total": 2 * job.support_per_class,
                        "draw": draw,
                        "outer_fold": fold,
                        "source_model_seed": source_seed,
                        "fit_patient_keys": support_keys,
                        "fit_labels": support_labels,
                        "fit_native_logits": offsets.tolist(),
                        "test_patient_ids_by_cohort": test_ids,
                        "selected_lambda": selection["selected_lambda"],
                        "inner_selection": selection,
                        "coefficients": weights.tolist(),
                        "bias": bias,
                        "solver_diagnostic": diagnostic,
                    }
                )
        for cohort in ("RIH", "SurGen"):
            if any(not np.isfinite(predictions[cohort][seed]).all() for seed in MODEL_SEEDS):
                raise ContractError("residual OOF prediction coverage is incomplete")
            native_by_seed = {
                seed: data[cohort][seed]["eta_native"].to_numpy(float) for seed in MODEL_SEEDS
            }
            for index, patient_id in enumerate(reference[cohort].index.astype(str)):
                rows.append(
                    {
                        "phase": PHASE,
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
                        "eta_residual_ridge": float(
                            np.mean([predictions[cohort][seed][index] for seed in MODEL_SEEDS])
                        ),
                        **{
                            f"eta_native_seed{seed}": float(native_by_seed[seed][index])
                            for seed in MODEL_SEEDS
                        },
                        **{
                            f"eta_residual_ridge_seed{seed}": float(
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
        "infinite_lambda_semantics": ("exact frozen native logit; zero residual correction"),
    }
    solver_calls = solver["n_finite_calls"] + solver["n_native_exact_calls"]
    if (
        len(rows) != expected_rows
        or len(fits) != expected_heads
        or len(supports) != expected_supports
        or solver_calls != _expected_shard_solver_calls(job)
        or solver.get("n_unaccepted") != 0
    ):
        raise ContractError(f"{job.key}: residual shard census/accounting drifted")
    return pd.DataFrame(rows), fits, supports, solver


def _residual_command(root: Path, job: ResidualJob) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_residual-one",
        "--output-root",
        str(root),
        "--layout-seed",
        str(job.layout_seed),
        "--support-per-class",
        str(job.support_per_class),
    ]


def _reported_artifact(path: Path, canonical_path: Path) -> dict[str, Any]:
    artifact = _artifact(path)
    artifact["path"] = str(canonical_path)
    return artifact


def _shard_payload(
    root: Path,
    job: ResidualJob,
    *,
    created_utc: str,
    execution: Mapping[str, Any],
    directory: Path | None = None,
) -> dict[str, Any]:
    canonical = shard_dir(root, job)
    source = canonical if directory is None else directory
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_LEAN_SOURCE_ANCHORED_RESIDUAL_SHARD",
        "created_utc": created_utc,
        "job": {
            "layout_seed": job.layout_seed,
            "support_per_class": job.support_per_class,
        },
        "command": _residual_command(root, job),
        "method": METHOD,
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "contract": _artifact(contract_path(root)),
        "preflight": _artifact(preflight_path(root)),
        "upstream_inference_seal": _artifact(broad.inference_seal_path(UPSTREAM_ROOT)),
        "upstream_target_open": _artifact(pure.target_open_path(UPSTREAM_ROOT)),
        "oof": _reported_artifact(source / "oof.parquet", canonical / "oof.parquet"),
        "fit_ledger": _reported_artifact(source / "fits.jsonl", canonical / "fits.jsonl"),
        "support_ledger": _reported_artifact(
            source / "supports.jsonl", canonical / "supports.jsonl"
        ),
        "solver": _reported_artifact(source / "solver.json", canonical / "solver.json"),
        "execution_record": _reported_artifact(
            source / "execution.json", canonical / "execution.json"
        ),
        "execution": dict(execution),
    }


def _validate_selection(record: Mapping[str, Any]) -> None:
    try:
        pure._validate_selection(record)  # noqa: SLF001
    except (pure.ContractError, broad.ContractError) as exc:
        raise ContractError(str(exc).replace("probe", "residual")) from exc


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
        "eta_residual_ridge",
        *(f"eta_native_seed{seed}" for seed in MODEL_SEEDS),
        *(f"eta_residual_ridge_seed{seed}" for seed in MODEL_SEEDS),
    ]


def _validate_final_head(
    record: Mapping[str, Any],
    *,
    features: np.ndarray,
    offsets: np.ndarray,
    labels: np.ndarray,
) -> None:
    coefficients = np.asarray(record["coefficients"], dtype=float)
    bias = float(record["bias"])
    diagnostic = record["solver_diagnostic"]
    if not isinstance(diagnostic, Mapping):
        raise ContractError("residual final diagnostic is absent")
    theta = np.asarray([*coefficients, bias], dtype=float)
    zero = np.zeros_like(theta)
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
        objective = float(np.mean(np.logaddexp(0.0, offsets) - labels * offsets))
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
            raise ContractError("infinite-lambda residual head is not exact native/no-correction")
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
    value, gradient = broad.adapter.residual_objective_and_gradient(
        theta, features, offsets, labels, lam
    )
    value0, _ = broad.adapter.residual_objective_and_gradient(zero, features, offsets, labels, lam)
    gradient_inf = float(np.max(np.abs(gradient)))
    if (
        set(diagnostic) != expected_keys
        or diagnostic["status"] != "finite_optimum"
        or not math.isclose(float(diagnostic["lambda"]), lam, abs_tol=0.0)
        or not math.isclose(float(diagnostic["objective_at_fit"]), value, abs_tol=1e-12)
        or not math.isclose(float(diagnostic["objective_at_zero"]), value0, abs_tol=1e-12)
        or not math.isclose(float(diagnostic["objective_decrease"]), value0 - value, abs_tol=1e-12)
        or not math.isclose(float(diagnostic["gradient_inf_norm"]), gradient_inf, abs_tol=1e-10)
        or not math.isclose(
            float(diagnostic["coefficient_l2"]),
            float(np.linalg.norm(coefficients)),
            abs_tol=1e-12,
        )
        or not math.isclose(float(diagnostic["bias"]), bias, abs_tol=0.0)
        or (not diagnostic["scipy_success"] and gradient_inf > broad.adapter.SOLVER_GRAD_TOL)
        or value0 - value < -broad.adapter.OBJECTIVE_TOL
    ):
        raise ContractError("finite residual-head optimum does not replay")


def _validate_shard(root: Path, job: ResidualJob, *, deep: bool) -> dict[str, Any] | None:
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
            not (directory / name).is_file() or (directory / name).is_symlink() for name in required
        )
    ):
        raise ContractError(f"{job.key}: partial or noncanonical residual shard")
    execution = _read_json(directory / "execution.json")
    if (
        set(execution)
        != {
            "job_id",
            "started_utc",
            "completed_utc",
            "started_unix_ns",
            "completed_unix_ns",
            "returncode",
            "configured_workers",
        }
        or execution["job_id"] != f"residual_ridge.{job.key}"
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
    expected_receipt = _shard_payload(root, job, created_utc=created, execution=execution)
    if stored != expected_receipt:
        raise ContractError(f"{job.key}: completion receipt does not replay")

    frame = pd.read_parquet(directory / "oof.parquet")
    numeric = [name for name in frame if name.startswith("eta_")]
    if (
        list(frame.columns) != _expected_oof_columns()
        or len(frame) != DRAWS_PER_LAYOUT * 159
        or frame.duplicated(["draw", "test_cohort", "patient_id"]).any()
        or set(frame["phase"].astype(str)) != {PHASE}
        or set(pd.to_numeric(frame["layout_seed"]).astype(int)) != {job.layout_seed}
        or set(pd.to_numeric(frame["support_per_class"]).astype(int)) != {job.support_per_class}
        or set(pd.to_numeric(frame["support_total"]).astype(int)) != {2 * job.support_per_class}
        or set(pd.to_numeric(frame["draw"]).astype(int)) != set(range(DRAWS_PER_LAYOUT))
        or set(frame["test_cohort"].astype(str)) != {"RIH", "SurGen"}
        or not np.isfinite(frame[numeric].to_numpy(float)).all()
    ):
        raise ContractError(f"{job.key}: residual OOF schema/census drifted")
    if not deep:
        return stored

    fits = _read_jsonl(directory / "fits.jsonl")
    supports = _read_jsonl(directory / "supports.jsonl")
    solver = _read_json(directory / "solver.json")
    expected_heads = DRAWS_PER_LAYOUT * len(FOLDS) * len(MODEL_SEEDS)
    expected_supports = DRAWS_PER_LAYOUT * len(FOLDS)
    if len(fits) != expected_heads or len(supports) != expected_supports:
        raise ContractError(f"{job.key}: fit/support census drifted")

    data = _load_upstream_patient_data()
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
                set(wanted_keys) & {f"{cohort}::{patient}" for patient in cohort_patients}
                for cohort, cohort_patients in wanted_tests.items()
            )
        ):
            raise ContractError(f"{job.key}: support/test replay failed at {key}")
        support_map[key] = record
    if set(support_map) != {(draw, fold) for draw in range(DRAWS_PER_LAYOUT) for fold in FOLDS}:
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
        "fit_native_logits",
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
        parsed = [item.split("::", 1) for item in record["fit_patient_keys"]]
        features = np.vstack(
            [
                data[cohort][key[2]].loc[patient, embedding_columns].to_numpy(float)
                for cohort, patient in parsed
            ]
        )
        offsets = np.asarray(
            [data[cohort][key[2]].loc[patient, "eta_native"] for cohort, patient in parsed],
            dtype=float,
        )
        if (
            record["phase"] != PHASE
            or record["method"] != METHOD
            or record["anchor_kind"] != "frozen_native_logit"
            or record["infinite_lambda_semantics"]
            != "exact frozen native logit; zero residual correction"
            or record["coefficient_order"] != "e0..e511"
            or record["layout_seed"] != job.layout_seed
            or record["support_per_class"] != job.support_per_class
            or record["support_total"] != 2 * job.support_per_class
            or key[2] not in MODEL_SEEDS
            or record["fit_patient_keys"] != support["support_patient_keys"]
            or record["fit_labels"] != support["support_labels"]
            or record["test_patient_ids_by_cohort"] != support["test_patient_ids_by_cohort"]
            or not np.array_equal(np.asarray(record["fit_native_logits"], dtype=float), offsets)
            or len(record["coefficients"]) != EMBED_DIM
            or not np.isfinite(
                np.asarray([*record["coefficients"], record["bias"]], dtype=float)
            ).all()
        ):
            raise ContractError(f"{job.key}: fitted-head identity/offset drift at {key}")
        _validate_selection(record)
        _validate_final_head(
            record,
            features=features,
            offsets=offsets,
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
                adapted: list[float] = []
                for seed in MODEL_SEEDS:
                    seed_frame = data[cohort][seed]
                    native = float(seed_frame.loc[patient, "eta_native"])
                    features = seed_frame.loc[patient, embedding_columns].to_numpy(float)
                    record = fit_map[(draw, fold, seed)]
                    eta = float(
                        native
                        + features @ np.asarray(record["coefficients"], dtype=float)
                        + float(record["bias"])
                    )
                    natives.append(native)
                    adapted.append(eta)
                    if not math.isclose(
                        float(row[f"eta_native_seed{seed}"]), native, abs_tol=0.0
                    ) or not math.isclose(
                        float(row[f"eta_residual_ridge_seed{seed}"]),
                        eta,
                        rel_tol=0,
                        abs_tol=1e-12,
                    ):
                        raise ContractError(f"{job.key}: per-seed residual OOF prediction drifted")
                if not math.isclose(
                    float(row["eta_native"]), float(np.mean(natives)), abs_tol=1e-15
                ) or not math.isclose(
                    float(row["eta_residual_ridge"]),
                    float(np.mean(adapted)),
                    abs_tol=1e-15,
                ):
                    raise ContractError(f"{job.key}: five-seed ensemble drifted")

    calls = int(solver.get("n_finite_calls", -1)) + int(solver.get("n_native_exact_calls", -1))
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
        != "exact frozen native logit; zero residual correction"
    ):
        raise ContractError(f"{job.key}: exact residual solver accounting drifted")
    return stored


def _publish_residual_shard(root: Path, job: ResidualJob) -> None:
    output = validate_output_root(root, must_exist=True)
    _validate_preflight(output, deep=False)
    if _validate_shard(output, job, deep=True) is not None:
        return
    final = shard_dir(output, job)
    if final.exists() or final.is_symlink():
        raise ContractError(f"{job.key}: partial final shard cannot be resumed")
    started_utc = _utcnow_precise()
    started_unix_ns = time.time_ns()
    data = _load_upstream_patient_data()
    frame, fits, supports, solver = _run_residual_shard(data, job)
    completed_utc = _utcnow_precise()
    completed_unix_ns = time.time_ns()
    execution = {
        "job_id": f"residual_ridge.{job.key}",
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": completed_unix_ns,
        "returncode": 0,
        "configured_workers": 1,
    }
    staging_parent = output / "adaptation/.staging"
    if staging_parent.exists() and (not staging_parent.is_dir() or staging_parent.is_symlink()):
        raise ContractError("residual staging namespace is noncanonical")
    staging_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"{job.key}.", dir=staging_parent))
    try:
        _write_parquet_once(stage / "oof.parquet", frame)
        _write_jsonl_once(stage / "fits.jsonl", fits)
        _write_jsonl_once(stage / "supports.jsonl", supports)
        _write_json_once(stage / "solver.json", solver)
        _write_json_once(stage / "execution.json", execution)
        _write_json_once(
            stage / "completion.json",
            _shard_payload(
                output,
                job,
                created_utc=_utcnow(),
                execution=execution,
                directory=stage,
            ),
        )
        final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, final)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    _validate_shard(output, job, deep=True)


def _write_or_reconcile_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        if path.is_symlink() or not pd.read_parquet(path).equals(frame):
            raise ContractError(f"immutable parquet drifted: {path}")
    else:
        _write_parquet_once(path, frame)


def _reconcile_shard_staging(root: Path) -> None:
    staging = root / "adaptation/.staging"
    if not staging.exists():
        return
    if not staging.is_dir() or staging.is_symlink():
        raise ContractError("residual staging namespace is noncanonical")
    prefixes = tuple(f"{job.key}." for job in residual_jobs())
    allowed = {
        "oof.parquet",
        "fits.jsonl",
        "supports.jsonl",
        "solver.json",
        "execution.json",
        "completion.json",
    }
    for candidate in staging.iterdir():
        if (
            not candidate.name.startswith(prefixes)
            or not candidate.is_dir()
            or candidate.is_symlink()
        ):
            raise ContractError(f"unsafe residual staging entry: {candidate}")
        contents = list(candidate.iterdir())
        if not {path.name for path in contents} <= allowed or any(
            path.is_symlink() or not path.is_file() for path in contents
        ):
            raise ContractError(f"unsafe residual staging contents: {candidate}")
        shutil.rmtree(candidate)


def _collect_residual_shards(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for job in residual_jobs():
        if _validate_shard(root, job, deep=False) is None:
            raise ContractError(f"{job.key}: required residual shard is missing")
        frames.append(pd.read_parquet(shard_dir(root, job) / "oof.parquet"))
    aggregate = (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            ["support_per_class", "layout_seed", "draw", "test_cohort", "patient_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    if (
        len(aggregate) != 63_600
        or aggregate.duplicated(
            ["support_per_class", "layout_seed", "draw", "test_cohort", "patient_id"]
        ).any()
        or set(pd.to_numeric(aggregate["support_per_class"]).astype(int)) != set(SUPPORT_PER_CLASS)
    ):
        raise ContractError("aggregate residual OOF census/key roster drifted")
    return aggregate


def _scheduler_payload(root: Path, *, created_utc: str) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for job in residual_jobs():
        receipt = _validate_shard(root, job, deep=False)
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
    if peak < 1 or peak > MAX_WORKERS:
        raise ContractError(f"observed residual-process peak was {peak}; maximum is {MAX_WORKERS}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_LEAN_SOURCE_ANCHORED_RESIDUAL_SCHEDULER",
        "created_utc": created_utc,
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "configured_max_workers": MAX_WORKERS,
        "configured_child_workers": 1,
        "observed_peak_workers": peak,
        "peak_requirement": "1 <= observed peak <= configured maximum 6",
        "shard_jobs": len(residual_jobs()),
        "events": events,
        "accounting": accounting(),
        "aggregate_oof": _artifact(aggregate_oof_path(root)),
    }


def _validate_scheduler(root: Path) -> dict[str, Any]:
    stored = _read_json(scheduler_path(root))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("residual scheduler lacks created_utc")
    _parse_utc(created, context="scheduler")
    expected = _scheduler_payload(root, created_utc=created)
    if stored != expected:
        raise ContractError("residual scheduler receipt does not exactly replay")
    return stored


def run_adaptation(root: Path, *, apply: bool, max_workers: int) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    _validate_preflight(output, deep=False)
    if max_workers != MAX_WORKERS:
        raise ContractError("governed residual adaptation requires --max-workers 6")
    commands = [_residual_command(output, job) for job in residual_jobs()]
    if not apply:
        return {
            "status": "DRY_RUN_COMBINED_SOURCE_ANCHORED_RESIDUAL",
            "shards": len(residual_jobs()),
            "configured_max_workers": MAX_WORKERS,
            "accounting": accounting(),
            "commands": commands,
        }
    _reconcile_shard_staging(output)

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
        return int(subprocess.run(list(command), cwd=REPO, env=environment, check=False).returncode)

    pending: list[list[str]] = []
    for job, command in zip(residual_jobs(), commands, strict=True):
        if _validate_shard(output, job, deep=True) is None:
            pending.append(command)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run, command) for command in pending]
        returncodes = [future.result() for future in concurrent.futures.as_completed(futures)]
    if any(returncodes):
        raise ContractError("one or more deterministic residual shards failed")
    aggregate = _collect_residual_shards(output)
    _write_or_reconcile_parquet(aggregate_oof_path(output), aggregate)
    if scheduler_path(output).exists():
        _validate_scheduler(output)
    else:
        _write_json_once(
            scheduler_path(output),
            _scheduler_payload(output, created_utc=_utcnow()),
        )
    return _validate_scheduler(output)


@contextlib.contextmanager
def _pure_summary_context() -> Iterable[None]:
    original = pure.METHOD
    try:
        pure.METHOD = METHOD
        yield
    finally:
        pure.METHOD = original


def _summary_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {"eta_residual_ridge": "eta_pure_ridge"}
    mapping.update(
        {f"eta_residual_ridge_seed{seed}": f"eta_pure_ridge_seed{seed}" for seed in MODEL_SEEDS}
    )
    if any(name in frame for name in mapping.values()):
        raise ContractError("summary input contains forbidden pure-probe columns")
    missing = [name for name in mapping if name not in frame]
    if missing:
        raise ContractError(f"summary input lacks residual columns: {missing}")
    return frame.rename(columns=mapping)


def _summarize_budget(
    frame: pd.DataFrame,
    *,
    support_per_class: int,
    bootstrap_draws: int | None = None,
) -> dict[str, Any]:
    proxy = _summary_proxy(frame)
    try:
        with _pure_summary_context():
            result = pure._summarize_budget(  # noqa: SLF001
                proxy,
                support_per_class=support_per_class,
                bootstrap_draws=bootstrap_draws,
            )
    except (pure.ContractError, broad.ContractError) as exc:
        raise ContractError(str(exc)) from exc
    result["method"] = METHOD
    result["native_offset_frozen"] = True
    result["infinity_semantics"] = "exact frozen native logit; zero residual correction"
    return result


def _build_results(root: Path, *, bootstrap_draws: int | None = None) -> dict[str, Any]:
    _validate_scheduler(root)
    aggregate = pd.read_parquet(aggregate_oof_path(root))
    if len(aggregate) != 63_600 or list(aggregate.columns) != _expected_oof_columns():
        raise ContractError("terminal aggregate residual OOF schema/census drifted")
    cells: dict[str, Any] = {}
    for support in SUPPORT_PER_CLASS:
        block = aggregate.loc[
            pd.to_numeric(aggregate["support_per_class"]).astype(int).eq(support)
        ].copy()
        if len(block) != PROCEDURES_PER_BUDGET * 159:
            raise ContractError(f"k={support}: residual procedure rows drifted")
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
                "paired patient/procedure bootstrap; frozen zero-shot point was sealed "
                "before target outcomes opened in the upstream campaign"
            ),
        }
        for scope, values in first["scopes"].items()
    }
    if (
        native_scopes["pooled_combined"]["census"] != {"patients": 159, "mutant": 67}
        or not math.isclose(
            native_scopes["pooled_combined"]["auroc"],
            EXPECTED_NATIVE_POOLED,
            abs_tol=1e-15,
        )
        or not math.isclose(
            native_scopes["equal_cohort_macro"]["auroc"],
            EXPECTED_NATIVE_EQUAL_MACRO,
            abs_tol=1e-15,
        )
    ):
        raise ContractError("native pooled/macro reporting authority drifted")
    draws = N_BOOTSTRAP if bootstrap_draws is None else bootstrap_draws
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "COMPLETE_COMBINED_MET_SOURCE_ANCHORED_RESIDUAL_FEWSHOT",
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "source_anchor": {
            "development_population": "TCGA+SurGen primaries only",
            "encoder": "UNI-v1",
            "model": "frozen p75 refit",
            "model_seeds": list(MODEL_SEEDS),
            "encoder_mil_classifier_native_logits_and_embeddings_frozen": True,
            "source_or_main_model_fits": 0,
            "upstream_root": str(UPSTREAM_ROOT),
        },
        "native_zero_shot": {
            "role": (
                "frozen comparator inside TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION; "
                "SurGen-M is source-family-exposed"
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
            "procedure_resampling": ("paired across native/residual and all report scopes"),
        },
        "fit_and_solver_accounting": accounting(),
        "firewall": {
            "primary_support_used": False,
            "allowed_target_labels": ["RIH-M", "SurGen-M"],
            "combined_support_only": True,
            "source_checkpoint_encoder_mil_classifier_or_native_logits_modified": False,
            "external_validation_claim_permitted": False,
            "surgen_metastatic_source_family_exposed": True,
            "adaptation_feedback_to_source_selection_or_external_claims": False,
            "pure_probe_local_mil_full_label_or_platt_present": False,
        },
    }


def _completion_payload(root: Path, *, created_utc: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_COMBINED_MET_SOURCE_ANCHORED_RESIDUAL_FEWSHOT",
        "created_utc": created_utc,
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "contract": _artifact(contract_path(root)),
        "preflight": _artifact(preflight_path(root)),
        "upstream_inference_seal": _artifact(broad.inference_seal_path(UPSTREAM_ROOT)),
        "upstream_target_open": _artifact(pure.target_open_path(UPSTREAM_ROOT)),
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
        raise ContractError("terminal residual results do not numerically replay")
    stored = _read_json(completion_path(root))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("terminal residual completion lacks created_utc")
    _parse_utc(created, context="completion")
    if stored != _completion_payload(root, created_utc=created):
        raise ContractError("terminal residual completion receipt does not replay")
    return stored_results


def analyze_campaign(root: Path, *, apply: bool) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    _validate_scheduler(output)
    if not apply:
        return {
            "status": "DRY_RUN_RESIDUAL_ANALYSIS",
            "bootstrap_draws": N_BOOTSTRAP,
            "budgets": list(SUPPORT_PER_CLASS),
            "all_budgets_reported": True,
        }
    results = _build_results(output)
    if results_path(output).exists():
        if _read_json(results_path(output)) != results:
            raise ContractError("persisted immutable residual results drifted")
    else:
        _write_json_once(results_path(output), results)
    if completion_path(output).exists():
        stored = _read_json(completion_path(output))
        created = stored.get("created_utc")
        if not isinstance(created, str) or stored != _completion_payload(
            output, created_utc=created
        ):
            raise ContractError("persisted residual completion drifted")
    else:
        _write_json_once(
            completion_path(output),
            _completion_payload(output, created_utc=_utcnow()),
        )
    return _validate_completion(output)


def _validate_terminal_namespace(root: Path) -> None:
    for forbidden in (
        root / "source_inference",
        root / "inputs",
        root / "adaptation/pure_ridge_oof.parquet",
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise ContractError(f"forbidden copied/upstream namespace exists: {forbidden}")
    staging = root / "adaptation/.staging"
    if staging.exists() and (
        not staging.is_dir() or staging.is_symlink() or any(staging.iterdir())
    ):
        raise ContractError("terminal residual staging namespace is not empty/canonical")


def campaign_plan(root: Path) -> dict[str, Any]:
    output = validate_output_root(root)
    return {
        "campaign": CAMPAIGN,
        "output_root": str(output),
        "sealed_upstream_root_read_only": str(UPSTREAM_ROOT),
        "result_role": INTERNAL_ROLE,
        "production_sequence": [
            "prepare --apply",
            "preflight --apply",
            "adapt --apply --max-workers 6",
            "analyze --apply",
            "verify",
        ],
        "method": METHOD,
        "formula": "eta_adapt = eta_native + H @ w + b",
        "support_regime": "COMBINED",
        "support_per_class_total": list(SUPPORT_PER_CLASS),
        "shards": len(residual_jobs()),
        "max_workers": MAX_WORKERS,
        "accounting": accounting(),
        "excluded": [
            "new source scoring or embedding",
            "pure linear probe",
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
        "preflight": preflight_path(output),
        "residual_scheduler_complete": scheduler_path(output),
        "analysis_complete": completion_path(output),
    }
    return {
        "output_root": str(output),
        "exists": output.is_dir(),
        "stages": {name: path.is_file() and not path.is_symlink() for name, path in stages.items()},
        "production_root_absent": not output.exists(),
    }


def verify_campaign(root: Path) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    load_contract(output)
    _validate_preflight(output, deep=True)
    _validate_scheduler(output)
    for job in residual_jobs():
        if _validate_shard(output, job, deep=True) is None:
            raise ContractError(f"{job.key}: residual shard missing at terminal verify")
    _validate_completion(output)
    _validate_terminal_namespace(output)
    return {
        "status": "VERIFIED_COMPLETE_SOURCE_ANCHORED_RESIDUAL_FEWSHOT",
        "campaign": CAMPAIGN,
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "results": _artifact(results_path(output)),
    }


def _print_json(value: Any) -> None:
    print(
        json.dumps(
            broad.adapter.json_ready(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _add_output_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare", "preflight", "adapt", "analyze", "verify", "status"):
        command = subparsers.add_parser(name)
        _add_output_root(command)
        if name in {"prepare", "preflight", "adapt", "analyze"}:
            command.add_argument("--apply", action="store_true")
        if name == "adapt":
            command.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    residual_one = subparsers.add_parser("_residual-one")
    _add_output_root(residual_one)
    residual_one.add_argument("--layout-seed", type=int, choices=LAYOUT_SEEDS, required=True)
    residual_one.add_argument(
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
        value = preflight(root, apply=args.apply)
    elif args.command == "adapt":
        value = run_adaptation(root, apply=args.apply, max_workers=args.max_workers)
    elif args.command == "analyze":
        value = analyze_campaign(root, apply=args.apply)
    elif args.command == "verify":
        value = verify_campaign(root)
    elif args.command == "_residual-one":
        job = ResidualJob(args.layout_seed, args.support_per_class)
        _publish_residual_shard(root, job)
        value = {"status": "COMPLETE", "job": job.key}
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _print_json(value)


if __name__ == "__main__":
    main()
