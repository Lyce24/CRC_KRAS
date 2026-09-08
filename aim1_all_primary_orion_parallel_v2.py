#!/usr/bin/env python3
"""Governed six-way fold runner for the all-primary-plus-Orion E0 experiment.

This additive v2 component keeps the v1 data, model, evaluation, and reporting
contract intact while replacing only its serialized execution path.  It runs
at most six distinct GPU trainer processes.  Cross-validation folds write to
disjoint directories; a seed's native finalization (including its p75 refit)
is eligible only after all five fold completion records authenticate.

The component must use a fresh output root.  It never imports or reuses partial
training artifacts from the v1 root.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, cast

import torch
from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import aim1_all_primary_orion_experiment as base  # noqa: E402, I001


PARALLEL_SCHEMA_VERSION = 1
MAX_PARALLEL_TRAINERS = 6
MIN_LOGICAL_CPUS = 36
MIN_AVAILABLE_RAM_GIB = 96.0
MIN_FREE_GPU_MIB = 20_000
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim1_e0_all_primary_orion_exp_parallel_v2_20260823"
)
PARALLEL_CONTRACT_NAME = "parallel_execution_contract.json"
PARALLEL_RECEIPT_NAME = "parallel_execution_receipt.json"
WORKER_AUTH_ENV = "OCEANPATH_E0_PARALLEL_V2_SESSION"
_LIBC = ctypes.CDLL(None, use_errno=True)


def _guard_v2_output_root(
    output_root: Path, *, before_contract: bool = False
) -> Path:
    """Production execution is restricted to the one fresh v2 lineage."""

    resolved = base._guard_output_root(output_root, before_contract=before_contract)
    if base._is_relative_to(resolved, Path("/tmp").resolve()):
        return resolved
    expected = DEFAULT_OUTPUT_ROOT.resolve()
    if resolved != expected:
        raise base.ContractError(
            "Parallel-v2 production output root must be exactly "
            f"{expected}; refusing v1/alternate lineage {resolved}"
        )
    return resolved


def _require_pretraining_v2_root(output_root: Path) -> None:
    """Before the supplement is sealed, permit only base-manifest inputs."""

    if _parallel_contract_path(output_root).is_file() or not output_root.exists():
        return
    input_root = (output_root / "inputs").resolve()
    unexpected = [
        path
        for path in output_root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and not base._is_relative_to(path.resolve(), input_root)
    ]
    if unexpected:
        raise base.ContractError(
            "Fresh parallel-v2 root already contains non-input artifacts; "
            f"refusing reuse: {[str(path) for path in unexpected[:10]]}"
        )


def _parallel_contract_path(output_root: Path) -> Path:
    return output_root / "inputs" / PARALLEL_CONTRACT_NAME


def _parallel_receipt_path(output_root: Path) -> Path:
    return output_root / "receipts" / "training_parallel" / PARALLEL_RECEIPT_NAME


def _pool_session_request_dir(output_root: Path) -> Path:
    return output_root / "requests" / "training_parallel" / "pool_sessions"


def _pool_session_receipt_dir(output_root: Path) -> Path:
    return output_root / "receipts" / "training_parallel" / "pool_sessions"


def _root_token(output_root: Path) -> str:
    return hashlib.sha256(str(output_root.resolve()).encode()).hexdigest()[:12]


def _orchestrator_lock_path(output_root: Path) -> Path:
    return Path(f"/tmp/oceanpath_e0_parallel_v2_{_root_token(output_root)}.lock")


def _job_lock_path(output_root: Path, key: str) -> Path:
    safe_key = key.replace(":", "_")
    return Path(
        f"/tmp/oceanpath_e0_parallel_v2_{_root_token(output_root)}_{safe_key}.lock"
    )


def _fold_key(seed: int, fold: int) -> str:
    return f"fold:s{seed}:f{fold}"


def _finalize_key(seed: int) -> str:
    return f"finalize:s{seed}"


def _job_graph(output_root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    order = 0
    for seed in base.SEEDS:
        for fold in range(base.N_FOLDS):
            key = _fold_key(seed, fold)
            jobs.append(
                {
                    "key": key,
                    "kind": "fold",
                    "seed": seed,
                    "fold": fold,
                    "order": order,
                    "depends_on": [],
                    "output_dir": str(
                        (base._run_dir(output_root, seed) / f"fold_{fold}").resolve()
                    ),
                    "lock": str(_job_lock_path(output_root, key)),
                }
            )
            order += 1
    for seed in base.SEEDS:
        key = _finalize_key(seed)
        jobs.append(
            {
                "key": key,
                "kind": "finalize",
                "seed": seed,
                "fold": None,
                "order": order,
                "depends_on": [
                    _fold_key(seed, fold) for fold in range(base.N_FOLDS)
                ],
                "output_dir": str(base._run_dir(output_root, seed).resolve()),
                "lock": str(_job_lock_path(output_root, key)),
            }
        )
        order += 1
    return jobs


def _component_implementation() -> list[dict[str, Any]]:
    from oceanpath.workflows import finalize, training

    paths = (
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(training.__file__).resolve(),
        Path(finalize.__file__).resolve(),
    )
    return [base._artifact(path) for path in paths]


def _build_parallel_contract(
    output_root: Path, *, created_utc: str | None = None
) -> dict[str, Any]:
    base_contract = base._read_json(base._contract_path(output_root))
    serial_policy = base_contract.get("recipe", {}).get("gpu_execution")
    expected_serial = {
        "device": "cuda:0",
        "host_wide_lock": str(base.GPU_LOCK_PATH.resolve()),
        "concurrent_training_processes": 1,
        "cuda_idle_precheck": True,
    }
    if serial_policy != expected_serial:
        raise base.ContractError("Base experiment GPU policy is not the sealed v1 policy")
    return {
        "schema_version": PARALLEL_SCHEMA_VERSION,
        "created_utc": created_utc or base._utc_now(),
        "status": "sealed before training",
        "lineage": output_root.name,
        "output_root": str(output_root.resolve()),
        "role": "additive execution component; model/data/report recipe unchanged",
        "base_experiment_contract": base._artifact(base._contract_path(output_root)),
        "component_implementation": _component_implementation(),
        "configs": {
            str(seed): base._artifact(base._config_path(output_root, seed))
            for seed in base.SEEDS
        },
        "operational_supersession": {
            "base_field": "recipe.gpu_execution",
            "base_value": serial_policy,
            "scope": "training invoked through this v2 component only",
            "model_recipe_changed": False,
            "data_recipe_changed": False,
        },
        "parallel_policy": {
            "device": "cuda:0",
            "max_concurrent_gpu_trainer_processes": MAX_PARALLEL_TRAINERS,
            "training_num_workers_per_process": base.TRAINING_NUM_WORKERS,
            "maximum_configured_train_loader_workers": (
                MAX_PARALLEL_TRAINERS * base.TRAINING_NUM_WORKERS
            ),
            "maximum_resident_train_plus_val_loader_processes": (
                MAX_PARALLEL_TRAINERS * base.TRAINING_NUM_WORKERS * 2
            ),
            "loader_process_accounting": (
                "num_workers=6 is per DataLoader; persistent train and validation "
                "loaders may coexist (up to 12 per trainer / 72 across six trainers)"
            ),
            "legacy_gpu_exclusive_lock_held_by_orchestrator": str(
                base.GPU_LOCK_PATH.resolve()
            ),
            "campaign_orchestrator_lock": str(
                _orchestrator_lock_path(output_root).resolve()
            ),
            "gpu_lease_inherited_by_children": True,
            "child_parent_death_signal": "SIGTERM via Linux PR_SET_PDEATHSIG",
            "parent_sigterm_sighup_cleanup": True,
            "cuda_idle_precheck_before_pool": True,
            "child_environment": {
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            "resource_admission": {
                "min_logical_cpus": MIN_LOGICAL_CPUS,
                "min_available_ram_gib": MIN_AVAILABLE_RAM_GIB,
                "min_free_gpu_mib": MIN_FREE_GPU_MIB,
            },
        },
        "schedule": {
            "dispatch_order": "ready finalizers first, then ascending frozen job order",
            "fold_retry": "quarantine incomplete fold directory; never overwrite it",
            "finalization": (
                "one native run_training call per seed after five authenticated folds; "
                "folds resume-skip, then aggregate, best-fold selection, and p75 refit"
            ),
            "jobs": _job_graph(output_root),
        },
        "fit_census": {
            "fold_fits": len(base.SEEDS) * base.N_FOLDS,
            "p75_refits": len(base.SEEDS),
            "total_fits": len(base.SEEDS) * (base.N_FOLDS + 1),
        },
        "v1_partial_artifacts_reused": False,
        "target_outcomes_opened": False,
    }


def _load_parallel_contract(output_root: Path) -> dict[str, Any]:
    output_root = _guard_v2_output_root(output_root)
    base._load_contract(output_root)
    path = _parallel_contract_path(output_root)
    observed = base._read_json(path)
    created_utc = observed.get("created_utc")
    if not isinstance(created_utc, str):
        raise base.ContractError("Parallel contract lacks created_utc")
    expected = _build_parallel_contract(output_root, created_utc=created_utc)
    if observed != expected:
        raise base.ContractError("Parallel execution contract changed after sealing")
    return observed


def _request_path(output_root: Path, job: dict[str, Any]) -> Path:
    if job["kind"] == "fold":
        return (
            output_root
            / "requests"
            / "training_parallel"
            / f"seed{job['seed']}"
            / f"fold{job['fold']}.json"
        )
    return (
        output_root
        / "requests"
        / "training_parallel"
        / f"seed{job['seed']}"
        / "finalize.json"
    )


def _job_receipt_path(output_root: Path, job: dict[str, Any]) -> Path:
    if job["kind"] == "fold":
        return (
            output_root
            / "receipts"
            / "training_parallel"
            / f"seed{job['seed']}"
            / f"fold{job['fold']}.json"
        )
    return (
        output_root
        / "receipts"
        / "training_parallel"
        / f"seed{job['seed']}"
        / "finalize.json"
    )


def _job_log_dir(output_root: Path, job: dict[str, Any]) -> Path:
    suffix = f"fold{job['fold']}" if job["kind"] == "fold" else "finalize"
    return output_root / "logs" / "training_parallel" / f"seed{job['seed']}" / suffix


def _job_request(output_root: Path, job: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "requested",
        "created_utc": base._utc_now(),
        "job": job,
        "parallel_contract": base._artifact(_parallel_contract_path(output_root)),
        "base_contract": base._artifact(base._contract_path(output_root)),
        "config": base._artifact(base._config_path(output_root, int(job["seed"]))),
        "training_num_workers": base.TRAINING_NUM_WORKERS,
        "target_outcomes_opened": False,
    }


def _ensure_job_request(output_root: Path, job: dict[str, Any]) -> dict[str, Any]:
    path = _request_path(output_root, job)
    live = _job_request(output_root, job)
    if path.is_file():
        existing = base._read_json(path)
        live["created_utc"] = existing.get("created_utc")
        if existing != live:
            raise base.ContractError(f"Existing request differs for {job['key']}")
    else:
        base._publish_json(path, live)
    return base._artifact(path)


def _load_cfg(output_root: Path, seed: int) -> DictConfig:
    cfg = OmegaConf.load(base._config_path(output_root, seed))
    base._validate_training_config(
        cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        seed,
        output_root,
    )
    return cfg


def _preinitialize_seed(output_root: Path, seed: int) -> str:
    from oceanpath.workflows.training import (
        _assert_training_output_identity,
        _write_training_identity,
        training_run_fingerprint,
    )

    cfg = _load_cfg(output_root, seed)
    directory = base._run_dir(output_root, seed)
    directory.mkdir(parents=True, exist_ok=True)
    expected = training_run_fingerprint(cfg)
    observed = _assert_training_output_identity(cfg, directory)
    if observed != expected:
        raise base.ContractError(f"Seed {seed} native training identity mismatch")
    identity_path = directory / "training_identity.json"
    if (
        not identity_path.is_file()
        and _write_training_identity(cfg, directory) != expected
    ):
        raise base.ContractError(f"Seed {seed} identity publication failed")
    root_cfg = directory / "config.yaml"
    frozen_text = base._config_path(output_root, seed).read_text(encoding="utf-8")
    base._publish_text(root_cfg, frozen_text)
    base._validate_partial_training_identity(output_root, seed)
    return expected


def _fold_complete(output_root: Path, seed: int, fold: int) -> bool:
    from oceanpath.workflows.training import _fold_complete as native_fold_complete
    from oceanpath.workflows.training import training_run_fingerprint

    cfg = _load_cfg(output_root, seed)
    return native_fold_complete(
        base._run_dir(output_root, seed) / f"fold_{fold}",
        fold_idx=fold,
        training_fingerprint=training_run_fingerprint(cfg),
    )


def _finalize_complete(output_root: Path, seed: int) -> bool:
    path = base._run_dir(output_root, seed) / "training_completion.json"
    if not path.is_file():
        return False
    try:
        base._training_record(output_root, seed)
    except (base.ContractError, FileNotFoundError) as exc:
        if _recoverable_finalization_failure(output_root, seed):
            return False
        raise base.ContractError(
            f"Seed {seed} has non-recoverable invalid finalization evidence"
        ) from exc
    return True


def _recoverable_finalization_failure(output_root: Path, seed: int) -> bool:
    """Identify a failed/missing refit while preserving valid fold evidence."""

    if not all(
        _fold_complete(output_root, seed, fold) for fold in range(base.N_FOLDS)
    ):
        return False
    final_dir = base._run_dir(output_root, seed) / "final"
    summary_path = final_dir / "finalize_summary.json"
    if not summary_path.is_file():
        return True
    try:
        summary = base._read_json(summary_path)
    except (OSError, json.JSONDecodeError, base.ContractError):
        return True
    refit = summary.get("refit") or {}
    return (
        "error" in refit
        or not (final_dir / "refit" / "model.ckpt").is_file()
        or not (final_dir / "refit" / "info.json").is_file()
    )


def _completed_keys(output_root: Path) -> set[str]:
    completed: set[str] = set()
    for job in _job_graph(output_root):
        native_complete = (
            _fold_complete(output_root, int(job["seed"]), int(job["fold"]))
            if job["kind"] == "fold"
            else _finalize_complete(output_root, int(job["seed"]))
        )
        if native_complete and _job_receipt_path(output_root, job).is_file():
            _ensure_job_receipt(output_root, job)
            completed.add(job["key"])
    return completed


def _ready_jobs(
    pending: list[dict[str, Any]], completed: set[str]
) -> list[dict[str, Any]]:
    ready = [
        job
        for job in pending
        if all(dependency in completed for dependency in job["depends_on"])
    ]
    return sorted(
        ready,
        key=lambda job: (0 if job["kind"] == "finalize" else 1, int(job["order"])),
    )


@contextmanager
def _nonblocking_job_lock(path: Path, operation: str) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise base.ContractError(f"Job lock is already held: {operation}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()} operation={operation} at={base._utc_now()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _parallel_gpu_lease(operation: str) -> Any:
    """Hold the legacy GPU flock and expose its inheritable lease descriptor."""

    base.GPU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with base.GPU_LOCK_PATH.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise base.ContractError(
                f"GPU0 is already reserved; cannot start {operation}"
            ) from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()} operation={operation} at={base._utc_now()}\n")
        stream.flush()
        try:
            yield stream.fileno()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _termination_as_exception() -> Any:
    """Route catchable parent termination through scheduler child cleanup."""

    watched = (signal.SIGTERM, signal.SIGHUP)
    previous = {signum: signal.getsignal(signum) for signum in watched}

    def handle(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    for signum in watched:
        signal.signal(signum, handle)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _configure_child_parent_death(parent_pid: int) -> None:
    """Ask Linux to terminate a GPU worker if its orchestrator disappears."""

    if _LIBC.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        os._exit(127)
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def _require_worker_authorization(contract_sha256: str) -> None:
    token = os.environ.get(WORKER_AUTH_ENV, "")
    expected = os.environ.get(f"{WORKER_AUTH_ENV}_SHA256", "")
    if not token or hashlib.sha256(token.encode()).hexdigest() != expected:
        raise base.ContractError("Internal worker lacks an authorized orchestrator session")
    if os.environ.get(f"{WORKER_AUTH_ENV}_CONTRACT") != contract_sha256:
        raise base.ContractError("Internal worker session binds a different contract")


def _run_fold_worker(
    output_root: Path, seed: int, fold: int, contract_sha256: str
) -> None:
    from oceanpath.runtime import run_context
    from oceanpath.workflows.training import (
        _fold_complete as native_fold_complete,
    )
    from oceanpath.workflows.training import (
        _setup_logging,
        fold_context,
        run_fold,
        training_run_fingerprint,
    )

    _require_worker_authorization(contract_sha256)
    contract = _load_parallel_contract(output_root)
    if base._artifact(_parallel_contract_path(output_root))["sha256"] != contract_sha256:
        raise base.ContractError("Fold worker contract digest mismatch")
    job = next(
        item
        for item in contract["schedule"]["jobs"]
        if item["key"] == _fold_key(seed, fold)
    )
    cfg = _load_cfg(output_root, seed)
    fingerprint = _preinitialize_seed(output_root, seed)
    directory = base._run_dir(output_root, seed)
    fold_dir = directory / f"fold_{fold}"
    with _nonblocking_job_lock(Path(job["lock"]), job["key"]):
        if native_fold_complete(
            fold_dir, fold_idx=fold, training_fingerprint=fingerprint
        ):
            print(f"{job['key']}: already complete; skipping")
            return
        if fold_dir.exists():
            raise base.ContractError(
                f"{job['key']} has an incomplete directory; orchestrator must quarantine it"
            )
        _setup_logging(cfg)
        torch.set_float32_matmul_precision("high")
        with run_context(
            cfg,
            stage="train_model",
            output_dir=fold_dir,
            persist=True,
        ), fold_context(fold):
            run_fold(cfg=cfg, fold_idx=fold, output_dir=directory)
        if not native_fold_complete(
            fold_dir,
            fold_idx=fold,
            training_fingerprint=training_run_fingerprint(cfg),
        ):
            raise base.ContractError(f"{job['key']} did not publish valid completion")


def _run_finalize_worker(
    output_root: Path, seed: int, contract_sha256: str
) -> None:
    from oceanpath.workflows.training import run_training

    _require_worker_authorization(contract_sha256)
    contract = _load_parallel_contract(output_root)
    if base._artifact(_parallel_contract_path(output_root))["sha256"] != contract_sha256:
        raise base.ContractError("Finalize worker contract digest mismatch")
    job = next(
        item
        for item in contract["schedule"]["jobs"]
        if item["key"] == _finalize_key(seed)
    )
    _preinitialize_seed(output_root, seed)
    with _nonblocking_job_lock(Path(job["lock"]), job["key"]):
        if _finalize_complete(output_root, seed):
            print(f"{job['key']}: already complete; skipping")
            return
        missing = [
            fold
            for fold in range(base.N_FOLDS)
            if not _fold_complete(output_root, seed, fold)
        ]
        if missing:
            raise base.ContractError(
                f"{job['key']} cannot run before authenticated folds {missing}"
            )
        print(run_training(_load_cfg(output_root, seed)).to_json())
        summary = base._read_json(
            base._run_dir(output_root, seed) / "final" / "finalize_summary.json"
        )
        refit = summary.get("refit") or {}
        if "error" in refit or not (
            base._run_dir(output_root, seed) / "final" / "refit" / "model.ckpt"
        ).is_file():
            raise base.ContractError(
                f"{job['key']} native finalization did not produce a valid p75 refit"
            )
        base._training_record(output_root, seed)


def _attempt_number(log_dir: Path) -> int:
    attempts = sorted(log_dir.glob("attempt-*.log")) if log_dir.is_dir() else []
    return len(attempts) + 1


def _quarantine_incomplete_fold(
    output_root: Path, job: dict[str, Any], attempt: int
) -> None:
    if job["kind"] != "fold":
        return
    seed = int(job["seed"])
    fold = int(job["fold"])
    fold_dir = base._run_dir(output_root, seed) / f"fold_{fold}"
    if not fold_dir.exists() or _fold_complete(output_root, seed, fold):
        return
    quarantine_root = base._run_dir(output_root, seed) / "failed_fold_attempts"
    index = max(1, attempt - 1)
    destination = quarantine_root / f"fold_{fold}" / f"attempt-{index:03d}"
    while destination.exists():
        index += 1
        destination = quarantine_root / f"fold_{fold}" / f"attempt-{index:03d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fold_dir.rename(destination)
    inventory = [
        base._artifact(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    base._publish_json(
        destination / "quarantine_receipt.json",
        {
            "schema_version": 1,
            "status": "incomplete attempt retained",
            "job": job["key"],
            "quarantined_utc": base._utc_now(),
            "artifacts_before_receipt": inventory,
        },
    )


def _quarantine_incomplete_finalization(
    output_root: Path, job: dict[str, Any], attempt: int
) -> None:
    if job["kind"] != "finalize":
        return
    seed = int(job["seed"])
    if _finalize_complete(output_root, seed):
        return
    directory = base._run_dir(output_root, seed)
    known = (
        "training_completion.json",
        "cv_summary.json",
        "es_val_predictions.parquet",
        "oof_predictions.parquet",
        "test_predictions.parquet",
        "final",
        "_run",
    )
    existing = [directory / name for name in known if (directory / name).exists()]
    if not existing:
        return
    if not all(
        _fold_complete(output_root, seed, fold) for fold in range(base.N_FOLDS)
    ):
        raise base.ContractError(
            f"Cannot recover seed {seed} finalization because a fold is invalid"
        )
    quarantine_root = directory / "failed_finalization_attempts"
    index = max(1, attempt - 1)
    destination = quarantine_root / f"attempt-{index:03d}"
    while destination.exists():
        index += 1
        destination = quarantine_root / f"attempt-{index:03d}"
    destination.mkdir(parents=True)
    for source in existing:
        source.rename(destination / source.name)
    inventory = [
        base._artifact(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    base._publish_json(
        destination / "quarantine_receipt.json",
        {
            "schema_version": 1,
            "status": "failed/incomplete finalization retained",
            "job": job["key"],
            "quarantined_utc": base._utc_now(),
            "artifacts_before_receipt": inventory,
            "authenticated_folds_preserved": base.N_FOLDS,
        },
    )


def _ensure_job_receipt(
    output_root: Path, job: dict[str, Any], log_path: Path | None = None
) -> dict[str, Any]:
    path = _job_receipt_path(output_root, job)
    observed = base._read_json(path) if path.is_file() else None
    if observed is not None:
        recorded_log = base._verify_artifact(observed.get("log"), f"{job['key']} log")
        if log_path is not None and log_path.resolve() != recorded_log.resolve():
            raise base.ContractError(f"Job receipt log differs for {job['key']}")
        log_path = recorded_log
        exit_path = base._verify_artifact(
            observed.get("exit"), f"{job['key']} exit evidence"
        )
    else:
        if log_path is None:
            raise base.ContractError(
                f"Native completion for {job['key']} needs a new certification attempt"
            )
        exit_path = log_path.with_suffix(".exit.json")
    exit_record = base._read_json(exit_path)
    expected_attempt = int(log_path.stem.split("-")[-1])
    expected_exit = {
        "job": job["key"],
        "attempt": expected_attempt,
        "returncode": 0,
        "log": base._artifact(log_path),
    }
    exit_mismatch = {
        key: {"expected": value, "observed": exit_record.get(key)}
        for key, value in expected_exit.items()
        if exit_record.get(key) != value
    }
    if exit_mismatch or not isinstance(exit_record.get("finished_utc"), str):
        raise base.ContractError(
            f"Job {job['key']} lacks matching returncode-0 exit evidence: {exit_mismatch}"
        )
    request = _ensure_job_request(output_root, job)
    if job["kind"] == "fold":
        seed = int(job["seed"])
        fold = int(job["fold"])
        if not _fold_complete(output_root, seed, fold):
            raise base.ContractError(f"Cannot receipt incomplete job {job['key']}")
        native = base._artifact(
            base._run_dir(output_root, seed) / f"fold_{fold}" / "completion.json"
        )
        base_training_receipt = None
    else:
        seed = int(job["seed"])
        if not _finalize_complete(output_root, seed):
            raise base.ContractError(f"Cannot receipt incomplete job {job['key']}")
        native = base._artifact(
            base._run_dir(output_root, seed) / "training_completion.json"
        )
        base_training_receipt = base._ensure_training_receipt(output_root, seed)
    expected = {
        "schema_version": 1,
        "status": "completed",
        "job": job,
        "request": request,
        "log": base._artifact(log_path),
        "exit": base._artifact(exit_path),
        "native_completion": native,
        "base_training_receipt": base_training_receipt,
        "target_outcomes_opened": False,
    }
    if observed is not None:
        expected["finished_utc"] = observed.get("finished_utc")
        if observed != expected:
            raise base.ContractError(f"Job receipt differs for {job['key']}")
    else:
        base._publish_json(path, {**expected, "finished_utc": base._utc_now()})
    return base._artifact(path)


def _publish_base_seed_requests(output_root: Path) -> None:
    for seed in base.SEEDS:
        path = output_root / "requests" / "training" / f"seed{seed}.json"
        live = base._train_request(output_root, seed)
        if path.is_file():
            existing = base._read_json(path)
            if existing != base._train_request_comparable(existing, live):
                raise base.ContractError(f"Base seed request differs for seed {seed}")
        else:
            base._publish_json(path, live)


def _available_ram_gib() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", maxsplit=1)
        values[key] = int(value.strip().split()[0])
    return values.get("MemAvailable", 0) / 1024**2


def _free_gpu_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise base.ContractError(f"GPU memory query failed: {result.stderr.strip()}")
    return int(result.stdout.strip().splitlines()[0])


def _resource_admission() -> dict[str, Any]:
    observed = {
        "logical_cpus": int(os.cpu_count() or 0),
        "available_ram_gib": _available_ram_gib(),
        "free_gpu_mib": _free_gpu_mib(),
    }
    failures = {
        "logical_cpus": (MIN_LOGICAL_CPUS, observed["logical_cpus"]),
        "available_ram_gib": (MIN_AVAILABLE_RAM_GIB, observed["available_ram_gib"]),
        "free_gpu_mib": (MIN_FREE_GPU_MIB, observed["free_gpu_mib"]),
    }
    failed = {
        key: {"minimum": minimum, "observed": value}
        for key, (minimum, value) in failures.items()
        if value < minimum
    }
    if failed:
        raise base.ContractError(f"Six-trainer resource admission failed: {failed}")
    return observed


def _begin_pool_session(
    output_root: Path, resources: dict[str, Any]
) -> tuple[int, Path]:
    directory = _pool_session_request_dir(output_root)
    attempts = sorted(directory.glob("attempt-*.json")) if directory.is_dir() else []
    attempt = len(attempts) + 1
    path = directory / f"attempt-{attempt:03d}.json"
    base._publish_json(
        path,
        {
            "schema_version": 1,
            "status": "six-way pool requested",
            "attempt": attempt,
            "started_utc": base._utc_now(),
            "parallel_contract": base._artifact(_parallel_contract_path(output_root)),
            "resource_admission": resources,
            "authorized_max_concurrent_gpu_trainers": MAX_PARALLEL_TRAINERS,
            "target_outcomes_opened": False,
        },
    )
    return attempt, path


def _finish_pool_session(
    output_root: Path,
    attempt: int,
    request_path: Path,
    execution: dict[str, int],
) -> dict[str, Any]:
    peak = execution.get("observed_peak_concurrent_gpu_trainers")
    if not isinstance(peak, int) or not 0 <= peak <= MAX_PARALLEL_TRAINERS:
        raise base.ContractError(f"Invalid observed pool concurrency: {peak}")
    path = _pool_session_receipt_dir(output_root) / f"attempt-{attempt:03d}.json"
    base._publish_json(
        path,
        {
            "schema_version": 1,
            "status": "pool completed",
            "attempt": attempt,
            "finished_utc": base._utc_now(),
            "request": base._artifact(request_path),
            "execution": execution,
            "target_outcomes_opened": False,
        },
    )
    return base._artifact(path)


def _worker_command(
    output_root: Path, job: dict[str, Any], contract_sha256: str
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_fold" if job["kind"] == "fold" else "_finalize",
        "--output-root",
        str(output_root),
        "--seed",
        str(job["seed"]),
        "--contract-sha256",
        contract_sha256,
    ]
    if job["kind"] == "fold":
        command.extend(["--fold", str(job["fold"])])
    return command


@dataclass
class _RunningJob:
    job: dict[str, Any]
    process: subprocess.Popen[str]
    log_path: Path
    log_stream: TextIO
    attempt: int


def _launch_job(
    output_root: Path,
    job: dict[str, Any],
    contract_sha256: str,
    session: str,
    gpu_lease_fd: int,
) -> _RunningJob:
    log_dir = _job_log_dir(output_root, job)
    attempt = _attempt_number(log_dir)
    _quarantine_incomplete_fold(output_root, job, attempt)
    _quarantine_incomplete_finalization(output_root, job, attempt)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"attempt-{attempt:03d}.log"
    log_stream = log_path.open("x", encoding="utf-8", buffering=1)
    command = _worker_command(output_root, job, contract_sha256)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        WORKER_AUTH_ENV: session,
        f"{WORKER_AUTH_ENV}_SHA256": hashlib.sha256(session.encode()).hexdigest(),
        f"{WORKER_AUTH_ENV}_CONTRACT": contract_sha256,
    }
    parent_pid = os.getpid()
    process = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        start_new_session=True,
        pass_fds=(gpu_lease_fd,),
        preexec_fn=lambda: _configure_child_parent_death(parent_pid),
    )
    print(f"launched {job['key']} pid={process.pid} log={log_path}", flush=True)
    return _RunningJob(job, process, log_path, log_stream, attempt)


def _record_exit(
    output_root: Path, running: _RunningJob, returncode: int
) -> None:
    running.log_stream.flush()
    running.log_stream.close()
    path = running.log_path.with_suffix(".exit.json")
    base._publish_json(
        path,
        {
            "schema_version": 1,
            "job": running.job["key"],
            "attempt": running.attempt,
            "returncode": returncode,
            "finished_utc": base._utc_now(),
            "log": base._artifact(running.log_path),
        },
    )


def _terminate_and_reap(
    output_root: Path, running: dict[str, _RunningJob]
) -> None:
    """Stop every child process group before the parent releases GPU0."""

    for active in running.values():
        if active.process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(active.process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30.0
    reaped: list[tuple[_RunningJob, int]] = []
    for active in running.values():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            returncode = active.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(active.process.pid, signal.SIGKILL)
            returncode = active.process.wait()
        reaped.append((active, int(returncode)))
    evidence_errors: list[str] = []
    for active, returncode in reaped:
        if active.log_stream.closed:
            continue
        try:
            _record_exit(output_root, active, returncode)
        except Exception as exc:  # safety cleanup must continue for every child
            with suppress(Exception):
                active.log_stream.close()
            evidence_errors.append(f"{active.job['key']}: {exc}")
    if evidence_errors:
        print(
            "WARNING: child exit evidence publication failed after all children "
            f"were reaped: {evidence_errors}",
            file=sys.stderr,
            flush=True,
        )


def _run_scheduler(
    output_root: Path, contract: dict[str, Any], gpu_lease_fd: int
) -> dict[str, int]:
    jobs = cast(list[dict[str, Any]], contract["schedule"]["jobs"])
    completed = _completed_keys(output_root)
    pending = [job for job in jobs if job["key"] not in completed]
    running: dict[str, _RunningJob] = {}
    failures: list[tuple[str, int]] = []
    contract_sha256 = base._artifact(_parallel_contract_path(output_root))["sha256"]
    session = secrets.token_hex(32)
    peak_concurrency = 0

    try:
        while pending or running:
            for key, active in list(running.items()):
                returncode = active.process.poll()
                if returncode is None:
                    continue
                _record_exit(output_root, active, int(returncode))
                del running[key]
                if returncode != 0:
                    failures.append((key, int(returncode)))
                    print(
                        f"FAILED {key} rc={returncode}; no new jobs will launch",
                        flush=True,
                    )
                    continue
                _ensure_job_receipt(output_root, active.job, active.log_path)
                completed.add(key)
                print(f"completed {key}", flush=True)

            if not failures:
                ready = _ready_jobs(pending, completed)
                while ready and len(running) < MAX_PARALLEL_TRAINERS:
                    job = ready.pop(0)
                    pending.remove(job)
                    running[job["key"]] = _launch_job(
                        output_root, job, contract_sha256, session, gpu_lease_fd
                    )
                    peak_concurrency = max(peak_concurrency, len(running))
            if failures and not running:
                raise RuntimeError(f"Parallel training jobs failed: {failures}")
            if pending and not running and not _ready_jobs(pending, completed):
                raise base.ContractError("Parallel job graph is deadlocked")
            if running:
                time.sleep(1.0)
    except BaseException:
        _terminate_and_reap(output_root, running)
        raise
    return {"observed_peak_concurrent_gpu_trainers": peak_concurrency}


def _ensure_parallel_receipt(output_root: Path) -> dict[str, Any]:
    contract = _load_parallel_contract(output_root)
    jobs = cast(list[dict[str, Any]], contract["schedule"]["jobs"])
    job_receipts = {
        job["key"]: _ensure_job_receipt(output_root, job) for job in jobs
    }
    training = {
        str(seed): base._ensure_training_receipt(output_root, seed)
        for seed in base.SEEDS
    }
    session_paths = sorted(_pool_session_receipt_dir(output_root).glob("attempt-*.json"))
    if not session_paths:
        raise base.ContractError("Parallel campaign has no completed pool-session receipt")
    sessions = [base._artifact(path) for path in session_paths]
    session_records = [base._read_json(path) for path in session_paths]
    observed_peak = max(
        int(record.get("execution", {}).get("observed_peak_concurrent_gpu_trainers", -1))
        for record in session_records
    )
    if not 0 <= observed_peak <= MAX_PARALLEL_TRAINERS:
        raise base.ContractError("Pool-session concurrency evidence is invalid")
    expected = {
        "schema_version": 1,
        "status": "18 fits completed under six-way parallel-v2 schedule",
        "parallel_contract": base._artifact(_parallel_contract_path(output_root)),
        "base_contract": base._artifact(base._contract_path(output_root)),
        "jobs": job_receipts,
        "base_training_receipts": training,
        "pool_sessions": sessions,
        "observed_peak_concurrent_gpu_trainers": observed_peak,
        "max_concurrent_gpu_trainer_processes": MAX_PARALLEL_TRAINERS,
        "target_outcomes_opened": False,
    }
    path = _parallel_receipt_path(output_root)
    if path.is_file():
        observed = base._read_json(path)
        expected["finished_utc"] = observed.get("finished_utc")
        if observed != expected:
            raise base.ContractError("Parallel execution receipt changed")
    else:
        base._publish_json(path, {**expected, "finished_utc": base._utc_now()})
    return base._artifact(path)


def _load_parallel_receipt(output_root: Path) -> dict[str, Any]:
    identity = _ensure_parallel_receipt(output_root)
    return {"identity": identity, "record": base._read_json(Path(identity["path"]))}


def cmd_plan(args: argparse.Namespace) -> None:
    _guard_v2_output_root(args.output_root, before_contract=True)
    base._validate_canonical_input_arguments(args)
    print("Exploratory E0 all-primary-plus-Orion parallel-v2 campaign")
    print(f"  fresh immutable root: {args.output_root}")
    print("  model/data/report recipe: identical to the sealed v1 runner")
    print(
        "  execution: 15 distinct fold fits + 3 dependency-gated p75 refits; "
        "at most 6 GPU trainer processes"
    )
    print(
        f"  each trainer retains training.num_workers={base.TRAINING_NUM_WORKERS} "
        "(up to 12 resident train+val loader processes per trainer / 72 total)"
    )
    print("  v1 partial artifacts: retained in place and never imported")


def cmd_manifest(args: argparse.Namespace) -> None:
    _guard_v2_output_root(args.output_root, before_contract=True)
    _require_pretraining_v2_root(args.output_root)
    base.cmd_manifest(args)
    _require_pretraining_v2_root(args.output_root)
    path = _parallel_contract_path(args.output_root)
    if path.is_file():
        _load_parallel_contract(args.output_root)
        print(f"Parallel-v2 contract already exists and is valid: {path}")
        return
    base._publish_json(path, _build_parallel_contract(args.output_root))
    _load_parallel_contract(args.output_root)
    print(f"Sealed parallel-v2 execution contract: {path}")


def cmd_preflight(args: argparse.Namespace) -> None:
    _guard_v2_output_root(args.output_root, before_contract=True)
    base.cmd_preflight(args)
    if not base._contract_path(args.output_root).is_file():
        print("  run `manifest` to seal the parallel-v2 execution component")
        return
    contract = _load_parallel_contract(args.output_root)
    print(
        "PASS: parallel-v2 job DAG is sealed: "
        f"{len(contract['schedule']['jobs'])} jobs, maximum {MAX_PARALLEL_TRAINERS} trainers"
    )
    print(
        f"  live resources: {os.cpu_count()} logical CPUs, "
        f"{_available_ram_gib():.1f} GiB available RAM"
    )


def cmd_train(args: argparse.Namespace) -> None:
    contract = _load_parallel_contract(args.output_root)
    base._validate_generated_splits(args.output_root)
    jobs = cast(list[dict[str, Any]], contract["schedule"]["jobs"])
    contract_sha256 = base._artifact(_parallel_contract_path(args.output_root))["sha256"]
    if args.dry_run:
        for job in jobs:
            print(" ".join(_worker_command(args.output_root, job, contract_sha256)))
        print(
            f"DRY RUN: {len(jobs)} dependency-governed jobs; "
            f"maximum concurrency={MAX_PARALLEL_TRAINERS}"
        )
        return
    if _parallel_receipt_path(args.output_root).is_file():
        receipt = _load_parallel_receipt(args.output_root)
        print(f"Parallel-v2 training already complete: {receipt['identity']['path']}")
        return

    _publish_base_seed_requests(args.output_root)
    for job in jobs:
        _ensure_job_request(args.output_root, job)
    for seed in base.SEEDS:
        _preinitialize_seed(args.output_root, seed)

    with _nonblocking_job_lock(
        _orchestrator_lock_path(args.output_root), "parallel-v2 orchestrator"
    ), _parallel_gpu_lease(
        "E0 all-primary-Orion parallel-v2 six-job pool"
    ) as gpu_lease_fd, _termination_as_exception():
        # Holding the legacy lock for the full pool prevents a serialized v1
        # job from entering GPU0 while the six authorized children are active.
        base._cuda_idle_precheck()
        resources = _resource_admission()
        print(f"six-trainer resource admission: {json.dumps(resources, sort_keys=True)}")
        attempt, session_request = _begin_pool_session(args.output_root, resources)
        execution = _run_scheduler(args.output_root, contract, gpu_lease_fd)
        _finish_pool_session(args.output_root, attempt, session_request, execution)
    receipt = _ensure_parallel_receipt(args.output_root)
    print(f"parallel execution: {json.dumps(execution, sort_keys=True)}")
    print(f"Parallel-v2 training complete: {receipt['path']}")


def cmd_score(args: argparse.Namespace) -> None:
    _load_parallel_receipt(args.output_root)
    base.cmd_score(args)


def cmd_report(args: argparse.Namespace) -> None:
    _load_parallel_receipt(args.output_root)
    base.cmd_report(args)


def cmd_verify(args: argparse.Namespace) -> None:
    receipt = _load_parallel_receipt(args.output_root)
    base.cmd_verify(args)
    print(
        "PASS: parallel-v2 execution lineage is complete and immutable: "
        f"{receipt['identity']['path']}"
    )


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    base._add_input_arguments(parser)
    parser.set_defaults(output_root=DEFAULT_OUTPUT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("plan", cmd_plan),
        ("manifest", cmd_manifest),
        ("preflight", cmd_preflight),
    ):
        sub = commands.add_parser(name)
        _add_input_arguments(sub)
        sub.set_defaults(func=function)

    train = commands.add_parser("train")
    train.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=cmd_train)

    fold = commands.add_parser("_fold")
    fold.add_argument("--output-root", type=Path, required=True)
    fold.add_argument("--seed", type=int, choices=base.SEEDS, required=True)
    fold.add_argument("--fold", type=int, choices=range(base.N_FOLDS), required=True)
    fold.add_argument("--contract-sha256", required=True)
    fold.set_defaults(
        func=lambda args: _run_fold_worker(
            args.output_root, args.seed, args.fold, args.contract_sha256
        )
    )

    finalize = commands.add_parser("_finalize")
    finalize.add_argument("--output-root", type=Path, required=True)
    finalize.add_argument("--seed", type=int, choices=base.SEEDS, required=True)
    finalize.add_argument("--contract-sha256", required=True)
    finalize.set_defaults(
        func=lambda args: _run_finalize_worker(
            args.output_root, args.seed, args.contract_sha256
        )
    )

    score = commands.add_parser("score")
    score.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    score.add_argument("--target", choices=base.SCORE_DATASETS)
    score.add_argument("--seed", type=int, choices=base.SEEDS)
    score.add_argument("--device", default="cuda")
    score.add_argument("--num-workers", type=int, default=base.TRAINING_NUM_WORKERS)
    score.set_defaults(func=cmd_score)

    report = commands.add_parser("report")
    report.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    report.add_argument("--n-bootstrap", type=int, default=base.N_BOOTSTRAP)
    report.add_argument("--e1a-n-bootstrap", type=int, default=base.E1A_N_BOOTSTRAP)
    report.set_defaults(func=cmd_report)

    verify = commands.add_parser("verify")
    verify.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
