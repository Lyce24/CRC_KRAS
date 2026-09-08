#!/usr/bin/env python3
"""Study-wide governed expansion of MIL model seeds 42--44 to 42--46.

This orchestrator composes three additive components without modifying any
previously sealed result tree:

* Aim 1: the canonical conventional-primary E0 baseline;
* Aim 2: every family, sibling/project, and size-matched LOCO arm;
* Aim 3: every fixed, repeated-control, and encoder ladder arm.

Only seeds 45 and 46 are trained.  All five model seeds reuse each experiment's
single frozen patient/slide fold layout.  The scheduler owns GPU 0 for the full
pool and permits at most six independent trainer processes at once.  Model
seeds are ensemble members, never inferential resampling units.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import aim2_loco_five_seed_extension as aim2  # noqa: E402
import aim3_ladders_five_seed_extension as aim3  # noqa: E402
from tools import aim1_e0_five_seed_extension as aim1  # noqa: E402

SCHEMA_VERSION = 1
MAX_CONCURRENT_GPU_TRAINERS = 6
EXPECTED_JOB_COUNT = 102
EXPECTED_NEW_FITS = 490

DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "final_v9_mil_5seed_expansion_v1_20260823"
)
GPU_LOCK_PATH = Path("/tmp/oceanpath_gpu0_exclusive.lock")
CAMPAIGN_LOCK_PATH = Path("/tmp/oceanpath_final_v9_mil_5seed_campaign.lock")
WORKER_AUTH_ENV = "OCEANPATH_FINALV9_5SEED_WORKER_AUTH"
GPU_LEASE_FD_ENV = "OCEANPATH_FINALV9_5SEED_GPU_LEASE_FD"

_LIBC = ctypes.CDLL(None, use_errno=True)


class ContractError(RuntimeError):
    """Fail-closed campaign contract violation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ContractError(f"Expected file artifact: {path}")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_strict_pairs)
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object: {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_json_once(path: Path, value: Any) -> None:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ContractError(f"Refusing to replace different immutable JSON: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if path.read_bytes() != payload:
                raise ContractError(
                    f"Concurrent immutable JSON collision: {path}"
                ) from exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_root(path: Path, *, must_exist: bool = False) -> Path:
    raw = Path(path)
    resolved = raw.resolve(strict=False)
    production = DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    if resolved != production and not _is_under(resolved, Path("/tmp").resolve()):
        raise ContractError(
            f"Production campaign root must be exactly {production}; tests may use /tmp"
        )
    if raw.is_symlink() or (resolved.exists() and resolved.is_symlink()):
        raise ContractError(f"Campaign root may not be a symlink: {raw}")
    if must_exist and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def aim1_root(root: Path) -> Path:
    return root / "aim1_e0"


def aim2_api_root(root: Path) -> Path:
    """Aim-2 accepts the shared lineage root and appends ``aim2_loco`` itself."""

    return root


def aim3_root(root: Path) -> Path:
    return root / "aim3_ladders"


def _contract_path(root: Path) -> Path:
    return root / "campaign/experiment_contract.json"


def _preflight_path(root: Path) -> Path:
    return root / "campaign/receipts/deep_preflight.json"


def _training_receipt_path(root: Path) -> Path:
    return root / "campaign/receipts/training_complete.json"


def _final_receipt_path(root: Path) -> Path:
    return root / "campaign/receipts/five_seed_results_complete.json"


def _component_contract_paths(root: Path) -> dict[str, Path]:
    return {
        "aim1_e0": aim1._contract_path(aim1_root(root)),
        "aim2_loco": aim2.contract_path(aim2_api_root(root)),
        "aim3_ladders": aim3_root(root) / "inputs/experiment_contract.json",
    }


def _normalize_jobs(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in aim1.build_job_inventory(aim1_root(root)):
        records.append(
            {
                "job_id": f"aim1.{raw['key']}",
                "component": "aim1_e0",
                "stage": "baseline",
                "seed": int(raw["seed"]),
                "fit_count": 6,
                "output": str(raw["run_dir"]),
                "component_job_key": str(raw["key"]),
                "command": list(raw["external_worker_command"]),
                "priority": 0,
            }
        )
    for raw in aim2.build_training_jobs(aim2_api_root(root)):
        records.append(
            {
                "job_id": str(raw["job_id"]),
                "component": "aim2_loco",
                "stage": str(raw["stage"]),
                "seed": int(raw["seed"]),
                "arm": str(raw["arm"]),
                "fit_count": int(raw["fit_count"]),
                "output": str(raw["output"]),
                "component_job_key": str(raw["job_id"]),
                "command": list(raw["external_worker_command"]),
                "priority": 1 if raw["stage"] == "source_cv" else 4,
            }
        )
    for raw in aim3.build_training_jobs(aim3_root(root)):
        stage = str(raw["stage"])
        records.append(
            {
                "job_id": str(raw["job_id"]),
                "component": "aim3_ladders",
                "stage": stage,
                "seed": int(raw["seed"]),
                "task": str(raw["task"]),
                "draw_seed": raw.get("draw_seed"),
                "fit_count": int(raw["fit_count"]),
                "output": str(raw["output"]),
                "component_job_key": str(raw["job_key"]),
                "command": list(raw["external_worker_command"]),
                "priority": {"e1v": 0, "e3v": 1, "fixed": 2, "repeated": 3}[stage],
            }
        )
    records.sort(
        key=lambda job: (
            int(job["priority"]),
            -int(job["fit_count"]),
            str(job["component"]),
            str(job["job_id"]),
        )
    )
    keys = [str(job["job_id"]) for job in records]
    outputs = [str(job["output"]) for job in records]
    if len(records) != EXPECTED_JOB_COUNT or len(set(keys)) != len(keys):
        raise ContractError("Cross-aim inventory is not exactly 102 unique jobs")
    if len(set(outputs)) != len(outputs):
        raise ContractError("Two cross-aim jobs share an output directory")
    if sum(int(job["fit_count"]) for job in records) != EXPECTED_NEW_FITS:
        raise ContractError("Cross-aim inventory is not exactly 490 new MIL fits")
    if {int(job["seed"]) for job in records} != {45, 46}:
        raise ContractError("Cross-aim scheduler may train only model seeds 45 and 46")
    return records


def build_job_inventory(output_root: Path = DEFAULT_OUTPUT_ROOT) -> list[dict[str, Any]]:
    """Public, side-effect-free exact cross-aim job inventory."""

    root = validate_output_root(Path(output_root), must_exist=False)
    return _normalize_jobs(root)


def _component_identities(root: Path) -> dict[str, dict[str, Any]]:
    return {name: _artifact(path) for name, path in _component_contract_paths(root).items()}


def _controller_identities() -> dict[str, dict[str, Any]]:
    return {
        "campaign": _artifact(Path(__file__)),
        "aim1_e0": _artifact(Path(aim1.__file__)),
        "aim2_loco": _artifact(Path(aim2.__file__)),
        "aim3_ladders": _artifact(Path(aim3.__file__)),
    }


def _build_contract(root: Path) -> dict[str, Any]:
    jobs = _normalize_jobs(root)
    by_component = {
        component: {
            "jobs": sum(job["component"] == component for job in jobs),
            "new_fits": sum(
                int(job["fit_count"]) for job in jobs if job["component"] == component
            ),
        }
        for component in ("aim1_e0", "aim2_loco", "aim3_ladders")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_before_new_fit",
        "created_utc": _utc_now(),
        "experiment": "final-v9 study-wide MIL five-seed expansion",
        "output_root": str(root),
        "model_seeds": {"adopted": [42, 43, 44], "new": [45, 46], "complete": [42, 43, 44, 45, 46]},
        "split_policy": {
            "outer_and_inner_membership_changes_across_model_seeds": False,
            "rule": "reuse each experiment's exact frozen patient/slide fold manifest for seeds 42-46",
        },
        "execution": {
            "gpu_index": 0,
            "max_concurrent_gpu_trainers": MAX_CONCURRENT_GPU_TRAINERS,
            "global_gpu_lock": str(GPU_LOCK_PATH),
            "fresh_process_per_chain": True,
            "child_parent_death_signal": "Linux PR_SET_PDEATHSIG SIGTERM",
            "catchable_termination_reaps_every_worker_process_group": True,
        },
        "counts": {
            "jobs": len(jobs),
            "new_mil_fits": sum(int(job["fit_count"]) for job in jobs),
            "by_component": by_component,
        },
        "component_contracts": _component_identities(root),
        "controllers": _controller_identities(),
        "jobs": jobs,
    }


def _load_contract(root: Path) -> dict[str, Any]:
    stored = _read_json(_contract_path(root))
    expected = _build_contract(root)
    expected["created_utc"] = stored.get("created_utc")
    if stored != expected:
        mismatch = {
            key: {"expected": expected.get(key), "observed": stored.get(key)}
            for key in sorted(set(expected) | set(stored))
            if expected.get(key) != stored.get(key)
        }
        raise ContractError(f"Campaign contract no longer replays exactly: {mismatch}")
    return stored


def _control_command(root: Path, component: str, operation: str) -> list[str]:
    if component == "aim1_e0":
        return [
            sys.executable,
            str(Path(aim1.__file__).resolve()),
            "--campaign-root",
            str(aim1_root(root)),
            operation,
        ]
    if component == "aim2_loco":
        command = [
            sys.executable,
            str(Path(aim2.__file__).resolve()),
            operation,
            "--output-root",
            str(aim2_api_root(root)),
        ]
        if operation in {"manifest", "report", "score"}:
            command.append("--apply")
        if operation in {"preflight", "verify", "seal-inference"}:
            command.append("--deep")
        if operation == "preflight":
            command.append("--apply")
        return command
    if component == "aim3_ladders":
        command = [
            sys.executable,
            str(Path(aim3.__file__).resolve()),
            operation,
            "--output-root",
            str(aim3_root(root)),
        ]
        if operation in {"manifest", "preflight", "analyze", "verify"}:
            command.extend(["--e0-root", str(aim1_root(root))])
        if operation in {"manifest", "preflight", "analyze"}:
            command.append("--apply")
        if operation == "preflight":
            command.append("--full-pack-check")
        return command
    raise ValueError(component)


def _run_control(
    root: Path,
    label: str,
    command: Sequence[str],
    *,
    gpu_lock_fd: int | None = None,
) -> Path:
    directory = root / "campaign/logs/control"
    directory.mkdir(parents=True, exist_ok=True)
    attempt = len(list(directory.glob(f"{label}__*.log"))) + 1
    path = directory / f"{label}__attempt-{attempt:03d}.log"
    with path.open("x", encoding="utf-8", buffering=1) as log:
        wrapper = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_control_worker",
            "--command-json",
            json.dumps(list(command)),
        ]
        parent_pid = os.getpid()
        process = subprocess.Popen(
            wrapper,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "MPLCONFIGDIR": "/tmp/oceanpath-mpl",
                "PYTHONUNBUFFERED": "1",
                **(
                    {}
                    if gpu_lock_fd is None
                    else {GPU_LEASE_FD_ENV: str(gpu_lock_fd)}
                ),
            },
            start_new_session=True,
            pass_fds=(() if gpu_lock_fd is None else (gpu_lock_fd,)),
            preexec_fn=lambda: _configure_parent_death(parent_pid),
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            returncode = int(process.wait())
            orphaned = _process_group_members(process.pid)
            if orphaned:
                _terminate_process_group_members(process.pid)
                if returncode == 0:
                    returncode = 70
        except BaseException:
            _terminate_group(process)
            raise
    if returncode:
        raise ContractError(f"Control stage {label} failed rc={returncode}; see {path}")
    return path


def _run_readonly(command: Sequence[str]) -> None:
    result = subprocess.run(list(command), cwd=REPO, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise ContractError(f"Read-only verification failed rc={result.returncode}: {command}")


def _available_ram_gib() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    return values.get("MemAvailable", 0) / 1024**2


def _gpu_resources() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=name,memory.total,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if query.returncode:
        raise ContractError(f"GPU query failed: {query.stderr.strip()}")
    fields = [value.strip() for value in query.stdout.strip().splitlines()[0].split(",")]
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if apps.returncode:
        raise ContractError(f"GPU process query failed: {apps.stderr.strip()}")
    active = [line.strip() for line in apps.stdout.splitlines() if line.strip()]
    return {
        "name": fields[0],
        "total_gpu_mib": int(fields[1]),
        "free_gpu_mib": int(fields[2]),
        "gpu_utilization_percent": int(fields[3]),
        "temperature_c": int(fields[4]),
        "active_compute_apps": active,
    }


def _resource_admission(root: Path) -> dict[str, Any]:
    parent = root
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    disk = shutil.disk_usage(parent)
    observed = {
        "logical_cpus": int(os.cpu_count() or 0),
        "available_ram_gib": _available_ram_gib(),
        "free_disk_gib": disk.free / 1024**3,
        **_gpu_resources(),
    }
    minima = {
        "logical_cpus": 36,
        "available_ram_gib": 96.0,
        "free_disk_gib": 100.0,
        "free_gpu_mib": 20_000,
    }
    failures = {
        key: {"minimum": minimum, "observed": observed[key]}
        for key, minimum in minima.items()
        if float(observed[key]) < minimum
    }
    if observed["active_compute_apps"]:
        failures["active_compute_apps"] = {
            "expected": [],
            "observed": observed["active_compute_apps"],
        }
    if int(observed["gpu_utilization_percent"]) > 20:
        failures["gpu_utilization_percent"] = {
            "maximum": 20,
            "observed": observed["gpu_utilization_percent"],
        }
    if failures:
        raise ContractError(f"Six-trainer resource admission failed: {failures}")
    return observed


def _validate_preflight(root: Path) -> dict[str, Any]:
    receipt = _read_json(_preflight_path(root))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "deep_preflight_passed",
        "contract": _artifact(_contract_path(root)),
        "aim2_preflight": _artifact(aim2.preflight_receipt_path(aim2_api_root(root))),
        "aim3_preflight": _artifact(aim3_root(root) / "receipts/preflight.json"),
        "maximum_concurrent_gpu_trainers": MAX_CONCURRENT_GPU_TRAINERS,
    }
    mismatch = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatch:
        raise ContractError(f"Deep-preflight receipt mismatch: {mismatch}")
    return receipt


@contextlib.contextmanager
def _exclusive_lock(path: Path, operation: str) -> Iterator[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f"Another process holds {operation}: {path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()} operation={operation} utc={_utc_now()}\n")
        stream.flush()
        try:
            yield stream.fileno()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _termination_as_exception() -> Iterator[None]:
    watched = (signal.SIGTERM, signal.SIGHUP)
    prior = {signum: signal.getsignal(signum) for signum in watched}

    def handler(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    for signum in watched:
        signal.signal(signum, handler)
    try:
        yield
    finally:
        for signum, previous in prior.items():
            signal.signal(signum, previous)


def _configure_parent_death(parent_pid: int) -> None:
    if _LIBC.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        os._exit(127)
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def _terminate_group(process: subprocess.Popen[Any], timeout: float = 30.0) -> int:
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    try:
        return int(process.wait(timeout=timeout))
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        return int(process.wait())


def _process_group_members(process_group: int, *, exclude: int | None = None) -> list[int]:
    """Return live Linux PIDs in one process group, excluding the supervisor."""

    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == exclude:
            continue
        try:
            # Everything after the final ')' starts at stat field 3 (state);
            # process group is field 5, hence index 2 in this remainder.
            remainder = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if remainder[0] != "Z" and int(remainder[2]) == process_group:
                members.append(pid)
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            continue
    return members


def _terminate_process_group_members(
    process_group: int,
    *,
    exclude: int | None = None,
    timeout: float = 30.0,
) -> None:
    """Terminate and then kill every descendant that remains in a governed group."""

    deadline = time.monotonic() + timeout
    while True:
        members = _process_group_members(process_group, exclude=exclude)
        if not members:
            return
        for pid in members:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    for pid in _process_group_members(process_group, exclude=exclude):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def _lease_fd_from_environment(*, required: bool) -> int | None:
    raw = os.environ.get(GPU_LEASE_FD_ENV)
    if raw is None:
        if required:
            raise ContractError("Authorized worker did not inherit the GPU campaign lease")
        return None
    try:
        descriptor = int(raw)
        os.fstat(descriptor)
    except (ValueError, OSError) as exc:
        raise ContractError("Inherited GPU campaign lease descriptor is invalid") from exc
    return descriptor


def _supervise_component(
    command: Sequence[str],
    *,
    environment: dict[str, str],
    gpu_lease_fd: int | None,
) -> None:
    """Keep a signal-aware supervisor alive around one same-group component tree."""

    process: subprocess.Popen[Any] | None = None
    try:
        with _termination_as_exception():
            # Install the parent-death handler before spawning so there is no
            # Popen→handler race in which a component could outlive its wrapper.
            process = subprocess.Popen(
                list(command),
                cwd=REPO,
                env=environment,
                start_new_session=False,
                pass_fds=(() if gpu_lease_fd is None else (gpu_lease_fd,)),
            )
            returncode = int(process.wait())
    except BaseException:
        _terminate_process_group_members(os.getpgrp(), exclude=os.getpid())
        if process is not None:
            with contextlib.suppress(Exception):
                process.wait(timeout=1)
        raise
    orphaned = _process_group_members(os.getpgrp(), exclude=os.getpid())
    if orphaned:
        _terminate_process_group_members(os.getpgrp(), exclude=os.getpid())
        raise RuntimeError(f"Component returned rc0 with live descendants: {orphaned}")
    if returncode:
        # A controller can be killed before it reaps its native trainer.  Clear
        # any remaining same-group descendants before exposing the failure.
        _terminate_process_group_members(os.getpgrp(), exclude=os.getpid())
        raise SystemExit(returncode)


def _require_worker_authorization(contract_sha256: str) -> None:
    token = os.environ.get(WORKER_AUTH_ENV, "")
    digest = os.environ.get(f"{WORKER_AUTH_ENV}_SHA256", "")
    bound = os.environ.get(f"{WORKER_AUTH_ENV}_CONTRACT", "")
    if not token or hashlib.sha256(token.encode()).hexdigest() != digest:
        raise ContractError("Internal campaign worker lacks scheduler authorization")
    if bound != contract_sha256:
        raise ContractError("Internal campaign worker binds a different contract")


def _job_by_id(contract: dict[str, Any], job_id: str) -> dict[str, Any]:
    matches = [job for job in contract["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise ContractError(f"Unknown or ambiguous campaign job: {job_id}")
    return matches[0]


def cmd_worker(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    contract = _load_contract(root)
    identity = _artifact(_contract_path(root))
    if identity["sha256"] != args.contract_sha256:
        raise ContractError("Worker contract SHA differs from scheduler request")
    _require_worker_authorization(args.contract_sha256)
    job = _job_by_id(contract, args.job_id)
    component_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(WORKER_AUTH_ENV) and key != GPU_LEASE_FD_ENV
    }
    environment = {
        **component_environment,
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/oceanpath-mpl",
        "PYTHONUNBUFFERED": "1",
    }
    _supervise_component(
        list(job["command"]),
        environment=environment,
        gpu_lease_fd=_lease_fd_from_environment(required=True),
    )


def cmd_control_worker(args: argparse.Namespace) -> None:
    """Contain one control-stage process tree under parent-death cleanup."""

    command = json.loads(args.command_json)
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ContractError("Control worker command must be a JSON string array")
    allowed = {
        str(Path(aim1.__file__).resolve()),
        str(Path(aim2.__file__).resolve()),
        str(Path(aim3.__file__).resolve()),
    }
    if len(command) < 2 or str(Path(command[1]).resolve()) not in allowed:
        raise ContractError("Control worker may invoke only the three governed controllers")
    environment = {key: value for key, value in os.environ.items() if key != GPU_LEASE_FD_ENV}
    _supervise_component(
        command,
        environment=environment,
        gpu_lease_fd=_lease_fd_from_environment(required=False),
    )


def _safe_job_name(job_id: str) -> str:
    return job_id.replace("/", "__").replace(":", "_")


def _job_receipt_path(root: Path, job_id: str) -> Path:
    return root / "campaign/receipts/jobs" / f"{_safe_job_name(job_id)}.json"


def _job_log_dir(root: Path, job_id: str) -> Path:
    return root / "campaign/logs/training" / _safe_job_name(job_id)


def _launch_request_payload(
    root: Path,
    job: dict[str, Any],
    *,
    attempt: int,
    started_utc: str,
    recovered_before_component_spawn: bool = False,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "authorized_by_global_max6_scheduler",
        "created_utc": started_utc,
        "started_utc": started_utc,
        "attempt": attempt,
        "job_id": job["job_id"],
        "component": job["component"],
        "component_job_key": job["component_job_key"],
        "contract": _artifact(_contract_path(root)),
        "command": job["command"],
        "output": job["output"],
        "maximum_concurrent_gpu_trainers": MAX_CONCURRENT_GPU_TRAINERS,
    }
    if recovered_before_component_spawn:
        payload["recovered_after_log_before_request_crash"] = True
    return payload


def _reconcile_launch_publication(root: Path, job: dict[str, Any]) -> None:
    """Close the log-create→request-publish crash window before dispatch."""

    directory = _job_log_dir(root, str(job["job_id"]))
    if not directory.exists():
        return
    for log in sorted(directory.glob("attempt-*.log")):
        attempt = int(log.stem.split("-")[-1])
        request = directory / f"attempt-{attempt:03d}.request.json"
        if request.exists():
            continue
        if (directory / f"attempt-{attempt:03d}.exit.json").exists():
            raise ContractError(f"Job exit exists without a launch request: {log}")
        started = datetime.fromtimestamp(log.stat().st_mtime, timezone.utc).isoformat()
        _write_json_once(
            request,
            _launch_request_payload(
                root,
                job,
                attempt=attempt,
                started_utc=started,
                recovered_before_component_spawn=True,
            ),
        )


def _validate_launch_request(
    root: Path, job: dict[str, Any], path: Path
) -> dict[str, Any]:
    request = _read_json(path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "authorized_by_global_max6_scheduler",
        "job_id": job["job_id"],
        "component": job["component"],
        "component_job_key": job["component_job_key"],
        "contract": _artifact(_contract_path(root)),
        "command": job["command"],
        "output": job["output"],
        "maximum_concurrent_gpu_trainers": MAX_CONCURRENT_GPU_TRAINERS,
    }
    allowed_keys = {
        *expected,
        "created_utc",
        "started_utc",
        "attempt",
        "recovered_after_log_before_request_crash",
    }
    mismatch = {
        key: {"expected": value, "observed": request.get(key)}
        for key, value in expected.items()
        if request.get(key) != value
    }
    try:
        path_attempt = int(path.name.removeprefix("attempt-").removesuffix(".request.json"))
    except ValueError:
        path_attempt = -1
    if (
        mismatch
        or set(request) - allowed_keys
        or not isinstance(request.get("attempt"), int)
        or request.get("attempt") != path_attempt
        or not request.get("started_utc")
        or request.get("created_utc") != request.get("started_utc")
        or not isinstance(
            request.get("recovered_after_log_before_request_crash", False), bool
        )
    ):
        raise ContractError(f"Invalid global launch request {path}: {mismatch}")
    return request


def _job_attempt_ledger(root: Path, job: dict[str, Any]) -> list[dict[str, Any]]:
    directory = _job_log_dir(root, str(job["job_id"]))
    request_paths = sorted(directory.glob("attempt-*.request.json"))
    log_paths = sorted(directory.glob("attempt-*.log"))
    if {path.name.removesuffix(".request.json") for path in request_paths} != {
        path.stem for path in log_paths
    }:
        raise ContractError(f"Launch request/log census differs for {job['job_id']}")
    ledger: list[dict[str, Any]] = []
    for request_path in request_paths:
        request = _validate_launch_request(root, job, request_path)
        attempt = int(request["attempt"])
        stem = directory / f"attempt-{attempt:03d}"
        log = stem.with_suffix(".log")
        exit_path = stem.with_suffix(".exit.json")
        if not log.is_file():
            raise ContractError(f"Launch request lacks its immutable log: {request_path}")
        record: dict[str, Any] = {
            "attempt": attempt,
            "request": _artifact(request_path),
            "log": _artifact(log),
            "exit": _artifact(exit_path) if exit_path.is_file() else None,
        }
        ledger.append(record)
    attempts = [int(record["attempt"]) for record in ledger]
    if attempts != list(range(1, len(attempts) + 1)):
        raise ContractError(f"Non-contiguous launch-attempt ledger for {job['job_id']}")
    return ledger


def _component_job_evidence(root: Path, job: dict[str, Any]) -> dict[str, Any]:
    component = job["component"]
    if component == "aim1_e0":
        seed = int(job["seed"])
        manifest, _splits, _records = aim1._load_manifest_and_splits(
            aim1.DEFAULT_MANIFEST.resolve(strict=True),
            aim1.DEFAULT_SPLIT_DIR.resolve(strict=True),
        )
        validated = aim1._validate_run(
            Path(job["output"]),
            manifest,
            seed,
            aim1.EXPECTED_NEW_FINGERPRINTS[seed],
        )
        return {"native_artifacts": validated["artifacts"]}
    if component == "aim2_loco":
        raw = {
            value["job_id"]: value
            for value in aim2.build_training_jobs(aim2_api_root(root))
        }[job["component_job_key"]]
        receipt = aim2._validate_scheduler_receipt(aim2_api_root(root), raw)
        if receipt is None:
            raise ContractError(f"Aim2 component receipt is missing: {job['job_id']}")
        path = aim2._scheduler_receipt_path(
            aim2_api_root(root), job["component_job_key"]
        )
        return {"component_receipt": _artifact(path)}
    if component == "aim3_ladders":
        raw = aim3.job_from_key(str(job["component_job_key"]))
        aim3._validate_job_receipt(aim3_root(root), raw)
        return {"component_receipt": _artifact(aim3.job_receipt_path(aim3_root(root), raw))}
    raise ValueError(component)


def _validate_top_job_receipt(root: Path, job: dict[str, Any]) -> dict[str, Any] | None:
    path = _job_receipt_path(root, str(job["job_id"]))
    if not path.exists():
        return None
    receipt = _read_json(path)
    exit_identity = receipt.get("exit")
    log_identity = receipt.get("log")
    request_identity = receipt.get("launch_request")
    if (
        not isinstance(exit_identity, dict)
        or not isinstance(log_identity, dict)
        or not isinstance(request_identity, dict)
    ):
        raise ContractError(f"Malformed campaign job receipt: {path}")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_rc0",
        "job_id": job["job_id"],
        "returncode": 0,
        "contract": _artifact(_contract_path(root)),
        "command": job["command"],
        "launch_request": _artifact(Path(request_identity["path"])),
        "exit": _artifact(Path(exit_identity["path"])),
        "log": _artifact(Path(log_identity["path"])),
        "attempt_ledger": _job_attempt_ledger(root, job),
        "component_evidence": _component_job_evidence(root, job),
    }
    mismatch = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatch:
        raise ContractError(f"Campaign job receipt mismatch {job['job_id']}: {mismatch}")
    exit_record = _read_json(Path(exit_identity["path"]))
    if exit_record.get("returncode") != 0 or exit_record.get("job_id") != job["job_id"]:
        raise ContractError(f"Campaign job lacks a matching rc0 exit: {job['job_id']}")
    request = _validate_launch_request(root, job, Path(receipt["launch_request"]["path"]))
    if exit_record.get("request") != receipt["launch_request"]:
        raise ContractError(f"Job exit is not bound to its launch request: {job['job_id']}")
    if exit_record.get("started_utc") != request.get("started_utc"):
        raise ContractError(f"Job exit/request start times differ: {job['job_id']}")
    return receipt


@dataclasses.dataclass
class RunningJob:
    job: dict[str, Any]
    process: subprocess.Popen[str]
    log_path: Path
    log_stream: TextIO
    request_path: Path
    attempt: int
    started_utc: str


def _launch_job(
    root: Path,
    job: dict[str, Any],
    contract_sha256: str,
    session_token: str,
    gpu_lock_fd: int,
) -> RunningJob:
    directory = _job_log_dir(root, str(job["job_id"]))
    directory.mkdir(parents=True, exist_ok=True)
    _reconcile_launch_publication(root, job)
    attempt = len(list(directory.glob("attempt-*.log"))) + 1
    log_path = directory / f"attempt-{attempt:03d}.log"
    prior_requests = sorted(directory.glob("attempt-*.request.json"))
    for prior in prior_requests:
        _validate_launch_request(root, job, prior)
    if Path(str(job["output"])).exists() and not prior_requests:
        raise ContractError(
            f"Component output exists without a prior global six-slot launch request: "
            f"{job['job_id']} -> {job['output']}"
        )
    log_stream = log_path.open("x", encoding="utf-8", buffering=1)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--output-root",
        str(root),
        "--job-id",
        str(job["job_id"]),
        "--contract-sha256",
        contract_sha256,
    ]
    environment = {
        **os.environ,
        WORKER_AUTH_ENV: session_token,
        f"{WORKER_AUTH_ENV}_SHA256": hashlib.sha256(session_token.encode()).hexdigest(),
        f"{WORKER_AUTH_ENV}_CONTRACT": contract_sha256,
        GPU_LEASE_FD_ENV: str(gpu_lock_fd),
        "PYTHONUNBUFFERED": "1",
    }
    parent_pid = os.getpid()
    started_utc = _utc_now()
    request_path = directory / f"attempt-{attempt:03d}.request.json"
    _write_json_once(
        request_path,
        _launch_request_payload(
            root,
            job,
            attempt=attempt,
            started_utc=started_utc,
        ),
    )
    process = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        start_new_session=True,
        pass_fds=(gpu_lock_fd,),
        preexec_fn=lambda: _configure_parent_death(parent_pid),
    )
    print(
        f"launched {job['job_id']} pid={process.pid} fit_count={job['fit_count']} log={log_path}",
        flush=True,
    )
    return RunningJob(job, process, log_path, log_stream, request_path, attempt, started_utc)


def _record_exit(root: Path, active: RunningJob, returncode: int) -> Path:
    active.log_stream.flush()
    active.log_stream.close()
    path = active.log_path.with_suffix(".exit.json")
    _write_json_once(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "job_id": active.job["job_id"],
            "attempt": active.attempt,
            "started_utc": active.started_utc,
            "returncode": int(returncode),
            "finished_utc": _utc_now(),
            "request": _artifact(active.request_path),
            "log": _artifact(active.log_path),
        },
    )
    return path


def _ensure_top_job_receipt(
    root: Path,
    active: RunningJob,
    exit_path: Path,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_rc0",
        "created_utc": _utc_now(),
        "job_id": active.job["job_id"],
        "returncode": 0,
        "contract": _artifact(_contract_path(root)),
        "command": active.job["command"],
        "launch_request": _artifact(active.request_path),
        "exit": _artifact(exit_path),
        "log": _artifact(active.log_path),
        "attempt_ledger": _job_attempt_ledger(root, active.job),
        "component_evidence": _component_job_evidence(root, active.job),
    }
    path = _job_receipt_path(root, str(active.job["job_id"]))
    _write_json_once(path, payload)
    validated = _validate_top_job_receipt(root, active.job)
    assert validated is not None
    return validated


def _terminate_and_reap(root: Path, running: dict[str, RunningJob]) -> None:
    for active in running.values():
        if active.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(active.process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 45.0
    for active in running.values():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            returncode = int(active.process.wait(timeout=remaining))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(active.process.pid, signal.SIGKILL)
            returncode = int(active.process.wait())
        if not active.log_stream.closed:
            with contextlib.suppress(Exception):
                _record_exit(root, active, returncode)


def _completed_job_ids(root: Path, jobs: list[dict[str, Any]]) -> set[str]:
    completed: set[str] = set()
    for job in jobs:
        if _validate_top_job_receipt(root, job) is not None:
            completed.add(str(job["job_id"]))
    return completed


def _run_scheduler(root: Path, gpu_lock_fd: int) -> dict[str, Any]:
    contract = _load_contract(root)
    jobs = list(contract["jobs"])
    completed = _completed_job_ids(root, jobs)
    pending = [job for job in jobs if job["job_id"] not in completed]
    running: dict[str, RunningJob] = {}
    failures: list[tuple[str, int]] = []
    token = secrets.token_hex(32)
    contract_sha = _artifact(_contract_path(root))["sha256"]
    peak = 0
    launched = 0
    start = _utc_now()
    try:
        while pending or running:
            for job_id, active in list(running.items()):
                returncode = active.process.poll()
                if returncode is None:
                    continue
                orphaned = _process_group_members(active.process.pid)
                if orphaned:
                    _terminate_process_group_members(active.process.pid)
                    if returncode == 0:
                        returncode = 70
                exit_path = _record_exit(root, active, int(returncode))
                del running[job_id]
                if returncode:
                    failures.append((job_id, int(returncode)))
                    print(f"FAILED {job_id} rc={returncode}; stopping new dispatch", flush=True)
                    continue
                _ensure_top_job_receipt(root, active, exit_path)
                completed.add(job_id)
                print(
                    f"completed {job_id}; {len(completed)}/{len(jobs)} jobs certified",
                    flush=True,
                )
            if not failures:
                while pending and len(running) < MAX_CONCURRENT_GPU_TRAINERS:
                    job = pending.pop(0)
                    active = _launch_job(root, job, contract_sha, token, gpu_lock_fd)
                    running[str(job["job_id"])] = active
                    launched += 1
                    peak = max(peak, len(running))
            if failures and not running:
                raise RuntimeError(f"Campaign training jobs failed: {failures}")
            if running:
                time.sleep(1.0)
        if len(completed) != EXPECTED_JOB_COUNT:
            raise ContractError(f"Training ended with only {len(completed)}/102 jobs")
    except BaseException:
        _terminate_and_reap(root, running)
        raise
    return {
        "started_utc": start,
        "finished_utc": _utc_now(),
        "jobs_launched_this_session": launched,
        "jobs_certified_total": len(completed),
        "observed_peak_concurrent_gpu_trainers": peak,
    }


def _reconstruct_peak_from_job_exits(root: Path) -> int:
    """Recover concurrency evidence from immutable per-job time intervals."""

    contract = _load_contract(root)
    events: list[tuple[datetime, int]] = []
    for job in contract["jobs"]:
        receipt = _read_json(_job_receipt_path(root, job["job_id"]))
        exit_record = _read_json(Path(receipt["exit"]["path"]))
        started = datetime.fromisoformat(str(exit_record["started_utc"]))
        finished = datetime.fromisoformat(str(exit_record["finished_utc"]))
        if finished < started:
            raise ContractError(f"Job exit interval is negative: {job['job_id']}")
        events.extend([(started, 1), (finished, -1)])
    active = 0
    peak = 0
    # At identical timestamps, count an exit before a new launch.  This avoids
    # inventing overlap at a scheduler handoff boundary.
    for _when, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        if active < 0:
            raise ContractError("Job exit intervals cannot be ordered consistently")
        peak = max(peak, active)
    if active != 0 or not 1 <= peak <= MAX_CONCURRENT_GPU_TRAINERS:
        raise ContractError(f"Reconstructed concurrency is invalid: active={active}, peak={peak}")
    return peak


def _write_training_receipt(root: Path, session: dict[str, Any]) -> None:
    session_dir = root / "campaign/receipts/pool_sessions"
    session_path = session_dir / f"session-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    _write_json_once(
        session_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "contract": _artifact(_contract_path(root)),
            **session,
        },
    )
    contract = _load_contract(root)
    job_receipts = {
        job["job_id"]: _artifact(_job_receipt_path(root, job["job_id"]))
        for job in contract["jobs"]
    }
    sessions = sorted(session_dir.glob("session-*.json"))
    records = [_read_json(path) for path in sessions]
    observed_peak = max(
        int(record.get("observed_peak_concurrent_gpu_trainers", 0)) for record in records
    )
    reconstructed_peak = _reconstruct_peak_from_job_exits(root)
    if not 0 <= observed_peak <= MAX_CONCURRENT_GPU_TRAINERS:
        raise ContractError("Pool-session concurrency evidence is invalid")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "490 new MIL fits completed and certified",
        "contract": _artifact(_contract_path(root)),
        "preflight": _artifact(_preflight_path(root)),
        "job_receipts": job_receipts,
        "pool_sessions": [_artifact(path) for path in sessions],
        "new_mil_fits": EXPECTED_NEW_FITS,
        "new_training_jobs": EXPECTED_JOB_COUNT,
        "maximum_concurrent_gpu_trainers": MAX_CONCURRENT_GPU_TRAINERS,
        "observed_peak_concurrent_gpu_trainers": max(observed_peak, reconstructed_peak),
        "reconstructed_peak_from_job_exit_intervals": reconstructed_peak,
    }
    _write_json_once(_training_receipt_path(root), payload)


def _validate_training_receipt(root: Path) -> dict[str, Any]:
    contract = _load_contract(root)
    stored = _read_json(_training_receipt_path(root))
    job_receipts: dict[str, dict[str, Any]] = {}
    for job in contract["jobs"]:
        if _validate_top_job_receipt(root, job) is None:
            raise ContractError(f"Missing certified campaign job: {job['job_id']}")
        job_receipts[job["job_id"]] = _artifact(_job_receipt_path(root, job["job_id"]))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "490 new MIL fits completed and certified",
        "contract": _artifact(_contract_path(root)),
        "preflight": _artifact(_preflight_path(root)),
        "job_receipts": job_receipts,
        "new_mil_fits": EXPECTED_NEW_FITS,
        "new_training_jobs": EXPECTED_JOB_COUNT,
        "maximum_concurrent_gpu_trainers": MAX_CONCURRENT_GPU_TRAINERS,
    }
    mismatch = {
        key: {"expected": value, "observed": stored.get(key)}
        for key, value in expected.items()
        if stored.get(key) != value
    }
    if mismatch:
        raise ContractError(f"Campaign training receipt mismatch: {mismatch}")
    peak = int(stored.get("observed_peak_concurrent_gpu_trainers", -1))
    if not 1 <= peak <= MAX_CONCURRENT_GPU_TRAINERS:
        raise ContractError("Invalid observed training concurrency")
    reconstructed_peak = _reconstruct_peak_from_job_exits(root)
    if stored.get("reconstructed_peak_from_job_exit_intervals") != reconstructed_peak:
        raise ContractError("Training receipt concurrency reconstruction changed")
    for identity in stored.get("pool_sessions", []):
        if _artifact(Path(identity["path"])) != identity:
            raise ContractError("Pool-session receipt changed")
    return stored


def cmd_plan(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=False)
    jobs = _normalize_jobs(root)
    by_component = {
        component: {
            "jobs": sum(job["component"] == component for job in jobs),
            "fits": sum(int(job["fit_count"]) for job in jobs if job["component"] == component),
        }
        for component in ("aim1_e0", "aim2_loco", "aim3_ladders")
    }
    print(
        json.dumps(
            {
                "status": "PLAN_ONLY",
                "output_root": str(root),
                "adopted_seeds": [42, 43, 44],
                "new_seeds": [45, 46],
                "folds_shared_across_all_model_seeds": True,
                "maximum_concurrent_gpu_trainers": MAX_CONCURRENT_GPU_TRAINERS,
                "jobs": len(jobs),
                "new_mil_fits": sum(int(job["fit_count"]) for job in jobs),
                "by_component": by_component,
            },
            indent=2,
        )
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=False)
    paths = _component_contract_paths(root)
    if not paths["aim1_e0"].is_file():
        _run_control(root, "prepare_aim1", _control_command(root, "aim1_e0", "prepare"))
    if not paths["aim2_loco"].is_file():
        _run_control(root, "prepare_aim2", _control_command(root, "aim2_loco", "manifest"))
    if not paths["aim3_ladders"].is_file():
        _run_control(root, "prepare_aim3", _control_command(root, "aim3_ladders", "manifest"))
    if not all(path.is_file() for path in paths.values()):
        raise ContractError("At least one component contract is absent after preparation")
    candidate = _build_contract(root)
    path = _contract_path(root)
    if path.exists():
        _load_contract(root)
    else:
        _write_json_once(path, candidate)
    _load_contract(root)
    print(f"PASS — sealed 102-job/490-fit campaign contract before new training: {path}")


def cmd_preflight(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    with _exclusive_lock(CAMPAIGN_LOCK_PATH, "five-seed campaign preflight"):
        _load_contract(root)
        if _preflight_path(root).is_file():
            _validate_preflight(root)
            print(f"PASS — existing deep preflight revalidated: {_preflight_path(root)}")
            return
        _run_control(root, "preflight_aim1", _control_command(root, "aim1_e0", "preflight"))
        _run_control(root, "preflight_aim2", _control_command(root, "aim2_loco", "preflight"))
        _run_control(root, "preflight_aim3", _control_command(root, "aim3_ladders", "preflight"))
        resources = _resource_admission(root)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "deep_preflight_passed",
            "created_utc": _utc_now(),
            "contract": _artifact(_contract_path(root)),
            "aim2_preflight": _artifact(aim2.preflight_receipt_path(aim2_api_root(root))),
            "aim3_preflight": _artifact(aim3_root(root) / "receipts/preflight.json"),
            "aim1_live_deep_contract_and_pack_authentication": True,
            "resource_admission": resources,
            "maximum_concurrent_gpu_trainers": MAX_CONCURRENT_GPU_TRAINERS,
        }
        _write_json_once(_preflight_path(root), payload)
        _validate_preflight(root)
    print(f"PASS — deep cross-aim preflight and six-trainer resource admission: {_preflight_path(root)}")


def cmd_train(args: argparse.Namespace) -> None:
    if not args.apply:
        raise ContractError("Training requires explicit --apply")
    root = validate_output_root(Path(args.output_root), must_exist=True)
    _load_contract(root)
    _validate_preflight(root)
    if _training_receipt_path(root).is_file():
        _validate_training_receipt(root)
        print("PASS — all training was already complete and has been revalidated")
        return
    if _final_receipt_path(root).exists():
        raise ContractError("Five-seed results are already sealed; refusing training")
    with (
        _exclusive_lock(CAMPAIGN_LOCK_PATH, "five-seed campaign launcher"),
        _exclusive_lock(GPU_LOCK_PATH, "host-wide GPU 0 campaign lease") as gpu_lock_fd,
        _termination_as_exception(),
    ):
        resources = _resource_admission(root)
        print(json.dumps({"resource_admission": resources}, indent=2), flush=True)
        session = _run_scheduler(root, gpu_lock_fd)
        _write_training_receipt(root, session)
    _validate_training_receipt(root)
    print(f"PASS — all 102 jobs / 490 new MIL fits certified: {_training_receipt_path(root)}")


def _build_final_receipt(root: Path) -> dict[str, Any]:
    aim1_files = {
        "training": aim1._training_receipt_path(aim1_root(root)),
        "results": aim1._results_path(aim1_root(root)),
        "analysis_receipt": aim1._analysis_receipt_path(aim1_root(root)),
    }
    aim2_files = {
        "inference_seal": aim2.inference_seal_path(aim2_api_root(root)),
        **{
            f"analysis_{key}": value
            for key, value in aim2._report_paths(aim2_api_root(root)).items()
        },
    }
    aim3_files = {
        "results": aim3_root(root) / "analysis/five_seed_results.json",
        "bootstrap_distributions": aim3_root(root) / "analysis/bootstrap_distributions.npz",
        "analysis_audit": aim3_root(root) / "analysis/analysis_audit.json",
        "completion": aim3_root(root) / "receipts/extension_complete.json",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "five_seed_results_complete",
        "created_utc": _utc_now(),
        "contract": _artifact(_contract_path(root)),
        "training": _artifact(_training_receipt_path(root)),
        "seeds": [42, 43, 44, 45, 46],
        "model_seeds_are_not_inference_units": True,
        "components": {
            "aim1_e0": {name: _artifact(path) for name, path in aim1_files.items()},
            "aim2_loco": {name: _artifact(path) for name, path in aim2_files.items()},
            "aim3_ladders": {name: _artifact(path) for name, path in aim3_files.items()},
        },
    }


def _validate_final_receipt(root: Path) -> dict[str, Any]:
    stored = _read_json(_final_receipt_path(root))
    expected = _build_final_receipt(root)
    for key in ("schema_version", "status", "contract", "training", "seeds", "model_seeds_are_not_inference_units", "components"):
        if stored.get(key) != expected.get(key):
            raise ContractError(f"Final campaign receipt differs at {key}")
    return stored


def cmd_postprocess(args: argparse.Namespace) -> None:
    if not args.apply:
        raise ContractError("Postprocessing requires explicit --apply")
    root = validate_output_root(Path(args.output_root), must_exist=True)
    _validate_training_receipt(root)
    if _final_receipt_path(root).is_file():
        cmd_verify(argparse.Namespace(output_root=root))
        print("PASS — postprocessing was already complete and has been reverified")
        return
    stages: list[tuple[str, list[str]]] = []
    if not aim1._analysis_receipt_path(aim1_root(root)).is_file():
        stages.extend(
            [
                ("aim1_validate", _control_command(root, "aim1_e0", "validate")),
                ("aim1_analyze", _control_command(root, "aim1_e0", "analyze")),
            ]
        )
    else:
        stages.append(("aim1_verify_resume", _control_command(root, "aim1_e0", "verify")))

    aim3_completion = aim3_root(root) / "receipts/extension_complete.json"
    if not aim3_completion.is_file():
        stages.append(("aim3_analyze", _control_command(root, "aim3_ladders", "analyze")))
    else:
        stages.append(("aim3_verify_resume", _control_command(root, "aim3_ladders", "verify")))

    if not aim2.inference_seal_path(aim2_api_root(root)).is_file():
        stages.extend(
            [
                ("aim2_score", _control_command(root, "aim2_loco", "score")),
                (
                    "aim2_seal_inference",
                    _control_command(root, "aim2_loco", "seal-inference"),
                ),
            ]
        )
    else:
        stages.append(
            (
                "aim2_verify_inference_resume",
                _control_command(root, "aim2_loco", "seal-inference"),
            )
        )
    if not aim2._report_paths(aim2_api_root(root))["receipt"].is_file():
        stages.append(("aim2_report", _control_command(root, "aim2_loco", "report")))
    stages.extend(
        [
            ("aim1_verify", _control_command(root, "aim1_e0", "verify")),
            ("aim3_verify", _control_command(root, "aim3_ladders", "verify")),
            ("aim2_verify", _control_command(root, "aim2_loco", "verify")),
        ]
    )
    with (
        _exclusive_lock(CAMPAIGN_LOCK_PATH, "five-seed campaign postprocessing"),
        _exclusive_lock(GPU_LOCK_PATH, "host-wide GPU 0 inference lease") as gpu_lock_fd,
        _termination_as_exception(),
    ):
        resources = _resource_admission(root)
        print(json.dumps({"postprocess_resource_admission": resources}, indent=2), flush=True)
        for label, command in stages:
            _run_control(root, label, command, gpu_lock_fd=gpu_lock_fd)
        payload = _build_final_receipt(root)
        _write_json_once(_final_receipt_path(root), payload)
        _validate_final_receipt(root)
    print(f"PASS — five-seed analyses and downstream inference sealed: {_final_receipt_path(root)}")


def cmd_verify(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    _load_contract(root)
    _validate_preflight(root)
    _validate_training_receipt(root)
    _run_readonly(_control_command(root, "aim1_e0", "verify"))
    _run_readonly(_control_command(root, "aim3_ladders", "verify"))
    _run_readonly(_control_command(root, "aim2_loco", "verify"))
    _validate_final_receipt(root)
    print("PASS — full study-wide five-seed campaign independently replays and verifies")


def cmd_status(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    contract = _load_contract(root)
    complete = 0
    partial = 0
    absent = 0
    for job in contract["jobs"]:
        receipt = _job_receipt_path(root, job["job_id"])
        if receipt.is_file():
            _validate_top_job_receipt(root, job)
            complete += 1
        elif Path(job["output"]).exists():
            partial += 1
        else:
            absent += 1
    print(
        json.dumps(
            {
                "certified_jobs": complete,
                "partial_or_component_complete_without_campaign_receipt": partial,
                "absent_jobs": absent,
                "training_sealed": _training_receipt_path(root).is_file(),
                "results_sealed": _final_receipt_path(root).is_file(),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, func in (
        ("plan", cmd_plan),
        ("prepare", cmd_prepare),
        ("preflight", cmd_preflight),
        ("status", cmd_status),
        ("verify", cmd_verify),
    ):
        command = commands.add_parser(name)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        command.set_defaults(func=func)
    train = commands.add_parser("train")
    train.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    train.add_argument("--apply", action="store_true")
    train.set_defaults(func=cmd_train)
    post = commands.add_parser("postprocess")
    post.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    post.add_argument("--apply", action="store_true")
    post.set_defaults(func=cmd_postprocess)
    worker = commands.add_parser("_worker")
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--job-id", required=True)
    worker.add_argument("--contract-sha256", required=True)
    worker.set_defaults(func=cmd_worker)
    control_worker = commands.add_parser("_control_worker")
    control_worker.add_argument("--command-json", required=True)
    control_worker.set_defaults(func=cmd_control_worker)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ContractError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
