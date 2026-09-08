from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim1_all_primary_orion_parallel_v2 as parallel


def test_parallel_graph_is_exactly_fifteen_distinct_folds_then_three_refits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "aim1_e0_all_primary_orion_exp_parallel_v2_test"
    jobs = parallel._job_graph(root)  # noqa: SLF001

    assert len(jobs) == 18
    assert len({job["key"] for job in jobs}) == 18
    folds = [job for job in jobs if job["kind"] == "fold"]
    finalizers = [job for job in jobs if job["kind"] == "finalize"]
    assert len(folds) == 15
    assert len(finalizers) == 3
    assert [(job["seed"], job["fold"]) for job in folds] == [
        (seed, fold)
        for seed in parallel.base.SEEDS
        for fold in range(parallel.base.N_FOLDS)
    ]
    for job in finalizers:
        assert job["depends_on"] == [
            parallel._fold_key(job["seed"], fold)  # noqa: SLF001
            for fold in range(parallel.base.N_FOLDS)
        ]


def test_initial_six_dispatches_are_unique_fold_jobs_and_refit_is_dependency_gated(
    tmp_path: Path,
) -> None:
    jobs = parallel._job_graph(tmp_path / "campaign")  # noqa: SLF001
    initial = parallel._ready_jobs(jobs, set())[:6]  # noqa: SLF001
    assert [job["key"] for job in initial] == [
        *(parallel._fold_key(42, fold) for fold in range(5)),  # noqa: SLF001
        parallel._fold_key(43, 0),  # noqa: SLF001
    ]
    assert all(job["kind"] == "fold" for job in initial)

    completed = {
        parallel._fold_key(42, fold)  # noqa: SLF001
        for fold in range(parallel.base.N_FOLDS)
    }
    pending = [job for job in jobs if job["key"] not in completed]
    ready = parallel._ready_jobs(pending, completed)  # noqa: SLF001
    assert ready[0]["key"] == parallel._finalize_key(42)  # noqa: SLF001
    assert parallel._finalize_key(43) not in {job["key"] for job in ready}  # noqa: SLF001


def test_parallel_contract_binds_six_trainers_worker_accounting_and_v1_supersession(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "aim1_e0_all_primary_orion_exp_parallel_v2_test"
    serial = {
        "device": "cuda:0",
        "host_wide_lock": str(parallel.base.GPU_LOCK_PATH.resolve()),
        "concurrent_training_processes": 1,
        "cuda_idle_precheck": True,
    }
    monkeypatch.setattr(
        parallel.base,
        "_read_json",
        lambda _path: {"recipe": {"gpu_execution": serial}},
    )
    monkeypatch.setattr(
        parallel.base,
        "_artifact",
        lambda path: {
            "path": str(Path(path).resolve()),
            "sha256": hashlib.sha256(str(path).encode()).hexdigest(),
            "size_bytes": 1,
        },
    )
    monkeypatch.setattr(
        parallel,
        "_component_implementation",
        lambda: [{"path": "component", "sha256": "abc", "size_bytes": 1}],
    )

    contract = parallel._build_parallel_contract(  # noqa: SLF001
        root, created_utc="2026-08-23T00:00:00Z"
    )
    policy = contract["parallel_policy"]
    assert policy["max_concurrent_gpu_trainer_processes"] == 6
    assert policy["training_num_workers_per_process"] == 6
    assert policy["maximum_configured_train_loader_workers"] == 36
    assert policy["maximum_resident_train_plus_val_loader_processes"] == 72
    assert contract["operational_supersession"]["base_value"] == serial
    assert contract["operational_supersession"]["model_recipe_changed"] is False
    assert contract["v1_partial_artifacts_reused"] is False


def test_train_dry_run_prints_bound_fresh_interpreter_commands_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "aim1_e0_all_primary_orion_exp_parallel_v2_test"
    jobs = parallel._job_graph(root)  # noqa: SLF001
    monkeypatch.setattr(
        parallel,
        "_load_parallel_contract",
        lambda _root: {"schedule": {"jobs": jobs}},
    )
    monkeypatch.setattr(
        parallel.base, "_validate_generated_splits", lambda _root: None
    )
    monkeypatch.setattr(
        parallel.base,
        "_artifact",
        lambda _path: {"sha256": "a" * 64},
    )
    monkeypatch.setattr(
        parallel,
        "_publish_base_seed_requests",
        lambda _root: pytest.fail("dry-run must not publish requests"),
    )

    parallel.cmd_train(argparse.Namespace(output_root=root, dry_run=True))
    lines = capsys.readouterr().out.strip().splitlines()
    commands = lines[:-1]
    assert len(commands) == 18
    assert len(set(commands)) == 18
    assert all(str(Path(parallel.__file__).resolve()) in command for command in commands)
    assert sum(" _fold " in command for command in commands) == 15
    assert sum(" _finalize " in command for command in commands) == 3
    assert lines[-1] == "DRY RUN: 18 dependency-governed jobs; maximum concurrency=6"


def test_internal_worker_requires_orchestrator_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(parallel.WORKER_AUTH_ENV, raising=False)
    monkeypatch.delenv(f"{parallel.WORKER_AUTH_ENV}_SHA256", raising=False)
    monkeypatch.delenv(f"{parallel.WORKER_AUTH_ENV}_CONTRACT", raising=False)
    with pytest.raises(parallel.base.ContractError, match="authorized orchestrator"):
        parallel._require_worker_authorization("contract")  # noqa: SLF001

    token = "test-session"
    monkeypatch.setenv(parallel.WORKER_AUTH_ENV, token)
    monkeypatch.setenv(
        f"{parallel.WORKER_AUTH_ENV}_SHA256", hashlib.sha256(token.encode()).hexdigest()
    )
    monkeypatch.setenv(f"{parallel.WORKER_AUTH_ENV}_CONTRACT", "contract")
    parallel._require_worker_authorization("contract")  # noqa: SLF001
    with pytest.raises(parallel.base.ContractError, match="different contract"):
        parallel._require_worker_authorization("other")  # noqa: SLF001


def test_retry_quarantines_incomplete_fold_instead_of_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "aim1_e0_all_primary_orion_exp_parallel_v2_test"
    job = parallel._job_graph(root)[0]  # noqa: SLF001
    fold_dir = parallel.base._run_dir(root, 42) / "fold_0"  # noqa: SLF001
    fold_dir.mkdir(parents=True)
    (fold_dir / "partial.ckpt").write_bytes(b"partial")
    monkeypatch.setattr(parallel, "_fold_complete", lambda *_args: False)

    parallel._quarantine_incomplete_fold(root, job, attempt=2)  # noqa: SLF001

    destination = (
        parallel.base._run_dir(root, 42)  # noqa: SLF001
        / "failed_fold_attempts"
        / "fold_0"
        / "attempt-001"
    )
    assert not fold_dir.exists()
    assert (destination / "partial.ckpt").read_bytes() == b"partial"
    assert (destination / "quarantine_receipt.json").is_file()


def test_resource_admission_fails_closed_below_any_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 35)
    monkeypatch.setattr(parallel, "_available_ram_gib", lambda: 200.0)
    monkeypatch.setattr(parallel, "_free_gpu_mib", lambda: 24_000)
    with pytest.raises(parallel.base.ContractError, match="resource admission failed"):
        parallel._resource_admission()  # noqa: SLF001


def test_production_guard_rejects_v1_and_allows_only_exact_v2_or_tmp(
    tmp_path: Path,
) -> None:
    with pytest.raises(parallel.base.ContractError, match="exactly"):
        parallel._guard_v2_output_root(  # noqa: SLF001
            parallel.base.DEFAULT_OUTPUT_ROOT
        )
    assert (
        parallel._guard_v2_output_root(parallel.DEFAULT_OUTPUT_ROOT)  # noqa: SLF001
        == parallel.DEFAULT_OUTPUT_ROOT.resolve()
    )
    assert parallel._guard_v2_output_root(tmp_path / "fixture") == (  # noqa: SLF001
        tmp_path / "fixture"
    ).resolve()


def test_parallel_supplement_refuses_a_pretrained_or_reused_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    (root / "inputs").mkdir(parents=True)
    (root / "inputs" / "base.json").write_text("{}", encoding="utf-8")
    parallel._require_pretraining_v2_root(root)  # noqa: SLF001
    train_artifact = root / "train" / "seed42" / "fold_0" / "completion.json"
    train_artifact.parent.mkdir(parents=True)
    train_artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(parallel.base.ContractError, match="refusing reuse"):
        parallel._require_pretraining_v2_root(root)  # noqa: SLF001


def test_native_complete_without_receipt_remains_pending_for_certification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    monkeypatch.setattr(parallel, "_fold_complete", lambda *_args: True)
    monkeypatch.setattr(parallel, "_finalize_complete", lambda *_args: True)
    assert parallel._completed_keys(root) == set()  # noqa: SLF001


def test_job_receipt_requires_matching_returncode_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    job = parallel._job_graph(root)[0]  # noqa: SLF001
    log_dir = parallel._job_log_dir(root, job)  # noqa: SLF001
    log_dir.mkdir(parents=True)
    log_path = log_dir / "attempt-001.log"
    log_path.write_text("completed native fold\n", encoding="utf-8")
    exit_path = log_path.with_suffix(".exit.json")
    exit_record = {
        "schema_version": 1,
        "job": job["key"],
        "attempt": 1,
        "returncode": 1,
        "finished_utc": "2026-08-23T00:00:00Z",
        "log": parallel.base._artifact(log_path),  # noqa: SLF001
    }
    exit_path.write_text(json.dumps(exit_record), encoding="utf-8")
    monkeypatch.setattr(parallel, "_fold_complete", lambda *_args: True)

    with pytest.raises(parallel.base.ContractError, match="returncode-0"):
        parallel._ensure_job_receipt(root, job, log_path)  # noqa: SLF001


def test_recoverable_refit_failure_is_quarantined_without_touching_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    job = next(
        item
        for item in parallel._job_graph(root)  # noqa: SLF001
        if item["key"] == parallel._finalize_key(42)  # noqa: SLF001
    )
    directory = parallel.base._run_dir(root, 42)  # noqa: SLF001
    for fold in range(parallel.base.N_FOLDS):
        fold_dir = directory / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        (fold_dir / "marker").write_text(str(fold), encoding="utf-8")
    (directory / "training_completion.json").write_text("{}", encoding="utf-8")
    (directory / "oof_predictions.parquet").write_bytes(b"oof")
    final = directory / "final"
    final.mkdir()
    (final / "finalize_summary.json").write_text(
        json.dumps({"refit": {"strategy": "refit", "error": "oom"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(parallel, "_fold_complete", lambda *_args: True)
    monkeypatch.setattr(
        parallel.base,
        "_training_record",
        lambda *_args: (_ for _ in ()).throw(parallel.base.ContractError("bad refit")),
    )

    assert parallel._finalize_complete(root, 42) is False  # noqa: SLF001
    parallel._quarantine_incomplete_finalization(  # noqa: SLF001
        root, job, attempt=2
    )
    destination = directory / "failed_finalization_attempts" / "attempt-001"
    assert (destination / "training_completion.json").is_file()
    assert (destination / "oof_predictions.parquet").is_file()
    assert (destination / "final" / "finalize_summary.json").is_file()
    assert all(
        (directory / f"fold_{fold}" / "marker").read_text(encoding="utf-8")
        == str(fold)
        for fold in range(parallel.base.N_FOLDS)
    )


def test_scheduler_exception_requests_cleanup_for_every_running_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    jobs = parallel._job_graph(root)[:2]  # noqa: SLF001
    launched: dict[str, Any] = {}

    class Process:
        pid = 123

        def poll(self) -> int | None:
            raise RuntimeError("poll failed")

    def launch(
        _root: Path,
        job: dict[str, Any],
        _digest: str,
        _session: str,
        _lease: int,
    ) -> Any:
        value = SimpleNamespace(job=job, process=Process())
        launched[job["key"]] = value
        return value

    cleaned: list[set[str]] = []
    monkeypatch.setattr(parallel, "_completed_keys", lambda _root: set())
    monkeypatch.setattr(parallel, "_launch_job", launch)
    monkeypatch.setattr(
        parallel,
        "_terminate_and_reap",
        lambda _root, running: cleaned.append(set(running)),
    )
    monkeypatch.setattr(
        parallel.base, "_artifact", lambda _path: {"sha256": "b" * 64}
    )
    with pytest.raises(RuntimeError, match="poll failed"):
        parallel._run_scheduler(  # noqa: SLF001
            root, {"schedule": {"jobs": jobs}}, gpu_lease_fd=9
        )
    assert set(launched) == {job["key"] for job in jobs}
    assert cleaned == [set(launched)]


def test_launch_captures_parent_pid_before_preexec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    job = parallel._job_graph(root)[0]  # noqa: SLF001
    captured: list[int] = []

    class Process:
        pid = 456

    def popen(_command: list[str], **kwargs: Any) -> Process:
        kwargs["preexec_fn"]()
        return Process()

    monkeypatch.setattr(parallel, "_quarantine_incomplete_fold", lambda *_args: None)
    monkeypatch.setattr(
        parallel, "_quarantine_incomplete_finalization", lambda *_args: None
    )
    monkeypatch.setattr(parallel.subprocess, "Popen", popen)
    monkeypatch.setattr(
        parallel,
        "_configure_child_parent_death",
        lambda parent_pid: captured.append(parent_pid),
    )
    parent_pid = parallel.os.getpid()
    running = parallel._launch_job(  # noqa: SLF001
        root, job, "c" * 64, "session", gpu_lease_fd=11
    )
    running.log_stream.close()
    assert captured == [parent_pid]


def test_fold_worker_uses_native_logging_matmul_and_fold_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oceanpath import runtime
    from oceanpath.workflows import training

    root = tmp_path / "campaign"
    job = parallel._job_graph(root)[0]  # noqa: SLF001
    calls: list[str] = []

    @contextmanager
    def context(name: str):
        calls.append(f"enter:{name}")
        yield
        calls.append(f"exit:{name}")

    monkeypatch.setattr(parallel, "_require_worker_authorization", lambda *_args: None)
    monkeypatch.setattr(
        parallel,
        "_load_parallel_contract",
        lambda _root: {"schedule": {"jobs": [job]}},
    )
    monkeypatch.setattr(
        parallel.base, "_artifact", lambda _path: {"sha256": "d" * 64}
    )
    monkeypatch.setattr(parallel, "_load_cfg", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(parallel, "_preinitialize_seed", lambda *_args: "fp")
    monkeypatch.setattr(
        parallel,
        "_nonblocking_job_lock",
        lambda *_args: context("lock"),
    )
    outcomes = iter((False, True))
    monkeypatch.setattr(training, "_fold_complete", lambda *_args, **_kwargs: next(outcomes))
    monkeypatch.setattr(training, "training_run_fingerprint", lambda _cfg: "fp")
    monkeypatch.setattr(training, "_setup_logging", lambda _cfg: calls.append("logging"))
    monkeypatch.setattr(training, "fold_context", lambda fold: context(f"fold:{fold}"))
    monkeypatch.setattr(
        training,
        "run_fold",
        lambda **_kwargs: calls.append("run_fold"),
    )
    monkeypatch.setattr(
        runtime,
        "run_context",
        lambda *_args, **_kwargs: context("run_context"),
    )
    monkeypatch.setattr(
        parallel.torch,
        "set_float32_matmul_precision",
        lambda value: calls.append(f"matmul:{value}"),
    )

    parallel._run_fold_worker(root, 42, 0, "d" * 64)  # noqa: SLF001
    assert "logging" in calls
    assert "matmul:high" in calls
    assert calls.index("enter:fold:0") < calls.index("run_fold") < calls.index("exit:fold:0")
