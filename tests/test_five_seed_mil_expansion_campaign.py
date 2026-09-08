from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim1_mil_five_seed_expansion as campaign


def test_exact_cross_aim_inventory(tmp_path: Path) -> None:
    jobs = campaign.build_job_inventory(tmp_path / "campaign")
    assert len(jobs) == 102
    assert sum(job["fit_count"] for job in jobs) == 490
    assert {job["seed"] for job in jobs} == {45, 46}


def test_component_fit_accounting(tmp_path: Path) -> None:
    jobs = campaign.build_job_inventory(tmp_path / "campaign")
    observed = {
        component: (
            sum(job["component"] == component for job in jobs),
            sum(job["fit_count"] for job in jobs if job["component"] == component),
        )
        for component in {job["component"] for job in jobs}
    }
    assert observed == {
        "aim1_e0": (2, 12),
        "aim2_loco": (36, 108),
        "aim3_ladders": (64, 370),
    }


def test_job_ids_and_outputs_are_disjoint(tmp_path: Path) -> None:
    jobs = campaign.build_job_inventory(tmp_path / "campaign")
    assert len({job["job_id"] for job in jobs}) == len(jobs)
    assert len({job["output"] for job in jobs}) == len(jobs)
    assert all("/aim2_loco/aim2_loco/" not in job["output"] for job in jobs)


def test_every_command_reenters_guarded_component_worker(tmp_path: Path) -> None:
    jobs = campaign.build_job_inventory(tmp_path / "campaign")
    for job in jobs:
        command = job["command"]
        assert "train-one" in command
        assert "--external-scheduler" in command
        if job["component"] == "aim2_loco":
            assert "--apply" in command
            assert "--recover-orphan" in command


def test_inventory_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    assert campaign.build_job_inventory(root) == campaign.build_job_inventory(root)


def test_production_root_is_exact() -> None:
    assert campaign.validate_output_root(campaign.DEFAULT_OUTPUT_ROOT) == (
        campaign.DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    )
    with pytest.raises(campaign.ContractError):
        campaign.validate_output_root(campaign.DEFAULT_OUTPUT_ROOT.parent / "wrong")


def test_tmp_roots_are_allowed(tmp_path: Path) -> None:
    assert campaign.validate_output_root(tmp_path / "nested") == (tmp_path / "nested").resolve()


def test_json_publication_is_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    value = {"schema_version": 1, "status": "PASS"}
    campaign._write_json_once(path, value)
    first = path.read_bytes()
    campaign._write_json_once(path, value)
    assert path.read_bytes() == first
    with pytest.raises(campaign.ContractError):
        campaign._write_json_once(path, {"schema_version": 1, "status": "CHANGED"})


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"status":"PASS","status":"FAIL"}\n', encoding="utf-8")
    with pytest.raises(campaign.ContractError, match="Duplicate JSON key"):
        campaign._read_json(path)


def test_plan_parser_defaults_to_governed_root() -> None:
    args = campaign.build_parser().parse_args(["plan"])
    assert args.output_root == campaign.DEFAULT_OUTPUT_ROOT
    assert args.func is campaign.cmd_plan


def test_aim3_verify_is_bound_to_the_campaign_e0_root(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    command = campaign._control_command(root, "aim3_ladders", "verify")
    assert "--e0-root" in command
    assert str(campaign.aim1_root(root)) in command


def test_safe_job_name_is_path_free() -> None:
    assert campaign._safe_job_name("aim2/source:seed45") == "aim2__source_seed45"


def test_campaign_constants() -> None:
    assert campaign.MAX_CONCURRENT_GPU_TRAINERS == 6
    assert campaign.EXPECTED_JOB_COUNT == 102
    assert campaign.EXPECTED_NEW_FITS == 490


def test_contract_loader_replays_every_sealed_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    path = campaign._contract_path(root)  # noqa: SLF001
    path.parent.mkdir(parents=True)
    expected = {
        "schema_version": 1,
        "created_utc": "fresh-build-value",
        "status": "sealed_before_new_fit",
        "execution": {"max_concurrent_gpu_trainers": 6},
    }
    stored = {**expected, "created_utc": "sealed-time"}
    path.write_text(json.dumps(stored), encoding="utf-8")
    monkeypatch.setattr(campaign, "_build_contract", lambda _root: expected.copy())
    assert campaign._load_contract(root) == stored  # noqa: SLF001

    stored["execution"] = {"max_concurrent_gpu_trainers": 7}
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(campaign.ContractError, match="no longer replays exactly"):
        campaign._load_contract(root)  # noqa: SLF001


def test_authorized_worker_supervises_component_without_leaking_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    job = {"job_id": "job", "command": ["/bin/echo", "go"]}
    monkeypatch.setattr(campaign, "_load_contract", lambda _root: {"jobs": [job]})
    monkeypatch.setattr(campaign, "_artifact", lambda _path: {"sha256": "contract"})
    monkeypatch.setattr(campaign, "_require_worker_authorization", lambda _sha: None)
    monkeypatch.setenv(campaign.WORKER_AUTH_ENV, "secret")
    monkeypatch.setenv(f"{campaign.WORKER_AUTH_ENV}_SHA256", "secret-hash")
    captured: dict[str, object] = {}

    def fake_supervise(
        command: list[str], *, environment: dict[str, str], gpu_lease_fd: int
    ) -> None:
        captured.update(command=command, environment=environment, gpu_lease_fd=gpu_lease_fd)

    monkeypatch.setattr(campaign, "_supervise_component", fake_supervise)
    monkeypatch.setattr(campaign, "_lease_fd_from_environment", lambda required: 123)
    campaign.cmd_worker(
        argparse.Namespace(
            output_root=root,
            contract_sha256="contract",
            job_id="job",
        )
    )
    assert captured["command"] == job["command"]
    assert captured["gpu_lease_fd"] == 123
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert not any(str(key).startswith(campaign.WORKER_AUTH_ENV) for key in environment)


def _minimal_launch_job(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "campaign"
    contract = campaign._contract_path(root)  # noqa: SLF001
    contract.parent.mkdir(parents=True)
    contract.write_text('{"status":"sealed"}\n', encoding="utf-8")
    job: dict[str, object] = {
        "job_id": "aim1.test",
        "component": "aim1_e0",
        "component_job_key": "test",
        "command": ["/bin/true"],
        "output": str(root / "component-output"),
    }
    return root, job


def test_lone_launch_log_is_reconciled_before_resume(tmp_path: Path) -> None:
    root, job = _minimal_launch_job(tmp_path)
    directory = campaign._job_log_dir(root, str(job["job_id"]))  # noqa: SLF001
    directory.mkdir(parents=True)
    (directory / "attempt-001.log").write_text("pre-request crash\n", encoding="utf-8")

    campaign._reconcile_launch_publication(root, job)  # noqa: SLF001
    ledger = campaign._job_attempt_ledger(root, job)  # noqa: SLF001
    assert len(ledger) == 1
    request = campaign._read_json(directory / "attempt-001.request.json")  # noqa: SLF001
    assert request["recovered_after_log_before_request_crash"] is True


def test_component_output_without_global_launch_request_is_rejected(tmp_path: Path) -> None:
    root, job = _minimal_launch_job(tmp_path)
    Path(str(job["output"])).mkdir(parents=True)
    with pytest.raises(campaign.ContractError, match="without a prior global six-slot"):
        campaign._launch_job(  # noqa: SLF001
            root,
            job,
            contract_sha256="unused",
            session_token="unused",
            gpu_lock_fd=-1,
        )
