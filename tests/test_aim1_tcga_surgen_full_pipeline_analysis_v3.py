from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import aim1_tcga_surgen_full_pipeline_analysis as legacy  # noqa: E402
from tools import aim1_tcga_surgen_full_pipeline_analysis_v2 as v2  # noqa: E402
from tools import aim1_tcga_surgen_full_pipeline_analysis_v3 as analysis  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_frozen_v2_source_and_published_bundle_pins() -> None:
    assert analysis._v2_source_identities() == {
        "controller": {
            "path": str(analysis.V2_CONTROLLER.resolve()),
            "sha256": analysis.V2_CONTROLLER_SHA256,
            "size_bytes": analysis.V2_CONTROLLER_SIZE,
        },
        "controller_test": {
            "path": str(analysis.V2_TEST.resolve()),
            "sha256": analysis.V2_TEST_SHA256,
            "size_bytes": analysis.V2_TEST_SIZE,
        },
    }
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    if not (root / "downstream_v2/contract.json").is_file():
        pytest.skip("Published downstream-v2 bundle is not mounted")
    identities = analysis._v2_artifact_identities(root)
    assert legacy._artifact(Path(identities["contract"]["path"])) == identities["contract"]
    assert (
        legacy._artifact(Path(identities["score_job_plan"]["path"])) == identities["score_job_plan"]
    )


def test_live_public_dry_prepare_typed_replays_v2_without_writes() -> None:
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    if not (root / "downstream_v2/contract.json").is_file():
        pytest.skip("Published downstream-v2 bundle is not mounted")
    destination = analysis.continuation_root(root)
    if destination.exists() or destination.is_symlink():
        pytest.skip("Live dry-prepare regression requires fresh continuation_v3")
    before = {
        name: legacy._artifact(Path(identity["path"]))
        for name, identity in analysis._v2_artifact_identities(root).items()
    }
    observed = analysis.prepare(root, apply=False)
    assert observed == {
        "status": "dry_run_ready_to_continue_from_immutable_v2_prepare",
        "campaign_root": str(root),
        "continuation_root": str(destination),
        "predecessor_v2_artifacts": 2,
        "legacy_prepared_artifacts": 7,
        "label_blind_targets": list(analysis.TARGET_ORDER),
        "score_jobs": 50,
        "score_slide_rows": 4_790,
        "outcome_files_opened": False,
        "predecessor_files_modified": False,
    }
    assert not destination.exists()
    assert {
        name: legacy._artifact(Path(identity["path"]))
        for name, identity in analysis._v2_artifact_identities(root).items()
    } == before


def test_predecessor_validation_never_calls_buggy_walkers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    if not (root / "downstream_v2/contract.json").is_file():
        pytest.skip("Published downstream-v2 bundle is not mounted")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("buggy generic walker called")

    monkeypatch.setattr(v2, "_load_contract_v2", forbidden)
    monkeypatch.setattr(v2, "_strict_continuation_json_tree", forbidden)
    monkeypatch.setattr(legacy, "_strict_json_tree", forbidden)
    observed = analysis.validate_predecessor_v2_bundle(root, deep=False)
    assert observed["status"] == "authenticated_immutable_two_file_predecessor"
    assert observed["score_jobs"] == 50


def test_predecessor_exact_byte_and_semantic_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    if not (root / "downstream_v2/contract.json").is_file():
        pytest.skip("Published downstream-v2 bundle is not mounted")
    original_sha = analysis.V2_CONTRACT_SHA256
    monkeypatch.setattr(analysis, "V2_CONTRACT_SHA256", "0" * 64)
    with pytest.raises(analysis.GovernanceError, match="contract bytes drifted"):
        analysis.validate_predecessor_v2_bundle(root, deep=False)
    monkeypatch.setattr(analysis, "V2_CONTRACT_SHA256", original_sha)
    original_payload = v2._contract_payload_v2

    def drifted_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = original_payload(*args, **kwargs)
        value["status"] = "semantic_drift"
        return value

    monkeypatch.setattr(v2, "_contract_payload_v2", drifted_payload)
    with pytest.raises(analysis.GovernanceError, match="typed-replay exactly"):
        analysis.validate_predecessor_v2_bundle(root, deep=False)


def test_score_factorization_and_child_command_are_v3_owned(tmp_path: Path) -> None:
    jobs = analysis._score_jobs_v3(tmp_path)
    assert len(jobs) == 50
    assert sum(job["expected_rows"] for job in jobs) == 4_790
    assert len({job["job_id"] for job in jobs}) == 50
    assert all("/downstream_v2/continuation_v3/scores/" in job["output"] for job in jobs)
    assert all("/downstream/inputs/label_blind/" in job["manifest"] for job in jobs)
    command = analysis._internal_score_command_v3(tmp_path, jobs[0], device="cuda", num_workers=4)
    assert command[0] == sys.executable
    assert Path(command[1]).resolve() == Path(analysis.__file__).resolve()
    assert command[2] == "_score-one"


def test_runtime_bindings_are_nested_v3_and_restore(tmp_path: Path) -> None:
    originals = {name: getattr(legacy, name) for name in analysis._RUNTIME_BINDINGS}
    with pytest.raises(RuntimeError, match="sentinel"), analysis._scoped_legacy_runtime():
        assert legacy.downstream_root(tmp_path) == tmp_path / analysis.NAMESPACE
        assert legacy.analysis_root(tmp_path) == tmp_path / analysis.NAMESPACE / "analysis"
        assert legacy.blind_path(tmp_path, "cptac_primary") == (
            tmp_path / "downstream/inputs/label_blind/cptac_primary.csv"
        )
        raise RuntimeError("sentinel")
    assert {name: getattr(legacy, name) for name in analysis._RUNTIME_BINDINGS} == originals


def test_nested_v2_reconstruction_restores_outer_v3_runtime() -> None:
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    if not (root / "downstream_v2/contract.json").is_file():
        pytest.skip("Published downstream-v2 bundle is not mounted")
    original = legacy.downstream_root
    with analysis._scoped_legacy_runtime():
        assert legacy.downstream_root(root) == analysis.continuation_root(root)
        analysis.validate_predecessor_v2_bundle(root, deep=False)
        assert legacy.downstream_root(root) == analysis.continuation_root(root)
    assert legacy.downstream_root is original


def test_typed_identity_graph_accepts_typed_baseline_generic_walker_rejects(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    _write_json(artifact, {"ok": True})
    identity = legacy._artifact(artifact)
    payload = {
        "prepared_downstream_baseline": {
            "root": str(tmp_path.resolve()),
            "canonical_record_schema": "sorted campaign-relative {path,sha256,size_bytes}",
            "artifact_count": 1,
            "total_size_bytes": identity["size_bytes"],
            "tree_sha256": "0" * 64,
            "artifacts": [identity],
        }
    }
    analysis._typed_identity_graph(payload)
    contract = tmp_path / "contract.json"
    _write_json(contract, payload)
    with pytest.raises(analysis.GovernanceError, match="ambiguous artifact-identity base"):
        legacy._strict_json_tree(contract)


def test_typed_identity_graph_rejects_relative_and_symlinked_paths(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    _write_json(artifact, {"ok": True})
    identity = legacy._artifact(artifact)
    relative = {**identity, "path": "artifact.json"}
    with pytest.raises(analysis.GovernanceError, match="non-absolute"):
        analysis._typed_identity_graph({"identity": relative})
    alias = tmp_path / "alias.json"
    alias.symlink_to(artifact)
    symlinked = {**identity, "path": str(alias)}
    with pytest.raises(analysis.GovernanceError, match="normalized/symlinked"):
        analysis._typed_identity_graph({"identity": symlinked})


def test_outer_namespace_allows_exact2_plus_v3_only(tmp_path: Path) -> None:
    base = tmp_path / "downstream_v2"
    _write_json(base / "contract.json", {})
    _write_json(base / "jobs/score_jobs.json", {})
    analysis._assert_predecessor_namespace(tmp_path)
    _write_json(base / "continuation_v3/contract.json", {})
    analysis._assert_predecessor_namespace(tmp_path)
    (base / "unexpected").mkdir()
    with pytest.raises(analysis.GovernanceError, match="Unexpected immutable"):
        analysis._assert_predecessor_namespace(tmp_path)


def test_outer_namespace_rejects_symlink_and_special_path(tmp_path: Path) -> None:
    base = tmp_path / "downstream_v2"
    _write_json(base / "contract.json", {})
    _write_json(base / "jobs/score_jobs.json", {})
    (base / "continuation_v3").symlink_to(tmp_path)
    with pytest.raises(analysis.GovernanceError, match="symlink"):
        analysis._assert_predecessor_namespace(tmp_path)


def test_outer_namespace_rejects_special_file(tmp_path: Path) -> None:
    base = tmp_path / "downstream_v2"
    _write_json(base / "contract.json", {})
    _write_json(base / "jobs/score_jobs.json", {})
    special = base / "continuation_v3/special.fifo"
    special.parent.mkdir(parents=True)
    os.mkfifo(special)
    with pytest.raises(analysis.GovernanceError, match="special path"):
        analysis._assert_predecessor_namespace(tmp_path)


def _synthetic_predecessor() -> dict[str, Any]:
    return {
        "status": "authenticated_immutable_two_file_predecessor",
        "namespace": "downstream_v2",
        "source_implementation": {},
        "artifacts": {},
        "created_utc": "2026-08-25T00:00:00+00:00",
        "contract_status": "prepared",
        "score_jobs": 50,
        "score_slide_rows": 4_790,
        "outcomes_opened": False,
        "training": {"terminal": {}},
        "legacy_prepared_artifacts": {},
        "legacy_label_blind_snapshots": {},
        "target_records": {},
        "snapshot_frames": {},
        "snapshots": {},
        "feature_stores": {},
    }


def test_prepare_publishes_exact2_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "downstream_v2").mkdir()
    predecessor = _synthetic_predecessor()
    monkeypatch.setattr(
        analysis,
        "validate_predecessor_v2_bundle",
        lambda root, deep=True: predecessor,
    )
    monkeypatch.setattr(analysis, "_score_jobs_v3", lambda root: [{"job_id": "one"}])

    def payload(*args: Any, created_utc: str) -> dict[str, Any]:
        return {"status": "prepared", "created_utc": created_utc, "plan": args[-1]}

    monkeypatch.setattr(analysis, "_contract_payload_v3", payload)

    def load(root: Path, *, deep: bool = True) -> dict[str, Any]:
        namespace = analysis.continuation_root(root)
        assert {
            path.relative_to(namespace).as_posix()
            for path in namespace.rglob("*")
            if path.is_file()
        } == {"contract.json", "jobs/score_jobs.json"}
        return legacy._read_json(namespace / "contract.json")

    monkeypatch.setattr(analysis, "_load_contract_v3", load)
    observed = analysis.prepare(tmp_path, apply=True)
    assert observed["status"] == "prepared"
    assert not list(tmp_path.parent.glob(f".{tmp_path.name}.continuation-v3-stage-*"))


def test_prepare_failure_never_publishes_partial_v3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        analysis,
        "validate_predecessor_v2_bundle",
        lambda root, deep=True: _synthetic_predecessor(),
    )
    monkeypatch.setattr(analysis, "_score_jobs_v3", lambda root: [{"job_id": "one"}])
    monkeypatch.setattr(
        analysis,
        "_contract_payload_v3",
        lambda *args, created_utc: {"created_utc": created_utc},
    )
    original = legacy._write_json_once

    def fail_contract(path: Path, value: Any) -> None:
        if path.name == "contract.json":
            raise analysis.GovernanceError("injected stage failure")
        original(path, value)

    monkeypatch.setattr(legacy, "_write_json_once", fail_contract)
    with pytest.raises(analysis.GovernanceError, match="injected stage failure"):
        analysis.prepare(tmp_path, apply=True)
    assert not analysis.continuation_root(tmp_path).exists()
    assert not list(tmp_path.parent.glob(f".{tmp_path.name}.continuation-v3-stage-*"))


def test_prepare_existing_partial_collision_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = analysis.continuation_root(tmp_path)
    _write_json(destination / "contract.json", {"partial": True})
    before = (destination / "contract.json").read_bytes()
    monkeypatch.setattr(
        analysis,
        "validate_predecessor_v2_bundle",
        lambda root, deep=True: _synthetic_predecessor(),
    )

    def reject(root: Path, *, deep: bool = True) -> dict[str, Any]:
        raise analysis.GovernanceError("partial collision")

    monkeypatch.setattr(analysis, "_load_contract_v3", reject)
    with pytest.raises(analysis.GovernanceError, match="partial collision"):
        analysis.prepare(tmp_path, apply=True)
    assert (destination / "contract.json").read_bytes() == before
    assert not (destination / "jobs/score_jobs.json").exists()


def test_strict_v3_graph_opens_only_owned_job_plan(tmp_path: Path) -> None:
    plan_path = analysis.continuation_root(tmp_path) / "jobs/score_jobs.json"
    _write_json(plan_path, {"jobs": analysis._score_jobs_v3(tmp_path)})
    external = tmp_path / "recovery_v2/typed.json"
    _write_json(external, {"root": str(tmp_path), "artifacts": [], "typed": True})
    contract = {
        "continuation": {"score_job_plan": legacy._artifact(plan_path)},
        "external_typed_leaf": legacy._artifact(external),
    }
    analysis._strict_continuation_graph(tmp_path, contract)
    _write_json(plan_path, {"jobs": []})
    with pytest.raises(analysis.GovernanceError, match="score-job plan|Artifact size drifted"):
        analysis._strict_continuation_graph(tmp_path, contract)


def test_analyze_deep_gates_before_delegated_outcome_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def load(root: Path, *, deep: bool = True) -> dict[str, Any]:
        assert deep is True
        events.append("deep_gate")
        return {}

    def delegated(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("analysis")
        return {"status": "dry"}

    monkeypatch.setattr(analysis, "_load_contract_v3", load)
    monkeypatch.setattr(legacy, "analyze", delegated)
    assert analysis.analyze(tmp_path, apply=False)["status"] == "dry"
    assert events == ["deep_gate", "analysis"]


def test_final_inventory_and_source_graph_rosters_are_exact(tmp_path: Path) -> None:
    expected = analysis._expected_final_inventory(tmp_path)
    assert len(expected) == 111
    assert all(analysis.continuation_root(tmp_path) in path.parents for path in expected)
    assert len(analysis.FINAL_SOURCE_GRAPH_KEYS) == 13
    assert len(set(analysis.FINAL_SOURCE_GRAPH_KEYS)) == 13
    assert (
        sum(
            key.startswith("continuation_v3_analysis_")
            or key
            in {
                "continuation_v3_patient_native_logits",
                "continuation_v3_bootstrap_distributions",
                "continuation_v3_results",
            }
            for key in analysis.FINAL_SOURCE_GRAPH_KEYS
        )
        == 5
    )


def test_delegation_contract_receipts_all_runtime_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis, "ANALYSIS_V3_TEST", Path(__file__).resolve())
    delegation = analysis._delegation_contract(Path("/tmp"))
    assert delegation["runtime_binding_symbol_roster"] == sorted(analysis._RUNTIME_BINDINGS)
    assert set(delegation["runtime_bindings"]) == {
        f"legacy.{name}" for name in analysis._RUNTIME_BINDINGS
    }
    assert delegation["predecessor_strict_walker_called"] is False
    assert delegation["root_analysis_allowed"] is False


def test_root_analysis_remains_forbidden(tmp_path: Path) -> None:
    (tmp_path / "analysis").mkdir()
    with pytest.raises(analysis.GovernanceError, match="root/analysis is forbidden"):
        analysis._assert_predecessor_namespace(tmp_path)
