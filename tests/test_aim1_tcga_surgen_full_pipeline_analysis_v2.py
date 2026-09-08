from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import aim1_tcga_surgen_full_pipeline_analysis as legacy  # noqa: E402
from tools import aim1_tcga_surgen_full_pipeline_analysis_v2 as analysis  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _identity(path: Path) -> dict[str, Any]:
    return legacy._artifact(path)


def _pin_recovery_sources(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    controller = analysis.RECOVERY_V2_CONTROLLER
    controller_test = analysis.RECOVERY_V2_TEST
    monkeypatch.setattr(analysis, "RECOVERY_V2_CONTROLLER_SHA256", legacy._sha256(controller))
    monkeypatch.setattr(analysis, "RECOVERY_V2_CONTROLLER_SIZE", controller.stat().st_size)
    monkeypatch.setattr(analysis, "RECOVERY_V2_TEST_SHA256", legacy._sha256(controller_test))
    monkeypatch.setattr(analysis, "RECOVERY_V2_TEST_SIZE", controller_test.stat().st_size)
    return analysis._recovery_v2_source_identities()


def _fake_recovery_terminal(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], SimpleNamespace]:
    sources = _pin_recovery_sources(monkeypatch)
    scope_contract = root / "recovery_v2/contract_scope_erratum.json"
    scope_adjudication = root / "recovery_v2/receipts/scope_adjudication.json"
    _write_json(scope_contract, {"status": "scope"})
    _write_json(scope_adjudication, {"status": "adjudicated"})
    v1_adjudication = root / "recovery_v1/receipts/validator_adjudication.json"
    _write_json(v1_adjudication, {"status": "v1-adjudication"})
    source_manifest = root / "inputs/source.csv"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text("slide_id,target_label\nS1,0\n", encoding="utf-8")
    _write_json(
        root / "downstream/contract.json",
        {"training": {"source_manifest_path": str(source_manifest.resolve())}},
    )
    baseline_records = [
        {
            "path": str((root / relative).resolve()),
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
        for relative, sha256, size_bytes in analysis.LEGACY_PREPARED_FILES
    ]
    terminal: dict[str, Any] = {
        "schema_version": 1,
        "recovery": "scope-recovery",
        "status": analysis.RECOVERY_V2_STATUS,
        "created_utc": "2026-08-25T00:00:00+00:00",
        "base_campaign": "base-campaign",
        "scope_contract": _identity(scope_contract),
        "scope_adjudication": _identity(scope_adjudication),
        "recovery_implementation": sources,
        "predecessor_v1_terminal": {
            "path": str((root / analysis.LEGACY_RECOVERY_TERMINAL).resolve()),
            "sha256": analysis.LEGACY_RECOVERY_TERMINAL_IDENTITY["sha256"],
            "size_bytes": analysis.LEGACY_RECOVERY_TERMINAL_IDENTITY["size_bytes"],
        },
        "training_scoped_census": {
            "root": str(root.resolve()),
            "artifact_count": 441,
            "total_size_bytes": 989_858_771,
            "tree_sha256": "a2ffd5b61eaf65dc8d0c5df6a1b29867ffabc57f0af869120603ea37591ac261",
            "original_census_source": _identity(v1_adjudication),
            "original_census_json_pointer": "/raw_artifact_hash_census/before",
            "all_original_records_rehashed_at_certification": True,
            "closed_roster_outside_exclusions": True,
        },
        "prepared_downstream_baseline": {
            "root": str(root.resolve()),
            "canonical_record_schema": "sorted campaign-relative {path,sha256,size_bytes}",
            "artifact_count": 7,
            "total_size_bytes": 95_802,
            "tree_sha256": analysis.LEGACY_PREPARED_TREE_SHA256,
            "artifacts": baseline_records,
        },
        "namespace_policy": {
            "immutable_training_scope": "exact original 441-file campaign census",
            "excluded_root_prefixes": [
                "recovery_v1/",
                "recovery_v2/",
                "downstream/",
                "downstream_v2/",
            ],
            "pinned_predecessor_namespace": "recovery_v1/ (exact nine-file graph)",
            "current_certificate_namespace": "recovery_v2/ (exact three-file graph)",
            "prepared_baseline_namespace": ("downstream/ (seven named bytes remain immutable)"),
            "delegated_growth_namespaces": ["downstream_v2/"],
            "delegated_growth_policy": (
                "regular non-symlink contents wholly delegated to the downstream_v2 "
                "controller; not training-authenticated"
            ),
            "root_analysis_namespace_authorized": False,
        },
        "fit_accounting": legacy.EXPECTED_FIT_ACCOUNTING,
        "execution_accounting": legacy.EXPECTED_EXECUTION_ACCOUNTING,
        "concurrency": {
            "maximum": 6,
            "observed_peak": 6,
            "witness_utc": "2026-08-25T01:10:34.730410+00:00",
        },
        "certification_boundary": (
            "training-only immutability over the exact original 441 records; recovery_v1 "
            "and recovery_v2 are pinned receipt namespaces; seven prepared downstream "
            "records and their namespace roster remain immutable; future downstream_v2 "
            "contents are delegated and are not authenticated as training evidence"
        ),
    }
    terminal_path = root / analysis.RECOVERY_V2_TERMINAL
    _write_json(terminal_path, terminal)
    monkeypatch.setattr(analysis, "RECOVERY_V2_TERMINAL_SHA256", legacy._sha256(terminal_path))
    monkeypatch.setattr(analysis, "RECOVERY_V2_TERMINAL_SIZE", terminal_path.stat().st_size)
    module = SimpleNamespace(
        __file__=str(analysis.RECOVERY_V2_CONTROLLER),
        SCHEMA_VERSION=1,
        RECOVERY="scope-recovery",
        RECOVERY_STATUS=analysis.RECOVERY_V2_STATUS,
        RECOVERY_V2_TERMINAL=analysis.RECOVERY_V2_TERMINAL,
        TERMINAL_FIELDS=analysis.RECOVERY_V2_TERMINAL_FIELDS,
        ContractError=RuntimeError,
        campaign=SimpleNamespace(CAMPAIGN="base-campaign"),
        validate_scoped_terminal=lambda candidate, deep_scope=True: terminal,
    )
    monkeypatch.setattr(analysis, "_recovery_v2_module", lambda: module)
    return terminal, module


def test_frozen_legacy_sources_are_unchanged() -> None:
    assert legacy._artifact(Path(legacy.__file__)) == {
        "path": str(Path(legacy.__file__).resolve()),
        "sha256": analysis.LEGACY_ANALYSIS_CONTROLLER_SHA256,
        "size_bytes": analysis.LEGACY_ANALYSIS_CONTROLLER_SIZE,
    }
    assert legacy._artifact(legacy.ANALYSIS_TEST) == {
        "path": str(legacy.ANALYSIS_TEST.resolve()),
        "sha256": analysis.LEGACY_ANALYSIS_TEST_SHA256,
        "size_bytes": analysis.LEGACY_ANALYSIS_TEST_SIZE,
    }


def test_frozen_recovery_v2_sources_match_compiled_pins() -> None:
    assert analysis._recovery_v2_source_identities() == {
        "controller": {
            "path": str(analysis.RECOVERY_V2_CONTROLLER.resolve()),
            "sha256": analysis.RECOVERY_V2_CONTROLLER_SHA256,
            "size_bytes": analysis.RECOVERY_V2_CONTROLLER_SIZE,
        },
        "controller_test": {
            "path": str(analysis.RECOVERY_V2_TEST.resolve()),
            "sha256": analysis.RECOVERY_V2_TEST_SHA256,
            "size_bytes": analysis.RECOVERY_V2_TEST_SIZE,
        },
    }


def test_published_recovery_v2_terminal_matches_compiled_pin_when_available() -> None:
    terminal = analysis.DEFAULT_CAMPAIGN_ROOT / analysis.RECOVERY_V2_TERMINAL
    if not terminal.is_file():
        pytest.skip("Published recovery-v2 terminal is not mounted")
    assert legacy._artifact(terminal) == {
        "path": str(terminal.resolve()),
        "sha256": analysis.RECOVERY_V2_TERMINAL_SHA256,
        "size_bytes": analysis.RECOVERY_V2_TERMINAL_SIZE,
    }


def test_prepared_bundle_canonical_roster_and_tree() -> None:
    records = analysis._expected_legacy_records(analysis.DEFAULT_CAMPAIGN_ROOT)
    assert len(records) == 7
    assert sum(record["size_bytes"] for record in records) == 95_802
    assert analysis._canonical_record_tree(records) == analysis.LEGACY_PREPARED_TREE_SHA256
    assert [record["path"] for record in records] == sorted(record["path"] for record in records)


def test_live_prepared_seven_file_bundle_when_available() -> None:
    if not (analysis.DEFAULT_CAMPAIGN_ROOT / "downstream/contract.json").is_file():
        pytest.skip("Production predecessor bundle is not mounted")
    observed = analysis.validate_legacy_prepared_bundle(analysis.DEFAULT_CAMPAIGN_ROOT)
    assert observed["artifact_count"] == 7
    assert observed["tree_sha256"] == analysis.LEGACY_PREPARED_TREE_SHA256
    assert set(observed["label_blind_snapshots"]) == set(analysis.TARGET_ORDER)


def test_recovery_v2_gate_accepts_exact_terminal_and_deep_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal, module = _fake_recovery_terminal(tmp_path, monkeypatch)
    calls: list[tuple[Path, bool]] = []

    def validate(candidate: Path, *, deep_scope: bool = True) -> dict[str, Any]:
        calls.append((candidate, deep_scope))
        return terminal

    module.validate_scoped_terminal = validate
    observed = analysis._validate_training_bundle_v2(tmp_path, deep=True)
    assert calls == [(tmp_path.resolve(), True)]
    assert observed["terminal"] == _identity(tmp_path / analysis.RECOVERY_V2_TERMINAL)
    assert observed["prepared_downstream_baseline"]["artifacts"][0]["path"].startswith(
        str(tmp_path.resolve())
    )


def test_v1_predecessor_path_is_immutable_under_scoped_runtime_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _terminal, _module = _fake_recovery_terminal(tmp_path, monkeypatch)
    monkeypatch.setattr(legacy, "TRAINING_RECOVERY_TERMINAL", analysis.RECOVERY_V2_TERMINAL)
    observed = analysis._validate_training_bundle_v2(tmp_path, deep=True)
    assert observed["predecessor_v1_terminal"]["path"] == str(
        (tmp_path / analysis.LEGACY_RECOVERY_TERMINAL).resolve()
    )


def test_live_public_dry_prepare_uses_v1_predecessor_inside_scoped_runtime() -> None:
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    if not (root / analysis.RECOVERY_V2_TERMINAL).is_file():
        pytest.skip("Published recovery-v2 terminal is not mounted")
    destination = root / "downstream_v2"
    if destination.exists() or destination.is_symlink():
        pytest.skip("Live public dry-prepare regression requires fresh downstream_v2")
    old_records = {
        relative: legacy._artifact(root / relative)
        for relative, _sha256, _size_bytes in analysis.LEGACY_PREPARED_FILES
    }
    observed = analysis.prepare(root, apply=False)
    assert observed["status"] == "dry_run_ready_to_adopt_legacy_label_blind_bundle"
    assert observed["outcome_files_opened"] is False
    assert not destination.exists()
    assert {
        relative: legacy._artifact(root / relative)
        for relative, _sha256, _size_bytes in analysis.LEGACY_PREPARED_FILES
    } == old_records


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda terminal: terminal.pop("concurrency"), "field roster"),
        (
            lambda terminal: terminal.__setitem__("fit_accounting", {"hidden_fits": 1}),
            "accounting/certification",
        ),
        (
            lambda terminal: terminal["prepared_downstream_baseline"].__setitem__(
                "artifacts", analysis._expected_legacy_records(Path("/tmp"))
            ),
            "baseline",
        ),
    ],
)
def test_recovery_v2_gate_rejects_schema_accounting_and_relative_baseline_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    terminal, _module = _fake_recovery_terminal(tmp_path, monkeypatch)
    mutation(terminal)
    terminal_path = tmp_path / analysis.RECOVERY_V2_TERMINAL
    _write_json(terminal_path, terminal)
    # Isolate the semantic validator under test after separately proving that
    # a byte mutation is rejected by the compiled terminal pin.
    monkeypatch.setattr(analysis, "RECOVERY_V2_TERMINAL_SHA256", legacy._sha256(terminal_path))
    monkeypatch.setattr(analysis, "RECOVERY_V2_TERMINAL_SIZE", terminal_path.stat().st_size)
    with pytest.raises(analysis.GovernanceError, match=message):
        analysis._validate_training_bundle_v2(tmp_path, deep=True)


def test_runtime_bindings_are_scoped_and_restored(tmp_path: Path) -> None:
    originals = {name: getattr(legacy, name) for name in analysis._RUNTIME_BINDINGS}
    with pytest.raises(RuntimeError, match="sentinel"), analysis._scoped_legacy_runtime():
        assert legacy.downstream_root(tmp_path) == tmp_path / "downstream_v2"
        assert legacy.analysis_root(tmp_path) == tmp_path / "downstream_v2/analysis"
        assert legacy.blind_path(tmp_path, "cptac_primary") == (
            tmp_path / "downstream/inputs/label_blind/cptac_primary.csv"
        )
        raise RuntimeError("sentinel")
    assert {name: getattr(legacy, name) for name in analysis._RUNTIME_BINDINGS} == originals


def test_score_jobs_and_worker_commands_are_v2_owned(tmp_path: Path) -> None:
    jobs = analysis._score_jobs_v2(tmp_path)
    assert len(jobs) == 50
    assert sum(job["expected_rows"] for job in jobs) == 4_790
    assert len({job["job_id"] for job in jobs}) == 50
    assert all("/downstream_v2/scores/" in job["output"] for job in jobs)
    assert all("/downstream/inputs/label_blind/" in job["manifest"] for job in jobs)
    command = analysis._internal_score_command_v2(tmp_path, jobs[0], device="cuda", num_workers=4)
    assert command[0] == sys.executable
    assert Path(command[1]).resolve() == Path(analysis.__file__).resolve()
    assert command[2] == "_score-one"


def test_expected_final_inventory_is_exact_and_isolated(tmp_path: Path) -> None:
    expected = analysis._expected_final_inventory(tmp_path)
    assert len(expected) == 111
    assert all((tmp_path / "downstream_v2") in path.parents for path in expected)
    assert {
        path.relative_to(tmp_path / "downstream_v2/analysis").as_posix()
        for path in expected
        if (tmp_path / "downstream_v2/analysis") in path.parents
    } == set(analysis.ANALYSIS_FILES)
    assert not any((tmp_path / "analysis") in path.parents for path in expected)
    assert len(analysis.FINAL_SOURCE_GRAPH_KEYS) == 13
    assert len(set(analysis.FINAL_SOURCE_GRAPH_KEYS)) == 13
    assert (
        sum(
            key.startswith("downstream_v2_analysis_")
            or key
            in {
                "downstream_v2_patient_native_logits",
                "downstream_v2_bootstrap_distributions",
                "downstream_v2_results",
            }
            for key in analysis.FINAL_SOURCE_GRAPH_KEYS
        )
        == 5
    )


def test_write_guard_refuses_old_and_root_analysis(tmp_path: Path) -> None:
    with pytest.raises(analysis.GovernanceError, match="outside downstream_v2"):
        analysis._write_json_v2_once(tmp_path / "downstream/new.json", {}, root=tmp_path)
    with pytest.raises(analysis.GovernanceError, match="outside downstream_v2"):
        analysis._write_json_v2_once(tmp_path / "analysis/new.json", {}, root=tmp_path)
    analysis._write_json_v2_once(tmp_path / "downstream_v2/new.json", {}, root=tmp_path)
    assert (tmp_path / "downstream_v2/new.json").is_file()


def test_root_analysis_namespace_is_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "analysis").mkdir()
    with pytest.raises(analysis.GovernanceError, match="root/analysis is forbidden"):
        analysis._assert_root_analysis_absent(tmp_path)


def test_prepare_publishes_two_files_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = {"terminal": {"path": "terminal"}}
    bundle = {"status": "legacy", "snapshot_frames": {}}
    monkeypatch.setattr(analysis, "_validate_training_bundle_v2", lambda root, deep=True: training)
    monkeypatch.setattr(analysis, "validate_legacy_prepared_bundle", lambda root: bundle)
    monkeypatch.setattr(
        analysis,
        "_target_state",
        lambda root, value: ({}, {}, {}, {}),
    )
    monkeypatch.setattr(analysis, "_score_jobs_v2", lambda root: [{"job_id": "one"}])

    def payload(*args: Any, created_utc: str) -> dict[str, Any]:
        return {
            "status": "prepared",
            "created_utc": created_utc,
            "plan": args[-1],
        }

    monkeypatch.setattr(analysis, "_contract_payload_v2", payload)

    def load(root: Path, *, deep: bool = True) -> dict[str, Any]:
        namespace = root / "downstream_v2"
        assert {
            path.relative_to(namespace).as_posix()
            for path in namespace.rglob("*")
            if path.is_file()
        } == {"contract.json", "jobs/score_jobs.json"}
        return legacy._read_json(namespace / "contract.json")

    monkeypatch.setattr(analysis, "_load_contract_v2", load)
    result = analysis.prepare(tmp_path, apply=True)
    assert result["status"] == "prepared"
    assert not list(tmp_path.parent.glob(f".{tmp_path.name}.downstream-v2-stage-*"))


def test_prepare_failure_never_publishes_partial_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        analysis, "_validate_training_bundle_v2", lambda root, deep=True: {"terminal": {}}
    )
    monkeypatch.setattr(
        analysis,
        "validate_legacy_prepared_bundle",
        lambda root: {"snapshot_frames": {}},
    )
    monkeypatch.setattr(analysis, "_target_state", lambda root, value: ({}, {}, {}, {}))
    monkeypatch.setattr(analysis, "_score_jobs_v2", lambda root: [{"job_id": "one"}])
    monkeypatch.setattr(
        analysis,
        "_contract_payload_v2",
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
    assert not (tmp_path / "downstream_v2").exists()
    assert not list(tmp_path.parent.glob(f".{tmp_path.name}.downstream-v2-stage-*"))


def test_strict_continuation_graph_traverses_job_plan_and_leaves_external_owned(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "downstream_v2"
    plan = namespace / "jobs/score_jobs.json"
    external = tmp_path / "recovery_v2/external.json"
    dangling = tmp_path / "missing.json"
    _write_json(plan, {"jobs": []})
    _write_json(external, {"nested": {"path": str(dangling), "sha256": "0" * 64, "size_bytes": 0}})
    contract = {
        "continuation": {"score_job_plan": _identity(plan)},
        "external_authenticated_leaf": _identity(external),
    }
    _write_json(namespace / "contract.json", contract)
    with analysis._scoped_legacy_runtime():
        analysis._strict_continuation_json_tree(tmp_path, contract)
        plan.write_text("{}\n", encoding="utf-8")
        with pytest.raises(analysis.GovernanceError, match="score-job plan"):
            analysis._strict_continuation_json_tree(tmp_path, contract)


def test_analyze_deep_gates_immediately_before_legacy_analysis(
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

    monkeypatch.setattr(analysis, "_load_contract_v2", load)
    monkeypatch.setattr(legacy, "analyze", delegated)
    assert analysis.analyze(tmp_path, apply=False)["status"] == "dry"
    assert events == ["deep_gate", "analysis"]


def test_delegation_contract_records_every_mutable_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis, "ANALYSIS_V2_TEST", Path(__file__).resolve())
    delegation = analysis._delegation_contract()
    assert delegation["runtime_binding_symbol_roster"] == sorted(analysis._RUNTIME_BINDINGS)
    assert set(delegation["runtime_bindings"]) == {
        f"legacy.{name}" for name in analysis._RUNTIME_BINDINGS
    }
    assert (
        str(Path(analysis.__file__).resolve())
        in delegation["runtime_bindings"]["legacy._internal_score_command"]
    )
    assert delegation["legacy_mutations_allowed"] is False
    assert delegation["root_analysis_allowed"] is False
