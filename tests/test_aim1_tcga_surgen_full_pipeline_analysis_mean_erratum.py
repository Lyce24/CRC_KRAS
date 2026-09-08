from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import aim1_tcga_surgen_full_pipeline_analysis as legacy  # noqa: E402
from tools import aim1_tcga_surgen_full_pipeline_analysis_mean_erratum as analysis  # noqa: E402
from tools import aim1_tcga_surgen_full_pipeline_analysis_v3 as v3  # noqa: E402


def _synthetic_patient_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in range(analysis.EXPECTED_EXACT_ROWS):
        rows.append(
            {
                "analysis_family": "source_restricted_tcga_surgen_oof",
                "encoder": "univ1",
                "dataset": "tcga_surgen_source",
                "patient_id": f"exact-{index:04d}",
                "n_slides": 1,
                **{column: 0.0 for column in analysis.SEED_COLUMNS},
                "mean_logit_5seed": 0.0,
            }
        )
    for record in analysis.EXPECTED_DISCREPANCY_RECORDS:
        rows.append(
            {
                "analysis_family": "target_refit_zero_shot_or_sensitivity",
                "encoder": record["encoder"],
                "dataset": record["dataset"],
                "patient_id": record["patient_id"],
                "n_slides": 2,
                **{column: record["recomputed_mean"] for column in analysis.SEED_COLUMNS},
                "mean_logit_5seed": record["stored_associative_mean"],
            }
        )
    return pd.DataFrame(rows)


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": "0" * 64, "size_bytes": 0}


def test_exact_profile_constants_and_synthetic_replay() -> None:
    frame = _synthetic_patient_table()
    profile = analysis._mean_discrepancy_profile(frame)
    assert profile == {
        "patient_rows": 6_342,
        "exact_rows": 6_324,
        "mismatch_rows": 18,
        "block_counts": {
            "univ1|cptac_primary": 3,
            "univ1|sr1482_metastatic": 10,
            "virchow2_cls|cptac_primary": 1,
            "virchow2_cls|sr1482_metastatic": 4,
        },
        "max_abs": 8.881784197001252e-16,
        "rtol": 0.0,
        "atol": 1e-15,
        "allclose": True,
        "records_sha256": ("ea5ef308db921565024fb1fa37598ab1d771d886d2b5a98bc6ba23fcb6b9874f"),
        "records": list(analysis.EXPECTED_DISCREPANCY_RECORDS),
    }


def test_bounded_validator_uses_copy_and_never_mutates_returned_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _synthetic_patient_table()
    before = frame.copy(deep=True)
    observed: list[pd.DataFrame] = []

    def frozen_validator(repaired: pd.DataFrame) -> None:
        assert repaired is not frame
        expected = repaired[list(analysis.SEED_COLUMNS)].mean(axis=1)
        assert np.array_equal(expected, repaired["mean_logit_5seed"])
        observed.append(repaired.copy(deep=True))
        repaired.loc[0, "mean_logit_5seed"] = 99.0

    monkeypatch.setattr(analysis, "_ORIGINAL_PATIENT_VALIDATOR", frozen_validator)
    analysis._bounded_patient_validator(frame)
    assert len(observed) == 1
    assert_frame_equal(frame, before, check_exact=True)


def _set_first_mismatch_exact(frame: pd.DataFrame) -> None:
    index = analysis.EXPECTED_EXACT_ROWS
    frame.loc[index, "mean_logit_5seed"] = frame.loc[index, list(analysis.SEED_COLUMNS)].mean()


def _add_nineteenth_mismatch(frame: pd.DataFrame) -> None:
    frame.loc[0, "mean_logit_5seed"] = np.nextafter(0.0, 1.0)


def _change_record(frame: pd.DataFrame) -> None:
    frame.loc[analysis.EXPECTED_EXACT_ROWS, "patient_id"] = "CPTAC:changed"


def _exceed_tolerance(frame: pd.DataFrame) -> None:
    index = analysis.EXPECTED_EXACT_ROWS
    recomputed = float(frame.loc[index, list(analysis.SEED_COLUMNS)].mean())
    frame.loc[index, "mean_logit_5seed"] = recomputed + 2e-15


def _change_block(frame: pd.DataFrame) -> None:
    frame.loc[analysis.EXPECTED_EXACT_ROWS, "dataset"] = "rih_primary"


def _change_family(frame: pd.DataFrame) -> None:
    frame.loc[analysis.EXPECTED_EXACT_ROWS, "analysis_family"] = "source"


def _change_slide_count(frame: pd.DataFrame) -> None:
    frame.loc[analysis.EXPECTED_EXACT_ROWS, "n_slides"] = 1


def _introduce_nonfinite_seed(frame: pd.DataFrame) -> None:
    frame.loc[analysis.EXPECTED_EXACT_ROWS, analysis.SEED_COLUMNS[0]] = np.inf


def _introduce_nonfinite_mean(frame: pd.DataFrame) -> None:
    frame.loc[analysis.EXPECTED_EXACT_ROWS, "mean_logit_5seed"] = np.nan


@pytest.mark.parametrize(
    "mutator",
    [
        _set_first_mismatch_exact,
        _add_nineteenth_mismatch,
        _change_record,
        _exceed_tolerance,
        _change_block,
        _change_family,
        _change_slide_count,
        _introduce_nonfinite_seed,
        _introduce_nonfinite_mean,
    ],
    ids=[
        "17-mismatches",
        "19-mismatches",
        "record-roster",
        "above-tolerance",
        "block",
        "family",
        "slide-count",
        "nonfinite-seed",
        "nonfinite-mean",
    ],
)
def test_profile_drift_fails_closed(mutator: Callable[[pd.DataFrame], None]) -> None:
    frame = _synthetic_patient_table()
    mutator(frame)
    with pytest.raises(analysis.GovernanceError, match="profile drifted|non-finite"):
        analysis._mean_discrepancy_profile(frame)


def test_profile_schema_and_row_census_fail_closed() -> None:
    frame = _synthetic_patient_table()
    with pytest.raises(analysis.GovernanceError, match="schema/row census"):
        analysis._mean_discrepancy_profile(frame.drop(columns=[analysis.SEED_COLUMNS[0]]))
    with pytest.raises(analysis.GovernanceError, match="schema/row census"):
        analysis._mean_discrepancy_profile(frame.iloc[:-1].copy())


def test_erratum_runtime_is_scoped_and_load_replay_uses_frozen_v3_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = (
        "_validate_patient_output_table",
        "_implementation_sources",
        "_analysis_contract_payload",
        "_load_contract",
        "EXPERIMENT",
    )
    originals = {name: getattr(legacy, name) for name in names}
    events: list[str] = []

    def load(root: Path, *, deep: bool = True) -> dict[str, Any]:
        assert root == tmp_path
        assert deep is True
        assert legacy._implementation_sources is v3._implementation_sources_v3
        assert legacy.EXPERIMENT == v3.EXPERIMENT
        events.append("v3-contract-replay")
        return {"status": "prepared"}

    monkeypatch.setattr(v3, "_load_contract_v3", load)
    with pytest.raises(RuntimeError, match="sentinel"), analysis._scoped_erratum_runtime():
        assert legacy._validate_patient_output_table is analysis._bounded_patient_validator
        assert legacy._implementation_sources is analysis._implementation_sources_erratum
        assert legacy._analysis_contract_payload is analysis._analysis_contract_payload_erratum
        assert legacy._load_contract is analysis._load_contract_erratum
        assert legacy.EXPERIMENT == analysis.EXPERIMENT
        assert legacy._load_contract(tmp_path, deep=True) == {"status": "prepared"}
        assert legacy._implementation_sources is analysis._implementation_sources_erratum
        assert legacy.EXPERIMENT == analysis.EXPERIMENT
        raise RuntimeError("sentinel")
    assert events == ["v3-contract-replay"]
    assert {name: getattr(legacy, name) for name in names} == originals


def test_erratum_contract_and_implementation_bind_exact_tool_and_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = {"analysis": {"bootstrap_draws": 10_000}}
    monkeypatch.setattr(
        analysis,
        "_ORIGINAL_ANALYSIS_CONTRACT_PAYLOAD",
        lambda *args, **kwargs: json.loads(json.dumps(base)),
    )
    payload = analysis._analysis_contract_payload_erratum()
    assert payload["analysis"]["mean_validator_erratum"] == analysis._erratum_contract_record()
    assert base == {"analysis": {"bootstrap_draws": 10_000}}
    implementation = analysis._implementation_sources_erratum()
    assert implementation[-2:] == [
        legacy._artifact(Path(analysis.__file__).resolve()),
        legacy._artifact(Path(__file__).resolve()),
    ]
    assert len({identity["path"] for identity in implementation}) == len(implementation)
    contract = {
        "experiment": analysis.EXPERIMENT,
        "analysis": {"mean_validator_erratum": analysis._erratum_contract_record()},
        "implementation": implementation,
    }
    analysis._validate_analysis_contract_erratum(contract)
    for drifted in (
        {**contract, "implementation": implementation[:-1]},
        {**contract, "implementation": [*implementation, implementation[0]]},
        {**contract, "implementation": list(reversed(implementation))},
    ):
        with pytest.raises(analysis.GovernanceError, match="exact ordered"):
            analysis._validate_analysis_contract_erratum(drifted)


def test_analysis_precondition_rejects_empty_partial_and_symlink(
    tmp_path: Path,
) -> None:
    analysis._analysis_precondition(tmp_path)
    destination = v3.continuation_analysis_root(tmp_path)
    destination.mkdir(parents=True)
    with pytest.raises(analysis.GovernanceError, match="Empty analysis"):
        analysis._analysis_precondition(tmp_path)
    (destination / "partial.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(analysis.GovernanceError, match="Partial or symlinked"):
        analysis._analysis_precondition(tmp_path)
    (destination / "analysis_completion_receipt.json").write_text("{}\n", encoding="utf-8")
    analysis._analysis_precondition(tmp_path)
    for path in destination.iterdir():
        path.unlink()
    destination.rmdir()
    destination.symlink_to(tmp_path)
    with pytest.raises(analysis.GovernanceError, match="Partial or symlinked"):
        analysis._analysis_precondition(tmp_path)


def test_analyze_deep_gates_before_any_delegated_outcome_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def gate(root: Path, *, deep: bool) -> dict[str, Any]:
        assert root == tmp_path
        assert deep is True
        events.append("deep-seal-gate")
        return {}

    def delegated(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("delegated-analysis")
        return {"status": "dry"}

    monkeypatch.setattr(analysis, "_validate_sealed_v3", gate)
    monkeypatch.setattr(legacy, "analyze", delegated)
    assert analysis.analyze(tmp_path, apply=False) == {"status": "dry"}
    assert events == ["deep-seal-gate", "delegated-analysis"]


def test_source_graph_is_exactly_eight_inference_plus_seven_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_identities = {
        "controller": _identity(tmp_path / "v3.py"),
        "controller_test": _identity(tmp_path / "test_v3.py"),
    }
    control_identities = {
        key: _identity(tmp_path / relative)
        for key, (relative, _, _) in analysis.V3_CONTROL_PINS.items()
    }
    monkeypatch.setattr(analysis, "_v3_source_identities", lambda: source_identities)
    monkeypatch.setattr(analysis, "_v3_control_identities", lambda root: control_identities)
    monkeypatch.setattr(legacy, "_artifact", _identity)
    inference, final_analysis = analysis._source_graph(tmp_path)
    assert tuple(inference) == analysis.INFERENCE_SOURCE_KEYS
    assert tuple(final_analysis) == analysis.ANALYSIS_SOURCE_KEYS
    assert len(inference) == 8
    assert len(final_analysis) == 7
    assert len(set(inference) | set(final_analysis)) == 15
    assert len(v3._expected_final_inventory(tmp_path)) == 111


def test_cli_exposes_analysis_only_surface() -> None:
    parser = analysis._build_parser()
    subparsers = next(action for action in parser._actions if getattr(action, "choices", None))
    assert set(subparsers.choices) == {"analyze", "verify", "status"}


def test_frozen_v3_sources_and_live_control_plane_are_exact() -> None:
    assert analysis._v3_source_identities() == {
        "controller": {
            "path": str(Path(v3.__file__).resolve()),
            "sha256": analysis.V3_CONTROLLER_SHA256,
            "size_bytes": analysis.V3_CONTROLLER_SIZE,
        },
        "controller_test": {
            "path": str(v3.ANALYSIS_V3_TEST.resolve()),
            "sha256": analysis.V3_TEST_SHA256,
            "size_bytes": analysis.V3_TEST_SIZE,
        },
    }
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    if not (v3.continuation_root(root) / "inference/inference_seal.json").is_file():
        pytest.skip("Sealed continuation-v3 production namespace is not mounted")
    controls = analysis._v3_control_identities(root)
    assert tuple(controls) == tuple(analysis.V3_CONTROL_PINS)
    assert all(
        legacy._artifact(Path(identity["path"])) == identity for identity in controls.values()
    )


def test_live_public_dry_analyze_is_read_only_and_keeps_outcomes_closed() -> None:
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    namespace = v3.continuation_root(root)
    seal = namespace / "inference/inference_seal.json"
    if not seal.is_file():
        pytest.skip("Sealed continuation-v3 production namespace is not mounted")
    analysis_root = v3.continuation_analysis_root(root)
    if analysis_root.exists() or analysis_root.is_symlink():
        pytest.skip("Live no-output regression requires pre-analysis production state")
    before = {
        path.relative_to(namespace).as_posix(): legacy._artifact(path)
        for path in namespace.rglob("*")
        if path.is_file()
    }
    assert len(before) == 106
    observed = analysis.analyze(root, apply=False)
    assert observed == {
        "status": "dry_run_ready_after_inference_seal",
        "analysis_files": list(v3.ANALYSIS_FILES),
        "bootstrap_draws": 10_000,
        "target_outcomes_opened": False,
    }
    after = {
        path.relative_to(namespace).as_posix(): legacy._artifact(path)
        for path in namespace.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not analysis_root.exists()


def test_live_patient_table_has_exact_bounded_profile_without_mutation() -> None:
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    seal = v3.continuation_root(root) / "inference/inference_seal.json"
    if not seal.is_file():
        pytest.skip("Sealed continuation-v3 production namespace is not mounted")
    with analysis._scoped_erratum_runtime():
        analysis._validate_sealed_v3(root, deep=True)
        outcomes, _ = legacy._open_target_outcomes_after_seal(root)
        source_covariates, _ = legacy._open_source_derived_covariates_after_seal(root)
        patient, _, _ = legacy.build_patient_table(root, outcomes, source_covariates)
    before = patient.copy(deep=True)
    assert analysis._mean_discrepancy_profile(patient)["records_sha256"] == (
        analysis.EXPECTED_PROFILE_SHA256
    )
    analysis._bounded_patient_validator(patient)
    assert_frame_equal(patient, before, check_exact=True)
