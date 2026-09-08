from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import aim1_tcga_surgen_full_pipeline_analysis as legacy  # noqa: E402
from tools import aim1_tcga_surgen_full_pipeline_analysis_mean_erratum as mean  # noqa: E402
from tools import (  # noqa: E402
    aim1_tcga_surgen_full_pipeline_analysis_report_order_erratum as analysis,
)


def _live_results() -> dict[str, Any]:
    path = mean.v3.continuation_analysis_root(analysis.DEFAULT_CAMPAIGN_ROOT) / "results.json"
    if not path.is_file():
        pytest.skip("Published five-file FINAL-v11 analysis is not mounted")
    return legacy._read_json(path)


def _live_inventory() -> dict[str, dict[str, Any]]:
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    namespace = mean.v3.continuation_root(root)
    if not namespace.is_dir():
        pytest.skip("Published continuation-v3 namespace is not mounted")
    return {
        path.relative_to(namespace).as_posix(): legacy._artifact(path)
        for path in namespace.rglob("*")
        if path.is_file()
    }


def test_frozen_mean_erratum_and_exact_five_analysis_pins() -> None:
    assert analysis._mean_erratum_source_identities() == {
        "mean_erratum_controller": {
            "path": str(Path(mean.__file__).resolve()),
            "sha256": analysis.MEAN_ERRATUM_SHA256,
            "size_bytes": analysis.MEAN_ERRATUM_SIZE,
        },
        "mean_erratum_controller_test": {
            "path": str(mean.ERRATUM_TEST.resolve()),
            "sha256": analysis.MEAN_ERRATUM_TEST_SHA256,
            "size_bytes": analysis.MEAN_ERRATUM_TEST_SIZE,
        },
    }
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    if not mean.v3.continuation_analysis_root(root).is_dir():
        pytest.skip("Published five-file FINAL-v11 analysis is not mounted")
    artifacts = analysis._analysis_artifact_identities(root)
    assert tuple(artifacts) == tuple(analysis.ANALYSIS_PINS)
    assert all(
        legacy._artifact(Path(identity["path"])) == identity for identity in artifacts.values()
    )


def test_exact_order_only_profile_replays_live_rows() -> None:
    observed = analysis._validate_order_only_defect(_live_results())
    assert observed == {
        "status": "exact_id_keyed_rows_with_serialization_order_only_drift",
        "performance_rows": 78,
        "performance_positional_mismatches": 12,
        "contrast_rows": 48,
        "contrast_positional_mismatches": 3,
        "why_d_rows": 2,
        "why_d_positional_mismatches": 0,
        "stored_claim_id_sequence_sha256": analysis.STORED_CLAIM_ID_SHA256,
        "stored_contrast_id_sequence_sha256": analysis.STORED_CONTRAST_ID_SHA256,
        "positional_profile_sha256": analysis.POSITIONAL_PROFILE_SHA256,
        "id_keyed_claim_rows_equal": True,
        "id_keyed_contrast_rows_equal": True,
        "why_d_rows_equal": True,
    }


def test_original_verifier_reproduces_only_report_binding_failure() -> None:
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    if not mean.v3.continuation_analysis_root(root).is_dir():
        pytest.skip("Published five-file FINAL-v11 analysis is not mounted")
    with pytest.raises(analysis.GovernanceError, match="Report bindings do not replay"):
        mean.verify(root)


def _swap_first_two_claims(results: dict[str, Any]) -> None:
    rows = results["report_claim_rows"]
    rows[0], rows[1] = rows[1], rows[0]


def _drop_claim(results: dict[str, Any]) -> None:
    results["report_claim_rows"].pop()


def _duplicate_claim_id(results: dict[str, Any]) -> None:
    results["report_claim_rows"][1]["row_id"] = results["report_claim_rows"][0]["row_id"]


def _alter_claim_value(results: dict[str, Any]) -> None:
    results["report_claim_rows"][0]["auroc"] += 0.001


def _swap_contrasts(results: dict[str, Any]) -> None:
    rows = results["report_contrast_rows"]
    rows[0], rows[1] = rows[1], rows[0]


def _alter_contrast_value(results: dict[str, Any]) -> None:
    metrics = results["report_contrast_rows"][0]["metrics"]
    first = next(iter(metrics.values()))
    first["delta_comparison_minus_reference"] += 0.001


def _swap_why_d(results: dict[str, Any]) -> None:
    rows = results["why_d_evidence_records"]
    rows[0], rows[1] = rows[1], rows[0]


def _alter_why_d(results: dict[str, Any]) -> None:
    results["why_d_evidence_records"][0]["n_patients"] += 1


@pytest.mark.parametrize(
    "mutator",
    [
        _swap_first_two_claims,
        _drop_claim,
        _duplicate_claim_id,
        _alter_claim_value,
        _swap_contrasts,
        _alter_contrast_value,
        _swap_why_d,
        _alter_why_d,
    ],
    ids=[
        "claim-order",
        "claim-count",
        "claim-duplicate",
        "claim-value",
        "contrast-order",
        "contrast-value",
        "why-d-order",
        "why-d-value",
    ],
)
def test_any_stored_row_drift_fails_closed(mutator: Any) -> None:
    results = copy.deepcopy(_live_results())
    mutator(results)
    with pytest.raises(analysis.GovernanceError, match="order-only defect|row census|unique"):
        analysis._validate_order_only_defect(results)


def test_replayed_order_or_mapping_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _live_results()
    original = analysis._ORIGINAL_REPORT_CLAIM_ROWS

    def reordered(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        rows = original(*args, **kwargs)
        rows[0], rows[1] = rows[1], rows[0]
        return rows

    monkeypatch.setattr(analysis, "_ORIGINAL_REPORT_CLAIM_ROWS", reordered)
    with pytest.raises(analysis.GovernanceError, match="order-only defect"):
        analysis._validate_order_only_defect(results)


def test_order_bridge_reorders_without_mutating_replayed_rows() -> None:
    claims, contrasts, _ = analysis._replay_rows(_live_results())
    claims_before = copy.deepcopy(claims)
    contrasts_before = copy.deepcopy(contrasts)
    ordered_claims = analysis._reorder_replayed_rows(
        claims,
        expected_ids=analysis.STORED_CLAIM_IDS,
        replay_sha256=analysis.REPLAY_CLAIM_ID_SHA256,
    )
    ordered_contrasts = analysis._reorder_replayed_rows(
        contrasts,
        expected_ids=analysis.STORED_CONTRAST_IDS,
        replay_sha256=analysis.REPLAY_CONTRAST_ID_SHA256,
    )
    assert claims == claims_before
    assert contrasts == contrasts_before
    assert tuple(row["row_id"] for row in ordered_claims) == analysis.STORED_CLAIM_IDS
    assert tuple(row["row_id"] for row in ordered_contrasts) == analysis.STORED_CONTRAST_IDS
    assert {row["row_id"]: row for row in ordered_claims} == {row["row_id"]: row for row in claims}
    assert {row["row_id"]: row for row in ordered_contrasts} == {
        row["row_id"]: row for row in contrasts
    }


def test_bridge_rejects_unexpected_replay_roster_or_order() -> None:
    claims, _, _ = analysis._replay_rows(_live_results())
    wrong_order = claims.copy()
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    with pytest.raises(analysis.GovernanceError, match="unexpected replay"):
        analysis._reorder_replayed_rows(
            wrong_order,
            expected_ids=analysis.STORED_CLAIM_IDS,
            replay_sha256=analysis.REPLAY_CLAIM_ID_SHA256,
        )
    missing = claims[:-1]
    with pytest.raises(analysis.GovernanceError, match="row census"):
        analysis._reorder_replayed_rows(
            missing,
            expected_ids=analysis.STORED_CLAIM_IDS,
            replay_sha256=analysis.REPLAY_CLAIM_ID_SHA256,
        )
    duplicate = copy.deepcopy(claims)
    duplicate[1]["row_id"] = duplicate[0]["row_id"]
    with pytest.raises(analysis.GovernanceError, match="not unique"):
        analysis._reorder_replayed_rows(
            duplicate,
            expected_ids=analysis.STORED_CLAIM_IDS,
            replay_sha256=analysis.REPLAY_CLAIM_ID_SHA256,
        )


def test_scoped_runtime_restores_both_helpers_after_failure() -> None:
    originals = {
        "_report_claim_rows": legacy._report_claim_rows,
        "_report_contrast_rows": legacy._report_contrast_rows,
    }
    with (
        pytest.raises(RuntimeError, match="sentinel"),
        analysis._scoped_order_verification_runtime(),
    ):
        assert legacy._report_claim_rows is analysis._ordered_claim_replay
        assert legacy._report_contrast_rows is analysis._ordered_contrast_replay
        assert legacy._validate_patient_output_table is mean._bounded_patient_validator
        raise RuntimeError("sentinel")
    assert {
        "_report_claim_rows": legacy._report_claim_rows,
        "_report_contrast_rows": legacy._report_contrast_rows,
    } == originals


def test_public_verify_passes_full_frozen_analysis_without_writes() -> None:
    before = _live_inventory()
    assert len(before) == 111
    observed = analysis.verify(analysis.DEFAULT_CAMPAIGN_ROOT)
    assert observed["status"] == "PASS_WITH_BOUNDED_REPORT_ORDER_VERIFICATION_ERRATUM"
    assert observed["analysis"]["status"] == "PASS"
    assert observed["campaign_artifact_count"] == 111
    assert observed["direct_source_graph_count"] == 17
    assert len(observed["inference_source_graph"]) == 8
    assert len(observed["analysis_source_graph"]) == 7
    assert len(observed["verification_erratum_source_graph"]) == 2
    assert _live_inventory() == before


def test_source_graph_is_exactly_seventeen_and_analysis_stays_exact_five() -> None:
    root = analysis.DEFAULT_CAMPAIGN_ROOT
    if not mean.v3.continuation_analysis_root(root).is_dir():
        pytest.skip("Published five-file FINAL-v11 analysis is not mounted")
    bundle = analysis._validate_exact_published_bundle(root)
    inference, governed_analysis, verification = analysis._verification_source_graph(root, bundle)
    assert tuple(inference) == mean.INFERENCE_SOURCE_KEYS
    assert tuple(governed_analysis) == mean.ANALYSIS_SOURCE_KEYS
    assert tuple(verification) == (
        "report_order_erratum_controller",
        "report_order_erratum_controller_test",
    )
    assert len(set(inference) | set(governed_analysis) | set(verification)) == 17
    assert tuple(bundle["analysis"]) == tuple(analysis.ANALYSIS_PINS)


def test_cli_and_status_are_strictly_read_only() -> None:
    parser = analysis._build_parser()
    subparsers = next(action for action in parser._actions if getattr(action, "choices", None))
    assert set(subparsers.choices) == {"verify", "status"}
    before = _live_inventory()
    observed = analysis.status(analysis.DEFAULT_CAMPAIGN_ROOT)
    assert observed["write_surface"] is False
    assert observed["exact_five_present"] is True
    assert observed["analysis_files_present"] == sorted(mean.v3.ANALYSIS_FILES)
    assert _live_inventory() == before
