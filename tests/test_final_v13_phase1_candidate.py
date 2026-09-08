from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import final_v13_bundle_receipt as verifier  # noqa: E402
from tools import final_v13_full_reconciler as reconciler  # noqa: E402
from tools import final_v13_phase1_candidate as candidate  # noqa: E402


@pytest.fixture(scope="module")
def payload() -> dict:
    return reconciler.build_full_payload()


@pytest.fixture(scope="module")
def evidence(payload: dict) -> dict:
    return verifier._validate_retained_incremental_evidence(
        candidate.candidate_paths(), payload["source_manifest"]["artifacts"]
    )


def test_incremental_ledger_is_exact_and_phase_partitioned() -> None:
    sources = candidate.extension_sources()
    ids = tuple(source.source_id for source in sources)
    assert len(sources) == 87
    assert ids == tuple(sorted(ids))
    assert len(set(ids)) == 87
    assert len({str(source.path) for source in sources}) == 87
    assert sum(not source.pending_phase2 for source in sources) == 83
    assert (
        tuple(source.source_id for source in sources if source.pending_phase2)
        == candidate.PENDING_PHASE2_SOURCE_IDS
    )
    additions = {source.source_id for source in candidate._incremental_sources()}
    assert additions == set(verifier.EXPECTED_INCREMENTAL_ADDITION_SOURCE_IDS)
    assert len(additions) == 40
    assert additions.isdisjoint(verifier.EXPECTED_PHASE1_BASE_EXTENSION_SOURCE_IDS)
    assert ids == verifier.EXPECTED_INCREMENTAL_EXTENSION_SOURCE_IDS
    assert len(verifier.EXPECTED_PHASE1_BASE_EXTENSION_SOURCE_IDS) == 47
    assert len(verifier.EXPECTED_EXTENSION_SOURCE_IDS) == 87


def test_candidate_paths_bind_independent_incremental_authority() -> None:
    paths = candidate.candidate_paths()
    assert tuple(paths.expected_extension_source_ids) == (
        verifier.EXPECTED_INCREMENTAL_EXTENSION_SOURCE_IDS
    )
    assert paths.expected_phase1_manifest_status == verifier.INCREMENTAL_MANIFEST_STATUS
    assert paths.aim3_results_source_id == candidate.AIM3_RESULTS_SOURCE_ID
    assert paths.aim3_controls_scheduler_source_id == (candidate.AIM3_CONTROLS_SCHEDULER_SOURCE_ID)


def test_terminal_payload_has_229_material_and_zero_pending(payload: dict) -> None:
    manifest = payload["source_manifest"]
    assert manifest["status"] == verifier.FINAL_MANIFEST_STATUS
    assert len(manifest["artifacts"]) == 229
    assert manifest["pending_artifacts"] == []
    assert payload["pending_source_count"] == 0
    assert not candidate.candidate_paths().destination.exists()
    assert candidate.candidate_paths().reconciliation_pins.is_file()


def test_terminal_docs_have_exact_topology_firewall_and_status(payload: dict) -> None:
    setup = payload["documents"]["Experimental_Setup.md"]
    results = payload["documents"]["Results.md"]
    for document in (setup, results):
        positions = [document.index(heading) for heading in verifier.PRIORITY_HEADINGS]
        assert positions == sorted(positions)
        assert all(document.count(marker) == 1 for marker in verifier.FIREWALL_MARKERS)
        for cohort, role in verifier.EXTERNAL_TARGET_ROLES.items():
            assert document.count(f"| {cohort} | {role} | frozen_model_zero_shot_only |") == 1
    assert "FINAL_V13_INCREMENTAL_STATUS" not in results
    assert results.count("<!-- AIM3_CONSENSUS ") == 5
    assert all(results.count(marker) == 1 for marker in verifier.SECONDARY_MARKERS)
    assert all(
        results.count(marker) == 1 for marker in verifier.SOURCE_ONLY_RESULTS_EXCLUSION_MARKERS
    )
    for document in payload["documents"].values():
        assert document.count(verifier.THEORETICAL_CEILING_NOT_RUN_MARKER) == 1
        assert document.count(verifier.AIM3_CEILING_SCOPE_MARKER) == 1
        assert document.count(verifier.AIM3_FIXED_NONOVERRIDE_MARKER) == 1


def _comments(text: str, prefix: str) -> list[tuple[str, str]]:
    return re.findall(
        rf"^<!-- {re.escape(prefix)} ([a-z0-9._-]+) (\S+) -->$",
        text,
        flags=re.MULTILINE,
    )


def test_numeric_bindings_are_exact_complete_and_ordered(payload: dict, evidence: dict) -> None:
    results = payload["documents"]["Results.md"]
    full = json.loads((candidate.RUN_ROOT / "analysis/results.json").read_text(encoding="utf-8"))
    external = json.loads(
        (candidate.FINE_EXTERNAL_ROOT / "analysis/results.json").read_text(encoding="utf-8")
    )
    expected_source = [
        (str(binding["id"]), verifier._canonical_number(binding["value"]))
        for binding in full["report_bindings"]
        if binding["metric"] != "consensus_verdict"
    ] + [
        (str(binding["id"]), verifier._canonical_number(binding["value"]))
        for binding in verifier._fine_external_report_bindings(external)
    ]
    expected_fixed: list[tuple[str, str]] = []
    expected_target = [
        (str(binding["id"]), verifier._canonical_number(binding["value"]))
        for binding in evidence["target_internal_bindings"]
    ]
    assert _comments(results, "AIM3_SOURCE_VALUE") == expected_source
    assert _comments(results, "AIM3_CONDITIONAL_FIXED_VALUE") == expected_fixed
    assert _comments(results, "AIM2_TARGET_INTERNAL_VALUE") == expected_target
    assert (len(expected_source), len(expected_fixed), len(expected_target)) == (
        430,
        0,
        348,
    )
    assert len({item[0] for item in expected_target}) == 348


def test_terminal_fixed_controls_do_not_override_repeated_consensus(payload: dict) -> None:
    results = payload["documents"]["Results.md"]
    section = results.split("### Controlling TCGA+SurGen-primary UNI-v1 five-seed ladder", 1)[
        1
    ].split("### Legacy all-valid-patient Aim 3", 1)[0]
    assert "Canonical matched-WT AUROC" in section
    assert "Repeated matched-WT AUROC" in section
    assert "Three-draw WT-control consensus" in section
    assert verifier.AIM3_CEILING_SCOPE_MARKER in section
    assert verifier.AIM3_FIXED_NONOVERRIDE_MARKER in section
    assert section.count("<!-- AIM3_CONSENSUS ") == 5


def test_stale_incremental_renderer_fails_closed_after_terminal_artifacts_exist() -> None:
    with pytest.raises(
        verifier.BundleVerificationError,
        match="Phase-2 pending artifact already exists",
    ):
        candidate.build_candidate_payload()


def test_target_internal_section_is_complete_and_accurately_qualified(
    payload: dict,
) -> None:
    results = payload["documents"]["Results.md"]
    section = results.split(
        "### Source-anchored combined-met TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION",
        maxsplit=1,
    )[1].split(verifier.PRIORITY_HEADINGS[4], maxsplit=1)[0]
    assert "Pure ridge linear probe" in section
    assert "Source-anchored residual ridge" in section
    assert all(f"| {budget} |" in section for budget in (2, 4, 8, 16))
    assert all(name in section for name in candidate.ADAPTATION_SCOPE_NAMES.values())
    assert "paired cohort-wise ordinary patient bootstrap, with one-class draws rejected" in section
    assert "stratified within each metastatic cohort" not in section
    assert "outcome-class-stratified" not in section
    assert "no positive repair claim is supported" in section
    assert candidate.TARGET_INTERNAL_ROLE in section
    assert "SurGen-M is source-family exposed" in section


def test_results_numeric_eligibility_and_aim3_construction_caveat(payload: dict) -> None:
    results = payload["documents"]["Results.md"]
    eligible = (
        "0.6929 [0.6619, 0.7219]",
        "-0.0272 [-0.0487, -0.0058]",
        "0.7261 [0.6947, 0.7579]",
        "+0.0280 [0.0112, 0.0457]",
        "p_ge_observed=9.999e-05",
        "+0.1527 [0.1167, 0.1889]",
    )
    assert all(value in results for value in eligible)
    assert all(
        branch in results
        for branch in (
            "E1v and cap robustness",
            "E1e positive control",
            "Worklist and DCA",
            "Extended-RAS/MAPK/pathway-quiet",
        )
    )
    assert not any(
        marker in results for marker in verifier.FORBIDDEN_INCREMENTAL_INELIGIBLE_NUMERIC_MARKERS
    )
    assert "### Frozen five-seed external/test performance" in results
    assert "### Post-outcome refit-versus-within-seed-fold5 robustness" in results
    assert "cannot supersede\nthe frozen five-seed target results" in results


def test_installed_full_bundle_supersedes_incremental_candidate() -> None:
    paths = candidate.candidate_paths()
    manifest = json.loads(
        (paths.final_v13 / verifier.SOURCE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["status"] == verifier.FINAL_MANIFEST_STATUS
    assert len(manifest["artifacts"]) == 229
    assert manifest["pending_artifacts"] == []
    assert paths.reconciliation_pins.is_file()
    assert not paths.destination.exists()


def test_exact_byte_guard_rejects_false_regeneration(tmp_path: Path) -> None:
    installed = tmp_path / "Results.md"
    installed.write_text("governed result\n", encoding="utf-8")
    with pytest.raises(
        verifier.BundleVerificationError,
        match="differs from trusted deterministic reconciliation",
    ):
        verifier._require_exact_installed_bytes(
            installed,
            b"Scientifically false claim: AUROC 0.9999.\n",
            label="isolated FINAL-v13 tamper fixture",
        )


def test_cli_install_requires_apply_and_exposes_no_seal() -> None:
    parser = candidate.build_parser()
    assert parser.parse_args(["install", "--apply"]).apply is True
    with pytest.raises(SystemExit):
        parser.parse_args(["seal"])
    with pytest.raises(verifier.BundleVerificationError, match="requires explicit --apply"):
        candidate.install_candidate_bundle(apply=False)
