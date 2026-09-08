from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from test_final_v13_bundle_receipt import _fixture, _write_json  # noqa: E402

from tools import final_v13_bundle_receipt as verifier  # noqa: E402
from tools import final_v13_full_reconciler as reconciler  # noqa: E402


def _retained_target_internal_fixture() -> dict:
    scopes = reconciler.phase1.ADAPTATION_SCOPES
    native = {
        scope: {
            "census": {
                "patients": 85 if scope == "RIH-M" else 74 if scope == "SurGen-M" else 159,
                "mutant": 37 if scope == "RIH-M" else 30 if scope == "SurGen-M" else 67,
            },
            "auroc": 0.6,
            "ci95": [0.5, 0.7],
        }
        for scope in scopes
    }

    def result(method: str) -> dict:
        cells = {}
        for budget in reconciler.phase1.ADAPTATION_BUDGETS:
            cells[str(budget)] = {
                "scopes": {
                    scope: {
                        method: {
                            "auroc": 0.55,
                            "procedure_sample_sd": 0.02,
                            "ci95": [0.50, 0.60],
                        },
                        f"{method}_minus_native": {
                            "auroc_gain": -0.05,
                            "ci95": [-0.10, 0.0],
                        },
                    }
                    for scope in scopes
                },
                "per_source_seed_expected_auroc": {
                    method: {scope: {"mean": 0.54, "sample_sd": 0.01} for scope in scopes}
                },
            }
        if method == "pure_ridge_linear_probe":
            accounting = {
                "label_blind_embedding_jobs": 10,
                "unique_support_procedures_by_fold": 2000,
                "final_probe_head_decisions": 10000,
                "fit_to_test_cohort_applications": 20000,
                "inner_plus_final_solver_calls": 1330000,
                "source_main_model_fits": 0,
                "residual_adapter_fits": 0,
                "local_mil_fits": 0,
                "full_label_fits": 0,
                "platt_fits": 0,
            }
        else:
            accounting = {
                "reused_sealed_embedding_artifacts": 10,
                "unique_support_procedures_by_fold": 2000,
                "final_residual_head_decisions": 10000,
                "fit_to_test_cohort_applications": 20000,
                "inner_plus_final_solver_calls": 1330000,
                "source_main_model_fits": 0,
                "pure_probe_fits": 0,
                "local_mil_fits": 0,
                "full_label_fits": 0,
                "platt_fits": 0,
            }
        return {
            "native_zero_shot": {"scopes": native},
            "few_shot": {"cells": cells},
            "fit_and_solver_accounting": accounting,
        }

    return {
        "pure_ridge": result("pure_ridge_linear_probe"),
        "residual_ridge": result("source_anchored_residual_ridge"),
        "target_internal_bindings": [
            {
                "id": f"aim2_target_internal.fixture.value_{index:03d}",
                "value": index,
            }
            for index in range(348)
        ],
    }


@pytest.fixture(autouse=True)
def _trust_live_reconciliation_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic fixtures bind the live generators before production freeze."""

    monkeypatch.setattr(
        verifier,
        "EXPECTED_FULL_RECONCILER_SHA256",
        verifier.sha256_file(verifier.FULL_RECONCILER_CODE),
    )
    monkeypatch.setattr(
        verifier,
        "EXPECTED_PHASE1_CANDIDATE_HELPER_SHA256",
        verifier.sha256_file(verifier.PHASE1_CANDIDATE_HELPER),
    )


def _fixture_extensions(paths: verifier.BundlePaths) -> tuple[reconciler.ExtensionSource, ...]:
    parent_manifest = json.loads(paths.parent_manifest.read_text(encoding="utf-8"))
    parent_ids = {record["id"] for record in parent_manifest["artifacts"]}
    manifest = json.loads(
        (paths.final_v13 / verifier.SOURCE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    extensions = []
    for record in manifest["artifacts"]:
        if record["id"] in parent_ids:
            continue
        extensions.append(
            reconciler.ExtensionSource(
                source_id=record["id"],
                aims=tuple(record["aims"]),
                experiments=tuple(record["experiments"]),
                role=record["role"],
                path=paths.repo / record["path"],
            )
        )
    return tuple(sorted(extensions, key=lambda source: source.source_id))


def _install_rendered(paths: verifier.BundlePaths, payload: dict) -> None:
    _write_json(
        paths.final_v13 / verifier.SOURCE_MANIFEST_NAME,
        payload["source_manifest"],
    )
    for name, text in payload["documents"].items():
        (paths.final_v13 / name).write_text(text, encoding="utf-8")
    _write_json(paths.reconciliation_pins, payload["reconciliation_pins"])


def _full_rendered_fixture(
    tmp_path: Path,
) -> tuple[verifier.BundlePaths, tuple[reconciler.ExtensionSource, ...], dict]:
    paths = _fixture(tmp_path, parent_source_count=142)
    extensions = _fixture_extensions(paths)
    payload = reconciler.build_full_payload(paths, extensions)
    _install_rendered(paths, payload)
    paths = dataclasses.replace(
        paths,
        expected_final_document_sha256=dict(payload["reconciliation_pins"]["documents"]),
    )
    return paths, extensions, payload


def test_production_terminal_pins_are_frozen_and_controller_completion_replays() -> None:
    assert len(reconciler.phase1.extension_sources()) == 87
    assert len(reconciler.reconciliation_paths().expected_extension_source_ids) == 87
    assert reconciler.FULL_EXTENSION_SOURCE_COUNT == 87
    assert reconciler.FULL_SOURCE_COUNT == 229
    assert len(reconciler.extension_sources()) == 87
    assert set(reconciler.EXPECTED_TERMINAL_PHASE2_IDENTITIES) == set(
        reconciler.TERMINAL_PHASE2_SOURCE_IDS
    )
    assert reconciler.EXPECTED_TERMINAL_PHASE2_IDENTITIES == {
        "aim3-source-primary-controls-scheduler": (
            27_563,
            "1c9086338a049e519572b44b9adabf76fd52ef3d74d7628e1943709b8cc540c9",
        ),
        "aim3-source-primary-results": (
            53_695,
            "9a8d1366eb00ea01aa85d246eb8aeddc2377e99650a8897deb3a547aa5de3c7e",
        ),
        "aim3-source-primary-scheduler": (
            28_065,
            "f7e380a88b98ad585f7f77fddf73f7fc04995e67268d252490b05435216d5b7f",
        ),
        "aim3-source-primary-training-completion": (
            37_004,
            "a66793097877a9d40fa0164c96706fa603b07aee23a08aabc87eeb5d92e4ea86",
        ),
    }
    paths = reconciler.reconciliation_paths()
    manifest = reconciler.build_source_manifest(paths)
    assert len(manifest["artifacts"]) == 229
    aim3, _, _ = verifier._validate_aim3_results(paths, manifest["artifacts"])
    assert aim3["terminal_analysis"]["analysis_completion"] == {
        "path": verifier.EXPECTED_AIM3_ANALYSIS_COMPLETION_PATH,
        "size_bytes": 909,
        "sha256": "ceff89363f891eb379bde3b675047e22c279158b800dd94e5a5b9cb1c418768d",
    }
    assert aim3["terminal_analysis"]["report_binding_count"] == 65


def test_aim2_conventional_primary_pool_replays_exact_pinned_patient_logits() -> None:
    paths = verifier.default_paths()
    parent_manifest = json.loads(paths.parent_manifest.read_text(encoding="utf-8"))
    pooled = verifier._validate_aim2_conventional_primary_pool(paths, parent_manifest["artifacts"])
    assert pooled == {
        "source_id": verifier.EXPECTED_AIM2_PRIMARY_POOL_SOURCE_ID,
        "encoder": "UNI-v1",
        "datasets": ["cptac_primary", "rih_primary"],
        "patients": 247,
        "mutant": 103,
        "wild_type": 144,
        "auroc": 0.7445051240560949,
        "aggregation": "patient_pooled_five_refit_mean_native_logit",
        "cohort_auroc_averaging": False,
        "governed_pooled_ci_available": False,
        "bindings": [
            {
                "id": "aim2_source.conventional_primary_pool.patients",
                "value": 247,
            },
            {
                "id": "aim2_source.conventional_primary_pool.mutant",
                "value": 103,
            },
            {
                "id": "aim2_source.conventional_primary_pool.wild_type",
                "value": 144,
            },
            {
                "id": "aim2_source.conventional_primary_pool.auroc",
                "value": 0.7445051240560949,
            },
        ],
    }
    rendered = reconciler._priority_2_results(pooled)
    assert "| UNI-v1 | CPTAC-primary + RIH-primary | 247 (103/144) | 0.745 |" in rendered
    assert "It is not an average" in rendered
    assert "No governed pooled\nconfidence interval exists" in rendered
    assert rendered.count("<!-- AIM2_PRIMARY_POOL_VALUE ") == 4
    assert "### Orion retrospective processing sensitivity" in rendered
    assert "| UNI-v1 | Orion | 40 (15/25) | 0.7120 [0.5440, 0.8613] |" in rendered


def test_aim2_conventional_primary_pool_rejects_numeric_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    paths = verifier.default_paths()
    parent_manifest = json.loads(paths.parent_manifest.read_text(encoding="utf-8"))
    source_path = Path(verifier.EXPECTED_AIM2_PRIMARY_POOL_SOURCE_PATH)
    original = pd.read_parquet(source_path)

    def _inverted_read_parquet(path: Path, *, columns: list[str]) -> object:
        assert Path(path) == source_path
        altered = original.copy()
        selected = (
            (altered["analysis_family"] == verifier.EXPECTED_AIM2_PRIMARY_POOL_ANALYSIS_FAMILY)
            & (altered["encoder"] == verifier.EXPECTED_AIM2_PRIMARY_POOL_ENCODER)
            & altered["dataset"].isin(verifier.EXPECTED_AIM2_PRIMARY_POOL_DATASETS)
        )
        for column in (
            *verifier.EXPECTED_AIM2_PRIMARY_POOL_SEED_COLUMNS,
            "mean_logit_5seed",
        ):
            altered.loc[selected, column] = -altered.loc[selected, column]
        return altered.loc[:, columns]

    monkeypatch.setattr(pd, "read_parquet", _inverted_read_parquet)
    with pytest.raises(
        verifier.BundleVerificationError,
        match="patient-pooled AUROC drift",
    ):
        verifier._validate_aim2_conventional_primary_pool(paths, parent_manifest["artifacts"])


def test_aim3_terminal_analysis_completion_pin_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = reconciler.reconciliation_paths()
    manifest = reconciler.build_source_manifest(paths)
    monkeypatch.setattr(
        verifier,
        "EXPECTED_AIM3_ANALYSIS_COMPLETION_SHA256",
        "0" * 64,
    )
    with pytest.raises(
        verifier.BundleVerificationError,
        match="analysis completion pinned identity drift",
    ):
        verifier._validate_aim3_results(paths, manifest["artifacts"])


def test_aim3_gate_replay_uses_99pct_fwer_not_descriptive_95pct_ci() -> None:
    def summary(
        estimate: float,
        ci95: list[float],
        fwer: list[float],
    ) -> dict:
        return {
            "estimate": estimate,
            "ci95_two_sided": ci95,
            "primary_fwer_one_sided": {
                "lower": fwer[0],
                "upper": fwer[1],
                "confidence": 0.99,
            },
        }

    fine = summary(0.53, [0.47, 0.59], [0.46, 0.596])
    control = summary(0.615, [0.559, 0.670], [0.548, 0.680])
    delta = summary(0.084, [0.0118, 0.1571], [-0.0015, 0.1702])
    replayed = verifier._replay_aim3_gate(fine, control, delta, context="audit-trap")
    assert replayed["verdict"] == "INCONCLUSIVE"
    delta["ci95_two_sided"] = [0.08, 0.09]
    assert (
        verifier._replay_aim3_gate(fine, control, delta, context="audit-trap-ci-change") == replayed
    )
    delta["primary_fwer_one_sided"]["lower"] = 0.001
    assert (
        verifier._replay_aim3_gate(fine, control, delta, context="audit-trap-fwer-change")[
            "verdict"
        ]
        == "CEILING"
    )

    g12c_control = summary(0.574, [0.5056, 0.6429], [0.4923, 0.6558])
    g12c = verifier._replay_aim3_gate(
        summary(0.577, [0.5050, 0.6502], [0.4912, 0.6618]),
        g12c_control,
        summary(-0.003, [-0.0755, 0.0698], [-0.0893, 0.0830]),
        context="g12c-audit-trap",
    )
    assert g12c["verdict"] == "UNDERPOWERED"


def test_complete_synthetic_229_source_render_passes_production_verifier(
    tmp_path: Path,
) -> None:
    paths, extensions, payload = _full_rendered_fixture(tmp_path)
    assert payload["scientific_source_count"] == 229
    assert payload["parent_source_count"] == 142
    assert payload["extension_source_count"] == 87
    assert payload["pending_source_count"] == 0
    assert payload["aim2_conventional_primary_pool_binding_count"] == 0
    assert len(payload["source_manifest"]["artifacts"]) == 229
    assert payload["source_manifest"]["pending_artifacts"] == []
    assert payload["reconciliation_pins"]["extension_source_count"] == 87

    checked = reconciler.check_full_bundle(paths, extensions)
    assert checked["status"] == "FULL_RECONCILIATION_VERIFIED_READ_ONLY"
    assert checked["seal_ready"] is True
    assert checked["source_count"] == 229
    assert checked["production_verifier"]["status"] == "READY_TO_SEAL"
    assert not paths.destination.exists()


def test_render_and_check_are_byte_read_only(tmp_path: Path) -> None:
    paths, extensions, _ = _full_rendered_fixture(tmp_path)
    governed = [
        paths.final_v13 / verifier.SOURCE_MANIFEST_NAME,
        *(paths.final_v13 / name for name in verifier.REPORT_DOCUMENTS),
        paths.reconciliation_pins,
    ]
    before = {path: path.read_bytes() for path in governed}
    first = reconciler.build_full_payload(paths, extensions)
    second = reconciler.build_full_payload(paths, extensions)
    assert first == second
    reconciler.check_full_bundle(paths, extensions)
    assert {path: path.read_bytes() for path in governed} == before
    assert not paths.destination.exists()


def test_full_results_render_all_source_bound_metrics_and_consensus(
    tmp_path: Path,
) -> None:
    _, _, payload = _full_rendered_fixture(tmp_path)
    results = payload["documents"]["Results.md"]
    binding_lines = [
        line for line in results.splitlines() if line.startswith("<!-- AIM3_SOURCE_VALUE ")
    ]
    consensus_lines = [
        line for line in results.splitlines() if line.startswith("<!-- AIM3_CONSENSUS ")
    ]
    target_lines = [
        line for line in results.splitlines() if line.startswith("<!-- AIM2_TARGET_INTERNAL_VALUE ")
    ]
    assert len(binding_lines) == 430
    assert len(consensus_lines) == 5
    assert target_lines == []
    assert results.count("### Controlling TCGA+SurGen-primary UNI-v1 five-seed ladder") == 1
    assert results.count("### Repeated three-draw WT-control consensus") == 1
    assert results.count("### Frozen-refit fine-task external/test performance") == 1
    assert "Canonical matched-WT AUROC" in results
    assert "Repeated matched-WT AUROC" in results
    assert "Three-draw WT-control consensus" in results
    assert all(results.count(marker) == 1 for marker in verifier.SECONDARY_MARKERS)
    assert all(
        results.count(marker) == 1 for marker in verifier.SOURCE_ONLY_RESULTS_EXCLUSION_MARKERS
    )
    assert not any(
        marker in results for marker in verifier.FORBIDDEN_INELIGIBLE_RESULTS_NUMERIC_MARKERS
    )
    for document in payload["documents"].values():
        assert document.count(verifier.THEORETICAL_CEILING_NOT_RUN_MARKER) == 1
        assert document.count(verifier.AIM3_CEILING_SCOPE_MARKER) == 1
        assert document.count(verifier.AIM3_FIXED_NONOVERRIDE_MARKER) == 1
    assert "no ceiling performance is reported" in results
    assert "zero pending governed source artifacts" in results
    assert "never establish biological or theoretical absence" in results
    assert "fixed single-draw CEILING cannot override" in results


def test_full_renderer_retains_source_anchored_target_internal_results(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, parent_source_count=142)
    fine = json.loads(
        (Path(paths.expected_aim3_run_root) / "analysis/fine_results.json").read_text()
    )
    full = json.loads((Path(paths.expected_aim3_run_root) / "analysis/results.json").read_text())
    external = json.loads(
        (Path(paths.expected_aim3_fine_external_run_root) / "analysis/results.json").read_text()
    )
    retained = _retained_target_internal_fixture()
    results = reconciler.render_results(
        fine,
        full,
        external,
        retained,
        k32=dict(paths.expected_aim4_k32_sha256),
        curated_from_sha256=paths.expected_aim4_curated_from_sha256,
    )
    assert results.count("<!-- AIM2_TARGET_INTERNAL_VALUE ") == 348
    assert results.count("| Pure ridge linear probe |") == 16
    assert results.count("| Source-anchored residual ridge |") == 16
    assert "TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION" in results
    assert "SurGen-M is source-family exposed" in results
    assert "no positive repair claim is supported" in results
    assert "Legacy non-source-anchored few-shot, local-training" in results
    assert "few-shot analyses above are\nnot a ceiling" in results
    assert "theoretical-ceiling estimand. Independently, target-label use" in results


def test_full_setup_and_results_cover_the_whole_aim1_pipeline(tmp_path: Path) -> None:
    _, _, payload = _full_rendered_fixture(tmp_path)
    setup = payload["documents"]["Experimental_Setup.md"]
    results = payload["documents"]["Results.md"]
    required_setup = (
        "E0 gene-level KRAS OOF\nranking",
        "E1a fixed-score molecular",
        "E1a-S common-support composition",
        "four zero-MIL-fit Why-D",
        "E1d cross-fitted",
        "E1v/cap, E1e, worklist/DCA",
        "extended-RAS/MAPK/pathway-quiet",
        "E1b did not fire",
        "E1c and E0b were not run",
    )
    assert all(marker in setup for marker in required_setup)
    required_results = (
        "### Whole Aim-1 pipeline summary",
        "| E0 main/baseline |",
        "| Paired encoder sensitivity |",
        "| E1a molecular restriction |",
        "| E1a-S composition standardization |",
        "| Why-D |",
        "| E1d clinical value |",
        "| E1v and cap robustness |",
        "| E1e positive control |",
        "| Worklist and DCA |",
        "| Extended-RAS/MAPK/pathway-quiet |",
        "| Conditional branches |",
    )
    assert all(marker in results for marker in required_results)
    for document in (setup, results):
        assert all(document.count(heading) == 1 for heading in verifier.PRIORITY_HEADINGS)
        assert all(document.count(marker) == 1 for marker in verifier.FIREWALL_MARKERS)
    assert "Inherited ALL-primary numeric rows are ineligible and omitted" in results
    assert not any(
        marker in results for marker in verifier.FORBIDDEN_INCREMENTAL_INELIGIBLE_NUMERIC_MARKERS
    )


def test_priority2_has_complete_exact_external_refit_fold5_table(
    tmp_path: Path,
) -> None:
    _, _, payload = _full_rendered_fixture(tmp_path)
    setup = payload["documents"]["Experimental_Setup.md"]
    results = payload["documents"]["Results.md"]
    table = reconciler._external_construction_table()
    assert table in results
    assert len(reconciler.EXTERNAL_CONSTRUCTION_ROWS) == 10
    assert len(reconciler.EXTERNAL_NATIVE_ROWS) == 10
    assert results.count("### Frozen five-seed external/test performance") == 1
    assert results.count("### Post-outcome refit-versus-within-seed-fold5 robustness") == 1
    assert "RIH-Pri + CPTAC" not in table
    assert "All-Met" not in table
    for row in reconciler.EXTERNAL_CONSTRUCTION_ROWS:
        encoder, target, patients, refit, refit_summary, fold5, fold5_summary, role = row
        expected = (
            f"| {encoder} | {target} | {patients} | {refit} | {refit_summary} | "
            f"{fold5} | {fold5_summary} | {role} |"
        )
        assert table.count(expected) == 1
    required_qualifications = (
        "SD is descriptive seed/partition dispersion, not a confidence",
        "post-outcome robustness sensitivity only",
        "every inference artifact was generated label-blind",
        "cannot justify\nchoosing a deployment construction after target outcomes",
        "contributes no main-model\ndevelopment feedback",
    )
    assert all(marker in results for marker in required_qualifications)
    assert "50 label-blind\nscore files and 4,790 slide rows" in setup
    assert "250 label-blind\ncheckpoint-target passes; zero fits" in setup


def test_reconciler_requires_every_repeated_draw(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, parent_source_count=142)
    extensions = _fixture_extensions(paths)
    result_path = Path(paths.expected_aim3_run_root) / "analysis/results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["repeated"]["rungs"]["codon"]["draws"].pop("20260825")
    _write_json(result_path, result)
    with pytest.raises(verifier.BundleVerificationError, match="exact three WT draws"):
        reconciler.build_full_payload(paths, extensions)


def test_reconciler_requires_phase1_and_full_fine_estimates_to_agree(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, parent_source_count=142)
    extensions = _fixture_extensions(paths)
    fine_path = Path(paths.expected_aim3_run_root) / "analysis/fine_results.json"
    fine = json.loads(fine_path.read_text(encoding="utf-8"))
    fine["fine"]["rungs"]["codon"]["five_seed_ensemble"]["estimate"] += 0.001
    fine["report_bindings"] = [
        {
            **binding,
            "value": fine["fine"]["rungs"]["codon"]["five_seed_ensemble"]["estimate"],
        }
        if binding["rung"] == "codon"
        else binding
        for binding in fine["report_bindings"]
    ]
    _write_json(fine_path, fine)
    with pytest.raises(
        verifier.BundleVerificationError,
        match="fine-external internal OOF|fine/full five-seed",
    ):
        reconciler.build_full_payload(paths, extensions)


def test_pin_metadata_breaks_cycle_is_outside_science_and_is_receipt_bound(
    tmp_path: Path,
) -> None:
    paths, extensions, payload = _full_rendered_fixture(tmp_path)
    manifest_paths = {record["path"] for record in payload["source_manifest"]["artifacts"]}
    assert str(paths.reconciliation_pins.relative_to(paths.repo)) not in manifest_paths
    audit = payload["documents"]["Audit.md"]
    assert audit.count(verifier.RECONCILIATION_PINS_BOUNDARY_MARKER) == 1

    receipt = verifier.build_receipt(paths)
    assert receipt["reconciliation_pins"]["identity"] == verifier._identity(
        paths.reconciliation_pins,
        display_path=str(paths.reconciliation_pins.relative_to(paths.repo)),
    )
    assert (
        receipt["reconciliation_pins"]["scientific_source_manifest_membership"]
        == verifier.RECONCILIATION_PINS_MEMBERSHIP
    )
    assert not paths.destination.exists()

    original = paths.verifier_code.read_bytes()
    paths.verifier_code.write_bytes(original + b"# post-pin verifier drift\n")
    with pytest.raises(verifier.BundleVerificationError, match="verifier SHA-256 drift"):
        verifier.check_bundle(paths)
    paths.verifier_code.write_bytes(original)
    assert reconciler.check_full_bundle(paths, extensions)["seal_ready"] is True


def test_pin_schema_is_strict_and_cannot_enter_scientific_ledger(
    tmp_path: Path,
) -> None:
    paths, extensions, payload = _full_rendered_fixture(tmp_path)
    pins = json.loads(paths.reconciliation_pins.read_text(encoding="utf-8"))
    pins["unexpected"] = True
    _write_json(paths.reconciliation_pins, pins)
    with pytest.raises(verifier.BundleVerificationError, match="pin schema is not exact"):
        verifier.check_bundle(paths)

    replacement = dataclasses.replace(
        extensions[-1],
        path=paths.reconciliation_pins,
    )
    invalid_extensions = (*extensions[:-1], replacement)
    invalid_extensions = tuple(sorted(invalid_extensions, key=lambda source: source.source_id))
    with pytest.raises(verifier.BundleVerificationError, match="outside the scientific"):
        reconciler.build_source_manifest(paths, invalid_extensions)
    assert payload["source_manifest"]["pending_artifacts"] == []


def test_render_emits_exact_pin_install_and_verifier_guidance(tmp_path: Path) -> None:
    _, _, payload = _full_rendered_fixture(tmp_path)
    pins = payload["reconciliation_pins"]
    guidance = payload["patch_guidance"]
    resolved = guidance["resolved_verifier_expectations"]
    assert resolved["EXPECTED_FINAL_DOCUMENT_SHA256"] == pins["documents"]
    assert resolved["EXPECTED_EXTENSION_SOURCE_IDS"] == pins["extension_source_ids"]
    assert guidance["install_targets"][verifier.RECONCILIATION_PINS_NAME].endswith(
        verifier.RECONCILIATION_PINS_NAME
    )
    assert "do not embed" in guidance["self_hash_cycle_resolution"]
    assert "--seal exactly once" in guidance["verification_order"][-2]


def test_cli_exposes_only_read_only_render_and_check() -> None:
    parser = reconciler.build_parser()
    assert parser.parse_args(["render"]).action == "render"
    assert parser.parse_args(["check"]).action == "check"
    with pytest.raises(SystemExit):
        parser.parse_args(["seal"])
