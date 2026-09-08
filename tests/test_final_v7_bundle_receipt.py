from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v7_bundle_receipt as bundle  # noqa: E402


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _metric(point: float) -> dict[str, object]:
    return {"point": point, "ci95": [point - 0.1, point + 0.1]}


def _procedure(auroc: float, log_loss: float, brier: float) -> dict[str, object]:
    return {
        "auroc": _metric(auroc),
        "log_loss": _metric(log_loss),
        "brier": _metric(brier),
    }


def _synthetic_report_contract() -> dict[str, object]:
    native = _procedure(0.60, 0.90, 0.30)
    adapted = _procedure(0.62, 0.70, 0.24)
    platt = _procedure(0.59, 0.68, 0.23)
    adapted_delta = _procedure(0.02, -0.20, -0.06)
    platt_delta = _procedure(-0.01, -0.22, -0.07)
    gates = {
        "fixed_gate_adapted": {"pass": True},
        "incremental_improvement_established": {"pass": False},
    }
    primary = {
        "primary_outer_seed": 102,
        "contrast_sign_convention": "procedure minus native",
        "per_cohort": {
            cohort: {
                "n": 3,
                "n_mutant": 1,
                "native": native,
                "adapted": adapted,
                "platt": platt,
                "adapted_minus_native": adapted_delta,
                "platt_minus_native": platt_delta,
            }
            for cohort in ("RIH", "SurGen")
        },
        "macro": {
            "procedures": {"native": native, "adapted": adapted, "platt": platt},
            "contrasts": {
                "adapted_minus_native": adapted_delta,
                "platt_minus_native": platt_delta,
            },
        },
        **gates,
        "bootstrap_draws_requested": 10,
        "bootstrap_draws_valid": 10,
        "bootstrap_seed": 17,
    }
    layouts = {
        str(seed): {
            "outer_seed": seed,
            "is_primary": seed == 102,
            "macro_adapted_auroc": _metric(0.62),
            "macro_adapted_minus_native_auroc": _metric(0.02),
            **gates,
            "bootstrap_draws_requested": 10,
            "bootstrap_draws_valid": 10,
        }
        for seed in (101, 102)
    }
    support = {
        "8": {
            "requested_support": 8,
            "realized_support_min": 8,
            "realized_support_max": 8,
            "exact_balanced_contract": True,
            "n_draws": 2,
            "macro_delta_auroc_mean": 0.01,
            "macro_delta_auroc_sd": 0.02,
            "per_cohort_delta_auroc_mean": {"RIH": -0.01, "SurGen": 0.03},
            "per_cohort_log_loss_mean": {
                "RIH": {"native": 0.9, "adapted": 0.8},
                "SurGen": {"native": 1.0, "adapted": 0.7},
            },
            "per_cohort_brier_mean": {
                "RIH": {"native": 0.3, "adapted": 0.27},
                "SurGen": {"native": 0.32, "adapted": 0.24},
            },
        }
    }
    component_states = {
        "e2f_v3": {"integrity_status": "PASS", "scientific_status": "EXECUTED"},
        "reviews_v5": {
            "integrity_status": "PASS",
            "scientific_status": "GENERATED_UNREAD",
            "analysis_executed": False,
            "unblinding_performed": False,
            "analysis_result": None,
        },
    }
    return {
        "schema_version": 1,
        "component_states": component_states,
        "e2f_v3": {
            "primary_metrics": primary,
            "outer_fold_layouts": layouts,
            "outer_fold_sensitivity_summary": {
                "n_layouts": 2,
                "n_gate_pass": 2,
                "macro_adapted_auroc_range": [0.62, 0.62],
                "macro_delta_auroc_range": [0.02, 0.02],
            },
            "label_efficiency_curve": support,
        },
        "reviews_v5": {
            **component_states["reviews_v5"],
            "census": {
                "cases": 2,
                "images": 2,
                "overviews": 2,
                "panels": 0,
                "exact_images_per_case": 1,
            },
            "sampling_strata": {
                "by_cohort": {"RIH": 1, "SurGen": 1},
                "by_kras": {"mutant": 1, "wild_type": 1},
                "by_p17_group": {"absent": 1, "positive_high": 1},
                "prior_exposed_union": 0,
                "selected_prior_exposure_overlap": 0,
            },
        },
    }


def _synthetic_prefix() -> str:
    return (
        "> **FINAL-v7 CONTROLLING STATUS.** E2f-v3 is `EXECUTED` with integrity "
        "`PASS` and passed the prespecified absolute discrimination gate; "
        "incremental improvement over the native model was not established. The "
        "absolute gate passed in 2 of 2 declared fold layouts; incremental improvement "
        "failed in all 2 declared fold layouts. reviews/v5 is `GENERATED_UNREAD` "
        "(`analysis_executed=false`, `unblinding_performed=false`, "
        "`analysis_result=null`). Accordingly, inherited statements are superseded.\n>\n"
        "> **INTERPRETIVE PRECEDENCE.** This prefix and appendix control inherited text.\n\n"
    )


def _fixture(tmp_path: Path) -> tuple[bundle.BundlePaths, dict[str, Path]]:
    final_v6 = tmp_path / "reports" / "final_v6"
    snapshot_root = (
        tmp_path / "reports" / "snapshots" / "final_v6_pre_v7_20260821"
    )
    snapshot_receipt = (
        tmp_path
        / "reports"
        / "snapshots"
        / "final_v6_pre_v7_20260821_receipt.json"
    )
    final_v7 = tmp_path / "reports" / "final_v7"
    additions = tmp_path / "reports" / "reruns" / "final_v7_additions_20260821"
    integration_root = additions / "integration"
    review_root = tmp_path / "reviews" / "v5"
    addendum_root = (
        tmp_path / "reviews" / "v5_pre_read_nuisance_addendum_20260821"
    )

    for filename in bundle.REPORT_DOCUMENTS:
        _write(final_v6 / filename, f"# final v6 {filename}\nSealed evidence.\n")
    _write_json(
        final_v6 / bundle.INHERITED_PARENT_NAME,
        {"schema_version": 1, "status": "PASS", "bundle": "final_v5"},
    )
    parent = {
        "schema_version": 1,
        "status": "PASS",
        "problems": [],
        "bundle": "reports/final_v6",
        "verification": {
            "markdown": {
                filename: {
                    "sha256": bundle.sha256_file(final_v6 / filename),
                    "bytes": (final_v6 / filename).stat().st_size,
                }
                for filename in bundle.REPORT_DOCUMENTS
            }
        },
    }
    _write_json(final_v6 / bundle.FINAL_RECEIPT_NAME, parent)

    snapshot_root.mkdir(parents=True)
    for source in sorted(final_v6.iterdir()):
        (snapshot_root / source.name).write_bytes(source.read_bytes())
    _write_json(
        snapshot_receipt,
        {
            "schema_version": 1,
            "status": "PASS",
            "problems": [],
            "source": str(final_v6.resolve()),
            "snapshot": str(snapshot_root.resolve()),
            "identities": {
                source.name: {
                    "sha256": bundle.sha256_file(source),
                    "size_bytes": source.stat().st_size,
                }
                for source in sorted(final_v6.iterdir())
            },
        },
    )

    for filename in bundle.REPORT_DOCUMENTS:
        inherited = (final_v6 / filename).read_text(encoding="utf-8")
        _write(
            final_v7 / filename,
            f"# Final-v7 controlling status: {filename}\n\n"
            f"{inherited}"
            "\n## Final-v7 addition\nComplete result.\n",
        )
    (final_v7 / bundle.PARENT_COPY_NAME).write_bytes(
        (final_v6 / bundle.FINAL_RECEIPT_NAME).read_bytes()
    )
    (final_v7 / bundle.INHERITED_PARENT_NAME).write_bytes(
        (final_v6 / bundle.INHERITED_PARENT_NAME).read_bytes()
    )

    shared_input = tmp_path / "inputs" / "sealed_source.csv"
    _write(shared_input, "case_id,label\nP1,1\n")

    e2f_root = additions / "e2f_v3"
    e2f_result = e2f_root / "results.json"
    e2f_code = e2f_root / "run_e2f_v3.py"
    _write(e2f_result, '{"fixed_gate": "PASS"}\n')
    _write(e2f_code, "# frozen E2f-v3 implementation\n")
    e2f_receipt = e2f_root / "receipt.json"
    _write_json(
        e2f_receipt,
        {
            "schema_version": 1,
            "status": "PASS",
            "append_only": True,
            "inputs": [bundle.identity(shared_input)],
            "outputs": [bundle.identity(e2f_result)],
            "code": [bundle.identity(e2f_code)],
        },
    )

    scoring_form = review_root / "FOR_PATHOLOGIST" / "scoring_form.csv"
    reviewer_info = review_root / "FOR_PATHOLOGIST" / "reviewer_info.csv"
    instructions = review_root / "FOR_PATHOLOGIST" / "INSTRUCTIONS.md"
    review_code = review_root / "build_reviews_v5.py"
    _write(
        scoring_form,
        "case_id,assessable,mucin_extent,note\nW001,,,\nW002,,,\n",
    )
    _write(
        reviewer_info,
        "reviewer_id,review_date,years_experience,attestation\n,,,\n",
    )
    _write(instructions, "# Reader instructions\nDo not open the key.\n")
    _write(review_code, "# deterministic packet builder\n")
    review_receipt = review_root / "KEYS_DO_NOT_DISTRIBUTE" / "packet_receipt.json"
    _write_json(
        review_receipt,
        {
            "schema_version": 1,
            "status": "PASS",
            "append_only": True,
            "inputs": [bundle.identity(shared_input)],
            "outputs": [
                bundle.identity(scoring_form),
                bundle.identity(reviewer_info),
                bundle.identity(instructions),
            ],
            "code": [bundle.identity(review_code)],
        },
    )

    parent_analyzer = tmp_path / "tools" / "analyze_reviews_v5.py"
    v5_case_key = review_root / "KEYS_DO_NOT_DISTRIBUTE" / "case_key.csv"
    development_manifest = tmp_path / "external" / "aim1_dev.csv"
    _write(parent_analyzer, "# frozen parent v5 analyzer\n")
    _write(v5_case_key, "case_id,slide_id\nQ00001,S1\n")
    _write(development_manifest, "slide_id,tissue_area_mm2\nS1,10\n")

    blocks = [
        f"{cohort}|{kras}"
        for cohort in ("CPTAC", "RIH", "SurGen", "TCGA")
        for kras in ("mutant", "wild_type")
    ]
    cells = [
        f"{block}|{group}"
        for block in blocks
        for group in ("absent", "positive_high", "positive_low")
    ]
    model_contract = {
        "link": "cumulative_logit_proportional_odds",
        "all_case_adjusted": "mucin ~ p17_z + log_tissue_area_z + blocks",
    }
    bootstrap_contract = {
        "draws": 2000,
        "seed": 20260829,
        "minimum_valid_draws": 1900,
    }
    missingness_contract = {
        "bootstrap_draws": 2000,
        "bootstrap_seed": 20260829,
        "minimum_valid_draws": 1900,
    }
    interpretation_contract = {
        "confirmatory_gate": False,
        "direction_reversal_is_concern": True,
    }
    design_constants = {
        "schema_version": 1,
        "addendum_id": addendum_root.name,
        "case_count": 60,
        "analysis_blocks": blocks,
        "reference_block": blocks[0],
        "sampling_cell_counts": {
            cell: 3 if index < 12 else 2 for index, cell in enumerate(cells)
        },
        "area_rank_tertile_counts": {
            "largest_20": 20,
            "middle_20": 20,
            "smallest_20": 20,
        },
        "scaling": {"p17_sd_all60_ddof0": 0.05},
        "model": model_contract,
        "bootstrap": bootstrap_contract,
        "missingness": missingness_contract,
        "interpretation": interpretation_contract,
    }
    addendum_files = {
        "ANALYSIS_PLAN.md": "# Frozen secondary plan\n",
        "README.md": "# Coordinator-only unread addendum\n",
        "analyze_nuisance.py": "# frozen secondary analyzer\n",
        "seal_addendum.py": "# receipt-last sealer\n",
        "covariate_manifest.csv": "case_id,tissue_area_mm2\nQ00001,10\n",
    }
    for filename, content in addendum_files.items():
        _write(addendum_root / filename, content)
    _write_json(addendum_root / "design_constants.json", design_constants)
    addendum_names = {*addendum_files, "design_constants.json"}
    addendum_test = tmp_path / "tests" / "test_v5_nuisance.py"
    _write(addendum_test, "# focused frozen test\n")
    addendum_outputs = [
        bundle.identity(addendum_root / filename)
        for filename in sorted(addendum_names)
    ]
    addendum_by_name = {
        Path(str(record["path"])).name: record for record in addendum_outputs
    }
    addendum_receipt = addendum_root / "ADDENDUM_RECEIPT.json"
    _write_json(
        addendum_receipt,
        {
            "schema_version": 1,
            "addendum_id": addendum_root.name,
            "status": "PASS",
            "problems": [],
            "scientific_status": "GENERATED_UNREAD_SECONDARY_ADDENDUM",
            "analysis_executed": False,
            "unblinding_performed": False,
            "analysis_result": None,
            "confirmatory_role": "NONE_SECONDARY_ROBUSTNESS_ONLY",
            "parent_primary_unchanged": True,
            "parent_v5_receipt": bundle.identity(review_receipt),
            "parent_v5_frozen_analyzer": bundle.identity(parent_analyzer),
            "v5_case_key": bundle.identity(v5_case_key),
            "development_manifest": bundle.identity(development_manifest),
            "analyzer": addendum_by_name["analyze_nuisance.py"],
            "sealer": addendum_by_name["seal_addendum.py"],
            "plan": addendum_by_name["ANALYSIS_PLAN.md"],
            "readme": addendum_by_name["README.md"],
            "covariate_manifest": addendum_by_name["covariate_manifest.csv"],
            "design_constants": addendum_by_name["design_constants.json"],
            "tests": [bundle.identity(addendum_test)],
            "frozen_inputs": [
                bundle.identity(review_receipt),
                bundle.identity(parent_analyzer),
                bundle.identity(v5_case_key),
                bundle.identity(development_manifest),
            ],
            "outputs": addendum_outputs,
            "model_contract": model_contract,
            "bootstrap_contract": bootstrap_contract,
            "missingness_contract": missingness_contract,
            "interpretation_contract": interpretation_contract,
        },
    )

    integration_input = integration_root / "integration_source.json"
    integration_output = integration_root / "results.json"
    integration_verification = integration_root / "verification.json"
    integration_code = integration_root / "build_final_v7_integration.py"
    _write(integration_input, '{"scope": "final_v7"}\n')
    report_contract = _synthetic_report_contract()
    claim_boundaries = {
        "e2f_v3": "Synthetic E2f claim boundary.",
        "reviews_v5": "Synthetic unread-review claim boundary.",
        "central_claim_boundary": "Synthetic central-claim boundary.",
    }
    report_ready_sentences = {
        "macro_native_auroc": "Synthetic native AUROC sentence.",
        "macro_adapted_auroc": "Synthetic adapted AUROC sentence.",
        "macro_adapted_minus_native_auroc": "Synthetic delta AUROC sentence.",
        "verdict": claim_boundaries["e2f_v3"],
        "pathology_state": claim_boundaries["reviews_v5"],
    }
    _write_json(
        integration_output,
        {
            "schema_version": 1,
            "status": "PASS",
            "component_states": report_contract["component_states"],
            "report_contract": report_contract,
            "claim_boundaries": claim_boundaries,
            "report_ready_sentences": report_ready_sentences,
        },
    )
    _write_json(integration_verification, {"schema_version": 1, "status": "PASS"})
    _write(integration_code, "# append-only integration code\n")
    integration_receipt = integration_root / "receipt.json"
    _write_json(
        integration_receipt,
        {
            "schema_version": 1,
            "status": "PASS",
            "append_only": True,
            "inputs": [bundle.identity(integration_input)],
            "outputs": [
                bundle.identity(integration_output),
                bundle.identity(integration_verification),
            ],
            "code": [bundle.identity(integration_code)],
            "components": {
                "e2f_v3": {
                    "integrity_status": "PASS",
                    "scientific_status": "EXECUTED",
                    "receipt": bundle.identity(e2f_receipt),
                },
                "reviews_v5": {
                    "integrity_status": "PASS",
                    "scientific_status": "GENERATED_UNREAD",
                    "analysis_executed": False,
                    "unblinding_performed": False,
                    "analysis_result": None,
                    "review_root": str(review_root.resolve()),
                    "receipt": bundle.identity(review_receipt),
                    "reader_scoring_form": bundle.identity(scoring_form),
                    "reviewer_info_form": bundle.identity(reviewer_info),
                },
            },
        },
    )

    contract_block = (
        bundle._REPORT_CONTRACT_START
        + bundle._canonical_report_contract(report_contract)
        + bundle._REPORT_CONTRACT_END
    )
    numeric_fragments = "\n".join(bundle._expected_numeric_fragments(report_contract))
    results_appendix = (
        "\n## Final-v7 addition\n"
        + "\n".join(claim_boundaries.values())
        + "\n"
        + "\n".join(report_ready_sentences.values())
        + "\n"
        + numeric_fragments
        + "\n"
        + "GENERATED_UNREAD_SECONDARY_ADDENDUM; "
        + "NONE_SECONDARY_ROBUSTNESS_ONLY; no result.\n"
        + contract_block
        + "\n"
    )
    audit_identities = [
        bundle.identity(e2f_receipt),
        bundle.identity(review_receipt),
        bundle.identity(addendum_receipt),
        bundle.identity(integration_receipt),
        bundle.identity(integration_output),
        bundle.identity(integration_verification),
    ]
    audit_appendix = (
        "\n## Final-v7 audit addition\n"
        + "\n".join(bundle._REQUIRED_AUDIT_CORRECTIONS)
        + "\n"
        + str(addendum_receipt.resolve())
        + "\nGENERATED_UNREAD_SECONDARY_ADDENDUM; analysis_executed=false; "
        + "unblinding_performed=false; analysis_result=null; "
        + "NONE_SECONDARY_ROBUSTNESS_ONLY; parent_primary_unchanged=true.\n"
        + "\n".join(
            f"{record['size_bytes']:,} {record['sha256']}" for record in audit_identities
        )
        + "\n"
    )
    for filename in bundle.REPORT_DOCUMENTS:
        inherited = (final_v6 / filename).read_text(encoding="utf-8")
        appendix = (
            results_appendix
            if filename == "Results.md"
            else audit_appendix
            if filename == "Audit.md"
            else "\n## Final-v7 methods addition\nComplete method.\n"
        )
        _write(final_v7 / filename, _synthetic_prefix() + inherited + appendix)

    paths = bundle.BundlePaths(
        final_v6=final_v6,
        snapshot_root=snapshot_root,
        snapshot_receipt=snapshot_receipt,
        final_v7=final_v7,
        integration_receipt=integration_receipt,
        reviews_v5=review_root,
        v5_pre_read_addendum=addendum_root,
        destination=final_v7 / bundle.FINAL_RECEIPT_NAME,
    )
    material = {
        "shared_input": shared_input,
        "snapshot_results": snapshot_root / "Results.md",
        "parent_copy": final_v7 / bundle.PARENT_COPY_NAME,
        "inherited_parent": final_v7 / bundle.INHERITED_PARENT_NAME,
        "e2f_code": e2f_code,
        "e2f_receipt": e2f_receipt,
        "review_receipt": review_receipt,
        "scoring_form": scoring_form,
        "reviewer_info": reviewer_info,
        "review_root": review_root,
        "addendum_receipt": addendum_receipt,
        "integration_receipt": integration_receipt,
        "integration_results": integration_output,
        "integration_verification": integration_verification,
        "final_v7_results": final_v7 / "Results.md",
        "final_v7_audit": final_v7 / "Audit.md",
    }
    return paths, material


def _append_current_integration_identities(material: dict[str, Path]) -> None:
    audit = material["final_v7_audit"]
    current = audit.read_text(encoding="utf-8")
    records = [
        bundle.identity(material["e2f_receipt"]),
        bundle.identity(material["review_receipt"]),
        bundle.identity(material["addendum_receipt"]),
        bundle.identity(material["integration_receipt"]),
        bundle.identity(material["integration_results"]),
        bundle.identity(material["integration_verification"]),
    ]
    _write(
        audit,
        current
        + "\n"
        + "\n".join(
            f"{record['size_bytes']:,} {record['sha256']}" for record in records
        )
        + "\n",
    )


def _update_integration_component_receipt(
    integration_receipt: Path,
    component: str,
    nested_receipt: Path,
) -> None:
    value = _read_json(integration_receipt)
    components = value["components"]
    assert isinstance(components, dict)
    metadata = components[component]
    assert isinstance(metadata, dict)
    metadata["receipt"] = bundle.identity(nested_receipt)
    _write_json(integration_receipt, value)


def _refresh_review_form_provenance(
    material: dict[str, Path],
    form_key: str,
) -> None:
    form = material[form_key]
    review_receipt = material["review_receipt"]
    review = _read_json(review_receipt)
    outputs = review["outputs"]
    assert isinstance(outputs, list)
    for index, record in enumerate(outputs):
        assert isinstance(record, dict)
        if Path(str(record["path"])).resolve() == form.resolve():
            outputs[index] = bundle.identity(form)
    _write_json(review_receipt, review)

    integration_receipt = material["integration_receipt"]
    integration = _read_json(integration_receipt)
    components = integration["components"]
    assert isinstance(components, dict)
    review_metadata = components["reviews_v5"]
    assert isinstance(review_metadata, dict)
    field = (
        "reader_scoring_form" if form_key == "scoring_form" else "reviewer_info_form"
    )
    review_metadata[field] = bundle.identity(form)
    review_metadata["receipt"] = bundle.identity(review_receipt)
    _write_json(integration_receipt, integration)


def test_complete_synthetic_bundle_verifies_recursively(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)

    observed = bundle.verify_bundle(paths)

    assert observed["status"] == "PASS"
    assert observed["documents"].keys() == set(bundle.REPORT_DOCUMENTS)
    integration = observed["integration_component"]
    assert integration["unique_declared_files"] >= 11
    assert len(integration["recursively_verified_receipts"]) == 3
    assert integration["identity_categories"]["inputs"]["identity_count"] == 1
    assert integration["identity_categories"]["outputs"]["identity_count"] == 2
    assert integration["identity_categories"]["code"]["identity_count"] == 1
    states = integration["scientific_states"]
    assert states["e2f_v3"]["scientific_status"] == "EXECUTED"
    assert states["reviews_v5"]["scientific_status"] == "GENERATED_UNREAD"
    assert states["reviews_v5"]["blank_reader_forms"]["case_rows"] == 2
    addendum = observed["v5_pre_read_nuisance_addendum"]
    assert addendum["scientific_status"] == "GENERATED_UNREAD_SECONDARY_ADDENDUM"
    assert addendum["confirmatory_role"] == "NONE_SECONDARY_ROBUSTNESS_ONLY"
    assert addendum["analysis_result"] is None
    assert observed["report_binding"]["status"] == "PASS"
    assert observed["report_binding"]["report_contract"]["exact_match"] is True
    assert observed["parent_final_v6_and_snapshot"][
        "inherited_parent_final_v5_byte_identical"
    ] is True


def test_pre_read_addendum_state_must_remain_unread_and_secondary(
    tmp_path: Path,
) -> None:
    paths, material = _fixture(tmp_path)
    receipt = _read_json(material["addendum_receipt"])
    receipt["analysis_executed"] = True
    _write_json(material["addendum_receipt"], receipt)

    with pytest.raises(
        bundle.BundleVerificationError,
        match="pre-read addendum analysis_executed",
    ):
        bundle.verify_bundle(paths)


def test_pre_read_addendum_output_tampering_is_rejected(tmp_path: Path) -> None:
    paths, material = _fixture(tmp_path)
    analyzer = paths.v5_pre_read_addendum / "analyze_nuisance.py"
    analyzer.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(
        bundle.BundleVerificationError,
        match="size mismatch|SHA-256 mismatch",
    ):
        bundle.verify_bundle(paths)


@pytest.mark.parametrize(
    ("field", "mutation", "message"),
    [
        ("schema_version", 2, "schema_version"),
        (
            "model_contract",
            {"link": "cumulative_logit_proportional_odds", "changed": True},
            "model contract differs",
        ),
        (
            "missingness_contract",
            {"bootstrap_draws": 999},
            "missingness contract differs",
        ),
    ],
)
def test_pre_read_addendum_schema_and_contracts_are_bound(
    tmp_path: Path,
    field: str,
    mutation: object,
    message: str,
) -> None:
    paths, material = _fixture(tmp_path)
    receipt = _read_json(material["addendum_receipt"])
    receipt[field] = mutation
    _write_json(material["addendum_receipt"], receipt)

    with pytest.raises(bundle.BundleVerificationError, match=message):
        bundle.verify_bundle(paths)


def test_pre_read_addendum_frozen_inputs_must_be_unique_and_named(
    tmp_path: Path,
) -> None:
    paths, material = _fixture(tmp_path)
    receipt = _read_json(material["addendum_receipt"])
    frozen = receipt["frozen_inputs"]
    assert isinstance(frozen, list) and len(frozen) == 4
    frozen[-1] = frozen[0]
    _write_json(material["addendum_receipt"], receipt)

    with pytest.raises(bundle.BundleVerificationError, match="four unique files"):
        bundle.verify_bundle(paths)


def test_pre_read_addendum_state_and_path_are_required_in_audit(
    tmp_path: Path,
) -> None:
    paths, material = _fixture(tmp_path)
    audit = material["final_v7_audit"]
    text = audit.read_text(encoding="utf-8").replace(
        "parent_primary_unchanged=true",
        "parent primary unchanged",
        1,
    )
    _write(audit, text)

    with pytest.raises(bundle.BundleVerificationError, match="unread addendum binding"):
        bundle.verify_bundle(paths)


def test_report_contract_tampering_is_rejected(tmp_path: Path) -> None:
    paths, material = _fixture(tmp_path)
    report = material["final_v7_results"]
    text = report.read_text(encoding="utf-8")
    start = text.index(bundle._REPORT_CONTRACT_START) + len(
        bundle._REPORT_CONTRACT_START
    )
    text = text[:start] + text[start:].replace(
        '"schema_version": 1', '"schema_version": 2', 1
    )
    _write(report, text)

    with pytest.raises(bundle.BundleVerificationError, match="does not exactly match"):
        bundle.verify_bundle(paths)


def test_human_numeric_fragment_must_match_contract(tmp_path: Path) -> None:
    paths, material = _fixture(tmp_path)
    report = material["final_v7_results"]
    text = report.read_text(encoding="utf-8")
    marker = "0.6000 [0.5000, 0.7000]"
    assert marker in text
    prefix, contract_and_after = text.split(bundle._REPORT_CONTRACT_START, 1)
    _write(report, prefix.replace(marker, "transcription-error") + bundle._REPORT_CONTRACT_START + contract_and_after)

    with pytest.raises(bundle.BundleVerificationError, match="not bound to integration fragment"):
        bundle.verify_bundle(paths)


def test_controlling_prefix_state_is_required(tmp_path: Path) -> None:
    paths, material = _fixture(tmp_path)
    report = material["final_v7_results"]
    text = report.read_text(encoding="utf-8").replace("superseded", "overridden", 1)
    _write(report, text)

    with pytest.raises(bundle.BundleVerificationError, match="controlling prefix"):
        bundle.verify_bundle(paths)


def test_audit_semantic_correction_is_required(tmp_path: Path) -> None:
    paths, material = _fixture(tmp_path)
    audit = material["final_v7_audit"]
    correction = bundle._REQUIRED_AUDIT_CORRECTIONS[0]
    text = audit.read_text(encoding="utf-8").replace(correction, "omitted", 1)
    _write(audit, text)

    with pytest.raises(bundle.BundleVerificationError, match="controlling correction"):
        bundle.verify_bundle(paths)


@pytest.mark.parametrize("target", ["snapshot_results", "parent_copy", "inherited_parent"])
def test_snapshot_and_parent_lineage_tampering_fails_closed(
    tmp_path: Path,
    target: str,
) -> None:
    paths, material = _fixture(tmp_path)
    material[target].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(bundle.BundleVerificationError, match="mismatch|byte-identical"):
        bundle.verify_bundle(paths)


@pytest.mark.parametrize("target", ["shared_input", "e2f_code"])
def test_recursive_input_and_code_tampering_fails_closed(
    tmp_path: Path,
    target: str,
) -> None:
    paths, material = _fixture(tmp_path)
    material[target].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(bundle.BundleVerificationError, match="mismatch"):
        bundle.verify_bundle(paths)


def test_nested_identity_without_size_is_rejected(tmp_path: Path) -> None:
    paths, material = _fixture(tmp_path)
    e2f_receipt = material["e2f_receipt"]
    value = _read_json(e2f_receipt)
    inputs = value["inputs"]
    assert isinstance(inputs, list) and isinstance(inputs[0], dict)
    del inputs[0]["size_bytes"]
    _write_json(e2f_receipt, value)
    _update_integration_component_receipt(
        material["integration_receipt"], "e2f_v3", e2f_receipt
    )

    with pytest.raises(bundle.BundleVerificationError, match="both sha256 and size_bytes"):
        bundle.verify_bundle(paths)


def test_explicit_legacy_receipt_is_rehashed_but_not_semantically_recursed(
    tmp_path: Path,
) -> None:
    paths, material = _fixture(tmp_path)
    legacy = material["e2f_receipt"].parent / "legacy_receipt.json"
    _write_json(
        legacy,
        {
            "status": "SEALED_LEGACY",
            "files": [
                {
                    "path": "/historical/location/no-longer-present.csv",
                    "sha256": "1" * 64,
                }
            ],
        },
    )
    e2f_receipt = material["e2f_receipt"]
    value = _read_json(e2f_receipt)
    value["legacy_receipts"] = [bundle.identity(legacy)]
    _write_json(e2f_receipt, value)
    _update_integration_component_receipt(
        material["integration_receipt"], "e2f_v3", e2f_receipt
    )
    _append_current_integration_identities(material)

    observed = bundle.verify_bundle(paths)

    skipped = observed["integration_component"][
        "lineage_only_receipts_directly_rehashed_not_recursed"
    ]
    assert skipped == [str(legacy.resolve())]
    legacy.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(bundle.BundleVerificationError, match="mismatch"):
        bundle.verify_bundle(paths)


@pytest.mark.parametrize(
    ("component", "field", "value", "message"),
    [
        ("e2f_v3", "integrity_status", "FAIL", "E2f-v3 integrity_status"),
        ("e2f_v3", "scientific_status", "PLANNED", "scientific_status"),
        ("reviews_v5", "integrity_status", "FAIL", "reviews/v5 integrity_status"),
        (
            "reviews_v5",
            "scientific_status",
            "EXECUTED",
            "GENERATED_UNREAD",
        ),
        ("reviews_v5", "analysis_executed", True, "analysis_executed"),
        ("reviews_v5", "unblinding_performed", True, "unblinding_performed"),
        ("reviews_v5", "analysis_result", "results.json", "explicit null"),
    ],
)
def test_scientific_and_integrity_states_are_fail_closed(
    tmp_path: Path,
    component: str,
    field: str,
    value: object,
    message: str,
) -> None:
    paths, material = _fixture(tmp_path)
    receipt = material["integration_receipt"]
    payload = _read_json(receipt)
    components = payload["components"]
    assert isinstance(components, dict)
    metadata = components[component]
    assert isinstance(metadata, dict)
    metadata[field] = value
    _write_json(receipt, payload)

    with pytest.raises(bundle.BundleVerificationError, match=message):
        bundle.verify_bundle(paths)


@pytest.mark.parametrize("form_key", ["scoring_form", "reviewer_info"])
def test_completed_reader_cells_are_rejected_even_with_refreshed_hashes(
    tmp_path: Path,
    form_key: str,
) -> None:
    paths, material = _fixture(tmp_path)
    if form_key == "scoring_form":
        _write(
            material[form_key],
            "case_id,assessable,mucin_extent,note\nW001,yes,,\nW002,,,\n",
        )
    else:
        _write(
            material[form_key],
            "reviewer_id,review_date,years_experience,attestation\nR1,,,\n",
        )
    _refresh_review_form_provenance(material, form_key)

    with pytest.raises(bundle.BundleVerificationError, match="is not blank"):
        bundle.verify_bundle(paths)


def test_analysis_result_file_is_rejected_while_metadata_remains_unread(
    tmp_path: Path,
) -> None:
    paths, material = _fixture(tmp_path)
    _write(material["review_root"] / "results.json", '{"unblinded": true}\n')

    with pytest.raises(bundle.BundleVerificationError, match="contains an analysis result"):
        bundle.verify_bundle(paths)


@pytest.mark.parametrize(
    "placeholder",
    ["TODO", "{{ effect_size }}", "SHA256_HERE", "0" * 64],
)
def test_markdown_placeholders_are_rejected(
    tmp_path: Path,
    placeholder: str,
) -> None:
    paths, material = _fixture(tmp_path)
    _write(material["final_v7_results"], f"# Results\n{placeholder}\n")

    with pytest.raises(bundle.BundleVerificationError, match="placeholder"):
        bundle.verify_bundle(paths)


def test_snapshot_receipt_inventory_must_cover_exact_tree(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    _write(paths.snapshot_root / "untracked.txt", "extra\n")

    with pytest.raises(bundle.BundleVerificationError, match="file-set mismatch"):
        bundle.verify_bundle(paths)


def test_final_v7_must_contain_complete_final_v6_bytes_exactly_once(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    inherited = (paths.final_v6 / "Results.md").read_text(encoding="utf-8")
    _write(paths.final_v7 / "Results.md", f"# Header\n{inherited}{inherited}")

    with pytest.raises(bundle.BundleVerificationError, match="exactly once; observed 2"):
        bundle.verify_bundle(paths)


def test_truncated_final_v6_inheritance_is_rejected(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    _write(paths.final_v7 / "Audit.md", "# Header\npartial old audit\n")

    with pytest.raises(bundle.BundleVerificationError, match="exactly once; observed 0"):
        bundle.verify_bundle(paths)


def test_actual_legacy_snapshot_field_names_are_supported(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    payload = _read_json(paths.snapshot_receipt)
    identities = payload["identities"]
    assert isinstance(identities, dict)
    for record in identities.values():
        assert isinstance(record, dict)
        record["bytes"] = record.pop("size_bytes")
    legacy = {
        "schema_version": 1,
        "diff_rq_clean": True,
        "source": "reports/final_v6",
        "snapshot": "snapshots/final_v6_pre_v7_20260821",
        "identities": identities,
    }
    _write_json(paths.snapshot_receipt, legacy)

    assert bundle.verify_bundle(paths)["status"] == "PASS"


def test_atomic_seal_is_written_once_and_refuses_overwrite(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    payload = bundle.verify_bundle(paths)

    bundle.write_json_once_atomic(paths.destination, payload)

    sealed = _read_json(paths.destination)
    assert sealed["status"] == "PASS"
    assert sealed["seal_protocol"]["receipt_written_last"] is True
    assert not list(paths.destination.parent.glob(f".{paths.destination.name}.*.tmp"))
    with pytest.raises(FileExistsError, match="overwrite"):
        bundle.write_json_once_atomic(paths.destination, payload)


def test_destination_must_be_directly_under_final_v7(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    invalid = replace(paths, destination=tmp_path / "elsewhere" / "receipt.json")

    with pytest.raises(bundle.BundleVerificationError, match="directly under final-v7"):
        bundle.verify_bundle(invalid)
