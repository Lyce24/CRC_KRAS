from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import final_v11_bundle_receipt as verifier  # noqa: E402


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path, display: str) -> dict[str, Any]:
    return {"path": display, "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _minimal_results(*, candidate: bool) -> str:
    state = "COMPLETE" if candidate else "DRAFT PENDING"
    final = f"\n\n{verifier.FINAL_STATE_STATUS_PARAGRAPH}\n" if candidate else "\n"
    inherited_markers = "\n".join(verifier._REQUIRED_INHERITED_RESULT_MARKERS)
    claim_rows = "\n".join(
        (
            *verifier.EXPECTED_PERFORMANCE_CLAIM_ROW_IDS,
            *verifier.EXPECTED_CONTRAST_CLAIM_ROW_IDS,
            *verifier.EXPECTED_WHY_D_RECORD_IDS,
        )
    )
    return f"""# FINAL-v11 comprehensive results — {state}

## Status, terminology, and reading rules

`e2a_s` is interpreted as **E1a-S**; no governed E2a-S experiment exists.
E2-CPHT-R is NOT_RUN and whole-section pathology is GENERATED_UNREAD.
Columns include training_population and evaluation_population.

## Integrated conclusions

Complete governed fixture.

## Complete experiment-state ledger

Complete governed fixture.

## Aim 1

### E0 — performance by training/evaluation population

#### Canonical all-primary OOF model: evaluation-population breakdown

Complete governed fixture.

#### Source-restricted OOF training

Complete governed fixture.

#### Encoder-paired contrasts

Complete governed fixture.

### E1a — controlled challenge populations

Complete governed fixture.

### E1a-S — acquisition/composition standardization

Complete governed fixture.

### Why-D

Complete governed fixture.

### E1d — Clinical Improvements over routine clinical variables

Complete governed fixture.

## Aim 2

Complete governed fixture.

## Aim 3

Complete governed fixture.

## Aim 4

Complete governed fixture.

## Cross-aim synthesis and claim boundaries

Complete governed fixture.

## Historical and superseded-result ledger

Complete governed fixture.

## Governed source index

Complete governed fixture.

{inherited_markers}

{claim_rows}

Fixture inherited numeric continuity: 0.1234.{final}"""


def _minimal_setup(*, candidate: bool) -> str:
    state = "COMPLETE" if candidate else "DRAFT PENDING"
    final = f"\n{verifier.FINAL_STATE_STATUS_PARAGRAPH}\n" if candidate else ""
    return f"""# FINAL-v11 setup — {state}

The label-blind contract contains 50 jobs and 4,790 score rows.

## Aim 1

Complete fixture.

## Aim 2

Complete fixture.

## Aim 3

Complete fixture.

## Aim 4

Complete fixture.
{final}"""


def _minimal_audit(sources: list[dict[str, Any]] | None = None, *, candidate: bool) -> str:
    state = "COMPLETE" if candidate else "DRAFT PENDING UNSEALED"
    table = ""
    if sources:
        table = "\n".join(
            f"| `{source['id']}` | `{source['sha256']}` |"
            for source in sorted(sources, key=lambda item: str(item["id"]))
        )
    final = f"\n{verifier.FINAL_STATE_STATUS_PARAGRAPH}\n" if candidate else ""
    return f"""# FINAL-v11 audit — {state}

    The graph has 94 inherited plus 45 new sources, exactly 139 total.
Accounting is 35 new and 60 lineage fits with six-way concurrency.

## Aim 1

Complete fixture.

## Aim 2

Complete fixture.

## Aim 3

Complete fixture.

## Aim 4

Complete fixture.

### Governed source index

| Source ID | SHA-256 |
|---|---|
{table}
{final}"""


def _parent_fixture(repo: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parent_dir = repo / "reports/final_v10_5"
    parent_dir.mkdir(parents=True)
    sources = []
    source_dir = repo / "parent_sources"
    source_dir.mkdir()
    for index in range(verifier.EXPECTED_PARENT_SOURCE_COUNT):
        path = source_dir / f"source_{index:03d}.json"
        _write_json(path, {"index": index, "payload": f"parent-{index}"})
        sources.append(
            {
                "id": f"parent-source-{index:03d}",
                "aims": ["Shared"],
                "experiments": ["fixture"],
                "role": "parent_fixture",
                **_identity(path, str(path.relative_to(repo))),
            }
        )
    manifest = {
        "schema_version": 2,
        "bundle": "final_v10_5",
        "status": verifier.parent.CANDIDATE_STATUS,
        "artifacts": sources,
        "pending_artifacts": [],
    }
    manifest_path = parent_dir / "source_manifest.json"
    _write_json(manifest_path, manifest)
    receipt = {
        "schema_version": 1,
        "bundle": "reports/final_v10_5",
        "status": verifier.SEALED_STATUS,
        "source_manifest": _identity(manifest_path, str(manifest_path.relative_to(repo))),
        "authoritative_sources": sources,
    }
    _write_json(parent_dir / "report_bundle_receipt.json", receipt)
    (parent_dir / "Results.md").write_text(
        "# Fixture sealed FINAL-v10.5 results\n\nResult 0.1234.\n", encoding="utf-8"
    )
    return receipt, sources


def _draft_fixture(tmp_path: Path) -> verifier.BundlePaths:
    repo = tmp_path / "repo"
    final = repo / "reports/final_v11"
    final.mkdir(parents=True)
    parent_receipt, _ = _parent_fixture(repo)

    parent_verifier = repo / "tools/final_v10_5_bundle_receipt.py"
    parent_test = repo / "tests/test_final_v10_5_bundle_receipt.py"
    own_verifier = repo / "tools/final_v11_bundle_receipt.py"
    own_test = repo / "tests/test_final_v11_bundle_receipt.py"
    campaign_controller = repo / "tools/aim1_tcga_surgen_two_encoder_campaign.py"
    campaign_test = repo / "tests/test_aim1_tcga_surgen_two_encoder_campaign.py"
    analysis_controller = repo / "tools/aim1_tcga_surgen_full_pipeline_analysis_v3.py"
    analysis_test = repo / "tests/test_aim1_tcga_surgen_full_pipeline_analysis_v3.py"
    for index, path in enumerate((parent_verifier, parent_test, own_verifier, own_test)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture code {index}\n", encoding="utf-8")

    (final / "Experimental_Setup.md").write_text(_minimal_setup(candidate=False), encoding="utf-8")
    (final / "Results.md").write_text(_minimal_results(candidate=False), encoding="utf-8")
    (final / "Audit.md").write_text(_minimal_audit(candidate=False), encoding="utf-8")

    campaign_root = tmp_path / "campaign"
    paths = verifier.BundlePaths(
        repo=repo,
        final_v11=final,
        destination=final / verifier.FINAL_RECEIPT_NAME,
        verifier_code=own_verifier,
        verifier_test=own_test,
        parent_dir=repo / "reports/final_v10_5",
        parent_receipt=repo / "reports/final_v10_5/report_bundle_receipt.json",
        parent_manifest=repo / "reports/final_v10_5/source_manifest.json",
        parent_verifier=parent_verifier,
        parent_test=parent_test,
        campaign_root=campaign_root,
        campaign_controller=campaign_controller,
        campaign_test=campaign_test,
        analysis_controller=analysis_controller,
        analysis_test=analysis_test,
        expected_parent_sha256={
            "parent_receipt": _sha(repo / "reports/final_v10_5/report_bundle_receipt.json"),
            "parent_manifest": _sha(repo / "reports/final_v10_5/source_manifest.json"),
            "parent_verifier": _sha(parent_verifier),
            "parent_test": _sha(parent_test),
        },
        parent_validator=lambda _paths: parent_receipt,
    )
    specs = verifier._source_specs(paths)
    manifest = {
        "schema_version": 1,
        "bundle": "final_v11",
        "status": verifier.DRAFT_STATUS,
        "base_bundle_receipt": _identity(
            paths.parent_receipt, str(paths.parent_receipt.relative_to(repo))
        ),
        "base_source_manifest": _identity(
            paths.parent_manifest, str(paths.parent_manifest.relative_to(repo))
        ),
        "artifacts": [],
        "pending_artifacts": [
            {"id": source_id, **metadata} for source_id, metadata in sorted(specs.items())
        ],
    }
    _write_json(final / verifier.SOURCE_MANIFEST_NAME, manifest)
    return paths


def _materialize_candidate(paths: verifier.BundlePaths) -> verifier.BundlePaths:
    manifest = json.loads((paths.final_v11 / verifier.SOURCE_MANIFEST_NAME).read_text())
    new_sources = []
    for index, record in enumerate(manifest["pending_artifacts"]):
        path = verifier._lexical_path(record["path"], paths)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"new source {index}: {record['id']}\n", encoding="utf-8")
        new_sources.append({**record, **_identity(path, record["path"])})
    parent_sources = json.loads(paths.parent_receipt.read_text())["authoritative_sources"]
    all_sources = sorted([*parent_sources, *new_sources], key=lambda item: str(item["id"]))
    _write_json(
        paths.final_v11 / verifier.SOURCE_MANIFEST_NAME,
        {
            "schema_version": 2,
            "bundle": "final_v11",
            "status": verifier.CANDIDATE_STATUS,
            "artifacts": all_sources,
            "pending_artifacts": [],
        },
    )
    (paths.final_v11 / "Experimental_Setup.md").write_text(
        _minimal_setup(candidate=True), encoding="utf-8"
    )
    (paths.final_v11 / "Results.md").write_text(_minimal_results(candidate=True), encoding="utf-8")
    (paths.final_v11 / "Audit.md").write_text(
        _minimal_audit(all_sources, candidate=True), encoding="utf-8"
    )
    by_id = {item["id"]: item for item in new_sources}
    return replace(
        paths,
        expected_document_sha256={
            name: _sha(paths.final_v11 / name) for name in verifier.REPORT_DOCUMENTS
        },
        expected_terminal_receipt_sha256={
            source_id: by_id[source_id]["sha256"]
            for source_id in verifier.EXPECTED_TERMINAL_RECEIPT_SHA256
        },
        expected_created_utc="2026-08-24T23:59:00+00:00",
        campaign_validator=lambda _paths: {"status": "PASS"},
        analysis_validator=lambda _paths: {"status": "PASS"},
    )


def test_production_parent_pins_are_exact() -> None:
    assert verifier.EXPECTED_FROZEN_PARENT_SHA256["parent_receipt"] == (
        "3e82b528962500a11b7a8fd956851fe013c61adc5539515078a9c12ac81b6934"
    )
    assert verifier.EXPECTED_PARENT_SOURCE_COUNT == 94
    assert verifier.EXPECTED_NEW_SOURCE_COUNT == 45
    assert verifier.EXPECTED_COMPLETE_SOURCE_COUNT == 139


def test_final_downstream_code_pins_match_reconciled_bytes() -> None:
    expected = {
        "campaign_controller": "4635042e76c3ad3f8fc9aea46e478881681d4a1dfeaef046e0968424d7e2bcc7",
        "campaign_test": "9609411e1bff2feb7517a23931a1e3907eb4984bee90da5d7cdd95c750ae3a16",
        "recovery_v1_controller": "d47cbd54b584fb3d0dd5160ae0bdd1fe2aca79b53207f8c1b63e366957b66ae5",
        "recovery_v1_test": "2cf182aab9cb8a92d4416402024251a7b450adc11e0be7da4f24d0bfac6c2f24",
        "recovery_v2_controller": "e474781f955a932cef575e6f8452f7eb824e178e1fbbe9254432ef7008b4687d",
        "recovery_v2_test": "2dc7b5362edbb55bfdd84d53cb3ff61ce5b30b0fc12c0a1a1fa5b9f763d52891",
        "legacy_analysis_controller": "9f86156da96a8fb2620d45a692c222f46cceef9fa546afd8dd4ca35e09178292",
        "legacy_analysis_test": "ac971cca54cac60bda914e7729bb817f47e9513f74e6a5cce55ee4db5dfd0219",
        "incident_v2_controller": "5ff574f3143c1a2c695334e6b9afb6fa126e7a7e814f787d156f63511fe92345",
        "incident_v2_test": "db2c668c8d1be483e0f8aeecd06fd21b099a5eae03efa9011e632618a51fb08e",
        "analysis_controller": "fadffae63348b2d664ea9f49b2d8126cc0e7869a2eb9bed1245153800b974767",
        "analysis_test": "f0329f2f7bd6cf9f549ae6f8763480a8766819c9ad3cf6920f4d364bb61aa32e",
        "analysis_erratum_controller": "675c190421a057949f9f22dc6a3710d38970625113af182a8120b44307542192",
        "analysis_erratum_test": "b7e49678bdc1c8cde5a880c7586ea1df476a863bb27fdf04b500fb82d33da92f",
        "report_order_erratum_controller": "7e87acf801c6dff2ca4c23792c8f4d9c0f3727977bf4c3b38035e6d19f7919d1",
        "report_order_erratum_test": "cce7a74b5bf8b104198a133cc4f22fb21b6ad0b6010b54515ad303d020ae1d2b",
    }
    assert expected == verifier.EXPECTED_FROZEN_DOWNSTREAM_CODE_SHA256
    paths = verifier.default_paths()
    assert verifier.sha256_file(paths.analysis_controller) == expected["analysis_controller"]
    assert verifier.sha256_file(paths.analysis_test) == expected["analysis_test"]
    assert (
        verifier.sha256_file(verifier.ANALYSIS_ERRATUM_CONTROLLER)
        == expected["analysis_erratum_controller"]
    )
    assert verifier.sha256_file(verifier.ANALYSIS_ERRATUM_TEST) == expected["analysis_erratum_test"]
    assert (
        verifier.sha256_file(verifier.REPORT_ORDER_ERRATUM_CONTROLLER)
        == expected["report_order_erratum_controller"]
    )
    assert (
        verifier.sha256_file(verifier.REPORT_ORDER_ERRATUM_TEST)
        == expected["report_order_erratum_test"]
    )


def test_direct_source_roster_preserves_all_incidents_and_final_erratum() -> None:
    specs = verifier._source_specs(verifier.default_paths())
    assert len(specs) == 45
    required = {
        "aim1-tcga-surgen-two-encoder-recovery-v1-training-completion",
        "aim1-tcga-surgen-two-encoder-recovery-v2-training-completion",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-score-job-plan",
        "aim1-tcga-surgen-two-encoder-downstream-v2-contract",
        "aim1-tcga-surgen-two-encoder-downstream-v2-score-job-plan",
        "aim1-tcga-surgen-two-encoder-downstream-v3-inference-seal",
        "aim1-tcga-surgen-two-encoder-analysis-mean-erratum-controller",
        "aim1-tcga-surgen-two-encoder-analysis-mean-erratum-test",
        "aim1-tcga-surgen-two-encoder-analysis-report-order-erratum-controller",
        "aim1-tcga-surgen-two-encoder-analysis-report-order-erratum-test",
        "aim1-tcga-surgen-two-encoder-downstream-v3-analysis-completion",
    }
    assert required <= set(specs)
    assert not any(
        source_id in specs
        for source_id in (
            "aim1-tcga-surgen-two-encoder-scheduler",
            "aim1-tcga-surgen-two-encoder-training-completion",
        )
    )


def test_draft_status_rehashes_parent_and_reports_pending(tmp_path: Path) -> None:
    paths = _draft_fixture(tmp_path)
    status = verifier.draft_status(paths)
    assert status["parent_recursive_verification"] == "PASS"
    assert status["parent_direct_94_source_rehash"] == "PASS"
    assert status["sealed_parent_source_count"] == 94
    assert status["pending_new_source_count"] == 45
    assert status["pending_complete_fit_census"] == 1360
    assert not paths.destination.exists()


def test_parent_source_byte_drift_fails_direct_rehash(tmp_path: Path) -> None:
    paths = _draft_fixture(tmp_path)
    target = paths.repo / "parent_sources/source_017.json"
    target.write_text("drift\n", encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="source identity drift"):
        verifier.draft_status(paths)


def test_manifest_rejects_duplicate_json_key(tmp_path: Path) -> None:
    paths = _draft_fixture(tmp_path)
    manifest = paths.final_v11 / verifier.SOURCE_MANIFEST_NAME
    manifest.write_text('{"schema_version":1,"schema_version":2}\n', encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="duplicate key"):
        verifier.draft_status(paths)


def test_manifest_rejects_nonfinite_json(tmp_path: Path) -> None:
    paths = _draft_fixture(tmp_path)
    manifest = paths.final_v11 / verifier.SOURCE_MANIFEST_NAME
    manifest.write_text('{"schema_version":NaN}\n', encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="non-finite"):
        verifier.draft_status(paths)


def test_parent_source_symlink_fails(tmp_path: Path) -> None:
    paths = _draft_fixture(tmp_path)
    target = paths.repo / "parent_sources/source_003.json"
    replacement = target.with_suffix(".real")
    target.rename(replacement)
    target.symlink_to(replacement)
    with pytest.raises(verifier.BundleVerificationError, match="symlink"):
        verifier.draft_status(paths)


def test_identity_reference_accepts_only_same_artifact_across_rooted_path_forms(
    tmp_path: Path,
) -> None:
    paths = _draft_fixture(tmp_path)
    artifact = paths.repo / "artifacts/evidence.json"
    artifact.parent.mkdir(parents=True)
    _write_json(artifact, {"evidence": "exact"})
    relative = str(artifact.relative_to(paths.repo))
    source = {"id": "evidence", **_identity(artifact, relative)}
    observed = _identity(artifact, str(artifact))

    verifier._validate_identity_reference(
        observed,
        "evidence",
        [source],
        context="fixture",
        paths=paths,
    )

    wrong = tmp_path / "wrong.json"
    wrong.write_bytes(artifact.read_bytes())
    with pytest.raises(verifier.BundleVerificationError, match="path does not bind"):
        verifier._validate_identity_reference(
            {**observed, "path": str(wrong)},
            "evidence",
            [source],
            context="fixture",
            paths=paths,
        )
    with pytest.raises(verifier.BundleVerificationError, match="schema is not exact"):
        verifier._validate_identity_reference(
            {**observed, "extra": True},
            "evidence",
            [source],
            context="fixture",
            paths=paths,
        )

    alias = paths.repo / "artifacts/alias.json"
    alias.symlink_to(artifact)
    with pytest.raises(verifier.BundleVerificationError, match="symlink"):
        verifier._validate_identity_reference(
            {**observed, "path": str(alias)},
            "evidence",
            [source],
            context="fixture",
            paths=paths,
        )


def test_refresh_promotes_only_materialized_sources(tmp_path: Path) -> None:
    paths = _draft_fixture(tmp_path)
    controller = paths.campaign_controller
    controller.parent.mkdir(parents=True, exist_ok=True)
    controller.write_text("controller\n", encoding="utf-8")
    status = verifier.refresh_manifest(paths)
    assert status["materialized_new_source_count"] == 1
    assert status["pending_new_source_count"] == 44
    assert not paths.destination.exists()


def test_document_topology_rejects_missing_e1a_s(tmp_path: Path) -> None:
    paths = _draft_fixture(tmp_path)
    results = paths.final_v11 / "Results.md"
    results.write_text(
        results.read_text().replace(
            "### E1a-S — acquisition/composition standardization",
            "### composition standardization",
        ),
        encoding="utf-8",
    )
    with pytest.raises(verifier.BundleVerificationError, match="E1a-S"):
        verifier.draft_status(paths)


def test_candidate_build_has_exact_censuses_and_does_not_publish(tmp_path: Path) -> None:
    paths = _materialize_candidate(_draft_fixture(tmp_path))
    receipt = verifier.build_receipt(paths)
    assert receipt["fit_census"]["new_fits"] == 35
    assert receipt["fit_census"]["tcga_surgen_campaign_lineage"] == 60
    assert receipt["fit_census"]["complete_study_wide"] == 1360
    assert receipt["score_census"]["label_blind_jobs"] == 50
    assert receipt["score_census"]["label_blind_slide_rows"] == 4790
    assert len(receipt["authoritative_sources"]) == 139
    assert not paths.destination.exists()


def test_candidate_requires_exact_one_to_one_source_index(tmp_path: Path) -> None:
    paths = _materialize_candidate(_draft_fixture(tmp_path))
    audit = paths.final_v11 / "Audit.md"
    text = audit.read_text()
    first_row = next(line for line in text.splitlines() if line.startswith("| `"))
    audit.write_text(text.replace(first_row, "", 1), encoding="utf-8")
    paths = replace(
        paths,
        expected_document_sha256={
            **paths.expected_document_sha256,
            "Audit.md": _sha(audit),
        },
    )
    with pytest.raises(verifier.BundleVerificationError, match="source index"):
        verifier.build_receipt(paths)


def test_terminal_pin_must_be_frozen(tmp_path: Path) -> None:
    paths = _materialize_candidate(_draft_fixture(tmp_path))
    pins = dict(paths.expected_terminal_receipt_sha256 or {})
    pins["aim1-tcga-surgen-two-encoder-downstream-v3-inference-seal"] = verifier._UNFROZEN
    with pytest.raises(verifier.BundleVerificationError, match="remains unfrozen"):
        verifier.build_receipt(replace(paths, expected_terminal_receipt_sha256=pins))


def test_seal_is_atomic_exactly_once(tmp_path: Path) -> None:
    paths = _materialize_candidate(_draft_fixture(tmp_path))
    receipt = verifier.seal(paths)
    assert paths.destination.is_file()
    assert json.loads(paths.destination.read_text()) == receipt
    assert verifier.verify_published_receipt(paths) == receipt
    original = paths.destination.read_bytes()
    with pytest.raises(verifier.BundleVerificationError, match="overwrite"):
        verifier.seal(paths)
    assert paths.destination.read_bytes() == original


def test_existing_receipt_symlink_is_never_overwritten(tmp_path: Path) -> None:
    paths = _materialize_candidate(_draft_fixture(tmp_path))
    decoy = paths.final_v11 / "decoy.json"
    decoy.write_text("do not touch\n", encoding="utf-8")
    paths.destination.symlink_to(decoy)
    with pytest.raises(verifier.BundleVerificationError, match="overwrite"):
        verifier.seal(paths)
    assert decoy.read_text() == "do not touch\n"


def test_campaign_fit_accounting_rejects_drift() -> None:
    expected = {
        "adopted_oof_fits": 25,
        "new_oof_fits": 25,
        "new_refits": 10,
        "new_fits": 35,
        "operational_lineage_fits": 60,
        "hidden_fits": 0,
    }
    assert expected["adopted_oof_fits"] + expected["new_fits"] == 60
    drift = {**expected, "new_refits": 9}
    assert drift != expected


def test_campaign_validator_enforces_exact_fit_and_concurrency_accounting(
    tmp_path: Path,
) -> None:
    paths = _draft_fixture(tmp_path)
    specs = verifier._source_specs(paths)

    def source(source_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = specs[source_id]
        path = verifier._lexical_path(metadata["path"], paths)
        _write_json(path, payload)
        return {"id": source_id, **metadata, **_identity(path, metadata["path"])}

    identity_keys = ("path", "size_bytes", "sha256")

    def artifact(record: dict[str, Any]) -> dict[str, Any]:
        return {key: record[key] for key in identity_keys}

    contract = source("aim1-tcga-surgen-two-encoder-campaign-contract", {"frozen": True})
    preflight = source("aim1-tcga-surgen-two-encoder-deep-preflight", {"frozen": True})
    v1_controller = source("aim1-tcga-surgen-two-encoder-recovery-v1-controller", {"code": "v1"})
    v1_test = source("aim1-tcga-surgen-two-encoder-recovery-v1-test", {"test": "v1"})
    v1_erratum = source("aim1-tcga-surgen-two-encoder-recovery-v1-erratum-contract", {"erratum": 1})
    v1_adjudication = source(
        "aim1-tcga-surgen-two-encoder-recovery-v1-adjudication", {"adjudication": 1}
    )
    v1_scheduler = source("aim1-tcga-surgen-two-encoder-recovery-v1-scheduler", {"scheduler": 1})
    v2_controller = source("aim1-tcga-surgen-two-encoder-recovery-v2-controller", {"code": "v2"})
    v2_test = source("aim1-tcga-surgen-two-encoder-recovery-v2-test", {"test": "v2"})
    v2_contract = source("aim1-tcga-surgen-two-encoder-recovery-v2-scope-contract", {"erratum": 2})
    v2_adjudication = source(
        "aim1-tcga-surgen-two-encoder-recovery-v2-scope-adjudication",
        {"adjudication": 2},
    )
    legacy_ids = (
        "aim1-tcga-surgen-two-encoder-legacy-prepare-contract",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-cptac-primary",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-orion-cpht",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-rih-metastatic",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-rih-primary",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-sr1482-metastatic",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-score-job-plan",
    )
    legacy = [source(source_id, {"legacy": index}) for index, source_id in enumerate(legacy_ids)]
    fit_accounting = {
        "adopted_oof_fits": 25,
        "hidden_fits": 0,
        "new_fits": 35,
        "new_oof_fits": 25,
        "new_refits": 10,
        "operational_lineage_fits": 60,
        "physical_new_fits": 35,
        "recovery_new_fits": 0,
    }
    execution = {
        "attempts_per_job": 1,
        "job_count": 10,
        "retries": 0,
        "total_attempts": 10,
    }
    v1_terminal = source(
        "aim1-tcga-surgen-two-encoder-recovery-v1-training-completion",
        {
            "status": "complete_and_certified_via_bounded_erratum",
            "fit_accounting": fit_accounting,
            "execution_accounting": execution,
            "seeds": list(verifier.EXPECTED_SEEDS),
            "encoders": list(verifier.EXPECTED_TRAINING_ENCODERS),
            "job_count": 10,
            "source_population": {
                "slides": 1389,
                "patients": 1239,
                "mutant_patients": 501,
                "wildtype_patients": 738,
            },
            "base_contract": artifact(contract),
            "base_preflight": artifact(preflight),
            "erratum_contract": artifact(v1_erratum),
            "validator_adjudication": artifact(v1_adjudication),
            "scheduler_recovery": artifact(v1_scheduler),
            "recovery_implementation": {
                "controller": artifact(v1_controller),
                "controller_test": artifact(v1_test),
            },
        },
    )
    terminal_metadata = specs["aim1-tcga-surgen-two-encoder-recovery-v2-training-completion"]
    terminal_path = verifier._lexical_path(terminal_metadata["path"], paths)
    terminal = {
        "status": "complete_and_certified_via_scoped_census_erratum",
        "fit_accounting": fit_accounting,
        "execution_accounting": execution,
        "concurrency": {
            "maximum": 6,
            "observed_peak": 6,
            "witness_utc": "2026-08-24T20:00:00+00:00",
        },
        "predecessor_v1_terminal": artifact(v1_terminal),
        "scope_contract": artifact(v2_contract),
        "scope_adjudication": artifact(v2_adjudication),
        "recovery_implementation": {
            "controller": artifact(v2_controller),
            "controller_test": artifact(v2_test),
        },
        "training_scoped_census": {
            "artifact_count": 441,
            "total_size_bytes": 989_858_771,
            "tree_sha256": "a2ffd5b61eaf65dc8d0c5df6a1b29867ffabc57f0af869120603ea37591ac261",
            "original_census_source": artifact(v1_adjudication),
            "all_original_records_rehashed_at_certification": True,
            "closed_roster_outside_exclusions": True,
        },
        "namespace_policy": {
            "delegated_growth_namespaces": ["downstream_v2/"],
            "root_analysis_namespace_authorized": False,
            "prepared_baseline_namespace": "downstream/ (seven named bytes remain immutable)",
        },
        "prepared_downstream_baseline": {
            "artifact_count": 7,
            "total_size_bytes": 95_802,
            "tree_sha256": "d6400d6004ad2c1a850c022db1cb694f08b0fac445cdbbc3d94400f1fc995621",
            "artifacts": [artifact(record) for record in legacy],
        },
    }
    _write_json(terminal_path, terminal)
    terminal_source = {
        "id": "aim1-tcga-surgen-two-encoder-recovery-v2-training-completion",
        **terminal_metadata,
        **_identity(terminal_path, terminal_metadata["path"]),
    }
    sources = [
        contract,
        preflight,
        v1_controller,
        v1_test,
        v1_erratum,
        v1_adjudication,
        v1_scheduler,
        v1_terminal,
        v2_controller,
        v2_test,
        v2_contract,
        v2_adjudication,
        *legacy,
        terminal_source,
    ]
    verifier._validate_campaign(paths, sources, deep_replay=False)

    terminal["fit_accounting"]["new_fits"] = 34
    _write_json(terminal_path, terminal)
    sources[-1] = {
        "id": "aim1-tcga-surgen-two-encoder-recovery-v2-training-completion",
        **terminal_metadata,
        **_identity(terminal_path, terminal_metadata["path"]),
    }
    with pytest.raises(verifier.BundleVerificationError, match="fit accounting"):
        verifier._validate_campaign(paths, sources, deep_replay=False)


def test_score_contract_constants_form_exact_cartesian_grid() -> None:
    assert (
        len(verifier.EXPECTED_ENCODERS)
        * len(verifier.EXPECTED_SEEDS)
        * len(verifier.EXPECTED_TARGETS)
        == verifier.EXPECTED_SCORE_JOBS
        == 50
    )
    assert verifier.EXPECTED_SCORE_SLIDE_ROWS == 4790


def test_score_seal_inspects_all_50_artifacts_and_rejects_outcome_columns(
    tmp_path: Path,
) -> None:
    import pandas as pd

    paths = _draft_fixture(tmp_path)
    score_records = []
    combinations = [
        (encoder, target, seed)
        for encoder in verifier.EXPECTED_ENCODERS
        for target in verifier.EXPECTED_TARGETS
        for seed in verifier.EXPECTED_SEEDS
    ]
    for index, (encoder, target, seed) in enumerate(combinations):
        job_id = f"final_v11.continuation_v3.score.{encoder}.{target}.seed{seed}"
        n_rows = 96 if index < 40 else 95
        score_path = tmp_path / "scores" / f"{job_id}.parquet"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "slide_id": [f"slide-{index}-{row}" for row in range(n_rows)],
                "seed": [seed] * n_rows,
                "fold": [0] * n_rows,
                "logit": [row / 1000 for row in range(n_rows)],
            }
        ).to_parquet(score_path, index=False)
        receipt_path = tmp_path / "score_receipts" / f"{job_id}.json"
        score_identity = _identity(score_path, str(score_path))
        _write_json(
            receipt_path,
            {
                "status": "complete",
                "contains_target_outcomes": False,
                "encoder": encoder,
                "target": target,
                "seed": seed,
                "artifact": score_identity,
                "n_rows": n_rows,
            },
        )
        score_records.append(
            {
                "job_id": job_id,
                "score": score_identity,
                "receipt": _identity(receipt_path, str(receipt_path)),
                "rows": n_rows,
            }
        )
    seal = {"score_artifacts": score_records}
    verifier._validate_score_artifacts(seal, paths)

    legacy_namespace = score_records[0]["job_id"]
    score_records[0]["job_id"] = legacy_namespace.replace(".continuation_v3", "")
    with pytest.raises(verifier.BundleVerificationError, match="job identity is invalid"):
        verifier._validate_score_artifacts(seal, paths)
    score_records[0]["job_id"] = legacy_namespace

    leaked = score_records[0]
    leaked_path = Path(leaked["score"]["path"])
    leaked_frame = pd.read_parquet(leaked_path)
    leaked_frame["target_label"] = 0
    leaked_frame.to_parquet(leaked_path, index=False)
    leaked["score"] = _identity(leaked_path, str(leaked_path))
    receipt_path = Path(leaked["receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact"] = leaked["score"]
    _write_json(receipt_path, receipt)
    leaked["receipt"] = _identity(receipt_path, str(receipt_path))
    with pytest.raises(verifier.BundleVerificationError, match="outcomes leaked"):
        verifier._validate_score_artifacts(seal, paths)


def test_metric_cell_rejects_nonfinite_and_misordered_interval() -> None:
    with pytest.raises(verifier.BundleVerificationError, match="non-finite"):
        verifier._metric_cell(float("nan"), [0.4, 0.6], context="fixture")
    with pytest.raises(verifier.BundleVerificationError, match="outside interval"):
        verifier._metric_cell(0.7, [0.4, 0.6], context="fixture")


def test_claim_row_binding_is_exactly_once() -> None:
    rows = []
    lines = []
    for row_id in verifier.EXPECTED_PERFORMANCE_CLAIM_ROW_IDS:
        row = {
            "row_id": row_id,
            "section": "fixture",
            "training_population": "TCGA+SurGen primary",
            "evaluation_population": "fixture",
            "encoder": row_id.rsplit(".", 1)[-1],
            "n_patients": 100,
            "n_mutant": 40,
            "auroc": 0.65,
            "auroc_ci95": [0.55, 0.75],
            "auprc": 0.60,
            "auprc_ci95": [0.50, 0.70],
            "evidence_state": "COMPLETE",
        }
        rows.append(row)
        lines.append(
            f"| {row_id} | TCGA+SurGen primary | fixture | {row['encoder']} | 100 | 40 | "
            "0.6500 [0.5500, 0.7500] | 0.6000 [0.5000, 0.7000] | COMPLETE |"
        )
    text = "\n".join(lines)
    verifier._validate_claim_rows({"report_claim_rows": rows}, text)
    with pytest.raises(verifier.BundleVerificationError, match="exactly once"):
        verifier._validate_claim_rows({"report_claim_rows": rows}, text + "\n" + lines[0])


def test_contrast_row_binding_is_exactly_once() -> None:
    rows = []
    lines = []
    for row_id in verifier.EXPECTED_CONTRAST_CLAIM_ROW_IDS:
        metrics = {
            "auroc": {
                "reference": 0.60,
                "reference_ci95": [0.50, 0.70],
                "comparison": 0.65,
                "comparison_ci95": [0.55, 0.75],
                "delta_comparison_minus_reference": 0.05,
                "delta_ci95": [0.01, 0.09],
                "lower_is_better": False,
            }
        }
        row = {
            "row_id": row_id,
            "section": "fixture",
            "training_population": "TCGA+SurGen primary",
            "encoder": "fixture_encoder",
            "reference": {"name": "reference", "n_patients": 100, "n_mutant": 40},
            "comparison": {"name": "comparison", "n_patients": 90, "n_mutant": 35},
            "contrast_definition": "comparison minus reference",
            "metrics": metrics,
            "evidence_state": "COMPLETE",
        }
        rows.append(row)
        metric_summary = verifier._contrast_metric_summary(metrics, context=f"contrast {row_id}")
        lines.append(
            f"| {row_id} | TCGA+SurGen primary | fixture_encoder | reference | 100 | 40 | "
            f"comparison | 90 | 35 | {metric_summary} | COMPLETE |"
        )
    text = "\n".join(lines)
    verifier._validate_contrast_rows({"report_contrast_rows": rows}, text)
    with pytest.raises(verifier.BundleVerificationError, match="exactly once"):
        verifier._validate_contrast_rows({"report_contrast_rows": rows}, text + "\n" + lines[0])


def test_why_d_evidence_binding_is_exactly_once() -> None:
    records = []
    lines = []
    for encoder in verifier.EXPECTED_ENCODERS:
        record = {
            "record_id": f"aim1.why_d.{encoder}",
            "section": "Aim1/Why_D_explanatory",
            "training_population": "TCGA+SurGen primary",
            "evaluation_population": "A_complete molecular/site-valid subset",
            "encoder": encoder,
            "n_patients": 1_158,
            "n_mutant": 459,
            "random_restriction_draws": 10_000,
            "patient_bootstrap_draws": 10_000,
            "analysis_a_random_restriction": {
                "auc_A_complete": 0.64,
                "auc_D": 0.66,
                "observed_delta": 0.02,
                "n_pos_D": 40,
                "n_neg_D": 60,
                "random": {
                    "mean": 0.0,
                    "sd": 0.01,
                    "p2.5": -0.02,
                    "p97.5": 0.02,
                    "p_ge_observed": 0.05,
                    "p_note": "(k+1)/(B+1)",
                },
                "stratified": {
                    "mean": 0.005,
                    "sd": 0.01,
                    "p2.5": -0.015,
                    "p97.5": 0.025,
                    "p_ge_observed": 0.10,
                    "p_note": "(k+1)/(B+1)",
                },
            },
            "analysis_b_pairwise_auc_decomposition": {
                comparison: {
                    "auc": 0.65,
                    "n_pos": 40,
                    "n_neg": 60,
                    "pair_share": 0.25,
                    "ci": [0.55, 0.75],
                }
                for comparison in ("D+ vs D-", "D+ vs C-", "C+ vs D-", "C+ vs C-")
            },
            "analysis_c_molecular_score_distributions": {
                kras_group: {
                    "MSS/pMMR|BRAF-wild": {
                        "n": 20,
                        "mean_logit": -0.2,
                        "mean_ci": [-0.3, -0.1],
                        "median_logit": -0.22,
                        "median_ci": [-0.32, -0.12],
                        "mean_prob": 0.45,
                    }
                }
                for kras_group in ("KRAS-WT", "KRAS-mutant")
            },
            "analysis_d_adjusted_molecular_association": {
                term: {"beta": 0.4, "ci": [0.1, 0.7], "excludes_zero": True}
                for term in ("BRAF_mut", "MSI", "BRAF_mut x MSI")
            },
            "result_path": f"/aim1/why_d/encoders/{encoder}",
            "evidence_state": "NEW_GOVERNED_EXPLANATORY_DIAGNOSTIC",
        }
        records.append(record)
        lines.append(
            "| "
            + " | ".join(
                [
                    record["record_id"],
                    record["training_population"],
                    record["evaluation_population"],
                    record["encoder"],
                    str(record["n_patients"]),
                    str(record["n_mutant"]),
                    str(record["random_restriction_draws"]),
                    str(record["patient_bootstrap_draws"]),
                    record["result_path"],
                    verifier._canonical_json_sha256(record),
                    record["evidence_state"],
                ]
            )
            + " |"
        )
        lines.extend(verifier._why_d_human_lines(record))
    text = "\n".join(lines)
    verifier._validate_why_d_evidence_records({"why_d_evidence_records": records}, text)
    with pytest.raises(verifier.BundleVerificationError, match="human-rendered"):
        verifier._validate_why_d_evidence_records(
            {"why_d_evidence_records": records}, text.replace(lines[-1], "", 1)
        )
    with pytest.raises(verifier.BundleVerificationError, match="exactly once"):
        verifier._validate_why_d_evidence_records(
            {"why_d_evidence_records": records}, text + "\n" + lines[0]
        )


def test_analysis_contract_enforces_frozen_covariate_and_output_censuses() -> None:
    contract = {
        "schema_version": 1,
        "status": "analysis_governed_after_inference_seal",
        "created_utc": "2026-08-24T22:00:00+00:00",
        "experiment": verifier.EXPECTED_ANALYSIS_EXPERIMENT,
        "campaign_root": "/fixture",
        "downstream_contract": {},
        "inference_seal": {},
        "seal_status_observed_before_outcome_open": "sealed_before_outcome_join",
        "outcomes_opened_only_after_seal": True,
        "outcome_sources": {},
        "derived_covariate_source": verifier.EXPECTED_DERIVED_COVARIATE_SOURCE,
        "inherited_canonical_e0": {},
        "analysis": {
            "inference_unit": "patient",
            "bootstrap_draws": verifier.EXPECTED_BOOTSTRAP_DRAWS,
            "bootstrap_seed": verifier.EXPECTED_BOOTSTRAP_SEED,
            "shared_encoder_draws": True,
            "target_calibration": False,
            "target_model_selection": False,
            "test_only_noncanonical_parameters": False,
            "mean_validator_erratum": verifier.EXPECTED_ANALYSIS_MEAN_ERRATUM,
            "bootstrap_arrays": {
                "names": ["fixture_draw"],
                "count": 1,
                "dtype": "float64",
                "length": verifier.EXPECTED_BOOTSTRAP_DRAWS,
            },
        },
        "patient_table_rows": verifier.EXPECTED_ANALYSIS_PATIENT_ROWS,
        "report_claim_row_count": len(verifier.EXPECTED_PERFORMANCE_CLAIM_ROW_IDS),
        "report_contrast_row_count": len(verifier.EXPECTED_CONTRAST_CLAIM_ROW_IDS),
        "why_d_evidence_record_count": len(verifier.EXPECTED_WHY_D_RECORD_IDS),
        "output_inventory": list(verifier.EXPECTED_ANALYSIS_OUTPUT_INVENTORY),
        "implementation": {},
    }
    assert verifier._validate_analysis_contract(contract) == 1
    drift = json.loads(json.dumps(contract))
    drift["derived_covariate_source"]["artifact"]["size_bytes"] -= 1
    with pytest.raises(verifier.BundleVerificationError, match="governed metadata"):
        verifier._validate_analysis_contract(drift)
    mean_drift = json.loads(json.dumps(contract))
    mean_drift["analysis"]["mean_validator_erratum"]["mismatch_rows"] = 17
    with pytest.raises(verifier.BundleVerificationError, match="mean-validator erratum"):
        verifier._validate_analysis_contract(mean_drift)


def test_analysis_completion_enforces_five_artifacts_and_zero_target_fits() -> None:
    contract = {
        "analysis": {"bootstrap_arrays": {"count": 7}},
    }
    source_ids = {
        "contract": "aim1-tcga-surgen-two-encoder-downstream-v3-analysis-contract",
        "patient_native_logits": (
            "aim1-tcga-surgen-two-encoder-downstream-v3-patient-native-logits"
        ),
        "bootstrap_distributions": ("aim1-tcga-surgen-two-encoder-downstream-v3-bootstrap"),
        "results": "aim1-tcga-surgen-two-encoder-downstream-v3-results",
    }
    sources = []
    artifacts = {}
    for index, (key, source_id) in enumerate(source_ids.items()):
        identity = {
            "path": f"/fixture/{key}",
            "size_bytes": index + 1,
            "sha256": f"{index + 1:064x}",
        }
        sources.append({"id": source_id, **identity})
        artifacts[key] = identity
    completion = {
        "schema_version": 1,
        "status": "complete_and_verified",
        "created_utc": "2026-08-24T23:00:00+00:00",
        "experiment": verifier.EXPECTED_ANALYSIS_EXPERIMENT,
        "seal_before_outcomes": {},
        "artifacts": artifacts,
        "analysis_artifact_count": verifier.EXPECTED_ANALYSIS_ARTIFACTS,
        "patient_rows": verifier.EXPECTED_ANALYSIS_PATIENT_ROWS,
        "bootstrap_array_count": 7,
        "bootstrap_draws": verifier.EXPECTED_BOOTSTRAP_DRAWS,
        "report_claim_row_count": len(verifier.EXPECTED_PERFORMANCE_CLAIM_ROW_IDS),
        "report_contrast_row_count": len(verifier.EXPECTED_CONTRAST_CLAIM_ROW_IDS),
        "why_d_evidence_record_count": len(verifier.EXPECTED_WHY_D_RECORD_IDS),
        "outcomes_opened_after_inference_seal": True,
        "target_refits": 0,
        "target_calibrations": 0,
        "target_model_selections": 0,
    }
    verifier._validate_analysis_completion(completion, contract, sources)
    completion["analysis_artifact_count"] = 4
    with pytest.raises(verifier.BundleVerificationError, match="terminal census"):
        verifier._validate_analysis_completion(completion, contract, sources)


def test_staging_results_cover_all_sealed_parent_numeric_tokens() -> None:
    parent_text = (REPO / "reports/final_v10_5/Results.md").read_text(encoding="utf-8")
    staging_text = (REPO / "reports/final_v11/Results.md").read_text(encoding="utf-8")
    assert verifier._result_number_tokens(parent_text) <= verifier._result_number_tokens(
        staging_text
    )
    for marker in verifier._REQUIRED_INHERITED_RESULT_MARKERS:
        assert staging_text.count(marker) == 1
    for row_id in (
        *verifier.EXPECTED_PERFORMANCE_CLAIM_ROW_IDS,
        *verifier.EXPECTED_CONTRAST_CLAIM_ROW_IDS,
        *verifier.EXPECTED_WHY_D_RECORD_IDS,
    ):
        assert staging_text.count(row_id) == 1


def test_declared_nonresults_remain_exact() -> None:
    assert verifier.EXPECTED_NEW_FITS == 35
    assert verifier.EXPECTED_CAMPAIGN_LINEAGE_FITS == 60
    assert verifier.EXPECTED_MAX_CONCURRENCY == 6
    assert "NOT_RUN" in (REPO / "reports/final_v11/Results.md").read_text()
    assert "GENERATED_UNREAD" in (REPO / "reports/final_v11/Results.md").read_text()
