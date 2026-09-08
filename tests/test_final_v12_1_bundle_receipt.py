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

from tools import final_v12_1_bundle_receipt as verifier  # noqa: E402


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path, display_path: str) -> dict[str, Any]:
    return {
        "path": display_path,
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _side_document(title: str) -> str:
    return f"""# FINAL-v12.1 {title}

The sealed FINAL-v12 receipt is present and authoritative.

## Aim 1

Integrated Aim 1 design.

## Aim 2

Integrated Aim 2 design.

## Aim 3

Integrated Aim 3 design.

## Aim 4

Integrated Aim 4 design.
"""


def _valid_results() -> str:
    return """# FINAL-v12.1 integrated results for paper selection

## Authority, scope, and reading rules

This standalone report presents results by aim, experiment, training design, and population.

### Cohort taxonomy used throughout

| Display label | Composition |
|---|---|
| **ALL primary** | all conventional primary cohorts |
| **TCGA** | TCGA-COAD and TCGA-READ |
| **SurGen** | SR386 and SR1482 |
| **TCGA+SurGen** | source-restricted union |

### Evidence tiers and inclusion rule

- Excluded from selectable results: superseded, audit-only, NOT_RUN, and GENERATED_UNREAD.

## Integrated conclusions

Complete integrated conclusions.

## Integrated experiment and population availability matrix

This matrix is the guardrail for paper selection.

| Aim | Experiment | Population |
|---|---|---|
| 1 | E0 canonical | ALL, TCGA, SurGen, TCGA+SurGen |
| 1 | E1a | ALL and TCGA+SurGen |
| 2 | E2d1–E2d6 | diagnostic populations |
| 3 | E3 fixed | molecular contrasts |
| 4 | E4 atlas | ALL A/D |

## Aim 1 — gene-level ranking

### E0 — performance by training/evaluation population

#### Design A: canonical ALL-primary OOF model — evaluation-population breakdown

aim1.canonical_e0.pooled_all_primary.univ1
aim1.canonical_e0.tcga.univ1
aim1.canonical_e0.surgen.univ1
aim1.canonical_e0.tcga_surgen.univ1

#### Design B: TCGA+SurGen source-restricted OOF training

aim1.source_restricted_e0.pooled_source.univ1
aim1.source_restricted_e0.tcga.univ1
aim1.source_restricted_e0.surgen.univ1
aim1.source_restricted_e0.tcga_surgen.univ1

#### Design C: separately trained within-source OOF arms

| training_population | evaluation_population | encoder |
|---|---|---|
| TCGA | TCGA | univ1 |
| SurGen | SurGen | univ1 |
| TCGA+SurGen | TCGA+SurGen | univ1 |

### E1a — controlled challenge populations

#### ALL-primary inherited challenge panel

Complete.

#### TCGA+SurGen two-encoder challenge panel

Complete.

### E1a-S — composition standardization

#### TCGA+SurGen two-encoder standardization

Complete.

#### ALL-primary inherited standardization

Complete.

### Why-D

#### Why-D A — matched restriction

Complete.

#### Why-D B — decomposition

Complete.

#### Why-D C — molecular context

Complete.

#### Why-D D — adjusted association

Complete.

### E1d — Clinical Improvements

Complete.

### E1v and cap robustness — encoder replication

Complete.

### E1e — MSI positive control

Complete.

### Worklist enrichment — fixed capacity

Complete.

### Decision-curve analysis

Complete.

### Extended-RAS, MAPK, and pathway-quiet relabeling

Complete.

## Aim 2 — transport

### Label-blind acquisition and composition evidence

Complete.

### TCGA+SurGen source-restricted target scoring

Complete.

#### Per-seed refit versus within-seed five-fold ensemble

Seed order is 42, 43, 44, 45, and 46. Values use patient-pooled AUROC where
specified. The summaries are arithmetic means and sample SD across seeds, not
confidence intervals.

##### UNI-v1

| Population | Refit AUROCs | Refit mean ± SD | Ensemble AUROCs | Ensemble mean ± SD |
|---|---|---|---|---|
| RIH-Pri + CPTAC | .7 | .7 ± .1 | .7 | .7 ± .1 |
| All-Met | .6 | .6 ± .1 | .6 | .6 ± .1 |

##### Virchow2-CLS

| Population | Refit AUROCs | Refit mean ± SD | Ensemble AUROCs | Ensemble mean ± SD |
|---|---|---|---|---|
| RIH-Pri + CPTAC | .7 | .7 ± .1 | .7 | .7 ± .1 |
| All-Met | .6 | .6 ± .1 | .6 | .6 ± .1 |

### E2a-F — family transport

Complete.

### E2a-D — sibling transport

Complete.

### Between-slide sampling — SR1482

Complete.

### E2-MET — metastatic transfer

Complete.

### E2c — sparse adaptation

Complete.

### E2d1–E2d6 — exploratory diagnostics

#### E2d1 — metastatic site

Complete.

#### E2d2 — peritoneal audit

Complete.

#### E2d3 — specimen role

Complete.

#### E2d4 — case mix

Complete.

#### E2d5 — paired specimens

Complete.

#### E2d6 — acquisition regimes

Complete.

### E2e — in-domain upper bound

Complete.

### E2f-v3 — residual adaptation

Complete.

### E2-CPHT — raw transfer

Complete.

### E2-CPHT-A — few-shot adaptation

Complete.

### E2-CPHT-R — unresolved robustness

NOT_RUN.

## Aim 3 — molecular-resolution ceiling

### E3 fixed five-seed ladder

Complete.

### E3 repeated controls

Complete.

### E3v replication

Complete.

## Aim 4 — morphology

### E4 canonical atlas

Complete.

### E4 primary-to-metastatic transport

Complete.

### E4 vocabulary-size stability

Complete.

### E4 score compressibility

Complete.

### E4 adopted montage naming

Complete.

### E4 pathway-context weld

Complete.

### Whole-section pathology state

GENERATED_UNREAD.

## Paper-building candidate map

Selectable experiments are mapped to manuscript roles.

## Cross-aim synthesis and claim boundaries

Complete.

## Excluded, superseded, and unavailable-result ledger

- Superseded headline fields are excluded.
- The audit-only campaign is not selectable.
- Missing arms are NOT_RUN.
- Whole-section pathology is GENERATED_UNREAD.

## Governed source index

The Audit document contains the exact source table.
"""


def _audit(sources: list[dict[str, Any]]) -> str:
    rows = "\n".join(f"| `{source['id']}` | `{source['sha256']}` |" for source in sources)
    return (
        _side_document("evidence audit and selection record")
        + f"""

## Governed source index

| Source ID | SHA-256 |
|---|---|
{rows}
"""
    )


def _fixture(tmp_path: Path, *, source_count: int = 4) -> verifier.BundlePaths:
    repo = tmp_path / "repo"
    parent_dir = repo / "reports/final_v12"
    final_dir = repo / "reports/final_v12_1"
    parent_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)

    sources: list[dict[str, Any]] = []
    for index in range(source_count):
        path = repo / f"evidence/source-{index:03d}.json"
        _write_json(path, {"index": index, "payload": f"source-{index}"})
        sources.append(
            {
                "id": f"fixture-source-{index:03d}",
                "aims": ["Shared"],
                "experiments": ["fixture"],
                "role": "fixture_source",
                **_identity(path, str(path.relative_to(repo))),
            }
        )

    parent_manifest = {
        "schema_version": 2,
        "bundle": "final_v12",
        "status": verifier.PARENT_MANIFEST_STATUS,
        "artifacts": sources,
        "pending_artifacts": [],
    }
    parent_manifest_path = parent_dir / verifier.SOURCE_MANIFEST_NAME
    _write_json(parent_manifest_path, parent_manifest)

    parent_documents = {
        "Experimental_Setup.md": "# FINAL-v12 setup\n\nSealed.\n",
        "Results.md": "# FINAL-v12 results\n\nSealed.\n",
        "Audit.md": "# FINAL-v12 audit\n\nSealed.\n",
    }
    for name, text in parent_documents.items():
        (parent_dir / name).write_text(text, encoding="utf-8")

    receipt = {
        "schema_version": 1,
        "bundle": "reports/final_v12",
        "status": verifier.PARENT_SEALED_STATUS,
        "checks": {"fixture_parent": "PASS"},
        "source_manifest": _identity(
            parent_manifest_path,
            str(parent_manifest_path.relative_to(repo)),
        ),
        "documents": {
            name: _identity(parent_dir / name, str((parent_dir / name).relative_to(repo)))
            for name in verifier.REPORT_DOCUMENTS
        },
        "authoritative_sources": sources,
    }
    parent_receipt_path = parent_dir / verifier.FINAL_RECEIPT_NAME
    _write_json(parent_receipt_path, receipt)

    promoted_path = repo / "evidence/promoted-fold5-results.json"
    _write_json(promoted_path, {"status": "complete", "rows": []})
    promoted = {
        "id": "promoted-fold5-results",
        "aims": ["Aim 2"],
        "experiments": ["refit versus five-fold ensemble"],
        "role": "promoted_results",
        **_identity(promoted_path, str(promoted_path.relative_to(repo))),
    }
    final_sources = sorted([*sources, promoted], key=lambda record: record["id"])
    final_manifest = {
        **parent_manifest,
        "bundle": "final_v12_1",
        "status": verifier.FINAL_MANIFEST_STATUS,
        "artifacts": final_sources,
    }
    _write_json(final_dir / verifier.SOURCE_MANIFEST_NAME, final_manifest)
    (final_dir / "Experimental_Setup.md").write_text(
        _side_document("integrated experimental setup"), encoding="utf-8"
    )
    (final_dir / "Results.md").write_text(_valid_results(), encoding="utf-8")
    (final_dir / "Audit.md").write_text(_audit(final_sources), encoding="utf-8")

    parent_verifier = repo / "tools/final_v12_bundle_receipt.py"
    parent_test = repo / "tests/test_final_v12_bundle_receipt.py"
    current_verifier = repo / "tools/final_v12_1_bundle_receipt.py"
    current_test = repo / "tests/test_final_v12_1_bundle_receipt.py"
    parent_verifier.parent.mkdir(parents=True)
    parent_test.parent.mkdir(parents=True)
    parent_verifier.write_text("# fixture parent verifier\n", encoding="utf-8")
    parent_test.write_text("# fixture parent verifier test\n", encoding="utf-8")
    current_verifier.write_text("# fixture current verifier\n", encoding="utf-8")
    current_test.write_text("# fixture current verifier test\n", encoding="utf-8")

    return verifier.BundlePaths(
        repo=repo,
        final_v12_1=final_dir,
        destination=final_dir / verifier.FINAL_RECEIPT_NAME,
        verifier_code=current_verifier,
        verifier_test=current_test,
        parent_dir=parent_dir,
        parent_receipt=parent_receipt_path,
        parent_manifest=parent_manifest_path,
        parent_verifier=parent_verifier,
        parent_test=parent_test,
        expected_parent_receipt_sha256=_sha(parent_receipt_path),
        expected_parent_manifest_sha256=_sha(parent_manifest_path),
        expected_parent_verifier_sha256=_sha(parent_verifier),
        expected_parent_test_sha256=_sha(parent_test),
        expected_parent_document_sha256={
            name: _sha(parent_dir / name) for name in verifier.REPORT_DOCUMENTS
        },
        expected_final_document_sha256={
            name: _sha(final_dir / name) for name in verifier.REPORT_DOCUMENTS
        },
        expected_extension_source_ids=("promoted-fold5-results",),
        expected_parent_source_count=source_count,
        replay_parent_verifier=False,
    )


def test_production_parent_pins_match_sealed_final_v12() -> None:
    paths = verifier.default_paths()
    assert verifier.sha256_file(paths.parent_receipt) == verifier.EXPECTED_PARENT_RECEIPT_SHA256
    assert verifier.sha256_file(paths.parent_manifest) == verifier.EXPECTED_PARENT_MANIFEST_SHA256
    assert {
        name: verifier.sha256_file(paths.parent_dir / name) for name in verifier.REPORT_DOCUMENTS
    } == verifier.EXPECTED_PARENT_DOCUMENT_SHA256
    assert verifier.EXPECTED_PARENT_SOURCE_COUNT == 139
    assert verifier.sha256_file(paths.parent_verifier) == verifier.EXPECTED_PARENT_VERIFIER_SHA256
    assert verifier.sha256_file(paths.parent_test) == verifier.EXPECTED_PARENT_TEST_SHA256


def test_build_receipt_is_deterministic_and_read_only(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = verifier.build_receipt(paths)
    second = verifier.build_receipt(paths)
    assert first == second
    assert verifier._receipt_bytes(first) == verifier._receipt_bytes(second)
    assert first["source_count"] == 5
    assert first["extension_source_count"] == 1
    assert set(first["authoritative_sources"][0]) == verifier._SOURCE_KEYS
    assert first["verification"] == {
        "verifier": _identity(
            paths.verifier_code,
            str(paths.verifier_code.relative_to(paths.repo)),
        ),
        "tests": _identity(
            paths.verifier_test,
            str(paths.verifier_test.relative_to(paths.repo)),
        ),
    }
    assert all(value == "PASS" for value in first["checks"].values())
    assert "created_utc" not in first
    assert not paths.destination.exists()


def test_parent_receipt_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths.parent_receipt.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    paths = replace(paths, expected_parent_receipt_sha256=_sha(paths.parent_receipt))
    with pytest.raises(verifier.BundleVerificationError, match="duplicate key"):
        verifier.build_receipt(paths)


def test_parent_manifest_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths.parent_manifest.write_text('{"schema_version":2,"schema_version":2}\n', encoding="utf-8")
    receipt = json.loads(paths.parent_receipt.read_text())
    receipt["source_manifest"] = _identity(
        paths.parent_manifest,
        str(paths.parent_manifest.relative_to(paths.repo)),
    )
    _write_json(paths.parent_receipt, receipt)
    paths = replace(
        paths,
        expected_parent_receipt_sha256=_sha(paths.parent_receipt),
        expected_parent_manifest_sha256=_sha(paths.parent_manifest),
    )
    with pytest.raises(verifier.BundleVerificationError, match="duplicate key"):
        verifier.build_receipt(paths)


def test_final_manifest_strict_json_rejects_nonfinite_constants(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = paths.final_v12_1 / verifier.SOURCE_MANIFEST_NAME
    manifest.write_text('{"schema_version":NaN}\n', encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="non-finite"):
        verifier.build_receipt(paths)


def test_parent_status_must_be_sealed_even_when_receipt_hash_is_updated(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt = json.loads(paths.parent_receipt.read_text())
    receipt["status"] = "candidate_ready"
    _write_json(paths.parent_receipt, receipt)
    paths = replace(paths, expected_parent_receipt_sha256=_sha(paths.parent_receipt))
    with pytest.raises(verifier.BundleVerificationError, match="status is not authoritative"):
        verifier.build_receipt(paths)


def test_parent_document_drift_fails_receipt_identity_replay(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths.parent_dir / "Results.md").write_text("drift\n", encoding="utf-8")
    with pytest.raises(
        verifier.BundleVerificationError, match="FINAL-v12 Results.md identity drift"
    ):
        verifier.build_receipt(paths)


def test_inherited_manifest_record_must_be_exact(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest_path = paths.final_v12_1 / verifier.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][1]["role"] = "mutated_role"
    _write_json(manifest_path, manifest)
    with pytest.raises(verifier.BundleVerificationError, match="inherited source record drift"):
        verifier.build_receipt(paths)


def test_all_sources_are_directly_rehashed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    source = paths.repo / "evidence/source-002.json"
    source.write_text("drift\n", encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="source identity drift"):
        verifier.build_receipt(paths)


def test_source_symlinks_are_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    source = paths.repo / "evidence/source-001.json"
    real = source.with_suffix(".real")
    source.rename(real)
    source.symlink_to(real)
    with pytest.raises(verifier.BundleVerificationError, match="symlink"):
        verifier.build_receipt(paths)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            "#### Design B: TCGA+SurGen source-restricted OOF training",
            "#### Source model",
            "E0 Design B",
        ),
        (
            "#### TCGA+SurGen two-encoder challenge panel",
            "#### Challenge panel",
            "E1a TCGA\\+SurGen population panel",
        ),
        ("| **SurGen** | SR386 and SR1482 |", "", "cohort-taxonomy row SurGen"),
        ("#### E2d4 — case mix", "#### case mix", "E2d4"),
        (
            "## Integrated experiment and population availability matrix",
            "## Availability",
            "paper-selection availability matrix",
        ),
        (
            "## Excluded, superseded, and unavailable-result ledger",
            "## Notes",
            "exclusion/supersession ledger",
        ),
    ),
)
def test_results_topology_fails_closed(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    paths = _fixture(tmp_path)
    results = paths.final_v12_1 / "Results.md"
    text = results.read_text()
    assert old in text
    results.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match=message):
        verifier.build_receipt(paths)


def test_addendum_form_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    results = paths.final_v12_1 / "Results.md"
    text = results.read_text()
    results.write_text(text.replace("## Integrated conclusions", "## Results addendum", 1))
    with pytest.raises(verifier.BundleVerificationError, match="addendum-form"):
        verifier.build_receipt(paths)


@pytest.mark.parametrize(
    "stale",
    (
        "The candidate contains a machine-checked ledger.",
        "The flat candidate manifest authenticates the ledger.",
        "No FINAL-v12 receipt has been published.",
        "FINAL-v12 is awaiting publication.",
    ),
)
def test_stale_candidate_and_parent_publication_wording_is_rejected(
    tmp_path: Path,
    stale: str,
) -> None:
    paths = _fixture(tmp_path)
    results = paths.final_v12_1 / "Results.md"
    results.write_text(results.read_text() + f"\n{stale}\n", encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="stale"):
        verifier.build_receipt(paths)


def test_audit_source_index_must_be_exact(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    audit = paths.final_v12_1 / "Audit.md"
    text = audit.read_text()
    source_id = "fixture-source-002"
    audit.write_text("\n".join(line for line in text.splitlines() if source_id not in line) + "\n")
    with pytest.raises(verifier.BundleVerificationError, match="source index"):
        verifier.build_receipt(paths)


def _governed_binding_fixture(
    paths: verifier.BundlePaths,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    claim_ids = [f"claim.{index:03d}" for index in range(78)]
    contrast_ids = [f"contrast.{index:03d}" for index in range(48)]
    why_ids = [f"why.{index:03d}" for index in range(2)]
    governed_path = paths.repo / "evidence/governed-results.json"
    _write_json(
        governed_path,
        {
            "report_claim_rows": [{"row_id": value} for value in claim_ids],
            "report_contrast_rows": [{"row_id": value} for value in contrast_ids],
            "why_d_evidence_records": [{"record_id": value} for value in why_ids],
        },
    )
    sources = [
        {
            "id": verifier.GOVERNED_RESULTS_SOURCE_ID,
            "path": str(governed_path.relative_to(paths.repo)),
        }
    ]
    record_ids = claim_ids + contrast_ids + why_ids
    return sources, "\n".join(record_ids), record_ids


def test_governed_78_48_2_record_ids_are_exactly_once_bound(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    sources, results, _ = _governed_binding_fixture(paths)
    verifier._validate_governed_record_bindings(paths, sources, results)


def test_governed_record_binding_rejects_missing_or_duplicate_id(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    sources, results, record_ids = _governed_binding_fixture(paths)
    bad_results = results.replace(record_ids[0], record_ids[1], 1)
    with pytest.raises(verifier.BundleVerificationError, match="not exact-once bound"):
        verifier._validate_governed_record_bindings(paths, sources, bad_results)


def test_seal_is_atomic_exactly_once_verifiable_and_rejects_tool_drift(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    expected = verifier.build_receipt(paths)
    published = verifier.seal(paths)
    assert published == expected
    assert paths.destination.read_bytes() == verifier._receipt_bytes(expected)
    assert verifier.verify_published_receipt(paths) == expected

    for artifact in (paths.verifier_code, paths.verifier_test):
        original_artifact = artifact.read_bytes()
        try:
            artifact.write_bytes(original_artifact + b"# adversarial post-seal drift\n")
            with pytest.raises(verifier.BundleVerificationError, match="byte identity drift"):
                verifier.verify_published_receipt(paths)
        finally:
            artifact.write_bytes(original_artifact)
        assert verifier.verify_published_receipt(paths) == expected

    original = paths.destination.read_bytes()
    with pytest.raises(verifier.BundleVerificationError, match="overwrite"):
        verifier.seal(paths)
    assert paths.destination.read_bytes() == original


def test_existing_receipt_symlink_is_never_overwritten(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    decoy = paths.final_v12_1 / "decoy.json"
    decoy.write_text("do not touch\n", encoding="utf-8")
    paths.destination.symlink_to(decoy)
    with pytest.raises(verifier.BundleVerificationError, match="overwrite"):
        verifier.seal(paths)
    assert decoy.read_text() == "do not touch\n"


def test_cli_requires_explicit_seal_action_to_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(verifier, "default_paths", lambda: paths)
    assert verifier.main([]) == 0
    assert not paths.destination.exists()
    assert json.loads(capsys.readouterr().out)["status"] == "READY_TO_SEAL"
    assert verifier.main(["--check"]) == 0
    assert not paths.destination.exists()
    capsys.readouterr()
    assert verifier.main(["--seal"]) == 0
    assert paths.destination.is_file()
    assert json.loads(capsys.readouterr().out)["status"] == verifier.FINAL_SEALED_STATUS


def test_published_receipt_requires_canonical_deterministic_bytes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    published = verifier.seal(paths)
    paths.destination.write_text(json.dumps(published), encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="byte identity drift"):
        verifier.verify_published_receipt(paths)
