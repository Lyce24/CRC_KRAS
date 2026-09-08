from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import final_v10_5_bundle_receipt as verifier  # noqa: E402


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


def _candidate_suffix(name: str, records: list[dict[str, Any]] | None = None) -> str:
    completion = verifier.FINAL_STATE_STATUS_PARAGRAPH
    if name == "Experimental_Setup.md":
        body = """
## Additive FINAL-v10.5 Aim 1 within-source primary-cohort OOF extension

The governed execution used 20 chains and exactly 100 OOF fits with zero refits.
This is not an external-transport experiment and is not a confirmatory gate.

| Arm | Inclusion rule | Slides | Patients | KRAS-mutant | Wild type |
|---|---|---:|---:|---:|---:|
| tcga_primary | TCGA conventional-primary only | 508 | 502 | 207 | 295 |
| sr386_primary | SurGen SR386 primary only | 413 | 413 | 147 | 266 |
| surgen_primary | SurGen SR386 + SR1482 primary | 881 | 737 | 294 | 443 |
| tcga_surgen_primary | TCGA + SurGen primary | 1,389 | 1,239 | 501 | 738 |
"""
    elif name == "Results.md":
        results = _analysis_results()
        arm_rows = []
        for arm in verifier.EXPECTED_ARMS:
            record = results["arm_performance"][arm]
            population = record["population"]
            metrics = record["five_seed_mean_native_logit"]
            arm_rows.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    arm,
                    population["patients"],
                    population["mutant"],
                    population["wild_type"],
                    verifier._metric_cell(
                        metrics["auroc"], metrics["auroc_ci95"], context="fixture"
                    ),
                    verifier._metric_cell(
                        metrics["auprc"], metrics["auprc_ci95"], context="fixture"
                    ),
                )
            )
        contrast_rows = []
        for key, record in results["paired_common_patient_contrasts"].items():
            contrast_rows.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    key,
                    record["evaluation_population"],
                    record["patients"],
                    record["mutant"],
                    record["wild_type"],
                    verifier._metric_cell(
                        record["auroc"]["delta_larger_minus_smaller"],
                        record["auroc"]["delta_ci95"],
                        context="fixture",
                        signed=True,
                    ),
                    verifier._metric_cell(
                        record["auprc"]["delta_larger_minus_smaller"],
                        record["auprc"]["delta_ci95"],
                        context="fixture",
                        signed=True,
                    ),
                )
            )
        body = """
## Additive FINAL-v10.5 Aim 1 within-source primary-cohort OOF extension

The extension is descriptive within source and neither an external-transport estimand
nor a change to canonical FINAL-v10 E0.

### Governed FINAL-v10.5 within-source OOF arm results

| Arm | Patients | Mutant | Wild type | AUROC [95% CI] | AUPRC [95% CI] |
|---|---:|---:|---:|---:|---:|
{arm_rows}

### Governed FINAL-v10.5 paired common-patient contrasts

| Contrast | Population | Patients | Mutant | Wild type | AUROC delta [95% CI] | AUPRC delta [95% CI] |
|---|---|---:|---:|---:|---:|---:|
{contrast_rows}
""".format(arm_rows="\n".join(arm_rows), contrast_rows="\n".join(contrast_rows))
    else:
        identity_rows = "\n".join(
            f"| {record['id']} | {record['sha256']} |"
            for record in sorted(records or [], key=lambda item: str(item["id"]))
        )
        body = f"""
## FINAL-v10.5 additive audit state

The extension used 20 arm-seed chains, 100 logical fold fits, and zero refits.
The extension is descriptive within source.

### Governed FINAL-v10.5 extension source identities

| Source ID | SHA-256 |
|---|---|
{identity_rows}
"""
    return f"\n{body.strip()}\n\n{completion}\n"


def _draft_suffix(name: str) -> str:
    if name == "Experimental_Setup.md":
        return """

## Additive FINAL-v10.5 Aim 1 within-source primary-cohort OOF extension (draft)

The design uses 20 chains and exactly 100 planned logical fits. This is not an external-transport experiment
and is not a confirmatory gate. The 1,325-fit
census is a pending target only.
"""
    if name == "Results.md":
        return """

## Additive FINAL-v10.5 Aim 1 within-source primary-cohort OOF extension (pending)

The analysis is descriptive within source and neither an external-transport estimand
nor a replacement for canonical E0. The value 1,325 is only the completion target.
"""
    return """

## FINAL-v10.5 additive audit state (unsealed)

The plan has 20 arm-seed chains, 100 logical fold fits, and zero refits. The
extension is descriptive within source and remains pending.
"""


def _campaign_contract() -> dict[str, Any]:
    definitions = {}
    for arm, census in verifier.EXPECTED_ARM_CENSUS.items():
        definitions[arm] = {
            "expected_slides": census[0],
            "expected_patients": census[1],
            "expected_mutant_patients": census[2],
            "expected_wildtype_patients": census[3],
        }
    jobs = []
    for arm in verifier.EXPECTED_ARMS:
        for seed in verifier.EXPECTED_SEEDS:
            jobs.append(
                {
                    "job_id": f"{arm}.seed{seed}",
                    "arm": arm,
                    "seed": seed,
                    "fit_count": 5,
                    "refit_count": 0,
                    "training_command": [
                        f"splits.seed={seed}",
                        f"training.seed={seed}",
                        "training.skip_finalize=true",
                    ],
                }
            )
    return {
        "campaign": "aim1_primary_cohort_oof_5seed",
        "arms": list(verifier.EXPECTED_ARMS),
        "seeds": list(verifier.EXPECTED_SEEDS),
        "n_folds": 5,
        "job_count": 20,
        "logical_fit_count": 100,
        "refit_count": 0,
        "max_parallel_training_processes": 6,
        "training_policy": ("five OOF folds only; training.skip_finalize=true; no final/refit"),
        "material_recipe": {"skip_finalize": True},
        "arm_definitions": definitions,
        "jobs": jobs,
    }


def _analysis_results() -> dict[str, Any]:
    arm_performance = {}
    for index, arm in enumerate(verifier.EXPECTED_ARMS):
        census = verifier.EXPECTED_ARM_CENSUS[arm]
        auroc = 0.61 + 0.01 * index
        auprc = 0.51 + 0.01 * index
        arm_performance[arm] = {
            "population": {
                "patients": census[1],
                "mutant": census[2],
                "wild_type": census[3],
            },
            "per_seed_descriptive": {},
            "five_seed_mean_native_logit": {
                "auroc": auroc,
                "auroc_ci95": [auroc - 0.05, auroc + 0.05],
                "auprc": auprc,
                "auprc_ci95": [auprc - 0.05, auprc + 0.05],
            },
        }

    def contrast_metrics(delta: float) -> dict[str, Any]:
        return {
            "larger": 0.65,
            "smaller": 0.65 - delta,
            "delta_larger_minus_smaller": delta,
            "delta_ci95": [delta - 0.03, delta + 0.03],
        }

    return {
        "schema_version": 1,
        "status": "complete",
        "experiment": "aim1_primary_cohort_oof_5seed_analysis",
        "design_status": "additive_FINAL_v10_5_source_cohort_OOF_analysis",
        "score_contract": {
            "probability_roundtrip_used": False,
            "model_seeds": list(verifier.EXPECTED_SEEDS),
        },
        "inference": {
            "unit": "patient",
            "bootstrap_draws": 10_000,
            "bootstrap_seed": 20260824,
            "stratification": "subcohort_x_KRAS_label",
            "model_seeds_are_inference_units": False,
            "folds_are_inference_units": False,
            "paired_indices_shared": True,
            "bootstrap_array_count": 26,
        },
        "arm_performance": arm_performance,
        "paired_common_patient_contrasts": {
            "surgen_minus_sr386_on_sr386": {
                "larger_arm": "surgen_primary",
                "smaller_arm": "sr386_primary",
                "evaluation_population": "sr386_primary",
                "patients": 413,
                "mutant": 147,
                "wild_type": 266,
                "paired_indices_shared": True,
                "auroc": contrast_metrics(0.01),
                "auprc": contrast_metrics(0.02),
            },
            "tcga_surgen_minus_surgen_on_surgen": {
                "larger_arm": "tcga_surgen_primary",
                "smaller_arm": "surgen_primary",
                "evaluation_population": "surgen_primary",
                "patients": 737,
                "mutant": 294,
                "wild_type": 443,
                "paired_indices_shared": True,
                "auroc": contrast_metrics(0.02),
                "auprc": contrast_metrics(0.03),
            },
            "tcga_surgen_minus_tcga_on_tcga": {
                "larger_arm": "tcga_surgen_primary",
                "smaller_arm": "tcga_primary",
                "evaluation_population": "tcga_primary",
                "patients": 502,
                "mutant": 207,
                "wild_type": 295,
                "paired_indices_shared": True,
                "auroc": contrast_metrics(-0.01),
                "auprc": contrast_metrics(0.01),
            },
        },
        "cross_population_ranking": {
            "role": "descriptive_only",
            "no_cross_population_inference_or_transport_claim": True,
        },
        "scope_boundary": {
            "canonical_final_v10_e0_unchanged": True,
            "append_only_to_final_v10": True,
            "supersedes_parent_fields": [],
        },
        "inputs": {},
    }


def _fixture(tmp_path: Path, *, candidate: bool) -> verifier.BundlePaths:
    repo = tmp_path / "repo"
    base_dir = repo / "reports/final_v10"
    extension_dir = repo / "reports/final_v10_5"
    campaign_root = tmp_path / "campaign"
    base_dir.mkdir(parents=True)
    extension_dir.mkdir(parents=True)

    for name in verifier.REPORT_DOCUMENTS:
        base_text = f"# sealed base {name}\n"
        (base_dir / name).write_text(base_text, encoding="utf-8")
        suffix = _candidate_suffix(name) if candidate else _draft_suffix(name)
        (extension_dir / name).write_text(base_text + suffix, encoding="utf-8")

    base_verifier = repo / "tools/final_v10_bundle_receipt.py"
    base_test = repo / "tests/test_final_v10_bundle_receipt.py"
    campaign_controller = repo / "tools/aim1_source_cohort_five_seed_campaign.py"
    campaign_test = repo / "tests/test_aim1_source_cohort_five_seed_campaign.py"
    analysis_implementation = repo / "tools/aim1_source_cohort_five_seed_analysis.py"
    analysis_test = repo / "tests/test_aim1_source_cohort_five_seed_analysis.py"
    own_verifier = repo / "tools/final_v10_5_bundle_receipt.py"
    own_test = repo / "tests/test_final_v10_5_bundle_receipt.py"
    for path in (
        base_verifier,
        base_test,
        campaign_controller,
        campaign_test,
        analysis_implementation,
        analysis_test,
        own_verifier,
        own_test,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture {path.name}\n", encoding="utf-8")

    base_sources = []
    for index in range(2):
        path = repo / f"base/source_{index}.json"
        _write_json(path, {"index": index})
        base_sources.append(
            {
                "id": f"base-source-{index}",
                "aims": ["Aim 1"],
                "experiments": ["sealed base"],
                "role": "sealed_base_evidence",
                **_identity(path, f"base/source_{index}.json"),
            }
        )
    base_manifest_path = base_dir / "source_manifest.json"
    _write_json(
        base_manifest_path,
        {
            "schema_version": 2,
            "bundle": "final_v10",
            "status": verifier.parent.CANDIDATE_STATUS,
            "artifacts": base_sources,
            "pending_artifacts": [],
        },
    )
    base_receipt_path = base_dir / "report_bundle_receipt.json"
    base_receipt = {
        "schema_version": 1,
        "bundle": "reports/final_v10",
        "status": verifier.SEALED_STATUS,
        "source_manifest": _identity(base_manifest_path, "reports/final_v10/source_manifest.json"),
        "authoritative_sources": base_sources,
    }
    _write_json(base_receipt_path, base_receipt)

    contract_path = campaign_root / "contract.json"
    _write_json(contract_path, _campaign_contract())
    preflight_path = campaign_root / "receipts/deep_preflight.json"
    _write_json(
        preflight_path,
        {
            "status": "deep_preflight_passed",
            "contract": _identity(contract_path, str(contract_path)),
            "job_count": 20,
            "logical_fit_count": 100,
            "refit_count": 0,
            "scheduler_ceiling": 6,
        },
    )
    scheduler_path = campaign_root / "receipts/scheduler.json"
    events = []
    for index, job in enumerate(_campaign_contract()["jobs"]):
        started = float((index // 6) * 2)
        events.append(
            {
                "job_id": job["job_id"],
                "pid": 10_000 + index,
                "returncode": 0,
                "started_utc": "2026-08-24T19:00:00+00:00",
                "finished_utc": "2026-08-24T19:01:00+00:00",
                "started_monotonic": started,
                "finished_monotonic": started + 1.0,
            }
        )
    _write_json(
        scheduler_path,
        {
            "schema_version": 1,
            "status": "completed_rc0",
            "created_utc": "2026-08-24T19:01:00+00:00",
            "configured_max_workers": 6,
            "observed_peak_parallel_workers": 6,
            "job_count": 20,
            "logical_fit_count": 100,
            "refit_count": 0,
            "events": events,
        },
    )
    job_receipts = []
    for arm in verifier.EXPECTED_ARMS:
        for seed in verifier.EXPECTED_SEEDS:
            job_path = campaign_root / f"receipts/source_cv/{arm}/seed{seed}.json"
            _write_json(job_path, {"arm": arm, "seed": seed, "status": "completed"})
            job_receipts.append(_identity(job_path, str(job_path)))
    training_path = campaign_root / "receipts/training_complete.json"
    _write_json(
        training_path,
        {
            "schema_version": 1,
            "status": "complete_and_certified",
            "created_utc": "2026-08-24T19:02:00+00:00",
            "contract": _identity(contract_path, str(contract_path)),
            "preflight": _identity(preflight_path, str(preflight_path)),
            "scheduler": _identity(scheduler_path, str(scheduler_path)),
            "arms": list(verifier.EXPECTED_ARMS),
            "seeds": list(verifier.EXPECTED_SEEDS),
            "job_count": 20,
            "folds_per_job": 5,
            "logical_fit_count": 100,
            "refit_count": 0,
            "job_receipts": job_receipts,
        },
    )
    analysis_root = campaign_root / "analysis"
    _write_json(analysis_root / "contract.json", {"schema_version": 1})
    (analysis_root / "patient_native_logits.parquet").write_bytes(b"synthetic parquet")
    _write_json(analysis_root / "results.json", _analysis_results())
    (analysis_root / "bootstrap_distributions.npz").write_bytes(b"synthetic npz")
    completion_path = analysis_root / "analysis_completion_receipt.json"
    _write_json(completion_path, {"status": "complete_and_certified"})

    preliminary = verifier.BundlePaths(
        repo=repo,
        final_v10_5=extension_dir,
        destination=extension_dir / verifier.FINAL_RECEIPT_NAME,
        verifier_code=own_verifier,
        verifier_test=own_test,
        base_final_v10=base_dir,
        base_receipt=base_receipt_path,
        base_manifest=base_manifest_path,
        base_verifier=base_verifier,
        base_test=base_test,
        campaign_root=campaign_root,
        campaign_controller=campaign_controller,
        campaign_test=campaign_test,
        analysis_implementation=analysis_implementation,
        analysis_test=analysis_test,
        expected_dependency_sha256={},
        expected_base_source_count=2,
        base_validator=lambda _: base_receipt,
        campaign_validator=lambda _: {"status": "PASS"},
        analysis_validator=lambda _: _analysis_results(),
    )
    dependency_paths = {
        "base_verifier": base_verifier,
        "base_test": base_test,
        "base_manifest": base_manifest_path,
        "base_receipt": base_receipt_path,
        "campaign_controller": campaign_controller,
        "campaign_test": campaign_test,
        "analysis_implementation": analysis_implementation,
        "analysis_test": analysis_test,
    }
    paths = replace(
        preliminary,
        expected_dependency_sha256={key: _sha(path) for key, path in dependency_paths.items()},
        expected_document_sha256={
            name: _sha(extension_dir / name) for name in verifier.REPORT_DOCUMENTS
        },
        expected_terminal_receipt_sha256={
            "aim1-source-cohort-five-seed-training-completion": _sha(training_path),
            "aim1-source-cohort-five-seed-analysis-completion": _sha(completion_path),
        },
        expected_created_utc="2026-08-24T20:00:00+00:00",
    )

    specs = verifier._source_specs(paths)
    records = []
    for source_id, metadata in specs.items():
        source_path = verifier._lexical_path(str(metadata["path"]), paths)
        records.append(
            {
                "id": source_id,
                **metadata,
                **_identity(source_path, str(metadata["path"])),
            }
        )
    if candidate:
        for name in verifier.REPORT_DOCUMENTS:
            base_text = (base_dir / name).read_text(encoding="utf-8")
            (extension_dir / name).write_text(
                base_text + _candidate_suffix(name, records), encoding="utf-8"
            )
        paths = replace(
            paths,
            expected_document_sha256={
                name: _sha(extension_dir / name) for name in verifier.REPORT_DOCUMENTS
            },
        )
        manifest = {
            "schema_version": 2,
            "bundle": "final_v10_5",
            "status": verifier.CANDIDATE_STATUS,
            "artifacts": sorted([*base_sources, *records], key=lambda item: str(item["id"])),
            "pending_artifacts": [],
        }
    else:
        present_ids = {
            "aim1-source-cohort-five-seed-campaign-controller",
            "aim1-source-cohort-five-seed-campaign-test",
            "aim1-source-cohort-five-seed-campaign-contract",
            "aim1-source-cohort-five-seed-deep-preflight",
        }
        manifest = {
            "schema_version": 1,
            "bundle": "final_v10_5",
            "status": verifier.DRAFT_STATUS,
            "base_bundle_receipt": _identity(
                base_receipt_path, "reports/final_v10/report_bundle_receipt.json"
            ),
            "base_source_manifest": _identity(
                base_manifest_path, "reports/final_v10/source_manifest.json"
            ),
            "artifacts": [record for record in records if record["id"] in present_ids],
            "pending_artifacts": [
                {key: value for key, value in record.items() if key in verifier._PENDING_KEYS}
                for record in records
                if record["id"] not in present_ids
            ],
        }
    _write_json(extension_dir / verifier.SOURCE_MANIFEST_NAME, manifest)
    return paths


def test_draft_recursively_rehashes_parent_and_stays_unsealed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, candidate=False)
    status = verifier.draft_status(paths)
    assert status["parent_recursive_verification"] == "PASS"
    assert status["parent_direct_81_source_rehash"] == "PASS"
    assert status["materialized_extension_source_count"] == 4
    assert status["pending_extension_source_count"] == 9
    assert status["authenticated_fit_census"] == 1225
    assert status["pending_complete_fit_census"] == 1325
    assert not paths.destination.exists()
    with pytest.raises(verifier.BundleVerificationError, match="blocked by 9 pending"):
        verifier.build_receipt(paths)


def test_full_candidate_seals_verifies_and_is_exactly_once(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, candidate=True)
    built = verifier.build_receipt(paths)
    assert built["fit_census"]["complete"] == 1325
    assert len(built["authoritative_sources"]) == 15
    sealed = verifier.seal(paths)
    assert sealed == built
    assert verifier.verify_published_receipt(paths) == built
    with pytest.raises(verifier.BundleVerificationError, match="overwrite"):
        verifier.seal(paths)


def test_base_source_byte_drift_is_directly_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, candidate=False)
    source = paths.repo / "base/source_0.json"
    source.write_text('{"coordinated": "tamper"}\n', encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="source identity drift"):
        verifier.draft_status(paths)


def test_extension_metadata_relabel_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, candidate=True)
    manifest_path = paths.final_v10_5 / verifier.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    target = next(
        item
        for item in manifest["artifacts"]
        if item["id"] == "aim1-source-cohort-five-seed-results"
    )
    target["role"] = "confirmatory_external_transport_result"
    _write_json(manifest_path, manifest)
    with pytest.raises(verifier.BundleVerificationError, match="frozen metadata"):
        verifier.build_receipt(paths)


@pytest.mark.parametrize(
    "payload",
    (
        '{"status":"shadow","status":"complete"}\n',
        '{"value":NaN}\n',
    ),
)
def test_json_ambiguity_fails_closed(tmp_path: Path, payload: str) -> None:
    source = tmp_path / "ambiguous.json"
    source.write_text(payload, encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError):
        verifier._load_json(source, label="ambiguous fixture")


def test_candidate_manifest_duplicate_key_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, candidate=True)
    manifest = paths.final_v10_5 / verifier.SOURCE_MANIFEST_NAME
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace('{\n', '{\n  "status": "adversarial-shadow-value",\n', 1),
        encoding="utf-8",
    )
    with pytest.raises(verifier.BundleVerificationError, match="duplicate JSON key"):
        verifier.build_receipt(paths)


def test_terminal_receipt_coordinated_rehash_cannot_bypass_pin(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, candidate=True)
    training = paths.campaign_root / "receipts/training_complete.json"
    value = json.loads(training.read_text())
    value["extra_unreviewed_field"] = True
    _write_json(training, value)
    manifest_path = paths.final_v10_5 / verifier.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    record = next(
        item
        for item in manifest["artifacts"]
        if item["id"] == "aim1-source-cohort-five-seed-training-completion"
    )
    record.update(_identity(training, str(training)))
    _write_json(manifest_path, manifest)
    with pytest.raises(verifier.BundleVerificationError, match="terminal receipt pin drift"):
        verifier.build_receipt(paths)


def test_same_byte_declared_source_symlink_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, candidate=True)
    results = paths.campaign_root / "analysis/results.json"
    copy = paths.campaign_root / "analysis/results-copy.json"
    copy.write_bytes(results.read_bytes())
    results.unlink()
    results.symlink_to(copy)
    with pytest.raises(verifier.BundleVerificationError, match="symlink"):
        verifier.build_receipt(paths)


def test_candidate_rejects_appended_draft_state_claim(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, candidate=True)
    results = paths.final_v10_5 / "Results.md"
    results.write_text(results.read_text() + "\nThe analysis remains pending.\n", encoding="utf-8")
    paths = replace(
        paths,
        expected_document_sha256={
            **paths.expected_document_sha256,
            "Results.md": _sha(results),
        },
    )
    with pytest.raises(verifier.BundleVerificationError, match="draft-state marker"):
        verifier.build_receipt(paths)


def test_candidate_rejects_mislabeled_arm_metric_row(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, candidate=True)
    results = paths.final_v10_5 / "Results.md"
    text = results.read_text(encoding="utf-8")
    assert "| tcga_primary | 502 | 207 | 295 |" in text
    results.write_text(
        text.replace(
            "| tcga_primary | 502 | 207 | 295 |",
            "| tcga_primary | 503 | 207 | 295 |",
            1,
        ),
        encoding="utf-8",
    )
    paths = replace(
        paths,
        expected_document_sha256={
            **paths.expected_document_sha256,
            "Results.md": _sha(results),
        },
    )
    with pytest.raises(verifier.BundleVerificationError, match="Markdown row"):
        verifier.build_receipt(paths)


def test_candidate_rejects_swapped_source_sha_rows(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, candidate=True)
    manifest = json.loads(
        (paths.final_v10_5 / verifier.SOURCE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    sources = [
        item
        for item in manifest["artifacts"]
        if str(item["id"]).startswith("aim1-source-cohort-five-seed-")
    ]
    first, second = sources[:2]
    audit = paths.final_v10_5 / "Audit.md"
    text = audit.read_text(encoding="utf-8")
    first_row = f"| {first['id']} | {first['sha256']} |"
    second_row = f"| {second['id']} | {second['sha256']} |"
    assert first_row in text and second_row in text
    text = text.replace(first_row, "__FIRST__", 1).replace(second_row, "__SECOND__", 1)
    text = text.replace("__FIRST__", f"| {first['id']} | {second['sha256']} |", 1).replace(
        "__SECOND__", f"| {second['id']} | {first['sha256']} |", 1
    )
    audit.write_text(text, encoding="utf-8")
    paths = replace(
        paths,
        expected_document_sha256={
            **paths.expected_document_sha256,
            "Audit.md": _sha(audit),
        },
    )
    with pytest.raises(verifier.BundleVerificationError, match="Markdown row"):
        verifier.build_receipt(paths)


def test_candidate_rejects_coexisting_external_transport_claim(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, candidate=True)
    results = paths.final_v10_5 / "Results.md"
    results.write_text(
        results.read_text(encoding="utf-8") + "\nExternal transport claim: TRUE.\n",
        encoding="utf-8",
    )
    paths = replace(
        paths,
        expected_document_sha256={
            **paths.expected_document_sha256,
            "Results.md": _sha(results),
        },
    )
    with pytest.raises(verifier.BundleVerificationError, match="scope contradiction"):
        verifier.build_receipt(paths)


def test_post_link_verification_failure_cleans_public_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, candidate=True)

    def fail_verify(_: verifier.BundlePaths) -> dict[str, Any]:
        raise verifier.BundleVerificationError("synthetic post-link failure")

    monkeypatch.setattr(verifier, "verify_published_receipt", fail_verify)
    with pytest.raises(verifier.BundleVerificationError, match="post-link"):
        verifier.seal(paths)
    assert not paths.destination.exists()


def test_timestamp_only_mutation_is_not_accepted(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, candidate=True)
    verifier.seal(paths)
    receipt = json.loads(paths.destination.read_text())
    receipt["created_utc"] = "2026-08-24T20:00:01+00:00"
    paths.destination.write_bytes(verifier._receipt_bytes(receipt))
    with pytest.raises(verifier.BundleVerificationError, match="byte identity drift"):
        verifier.verify_published_receipt(paths)


def test_bootstrap_archive_roster_dtype_shape_and_paired_delta_are_enforced(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, candidate=True)
    archive_path = paths.campaign_root / "analysis/bootstrap_distributions.npz"
    names = {
        f"arm__{arm}__ensemble__{metric}"
        for arm in verifier.EXPECTED_ARMS
        for metric in ("auroc", "auprc")
    }
    contrast_keys = (
        "surgen_minus_sr386_on_sr386",
        "tcga_surgen_minus_surgen_on_surgen",
        "tcga_surgen_minus_tcga_on_tcga",
    )
    names |= {
        f"contrast__{key}__{component}__{metric}"
        for key in contrast_keys
        for component in ("larger", "smaller", "delta")
        for metric in ("auroc", "auprc")
    }
    arrays = {name: np.zeros(10_000, dtype=np.float64) for name in names}
    for key in contrast_keys:
        for metric in ("auroc", "auprc"):
            prefix = f"contrast__{key}"
            arrays[f"{prefix}__larger__{metric}"][:] = 0.6
            arrays[f"{prefix}__smaller__{metric}"][:] = 0.5
            arrays[f"{prefix}__delta__{metric}"] = (
                arrays[f"{prefix}__larger__{metric}"]
                - arrays[f"{prefix}__smaller__{metric}"]
            )
    np.savez(archive_path, **arrays)
    verifier._validate_bootstrap_archive(paths)
    arrays["contrast__surgen_minus_sr386_on_sr386__delta__auroc"][0] = 0.2
    np.savez(archive_path, **arrays)
    with pytest.raises(verifier.BundleVerificationError, match="delta identity drift"):
        verifier._validate_bootstrap_archive(paths)
