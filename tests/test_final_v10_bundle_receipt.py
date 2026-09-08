from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests import test_final_v9_bundle_receipt as v9_fixture
from tools import final_v10_bundle_receipt as bundle

_FIXTURE_CREATED_UTC = "2026-08-24T00:00:00+00:00"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": bundle.sha256_file(path),
    }


def _document_sha256_pins(paths: bundle.BundlePaths) -> dict[str, str]:
    return {
        filename: bundle.sha256_file(paths.final_v10 / filename)
        for filename in bundle.REPORT_DOCUMENTS
    }


def _terminal_receipt_sha256_pins(paths: bundle.BundlePaths) -> dict[str, str]:
    requirements = bundle._required_sources(paths)  # noqa: SLF001
    return {
        source_id: bundle.sha256_file(requirements[source_id][0])
        for source_id in bundle.EXPECTED_TERMINAL_VERIFICATION_RECEIPT_SHA256
    }


def _write_split_integrity(manifest_path: Path, splits_path: Path, output_path: Path) -> None:
    digest = hashlib.sha256()
    digest.update(manifest_path.read_bytes())
    digest.update(splits_path.read_bytes())
    _write_json(
        output_path,
        {"hash": digest.hexdigest(), "csv_path": str(manifest_path.resolve())},
    )


def _paths(tmp_path: Path) -> bundle.BundlePaths:
    final_v10 = tmp_path / "reports/final_v10"
    final_v10.mkdir(parents=True)
    return bundle.BundlePaths(
        repo=tmp_path,
        final_v10=final_v10,
        destination=final_v10 / bundle.FINAL_RECEIPT_NAME,
        verifier_code=tmp_path / "tools/final_v10_bundle_receipt.py",
        verifier_test=tmp_path / "tests/test_final_v10_bundle_receipt.py",
        base_manifest=tmp_path / "reports/final_v9/source_manifest.json",
        campaign_root=tmp_path / "campaign",
        adjudication_root=tmp_path / "adjudication",
        expected_created_utc=_FIXTURE_CREATED_UTC,
    )


def _pending(source_id: str, path: Path) -> dict[str, object]:
    return {
        "id": source_id,
        "aims": ["Shared"],
        "experiments": ["fixture"],
        "role": "fixture",
        "path": str(path),
    }


def _artifact(source_id: str, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source_id, encoding="utf-8")
    return {
        **_pending(source_id, path),
        "size_bytes": path.stat().st_size,
        "sha256": bundle.sha256_file(path),
    }


def test_production_draft_replays_training_and_has_no_receipt() -> None:
    status = bundle.draft_status()
    assert status["training_stage_replay"] == "PASS"
    assert status["adopted_source_count"] == 52
    assert status["materialized_five_seed_source_count"] + status["pending_source_count"] == 29
    assert status["published_receipt_present"] is False
    assert not bundle.default_paths().destination.exists()


def test_exact_source_rosters_and_precedence_are_frozen() -> None:
    assert len(bundle._EXPECTED_BASE_SOURCE_IDS) == 52  # noqa: SLF001
    assert len(bundle._required_sources(bundle.default_paths())) == 29  # noqa: SLF001
    assert set(bundle.core.SUPERSEDED_BY_FIVE_SEED_SOURCE_IDS).issubset(  # noqa: SLF001
        bundle._EXPECTED_ROLE_OVERRIDES  # noqa: SLF001
    )
    assert bundle._EXPECTED_ROLE_OVERRIDES["aim3-actionability-results"].startswith(  # noqa: SLF001
        "legacy_three_seed"
    )
    assert len(bundle._EXPECTED_NEW_SOURCE_METADATA) == 29  # noqa: SLF001
    orion_experiments, orion_role = bundle._EXPECTED_NEW_SOURCE_METADATA[  # noqa: SLF001
        "aim2-loco-five-seed-orion-patients"
    ]
    assert orion_experiments == ("Orion LOCO sensitivity",)
    assert orion_role == "governed_five_seed_orion_loco_sensitivity_scores"
    primary_experiments, _ = bundle._EXPECTED_NEW_SOURCE_METADATA[  # noqa: SLF001
        "aim2-loco-five-seed-primary-patients"
    ]
    assert "RIH size-matched sensitivity" in primary_experiments


def test_production_candidate_builds_without_publishing_receipt() -> None:
    destination = bundle.default_paths().destination
    receipt = bundle.build_receipt()
    assert receipt["status"] == "SEALED_COMPLETED_RESULTS_WITH_DECLARED_NOT_RUN_ARM"
    assert receipt["checks"]["exact_flat_81_source_inventory"] == "PASS"
    assert not destination.exists()


def test_refresh_keeps_layered_schema_after_partial_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    ready = tmp_path / "ready.json"
    ready.write_text("ready", encoding="utf-8")
    missing = tmp_path / "missing.json"
    manifest = {
        "schema_version": 2,
        "bundle": "final_v10",
        "status": bundle.DRAFT_STATUS,
        "base_manifest": {"path": "base", "size_bytes": 1, "sha256": "a" * 64},
        "role_overrides": dict(bundle._EXPECTED_ROLE_OVERRIDES),  # noqa: SLF001
        "artifacts": [],
        "pending_artifacts": [
            _pending("ready-source", ready),
            _pending("missing-source", missing),
        ],
    }
    _write_json(paths.final_v10 / bundle.SOURCE_MANIFEST_NAME, manifest)

    def fake_validate(
        _paths_value: bundle.BundlePaths, *, require_candidate: bool
    ) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
        assert require_candidate is False
        current = json.loads(
            (paths.final_v10 / bundle.SOURCE_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        return current, [], list(current["pending_artifacts"])

    monkeypatch.setattr(bundle, "_validate_manifest", fake_validate)
    monkeypatch.setattr(bundle, "draft_status", lambda _paths_value: {"status": "checked"})
    assert bundle.refresh_manifest(paths) == {"status": "checked"}

    refreshed = json.loads(
        (paths.final_v10 / bundle.SOURCE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert set(refreshed) == bundle._DRAFT_MANIFEST_KEYS  # noqa: SLF001
    assert refreshed["status"] == bundle.DRAFT_STATUS
    assert [item["id"] for item in refreshed["artifacts"]] == ["ready-source"]
    assert refreshed["artifacts"][0]["sha256"] == bundle.sha256_file(ready)
    assert [item["id"] for item in refreshed["pending_artifacts"]] == ["missing-source"]


def test_refresh_atomically_flattens_exact_81_records_on_final_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    ready = tmp_path / "last.json"
    ready.write_text("last", encoding="utf-8")

    adopted = [
        {
            "id": source_id,
            "aims": ["Shared"],
            "experiments": ["fixture"],
            "role": "adopted_fixture",
            "path": str(tmp_path / f"adopted-{index}"),
            "size_bytes": index,
            "sha256": f"{index:064x}",
        }
        for index, source_id in enumerate(sorted(bundle._EXPECTED_BASE_SOURCE_IDS), start=1)  # noqa: SLF001
    ]
    existing_new = [
        {
            "id": f"new-{index}",
            "aims": ["Shared"],
            "experiments": ["fixture"],
            "role": "new_fixture",
            "path": str(tmp_path / f"new-{index}"),
            "size_bytes": index,
            "sha256": f"{index + 100:064x}",
        }
        for index in range(28)
    ]
    manifest = {
        "schema_version": 2,
        "bundle": "final_v10",
        "status": bundle.DRAFT_STATUS,
        "base_manifest": {"path": "base", "size_bytes": 1, "sha256": "a" * 64},
        "role_overrides": dict(bundle._EXPECTED_ROLE_OVERRIDES),  # noqa: SLF001
        "artifacts": existing_new,
        "pending_artifacts": [_pending("last-new", ready)],
    }
    _write_json(paths.final_v10 / bundle.SOURCE_MANIFEST_NAME, manifest)

    def fake_validate(
        _paths_value: bundle.BundlePaths, *, require_candidate: bool
    ) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
        assert require_candidate is False
        return manifest, [*adopted, *existing_new], list(manifest["pending_artifacts"])

    monkeypatch.setattr(bundle, "_validate_manifest", fake_validate)
    monkeypatch.setattr(bundle, "draft_status", lambda _paths_value: {"status": "flat"})
    monkeypatch.setattr(bundle, "_validate_documents", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(bundle, "_validate_complete_source_graph", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bundle.core, "_validate_five_seed_source_inventory", lambda *_args: None)
    assert bundle.refresh_manifest(paths) == {"status": "flat"}

    refreshed = json.loads(
        (paths.final_v10 / bundle.SOURCE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert set(refreshed) == bundle._FLAT_MANIFEST_KEYS  # noqa: SLF001
    assert refreshed["status"] == bundle.CANDIDATE_STATUS
    assert refreshed["pending_artifacts"] == []
    assert len(refreshed["artifacts"]) == 81
    assert "base_manifest" not in refreshed
    assert "role_overrides" not in refreshed


def test_layered_manifest_transitions_through_partial_state_to_flat_81(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("first", encoding="utf-8")
    adopted = [
        {
            "id": source_id,
            "aims": ["Shared"],
            "experiments": ["fixture"],
            "role": "adopted_fixture",
            "path": str(tmp_path / f"adopted-{index}"),
            "size_bytes": index,
            "sha256": f"{index:064x}",
        }
        for index, source_id in enumerate(sorted(bundle._EXPECTED_BASE_SOURCE_IDS), start=1)  # noqa: SLF001
    ]
    existing_new = [
        {
            "id": f"new-{index}",
            "aims": ["Shared"],
            "experiments": ["fixture"],
            "role": "new_fixture",
            "path": str(tmp_path / f"new-{index}"),
            "size_bytes": index,
            "sha256": f"{index + 100:064x}",
        }
        for index in range(27)
    ]
    manifest_path = paths.final_v10 / bundle.SOURCE_MANIFEST_NAME
    _write_json(
        manifest_path,
        {
            "schema_version": 2,
            "bundle": "final_v10",
            "status": bundle.DRAFT_STATUS,
            "base_manifest": {"path": "base", "size_bytes": 1, "sha256": "a" * 64},
            "role_overrides": dict(bundle._EXPECTED_ROLE_OVERRIDES),  # noqa: SLF001
            "artifacts": existing_new,
            "pending_artifacts": [
                _pending("first-new", first),
                _pending("second-new", second),
            ],
        },
    )

    def fake_validate(
        _paths_value: bundle.BundlePaths, *, require_candidate: bool
    ) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
        assert require_candidate is False
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        return current, [*adopted, *current["artifacts"]], list(current["pending_artifacts"])

    monkeypatch.setattr(bundle, "_validate_manifest", fake_validate)
    monkeypatch.setattr(bundle, "draft_status", lambda _paths_value: {"status": "checked"})
    monkeypatch.setattr(bundle, "_validate_documents", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(bundle, "_validate_complete_source_graph", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bundle.core, "_validate_five_seed_source_inventory", lambda *_args: None)

    bundle.refresh_manifest(paths)
    partial = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(partial) == bundle._DRAFT_MANIFEST_KEYS  # noqa: SLF001
    assert len(partial["artifacts"]) == 28
    assert [item["id"] for item in partial["pending_artifacts"]] == ["second-new"]

    second.write_text("second", encoding="utf-8")
    bundle.refresh_manifest(paths)
    flat = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(flat) == bundle._FLAT_MANIFEST_KEYS  # noqa: SLF001
    assert flat["status"] == bundle.CANDIDATE_STATUS
    assert flat["pending_artifacts"] == []
    assert len(flat["artifacts"]) == 81


def test_build_receipt_accepts_flat_manifest_shape_without_base_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    (paths.verifier_code.parent).mkdir(parents=True)
    (paths.verifier_test.parent).mkdir(parents=True)
    paths.verifier_code.write_text("verifier", encoding="utf-8")
    paths.verifier_test.write_text("tests", encoding="utf-8")
    _write_json(paths.base_manifest, {"fixture": "adopted ledger"})
    manifest_path = paths.final_v10 / bundle.SOURCE_MANIFEST_NAME
    flat = {
        "schema_version": 2,
        "bundle": "final_v10",
        "status": bundle.CANDIDATE_STATUS,
        "artifacts": [],
        "pending_artifacts": [],
    }
    _write_json(manifest_path, flat)
    sources = [
        {
            "id": f"source-{index}",
            "aims": ["Shared"],
            "experiments": ["fixture"],
            "role": "fixture",
            "path": str(tmp_path / f"source-{index}"),
            "size_bytes": 0,
            "sha256": "a" * 64,
        }
        for index in range(81)
    ]
    monkeypatch.setattr(bundle, "_validate_documents", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        bundle,
        "_validate_manifest",
        lambda *_args, **_kwargs: (flat, sources, []),
    )
    receipt = bundle.build_receipt(paths)
    assert len(receipt["authoritative_sources"]) == 81
    assert receipt["adopted_base_manifest"] == {
        **_identity(paths.base_manifest),
        "path": "reports/final_v9/source_manifest.json",
    }
    assert receipt["checks"]["exact_flat_81_source_inventory"] == "PASS"


def test_metric_claim_formatter_uses_report_precision() -> None:
    assert bundle._metric_ci_text(0.6528605, [0.6191418, 0.6833152]) == (  # noqa: SLF001
        "0.6529 [0.6191, 0.6833]"
    )


def test_claim_row_validation_rejects_correct_numbers_under_wrong_label() -> None:
    text = "| G12D vs other KRAS | 0.5000 | 0.6000 | +0.1000 | `CEILING` |\n"
    bundle._require_markdown_row(  # noqa: SLF001
        text,
        "G12D vs other KRAS",
        ("0.5000", "0.6000", "+0.1000", "`CEILING`"),
        label="fixture",
    )
    with pytest.raises(bundle.BundleVerificationError, match="does not map governed"):
        bundle._require_markdown_row(  # noqa: SLF001
            text,
            "G12 vs non-G12 KRAS",
            ("0.5000", "0.6000", "+0.1000", "`CEILING`"),
            label="fixture",
        )


@pytest.mark.parametrize(
    ("text", "row_label", "expected"),
    [
        ("| NOT CPTAC | 0.6100 |\n", "CPTAC", "0.6100"),
        ("| CPTAC | 0.61001 |\n", "CPTAC", "0.6100"),
    ],
)
def test_claim_row_validation_requires_exact_cells(
    text: str, row_label: str, expected: str
) -> None:
    with pytest.raises(bundle.BundleVerificationError, match="does not map governed"):
        bundle._require_markdown_row(  # noqa: SLF001
            text,
            row_label,
            (expected,),
            label="fixture exact-cell binding",
        )


def _metric(
    estimate: float,
    *,
    lower: float,
    upper: float,
    two_sided_half_width: float = 0.04,
) -> dict[str, object]:
    return {
        "estimate": estimate,
        "ci95_two_sided": [
            max(-1.0, estimate - two_sided_half_width),
            min(1.0, estimate + two_sided_half_width),
        ],
        "primary_fwer_one_sided": {
            "confidence": 0.99,
            "lower": lower,
            "upper": upper,
        },
    }


_CONTROL_TASKS = {
    "codon": "ctrl_codon",
    "g12d_broad": "ctrl_g12d_broad",
    "allele1": "ctrl_allele1",
    "allele2": "ctrl_allele2",
    "g12c": "ctrl_g12c",
}


def _rung(task: str, *, outcome: str, include_control_task: bool = True) -> dict[str, object]:
    if outcome == "CEILING":
        fine, control = 0.51, 0.66
        fine_bounds, control_bounds, delta_bounds = (0.43, 0.57), (0.56, 0.74), (0.05, 0.25)
    elif outcome == "INCONCLUSIVE":
        fine, control = 0.58, 0.65
        fine_bounds, control_bounds, delta_bounds = (0.49, 0.62), (0.55, 0.73), (-0.01, 0.15)
    elif outcome == "UNDERPOWERED":
        fine, control = 0.55, 0.54
        fine_bounds, control_bounds, delta_bounds = (0.47, 0.59), (0.48, 0.62), (-0.10, 0.08)
    else:  # pragma: no cover - fixture misuse
        raise AssertionError(outcome)
    delta = control - fine
    conditions = {
        "fine_upper_lt_0p60": fine_bounds[1] < 0.60,
        "control_lower_gt_0p50": control_bounds[0] > 0.50,
        "delta_lower_gt_zero": delta_bounds[0] > 0.0,
    }
    value: dict[str, object] = {
        "fine": _metric(fine, lower=fine_bounds[0], upper=fine_bounds[1]),
        "control": _metric(control, lower=control_bounds[0], upper=control_bounds[1]),
        "delta_control_minus_fine": _metric(
            delta,
            lower=delta_bounds[0],
            upper=delta_bounds[1],
            two_sided_half_width=0.05,
        ),
        "fine_per_seed": {str(seed): fine + (seed - 44) * 0.001 for seed in range(42, 47)},
        "control_per_seed": {str(seed): control + (seed - 44) * 0.001 for seed in range(42, 47)},
        "gate": {
            "conditions": conditions,
            "ceiling": outcome == "CEILING",
            "verdict": outcome,
        },
    }
    if include_control_task:
        value["control_task"] = _CONTROL_TASKS[task]
    return value


def _direction(point: float, lower: float) -> dict[str, object]:
    passed = lower > 0.5
    return {
        "auroc": point,
        "auroc_ci95": [lower, min(0.99, point + 0.12)],
        "directional_gate": {
            "rule": "patient-bootstrap AUROC lower 95% bound > 0.50",
            "lower_ci_above_0p5": passed,
            "passes": passed,
        },
    }


def _update_json(path: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    _write_json(path, value)


def _upgrade_v9_dag_for_v10(
    core_paths: Any,
    original_records: dict[str, dict[str, Any]],
) -> None:
    required = bundle.core._required_five_seed_sources(core_paths)  # noqa: SLF001

    def original_path(source_id: str) -> Path:
        raw = Path(str(original_records[source_id]["path"]))
        return raw.resolve() if raw.is_absolute() else (core_paths.repo / raw).resolve()

    aim1_result_path = required["aim1-e0-five-seed-results"][0]

    def upgrade_aim1(result: dict[str, Any]) -> None:
        domains = ("CPTAC", "RIH", "SR1482", "SR386", "TCGA")
        per_seed_domains = {
            str(seed): {
                domain: 0.58 + 0.01 * (seed - 42) + 0.02 * domain_index
                for domain_index, domain in enumerate(domains)
            }
            for seed in range(42, 47)
        }
        per_seed_macro = {
            seed: sum(values.values()) / len(values) for seed, values in per_seed_domains.items()
        }
        result.update(
            schema_version=1,
            score_scale="native_logit",
            population={
                "patients": 1486,
                "mutant": 604,
                "wild_type": 882,
                "domain_counts": {
                    "CPTAC": 94,
                    "RIH": 153,
                    "SR1482": 324,
                    "SR386": 413,
                    "TCGA": 502,
                },
            },
            per_seed_domain_auroc=per_seed_domains,
            per_seed_macro5=per_seed_macro,
            primary_median_seed_macro5={"point": 0.64, "ci95": [0.57, 0.71]},
            primitive_median_seed_auroc={
                domain: {
                    "point": per_seed_domains["44"][domain],
                    "ci95": [
                        per_seed_domains["44"][domain] - 0.08,
                        per_seed_domains["44"][domain] + 0.08,
                    ],
                }
                for domain in domains
            },
            five_seed_ensemble={
                "domain_auroc": {
                    domain: 0.605 + 0.02 * index for index, domain in enumerate(domains)
                },
                "macro5_auroc": 0.645,
                "macro5_ci95": [0.58, 0.71],
                "pooled_auroc": 0.655,
                "pooled_auprc": 0.585,
            },
            family_macro4_median_seed={"point": 0.646, "ci95": [0.576, 0.716]},
            patient_count_weighted_macro_median_seed={
                "point": 0.647,
                "ci95": [0.577, 0.717],
            },
        )

    _update_json(aim1_result_path, upgrade_aim1)

    aim1_contract_path = required["aim1-e0-five-seed-contract"][0]
    aim1_input_root = core_paths.repo / "fixture_inputs/aim1_splits"
    aim1_manifest_path = aim1_input_root / "canonical_manifest.csv"
    aim1_splits_path = aim1_input_root / "canonical_splits.parquet"
    aim1_integrity_path = aim1_input_root / "split_integrity.txt"
    aim1_summary_path = aim1_input_root / "split_summary.json"
    aim1_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    aim1_manifest_path.write_text(
        "patient_id,slide_id,fold\nfixture,fixture-slide,0\n", encoding="utf-8"
    )
    aim1_splits_path.write_bytes(b"fixture canonical Aim1 split membership")
    _write_split_integrity(aim1_manifest_path, aim1_splits_path, aim1_integrity_path)
    aim1_manifest_sha = bundle.sha256_file(aim1_manifest_path)
    aim1_integrity_sha = bundle.sha256_file(aim1_integrity_path)
    _write_json(
        aim1_summary_path,
        {
            "n_slides": 1642,
            "n_groups": 1486,
            "n_folds": 5,
            "seed": 42,
            "identity": {"payload": {"manifest_sha256": aim1_manifest_sha}},
        },
    )
    _update_json(
        aim1_contract_path,
        lambda contract: contract.update(
            inputs={
                "manifest": _identity(aim1_manifest_path),
                "splits": _identity(aim1_splits_path),
                "split_integrity": _identity(aim1_integrity_path),
                "split_summary": _identity(aim1_summary_path),
            }
        ),
    )

    aim1_training_path = required["aim1-e0-five-seed-training-validation"][0]

    def upgrade_aim1_training(training: dict[str, Any]) -> None:
        runs: dict[str, Any] = {}
        for seed in range(42, 47):
            identity_path = aim1_training_path.parent / "fixture_identities" / f"seed{seed}.json"
            _write_json(
                identity_path,
                {
                    "schema_version": 2,
                    "fingerprint": f"fixture-{seed}",
                    "payload": {
                        "input_evidence": {
                            "manifest_sha256": aim1_manifest_sha,
                            "split_integrity_sha256": aim1_integrity_sha,
                            "feature_inventory_sha256": "3" * 64,
                        },
                        "material_config": {
                            "splits": {"seed": 42},
                            "training": {"seed": seed},
                        },
                    },
                },
            )
            test_counts = (329, 328, 328, 328, 329)
            runs[str(seed)] = {
                "seed": seed,
                "run_dir": str(aim1_training_path.parent / f"seed{seed}"),
                "artifacts": {
                    "identity": _identity(identity_path),
                    "folds": {
                        str(fold): {
                            "n_test_slides": test_counts[fold],
                            "n_val_slides": 196 + (fold == 0),
                        }
                        for fold in range(5)
                    },
                },
            }
        training["runs"] = runs

    _update_json(aim1_training_path, upgrade_aim1_training)

    aim2_result_path = required["aim2-loco-five-seed-results"][0]

    paired_keys = (
        "sibling_sr386_minus_family_surgen_SR386",
        "sibling_sr1482_minus_family_surgen_SR1482",
        "sibling_tcga_coad_minus_family_tcga_TCGA-COAD",
        "sibling_tcga_read_minus_family_tcga_TCGA-READ",
        "family_rih_sm_minus_family_rih",
    )
    paired_values = {
        key: {
            "auroc_left": 0.60 + 0.005 * index,
            "auroc_right": 0.62 + 0.01 * index,
            "delta_auroc": (0.62 + 0.01 * index) - (0.60 + 0.005 * index),
            "ci_low": -0.02 + 0.002 * index,
            "ci_high": 0.09 + 0.005 * index,
            "n_patients": 90 + index,
        }
        for index, key in enumerate(paired_keys)
    }
    met_targets = {
        "rih_m": {
            "scorer": "family_rih",
            "n": 85,
            "n_mutant": 37,
            "auroc": 0.60,
            "auroc_ci95": [0.48, 0.72],
        },
        "sr1482_m": {
            "scorer": "family_surgen",
            "n": 74,
            "n_mutant": 30,
            "auroc": 0.56,
            "auroc_ci95": [0.43, 0.69],
        },
    }
    met_gate = {
        "equal_cohort_metastatic_macro_auroc": 0.58,
        "macro_auroc_ci95": [0.49, 0.67],
        "both_target_points_above_0p5": True,
        "macro_lower_bound_above_0p5": False,
        "claim_metastatic_transport": False,
        "gate": "both target AUROC points > 0.5 AND macro AUROC CI95 lower bound > 0.5",
    }

    def upgrade_aim2(result: dict[str, Any]) -> None:
        result["paired_sibling_and_size_matched_contrasts"] = paired_values
        result["e2met_confirmatory"] = met_gate
        result["e2met_confirmatory_family_naive"] = met_targets

    _update_json(aim2_result_path, upgrade_aim2)

    aim2_contract_path = required["aim2-loco-five-seed-contract"][0]
    aim2_contract = json.loads(aim2_contract_path.read_text(encoding="utf-8"))
    aim2_adoptions: list[dict[str, Any]] = []
    split_fixture_root = core_paths.repo / "fixture_inputs/aim2_splits"
    top_receipt_root = core_paths.campaign_root / "campaign/receipts/jobs"
    for arm in bundle.core.AIM2_ALL_ARMS:
        arm_root = split_fixture_root / arm
        source_manifest_path = arm_root / "source_manifest.csv"
        split_path = arm_root / "splits.parquet"
        integrity_path = arm_root / ".integrity_hash"
        source_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        source_manifest_path.write_text(f"patient_id,arm\nfixture,{arm}\n", encoding="utf-8")
        split_path.write_bytes(f"fixture split bytes for {arm}".encode())
        _write_split_integrity(source_manifest_path, split_path, integrity_path)
        source_record = _identity(source_manifest_path)
        split_record = _identity(split_path)
        aim2_contract["arms"][arm].update(
            source_sha256=source_record["sha256"],
            split_sha256=split_record["sha256"],
            inputs={
                "source_manifest": source_record,
                "split_file": split_record,
                "fold_manifest_reused_for_every_seed": True,
            },
        )
        for seed in range(42, 47):
            identity_path = arm_root / f"seed{seed}_training_identity.json"
            _write_json(
                identity_path,
                {
                    "schema_version": 2,
                    "payload": {
                        "input_evidence": {
                            "manifest_sha256": source_record["sha256"],
                            "split_integrity_sha256": bundle.sha256_file(integrity_path),
                        }
                    },
                },
            )
            if seed in bundle.ADOPTED_MODEL_SEEDS:
                aim2_adoptions.append(
                    {
                        "kind": "training",
                        "arm": arm,
                        "seed": seed,
                        "source_cv": {
                            "identity": _identity(identity_path),
                            "semantic_validation": {
                                "source_manifest": source_record,
                                "split_file": split_record,
                                "test_roster_exact_for_every_fold": True,
                                "validation_roster_exact_for_every_fold": True,
                                "oof_label_and_fold_exact_per_slide": True,
                            },
                        },
                    }
                )
                continue
            job_id = f"aim2.source_cv.{arm}.seed{seed}"
            component_path = arm_root / f"seed{seed}_component_receipt.json"
            _write_json(
                component_path,
                {
                    "status": "completed_rc0",
                    "job_key": job_id,
                    "artifacts": {
                        "arm": arm,
                        "seed": seed,
                        "identity": _identity(identity_path),
                    },
                },
            )
            top_path = top_receipt_root / f"{bundle.core._safe_campaign_job_name(job_id)}.json"  # noqa: SLF001
            _update_json(
                top_path,
                lambda top, component_path=component_path: top.update(
                    component_evidence={"component_receipt": _identity(component_path)}
                ),
            )
    aim2_contract["adoptions"] = aim2_adoptions
    _write_json(aim2_contract_path, aim2_contract)

    adjudication_path = required["aim2-e2a-five-seed-adjudication-result"][0]
    family_names = (
        "CPTAC",
        "RIH",
        "SR386_given_whole_SurGen_holdout",
        "SR1482_given_whole_SurGen_holdout",
        "TCGA_pooled_COAD_READ",
    )
    sibling_slugs = ("sr386", "sr1482", "tcga_coad", "tcga_read")
    input_paths = {
        "final_v8_preregistration": core_paths.repo
        / "fixture_inputs/final_v8_preregistration.json",
        "frozen_aim2_five_seed_extension": core_paths.repo / "fixture_inputs/frozen_aim2.py",
        "frozen_five_seed_campaign_controller": core_paths.repo / "fixture_inputs/controller.py",
        "frozen_final_v8_e2ad_topology": core_paths.repo / "fixture_inputs/e2ad.py",
        "campaign_contract": required["final-v9-five-seed-campaign-contract"][0],
        "campaign_final_receipt": required["final-v9-five-seed-campaign-results-completion"][0],
        "aim2_contract": required["aim2-loco-five-seed-contract"][0],
        "aim2_inference_seal": required["aim2-loco-five-seed-inference-seal"][0],
        "aim2_results": aim2_result_path,
        "aim2_report_receipt": required["aim2-loco-five-seed-report-receipt"][0],
        "aim2_primary_patient_table": required["aim2-loco-five-seed-primary-patients"][0],
        "aim2_metastatic_patient_table": required["aim2-loco-five-seed-metastatic-patients"][0],
        "raw_cpht_three_seed_result": original_path("aim2-e2cpht-results"),
    }
    for name, path in input_paths.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture input {name}\n", encoding="utf-8")

    def upgrade_adjudication(result: dict[str, Any]) -> None:
        result["schema_version"] = 1
        result["inputs"] = {name: _identity(path) for name, path in input_paths.items()}
        result["precedence"]["retained_source_paired_delta_replay"] = (
            "/e2a_d/source_report_paired_point_confirmation"
        )
        result["e2a_f"]["directions"] = {
            name: _direction(0.61 + 0.01 * index, 0.51 + 0.005 * index)
            for index, name in enumerate(family_names)
        }
        result["e2a_f"]["nested_four_family_macro"] = {
            "auroc": 0.651,
            "auroc_ci95": [0.571, 0.731],
            "role": "panel summary; cannot rescue a failed direction",
        }
        result["e2a_f"]["source_report_confirmation"] = {
            "source_report_macro_point_and_interval_unchanged": True,
            "source_report_direction_points_and_intervals_unchanged": True,
            "only_gate_interpretation_superseded": True,
        }
        result["e2a_d"]["targets"] = {
            slug: _direction(0.66 + 0.01 * index, 0.52 + 0.005 * index)
            for index, slug in enumerate(sibling_slugs)
        }
        result["e2a_d"]["adjudication"] = {
            "rule": (
                "all four sibling-stratum patient-bootstrap AUROC lower 95% bounds "
                "must exceed 0.50; neither macro can rescue a failed direction"
            ),
            "passed_directions": list(sibling_slugs),
            "required_directions": list(sibling_slugs),
            "all_directional_lower_bounds_above_0p5": True,
            "claim_sibling_stratum_transport": True,
        }
        result["e2a_d"]["macros"] = {
            "secondary_five_acquisition_domain": {
                "auroc": 0.661,
                "auroc_ci95": [0.581, 0.741],
                "role": "secondary panel summary; cannot rescue a failed direction",
            },
            "descriptive_equal_six_stratum": {
                "auroc": 0.662,
                "auroc_ci95": [0.582, 0.742],
                "role": "descriptive; gives TCGA and SurGen two votes each",
            },
        }
        paired: dict[str, Any] = {}
        confirmations: dict[str, Any] = {}
        for slug, source_key in zip(sibling_slugs, paired_keys[:4], strict=True):
            source_block = paired_values[source_key]
            paired[slug] = {
                "estimand": "AUROC(sibling retained) - AUROC(whole family held out)",
                "n_paired_patients": source_block["n_patients"],
                "sibling_retained_auroc": source_block["auroc_right"],
                "whole_family_held_out_auroc": source_block["auroc_left"],
                "delta_auroc": source_block["delta_auroc"],
                "delta_auroc_ci95": [source_block["ci_low"], source_block["ci_high"]],
                "bootstrap": "identical indices shared",
            }
            confirmations[source_key] = {
                "delta_auroc": source_block["delta_auroc"],
                "auroc_left": source_block["auroc_left"],
                "auroc_right": source_block["auroc_right"],
                "n_patients": source_block["n_patients"],
                "matches": True,
            }
        result["e2a_d"]["paired_sibling_minus_family"] = paired
        size_source = paired_values[paired_keys[4]]
        result["e2a_d"]["size_matched_rih_sensitivity"] = {
            "estimand": "AUROC(size-matched RIH holdout) - AUROC(full-source RIH holdout)",
            "n_paired_patients": size_source["n_patients"],
            "size_matched_auroc": size_source["auroc_right"],
            "full_source_auroc": size_source["auroc_left"],
            "delta_auroc": size_source["delta_auroc"],
            "delta_auroc_ci95": [size_source["ci_low"], size_source["ci_high"]],
            "bootstrap": "identical indices shared",
            "role": "secondary sensitivity; not an E2a-D directional gate",
        }
        confirmations[paired_keys[4]] = {
            "delta_auroc": size_source["delta_auroc"],
            "auroc_left": size_source["auroc_left"],
            "auroc_right": size_source["auroc_right"],
            "n_patients": size_source["n_patients"],
            "matches": True,
        }
        result["e2a_d"]["source_report_paired_point_confirmation"] = {
            "all_five_retained_delta_points_recomputed_and_match": True,
            "comparisons": confirmations,
        }
        result["e2met_gate_confirmation"] = {
            "targets": met_targets,
            **met_gate,
            "source_report_pointer": "/e2met_confirmatory",
            "matches_and_remains_authoritative": True,
        }

    _update_json(adjudication_path, upgrade_adjudication)

    aim3_path = required["aim3-ladders-five-seed-results"][0]

    def upgrade_aim3(result: dict[str, Any]) -> None:
        outcomes = {
            "codon": "CEILING",
            "g12d_broad": "CEILING",
            "allele1": "CEILING",
            "allele2": "INCONCLUSIVE",
            "g12c": "UNDERPOWERED",
        }
        result["fixed_univ1"] = {
            "rungs": {task: _rung(task, outcome=outcome) for task, outcome in outcomes.items()}
        }
        repeated: dict[str, Any] = {}
        for task, outcome in outcomes.items():
            draw_outcomes = [outcome, outcome, outcome]
            if task == "allele2":
                draw_outcomes = ["CEILING", "INCONCLUSIVE", "INCONCLUSIVE"]
            repeated[task] = {
                "control_task": _CONTROL_TASKS[task],
                "fine_per_seed": {str(seed): 0.52 + (seed - 44) * 0.001 for seed in range(42, 47)},
                "draws": {
                    str(draw_seed): _rung(task, outcome=draw_outcome, include_control_task=False)
                    for draw_seed, draw_outcome in zip(
                        (20260823, 20260824, 20260825), draw_outcomes, strict=True
                    )
                },
                "consensus_verdict": (
                    "CONSENSUS_CEILING"
                    if all(value == "CEILING" for value in draw_outcomes)
                    else "NO_CEILING_CONSENSUS"
                ),
            }
        result["repeated_univ1"] = {"rungs": repeated}
        e3v_rungs: dict[str, Any] = {}
        for index, task in enumerate(("codon", "g12d_broad", "allele1")):
            fine = 0.48 + 0.005 * index
            control = 0.67 + 0.005 * index
            delta = control - fine
            e3v_rungs[task] = {
                "control_task": _CONTROL_TASKS[task],
                "fine": _metric(fine, lower=0.40, upper=0.56 + 0.005 * index),
                "control": _metric(control, lower=0.57 + 0.005 * index, upper=0.77),
                "delta_control_minus_fine": _metric(delta, lower=0.08 + 0.005 * index, upper=0.29),
                "fine_per_seed": {str(seed): fine + (seed - 44) * 0.001 for seed in range(42, 47)},
                "control_per_seed": {
                    str(seed): control + (seed - 44) * 0.001 for seed in range(42, 47)
                },
                "gate": {
                    "conditions": {
                        "fine_upper_lt_0p60": True,
                        "control_lower_gt_0p50": True,
                        "delta_lower_gt_zero": True,
                    },
                    "ceiling": True,
                    "verdict": "CEILING",
                },
            }
        result["e3v_virchow2_cls"] = {"rungs": e3v_rungs}
        result["e1v_virchow2_cls_gene_reference"] = {
            "five_seed_ensemble_A": {
                "n": 1486,
                "n_positive": 604,
                "auroc": 0.681,
                "ci_low": 0.641,
                "ci_high": 0.721,
            },
            "per_seed_auroc_A": {str(seed): 0.67 + (seed - 42) * 0.005 for seed in range(42, 47)},
            "five_seed_ensemble_D_auroc": 0.719,
        }

    _update_json(aim3_path, upgrade_aim3)

    aim3_contract_path = required["aim3-ladders-five-seed-contract"][0]
    aim3_audit_path = required["aim3-ladders-five-seed-analysis-audit"][0]
    aim3_contract = json.loads(aim3_contract_path.read_text(encoding="utf-8"))
    aim3_audit = json.loads(aim3_audit_path.read_text(encoding="utf-8"))
    group_artifacts: dict[tuple[str, str, int | None], dict[int, dict[str, Any]]] = {}
    aim3_split_root = core_paths.repo / "fixture_inputs/aim3_splits"
    all_jobs = [
        *(record["job"] for record in aim3_contract["adopted_old_chains"]),
        *aim3_contract["new_jobs"],
    ]
    for job in all_jobs:
        component = str(job["component"])
        task = str(job["task"])
        draw_seed = job.get("draw_seed")
        seed = int(job["model_seed"])
        group = (component, task, draw_seed)
        draw_label = "none" if draw_seed is None else str(draw_seed)
        group_root = aim3_split_root / component / task / draw_label
        manifest_path = group_root / "manifest.csv"
        split_path = group_root / "splits.parquet"
        integrity_path = group_root / "split_integrity.txt"
        if not manifest_path.exists():
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                f"patient_id,task\nfixture,{component}-{task}-{draw_label}\n",
                encoding="utf-8",
            )
            split_path.write_bytes(f"fixture Aim3 split {component}/{task}/{draw_label}".encode())
            _write_split_integrity(manifest_path, split_path, integrity_path)
        training_identity_path = group_root / f"seed{seed}_training_identity.json"
        _write_json(
            training_identity_path,
            {
                "schema_version": 2,
                "payload": {
                    "input_evidence": {
                        "manifest_sha256": bundle.sha256_file(manifest_path),
                        "split_integrity_sha256": bundle.sha256_file(integrity_path),
                    }
                },
            },
        )
        group_artifacts.setdefault(group, {})[seed] = {
            "manifest": _identity(manifest_path),
            "splits": _identity(split_path),
            "split_integrity": _identity(integrity_path),
            "training_identity": _identity(training_identity_path),
        }
    for record in aim3_contract["adopted_old_chains"]:
        job = record["job"]
        record["artifacts"] = group_artifacts[
            (job["component"], job["task"], job.get("draw_seed"))
        ][job["model_seed"]]
    aim3_audit["new_chains"] = [
        {
            "job": {
                "component": job["component"],
                "task": job["task"],
                "draw_seed": job.get("draw_seed"),
                "model_seed": job["model_seed"],
            },
            "artifacts": group_artifacts[(job["component"], job["task"], job.get("draw_seed"))][
                job["model_seed"]
            ],
        }
        for job in aim3_contract["new_jobs"]
    ]
    _write_json(aim3_contract_path, aim3_contract)
    _write_json(aim3_audit_path, aim3_audit)

    raw_cpht_path = original_path("aim2-e2cpht-results")

    def upgrade_cpht(result: dict[str, Any]) -> None:
        result["populations"] = {
            "all_40": {
                "metrics": {
                    "all_conventional": {
                        "auroc": 0.789,
                        "auroc_ci95": [0.619, 0.928],
                    }
                }
            }
        }

    _update_json(raw_cpht_path, upgrade_cpht)
    v9_fixture._rebind_source_graph(core_paths)  # noqa: SLF001


def _fmt_ci(block: dict[str, Any], point_key: str = "auroc", ci_key: str = "auroc_ci95") -> str:
    return f"{float(block[point_key]):.4f} [{float(block[ci_key][0]):.4f}, {float(block[ci_key][1]):.4f}]"


def _write_v10_documents(
    paths: bundle.BundlePaths,
    sources: list[dict[str, Any]],
) -> None:
    by_id = {str(source["id"]): source for source in sources}

    def source_path(source_id: str) -> Path:
        raw = Path(str(by_id[source_id]["path"]))
        return raw.resolve() if raw.is_absolute() else (paths.repo / raw).resolve()

    common = (
        f"{bundle.OFFICIAL_CPHT_NAME}.\n\n"
        "E2-CPHT-R is NOT RUN. Aim 4 whole-section pathology validation is "
        "GENERATED_UNREAD.\n"
    )
    setup = (
        "# FINAL-v10 fixture experimental setup\n\n"
        f"{bundle.FINAL_STATE_STATUS_PARAGRAPH}\n\n"
        "The scheduler uses `max_concurrent_gpu_trainers=6`. Orion is excluded from canonical Aim 1.\n\n"
        "The study-wide MIL census is 1,225 fits: 735 adopted three-seed fits plus 490 new fits for seeds 45 and 46. "
        "The new-fit allocation is Aim 1, 12; Aim 2, 108; and Aim 3, 370.\n\n"
        "## Aim 1 — conventional-primary KRAS ranking\n\n"
        "Canonical Aim 1 contains 1,642 conventional-primary slides from 1,486 patients, including 604 KRAS-mutant and 882 wild-type patients.\n\n"
        "## Aim 2 — transfer, adaptation, and deployment boundaries\n\n"
        "The five-seed scope contains eight controlling LOCO score ensembles plus the RIH "
        "size-matched sensitivity ensemble.\n\n"
        "## Aim 3 — molecular-resolution ceiling\n\n"
        "Fixed, repeated, and E3v tasks use identical within-task folds across seeds.\n\n"
        "## Aim 4 — quantitative morphology and pathology interpretation\n\n" + common
    )
    (paths.final_v10 / "Experimental_Setup.md").write_text(setup, encoding="utf-8")

    aim1 = json.loads(source_path("aim1-e0-five-seed-results").read_text(encoding="utf-8"))
    aim3 = json.loads(source_path("aim3-ladders-five-seed-results").read_text(encoding="utf-8"))
    adjud = json.loads(
        source_path("aim2-e2a-five-seed-adjudication-result").read_text(encoding="utf-8")
    )
    cpht = json.loads(source_path("aim2-e2cpht-results").read_text(encoding="utf-8"))
    primary = aim1["primary_median_seed_macro5"]
    ensemble = aim1["five_seed_ensemble"]
    macro4 = aim1["family_macro4_median_seed"]
    weighted = aim1["patient_count_weighted_macro_median_seed"]
    cptac = aim1["primitive_median_seed_auroc"]["CPTAC"]
    e1v_root = aim3["e1v_virchow2_cls_gene_reference"]
    e1v = e1v_root["five_seed_ensemble_A"]
    aim2_family_macro = adjud["e2a_f"]["nested_four_family_macro"]
    met = adjud["e2met_gate_confirmation"]

    lines = [
        "# FINAL-v10 fixture results",
        "",
        bundle.FINAL_STATE_STATUS_PARAGRAPH,
        "",
        "## Result status and interpretation rule",
        "",
        "| Analysis | Governed top-line result | Governed interpretation |",
        "|---|---|---|",
        f"| Aim 2 E2a | Family macro {_fmt_ci(aim2_family_macro)}; all four sibling directions pass | "
        f"E2a-F `{str(adjud['e2a_f']['adjudication']['claim_family_loco_transport']).upper()}`; "
        f"E2a-D `{str(adjud['e2a_d']['adjudication']['claim_sibling_stratum_transport']).upper()}` |",
        f"| Aim 2 E2-MET | Five-seed conformant macro {met['equal_cohort_metastatic_macro_auroc']:.4f} "
        f"[{met['macro_auroc_ci95'][0]:.4f}, {met['macro_auroc_ci95'][1]:.4f}] | "
        f"Metastatic transport claim `{str(met['claim_metastatic_transport']).upper()}` within the "
        "declared LOCO-sensitivity scope |",
        "",
        "## Aim 1 — conventional-primary KRAS ranking",
        "",
        "### Five-seed E0 governed result",
        "",
        "The authenticated roster contains 1,486 patients and 1,642 slides, including 604 KRAS-mutant and 882 wild-type patients. "
        f"The CPTAC primitive was {bundle._metric_ci_text(cptac['point'], cptac['ci95'])}.",  # noqa: SLF001
        "",
        "| Estimand | AUROC [95% CI] | AUPRC |",
        "|---|---:|---:|",
        f"| **Primary: median of five seed-specific equal-five-domain macros** | {bundle._metric_ci_text(primary['point'], primary['ci95'])} | — |",  # noqa: SLF001
        f"| Five-seed mean-native-logit equal-five-domain macro sensitivity | {bundle._metric_ci_text(ensemble['macro5_auroc'], ensemble['macro5_ci95'])} | — |",  # noqa: SLF001
        f"| Five-seed pooled continuity sensitivity | {ensemble['pooled_auroc']:.4f} | {ensemble['pooled_auprc']:.4f} |",
        f"| Median-seed equal-four-family sensitivity | {bundle._metric_ci_text(macro4['point'], macro4['ci95'])} | — |",  # noqa: SLF001
        f"| Median-seed patient-count-weighted sensitivity | {bundle._metric_ci_text(weighted['point'], weighted['ci95'])} | — |",  # noqa: SLF001
        "",
        "### Five-seed E1v encoder-sensitivity reference",
        "",
        f"The five-seed result was {bundle._metric_ci_text(e1v['auroc'], [e1v['ci_low'], e1v['ci_high']])} on all {e1v['n']:,} patients, including {e1v['n_positive']} mutants, with AUROC {e1v_root['five_seed_ensemble_D_auroc']:.4f} in Set D.",  # noqa: SLF001
        "",
        "## Aim 2 — transfer, adaptation, and deployment boundaries",
        "",
        f"The unchanged raw E2-CPHT AUROC was {_fmt_ci(cpht['populations']['all_40']['metrics']['all_conventional'])}.",
        "",
        "### Five-seed E2a-F family directions",
        "",
    ]
    for name, block in adjud["e2a_f"]["directions"].items():
        lines.append(
            f"| {name} | {_fmt_ci(block)} | `{str(block['directional_gate']['passes']).upper()}` |"
        )
    lines.append(
        f"| Nested four-family macro | {_fmt_ci(adjud['e2a_f']['nested_four_family_macro'])} | panel only |"
    )
    lines.append(
        "| Family LOCO transport claim | "
        f"`{str(adjud['e2a_f']['adjudication']['claim_family_loco_transport']).upper()}` |"
    )
    lines.extend(["", "### Five-seed E2a-D sibling directions", ""])
    for name, block in adjud["e2a_d"]["targets"].items():
        lines.append(
            f"| {name} | {_fmt_ci(block)} | `{str(block['directional_gate']['passes']).upper()}` |"
        )
    for name, block in adjud["e2a_d"]["macros"].items():
        lines.append(f"| {name} | {_fmt_ci(block)} | panel only |")
    lines.append(
        "| Sibling-stratum transport claim | "
        f"`{str(adjud['e2a_d']['adjudication']['claim_sibling_stratum_transport']).upper()}` |"
    )
    lines.extend(["", "### Corrected paired contrasts", ""])
    for slug, block in adjud["e2a_d"]["paired_sibling_minus_family"].items():
        lines.append(
            f"| {slug} sibling minus family | {block['delta_auroc']:+.4f} "
            f"[{block['delta_auroc_ci95'][0]:.4f}, {block['delta_auroc_ci95'][1]:.4f}] |"
        )
    size = adjud["e2a_d"]["size_matched_rih_sensitivity"]
    lines.append(
        f"| RIH size-matched minus full-source | {size['delta_auroc']:+.4f} "
        f"[{size['delta_auroc_ci95'][0]:.4f}, {size['delta_auroc_ci95'][1]:.4f}] |"
    )
    met_rows = [f"| {target} | {_fmt_ci(block)} |" for target, block in met["targets"].items()]
    lines.extend(
        [
            "",
            "### E2-MET conformant gate",
            "",
            *met_rows,
            f"| Equal-cohort metastatic macro | {met['equal_cohort_metastatic_macro_auroc']:.4f} "
            f"[{met['macro_auroc_ci95'][0]:.4f}, {met['macro_auroc_ci95'][1]:.4f}] |",
            f"| Metastatic transport claim | `{str(met['claim_metastatic_transport']).upper()}` |",
            "",
            "### Three-seed E2-MET continuity and unchanged detailed analyses",
            "",
            "The former three-seed family-naive gate is retained as historical continuity, "
            "not as the controlling FINAL-v10 gate.",
            "",
            "## Aim 3 — molecular-resolution ceiling",
            "",
            "### Five-seed fixed UNI-v1 ladder",
            "",
            "| Task | Fine | Control | Delta | Verdict |",
            "|---|---:|---:|---:|---|",
        ]
    )
    task_labels = {
        "codon": "G12 vs non-G12 KRAS",
        "g12d_broad": "G12D vs other KRAS",
        "allele1": "G12D vs other G12",
        "allele2": "G12V vs other G12",
        "g12c": "G12C vs other G12",
    }
    fixed = aim3["fixed_univ1"]["rungs"]
    for task, rung in fixed.items():
        lines.append(
            f"| {task_labels[task]} | {rung['fine']['estimate']:.4f} | "
            f"{rung['control']['estimate']:.4f} | {rung['delta_control_minus_fine']['estimate']:+.4f} | "
            f"`{rung['gate']['verdict']}` |"
        )
    ceiling_order = ("codon", "g12d_broad", "allele1")
    fine_bounds = ", ".join(
        f"{fixed[task]['fine']['primary_fwer_one_sided']['upper']:.4f}" for task in ceiling_order
    )
    control_bounds = ", ".join(
        f"{fixed[task]['control']['primary_fwer_one_sided']['lower']:.4f}" for task in ceiling_order
    )
    delta_bounds = ", ".join(
        f"{fixed[task]['delta_control_minus_fine']['primary_fwer_one_sided']['lower']:.4f}"
        for task in ceiling_order
    )
    lines.extend(
        [
            "",
            "For the three ceiling rungs, the one-sided familywise fine upper bounds were "
            f"{fine_bounds}; control lower bounds were {control_bounds}; and delta lower bounds were {delta_bounds}.",
            "G12V missed the fine-ceiling and delta conditions. G12C did not establish a learnable control contrast.",
            "",
            "### Five-seed repeated-control consensus",
            "",
            "| Task | Consensus |",
            "|---|---|",
        ]
    )
    for task, rung in aim3["repeated_univ1"]["rungs"].items():
        lines.append(f"| {task_labels[task]} | `{rung['consensus_verdict']}` |")
    lines.extend(
        [
            "",
            "### Five-seed Virchow2-CLS replication",
            "",
            "| Task | Fine | Control | Delta | Verdict |",
            "|---|---:|---:|---:|---|",
        ]
    )
    e3v = aim3["e3v_virchow2_cls"]["rungs"]
    for task, rung in e3v.items():
        lines.append(
            f"| {task_labels[task]} | {rung['fine']['estimate']:.4f} | "
            f"{rung['control']['estimate']:.4f} | {rung['delta_control_minus_fine']['estimate']:+.4f} | "
            f"`{rung['gate']['verdict']}` |"
        )
    worst_fine = max(rung["fine"]["primary_fwer_one_sided"]["upper"] for rung in e3v.values())
    worst_control = min(rung["control"]["primary_fwer_one_sided"]["lower"] for rung in e3v.values())
    worst_delta = min(
        rung["delta_control_minus_fine"]["primary_fwer_one_sided"]["lower"] for rung in e3v.values()
    )
    lines.extend(
        [
            "",
            f"The worst fine upper bound was {worst_fine:.4f}, the worst control lower bound was {worst_control:.4f}, and the worst delta lower bound was {worst_delta:.4f}.",
            "",
            "## Aim 4 — quantitative morphology and pathology interpretation",
            "",
            common,
        ]
    )
    (paths.final_v10 / "Results.md").write_text("\n".join(lines), encoding="utf-8")

    audit_hashes = "\n".join(
        [
            "### Governed five-seed source identities",
            "",
            "| Source ID | SHA-256 |",
            "|---|---|",
            *[
                f"| `{source_id}` | `{by_id[source_id]['sha256']}` |"
                for source_id in sorted(bundle._required_sources(paths))  # noqa: SLF001
            ],
        ]
    )
    audit = (
        "# FINAL-v10 fixture audit\n\n"
        f"{bundle.FINAL_STATE_STATUS_PARAGRAPH}\n\n"
        "Five-seed scope: E0; E1v; all nine LOCO primary score ensembles, comprising eight controlling arms plus the RIH size-matched sensitivity; E2-MET; Orion LOCO sensitivity; Aim 3 fixed, repeated, and E3v.\n"
        "Three-seed unchanged scope: raw E2-CPHT; E2-CPHT-A; 15-fold Orion sensitivity; E2e; E2f-v3; between-slide; detailed E2-MET role/organ analyses; unaffected Aim 1 analyses.\n"
        "Study-wide MIL census: 1,225 = 735 adopted + 490 new fits in 102 chains; seeds 42, 43, 44, 45, 46; patient/slide folds unchanged; maximum concurrency 6; observed peak 6.\n\n"
        "## Aim 1 — conventional-primary KRAS ranking\n\n"
        "The canonical patient roster remains 1,486 patients and 1,642 slides, with 604 KRAS-mutant and 882 wild-type patients. "
        f"The primary was {bundle._metric_ci_text(primary['point'], primary['ci95'])}.\n\n"  # noqa: SLF001
        + audit_hashes
        + "\n\n## Aim 2 — transfer, adaptation, and deployment boundaries\n\n"
        "Five-seed analysis controls eight LOCO arms; the ninth ensemble, RIH size-matched, "
        "remains a prespecified sensitivity.\n\n"
        "Corrected field-scoped adjudication controls paired intervals.\n\n"
        "## Aim 3 — molecular-resolution ceiling\n\n"
        "All fixed, repeated, and E3v gates replayed.\n\n"
        "## Aim 4 — quantitative morphology and pathology interpretation\n\n" + common
    )
    (paths.final_v10 / "Audit.md").write_text(audit, encoding="utf-8")


def _new_source_records(paths: bundle.BundlePaths) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_id, (path, core_aim) in bundle._required_sources(paths).items():  # noqa: SLF001
        experiments, role = bundle._EXPECTED_NEW_SOURCE_METADATA[source_id]  # noqa: SLF001
        records.append(
            {
                "id": source_id,
                "aims": bundle._expected_source_aims(source_id, core_aim),  # noqa: SLF001
                "experiments": list(experiments),
                "role": role,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": bundle.sha256_file(path),
            }
        )
    return records


def _make_full_v10_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    layered: bool = False,
) -> SimpleNamespace:
    core_paths, _source_paths = v9_fixture._fixture(tmp_path)  # noqa: SLF001
    original_manifest = json.loads(
        (core_paths.final_v9 / bundle.SOURCE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    original_records = {str(record["id"]): record for record in original_manifest["artifacts"]}
    _upgrade_v9_dag_for_v10(core_paths, original_records)
    rebound_manifest = json.loads(
        (core_paths.final_v9 / bundle.SOURCE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    rebound_records = {str(record["id"]): record for record in rebound_manifest["artifacts"]}

    base_records: list[dict[str, Any]] = []
    base_root = core_paths.repo / "authoritative/v10_base_fixture"
    for source_id in sorted(bundle._EXPECTED_BASE_SOURCE_IDS):  # noqa: SLF001
        if source_id in rebound_records:
            source = dict(rebound_records[source_id])
        else:
            path = base_root / f"{source_id}.json"
            _write_json(path, {"status": "complete", "source": source_id})
            source = {
                "id": source_id,
                "aims": ["Shared"],
                "experiments": ["adopted FINAL-v9 fixture"],
                "role": "adopted_fixture_evidence",
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": bundle.sha256_file(path),
            }
        base_records.append(source)
    _write_json(
        core_paths.final_v9 / bundle.SOURCE_MANIFEST_NAME,
        {"schema_version": 1, "bundle": "final_v9", "artifacts": base_records},
    )
    base_manifest = core_paths.final_v9 / bundle.SOURCE_MANIFEST_NAME
    monkeypatch.setattr(bundle, "EXPECTED_BASE_MANIFEST_SHA256", bundle.sha256_file(base_manifest))

    final_v10 = core_paths.repo / "reports/final_v10"
    final_v10.mkdir(parents=True)
    verifier_code = core_paths.repo / "tools/final_v10_bundle_receipt.py"
    verifier_test = core_paths.repo / "tests/test_final_v10_bundle_receipt.py"
    shutil.copyfile(Path(bundle.__file__), verifier_code)
    shutil.copyfile(Path(__file__), verifier_test)
    paths = bundle.BundlePaths(
        repo=core_paths.repo,
        final_v10=final_v10,
        destination=final_v10 / bundle.FINAL_RECEIPT_NAME,
        verifier_code=verifier_code,
        verifier_test=verifier_test,
        base_manifest=base_manifest,
        core_verifier=bundle.CORE_VERIFIER,
        campaign_root=core_paths.campaign_root,
        adjudication_root=core_paths.adjudication_root,
        expected_created_utc=_FIXTURE_CREATED_UTC,
    )
    paths = dataclasses.replace(
        paths,
        expected_terminal_receipt_sha256=_terminal_receipt_sha256_pins(paths),
    )
    adopted = bundle._validate_base_sources(  # noqa: SLF001
        paths,
        dict(bundle._EXPECTED_ROLE_OVERRIDES),  # noqa: SLF001
    )
    new = _new_source_records(paths)
    combined = sorted([*adopted, *new], key=lambda value: str(value["id"]))
    _write_v10_documents(paths, combined)
    paths = dataclasses.replace(
        paths,
        expected_document_sha256=_document_sha256_pins(paths),
    )
    manifest_path = final_v10 / bundle.SOURCE_MANIFEST_NAME
    if layered:
        _write_json(
            manifest_path,
            {
                "schema_version": 2,
                "bundle": "final_v10",
                "status": bundle.DRAFT_STATUS,
                "base_manifest": _identity(base_manifest),
                "role_overrides": dict(bundle._EXPECTED_ROLE_OVERRIDES),  # noqa: SLF001
                "artifacts": new,
                "pending_artifacts": [],
            },
        )
    else:
        _write_json(
            manifest_path,
            {
                "schema_version": 2,
                "bundle": "final_v10",
                "status": bundle.CANDIDATE_STATUS,
                "artifacts": combined,
                "pending_artifacts": [],
            },
        )
    return SimpleNamespace(
        paths=paths,
        core_paths=core_paths,
        base=adopted,
        new=new,
        combined=combined,
        manifest_path=manifest_path,
    )


def _resolve_fixture_path(raw_path: object, repo: Path) -> Path:
    raw = Path(str(raw_path))
    return raw.resolve() if raw.is_absolute() else (repo / raw).resolve()


def _rebind_v10_new_graph(
    fixture: SimpleNamespace,
    *,
    rewrite_documents: bool = True,
    refresh_terminal_pins: bool = False,
) -> None:
    """Rehash a coordinated synthetic DAG so semantic tampering reaches the deep gates."""

    paths = fixture.paths
    required_paths = {
        path.resolve()
        for path, _aim in bundle._required_sources(paths).values()  # noqa: SLF001
    }
    required_paths.update(
        path.resolve() for path in (paths.campaign_root / "campaign/receipts/jobs").glob("*.json")
    )
    known = set(required_paths)

    def discover(value: object) -> None:
        if isinstance(value, dict):
            if {"path", "size_bytes", "sha256"}.issubset(value):
                candidate = _resolve_fixture_path(value["path"], paths.repo)
                if candidate.is_file():
                    known.add(candidate)
            for child in value.values():
                discover(child)
        elif isinstance(value, list):
            for child in value:
                discover(child)

    for _iteration in range(30):
        before_known = len(known)
        for source_path in list(known):
            if source_path.suffix != ".json":
                continue
            try:
                discover(json.loads(source_path.read_text(encoding="utf-8")))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue

        changed = False

        def refresh(value: object) -> None:
            nonlocal changed
            if isinstance(value, dict):
                if {"path", "size_bytes", "sha256"}.issubset(value):
                    target = _resolve_fixture_path(value["path"], paths.repo)
                    if target in known and target.is_file():
                        wanted_size = target.stat().st_size
                        wanted_hash = bundle.sha256_file(target)
                        if value["size_bytes"] != wanted_size or value["sha256"] != wanted_hash:
                            value["size_bytes"] = wanted_size
                            value["sha256"] = wanted_hash
                            changed = True
                for child in value.values():
                    refresh(child)
            elif isinstance(value, list):
                for child in value:
                    refresh(child)

        for source_path in list(known):
            if source_path.suffix != ".json":
                continue
            try:
                value = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            before = json.dumps(value, sort_keys=True)
            refresh(value)
            if json.dumps(value, sort_keys=True) != before:
                _write_json(source_path, value)
        if not changed and len(known) == before_known:
            break
    else:  # pragma: no cover - fixture bug
        raise AssertionError("v10 synthetic DAG did not converge")

    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    required_ids = set(bundle._required_sources(paths))  # noqa: SLF001
    for record in manifest["artifacts"]:
        if record["id"] not in required_ids:
            continue
        source_path = _resolve_fixture_path(record["path"], paths.repo)
        record["size_bytes"] = source_path.stat().st_size
        record["sha256"] = bundle.sha256_file(source_path)
    _write_json(fixture.manifest_path, manifest)
    fixture.combined = sorted(manifest["artifacts"], key=lambda value: str(value["id"]))
    updated_paths = paths
    if refresh_terminal_pins:
        updated_paths = dataclasses.replace(
            updated_paths,
            expected_terminal_receipt_sha256=_terminal_receipt_sha256_pins(updated_paths),
        )
    if rewrite_documents:
        _write_v10_documents(paths, fixture.combined)
        updated_paths = dataclasses.replace(
            updated_paths,
            expected_document_sha256=_document_sha256_pins(updated_paths),
        )
    fixture.paths = updated_paths


def _mutate_required_json(
    fixture: SimpleNamespace,
    source_id: str,
    mutation: Callable[[dict[str, Any]], None],
    *,
    rewrite_documents: bool = True,
) -> None:
    source_path = bundle._required_sources(fixture.paths)[source_id][0]  # noqa: SLF001
    _update_json(source_path, mutation)
    _rebind_v10_new_graph(
        fixture,
        rewrite_documents=rewrite_documents,
        refresh_terminal_pins=True,
    )


def test_full_dag_flat_candidate_seals_verifies_and_is_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    candidate = bundle.build_receipt(fixture.paths)
    assert len(candidate["authoritative_sources"]) == 81
    assert candidate["adopted_base_manifest"] == {
        **_identity(fixture.paths.base_manifest),
        "path": "reports/final_v9/source_manifest.json",
    }
    assert candidate["verification"]["frozen_semantic_core"]["sha256"] == (
        bundle.EXPECTED_CORE_VERIFIER_SHA256
    )
    assert candidate["study_wide_mil_census"]["complete_fits"] == 1225
    assert candidate["checks"]["direct_json_and_split_semantic_replay"] == "PASS"
    assert candidate["checks"]["pinned_full_component_verification_receipts"] == "PASS"
    assert candidate["checks"]["upgraded_and_controlling_headline_validation"] == "PASS"
    assert (
        candidate["checks"][
            "unchanged_continuity_claims_independently_reviewed_and_final_document_pinned"
        ]
        == "PASS"
    )
    assert "corrected_aim2_adjudication_replay" not in candidate["checks"]
    assert "reported_number_to_source_validation" not in candidate["checks"]

    sealed = bundle.seal(fixture.paths)
    assert fixture.paths.destination.is_file()
    rehashed_paths: list[Path] = []
    direct_sha256 = bundle.sha256_file

    def record_rehash(path: Path) -> str:
        rehashed_paths.append(Path(path).resolve())
        return direct_sha256(path)

    monkeypatch.setattr(bundle, "sha256_file", record_rehash)
    assert bundle.verify_published_receipt(fixture.paths) == sealed
    authoritative_paths = {
        _resolve_fixture_path(source["path"], fixture.paths.repo)
        for source in sealed["authoritative_sources"]
    }
    assert len(authoritative_paths) == 81
    assert authoritative_paths.issubset(set(rehashed_paths))
    with pytest.raises(bundle.BundleVerificationError, match="refusing to overwrite"):
        bundle.seal(fixture.paths)


def test_real_layered_partial_transition_flattens_to_exact_81(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    by_id = {str(record["id"]): record for record in fixture.new}
    pending_ids = {
        "aim2-e2a-five-seed-adjudication-result",
        "aim2-e2a-five-seed-adjudication-receipt",
    }
    result_path = bundle._required_sources(fixture.paths)[  # noqa: SLF001
        "aim2-e2a-five-seed-adjudication-result"
    ][0]
    held_path = result_path.with_name(f".{result_path.name}.held")
    result_path.rename(held_path)
    pending = [
        {
            key: value
            for key, value in by_id[source_id].items()
            if key not in {"size_bytes", "sha256"}
        }
        for source_id in sorted(pending_ids)
    ]
    _write_json(
        fixture.manifest_path,
        {
            "schema_version": 2,
            "bundle": "final_v10",
            "status": bundle.DRAFT_STATUS,
            "base_manifest": _identity(fixture.paths.base_manifest),
            "role_overrides": dict(bundle._EXPECTED_ROLE_OVERRIDES),  # noqa: SLF001
            "artifacts": [record for record in fixture.new if str(record["id"]) not in pending_ids],
            "pending_artifacts": pending,
        },
    )

    partial = bundle.refresh_manifest(fixture.paths)
    partial_manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    assert partial["pending_source_count"] == 1
    assert set(partial_manifest) == bundle._DRAFT_MANIFEST_KEYS  # noqa: SLF001
    assert len(partial_manifest["artifacts"]) == 28

    held_path.rename(result_path)
    complete = bundle.refresh_manifest(fixture.paths)
    flat = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    assert complete["pending_source_count"] == 0
    assert set(flat) == bundle._FLAT_MANIFEST_KEYS  # noqa: SLF001
    assert flat["status"] == bundle.CANDIDATE_STATUS
    assert len(flat["artifacts"]) == 81
    assert flat["pending_artifacts"] == []
    assert "base_manifest" not in flat and "role_overrides" not in flat
    assert len(bundle.build_receipt(fixture.paths)["authoritative_sources"]) == 81


def test_invalid_last_source_promotion_keeps_recoverable_layered_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    source_id = "aim1-e0-five-seed-results"
    _mutate_required_json(
        fixture,
        source_id,
        lambda value: value["primary_median_seed_macro5"].__setitem__("point", 0.641),
    )
    flat = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in flat["artifacts"]}
    result_path = bundle._required_sources(fixture.paths)[source_id][0]  # noqa: SLF001
    held_path = result_path.with_name(f".{result_path.name}.held")
    result_path.rename(held_path)
    new_ids = set(bundle._required_sources(fixture.paths))  # noqa: SLF001
    _write_json(
        fixture.manifest_path,
        {
            "schema_version": 2,
            "bundle": "final_v10",
            "status": bundle.DRAFT_STATUS,
            "base_manifest": _identity(fixture.paths.base_manifest),
            "role_overrides": dict(bundle._EXPECTED_ROLE_OVERRIDES),  # noqa: SLF001
            "artifacts": [by_id[artifact_id] for artifact_id in sorted(new_ids - {source_id})],
            "pending_artifacts": [
                {
                    key: value
                    for key, value in by_id[source_id].items()
                    if key not in {"size_bytes", "sha256"}
                }
            ],
        },
    )
    held_path.rename(result_path)

    with pytest.raises(
        bundle.BundleVerificationError, match="primary is not the median seed macro"
    ):
        bundle.refresh_manifest(fixture.paths)
    persisted = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    assert set(persisted) == bundle._DRAFT_MANIFEST_KEYS  # noqa: SLF001
    assert persisted["status"] == bundle.DRAFT_STATUS
    assert [record["id"] for record in persisted["pending_artifacts"]] == [source_id]


def test_recorded_identity_rejects_same_byte_lexical_symlink(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    declared = tmp_path / "governed.json"
    _write_json(declared, {"status": "complete"})
    record = _identity(declared)
    backing = tmp_path / "governed.same-bytes.json"
    shutil.copyfile(declared, backing)
    declared.unlink()
    declared.symlink_to(backing)
    assert declared.read_bytes() == backing.read_bytes()
    with pytest.raises(bundle.BundleVerificationError, match="declared path.*symlink"):
        bundle._validate_recorded_identity(  # noqa: SLF001
            record,
            paths,
            context="synthetic governed dependency",
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("role", "confirmatory_five_seed_orion_raw_cpht_scores"),
        ("experiments", ["Raw E2-CPHT confirmatory analysis"]),
    ],
)
def test_orion_source_metadata_cannot_be_relabelled_as_confirmatory_raw_cpht(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    orion = next(
        record
        for record in manifest["artifacts"]
        if record["id"] == "aim2-loco-five-seed-orion-patients"
    )
    orion[field] = replacement
    _write_json(fixture.manifest_path, manifest)
    with pytest.raises(bundle.BundleVerificationError, match="exact experiments/role metadata"):
        bundle.build_receipt(fixture.paths)


def test_frozen_base_ledger_and_semantic_core_are_directly_authenticated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    bundle.build_receipt(fixture.paths)

    fixture.paths.base_manifest.write_text(
        fixture.paths.base_manifest.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(bundle.BundleVerificationError, match="source-manifest identity drift"):
        bundle.build_receipt(fixture.paths)

    fixture = _make_full_v10_fixture(tmp_path / "core", monkeypatch)
    copied_core = fixture.paths.repo / "tools/frozen_final_v9_core.py"
    shutil.copyfile(bundle.CORE_VERIFIER, copied_core)
    paths = dataclasses.replace(fixture.paths, core_verifier=copied_core)
    monkeypatch.setattr(bundle, "EXPECTED_CORE_VERIFIER_SHA256", bundle.sha256_file(copied_core))
    bundle.build_receipt(paths)
    copied_core.write_text(
        copied_core.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8"
    )
    with pytest.raises(bundle.BundleVerificationError, match="semantic verifier identity drift"):
        bundle.build_receipt(paths)


@pytest.mark.parametrize(
    "target",
    ["adopted_source", "new_source", "results_document", "verifier_code", "verifier_test"],
)
def test_published_receipt_rejects_direct_post_seal_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    bundle.seal(fixture.paths)
    if target == "adopted_source":
        target_path = _resolve_fixture_path(fixture.base[0]["path"], fixture.paths.repo)
    elif target == "new_source":
        target_path = bundle._required_sources(fixture.paths)[  # noqa: SLF001
            "aim1-e0-five-seed-results"
        ][0]
    elif target == "results_document":
        target_path = fixture.paths.final_v10 / "Results.md"
    elif target == "verifier_code":
        target_path = fixture.paths.verifier_code
    else:
        target_path = fixture.paths.verifier_test
    target_path.write_bytes(target_path.read_bytes() + b"\npost-seal drift\n")
    with pytest.raises(
        bundle.BundleVerificationError,
        match="drift|identity|hash|verification|changed|does not match|mismatch",
    ):
        bundle.verify_published_receipt(fixture.paths)


@pytest.mark.parametrize(
    "target",
    ["new_source", "terminal_source", "base_manifest", "document"],
)
def test_candidate_rejects_same_byte_lexical_dependency_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    if target == "new_source":
        target_path = bundle._required_sources(fixture.paths)[  # noqa: SLF001
            "aim1-e0-five-seed-patient-logits"
        ][0]
    elif target == "terminal_source":
        target_path = bundle._required_sources(fixture.paths)[  # noqa: SLF001
            "aim2-e2a-five-seed-adjudication-receipt"
        ][0]
    elif target == "base_manifest":
        target_path = fixture.paths.base_manifest
    else:
        target_path = fixture.paths.final_v10 / "Results.md"
    backing = target_path.with_name(f".{target_path.name}.same-bytes")
    shutil.copyfile(target_path, backing)
    target_path.unlink()
    target_path.symlink_to(backing)
    assert target_path.read_bytes() == backing.read_bytes()
    with pytest.raises(bundle.BundleVerificationError, match="non-symlink|symlink"):
        bundle.build_receipt(fixture.paths)


def test_published_receipt_rejects_timestamp_only_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    bundle.seal(fixture.paths)
    published = json.loads(fixture.paths.destination.read_text(encoding="utf-8"))
    assert published["created_utc"] == _FIXTURE_CREATED_UTC
    published["created_utc"] = "2026-08-24T00:00:01+00:00"
    fixture.paths.destination.write_bytes(bundle._receipt_bytes(published))  # noqa: SLF001
    with pytest.raises(bundle.BundleVerificationError, match="frozen candidate receipt"):
        bundle.verify_published_receipt(fixture.paths)


def test_published_receipt_rejects_same_byte_lexical_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    bundle.seal(fixture.paths)
    backing = fixture.paths.destination.with_name(f".{fixture.paths.destination.name}.same-bytes")
    shutil.copyfile(fixture.paths.destination, backing)
    fixture.paths.destination.unlink()
    fixture.paths.destination.symlink_to(backing)
    assert fixture.paths.destination.read_bytes() == backing.read_bytes()
    with pytest.raises(bundle.BundleVerificationError, match="regular non-symlink"):
        bundle.verify_published_receipt(fixture.paths)


@pytest.mark.parametrize(
    ("document", "appended_claim"),
    [
        (
            "Results.md",
            "Unchanged continuity metric: 0.7893 [0.6187, 0.9280].",
        ),
        (
            "Audit.md",
            "One unresolved review still gates final publication.",
        ),
    ],
)
def test_candidate_rejects_unreviewed_appended_document_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: str,
    appended_claim: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    document_path = fixture.paths.final_v10 / document
    document_path.write_text(
        document_path.read_text(encoding="utf-8") + f"\n{appended_claim}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        bundle.BundleVerificationError,
        match="frozen FINAL-v10 document identity drift",
    ):
        bundle.build_receipt(fixture.paths)


@pytest.mark.parametrize(
    "stale_state",
    [
        "Governed replay is in progress.",
        "The FINAL-v10 receipt remains blocked.",
        "Immutable publication and reconciliation are active seal blockers.",
        "The FINAL-v10 receipt is absent by design.",
        "Four governed artifacts remain pending.",
        "This is the Working Copy for review.",
        "This Working Document is provisional.",
        "This Working Bundle is not final.",
        "The receipt MUST REMAIN ABSENT UNTIL approval.",
        "The receipt remains absent until review.",
        "Before Sealing, reconcile the inventory.",
        "The Current Staged Manifest is provisional.",
        "Final Materialization Must follow the last replay.",
        "Source-Manifest Materialization follows reconciliation.",
        "Final Reconciliation is required.",
        "There is one Seal Blocker.",
        "Awaiting completion.",
        "The adjudications ARE STILL IN PROGRESS.",
        "THE FINAL-v10 RECEIPT REMAINS BLOCKED.",
        "FOUR GOVERNED ARTIFACTS REMAIN PENDING.",
        "THIS BUNDLE IS draft_unsealed.",
        "The draft manifest has not been promoted.",
        "Document reconciliation will follow.",
    ],
)
def test_candidate_rejects_explicit_stale_finalization_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_state: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    audit_path = fixture.paths.final_v10 / "Audit.md"
    audit_path.write_text(
        audit_path.read_text(encoding="utf-8") + f"\n{stale_state}\n",
        encoding="utf-8",
    )
    with pytest.raises(bundle.BundleVerificationError, match="draft marker"):
        bundle.build_receipt(fixture.paths)


def test_candidate_requires_exact_completion_status_paragraph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    for filename in bundle.REPORT_DOCUMENTS:
        document_path = fixture.paths.final_v10 / filename
        original = document_path.read_text(encoding="utf-8")
        document_path.write_text(
            original.replace(bundle.FINAL_STATE_STATUS_PARAGRAPH, "Evidence is complete."),
            encoding="utf-8",
        )
        with pytest.raises(
            bundle.BundleVerificationError,
            match="exact FINAL-v10 completion-status paragraph",
        ):
            bundle.build_receipt(fixture.paths)
        document_path.write_text(original, encoding="utf-8")


@pytest.mark.parametrize(
    ("payload_source_id", "terminal_receipt_id"),
    [
        (
            "aim1-e0-five-seed-patient-logits",
            "final-v9-five-seed-campaign-results-completion",
        ),
        (
            "aim2-loco-five-seed-primary-patients",
            "final-v9-five-seed-campaign-results-completion",
        ),
        (
            "aim3-ladders-five-seed-bootstrap",
            "final-v9-five-seed-campaign-results-completion",
        ),
        (
            "aim2-e2a-five-seed-adjudication-bootstrap",
            "aim2-e2a-five-seed-adjudication-receipt",
        ),
    ],
)
def test_coordinated_junk_payload_and_receipt_rehash_cannot_bypass_terminal_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload_source_id: str,
    terminal_receipt_id: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    requirements = bundle._required_sources(fixture.paths)  # noqa: SLF001
    requirements[payload_source_id][0].write_bytes(b"J" * 23)
    _rebind_v10_new_graph(fixture)
    pinned = fixture.paths.expected_terminal_receipt_sha256
    assert pinned is not None
    assert bundle.sha256_file(requirements[terminal_receipt_id][0]) != pinned[terminal_receipt_id]
    with pytest.raises(
        bundle.BundleVerificationError,
        match="pinned full-component verification receipt changed",
    ):
        bundle.build_receipt(fixture.paths)


def test_production_defaults_cannot_disable_candidate_byte_pins() -> None:
    paths = bundle.default_paths()
    assert paths.expected_document_sha256 == bundle.EXPECTED_FINAL_DOCUMENT_SHA256
    assert (
        paths.expected_terminal_receipt_sha256
        == bundle.EXPECTED_TERMINAL_VERIFICATION_RECEIPT_SHA256
    )
    assert paths.expected_created_utc == bundle.EXPECTED_RECEIPT_CREATED_UTC
    with pytest.raises(bundle.BundleVerificationError, match="pin roster is not frozen"):
        bundle._validate_frozen_document_bytes(  # noqa: SLF001
            dataclasses.replace(paths, expected_document_sha256=None)
        )
    with pytest.raises(bundle.BundleVerificationError, match="pin roster is not frozen"):
        bundle._validate_terminal_verification_receipts(  # noqa: SLF001
            dataclasses.replace(paths, expected_terminal_receipt_sha256=None),
            [],
        )
    with pytest.raises(
        bundle.BundleVerificationError,
        match="production FINAL-v10 documents must use the frozen verifier constants",
    ):
        bundle._validate_frozen_document_bytes(  # noqa: SLF001
            dataclasses.replace(
                paths,
                expected_document_sha256={
                    filename: "0" * 64 for filename in bundle.REPORT_DOCUMENTS
                },
            )
        )
    with pytest.raises(
        bundle.BundleVerificationError,
        match="production terminal verification receipts must use the frozen verifier constants",
    ):
        bundle._validate_terminal_verification_receipts(  # noqa: SLF001
            dataclasses.replace(
                paths,
                expected_terminal_receipt_sha256={
                    source_id: "0" * 64
                    for source_id in bundle.EXPECTED_TERMINAL_VERIFICATION_RECEIPT_SHA256
                },
            ),
            [],
        )
    with pytest.raises(
        bundle.BundleVerificationError,
        match="production FINAL-v10 receipt must use the frozen verifier timestamp constant",
    ):
        bundle._validated_receipt_created_utc(  # noqa: SLF001
            dataclasses.replace(paths, expected_created_utc=_FIXTURE_CREATED_UTC)
        )


@pytest.mark.parametrize(
    ("source_id", "mutation", "message", "rewrite_documents"),
    [
        (
            "aim1-e0-five-seed-results",
            lambda value: value["primary_median_seed_macro5"].__setitem__("point", 0.641),
            "primary is not the median seed macro",
            True,
        ),
        (
            "aim3-ladders-five-seed-results",
            lambda value: value["fixed_univ1"]["rungs"]["codon"]["gate"].__setitem__(
                "verdict", "INCONCLUSIVE"
            ),
            "verdict does not replay",
            True,
        ),
        (
            "aim3-ladders-five-seed-results",
            lambda value: value["repeated_univ1"]["rungs"]["codon"].__setitem__(
                "consensus_verdict", "NO_CEILING_CONSENSUS"
            ),
            "consensus does not replay",
            True,
        ),
        (
            "aim2-e2a-five-seed-adjudication-result",
            lambda value: value["e2a_d"].pop("paired_sibling_minus_family"),
            "paired-contrast roster changed",
            False,
        ),
        (
            "aim2-e2a-five-seed-adjudication-result",
            lambda value: value["e2met_gate_confirmation"].__setitem__(
                "claim_metastatic_transport", True
            ),
            "E2-MET gate does not replay",
            True,
        ),
    ],
)
def test_full_dag_semantic_tampering_fails_after_coordinated_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_id: str,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
    rewrite_documents: bool,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    _mutate_required_json(
        fixture,
        source_id,
        mutation,
        rewrite_documents=rewrite_documents,
    )
    with pytest.raises(bundle.BundleVerificationError, match=message):
        bundle.build_receipt(fixture.paths)


@pytest.mark.parametrize("component", ["aim1", "aim2", "aim3"])
def test_count_preserving_split_membership_hash_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    requirements = bundle._required_sources(fixture.paths)  # noqa: SLF001
    if component == "aim1":
        training_path = requirements["aim1-e0-five-seed-training-validation"][0]
        training = json.loads(training_path.read_text(encoding="utf-8"))
        identity_path = Path(training["runs"]["46"]["artifacts"]["identity"]["path"])
    elif component == "aim2":
        contract_path = requirements["aim2-loco-five-seed-contract"][0]
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        adoption = next(
            record
            for record in contract["adoptions"]
            if record["kind"] == "training"
            and record["arm"] == "family_cptac"
            and record["seed"] == 44
        )
        identity_path = Path(adoption["source_cv"]["identity"]["path"])
    else:
        audit_path = requirements["aim3-ladders-five-seed-analysis-audit"][0]
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        record = next(
            record
            for record in audit["new_chains"]
            if record["job"]["component"] == "fixed"
            and record["job"]["task"] == "codon"
            and record["job"]["model_seed"] == 46
        )
        identity_path = Path(record["artifacts"]["training_identity"]["path"])
    _update_json(
        identity_path,
        lambda value: value["payload"]["input_evidence"].__setitem__(
            "split_integrity_sha256", "f" * 64
        ),
    )
    _rebind_v10_new_graph(fixture, refresh_terminal_pins=True)
    with pytest.raises(bundle.BundleVerificationError, match="split|membership|input evidence"):
        bundle.build_receipt(fixture.paths)


def test_correct_row_in_wrong_aim3_section_cannot_rescue_corrupted_fixed_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    results_path = fixture.paths.final_v10 / "Results.md"
    text = results_path.read_text(encoding="utf-8")
    correct = "| G12 vs non-G12 KRAS | 0.5100 | 0.6600 | +0.1500 | `CEILING` |"
    corrupted = "| G12 vs non-G12 KRAS | 0.5110 | 0.6600 | +0.1490 | `CEILING` |"
    assert text.count(correct) == 1
    text = text.replace(correct, corrupted)
    e3v_heading = "### Five-seed Virchow2-CLS replication"
    text = text.replace(e3v_heading, f"{e3v_heading}\n\n{correct}", 1)
    results_path.write_text(text, encoding="utf-8")
    with pytest.raises(bundle.BundleVerificationError, match="does not map governed"):
        bundle.build_receipt(fixture.paths)


def test_duplicate_correct_row_cannot_rescue_corrupted_detailed_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    results_path = fixture.paths.final_v10 / "Results.md"
    text = results_path.read_text(encoding="utf-8")
    correct = "| G12 vs non-G12 KRAS | 0.5100 | 0.6600 | +0.1500 | `CEILING` |"
    corrupted = "| G12 vs non-G12 KRAS | 0.5110 | 0.6600 | +0.1490 | `CEILING` |"
    assert text.count(correct) == 1
    text = text.replace(correct, f"{corrupted}\n{correct}", 1)
    results_path.write_text(text, encoding="utf-8")
    with pytest.raises(bundle.BundleVerificationError, match="duplicate|exactly one|does not map"):
        bundle.build_receipt(fixture.paths)


@pytest.mark.parametrize("claim", ["paired_interval", "directional_gate", "e2met_gate"])
def test_aim2_governed_gate_and_paired_claims_are_document_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    claim: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    results_path = fixture.paths.final_v10 / "Results.md"
    text = results_path.read_text(encoding="utf-8")
    if claim == "paired_interval":
        old = "| sr386 sibling minus family | +0.0200 [-0.0200, 0.0900] |"
        new = "| sr386 sibling minus family | +0.0200 [-0.0100, 0.0900] |"
    elif claim == "directional_gate":
        old = "| CPTAC | 0.6100 [0.5100, 0.7300] | `TRUE` |"
        new = "| CPTAC | 0.6100 [0.5100, 0.7300] | `FALSE` |"
    else:
        old = "| Metastatic transport claim | `FALSE` |"
        new = "| Metastatic transport claim | `TRUE` |"
    assert old in text
    results_path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(
        bundle.BundleVerificationError, match="Aim-2|paired|gate|claim|does not map"
    ):
        bundle.build_receipt(fixture.paths)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "Family macro 0.6510 [0.5710, 0.7310]; all four sibling directions pass",
            "Family macro 0.6520 [0.5710, 0.7310]; all four sibling directions pass",
        ),
        (
            "Metastatic transport claim `FALSE` within the declared LOCO-sensitivity scope",
            "Metastatic transport claim `TRUE` within the declared LOCO-sensitivity scope",
        ),
    ],
)
def test_wrong_aim2_top_summary_cannot_be_rescued_by_correct_detailed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    results_path = fixture.paths.final_v10 / "Results.md"
    text = results_path.read_text(encoding="utf-8")
    assert old in text
    results_path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(bundle.BundleVerificationError, match="Aim-2|summary|does not map"):
        bundle.build_receipt(fixture.paths)


@pytest.mark.parametrize(
    "stale_prose",
    [
        "Metastatic transport was not established.",
        "These three-seed results remain the scientific E2-MET gate record.",
    ],
)
def test_correct_e2met_precedence_cannot_rescue_stale_scientific_gate_prose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_prose: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    results_path = fixture.paths.final_v10 / "Results.md"
    results_path.write_text(
        results_path.read_text(encoding="utf-8") + f"\n{stale_prose}\n",
        encoding="utf-8",
    )
    with pytest.raises(bundle.BundleVerificationError, match="contradictory Aim-2 prose"):
        bundle.build_receipt(fixture.paths)


@pytest.mark.parametrize("tamper", ["missing", "wrong"])
def test_audit_must_bind_exact_aim2_new_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    source_id = "aim2-loco-five-seed-results"
    source = next(record for record in fixture.combined if record["id"] == source_id)
    governed_line = f"| `{source_id}` | `{source['sha256']}` |"
    audit_path = fixture.paths.final_v10 / "Audit.md"
    text = audit_path.read_text(encoding="utf-8")
    assert text.count(governed_line) == 1
    replacement = "" if tamper == "missing" else f"| `{source_id}` | `{'f' * 64}` |"
    audit_path.write_text(text.replace(governed_line, replacement, 1), encoding="utf-8")
    with pytest.raises(bundle.BundleVerificationError, match="Audit source identity table"):
        bundle.build_receipt(fixture.paths)


@pytest.mark.parametrize("tamper", ["swap", "mislabel", "duplicate"])
def test_audit_source_identity_table_cannot_be_rebound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    by_id = {str(record["id"]): record for record in fixture.combined}
    first_id = "aim2-loco-five-seed-results"
    second_id = "aim3-ladders-five-seed-results"
    first_row = f"| `{first_id}` | `{by_id[first_id]['sha256']}` |"
    second_row = f"| `{second_id}` | `{by_id[second_id]['sha256']}` |"
    audit_path = fixture.paths.final_v10 / "Audit.md"
    text = audit_path.read_text(encoding="utf-8")
    if tamper == "swap":
        text = text.replace(first_row, "__FIRST_SOURCE_ROW__", 1)
        text = text.replace(
            second_row,
            f"| `{second_id}` | `{by_id[first_id]['sha256']}` |",
            1,
        )
        text = text.replace(
            "__FIRST_SOURCE_ROW__",
            f"| `{first_id}` | `{by_id[second_id]['sha256']}` |",
            1,
        )
    elif tamper == "mislabel":
        text = text.replace(
            first_row,
            f"| `aim2-loco-five-seed-resultz` | `{by_id[first_id]['sha256']}` |",
            1,
        )
    else:
        text = text.replace(first_row, f"{first_row}\n{first_row}", 1)
    audit_path.write_text(text, encoding="utf-8")
    with pytest.raises(bundle.BundleVerificationError, match="Audit source identity table"):
        bundle.build_receipt(fixture.paths)


@pytest.mark.parametrize(
    ("document", "old", "new"),
    [
        (
            "Experimental_Setup.md",
            "eight controlling LOCO score ensembles plus the RIH size-matched sensitivity ensemble",
            "nine controlling LOCO score ensembles",
        ),
        (
            "Audit.md",
            "Five-seed analysis controls eight LOCO arms; the ninth ensemble, RIH size-matched, "
            "remains a prespecified sensitivity.",
            "Five-seed analysis controls the nine LOCO ensembles.",
        ),
        (
            "Experimental_Setup.md",
            "The five-seed scope contains eight controlling LOCO score ensembles plus the RIH "
            "size-matched sensitivity ensemble.",
            "All nine LOCO arms are controlling.",
        ),
    ],
)
def test_mixed_scope_documents_reject_nine_controlling_loco_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: str,
    old: str,
    new: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    document_path = fixture.paths.final_v10 / document
    text = document_path.read_text(encoding="utf-8")
    assert old in text
    document_path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(bundle.BundleVerificationError, match="LOCO|scope|controlling|sensitivity"):
        bundle.build_receipt(fixture.paths)


@pytest.mark.parametrize("document", ["Experimental_Setup.md", "Results.md"])
def test_correct_scope_text_cannot_rescue_coexisting_rih_sm_controlling_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    document_path = fixture.paths.final_v10 / document
    document_path.write_text(
        document_path.read_text(encoding="utf-8")
        + "\nRIH size-matched is a controlling LOCO arm.\n",
        encoding="utf-8",
    )
    with pytest.raises(bundle.BundleVerificationError, match="RIH|LOCO|controlling|sensitivity"):
        bundle.build_receipt(fixture.paths)


@pytest.mark.parametrize(
    ("document", "false_claim"),
    [
        ("Experimental_Setup.md", "Canonical Aim 1 INCLUDES Orion."),
        ("Results.md", "Orion LOCO is CONFIRMATORY raw E2-CPHT evidence."),
        ("Audit.md", "E2-CPHT-R COMPLETED and PASSED."),
        (
            "Experimental_Setup.md",
            "Whole-section pathology validation was READ and CONFIRMED.",
        ),
        ("Results.md", "Model seeds are INDEPENDENT INFERENCE UNITS."),
        ("Audit.md", "Patient/slide folds CHANGED across model seeds."),
    ],
)
def test_candidate_rejects_coexisting_false_scientific_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: str,
    false_claim: str,
) -> None:
    fixture = _make_full_v10_fixture(tmp_path, monkeypatch)
    document_path = fixture.paths.final_v10 / document
    document_path.write_text(
        document_path.read_text(encoding="utf-8") + f"\n{false_claim}\n",
        encoding="utf-8",
    )
    with pytest.raises(bundle.BundleVerificationError, match="contradictory scientific claim"):
        bundle.build_receipt(fixture.paths)


def test_post_link_verification_failure_does_not_leave_poisoned_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        bundle,
        "build_receipt",
        lambda _paths_value: {"created_utc": "2026-08-24T00:00:00+00:00"},
    )
    monkeypatch.setattr(
        bundle,
        "verify_published_receipt",
        lambda _paths_value: (_ for _ in ()).throw(
            bundle.BundleVerificationError("post-link identity drift")
        ),
    )
    with pytest.raises(bundle.BundleVerificationError, match="post-link identity drift"):
        bundle.seal(paths)
    assert not paths.destination.exists()


def test_first_publication_directory_fsync_failure_removes_link_and_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        bundle,
        "build_receipt",
        lambda _paths_value: {"created_utc": "2026-08-24T00:00:00+00:00"},
    )
    calls = 0

    def fail_first_directory_fsync(_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("first directory fsync fault")

    monkeypatch.setattr(bundle, "_fsync_directory", fail_first_directory_fsync)
    with pytest.raises(OSError, match="first directory fsync fault"):
        bundle.seal(paths)
    assert calls == 2
    assert not paths.destination.exists()
    assert not list(paths.destination.parent.glob(f".{paths.destination.name}.*.tmp"))


def test_first_temp_unlink_failure_removes_link_and_retries_temp_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        bundle,
        "build_receipt",
        lambda _paths_value: {"created_utc": "2026-08-24T00:00:00+00:00"},
    )
    direct_unlink = Path.unlink
    failed = False

    def fail_first_temp_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal failed
        is_private_temp = (
            path.parent == paths.destination.parent
            and path.name.startswith(f".{paths.destination.name}.")
            and path.name.endswith(".tmp")
        )
        if is_private_temp and not failed:
            failed = True
            raise OSError("first temp unlink fault")
        direct_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_temp_unlink)
    with pytest.raises(OSError, match="first temp unlink fault"):
        bundle.seal(paths)
    assert failed is True
    assert not paths.destination.exists()
    assert not list(paths.destination.parent.glob(f".{paths.destination.name}.*.tmp"))


def test_link_created_before_control_loss_is_removed_by_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        bundle,
        "build_receipt",
        lambda _paths_value: {"created_utc": "2026-08-24T00:00:00+00:00"},
    )
    direct_link = bundle.os.link

    def create_link_then_lose_control(source: Path, destination: Path) -> None:
        direct_link(source, destination)
        raise KeyboardInterrupt("control lost immediately after link creation")

    monkeypatch.setattr(bundle.os, "link", create_link_then_lose_control)
    with pytest.raises(KeyboardInterrupt, match="control lost immediately after link creation"):
        bundle.seal(paths)
    assert not paths.destination.exists()
    assert not list(paths.destination.parent.glob(f".{paths.destination.name}.*.tmp"))
