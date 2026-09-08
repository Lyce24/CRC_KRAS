from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import aim1_tcga_surgen_full_pipeline_analysis as analysis  # noqa: E402


def _file_identity(path: Path, *, stored_path: str | None = None) -> dict[str, object]:
    return {
        "path": str(path if stored_path is None else stored_path),
        "sha256": analysis._sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _patient_frame(encoder: str, n: int = 60) -> pd.DataFrame:
    index = np.arange(n)
    labels = (index % 3 == 0).astype(int)
    logits = labels * 1.2 + np.sin(index / 3) - 0.4
    frame = pd.DataFrame(
        {
            "analysis_family": "source_restricted_tcga_surgen_oof",
            "encoder": encoder,
            "dataset": "tcga_surgen_source",
            "patient_id": [f"P{item:03d}" for item in index],
            "label": labels,
            "cohort": np.where(index < n // 2, "TCGA", "SurGen"),
            "subcohort": np.where(index < n // 2, "TCGA-COAD", "SR386"),
            "specimen_role": "primary",
            "k_fold": index % 5,
            "msi_dmmr": np.where(index % 5 == 0, "MSI/dMMR", "MSS/pMMR"),
            "braf": np.where(index % 7 == 0, "mutant", "wild_type"),
            "tumor_site_group": np.where(index % 4 == 0, "Rectum", "Colon"),
            "site_class": np.where(index % 4 == 0, "Rectum", "Colon"),
            "stage_group_major": np.where(index % 6 == 0, "IV", "II"),
            "stage_group_major_filled": np.where(index % 6 == 0, "IV", "II"),
            "stage_class": np.where(index % 6 == 0, "III-IV", "I-II"),
            "sidedness": np.select(
                [index % 5 == 0, index % 5 == 1],
                ["right", "transverse"],
                default="left",
            ),
            "age_at_diagnosis": 40 + index,
            "sex": np.where(index % 2 == 0, "female", "male"),
            "n_slides": 1,
            "slide_roster_sha256": [f"{item:064x}" for item in index],
            "mean_logit_5seed": logits,
            "source_platt_probability": 1 / (1 + np.exp(-logits)),
            "exclude_neoadjuvant": False,
            "exclude_ambiguous_crc15": False,
        }
    )
    for seed in analysis.SEEDS:
        frame[f"logit_seed{seed}"] = logits + (seed - 44) * 0.01
    return frame


def _metric_block() -> dict[str, object]:
    return {
        "patients": 20,
        "mutant": 8,
        "auroc": 0.7,
        "auroc_ci95": [0.6, 0.8],
        "auprc": 0.65,
        "auprc_ci95": [0.5, 0.75],
    }


def _exact_e1d_source_frame(encoder: str) -> pd.DataFrame:
    n = analysis.SOURCE_PATIENTS
    index = np.arange(n)
    labels = np.zeros(n, dtype=int)
    labels[: analysis.SOURCE_MUTANT] = 1
    derived_known = np.zeros(n, dtype=bool)
    derived_known[:422] = True
    derived_known[analysis.SOURCE_MUTANT : analysis.SOURCE_MUTANT + 638] = True
    logits = labels * 0.8 + np.sin(index / 19) + (0.03 if encoder == "virchow2_cls" else 0)
    return pd.DataFrame(
        {
            "patient_id": [f"P{item:04d}" for item in index],
            "label": labels,
            "cohort": np.where(index % 3 == 0, "TCGA", "SurGen"),
            "subcohort": np.select(
                [index % 4 == 0, index % 4 == 1, index % 4 == 2],
                ["TCGA-COAD", "TCGA-READ", "SR386"],
                default="SR1482",
            ),
            "specimen_role": "primary",
            "k_fold": index % 5,
            "mean_logit_5seed": logits,
            "tumor_site_group": np.where(index % 4 == 0, "Rectum", "Colon"),
            "site_class": np.where(index % 4 == 0, "Rectum", "Colon"),
            "stage_class": np.where(derived_known, "I-II", "unknown"),
            "stage_group_major_filled": np.where(derived_known, "II", pd.NA),
            "age_at_diagnosis": 40 + index % 45,
            "sex": np.where(index % 2 == 0, "female", "male"),
        }
    )


def _clinical_metric_block() -> dict[str, object]:
    return {
        **_metric_block(),
        "brier": 0.2,
        "brier_ci95": [0.15, 0.25],
        "log_loss": 0.6,
        "log_loss_ci95": [0.5, 0.7],
        "cal_intercept": 0.0,
        "cal_slope": 1.0,
    }


def _binding_fixtures() -> tuple[dict[str, object], ...]:
    encoders = {encoder: _metric_block() for encoder in analysis.ENCODERS}
    canonical = {
        key: encoders
        for key in ("pooled_all_primary", "tcga", "sr386", "sr1482", "surgen", "tcga_surgen")
    }
    source = {
        key: encoders
        for key in ("pooled_source", "tcga", "sr386", "sr1482", "surgen", "tcga_surgen")
    }
    targets = {key: {"encoders": encoders} for key in analysis.TARGET_ORDER}
    e1a = {
        "encoders": {
            encoder: {
                key: {
                    **_metric_block(),
                    "delta_A_minus_subset_auroc": 0.01,
                    "delta_A_minus_subset_auroc_ci95": [-0.02, 0.04],
                    "delta_A_minus_subset_auprc": 0.02,
                    "delta_A_minus_subset_auprc_ci95": [-0.01, 0.05],
                }
                for key in analysis.E1A_SETS
            }
            for encoder in analysis.ENCODERS
        }
    }
    model = _clinical_metric_block()
    e1d_populations: dict[str, object] = {}
    for population in ("A_all_primary", "G_stage_known_derived"):
        encoder_blocks: dict[str, object] = {}
        for encoder in analysis.ENCODERS:
            comparisons = {}
            for name in (
                "wsi_minus_clinical",
                "fusion_minus_clinical",
                "fusion_minus_wsi",
            ):
                comparisons[name] = {
                    metric: {
                        "left": float(model[metric]),
                        "right": float(model[metric]) - 0.01,
                        "delta": 0.01,
                        "delta_ci95": [-0.01, 0.03],
                        "lower_is_better": metric in {"brier", "log_loss"},
                    }
                    for metric in ("auroc", "auprc", "brier", "log_loss")
                }
            encoder_blocks[encoder] = {
                "wsi": dict(model),
                "fusion": dict(model),
                "incremental_contrasts": comparisons,
            }
        e1d_populations[population] = {
            "population": {"patients": 20, "mutant": 8},
            "clinical": dict(model),
            "encoders": encoder_blocks,
        }
    e1d = {"populations": e1d_populations}
    rih = {
        "encoders": {
            encoder: {
                "primary": {**_metric_block(), "patients": 145, "mutant": 65},
                "metastatic": {**_metric_block(), "patients": 77, "mutant": 33},
                "delta_metastatic_minus_primary": {
                    "auroc": 0.01,
                    "auroc_ci95": [-0.05, 0.07],
                    "auprc": 0.02,
                    "auprc_ci95": [-0.04, 0.08],
                },
            }
            for encoder in analysis.ENCODERS
        }
    }
    cpht = {
        "populations": {
            population: encoders
            for population in ("all_40", "exclude_neoadjuvant", "exclude_ambiguous_crc15")
        }
    }
    source_contrasts = {
        population: {
            "patients": 20,
            "mutant": 8,
            "metrics": {
                metric: {
                    "univ1": 0.7,
                    "univ1_ci95": [0.6, 0.8],
                    "virchow2_cls": 0.71,
                    "virchow2_cls_ci95": [0.61, 0.81],
                    "delta_virchow2_cls_minus_univ1": 0.01,
                    "delta_ci95": [-0.02, 0.04],
                }
                for metric in ("auroc", "auprc")
            },
        }
        for population in ("pooled_source", "tcga", "sr386", "sr1482", "surgen", "tcga_surgen")
    }
    e1a_s = {
        "references": {
            reference: {
                "encoders": {
                    encoder: {
                        "patients_A": 20,
                        "mutant_A": 8,
                        "patients_D": 15,
                        "mutant_D": 6,
                        "standardized_A_auroc": 0.7,
                        "standardized_A_auroc_ci95": [0.6, 0.8],
                        "standardized_D_auroc": 0.72,
                        "standardized_D_auroc_ci95": [0.62, 0.82],
                        "delta_D_minus_A_auroc": 0.02,
                        "delta_D_minus_A_auroc_ci95": [-0.02, 0.06],
                    }
                    for encoder in analysis.ENCODERS
                }
            }
            for reference in ("A_all_primary", "A_complete")
        }
    }
    why_d = {
        "random_restriction_draws": 10_000,
        "patient_bootstrap_draws": 10_000,
        "encoders": {
            encoder: {
                "status": "DERIVED",
                "A_random_restriction_negative_control": {"observed_delta": 0.01},
                "B_pairwise_auc_decomposition": {"D+ vs D-": {"auc": 0.7}},
                "C_molecular_score_distributions": {"KRAS-WT": {}},
                "D_adjusted_molecular_association": {"BRAF_mut": {"beta": 0.1}},
            }
            for encoder in analysis.ENCODERS
        },
    }
    return canonical, source, targets, e1a, e1d, rih, cpht, source_contrasts, e1a_s, why_d


def test_exact_scoring_factorization_and_target_roster(tmp_path: Path) -> None:
    jobs = analysis._score_jobs(tmp_path)
    assert analysis.TARGET_ORDER == (
        "cptac_primary",
        "rih_primary",
        "rih_metastatic",
        "sr1482_metastatic",
        "orion_cpht",
    )
    assert len(jobs) == 50
    assert sum(int(job["expected_rows"]) for job in jobs) == 4_790
    assert len({(job["encoder"], job["target"], job["seed"]) for job in jobs}) == 50
    assert {job["fit_count"] for job in jobs} == {0}
    assert {job["contains_target_outcomes"] for job in jobs} == {False}


def test_training_terminal_contract_is_exact() -> None:
    assert analysis.EXPECTED_FIT_ACCOUNTING == {
        "adopted_oof_fits": 25,
        "new_oof_fits": 25,
        "new_refits": 10,
        "new_fits": 35,
        "physical_new_fits": 35,
        "operational_lineage_fits": 60,
        "recovery_new_fits": 0,
        "hidden_fits": 0,
    }
    assert analysis.EXPECTED_EXECUTION_ACCOUNTING == {
        "job_count": 10,
        "attempts_per_job": 1,
        "total_attempts": 10,
        "retries": 0,
    }
    assert Path("recovery_v1/receipts/training_complete_recovered.json") == (
        analysis.TRAINING_RECOVERY_TERMINAL
    )
    assert analysis.TRAINING_RECOVERY_CONTROLLER_SHA256 == (
        "d47cbd54b584fb3d0dd5160ae0bdd1fe2aca79b53207f8c1b63e366957b66ae5"
    )
    assert analysis.TRAINING_RECOVERY_TEST_SHA256 == (
        "2cf182aab9cb8a92d4416402024251a7b450adc11e0be7da4f24d0bfac6c2f24"
    )
    expected_training_top_level = {
        "schema_version",
        "recovery",
        "base_campaign",
        "status",
        "created_utc",
        "base_contract",
        "base_preflight",
        "erratum_contract",
        "scheduler_recovery",
        "validator_adjudication",
        "recovery_implementation",
        "source_population",
        "seeds",
        "encoders",
        "job_count",
        "fit_accounting",
        "execution_accounting",
        "concurrency",
        "adopted_univ1_oof_chains",
        "stock_univ1_job_receipts",
        "recovered_virchow2_job_receipts",
        "new_chains",
        "new_job_receipts",
        "raw_artifact_hash_census",
        "certification_boundary",
        "stock_receipts_fabricated",
    }
    assert expected_training_top_level == analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "base_contract" in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "base_preflight" in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "erratum_contract" in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "scheduler_recovery" in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "validator_adjudication" in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "recovery_implementation" in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "stock_univ1_job_receipts" in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "recovered_virchow2_job_receipts" in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "raw_artifact_hash_census" in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "stock_receipts_fabricated" in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "adopted_univ1_oof_chains" in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "new_job_receipts" in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "contract" not in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "preflight" not in analysis.EXPECTED_TRAINING_TOP_LEVEL
    assert "scheduler" not in analysis.EXPECTED_TRAINING_TOP_LEVEL


def test_training_recovery_source_pins_match_frozen_bytes() -> None:
    assert analysis._sha256(analysis.TRAINING_RECOVERY_CONTROLLER) == (
        analysis.TRAINING_RECOVERY_CONTROLLER_SHA256
    )
    assert analysis.TRAINING_RECOVERY_CONTROLLER.stat().st_size == (
        analysis.TRAINING_RECOVERY_CONTROLLER_SIZE
    )
    assert analysis._sha256(analysis.TRAINING_RECOVERY_TEST) == (
        analysis.TRAINING_RECOVERY_TEST_SHA256
    )
    assert analysis.TRAINING_RECOVERY_TEST.stat().st_size == (analysis.TRAINING_RECOVERY_TEST_SIZE)


@pytest.mark.parametrize(
    "relative",
    (
        "receipts/training_complete.json",
        "receipts/scheduler.json",
        "receipts/jobs/virchow2_full/seed42.json",
        "requests/virchow2_full/seed42.failure.json",
    ),
)
def test_training_gate_refuses_fabricated_stock_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    monkeypatch.setattr(analysis, "_safe_campaign_root", lambda _: tmp_path)
    forbidden = tmp_path / relative
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_text("{}\n", encoding="utf-8")
    with pytest.raises(analysis.GovernanceError, match="requires absent stock"):
        analysis._validate_training_bundle(tmp_path, deep=False)


def test_training_gate_calls_exact_recovery_replay_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analysis, "_safe_campaign_root", lambda _: tmp_path)
    terminal = tmp_path / analysis.TRAINING_RECOVERY_TERMINAL
    terminal.parent.mkdir(parents=True, exist_ok=True)
    terminal.write_text("{}\n", encoding="utf-8")
    called: dict[str, object] = {}

    def replay(root: Path, *, deep_pack: bool) -> dict[str, object]:
        called.update(root=root, deep_pack=deep_pack)
        return {}

    monkeypatch.setattr(
        analysis.training_recovery,
        "validate_recovered_terminal",
        replay,
    )
    with pytest.raises(analysis.GovernanceError, match="top-level field roster"):
        analysis._validate_training_bundle(tmp_path, deep=False)
    assert called == {"root": tmp_path, "deep_pack": False}


def test_status_uses_recovered_terminal_not_stock_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analysis, "_safe_campaign_root", lambda _: tmp_path)
    stock = tmp_path / "receipts/training_complete.json"
    stock.parent.mkdir(parents=True, exist_ok=True)
    stock.write_text("{}\n", encoding="utf-8")
    assert analysis.status(tmp_path)["state"] == "INVALID_FORBIDDEN_STOCK_RECEIPTS"
    stock.unlink()
    recovered = tmp_path / analysis.TRAINING_RECOVERY_TERMINAL
    recovered.parent.mkdir(parents=True, exist_ok=True)
    recovered.write_text("{}\n", encoding="utf-8")
    assert analysis.status(tmp_path)["state"] == "READY_TO_PREPARE"


def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(analysis.GovernanceError, match="Duplicate JSON key"):
        analysis._read_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(analysis.GovernanceError, match="Non-finite JSON"):
        analysis._read_json(nonfinite)


def test_strict_json_tree_resolves_native_completion_paths_from_each_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    fold = run / "fold_0"
    fold.mkdir(parents=True)
    config = fold / "config.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")
    metrics = fold / "fold_metrics.json"
    metrics.write_text('{"status": "complete"}\n', encoding="utf-8")
    fold_completion = fold / "completion.json"
    fold_completion.write_text(
        json.dumps(
            {
                "artifacts": {
                    "config": _file_identity(config, stored_path="config.yaml"),
                    "metrics": _file_identity(metrics, stored_path="fold_metrics.json"),
                },
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    cv_summary = run / "cv_summary.json"
    cv_summary.write_text('{"status": "complete"}\n', encoding="utf-8")
    completion = run / "training_completion.json"
    completion.write_text(
        json.dumps(
            {
                "artifacts": {
                    "cv_summary": _file_identity(cv_summary, stored_path="cv_summary.json")
                },
                "fold_completions": [
                    _file_identity(fold_completion, stored_path="fold_0/completion.json")
                ],
            }
        ),
        encoding="utf-8",
    )
    decoy = tmp_path / "cwd"
    decoy.mkdir()
    monkeypatch.chdir(decoy)
    assert analysis._strict_json_tree(completion)["fold_completions"][0]["path"] == (
        "fold_0/completion.json"
    )


def test_strict_json_tree_uses_exact_census_root_for_root_relative_records(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    contract = campaign / "contract.json"
    contract.write_text('{"status": "sealed"}\n', encoding="utf-8")
    receipt_dir = tmp_path / "recovery" / "receipts"
    receipt_dir.mkdir(parents=True)
    adjudication = receipt_dir / "validator_adjudication.json"
    record = _file_identity(contract, stored_path="contract.json")
    adjudication.write_text(
        json.dumps(
            {
                "raw_artifact_hash_census": {
                    "root": str(campaign),
                    "excluded_prefix": "recovery_v1/",
                    "artifact_count": 1,
                    "total_size_bytes": contract.stat().st_size,
                    "tree_sha256": "0" * 64,
                    "artifacts": [record],
                }
            }
        ),
        encoding="utf-8",
    )
    assert analysis._strict_json_tree(adjudication)["raw_artifact_hash_census"]["artifacts"] == [
        record
    ]


def test_strict_json_tree_keeps_absolute_identity_paths_unchanged(tmp_path: Path) -> None:
    external = tmp_path / "external.json"
    external.write_text('{"status": "absolute"}\n', encoding="utf-8")
    owner = tmp_path / "owner"
    owner.mkdir()
    root = owner / "root.json"
    root.write_text(json.dumps({"external": _file_identity(external)}), encoding="utf-8")
    assert analysis._strict_json_tree(root)["external"]["path"] == str(external)


def test_strict_json_tree_rejects_non_normalized_absolute_identity_path(
    tmp_path: Path,
) -> None:
    owner = tmp_path / "owner"
    owner.mkdir()
    child = tmp_path / "outside.json"
    child.write_text('{"status": "absolute"}\n', encoding="utf-8")
    non_normalized = f"{owner}/../outside.json"
    root = owner / "root.json"
    root.write_text(
        json.dumps({"child": _file_identity(child, stored_path=non_normalized)}),
        encoding="utf-8",
    )
    with pytest.raises(analysis.GovernanceError, match="absolute artifact path.*non-normalized"):
        analysis._strict_json_tree(root)


def test_strict_json_tree_rejects_absolute_directory_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    child = outside / "child.json"
    child.write_text('{"status": "absolute"}\n', encoding="utf-8")
    owner = tmp_path / "owner"
    owner.mkdir()
    alias = owner / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    root = owner / "root.json"
    root.write_text(
        json.dumps({"child": _file_identity(child, stored_path=str(alias / "child.json"))}),
        encoding="utf-8",
    )
    with pytest.raises(analysis.GovernanceError, match="absolute artifact path.*symlinked"):
        analysis._strict_json_tree(root)


def test_strict_json_tree_rejects_relative_traversal_even_with_matching_bytes(
    tmp_path: Path,
) -> None:
    owner = tmp_path / "owner"
    owner.mkdir()
    escaped = tmp_path / "escaped.json"
    escaped.write_text('{"status": "outside"}\n', encoding="utf-8")
    root = owner / "root.json"
    root.write_text(
        json.dumps({"escaped": _file_identity(escaped, stored_path="../escaped.json")}),
        encoding="utf-8",
    )
    with pytest.raises(analysis.GovernanceError, match="non-normalized/traversing"):
        analysis._strict_json_tree(root)


def test_strict_json_tree_rejects_symlinked_relative_path_escape(tmp_path: Path) -> None:
    owner = tmp_path / "owner"
    owner.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "child.json"
    external.write_text('{"status": "outside"}\n', encoding="utf-8")
    (owner / "alias").symlink_to(outside, target_is_directory=True)
    root = owner / "root.json"
    root.write_text(
        json.dumps({"child": _file_identity(external, stored_path="alias/child.json")}),
        encoding="utf-8",
    )
    with pytest.raises(analysis.GovernanceError, match="escapes|symlink"):
        analysis._strict_json_tree(root)


def test_strict_json_tree_rejects_ambiguous_census_base(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    child = campaign / "child.json"
    child.write_text('{"status": "sealed"}\n', encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "census_like": {
                    "root": str(campaign),
                    "artifact_count": 1,
                    "artifacts": [_file_identity(child, stored_path="child.json")],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(analysis.GovernanceError, match="ambiguous artifact-identity base"):
        analysis._strict_json_tree(receipt)


def test_strict_json_tree_rejects_malformed_census_artifact_record(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    child = campaign / "child.json"
    child.write_text('{"status": "sealed"}\n', encoding="utf-8")
    malformed = {**_file_identity(child, stored_path="child.json"), "unexpected": True}
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "census": {
                    "root": str(campaign),
                    "excluded_prefix": "recovery_v1/",
                    "artifact_count": 1,
                    "total_size_bytes": child.stat().st_size,
                    "tree_sha256": "0" * 64,
                    "artifacts": [malformed],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(analysis.GovernanceError, match="malformed raw artifact census records"):
        analysis._strict_json_tree(receipt)


def test_strict_json_tree_rejects_absolute_census_record_outside_root(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"status": "outside"}\n', encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "census": {
                    "root": str(campaign),
                    "excluded_prefix": "recovery_v1/",
                    "artifact_count": 1,
                    "total_size_bytes": outside.stat().st_size,
                    "tree_sha256": "0" * 64,
                    "artifacts": [_file_identity(outside)],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(analysis.GovernanceError, match="census paths must be relative"):
        analysis._strict_json_tree(receipt)


def test_strict_json_tree_census_does_not_fall_back_to_receipt_parent(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    campaign_child = campaign / "child.json"
    campaign_child.write_text('{"value": "aaa"}\n', encoding="utf-8")
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    receipt_child = receipt_dir / "child.json"
    receipt_child.write_text('{"value": "bbb"}\n', encoding="utf-8")
    receipt = receipt_dir / "adjudication.json"
    receipt.write_text(
        json.dumps(
            {
                "census": {
                    "root": str(campaign),
                    "excluded_prefix": "recovery_v1/",
                    "artifact_count": 1,
                    "total_size_bytes": campaign_child.stat().st_size,
                    "tree_sha256": "0" * 64,
                    "artifacts": [_file_identity(receipt_child, stored_path="child.json")],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(analysis.GovernanceError, match="SHA drifted"):
        analysis._strict_json_tree(receipt)


def test_strict_json_tree_does_not_fall_back_to_cwd_for_relative_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = tmp_path / "owner"
    owner.mkdir()
    owner_child = owner / "child.json"
    owner_child.write_text('{"value": "aaa"}\n', encoding="utf-8")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    cwd_child = cwd / "child.json"
    cwd_child.write_text('{"value": "bbb"}\n', encoding="utf-8")
    root = owner / "root.json"
    root.write_text(
        json.dumps({"child": _file_identity(cwd_child, stored_path="child.json")}),
        encoding="utf-8",
    )
    monkeypatch.chdir(cwd)
    with pytest.raises(analysis.GovernanceError, match="SHA drifted"):
        analysis._strict_json_tree(root)


def test_strict_json_tree_cycle_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps({"child": {"path": "second.json", "sha256": "0" * 64, "size_bytes": 1}}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"child": {"path": "first.json", "sha256": "0" * 64, "size_bytes": 1}}),
        encoding="utf-8",
    )
    validated: list[Path] = []
    monkeypatch.setattr(
        analysis,
        "_validate_identity_at_path",
        lambda _identity, path: validated.append(Path(path)),
    )
    analysis._strict_json_tree(first)
    assert validated == [second, first]


LIVE_RECOVERED_TERMINAL = analysis.DEFAULT_CAMPAIGN_ROOT / analysis.TRAINING_RECOVERY_TERMINAL


@pytest.mark.skipif(
    not LIVE_RECOVERED_TERMINAL.is_file(), reason="published recovery terminal is unavailable"
)
def test_live_recovered_terminal_strict_json_tree_regression() -> None:
    terminal = analysis._strict_json_tree(LIVE_RECOVERED_TERMINAL)
    assert terminal["status"] == "complete_and_certified_via_bounded_erratum"
    assert terminal["raw_artifact_hash_census"]["artifact_count"] == 441


def test_write_once_is_idempotent_and_refuses_drift(tmp_path: Path) -> None:
    path = tmp_path / "immutable.bin"
    analysis._write_bytes_once(path, b"alpha")
    analysis._write_bytes_once(path, b"alpha")
    with pytest.raises(analysis.GovernanceError, match="overwrite"):
        analysis._write_bytes_once(path, b"beta")


def test_label_blind_allowlist_and_census() -> None:
    spec = replace(analysis.TARGETS["cptac_primary"], slides=2, patients=2)
    good = pd.DataFrame(
        {
            "slide_id": ["s1", "s2"],
            "patient_id": ["p1", "p2"],
            "cohort": ["CPTAC", "CPTAC"],
        }
    )
    assert len(analysis._blind_frame(good, spec=spec, context="test")) == 2
    with pytest.raises(analysis.GovernanceError, match="label-blind schema violation"):
        analysis._blind_frame(good.assign(target_label=[0, 1]), spec=spec, context="leak")
    with pytest.raises(analysis.GovernanceError, match="target census drifted"):
        analysis._blind_frame(good.iloc[:1], spec=spec, context="short")


def test_score_schema_is_native_logit_only() -> None:
    manifest = pd.DataFrame({"slide_id": ["s1", "s2"], "patient_id": ["p1", "p2"]})
    scores = pd.DataFrame(
        {
            "slide_id": ["s1", "s2"],
            "seed": [42, 42],
            "fold": [0, 0],
            "logit": [-1.0, 1.0],
        }
    )
    analysis._validate_score_frame(scores, manifest, seed=42, context="test")
    with pytest.raises(analysis.GovernanceError, match="score schema drifted"):
        analysis._validate_score_frame(
            scores.assign(probability=[0.2, 0.8]), manifest, seed=42, context="test"
        )


def test_stratified_bootstrap_is_deterministic_and_cell_preserving() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1])
    strata = np.array(["A", "A", "B", "B", "B", "B"])
    first = analysis.stratified_bootstrap_indices(labels, strata, n_bootstrap=20, stream="unit")
    second = analysis.stratified_bootstrap_indices(labels, strata, n_bootstrap=20, stream="unit")
    assert np.array_equal(first, second)
    original = sorted(zip(strata, labels, strict=True))
    for draw in first:
        assert sorted(zip(strata[draw], labels[draw], strict=True)) == original


def test_performance_panel_shares_encoder_draws() -> None:
    uni = _patient_frame("univ1")
    v2 = _patient_frame("virchow2_cls")
    arrays: dict[str, np.ndarray] = {}
    panel = analysis._performance_panel(
        {"univ1": uni, "virchow2_cls": v2},
        {"all": np.ones(len(uni), dtype=bool)},
        prefix="unit",
        n_bootstrap=25,
        arrays=arrays,
        include_probability_metrics=True,
    )
    assert panel["all"]["univ1"]["shared_encoder_bootstrap_indices"] is True
    assert np.array_equal(
        arrays["unit__all__univ1__auroc"],
        arrays["unit__all__virchow2_cls__auroc"],
    )
    assert all(value.shape == (25,) for value in arrays.values())


def test_duplicate_source_population_alias_reuses_draws() -> None:
    uni = _patient_frame("univ1")
    v2 = _patient_frame("virchow2_cls")
    arrays: dict[str, np.ndarray] = {}
    mask = np.ones(len(uni), dtype=bool)
    analysis._performance_panel(
        {"univ1": uni, "virchow2_cls": v2},
        {"pooled_source": mask, "tcga_surgen": mask.copy()},
        prefix="source",
        n_bootstrap=20,
        arrays=arrays,
        include_probability_metrics=True,
    )
    assert np.array_equal(
        arrays["source__pooled_source__univ1__auroc"],
        arrays["source__tcga_surgen__univ1__auroc"],
    )


def test_source_encoder_contrast_subtracts_shared_patient_draws() -> None:
    populations = ("pooled_source", "tcga", "sr386", "sr1482", "surgen", "tcga_surgen")
    performance = {
        population: {
            "univ1": _metric_block(),
            "virchow2_cls": {
                **_metric_block(),
                "auroc": 0.72,
                "auprc": 0.68,
            },
        }
        for population in populations
    }
    arrays: dict[str, np.ndarray] = {}
    for population in populations:
        for metric in ("auroc", "auprc"):
            arrays[f"aim1_source_restricted_e0__{population}__univ1__{metric}"] = np.array(
                [0.6, 0.7, 0.8], dtype=np.float64
            )
            arrays[f"aim1_source_restricted_e0__{population}__virchow2_cls__{metric}"] = np.array(
                [0.65, 0.72, 0.78], dtype=np.float64
            )
    result = analysis._source_encoder_contrast_panel(performance, arrays=arrays)
    assert np.allclose(
        arrays["aim1_source_restricted_e0__pooled_source__encoder_delta_v2_minus_uni__auroc"],
        [0.05, 0.02, -0.02],
    )
    assert result["pooled_source"]["shared_patient_bootstrap"] is True


def test_patient_native_logit_aggregation_uses_all_five_seeds(tmp_path: Path) -> None:
    rows = []
    for item in range(10):
        rows.append(
            {
                "slide_id": f"s{item}",
                "patient_id": f"p{item}",
                "target_label": item % 2,
                "cohort": "TCGA" if item < 5 else "SurGen",
                "subcohort": "TCGA-COAD" if item < 5 else "SR386",
                "specimen_role": "primary",
                "k_fold": item % 5,
                "msi_dmmr": "MSS/pMMR",
                "braf": "wild_type",
                "tumor_site_group": "Colon",
                "site_class": "Colon",
                "stage_group_major": "II",
                "stage_class": "I-II",
                "age_at_diagnosis": 50,
                "sex": "female",
            }
        )
    manifest = pd.DataFrame(rows)
    paths: dict[int, Path] = {}
    for seed in analysis.SEEDS:
        frame = pd.DataFrame(
            {
                "slide_id": manifest["slide_id"],
                "label": manifest["target_label"],
                "prob_1": 0.5,
                "logit": np.arange(10, dtype=float) + seed,
                "fold": manifest["k_fold"],
            }
        )
        path = tmp_path / f"seed{seed}.parquet"
        frame.to_parquet(path, index=False)
        paths[seed] = path
    patient = analysis._aggregate_oof_patients(
        manifest,
        paths,
        encoder="univ1",
        analysis_family="unit",
        dataset="unit",
    )
    assert len(patient) == 10
    assert np.allclose(
        patient["mean_logit_5seed"],
        patient[[f"logit_seed{seed}" for seed in analysis.SEEDS]].mean(axis=1),
    )


def test_source_platt_matches_unregularized_logistic_map() -> None:
    frame = _patient_frame("univ1", n=90)
    calibration = analysis._fit_source_platt(frame)
    probability = analysis._apply_platt(frame["mean_logit_5seed"], calibration)
    assert calibration["slope"] > 0
    assert np.isfinite(probability).all()
    assert ((probability > 0) & (probability < 1)).all()


def test_outcomes_are_not_opened_when_seal_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = False

    def fail_seal(_root: Path) -> dict[str, object]:
        raise analysis.GovernanceError("seal missing")

    def spy_read(*_args: object, **_kwargs: object) -> pd.DataFrame:
        nonlocal opened
        opened = True
        return pd.DataFrame()

    monkeypatch.setattr(analysis, "verify_inference_seal", fail_seal)
    monkeypatch.setattr(pd, "read_csv", spy_read)
    with pytest.raises(analysis.GovernanceError, match="seal missing"):
        analysis._open_target_outcomes_after_seal(tmp_path)
    assert opened is False


def test_analyze_dry_run_does_not_open_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        analysis,
        "verify_inference_seal",
        lambda _root: {"status": "sealed_before_outcome_join"},
    )
    monkeypatch.setattr(
        analysis,
        "_open_target_outcomes_after_seal",
        lambda _root: pytest.fail("dry analysis opened target outcomes"),
    )
    result = analysis.analyze(tmp_path, apply=False)
    assert result["status"] == "dry_run_ready_after_inference_seal"
    assert result["target_outcomes_opened"] is False


def test_score_dry_run_is_50_jobs_without_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analysis, "_load_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        analysis,
        "_read_json",
        lambda _path: {"status": "ready_for_label_blind_inference"},
    )
    monkeypatch.setattr(
        analysis,
        "_run_score_subprocess",
        lambda _command: pytest.fail("dry score launched a worker"),
    )
    result = analysis.score(tmp_path, apply=False, max_workers=6, device="cuda", num_workers=0)
    assert result["jobs"] == 50
    assert result["slide_rows"] == 4_790
    assert len(result["commands"]) == 50


def test_inference_seal_payload_is_exactly_50_files_and_4790_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis._write_json_once(analysis.contract_path(tmp_path), {"contract": True})
    analysis._write_json_once(analysis.preflight_path(tmp_path), {"preflight": True})
    analysis._write_json_once(
        analysis.scoring_receipt_path(tmp_path),
        {
            "configured_max_workers": 6,
            "max_observed_parallel_workers": 4,
        },
    )
    for job in analysis._score_jobs(tmp_path):
        path = analysis.score_path(
            tmp_path, str(job["encoder"]), str(job["target"]), int(job["seed"])
        )
        receipt = analysis.score_receipt_path(
            tmp_path, str(job["encoder"]), str(job["target"]), int(job["seed"])
        )
        analysis._write_bytes_once(path, str(job["job_id"]).encode())
        analysis._write_json_once(receipt, {"job_id": job["job_id"]})

    def cached(_root: Path, _encoder: str, target: str, _seed: int) -> pd.DataFrame:
        return pd.DataFrame(index=range(analysis.TARGETS[target].slides))

    monkeypatch.setattr(analysis, "_validate_cached_score", cached)
    payload = analysis._seal_payload(tmp_path)
    assert payload["status"] == "sealed_before_outcome_join"
    assert payload["target_outcomes_present"] is False
    assert payload["score_artifact_count"] == 50
    assert payload["score_slide_rows"] == 4_790
    assert len(payload["score_artifacts"]) == 50


def test_npz_contract_is_pickle_free_and_tamper_evident(tmp_path: Path) -> None:
    path = tmp_path / "draws.npz"
    arrays = {"a": np.arange(10, dtype=np.float64), "b": np.ones(10, dtype=np.float64)}
    analysis._write_bytes_once(path, analysis._npz_bytes(arrays))
    analysis._validate_npz(path, ["a", "b"], length=10)
    with pytest.raises(analysis.GovernanceError, match="key roster"):
        analysis._validate_npz(path, ["a"], length=10)


def test_report_claim_rows_are_one_to_one_and_boundary_complete() -> None:
    canonical, source, targets, e1a, e1d, rih, cpht, *_ = _binding_fixtures()
    rows = analysis._report_claim_rows(canonical, source, targets, e1a, e1d, rih, cpht)
    assert len(rows) == analysis.REPORT_PERFORMANCE_CLAIM_COUNT == 78
    assert len({row["row_id"] for row in rows}) == 78
    assert {row["row_id"] for row in rows} == analysis._expected_performance_claim_ids()
    assert {
        "aim1.canonical_e0.pooled_all_primary.univ1",
        "aim1.source_restricted_e0.tcga_surgen.virchow2_cls",
        "aim1.e1a.K_transverse.univ1",
        "aim1.e1d.G_stage_known_derived.virchow2_cls.fusion",
        "aim2.rih_disjoint_role.metastatic.univ1",
        "aim2.cpht_raw.exclude_neoadjuvant.virchow2_cls",
        "aim2.orion_cpht.univ1",
    } <= {row["row_id"] for row in rows}


def test_report_contrast_rows_are_complete_and_directional() -> None:
    (
        _canonical,
        _source,
        _targets,
        e1a,
        e1d,
        rih,
        _cpht,
        source_contrasts,
        e1a_s,
        _why_d,
    ) = _binding_fixtures()
    rows = analysis._report_contrast_rows(source_contrasts, e1a, e1a_s, e1d, rih)
    assert len(rows) == analysis.REPORT_CONTRAST_CLAIM_COUNT == 48
    assert {row["row_id"] for row in rows} == analysis._expected_contrast_claim_ids()
    by_id = {row["row_id"]: row for row in rows}
    source = by_id["aim1.source_restricted_e0_encoder_delta.pooled_source.virchow2_cls_minus_univ1"]
    assert source["reference"]["name"] == "pooled_source:univ1"
    assert source["comparison"]["name"] == "pooled_source:virchow2_cls"
    assert source["metrics"]["auroc"]["delta_comparison_minus_reference"] == 0.01
    e1a_row = by_id["aim1.e1a_delta_A_minus_set.K_transverse.univ1"]
    assert e1a_row["reference"]["name"] == "K_transverse"
    assert e1a_row["comparison"]["name"] == "A_all_primary"


def test_why_d_evidence_records_embed_human_renderable_results() -> None:
    *_, why_d = _binding_fixtures()
    records = analysis._why_d_evidence_records(why_d)
    assert len(records) == analysis.WHY_D_EVIDENCE_RECORD_COUNT == 2
    assert {record["record_id"] for record in records} == {
        "aim1.why_d.univ1",
        "aim1.why_d.virchow2_cls",
    }
    assert all("observed_delta" in record["analysis_a_random_restriction"] for record in records)


def test_e1a_true_gap_masks_use_derived_stage_and_three_way_sidedness() -> None:
    frame = pd.DataFrame(
        {
            "stage_group_major": [pd.NA, "IV", "II", pd.NA],
            "stage_group_major_filled": ["IV", "IV", "II", pd.NA],
            "sidedness": ["right", "left", "transverse", pd.NA],
        }
    )
    assert analysis._e1a_mask(frame, "G_stage_known_derived").tolist() == [True, True, True, False]
    assert analysis._e1a_mask(frame, "G_stage_known_frozen").tolist() == [False, True, True, False]
    assert analysis._e1a_mask(frame, "H_stage_iv").tolist() == [True, True, False, False]
    assert analysis._e1a_mask(frame, "I_right_proximal").tolist() == [True, False, False, False]
    assert analysis._e1a_mask(frame, "J_left_distal").tolist() == [False, True, False, False]
    assert analysis._e1a_mask(frame, "K_transverse").tolist() == [False, False, True, False]


def test_e1d_stage_known_is_fixed_subset_of_full_cross_fitted_scores() -> None:
    frames = {encoder: _exact_e1d_source_frame(encoder) for encoder in analysis.ENCODERS}
    arrays: dict[str, np.ndarray] = {}
    result = analysis._e1d_clinical(frames, n_bootstrap=8, arrays=arrays)
    assert result["stage_known_is_fixed_evaluation_subset_not_refit"] is True
    assert result["populations"]["A_all_primary"]["population"] == {
        "patients": 1_239,
        "mutant": 501,
        "stage_unknown_frozen": 179,
        "age_missing": 0,
    }
    assert result["populations"]["G_stage_known_derived"]["population"]["patients"] == 1_060
    assert result["populations"]["G_stage_known_derived"]["population"]["mutant"] == 422
    assert all(value.shape == (8,) and np.isfinite(value).all() for value in arrays.values())


def test_cpht_all40_is_exact_alias_and_only_exclusions_get_new_draws() -> None:
    index = np.arange(40)
    labels = (index < 15).astype(int)
    exclude_neoadjuvant = np.zeros(40, dtype=bool)
    exclude_neoadjuvant[[0, 1, 2, 3, 15, 16]] = True
    exclude_ambiguous = np.zeros(40, dtype=bool)
    exclude_ambiguous[17] = True
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    overall: dict[str, dict[str, object]] = {}
    for encoder in analysis.ENCODERS:
        score = labels * 0.8 + np.sin(index / 5) + (encoder == "virchow2_cls") * 0.02
        frames[encoder] = {
            "orion_cpht": pd.DataFrame(
                {
                    "patient_id": [f"O{item:02d}" for item in index],
                    "label": labels,
                    "cohort": "Orion",
                    "subcohort": "Orion-CRC",
                    "specimen_role": "primary",
                    "mean_logit_5seed": score,
                    "source_platt_probability": 1 / (1 + np.exp(-score)),
                    "exclude_neoadjuvant": exclude_neoadjuvant,
                    "exclude_ambiguous_crc15": exclude_ambiguous,
                    **{f"logit_seed{seed}": score + (seed - 44) * 0.001 for seed in analysis.SEEDS},
                }
            )
        }
        overall[encoder] = {**_metric_block(), "patients": 40, "mutant": 15}
    arrays: dict[str, np.ndarray] = {}
    result = analysis._orion_sensitivity_populations(
        frames,
        overall,
        n_bootstrap=8,
        arrays=arrays,
    )
    assert result["populations"]["all_40"] == overall
    assert all("all_40" not in key for key in arrays)
    assert result["populations"]["exclude_neoadjuvant"]["univ1"]["patients"] == 34
    assert result["populations"]["exclude_ambiguous_crc15"]["univ1"]["patients"] == 39


def test_analysis_inventory_is_exactly_five_governed_artifacts() -> None:
    assert analysis.ANALYSIS_FILES == (
        "contract.json",
        "patient_native_logits.parquet",
        "bootstrap_distributions.npz",
        "results.json",
        "analysis_completion_receipt.json",
    )


def test_status_waits_for_training_without_writes(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())
    observed = analysis.status(tmp_path)
    assert observed["state"] == "WAITING_FOR_TRAINING"
    assert list(tmp_path.iterdir()) == before


def test_json_payload_refuses_nan() -> None:
    with pytest.raises(analysis.GovernanceError, match="not finite"):
        analysis._json_bytes({"bad": float("nan")})


def test_default_campaign_root_and_no_production_launch() -> None:
    assert (
        Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim1_tcga_surgen_two_encoder_v1_20260824")
        == analysis.DEFAULT_CAMPAIGN_ROOT
    )


def test_strict_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    value = {"schema_version": 1, "status": "PASS"}
    analysis._write_json_once(path, value)
    assert analysis._read_json(path) == value
    assert json.loads(path.read_text()) == value
