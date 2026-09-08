from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import aim1_e0_five_seed_extension as extension  # noqa: E402


def test_public_job_inventory_is_exactly_two_overlay_seed_chains(tmp_path: Path) -> None:
    root = tmp_path / "final_v9_mil_5seed_expansion_v1_20260823/aim1_e0"
    jobs = extension.build_job_inventory(root)

    assert [job["seed"] for job in jobs] == [45, 46]
    assert [job["key"] for job in jobs] == ["aim1_e0_seed45", "aim1_e0_seed46"]
    assert all(job["scheduler_slots"] == 1 for job in jobs)
    assert all(job["component"] == "aim1_e0" for job in jobs)
    assert all(str(root.resolve()) in job["run_dir"] for job in jobs)
    assert not any(str(extension.DEFAULT_LEGACY_ROOT) in job["run_dir"] for job in jobs)

    for seed, job in zip(extension.NEW_SEEDS, jobs, strict=True):
        command = job["training_command"]
        assert f"splits.seed={seed}" in command
        assert f"training.seed={seed}" in command
        assert "splits=aim1_balanced" in command
        assert "training.dataset_max_instances=8192" in command
        assert "training.train_sampling_strategy=patient_natural" in command
        assert "training.sample_weight_column=null" in command
        assert "training.refit_epoch_rule=p75" not in command  # inherited from training=aim1
        assert f"train_dir={job['run_dir']}" in command
        assert command[-1] == "hydra.job.chdir=false"
        assert job["expected_training_fingerprint"] == extension.EXPECTED_NEW_FINGERPRINTS[seed]


def test_five_seed_resolver_never_places_new_runs_in_inherited_tree(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay"
    legacy = tmp_path / "legacy"

    for seed in extension.INHERITED_SEEDS:
        assert extension.resolve_run_dir(overlay, legacy, seed) == legacy / f"seed{seed}"
    for seed in extension.NEW_SEEDS:
        assert extension.resolve_run_dir(overlay, legacy, seed) == (
            overlay / f"train/1a_pb_cap8192/univ1/seed{seed}"
        )
    with pytest.raises(extension.ContractError, match="outside the frozen"):
        extension.resolve_run_dir(overlay, legacy, 47)


def test_partial_extension_run_must_be_absent_or_have_expected_fingerprint(
    tmp_path: Path,
) -> None:
    run = tmp_path / "seed45"
    extension._validate_partial_new_run(run, 45)  # noqa: SLF001

    run.mkdir()
    with pytest.raises(extension.ContractError, match="without an authenticated identity"):
        extension._validate_partial_new_run(run, 45)  # noqa: SLF001

    (run / "training_identity.json").write_text(
        json.dumps({"fingerprint": "foreign"}), encoding="utf-8"
    )
    with pytest.raises(extension.ContractError, match="foreign training fingerprint"):
        extension._validate_partial_new_run(run, 45)  # noqa: SLF001

    (run / "training_identity.json").write_text(
        json.dumps({"fingerprint": extension.EXPECTED_NEW_FINGERPRINTS[45]}),
        encoding="utf-8",
    )
    extension._validate_partial_new_run(run, 45)  # noqa: SLF001


def test_refit_validator_requires_exact_p75_epoch_rule_and_full_source(tmp_path: Path) -> None:
    run = tmp_path / "seed45"
    refit = run / "final/refit"
    refit.mkdir(parents=True)
    (refit / "model.ckpt").write_bytes(b"checkpoint")
    info = {
        "strategy": "refit",
        "refit_epoch_rule": "p75",
        "fold_best_epochs": [5, 6, 5, 8, 20],
        "refit_epochs": 8,
        "n_train_slides": 1642,
        "label_counts": {"0": 966, "1": 676},
    }
    (refit / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (run / "final/finalize_summary.json").write_text(json.dumps({"refit": info}), encoding="utf-8")

    validated = extension._validate_refit(run)  # noqa: SLF001
    assert validated["refit_epochs"] == 8
    assert validated["fold_best_epochs"] == [5, 6, 5, 8, 20]

    info["refit_epochs"] = 7
    (refit / "info.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(extension.ContractError, match="Invalid p75 refit"):
        extension._validate_refit(run)  # noqa: SLF001


def _synthetic_patient_table() -> pd.DataFrame:
    sizes = dict(zip(extension.DOMAINS, [500, 300, 250, 220, 216], strict=True))
    rows = []
    patient_number = 0
    for domain, size in sizes.items():
        for within in range(size):
            label = within % 2
            row: dict[str, object] = {
                "patient_id": f"P{patient_number:04d}",
                "label": label,
                "cohort": "SurGen" if domain.startswith("SR") else domain,
                "subcohort": domain,
                "k_fold": patient_number % 5,
                "domain": domain,
            }
            for seed in extension.ALL_SEEDS:
                # Perfect within-domain ranking on the native-logit scale.  A
                # seed-specific offset confirms absolute scale is irrelevant.
                row[f"logit_seed{seed}"] = (2 * label - 1) + (seed - 44) * 0.01
            rows.append(row)
            patient_number += 1
    return pd.DataFrame(rows)


def test_result_builder_uses_five_native_logits_and_patient_bootstrap() -> None:
    patients = _synthetic_patient_table()
    result = extension.build_five_seed_results(
        patients,
        n_bootstrap=20,
        bootstrap_seed=123,
    )

    assert result["score_scale"] == "native_logit_only"
    assert result["model_seeds"] == [42, 43, 44, 45, 46]
    assert result["population"]["patients"] == 1486
    assert result["primary_median_seed_macro5"] == {"point": 1.0, "ci95": [1.0, 1.0]}
    assert result["five_seed_ensemble"]["macro5_auroc"] == 1.0
    assert result["five_seed_ensemble"]["pooled_auroc"] == 1.0
    assert result["five_seed_ensemble"]["pooled_auprc"] == 1.0
    assert result["inference"]["unit"] == "patient"
    assert result["inference"]["model_seeds_are_inferential_units"] is False
    assert result["inference"]["shared_indices_across_model_seeds"] is True


def test_result_builder_fails_if_any_native_seed_logit_is_missing() -> None:
    patients = _synthetic_patient_table().drop(columns="logit_seed46")
    with pytest.raises(extension.ContractError, match="incomplete"):
        extension.build_five_seed_results(patients, n_bootstrap=2)


def test_publish_json_is_additive_and_refuses_changed_seal(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    extension._publish_json(path, {"status": "sealed"})  # noqa: SLF001
    extension._publish_json(path, {"status": "sealed"})  # noqa: SLF001
    with pytest.raises(extension.ContractError, match="Refusing to replace"):
        extension._publish_json(path, {"status": "changed"})  # noqa: SLF001


def test_frozen_manifest_and_split_layout_authenticate_when_available() -> None:
    if not extension.DEFAULT_MANIFEST.is_file():
        pytest.skip("production Aim1 manifest is not mounted")
    manifest, splits, inputs = extension._load_manifest_and_splits(  # noqa: SLF001
        extension.DEFAULT_MANIFEST, extension.DEFAULT_SPLIT_DIR
    )

    assert len(manifest) == len(splits) == 1642
    assert manifest["patient_id"].nunique() == 1486
    assert inputs["splits"]["sha256"] == extension.EXPECTED_INPUT_HASHES["splits"]
    assert [int(manifest["k_fold"].eq(fold).sum()) for fold in range(5)] == [
        329,
        328,
        328,
        328,
        329,
    ]
    assert [int(manifest[f"val_fold_{fold}"].sum()) for fold in range(5)] == [
        197,
        196,
        196,
        196,
        196,
    ]


def test_auc_rejects_single_class_cells() -> None:
    with pytest.raises(extension.ContractError, match="lacks both"):
        extension._auc(np.zeros(3, dtype=int), np.arange(3, dtype=float))  # noqa: SLF001


def test_incomplete_new_run_is_quarantined_for_clean_retry(tmp_path: Path) -> None:
    campaign = tmp_path / "aim1_e0"
    partial = extension._overlay_run_dir(campaign, 45)  # noqa: SLF001
    partial.mkdir(parents=True)
    (partial / "partial.txt").write_text("interrupted\n", encoding="utf-8")
    destination = extension._quarantine_incomplete_new_run(campaign, 45)  # noqa: SLF001
    assert destination == campaign / "quarantine/seed45/attempt-001"
    assert not partial.exists()
    assert (destination / "partial.txt").read_text(encoding="utf-8") == "interrupted\n"
    receipt = campaign / "quarantine/seed45/attempt-001.receipt.json"
    assert receipt.is_file()
    assert extension._validate_quarantine_receipts(campaign) == [  # noqa: SLF001
        extension._artifact(receipt)  # noqa: SLF001
    ]

    (destination / "partial.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(extension.ContractError, match="quarantine evidence changed"):
        extension._validate_quarantine_receipts(campaign)  # noqa: SLF001


def test_interrupted_quarantine_receipt_publication_is_recovered(tmp_path: Path) -> None:
    campaign = tmp_path / "aim1_e0"
    destination = campaign / "quarantine/seed46/attempt-001"
    destination.mkdir(parents=True)
    (destination / "partial.txt").write_text("preserved\n", encoding="utf-8")

    with pytest.raises(extension.ContractError, match="census is not one-to-one"):
        extension._validate_quarantine_receipts(campaign)  # noqa: SLF001
    extension._recover_unreceipted_quarantine_attempts(campaign, 46)  # noqa: SLF001
    receipts = extension._validate_quarantine_receipts(campaign)  # noqa: SLF001
    assert len(receipts) == 1
    record = json.loads(Path(receipts[0]["path"]).read_text(encoding="utf-8"))
    assert record["recovered_after_interrupted_receipt_publication"] is True
