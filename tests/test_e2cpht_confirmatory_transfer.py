from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_confirmatory_transfer as cpht


def _orion_manifest() -> pd.DataFrame:
    rows: list[dict] = []
    for patient in range(1, 41):
        slides = [f"CRC{patient:02d}"]
        if patient == 33:
            slides = ["CRC33_01", "CRC33_02"]
        for slide in slides:
            rows.append(
                {
                    "slide_id": slide,
                    "patient_id": f"ORION:C{patient}",
                    "exclude_neoadjuvant": patient in {5, 12, 18, 23, 28, 32},
                    "exclude_ambiguous_crc15": patient == 15,
                }
            )
    return pd.DataFrame(rows)


def _labels() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "patient_id": [f"ORION:C{i}" for i in range(1, 41)],
                "label": [int(i <= 15) for i in range(1, 41)],
            }
        )
        .sort_values("patient_id", kind="stable")
        .reset_index(drop=True)
    )


def _score_frame(scale: float, offset: float = 0.0) -> pd.DataFrame:
    labels = dict(zip(_labels()["patient_id"], _labels()["label"], strict=True))
    rows = []
    for _, slide in _orion_manifest().iterrows():
        patient_number = int(str(slide["patient_id"]).split("C", maxsplit=1)[1])
        noise = ((patient_number * 17) % 13 - 6) / 20
        rows.append(
            {
                "slide_id": slide["slide_id"],
                "seed": 42,
                "fold": 0,
                "logit": offset + scale * labels[slide["patient_id"]] + noise,
            }
        )
    return pd.DataFrame(rows)


def test_component_layout_matches_shared_aim2_lineage_contract(tmp_path: Path) -> None:
    assert cpht.run_dir(tmp_path, 42) == (
        tmp_path / "cpht/train/pb_cap8192/all_conventional/seed42"
    )
    assert cpht.calibrator_path(tmp_path) == (
        tmp_path / "cpht/calibrators/cap8192_all_conventional.json"
    )
    assert cpht.checkpoint_path(tmp_path, 44).name == "model.ckpt"
    assert cpht.run_config_path(tmp_path, 44) == (
        tmp_path / "cpht/train/pb_cap8192/all_conventional/seed44/resolved_config.yaml"
    )


def test_cli_has_separate_seal_train_score_and_report_boundaries(tmp_path: Path) -> None:
    parser = cpht.build_parser()
    for command in ("preflight", "seal", "train"):
        args = parser.parse_args([command, "--output-root", str(tmp_path)])
        assert args.command == command
    for command in ("score", "report"):
        args = parser.parse_args(
            [command, "--output-root", str(tmp_path), "--sibling-root", str(tmp_path / "e2ad")]
        )
        assert args.sibling_root == tmp_path / "e2ad"


def test_frozen_config_is_exact_step_univ1_patient_natural(tmp_path: Path) -> None:
    config = cpht.yaml.safe_load(cpht._compose_refit_config(tmp_path, 43))  # noqa: SLF001

    assert config["encoder"]["name"] == "uni_v1"
    assert config["encoder"]["feature_dim"] == 1024
    assert config["model"]["arch"] == "abmil"
    assert config["model"]["embed_dim"] == 512
    assert config["model"]["attn_dim"] == 384
    assert config["training"]["seed"] == 43
    assert config["training"]["dataset_max_instances"] == 8192
    assert config["training"]["refit_max_steps"] == 6060
    assert config["training"]["train_sampling_strategy"] == "patient_natural"
    assert config["training"]["sample_weight_column"] is None
    assert config["data"]["csv_path"] == str(cpht.source_copy_path(tmp_path))


def test_source_platt_uses_three_seed_mean_honest_patient_oof(tmp_path: Path) -> None:
    patient_ids = [f"P{i:04d}" for i in range(cpht.EXPECTED_SOURCE_PATIENTS)]
    labels = np.array(
        [1] * cpht.EXPECTED_SOURCE_MUTANT
        + [0] * (cpht.EXPECTED_SOURCE_PATIENTS - cpht.EXPECTED_SOURCE_MUTANT)
    )
    manifest = pd.DataFrame(
        {
            "slide_id": [f"S{i:04d}" for i in range(cpht.EXPECTED_SOURCE_PATIENTS)],
            "patient_id": patient_ids,
            "target_label": labels,
            "cohort": "TCGA",
            "subcohort": "TCGA-COAD",
        }
    )
    for seed, shift in zip(cpht.SEEDS, (-0.2, 0.0, 0.2), strict=True):
        directory = tmp_path / f"seed{seed}"
        directory.mkdir(parents=True)
        logits = labels * 1.25 + np.linspace(-0.8, 0.8, len(labels)) + shift
        pd.DataFrame(
            {
                "slide_id": manifest["slide_id"],
                "label": labels,
                "prob_1": 1 / (1 + np.exp(-logits)),
                "logit": logits,
                "fold": np.arange(len(labels)) % 5,
            }
        ).to_parquet(directory / "oof_predictions.parquet", index=False)

    record, table = cpht.build_source_calibrator(manifest, aim1_root=tmp_path)

    assert record["n_source"] == 1486
    assert record["n_mutant"] == 604
    assert record["target_labels_used"] is False
    assert record["b"] > 0
    assert len(record["source_oof_inputs"]) == 3
    assert record["source_cv_inputs"] == record["source_oof_inputs"]
    assert table["patient_id"].is_unique
    assert table["mean_oof_logit"].to_numpy() == pytest.approx(
        labels * 1.25 + np.linspace(-0.8, 0.8, len(labels))
    )


def test_patient_embedding_export_means_c33_slides_within_each_seed() -> None:
    manifest = _orion_manifest()
    rows = []
    for seed in cpht.SEEDS:
        for _, slide in manifest.iterrows():
            value = 3.0 if slide["slide_id"] == "CRC33_02" else 1.0
            rows.append(
                {
                    "slide_id": slide["slide_id"],
                    "seed": seed,
                    "fold": 0,
                    "logit": value,
                    **{f"e{i}": value + i / 1000 for i in range(cpht.EMBED_DIM)},
                }
            )
    patient = cpht.aggregate_patient_features(pd.DataFrame(rows), manifest)

    assert len(patient) == 120
    c33 = patient[patient["patient_id"].eq("ORION:C33")]
    assert len(c33) == 3
    assert c33["n_slides"].eq(2).all()
    assert c33["logit"].tolist() == pytest.approx([2.0, 2.0, 2.0])
    assert c33["e511"].tolist() == pytest.approx([2.511, 2.511, 2.511])


def test_analysis_reports_confirmatory_outer15_and_complete_eight_matrix() -> None:
    names = [
        "all_conventional",
        "aim1_outer15",
        *(f"loco_family_{target.lower()}" for target in cpht.FAMILY_TARGETS),
        *(f"loco_sibling_{arm}" for arm in cpht.SIBLING_ARMS),
    ]
    scores = {
        name: _score_frame(1.0 - index * 0.025, offset=index / 20)
        for index, name in enumerate(names)
    }
    calibrators = {name: (0.0, 1.0) for name in names}
    calibrators["aim1_outer15"] = None

    patient, results = cpht.analyse_transfer(
        scores, calibrators, _orion_manifest(), _labels(), n_bootstrap=40
    )

    assert len(patient) == 40 * 10
    assert results["confirmatory_scorer"] == "all_conventional"
    assert results["complete_eight_model_matrix"] is True
    assert results["outer15_rank_sensitivity_present"] is True
    metrics = results["populations"]["all_40"]["metrics"]
    assert len(metrics) == 10
    assert "source_calibrated" in metrics["all_conventional"]
    assert "source_calibrated" not in metrics["aim1_outer15"]
    assert len(results["populations"]["all_40"]["paired_auroc_vs_confirmatory"]) == 9
    assert results["outcome_access_caveat"] == cpht.OUTCOME_ACCESS_CAVEAT


def test_sibling_resolver_requires_all_models_calibrator_and_receipts(tmp_path: Path) -> None:
    sibling_root = tmp_path / "e2ad"
    seal = sibling_root / "inputs/manifest_contract.json"
    seal.parent.mkdir(parents=True)
    seal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "E2a-D sibling-stratum LOCO",
                "cap": 8192,
                "optimizer_step_budget": 6060,
                "seeds": [42, 43, 44],
                "arms": {arm: {} for arm in cpht.SIBLING_ARMS},
            }
        )
    )
    for arm in cpht.SIBLING_ARMS:
        receipts = {}
        for seed in cpht.SEEDS:
            directory = sibling_root / f"train/pb_cap8192/{arm}/seed{seed}"
            checkpoint = directory / "final/refit/model.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"{arm}-{seed}".encode())
            resolved_config = directory / "resolved_config.yaml"
            resolved_config.write_text("model: test\n")
            run_request = directory / "run_request.json"
            run_request.write_text(json.dumps({"status": "requested"}))
            summary = {
                "schema_version": 1,
                "status": "completed",
                "lineage": "test",
                "arm": arm,
                "target_subcohort": cpht.SIBLING_DISPLAY[arm],
                "seed": seed,
                "cap": 8192,
                "optimizer_step_budget": 6060,
                "model": cpht._artifact(checkpoint),  # noqa: SLF001
                "resolved_config": cpht._artifact(resolved_config),  # noqa: SLF001
                "run_request": cpht._artifact(run_request),  # noqa: SLF001
                "result": {
                    "actual_optimizer_steps": 6060,
                    "refit_max_steps": 6060,
                    "train_sampling_strategy": "patient_natural",
                    "dataset_max_instances": 8192,
                    "eval_full_bags": True,
                },
            }
            (directory / "fit_summary.json").write_text(json.dumps(summary))
            receipt = sibling_root / f"source_cv/{arm}_seed{seed}.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({"status": "completed", "seed": seed}))
            receipts[str(seed)] = cpht._artifact(receipt)  # noqa: SLF001
        calibrator = sibling_root / f"calibrators/cap8192_{arm}.json"
        calibrator.parent.mkdir(parents=True, exist_ok=True)
        calibrator.write_text(
            json.dumps(
                {
                    "a": -0.2,
                    "b": 0.3,
                    "n_source": 100,
                    "source_cv_receipts": receipts,
                    "target_labels_used_for_fit": False,
                }
            )
        )

    models = cpht.resolve_sibling_models(sibling_root)

    assert set(models) == {*cpht.SIBLING_ARMS, "_seal"}
    for arm in cpht.SIBLING_ARMS:
        assert set(models[arm]) == {"42", "43", "44", "calibrator"}


def test_governance_caveat_is_explicit_and_not_a_preoutcome_claim() -> None:
    assert "outcomes were accessed" in cpht.OUTCOME_ACCESS_CAVEAT
    assert "not a pristine pre-outcome" in cpht.OUTCOME_ACCESS_CAVEAT


def test_inference_seal_rejects_runtime_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = cpht.inference_seal_path(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_text(
        json.dumps(
            {
                "complete_eight_model_matrix": True,
                "confirmatory_cpht_model_present": True,
                "target_outcomes_present": False,
                "inference_environment": {"device": "recorded"},
            }
        )
    )
    monkeypatch.setattr(cpht, "_inference_environment", lambda: {"device": "current"})

    with pytest.raises(cpht.ContractError, match="runtime changed"):
        cpht.verify_inference_seal(tmp_path, tmp_path / "e2ad")


def test_analysis_environment_binds_versions_and_executed_sources() -> None:
    environment = cpht._analysis_environment()  # noqa: SLF001
    assert all(
        key in environment
        for key in (
            "python_version",
            "numpy_version",
            "pandas_version",
            "scipy_version",
            "sklearn_version",
        )
    )
    source_names = {Path(item["path"]).name for item in environment["implementation_sources"]}
    assert {
        "aim2_confirmatory_transfer.py",
        "aim2_cross_protocol_transfer.py",
        "core.py",
        "external.py",
    }.issubset(source_names)
