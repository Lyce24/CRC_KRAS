from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_sibling_loco as e2ad
from oceanpath.aim1 import lineage

DOMAIN_COUNTS = {
    "TCGA-COAD": (374, 160, "TCGA"),
    "TCGA-READ": (128, 47, "TCGA"),
    "SR386": (413, 147, "SurGen"),
    "SR1482": (324, 147, "SurGen"),
    "RIH-Colon": (153, 70, "RIH"),
    "CPTAC-COAD": (94, 33, "CPTAC"),
}


def _nested_record(fields: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for dotted, value in fields.items():
        destination = record
        keys = dotted.split(".")
        for key in keys[:-1]:
            destination = destination.setdefault(key, {})
        destination[keys[-1]] = value
    return record


def _conventional_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subcohort, (n_patients, n_mutant, cohort) in DOMAIN_COUNTS.items():
        for index in range(n_patients):
            mutant = index < n_mutant
            patient_id = f"{subcohort}:P{index:04d}"
            rows.append(
                {
                    "slide_id": f"{patient_id}:S0",
                    "patient_id": patient_id,
                    "target_label": int(mutant),
                    "slide_weight": 1.0,
                    "kras": "mutant" if mutant else "wild_type",
                    "cohort": cohort,
                    "subcohort": subcohort,
                    "specimen_role": "primary",
                    "site_class": "colon",
                    "stage_class": "I-II",
                    "mpp_bin": "0.50",
                    "technical_class": "standard",
                    "n_slides_class": "1",
                }
            )
    return pd.DataFrame(rows)


def test_four_arm_censuses_and_sibling_retention_are_exact() -> None:
    primary = _conventional_frame()

    for arm, spec in e2ad.ARMS.items():
        source, target = e2ad.build_arm_frames(primary, arm)

        assert target["patient_id"].nunique() == spec.expected_target_patients
        assert int(target.drop_duplicates("patient_id")["target_label"].sum()) == (
            spec.expected_target_mutant
        )
        assert source["patient_id"].nunique() == spec.expected_source_patients
        assert int(source.drop_duplicates("patient_id")["target_label"].sum()) == (
            spec.expected_source_mutant
        )
        assert set(target["subcohort"]) == {spec.target_subcohort}
        assert spec.target_subcohort not in set(source["subcohort"])
        assert spec.related_retained in set(source["subcohort"])
        assert set(source["patient_id"]).isdisjoint(target["patient_id"])


@pytest.mark.parametrize(
    ("n_patients", "expected_epochs"),
    [(1_073, 6), (1_162, 6), (1_112, 6), (1_358, 5)],
)
def test_refit_epoch_ceiling_reaches_exact_step_budget(
    n_patients: int, expected_epochs: int
) -> None:
    epochs = e2ad.epochs_for_n(n_patients)

    assert epochs == expected_epochs
    assert epochs * n_patients >= e2ad.STEP_BUDGET
    assert (epochs - 1) * n_patients < e2ad.STEP_BUDGET


def test_source_cv_layout_is_patient_grouped_and_excludes_test_from_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = e2ad.build_arm_frames(_conventional_frame(), "sr386")
    monkeypatch.setattr(
        e2ad.balance,
        "carve_out_validation",
        lambda pool, _columns, _patients, _ratio: (
            pool.drop_duplicates("patient_id")["patient_id"].iloc[::7].tolist()
        ),
    )

    manifest = e2ad.add_fold_columns(source)

    assert set(manifest["k_fold"]) == set(range(5))
    assert manifest.groupby("patient_id")["k_fold"].nunique().eq(1).all()
    for fold in range(5):
        validation = manifest[f"val_fold_{fold}"].astype(int)
        assert not (validation.eq(1) & manifest["k_fold"].eq(fold)).any()
        assert manifest.groupby("patient_id")[f"val_fold_{fold}"].nunique().eq(1).all()


def test_chunked_bootstrap_auroc_matches_sklearn_with_resampling_ties() -> None:
    labels = np.repeat([0, 1], [11, 9])
    score = np.array(
        [-2, -1, -1, -0.5, 0, 0, 0.2, 0.4, 0.4, 1, 1.2, -0.8, -0.1, 0, 0.2, 0.2, 0.7, 1, 1, 1.4],
        dtype=float,
    )
    rng = np.random.default_rng(77)
    indices = e2ad.stratified_bootstrap_indices(labels, n_bootstrap=80, rng=rng)

    observed = e2ad.bootstrap_auroc_samples(labels, score, indices, chunk_size=7)
    expected = np.array([roc_auc_score(labels[index], score[index]) for index in indices])

    np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-16)


def test_paired_alignment_requires_identical_patient_and_label_roster() -> None:
    sibling = pd.DataFrame({"patient_id": ["a", "b"], "label": [0, 1], "mean_logit": [-1.0, 1.0]})
    family = sibling.copy()

    left, right = e2ad._align_pair(sibling, family, arm="sr386")
    assert list(left["patient_id"]) == list(right["patient_id"])

    with pytest.raises(RuntimeError, match="patients differ"):
        e2ad._align_pair(
            sibling,
            family.assign(patient_id=["a", "c"]),
            arm="sr386",
        )
    with pytest.raises(RuntimeError, match="labels differ"):
        e2ad._align_pair(
            sibling,
            family.assign(label=[1, 1]),
            arm="sr386",
        )


def test_all_artifact_paths_are_lineage_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(lineage.AIM2_LINEAGE_ENV, "e2ad_unit_test")
    root = lineage.paths.OUTPUT_ROOT / "reruns/e2ad_unit_test/e2ad"

    assert e2ad.component_root() == root
    assert e2ad.source_manifest("sr386").is_relative_to(root)
    assert e2ad.target_manifest("tcga_read").is_relative_to(root)
    assert e2ad.model_ckpt("sr1482", 43).is_relative_to(root)
    assert e2ad.calibrator_path("tcga_coad").is_relative_to(root)
    assert e2ad.report_path().is_relative_to(root)


def test_governed_scorer_is_packed_full_bag_cuda_bfloat16() -> None:
    source = inspect.getsource(e2ad._score_packed_checkpoint)

    assert "PackedFeatureStore" in source
    assert "force_float32=True" in source
    assert "max_instances=None" in source
    assert 'map_location="cuda"' in source
    assert "dtype=torch.bfloat16" in source
    assert "score_manifest_with_checkpoints" not in source


def test_inference_and_analysis_seals_bind_direct_implementation_sources() -> None:
    inference = inspect.getsource(e2ad._current_inference_environment)
    analysis = inspect.getsource(e2ad._current_analysis_environment)
    report = inspect.getsource(e2ad.cmd_report)

    for path in (
        "models/__init__.py",
        "models/wsi_classifier.py",
        "contracts/__init__.py",
        "contracts/slide_ids.py",
        "datasets/datamodule.py",
        "datasets/packed.py",
    ):
        assert path in inference
    assert "eval/external.py" in analysis
    assert "_seal_inference_environment" in report
    assert "_seal_analysis_environment" in report


def test_source_snapshot_covers_launcher_and_every_hydra_yaml() -> None:
    observed = e2ad._source_files()
    expected_configs = {
        str(path.relative_to(e2ad.REPO)) for path in (e2ad.REPO / "configs").rglob("*.yaml")
    }

    assert observed == tuple(sorted(set(observed)))
    assert "tools/study_train.py" in observed
    assert expected_configs <= set(observed)


def test_source_cv_training_identity_proves_five_fold_recipe_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm, seed = "sr386", 42
    source = tmp_path / "source.csv"
    source.write_text("slide_id,patient_id,target_label\ns,p,0\n")
    splits_root = tmp_path / "splits"
    splits = splits_root / f"aim1_{e2ad.data_name(arm)}" / e2ad.paths.SPLIT_NAME
    splits.mkdir(parents=True)
    (splits / ".integrity_hash").write_text("sealed split integrity\n")
    directory = tmp_path / "source_cv"
    directory.mkdir()
    monkeypatch.setattr(e2ad, "source_manifest", lambda _arm: source)
    monkeypatch.setattr(e2ad, "split_root", lambda: splits_root)
    monkeypatch.setattr(e2ad, "split_dir", lambda _arm: splits)

    material = _nested_record(e2ad._source_cv_material_expectations(arm, seed))
    fingerprint = "0123456789abcdef"
    identity = {
        "fingerprint": fingerprint,
        "payload": {
            "material_config": material,
            "input_evidence": {
                "manifest_sha256": lineage.sha256_file(source),
                "split_integrity_sha256": lineage.sha256_file(splits / ".integrity_hash"),
            },
        },
    }
    (directory / "training_identity.json").write_text(json.dumps(identity))
    for fold in range(e2ad.paths.N_FOLDS):
        config = _nested_record(e2ad._source_cv_material_expectations(arm, seed))
        config["platform"]["splits_root"] = str(splits_root.resolve())
        config["train_dir"] = str(directory.resolve())
        fold_dir = directory / f"fold_{fold}"
        fold_dir.mkdir()
        (fold_dir / "config.yaml").write_text(e2ad.yaml.safe_dump(config))
    completion = {
        "training_fingerprint": fingerprint,
        "n_folds": e2ad.paths.N_FOLDS,
        "skip_finalize": True,
        "fold_completions": [{} for _ in range(e2ad.paths.N_FOLDS)],
    }

    e2ad._validate_source_cv_training_identity(arm, seed, directory, completion)

    bad_completion = {**completion, "n_folds": e2ad.paths.N_FOLDS - 1}
    with pytest.raises(RuntimeError, match="exactly five"):
        e2ad._validate_source_cv_training_identity(arm, seed, directory, bad_completion)
    material["training"]["dataset_max_instances"] = 4_096
    identity["payload"]["material_config"] = material
    (directory / "training_identity.json").write_text(json.dumps(identity))
    with pytest.raises(RuntimeError, match="recipe mismatch"):
        e2ad._validate_source_cv_training_identity(arm, seed, directory, completion)


def test_downstream_checkpoint_and_calibrator_schemas_are_explicit() -> None:
    fit = inspect.getsource(e2ad.cmd_fit)
    completion = inspect.getsource(e2ad._completed_refit)
    calibration = inspect.getsource(e2ad.calibrate_one)

    for field in (
        "target_subcohort",
        "optimizer_step_budget",
        "sampling_seed",
        "resolved_config",
        "run_request",
    ):
        assert field in fit
        assert field in completion
    for field in (
        "source_cv_receipts",
        "target_labels_used_for_fit",
        "n_source",
        '"a"',
        '"b"',
    ):
        assert field in calibration


def test_smoke_is_explicitly_non_governed_and_bounded() -> None:
    source = inspect.getsource(e2ad.cmd_smoke)

    assert "PASS_NON_GOVERNED_SMOKE_ONLY" in source
    assert '"governed_fit": False' in source
    assert '"may_enter_e2ad_roster": False' in source
    assert "steps > 10" in source
    assert "e2ad_non_governed_smoke_" in source


def test_cli_exposes_dry_runs_and_non_governed_smoke() -> None:
    parser = e2ad.build_parser()

    source_cv = parser.parse_args(["source-cv", "--arm", "sr386", "--dry-run"])
    train = parser.parse_args(["train", "--seed", "42", "--dry-run"])
    smoke = parser.parse_args(["smoke", "--steps", "2", "--dry-run"])

    assert source_cv.dry_run is True
    assert train.dry_run is True
    assert smoke.steps == 2 and smoke.dry_run is True
