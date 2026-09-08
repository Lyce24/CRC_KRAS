from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim1_all_primary_orion_experiment as experiment


def _synthetic_orion_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    slide_index = 0
    for outer_fold in range(experiment.N_FOLDS):
        for within_fold in range(8):
            patient_id = f"ORION:P{outer_fold}{within_fold}"
            label = int(within_fold < 3)
            kras = "mutant" if label else "wild_type"
            fold_rows.append(
                {"patient_id": patient_id, "label": label, "outer_fold": outer_fold}
            )
            master_rows.append(
                {
                    "slide_uid": f"ORION:S{slide_index:03d}",
                    "output_id": f"S{slide_index:03d}",
                    "patient_uid": patient_id,
                    "cohort": "Orion",
                    "subcohort": "Orion-CRC",
                    "specimen_role": "primary",
                    "include": "yes",
                    "available": "yes",
                    "used_kras": "yes",
                    "qc_slides": "pass",
                    "kras": kras,
                    "kras_subvariant": "G12D" if label else np.nan,
                    "msi_dmmr": "MSS/pMMR",
                    "braf": "wild_type",
                    "nras": "wild_type",
                    "ras": kras,
                    "tumor_site_group": "Colon",
                    "tumor_site_raw": "Sigmoid",
                    "stage_group_major": "II",
                    "stage_group_major_filled": "II",
                    "sidedness": "left",
                    "age_at_diagnosis": 60.0,
                    "sex": "female",
                    "mpp": 0.325,
                }
            )
            slide_index += 1

    # The governed Orion cohort has one patient represented by two slides.
    duplicate = dict(master_rows[-1])
    duplicate["slide_uid"] = "ORION:S040"
    duplicate["output_id"] = "S040"
    master_rows.append(duplicate)

    pack = pd.DataFrame(
        {
            "slide_id": [row["output_id"] for row in master_rows],
            "n_patches": [1_000] * 40 + [10_000],
        }
    )
    return pd.DataFrame(master_rows), pack, pd.DataFrame(fold_rows)


def _synthetic_conventional() -> pd.DataFrame:
    patient_ids = [f"CONV:P{index:04d}" for index in range(1_486)]
    labels = {patient_id: int(index < 604) for index, patient_id in enumerate(patient_ids)}
    rows: list[dict[str, object]] = []
    for index, patient_id in enumerate(patient_ids):
        outer_fold = index % experiment.N_FOLDS
        row: dict[str, object] = {
            "slide_id": f"CONV:S{index:04d}",
            "patient_id": patient_id,
            "target_label": labels[patient_id],
            "specimen_role": "primary",
            "cohort": "TCGA",
            "subcohort": "TCGA-COAD",
            "k_fold": outer_fold,
        }
        for fold in range(experiment.N_FOLDS):
            row[f"val_fold_{fold}"] = int(outer_fold == (fold + 1) % experiment.N_FOLDS)
        rows.append(row)

    # 1,486 patients and 1,642 slides means 156 patients have a second slide.
    duplicate_indices = [*range(72), *range(604, 688)]
    assert len(duplicate_indices) == 156
    for index in duplicate_indices:
        duplicate = dict(rows[index])
        duplicate["slide_id"] = f"CONV:S{len(rows):04d}"
        rows.append(duplicate)
    return pd.DataFrame(rows)


def test_orion_rows_preserve_cpht_a_outer_folds_and_use_next_block_for_validation() -> None:
    master, pack, frozen_folds = _synthetic_orion_inputs()

    rows = experiment._build_orion_rows(  # noqa: SLF001
        master, pack, frozen_folds, section_threshold=106.0
    )
    patients = rows.drop_duplicates("patient_id")

    assert len(rows) == experiment.EXPECTED_ORION_SLIDES == 41
    assert len(patients) == experiment.EXPECTED_ORION_PATIENTS == 40
    assert int(patients["target_label"].sum()) == experiment.EXPECTED_ORION_MUTANTS == 15
    for outer_fold in range(experiment.N_FOLDS):
        test = patients.loc[patients["k_fold"].eq(outer_fold), "target_label"]
        validation = patients.loc[patients[f"val_fold_{outer_fold}"].eq(1)]
        assert test.value_counts().sort_index().to_dict() == {0: 5, 1: 3}
        assert set(validation["k_fold"]) == {(outer_fold + 1) % experiment.N_FOLDS}
        assert validation["target_label"].value_counts().sort_index().to_dict() == {0: 5, 1: 3}
        assert not rows.loc[rows["k_fold"].eq(outer_fold), f"val_fold_{outer_fold}"].any()

    duplicated_patient = master.iloc[-1]["patient_uid"]
    duplicate_rows = rows.loc[rows["patient_id"].eq(duplicated_patient)]
    assert duplicate_rows["slide_weight"].tolist() == [0.5, 0.5]
    assert set(rows["section_size_class"]) == {"small_fragment", "large_section"}


def test_combined_source_has_locked_slide_patient_and_mutant_census() -> None:
    master, pack, frozen_folds = _synthetic_orion_inputs()
    orion = experiment._build_orion_rows(master, pack, frozen_folds)  # noqa: SLF001

    combined = experiment._combine_source(_synthetic_conventional(), orion)  # noqa: SLF001
    patient = combined.drop_duplicates("patient_id")

    assert len(combined) == experiment.EXPECTED_SOURCE_SLIDES == 1_683
    assert len(patient) == experiment.EXPECTED_SOURCE_PATIENTS == 1_526
    assert int(patient["target_label"].sum()) == experiment.EXPECTED_SOURCE_MUTANTS == 619
    assert int(combined["target_label"].sum()) == experiment.EXPECTED_SOURCE_MUTANT_SLIDES == 691
    assert set(combined["specimen_role"]) == {"primary"}
    assert not combined["slide_id"].duplicated().any()
    assert combined.groupby("patient_id")["target_label"].nunique().eq(1).all()


def test_combined_source_rejects_a_multislide_patient_spanning_outer_folds() -> None:
    master, pack, frozen_folds = _synthetic_orion_inputs()
    orion = experiment._build_orion_rows(master, pack, frozen_folds)  # noqa: SLF001
    conventional = _synthetic_conventional()
    duplicate = conventional["patient_id"].eq("CONV:P0000")
    assert int(duplicate.sum()) == 2
    second = conventional.index[duplicate][-1]
    conventional.loc[second, "k_fold"] = 1

    with pytest.raises(experiment.ContractError, match="spanning outer folds"):
        experiment._combine_source(conventional, orion)  # noqa: SLF001


def test_combined_source_rejects_inconsistent_multislide_validation_membership() -> None:
    master, pack, frozen_folds = _synthetic_orion_inputs()
    orion = experiment._build_orion_rows(master, pack, frozen_folds)  # noqa: SLF001
    conventional = _synthetic_conventional()
    duplicate = conventional["patient_id"].eq("CONV:P0000")
    assert int(duplicate.sum()) == 2
    second = conventional.index[duplicate][-1]
    conventional.loc[second, "val_fold_1"] = 1 - int(
        conventional.loc[second, "val_fold_1"]
    )

    with pytest.raises(experiment.ContractError, match="spanning val_fold_1 membership"):
        experiment._combine_source(conventional, orion)  # noqa: SLF001


def test_oof_validation_rejects_full_coverage_predictions_from_the_wrong_fold(
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "slide_id": [f"S{index}" for index in range(5)],
            "target_label": [0, 1, 0, 1, 0],
            "k_fold": list(range(5)),
        }
    )
    oof = pd.DataFrame(
        {
            "slide_id": source["slide_id"],
            "label": source["target_label"],
            "logit": np.linspace(-1, 1, len(source)),
            "fold": [1, 2, 3, 4, 0],
        }
    )
    path = tmp_path / "wrong_fold.parquet"
    oof.to_parquet(path, index=False)

    with pytest.raises(experiment.ContractError, match="fold identities disagree"):
        experiment._validate_oof_frame(path, source, seed=42)  # noqa: SLF001


def test_root_oof_must_equal_authenticated_fold_prediction_concatenation(
    tmp_path: Path,
) -> None:
    pieces = []
    for fold in range(experiment.N_FOLDS):
        frame = pd.DataFrame(
            {
                "slide_id": [f"S{fold}"],
                "label": [fold % 2],
                "prob_1": [0.1 + fold / 10],
                "logit": [float(fold - 2)],
            }
        )
        directory = tmp_path / f"fold_{fold}"
        directory.mkdir()
        frame.to_parquet(directory / "preds_test.parquet", index=False)
        pieces.append(frame.assign(fold=fold))
    root = pd.concat(pieces, ignore_index=True)
    root.to_parquet(tmp_path / "oof_predictions.parquet", index=False)
    experiment._validate_root_oof_against_folds(tmp_path, seed=42)  # noqa: SLF001

    root.loc[root["slide_id"].eq("S3"), "logit"] += 1.0
    root.to_parquet(tmp_path / "oof_predictions.parquet", index=False)
    with pytest.raises(experiment.ContractError, match="root OOF differs"):
        experiment._validate_root_oof_against_folds(tmp_path, seed=42)  # noqa: SLF001


@pytest.mark.parametrize("seed", experiment.SEEDS)
def test_training_config_is_exact_e0_recipe_with_six_workers(tmp_path: Path, seed: int) -> None:
    output_root = tmp_path / "experiment"
    config = experiment._compose_training_cfg(  # noqa: SLF001
        seed,
        experiment._source_path(output_root),  # noqa: SLF001
        experiment._split_root(output_root),  # noqa: SLF001
        experiment._run_dir(output_root, seed),  # noqa: SLF001
    )
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)

    experiment._validate_training_config(resolved, seed, output_root)  # noqa: SLF001
    assert resolved["training"]["dataset_max_instances"] == 8_192
    assert resolved["training"]["train_sampling_strategy"] == "patient_natural"
    assert resolved["training"]["num_workers"] == 6
    assert resolved["training"]["refit_epoch_rule"] == "p75"
    assert resolved["training"]["refit_max_steps"] is None
    assert resolved["training"]["seed"] == seed


def test_source_platt_uses_only_three_seed_honest_oof_patient_logits() -> None:
    labels = np.array([0] * 10 + [1] * 10, dtype=int)
    patients = [f"P{index:02d}" for index in range(len(labels))]
    seed_frames: dict[int, pd.DataFrame] = {}
    for seed_index, seed in enumerate(experiment.SEEDS):
        seed_frames[seed] = pd.DataFrame(
            {
                "patient_id": patients,
                "label": labels,
                "mean_logit": labels * 1.2 - 0.6 + seed_index * 0.15,
            }
        )

    record, table = experiment._fit_source_platt(  # noqa: SLF001
        seed_frames,
        {str(seed): {"sha256": f"oof-{seed}"} for seed in experiment.SEEDS},
    )

    expected = np.mean(
        np.vstack([seed_frames[seed]["mean_logit"] for seed in experiment.SEEDS]), axis=0
    )
    np.testing.assert_allclose(table["mean_oof_logit"], expected)
    assert table.columns.tolist() == ["patient_id", "label", "mean_oof_logit"]
    assert record["n_source"] == 20
    assert record["n_mutant"] == 10
    assert record["b"] > 0
    assert record["target_labels_used"] is False
    assert record["target_labels_used_for_fit"] is False
    assert set(record["source_oof_inputs"]) == {"42", "43", "44"}


def test_source_platt_fails_closed_if_a_seed_patient_roster_changes() -> None:
    base = pd.DataFrame(
        {
            "patient_id": ["A", "B", "C", "D"],
            "label": [0, 0, 1, 1],
            "mean_logit": [-1.0, -0.5, 0.5, 1.0],
        }
    )
    frames = {seed: base.copy() for seed in experiment.SEEDS}
    frames[43].loc[0, "patient_id"] = "changed"

    with pytest.raises(experiment.ContractError, match="roster/labels differ"):
        experiment._fit_source_platt(frames)  # noqa: SLF001


def _synthetic_oof_and_refit_patient_logits() -> tuple[
    dict[int, pd.DataFrame], dict[int, pd.DataFrame]
]:
    patients = ["P3", "P1", "P4", "P2"]
    labels = [1, 0, 1, 0]
    base = np.array([0.5, -1.5, 1.5, -0.5])
    oof: dict[int, pd.DataFrame] = {}
    refit: dict[int, pd.DataFrame] = {}
    for seed_index, seed in enumerate(experiment.SEEDS, start=1):
        oof[seed] = pd.DataFrame(
            {
                "patient_id": patients,
                "label": labels,
                "mean_logit": base * seed_index,
            }
        )
        # Reverse row order to prove that alignment is by patient identity,
        # not by incidental dataframe order.
        refit[seed] = pd.DataFrame(
            {
                "patient_id": patients,
                "label": labels,
                "mean_logit": base * seed_index * 2.0,
            }
        ).iloc[::-1]
    return oof, refit


def test_refit_vs_honest_oof_native_logit_sd_is_patient_level_and_seed_aligned() -> None:
    oof, refit = _synthetic_oof_and_refit_patient_logits()

    result = experiment._refit_oof_logit_sd_diagnostic(oof, refit)  # noqa: SLF001

    assert result["unit"] == "patient; mean native slide logit"
    assert result["sd_definition"] == "sample standard deviation (ddof=1)"
    for seed_index, seed in enumerate(experiment.SEEDS, start=1):
        block = result["per_seed"][seed]
        expected_oof_sd = float(np.std(oof[seed]["mean_logit"], ddof=1))
        assert block["n_patients"] == 4
        assert block["honest_oof_logit_sd"] == pytest.approx(expected_oof_sd)
        assert block["full_source_refit_logit_sd"] == pytest.approx(2 * expected_oof_sd)
        assert block["refit_to_oof_sd_ratio"] == pytest.approx(2.0)
        assert expected_oof_sd == pytest.approx(np.sqrt(5 / 3) * seed_index)

    ensemble = result["three_seed_mean_native_logit"]
    assert ensemble["n_patients"] == 4
    assert ensemble["honest_oof_logit_sd"] == pytest.approx(2 * np.sqrt(5 / 3))
    assert ensemble["full_source_refit_logit_sd"] == pytest.approx(4 * np.sqrt(5 / 3))
    assert ensemble["refit_to_oof_sd_ratio"] == pytest.approx(2.0)


@pytest.mark.parametrize("failure", ["patient", "label", "nonfinite", "zero_oof_sd"])
def test_refit_vs_oof_logit_sd_fails_closed_on_invalid_alignment_or_scale(
    failure: str,
) -> None:
    oof, refit = _synthetic_oof_and_refit_patient_logits()
    if failure == "patient":
        refit[43].loc[refit[43].index[0], "patient_id"] = "changed"
        expected = "roster/labels differ"
    elif failure == "label":
        refit[43].loc[refit[43].index[0], "label"] ^= 1
        expected = "roster/labels differ"
    elif failure == "nonfinite":
        refit[43].loc[refit[43].index[0], "mean_logit"] = np.inf
        expected = "insufficient/non-finite logits"
    else:
        for seed in experiment.SEEDS:
            oof[seed]["mean_logit"] = 1.0
        expected = "SD is non-positive"

    with pytest.raises(experiment.ContractError, match=expected):
        experiment._refit_oof_logit_sd_diagnostic(oof, refit)  # noqa: SLF001


def test_rih_headline_excludes_exact_eight_dual_role_patients() -> None:
    assert len(experiment.RIH_DUAL_PATIENTS) == 8
    retained = {"RIH:new-1", "RIH:new-2"}
    rih = pd.DataFrame(
        {"patient_id": [*sorted(experiment.RIH_DUAL_PATIENTS), *sorted(retained)]}
    )

    headline = experiment._headline_target_frame("rih_m", rih)  # noqa: SLF001
    assert set(headline["patient_id"]) == retained
    assert not set(headline["patient_id"]) & set(experiment.RIH_DUAL_PATIENTS)

    sr = pd.DataFrame({"patient_id": ["SR:1", "SR:2"]})
    pd.testing.assert_frame_equal(
        experiment._headline_target_frame("sr1482_m", sr),  # noqa: SLF001
        sr,
    )


def _synthetic_target_overlap_inputs() -> tuple[
    pd.DataFrame, dict[str, pd.DataFrame], list[str], list[str]
]:
    dual = sorted(experiment.RIH_DUAL_PATIENTS)
    rih_unique = [f"RIH:MET{index:03d}" for index in range(77)]
    rih_ids = [*dual, *rih_unique]
    rih = pd.DataFrame(
        {
            "slide_id": [f"RIH:S{index:03d}" for index in range(len(rih_ids))],
            "patient_id": rih_ids,
        }
    )
    sr_ids = [f"SR:P{index:03d}" for index in range(74)]
    sr_rows = [*sr_ids, *sr_ids[:26]]
    sr = pd.DataFrame(
        {
            "slide_id": [f"SR:S{index:03d}" for index in range(len(sr_rows))],
            "patient_id": sr_rows,
        }
    )
    source = pd.DataFrame({"patient_id": [*dual, "SOURCE:OTHER"]})
    return source, {"rih_m": rih, "sr1482_m": sr}, rih_unique, sr_ids


def test_pretraining_target_overlap_guard_freezes_rih77_and_sr74() -> None:
    source, targets, _, _ = _synthetic_target_overlap_inputs()

    headline = experiment._validate_target_overlap(source, targets)  # noqa: SLF001

    assert (len(headline["rih_m"]), headline["rih_m"]["patient_id"].nunique()) == (77, 77)
    assert (len(headline["sr1482_m"]), headline["sr1482_m"]["patient_id"].nunique()) == (
        100,
        74,
    )


@pytest.mark.parametrize("mutation", ["missing_rih", "extra_rih", "sr_overlap"])
def test_pretraining_target_overlap_guard_fails_on_any_roster_change(mutation: str) -> None:
    source, targets, rih_unique, sr_ids = _synthetic_target_overlap_inputs()
    if mutation == "missing_rih":
        source = source[source["patient_id"].ne(sorted(experiment.RIH_DUAL_PATIENTS)[0])]
        expected = "frozen eight"
    elif mutation == "extra_rih":
        source = pd.concat(
            [source, pd.DataFrame({"patient_id": [rih_unique[0]]})], ignore_index=True
        )
        expected = "frozen eight"
    else:
        source = pd.concat(
            [source, pd.DataFrame({"patient_id": [sr_ids[0]]})], ignore_index=True
        )
        expected = "overlaps SR1482"

    with pytest.raises(experiment.ContractError, match=expected):
        experiment._validate_target_overlap(source, targets)  # noqa: SLF001


def _patient_metric_rows(patient_ids: list[str], labels: list[int]) -> pd.DataFrame:
    label_array = np.asarray(labels, dtype=int)
    jitter = np.linspace(-0.2, 0.2, len(labels))
    logits = np.where(label_array == 1, 0.8, -0.8) + jitter
    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "label": label_array,
            "mean_logit": logits,
            "prob_raw": experiment.sigmoid(logits),
            "prob_source_calibrated": experiment.sigmoid(-0.1 + 0.7 * logits),
        }
    )


def test_metastatic_probability_outputs_are_labeled_deployment_calibration_sensitivities() -> None:
    dual = sorted(experiment.RIH_DUAL_PATIENTS)
    rih_other = [f"RIH:MET{index:03d}" for index in range(77)]
    rih = _patient_metric_rows(
        [*dual, *rih_other],
        [1] * 4 + [0] * 4 + [1] * 33 + [0] * 44,
    )
    sr_ids = [f"SR:P{index:03d}" for index in range(74)]
    sr = _patient_metric_rows(sr_ids, [1] * 30 + [0] * 44)

    result = experiment._metastatic_analysis(  # noqa: SLF001
        {"rih_m": rih, "sr1482_m": sr},
        n_bootstrap=20,
        bootstrap_seed=19,
    )

    assert "deployment-calibration sensitivities only" in result[
        "probability_and_calibration_claim"
    ]
    assert "Raw native-logit AUROC/AUPRC are primary" in result[
        "probability_and_calibration_claim"
    ]
    for target in experiment.TARGETS:
        block = result["headline"][target]
        assert "deployment-calibration sensitivity" in block["probability_metrics_role"]
        assert block["source_calibrated"]["interpretation"] == block[
            "probability_metrics_role"
        ]
    assert "deployment-calibration sensitivity" in result["equal_cohort_macro"][
        "probability_metrics_role"
    ]


def test_output_guard_rejects_final_v8_descendants_and_allows_tmp(tmp_path: Path) -> None:
    protected = experiment.FINAL_V8_ROOT / "must_not_write_here"
    with pytest.raises(experiment.ContractError, match="protected canonical tree"):
        experiment._guard_output_root(protected)  # noqa: SLF001

    assert experiment._guard_output_root(tmp_path) == tmp_path.resolve()  # noqa: SLF001


def test_production_input_overrides_fail_closed_but_tmp_fixtures_are_allowed(
    tmp_path: Path,
) -> None:
    canonical = {
        "conventional_manifest": experiment.CONVENTIONAL_MANIFEST,
        "label_source": experiment.LABEL_SOURCE,
        "orion_folds": experiment.CPHT_A_FOLDS,
        "conventional_splits": experiment.CONVENTIONAL_SPLITS,
        "met_input_root": experiment.FINAL_V8_MET_INPUT_ROOT,
        "pack_dir": experiment.PACK_DIR,
    }
    production = argparse.Namespace(
        output_root=experiment.DEFAULT_OUTPUT_ROOT,
        **{**canonical, "conventional_manifest": tmp_path / "permuted.csv"},
    )
    with pytest.raises(experiment.ContractError, match="canonical frozen input paths"):
        experiment._validate_canonical_input_arguments(production)  # noqa: SLF001

    fixture = argparse.Namespace(
        output_root=tmp_path / "campaign",
        **{name: tmp_path / name for name in canonical},
    )
    experiment._validate_canonical_input_arguments(fixture)  # noqa: SLF001


def _saturated_source_frames() -> dict[int, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for index in range(20):
        label = index % 2
        rows.append(
            {
                "patient_id": f"SAT:P{index:02d}",
                "label": label,
                # Both classes saturate to probability 1.0 in float64, while
                # their native logits retain perfect rank separation.
                "mean_logit": 40.0 if label else 37.0,
                "k_fold": (index // 2) % experiment.N_FOLDS,
                "cohort": "TCGA",
                "subcohort": "TCGA-COAD",
                "msi_dmmr": "MSS/pMMR",
                "braf": "wild_type",
                "tumor_site_group": "Colon",
                "stage_class": "I-II",
                "stage_group_major": "II",
                "stage_group_major_filled": "II",
                "sidedness": "left",
            }
        )
    base = pd.DataFrame(rows)
    base["prob_raw"] = experiment.sigmoid(base["mean_logit"].to_numpy(dtype=float))
    assert base["prob_raw"].nunique() == 1
    return {
        seed: base.assign(mean_logit=base["mean_logit"] + 0.1 * seed_index)
        for seed_index, seed in enumerate(experiment.SEEDS)
    }


@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
def test_source_rank_diagnostics_never_roundtrip_through_saturated_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _saturated_source_frames()
    # This is the regression discriminator: the legacy probability path sees
    # one giant tie and scores 0.5, while the persisted native logits score 1.
    assert experiment.evaluate.patient_auroc(frames[42]) == pytest.approx(0.5)
    assert experiment.evaluate.patient_auroc(
        frames[42], score_column="mean_logit"
    ) == pytest.approx(1.0)

    monkeypatch.setattr(
        experiment,
        "_acquisition_domains",
        lambda frame: {"synthetic": frame},
    )
    e0_result = experiment._source_e0_diagnostics(frames, n_bootstrap=20)  # noqa: SLF001
    assert all(
        e0_result["per_seed"][seed]["pooled_oof"]["auroc"] == pytest.approx(1.0)
        for seed in experiment.SEEDS
    )
    assert e0_result["three_seed_mean_native_logit_oof_sensitivity"]["auroc"][
        "auroc"
    ] == pytest.approx(1.0)
    ensemble = experiment._mean_native_logit_ensemble(frames)  # noqa: SLF001
    assert ensemble["prob_raw"].nunique() == 1
    assert experiment.evaluate.patient_auroc(
        ensemble, score_column="mean_logit"
    ) == pytest.approx(1.0)

    # Keep the integration test focused on score routing: A and D are enough
    # to exercise shared-resample AUROC and AUPRC, while the production gate
    # itself is tested elsewhere.
    monkeypatch.setattr(
        experiment.e1a,
        "SETS",
        {
            "A_all_primary": ("All", "main"),
            "D_mss_braf_wt": ("MSS/BRAF-WT", "main"),
        },
    )
    monkeypatch.setattr(experiment.e0, "evaluate_gate", lambda _: {"verdict": "test"})
    e1a_result = experiment._source_e1a_diagnostics(frames, n_bootstrap=20)  # noqa: SLF001
    for seed in experiment.SEEDS:
        for key in ("A_all_primary", "D_mss_braf_wt"):
            assert e1a_result["per_seed"][seed][key]["auroc"] == pytest.approx(1.0)
            assert e1a_result["per_seed"][seed][key]["auprc"] == pytest.approx(1.0)
