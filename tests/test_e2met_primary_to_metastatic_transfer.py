from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_primary_to_metastatic_transfer as met
from oceanpath.eval.external import sigmoid


def _patient_frame(
    prefix: str,
    n_per_class: int,
    *,
    separation: float,
    scorer_offset: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for label in (0, 1):
        for index in range(n_per_class):
            patient_id = f"{prefix}:{label}:{index}"
            # Both organ bins retain both outcome classes.
            organ = "liver" if index % 2 == 0 else "non_liver"
            logit = separation * label + 0.07 * index + scorer_offset
            rows.append(
                {
                    "patient_id": patient_id,
                    "label": label,
                    "mean_logit": logit,
                    "prob_raw": float(sigmoid(np.array([logit]))[0]),
                    "prob_source_calibrated": float(sigmoid(np.array([-0.1 + 0.25 * logit]))[0]),
                    "liver_class": organ,
                    "n_slides": 1,
                    "specimen_role": "metastatic",
                    "subcohort": "RIH-Colon" if prefix == "rih" else "SR1482",
                }
            )
    return pd.DataFrame(rows).sort_values("patient_id", kind="stable").reset_index(drop=True)


def _analysis_inputs() -> tuple[
    dict[str, dict[str, pd.DataFrame]],
    dict[tuple[str, str], pd.DataFrame],
    set[str],
]:
    patient_scores: dict[str, dict[str, pd.DataFrame]] = {}
    for scorer_index, scorer in enumerate(met.ALL_SCORERS):
        patient_scores[scorer] = {
            "rih_m": _patient_frame(
                "rih",
                6,
                separation=0.35 + 0.02 * scorer_index,
                scorer_offset=0.01 * scorer_index,
            ),
            "sr1482_m": _patient_frame(
                "sr",
                6,
                separation=0.30 + 0.03 * scorer_index,
                scorer_offset=-0.01 * scorer_index,
            ),
        }

    # One dual patient per class leaves a leakage-free 10-patient RIH arm.
    dual = {"rih:0:0", "rih:1:0"}
    primary_scores = {
        ("heldout_rih", "rih_primary"): _patient_frame("rih", 6, separation=0.65).assign(
            specimen_role="primary"
        ),
        ("heldout_surgen", "sr1482_primary"): _patient_frame("srp", 6, separation=0.55).assign(
            specimen_role="primary", subcohort="SR1482"
        ),
        ("heldout_sr1482", "sr1482_primary"): _patient_frame("srp", 6, separation=0.60).assign(
            specimen_role="primary", subcohort="SR1482"
        ),
    }
    return patient_scores, primary_scores, dual


def test_exposure_labels_match_final_v8_hierarchy() -> None:
    assert met.exposure_class("heldout_rih", "rih_m") == "family-naive"
    assert met.exposure_class("heldout_surgen", "rih_m") == "target-primary-exposed"
    assert met.exposure_class("heldout_surgen", "sr1482_m") == "family-naive"
    assert met.exposure_class("heldout_sr1482", "sr1482_m") == "sibling-exposed"
    assert met.exposure_class("heldout_sr386", "sr1482_m") == "target-primary-exposed"
    assert met.exposure_class("all_conventional", "sr1482_m") == ("target-primary-exposed")


def test_score_roster_is_eight_by_two_plus_all_conventional_and_primary_roles() -> None:
    jobs = met.score_jobs()

    # 9 scorers x 2 metastatic targets x 3 seeds, plus 3 role jobs x 3 seeds.
    assert len(jobs) == 63
    for scorer in met.ALL_SCORERS:
        for target in met.MET_TARGETS:
            assert sum(job[:2] == (scorer, target) for job in jobs) == 3
    assert ("heldout_rih", "rih_primary", 42) in jobs
    assert ("heldout_surgen", "sr1482_primary", 43) in jobs
    assert ("heldout_sr1482", "sr1482_primary", 44) in jobs
    assert not any(
        scorer == met.ALL_CONVENTIONAL_SCORER and target.endswith("primary")
        for scorer, target, _ in jobs
    )


def test_analysis_keeps_85_style_standalone_separate_from_common_model_population() -> None:
    patient_scores, primary_scores, dual = _analysis_inputs()

    results = met.analyse_patient_scores(
        patient_scores,
        primary_scores,
        dual,
        n_bootstrap=100,
        bootstrap_seed=41,
        enforce_live_census=False,
    )

    assert results["confirmatory_family_naive"]["rih_m"]["n"] == 12
    matrix = results["heldout_source_composition_matrix"]
    assert matrix["complete_eight_model_matrix"] is True
    assert set(matrix["scorers"]) == set(met.HELDOUT_SCORERS)
    assert met.ALL_CONVENTIONAL_SCORER not in matrix["scorers"]
    assert matrix["targets"]["rih_m"]["n_common"] == 10
    assert len(matrix["targets"]["rih_m"]["metrics"]) == 8
    assert all(block["n"] == 10 for block in matrix["targets"]["rih_m"]["metrics"].values())
    assert results["all_conventional_deployment_sensitivity"]["rih_m"]["n"] == 10
    assert results["all_conventional_deployment_sensitivity"]["sr1482_m"]["n"] == 12
    assert results["primary_to_metastatic_role_contrasts"][
        "all_conventional_role_delta"
    ].startswith("not estimated")


def test_all_pairwise_rih_heldout_contrasts_share_leakage_free_patients() -> None:
    patient_scores, primary_scores, dual = _analysis_inputs()
    results = met.analyse_patient_scores(
        patient_scores,
        primary_scores,
        dual,
        n_bootstrap=40,
        bootstrap_seed=9,
        enforce_live_census=False,
    )

    rih = results["heldout_source_composition_matrix"]["targets"]["rih_m"]
    # C(8, 2) complete descriptive contrast roster.
    assert len(rih["paired_auroc_contrasts"]) == 28
    assert all(
        block["paired_shared_patient_bootstrap"] is True
        for block in rih["paired_auroc_contrasts"].values()
    )
    assert (
        results["all_conventional_deployment_sensitivity"]["rih_m"]["paired_minus_family_naive"][
            "common_patients"
        ]
        == 10
    )


def test_organ_macro_uses_confirmatory_scorers_and_declared_sr1482_sensitivity() -> None:
    patient_scores, primary_scores, dual = _analysis_inputs()

    first = met.analyse_patient_scores(
        patient_scores,
        primary_scores,
        dual,
        n_bootstrap=75,
        bootstrap_seed=27,
        enforce_live_census=False,
    )
    second = met.analyse_patient_scores(
        patient_scores,
        primary_scores,
        dual,
        n_bootstrap=75,
        bootstrap_seed=27,
        enforce_live_census=False,
    )

    assert first == second
    organ = first["liver_non_liver"]
    assert organ["confirmatory_family_naive"]["cohorts"]["rih_m"]["scorer"] == ("heldout_rih")
    assert organ["confirmatory_family_naive"]["cohorts"]["sr1482_m"]["scorer"] == ("heldout_surgen")
    assert (
        organ["sr1482_sibling_exposed_sensitivity"]["cohorts"]["sr1482_m"]["scorer"]
        == "heldout_sr1482"
    )
    assert all(
        block["shared_cohort_organ_kras_bootstrap"] is True
        for block in organ["paired_sensitivity_contrasts"].values()
    )


def test_confirmatory_bootstrap_stream_is_independent_of_exploratory_matrix_size() -> None:
    patient_scores, primary_scores, dual = _analysis_inputs()
    first = met.analyse_patient_scores(
        patient_scores,
        primary_scores,
        dual,
        n_bootstrap=80,
        bootstrap_seed=19,
        enforce_live_census=False,
    )
    # Changing the leakage-free descriptive-matrix roster must not consume a
    # different prefix of the confirmatory bootstrap stream.
    expanded_dual = {*dual, "rih:0:1"}
    second = met.analyse_patient_scores(
        patient_scores,
        primary_scores,
        expanded_dual,
        n_bootstrap=80,
        bootstrap_seed=19,
        enforce_live_census=False,
    )

    assert first["confirmatory_family_naive"] == second["confirmatory_family_naive"]
    assert first["confirmatory_conclusion"] == second["confirmatory_conclusion"]


def test_analysis_fails_closed_on_incomplete_heldout_matrix() -> None:
    patient_scores, primary_scores, dual = _analysis_inputs()
    del patient_scores["heldout_tcga_read"]

    with pytest.raises(met.ContractError, match="Scorer roster"):
        met.analyse_patient_scores(
            patient_scores,
            primary_scores,
            dual,
            n_bootstrap=10,
            enforce_live_census=False,
        )


def test_patient_aggregation_averages_three_seeds_then_slides_and_applies_platt() -> None:
    manifest = pd.DataFrame(
        {
            "slide_id": ["A1", "A2", "B1"],
            "patient_id": ["A", "A", "B"],
            "specimen_role": ["metastatic"] * 3,
            "subcohort": ["SR1482"] * 3,
            "liver_class": ["liver", "liver", "non_liver"],
        }
    )
    outcomes = manifest.assign(target_label=[1, 1, 0])
    frames = []
    for seed_index, seed in enumerate(met.SEEDS):
        frames.append(
            pd.DataFrame(
                {
                    "slide_id": manifest["slide_id"],
                    "seed": seed,
                    "fold": 0,
                    "logit": [1.0 + seed_index, 3.0 + seed_index, -1.0 + seed_index],
                }
            )
        )

    patient = met.aggregate_patient_scores(
        frames, manifest, outcomes, calibrator={"a": -0.2, "b": 0.5}
    ).set_index("patient_id")

    # Seed means A1=2, A2=4; patient mean=3. B's seed mean is 0.
    assert patient.loc["A", "mean_logit"] == pytest.approx(3.0)
    assert patient.loc["A", "n_slides"] == 2
    assert patient.loc["B", "mean_logit"] == pytest.approx(0.0)
    assert patient.loc["A", "prob_source_calibrated"] == pytest.approx(
        float(sigmoid(np.array([-0.2 + 0.5 * 3.0]))[0])
    )


def test_cli_requires_explicit_new_model_roots_before_contract_seal() -> None:
    parser = met.build_parser()
    for command in ("preflight", "manifest"):
        args = parser.parse_args(
            [
                command,
                "--sibling-root",
                "/read-only/e2ad",
                "--all-conventional-root",
                "/read-only/cpht",
            ]
        )
        assert args.command == command
    for command in ("score", "seal", "report"):
        assert parser.parse_args([command]).command == command


def test_refit_validation_rejects_requested_but_under_run_fit() -> None:
    result = {
        "actual_optimizer_steps": met.STEP_BUDGET - 1,
        "refit_max_steps": met.STEP_BUDGET,
        "batch_size": 1,
        "accumulate_grad_batches": 1,
        "seed": 42,
        "sampling_seed": 42,
        "lr_scheduler": "cosine",
        "lr_scheduler_interval": "step",
        "lr_scheduler_total_steps": met.STEP_BUDGET,
        "train_sampling_strategy": "patient_natural",
        "sample_weight_column": None,
        "class_weights": None,
        "dataset_max_instances": met.CAP,
        "max_instances": None,
        "eval_full_bags": True,
    }

    with pytest.raises(met.ContractError, match="actual_optimizer_steps"):
        met._validate_completed_result(  # noqa: SLF001
            {"result": result}, "all_conventional", 42
        )


def test_source_population_guard_checks_holdout_and_mutant_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.csv"
    frame = pd.DataFrame(
        {
            "slide_id": ["A", "B", "C"],
            "patient_id": ["P1", "P2", "P3"],
            "target_label": [0, 1, 0],
            "cohort": ["RIH", "CPTAC", "SurGen"],
            "subcohort": ["RIH-Colon", "CPTAC-COAD", "SR1482"],
            "specimen_role": ["primary"] * 3,
        }
    )
    frame.to_csv(path, index=False)
    monkeypatch.setitem(met.EXPECTED_SOURCE_PATIENTS, "heldout_tcga", 3)
    monkeypatch.setitem(met.EXPECTED_SOURCE_MUTANTS, "heldout_tcga", 1)
    met._validate_source_population(path, "heldout_tcga")  # noqa: SLF001

    frame.loc[0, "cohort"] = "TCGA"
    frame.to_csv(path, index=False)
    with pytest.raises(met.ContractError, match="leaked"):
        met._validate_source_population(path, "heldout_tcga")  # noqa: SLF001


def test_report_parquet_publish_is_exactly_resumable(tmp_path: Path) -> None:
    path = tmp_path / "patients.parquet"
    frame = pd.DataFrame({"patient_id": ["A", "B"], "mean_logit": [0.1, 0.2]})
    met._publish_parquet_resumable(path, frame)  # noqa: SLF001
    met._publish_parquet_resumable(path, frame.copy())  # noqa: SLF001

    with pytest.raises(FileExistsError, match="non-identical"):
        met._publish_parquet_resumable(  # noqa: SLF001
            path, frame.assign(mean_logit=[0.1, 0.3])
        )


@pytest.mark.skipif(
    not all(path.is_file() for path in met.TARGET_SOURCES.values()),
    reason="governed metastatic manifests unavailable",
)
def test_live_input_manifest_census_and_dual_role_guard() -> None:
    frames, sources, dual = met.build_input_manifests()

    assert set(frames) == set(met.TARGET_SOURCES)
    assert all("target_label" not in frame.columns for frame in frames.values())
    assert frames["rih_m"]["patient_id"].nunique() == 85
    assert frames["sr1482_m"]["patient_id"].nunique() == 74
    assert len(dual) == 8
    assert sources["rih_m"]["n_mutant"] == 37
    assert sources["sr1482_m"]["n_mutant"] == 30
