from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_surgen_subcohort_gap  # noqa: E402


def _synthetic_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for subcohort, group_shift in (("SR386", 0.15), ("SR1482", -0.15)):
        for label in (0, 1):
            for index in range(14):
                stage = ("I", "II", "III", "IV")[index % 4]
                if (subcohort == "SR1482" and index % 4 == 0) or (
                    subcohort == "SR386" and index == 0
                ):
                    stage = np.nan
                score = 0.8 * label + group_shift + rng.normal(0, 0.8)
                rows.append(
                    {
                        "patient_id": f"{subcohort}-{label}-{index}",
                        "subcohort": subcohort,
                        "label": label,
                        "mean_logit": score,
                        "age_at_diagnosis": 62 + 5 * group_shift + rng.normal(0, 7),
                        "tumor_site_group": ("colon", "rectum")[index % 2],
                        "msi_dmmr": ("MSS/pMMR", "MSI/dMMR")[index % 5 == 0],
                        "braf": ("wild_type", "mutant")[index % 6 == 0],
                        "n_slides_true": 1 + (index % 5 == 0),
                        "mean_patch_count": 10_000 + 500 * index + rng.normal(0, 300),
                        "mean_tissue_area": 40 + index + rng.normal(0, 2),
                        "sex": ("female", "male")[index % 2],
                        aim2_surgen_subcohort_gap.STAGE_COLUMN: stage,
                    }
                )
    return pd.DataFrame(rows)


def test_weighted_auroc_matches_sklearn_for_unit_weights_and_ties() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.0, 0.5, 1.0, 0.5, 1.0, 1.0])

    observed = aim2_surgen_subcohort_gap._weighted_auroc(labels, scores, np.ones(len(labels)))

    assert observed == pytest.approx(roc_auc_score(labels, scores))


def test_load_joins_current_filled_stage_and_audits_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patients = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "subcohort": ["SR386", "SR386", "SR1482", "SR1482"],
            "label": [0, 1, 0, 1],
            "mean_logit": [-1.0, 1.0, -0.5, 0.5],
            "prob_raw": [0.2, 0.8, 0.3, 0.7],
        }
    )
    monkeypatch.setattr(
        aim2_surgen_subcohort_gap.aim2_loco_transport,
        "seed_ensemble",
        lambda *_args, **_kwargs: (
            patients.copy(),
            {seed: patients.copy() for seed in aim2_surgen_subcohort_gap.aim2_loco_transport.SEEDS},
        ),
    )
    development = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3", "p4"],
            "age_at_diagnosis": [60, 60, 70, 55, 65],
            "tumor_site_group": ["colon", "colon", "rectum", "colon", "rectum"],
            "msi_dmmr": ["MSS/pMMR"] * 5,
            "braf": ["wild_type"] * 5,
            "stage_class": ["unknown"] * 5,
            "sex": ["female", "female", "male", "female", "male"],
            "patch_count": [100, 120, 130, 140, 150],
            "tissue_area_mm2": [10, 12, 13, 14, 15],
        }
    )
    label_source = pd.DataFrame(
        {
            "patient_uid": ["p1", "p1", "p2", "p3", "p4", "p4"],
            "specimen_role": ["primary", "primary", "primary", "primary", "primary", "metastatic"],
            "subcohort": ["SR386", "SR386", "SR386", "SR1482", "SR1482", "SR1482"],
            aim2_surgen_subcohort_gap.STAGE_COLUMN: ["II", "II", "III", np.nan, "IV", "I"],
            aim2_surgen_subcohort_gap.STAGE_SOURCE_COLUMN: [
                "original",
                "original",
                "original",
                "unknown",
                "derived_ajcc_from_tnm",
                "original",
            ],
        }
    )
    development_path = tmp_path / "development.csv"
    label_source_path = tmp_path / "labels.csv"
    development.to_csv(development_path, index=False)
    label_source.to_csv(label_source_path, index=False)
    monkeypatch.setattr(aim2_surgen_subcohort_gap.paths, "DEV_MANIFEST", development_path)
    monkeypatch.setattr(aim2_surgen_subcohort_gap.paths, "LABEL_SOURCE", label_source_path)

    loaded = aim2_surgen_subcohort_gap.load(8192)
    coverage = aim2_surgen_subcohort_gap.stage_coverage(loaded)

    assert len(loaded) == 4
    assert loaded.set_index("patient_id").loc["p4", aim2_surgen_subcohort_gap.STAGE_COLUMN] == "IV"
    assert pd.isna(loaded.set_index("patient_id").loc["p3", aim2_surgen_subcohort_gap.STAGE_COLUMN])
    assert coverage["by_subcohort"]["SR386"]["n_known"] == 2
    assert coverage["by_subcohort"]["SR1482"]["n_known"] == 1
    assert coverage["by_subcohort"]["SR1482"]["n_unknown"] == 1


def test_standardized_delta_refits_propensity_and_reports_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _synthetic_frame()
    covariates = ["age_at_diagnosis", "tumor_site_group", aim2_surgen_subcohort_gap.STAGE_COLUMN]
    original = aim2_surgen_subcohort_gap._fit_standardization_once
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aim2_surgen_subcohort_gap, "_fit_standardization_once", counted)
    result = aim2_surgen_subcohort_gap.standardized_delta(
        frame,
        covariates,
        seed=19,
        n_boot=20,
    )

    assert calls == 21
    assert result["n_propensity_fits"] == 21
    assert result["n_bootstrap"] == 20
    assert len(result["delta_ci"]) == 2
    assert result["delta_ci"][0] <= result["delta_ci"][1]
    assert "diagnostics" in result
    diagnostics = result["diagnostics"]
    assert diagnostics["estimand"] == "ATT on the SR386 covariate distribution"
    assert diagnostics["SR1482_effective_sample_size"] > 0
    assert diagnostics["SR1482_max_weight"] > 0
    assert set(diagnostics["propensity_by_subcohort"]) == {"SR386", "SR1482"}
    assert any(
        name.endswith("_unknown")
        for name in result["propensity_model"]["design_columns"]
    )
    assert "overlap_assessment" in result
    assert isinstance(result["overlap_assessment"]["estimable_for_inference"], bool)


def test_overlap_assessment_blocks_severe_positivity_failure() -> None:
    diagnostics = {
        "propensity_by_subcohort": {
            "SR386": {},
            "SR1482": {
                "fraction_clipped": 0.50,
                "fraction_outside_empirical_common_support": 0.84,
            },
        },
        "SR1482_ess_fraction": 0.06,
        "SR1482_max_weight": 29.0,
    }

    assessment = aim2_surgen_subcohort_gap._overlap_assessment(diagnostics)

    assert assessment["estimable_for_inference"] is False
    assert assessment["status"] == "limited overlap; positivity diagnostic only"
    assert set(assessment["failed_checks"]) == {
        "clipping",
        "common_support",
        "effective_sample_size",
        "maximum_weight",
    }


def test_standardized_delta_is_deterministic() -> None:
    frame = _synthetic_frame()
    covariates = ["age_at_diagnosis", "tumor_site_group", aim2_surgen_subcohort_gap.STAGE_COLUMN]

    first = aim2_surgen_subcohort_gap.standardized_delta(frame, covariates, seed=29, n_boot=12)
    second = aim2_surgen_subcohort_gap.standardized_delta(frame, covariates, seed=29, n_boot=12)

    assert first == second


def test_standardization_failure_aborts_instead_of_returning_partial_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _synthetic_frame()
    original = aim2_surgen_subcohort_gap._fit_standardization_once
    calls = 0

    def fail_first_bootstrap(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic propensity failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(aim2_surgen_subcohort_gap, "_fit_standardization_once", fail_first_bootstrap)
    with pytest.raises(RuntimeError, match="bootstrap replicate 0"):
        aim2_surgen_subcohort_gap.standardized_delta(
            frame,
            ["age_at_diagnosis", "tumor_site_group"],
            seed=31,
            n_boot=3,
        )


def test_standardization_ladder_propagates_any_model_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _synthetic_frame()

    def fail(*_args, **_kwargs):
        raise RuntimeError("do not publish")

    monkeypatch.setattr(aim2_surgen_subcohort_gap, "standardized_delta", fail)
    with pytest.raises(RuntimeError, match="do not publish"):
        aim2_surgen_subcohort_gap.run_standardizations(frame, n_boot=5, seed=3)
