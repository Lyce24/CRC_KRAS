from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_paired_specimen_concordance


def _pairs(primary: list[float], metastatic: list[float]) -> pd.DataFrame:
    primary_array = np.asarray(primary, dtype=float)
    metastatic_array = np.asarray(metastatic, dtype=float)
    return pd.DataFrame(
        {
            "patient_id": [f"p{index}" for index in range(len(primary))],
            "cohort": "RIH",
            "mean_logit_primary": primary_array,
            "mean_logit_metastatic": metastatic_array,
            "delta_logit_metastatic_minus_primary": metastatic_array - primary_array,
        }
    )


def test_exact_sign_test_is_exact_and_excludes_ties() -> None:
    result = aim2_paired_specimen_concordance.exact_sign_test(np.array([1, 2, 3, 4, 5, -1, 0], dtype=float))

    assert result["positive"] == 5
    assert result["negative"] == 1
    assert result["ties_excluded"] == 1
    assert result["n_nonzero"] == 6
    assert result["two_sided_exact_p"] == pytest.approx(14 / 64)


def test_ccc_penalizes_level_shift_that_pearson_ignores() -> None:
    primary = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    metastatic = primary + 2.0

    assert aim2_paired_specimen_concordance._pearson(primary, metastatic) == pytest.approx(1.0)
    assert aim2_paired_specimen_concordance.lins_concordance_correlation(primary, metastatic) == pytest.approx(0.5)


def test_paired_bootstrap_is_deterministic_and_resamples_patient_pairs() -> None:
    pairs = _pairs(
        [-2.0, -1.3, -0.7, -0.1, 0.4, 1.0, 1.5, 2.1],
        [-1.5, -1.0, -0.4, 0.2, 0.7, 1.4, 1.8, 2.4],
    )

    first = aim2_paired_specimen_concordance.paired_bootstrap(pairs, n_bootstrap=250, seed=17)
    second = aim2_paired_specimen_concordance.paired_bootstrap(pairs, n_bootstrap=250, seed=17)

    assert first == second
    assert first["n_pairs"] == 8
    assert first["mean_logit_shift"] == pytest.approx(0.3375)
    assert first["bootstrap"]["sampling_unit"] == "patient pair"
    assert first["bootstrap"]["n_bootstrap_requested"] == 250
    mean_interval = first["bootstrap"]["intervals"]["mean_logit_shift"]
    assert mean_interval["n_bootstrap_valid"] == 250
    assert mean_interval["ci_95_percentile"][0] > 0
    assert "auroc" not in first


def test_make_pair_table_preserves_role_labels_and_flags_discordance() -> None:
    primary = pd.DataFrame(
        {
            "patient_id": ["same", "changed"],
            "label": [1, 1],
            "mean_logit": [2.0, 1.0],
            "prob_raw": [0.88, 0.73],
        }
    )
    metastatic = pd.DataFrame(
        {
            "patient_id": ["same", "changed"],
            "label": [1, 0],
            "mean_logit": [1.5, -0.5],
            "prob_raw": [0.82, 0.38],
        }
    )

    pairs = aim2_paired_specimen_concordance.make_pair_table(primary, metastatic, cohort="RIH")

    changed = pairs.set_index("patient_id").loc["changed"]
    assert bool(changed["label_concordant"]) is False
    assert changed["label_primary"] == 1
    assert changed["label_metastatic"] == 0
    assert changed["delta_logit_metastatic_minus_primary"] == pytest.approx(-1.5)


def test_make_pair_table_fails_closed_on_mismatched_patient_sets() -> None:
    primary = pd.DataFrame(
        {"patient_id": ["a"], "label": [0], "mean_logit": [0.1], "prob_raw": [0.52]}
    )
    metastatic = pd.DataFrame(
        {"patient_id": ["b"], "label": [0], "mean_logit": [0.2], "prob_raw": [0.55]}
    )

    with pytest.raises(ValueError, match="do not match exactly"):
        aim2_paired_specimen_concordance.make_pair_table(primary, metastatic, cohort="RIH")


def test_pair_bootstrap_fails_closed_on_duplicate_patient_rows() -> None:
    pairs = _pairs([0.0, 1.0], [0.2, 1.1])
    pairs.loc[1, "patient_id"] = pairs.loc[0, "patient_id"]

    with pytest.raises(ValueError, match="one row per patient"):
        aim2_paired_specimen_concordance.paired_bootstrap(pairs, n_bootstrap=10)


def test_verified_population_requires_all_eight_rih_and_one_tcga(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [f"rih-{index}" for index in range(8)] + ["tcga-0"]
    cohorts = ["RIH"] * 8 + ["TCGA"]
    primary = pd.DataFrame(
        {
            "patient_id": ids,
            "cohort": cohorts,
            "target_label": [1, 0, 1, 0, 1, 0, 1, 1, 1],
        }
    )
    metastatic = primary.copy()
    metastatic.loc[metastatic["patient_id"].eq("rih-7"), "target_label"] = 0
    monkeypatch.setattr(aim2_paired_specimen_concordance.population, "eligible", lambda: primary.copy())
    monkeypatch.setattr(
        aim2_paired_specimen_concordance.population, "add_context_columns", lambda frame: frame.copy()
    )
    monkeypatch.setattr(
        aim2_paired_specimen_concordance.population, "eligible_metastatic", lambda: metastatic.copy()
    )

    _, _, audit = aim2_paired_specimen_concordance.verified_pair_population()

    assert audit["n_pairs"] == 9
    assert audit["cohort_counts"] == {"RIH": 8, "TCGA": 1}
    assert audit["n_label_discordant"] == 1
    assert audit["label_discordant_patient_ids"] == ["rih-7"]


def test_verified_population_fails_if_a_pair_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = pd.DataFrame(
        {
            "patient_id": [f"rih-{index}" for index in range(8)],
            "cohort": "RIH",
            "target_label": [0, 1] * 4,
        }
    )
    metastatic = primary.copy()
    monkeypatch.setattr(aim2_paired_specimen_concordance.population, "eligible", lambda: primary.copy())
    monkeypatch.setattr(
        aim2_paired_specimen_concordance.population, "add_context_columns", lambda frame: frame.copy()
    )
    monkeypatch.setattr(
        aim2_paired_specimen_concordance.population, "eligible_metastatic", lambda: metastatic.copy()
    )

    with pytest.raises(RuntimeError, match="population changed"):
        aim2_paired_specimen_concordance.verified_pair_population()


def test_native_score_validation_rejects_probability_only_scores() -> None:
    manifest = pd.DataFrame({"slide_id": ["slide-1"]})
    scores = pd.DataFrame({"slide_id": ["slide-1"], "seed": [42], "prob_1": [0.9]})

    with pytest.raises(RuntimeError, match="lack columns"):
        aim2_paired_specimen_concordance._validate_native_scores(scores, manifest, seed=42)
