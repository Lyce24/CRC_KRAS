from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_metastatic_indomain_bound  # noqa: E402


def _patients(n_rih_mut=37, n_rih_wt=48, n_sg_mut=30, n_sg_wt=44) -> pd.DataFrame:
    rows = []
    for cohort, kras, count in (
        ("RIH", "mutant", n_rih_mut),
        ("RIH", "wild_type", n_rih_wt),
        ("SurGen", "mutant", n_sg_mut),
        ("SurGen", "wild_type", n_sg_wt),
    ):
        for i in range(count):
            rows.append(
                {
                    "patient_id": f"{cohort}_{kras}_{i:03d}",
                    "cohort": cohort,
                    "kras": kras,
                    "label": 1 if kras == "mutant" else 0,
                }
            )
    return pd.DataFrame(rows)


def test_outer_fold_deal_is_deterministic_and_balanced() -> None:
    patients = _patients()
    fold_a = aim2_metastatic_indomain_bound._deal_outer_folds(patients)
    fold_b = aim2_metastatic_indomain_bound._deal_outer_folds(patients)
    assert fold_a.equals(fold_b)
    sizes = fold_a.value_counts()
    assert sizes.max() - sizes.min() <= 2
    mutants = patients.groupby(fold_a)["label"].sum()
    assert mutants.max() - mutants.min() <= 2
    # patient-grouped by construction: each patient appears exactly once
    assert len(fold_a) == len(patients)


def test_carve_val_enforces_minimum_positive_patients() -> None:
    patients = _patients()
    fold = aim2_metastatic_indomain_bound._deal_outer_folds(patients)
    for fold_idx in range(aim2_metastatic_indomain_bound.N_FOLDS):
        flags = aim2_metastatic_indomain_bound._carve_val(patients, fold, fold_idx)
        val_mutants = int(flags[(patients["kras"] == "mutant")].sum())
        assert val_mutants >= aim2_metastatic_indomain_bound.MIN_VAL_POSITIVES
        # no validation patient may sit inside the outer test fold
        assert int(flags[fold == fold_idx].sum()) == 0


def test_carve_val_fails_loudly_when_underpowered() -> None:
    patients = _patients(n_rih_mut=5, n_rih_wt=48, n_sg_mut=5, n_sg_wt=44)
    fold = aim2_metastatic_indomain_bound._deal_outer_folds(patients)
    with pytest.raises(AssertionError, match="mutant validation patients"):
        aim2_metastatic_indomain_bound._carve_val(patients, fold, 0)


def test_hanley_mcneil_sample_sizes_are_sane() -> None:
    n_060 = aim2_metastatic_indomain_bound._hanley_mcneil_n(0.60, 0.42, 0.05, 0.80)
    n_065 = aim2_metastatic_indomain_bound._hanley_mcneil_n(0.65, 0.42, 0.05, 0.80)
    n_070 = aim2_metastatic_indomain_bound._hanley_mcneil_n(0.70, 0.42, 0.05, 0.80)
    assert n_060 > n_065 > n_070 > 0
    # the corrected null variance makes 0.60 undetectable at n=159
    assert n_060 > 159
    assert 50 < n_065 < 159
