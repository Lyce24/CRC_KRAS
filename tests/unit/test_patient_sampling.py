from __future__ import annotations

from collections import Counter

import pytest

from oceanpath.datasets.sampling import PatientSlideSampler


def _metadata():
    slide_ids: list[str] = []
    labels: list[int] = []
    patients: list[str] = []
    cohorts: list[str] = []
    for cohort in ("A", "B"):
        for index in range(6):
            patient = f"{cohort}{index}"
            label = int(index < (2 if cohort == "A" else 4))
            n_slides = 2 if patient == "A0" else 1
            for slide in range(n_slides):
                slide_ids.append(f"{patient}_s{slide}")
                labels.append(label)
                patients.append(patient)
                cohorts.append(cohort)
    return slide_ids, labels, patients, cohorts


def _sampler(strategy: str, *, seed: int = 7):
    slide_ids, labels, patients, cohorts = _metadata()
    return PatientSlideSampler(
        slide_ids=slide_ids,
        labels=labels,
        patient_ids=patients,
        cohorts=cohorts,
        strategy=strategy,
        target_positive_prevalence=0.4,
        seed=seed,
    )


def test_patient_natural_visits_each_patient_once_despite_duplicate_slides():
    sampler = _sampler("patient_natural")
    indices = list(sampler)
    drawn_patients = [sampler.patient_ids[index] for index in indices]

    assert len(indices) == 12
    assert Counter(drawn_patients) == Counter(
        {f"{cohort}{i}": 1 for cohort in "AB" for i in range(6)}
    )
    assert sum(index in (0, 1) for index in indices) == 1


def test_cohort_balanced_uses_equal_integer_quotas():
    sampler = _sampler("cohort_balanced")
    indices = list(sampler)
    counts = Counter(sampler.cohorts[index] for index in indices)
    assert counts == {"A": 6, "B": 6}


def test_cohort_label_balanced_hits_requested_prevalence_with_rounding():
    sampler = _sampler("cohort_label_balanced")
    indices = list(sampler)
    cells = Counter((sampler.cohorts[index], sampler.labels[index]) for index in indices)
    assert cells == {("A", 0): 4, ("A", 1): 2, ("B", 0): 4, ("B", 1): 2}


def test_sampling_is_reproducible_by_seed_and_changes_by_epoch():
    first = _sampler("cohort_label_balanced", seed=19)
    second = _sampler("cohort_label_balanced", seed=19)
    epoch_zero = list(first)
    assert epoch_zero == list(second)
    assert list(first) != epoch_zero
    first.set_epoch(0)
    assert list(first) == epoch_zero


def test_summary_records_realized_epoch_without_slide_overweighting():
    sampler = _sampler("patient_natural")
    list(sampler)
    summary = sampler.summary()
    assert summary["n_training_slides"] == 13
    assert summary["n_training_patients"] == 12
    assert summary["samples_per_epoch"] == 12
    assert summary["realized_epochs"][0]["n_unique_patients"] == 12


@pytest.mark.parametrize(
    ("field", "message"),
    [("label", "conflicting labels"), ("cohort", "crosses cohorts")],
)
def test_patient_conflicts_fail_closed(field: str, message: str):
    slide_ids, labels, patients, cohorts = _metadata()
    duplicate_index = patients.index("A0", 1)
    if field == "label":
        labels[duplicate_index] = 1 - labels[duplicate_index]
    else:
        cohorts[duplicate_index] = "B"
    with pytest.raises(ValueError, match=message):
        PatientSlideSampler(
            slide_ids=slide_ids,
            labels=labels,
            patient_ids=patients,
            cohorts=cohorts,
            strategy="patient_natural",
        )


def test_cohort_label_balance_rejects_empty_cell():
    slide_ids = ["a", "b", "c"]
    with pytest.raises(ValueError, match="both labels"):
        PatientSlideSampler(
            slide_ids=slide_ids,
            labels=[0, 0, 1],
            patient_ids=["a", "b", "c"],
            cohorts=["A", "A", "B"],
            strategy="cohort_label_balanced",
        )


def _multiclass_metadata(n_classes: int = 4):
    """One patient per slide, labels cycling through `n_classes`."""
    slide_ids, labels, patients, cohorts = [], [], [], []
    for cohort in ("A", "B"):
        for index in range(8):
            patient = f"{cohort}{index}"
            slide_ids.append(f"{patient}_s0")
            labels.append(index % n_classes)
            patients.append(patient)
            cohorts.append(cohort)
    return slide_ids, labels, patients, cohorts


@pytest.mark.parametrize("strategy", ["patient_natural", "cohort_balanced"])
def test_label_agnostic_strategies_accept_multiclass(strategy: str):
    """Neither strategy reads the label, so a multiclass target must work.

    E1e's 4-class cohort probe needs exactly this; the binary restriction used
    to be enforced for every strategy, which blocked it for no reason.
    """
    slide_ids, labels, patients, cohorts = _multiclass_metadata()
    sampler = PatientSlideSampler(
        slide_ids=slide_ids,
        labels=labels,
        patient_ids=patients,
        cohorts=cohorts,
        strategy=strategy,
        seed=3,
    )
    draws = list(iter(sampler))
    assert len(draws) == len(set(patients))
    summary = sampler.summary()
    assert summary["label_values"] == [0, 1, 2, 3]
    assert sum(summary["empirical_label_counts_by_cohort"].values()) == len(set(patients))


def test_cohort_label_balanced_still_requires_binary():
    """The one strategy that needs a 'positive' class must keep refusing."""
    slide_ids, labels, patients, cohorts = _multiclass_metadata()
    with pytest.raises(ValueError, match="binary labels 0/1 only"):
        PatientSlideSampler(
            slide_ids=slide_ids,
            labels=labels,
            patient_ids=patients,
            cohorts=cohorts,
            strategy="cohort_label_balanced",
        )


def test_multiclass_draws_are_deterministic_per_seed():
    slide_ids, labels, patients, cohorts = _multiclass_metadata()

    def draw(seed: int) -> list[int]:
        return list(
            iter(
                PatientSlideSampler(
                    slide_ids=slide_ids,
                    labels=labels,
                    patient_ids=patients,
                    cohorts=cohorts,
                    strategy="patient_natural",
                    seed=seed,
                )
            )
        )

    assert draw(11) == draw(11)
    assert draw(11) != draw(12)
