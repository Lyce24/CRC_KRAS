"""Patient-level samplers for cohort-shortcut experiments.

The model consumes one slide bag at a time, but the experimental unit is a
patient.  This sampler assigns quotas to patients, then chooses one of each
drawn patient's eligible slides uniformly.  It therefore prevents patients
with multiple slides from receiving extra training mass.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from torch.utils.data import Sampler

PATIENT_SAMPLING_STRATEGIES = frozenset(
    {"patient_natural", "cohort_balanced", "cohort_label_balanced"}
)


def _equal_quotas(total: int, groups: Sequence[str], epoch: int) -> dict[str, int]:
    """Allocate ``total`` integer draws as evenly as possible across groups."""
    ordered = sorted(groups)
    base, remainder = divmod(total, len(ordered))
    quotas = {group: base for group in ordered}
    for offset in range(remainder):
        quotas[ordered[(epoch + offset) % len(ordered)]] += 1
    return quotas


def _cyclic_draw(pool: np.ndarray, count: int, rng: np.random.Generator) -> list[str]:
    """Draw from shuffled cycles, exhausting a group before repeating it."""
    if count < 0 or len(pool) == 0:
        raise ValueError("Sampling pools must be non-empty and quotas non-negative")
    draws: list[str] = []
    while len(draws) < count:
        cycle = rng.permutation(pool).tolist()
        draws.extend(str(value) for value in cycle[: count - len(draws)])
    return draws


class PatientSlideSampler(Sampler[int]):
    """Draw a fixed patient budget and return one slide index per patient draw.

    ``patient_natural`` visits every training patient exactly once per epoch.
    The balanced policies allocate the same total number of draws across
    cohorts or cohort-label cells and use shuffled cycles within each cell.
    Thus balance is exact up to integer rounding, repeated exposure is kept to
    the minimum needed by a quota, and the optimizer step count is identical
    across policies.
    """

    def __init__(
        self,
        *,
        slide_ids: Sequence[str],
        labels: Sequence[int],
        patient_ids: Sequence[str],
        cohorts: Sequence[str],
        strategy: str,
        target_positive_prevalence: float = 0.4,
        num_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        lengths = {len(slide_ids), len(labels), len(patient_ids), len(cohorts)}
        if len(lengths) != 1 or not slide_ids:
            raise ValueError("Sampling metadata must be non-empty and have equal lengths")
        if strategy not in PATIENT_SAMPLING_STRATEGIES:
            raise ValueError(f"strategy={strategy!r} not in {sorted(PATIENT_SAMPLING_STRATEGIES)}")
        if not 0.0 < float(target_positive_prevalence) < 1.0:
            raise ValueError("target_positive_prevalence must be in (0, 1)")

        self.slide_ids = tuple(str(value) for value in slide_ids)
        self.labels = tuple(int(value) for value in labels)
        self.patient_ids = tuple(str(value).strip() for value in patient_ids)
        self.cohorts = tuple(str(value).strip() for value in cohorts)
        self.strategy = strategy
        self.target_positive_prevalence = float(target_positive_prevalence)
        self.seed = int(seed)
        self._next_epoch = 0
        self._history: list[dict[str, Any]] = []

        if any(not value for value in self.patient_ids) or any(not value for value in self.cohorts):
            raise ValueError("Patient and cohort identifiers must not be blank")
        self._label_values = tuple(sorted(set(self.labels)))
        # Only the label-quota strategy needs a binary label: it has to know
        # which class is "positive" to hit target_positive_prevalence.
        # patient_natural draws patients uniformly and cohort_balanced draws
        # within cohort, and neither reads the label at all — so a multiclass
        # probe (e.g. the E1e 4-class cohort head) can use them unchanged.
        if strategy == "cohort_label_balanced" and set(self.labels) - {0, 1}:
            raise ValueError(
                "cohort_label_balanced supports binary labels 0/1 only "
                f"(got labels {self._label_values}); use patient_natural or "
                "cohort_balanced for a multiclass target"
            )

        slides_by_patient: dict[str, list[int]] = {}
        label_by_patient: dict[str, int] = {}
        cohort_by_patient: dict[str, str] = {}
        for index, (patient, label, cohort) in enumerate(
            zip(self.patient_ids, self.labels, self.cohorts, strict=True)
        ):
            if patient in label_by_patient and label_by_patient[patient] != label:
                raise ValueError(f"Patient {patient!r} has conflicting labels")
            if patient in cohort_by_patient and cohort_by_patient[patient] != cohort:
                raise ValueError(f"Patient {patient!r} crosses cohorts")
            slides_by_patient.setdefault(patient, []).append(index)
            label_by_patient[patient] = label
            cohort_by_patient[patient] = cohort

        self._slides_by_patient = {
            patient: np.asarray(indices, dtype=np.int64)
            for patient, indices in slides_by_patient.items()
        }
        self._label_by_patient = label_by_patient
        self._cohort_by_patient = cohort_by_patient
        self._patients = np.asarray(sorted(slides_by_patient), dtype=object)
        self._cohort_names = tuple(sorted(set(cohort_by_patient.values())))
        self.num_samples = len(self._patients) if num_samples is None else int(num_samples)
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")

        self._pools: dict[tuple[str, int], np.ndarray] = {}
        for cohort in self._cohort_names:
            for label in self._label_values:
                pool = [
                    patient
                    for patient in self._patients
                    if cohort_by_patient[str(patient)] == cohort
                    and label_by_patient[str(patient)] == label
                ]
                self._pools[(cohort, label)] = np.asarray(pool, dtype=object)
        if strategy == "cohort_label_balanced":
            missing = [(cohort, label)
                       for cohort in self._cohort_names for label in (0, 1)
                       if len(self._pools.get((cohort, label), ())) == 0]
            if missing:
                raise ValueError(
                    "cohort_label_balanced requires both labels in every cohort; "
                    f"empty cells: {missing}"
                )

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic epoch used by the next iterator."""
        self._next_epoch = int(epoch)

    def _patient_draws(self, epoch: int, rng: np.random.Generator) -> list[str]:
        if self.strategy == "patient_natural":
            if self.num_samples == len(self._patients):
                return [str(value) for value in rng.permutation(self._patients)]
            return _cyclic_draw(self._patients, self.num_samples, rng)

        cohort_quotas = _equal_quotas(self.num_samples, self._cohort_names, epoch)
        draws: list[str] = []
        for cohort in self._cohort_names:
            cohort_quota = cohort_quotas[cohort]
            if self.strategy == "cohort_balanced":
                pool = np.concatenate([self._pools[(cohort, 0)], self._pools[(cohort, 1)]])
                draws.extend(_cyclic_draw(pool, cohort_quota, rng))
                continue

            positive_quota = int(math.floor(cohort_quota * self.target_positive_prevalence + 0.5))
            positive_quota = min(max(positive_quota, 1), cohort_quota - 1)
            draws.extend(_cyclic_draw(self._pools[(cohort, 1)], positive_quota, rng))
            draws.extend(_cyclic_draw(self._pools[(cohort, 0)], cohort_quota - positive_quota, rng))
        return [str(value) for value in rng.permutation(np.asarray(draws, dtype=object))]

    def __iter__(self) -> Iterator[int]:
        epoch = self._next_epoch
        self._next_epoch += 1
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        patients = self._patient_draws(epoch, rng)
        indices = [int(rng.choice(self._slides_by_patient[patient])) for patient in patients]

        cohort_counts = Counter(self._cohort_by_patient[patient] for patient in patients)
        cell_counts = Counter(
            (self._cohort_by_patient[patient], self._label_by_patient[patient])
            for patient in patients
        )
        self._history.append(
            {
                "epoch": epoch,
                "n_draws": len(indices),
                "n_unique_patients": len(set(patients)),
                "cohort_counts": dict(sorted(cohort_counts.items())),
                "cohort_label_counts": {
                    f"{cohort}|{label}": count
                    for (cohort, label), count in sorted(cell_counts.items())
                },
            }
        )
        return iter(indices)

    def summary(self) -> dict[str, Any]:
        """Return the frozen plan plus realized counts for completed epochs."""
        patient_counts = Counter(self._cohort_by_patient.values())
        positives = Counter(
            cohort
            for patient, cohort in self._cohort_by_patient.items()
            if self._label_by_patient[patient] == 1
        )
        # For a multiclass target "positive fraction" is not a meaningful
        # summary, so the full per-cohort label distribution is emitted too.
        label_counts: Counter = Counter(
            (cohort, self._label_by_patient[patient])
            for patient, cohort in self._cohort_by_patient.items()
        )
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "sampling_unit": "patient; one uniformly selected slide per draw",
            "quota_policy": "shuffled cyclic quotas; exact up to integer rounding",
            "samples_per_epoch": self.num_samples,
            "n_training_slides": len(self.slide_ids),
            "n_training_patients": len(self._patients),
            "target_positive_prevalence": (
                self.target_positive_prevalence
                if self.strategy == "cohort_label_balanced"
                else None
            ),
            "empirical_patient_counts": dict(sorted(patient_counts.items())),
            "empirical_positive_fraction_by_cohort": {
                cohort: positives[cohort] / patient_counts[cohort]
                for cohort in sorted(patient_counts)
            },
            "label_values": list(self._label_values),
            "empirical_label_counts_by_cohort": {
                f"{cohort}|{label}": count
                for (cohort, label), count in sorted(label_counts.items())
            },
            "realized_epochs": list(self._history),
        }


__all__ = ["PATIENT_SAMPLING_STRATEGIES", "PatientSlideSampler"]
