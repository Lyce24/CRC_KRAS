"""Deterministic multi-marginal fold assignment (Aim1_Setup.md §4.4).

``StratifiedGroupKFold`` on KRAS alone leaves every other axis to chance, and
a fully crossed stratum (KRAS x subcohort x site x stage x MPP x technical
class) is mostly singleton cells at n=909. This module takes the third route
the plan asks for: assign patients greedily so that **several marginal
distributions** sit close to their population proportions in every fold, then
improve the assignment with a bounded deterministic swap search.

Balanced marginals (all label-independent except KRAS itself):

    kras              mutant / wild-type
    subcohort         TCGA-COAD / TCGA-READ / SR386
    site_class        Colon / Rectum
    stage_class       I-II / III-IV / unknown
    mpp_bin           native acquisition-resolution bin
    technical_class   the §4.5 per-subcohort technical axis
    n_slides_class    1 vs 2+ slides for that patient

The same machinery draws the 15% early-stopping carve-out inside each outer
fold's training pool, so the inner validation split is balanced too rather
than being a random 15%.

Nothing here is random: given the same patients and attributes it returns the
same assignment on every machine, which is what lets the fold manifest be
frozen before training.

Implementation note — all balancing counts live in one ``(level, group)``
matrix ``C`` with matching target matrix ``T``. Adding or removing one
patient changes the squared-error objective by a closed form,
``(x +/- 1 - t)^2 - (x - t)^2 = +/-2(x - t) + 1``, so both the greedy pass and
the O(n^2) swap search evaluate a move in a handful of array reads instead of
recomputing the objective.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

# Fold-size balance matters more than any single marginal: an undersized fold
# distorts every metric computed on it.
SIZE_WEIGHT = 4.0
SWAP_PASSES = 20


def patient_table(rows: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """One row per patient; each balancing column collapsed to its mode.

    A patient's slides normally agree on every balancing variable. Where two
    blocks of the same patient disagree (a technical class can differ between
    sections), the mode keeps the patient a single balancing unit.
    """
    table = rows.groupby("patient_id")[list(columns)].agg(
        lambda values: values.astype("string").fillna("NA").mode().iat[0]
    )
    return table.sort_index()


class _Balancer:
    """Flat level x group count matrix with closed-form move deltas."""

    def __init__(self, table: pd.DataFrame, columns: Sequence[str], shares: np.ndarray):
        self.columns = list(columns)
        self.n_groups = len(shares)
        self.n_patients = len(table)

        offsets: list[int] = []
        levels: list[list[str]] = []
        offset = 0
        for column in self.columns:
            values = sorted(table[column].unique().tolist())
            levels.append(values)
            offsets.append(offset)
            offset += len(values)
        self.levels = levels
        self.total_levels = offset

        # rows[p] = the flat level index of patient p on each balancing column
        self.rows = np.zeros((self.n_patients, len(self.columns)), dtype=np.intp)
        for position, (column, values, base) in enumerate(
            zip(self.columns, levels, offsets, strict=True)
        ):
            index_of = {value: base + rank for rank, value in enumerate(values)}
            self.rows[:, position] = table[column].map(index_of).to_numpy(dtype=np.intp)

        self.counts = np.zeros((self.total_levels, self.n_groups), dtype=np.float64)
        self.targets = np.zeros((self.total_levels, self.n_groups), dtype=np.float64)
        for column, values, base in zip(self.columns, levels, offsets, strict=True):
            totals = table[column].value_counts().reindex(values).to_numpy(dtype=np.float64)
            self.targets[base : base + len(values), :] = totals[:, None] * shares[None, :]

        self.sizes = np.zeros(self.n_groups, dtype=np.float64)
        self.target_sizes = self.n_patients * shares
        # Hard group capacities, summing to exactly n_patients. Without them
        # the greedy drifts: a patient carrying a singleton level always looks
        # best in the LARGEST group (its level's target is biggest there), so
        # unequal shares systematically under-fill the small group — and swaps
        # can never repair it, since a swap preserves group sizes by
        # construction. Capacities pin the sizes and leave the objective to do
        # what it is good at, which is balancing the marginals within them.
        self.capacities = self._largest_remainder(self.n_patients, shares)

    @staticmethod
    def _largest_remainder(total: int, shares: np.ndarray) -> np.ndarray:
        """Integer group sizes summing to ``total``, closest to ``shares``."""
        exact = total * shares
        base = np.floor(exact).astype(int)
        remaining = total - int(base.sum())
        if remaining:
            order = np.lexsort((np.arange(len(shares)), -(exact - base)))
            base[order[:remaining]] += 1
        return base

    def rarity_order(self, table: pd.DataFrame) -> np.ndarray:
        """Rarest-first: a patient with a scarce level gets placed while every
        group is still open. Ties resolve by table order, so it is stable."""
        rarity = np.zeros(self.n_patients, dtype=np.float64)
        for column in self.columns:
            counts = table[column].map(table[column].value_counts()).to_numpy(dtype=np.float64)
            rarity += 1.0 / counts
        return np.lexsort((np.arange(self.n_patients), -rarity))

    def delta(self, patient: int, group: int, direction: int) -> float:
        rows = self.rows[patient]
        residual = self.counts[rows, group] - self.targets[rows, group]
        marginal = float((direction * 2.0 * residual + 1.0).sum())
        size_residual = self.sizes[group] - self.target_sizes[group]
        return marginal + SIZE_WEIGHT * (direction * 2.0 * size_residual + 1.0)

    def apply(self, patient: int, group: int, direction: int) -> None:
        self.counts[self.rows[patient], group] += direction
        self.sizes[group] += direction

    def objective(self) -> float:
        marginal = float(((self.counts - self.targets) ** 2).sum())
        size = float(((self.sizes - self.target_sizes) ** 2).sum())
        return marginal + SIZE_WEIGHT * size

    def greedy(self, table: pd.DataFrame) -> np.ndarray:
        assignment = np.full(self.n_patients, -1, dtype=int)
        for patient in self.rarity_order(table):
            deltas = np.array(
                [self.delta(int(patient), group, +1) for group in range(self.n_groups)]
            )
            # A group at capacity is out of the running entirely.
            deltas = np.where(self.sizes < self.capacities, deltas, np.inf)
            if not np.isfinite(deltas).any():
                raise RuntimeError("every group is at capacity before all patients are placed")
            # Round before comparing so float noise cannot make the choice
            # machine-dependent; ties go to the emptiest, then lowest, group.
            best = int(np.lexsort((np.arange(self.n_groups), self.sizes, deltas.round(9)))[0])
            assignment[patient] = best
            self.apply(int(patient), best, +1)
        return assignment

    def swap_refine(self, assignment: np.ndarray, max_passes: int = SWAP_PASSES) -> int:
        """Accept only strictly improving swaps, so passes cannot oscillate."""
        signature = np.array(
            [hash(tuple(row)) for row in self.rows], dtype=np.int64
        )  # identical patients are interchangeable; skip those pairs
        passes = 0
        improved = True
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for left in range(self.n_patients):
                group_left = int(assignment[left])
                for right in range(left + 1, self.n_patients):
                    group_right = int(assignment[right])
                    if group_left == group_right or signature[left] == signature[right]:
                        continue
                    delta = self.delta(left, group_left, -1)
                    self.apply(left, group_left, -1)
                    delta += self.delta(right, group_right, -1)
                    self.apply(right, group_right, -1)
                    delta += self.delta(left, group_right, +1)
                    self.apply(left, group_right, +1)
                    delta += self.delta(right, group_left, +1)
                    self.apply(right, group_left, +1)
                    if delta < -1e-9:
                        assignment[left] = group_right
                        assignment[right] = group_left
                        group_left = group_right
                        improved = True
                    else:  # revert
                        self.apply(right, group_left, -1)
                        self.apply(left, group_right, -1)
                        self.apply(right, group_right, +1)
                        self.apply(left, group_left, +1)
        return passes


def assign_folds(
    rows: pd.DataFrame, columns: Sequence[str], n_folds: int
) -> tuple[pd.Series, dict[str, Any]]:
    """Assign every patient to one of ``n_folds`` folds, balancing marginals."""
    table = patient_table(rows, columns)
    balancer = _Balancer(table, columns, np.full(n_folds, 1.0 / n_folds))
    assignment = balancer.greedy(table)
    initial = balancer.objective()
    passes = balancer.swap_refine(assignment)

    folds = pd.Series(assignment, index=table.index, name="k_fold")
    report = balance_report(table, folds, columns, group_column="k_fold")
    report["objective"] = {
        "after_greedy": initial,
        "after_swaps": balancer.objective(),
        "swap_passes": passes,
    }
    return folds, report


def carve_out_validation(
    rows: pd.DataFrame,
    columns: Sequence[str],
    pool_patients: Sequence[str],
    ratio: float,
) -> list[str]:
    """Pick a balanced ``ratio`` share of ``pool_patients`` for early stopping.

    A two-group assignment with the same objective: group 0 trains, group 1
    validates, with targets in proportion ``(1 - ratio) : ratio``.
    """
    pool = rows[rows["patient_id"].isin(set(pool_patients))]
    table = patient_table(pool, columns)
    balancer = _Balancer(table, columns, np.array([1.0 - ratio, ratio], dtype=float))
    assignment = balancer.greedy(table)
    balancer.swap_refine(assignment)
    return sorted(table.index[assignment == 1].tolist())


def balance_report(
    table: pd.DataFrame,
    groups: pd.Series,
    columns: Sequence[str],
    group_column: str = "group",
) -> dict[str, Any]:
    """Per-group marginal distributions and the worst proportion deviation."""
    joined = table.join(groups.rename(group_column))
    report: dict[str, Any] = {
        "n_patients": int(len(table)),
        "group_sizes": {
            str(key): int(value)
            for key, value in joined[group_column].value_counts().sort_index().items()
        },
        "marginals": {},
    }
    worst = 0.0
    for column in columns:
        overall = joined[column].value_counts(normalize=True)
        proportions = pd.crosstab(joined[column], joined[group_column], normalize="columns")
        deviation = float(proportions.sub(overall, axis=0).abs().to_numpy().max())
        worst = max(worst, deviation)
        counts = pd.crosstab(joined[column], joined[group_column])
        report["marginals"][column] = {
            "counts": {
                str(key): {str(k): int(v) for k, v in value.items()}
                for key, value in counts.to_dict().items()
            },
            "max_abs_proportion_deviation": deviation,
        }
    report["worst_marginal_deviation"] = worst
    return report
