from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyze_reviews_v5 import (  # noqa: E402
    restricted_pair_tau_b,
    stratified_kendall_tau_b,
)


def _frame(blocks: dict[str, tuple[list[float], list[float]]]) -> pd.DataFrame:
    rows = []
    for block, (exposure, outcome) in blocks.items():
        for x_value, y_value in zip(exposure, outcome, strict=True):
            rows.append(
                {
                    "analysis_block": block,
                    "exposure": x_value,
                    "outcome": y_value,
                    "complete": True,
                }
            )
    return pd.DataFrame(rows)


def test_constant_outcome_block_is_penalized_not_dropped() -> None:
    frame = _frame(
        {
            "A": ([1, 2, 3], [1, 2, 3]),
            "B": ([1, 2, 3], [0, 0, 0]),
        }
    )
    statistic, rows, components = stratified_kendall_tau_b(
        frame,
        "exposure",
        "outcome",
        "complete",
        {"A": 3, "B": 3},
    )
    assert math.isclose(statistic, 3 / math.sqrt(6 * 3))
    assert statistic < 1.0
    assert components["non_tied_exposure_pairs"] == 6
    assert components["non_tied_outcome_pairs"] == 3
    assert rows[1]["informative"] is False
    assert rows[1]["non_tied_exposure_pairs"] == 3


def test_between_cohort_location_shifts_do_not_change_restricted_pair_tau() -> None:
    base = _frame(
        {
            "cohort1|mutant": ([1, 2, 3], [1, 2, 3]),
            "cohort2|mutant": ([1, 2, 3], [3, 2, 1]),
        }
    )
    shifted = base.copy()
    in_second = shifted["analysis_block"] == "cohort2|mutant"
    shifted.loc[in_second, "exposure"] += 10_000
    shifted.loc[in_second, "outcome"] += 1_000
    design = {"cohort1|mutant": 3, "cohort2|mutant": 3}
    first, _, _ = stratified_kendall_tau_b(base, "exposure", "outcome", "complete", design)
    second, _, _ = stratified_kendall_tau_b(shifted, "exposure", "outcome", "complete", design)
    assert first == second


def test_identity_within_block_permutation_preserves_exact_statistic() -> None:
    blocks = [
        (np.array([0.0, 0.2, 0.8, 1.1]), np.array([0.0, 1.0, 1.0, 3.0])),
        (np.array([0.1, 0.1, 0.7]), np.array([2.0, 0.0, 1.0])),
    ]
    observed, observed_components = restricted_pair_tau_b(blocks)
    identity_permutation = np.arange(4), np.arange(3)
    replay, replay_components = restricted_pair_tau_b(
        (x_values, y_values[identity])
        for (x_values, y_values), identity in zip(blocks, identity_permutation, strict=True)
    )
    assert replay == observed
    assert replay_components == observed_components
