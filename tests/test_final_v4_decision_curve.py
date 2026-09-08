from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v4_decision_curve as dca  # noqa: E402


def test_perfect_predictor_net_benefit_equals_prevalence() -> None:
    y = np.array([1.0] * 40 + [0.0] * 60)
    p = y.copy()  # perfect probabilities: 1 for positives, 0 for negatives
    rows = dca.net_benefit_curves(y, p, n_draws=50, seed=0, thresholds=(0.1, 0.4, 0.6))
    for row in rows:
        assert row["net_benefit"] == pytest.approx(0.40)
        assert row["flagged_fraction"] == pytest.approx(0.40)


def test_prioritize_all_formula_and_pairing() -> None:
    y = np.array([1.0] * 30 + [0.0] * 70)
    p = np.full(100, 1.0)  # model that flags everyone == prioritize-all
    rows = dca.net_benefit_curves(y, p, n_draws=200, seed=1, thresholds=(0.2, 0.5))
    for row in rows:
        odds = row["threshold"] / (1 - row["threshold"])
        assert row["net_benefit_all"] == pytest.approx(0.30 - 0.70 * odds)
        assert row["net_benefit"] == pytest.approx(row["net_benefit_all"])
        # flag-everyone can never beat the best default: delta <= 0, and its
        # paired bootstrap interval must contain no strictly positive mass
        # beyond numerical noise at thresholds where treat-all is the best.
        assert row["delta_vs_best_default"] <= 1e-12
        assert row["delta_vs_best_default_ci"][1] <= 1e-12


def test_summarize_range_reports_point_and_ci_thresholds() -> None:
    rows = [
        {"threshold": 0.2, "delta_vs_best_default": -0.01, "delta_vs_best_default_ci": [-0.02, 0.0]},
        {"threshold": 0.3, "delta_vs_best_default": 0.02, "delta_vs_best_default_ci": [-0.01, 0.05]},
        {"threshold": 0.4, "delta_vs_best_default": 0.05, "delta_vs_best_default_ci": [0.02, 0.08]},
    ]
    summary = dca.summarize_range(rows)
    assert summary["thresholds_model_above_best_default_point"] == [0.3, 0.4]
    assert summary["thresholds_model_above_best_default_ci_lower"] == [0.4]


def test_median_of_seed_curves_is_elementwise_median() -> None:
    def curve(shift: float) -> list[dict]:
        return [
            {
                "threshold": 0.4,
                "net_benefit": 0.10 + shift,
                "net_benefit_ci": [0.05 + shift, 0.15 + shift],
                "net_benefit_all": 0.01,
                "net_benefit_all_ci": [0.0, 0.02],
                "net_benefit_none": 0.0,
                "delta_vs_best_default": 0.09 + shift,
                "delta_vs_best_default_ci": [0.04 + shift, 0.14 + shift],
                "flagged_fraction": 0.3,
            }
        ]

    merged = dca.median_of_seed_curves([curve(0.0), curve(0.01), curve(0.05)])
    assert merged[0]["net_benefit"] == pytest.approx(0.11)
    assert merged[0]["net_benefit_ci"] == [pytest.approx(0.06), pytest.approx(0.16)]
    assert merged[0]["delta_vs_best_default"] == pytest.approx(0.10)
