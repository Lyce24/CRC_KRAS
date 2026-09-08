"""Contract tests for E3a's conclusion rule and its control-minus-fine contrast.

The verdict table is the whole experiment: it decides whether a fine rung at
chance is an empirical resolution ceiling, an underpowered non-result, or weak
but real signal. These tests pin the rule so a later edit cannot quietly turn a
null into a claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim3_resolution_ladder as e3a  # noqa: E402


def _stats(auroc: float, low: float, high: float) -> dict:
    return {"auroc": auroc, "ci": [low, high]}


def _delta(low: float, high: float) -> dict:
    return {"delta": (low + high) / 2, "ci": [low, high]}


# ── the verdict table ────────────────────────────────────────────────────────
def test_unlearnable_control_dominates_every_other_condition():
    # Even a fine rung that looks bounded cannot be a ceiling if the matched
    # easy task is itself at chance: the null carries no information.
    code, _ = e3a.verdict_for(
        _stats(0.51, 0.46, 0.55), _stats(0.53, 0.47, 0.59), _delta(-0.02, 0.06)
    )
    assert code == "UNDERPOWERED"


def test_bounded_fine_learnable_control_and_positive_delta_is_a_ceiling():
    code, _ = e3a.verdict_for(
        _stats(0.527, 0.475, 0.578), _stats(0.658, 0.612, 0.704), _delta(0.061, 0.199)
    )
    assert code == "CEILING"


def test_a_ceiling_is_refused_when_the_delta_ci_touches_zero():
    # Third condition of the revised rule: two overlapping CIs are not evidence
    # that the control is better, so this is inconclusive rather than a ceiling.
    code, _ = e3a.verdict_for(
        _stats(0.54, 0.48, 0.59), _stats(0.60, 0.52, 0.67), _delta(-0.01, 0.14)
    )
    assert code == "INCONCLUSIVE"


def test_a_bounded_but_above_chance_fine_rung_is_reported_as_its_own_outcome():
    # Conditions 2 and 3 both hold. Calling this a plain ceiling would hide a
    # fine rung that is demonstrably above chance.
    code, _ = e3a.verdict_for(
        _stats(0.56, 0.52, 0.59), _stats(0.66, 0.61, 0.70), _delta(0.03, 0.17)
    )
    assert code == "CEILING_WITH_RESIDUAL_SIGNAL"


def test_a_fine_rung_above_chance_but_unbounded_is_positive_evidence():
    code, prose = e3a.verdict_for(
        _stats(0.64, 0.58, 0.70), _stats(0.66, 0.61, 0.71), _delta(-0.04, 0.08)
    )
    assert code == "FINE_RESOLUTION_EVIDENCE"
    assert "practically meaningful" in prose


def test_everything_else_is_inconclusive():
    code, _ = e3a.verdict_for(
        _stats(0.55, 0.49, 0.62), _stats(0.66, 0.61, 0.71), _delta(0.02, 0.20)
    )
    assert code == "INCONCLUSIVE"


# ── the control-minus-fine contrast ──────────────────────────────────────────
def _arm(n: int, separation: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = np.repeat([0, 1], n // 2)
    score = rng.normal(0.0, 1.0, n) + separation * label
    return pd.DataFrame(
        {
            "patient_id": [f"p{seed}_{i}" for i in range(n)],
            "cohort": np.resize(["A", "B"], n),
            "label": label,
            "mean_logit": score,
        }
    )


def _partially_paired_arms(
    n_positive: int,
    n_fine_negative: int,
    n_control_negative: int,
    fine_separation: float,
    control_separation: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    positive_ids = [f"shared_{i}" for i in range(n_positive)]
    positive_cohort = np.resize(["A", "B"], n_positive)
    latent = rng.normal(size=n_positive)

    def build(kind: str, n_negative: int, separation: float) -> pd.DataFrame:
        positive = pd.DataFrame(
            {
                "patient_id": positive_ids,
                "cohort": positive_cohort,
                "label": 1,
                "mean_logit": latent + separation,
            }
        )
        negative = pd.DataFrame(
            {
                "patient_id": [f"{kind}_negative_{i}" for i in range(n_negative)],
                "cohort": np.resize(["A", "B"], n_negative),
                "label": 0,
                "mean_logit": rng.normal(size=n_negative),
            }
        )
        return pd.concat([positive, negative], ignore_index=True)

    return (
        build("fine", n_fine_negative, fine_separation),
        build("control", n_control_negative, control_separation),
    )


def test_delta_bootstrap_recovers_a_real_control_advantage():
    fine, control = _partially_paired_arms(200, 200, 200, 0.0, 1.0, seed=1)
    out = e3a.delta_bootstrap(fine, control, n_bootstrap=400, seed=11)
    assert out["delta"] > 0.15
    assert out["ci"][0] > 0.0
    assert out["bootstrap_fraction_delta_gt_0"] > 0.99
    assert out["shared_positive_n"] == 200


def test_delta_bootstrap_ci_covers_zero_when_the_arms_match():
    fine, control = _partially_paired_arms(200, 200, 200, 0.8, 0.8, seed=3)
    out = e3a.delta_bootstrap(fine, control, n_bootstrap=400, seed=12)
    assert out["ci"][0] < 0.0 < out["ci"][1]


def test_delta_bootstrap_pairs_positives_but_accepts_different_negative_sizes():
    fine, control = _partially_paired_arms(100, 100, 300, 0.0, 1.0, seed=5)
    out = e3a.delta_bootstrap(fine, control, n_bootstrap=200, seed=13)
    assert np.isfinite(out["delta"])
    assert out["n_bootstrap"] > 0


def test_delta_bootstrap_refuses_nonidentical_positive_patients():
    fine, control = _partially_paired_arms(100, 100, 100, 0.0, 1.0, seed=6)
    control.loc[control["label"].eq(1).idxmax(), "patient_id"] = "not_shared"
    with pytest.raises(ValueError, match="positive patient sets are not identical"):
        e3a.delta_bootstrap(fine, control, n_bootstrap=10, seed=14)


def test_delta_bootstrap_uses_the_same_positive_draw_for_both_models():
    fine, control = _partially_paired_arms(100, 100, 100, 0.0, 0.0, seed=7)
    # Identical paired positive scores plus constant, identical negative scores
    # must give exactly zero in every replicate. Independent positive draws
    # would create artificial delta variance and fail this assertion.
    control.loc[control["label"].eq(1), "mean_logit"] = fine.loc[
        fine["label"].eq(1), "mean_logit"
    ].to_numpy()
    fine.loc[fine["label"].eq(0), "mean_logit"] = 0.0
    control.loc[control["label"].eq(0), "mean_logit"] = 0.0
    out = e3a.delta_bootstrap(fine, control, n_bootstrap=100, seed=15)
    assert out["delta"] == 0.0
    assert out["ci"] == [0.0, 0.0]


def test_legacy_in_place_report_is_disabled():
    with pytest.raises(SystemExit, match="Legacy in-place E3 reporting is disabled"):
        e3a.cmd_report(argparse.Namespace(output_root=None, n_bootstrap=10_000))


def test_corrected_report_refuses_too_few_bootstrap_draws(tmp_path: Path):
    with pytest.raises(SystemExit, match="at least 10,000"):
        e3a.cmd_report(
            argparse.Namespace(output_root=str(tmp_path / "new"), n_bootstrap=9_999)
        )


# ── the ladder definition ────────────────────────────────────────────────────
def test_every_fine_rung_has_exactly_one_matched_control():
    fine = {name for name, (_, kind) in e3a.TASKS.items() if kind == "fine"}
    control = {name for name, (_, kind) in e3a.TASKS.items() if kind == "control"}
    assert {f for f, _ in e3a.PAIRS} == fine
    assert {c for _, c in e3a.PAIRS} == control
    assert len(e3a.PAIRS) == len(fine) == len(control)


def test_every_control_draws_its_wild_types_with_its_own_seed():
    controls = [name for name, (_, kind) in e3a.TASKS.items() if kind == "control"]
    seeds = [e3a.CONTROL_WT_SEED[name] for name in controls]
    assert set(controls) == set(e3a.CONTROL_WT_SEED)
    assert len(set(seeds)) == len(seeds)


def test_allele_membership_keeps_multi_substitution_calls():
    # "G12V;G12C" is both a G12V and a G12C patient, and is codon-12 either way.
    assert e3a.is_g12("G12V;G12C")
    assert e3a.has_allele("G12V;G12C", "G12C")
    assert e3a.is_g12d("G12D;G13D")
    assert not e3a.is_g12d("G13D")
    assert not e3a.is_g12("A146T")
