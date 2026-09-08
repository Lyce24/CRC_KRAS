"""Contracts for the deployment-equivalent E2c procedural estimator."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_head_adaptation_base  # noqa: E402


def test_vectorized_bootstrap_auroc_matches_sklearn_with_ties():
    y = np.array([0, 0, 0, 1, 1, 1])
    draws = np.array(
        [
            [0, 0, 1, 1, 2, 2],
            [3, 2, 1, 1, 2, 3],
        ],
        dtype=float,
    )
    expected = np.array([roc_auc_score(y, row) for row in draws])
    assert aim2_head_adaptation_base._auroc_rows(y, draws) == pytest.approx(expected)


def test_headline_is_mean_metric_not_metric_of_averaged_predictions():
    y = np.array([0, 0, 0, 1, 1, 1])
    draws = np.array(
        [
            [0, 1, 3, 2, 4, 5],
            [4, 2, 0, 5, 1, 3],
        ],
        dtype=float,
    )

    out = aim2_head_adaptation_base.procedure_summary(y, draws, n_boot=40, random_seed=17)
    expected = np.mean([roc_auc_score(y, row) for row in draws])
    invalid_ensemble = roc_auc_score(y, draws.mean(axis=0))

    assert out["metrics"]["auroc"] == pytest.approx(expected)
    assert out["metrics"]["auroc"] == pytest.approx(7 / 9)
    assert invalid_ensemble == 1.0
    assert out["metrics"]["auroc"] != invalid_ensemble
    assert len(out["per_draw_metrics"]) == 2


def test_two_way_bootstrap_is_reproducible_and_tracks_both_random_sources():
    rng = np.random.default_rng(3)
    y = np.repeat([0, 1], 20)
    draws = np.vstack([rng.normal(size=len(y)) + effect * y for effect in (0.0, 0.4, 0.8, 1.2)])

    first = aim2_head_adaptation_base.procedure_summary(y, draws, n_boot=100, random_seed=91)
    second = aim2_head_adaptation_base.procedure_summary(y, draws, n_boot=100, random_seed=91)

    assert first == second
    assert first["single_procedure_draw_auroc"]["observed_sd"] > 0
    lo, hi = first["expected_auroc_ci"]
    assert lo < first["metrics"]["auroc"] < hi
    pred_lo, pred_hi = first["single_procedure_draw_auroc"][
        "patient_and_support_variability_interval"
    ]
    assert pred_hi - pred_lo > 0


def test_contrast_pairs_patients_but_independently_resamples_support_draws():
    rng = np.random.default_rng(5)
    y = np.repeat([0, 1], 30)
    a = np.vstack([rng.normal(size=len(y)) + effect * y for effect in (0.6, 0.8, 1.0)])
    b = np.vstack([rng.normal(size=len(y)) + effect * y for effect in (0.0, 0.1, 0.2, 0.3)])

    out = aim2_head_adaptation_base.procedure_contrast(
        y, a, b, n_boot=100, random_seed=29, support_resampling="independent"
    )
    expected = np.mean([roc_auc_score(y, row) for row in a]) - np.mean(
        [roc_auc_score(y, row) for row in b]
    )

    assert out["delta"] == pytest.approx(expected)
    assert out["patient_resampling"] == "paired"
    assert out["support_resampling"] == "independent"
    assert len(out["ci"]) == 2


def test_paired_support_bootstrap_requires_naturally_aligned_draws():
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="equal lengths"):
        aim2_head_adaptation_base.procedure_contrast(
            y,
            np.ones((2, 4)),
            np.ones((3, 4)),
            n_boot=5,
            support_resampling="paired",
        )


def test_run_cohort_retains_exact_budget_and_patient_leakage_audit(monkeypatch):
    monkeypatch.setenv("OCEANPATH_AIM2_LINEAGE", "aim2_cap8192_v2_test")
    n = 20
    labels = np.tile([0, 1], n // 2)
    met_ids = [f"m{i}" for i in range(n)]
    pri_ids = ["m0", "m1"] + [f"p{i}" for i in range(n - 2)]
    met_pat = pd.DataFrame({"patient_id": met_ids, "label": labels})
    pri_pat = pd.DataFrame({"patient_id": pri_ids, "label": labels})
    met_h = np.column_stack([np.linspace(-2, 2, n), np.sin(np.arange(n))])
    pri_h = np.column_stack([np.linspace(2, -2, n), np.cos(np.arange(n))])

    def fake_patient_table(target, kind, cap, subcohort=None):
        del target, cap, subcohort
        if kind == "metastatic":
            return met_pat.copy(), {42: met_h.copy()}
        return pri_pat.copy(), {42: pri_h.copy()}

    monkeypatch.setattr(aim2_head_adaptation_base, "patient_table", fake_patient_table)
    monkeypatch.setattr(
        aim2_head_adaptation_base,
        "patient_native_logits",
        lambda target, kind, cap: {42: met_h @ np.array([1.0, -0.25])},
    )
    monkeypatch.setattr(aim2_head_adaptation_base.aim2_loco_transport, "SEEDS", (42,))
    monkeypatch.setattr(aim2_head_adaptation_base, "BUDGETS", (2,))
    monkeypatch.setattr(
        aim2_head_adaptation_base.np,
        "load",
        lambda _path: {"w": np.array([[1.0, -0.25]]), "b": np.array([0.0])},
    )
    monkeypatch.setattr(aim2_head_adaptation_base, "select_lambda", lambda *args: np.inf)
    monkeypatch.setattr(
        aim2_head_adaptation_base,
        "fit_l2sp",
        lambda H, y, w0, b0, lam: (w0.copy(), float(b0)),
    )

    out = aim2_head_adaptation_base.run_cohort("RIH", None, 8192, reps=3, n_bootstrap=10)

    assert np.asarray(out["prediction_draws"]["S1_k2"]).shape == (3, n)
    assert np.asarray(out["prediction_draws"]["S2_k2"]).shape == (3, n)
    assert "per_patient" not in out
    assert "mean_auroc" not in out["arms"]["S1_k2"]
    for arm in ("S1_k2", "S2_k2"):
        assert len(out["support_draws"][arm]) == 3
        for rep in out["support_draws"][arm]:
            for fold in rep["folds"]:
                support = fold["support_patient_ids"]
                assert len(support) == len(set(support)) == 4
                assert fold["support_labels"].count(0) == 2
                assert fold["support_labels"].count(1) == 2
                test_ids = {
                    out["patient_ids"][i]
                    for i, assigned in enumerate(out["fold_of_patient"])
                    if assigned == fold["fold"]
                }
                assert test_ids.isdisjoint(support)


def test_e2c_result_path_is_inside_required_immutable_lineage(monkeypatch):
    monkeypatch.setenv("OCEANPATH_AIM2_LINEAGE", "aim2_cap8192_v2_test")
    path = aim2_head_adaptation_base.result_path(8192)
    assert "/reruns/aim2_cap8192_v2_test/e2c/analysis/" in str(path)
    assert path.name == "e2c_true_kshot_cap8192.json"


def test_run_refuses_to_overwrite_an_existing_result(monkeypatch, tmp_path):
    existing = tmp_path / "e2c_true_kshot_cap8192.json"
    existing.write_text("frozen evidence")
    monkeypatch.setattr(aim2_head_adaptation_base, "result_path", lambda cap: existing)
    monkeypatch.setattr(
        aim2_head_adaptation_base,
        "run_cohort",
        lambda *args, **kwargs: pytest.fail("calculation started before overwrite guard"),
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        aim2_head_adaptation_base.cmd_run(SimpleNamespace(cap=8192, reps=100, n_bootstrap=2000))
    assert existing.read_text() == "frozen evidence"
