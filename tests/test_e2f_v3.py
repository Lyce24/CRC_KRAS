from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_v3_fulllabel_residual_adaptation  # noqa: E402


def test_mean_loss_objective_gradient_and_duplicate_invariance() -> None:
    features = np.array(
        [[0.2, -0.4], [1.1, 0.3], [-0.7, 0.5], [0.4, 0.9]], dtype=float
    )
    offset = np.array([-0.3, 0.2, 0.6, -0.1], dtype=float)
    labels = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)
    theta = np.array([0.15, -0.25, 0.05], dtype=float)
    lam = 0.3
    value, gradient = aim2_v3_fulllabel_residual_adaptation.residual_objective_and_gradient(
        theta, features, offset, labels, lam
    )

    epsilon = 1e-6
    finite_difference = np.empty_like(theta)
    for index in range(len(theta)):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_value, _ = aim2_v3_fulllabel_residual_adaptation.residual_objective_and_gradient(
            plus, features, offset, labels, lam
        )
        minus_value, _ = aim2_v3_fulllabel_residual_adaptation.residual_objective_and_gradient(
            minus, features, offset, labels, lam
        )
        finite_difference[index] = (plus_value - minus_value) / (2 * epsilon)
    assert np.isfinite(value)
    assert np.allclose(gradient, finite_difference, atol=1e-9, rtol=1e-7)

    duplicated_value, duplicated_gradient = aim2_v3_fulllabel_residual_adaptation.residual_objective_and_gradient(
        theta,
        np.vstack([features, features]),
        np.concatenate([offset, offset]),
        np.concatenate([labels, labels]),
        lam,
    )
    assert duplicated_value == pytest.approx(value, abs=1e-15)
    assert np.allclose(duplicated_gradient, gradient, atol=1e-15, rtol=0)


def test_exact_support_contract_and_infeasible_failure() -> None:
    labels = np.array([0] * 6 + [1] * 6, dtype=int)
    train = np.arange(len(labels))
    support = aim2_v3_fulllabel_residual_adaptation.draw_exact_support(labels, train, 8, seed=123)
    assert len(support) == len(np.unique(support)) == 8
    assert int((labels[support] == 0).sum()) == 4
    assert int((labels[support] == 1).sum()) == 4

    insufficient_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(RuntimeError, match="infeasible"):
        aim2_v3_fulllabel_residual_adaptation.draw_exact_support(labels, insufficient_train, 8, seed=123)


def _toy_seed_frame(model_seed: int) -> pd.DataFrame:
    labels = np.array([0, 1] * 5, dtype=int)
    patient_ids = [f"P{index:02d}" for index in range(len(labels))]
    native = np.linspace(-1.0, 1.0, len(labels)) + (model_seed - 43) * 0.01
    payload: dict[str, np.ndarray] = {
        "label": labels,
        "eta_native": native,
    }
    for dimension in range(512):
        payload[f"e{dimension}"] = np.linspace(-0.2, 0.2, len(labels)) + dimension * 1e-5
    return pd.DataFrame(payload, index=patient_ids)


def test_full_layout_has_no_fold_leakage_and_complete_oof_census(monkeypatch) -> None:
    monkeypatch.setattr(aim2_v3_fulllabel_residual_adaptation, "LAMBDA_GRID", (np.inf,))
    data = {
        "Toy": {
            model_seed: _toy_seed_frame(model_seed) for model_seed in aim2_v3_fulllabel_residual_adaptation.SEEDS
        }
    }
    ledger = aim2_v3_fulllabel_residual_adaptation.SolverLedger()
    blocks, oof_rows, fits = aim2_v3_fulllabel_residual_adaptation.run_full_layout(
        data, outer_seed=aim2_v3_fulllabel_residual_adaptation.PRIMARY_OUTER_SEED, ledger=ledger
    )

    assert len(blocks["Toy"]) == 10
    assert len(oof_rows) == 10
    assert len({row["patient_id"] for row in oof_rows}) == 10
    assert {row["fold"] for row in oof_rows} == set(range(5))
    assert len(fits) == 20  # 3 seeds x 5 adapter fits + 5 Platt fits
    assert all(
        not (set(row["fit_patient_ids"]) & set(row["test_patient_ids"]))
        for row in fits
    )
    all_patients = {f"P{index:02d}" for index in range(10)}
    for model_seed in aim2_v3_fulllabel_residual_adaptation.SEEDS:
        seed_fits = [
            row
            for row in fits
            if row["model_kind"] == "residual_adapter"
            and row["model_seed"] == model_seed
        ]
        held_out = [patient for row in seed_fits for patient in row["test_patient_ids"]]
        assert len(held_out) == len(set(held_out)) == 10
        assert set(held_out) == all_patients


def test_full_label_lower_boundary_is_fail_closed() -> None:
    safe = {
        "phase": "full_label",
        "model_kind": "residual_adapter",
        "selected_lambda": "0.003",
    }
    support_boundary = {
        "phase": "support_curve",
        "model_kind": "residual_adapter",
        "selected_lambda": "0.001",
    }
    aim2_v3_fulllabel_residual_adaptation.assert_full_label_lambda_grid_closed([safe, support_boundary])

    unsafe = {
        "phase": "full_label",
        "model_kind": "residual_adapter",
        "selected_lambda": "0.001",
    }
    with pytest.raises(RuntimeError, match="lower boundary 0.001"):
        aim2_v3_fulllabel_residual_adaptation.assert_full_label_lambda_grid_closed([unsafe])


def test_solver_nonfinite_solution_is_rejected(monkeypatch) -> None:
    def fake_minimize(*_args, **_kwargs):
        return SimpleNamespace(
            x=np.array([np.nan, 0.0, 0.0]),
            success=True,
            status=0,
            message="fake",
            nit=1,
            nfev=1,
        )

    monkeypatch.setattr(aim2_v3_fulllabel_residual_adaptation, "minimize", fake_minimize)
    features = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    offset = np.zeros(4)
    labels = np.array([0, 1, 0, 1])
    with np.errstate(invalid="ignore"), pytest.raises(
        RuntimeError, match="non-finite solution"
    ):
        aim2_v3_fulllabel_residual_adaptation.fit_residual(
            features,
            offset,
            labels,
            1.0,
            ledger=aim2_v3_fulllabel_residual_adaptation.SolverLedger(),
            context={"test": True},
        )


def test_incremental_improvement_semantic_verdict_is_mechanical() -> None:
    established = aim2_v3_fulllabel_residual_adaptation.derive_incremental_improvement([0.01, 0.08], [0.02, 0.03])
    assert established == {
        "macro_auroc_delta_ci_lower_above_zero": True,
        "both_cohort_auroc_delta_points_above_zero": True,
        "pass": True,
    }
    assert not aim2_v3_fulllabel_residual_adaptation.derive_incremental_improvement(
        [-0.01, 0.08], [0.02, 0.03]
    )["pass"]
    assert not aim2_v3_fulllabel_residual_adaptation.derive_incremental_improvement(
        [0.01, 0.08], [-0.02, 0.03]
    )["pass"]
