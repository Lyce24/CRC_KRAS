"""Contracts for the append-only native-logit E2c recovery component."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_residual_adaptation  # noqa: E402


def _objective(
    theta: np.ndarray,
    H: np.ndarray,
    y: np.ndarray,
    native: np.ndarray,
    lam: float,
) -> float:
    w, b = theta[:-1], theta[-1]
    z = native + H @ w + b
    return float(
        np.mean(np.logaddexp(0.0, z) - y * z)
        + lam / 2 * (np.dot(w, w) + b**2)
    )


def test_native_offset_solver_matches_independent_full_space_optimizer():
    rng = np.random.default_rng(19)
    H = rng.normal(size=(8, 4))
    y = np.tile([0.0, 1.0], 4)
    native = rng.normal(scale=0.7, size=len(y))
    lam = 3.0

    delta_w, delta_b = aim2_residual_adaptation.fit_native_offset(H, y, native, lam)
    reference = minimize(
        _objective,
        np.zeros(H.shape[1] + 1),
        args=(H, y, native, lam),
        method="BFGS",
        options={"gtol": 1e-11, "maxiter": 1000},
    )
    fitted = np.r_[delta_w, delta_b]
    assert _objective(fitted, H, y, native, lam) == pytest.approx(
        reference.fun, abs=1e-10
    )
    z = native + H @ delta_w + delta_b
    p = 1 / (1 + np.exp(-z))
    gradient = np.r_[H.T @ (p - y) / len(y) + lam * delta_w, np.mean(p - y) + lam * delta_b]
    assert np.max(np.abs(gradient)) < 1e-8


def test_infinite_lambda_is_exact_zero_delta_and_preserves_native_bf16_fixture():
    H = np.array([[1.234567, -0.333333], [-0.777777, 2.345678]])
    y = np.array([0.0, 1.0])
    source_w = np.array([0.912345, -1.223456])
    reconstructed = H @ source_w + 0.123456
    native = np.round(reconstructed * 128) / 128  # representative BF16-like rounding
    assert not np.array_equal(native, reconstructed)

    delta_w, delta_b = aim2_residual_adaptation.fit_native_offset(H, y, native, np.inf)
    prediction = native + H @ delta_w + delta_b

    assert np.array_equal(delta_w, np.zeros(H.shape[1]))
    assert delta_b == 0.0
    assert np.array_equal(prediction, native)
    diagnostic = aim2_residual_adaptation.native_offset_fit_diagnostics(
        H, y, native, delta_w, delta_b, np.inf
    )
    assert diagnostic == {
        "status": "native_exact",
        "delta_weight_l2": 0.0,
        "delta_bias": 0.0,
        "prediction_rule": "eta_native (no residual arithmetic)",
    }


def test_large_lambda_continuously_approaches_native_predictor():
    rng = np.random.default_rng(21)
    H = rng.normal(size=(10, 5))
    y = np.tile([0.0, 1.0], 5)
    native = rng.normal(size=len(y))

    delta_w, delta_b = aim2_residual_adaptation.fit_native_offset(H, y, native, 1e12)

    assert np.linalg.norm(delta_w) < 1e-10
    assert abs(delta_b) < 1e-10
    assert np.max(np.abs(H @ delta_w + delta_b)) < 1e-9


def test_loo_infinite_lambda_scores_held_out_native_offset_exactly():
    H = np.arange(16, dtype=float).reshape(4, 4) / 10
    y = np.array([0.0, 0.0, 1.0, 1.0])
    native = np.array([-2.0, -1.0, 0.5, 1.5])

    observed = aim2_residual_adaptation.loo_loss_native_offset(H, y, native, np.inf)
    p = np.clip(1 / (1 + np.exp(-native)), aim2_residual_adaptation.EPS, 1 - aim2_residual_adaptation.EPS)
    expected = np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

    assert observed == pytest.approx(expected)


def test_lambda_ties_prefer_infinite_shrinkage(monkeypatch):
    monkeypatch.setattr(aim2_residual_adaptation, "loo_loss_native_offset", lambda *args: 0.5)
    selected = aim2_residual_adaptation.select_lambda_native_offset(
        np.ones((4, 2)), np.array([0, 0, 1, 1]), np.zeros(4)
    )
    assert np.isinf(selected)


def test_all_infinite_procedure_predictions_are_bit_exact_s0(monkeypatch):
    n = 20
    labels = np.tile([0, 1], n // 2)
    met_pat = pd.DataFrame(
        {"patient_id": [f"m{i}" for i in range(n)], "label": labels}
    )
    pri_pat = pd.DataFrame(
        {
            "patient_id": ["m0", "m1", *[f"p{i}" for i in range(n - 2)]],
            "label": labels,
        }
    )
    rng = np.random.default_rng(4)
    met_h = {seed: rng.normal(size=(n, 3)) for seed in aim2_residual_adaptation.SEEDS}
    pri_h = {seed: rng.normal(size=(n, 3)) for seed in aim2_residual_adaptation.SEEDS}
    met_native = {
        seed: rng.normal(size=n) + (seed - aim2_residual_adaptation.SEEDS[0]) / 10
        for seed in aim2_residual_adaptation.SEEDS
    }
    pri_native = {seed: rng.normal(size=n) for seed in aim2_residual_adaptation.SEEDS}

    def fake_patient_data(bundle, target, kind, *, subcohort=None):
        del bundle, target, subcohort
        if kind == "metastatic":
            return met_pat.copy(), met_h, met_native
        return pri_pat.copy(), pri_h, pri_native

    monkeypatch.setattr(aim2_residual_adaptation, "_patient_data", fake_patient_data)
    monkeypatch.setattr(aim2_residual_adaptation, "BUDGETS", (2,))
    monkeypatch.setattr(aim2_residual_adaptation, "select_lambda_native_offset", lambda *args: np.inf)

    out = aim2_residual_adaptation.run_cohort(
        SimpleNamespace(), "RIH", None, reps=2, n_bootstrap=5
    )
    expected = np.mean(np.vstack([met_native[seed] for seed in aim2_residual_adaptation.SEEDS]), axis=0)

    assert np.array_equal(np.asarray(out["prediction_draws"]["S0"])[0], expected)
    assert np.array_equal(np.asarray(out["prediction_draws"]["S1_k2"]), np.vstack([expected] * 2))
    assert np.array_equal(np.asarray(out["prediction_draws"]["S2_k2"]), np.vstack([expected] * 2))
    assert out["arms"]["S1_k2"]["lambda_frac_declined"] == 1.0
    assert out["arms"]["S2_k2"]["lambda_frac_declined"] == 1.0
    for arm in ("S1_k2", "S2_k2"):
        solver = out["arms"][arm]["solver_audit"]
        assert solver["finite_fit_count"] == 0
        assert solver["declined_exact_native_count"] == 2 * 5 * 3
        assert solver["max_declined_delta_weight_l2"] == 0.0
        assert solver["max_declined_abs_delta_bias"] == 0.0


def test_identity_validation_fails_after_content_tampering(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    recorded = aim2_residual_adaptation._artifact_identity(artifact)
    artifact.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        aim2_residual_adaptation._validate_file_identity(recorded, label="fixture")


def test_atomic_writer_refuses_to_overwrite(tmp_path):
    destination = tmp_path / "evidence.json"
    destination.write_text("frozen")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        aim2_residual_adaptation._write_json_once_atomic(destination, {"new": "data"})
    assert destination.read_text() == "frozen"


def test_run_rejects_existing_output_before_preflight(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    sentinel = output_root / "evidence"
    sentinel.write_text("preserve")
    monkeypatch.setattr(
        aim2_residual_adaptation,
        "preflight_inputs",
        lambda *args: pytest.fail("preflight ran before overwrite guard"),
    )
    args = argparse.Namespace(
        input_lineage_root=str(input_root.resolve()),
        output_root=str(output_root.resolve()),
        cap=8192,
        reps=2,
        n_bootstrap=5,
    )

    with pytest.raises(FileExistsError, match="existing output root"):
        aim2_residual_adaptation.cmd_run(args)
    assert sentinel.read_text() == "preserve"


@pytest.mark.parametrize(
    ("reps", "n_bootstrap", "message"),
    [(1, 5, "reps must be at least 2"), (2, 0, "n-bootstrap must be positive")],
)
def test_invalid_run_arguments_do_not_reserve_output_root(
    monkeypatch, tmp_path, reps, n_bootstrap, message
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "new-output"
    input_root.mkdir()
    monkeypatch.setattr(
        aim2_residual_adaptation,
        "preflight_inputs",
        lambda *args: pytest.fail("preflight ran before argument validation"),
    )
    args = argparse.Namespace(
        input_lineage_root=str(input_root.resolve()),
        output_root=str(output_root.resolve()),
        cap=8192,
        reps=reps,
        n_bootstrap=n_bootstrap,
    )

    with pytest.raises(ValueError, match=message):
        aim2_residual_adaptation.cmd_run(args)
    assert not output_root.exists()


def test_source_byte_recheck_is_fail_closed():
    source = aim2_residual_adaptation._current_source_payloads()
    changed = dict(source)
    changed["aim2_residual_adaptation.py"] += b"concurrent edit"

    with pytest.raises(RuntimeError, match="source bytes changed"):
        aim2_residual_adaptation._assert_source_bytes_unchanged(changed)


def test_explicit_roots_must_be_absolute(tmp_path):
    with pytest.raises(ValueError, match="explicit absolute"):
        aim2_residual_adaptation._require_absolute(Path("relative"), label="fixture", must_exist=False)
