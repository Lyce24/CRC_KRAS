"""Numerical and provenance checks for governed v14 context associations."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import final_v14_context as context  # noqa: E402


def leave_one_out_hc3(X, y):
    """Independent HC3 identity using n actual leave-one-patient-out refits."""
    coefficient = np.linalg.solve(X.T @ X, X.T @ y)
    differences = []
    for omitted in range(len(X)):
        kept = np.arange(len(X)) != omitted
        beta = np.linalg.solve(X[kept].T @ X[kept], X[kept].T @ y[kept])
        differences.append(beta - coefficient)
    differences = np.asarray(differences)
    return coefficient, differences.T @ differences


def test_multioutput_hc3_matches_independent_leave_one_out_identity():
    rng = np.random.default_rng(74)
    X = np.column_stack([np.ones(120), rng.normal(size=(120, 3))])
    Y = X @ rng.normal(size=(4, 3)) + rng.normal(size=(120, 3)) * (0.3 + abs(X[:, 1, None]))
    actual = context.hc3_multioutput(X, Y, [3])
    for j in range(3):
        coefficient, covariance = leave_one_out_hc3(X, Y[:, j])
        np.testing.assert_allclose(actual["coefficient"][:, j], coefficient, atol=1e-12)
        np.testing.assert_allclose(actual["covariance"][j], covariance, atol=1e-12)
        p = 2 * norm.sf(abs(coefficient[3] / np.sqrt(covariance[3, 3])))
        np.testing.assert_allclose(actual["pvalue"][j], p, atol=1e-12)
        expected_effect = coefficient[3] / np.std(Y[:, j], ddof=1)
        np.testing.assert_allclose(actual["standardized_contrast"][0, j], expected_effect, atol=1e-12)


def test_joint_hc3_wald_and_partial_r2_match_reduced_model():
    rng = np.random.default_rng(41)
    X = np.column_stack([np.ones(100), rng.normal(size=(100, 3))])
    Y = rng.normal(size=(100, 2)) + X[:, 2, None]
    actual = context.hc3_multioutput(X, Y, [2, 3])
    for j in range(2):
        coefficient, covariance = leave_one_out_hc3(X, Y[:, j])
        R = np.eye(4)[[2, 3]]
        b, cov = R @ coefficient, R @ covariance @ R.T
        wald = b @ np.linalg.solve(cov, b)
        np.testing.assert_allclose(actual["pvalue"][j], chi2.sf(wald, 2), atol=1e-12)
        reduced_beta = np.linalg.solve(X[:, :2].T @ X[:, :2], X[:, :2].T @ Y[:, j])
        reduced_sse = np.sum((Y[:, j] - X[:, :2] @ reduced_beta) ** 2)
        full_sse = np.sum((Y[:, j] - X @ coefficient) ** 2)
        np.testing.assert_allclose(actual["partial_r2"][j], (reduced_sse - full_sse) / reduced_sse, atol=1e-12)


def test_bootstrap_refits_ols_on_one_shared_patient_draw():
    rng = np.random.default_rng(53)
    X = np.column_stack([np.ones(30), rng.normal(size=30)])
    Y = np.column_stack([X[:, 1] + rng.normal(size=30), 2 * X[:, 1] + rng.normal(size=30)])
    strata = [("a" if i < 15 else "b", i % 2) for i in range(30)]
    indices = context.stratified_bootstrap_indices(strata, 7)
    np.testing.assert_array_equal(indices, context.stratified_bootstrap_indices(strata, 7))
    result = context.bootstrap_ols(X, Y, [1], indices)
    for draw, sampled in enumerate(indices):
        beta = np.linalg.lstsq(X[sampled], Y[sampled], rcond=None)[0][1]
        np.testing.assert_allclose(result["raw_contrasts"][draw, 0], beta)
        np.testing.assert_allclose(result["standardized_contrasts"][draw, 0], beta / Y[sampled].std(0, ddof=1))
        for group in set(strata):
            assert sum(strata[i] == group for i in sampled) == strata.count(group)


def test_rank_deficient_bootstrap_is_not_redrawn_or_imputed():
    X = np.column_stack([np.ones(5), [0, 0, 0, 0, 1]])
    Y = np.arange(5, dtype=float).reshape(-1, 1)
    indices = np.array([[0, 1, 2, 3, 0], [0, 1, 2, 3, 4]])
    result = context.bootstrap_ols(X, Y, [1], indices)
    assert np.isnan(result["raw_contrasts"][0]).all()
    assert np.isfinite(result["raw_contrasts"][1]).all()
    assert context.finite_interval(result["raw_contrasts"][:, 0, 0])["undefined_draws"] == 1


def test_bh_fixed_family_matches_known_adjustment():
    p = np.array([0.0004, 0.7, 1.0, 0.022, 0.05, 0.13])
    np.testing.assert_allclose(context.bh_adjust(p), [0.0024, 0.84, 1.0, 0.066, 0.1, 0.195])


def test_normalization_preserves_unknown_missingness_and_frozen_references():
    frame = pd.DataFrame({"subcohort": ["a", "a", "b", "b", "b", "b"],
                          "msi_dmmr": ["MSS/pMMR", "MSI/dMMR", "MSI-H/dMMR", "unknown", None, " MSS/pMMR "]})
    dictionary = {"variables": {"source_subcohort": {"levels": ["a", "b"]},
                                "msi_dmmr": {"type": "binary", "levels": ["MSS/pMMR", "MSI-H/dMMR"]}}}
    design = context.association_design(frame, "msi_dmmr", dictionary)
    np.testing.assert_array_equal(design["kept"], [0, 1, 2, 5])
    np.testing.assert_array_equal(design["X"][:, -1], [0, 1, 1, 0])
    assert design["reference_level"] == "MSS/pMMR"
    assert design["contrast_names"] == ["MSI-H/dMMR minus MSS/pMMR"]


def test_fold_mapping_uses_heldout_local_coordinates_only():
    frame = pd.DataFrame({"patient_id": [f"p{f}" for f in range(5)], "k_fold": range(5)})
    local = frame.rename(columns={"k_fold": "fold"}).copy()
    local["prototype_00"] = np.arange(1, 6) / 10
    local["prototype_01"] = 0.8
    mapping = pd.DataFrame([{"outer_fold": f, "reference_prototype_id": 10,
                             "source_prototype_id": f % 2, "cosine_similarity": 0.9} for f in range(5)])
    result = context.aligned_context(frame, mapping, local, [10])
    expected = np.array([0.1, 0.8, 0.3, 0.8, 0.5])
    np.testing.assert_allclose(result[:, 0], np.arcsin(np.sqrt(expected)))
    mapping.loc[0, "cosine_similarity"] = 0.79
    with pytest.raises(ValueError, match="qualifying"):
        context.aligned_context(frame, mapping, local, [10])


def test_attention_boundary_roundoff_clipped_only_with_explicit_tolerance():
    frame = pd.DataFrame({"patient_id": [f"p{f}" for f in range(5)], "k_fold": range(5)})
    local = frame.rename(columns={"k_fold": "fold"}).copy()
    local["prototype_00"] = 1.000000006
    mapping = pd.DataFrame([{"outer_fold": f, "reference_prototype_id": 0,
                             "source_prototype_id": 0, "cosine_similarity": 0.9} for f in range(5)])
    with pytest.raises(ValueError, match="proportions invalid"):
        context.aligned_context(frame, mapping, local, [0])
    result = context.aligned_context(frame, mapping, local, [0], boundary_tolerance=2e-6)
    np.testing.assert_allclose(result, np.pi / 2)
    assert local["prototype_00"].gt(1).all()
