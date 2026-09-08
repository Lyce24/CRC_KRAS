from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from tools import final_v14_modeling as m


def test_heldout_values_cannot_change_selection_or_scaler():
    rng = np.random.default_rng(45)
    X = rng.normal(size=(60, 3))
    y = np.tile([0, 1], 30)
    splits = list(StratifiedKFold(4, shuffle=True, random_state=19).split(X, y))
    first = m.nested_logistic(X, y, X[:3], splits)
    second = m.nested_logistic(X, y, X[:3] + 1e6, splits)
    assert first["status"] == "ESTIMABLE"
    for key in ["selected_penalty", "scaler_mean", "scaler_scale", "coef", "intercept", "candidates"]:
        assert first[key] == second[key]
    np.testing.assert_allclose(first["scaler_mean"], X.mean(axis=0))


def test_constant_features_zero_coef_and_strongest_tie():
    X = np.ones((40, 2))
    y = np.tile([0, 1], 20)
    splits = list(StratifiedKFold(4).split(X, y))
    logistic = m.nested_logistic(X, y, X[:2], splits)
    ridge = m.nested_ridge(X, y, X[:2], splits)
    assert logistic["selected_penalty"] == min(m.PENALTIES)
    assert ridge["selected_penalty"] == max(m.PENALTIES)
    assert logistic["coef"] == [0, 0]
    assert logistic["scaler_scale"] == [1, 1]


def test_nonconvergence_is_not_an_eligible_candidate(monkeypatch):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    def fail(*args, **kwargs):
        warnings.warn("did not converge", ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(m.LogisticRegression, "fit", fail)
    X = np.arange(40).reshape(20, 2)
    y = np.tile([0, 1], 10)
    result = m.nested_logistic(X, y, X[:2], list(StratifiedKFold(2).split(X, y)))
    assert result["status"] == "NOT_ESTIMABLE"
    assert all(not c["eligible"] for c in result["candidates"])


def test_inner_leakage_is_rejected():
    with pytest.raises(ValueError, match="partition"):
        m.nested_logistic(np.ones((4, 1)), [0, 1, 0, 1], [[1]],
                          [([0, 1, 2], [2, 3]), ([2, 3], [0, 1])])
