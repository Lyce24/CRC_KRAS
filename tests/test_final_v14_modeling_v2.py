"""Numerical regression for exact constant columns in pending v14 refits."""

from pathlib import Path

import numpy as np
import pytest

from tools import final_v14_modeling_v2 as modeling


@pytest.mark.parametrize("kind", ["logistic", "ridge"])
def test_decimal_constant_has_unit_scale_and_exact_zero_coefficient(kind):
    # Repeated 0.1 acquires a tiny floating standard deviation although every
    # observed value is exactly equal. Scale==0 alone misses this constant.
    count = 1239
    X = np.column_stack([np.full(count, 0.1), np.linspace(-1, 1, count)])
    assert X[:, 0].std() > 0
    assert np.ptp(X[:, 0]) == 0
    y = np.arange(count) % 2
    fitted = modeling._fit(X, y, 1.0, kind)
    assert fitted["status"] == "ESTIMABLE"
    assert fitted["scaler_scale"][0] == 1.0
    assert fitted["coef"][0] == 0.0
    assert fitted["zero_variance_coordinates"] == [0]
    np.testing.assert_allclose(fitted["predictions"],
                               ((X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1]))
                               * fitted["coef"][1] + fitted["intercept"], atol=1e-12)


def test_v2_changes_only_the_constant_detection_line():
    root = Path(__file__).resolve().parents[1]
    old = (root / "tools/final_v14_modeling.py").read_bytes()
    new = (root / "tools/final_v14_modeling_v2.py").read_bytes()
    assert new == old.replace(b"    constant = scale == 0",
                              b"    constant = (np.ptp(X, axis=0) == 0) | (scale == 0)")
