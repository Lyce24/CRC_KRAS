from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from tools import final_v14_module1_evaluate as m


def test_auc_ties_match_standard_estimator():
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([1, 2, 2, 3, 1, 1])
    assert m.auc(y, s) == roc_auc_score(y, s)


def test_fidelity_can_be_negative_and_ratio_not_clipped():
    y = np.array([0, 0, 1, 1])
    actual = np.array([0.0, 2.0, 1.0, 3.0])
    reconstructed = np.array([0.0, 1.0, 10.0, 11.0])
    d = m.fidelity(y, actual, reconstructed)
    assert d["r2_oof"] < 0
    assert d["retention_ratio"] == 2.0
    assert np.isnan(m.fidelity(y, -actual, reconstructed)["retention_ratio"])


def test_ratio_does_not_redraw_undefined_bootstraps():
    a = np.r_[np.ones(9499), np.full(501, np.nan)]
    r = m.interval(1, a, ratio=True)
    assert r["status"] == "RATIO_CI_NOT_ESTIMABLE"
    assert r["finite_draws"] == 9499 and r["undefined_draws"] == 501
    a[9499] = 1
    assert m.interval(1, a, ratio=True)["ci95"] == [1, 1]
