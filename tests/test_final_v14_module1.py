from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from tools import final_v14_module1 as m


def test_joint_heldout_clinical_values_do_not_change_fit():
    rng = np.random.default_rng(18)
    frame = pd.DataFrame({"age_at_diagnosis": rng.normal(60, 10, 48),
                          "sex": ["male", "female"] * 24,
                          "site_class": ["Colon"] * 48,
                          "stage_class": ["I-II", "III-IV"] * 24,
                          "label": [0, 1] * 24})
    X = rng.normal(size=(48, 3))
    splits = list(StratifiedKFold(4, shuffle=True, random_state=19).split(X, frame.label))
    test = frame.iloc[:4].copy()
    first = m.joint_fit(X, frame.label.to_numpy(), X[:4], frame, test, splits)
    test["age_at_diagnosis"] = 2000
    test["site_class"] = "TEST_ONLY"
    second = m.joint_fit(X, frame.label.to_numpy(), X[:4], frame, test, splits)
    assert first["status"] == "ESTIMABLE"
    for field in ["coef", "intercept", "selected_penalty", "clinical_encoder", "candidates"]:
        assert first[field] == second[field]
    assert "TEST_ONLY" not in str(second["clinical_encoder"])


def test_named_set_cannot_be_unsealed(tmp_path):
    import pytest

    path = tmp_path / "names.json"
    m.seal_json(path, {"status": "NAME_GATE_FAIL"})
    assert m.read_sealed(path)["status"] == "NAME_GATE_FAIL"
    path.chmod(0o600)
    path.write_text('{"status":"NAME_GATE_PASS"}')
    with pytest.raises(ValueError, match="seal mismatch"):
        m.read_sealed(path)
