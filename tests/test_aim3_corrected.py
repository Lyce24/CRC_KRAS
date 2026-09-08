from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim3_fixed_control_analysis as a3  # noqa: E402


def _frame(
    positive: list[tuple[str, str, str, float]],
    negative: list[tuple[str, str, str, float]],
) -> pd.DataFrame:
    rows = [(*item, 1) for item in positive] + [(*item, 0) for item in negative]
    return pd.DataFrame(rows, columns=["patient_id", "cohort", "subcohort", "mean_logit", "label"])


def test_auc_handles_ties_exactly():
    assert a3._auc(np.array([0, 0, 1, 1]), np.array([0.0, 1.0, 1.0, 2.0])) == pytest.approx(0.875)


def test_task_bootstrap_is_reproducible_and_complete():
    frame = _frame(
        [("p1", "A", "A1", 2.0), ("p2", "B", "B1", 3.0)],
        [("n1", "A", "A1", -2.0), ("n2", "B", "B1", -3.0)],
    )
    first = a3.task_bootstrap(frame, n_bootstrap=20, seed=7)
    second = a3.task_bootstrap(frame, n_bootstrap=20, seed=7)
    assert np.array_equal(first, second)
    assert len(first) == 20
    assert np.all(first == 1.0)


def test_partially_paired_bootstrap_requires_shared_positive_patients():
    fine = _frame([("p1", "A", "A1", 1.0)], [("n1", "A", "A1", 0.0)])
    control = _frame([("different", "A", "A1", 1.0)], [("c1", "A", "A1", 0.0)])
    with pytest.raises(ValueError, match="identical"):
        a3.partially_paired_bootstrap(fine, control, n_bootstrap=2, seed=1)


def test_partially_paired_bootstrap_accepts_different_negative_patients():
    fine = _frame(
        [("p1", "A", "A1", 2.0), ("p2", "B", "B1", 2.0)],
        [("n1", "A", "A1", 1.0), ("n2", "B", "B1", 1.0)],
    )
    control = _frame(
        [("p1", "A", "A1", 3.0), ("p2", "B", "B1", 3.0)],
        [("c1", "A", "A1", 0.0), ("c2", "B", "B1", 0.0)],
    )
    out = a3.partially_paired_bootstrap(fine, control, n_bootstrap=10, seed=3)
    assert set(out) == {"fine_auc", "control_auc", "delta"}
    assert np.all(out["fine_auc"] == 1.0)
    assert np.all(out["control_auc"] == 1.0)
    assert np.all(out["delta"] == 0.0)


def test_subcohort_standardization_target_must_have_support():
    fine = _frame([("p1", "A", "A1", 2.0)], [("n1", "A", "A1", 1.0)])
    control = _frame([("p1", "A", "A1", 2.0)], [("c1", "A", "A1", 0.0)])
    with pytest.raises(ValueError, match="invalid or unsupported"):
        a3.partially_paired_bootstrap(
            fine,
            control,
            n_bootstrap=2,
            seed=1,
            stratum="subcohort",
            target_control_negative_counts={"missing": 1},
        )


def test_verdict_underpowered_control_dominates():
    fine = {"ci95": [0.40, 0.55], "auroc": 0.48}
    control = {"ci95": [0.49, 0.70], "auroc": 0.60}
    delta = {"ci95": [0.01, 0.20], "estimate": 0.12}
    assert a3.verdict_for(fine, control, delta) == "UNDERPOWERED"


def test_verdict_requires_positive_delta_for_ceiling():
    fine = {"ci95": [0.45, 0.58], "auroc": 0.52}
    control = {"ci95": [0.55, 0.70], "auroc": 0.62}
    delta = {"ci95": [-0.01, 0.21], "estimate": 0.10}
    assert a3.verdict_for(fine, control, delta) == "INCONCLUSIVE"


def test_atomic_writer_refuses_overwrite(tmp_path: Path):
    path = tmp_path / "receipt.json"
    a3._write_json_once_atomic(path, {"status": "PASS"})
    with pytest.raises(FileExistsError):
        a3._write_json_once_atomic(path, {"status": "changed"})
    assert a3._read_json(path) == {"status": "PASS"}


def test_stable_seed_changes_by_endpoint():
    assert a3._stable_seed(a3.BOOTSTRAP_SEED, "a") == a3._stable_seed(a3.BOOTSTRAP_SEED, "a")
    assert a3._stable_seed(a3.BOOTSTRAP_SEED, "a") != a3._stable_seed(a3.BOOTSTRAP_SEED, "b")
