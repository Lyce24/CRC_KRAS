from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v6_encoder_and_control as v6  # noqa: E402


def _arm(etas: list[float], labels: list[int], ids: list[str] | None = None) -> pd.DataFrame:
    n = len(etas)
    return pd.DataFrame(
        {
            "patient_id": ids or [f"P{i:03d}" for i in range(n)],
            "label": labels,
            "msi_dmmr": ["MSS/pMMR"] * n,
            "braf": ["wild_type"] * n,
            "eta": etas,
        }
    )


def test_paired_contrast_rejects_mismatched_patients() -> None:
    left = _arm([0.1, 0.2], [0, 1])
    right = _arm([0.1, 0.2], [0, 1], ids=["X", "Y"])
    with pytest.raises(AssertionError, match="identical patients"):
        v6.paired_contrast(left, right)


def test_paired_contrast_rejects_label_disagreement() -> None:
    left = _arm([0.1, 0.2], [0, 1])
    right = _arm([0.1, 0.2], [1, 0])
    with pytest.raises(AssertionError, match="disagree on labels"):
        v6.paired_contrast(left, right)


def test_paired_contrast_zero_when_arms_identical() -> None:
    rng = np.random.default_rng(0)
    n = 200
    labels = (rng.random(n) > 0.5).astype(int).tolist()
    etas = rng.normal(size=n).tolist()
    frame = _arm(etas, labels)
    out = v6.paired_contrast(frame, frame.copy())
    assert out["A"]["delta_virchow2_minus_univ1"] == pytest.approx(0.0, abs=1e-12)
    assert out["A"]["delta_ci"] == [pytest.approx(0.0, abs=1e-12)] * 2
    assert out["A"]["excludes_zero"] is False
    assert out["spearman_between_ensembles"] == pytest.approx(1.0)


def _e3v(fine: float, control: float, gate: bool) -> dict:
    return {
        "univ1_anchor_points": {t: v["fine"] for t, v in v6.AIM3_SEALED_UNIV1.items()},
        "rungs": {
            task: {
                "n": 604,
                "n_positive": 420,
                "fine": {"auroc": fine, "ci95": [fine - 0.05, fine + 0.05]},
                "control": {"auroc": control, "ci95": [control - 0.05, control + 0.05]},
                "delta_control_minus_fine": {
                    "auroc": control - fine,
                    "ci95": [control - fine - 0.06, control - fine + 0.06],
                },
                "fixed_gate_one_sided_99": {
                    "fine_upper99_below_0p60": gate,
                    "control_lower99_above_0p50": gate,
                    "delta_lower99_above_zero": gate,
                    "ceiling": gate,
                },
            }
            for task in v6.AIM3_SEALED_UNIV1
        },
    }


def test_e3v_block_flags_anchor_drift() -> None:
    payload = _e3v(0.52, 0.65, True)
    payload["univ1_anchor_points"]["codon"] = 0.60  # drifted
    with pytest.raises(AssertionError, match="anchor drift"):
        v6.e3v_block(payload)


def test_e3v_block_reports_full_replication() -> None:
    out = v6.e3v_block(_e3v(0.52, 0.65, True))
    assert out["n_replicated"] == 3
    assert out["verdict"] == "ENCODER_ROBUST_CEILINGS"
    assert all(r["replication"] == "REPLICATED" for r in out["rungs"].values())


def test_e3v_block_reports_opened_rung_without_defending_it() -> None:
    out = v6.e3v_block(_e3v(0.62, 0.65, False))
    assert out["n_replicated"] == 0
    assert "revision" in out["verdict"]
    assert all(r["replication"] == "NOT_REPLICATED" for r in out["rungs"].values())


def test_rung_labels_cover_the_three_consensus_ceilings() -> None:
    assert set(v6.RUNG_LABEL) == set(v6.AIM3_SEALED_UNIV1) == {"codon", "g12d_broad", "allele1"}
