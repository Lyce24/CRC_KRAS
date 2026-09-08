from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v4_aim1_ras_composite as composite  # noqa: E402


def _frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "patient_id": [f"P{i}" for i in range(len(rows))],
        "label": [1.0 if r["kras"] == "mutant" else 0.0 for r in rows],
    }
    frame = pd.DataFrame(base)
    for column in ("kras", "nras", "braf"):
        frame[column] = [r[column] for r in rows]
    frame["ras"] = [
        "mutant"
        if r["kras"] == "mutant" or r["nras"] == "mutant"
        else ("wild_type" if r["kras"] == "wild_type" and r["nras"] == "wild_type" else "unknown")
        for r in rows
    ]
    return frame


def test_derive_endpoints_composites_and_context() -> None:
    rows = [
        {"kras": "mutant", "nras": "wild_type", "braf": "wild_type"},
        {"kras": "wild_type", "nras": "mutant", "braf": "wild_type"},
        {"kras": "wild_type", "nras": "wild_type", "braf": "mutant"},
        {"kras": "wild_type", "nras": "wild_type", "braf": "wild_type"},
        {"kras": "wild_type", "nras": "unknown", "braf": "wild_type"},
    ]
    out = composite.derive_endpoints(_frame(rows))
    assert out["endpoint_ras"].tolist()[:4] == [1.0, 1.0, 0.0, 0.0]
    assert np.isnan(out["endpoint_ras"].iloc[4])
    assert out["endpoint_mapk"].tolist()[:4] == [1.0, 1.0, 1.0, 0.0]
    assert np.isnan(out["endpoint_mapk"].iloc[4])
    assert out["wt_context"].tolist() == [
        "kras_mutant",
        "nras_mutant",
        "braf_mutant_nras_wt",
        "pathway_quiet",
        "incomplete_labels",
    ]


def test_derive_endpoints_fails_closed_on_manifest_disagreement() -> None:
    frame = _frame(
        [
            {"kras": "mutant", "nras": "wild_type", "braf": "wild_type"},
            {"kras": "wild_type", "nras": "wild_type", "braf": "wild_type"},
        ]
    )
    frame.loc[1, "ras"] = "mutant"  # contradicts kras|nras derivation
    with pytest.raises(AssertionError, match="disagrees"):
        composite.derive_endpoints(frame)


def test_endpoint_auroc_block_delta_is_relabeling_only() -> None:
    rng = np.random.default_rng(0)
    n = 400
    score = rng.normal(size=n)
    kras = (score + rng.normal(scale=1.0, size=n) > 0).astype(float)
    rows = [
        {
            "kras": "mutant" if k else "wild_type",
            "nras": "wild_type",
            "braf": "wild_type",
        }
        for k in kras
    ]
    frame = composite.derive_endpoints(_frame(rows))
    frame["s"] = score
    block = composite.endpoint_auroc_block(
        frame, "ras", {"only": "s"}, n_draws=200, seed=1
    )
    entry = block["scores"]["only"]
    # with nras all wild-type, ras == kras, so the relabeling delta is exactly 0
    assert entry["delta_vs_kras_on_shared_population"] == pytest.approx(0.0, abs=1e-12)
    assert entry["delta_vs_kras_ci"][0] == pytest.approx(0.0, abs=1e-12)
    assert entry["delta_vs_kras_ci"][1] == pytest.approx(0.0, abs=1e-12)


def test_median_of_endpoints_uses_seedwise_medians() -> None:
    per_seed = [
        {"auroc": 0.60, "auroc_ci": [0.55, 0.65]},
        {"auroc": 0.62, "auroc_ci": [0.57, 0.67]},
        {"auroc": 0.70, "auroc_ci": [0.66, 0.74]},
    ]
    out = composite.median_of_endpoints(per_seed)
    assert out["auroc"] == pytest.approx(0.62)
    assert out["auroc_ci"] == [pytest.approx(0.57), pytest.approx(0.67)]
