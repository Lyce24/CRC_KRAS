from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v5_aim4_context_weld as weld  # noqa: E402
from tools import final_v5_aim4_human_read as human  # noqa: E402


def test_bh_adjust_matches_hand_computation() -> None:
    contrasts = [
        {"p_two_sided": 0.01},
        {"p_two_sided": 0.04},
        {"p_two_sided": 0.03},
        {"p_two_sided": 0.005},
    ]
    weld.bh_adjust(contrasts)
    # sorted p: .005, .01, .03, .04 -> raw q: .02, .02, .04, .04 (after monotone step-up)
    assert contrasts[3]["q_bh"] == pytest.approx(0.02)
    assert contrasts[0]["q_bh"] == pytest.approx(0.02)
    assert contrasts[2]["q_bh"] == pytest.approx(0.04)
    assert contrasts[1]["q_bh"] == pytest.approx(0.04)


def test_weld_cell_block_reports_quiet_contrasts() -> None:
    frame = pd.DataFrame(
        {
            "cell": ["kras_mutant"] * 20 + ["nras_mutant"] * 20
            + ["braf_mutant_nras_wt"] * 20 + ["pathway_quiet"] * 40,
            "abundance_p17": [0.3] * 20 + [0.1] * 20 + [0.4] * 20 + [0.1] * 40,
        }
    )
    cells, contrasts = weld.cell_block(frame, "abundance_p17", rng_seed=0)
    assert set(cells) == set(weld.CELLS)
    assert len(contrasts) == 3  # every non-quiet cell versus quiet
    braf = next(c for c in contrasts if c["cell"] == "braf_mutant_nras_wt")
    assert braf["mean_difference_vs_quiet"] == pytest.approx(0.3)
    assert braf["auroc_cell_vs_quiet"] == pytest.approx(1.0)
    nras = next(c for c in contrasts if c["cell"] == "nras_mutant")
    assert nras["mean_difference_vs_quiet"] == pytest.approx(0.0)


def test_human_concordance_covers_full_packet_prototype_set() -> None:
    assert set(human.CONCORDANCE) == {20, 6, 26, 17, 9, 1, 28, 12, 8, 15, 5}
    assert all(v[0] in {"agree", "partial", "differ"} for v in human.CONCORDANCE.values())


def test_human_montage_map_rejects_ambiguity(tmp_path: Path) -> None:
    key = pd.DataFrame({"montage_id": ["M01", "M01"], "prototype": [17, 28]})
    path = tmp_path / "key.csv"
    key.to_csv(path, index=False)
    with pytest.raises(AssertionError, match="more than one prototype"):
        human.montage_map(path)


def test_human_montage_map_reads_assignment(tmp_path: Path) -> None:
    key = pd.DataFrame({"montage_id": ["M04"] * 3 + ["M07"] * 3, "prototype": [17] * 3 + [28] * 3})
    path = tmp_path / "key.csv"
    key.to_csv(path, index=False)
    assert human.montage_map(path) == {"M04": 17, "M07": 28}
