from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v4_aim4_pathology_ingest as ingest  # noqa: E402


def _complete_row(montage_id: str) -> dict:
    return {
        "montage_id": montage_id,
        "n_tiles": "12",
        "review_status": "complete",
        "interpretable": "yes",
        "dominant_pattern": "glandular tumour",
        "heterogeneity": "homogeneous",
        "architecture": "glandular",
        "mucin": "absent",
        "dirty_necrosis": "absent",
        "desmoplasia_stroma": "present",
        "budding_invasion": "absent",
        "immune_infiltration": "absent",
        "normal_organ_tissue": "absent",
        "artifact": "absent",
        "differentiation": "moderate",
        "confidence": "high",
        "reviewer_id": "R1",
        "review_date": "2026-08-20",
        "blinding_attestation": "confirmed_no_key_access",
        "free_text": "none",
    }


def _form(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=ingest.FORM_COLUMNS)


def _blank(montages: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"montage_id": montages})


def test_valid_form_passes() -> None:
    form = _form([_complete_row("M01"), _complete_row("M02")])
    assert ingest.validate_form(form, _blank(["M01", "M02"]), "base") == []


def test_vocabulary_violation_is_reported() -> None:
    row = _complete_row("M01")
    row["architecture"] = "tubular"  # not in the controlled vocabulary
    problems = ingest.validate_form(_form([row]), _blank(["M01"]), "base")
    assert any("architecture" in p for p in problems)


def test_interpretable_no_cascade_is_enforced() -> None:
    row = _complete_row("M01")
    row["interpretable"] = "no"
    row["dominant_pattern"] = "not_assessable"
    # every cascade field must be not_assessable; leave architecture glandular
    problems = ingest.validate_form(_form([row]), _blank(["M01"]), "base")
    assert any("interpretable=no requires" in p for p in problems)


def test_montage_set_mismatch_is_reported() -> None:
    form = _form([_complete_row("M01")])
    problems = ingest.validate_form(form, _blank(["M01", "M02"]), "base")
    assert any("montage ids" in p for p in problems)


def test_pending_field_is_reported() -> None:
    row = _complete_row("M01")
    row["free_text"] = "pending"
    problems = ingest.validate_form(_form([row]), _blank(["M01"]), "base")
    assert any("free_text incomplete" in p for p in problems)


def test_montage_prototype_map_requires_unique_prototype(tmp_path: Path) -> None:
    key = pd.DataFrame(
        {
            "montage_id": ["M01", "M01"],
            "prototype": [17, 28],
            "selection_stratum": ["candidate", "candidate"],
        }
    )
    path = tmp_path / "key.csv"
    key.to_csv(path, index=False)
    with pytest.raises(AssertionError, match="more than one prototype"):
        ingest.montage_prototype_map(path)


def test_montage_prototype_map_joins_stratum(tmp_path: Path) -> None:
    key = pd.DataFrame(
        {
            "montage_id": ["M01"] * 3,
            "prototype": [17] * 3,
            "selection_stratum": ["candidate"] * 3,
        }
    )
    path = tmp_path / "key.csv"
    key.to_csv(path, index=False)
    out = ingest.montage_prototype_map(path)
    assert out.iloc[0]["prototype"] == 17
    assert out.iloc[0]["n_key_tiles"] == 3
    assert out.iloc[0]["selection_stratum"] == "candidate"
