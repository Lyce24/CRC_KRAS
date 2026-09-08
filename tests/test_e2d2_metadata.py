from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_peritoneal_stability  # noqa: E402


def test_patient_metadata_uses_explicit_multislide_aggregation() -> None:
    source = pd.DataFrame(
        {
            "patient_uid": ["p1", "p1", "p1", "p2"],
            "specimen_role": ["metastatic", "metastatic", "primary", "metastatic"],
            "braf": ["wild_type", "wild_type", "wild_type", "mutant"],
            "mpp": [0.25, 0.50, 9.0, 0.30],
            "slide_size_bytes": [100, 250, 999, 400],
            "image_format": ["svs", "tiff", "svs", "svs"],
        }
    )

    observed = aim2_peritoneal_stability._patient_metadata(source, {"p1", "p2"})

    assert observed.loc["p1", "mpp"] == pytest.approx(0.375)
    assert observed.loc["p1", "slide_size_bytes"] == 350
    assert observed.loc["p1", "image_format"] == "svs+tiff"
    assert observed.loc["p1", "metadata_slide_count"] == 2


def test_patient_metadata_fails_on_clinical_conflict() -> None:
    source = pd.DataFrame(
        {
            "patient_uid": ["p1", "p1"],
            "specimen_role": ["metastatic", "metastatic"],
            "braf": ["mutant", "wild_type"],
        }
    )

    with pytest.raises(RuntimeError, match="not patient-constant"):
        aim2_peritoneal_stability._patient_metadata(source, {"p1"})
