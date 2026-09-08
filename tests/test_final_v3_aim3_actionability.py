from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "final_v3_aim3_actionability.py"
SPEC = importlib.util.spec_from_file_location("final_v3_aim3_actionability", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_authoritative_input_hashes_match() -> None:
    inputs = MODULE.validate_inputs(MODULE.DEFAULT_FIXED, MODULE.DEFAULT_REPEATED)
    assert inputs["fixed"]["sha256"] == MODULE.EXPECTED_INPUT_SHA256["fixed"]
    assert inputs["repeated"]["sha256"] == MODULE.EXPECTED_INPUT_SHA256["repeated"]


def test_actionability_rows_preserve_aim3_verdicts() -> None:
    fixed = MODULE.read_json(MODULE.DEFAULT_FIXED)
    repeated = MODULE.read_json(MODULE.DEFAULT_REPEATED)
    rows = MODULE.build_rows(fixed, repeated)
    by_level = {row["molecular_level"]: row for row in rows}

    assert len(rows) == 8
    assert by_level["Codon 12 versus other KRAS"]["repeated_control_consensus"] == (
        "CONSENSUS_CEILING"
    )
    assert by_level["G12D versus all other KRAS"]["repeated_control_consensus"] == (
        "CONSENSUS_CEILING"
    )
    assert by_level["G12D versus other G12"]["repeated_control_consensus"] == ("CONSENSUS_CEILING")
    assert by_level["G12V versus other G12"]["repeated_control_consensus"] == (
        "NO_CEILING_CONSENSUS"
    )
    assert by_level["G12C versus other G12"]["repeated_control_consensus"] == (
        "NO_CEILING_CONSENSUS"
    )
    assert {row["direct_clinical_decision_supported"] for row in rows} == {"No"}


def test_run_is_write_once_and_receipted(tmp_path: Path) -> None:
    output = tmp_path / "actionability"
    result = MODULE.run(MODULE.DEFAULT_FIXED, MODULE.DEFAULT_REPEATED, output)
    assert result["status"] == "PASS"
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert set(receipt["outputs"]) == {
        "actionability_ladder.csv",
        "actionability_ladder.json",
        "design.md",
    }
    with pytest.raises(FileExistsError):
        MODULE.run(MODULE.DEFAULT_FIXED, MODULE.DEFAULT_REPEATED, output)
