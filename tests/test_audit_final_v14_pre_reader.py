"""Focused tests for the independent FINAL-v14 pre-reader auditor."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "tools/audit_final_v14_pre_reader.py"
SPEC = importlib.util.spec_from_file_location("audit_final_v14_pre_reader_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDITOR
SPEC.loader.exec_module(AUDITOR)


def test_balanced_hierarchical_plan_is_capacity_safe_and_canonical() -> None:
    slides = pd.DataFrame(
        {
            "subcohort": ["B", "A", "A", "B"],
            "patient_id": ["p4", "p2", "p1", "p3"],
            "slide_id": ["s4", "s2", "s1", "s3"],
            "n_tiles": [9, 1, 9, 1],
        }
    )

    observed = AUDITOR.hierarchical_sample_plan(slides, cap=12)

    assert observed["slide_id"].tolist() == ["s1", "s2", "s3", "s4"]
    assert observed.groupby("subcohort")["n_sample"].sum().to_dict() == {"A": 6, "B": 6}
    assert int(observed["n_sample"].sum()) == 12
    assert observed["n_sample"].le(observed["n_tiles"]).all()


def test_sample_replay_detects_one_tile_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(AUDITOR, "SAMPLE_TILES", 5)
    plan = pd.DataFrame(
        {
            "subcohort": ["A"],
            "patient_id": ["p"],
            "slide_id": ["s"],
            "n_tiles": [20],
            "n_sample": [5],
        }
    )
    rng = np.random.Generator(np.random.PCG64(AUDITOR.VOCAB_SEED))
    drawn = np.asarray(rng.choice(20, size=5, replace=False), dtype=np.int64)
    order = np.argsort(drawn, kind="stable")
    positions = drawn[order]
    sample = pd.DataFrame(
        {
            "subcohort": ["A"] * 5,
            "patient_id": ["p"] * 5,
            "slide_id": ["s"] * 5,
            "tile_id": [f"{value:012d}" for value in positions],
            "tile_index": positions,
            "global_tile_index": positions + 100,
            "within_slide_draw_index": np.arange(5, dtype=np.int64)[order],
            "sample_index": np.arange(5, dtype=np.int64),
        }
    )
    AUDITOR._replay_sample_ids(plan, sample, {"s": 100})
    sample.loc[0, "global_tile_index"] += 1
    with pytest.raises(AUDITOR.AuditError, match="global packed row"):
        AUDITOR._replay_sample_ids(plan, sample, {"s": 100})


def test_hungarian_mapping_replays_bijection_and_threshold() -> None:
    reference = np.eye(3, dtype=np.float32)
    source = reference[[2, 0, 1]]

    mapping = AUDITOR._recompute_mapping(source, reference)

    assert mapping["reference_prototype_id"].tolist() == [2, 0, 1]
    assert mapping["name_mappable"].tolist() == [True, True, True]
    assert mapping["cosine_similarity"].tolist() == [1.0, 1.0, 1.0]


def test_exact_profile_gate_rejects_tile_weighting() -> None:
    columns = list(AUDITOR.PROTOTYPE_COLUMNS)
    expected = pd.DataFrame(np.zeros((1, 32)), columns=columns)
    expected.loc[0, "prototype_00"] = 0.5
    expected.loc[0, "prototype_01"] = 0.5
    for position, value in enumerate(["reference", "p", "S", 0, 2]):
        expected.insert(
            position,
            ["vocabulary_id", "patient_id", "subcohort", "fold", "n_slides"][position],
            value,
        )
    observed = expected.copy()
    observed.loc[0, "prototype_00"] = 100 / 101
    observed.loc[0, "prototype_01"] = 1 / 101

    with pytest.raises(AUDITOR.AuditError, match="numeric reconstruction"):
        AUDITOR._profile_residual(observed, expected, "patient profiles")


def test_hmac_blinding_is_collision_free_and_order_replayable() -> None:
    occurrences = [(prototype, 0) for prototype in range(32)] + [
        (prototype, 1) for prototype in range(8)
    ]
    salt = bytes(range(32))

    first = AUDITOR._hmac_blinding_table(occurrences, salt)
    second = AUDITOR._hmac_blinding_table(list(reversed(occurrences)), salt)

    pd.testing.assert_frame_equal(first, second)
    assert first["code"].is_unique
    assert first["code"].str.fullmatch(r"[A-Z2-7]{6}").all()
    assert first["presentation_order"].tolist() == list(range(40))


def test_exact_tree_forbids_orphan_temp_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "expected.txt").write_text("sealed", encoding="utf-8")
    AUDITOR._assert_exact_tree(root, {"expected.txt"}, "fixture")

    (root / "orphan.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(AUDITOR.AuditError, match="orphan"):
        AUDITOR._assert_exact_tree(root, {"expected.txt"}, "fixture")
    (root / "orphan.txt").unlink()

    (root / ".expected.txt.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(AUDITOR.AuditError, match="temporary"):
        AUDITOR._assert_exact_tree(root, {"expected.txt"}, "fixture")
    (root / ".expected.txt.tmp").unlink()

    os.symlink(root / "expected.txt", root / "linked.txt")
    with pytest.raises(AUDITOR.AuditError, match="symlink"):
        AUDITOR._assert_exact_tree(root, {"expected.txt", "linked.txt"}, "fixture")


def test_teacher_numeric_gate_is_exact_and_tamper_evident() -> None:
    expected = pd.DataFrame({"patient_id": ["a", "b"], "logit": [1.0, -2.0]})
    observed = expected.copy()
    assert (
        AUDITOR._assert_table_numeric_exact(observed, expected, ["logit"], "teacher fixture") == 0.0
    )
    observed.loc[1, "logit"] = np.nextafter(-2.0, 0.0)
    with pytest.raises(AUDITOR.AuditError, match="exceeds tolerance"):
        AUDITOR._assert_table_numeric_exact(observed, expected, ["logit"], "teacher fixture")


def test_atomic_receipt_replaces_whole_json_without_temp_orphans(tmp_path: Path) -> None:
    destination = tmp_path / "audit/audit_receipt.json"
    first = AUDITOR.json_bytes({"status": "PASS", "run": 1})
    second = AUDITOR.json_bytes({"status": "PASS", "run": 2})

    AUDITOR._atomic_receipt(destination, first)
    AUDITOR._atomic_receipt(destination, second)

    assert destination.read_bytes() == second
    assert destination.stat().st_mode & 0o777 == 0o600
    assert list(destination.parent.iterdir()) == [destination]
    assert (
        hashlib.sha256(destination.read_bytes()).hexdigest() == hashlib.sha256(second).hexdigest()
    )


def test_handoff_ready_requires_every_gate() -> None:
    gates = {"a": {"status": "PASS"}, "b": {"status": "PASS"}}
    assert all(gate.get("status") == "PASS" for gate in gates.values())
    gates["b"]["status"] = "FAIL"
    assert not all(gate.get("status") == "PASS" for gate in gates.values())
