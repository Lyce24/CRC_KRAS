from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import analyze_reviews_v5 as analyzer  # noqa: E402
from tools import build_reviews_v5 as builder  # noqa: E402


def _assert_partition(rectangles: list[dict[str, int]], width: int, height: int) -> None:
    frame = pd.DataFrame(rectangles)
    assert builder.rectangles_are_partition(frame, width, height)
    assert sorted(frame["panel_number"].to_list()) == [1, 2, 3, 4, 5, 6]


def test_landscape_and_portrait_are_exact_six_panel_partitions() -> None:
    landscape = builder.panel_rectangles(1001, 701)
    portrait = builder.panel_rectangles(701, 1001)
    _assert_partition(landscape, 1001, 701)
    _assert_partition(portrait, 701, 1001)
    assert {(row["grid_rows"], row["grid_columns"]) for row in landscape} == {(2, 3)}
    assert {(row["grid_rows"], row["grid_columns"]) for row in portrait} == {(3, 2)}


def test_irregular_pyramid_uses_coarsest_level_not_coarser_than_target() -> None:
    level, downsample_x, downsample_y = builder.select_source_level(
        (1000, 800),
        ((1000, 800), (400, 400), (260, 210), (100, 80)),
        mpp_x=0.5,
        mpp_y=0.5,
        target_mpp=2.0,
    )
    assert level == 2
    assert downsample_x * 0.5 <= 2.0 + builder.SOURCE_MPP_TOLERANCE
    assert downsample_y * 0.5 <= 2.0 + builder.SOURCE_MPP_TOLERANCE
    assert (1000 / 100) * 0.5 > 2.0


def test_staged_identity_declares_intended_absolute_final_path(tmp_path: Path) -> None:
    staging = tmp_path / ".building"
    final = tmp_path / "v5"
    staged_file = staging / "FOR_PATHOLOGIST" / "payload.txt"
    staged_file.parent.mkdir(parents=True)
    staged_file.write_text("sealed\n")
    record = builder.staged_file_record(staged_file, staging, final)
    assert Path(record["path"]).is_absolute()
    assert Path(record["path"]) == final.resolve() / "FOR_PATHOLOGIST" / "payload.txt"
    assert record["sha256"] == builder.sha256(staged_file)


def test_analyzer_resolves_and_verifies_sealed_key_and_manifest(tmp_path: Path) -> None:
    packet = tmp_path / "v5"
    keys = packet / "KEYS_DO_NOT_DISTRIBUTE"
    keys.mkdir(parents=True)
    key = keys / "case_key.csv"
    manifest = keys / "panel_manifest.csv"
    key.write_text("case_id\nQ10000\n")
    manifest.write_text("case_id,panel_number\nQ10000,1\n")
    receipt_path = keys / "packet_receipt.json"
    receipt = {
        "status": "PASS",
        "problems": [],
        "scientific_status": "GENERATED_UNREAD",
        "analysis_executed": False,
        "unblinding_performed": False,
        "analysis_result": None,
        "builder": builder.file_record(Path(builder.__file__).resolve()),
        "frozen_analyzer": builder.file_record(Path(analyzer.__file__).resolve()),
        "outputs": [builder.file_record(key)],
        "manifests": [builder.file_record(manifest)],
    }
    receipt_path.write_text(json.dumps(receipt))
    analyzer.verify_packet_receipt(receipt_path, key)
    manifest.write_text("tampered\n")
    with pytest.raises(analyzer.ValidationError, match="identity does not match"):
        analyzer.verify_packet_receipt(receipt_path, key)


def test_forced_post_rename_mismatch_leaves_no_pass_receipt(tmp_path: Path) -> None:
    staging = tmp_path / ".building"
    output = tmp_path / "v5"
    keys = staging / "KEYS_DO_NOT_DISTRIBUTE"
    keys.mkdir(parents=True)
    payload = staging / "payload.txt"
    payload.write_text("original\n")
    bad_record = builder.staged_file_record(payload, staging, output)
    bad_record["sha256"] = "0" * 64
    tool_record = builder.file_record(Path(builder.__file__).resolve())
    analyzer_record = builder.file_record(Path(analyzer.__file__).resolve())
    receipt = {
        "status": "PASS",
        "problems": [],
        "scientific_status": "GENERATED_UNREAD",
        "analysis_executed": False,
        "unblinding_performed": False,
        "analysis_result": None,
        "builder": tool_record,
        "frozen_analyzer": analyzer_record,
        "frozen_inputs": [],
        "source_wsi_inputs": [],
        "outputs": [bad_record],
        "manifests": [],
    }
    problems = builder.finalize_staged_packet(staging, output, receipt)
    assert problems
    assert output.is_dir()
    assert not (output / "KEYS_DO_NOT_DISTRIBUTE" / "packet_receipt.json").exists()
    assert (output / "DO_NOT_RELEASE_BUILD_FAILED.txt").is_file()


def test_blank_generated_forms_are_refused(tmp_path: Path) -> None:
    case_ids = [f"Q{10_000 + index:05d}" for index in range(60)]
    key = pd.DataFrame(
        {
            "case_id": case_ids,
            "cohort": ["CPTAC"] * 60,
            "kras": ["mutant"] * 60,
            "p17_group": ["absent"] * 60,
            "p17_abundance": [0.0] * 60,
            "p28_abundance": [0.0] * 60,
        }
    )
    scores = pd.DataFrame({column: [""] * 60 for column in analyzer.SCORE_COLUMNS})
    scores["case_id"] = case_ids
    reviewer = pd.DataFrame({column: [""] for column in analyzer.REVIEWER_COLUMNS})
    key_path = tmp_path / "key.csv"
    scores_path = tmp_path / "scores.csv"
    reviewer_path = tmp_path / "reviewer.csv"
    key.to_csv(key_path, index=False)
    scores.to_csv(scores_path, index=False)
    reviewer.to_csv(reviewer_path, index=False)
    with pytest.raises(analyzer.ValidationError, match="CONFIRMATORY ANALYSIS REFUSED"):
        analyzer.read_returned_forms(scores_path, reviewer_path, key_path)


def test_deterministic_60_case_sampling_contract() -> None:
    exposed = builder.previously_exposed_patients()
    frame = builder.assign_p17_groups(builder.load_case_frame())
    first = builder.select_cases(frame)
    second = builder.select_cases(frame)
    assert len(exposed) == 191
    assert first[["case_id", "patient_id"]].equals(second[["case_id", "patient_id"]])
    assert len(first) == first["patient_id"].nunique() == first["slide_id"].nunique() == 60
    assert first.groupby("p17_group").size().to_dict() == {
        "absent": 20,
        "positive_high": 20,
        "positive_low": 20,
    }
    assert first.groupby("kras").size().to_dict() == {"mutant": 30, "wild_type": 30}
    assert set(first["patient_id"]) & exposed == set()
    cells = first.groupby(["cohort", "kras", "p17_group"]).size()
    assert len(cells) == 24
    assert set(cells.to_list()) == {2, 3}
    assert all(builder.resolve_slide(slide_id).is_file() for slide_id in first["slide_id"])
