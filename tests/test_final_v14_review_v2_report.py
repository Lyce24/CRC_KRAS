"""Regression checks for annotation-only publication of the finalized review."""
import csv
import shutil
from pathlib import Path

import pytest

from tools import final_v14_review_v2_report as report


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    output = tmp_path_factory.mktemp("reader-v2-report")
    result = report.build(output)
    assert result["status"] == "READY_FOR_READER_V2_PUBLICATION"
    return output


def test_finalized_ratings_replace_old_missing_data(rendered):
    summary = report.read(rendered / "reader_revision.json")
    assert (summary["primary_category_count"], summary["confidence_count"], summary["artifact_response_count"]) == (40, 40, 40)
    assert summary["controlling_eligible_count"] == 31
    assert (summary["repeatability_exact"], summary["repeatability_partial"]) == (6, 1)
    assert summary["repeatability_numerator"] == 7
    assert summary["historical_name_gate"] == "NAME_GATE_FAIL"
    assert summary["response_only_gate_status"] == "NAME_GATE_PASS"
    assert set(summary["metadata_recorded_counts"].values()) == {0}


def test_reader_supplied_discrepancy_and_stromal_split_are_preserved(rendered):
    values = {int(r["Concept"]): r for r in report.rows(rendered / "tables/Mentor_annotations.csv")}
    assert values[0]["Primary category"] == "malignant solid or poorly differentiated epithelium"
    assert values[0]["Mentor comment"].startswith("Most are benign fibrosis")
    assert values[1]["Primary category"] == values[2]["Primary category"] == "desmoplastic"
    assert values[4]["Primary category"] == "benign fibrosis"
    assert values[11]["Artifact flag"] == "yes"
    assert values[11]["V2 response eligibility"] == "False"
    evidence = {int(r["prototype_id"]): r for r in report.rows(rendered / "tables/All32_evidence.csv")}
    assert evidence[6]["eligible_for_named_ref"] == values[6]["Historical named-set eligible"] == "True"
    assert sum(r["eligible_for_named_ref"] == "True" for r in evidence.values()) == 1
    assert sum(r["response_eligibility_descriptive"] == "True" for r in evidence.values()) == 31


def test_every_preexisting_statistical_field_is_unchanged(rendered):
    old, new = report.rows(report.OLD_WIDE), report.rows(rendered / "tables/All32_evidence.csv")
    statistical = [k for k in old[0] if k.startswith(("abundance__", "attention__", "ridge__", "logistic__"))]
    assert len(statistical) > 50
    for a, b in zip(old, new, strict=True):
        assert {k: a[k] for k in statistical} == {k: b[k] for k in statistical}
    assert report.verify_outputs(rendered)["all_original_numerical_workbook_values_identical"]


def test_numerical_table_tampering_fails_closed(rendered, tmp_path):
    copy = tmp_path / "tampered"
    shutil.copytree(rendered, copy)
    path = copy / "tables/M1_all_estimands.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Numerical table changed"):
        report.verify_outputs(copy)


def test_statistic_join_tampering_is_rejected(rendered, tmp_path):
    copy = tmp_path / "tampered"
    shutil.copytree(rendered, copy)
    path = copy / "tables/All32_evidence.csv"
    values = report.rows(path)
    values[22]["abundance__kras__standardized_effect"] = "0.999"
    report.write_csv(path, values)
    with pytest.raises(ValueError, match="Reader/statistic mismatch"):
        report.verify_outputs(copy)


def test_historical_report_snapshot_is_byte_bound():
    mapping = report.snapshot_mapping()
    assert len(mapping) == 52
    assert str(report.REPORT / "final_bundle_receipt.json") in mapping
    assert mapping[str(report.REPORT / "Results.md")]["path"].startswith(str(report.SNAPSHOT))
