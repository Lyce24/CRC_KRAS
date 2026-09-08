"""Regression checks for the actual finalized reader revision and its boundaries."""
from __future__ import annotations

import copy
import io
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

from tools import final_v14_reader_v2 as reader


@pytest.fixture(scope="module")
def inputs():
    context, historical, _ = reader.sources()
    payload = reader.WORKBOOK.read_bytes()
    baseline = Path(context["raw"]["path"]).read_bytes()
    parsed = reader.parse_workbook(payload, baseline, context["literal"]["rows"])
    return context, historical, payload, baseline, parsed


def changed_workbook(payload, coordinate, value):
    workbook = load_workbook(io.BytesIO(payload), data_only=False)
    workbook[reader.handoff.REVIEW_SHEET][coordinate] = value
    buffer = io.BytesIO()
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


def test_actual_finalized_workbook_preserves_added_columns_and_revised_ontology(inputs):
    _, _, _, _, parsed = inputs
    assert len(parsed["rows"]) == 40
    assert parsed["workbook_context"]["headers"][2:4] == ["Column1", "Column2"]
    assert parsed["helper_columns"][0]["Column2"] == '=HYPERLINK(C2&B2&".jpg", "Open")'
    assert parsed["helper_columns"][0]["Column1"].startswith("G:")
    assert parsed["validation"]["helper_formulas_evaluated"] is False
    ontology = parsed["validation"]["ontology"]
    assert ontology["added"] == ["desmoplastic", "benign fibrosis"]
    assert ontology["removed"] == ["desmoplastic/fibrous stroma"]
    assert parsed["rows"][1]["primary_category"] == "benign fibrosis"
    assert parsed["validation"]["scientific_rating_fields_complete"] is True
    assert parsed["validation"]["confidence_counts"] == {4: 6, 5: 34}
    assert parsed["validation"]["artifact_counts"] == {"no": 39, "yes": 1}
    assert parsed["changed_field_counts"] == {"artifact_uninterpretable": 39, "confidence_1_to_5": 39,
        "free_text_description": 7, "primary_category": 15, "secondary_category_1": 22, "secondary_category_2": 8}
    for row in parsed["rows"]:
        assert all(row[field] is None for field in ("review_status", "reviewer_id", "review_date", "blinding_attestation"))


def test_revised_response_rules_do_not_retroactively_upgrade_original_gate(inputs):
    context, historical, _, _, parsed = inputs
    prior = copy.deepcopy(historical)
    summary = reader.revised_summary(parsed, historical, pd.read_csv(context["key"]["path"]),
                                     pd.read_csv(context["mapping"]["path"]), context["duplicate_preflight"])
    assert historical == prior
    assert summary["name_gate_status"] == "NAME_GATE_FAIL"
    assert summary["response_only_gate_status"] == "NAME_GATE_PASS"
    assert summary["NAMED_REF"] == historical["NAMED_REF"] == [6]
    assert summary["descriptive_eligible_ref"] == [value for value in range(32) if value != 11]
    repeat = summary["repeatability"]
    assert (repeat["numerator"], repeat["denominator"], repeat["exact"], repeat["partial"]) == (7, 8, 6, 1)
    assert repeat["ci95_clopper_pearson"] == pytest.approx([0.47349032912479344, 0.9968402764687481])
    assert summary["historical_name_gate"]["repeatability"]["numerator"] == 0
    p11 = summary["controlling_reads"][11]
    assert p11["response"]["artifact_uninterpretable"] == "yes"
    assert p11["response_eligibility_descriptive"] is False
    assert p11["response_exclusion_reasons"] == ["ARTIFACT_UNINTERPRETABLE"]
    assert [row["prototype_id"] for row in summary["controlling_reads"] if row["eligible_for_named_ref"]] == [6]
    assert all(count == 40 for count in summary["missing_metadata_counts"].values())


@pytest.mark.parametrize("coordinate,value,error", [
    ("B2", "CHANGED", "Frozen presentation identity"),
    ("G2", "desmoplastic/fibrous stroma", "outside revised ontology"),
    ("J2", '=HYPERLINK("https://example.com","Open")', "Formula outside"),
    ("D2", '=HYPERLINK("https://example.com","Open")', "Unexpected helper formula"),
])
def test_rejects_identity_drift_unknown_categories_and_response_formulas(inputs, coordinate, value, error):
    context, _, payload, baseline, _ = inputs
    with pytest.raises(reader.original.PostReaderError, match=error):
        reader.parse_workbook(changed_workbook(payload, coordinate, value), baseline, context["literal"]["rows"])


def test_raw_return_is_byte_identical_and_cannot_be_overwritten(tmp_path, inputs):
    _, _, payload, _, _ = inputs
    source = tmp_path / "input.xlsx"
    source.write_bytes(payload)
    output = tmp_path / "successor"
    pin = reader.seal_raw_return(source, output)
    raw = Path(pin["path"])
    assert raw.read_bytes() == payload
    assert reader.seal_raw_return(source, output) == pin
    source.write_bytes(payload + b"changed")
    with pytest.raises(reader.original.PostReaderError, match="Refusing to replace"):
        reader.seal_raw_return(source, output)
    assert raw.read_bytes() == payload
    assert reader.original.verify_seal(output / "raw_return/receipt.json")["artifact"] == pin


def test_successor_ingestion_and_verification_are_replayable_and_detect_raw_tampering(tmp_path, inputs):
    _, _, payload, _, _ = inputs
    source = tmp_path / "input.xlsx"
    source.write_bytes(payload)
    output = tmp_path / "successor"
    receipt = reader.ingest(source, output)
    assert reader.ingest(source, output) == receipt
    assert reader.verify(output) == receipt
    assert receipt["descriptive_repeatability"]["numerator"] == 7
    amendment = reader.original.verify_seal(output / "governance_amendment.json")
    assert amendment["status"] == "USER_AUTHORIZED_FINAL_REVIEW_REVISION_AFTER_EXISTING_ANALYSES"
    assert amendment["prior_key_access_and_completed_analyses"] is True
    assert amendment["revision_session_timing_established"] is False
    raw = Path(receipt["raw_workbook"]["path"])
    raw.chmod(0o600)
    raw.write_bytes(payload + b"changed")
    with pytest.raises(reader.original.PostReaderError, match="Identity drift"):
        reader.verify(output)
