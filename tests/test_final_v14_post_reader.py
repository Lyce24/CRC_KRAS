"""The amended reader intake cannot manufacture missing naming evidence."""
from __future__ import annotations

import copy
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import final_v14_post_reader as post  # noqa: E402


def fixture_naming():
    rows, keys = [], []
    for index, (prototype, occurrence) in enumerate(
        [(p, 0) for p in range(32)] + [(p, 1) for p in range(8)], start=1
    ):
        code = f"A{index:05d}"
        rows.append({"presentation_order": index, "blinded_code": code, "n_tiles": 12,
                     "primary_category": post.handoff.ONTOLOGY[0],
                     "secondary_category_1": None, "secondary_category_2": None,
                     "confidence_1_to_5": 4, "artifact_uninterpretable": "no",
                     "free_text_description": "Exact original text.  "})
        keys.append({"presentation_order": index - 1, "code": code, "prototype_id": prototype,
                     "occurrence": occurrence, "is_controlling_read": occurrence == 0})
    mapping = pd.DataFrame([{"outer_fold": fold, "source_prototype_id": p,
                             "reference_prototype_id": p, "cosine_similarity": 0.9}
                            for fold in range(5) for p in range(32)])
    return rows, pd.DataFrame(keys), mapping


def derive(rows, key, mapping):
    return post.derive_naming(rows, key, mapping, "DUPLICATE_PREFLIGHT_PASS")


def test_complete_fixture_passes_all_three_quality_conjuncts():
    result = derive(*fixture_naming())
    assert result["name_gate_status"] == "NAME_GATE_PASS"
    assert result["NAMED_REF"] == list(range(32))
    assert result["repeatability"]["numerator"] == 8
    assert result["repeatability"]["ci95_clopper_pearson"][1] == 1


def test_missing_evidence_is_not_repaired_by_free_text_or_duplicate():
    rows, key, mapping = fixture_naming()
    rows[0]["primary_category"] = None
    rows[0]["confidence_1_to_5"] = None
    rows[0]["artifact_uninterpretable"] = None
    rows[0]["free_text_description"] = "Clearly malignant glands, confidence 5, no artifacts."
    before = copy.deepcopy(rows)
    result = derive(rows, key, mapping)
    assert rows == before
    assert 0 not in result["NAMED_REF"]
    assert result["controlling_reads"][0]["exclusion_reasons"] == [
        "MISSING_PRIMARY_CATEGORY", "MISSING_CONFIDENCE", "MISSING_ARTIFACT_RESPONSE"]
    assert result["repeatability"]["denominator"] == 8
    assert result["repeatability"]["numerator"] == 7


def test_missing_artifact_flags_do_not_establish_formal_repeatability():
    rows, key, mapping = fixture_naming()
    for row in rows:
        row["artifact_uninterpretable"] = None
    result = derive(rows, key, mapping)
    assert result["name_gate_status"] == "NAME_GATE_FAIL"
    assert result["NAMED_REF"] == []
    assert result["repeatability"]["numerator"] == 0
    assert result["repeatability"]["categorical_concordance_ignoring_missing_artifact_fields_descriptive_only"] == {"exact": 8}


def test_eight_duplicate_denominator_cannot_shrink():
    rows, key, mapping = fixture_naming()
    with pytest.raises(post.PostReaderError, match="bijection"):
        derive(rows[:-1], key.iloc[:-1], mapping)


def test_partial_agreement_uses_only_structured_secondary_categories():
    rows, _, _ = fixture_naming()
    a, b = copy.deepcopy(rows[:2])
    b["primary_category"] = post.handoff.ONTOLOGY[1]
    b["secondary_category_1"] = a["primary_category"]
    assert post.category_agreement(a, b) == "partial"
    b["secondary_category_1"] = None
    b["free_text_description"] = a["primary_category"]
    assert post.category_agreement(a, b) == "different"


def test_mapping_threshold_and_all_fold_attribution_are_separate():
    rows, key, mapping = fixture_naming()
    mapping.loc[(mapping.outer_fold == 1) & (mapping.source_prototype_id == 3), "cosine_similarity"] = 0.7999
    result = derive(rows, key, mapping)
    assert 3 in result["NAMED_REF"]
    assert 3 in result["NAMED_OOF"]["0"]
    assert 3 not in result["NAMED_OOF"]["1"]
    assert 3 not in result["ATTRIBUTABLE_REF"]
    assert result["ALL32"] == list(range(32))


def test_joint_legacy_matching_resolves_collision_and_ties_deterministically():
    assert post.joint_anchor_assignment(np.array([[1.0, 0.8], [1.0, 0.7]])) == (1, 0)
    assert post.joint_anchor_assignment(np.ones((2, 4))) == (0, 1)
    with pytest.raises(ValueError, match="finite"):
        post.joint_anchor_assignment(np.array([[np.nan, 1], [1, 1]]))


def test_literal_xlsx_preserves_missingness_and_comment_whitespace(tmp_path, monkeypatch):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    expected = [{"presentation_order": str(i + 1), "blinded_code": "AAAA" + alphabet[i // 32] + alphabet[i % 32],
                 "n_tiles": "12", **{field: "" for field in post.handoff.RESPONSE_COLUMNS}}
                for i in range(40)]
    rubric = "# Rubric\n" + "\n".join(f"- {value}" for value in post.handoff.ONTOLOGY)
    blank = post.handoff.build_workbook(expected, rubric, "2026-09-04T05:15:42+00:00")
    mirror = tmp_path / "public"
    mirror.mkdir()
    (mirror / post.handoff.WORKBOOK_NAME).write_bytes(blank)
    monkeypatch.setattr(post.handoff, "AUDIT_ROOT", tmp_path)
    workbook = load_workbook(io.BytesIO(blank), data_only=False)
    worksheet = workbook[post.handoff.REVIEW_SHEET]
    worksheet["H2"] = "  Literal original comment.  "
    worksheet["E2"] = post.handoff.ONTOLOGY[0]
    buffer = io.BytesIO()
    workbook.save(buffer)
    rows, validation = post.literal_workbook_rows(buffer.getvalue(), expected, rubric)
    assert rows[0]["free_text_description"] == "  Literal original comment.  "
    assert rows[0]["confidence_1_to_5"] is None
    assert rows[0]["artifact_uninterpretable"] is None
    assert rows[0]["review_date"] is None
    assert validation["missing_response_counts"]["primary_category"] == 39
    worksheet["B2"] = "DRIFTD"
    buffer = io.BytesIO()
    workbook.save(buffer)
    with pytest.raises(post.PostReaderError, match="identity changed"):
        post.literal_workbook_rows(buffer.getvalue(), expected, rubric)


def test_seal_detects_payload_drift(tmp_path):
    path = tmp_path / "frozen.json"
    post.publish(path, {"value": 1})
    assert post.verify_seal(path) == {"value": 1}
    path.chmod(0o600)
    path.write_text('{"value": 2}\n')
    with pytest.raises(post.PostReaderError, match="Identity drift"):
        post.verify_seal(path)
