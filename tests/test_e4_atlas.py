"""Contract tests for E3b's analysis layer: the effect table and the four buckets.

The bucket a prototype lands in is the paper's claim about it — "KRAS-associated
and dependency-robust" versus "shared MSI/BRAF-context morphology" is the whole
point of running the A-and-D pair. These tests pin that mapping, and the
significance rule (FDR *and* a CI excluding 0.5) that feeds it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim4_morphologic_atlas_base as e3b  # noqa: E402


def _frame(n: int, effects: dict[int, float], seed: int = 0) -> pd.DataFrame:
    """Patient x prototype abundances; ``effects`` shifts mutants for some prototypes."""
    rng = np.random.default_rng(seed)
    rows = []
    for patient in range(n):
        mutant = int(patient % 2 == 0)
        for prototype in range(3):
            rows.append({
                "patient_id": f"p{patient}",
                "prototype": prototype,
                "is_mutant": mutant,
                "abundance": rng.normal(0.0, 1.0) + effects.get(prototype, 0.0) * mutant,
            })
    return pd.DataFrame(rows)


# ── the effect table ─────────────────────────────────────────────────────────
def test_effect_table_finds_the_planted_prototype_and_spares_the_others():
    frame = _frame(200, effects={1: 1.5}, seed=1)
    table = e3b._effect_table(frame, "is_mutant", "abundance", n_bootstrap=200, seed=7)
    assert list(table["prototype"]) == [0, 1, 2]
    planted = table[table["prototype"].eq(1)].iloc[0]
    assert planted["auc"] > 0.7
    assert planted["significant"]
    assert not table[table["prototype"].ne(1)]["significant"].any()


def test_effect_table_requires_both_fdr_and_a_ci_away_from_chance():
    # Every prototype is null: nothing may be called significant even though
    # three simultaneous tests would produce a small raw p sooner or later.
    table = e3b._effect_table(_frame(200, effects={}, seed=2), "is_mutant", "abundance",
                              n_bootstrap=200, seed=8)
    assert not table["significant"].any()
    assert (table["q"] >= table["p"]).all()


# ── the four buckets ─────────────────────────────────────────────────────────
def _block(a, d, ctx, attn=None, top_share=0.3, top_key="TCGA"):
    def stats(spec):
        if spec is None:
            return {}
        auc, significant = spec
        return {"auc": auc, "significant": significant, "ci_low": auc - 0.05,
                "ci_high": auc + 0.05, "q": 0.001 if significant else 0.9}
    return {
        "abundance": {"A": stats(a), "D": stats(d), "context_in_wt": stats(ctx)},
        "attn_mass_mean": {"A": stats(attn)},
        "cohort_concentration": {"top_share": top_share, "top_key": top_key},
    }


def test_an_association_that_survives_set_D_is_dependency_robust():
    out = e3b._classify_prototype(_block(a=(0.62, True), d=(0.60, True), ctx=(0.52, False)))
    assert out["bucket"] == "kras_associated_dependency_robust"


def test_an_association_lost_in_D_with_a_context_effect_is_shared_context_morphology():
    out = e3b._classify_prototype(_block(a=(0.62, True), d=(0.52, False), ctx=(0.65, True)))
    assert out["bucket"] == "shared_msi_braf_context_morphology"
    assert "association_lost_in_D" in out["flags"]
    assert "context_associated_among_WT" in out["flags"]


def test_an_association_lost_in_D_without_a_context_effect_is_kept_separate():
    # Not the same claim as "shared MSI/BRAF morphology": the association simply
    # does not survive restriction, and no measured context explains it.
    out = e3b._classify_prototype(_block(a=(0.62, True), d=(0.52, False), ctx=(0.51, False)))
    assert out["bucket"] == "kras_associated_attenuates_in_D"


def test_a_sign_flip_between_A_and_D_is_not_dependency_robust():
    out = e3b._classify_prototype(_block(a=(0.62, True), d=(0.38, True), ctx=(0.51, False)))
    assert out["bucket"] != "kras_associated_dependency_robust"


def test_a_single_cohort_prototype_with_no_kras_effect_is_flagged_as_cohort_associated():
    out = e3b._classify_prototype(
        _block(a=(0.51, False), d=(0.50, False), ctx=(0.50, False), top_share=0.92,
               top_key="SurGen")
    )
    assert out["bucket"] == "cohort_associated"
    assert any("cohort_dominated" in f for f in out["flags"])


def test_attention_is_a_flag_and_never_promotes_a_prototype_to_a_bucket():
    # The model leaning on a morphology is a statement about the MODEL; the
    # bucket is a statement about the TISSUE, so attention must not decide it.
    out = e3b._classify_prototype(
        _block(a=(0.51, False), d=(0.50, False), ctx=(0.50, False), attn=(0.70, True))
    )
    assert out["bucket"] == "not_kras_associated"
    assert "attention_differs_by_kras(higher_in_mutants)" in out["flags"]


def test_attention_flag_preserves_the_direction_of_the_effect():
    out = e3b._classify_prototype(
        _block(a=(0.51, False), d=(0.50, False), ctx=(0.50, False), attn=(0.40, True))
    )
    assert "attention_differs_by_kras(higher_in_wild_type)" in out["flags"]


def test_attention_only_prototype_is_selected_for_blinded_followup():
    readout = {
        5: {
            "group": "not_kras_associated",
            "kras_association": {"significant": False},
            "model_attends": True,
            "shortcut_flags": [],
        },
        7: {
            "group": "dependency_robust_transport_inconclusive",
            "kras_association": {"significant": True},
            "model_attends": True,
            "shortcut_flags": [],
        },
        9: {
            "group": "shortcut_technical",
            "kras_association": {"significant": False},
            "model_attends": True,
            "shortcut_flags": ["cohort_dominated(TCGA 0.90)"],
        },
    }
    selected = e3b.review_prototypes(readout, n_technical=0)
    assert selected["attention_followup"] == [5]
    assert selected["claimed"] == [7]
    assert selected["suppressed"] == [9]


def test_montage_addendum_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    review_root = tmp_path / "reviews"
    montage_root = tmp_path / "montages"
    image = review_root / "k32" / "montages" / "M11.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"locked")
    monkeypatch.setattr(e3b, "REVIEW_ROOT", review_root)
    monkeypatch.setattr(e3b, "MONTAGE_DIR", montage_root)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        e3b.cmd_montage_addendum(SimpleNamespace(
            k=32, montage_id="M11", prototype=5, force=False,
        ))


# ── arm wiring ───────────────────────────────────────────────────────────────
def test_the_surgen_transport_arm_is_sr1482_only_on_both_sides():
    assert e3b.ARMS["sr1482_primary"].subcohort == "SR1482"
    assert e3b.ARMS["sr1482_primary"].target == "SurGen"
    assert e3b.ARMS["sr1482_metastatic"].target == "SurGen"
    assert ("SR1482", "sr1482_primary", "sr1482_metastatic") in e3b.TRANSPORT_PAIRS


def test_only_the_development_arm_uses_out_of_fold_attention():
    assert e3b.ARMS["e0"].mode == "oof"
    assert all(a.mode == "refit" for name, a in e3b.ARMS.items() if name != "e0")
    assert all(a.target is not None for a in e3b.ARMS.values() if a.mode == "refit")


def test_patient_level_aggregation_weights_slides_equally_within_a_patient():
    slides = pd.DataFrame([
        {"arm": "e0", "slide_id": "s1", "patient_id": "p", "prototype": 0,
         "abundance": 0.2, "attn_mass_mean": 0.1, "n_tiles_prototype": 10,
         "n_tiles_slide": 100, "role": "primary", "cohort": "TCGA", "label": 1},
        {"arm": "e0", "slide_id": "s2", "patient_id": "p", "prototype": 0,
         "abundance": 0.4, "attn_mass_mean": 0.3, "n_tiles_prototype": 4000,
         "n_tiles_slide": 10000, "role": "primary", "cohort": "TCGA", "label": 1},
    ])
    out = e3b._to_patient_level(slides)
    assert len(out) == 1
    # The 10,000-tile slide does not outvote the 100-tile slide.
    assert out.iloc[0]["abundance"] == pytest.approx(0.3)
    assert out.iloc[0]["attn_mass_mean"] == pytest.approx(0.2)
    assert out.iloc[0]["n_slides"] == 2


# ── transport-state contract ─────────────────────────────────────────────────
def _transport_row(*, primary=False, metastatic=False, same=True, conserved=False,
                   changed=False, delta=-0.10):
    return {
        "prototype": 7,
        "auc_primary": 0.62,
        "auc_metastatic": 0.62 + delta,
        "significant_primary": primary,
        "significant_metastatic": metastatic,
        "same_direction": same,
        "conserved": conserved,
        "changed": changed,
        "delta_auc_metastatic_minus_primary": delta,
        "delta_ci_low": delta - 0.03,
        "delta_ci_high": delta + 0.03,
        "delta_q": 0.01 if changed else 0.9,
    }


def _transport(**row):
    return {
        "conservation": {
            "RIH": {
                "n_primary": 145,
                "n_metastatic": 77,
                "by_quantity": {"abundance": [_transport_row(**row)]},
            }
        }
    }


@pytest.mark.parametrize(
    ("row", "state"),
    [
        ({}, "underpowered"),
        ({"primary": True}, "inconclusive"),
        ({"primary": True, "changed": True}, "changed"),
        ({"primary": True, "metastatic": True, "conserved": True}, "conserved"),
    ],
)
def test_conservation_states_do_not_turn_missing_significance_into_change(row, state):
    assert e3b._conservation_state(7, _transport(**row))["state"] == state


def test_conservation_is_heterogeneous_when_cohorts_directly_conflict():
    transport = _transport(primary=True, metastatic=True, conserved=True, delta=0.0)
    transport["conservation"]["SR1482"] = {
        "n_primary": 324,
        "n_metastatic": 74,
        "by_quantity": {
            "abundance": [_transport_row(primary=True, changed=True, delta=-0.15)]
        },
    }
    assert e3b._conservation_state(7, transport)["state"] == "heterogeneous"


def test_direct_auc_difference_detects_a_planted_attenuation():
    rng = np.random.default_rng(11)
    labels = np.tile([0, 1], 200)
    primary = rng.normal(size=400) + labels * 1.5
    metastatic = rng.normal(size=400)
    result = e3b._auc_difference(
        primary, labels, metastatic, labels, n_bootstrap=400, seed=12
    )
    assert result["delta_auc_metastatic_minus_primary"] < -0.20
    assert result["delta_ci_high"] < 0


def _final_spec(*, a=(0.62, True), d=(0.60, True), ctx=(0.50, False),
                attn=(0.55, False)):
    block = _block(a=a, d=d, ctx=ctx, attn=attn)
    block["classification"] = e3b._classify_prototype(block)
    block["cohort_reproducibility"] = {}
    return {"prototypes": {"7": block}}


def test_final_readout_keeps_transport_inconclusive_separate_from_changed():
    row = e3b.final_readout(_final_spec(), _transport(primary=True))[7]
    assert row["group"] == "dependency_robust_transport_inconclusive"


def test_final_readout_does_not_call_unexplained_attenuation_context_dependent():
    row = e3b.final_readout(
        _final_spec(d=(0.52, False), ctx=(0.50, False)), _transport()
    )[7]
    assert row["group"] == "restriction_sensitive_unexplained"


def test_final_readout_preserves_attention_cohort_reproducibility():
    spec = _final_spec(attn=(0.45, True))
    spec["prototypes"]["7"]["cohort_reproducibility"] = {
        "abundance": {"agreeing": 2, "n_cohorts": 4, "direction": "higher_in_mutant"},
        "attn_mass_mean": {"agreeing": 4, "n_cohorts": 4, "direction": "higher_in_wt"},
    }
    row = e3b.final_readout(spec, _transport())[7]
    assert row["cohort_reproducibility"]["agreeing"] == 2
    assert row["attention_cohort_reproducibility"] == {
        "agreeing": 4,
        "n_cohorts": 4,
        "direction": "higher_in_wt",
    }


# ── completed-review contract ────────────────────────────────────────────────
def _write_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> None:
    review_root = tmp_path / "reviews"
    montage_root = tmp_path / "montages"
    (review_root / "k32").mkdir(parents=True)
    (montage_root / "k32").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(review_root / "k32" / "review_form.csv", index=False)
    key_rows = [
        {"montage_id": row["montage_id"], "slot": slot,
         "prototype": int(row["montage_id"][1:])}
        for row in rows for slot in range(int(row["n_tiles"]))
    ]
    pd.DataFrame(key_rows).to_csv(
        montage_root / "k32" / "KEY_do_not_open_before_review.csv", index=False
    )
    monkeypatch.setattr(e3b, "REVIEW_ROOT", review_root)
    monkeypatch.setattr(e3b, "MONTAGE_DIR", montage_root)


def test_review_join_is_complete_structured_and_not_truncated(tmp_path, monkeypatch):
    long_text = "well-differentiated tumour at a boundary between tumour and nearby blank space"
    _write_review(tmp_path, monkeypatch, [
        {"montage_id": "M07", "n_tiles": 2, "architecture": long_text, "artifact": ""},
    ])
    review = e3b._load_review(32)
    assert review["status"] == "complete"
    assert review["annotations"][7]["architecture"] == long_text
    assert long_text in e3b._load_annotations(32)[7]
    assert len(review["form_sha256"]) == 64


def _write_completed_followup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_review(tmp_path, monkeypatch, [
        {"montage_id": "M01", "n_tiles": 1, "architecture": "glandular"},
    ])
    packet = tmp_path / "reviews" / "k32"
    key_dir = tmp_path / "montages" / "k32"
    (packet / "montages").mkdir()
    (packet / "montages" / "M01.jpg").write_bytes(b"m01")
    (packet / "montages" / "M11.jpg").write_bytes(b"m11")
    raw = packet / "completed_review.md"
    raw.write_text("completed blinded answers")
    addendum_key = key_dir / "KEY_M11_followup_do_not_open_before_review.csv"
    pd.DataFrame([
        {"montage_id": "M11", "slot": 0, "prototype": 11,
         "selection_stratum": "attention_followup"},
    ]).to_csv(addendum_key, index=False)
    base_form = packet / "review_form.csv"
    base_key = key_dir / "KEY_do_not_open_before_review.csv"
    structured = {
        "schema_version": 1,
        "extraction_status": "curated_from_completed_blinded_followup",
        "source_document": raw.name,
        "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "reviewer_metadata": {"blinding_confirmation": "not_recorded"},
        "provenance": {
            "base_form_sha256": hashlib.sha256(base_form.read_bytes()).hexdigest(),
            "base_key_sha256": hashlib.sha256(base_key.read_bytes()).hexdigest(),
            "addendum_key": addendum_key.name,
            "addendum_key_sha256": hashlib.sha256(addendum_key.read_bytes()).hexdigest(),
            "montage_sha256": {
                montage_id: hashlib.sha256(
                    (packet / "montages" / f"{montage_id}.jpg").read_bytes()
                ).hexdigest()
                for montage_id in ("M01", "M11")
            },
        },
        "montages": {
            "M01": {"n_tiles": 1, "canonical_description": "tumour glands"},
            "M11": {"n_tiles": 1, "canonical_description": "benign glands"},
        },
        "cross_montage": {},
        "global_quality_flags": ["single reader"],
    }
    structured_path = packet / "completed_review_structured.json"
    structured_path.write_text(json.dumps(structured))
    return raw


def test_completed_followup_unblinds_only_through_external_keys(tmp_path, monkeypatch):
    _write_completed_followup(tmp_path, monkeypatch)
    followup = e3b._load_completed_followup(32)
    assert followup["status"] == "complete"
    assert followup["n_assessments"] == 2
    assert followup["annotations"][1]["canonical_description"] == "tumour glands"
    assert followup["annotations"][11]["canonical_description"] == "benign glands"
    assert followup["montage_by_prototype"] == {1: "M01", 11: "M11"}
    assert e3b._format_annotation(followup["annotations"][11]) == "benign glands"


def test_completed_followup_rejects_a_changed_raw_source(tmp_path, monkeypatch):
    raw = _write_completed_followup(tmp_path, monkeypatch)
    raw.write_text("answers changed after structured extraction")
    with pytest.raises(SystemExit, match="source checksum"):
        e3b._load_completed_followup(32)


def test_completed_followup_requires_attention_followup_key_stratum(
    tmp_path, monkeypatch
):
    _write_completed_followup(tmp_path, monkeypatch)
    key = tmp_path / "montages" / "k32" / "KEY_M11_followup_do_not_open_before_review.csv"
    rows = pd.read_csv(key)
    rows["selection_stratum"] = "technical"
    rows.to_csv(key, index=False)
    structured_path = tmp_path / "reviews" / "k32" / "completed_review_structured.json"
    structured = json.loads(structured_path.read_text())
    structured["provenance"]["addendum_key_sha256"] = hashlib.sha256(
        key.read_bytes()
    ).hexdigest()
    structured_path.write_text(json.dumps(structured))
    with pytest.raises(SystemExit, match="attention_followup selection stratum"):
        e3b._load_completed_followup(32)


def test_completed_followup_rejects_montage_id_reused_across_keys(
    tmp_path, monkeypatch
):
    _write_completed_followup(tmp_path, monkeypatch)
    key = tmp_path / "montages" / "k32" / "KEY_M11_followup_do_not_open_before_review.csv"
    rows = pd.read_csv(key)
    rows["montage_id"] = "M01"
    rows["prototype"] = 1
    rows.to_csv(key, index=False)
    structured_path = tmp_path / "reviews" / "k32" / "completed_review_structured.json"
    structured = json.loads(structured_path.read_text())
    structured["provenance"]["addendum_key_sha256"] = hashlib.sha256(
        key.read_bytes()
    ).hexdigest()
    structured_path.write_text(json.dumps(structured))
    with pytest.raises(SystemExit, match="reuse montage ID"):
        e3b._load_completed_followup(32)


def test_partially_completed_review_is_rejected(tmp_path, monkeypatch):
    _write_review(tmp_path, monkeypatch, [
        {"montage_id": "M01", "n_tiles": 1, "architecture": "glandular"},
        {"montage_id": "M02", "n_tiles": 1, "architecture": ""},
    ])
    with pytest.raises(SystemExit, match="partially completed"):
        e3b._load_review(32)


def test_completed_review_packet_cannot_be_overwritten_without_force(tmp_path, monkeypatch):
    _write_review(tmp_path, monkeypatch, [
        {"montage_id": "M01", "n_tiles": 1, "architecture": "glandular"},
    ])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        e3b.cmd_montages(SimpleNamespace(k=32, force=False))
