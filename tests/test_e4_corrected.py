"""Focused contracts for the append-only corrected Aim 4 runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim4_morphologic_atlas as corrected  # noqa: E402


def _effect_frame(*, one_class: bool = False) -> pd.DataFrame:
    rows: list[dict] = []
    for patient in range(12):
        label = 0 if one_class else patient % 2
        for prototype in range(corrected.K):
            rows.append({
                "patient_id": f"p{patient}",
                "prototype": prototype,
                "label": label,
                "abundance": float(label if prototype == 7 else patient % 3) / 3.0,
                "attn_mass_mean": 1.0 / corrected.K,
            })
    return pd.DataFrame(rows)


def test_fixed_effect_family_always_contains_exactly_32_hypotheses():
    table = corrected.fixed_effect_table(
        _effect_frame(), "label", "abundance", n_bootstrap=50, seed=10
    )
    assert table["prototype"].tolist() == list(range(32))
    assert len(table) == 32
    assert np.isfinite(table[["p", "q"]].to_numpy()).all()
    assert table.loc[table["prototype"].eq(7), "significant"].item()


def test_structural_rows_stay_in_family_with_p_one_and_cannot_be_significant():
    table = corrected.fixed_effect_table(
        _effect_frame(one_class=True), "label", "abundance", n_bootstrap=20, seed=11
    )
    assert len(table) == 32
    assert not table["estimable"].any()
    assert (table["p"] == 1.0).all()
    assert (table["q"] == 1.0).all()
    assert not table["significant"].any()
    assert set(table["structural_p_policy"]) == {"p=1_in_fixed_32_family"}


def test_effect_family_fails_closed_when_a_prototype_row_is_missing():
    frame = _effect_frame()
    frame = frame[~frame["prototype"].eq(31)]
    with pytest.raises(RuntimeError, match="prototype family is incomplete"):
        corrected.fixed_effect_table(
            frame, "label", "abundance", n_bootstrap=10, seed=12
        )


def test_fixed_direct_contrast_has_32_rows_and_structural_p_one():
    primary = _effect_frame(one_class=True)
    metastatic = _effect_frame(one_class=True)
    table = corrected.fixed_auc_difference_table(
        primary, metastatic, "abundance", n_bootstrap=10, seed=13
    )
    assert table["prototype"].tolist() == list(range(32))
    assert (table["delta_p"] == 1.0).all()
    assert (table["delta_q"] == 1.0).all()
    assert not table["changed"].any()


def test_output_roots_must_be_absolute_distinct_and_non_nested(tmp_path):
    input_root = tmp_path / "legacy"
    aim2_root = tmp_path / "aim2"
    input_root.mkdir()
    aim2_root.mkdir()
    with pytest.raises(ValueError, match="explicit absolute"):
        corrected._validate_roots(
            Path("legacy"), aim2_root, tmp_path / "new", output_exists=False
        )
    with pytest.raises(ValueError, match="distinct and non-nested"):
        corrected._validate_roots(
            input_root, aim2_root, input_root / "new", output_exists=False
        )


def test_archival_root_resolution_allows_consumed_legacy_root_to_be_absent(tmp_path):
    missing_legacy = tmp_path / "retired-legacy"
    aim2_root = tmp_path / "aim2"
    output = tmp_path / "corrected"
    aim2_root.mkdir()
    output.mkdir()
    resolved = corrected._validate_roots(
        missing_legacy,
        aim2_root,
        output,
        output_exists=True,
        input_exists=False,
    )
    assert resolved[0] == missing_legacy


def test_post_seal_review_commands_use_archival_inputs():
    assert {"import-reviews", "finalize"} <= corrected.ARCHIVAL_INPUT_COMMANDS
    assert "review-packets" not in corrected.ARCHIVAL_INPUT_COMMANDS


def test_post_seal_import_and_finalize_do_not_require_live_worktree(
    tmp_path, monkeypatch
):
    calls = []

    class StopAfterPrepared(RuntimeError):
        pass

    def prepared(_output, _bundle, *, require_live_source=True):
        calls.append(require_live_source)
        raise StopAfterPrepared

    monkeypatch.setattr(corrected, "_validate_prepared", prepared)
    bundle = _profile_bundle(tmp_path)
    with pytest.raises(StopAfterPrepared):
        corrected.import_completed_reviews(
            bundle,
            tmp_path,
            base_form=tmp_path / "base.csv",
            attention_addendum_form=tmp_path / "addendum.csv",
            apply=False,
        )
    with pytest.raises(StopAfterPrepared):
        corrected.finalize(bundle, tmp_path, apply=False)
    assert calls == [False, False]


def test_checkpoint_mapping_is_explicitly_inside_supplied_aim2_root(tmp_path):
    path = corrected._checkpoint_path(tmp_path, "RIH", 43)
    assert path == (
        tmp_path
        / "e2a/train/pb_cap8192/rih/seed43/final/refit/model.ckpt"
    )
    assert "/outputs/aim1/e2a/" not in str(path)


def test_malformed_interrupted_attention_stage_is_preserved_and_retryable(
    tmp_path, monkeypatch
):
    output = tmp_path / "output"
    stage = output / "_staging/attention/rih_primary_seed42.h5"
    stage.parent.mkdir(parents=True)
    with h5py.File(stage, "w") as handle:
        handle.create_group("unexpected_slide")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    bundle = corrected.InputBundle(
        input_root=tmp_path / "legacy",
        aim2_root=tmp_path / "aim2",
        manifests={
            "rih_primary": pd.DataFrame({"slide_id": ["expected_slide"]})
        },
        feature_files={},
        checkpoints={("RIH", 42): checkpoint},
        receipt={},
    )
    monkeypatch.setattr(corrected, "_validate_prepared", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(corrected, "_assert_numeric_not_completed", lambda *_args: None)
    monkeypatch.setattr(corrected, "_validate_attention_file", lambda *_args, **_kwargs: {})
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fresh_export(*, destination, **_kwargs):
        assert not destination.exists()
        destination.write_bytes(b"fresh-complete-h5")
        destination.with_suffix(".json").write_text("{}")

    monkeypatch.setattr(corrected.attention, "export_attention_refit", fresh_export)
    corrected.export_target_attention(
        bundle,
        output,
        arms=["rih_primary"],
        seeds=[42],
        apply=True,
    )
    assert (output / "attention/rih_primary_seed42.h5").read_bytes() == (
        b"fresh-complete-h5"
    )
    archived = list((stage.parent / "interrupted").rglob(stage.name))
    assert len(archived) == 1
    with h5py.File(archived[0], "r") as handle:
        assert set(handle) == {"unexpected_slide"}


def test_write_once_refuses_to_replace_even_identical_content(tmp_path):
    path = tmp_path / "artifact.json"
    corrected._write_json_once(path, {"status": "PASS"})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        corrected._write_json_once(path, {"status": "PASS"})


def test_strict_json_sanitizer_preserves_structural_null_without_nan():
    value = corrected._sanitize_json({"auc": float("nan"), "q": np.float64(1.0)})
    assert value == {"auc": None, "q": 1.0}
    assert json.dumps(value, allow_nan=False)


def test_replay_restores_empty_organ_concentration_without_restoring_real_nulls():
    serialized = {
        "organ_concentration": {"7": {"top_share": None, "top_key": None}},
        "effect": {"estimable": True, "structural_p_policy": None},
    }
    restored = corrected._restore_structural_nan(serialized)
    assert np.isnan(restored["organ_concentration"]["7"]["top_share"])
    assert restored["organ_concentration"]["7"]["top_key"] is None
    assert restored["effect"]["structural_p_policy"] is None
    assert corrected.legacy._shortcut_flags(
        7,
        {"classification": {"flags": []}},
        restored,
    ) == {"source": [], "organ": []}


def test_actual_shaped_review_selection_survives_sort_keys_json_round_trip():
    """Equal 3.0x technical ties retain the sealed production membership."""
    readout = {
        prototype: {
            "group": "not_kras_associated",
            "kras_association": {"significant": False},
            "model_attends": False,
            "shortcut_flags": [],
        }
        for prototype in range(corrected.K)
    }
    # These are the equal-dominance contenders in the corrected k32 analysis.
    # Before the numeric tie-break, sort_keys JSON order selected
    # 1,12,15,16,18,23 instead of the generated/sealed 1,6,8,9,12,15 packet.
    for prototype in (1, 6, 8, 9, 12, 15, 16, 18, 23):
        readout[prototype] = {
            **readout[prototype],
            "group": "shortcut_technical",
            "shortcut_flags": [
                "acquisition_dominated(mpp_bin@RIH=mpp~0.19 1.00, 3.0x)"
            ],
        }
    readout[0] = {
        **readout[0],
        "group": "shortcut_technical",
        "shortcut_flags": [
            "acquisition_dominated(mpp_bin@RIH=mpp~0.50 0.94, 2.8x)"
        ],
    }
    for prototype, group in (
        (17, "dependency_robust_transport_underpowered"),
        (28, "dependency_robust_transport_inconclusive"),
    ):
        readout[prototype] = {
            **readout[prototype],
            "group": group,
            "kras_association": {"significant": True},
        }
    readout[5] = {**readout[5], "model_attends": True}
    for prototype in (20, 26):
        readout[prototype] = {
            **readout[prototype],
            "group": "shortcut_technical",
            "model_attends": True,
            "shortcut_flags": ["cohort_dominated(SurGen 1.00)"],
        }

    expected = {
        "claimed": [17, 28],
        "suppressed": [20, 26],
        "attention_followup": [5],
        "technical": [1, 6, 8, 9, 12, 15],
    }
    before = corrected._corrected_review_selection(readout)
    strict_json = json.dumps(readout, sort_keys=True, allow_nan=False)
    reloaded = {
        int(prototype): row
        for prototype, row in json.loads(strict_json).items()
    }
    after = corrected._corrected_review_selection(reloaded)

    assert corrected.legacy.review_prototypes(reloaded) == expected
    assert before == after == expected


def _profile_bundle(tmp_path: Path) -> corrected.InputBundle:
    manifests: dict[str, pd.DataFrame] = {}
    for index, arm in enumerate(corrected.ALL_ARMS):
        manifests[arm] = pd.DataFrame([{
            "slide_id": f"s{index}",
            "patient_id": f"p{index}",
            "target_label": index % 2,
            "specimen_role": "primary" if arm == "e0" or "primary" in arm else "metastatic",
            "cohort": "TCGA" if arm == "e0" else ("RIH" if arm.startswith("rih") else "SurGen"),
            "subcohort": "TCGA-COAD" if arm == "e0" else (
                "RIH-Colon" if arm.startswith("rih") else "SR1482"
            ),
        }])
    return corrected.InputBundle(
        input_root=tmp_path / "legacy",
        aim2_root=tmp_path / "aim2",
        manifests=manifests,
        feature_files={},
        checkpoints={},
        receipt={},
    )


def _write_profiles(tmp_path: Path, *, missing_prototype: bool = False) -> Path:
    output = tmp_path / "output"
    (output / "profiles").mkdir(parents=True)
    slide_rows: list[dict] = []
    patient_rows: list[dict] = []
    candidate_rows: list[dict] = []
    prototypes = range(31) if missing_prototype else range(32)
    for index, arm in enumerate(corrected.ALL_ARMS):
        role = "primary" if arm == "e0" or "primary" in arm else "metastatic"
        cohort = "TCGA" if arm == "e0" else ("RIH" if arm.startswith("rih") else "SurGen")
        subcohort = "TCGA-COAD" if arm == "e0" else (
            "RIH-Colon" if arm.startswith("rih") else "SR1482"
        )
        for prototype in prototypes:
            base = {
                "arm": arm,
                "role": role,
                "cohort": cohort,
                "subcohort": subcohort,
                "prototype": prototype,
                "abundance": 1 / 32,
                "attn_mass_mean": 1 / 32,
                "attn_mass_seed42": 1 / 32,
                "attn_mass_seed43": 1 / 32,
                "attn_mass_seed44": 1 / 32,
                "n_attention_seeds": 3,
                "label": index % 2,
            }
            slide_rows.append({
                **base,
                "slide_id": f"s{index}",
                "patient_id": f"p{index}",
                "n_tiles_prototype": 1,
                "n_tiles_slide": 32,
            })
            patient_rows.append({
                **base,
                "patient_id": f"p{index}",
                "n_tiles_prototype": 1,
                "n_tiles_slide": 32,
                "n_slides": 1,
            })
            for kind in ("medoid", "top_attention"):
                candidate_rows.append({
                    "arm": arm,
                    "slide_id": f"s{index}",
                    "patient_id": f"p{index}",
                    "role": role,
                    "cohort": cohort,
                    "subcohort": subcohort,
                    "label": index % 2,
                    "prototype": prototype,
                    "kind": kind,
                    "tile_index": prototype,
                    "x": prototype,
                    "y": prototype,
                    "distance_to_centroid": float(prototype),
                    "attention": 1 / 32,
                })
    pd.DataFrame(slide_rows).to_parquet(
        output / "profiles/slide_profiles_k32.parquet", index=False
    )
    pd.DataFrame(patient_rows).to_parquet(
        output / "profiles/patient_profiles_k32.parquet", index=False
    )
    pd.DataFrame(candidate_rows).to_parquet(
        output / "profiles/tile_candidates_k32.parquet", index=False
    )
    return output


def test_profile_validation_fails_closed_and_accepts_only_exact_blocks(tmp_path):
    bundle = _profile_bundle(tmp_path)
    output = _write_profiles(tmp_path)
    result = corrected.validate_profiles(output, bundle, deep_candidates=False)
    assert result["status"] == "PASS"
    assert result["slide_rows"] == len(corrected.ALL_ARMS) * 32
    assert result["patient_rows"] == len(corrected.ALL_ARMS) * 32


def test_profile_validation_rejects_one_missing_prototype(tmp_path):
    bundle = _profile_bundle(tmp_path)
    output = _write_profiles(tmp_path, missing_prototype=True)
    with pytest.raises(RuntimeError, match="prototype inventory|32-prototype"):
        corrected.validate_profiles(output, bundle, deep_candidates=False)


def test_review_form_contains_no_prototype_or_selection_information():
    form = corrected._review_form(["M01", "M02"], [12, 12])
    assert "prototype" not in form.columns
    assert "selection_stratum" not in form.columns
    assert form["montage_id"].tolist() == ["M01", "M02"]
    assert (form.drop(columns=["montage_id", "n_tiles"]) == "pending").all().all()


def test_packet_instructions_require_an_external_completed_copy():
    instructions = corrected._packet_instructions("base", 3, 12, 256)
    assert "immutable blank master" in instructions
    assert "copy `review_form.csv`" in instructions
    assert "do not edit, rename, or add files inside the sealed packet" in instructions


def _complete_form(form: pd.DataFrame) -> pd.DataFrame:
    completed = form.copy()
    completed["review_status"] = "complete"
    completed["interpretable"] = "yes"
    completed["dominant_pattern"] = (
        "well-formed glands with luminal debris and a desmoplastic interface"
    )
    completed["heterogeneity"] = "mixed"
    completed["architecture"] = "glandular"
    for field in corrected.PRESENCE_REVIEW_FIELDS:
        completed[field] = "absent"
    completed["differentiation"] = "moderate"
    completed["confidence"] = "high"
    completed["reviewer_id"] = "pathologist-01"
    completed["review_date"] = "2026-08-20"
    completed["blinding_attestation"] = "confirmed_no_key_access"
    completed["free_text"] = "none"
    return completed


def test_completed_review_requires_explicit_complete_state_and_no_pending_cells(tmp_path):
    generated = tmp_path / "generated.csv"
    submitted = tmp_path / "submitted.csv"
    form = corrected._review_form(["M01"], [12])
    form.to_csv(generated, index=False)
    form.to_csv(submitted, index=False)
    with pytest.raises(RuntimeError, match="blank or pending"):
        corrected._completed_review_form(submitted, generated, label="base")


def test_completed_review_is_structured_and_preserves_full_text(tmp_path):
    generated = tmp_path / "generated.csv"
    submitted = tmp_path / "submitted.csv"
    form = corrected._review_form(["M01"], [12])
    form.to_csv(generated, index=False)
    completed = _complete_form(form)
    completed.to_csv(submitted, index=False)
    loaded = corrected._completed_review_form(submitted, generated, label="base")
    assert loaded.loc[0, "review_status"] == "complete"
    assert loaded.loc[0, "dominant_pattern"] == completed.loc[0, "dominant_pattern"]
    assert loaded.loc[0, "blinding_attestation"] == "confirmed_no_key_access"


def test_completed_review_rejects_blank_as_not_assessable_substitute(tmp_path):
    generated = tmp_path / "generated.csv"
    submitted = tmp_path / "submitted.csv"
    form = corrected._review_form(["M01"], [12])
    form.to_csv(generated, index=False)
    completed = _complete_form(form)
    completed.loc[0, "artifact"] = ""
    completed.to_csv(submitted, index=False)
    with pytest.raises(RuntimeError, match="blank or pending"):
        corrected._completed_review_form(submitted, generated, label="base")


def test_non_interpretable_review_requires_explicit_not_assessable_semantics(tmp_path):
    generated = tmp_path / "generated.csv"
    submitted = tmp_path / "submitted.csv"
    form = corrected._review_form(["M01"], [12])
    form.to_csv(generated, index=False)
    completed = _complete_form(form)
    completed.loc[0, "interpretable"] = "no"
    completed.loc[0, "dominant_pattern"] = "not_assessable"
    completed.to_csv(submitted, index=False)
    with pytest.raises(RuntimeError, match="morphology fields"):
        corrected._completed_review_form(submitted, generated, label="base")
    fields = [
        "dominant_pattern", "heterogeneity", "architecture",
        *corrected.PRESENCE_REVIEW_FIELDS, "differentiation", "confidence",
    ]
    completed.loc[0, fields] = "not_assessable"
    completed.to_csv(submitted, index=False)
    assert corrected._completed_review_form(submitted, generated, label="base").loc[
        0, "review_status"
    ] == "complete"


def test_tile_scale_is_bound_to_256_samples_at_target_mpp_half(tmp_path, monkeypatch):
    feature_dir = tmp_path / "store" / "features"
    patch_dir = feature_dir.parent / "patches"
    patch_dir.mkdir(parents=True)
    with h5py.File(patch_dir / "s1_patches.h5", "w") as handle:
        coords = handle.create_dataset("coords", data=np.zeros((1, 2), dtype=np.int64))
        coords.attrs["patch_size"] = 256
        coords.attrs["coords_schema_version"] = 2
        coords.attrs["sampling_mode"] = "exact_mpp"
        coords.attrs["level0_mpp"] = 0.2499
        coords.attrs["target_mpp"] = 0.5
        coords.attrs["effective_target_mpp"] = 0.4998
        coords.attrs["coordinate_units"] = "level0_pixels"
        coords.attrs["patch_size_level0"] = 512
        coords.attrs["read_level"] = 0
        coords.attrs["read_level_downsample"] = 1.0
        coords.attrs["read_patch_size"] = 512
        coords.attrs["actual_read_level0_pixels"] = 512
        coords.attrs["mask_simplification"] = "none"
    monkeypatch.setattr(corrected.paths, "PINNED_FEATURE_DIR", feature_dir)
    scale = corrected._tile_scale("s1")
    assert scale["field_width_um"] == pytest.approx(127.9488)
    assert scale["coordinate_units"] == "level0_pixels"


def test_patch_coordinate_must_match_exact_patcher_row(tmp_path, monkeypatch):
    feature_dir = tmp_path / "store" / "features"
    patch_dir = feature_dir.parent / "patches"
    patch_dir.mkdir(parents=True)
    with h5py.File(patch_dir / "s1_patches.h5", "w") as handle:
        handle.create_dataset("coords", data=np.asarray([[11, 22], [33, 44]], dtype=np.int64))
    monkeypatch.setattr(corrected.paths, "PINNED_FEATURE_DIR", feature_dir)
    corrected._validate_patch_coordinate("s1", 1, 33, 44)
    with pytest.raises(RuntimeError, match="differs from patch row"):
        corrected._validate_patch_coordinate("s1", 1, 33, 45)


def _source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    (output / "receipts").mkdir(parents=True)
    fixed = ["aim4_morphologic_atlas.py", "aim4_morphologic_atlas_base.py", "aim2_loco_transport.py", "pyproject.toml", "uv.lock"]
    rows = []
    for name in fixed:
        live = repo / name
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text(f"frozen {name}\n")
        snapshot = output / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(live.read_bytes())
        rows.append({
            "relative_path": name,
            "source": corrected._identity(live),
            "imported": corrected._identity(snapshot),
        })
    corrected._write_json_once(
        output / "receipts/source_snapshot.json",
        {"schema_version": 1, "component": "aim4_corrected_cap8192", "files": rows},
    )
    monkeypatch.setattr(corrected, "REPO", repo)
    return repo, output


def test_archival_source_verification_survives_live_drift_but_not_snapshot_tamper(
    tmp_path, monkeypatch
):
    repo, output = _source_fixture(tmp_path, monkeypatch)
    corrected._validate_source_snapshot(output, require_live_source=True)
    (repo / "aim4_morphologic_atlas.py").write_text("future worktree revision\n")
    corrected._validate_source_snapshot(output, require_live_source=False)
    with pytest.raises(RuntimeError, match="source changed"):
        corrected._validate_source_snapshot(output, require_live_source=True)
    (output / "source_snapshot/aim4_morphologic_atlas.py").write_text("tampered snapshot\n")
    with pytest.raises(RuntimeError, match="frozen source snapshot changed"):
        corrected._validate_source_snapshot(output, require_live_source=False)


def test_archival_inputs_use_imported_content_identity_not_retired_source_path(
    tmp_path, monkeypatch
):
    output = tmp_path / "corrected"
    aim2 = tmp_path / "aim2"
    original = tmp_path / "retired-legacy"
    (output / "receipts").mkdir(parents=True)
    aim2.mkdir()
    original_identity = {
        "path": str(original / "artifact"), "size_bytes": 7, "sha256": "a" * 64,
    }
    imported_identity = {
        "path": str(output / "artifact"), "size_bytes": 7, "sha256": "a" * 64,
    }
    common = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "aim2_root": str(aim2),
        "protocol": {},
        "manifest_files": {},
        "manifest_counts": {},
        "aim2_lineage_start": {},
        "aim2_refits": {},
        "feature_store": {},
        "validation": {},
    }
    recorded = {
        **common,
        "input_e4_root": str(original),
        "vocabulary": {
            "vocab_npz": original_identity,
            "vocab_json": original_identity,
            "sample_plan": original_identity,
            "n_plan_slides": 1,
            "n_sample_tiles": 1,
        },
        "legacy_e0_attention": {
            str(seed): {
                "h5": original_identity,
                "summary": original_identity,
                "n_slides": 1,
                "deep_validation": True,
            }
            for seed in corrected.SEEDS
        },
    }
    corrected._write_json_once(output / "receipts/inputs.json", recorded)
    candidate_receipt = {
        **common,
        "input_e4_root": str(output),
        "vocabulary": {
            **recorded["vocabulary"],
            "vocab_npz": imported_identity,
            "vocab_json": imported_identity,
            "sample_plan": imported_identity,
        },
        "legacy_e0_attention": {
            str(seed): {
                **recorded["legacy_e0_attention"][str(seed)],
                "h5": imported_identity,
                "summary": imported_identity,
            }
            for seed in corrected.SEEDS
        },
    }
    candidate = corrected.InputBundle(
        input_root=output,
        aim2_root=aim2,
        manifests={},
        feature_files={},
        checkpoints={},
        receipt=candidate_receipt,
    )
    monkeypatch.setattr(corrected, "validate_inputs", lambda *_args, **_kwargs: candidate)
    archived = corrected.validate_archival_inputs(original, aim2, output)
    assert archived.input_root == original
    assert archived.receipt == recorded


def _packet_candidates(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "arm": "e0", "role": "primary", "slide_id": f"s{index}",
            "patient_id": f"p{index}", "cohort": "TCGA", "subcohort": "TCGA-COAD",
            "label": index % 2, "prototype": 7, "kind": "medoid", "tile_index": index,
            "x": index, "y": index, "distance_to_centroid": float(index),
            "attention": 1 / n,
        }
        for index in range(n)
    ])


def test_corrected_picker_is_deterministic_and_patient_slide_unique():
    rows = []
    for patient in range(18):
        # Multiple slides and both candidate kinds reproduce the real k32 pool
        # condition that made the legacy picker return only 10-11 patients.
        for block in range(2):
            for kind in ("medoid", "top_attention"):
                rows.append({
                    "patient_id": f"p{patient}",
                    "slide_id": f"s{patient}_{block}",
                    "cohort": "TCGA" if patient % 2 else "CPTAC",
                    "role": "primary",
                    "kind": kind,
                    "x": patient * 100 + block,
                    "y": patient * 100 + block,
                })
    pool = pd.DataFrame(rows)
    first = corrected._diverse_patient_picks(pool, 12, np.random.default_rng(7))
    second = corrected._diverse_patient_picks(pool, 12, np.random.default_rng(7))
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 12
    assert first["patient_id"].nunique() == 12
    assert first["slide_id"].nunique() == 12
    assert not first.duplicated(["slide_id", "x", "y"]).any()


def test_slide_index_cache_is_bound_to_exact_root(tmp_path):
    output = tmp_path / "output"
    root_one = tmp_path / "slides-one"
    root_two = tmp_path / "slides-two"
    for root in (root_one, root_two):
        directory = root / "SURGEN"
        directory.mkdir(parents=True)
        (directory / "s1.svs").write_bytes(b"wsi")
    index = corrected._load_slide_index(output, root_one)
    assert Path(index["s1"]).is_relative_to(root_one)
    with pytest.raises(RuntimeError, match="not bound"):
        corrected._load_slide_index(output, root_two)


def test_slide_index_cache_rejects_wsi_content_drift(tmp_path):
    output = tmp_path / "output"
    root = tmp_path / "slides"
    directory = root / "rih"
    directory.mkdir(parents=True)
    slide = directory / "s1.svs"
    slide.write_bytes(b"first")
    corrected._load_slide_index(output, root)
    slide.write_bytes(b"changed-size")
    with pytest.raises(RuntimeError, match="size/mtime inventory changed"):
        corrected._load_slide_index(output, root)


def test_slide_index_cache_rejects_new_higher_precedence_copy(tmp_path):
    output = tmp_path / "output"
    root = tmp_path / "slides"
    original = root / "rih"
    original.mkdir(parents=True)
    (original / "s1.svs").write_bytes(b"original")
    first = corrected._load_slide_index(output, root)
    assert Path(first["s1"]).parent.name == "rih"
    corrected_copy = root / "SURGEN"
    corrected_copy.mkdir()
    (corrected_copy / "s1.svs").write_bytes(b"corrected")
    with pytest.raises(RuntimeError, match="precedence/inventory"):
        corrected._load_slide_index(output, root)


def _patch_packet_rendering(monkeypatch: pytest.MonkeyPatch, *, fail: bool) -> None:
    monkeypatch.setattr(
        corrected,
        "_load_slide_index",
        lambda _output, _root: {
            f"s{index}": f"/slides/s{index}.svs" for index in range(12)
        },
    )
    monkeypatch.setattr(
        corrected.legacy,
        "_diverse_picks",
        lambda pool, n, _rng: pool.head(n).copy(),
    )
    if fail:
        monkeypatch.setattr(
            corrected.legacy,
            "_read_tile",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("render failed")),
        )
    else:
        monkeypatch.setattr(corrected.legacy, "_read_tile", lambda *_args, **_kwargs: object())

        class _Grid:
            @staticmethod
            def save(path, quality):
                assert quality == 92
                Path(path).write_bytes(b"jpeg")

        monkeypatch.setattr(corrected.legacy, "_grid", lambda *_args, **_kwargs: _Grid())
    monkeypatch.setattr(corrected, "_tile_scale", lambda _slide: {
        "patch_size_samples": 256,
        "target_mpp": 0.5,
        "effective_target_mpp": 0.5,
        "field_width_um": 128.0,
        "patch_size_level0": 512,
        "coordinate_units": "level0_pixels",
    })
    monkeypatch.setattr(
        corrected,
        "_validate_patch_coordinate",
        lambda _slide, _index, _x, _y: None,
    )
    monkeypatch.setitem(sys.modules, "openslide", SimpleNamespace())


def test_packet_and_external_key_promote_as_one_atomic_bundle(tmp_path, monkeypatch):
    _patch_packet_rendering(monkeypatch, fail=False)
    slide_root = tmp_path / "slides"
    slide_root.mkdir()
    receipt = corrected._render_packet(
        output=tmp_path,
        candidates=_packet_candidates(),
        prototypes=[7],
        packet_slug="base",
        id_prefix="M",
        selection_strata={7: "claimed"},
        slide_root=slide_root,
        tiles_per_montage=12,
        tile_px=256,
        seed=1,
    )
    packet = Path(receipt["packet"])
    key = Path(receipt["key"]["path"])
    assert packet.is_dir()
    assert key.is_file()
    assert key.parent == packet.parent
    assert key.parent.name == "base"
    assert not key.is_relative_to(packet)
    assert (packet / "scale_info.csv").is_file()


def _write_complete_review_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_packet_rendering(monkeypatch, fail=False)
    slide_root = tmp_path / "slides"
    slide_root.mkdir()
    base = corrected._render_packet(
        output=tmp_path,
        candidates=_packet_candidates(),
        prototypes=[7],
        packet_slug="base",
        id_prefix="M",
        selection_strata={7: "claimed"},
        slide_root=slide_root,
        tiles_per_montage=12,
        tile_px=256,
        seed=1,
    )
    addendum = corrected._render_packet(
        output=tmp_path,
        candidates=_packet_candidates(),
        prototypes=[],
        packet_slug="attention_addendum",
        id_prefix="A",
        selection_strata={7: "claimed"},
        slide_root=slide_root,
        tiles_per_montage=12,
        tile_px=256,
        seed=2,
    )
    selection = {
        "base_packet": [7],
        "attention_addendum_packet": [],
        "claimed": [7],
        "suppressed": [],
        "technical": [],
    }
    plan = {
        "status": "PASS",
        "base_packet_prototypes": [7],
        "attention_addendum_prototypes": [],
        "separate_keys": True,
        "tiles_per_montage": 12,
    }
    receipt = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "PASS",
        "selection": plan,
        "base": base,
        "attention_addendum": addendum,
        "blinding": "prototype IDs occur only in external keys, never packet files",
    }
    campaign = tmp_path / "review_bundles/k32"
    corrected._write_json_once(
        campaign / corrected.REVIEW_TRANSACTION_RECEIPT, receipt
    )
    corrected._copy_once(
        campaign / corrected.REVIEW_TRANSACTION_RECEIPT,
        tmp_path / "receipts/review_packets.json",
    )
    monkeypatch.setattr(
        corrected, "_selection_from_report", lambda _output: ({}, selection)
    )


def test_review_validator_rejects_any_extra_file_inside_blinded_packet(
    tmp_path, monkeypatch
):
    _write_complete_review_campaign(tmp_path, monkeypatch)
    assert corrected.validate_review_packets(tmp_path)["status"] == "PASS"
    (tmp_path / "review_bundles/k32/base/packet/LEAK.txt").write_text("prototype 7")
    with pytest.raises(RuntimeError, match="exact blinded bundle inventory differs"):
        corrected.validate_review_packets(tmp_path)


def test_review_validator_binds_internal_and_external_transaction_receipts(
    tmp_path, monkeypatch
):
    _write_complete_review_campaign(tmp_path, monkeypatch)
    transaction = tmp_path / "review_bundles/k32" / corrected.REVIEW_TRANSACTION_RECEIPT
    payload = json.loads(transaction.read_text())
    payload["blinding"] = "drifted"
    transaction.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="transaction receipt differs"):
        corrected.validate_review_packets(tmp_path)


def test_render_failure_never_creates_a_canonical_partial_packet(tmp_path, monkeypatch):
    _patch_packet_rendering(monkeypatch, fail=True)
    slide_root = tmp_path / "slides"
    slide_root.mkdir()
    with pytest.raises(RuntimeError, match="could not render"):
        corrected._render_packet(
            output=tmp_path,
            candidates=_packet_candidates(),
            prototypes=[7],
            packet_slug="base",
            id_prefix="M",
            selection_strata={7: "claimed"},
            slide_root=slide_root,
            tiles_per_montage=12,
            tile_px=256,
            seed=1,
        )
    assert not (tmp_path / "review_bundles/k32/base").exists()


def test_two_packet_campaign_can_stage_base_without_publishing_it(tmp_path, monkeypatch):
    _patch_packet_rendering(monkeypatch, fail=False)
    slide_root = tmp_path / "slides"
    slide_root.mkdir()
    staged_parent = tmp_path / "_staging/review_campaigns/k32.test"
    corrected._render_packet(
        output=tmp_path,
        candidates=_packet_candidates(),
        prototypes=[7],
        packet_slug="base",
        id_prefix="M",
        selection_strata={7: "claimed"},
        slide_root=slide_root,
        tiles_per_montage=12,
        tile_px=256,
        seed=1,
        bundle_parent=staged_parent,
    )
    assert (staged_parent / "base/packet/review_form.csv").is_file()
    assert not (tmp_path / "review_bundles/k32").exists()


def test_second_packet_failure_leaves_no_canonical_campaign(tmp_path, monkeypatch):
    _patch_packet_rendering(monkeypatch, fail=False)
    slide_root = tmp_path / "slides"
    slide_root.mkdir()
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    base = _packet_candidates()
    addendum = base.assign(prototype=8)
    pd.concat([base, addendum], ignore_index=True).to_parquet(
        profiles / "tile_candidates_k32.parquet", index=False
    )
    monkeypatch.setattr(corrected, "_validate_prepared", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        corrected,
        "validate_profiles",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(corrected, "_selection_from_report", lambda _output: ({}, {
        "base_packet": [7],
        "attention_addendum_packet": [8],
        "claimed": [7],
        "suppressed": [],
        "technical": [],
    }))
    calls = 0

    def fail_after_base(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls > 12:
            raise RuntimeError("second packet render failed")
        return object()

    monkeypatch.setattr(corrected.legacy, "_read_tile", fail_after_base)
    with pytest.raises(RuntimeError, match="second packet render failed"):
        corrected.make_review_packets(
            _profile_bundle(tmp_path),
            tmp_path,
            slide_root=slide_root,
            tiles_per_montage=12,
            tile_px=256,
            seed=corrected.BOOTSTRAP_SEED,
            apply=True,
        )
    assert not (tmp_path / "review_bundles/k32").exists()
    assert not (tmp_path / "receipts/review_packets.json").exists()


def test_assignment_failure_leaves_no_canonical_profile_campaign(tmp_path, monkeypatch):
    bundle = _profile_bundle(tmp_path)
    bundle.checkpoints = {
        (target, seed): tmp_path / f"{target}-{seed}.ckpt"
        for target in ("RIH", "SurGen")
        for seed in corrected.SEEDS
    }
    monkeypatch.setattr(corrected, "_validate_prepared", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(corrected, "_validate_attention_file", lambda *_args, **_kwargs: {})

    def fail_in_stage(_args):
        corrected.legacy.PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        (corrected.legacy.PROFILE_DIR / "partial.parquet").write_bytes(b"partial")
        raise RuntimeError("assignment failed")

    monkeypatch.setattr(corrected.legacy, "cmd_assign", fail_in_stage)
    with pytest.raises(RuntimeError, match="assignment failed"):
        corrected.assign_profiles(bundle, tmp_path, apply=True)
    assert not (tmp_path / "profiles").exists()
    assert not (tmp_path / "receipts/profiles.json").exists()


def test_analysis_failure_leaves_no_canonical_analysis_campaign(tmp_path, monkeypatch):
    bundle = _profile_bundle(tmp_path)
    monkeypatch.setattr(corrected, "_validate_prepared", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        corrected,
        "validate_profiles",
        lambda *_args, **_kwargs: {"status": "PASS", "artifacts": {}},
    )

    def write_specificity(_args):
        corrected.legacy.paths.EVAL_ROOT.mkdir(parents=True, exist_ok=True)
        corrected._write_json_once(
            corrected.legacy.paths.EVAL_ROOT / "e3b_specificity_k32.json", {}
        )

    monkeypatch.setattr(corrected.legacy, "cmd_specificity", write_specificity)
    monkeypatch.setattr(
        corrected.legacy,
        "cmd_transport",
        lambda _args: (_ for _ in ()).throw(RuntimeError("transport failed")),
    )
    with pytest.raises(RuntimeError, match="transport failed"):
        corrected.analyze(
            bundle,
            tmp_path,
            n_bootstrap=corrected.DEFAULT_BOOTSTRAP,
            apply=True,
        )
    assert not (tmp_path / "analysis").exists()
    assert not (tmp_path / "receipts/analysis.json").exists()


def test_review_import_failure_leaves_no_canonical_completed_bundle(tmp_path, monkeypatch):
    bundle = _profile_bundle(tmp_path)
    base = tmp_path / "base.csv"
    addendum = tmp_path / "addendum.csv"
    base.write_text("completed\n")
    addendum.write_text("completed\n")
    monkeypatch.setattr(corrected, "_validate_prepared", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(corrected, "validate_numeric_verification", lambda *_args: {})
    monkeypatch.setattr(corrected, "validate_review_packets", lambda *_args: {})
    monkeypatch.setattr(corrected, "_assert_not_completed", lambda *_args: None)
    monkeypatch.setattr(
        corrected,
        "_review_bundle_paths",
        lambda _output, slug: (tmp_path / f"generated-{slug}.csv", tmp_path / "key.csv"),
    )
    for slug in ("base", "attention_addendum"):
        (tmp_path / f"generated-{slug}.csv").write_text("pending\n")
    monkeypatch.setattr(
        corrected,
        "_completed_review_form",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    original_copy = corrected._copy_once
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second import failed")
        return original_copy(source, destination)

    monkeypatch.setattr(corrected, "_copy_once", fail_second)
    with pytest.raises(RuntimeError, match="second import failed"):
        corrected.import_completed_reviews(
            bundle,
            tmp_path,
            base_form=base,
            attention_addendum_form=addendum,
            apply=True,
        )
    assert not (tmp_path / "review_completion/k32").exists()
    assert not (tmp_path / "receipts/review_import.json").exists()


def test_completed_review_validator_binds_transaction_receipt_copy(tmp_path):
    canonical = tmp_path / "review_completion/k32"
    external = tmp_path / "receipts/review_import.json"
    payload = {
        "schema_version": 1,
        "component": "aim4_corrected_pathology_review",
        "status": "PASS",
    }
    corrected._write_json_once(
        canonical / corrected.REVIEW_IMPORT_TRANSACTION_RECEIPT, payload
    )
    corrected._write_json_once(external, {**payload, "status": "DRIFTED"})
    with pytest.raises(RuntimeError, match="transaction receipt differs"):
        corrected.validate_imported_reviews(tmp_path)


def test_numeric_seal_and_full_replay_verification_are_immutable(tmp_path):
    output = tmp_path / "corrected"
    (output / "analysis").mkdir(parents=True)
    (output / "analysis/result.json").write_text('{"status":"PASS"}\n')
    files = corrected._manifest_files(output, phase="numeric")
    completion = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "numeric_completed_pathology_review_pending",
        "output_root": str(output),
        "artifacts": files,
        "aggregate_sha256": corrected.hashlib.sha256(
            corrected._canonical(files).encode()
        ).hexdigest(),
    }
    corrected._write_json_once(output / corrected.NUMERIC_COMPLETION_NAME, completion)
    result = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "PASS",
        "receipt_role": "immutable_numeric_verification_addendum",
        "output_root": str(output),
        "numeric_completion": corrected._identity(
            output / corrected.NUMERIC_COMPLETION_NAME
        ),
        "numeric_aggregate_sha256": completion["aggregate_sha256"],
        "checks": {"deterministic_statistical_replay": "PASS"},
        "replay": {"status": "PASS"},
    }
    identity = corrected.seal_numeric_verification(output, result)
    assert Path(identity["path"]).name == corrected.NUMERIC_VERIFICATION_NAME
    assert corrected.validate_numeric_verification(output)["status"] == "PASS"
    assert corrected.seal_numeric_verification(output, result) == identity
    changed = {**result, "numeric_aggregate_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="immutable verification receipt differs"):
        corrected.seal_numeric_verification(output, changed)
    (output / "analysis/result.json").write_text('{"status":"TAMPERED"}\n')
    with pytest.raises(RuntimeError, match="completed artifact identity changed"):
        corrected.validate_numeric_completion(output)


def test_numeric_and_final_manifests_reject_unexpected_unsealed_files(tmp_path):
    numeric = tmp_path / "numeric"
    numeric.mkdir()
    (numeric / "result.json").write_text("locked\n")
    numeric_files = corrected._manifest_files(numeric, phase="numeric")
    (numeric / "unexpected.txt").write_text("not an allowed pathology addendum\n")
    with pytest.raises(RuntimeError, match="unexpected artifact after numeric seal"):
        corrected._validate_manifest(numeric, numeric_files, phase="numeric")

    final = tmp_path / "final"
    final.mkdir()
    (final / "result.json").write_text("locked\n")
    final_files = corrected._manifest_files(final, phase="final")
    (final / "unexpected.txt").write_text("post-seal drift\n")
    with pytest.raises(RuntimeError, match="unexpected artifact after final seal"):
        corrected._validate_manifest(final, final_files, phase="final")


def test_skipped_replay_can_never_be_sealed(tmp_path):
    result = {
        "status": "PASS",
        "receipt_role": "immutable_completion_verification_addendum",
        "checks": {"deterministic_statistical_replay": "SKIPPED_BY_FLAG"},
        "replay": {"status": "SKIPPED_BY_FLAG"},
    }
    with pytest.raises(RuntimeError, match="full deterministic verification"):
        corrected.seal_verification(tmp_path, result)
    assert not (tmp_path / corrected.VERIFICATION_NAME).exists()


def _effect_records(*, delta: bool = False) -> list[dict]:
    records = []
    for prototype in range(32):
        if delta:
            records.append({
                "prototype": prototype,
                "delta_p": 1.0,
                "delta_q": 1.0,
                "estimable": False,
                "changed": False,
            })
        else:
            records.append({
                "prototype": prototype,
                "p": 1.0,
                "q": 1.0,
                "estimable": False,
                "significant": False,
            })
    return records


def _analysis_artifacts() -> tuple[dict, dict]:
    protocol = {
        "cap": corrected.CAP,
        "k": corrected.K,
        "seeds": list(corrected.SEEDS),
        "n_bootstrap": corrected.DEFAULT_BOOTSTRAP,
        "bootstrap_seed": corrected.BOOTSTRAP_SEED,
        "inference_unit": "patient",
        "multiplicity": "BH independently within each quantity x population/arm/contrast",
        "bh_family": "fixed prototypes 0..31, including structural p=1 rows",
    }
    prototypes = {}
    for prototype in range(32):
        prototypes[str(prototype)] = {
            quantity: {
                panel: _effect_records()[prototype]
                for panel in ("A", "D", "context_in_wt")
            }
            for quantity in corrected.QUANTITIES
        }
        prototypes[str(prototype)]["by_cohort"] = {
            quantity: {
                cohort: _effect_records()[prototype]
                for cohort in ("CPTAC", "RIH", "SurGen", "TCGA")
            }
            for quantity in corrected.QUANTITIES
        }
    spec = {
        "component": "aim4_corrected_cap8192",
        "protocol": protocol,
        "k": 32,
        "prototypes": prototypes,
    }
    arms = {
        arm: {"effects": {quantity: _effect_records() for quantity in corrected.QUANTITIES}}
        for arm in corrected.TARGET_ARMS
    }
    merged = []
    for prototype in range(32):
        effect = _effect_records()[prototype]
        merged.append({
            "prototype": prototype,
            "p_primary": effect["p"],
            "q_primary": effect["q"],
            "estimable_primary": effect["estimable"],
            "significant_primary": effect["significant"],
            "p_metastatic": effect["p"],
            "q_metastatic": effect["q"],
            "estimable_metastatic": effect["estimable"],
            "significant_metastatic": effect["significant"],
            **_effect_records(delta=True)[prototype],
        })
    transport = {
        "component": "aim4_corrected_cap8192",
        "protocol": protocol,
        "k": 32,
        "arms": arms,
        "conservation": {
            cohort: {"by_quantity": {quantity: merged for quantity in corrected.QUANTITIES}}
            for cohort in ("RIH", "SR1482")
        },
    }
    return spec, transport


def test_analysis_validator_requires_every_fixed_family_row():
    spec, transport = _analysis_artifacts()
    assert corrected.validate_analysis(spec, transport)["status"] == "PASS"
    transport["arms"]["rih_primary"]["effects"]["abundance"].pop()
    with pytest.raises(RuntimeError, match="exactly 32"):
        corrected.validate_analysis(spec, transport)
