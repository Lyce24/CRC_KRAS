"""Synthetic checks for the preregistered source grid sensitivity."""

import numpy as np
import pandas as pd
import pytest

from tools import final_v14_grid_offset as grid


def geometry(footprint=512, width=1600, height=1600):
    return {"level0_width": width, "level0_height": height,
            "patch_size_level0": footprint, "overlap_level0": 0,
            "patch_size": 256, "target_mpp": .5, "level0_mpp": 128 / footprint,
            "min_tissue_proportion": .5, "mask_simplification": "none"}


def test_half_stride_shift_preserves_nonoverlap_and_omits_boundary_tiles():
    coordinates = grid.shifted_grid_coordinates(geometry())
    np.testing.assert_array_equal(coordinates, [[256, 256], [256, 768], [768, 256], [768, 768]])
    assert np.all(coordinates + 512 <= 1600)
    assert not np.any(coordinates % 512 == 0)


def test_odd_stride_floors_origin_and_keeps_full_tiles():
    coords = grid.shifted_grid_coordinates(geometry(511, 2000, 1300))
    np.testing.assert_array_equal(coords, [[255, 255], [255, 766], [766, 255], [766, 766], [1277, 255], [1277, 766]])


def test_geometry_rejects_overlap_changed_mpp_and_noninteger_pixels():
    for key, value in [("overlap_level0", 256), ("target_mpp", .25), ("patch_size_level0", 511.5)]:
        attrs = geometry() | {key: value}
        with pytest.raises(ValueError):
            grid.shifted_grid_coordinates(attrs)
    assert grid.shifted_grid_coordinates(geometry(width=400)).shape == (0, 2)


def test_tissue_filter_receives_canonical_rule_after_boundary_clipping():
    calls = []
    mask = object()

    def filter_coordinates(candidates, **kwargs):
        calls.append((candidates.copy(), kwargs))
        return candidates[:1]

    result = grid.shifted_tissue_coordinates(geometry(), mask, filter_fn=filter_coordinates)
    np.testing.assert_array_equal(result, [[256, 256]])
    assert calls[0][1] == {"mask": mask, "footprint": 512, "threshold": .5}
    assert np.all(calls[0][0] + 512 <= 1600)


def eligible(count=30):
    return pd.DataFrame([{"patient_id": f"{group}:{i:03d}", "subcohort": group, "eligible": True}
                         for group in grid.SUBCOHORTS for i in range(count)])


def test_roster_exact_25_per_group_deterministic_and_input_order_invariant():
    frame = eligible()
    selected = grid.select_source_patients(frame, canonical_root_verified=True)
    assert len(selected) == selected.patient_id.nunique() == 100
    assert selected.groupby("subcohort").size().eq(25).all()
    pd.testing.assert_frame_equal(selected, grid.select_source_patients(frame.iloc[::-1], canonical_root_verified=True))


def test_missing_archive_is_operational_not_insufficient_eligibility(tmp_path):
    with pytest.raises(grid.OperationalBlock):
        grid.require_canonical_root(tmp_path / "missing_archive")
    with pytest.raises(grid.OperationalBlock):
        grid.select_source_patients(eligible(20), canonical_root_verified=False)
    with pytest.raises(grid.InsufficientEligiblePatients):
        grid.select_source_patients(eligible(20), canonical_root_verified=True)


def test_excluded_patients_never_enter_draw_and_groups_are_not_substituted():
    frame = eligible(26)
    frame.loc[frame.patient_id.str.endswith("025"), "eligible"] = False
    selected = grid.select_source_patients(frame, canonical_root_verified=True)
    assert not selected.patient_id.str.endswith("025").any()
    frame.loc[0, "eligible"] = False
    with pytest.raises(grid.InsufficientEligiblePatients):
        grid.select_source_patients(frame, canonical_root_verified=True)


def test_quantization_matches_canonical_float16_store():
    features = np.full((2, 1024), 0.12345678, dtype=np.float32)
    observed = grid.canonical_quantize(features)
    np.testing.assert_array_equal(observed, features.astype(np.float16).astype(np.float32))
    assert np.any(observed != features)
    with pytest.raises(ValueError, match="overflow"):
        grid.canonical_quantize(np.full((1, 1024), 1e10))


def profile_rows():
    rows = []
    for slide, patient, a in [("s1", "p1", 1.), ("s2", "p1", 0.), ("s3", "p2", .3)]:
        row = {"slide_id": slide, "patient_id": patient, "subcohort": "SR386", "n_tiles": 1000 if slide == "s1" else 1}
        row.update({column: 0. for column in grid.PROTOTYPES})
        row["prototype_00"], row["prototype_01"] = a, 1-a
        rows.append(row)
    return pd.DataFrame(rows)


def test_patient_abundance_weights_slides_equally_and_pairs_exact_roster():
    patients = grid.equal_slide_profiles(profile_rows())
    assert patients.loc[0, "prototype_00"] == .5
    assert patients.loc[0, "n_slides"] == 2
    result = grid.paired_profile_summary(patients, patients.iloc[::-1])
    assert result["cosine_mean"] == pytest.approx(1.)
    assert result["prototype_icc"][0]["icc"] == pytest.approx(1.)
    assert result["prototype_icc"][2]["icc"] is None
    with pytest.raises(ValueError, match="same patient roster"):
        grid.paired_profile_summary(patients, patients.iloc[:1])


def test_icc_is_absolute_agreement_and_uses_between_patient_variance_guard():
    result = grid.icc_2_1(np.array([1., 2., 3.]), np.array([2., 3., 4.]))
    assert result["icc"] == pytest.approx(2/3)
    assert grid.icc_2_1(np.array([1., 2., 3.]), np.array([1., 2., 3.]))["icc"] == pytest.approx(1.)
    missing = grid.icc_2_1(np.full(1239, .1), np.linspace(0, 1, 1239))
    assert missing["status"] == "ICC_UNDEFINED_ZERO_BETWEEN_PATIENT_VARIANCE"
    assert missing["n_nonzero_original"] == 1239
    assert missing["icc"] is None


def test_preflight_does_not_select_or_report_scientific_failure(tmp_path):
    result = grid.preflight(tmp_path / "missing", tmp_path / "pre", tmp_path / "uni.bin")
    assert result["status"] == "OPERATIONALLY_BLOCKED"
    assert result["scientific_status"] == "PENDING_NOT_TESTED"
    assert result["patients_sampled"] == result["slides_extracted"] == 0
