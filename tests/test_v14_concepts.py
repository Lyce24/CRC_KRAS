from __future__ import annotations

import base64
import hashlib
import hmac

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from oceanpath.aim1.v14_concepts import (
    HMACCodeExhaustedError,
    V14Vocabulary,
    balanced_integer_quotas,
    blinded_occurrences,
    choose_hmac_code,
    closest_per_patient_candidates,
    deterministic_sample_indices,
    draw_collision_free_hmac_salt,
    equal_slide_patient_profiles,
    fit_v14_vocabulary,
    hierarchical_sample_plan,
    hmac_blinding_table,
    lloyd_kmeans_parameters,
    map_vocabulary_to_reference,
    maximum_cosine_hungarian_mapping,
    pca64_parameters,
    salt_sha256,
    sample_id_table,
    select_duplicate_montages,
    select_montage_tiles,
    slide_abundance,
    zero_based_prototype_index,
)


def test_balanced_integer_quotas_hit_budget_and_redistribute_in_sorted_rounds() -> None:
    quotas = balanced_integer_quotas({"c": 10, "a": 1, "b": 10}, 10)

    assert quotas == {"a": 1, "b": 5, "c": 4}
    assert sum(quotas.values()) == 10
    assert all(quotas[key] <= capacity for key, capacity in {"a": 1, "b": 10, "c": 10}.items())
    assert balanced_integer_quotas({"a": 2, "b": 1}, 99) == {"a": 2, "b": 1}


def test_hierarchical_sample_plan_is_exact_capacity_safe_and_row_order_invariant() -> None:
    slides = pd.DataFrame(
        [
            {"subcohort": "B", "patient_id": "p3", "slide_id": "s5", "n_tiles": 20},
            {"subcohort": "A", "patient_id": "p1", "slide_id": "s1", "n_tiles": 1},
            {"subcohort": "B", "patient_id": "p2", "slide_id": "s3", "n_tiles": 1},
            {"subcohort": "A", "patient_id": "p1", "slide_id": "s2", "n_tiles": 1},
            {"subcohort": "B", "patient_id": "p3", "slide_id": "s4", "n_tiles": 20},
        ]
    )

    first = hierarchical_sample_plan(slides, cap=10)
    second = hierarchical_sample_plan(slides.sample(frac=1, random_state=7), cap=10)

    assert_frame_equal(first, second)
    assert first["n_sample"].sum() == 10
    assert (first["n_sample"] <= first["n_tiles"]).all()
    assert first.groupby("subcohort")["n_sample"].sum().to_dict() == {"A": 2, "B": 8}
    # B's one-tile p2 returns the remainder to p3; p3 then splits seven 4/3.
    assert first.set_index("slide_id")["n_sample"].to_dict() == {
        "s1": 1,
        "s2": 1,
        "s3": 1,
        "s4": 4,
        "s5": 3,
    }


def test_hierarchical_sample_plan_rejects_cross_subcohort_patient() -> None:
    rows = pd.DataFrame(
        {
            "subcohort": ["A", "B"],
            "patient_id": ["p", "p"],
            "slide_id": ["s1", "s2"],
            "n_tiles": [1, 1],
        }
    )
    with pytest.raises(ValueError, match="more than one subcohort"):
        hierarchical_sample_plan(rows)


def test_within_slide_sampling_depends_on_sorted_ids_not_input_order() -> None:
    tile_ids = ["tile-10", "tile-02", "tile-01", "tile-20", "tile-03"]
    shuffled = [tile_ids[index] for index in [3, 0, 4, 1, 2]]

    first_indices = deterministic_sample_indices(tile_ids, 3, seed=17)
    second_indices = deterministic_sample_indices(shuffled, 3, seed=17)
    first_ids = [tile_ids[index] for index in first_indices]
    second_ids = [shuffled[index] for index in second_indices]

    assert first_ids == second_ids
    assert first_ids == sorted(first_ids)
    assert len(set(first_indices.tolist())) == 3


def test_sample_id_table_uses_one_canonical_stream_and_is_replayable() -> None:
    plan = pd.DataFrame(
        [
            {
                "subcohort": "B",
                "patient_id": "p2",
                "slide_id": "s2",
                "n_tiles": 4,
                "n_sample": 2,
            },
            {
                "subcohort": "A",
                "patient_id": "p1",
                "slide_id": "s1",
                "n_tiles": 4,
                "n_sample": 3,
            },
        ]
    )
    ids = {"s1": ["d", "a", "c", "b"], "s2": ["4", "2", "1", "3"]}

    first = sample_id_table(plan, ids, seed=23)
    second = sample_id_table(plan.iloc[::-1], ids, seed=23)

    assert_frame_equal(first, second)
    assert len(first) == 5
    assert first["sample_index"].tolist() == list(range(5))
    assert first.groupby("slide_id").size().to_dict() == {"s1": 3, "s2": 2}
    assert not first.duplicated(["slide_id", "tile_id"]).any()
    assert first.equals(first.sort_values(["subcohort", "patient_id", "slide_id", "tile_id"]))


def test_fixed_pca_and_kmeans_parameter_contracts_are_explicit() -> None:
    assert pca64_parameters() == {
        "n_components": 64,
        "whiten": False,
        "svd_solver": "randomized",
        "random_state": 20260819,
    }
    assert lloyd_kmeans_parameters() == {
        "n_clusters": 32,
        "n_init": 10,
        "max_iter": 300,
        "tol": 1e-4,
        "algorithm": "lloyd",
        "random_state": 20260819,
    }


def test_vocabulary_projects_assigns_and_returns_euclidean_distances() -> None:
    vocabulary = V14Vocabulary(
        centroids=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        pca_mean=np.zeros(2, dtype=np.float32),
        pca_components=np.eye(2, dtype=np.float32),
    )
    features = np.asarray([[3.0, 0.0], [0.0, 8.0], [1.0, 1.0]], dtype=np.float32)

    projected = vocabulary.project(features)
    labels, distances = vocabulary.assign_with_distances(features, batch_size=1)

    np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1.0)
    np.testing.assert_array_equal(labels, [0, 1, 0])
    np.testing.assert_allclose(distances[:2], 0.0, atol=1e-6)
    np.testing.assert_allclose(distances[2], np.sqrt(2.0 - np.sqrt(2.0)), rtol=1e-6)
    np.testing.assert_array_equal(vocabulary.assign(features), labels)
    np.testing.assert_allclose(vocabulary.backprojected_centroids(), vocabulary.centroids)


def test_fitted_vocabulary_uses_the_same_pca_coordinates_as_assignment() -> None:
    rng = np.random.default_rng(19)
    sample = rng.normal(size=(96, 64)).astype(np.float32)

    vocabulary = fit_v14_vocabulary(sample, n_prototypes=3, seed=23)

    projected = vocabulary.project(sample)
    centers = np.asarray(vocabulary.centroids)
    squared = (
        np.sum(projected * projected, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1)[None, :]
        - 2 * projected @ centers.T
    )
    labels, distances = vocabulary.assign_with_distances(sample)
    np.testing.assert_array_equal(labels, np.argmin(squared, axis=1))
    np.testing.assert_array_equal(
        np.bincount(labels, minlength=3), vocabulary.config["cluster_sizes"]
    )
    np.testing.assert_allclose(
        np.sum(np.square(distances, dtype=np.float64)),
        vocabulary.config["inertia"],
        rtol=2e-5,
    )
    assert np.isfinite(projected).all()


def test_backprojection_and_hungarian_mapping_recover_permuted_coordinates() -> None:
    reference = V14Vocabulary(
        centroids=np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        pca_mean=np.zeros(2, dtype=np.float32),
        pca_components=np.eye(2, dtype=np.float32),
    )
    source = V14Vocabulary(
        centroids=np.asarray([[0.0, 5.0], [4.0, 0.0]], dtype=np.float32),
        pca_mean=np.zeros(2, dtype=np.float32),
        pca_components=np.eye(2, dtype=np.float32),
    )

    mapping = map_vocabulary_to_reference(source, reference)

    assert mapping["source_prototype_id"].tolist() == [0, 1]
    assert mapping["reference_prototype_id"].tolist() == [1, 0]
    np.testing.assert_allclose(mapping["cosine_similarity"], 1.0)
    assert mapping["name_mappable"].all()


def test_rectangular_hungarian_mapping_marks_unmatched_source_nullable() -> None:
    source = np.eye(3, dtype=np.float32)
    reference = np.eye(3, dtype=np.float32)[:2]

    mapping = maximum_cosine_hungarian_mapping(source, reference)

    assert mapping["reference_prototype_id"].isna().sum() == 1
    assert mapping.loc[mapping["reference_prototype_id"].isna(), "name_mappable"].eq(False).all()


def test_slide_abundance_and_patient_aggregation_are_equal_slide_not_tile_weighted() -> None:
    large_slide = slide_abundance(np.asarray([0] * 90 + [1] * 10), 2)
    small_slide = slide_abundance(np.asarray([1]), 2)
    other_patient = slide_abundance(np.asarray([0, 0]), 2)

    patients = equal_slide_patient_profiles(
        np.vstack([large_slide, small_slide, other_patient]), ["p1", "p1", "p2"]
    )

    assert patients.patient_ids.tolist() == ["p1", "p2"]
    assert patients.n_slides.tolist() == [2, 1]
    np.testing.assert_allclose(patients.profiles[0], [0.45, 0.55])
    np.testing.assert_allclose(patients.profiles[1], [1.0, 0.0])


def test_closest_candidate_census_uses_all_assignment_linear_quantiles() -> None:
    rows = pd.DataFrame(
        {
            "patient_id": [f"p{index:02d}" for index in range(20)] + ["p00"],
            "subcohort": ["A"] * 21,
            "slide_id": [f"s{index:02d}" for index in range(20)] + ["s-extra"],
            "tile_id": [f"t{index:02d}" for index in range(20)] + ["t-extra"],
            "distance": list(range(20)) + [9.5],
        }
    )

    candidates = closest_per_patient_candidates(rows)

    assert len(candidates) == 20
    assert candidates.loc[candidates.patient_id.eq("p00"), "tile_id"].item() == "t00"
    # All 21 assignment distances define the thresholds: q10=2 and q25=5.
    assert candidates["prototype_q10_distance"].unique().tolist() == [2.0]
    assert candidates["prototype_q25_distance"].unique().tolist() == [5.0]
    assert (candidates["eligibility_tier"] == 0).sum() == 3
    assert (candidates["eligibility_tier"] <= 1).sum() == 6


def _montage_candidates(capacities: dict[str, int], *, tier: int = 2) -> pd.DataFrame:
    records = []
    distance = 0.0
    for subcohort in sorted(capacities):
        for index in range(capacities[subcohort]):
            patient = f"{subcohort}-p{index:02d}"
            records.append(
                {
                    "patient_id": patient,
                    "subcohort": subcohort,
                    "slide_id": f"{patient}-s",
                    "tile_id": f"{patient}-t",
                    "distance": distance,
                    "eligibility_tier": tier,
                }
            )
            distance += 1.0
    return pd.DataFrame.from_records(records)


def test_montage_expands_and_redistributes_deficits_by_largest_pool() -> None:
    candidates = _montage_candidates({"A": 1, "B": 2, "C": 7, "D": 7})
    # Make exactly 11 candidates available by quartile, forcing all-patient expansion.
    candidates.loc[candidates.index[:11], "eligibility_tier"] = 1

    selection = select_montage_tiles(
        candidates,
        prototype_index=4,
        occurrence=0,
        subcohort_order=("A", "B", "C", "D"),
    )
    replay = select_montage_tiles(
        candidates,
        prototype_index=4,
        occurrence=0,
        subcohort_order=("A", "B", "C", "D"),
    )

    assert_frame_equal(selection.tiles, replay.tiles)
    assert selection.stage == "all"
    assert selection.seed == 20260819 + 400
    assert selection.support_status == "MONTAGE_SUPPORT_SUFFICIENT"
    assert selection.tiles["subcohort"].value_counts().to_dict() == {
        "C": 5,
        "D": 4,
        "B": 2,
        "A": 1,
    }
    assert len(selection.tiles) == 12
    assert not selection.tiles["patient_id"].duplicated().any()


def test_duplicate_montages_are_disjoint_when_24_patients_are_available() -> None:
    candidates = _montage_candidates({"A": 6, "B": 6, "C": 6, "D": 6})

    original, duplicate = select_duplicate_montages(
        candidates,
        prototype_index=0,
        subcohort_order=("A", "B", "C", "D"),
    )

    first_patients = set(original.tiles["patient_id"])
    second_patients = set(duplicate.tiles["patient_id"])
    assert len(first_patients) == len(second_patients) == 12
    assert first_patients.isdisjoint(second_patients)
    assert duplicate.overlap_count == 0


def test_duplicate_montages_attain_mathematical_minimum_overlap_below_24() -> None:
    candidates = _montage_candidates({"A": 5, "B": 5, "C": 4, "D": 4})

    original, duplicate = select_duplicate_montages(
        candidates,
        prototype_index=2,
        subcohort_order=("A", "B", "C", "D"),
    )

    assert len(original.tiles) == len(duplicate.tiles) == 12
    assert duplicate.overlap_count == 6  # 12 + 12 - 18 distinct patients
    assert len(set(original.tiles.patient_id) | set(duplicate.tiles.patient_id)) == 18


def test_insufficient_montage_shows_every_patient_and_records_status() -> None:
    candidates = _montage_candidates({"A": 3, "B": 2, "C": 2, "D": 2})

    selection = select_montage_tiles(
        candidates,
        prototype_index=1,
        occurrence=0,
        subcohort_order=("A", "B", "C", "D"),
    )

    assert len(selection.tiles) == len(candidates) == 9
    assert set(selection.tiles.patient_id) == set(candidates.patient_id)
    assert selection.support_status == "MONTAGE_SUPPORT_INSUFFICIENT"


def test_prototype_index_is_zero_based_position_in_sorted_inventory() -> None:
    assert zero_based_prototype_index(20, [30, 10, 20]) == 1
    with pytest.raises(ValueError, match="duplicates"):
        zero_based_prototype_index(10, [10, 10])


def test_hmac_codes_and_order_follow_exact_messages_and_raw_digest_sort() -> None:
    salt = bytes(range(32))
    occurrences = blinded_occurrences([2, 0, 1], [2])

    table = hmac_blinding_table(occurrences, salt)

    assert occurrences == [(0, 0), (1, 0), (2, 0), (2, 1)]
    assert table["code"].str.fullmatch(r"[A-Z2-7]{6}").all()
    assert table["code"].is_unique
    assert table["presentation_order"].tolist() == list(range(4))
    expected_order = sorted(
        occurrences,
        key=lambda item: (
            hmac.new(
                salt, f"order|{item[0]}|{item[1]}".encode(), hashlib.sha256
            ).digest(),
            item,
        ),
    )
    assert list(zip(table.prototype_id, table.occurrence, strict=True)) == expected_order
    for row in table.itertuples(index=False):
        digest = hmac.new(
            salt,
            f"code|{row.prototype_id}|{row.occurrence}".encode(),
            hashlib.sha256,
        ).digest()
        expected = base64.b32encode(digest).decode("ascii").rstrip("=")[:6]
        assert row.code == expected
        assert row.code_window_index == 0
        assert row.is_controlling_read == (row.occurrence == 0)
    assert salt_sha256(salt) == hashlib.sha256(salt).hexdigest()


def test_hmac_window_allocator_advances_nonoverlapping_and_exhausts_at_eight() -> None:
    chunks = ["AAAAAA", "BBBBBB", "CCCCCC", "DDDDDD", "EEEEEE", "FFFFFF", "GGGGGG", "HHHHHH"]
    encoded = "".join(chunks) + "IIII"

    assert choose_hmac_code(encoded, {"AAAAAA", "BBBBBB"}) == ("CCCCCC", 2)
    with pytest.raises(HMACCodeExhaustedError, match="eight"):
        choose_hmac_code(encoded, set(chunks))


def test_collision_free_salt_draw_uses_exactly_32_bytes() -> None:
    calls: list[int] = []

    def token_bytes(n_bytes: int) -> bytes:
        calls.append(n_bytes)
        return b"x" * n_bytes

    salt, table = draw_collision_free_hmac_salt(
        [(0, 0), (1, 0)], token_bytes=token_bytes
    )

    assert calls == [32]
    assert salt == b"x" * 32
    assert len(table) == 2
