import numpy as np
import pytest

from tools.final_v14_variant_mapping import match_matrix


def test_equal_size_matching_maximizes_joint_objective():
    matrix = np.array([[.91, .90], [.89, .10]])
    rows = match_matrix(matrix)
    assert [r["variant_prototype_id"] for r in rows] == [1, 0]
    assert all(r["geometrically_mappable"] for r in rows)


def test_unequal_size_keeps_only_reciprocal_matches():
    rows = match_matrix(np.array([[.99, .2], [.98, .1], [.1, .79]]))
    assert [r["variant_prototype_id"] for r in rows] == [0, None, 1]
    assert [r["geometrically_mappable"] for r in rows] == [True, False, False]
    assert rows[1]["cosine_similarity"] is None


def test_mutual_ties_use_lowest_ids_and_invalid_matrices_fail():
    rows = match_matrix(np.ones((3, 2)))
    assert [r["variant_prototype_id"] for r in rows] == [0, None, None]
    with pytest.raises(ValueError):
        match_matrix(np.array([[np.nan]]))
