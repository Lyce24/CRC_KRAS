"""Contract tests for the Aim-3 atlas primitives (E3b) and the E3a verdict rule.

These cover the pieces where a silent error would be invisible in the output:
the sampling allocation that keeps one cohort from defining the vocabulary, the
projection/assignment round trip, the two per-slide quantities, and the
statistics that decide which prototypes get called KRAS-associated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oceanpath.aim1 import atlas


# ── sampling allocation ──────────────────────────────────────────────────────
def _slides(groups: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """One row per slide; ``groups`` maps group -> (n_patients, tiles_per_slide)."""
    rows = []
    for group, (n_patients, tiles) in groups.items():
        for patient in range(n_patients):
            rows.append({
                "slide_id": f"{group}_p{patient}",
                "patient_id": f"{group}_p{patient}",
                "atlas_group": group,
                "n_tiles": tiles,
            })
    return pd.DataFrame(rows)


def test_sample_plan_splits_the_budget_evenly_across_groups():
    # SurGen-shaped imbalance: one group has 40x the patients and 3x the tiles.
    slides = _slides({"big": (400, 15000), "small": (10, 5000)})
    plan = atlas.sample_plan(slides, total_tiles=40_000)
    by_group = plan.groupby("atlas_group")["n_sample"].sum()
    assert by_group["small"] > 0
    # Neither group may take more than ~its equal share of the budget.
    assert by_group["big"] <= 21_000
    assert by_group["small"] <= 21_000


def test_sample_plan_never_asks_a_slide_for_more_tiles_than_it_has():
    slides = _slides({"tiny": (3, 5), "large": (3, 100_000)})
    plan = atlas.sample_plan(slides, total_tiles=100_000)
    assert (plan["n_sample"] <= plan["n_tiles"]).all()


def test_sample_plan_requires_its_columns():
    with pytest.raises(ValueError, match="sample_plan needs columns"):
        atlas.sample_plan(pd.DataFrame({"slide_id": ["a"]}), total_tiles=10)


# ── vocabulary ───────────────────────────────────────────────────────────────
def _toy_sample(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = np.eye(4, 16, dtype=np.float32) * 5.0
    blocks = [centres[i] + rng.normal(scale=0.1, size=(200, 16)) for i in range(4)]
    return np.concatenate(blocks).astype(np.float32)


def test_vocabulary_recovers_well_separated_clusters():
    vocab = atlas.fit_vocabulary(_toy_sample(), n_prototypes=4, n_components=8, seed=1)
    labels = vocab.assign(_toy_sample())
    # Each true block of 200 must land on a single prototype.
    for block in range(4):
        assert len(np.unique(labels[block * 200:(block + 1) * 200])) == 1
    assert len(np.unique(labels)) == 4


def test_vocabulary_round_trips_through_disk(tmp_path):
    vocab = atlas.fit_vocabulary(_toy_sample(), n_prototypes=4, n_components=8, seed=1)
    path = tmp_path / "vocab.npz"
    vocab.save(path)
    reloaded = atlas.Vocabulary.load(path)
    assert reloaded.normalize == vocab.normalize
    assert reloaded.n_prototypes == 4
    sample = _toy_sample(seed=7)
    np.testing.assert_array_equal(vocab.assign(sample), reloaded.assign(sample))


def test_vocabulary_rejects_an_unknown_normalisation():
    with pytest.raises(ValueError, match="normalize must be"):
        atlas.fit_vocabulary(_toy_sample(), n_prototypes=2, n_components=4, normalize="zscore")


# ── per-slide quantities ─────────────────────────────────────────────────────
def test_slide_profile_separates_abundance_from_attention_mass():
    labels = np.array([0, 0, 0, 1], dtype=np.int16)
    # Prototype 1 is a quarter of the tissue but carries 70% of the attention:
    # the two quantities must not be collapsed into one number.
    weights = {42: np.array([0.1, 0.1, 0.1, 0.7])}
    profile = atlas.slide_profile(labels, weights, n_prototypes=3)
    np.testing.assert_allclose(profile["abundance"], [0.75, 0.25, 0.0])
    np.testing.assert_allclose(profile["attn_mass_seed42"], [0.3, 0.7, 0.0])
    assert profile["abundance"].sum() == pytest.approx(1.0)


def test_slide_profile_rejects_mismatched_attention_length():
    with pytest.raises(ValueError, match="attention values"):
        atlas.slide_profile(np.array([0, 1]), {42: np.array([1.0])}, n_prototypes=2)


# ── statistics ───────────────────────────────────────────────────────────────
def test_auc_effect_is_one_when_positives_are_uniformly_higher():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    positive = np.array([0, 0, 1, 1])
    assert atlas.auc_effect(values, positive) == pytest.approx(1.0)
    assert atlas.auc_effect(-values, positive) == pytest.approx(0.0)


def test_auc_effect_is_half_when_every_value_ties():
    assert atlas.auc_effect(np.zeros(6), np.array([0, 0, 0, 1, 1, 1])) == pytest.approx(0.5)


def test_bootstrap_auc_effect_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    positive = np.repeat([0, 1], 80)
    values = np.concatenate([rng.normal(0, 1, 80), rng.normal(1.2, 1, 80)])
    stats = atlas.bootstrap_auc_effect(values, positive, n_bootstrap=300, seed=5)
    assert stats["ci_low"] < stats["auc"] < stats["ci_high"]
    assert stats["ci_low"] > 0.5           # a real effect this size must clear chance
    assert stats["delta"] == pytest.approx(stats["auc"] - 0.5)


def test_benjamini_hochberg_matches_a_hand_computed_example():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    # q_i = min over j>=i of p_j * m / j, capped at 1.
    expected = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    np.testing.assert_allclose(atlas.benjamini_hochberg(p), expected)


def test_benjamini_hochberg_passes_nan_through():
    out = atlas.benjamini_hochberg(np.array([0.01, np.nan, 0.5]))
    assert np.isnan(out[1])
    assert np.isfinite(out[0]) and np.isfinite(out[2])


def test_concentration_reports_the_dominant_source():
    shares = pd.Series({"SurGen": 90.0, "TCGA": 10.0})
    result = atlas.concentration(shares)
    assert result["top_key"] == "SurGen"
    assert result["top_share"] == pytest.approx(0.9)


def test_concentration_is_undefined_for_an_empty_prototype():
    assert np.isnan(atlas.concentration(pd.Series({"SurGen": 0.0}))["top_share"])


def test_auc_effect_agrees_with_sklearn_including_heavy_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(11)
    positive = (rng.random(400) < 0.4).astype(int)
    # Most prototypes are absent from most slides, so exact zeros are the common
    # case and the tie handling is what has to match, not the generic path.
    tied = np.where(rng.random(400) < 0.6, 0.0, rng.random(400))
    continuous = rng.random(400)
    for values in (tied, continuous):
        assert atlas.auc_effect(values, positive) == pytest.approx(
            roc_auc_score(positive, values)
        )


def test_auc_effect_is_undefined_without_both_classes():
    assert np.isnan(atlas.auc_effect(np.arange(5.0), np.zeros(5, dtype=int)))
    assert np.isnan(atlas.auc_effect(np.arange(5.0), np.ones(5, dtype=int)))
