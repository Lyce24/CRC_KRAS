"""
Comprehensive tests for dataset.py and datamodule.py.

Coverage:
  SlideDataset          — loading, shapes, dtypes, subsampling, augmentation, caching, edge cases
  MILCollator          — pre-allocated batching, padding, truncation, buffer reuse
  SimpleMILCollator    — dynamic padding, max_instances cap, variable bag sizes
  MILDataModule        — setup wiring, dataset_max_instances vs max_instances, collator selection,
                         weighted sampling, dataloader construction, properties

Fixture strategy:
  build_feature_dir()  — creates per-slide H5 feature files with known data,
                         matching the real format produced by the extraction
                         stage.  Every test that touches SlideDataset or
                         MILDataModule uses this fixture.
  make_sample()        — creates a single dataset sample dict (for collator-only tests).
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════


FEAT_DIM = 8
COORD_DIM = 2
FEAT_DTYPE = "float16"
COORD_DTYPE = "int32"


@pytest.fixture
def build_feature_dir(tmp_path):
    """
    Factory fixture: call it with slide specs to build per-slide H5 files.

    Usage:
        feature_dir = build_feature_dir([
            ("slide_A", 100, 0),   # (slide_id, n_patches, label)
            ("slide_B", 50, 1),
        ])

    Features are sequential floats so we can verify reads:
        slide_A features[i, j] ≈ A_global_start + i * FEAT_DIM + j  (as float16)
    Coords are (row, col) = (patch_idx, patch_idx+1).
    """

    def _build(
        specs: list[tuple[str, int, int]],
        feat_dim: int = FEAT_DIM,
        coord_dim: int = COORD_DIM,
        feat_dtype: str = FEAT_DTYPE,
        coord_dtype: str = COORD_DTYPE,
    ) -> tuple[Path, dict[str, int]]:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir(exist_ok=True)

        labels_map = {}
        total_patches = 0

        for sid, n_patches, lbl in specs:
            labels_map[sid] = lbl

            # Sequential features so we can verify correct reads
            feats = (
                np.arange(
                    total_patches * feat_dim,
                    (total_patches + n_patches) * feat_dim,
                    dtype=np.float32,
                )
                .reshape(n_patches, feat_dim)
                .astype(feat_dtype)
            )

            # Coords: (patch_global_idx, patch_global_idx+1)
            coords = np.stack(
                [
                    np.arange(total_patches, total_patches + n_patches),
                    np.arange(total_patches + 1, total_patches + n_patches + 1),
                ],
                axis=1,
            ).astype(coord_dtype)

            with h5py.File(feature_dir / f"{sid}.h5", "w") as h5:
                h5.create_dataset("features", data=feats)
                h5.create_dataset("coords", data=coords)

            total_patches += n_patches

        return feature_dir, labels_map

    return _build


def make_sample(
    n_patches: int,
    feat_dim: int = FEAT_DIM,
    label: int = 0,
    slide_id: str = "slide_X",
    with_coords: bool = False,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Create a single dataset-like sample dict for collator tests."""
    d = {
        "features": torch.randn(n_patches, feat_dim, dtype=dtype),
        "label": label,
        "slide_id": slide_id,
        "length": n_patches,
    }
    if with_coords:
        d["coords"] = torch.randint(0, 1000, (n_patches, COORD_DIM), dtype=torch.int32)
    return d


# ═════════════════════════════════════════════════════════════════════════════
# SlideDataset
# ═════════════════════════════════════════════════════════════════════════════


class TestSlideDatasetBasic:
    """Core loading, shapes, types, and output contract."""

    SPECS = (
        ("slide_A", 100, 0),
        ("slide_B", 50, 1),
        ("slide_C", 200, 2),
    )

    def _make_ds(self, build_feature_dir, **kwargs):
        feature_dir, labels = build_feature_dir(self.SPECS)
        defaults = {
            "feature_dir": str(feature_dir),
            "slide_ids": [s[0] for s in self.SPECS],
            "labels": labels,
            "is_train": False,
        }
        defaults.update(kwargs)
        from oceanpath.datasets import SlideDataset

        return SlideDataset(**defaults)

    def test_len(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir)
        assert len(ds) == 3

    def test_getitem_keys(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir)
        sample = ds[0]
        assert set(sample.keys()) == {"features", "label", "slide_id", "length"}

    def test_getitem_keys_with_coords(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir, return_coords=True)
        sample = ds[0]
        assert "coords" in sample

    def test_features_shape(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir)
        for i, (_, n_patches, _) in enumerate(self.SPECS):
            sample = ds[i]
            assert sample["features"].shape == (n_patches, FEAT_DIM)

    def test_features_dtype_always_float32(self, build_feature_dir):
        """Features stored as float16 must be upcast to float32."""
        ds = self._make_ds(build_feature_dir)
        assert ds[0]["features"].dtype == torch.float32

    def test_features_data_correctness(self, build_feature_dir):
        """Verify the sequential pattern written by the fixture is read correctly."""
        ds = self._make_ds(build_feature_dir)
        sample_a = ds[0]
        # slide_A starts at global patch 0, so features[0, :] ≈ [0,1,...,7] (as fp16→fp32)
        expected_first_row = np.arange(FEAT_DIM, dtype=np.float16).astype(np.float32)
        np.testing.assert_allclose(
            sample_a["features"][0].numpy(),
            expected_first_row,
            atol=1e-2,
        )

    def test_coords_shape_and_dtype(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir, return_coords=True)
        sample = ds[0]
        assert sample["coords"].shape == (100, COORD_DIM)
        assert sample["coords"].dtype == torch.int32

    def test_label_mapping(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir)
        for i, (_, _, expected_label) in enumerate(self.SPECS):
            assert ds[i]["label"] == expected_label

    def test_slide_id_preserved(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir)
        for i, (sid, _, _) in enumerate(self.SPECS):
            assert ds[i]["slide_id"] == sid

    def test_length_field_matches_features(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["length"] == sample["features"].shape[0]


class TestSlideDatasetSubsampling:
    """Subsampling behavior: stochastic (train), deterministic (val), boundaries."""

    SPECS = (("big_slide", 500, 0), ("small_slide", 10, 1))

    def _make_ds(self, build_feature_dir, **kwargs):
        feature_dir, labels = build_feature_dir(self.SPECS)
        defaults = {
            "feature_dir": str(feature_dir),
            "slide_ids": [s[0] for s in self.SPECS],
            "labels": labels,
        }
        defaults.update(kwargs)
        from oceanpath.datasets import SlideDataset

        return SlideDataset(**defaults)

    def test_no_subsampling_when_none(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir, max_instances=None, is_train=False)
        assert ds[0]["features"].shape[0] == 500

    def test_no_subsampling_when_under_cap(self, build_feature_dir):
        """Slide with 10 patches should not be subsampled even if max_instances=50."""
        ds = self._make_ds(build_feature_dir, max_instances=50, is_train=False)
        assert ds[1]["features"].shape[0] == 10

    def test_val_subsampling_deterministic_first_n(self, build_feature_dir):
        """Val (is_train=False): takes first N patches deterministically."""
        ds = self._make_ds(build_feature_dir, max_instances=20, is_train=False)
        s1 = ds[0]
        s2 = ds[0]
        assert s1["features"].shape[0] == 20
        torch.testing.assert_close(s1["features"], s2["features"])

    def test_val_subsampling_takes_first_patches(self, build_feature_dir):
        """Verify val actually uses indices [0..k) not random."""
        ds = self._make_ds(build_feature_dir, max_instances=5, is_train=False)
        sample = ds[0]
        # First patch of big_slide: sequential starting from 0
        expected = np.arange(FEAT_DIM, dtype=np.float16).astype(np.float32)
        np.testing.assert_allclose(sample["features"][0].numpy(), expected, atol=1e-2)

    def test_train_subsampling_caps_size(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir, max_instances=30, is_train=True)
        assert ds[0]["features"].shape[0] <= 30

    def test_train_subsampling_is_stochastic(self, build_feature_dir):
        """Two reads of the same slide should (almost certainly) differ."""
        ds = self._make_ds(build_feature_dir, max_instances=30, is_train=True)
        np.random.seed(None)  # noqa: NPY002
        s1 = ds[0]["features"].clone()
        s2 = ds[0]["features"].clone()
        # With 500 patches and k=30, collision probability is negligible
        assert not torch.equal(s1, s2)

    def test_subsampling_respects_coords(self, build_feature_dir):
        """When subsampling features, coords must be subsampled identically."""
        ds = self._make_ds(
            build_feature_dir,
            max_instances=20,
            is_train=False,
            return_coords=True,
        )
        sample = ds[0]
        assert sample["features"].shape[0] == sample["coords"].shape[0] == 20

    def test_length_field_reflects_subsampled_count(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir, max_instances=25, is_train=False)
        sample = ds[0]
        assert sample["length"] == 25


class TestSlideDatasetAugmentation:
    """Instance dropout and feature noise (train only)."""

    SPECS = (("slide_X", 200, 0),)

    def _make_ds(self, build_feature_dir, **kwargs):
        feature_dir, labels = build_feature_dir(self.SPECS)
        defaults = {
            "feature_dir": str(feature_dir),
            "slide_ids": ["slide_X"],
            "labels": labels,
        }
        defaults.update(kwargs)
        from oceanpath.datasets import SlideDataset

        return SlideDataset(**defaults)

    def test_instance_dropout_reduces_patches(self, build_feature_dir):
        """With dropout=0.5, most reads should have < 200 patches."""
        ds = self._make_ds(build_feature_dir, is_train=True, instance_dropout=0.5)
        sizes = [ds[0]["features"].shape[0] for _ in range(20)]
        # Very unlikely that ALL 20 reads keep exactly 200 patches
        assert any(s < 200 for s in sizes)

    def test_instance_dropout_guarantees_one_patch(self, build_feature_dir):
        """Even with dropout=0.99, at least 1 patch must survive."""
        ds = self._make_ds(build_feature_dir, is_train=True, instance_dropout=0.99)
        for _ in range(50):
            assert ds[0]["features"].shape[0] >= 1

    def test_instance_dropout_skipped_for_single_patch(self, build_feature_dir):
        """Slides with 1 patch: dropout must not remove it."""
        feature_dir, labels = build_feature_dir([("tiny", 1, 0)])
        from oceanpath.datasets import SlideDataset

        ds = SlideDataset(
            feature_dir=str(feature_dir),
            slide_ids=["tiny"],
            labels=labels,
            is_train=True,
            instance_dropout=0.99,
        )
        assert ds[0]["features"].shape[0] == 1

    def test_instance_dropout_syncs_coords(self, build_feature_dir):
        """After dropout, features and coords must have same length."""
        ds = self._make_ds(
            build_feature_dir,
            is_train=True,
            instance_dropout=0.5,
            return_coords=True,
        )
        for _ in range(10):
            sample = ds[0]
            assert sample["features"].shape[0] == sample["coords"].shape[0]

    def test_feature_noise_changes_values(self, build_feature_dir):
        """Feature noise should perturb features (train only)."""
        ds_noisy = self._make_ds(
            build_feature_dir,
            is_train=True,
            feature_noise_std=1.0,
            instance_dropout=0.0,
        )
        ds_clean = self._make_ds(
            build_feature_dir,
            is_train=True,
            feature_noise_std=0.0,
            instance_dropout=0.0,
        )
        # With noise_std=1.0 and no dropout, features should differ
        noisy = ds_noisy[0]["features"]
        clean = ds_clean[0]["features"]
        assert not torch.allclose(noisy, clean, atol=0.01)

    def test_no_augmentation_for_val(self, build_feature_dir):
        """is_train=False: no dropout, no noise even if configured."""
        ds = self._make_ds(
            build_feature_dir,
            is_train=False,
            instance_dropout=0.9,
            feature_noise_std=5.0,
        )
        s1 = ds[0]["features"]
        s2 = ds[0]["features"]
        # Val should be deterministic → same result
        torch.testing.assert_close(s1, s2)
        assert s1.shape[0] == 200  # no dropout


class TestSlideDatasetCaching:
    """LRU cache behavior for val/test."""

    SPECS = (("s1", 20, 0), ("s2", 30, 1))

    def _make_ds(self, build_feature_dir, **kwargs):
        feature_dir, labels = build_feature_dir(self.SPECS)
        defaults = {
            "feature_dir": str(feature_dir),
            "slide_ids": [s[0] for s in self.SPECS],
            "labels": labels,
        }
        defaults.update(kwargs)
        from oceanpath.datasets import SlideDataset

        return SlideDataset(**defaults)

    def test_cache_disabled_for_train(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir, is_train=True, cache_size_mb=100)
        assert ds._cache is None

    def test_cache_enabled_for_val(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir, is_train=False, cache_size_mb=100)
        assert ds._cache is not None
        assert len(ds._cache) == 0

    def test_cache_disabled_when_zero_mb(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir, is_train=False, cache_size_mb=0)
        assert ds._cache is None

    def test_cache_stores_after_first_read(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir, is_train=False, cache_size_mb=100)
        _ = ds[0]
        assert 0 in ds._cache

    def test_cache_returns_same_object(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir, is_train=False, cache_size_mb=100)
        s1 = ds[0]
        s2 = ds[0]
        # Cached → should be the exact same dict
        assert s1 is s2

    def test_cache_eviction_under_pressure(self, build_feature_dir):
        """With tiny cache limit, earlier items get evicted."""
        # s1: 20 patches x 8 dims x 4 bytes = 640 bytes
        # s2: 30 patches x 8 dims x 4 bytes = 960 bytes
        # Set cache to ~1000 bytes → fits either sample alone, not both
        ds = self._make_ds(build_feature_dir, is_train=False, cache_size_mb=1000e-6)
        _ = ds[0]
        _ = ds[1]
        # First sample should have been evicted
        assert 0 not in ds._cache
        assert 1 in ds._cache


class TestSlideDatasetEdgeCases:
    """Edge cases: missing IDs, zero-patch slides, unmapped labels."""

    def test_missing_slide_ids_warns(self, build_feature_dir, caplog):
        feature_dir, labels = build_feature_dir([("slide_A", 10, 0)])
        from oceanpath.datasets import SlideDataset

        with caplog.at_level(logging.WARNING):
            ds = SlideDataset(
                feature_dir=str(feature_dir),
                slide_ids=["slide_A", "nonexistent_1", "nonexistent_2"],
                labels=labels,
                is_train=False,
            )
        assert len(ds) == 1
        assert any("have no H5" in r.message for r in caplog.records)

    def test_zero_patch_slide_skipped_with_warning(self, build_feature_dir, caplog):
        feature_dir, labels = build_feature_dir(
            [
                ("empty_slide", 0, 0),
                ("normal_slide", 10, 1),
            ]
        )
        from oceanpath.datasets import SlideDataset

        with caplog.at_level(logging.WARNING):
            ds = SlideDataset(
                feature_dir=str(feature_dir),
                slide_ids=["empty_slide", "normal_slide"],
                labels=labels,
                is_train=False,
            )
        assert len(ds) == 1
        assert ds.slide_ids == ["normal_slide"]
        assert any("0 patches" in r.message for r in caplog.records)

    def test_unmapped_label_defaults_to_minus_one(self, build_feature_dir, caplog):
        feature_dir, _ = build_feature_dir([("slide_A", 10, 0)])
        from oceanpath.datasets import SlideDataset

        with caplog.at_level(logging.WARNING):
            ds = SlideDataset(
                feature_dir=str(feature_dir),
                slide_ids=["slide_A"],
                labels={},  # empty → label defaults to -1
                is_train=False,
            )
        assert ds[0]["label"] == -1
        assert any("no label" in r.message for r in caplog.records)

    def test_empty_slide_ids_list(self, build_feature_dir):
        feature_dir, labels = build_feature_dir([("slide_A", 10, 0)])
        from oceanpath.datasets import SlideDataset

        ds = SlideDataset(
            feature_dir=str(feature_dir),
            slide_ids=[],
            labels=labels,
            is_train=False,
        )
        assert len(ds) == 0

    def test_single_patch_slide(self, build_feature_dir):
        feature_dir, labels = build_feature_dir([("tiny", 1, 0)])
        from oceanpath.datasets import SlideDataset

        ds = SlideDataset(
            feature_dir=str(feature_dir),
            slide_ids=["tiny"],
            labels=labels,
            is_train=False,
        )
        sample = ds[0]
        assert sample["features"].shape == (1, FEAT_DIM)
        assert sample["length"] == 1


class TestSlideDatasetLabelUtilities:
    """get_label_counts, get_all_labels, get_bag_sizes."""

    SPECS = (
        ("a", 10, 0),
        ("b", 20, 0),
        ("c", 30, 1),
        ("d", 40, 2),
    )

    def _make_ds(self, build_feature_dir):
        feature_dir, labels = build_feature_dir(self.SPECS)
        from oceanpath.datasets import SlideDataset

        return SlideDataset(
            feature_dir=str(feature_dir),
            slide_ids=[s[0] for s in self.SPECS],
            labels=labels,
            is_train=False,
        )

    def test_get_label_counts(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir)
        counts = ds.get_label_counts()
        assert counts == {0: 2, 1: 1, 2: 1}

    def test_get_all_labels(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir)
        labels = ds.get_all_labels()
        assert isinstance(labels, np.ndarray)
        np.testing.assert_array_equal(labels, [0, 0, 1, 2])

    def test_get_bag_sizes(self, build_feature_dir):
        ds = self._make_ds(build_feature_dir)
        sizes = ds.get_bag_sizes()
        assert isinstance(sizes, np.ndarray)
        np.testing.assert_array_equal(sizes, [10, 20, 30, 40])


# ═════════════════════════════════════════════════════════════════════════════
# MILCollator (pre-allocated)
# ═════════════════════════════════════════════════════════════════════════════


class TestMILCollator:
    """Pre-allocated collator for fixed-size training batches."""

    @pytest.fixture
    def collator(self):
        from oceanpath.datasets.datamodule import MILCollator

        return MILCollator(
            max_instances=50,
            feat_dim=FEAT_DIM,
            batch_size=4,
            pin_memory=False,
        )

    def test_output_keys(self, collator):
        batch = [make_sample(30), make_sample(40)]
        out = collator(batch)
        assert set(out.keys()) == {"features", "mask", "labels", "lengths", "slide_ids"}

    def test_features_shape_padded(self, collator):
        batch = [make_sample(30), make_sample(40)]
        out = collator(batch)
        assert out["features"].shape == (2, 50, FEAT_DIM)

    def test_mask_shape(self, collator):
        batch = [make_sample(30), make_sample(40)]
        out = collator(batch)
        assert out["mask"].shape == (2, 50)

    def test_mask_correctness(self, collator):
        batch = [make_sample(30), make_sample(10)]
        out = collator(batch)
        # First sample: 30 valid, rest padding
        assert out["mask"][0, :30].sum() == 30
        assert out["mask"][0, 30:].sum() == 0
        # Second sample: 10 valid
        assert out["mask"][1, :10].sum() == 10
        assert out["mask"][1, 10:].sum() == 0

    def test_padding_is_zeros(self, collator):
        batch = [make_sample(10)]
        out = collator(batch)
        # Padding region should be all zeros
        assert torch.all(out["features"][0, 10:] == 0)

    def test_truncation_when_exceeds_max(self, collator):
        """Bags larger than max_instances get truncated."""
        batch = [make_sample(100)]  # 100 > max_instances=50
        out = collator(batch)
        assert out["features"].shape == (1, 50, FEAT_DIM)
        assert out["lengths"][0] == 50
        assert out["mask"][0].sum() == 50

    def test_labels_dtype(self, collator):
        batch = [make_sample(10, label=0), make_sample(10, label=2)]
        out = collator(batch)
        assert out["labels"].dtype == torch.long
        torch.testing.assert_close(out["labels"], torch.tensor([0, 2]))

    def test_lengths_dtype(self, collator):
        batch = [make_sample(30), make_sample(10)]
        out = collator(batch)
        assert out["lengths"].dtype == torch.int32
        torch.testing.assert_close(
            out["lengths"],
            torch.tensor([30, 10], dtype=torch.int32),
        )

    def test_slide_ids_preserved(self, collator):
        batch = [make_sample(10, slide_id="A"), make_sample(10, slide_id="B")]
        out = collator(batch)
        assert out["slide_ids"] == ["A", "B"]

    def test_smaller_batch_than_buffer(self, collator):
        """Last batch may be smaller than the pre-allocated buffer (batch_size=4)."""
        batch = [make_sample(10)]
        out = collator(batch)
        assert out["features"].shape[0] == 1

    def test_features_cloned_from_buffer(self, collator):
        """Output must be cloned — not a view of the shared buffer."""
        batch1 = [make_sample(10)]
        out1 = collator(batch1)
        feats1 = out1["features"].clone()

        batch2 = [make_sample(10)]
        _ = collator(batch2)
        # If not cloned, out1["features"] would have been mutated by the second call
        # (We compare with the saved clone to check independence)
        torch.testing.assert_close(feats1, out1["features"])

    def test_coords_collation(self, collator):
        batch = [make_sample(20, with_coords=True), make_sample(30, with_coords=True)]
        out = collator(batch)
        assert "coords" in out
        assert out["coords"].shape == (2, 50, COORD_DIM)
        assert out["coords"].dtype == torch.int32

    def test_coords_padding_is_zeros(self, collator):
        batch = [make_sample(10, with_coords=True)]
        out = collator(batch)
        assert torch.all(out["coords"][0, 10:] == 0)

    def test_preserves_input_feature_dtype(self, collator):
        batch = [make_sample(10, dtype=torch.float16), make_sample(12, dtype=torch.float16)]
        out = collator(batch)
        assert out["features"].dtype == torch.float16


# ═════════════════════════════════════════════════════════════════════════════
# SimpleMILCollator (dynamic)
# ═════════════════════════════════════════════════════════════════════════════


class TestSimpleMILCollator:
    """Dynamic collator with optional max_instances cap."""

    def test_dynamic_padding_to_batch_max(self):
        from oceanpath.datasets.datamodule import SimpleMILCollator

        collator = SimpleMILCollator(max_instances=None)
        batch = [make_sample(10), make_sample(30), make_sample(20)]
        out = collator(batch)
        # Should pad to the largest bag in the batch = 30
        assert out["features"].shape == (3, 30, FEAT_DIM)

    def test_cap_applied(self):
        from oceanpath.datasets.datamodule import SimpleMILCollator

        collator = SimpleMILCollator(max_instances=15)
        batch = [make_sample(10), make_sample(30), make_sample(20)]
        out = collator(batch)
        # All truncated to cap=15, so max is 15
        assert out["features"].shape == (3, 15, FEAT_DIM)

    def test_no_cap_when_none(self):
        from oceanpath.datasets.datamodule import SimpleMILCollator

        collator = SimpleMILCollator(max_instances=None)
        batch = [make_sample(100)]
        out = collator(batch)
        assert out["features"].shape == (1, 100, FEAT_DIM)

    def test_cap_does_not_affect_smaller_bags(self):
        """A bag of size 10 with cap=50 should not be extended to 50."""
        from oceanpath.datasets.datamodule import SimpleMILCollator

        collator = SimpleMILCollator(max_instances=50)
        batch = [make_sample(10)]
        out = collator(batch)
        # Dynamic: only pads to batch max, which is 10
        assert out["features"].shape == (1, 10, FEAT_DIM)

    def test_mask_correctness_variable_sizes(self):
        from oceanpath.datasets.datamodule import SimpleMILCollator

        collator = SimpleMILCollator(max_instances=None)
        batch = [make_sample(5), make_sample(15)]
        out = collator(batch)
        # Padded to 15
        assert out["mask"][0, :5].sum() == 5
        assert out["mask"][0, 5:].sum() == 0
        assert out["mask"][1, :15].sum() == 15

    def test_mask_correctness_with_cap(self):
        from oceanpath.datasets.datamodule import SimpleMILCollator

        collator = SimpleMILCollator(max_instances=8)
        batch = [make_sample(5), make_sample(20)]
        out = collator(batch)
        # batch 0: 5 valid (< cap), batch 1: 8 valid (capped)
        assert out["lengths"][0] == 5
        assert out["lengths"][1] == 8
        assert out["mask"][0, :5].sum() == 5
        assert out["mask"][1, :8].sum() == 8

    def test_output_keys(self):
        from oceanpath.datasets.datamodule import SimpleMILCollator

        collator = SimpleMILCollator()
        batch = [make_sample(10)]
        out = collator(batch)
        assert set(out.keys()) == {"features", "mask", "labels", "lengths", "slide_ids"}

    def test_dtypes(self):
        from oceanpath.datasets.datamodule import SimpleMILCollator

        collator = SimpleMILCollator()
        batch = [make_sample(10)]
        out = collator(batch)
        assert out["features"].dtype == torch.float32
        assert out["mask"] is None  # uniform bag: all tokens are valid
        assert out["labels"].dtype == torch.long
        assert out["lengths"].dtype == torch.int32

    def test_padded_mask_is_bool(self):
        from oceanpath.datasets.datamodule import SimpleMILCollator

        out = SimpleMILCollator()([make_sample(5), make_sample(10)])
        assert out["mask"].dtype == torch.bool

    def test_coords_collation_with_cap(self):
        from oceanpath.datasets.datamodule import SimpleMILCollator

        collator = SimpleMILCollator(max_instances=10)
        batch = [make_sample(5, with_coords=True), make_sample(20, with_coords=True)]
        out = collator(batch)
        assert "coords" in out
        # Capped at 10, dynamic pad to max(5, 10) = 10
        assert out["coords"].shape == (2, 10, COORD_DIM)

    def test_single_sample_batch(self):
        from oceanpath.datasets.datamodule import SimpleMILCollator

        collator = SimpleMILCollator()
        batch = [make_sample(7)]
        out = collator(batch)
        assert out["features"].shape == (1, 7, FEAT_DIM)

    def test_preserves_input_feature_dtype(self):
        from oceanpath.datasets.datamodule import SimpleMILCollator

        collator = SimpleMILCollator()
        batch = [make_sample(7, dtype=torch.float16), make_sample(5, dtype=torch.float16)]
        out = collator(batch)
        assert out["features"].dtype == torch.float16


# ═════════════════════════════════════════════════════════════════════════════
# MILDataModule
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_splits_and_labels(tmp_path):
    """
    Create a mock CSV + splits parquet for MILDataModule tests.

    10 slides: slide_0 through slide_9.
    Labels: 0-4 cycle. Fold: kfold5 with fold = slide_idx % 5.
    """
    csv_path = tmp_path / "manifest.csv"
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    # CSV
    rows = []
    for i in range(10):
        rows.append({"filename": f"slide_{i}.svs", "label": i % 3})
    df_csv = pd.DataFrame(rows)
    df_csv.to_csv(csv_path, index=False)

    # Splits parquet (kfold with 5 folds)
    split_rows = []
    for i in range(10):
        split_rows.append({"slide_id": f"slide_{i}", "fold": i % 5})
    df_splits = pd.DataFrame(split_rows)
    df_splits.to_parquet(splits_dir / "splits.parquet", index=False)

    # Integrity hash file (skip verification in tests)
    return csv_path, splits_dir


class TestMILDataModule:
    """MILDataModule wiring: dataset construction, collator selection, sampling."""

    SLIDE_SPECS = tuple((f"slide_{i}", 50 + i * 10, i % 3) for i in range(10))

    @pytest.fixture
    def datamodule_env(self, build_feature_dir, mock_splits_and_labels):
        """Build feature dir + splits, return (feature_dir, csv_path, splits_dir, labels)."""
        feature_dir, labels = build_feature_dir(self.SLIDE_SPECS)
        csv_path, splits_dir = mock_splits_and_labels
        return feature_dir, csv_path, splits_dir, labels

    def _make_dm(self, datamodule_env, **kwargs):
        feature_dir, csv_path, splits_dir, _ = datamodule_env
        from oceanpath.datasets.datamodule import MILDataModule

        defaults = {
            "feature_dir": str(feature_dir),
            "splits_dir": str(splits_dir),
            "csv_path": str(csv_path),
            "label_column": "label",
            "filename_column": "filename",
            "scheme": "kfold",
            "fold": 0,
            "batch_size": 2,
            "max_instances": 100,
            "dataset_max_instances": None,
            "num_workers": 0,
            "class_weighted_sampling": False,
            "verify_splits": False,
            "use_preallocated_collator": True,
        }
        defaults.update(kwargs)
        return MILDataModule(**defaults)

    def test_setup_creates_datasets(self, datamodule_env):
        dm = self._make_dm(datamodule_env)
        dm.setup(stage="fit")
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_train_dataset_is_train(self, datamodule_env):
        dm = self._make_dm(datamodule_env)
        dm.setup(stage="fit")
        assert dm.train_dataset.is_train is True

    def test_val_dataset_is_not_train(self, datamodule_env):
        dm = self._make_dm(datamodule_env)
        dm.setup(stage="fit")
        assert dm.val_dataset.is_train is False

    def test_train_gets_dataset_max_instances(self, datamodule_env):
        """dataset_max_instances should be passed to train dataset only."""
        dm = self._make_dm(datamodule_env, dataset_max_instances=25)
        dm.setup(stage="fit")
        assert dm.train_dataset.max_instances == 25

    def test_val_gets_dataset_max_instances_for_fixed_n_eval(self, datamodule_env):
        """Val uses the same dataset cap so fixed-N evaluation stays memory-bounded."""
        dm = self._make_dm(datamodule_env, dataset_max_instances=25)
        dm.setup(stage="fit")
        assert dm.val_dataset.max_instances == 25

    def test_val_dataset_no_cache_when_zero(self, datamodule_env):
        dm = self._make_dm(datamodule_env, cache_size_mb=0)
        dm.setup(stage="fit")
        assert dm.val_dataset._cache is None

    def test_val_dataset_has_cache(self, datamodule_env):
        dm = self._make_dm(datamodule_env, cache_size_mb=100)
        dm.setup(stage="fit")
        assert dm.val_dataset._cache is not None

    def test_train_dataset_never_cached(self, datamodule_env):
        """Train is always stochastic — caching must be disabled."""
        dm = self._make_dm(datamodule_env, cache_size_mb=100)
        dm.setup(stage="fit")
        assert dm.train_dataset._cache is None

    def test_train_dataloader_works(self, datamodule_env):
        dm = self._make_dm(datamodule_env)
        dm.setup(stage="fit")
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert "features" in batch
        assert "mask" in batch
        assert batch["features"].ndim == 3

    def test_val_dataloader_works(self, datamodule_env):
        dm = self._make_dm(datamodule_env)
        dm.setup(stage="fit")
        dl = dm.val_dataloader()
        batch = next(iter(dl))
        assert "features" in batch

    def test_test_dataloader_raises_when_empty(self, datamodule_env):
        """kfold scheme has no test set → should raise RuntimeError."""
        dm = self._make_dm(datamodule_env)
        dm.setup(stage="fit")
        with pytest.raises(RuntimeError, match="No test set"):
            dm.test_dataloader()

    def test_explicit_test_slide_ids_override_split_scheme(self, datamodule_env):
        dm = self._make_dm(datamodule_env, test_slide_ids=["slide_0"])
        dm.setup(stage="test")
        assert dm.test_dataset.slide_ids == ["slide_0"]
        batch = next(iter(dm.test_dataloader()))
        assert batch["slide_ids"] == ["slide_0"]

    def test_setup_fails_when_declared_split_feature_is_missing(self, datamodule_env):
        feature_dir, _csv_path, _splits_dir, _labels = datamodule_env
        (feature_dir / "slide_1.h5").unlink()
        dm = self._make_dm(datamodule_env)
        with pytest.raises(FileNotFoundError, match="train split is missing"):
            dm.setup(stage="fit")

    def test_train_collator_is_preallocated(self, datamodule_env):
        dm = self._make_dm(datamodule_env, use_preallocated_collator=True, max_instances=100)
        from oceanpath.datasets.datamodule import MILCollator

        collator = dm._train_collator()
        assert isinstance(collator, MILCollator)

    def test_train_collator_falls_back_to_simple(self, datamodule_env):
        """Without max_instances, can't pre-allocate → SimpleMILCollator."""
        dm = self._make_dm(datamodule_env, max_instances=None)
        from oceanpath.datasets.datamodule import SimpleMILCollator

        collator = dm._train_collator()
        assert isinstance(collator, SimpleMILCollator)

    def test_eval_collator_matches_preallocation_setting(self, datamodule_env):
        dm = self._make_dm(datamodule_env)
        from oceanpath.datasets.datamodule import MILCollator

        collator = dm._eval_collator()
        assert isinstance(collator, MILCollator)

    def test_eval_collator_has_cap(self, datamodule_env):
        dm = self._make_dm(datamodule_env, max_instances=200)
        collator = dm._eval_collator()
        assert collator.max_instances == 200

    def test_num_classes_property(self, datamodule_env):
        dm = self._make_dm(datamodule_env)
        # Labels cycle through 0,1,2 → 3 classes
        assert dm.num_classes == 3

    def test_feat_dim_property(self, datamodule_env):
        dm = self._make_dm(datamodule_env)
        assert dm.feat_dim == FEAT_DIM

    def test_force_float32_false_preserves_native_dtype(self, datamodule_env):
        dm = self._make_dm(
            datamodule_env,
            force_float32=False,
            use_preallocated_collator=False,
        )
        dm.setup(stage="fit")
        batch = next(iter(dm.val_dataloader()))
        assert batch["features"].dtype == torch.float16

    def test_loader_kwargs_respect_memory_overrides(self, datamodule_env):
        dm = self._make_dm(
            datamodule_env,
            num_workers=2,
            prefetch_factor=1,
            pin_memory=False,
            persistent_workers=False,
        )
        loader_kwargs = dm._loader_kwargs()
        assert loader_kwargs["num_workers"] == 2
        assert loader_kwargs["prefetch_factor"] == 1
        assert loader_kwargs["pin_memory"] is False
        assert loader_kwargs["persistent_workers"] is False


class TestMILDataModuleSampler:
    """Class-weighted sampling logic."""

    SLIDE_SPECS = (
        ("s0", 10, 0),
        ("s1", 10, 0),
        ("s2", 10, 0),
        ("s3", 10, 0),
        ("s4", 10, 0),
        ("s5", 10, 0),
        ("s6", 10, 0),
        ("s7", 10, 0),
        ("s8", 10, 1),
        ("s9", 10, 1),
    )

    @pytest.fixture
    def datamodule_env(self, build_feature_dir, tmp_path):
        feature_dir, labels = build_feature_dir(self.SLIDE_SPECS)

        csv_path = tmp_path / "manifest.csv"
        rows = [{"filename": f"{sid}.svs", "label": lbl} for sid, _, lbl in self.SLIDE_SPECS]
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        split_rows = [
            {"slide_id": sid, "fold": i % 5} for i, (sid, _, _) in enumerate(self.SLIDE_SPECS)
        ]
        pd.DataFrame(split_rows).to_parquet(splits_dir / "splits.parquet", index=False)

        return feature_dir, csv_path, splits_dir, labels

    def _make_dm(self, datamodule_env, **kwargs):
        feature_dir, csv_path, splits_dir, _ = datamodule_env
        from oceanpath.datasets.datamodule import MILDataModule

        defaults = {
            "feature_dir": str(feature_dir),
            "splits_dir": str(splits_dir),
            "csv_path": str(csv_path),
            "label_column": "label",
            "filename_column": "filename",
            "scheme": "kfold",
            "fold": 0,
            "batch_size": 2,
            "max_instances": 100,
            "num_workers": 0,
            "verify_splits": False,
            "use_preallocated_collator": False,
        }
        defaults.update(kwargs)
        return MILDataModule(**defaults)

    def test_weighted_sampler_created(self, datamodule_env):
        dm = self._make_dm(datamodule_env, class_weighted_sampling=True)
        dm.setup(stage="fit")
        from torch.utils.data import WeightedRandomSampler

        sampler = dm._get_sampler(dm.train_dataset)
        assert isinstance(sampler, WeightedRandomSampler)

    def test_no_sampler_when_disabled(self, datamodule_env):
        dm = self._make_dm(datamodule_env, class_weighted_sampling=False)
        dm.setup(stage="fit")
        sampler = dm._get_sampler(dm.train_dataset)
        assert sampler is None

    def test_weighted_sampler_balances_classes(self, datamodule_env):
        """With 8:2 class imbalance, weighted sampler should over-sample minority."""
        dm = self._make_dm(datamodule_env, class_weighted_sampling=True, batch_size=2)
        dm.setup(stage="fit")
        dl = dm.train_dataloader()

        # Sample over multiple epochs to reduce variance
        label_counts = {0: 0, 1: 0}
        for _epoch in range(10):
            for batch in dl:
                for lbl in batch["labels"].tolist():
                    if lbl in label_counts:
                        label_counts[lbl] += 1

        # With weighting over many samples, minority class should be well above
        # its raw 20% proportion (weighted sampling targets ~50%).
        total = label_counts[0] + label_counts[1]
        if total > 0:
            minority_frac = label_counts[1] / total
            assert minority_frac > 0.25, f"Minority class only got {minority_frac:.1%}"


class TestMILDataModuleBatchContract:
    """End-to-end: batches from dataloaders have the right structure for model forward."""

    SLIDE_SPECS = tuple((f"slide_{i}", 30 + i * 5, i % 2) for i in range(6))

    @pytest.fixture
    def datamodule_env(self, build_feature_dir, tmp_path):
        feature_dir, labels = build_feature_dir(self.SLIDE_SPECS)

        csv_path = tmp_path / "manifest.csv"
        rows = [{"filename": f"{sid}.svs", "label": lbl} for sid, _, lbl in self.SLIDE_SPECS]
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        split_rows = [
            {"slide_id": sid, "fold": i % 3} for i, (sid, _, _) in enumerate(self.SLIDE_SPECS)
        ]
        pd.DataFrame(split_rows).to_parquet(splits_dir / "splits.parquet", index=False)

        return feature_dir, csv_path, splits_dir, labels

    def _make_dm(self, datamodule_env, **kwargs):
        feature_dir, csv_path, splits_dir, _ = datamodule_env
        from oceanpath.datasets.datamodule import MILDataModule

        defaults = {
            "feature_dir": str(feature_dir),
            "splits_dir": str(splits_dir),
            "csv_path": str(csv_path),
            "label_column": "label",
            "filename_column": "filename",
            "scheme": "kfold",
            "fold": 0,
            "batch_size": 2,
            "max_instances": 80,
            "num_workers": 0,
            "class_weighted_sampling": False,
            "verify_splits": False,
            "use_preallocated_collator": True,
        }
        defaults.update(kwargs)
        return MILDataModule(**defaults)

    def test_batch_features_3d(self, datamodule_env):
        dm = self._make_dm(datamodule_env)
        dm.setup(stage="fit")
        batch = next(iter(dm.train_dataloader()))
        assert batch["features"].ndim == 3  # [B, N, D]

    def test_batch_mask_matches_features(self, datamodule_env):
        dm = self._make_dm(datamodule_env)
        dm.setup(stage="fit")
        batch = next(iter(dm.train_dataloader()))
        B, N, _ = batch["features"].shape
        assert batch["mask"].shape == (B, N)

    def test_batch_mask_binary(self, datamodule_env):
        dm = self._make_dm(datamodule_env)
        dm.setup(stage="fit")
        batch = next(iter(dm.train_dataloader()))
        unique = torch.unique(batch["mask"])
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_batch_labels_long(self, datamodule_env):
        dm = self._make_dm(datamodule_env)
        dm.setup(stage="fit")
        batch = next(iter(dm.train_dataloader()))
        assert batch["labels"].dtype == torch.long

    def test_masked_features_are_zero(self, datamodule_env):
        """Padding region (mask=0) must have zero features."""
        dm = self._make_dm(datamodule_env)
        dm.setup(stage="fit")
        batch = next(iter(dm.val_dataloader()))
        for i in range(batch["features"].shape[0]):
            padding_mask = batch["mask"][i] == 0
            if padding_mask.any():
                padded_feats = batch["features"][i][padding_mask]
                assert torch.all(padded_feats == 0)

    def test_val_batch_no_truncation_below_cap(self, datamodule_env):
        """Val slides under max_instances shouldn't lose patches."""
        dm = self._make_dm(datamodule_env, max_instances=500)  # all slides < 500
        dm.setup(stage="fit")
        for batch in dm.val_dataloader():
            for i in range(batch["features"].shape[0]):
                valid = int(batch["mask"][i].sum())
                assert valid == batch["lengths"][i]

    def test_train_batch_capped_at_max_instances(self, datamodule_env):
        """Train collator must not exceed max_instances dimension."""
        dm = self._make_dm(datamodule_env, max_instances=20)
        dm.setup(stage="fit")
        for batch in dm.train_dataloader():
            assert batch["features"].shape[1] == 20

    def test_full_epoch_iteration(self, datamodule_env):
        """All batches in one epoch should be consumable without error."""
        dm = self._make_dm(datamodule_env)
        dm.setup(stage="fit")
        n_train_batches = 0
        for batch in dm.train_dataloader():
            n_train_batches += 1
            assert batch["features"].shape[0] <= 2  # batch_size=2
        assert n_train_batches > 0

        n_val_batches = 0
        for _batch in dm.val_dataloader():
            n_val_batches += 1
        assert n_val_batches > 0


# ── predefined_kfold: adopt manifest fold assignments ────────────────────────


def _predefined_manifest(tmp_path, folds, patients=None):
    import pandas as pd

    n = len(folds)
    patients = patients or [f"p{i}" for i in range(n)]
    csv = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "filename": [f"s{i}.svs" for i in range(n)],
            "patient": patients,
            "label": [i % 2 for i in range(n)],
            "k_fold": folds,
        }
    ).to_csv(csv, index=False)
    return csv


def _predefined_cfg(csv, tmp_path, **kw):
    from oceanpath.splitting import SplitConfig

    return SplitConfig(
        scheme="predefined_kfold",
        name="pre",
        csv_path=str(csv),
        output_dir=str(tmp_path / "out"),
        filename_column="filename",
        label_column="label",
        group_column="patient",
        fold_column="k_fold",
        **kw,
    )


class TestPredefinedKFold:
    def test_adopts_folds_verbatim_and_marks_holdout(self, tmp_path):
        from oceanpath.splitting import generate_splits, get_slide_ids_for_fold

        folds = [-1, -1, 0, 0, 1, 1]
        csv = _predefined_manifest(tmp_path, folds)
        generate_splits(_predefined_cfg(csv, tmp_path, n_folds=2))

        import pandas as pd

        sp = pd.read_parquet(tmp_path / "out" / "splits.parquet")
        assert sorted(sp["fold"].tolist()) == sorted(folds)
        ids = get_slide_ids_for_fold(sp, 0, scheme="predefined_kfold")
        assert sorted(ids["test"]) == ["s0", "s1"]
        assert sorted(ids["val"]) == ["s2", "s3"]
        assert sorted(ids["train"]) == ["s4", "s5"]

    def test_rejects_group_spanning_folds(self, tmp_path):
        from oceanpath.splitting import generate_splits

        csv = _predefined_manifest(
            tmp_path, [-1, -1, 0, 1, 0, 1], patients=["a", "b", "c", "c", "d", "e"]
        )
        with pytest.raises(ValueError, match="span multiple folds"):
            generate_splits(_predefined_cfg(csv, tmp_path, n_folds=2))

    def test_allow_group_overlap_adopts_anyway(self, tmp_path):
        from oceanpath.splitting import generate_splits

        csv = _predefined_manifest(
            tmp_path, [-1, -1, 0, 1, 0, 1], patients=["a", "b", "c", "c", "d", "e"]
        )
        result = generate_splits(
            _predefined_cfg(csv, tmp_path, allow_group_overlap=True, n_folds=2)
        )
        assert result.n_slides == 6

    def test_group_shared_with_holdout_always_rejected(self, tmp_path):
        """Even with allow_group_overlap, a patient in both CV and the held-out
        cohort contaminates the external test and must fail."""
        from oceanpath.splitting import generate_splits

        csv = _predefined_manifest(
            tmp_path, [-1, 0, 0, 1, 1, -1], patients=["x", "x", "c", "d", "e", "f"]
        )
        with pytest.raises(ValueError, match="BOTH the CV pool and the held-out"):
            generate_splits(
                _predefined_cfg(csv, tmp_path, allow_group_overlap=True, n_folds=2)
            )

    def test_requires_fold_column(self, tmp_path):
        from oceanpath.splitting import SplitConfig, generate_splits

        csv = _predefined_manifest(tmp_path, [-1, 0, 0, 1, 1, -1])
        cfg = SplitConfig(
            scheme="predefined_kfold",
            name="pre",
            csv_path=str(csv),
            output_dir=str(tmp_path / "out"),
            filename_column="filename",
            label_column="label",
        )
        with pytest.raises(ValueError, match="requires 'fold_column'"):
            generate_splits(cfg)

    def test_rejects_non_contiguous_folds(self, tmp_path):
        from oceanpath.splitting import generate_splits

        csv = _predefined_manifest(tmp_path, [-1, 0, 0, 3, 3, -1])
        with pytest.raises(ValueError, match="contiguous"):
            generate_splits(_predefined_cfg(csv, tmp_path, n_folds=2))

    def test_rejects_missing_fold_values(self, tmp_path):
        import pandas as pd

        from oceanpath.splitting import generate_splits

        csv = _predefined_manifest(tmp_path, [-1, 0, 0, 1, 1, -1])
        df = pd.read_csv(csv)
        df.loc[2, "k_fold"] = None
        df.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="missing value"):
            generate_splits(_predefined_cfg(csv, tmp_path, n_folds=2))

    def test_n_folds_must_match_the_column(self, tmp_path):
        """Trainer plans folds from n_folds; a mismatch would silently run a
        partial CV and report it as complete."""
        from oceanpath.splitting import generate_splits

        csv = _predefined_manifest(tmp_path, [-1, 0, 0, 1, 1, -1])  # 2 CV folds
        with pytest.raises(ValueError, match="splits.n_folds"):
            generate_splits(_predefined_cfg(csv, tmp_path, n_folds=5))
