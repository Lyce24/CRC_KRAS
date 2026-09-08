"""
Tests for the packed feature store and the fast dataloading paths built on it.

Coverage:
  pack_features        — round-trip fidelity, dtype, coords, staleness evidence,
                         atomic staging, refusal to pack non-finite features
  PackedFeatureStore   — index, row gathers, fork-safety, corruption detection
  SlideDataset(store=) — parity with the H5 path, subsample-before-read
  fixed_bag_size       — uniform bags, short-bag policies, collator fast path
  MILDataModule        — packed wiring, eval batching, stale-pack refusal
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

import oceanpath.datasets.packed as packed_module
from oceanpath.datasets.datamodule import MILCollator, SlideDataset
from oceanpath.datasets.packed import (
    COORDS_FILE,
    FEATURES_FILE,
    META_FILE,
    PackedFeatureStore,
    PackedStoreError,
    feature_inventory_sha256,
    pack_features,
    validate_packed_dir,
)

FEAT_DIM = 8


@pytest.fixture
def feature_dir(tmp_path) -> Path:
    """Three slides with distinctive, exactly-float16-representable values."""
    d = tmp_path / "features_test"
    d.mkdir()
    specs = {"slide_A": 100, "slide_B": 40, "slide_C": 7}
    base = 0
    for sid, n in specs.items():
        feats = (np.arange(n * FEAT_DIM, dtype=np.float32).reshape(n, FEAT_DIM) + base) / 8.0
        coords = np.stack([np.arange(n), np.arange(n) + 1], axis=1).astype(np.int64)
        with h5py.File(d / f"{sid}.h5", "w") as h5:
            h5.create_dataset("features", data=feats)
            h5.create_dataset("coords", data=coords)
        base += 10_000
    return d


@pytest.fixture
def pack_dir(tmp_path, feature_dir) -> Path:
    out = tmp_path / "packed_test"
    pack_features(feature_dir, out)
    return out


def _h5_features(feature_dir: Path, slide_id: str) -> np.ndarray:
    with h5py.File(feature_dir / f"{slide_id}.h5", "r") as h5:
        return h5["features"][...]


# ═════════════════════════════════════════════════════════════════════════════
# pack_features
# ═════════════════════════════════════════════════════════════════════════════


class TestPackFeatures:
    def test_creates_expected_files(self, pack_dir):
        for name in ("meta.json", "index.parquet", "features.bin", "coords.bin"):
            assert (pack_dir / name).is_file(), name

    def test_meta_counts(self, pack_dir):
        meta = json.loads((pack_dir / META_FILE).read_text())
        assert meta["n_slides"] == 3
        assert meta["total_patches"] == 147
        assert meta["feat_dim"] == FEAT_DIM
        assert meta["feat_dtype"] == "float16"
        assert meta["has_coords"] is True

    def test_index_offsets_are_cumulative(self, pack_dir):
        index = pd.read_parquet(pack_dir / "index.parquet")
        assert index["offset"].tolist() == [0, 100, 140]
        assert index["n_patches"].tolist() == [100, 40, 7]

    def test_features_bin_size_matches_index(self, pack_dir):
        meta = json.loads((pack_dir / META_FILE).read_text())
        expected = meta["total_patches"] * meta["feat_dim"] * 2  # float16
        assert (pack_dir / FEATURES_FILE).stat().st_size == expected

    def test_roundtrip_values_match_h5(self, feature_dir, pack_dir):
        store = PackedFeatureStore(pack_dir)
        for sid in ("slide_A", "slide_B", "slide_C"):
            packed = store.read_features(store.position(sid))
            np.testing.assert_allclose(
                packed.astype(np.float32),
                _h5_features(feature_dir, sid).astype(np.float16).astype(np.float32),
            )

    def test_float32_pack_is_bit_exact(self, tmp_path, feature_dir):
        out = tmp_path / "packed_f32"
        pack_features(feature_dir, out, feat_dtype="float32")
        store = PackedFeatureStore(out)
        np.testing.assert_array_equal(
            store.read_features(store.position("slide_A")),
            _h5_features(feature_dir, "slide_A"),
        )

    def test_streams_h5_in_bounded_blocks(self, tmp_path, feature_dir, monkeypatch):
        calls = []
        original = packed_module._read_h5_block

        def record_read(dataset, start, end):
            calls.append((dataset.name, start, end))
            return original(dataset, start, end)

        monkeypatch.setattr(packed_module, "_read_h5_block", record_read)
        pack_features(feature_dir, tmp_path / "packed_streamed", stream_chunk_size=13)

        assert len(calls) > 6
        assert all(end - start <= 13 for _, start, end in calls)
        assert any(name.endswith("/features") for name, _, _ in calls)
        assert any(name.endswith("/coords") for name, _, _ in calls)

    def test_accepts_legacy_leading_singleton_axis(self, tmp_path):
        source = tmp_path / "legacy_features"
        source.mkdir()
        features = np.arange(5 * FEAT_DIM, dtype=np.float32).reshape(1, 5, FEAT_DIM)
        coords = np.arange(10, dtype=np.int64).reshape(1, 5, 2)
        with h5py.File(source / "legacy.h5", "w") as h5:
            h5.create_dataset("features", data=features)
            h5.create_dataset("coords", data=coords)

        out = tmp_path / "packed_legacy"
        pack_features(source, out, stream_chunk_size=2)
        store = PackedFeatureStore(out)
        np.testing.assert_array_equal(store.read_features(0), features[0].astype(np.float16))
        np.testing.assert_array_equal(store.read_coords(0), coords[0].astype(np.int32))

    def test_coords_roundtrip(self, pack_dir):
        store = PackedFeatureStore(pack_dir)
        coords = store.read_coords(store.position("slide_B"))
        assert coords.shape == (40, 2)
        np.testing.assert_array_equal(coords[:, 0], np.arange(40))
        np.testing.assert_array_equal(coords[:, 1], np.arange(40) + 1)

    def test_no_coords_option(self, tmp_path, feature_dir):
        out = tmp_path / "packed_nocoords"
        pack_features(feature_dir, out, include_coords=False)
        store = PackedFeatureStore(out)
        assert store.has_coords is False
        assert not (out / "coords.bin").exists()
        with pytest.raises(PackedStoreError, match="no coords"):
            store.read_coords(0)

    def test_slide_ids_subset(self, tmp_path, feature_dir):
        out = tmp_path / "packed_subset"
        meta = pack_features(feature_dir, out, slide_ids=["slide_A", "slide_C"])
        assert meta.n_slides == 2
        assert set(PackedFeatureStore(out).slide_ids) == {"slide_A", "slide_C"}

    def test_refuses_existing_dir_without_overwrite(self, tmp_path, feature_dir, pack_dir):
        with pytest.raises(FileExistsError):
            pack_features(feature_dir, pack_dir)

    def test_overwrite_rebuilds(self, feature_dir, pack_dir):
        meta = pack_features(feature_dir, pack_dir, overwrite=True)
        assert meta.n_slides == 3

    def test_failed_overwrite_preserves_previous_valid_pack(self, feature_dir, pack_dir):
        bad = np.full((5, FEAT_DIM), np.inf, dtype=np.float32)
        with h5py.File(feature_dir / "slide_zbad.h5", "w") as h5:
            h5.create_dataset("features", data=bad)
            h5.create_dataset("coords", data=np.zeros((5, 2), dtype=np.int64))

        with pytest.raises(ValueError, match="non-finite"):
            pack_features(feature_dir, pack_dir, overwrite=True, stream_chunk_size=2)

        assert PackedFeatureStore(pack_dir).meta.n_slides == 3

    def test_source_change_during_build_aborts_publication(
        self, tmp_path, feature_dir, monkeypatch
    ):
        hashes = iter(["a" * 64, "b" * 64])
        monkeypatch.setattr(packed_module, "feature_inventory_sha256", lambda _: next(hashes))
        target = tmp_path / "packed_racing_extraction"

        with pytest.raises(RuntimeError, match="changed while packing"):
            pack_features(feature_dir, target)

        assert not target.exists()

    def test_refuses_non_finite_features(self, tmp_path, feature_dir):
        bad = np.full((5, FEAT_DIM), np.nan, dtype=np.float32)
        with h5py.File(feature_dir / "slide_bad.h5", "w") as h5:
            h5.create_dataset("features", data=bad)
            h5.create_dataset("coords", data=np.zeros((5, 2), dtype=np.int64))
        with pytest.raises(ValueError, match="non-finite"):
            pack_features(feature_dir, tmp_path / "packed_bad")

    def test_failed_pack_leaves_no_usable_dir(self, tmp_path, feature_dir):
        """A crash mid-pack must not leave something a reader would accept."""
        bad = np.full((5, FEAT_DIM), np.inf, dtype=np.float32)
        with h5py.File(feature_dir / "slide_zbad.h5", "w") as h5:
            h5.create_dataset("features", data=bad)
            h5.create_dataset("coords", data=np.zeros((5, 2), dtype=np.int64))
        target = tmp_path / "packed_crash"
        with pytest.raises(ValueError):
            pack_features(feature_dir, target)
        assert not target.exists()

    def test_skips_zero_patch_slides(self, tmp_path, feature_dir):
        with h5py.File(feature_dir / "slide_empty.h5", "w") as h5:
            h5.create_dataset("features", data=np.zeros((0, FEAT_DIM), dtype=np.float32))
            h5.create_dataset("coords", data=np.zeros((0, 2), dtype=np.int64))
        meta = pack_features(feature_dir, tmp_path / "packed_skip")
        assert meta.n_slides == 3

    def test_rejects_mixed_feature_dims(self, tmp_path, feature_dir):
        with h5py.File(feature_dir / "slide_wide.h5", "w") as h5:
            h5.create_dataset("features", data=np.zeros((3, FEAT_DIM * 2), dtype=np.float32))
            h5.create_dataset("coords", data=np.zeros((3, 2), dtype=np.int64))
        with pytest.raises(ValueError, match="Inconsistent feature dim"):
            pack_features(feature_dir, tmp_path / "packed_mixed")


# ═════════════════════════════════════════════════════════════════════════════
# PackedFeatureStore
# ═════════════════════════════════════════════════════════════════════════════


class TestPackedFeatureStore:
    def test_lengths_and_membership(self, pack_dir):
        store = PackedFeatureStore(pack_dir)
        assert len(store) == 3
        assert "slide_A" in store and "nope" not in store
        assert store.length_of("slide_C") == 7
        assert store.lengths.tolist() == [100, 40, 7]

    def test_unknown_slide_id_raises(self, pack_dir):
        with pytest.raises(KeyError, match="nope"):
            PackedFeatureStore(pack_dir).position("nope")

    def test_row_gather_selects_exact_rows(self, feature_dir, pack_dir):
        store = PackedFeatureStore(pack_dir)
        rows = np.array([3, 17, 99])
        gathered = store.read_features(store.position("slide_A"), rows)
        full = _h5_features(feature_dir, "slide_A").astype(np.float16)
        np.testing.assert_array_equal(gathered, full[rows])

    def test_reads_never_alias_the_memmap(self, pack_dir):
        """Callers mutate features (noise augmentation); reads must be copies."""
        store = PackedFeatureStore(pack_dir)
        a = store.read_features(store.position("slide_A"))
        a += 1.0  # would raise on a read-only memmap view
        b = store.read_features(store.position("slide_A"))
        assert not np.shares_memory(a, b)
        assert b.max() < a.max()

    def test_missing_pack_dir_message_is_actionable(self, tmp_path):
        with pytest.raises(PackedStoreError, match="scripts/pack_features"):
            PackedFeatureStore(tmp_path / "does_not_exist")

    def test_truncated_features_file_detected(self, pack_dir):
        target = pack_dir / FEATURES_FILE
        with open(target, "r+b") as fh:
            fh.truncate(target.stat().st_size - 64)
        with pytest.raises(PackedStoreError, match="truncated or corrupt"):
            PackedFeatureStore(pack_dir)

    def test_truncated_coords_file_detected(self, pack_dir):
        target = pack_dir / COORDS_FILE
        with open(target, "r+b") as fh:
            fh.truncate(target.stat().st_size - 8)
        with pytest.raises(PackedStoreError, match="coords.bin.*truncated or corrupt"):
            PackedFeatureStore(pack_dir)

    def test_noncontiguous_index_offsets_detected(self, pack_dir):
        index_path = pack_dir / "index.parquet"
        index = pd.read_parquet(index_path)
        index.loc[1, "offset"] += 1
        index.to_parquet(index_path, index=False)
        with pytest.raises(PackedStoreError, match="offsets are not contiguous"):
            PackedFeatureStore(pack_dir)

    def test_public_validator_returns_metadata(self, pack_dir):
        meta = validate_packed_dir(pack_dir)
        assert meta.n_slides == 3
        assert meta.total_patches == 147

    def test_schema_version_mismatch_detected(self, pack_dir):
        meta = json.loads((pack_dir / META_FILE).read_text())
        meta["schema_version"] = 999
        (pack_dir / META_FILE).write_text(json.dumps(meta))
        with pytest.raises(PackedStoreError, match="schema"):
            PackedFeatureStore(pack_dir)

    def test_stale_pack_refused(self, feature_dir, pack_dir):
        store_hash = json.loads((pack_dir / META_FILE).read_text())["source_inventory_sha256"]
        assert store_hash == feature_inventory_sha256(feature_dir)
        # Re-extracting one slide changes size+mtime, hence the inventory hash.
        with h5py.File(feature_dir / "slide_A.h5", "w") as h5:
            h5.create_dataset("features", data=np.zeros((120, FEAT_DIM), dtype=np.float32))
            h5.create_dataset("coords", data=np.zeros((120, 2), dtype=np.int64))
        with pytest.raises(PackedStoreError, match="STALE"):
            PackedFeatureStore(pack_dir, verify_source=feature_inventory_sha256(feature_dir))

    def test_matching_source_hash_accepted(self, feature_dir, pack_dir):
        PackedFeatureStore(pack_dir, verify_source=feature_inventory_sha256(feature_dir))

    def test_pickles_without_open_memmaps(self, pack_dir):
        """DataLoader workers receive the store by pickle; memmaps must not ride along."""
        import pickle

        store = PackedFeatureStore(pack_dir)
        _ = store.read_features(0)  # force the memmap open
        assert store._features is not None
        revived = pickle.loads(pickle.dumps(store))
        assert revived._features is None
        np.testing.assert_array_equal(revived.read_features(0), store.read_features(0))


# ═════════════════════════════════════════════════════════════════════════════
# SlideDataset over a packed store
# ═════════════════════════════════════════════════════════════════════════════


LABELS = {"slide_A": 0, "slide_B": 1, "slide_C": 0}


def _dataset(feature_dir, store=None, **kw):
    return SlideDataset(
        feature_dir=str(feature_dir),
        slide_ids=["slide_A", "slide_B", "slide_C"],
        labels=LABELS,
        store=store,
        **kw,
    )


class TestSlideDatasetPacked:
    def test_parity_with_h5_path(self, feature_dir, pack_dir):
        packed = _dataset(feature_dir, PackedFeatureStore(pack_dir), is_train=False)
        plain = _dataset(feature_dir, is_train=False)
        assert packed.slide_ids == plain.slide_ids
        assert packed.lengths == plain.lengths
        assert packed.feat_dim == plain.feat_dim
        for i in range(len(plain)):
            torch.testing.assert_close(
                packed[i]["features"],
                plain[i]["features"].half().float(),  # pack is float16
            )

    def test_setup_opens_no_h5_files(self, feature_dir, pack_dir):
        """The whole point of the index: constructing a split touches no slides."""
        store = PackedFeatureStore(pack_dir)
        for path in feature_dir.glob("*.h5"):
            path.unlink()
        ds = _dataset(feature_dir, store, is_train=False)
        assert ds.lengths == [100, 40, 7]
        assert ds[0]["features"].shape == (100, FEAT_DIM)

    def test_missing_slide_in_store_is_reported(self, feature_dir, pack_dir):
        ds = SlideDataset(
            feature_dir=str(feature_dir),
            slide_ids=["slide_A", "ghost"],
            labels=LABELS,
            store=PackedFeatureStore(pack_dir),
            is_train=False,
        )
        assert ds.slide_ids == ["slide_A"]

    def test_subsample_caps_and_keeps_real_rows(self, feature_dir, pack_dir):
        store = PackedFeatureStore(pack_dir)
        ds = _dataset(feature_dir, store, is_train=True, max_instances=20)
        sample = ds[0]
        assert sample["features"].shape == (20, FEAT_DIM)
        full = set(map(tuple, store.read_features(0).astype(np.float32).tolist()))
        for row in sample["features"].numpy().tolist():
            assert tuple(row) in full

    def test_spatial_stratified_requires_packed_coords(self, tmp_path, feature_dir):
        out = tmp_path / "packed_nc"
        pack_features(feature_dir, out, include_coords=False)
        with pytest.raises(ValueError, match="no coords"):
            _dataset(feature_dir, PackedFeatureStore(out), cap_strategy="spatial_stratified")

    def test_spatial_stratified_works_with_coords(self, feature_dir, pack_dir):
        ds = _dataset(
            feature_dir,
            PackedFeatureStore(pack_dir),
            is_train=True,
            max_instances=20,
            cap_strategy="spatial_stratified",
            return_coords=True,
        )
        sample = ds[0]
        assert sample["features"].shape == (20, FEAT_DIM)
        assert sample["coords"].shape == (20, 2)

    def test_float16_preserved_when_not_forcing_fp32(self, feature_dir, pack_dir):
        ds = _dataset(
            feature_dir, PackedFeatureStore(pack_dir), is_train=False, force_float32=False
        )
        assert ds[0]["features"].dtype == torch.float16


class TestSpatialStratifiedRespectsCap:
    """Regression: the quota balancer could return MORE indices than the cap.

    When `k` is smaller than the number of occupied grid cells every quota is
    already 1, and the old surplus loop refused to take a cell below 1 — so it
    removed nothing and the bag came back oversized. The collator hid this by
    truncating, but the cap was not being honoured and `fixed_bag_size` bags
    came out ragged.
    """

    @pytest.mark.parametrize("k", [1, 3, 20, 37, 99])
    @pytest.mark.parametrize("is_train", [True, False])
    def test_never_exceeds_k(self, feature_dir, pack_dir, k, is_train):
        ds = _dataset(
            feature_dir,
            PackedFeatureStore(pack_dir),
            is_train=is_train,
            max_instances=k,
            cap_strategy="spatial_stratified",
            eval_crop_seed=0,
        )
        assert ds[0]["features"].shape[0] == k

    def test_indices_are_unique(self, feature_dir, pack_dir):
        ds = _dataset(
            feature_dir,
            PackedFeatureStore(pack_dir),
            is_train=True,
            max_instances=20,
            cap_strategy="spatial_stratified",
        )
        coords = ds._spatial_stratified_indices(
            np.stack([np.arange(100), np.arange(100) + 1], axis=1), 100, 20, 0
        )
        assert len(coords) == 20
        assert len(np.unique(coords)) == 20

    def test_dropped_cells_vary_across_draws(self, feature_dir, pack_dir):
        """Shedding must not always penalise the same corner of the slide."""
        ds = _dataset(
            feature_dir,
            PackedFeatureStore(pack_dir),
            is_train=True,
            max_instances=10,
            cap_strategy="spatial_stratified",
        )
        coords = np.stack([np.arange(100), np.arange(100) + 1], axis=1)
        draws = {
            tuple(sorted(ds._spatial_stratified_indices(coords, 100, 10, 0))) for _ in range(8)
        }
        assert len(draws) > 1


# ═════════════════════════════════════════════════════════════════════════════
# fixed_bag_size
# ═════════════════════════════════════════════════════════════════════════════


class TestFixedBagSize:
    @pytest.mark.parametrize("packed", [False, True])
    def test_every_bag_is_exactly_n(self, feature_dir, pack_dir, packed):
        store = PackedFeatureStore(pack_dir) if packed else None
        ds = _dataset(feature_dir, store, is_train=True, fixed_bag_size=32)
        # 100 (subsample), 40 (subsample), 7 (repeat-fill) all land on 32
        for i in range(len(ds)):
            assert ds[i]["features"].shape == (32, FEAT_DIM)

    def test_short_bag_repeat_uses_only_real_rows(self, feature_dir, pack_dir):
        store = PackedFeatureStore(pack_dir)
        ds = _dataset(feature_dir, store, is_train=True, fixed_bag_size=32)
        real = {tuple(r) for r in store.read_features(store.position("slide_C")).tolist()}
        sample = ds[ds.slide_ids.index("slide_C")]
        assert sample["features"].shape[0] == 32
        for row in sample["features"].half().numpy().tolist():
            assert tuple(row) in real

    def test_short_bag_pad_policy_leaves_bag_short(self, feature_dir, pack_dir):
        ds = _dataset(
            feature_dir,
            PackedFeatureStore(pack_dir),
            is_train=True,
            fixed_bag_size=32,
            short_bag_policy="pad",
        )
        assert ds[ds.slide_ids.index("slide_C")]["features"].shape == (7, FEAT_DIM)

    def test_train_sampling_is_stochastic(self, feature_dir, pack_dir):
        ds = _dataset(feature_dir, PackedFeatureStore(pack_dir), is_train=True, fixed_bag_size=32)
        a, b = ds[0]["features"], ds[0]["features"]
        assert not torch.equal(a, b), "train bags must be a fresh random view each epoch"

    def test_eval_sampling_is_deterministic(self, feature_dir, pack_dir):
        ds = _dataset(
            feature_dir,
            PackedFeatureStore(pack_dir),
            is_train=False,
            fixed_bag_size=32,
            eval_crop_seed=0,
        )
        torch.testing.assert_close(ds[0]["features"], ds[0]["features"])

    def test_rejects_instance_dropout(self, feature_dir, pack_dir):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _dataset(
                feature_dir,
                PackedFeatureStore(pack_dir),
                fixed_bag_size=32,
                instance_dropout=0.1,
            )

    def test_rejects_nonpositive_size(self, feature_dir):
        with pytest.raises(ValueError, match="fixed_bag_size"):
            _dataset(feature_dir, fixed_bag_size=0)

    def test_rejects_unknown_short_bag_policy(self, feature_dir):
        with pytest.raises(ValueError, match="short_bag_policy"):
            _dataset(feature_dir, short_bag_policy="truncate")


# ═════════════════════════════════════════════════════════════════════════════
# MILCollator uniform fast path
# ═════════════════════════════════════════════════════════════════════════════


def _sample(n, label=0, sid="s", dtype=torch.float32, coords=False):
    out = {
        "features": torch.arange(n * FEAT_DIM, dtype=dtype).reshape(n, FEAT_DIM),
        "label": label,
        "slide_id": sid,
        "length": n,
    }
    if coords:
        out["coords"] = torch.zeros(n, 2, dtype=torch.int32)
    return out


class TestUniformCollation:
    @pytest.fixture
    def collator(self):
        return MILCollator(max_instances=32, feat_dim=FEAT_DIM, batch_size=4, pin_memory=False)

    def test_uniform_batch_is_all_valid(self, collator):
        out = collator([_sample(32), _sample(32), _sample(32)])
        assert out["features"].shape == (3, 32, FEAT_DIM)
        assert out["mask"] is None
        assert out["lengths"].tolist() == [32, 32, 32]

    def test_uniform_batch_preserves_values(self, collator):
        s0, s1 = _sample(32), _sample(32)
        out = collator([s0, s1])
        torch.testing.assert_close(out["features"][0], s0["features"])
        torch.testing.assert_close(out["features"][1], s1["features"])

    def test_over_length_bag_is_not_stacked(self, collator):
        """A 40-row bag clamps to length 32 but must be truncated, not stacked."""
        out = collator([_sample(40), _sample(32)])
        assert out["features"].shape == (2, 32, FEAT_DIM)
        torch.testing.assert_close(out["features"][0], _sample(40)["features"][:32])

    def test_mixed_lengths_pad_and_mask(self, collator):
        out = collator([_sample(32), _sample(10)])
        assert out["mask"][0].sum() == 32
        assert out["mask"][1].sum() == 10
        assert torch.all(out["features"][1, 10:] == 0)

    def test_output_is_not_a_shared_buffer(self, collator):
        first = collator([_sample(32)])["features"]
        snapshot = first.clone()
        collator([_sample(32, label=1)])
        torch.testing.assert_close(first, snapshot)

    def test_output_is_pageable_for_the_pin_thread(self, collator):
        """The collator must not hand back pinned memory it cannot reuse."""
        assert collator([_sample(32)])["features"].is_pinned() is False

    def test_coords_follow_the_uniform_path(self, collator):
        out = collator([_sample(32, coords=True), _sample(32, coords=True)])
        assert out["coords"].shape == (2, 32, 2)


# ═════════════════════════════════════════════════════════════════════════════
# MILDataModule wiring
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def dm_env(tmp_path):
    """A 10-slide cohort with manifest, splits, H5 features and a matching pack."""
    from oceanpath.datasets.datamodule import MILDataModule

    feature_dir = tmp_path / "features_dm"
    feature_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(10):
        n = 20 + i * 10
        with h5py.File(feature_dir / f"slide_{i}.h5", "w") as h5:
            h5.create_dataset(
                "features", data=rng.standard_normal((n, FEAT_DIM)).astype(np.float32)
            )
            h5.create_dataset("coords", data=rng.integers(0, 500, (n, 2)).astype(np.int64))

    csv_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"filename": f"slide_{i}.svs", "label": i % 2} for i in range(10)]).to_csv(
        csv_path, index=False
    )

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    pd.DataFrame([{"slide_id": f"slide_{i}", "fold": i % 5} for i in range(10)]).to_parquet(
        splits_dir / "splits.parquet", index=False
    )

    packed = tmp_path / "packed_dm"
    pack_features(feature_dir, packed)

    def _make(**kw):
        defaults = dict(
            feature_dir=str(feature_dir),
            splits_dir=str(splits_dir),
            csv_path=str(csv_path),
            label_column="label",
            filename_column="filename",
            scheme="kfold",
            fold=0,
            batch_size=2,
            num_workers=0,
            class_weighted_sampling=False,
            verify_splits=False,
        )
        defaults.update(kw)
        return MILDataModule(**defaults)

    _make.feature_dir = feature_dir
    _make.pack_dir = packed
    return _make


class TestDataModulePacked:
    def test_reads_from_pack_when_configured(self, dm_env):
        dm = dm_env(packed_dir=str(dm_env.pack_dir))
        dm.setup(stage="fit")
        assert dm.train_dataset.store is not None
        assert dm.val_dataset.store is not None
        assert dm.feat_dim == FEAT_DIM

    def test_h5_path_still_default(self, dm_env):
        dm = dm_env()
        dm.setup(stage="fit")
        assert dm.train_dataset.store is None

    def test_stale_pack_refused(self, dm_env):
        pack = dm_env.pack_dir
        with h5py.File(dm_env.feature_dir / "slide_0.h5", "w") as h5:
            h5.create_dataset("features", data=np.zeros((99, FEAT_DIM), dtype=np.float32))
            h5.create_dataset("coords", data=np.zeros((99, 2), dtype=np.int64))
        dm = dm_env(packed_dir=str(pack))
        with pytest.raises(PackedStoreError, match="STALE"):
            dm.setup(stage="fit")

    def test_staleness_check_can_be_waived(self, dm_env):
        with h5py.File(dm_env.feature_dir / "slide_0.h5", "w") as h5:
            h5.create_dataset("features", data=np.zeros((99, FEAT_DIM), dtype=np.float32))
            h5.create_dataset("coords", data=np.zeros((99, 2), dtype=np.int64))
        dm = dm_env(packed_dir=str(dm_env.pack_dir), verify_packed_source=False)
        dm.setup(stage="fit")
        assert dm.train_dataset.store is not None

    def test_fixed_bag_size_yields_uniform_batches(self, dm_env):
        dm = dm_env(packed_dir=str(dm_env.pack_dir), fixed_bag_size=64, batch_size=4)
        dm.setup(stage="fit")
        for batch in dm.train_dataloader():
            assert batch["features"].shape[1] == 64
            assert batch["mask"] is None

    def test_drop_last_keeps_static_training_batch_dimension(self, dm_env):
        dm = dm_env(
            packed_dir=str(dm_env.pack_dir),
            fixed_bag_size=32,
            batch_size=3,
            drop_last=True,
        )
        dm.setup(stage="fit")
        assert all(batch["features"].shape[0] == 3 for batch in dm.train_dataloader())

    def test_final_inference_datamodule_uses_packed_store(self, dm_env):
        from oceanpath.eval.inference import _build_test_datamodule

        template = dm_env()
        dm = _build_test_datamodule(
            feature_dir=template.feature_dir,
            splits_dir=template.splits_dir,
            csv_path=template.csv_path,
            label_column=template.label_column,
            filename_column=template.filename_column,
            scheme=template.scheme,
            test_slide_ids=["slide_0"],
            batch_size=1,
            max_instances=None,
            num_workers=0,
            packed_dir=str(dm_env.pack_dir),
            force_float32=False,
        )
        assert dm.test_dataset.store is not None
        assert dm.test_dataset[0]["features"].dtype == torch.float16

    def test_eval_stays_on_full_bags_by_default(self, dm_env):
        """Capping the training signal must not silently cap evaluation."""
        dm = dm_env(packed_dir=str(dm_env.pack_dir), fixed_bag_size=64)
        dm.setup(stage="fit")
        assert dm.train_dataset.fixed_bag_size == 64
        assert dm.val_dataset.fixed_bag_size is None

    def test_eval_batch_size_defaults_to_train(self, dm_env):
        assert dm_env(batch_size=7).eval_batch_size == 7

    def test_eval_batches_are_bag_size_sorted(self, dm_env):
        """Sorted eval order keeps padding small when eval_batch_size > 1."""
        dm = dm_env(eval_batch_size=3)
        dm.setup(stage="fit")
        sizes = dm.val_dataset.get_bag_sizes()
        order = dm._eval_sampler(dm.val_dataset)
        assert order is not None
        assert list(sizes[order]) == sorted(sizes)

    def test_no_eval_sampler_for_batch_size_one(self, dm_env):
        dm = dm_env(eval_batch_size=1)
        dm.setup(stage="fit")
        assert dm._eval_sampler(dm.val_dataset) is None

    def test_eval_covers_every_slide_exactly_once(self, dm_env):
        dm = dm_env(eval_batch_size=3)
        dm.setup(stage="fit")
        seen = [sid for batch in dm.val_dataloader() for sid in batch["slide_ids"]]
        assert sorted(seen) == sorted(dm.val_dataset.slide_ids)

    def test_packed_and_h5_produce_the_same_eval_batches(self, dm_env):
        def collect(**kw):
            dm = dm_env(eval_batch_size=1, **kw)
            dm.setup(stage="fit")
            return {b["slide_ids"][0]: b["features"][0].float() for b in dm.val_dataloader()}

        h5_out = collect()
        packed_out = collect(packed_dir=str(dm_env.pack_dir))
        assert h5_out.keys() == packed_out.keys()
        for sid in h5_out:
            torch.testing.assert_close(packed_out[sid], h5_out[sid].half().float())


# ── Resident store ────────────────────────────────────────────────────────────


class TestResidentPackedStore:
    """CPU-device coverage of the device-resident store (CUDA not required)."""

    @pytest.fixture
    def resident(self, pack_dir):
        from oceanpath.datasets.packed import ResidentPackedStore

        return ResidentPackedStore(PackedFeatureStore(pack_dir), device="cpu")

    def test_full_read_parity_with_memmap(self, pack_dir, resident):
        base = PackedFeatureStore(pack_dir)
        for pos in range(len(base)):
            np.testing.assert_array_equal(
                resident.read_features(pos).numpy(), base.read_features(pos)
            )
            np.testing.assert_array_equal(
                resident.read_coords(pos).numpy(), base.read_coords(pos)
            )

    def test_row_gather_parity_with_memmap(self, pack_dir, resident):
        base = PackedFeatureStore(pack_dir)
        rows = np.array([1, 4, 17, 63], dtype=np.int64)
        np.testing.assert_array_equal(
            resident.read_features(0, rows).numpy(), base.read_features(0, rows)
        )
        np.testing.assert_array_equal(
            resident.read_coords(0, rows).numpy(), base.read_coords(0, rows)
        )

    def test_cpu_residency_owns_its_memory(self, pack_dir, resident):
        """RAM residency must be a real copy, not a disguised memmap view."""
        base = PackedFeatureStore(pack_dir)
        assert not np.shares_memory(resident._features.numpy(), np.asarray(base._feature_map()))

    def test_index_interface_matches_store(self, pack_dir, resident):
        base = PackedFeatureStore(pack_dir)
        assert resident.slide_ids == base.slide_ids
        assert list(resident.lengths) == list(base.lengths)
        assert resident.feat_dim == base.feat_dim
        assert resident.has_coords == base.has_coords
        assert "slide_A" in resident
        assert resident.length_of("slide_B") == base.length_of("slide_B")
        with pytest.raises(KeyError):
            resident.position("ghost")

    def test_dataset_returns_tensors_and_fast_collates(self, feature_dir, resident):
        ds = _dataset(
            feature_dir,
            resident,
            is_train=True,
            fixed_bag_size=16,
            force_float32=False,
        )
        s0, s1 = ds[0], ds[1]
        assert isinstance(s0["features"], torch.Tensor)
        assert s0["features"].dtype == torch.float16
        collator = MILCollator(max_instances=16, feat_dim=FEAT_DIM, batch_size=2)
        batch = collator([s0, s1])
        assert batch["features"].shape == (2, 16, FEAT_DIM)
        assert batch["features"].dtype == torch.float16
        assert batch["mask"] is None

    def test_spatial_stratified_accepts_tensor_coords(self, feature_dir, resident):
        ds = _dataset(
            feature_dir,
            resident,
            is_train=True,
            max_instances=20,
            cap_strategy="spatial_stratified",
            return_coords=True,
        )
        sample = ds[0]
        assert sample["features"].shape == (20, FEAT_DIM)
        assert sample["coords"].shape == (20, 2)

    def test_augment_tensor_path(self, feature_dir, resident):
        ds = _dataset(
            feature_dir,
            resident,
            is_train=True,
            instance_dropout=0.5,
            feature_noise_std=0.1,
            force_float32=False,
        )
        sample = ds[0]
        assert isinstance(sample["features"], torch.Tensor)
        assert sample["features"].dtype == torch.float16
        assert 1 <= sample["features"].shape[0] <= 100

    def test_resident_requires_packed_dir(self, feature_dir, tmp_path):
        from oceanpath.datasets.datamodule import MILDataModule

        with pytest.raises(ValueError, match="resident_device requires packed_dir"):
            MILDataModule(
                feature_dir=str(feature_dir),
                splits_dir=str(tmp_path),
                csv_path=str(tmp_path / "labels.csv"),
                resident_device="cpu",
            )


class TestAugmentDtypePreservation:
    def test_numpy_noise_keeps_float16(self, feature_dir, pack_dir):
        """Feature noise must not upcast fp16 bags to fp32 in the worker."""
        ds = _dataset(
            feature_dir,
            PackedFeatureStore(pack_dir),
            is_train=True,
            feature_noise_std=0.1,
            force_float32=False,
        )
        assert ds[0]["features"].dtype == torch.float16

    def test_noise_actually_perturbs(self, feature_dir, pack_dir):
        store = PackedFeatureStore(pack_dir)
        noisy = _dataset(
            feature_dir, store, is_train=True, feature_noise_std=0.5, force_float32=False
        )
        clean = _dataset(feature_dir, store, is_train=False, force_float32=False)
        assert not torch.equal(noisy[0]["features"], clean[0]["features"])

    def test_resident_store_refuses_to_pickle(self, pack_dir):
        import pickle

        from oceanpath.datasets.packed import ResidentPackedStore

        resident = ResidentPackedStore(PackedFeatureStore(pack_dir), device="cpu")
        with pytest.raises(PackedStoreError, match="cannot be pickled"):
            pickle.dumps(resident)

    def test_workers0_sampling_is_reproducible(self, feature_dir, pack_dir):
        """num_workers=0 (the resident mode) must reproduce bag sampling
        under the same global seed — _worker_init_fn never runs there."""

        def draw():
            torch.manual_seed(1234)
            ds = _dataset(
                feature_dir,
                PackedFeatureStore(pack_dir),
                is_train=True,
                fixed_bag_size=16,
                force_float32=False,
            )
            return ds[0]["features"]

        torch.testing.assert_close(draw(), draw())


class TestProcessStoreCache:
    """The packed store is opened once per process, not once per fold."""

    def test_second_datamodule_reuses_the_store(self, dm_env):
        dm1 = dm_env(packed_dir=str(dm_env.pack_dir))
        dm2 = dm_env(packed_dir=str(dm_env.pack_dir))
        s1, s2 = dm1._get_store(), dm2._get_store()
        assert s1 is s2, "CV folds must share one store instance per process"

    def test_rebuilt_pack_is_not_served_stale(self, dm_env):
        import time as _time

        dm1 = dm_env(packed_dir=str(dm_env.pack_dir))
        s1 = dm1._get_store()
        _time.sleep(0.01)  # ensure mtime_ns advances across the rebuild
        pack_features(dm_env.feature_dir, dm_env.pack_dir, overwrite=True)
        dm2 = dm_env(packed_dir=str(dm_env.pack_dir))
        s2 = dm2._get_store()
        assert s1 is not s2, "a rebuilt pack must invalidate the cache stamp"
