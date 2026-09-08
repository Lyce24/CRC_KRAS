from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from oceanpath.datasets.packed import (
    COORDS_FILE,
    FEATURES_FILE,
    INDEX_FILE,
    PackedFeatureStore,
    feature_inventory_sha256,
    pack_features,
)

SCRIPT = Path(__file__).parents[2] / "tools" / "pack_virchow2_after_encoding.py"
SPEC = importlib.util.spec_from_file_location("pack_virchow2_after_encoding", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
packer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packer)


def _write_source(root: Path, receipt_root: Path, slide_id: str, values: np.ndarray) -> None:
    coords = np.arange(len(values) * 2, dtype=np.int64).reshape(len(values), 2)
    with h5py.File(root / f"{slide_id}.h5", "w") as handle:
        features = handle.create_dataset("features", data=np.asarray(values, dtype=np.float32))
        features.attrs["encoder"] = "virchow2"
        features.attrs["name"] = slide_id
        handle.create_dataset("coords", data=coords)
    receipt = {
        "stage": "virchow2_feat",
        "implementation_hash": packer.IMPLEMENTATION_SHA256,
        "checkpoint_hashes": {"virchow2": packer.CHECKPOINT_SHA256},
        "source": {"output_id": slide_id},
        "validation": {"feature_dim": 4, "patch_count": len(values)},
    }
    path = receipt_root / slide_id / "virchow2_feat.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt))


@pytest.fixture
def tiny_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    feature_dir = tmp_path / "features"
    receipt_root = tmp_path / "receipts"
    feature_dir.mkdir()
    _write_source(
        feature_dir,
        receipt_root,
        "slide_a",
        np.arange(20, dtype=np.float32).reshape(5, 4),
    )
    _write_source(
        feature_dir,
        receipt_root,
        "slide_b",
        (np.arange(12, dtype=np.float32) + 100).reshape(3, 4),
    )
    monkeypatch.setattr(packer, "FEATURE_DIR", feature_dir)
    monkeypatch.setattr(packer, "RECEIPT_ROOT", receipt_root)
    monkeypatch.setattr(packer, "FULL_PACK_DIR", tmp_path / "packed_full")
    monkeypatch.setattr(packer, "CLS_PACK_DIR", tmp_path / "packed_cls")
    monkeypatch.setattr(packer, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(packer, "STATE_PATH", tmp_path / "run" / "state.json")
    monkeypatch.setattr(packer, "COMPLETION_PATH", tmp_path / "run" / "completion.json")
    monkeypatch.setattr(packer, "EXPECTED_SLIDES", 2)
    monkeypatch.setattr(packer, "SOURCE_DIM", 4)
    monkeypatch.setattr(packer, "CLS_DIM", 2)
    monkeypatch.setattr(packer, "COPY_CHUNK_ROWS", 2)
    return feature_dir, receipt_root


def test_cls_pack_is_exact_prefix_with_identical_index_and_coords(tiny_environment) -> None:
    feature_dir, _ = tiny_environment
    source_hash, counts = packer._validate_source(["slide_a", "slide_b"])
    assert source_hash == feature_inventory_sha256(feature_dir)

    cls_meta = packer._build_cls_pack(source_hash, counts)
    assert not packer.FULL_PACK_DIR.exists()
    full_meta = pack_features(
        feature_dir,
        packer.FULL_PACK_DIR,
        feat_dtype="float16",
        include_coords=True,
    )
    packer._validate_cls_relationship(
        full_meta,
        source_hash,
        counts,
        verify_full_prefix=True,
    )

    full = PackedFeatureStore(packer.FULL_PACK_DIR, verify_source=source_hash)
    cls = PackedFeatureStore(packer.CLS_PACK_DIR, verify_source=source_hash)
    assert cls_meta.feat_dim == 2
    assert np.array_equal(cls._feature_map(), full._feature_map()[:, :2])
    assert (packer.CLS_PACK_DIR / INDEX_FILE).read_bytes() == (
        packer.FULL_PACK_DIR / INDEX_FILE
    ).read_bytes()
    assert (packer.CLS_PACK_DIR / COORDS_FILE).read_bytes() == (
        packer.FULL_PACK_DIR / COORDS_FILE
    ).read_bytes()

    representation = json.loads((packer.CLS_PACK_DIR / "representation.json").read_text())
    assert representation["selected_columns"] == {
        "start_inclusive": 0,
        "stop_exclusive": 2,
    }
    assert representation["features_sha256"] == packer._sha256(
        packer.CLS_PACK_DIR / FEATURES_FILE
    )
    assert representation["full_prefix_sha256"] == representation["features_sha256"]
    assert representation["derivation"] == (
        "float16 cast of validated source H5 columns [0:1280]"
    )


def test_relationship_is_by_slide_id_when_physical_pack_orders_differ(
    tiny_environment,
) -> None:
    feature_dir, _ = tiny_environment
    # Model the production edge case: the queue/CLS writer and the generic full
    # packer can choose different valid physical block orders.
    source_hash, counts = packer._validate_source(["slide_b", "slide_a"])
    packer._build_cls_pack(source_hash, counts)
    full_meta = pack_features(
        feature_dir,
        packer.FULL_PACK_DIR,
        feat_dtype="float16",
        include_coords=True,
    )

    cls_index = json.loads(
        pd.read_parquet(packer.CLS_PACK_DIR / INDEX_FILE).to_json(orient="records")
    )
    full_index = json.loads(
        pd.read_parquet(packer.FULL_PACK_DIR / INDEX_FILE).to_json(orient="records")
    )
    assert [row["slide_id"] for row in cls_index] == ["slide_b", "slide_a"]
    assert [row["slide_id"] for row in full_index] == ["slide_a", "slide_b"]

    packer._validate_cls_relationship(
        full_meta,
        source_hash,
        counts,
        verify_full_prefix=True,
    )


def test_existing_cls_pack_detects_same_size_corruption(tiny_environment) -> None:
    _, _ = tiny_environment
    source_hash, counts = packer._validate_source(["slide_a", "slide_b"])
    cls_meta = packer._build_cls_pack(source_hash, counts)

    values = np.memmap(
        packer.CLS_PACK_DIR / FEATURES_FILE,
        dtype=np.float16,
        mode="r+",
        shape=(cls_meta.total_patches, 2),
    )
    values[1, 1] += np.float16(1)
    values.flush()
    del values
    with pytest.raises(RuntimeError, match="recorded SHA-256"):
        packer._validate_cls_source_pack(source_hash, counts)


def test_cls_first_rejects_nonfinite_value_outside_cls_half(
    tiny_environment,
) -> None:
    feature_dir, _ = tiny_environment
    source_hash, counts = packer._validate_source(["slide_a", "slide_b"])
    with h5py.File(feature_dir / "slide_a.h5", "r+") as handle:
        handle["features"][0, 3] = np.nan

    with pytest.raises(RuntimeError, match="Non-finite Virchow2 values"):
        packer._build_cls_pack(source_hash, counts)
    assert not packer.CLS_PACK_DIR.exists()
    assert not list(packer.CLS_PACK_DIR.parent.glob(".packed_cls.tmp-*"))


def test_cls_first_copy_failure_is_atomic(
    tiny_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _ = tiny_environment
    source_hash, counts = packer._validate_source(["slide_a", "slide_b"])
    original = packer._read_h5_rows
    calls = 0

    def fail_on_second_block(dataset, start: int, stop: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected copy failure")
        return original(dataset, start, stop)

    monkeypatch.setattr(packer, "_read_h5_rows", fail_on_second_block)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        packer._build_cls_pack(source_hash, counts)
    assert not packer.CLS_PACK_DIR.exists()
    assert not list(packer.CLS_PACK_DIR.parent.glob(".packed_cls.tmp-*"))


def test_source_contract_rejects_encoder_and_implementation_drift(
    tiny_environment,
) -> None:
    feature_dir, receipt_root = tiny_environment
    h5_path = feature_dir / "slide_a.h5"
    with h5py.File(h5_path, "r+") as handle:
        handle["features"].attrs["encoder"] = "wrong"
    with pytest.raises(RuntimeError, match="encoder='virchow2'"):
        packer._validate_source(["slide_a", "slide_b"])

    with h5py.File(h5_path, "r+") as handle:
        handle["features"].attrs["encoder"] = "virchow2"
    receipt_path = receipt_root / "slide_a" / "virchow2_feat.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["implementation_hash"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="implementation hash mismatch"):
        packer._validate_source(["slide_a", "slide_b"])


def test_source_contract_rejects_wrong_feature_dtype(tiny_environment) -> None:
    feature_dir, _ = tiny_environment
    h5_path = feature_dir / "slide_a.h5"
    with h5py.File(h5_path, "r+") as handle:
        values = handle["features"][:].astype(np.float16)
        del handle["features"]
        features = handle.create_dataset("features", data=values)
        features.attrs["encoder"] = "virchow2"
        features.attrs["name"] = "slide_a"
    with pytest.raises(RuntimeError, match="Expected float32"):
        packer._validate_source(["slide_a", "slide_b"])


def test_queue_completion_gate_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packer, "EXPECTED_SLIDES", 2)
    complete = {
        "total": 2,
        "complete": 2,
        "pending": 0,
        "processing": 0,
        "retry": 0,
        "failed": 0,
        "blocked": 0,
    }
    assert packer._queue_is_complete(complete)
    for status in ("pending", "processing", "retry", "failed", "blocked"):
        changed = dict(complete)
        changed["complete"] = 1
        changed[status] = 1
        assert not packer._queue_is_complete(changed)
