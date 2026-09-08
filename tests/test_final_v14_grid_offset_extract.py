import errno
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tools import final_v14_grid_offset_extract as x


def test_archive_absence_blocks_before_runtime_sampling_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(x, "runtime", lambda **kw: pytest.fail("runtime/source preparation should not run"))
    monkeypatch.setattr(x.grid, "select_source_patients", lambda *a, **kw: pytest.fail("sampling forbidden"))
    with pytest.raises(x.grid.OperationalBlock, match="Canonical archive"):
        x.prepare(tmp_path / "pre", tmp_path / "missing", tmp_path / "uni", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_atomic_write_and_receipt_orphan_resume(tmp_path):
    artifact = tmp_path / "artifact.npz"
    x.atomic_write_once(artifact, b"finished immutable artifact")
    expected = {"contract_sha256": "contract", "stage": "OFFSET_UNI_FEATURES"}
    receipt = x._recover_or_publish_stage(artifact, expected)
    sidecar = Path(str(artifact) + ".receipt.json.seal.json")
    sidecar.unlink()
    assert x._recover_or_publish_stage(artifact, expected) == receipt
    assert sidecar.exists()
    with pytest.raises(x.common.ContractError, match="Refusing"):
        x.atomic_write_once(artifact, b"different")
    assert not list(tmp_path.glob("*.pending"))


def test_individual_missing_input_vs_archive_io_outage(tmp_path, monkeypatch):
    meta = tmp_path / x.PROFILE / "packed_uni_v1/meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text("present")
    monkeypatch.setattr(x.grid, "require_canonical_root", lambda root: None)
    monkeypatch.setattr(x.common, "digest", lambda path: x.PACK_META_SHA)
    raw_root = tmp_path / "raw_source"
    raw_root.mkdir()
    ready = tmp_path / "ready.csv"
    ready.write_text("synthetic inventory")
    monkeypatch.setattr(x, "WSI_ROOT", raw_root)
    monkeypatch.setattr(x, "READY", ready)
    row = {"slide_id": "s", "patient_id": "p", "subcohort": "SR1482"}
    result = x.eligibility_failure(row, FileNotFoundError(errno.ENOENT, "one missing WSI"), tmp_path)
    assert result["eligible"] is False
    with pytest.raises(x.grid.OperationalBlock, match="Systemic"):
        x.eligibility_failure(row, OSError(errno.EIO, "archive disconnected"), tmp_path)
    with pytest.raises(x.grid.OperationalBlock, match="Systemic"):
        x.eligibility_failure(row, PermissionError(errno.EACCES, "permission denied"), tmp_path)
    monkeypatch.setattr(x, "WSI_ROOT", tmp_path / "unmounted_raw_source")
    with pytest.raises(x.grid.OperationalBlock, match="Raw WSI root"):
        x.eligibility_failure(row, FileNotFoundError(errno.ENOENT, "drive disappeared"), tmp_path)


def test_feature_checkpoint_requires_same_coords_encoder_and_finite_features(tmp_path):
    path = tmp_path / "features.h5"
    coords = np.array([[128, 128], [384, 128]])
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=np.ones((2, 1024), np.float32)).attrs["encoder"] = "uni_v1"
        f.create_dataset("coords", data=coords)
    x.validate_feature_checkpoint(path, coords)
    with pytest.raises(x.common.ContractError, match="geometry"):
        x.validate_feature_checkpoint(path, coords[::-1])
    with h5py.File(path, "a") as f:
        f["features"][0, 0] = np.nan
    with pytest.raises(x.common.ContractError, match="Nonfinite"):
        x.validate_feature_checkpoint(path, coords)


def test_numpy_coordinate_metadata_serializes_without_losing_integer_geometry():
    value = x.normalize_json({"patch_size_level0": np.int64(513), "level0_mpp": np.float64(.2495),
                               "mode": b"exact_mpp", "shape": np.array([10, 20])})
    assert json.loads(json.dumps(value)) == {"patch_size_level0": 513, "level0_mpp": .2495, "mode": "exact_mpp", "shape": [10, 20]}


def test_historical_hest_receipt_without_new_policy_field_is_recognized(tmp_path):
    row = {"slide_id": "s", "mpp": .5}
    raw = {"size_bytes": 100, "mtime_ns": 123}
    paths = {}
    for stage in ("seg", "uni_v1_coords", "uni_v1_feat"):
        path = tmp_path / f"{stage}.json"
        paths[f"receipt_{stage}"] = path
        x.write_json(path, {"stage": stage, "source": {"output_id": "s", "size": 100, "mtime_ns": 123, "mpp": .5},
                           "checkpoint_hashes": {"uni_v1": x.grid.UNI_SHA256, "hest": x.HEST_SHA},
                           "implementation_hash": "11169bd9db126013246a80e74535020ee4f1cbe40eb8dd460ac2ba24da93f96d"})
    assert len(x.verify_receipts(paths, row, raw)) == 3
    path = paths["receipt_seg"]
    data = json.loads(path.read_text())
    data["implementation_hash"] = "unrecognized_history"
    path.write_text(json.dumps(data))
    with pytest.raises(x.grid.OperationalBlock, match="provenance schema"):
        x.verify_receipts(paths, row, raw)


def test_process_slide_quantizes_before_assignment_and_resumes_without_encoding(tmp_path, monkeypatch):
    # Synthetic adapter writes two arbitrary embeddings, never loads a WSI/model.
    from oceanpath.extraction import mpp_sampling
    coords = np.array([[128, 128], [384, 128]], dtype=np.int64)
    calls = []
    class FakeAdapter:
        def __init__(self, wsi, target_mpp):
            assert target_mpp == .5
        def extract_patch_features(self, model, coords_path, directory, **kwargs):
            calls.append(kwargs)
            path = Path(directory) / "s.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as f:
                f.create_dataset("features", data=np.full((2, 1024), .10001, dtype=np.float32)).attrs["encoder"] = "uni_v1"
                f.create_dataset("coords", data=coords)
    seen = []
    class FakeVocabulary:
        def assign_with_distances(self, features):
            seen.append(features.copy())
            return np.array([0, 1]), np.array([.5, 2.], dtype=np.float32)
    monkeypatch.setitem(sys.modules, "geopandas", SimpleNamespace(read_file=lambda path: "mask"))
    monkeypatch.setitem(sys.modules, "trident", SimpleNamespace(load_wsi=lambda *a, **kw: SimpleNamespace(release=lambda: None)))
    monkeypatch.setattr(mpp_sampling, "ExactMppWSIAdapter", FakeAdapter)
    monkeypatch.setattr(x.grid, "shifted_tissue_coordinates", lambda attrs, mask: coords)
    monkeypatch.setattr(x, "verify_slide_inputs", lambda row: None)
    row = {"slide_id": "s", "patient_id": "p", "subcohort": "SR1482", "mpp": .5,
           "wsi": {"path": "/nonexistent.svs"}, "mask": {"path": "/nonexistent.geojson"},
           "coordinates": {"sha256": "original"}, "canonical_attributes": {"patch_size_level0": 256}}
    first = x.process_slide(row, object(), FakeVocabulary(), np.ones(32), "contract", tmp_path)
    assert len(calls) == 1 and calls[0]["batch_limit"] == 64
    np.testing.assert_array_equal(seen[0], np.full((2, 1024), np.float16(.10001), dtype=np.float32))
    assert first["distance_support"][2]["nonempty_slides"] == 0
    assert first["distance_support"][2]["median_distance"] is None
    second = x.process_slide(row, object(), FakeVocabulary(), np.ones(32), "contract", tmp_path)
    assert first == second and len(calls) == 1
    # Crash after final JSON publication but before its seal: replay stage
    # checkpoints and reconstruct the same receipt without running the encoder.
    (tmp_path / "slides/s/receipt.json.seal.json").unlink()
    third = x.process_slide(row, object(), FakeVocabulary(), np.ones(32), "contract", tmp_path)
    assert third == first and len(calls) == 1
    with pytest.raises(x.common.ContractError, match="contract drift"):
        x.process_slide(row, object(), FakeVocabulary(), np.ones(32), "other_contract", tmp_path)
