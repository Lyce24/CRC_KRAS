"""Correspondence resume preserves operational/scientific status separation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import final_v14_correspondence_replay as replay  # noqa: E402


def test_missing_mount_blocks_before_any_geometry_load(monkeypatch, tmp_path):
    receipt = {"status": "BLOCKED_OPERATIONAL_DEPENDENCY", "scientific_status": "PENDING_NOT_TESTED"}
    monkeypatch.setattr(replay, "preflight", lambda: (receipt, tmp_path / "receipt.json"))
    monkeypatch.setattr(replay, "normalized_centroids", lambda path: pytest.fail("Geometry opened while archive absent"))
    with pytest.raises(replay.PendingDependency, match="scientific geometry remains pending"):
        replay.run()


def test_versioned_payload_replay_preserves_initial_timestamp(tmp_path):
    path = tmp_path / "results.json"
    first = replay.publish_version(path, {"status": "READY", "value": 1})
    assert replay.publish_version(path, {"status": "READY", "value": 1}) == first
    with pytest.raises(replay.original.PostReaderError, match="conflict"):
        replay.publish_version(path, {"status": "READY", "value": 2})


def test_resolved_archive_link_still_requires_exact_digest(tmp_path):
    original = tmp_path / "archive.npz"
    original.write_bytes(b"frozen geometry")
    link = tmp_path / "geometry-link.npz"
    link.symlink_to(original)
    digest = replay.original.identity(original)["sha256"]
    assert replay.pinned_file(link, digest)["path"] == str(original)
    with pytest.raises(replay.original.PostReaderError, match="digest mismatch"):
        replay.pinned_file(link, "0" * 64)


def test_direct_joint_pair_collision_and_nine_variant_reporting():
    anchors = np.array([[1, 0, 0], [1, 0, 0]], dtype=float)
    variants = []
    for k in (24, 32, 40):
        for seed in (20260819, 20260820, 20260821):
            vectors = np.zeros((k, 3))
            vectors[:, 2] = 1
            vectors[0], vectors[1] = [1, 0, 0], [0.9, np.sqrt(0.19), 0]
            variants.append((k, seed, vectors))
    rows = replay.geometry_table(anchors, variants)
    assert len(rows) == 9
    assert all((row["p17"]["prototype_id"], row["p28"]["prototype_id"]) == (0, 1) for row in rows)
    axes = replay.axis_statuses({"name_gate_status": "NAME_GATE_FAIL"}, rows)
    assert axes["p17"]["geometry_passes"] == 9
    assert axes["p28"]["geometry_passes"] == 9
    assert all(axis["status"] == "CORRESPONDENCE_NOT_EVALUABLE" for axis in axes.values())


def test_geometry_backprojection_and_normalization(tmp_path):
    path = tmp_path / "geometry.npz"
    np.savez(path, pca_mean=np.array([1, 0, 0]), pca_components=np.array([[0, 2, 0], [0, 0, 1]]), centroids=np.array([[1, 0], [0, 1]]))
    actual = replay.normalized_centroids(path)
    expected = np.array([[1, 2, 0], [1, 0, 1]], dtype=float)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected)
