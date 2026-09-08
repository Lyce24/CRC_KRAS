"""Focused training-identity tests for manifest-selected H5 inventories."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from oceanpath.workflows.training import (
    _feature_inventory_sha256,
    training_identity_payload,
)


def _inventory_hash(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def test_selected_inventory_normalizes_extensions_deduplicates_and_ignores_unrelated_h5s(
    tmp_path,
):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    selected_a = feature_dir / "selected-a.h5"
    selected_b = feature_dir / "selected-b.h5"
    unrelated = feature_dir / "unrelated.h5"
    selected_a.write_bytes(b"a-v1")
    selected_b.write_bytes(b"b-v1")
    unrelated.write_bytes(b"unrelated-v1")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "case_file": [
                "selected-a.SVS",
                "selected-a.h5",  # same canonical ID, intentionally duplicated
                "selected-b.tiff",
            ]
        }
    ).to_csv(manifest, index=False)

    first = _feature_inventory_sha256(feature_dir, manifest, "case_file")

    assert first == _inventory_hash(selected_a, selected_b)
    unrelated.write_bytes(b"unrelated-v2-with-a-new-size")
    (feature_dir / "later-extraction.h5").write_bytes(b"later")
    assert _feature_inventory_sha256(feature_dir, manifest, "case_file") == first

    selected_a.write_bytes(b"a-v2-with-a-new-size")
    assert _feature_inventory_sha256(feature_dir, manifest, "case_file") != first


def test_selected_but_missing_h5_has_stable_evidence_and_changes_when_published(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    present = feature_dir / "present.h5"
    present.write_bytes(b"present")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"slide": ["present.svs", "not-yet-published.tif"]}).to_csv(
        manifest, index=False
    )

    missing_hash = _feature_inventory_sha256(feature_dir, manifest, "slide")

    assert len(missing_hash) == 64
    assert _feature_inventory_sha256(feature_dir, manifest, "slide") == missing_hash
    (feature_dir / "not-yet-published.h5").write_bytes(b"now-present")
    assert _feature_inventory_sha256(feature_dir, manifest, "slide") != missing_hash


@pytest.mark.parametrize("manifest_state", ["missing", "unreadable", "no_column"])
def test_unavailable_manifest_selection_retains_full_directory_fallback(
    tmp_path, manifest_state
):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    first_h5 = feature_dir / "first.h5"
    first_h5.write_bytes(b"first")
    manifest = tmp_path / "manifest.csv"
    if manifest_state == "unreadable":
        manifest.write_bytes(b"\xff\xfe\x00not-a-utf8-csv")
    elif manifest_state == "no_column":
        pd.DataFrame({"other": ["first"]}).to_csv(manifest, index=False)

    first = _feature_inventory_sha256(feature_dir, manifest, "slide_id")

    assert first == _feature_inventory_sha256(feature_dir)
    (feature_dir / "second.h5").write_bytes(b"second")
    assert _feature_inventory_sha256(feature_dir, manifest, "slide_id") != first


def test_readable_empty_manifest_selects_empty_inventory_even_with_unrelated_h5(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    (feature_dir / "unrelated.h5").write_bytes(b"unrelated")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"slide_id": pd.Series(dtype=str)}).to_csv(manifest, index=False)

    assert _feature_inventory_sha256(feature_dir, manifest, "slide_id") == "empty"


def test_training_identity_uses_configured_filename_column_for_selected_inventory(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"custom_slide_column": ["selected.svs"], "label": [1]}).to_csv(
        manifest, index=False
    )
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    selected = feature_dir / "selected.h5"
    selected.write_bytes(b"selected-v1")

    cfg = OmegaConf.create(
        {
            "exp_name": "identity-test",
            "platform": {
                "project_root": str(tmp_path),
                "output_root": str(tmp_path / "outputs"),
                "accelerator": "cpu",
                "devices": 1,
                "strategy": "auto",
                "precision": "32-true",
                "num_nodes": 1,
            },
            "data": {
                "name": "synthetic",
                "csv_path": str(manifest),
                "feature_h5_dir": str(feature_dir),
                "filename_column": "custom_slide_column",
            },
            "encoder": {"name": "synthetic-encoder"},
            "extraction": {},
            "splits": {
                "name": "synthetic-split",
                "output_dir": str(tmp_path / "splits"),
            },
            "model": {"arch": "abmil"},
            "training": {"seed": 42},
        }
    )

    first = training_identity_payload(cfg)["input_evidence"]["feature_inventory_sha256"]

    (feature_dir / "unrelated.h5").write_bytes(b"unrelated")
    second = training_identity_payload(cfg)["input_evidence"]["feature_inventory_sha256"]
    assert second == first

    selected.write_bytes(b"selected-v2-with-a-new-size")
    third = training_identity_payload(cfg)["input_evidence"]["feature_inventory_sha256"]
    assert third != first
