"""Publish-ordering and safety tests for local job-output staging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oceanpath.extraction.local_output import LocalOutputSlideEncoder
from oceanpath.extraction.stream_encoder import SlideEncoderConfig, SlideEncodingError


def _config(tmp_path: Path, **overrides) -> SlideEncoderConfig:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for name in ("hest.ckpt", "uni.bin", "conch.bin", "virchow2.bin"):
        (checkpoints / name).write_bytes(name.encode())
    defaults = dict(
        output_root=tmp_path / "durable",
        state_root=tmp_path / "state",
        scratch_root=tmp_path / "scratch",
        hest_checkpoint_path=checkpoints / "hest.ckpt",
        uni_checkpoint_path=checkpoints / "uni.bin",
        conch_v15_checkpoint_path=checkpoints / "conch.bin",
        virchow2_checkpoint_path=checkpoints / "virchow2.bin",
    )
    defaults.update(overrides)
    return SlideEncoderConfig(**defaults)


def _write_local_artifacts(encoder: LocalOutputSlideEncoder, output_id: str) -> list[Path]:
    files = encoder._publish_order(output_id)
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(json.dumps({"artifact": path.name}), encoding="utf-8")
        else:
            path.write_bytes(path.name.encode())
    return files


def test_local_root_defaults_under_scratch_and_all_paths_are_local(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    encoder = LocalOutputSlideEncoder(cfg)

    assert encoder.durable_root == (tmp_path / "durable").resolve()
    assert encoder.cfg.output_root.is_relative_to((tmp_path / "scratch").resolve())
    for path in encoder._publish_order("SL-1"):
        assert path.is_relative_to(encoder.cfg.output_root)


def test_local_root_must_be_disjoint_from_durable_root(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    with pytest.raises(ValueError):
        LocalOutputSlideEncoder(cfg, local_output_root=cfg.output_root / "nested")


def test_publish_moves_every_artifact_and_clears_local_copies(tmp_path: Path) -> None:
    encoder = LocalOutputSlideEncoder(_config(tmp_path))
    files = _write_local_artifacts(encoder, "SL-9")

    metrics = encoder._publish("SL-9")

    assert metrics["files"] == len(files)
    for path in files:
        assert not path.exists()
        durable = encoder.durable_root / path.relative_to(encoder.cfg.output_root)
        assert durable.is_file() and durable.stat().st_size > 0
    assert not list(encoder.durable_root.rglob("*.tmp"))
    completion = encoder.durable_root / "_stream_receipts" / "SL-9" / "complete.json"
    assert completion.is_file()


def test_publish_orders_completion_receipt_last(tmp_path: Path) -> None:
    encoder = LocalOutputSlideEncoder(_config(tmp_path))
    ordered = encoder._publish_order("SL-2")

    assert ordered[-1].name == "complete.json"
    receipt_positions = [
        index for index, path in enumerate(ordered) if "_stream_receipts" in path.parts
    ]
    artifact_positions = [
        index for index, path in enumerate(ordered) if "_stream_receipts" not in path.parts
    ]
    assert artifact_positions and receipt_positions
    assert max(artifact_positions) < min(receipt_positions)


def test_publish_archives_existing_durable_artifacts(tmp_path: Path) -> None:
    encoder = LocalOutputSlideEncoder(_config(tmp_path))
    _write_local_artifacts(encoder, "SL-3")
    stale = encoder.durable_root / "contours" / "SL-3.jpg"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale-version")

    encoder._publish("SL-3")

    archived = list((encoder.durable_root / ".archive").rglob("SL-3.jpg"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == b"stale-version"
    assert stale.read_bytes() != b"stale-version"


def test_publish_refuses_incomplete_local_artifacts(tmp_path: Path) -> None:
    encoder = LocalOutputSlideEncoder(_config(tmp_path))
    files = _write_local_artifacts(encoder, "SL-4")
    files[0].unlink()

    with pytest.raises(SlideEncodingError):
        encoder._publish("SL-4")

    assert not (encoder.durable_root / "_stream_receipts" / "SL-4" / "complete.json").exists()
