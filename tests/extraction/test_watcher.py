"""Focused process-status tests for the continuous colon watcher."""

from __future__ import annotations

from pathlib import Path

from oceanpath.extraction.inventory import InventoryPaths
from oceanpath.extraction.stream_encoder import SlideEncoderConfig
from oceanpath.extraction.stream_state import exclusive_process_lock
from oceanpath.extraction.watcher import (
    WatcherConfig,
    _MetadataCache,
    watcher_status,
)
from oceanpath.workflows.streaming import _common_child_arguments, _parser
from oceanpath.workflows.streaming import _config as cli_config


def _config(tmp_path: Path) -> WatcherConfig:
    state_root = tmp_path / "state"
    manifests = tmp_path / "manifests"
    encoder = SlideEncoderConfig(
        output_root=tmp_path / "features",
        state_root=state_root,
        scratch_root=tmp_path / "scratch",
        hest_checkpoint_path=tmp_path / "hest.ckpt",
        uni_checkpoint_path=tmp_path / "uni.bin",
        conch_v15_checkpoint_path=tmp_path / "conch.bin",
        virchow2_checkpoint_path=tmp_path / "virchow2.bin",
    )
    return WatcherConfig(
        inventory=InventoryPaths.from_roots(tmp_path / "slides", manifests),
        encoder=encoder,
        mpp_sheet_path=manifests / "ready.csv",
        inventory_status_path=manifests / "inventory.csv",
        queue_status_path=manifests / "queue.csv",
        database_path=state_root / "queue.sqlite",
        lock_path=state_root / "watcher.lock",
    )


def _pid_is_invisible(_pid: int, _signal: int) -> None:
    raise ProcessLookupError


def test_cli_defaults_to_preserved_holes_and_propagates_override() -> None:
    default_args = _parser().parse_args(["status"])
    filled_args = _parser().parse_args(["status", "--no-remove-holes"])

    assert cli_config(default_args).encoder.remove_holes is True
    assert "--remove-holes" in _common_child_arguments(default_args)
    assert cli_config(filled_args).encoder.remove_holes is False
    assert "--no-remove-holes" in _common_child_arguments(filled_args)


def test_cli_frozen_inventory_disables_background_discovery() -> None:
    default_args = _parser().parse_args(["status"])
    frozen_args = _parser().parse_args(["status", "--frozen-inventory"])

    assert cli_config(default_args).continuous_discovery is True
    assert "--frozen-inventory" not in _common_child_arguments(default_args)
    assert cli_config(frozen_args).continuous_discovery is False
    assert "--frozen-inventory" in _common_child_arguments(frozen_args)


def test_status_uses_held_lock_when_pid_is_invisible(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr("oceanpath.extraction.watcher.os.kill", _pid_is_invisible)

    with exclusive_process_lock(cfg.lock_path):
        original = cfg.lock_path.read_bytes()
        status = watcher_status(cfg)
        assert cfg.lock_path.read_bytes() == original

    assert status["running"] is True
    assert status["lock_held"] is True
    assert status["pid_visible"] is False
    assert status["pid"] is not None


def test_status_reports_free_stale_lock_without_rewriting_it(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr("oceanpath.extraction.watcher.os.kill", _pid_is_invisible)
    with exclusive_process_lock(cfg.lock_path):
        pass
    original = cfg.lock_path.read_bytes()

    status = watcher_status(cfg)

    assert status["running"] is False
    assert status["lock_held"] is False
    assert status["pid_visible"] is False
    assert cfg.lock_path.read_bytes() == original


def test_metadata_cache_warms_only_matching_file_identity(tmp_path, monkeypatch):
    slide_root = tmp_path / "slides"
    slide = slide_root / "rih" / "SL-1.svs"
    slide.parent.mkdir(parents=True)
    slide.write_bytes(b"slide")
    stat = slide.stat()
    status = tmp_path / "inventory.csv"
    status.write_text(
        "wsi,mpp,cohort,status,source_status,size_bytes,mtime_ns\n"
        f"rih/SL-1.svs,0.5016,RIH,ready,original_standard,"
        f"{stat.st_size},{stat.st_mtime_ns}\n",
        encoding="utf-8",
    )
    cache = _MetadataCache()
    assert cache.seed_from_status(status, slide_root) == 1
    monkeypatch.setattr(
        "oceanpath.extraction.watcher.read_slide_mpp",
        lambda _path: (_ for _ in ()).throw(AssertionError("metadata reread")),
    )
    monkeypatch.setattr(
        "oceanpath.extraction.watcher.rih_original_requires_repair",
        lambda _path: (_ for _ in ()).throw(AssertionError("tile header reread")),
    )

    assert cache.mpp(slide) == 0.5016
    assert cache.repair(slide) is False


def test_metadata_cache_rechecks_changed_file(tmp_path, monkeypatch):
    slide_root = tmp_path / "slides"
    slide = slide_root / "TCGA" / "sample.svs"
    slide.parent.mkdir(parents=True)
    slide.write_bytes(b"old")
    stat = slide.stat()
    status = tmp_path / "inventory.csv"
    status.write_text(
        "wsi,mpp,cohort,status,source_status,size_bytes,mtime_ns\n"
        f"TCGA/sample.svs,0.25,TCGA,ready,download_verified,"
        f"{stat.st_size},{stat.st_mtime_ns}\n",
        encoding="utf-8",
    )
    cache = _MetadataCache()
    cache.seed_from_status(status, slide_root)
    slide.write_bytes(b"changed")
    monkeypatch.setattr("oceanpath.extraction.watcher.read_slide_mpp", lambda _path: 0.252)

    assert cache.mpp(slide) == 0.252
