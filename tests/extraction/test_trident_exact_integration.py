"""Integration tests for OceanPath's exact-MPP TRIDENT boundary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from oceanpath.extraction import TridentExtractionConfig, ValidationError, validate_inputs
from oceanpath.extraction import trident as extraction
from oceanpath.workflows.extraction import _run_metadata_output_dir


def _touch_slides(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _write_selection(path: Path, rows: list[tuple[str, float | None]]) -> Path:
    include_mpp = any(mpp is not None for _, mpp in rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["wsi", "mpp"] if include_mpp else ["wsi"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for wsi, mpp in rows:
            row: dict[str, str | float] = {"wsi": wsi}
            if include_mpp:
                row["mpp"] = "" if mpp is None else mpp
            writer.writerow(row)
    return path


def _config(slides: Path, job: Path, selection: Path, **kwargs) -> TridentExtractionConfig:
    values = {
        "wsi_dir": str(slides),
        "job_dir": str(job),
        "custom_list_of_wsis": str(selection),
        "wsi_ext": [".svs", ".tiff"],
    }
    values.update(kwargs)
    return TridentExtractionConfig(**values)


def _write_generic_feature(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.ones((1, 3), dtype=np.float32))
        handle.create_dataset("coords", data=np.asarray([[0, 0]], dtype=np.int64))


def _write_generic_coords(cfg: TridentExtractionConfig, *slide_ids: str) -> None:
    for slide_id in slide_ids:
        path = Path(cfg.job_dir) / cfg.coords_subdir / "patches" / f"{slide_id}_patches.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("coords", data=np.asarray([[0, 0]], dtype=np.int64))


def _write_current_manifest(
    cfg: TridentExtractionConfig,
    *,
    path: Path | None = None,
    extraction_fingerprint: str | None = None,
) -> Path:
    stages = extraction.compute_stage_fingerprints(cfg)
    manifest_path = path or Path(cfg.job_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "extraction_fingerprint": extraction_fingerprint
                or extraction.compute_extraction_fingerprint(cfg),
                "stage_fingerprints": stages,
                "completed_stage_fingerprints": stages,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_exact_mode_validates_custom_wsi_mpp_contract(tmp_path: Path) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "first.svs", "second.tiff")
    valid = _write_selection(
        tmp_path / "valid.csv",
        [("first.svs", 0.252), ("second.tiff", 0.189012)],
    )

    summary = validate_inputs(
        _config(slides, tmp_path / "job", valid, target_mpp=0.5),
        ["coords"],
        create_job_dir=False,
    )

    assert summary["slide_count"] == 2
    assert summary["target_mpp"] == pytest.approx(0.5)

    missing_mpp = _write_selection(
        tmp_path / "missing.csv",
        [("first.svs", None), ("second.tiff", None)],
    )
    with pytest.raises(ValidationError, match="numeric 'mpp' column"):
        validate_inputs(
            _config(slides, tmp_path / "job", missing_mpp, target_mpp=0.5),
            ["coords"],
            create_job_dir=False,
        )


def test_diff_and_shard_csvs_preserve_explicit_mpp(tmp_path: Path) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "alpha.svs", "beta.svs", "gamma.svs")
    selection = _write_selection(
        tmp_path / "selection.csv",
        [("gamma.svs", 0.5), ("alpha.svs", 0.25), ("beta.svs", 0.189)],
    )
    cfg = _config(slides, tmp_path / "job", selection, patch_encoder="smoke")
    fingerprint = extraction.compute_extraction_fingerprint(cfg)
    output_dir = Path(cfg.job_dir) / cfg.coords_subdir / "features_smoke"
    output_dir.mkdir(parents=True)
    _write_generic_feature(output_dir / "alpha.h5")
    (Path(cfg.job_dir) / "manifest.json").write_text(
        json.dumps({"extraction_fingerprint": fingerprint}),
        encoding="utf-8",
    )

    diff_path = extraction.compute_extraction_diff(cfg, fingerprint)
    assert diff_path is not None
    with diff_path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"wsi": "beta.svs", "mpp": "0.189"},
            {"wsi": "gamma.svs", "mpp": "0.5"},
        ]

    shard_path = extraction.shard_wsi_list(cfg, shard_id=1, total_shards=2)
    with shard_path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"wsi": "beta.svs", "mpp": "0.189"},
        ]


def test_diff_mode_is_task_aware_for_missing_coordinates(tmp_path: Path) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "case.svs")
    selection = _write_selection(tmp_path / "selection.csv", [("case.svs", 0.25)])
    cfg = _config(slides, tmp_path / "job", selection)
    fingerprint = extraction.compute_extraction_fingerprint(cfg)
    (Path(cfg.job_dir) / "manifest.json").parent.mkdir(parents=True)
    (Path(cfg.job_dir) / "manifest.json").write_text(
        json.dumps({"extraction_fingerprint": fingerprint}),
        encoding="utf-8",
    )

    diff_path = extraction.compute_extraction_diff(
        cfg,
        fingerprint,
        tasks=["coords"],
    )

    assert diff_path is not None
    with diff_path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"wsi": "case.svs", "mpp": "0.25"}]


def test_processor_receives_a_canonical_validated_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "nested/case.svs")
    selection = _write_selection(tmp_path / "selection.csv", [(r"nested\case.svs", 0.25)])
    captured: dict[str, object] = {}

    class FakeProcessor:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.wsis = []

    monkeypatch.setattr(extraction, "_trident_imported", True)
    monkeypatch.setattr(extraction, "_Processor", FakeProcessor)

    extraction.create_processor(_config(slides, tmp_path / "job", selection))

    processor_selection = Path(str(captured["custom_list_of_wsis"]))
    assert processor_selection != selection
    with processor_selection.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"wsi": "nested/case.svs", "mpp": "0.25"}]


def test_empty_shard_is_a_clean_noop_with_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "only.svs")
    selection = _write_selection(tmp_path / "selection.csv", [("only.svs", 0.25)])
    cfg = _config(slides, tmp_path / "job", selection)
    monkeypatch.setattr(
        extraction,
        "create_processor",
        lambda _cfg: pytest.fail("empty shard must not create a TRIDENT Processor"),
    )

    result = extraction.run_pipeline(
        cfg,
        tasks=["coords"],
        shard_id=2,
        total_shards=3,
    )

    assert result == Path(cfg.job_dir)
    manifest = json.loads((Path(cfg.job_dir) / "manifest_shard_0002.json").read_text())
    assert manifest["requested_selection"]["path"] == str(selection)
    assert manifest["execution_selection"]["slide_count"] == 0
    assert manifest["execution_selection"]["path"].endswith("shard_0002.csv")


def test_force_archives_existing_features_before_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "case.svs")
    selection = _write_selection(tmp_path / "selection.csv", [("case.svs", 0.25)])
    cfg = _config(slides, tmp_path / "job", selection, patch_encoder="smoke")
    feature_path = Path(cfg.job_dir) / cfg.coords_subdir / "features_smoke" / "case.h5"
    feature_path.parent.mkdir(parents=True)
    feature_path.write_bytes(b"stale")
    _write_generic_coords(cfg, "case")
    _write_current_manifest(cfg)
    monkeypatch.setattr(extraction, "create_processor", lambda _cfg: object())

    def write_new_feature(_processor, _cfg) -> None:
        _write_generic_feature(feature_path)

    monkeypatch.setattr(extraction, "run_feature_extraction", write_new_feature)

    extraction.run_pipeline(cfg, tasks=["feat"], force=True)

    with h5py.File(feature_path, "r") as handle:
        assert handle["features"].shape == (1, 3)
    archived = list((Path(cfg.job_dir) / ".archive").glob("**/case.h5"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == b"stale"
    manifest = json.loads((Path(cfg.job_dir) / "manifest.json").read_text())
    assert len(manifest["archived_artifact_dirs"]) == 1


def test_default_rerun_refuses_to_bless_stale_filename_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "case.svs")
    selection = _write_selection(tmp_path / "selection.csv", [("case.svs", 0.25)])
    cfg = _config(slides, tmp_path / "job", selection, patch_encoder="smoke")
    _write_generic_feature(Path(cfg.job_dir) / cfg.coords_subdir / "features_smoke" / "case.h5")
    _write_generic_coords(cfg, "case")
    _write_current_manifest(cfg, extraction_fingerprint="stale")
    monkeypatch.setattr(
        extraction,
        "create_processor",
        lambda _cfg: pytest.fail("stale artifacts must be rejected before Processor creation"),
    )

    with pytest.raises(RuntimeError, match="pinned TRIDENT would skip"):
        extraction.run_pipeline(cfg, tasks=["feat"])


def test_shared_feature_directory_is_valid_for_each_nonempty_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "alpha.svs", "beta.svs")
    selection = _write_selection(
        tmp_path / "selection.csv",
        [("alpha.svs", 0.25), ("beta.svs", 0.5)],
    )
    cfg = _config(slides, tmp_path / "job", selection, patch_encoder="smoke")
    feature_dir = Path(cfg.job_dir) / cfg.coords_subdir / "features_smoke"
    _write_generic_feature(feature_dir / "alpha.h5")
    _write_generic_feature(feature_dir / "beta.h5")
    _write_generic_coords(cfg, "alpha", "beta")
    _write_current_manifest(
        cfg,
        path=Path(cfg.job_dir) / "manifest_shard_0000.json",
    )
    monkeypatch.setattr(extraction, "create_processor", lambda _cfg: object())
    monkeypatch.setattr(extraction, "run_feature_extraction", lambda _processor, _cfg: None)

    result = extraction.run_pipeline(
        cfg,
        tasks=["feat"],
        shard_id=0,
        total_shards=2,
    )

    assert result == Path(cfg.job_dir)
    manifest = json.loads((Path(cfg.job_dir) / "manifest_shard_0000.json").read_text())
    assert manifest["execution_selection"]["slide_count"] == 1


def test_array_workers_have_disjoint_run_metadata_directories(tmp_path: Path) -> None:
    first = _run_metadata_output_dir(tmp_path, 0, 8)
    second = _run_metadata_output_dir(tmp_path, 1, 8)

    assert first != second
    assert first == tmp_path / ".shard_runs" / "shard_0000_of_0008"
    assert second == tmp_path / ".shard_runs" / "shard_0001_of_0008"
    assert _run_metadata_output_dir(tmp_path, None, None) == tmp_path


def test_partial_feature_run_rejects_unproven_coordinate_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "case.svs")
    selection = _write_selection(tmp_path / "selection.csv", [("case.svs", 0.25)])
    cfg = _config(slides, tmp_path / "job", selection, patch_encoder="smoke")
    coords = Path(cfg.job_dir) / cfg.coords_subdir / "patches" / "case_patches.h5"
    coords.parent.mkdir(parents=True)
    with h5py.File(coords, "w") as handle:
        handle.create_dataset("coords", data=np.asarray([[0, 0]], dtype=np.int64))
    monkeypatch.setattr(
        extraction,
        "create_processor",
        lambda _cfg: pytest.fail("unproven coordinates must be rejected before Processor creation"),
    )

    with pytest.raises(RuntimeError, match="no manifest proves"):
        extraction.run_pipeline(cfg, tasks=["feat"], force=True)


def test_source_change_during_run_prevents_manifest_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "case.svs")
    selection = _write_selection(tmp_path / "selection.csv", [("case.svs", 0.25)])
    cfg = _config(slides, tmp_path / "job", selection, patch_encoder="smoke")
    feature_path = Path(cfg.job_dir) / cfg.coords_subdir / "features_smoke" / "case.h5"
    _write_generic_coords(cfg, "case")
    manifest_path = _write_current_manifest(cfg)
    original_manifest = manifest_path.read_text(encoding="utf-8")
    monkeypatch.setattr(extraction, "create_processor", lambda _cfg: object())

    def mutate_source_and_write_feature(_processor, _cfg) -> None:
        (slides / "case.svs").write_bytes(b"changed during extraction")
        _write_generic_feature(feature_path)

    monkeypatch.setattr(extraction, "run_feature_extraction", mutate_source_and_write_feature)

    with pytest.raises(RuntimeError, match="changed during extraction"):
        extraction.run_pipeline(cfg, tasks=["feat"])
    assert manifest_path.read_text(encoding="utf-8") == original_manifest


def test_slide_feature_invalidation_also_archives_patch_prerequisites(tmp_path: Path) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "case.svs")
    selection = _write_selection(tmp_path / "selection.csv", [("case.svs", 0.25)])
    cfg = _config(
        slides,
        tmp_path / "job",
        selection,
        patch_encoder="uni_v1",
        slide_encoder="titan",
    )
    slide_feature = Path(cfg.job_dir) / cfg.coords_subdir / "slide_features_titan" / "case.h5"
    patch_feature = Path(cfg.job_dir) / cfg.coords_subdir / "features_conch_v15" / "case.h5"
    unrelated_feature = Path(cfg.job_dir) / cfg.coords_subdir / "features_uni_v1" / "case.h5"
    _write_generic_feature(slide_feature)
    _write_generic_feature(patch_feature)
    _write_generic_feature(unrelated_feature)

    archive = extraction._archive_selected_artifacts(
        cfg,
        extraction._load_records(cfg),
        ["feat"],
        reason="test",
    )

    assert archive is not None
    assert not slide_feature.exists()
    assert not patch_feature.exists()
    assert unrelated_feature.is_file()
    assert (archive / slide_feature.relative_to(cfg.job_dir)).is_file()
    assert (archive / patch_feature.relative_to(cfg.job_dir)).is_file()


def test_partial_shard_arguments_are_rejected(tmp_path: Path) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "case.svs")
    selection = _write_selection(tmp_path / "selection.csv", [("case.svs", 0.25)])
    cfg = _config(slides, tmp_path / "job", selection)

    with pytest.raises(ValidationError, match="provided together"):
        extraction.run_pipeline(cfg, tasks=["coords"], shard_id=0)


def test_empty_task_list_is_rejected(tmp_path: Path) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "case.svs")
    selection = _write_selection(tmp_path / "selection.csv", [("case.svs", 0.25)])
    cfg = _config(slides, tmp_path / "job", selection)

    with pytest.raises(ValidationError, match="At least one extraction task"):
        extraction.run_pipeline(cfg, tasks=[])


def test_completed_stage_lineage_never_claims_unrun_downstream_stages() -> None:
    current = {"seg": "seg-new", "coords": "coords-new", "feat": "feat-new"}
    previous = {"completed_stage_fingerprints": dict(current)}

    completed = extraction._derive_completed_stage_fingerprints(
        previous,
        current,
        ["seg"],
        run_complete=True,
    )

    assert completed == {"seg": "seg-new"}


def _write_exact_feature(path: Path, *, source_mpp: float, target_mpp: float = 0.5) -> None:
    footprint = round(256 * target_mpp / source_mpp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.ones((1, 3), dtype=np.float32))
        coords = handle.create_dataset("coords", data=np.asarray([[0, 0]], dtype=np.int64))
        attrs = {
            "sampling_mode": "exact_mpp",
            "coords_schema_version": 2,
            "patch_size": 256,
            "patch_size_level0": footprint,
            "level0_mpp": source_mpp,
            "target_mpp": target_mpp,
            "overlap": 0,
            "min_tissue_proportion": 0.0,
            "read_level": 0,
            "read_level_downsample": 1.0,
            "read_patch_size": footprint,
            "actual_read_level0_pixels": footprint,
            "effective_target_mpp": source_mpp * footprint / 256,
            "mask_simplification": "none",
        }
        for key, value in attrs.items():
            coords.attrs[key] = value


def test_exact_output_validation_checks_physical_metadata(tmp_path: Path) -> None:
    slides = tmp_path / "slides"
    slides.mkdir()
    _touch_slides(slides, "case.svs")
    selection = _write_selection(tmp_path / "selection.csv", [("case.svs", 0.189012)])
    cfg = _config(
        slides,
        tmp_path / "job",
        selection,
        target_mpp=0.5,
        patch_encoder="smoke",
    )
    feature_path = Path(cfg.job_dir) / cfg.coords_subdir / "features_smoke" / "case.h5"
    _write_exact_feature(feature_path, source_mpp=0.189012)

    valid = extraction.validate_outputs(cfg, ["feat"])
    assert valid == {
        "total": 1,
        "found": 1,
        "missing": [],
        "invalid": [],
        "unexpected": [],
    }

    with h5py.File(feature_path, "r+") as handle:
        attrs = handle["coords"].attrs
        attrs["level0_mpp"] = 0.25
        attrs["patch_size_level0"] = 512
        attrs["read_patch_size"] = 512
        attrs["actual_read_level0_pixels"] = 512
        attrs["effective_target_mpp"] = 0.5
    invalid = extraction.validate_outputs(cfg, ["feat"])
    assert invalid["found"] == 0
    assert "level0_mpp mismatch" in invalid["invalid"][0]
