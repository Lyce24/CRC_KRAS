from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from oceanpath.extraction.stream_encoder import (
    SlideEncoder,
    SlideEncoderConfig,
    SlideEncodingError,
    SourceChangedError,
    SourceSnapshot,
)


@dataclass(frozen=True)
class FakeSlide:
    wsi: str
    path: Path
    output_id: str
    mpp: float
    cohort: str = "test"


def _config(tmp_path: Path) -> SlideEncoderConfig:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    paths = []
    for name in ("hest.ckpt", "uni.bin", "conch.bin", "virchow2.bin"):
        path = checkpoints / name
        path.write_bytes(name.encode())
        paths.append(path)
    return SlideEncoderConfig(
        output_root=tmp_path / "features",
        state_root=tmp_path / "state",
        scratch_root=tmp_path / "scratch",
        hest_checkpoint_path=paths[0],
        uni_checkpoint_path=paths[1],
        conch_v15_checkpoint_path=paths[2],
        virchow2_checkpoint_path=paths[3],
    )


def _slide(tmp_path: Path, *, mpp: float = 0.25) -> FakeSlide:
    source = tmp_path / "case.svs"
    source.write_bytes(b"slide")
    return FakeSlide(wsi="rih/case.svs", path=source, output_id="case", mpp=mpp)


def _write_coordinates(
    encoder: SlideEncoder,
    snapshot: SourceSnapshot,
    *,
    encoder_index: int,
) -> np.ndarray:
    spec = encoder.cfg.encoders[encoder_index]
    native = round(spec.patch_size * encoder.cfg.target_mpp / snapshot.mpp)
    coordinates = np.asarray([[0, 0], [native, 0]], dtype=np.int64)
    path = encoder._coordinate_path(snapshot.output_id, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("coords", data=coordinates)
        dataset.attrs.update(
            {
                "coords_schema_version": 2,
                "sampling_mode": "exact_mpp",
                "level0_mpp": snapshot.mpp,
                "target_mpp": encoder.cfg.target_mpp,
                "patch_size": spec.patch_size,
                "patch_size_level0": native,
                "overlap": 0,
                "overlap_level0": 0,
                "min_tissue_proportion": 0.5,
                "effective_target_mpp": snapshot.mpp * native / spec.patch_size,
                "read_level": 0,
                "read_level_downsample": 1.0,
                "read_patch_size": native,
                "actual_read_level0_pixels": native,
                "mask_simplification": "none",
            }
        )
    visualization = encoder.cfg.output_root / spec.coords_dir / "visualization" / "case.jpg"
    visualization.parent.mkdir(parents=True, exist_ok=True)
    visualization.write_bytes(b"viz")
    return coordinates


def _write_features(
    encoder: SlideEncoder,
    snapshot: SourceSnapshot,
    coordinates: np.ndarray,
    *,
    encoder_index: int,
    feature_dim: int | None = None,
) -> Path:
    spec = encoder.cfg.encoders[encoder_index]
    path = encoder._feature_path(snapshot.output_id, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        features = handle.create_dataset(
            "features",
            data=np.ones((len(coordinates), feature_dim or spec.feature_dim), dtype=np.float32),
        )
        features.attrs["encoder"] = spec.name
        handle.create_dataset("coords", data=coordinates)
    return path


def test_native_colon_profile_tracks_encoder_patch_size() -> None:
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        uni = compose(
            config_name="extract",
            overrides=["data=colon", "extraction=colon_native", "encoder=univ1"],
        )
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        conch = compose(
            config_name="extract",
            overrides=["data=colon", "extraction=colon_native", "encoder=conch_v15"],
        )
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        virchow2 = compose(
            config_name="extract",
            overrides=["data=colon", "extraction=colon_native", "encoder=virchow2"],
        )

    assert uni.extraction.patch_size == 256
    assert conch.extraction.patch_size == 512
    assert virchow2.extraction.patch_size == 224
    assert (
        uni.extraction.target_mpp
        == conch.extraction.target_mpp
        == virchow2.extraction.target_mpp
        == 0.5
    )
    assert (
        uni.extraction.min_tissue_proportion
        == conch.extraction.min_tissue_proportion
        == virchow2.extraction.min_tissue_proportion
        == 0.5
    )
    assert (
        uni.extraction.reader_type
        == conch.extraction.reader_type
        == virchow2.extraction.reader_type
        == "openslide"
    )
    assert (
        uni.extraction.remove_holes
        is conch.extraction.remove_holes
        is virchow2.extraction.remove_holes
        is True
    )
    assert (
        uni.extraction.remove_artifacts
        is conch.extraction.remove_artifacts
        is virchow2.extraction.remove_artifacts
        is False
    )
    assert (
        uni.extraction.remove_penmarks
        is conch.extraction.remove_penmarks
        is virchow2.extraction.remove_penmarks
        is False
    )
    assert uni.extraction.feat_batch_size == 64
    assert conch.extraction.feat_batch_size == 32
    assert virchow2.extraction.feat_batch_size == 8


def test_stage_keys_share_hest_but_separate_native_geometries(tmp_path: Path) -> None:
    encoder = SlideEncoder(_config(tmp_path))
    snapshot = SourceSnapshot.capture(_slide(tmp_path, mpp=0.189012))

    keys = encoder.stage_keys(snapshot)

    assert set(keys) == {
        "seg",
        "uni_v1_coords",
        "uni_v1_feat",
        "conch_v15_coords",
        "conch_v15_feat",
        "virchow2_coords",
        "virchow2_feat",
    }
    assert keys["uni_v1_coords"] != keys["conch_v15_coords"]
    assert encoder.cfg.encoders[0].coords_dir != encoder.cfg.encoders[1].coords_dir
    assert encoder.cfg.encoders[2].coords_dir not in {
        encoder.cfg.encoders[0].coords_dir,
        encoder.cfg.encoders[1].coords_dir,
    }


def test_hole_policy_invalidates_segmentation_and_both_feature_lineages(
    tmp_path: Path,
) -> None:
    preserved = SlideEncoder(_config(tmp_path))
    filled = SlideEncoder(replace(preserved.cfg, remove_holes=False))
    snapshot = SourceSnapshot.capture(_slide(tmp_path, mpp=0.189012))

    preserved_keys = preserved.stage_keys(snapshot)
    filled_keys = filled.stage_keys(snapshot)

    assert preserved.segmentation_policy == {
        "segmenter": "hest",
        "confidence": 0.5,
        "target_mag": 10,
        "reader": "openslide",
        "remove_holes": True,
        "holes_are_tissue": False,
        "remove_artifacts": False,
        "remove_penmarks": False,
        "artifact_remover_model": None,
    }
    assert filled.segmentation_policy["holes_are_tissue"] is True
    assert all(preserved_keys[stage] != filled_keys[stage] for stage in preserved_keys)
    assert preserved.config_key(snapshot) != filled.config_key(snapshot)


def test_segmentation_call_and_receipt_use_the_fingerprinted_policy(tmp_path: Path) -> None:
    encoder = SlideEncoder(_config(tmp_path))
    snapshot = SourceSnapshot.capture(_slide(tmp_path))
    hest = SimpleNamespace(target_mag=10)
    encoder._models["hest"] = hest

    class Processor:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def run_segmentation_job(self, *args: object, **kwargs: object) -> None:
            self.calls.append((args, kwargs))

    processor = Processor()
    encoder._run_segmentation_job(processor, batch_size=23)

    assert processor.calls == [
        (
            (hest,),
            {
                "seg_mag": 10,
                "holes_are_tissue": False,
                "artifact_remover_model": None,
                "batch_size": 23,
                "device": "cuda:0",
            },
        )
    ]

    keys = encoder.stage_keys(snapshot)
    encoder._commit_stage(snapshot, "seg", keys["seg"], {"polygon_count": 1})
    receipt = json.loads(encoder._receipt_path("case", "seg").read_text(encoding="utf-8"))

    assert receipt["stage_key"] == keys["seg"]
    assert receipt["segmentation_policy"] == encoder.segmentation_policy


def test_exact_geometry_and_feature_validation(tmp_path: Path) -> None:
    encoder = SlideEncoder(_config(tmp_path))
    snapshot = SourceSnapshot.capture(_slide(tmp_path, mpp=0.189012))
    coordinates = _write_coordinates(encoder, snapshot, encoder_index=0)
    _write_features(encoder, snapshot, coordinates, encoder_index=0)

    geometry = encoder._validate_coordinates(snapshot, encoder.cfg.encoders[0])
    features = encoder._validate_features(snapshot, encoder.cfg.encoders[0])

    assert geometry["native_pixels"] == round(256 * 0.5 / 0.189012)
    assert geometry["effective_mpp"] == pytest.approx(0.5, abs=0.001)
    assert features == {"patch_count": 2, "feature_dim": 1024}


def test_feature_validation_rejects_wrong_dimension_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    encoder = SlideEncoder(_config(tmp_path))
    snapshot = SourceSnapshot.capture(_slide(tmp_path))
    coordinates = _write_coordinates(encoder, snapshot, encoder_index=1)
    path = _write_features(
        encoder,
        snapshot,
        coordinates,
        encoder_index=1,
        feature_dim=7,
    )
    with pytest.raises(SlideEncodingError, match="Wrong conch_v15 feature shape"):
        encoder._validate_features(snapshot, encoder.cfg.encoders[1])

    path.unlink()
    path = _write_features(encoder, snapshot, coordinates, encoder_index=1)
    with h5py.File(path, "r+") as handle:
        handle["features"][0, 0] = np.nan
    with pytest.raises(SlideEncodingError, match="Non-finite embeddings"):
        encoder._validate_features(snapshot, encoder.cfg.encoders[1])


def test_source_snapshot_detects_replacement(tmp_path: Path) -> None:
    slide = _slide(tmp_path)
    snapshot = SourceSnapshot.capture(slide)
    slide.path.write_bytes(b"replacement")

    with pytest.raises(SourceChangedError, match="Source changed"):
        snapshot.assert_unchanged()


def test_source_key_ignores_wsl_remount_metadata(tmp_path: Path) -> None:
    snapshot = SourceSnapshot.capture(_slide(tmp_path))
    remounted = replace(
        snapshot,
        ctime_ns=snapshot.ctime_ns + 1,
        device=snapshot.device + 1,
        inode=snapshot.inode + 1,
    )

    assert remounted.key == snapshot.key
    assert replace(snapshot, mtime_ns=snapshot.mtime_ns + 1).key != snapshot.key


def test_staged_source_is_ephemeral_and_does_not_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = SlideEncoder(replace(_config(tmp_path), scratch_reserve_gib=0))
    slide = _slide(tmp_path)
    snapshot = SourceSnapshot.capture(slide)

    def unexpected_fsync(_descriptor: int) -> None:
        pytest.fail("ephemeral slide staging must not force durable storage")

    monkeypatch.setattr("oceanpath.extraction.stream_encoder.os.fsync", unexpected_fsync)
    with encoder._staged_source(snapshot) as staged:
        assert staged != slide.path
        assert staged.read_bytes() == slide.path.read_bytes()
        staged_path = staged

    assert slide.path.is_file()
    assert not staged_path.exists()


def test_crash_after_stage_receipts_repairs_completion_without_loading_models(
    tmp_path: Path,
) -> None:
    encoder = SlideEncoder(_config(tmp_path))
    slide = _slide(tmp_path)
    snapshot = SourceSnapshot.capture(slide)
    keys = encoder.stage_keys(snapshot)
    for directory, suffix, contents in (
        ("contours", ".jpg", b"preview"),
        ("thumbnails", ".jpg", b"thumbnail"),
    ):
        path = encoder.cfg.output_root / directory / f"case{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    geojson = encoder.cfg.output_root / "contours_geojson" / "case.geojson"
    geojson.parent.mkdir(parents=True, exist_ok=True)
    geojson.write_text(json.dumps({"features": [{"type": "Feature"}]}), encoding="utf-8")
    encoder._commit_stage(snapshot, "seg", keys["seg"], {"polygon_count": 1})
    for index, spec in enumerate(encoder.cfg.encoders):
        coordinates = _write_coordinates(encoder, snapshot, encoder_index=index)
        _write_features(encoder, snapshot, coordinates, encoder_index=index)
        encoder._commit_stage(
            snapshot,
            f"{spec.name}_coords",
            keys[f"{spec.name}_coords"],
            {"patch_count": 2},
        )
        encoder._commit_stage(
            snapshot,
            f"{spec.name}_feat",
            keys[f"{spec.name}_feat"],
            {"patch_count": 2},
        )

    result = encoder.process(slide)

    assert result.stage_metrics == {"cache": {"reused": True}}
    assert encoder._completion_path("case").is_file()
    assert encoder.validate_complete(slide)


def test_complete_boundary_returns_false_for_corrupt_h5(tmp_path: Path) -> None:
    encoder = SlideEncoder(_config(tmp_path))
    slide = _slide(tmp_path)
    snapshot = SourceSnapshot.capture(slide)
    keys = encoder.stage_keys(snapshot)
    completion = encoder._completion_path("case")
    completion.parent.mkdir(parents=True, exist_ok=True)
    completion.write_text(
        json.dumps({"source_key": snapshot.key, "config_key": encoder.config_key(snapshot)}),
        encoding="utf-8",
    )
    for stage, key in keys.items():
        receipt = encoder._receipt_path("case", stage)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(
            json.dumps({"source_key": snapshot.key, "stage_key": key}), encoding="utf-8"
        )
    corrupt = encoder._coordinate_path("case", encoder.cfg.encoders[0])
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"not hdf5")

    assert encoder.validate_complete(slide) is False


def test_startup_archives_orphan_locks_and_atomic_temporaries(tmp_path: Path) -> None:
    encoder = SlideEncoder(_config(tmp_path))
    locked = encoder.cfg.output_root / "contours" / "case.jpg.lock"
    temporary = encoder.cfg.output_root / "coords" / ".case.h5.deadbeef.tmp"
    locked.parent.mkdir(parents=True, exist_ok=True)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    locked.touch()
    temporary.write_bytes(b"partial")

    assert encoder.recover_orphan_work_files() == 2
    assert not locked.exists()
    assert not temporary.exists()
    assert len(list((encoder.cfg.output_root / ".orphan_work").rglob("*.*"))) >= 2
