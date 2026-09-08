from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest
import torch
from shapely import Polygon, box

from oceanpath.extraction import mpp_sampling
from oceanpath.extraction.mpp_sampling import (
    COORDS_SCHEMA_VERSION,
    EXACT_MPP_SAMPLING_MODE,
    ExactMppSamplingError,
    ExactMppWSIAdapter,
    is_exact_mpp_metadata,
    read_coordinate_h5,
    validate_exact_mpp_coordinate_file,
    wrap_processor_wsis_for_exact_mpp,
)


class FakePatcher:
    def __init__(
        self,
        *,
        wsi: FakeWSI,
        patch_size: int,
        src_pixel_size: float,
        dst_pixel_size: float,
        overlap: int = 0,
        custom_coords: np.ndarray | None = None,
        empty: bool = False,
    ) -> None:
        self.wsi = wsi
        self.patch_size_target = patch_size
        self.patch_size_src = round(patch_size * dst_pixel_size / src_pixel_size)
        self.overlap_src = round(overlap * dst_pixel_size / src_pixel_size)
        self.level = wsi.force_level
        # Emulate pinned TRIDENT choosing a pyramid level. Exact mode must
        # replace this with a native level-0 read.
        floored_downsample = int(wsi.level_downsamples[self.level])
        self.patch_size_level = round(self.patch_size_src / floored_downsample)
        self.overlap_level = round(self.overlap_src / floored_downsample)
        if custom_coords is not None:
            self.valid_coords = np.asarray(custom_coords, dtype=np.int64)
        elif empty:
            self.valid_coords = np.empty((0, 2), dtype=np.int64)
        else:
            self.valid_coords = np.asarray(
                [[0, 0], [self.patch_size_src - self.overlap_src, 0]], dtype=np.int64
            )
        self.mask_simplify_shape: bool | None = None

    def _compute_masked(
        self,
        coords: np.ndarray,
        threshold: float,
        *,
        simplify_shape: bool,
    ) -> tuple[int, np.ndarray]:
        self.mask_simplify_shape = simplify_shape
        return len(coords), coords

    def __iter__(self):
        return iter(self.valid_coords)

    def __len__(self) -> int:
        return len(self.valid_coords)


class FakeWSI:
    def __init__(
        self,
        mpp: float,
        *,
        name: str = "sample",
        empty: bool = False,
        level_downsamples: list[float] | None = None,
        force_level: int = 0,
    ) -> None:
        self.mpp = mpp
        self.name = name
        self.width = 4096
        self.height = 2048
        self.gdf_contours = object()
        self.max_workers = 0
        self.empty = empty
        self.level_downsamples = level_downsamples or [1.0, 2.0, 4.0]
        self.force_level = force_level
        self.initialize_calls = 0
        self.create_calls: list[dict[str, Any]] = []
        self.patchers: list[FakePatcher] = []
        self.legacy_feature_calls: list[dict[str, Any]] = []

    def _lazy_initialize(self) -> None:
        self.initialize_calls += 1

    def create_patcher(self, **kwargs: Any) -> FakePatcher:
        self.create_calls.append(dict(kwargs))
        patcher = FakePatcher(
            wsi=self,
            patch_size=int(kwargs["patch_size"]),
            src_pixel_size=float(kwargs["src_pixel_size"]),
            dst_pixel_size=float(kwargs["dst_pixel_size"]),
            overlap=int(kwargs.get("overlap", 0)),
            custom_coords=kwargs.get("custom_coords"),
            empty=self.empty,
        )
        self.patchers.append(patcher)
        return patcher

    def extract_patch_features(self, **kwargs: Any) -> str:
        self.legacy_feature_calls.append(kwargs)
        return "legacy-output.h5"


class FakePatcherDataset(torch.utils.data.Dataset):
    def __init__(self, patcher: FakePatcher, transform: Any) -> None:
        self.patcher = patcher
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patcher)

    def __getitem__(self, index: int):
        image = torch.full((3, 4, 4), float(index + 1), dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image)
        x, y = self.patcher.valid_coords[index]
        return image, (int(x), int(y))


class FakeEncoder(torch.nn.Module):
    enc_name = "fake_encoder"
    precision = torch.float32
    eval_transforms = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(2, 3))


class BFloat16Encoder(FakeEncoder):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return super().forward(images).to(dtype=torch.bfloat16)


def test_resizable_h5_datasets_use_io_efficient_row_chunks(tmp_path: Path) -> None:
    path = tmp_path / "chunks.h5"
    with h5py.File(path, "w") as handle:
        coords = mpp_sampling._create_resizable_dataset(
            handle,
            "coords",
            np.zeros((10_000, 2), dtype=np.int64),
        )
        features = mpp_sampling._create_resizable_dataset(
            handle,
            "features",
            np.zeros((1_000, 1_024), dtype=np.float32),
        )

        assert coords.chunks == (8_192, 2)
        assert features.chunks == (256, 1_024)
        assert coords.maxshape == (None, 2)
        assert features.maxshape == (None, 1_024)


@pytest.mark.parametrize(
    ("source_mpp", "expected_level0_size"),
    [
        (0.13899, 921),
        (0.189012, 677),
        (0.378, 339),
        (0.5016, 255),
        (0.25, 512),
        (0.2325, 551),
    ],
)
def test_exact_mpp_coordinates_use_physical_pixel_sizes(
    tmp_path: Path, source_mpp: float, expected_level0_size: int
) -> None:
    wsi = FakeWSI(source_mpp)
    adapter = ExactMppWSIAdapter(wsi, target_mpp=0.5)

    output = adapter.extract_tissue_coords(
        target_mag=20,
        patch_size=256,
        save_coords=str(tmp_path),
        overlap=0,
    )

    call = wsi.create_calls[-1]
    assert call["src_pixel_size"] == pytest.approx(source_mpp)
    assert call["dst_pixel_size"] == pytest.approx(0.5)
    assert "src_mag" not in call
    assert "dst_mag" not in call
    assert call["mask"] is None
    assert wsi.patchers[-1].mask_simplify_shape is False

    attributes, coordinates = read_coordinate_h5(output)
    assert coordinates.shape == (2, 2)
    assert is_exact_mpp_metadata(attributes)
    assert attributes["coords_schema_version"] == COORDS_SCHEMA_VERSION
    assert attributes["sampling_mode"] == EXACT_MPP_SAMPLING_MODE
    assert attributes["patch_size_level0"] == expected_level0_size
    assert attributes["level0_mpp"] == pytest.approx(source_mpp)
    assert attributes["target_mpp"] == pytest.approx(0.5)
    assert attributes["read_level"] == 0
    assert attributes["read_level_downsample"] == pytest.approx(1.0)
    assert attributes["read_patch_size"] == expected_level0_size
    assert attributes["actual_read_level0_pixels"] == expected_level0_size
    assert attributes["mask_simplification"] == "none"
    assert 10.0 / attributes["level0_magnification"] == pytest.approx(source_mpp)
    assert 10.0 / attributes["target_magnification"] == pytest.approx(0.5)
    expected_effective = source_mpp * expected_level0_size / 256
    assert attributes["effective_target_mpp"] == pytest.approx(expected_effective)


def test_fast_unsimplified_mask_matches_exact_patch_area_and_preserves_order() -> None:
    candidates = np.asarray(
        [[0, 0], [10, 0], [20, 0], [0, 10], [10, 10], [20, 10]],
        dtype=np.int64,
    )
    # A 30x20 exterior with a 10x10 interior hole. At threshold 0.5 the
    # hole-only patch is excluded while every full-tissue patch remains.
    mask = Polygon(
        [(0, 0), (30, 0), (30, 20), (0, 20)],
        holes=[[(10, 1), (20, 1), (20, 9), (10, 9)]],
    )

    selected = mpp_sampling._filter_coordinates_by_mask(
        candidates,
        mask=mask,
        footprint=10,
        threshold=0.5,
        chunk_size=2,
    )

    assert selected is not None
    np.testing.assert_array_equal(
        selected,
        np.asarray([[0, 0], [20, 0], [0, 10], [10, 10], [20, 10]], dtype=np.int64),
    )


def test_fast_unsimplified_mask_keeps_edge_square_semantics() -> None:
    candidates = np.asarray([[0, 0], [10, 0], [20, 0]], dtype=np.int64)
    # The final 10x10 candidate extends beyond this geometry. Exactly 60% of
    # the full candidate square is tissue, so it remains at threshold 0.5.
    mask = box(0, 0, 26, 10)

    selected = mpp_sampling._filter_coordinates_by_mask(
        candidates,
        mask=mask,
        footprint=10,
        threshold=0.5,
    )

    assert selected is not None
    np.testing.assert_array_equal(selected, candidates)


def test_unknown_mask_type_uses_reference_patcher_path(tmp_path: Path) -> None:
    wsi = FakeWSI(0.25)
    adapter = ExactMppWSIAdapter(wsi, target_mpp=0.5)

    adapter.extract_tissue_coords(20, 256, str(tmp_path))

    assert wsi.patchers[-1].mask_simplify_shape is False


def test_empty_coordinate_set_is_rejected_before_cache_write(tmp_path: Path) -> None:
    adapter = ExactMppWSIAdapter(FakeWSI(0.25, empty=True), target_mpp=0.5)

    with pytest.raises(ExactMppSamplingError, match="No tissue coordinates"):
        adapter.extract_tissue_coords(20, 256, str(tmp_path))
    assert not (tmp_path / "patches" / "sample_patches.h5").exists()


@pytest.mark.parametrize("bad_mpp", [0.0, -0.5, float("nan"), float("inf"), None])
def test_invalid_target_mpp_is_rejected(bad_mpp: Any) -> None:
    with pytest.raises(ExactMppSamplingError, match="target_mpp"):
        ExactMppWSIAdapter(FakeWSI(0.25), target_mpp=bad_mpp)


def test_conflicting_target_magnification_is_rejected(tmp_path: Path) -> None:
    adapter = ExactMppWSIAdapter(FakeWSI(0.25), target_mpp=0.5)
    lock_path = tmp_path / "patches" / "sample_patches.h5.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.touch()

    with pytest.raises(ExactMppSamplingError, match="conflicts"):
        adapter.extract_tissue_coords(40, 256, str(tmp_path))
    assert not lock_path.exists()


def test_keyboard_interrupt_cleans_coordinate_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = ExactMppWSIAdapter(FakeWSI(0.25), target_mpp=0.5)
    lock_path = tmp_path / "patches" / "sample_patches.h5.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.touch()

    def interrupt(*_args: Any, **_kwargs: Any) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(adapter, "_extract_tissue_coords", interrupt)
    with pytest.raises(KeyboardInterrupt):
        adapter.extract_tissue_coords(20, 256, str(tmp_path))
    assert not lock_path.exists()


def test_exact_mode_forces_native_level_read(tmp_path: Path) -> None:
    wsi = FakeWSI(
        0.125,
        level_downsamples=[1.0, 3.999],
        force_level=1,
    )
    adapter = ExactMppWSIAdapter(wsi, target_mpp=0.5)

    output = adapter.extract_tissue_coords(20, 256, str(tmp_path))

    attributes, _ = read_coordinate_h5(output)
    assert attributes["patch_size_level0"] == 1024
    assert attributes["read_level"] == 0
    assert attributes["read_patch_size"] == 1024
    assert attributes["read_level_downsample"] == pytest.approx(1.0)
    assert attributes["actual_read_level0_pixels"] == 1024
    assert wsi.patchers[-1].level == 0
    assert wsi.patchers[-1].patch_size_level == 1024


def test_exact_coordinate_cache_validation_detects_conflicts(tmp_path: Path) -> None:
    adapter = ExactMppWSIAdapter(FakeWSI(0.25), target_mpp=0.5)
    output = adapter.extract_tissue_coords(
        20,
        256,
        str(tmp_path),
        overlap=32,
        min_tissue_proportion=0.2,
    )

    attributes = validate_exact_mpp_coordinate_file(
        output,
        target_mpp=0.5,
        patch_size=256,
        overlap=32,
        min_tissue_proportion=0.2,
    )
    assert attributes["target_mpp"] == pytest.approx(0.5)

    with pytest.raises(ExactMppSamplingError, match="target_mpp mismatch"):
        validate_exact_mpp_coordinate_file(output, target_mpp=0.25)
    with pytest.raises(ExactMppSamplingError, match="overlap mismatch"):
        validate_exact_mpp_coordinate_file(output, target_mpp=0.5, overlap=0)


def test_exact_coordinate_cache_validation_detects_source_mpp_change(tmp_path: Path) -> None:
    adapter = ExactMppWSIAdapter(FakeWSI(0.25), target_mpp=0.5)
    output = adapter.extract_tissue_coords(20, 256, str(tmp_path))

    validate_exact_mpp_coordinate_file(output, target_mpp=0.5, source_mpp=0.25)
    with pytest.raises(ExactMppSamplingError, match="level0_mpp mismatch"):
        validate_exact_mpp_coordinate_file(output, target_mpp=0.5, source_mpp=0.2501)


def test_exact_coordinate_cache_rejects_non_native_read_metadata(tmp_path: Path) -> None:
    adapter = ExactMppWSIAdapter(FakeWSI(0.125), target_mpp=0.5)
    output = Path(adapter.extract_tissue_coords(20, 256, str(tmp_path)))
    with h5py.File(output, "r+") as handle:
        handle["coords"].attrs["read_level"] = 1
        handle["coords"].attrs["read_level_downsample"] = 4.0
        handle["coords"].attrs["read_patch_size"] = 256

    with pytest.raises(ExactMppSamplingError, match="native level-0 read footprint"):
        validate_exact_mpp_coordinate_file(output, target_mpp=0.5)


def test_exact_coordinate_cache_rejects_self_consistent_wrong_footprint(tmp_path: Path) -> None:
    adapter = ExactMppWSIAdapter(FakeWSI(0.25), target_mpp=0.5)
    output = Path(adapter.extract_tissue_coords(20, 256, str(tmp_path)))
    with h5py.File(output, "r+") as handle:
        attrs = handle["coords"].attrs
        attrs["patch_size_level0"] = 513
        attrs["read_patch_size"] = 513
        attrs["actual_read_level0_pixels"] = 513
        attrs["effective_target_mpp"] = 0.25 * 513 / 256

    with pytest.raises(ExactMppSamplingError, match="native level-0 read footprint"):
        validate_exact_mpp_coordinate_file(output, target_mpp=0.5)


def _write_legacy_coords(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("coords", data=np.asarray([[0, 0]], dtype=np.int64))
        dataset.attrs["patch_size"] = 256
        dataset.attrs["level0_magnification"] = 40
        dataset.attrs["target_magnification"] = 20


def test_exact_mode_rejects_legacy_coordinate_replay(tmp_path: Path) -> None:
    coordinate_path = tmp_path / "legacy.h5"
    _write_legacy_coords(coordinate_path)
    adapter = ExactMppWSIAdapter(FakeWSI(0.25), target_mpp=0.5)
    lock_path = tmp_path / "sample.h5.lock"
    lock_path.touch()

    with pytest.raises(ExactMppSamplingError, match="Legacy coordinate"):
        adapter.extract_patch_features(
            FakeEncoder(), str(coordinate_path), str(tmp_path), device="cpu"
        )
    assert not lock_path.exists()

    with pytest.raises(ExactMppSamplingError, match="Legacy or unknown"):
        validate_exact_mpp_coordinate_file(coordinate_path, target_mpp=0.5)


def test_legacy_replay_can_be_explicitly_delegated(tmp_path: Path) -> None:
    coordinate_path = tmp_path / "legacy.h5"
    _write_legacy_coords(coordinate_path)
    wsi = FakeWSI(0.25)
    adapter = ExactMppWSIAdapter(wsi, target_mpp=0.5, allow_legacy_replay=True)

    output = adapter.extract_patch_features(
        FakeEncoder(), str(coordinate_path), str(tmp_path), device="cpu"
    )

    assert output == "legacy-output.h5"
    assert len(wsi.legacy_feature_calls) == 1


def test_exact_feature_replay_uses_stored_mpp_and_preserves_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wsi = FakeWSI(0.378)
    adapter = ExactMppWSIAdapter(wsi, target_mpp=0.5)
    coordinate_path = adapter.extract_tissue_coords(20, 256, str(tmp_path / "coords"))
    monkeypatch.setattr(
        mpp_sampling,
        "_get_wsi_patcher_dataset_class",
        lambda: FakePatcherDataset,
    )

    output = adapter.extract_patch_features(
        FakeEncoder(),
        coordinate_path,
        str(tmp_path / "features"),
        device="cpu",
        batch_limit=2,
    )

    replay_call = wsi.create_calls[-1]
    assert replay_call["src_pixel_size"] == pytest.approx(0.378)
    assert replay_call["dst_pixel_size"] == pytest.approx(0.5)
    assert np.array_equal(replay_call["custom_coords"], np.asarray([[0, 0], [339, 0]]))
    assert wsi.patchers[-1].level == 0
    assert wsi.patchers[-1].patch_size_level == 339
    with h5py.File(output, "r") as handle:
        assert handle["features"].shape == (2, 3)
        assert np.allclose(handle["features"][:], [[1, 1, 1], [2, 2, 2]])
        assert handle["features"].attrs["encoder"] == "fake_encoder"
        assert handle["coords"].attrs["sampling_mode"] == EXACT_MPP_SAMPLING_MODE
        assert handle["coords"].attrs["target_mpp"] == pytest.approx(0.5)


def test_bfloat16_encoder_outputs_are_serialized_as_float32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wsi = FakeWSI(0.25)
    adapter = ExactMppWSIAdapter(wsi, target_mpp=0.5)
    coordinate_path = adapter.extract_tissue_coords(20, 256, str(tmp_path / "coords"))
    monkeypatch.setattr(
        mpp_sampling,
        "_get_wsi_patcher_dataset_class",
        lambda: FakePatcherDataset,
    )

    output = adapter.extract_patch_features(
        BFloat16Encoder(),
        coordinate_path,
        str(tmp_path / "features"),
        device="cpu",
        batch_limit=2,
    )

    with h5py.File(output, "r") as handle:
        assert handle["features"].dtype == np.dtype("float32")


def test_feature_replay_detects_changed_source_mpp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wsi = FakeWSI(0.25)
    adapter = ExactMppWSIAdapter(wsi, target_mpp=0.5)
    coordinate_path = adapter.extract_tissue_coords(20, 256, str(tmp_path / "coords"))
    wsi.mpp = 0.5
    monkeypatch.setattr(
        mpp_sampling,
        "_get_wsi_patcher_dataset_class",
        lambda: FakePatcherDataset,
    )

    with pytest.raises(ExactMppSamplingError, match="Slide MPP changed"):
        adapter.extract_patch_features(
            FakeEncoder(), coordinate_path, str(tmp_path / "features"), device="cpu"
        )


def test_processor_wsi_wrapping_is_in_place() -> None:
    first = FakeWSI(0.25, name="first")
    second = FakeWSI(0.5, name="second")
    processor = SimpleNamespace(wsis=[first, second])

    returned = wrap_processor_wsis_for_exact_mpp(processor, 0.5)

    assert returned is processor
    assert [item.name for item in processor.wsis] == ["first", "second"]
    assert all(isinstance(item, ExactMppWSIAdapter) for item in processor.wsis)
    assert processor.wsis[0].wrapped_wsi is first


def test_processor_without_wsis_is_rejected() -> None:
    with pytest.raises(ExactMppSamplingError, match="no 'wsis'"):
        wrap_processor_wsis_for_exact_mpp(SimpleNamespace(), 0.5)
