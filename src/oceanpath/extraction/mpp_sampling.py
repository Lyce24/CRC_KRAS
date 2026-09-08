"""Exact-MPP sampling compatibility for the pinned TRIDENT release.

TRIDENT 0.2.3 accepts an exact source MPP, but its public coordinate path
first reduces that value to a nominal magnification bucket.  This module keeps
the dependency unmodified and adapts only the two operations for which the
physical sampling geometry matters: coordinate generation and patch replay.

Imports from TRIDENT remain lazy so importing :mod:`oceanpath.extraction` does
not require the optional extraction dependency group.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

COORDS_SCHEMA_VERSION = 2
EXACT_MPP_SAMPLING_MODE = "exact_mpp"
_MPP_ABS_TOL = 1e-6
_MPP_REL_TOL = 1e-6
_SOURCE_MPP_REL_TOL = 1e-6
_H5_TARGET_CHUNK_BYTES = 1024 * 1024
_H5_MAX_CHUNK_ROWS = 8192


class ExactMppSamplingError(ValueError):
    """Raised when exact physical sampling cannot be guaranteed."""


def _positive_finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExactMppSamplingError(
            f"{field} must be a finite positive number, got {value!r}"
        ) from exc
    if not math.isfinite(result) or result <= 0:
        raise ExactMppSamplingError(f"{field} must be a finite positive number, got {value!r}")
    return result


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _is_close(first: float, second: float, *, source_mpp: bool = False) -> bool:
    return math.isclose(
        first,
        second,
        rel_tol=_SOURCE_MPP_REL_TOL if source_mpp else _MPP_REL_TOL,
        abs_tol=_MPP_ABS_TOL,
    )


def is_exact_mpp_metadata(attributes: Mapping[str, Any]) -> bool:
    """Return whether coordinate attributes declare the exact-MPP schema."""

    mode = attributes.get("sampling_mode")
    if mode is None or _as_text(mode) != EXACT_MPP_SAMPLING_MODE:
        return False
    return "level0_mpp" in attributes and "target_mpp" in attributes


def _validate_exact_schema_attributes(attributes: Mapping[str, Any], coordinate_path: Path) -> None:
    required = {
        "coords_schema_version",
        "patch_size",
        "patch_size_level0",
        "read_level",
        "read_level_downsample",
        "read_patch_size",
        "actual_read_level0_pixels",
        "effective_target_mpp",
        "mask_simplification",
    }
    missing = sorted(required.difference(attributes))
    if missing:
        raise ExactMppSamplingError(
            f"Exact-MPP coordinate H5 is missing {missing} in {coordinate_path}"
        )

    schema_version = int(attributes["coords_schema_version"])
    if schema_version != COORDS_SCHEMA_VERSION:
        raise ExactMppSamplingError(
            f"Unsupported exact-MPP coordinate schema in {coordinate_path}: "
            f"stored={schema_version}, supported={COORDS_SCHEMA_VERSION}"
        )

    patch_size_level0 = int(attributes["patch_size_level0"])
    patch_size = int(attributes["patch_size"])
    source_mpp = _positive_finite_float(attributes["level0_mpp"], field="stored level0_mpp")
    target_mpp = _positive_finite_float(attributes["target_mpp"], field="stored target_mpp")
    expected_level0 = round(patch_size * target_mpp / source_mpp)
    effective_target_mpp = _positive_finite_float(
        attributes["effective_target_mpp"], field="stored effective_target_mpp"
    )
    expected_effective_mpp = source_mpp * patch_size_level0 / patch_size
    read_level = int(attributes["read_level"])
    read_downsample = _positive_finite_float(
        attributes["read_level_downsample"], field="stored read_level_downsample"
    )
    read_patch_size = int(attributes["read_patch_size"])
    actual_level0_pixels = int(attributes["actual_read_level0_pixels"])
    mask_simplification = _as_text(attributes["mask_simplification"])
    if (
        patch_size_level0 <= 0
        or read_level != 0
        or not _is_close(read_downsample, 1.0)
        or read_patch_size != patch_size_level0
        or actual_level0_pixels != patch_size_level0
        or patch_size_level0 != expected_level0
        or not _is_close(effective_target_mpp, expected_effective_mpp)
        or mask_simplification != "none"
    ):
        raise ExactMppSamplingError(
            f"Exact-MPP coordinates in {coordinate_path} do not guarantee a native level-0 "
            f"read footprint of {patch_size_level0} pixels"
        )


def read_coordinate_h5(path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    """Read TRIDENT-compatible coordinate attributes and level-0 coordinates."""

    coordinate_path = Path(path)
    with h5py.File(coordinate_path, "r") as handle:
        if "coords" not in handle:
            raise ExactMppSamplingError(f"Coordinate H5 has no 'coords' dataset: {coordinate_path}")
        dataset = handle["coords"]
        attributes = dict(dataset.attrs)
        coordinates = np.asarray(dataset[:], dtype=np.int64)

    if coordinates.size == 0:
        coordinates = np.empty((0, 2), dtype=np.int64)
    elif coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ExactMppSamplingError(
            f"Coordinates must have shape (N, 2), got {coordinates.shape} in {coordinate_path}"
        )
    return attributes, coordinates


def validate_exact_mpp_coordinate_file(
    path: str | Path,
    *,
    target_mpp: float,
    source_mpp: float | None = None,
    patch_size: int | None = None,
    overlap: int | None = None,
    min_tissue_proportion: float | None = None,
) -> dict[str, Any]:
    """Validate an existing coordinate file before exact-MPP cache reuse."""

    coordinate_path = Path(path)
    attributes, _ = read_coordinate_h5(coordinate_path)
    if not is_exact_mpp_metadata(attributes):
        raise ExactMppSamplingError(
            f"Legacy or unknown coordinate sampling metadata in {coordinate_path}; "
            "exact-MPP mode requires newly generated coordinates"
        )
    _validate_exact_schema_attributes(attributes, coordinate_path)

    stored_target = _positive_finite_float(attributes["target_mpp"], field="stored target_mpp")
    expected_target = _positive_finite_float(target_mpp, field="target_mpp")
    if not _is_close(stored_target, expected_target):
        raise ExactMppSamplingError(
            f"Coordinate target_mpp mismatch in {coordinate_path}: "
            f"stored={stored_target}, requested={expected_target}"
        )

    if source_mpp is not None:
        stored_source = _positive_finite_float(attributes["level0_mpp"], field="stored level0_mpp")
        expected_source = _positive_finite_float(source_mpp, field="source_mpp")
        if not _is_close(stored_source, expected_source, source_mpp=True):
            raise ExactMppSamplingError(
                f"Coordinate level0_mpp mismatch in {coordinate_path}: "
                f"stored={stored_source}, requested={expected_source}"
            )

    expected_values = {
        "patch_size": patch_size,
        "overlap": overlap,
        "min_tissue_proportion": min_tissue_proportion,
    }
    for key, expected in expected_values.items():
        if expected is None:
            continue
        if key not in attributes:
            raise ExactMppSamplingError(f"Coordinate H5 is missing '{key}': {coordinate_path}")
        stored = float(attributes[key])
        if not math.isclose(stored, float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise ExactMppSamplingError(
                f"Coordinate {key} mismatch in {coordinate_path}: stored={stored}, requested={expected}"
            )
    return attributes


@contextmanager
def _atomic_output(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_attributes(dataset: h5py.Dataset, attributes: Mapping[str, Any]) -> None:
    for key, value in attributes.items():
        if value is None:
            value = "None"
        elif isinstance(value, (dict, list, tuple)):
            value = json.dumps(value)
        dataset.attrs[key] = value


def _create_resizable_dataset(handle: h5py.File, name: str, values: np.ndarray) -> h5py.Dataset:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    row_elements = math.prod(array.shape[1:]) if array.ndim > 1 else 1
    row_bytes = max(1, row_elements * array.dtype.itemsize)
    chunk_rows = max(
        1,
        min(
            max(1, array.shape[0]),
            _H5_MAX_CHUNK_ROWS,
            max(1, _H5_TARGET_CHUNK_BYTES // row_bytes),
        ),
    )
    chunk_shape = (chunk_rows, *array.shape[1:])
    max_shape = (None, *array.shape[1:])
    dataset = handle.create_dataset(
        name,
        shape=array.shape,
        maxshape=max_shape,
        chunks=chunk_shape,
        dtype=array.dtype,
    )
    dataset[...] = array
    return dataset


def _force_level0_read_geometry(patcher: Any) -> Any:
    """Read the recorded level-0 footprint without pyramid resampling."""

    try:
        level_downsample = _positive_finite_float(
            patcher.wsi.level_downsamples[0], field="level_downsamples[0]"
        )
    except (AttributeError, IndexError) as exc:
        raise ExactMppSamplingError("Cannot resolve the level-0 pyramid downsample") from exc

    if not _is_close(level_downsample, 1.0):
        raise ExactMppSamplingError(
            f"Level 0 must represent native pixels, got downsample={level_downsample}"
        )

    patcher.level = 0
    patcher.patch_size_level = max(1, int(patcher.patch_size_src))
    patcher.overlap_level = max(0, int(patcher.overlap_src))
    return patcher


def _coordinate_array(patcher: Any) -> np.ndarray:
    coordinates = np.asarray(list(patcher), dtype=np.int64)
    if coordinates.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ExactMppSamplingError(
            f"Patcher yielded coordinates with shape {coordinates.shape}; expected (N, 2)"
        )
    return coordinates


def _polygonal_mask_geometry(mask: Any) -> Any | None:
    """Return one valid polygonal Shapely geometry when ``mask`` supports it.

    TRIDENT's mask filter intersects every candidate patch with the complete
    GeoDataFrame geometry.  On complex whole-slide contours that becomes
    quadratic enough to leave the GPU idle for minutes.  Shapely 2 can perform
    the identical area test efficiently when the geometry is partitioned and
    indexed.  ``None`` deliberately means "unsupported mask type" so test
    doubles and older third-party objects retain TRIDENT's reference path.
    """

    try:
        from shapely import GeometryCollection, MultiPolygon, Polygon, make_valid, union_all
        from shapely.geometry.base import BaseGeometry
    except ImportError:
        return None

    if isinstance(mask, BaseGeometry):
        values = [mask]
    elif hasattr(mask, "geometry"):
        values = [
            geometry for geometry in mask.geometry if geometry is not None and not geometry.is_empty
        ]
    else:
        return None
    if not values:
        return GeometryCollection()

    def polygonal(value: Any) -> Any:
        if value is None or value.is_empty:
            return GeometryCollection()
        valid = make_valid(value)
        if isinstance(valid, (Polygon, MultiPolygon)):
            return valid
        if isinstance(valid, GeometryCollection):
            parts = [polygonal(part) for part in valid.geoms]
            parts = [part for part in parts if not part.is_empty]
            return polygonal(union_all(parts)) if parts else GeometryCollection()
        return GeometryCollection()

    return polygonal(union_all(values))


def _partition_mask_geometry(geometry: Any, *, cell_size: int = 4096) -> np.ndarray:
    """Clip a WSI mask into disjoint pieces for exact indexed area queries."""

    from shapely import area, box, intersection, is_empty

    if cell_size <= 0:
        raise ExactMppSamplingError(f"mask partition cell_size must be positive: {cell_size}")
    if geometry.is_empty:
        return np.empty(0, dtype=object)
    min_x, min_y, max_x, max_y = geometry.bounds
    start_x = math.floor(min_x / cell_size) * cell_size
    start_y = math.floor(min_y / cell_size) * cell_size
    x_values = np.arange(start_x, max_x + cell_size, cell_size, dtype=np.float64)
    y_values = np.arange(start_y, max_y + cell_size, cell_size, dtype=np.float64)
    repeated_x = np.repeat(x_values, len(y_values))
    tiled_y = np.tile(y_values, len(x_values))
    cells = box(
        repeated_x,
        tiled_y,
        repeated_x + cell_size,
        tiled_y + cell_size,
    )
    pieces = intersection(cells, geometry)
    keep = (~is_empty(pieces)) & (area(pieces) > 0)
    return np.asarray(pieces[keep], dtype=object)


def _filter_coordinates_by_mask(
    candidates: np.ndarray,
    *,
    mask: Any,
    footprint: int,
    threshold: float,
    chunk_size: int = 8192,
) -> np.ndarray | None:
    """Apply TRIDENT's unsimplified patch-area rule using an exact STRtree.

    The returned coordinates preserve the input order.  Edge patches retain
    TRIDENT's behavior: their full square is tested even when it extends past
    the slide bounds.  ``None`` requests the reference implementation because
    the supplied mask is not a supported Shapely/GeoPandas object.
    """

    geometry = _polygonal_mask_geometry(mask)
    if geometry is None:
        return None
    points = np.asarray(candidates, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ExactMppSamplingError(
            f"Candidate coordinates must have shape (N, 2), got {points.shape}"
        )
    if footprint <= 0 or chunk_size <= 0:
        raise ExactMppSamplingError("mask footprint and chunk_size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ExactMppSamplingError(f"mask threshold must be in [0, 1], got {threshold}")
    if points.shape[0] == 0 or geometry.is_empty:
        return np.empty((0, 2), dtype=np.int64)

    from shapely import STRtree, area, box, intersection

    pieces = _partition_mask_geometry(geometry)
    if len(pieces) == 0:
        return np.empty((0, 2), dtype=np.int64)
    spatial_index = STRtree(pieces)
    selected_mask = np.zeros(points.shape[0], dtype=bool)
    min_x, min_y, max_x, max_y = geometry.bounds
    plausible = (
        (points[:, 0] >= min_x - footprint)
        & (points[:, 0] <= max_x + footprint)
        & (points[:, 1] >= min_y - footprint)
        & (points[:, 1] <= max_y + footprint)
    )
    plausible_indices = np.flatnonzero(plausible)
    required_area = threshold * footprint * footprint
    for start in range(0, len(plausible_indices), chunk_size):
        indices = plausible_indices[start : start + chunk_size]
        xy = points[indices]
        squares = box(
            xy[:, 0],
            xy[:, 1],
            xy[:, 0] + footprint,
            xy[:, 1] + footprint,
        )
        pairs = spatial_index.query(squares, predicate="intersects")
        if threshold == 0:
            intersecting_squares = np.zeros(len(squares), dtype=bool)
            if pairs.shape[1]:
                intersecting_squares[np.unique(pairs[0])] = True
            selected_mask[indices] = intersecting_squares
        else:
            intersection_areas = np.zeros(len(squares), dtype=np.float64)
            if pairs.shape[1]:
                pair_areas = area(intersection(squares[pairs[0]], pieces[pairs[1]]))
                intersection_areas[:] = np.bincount(
                    pairs[0], weights=pair_areas, minlength=len(squares)
                )
            selected_mask[indices] = np.asarray(intersection_areas >= required_area, dtype=bool)
    return np.asarray(points[selected_mask], dtype=np.int64)


def _apply_unsimplified_mask(patcher: Any, mask: Any, threshold: float) -> Any:
    """Apply tissue geometry without a native-MPP-dependent simplify tolerance."""

    if mask is None:
        return patcher
    compute_masked = getattr(patcher, "_compute_masked", None)
    if compute_masked is None:
        raise ExactMppSamplingError("TRIDENT patcher does not expose exact tissue-mask filtering")
    patcher.mask = mask
    candidates = np.asarray(patcher.valid_coords, dtype=np.int64)
    exact_coordinates = _filter_coordinates_by_mask(
        candidates,
        mask=mask,
        footprint=int(patcher.patch_size_src),
        threshold=threshold,
    )
    if exact_coordinates is None:
        patcher.valid_patches_nb, patcher.valid_coords = compute_masked(
            candidates,
            threshold,
            simplify_shape=False,
        )
    else:
        patcher.valid_coords = exact_coordinates
        patcher.valid_patches_nb = len(exact_coordinates)
    return patcher


def _write_coordinate_h5(
    path: Path,
    *,
    coordinates: np.ndarray,
    attributes: Mapping[str, Any],
) -> None:
    with _atomic_output(path) as temporary, h5py.File(temporary, "w") as handle:
        dataset = _create_resizable_dataset(handle, "coords", coordinates)
        _write_attributes(dataset, attributes)


def _read_level_downsample(wsi: Any, level: int) -> float:
    return _positive_finite_float(wsi.level_downsamples[level], field=f"level_downsamples[{level}]")


def _write_feature_h5(
    path: Path,
    *,
    features: np.ndarray,
    coordinates: np.ndarray,
    coordinate_attributes: Mapping[str, Any],
    feature_attributes: Mapping[str, Any],
) -> None:
    with _atomic_output(path) as temporary, h5py.File(temporary, "w") as handle:
        feature_dataset = _create_resizable_dataset(handle, "features", features)
        _write_attributes(feature_dataset, feature_attributes)
        coordinate_dataset = _create_resizable_dataset(handle, "coords", coordinates)
        _write_attributes(coordinate_dataset, coordinate_attributes)


def _get_wsi_patcher_dataset_class() -> type[Any]:
    from trident.wsi_objects.WSIPatcherDataset import WSIPatcherDataset

    return cast(type[Any], WSIPatcherDataset)


def _resolve_num_workers(batch_size: int, configured: int | None) -> int:
    if configured is not None:
        return max(0, int(configured))
    if os.name == "nt":
        return 0
    cores = os.cpu_count() or 16
    return max(1, min(int(0.75 * cores), 2 * batch_size))


class ExactMppWSIAdapter:
    """Delegate a TRIDENT WSI while enforcing physical MPP for sampling."""

    def __init__(
        self,
        wsi: Any,
        target_mpp: float,
        *,
        allow_legacy_replay: bool = False,
        validate_source_mpp: bool = True,
    ) -> None:
        self._wsi = wsi
        self.target_mpp = _positive_finite_float(target_mpp, field="target_mpp")
        self.allow_legacy_replay = allow_legacy_replay
        self.validate_source_mpp = validate_source_mpp

    @property
    def wrapped_wsi(self) -> Any:
        return self._wsi

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wsi, name)

    def __repr__(self) -> str:
        return f"<ExactMppWSIAdapter target_mpp={self.target_mpp} wsi={self._wsi!r}>"

    def _initialize(self) -> None:
        initializer = getattr(self._wsi, "_lazy_initialize", None)
        if initializer is None:
            raise ExactMppSamplingError("Wrapped WSI does not expose TRIDENT's lazy initializer")
        initializer()

    def _source_mpp(self) -> float:
        return _positive_finite_float(
            getattr(self._wsi, "mpp", None), field=f"{self.name} level0_mpp"
        )

    def _validate_target_magnification(self, target_mag: float) -> None:
        requested_magnification = _positive_finite_float(target_mag, field="target_mag")
        implied_magnification = 10.0 / self.target_mpp
        if not _is_close(requested_magnification, implied_magnification):
            raise ExactMppSamplingError(
                f"target_mag={requested_magnification} conflicts with target_mpp={self.target_mpp}; "
                f"expected target_mag={implied_magnification}"
            )

    def segment_tissue(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate segmentation and clear TRIDENT's lock after ordinary failures."""

        try:
            return self._wsi.segment_tissue(*args, **kwargs)
        except BaseException:
            job_dir = kwargs.get("job_dir")
            if job_dir is not None:
                Path(f"{Path(job_dir) / 'contours' / f'{self.name}.jpg'}.lock").unlink(
                    missing_ok=True
                )
            raise

    def extract_tissue_coords(
        self,
        target_mag: int,
        patch_size: int,
        save_coords: str,
        overlap: int = 0,
        min_tissue_proportion: float = 0.0,
    ) -> str:
        """Generate coordinates while cleaning a failed pinned-TRIDENT lock."""

        output_path = Path(save_coords) / "patches" / f"{self.name}_patches.h5"
        try:
            return self._extract_tissue_coords(
                target_mag,
                patch_size,
                save_coords,
                overlap,
                min_tissue_proportion,
            )
        except BaseException:
            Path(f"{output_path}.lock").unlink(missing_ok=True)
            raise

    def _extract_tissue_coords(
        self,
        target_mag: int,
        patch_size: int,
        save_coords: str,
        overlap: int = 0,
        min_tissue_proportion: float = 0.0,
    ) -> str:
        """Generate a level-0 grid using exact source and destination MPP."""

        self._validate_target_magnification(target_mag)
        if patch_size <= 0:
            raise ExactMppSamplingError(f"patch_size must be positive, got {patch_size}")
        if overlap < 0 or overlap >= patch_size:
            raise ExactMppSamplingError(
                f"overlap must satisfy 0 <= overlap < patch_size, got {overlap} and {patch_size}"
            )
        if not 0.0 <= min_tissue_proportion <= 1.0:
            raise ExactMppSamplingError(
                f"min_tissue_proportion must be between 0 and 1, got {min_tissue_proportion}"
            )

        self._initialize()
        source_mpp = self._source_mpp()
        tissue_mask = getattr(self._wsi, "gdf_contours", None)
        patcher = self._wsi.create_patcher(
            patch_size=patch_size,
            src_pixel_size=source_mpp,
            dst_pixel_size=self.target_mpp,
            overlap=overlap,
            mask=None,
            coords_only=True,
            threshold=min_tissue_proportion,
        )
        _apply_unsimplified_mask(patcher, tissue_mask, min_tissue_proportion)
        _force_level0_read_geometry(patcher)
        coordinates = _coordinate_array(patcher)
        if coordinates.shape[0] == 0:
            raise ExactMppSamplingError(f"No tissue coordinates generated for {self.name}")

        output_path = Path(save_coords) / "patches" / f"{self.name}_patches.h5"
        patch_size_level0 = int(patcher.patch_size_src)
        overlap_level0 = int(patcher.overlap_src)
        attributes = {
            # Existing TRIDENT fields. Floating magnification proxies make old
            # consumers reconstruct the same MPP instead of a coarse bucket.
            "patch_size": int(patch_size),
            "patch_size_level0": patch_size_level0,
            "level0_magnification": 10.0 / source_mpp,
            "target_magnification": 10.0 / self.target_mpp,
            "overlap": int(overlap),
            "name": str(self.name),
            "savetodir": str(save_coords),
            "level0_width": int(self.width),
            "level0_height": int(self.height),
            # OceanPath exact-MPP schema.
            "coords_schema_version": COORDS_SCHEMA_VERSION,
            "sampling_mode": EXACT_MPP_SAMPLING_MODE,
            "coordinate_units": "level0_pixels",
            "level0_mpp": source_mpp,
            "target_mpp": self.target_mpp,
            "effective_target_mpp": source_mpp * patch_size_level0 / patch_size,
            "overlap_level0": overlap_level0,
            "min_tissue_proportion": float(min_tissue_proportion),
            "mask_simplification": "none",
            "read_level": int(patcher.level),
            "read_level_downsample": _read_level_downsample(self._wsi, int(patcher.level)),
            "read_patch_size": int(patcher.patch_size_level),
            "actual_read_level0_pixels": int(patcher.patch_size_level),
        }
        _write_coordinate_h5(output_path, coordinates=coordinates, attributes=attributes)
        return str(output_path)

    def visualize_coords(self, coords_path: str, save_patch_viz: str) -> str:
        """Delegate visualization and clear the coordinate lock after failures."""

        try:
            return str(self._wsi.visualize_coords(coords_path, save_patch_viz))
        except BaseException:
            Path(f"{coords_path}.lock").unlink(missing_ok=True)
            raise

    @torch.inference_mode()
    def extract_patch_features(
        self,
        patch_encoder: torch.nn.Module,
        coords_path: str,
        save_features: str,
        device: str = "cuda:0",
        saveas: str = "h5",
        batch_limit: int = 512,
        verbose: bool = False,
    ) -> str:
        """Replay coordinates while cleaning a failed pinned-TRIDENT lock."""

        output_path = Path(save_features) / f"{self.name}.{saveas}"
        try:
            return self._extract_patch_features(
                patch_encoder,
                coords_path,
                save_features,
                device,
                saveas,
                batch_limit,
                verbose,
            )
        except BaseException:
            Path(f"{output_path}.lock").unlink(missing_ok=True)
            raise

    @torch.inference_mode()
    def _extract_patch_features(
        self,
        patch_encoder: torch.nn.Module,
        coords_path: str,
        save_features: str,
        device: str = "cuda:0",
        saveas: str = "h5",
        batch_limit: int = 512,
        verbose: bool = False,
    ) -> str:
        """Replay exact-MPP coordinates and extract patch embeddings."""

        if saveas not in {"h5", "pt"}:
            raise ExactMppSamplingError(
                f"Unsupported feature format: {saveas!r}; expected 'h5' or 'pt'"
            )
        if batch_limit <= 0:
            raise ExactMppSamplingError(f"batch_limit must be positive, got {batch_limit}")

        attributes, coordinates = read_coordinate_h5(coords_path)
        if not is_exact_mpp_metadata(attributes):
            if self.allow_legacy_replay:
                return str(
                    self._wsi.extract_patch_features(
                        patch_encoder=patch_encoder,
                        coords_path=coords_path,
                        save_features=save_features,
                        device=device,
                        saveas=saveas,
                        batch_limit=batch_limit,
                        verbose=verbose,
                    )
                )
            raise ExactMppSamplingError(
                f"Legacy coordinate file cannot be replayed in exact-MPP mode: {coords_path}"
            )
        _validate_exact_schema_attributes(attributes, Path(coords_path))

        stored_target_mpp = _positive_finite_float(
            attributes["target_mpp"], field="stored target_mpp"
        )
        if not _is_close(stored_target_mpp, self.target_mpp):
            raise ExactMppSamplingError(
                f"Coordinate target_mpp={stored_target_mpp} conflicts with adapter "
                f"target_mpp={self.target_mpp}: {coords_path}"
            )

        self._initialize()
        stored_source_mpp = _positive_finite_float(
            attributes["level0_mpp"], field="stored level0_mpp"
        )
        if self.validate_source_mpp:
            current_source_mpp = self._source_mpp()
            if not _is_close(current_source_mpp, stored_source_mpp, source_mpp=True):
                raise ExactMppSamplingError(
                    f"Slide MPP changed since coordinate generation for {self.name}: "
                    f"stored={stored_source_mpp}, current={current_source_mpp}"
                )

        if coordinates.shape[0] == 0:
            raise ExactMppSamplingError(f"No patch coordinates to encode: {coords_path}")

        patch_size = int(attributes["patch_size"])
        patcher = self._wsi.create_patcher(
            patch_size=patch_size,
            src_pixel_size=stored_source_mpp,
            dst_pixel_size=stored_target_mpp,
            custom_coords=coordinates,
            coords_only=False,
            pil=True,
        )
        _force_level0_read_geometry(patcher)
        if int(patcher.patch_size_src) != int(attributes["patch_size_level0"]):
            raise ExactMppSamplingError(
                "Replay footprint conflicts with stored exact-MPP coordinates: "
                f"derived={int(patcher.patch_size_src)}, "
                f"stored={int(attributes['patch_size_level0'])}"
            )

        patch_encoder.to(device)
        patch_encoder.eval()
        precision = getattr(patch_encoder, "precision", torch.float32)
        transforms = getattr(patch_encoder, "eval_transforms", None)
        dataset_class = _get_wsi_patcher_dataset_class()
        dataset = dataset_class(patcher, transforms)
        dataloader: Any = DataLoader(
            dataset,
            batch_size=batch_limit,
            num_workers=_resolve_num_workers(batch_limit, getattr(self._wsi, "max_workers", None)),
            pin_memory=torch.device(device).type == "cuda",
        )
        if verbose:
            from tqdm import tqdm

            dataloader = tqdm(dataloader)

        device_type = torch.device(device).type
        encoded_batches = []
        for images, _ in dataloader:
            images = images.to(device)
            with torch.autocast(
                device_type=device_type,
                dtype=precision,
                enabled=device_type == "cuda" and precision != torch.float32,
            ):
                encoded = patch_encoder(images)
            encoded_batches.append(encoded.detach().to(dtype=torch.float32).cpu().numpy())

        features = np.concatenate(encoded_batches, axis=0)
        output_path = Path(save_features) / f"{self.name}.{saveas}"
        model_name = getattr(patch_encoder, "enc_name", None)
        if saveas == "h5":
            _write_feature_h5(
                output_path,
                features=features,
                coordinates=coordinates,
                coordinate_attributes=attributes,
                feature_attributes={
                    "name": str(self.name),
                    "savetodir": str(save_features),
                    "encoder": model_name,
                },
            )
        elif saveas == "pt":
            with _atomic_output(output_path) as temporary:
                torch.save(features, temporary)
        return str(output_path)

    def extract_slide_features(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate slide encoding and clear TRIDENT's lock after failures."""

        save_features = kwargs.get("save_features")
        try:
            return self._wsi.extract_slide_features(*args, **kwargs)
        except BaseException:
            if save_features is not None:
                output_path = Path(save_features) / f"{self.name}.h5"
                Path(f"{output_path}.lock").unlink(missing_ok=True)
            raise


def wrap_processor_wsis_for_exact_mpp(
    processor: Any,
    target_mpp: float,
    *,
    allow_legacy_replay: bool = False,
    validate_source_mpp: bool = True,
) -> Any:
    """Replace a Processor's WSI list with exact-MPP delegating adapters."""

    if not hasattr(processor, "wsis"):
        raise ExactMppSamplingError("Processor has no 'wsis' collection")
    processor.wsis = [
        ExactMppWSIAdapter(
            wsi,
            target_mpp,
            allow_legacy_replay=allow_legacy_replay,
            validate_source_mpp=validate_source_mpp,
        )
        for wsi in processor.wsis
    ]
    return processor


__all__ = [
    "COORDS_SCHEMA_VERSION",
    "EXACT_MPP_SAMPLING_MODE",
    "ExactMppSamplingError",
    "ExactMppWSIAdapter",
    "is_exact_mpp_metadata",
    "read_coordinate_h5",
    "validate_exact_mpp_coordinate_file",
    "wrap_processor_wsis_for_exact_mpp",
]
