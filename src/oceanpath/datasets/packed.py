"""
Packed feature store: one flat memory-mapped array for a whole cohort.

Why this exists
═══════════════════════════════════════════════════════════════════════════

The extraction stage writes one H5 per slide. That is the right *publication*
format — atomic, resumable, one file per unit of work — but it is a poor
*training* format:

  - h5py costs ~40 ms per slide even when the file is entirely in page cache
    (chunk assembly + Python-level overhead), against ~1-11 ms of GPU work
    per slide for an ABMIL step. Training is therefore reader-bound.
  - h5py fancy indexing (``dset[sorted_random_rows]``) is ~13x SLOWER than
    reading the whole dataset and slicing in numpy, so "read only the K
    patches I sampled" is a pessimisation, not an optimisation.
  - Features are stored float32. Patch encoders emit values well inside the
    float16 range, so half the bytes moved off disk, through the DataLoader
    IPC queue, and over PCIe are carrying no information.

Packing rewrites the cohort once into:

    {pack_dir}/meta.json      — feat_dim, dtypes, counts, source evidence
    {pack_dir}/index.parquet  — slide_id -> (offset, n_patches)
    {pack_dir}/features.bin   — flat [total_patches, D] contiguous
    {pack_dir}/coords.bin     — flat [total_patches, 2] int32

Reads then become numpy memmap slices: no decoder, no per-file open, and the
OS page cache holds the whole store (27 GB in float16 for the colon cohort,
against 188 GB of RAM), so after the first epoch every read is a memcpy.

The store is append-only and content-addressed by ``source_inventory_sha256``
so a stale pack cannot be silently trained against after re-extraction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
META_FILE = "meta.json"
INDEX_FILE = "index.parquet"
FEATURES_FILE = "features.bin"
COORDS_FILE = "coords.bin"

H5_FEATURES_KEY = "features"
H5_COORDS_KEY = "coords"

_COORD_DTYPE = np.int32


class PackedStoreError(RuntimeError):
    """Raised when a pack directory is missing, malformed, or stale."""


def feature_inventory_sha256(feature_dir: Path) -> str:
    """Fingerprint a per-slide H5 directory by (name, size, mtime_ns).

    Mirrors ``workflows.training._feature_inventory_sha256`` so a pack can be
    proven to correspond to the exact feature inventory it was built from.
    """
    feature_dir = Path(feature_dir)
    if not feature_dir.is_dir():
        return "missing"
    paths = sorted(feature_dir.glob("*.h5"))
    if not paths:
        return "empty"
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


# ── Reader ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PackedMeta:
    schema_version: int
    feat_dim: int
    feat_dtype: str
    coord_dim: int
    n_slides: int
    total_patches: int
    source_dir: str
    source_inventory_sha256: str
    has_coords: bool


@dataclass(frozen=True)
class _ScannedSlide:
    """Shape-only description collected before any output bytes are written."""

    slide_id: str
    path: Path
    n_patches: int
    feat_dim: int
    coord_dim: int


def _shape_2d(dataset, *, path: Path, key: str) -> tuple[int, int]:
    """Normalize ``(N, D)`` and legacy ``(1, N, D)`` H5 layouts."""

    shape = tuple(int(value) for value in dataset.shape)
    if len(shape) == 3 and shape[0] == 1:
        return shape[1], shape[2]
    if len(shape) == 2:
        return shape
    raise ValueError(f"{path.name}:{key} must have shape (N, D) or (1, N, D), got {shape}")


def _read_h5_block(dataset, start: int, end: int) -> np.ndarray:
    """Read one row block without materialising the complete slide."""

    if dataset.ndim == 3:
        return np.asarray(dataset[0, start:end, :])
    return np.asarray(dataset[start:end, :])


def _packed_layout(pack_dir: Path) -> tuple[PackedMeta, Any]:
    """Load and structurally validate packed metadata and its slide index."""

    import pandas as pd

    meta_path = pack_dir / META_FILE
    if not meta_path.is_file():
        raise PackedStoreError(
            f"No packed feature store at {pack_dir} (missing {META_FILE}). "
            "Build one with: python scripts/pack_features.py "
            f"--feature-dir <features_*> --pack-dir {pack_dir}"
        )

    try:
        raw = json.loads(meta_path.read_text())
        meta = PackedMeta(
            schema_version=int(raw["schema_version"]),
            feat_dim=int(raw["feat_dim"]),
            feat_dtype=str(raw["feat_dtype"]),
            coord_dim=int(raw["coord_dim"]),
            n_slides=int(raw["n_slides"]),
            total_patches=int(raw["total_patches"]),
            source_dir=str(raw["source_dir"]),
            source_inventory_sha256=str(raw["source_inventory_sha256"]),
            has_coords=bool(raw["has_coords"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PackedStoreError(f"Malformed {META_FILE} in {pack_dir}: {exc}") from exc

    if meta.schema_version != SCHEMA_VERSION:
        raise PackedStoreError(
            f"Packed store schema {meta.schema_version} != expected "
            f"{SCHEMA_VERSION} at {pack_dir}; rebuild the pack."
        )
    try:
        feat_dtype = np.dtype(meta.feat_dtype)
    except TypeError as exc:
        raise PackedStoreError(f"Invalid feature dtype {meta.feat_dtype!r}") from exc
    if feat_dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise PackedStoreError(
            f"Packed features must be float16 or float32, got {meta.feat_dtype!r}"
        )
    if meta.n_slides <= 0 or meta.total_patches <= 0 or meta.feat_dim <= 0:
        raise PackedStoreError(
            "Packed metadata counts must be positive: "
            f"slides={meta.n_slides}, patches={meta.total_patches}, dim={meta.feat_dim}"
        )
    if meta.has_coords != (meta.coord_dim > 0):
        raise PackedStoreError(
            f"Inconsistent coordinate metadata: has_coords={meta.has_coords}, "
            f"coord_dim={meta.coord_dim}"
        )

    index_path = pack_dir / INDEX_FILE
    if not index_path.is_file():
        raise PackedStoreError(f"Packed store is missing {INDEX_FILE}: {pack_dir}")
    try:
        index = pd.read_parquet(index_path)
    except Exception as exc:
        raise PackedStoreError(f"Cannot read {INDEX_FILE} in {pack_dir}: {exc}") from exc
    required = {"slide_id", "offset", "n_patches"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise PackedStoreError(f"{INDEX_FILE} is missing columns: {missing}")
    if len(index) != meta.n_slides:
        raise PackedStoreError(
            f"{INDEX_FILE} has {len(index)} slides but metadata declares {meta.n_slides}"
        )

    slide_ids = [str(value) for value in index["slide_id"].tolist()]
    if any(not slide_id for slide_id in slide_ids) or len(set(slide_ids)) != len(slide_ids):
        raise PackedStoreError(f"{INDEX_FILE} contains empty or duplicate slide IDs")
    try:
        offsets = index["offset"].to_numpy(dtype=np.int64)
        lengths = index["n_patches"].to_numpy(dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise PackedStoreError(f"{INDEX_FILE} offsets and lengths must be integers") from exc
    if np.any(lengths <= 0):
        raise PackedStoreError(f"{INDEX_FILE} contains non-positive slide lengths")
    expected_offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
    )
    if not np.array_equal(offsets, expected_offsets):
        raise PackedStoreError(f"{INDEX_FILE} offsets are not contiguous cumulative row offsets")
    if int(lengths.sum()) != meta.total_patches:
        raise PackedStoreError(
            f"{INDEX_FILE} contains {int(lengths.sum())} patches but metadata declares "
            f"{meta.total_patches}"
        )

    expected_features = meta.total_patches * meta.feat_dim * feat_dtype.itemsize
    feature_path = pack_dir / FEATURES_FILE
    actual_features = feature_path.stat().st_size if feature_path.is_file() else -1
    if actual_features != expected_features:
        raise PackedStoreError(
            f"{FEATURES_FILE} is {actual_features} bytes but the index implies "
            f"{expected_features}; the pack is truncated or corrupt. Rebuild it."
        )

    coord_path = pack_dir / COORDS_FILE
    if meta.has_coords:
        expected_coords = (
            meta.total_patches * meta.coord_dim * np.dtype(_COORD_DTYPE).itemsize
        )
        actual_coords = coord_path.stat().st_size if coord_path.is_file() else -1
        if actual_coords != expected_coords:
            raise PackedStoreError(
                f"{COORDS_FILE} is {actual_coords} bytes but the index implies "
                f"{expected_coords}; the pack is truncated or corrupt. Rebuild it."
            )
    elif coord_path.exists():
        raise PackedStoreError(f"Unexpected {COORDS_FILE} in a feature-only packed store")

    return meta, index


def validate_packed_dir(
    pack_dir: str | Path,
    *,
    verify_source: str | None = None,
) -> PackedMeta:
    """Validate a complete packed store and optionally its live-source identity."""

    meta, _ = _packed_layout(Path(pack_dir))
    if verify_source is not None and verify_source != meta.source_inventory_sha256:
        raise PackedStoreError(
            f"Packed store at {pack_dir} is STALE: it was built from feature "
            f"inventory {meta.source_inventory_sha256[:12]}, but the live "
            f"inventory is {verify_source[:12]}. Re-run scripts/pack_features.py."
        )
    return meta


class PackedFeatureStore:
    """Read-only view over a packed cohort.

    Memmaps are opened lazily on first access so an instance can be
    constructed in the parent process and used after a DataLoader fork
    without carrying open file objects across the boundary.

    Parameters
    ----------
    pack_dir : str or Path
        Directory produced by :func:`pack_features`.
    verify_source : str or None
        When given, the ``source_inventory_sha256`` recorded at pack time must
        equal this value or :class:`PackedStoreError` is raised. Pass the live
        inventory hash of the feature directory to guarantee the pack is not
        stale.
    """

    def __init__(self, pack_dir: str | Path, verify_source: str | None = None):
        self.pack_dir = Path(pack_dir)
        self.meta, index = _packed_layout(self.pack_dir)
        if verify_source is not None and verify_source != self.meta.source_inventory_sha256:
            raise PackedStoreError(
                f"Packed store at {self.pack_dir} is STALE: it was built from feature "
                f"inventory {self.meta.source_inventory_sha256[:12]}, but the live "
                f"inventory is {verify_source[:12]}. Re-run scripts/pack_features.py."
            )
        self._slide_ids: list[str] = [str(s) for s in index["slide_id"].tolist()]
        self._offsets: np.ndarray = index["offset"].to_numpy(dtype=np.int64)
        self._lengths: np.ndarray = index["n_patches"].to_numpy(dtype=np.int64)
        self._pos: dict[str, int] = {sid: i for i, sid in enumerate(self._slide_ids)}

        self._features: np.memmap | None = None
        self._coords: np.memmap | None = None

    # -- memmap lifecycle (fork-safe) -----------------------------------------

    def _feature_map(self) -> np.memmap:
        if self._features is None:
            self._features = np.memmap(
                self.pack_dir / FEATURES_FILE,
                dtype=np.dtype(self.meta.feat_dtype),
                mode="r",
                shape=(self.meta.total_patches, self.meta.feat_dim),
            )
        return self._features

    def _coord_map(self) -> np.memmap:
        if not self.meta.has_coords:
            raise PackedStoreError(f"Packed store {self.pack_dir} has no coords")
        if self._coords is None:
            self._coords = np.memmap(
                self.pack_dir / COORDS_FILE,
                dtype=_COORD_DTYPE,
                mode="r",
                shape=(self.meta.total_patches, self.meta.coord_dim),
            )
        return self._coords

    def __getstate__(self) -> dict:
        # Never pickle live memmaps across the DataLoader worker boundary.
        state = self.__dict__.copy()
        state["_features"] = None
        state["_coords"] = None
        return state

    # -- accessors -------------------------------------------------------------

    @property
    def slide_ids(self) -> list[str]:
        return self._slide_ids

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    @property
    def feat_dim(self) -> int:
        return self.meta.feat_dim

    @property
    def has_coords(self) -> bool:
        return self.meta.has_coords

    def __contains__(self, slide_id: str) -> bool:
        return slide_id in self._pos

    def __len__(self) -> int:
        return len(self._slide_ids)

    def position(self, slide_id: str) -> int:
        try:
            return self._pos[slide_id]
        except KeyError:
            raise KeyError(
                f"slide_id {slide_id!r} is not in packed store {self.pack_dir}"
            ) from None

    def length_of(self, slide_id: str) -> int:
        return int(self._lengths[self.position(slide_id)])

    def read_features(self, pos: int, rows: np.ndarray | None = None) -> np.ndarray:
        """Return this slide's features, optionally only ``rows`` of them.

        ``rows`` are slide-local indices. Gathering a subset touches only the
        pages those rows live on, which is the whole point of the flat layout:
        a 2048-of-12500 sample moves ~4 MB instead of ~25 MB.

        The returned array is always a fresh, contiguous, writable copy — never
        a view into the memmap — so downstream augmentation cannot fault pages
        back in or attempt to write to read-only memory.
        """
        start = int(self._offsets[pos])
        n = int(self._lengths[pos])
        block = self._feature_map()[start : start + n]
        if rows is None:
            return np.array(block, copy=True)
        # NumPy advanced indexing already returns a fresh contiguous ndarray.
        # Do not wrap it in np.array(copy=True): that silently doubles every
        # selected-row memory copy in the training hot path.
        return cast(np.ndarray, block[rows])

    def read_coords(self, pos: int, rows: np.ndarray | None = None) -> np.ndarray:
        start = int(self._offsets[pos])
        n = int(self._lengths[pos])
        block = self._coord_map()[start : start + n]
        if rows is None:
            return np.array(block, copy=True)
        return cast(np.ndarray, block[rows])


class ResidentPackedStore:
    """A packed cohort held resident in device memory as one torch tensor.

    Where :class:`PackedFeatureStore` serves memmap gathers that must then
    cross the DataLoader IPC queue, the pin thread, and the PCIe bus on every
    step, this store loads ``features.bin`` ONCE into a single tensor on
    ``device`` and serves per-slide views / ``index_select`` gathers from it:

      - ``device="cuda"``: the steady-state per-sample cost becomes a
        device-side gather — no worker IPC, no pinning, no H2D copy. The
        pack must fit in VRAM next to the model (CONCH at ~8 GB does on a
        24 GB card; UNI at ~39 GB does not).
      - ``device="cpu"``: a plain RAM-resident tensor — same read interface,
        removes memmap page-fault jitter, and composes with num_workers=0.

    Requires ``num_workers=0``: CUDA tensors cannot be produced inside forked
    DataLoader workers, and a resident store makes workers pointless anyway
    (the read is an index op, not I/O). :class:`~oceanpath.datasets.datamodule.
    MILDataModule` enforces this.

    ``read_features``/``read_coords`` mirror :class:`PackedFeatureStore` but
    return ``torch.Tensor`` on ``device``. Full-slide reads (``rows=None``)
    return a zero-copy view; callers must not mutate it (the training
    collators always copy into the batch tensor).
    """

    def __init__(self, store: PackedFeatureStore, device: str = "cuda"):
        import torch

        self._torch = torch
        self.pack_dir = store.pack_dir
        self.meta = store.meta
        self.device = torch.device(device)
        self._slide_ids = store.slide_ids
        self._offsets = store._offsets
        self._lengths = store._lengths
        self._pos = store._pos

        feat_bytes = self.meta.total_patches * self.meta.feat_dim * np.dtype(
            self.meta.feat_dtype
        ).itemsize
        coord_bytes = (
            self.meta.total_patches * self.meta.coord_dim * np.dtype(_COORD_DTYPE).itemsize
            if self.meta.has_coords
            else 0
        )
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            needed = feat_bytes + coord_bytes
            if needed > free:
                raise PackedStoreError(
                    f"Resident store needs {needed / 1e9:.1f} GB on {self.device} but only "
                    f"{free / 1e9:.1f} GB of {total / 1e9:.1f} GB is free. Use the memmap "
                    "store (resident_device=null) or a smaller encoder pack."
                )

        logger.info(
            "Loading packed store %s into %s memory (%.1f GB features%s)",
            self.pack_dir,
            self.device,
            feat_bytes / 1e9,
            f" + {coord_bytes / 1e9:.1f} GB coords" if coord_bytes else "",
        )
        # from_numpy on the memmap is zero-copy; .to(device) streams the pages
        # once. For CPU residency an explicit empty+copy_ forces real RAM
        # backing (plain .to("cpu") would return the memmap view unchanged).
        # The read-only-array warning is expected: the memmap is mode="r" and
        # we only ever read it into the resident copy.
        import warnings

        def _load(mm: np.memmap):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                src = torch.from_numpy(np.asarray(mm))
            if self.device.type == "cpu":
                out = torch.empty_like(src, device="cpu")
                out.copy_(src)
                return out
            return src.to(self.device)

        self._features = _load(store._feature_map())
        self._coords = _load(store._coord_map()) if self.meta.has_coords else None

    def __getstate__(self) -> dict:
        # A resident store must never cross a process boundary: pickling would
        # serialize the full tensor (or crash on CUDA-after-fork). Fail here
        # with an actionable message instead of deep inside a DataLoader worker.
        raise PackedStoreError(
            "ResidentPackedStore cannot be pickled across process boundaries. "
            "Use num_workers=0 (MILDataModule enforces this automatically), or "
            "use the memmap PackedFeatureStore for worker-based loading."
        )

    # -- accessors (interface-compatible with PackedFeatureStore) --------------

    @property
    def slide_ids(self) -> list[str]:
        return self._slide_ids

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    @property
    def feat_dim(self) -> int:
        return self.meta.feat_dim

    @property
    def has_coords(self) -> bool:
        return self.meta.has_coords

    def __contains__(self, slide_id: str) -> bool:
        return slide_id in self._pos

    def __len__(self) -> int:
        return len(self._slide_ids)

    def position(self, slide_id: str) -> int:
        try:
            return self._pos[slide_id]
        except KeyError:
            raise KeyError(
                f"slide_id {slide_id!r} is not in packed store {self.pack_dir}"
            ) from None

    def length_of(self, slide_id: str) -> int:
        return int(self._lengths[self.position(slide_id)])

    def _rows_to_index(self, rows: np.ndarray):
        torch = self._torch
        return torch.from_numpy(np.ascontiguousarray(rows, dtype=np.int64)).to(
            self.device, non_blocking=True
        )

    def read_features(self, pos: int, rows: np.ndarray | None = None):
        start = int(self._offsets[pos])
        n = int(self._lengths[pos])
        block = self._features.narrow(0, start, n)
        if rows is None:
            return block  # zero-copy view; collators copy on stack
        return block.index_select(0, self._rows_to_index(rows))

    def read_coords(self, pos: int, rows: np.ndarray | None = None):
        if self._coords is None:
            raise PackedStoreError(f"Packed store {self.pack_dir} has no coords")
        start = int(self._offsets[pos])
        n = int(self._lengths[pos])
        block = self._coords.narrow(0, start, n)
        if rows is None:
            return block
        return block.index_select(0, self._rows_to_index(rows))


# ── Writer ────────────────────────────────────────────────────────────────────


def pack_features(
    feature_dir: str | Path,
    pack_dir: str | Path,
    feat_dtype: str = "float16",
    slide_ids: list[str] | None = None,
    include_coords: bool = True,
    overwrite: bool = False,
    log_every: int = 50,
    stream_chunk_size: int = 65_536,
    verify_source_unchanged: bool = True,
) -> PackedMeta:
    """Rewrite a directory of per-slide H5 features into a flat packed store.

    Uses a two-pass scan/write design and streams bounded row blocks from H5.
    Peak feature memory is therefore ``stream_chunk_size * feat_dim`` rather
    than one complete slide. Every patch is retained; bag capping remains a
    training-time operation so each epoch can draw a different view.

    Parameters
    ----------
    feature_dir : str or Path
        TRIDENT-style ``features_{encoder}`` directory of ``{slide_id}.h5``.
    pack_dir : str or Path
        Destination. Written to a ``.tmp`` sibling and renamed on success so a
        crashed pack can never be mistaken for a complete one.
    feat_dtype : str
        Storage dtype for features. ``float16`` halves bytes at every level of
        the pipeline; patch-encoder outputs sit far inside its range. Values
        are checked for overflow and non-finite entries during the copy.
    slide_ids : list[str] or None
        Restrict the pack to these IDs. None packs every H5 present.
    include_coords : bool
        Also pack the ``coords`` dataset (needed for spatial-stratified
        sampling and attention heatmaps).
    stream_chunk_size : int
        Maximum rows read from an H5 dataset per write operation.
    verify_source_unchanged : bool
        Re-hash the H5 inventory after writing and abort publication if feature
        extraction changed the source directory during the build.
    """
    import h5py
    import pandas as pd

    from oceanpath.contracts import normalize_slide_id

    feature_dir = Path(feature_dir)
    pack_dir = Path(pack_dir)
    if stream_chunk_size < 1:
        raise ValueError(f"stream_chunk_size must be >= 1, got {stream_chunk_size}")
    try:
        np_feat_dtype = np.dtype(feat_dtype)
    except TypeError as exc:
        raise ValueError(f"Invalid feature dtype: {feat_dtype!r}") from exc
    if np_feat_dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError(f"feat_dtype must be float16 or float32, got {feat_dtype!r}")
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")
    if pack_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{pack_dir} already exists. Pass overwrite=True (or --overwrite) to rebuild."
            )
        if pack_dir.is_symlink():
            raise ValueError(f"Refusing to overwrite symlinked pack directory: {pack_dir}")

    inventory_hash = feature_inventory_sha256(feature_dir)

    available: dict[str, Path] = {}
    for source_path in sorted(feature_dir.glob("*.h5")):
        slide_id = normalize_slide_id(source_path.name)
        if slide_id in available:
            raise ValueError(
                f"Multiple H5 files normalize to slide_id {slide_id!r}: "
                f"{available[slide_id].name}, {source_path.name}"
            )
        available[slide_id] = source_path
    wanted = list(available) if slide_ids is None else [normalize_slide_id(s) for s in slide_ids]
    if len(set(wanted)) != len(wanted):
        raise ValueError("slide_ids contains duplicates after normalization")

    # Pass 1: shape-only scan. Fail before writing gigabytes if one slide is
    # malformed or incompatible with the cohort.
    scanned: list[_ScannedSlide] = []
    skipped: list[str] = []
    feat_dim: int | None = None
    coord_dim = 0
    for sid in wanted:
        path = available.get(sid)
        if path is None:
            skipped.append(sid)
            continue
        with h5py.File(path, "r") as h5:
            if H5_FEATURES_KEY not in h5:
                raise ValueError(f"{path.name} is missing {H5_FEATURES_KEY!r}")
            n_patches, current_feat_dim = _shape_2d(
                h5[H5_FEATURES_KEY], path=path, key=H5_FEATURES_KEY
            )
            if n_patches == 0:
                skipped.append(sid)
                continue
            if current_feat_dim <= 0:
                raise ValueError(f"Invalid feature dimension in {path.name}: {current_feat_dim}")
            if feat_dim is None:
                feat_dim = current_feat_dim
            elif current_feat_dim != feat_dim:
                raise ValueError(
                    f"Inconsistent feature dim in {path.name}: {current_feat_dim} != {feat_dim} "
                    "(mixed encoders in one directory?)"
                )

            current_coord_dim = 0
            if include_coords:
                if H5_COORDS_KEY not in h5:
                    raise ValueError(f"{path.name} is missing requested {H5_COORDS_KEY!r}")
                coord_n, current_coord_dim = _shape_2d(
                    h5[H5_COORDS_KEY], path=path, key=H5_COORDS_KEY
                )
                if coord_n != n_patches:
                    raise ValueError(
                        f"Patch count mismatch in {path.name}: "
                        f"features={n_patches}, coords={coord_n}"
                    )
                if current_coord_dim <= 0:
                    raise ValueError(
                        f"Invalid coordinate dimension in {path.name}: {current_coord_dim}"
                    )
                if coord_dim == 0:
                    coord_dim = current_coord_dim
                elif current_coord_dim != coord_dim:
                    raise ValueError(
                        f"Inconsistent coordinate dim in {path.name}: "
                        f"{current_coord_dim} != {coord_dim}"
                    )
            scanned.append(
                _ScannedSlide(
                    slide_id=sid,
                    path=path,
                    n_patches=n_patches,
                    feat_dim=current_feat_dim,
                    coord_dim=current_coord_dim,
                )
            )

    if feat_dim is None or not scanned:
        raise ValueError(f"No usable H5 feature files found in {feature_dir}")

    pack_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{pack_dir.name}.tmp-", dir=str(pack_dir.parent))
    )
    feat_max = float(np.finfo(np_feat_dtype).max) if np_feat_dtype.kind == "f" else None

    rows: list[dict] = []
    offset = 0
    n_clipped = 0

    features_path = staging / FEATURES_FILE
    coords_path = staging / COORDS_FILE

    backup: Path | None = None
    try:
        with ExitStack() as stack:
            fout = stack.enter_context(open(features_path, "wb"))
            coord_context = (
                stack.enter_context(open(coords_path, "wb")) if include_coords else None
            )
            for i, slide in enumerate(scanned):
                with h5py.File(slide.path, "r") as h5:
                    feat_dataset = h5[H5_FEATURES_KEY]
                    for start in range(0, slide.n_patches, stream_chunk_size):
                        end = min(start + stream_chunk_size, slide.n_patches)
                        block = _read_h5_block(feat_dataset, start, end)
                        if not np.isfinite(block).all():
                            raise ValueError(
                                f"{slide.path.name} contains non-finite features; "
                                "refusing to pack. Re-extract this slide."
                            )
                        if feat_max is not None:
                            over = int(np.count_nonzero(np.abs(block) > feat_max))
                            if over:
                                n_clipped += over
                                block = np.clip(block, -feat_max, feat_max)
                        np.ascontiguousarray(block, dtype=np_feat_dtype).tofile(fout)

                    # Finish the feature dataset before moving to coords. HDF5
                    # chunk reads stay sequential instead of bouncing between
                    # two datasets for every block.
                    if include_coords:
                        coord_dataset = h5[H5_COORDS_KEY]
                        for start in range(0, slide.n_patches, stream_chunk_size):
                            end = min(start + stream_chunk_size, slide.n_patches)
                            coord_block = _read_h5_block(coord_dataset, start, end)
                            if not np.isfinite(coord_block).all():
                                raise ValueError(
                                    f"{slide.path.name} contains non-finite coordinates"
                                )
                            coord_out = np.ascontiguousarray(coord_block, dtype=_COORD_DTYPE)
                            assert coord_context is not None
                            coord_out.tofile(coord_context)

                rows.append(
                    {
                        "slide_id": slide.slide_id,
                        "offset": offset,
                        "n_patches": slide.n_patches,
                    }
                )
                offset += slide.n_patches

                if log_every and (i + 1) % log_every == 0:
                    logger.info(
                        "pack_features: %d/%d slides, %d patches, %.1f GB written",
                        i + 1,
                        len(scanned),
                        offset,
                        offset * feat_dim * np_feat_dtype.itemsize / 1e9,
                    )

        if skipped:
            logger.warning(
                "pack_features skipped %d slide(s) (missing H5 or 0 patches). First 5: %s",
                len(skipped),
                skipped[:5],
            )
        if n_clipped:
            logger.warning(
                "pack_features clipped %d feature value(s) to the %s range.",
                n_clipped,
                feat_dtype,
            )

        meta = PackedMeta(
            schema_version=SCHEMA_VERSION,
            feat_dim=feat_dim,
            feat_dtype=np_feat_dtype.name,
            coord_dim=coord_dim,
            n_slides=len(rows),
            total_patches=offset,
            source_dir=str(feature_dir),
            source_inventory_sha256=inventory_hash,
            has_coords=include_coords,
        )
        pd.DataFrame(rows).to_parquet(staging / INDEX_FILE, index=False)
        (staging / META_FILE).write_text(json.dumps(meta.__dict__, indent=2, sort_keys=True))

        current_hash = feature_inventory_sha256(feature_dir)
        if verify_source_unchanged and current_hash != inventory_hash:
            raise RuntimeError(
                "Feature inventory changed while packing; extraction is still publishing H5 "
                "files. Wait for extraction to finish and rebuild the pack."
            )
        validate_packed_dir(staging, verify_source=inventory_hash)

        # Keep the previous valid pack until its replacement has been fully
        # written and validated. Renames stay on the same filesystem.
        if pack_dir.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{pack_dir.name}.old-", dir=str(pack_dir.parent))
            )
            backup.rmdir()
            pack_dir.rename(backup)
        try:
            staging.rename(pack_dir)
        except Exception:
            if backup is not None and backup.exists() and not pack_dir.exists():
                backup.rename(pack_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not pack_dir.exists():
            backup.rename(pack_dir)

    logger.info(
        "Packed %d slides / %d patches / D=%d as %s → %s (%.1f GB)",
        meta.n_slides,
        meta.total_patches,
        meta.feat_dim,
        meta.feat_dtype,
        pack_dir,
        meta.total_patches * meta.feat_dim * np_feat_dtype.itemsize / 1e9,
    )
    return meta
