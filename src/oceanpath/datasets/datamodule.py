"""
Lightning DataModule for MIL training.

Slides are loaded one by one from per-slide H5 feature files — the exact
artifacts the extraction stage writes ({feature_dir}/{slide_id}.h5 with
``features`` [N, D] and ``coords`` [N, 2] datasets). There is no intermediate
packed store; adding a slide to the feature directory is enough to train on it.

Subsampling architecture
═══════════════════════════════════════════════════════════════════════════

Two distinct caps control how many patches each slide uses:

  1. dataset_max_instances (SlideDataset level)
     - Train: stochastic random subsampling for regularization. Each epoch
       sees a different random subset of patches.
     - Val/Test: deterministic subsampling with the same cap. Set this to
       None to use every patch in every split.
     - Default: None (no dataset-level subsampling).

  2. max_instances (Collator level)
     - Both train and eval. Hard ceiling on the padded tensor dimension.
       Prevents OOM from slides with >10k patches.
     - MILCollator (train): pre-allocated [B, max_instances, D] buffer.
     - SimpleMILCollator (eval): dynamic padding but capped at max_instances.
     - Default: 8000.

The model has NO subsampling logic — it receives pre-subsampled, padded
tensors with a mask and processes them as-is.

Collation:
  MILCollator: direct fixed-width output (training)
  SimpleMILCollator: dynamic padding to batch-max, capped (val/test)

Uniform batches use ``mask=None`` (all tokens valid). Ragged batches carry a
compact boolean mask. DataLoader's pin thread performs the one required pinned
copy when CUDA is active.
"""

import logging
from collections import Counter, OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from oceanpath.contracts import normalize_slide_id
from oceanpath.splitting import get_slide_ids_for_fold, load_splits

logger = logging.getLogger(__name__)

_TRAIN_SAMPLING_STRATEGIES = frozenset(
    {
        "slide_uniform",
        "patient_natural",
        "cohort_balanced",
        "cohort_label_balanced",
    }
)


@lru_cache(maxsize=1)
def _cuda_pin_memory_available() -> bool:
    """Return True only when CUDA pinned-memory transfer is genuinely usable."""
    try:
        if not torch.cuda.is_available():
            return False
        _ = torch.empty(1).pin_memory()
        return True
    except Exception:
        return False


# ── Collation ─────────────────────────────────────────────────────────────────


class MILCollator:
    """
    Fixed-width collator: every batch is [B, max_instances, D].

    Bags shorter than ``max_instances`` are zero-padded and masked; longer
    bags are truncated to the first ``max_instances`` rows (the dataset has
    already done any stochastic selection).

    Used for TRAINING where bag sizes are bounded by dataset subsampling.

    Allocation strategy
    ═══════════════════════════════════════════════════════════════════
    This collator deliberately does NOT hold a pinned scratch buffer.
    An earlier version pre-allocated ``[batch_size, max_instances, D]`` in
    pinned memory and returned ``buffer[:B].clone()`` — but ``clone()``
    allocates through the ordinary CPU allocator and drops the pinning, so
    the output was pageable anyway and ``DataLoader(pin_memory=True)`` had
    to copy it a second time. The net effect was four passes over every
    batch (zero, fill, clone, pin) and, with ``num_workers>0``, one pinned
    buffer per worker of unswappable host memory.

    Allocating the output directly and letting the DataLoader's pin thread
    perform the single pinning copy is both faster and safer. Padding is
    zeroed by region rather than by zeroing the whole tensor, so a batch of
    full-length bags costs no zeroing at all.
    """

    def __init__(
        self,
        max_instances: int,
        feat_dim: int,
        batch_size: int,
        pin_memory: bool = True,
    ):
        self.max_instances = max_instances
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        # Retained for API compatibility; pinning is the DataLoader's job.
        self.pin_memory = pin_memory

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)
        M = self.max_instances
        feat_dtype = batch[0]["features"].dtype
        # Resident-store samples arrive on GPU; padded buffers must live there
        # too (cross-device index assignment raises).
        device = batch[0]["features"].device
        has_coords = "coords" in batch[0]

        raw_lengths = [int(s["features"].shape[0]) for s in batch]
        lengths = [min(n, M) for n in raw_lengths]
        labels = [s["label"] for s in batch]
        slide_ids = [s["slide_id"] for s in batch]

        # Uniform full-length bags (the fixed_bag_size regime): a single
        # contiguous stack, no padding, no masking work. The check must use
        # the RAW row counts — a bag longer than M also has a clamped length
        # of M but cannot be stacked without truncation.
        uniform = all(n == M for n in raw_lengths)
        if uniform:
            features = torch.stack([s["features"] for s in batch])
            # None is the model contract for "every token is valid". Avoid an
            # all-ones allocation, worker IPC/H2D transfer, bool conversion and
            # masked_fill in every fixed-bag training step.
            mask = None
        else:
            features = torch.empty(B, M, self.feat_dim, dtype=feat_dtype, device=device)
            mask = torch.zeros(B, M, dtype=torch.bool, device=device)
            for i, sample in enumerate(batch):
                n = lengths[i]
                features[i, :n] = sample["features"][:n]
                if n < M:
                    features[i, n:].zero_()
                mask[i, :n] = True

        result = {
            "features": features,
            "mask": mask,
            "labels": torch.tensor(labels, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.int32),
            "slide_ids": slide_ids,
        }
        if "weight" in batch[0]:
            result["weights"] = torch.tensor(
                [s["weight"] for s in batch], dtype=torch.float32
            )

        if has_coords:
            coord_dim = batch[0]["coords"].shape[-1]
            if uniform:
                coords_t = torch.stack([sample["coords"] for sample in batch])
            else:
                coords_t = torch.empty(B, M, coord_dim, dtype=torch.int32, device=device)
                for i, sample in enumerate(batch):
                    n = lengths[i]
                    coords_t[i, :n] = sample["coords"][:n]
                    if n < M:
                        coords_t[i, n:].zero_()
            result["coords"] = coords_t

        return result


_VALID_SAMPLING_MODES = ("contiguous", "random", "spatial_stratified")

# Packed stores opened this process, keyed by (resolved pack_dir, device) and
# stamped with (features.bin size, mtime_ns) so a rebuilt pack is never served
# stale. Deliberately process-lifetime: a resident CUDA store staying warm
# across CV folds is the point of the cache.
_PROCESS_STORE_CACHE: dict[tuple[str, str | None], tuple[object, tuple[int, int]]] = {}


def _seed_for_slide(seed: int, slide_id) -> int:
    """Stable per-slide seed derived from (collator_seed, slide_id).

    Ensures the same slide gets the same subsample across DataLoader
    workers, across DataLoader rebuilds, and across epochs — so the LP
    callback's score reflects representation drift, not sampling drift.
    """
    import hashlib

    h = hashlib.sha256(f"{int(seed)}:{slide_id}".encode()).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def _spatial_stratified_indices(
    coords: torch.Tensor,
    n_target: int,
    generator: torch.Generator,
    grid_size: int = 32,
) -> torch.Tensor:
    """Sample ``n_target`` indices spread evenly across an ``grid_size`` x
    ``grid_size`` grid over the bag's bounding box.

    Falls back gracefully when bag size <= n_target (returns all indices)
    or when coords are degenerate (single point).
    """
    n = coords.shape[0]
    if n <= n_target:
        return torch.arange(n, dtype=torch.long)

    xy = coords[:, :2].to(torch.float32)
    mn = xy.min(dim=0).values
    mx = xy.max(dim=0).values
    rng = (mx - mn).clamp(min=1.0)

    cells = ((xy - mn) / rng * grid_size).floor().clamp(max=grid_size - 1).to(torch.long)
    cell_id = (cells[:, 0] * grid_size + cells[:, 1]).tolist()

    bins: dict[int, list[int]] = {}
    for i, c in enumerate(cell_id):
        bins.setdefault(c, []).append(i)

    non_empty = [k for k, v in bins.items() if v]
    k = len(non_empty)
    if k == 0:
        return torch.arange(min(n, n_target), dtype=torch.long)

    base = n_target // k
    extra = n_target - base * k

    # Random cell order — distributes the `extra` slots stochastically
    # so we don't always over-sample the same grid cells.
    cell_order = torch.randperm(k, generator=generator).tolist()

    chosen: list[int] = []
    for rank, cell_idx in enumerate(cell_order):
        bucket = bins[non_empty[cell_idx]]
        n_take = base + (1 if rank < extra else 0)
        n_take = min(n_take, len(bucket))
        if n_take == 0:
            continue
        sel = torch.randperm(len(bucket), generator=generator)[:n_take].tolist()
        chosen.extend(bucket[s] for s in sel)

    # Top up with random remaining indices if some cells were too small.
    if len(chosen) < n_target:
        chosen_set = set(chosen)
        remaining = [i for i in range(n) if i not in chosen_set]
        if remaining:
            need = n_target - len(chosen)
            sel = torch.randperm(len(remaining), generator=generator)[:need].tolist()
            chosen.extend(remaining[s] for s in sel)

    return torch.tensor(chosen[:n_target], dtype=torch.long)


class SimpleMILCollator:
    """
    Dynamic collator — pads to the max bag size within each batch.

    No pre-allocation. Used for val/test where batch composition varies
    and the full bag should be used for the best predictions.

    Parameters
    ----------
    max_instances : int or None
        Hard ceiling on the sequence dimension. If a bag exceeds this,
        it is reduced to ``max_instances`` according to ``sampling_mode``.
        None = no ceiling (dynamic padding to batch-max), and
        ``sampling_mode`` is irrelevant.
    sampling_mode : str
        How to choose which tokens to keep when a bag exceeds
        ``max_instances``.
          - ``contiguous`` (default): deterministic first-N. Matches the
            ``contiguous`` pretraining pre-cap; fastest.
          - ``random``: uniform without replacement, deterministic per
            slide via a slide-id-derived seed.
          - ``spatial_stratified``: bucket tokens into a grid over the
            bag's bounding box and sample evenly across cells. Requires
            ``coords`` in the sample dict; falls back to ``random`` when
            coords are absent.
    seed : int
        Base seed combined with each slide_id to derive the per-slide
        RNG. The default keeps eval-time sampling reproducible across
        runs / epochs / workers.
    spatial_grid_size : int
        Grid resolution for ``spatial_stratified`` (default 32 → 1024 cells,
        matching SlideDataset.cap_grid_size used during pretraining).
    """

    def __init__(
        self,
        max_instances: int | None = None,
        sampling_mode: str = "contiguous",
        seed: int = 42,
        spatial_grid_size: int = 32,
    ):
        if sampling_mode not in _VALID_SAMPLING_MODES:
            raise ValueError(f"sampling_mode={sampling_mode!r} not in {_VALID_SAMPLING_MODES}")
        self.max_instances = max_instances
        self.sampling_mode = sampling_mode
        self.seed = int(seed)
        self.spatial_grid_size = int(spatial_grid_size)

    def _select_indices(
        self,
        n: int,
        n_target: int,
        coords: torch.Tensor | None,
        slide_id,
    ) -> torch.Tensor | slice | None:
        if n <= n_target:
            return None
        if self.sampling_mode == "contiguous":
            return slice(0, n_target)

        gen = torch.Generator()
        gen.manual_seed(_seed_for_slide(self.seed, slide_id))

        if self.sampling_mode == "spatial_stratified" and coords is not None:
            return _spatial_stratified_indices(
                coords, n_target, gen, grid_size=self.spatial_grid_size
            )
        # random (also covers spatial_stratified without coords)
        return torch.randperm(n, generator=gen)[:n_target].clone()

    @staticmethod
    def _selected_rows(tensor: torch.Tensor, selection: torch.Tensor | slice | None):
        """Return selected rows without an identity advanced-index copy."""
        if selection is None:
            return tensor
        return tensor[selection]

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)
        feat_dim = batch[0]["features"].shape[1]
        feat_dtype = batch[0]["features"].dtype
        device = batch[0]["features"].device
        has_coords = "coords" in batch[0]

        # Per-sample selection. ``None`` represents the common no-cap/identity
        # case and ``slice`` represents the contiguous cap, avoiding arange and
        # a full advanced-index gather before collation.
        sample_indices: list[torch.Tensor | slice | None] = []
        actual_lengths: list[int] = []
        for s in batch:
            n_full = s["features"].shape[0]
            n_target = n_full if self.max_instances is None else min(n_full, self.max_instances)
            idx = self._select_indices(
                n=n_full,
                n_target=n_target,
                coords=s.get("coords") if has_coords else None,
                slide_id=s.get("slide_id"),
            )
            sample_indices.append(idx)
            actual_lengths.append(int(n_target))
        max_n = max(actual_lengths)
        uniform = all(n == max_n for n in actual_lengths)

        if uniform:
            features = torch.stack(
                [
                    self._selected_rows(sample["features"], selection)
                    for sample, selection in zip(batch, sample_indices, strict=True)
                ]
            )
            mask = None
        else:
            features = torch.empty(B, max_n, feat_dim, dtype=feat_dtype, device=device)
            mask = torch.zeros(B, max_n, dtype=torch.bool, device=device)
            for i, (sample, selection, n) in enumerate(
                zip(batch, sample_indices, actual_lengths, strict=True)
            ):
                features[i, :n] = self._selected_rows(sample["features"], selection)
                if n < max_n:
                    features[i, n:].zero_()
                mask[i, :n] = True

        result = {
            "features": features,
            "mask": mask,
            "labels": torch.tensor([sample["label"] for sample in batch], dtype=torch.long),
            "lengths": torch.tensor(actual_lengths, dtype=torch.int32),
            "slide_ids": [sample["slide_id"] for sample in batch],
        }
        if "weight" in batch[0]:
            result["weights"] = torch.tensor(
                [sample["weight"] for sample in batch], dtype=torch.float32
            )

        if has_coords:
            coord_dim = batch[0]["coords"].shape[-1]
            if uniform:
                coords_t = torch.stack(
                    [
                        self._selected_rows(sample["coords"], selection)
                        for sample, selection in zip(batch, sample_indices, strict=True)
                    ]
                )
            else:
                coords_t = torch.empty(B, max_n, coord_dim, dtype=torch.int32, device=device)
                for i, (sample, selection, n) in enumerate(
                    zip(batch, sample_indices, actual_lengths, strict=True)
                ):
                    coords_t[i, :n] = self._selected_rows(sample["coords"], selection)
                    if n < max_n:
                        coords_t[i, n:].zero_()
            result["coords"] = coords_t

        return result


# ── Length-bucket batch sampler ────────────────────────────────────────────────


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group slides by patch count so that batches have similar lengths.

    Sorts slides by bag size, chunks them into buckets of ``bucket_size``
    slides, shuffles buckets, then yields batches of ``batch_size`` within
    each bucket.  This minimises wasted padding while preserving
    stochasticity across epochs.

    Compatible with ``WeightedRandomSampler`` via *pre-sorted indices* — the
    sampler itself handles shuffling, so set ``shuffle=False`` on DataLoader.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        bucket_size: int = 64,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.bucket_size = max(bucket_size, batch_size)
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        # Sort by length, chunk into buckets, shuffle within bucket
        order = np.argsort(self.lengths)
        n = len(order)
        # Chunk into buckets
        buckets = [order[i : i + self.bucket_size] for i in range(0, n, self.bucket_size)]
        # Shuffle bucket order
        self.rng.shuffle(buckets)
        for bucket in buckets:
            # Shuffle within bucket
            self.rng.shuffle(bucket)
            # Yield batches
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size].tolist()
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


# ── Distributed sampler wrapper ───────────────────────────────────────────────


class DistributedSamplerWrapper(DistributedSampler):
    """Shard the output of an arbitrary inner Sampler across DDP ranks.

    Replays the inner sampler at every ``__iter__`` so a stochastic inner
    sampler (e.g. ``WeightedRandomSampler``) draws a fresh epoch each time.
    The inner indices are then sliced rank-wise via the standard
    ``DistributedSampler`` machinery — each rank sees a disjoint chunk and
    the per-epoch sample budget across all ranks matches the inner sampler's
    own budget (no replication).

    Required for DDP because Lightning will not auto-wrap a user-provided
    sampler with ``DistributedSampler``; without this wrapper every rank
    would draw the same global indices and effectively replicate the data.
    """

    def __init__(
        self,
        sampler: Sampler,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset=range(len(sampler)),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.inner_sampler = sampler

    def __iter__(self):
        # One full pass of the inner sampler this epoch.
        inner_indices = list(self.inner_sampler)
        for pos in super().__iter__():
            yield inner_indices[pos]


def _ddp_active() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


# ── Dataset ───────────────────────────────────────────────────────────────────


H5_FEATURES_KEY = "features"
H5_COORDS_KEY = "coords"


class SlideDataset(Dataset):
    """Supervised slide-level dataset over per-slide H5 feature files.

    Each slide is one ``{feature_dir}/{slide_id}.h5`` (the extraction-stage
    artifact) holding a ``features`` [N, D] and a ``coords`` [N, 2] dataset.
    Files are opened lazily inside ``__getitem__`` so no handles cross the
    DataLoader fork boundary.

    Subsampling contract
    ════════════════════════════════════════════════════════════════════
    ALL patch-level subsampling happens in the DATA LAYER (here + collator).
    The model receives exactly what it gets — no further truncation.

      - max_instances (dataset level):
          Train: stochastic random sampling (different view each epoch)
          Val/Test: deterministic first-N (reproducible)
      - instance_dropout: randomly drops patches AFTER subsampling (train only)
      - The collator then pads/truncates to a uniform batch dimension.

    Parameters
    ----------
    feature_dir : str
        Directory of per-slide H5 feature files for one encoder
        (TRIDENT's ``features_{encoder}`` output directory).
    slide_ids : list[str] or None
        Slide IDs to include (subset for this split/fold). None = every
        H5 file present in ``feature_dir``.
    labels : dict[str, int]
        Mapping slide_id -> integer label.
    max_instances : int or None
        Cap patches per slide at load time. None = use all patches.
        Train: stochastic random sampling. Val/Test: deterministic first-N.
    is_train : bool
        If True, subsampling is stochastic and augmentations apply.
    cap_strategy : str
        Subsampling strategy when max_instances < n_patches.
        "random": random permutation (train) or first-K (val/test).
        "spatial_stratified": grid-proportional sampling using coords.
    cap_grid_size : int
        Grid resolution for spatial_stratified sampling (default 32).
    instance_dropout : float
        Probability of dropping each patch after subsampling (train only).
    feature_noise_std : float
        Std of additive Gaussian noise on features (train only).
    cache_size_mb : int
        LRU cache size in MB for val/test. 0 = disabled. Ignored for train.
    return_coords : bool
        Whether to return spatial coordinates alongside features.
    force_float32 : bool
        If True, cast features to float32 in __getitem__. Default True to
        match supervised collators that pre-allocate float32 buffers.
    eval_crop_seed : int or None
        Seed for deterministic eval subsampling. When set, val/test
        subsampling uses this seed (+ slide index) for reproducible but
        non-trivial sampling. Different seeds produce different crops,
        enabling multi-crop evaluation. None = deterministic first-K behaviour
        for "random" strategy; for "spatial_stratified" defaults to 0.
    store : PackedFeatureStore or None
        When provided, features are read from a flat packed memmap instead of
        per-slide H5 files. This is the fast path: no h5py decode, no per-item
        file open, and subsampling gathers ONLY the selected rows out of the
        memmap rather than reading the whole slide and discarding most of it.
        ``feature_dir`` is still required (it identifies the inventory) but is
        not read from.
    fixed_bag_size : int or None
        When set, every sample returns EXACTLY this many patches, so batches
        are uniform and need no padding or masking. Slides with more patches
        are subsampled per ``cap_strategy``; slides with fewer are handled per
        ``short_bag_policy``. This is what makes ``batch_size > 1`` free of
        padding waste and makes shapes static enough for ``torch.compile``.
        Takes precedence over ``max_instances``.
    short_bag_policy : str
        What to do when a slide has fewer patches than ``fixed_bag_size``.
        ``"repeat"`` (default) resamples with replacement to fill the bag,
        keeping every batch uniform. ``"pad"`` returns the short bag as-is and
        lets the collator zero-pad and mask it, which is exact but gives up
        the uniform-batch fast path.
    sample_weights : dict[str, float] or None
        Per-slide loss weight, e.g. ``1 / n_slides`` for the slide's patient
        so that a patient contributing several blocks does not outvote a
        patient with one. Slides absent from the mapping weigh 1.0. None
        disables weighting entirely (no ``weight`` key is emitted, so the
        collators and the loss keep their unweighted fast path).
    """

    def __init__(
        self,
        feature_dir: str,
        slide_ids: list[str] | None = None,
        labels: dict[str, int] | None = None,
        max_instances: int | None = None,
        is_train: bool = True,
        cap_strategy: str = "random",
        cap_grid_size: int = 32,
        instance_dropout: float = 0.0,
        feature_noise_std: float = 0.0,
        cache_size_mb: int = 0,
        return_coords: bool = False,
        force_float32: bool = True,
        eval_crop_seed: int | None = None,
        store=None,
        fixed_bag_size: int | None = None,
        short_bag_policy: str = "repeat",
        sample_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.feature_dir = Path(feature_dir)
        if not self.feature_dir.is_dir():
            raise FileNotFoundError(f"Feature directory does not exist: {self.feature_dir}")
        if short_bag_policy not in ("repeat", "pad"):
            raise ValueError(f"short_bag_policy={short_bag_policy!r} not in ('repeat', 'pad')")
        if fixed_bag_size is not None and fixed_bag_size < 1:
            raise ValueError(f"fixed_bag_size must be >= 1, got {fixed_bag_size}")
        if fixed_bag_size is not None and instance_dropout > 0:
            raise ValueError(
                "fixed_bag_size and instance_dropout are mutually exclusive: dropout would "
                "make bag sizes ragged again, defeating the uniform-batch fast path. The "
                "stochastic fixed-size subsample already provides the same regularisation."
            )
        # Seeded from the process torch seed so num_workers=0 runs (the
        # resident-store mode) are reproducible under seed_everything. With
        # workers > 0 this is overridden per worker by _worker_init_fn.
        self.rng = np.random.default_rng(torch.initial_seed() & 0xFFFF_FFFF)
        self.max_instances = max_instances
        self.is_train = is_train
        self.cap_strategy = cap_strategy
        self.cap_grid_size = cap_grid_size
        self.instance_dropout = instance_dropout
        self.feature_noise_std = feature_noise_std
        self.return_coords = return_coords
        self.force_float32 = force_float32
        self.eval_crop_seed = eval_crop_seed
        self.store = store
        self.fixed_bag_size = fixed_bag_size
        self.short_bag_policy = short_bag_policy

        labels = labels or {}

        self.slide_ids: list[str] = []
        self.paths: list[Path] = []
        self.lengths: list[int] = []
        self.positions: list[int] = []
        self._feat_dim: int | None = None

        missing = []
        if store is not None:
            # Packed path: the index already carries every length and the
            # feature dim, so setup opens no slide files at all.
            needs_coords = return_coords or cap_strategy == "spatial_stratified"
            if needs_coords and not store.has_coords:
                raise ValueError(
                    f"Packed store {store.pack_dir} has no coords, but "
                    f"return_coords={return_coords} / cap_strategy={cap_strategy!r} "
                    "requires them. Re-pack with include_coords=True."
                )
            self._feat_dim = int(store.feat_dim)
            if slide_ids is None:
                slide_ids = list(store.slide_ids)
            for sid in slide_ids:
                sid = normalize_slide_id(sid)
                if sid not in store:
                    missing.append(sid)
                    continue
                pos = store.position(sid)
                n_patches = store.length_of(sid)
                if n_patches == 0:
                    logger.warning("Slide '%s' has 0 patches — skipping.", sid)
                    continue
                self.slide_ids.append(sid)
                self.positions.append(pos)
                self.lengths.append(int(n_patches))
                self.paths.append(self.feature_dir / f"{sid}.h5")
        else:
            available = {
                normalize_slide_id(path.name): path
                for path in sorted(self.feature_dir.glob("*.h5"))
            }
            if slide_ids is None:
                slide_ids = list(available)

            for sid in slide_ids:
                path = available.get(sid)
                if path is None:
                    missing.append(sid)
                    continue
                with h5py.File(path, "r") as h5:
                    n_patches, feat_dim = h5[H5_FEATURES_KEY].shape
                if n_patches == 0:
                    logger.warning("Slide '%s' has 0 patches — skipping.", sid)
                    continue
                if self._feat_dim is None:
                    self._feat_dim = int(feat_dim)
                elif int(feat_dim) != self._feat_dim:
                    raise ValueError(
                        f"Inconsistent feature dim in {path.name}: "
                        f"{feat_dim} != {self._feat_dim} (mixed encoders in one directory?)"
                    )
                self.slide_ids.append(sid)
                self.paths.append(path)
                self.positions.append(len(self.positions))
                self.lengths.append(int(n_patches))

        if missing:
            logger.warning(
                "%d/%d slide_ids have no H5 in %s. First 5: %s",
                len(missing),
                len(slide_ids),
                self.feature_dir,
                missing[:5],
            )

        self.labels_list: list[int] = [labels.get(sid, -1) for sid in self.slide_ids]
        self.weights_list: list[float] | None = (
            [float(sample_weights.get(sid, 1.0)) for sid in self.slide_ids]
            if sample_weights
            else None
        )
        n_unlabeled = sum(1 for label in self.labels_list if label < 0)
        if n_unlabeled > 0:
            logger.warning(
                "%d/%d slides have no label (label=-1). Check your CSV mapping.",
                n_unlabeled,
                len(self.labels_list),
            )

        logger.info(
            "SlideDataset: %d slides, feat_dim=%s, max_instances=%s, fixed_bag_size=%s, "
            "cap_strategy=%s, is_train=%s, source=%s",
            len(self.slide_ids),
            self._feat_dim,
            self.max_instances,
            self.fixed_bag_size,
            self.cap_strategy,
            self.is_train,
            "packed" if store is not None else "h5",
        )

        # LRU cache (val/test only — train is stochastic, caching freezes
        # augmentation). Pointless for a resident store: reads are index ops
        # on memory already held, so caching would only duplicate it.
        self._cache: OrderedDict | None = None
        self._cache_max_bytes = int(cache_size_mb * 1e6) if cache_size_mb > 0 else 0
        self._cache_current_bytes = 0
        if self._cache_max_bytes > 0 and not is_train and getattr(store, "device", None) is None:
            self._cache = OrderedDict()

    @property
    def feat_dim(self) -> int:
        if self._feat_dim is None:
            raise RuntimeError(f"Empty dataset — no H5 features matched in {self.feature_dir}")
        return self._feat_dim

    def __len__(self) -> int:
        return len(self.slide_ids)

    def _target_bag_size(self) -> int | None:
        """Rows this dataset wants per slide (fixed_bag_size wins over the cap)."""
        if self.fixed_bag_size is not None:
            return self.fixed_bag_size
        return self.max_instances

    def _fill_indices(self, n_patches: int, target: int, idx: int) -> np.ndarray:
        """Indices that grow a short bag to exactly ``target`` rows.

        Every real patch is kept once; the shortfall is drawn with
        replacement. Duplicating a patch in an attention pool is equivalent
        to giving it proportionally more mass, which is the accepted cost of
        keeping every batch the same shape.
        """
        rng = self.rng if self.is_train else np.random.default_rng((self.eval_crop_seed or 0) + idx)
        extra = rng.choice(n_patches, size=target - n_patches, replace=True)
        return np.concatenate([np.arange(n_patches), extra])

    def _read_packed(
        self,
        idx: int,
        need_coords: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Select rows first, then gather only those rows from the memmap."""
        pos = self.positions[idx]
        n_patches = self.lengths[idx]
        target = self._target_bag_size()

        rows: np.ndarray | None = None
        if target is not None and n_patches > target:
            if self.cap_strategy == "spatial_stratified":
                coords_full = self.store.read_coords(pos, None)
                rows = self._spatial_stratified_indices(coords_full, n_patches, target, idx)
            else:
                rows = self._random_indices(n_patches, target, idx)
            # Sorting costs microseconds and turns a scattered gather into a
            # near-sequential one over the memmap. A bag is an unordered set
            # for every aggregator here, so the order carries no information.
            rows = np.asarray(rows)
            rows.sort()
        elif (
            self.fixed_bag_size is not None
            and n_patches < self.fixed_bag_size
            and self.short_bag_policy == "repeat"
        ):
            rows = self._fill_indices(n_patches, self.fixed_bag_size, idx)
            rows.sort()

        features = self.store.read_features(pos, rows)
        coords = self.store.read_coords(pos, rows) if need_coords else None
        return features, coords

    def _read_h5(
        self,
        idx: int,
        need_coords: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Read the whole slide, then slice in numpy.

        h5py fancy indexing is roughly an order of magnitude slower than
        reading the dataset whole and slicing, so partial reads are NOT an
        optimisation here — that is what the packed store is for.
        """
        with h5py.File(self.paths[idx], "r") as h5:
            features = h5[H5_FEATURES_KEY][...]
            coords = h5[H5_COORDS_KEY][...] if need_coords else None
        n_patches = features.shape[0]
        target = self._target_bag_size()

        if target is not None and n_patches > target:
            features, coords = self._subsample(features, coords, n_patches, idx, k=target)
        elif (
            self.fixed_bag_size is not None
            and n_patches < self.fixed_bag_size
            and self.short_bag_policy == "repeat"
        ):
            rows = self._fill_indices(n_patches, self.fixed_bag_size, idx)
            features = features[rows]
            if coords is not None:
                coords = coords[rows]
        return features, coords

    def __getitem__(self, idx: int) -> dict:
        """Load one slide's features + label.

        Pipeline: select rows -> read -> augment -> tensor conversion

        Returns dict: features [N, D], label, slide_id, length, [coords]
        """
        if self._cache is not None and idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        slide_id = self.slide_ids[idx]
        label = self.labels_list[idx]

        # 1-2. Read + subsample (order depends on whether partial reads are cheap)
        need_coords = self.return_coords or self.cap_strategy == "spatial_stratified"
        if self.store is not None:
            features, coords = self._read_packed(idx, need_coords)
        else:
            features, coords = self._read_h5(idx, need_coords)
        n_patches = features.shape[0]

        # 3. Augmentation (train only)
        if self.is_train:
            features, coords, n_patches = self._augment(features, coords, n_patches)

        # 4. Convert to tensors. A resident store already returns torch
        # tensors on its device — no numpy round-trip, no host copy.
        if isinstance(features, torch.Tensor):
            features_t = features.float() if self.force_float32 else features
        else:
            features = np.ascontiguousarray(features)
            if self.force_float32:
                features = features.astype(np.float32, copy=False)
            features_t = torch.from_numpy(features)

        result = {
            "features": features_t,
            "label": label,
            "slide_id": slide_id,
            "length": n_patches,
        }
        if self.weights_list is not None:
            result["weight"] = self.weights_list[idx]
        if self.return_coords and coords is not None:
            if isinstance(coords, torch.Tensor):
                result["coords"] = coords.to(torch.int32)
            else:
                result["coords"] = torch.from_numpy(np.ascontiguousarray(coords, dtype=np.int32))

        # Cache (val/test only)
        if self._cache is not None:
            item_bytes = features_t.nbytes
            if "coords" in result:
                item_bytes += result["coords"].nbytes
            while self._cache_current_bytes + item_bytes > self._cache_max_bytes and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._cache_current_bytes -= evicted["features"].nbytes
                if "coords" in evicted:
                    self._cache_current_bytes -= evicted["coords"].nbytes
            if item_bytes <= self._cache_max_bytes:
                self._cache[idx] = result
                self._cache_current_bytes += item_bytes

        return result

    # -- Subsampling -----------------------------------------------------------

    def _subsample(
        self,
        features: np.ndarray,
        coords: np.ndarray | None,
        n_patches: int,
        idx: int = 0,
        k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Subsample to ``k`` patches (default: self.max_instances).

        Dispatches to random or spatial-stratified strategy.
        """
        k = self.max_instances if k is None else k
        if self.cap_strategy == "spatial_stratified":
            indices = self._spatial_stratified_indices(coords, n_patches, k, idx)
        else:
            indices = self._random_indices(n_patches, k, idx)
        features = features[indices]
        if coords is not None:
            coords = coords[indices]
        return features, coords

    def _random_indices(self, n_patches: int, k: int, idx: int) -> np.ndarray:
        """Uniform subsampling without constructing an O(N) permutation."""
        if self.is_train:
            return self.rng.choice(n_patches, size=k, replace=False, shuffle=False)
        if self.eval_crop_seed is not None:
            rng = np.random.default_rng(self.eval_crop_seed + idx)
            return rng.choice(n_patches, size=k, replace=False, shuffle=False)
        return np.arange(k)

    def _spatial_stratified_indices(
        self,
        coords: np.ndarray,
        n_patches: int,
        k: int,
        idx: int,
    ) -> np.ndarray:
        """Grid-proportional spatial sampling.

        Divides the coordinate space into a ``cap_grid_size x cap_grid_size``
        grid, then allocates a quota to each occupied cell proportional to
        its patch count.  Within each cell the selection is random (train)
        or deterministic (val/test, seeded by ``eval_crop_seed + idx``).
        """
        gs = self.cap_grid_size
        if isinstance(coords, torch.Tensor):
            # Resident-store path: coords arrive as a device tensor. The
            # selection logic is index arithmetic on ~200 KB — one D2H copy
            # here is far cheaper than porting the allocator to torch.
            coords = coords.detach().cpu().numpy()
        xy = coords[:n_patches, :2].astype(np.float64)

        # Normalise coords to [0, gs) grid indices
        lo = xy.min(axis=0)
        span = xy.max(axis=0) - lo
        span[span == 0] = 1.0  # degenerate axis
        normed = (xy - lo) / span * (gs - 1e-9)
        gx = normed[:, 0].astype(np.int32)
        gy = normed[:, 1].astype(np.int32)
        cell_ids = gx * gs + gy

        if self.is_train:
            rng = self.rng
        else:
            seed = (self.eval_crop_seed if self.eval_crop_seed is not None else 0) + idx
            rng = np.random.default_rng(seed)

        # Group patches by cell AND pre-shuffle within each cell in one
        # lexsort over (cell_id, random_key). Per-cell selection then reduces
        # to "keep the first quota_i rows of each group" — pure vectorized
        # arithmetic, no Python loop over cells and no per-cell rng.choice.
        # (The previous per-cell np.where scan was O(N * occupied_cells);
        # the intermediate argsort+choice version still paid ~1k Generator
        # calls per sample.)
        # One argsort, not a two-key lexsort: cell ids are integers and the
        # random tiebreak lives in [0, 1), so cell_id + key orders by cell
        # first and uniformly at random within each cell.
        order = np.argsort(cell_ids + rng.random(n_patches))
        _, first = np.unique(cell_ids[order], return_index=True)

        # Allocate quotas proportional to cell size
        cell_sizes = np.diff(np.append(first, n_patches))
        quotas = np.round(cell_sizes / n_patches * k).astype(int)
        # Clamp each quota to actual cell size
        quotas = np.minimum(quotas, cell_sizes)
        # Adjust total to exactly k
        diff = int(k - quotas.sum())
        if diff > 0:
            # Add to cells that still have room, largest-room-first
            room = cell_sizes - quotas
            by_room = np.argsort(-room)
            for i in by_room:
                add = min(diff, int(room[i]))
                quotas[i] += add
                diff -= add
                if diff == 0:
                    break
        elif diff < 0:
            # Shed the surplus one patch at a time, visiting cells in a random
            # (train) or seeded (eval) order so the loss is spread evenly and
            # does not always fall on the same corner of the slide.
            #
            # A quota MAY reach zero here. The previous implementation refused
            # to take a cell below 1, which made the whole adjustment a no-op
            # whenever `k` was smaller than the number of occupied cells (every
            # quota is already 1, so `min(surplus, quota - 1)` is 0) — and the
            # function then returned MORE than `k` indices, silently violating
            # the cap. When k < n_occupied_cells, some cells must go
            # unrepresented; there is no allocation that keeps them all.
            surplus = -diff
            visit = rng.permutation(len(quotas))
            cursor = 0
            while surplus > 0:
                i = visit[cursor % len(visit)]
                if quotas[i] > 0:
                    quotas[i] -= 1
                    surplus -= 1
                cursor += 1

        # Take the first quota_i rows of each group. Within-cell order is
        # already random via the lexsort key, so this IS the uniform
        # without-replacement draw.
        rank = np.arange(n_patches) - np.repeat(first, cell_sizes)
        chosen = order[rank < np.repeat(quotas, cell_sizes)]
        return chosen if chosen.size else np.arange(min(k, n_patches))

    # -- Augmentation ----------------------------------------------------------

    def _augment(
        self,
        features,
        coords,
        n_patches: int,
    ) -> tuple[Any, Any, int]:
        """Apply instance dropout and feature noise (train only).

        Works on numpy arrays (memmap/h5 path) and torch tensors (resident
        store path). Noise is added IN the stored dtype: upcasting a float16
        bag to float32 here would double every byte crossing the worker IPC
        queue and the PCIe bus, defeating the entire packed-FP16 design.
        """
        is_tensor = isinstance(features, torch.Tensor)

        if self.instance_dropout > 0 and n_patches > 1:
            keep = self.rng.random(n_patches) > self.instance_dropout
            if not keep.any():
                keep[self.rng.integers(n_patches)] = True
            if is_tensor:
                idx = torch.from_numpy(np.flatnonzero(keep)).to(features.device)
                features = features.index_select(0, idx)
                if coords is not None:
                    coords = coords.index_select(0, idx)
            else:
                features = features[keep]
                if coords is not None:
                    coords = coords[keep]
            n_patches = int(keep.sum())

        if self.feature_noise_std > 0:
            if is_tensor:
                # Device-side noise; uses torch's global RNG (seeded by
                # seed_everything) rather than self.rng — generating on the
                # host and shipping it over PCIe would cost more than the bag.
                features = features + torch.randn_like(features) * self.feature_noise_std
            else:
                # float32 draw (no float64 intermediate), scaled in place, then
                # added into the existing array so float16 stays float16.
                noise = self.rng.standard_normal(features.shape, dtype=np.float32)
                noise *= self.feature_noise_std
                np.add(features, noise, out=features, casting="same_kind")

        return features, coords, n_patches

    # -- Utilities -------------------------------------------------------------

    def get_bag_sizes(self) -> np.ndarray:
        """Return array of patch counts per slide."""
        return np.array(self.lengths)

    def get_label_counts(self) -> dict[int, int]:
        """Return {label: count} for all slides in this dataset."""
        return dict(Counter(self.labels_list))

    def get_all_labels(self) -> np.ndarray:
        """Return flat array of all labels (for sampler construction)."""
        return np.array(self.labels_list)


# ── DataModule ────────────────────────────────────────────────────────────────


class MILDataModule(L.LightningDataModule):
    """
    Lightning DataModule for MIL training from per-slide H5 feature files.

    Parameters
    ----------
    feature_dir : str
        Directory of per-slide H5 feature files for one encoder
        (TRIDENT's ``features_{encoder}`` output directory).
    splits_dir : str
        Path containing splits.parquet.
    csv_path : str
        Manifest CSV with labels.
    label_column : str
        Column name for integer labels.
    filename_column : str
        Column used to derive slide_id (stem of filename).
    scheme : str
        Split scheme (kfold, holdout, custom_kfold, etc.).
    fold : int
        Current fold index.
    test_slide_ids : list[str] or None
        Explicit held-out IDs for test/predict. When None, use the IDs from
        the configured split scheme.
    batch_size : int
        Training batch size.
    max_instances : int or None
        Collation cap — hard ceiling on padded tensor dimension.
        Both train and eval. Prevents OOM on large bags.
    dataset_max_instances : int or None
        Dataset-level subsampling. TRAIN ONLY.
        None = no subsampling (dataset returns all patches).
    num_workers : int
        DataLoader workers.
    prefetch_factor : int
        Number of prefetched batches per worker when num_workers > 0.
    pin_memory : bool or None
        Whether to pin host-memory batches for faster H2D copies.
        None = auto-detect CUDA safety.
    persistent_workers : bool or None
        Keep DataLoader workers alive across epochs. None = enabled iff
        num_workers > 0.
    class_weighted_sampling : bool
        Inverse-frequency weighted random sampling for training.
    train_sampling_strategy : str
        Training-only patient sampling policy. ``slide_uniform`` preserves
        the historical shuffled-slide loader. ``patient_natural`` samples
        patients uniformly, ``cohort_balanced`` gives every cohort equal
        mass while preserving its empirical target prevalence, and
        ``cohort_label_balanced`` gives every cohort equal mass and fixes its
        positive prevalence to ``sampling_target_positive_prevalence``.
        Patient mass is divided across that patient's slides.
    instance_dropout : float
        Patch dropout rate (train only).
    feature_noise_std : float
        Feature noise std (train only).
    cache_size_mb : int
        LRU cache size per dataset (val/test only).
    return_coords : bool
        Return spatial coordinates.
    verify_splits : bool
        Verify split integrity hash before loading.
    use_preallocated_collator : bool
        Use pre-allocated collation buffers for training.
    force_float32 : bool
        Cast features to fp32 in the dataset. False preserves the stored
        dtype (e.g. float16) to reduce RAM usage in the worker queue.
        Set this False when reading a float16 packed store: upcasting in the
        worker doubles the bytes crossing the IPC queue and the PCIe bus for
        no gain, since the model casts anyway.
    packed_dir : str or None
        Directory produced by ``scripts/pack_features.py`` (or the RAS study's
        ``scripts/pack_features.py``). When set, every
        split reads from the flat packed memmap instead of per-slide H5
        files. ``feature_dir`` must still point at the source H5 directory —
        it identifies the inventory the pack is verified against.
    verify_packed_source : bool
        Refuse to train when the pack's recorded source inventory hash does
        not match the live feature directory (i.e. the pack is stale).
    resident_device : str or None
        When set ("cuda" or "cpu"), the packed store is loaded ONCE into a
        single tensor on that device and every read becomes a device-side
        gather — no DataLoader workers, no IPC, no per-step host-to-device
        copy. Requires ``packed_dir``; forces ``num_workers=0`` and
        ``pin_memory=False``. "cuda" needs the pack to fit in VRAM beside
        the model.
    fixed_bag_size : int or None
        Train-split bag size. When set, every training sample is exactly this
        many patches, making batches uniform, padding-free, and static in
        shape. None = current behaviour (ragged bags capped by
        ``dataset_max_instances`` / ``max_instances``).
    eval_fixed_bag_size : int or None
        Same for val/test. Deliberately separate and defaulting to None so
        capping the training signal never silently caps evaluation.
    short_bag_policy : str
        ``"repeat"`` or ``"pad"`` — how slides shorter than the fixed bag
        size are handled. See :class:`SlideDataset`.
    eval_batch_size : int or None
        Batch size for val/test. None = same as ``batch_size``. When > 1,
        eval slides are visited in bag-size order so batches pad very little.
    drop_last : bool
        Drop the final partial training batch. This keeps the batch dimension
        static for fixed-bag compiled training.
    """

    def __init__(
        self,
        feature_dir: str,
        splits_dir: str,
        csv_path: str,
        label_column: str = "label",
        filename_column: str = "filename",
        patient_column: str | None = None,
        scheme: str = "kfold",
        fold: int = 0,
        test_slide_ids: list[str] | None = None,
        batch_size: int = 1,
        max_instances: int | None = None,
        dataset_max_instances: int | None = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        class_weighted_sampling: bool = True,
        train_sampling_strategy: str = "slide_uniform",
        cohort_column: str | None = None,
        sampling_target_positive_prevalence: float = 0.4,
        sampling_seed: int = 42,
        instance_dropout: float = 0.0,
        feature_noise_std: float = 0.0,
        cache_size_mb: int = 0,
        return_coords: bool = False,
        verify_splits: bool = True,
        use_preallocated_collator: bool = True,
        force_float32: bool = True,
        refit_mode: bool = False,
        cap_strategy: str = "random",
        cap_grid_size: int = 32,
        eval_n_crops: int = 1,
        length_bucket: bool = False,
        length_bucket_size: int = 64,
        packed_dir: str | None = None,
        verify_packed_source: bool = True,
        resident_device: str | None = None,
        fixed_bag_size: int | None = None,
        eval_fixed_bag_size: int | None = None,
        short_bag_policy: str = "repeat",
        eval_batch_size: int | None = None,
        drop_last: bool = False,
        eval_full_bags: bool = False,
        sample_weight_column: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")
        if train_sampling_strategy not in _TRAIN_SAMPLING_STRATEGIES:
            raise ValueError(
                f"train_sampling_strategy={train_sampling_strategy!r} not in "
                f"{sorted(_TRAIN_SAMPLING_STRATEGIES)}"
            )
        if class_weighted_sampling and train_sampling_strategy != "slide_uniform":
            raise ValueError(
                "class_weighted_sampling and patient-level train_sampling_strategy "
                "are mutually exclusive"
            )
        if train_sampling_strategy != "slide_uniform":
            if patient_column is None:
                raise ValueError(
                    f"train_sampling_strategy={train_sampling_strategy!r} requires patient_column"
                )
            if cohort_column is None:
                raise ValueError(
                    f"train_sampling_strategy={train_sampling_strategy!r} requires cohort_column"
                )
            if length_bucket:
                raise ValueError(
                    "Patient-level train_sampling_strategy is incompatible with "
                    "length_bucket=True because the bucket sampler would replace it"
                )
        if not 0.0 < float(sampling_target_positive_prevalence) < 1.0:
            raise ValueError("sampling_target_positive_prevalence must be in (0, 1)")
        if resident_device is not None:
            if packed_dir is None:
                raise ValueError(
                    "resident_device requires packed_dir: only a packed store can be "
                    "loaded resident. Run scripts/pack_features.py first."
                )
            if num_workers != 0:
                # CUDA tensors cannot cross a fork, and a resident store makes
                # workers pointless: the per-sample read is an index op, not I/O.
                logger.info(
                    "resident_device=%s: overriding num_workers %d -> 0",
                    resident_device,
                    num_workers,
                )
                num_workers = 0
            # Pinning would try to pin device tensors (an error) and buys
            # nothing when batches are already on / headed straight to device.
            pin_memory = False

        self.feature_dir = feature_dir
        self.splits_dir = splits_dir
        self.csv_path = csv_path
        self.label_column = label_column
        self.filename_column = filename_column
        self.patient_column = patient_column
        self.cohort_column = cohort_column
        self.scheme = scheme
        self.fold = fold
        self.test_slide_ids = test_slide_ids
        self.batch_size = batch_size
        self.max_instances = max_instances
        self.dataset_max_instances = dataset_max_instances
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.class_weighted_sampling = class_weighted_sampling
        self.train_sampling_strategy = train_sampling_strategy
        self.sampling_target_positive_prevalence = float(sampling_target_positive_prevalence)
        self.sampling_seed = int(sampling_seed)
        self.instance_dropout = instance_dropout
        self.feature_noise_std = feature_noise_std
        self.cache_size_mb = cache_size_mb
        self.return_coords = return_coords
        self.verify_splits = verify_splits
        self.use_preallocated_collator = use_preallocated_collator
        self.force_float32 = force_float32
        self.refit_mode = refit_mode
        self.cap_strategy = cap_strategy
        self.cap_grid_size = cap_grid_size
        self.eval_n_crops = eval_n_crops
        self.length_bucket = length_bucket
        self.length_bucket_size = length_bucket_size
        self.packed_dir = packed_dir
        self.verify_packed_source = verify_packed_source
        self.resident_device = resident_device
        self.fixed_bag_size = fixed_bag_size
        # Eval defaults to full bags even when training is capped: the cap is a
        # regulariser for the training signal, not a property of the task.
        self.eval_fixed_bag_size = eval_fixed_bag_size
        self.short_bag_policy = short_bag_policy
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.drop_last = bool(drop_last)
        # Eval on ALL patches regardless of the training-time dataset cap
        # (dataset_max_instances): the cap is a train-time regulariser; a
        # study whose protocol scores full bags sets this true.
        self.eval_full_bags = bool(eval_full_bags)
        self._store = None
        self.train_dataset: SlideDataset | None = None
        self.val_dataset: SlideDataset | None = None
        self.test_dataset: SlideDataset | None = None
        self._labels_map: dict | None = None
        self._patient_map: dict | None = None
        # Per-slide loss weights (e.g. 1/n_slides for the patient). TRAIN only:
        # validation and test metrics are aggregated to the patient level
        # downstream, so re-weighting them would double-count the correction.
        self.sample_weight_column = sample_weight_column
        self._sample_weights_map: dict[str, float] | None = None
        self._sampling_summary: dict[str, Any] | None = None
        self._train_sampler: Sampler | None = None
        self._feat_dim: int | None = None

    def _get_store(self):
        """Open the packed feature store once per PROCESS, if configured.

        Returns None when ``packed_dir`` is unset, in which case every split
        falls back to reading per-slide H5 files.

        The store is cached at module level keyed by (pack_dir, device): a CV
        run constructs a fresh datamodule per fold, and without the cache a
        ``resident_device="cuda"`` run would reload the multi-GB pack into
        VRAM once per fold. Cache hits re-verify cheaply: the features.bin
        (size, mtime) stamp catches a rebuilt pack, and the live-inventory
        hash check is identical to what a fresh open would enforce.
        """
        if self.packed_dir is None:
            return None
        if self._store is None:
            from oceanpath.datasets.packed import (
                FEATURES_FILE,
                PackedFeatureStore,
                PackedStoreError,
                ResidentPackedStore,
                feature_inventory_sha256,
            )

            verify = (
                feature_inventory_sha256(Path(self.feature_dir))
                if self.verify_packed_source
                else None
            )
            key = (str(Path(self.packed_dir).resolve()), self.resident_device)
            feat_path = Path(self.packed_dir) / FEATURES_FILE
            stamp = None
            if feat_path.is_file():
                stat = feat_path.stat()
                stamp = (stat.st_size, stat.st_mtime_ns)
            # A missing features.bin falls through to PackedFeatureStore,
            # which raises the actionable "build one with pack_features" error.

            cached = _PROCESS_STORE_CACHE.get(key)
            if cached is not None and stamp is not None and cached[1] == stamp:
                store = cached[0]
                if verify is not None and verify != store.meta.source_inventory_sha256:
                    raise PackedStoreError(
                        f"Packed store at {self.packed_dir} is STALE: built from "
                        f"inventory {store.meta.source_inventory_sha256[:12]}, live "
                        f"inventory is {verify[:12]}. Re-run scripts/pack_features.py."
                    )
                logger.info("Reusing process-cached packed store %s", self.packed_dir)
            else:
                store = PackedFeatureStore(self.packed_dir, verify_source=verify)
                if self.resident_device is not None:
                    store = ResidentPackedStore(store, device=self.resident_device)
                if stamp is not None:
                    _PROCESS_STORE_CACHE[key] = (store, stamp)
            self._store = store
            logger.info(
                "Using packed feature store %s (%d slides, %d patches, %s%s)",
                self.packed_dir,
                self._store.meta.n_slides,
                self._store.meta.total_patches,
                self._store.meta.feat_dtype,
                f", resident on {self.resident_device}" if self.resident_device else "",
            )
        return self._store

    @property
    def feat_dim(self) -> int:
        if self._feat_dim is None:
            store = self._get_store()
            if store is not None:
                self._feat_dim = int(store.feat_dim)
                return self._feat_dim
            first = next(iter(sorted(Path(self.feature_dir).glob("*.h5"))), None)
            if first is None:
                raise FileNotFoundError(f"No .h5 feature files in {self.feature_dir}")
            with h5py.File(first, "r") as h5:
                self._feat_dim = int(h5[H5_FEATURES_KEY].shape[1])
        return self._feat_dim

    @property
    def num_classes(self) -> int:
        labels = self._load_labels()
        return len(set(labels.values()))

    def _load_labels(self) -> dict[str, int]:
        if self._labels_map is not None:
            return self._labels_map
        df = pd.read_csv(self.csv_path)
        if self.filename_column not in df.columns:
            raise ValueError(
                f"filename_column='{self.filename_column}' not found. Columns: {list(df.columns)}"
            )
        if self.label_column not in df.columns:
            raise ValueError(
                f"label_column='{self.label_column}' not found. Columns: {list(df.columns)}"
            )
        if df[self.filename_column].isna().any() or df[self.label_column].isna().any():
            raise ValueError(
                f"Manifest columns '{self.filename_column}' and '{self.label_column}' "
                "must not contain missing values"
            )
        df["slide_id"] = df[self.filename_column].astype(str).map(normalize_slide_id)
        labels = df[self.label_column].astype(int)
        conflicting = (
            pd.DataFrame({"slide_id": df["slide_id"], "label": labels})
            .groupby("slide_id")["label"]
            .nunique()
        )
        conflicting_ids = conflicting[conflicting > 1].index.tolist()
        if conflicting_ids:
            raise ValueError(
                f"Manifest has conflicting labels for {len(conflicting_ids)} slide IDs; "
                f"first 5: {conflicting_ids[:5]}"
            )
        unique_labels = sorted(int(value) for value in labels.unique())
        expected_labels = list(range(len(unique_labels)))
        if unique_labels != expected_labels:
            raise ValueError(
                "Labels must be zero-based contiguous integers for cross-entropy; "
                f"found {unique_labels}, expected {expected_labels}"
            )
        self._labels_map = dict(zip(df["slide_id"], labels, strict=True))
        return self._labels_map

    def _load_sample_weights(self) -> dict[str, float] | None:
        """Read the per-slide loss weight column, if the study declares one."""
        if not self.sample_weight_column:
            return None
        if self._sample_weights_map is not None:
            return self._sample_weights_map
        df = pd.read_csv(self.csv_path)
        if self.sample_weight_column not in df.columns:
            raise ValueError(
                f"sample_weight_column='{self.sample_weight_column}' not found. "
                f"Columns: {list(df.columns)}"
            )
        weights = pd.to_numeric(df[self.sample_weight_column], errors="coerce")
        if weights.isna().any() or (weights <= 0).any():
            raise ValueError(
                f"'{self.sample_weight_column}' must be positive and complete; found "
                f"{int(weights.isna().sum())} missing and {int((weights <= 0).sum())} "
                "non-positive value(s)"
            )
        slide_ids = df[self.filename_column].astype(str).map(normalize_slide_id)
        self._sample_weights_map = dict(
            zip(slide_ids, weights.astype(float), strict=True)
        )
        return self._sample_weights_map

    @property
    def patient_of_slide(self) -> dict[str, str] | None:
        """slide_id → patient_id, when the manifest declares a patient column.

        Consumed by the patient-level validation AUROC monitor and the
        small-class early-stopping rule; None when no column is configured
        (slide-level fallback everywhere).
        """
        if self.patient_column is None:
            return None
        if self._patient_map is None:
            df = pd.read_csv(self.csv_path)
            if self.patient_column not in df.columns:
                raise ValueError(
                    f"patient_column='{self.patient_column}' not found. Columns: {list(df.columns)}"
                )
            if df[self.patient_column].isna().any():
                raise ValueError(f"Manifest column '{self.patient_column}' has missing values")
            df["slide_id"] = df[self.filename_column].astype(str).map(normalize_slide_id)
            self._patient_map = dict(
                zip(df["slide_id"], df[self.patient_column].astype(str), strict=True)
            )
        return self._patient_map

    # ── Setup ─────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_split_dataset(
        dataset: SlideDataset,
        requested_ids: list[str],
        split_name: str,
    ) -> None:
        """Fail before training when a declared split cannot be loaded exactly."""
        requested = {normalize_slide_id(slide_id) for slide_id in requested_ids}
        loaded = set(dataset.slide_ids)
        missing = sorted(requested - loaded)
        if missing:
            raise FileNotFoundError(
                f"{split_name} split is missing {len(missing)} usable H5 feature files; "
                f"first 5: {missing[:5]}"
            )
        if len(dataset) == 0:
            raise RuntimeError(f"{split_name} split has no usable slides")
        unlabeled = [
            slide_id
            for slide_id, label in zip(dataset.slide_ids, dataset.labels_list, strict=True)
            if label < 0
        ]
        if unlabeled:
            raise ValueError(
                f"{split_name} split has {len(unlabeled)} slides without labels; "
                f"first 5: {unlabeled[:5]}"
            )

    def setup(self, stage: str | None = None) -> None:
        labels = self._load_labels()
        store = self._get_store()

        splits_df = load_splits(
            self.splits_dir,
            csv_path=self.csv_path if self.verify_splits else None,
            verify=self.verify_splits,
        )
        fold_ids = get_slide_ids_for_fold(
            splits_df,
            self.fold,
            scheme=self.scheme,
        )

        if stage in ("fit", None):
            if self.refit_mode:
                # REFIT MODE: ALL non-test slides go to train, no val
                all_train_ids = fold_ids["train"] + fold_ids["val"]
                if self.scheme in ("oof_kfold", "predefined_oof_kfold"):
                    # oof_kfold has no global holdout: fold i's "test" is just
                    # the OOF fold and is development data — a refit trains on
                    # the complete development cohort.
                    all_train_ids = all_train_ids + fold_ids["test"]
                logger.info(
                    f"Refit mode: {len(all_train_ids)} slides "
                    f"(train={len(fold_ids['train'])} + val={len(fold_ids['val'])}"
                    + (
                        f" + oof_test={len(fold_ids['test'])}"
                        if self.scheme in ("oof_kfold", "predefined_oof_kfold")
                        else ""
                    )
                    + ")"
                )
            else:
                all_train_ids = fold_ids["train"]

            # TRAIN: stochastic subsampling + augmentation
            # dataset_max_instances provides regularization via random views
            self.train_dataset = SlideDataset(
                feature_dir=self.feature_dir,
                slide_ids=all_train_ids,  # <── was fold_ids["train"]
                labels=labels,
                max_instances=self.dataset_max_instances,
                is_train=True,
                cap_strategy=self.cap_strategy,
                cap_grid_size=self.cap_grid_size,
                instance_dropout=self.instance_dropout,
                feature_noise_std=self.feature_noise_std,
                cache_size_mb=0,  # never cache train (stochastic)
                return_coords=self.return_coords,
                force_float32=self.force_float32,
                store=store,
                fixed_bag_size=self.fixed_bag_size,
                short_bag_policy=self.short_bag_policy,
                sample_weights=self._load_sample_weights(),
            )
            self._validate_split_dataset(self.train_dataset, all_train_ids, "train")

            if self.refit_mode:
                # No validation in refit mode
                self.val_dataset = None
            else:
                # VAL: deterministic subsampling with the same dataset cap as train.
                # Collation is still dynamic unless use_preallocated_collator=True,
                # so batches are [B, N_batch, D] with N_batch <= max_instances.
                self.val_dataset = SlideDataset(
                    feature_dir=self.feature_dir,
                    slide_ids=fold_ids["val"],
                    labels=labels,
                    max_instances=None if self.eval_full_bags else self.dataset_max_instances,
                    is_train=False,
                    cap_strategy=self.cap_strategy,
                    cap_grid_size=self.cap_grid_size,
                    eval_crop_seed=0,
                    cache_size_mb=self.cache_size_mb if self.eval_n_crops <= 1 else 0,
                    return_coords=self.return_coords,
                    force_float32=self.force_float32,
                    store=store,
                    fixed_bag_size=self.eval_fixed_bag_size,
                    short_bag_policy=self.short_bag_policy,
                )
                self._validate_split_dataset(self.val_dataset, fold_ids["val"], "validation")

            # Log bag size statistics
            train_sizes = self.train_dataset.get_bag_sizes()
            logger.info(
                f"Fold {self.fold}: "
                f"train={len(self.train_dataset)} ({self.train_dataset.get_label_counts()})"
                + (
                    f", val={len(self.val_dataset)} ({self.val_dataset.get_label_counts()})"
                    if self.val_dataset
                    else ", val=None (refit mode)"
                )
            )
            if len(train_sizes) > 0:
                logger.info(
                    f"Bag sizes — train: mean={train_sizes.mean():.0f}, max={train_sizes.max()}"
                )
                if self.val_dataset:
                    val_sizes = self.val_dataset.get_bag_sizes()
                    logger.info(
                        f"Bag sizes — val: mean={val_sizes.mean():.0f}, max={val_sizes.max()}"
                    )

        if stage in ("test", "predict", None):
            test_ids = (
                self.test_slide_ids if self.test_slide_ids is not None else fold_ids.get("test", [])
            )
            if test_ids:
                self.test_dataset = SlideDataset(
                    feature_dir=self.feature_dir,
                    slide_ids=test_ids,
                    labels=labels,
                    max_instances=None if self.eval_full_bags else self.dataset_max_instances,
                    is_train=False,
                    cap_strategy=self.cap_strategy,
                    cap_grid_size=self.cap_grid_size,
                    eval_crop_seed=0,
                    cache_size_mb=self.cache_size_mb if self.eval_n_crops <= 1 else 0,
                    return_coords=self.return_coords,
                    force_float32=self.force_float32,
                    store=store,
                    fixed_bag_size=self.eval_fixed_bag_size,
                    short_bag_policy=self.short_bag_policy,
                )
                self._validate_split_dataset(self.test_dataset, test_ids, "test")
                logger.info(f"Test set: {len(self.test_dataset)} slides")

    # ── Collators ─────────────────────────────────────────────────────────

    def _resolve_pin_memory(self) -> bool:
        return _cuda_pin_memory_available() if self.pin_memory is None else bool(self.pin_memory)

    def _resolve_persistent_workers(self) -> bool:
        if self.persistent_workers is None:
            return self.num_workers > 0
        return bool(self.persistent_workers) and self.num_workers > 0

    def _loader_kwargs(self) -> dict:
        kw = {
            "num_workers": self.num_workers,
            "pin_memory": self._resolve_pin_memory(),
            "persistent_workers": self._resolve_persistent_workers(),
        }
        if self.num_workers > 0:
            kw["prefetch_factor"] = self.prefetch_factor
            kw["worker_init_fn"] = self._worker_init_fn
        return kw

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """Reset per-worker RNG after fork so slide sampling diverges by worker."""
        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        ds = info.dataset
        if hasattr(ds, "rng"):
            ds.rng = np.random.default_rng(info.seed & 0xFFFF_FFFF)

    def _train_collator(self):
        """Fixed-width collator for training.

        ``fixed_bag_size`` wins over ``max_instances``: when it is set the
        dataset already returns exactly that many rows, so the collator's
        uniform-batch path applies and no padding or masking work happens.
        """
        width = self.fixed_bag_size if self.fixed_bag_size is not None else self.max_instances
        if self.use_preallocated_collator and width is not None:
            return MILCollator(
                max_instances=width,
                feat_dim=self.feat_dim,
                batch_size=self.batch_size,
                pin_memory=self._resolve_pin_memory(),
            )
        return SimpleMILCollator(max_instances=width)

    def _eval_collator(self):
        """Collator for eval.

        Uses fixed-width batching only when a width is known
        (``eval_fixed_bag_size`` or ``max_instances``) and
        ``use_preallocated_collator=True``. Otherwise pads dynamically to the
        max bag size in the current batch.
        """
        width = (
            self.eval_fixed_bag_size if self.eval_fixed_bag_size is not None else self.max_instances
        )
        if self.use_preallocated_collator and width is not None:
            return MILCollator(
                max_instances=width,
                feat_dim=self.feat_dim,
                batch_size=self.eval_batch_size,
                pin_memory=self._resolve_pin_memory(),
            )
        return SimpleMILCollator(max_instances=width)

    def _eval_sampler(self, dataset: SlideDataset):
        """Order eval slides by bag size so multi-slide batches pad very little.

        Predictions are keyed by ``slide_id``, so evaluation order carries no
        meaning and sorting is free. Skipped for ``eval_batch_size == 1`` (no
        padding to save) and under DDP (Lightning owns the eval sampler there).
        """
        if self.eval_batch_size <= 1 or _ddp_active():
            return None
        order = np.argsort(dataset.get_bag_sizes(), kind="stable")
        return list(order.tolist())

    # ── Sampler ───────────────────────────────────────────────────────────

    def _patient_sampling_metadata(self, dataset: SlideDataset) -> tuple[list[str], list[str]]:
        """Load patient/cohort metadata in exact dataset index order."""
        if self.patient_column is None or self.cohort_column is None:
            raise RuntimeError("Patient sampling columns were not configured")
        patient_column = self.patient_column
        cohort_column = self.cohort_column
        requested_columns = [
            self.filename_column,
            self.label_column,
            patient_column,
            cohort_column,
        ]
        frame = pd.read_csv(self.csv_path)
        missing_columns = [column for column in requested_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Patient sampling manifest is missing required columns: {missing_columns}"
            )
        frame = frame[requested_columns].copy()
        frame["slide_id"] = frame[self.filename_column].astype(str).map(normalize_slide_id)
        duplicate_slides = frame["slide_id"].duplicated(keep=False)
        if duplicate_slides.any():
            examples = sorted(frame.loc[duplicate_slides, "slide_id"].unique())[:5]
            raise ValueError(
                "Patient sampling requires one manifest row per slide; duplicate "
                f"slide IDs include {examples}"
            )
        frame = frame.set_index("slide_id").reindex(dataset.slide_ids)
        metadata_columns = [self.label_column, patient_column, cohort_column]
        if frame[metadata_columns].isna().any().any():
            missing_ids = frame.index[frame[metadata_columns].isna().any(axis=1)].tolist()
            raise ValueError(
                "Patient sampling metadata is missing for training slides; "
                f"examples: {missing_ids[:5]}"
            )
        frame["patient_id"] = frame[patient_column].astype(str).str.strip()
        frame["cohort"] = frame[cohort_column].astype(str).str.strip()
        frame["label"] = pd.to_numeric(frame[self.label_column], errors="raise").astype(int)
        if frame["patient_id"].eq("").any() or frame["cohort"].eq("").any():
            raise ValueError("Patient sampling metadata contains blank patient/cohort values")
        # The binary restriction belongs to the strategy, not to patient
        # sampling as such: patient_natural and cohort_balanced never read the
        # label, so a multiclass target is fine for them. PatientSlideSampler
        # raises for the one strategy that does need 0/1.
        if (
            self.train_sampling_strategy == "cohort_label_balanced"
            and not frame["label"].isin([0, 1]).all()
        ):
            raise ValueError(
                "cohort_label_balanced sampling supports binary labels 0/1 only; "
                "use patient_natural or cohort_balanced for a multiclass target"
            )

        return frame["patient_id"].tolist(), frame["cohort"].tolist()

    @property
    def training_sampling_summary(self) -> dict[str, Any] | None:
        """Auditable expected exposure distribution for the train loader."""
        if self._train_sampler is not None:
            from oceanpath.datasets.sampling import PatientSlideSampler

            if isinstance(self._train_sampler, PatientSlideSampler):
                return self._train_sampler.summary()
        return self._sampling_summary

    def _get_sampler(self, dataset: SlideDataset):
        if self.train_sampling_strategy != "slide_uniform":
            from oceanpath.datasets.sampling import PatientSlideSampler

            if isinstance(self._train_sampler, PatientSlideSampler):
                return self._train_sampler
            patient_ids, cohorts = self._patient_sampling_metadata(dataset)
            sampler = PatientSlideSampler(
                slide_ids=dataset.slide_ids,
                labels=dataset.labels_list,
                patient_ids=patient_ids,
                cohorts=cohorts,
                strategy=self.train_sampling_strategy,
                target_positive_prevalence=self.sampling_target_positive_prevalence,
                seed=self.sampling_seed,
            )
            self._train_sampler = sampler
            return sampler
        if not self.class_weighted_sampling:
            self._sampling_summary = {
                "strategy": "slide_uniform",
                "sampling_unit": "slide",
                "replacement": False,
                "samples_per_epoch": int(len(dataset)),
            }
            return None
        labels = dataset.get_all_labels()
        if len(labels) == 0:
            return None
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2:
            logger.warning(
                f"Only {len(unique)} class(es) in training set — "
                f"class-weighted sampling may not be meaningful."
            )
        weights_per_class = 1.0 / counts.astype(float)
        class_to_weight = dict(zip(unique, weights_per_class, strict=True))
        sample_weights = np.array([class_to_weight[label] for label in labels])
        return WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(dataset),
            replacement=True,
        )

    # ── DataLoaders ───────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        if self.length_bucket:
            if _ddp_active():
                # LengthBucketBatchSampler yields whole batches; sharding it
                # across ranks would require a custom DistributedBatchSampler
                # that we do not have. Fail loudly rather than silently
                # replicating batches across ranks.
                raise NotImplementedError(
                    "length_bucket=True is not compatible with DDP. "
                    "Set training.length_bucket=false (the supervised "
                    "fixed-N regime already pads to a constant length)."
                )
            if self.class_weighted_sampling or self.train_sampling_strategy != "slide_uniform":
                logger.warning(
                    "length_bucket=True overrides weighted/patient sampling. "
                    "Set length_bucket=false when a sampling strategy is part of "
                    "the experimental protocol."
                )
            # Length-bucketed batching: group similar-length slides together
            batch_sampler = LengthBucketBatchSampler(
                lengths=self.train_dataset.get_bag_sizes(),
                batch_size=self.batch_size,
                bucket_size=self.length_bucket_size,
                drop_last=self.drop_last,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self._train_collator(),
                **self._loader_kwargs(),
            )

        sampler = self._get_sampler(self.train_dataset)
        # Under DDP, Lightning auto-wraps eval loaders (no sampler) but skips
        # train loaders that already carry a custom sampler. Wrap manually so
        # WeightedRandomSampler (or any future custom sampler) shards across
        # ranks instead of replicating its full draw on every rank.
        if _ddp_active():
            if sampler is None:
                # Vanilla random shuffle — DistributedSampler does the work.
                sampler = DistributedSampler(self.train_dataset, shuffle=True)
            else:
                sampler = DistributedSamplerWrapper(sampler, shuffle=False)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=self._train_collator(),
            drop_last=self.drop_last,
            **self._loader_kwargs(),
        )

    def set_eval_crop_seed(self, dataset: SlideDataset, seed: int) -> None:
        """Set the eval crop seed on a dataset (for multi-crop evaluation)."""
        dataset.eval_crop_seed = seed
        # Invalidate cache since different seed → different crops
        if dataset._cache is not None:
            dataset._cache.clear()
            dataset._cache_current_bytes = 0

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            return None  # refit mode has no validation set
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            sampler=self._eval_sampler(self.val_dataset),
            collate_fn=self._eval_collator(),
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("No test set for this scheme/fold")
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            sampler=self._eval_sampler(self.test_dataset),
            collate_fn=self._eval_collator(),
            **self._loader_kwargs(),
        )
