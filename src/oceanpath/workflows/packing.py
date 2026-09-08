"""Feature-packing workflow boundary.

Packing rewrites the per-slide H5 features published by extraction into one
flat memory-mapped store optimised for the training read pattern. It produces
no new labels or representations: the pack is a deterministic FP16
re-materialisation of an existing extraction output, so it reports under the
extraction stage. FP16 quantisation is intentionally lossy and must retain its
own downstream quality A/B evidence.

See :mod:`oceanpath.datasets.packed` for the storage format and the reasoning
behind it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from oceanpath.config import FoundationPaths, cfg_select
from oceanpath.contracts import PipelineStage, StageResult
from oceanpath.datasets.packed import pack_features
from oceanpath.runtime import setup_logging

logger = logging.getLogger(__name__)


def resolve_pack_dir(cfg: Any) -> Path:
    """Where the pack for this (data, encoder, extraction) triple lives.

    Defaults to a ``packed_{encoder}`` sibling of the ``features_{encoder}``
    directory so the pack travels with the features it was built from.
    """
    configured = cfg_select(cfg, "training.packed_dir", None)
    if configured not in (None, "null", ""):
        return Path(str(configured))
    feature_dir = Path(FoundationPaths.from_config(cfg).feature_h5_dir)
    name = feature_dir.name
    packed_name = (
        f"packed_{name[len('features_') :]}" if name.startswith("features_") else f"{name}_packed"
    )
    return feature_dir.with_name(packed_name)


def run_packing(cfg: Any) -> StageResult:
    """Build a packed feature store from the configured feature directory."""
    stage = PipelineStage.EXTRACT_FEATURES
    setup_logging(cfg, stage="pack_features")
    paths = FoundationPaths.from_config(cfg)
    feature_dir = Path(paths.feature_h5_dir)
    pack_dir = resolve_pack_dir(cfg)

    feat_dtype = str(cfg_select(cfg, "pack.dtype", "float16"))
    include_coords = bool(cfg_select(cfg, "pack.include_coords", True))
    stream_chunk_size = int(cfg_select(cfg, "pack.stream_chunk_size", 65_536))
    verify_source_unchanged = bool(cfg_select(cfg, "pack.verify_source_unchanged", True))
    overwrite = bool(cfg_select(cfg, "pack.overwrite", False))
    dry_run = bool(cfg_select(cfg, "dry_run", False))

    logger.info("Source features : %s", feature_dir)
    logger.info("Pack destination: %s", pack_dir)

    started = time.monotonic()
    if dry_run:
        n_slides = len(list(feature_dir.glob("*.h5"))) if feature_dir.is_dir() else 0
        return StageResult(
            stage=stage,
            status="dry_run",
            output_dir=pack_dir,
            elapsed_seconds=time.monotonic() - started,
            details={
                "feature_dir": str(feature_dir),
                "pack_dir": str(pack_dir),
                "n_source_h5": n_slides,
                "dtype": feat_dtype,
                "stream_chunk_size": stream_chunk_size,
            },
        )

    meta = pack_features(
        feature_dir=feature_dir,
        pack_dir=pack_dir,
        feat_dtype=feat_dtype,
        include_coords=include_coords,
        overwrite=overwrite,
        stream_chunk_size=stream_chunk_size,
        verify_source_unchanged=verify_source_unchanged,
    )

    logger.info(
        "Train against this pack with: training.packed_dir=%s training.force_float32=false",
        pack_dir,
    )
    return StageResult(
        stage=stage,
        status="completed",
        output_dir=pack_dir,
        elapsed_seconds=time.monotonic() - started,
        details={
            "feature_dir": str(feature_dir),
            "pack_dir": str(pack_dir),
            "n_slides": meta.n_slides,
            "total_patches": meta.total_patches,
            "feat_dim": meta.feat_dim,
            "feat_dtype": meta.feat_dtype,
            "has_coords": meta.has_coords,
            "stream_chunk_size": stream_chunk_size,
            "source_inventory_sha256": meta.source_inventory_sha256,
        },
    )
