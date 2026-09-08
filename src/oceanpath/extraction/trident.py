"""
TRIDENT adapter for config-driven feature extraction.

This module provides a clean, config-driven TRIDENT interface that:
  1. Replaces argparse with structured configs
  2. Adds provenance tracking (manifest.json + encoder fingerprint)
  3. Makes extraction resumable at the slide level with stale-detection
  4. Separates "what to extract" from "how to run it"
  5. Supports Slurm array sharding for parallel extraction
  6. Validates all inputs before committing compute
  7. Adds an exact-MPP compatibility layer for coordinate generation and replay
"""

import datetime
import hashlib
import json
import logging
import math
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from oceanpath.extraction.slide_records import (
    SlideRecord,
    SlideRecordError,
    load_slide_records,
    shard_slide_records,
    write_slide_records_csv,
)

logger = logging.getLogger(__name__)


# ── Schema version ────────────────────────────────────────────────────────────
H5_SCHEMA_VERSION = 2
MANIFEST_SCHEMA_VERSION = 2
SUPPORTED_ENCODER_ADAPTERS = {"trident"}
# Mirrored from pinned TRIDENT 0.2.3. Keeping this tiny compatibility map local
# lets cache invalidation work before the optional extraction dependency loads.
PINNED_SLIDE_TO_PATCH_ENCODER = {
    "threads": "conch_v15",
    "titan": "conch_v15",
    "tcga": "conch_v15",
    "prism": "virchow",
    "chief": "ctranspath",
    "gigapath": "gigapath",
    "madeleine": "conch_v1",
    "feather": "conch_v15",
}


# ── Config dataclass ──────────────────────────────────────────────────────────


@dataclass
class TridentExtractionConfig:
    """All parameters needed to run TRIDENT feature extraction."""

    # Slide source
    wsi_dir: str
    job_dir: str
    wsi_ext: list[str] | None = None
    custom_list_of_wsis: str | None = None
    custom_mpp_keys: list[str] | None = None
    reader_type: str | None = None
    search_nested: bool = False

    # Segmentation
    segmenter: str = "hest"
    seg_conf_thresh: float = 0.5
    remove_holes: bool = False
    remove_artifacts: bool = False
    remove_penmarks: bool = False

    # Patching
    mag: int = 20
    target_mpp: float | None = None
    patch_size: int = 256
    overlap: int = 0
    min_tissue_proportion: float = 0.0
    coords_dir: str | None = None

    # Feature extraction
    patch_encoder: str = "uni_v1"
    patch_encoder_ckpt_path: str | None = None
    slide_encoder: str | None = None
    encoder_source: str | None = None
    encoder_adapter: str = "trident"

    # Runtime
    gpu: int = 0
    batch_size: int = 64
    seg_batch_size: int | None = None
    feat_batch_size: int | None = None
    max_workers: int | None = None
    skip_errors: bool = True
    require_complete: bool = True

    # Caching
    wsi_cache: str | None = None
    cache_batch_size: int = 32

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return f"cuda:{self.gpu}"
        return "cpu"

    @property
    def coords_subdir(self) -> str:
        if self.coords_dir:
            return self.coords_dir
        base = f"{self.mag}x_{self.patch_size}px_{self.overlap}px_overlap"
        if self.target_mpp is not None:
            return f"{base}_mpp{self.target_mpp:g}"
        return base

    @property
    def encoder_name(self) -> str:
        return self.slide_encoder or self.patch_encoder

    @property
    def encoder_type(self) -> str:
        return "slide" if self.slide_encoder else "patch"


# ── Encoder fingerprinting ────────────────────────────────────────────────────


def compute_encoder_fingerprint(cfg: TridentExtractionConfig) -> str:
    ckpt_path = cfg.patch_encoder_ckpt_path
    if ckpt_path and Path(ckpt_path).is_file():
        return _hash_file(Path(ckpt_path))
    id_string = ":".join(
        [
            cfg.encoder_adapter,
            cfg.encoder_name,
            cfg.encoder_source or "unknown-source",
            ckpt_path or "default",
        ]
    )
    return hashlib.sha256(id_string.encode()).hexdigest()[:16]


def _fingerprint_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def compute_stage_fingerprints(cfg: TridentExtractionConfig) -> dict[str, str]:
    """Fingerprint segmentation, coordinate, and feature lineage separately."""

    source_inventory: dict[str, Any] | None = None
    with suppress(OSError, SlideRecordError):
        source_inventory = {
            "wsi_dir": str(Path(cfg.wsi_dir).expanduser().resolve(strict=True)),
            "slides": [],
        }
        for record in _load_records(cfg):
            stat = record.path.stat()
            source_inventory["slides"].append(
                {
                    "wsi": record.wsi,
                    "path": str(record.path),
                    "mpp": record.mpp,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                }
            )

    segmentation = _fingerprint_payload(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source_inventory": source_inventory,
            "segmenter": cfg.segmenter,
            "seg_conf_thresh": cfg.seg_conf_thresh,
            "remove_holes": cfg.remove_holes,
            "remove_artifacts": cfg.remove_artifacts,
            "remove_penmarks": cfg.remove_penmarks,
            "reader_type": cfg.reader_type,
            "custom_mpp_keys": cfg.custom_mpp_keys,
        }
    )
    coordinates = _fingerprint_payload(
        {
            "segmentation": segmentation,
            "mag": cfg.mag,
            "target_mpp": cfg.target_mpp,
            "patch_size": cfg.patch_size,
            "overlap": cfg.overlap,
            "min_tissue_proportion": cfg.min_tissue_proportion,
            "coords_subdir": cfg.coords_subdir,
            "h5_schema_version": H5_SCHEMA_VERSION,
        }
    )
    features = _fingerprint_payload(
        {
            "coordinates": coordinates,
            "encoder_fingerprint": compute_encoder_fingerprint(cfg),
            "encoder_type": cfg.encoder_type,
            "patch_encoder": cfg.patch_encoder,
            "slide_encoder": cfg.slide_encoder,
        }
    )
    return {
        "seg": segmentation,
        "coords": coordinates,
        "feat": features,
    }


def compute_extraction_fingerprint(cfg: TridentExtractionConfig) -> str:
    """Fingerprint every semantic input that changes extracted pixels/features."""
    return _fingerprint_payload(compute_stage_fingerprints(cfg))


def _assert_inputs_unchanged(
    cfg: TridentExtractionConfig,
    expected_stages: dict[str, str],
    expected_extraction: str,
) -> None:
    current_stages = compute_stage_fingerprints(cfg)
    current_extraction = _fingerprint_payload(current_stages)
    if current_stages != expected_stages or current_extraction != expected_extraction:
        raise RuntimeError(
            "Slides or the WSI selection changed during extraction; refusing to commit a "
            "mixed-input result. Rerun against a frozen canonical selection."
        )


def _hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


# ── Input validation ──────────────────────────────────────────────────────────


class ValidationError(Exception):
    pass


def _load_records(cfg: TridentExtractionConfig) -> list[SlideRecord]:
    """Resolve one authoritative slide selection for every adapter operation."""

    return load_slide_records(
        cfg.wsi_dir,
        custom_list_path=cfg.custom_list_of_wsis,
        wsi_ext=cfg.wsi_ext,
        search_nested=cfg.search_nested,
    )


def _records_include_mpp(records: list[SlideRecord]) -> bool:
    return bool(records) and all(record.mpp is not None for record in records)


def _records_fingerprint(records: list[SlideRecord]) -> str:
    payload = [
        {"wsi": record.wsi, "mpp": record.mpp}
        for record in sorted(records, key=lambda record: (record.wsi.casefold(), record.wsi))
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _canonical_processor_selection(
    cfg: TridentExtractionConfig,
    records: list[SlideRecord],
) -> str | None:
    """Materialize validated paths before handing a custom CSV to TRIDENT."""

    if cfg.custom_list_of_wsis is None:
        return None
    digest = _records_fingerprint(records)
    selection_path = Path(cfg.job_dir) / ".selections" / f"selection_{digest}.csv"
    write_slide_records_csv(
        records,
        selection_path,
        include_mpp=_records_include_mpp(records),
    )
    return str(selection_path)


def validate_inputs(
    cfg: TridentExtractionConfig,
    tasks: list[str],
    *,
    create_job_dir: bool = True,
) -> dict:
    errors = []

    if cfg.mag <= 0:
        errors.append(f"Magnification must be positive, got {cfg.mag}")
    if cfg.patch_size <= 0:
        errors.append(f"Patch size must be positive, got {cfg.patch_size}")
    if cfg.overlap < 0 or cfg.overlap >= cfg.patch_size:
        errors.append(
            f"Overlap must satisfy 0 <= overlap < patch_size, got "
            f"overlap={cfg.overlap}, patch_size={cfg.patch_size}"
        )
    if not 0.0 <= cfg.min_tissue_proportion <= 1.0:
        errors.append(
            f"Minimum tissue proportion must be between 0 and 1, got {cfg.min_tissue_proportion}"
        )
    if cfg.target_mpp is not None:
        if not math.isfinite(cfg.target_mpp) or cfg.target_mpp <= 0:
            errors.append(f"Target MPP must be finite and positive, got {cfg.target_mpp}")
        elif not math.isclose(cfg.target_mpp, 10.0 / cfg.mag, rel_tol=0.0, abs_tol=1e-6):
            errors.append(
                f"Conflicting physical scale: mag={cfg.mag} implies "
                f"target_mpp={10.0 / cfg.mag:g}, got {cfg.target_mpp:g}"
            )

    if "feat" in tasks and cfg.encoder_adapter not in SUPPORTED_ENCODER_ADAPTERS:
        errors.append(
            f"Encoder adapter '{cfg.encoder_adapter}' is not implemented. "
            "Gemma 4 profiles are metadata-only until OceanPath has a tested "
            "Hugging Face adapter for image preprocessing and soft-token pooling; "
            "use an encoder profile with adapter=trident for extraction."
        )

    wsi_dir = Path(cfg.wsi_dir)
    records: list[SlideRecord] = []
    try:
        records = _load_records(cfg)
    except SlideRecordError as exc:
        errors.append(str(exc))

    if cfg.target_mpp is not None and cfg.custom_list_of_wsis and records:
        missing_mpp = [record.wsi for record in records if record.mpp is None]
        if missing_mpp:
            errors.append(
                "Exact-MPP extraction requires a numeric 'mpp' column for every row "
                f"of the custom WSI CSV; missing for {len(missing_mpp)} slide(s)"
            )

    slide_count = len(records)

    if "cuda" in cfg.device:
        if not torch.cuda.is_available():
            errors.append("CUDA requested but torch.cuda.is_available() is False")
        elif cfg.gpu >= torch.cuda.device_count():
            errors.append(f"GPU {cfg.gpu} requested but only {torch.cuda.device_count()} available")

    job_dir = Path(cfg.job_dir)
    if create_job_dir:
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            errors.append(f"Cannot create job directory {job_dir}: {e}")

    valid_tasks = {"seg", "coords", "feat"}
    if not tasks:
        errors.append("At least one extraction task is required")
    for t in tasks:
        if t not in valid_tasks:
            errors.append(f"Unknown task '{t}'. Must be one of {valid_tasks}")
    if len(tasks) != len(set(tasks)):
        errors.append(f"Tasks may not contain duplicates, got {tasks}")
    task_order = {"seg": 0, "coords": 1, "feat": 2}
    known_tasks = [task for task in tasks if task in task_order]
    if known_tasks != sorted(known_tasks, key=task_order.__getitem__):
        errors.append(f"Tasks must follow dependency order [seg, coords, feat], got {tasks}")
    if "seg" in tasks and "feat" in tasks and "coords" not in tasks:
        errors.append("A run containing both seg and feat must also contain coords")

    if errors:
        raise ValidationError("Input validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    return {
        "slide_count": slide_count,
        "slide_dir": str(wsi_dir),
        "encoder": cfg.encoder_name,
        "encoder_type": cfg.encoder_type,
        "encoder_source": cfg.encoder_source,
        "encoder_adapter": cfg.encoder_adapter,
        "device": cfg.device,
        "tasks": tasks,
        "job_dir": str(cfg.job_dir),
        "mag": cfg.mag,
        "target_mpp": cfg.target_mpp,
        "patch_size": cfg.patch_size,
    }


# ── Diff mode ─────────────────────────────────────────────────────────────────


def compute_extraction_diff(
    cfg: TridentExtractionConfig,
    extraction_fingerprint: str,
    *,
    tasks: list[str] | None = None,
    manifest_path: Path | None = None,
) -> Path | None:
    """Write the immutable subset whose requested outputs need recomputation."""

    tasks = tasks or ["feat"]
    manifest_path = manifest_path or Path(cfg.job_dir) / "manifest.json"

    records = _load_records(cfg)

    prev_fingerprint = None
    if manifest_path.is_file():
        try:
            prev_manifest = json.loads(manifest_path.read_text())
            prev_fingerprint = prev_manifest.get("extraction_fingerprint")
        except (json.JSONDecodeError, KeyError):
            pass

    output_validation = validate_outputs(cfg, tasks)
    unhealthy_ids = set(output_validation["missing"])
    unhealthy_ids.update(item.split(":", 1)[0] for item in output_validation["invalid"])

    needs_extraction: list[SlideRecord] = []
    for record in records:
        if record.output_id in unhealthy_ids or prev_fingerprint != extraction_fingerprint:
            needs_extraction.append(record)

    if not needs_extraction:
        logger.info(
            f"All {len(records)} slides already extracted "
            f"with extraction fingerprint {extraction_fingerprint[:8]}..."
        )
        return None

    logger.info(f"Diff mode: {len(needs_extraction)}/{len(records)} slides need extraction")

    requested_tasks = sorted(set(tasks))
    task_fingerprint = _fingerprint_payload({"tasks": requested_tasks})
    needs_fingerprint = _records_fingerprint(needs_extraction)
    diff_name = (
        f"diff_{_records_fingerprint(records)}_{extraction_fingerprint}_"
        f"{task_fingerprint}_{needs_fingerprint}.csv"
    )
    diff_list_path = Path(cfg.job_dir) / ".diff" / diff_name
    return write_slide_records_csv(
        needs_extraction,
        diff_list_path,
        include_mpp=_records_include_mpp(records),
    )


# ── Slurm sharding ────────────────────────────────────────────────────────────


def shard_wsi_list(
    cfg: TridentExtractionConfig,
    shard_id: int,
    total_shards: int,
) -> Path:
    records = _load_records(cfg)
    shard_records = shard_slide_records(
        records,
        shard_id=shard_id,
        total_shards=total_shards,
    )
    logger.info(
        f"Slurm shard {shard_id}/{total_shards}: {len(shard_records)}/{len(records)} slides"
    )

    shard_dir = Path(cfg.job_dir) / ".shards"
    shard_path = shard_dir / f"shard_{shard_id:04d}.csv"
    return write_slide_records_csv(
        shard_records,
        shard_path,
        include_mpp=_records_include_mpp(records),
    )


def _csv_has_records(path: Path) -> bool:
    """Return whether a generated TRIDENT subset CSV contains a data row."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        next(handle, None)  # header
        return any(line.strip() for line in handle)


def _feature_h5_dir(cfg: TridentExtractionConfig) -> Path:
    prefix = "slide_features" if cfg.encoder_type == "slide" else "features"
    return Path(cfg.job_dir) / cfg.coords_subdir / f"{prefix}_{cfg.encoder_name}"


def _required_patch_feature_dir(cfg: TridentExtractionConfig) -> Path:
    if cfg.slide_encoder is None:
        raise ValidationError("A patch-feature dependency exists only for slide encoders")
    if cfg.slide_encoder.startswith("mean-"):
        patch_encoder = cfg.slide_encoder.split("mean-", 1)[1]
    else:
        try:
            patch_encoder = PINNED_SLIDE_TO_PATCH_ENCODER[cfg.slide_encoder]
        except KeyError as exc:
            raise ValidationError(
                f"Unknown patch-feature dependency for slide encoder {cfg.slide_encoder!r}"
            ) from exc
    return Path(cfg.job_dir) / cfg.coords_subdir / f"features_{patch_encoder}"


def _artifact_paths(
    cfg: TridentExtractionConfig,
    records: list[SlideRecord],
    tasks: list[str],
) -> list[Path]:
    """Return selected artifacts invalidated by a forced or stale rerun."""

    invalidated = set(tasks)
    if "seg" in invalidated:
        invalidated.update({"coords", "feat"})
    if "coords" in invalidated:
        invalidated.add("feat")

    job_dir = Path(cfg.job_dir)
    paths: list[Path] = []
    for record in records:
        slide_id = record.output_id
        if "seg" in invalidated:
            paths.extend(
                [
                    job_dir / "contours" / f"{slide_id}.jpg",
                    job_dir / "contours_geojson" / f"{slide_id}.geojson",
                    job_dir / "thumbnails" / f"{slide_id}.jpg",
                ]
            )
        if "coords" in invalidated:
            paths.extend(
                [
                    job_dir / cfg.coords_subdir / "patches" / f"{slide_id}_patches.h5",
                    job_dir / cfg.coords_subdir / "visualization" / f"{slide_id}.jpg",
                ]
            )
        if "feat" in invalidated:
            paths.append(_feature_h5_dir(cfg) / f"{slide_id}.h5")
            if cfg.encoder_type == "slide":
                paths.append(_required_patch_feature_dir(cfg) / f"{slide_id}.h5")
    return paths


def _archive_selected_artifacts(
    cfg: TridentExtractionConfig,
    records: list[SlideRecord],
    tasks: list[str],
    *,
    reason: str,
) -> Path | None:
    """Atomically move invalidated artifacts aside so pinned TRIDENT cannot skip them."""

    candidates = _artifact_paths(cfg, records, tasks)
    locked = [Path(f"{path}.lock") for path in candidates if Path(f"{path}.lock").exists()]
    if locked:
        preview = ", ".join(str(path) for path in locked[:3])
        raise RuntimeError(
            f"Cannot {reason} while {len(locked)} selected artifact(s) are locked: {preview}"
        )

    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None

    job_dir = Path(cfg.job_dir).resolve()
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = job_dir / ".archive" / f"{timestamp}_{uuid.uuid4().hex[:8]}_{reason}"
    for source in existing:
        relative = source.resolve().relative_to(job_dir)
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)

    logger.warning(
        "Archived %d selected artifact(s) to %s before %s",
        len(existing),
        archive_root,
        reason,
    )
    return archive_root


def _run_manifest_path(cfg: TridentExtractionConfig, shard_id: int | None) -> Path:
    name = f"manifest_shard_{shard_id:04d}.json" if shard_id is not None else "manifest.json"
    return Path(cfg.job_dir) / name


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _require_current_prerequisites(
    cfg: TridentExtractionConfig,
    records: list[SlideRecord],
    tasks: list[str],
    previous_manifest: dict[str, Any] | None,
    stage_fingerprints: dict[str, str],
) -> None:
    """Fail closed when a partial run would consume unproven upstream artifacts."""

    requested = set(tasks)
    prerequisite: str | None = None
    consumer: str | None = None
    if "feat" in requested and "coords" not in requested:
        prerequisite, consumer = "coords", "feat"
    elif "coords" in requested and "seg" not in requested:
        prerequisite, consumer = "seg", "coords"
    if prerequisite is None:
        return

    job_dir = Path(cfg.job_dir)
    if prerequisite == "coords":
        prerequisite_complete = all(
            (job_dir / cfg.coords_subdir / "patches" / f"{record.output_id}_patches.h5").is_file()
            for record in records
        )
    else:
        prerequisite_complete = all(
            (job_dir / "contours_geojson" / f"{record.output_id}.geojson").is_file()
            and (job_dir / "contours" / f"{record.output_id}.jpg").is_file()
            for record in records
        )
    if not prerequisite_complete:
        prefix = "Feature extraction is incomplete: " if consumer == "feat" else ""
        raise RuntimeError(
            f"{prefix}cannot run {consumer} without complete {prerequisite} artifacts "
            "for every selected slide. Run the prerequisite stage(s) first."
        )

    previous_stages = (
        previous_manifest.get("completed_stage_fingerprints", {})
        if previous_manifest is not None
        else {}
    )
    if (
        not isinstance(previous_stages, dict)
        or previous_stages.get(prerequisite) != (stage_fingerprints[prerequisite])
    ):
        raise RuntimeError(
            f"Cannot run {consumer} without {prerequisite}: no manifest proves that the "
            f"existing {prerequisite} artifacts match the current slides and configuration. "
            "Run the prerequisite stage(s) in dependency order."
        )


def _derive_completed_stage_fingerprints(
    previous_manifest: dict[str, Any] | None,
    current: dict[str, str],
    tasks: list[str],
    *,
    run_complete: bool,
) -> dict[str, str]:
    """Carry forward only proven lineage and add successfully completed stages."""

    previous = (
        previous_manifest.get("completed_stage_fingerprints", {})
        if previous_manifest is not None
        else {}
    )
    if not isinstance(previous, dict):
        previous = {}
    completed = {
        stage: fingerprint
        for stage, fingerprint in previous.items()
        if stage in current and current[stage] == fingerprint
    }
    invalidated = set(tasks)
    if "seg" in invalidated:
        invalidated.update({"coords", "feat"})
    if "coords" in invalidated:
        invalidated.add("feat")
    for stage in invalidated:
        completed.pop(stage, None)
    if run_complete:
        for stage in tasks:
            completed[stage] = current[stage]
    return completed


def _has_existing_artifacts(
    cfg: TridentExtractionConfig,
    records: list[SlideRecord],
    tasks: list[str],
) -> bool:
    return any(path.is_file() for path in _artifact_paths(cfg, records, tasks))


# ── Core functions (lazy TRIDENT import) ──────────────────────────────────────

_trident_imported = False
_Processor: Any = None


def _lazy_import_trident() -> None:
    global _Processor, _trident_imported
    if _trident_imported:
        return
    from trident import Processor as _Proc

    _Processor = _Proc
    _trident_imported = True


def create_processor(cfg: TridentExtractionConfig) -> Any:
    _lazy_import_trident()
    records = _load_records(cfg)
    records_by_id = {record.output_id: record for record in records}
    processor_selection = _canonical_processor_selection(cfg, records)
    processor = _Processor(
        job_dir=cfg.job_dir,
        wsi_source=cfg.wsi_dir,
        wsi_ext=cfg.wsi_ext,
        wsi_cache=cfg.wsi_cache,
        skip_errors=cfg.skip_errors,
        custom_mpp_keys=cfg.custom_mpp_keys,
        custom_list_of_wsis=processor_selection,
        max_workers=cfg.max_workers,
        reader_type=cfg.reader_type,
        search_nested=cfg.search_nested,
    )
    if cfg.target_mpp is not None:
        from oceanpath.extraction.mpp_sampling import (
            validate_exact_mpp_coordinate_file,
            wrap_processor_wsis_for_exact_mpp,
        )

        coords_root = Path(cfg.job_dir) / cfg.coords_subdir / "patches"
        for wsi in processor.wsis:
            coords_path = coords_root / f"{wsi.name}_patches.h5"
            if coords_path.is_file():
                record = records_by_id.get(wsi.name)
                source_mpp = record.mpp if record is not None else None
                if source_mpp is None:
                    wsi._lazy_initialize()
                    source_mpp = wsi.mpp
                validate_exact_mpp_coordinate_file(
                    coords_path,
                    target_mpp=cfg.target_mpp,
                    source_mpp=source_mpp,
                    patch_size=cfg.patch_size,
                    overlap=cfg.overlap,
                    min_tissue_proportion=cfg.min_tissue_proportion,
                )
        wrap_processor_wsis_for_exact_mpp(processor, cfg.target_mpp)
    return processor


def run_segmentation(processor, cfg: TridentExtractionConfig) -> None:
    from trident.segmentation_models.load import segmentation_model_factory

    seg_model = segmentation_model_factory(
        cfg.segmenter,
        confidence_thresh=cfg.seg_conf_thresh,
    )
    artifact_model = None
    if cfg.remove_artifacts or cfg.remove_penmarks:
        artifact_model = segmentation_model_factory(
            "grandqc_artifact",
            remove_penmarks_only=cfg.remove_penmarks and not cfg.remove_artifacts,
        )
    processor.run_segmentation_job(
        seg_model,
        seg_mag=seg_model.target_mag,
        holes_are_tissue=not cfg.remove_holes,
        artifact_remover_model=artifact_model,
        batch_size=cfg.seg_batch_size or cfg.batch_size,
        device=cfg.device,
    )


def run_patching(processor, cfg: TridentExtractionConfig) -> None:
    processor.run_patching_job(
        target_magnification=cfg.mag,
        patch_size=cfg.patch_size,
        overlap=cfg.overlap,
        saveto=cfg.coords_subdir,
        min_tissue_proportion=cfg.min_tissue_proportion,
    )


def run_feature_extraction(processor, cfg: TridentExtractionConfig) -> None:
    if cfg.slide_encoder is None:
        from trident.patch_encoder_models.load import encoder_factory

        encoder = encoder_factory(cfg.patch_encoder, weights_path=cfg.patch_encoder_ckpt_path)
        processor.run_patch_feature_extraction_job(
            coords_dir=cfg.coords_subdir,
            patch_encoder=encoder,
            device=cfg.device,
            saveas="h5",
            batch_limit=cfg.feat_batch_size or cfg.batch_size,
        )
    else:
        from trident.slide_encoder_models.load import encoder_factory

        encoder = encoder_factory(cfg.slide_encoder)
        processor.run_slide_feature_extraction_job(
            slide_encoder=encoder,
            coords_dir=cfg.coords_subdir,
            device=cfg.device,
            saveas="h5",
            batch_limit=cfg.feat_batch_size or cfg.batch_size,
        )


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_pipeline(
    cfg: TridentExtractionConfig,
    tasks: list[str] | None = None,
    diff_mode: bool = False,
    shard_id: int | None = None,
    total_shards: int | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> Path | None:
    if tasks is None:
        tasks = ["seg", "coords", "feat"]
    if (shard_id is None) != (total_shards is None):
        raise ValidationError("shard_id and total_shards must be provided together")

    start_time = time.monotonic()
    requested_cfg = cfg
    archive_dirs: list[str] = []

    # Step 1: Validate
    logger.info("Validating inputs...")
    summary = validate_inputs(cfg, tasks, create_job_dir=not dry_run)
    encoder_fp = compute_encoder_fingerprint(cfg)
    stage_fingerprints = compute_stage_fingerprints(cfg)
    extraction_fp = _fingerprint_payload(stage_fingerprints)
    summary["encoder_fingerprint"] = encoder_fp
    summary["extraction_fingerprint"] = extraction_fp

    logger.info(
        f"Validation passed: {summary['slide_count']} slides, "
        f"encoder={summary['encoder']} (fp={encoder_fp[:8]}...), "
        f"device={summary['device']}, tasks={summary['tasks']}"
    )

    if dry_run:
        logger.info("DRY RUN — validation passed, exiting before compute")
        _print_dry_run_summary(summary)
        return None

    # Step 2: Slurm sharding. Keep requested_cfg intact for provenance and,
    # outside Slurm, full-cohort completeness validation.
    execution_cfg = cfg
    validation_cfg = requested_cfg
    allowed_feature_ids = {record.output_id for record in _load_records(requested_cfg)}
    if shard_id is not None and total_shards is not None:
        shard_list = shard_wsi_list(execution_cfg, shard_id, total_shards)
        execution_cfg = TridentExtractionConfig(
            **{**asdict(execution_cfg), "custom_list_of_wsis": str(shard_list)}
        )
        validation_cfg = execution_cfg
        if not _csv_has_records(shard_list):
            _assert_inputs_unchanged(requested_cfg, stage_fingerprints, extraction_fp)
            total_elapsed = time.monotonic() - start_time
            _write_manifest(
                requested_cfg,
                encoder_fp,
                extraction_fp,
                tasks,
                total_elapsed,
                shard_id,
                total_shards,
                execution_cfg=execution_cfg,
                selected_slide_count=0,
                stage_fingerprints=stage_fingerprints,
            )
            logger.info("Slurm shard is empty; completed as a clean no-op")
            return Path(requested_cfg.job_dir)

    # Step 3: Prove partial-run lineage and guard pinned TRIDENT's filename cache.
    run_manifest_path = _run_manifest_path(requested_cfg, shard_id)
    previous_manifest = _read_manifest(run_manifest_path)
    execution_records = _load_records(execution_cfg)
    _require_current_prerequisites(
        execution_cfg,
        execution_records,
        tasks,
        previous_manifest,
        stage_fingerprints,
    )
    previous_fingerprint = (
        previous_manifest.get("extraction_fingerprint") if previous_manifest is not None else None
    )
    if (
        not diff_mode
        and not force
        and previous_fingerprint != extraction_fp
        and _has_existing_artifacts(execution_cfg, execution_records, tasks)
    ):
        raise RuntimeError(
            "Existing extraction artifacts do not match the current slides/configuration, "
            "and pinned TRIDENT would skip them by filename. Rerun with diff_mode=true for "
            "a recoverable stale-only rebuild or force=true for the selected stages."
        )

    # Step 4: Diff/force invalidation.
    if diff_mode and not force:
        diff_list = compute_extraction_diff(
            execution_cfg,
            extraction_fp,
            tasks=tasks,
            manifest_path=run_manifest_path,
        )
        if diff_list is None:
            if requested_cfg.require_complete:
                _raise_for_incomplete_outputs(
                    validate_outputs(
                        validation_cfg,
                        tasks,
                        allowed_feature_ids=allowed_feature_ids,
                    )
                )
            _assert_inputs_unchanged(requested_cfg, stage_fingerprints, extraction_fp)
            return Path(requested_cfg.job_dir)
        execution_cfg = TridentExtractionConfig(
            **{**asdict(execution_cfg), "custom_list_of_wsis": str(diff_list)}
        )
        archive_path = _archive_selected_artifacts(
            execution_cfg,
            _load_records(execution_cfg),
            tasks,
            reason=f"diff_{extraction_fp}",
        )
        if archive_path is not None:
            archive_dirs.append(str(archive_path))
    elif force:
        archive_path = _archive_selected_artifacts(
            execution_cfg,
            _load_records(execution_cfg),
            tasks,
            reason=f"force_{extraction_fp}",
        )
        if archive_path is not None:
            archive_dirs.append(str(archive_path))

    # Step 5: Run tasks
    task_fn = {
        "seg": run_segmentation,
        "coords": run_patching,
        "feat": run_feature_extraction,
    }

    processor = create_processor(execution_cfg)

    for task_name in tasks:
        if task_name not in task_fn:
            raise ValueError(f"Unknown task: {task_name}. Must be one of {list(task_fn)}")
        task_start = time.monotonic()
        logger.info(f"{'=' * 60}")
        logger.info(f"  Starting task: {task_name}")
        logger.info(f"{'=' * 60}")
        task_fn[task_name](processor, execution_cfg)
        logger.info(f"  Task '{task_name}' completed in {time.monotonic() - task_start:.1f}s")

    output_validation = validate_outputs(
        validation_cfg,
        tasks,
        allowed_feature_ids=allowed_feature_ids,
    )
    run_complete = not any(output_validation[key] for key in ("missing", "invalid", "unexpected"))
    if requested_cfg.require_complete:
        _raise_for_incomplete_outputs(output_validation)
    completed_stage_fingerprints = _derive_completed_stage_fingerprints(
        previous_manifest,
        stage_fingerprints,
        tasks,
        run_complete=run_complete,
    )
    _assert_inputs_unchanged(requested_cfg, stage_fingerprints, extraction_fp)

    # Step 6: Provenance
    total_elapsed = time.monotonic() - start_time
    _write_manifest(
        requested_cfg,
        encoder_fp,
        extraction_fp,
        tasks,
        total_elapsed,
        shard_id,
        total_shards,
        execution_cfg=execution_cfg,
        selected_slide_count=len(_load_records(execution_cfg)),
        archive_dirs=archive_dirs,
        stage_fingerprints=stage_fingerprints,
        completed_stage_fingerprints=completed_stage_fingerprints,
    )
    logger.info(f"Pipeline completed in {total_elapsed:.1f}s → {requested_cfg.job_dir}")
    return Path(requested_cfg.job_dir)


# ── Provenance ────────────────────────────────────────────────────────────────


def _write_manifest(
    cfg: TridentExtractionConfig,
    encoder_fingerprint: str,
    extraction_fingerprint: str,
    tasks: list[str],
    elapsed_seconds: float,
    shard_id: int | None = None,
    total_shards: int | None = None,
    *,
    execution_cfg: TridentExtractionConfig | None = None,
    selected_slide_count: int | None = None,
    archive_dirs: list[str] | None = None,
    stage_fingerprints: dict[str, str] | None = None,
    completed_stage_fingerprints: dict[str, str] | None = None,
) -> None:
    git_sha = _get_git_sha()
    execution_cfg = execution_cfg or cfg

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "encoder": cfg.encoder_name,
        "encoder_type": cfg.encoder_type,
        "encoder_fingerprint": encoder_fingerprint,
        "extraction_fingerprint": extraction_fingerprint,
        "stage_fingerprints": stage_fingerprints or compute_stage_fingerprints(cfg),
        "completed_stage_fingerprints": completed_stage_fingerprints or {},
        "magnification": cfg.mag,
        "target_mpp": cfg.target_mpp,
        "sampling_mode": "exact_mpp" if cfg.target_mpp is not None else "magnification",
        "patch_size": cfg.patch_size,
        "overlap": cfg.overlap,
        "segmenter": cfg.segmenter,
        "seg_conf_thresh": cfg.seg_conf_thresh,
        "wsi_dir": cfg.wsi_dir,
        "tasks_run": tasks,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "extracted_at": datetime.datetime.now().isoformat(),
        "device": cfg.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "git_sha": git_sha,
        "h5_schema_version": H5_SCHEMA_VERSION,
        "full_config": asdict(cfg),
        "requested_selection": {
            "path": cfg.custom_list_of_wsis,
            "fingerprint": (
                _hash_file(Path(cfg.custom_list_of_wsis))
                if cfg.custom_list_of_wsis and Path(cfg.custom_list_of_wsis).is_file()
                else None
            ),
        },
        "execution_selection": {
            "path": execution_cfg.custom_list_of_wsis,
            "fingerprint": (
                _hash_file(Path(execution_cfg.custom_list_of_wsis))
                if execution_cfg.custom_list_of_wsis
                and Path(execution_cfg.custom_list_of_wsis).is_file()
                else None
            ),
            "slide_count": selected_slide_count,
        },
        "archived_artifact_dirs": archive_dirs or [],
    }

    if shard_id is not None:
        manifest["shard"] = {"id": shard_id, "total": total_shards}
        manifest_name = f"manifest_shard_{shard_id:04d}.json"
    else:
        manifest_name = "manifest.json"

    manifest_path = Path(cfg.job_dir) / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, default=str))
    tmp_path.replace(manifest_path)
    logger.info(f"Manifest saved: {manifest_path}")


# ── Output validation ─────────────────────────────────────────────────────────


def validate_outputs(
    cfg: TridentExtractionConfig,
    tasks: list[str],
    *,
    allowed_feature_ids: set[str] | None = None,
) -> dict:
    required_tasks = set(tasks) & {"seg", "coords", "feat"}
    if not required_tasks:
        return {"total": 0, "found": 0, "missing": [], "invalid": [], "unexpected": []}

    records = _load_records(cfg)
    if (
        cfg.target_mpp is not None
        and cfg.custom_list_of_wsis is not None
        and any(record.mpp is None for record in records)
    ):
        raise ValidationError(
            "Exact-MPP output validation requires a numeric 'mpp' column for every "
            "row of the custom WSI CSV"
        )

    expected_records = {record.output_id: record for record in records}
    expected = set(expected_records)
    allowed_features = expected if allowed_feature_ids is None else set(allowed_feature_ids)
    if not expected.issubset(allowed_features):
        raise ValidationError("allowed_feature_ids must include every selected slide")
    missing_ids: set[str] = set()
    invalid: list[str] = []

    if "seg" in required_tasks:
        for record in records:
            reason, is_missing = _validate_segmentation_output(cfg, record)
            if is_missing:
                missing_ids.add(record.output_id)
            elif reason is not None:
                invalid.append(f"{record.output_id}: segmentation: {reason}")

    if "coords" in required_tasks:
        for record in records:
            coords_path = (
                Path(cfg.job_dir) / cfg.coords_subdir / "patches" / f"{record.output_id}_patches.h5"
            )
            if not coords_path.is_file():
                missing_ids.add(record.output_id)
                continue
            reason = _validate_coordinate_output(coords_path, record, cfg)
            if reason is not None:
                invalid.append(f"{record.output_id}: coordinates: {reason}")

    unexpected: list[str] = []
    if "feat" in required_tasks:
        h5_dir = _feature_h5_dir(cfg)
        feature_ids = {path.stem for path in h5_dir.glob("*.h5")} if h5_dir.is_dir() else set()
        missing_ids.update(expected - feature_ids)
        unexpected = sorted(feature_ids - allowed_features)

        for slide_id in sorted(expected & feature_ids):
            feature_path = h5_dir / f"{slide_id}.h5"
            if cfg.target_mpp is not None:
                validator = (
                    _validate_exact_slide_feature
                    if cfg.encoder_type == "slide"
                    else _validate_exact_patch_feature
                )
                reason = validator(feature_path, expected_records[slide_id], cfg)
            else:
                reason = _validate_generic_feature(feature_path, cfg.encoder_type)
            if reason is not None:
                invalid.append(f"{slide_id}: features: {reason}")

    missing = sorted(missing_ids)
    invalid_ids = {item.split(":", 1)[0] for item in invalid}
    result = {
        "total": len(expected),
        "found": len(expected - missing_ids - invalid_ids),
        "missing": missing,
        "invalid": invalid,
        "unexpected": unexpected,
    }

    if missing or invalid or unexpected:
        logger.warning(
            "Output validation: %d/%d slides missing, %d invalid, %d unexpected",
            len(missing),
            len(expected),
            len(invalid),
            len(unexpected),
        )
    else:
        logger.info("Output validation: all %d selected slides are complete", len(expected))

    return result


def _raise_for_incomplete_outputs(validation: dict[str, Any]) -> None:
    missing = validation["missing"]
    invalid = validation["invalid"]
    unexpected = validation["unexpected"]
    if not missing and not invalid and not unexpected:
        return
    raise RuntimeError(
        "Feature extraction is incomplete or contaminated: "
        f"{len(missing)} missing, {len(invalid)} invalid, and "
        f"{len(unexpected)} unexpected out of {validation['total']} expected slides. "
        "Inspect processor errors and use an isolated job directory for each cohort selection."
    )


def _validate_segmentation_output(
    cfg: TridentExtractionConfig,
    record: SlideRecord,
) -> tuple[str | None, bool]:
    geojson_path = Path(cfg.job_dir) / "contours_geojson" / f"{record.output_id}.geojson"
    preview_path = Path(cfg.job_dir) / "contours" / f"{record.output_id}.jpg"
    if not geojson_path.is_file() or not preview_path.is_file():
        return None, True
    try:
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))
        features = payload.get("features")
        if not isinstance(features, list) or not features:
            return "GeoJSON contains no tissue features", False
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
        return str(exc), False
    return None, False


def _validate_coordinate_output(
    path: Path,
    record: SlideRecord,
    cfg: TridentExtractionConfig,
) -> str | None:
    import h5py

    try:
        if cfg.target_mpp is not None:
            from oceanpath.extraction.mpp_sampling import (
                validate_exact_mpp_coordinate_file,
            )

            attributes = validate_exact_mpp_coordinate_file(
                path,
                target_mpp=cfg.target_mpp,
                source_mpp=record.mpp,
                patch_size=cfg.patch_size,
                overlap=cfg.overlap,
                min_tissue_proportion=cfg.min_tissue_proportion,
            )
            if int(attributes["patch_size_level0"]) <= 0:
                return "invalid level-0 footprint"
        with h5py.File(path, "r") as handle:
            if "coords" not in handle:
                return "missing 'coords' dataset"
            coords = handle["coords"]
            if coords.ndim != 2 or coords.shape[1] != 2:
                return f"coords must have shape (N, 2), got {coords.shape}"
            if coords.shape[0] == 0:
                return "contains no tissue coordinates"
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return str(exc)
    return None


def _validate_generic_feature(path: Path, encoder_type: str) -> str | None:
    """Validate the TRIDENT H5 shape contract independently of sampling mode."""

    import h5py
    import numpy as np

    try:
        with h5py.File(path, "r") as handle:
            if "features" not in handle:
                return "missing 'features' dataset"
            if "coords" not in handle:
                return "missing 'coords' dataset"
            features = handle["features"]
            coords = handle["coords"]
            if not np.issubdtype(features.dtype, np.number):
                return f"features must be numeric, got {features.dtype}"
            if features.size == 0:
                return "contains no features"
            if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
                return f"coords must have nonempty shape (N, 2), got {coords.shape}"
            if encoder_type == "patch":
                if features.ndim != 2:
                    return f"patch features must be 2-D, got shape {features.shape}"
                if features.shape[0] != coords.shape[0]:
                    return (
                        f"feature/coordinate row mismatch: {features.shape[0]} != {coords.shape[0]}"
                    )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return str(exc)
    return None


def _validate_exact_patch_feature(
    path: Path,
    record: SlideRecord,
    cfg: TridentExtractionConfig,
) -> str | None:
    """Return a concise error when an exact-MPP patch feature file is unsafe."""

    import h5py

    from oceanpath.extraction.mpp_sampling import (
        ExactMppSamplingError,
        validate_exact_mpp_coordinate_file,
    )

    if cfg.target_mpp is None:
        return "exact-MPP validation requested without target_mpp"
    if generic_reason := _validate_generic_feature(path, "patch"):
        return generic_reason

    try:
        attributes = validate_exact_mpp_coordinate_file(
            path,
            target_mpp=cfg.target_mpp,
            source_mpp=record.mpp,
            patch_size=cfg.patch_size,
            overlap=cfg.overlap,
            min_tissue_proportion=cfg.min_tissue_proportion,
        )
        with h5py.File(path, "r") as handle:
            if "features" not in handle:
                return "missing 'features' dataset"
            features = handle["features"]
            coords = handle["coords"]
            if features.ndim != 2:
                return f"features must be 2-D, got shape {features.shape}"
            if coords.ndim != 2 or coords.shape[1] != 2:
                return f"coords must have shape (N, 2), got {coords.shape}"
            if features.shape[0] != coords.shape[0]:
                return f"feature/coordinate row mismatch: {features.shape[0]} != {coords.shape[0]}"
            if features.shape[0] == 0:
                return "contains no patch features"
            if int(attributes["actual_read_level0_pixels"]) != int(attributes["patch_size_level0"]):
                return "recorded read footprint differs from coordinate stride"
    except (ExactMppSamplingError, OSError, KeyError, TypeError, ValueError) as exc:
        return str(exc)
    return None


def _validate_exact_slide_feature(
    path: Path,
    record: SlideRecord,
    cfg: TridentExtractionConfig,
) -> str | None:
    """Validate that a slide embedding was derived from exact-MPP patch metadata."""

    import h5py
    import numpy as np

    from oceanpath.extraction.mpp_sampling import (
        ExactMppSamplingError,
        validate_exact_mpp_coordinate_file,
    )

    if cfg.target_mpp is None:
        return "exact-MPP validation requested without target_mpp"
    if generic_reason := _validate_generic_feature(path, "slide"):
        return generic_reason
    try:
        validate_exact_mpp_coordinate_file(
            path,
            target_mpp=cfg.target_mpp,
            source_mpp=record.mpp,
            patch_size=cfg.patch_size,
            overlap=cfg.overlap,
            min_tissue_proportion=cfg.min_tissue_proportion,
        )
        with h5py.File(path, "r") as handle:
            if "features" not in handle:
                return "missing 'features' dataset"
            features = handle["features"]
            coords = handle["coords"]
            if features.size == 0:
                return "contains no slide features"
            if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
                return f"coords must have nonempty shape (N, 2), got {coords.shape}"
            if not np.issubdtype(features.dtype, np.number):
                return f"features must be numeric, got {features.dtype}"
    except (ExactMppSamplingError, OSError, KeyError, TypeError, ValueError) as exc:
        return str(exc)
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_git_sha() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _print_dry_run_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("  DRY RUN SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key:>25s}: {value}")
    print("=" * 60 + "\n")
