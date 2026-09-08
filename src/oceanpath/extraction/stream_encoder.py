"""Per-slide native-geometry encoder used by the live streaming queue.

The queue deliberately does not call :func:`oceanpath.extraction.run_pipeline`.
That API fingerprints a frozen cohort selection, whereas the streaming
inventory grows every time a producer commits another slide.  This module
instead owns immutable per-stage receipts for one source snapshot and uses
TRIDENT's low-level processor jobs with the exact-MPP WSI adapter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import h5py
import numpy as np
import torch

from oceanpath.extraction.mpp_sampling import validate_exact_mpp_coordinate_file
from oceanpath.extraction.slide_records import SlideRecord, write_slide_records_csv
from oceanpath.extraction.trident import TridentExtractionConfig, create_processor

logger = logging.getLogger(__name__)

# Frozen cache identity folded into every stage receipt via ``stage_keys``.
#
# Historically this was computed live as a hash of (this module, mpp_sampling.py,
# trident.py, uv.lock), so ANY edit to those files — including pure refactors —
# silently invalidated every committed slide and re-encoded the whole cohort on
# the next watcher restart. The value below is the exact fingerprint the
# production stream has stamped into receipts since the v2 receipt migration.
#
# Bump this ONLY when encoding semantics change (a given source snapshot would
# produce different artifacts); doing so deliberately re-encodes all completed
# slides on the next reconcile.
IMPLEMENTATION_HASH = "c186b18b06c755fead1c3f6ea9a7e8a9a2ce167e6b2a8c04525599d79fd482de"


class ReadySlideLike(Protocol):
    """Inventory fields consumed by the encoder."""

    @property
    def wsi(self) -> str: ...

    @property
    def path(self) -> Path: ...

    @property
    def output_id(self) -> str: ...

    @property
    def mpp(self) -> float: ...

    @property
    def cohort(self) -> str: ...


class SlideEncodingError(RuntimeError):
    """Raised when a stage cannot be safely committed."""


class SourceChangedError(SlideEncodingError):
    """Raised when a producer-visible source changes during processing."""


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    name: str
    patch_size: int
    feature_dim: int
    checkpoint_path: Path
    batch_size: int

    @property
    def coords_dir(self) -> str:
        return f"20x_{self.patch_size}px_0px_overlap_mpp0.5"


@dataclass(frozen=True, slots=True)
class SlideEncoderConfig:
    output_root: Path
    state_root: Path
    scratch_root: Path
    hest_checkpoint_path: Path
    uni_checkpoint_path: Path
    conch_v15_checkpoint_path: Path
    virchow2_checkpoint_path: Path | None = None
    target_mpp: float = 0.5
    target_mag: int = 20
    overlap: int = 0
    min_tissue_proportion: float = 0.5
    segmentation_confidence: float = 0.5
    remove_holes: bool = True
    segmentation_batch_size: int = 16
    uni_batch_size: int = 64
    conch_v15_batch_size: int = 32
    virchow2_batch_size: int = 8
    max_workers: int = 6
    low_memory_workers: int = 4
    low_memory_threshold_gib: float = 40.0
    scratch_reserve_gib: float = 30.0
    stage_locally: bool = True
    gpu: int = 0

    @property
    def encoders(self) -> tuple[EncoderSpec, ...]:
        specs = [
            EncoderSpec(
                name="uni_v1",
                patch_size=256,
                feature_dim=1024,
                checkpoint_path=self.uni_checkpoint_path,
                batch_size=self.uni_batch_size,
            ),
            EncoderSpec(
                name="conch_v15",
                patch_size=512,
                feature_dim=768,
                checkpoint_path=self.conch_v15_checkpoint_path,
                batch_size=self.conch_v15_batch_size,
            ),
        ]
        if self.virchow2_checkpoint_path is not None:
            specs.append(
                EncoderSpec(
                    name="virchow2",
                    patch_size=224,
                    feature_dim=2560,
                    checkpoint_path=self.virchow2_checkpoint_path,
                    batch_size=self.virchow2_batch_size,
                )
            )
        return tuple(specs)


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    wsi: str
    path: Path
    output_id: str
    cohort: str
    mpp: float
    size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int

    @classmethod
    def capture(cls, slide: ReadySlideLike) -> SourceSnapshot:
        path = Path(slide.path).expanduser().resolve(strict=True)
        stat = path.stat()
        mpp = float(slide.mpp)
        if not math.isfinite(mpp) or mpp <= 0:
            raise SlideEncodingError(f"Invalid source MPP for {slide.wsi}: {slide.mpp!r}")
        if path.name.endswith(".part") or ".part." in path.name:
            raise SlideEncodingError(f"Refusing an uncommitted producer file: {path}")
        return cls(
            wsi=str(slide.wsi),
            path=path,
            output_id=str(slide.output_id),
            cohort=str(slide.cohort),
            mpp=mpp,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            ctime_ns=stat.st_ctime_ns,
            device=stat.st_dev,
            inode=stat.st_ino,
        )

    @property
    def durable_identity(self) -> dict[str, Any]:
        """Return source facts that survive a WSL/DrvFS remount.

        ``ctime``, device, and inode remain part of the in-process mutation
        guard in :meth:`assert_unchanged`, but DrvFS may assign new values for
        them whenever WSL restarts.  They therefore cannot participate in a
        durable queue or receipt key.
        """

        return {
            "schema_version": 2,
            "wsi": self.wsi,
            "path": self.path,
            "output_id": self.output_id,
            "cohort": self.cohort,
            "mpp": self.mpp,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }

    @property
    def key(self) -> str:
        return _fingerprint(self.durable_identity)

    def assert_unchanged(self) -> None:
        try:
            current = self.path.stat()
        except FileNotFoundError as exc:
            raise SourceChangedError(f"Source disappeared during encoding: {self.path}") from exc
        observed = (
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
            current.st_dev,
            current.st_ino,
        )
        expected = (self.size, self.mtime_ns, self.ctime_ns, self.device, self.inode)
        if observed != expected:
            raise SourceChangedError(
                f"Source changed during encoding: {self.path}; expected={expected}, observed={observed}"
            )


@dataclass(frozen=True, slots=True)
class SlideEncodingResult:
    output_id: str
    source_key: str
    config_key: str
    elapsed_seconds: float
    stage_metrics: dict[str, dict[str, Any]]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def _fingerprint(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, default=_json_default, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _available_memory_gib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    return float("inf")


class SlideEncoder:
    """Encode one immutable slide snapshot with one shared HEST segmentation."""

    def __init__(self, cfg: SlideEncoderConfig) -> None:
        self.cfg = cfg
        self.cfg.output_root.mkdir(parents=True, exist_ok=True)
        self.cfg.state_root.mkdir(parents=True, exist_ok=True)
        self.cfg.scratch_root.mkdir(parents=True, exist_ok=True)
        self._validate_config()
        self._weight_hashes = {
            "hest": sha256_file(cfg.hest_checkpoint_path),
            "uni_v1": sha256_file(cfg.uni_checkpoint_path),
            "conch_v15": sha256_file(cfg.conch_v15_checkpoint_path),
        }
        if cfg.virchow2_checkpoint_path is not None:
            self._weight_hashes["virchow2"] = sha256_file(cfg.virchow2_checkpoint_path)
        self._implementation_hash = IMPLEMENTATION_HASH
        self._models: dict[str, Any] = {}
        self._stage_batches = {
            "seg": cfg.segmentation_batch_size,
            "uni_v1_feat": cfg.uni_batch_size,
            "conch_v15_feat": cfg.conch_v15_batch_size,
        }
        if cfg.virchow2_checkpoint_path is not None:
            self._stage_batches["virchow2_feat"] = cfg.virchow2_batch_size

    def _validate_config(self) -> None:
        if not math.isclose(self.cfg.target_mpp, 0.5, rel_tol=0.0, abs_tol=1e-12):
            raise SlideEncodingError("The colon native contract requires target_mpp=0.5")
        if self.cfg.target_mag != 20 or self.cfg.overlap != 0:
            raise SlideEncodingError("The colon native contract requires 20x and zero overlap")
        if not 0 <= self.cfg.min_tissue_proportion <= 1:
            raise SlideEncodingError("min_tissue_proportion must be in [0, 1]")
        if not isinstance(self.cfg.remove_holes, bool):
            raise SlideEncodingError("remove_holes must be a boolean")
        required_paths = [
            self.cfg.hest_checkpoint_path,
            self.cfg.uni_checkpoint_path,
            self.cfg.conch_v15_checkpoint_path,
        ]
        if self.cfg.virchow2_checkpoint_path is not None:
            required_paths.append(self.cfg.virchow2_checkpoint_path)
        for path in required_paths:
            if not path.is_file():
                raise SlideEncodingError(f"Required checkpoint does not exist: {path}")

    @property
    def segmentation_policy(self) -> dict[str, Any]:
        """Return the complete, fingerprinted colon segmentation contract."""

        return {
            "segmenter": "hest",
            "confidence": self.cfg.segmentation_confidence,
            "target_mag": 10,
            "reader": "openslide",
            "remove_holes": self.cfg.remove_holes,
            "holes_are_tissue": not self.cfg.remove_holes,
            # Artifact cleanup is deliberately outside the approved colon
            # production contract. Keep both user-facing flags and the direct
            # TRIDENT argument explicit in receipts and cache identity.
            "remove_artifacts": False,
            "remove_penmarks": False,
            "artifact_remover_model": None,
        }

    def stage_keys(self, snapshot: SourceSnapshot) -> dict[str, str]:
        seg = _fingerprint(
            {
                "source": snapshot.key,
                "implementation": self._implementation_hash,
                "checkpoint": self._weight_hashes["hest"],
                "policy": self.segmentation_policy,
            }
        )
        result = {"seg": seg}
        for spec in self.cfg.encoders:
            coords = _fingerprint(
                {
                    "seg": seg,
                    "mpp": snapshot.mpp,
                    "target_mpp": self.cfg.target_mpp,
                    "target_mag": self.cfg.target_mag,
                    "patch_size": spec.patch_size,
                    "overlap": self.cfg.overlap,
                    "min_tissue_proportion": self.cfg.min_tissue_proportion,
                }
            )
            result[f"{spec.name}_coords"] = coords
            result[f"{spec.name}_feat"] = _fingerprint(
                {
                    "coords": coords,
                    "encoder": spec.name,
                    "feature_dim": spec.feature_dim,
                    "checkpoint": self._weight_hashes[spec.name],
                }
            )
        return result

    def config_key(self, snapshot: SourceSnapshot) -> str:
        return _fingerprint(self.stage_keys(snapshot))

    def validate_complete(self, slide: ReadySlideLike, *, deep: bool = True) -> bool:
        """Return whether every encoder lineage is complete for this snapshot.

        ``deep=False`` is suitable for periodic queue reconciliation: it checks
        immutable receipt identity, nonempty artifacts, and locks without
        rereading every embedding. Stage commit and cache reuse always use the
        full validators.
        """

        try:
            snapshot = SourceSnapshot.capture(slide)
            keys = self.stage_keys(snapshot)
            completion = _load_json(self._completion_path(snapshot.output_id))
            if not completion or completion.get("source_key") != snapshot.key:
                return False
            if completion.get("config_key") != self.config_key(snapshot):
                return False
            if not self._stage_receipt_matches(snapshot, "seg", keys["seg"]):
                return False
            if deep:
                self._validate_segmentation(snapshot.output_id)
            else:
                self._validate_stage_artifacts_present(snapshot.output_id, "seg")
            for spec in self.cfg.encoders:
                coords_stage = f"{spec.name}_coords"
                feat_stage = f"{spec.name}_feat"
                if not self._stage_receipt_matches(snapshot, coords_stage, keys[coords_stage]):
                    return False
                if not self._stage_receipt_matches(snapshot, feat_stage, keys[feat_stage]):
                    return False
                if deep:
                    self._validate_coordinates(snapshot, spec)
                    self._validate_features(snapshot, spec)
                else:
                    self._validate_stage_artifacts_present(snapshot.output_id, coords_stage)
                    self._validate_stage_artifacts_present(snapshot.output_id, feat_stage)
            snapshot.assert_unchanged()
            return True
        except Exception:
            return False

    @property
    def checkpoint_hashes(self) -> dict[str, str]:
        return dict(self._weight_hashes)

    @property
    def implementation_hash(self) -> str:
        return self._implementation_hash

    def recover_orphan_work_files(self) -> int:
        """Archive locks/atomic temporaries left by an interrupted watcher.

        Call this only while holding the watcher's process-wide lock. The
        ``colon_stream`` feature root is dedicated to this watcher; generic
        TRIDENT jobs must use another root.
        """

        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        recovery_root = self.cfg.output_root / ".orphan_work" / stamp
        candidates = [
            path
            for path in self.cfg.output_root.rglob("*")
            if path.is_file()
            and (
                path.name.endswith(".lock")
                or (path.name.startswith(".") and path.name.endswith(".tmp"))
            )
            and ".archive" not in path.parts
            and ".orphan_work" not in path.parts
        ]
        for path in candidates:
            relative = path.relative_to(self.cfg.output_root)
            destination = recovery_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}")
            os.replace(path, destination)
        return len(candidates)

    def load_models(self) -> None:
        """Load segmentation and all patch encoders once for process reuse."""

        if self._models:
            return
        if not torch.cuda.is_available():
            raise SlideEncodingError("CUDA is required for the live colon encoder")
        from trident.patch_encoder_models.load import encoder_factory
        from trident.segmentation_models import load as segmentation_load

        original_get_weights_path = segmentation_load.get_weights_path

        def explicit_hest(model_type: str, model_name: str) -> str:
            if model_type == "seg" and model_name == "hest":
                return str(self.cfg.hest_checkpoint_path)
            return cast("str", original_get_weights_path(model_type, model_name))

        segmentation_load.get_weights_path = explicit_hest
        try:
            hest = segmentation_load.segmentation_model_factory(
                "hest", confidence_thresh=self.cfg.segmentation_confidence
            )
        finally:
            segmentation_load.get_weights_path = original_get_weights_path

        self._models = {
            "hest": hest,
            "uni_v1": encoder_factory("uni_v1", weights_path=str(self.cfg.uni_checkpoint_path)),
            "conch_v15": encoder_factory(
                "conch_v15", weights_path=str(self.cfg.conch_v15_checkpoint_path)
            ),
        }
        if self.cfg.virchow2_checkpoint_path is not None:
            self._models["virchow2"] = encoder_factory(
                "virchow2", weights_path=str(self.cfg.virchow2_checkpoint_path)
            )
        logger.info(
            "Loaded frozen encoder models %s with checkpoint hashes %s",
            sorted(self._models),
            self._weight_hashes,
        )

    def process(self, slide: ReadySlideLike) -> SlideEncodingResult:
        started = time.monotonic()
        snapshot = SourceSnapshot.capture(slide)
        keys = self.stage_keys(snapshot)
        needed = self._prepare_stages(snapshot, keys)
        metrics: dict[str, dict[str, Any]] = {}
        if not any(needed.values()):
            result = SlideEncodingResult(
                output_id=snapshot.output_id,
                source_key=snapshot.key,
                config_key=self.config_key(snapshot),
                elapsed_seconds=time.monotonic() - started,
                stage_metrics={"cache": {"reused": True}},
            )
            self._write_completion_receipt(snapshot, keys, result)
            return result

        self.load_models()
        with self._staged_source(snapshot) as staged_path:
            processor = self._create_single_slide_processor(snapshot, staged_path)
            if needed["seg"]:
                metrics["seg"] = self._run_gpu_stage(
                    "seg",
                    lambda batch_size: self._run_segmentation_job(processor, batch_size),
                )
                snapshot.assert_unchanged()
                validation = self._validate_segmentation(snapshot.output_id)
                metrics["seg"].update(validation)
                self._commit_stage(snapshot, "seg", keys["seg"], validation)

            for spec in self.cfg.encoders:
                coords_stage = f"{spec.name}_coords"
                feat_stage = f"{spec.name}_feat"
                if needed[coords_stage]:
                    stage_started = time.monotonic()
                    processor.run_patching_job(
                        target_magnification=self.cfg.target_mag,
                        patch_size=spec.patch_size,
                        overlap=self.cfg.overlap,
                        saveto=spec.coords_dir,
                        min_tissue_proportion=self.cfg.min_tissue_proportion,
                    )
                    snapshot.assert_unchanged()
                    validation = self._validate_coordinates(snapshot, spec)
                    metrics[coords_stage] = {
                        "seconds": time.monotonic() - stage_started,
                        **validation,
                    }
                    self._commit_stage(snapshot, coords_stage, keys[coords_stage], validation)

                if needed[feat_stage]:

                    def _extract_features(batch_size: int, spec: EncoderSpec = spec) -> Any:
                        return processor.run_patch_feature_extraction_job(
                            coords_dir=spec.coords_dir,
                            patch_encoder=self._models[spec.name],
                            device=f"cuda:{self.cfg.gpu}",
                            saveas="h5",
                            batch_limit=batch_size,
                        )

                    metrics[feat_stage] = self._run_gpu_stage(feat_stage, _extract_features)
                    snapshot.assert_unchanged()
                    validation = self._validate_features(snapshot, spec)
                    metrics[feat_stage].update(validation)
                    self._commit_stage(snapshot, feat_stage, keys[feat_stage], validation)

        snapshot.assert_unchanged()
        result = SlideEncodingResult(
            output_id=snapshot.output_id,
            source_key=snapshot.key,
            config_key=self.config_key(snapshot),
            elapsed_seconds=time.monotonic() - started,
            stage_metrics=metrics,
        )
        self._write_completion_receipt(snapshot, keys, result)
        return result

    def _create_single_slide_processor(self, snapshot: SourceSnapshot, staged_path: Path) -> Any:
        selection_dir = self.cfg.scratch_root / "selections"
        selection = selection_dir / f"{snapshot.output_id}.{uuid.uuid4().hex}.csv"
        staged_record = SlideRecord(
            wsi=staged_path.name,
            path=staged_path.resolve(strict=True),
            output_id=snapshot.output_id,
            mpp=snapshot.mpp,
        )
        write_slide_records_csv([staged_record], selection, include_mpp=True)
        workers = (
            self.cfg.low_memory_workers
            if _available_memory_gib() < self.cfg.low_memory_threshold_gib
            else self.cfg.max_workers
        )
        cfg = TridentExtractionConfig(
            wsi_dir=str(staged_path.parent),
            job_dir=str(self.cfg.output_root),
            custom_list_of_wsis=str(selection),
            wsi_ext=[staged_path.suffix.lower()],
            reader_type="openslide",
            search_nested=False,
            segmenter="hest",
            seg_conf_thresh=self.cfg.segmentation_confidence,
            remove_holes=self.cfg.remove_holes,
            remove_artifacts=False,
            remove_penmarks=False,
            mag=self.cfg.target_mag,
            target_mpp=self.cfg.target_mpp,
            patch_size=256,
            overlap=self.cfg.overlap,
            min_tissue_proportion=self.cfg.min_tissue_proportion,
            coords_dir=self.cfg.encoders[0].coords_dir,
            patch_encoder="uni_v1",
            patch_encoder_ckpt_path=str(self.cfg.uni_checkpoint_path),
            gpu=self.cfg.gpu,
            seg_batch_size=self.cfg.segmentation_batch_size,
            feat_batch_size=self.cfg.uni_batch_size,
            max_workers=workers,
            skip_errors=False,
            require_complete=True,
        )
        try:
            processor = create_processor(cfg)
        finally:
            selection.unlink(missing_ok=True)
        if len(processor.wsis) != 1 or processor.wsis[0].name != snapshot.output_id:
            raise SlideEncodingError(
                f"Processor selection mismatch for {snapshot.output_id}: {processor.wsis!r}"
            )
        return processor

    def _run_segmentation_job(self, processor: Any, batch_size: int) -> None:
        """Run HEST with the same policy recorded by ``stage_keys``."""

        processor.run_segmentation_job(
            self._models["hest"],
            seg_mag=self._models["hest"].target_mag,
            holes_are_tissue=not self.cfg.remove_holes,
            artifact_remover_model=None,
            batch_size=batch_size,
            device=f"cuda:{self.cfg.gpu}",
        )

    @contextmanager
    def _staged_source(self, snapshot: SourceSnapshot) -> Iterator[Path]:
        if not self.cfg.stage_locally:
            yield snapshot.path
            return
        usage = shutil.disk_usage(self.cfg.scratch_root)
        reserve = int(self.cfg.scratch_reserve_gib * 1024**3)
        if usage.free - snapshot.size < reserve:
            logger.warning(
                "Not staging %s: scratch free=%d, source=%d, reserve=%d",
                snapshot.output_id,
                usage.free,
                snapshot.size,
                reserve,
            )
            yield snapshot.path
            return

        slide_dir = self.cfg.scratch_root / "slides" / snapshot.key[:16]
        slide_dir.mkdir(parents=True, exist_ok=True)
        staged = slide_dir / snapshot.path.name
        temporary = slide_dir / f".{snapshot.path.name}.{uuid.uuid4().hex}.part"
        try:
            with snapshot.path.open("rb") as source, temporary.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
            if temporary.stat().st_size != snapshot.size:
                raise SourceChangedError(
                    f"Staged size mismatch for {snapshot.output_id}: "
                    f"{temporary.stat().st_size} != {snapshot.size}"
                )
            snapshot.assert_unchanged()
            os.replace(temporary, staged)
            yield staged
        finally:
            temporary.unlink(missing_ok=True)
            staged.unlink(missing_ok=True)
            with suppress(OSError):
                slide_dir.rmdir()

    def _run_gpu_stage(self, stage: str, operation: Callable[[int], Any]) -> dict[str, Any]:
        device = torch.device(f"cuda:{self.cfg.gpu}")
        batch_size = self._stage_batches[stage]
        while True:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            free_before, total = torch.cuda.mem_get_info(device)
            started = time.monotonic()
            try:
                operation(batch_size)
                torch.cuda.synchronize(device)
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                reduced = max(1, batch_size // 2)
                logger.warning(
                    "CUDA OOM in %s at batch=%d; clearing cache and retrying at batch=%d",
                    stage,
                    batch_size,
                    reduced,
                )
                batch_size = reduced
                self._stage_batches[stage] = reduced
                torch.cuda.empty_cache()
        elapsed = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        metrics = {
            "seconds": elapsed,
            "batch_size": batch_size,
            "peak_allocated_mib": peak / 1024**2,
            "free_before_mib": free_before / 1024**2,
            "free_after_mib": free_after / 1024**2,
            "total_vram_mib": total / 1024**2,
        }
        telemetry = self.cfg.state_root / "telemetry.jsonl"
        with telemetry.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": stage, "time": time.time(), **metrics}) + "\n")
            handle.flush()
        return metrics

    def _receipt_path(self, output_id: str, stage: str) -> Path:
        return self.cfg.output_root / "_stream_receipts" / output_id / f"{stage}.json"

    def _completion_path(self, output_id: str) -> Path:
        return self.cfg.output_root / "_stream_receipts" / output_id / "complete.json"

    def _stage_receipt_matches(self, snapshot: SourceSnapshot, stage: str, stage_key: str) -> bool:
        payload = _load_json(self._receipt_path(snapshot.output_id, stage))
        return bool(
            payload
            and payload.get("source_key") == snapshot.key
            and payload.get("stage_key") == stage_key
        )

    def _prepare_stages(self, snapshot: SourceSnapshot, keys: Mapping[str, str]) -> dict[str, bool]:
        needed = {stage: True for stage in keys}
        if self._stage_receipt_matches(snapshot, "seg", keys["seg"]):
            try:
                self._validate_segmentation(snapshot.output_id)
                needed["seg"] = False
            except Exception as exc:
                logger.warning("Invalid cached segmentation for %s: %s", snapshot.output_id, exc)
        if needed["seg"]:
            self._archive_from_stage(snapshot.output_id, "seg")
            return needed

        for spec in self.cfg.encoders:
            coords_stage = f"{spec.name}_coords"
            feat_stage = f"{spec.name}_feat"
            if self._stage_receipt_matches(snapshot, coords_stage, keys[coords_stage]):
                try:
                    self._validate_coordinates(snapshot, spec)
                    needed[coords_stage] = False
                except Exception as exc:
                    logger.warning(
                        "Invalid cached %s coordinates for %s: %s",
                        spec.name,
                        snapshot.output_id,
                        exc,
                    )
            if needed[coords_stage]:
                self._archive_from_stage(snapshot.output_id, coords_stage)
                continue
            if self._stage_receipt_matches(snapshot, feat_stage, keys[feat_stage]):
                try:
                    self._validate_features(snapshot, spec)
                    needed[feat_stage] = False
                except Exception as exc:
                    logger.warning(
                        "Invalid cached %s features for %s: %s",
                        spec.name,
                        snapshot.output_id,
                        exc,
                    )
            if needed[feat_stage]:
                self._archive_from_stage(snapshot.output_id, feat_stage)
        return needed

    def _stage_artifacts(self, output_id: str, stage: str) -> list[Path]:
        if stage == "seg":
            paths = [
                self.cfg.output_root / "contours" / f"{output_id}.jpg",
                self.cfg.output_root / "contours_geojson" / f"{output_id}.geojson",
                self.cfg.output_root / "thumbnails" / f"{output_id}.jpg",
                self._receipt_path(output_id, "seg"),
            ]
            for spec in self.cfg.encoders:
                paths.extend(self._stage_artifacts(output_id, f"{spec.name}_coords"))
            paths.append(self._completion_path(output_id))
            return paths
        encoder_name, kind = stage.rsplit("_", 1)
        spec = next(item for item in self.cfg.encoders if item.name == encoder_name)
        if kind == "coords":
            return [
                self.cfg.output_root / spec.coords_dir / "patches" / f"{output_id}_patches.h5",
                self.cfg.output_root / spec.coords_dir / "visualization" / f"{output_id}.jpg",
                self._receipt_path(output_id, stage),
                *self._stage_artifacts(output_id, f"{encoder_name}_feat"),
            ]
        return [
            self.cfg.output_root / spec.coords_dir / f"features_{spec.name}" / f"{output_id}.h5",
            self._receipt_path(output_id, stage),
            self._completion_path(output_id),
        ]

    def _validate_stage_artifacts_present(self, output_id: str, stage: str) -> None:
        for path in self._stage_artifacts(output_id, stage):
            if path == self._completion_path(output_id):
                continue
            if not path.is_file() or path.stat().st_size <= 0:
                raise SlideEncodingError(f"Missing or empty committed artifact: {path}")
            if Path(f"{path}.lock").exists():
                raise SlideEncodingError(f"Committed artifact is locked: {path}.lock")

    def _archive_from_stage(self, output_id: str, stage: str) -> None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        archive_root = self.cfg.output_root / ".archive" / stamp / output_id
        seen: set[Path] = set()
        for path in self._stage_artifacts(output_id, stage):
            lock = Path(f"{path}.lock")
            if lock.exists():
                raise SlideEncodingError(
                    f"Refusing to invalidate a locked artifact owned by an unknown process: {lock}. "
                    "Do not run another TRIDENT process in the colon_stream root."
                )
            if path in seen or not path.exists():
                continue
            seen.add(path)
            relative = path.relative_to(self.cfg.output_root)
            destination = archive_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}")
            os.replace(path, destination)

    def _validate_segmentation(self, output_id: str) -> dict[str, Any]:
        contours = self.cfg.output_root / "contours" / f"{output_id}.jpg"
        geojson = self.cfg.output_root / "contours_geojson" / f"{output_id}.geojson"
        thumbnail = self.cfg.output_root / "thumbnails" / f"{output_id}.jpg"
        for path in (contours, geojson, thumbnail):
            if not path.is_file() or path.stat().st_size <= 0:
                raise SlideEncodingError(f"Missing or empty segmentation artifact: {path}")
            if Path(f"{path}.lock").exists():
                raise SlideEncodingError(f"Active segmentation lock: {path}.lock")
        payload = json.loads(geojson.read_text(encoding="utf-8"))
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, list) or not features:
            raise SlideEncodingError(f"HEST produced no tissue polygons for {output_id}")
        return {"polygon_count": len(features)}

    def _coordinate_path(self, output_id: str, spec: EncoderSpec) -> Path:
        return self.cfg.output_root / spec.coords_dir / "patches" / f"{output_id}_patches.h5"

    def _feature_path(self, output_id: str, spec: EncoderSpec) -> Path:
        return self.cfg.output_root / spec.coords_dir / f"features_{spec.name}" / f"{output_id}.h5"

    def _validate_coordinates(self, snapshot: SourceSnapshot, spec: EncoderSpec) -> dict[str, Any]:
        path = self._coordinate_path(snapshot.output_id, spec)
        visualization = (
            self.cfg.output_root / spec.coords_dir / "visualization" / f"{snapshot.output_id}.jpg"
        )
        if Path(f"{path}.lock").exists():
            raise SlideEncodingError(f"Active coordinate lock: {path}.lock")
        if not visualization.is_file() or visualization.stat().st_size <= 0:
            raise SlideEncodingError(f"Missing coordinate visualization: {visualization}")
        if Path(f"{visualization}.lock").exists():
            raise SlideEncodingError(f"Active coordinate visualization lock: {visualization}.lock")
        attrs = validate_exact_mpp_coordinate_file(
            path,
            target_mpp=self.cfg.target_mpp,
            source_mpp=snapshot.mpp,
            patch_size=spec.patch_size,
            overlap=self.cfg.overlap,
            min_tissue_proportion=self.cfg.min_tissue_proportion,
        )
        with h5py.File(path, "r") as handle:
            coordinates = np.asarray(handle["coords"][:], dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
            raise SlideEncodingError(f"Invalid or empty coordinate matrix in {path}")
        if np.any(coordinates < 0) or len(np.unique(coordinates, axis=0)) != len(coordinates):
            raise SlideEncodingError(f"Coordinates are negative or duplicated in {path}")
        stride = int(attrs["patch_size_level0"]) - int(attrs["overlap_level0"])
        if stride <= 0 or np.any(coordinates % stride):
            raise SlideEncodingError(f"Coordinates are not on the exact level-0 lattice in {path}")
        native = int(attrs["patch_size_level0"])
        physical_width = native * snapshot.mpp
        return {
            "patch_count": len(coordinates),
            "native_pixels": native,
            "physical_width_um": physical_width,
            "effective_mpp": physical_width / spec.patch_size,
        }

    def _validate_features(self, snapshot: SourceSnapshot, spec: EncoderSpec) -> dict[str, Any]:
        coords_path = self._coordinate_path(snapshot.output_id, spec)
        feature_path = self._feature_path(snapshot.output_id, spec)
        if Path(f"{feature_path}.lock").exists():
            raise SlideEncodingError(f"Active feature lock: {feature_path}.lock")
        if not feature_path.is_file():
            raise SlideEncodingError(f"Missing feature file: {feature_path}")
        with (
            h5py.File(coords_path, "r") as coords_handle,
            h5py.File(feature_path, "r") as feature_handle,
        ):
            if "features" not in feature_handle or "coords" not in feature_handle:
                raise SlideEncodingError(f"Feature H5 lacks required datasets: {feature_path}")
            expected_coords = coords_handle["coords"]
            observed_coords = feature_handle["coords"]
            features = feature_handle["features"]
            if features.ndim != 2 or features.shape[1] != spec.feature_dim:
                raise SlideEncodingError(
                    f"Wrong {spec.name} feature shape {features.shape} in {feature_path}; "
                    f"expected (*, {spec.feature_dim})"
                )
            if len(features) == 0 or len(features) != len(expected_coords):
                raise SlideEncodingError(f"Feature/coordinate row mismatch in {feature_path}")
            if observed_coords.shape != expected_coords.shape:
                raise SlideEncodingError(
                    f"Stored feature coordinates have wrong shape: {feature_path}"
                )
            stored_encoder = features.attrs.get("encoder")
            if isinstance(stored_encoder, bytes):
                stored_encoder = stored_encoder.decode("utf-8")
            if str(stored_encoder) != spec.name:
                raise SlideEncodingError(
                    f"Encoder metadata mismatch in {feature_path}: {stored_encoder!r}"
                )
            for start in range(0, len(features), 8192):
                stop = min(start + 8192, len(features))
                if not np.array_equal(observed_coords[start:stop], expected_coords[start:stop]):
                    raise SlideEncodingError(f"Coordinate replay mismatch in {feature_path}")
                if not np.isfinite(features[start:stop]).all():
                    raise SlideEncodingError(f"Non-finite embeddings in {feature_path}")
            patch_count = len(features)
        return {"patch_count": patch_count, "feature_dim": spec.feature_dim}

    def _commit_stage(
        self,
        snapshot: SourceSnapshot,
        stage: str,
        stage_key: str,
        validation: Mapping[str, Any],
    ) -> None:
        snapshot.assert_unchanged()
        receipt = {
            "schema_version": 1,
            "stage": stage,
            "stage_key": stage_key,
            "source_key": snapshot.key,
            "source": asdict(snapshot),
            "validation": dict(validation),
            "completed_at": time.time(),
            "implementation_hash": self._implementation_hash,
            "checkpoint_hashes": self._weight_hashes,
        }
        if stage == "seg":
            receipt["segmentation_policy"] = self.segmentation_policy
        _atomic_json(self._receipt_path(snapshot.output_id, stage), receipt)

    def _write_completion_receipt(
        self,
        snapshot: SourceSnapshot,
        keys: Mapping[str, str],
        result: SlideEncodingResult,
    ) -> None:
        snapshot.assert_unchanged()
        _atomic_json(
            self._completion_path(snapshot.output_id),
            {
                "schema_version": 1,
                "status": "complete",
                "source": asdict(snapshot),
                "source_key": snapshot.key,
                "config_key": result.config_key,
                "stage_keys": dict(keys),
                "stage_metrics": result.stage_metrics,
                "elapsed_seconds": result.elapsed_seconds,
                "completed_at": time.time(),
                "implementation_hash": self._implementation_hash,
                "checkpoint_hashes": self._weight_hashes,
                "segmentation_policy": self.segmentation_policy,
            },
        )


__all__ = [
    "SlideEncoderConfig",
    "SlideEncodingError",
    "SlideEncoder",
    "EncoderSpec",
    "SlideEncodingResult",
    "SourceChangedError",
    "SourceSnapshot",
    "sha256_file",
]
