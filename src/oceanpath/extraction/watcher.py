"""Continuous discovery and encoding for the three live colon producers."""

from __future__ import annotations

import csv
import fcntl
import logging
import math
import os
import signal
import socket
import threading
import time
import traceback
import uuid
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from oceanpath.extraction.inventory import (
    InventoryPaths,
    InventoryResult,
    ReadySlide,
    discover_ready_slides,
    read_slide_mpp,
    rih_original_requires_repair,
    write_inventory_csvs,
)
from oceanpath.extraction.local_output import LocalOutputSlideEncoder
from oceanpath.extraction.prefetch import (
    GIB,
    PrefetchingSlideEncoder,
    SlidePrefetcher,
    plan_due_jobs,
)
from oceanpath.extraction.stream_encoder import (
    SlideEncoderConfig,
    SourceSnapshot,
)
from oceanpath.extraction.stream_state import (
    DiscoveredSlide,
    StateTransitionError,
    StreamState,
    exclusive_process_lock,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WatcherConfig:
    inventory: InventoryPaths
    encoder: SlideEncoderConfig
    mpp_sheet_path: Path
    inventory_status_path: Path
    queue_status_path: Path
    database_path: Path
    lock_path: Path
    poll_seconds: float = 60.0
    idle_seconds: float = 10.0
    heartbeat_seconds: float = 30.0
    max_attempts: int = 4
    newest_jobs_per_oldest: int = 3
    reconcile_every_polls: int = 30
    continuous_discovery: bool = True
    prefetch_depth: int = 3
    prefetch_max_gib: float = 64.0
    prefetch_copy_workers: int = 1
    local_job_output: bool = False

    def __post_init__(self) -> None:
        if self.prefetch_depth < 0:
            raise ValueError("prefetch_depth must be non-negative")
        if not math.isfinite(self.prefetch_max_gib) or self.prefetch_max_gib <= 0:
            raise ValueError("prefetch_max_gib must be finite and positive")
        if self.prefetch_depth and not self.encoder.stage_locally:
            raise ValueError("prefetch_depth must be 0 when local staging is disabled")
        if self.prefetch_copy_workers < 1:
            raise ValueError("prefetch_copy_workers must be positive")


@dataclass(frozen=True, slots=True)
class _QueueSlide:
    wsi: str
    path: Path
    output_id: str
    mpp: float
    cohort: str


class _MetadataCache:
    """Cache header-only MPP/tile audits by immutable file stat identity."""

    def __init__(self) -> None:
        self._mpp: dict[str, tuple[int, int, float | None]] = {}
        self._repair: dict[str, tuple[int, int, bool]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(path: Path) -> str:
        # Inventory paths are already constructed beneath the configured slide
        # root.  Resolving every path adds an avoidable mounted-drive metadata
        # round trip during each discovery pass.
        return str(path.absolute())

    def seed_from_status(self, status_path: Path, slide_root: Path) -> int:
        """Warm immutable metadata from the last atomically committed ledger."""

        seeded = 0
        try:
            with status_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    wsi = row.get("wsi", "")
                    size = row.get("size_bytes", "")
                    mtime_ns = row.get("mtime_ns", "")
                    if not wsi or not size or not mtime_ns:
                        continue
                    try:
                        identity = (int(size), int(mtime_ns))
                    except ValueError:
                        continue
                    path = slide_root / wsi
                    key = self._key(path)
                    status = row.get("status")
                    mpp_text = row.get("mpp", "")
                    if mpp_text:
                        with suppress(ValueError):
                            self._mpp[key] = (*identity, float(mpp_text))
                    elif row.get("source_status") == "invalid_mpp":
                        self._mpp[key] = (*identity, None)

                    if row.get("cohort") == "RIH":
                        source_status = row.get("source_status")
                        if source_status in {"original_standard", "fixed_verified"}:
                            self._repair[key] = (*identity, False)
                        elif source_status == "invalid_fixed_tiles":
                            self._repair[key] = (*identity, True)
                    if status == "ready":
                        seeded += 1
        except (FileNotFoundError, OSError, csv.Error):
            return 0
        return seeded

    @staticmethod
    def _identity(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    def mpp(self, path: Path) -> float | None:
        size, mtime_ns = self._identity(path)
        key = self._key(path)
        with self._lock:
            cached = self._mpp.get(key)
            if cached is not None and cached[:2] == (size, mtime_ns):
                return cached[2]
        value = read_slide_mpp(path)
        with self._lock:
            self._mpp[key] = (size, mtime_ns, value)
        return value

    def repair(self, path: Path) -> bool:
        size, mtime_ns = self._identity(path)
        key = self._key(path)
        with self._lock:
            cached = self._repair.get(key)
            if cached is not None and cached[:2] == (size, mtime_ns):
                return cached[2]
        value = rih_original_requires_repair(path)
        with self._lock:
            self._repair[key] = (size, mtime_ns, value)
        return value


class EncodingWatcher:
    """One-GPU worker plus an independent producer-discovery thread."""

    def __init__(self, cfg: WatcherConfig) -> None:
        self.cfg = cfg
        self.state = StreamState(cfg.database_path)
        # ``encoder`` always observes the durable feature root; discovery,
        # reconciliation, and cache-identity checks must never see the local
        # job root while a slide is mid-flight on the worker thread.
        self.encoder = PrefetchingSlideEncoder(cfg.encoder)
        self.process_encoder: PrefetchingSlideEncoder = (
            LocalOutputSlideEncoder(cfg.encoder) if cfg.local_job_output else self.encoder
        )
        if self.process_encoder.implementation_hash != self.encoder.implementation_hash:
            raise RuntimeError("Durable and local-output encoders disagree on cache identity")
        self.stop_event = threading.Event()
        self._cache = _MetadataCache()
        seeded = self._cache.seed_from_status(
            cfg.inventory_status_path,
            cfg.inventory.slide_root,
        )
        if seeded:
            logger.info("Warmed metadata cache from %d ready ledger rows", seeded)
        self._poll_count = 0
        self._schedule_count = 0
        self._discovery_thread: threading.Thread | None = None
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"

    @staticmethod
    def _queue_slide(job: Any) -> _QueueSlide:
        return _QueueSlide(
            wsi=job.source_relpath,
            path=Path(job.source_abspath),
            output_id=job.output_id,
            mpp=job.mpp,
            cohort=job.cohort,
        )

    def _prefetch_snapshots(self) -> list[SourceSnapshot]:
        jobs = self.state.list_jobs(statuses=["pending", "retry"])
        planned = plan_due_jobs(
            jobs,
            schedule_count=self._schedule_count,
            newest_jobs_per_oldest=self.cfg.newest_jobs_per_oldest,
            count=self.cfg.prefetch_depth + 1,
        )
        snapshots: list[SourceSnapshot] = []
        for job in planned:
            try:
                snapshot = SourceSnapshot.capture(self._queue_slide(job))
            except Exception as exc:
                logger.warning("Cannot prefetch %s: %s", job.output_id, exc)
                continue
            if (
                snapshot.key != job.source_key
                or self.encoder.config_key(snapshot) != job.config_key
            ):
                logger.warning("Skipping stale prefetch candidate %s", job.output_id)
                continue
            snapshots.append(snapshot)
        return snapshots

    def request_stop(self, signum: int | None = None, _frame: object = None) -> None:
        if signum is not None:
            logger.info("Received signal %s; stopping after the active slide", signum)
        self.stop_event.set()

    def refresh(self, *, reconcile: bool = False) -> InventoryResult:
        started = time.monotonic()
        result = discover_ready_slides(
            self.cfg.inventory,
            require_label_match=False,
            mpp_resolver=self._cache.mpp,
            rih_requires_repair=self._cache.repair,
        )
        write_inventory_csvs(
            result,
            mpp_path=self.cfg.mpp_sheet_path,
            status_path=self.cfg.inventory_status_path,
        )
        discovered: list[DiscoveredSlide] = []
        ready_by_id: dict[str, ReadySlide] = {}
        for slide in result.ready:
            snapshot = SourceSnapshot.capture(slide)
            ready_by_id[slide.output_id] = slide
            discovered.append(
                DiscoveredSlide(
                    output_id=slide.output_id,
                    source_abspath=str(snapshot.path),
                    source_relpath=slide.wsi,
                    mpp=slide.mpp,
                    source_size=snapshot.size,
                    source_mtime_ns=snapshot.mtime_ns,
                    source_key=snapshot.key,
                    config_key=self.encoder.config_key(snapshot),
                    cohort=slide.cohort,
                    readiness_evidence={
                        "source_status": slide.source_status,
                        "evidence": slide.evidence,
                        "label_match": slide.label_match,
                        "details": dict(slide.details),
                    },
                )
            )
        changes = self.state.upsert_many(discovered)
        inserted = sum(item.inserted for item in changes)
        reset = sum(item.reset for item in changes)

        if reconcile:
            for job in self.state.list_jobs():
                ready = ready_by_id.get(job.output_id)
                if ready is None:
                    continue
                valid = self.encoder.validate_complete(ready, deep=False)
                if job.status == "complete" and not valid:
                    self.state.requeue(job.output_id, "committed artifacts failed validation")
                    reset += 1
                elif job.status != "processing" and job.status != "complete" and valid:
                    snapshot = SourceSnapshot.capture(ready)
                    self.state.restore_complete(
                        job.output_id,
                        source_key=snapshot.key,
                        config_key=self.encoder.config_key(snapshot),
                        receipt_path=(
                            self.cfg.encoder.output_root
                            / "_stream_receipts"
                            / job.output_id
                            / "complete.json"
                        ),
                        result={"restored_from_validated_receipt": True},
                    )

        self._write_queue_status()
        logger.info(
            "Inventory refreshed in %.1fs: ready=%d (TCGA=%d SurGen=%d RIH=%d), "
            "inserted=%d reset=%d queue=%s",
            time.monotonic() - started,
            result.summary["ready"],
            result.summary["ready_tcga"],
            result.summary["ready_surgen"],
            result.summary["ready_rih"],
            inserted,
            reset,
            self.state.summary(),
        )
        return result

    def _write_queue_status(self) -> None:
        path = self.cfg.queue_status_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        fields = (
            "output_id",
            "wsi",
            "mpp",
            "cohort",
            "status",
            "attempts",
            "discovered_at",
            "started_at",
            "finished_at",
            "next_retry_at",
            "error",
            "receipt_path",
        )
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for job in self.state.list_jobs():
                    writer.writerow(
                        {
                            "output_id": job.output_id,
                            "wsi": job.source_relpath,
                            "mpp": format(job.mpp, ".17g"),
                            "cohort": job.cohort,
                            "status": job.status,
                            "attempts": job.attempts,
                            "discovered_at": job.discovered_at,
                            "started_at": job.started_at,
                            "finished_at": job.finished_at,
                            "next_retry_at": job.next_retry_at,
                            "error": job.error,
                            "receipt_path": job.receipt_path,
                        }
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _discovery_loop(self) -> None:
        while not self.stop_event.wait(self.cfg.poll_seconds):
            try:
                self._poll_count += 1
                reconcile = self._poll_count % self.cfg.reconcile_every_polls == 0
                self.refresh(reconcile=reconcile)
            except Exception:
                logger.exception("Producer discovery failed; the GPU worker will continue safely")

    def _shutdown_background(self) -> None:
        self.stop_event.set()
        if self._discovery_thread is not None:
            self._discovery_thread.join(timeout=min(self.cfg.poll_seconds, 10.0))

    def run(self, *, max_slides: int | None = None) -> dict[str, int]:
        with exclusive_process_lock(self.cfg.lock_path), ExitStack() as lifecycle:
            prefetcher: SlidePrefetcher | None = None
            orphan_files = self.encoder.recover_orphan_work_files()
            if self.process_encoder is not self.encoder:
                orphan_files += self.process_encoder.recover_orphan_work_files()
            if orphan_files:
                logger.warning("Archived %d orphaned TRIDENT lock/temp files", orphan_files)
            recovered = self.state.recover_orphans()
            if recovered:
                logger.warning("Recovered %d orphaned processing jobs", recovered)
            self.refresh(reconcile=True)
            if self.cfg.prefetch_depth:
                try:
                    prefetcher = lifecycle.enter_context(
                        SlidePrefetcher(
                            self.cfg.encoder.scratch_root,
                            depth=self.cfg.prefetch_depth,
                            max_bytes=int(self.cfg.prefetch_max_gib * GIB),
                            reserve_bytes=int(self.cfg.encoder.scratch_reserve_gib * GIB),
                            telemetry_path=self.cfg.encoder.state_root / "prefetch_telemetry.jsonl",
                            copy_workers=self.cfg.prefetch_copy_workers,
                        )
                    )
                    self.process_encoder.bind_prefetcher(prefetcher)
                    lifecycle.callback(self.process_encoder.bind_prefetcher, None)
                    logger.info(
                        "Enabled slide prefetch: depth=%d max=%.1fGiB reserve=%.1fGiB workers=%d",
                        self.cfg.prefetch_depth,
                        self.cfg.prefetch_max_gib,
                        self.cfg.encoder.scratch_reserve_gib,
                        self.cfg.prefetch_copy_workers,
                    )
                except Exception:
                    logger.exception("Prefetch startup failed; continuing with serial staging")
                    self.process_encoder.bind_prefetcher(None)
                    prefetcher = None
            if self.cfg.continuous_discovery:
                self._discovery_thread = threading.Thread(
                    target=self._discovery_loop,
                    name="producer-discovery",
                    daemon=True,
                )
                self._discovery_thread.start()
            else:
                logger.info("Frozen inventory mode: producer polling is disabled after startup")
            lifecycle.callback(self._shutdown_background)
            processed = 0
            while not self.stop_event.is_set():
                if max_slides is not None and processed >= max_slides:
                    break
                if prefetcher is not None:
                    try:
                        prefetcher.plan(self._prefetch_snapshots())
                    except Exception:
                        logger.exception("Prefetch planning failed; continuing with serial staging")
                        self.process_encoder.bind_prefetcher(None)
                        prefetcher.close()
                        prefetcher = None
                order: Literal["oldest", "newest"] = (
                    "oldest"
                    if self._schedule_count % (self.cfg.newest_jobs_per_oldest + 1)
                    == self.cfg.newest_jobs_per_oldest
                    else "newest"
                )
                self._schedule_count += 1
                job = self.state.claim_next(order=order, worker_id=self.worker_id)
                if job is None:
                    self.stop_event.wait(self.cfg.idle_seconds)
                    continue
                self._process_claim(job)
                processed += 1
                self._write_queue_status()
            self._shutdown_background()
            summary = self.state.summary()
            logger.info("Watcher stopped cleanly after %d claims: %s", processed, summary)
            return summary

    def _process_claim(self, job: Any) -> None:
        slide = self._queue_slide(job)
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(self.cfg.heartbeat_seconds):
                if not self.state.heartbeat(job.output_id, worker_id=self.worker_id):
                    logger.warning("Lost processing lease for %s", job.output_id)
                    return

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"heartbeat-{job.output_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        logger.info(
            "Encoding %s (%s, mpp=%.9g, attempt=%d, source=%s)",
            job.output_id,
            job.cohort,
            job.mpp,
            job.attempts,
            job.source_relpath,
        )
        try:
            current = SourceSnapshot.capture(slide)
            if current.key != job.source_key or self.encoder.config_key(current) != job.config_key:
                raise RuntimeError("source/config identity changed after queue claim")
            result = self.process_encoder.process(slide)
            if result.source_key != job.source_key or result.config_key != job.config_key:
                raise RuntimeError("encoder committed a different source/config identity")
            receipt = (
                self.cfg.encoder.output_root / "_stream_receipts" / job.output_id / "complete.json"
            )
            self.state.mark_complete(
                job.output_id,
                worker_id=self.worker_id,
                receipt_path=receipt,
                result={
                    "elapsed_seconds": result.elapsed_seconds,
                    "stage_metrics": result.stage_metrics,
                },
            )
            logger.info(
                "Completed %s in %.1fs: %s",
                job.output_id,
                result.elapsed_seconds,
                result.stage_metrics,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                self.stop_event.set()
            error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Encoding failed for %s: %s\n%s", job.output_id, error, traceback.format_exc()
            )
            try:
                if job.attempts >= self.cfg.max_attempts:
                    self.state.mark_failed(job.output_id, error, worker_id=self.worker_id)
                else:
                    delay = min(3600.0, 60.0 * 2 ** max(0, job.attempts - 1))
                    self.state.mark_retry(
                        job.output_id,
                        error,
                        delay_seconds=delay,
                        worker_id=self.worker_id,
                    )
            except StateTransitionError:
                logger.warning(
                    "Queue identity changed while %s was running; discovery owns its new state",
                    job.output_id,
                )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2.0)


def install_signal_handlers(watcher: EncodingWatcher) -> None:
    signal.signal(signal.SIGINT, watcher.request_stop)
    signal.signal(signal.SIGTERM, watcher.request_stop)


def _advisory_lock_held(path: Path) -> bool | None:
    """Probe an existing flock without creating or modifying its lock file.

    ``None`` means the lock state could not be inspected, in which case the
    caller may fall back to PID visibility.  Acquiring a free advisory lock
    briefly is safe: no bytes are written and the descriptor is always
    unlocked before it is closed.
    """

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        return False
    except OSError:
        return None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return None
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def watcher_status(cfg: WatcherConfig) -> dict[str, Any]:
    state = StreamState(cfg.database_path)
    owner = ""
    with suppress(OSError):
        owner = cfg.lock_path.read_text(encoding="utf-8").strip()
    pid: int | None = None
    if owner.startswith("pid="):
        try:
            pid = int(owner.split()[0].split("=", maxsplit=1)[1])
        except (IndexError, ValueError):
            pid = None
    pid_visible = False
    if pid is not None:
        try:
            os.kill(pid, 0)
            pid_visible = True
        except OSError:
            pass
    lock_held = _advisory_lock_held(cfg.lock_path)
    running = pid_visible if lock_held is None else lock_held
    return {
        "prefetch": {
            "depth": cfg.prefetch_depth,
            "max_gib": cfg.prefetch_max_gib,
            "scratch_reserve_gib": cfg.encoder.scratch_reserve_gib,
            "copy_workers": cfg.prefetch_copy_workers,
        },
        "local_job_output": cfg.local_job_output,
        "running": running,
        "pid": pid,
        "pid_visible": pid_visible,
        "lock_held": lock_held,
        "lock_owner": owner,
        "queue": state.summary(),
        "database": str(cfg.database_path),
        "features": str(cfg.encoder.output_root),
        "mpp_sheet": str(cfg.mpp_sheet_path),
    }


__all__ = [
    "EncodingWatcher",
    "WatcherConfig",
    "install_signal_handlers",
    "watcher_status",
]
