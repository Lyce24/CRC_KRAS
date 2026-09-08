"""Operational local-slide prefetch for the continuous slide encoder.

Prefetching is an operational detail — copying an immutable source to local
scratch never changes the encoded artifacts, so it lives outside
:mod:`stream_encoder`, whose pinned ``IMPLEMENTATION_HASH`` defines the cache
identity recorded in every stage receipt.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

from oceanpath.extraction.stream_encoder import SlideEncoder, SourceSnapshot

logger = logging.getLogger(__name__)

COPY_CHUNK_BYTES = 16 * 1024 * 1024
GIB = 1024**3
PREFETCH_DIRECTORY = "prefetch-v1"


class QueueJobLike(Protocol):
    """Queue fields needed to reproduce the watcher's claim order."""

    @property
    def output_id(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def queued_at(self) -> float: ...

    @property
    def next_retry_at(self) -> float | None: ...


QueueJobT = TypeVar("QueueJobT", bound=QueueJobLike)


def plan_due_jobs(
    jobs: Iterable[QueueJobT],
    *,
    schedule_count: int,
    newest_jobs_per_oldest: int,
    count: int,
    now: float | None = None,
) -> list[QueueJobT]:
    """Return a read-only lookahead matching successive ``claim_next`` calls.

    The result is only a hint.  The watcher still performs the authoritative
    atomic claim immediately before processing, so a concurrent discovery or a
    newly due retry can safely make this plan stale.
    """

    if schedule_count < 0:
        raise ValueError("schedule_count must be non-negative")
    if newest_jobs_per_oldest < 0:
        raise ValueError("newest_jobs_per_oldest must be non-negative")
    if count < 0:
        raise ValueError("count must be non-negative")
    timestamp = time.time() if now is None else float(now)
    remaining = [
        job
        for job in jobs
        if job.status == "pending"
        or (job.status == "retry" and (job.next_retry_at is None or job.next_retry_at <= timestamp))
    ]
    selected: list[QueueJobT] = []
    cadence = newest_jobs_per_oldest + 1
    for offset in range(min(count, len(remaining))):
        sequence = schedule_count + offset
        oldest = sequence % cadence == newest_jobs_per_oldest
        if oldest:
            candidate = min(remaining, key=lambda job: (job.queued_at, job.output_id))
        else:
            candidate = min(remaining, key=lambda job: (-job.queued_at, job.output_id))
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


class _CopyCancelled(RuntimeError):
    """Internal cooperative cancellation marker."""


@dataclass(slots=True)
class _StageEntry:
    snapshot: SourceSnapshot
    staged: Path
    temporary: Path
    cancel: threading.Event = field(default_factory=threading.Event)
    future: Future[Path] | None = None
    leased: bool = False
    wanted: bool = True
    status: str = "queued"
    scheduled_at: float = field(default_factory=time.monotonic)
    copy_started_at: float | None = None
    copy_finished_at: float | None = None
    copy_seconds: float | None = None
    error: str | None = None


class SlidePrefetcher:
    """Bounded, single-reader staging queue for immutable slide snapshots."""

    def __init__(
        self,
        scratch_root: str | Path,
        *,
        depth: int = 3,
        max_bytes: int = 64 * GIB,
        reserve_bytes: int = 30 * GIB,
        telemetry_path: str | Path | None = None,
        copy_workers: int = 1,
    ) -> None:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if reserve_bytes < 0:
            raise ValueError("reserve_bytes must be non-negative")
        if copy_workers < 1:
            raise ValueError("copy_workers must be positive")
        self.depth = int(depth)
        self.copy_workers = int(copy_workers)
        self.max_bytes = int(max_bytes)
        self.reserve_bytes = int(reserve_bytes)
        self.root = Path(scratch_root).expanduser().resolve(strict=False) / PREFETCH_DIRECTORY
        self.telemetry_path = (
            None
            if telemetry_path is None
            else Path(telemetry_path).expanduser().resolve(strict=False)
        )
        self._lock = threading.RLock()
        self._telemetry_lock = threading.Lock()
        self._entries: dict[str, _StageEntry] = {}
        self._stats: Counter[str] = Counter()
        self._wait_seconds = 0.0
        self._closed = False
        self._telemetry_failed = False
        self.root.mkdir(parents=True, exist_ok=True)
        self._recover_orphans()
        self._executor = (
            ThreadPoolExecutor(
                max_workers=self.copy_workers, thread_name_prefix="colon-slide-prefetch"
            )
            if self.depth > 0
            else None
        )

    @property
    def stats(self) -> dict[str, int | float]:
        with self._lock:
            result: dict[str, int | float] = dict(self._stats)
            result["wait_seconds"] = self._wait_seconds
            result["entries"] = len(self._entries)
            result["reserved_bytes"] = sum(entry.snapshot.size for entry in self._entries.values())
            return result

    def _emit(self, event: str, **payload: Any) -> None:
        if self.telemetry_path is None or self._telemetry_failed:
            return
        record = {"time": time.time(), "event": event, **payload}
        try:
            with self._telemetry_lock:
                self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                with self.telemetry_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
        except OSError as exc:
            self._telemetry_failed = True
            logger.warning("Disabling prefetch telemetry after write failure: %s", exc)

    def _recover_orphans(self) -> None:
        """Remove only files created beneath the dedicated prefetch subtree."""

        removed = 0
        for directory in self.root.iterdir():
            if directory.is_symlink():
                directory.unlink(missing_ok=True)
                removed += 1
                continue
            if not directory.is_dir() or len(directory.name) != 64:
                logger.warning("Leaving unknown prefetch scratch entry untouched: %s", directory)
                continue
            try:
                int(directory.name, 16)
            except ValueError:
                logger.warning(
                    "Leaving unknown prefetch scratch directory untouched: %s", directory
                )
                continue
            unknown_directory = False
            for path in directory.iterdir():
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)
                    removed += 1
                else:
                    unknown_directory = True
                    logger.warning("Leaving nested prefetch scratch entry untouched: %s", path)
            if not unknown_directory:
                with suppress(OSError):
                    directory.rmdir()
        if removed:
            logger.warning("Removed %d orphaned prefetched slide files", removed)

    def _cleanup_paths(self, entry: _StageEntry) -> None:
        entry.temporary.unlink(missing_ok=True)
        entry.staged.unlink(missing_ok=True)
        with suppress(OSError):
            entry.staged.parent.rmdir()

    def _discard(self, entry: _StageEntry) -> None:
        with self._lock:
            current = self._entries.get(entry.snapshot.key)
            if current is entry and not entry.leased:
                self._entries.pop(entry.snapshot.key, None)
        if not entry.leased:
            self._cleanup_paths(entry)

    def _copy_entry(self, entry: _StageEntry) -> Path:
        entry.copy_started_at = time.monotonic()
        entry.status = "copying"
        try:
            entry.snapshot.assert_unchanged()
            with entry.snapshot.path.open("rb") as source, entry.temporary.open("xb") as target:
                while True:
                    if entry.cancel.is_set():
                        raise _CopyCancelled("prefetch plan changed")
                    chunk = source.read(COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    target.write(chunk)
            if entry.temporary.stat().st_size != entry.snapshot.size:
                raise RuntimeError(
                    f"Prefetched size mismatch for {entry.snapshot.output_id}: "
                    f"{entry.temporary.stat().st_size} != {entry.snapshot.size}"
                )
            entry.snapshot.assert_unchanged()
            if entry.cancel.is_set():
                raise _CopyCancelled("prefetch plan changed")
            os.replace(entry.temporary, entry.staged)
            entry.status = "ready"
            entry.copy_finished_at = time.monotonic()
            entry.copy_seconds = entry.copy_finished_at - entry.copy_started_at
            with self._lock:
                self._stats["copied"] += 1
                self._stats["bytes_copied"] += entry.snapshot.size
            self._emit(
                "copy_complete",
                output_id=entry.snapshot.output_id,
                source_key=entry.snapshot.key,
                bytes=entry.snapshot.size,
                seconds=entry.copy_seconds,
            )
            return entry.staged
        except _CopyCancelled as exc:
            entry.status = "cancelled"
            entry.error = str(exc)
            with self._lock:
                self._stats["cancelled"] += 1
            raise
        except Exception as exc:
            entry.status = "failed"
            entry.error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._stats["copy_failed"] += 1
            self._emit(
                "copy_failed",
                output_id=entry.snapshot.output_id,
                source_key=entry.snapshot.key,
                error=entry.error,
            )
            logger.warning("Prefetch failed for %s: %s", entry.snapshot.output_id, entry.error)
            raise
        finally:
            entry.temporary.unlink(missing_ok=True)

    def _reap_unwanted(self) -> None:
        with self._lock:
            entries = [
                entry
                for entry in self._entries.values()
                if not entry.wanted
                and not entry.leased
                and (entry.future is None or entry.future.done())
            ]
        for entry in entries:
            self._discard(entry)

    def plan(self, snapshots: Sequence[SourceSnapshot]) -> None:
        """Reconcile the bounded copy queue with current plus ``depth`` lookahead."""

        if self.depth == 0:
            return
        unique: list[SourceSnapshot] = []
        seen: set[str] = set()
        for snapshot in snapshots[: self.depth + 1]:
            if snapshot.key not in seen:
                unique.append(snapshot)
                seen.add(snapshot.key)

        with self._lock:
            if self._closed:
                return
            for entry in self._entries.values():
                entry.wanted = entry.snapshot.key in seen
                if not entry.wanted and not entry.leased:
                    entry.cancel.set()
                    if entry.future is not None:
                        entry.future.cancel()
        self._reap_unwanted()

        for snapshot in unique:
            with self._lock:
                if self._closed:
                    return
                existing = self._entries.get(snapshot.key)
                if existing is not None:
                    existing.wanted = True
                    continue
                live_entries = list(self._entries.values())
                if len(live_entries) >= self.depth + 1:
                    break
                reserved = sum(entry.snapshot.size for entry in live_entries)
                if reserved + snapshot.size > self.max_bytes:
                    self._stats["max_bytes_denied"] += 1
                    self._emit(
                        "admission_denied",
                        reason="max_bytes",
                        output_id=snapshot.output_id,
                        bytes=snapshot.size,
                        reserved_bytes=reserved,
                    )
                    break
                pending = sum(
                    entry.snapshot.size
                    for entry in live_entries
                    if entry.future is None or not entry.future.done()
                )
            try:
                free = shutil.disk_usage(self.root).free
            except OSError as exc:
                logger.warning("Cannot inspect prefetch scratch capacity: %s", exc)
                with self._lock:
                    self._stats["disk_probe_failed"] += 1
                break
            if free - pending - snapshot.size < self.reserve_bytes:
                with self._lock:
                    self._stats["reserve_denied"] += 1
                self._emit(
                    "admission_denied",
                    reason="scratch_reserve",
                    output_id=snapshot.output_id,
                    bytes=snapshot.size,
                    free_bytes=free,
                    pending_bytes=pending,
                    reserve_bytes=self.reserve_bytes,
                )
                break

            directory = self.root / snapshot.key
            directory.mkdir(parents=True, exist_ok=True)
            staged = directory / snapshot.path.name
            temporary = directory / f".{snapshot.path.name}.{uuid.uuid4().hex}.part"
            entry = _StageEntry(snapshot=snapshot, staged=staged, temporary=temporary)
            with self._lock:
                if self._closed or snapshot.key in self._entries:
                    self._cleanup_paths(entry)
                    continue
                self._entries[snapshot.key] = entry
                self._stats["scheduled"] += 1
                assert self._executor is not None
                entry.future = self._executor.submit(self._copy_entry, entry)
            self._emit(
                "scheduled",
                output_id=snapshot.output_id,
                source_key=snapshot.key,
                bytes=snapshot.size,
            )

    def _cancel_unleased(self) -> None:
        with self._lock:
            entries = [entry for entry in self._entries.values() if not entry.leased]
            for entry in entries:
                entry.wanted = False
                entry.cancel.set()
                if entry.future is not None:
                    entry.future.cancel()
        self._reap_unwanted()

    def acquire(self, snapshot: SourceSnapshot) -> Path | None:
        """Return and pin an exact prefetched copy, or ``None`` for safe fallback."""

        with self._lock:
            entry = self._entries.get(snapshot.key)
            if entry is None or entry.cancel.is_set() or entry.future is None:
                self._cancel_unleased()
                self._stats["misses"] += 1
                return None
            future = entry.future
        wait_started = time.monotonic()
        try:
            staged = future.result()
            snapshot.assert_unchanged()
            if not staged.is_file() or staged.stat().st_size != snapshot.size:
                raise RuntimeError(f"Missing or invalid prefetched source: {staged}")
        except Exception as exc:
            wait_seconds = time.monotonic() - wait_started
            with self._lock:
                self._stats["misses"] += 1
                self._wait_seconds += wait_seconds
            self._emit(
                "miss",
                output_id=snapshot.output_id,
                source_key=snapshot.key,
                wait_seconds=wait_seconds,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._cancel_unleased()
            self._discard(entry)
            return None
        wait_seconds = time.monotonic() - wait_started
        with self._lock:
            current = self._entries.get(snapshot.key)
            if current is not entry or entry.cancel.is_set():
                self._cancel_unleased()
                self._stats["misses"] += 1
                return None
            entry.leased = True
            self._stats["hits"] += 1
            self._wait_seconds += wait_seconds
        self._emit(
            "hit",
            output_id=snapshot.output_id,
            source_key=snapshot.key,
            wait_seconds=wait_seconds,
            copy_seconds=entry.copy_seconds,
        )
        logger.info(
            "Using prefetched source %s (copy=%.1fs, wait=%.1fs)",
            snapshot.output_id,
            entry.copy_seconds or 0.0,
            wait_seconds,
        )
        return staged

    def release(self, snapshot: SourceSnapshot) -> None:
        """Release and delete a prefetched file after all slide workers stop using it."""

        with self._lock:
            entry = self._entries.get(snapshot.key)
            if entry is None:
                return
            entry.leased = False
            self._entries.pop(snapshot.key, None)
            self._stats["released"] += 1
        self._cleanup_paths(entry)
        self._emit("released", output_id=snapshot.output_id, source_key=snapshot.key)

    def close(self) -> None:
        """Cooperatively stop copies and remove every unleased staged artifact."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = list(self._entries.values())
            for entry in entries:
                if not entry.leased:
                    entry.cancel.set()
                    if entry.future is not None:
                        entry.future.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        for entry in entries:
            if not entry.leased:
                self._discard(entry)
        logger.info("Slide prefetcher stopped: %s", self.stats)

    def __enter__(self) -> SlidePrefetcher:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


class PrefetchingSlideEncoder(SlideEncoder):
    """Slide encoder whose staging source may be supplied by a prefetcher."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._slide_prefetcher: SlidePrefetcher | None = None

    def bind_prefetcher(self, prefetcher: SlidePrefetcher | None) -> None:
        self._slide_prefetcher = prefetcher

    @contextmanager
    def _staged_source(self, snapshot: SourceSnapshot) -> Iterator[Path]:
        prefetcher = self._slide_prefetcher
        if prefetcher is not None:
            staged = prefetcher.acquire(snapshot)
            if staged is not None:
                try:
                    yield staged
                finally:
                    prefetcher.release(snapshot)
                return
        with super()._staged_source(snapshot) as staged:
            yield staged


__all__ = [
    "GIB",
    "PrefetchingSlideEncoder",
    "SlidePrefetcher",
    "plan_due_jobs",
]
