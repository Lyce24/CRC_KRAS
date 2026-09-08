"""Durable per-slide state for continuously growing extraction queues.

This module deliberately has no imaging, model, or third-party dependencies.
Discovery can run in one thread while a worker claims slides in another: every
operation opens its own SQLite connection, and claims use ``BEGIN IMMEDIATE``
to make selection plus transition to ``processing`` atomic.

Times are Unix timestamps in UTC.  Callers may supply explicit times to make
tests and operational recovery deterministic.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import socket
import sqlite3
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Literal

JobStatus = Literal[
    "pending",
    "processing",
    "complete",
    "retry",
    "failed",
    "blocked",
]
ClaimOrder = Literal["oldest", "newest"]

JOB_STATUSES: tuple[JobStatus, ...] = (
    "pending",
    "processing",
    "complete",
    "retry",
    "failed",
    "blocked",
)


class StreamStateError(RuntimeError):
    """Base exception for durable stream-state operations."""


class StateTransitionError(StreamStateError):
    """Raised when a stale or invalid worker tries to transition a job."""


class LockUnavailableError(StreamStateError):
    """Raised when another process already owns the global watcher lock."""


@dataclass(frozen=True, slots=True)
class DiscoveredSlide:
    """Immutable identity and readiness evidence for one ready source slide."""

    output_id: str
    source_abspath: str
    source_relpath: str
    mpp: float
    source_size: int
    source_mtime_ns: int
    source_key: str
    config_key: str
    cohort: str
    readiness_evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nonempty = {
            "output_id": self.output_id,
            "source_abspath": self.source_abspath,
            "source_relpath": self.source_relpath,
            "source_key": self.source_key,
            "config_key": self.config_key,
            "cohort": self.cohort,
        }
        for name, value in nonempty.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        if not Path(self.source_abspath).is_absolute():
            raise ValueError("source_abspath must be absolute")
        relative = PurePath(self.source_relpath)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source_relpath must be a contained relative path")
        if isinstance(self.mpp, bool) or not isinstance(self.mpp, (int, float)):
            raise ValueError("mpp must be a finite positive number")
        if not math.isfinite(float(self.mpp)) or float(self.mpp) <= 0:
            raise ValueError("mpp must be a finite positive number")
        if isinstance(self.source_size, bool) or not isinstance(self.source_size, int):
            raise ValueError("source_size must be a non-negative integer")
        if self.source_size < 0:
            raise ValueError("source_size must be a non-negative integer")
        if isinstance(self.source_mtime_ns, bool) or not isinstance(self.source_mtime_ns, int):
            raise ValueError("source_mtime_ns must be a non-negative integer")
        if self.source_mtime_ns < 0:
            raise ValueError("source_mtime_ns must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class JobRecord:
    """A snapshot of one row in the durable queue."""

    output_id: str
    source_abspath: str
    source_relpath: str
    mpp: float
    source_size: int
    source_mtime_ns: int
    source_key: str
    config_key: str
    cohort: str
    readiness_evidence: dict[str, object]
    status: JobStatus
    attempts: int
    discovered_at: float
    queued_at: float
    updated_at: float
    started_at: float | None
    heartbeat_at: float | None
    finished_at: float | None
    next_retry_at: float | None
    error: str | None
    worker_id: str | None
    receipt_path: str | None
    result: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class UpsertResult:
    """Outcome of incorporating one discovery into the queue."""

    record: JobRecord
    inserted: bool
    reset: bool


_SCHEMA = """
CREATE TABLE IF NOT EXISTS stream_jobs (
    output_id TEXT PRIMARY KEY,
    source_abspath TEXT NOT NULL,
    source_relpath TEXT NOT NULL,
    mpp REAL NOT NULL CHECK (mpp > 0),
    source_size INTEGER NOT NULL CHECK (source_size >= 0),
    source_mtime_ns INTEGER NOT NULL CHECK (source_mtime_ns >= 0),
    source_key TEXT NOT NULL,
    config_key TEXT NOT NULL,
    cohort TEXT NOT NULL,
    readiness_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'processing', 'complete', 'retry', 'failed', 'blocked')
    ),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    discovered_at REAL NOT NULL,
    queued_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    heartbeat_at REAL,
    finished_at REAL,
    next_retry_at REAL,
    error TEXT,
    worker_id TEXT,
    receipt_path TEXT,
    result_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_stream_jobs_claim
    ON stream_jobs(status, next_retry_at, queued_at);
CREATE INDEX IF NOT EXISTS idx_stream_jobs_cohort_status
    ON stream_jobs(cohort, status);
"""

_SELECT_COLUMNS = """
output_id, source_abspath, source_relpath, mpp, source_size, source_mtime_ns,
source_key, config_key, cohort, readiness_json, status, attempts,
discovered_at, queued_at, updated_at, started_at, heartbeat_at, finished_at,
next_retry_at, error, worker_id, receipt_path, result_json
"""


class StreamState:
    """SQLite-backed queue with one connection per public operation."""

    def __init__(self, path: str | Path, *, busy_timeout_seconds: float = 30.0) -> None:
        self.path = Path(path)
        if busy_timeout_seconds <= 0 or not math.isfinite(busy_timeout_seconds):
            raise ValueError("busy_timeout_seconds must be finite and positive")
        self._busy_timeout_ms = round(busy_timeout_seconds * 1000)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(_SCHEMA)
            connection.execute("PRAGMA user_version = 1")
        finally:
            connection.close()

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def upsert(self, slide: DiscoveredSlide, *, now: float | None = None) -> UpsertResult:
        """Insert a discovery, preserving terminal state when identity is unchanged."""

        return self.upsert_many([slide], now=now)[0]

    def upsert_many(
        self,
        slides: Iterable[DiscoveredSlide],
        *,
        now: float | None = None,
    ) -> list[UpsertResult]:
        """Atomically incorporate a discovery snapshot.

        A change to source identity or ``config_key`` invalidates prior work and
        resets the job to a fresh ``pending`` state.  Readiness evidence and
        cohort metadata may evolve without invalidating completed artifacts.
        """

        discoveries = list(slides)
        duplicate_ids = _duplicates(slide.output_id for slide in discoveries)
        if duplicate_ids:
            joined = ", ".join(sorted(duplicate_ids))
            raise ValueError(f"duplicate output_id values in discovery batch: {joined}")
        timestamp = _timestamp(now)
        encoded = [(slide, _json_dumps(dict(slide.readiness_evidence))) for slide in discoveries]
        results: list[UpsertResult] = []

        with self._write_transaction() as connection:
            for slide, readiness_json in encoded:
                row = connection.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM stream_jobs WHERE output_id = ?",
                    (slide.output_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO stream_jobs (
                            output_id, source_abspath, source_relpath, mpp,
                            source_size, source_mtime_ns, source_key, config_key,
                            cohort, readiness_json, status, attempts,
                            discovered_at, queued_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
                        """,
                        (
                            slide.output_id,
                            slide.source_abspath,
                            slide.source_relpath,
                            float(slide.mpp),
                            slide.source_size,
                            slide.source_mtime_ns,
                            slide.source_key,
                            slide.config_key,
                            slide.cohort,
                            readiness_json,
                            timestamp,
                            timestamp,
                            timestamp,
                        ),
                    )
                    inserted = True
                    reset = False
                else:
                    reset = _identity_changed(row, slide)
                    inserted = False
                    if reset:
                        connection.execute(
                            """
                            UPDATE stream_jobs SET
                                source_abspath = ?, source_relpath = ?, mpp = ?,
                                source_size = ?, source_mtime_ns = ?, source_key = ?,
                                config_key = ?, cohort = ?, readiness_json = ?,
                                status = 'pending', attempts = 0, queued_at = ?,
                                updated_at = ?, started_at = NULL,
                                heartbeat_at = NULL, finished_at = NULL,
                                next_retry_at = NULL, error = NULL, worker_id = NULL,
                                receipt_path = NULL, result_json = NULL
                            WHERE output_id = ?
                            """,
                            (
                                slide.source_abspath,
                                slide.source_relpath,
                                float(slide.mpp),
                                slide.source_size,
                                slide.source_mtime_ns,
                                slide.source_key,
                                slide.config_key,
                                slide.cohort,
                                readiness_json,
                                timestamp,
                                timestamp,
                                slide.output_id,
                            ),
                        )
                    elif row["cohort"] != slide.cohort or row["readiness_json"] != readiness_json:
                        connection.execute(
                            """
                            UPDATE stream_jobs
                            SET cohort = ?, readiness_json = ?, updated_at = ?
                            WHERE output_id = ?
                            """,
                            (slide.cohort, readiness_json, timestamp, slide.output_id),
                        )

                updated = connection.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM stream_jobs WHERE output_id = ?",
                    (slide.output_id,),
                ).fetchone()
                assert updated is not None
                results.append(UpsertResult(_row_to_record(updated), inserted, reset))

        return results

    def get(self, output_id: str) -> JobRecord | None:
        """Return a current job snapshot, or ``None`` when it is unknown."""

        connection = self._connect()
        try:
            row = connection.execute(
                f"SELECT {_SELECT_COLUMNS} FROM stream_jobs WHERE output_id = ?",
                (output_id,),
            ).fetchone()
            return None if row is None else _row_to_record(row)
        finally:
            connection.close()

    def list_jobs(
        self,
        *,
        statuses: Iterable[JobStatus] | None = None,
        newest_first: bool = False,
    ) -> list[JobRecord]:
        """List jobs deterministically by queue time and output ID."""

        selected = tuple(statuses) if statuses is not None else ()
        _validate_statuses(selected)
        direction = "DESC" if newest_first else "ASC"
        where = ""
        parameters: tuple[object, ...] = ()
        if selected:
            placeholders = ", ".join("?" for _ in selected)
            where = f"WHERE status IN ({placeholders})"
            parameters = selected

        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS} FROM stream_jobs
                {where}
                ORDER BY queued_at {direction}, output_id ASC
                """,
                parameters,
            ).fetchall()
            return [_row_to_record(row) for row in rows]
        finally:
            connection.close()

    def claim_next(
        self,
        *,
        order: ClaimOrder = "oldest",
        worker_id: str | None = None,
        now: float | None = None,
    ) -> JobRecord | None:
        """Atomically claim one due job, choosing the newest or oldest queued slide."""

        if order not in ("oldest", "newest"):
            raise ValueError("order must be 'oldest' or 'newest'")
        timestamp = _timestamp(now)
        owner = worker_id or _default_worker_id()
        direction = "ASC" if order == "oldest" else "DESC"

        with self._write_transaction() as connection:
            selected = connection.execute(
                f"""
                SELECT output_id FROM stream_jobs
                WHERE status = 'pending'
                   OR (status = 'retry' AND (next_retry_at IS NULL OR next_retry_at <= ?))
                ORDER BY queued_at {direction}, output_id ASC
                LIMIT 1
                """,
                (timestamp,),
            ).fetchone()
            if selected is None:
                return None

            output_id = str(selected["output_id"])
            connection.execute(
                """
                UPDATE stream_jobs SET
                    status = 'processing', attempts = attempts + 1,
                    started_at = ?, heartbeat_at = ?, updated_at = ?,
                    finished_at = NULL, next_retry_at = NULL,
                    error = NULL, worker_id = ?
                WHERE output_id = ?
                """,
                (timestamp, timestamp, timestamp, owner, output_id),
            )
            row = connection.execute(
                f"SELECT {_SELECT_COLUMNS} FROM stream_jobs WHERE output_id = ?",
                (output_id,),
            ).fetchone()
            assert row is not None
            return _row_to_record(row)

    def heartbeat(
        self,
        output_id: str,
        *,
        worker_id: str | None = None,
        now: float | None = None,
    ) -> bool:
        """Refresh a processing lease; return false for a stale or different worker."""

        timestamp = _timestamp(now)
        worker_clause, parameters = _worker_condition(worker_id)
        with self._write_transaction() as connection:
            cursor = connection.execute(
                f"""
                UPDATE stream_jobs SET heartbeat_at = ?, updated_at = ?
                WHERE output_id = ? AND status = 'processing' {worker_clause}
                """,
                (timestamp, timestamp, output_id, *parameters),
            )
            return cursor.rowcount == 1

    def requeue(
        self,
        output_id: str,
        reason: str,
        *,
        now: float | None = None,
    ) -> JobRecord:
        """Requeue non-processing work whose committed artifacts failed validation."""

        timestamp = _timestamp(now)
        with self._write_transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE stream_jobs SET
                    status = 'pending', queued_at = ?, updated_at = ?,
                    started_at = NULL, heartbeat_at = NULL, finished_at = NULL,
                    next_retry_at = NULL, error = ?, worker_id = NULL,
                    receipt_path = NULL, result_json = NULL
                WHERE output_id = ? AND status != 'processing'
                """,
                (timestamp, timestamp, _required_error(reason), output_id),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(
                    f"Cannot requeue {output_id!r}: it is missing or actively processing"
                )
            row = connection.execute(
                f"SELECT {_SELECT_COLUMNS} FROM stream_jobs WHERE output_id = ?",
                (output_id,),
            ).fetchone()
            assert row is not None
            return _row_to_record(row)

    def mark_complete(
        self,
        output_id: str,
        *,
        worker_id: str | None = None,
        receipt_path: str | Path | None = None,
        result: Mapping[str, object] | None = None,
        now: float | None = None,
    ) -> JobRecord:
        """Finalize a processing job after its artifacts have been validated."""

        return self._finish_processing(
            output_id,
            status="complete",
            worker_id=worker_id,
            error=None,
            next_retry_at=None,
            receipt_path=None if receipt_path is None else str(receipt_path),
            result=None if result is None else dict(result),
            now=now,
        )

    def restore_complete(
        self,
        output_id: str,
        *,
        source_key: str,
        config_key: str,
        receipt_path: str | Path,
        result: Mapping[str, object] | None = None,
        now: float | None = None,
    ) -> JobRecord:
        """Restore an exact-identity completion after a queue/database reset.

        Callers must validate the committed receipt and artifacts first.  The
        identity predicate prevents a stale receipt from completing a changed
        source or configuration, and active processing work is never stolen.
        """

        timestamp = _timestamp(now)
        result_json = None if result is None else _json_dumps(dict(result))
        with self._write_transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE stream_jobs SET
                    status = 'complete', updated_at = ?, finished_at = ?,
                    started_at = NULL, heartbeat_at = NULL,
                    next_retry_at = NULL, error = NULL, worker_id = NULL,
                    receipt_path = ?, result_json = ?
                WHERE output_id = ? AND status != 'processing'
                  AND source_key = ? AND config_key = ?
                """,
                (
                    timestamp,
                    timestamp,
                    str(receipt_path),
                    result_json,
                    output_id,
                    source_key,
                    config_key,
                ),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(
                    f"Cannot restore {output_id!r} complete: identity changed, "
                    "work is active, or the job is missing"
                )
            row = connection.execute(
                f"SELECT {_SELECT_COLUMNS} FROM stream_jobs WHERE output_id = ?",
                (output_id,),
            ).fetchone()
            assert row is not None
            return _row_to_record(row)

    def mark_retry(
        self,
        output_id: str,
        error: str,
        *,
        delay_seconds: float = 0.0,
        worker_id: str | None = None,
        now: float | None = None,
    ) -> JobRecord:
        """Return a processing job to the queue after a non-terminal failure."""

        if delay_seconds < 0 or not math.isfinite(delay_seconds):
            raise ValueError("delay_seconds must be finite and non-negative")
        timestamp = _timestamp(now)
        return self._finish_processing(
            output_id,
            status="retry",
            worker_id=worker_id,
            error=_required_error(error),
            next_retry_at=timestamp + delay_seconds,
            receipt_path=None,
            result=None,
            now=timestamp,
        )

    def mark_failed(
        self,
        output_id: str,
        error: str,
        *,
        worker_id: str | None = None,
        now: float | None = None,
    ) -> JobRecord:
        """Move a processing job to terminal failure."""

        return self._finish_processing(
            output_id,
            status="failed",
            worker_id=worker_id,
            error=_required_error(error),
            next_retry_at=None,
            receipt_path=None,
            result=None,
            now=now,
        )

    def mark_blocked(
        self,
        output_id: str,
        error: str,
        *,
        worker_id: str | None = None,
        now: float | None = None,
    ) -> JobRecord:
        """Move a processing job to a terminal externally-blocked state."""

        return self._finish_processing(
            output_id,
            status="blocked",
            worker_id=worker_id,
            error=_required_error(error),
            next_retry_at=None,
            receipt_path=None,
            result=None,
            now=now,
        )

    def _finish_processing(
        self,
        output_id: str,
        *,
        status: Literal["complete", "retry", "failed", "blocked"],
        worker_id: str | None,
        error: str | None,
        next_retry_at: float | None,
        receipt_path: str | None,
        result: dict[str, object] | None,
        now: float | None,
    ) -> JobRecord:
        timestamp = _timestamp(now)
        result_json = None if result is None else _json_dumps(result)
        worker_clause, parameters = _worker_condition(worker_id)

        with self._write_transaction() as connection:
            cursor = connection.execute(
                f"""
                UPDATE stream_jobs SET
                    status = ?, updated_at = ?, finished_at = ?,
                    next_retry_at = ?, error = ?, worker_id = NULL,
                    receipt_path = ?, result_json = ?
                WHERE output_id = ? AND status = 'processing' {worker_clause}
                """,
                (
                    status,
                    timestamp,
                    timestamp,
                    next_retry_at,
                    error,
                    receipt_path,
                    result_json,
                    output_id,
                    *parameters,
                ),
            )
            if cursor.rowcount != 1:
                raise StateTransitionError(
                    f"Cannot mark {output_id!r} {status}: it is not owned processing work"
                )
            row = connection.execute(
                f"SELECT {_SELECT_COLUMNS} FROM stream_jobs WHERE output_id = ?",
                (output_id,),
            ).fetchone()
            assert row is not None
            return _row_to_record(row)

    def recover_orphans(
        self,
        *,
        stale_after_seconds: float | None = None,
        now: float | None = None,
        reason: str = "recovered orphaned processing job",
    ) -> int:
        """Make abandoned processing rows immediately claimable again.

        Passing ``None`` recovers every processing row, which is appropriate at
        startup while holding :func:`exclusive_process_lock`.  Otherwise only
        rows whose last heartbeat is at least ``stale_after_seconds`` old are
        recovered.
        """

        if stale_after_seconds is not None and (
            stale_after_seconds < 0 or not math.isfinite(stale_after_seconds)
        ):
            raise ValueError("stale_after_seconds must be finite and non-negative")
        timestamp = _timestamp(now)
        condition = "status = 'processing'"
        parameters: list[object] = []
        if stale_after_seconds is not None:
            condition += " AND (heartbeat_at IS NULL OR heartbeat_at <= ?)"
            parameters.append(timestamp - stale_after_seconds)

        with self._write_transaction() as connection:
            cursor = connection.execute(
                f"""
                UPDATE stream_jobs SET
                    status = 'retry', updated_at = ?, finished_at = ?,
                    next_retry_at = ?, error = ?, worker_id = NULL
                WHERE {condition}
                """,
                (timestamp, timestamp, timestamp, reason, *parameters),
            )
            return cursor.rowcount

    def summary(self, *, now: float | None = None) -> dict[str, int]:
        """Return stable status counts plus total and currently claimable jobs."""

        timestamp = _timestamp(now)
        counts: dict[str, int] = {status: 0 for status in JOB_STATUSES}
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM stream_jobs GROUP BY status"
            ).fetchall()
            for row in rows:
                counts[str(row["status"])] = int(row["count"])
            claimable = connection.execute(
                """
                SELECT COUNT(*) FROM stream_jobs
                WHERE status = 'pending'
                   OR (status = 'retry' AND (next_retry_at IS NULL OR next_retry_at <= ?))
                """,
                (timestamp,),
            ).fetchone()[0]
        finally:
            connection.close()
        counts["claimable"] = int(claimable)
        counts["total"] = sum(counts[status] for status in JOB_STATUSES)
        return counts


def atomic_write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Durably write JSON via a same-directory UUID temporary and ``replace``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        dict(payload),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return destination


def write_stage_receipt(
    path: str | Path,
    *,
    output_id: str,
    stage: str,
    source_key: str,
    config_key: str,
    payload: Mapping[str, object] | None = None,
    now: float | None = None,
) -> dict[str, object]:
    """Write a versioned completion receipt and return its in-memory content."""

    identity = {
        "output_id": output_id,
        "stage": stage,
        "source_key": source_key,
        "config_key": config_key,
    }
    for name, value in identity.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    receipt: dict[str, object] = {
        "schema_version": 1,
        **identity,
        "completed_at": _timestamp(now),
        "payload": {} if payload is None else dict(payload),
    }
    atomic_write_json(path, receipt)
    return receipt


def receipt_matches(
    path: str | Path,
    *,
    output_id: str,
    stage: str,
    source_key: str,
    config_key: str,
    payload: Mapping[str, object] | None = None,
) -> bool:
    """Return whether a readable receipt contains the expected identity/payload."""

    try:
        with Path(path).open(encoding="utf-8") as handle:
            actual = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    expected: dict[str, object] = {
        "schema_version": 1,
        "output_id": output_id,
        "stage": stage,
        "source_key": source_key,
        "config_key": config_key,
    }
    if payload is not None:
        expected["payload"] = dict(payload)
    return isinstance(actual, dict) and _mapping_contains(actual, expected)


@contextmanager
def exclusive_process_lock(
    path: str | Path,
    *,
    blocking: bool = False,
) -> Iterator[Path]:
    """Hold a process-wide advisory lock without deleting its stable lock file."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    operation = fcntl.LOCK_EX
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as error:
            raise LockUnavailableError(f"Another process holds {lock_path}") from error
        owner = f"pid={os.getpid()} host={socket.gethostname()} acquired={time.time():.6f}\n"
        os.ftruncate(descriptor, 0)
        os.write(descriptor, owner.encode("utf-8"))
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _timestamp(value: float | None) -> float:
    timestamp = time.time() if value is None else value
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise ValueError("timestamp must be a finite number")
    converted = float(timestamp)
    if not math.isfinite(converted):
        raise ValueError("timestamp must be a finite number")
    return converted


def _json_dumps(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("receipt/state metadata must be JSON-serializable without NaN") from error


def _identity_changed(row: sqlite3.Row, slide: DiscoveredSlide) -> bool:
    return any(
        (
            row["source_abspath"] != slide.source_abspath,
            row["source_relpath"] != slide.source_relpath,
            float(row["mpp"]) != float(slide.mpp),
            int(row["source_size"]) != slide.source_size,
            int(row["source_mtime_ns"]) != slide.source_mtime_ns,
            row["source_key"] != slide.source_key,
            row["config_key"] != slide.config_key,
        )
    )


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    readiness = json.loads(row["readiness_json"])
    result = None if row["result_json"] is None else json.loads(row["result_json"])
    return JobRecord(
        output_id=str(row["output_id"]),
        source_abspath=str(row["source_abspath"]),
        source_relpath=str(row["source_relpath"]),
        mpp=float(row["mpp"]),
        source_size=int(row["source_size"]),
        source_mtime_ns=int(row["source_mtime_ns"]),
        source_key=str(row["source_key"]),
        config_key=str(row["config_key"]),
        cohort=str(row["cohort"]),
        readiness_evidence=dict(readiness),
        status=row["status"],
        attempts=int(row["attempts"]),
        discovered_at=float(row["discovered_at"]),
        queued_at=float(row["queued_at"]),
        updated_at=float(row["updated_at"]),
        started_at=None if row["started_at"] is None else float(row["started_at"]),
        heartbeat_at=None if row["heartbeat_at"] is None else float(row["heartbeat_at"]),
        finished_at=None if row["finished_at"] is None else float(row["finished_at"]),
        next_retry_at=(None if row["next_retry_at"] is None else float(row["next_retry_at"])),
        error=None if row["error"] is None else str(row["error"]),
        worker_id=None if row["worker_id"] is None else str(row["worker_id"]),
        receipt_path=None if row["receipt_path"] is None else str(row["receipt_path"]),
        result=None if result is None else dict(result),
    )


def _worker_condition(worker_id: str | None) -> tuple[str, tuple[object, ...]]:
    if worker_id is None:
        return "", ()
    if not worker_id:
        raise ValueError("worker_id must be non-empty when supplied")
    return "AND worker_id = ?", (worker_id,)


def _required_error(error: str) -> str:
    if not isinstance(error, str) or not error.strip():
        raise ValueError("error must be a non-empty string")
    return error


def _default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _validate_statuses(statuses: Iterable[str]) -> None:
    invalid = set(statuses) - set(JOB_STATUSES)
    if invalid:
        raise ValueError(f"invalid job statuses: {', '.join(sorted(invalid))}")


def _mapping_contains(actual: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                return False
            if not _mapping_contains(actual_value, expected_value):
                return False
        elif type(actual_value) is not type(expected_value) or actual_value != expected_value:
            return False
    return True


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
