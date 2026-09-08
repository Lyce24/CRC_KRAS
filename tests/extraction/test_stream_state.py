"""Tests for durable, growing per-slide extraction state."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from oceanpath.extraction.stream_state import (
    DiscoveredSlide,
    LockUnavailableError,
    StateTransitionError,
    StreamState,
    exclusive_process_lock,
    receipt_matches,
    write_stage_receipt,
)


def _slide(
    tmp_path: Path,
    output_id: str,
    *,
    size: int = 100,
    mtime_ns: int = 1_000,
    source_key: str | None = None,
    config_key: str = "config-v1",
) -> DiscoveredSlide:
    source = (tmp_path / "slides" / f"{output_id}.svs").resolve()
    return DiscoveredSlide(
        output_id=output_id,
        source_abspath=str(source),
        source_relpath=f"cohort/{output_id}.svs",
        mpp=0.25,
        source_size=size,
        source_mtime_ns=mtime_ns,
        source_key=source_key or f"source-{output_id}-v1",
        config_key=config_key,
        cohort="example",
        readiness_evidence={"producer": "verified", "size": size},
    )


def test_growing_discovery_is_idempotent_and_preserves_completion(tmp_path):
    state = StreamState(tmp_path / "state.sqlite")
    alpha = _slide(tmp_path, "alpha")

    first = state.upsert(alpha, now=10)
    repeated = state.upsert(alpha, now=20)

    assert first.inserted and not first.reset
    assert not repeated.inserted and not repeated.reset
    assert repeated.record.discovered_at == 10
    assert repeated.record.queued_at == 10
    assert repeated.record.updated_at == 10

    claimed = state.claim_next(worker_id="worker-a", now=30)
    assert claimed is not None
    complete = state.mark_complete(
        "alpha",
        worker_id="worker-a",
        receipt_path="receipts/alpha.json",
        result={"patches": 42},
        now=40,
    )
    after_complete = state.upsert(alpha, now=50)
    beta = state.upsert(_slide(tmp_path, "beta"), now=60)

    assert complete.status == "complete"
    assert complete.attempts == 1
    assert complete.result == {"patches": 42}
    assert after_complete.record == complete
    assert beta.inserted
    assert [job.output_id for job in state.list_jobs()] == ["alpha", "beta"]
    assert state.summary(now=60) == {
        "pending": 1,
        "processing": 0,
        "complete": 1,
        "retry": 0,
        "failed": 0,
        "blocked": 0,
        "claimable": 1,
        "total": 2,
    }


@pytest.mark.parametrize(
    "replacement",
    [
        {"size": 101, "mtime_ns": 2_000, "source_key": "source-alpha-v2"},
        {"config_key": "config-v2"},
    ],
)
def test_source_or_config_change_resets_completed_job(tmp_path, replacement):
    state = StreamState(tmp_path / "state.sqlite")
    state.upsert(_slide(tmp_path, "alpha"), now=10)
    claimed = state.claim_next(worker_id="worker-a", now=20)
    assert claimed is not None
    state.mark_complete(
        "alpha",
        worker_id="worker-a",
        receipt_path="old.json",
        result={"old": True},
        now=30,
    )

    reset = state.upsert(_slide(tmp_path, "alpha", **replacement), now=40)

    assert reset.reset and not reset.inserted
    assert reset.record.status == "pending"
    assert reset.record.attempts == 0
    assert reset.record.discovered_at == 10
    assert reset.record.queued_at == 40
    assert reset.record.started_at is None
    assert reset.record.finished_at is None
    assert reset.record.receipt_path is None
    assert reset.record.result is None


def test_claim_is_exclusive_across_connections_and_supports_fair_order(tmp_path):
    database = tmp_path / "state.sqlite"
    state = StreamState(database)
    state.upsert(_slide(tmp_path, "oldest"), now=10)
    state.upsert(_slide(tmp_path, "middle"), now=20)
    state.upsert(_slide(tmp_path, "newest"), now=30)

    newest = state.claim_next(order="newest", worker_id="new-worker", now=40)
    oldest = state.claim_next(order="oldest", worker_id="old-worker", now=40)

    assert newest is not None and newest.output_id == "newest"
    assert oldest is not None and oldest.output_id == "oldest"

    first_connection = StreamState(database)
    second_connection = StreamState(database)
    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(
            executor.map(
                lambda item: item[0].claim_next(worker_id=item[1], now=50),
                [(first_connection, "one"), (second_connection, "two")],
            )
        )

    nonempty = [claim for claim in claims if claim is not None]
    assert len(nonempty) == 1
    assert nonempty[0].output_id == "middle"
    assert state.claim_next(now=50) is None
    assert state.summary(now=50)["processing"] == 3


def test_retry_heartbeat_and_orphan_recovery(tmp_path):
    state = StreamState(tmp_path / "state.sqlite")
    state.upsert_many([_slide(tmp_path, "alpha"), _slide(tmp_path, "beta")], now=0)
    alpha = state.claim_next(worker_id="worker-a", now=10)
    assert alpha is not None and alpha.output_id == "alpha"
    assert state.heartbeat("alpha", worker_id="wrong-worker", now=12) is False
    assert state.heartbeat("alpha", worker_id="worker-a", now=12) is True
    assert state.recover_orphans(stale_after_seconds=5, now=16) == 0
    assert state.recover_orphans(stale_after_seconds=5, now=18) == 1

    recovered = state.claim_next(order="oldest", worker_id="worker-b", now=18)
    assert recovered is not None and recovered.output_id == "alpha"
    assert recovered.attempts == 2
    retried = state.mark_retry(
        "alpha", "temporary I/O error", worker_id="worker-b", delay_seconds=10, now=20
    )
    assert retried.status == "retry"
    assert retried.next_retry_at == 30

    beta = state.claim_next(order="oldest", worker_id="worker-b", now=25)
    assert beta is not None and beta.output_id == "beta"
    state.mark_failed("beta", "invalid source", worker_id="worker-b", now=26)
    assert state.claim_next(now=29) is None
    alpha_again = state.claim_next(worker_id="worker-c", now=30)
    assert alpha_again is not None and alpha_again.output_id == "alpha"

    with pytest.raises(StateTransitionError):
        state.mark_complete("alpha", worker_id="stale-worker", now=31)
    blocked = state.mark_blocked("alpha", "operator action needed", worker_id="worker-c", now=32)
    assert blocked.status == "blocked"
    assert state.summary(now=32)["claimable"] == 0


def test_stage_receipt_is_atomic_and_identity_matched(tmp_path):
    receipt_path = tmp_path / "nested" / "alpha.features.json"
    written = write_stage_receipt(
        receipt_path,
        output_id="alpha",
        stage="features",
        source_key="source-v1",
        config_key="config-v1",
        payload={"rows": 42, "models": {"uni": 1024, "conch": 768}},
        now=100,
    )

    assert json.loads(receipt_path.read_text(encoding="utf-8")) == written
    assert not list(receipt_path.parent.glob(f".{receipt_path.name}.*.tmp"))
    assert receipt_matches(
        receipt_path,
        output_id="alpha",
        stage="features",
        source_key="source-v1",
        config_key="config-v1",
        payload={"rows": 42},
    )
    assert not receipt_matches(
        receipt_path,
        output_id="alpha",
        stage="features",
        source_key="source-v2",
        config_key="config-v1",
    )

    receipt_path.write_text("not json", encoding="utf-8")
    assert not receipt_matches(
        receipt_path,
        output_id="alpha",
        stage="features",
        source_key="source-v1",
        config_key="config-v1",
    )


def test_complete_job_can_be_requeued_after_artifact_validation_fails(tmp_path):
    state = StreamState(tmp_path / "state.sqlite")
    state.upsert(_slide(tmp_path, "alpha"), now=1)
    state.claim_next(worker_id="worker", now=2)
    state.mark_complete("alpha", worker_id="worker", now=3)

    requeued = state.requeue("alpha", "feature H5 missing", now=4)

    assert requeued.status == "pending"
    assert requeued.error == "feature H5 missing"
    assert requeued.queued_at == 4
    claimed = state.claim_next(worker_id="repair", now=5)
    assert claimed is not None and claimed.output_id == "alpha"
    with pytest.raises(StateTransitionError, match="actively processing"):
        state.requeue("alpha", "cannot steal", now=6)


def test_validated_completion_can_be_restored_after_queue_reset(tmp_path):
    state = StreamState(tmp_path / "state.sqlite")
    alpha = _slide(tmp_path, "alpha")
    state.upsert(alpha, now=1)

    restored = state.restore_complete(
        "alpha",
        source_key=alpha.source_key,
        config_key=alpha.config_key,
        receipt_path="receipts/alpha/complete.json",
        result={"restored": True},
        now=2,
    )

    assert restored.status == "complete"
    assert restored.attempts == 0
    assert restored.result == {"restored": True}
    with pytest.raises(StateTransitionError, match="identity changed"):
        state.restore_complete(
            "alpha",
            source_key="stale-source",
            config_key=alpha.config_key,
            receipt_path="stale.json",
            now=3,
        )


def test_process_lock_rejects_a_second_owner(tmp_path):
    lock_path = tmp_path / "watcher.lock"
    with (
        exclusive_process_lock(lock_path),
        pytest.raises(LockUnavailableError),
        exclusive_process_lock(lock_path),
    ):
        pytest.fail("second lock owner should not enter")

    with exclusive_process_lock(lock_path):
        assert "pid=" in lock_path.read_text(encoding="utf-8")


def test_rejects_duplicate_discovery_ids_before_writing(tmp_path):
    state = StreamState(tmp_path / "state.sqlite")
    alpha = _slide(tmp_path, "alpha")
    with pytest.raises(ValueError, match="duplicate output_id"):
        state.upsert_many([alpha, alpha], now=1)
    assert state.summary(now=1)["total"] == 0
