"""Safety and scheduling tests for operational colon slide prefetch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from oceanpath.extraction.prefetch import SlidePrefetcher, plan_due_jobs
from oceanpath.extraction.stream_encoder import SourceSnapshot
from oceanpath.workflows.streaming import _common_child_arguments, _config, _parser


@dataclass(frozen=True)
class _Job:
    output_id: str
    queued_at: float
    status: str = "pending"
    next_retry_at: float | None = None


def _snapshot(tmp_path: Path, output_id: str, contents: bytes) -> SourceSnapshot:
    path = tmp_path / f"{output_id}.svs"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    slide = SimpleNamespace(
        wsi=path.name,
        path=path,
        output_id=output_id,
        mpp=0.5,
        cohort="test",
    )
    return SourceSnapshot.capture(slide)


def _scratch_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()]


def test_plan_due_jobs_matches_three_newest_then_oldest() -> None:
    jobs = [_Job(f"slide-{index}", float(index)) for index in range(1, 6)]

    planned = plan_due_jobs(
        jobs,
        schedule_count=0,
        newest_jobs_per_oldest=3,
        count=4,
        now=100.0,
    )

    assert [job.output_id for job in planned] == [
        "slide-5",
        "slide-4",
        "slide-3",
        "slide-1",
    ]
    assert all(job.status == "pending" for job in jobs)


def test_plan_due_jobs_excludes_retry_that_is_not_due() -> None:
    jobs = [
        _Job("pending", 1.0),
        _Job("retry-due", 2.0, status="retry", next_retry_at=9.0),
        _Job("retry-later", 3.0, status="retry", next_retry_at=11.0),
    ]

    planned = plan_due_jobs(
        jobs,
        schedule_count=0,
        newest_jobs_per_oldest=3,
        count=3,
        now=10.0,
    )

    assert [job.output_id for job in planned] == ["retry-due", "pending"]


def test_prefetch_stages_atomically_and_releases_without_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = [
        _snapshot(tmp_path / "sources", "current", b"current-slide"),
        _snapshot(tmp_path / "sources", "next", b"next-slide"),
    ]

    def unexpected_fsync(_descriptor: int) -> None:
        pytest.fail("ephemeral prefetch must not fsync")

    monkeypatch.setattr("oceanpath.extraction.prefetch.os.fsync", unexpected_fsync)
    scratch = tmp_path / "scratch"
    with SlidePrefetcher(
        scratch,
        depth=1,
        max_bytes=1024,
        reserve_bytes=0,
        telemetry_path=tmp_path / "state" / "prefetch.jsonl",
    ) as prefetcher:
        prefetcher.plan(snapshots)
        staged = prefetcher.acquire(snapshots[0])
        assert staged is not None
        assert staged.read_bytes() == b"current-slide"
        assert not list(scratch.rglob("*.part"))
        prefetcher.release(snapshots[0])
        assert not staged.exists()

    assert not _scratch_files(scratch)


def test_source_replacement_falls_back_and_cleans_all_speculative_files(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    current = _snapshot(source_root, "current", b"old")
    future = _snapshot(source_root, "future", b"future")
    current.path.write_bytes(b"replacement")
    scratch = tmp_path / "scratch"
    prefetcher = SlidePrefetcher(
        scratch,
        depth=1,
        max_bytes=1024,
        reserve_bytes=0,
    )
    try:
        prefetcher.plan([current, future])
        assert prefetcher.acquire(current) is None
    finally:
        prefetcher.close()

    assert prefetcher.stats.get("copy_failed", 0) == 1
    assert not _scratch_files(scratch)


def test_max_bytes_denial_never_creates_a_partial_copy(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "sources", "too-large", b"12345")
    scratch = tmp_path / "scratch"
    with SlidePrefetcher(
        scratch,
        depth=1,
        max_bytes=4,
        reserve_bytes=0,
    ) as prefetcher:
        prefetcher.plan([snapshot])
        assert prefetcher.acquire(snapshot) is None
        assert prefetcher.stats.get("max_bytes_denied", 0) == 1

    assert not _scratch_files(scratch)


def test_cli_prefetch_defaults_propagate_and_no_local_stage_disables_it() -> None:
    args = _parser().parse_args(["launch"])
    cfg = _config(args)
    child = _common_child_arguments(args)

    assert cfg.prefetch_depth == 3
    assert cfg.prefetch_max_gib == 64.0
    assert cfg.prefetch_copy_workers == 1
    assert cfg.local_job_output is False
    depth_index = child.index("--prefetch-depth")
    max_index = child.index("--prefetch-max-gib")
    workers_index = child.index("--prefetch-copy-workers")
    assert child[depth_index + 1] == "3"
    assert child[max_index + 1] == "64.0"
    assert child[workers_index + 1] == "1"
    assert "--local-job-output" not in child

    no_stage = _parser().parse_args(["status", "--no-local-stage"])
    assert _config(no_stage).prefetch_depth == 0


def test_cli_copy_workers_and_local_job_output_propagate() -> None:
    args = _parser().parse_args(["launch", "--prefetch-copy-workers", "3", "--local-job-output"])
    cfg = _config(args)
    child = _common_child_arguments(args)

    assert cfg.prefetch_copy_workers == 3
    assert cfg.local_job_output is True
    workers_index = child.index("--prefetch-copy-workers")
    assert child[workers_index + 1] == "3"
    assert "--local-job-output" in child


def test_parallel_copy_workers_stage_all_planned_slides(tmp_path: Path) -> None:
    snapshots = [
        _snapshot(tmp_path / "sources", f"slide-{index}", f"payload-{index}".encode())
        for index in range(3)
    ]
    scratch = tmp_path / "scratch"
    with SlidePrefetcher(
        scratch,
        depth=2,
        max_bytes=1024,
        reserve_bytes=0,
        copy_workers=2,
    ) as prefetcher:
        prefetcher.plan(snapshots)
        for index, snapshot in enumerate(snapshots):
            staged = prefetcher.acquire(snapshot)
            assert staged is not None
            assert staged.read_bytes() == f"payload-{index}".encode()
            prefetcher.release(snapshot)

    assert not _scratch_files(scratch)


def test_copy_workers_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SlidePrefetcher(tmp_path / "scratch", copy_workers=0)
