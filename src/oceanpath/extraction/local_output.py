"""Operational local job-output staging for the continuous slide encoder.

Like :mod:`prefetch`, this module is deliberately kept outside
:mod:`stream_encoder`.  *Where* TRIDENT writes its artifacts while a slide is
in flight is an operational detail with no bearing on the cache identity
(:data:`stream_encoder.IMPLEMENTATION_HASH`) recorded in every stage receipt.

The encoder here runs every stage — TRIDENT output, validation re-reads, and
receipt writes — against a local scratch root, then publishes the finished
slide's artifacts to the durable feature root in a crash-safe order: data
artifacts first, stage receipts next, and the completion receipt last.  A
reconcile pass against the durable root therefore only ever observes a slide
as complete once every byte behind that claim is durable.

TRIDENT's shared run-level sidecars (``_config_*.json``, ``_logs_*.txt``) are
not per-slide artifacts and are not validated; they remain in the local job
root.  Stage receipts record the full policy, implementation hash, and
checkpoint hashes, so provenance on the durable side is unaffected.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from oceanpath.extraction.prefetch import PrefetchingSlideEncoder
from oceanpath.extraction.stream_encoder import (
    ReadySlideLike,
    SlideEncoderConfig,
    SlideEncodingError,
    SlideEncodingResult,
)

logger = logging.getLogger(__name__)

LOCAL_OUTPUT_DIRECTORY = "joboutput-v1"


def _is_disjoint(left: Path, right: Path) -> bool:
    return left != right and left not in right.parents and right not in left.parents


class LocalOutputSlideEncoder(PrefetchingSlideEncoder):
    """Slide encoder that stages all slide outputs locally, then publishes.

    Every inherited path helper, validator, and receipt writer operates on the
    local root because ``super().__init__`` receives a config whose
    ``output_root`` is the local scratch directory.  Callers that must observe
    the durable root (queue reconciliation, ``validate_complete`` at restart)
    should use a separate encoder built on the unmodified config.
    """

    def __init__(
        self,
        cfg: SlideEncoderConfig,
        *,
        local_output_root: str | Path | None = None,
    ) -> None:
        durable_root = cfg.output_root.expanduser().resolve(strict=False)
        local_root = (
            Path(local_output_root).expanduser().resolve(strict=False)
            if local_output_root is not None
            else cfg.scratch_root.expanduser().resolve(strict=False) / LOCAL_OUTPUT_DIRECTORY
        )
        if not _is_disjoint(local_root, durable_root):
            raise ValueError(
                f"Local output root {local_root} must be disjoint from the "
                f"durable output root {durable_root}"
            )
        super().__init__(dataclasses.replace(cfg, output_root=local_root))
        self.durable_root = durable_root
        self.durable_root.mkdir(parents=True, exist_ok=True)

    def process(self, slide: ReadySlideLike) -> SlideEncodingResult:
        result = super().process(slide)
        result.stage_metrics["publish"] = self._publish(result.output_id)
        return result

    def _publish_order(self, output_id: str) -> list[Path]:
        """Local artifact files for one slide: data, stage receipts, completion."""

        completion = self._completion_path(output_id)
        artifacts: list[Path] = []
        receipts: list[Path] = []
        seen: set[Path] = set()
        for path in self._stage_artifacts(output_id, "seg"):
            if path in seen or path == completion:
                continue
            seen.add(path)
            if "_stream_receipts" in path.parts:
                receipts.append(path)
            else:
                artifacts.append(path)
        return [*artifacts, *receipts, completion]

    def _publish(self, output_id: str) -> dict[str, Any]:
        """Copy one finished slide's artifacts from local scratch to durable.

        Any pre-existing durable file is archived (never silently replaced) so
        a re-encode after a source change keeps the old artifacts, matching
        ``_archive_from_stage`` semantics on the durable side.
        """

        started = time.monotonic()
        files = self._publish_order(output_id)
        for path in files:
            if not path.is_file() or path.stat().st_size <= 0:
                raise SlideEncodingError(f"Missing or empty local artifact to publish: {path}")
        archive_stamp: str | None = None
        bytes_published = 0
        for path in files:
            relative = path.relative_to(self.cfg.output_root)
            destination = self.durable_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if archive_stamp is None:
                    archive_stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                archived = self.durable_root / ".archive" / archive_stamp / output_id / relative
                archived.parent.mkdir(parents=True, exist_ok=True)
                if archived.exists():
                    archived = archived.with_name(f"{archived.name}.{uuid.uuid4().hex}")
                os.replace(destination, archived)
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                shutil.copyfile(path, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            bytes_published += path.stat().st_size
        for path in files:
            path.unlink(missing_ok=True)
        with suppress(OSError):
            self._completion_path(output_id).parent.rmdir()
        metrics = {
            "seconds": time.monotonic() - started,
            "files": len(files),
            "bytes": bytes_published,
        }
        logger.info(
            "Published %s to %s: %d files, %.1f MiB in %.1fs",
            output_id,
            self.durable_root,
            metrics["files"],
            bytes_published / 1024**2,
            metrics["seconds"],
        )
        return metrics


__all__ = [
    "LOCAL_OUTPUT_DIRECTORY",
    "LocalOutputSlideEncoder",
]
