"""Immutable output-lineage helpers for Aim 2 reruns.

Aim 2 has expensive upstream fits and many derived caches.  A rerun must never
silently mix those objects with an older checkpoint, nor replace the evidence
that motivated the rerun.  The environment variable below selects a *new*
lineage rooted below ``outputs/aim1/reruns``.  Individual writers use the
exclusive helpers so an existing artifact is an error rather than a cache hit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oceanpath.aim1 import paths

AIM2_LINEAGE_ENV = "OCEANPATH_AIM2_LINEAGE"
AIM2_SOURCE_CV_ROOT_ENV = "OCEANPATH_AIM2_SOURCE_CV_ROOT"

_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def lineage_name(*, required: bool = True) -> str | None:
    """Return and validate the selected Aim-2 lineage name."""

    value = os.environ.get(AIM2_LINEAGE_ENV, "").strip()
    if not value:
        if required:
            raise RuntimeError(
                f"{AIM2_LINEAGE_ENV} is required for every mutating Aim-2 command; "
                "choose a new immutable lineage name"
            )
        return None
    if not _SLUG.fullmatch(value):
        raise ValueError(
            f"Invalid {AIM2_LINEAGE_ENV}={value!r}; use letters, digits, '.', '_' or '-'"
        )
    return value


def aim2_root(*, required: bool = True) -> Path:
    """Root for the selected rerun, never the legacy Aim-2 output tree."""

    name = lineage_name(required=required)
    if name is None:
        return paths.OUTPUT_ROOT
    root = paths.OUTPUT_ROOT / "reruns" / name
    # Defensive containment check: a malformed future refactor must not turn a
    # rerun command into a writer for the frozen baseline tree.
    if root.parent != paths.OUTPUT_ROOT / "reruns":
        raise RuntimeError(f"Unsafe Aim-2 lineage root: {root}")
    return root


def component_root(component: str) -> Path:
    """Return ``<lineage>/<component>`` for a simple component name."""

    if not _SLUG.fullmatch(component):
        raise ValueError(f"Invalid Aim-2 component name: {component!r}")
    return aim2_root() / component


def eval_root() -> Path:
    return component_root("eval")


def source_cv_input_root() -> Path:
    """Explicit immutable source-CV input for a rerun.

    The 60 source-CV fits are valid and need not be repeated, but permitting an
    implicit fallback would make lineage provenance ambiguous.  New lineages
    must therefore name their source-CV tree explicitly.
    """

    selected = os.environ.get(AIM2_SOURCE_CV_ROOT_ENV, "").strip()
    if not selected:
        raise RuntimeError(
            f"{AIM2_SOURCE_CV_ROOT_ENV} is required; point it at the frozen "
            "source_cv directory used as input"
        )
    root = Path(selected).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Aim-2 source-CV input does not exist: {root}")
    return root


def ensure_absent(path: Path) -> Path:
    """Fail closed if ``path`` already exists."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable Aim-2 artifact: {path}")
    return path


def write_text_once(path: Path, value: str) -> None:
    """Create a text artifact with OS-level exclusive-create semantics."""

    ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def write_bytes_once(path: Path, value: bytes) -> None:
    """Create a binary artifact with OS-level exclusive-create semantics."""

    ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)


def write_json_once(path: Path, value: Any) -> None:
    write_text_once(path, json.dumps(value, indent=2, default=str) + "\n")


def write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    """Atomically publish a parquet without ever replacing an existing path."""

    ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_parquet(temporary, index=False)
        # hard-link creation is atomic and fails with FileExistsError if a
        # concurrent or resumed command already published the destination.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_npz_once(path: Path, **arrays: Any) -> None:
    """Atomically publish a NumPy archive without replacement."""

    ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **arrays)
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }
