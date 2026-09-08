"""Canonical slide selection records for TRIDENT extraction jobs.

TRIDENT accepts an optional CSV with a required ``wsi`` column and an
optional ``mpp`` column.  It also writes every downstream artifact using the
slide's basename (without its final extension), even when slides live in
nested directories.  This module centralizes those details so validation,
sharding, diff selection, and output checks can share one interpretation of
the input cohort.

The module intentionally has no TRIDENT or pandas dependency.  In particular,
it validates a fully populated MPP column before TRIDENT sees it: the pinned
TRIDENT implementation drops missing MPP values and then aligns the remaining
values positionally, which can otherwise assign an MPP to the wrong slide.
"""

from __future__ import annotations

import csv
import math
import posixpath
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

DEFAULT_WSI_EXTENSIONS = (
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".vms",
    ".vmu",
    ".scn",
    ".bif",
    ".dcm",
    ".dicom",
    ".sdpc",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
)


class SlideRecordError(ValueError):
    """Raised when a slide selection cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class SlideRecord:
    """One validated slide row in TRIDENT's input contract.

    Parameters
    ----------
    wsi:
        POSIX-style path relative to ``wsi_dir``.
    path:
        Resolved absolute path to the existing slide.
    output_id:
        Basename without its final extension, matching TRIDENT's flat output
        naming behavior.
    mpp:
        Optional explicit source microns-per-pixel override.
    """

    wsi: str
    path: Path
    output_id: str
    mpp: float | None = None


def load_slide_records(
    wsi_dir: str | Path,
    *,
    custom_list_path: str | Path | None = None,
    wsi_ext: Sequence[str] | None = None,
    search_nested: bool = False,
) -> list[SlideRecord]:
    """Load and validate the complete slide selection for an extraction job.

    When ``custom_list_path`` is supplied, it is authoritative and
    ``search_nested`` must be false.  Without a custom CSV, the filesystem is
    scanned deterministically using ``wsi_ext`` (or the default extensions).
    """

    root = _resolve_wsi_root(wsi_dir)
    extensions = _normalize_extensions(wsi_ext)

    if custom_list_path is not None:
        if search_nested:
            raise SlideRecordError(
                "search_nested must be false when custom_list_path is provided; "
                "the CSV already defines the complete slide selection"
            )
        records = _load_custom_csv(root, Path(custom_list_path), extensions)
    else:
        records = _scan_wsi_dir(root, extensions, search_nested=search_nested)

    if not records:
        source = custom_list_path if custom_list_path is not None else root
        raise SlideRecordError(f"No slides found in {source}")

    return _validated_sorted_records(records)


def shard_slide_records(
    records: Iterable[SlideRecord],
    *,
    shard_id: int,
    total_shards: int,
) -> list[SlideRecord]:
    """Return a stable, disjoint strided shard of ``records``.

    Sorting before slicing makes assignment independent of the source CSV's
    row order.  Empty shards are valid; callers can treat them as clean no-op
    jobs without constructing a TRIDENT ``Processor``.
    """

    if isinstance(total_shards, bool) or not isinstance(total_shards, int):
        raise SlideRecordError("total_shards must be an integer")
    if total_shards <= 0:
        raise SlideRecordError("total_shards must be greater than zero")
    if isinstance(shard_id, bool) or not isinstance(shard_id, int):
        raise SlideRecordError("shard_id must be an integer")
    if not 0 <= shard_id < total_shards:
        raise SlideRecordError(
            f"shard_id must satisfy 0 <= shard_id < total_shards; "
            f"got shard_id={shard_id}, total_shards={total_shards}"
        )

    ordered = _validated_sorted_records(list(records))
    return ordered[shard_id::total_shards]


def write_slide_records_csv(
    records: Iterable[SlideRecord],
    path: str | Path,
    *,
    include_mpp: bool | None = None,
) -> Path:
    """Atomically write a deterministic TRIDENT-compatible subset CSV.

    ``records`` may be the full cohort, a diff selection, or a shard.  MPP
    values are never silently discarded.  For an empty MPP-aware shard, pass
    ``include_mpp=True`` to retain the ``wsi,mpp`` schema.
    """

    ordered = _validated_sorted_records(list(records))
    populated_mpp = [record.mpp is not None for record in ordered]
    if any(populated_mpp) and not all(populated_mpp):
        raise SlideRecordError(
            "Cannot write a partially populated mpp column; every record must "
            "have an MPP value or none may have one"
        )

    inferred_include_mpp = bool(populated_mpp and all(populated_mpp))
    if include_mpp is None:
        include_mpp = inferred_include_mpp
    elif not isinstance(include_mpp, bool):
        raise SlideRecordError("include_mpp must be true, false, or None")

    if include_mpp and any(record.mpp is None for record in ordered):
        raise SlideRecordError("include_mpp=true requires an MPP value for every record")
    if not include_mpp and any(record.mpp is not None for record in ordered):
        raise SlideRecordError("Refusing to discard populated MPP values")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    fieldnames = ["wsi", "mpp"] if include_mpp else ["wsi"]

    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for record in ordered:
                row = {"wsi": record.wsi}
                if include_mpp:
                    assert record.mpp is not None  # checked above; narrows the type
                    row["mpp"] = format(record.mpp, ".17g")
                writer.writerow(row)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return output_path


def _resolve_wsi_root(wsi_dir: str | Path) -> Path:
    root = Path(wsi_dir).expanduser()
    if not root.is_dir():
        raise SlideRecordError(f"Slide directory does not exist: {root}")
    return root.resolve(strict=True)


def _normalize_extensions(wsi_ext: Sequence[str] | None) -> tuple[str, ...]:
    raw_extensions = DEFAULT_WSI_EXTENSIONS if wsi_ext is None else tuple(wsi_ext)
    normalized: set[str] = set()
    for raw_extension in raw_extensions:
        extension = str(raw_extension).strip().lower()
        if not extension or not extension.startswith("."):
            raise SlideRecordError(
                f"Invalid WSI extension {raw_extension!r}; extensions must start with '.'"
            )
        normalized.add(extension)
    if not normalized:
        raise SlideRecordError("At least one WSI extension is required")
    return tuple(sorted(normalized))


def _load_custom_csv(
    root: Path,
    csv_path: Path,
    extensions: tuple[str, ...],
) -> list[SlideRecord]:
    if not csv_path.is_file():
        raise SlideRecordError(f"Custom WSI CSV does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SlideRecordError(f"Custom WSI CSV has no header: {csv_path}")

        fieldnames = [field.strip() if field is not None else "" for field in reader.fieldnames]
        if any(not field for field in fieldnames):
            raise SlideRecordError(f"Custom WSI CSV contains a blank column name: {csv_path}")
        if len(set(fieldnames)) != len(fieldnames):
            raise SlideRecordError(f"Custom WSI CSV contains duplicate column names: {csv_path}")
        reader.fieldnames = fieldnames

        if "wsi" not in fieldnames:
            raise SlideRecordError(f"Custom WSI CSV must contain a 'wsi' column: {csv_path}")
        has_mpp = "mpp" in fieldnames

        records: list[SlideRecord] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise SlideRecordError(
                    f"Malformed CSV row {row_number} in {csv_path}: too many fields"
                )
            raw_wsi = row.get("wsi")
            if raw_wsi is None or not raw_wsi.strip():
                raise SlideRecordError(f"Missing wsi value at row {row_number} in {csv_path}")

            mpp = None
            if has_mpp:
                raw_mpp = row.get("mpp")
                if raw_mpp is None or not raw_mpp.strip():
                    raise SlideRecordError(
                        f"Missing mpp value at row {row_number} in {csv_path}; "
                        "omit the mpp column entirely or populate every row"
                    )
                mpp = _parse_mpp(raw_mpp, csv_path=csv_path, row_number=row_number)

            records.append(
                _record_from_relative_path(
                    root,
                    raw_wsi,
                    extensions,
                    mpp=mpp,
                    context=f"row {row_number} in {csv_path}",
                )
            )

    return records


def _parse_mpp(raw_mpp: str, *, csv_path: Path, row_number: int) -> float:
    try:
        mpp = float(raw_mpp)
    except ValueError as exc:
        raise SlideRecordError(
            f"Invalid mpp value {raw_mpp!r} at row {row_number} in {csv_path}"
        ) from exc
    if not math.isfinite(mpp) or mpp <= 0:
        raise SlideRecordError(
            f"MPP must be finite and positive at row {row_number} in {csv_path}; got {raw_mpp!r}"
        )
    return mpp


def _scan_wsi_dir(
    root: Path,
    extensions: tuple[str, ...],
    *,
    search_nested: bool,
) -> list[SlideRecord]:
    candidates = root.rglob("*") if search_nested else root.iterdir()
    relative_paths = sorted(
        (
            candidate.relative_to(root).as_posix()
            for candidate in candidates
            if candidate.is_file() and _matches_extension(candidate.name, extensions)
        ),
        key=_sort_key,
    )
    return [
        _record_from_relative_path(root, relative, extensions, context=str(root))
        for relative in relative_paths
    ]


def _record_from_relative_path(
    root: Path,
    raw_wsi: str,
    extensions: tuple[str, ...],
    *,
    mpp: float | None = None,
    context: str,
) -> SlideRecord:
    normalized = str(raw_wsi).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    relative = PurePosixPath(normalized)

    if (
        not normalized
        or relative.is_absolute()
        or PureWindowsPath(normalized).is_absolute()
        or ".." in relative.parts
    ):
        raise SlideRecordError(
            f"WSI path must be relative to the slide directory and may not traverse it: "
            f"{raw_wsi!r} ({context})"
        )

    canonical_wsi = relative.as_posix()
    if not _matches_extension(canonical_wsi, extensions):
        raise SlideRecordError(
            f"Unsupported WSI extension for {canonical_wsi!r} ({context}); "
            f"allowed extensions: {list(extensions)}"
        )

    try:
        resolved_path = (root / Path(*relative.parts)).resolve(strict=True)
    except FileNotFoundError as exc:
        raise SlideRecordError(f"Slide does not exist: {canonical_wsi!r} ({context})") from exc
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise SlideRecordError(
            f"Slide resolves outside the configured slide directory: {canonical_wsi!r} ({context})"
        ) from exc
    if not resolved_path.is_file():
        raise SlideRecordError(f"Slide is not a regular file: {canonical_wsi!r} ({context})")

    output_id, _ = posixpath.splitext(relative.name)
    if not output_id:
        raise SlideRecordError(f"Cannot derive an output ID from {canonical_wsi!r} ({context})")

    return SlideRecord(
        wsi=canonical_wsi,
        path=resolved_path,
        output_id=output_id,
        mpp=mpp,
    )


def _matches_extension(filename: str, extensions: tuple[str, ...]) -> bool:
    lowered = filename.lower()
    return any(lowered.endswith(extension) for extension in extensions)


def _sort_key(value: str) -> tuple[str, str]:
    return value.casefold(), value


def _validated_sorted_records(records: list[SlideRecord]) -> list[SlideRecord]:
    ordered = sorted(records, key=lambda record: _sort_key(record.wsi))
    seen_paths: dict[str, str] = {}
    seen_output_ids: dict[str, str] = {}
    has_mpp = {record.mpp is not None for record in ordered}
    if len(has_mpp) > 1:
        raise SlideRecordError(
            "MPP values are partially populated; every record must have an MPP "
            "value or none may have one"
        )

    for record in ordered:
        path_key = record.wsi.casefold()
        if path_key in seen_paths:
            raise SlideRecordError(
                f"Duplicate WSI path {record.wsi!r}; first occurrence: {seen_paths[path_key]!r}"
            )
        seen_paths[path_key] = record.wsi

        output_key = record.output_id.casefold()
        if output_key in seen_output_ids:
            raise SlideRecordError(
                "TRIDENT flattens nested outputs by basename; slides "
                f"{seen_output_ids[output_key]!r} and {record.wsi!r} would both write "
                f"output ID {record.output_id!r}"
            )
        seen_output_ids[output_key] = record.wsi

        if record.mpp is not None and (not math.isfinite(record.mpp) or record.mpp <= 0):
            raise SlideRecordError(
                f"MPP must be finite and positive for {record.wsi!r}; got {record.mpp!r}"
            )

    return ordered


__all__ = [
    "DEFAULT_WSI_EXTENSIONS",
    "SlideRecord",
    "SlideRecordError",
    "load_slide_records",
    "shard_slide_records",
    "write_slide_records_csv",
]
