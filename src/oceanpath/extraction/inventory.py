"""Authoritative, fail-closed inventory for the live colon extraction queue.

The producers for TCGA downloads, SurGen colour correction, and RIH tile
repair can run concurrently with extraction.  This module turns their
partially completed directories into one deterministic readiness ledger:

* TCGA: a GDC diagnostic slide is ready only when the final ``.svs`` size
  matches the GDC manifest and MPP can be read from slide metadata.
* SurGen: a corrected ``SURGEN_fixed`` TIFF is ready only after the append-only
  conversion state says ``done``.
* RIH: a completed fixed TIFF supersedes its original.  Otherwise the
  original is ready only when a header-only audit proves its L0 tiles are
  TIFF-compliant (both dimensions are multiples of 16).
* CPTAC: a curated transfer-manifest slide is ready only when its final SVS
  byte size matches the manifest and MPP can be read from slide metadata.

The three official cohort workbooks are annotative by default: all valid
canonical slides are queued and every ledger row records ``label_match``.
Callers may opt into a label-only inventory with ``require_label_match=True``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import posixpath
import re
import struct
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

_WSI_SUFFIXES = (".svs", ".tiff", ".tif")
_ID_SUFFIXES = (*_WSI_SUFFIXES, ".czi", ".jpeg", ".jpg", ".png")
_COHORT_ORDER = {"RIH": 0, "SurGen": 1, "TCGA": 2}
_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_OFFICE_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_CELL_COLUMN = re.compile(r"^([A-Za-z]+)")
_APERIO_MPP = re.compile(
    r"(?:^|\|)\s*MPP\s*=\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)",
    re.IGNORECASE,
)
_MIN_COLON_MPP = 0.1
_MAX_COLON_MPP = 1.0

JsonScalar = str | int | float | bool | None
SlideMppResolver = Callable[[Path], float | None]
RihRepairAudit = Callable[[Path], bool]


class InventoryError(ValueError):
    """Raised when an authoritative colon input cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class InventoryPaths:
    """Roots and authoritative manifests used by colon readiness discovery."""

    slide_root: Path
    manifest_root: Path
    tcga_clinical_workbook: Path
    surgen_clinical_workbook: Path
    rih_clinical_workbook: Path
    surgen_state: Path
    rih_state: Path
    gdc_manifest: Path

    @classmethod
    def from_roots(
        cls,
        slide_root: str | Path,
        manifest_root: str | Path,
    ) -> InventoryPaths:
        """Build the production layout from the two colon root directories."""

        slides = Path(slide_root)
        manifests = Path(manifest_root)
        return cls(
            slide_root=slides,
            manifest_root=manifests,
            tcga_clinical_workbook=manifests / "TCGA_COAD_READ_clinical_by_slide.xlsx",
            surgen_clinical_workbook=(
                manifests / "SurGen-1020_clinical_by_slide_paper_reconciled.xlsx"
            ),
            rih_clinical_workbook=manifests / "RIH_clinical_reconciliation.xlsx",
            surgen_state=slides / "fix_surgen_state.jsonl",
            rih_state=slides / "fix_rih_state.jsonl",
            gdc_manifest=slides / "TCGA" / "gdc_manifest.json",
        )

    @property
    def tcga_dir(self) -> Path:
        return Path(self.slide_root) / "TCGA"

    @property
    def surgen_original_dir(self) -> Path:
        swapped_backup = Path(self.slide_root) / "SURGEN_rb_swapped_original"
        return swapped_backup if swapped_backup.is_dir() else Path(self.slide_root) / "SURGEN"

    @property
    def surgen_fixed_dir(self) -> Path:
        active_output = Path(self.slide_root) / "SURGEN_fixed"
        if active_output.is_dir():
            return active_output
        canonical = Path(self.slide_root) / "SURGEN"
        swapped_backup = Path(self.slide_root) / "SURGEN_rb_swapped_original"
        return canonical if canonical.is_dir() and swapped_backup.is_dir() else active_output

    @property
    def rih_original_dir(self) -> Path:
        return Path(self.slide_root) / "rih"

    @property
    def rih_fixed_dir(self) -> Path:
        return Path(self.slide_root) / "rih_fixed"

    @property
    def cptac_dir(self) -> Path:
        return Path(self.slide_root) / "CPTAC_COAD"

    @property
    def cptac_transfer_manifest(self) -> Path:
        return self.cptac_dir / "metadata" / "transfer_manifest_98.csv"

    @property
    def cptac_label_manifest(self) -> Path:
        return self.cptac_dir / "metadata" / "master_labels_98.csv"


@dataclass(frozen=True, slots=True)
class ReadySlide:
    """One immutable input that is safe to submit to extraction now."""

    wsi: str
    path: Path
    output_id: str
    mpp: float
    cohort: str
    source_status: str
    evidence: str
    label_match: bool
    details: Mapping[str, JsonScalar]
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class InventoryStatus:
    """One audit-ledger row, including waiting and deliberately excluded data."""

    wsi: str
    path: Path
    output_id: str
    mpp: float | None
    cohort: str
    status: str
    source_status: str
    reason: str
    evidence: str
    label_match: bool
    details: Mapping[str, JsonScalar]
    size_bytes: int | None = None
    mtime_ns: int | None = None


@dataclass(frozen=True, slots=True)
class InventoryResult:
    """Ready queue plus the full deterministic discovery ledger and counts."""

    ready: tuple[ReadySlide, ...]
    status_rows: tuple[InventoryStatus, ...]
    summary: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _GdcSlide:
    file_name: str
    output_id: str
    size_bytes: int
    md5: str
    project: str


@dataclass(frozen=True, slots=True)
class _CptacSlide:
    file_name: str
    output_id: str
    size_bytes: int
    md5: str
    transfer_group: str


def discover_ready_slides(
    paths: InventoryPaths,
    *,
    require_label_match: bool = False,
    mpp_resolver: SlideMppResolver | None = None,
    rih_requires_repair: RihRepairAudit | None = None,
) -> InventoryResult:
    """Discover the canonical slides that are ready at this instant.

    Slide metadata, not cohort averages or nominal magnification, supplies
    source MPP.  The two callbacks are injectable for focused fixtures; the
    defaults inspect only TIFF metadata and never decode slide pixels.
    """

    resolve_mpp = read_slide_mpp if mpp_resolver is None else mpp_resolver
    audit_rih = rih_original_requires_repair if rih_requires_repair is None else rih_requires_repair

    slide_root = Path(paths.slide_root)
    if not slide_root.is_dir():
        raise InventoryError(f"Colon slide root does not exist: {slide_root}")

    tcga_labels = _load_label_ids(
        Path(paths.tcga_clinical_workbook),
        sheet_name="TCGA_Clinical",
        id_column="Slide_ID",
    )
    surgen_labels = _load_label_ids(
        Path(paths.surgen_clinical_workbook),
        sheet_name="SurGen_Clinical",
        id_column="Slide_ID",
    )
    rih_labels = _load_label_ids(
        Path(paths.rih_clinical_workbook),
        sheet_name="RIH_Clinical",
        id_column="Slide_ID",
    )
    gdc_slides = _read_gdc_manifest(Path(paths.gdc_manifest))
    surgen_state = _read_jsonl_state(Path(paths.surgen_state))
    rih_state = _read_jsonl_state(Path(paths.rih_state))

    status_rows: list[InventoryStatus] = []
    status_rows.extend(
        _discover_tcga(
            paths,
            gdc_slides,
            tcga_labels,
            resolve_mpp,
            require_label_match=require_label_match,
        )
    )
    status_rows.extend(
        _discover_surgen(
            paths,
            surgen_labels,
            surgen_state,
            resolve_mpp,
            require_label_match=require_label_match,
        )
    )
    status_rows.extend(
        _discover_rih(
            paths,
            rih_labels,
            rih_state,
            resolve_mpp,
            audit_rih,
            require_label_match=require_label_match,
        )
    )
    if paths.cptac_dir.is_dir():
        cptac_slides = _read_cptac_manifest(paths.cptac_transfer_manifest)
        cptac_labels = _read_cptac_label_ids(paths.cptac_label_manifest)
        status_rows.extend(
            _discover_cptac(
                paths,
                cptac_slides,
                cptac_labels,
                resolve_mpp,
                require_label_match=require_label_match,
            )
        )

    ordered_status = tuple(sorted(status_rows, key=_status_sort_key))
    ready = tuple(_ready_from_status(row) for row in ordered_status if row.status == "ready")
    _validate_ready(ready)
    return InventoryResult(
        ready=ready,
        status_rows=ordered_status,
        summary=_summarize(ordered_status),
    )


def write_inventory_csvs(
    result: InventoryResult,
    *,
    mpp_path: str | Path,
    status_path: str | Path,
) -> tuple[Path, Path]:
    """Atomically write TRIDENT's MPP sheet and the complete audit ledger."""

    _validate_ready(result.ready)
    mpp_csv = Path(mpp_path)
    status_csv = Path(status_path)
    _atomic_write_csv(
        mpp_csv,
        fieldnames=("wsi", "mpp"),
        rows=({"wsi": record.wsi, "mpp": format(record.mpp, ".17g")} for record in result.ready),
    )
    _atomic_write_csv(
        status_csv,
        fieldnames=(
            "wsi",
            "mpp",
            "output_id",
            "cohort",
            "status",
            "source_status",
            "label_match",
            "reason",
            "evidence",
            "size_bytes",
            "mtime_ns",
            "details",
        ),
        rows=(_status_csv_row(row) for row in result.status_rows),
    )
    return mpp_csv, status_csv


def refresh_inventory(
    paths: InventoryPaths,
    *,
    mpp_path: str | Path,
    status_path: str | Path,
    require_label_match: bool = False,
    mpp_resolver: SlideMppResolver | None = None,
    rih_requires_repair: RihRepairAudit | None = None,
) -> InventoryResult:
    """Discover ready slides, atomically refresh both sheets, and return them."""

    result = discover_ready_slides(
        paths,
        require_label_match=require_label_match,
        mpp_resolver=mpp_resolver,
        rih_requires_repair=rih_requires_repair,
    )
    write_inventory_csvs(result, mpp_path=mpp_path, status_path=status_path)
    return result


def read_slide_mpp(path: Path) -> float | None:
    """Read physical pixel size from TIFF/SVS metadata without pixel decoding."""

    tags = _cached_tiff_ifd0_tags(path)
    description = tags.get(270)
    if isinstance(description, str):
        match = _APERIO_MPP.search(description)
        if match is not None:
            mpp = _positive_float_or_none(match.group(1))
            if mpp is not None:
                return mpp

    x_resolution = _positive_float_or_none(tags.get(282))
    y_resolution = _positive_float_or_none(tags.get(283))
    resolution_unit = tags.get(296)
    resolutions = [value for value in (x_resolution, y_resolution) if value is not None]
    if not resolutions or resolution_unit not in (2, 3):
        return None
    if len(resolutions) == 2 and not math.isclose(
        resolutions[0], resolutions[1], rel_tol=0.01, abs_tol=1e-12
    ):
        return None
    pixels_per_unit = sum(resolutions) / len(resolutions)
    microns_per_unit = 25_400.0 if resolution_unit == 2 else 10_000.0
    return _positive_float_or_none(microns_per_unit / pixels_per_unit)


def rih_original_requires_repair(path: Path) -> bool:
    """Fail closed unless IFD0 tile width and height are positive multiples of 16."""

    tags = _cached_tiff_ifd0_tags(path)
    width = tags.get(322)
    height = tags.get(323)
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return True
    return width % 16 != 0 or height % 16 != 0


def _discover_tcga(
    paths: InventoryPaths,
    gdc_slides: list[_GdcSlide],
    labels: set[str],
    resolve_mpp: SlideMppResolver,
    *,
    require_label_match: bool,
) -> list[InventoryStatus]:
    rows: list[InventoryStatus] = []
    for source in gdc_slides:
        slide_id = _canonical_slide_id(source.file_name)
        slide = paths.tcga_dir / source.file_name
        relative = _expected_relative(paths, slide)
        label_match = slide_id in labels
        base_details: dict[str, JsonScalar] = {
            "expected_size_bytes": source.size_bytes,
            "gdc_md5": source.md5,
            "gdc_project": source.project,
        }
        if require_label_match and not label_match:
            rows.append(_label_excluded(relative, slide, source.output_id, "TCGA", base_details))
            continue
        if not slide.is_file():
            rows.append(
                _status(
                    relative,
                    slide,
                    source.output_id,
                    cohort="TCGA",
                    status="waiting",
                    source_status="waiting_download",
                    reason="verified final .svs has not been published yet",
                    evidence="GDC manifest target; .part files are ignored",
                    label_match=label_match,
                    details=base_details,
                )
            )
            continue
        stat = slide.stat()
        details = {**base_details, "actual_size_bytes": stat.st_size}
        if stat.st_size != source.size_bytes:
            rows.append(
                _status(
                    relative,
                    slide,
                    source.output_id,
                    cohort="TCGA",
                    status="waiting",
                    source_status="size_mismatch",
                    reason="final .svs byte size does not match the GDC manifest",
                    evidence="GDC data.hits.file_size",
                    label_match=label_match,
                    details=details,
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
            continue
        mpp = _resolve_valid_mpp(resolve_mpp, slide)
        details.update(_mpp_details(mpp))
        if mpp is None:
            rows.append(
                _status(
                    relative,
                    slide,
                    source.output_id,
                    cohort="TCGA",
                    status="waiting",
                    source_status="invalid_mpp",
                    reason="source MPP cannot be proven from slide metadata",
                    evidence="TIFF ImageDescription/resolution tags",
                    label_match=label_match,
                    details=details,
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
            continue
        rows.append(
            _status(
                relative,
                slide,
                source.output_id,
                cohort="TCGA",
                status="ready",
                source_status="download_verified",
                reason="final .svs matches GDC size and has metadata MPP",
                evidence="GDC file_size + atomic final name + TIFF metadata",
                label_match=label_match,
                details=details,
                mpp=mpp,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    return rows


def _discover_surgen(
    paths: InventoryPaths,
    labels: set[str],
    state: dict[str, str],
    resolve_mpp: SlideMppResolver,
    *,
    require_label_match: bool,
) -> list[InventoryStatus]:
    rows: list[InventoryStatus] = []
    original_by_id = _index_files(paths.surgen_original_dir, suffix=".tiff")
    fixed_by_id = _index_files(paths.surgen_fixed_dir, suffix=".tiff")
    candidates = {
        slide_id: _strip_id_suffix(path.name) for slide_id, path in original_by_id.items()
    }
    # State entries keep fixtures and a future post-swap layout discoverable,
    # while the original directory remains the authoritative full target set.
    for slide_id in state:
        candidates.setdefault(slide_id, slide_id)

    for slide_id, output_id in candidates.items():
        fixed = fixed_by_id.get(slide_id)
        expected = fixed or (paths.surgen_fixed_dir / f"{output_id}.tiff")
        relative = _expected_relative(paths, expected)
        label_match = slide_id in labels
        details: dict[str, JsonScalar] = {"conversion_state": state.get(slide_id, "pending")}
        if require_label_match and not label_match:
            rows.append(_label_excluded(relative, expected, output_id, "SurGen", details))
            continue
        if state.get(slide_id) != "done" or fixed is None:
            rows.append(
                _status(
                    relative,
                    expected,
                    output_id,
                    cohort="SurGen",
                    status="waiting",
                    source_status="waiting_conversion",
                    reason="colour-corrected final TIFF is not verified done",
                    evidence="fix_surgen_state.jsonl + SURGEN_fixed final name",
                    label_match=label_match,
                    details=details,
                )
            )
            continue
        stat = fixed.stat()
        mpp = _resolve_valid_mpp(resolve_mpp, fixed)
        details.update(_mpp_details(mpp, 0.25))
        if mpp is None:
            rows.append(
                _status(
                    relative,
                    fixed,
                    output_id,
                    cohort="SurGen",
                    status="waiting",
                    source_status="invalid_mpp",
                    reason="fixed TIFF has no valid physical resolution metadata",
                    evidence="TIFF resolution tags",
                    label_match=label_match,
                    details=details,
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
            continue
        rows.append(
            _status(
                relative,
                fixed,
                output_id,
                cohort="SurGen",
                status="ready",
                source_status="fixed_verified",
                reason="colour-corrected TIFF has state=done and metadata MPP",
                evidence="fix_surgen_state.jsonl + final TIFF + TIFF metadata",
                label_match=label_match,
                details=details,
                mpp=mpp,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    return rows


def _discover_cptac(
    paths: InventoryPaths,
    manifest: list[_CptacSlide],
    labels: set[str],
    resolve_mpp: SlideMppResolver,
    *,
    require_label_match: bool,
) -> list[InventoryStatus]:
    expected_names = {source.file_name.casefold() for source in manifest}
    actual_names = {path.name.casefold() for path in _files_with_suffix(paths.cptac_dir, ".svs")}
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        raise InventoryError(
            f"CPTAC_COAD contains SVS files absent from the curated transfer manifest: {unexpected}"
        )

    rows: list[InventoryStatus] = []
    for source in manifest:
        slide = paths.cptac_dir / source.file_name
        relative = _expected_relative(paths, slide)
        label_match = _canonical_slide_id(source.file_name) in labels
        details: dict[str, JsonScalar] = {
            "expected_size_bytes": source.size_bytes,
            "source_md5": source.md5,
            "transfer_group": source.transfer_group,
        }
        if require_label_match and not label_match:
            rows.append(_label_excluded(relative, slide, source.output_id, "CPTAC", details))
            continue
        if not slide.is_file():
            rows.append(
                _status(
                    relative,
                    slide,
                    source.output_id,
                    cohort="CPTAC",
                    status="waiting",
                    source_status="waiting_transfer",
                    reason="curated final .svs has not been transferred yet",
                    evidence="transfer_manifest_98.csv target; partial files are ignored",
                    label_match=label_match,
                    details=details,
                )
            )
            continue
        stat = slide.stat()
        details["actual_size_bytes"] = stat.st_size
        if stat.st_size != source.size_bytes:
            rows.append(
                _status(
                    relative,
                    slide,
                    source.output_id,
                    cohort="CPTAC",
                    status="waiting",
                    source_status="size_mismatch",
                    reason="final .svs byte size does not match the curated transfer manifest",
                    evidence="transfer_manifest_98.csv local_bytes",
                    label_match=label_match,
                    details=details,
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
            continue
        mpp = _resolve_valid_mpp(resolve_mpp, slide)
        details.update(_mpp_details(mpp))
        if mpp is None:
            rows.append(
                _status(
                    relative,
                    slide,
                    source.output_id,
                    cohort="CPTAC",
                    status="waiting",
                    source_status="invalid_mpp",
                    reason="source MPP cannot be proven from slide metadata",
                    evidence="TIFF ImageDescription/resolution tags",
                    label_match=label_match,
                    details=details,
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
            continue
        rows.append(
            _status(
                relative,
                slide,
                source.output_id,
                cohort="CPTAC",
                status="ready",
                source_status="transfer_verified",
                reason="final .svs matches curated transfer size and has metadata MPP",
                evidence="transfer_manifest_98.csv + master_labels_98.csv + TIFF metadata",
                label_match=label_match,
                details=details,
                mpp=mpp,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    return rows


def _discover_rih(
    paths: InventoryPaths,
    labels: set[str],
    state: dict[str, str],
    resolve_mpp: SlideMppResolver,
    audit: RihRepairAudit,
    *,
    require_label_match: bool,
) -> list[InventoryStatus]:
    rows: list[InventoryStatus] = []
    original_by_id = _index_files(paths.rih_original_dir, suffix=".svs")
    legacy_fixed_by_id = _index_files(paths.rih_fixed_dir, suffix=".tiff")
    inplace_fixed_by_id = _index_files(paths.rih_original_dir, suffix=".tiff")
    repair_named = set(state)
    slide_ids = sorted(
        set(original_by_id) | set(legacy_fixed_by_id) | set(inplace_fixed_by_id) | repair_named
    )
    for slide_id in slide_ids:
        original = original_by_id.get(slide_id)
        # The repair producer writes to rih_fixed while active, then atomically
        # consolidates verified TIFFs into rih.  The final in-place path wins if
        # both are briefly visible during that handoff.
        fixed = inplace_fixed_by_id.get(slide_id, legacy_fixed_by_id.get(slide_id))
        named_source = fixed if fixed is not None else original
        output_id = _strip_id_suffix(named_source.name) if named_source is not None else slide_id
        label_match = slide_id in labels
        fixed_done = state.get(slide_id) == "done" and fixed is not None
        selected = fixed if fixed_done else original
        expected_fixed = paths.rih_original_dir / f"{output_id}.tiff"
        status_path = selected if selected is not None else expected_fixed
        relative = _expected_relative(paths, status_path)
        details: dict[str, JsonScalar] = {
            "repair_state": state.get(slide_id, "not_selected"),
            "canonical_source": "fixed" if fixed_done else "original",
        }
        if require_label_match and not label_match:
            rows.append(_label_excluded(relative, status_path, output_id, "RIH", details))
            continue

        if fixed_done and selected is not None:
            if audit(selected):
                stat = selected.stat()
                rows.append(
                    _status(
                        relative,
                        selected,
                        output_id,
                        cohort="RIH",
                        status="waiting",
                        source_status="invalid_fixed_tiles",
                        reason="state=done fixed TIFF did not pass the standard-tile audit",
                        evidence="IFD0 TileWidth/TileLength",
                        label_match=label_match,
                        details=details,
                        size_bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                    )
                )
                continue
        elif original is None or slide_id in repair_named or audit(original):
            details["repair_detected_by"] = (
                "state_log" if slide_id in repair_named or original is None else "tile_header_audit"
            )
            rows.append(
                _status(
                    _expected_relative(paths, expected_fixed),
                    expected_fixed,
                    output_id,
                    cohort="RIH",
                    status="waiting",
                    source_status="waiting_repair",
                    reason="nonstandard original is waiting for a verified fixed TIFF",
                    evidence="repair state or IFD0 tile audit; original is never queued",
                    label_match=label_match,
                    details=details,
                )
            )
            continue

        assert selected is not None
        stat = selected.stat()
        mpp = _resolve_valid_mpp(resolve_mpp, selected)
        details.update(_mpp_details(mpp))
        if mpp is None:
            rows.append(
                _status(
                    relative,
                    selected,
                    output_id,
                    cohort="RIH",
                    status="waiting",
                    source_status="invalid_mpp",
                    reason="canonical RIH slide has no valid physical resolution metadata",
                    evidence="TIFF ImageDescription/resolution tags",
                    label_match=label_match,
                    details=details,
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
            continue
        source_status = "fixed_verified" if fixed_done else "original_standard"
        source_reason = (
            "standard-tile replacement has state=done and metadata MPP"
            if fixed_done
            else "original passed the standard-tile audit and has metadata MPP"
        )
        rows.append(
            _status(
                relative,
                selected,
                output_id,
                cohort="RIH",
                status="ready",
                source_status=source_status,
                reason=source_reason,
                evidence="repair state + IFD0 tile audit + TIFF metadata",
                label_match=label_match,
                details=details,
                mpp=mpp,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    return rows


def _status(
    wsi: str,
    path: Path,
    output_id: str,
    *,
    cohort: str,
    status: str,
    source_status: str,
    reason: str,
    evidence: str,
    label_match: bool,
    details: Mapping[str, JsonScalar],
    mpp: float | None = None,
    size_bytes: int | None = None,
    mtime_ns: int | None = None,
) -> InventoryStatus:
    return InventoryStatus(
        wsi=wsi,
        # Paths are constructed beneath the configured root.  Avoid
        # ``Path.resolve`` here: on the mounted slide drive it incurs several
        # metadata round trips for every one of ~2,000 ledger rows.
        path=path.absolute(),
        output_id=output_id,
        mpp=mpp,
        cohort=cohort,
        status=status,
        source_status=source_status,
        reason=reason,
        evidence=evidence,
        label_match=label_match,
        details=dict(details),
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
    )


def _label_excluded(
    wsi: str,
    path: Path,
    output_id: str,
    cohort: str,
    details: Mapping[str, JsonScalar],
) -> InventoryStatus:
    official_sheet = {
        "TCGA": "TCGA_COAD_READ_clinical_by_slide.xlsx:TCGA_Clinical",
        "SurGen": ("SurGen-1020_clinical_by_slide_paper_reconciled.xlsx:SurGen_Clinical"),
        "RIH": "RIH_clinical_reconciliation.xlsx:RIH_Clinical",
        "CPTAC": "CPTAC_COAD/metadata/master_labels_98.csv:slide_uid",
    }[cohort]
    return _status(
        wsi,
        path,
        output_id,
        cohort=cohort,
        status="excluded",
        source_status="label_excluded",
        reason="not present in the cohort's official Slide_ID column",
        evidence=official_sheet,
        label_match=False,
        details=details,
    )


def _ready_from_status(row: InventoryStatus) -> ReadySlide:
    if row.mpp is None or row.size_bytes is None or row.mtime_ns is None:
        raise InventoryError(f"Ready status lacks MPP/stat identity: {row.wsi}")
    return ReadySlide(
        wsi=row.wsi,
        path=row.path,
        output_id=row.output_id,
        mpp=row.mpp,
        cohort=row.cohort,
        source_status=row.source_status,
        evidence=row.evidence,
        label_match=row.label_match,
        details=row.details,
        size_bytes=row.size_bytes,
        mtime_ns=row.mtime_ns,
    )


def _resolve_valid_mpp(resolve_mpp: SlideMppResolver, path: Path) -> float | None:
    try:
        mpp = _positive_float_or_none(resolve_mpp(path))
    except (OSError, ValueError):
        return None
    if mpp is None or not _MIN_COLON_MPP <= mpp <= _MAX_COLON_MPP:
        return None
    return mpp


def _mpp_details(
    metadata_mpp: float | None,
    expected_mpp: float | None = None,
) -> dict[str, JsonScalar]:
    details: dict[str, JsonScalar] = {
        "metadata_mpp": metadata_mpp,
        "expected_mpp": expected_mpp,
    }
    if metadata_mpp is not None and expected_mpp is not None:
        difference = abs(metadata_mpp - expected_mpp)
        details["mpp_abs_difference"] = difference
        details["mpp_crosscheck"] = "match" if difference <= 0.002 else "different"
    elif expected_mpp is None:
        details["mpp_crosscheck"] = "metadata_only"
    else:
        details["mpp_crosscheck"] = "metadata_missing"
    return details


def _expected_relative(paths: InventoryPaths, path: Path) -> str:
    root = Path(paths.slide_root).absolute()
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise InventoryError(f"Slide target resolves outside slide_root: {path}") from exc
    if ".." in relative.parts:
        raise InventoryError(f"Slide target escapes slide_root: {path}")
    return relative.as_posix()


def _load_label_ids(
    path: Path,
    *,
    sheet_name: str,
    id_column: str,
) -> set[str]:
    if not path.is_file():
        raise InventoryError(f"Official clinical workbook does not exist: {path}")
    try:
        with ZipFile(path) as archive:
            worksheet = _worksheet_path(archive, sheet_name)
            shared_strings = _read_shared_strings(archive)
            root = ElementTree.fromstring(archive.read(worksheet))
    except (BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise InventoryError(f"Cannot parse official clinical workbook: {path}") from exc

    slide_column: str | None = None
    labels: set[str] = set()
    for row in root.findall(f".//{_SHEET_NS}row"):
        values = {
            _cell_column(cell): _cell_value(cell, shared_strings)
            for cell in row.findall(f"{_SHEET_NS}c")
        }
        if slide_column is None:
            matches = [
                column
                for column, value in values.items()
                if value.strip().casefold() == id_column.casefold()
            ]
            if matches:
                slide_column = matches[0]
            continue
        raw_value = values.get(slide_column, "").strip()
        if raw_value:
            slide_id = _canonical_slide_id(raw_value)
            if slide_id in labels:
                raise InventoryError(
                    f"Duplicate {id_column} {raw_value!r} in {sheet_name!r}: {path}"
                )
            labels.add(slide_id)

    if slide_column is None:
        raise InventoryError(f"Official sheet {sheet_name!r} has no {id_column!r} column: {path}")
    if not labels:
        raise InventoryError(
            f"Official sheet {sheet_name!r} has no populated {id_column!r} values: {path}"
        )
    return labels


def _worksheet_path(archive: ZipFile, sheet_name: str) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    matches = [
        sheet
        for sheet in workbook.findall(f".//{_SHEET_NS}sheet")
        if sheet.attrib.get("name") == sheet_name
    ]
    if len(matches) != 1 or _OFFICE_REL_ID not in matches[0].attrib:
        raise InventoryError(f"Workbook has no unique sheet named {sheet_name!r}")
    relationship_id = matches[0].attrib[_OFFICE_REL_ID]
    relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relationship in relationships.findall(f"{_REL_NS}Relationship"):
        if relationship.attrib.get("Id") != relationship_id:
            continue
        target = relationship.attrib.get("Target", "")
        if target.startswith("/"):
            return target.lstrip("/")
        return posixpath.normpath(posixpath.join("xl", target))
    raise InventoryError(f"Workbook relationship for sheet {sheet_name!r} is missing")


def _read_shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(node.text or "" for node in item.findall(f".//{_SHEET_NS}t")) for item in root]


def _cell_column(cell: ElementTree.Element) -> str:
    reference = cell.attrib.get("r", "")
    match = _CELL_COLUMN.match(reference)
    if match is None:
        raise InventoryError(f"Worksheet cell has invalid reference: {reference!r}")
    return match.group(1).upper()


def _cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    if cell.attrib.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(f".//{_SHEET_NS}t"))
    value = cell.find(f"{_SHEET_NS}v")
    if value is None or value.text is None:
        return ""
    if cell.attrib.get("t") == "s":
        try:
            return shared_strings[int(value.text)]
        except (IndexError, ValueError) as exc:
            raise InventoryError("Workbook contains an invalid shared-string index") from exc
    return value.text


def _read_gdc_manifest(path: Path) -> list[_GdcSlide]:
    if not path.is_file():
        raise InventoryError(f"GDC manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        hits = payload["data"]["hits"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise InventoryError(f"Cannot parse GDC manifest: {path}") from exc
    if not isinstance(hits, list):
        raise InventoryError(f"GDC manifest data.hits is not a list: {path}")

    slides: list[_GdcSlide] = []
    seen: set[str] = set()
    for index, hit in enumerate(hits):
        if not isinstance(hit, dict):
            raise InventoryError(f"Invalid GDC hit {index} in {path}")
        file_name = hit.get("file_name")
        file_size = hit.get("file_size")
        md5 = hit.get("md5sum", "")
        if not isinstance(file_name, str) or not file_name.casefold().endswith(".svs"):
            continue
        if isinstance(file_size, bool) or not isinstance(file_size, int) or file_size <= 0:
            raise InventoryError(f"Invalid GDC file_size for {file_name!r} in {path}")
        slide_id = _canonical_slide_id(file_name)
        if slide_id in seen:
            raise InventoryError(f"Duplicate GDC slide ID for {file_name!r} in {path}")
        seen.add(slide_id)
        cases = hit.get("cases")
        project = ""
        if isinstance(cases, list) and cases and isinstance(cases[0], dict):
            project_node = cases[0].get("project")
            if isinstance(project_node, dict):
                project = str(project_node.get("project_id", ""))
        slides.append(
            _GdcSlide(
                file_name=file_name,
                output_id=_strip_id_suffix(file_name),
                size_bytes=file_size,
                md5=str(md5),
                project=project,
            )
        )
    if not slides:
        raise InventoryError(f"GDC manifest contains no SVS records: {path}")
    return sorted(slides, key=lambda item: item.output_id.casefold())


def _read_cptac_manifest(path: Path) -> list[_CptacSlide]:
    if not path.is_file():
        raise InventoryError(f"CPTAC transfer manifest does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "decision",
                "include",
                "expected_svs_filename",
                "local_bytes",
                "source_md5",
                "transfer_group",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise InventoryError(f"CPTAC transfer manifest has an invalid schema: {path}")
            source_rows = list(reader)
    except OSError as exc:
        raise InventoryError(f"Cannot read CPTAC transfer manifest: {path}") from exc

    slides: list[_CptacSlide] = []
    seen: set[str] = set()
    for index, row in enumerate(source_rows, start=2):
        if row["decision"].strip().casefold() != "keep" or row["include"].strip().casefold() != "yes":
            continue
        file_name = row["expected_svs_filename"].strip()
        if not file_name.casefold().endswith(".svs") or Path(file_name).name != file_name:
            raise InventoryError(f"Invalid CPTAC filename on row {index}: {file_name!r}")
        try:
            size_bytes = int(row["local_bytes"])
        except ValueError as exc:
            raise InventoryError(f"Invalid CPTAC local_bytes on row {index}: {path}") from exc
        if size_bytes <= 0:
            raise InventoryError(f"Invalid CPTAC local_bytes on row {index}: {path}")
        slide_id = _canonical_slide_id(file_name)
        if slide_id in seen:
            raise InventoryError(f"Duplicate CPTAC slide ID for {file_name!r} in {path}")
        seen.add(slide_id)
        slides.append(
            _CptacSlide(
                file_name=file_name,
                output_id=_strip_id_suffix(file_name),
                size_bytes=size_bytes,
                md5=row["source_md5"].strip(),
                transfer_group=row["transfer_group"].strip(),
            )
        )
    if not slides:
        raise InventoryError(f"CPTAC transfer manifest contains no selected SVS records: {path}")
    return sorted(slides, key=lambda item: item.output_id.casefold())


def _read_cptac_label_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise InventoryError(f"CPTAC label manifest does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"slide_uid", "include"}.issubset(reader.fieldnames):
                raise InventoryError(f"CPTAC label manifest has an invalid schema: {path}")
            rows = list(reader)
    except OSError as exc:
        raise InventoryError(f"Cannot read CPTAC label manifest: {path}") from exc
    labels: set[str] = set()
    for index, row in enumerate(rows, start=2):
        if row["include"].strip().casefold() != "yes":
            continue
        raw_id = row["slide_uid"].strip().split(":", maxsplit=1)[-1]
        slide_id = _canonical_slide_id(raw_id)
        if not slide_id or slide_id in labels:
            raise InventoryError(f"Invalid or duplicate CPTAC slide_uid on row {index}: {path}")
        labels.add(slide_id)
    if not labels:
        raise InventoryError(f"CPTAC label manifest contains no included slides: {path}")
    return labels


def _read_jsonl_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InventoryError(f"Cannot read state log: {path}") from exc
    result: dict[str, str] = {}
    last_nonempty = max((index for index, line in enumerate(lines) if line.strip()), default=-1)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == last_nonempty:
                continue
            raise InventoryError(f"Malformed JSONL row {index + 1} in {path}") from exc
        if not isinstance(item, dict):
            raise InventoryError(f"Invalid JSONL row {index + 1} in {path}")
        file_name = item.get("file")
        status = item.get("status")
        if not isinstance(file_name, str) or not isinstance(status, str):
            raise InventoryError(f"State row {index + 1} lacks file/status in {path}")
        result[_canonical_slide_id(file_name)] = status.strip().casefold()
    return result


def _canonical_slide_id(value: str) -> str:
    name = value.strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    return _strip_id_suffix(name).casefold()


def _strip_id_suffix(name: str) -> str:
    folded = name.casefold()
    for suffix in _ID_SUFFIXES:
        if folded.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _files_with_suffix(directory: Path, suffix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.name.casefold().endswith(suffix)
        ),
        key=lambda path: path.name.casefold(),
    )


def _index_files(directory: Path, *, suffix: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for candidate in _files_with_suffix(directory, suffix):
        slide_id = _canonical_slide_id(candidate.name)
        previous = result.get(slide_id)
        if previous is not None:
            raise InventoryError(
                f"Duplicate slide ID in {directory}: {previous.name!r}, {candidate.name!r}"
            )
        result[slide_id] = candidate
    return result


def _status_sort_key(row: InventoryStatus) -> tuple[int, str, str]:
    return (
        _COHORT_ORDER.get(row.cohort, len(_COHORT_ORDER)),
        row.output_id.casefold(),
        row.wsi.casefold(),
    )


def _validate_ready(records: Iterable[ReadySlide]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        if not math.isfinite(record.mpp) or record.mpp <= 0:
            raise InventoryError(f"Invalid ready-slide MPP for {record.wsi}: {record.mpp}")
        key = record.output_id.casefold()
        if key in seen:
            raise InventoryError(
                "TRIDENT output ID collision between "
                f"{seen[key]!r} and {record.wsi!r}: {record.output_id!r}"
            )
        seen[key] = record.wsi


def _summarize(rows: Iterable[InventoryStatus]) -> dict[str, int]:
    rows_tuple = tuple(rows)
    counts = Counter((row.status, row.cohort) for row in rows_tuple)
    summary = {
        "total": len(rows_tuple),
        "ready": sum(row.status == "ready" for row in rows_tuple),
        "waiting": sum(row.status == "waiting" for row in rows_tuple),
        "excluded": sum(row.status == "excluded" for row in rows_tuple),
        "label_match": sum(row.label_match for row in rows_tuple),
    }
    cohorts = [*_COHORT_ORDER, *sorted({row.cohort for row in rows_tuple} - _COHORT_ORDER.keys())]
    for cohort in cohorts:
        key = cohort.casefold().replace(" ", "_")
        summary[f"ready_{key}"] = counts[("ready", cohort)]
        summary[f"waiting_{key}"] = counts[("waiting", cohort)]
        summary[f"excluded_{key}"] = counts[("excluded", cohort)]
    return summary


def _status_csv_row(row: InventoryStatus) -> dict[str, JsonScalar]:
    return {
        "wsi": row.wsi,
        "mpp": "" if row.mpp is None else format(row.mpp, ".17g"),
        "output_id": row.output_id,
        "cohort": row.cohort,
        "status": row.status,
        "source_status": row.source_status,
        "label_match": row.label_match,
        "reason": row.reason,
        "evidence": row.evidence,
        "size_bytes": row.size_bytes,
        "mtime_ns": row.mtime_ns,
        "details": json.dumps(row.details, sort_keys=True, separators=(",", ":")),
    }


def _atomic_write_csv(
    path: Path,
    *,
    fieldnames: tuple[str, ...],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _positive_float_or_none(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _cached_tiff_ifd0_tags(path: Path) -> dict[int, object]:
    try:
        stat = path.stat()
    except OSError:
        return {}
    return _cached_tiff_ifd0_tags_by_identity(
        str(path.resolve(strict=False)), stat.st_size, stat.st_mtime_ns
    )


@lru_cache(maxsize=8192)
def _cached_tiff_ifd0_tags_by_identity(
    path: str,
    _size_bytes: int,
    _mtime_ns: int,
) -> dict[int, object]:
    # Size and nanosecond mtime participate in the cache key so a producer's
    # atomic replacement cannot inherit metadata from an older file.
    return _read_tiff_ifd0_tags(Path(path), wanted={270, 282, 283, 296, 322, 323})


def _read_tiff_ifd0_tags(path: Path, *, wanted: set[int]) -> dict[int, object]:
    """Read selected first-IFD TIFF fields without decoding pixels."""

    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(16)
            if len(header) < 8:
                return {}
            if header[:2] == b"II":
                endian = "<"
            elif header[:2] == b"MM":
                endian = ">"
            else:
                return {}

            magic = struct.unpack(f"{endian}H", header[2:4])[0]
            if magic == 42:
                ifd_offset = struct.unpack(f"{endian}I", header[4:8])[0]
                count_format, count_size = "H", 2
                entry_size, value_size = 12, 4
                entry_count_format, offset_format = "I", "I"
            elif magic == 43:
                if len(header) < 16 or struct.unpack(f"{endian}H", header[4:6])[0] != 8:
                    return {}
                ifd_offset = struct.unpack(f"{endian}Q", header[8:16])[0]
                count_format, count_size = "Q", 8
                entry_size, value_size = 20, 8
                entry_count_format, offset_format = "Q", "Q"
            else:
                return {}
            if ifd_offset <= 0 or ifd_offset + count_size > file_size:
                return {}
            handle.seek(ifd_offset)
            raw_count = handle.read(count_size)
            if len(raw_count) != count_size:
                return {}
            entry_count = struct.unpack(f"{endian}{count_format}", raw_count)[0]
            if entry_count > 4096 or ifd_offset + count_size + entry_count * entry_size > file_size:
                return {}

            values: dict[int, object] = {}
            count_width = struct.calcsize(entry_count_format)
            type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 16: 8}
            for _ in range(entry_count):
                entry = handle.read(entry_size)
                if len(entry) != entry_size:
                    return {}
                tag, value_type = struct.unpack(f"{endian}HH", entry[:4])
                if tag not in wanted or value_type not in type_sizes:
                    continue
                item_count = struct.unpack(
                    f"{endian}{entry_count_format}", entry[4 : 4 + count_width]
                )[0]
                item_size = type_sizes[value_type]
                byte_count = item_count * item_size
                if item_count < 1 or byte_count > 1_048_576:
                    continue
                value_field = entry[4 + count_width : 4 + count_width + value_size]
                if byte_count <= value_size:
                    raw_value = value_field[:byte_count]
                else:
                    external_offset = struct.unpack(f"{endian}{offset_format}", value_field)[0]
                    if external_offset + byte_count > file_size:
                        continue
                    resume = handle.tell()
                    handle.seek(external_offset)
                    raw_value = handle.read(byte_count)
                    handle.seek(resume)
                if len(raw_value) != byte_count:
                    continue
                decoded = _decode_tiff_value(raw_value, endian, value_type, item_count)
                if decoded is not None:
                    values[tag] = decoded
            return values
    except (OSError, struct.error, OverflowError):
        return {}


def _decode_tiff_value(
    raw: bytes,
    endian: str,
    value_type: int,
    count: int,
) -> object | None:
    if value_type == 2:
        return raw.rstrip(b"\x00").decode("utf-8", errors="replace")
    if value_type == 5:
        numerator, denominator = struct.unpack(f"{endian}II", raw[:8])
        return None if denominator == 0 else numerator / denominator
    formats = {1: "B", 3: "H", 4: "I", 16: "Q"}
    value_format = formats.get(value_type)
    if value_format is None:
        return None
    item_size = struct.calcsize(value_format)
    values = [
        struct.unpack(f"{endian}{value_format}", raw[index * item_size : (index + 1) * item_size])[
            0
        ]
        for index in range(count)
    ]
    return values[0] if len(values) == 1 else values
