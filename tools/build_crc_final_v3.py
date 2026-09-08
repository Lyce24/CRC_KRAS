#!/usr/bin/env python3
"""Build CRC label master v3 by attaching audited slide/inventory fields.

The v3 table preserves every v2 label row.  ``used`` is deliberately stricter
than ``available``: a physical WSI may exist but remain unusable when its MPP
cannot be proven from slide metadata.  Task-specific ``used_kras`` and
``used_ras`` columns additionally require a known binary endpoint.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import uuid
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

MANIFEST_ROOT = Path("/mnt/d/YC.Liu/manifests/colon")
SLIDE_ROOT = Path("/mnt/d/YC.Liu/slides/colon")
V2_CSV = MANIFEST_ROOT / "crc_final_v2.csv"
V2_XLSX = MANIFEST_ROOT / "crc_final_v2.xlsx"
INVENTORY_CSV = MANIFEST_ROOT / "colon_ready_inventory.csv"
V3_CSV = MANIFEST_ROOT / "crc_final_v3.csv"
V3_XLSX = MANIFEST_ROOT / "crc_final_v3.xlsx"

SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = f"{{{SHEET_NS}}}"
ElementTree.register_namespace("x", SHEET_NS)

SLIDE_COLUMNS = [
    "wsi",
    "output_id",
    "image_filename",
    "image_format",
    "available",
    "used",
    "used_kras",
    "used_ras",
    "mpp",
    "mpp_valid",
    "mpp_source",
    "qc_slides",
    "inventory_status",
    "inventory_source_status",
    "inventory_reason",
    "inventory_evidence",
    "slide_size_bytes",
    "slide_mtime_ns",
]

NUMERIC_SLIDE_COLUMNS = {"mpp", "slide_size_bytes", "slide_mtime_ns"}

CODEBOOK_ROWS = [
    (
        "wsi / output_id",
        "Canonical relative WSI path and collision-checked feature output identity",
        "Resolved from colon_ready_inventory.csv; wsi is relative to the colon slide root",
    ),
    (
        "image_filename / image_format",
        "Filename and actual model-input container format",
        ".tiff for corrected SurGen/RIH inputs where applicable; .svs otherwise",
    ),
    (
        "available",
        "Whether the canonical physical WSI currently exists",
        "yes | no; existence alone does not make a slide usable",
    ),
    (
        "used",
        "General label/slide intersection flag",
        "yes only when include=yes, inventory_status=ready, and metadata MPP is valid",
    ),
    (
        "used_kras / used_ras",
        "Binary-task usability flags",
        "used=yes plus the corresponding endpoint is mutant or wild_type",
    ),
    (
        "mpp / mpp_valid / mpp_source",
        "Physical pixel size and its validation provenance",
        "MPP is blank unless valid; source is slide_metadata, never a cohort default",
    ),
    (
        "qc_slides",
        "Slide-level technical QC outcome",
        "pass for ready inputs; otherwise the inventory source status such as invalid_mpp",
    ),
    (
        "inventory_status / inventory_source_status / inventory_reason / inventory_evidence",
        "Audited readiness decision and supporting provenance",
        "Copied from colon_ready_inventory.csv",
    ),
    (
        "slide_size_bytes / slide_mtime_ns",
        "Physical file identity recorded during inventory validation",
        "Integer byte size and nanosecond modification timestamp",
    ),
]

MODELLING_ROWS = [
    (
        "Availability is not usability",
        "available=yes means the physical file exists. used=yes additionally requires "
        "include=yes, a ready inventory decision, and valid metadata-derived MPP.",
    ),
    (
        "Use the task-specific flags",
        "used_kras and used_ras exclude unknown endpoints. The general used flag retains "
        "slides that are technically usable for other CRC endpoints.",
    ),
    (
        "Slide QC and label QC are separate",
        "qc_slides records image/inventory QC. qc_flags and qc_note retain molecular and "
        "clinical-label review flags; neither field should overwrite the other.",
    ),
]


def _slide_id(value: str) -> str:
    return value.strip().split(":", maxsplit=1)[-1]


def _inventory_slide_id(cohort: str, output_id: str) -> str:
    value = output_id.strip()
    return value.split(".", maxsplit=1)[0] if cohort.strip().casefold() == "tcga" else value


def _join_key(cohort: str, slide_id: str) -> str:
    return f"{cohort.strip().casefold()}::{slide_id.strip().casefold()}"


def build_table() -> pd.DataFrame:
    labels = pd.read_csv(V2_CSV, dtype=str, keep_default_na=False)
    inventory = pd.read_csv(INVENTORY_CSV, dtype=str, keep_default_na=False)
    original_columns = list(labels.columns)

    labels["_join_key"] = [
        _join_key(cohort, _slide_id(slide_uid))
        for cohort, slide_uid in zip(labels["cohort"], labels["slide_uid"], strict=True)
    ]
    inventory["_join_key"] = [
        _join_key(cohort, _inventory_slide_id(cohort, output_id))
        for cohort, output_id in zip(inventory["cohort"], inventory["output_id"], strict=True)
    ]
    if labels["_join_key"].duplicated().any():
        raise ValueError("crc_final_v2 contains duplicate cohort-aware slide identities")
    if inventory["_join_key"].duplicated().any():
        duplicates = inventory.loc[inventory["_join_key"].duplicated(False), "wsi"].tolist()
        raise ValueError(f"Inventory contains ambiguous slide identities: {duplicates[:10]}")

    source_columns = [
        "_join_key",
        "wsi",
        "output_id",
        "mpp",
        "status",
        "source_status",
        "reason",
        "evidence",
        "size_bytes",
        "mtime_ns",
    ]
    merged = labels.merge(
        inventory[source_columns],
        on="_join_key",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        missing = merged.loc[merged["_merge"] != "both", "slide_uid"].tolist()
        raise ValueError(f"Label rows missing from the audited inventory: {missing[:10]}")

    physical_paths = [SLIDE_ROOT / value for value in merged["wsi"]]
    merged["image_filename"] = [path.name for path in physical_paths]
    merged["image_format"] = [path.suffix.casefold() for path in physical_paths]
    merged["available"] = ["yes" if path.is_file() else "no" for path in physical_paths]
    mpp_valid = (merged["status"] == "ready") & (merged["mpp"].str.strip() != "")
    used = (merged["include"].str.casefold() == "yes") & mpp_valid
    merged["used"] = used.map({True: "yes", False: "no"})
    merged["used_kras"] = (used & merged["kras"].isin(["mutant", "wild_type"])).map(
        {True: "yes", False: "no"}
    )
    merged["used_ras"] = (used & merged["ras"].isin(["mutant", "wild_type"])).map(
        {True: "yes", False: "no"}
    )
    merged["mpp_valid"] = mpp_valid.map({True: "yes", False: "no"})
    merged["mpp_source"] = mpp_valid.map({True: "slide_metadata", False: ""})
    merged["qc_slides"] = merged["source_status"].where(~mpp_valid, "pass")
    merged["inventory_status"] = merged.pop("status")
    merged["inventory_source_status"] = merged.pop("source_status")
    merged["inventory_reason"] = merged.pop("reason")
    merged["inventory_evidence"] = merged.pop("evidence")
    merged["slide_size_bytes"] = merged.pop("size_bytes")
    merged["slide_mtime_ns"] = merged.pop("mtime_ns")

    if (merged["available"] != "yes").any():
        missing_files = merged.loc[merged["available"] != "yes", "wsi"].tolist()
        raise ValueError(f"Inventory paths are not physically available: {missing_files[:10]}")
    if int((merged["used"] == "yes").sum()) != 2003:
        raise ValueError("Expected exactly 2,003 generally usable slides")
    if int((merged["used_kras"] == "yes").sum()) != 1828:
        raise ValueError("Expected exactly 1,828 KRAS-usable slides")
    if int((merged["used_ras"] == "yes").sum()) != 1826:
        raise ValueError("Expected exactly 1,826 RAS-usable slides")
    if set(merged["qc_slides"]) != {"pass", "invalid_mpp"}:
        raise ValueError(f"Unexpected slide QC values: {set(merged['qc_slides'])}")

    return merged[[*original_columns, *SLIDE_COLUMNS]]


def _column_name(index: int) -> str:
    result = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _append_cell(row: ElementTree.Element, reference: str, value: object, *, numeric: bool) -> None:
    cell = ElementTree.SubElement(row, f"{NS}c", {"r": reference})
    text = "" if value is None else str(value)
    if not text:
        return
    if numeric:
        cell.set("t", "n")
        ElementTree.SubElement(cell, f"{NS}v").text = text
        return
    cell.set("t", "inlineStr")
    inline = ElementTree.SubElement(cell, f"{NS}is")
    node = ElementTree.SubElement(inline, f"{NS}t")
    node.set(f"{{{XML_NS}}}space", "preserve")
    node.text = text


def _append_sheet_rows(
    xml_bytes: bytes,
    rows_to_add: list[tuple[str, ...]],
) -> tuple[bytes, int]:
    root = ElementTree.fromstring(xml_bytes)
    sheet_data = root.find(f"{NS}sheetData")
    if sheet_data is None:
        raise ValueError("Workbook sheet has no sheetData")
    existing_rows = list(sheet_data.findall(f"{NS}row"))
    last_row = max(int(row.attrib["r"]) for row in existing_rows)
    for offset, values in enumerate(rows_to_add, start=1):
        row_number = last_row + offset
        row = ElementTree.SubElement(sheet_data, f"{NS}row", {"r": str(row_number)})
        for column_index, value in enumerate(values, start=1):
            _append_cell(row, f"{_column_name(column_index)}{row_number}", value, numeric=False)
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True), last_row + len(
        rows_to_add
    )


def _extend_labels_sheet(xml_bytes: bytes, table: pd.DataFrame) -> bytes:
    root = ElementTree.fromstring(xml_bytes)
    sheet_data = root.find(f"{NS}sheetData")
    if sheet_data is None:
        raise ValueError("Labels sheet has no sheetData")
    rows = list(sheet_data.findall(f"{NS}row"))
    if len(rows) != len(table) + 1:
        raise ValueError("v2 XLSX and v2 CSV row counts differ")
    start_index = len(table.columns) - len(SLIDE_COLUMNS) + 1
    for row_index, (xml_row, values) in enumerate(
        zip(
            rows,
            [SLIDE_COLUMNS, *table[SLIDE_COLUMNS].itertuples(index=False, name=None)],
            strict=True,
        ),
        start=1,
    ):
        for offset, (column, value) in enumerate(zip(SLIDE_COLUMNS, values, strict=True)):
            column_index = start_index + offset
            numeric = row_index > 1 and column in NUMERIC_SLIDE_COLUMNS and str(value) != ""
            _append_cell(
                xml_row,
                f"{_column_name(column_index)}{row_index}",
                value,
                numeric=numeric,
            )
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _extend_table_columns(xml_bytes: bytes, columns: list[str], last_row: int) -> bytes:
    root = ElementTree.fromstring(xml_bytes)
    table_columns = root.find(f"{NS}tableColumns")
    if table_columns is None:
        raise ValueError("Workbook table has no tableColumns")
    existing = list(table_columns.findall(f"{NS}tableColumn"))
    for offset, column in enumerate(columns, start=1):
        ElementTree.SubElement(
            table_columns,
            f"{NS}tableColumn",
            {"id": str(len(existing) + offset), "name": column},
        )
    count = len(existing) + len(columns)
    table_columns.set("count", str(count))
    root.set("ref", f"A1:{_column_name(count)}{last_row}")
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _extend_table_rows(xml_bytes: bytes, last_row: int) -> bytes:
    root = ElementTree.fromstring(xml_bytes)
    reference = root.attrib["ref"]
    end_column = reference.split(":", maxsplit=1)[1].rstrip("0123456789")
    root.set("ref", f"A1:{end_column}{last_row}")
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def write_xlsx(table: pd.DataFrame, destination: Path) -> None:
    replacements: dict[str, bytes] = {}
    with ZipFile(V2_XLSX, "r") as source:
        replacements["xl/worksheets/sheet1.xml"] = _extend_labels_sheet(
            source.read("xl/worksheets/sheet1.xml"), table
        )
        replacements["xl/tables/table1.xml"] = _extend_table_columns(
            source.read("xl/tables/table1.xml"), SLIDE_COLUMNS, len(table) + 1
        )
        codebook, codebook_last = _append_sheet_rows(
            source.read("xl/worksheets/sheet2.xml"), CODEBOOK_ROWS
        )
        replacements["xl/worksheets/sheet2.xml"] = codebook
        replacements["xl/tables/table2.xml"] = _extend_table_rows(
            source.read("xl/tables/table2.xml"), codebook_last
        )
        notes, notes_last = _append_sheet_rows(
            source.read("xl/worksheets/sheet3.xml"), MODELLING_ROWS
        )
        replacements["xl/worksheets/sheet3.xml"] = notes
        replacements["xl/tables/table3.xml"] = _extend_table_rows(
            source.read("xl/tables/table3.xml"), notes_last
        )

        with ZipFile(destination, "w", compression=ZIP_DEFLATED) as target:
            for item in source.infolist():
                target.writestr(item, replacements.get(item.filename, source.read(item.filename)))


def atomic_csv(table: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        table.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_xlsx(table: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_xlsx(table, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_outputs(csv_path: Path, xlsx_path: Path) -> None:
    output = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if len(output) != 2028 or list(output.columns[-len(SLIDE_COLUMNS) :]) != SLIDE_COLUMNS:
        raise ValueError("Generated v3 CSV has the wrong shape or columns")
    with ZipFile(xlsx_path) as archive:
        labels = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = labels.findall(f".//{NS}row")
        cells = rows[0].findall(f"{NS}c")
        table = ElementTree.fromstring(archive.read("xl/tables/table1.xml"))
        if len(rows) != 2029 or len(cells) != len(output.columns):
            raise ValueError("Generated v3 XLSX labels sheet has the wrong shape")
        if table.attrib["ref"] != f"A1:{_column_name(len(output.columns))}2029":
            raise ValueError("Generated v3 XLSX table range is incorrect")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write crc_final_v3.csv/.xlsx")
    args = parser.parse_args()
    table = build_table()
    counts = {
        "rows": len(table),
        "available": int((table["available"] == "yes").sum()),
        "used": int((table["used"] == "yes").sum()),
        "used_kras": int((table["used_kras"] == "yes").sum()),
        "used_ras": int((table["used_ras"] == "yes").sum()),
        "slide_qc": table["qc_slides"].value_counts().to_dict(),
    }
    print(counts)
    if not args.apply:
        with tempfile.TemporaryDirectory(prefix="crc-final-v3-") as directory:
            csv_path = Path(directory) / V3_CSV.name
            xlsx_path = Path(directory) / V3_XLSX.name
            table.to_csv(csv_path, index=False)
            write_xlsx(table, xlsx_path)
            verify_outputs(csv_path, xlsx_path)
        print("Dry-run verification passed; use --apply to write outputs")
        return
    if V3_CSV.exists() or V3_XLSX.exists():
        raise FileExistsError("Refusing to overwrite an existing crc_final_v3 output")
    atomic_csv(table, V3_CSV)
    atomic_xlsx(table, V3_XLSX)
    verify_outputs(V3_CSV, V3_XLSX)
    print(f"Wrote {V3_CSV}")
    print(f"Wrote {V3_XLSX}")


if __name__ == "__main__":
    main()
