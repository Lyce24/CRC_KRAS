"""Tests for the live, multi-producer colon slide inventory."""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from oceanpath.extraction.inventory import (
    InventoryPaths,
    discover_ready_slides,
    read_slide_mpp,
    rih_original_requires_repair,
    write_inventory_csvs,
)


def _write_labels_xlsx(path: Path, *, sheet_name: str, slide_ids: list[str]) -> None:
    rows = [
        ("Slide_ID", "Patient_ID"),
        *((slide_id, f"patient-{index}") for index, slide_id in enumerate(slide_ids)),
    ]
    sheet_rows = []
    for row_number, values in enumerate(rows, start=1):
        cells = "".join(
            f'<c r="{column}{row_number}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
            for column, value in zip(("A", "B"), values, strict=True)
        )
        sheet_rows.append(f'<row r="{row_number}">{cells}</row>')
    worksheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows)}</sheetData></worksheet>"
    )
    workbook = (
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{sheet_name}" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    relationships = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_state(path: Path, rows: list[dict[str, str]], *, incomplete_tail: bool = False) -> None:
    text = "".join(f"{json.dumps(row)}\n" for row in rows)
    if incomplete_tail:
        text += '{"file": "producer-is-writing"'
    path.write_text(text, encoding="utf-8")


def _write_fake_slide(path: Path, size: int = 11) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _fixture_paths(tmp_path: Path) -> tuple[InventoryPaths, dict[str, float]]:
    slides = tmp_path / "slides"
    manifests = tmp_path / "manifests"
    for directory in (
        slides / "TCGA",
        slides / "SURGEN",
        slides / "SURGEN_fixed",
        slides / "rih",
        slides / "rih_fixed",
        slides / "rih_quarantine",
        manifests / "TCGA",
    ):
        directory.mkdir(parents=True)

    tcga_ready = "TCGA-AA-0001-01Z-00-DX1.12345678-AAAA-BBBB-CCCC-123456789ABC.svs"
    tcga_discarded = "TCGA-AA-0002-01Z-00-DX1.deadbeef-aaaa-bbbb-cccc-123456789abc.svs"
    tcga_pending = "TCGA-AA-0003-01Z-00-DX1.aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.svs"
    tcga_wrong_size = "TCGA-AA-0004-01Z-00-DX1.bbbbbbbb-cccc-dddd-eeee-ffffffffffff.svs"
    tcga_no_mpp = "TCGA-AA-0005-01Z-00-DX1.cccccccc-dddd-eeee-ffff-000000000000.svs"
    gdc_rows = [
        {
            "file_name": name,
            "file_size": size,
            "md5sum": f"md5-{index}",
            "cases": [{"project": {"project_id": "TCGA-COAD"}}],
        }
        for index, (name, size) in enumerate(
            [
                (tcga_ready, 17),
                (tcga_discarded, 19),
                (tcga_pending, 23),
                (tcga_wrong_size, 29),
                (tcga_no_mpp, 31),
            ]
        )
    ]
    (slides / "TCGA" / "gdc_manifest.json").write_text(
        json.dumps({"data": {"hits": gdc_rows}}), encoding="utf-8"
    )
    _write_fake_slide(slides / "TCGA" / tcga_ready, 17)
    _write_fake_slide(slides / "TCGA" / tcga_discarded, 19)
    _write_fake_slide(slides / "TCGA" / f"{tcga_pending}.part", 7)
    _write_fake_slide(slides / "TCGA" / tcga_wrong_size, 7)
    _write_fake_slide(slides / "TCGA" / tcga_no_mpp, 31)
    _write_labels_xlsx(
        manifests / "TCGA.xlsx",
        sheet_name="TCGA_Clinical",
        slide_ids=[
            tcga_ready,
            tcga_discarded,
            tcga_pending,
            tcga_wrong_size,
            tcga_no_mpp,
        ],
    )
    _write_labels_xlsx(
        manifests / "SurGen.xlsx",
        sheet_name="SurGen_Clinical",
        # The official workbook retains the source CZI suffix while the
        # canonical corrected slides are TIFFs.
        slide_ids=["SR_A.czi", "SR_B.czi", "SR_C.czi"],
    )
    _write_labels_xlsx(
        manifests / "RIH.xlsx",
        sheet_name="RIH_Clinical",
        slide_ids=["SL-1", "SL-301", "SL-302"],
    )

    for name in ("SR_A.tiff", "SR_B.tiff", "SR_C.tiff", "SR_UNLAB.tiff"):
        _write_fake_slide(slides / "SURGEN" / name)
    for name in ("SR_A.tiff", "SR_B.tiff", "SR_UNLAB.tiff"):
        _write_fake_slide(slides / "SURGEN_fixed" / name)
    _write_state(
        slides / "surgen.jsonl",
        [
            {"file": "SR_A.tiff", "status": "done"},
            {"file": "SR_B.tiff", "status": "failed"},
            {"file": "SR_C.tiff", "status": "done"},
            {"file": "SR_UNLAB.tiff", "status": "done"},
        ],
        incomplete_tail=True,
    )

    for name in ("SL-1.svs", "SL-301.svs", "SL-302.svs", "SL-400.svs"):
        _write_fake_slide(slides / "rih" / name)
    _write_fake_slide(slides / "rih_fixed" / "SL-301.tiff")
    _write_fake_slide(slides / "rih_quarantine" / "SHOULD-NOT-APPEAR.svs")
    _write_state(slides / "rih.jsonl", [{"file": "SL-301.svs", "status": "done"}])
    paths = InventoryPaths(
        slide_root=slides,
        manifest_root=manifests,
        tcga_clinical_workbook=manifests / "TCGA.xlsx",
        surgen_clinical_workbook=manifests / "SurGen.xlsx",
        rih_clinical_workbook=manifests / "RIH.xlsx",
        surgen_state=slides / "surgen.jsonl",
        rih_state=slides / "rih.jsonl",
        gdc_manifest=slides / "TCGA" / "gdc_manifest.json",
    )
    mpp_by_name = {
        tcga_ready: 0.2525,
        tcga_discarded: 0.252,
        tcga_no_mpp: 2.0,  # positive but outside the audited colon plausibility range
        "SR_A.tiff": 0.25,
        "SR_UNLAB.tiff": 0.25,
        "SL-1.svs": 0.5016,
        "SL-301.tiff": 0.189012,
        "SL-400.svs": 0.5016,
    }
    return paths, mpp_by_name


def test_discovers_all_valid_ready_slides_and_audits_waiting_inputs(tmp_path):
    paths, mpp_by_name = _fixture_paths(tmp_path)

    result = discover_ready_slides(
        paths,
        mpp_resolver=lambda path: mpp_by_name.get(path.name),
        rih_requires_repair=lambda path: path.name in {"SL-301.svs", "SL-302.svs"},
    )

    assert [record.output_id for record in result.ready] == [
        "SL-1",
        "SL-301",
        "SL-400",
        "SR_A",
        "SR_UNLAB",
        "TCGA-AA-0001-01Z-00-DX1.12345678-AAAA-BBBB-CCCC-123456789ABC",
        "TCGA-AA-0002-01Z-00-DX1.deadbeef-aaaa-bbbb-cccc-123456789abc",
    ]
    assert result.summary == {
        "total": 13,
        "ready": 7,
        "waiting": 6,
        "excluded": 0,
        "label_match": 11,
        "ready_rih": 3,
        "waiting_rih": 1,
        "excluded_rih": 0,
        "ready_surgen": 2,
        "waiting_surgen": 2,
        "excluded_surgen": 0,
        "ready_tcga": 2,
        "waiting_tcga": 3,
        "excluded_tcga": 0,
    }

    by_id = {row.output_id: row for row in result.status_rows}
    assert by_id["SL-301"].wsi == "rih_fixed/SL-301.tiff"
    assert by_id["SL-302"].source_status == "waiting_repair"
    assert by_id["SL-302"].details["repair_detected_by"] == "tile_header_audit"
    assert by_id["SL-400"].label_match is False
    assert by_id["SR_UNLAB"].status == "ready"
    assert by_id["SR_UNLAB"].label_match is False
    tcga = by_id["TCGA-AA-0001-01Z-00-DX1.12345678-AAAA-BBBB-CCCC-123456789ABC"]
    assert tcga.label_match is True  # dot before UUID was not mistaken for an extension
    assert tcga.details["metadata_mpp"] == pytest.approx(0.2525)
    assert tcga.details["expected_mpp"] is None
    assert all("part" not in row.wsi for row in result.status_rows)
    assert all("quarantine" not in row.wsi for row in result.status_rows)


def test_discovers_curated_cptac_slide_with_metadata_mpp(tmp_path):
    paths, mpp_by_name = _fixture_paths(tmp_path)
    file_name = "01CO005-example.svs"
    slide = _write_fake_slide(paths.cptac_dir / file_name, 37)
    _write_csv(
        paths.cptac_transfer_manifest,
        [
            "decision",
            "include",
            "expected_svs_filename",
            "local_bytes",
            "source_md5",
            "transfer_group",
        ],
        [
            {
                "decision": "KEEP",
                "include": "yes",
                "expected_svs_filename": file_name,
                "local_bytes": slide.stat().st_size,
                "source_md5": "example-md5",
                "transfer_group": "core",
            }
        ],
    )
    _write_csv(
        paths.cptac_label_manifest,
        ["slide_uid", "include"],
        [{"slide_uid": "CPTAC:01CO005-example", "include": "yes"}],
    )
    mpp_by_name[file_name] = 0.2501

    result = discover_ready_slides(
        paths,
        mpp_resolver=lambda path: mpp_by_name.get(path.name),
        rih_requires_repair=lambda path: path.name in {"SL-301.svs", "SL-302.svs"},
    )

    cptac = [row for row in result.ready if row.cohort == "CPTAC"]
    assert [(row.wsi, row.output_id, row.mpp) for row in cptac] == [
        ("CPTAC_COAD/01CO005-example.svs", "01CO005-example", 0.2501)
    ]
    assert result.summary["ready_cptac"] == 1

def test_discovers_state_verified_rih_tiff_after_final_inplace_handoff(tmp_path):
    paths, mpp_by_name = _fixture_paths(tmp_path)
    legacy = paths.rih_fixed_dir / "SL-301.tiff"
    inplace = paths.rih_original_dir / "SL-301.tiff"
    legacy.replace(inplace)
    (paths.rih_original_dir / "SL-301.svs").unlink()

    result = discover_ready_slides(
        paths,
        mpp_resolver=lambda path: mpp_by_name.get(path.name),
        rih_requires_repair=lambda path: path.name in {"SL-301.svs", "SL-302.svs"},
    )

    record = next(item for item in result.ready if item.output_id == "SL-301")
    assert record.wsi == "rih/SL-301.tiff"
    assert record.source_status == "fixed_verified"


def test_label_filter_is_opt_in_and_keeps_full_status_ledger(tmp_path):
    paths, mpp_by_name = _fixture_paths(tmp_path)

    result = discover_ready_slides(
        paths,
        require_label_match=True,
        mpp_resolver=lambda path: mpp_by_name.get(path.name),
        rih_requires_repair=lambda path: path.name in {"SL-301.svs", "SL-302.svs"},
    )

    assert {record.output_id for record in result.ready} == {
        "SL-1",
        "SL-301",
        "SR_A",
        "TCGA-AA-0001-01Z-00-DX1.12345678-AAAA-BBBB-CCCC-123456789ABC",
        "TCGA-AA-0002-01Z-00-DX1.deadbeef-aaaa-bbbb-cccc-123456789abc",
    }
    assert result.summary["total"] == 13
    assert result.summary["excluded"] == 2
    label_excluded = {
        row.output_id for row in result.status_rows if row.source_status == "label_excluded"
    }
    assert {"SL-400", "SR_UNLAB"} <= label_excluded
    assert len(label_excluded) == 2


def test_writes_deterministic_atomic_mpp_and_status_csvs(tmp_path):
    paths, mpp_by_name = _fixture_paths(tmp_path)
    result = discover_ready_slides(
        paths,
        mpp_resolver=lambda path: mpp_by_name.get(path.name),
        rih_requires_repair=lambda path: path.name in {"SL-301.svs", "SL-302.svs"},
    )
    mpp_path = tmp_path / "published" / "ready.csv"
    status_path = tmp_path / "published" / "status.csv"
    mpp_path.parent.mkdir()
    mpp_path.write_text("stale\n", encoding="utf-8")
    status_path.write_text("stale\n", encoding="utf-8")

    returned = write_inventory_csvs(result, mpp_path=mpp_path, status_path=status_path)

    assert returned == (mpp_path, status_path)
    with mpp_path.open(newline="", encoding="utf-8") as handle:
        mpp_rows = list(csv.DictReader(handle))
    with status_path.open(newline="", encoding="utf-8") as handle:
        status_rows = list(csv.DictReader(handle))
    assert list(mpp_rows[0]) == ["wsi", "mpp"]
    assert len(mpp_rows) == result.summary["ready"]
    assert [row["wsi"] for row in mpp_rows] == [record.wsi for record in result.ready]
    assert len(status_rows) == result.summary["total"]
    assert json.loads(status_rows[0]["details"])
    assert not list(mpp_path.parent.glob("*.tmp"))


def _write_classic_tiff(
    path: Path,
    *,
    tile_shape: tuple[int, int],
    pixels_per_centimeter: int,
    description_mpp: float | None = None,
) -> None:
    description = (
        f"Aperio Image Library|AppMag = 40|MPP = {description_mpp}|".encode() + b"\x00"
        if description_mpp is not None
        else None
    )
    tags: list[tuple[int, int, int, bytes | int]] = [
        (282, 5, 1, struct.pack("<II", pixels_per_centimeter, 1)),
        (283, 5, 1, struct.pack("<II", pixels_per_centimeter, 1)),
        (296, 3, 1, struct.pack("<H", 3)),
        (322, 4, 1, struct.pack("<I", tile_shape[0])),
        (323, 4, 1, struct.pack("<I", tile_shape[1])),
    ]
    if description is not None:
        tags.append((270, 2, len(description), description))
    tags.sort()

    ifd_offset = 8
    external_offset = ifd_offset + 2 + len(tags) * 12 + 4
    entries = bytearray()
    external = bytearray()
    for tag, value_type, count, value in tags:
        assert isinstance(value, bytes)
        entries.extend(struct.pack("<HHI", tag, value_type, count))
        if len(value) <= 4:
            entries.extend(value.ljust(4, b"\x00"))
        else:
            entries.extend(struct.pack("<I", external_offset + len(external)))
            external.extend(value)
    payload = (
        b"II"
        + struct.pack("<H", 42)
        + struct.pack("<I", ifd_offset)
        + struct.pack("<H", len(tags))
        + bytes(entries)
        + struct.pack("<I", 0)
        + bytes(external)
    )
    path.write_bytes(payload)


def test_header_only_mpp_and_tile_audit(tmp_path):
    standard = tmp_path / "standard.tiff"
    nonstandard = tmp_path / "nonstandard.svs"
    description_wins = tmp_path / "aperio.svs"
    _write_classic_tiff(
        standard,
        tile_shape=(256, 256),
        pixels_per_centimeter=40_000,
    )
    _write_classic_tiff(
        nonstandard,
        tile_shape=(620, 740),
        pixels_per_centimeter=50_000,
    )
    _write_classic_tiff(
        description_wins,
        tile_shape=(240, 240),
        pixels_per_centimeter=20_000,
        description_mpp=0.2525,
    )

    assert read_slide_mpp(standard) == pytest.approx(0.25)
    assert read_slide_mpp(nonstandard) == pytest.approx(0.2)
    assert read_slide_mpp(description_wins) == pytest.approx(0.2525)
    assert rih_original_requires_repair(standard) is False
    assert rih_original_requires_repair(nonstandard) is True
    assert rih_original_requires_repair(tmp_path / "missing.svs") is True


def test_from_roots_matches_live_colon_layout(tmp_path):
    paths = InventoryPaths.from_roots(tmp_path / "slides", tmp_path / "manifests")

    assert paths.tcga_dir == tmp_path / "slides" / "TCGA"
    assert paths.surgen_fixed_dir == tmp_path / "slides" / "SURGEN_fixed"
    assert paths.rih_original_dir == tmp_path / "slides" / "rih"
    assert paths.gdc_manifest == tmp_path / "slides" / "TCGA" / "gdc_manifest.json"
    assert paths.tcga_clinical_workbook == (
        tmp_path / "manifests" / "TCGA_COAD_READ_clinical_by_slide.xlsx"
    )
    assert paths.surgen_clinical_workbook == (
        tmp_path / "manifests" / "SurGen-1020_clinical_by_slide_paper_reconciled.xlsx"
    )
    assert paths.rih_clinical_workbook == (
        tmp_path / "manifests" / "RIH_clinical_reconciliation.xlsx"
    )


def test_from_roots_never_uses_archived_legacy_manifests(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    legacy = manifests / "legacy"
    (legacy / "TCGA").mkdir(parents=True)
    for relative in (
        "combined_labels_with_patient_id.xlsx",
        "TCGA/slides_with_mpp_filtered.csv",
        "TCGA/sl.csv",
        "TCGA/discard.tsv",
    ):
        path = legacy / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    paths = InventoryPaths.from_roots(tmp_path / "slides", manifests)

    assert paths.tcga_clinical_workbook.parent == manifests
    assert paths.surgen_clinical_workbook.parent == manifests
    assert paths.rih_clinical_workbook.parent == manifests
    assert "legacy" not in paths.tcga_clinical_workbook.parts


def test_surgen_paths_follow_verified_final_directory_swap(tmp_path: Path) -> None:
    slides = tmp_path / "slides"
    (slides / "SURGEN").mkdir(parents=True)
    (slides / "SURGEN_rb_swapped_original").mkdir()
    paths = InventoryPaths.from_roots(slides, tmp_path / "manifests")

    assert paths.surgen_fixed_dir == slides / "SURGEN"
    assert paths.surgen_original_dir == slides / "SURGEN_rb_swapped_original"


def test_surgen_paths_prefer_active_conversion_output(tmp_path: Path) -> None:
    slides = tmp_path / "slides"
    (slides / "SURGEN").mkdir(parents=True)
    (slides / "SURGEN_fixed").mkdir()
    paths = InventoryPaths.from_roots(slides, tmp_path / "manifests")

    assert paths.surgen_fixed_dir == slides / "SURGEN_fixed"
    assert paths.surgen_original_dir == slides / "SURGEN"
