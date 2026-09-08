"""Focused tests for the TRIDENT slide-selection CSV contract."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from oceanpath.extraction.slide_records import (
    SlideRecord,
    SlideRecordError,
    load_slide_records,
    shard_slide_records,
    write_slide_records_csv,
)


def _touch(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_loads_trident_csv_with_nested_paths_and_mpp(tmp_path):
    slides = tmp_path / "slides"
    _touch(slides, "TCGA/case.with.dots.SVS")
    _touch(slides, "RIH/slide,comma.svs")
    custom = _write_csv(
        tmp_path / "slides.csv",
        [
            {"wsi": r"RIH\slide,comma.svs", "mpp": "0.502"},
            {"wsi": "./TCGA/case.with.dots.SVS", "mpp": "0.2325"},
        ],
        ["wsi", "mpp"],
    )

    records = load_slide_records(
        slides,
        custom_list_path=custom,
        wsi_ext=[".svs"],
    )

    assert [(record.wsi, record.output_id, record.mpp) for record in records] == [
        ("RIH/slide,comma.svs", "slide,comma", 0.502),
        ("TCGA/case.with.dots.SVS", "case.with.dots", 0.2325),
    ]
    assert all(record.path.is_absolute() for record in records)


def test_requires_wsi_header(tmp_path):
    slides = tmp_path / "slides"
    slides.mkdir()
    custom = _write_csv(
        tmp_path / "slides.csv",
        [{"filename": "case.svs"}],
        ["filename"],
    )

    with pytest.raises(SlideRecordError, match="'wsi' column"):
        load_slide_records(slides, custom_list_path=custom, wsi_ext=[".svs"])


@pytest.mark.parametrize("mpp", ["", "0", "-0.1", "nan", "inf", "not-a-number"])
def test_rejects_missing_nonfinite_or_nonpositive_mpp(tmp_path, mpp):
    slides = tmp_path / "slides"
    _touch(slides, "case.svs")
    custom = _write_csv(
        tmp_path / "slides.csv",
        [{"wsi": "case.svs", "mpp": mpp}],
        ["wsi", "mpp"],
    )

    with pytest.raises(SlideRecordError, match="[Mm][Pp][Pp]"):
        load_slide_records(slides, custom_list_path=custom, wsi_ext=[".svs"])


def test_rejects_partial_mpp_column(tmp_path):
    slides = tmp_path / "slides"
    _touch(slides, "first.svs")
    _touch(slides, "second.svs")
    custom = _write_csv(
        tmp_path / "slides.csv",
        [
            {"wsi": "first.svs", "mpp": "0.25"},
            {"wsi": "second.svs", "mpp": ""},
        ],
        ["wsi", "mpp"],
    )

    with pytest.raises(SlideRecordError, match="populate every row"):
        load_slide_records(slides, custom_list_path=custom, wsi_ext=[".svs"])


@pytest.mark.parametrize("wsi", ["../outside.svs", "/tmp/outside.svs", r"C:\outside.svs"])
def test_rejects_paths_outside_slide_root(tmp_path, wsi):
    slides = tmp_path / "slides"
    slides.mkdir()
    custom = _write_csv(
        tmp_path / "slides.csv",
        [{"wsi": wsi}],
        ["wsi"],
    )

    with pytest.raises(SlideRecordError, match="relative to the slide directory"):
        load_slide_records(slides, custom_list_path=custom, wsi_ext=[".svs"])


def test_rejects_symlink_that_resolves_outside_slide_root(tmp_path):
    slides = tmp_path / "slides"
    slides.mkdir()
    outside = tmp_path / "outside.svs"
    outside.touch()
    (slides / "linked.svs").symlink_to(outside)
    custom = _write_csv(
        tmp_path / "slides.csv",
        [{"wsi": "linked.svs"}],
        ["wsi"],
    )

    with pytest.raises(SlideRecordError, match="resolves outside"):
        load_slide_records(slides, custom_list_path=custom, wsi_ext=[".svs"])


def test_rejects_nested_output_id_collisions(tmp_path):
    slides = tmp_path / "slides"
    _touch(slides, "TCGA/shared.svs")
    _touch(slides, "RIH/SHARED.tiff")

    with pytest.raises(SlideRecordError, match="flattens nested outputs"):
        load_slide_records(
            slides,
            wsi_ext=[".svs", ".tiff"],
            search_nested=True,
        )


def test_scan_is_deterministic_and_respects_search_nested(tmp_path):
    slides = tmp_path / "slides"
    _touch(slides, "zeta.SVS")
    _touch(slides, "alpha.svs")
    _touch(slides, "nested/beta.svs")
    _touch(slides, "ignore.txt")

    top_level = load_slide_records(slides, wsi_ext=[".svs"])
    nested = load_slide_records(slides, wsi_ext=[".svs"], search_nested=True)

    assert [record.wsi for record in top_level] == ["alpha.svs", "zeta.SVS"]
    assert [record.wsi for record in nested] == [
        "alpha.svs",
        "nested/beta.svs",
        "zeta.SVS",
    ]


def test_custom_csv_and_search_nested_are_mutually_exclusive(tmp_path):
    slides = tmp_path / "slides"
    _touch(slides, "case.svs")
    custom = _write_csv(
        tmp_path / "slides.csv",
        [{"wsi": "case.svs"}],
        ["wsi"],
    )

    with pytest.raises(SlideRecordError, match="CSV already defines"):
        load_slide_records(
            slides,
            custom_list_path=custom,
            wsi_ext=[".svs"],
            search_nested=True,
        )


def test_shards_are_stable_disjoint_and_cover_all_records(tmp_path):
    slides = tmp_path / "slides"
    records = []
    for name in ["delta.svs", "alpha.svs", "charlie.svs", "bravo.svs", "echo.svs"]:
        path = _touch(slides, name).resolve()
        records.append(SlideRecord(wsi=name, path=path, output_id=path.stem, mpp=0.25))

    shards = [
        shard_slide_records(list(reversed(records)), shard_id=shard_id, total_shards=3)
        for shard_id in range(3)
    ]
    shard_ids = [{record.output_id for record in shard} for shard in shards]

    assert not (shard_ids[0] & shard_ids[1])
    assert not (shard_ids[0] & shard_ids[2])
    assert not (shard_ids[1] & shard_ids[2])
    assert set.union(*shard_ids) == {record.output_id for record in records}
    assert [record.wsi for record in shards[0]] == ["alpha.svs", "delta.svs"]


@pytest.mark.parametrize(
    ("shard_id", "total_shards"),
    [(-1, 2), (2, 2), (0, 0), (True, 2), (0, True)],
)
def test_rejects_invalid_shard_coordinates(shard_id, total_shards):
    with pytest.raises(SlideRecordError):
        shard_slide_records([], shard_id=shard_id, total_shards=total_shards)


def test_subset_writer_is_deterministic_and_preserves_mpp(tmp_path):
    slides = tmp_path / "slides"
    beta = _touch(slides, "beta,quoted.svs").resolve()
    alpha = _touch(slides, "alpha.svs").resolve()
    records = [
        SlideRecord("beta,quoted.svs", beta, "beta,quoted", 0.502),
        SlideRecord("alpha.svs", alpha, "alpha", 0.189),
    ]

    output = write_slide_records_csv(records, tmp_path / "subset.csv")

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"wsi": "alpha.svs", "mpp": "0.189"},
        {"wsi": "beta,quoted.svs", "mpp": "0.502"},
    ]
    assert not (tmp_path / ".subset.csv.tmp").exists()


def test_empty_mpp_aware_shard_keeps_csv_schema(tmp_path):
    output = write_slide_records_csv([], tmp_path / "empty.csv", include_mpp=True)

    assert output.read_text(encoding="utf-8") == "wsi,mpp\n"


def test_writer_refuses_to_drop_or_partially_write_mpp(tmp_path):
    slides = tmp_path / "slides"
    first = _touch(slides, "first.svs").resolve()
    second = _touch(slides, "second.svs").resolve()
    populated = SlideRecord("first.svs", first, "first", 0.25)
    missing = SlideRecord("second.svs", second, "second", None)

    with pytest.raises(SlideRecordError, match="partially populated"):
        write_slide_records_csv([populated, missing], tmp_path / "partial.csv")
    with pytest.raises(SlideRecordError, match="discard"):
        write_slide_records_csv([populated], tmp_path / "dropped.csv", include_mpp=False)
