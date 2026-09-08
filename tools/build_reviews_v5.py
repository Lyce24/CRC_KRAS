#!/usr/bin/env python3
"""Build the append-only reviews/v5 whole-section pathology packet.

reviews/v5 replaces the exposure-dependent selected-field rendering in v4.
For each of the same deterministically selected 60 eligible single-slide
patients, it renders one numbered overview and exactly six fresh JPEG panels.
The six level-0 rectangles form a disjoint 2x3 or 3x2 partition of the entire
OpenSlide main-image canvas.  No tissue detector, tumour detector, prototype,
or outcome is used to choose pixels.  Panel output is standardized to about
2 micrometres per pixel in both axes.

The reader receives only FOR_PATHOLOGIST.  Patient identities, molecular
variables, source paths, source identities, coordinates, and the unblinding
key remain under KEYS_DO_NOT_DISTRIBUTE.  The build is staged and atomically
renamed to reviews/v5 only after all census, geometry, decode, metadata, hash,
sampling, exposure, and leakage checks pass.

This builder is self-contained: case loading uses only the frozen e0 patient
profile and development manifest, plus the four prior-reader keys needed to
exclude their 191-patient union.  It does not import v2, E2A, or any score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "reviews" / "v5"
STAGING_NAME = ".v5.building"
ANALYZER = REPO / "tools" / "analyze_reviews_v5.py"

PROFILES = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim4_cap8192_corrected_v2_20260820/profiles/patient_profiles_k32.parquet"
)
DEV_MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/aim1_dev.csv")
MONTAGE_KEYS = (
    Path(
        "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
        "aim4_cap8192_corrected_v2_20260820/review_bundles/k32/base/"
        "unblinding_key_DO_NOT_SHARE.csv"
    ),
    Path(
        "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
        "aim4_cap8192_corrected_v2_20260820/review_bundles/k32/attention_addendum/"
        "unblinding_key_DO_NOT_SHARE.csv"
    ),
    Path("/mnt/wsl/oceanpath-hot/outputs/aim1/e3b/montages/k32/KEY_do_not_open_before_review.csv"),
    Path(
        "/mnt/wsl/oceanpath-hot/outputs/aim1/e3b/montages/k32/"
        "KEY_M11_followup_do_not_open_before_review.csv"
    ),
)
SLIDE_ROOT = Path("/mnt/d/YC.Liu/slides/colon")
CANONICAL_SLIDE_DIRS = ("CPTAC_COAD", "TCGA", "SURGEN", "rih")
SLIDE_SUFFIXES = (".svs", ".tiff", ".tif", ".scn")

SELECTION_SEED = 20260821
CASE_ID_SEED = 20260822
PER_EXPOSURE_GROUP = 20
EXPECTED_CASES = 60
EXPECTED_PRIOR_EXPOSED = 191
N_PANELS = 6
TARGET_MPP = 2.0
SOURCE_MPP_TOLERANCE = 1e-6
OVERVIEW_MAX_PX = 2400
JPEG_QUALITY = 90
ATTESTATION = "confirmed_no_key_or_source_access"

FORM_COLUMNS = (
    "case_id",
    "assessable",
    "extracellular_mucin_extent",
    "gland_formation",
    "note_if_unusual",
)
REVIEWER_COLUMNS = (
    "reviewer_id",
    "review_date",
    "viewer_software",
    "years_experience_gi_pathology",
    "elapsed_minutes",
    "blinding_attestation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, relative_to: Path | None = None) -> dict[str, Any]:
    display = path.relative_to(relative_to).as_posix() if relative_to else str(path.resolve())
    return {"path": display, "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def staged_file_record(path: Path, staging: Path, final_root: Path) -> dict[str, Any]:
    """Hash staged bytes while declaring their intended absolute final path."""
    final_path = final_root / path.relative_to(staging)
    return {
        "path": str(final_path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=REPO, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def software_and_git_metadata() -> dict[str, Any]:
    import openslide
    import PIL
    import scipy

    status = git_value("status", "--porcelain=v1") or ""
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
            "openslide_python": openslide.__version__,
            "openslide_library": getattr(openslide, "__library_version__", None),
            "scipy": scipy.__version__,
        },
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "is_dirty": bool(status),
            "status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
        },
    }


# ── self-contained frozen case loading and selection ────────────────────────
def previously_exposed_patients() -> set[str]:
    seen: set[str] = set()
    for key_path in MONTAGE_KEYS:
        table = pd.read_csv(key_path, dtype={"patient_id": str})
        if "patient_id" not in table.columns:
            raise AssertionError(f"prior-reader key lacks patient_id: {key_path}")
        seen.update(table["patient_id"].dropna().astype(str).str.strip())
    return seen


def load_case_frame() -> pd.DataFrame:
    """One row per development patient, using profile + manifest only."""
    profiles = pd.read_parquet(PROFILES)
    required_profile = {"arm", "patient_id", "prototype", "abundance"}
    if not required_profile.issubset(profiles.columns):
        raise AssertionError(f"profile lacks {sorted(required_profile - set(profiles.columns))}")
    e0 = profiles[profiles["arm"] == "e0"]
    wide = e0.pivot_table(
        index="patient_id", columns="prototype", values="abundance", aggfunc="first"
    )
    if 17 not in wide.columns or 28 not in wide.columns:
        raise AssertionError("frozen e0 profile lacks prototype 17 or 28")
    wide = wide[[17, 28]].rename(columns={17: "p17_abundance", 28: "p28_abundance"})

    manifest = pd.read_csv(DEV_MANIFEST, low_memory=False)
    required_manifest = {
        "slide_id",
        "patient_id",
        "cohort",
        "subcohort",
        "kras",
        "native_mpp",
        "specimen_role",
    }
    if not required_manifest.issubset(manifest.columns):
        raise AssertionError(
            f"development manifest lacks {sorted(required_manifest - set(manifest.columns))}"
        )
    if set(manifest["specimen_role"].dropna().astype(str)) != {"primary"}:
        raise AssertionError("development manifest contains non-primary specimens")

    patient_fields = manifest.drop_duplicates("patient_id")[
        [
            "patient_id",
            "cohort",
            "subcohort",
            "kras",
        ]
    ]
    slides = (
        manifest.groupby("patient_id", sort=False)["slide_id"]
        .apply(lambda values: sorted(values.astype(str)))
        .rename("slide_ids")
        .reset_index()
    )
    frame = patient_fields.merge(wide.reset_index(), on="patient_id", validate="one_to_one")
    frame = frame.merge(slides, on="patient_id", validate="one_to_one")
    frame = frame[frame["slide_ids"].map(len) == 1].copy()
    frame["slide_id"] = frame["slide_ids"].str[0]
    slide_mpp = manifest.drop_duplicates("slide_id")[["slide_id", "native_mpp"]]
    frame = frame.merge(slide_mpp, on="slide_id", validate="one_to_one")
    frame = frame[~frame["patient_id"].astype(str).isin(previously_exposed_patients())]
    if frame[["p17_abundance", "p28_abundance", "native_mpp"]].isna().any().any():
        raise AssertionError("eligible frame contains missing p17, p28, or native_mpp")
    return frame.reset_index(drop=True)


def assign_p17_groups(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["p17_group"] = "absent"
    positive = work["p17_abundance"] > 0
    for _cohort, block in work[positive].groupby("cohort", sort=True):
        median = float(block["p17_abundance"].median())
        work.loc[block.index[block["p17_abundance"] <= median], "p17_group"] = "positive_low"
        work.loc[block.index[block["p17_abundance"] > median], "p17_group"] = "positive_high"
    return work


def select_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the frozen 60-case v4 selection, then assign wholly new IDs."""
    rng = np.random.default_rng(SELECTION_SEED)
    picks: list[int] = []
    audit: list[dict[str, Any]] = []
    for group in ("absent", "positive_low", "positive_high"):
        per_class = PER_EXPOSURE_GROUP // 2
        for kras in ("mutant", "wild_type"):
            block = frame[(frame["p17_group"] == group) & (frame["kras"] == kras)]
            by_cohort = {
                cohort: rng.permutation(sub.sort_values("patient_id").index.to_numpy())
                for cohort, sub in block.groupby("cohort", sort=True)
            }
            chosen: list[int] = []
            position = 0
            while len(chosen) < per_class and any(
                position < len(values) for values in by_cohort.values()
            ):
                for cohort in sorted(by_cohort):
                    if len(chosen) >= per_class:
                        break
                    if position < len(by_cohort[cohort]):
                        chosen.append(int(by_cohort[cohort][position]))
                position += 1
            if len(chosen) != per_class:
                raise AssertionError(f"insufficient eligible cases for {group} x {kras}")
            picks.extend(chosen)
            audit.append(
                {
                    "p17_group": group,
                    "kras": kras,
                    "available": int(len(block)),
                    "selected": int(len(chosen)),
                }
            )
    sample = frame.loc[picks].copy()
    id_rng = np.random.default_rng(CASE_ID_SEED)
    sample = sample.iloc[id_rng.permutation(len(sample))].reset_index(drop=True)
    numeric_tokens = id_rng.choice(np.arange(10_000, 100_000), size=len(sample), replace=False)
    sample.insert(0, "case_id", [f"Q{value:05d}" for value in numeric_tokens])
    sample["sampling_cell"] = (
        sample["cohort"].astype(str)
        + "|"
        + sample["kras"].astype(str)
        + "|"
        + sample["p17_group"].astype(str)
    )
    sample.attrs["selection_audit"] = audit
    return sample


def resolve_slide(slide_id: str) -> Path:
    matches = []
    for directory in CANONICAL_SLIDE_DIRS:
        for suffix in SLIDE_SUFFIXES:
            candidate = SLIDE_ROOT / directory / f"{slide_id}{suffix}"
            if candidate.is_file():
                matches.append(candidate)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one canonical source for {slide_id}, found {matches}")
    return matches[0]


# ── exhaustive panel geometry and rendering ─────────────────────────────────
def grid_for_dimensions(width: int, height: int) -> tuple[int, int]:
    return (2, 3) if width >= height else (3, 2)


def panel_rectangles(width: int, height: int) -> list[dict[str, int]]:
    rows, columns = grid_for_dimensions(width, height)
    x_edges = [(column * width) // columns for column in range(columns + 1)]
    y_edges = [(row * height) // rows for row in range(rows + 1)]
    rectangles: list[dict[str, int]] = []
    number = 1
    for row in range(rows):
        for column in range(columns):
            rectangles.append(
                {
                    "panel_number": number,
                    "grid_rows": rows,
                    "grid_columns": columns,
                    "grid_row": row + 1,
                    "grid_column": column + 1,
                    "level0_x0": x_edges[column],
                    "level0_y0": y_edges[row],
                    "level0_x1": x_edges[column + 1],
                    "level0_y1": y_edges[row + 1],
                }
            )
            number += 1
    return rectangles


def select_source_level(
    level0_dimensions: tuple[int, int],
    level_dimensions: tuple[tuple[int, int], ...],
    mpp_x: float,
    mpp_y: float,
    target_mpp: float = TARGET_MPP,
) -> tuple[int, float, float]:
    """Choose the coarsest pyramid level that is not coarser than output."""
    width, height = level0_dimensions
    eligible: list[tuple[int, float, float, int]] = []
    for level, (level_width, level_height) in enumerate(level_dimensions):
        downsample_x = width / level_width
        downsample_y = height / level_height
        if (
            downsample_x * mpp_x <= target_mpp + SOURCE_MPP_TOLERANCE
            and downsample_y * mpp_y <= target_mpp + SOURCE_MPP_TOLERANCE
        ):
            eligible.append((level, downsample_x, downsample_y, level_width * level_height))
    if not eligible:
        raise AssertionError(
            f"even source level 0 is coarser than {target_mpp} um/px in at least one axis"
        )
    level, downsample_x, downsample_y, _area = min(eligible, key=lambda item: item[3])
    return level, downsample_x, downsample_y


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def physical_mpp(slide: Any, manifest_mpp: float) -> tuple[float, float, str]:
    import openslide

    mpp_x = safe_float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
    mpp_y = safe_float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y))
    if mpp_x is None or mpp_y is None:
        return manifest_mpp, manifest_mpp, "development_manifest_native_mpp"
    if abs(mpp_x - manifest_mpp) / manifest_mpp > 0.10:
        raise AssertionError(
            f"OpenSlide mpp-x {mpp_x} disagrees with manifest {manifest_mpp} by >10%"
        )
    if abs(mpp_y - manifest_mpp) / manifest_mpp > 0.10:
        raise AssertionError(
            f"OpenSlide mpp-y {mpp_y} disagrees with manifest {manifest_mpp} by >10%"
        )
    return mpp_x, mpp_y, "openslide.mpp-x_y"


def rgba_to_white_rgb(image: Any) -> Any:
    from PIL import Image

    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        image.close()
        return background
    converted = image.convert("RGB")
    image.close()
    return converted


def save_fresh_jpeg(image: Any, path: Path, quality: int = JPEG_QUALITY) -> None:
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    rgb.save(
        path,
        format="JPEG",
        quality=quality,
        optimize=False,
        progressive=False,
    )
    if rgb is not image:
        rgb.close()


def overview_with_numbered_grid(slide: Any, rectangles: list[dict[str, int]]) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    Image.MAX_IMAGE_PIXELS = None
    overview = rgba_to_white_rgb(
        slide.get_thumbnail((OVERVIEW_MAX_PX, OVERVIEW_MAX_PX)).convert("RGBA")
    )
    draw = ImageDraw.Draw(overview)
    width, height = slide.dimensions
    scale_x = overview.width / width
    scale_y = overview.height / height
    line_width = max(3, round(max(overview.size) / 500))
    font_size = max(28, round(max(overview.size) / 30))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    for rectangle in rectangles:
        x0 = round(rectangle["level0_x0"] * scale_x)
        y0 = round(rectangle["level0_y0"] * scale_y)
        x1 = round(rectangle["level0_x1"] * scale_x) - 1
        y1 = round(rectangle["level0_y1"] * scale_y) - 1
        draw.rectangle((x0, y0, x1, y1), outline=(255, 210, 0), width=line_width)
        label = str(rectangle["panel_number"])
        label_box = draw.textbbox((0, 0), label, font=font, stroke_width=2)
        label_w = label_box[2] - label_box[0]
        label_h = label_box[3] - label_box[1]
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        padding = max(5, line_width)
        draw.rounded_rectangle(
            (
                center_x - label_w // 2 - padding,
                center_y - label_h // 2 - padding,
                center_x + label_w // 2 + padding,
                center_y + label_h // 2 + padding,
            ),
            radius=padding,
            fill=(0, 0, 0),
        )
        draw.text(
            (center_x - label_w / 2, center_y - label_h / 2 - label_box[1]),
            label,
            font=font,
            fill=(255, 225, 0),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    return overview


def render_case(
    row: Any, slide_path: Path, image_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import openslide
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    image_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    slide = openslide.OpenSlide(str(slide_path))
    try:
        width, height = map(int, slide.dimensions)
        rectangles = panel_rectangles(width, height)
        mpp_x, mpp_y, mpp_source = physical_mpp(slide, float(row.native_mpp))

        overview = overview_with_numbered_grid(slide, rectangles)
        overview_name = f"{row.case_id}_overview.jpg"
        overview_path = image_dir / overview_name
        save_fresh_jpeg(overview, overview_path, quality=90)
        overview.close()
        image_rows.append(
            {
                "file": overview_name,
                "case_id": row.case_id,
                "image_role": "numbered_overview",
                "panel_number": "",
                "width_px": 0,
                "height_px": 0,
                "mode": "RGB",
                "size_bytes": overview_path.stat().st_size,
                "sha256": sha256(overview_path),
            }
        )

        level, downsample_x, downsample_y = select_source_level(
            (width, height), tuple(slide.level_dimensions), mpp_x, mpp_y
        )

        for rectangle in rectangles:
            region_width = rectangle["level0_x1"] - rectangle["level0_x0"]
            region_height = rectangle["level0_y1"] - rectangle["level0_y0"]
            read_width = max(1, math.ceil(region_width / downsample_x))
            read_height = max(1, math.ceil(region_height / downsample_y))
            output_width = max(1, round(region_width * mpp_x / TARGET_MPP))
            output_height = max(1, round(region_height * mpp_y / TARGET_MPP))

            rgba = slide.read_region(
                (rectangle["level0_x0"], rectangle["level0_y0"]),
                level,
                (read_width, read_height),
            )
            rgb = rgba_to_white_rgb(rgba)
            if rgb.size != (output_width, output_height):
                resized = rgb.resize(
                    (output_width, output_height), resample=Image.Resampling.LANCZOS
                )
                rgb.close()
                rgb = resized
            panel_number = rectangle["panel_number"]
            panel_name = f"{row.case_id}_panel{panel_number}.jpg"
            panel_path = image_dir / panel_name
            save_fresh_jpeg(rgb, panel_path)
            rgb.close()
            panel_sha = sha256(panel_path)
            panel_bytes = panel_path.stat().st_size
            actual_mpp_x = region_width * mpp_x / output_width
            actual_mpp_y = region_height * mpp_y / output_height
            panel_rows.append(
                {
                    "case_id": row.case_id,
                    "panel_number": panel_number,
                    "image_file": panel_name,
                    **rectangle,
                    "level0_width_px": region_width,
                    "level0_height_px": region_height,
                    "source_level": level,
                    "source_level_downsample_x": downsample_x,
                    "source_level_downsample_y": downsample_y,
                    "read_width_px": read_width,
                    "read_height_px": read_height,
                    "output_width_px": output_width,
                    "output_height_px": output_height,
                    "manifest_native_mpp": float(row.native_mpp),
                    "mpp_x_used": mpp_x,
                    "mpp_y_used": mpp_y,
                    "mpp_source": mpp_source,
                    "target_mpp": TARGET_MPP,
                    "actual_output_mpp_x": actual_mpp_x,
                    "actual_output_mpp_y": actual_mpp_y,
                    "size_bytes": panel_bytes,
                    "sha256": panel_sha,
                }
            )
            image_rows.append(
                {
                    "file": panel_name,
                    "case_id": row.case_id,
                    "image_role": "whole_section_panel",
                    "panel_number": panel_number,
                    "width_px": output_width,
                    "height_px": output_height,
                    "mode": "RGB",
                    "size_bytes": panel_bytes,
                    "sha256": panel_sha,
                }
            )

        quickhash = slide.properties.get(openslide.PROPERTY_NAME_QUICKHASH1)
        source_identity = {
            "case_id": row.case_id,
            "patient_id": row.patient_id,
            "slide_id": row.slide_id,
            "source_path": str(slide_path.resolve()),
            "source_suffix": slide_path.suffix.lower(),
            "source_size_bytes": slide_path.stat().st_size,
            "source_mtime_ns": slide_path.stat().st_mtime_ns,
            "source_sha256": "PENDING",
            "openslide_quickhash1": quickhash or "",
            "openslide_vendor": slide.properties.get(openslide.PROPERTY_NAME_VENDOR, ""),
            "level0_width_px": width,
            "level0_height_px": height,
            "level_count": slide.level_count,
            "level_dimensions_json": json.dumps(
                [list(values) for values in slide.level_dimensions]
            ),
            "level_downsamples_json": json.dumps(list(map(float, slide.level_downsamples))),
            "manifest_native_mpp": float(row.native_mpp),
            "mpp_x_used": mpp_x,
            "mpp_y_used": mpp_y,
            "mpp_source": mpp_source,
            "grid_rows": rectangles[0]["grid_rows"],
            "grid_columns": rectangles[0]["grid_columns"],
        }
    finally:
        slide.close()

    with Image.open(image_dir / overview_name) as check:
        image_rows[0]["width_px"], image_rows[0]["height_px"] = check.size
    return image_rows, panel_rows, source_identity


# ── packet documents ────────────────────────────────────────────────────────
def write_blank_csvs(reader: Path, keys: Path, sample: pd.DataFrame) -> None:
    score_path = reader / "scoring_form.csv"
    with score_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORM_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for case_id in sample["case_id"]:
            writer.writerow(
                {column: case_id if column == "case_id" else "" for column in FORM_COLUMNS}
            )
    reviewer_path = reader / "reviewer_info.csv"
    with reviewer_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEWER_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerow({column: "" for column in REVIEWER_COLUMNS})
    shutil.copyfile(score_path, keys / "scoring_form_TEMPLATE.csv")
    shutil.copyfile(reviewer_path, keys / "reviewer_info_TEMPLATE.csv")


def write_reader_instructions(reader: Path) -> None:
    (reader / "INSTRUCTIONS.md").write_text(
        f"""# Blinded whole-section panel read — 60 cases

Please complete all 60 cases. Plan approximately **3–5 hours total**, depending
on your viewer and reading pace. You may split the work across sessions. Save
only between cases; do not skip cases. A partial return cannot produce the
prespecified confirmatory result.

## What each case contains

The `images/` folder contains exactly seven RGB JPEGs per opaque case ID:

* `Qxxxxx_overview.jpg` shows the entire main slide image at low resolution.
  Yellow boxes and numbers show the six-panel layout.
* `Qxxxxx_panel1.jpg` through `Qxxxxx_panel6.jpg` are the numbered panels at
  approximately {TARGET_MPP:g} micrometres per pixel.

The six panels are not selected fields. They are contiguous, non-overlapping
rectangles that together cover the entire main slide image without gaps.
Landscape slides use a 2-row × 3-column layout; portrait slides use 3 rows × 2
columns. Panels are numbered left-to-right, then top-to-bottom. Blank glass,
pen marks, normal tissue, necrosis, and other non-tumour regions are retained
because nothing was selected or omitted. Inspect all six panels; use the
numbered overview to keep your place.

## Record one row per case in `scoring_form.csv`

**`assessable`** — enter `yes`, or enter `no` only if the complete overview and
all six panels are unreadable or do not permit tumour assessment. If `no`,
leave both score fields blank and briefly explain in `note_if_unusual`.

**`extracellular_mucin_extent`** — estimate the share of tumour area across the
whole represented section occupied by extracellular mucin (pools, lakes, or
stromal mucin; exclude intracytoplasmic mucin and ordinary goblet cells). Enter
exactly one:

* `none`
* `focal_lt10`
* `moderate_10_50`
* `extensive_gt50`

**`gland_formation`** — estimate the share of tumour across the whole
represented section forming recognisable glands. Enter exactly one:

* `gt95`
* `pct50_95`
* `lt50`

**`note_if_unusual`** is optional except when `assessable=no`.

## Blinding and completion

Before viewing images, fill `reviewer_id`, `review_date`, `viewer_software`, and
`years_experience_gi_pathology` in `reviewer_info.csv`. After confirming that
you received only this reader folder and had no key or source-slide access,
enter exactly `{ATTESTATION}` for `blinding_attestation`. When all 60 cases are
finished, record total active reading time in `elapsed_minutes`.

Case IDs are random and carry no patient, institution, molecular, or model
information. Do not try to identify cases or seek source slides, clinical data,
expected associations, or an unblinding key. Return only the two completed CSV
files to the coordinator and retain no local copy unless instructed by the
study team.
"""
    )


def write_analysis_plan(root: Path) -> None:
    (root / "ANALYSIS_PLAN.md").write_text(
        """# Frozen analysis plan — reviews/v5

This plan and `tools/analyze_reviews_v5.py` are frozen before any v5 case is
read. The generated-but-unread packet does not execute the analyzer.

## Design and measurement

The unit is the patient; all 60 patients have one slide. Each slide is shown as
one numbered overview plus exactly six exhaustive, non-overlapping panels that
partition the full OpenSlide level-0 main-image canvas. Panels are rendered at
approximately 2 micrometres per pixel. No tissue, tumour, model, prototype, or
outcome-dependent selection is used. The sample is balanced on p17 exposure
group and KRAS; it is not a prevalence sample and its effects are not directly
transportable to an unselected clinical series.

Mucin is coded none=0, focal_lt10=1, moderate_10_50=2,
extensive_gt50=3. Gland formation is coded lt50=1, pct50_95=2, gt95=3.

## Sole confirmatory endpoint

Higher continuous frozen p17 abundance is prespecified to associate with a
higher whole-section extracellular-mucin score.

1. Compute Kendall tau-b separately in each cohort × KRAS block for descriptive
   reporting.
2. Use a prespecified pair-weighted, restricted-pair Kendall tau-b as the
   primary summary. Within each block calculate `S_b=C_b-D_b`, the number of
   exposure-nontied pairs, and the number of outcome-nontied pairs. Sum each of
   those three components across all blocks, then calculate
   `sum(S_b) / sqrt(sum(non-tied-x_b) * sum(non-tied-y_b))`. No cross-block pair
   is compared. A constant-outcome or constant-exposure block remains in the
   relevant tie denominator and is never dropped or renormalized away.
3. Test the stratified summary with 20,000 permutations of mucin score within
   cohort × KRAS blocks, seed 20260824. The one-sided p-value is
   `(1 + count(null >= observed)) / (20,000 + 1)`.
4. Report the stratified tau-b and a 95% percentile CI from 2,000 patient
   bootstrap draws, seed 20260825, resampling with replacement separately
   within every cohort × KRAS × p17-group sampling cell.
5. Report a clinical-scale contrast: within each cohort × KRAS block, mean
   mucin score in positive-high minus absent p17; combine with the same fixed
   block design-n weights and report the same stratified-bootstrap CI plus the
   fraction of prespecified design weight covered by estimable block contrasts.

The minimum clinically meaningful stratified tau-b is 0.20. The result is
`SUPPORTS_PRIMARY_CLAIM` only if all four mechanical gates pass: stratified
tau-b >=0.20; one-sided permutation p<0.05; bootstrap CI lower bound >0; and
no more than 10% of cases are non-assessable. More than 10% non-assessable
forces `INCONCLUSIVE_MISSINGNESS` while retaining all estimates. Otherwise,
failure of any evidence gate yields `DOES_NOT_SUPPORT_PRIMARY_CLAIM`.

## Missingness and completion

All 60 case rows and a complete blinded-review record are required before the
script will unblind or emit any confirmatory result. `assessable=no` is a
completed disposition, but such patients are excluded from score analyses.
Analysis is complete-case only. Missingness is tabulated by the three p17
exposure groups. There is no modal or other imputation. A blank or invalid row,
missing case, duplicate case, or absent blinding attestation causes the analyzer
to refuse analysis before producing effect estimates.

## Explicitly exploratory endpoint

Gland-formation score versus continuous frozen p28 abundance is exploratory.
The script reports the same block-specific and restricted-pair stratified
tau-b plus its sampling-cell-stratified bootstrap CI, but no confirmatory test,
multiplicity-protected p-value, or error-controlled claim. It cannot rescue or
modify the primary conclusion.

No mediation, attenuation, prevalence, external-validity, inter-reader, or
intra-reader claim is made. This is blinded single-reader construct validation
in selected development-cohort cases.
"""
    )


def write_key_readme(keys: Path) -> None:
    (keys / "README.md").write_text(
        """# DO NOT DISTRIBUTE OR OPEN BEFORE THE RETURN IS SEALED

This directory contains the case-to-patient mapping, cohort/KRAS blocks,
continuous p17/p28 values, source paths, exact panel coordinates, source-slide
identities, and packet receipt. It must never be copied into the reader folder
or made accessible to the pathologist.

The two `*_TEMPLATE.csv` files are immutable copies of the original blank
forms. They make the pre-read packet auditable after the reader edits copies.
On return, hash and preserve the completed CSVs in a new access-controlled,
write-once return directory before opening this key or running the analyzer.
"""
    )


def write_coordinator_readme(
    root: Path, sample: pd.DataFrame, image_count: int, created: str
) -> None:
    counts = sample.groupby("cohort", sort=True).size().to_dict()
    (root / "README_COORDINATOR.md").write_text(
        f"""# reviews/v5 — sealed whole-section panel validation

This append-only packet supersedes v4 before distribution. v4 selected up to
six tissue-dense fields, and field count was associated with p17. v5 removes
that channel: every one of the {len(sample)} cases has one numbered overview
and exactly six disjoint panels that tile the full main-image canvas at about
{TARGET_MPP:g} micrometres per pixel ({image_count} JPEGs total).

## Release boundary

Hand the pathologist **only a fresh copy or archive of `FOR_PATHOLOGIST/`**.
Never grant repository access and never include `KEYS_DO_NOT_DISTRIBUTE/`, this
coordinator file, the analysis plan, or either tool script. Reader material has
opaque IDs and fresh RGB JPEGs only; it contains no source path, WSI extension,
EXIF, vendor metadata, associated label/macro image, cohort, patient ID, KRAS,
p17, or p28 value.

Before release, visually inspect the 60 numbered overviews for burned-in
identifiers or gross rendering failure. If any problem exists, **do not delete,
edit, withdraw, or substitute a case in this sealed packet**. Withhold the
entire version and build a new version with a new receipt.

## Reader-only copy/archive plan

1. Preserve this master packet unchanged and access-control the key directory.
2. Copy only `FOR_PATHOLOGIST/` to a new reader-delivery location, or archive
   exactly that directory. Do not archive the `reviews/v5` parent.
3. In the copied folder, verify every original payload line with
   `sha256sum -c HANDOFF_MANIFEST.sha256` before delivery. The manifest excludes
   itself and describes the pristine blank forms plus all 420 JPEGs.
4. Let the reader edit only the copied CSVs. The manifest is a pre-read seal and
   is not expected to validate edited returned forms.
5. Require all 60 case dispositions and the reviewer form. On return, copy the
   two completed CSVs into a new write-once, access-controlled return directory,
   hash them, and only then open the key.
6. Run the frozen analyzer into a new output directory; never run it in place:

       .venv/bin/python tools/analyze_reviews_v5.py \
         --scores RETURN/scoring_form.csv \
         --reviewer-info RETURN/reviewer_info.csv \
         --output-dir RETURN/analysis_run_01

Do not run the analyzer on the generated unread packet. Its blank forms must be
rejected, which is a deliberate completion gate.

## Frozen sample and scope

The sample uses seed {SELECTION_SEED} and the same deterministic eligible
single-slide patient selection as undistributed v4, with newly randomized
opaque IDs from an independent seed. It contains 20 absent, 20 positive-low,
and 20 positive-high p17 cases, each group split 10/10 by KRAS. Cohort counts
are {counts}. The full union of **{EXPECTED_PRIOR_EXPOSED}** patients exposed in
any of four earlier montage keys is excluded; selected overlap is zero.

Primary: p17 versus whole-section mucin. p28 versus gland formation is
explicitly exploratory. The frozen analysis plan and executable analyzer were
sealed at {created}. This remains a selected-development-cohort, single-reader
construct-validation study—not prevalence estimation, external validation, or
inter-reader reliability.
"""
    )


# ── final manifests and release audit ───────────────────────────────────────
def write_handoff_manifest(reader: Path) -> Path:
    manifest = reader / "HANDOFF_MANIFEST.sha256"
    payload = sorted(path for path in reader.rglob("*") if path.is_file() and path != manifest)
    lines = [f"{sha256(path)}  {path.relative_to(reader).as_posix()}" for path in payload]
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def rectangles_are_partition(panel_block: pd.DataFrame, width: int, height: int) -> bool:
    if len(panel_block) != N_PANELS:
        return False
    area = 0
    rectangles = []
    for row in panel_block.itertuples(index=False):
        x0, y0, x1, y1 = row.level0_x0, row.level0_y0, row.level0_x1, row.level0_y1
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            return False
        area += (x1 - x0) * (y1 - y0)
        rectangles.append((x0, y0, x1, y1))
    if area != width * height:
        return False
    for index, first in enumerate(rectangles):
        for second in rectangles[index + 1 :]:
            overlap_w = max(0, min(first[2], second[2]) - max(first[0], second[0]))
            overlap_h = max(0, min(first[3], second[3]) - max(first[1], second[1]))
            if overlap_w * overlap_h:
                return False
    return True


def release_audit(
    staging: Path,
    sample: pd.DataFrame,
    image_manifest: pd.DataFrame,
    panel_manifest: pd.DataFrame,
    source_manifest: pd.DataFrame,
) -> list[str]:
    from PIL import Image

    problems: list[str] = []
    reader = staging / "FOR_PATHOLOGIST"
    images = reader / "images"
    expected_ids = set(sample["case_id"].astype(str))
    actual_files = sorted(images.glob("*.jpg"))

    if len(sample) != EXPECTED_CASES or sample["patient_id"].nunique() != EXPECTED_CASES:
        problems.append("sample is not 60 unique patients")
    if sample["slide_id"].nunique() != EXPECTED_CASES:
        problems.append("sample is not 60 unique single slides")
    expected_group = {"absent": 20, "positive_low": 20, "positive_high": 20}
    if sample.groupby("p17_group").size().to_dict() != expected_group:
        problems.append("p17 exposure groups are not exactly 20/20/20")
    if sample.groupby("kras").size().to_dict() != {"mutant": 30, "wild_type": 30}:
        problems.append("KRAS is not exactly 30/30")
    cross = sample.groupby(["p17_group", "kras"]).size()
    if len(cross) != 6 or set(cross.to_list()) != {10}:
        problems.append("each p17 group is not exactly 10/10 by KRAS")
    expected_cells = {
        (cohort, kras, group): 3 if cohort in {"CPTAC", "RIH"} else 2
        for cohort in ("CPTAC", "RIH", "SurGen", "TCGA")
        for kras in ("mutant", "wild_type")
        for group in ("absent", "positive_low", "positive_high")
    }
    observed_cells = sample.groupby(["cohort", "kras", "p17_group"]).size().to_dict()
    if observed_cells != expected_cells:
        problems.append("24 cohort x KRAS x p17 sampling cells do not have expected 3/2 counts")
    if not sample.loc[sample["p17_group"] == "absent", "p17_abundance"].eq(0).all():
        problems.append("p17-absent group contains a nonzero exposure")
    if not sample.loc[sample["p17_group"] != "absent", "p17_abundance"].gt(0).all():
        problems.append("a p17-positive group contains a zero exposure")
    if sample["case_id"].duplicated().any() or not sample["case_id"].str.fullmatch(r"Q\d{5}").all():
        problems.append("case IDs are not unique opaque Q-plus-five-digit tokens")
    if len(previously_exposed_patients()) != EXPECTED_PRIOR_EXPOSED:
        problems.append("prior-exposed union is not exactly 191")
    overlap = set(sample["patient_id"].astype(str)) & previously_exposed_patients()
    if overlap:
        problems.append(f"selected sample overlaps prior-exposed patients: {sorted(overlap)}")

    if len(actual_files) != EXPECTED_CASES * (N_PANELS + 1):
        problems.append(f"image census is {len(actual_files)}, expected 420")
    if len(image_manifest) != len(actual_files):
        problems.append("image manifest row count differs from image census")
    if len(panel_manifest) != EXPECTED_CASES * N_PANELS:
        problems.append("panel manifest does not contain exactly 360 rows")
    if image_manifest["file"].duplicated().any() or panel_manifest["image_file"].duplicated().any():
        problems.append("duplicate image filename in a manifest")
    if (
        set(image_manifest["case_id"]) != expected_ids
        or set(panel_manifest["case_id"]) != expected_ids
    ):
        problems.append("manifest case IDs do not match the selected sample")
    counts = image_manifest.groupby(["case_id", "image_role"]).size().unstack(fill_value=0)
    if not all(counts.get("numbered_overview", pd.Series(dtype=int)).eq(1)) or not all(
        counts.get("whole_section_panel", pd.Series(dtype=int)).eq(6)
    ):
        problems.append("a case does not have exactly one overview and six panels")
    exposure_image_counts = (
        image_manifest[["case_id"]]
        .merge(sample[["case_id", "p17_group"]], on="case_id", validate="many_to_one")
        .groupby("p17_group")
        .size()
        .to_dict()
    )
    if exposure_image_counts != {
        "absent": 140,
        "positive_low": 140,
        "positive_high": 140,
    }:
        problems.append("rendered image exposure is not exactly 140 images per p17 group")

    source_by_case = source_manifest.set_index("case_id")
    for case_id, block in panel_manifest.groupby("case_id", sort=False):
        source = source_by_case.loc[case_id]
        if not rectangles_are_partition(
            block, int(source["level0_width_px"]), int(source["level0_height_px"])
        ):
            problems.append(f"{case_id}: panel level-0 rectangles do not exactly partition slide")
        if sorted(block["panel_number"].astype(int).to_list()) != list(range(1, 7)):
            problems.append(f"{case_id}: panel numbering is not exactly 1..6")
        if (block["actual_output_mpp_x"].sub(TARGET_MPP).abs() > 0.01).any() or (
            block["actual_output_mpp_y"].sub(TARGET_MPP).abs() > 0.01
        ).any():
            problems.append(f"{case_id}: output scale differs from 2 mpp by >0.01")
        if (
            block["source_level_downsample_x"] * block["mpp_x_used"]
            > TARGET_MPP + SOURCE_MPP_TOLERANCE
        ).any() or (
            block["source_level_downsample_y"] * block["mpp_y_used"]
            > TARGET_MPP + SOURCE_MPP_TOLERANCE
        ).any():
            problems.append(f"{case_id}: selected source pyramid level is coarser than output")

    manifest_by_file = image_manifest.set_index("file")
    allowed_info = {"jfif", "jfif_version", "jfif_unit", "jfif_density"}
    for path in actual_files:
        try:
            with Image.open(path) as image:
                image.load()
                row = manifest_by_file.loc[path.name]
                if image.mode != "RGB":
                    problems.append(f"{path.name}: mode is {image.mode}, not RGB")
                if image.size != (int(row["width_px"]), int(row["height_px"])):
                    problems.append(f"{path.name}: dimensions differ from image manifest")
                if len(image.getexif()) != 0:
                    problems.append(f"{path.name}: EXIF is not empty")
                unexpected = set(image.info) - allowed_info
                if unexpected:
                    problems.append(f"{path.name}: unexpected JPEG metadata {sorted(unexpected)}")
        except Exception as exc:  # noqa: BLE001 - every decode failure belongs in audit
            problems.append(f"{path.name}: decode failed: {exc}")
            continue
        if path.stat().st_size != int(manifest_by_file.loc[path.name, "size_bytes"]):
            problems.append(f"{path.name}: byte size differs from image manifest")
        if sha256(path) != manifest_by_file.loc[path.name, "sha256"]:
            problems.append(f"{path.name}: SHA-256 differs from image manifest")

    if image_manifest["sha256"].duplicated().any():
        problems.append("two rendered images have identical SHA-256")
    if not all(Path(path).is_file() for path in source_manifest["source_path"]):
        problems.append("one or more source slides no longer exists")
    if (source_manifest["source_sha256"] == "PENDING").any():
        problems.append("source-slide SHA-256 is pending")
    if not all(source_manifest["case_id"].astype(str).isin(expected_ids)):
        problems.append("source identity manifest contains an unexpected case")

    # No reader-side text may contain a source identity or path fragment.
    reader_text = "\n".join(
        path.read_text(errors="replace")
        for path in reader.rglob("*")
        if path.is_file() and path.suffix.lower() != ".jpg"
    )
    forbidden = set(sample["patient_id"].astype(str)) | set(sample["slide_id"].astype(str))
    forbidden |= {str(path) for path in source_manifest["source_path"]}
    leaked = sorted(value for value in forbidden if value and value in reader_text)
    if leaked:
        problems.append(f"reader text contains source identity/path values: {leaked[:3]}")
    if any(path.suffix.lower() != ".jpg" for path in images.iterdir() if path.is_file()):
        problems.append("reader image directory contains a non-JPEG file")

    return problems


def lock_key_permissions(keys: Path) -> None:
    os.chmod(keys, 0o700)
    for path in keys.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o700)
        else:
            os.chmod(path, 0o600)


def verify_final_receipt_payload(receipt: dict[str, Any]) -> list[str]:
    """Rehash every declared final identity before publishing a PASS receipt."""
    import openslide

    problems: list[str] = []
    if receipt.get("status") != "PASS" or receipt.get("problems") != []:
        problems.append("intended packet receipt is not PASS with zero problems")
    if (
        receipt.get("scientific_status") != "GENERATED_UNREAD"
        or receipt.get("analysis_executed") is not False
        or receipt.get("unblinding_performed") is not False
        or receipt.get("analysis_result") is not None
    ):
        problems.append("final packet receipt violates GENERATED_UNREAD state")

    records: list[tuple[str, dict[str, Any]]] = []
    for label in ("builder", "frozen_analyzer"):
        record = receipt.get(label)
        if isinstance(record, dict):
            records.append((label, record))
        else:
            problems.append(f"receipt lacks {label} identity")
    for group in ("frozen_inputs", "source_wsi_inputs", "outputs", "manifests"):
        values = receipt.get(group)
        if not isinstance(values, list):
            problems.append(f"receipt lacks {group} identities")
            continue
        records.extend((group, record) for record in values if isinstance(record, dict))

    for label, record in records:
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            problems.append(f"{label} record is not an absolute final path: {path}")
            continue
        if not path.is_file():
            problems.append(f"{label} path is missing after rename: {path}")
            continue
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            problems.append(f"{label} size mismatch after rename: {path}")
            continue
        if sha256(path) != record.get("sha256"):
            problems.append(f"{label} SHA-256 mismatch after rename: {path}")

    for record in receipt.get("source_wsi_inputs", []):
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            continue
        try:
            slide = openslide.OpenSlide(str(path))
            try:
                observed = slide.properties.get(openslide.PROPERTY_NAME_QUICKHASH1, "")
            finally:
                slide.close()
        except Exception as exc:  # noqa: BLE001 - source verification must report any failure
            problems.append(f"OpenSlide quickhash verification failed for {path}: {exc}")
            continue
        if observed != record.get("openslide_quickhash1", ""):
            problems.append(f"OpenSlide quickhash changed after rename: {path}")
    return problems


def publish_packet_receipt(receipt_path: Path, receipt: dict[str, Any]) -> None:
    """Publish the already-verified receipt once, using an atomic rename."""
    if receipt_path.exists():
        raise FileExistsError(f"write-once receipt already exists: {receipt_path}")
    temporary = receipt_path.with_name(".packet_receipt.json.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale receipt staging file exists: {temporary}")
    payload = (json.dumps(receipt, indent=2) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, receipt_path)
        os.chmod(receipt_path, 0o600)
        directory_descriptor = os.open(receipt_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def finalize_staged_packet(staging: Path, output: Path, receipt: dict[str, Any]) -> list[str]:
    """Rename, rehash against memory, then—and only then—publish PASS receipt."""
    if output.exists():
        return [f"append-only final output already exists: {output}"]
    staged_receipt = staging / "KEYS_DO_NOT_DISTRIBUTE" / "packet_receipt.json"
    if staged_receipt.exists():
        return ["staging unexpectedly contains a packet receipt before final verification"]
    staging.rename(output)
    problems = verify_final_receipt_payload(receipt)
    receipt_path = output / "KEYS_DO_NOT_DISTRIBUTE" / "packet_receipt.json"
    if problems:
        marker = output / "DO_NOT_RELEASE_BUILD_FAILED.txt"
        marker.write_text(
            "NO PASS RECEIPT WAS PUBLISHED. Post-rename verification failed:\n- "
            + "\n- ".join(problems)
            + "\n"
        )
        return problems
    try:
        publish_packet_receipt(receipt_path, receipt)
    except Exception as exc:  # noqa: BLE001 - publication failure must leave explicit no-go
        marker = output / "DO_NOT_RELEASE_BUILD_FAILED.txt"
        marker.write_text(f"NO PASS RECEIPT WAS PUBLISHED: {exc}\n")
        return [f"atomic packet-receipt publication failed: {exc}"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"append-only: final output already exists: {output}")
    if not ANALYZER.is_file():
        raise SystemExit(f"frozen analyzer is missing: {ANALYZER}")
    tool_freeze = {
        "builder": file_record(Path(__file__).resolve()),
        "frozen_analyzer": file_record(ANALYZER.resolve()),
    }
    staging = output.parent / STAGING_NAME
    if staging.exists():
        raise SystemExit(f"stale staging directory exists; inspect before proceeding: {staging}")

    frame = assign_p17_groups(load_case_frame())
    sample = select_cases(frame)
    if len(sample) != EXPECTED_CASES:
        raise AssertionError(f"selected {len(sample)} cases, expected {EXPECTED_CASES}")
    sample["source_path"] = sample["slide_id"].map(lambda value: str(resolve_slide(value)))

    reader = staging / "FOR_PATHOLOGIST"
    image_dir = reader / "images"
    keys = staging / "KEYS_DO_NOT_DISTRIBUTE"
    image_dir.mkdir(parents=True)
    keys.mkdir(parents=True)

    write_blank_csvs(reader, keys, sample)
    write_reader_instructions(reader)
    write_analysis_plan(staging)
    write_key_readme(keys)

    image_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for number, row in enumerate(sample.itertuples(index=False), start=1):
        case_images, case_panels, source = render_case(row, Path(row.source_path), image_dir)
        image_rows.extend(case_images)
        panel_rows.extend(case_panels)
        source_rows.append(source)
        total_bytes = sum(item["size_bytes"] for item in image_rows)
        print(
            f"rendered {number:02d}/{EXPECTED_CASES} {row.case_id}: 7 images; "
            f"cumulative {total_bytes / 1e9:.2f} GB",
            flush=True,
        )

    print("hashing 60 source WSI inputs (full SHA-256)", flush=True)
    for number, source in enumerate(source_rows, start=1):
        source["source_sha256"] = sha256(Path(source["source_path"]))
        if number % 5 == 0 or number == EXPECTED_CASES:
            print(f"hashed source {number:02d}/{EXPECTED_CASES}", flush=True)

    image_manifest = pd.DataFrame(image_rows).sort_values("file").reset_index(drop=True)
    panel_manifest = (
        pd.DataFrame(panel_rows).sort_values(["case_id", "panel_number"]).reset_index(drop=True)
    )
    source_manifest = pd.DataFrame(source_rows).sort_values("case_id").reset_index(drop=True)

    case_key = sample.copy()
    case_key["slide_ids"] = case_key["slide_ids"].map(lambda values: ";".join(values))
    case_key.to_csv(keys / "case_key.csv", index=False)
    pd.DataFrame(sample.attrs["selection_audit"]).to_csv(keys / "selection_audit.csv", index=False)
    pd.DataFrame(sorted(previously_exposed_patients()), columns=["patient_id"]).to_csv(
        keys / "excluded_previously_exposed_patients.csv", index=False
    )
    image_manifest.to_csv(keys / "image_manifest.csv", index=False)
    panel_manifest.to_csv(keys / "panel_manifest.csv", index=False)
    source_manifest.to_csv(keys / "source_identity_manifest.csv", index=False)

    created = datetime.now(timezone.utc).isoformat()
    write_coordinator_readme(staging, sample, len(image_manifest), created)
    handoff_manifest = write_handoff_manifest(reader)

    problems = release_audit(staging, sample, image_manifest, panel_manifest, source_manifest)
    for label, path in (
        ("builder", Path(__file__).resolve()),
        ("frozen_analyzer", ANALYZER.resolve()),
    ):
        current = file_record(path)
        frozen = tool_freeze[label]
        if current["size_bytes"] != frozen["size_bytes"] or current["sha256"] != frozen["sha256"]:
            problems.append(f"{label} changed during the long-running build")
    status = "PASS" if not problems else "FAIL"
    summary = {
        "schema_version": 1,
        "created_utc": created,
        "status": status,
        "problems": problems,
        "cases": int(len(sample)),
        "images": int(len(image_manifest)),
        "overviews": int((image_manifest["image_role"] == "numbered_overview").sum()),
        "panels": int((image_manifest["image_role"] == "whole_section_panel").sum()),
        "images_per_case": 7,
        "target_mpp": TARGET_MPP,
        "prior_exposed_union": len(previously_exposed_patients()),
        "selected_prior_exposure_overlap": 0,
        "by_cohort": sample.groupby("cohort", sort=True).size().to_dict(),
        "by_kras": sample.groupby("kras", sort=True).size().to_dict(),
        "by_p17_group": sample.groupby("p17_group", sort=True).size().to_dict(),
        "panel_output_pixels": int(
            (panel_manifest["output_width_px"] * panel_manifest["output_height_px"]).sum()
        ),
        "reader_payload_size_bytes": sum(
            path.stat().st_size for path in reader.rglob("*") if path.is_file()
        ),
        "scientific_status": "GENERATED_UNREAD",
        "analysis_executed": False,
        "unblinding_performed": False,
        "analysis_result": None,
    }
    (staging / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if problems:
        raise SystemExit(
            "release audit failed; staging preserved and final v5 not created:\n- "
            + "\n- ".join(problems)
        )

    frozen_inputs = [PROFILES, DEV_MANIFEST, *MONTAGE_KEYS]
    output_files = sorted(
        path for path in staging.rglob("*") if path.is_file() and path.name != "packet_receipt.json"
    )
    manifest_paths = [
        keys / "image_manifest.csv",
        keys / "panel_manifest.csv",
        keys / "source_identity_manifest.csv",
        keys / "case_key.csv",
        keys / "selection_audit.csv",
        keys / "excluded_previously_exposed_patients.csv",
        handoff_manifest,
    ]
    receipt = {
        "schema_version": 2,
        "created_utc": created,
        "status": "PASS",
        "problems": [],
        "scientific_status": "GENERATED_UNREAD",
        "analysis_executed": False,
        "unblinding_performed": False,
        "analysis_result": None,
        "supersedes": "undistributed reviews/v4 selected-field packet",
        "builder": tool_freeze["builder"],
        "frozen_analyzer": tool_freeze["frozen_analyzer"],
        "frozen_inputs": [file_record(path) for path in frozen_inputs],
        "source_wsi_inputs": [
            {
                "path": row["source_path"],
                "size_bytes": int(row["source_size_bytes"]),
                "sha256": row["source_sha256"],
                "openslide_quickhash1": row["openslide_quickhash1"],
            }
            for row in source_rows
        ],
        "outputs": [staged_file_record(path, staging, output) for path in output_files],
        "manifests": [staged_file_record(path, staging, output) for path in manifest_paths],
        "census": {
            "cases": EXPECTED_CASES,
            "images": EXPECTED_CASES * (N_PANELS + 1),
            "overviews": EXPECTED_CASES,
            "panels": EXPECTED_CASES * N_PANELS,
            "exact_images_per_case": N_PANELS + 1,
        },
        "panel_design": (
            "six disjoint level-0 rectangles exactly partition the full main-image "
            "canvas; 2x3 landscape or 3x2 portrait; output ~2 um/px"
        ),
        "blinding": (
            "reader-only folder contains opaque random case IDs, fresh RGB JPEGs, "
            "blank forms, instructions, and reader-side hashes; no raw WSI/path/key"
        ),
        "software_and_git": software_and_git_metadata(),
        "write_policy": "staged build atomically renamed to append-only final directory",
    }
    lock_key_permissions(keys)
    print(
        "post-rename: rehashing every intended receipt identity before publication",
        flush=True,
    )
    final_problems = finalize_staged_packet(staging, output, receipt)
    if final_problems:
        raise SystemExit(
            "post-rename receipt verification failed; v5 must not be released:\n- "
            + "\n- ".join(final_problems)
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
