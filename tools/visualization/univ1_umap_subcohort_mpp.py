#!/usr/bin/env python3
"""Replot the all-slide UNI-v1 UMAP by subcohort shape and MPP-bin color."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

COORDINATES = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/cohort_umap_all_slides/coordinates_univ1.parquet"
)
QUEUE_DB = Path("/home/yc_liu/projects/OceanPath-colon/outputs/colon_stream/queue.sqlite")
MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v4.csv")
OUTPUT_DIR = Path("/mnt/wsl/oceanpath-hot/outputs/eval/univ1_umap_subcohort_mpp_clear")

SUBCOHORT_ORDER = (
    "CPTAC-COAD",
    "TCGA-COAD",
    "TCGA-READ",
    "SR386",
    "SR1482",
    "RIH-Colon",
)
SUBCOHORT_MARKERS = {
    "CPTAC-COAD": "P",
    "TCGA-COAD": "o",
    "TCGA-READ": "s",
    "SR386": "^",
    "SR1482": "v",
    "RIH-Colon": "D",
}
EXPECTED_SUBCOHORT_COUNTS = {
    "CPTAC-COAD": 98,
    "TCGA-COAD": 442,
    "TCGA-READ": 158,
    "SR386": 427,
    "SR1482": 593,
    "RIH-Colon": 369,
}

MPP_EDGES = (-np.inf, 0.21, 0.24, 0.251, 0.30, np.inf)
MPP_LABELS = (
    "<0.21",
    "0.21–<0.24",
    "0.24–<0.251",
    "0.251–<0.30",
    "≥0.30",
)
MPP_COLORS = {
    label: color
    for label, color in zip(
        MPP_LABELS,
        ("#542788", "#2166ac", "#1b9e77", "#e08214", "#b2182b"),
        strict=True,
    )
}
EXPECTED_MPP_BIN_COUNTS = {
    "<0.21": 99,
    "0.21–<0.24": 231,
    "0.24–<0.251": 1165,
    "0.251–<0.30": 322,
    "≥0.30": 270,
}

INK = "#1a1a19"
MUTED = "#6b6a62"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def assemble_metadata(
    coordinates_path: Path,
    queue_db: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    coordinates = pd.read_parquet(coordinates_path)
    required_coords = {"slide_id", "cohort", "umap_1", "umap_2"}
    if not required_coords.issubset(coordinates.columns):
        raise ValueError(
            f"Coordinates lack columns {sorted(required_coords - set(coordinates.columns))}"
        )
    if coordinates["slide_id"].astype(str).duplicated().any():
        raise ValueError("UNI-v1 coordinates contain duplicate slide IDs")

    uri = f"file:{queue_db}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        queue = pd.read_sql_query(
            "SELECT output_id AS slide_id, cohort, mpp FROM stream_jobs", connection
        )
    queue["slide_id"] = queue["slide_id"].astype(str)
    if queue["slide_id"].duplicated().any():
        raise ValueError("Encoder queue contains duplicate slide IDs")

    manifest = pd.read_csv(manifest_path, usecols=["output_id", "subcohort"])
    manifest = manifest.rename(columns={"output_id": "slide_id"})
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    if manifest["slide_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate output_id values")

    frame = coordinates.merge(queue, on=["slide_id", "cohort"], validate="one_to_one")
    frame = frame.merge(manifest, on="slide_id", how="left", validate="one_to_one")
    # crc_final_v4 covers label-eligible RIH slides; the additional encoded
    # RIH slides share the same sole subcohort identity.
    frame.loc[frame["subcohort"].isna() & frame["cohort"].eq("RIH"), "subcohort"] = "RIH-Colon"

    if len(frame) != 2087:
        raise ValueError(f"Expected 2,087 UNI-v1 slides, got {len(frame)}")
    if frame[["subcohort", "mpp"]].isna().any().any():
        missing = frame[frame[["subcohort", "mpp"]].isna().any(axis=1)]
        raise ValueError(
            f"Slides lack subcohort/MPP metadata: {missing['slide_id'].head().tolist()}"
        )
    if not np.isfinite(frame[["mpp", "umap_1", "umap_2"]]).all().all():
        raise ValueError("MPP or UMAP coordinates contain non-finite values")

    frame["mpp_bin"] = pd.cut(
        frame["mpp"],
        bins=MPP_EDGES,
        labels=MPP_LABELS,
        right=False,
        ordered=True,
    )
    subcohort_counts = frame["subcohort"].value_counts().to_dict()
    if subcohort_counts != EXPECTED_SUBCOHORT_COUNTS:
        raise ValueError(
            f"Unexpected subcohort counts: {subcohort_counts} != {EXPECTED_SUBCOHORT_COUNTS}"
        )
    mpp_counts = frame["mpp_bin"].value_counts(sort=False).to_dict()
    if mpp_counts != EXPECTED_MPP_BIN_COUNTS:
        raise ValueError(f"Unexpected MPP-bin counts: {mpp_counts} != {EXPECTED_MPP_BIN_COUNTS}")
    return frame


def _draw_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    point_size: float,
    alpha: float,
    outlined: bool,
) -> None:
    for subcohort in SUBCOHORT_ORDER:
        subcohort_frame = frame[frame["subcohort"].eq(subcohort)]
        counts = subcohort_frame["mpp_bin"].astype(str).value_counts()
        # Draw common colors first so uncommon acquisition settings remain visible.
        for mpp_bin in sorted(MPP_LABELS, key=lambda label: counts.get(label, 0), reverse=True):
            selection = subcohort_frame[subcohort_frame["mpp_bin"].astype(str).eq(mpp_bin)]
            if selection.empty:
                continue
            ax.scatter(
                selection["umap_1"],
                selection["umap_2"],
                marker=SUBCOHORT_MARKERS[subcohort],
                s=point_size,
                c=MPP_COLORS[mpp_bin],
                alpha=alpha,
                edgecolors="white" if outlined else "none",
                linewidths=0.3 if outlined else 0,
                rasterized=True,
            )


def _style_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d8d7d0")
        spine.set_linewidth(0.8)
    ax.set_facecolor("#fbfbf9")


def _zoom_to_frame(ax: plt.Axes, frame: pd.DataFrame, padding: float = 0.08) -> None:
    x_min, x_max = frame["umap_1"].agg(["min", "max"])
    y_min, y_max = frame["umap_2"].agg(["min", "max"])
    x_pad = max((x_max - x_min) * padding, 0.08)
    y_pad = max((y_max - y_min) * padding, 0.08)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")


def save_figure(frame: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    fig = plt.figure(figsize=(18, 10.5), facecolor="white")
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.18, 1, 1, 1),
        left=0.045,
        right=0.985,
        top=0.87,
        bottom=0.16,
        wspace=0.16,
        hspace=0.25,
    )

    overview = fig.add_subplot(grid[:, 0])
    _draw_points(overview, frame, point_size=15, alpha=0.78, outlined=False)
    _style_axis(overview)
    _zoom_to_frame(overview, frame, padding=0.04)
    overview.set_title(
        "A  All slides · global geometry",
        fontsize=12,
        color=INK,
        loc="left",
        weight="semibold",
        pad=10,
    )
    overview.set_xlabel("UMAP 1", fontsize=9, color=MUTED)
    overview.set_ylabel("UMAP 2", fontsize=9, color=MUTED)

    panel_letters = "BCDEFG"
    for index, subcohort in enumerate(SUBCOHORT_ORDER):
        ax = fig.add_subplot(grid[index // 3, 1 + index % 3])
        subset = frame[frame["subcohort"].eq(subcohort)].copy()
        _draw_points(ax, subset, point_size=28, alpha=0.88, outlined=True)
        _style_axis(ax)
        _zoom_to_frame(ax, subset)
        ax.set_title(
            f"{panel_letters[index]}  {subcohort} · n={len(subset):,}",
            fontsize=11,
            color=INK,
            loc="left",
            weight="semibold",
            pad=8,
        )

    shape_handles = [
        Line2D(
            [0],
            [0],
            marker=SUBCOHORT_MARKERS[subcohort],
            linestyle="none",
            markerfacecolor="#777771",
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=8,
            label=subcohort,
        )
        for subcohort in SUBCOHORT_ORDER
    ]
    color_handles = [
        Patch(
            facecolor=MPP_COLORS[label],
            edgecolor="none",
            label=f"{label} µm/px (n={EXPECTED_MPP_BIN_COUNTS[label]:,})",
        )
        for label in MPP_LABELS
    ]
    shape_legend = fig.legend(
        handles=shape_handles,
        title="Shape = subcohort",
        loc="lower center",
        bbox_to_anchor=(0.29, 0.035),
        ncol=3,
        fontsize=8.5,
        title_fontsize=10,
        frameon=False,
        labelcolor=MUTED,
    )
    fig.add_artist(shape_legend)
    fig.legend(
        handles=color_handles,
        title="Color = MPP bin",
        loc="lower center",
        bbox_to_anchor=(0.75, 0.035),
        ncol=3,
        fontsize=8.5,
        title_fontsize=10,
        frameon=False,
        labelcolor=MUTED,
    )
    fig.suptitle(
        "UNI-v1 slide embeddings by subcohort and scan resolution",
        x=0.045,
        y=0.965,
        ha="left",
        fontsize=18,
        color=INK,
        weight="semibold",
    )
    fig.text(
        0.045,
        0.915,
        "Shape identifies subcohort; color identifies MPP bin. Panels B–G zoom locally while preserving the original UMAP coordinates.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    fig.text(
        0.985,
        0.012,
        "All 2,087 slides · existing cosine UMAP · exact slide mean pooling · seed 42",
        fontsize=8,
        color=MUTED,
        ha="right",
    )
    png_path = output_dir / "umap_univ1_subcohort_shape_mpp_color_clear.png"
    pdf_path = output_dir / "umap_univ1_subcohort_shape_mpp_color_clear.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    subset_subcohorts = ("TCGA-COAD", "TCGA-READ", "RIH-Colon")
    informative = frame[frame["subcohort"].isin(subset_subcohorts)].copy()
    subset_fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.7), facecolor="white")
    for index, (ax, subcohort) in enumerate(zip(axes, subset_subcohorts, strict=True)):
        subset = informative[informative["subcohort"].eq(subcohort)]
        _draw_points(ax, subset, point_size=34, alpha=0.9, outlined=True)
        _style_axis(ax)
        _zoom_to_frame(ax, subset)
        ax.set_title(
            f"{chr(ord('A') + index)}  {subcohort} · n={len(subset):,}",
            fontsize=12,
            color=INK,
            loc="left",
            weight="semibold",
            pad=9,
        )

    subset_counts = informative["mpp_bin"].value_counts(sort=False).to_dict()
    subset_color_handles = [
        Patch(
            facecolor=MPP_COLORS[label],
            edgecolor="none",
            label=f"{label} µm/px (n={subset_counts[label]:,})",
        )
        for label in MPP_LABELS
    ]
    subset_shape_handles = [
        Line2D(
            [0],
            [0],
            marker=SUBCOHORT_MARKERS[subcohort],
            linestyle="none",
            markerfacecolor="#777771",
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=8,
            label=subcohort,
        )
        for subcohort in subset_subcohorts
    ]
    subset_fig.legend(
        handles=subset_shape_handles,
        title="Shape = subcohort",
        loc="lower center",
        bbox_to_anchor=(0.25, 0.005),
        ncol=3,
        fontsize=8.5,
        title_fontsize=9.5,
        frameon=False,
        labelcolor=MUTED,
    )
    subset_fig.legend(
        handles=subset_color_handles,
        title="Color = MPP bin",
        loc="lower center",
        bbox_to_anchor=(0.72, 0.005),
        ncol=3,
        fontsize=8.5,
        title_fontsize=9.5,
        frameon=False,
        labelcolor=MUTED,
    )
    subset_fig.suptitle(
        "UNI-v1: MPP-informative subcohorts",
        x=0.04,
        y=0.975,
        ha="left",
        fontsize=17,
        color=INK,
        weight="semibold",
    )
    subset_fig.text(
        0.04,
        0.91,
        "Local UMAP zooms for the three subcohorts containing more than one MPP bin; coordinates are unchanged and panel limits differ.",
        fontsize=9.5,
        color=MUTED,
        ha="left",
    )
    subset_fig.subplots_adjust(left=0.04, right=0.985, top=0.82, bottom=0.23, wspace=0.15)
    subset_png_path = output_dir / "umap_univ1_mpp_informative_subcohorts.png"
    subset_pdf_path = output_dir / "umap_univ1_mpp_informative_subcohorts.pdf"
    subset_fig.savefig(subset_png_path, dpi=240, bbox_inches="tight")
    subset_fig.savefig(subset_pdf_path, dpi=240, bbox_inches="tight")
    plt.close(subset_fig)

    return {
        "all_slides_png": png_path,
        "all_slides_pdf": pdf_path,
        "mpp_informative_subset_png": subset_png_path,
        "mpp_informative_subset_pdf": subset_pdf_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, default=COORDINATES)
    parser.add_argument("--queue-db", type=Path, default=QUEUE_DB)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Replace this script's outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = assemble_metadata(args.coordinates, args.queue_db, args.manifest)
    frame.to_parquet(args.output_dir / "coordinates_univ1_subcohort_mpp.parquet", index=False)
    figures = save_figure(frame, args.output_dir)

    crosstab = pd.crosstab(frame["subcohort"], frame["mpp_bin"], dropna=False)
    summary = {
        "analysis": "univ1_umap_subcohort_shape_mpp_color",
        "n_slides": len(frame),
        "coordinates_source": str(args.coordinates),
        "coordinates_sha256": _sha256_file(args.coordinates),
        "metadata_sources": {
            "queue_db": str(args.queue_db),
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256_file(args.manifest),
        },
        "subcohort_marker_shapes": SUBCOHORT_MARKERS,
        "subcohort_counts": EXPECTED_SUBCOHORT_COUNTS,
        "mpp_bin_edges": ["-inf", 0.21, 0.24, 0.251, 0.30, "+inf"],
        "mpp_bin_colors": MPP_COLORS,
        "mpp_bin_counts": EXPECTED_MPP_BIN_COUNTS,
        "subcohort_by_mpp_bin": crosstab.astype(int).to_dict(orient="index"),
        "figures": {name: str(path) for name, path in figures.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(
        "\n".join(
            [
                "# UNI-v1 UMAP: subcohort shape and MPP-bin color",
                "",
                "The existing all-slide UNI-v1 UMAP is replotted for 2,087 slides.",
                "Marker shape encodes subcohort; ordered color encodes MPP bin.",
                "Panel A preserves the global view. Panels B–G use local axis limits",
                "for clarity but do not refit or otherwise change the coordinates.",
                "The focused three-panel figure includes only TCGA-COAD, TCGA-READ,",
                "and RIH-Colon, the subcohorts represented by more than one MPP bin.",
                "",
                "| MPP bin (µm/px) | Slides |",
                "|---|---:|",
                *[f"| {label} | {EXPECTED_MPP_BIN_COUNTS[label]:,} |" for label in MPP_LABELS],
                "",
                "MPP and subcohort are strongly correlated acquisition attributes, so the",
                "two visual encodings should not be interpreted as independent effects.",
                "",
            ]
        )
    )
    print(f"Wrote UNI-v1 subcohort/MPP UMAP to {args.output_dir}")


if __name__ == "__main__":
    main()
