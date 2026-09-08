#!/usr/bin/env python3
"""Combine the UNI-v1 subcohort/MPP overview with audited residual rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from tools.visualization.univ1_umap_subcohort_mpp import (  # noqa: E402
    EXPECTED_MPP_BIN_COUNTS,
    EXPECTED_SUBCOHORT_COUNTS,
    MPP_COLORS,
    MPP_LABELS,
    SUBCOHORT_MARKERS,
    SUBCOHORT_ORDER,
    _style_axis,
    _zoom_to_frame,
)

OVERVIEW_COORDINATES = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/univ1_umap_subcohort_mpp_clear/"
    "coordinates_univ1_subcohort_mpp.parquet"
)
AUDIT_ASSIGNMENTS = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/univ1_umap_separation_audit/"
    "slide_cluster_assignments.parquet"
)
AUDIT_SUMMARY = Path("/mnt/wsl/oceanpath-hot/outputs/eval/univ1_umap_separation_audit/summary.json")
MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v4.csv")
OUTPUT_DIR = Path("/mnt/wsl/oceanpath-hot/outputs/eval/univ1_umap_comprehensive")

INK = "#1a1a19"
MUTED = "#6b6a62"
NEGATIVE_COLOR = "#4d4d48"
POSITIVE_COLOR = "#cc4678"

EXPECTED_SEMANTIC_COUNTS = {
    "SR386": {"main_stain_island": 379, "cool_stain_island": 48},
    "SR1482": {
        "large_section_island": 443,
        "small_fragment_island": 149,
        "atypical_outlier": 1,
    },
    "RIH-Colon": {
        "large_section_island": 177,
        "small_fragment_island": 88,
        "different_mpp": 104,
    },
}
EXPECTED_TISSUE_CLASS_COUNTS = {
    "large/full section": 620,
    "small biopsy/fragment": 237,
    "not audited": 1230,
}
EXPECTED_ROLE_COUNTS = {
    "SR1482": {"primary": 483, "metastatic": 110},
    "RIH-Colon": {"primary": 186, "metastatic": 99},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_inputs(
    overview_path: Path,
    assignments_path: Path,
    summary_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    overview = pd.read_parquet(overview_path)
    assignments = pd.read_parquet(assignments_path)
    summary = json.loads(summary_path.read_text())
    manifest = pd.read_csv(
        manifest_path,
        usecols=["output_id", "subcohort", "specimen_role", "used"],
    ).rename(columns={"output_id": "slide_id"})

    required_overview = {"slide_id", "subcohort", "mpp_bin", "umap_1", "umap_2"}
    required_assignments = {
        "slide_id",
        "subcohort",
        "visible_island",
        "umap_1",
        "umap_2",
        "best_simple_rule_correct",
    }
    if missing := required_overview - set(overview.columns):
        raise ValueError(f"Overview coordinates missing columns: {sorted(missing)}")
    if missing := required_assignments - set(assignments.columns):
        raise ValueError(f"Audit assignments missing columns: {sorted(missing)}")
    if len(overview) != 2087 or overview["slide_id"].nunique() != 2087:
        raise ValueError("Overview must contain exactly 2,087 unique slides")
    if len(assignments) != 1389 or assignments["slide_id"].nunique() != 1389:
        raise ValueError("Audit assignments must contain exactly 1,389 unique slides")
    if overview["subcohort"].value_counts().to_dict() != EXPECTED_SUBCOHORT_COUNTS:
        raise ValueError("Overview subcohort counts changed")
    observed_mpp = overview["mpp_bin"].value_counts(sort=False).to_dict()
    if observed_mpp != EXPECTED_MPP_BIN_COUNTS:
        raise ValueError(f"Overview MPP-bin counts changed: {observed_mpp}")

    observed_semantic = {
        subcohort: subset["visible_island"].value_counts().astype(int).to_dict()
        for subcohort, subset in assignments.groupby("subcohort")
    }
    if observed_semantic != EXPECTED_SEMANTIC_COUNTS:
        raise ValueError(f"Audit semantic counts changed: {observed_semantic}")

    coordinate_check = overview[["slide_id", "umap_1", "umap_2"]].merge(
        assignments[["slide_id", "umap_1", "umap_2"]],
        on="slide_id",
        how="inner",
        suffixes=("_overview", "_audit"),
        validate="one_to_one",
    )
    if len(coordinate_check) != len(assignments):
        raise ValueError("Not every audited slide is present in the overview")
    for axis in ("umap_1", "umap_2"):
        if not np.allclose(
            coordinate_check[f"{axis}_overview"],
            coordinate_check[f"{axis}_audit"],
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(f"Overview and audit {axis} coordinates differ")

    expected_assignment_hash = summary.get("artifacts", {}).get("sha256", {}).get("assignments")
    if expected_assignment_hash and _sha256(assignments_path) != expected_assignment_hash:
        raise ValueError("Audit assignment hash does not match its summary")
    tissue_lookup = assignments[["slide_id", "visible_island"]].copy()
    tissue_lookup["tissue_class"] = np.select(
        [
            tissue_lookup["visible_island"].eq("large_section_island"),
            tissue_lookup["visible_island"].eq("small_fragment_island"),
        ],
        ["large/full section", "small biopsy/fragment"],
        default="not audited",
    )
    overview = overview.merge(
        tissue_lookup[["slide_id", "tissue_class"]],
        on="slide_id",
        how="left",
        validate="one_to_one",
    )
    overview["tissue_class"] = overview["tissue_class"].fillna("not audited")
    observed_tissue = overview["tissue_class"].value_counts().astype(int).to_dict()
    if observed_tissue != EXPECTED_TISSUE_CLASS_COUNTS:
        raise ValueError(f"Audited tissue-class counts changed: {observed_tissue}")

    manifest["slide_id"] = manifest["slide_id"].astype(str)
    if manifest["slide_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate output_id values")
    role_lookup = manifest[manifest["subcohort"].isin(["SR1482", "RIH-Colon"])][
        ["slide_id", "subcohort", "specimen_role", "used"]
    ].rename(columns={"subcohort": "manifest_subcohort"})
    overview = overview.merge(role_lookup, on="slide_id", how="left", validate="one_to_one")
    matched = overview["manifest_subcohort"].notna()
    if not overview.loc[matched, "subcohort"].eq(overview.loc[matched, "manifest_subcohort"]).all():
        raise ValueError("Manifest and overview subcohort labels disagree")
    role_subset = overview[
        overview["subcohort"].isin(["SR1482", "RIH-Colon"])
        & overview["specimen_role"].isin(["primary", "metastatic"])
    ]
    observed_roles = {
        subcohort: subset["specimen_role"].value_counts().astype(int).to_dict()
        for subcohort, subset in role_subset.groupby("subcohort", sort=True)
    }
    if observed_roles != EXPECTED_ROLE_COUNTS:
        raise ValueError(f"SR1482/RIH specimen-role counts changed: {observed_roles}")
    if not role_subset["used"].eq("yes").all():
        raise ValueError("A plotted SR1482/RIH specimen-role slide is not QC eligible")
    return overview, assignments, summary


def _draw_three_factor_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    point_size: float,
) -> None:
    """Encode subcohort by shape, MPP by color, and audited tissue class by fill."""
    tissue_order = ("not audited", "large/full section", "small biopsy/fragment")
    for subcohort in SUBCOHORT_ORDER:
        subcohort_frame = frame[frame["subcohort"].eq(subcohort)]
        mpp_counts = subcohort_frame["mpp_bin"].astype(str).value_counts()
        for mpp_bin in sorted(
            MPP_LABELS,
            key=lambda label: mpp_counts.get(label, 0),
            reverse=True,
        ):
            mpp_frame = subcohort_frame[subcohort_frame["mpp_bin"].astype(str).eq(mpp_bin)]
            for tissue_class in tissue_order:
                selection = mpp_frame[mpp_frame["tissue_class"].eq(tissue_class)]
                if selection.empty:
                    continue
                common = {
                    "x": selection["umap_1"],
                    "y": selection["umap_2"],
                    "marker": SUBCOHORT_MARKERS[subcohort],
                    "s": point_size,
                    "rasterized": True,
                }
                if tissue_class == "small biopsy/fragment":
                    ax.scatter(
                        **common,
                        facecolors="white",
                        edgecolors=MPP_COLORS[mpp_bin],
                        linewidths=1.05,
                        alpha=1.0,
                    )
                elif tissue_class == "large/full section":
                    ax.scatter(
                        **common,
                        c=MPP_COLORS[mpp_bin],
                        edgecolors="white",
                        linewidths=0.3,
                        alpha=0.92,
                    )
                else:
                    ax.scatter(
                        **common,
                        c=MPP_COLORS[mpp_bin],
                        edgecolors="none",
                        linewidths=0,
                        alpha=0.52,
                    )


def _overview_legends(ax: plt.Axes) -> None:
    ax.axis("off")
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
    tissue_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#777771",
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=7,
            label="large/full section",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#555550",
            markeredgewidth=1.2,
            markersize=7,
            label="small biopsy/fragment",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#aaa9a3",
            markeredgecolor="none",
            alpha=0.5,
            markersize=7,
            label="not audited / other MPP",
        ),
    ]
    shape_legend = ax.legend(
        handles=shape_handles,
        title="Shape = subcohort",
        loc="center left",
        bbox_to_anchor=(0.0, 0.60),
        ncol=3,
        fontsize=8.2,
        title_fontsize=9.5,
        frameon=False,
        labelcolor=MUTED,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    ax.add_artist(shape_legend)
    color_legend = ax.legend(
        handles=color_handles,
        title="Color = native MPP bin",
        loc="center",
        bbox_to_anchor=(0.61, 0.60),
        ncol=3,
        fontsize=8.2,
        title_fontsize=9.5,
        frameon=False,
        labelcolor=MUTED,
        handletextpad=0.55,
        columnspacing=1.1,
    )
    ax.add_artist(color_legend)
    ax.legend(
        handles=tissue_handles,
        title="Fill = audited tissue class",
        loc="center right",
        bbox_to_anchor=(1.0, 0.60),
        ncol=1,
        fontsize=8.2,
        title_fontsize=9.5,
        frameon=False,
        labelcolor=MUTED,
        handletextpad=0.55,
    )
    ax.text(
        0.0,
        -0.08,
        "H–J  Residual within-subcohort rules after separating the explicit MPP effect",
        transform=ax.transAxes,
        fontsize=12,
        weight="semibold",
        color=INK,
        va="top",
    )


def _draw_rule_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    title: str,
    positive_island: str,
    negative_island: str,
    negative_label: str,
    positive_label: str,
    annotation: str,
) -> None:
    subset = frame[frame["visible_island"].isin([negative_island, positive_island])].copy()
    subset["actual_island"] = subset["visible_island"].eq(positive_island).astype(int)
    for value, color, label in (
        (0, NEGATIVE_COLOR, negative_label),
        (1, POSITIVE_COLOR, positive_label),
    ):
        selected = subset[subset["actual_island"].eq(value)]
        ax.scatter(
            selected["umap_1"],
            selected["umap_2"],
            s=25,
            c=color,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.3,
            label=f"{label} (n={len(selected):,})",
            rasterized=True,
        )
    mismatches = subset[subset["best_simple_rule_correct"].eq(False)]  # noqa: E712
    ax.scatter(
        mismatches["umap_1"],
        mismatches["umap_2"],
        s=43,
        facecolors="none",
        edgecolors=INK,
        linewidths=0.7,
        label=f"rule mismatch (n={len(mismatches):,})",
        rasterized=True,
    )
    _style_axis(ax)
    x_min, x_max = subset["umap_1"].agg(["min", "max"])
    y_min, y_max = subset["umap_2"].agg(["min", "max"])
    x_pad = max((x_max - x_min) * 0.05, 0.08)
    y_pad = max((y_max - y_min) * 0.05, 0.08)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("auto")
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", weight="semibold", pad=8)
    ax.legend(
        loc="best",
        fontsize=7.8,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        labelcolor=MUTED,
    )
    ax.text(
        0.01,
        -0.10,
        annotation,
        transform=ax.transAxes,
        fontsize=8.3,
        color=MUTED,
        va="top",
        clip_on=False,
    )


def _draw_specimen_role_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    subcohort: str,
    panel_letter: str,
) -> None:
    subset = frame[
        frame["subcohort"].eq(subcohort) & frame["specimen_role"].isin(["primary", "metastatic"])
    ].copy()
    for role, color, label in (
        ("primary", NEGATIVE_COLOR, "Primary"),
        ("metastatic", POSITIVE_COLOR, "Metastatic"),
    ):
        selected = subset[subset["specimen_role"].eq(role)]
        ax.scatter(
            selected["umap_1"],
            selected["umap_2"],
            s=25,
            c=color,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.3,
            label=f"{label} (n={len(selected):,})",
            rasterized=True,
        )
    _style_axis(ax)
    x_min, x_max = subset["umap_1"].agg(["min", "max"])
    y_min, y_max = subset["umap_2"].agg(["min", "max"])
    x_pad = max((x_max - x_min) * 0.05, 0.08)
    y_pad = max((y_max - y_min) * 0.05, 0.08)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("auto")
    display_name = "RIH" if subcohort == "RIH-Colon" else subcohort
    ax.set_title(
        f"{panel_letter}  {display_name} · specimen role · n={len(subset):,}",
        fontsize=11.5,
        color=INK,
        loc="left",
        weight="semibold",
        pad=8,
    )
    ax.legend(
        loc="best",
        fontsize=7.8,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        labelcolor=MUTED,
    )
    role_counts = EXPECTED_ROLE_COUNTS[subcohort]
    annotation = (
        f"Manifest specimen_role · {role_counts['primary']} primary / "
        f"{role_counts['metastatic']} metastatic"
    )
    if subcohort == "RIH-Colon":
        annotation += (
            "\n84 additional encoded RIH slides lack governed role labels and are excluded"
        )
    else:
        annotation += "\nAll encoded SR1482 slides have governed role labels"
    ax.text(
        0.01,
        -0.10,
        annotation,
        transform=ax.transAxes,
        fontsize=8.3,
        color=MUTED,
        va="top",
        clip_on=False,
    )


def save_figure(
    overview: pd.DataFrame,
    assignments: pd.DataFrame,
    audit_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    fig = plt.figure(figsize=(20, 21), facecolor="white")
    grid = fig.add_gridspec(
        6,
        8,
        height_ratios=(1.0, 1.0, 0.30, 1.22, 0.18, 1.22),
        left=0.035,
        right=0.985,
        top=0.845,
        bottom=0.070,
        wspace=0.18,
        hspace=0.27,
    )

    global_ax = fig.add_subplot(grid[0:2, 0:2])
    _draw_three_factor_points(global_ax, overview, point_size=11)
    _style_axis(global_ax)
    _zoom_to_frame(global_ax, overview, padding=0.04)
    global_ax.set_title(
        "A  All 2,087 slides · global geometry",
        fontsize=12,
        color=INK,
        loc="left",
        weight="semibold",
        pad=9,
    )
    global_ax.set_xlabel("UMAP 1", fontsize=8.5, color=MUTED)
    global_ax.set_ylabel("UMAP 2", fontsize=8.5, color=MUTED)

    panel_letters = "BCDEFG"
    facet_links = {
        "TCGA-COAD": "MPP/calibration separation",
        "TCGA-READ": "MPP-associated separation",
        "SR386": "fixed MPP; residual color rule → H",
        "SR1482": "fill distinguishes tissue class → I",
        "RIH-Colon": "MPP color + tissue fill → J",
    }
    for index, subcohort in enumerate(SUBCOHORT_ORDER):
        ax = fig.add_subplot(grid[index // 3, 2 + 2 * (index % 3) : 4 + 2 * (index % 3)])
        subset = overview[overview["subcohort"].eq(subcohort)].copy()
        _draw_three_factor_points(ax, subset, point_size=22)
        _style_axis(ax)
        _zoom_to_frame(ax, subset)
        ax.set_title(
            f"{panel_letters[index]}  {subcohort} · n={len(subset):,}",
            fontsize=10.5,
            color=INK,
            loc="left",
            weight="semibold",
            pad=7,
        )
        if subcohort in facet_links:
            ax.text(
                0.02,
                0.02,
                facet_links[subcohort],
                transform=ax.transAxes,
                fontsize=7.5,
                color=MUTED,
                va="bottom",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                    "pad": 2,
                },
            )

    legend_ax = fig.add_subplot(grid[2, :])
    _overview_legends(legend_ax)

    rule_grid = grid[3, :].subgridspec(1, 3, wspace=0.16)
    metrics = audit_summary["rule_metrics"]
    categorical = audit_summary["categorical_tests"]
    sr386 = metrics["SR386"]
    sr1482 = metrics["SR1482"]
    rih = metrics["RIH fixed MPP"]
    sr1482_biopsy = categorical["sr1482_biopsy_text"]
    rih_biopsy = categorical["rih_raw_source_1_biopsy_known_only"]

    _draw_rule_panel(
        fig.add_subplot(rule_grid[0, 0]),
        assignments[assignments["subcohort"].eq("SR386")],
        title="H  SR386 · residual tissue-color signature · n=427",
        positive_island="cool_stain_island",
        negative_island="main_stain_island",
        negative_label="main-like",
        positive_label="cool/blue-shifted",
        annotation=(
            f"Median tissue hue ≤ {sr386['threshold']:.4f} · grouped OOF BA "
            f"{sr386['cv_balanced_accuracy']:.3f} · AUC {sr386['roc_auc_oriented']:.3f}\n"
            "Consistent with an unrecorded staining/acquisition batch"
        ),
    )
    _draw_rule_panel(
        fig.add_subplot(rule_grid[0, 1]),
        assignments[assignments["subcohort"].eq("SR1482")],
        title="I  SR1482 · biopsy/tissue-section size\nn=592 (1 outlier excluded)",
        positive_island="small_fragment_island",
        negative_island="large_section_island",
        negative_label="large/full-section",
        positive_label="small/fragment",
        annotation=(
            f"UNI patches ≤ {sr1482['threshold']:,.1f} · grouped OOF BA "
            f"{sr1482['cv_balanced_accuracy']:.3f} · AUC {sr1482['roc_auc_oriented']:.3f}\n"
            f"Biopsy wording: {sr1482_biopsy['table_rows_small_large_columns_biopsy_other'][0][0]}/149 "
            f"small vs {sr1482_biopsy['table_rows_small_large_columns_biopsy_other'][1][0]}/443 large"
        ),
    )
    _draw_rule_panel(
        fig.add_subplot(rule_grid[0, 2]),
        assignments[assignments["subcohort"].eq("RIH-Colon")],
        title="J  RIH · biopsy/section size\nn=265 at MPP 0.5016 (104 other-MPP excluded)",
        positive_island="small_fragment_island",
        negative_island="large_section_island",
        negative_label="large/full-section",
        positive_label="small/fragment",
        annotation=(
            f"UNI patches ≤ {rih['threshold']:,.1f} · grouped OOF BA "
            f"{rih['cv_balanced_accuracy']:.3f} · AUC {rih['roc_auc_oriented']:.3f}\n"
            f"Raw biopsy field: {rih_biopsy['table_rows_small_large_columns_biopsy_nonbiopsy'][0][0]}/58 "
            f"small vs {rih_biopsy['table_rows_small_large_columns_biopsy_nonbiopsy'][1][0]}/162 large known"
        ),
    )

    role_grid = grid[5, :].subgridspec(1, 2, wspace=0.16)
    _draw_specimen_role_panel(
        fig.add_subplot(role_grid[0, 0]),
        overview,
        subcohort="SR1482",
        panel_letter="K",
    )
    _draw_specimen_role_panel(
        fig.add_subplot(role_grid[0, 1]),
        overview,
        subcohort="RIH-Colon",
        panel_letter="L",
    )

    fig.suptitle(
        "UNI-v1 slide embeddings: subcohort, native MPP, tissue class, and specimen role",
        x=0.035,
        y=0.975,
        ha="left",
        fontsize=19,
        color=INK,
        weight="semibold",
    )
    fig.text(
        0.035,
        0.932,
        "A–G: shape = subcohort, color = native MPP, fill = audited tissue class "
        "(filled large section; hollow small biopsy/fragment). H–J show the tested rules; K–L separately show SR1482 and RIH primary versus metastatic specimens using the same UMAP coordinates.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    fig.text(
        0.035,
        0.885,
        "A–G  Joint view of subcohort, scan resolution, and audited tissue-section class",
        fontsize=12,
        color=INK,
        weight="semibold",
        ha="left",
    )
    fig.text(
        0.985,
        0.018,
        "Existing cosine UMAP · exact UNI-v1 slide-mean pooling · seed 42 · panels K–L are metadata overlays, not refitted UMAPs",
        fontsize=7.8,
        color=MUTED,
        ha="right",
    )

    png_path = output_dir / "umap_univ1_comprehensive_separation_rules.png"
    pdf_path = output_dir / "umap_univ1_comprehensive_separation_rules.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overview", type=Path, default=OVERVIEW_COORDINATES)
    parser.add_argument("--assignments", type=Path, default=AUDIT_ASSIGNMENTS)
    parser.add_argument("--audit-summary", type=Path, default=AUDIT_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Replace this script's outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overview, assignments, audit_summary = _load_inputs(
        args.overview,
        args.assignments,
        args.audit_summary,
        args.manifest,
    )
    figures = save_figure(overview, assignments, audit_summary, args.output_dir)
    summary = {
        "analysis": "univ1_comprehensive_separation_rules",
        "n_overview_slides": int(len(overview)),
        "n_residual_audit_slides": int(len(assignments)),
        "sources": {
            "overview": {"path": str(args.overview), "sha256": _sha256(args.overview)},
            "assignments": {
                "path": str(args.assignments),
                "sha256": _sha256(args.assignments),
            },
            "audit_summary": {
                "path": str(args.audit_summary),
                "sha256": _sha256(args.audit_summary),
            },
            "manifest": {
                "path": str(args.manifest),
                "sha256": _sha256(args.manifest),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "semantic_counts": EXPECTED_SEMANTIC_COUNTS,
        "tissue_class_counts": EXPECTED_TISSUE_CLASS_COUNTS,
        "sr1482_rih_specimen_role_counts": EXPECTED_ROLE_COUNTS,
        "rule_metrics": audit_summary["rule_metrics"],
        "figures": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in figures.items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Comprehensive UNI-v1 UMAP separation figure",
                "",
                "Panels A–G show all 2,087 slides with subcohort encoded by shape,",
                "native MPP by color, and audited tissue class by marker fill:",
                "large/full sections are filled and small biopsy/fragments are hollow.",
                "Tissue class is shown only for SR1482 and fixed-MPP RIH; it is not",
                "extrapolated to unaudited cohorts or RIH slides at other MPP values.",
                "Panels H–J show the audited residual rules:",
                "SR386 residual tissue color, SR1482 biopsy/tissue-section size, and RIH",
                "biopsy/tissue-section size after restricting native MPP to 0.5016.",
                "Panel K overlays specimen_role for SR1482 only: 483 primary and 110",
                "metastatic slides. Panel L separately overlays RIH: 186 primary and 99",
                "metastatic slides. The 84 additional encoded RIH slides without governed",
                "specimen-role metadata are excluded from L.",
                "All panels reuse the frozen UMAP coordinates; no embedding was refit.",
                "",
                f"- PNG: `{figures['png']}`",
                f"- PDF: `{figures['pdf']}`",
                "",
            ]
        )
    )
    print(f"Wrote comprehensive UNI-v1 figure to {args.output_dir}")


if __name__ == "__main__":
    main()
