#!/usr/bin/env python3
"""Plot CONCH v1.5 by subcohort, native MPP, and frozen tissue class."""

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
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from tools.visualization.univ1_umap_comprehensive import (  # noqa: E402
    EXPECTED_SEMANTIC_COUNTS,
    EXPECTED_TISSUE_CLASS_COUNTS,
    INK,
    MUTED,
    _draw_rule_panel,
    _draw_three_factor_points,
    _overview_legends,
)
from tools.visualization.univ1_umap_subcohort_mpp import (  # noqa: E402
    EXPECTED_MPP_BIN_COUNTS,
    EXPECTED_SUBCOHORT_COUNTS,
    SUBCOHORT_ORDER,
    _style_axis,
    _zoom_to_frame,
)

CONCH_COORDINATES = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/cohort_umap_all_slides/coordinates_conch_v15.parquet"
)
CONCH_SLIDE_MEANS = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/cohort_umap_all_slides/slide_means_conch_v15.npz"
)
METADATA_REFERENCE = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/univ1_umap_subcohort_mpp_clear/"
    "coordinates_univ1_subcohort_mpp.parquet"
)
AUDIT_ASSIGNMENTS = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/univ1_umap_separation_audit/"
    "slide_cluster_assignments.parquet"
)
AUDIT_SUMMARY = Path("/mnt/wsl/oceanpath-hot/outputs/eval/univ1_umap_separation_audit/summary.json")
OUTPUT_DIR = Path("/mnt/wsl/oceanpath-hot/outputs/eval/conch_v15_umap_comprehensive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_inputs(
    coordinates_path: Path,
    means_path: Path,
    metadata_path: Path,
    assignments_path: Path,
    audit_summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    coordinates = pd.read_parquet(coordinates_path)
    metadata = pd.read_parquet(metadata_path)
    assignments = pd.read_parquet(assignments_path)
    audit_summary = json.loads(audit_summary_path.read_text())

    required_coordinates = {"slide_id", "cohort", "n_patches", "umap_1", "umap_2"}
    required_metadata = {"slide_id", "subcohort", "mpp", "mpp_bin"}
    required_assignments = {
        "slide_id",
        "patient_uid",
        "subcohort",
        "visible_island",
        "thumbnail_hue_median",
    }
    for name, frame, required in (
        ("coordinates", coordinates, required_coordinates),
        ("metadata", metadata, required_metadata),
        ("assignments", assignments, required_assignments),
    ):
        if missing := required - set(frame.columns):
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        if frame["slide_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate slide IDs")

    if len(coordinates) != 2087 or len(metadata) != 2087:
        raise ValueError("CONCH coordinates and frozen metadata must each contain 2,087 slides")
    if set(coordinates["slide_id"].astype(str)) != set(metadata["slide_id"].astype(str)):
        raise ValueError("CONCH and metadata slide inventories differ")
    if len(assignments) != 1389:
        raise ValueError("Frozen audit assignments must contain 1,389 slides")
    if not np.isfinite(coordinates[["n_patches", "umap_1", "umap_2"]].to_numpy(dtype=float)).all():
        raise ValueError("CONCH coordinates contain non-finite values")

    overview = coordinates.merge(
        metadata[["slide_id", "subcohort", "mpp", "mpp_bin"]],
        on="slide_id",
        how="left",
        validate="one_to_one",
    )
    if overview["subcohort"].value_counts().to_dict() != EXPECTED_SUBCOHORT_COUNTS:
        raise ValueError("CONCH subcohort counts changed")
    observed_mpp = overview["mpp_bin"].value_counts(sort=False).to_dict()
    if observed_mpp != EXPECTED_MPP_BIN_COUNTS:
        raise ValueError(f"CONCH MPP-bin counts changed: {observed_mpp}")

    transferred = assignments.drop(
        columns=["umap_1", "umap_2", "n_patches"], errors="ignore"
    ).merge(
        coordinates[["slide_id", "n_patches", "umap_1", "umap_2"]],
        on="slide_id",
        how="inner",
        validate="one_to_one",
    )
    if len(transferred) != len(assignments):
        raise ValueError("Not every frozen audit label transfers to CONCH")
    subcohort_check = overview[["slide_id", "subcohort"]].merge(
        transferred[["slide_id", "subcohort"]],
        on="slide_id",
        suffixes=("_overview", "_audit"),
        validate="one_to_one",
    )
    if not subcohort_check["subcohort_overview"].eq(subcohort_check["subcohort_audit"]).all():
        raise ValueError("Transferred audit subcohort labels disagree with CONCH metadata")

    observed_semantic = {
        subcohort: subset["visible_island"].value_counts().astype(int).to_dict()
        for subcohort, subset in transferred.groupby("subcohort")
    }
    if observed_semantic != EXPECTED_SEMANTIC_COUNTS:
        raise ValueError(f"Frozen semantic counts changed: {observed_semantic}")

    tissue_lookup = transferred[["slide_id", "visible_island"]].copy()
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
    if overview["tissue_class"].value_counts().astype(int).to_dict() != (
        EXPECTED_TISSUE_CLASS_COUNTS
    ):
        raise ValueError("Transferred tissue-class counts changed")

    with np.load(means_path, allow_pickle=False) as archive:
        mean_slide_ids = archive["slide_ids"].astype(str)
        means = archive["means"].copy()
    if means.shape != (2087, 768) or not np.isfinite(means).all():
        raise ValueError(f"Invalid CONCH slide-mean matrix: {means.shape}")
    if len(np.unique(mean_slide_ids)) != 2087:
        raise ValueError("CONCH slide-mean archive contains duplicate slide IDs")
    if set(mean_slide_ids) != set(coordinates["slide_id"].astype(str)):
        raise ValueError("CONCH slide-mean and coordinate inventories differ")
    return overview, transferred, mean_slide_ids, means, audit_summary


def _rule_metrics(
    frame: pd.DataFrame,
    *,
    predictor: str,
    positive_island: str,
    negative_island: str,
) -> tuple[dict[str, Any], pd.Series]:
    binary = frame[frame["visible_island"].isin([positive_island, negative_island])].copy()
    if binary[predictor].isna().any():
        raise ValueError(f"Missing {predictor} in CONCH rule contrast")
    values = binary[predictor].to_numpy(dtype=float)
    target = binary["visible_island"].eq(positive_island).astype(int).to_numpy()
    groups = binary["patient_uid"].fillna(binary["slide_id"]).astype(str).to_numpy()

    model = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=42)
    model.fit(values[:, None], target)
    fitted = model.predict(values[:, None])
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
            values[:, None], target, groups
        )
    )
    oof = cross_val_predict(
        DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=42),
        values[:, None],
        target,
        cv=folds,
    )
    raw_auc = roc_auc_score(target, values)
    threshold = float(model.tree_.threshold[0])
    left_class = int(np.argmax(model.tree_.value[1][0]))
    operator = "<=" if left_class == 1 else ">"
    matrix = confusion_matrix(target, fitted, labels=[0, 1])
    metrics = {
        "predictor": predictor,
        "positive_island": positive_island,
        "negative_island": negative_island,
        "n": int(len(binary)),
        "n_patients": int(pd.Series(groups).nunique()),
        "threshold": threshold,
        "positive_rule_operator": operator,
        "roc_auc_oriented": float(max(raw_auc, 1.0 - raw_auc)),
        "balanced_accuracy": float(balanced_accuracy_score(target, fitted)),
        "cv_balanced_accuracy": float(balanced_accuracy_score(target, oof)),
        "confusion_matrix_actual_negative_positive": matrix.astype(int).tolist(),
    }
    correct = pd.Series(fitted == target, index=binary.index, dtype="boolean")
    return metrics, correct


def _embedding_validation(
    frame: pd.DataFrame,
    slide_ids: np.ndarray,
    means: np.ndarray,
    *,
    positive_island: str,
    negative_island: str,
) -> dict[str, float | int]:
    binary = frame[frame["visible_island"].isin([positive_island, negative_island])].copy()
    index = {slide_id: position for position, slide_id in enumerate(slide_ids)}
    embeddings = means[[index[slide_id] for slide_id in binary["slide_id"].astype(str)]].copy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("CONCH slide means contain invalid norms")
    embeddings /= norms
    target = binary["visible_island"].eq(positive_island).astype(int).to_numpy()
    groups = binary["patient_uid"].fillna(binary["slide_id"]).astype(str).to_numpy()
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
            embeddings, target, groups
        )
    )
    predictions = cross_val_predict(
        make_pipeline(
            PCA(n_components=50, random_state=42),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        ),
        embeddings,
        target,
        cv=folds,
    )
    pca_embeddings = PCA(n_components=50, random_state=42).fit_transform(embeddings)
    clusters = KMeans(n_clusters=2, n_init=50, random_state=42).fit_predict(pca_embeddings)
    return {
        "n": int(len(binary)),
        "n_patients": int(pd.Series(groups).nunique()),
        "umap_2d_silhouette": float(silhouette_score(binary[["umap_1", "umap_2"]], target)),
        "original_768d_cosine_silhouette": float(
            silhouette_score(embeddings, target, metric="cosine")
        ),
        "pca50_logistic_patient_grouped_oof_balanced_accuracy": float(
            balanced_accuracy_score(target, predictions)
        ),
        "pca50_kmeans_adjusted_rand_index": float(adjusted_rand_score(target, clusters)),
    }


def _prepare_metrics(
    transferred: pd.DataFrame,
    slide_ids: np.ndarray,
    means: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, float | int]]]:
    transferred = transferred.copy()
    transferred["best_simple_rule_correct"] = pd.Series(
        pd.NA, index=transferred.index, dtype="boolean"
    )
    specs = {
        "SR386": ("thumbnail_hue_median", "cool_stain_island", "main_stain_island"),
        "SR1482": ("n_patches", "small_fragment_island", "large_section_island"),
        "RIH fixed MPP": (
            "n_patches",
            "small_fragment_island",
            "large_section_island",
        ),
    }
    rules: dict[str, dict[str, Any]] = {}
    validation: dict[str, dict[str, float | int]] = {}
    for name, (predictor, positive, negative) in specs.items():
        if name == "SR386":
            contrast = transferred[transferred["subcohort"].eq("SR386")]
        elif name == "SR1482":
            contrast = transferred[transferred["subcohort"].eq("SR1482")]
        else:
            contrast = transferred[transferred["subcohort"].eq("RIH-Colon")]
        rules[name], correct = _rule_metrics(
            contrast,
            predictor=predictor,
            positive_island=positive,
            negative_island=negative,
        )
        transferred.loc[correct.index, "best_simple_rule_correct"] = correct
        validation[name] = _embedding_validation(
            contrast,
            slide_ids,
            means,
            positive_island=positive,
            negative_island=negative,
        )
    return transferred, rules, validation


def save_figure(
    overview: pd.DataFrame,
    transferred: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
    validation: dict[str, dict[str, float | int]],
    audit_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    fig = plt.figure(figsize=(20, 15.5), facecolor="white")
    grid = fig.add_gridspec(
        4,
        8,
        height_ratios=(1.0, 1.0, 0.30, 1.22),
        left=0.035,
        right=0.985,
        top=0.845,
        bottom=0.085,
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
        "TCGA-COAD": "native-MPP association",
        "TCGA-READ": "native-MPP association",
        "SR386": "frozen color groups → H",
        "SR1482": "tissue fill → I",
        "RIH-Colon": "MPP color + tissue fill → J",
    }
    for index, subcohort in enumerate(SUBCOHORT_ORDER):
        ax = fig.add_subplot(grid[index // 3, 2 + 2 * (index % 3) : 4 + 2 * (index % 3)])
        subset = overview[overview["subcohort"].eq(subcohort)]
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
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2},
            )

    legend_ax = fig.add_subplot(grid[2, :])
    _overview_legends(legend_ax)
    rule_grid = grid[3, :].subgridspec(1, 3, wspace=0.16)
    categorical = audit_summary["categorical_tests"]
    sr386 = rules["SR386"]
    sr1482 = rules["SR1482"]
    rih = rules["RIH fixed MPP"]
    sr1482_biopsy = categorical["sr1482_biopsy_text"]
    rih_biopsy = categorical["rih_raw_source_1_biopsy_known_only"]

    _draw_rule_panel(
        fig.add_subplot(rule_grid[0, 0]),
        transferred[transferred["subcohort"].eq("SR386")],
        title="H  SR386 · transferred tissue-color groups · n=427",
        positive_island="cool_stain_island",
        negative_island="main_stain_island",
        negative_label="main-like",
        positive_label="cool/blue-shifted",
        annotation=(
            f"Median tissue hue ≤ {sr386['threshold']:.4f} · grouped OOF BA "
            f"{sr386['cv_balanced_accuracy']:.3f} · CONCH UMAP silhouette "
            f"{validation['SR386']['umap_2d_silhouette']:.3f}\n"
            "Frozen external labels; CONCH coordinates fitted independently"
        ),
    )
    _draw_rule_panel(
        fig.add_subplot(rule_grid[0, 1]),
        transferred[transferred["subcohort"].eq("SR1482")],
        title="I  SR1482 · transferred biopsy/section class\nn=592 (1 outlier excluded)",
        positive_island="small_fragment_island",
        negative_island="large_section_island",
        negative_label="large/full-section",
        positive_label="small/fragment",
        annotation=(
            f"CONCH patches ≤ {sr1482['threshold']:,.1f} · grouped OOF BA "
            f"{sr1482['cv_balanced_accuracy']:.3f} · UMAP silhouette "
            f"{validation['SR1482']['umap_2d_silhouette']:.3f}\n"
            f"Biopsy wording: {sr1482_biopsy['table_rows_small_large_columns_biopsy_other'][0][0]}/149 "
            f"small vs {sr1482_biopsy['table_rows_small_large_columns_biopsy_other'][1][0]}/443 large"
        ),
    )
    _draw_rule_panel(
        fig.add_subplot(rule_grid[0, 2]),
        transferred[transferred["subcohort"].eq("RIH-Colon")],
        title="J  RIH · transferred biopsy/section class\nn=265 at MPP 0.5016 (104 other-MPP excluded)",
        positive_island="small_fragment_island",
        negative_island="large_section_island",
        negative_label="large/full-section",
        positive_label="small/fragment",
        annotation=(
            f"CONCH patches ≤ {rih['threshold']:,.1f} · grouped OOF BA "
            f"{rih['cv_balanced_accuracy']:.3f} · UMAP silhouette "
            f"{validation['RIH fixed MPP']['umap_2d_silhouette']:.3f}\n"
            f"Raw biopsy: {rih_biopsy['table_rows_small_large_columns_biopsy_nonbiopsy'][0][0]}/58 "
            f"small vs {rih_biopsy['table_rows_small_large_columns_biopsy_nonbiopsy'][1][0]}/162 large known"
        ),
    )

    fig.suptitle(
        "CONCH v1.5 slide embeddings: subcohort, native MPP, and tissue-section class",
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
        "Shape = subcohort, color = native/source MPP, fill = frozen audited tissue class. "
        "Labels transfer one-to-one by slide ID; the CONCH UMAP was independently fitted.",
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
        "CONCH v1.5 · 768-D · 512px tiles · target MPP 0.5 · exact slide means · cosine UMAP · seed 42",
        fontsize=7.8,
        color=MUTED,
        ha="right",
    )

    png_path = output_dir / "umap_conch_v15_comprehensive_separation_rules.png"
    pdf_path = output_dir / "umap_conch_v15_comprehensive_separation_rules.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, default=CONCH_COORDINATES)
    parser.add_argument("--slide-means", type=Path, default=CONCH_SLIDE_MEANS)
    parser.add_argument("--metadata-reference", type=Path, default=METADATA_REFERENCE)
    parser.add_argument("--assignments", type=Path, default=AUDIT_ASSIGNMENTS)
    parser.add_argument("--audit-summary", type=Path, default=AUDIT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Replace this script's outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview, transferred, slide_ids, means, audit_summary = _load_inputs(
        args.coordinates,
        args.slide_means,
        args.metadata_reference,
        args.assignments,
        args.audit_summary,
    )
    transferred, rules, validation = _prepare_metrics(transferred, slide_ids, means)
    figures = save_figure(
        overview,
        transferred,
        rules,
        validation,
        audit_summary,
        args.output_dir,
    )
    transferred_path = args.output_dir / "coordinates_conch_v15_with_frozen_labels.parquet"
    transferred.to_parquet(transferred_path, index=False)
    summary = {
        "analysis": "conch_v15_comprehensive_separation_rules",
        "n_slides": int(len(overview)),
        "encoder": {
            "name": "CONCH v1.5",
            "feature_dim": 768,
            "tile_size_px": 512,
            "target_mpp": 0.5,
        },
        "label_policy": (
            "Frozen UNI/source-grounded audit labels transferred one-to-one by slide_id; "
            "no CONCH reclustering used for assignment"
        ),
        "tissue_class_counts": EXPECTED_TISSUE_CLASS_COUNTS,
        "rule_metrics": rules,
        "embedding_validation": validation,
        "sources": {
            "coordinates": {"path": str(args.coordinates), "sha256": _sha256(args.coordinates)},
            "slide_means": {"path": str(args.slide_means), "sha256": _sha256(args.slide_means)},
            "metadata_reference": {
                "path": str(args.metadata_reference),
                "sha256": _sha256(args.metadata_reference),
            },
            "assignments": {
                "path": str(args.assignments),
                "sha256": _sha256(args.assignments),
            },
            "audit_summary": {
                "path": str(args.audit_summary),
                "sha256": _sha256(args.audit_summary),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "artifacts": {
            "transferred_coordinates": {
                "path": str(transferred_path),
                "sha256": _sha256(transferred_path),
            },
            "figures": {
                name: {"path": str(path), "sha256": _sha256(path)} for name, path in figures.items()
            },
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(
        "\n".join(
            [
                "# CONCH v1.5 UMAP: subcohort, MPP, and tissue-section class",
                "",
                "The independently fitted 2,087-slide CONCH UMAP uses the same visual",
                "encoding as UNI-v1: subcohort shape, native/source-MPP color, and",
                "audited tissue-class fill. Frozen labels transfer by slide ID and were",
                "not redefined from CONCH clustering.",
                "",
                f"- PNG: `{figures['png']}`",
                f"- PDF: `{figures['pdf']}`",
                "",
            ]
        )
    )
    print(f"Wrote comprehensive CONCH v1.5 figure to {args.output_dir}")


if __name__ == "__main__":
    main()
