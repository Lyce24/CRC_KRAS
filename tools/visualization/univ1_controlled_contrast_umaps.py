#!/usr/bin/env python3
"""Controlled UNI-v1 cohort/scanner/specimen-role contrast UMAPs.

Four independently fitted UMAPs isolate scanner calibration, collection
episode/stain, and specimen role while enforcing predeclared metadata and
group counts. Slide vectors are the exact all-tile UNI-v1 means previously
computed from the complete 2,087-slide packed store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.visualization.cohort_umap_all_slides import MUTED, fit_umap  # noqa: E402

MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v4.csv")
MEANS_CACHE = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/cohort_umap_all_slides/slide_means_univ1.npz"
)
ALL_SLIDE_SUMMARY = Path("/mnt/wsl/oceanpath-hot/outputs/eval/cohort_umap_all_slides/summary.json")
PACK_INDEX = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/"
    "20x_256px_0px_overlap_mpp0.5/packed_uni_v1/index.parquet"
)
OUTPUT_DIR = Path("/mnt/wsl/oceanpath-hot/outputs/eval/univ1_controlled_contrast_umaps")

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#1a1a19"


@dataclass(frozen=True)
class ContrastSpec:
    key: str
    title: str
    held_fixed: str
    isolates: str
    order: tuple[str, str]
    palette: tuple[str, str]
    expected_counts: tuple[int, int]


SPECS = {
    "A": ContrastSpec(
        key="A",
        title="TCGA-COAD primary: MPP 0.2325 vs ≈0.252",
        held_fixed="cohort, subcohort, format, role",
        isolates="scanner calibration alone",
        order=("MPP 0.2325", "MPP ≈0.252"),
        palette=(BLUE, ORANGE),
        expected_counts=(159, 245),
    ),
    "B": ContrastSpec(
        key="B",
        title="SurGen primary: SR1482 vs SR386",
        held_fixed="cohort, MPP (0.250), format, role",
        isolates="collection episode / stain alone",
        order=("SR1482", "SR386"),
        palette=(ORANGE, BLUE),
        expected_counts=(483, 427),
    ),
    "C": ContrastSpec(
        key="C",
        title="SR1482: primary vs metastatic",
        held_fixed="cohort, subcohort, MPP (0.250), format",
        isolates="specimen role alone",
        order=("Primary", "Metastatic"),
        palette=(BLUE, ORANGE),
        expected_counts=(483, 110),
    ),
    "D": ContrastSpec(
        key="D",
        title="RIH: primary vs metastatic at MPP 0.502",
        held_fixed="cohort, subcohort, MPP (0.5016), format",
        isolates="specimen role, replicated",
        order=("Primary", "Metastatic"),
        palette=(BLUE, ORANGE),
        expected_counts=(138, 66),
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _base_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "output_id",
        "cohort",
        "subcohort",
        "specimen_role",
        "mpp",
        "image_format",
        "used",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest lacks required columns: {sorted(missing)}")
    if frame["output_id"].astype(str).duplicated().any():
        raise ValueError("Manifest contains duplicate output_id values")
    frame = frame[frame["used"].eq("yes")].copy()
    frame["slide_id"] = frame["output_id"].astype(str)
    return frame


def _with_common_columns(frame: pd.DataFrame, *, key: str, group: pd.Series) -> pd.DataFrame:
    selected = frame[
        ["slide_id", "cohort", "subcohort", "specimen_role", "mpp", "image_format"]
    ].copy()
    selected["contrast"] = key
    selected["group"] = group.astype(str).to_numpy()
    return selected.sort_values("slide_id", kind="stable").reset_index(drop=True)


def build_contrasts(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    contrasts: dict[str, pd.DataFrame] = {}

    # A: the requested 245-slide arm is the tight observed scanner band near
    # 0.252, comprising 0.2520, 0.2525, 0.2527, and 0.2533 µm/px.
    a_base = manifest[
        manifest["cohort"].eq("TCGA")
        & manifest["subcohort"].eq("TCGA-COAD")
        & manifest["specimen_role"].eq("primary")
        & manifest["image_format"].eq(".svs")
    ]
    a_low = a_base[np.isclose(a_base["mpp"], 0.2325)].copy()
    a_low["_group"] = "MPP 0.2325"
    a_high = a_base[a_base["mpp"].between(0.251, 0.254)].copy()
    a_high["_group"] = "MPP ≈0.252"
    a = pd.concat([a_low, a_high], ignore_index=True)
    contrasts["A"] = _with_common_columns(a, key="A", group=a["_group"])

    b = manifest[
        manifest["cohort"].eq("SurGen")
        & manifest["specimen_role"].eq("primary")
        & manifest["image_format"].eq(".tiff")
        & np.isclose(manifest["mpp"], 0.250)
        & manifest["subcohort"].isin(["SR1482", "SR386"])
    ].copy()
    contrasts["B"] = _with_common_columns(b, key="B", group=b["subcohort"])

    c = manifest[
        manifest["cohort"].eq("SurGen")
        & manifest["subcohort"].eq("SR1482")
        & manifest["image_format"].eq(".tiff")
        & np.isclose(manifest["mpp"], 0.250)
        & manifest["specimen_role"].isin(["primary", "metastatic"])
    ].copy()
    c_group = c["specimen_role"].map({"primary": "Primary", "metastatic": "Metastatic"})
    contrasts["C"] = _with_common_columns(c, key="C", group=c_group)

    d = manifest[
        manifest["cohort"].eq("RIH")
        & manifest["subcohort"].eq("RIH-Colon")
        & manifest["image_format"].eq(".svs")
        & np.isclose(manifest["mpp"], 0.5016)
        & manifest["specimen_role"].isin(["primary", "metastatic"])
    ].copy()
    d_group = d["specimen_role"].map({"primary": "Primary", "metastatic": "Metastatic"})
    contrasts["D"] = _with_common_columns(d, key="D", group=d_group)

    for key, frame in contrasts.items():
        spec = SPECS[key]
        observed = tuple(int((frame["group"] == group).sum()) for group in spec.order)
        if observed != spec.expected_counts:
            raise ValueError(
                f"Contrast {key} count mismatch: observed {observed}, "
                f"expected {spec.expected_counts}"
            )
        if frame["slide_id"].duplicated().any():
            raise ValueError(f"Contrast {key} contains duplicate slide IDs")
    return contrasts


def load_means(cache_path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as cache:
        slide_ids = cache["slide_ids"].astype(str).tolist()
        means = cache["means"].astype(np.float32)
    if len(slide_ids) != len(set(slide_ids)):
        raise ValueError("UNI-v1 means cache contains duplicate slide IDs")
    if means.shape != (len(slide_ids), 1024):
        raise ValueError(f"Unexpected UNI-v1 means shape: {means.shape}")
    if not np.isfinite(means).all():
        raise ValueError("UNI-v1 means contain non-finite values")
    return slide_ids, means


def scatter_contrast(ax, frame: pd.DataFrame, spec: ContrastSpec) -> None:
    for group, color in zip(spec.order, spec.palette, strict=True):
        subset = frame[frame["group"] == group]
        ax.scatter(
            subset["umap_1"],
            subset["umap_2"],
            s=16,
            c=color,
            alpha=0.72,
            linewidths=0,
            label=f"{group} (n={len(subset):,})",
            rasterized=True,
        )
    ax.set_title(
        f"{spec.key} · {spec.title}\nHeld fixed: {spec.held_fixed}\nIsolates: {spec.isolates}",
        fontsize=10,
        color=INK,
        loc="left",
        linespacing=1.35,
    )
    ax.set_xlabel("UMAP 1", fontsize=9, color=MUTED)
    ax.set_ylabel("UMAP 2", fontsize=9, color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d8d7d0")
    ax.legend(
        loc="best",
        fontsize=8,
        frameon=False,
        handletextpad=0.25,
        markerscale=1.6,
        labelcolor=MUTED,
    )


def save_individual(frame: pd.DataFrame, spec: ContrastSpec, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 7.0), facecolor="white")
    scatter_contrast(ax, frame, spec)
    fig.text(
        0.01,
        0.01,
        "UNI-v1 exact slide means · per-slide L2 normalization · independent cosine UMAP",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = (
        output_dir
        / f"umap_{spec.key.lower()}_{spec.isolates.replace(' ', '_').replace('/', 'or')}.png"
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_combined(frames: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 13.0), facecolor="white")
    for ax, key in zip(axes.flat, SPECS, strict=True):
        scatter_contrast(ax, frames[key], SPECS[key])
    fig.suptitle(
        "UNI-v1 controlled cohort, scanner, stain, and specimen-role contrasts",
        fontsize=14,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.01,
        "Each panel is fitted independently on its stated, count-locked population · seed 42",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    path = output_dir / "umap_univ1_controlled_contrasts.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--means-cache", type=Path, default=MEANS_CACHE)
    parser.add_argument("--all-slide-summary", type=Path, default=ALL_SLIDE_SUMMARY)
    parser.add_argument("--pack-index", type=Path, default=PACK_INDEX)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    from sklearn.metrics import silhouette_score

    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _base_manifest(args.manifest)
    contrasts = build_contrasts(manifest)
    mean_ids, means = load_means(args.means_cache)
    mean_position = {slide_id: position for position, slide_id in enumerate(mean_ids)}
    pack_index = pd.read_parquet(args.pack_index)
    patch_counts = dict(
        zip(
            pack_index["slide_id"].astype(str),
            pack_index["n_patches"].astype(int),
            strict=True,
        )
    )

    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}
    for key, membership in contrasts.items():
        missing = sorted(set(membership["slide_id"]) - set(mean_position))
        if missing:
            raise ValueError(f"Contrast {key} lacks {len(missing)} UNI-v1 means: {missing[:5]}")
        positions = [mean_position[slide_id] for slide_id in membership["slide_id"]]
        coordinates = fit_umap(means[positions], seed=args.seed)
        frame = membership.copy()
        frame["n_patches"] = frame["slide_id"].map(patch_counts).astype(int)
        frame["umap_1"] = coordinates[:, 0]
        frame["umap_2"] = coordinates[:, 1]
        frame.to_parquet(args.output_dir / f"coordinates_{key.lower()}.parquet", index=False)
        frames[key] = frame

        silhouette = float(silhouette_score(coordinates, frame["group"].to_numpy()))
        spec = SPECS[key]
        summaries[key] = {
            "title": spec.title,
            "held_fixed": spec.held_fixed,
            "isolates": spec.isolates,
            "group_counts": {group: int((frame["group"] == group).sum()) for group in spec.order},
            "n_slides": len(frame),
            "total_patches": int(frame["n_patches"].sum()),
            "umap_2d_group_silhouette": silhouette,
        }
        save_individual(frame, spec, args.output_dir)

    combined_coordinates = pd.concat(frames.values(), ignore_index=True)
    combined_coordinates.to_parquet(
        args.output_dir / "coordinates_all_contrasts.parquet", index=False
    )
    combined_figure = save_combined(frames, args.output_dir)
    all_slide_summary = json.loads(args.all_slide_summary.read_text())

    summary = {
        "analysis": "univ1_controlled_contrast_umaps",
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "means_cache": str(args.means_cache),
        "means_cache_sha256": _sha256_file(args.means_cache),
        "uni_source_inventory_sha256": all_slide_summary["encoders"]["univ1"]["pack_meta"][
            "source_inventory_sha256"
        ],
        "parameters": {
            "slide_pooling": "exact arithmetic mean of every tile embedding",
            "preprocessing": "per-slide L2 normalization",
            "method": "UMAP",
            "metric": "cosine",
            "n_neighbors": 30,
            "min_dist": 0.1,
            "seed": args.seed,
            "fits": "independent per contrast",
        },
        "contrast_A_mpp_definition": {
            "low": [0.2325],
            "approximately_0.252": [0.2520, 0.2525, 0.2527, 0.2533],
        },
        "contrasts": summaries,
        "combined_figure": str(combined_figure),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "# UNI-v1 controlled contrast UMAPs",
        "",
        "Each panel is fitted independently using exact all-tile slide means.",
        "",
        "| Contrast | Groups | n | Held fixed | Isolates | 2D silhouette |",
        "|---|---|---:|---|---|---:|",
    ]
    for key, spec in SPECS.items():
        item = summaries[key]
        counts = " vs ".join(str(item["group_counts"][group]) for group in spec.order)
        report.append(
            f"| {key} | {' vs '.join(spec.order)} | {counts} | {spec.held_fixed} | "
            f"{spec.isolates} | {item['umap_2d_group_silhouette']:.3f} |"
        )
    report.extend(
        [
            "",
            "Contrast A's ≈0.252 arm contains observed MPP values 0.2520, 0.2525,",
            "0.2527, and 0.2533, which reproduces the predeclared n=245.",
            "Silhouettes describe the nonlinear 2D layouts only and are not held-out",
            "estimates of separability in the full UNI-v1 feature space.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(f"Wrote UNI-v1 controlled contrast UMAPs to {args.output_dir}")


if __name__ == "__main__":
    main()
