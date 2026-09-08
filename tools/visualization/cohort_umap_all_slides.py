#!/usr/bin/env python3
"""Matched UNI-v1 and CONCH-v1.5 UMAPs for every encoded colon slide.

The population is the authoritative 2,087-slide production encoder queue,
not a task-specific label manifest. Each slide is represented by the exact
mean of all of its tile embeddings from the frozen packed store. Slide means
are L2-normalized before fitting a deterministic cosine-distance UMAP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.datasets.packed import validate_packed_dir  # noqa: E402

QUEUE_DB = Path("/home/yc_liu/projects/OceanPath-colon/outputs/colon_stream/queue.sqlite")
FEATURE_ROOT = Path("/mnt/wsl/oceanpath-hot/features/colon_stream")
DEFAULT_OUTPUT_DIR = Path("/mnt/wsl/oceanpath-hot/outputs/eval/cohort_umap_all_slides")

COHORT_ORDER = ("CPTAC", "TCGA", "SurGen", "RIH")
EXPECTED_COUNTS = {"CPTAC": 98, "TCGA": 600, "SurGen": 1020, "RIH": 369}
COHORT_COLORS = {
    "CPTAC": "#7b2cbf",
    "TCGA": "#2a78d6",
    "SurGen": "#eb6834",
    "RIH": "#1baf7a",
}
INK = "#1a1a19"
MUTED = "#6b6a62"


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    title: str
    pack_dir: Path


ENCODERS = (
    EncoderSpec(
        key="univ1",
        title="UNI-v1",
        pack_dir=FEATURE_ROOT / "20x_256px_0px_overlap_mpp0.5" / "packed_uni_v1",
    ),
    EncoderSpec(
        key="conch_v15",
        title="CONCH v1.5",
        pack_dir=FEATURE_ROOT / "20x_512px_0px_overlap_mpp0.5" / "packed_conch_v15",
    ),
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_population(queue_db: Path) -> tuple[pd.DataFrame, str]:
    """Load the stable slide/cohort identity from the live queue read-only."""
    uri = f"file:{queue_db}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        population = pd.read_sql_query(
            "SELECT output_id AS slide_id, cohort FROM stream_jobs", connection
        )

    population["slide_id"] = population["slide_id"].astype(str)
    population["cohort"] = population["cohort"].astype(str)
    population = population.sort_values("slide_id", kind="stable").reset_index(drop=True)

    if population["slide_id"].duplicated().any():
        duplicates = population.loc[population["slide_id"].duplicated(), "slide_id"].head()
        raise ValueError(f"Production queue has duplicate slide IDs: {duplicates.tolist()}")
    counts = population["cohort"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected production cohort counts: {counts} != {EXPECTED_COUNTS}")

    canonical = population.to_csv(index=False, lineterminator="\n").encode()
    return population, _sha256_bytes(canonical)


def _load_pack(pack_dir: Path) -> tuple[dict, pd.DataFrame]:
    validate_packed_dir(pack_dir)
    meta = json.loads((pack_dir / "meta.json").read_text())
    index = pd.read_parquet(pack_dir / "index.parquet")
    expected_columns = {"slide_id", "offset", "n_patches"}
    if not expected_columns.issubset(index.columns):
        raise ValueError(
            f"{pack_dir}/index.parquet lacks columns {sorted(expected_columns - set(index.columns))}"
        )
    if index["slide_id"].astype(str).duplicated().any():
        raise ValueError(f"Packed store has duplicate slide IDs: {pack_dir}")
    return meta, index


def _cache_matches(cache: Path, *, index_sha256: str, source_sha256: str) -> bool:
    if not cache.is_file():
        return False
    with np.load(cache, allow_pickle=False) as stored:
        return (
            str(stored["index_sha256"].item()) == index_sha256
            and str(stored["source_inventory_sha256"].item()) == source_sha256
        )


def slide_means(
    pack_dir: Path,
    cache: Path,
    *,
    block_rows: int = 4096,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Compute exact slide means from a packed FP16 store in bounded memory."""
    meta, index = _load_pack(pack_dir)
    index_sha256 = _sha256_file(pack_dir / "index.parquet")
    source_sha256 = str(meta["source_inventory_sha256"])

    if _cache_matches(cache, index_sha256=index_sha256, source_sha256=source_sha256):
        with np.load(cache, allow_pickle=False) as stored:
            cached_ids = stored["slide_ids"].astype(str)
            if cached_ids.tolist() == index["slide_id"].astype(str).tolist():
                return index, stored["means"].astype(np.float32), meta

    feat_dim = int(meta["feat_dim"])
    total_patches = int(meta["total_patches"])
    features = np.memmap(
        pack_dir / "features.bin",
        dtype=np.dtype(meta["feat_dtype"]),
        mode="r",
        shape=(total_patches, feat_dim),
    )
    means = np.empty((len(index), feat_dim), dtype=np.float32)

    for position, row in enumerate(index.itertuples(index=False)):
        start = int(row.offset)
        n_patches = int(row.n_patches)
        if n_patches <= 0:
            raise ValueError(f"Slide {row.slide_id!r} has no patches in {pack_dir}")
        accumulator = np.zeros(feat_dim, dtype=np.float64)
        for block_start in range(start, start + n_patches, block_rows):
            block_end = min(block_start + block_rows, start + n_patches)
            block = np.asarray(features[block_start:block_end], dtype=np.float32)
            accumulator += block.sum(axis=0, dtype=np.float64)
        means[position] = (accumulator / n_patches).astype(np.float32)

        if (position + 1) % 250 == 0 or position + 1 == len(index):
            print(f"  mean pooled {position + 1:,}/{len(index):,} slides from {pack_dir.name}")

    del features
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        slide_ids=index["slide_id"].astype(str).to_numpy(dtype=str),
        means=means,
        index_sha256=np.asarray(index_sha256),
        source_inventory_sha256=np.asarray(source_sha256),
    )
    return index, means, meta


def fit_umap(means: np.ndarray, *, seed: int) -> np.ndarray:
    import umap
    from sklearn.preprocessing import normalize

    if not np.isfinite(means).all():
        raise ValueError("Slide means contain non-finite values")
    normalized = normalize(means, norm="l2", axis=1, copy=True)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    )
    return np.asarray(reducer.fit_transform(normalized), dtype=np.float32)


def scatter_cohorts(ax, frame: pd.DataFrame, *, title: str) -> None:
    for cohort in COHORT_ORDER:
        subset = frame[frame["cohort"] == cohort]
        ax.scatter(
            subset["umap_1"],
            subset["umap_2"],
            s=13,
            c=COHORT_COLORS[cohort],
            alpha=0.72,
            linewidths=0,
            label=f"{cohort} (n={len(subset):,})",
            rasterized=True,
        )
    ax.set_title(title, fontsize=12, color=INK, loc="left", weight="semibold")
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
        markerscale=1.7,
        labelcolor=MUTED,
    )


def save_single_figure(frame: pd.DataFrame, spec: EncoderSpec, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 7.0), facecolor="white")
    scatter_cohorts(ax, frame, title=f"{spec.title} — all colon slides by cohort")
    fig.text(
        0.01,
        0.01,
        "Exact mean of all tile embeddings per slide · L2 normalization · cosine UMAP",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = output_dir / f"umap_{spec.key}_by_cohort.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_combined_figure(frames: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.8), facecolor="white")
    for ax, spec in zip(axes, ENCODERS, strict=True):
        scatter_cohorts(
            ax,
            frames[spec.key],
            title=f"{spec.title} — all colon slides by cohort",
        )
    fig.suptitle(
        "Matched cohort structure in slide-level foundation embeddings",
        fontsize=13,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.01,
        "n=2,087 slides per encoder · exact mean pooling · independently fitted UMAPs",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    path = output_dir / "umap_univ1_conch_by_cohort.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-db", type=Path, default=QUEUE_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    from sklearn.metrics import silhouette_score

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    population, population_sha256 = load_population(args.queue_db)
    population_ids = set(population["slide_id"])

    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}
    for spec in ENCODERS:
        print(f"Processing {spec.title}: {spec.pack_dir}")
        index, means, meta = slide_means(
            spec.pack_dir,
            args.output_dir / f"slide_means_{spec.key}.npz",
        )
        packed_ids = set(index["slide_id"].astype(str))
        if packed_ids != population_ids:
            raise ValueError(
                f"{spec.title} population mismatch: "
                f"{len(population_ids - packed_ids)} queue-only, "
                f"{len(packed_ids - population_ids)} pack-only"
            )

        coordinates = fit_umap(means, seed=args.seed)
        frame = index[["slide_id", "n_patches"]].copy()
        frame["slide_id"] = frame["slide_id"].astype(str)
        frame = frame.merge(population, on="slide_id", how="left", validate="one_to_one")
        frame["encoder"] = spec.key
        frame["umap_1"] = coordinates[:, 0]
        frame["umap_2"] = coordinates[:, 1]
        frame = frame[["slide_id", "cohort", "encoder", "n_patches", "umap_1", "umap_2"]]
        frame.to_parquet(args.output_dir / f"coordinates_{spec.key}.parquet", index=False)
        frames[spec.key] = frame

        silhouette = float(silhouette_score(coordinates, frame["cohort"].to_numpy()))
        summaries[spec.key] = {
            "title": spec.title,
            "pack_dir": str(spec.pack_dir),
            "pack_meta": meta,
            "index_sha256": _sha256_file(spec.pack_dir / "index.parquet"),
            "n_slides": len(frame),
            "feature_dim": int(meta["feat_dim"]),
            "total_patches": int(meta["total_patches"]),
            "umap_2d_cohort_silhouette": silhouette,
        }
        save_single_figure(frame, spec, args.output_dir)

    combined = pd.concat([frames[spec.key] for spec in ENCODERS], ignore_index=True)
    combined.to_parquet(args.output_dir / "coordinates_all.parquet", index=False)
    combined_figure = save_combined_figure(frames, args.output_dir)

    summary = {
        "analysis": "all_slide_cohort_umap",
        "population_source": str(args.queue_db),
        "population_sha256": population_sha256,
        "cohort_counts": EXPECTED_COUNTS,
        "n_slides": len(population),
        "parameters": {
            "slide_pooling": "exact arithmetic mean of every tile embedding",
            "preprocessing": "per-slide L2 normalization",
            "method": "UMAP",
            "metric": "cosine",
            "n_neighbors": 30,
            "min_dist": 0.1,
            "seed": args.seed,
        },
        "encoders": summaries,
        "combined_figure": str(combined_figure),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report_lines = [
        "# All-slide cohort UMAP",
        "",
        "Both panels use the same 2,087-slide production population. Coordinates are",
        "fitted independently per encoder and are not directly aligned across panels.",
        "",
        "| Encoder | Slides | Tile embeddings | D | 2D cohort silhouette |",
        "|---|---:|---:|---:|---:|",
    ]
    for spec in ENCODERS:
        item = summaries[spec.key]
        report_lines.append(
            f"| {spec.title} | {item['n_slides']:,} | {item['total_patches']:,} | "
            f"{item['feature_dim']:,} | {item['umap_2d_cohort_silhouette']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "Cohort counts: CPTAC 98, TCGA 600, SurGen 1,020, RIH 369.",
            "The silhouette is descriptive of the displayed nonlinear 2D layout; it is",
            "not a held-out estimate of cohort predictability in the full feature space.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report_lines))
    print(f"Wrote cohort UMAP artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
