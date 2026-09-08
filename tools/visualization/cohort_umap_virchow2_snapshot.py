#!/usr/bin/env python3
"""UMAP of a frozen snapshot of completed Virchow2 slide embeddings.

The encoder may continue writing new slides while this command runs. The
population is therefore frozen from queue rows marked ``complete`` at startup;
the actively processed slide and all pending slides are excluded. Every
selected H5 is validated and mean-pooled exactly over all tile embeddings.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.visualization.cohort_umap_all_slides import (  # noqa: E402
    COHORT_ORDER,
    MUTED,
    fit_umap,
    scatter_cohorts,
)

QUEUE_DB = Path("/home/yc_liu/projects/OceanPath-colon/outputs/colon_stream/queue.sqlite")
FEATURE_DIR = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/20x_224px_0px_overlap_mpp0.5/features_virchow2"
)
OUTPUT_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/eval/cohort_umap_virchow2_snapshots")
EXPECTED_DIM = 2560


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot_id() -> str:
    return dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def freeze_population(queue_db: Path) -> tuple[pd.DataFrame, str, str]:
    uri = f"file:{queue_db}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        population = pd.read_sql_query(
            """
            SELECT output_id AS slide_id, cohort, finished_at
            FROM stream_jobs
            WHERE status = 'complete'
            ORDER BY output_id
            """,
            connection,
        )

    frozen_at = dt.datetime.now().astimezone().isoformat()
    population["slide_id"] = population["slide_id"].astype(str)
    population["cohort"] = population["cohort"].astype(str)
    if population["slide_id"].duplicated().any():
        raise ValueError("Completed Virchow2 queue population has duplicate slide IDs")
    if not set(population["cohort"]).issubset(COHORT_ORDER):
        raise ValueError(
            f"Unexpected cohorts: {sorted(set(population['cohort']) - set(COHORT_ORDER))}"
        )
    canonical = population[["slide_id", "cohort"]].to_csv(index=False, lineterminator="\n")
    return population, frozen_at, _sha256_bytes(canonical.encode())


def validate_inventory(population: pd.DataFrame, feature_dir: Path) -> tuple[list[Path], str, int]:
    paths: list[Path] = []
    digest = hashlib.sha256()
    total_bytes = 0
    for slide_id in population["slide_id"]:
        path = feature_dir / f"{slide_id}.h5"
        if not path.is_file():
            raise FileNotFoundError(f"Completed queue slide lacks Virchow2 H5: {path}")
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        total_bytes += int(stat.st_size)
        paths.append(path)
    return paths, digest.hexdigest(), total_bytes


def exact_h5_means(
    paths: list[Path],
    *,
    block_rows: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty((len(paths), EXPECTED_DIM), dtype=np.float32)
    patch_counts = np.empty(len(paths), dtype=np.int64)

    for position, path in enumerate(paths):
        with h5py.File(path, "r") as handle:
            if "features" not in handle:
                raise ValueError(f"{path} lacks a 'features' dataset")
            dataset = handle["features"]
            if dataset.ndim != 2 or int(dataset.shape[1]) != EXPECTED_DIM:
                raise ValueError(
                    f"{path}:features has shape {dataset.shape}; expected (N, {EXPECTED_DIM})"
                )
            n_patches = int(dataset.shape[0])
            if n_patches <= 0:
                raise ValueError(f"{path} has no Virchow2 tile embeddings")
            accumulator = np.zeros(EXPECTED_DIM, dtype=np.float64)
            for start in range(0, n_patches, block_rows):
                block = np.asarray(dataset[start : start + block_rows], dtype=np.float32)
                accumulator += block.sum(axis=0, dtype=np.float64)
            means[position] = (accumulator / n_patches).astype(np.float32)
            patch_counts[position] = n_patches

        if (position + 1) % 100 == 0 or position + 1 == len(paths):
            print(f"  mean pooled {position + 1:,}/{len(paths):,} completed slides", flush=True)

    return means, patch_counts


def save_figure(frame: pd.DataFrame, output_dir: Path, *, frozen_at: str) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 7.0), facecolor="white")
    scatter_cohorts(
        ax,
        frame,
        title="Virchow2 — currently completed colon slides by cohort",
    )
    fig.text(
        0.01,
        0.012,
        "Exact mean of all 2,560-D tile embeddings · L2 normalization · cosine UMAP\n"
        f"Snapshot {frozen_at} · CPTAC not yet encoded",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    path = output_dir / "umap_virchow2_completed_by_cohort.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-db", type=Path, default=QUEUE_DB)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    from sklearn.metrics import silhouette_score

    args = parse_args()
    output_dir = args.output_dir or (args.output_root / f"snapshot_{_snapshot_id()}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty snapshot directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    population, frozen_at, population_sha256 = freeze_population(args.queue_db)
    paths, inventory_sha256, total_bytes = validate_inventory(population, args.feature_dir)
    population.to_parquet(output_dir / "population.parquet", index=False)

    cohort_counts = {cohort: int((population["cohort"] == cohort).sum()) for cohort in COHORT_ORDER}
    print(f"Frozen completed population at {frozen_at}: {cohort_counts}", flush=True)
    means, patch_counts = exact_h5_means(paths)
    np.savez_compressed(
        output_dir / "slide_means_virchow2.npz",
        slide_ids=population["slide_id"].to_numpy(dtype=str),
        means=means,
        patch_counts=patch_counts,
        population_sha256=np.asarray(population_sha256),
        feature_inventory_sha256=np.asarray(inventory_sha256),
    )

    coordinates = fit_umap(means, seed=args.seed)
    frame = population[["slide_id", "cohort"]].copy()
    frame["encoder"] = "virchow2"
    frame["n_patches"] = patch_counts
    frame["umap_1"] = coordinates[:, 0]
    frame["umap_2"] = coordinates[:, 1]
    frame.to_parquet(output_dir / "coordinates_virchow2.parquet", index=False)
    figure = save_figure(frame, output_dir, frozen_at=frozen_at)

    silhouette = float(silhouette_score(coordinates, frame["cohort"].to_numpy()))
    summary = {
        "analysis": "completed_virchow2_cohort_umap_snapshot",
        "frozen_at": frozen_at,
        "population_source": str(args.queue_db),
        "population_sha256": population_sha256,
        "feature_dir": str(args.feature_dir),
        "feature_inventory_sha256": inventory_sha256,
        "feature_inventory_bytes": total_bytes,
        "n_slides": len(frame),
        "cohort_counts": cohort_counts,
        "feature_dim": EXPECTED_DIM,
        "total_patches": int(patch_counts.sum()),
        "umap_2d_cohort_silhouette": silhouette,
        "parameters": {
            "slide_pooling": "exact arithmetic mean of every tile embedding",
            "preprocessing": "per-slide L2 normalization",
            "method": "UMAP",
            "metric": "cosine",
            "n_neighbors": 30,
            "min_dist": 0.1,
            "seed": args.seed,
        },
        "figure": str(figure),
        "limitations": [
            "Snapshot includes only queue rows complete at frozen_at.",
            "CPTAC has no completed Virchow2 slides in this snapshot.",
            "Incomplete cohort composition prevents direct quantitative comparison with all-slide UNI/CONCH maps.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Virchow2 cohort UMAP — completed-slide snapshot",
                "",
                f"Frozen at: {frozen_at}",
                f"Slides: {len(frame):,}; tile embeddings: {int(patch_counts.sum()):,}; D={EXPECTED_DIM:,}.",
                "",
                "| Cohort | Completed slides |",
                "|---|---:|",
                *[f"| {cohort} | {cohort_counts[cohort]:,} |" for cohort in COHORT_ORDER],
                "",
                f"Descriptive 2D cohort silhouette: {silhouette:.3f}.",
                "",
                "CPTAC has not been encoded yet. This current snapshot is not directly",
                "comparable to the complete 2,087-slide UNI/CONCH maps.",
                "",
            ]
        )
    )
    print(f"Wrote Virchow2 snapshot UMAP to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
