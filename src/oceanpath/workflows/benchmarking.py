"""Loader benchmarking workflow: measure the training read path against a
packed feature store.

Answers "are we loader-bound or compute-bound?" in one command by measuring
slides/s through the exact Dataset -> Collator -> DataLoader -> (optional H2D)
stack that training uses, across worker counts and store residency modes.

Usage:
    python scripts/bench_loader.py --pack-dir /path/to/packed_uni_v1
    python scripts/bench_loader.py --pack-dir ... --workers 0 4 --resident cuda
    python scripts/bench_loader.py --pack-dir ... --fixed-bag-size 2048 --batch-size 32

Interpretation: if slides/s stops improving between worker settings and far
exceeds the model's training-step throughput, the loader is not the
bottleneck and further loader work buys nothing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _build_dataset(store, feature_dir: Path, args):
    from oceanpath.datasets.datamodule import SlideDataset

    return SlideDataset(
        feature_dir=str(feature_dir),
        slide_ids=None,
        labels={},
        is_train=True,
        cap_strategy=args.cap_strategy,
        store=store,
        fixed_bag_size=args.fixed_bag_size,
        short_bag_policy="repeat",
        force_float32=False,
    )


def _bench_one(dataset, feat_dim: int, args, num_workers: int, device) -> dict:
    from oceanpath.datasets.datamodule import MILCollator

    collator = MILCollator(
        max_instances=args.fixed_bag_size,
        feat_dim=feat_dim,
        batch_size=args.batch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda" and args.resident is None),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    if len(loader) == 0:
        raise ValueError(
            f"DataLoader yields no batches: {len(dataset)} slides with "
            f"batch_size={args.batch_size} and drop_last=True."
        )

    def run(n_batches: int) -> tuple[float, int]:
        """Consume exactly n_batches, cycling the loader if the split is small.

        Returns (elapsed, batches_actually_consumed). Counting delivered
        batches rather than assuming n_batches matters whenever the dataset
        holds fewer than n_batches worth of slides — otherwise a short epoch
        is timed but a full one is reported, inflating throughput.
        """
        done = 0
        start = time.perf_counter()
        while done < n_batches:
            for batch in loader:
                feats = batch["features"]
                if device.type == "cuda":
                    feats = feats.to(device, non_blocking=True)
                    # A tiny reduction forces the copy to complete so we measure
                    # delivered bandwidth, not enqueue time.
                    _ = feats[..., 0].float().sum().item()
                done += 1
                if done >= n_batches:
                    break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - start, done

    run(args.warmup_batches)  # warm page cache / workers / CUDA context
    elapsed, n_done = run(args.batches)
    slides = n_done * args.batch_size
    mb = slides * (args.fixed_bag_size or 0) * feat_dim * 2 / 1e6  # fp16 bytes
    return {
        "num_workers": num_workers,
        "slides_per_s": slides / elapsed,
        "batches_per_s": n_done / elapsed,
        "mb_per_s": mb / elapsed if mb else float("nan"),
        "elapsed_s": elapsed,
        "n_batches": n_done,
    }


def run_loader_benchmark(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", required=True, type=Path)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=None,
        help="Source H5 dir (default: the pack's recorded source_dir)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fixed-bag-size", type=int, default=2048)
    parser.add_argument("--cap-strategy", choices=["random", "spatial_stratified"], default="random")
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4])
    parser.add_argument(
        "--resident",
        choices=["cuda", "cpu"],
        default=None,
        help="Also/only bench a resident store on this device (forces workers=0)",
    )
    parser.add_argument(
        "--h5",
        action="store_true",
        help="Also bench the per-slide H5 lane (store=None) over the same worker counts. "
        "For a fair format-only comparison the H5 files must already be float16.",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Read every source byte before measuring so both lanes start page-cache warm.",
    )
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    from oceanpath.datasets.packed import PackedFeatureStore, ResidentPackedStore

    device = torch.device(args.device)
    base = PackedFeatureStore(args.pack_dir)
    feature_dir = args.feature_dir or Path(base.meta.source_dir)
    print(
        f"pack: {args.pack_dir} | {base.meta.n_slides} slides, "
        f"{base.meta.total_patches} patches, D={base.meta.feat_dim}, "
        f"{base.meta.feat_dtype} | bag={args.fixed_bag_size} bs={args.batch_size} "
        f"| measuring {args.batches} batches after {args.warmup_batches} warmup"
    )

    if args.warm_cache:
        warmed = 0
        for path in [*sorted(feature_dir.glob("*.h5")), *sorted(args.pack_dir.glob("*.bin"))]:
            with open(path, "rb") as handle:
                while handle.read(64 * 1024 * 1024):
                    pass
            warmed += 1
        print(f"warmed {warmed} file(s) into page cache")

    results = []
    if args.h5:
        h5_dataset = _build_dataset(None, feature_dir, args)
        for w in args.workers:
            r = _bench_one(h5_dataset, h5_dataset.feat_dim, args, w, device)
            r["store"] = "h5"
            results.append(r)
            print(
                f"h5       workers={w}: {r['slides_per_s']:7.1f} slides/s "
                f"({r['mb_per_s']:.0f} MB/s, {r['batches_per_s']:.2f} batches/s)"
            )

    if args.resident is None:
        dataset = _build_dataset(base, feature_dir, args)
        for w in args.workers:
            r = _bench_one(dataset, base.feat_dim, args, w, device)
            r["store"] = "memmap"
            results.append(r)
            print(
                f"memmap   workers={w}: {r['slides_per_s']:7.1f} slides/s "
                f"({r['mb_per_s']:.0f} MB/s, {r['batches_per_s']:.2f} batches/s)"
            )
    else:
        resident = ResidentPackedStore(base, device=args.resident)
        dataset = _build_dataset(resident, feature_dir, args)
        r = _bench_one(dataset, base.feat_dim, args, 0, device)
        r["store"] = f"resident-{args.resident}"
        results.append(r)
        print(
            f"resident-{args.resident} workers=0: {r['slides_per_s']:7.1f} slides/s "
            f"({r['mb_per_s']:.0f} MB/s, {r['batches_per_s']:.2f} batches/s)"
        )

    best = max(results, key=lambda r: r["slides_per_s"])
    print(f"\nbest: {best['store']} workers={best['num_workers']} @ {best['slides_per_s']:.1f} slides/s")
