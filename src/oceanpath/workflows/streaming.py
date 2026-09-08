#!/usr/bin/env python3
"""Operate the continuous colon download/repair-to-encoding queue."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from oceanpath.extraction.inventory import InventoryPaths
from oceanpath.extraction.stream_encoder import SlideEncoderConfig
from oceanpath.extraction.watcher import (
    EncodingWatcher,
    WatcherConfig,
    install_signal_handlers,
    watcher_status,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SLIDE_ROOT = Path("/mnt/d/YC.Liu/slides/colon")
DEFAULT_MANIFEST_ROOT = Path("/mnt/d/YC.Liu/manifests/colon")
DEFAULT_FEATURE_ROOT = Path("/mnt/wsl/oceanpath-hot/features/colon_stream")
DEFAULT_STATE_ROOT = REPO_ROOT / "outputs" / "colon_stream"
DEFAULT_SCRATCH_ROOT = Path("/mnt/wsl/oceanpath-hot/scratch/colon_stream")
DEFAULT_HEST = Path("/home/yc_liu/.cache/trident/deeplabv3_seg_v4.ckpt")
DEFAULT_UNI = Path(
    "/home/yc_liu/.cache/huggingface/hub/models--MahmoodLab--uni/snapshots/"
    "b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin"
)
DEFAULT_CONCH = Path(
    "/home/yc_liu/.cache/huggingface/hub/models--MahmoodLab--conchv1_5/snapshots/"
    "3e5766a5d1500d53c73c03005e24c30c1f27be13/pytorch_model_vision.bin"
)
# Official ``pytorch_model.bin`` from paige-ai/Virchow2 revision
# 3158645804b69e3f3bc4439d4116edddf0840a72 (SHA-256 14244fbaa540...).
DEFAULT_VIRCHOW2 = Path("/mnt/wsl/oceanpath-hot/models/virchow2/pytorch_model.bin")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "run", "launch", "status", "stop"))
    parser.add_argument("--slide-root", type=Path, default=DEFAULT_SLIDE_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--hest-checkpoint", type=Path, default=DEFAULT_HEST)
    parser.add_argument("--uni-checkpoint", type=Path, default=DEFAULT_UNI)
    parser.add_argument("--conch-v15-checkpoint", type=Path, default=DEFAULT_CONCH)
    parser.add_argument("--virchow2-checkpoint", type=Path, default=DEFAULT_VIRCHOW2)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-slides", type=int)
    parser.add_argument("--seg-batch-size", type=int, default=64)
    parser.add_argument("--uni-batch-size", type=int, default=256)
    parser.add_argument("--conch-batch-size", type=int, default=128)
    parser.add_argument("--virchow2-batch-size", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument(
        "--frozen-inventory",
        action="store_true",
        help="Audit once at startup, then disable producer polling for a finalized cohort.",
    )
    parser.add_argument(
        "--remove-holes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve HEST mask holes as non-tissue (production default).",
    )
    parser.add_argument("--no-local-stage", action="store_true")
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=3,
        help="Number of future slides to stage with one serial copy worker.",
    )
    parser.add_argument(
        "--prefetch-max-gib",
        type=float,
        default=64.0,
        help="Maximum aggregate bytes reserved by current plus prefetched slides.",
    )
    parser.add_argument(
        "--prefetch-copy-workers",
        type=int,
        default=1,
        help="Concurrent prefetch copy streams from the slide root.",
    )
    parser.add_argument(
        "--local-job-output",
        action="store_true",
        help="Run TRIDENT and validation against local scratch, then publish "
        "each finished slide's artifacts to the feature root.",
    )
    return parser


def _config(args: argparse.Namespace) -> WatcherConfig:
    state_root = args.state_root.expanduser().resolve(strict=False)
    manifest_root = args.manifest_root.expanduser().resolve(strict=False)
    encoder = SlideEncoderConfig(
        output_root=args.feature_root.expanduser().resolve(strict=False),
        state_root=state_root,
        scratch_root=args.scratch_root.expanduser().resolve(strict=False),
        hest_checkpoint_path=args.hest_checkpoint.expanduser().resolve(strict=False),
        uni_checkpoint_path=args.uni_checkpoint.expanduser().resolve(strict=False),
        conch_v15_checkpoint_path=args.conch_v15_checkpoint.expanduser().resolve(strict=False),
        virchow2_checkpoint_path=args.virchow2_checkpoint.expanduser().resolve(strict=False),
        segmentation_batch_size=args.seg_batch_size,
        uni_batch_size=args.uni_batch_size,
        conch_v15_batch_size=args.conch_batch_size,
        virchow2_batch_size=args.virchow2_batch_size,
        max_workers=args.max_workers,
        remove_holes=args.remove_holes,
        stage_locally=not args.no_local_stage,
    )
    return WatcherConfig(
        inventory=InventoryPaths.from_roots(args.slide_root, manifest_root),
        encoder=encoder,
        mpp_sheet_path=manifest_root / "colon_ready_wsi_mpp.csv",
        inventory_status_path=manifest_root / "colon_ready_inventory.csv",
        queue_status_path=manifest_root / "colon_encoding_queue.csv",
        database_path=state_root / "queue.sqlite",
        lock_path=state_root / "watcher.lock",
        poll_seconds=args.poll_seconds,
        max_attempts=args.max_attempts,
        continuous_discovery=not args.frozen_inventory,
        prefetch_depth=0 if args.no_local_stage else args.prefetch_depth,
        prefetch_max_gib=args.prefetch_max_gib,
        prefetch_copy_workers=args.prefetch_copy_workers,
        local_job_output=args.local_job_output,
    )


def _common_child_arguments(args: argparse.Namespace) -> list[str]:
    values = [
        "--slide-root",
        str(args.slide_root),
        "--manifest-root",
        str(args.manifest_root),
        "--feature-root",
        str(args.feature_root),
        "--state-root",
        str(args.state_root),
        "--scratch-root",
        str(args.scratch_root),
        "--hest-checkpoint",
        str(args.hest_checkpoint),
        "--uni-checkpoint",
        str(args.uni_checkpoint),
        "--conch-v15-checkpoint",
        str(args.conch_v15_checkpoint),
        "--virchow2-checkpoint",
        str(args.virchow2_checkpoint),
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-attempts",
        str(args.max_attempts),
        "--seg-batch-size",
        str(args.seg_batch_size),
        "--uni-batch-size",
        str(args.uni_batch_size),
        "--conch-batch-size",
        str(args.conch_batch_size),
        "--virchow2-batch-size",
        str(args.virchow2_batch_size),
        "--max-workers",
        str(args.max_workers),
        "--prefetch-depth",
        str(args.prefetch_depth),
        "--prefetch-max-gib",
        str(args.prefetch_max_gib),
        "--prefetch-copy-workers",
        str(args.prefetch_copy_workers),
        "--remove-holes" if args.remove_holes else "--no-remove-holes",
    ]
    if args.no_local_stage:
        values.append("--no-local-stage")
    if args.frozen_inventory:
        values.append("--frozen-inventory")
    if args.local_job_output:
        values.append("--local-job-output")
    return values


def _launch(args: argparse.Namespace, cfg: WatcherConfig) -> int:
    current = watcher_status(cfg)
    if current["running"]:
        print(json.dumps(current, indent=2))
        return 0
    cfg.encoder.state_root.mkdir(parents=True, exist_ok=True)
    log_path = cfg.encoder.state_root / "watcher.log"
    child = [sys.executable, str(Path(__file__).resolve()), "run", *_common_child_arguments(args)]
    if shutil.which("ionice") and shutil.which("nice"):
        child = ["ionice", "-c", "2", "-n", "5", "nice", "-n", "5", *child]
    environment = dict(os.environ)
    environment.update({"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            child,
            cwd=REPO_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Watcher exited during startup with code {process.returncode}; inspect {log_path}"
            )
        status = watcher_status(cfg)
        if status["running"]:
            status["log"] = str(log_path)
            print(json.dumps(status, indent=2))
            return 0
        time.sleep(1)
    raise RuntimeError(f"Watcher did not acquire its lock within 45 seconds; inspect {log_path}")


def _stop(cfg: WatcherConfig) -> int:
    status = watcher_status(cfg)
    if not status["running"] or status["pid"] is None:
        print(json.dumps(status, indent=2))
        return 0
    os.kill(int(status["pid"]), signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(int(status["pid"]), 0)
        except OSError:
            break
        time.sleep(0.5)
    print(json.dumps(watcher_status(cfg), indent=2))
    return 0


def main() -> int:
    args = _parser().parse_args()
    cfg = _config(args)
    if args.command == "status":
        print(json.dumps(watcher_status(cfg), indent=2))
        return 0
    if args.command == "stop":
        return _stop(cfg)
    if args.command == "launch":
        return _launch(args, cfg)

    import logging

    cfg.encoder.state_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    watcher = EncodingWatcher(cfg)
    if args.command == "inventory":
        result = watcher.refresh(reconcile=True)
        print(
            json.dumps(
                {"inventory": dict(result.summary), "queue": watcher.state.summary()}, indent=2
            )
        )
        return 0
    install_signal_handlers(watcher)
    summary = watcher.run(max_slides=args.max_slides)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
