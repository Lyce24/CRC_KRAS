#!/usr/bin/env python3
"""Rebuild all four colon packs after Orion takes the cohort to 2,128 slides.

Packs rebuilt, in order:

* ``packed_uni_v1``            2,128 x 1,024   (20x_256px)
* ``packed_conch_v15``         2,128 x   768   (20x_512px)
* ``packed_virchow2_full_2560``2,128 x 2,560   (20x_224px)  -- delegated
* ``packed_virchow2_cls_1280`` 2,128 x 1,280   (20x_224px)  -- delegated

The two Virchow2 packs are NOT rebuilt here. They are produced by
``tools/pack_virchow2_after_encoding.py``, which additionally proves the CLS
pack is the byte-identical float16 [0:1280] prefix of the full pack and refuses
to publish unless every slide receipt carries the pinned implementation hash.
Duplicating that gate would weaken it, so this program calls it.

This is a completion gate, not an encoder controller: it never starts or stops
the watcher, and it refuses to touch a pack while the queue still holds
pending, processing, retry, failed or blocked rows.

Usage::

    python tools/pack_orion_expanded.py check     # readiness, changes nothing
    python tools/pack_orion_expanded.py run       # preflight, then rebuild all four
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.datasets.packed import (  # noqa: E402
    feature_inventory_sha256,
    validate_packed_dir,
)

# The extraction/packing interpreter, matching how every existing pack in this
# tree was built (production venv, development sources on PYTHONPATH).
PACK_PYTHON = Path("/home/yc_liu/projects/OceanPath-colon/.venv/bin/python")
QUEUE_DB = Path("/home/yc_liu/projects/OceanPath-colon/outputs/colon_stream/queue.sqlite")
FEATURE_ROOT = Path("/mnt/wsl/oceanpath-hot/features/colon_stream")
RECEIPTS = FEATURE_ROOT / "_stream_receipts"

EXPECTED_SLIDES = 2_128          # 2,087 four-cohort study + 41 Orion H&E
ORION_SLIDES = 41

# Pins asserted before anything is published. Identical to the values the
# sealed four-cohort packs were built under; a mismatch means the new slides
# are not comparable with the old ones and must not enter the same store.
IMPLEMENTATION_SHA256 = "c186b18b06c755fead1c3f6ea9a7e8a9a2ce167e6b2a8c04525599d79fd482de"
CHECKPOINTS = {
    "hest": "4ddf8be82384544ae9ef9af7ad0cc1e8190c781b945eea9826c778f93d08fd5a",
    "uni_v1": "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40",
    "conch_v15": "0c2588cef394daab46cbd5b64a556bf1d5d032262492d81d0e0d19c3118a5520",
    "virchow2": "14244fbaa5409452f6a6ae01b5d2dc452f2399e2210e696d0f7ef04bb4838666",
}

# encoder config name -> (coords subdir, feature dim)
ENCODERS = {
    "univ1": ("20x_256px_0px_overlap_mpp0.5", "uni_v1", 1024),
    "conch_v15": ("20x_512px_0px_overlap_mpp0.5", "conch_v15", 768),
}
VIRCHOW2_LAYOUT = ("20x_224px_0px_overlap_mpp0.5", "virchow2", 2560)


def feature_dir(coords: str, name: str) -> Path:
    return FEATURE_ROOT / coords / f"features_{name}"


def pack_dir(coords: str, name: str) -> Path:
    return FEATURE_ROOT / coords / f"packed_{name}"


class GateError(RuntimeError):
    """A precondition for publishing failed; nothing was written."""


def queue_counts() -> dict[str, int]:
    if not QUEUE_DB.is_file():
        raise GateError(f"queue database missing: {QUEUE_DB}")
    with sqlite3.connect(f"file:{QUEUE_DB}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT status, COUNT(*) FROM stream_jobs GROUP BY status").fetchall()
    counts = {str(s): int(n) for s, n in rows}
    counts["total"] = sum(counts.values())
    return counts


def check_queue(counts: dict[str, int]) -> list[str]:
    problems = []
    if counts.get("total") != EXPECTED_SLIDES:
        problems.append(f"queue holds {counts.get('total')} rows, expected {EXPECTED_SLIDES}")
    busy = {k: counts.get(k, 0) for k in ("pending", "processing", "retry", "failed", "blocked")}
    if any(busy.values()):
        problems.append(f"queue not drained: {busy}")
    if counts.get("complete") != EXPECTED_SLIDES:
        problems.append(f"{counts.get('complete')} complete, expected {EXPECTED_SLIDES}")
    return problems


def check_features() -> list[str]:
    problems = []
    layouts = [(c, n) for c, n, _ in ENCODERS.values()] + [VIRCHOW2_LAYOUT[:2]]
    for coords, name in layouts:
        d = feature_dir(coords, name)
        n = len(list(d.glob("*.h5"))) if d.is_dir() else 0
        if n != EXPECTED_SLIDES:
            problems.append(f"{name}: {n} feature H5 files, expected {EXPECTED_SLIDES}")
    return problems


def check_orion_receipts() -> list[str]:
    """Every Orion slide must carry the pinned implementation and checkpoints."""
    problems = []
    orion = sorted(RECEIPTS.glob("CRC*/complete.json"))
    if len(orion) != ORION_SLIDES:
        problems.append(f"{len(orion)} Orion receipts, expected {ORION_SLIDES}")
    impl, ckpt, mpp = set(), set(), set()
    for path in orion:
        r = json.loads(path.read_text())
        impl.add(r.get("implementation_hash"))
        ckpt.add(json.dumps(r.get("checkpoint_hashes"), sort_keys=True))
        mpp.add(r.get("source", {}).get("mpp"))
    if orion and impl != {IMPLEMENTATION_SHA256}:
        problems.append(f"Orion implementation hash drift: {sorted(impl)}")
    if orion and ckpt != {json.dumps(CHECKPOINTS, sort_keys=True)}:
        problems.append("Orion checkpoint hashes differ from the sealed pins")
    if orion and mpp != {0.325}:
        problems.append(f"Orion source MPP is not uniformly 0.325: {sorted(mpp)}")
    return problems


def preflight(*, strict: bool) -> list[str]:
    problems: list[str] = []
    try:
        problems += check_queue(queue_counts())
    except GateError as exc:
        problems.append(str(exc))
    problems += check_features()
    problems += check_orion_receipts()
    if problems and strict:
        raise GateError("preflight failed:\n  - " + "\n  - ".join(problems))
    return problems


def rebuild(encoder: str) -> None:
    coords, name, dim = ENCODERS[encoder]
    command = [
        str(PACK_PYTHON), "scripts/pack_features.py",
        "platform=colon_workstation", "data=colon",
        f"encoder={encoder}", "extraction=colon_native", "pack.overwrite=true",
    ]
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True,
                   env={**os.environ, "HF_HUB_OFFLINE": "1",
                        "PYTHONPATH": str(REPO / "src")})
    source = feature_dir(coords, name)
    meta = validate_packed_dir(pack_dir(coords, name),
                               verify_source=feature_inventory_sha256(source))
    if meta.n_slides != EXPECTED_SLIDES or meta.feat_dim != dim:
        raise GateError(f"{name} pack metadata wrong: {meta}")
    index = pd.read_parquet(pack_dir(coords, name) / "index.parquet", columns=["slide_id"])
    packed = set(index["slide_id"].astype(str))
    missing = {p.stem for p in RECEIPTS.glob("CRC*") if p.is_dir()} - packed
    if missing:
        raise GateError(f"{name} pack lacks Orion IDs: {sorted(missing)[:5]}")
    print(f"OK  {name}: {meta.n_slides} slides x {meta.feat_dim}", flush=True)


def rebuild_virchow2() -> None:
    command = [str(REPO / ".venv/bin/python"),
               "tools/pack_virchow2_after_encoding.py", "run",
               "--expected-slides", str(EXPECTED_SLIDES)]
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})


def cmd_check(_: argparse.Namespace) -> int:
    try:
        counts = queue_counts()
        print("queue:", {k: v for k, v in sorted(counts.items())})
    except GateError as exc:
        print("queue:", exc)
    for coords, name in [(c, n) for c, n, _ in ENCODERS.values()] + [VIRCHOW2_LAYOUT[:2]]:
        d = feature_dir(coords, name)
        n = len(list(d.glob("*.h5"))) if d.is_dir() else 0
        print(f"features_{name}: {n}/{EXPECTED_SLIDES}")
    problems = preflight(strict=False)
    if problems:
        print("\nNOT READY:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nREADY: preflight clean; `run` would rebuild all four packs.")
    return 0


def cmd_run(_: argparse.Namespace) -> int:
    preflight(strict=True)
    print(f"preflight clean at {EXPECTED_SLIDES} slides\n")
    for encoder in ENCODERS:
        rebuild(encoder)
    rebuild_virchow2()
    print("\nAll four packs rebuilt and validated.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(func=cmd_check)
    sub.add_parser("run").set_defaults(func=cmd_run)
    args = p.parse_args()
    try:
        return args.func(args)
    except GateError as exc:
        print(f"\nGATE FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
