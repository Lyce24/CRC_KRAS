#!/usr/bin/env python3
"""Run every Aim 1-3 MIL training chain back to back through ONE global queue.

Why this exists. The campaign launchers (`aim3_resolution_ladder.py --jobs`, `aim3_repeated_control_campaign.py
--jobs`, ...) each parallelise *within* one arm. Running the arms one after
another therefore drains the slot pool at the tail of every arm: E0 has only 3
chains, so a 6-slot pool sits half idle for over an hour, and the same happens
at the end of E2a, E1v, E1e and E2e. Pooling all 117 chains into a single
longest-first queue removes those bubbles.

Measured on this repo's own logs (117 chains, 62.6 GPU-slot-hours):

    slots   arm-by-arm   one global queue
      4        18.2 h         15.7 h
      6        14.3 h         10.5 h      <- 27% saved

Chains are independent by construction -- separate processes, separate
train_dirs, pre-generated splits, no shared mutable state -- so pooling them
cannot change a result, only the wall clock. This program never redraws a
split, never edits a manifest and never writes into an existing train_dir: it
only decides the order in which the existing launchers are invoked.

The per-chain `train_dir` is resolved by importing each launcher and calling
that launcher's own path function, so the completion check cannot drift from
what the launcher actually writes.

Usage::

    python tools/train_all_pipeline.py plan                  # what would run
    python tools/train_all_pipeline.py plan --slots 6        # + wall-clock model
    python tools/train_all_pipeline.py run  --slots 6        # execute
    python tools/train_all_pipeline.py run  --slots 6 --only E3,E3v
    python tools/train_all_pipeline.py status                # progress of a live run
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

PYTHON = REPO / ".venv/bin/python"
STATE_DIR = REPO / "outputs" / "train_all_pipeline"
LOG_DIR = STATE_DIR / "logs"
STATE_PATH = STATE_DIR / "state.jsonl"

# The Aim-2 rerun lineage that final-v6 was sealed against. Every mutating E2a
# command requires it; it selects the immutable rerun root.
AIM2_LINEAGE = "aim2_cap8192_v4_20260819"
E3_REPEAT_ROOT = "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim3_controls_repeated_v1_20260819"

# Median minutes per chain, measured from this repository's own launcher logs
# at the 5-6 concurrency these arms actually ran at. Used ONLY to order the
# queue longest-first and to project a finish time; never to gate anything.
MINUTES = {
    "E0": 71.6,
    "E1v": 84.6,
    "E1e": 68.5,
    "E2a:SurGen": 23.4,
    "E2a:TCGA": 48.7,
    "E2a:RIH": 75.1,
    "E2a:CPTAC": 59.3,
    "E2a:RIH_sm": 23.4,
    "E2e": 4.7,
    "E3": 26.2,
    "E3rep": 26.2,
    "E3v": 26.7,
}

SEEDS = (42, 43, 44)

# Folds per chain. E2a chains carry one extra full-source refit on top of the
# five source-only CV folds -- `4 x 3 x (5 + 1) = 72` in the setup document.
FITS_PER_CHAIN = {"E2a": 6}
DEFAULT_FITS = 5


def fits_of(arm: str) -> int:
    return FITS_PER_CHAIN.get(arm.split(":")[0], DEFAULT_FITS)


@dataclass
class Chain:
    arm: str          # scheduling/estimate key, e.g. "E3" or "E2a:RIH"
    name: str         # unique human-readable id, e.g. "E3:g12c:seed42"
    command: list[str]
    train_dirs: list[Path]
    minutes: float
    env: dict[str, str] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Complete when any candidate location holds the launcher's OOF file.

        E2a is the reason this is a list: its sealed source-CV chains live at
        the legacy root while the size-matched sensitivity arm lives under the
        active rerun lineage, and `aim2_loco_transport.source_cv_dir` resolves between them
        only when an extra input-root variable is exported.
        """
        return any((d / "oof_predictions.parquet").is_file() for d in self.train_dirs)


def _py(script: str, *args: str) -> list[str]:
    return [str(PYTHON), str(REPO / script), *args]


def build_chains() -> list[Chain]:
    """Enumerate all Aim 1-3 chains, resolving train_dirs via the launchers."""
    import aim1_msi_positive_control
    import aim1_virchow2_encoder_arm
    import aim2_metastatic_indomain_bound
    import aim3_resolution_ladder
    import aim3_virchow2_replication
    from oceanpath.aim1 import paths as aim1_paths

    chains: list[Chain] = []

    # --- Aim 1 -------------------------------------------------------------
    # E0 has no chain-level CLI of its own; it is the locked baseline trained
    # straight through scripts/train.py with the frozen override set.
    e0_root = aim1_paths.OUTPUT_ROOT / "train" / "1a_pb_cap8192" / "univ1"
    for seed in SEEDS:
        d = e0_root / f"seed{seed}"
        chains.append(Chain(
            "E0", f"E0:seed{seed}",
            _py("scripts/train.py",
                "platform=colon_workstation", "data=aim1", "data.aim1_model=1a",
                "data.manifest_stem=aim1_dev", "+data.cohort_column=cohort",
                "encoder=univ1", "splits=aim1_balanced", f"splits.seed={seed}",
                "model=abmil", "model.embed_dim=512", "model.attn_dim=384",
                "model.input_dropout=0.10", "model.dropout=0.25",
                "training=aim1", "training.lr=1e-4", "training.weight_decay=1e-5",
                f"training.seed={seed}", "training.dataset_max_instances=8192",
                "training.eval_full_bags=true",
                "training.train_sampling_strategy=patient_natural",
                "training.sample_weight_column=null",
                f"train_dir={d}", f"exp_name=aim1_1a_pb_cap8192_seed{seed}"),
            [d], MINUTES["E0"]))

    for seed in SEEDS:
        chains.append(Chain("E1v", f"E1v:seed{seed}",
                            _py("aim1_virchow2_encoder_arm.py", "train", "--seed", str(seed)),
                            [aim1_virchow2_encoder_arm.train_dir(seed)], MINUTES["E1v"]))
        chains.append(Chain("E1e", f"E1e:seed{seed}",
                            _py("aim1_msi_positive_control.py", "train", "--seed", str(seed)),
                            [aim1_msi_positive_control.train_dir(seed)], MINUTES["E1e"]))

    # --- Aim 2 -------------------------------------------------------------
    env2 = {"OCEANPATH_AIM2_LINEAGE": AIM2_LINEAGE}
    os.environ.setdefault("OCEANPATH_AIM2_LINEAGE", AIM2_LINEAGE)
    import aim2_loco_transport  # imported after the lineage is set: its roots resolve at import
    def e2a_candidates(arm: str, seed: int) -> list[Path]:
        rel = Path("source_cv") / "cap8192" / arm / f"seed{seed}"
        return [aim2_loco_transport.e2a_root() / rel, aim2_loco_transport.LEGACY_E2A_ROOT / rel]

    for target in aim2_loco_transport.TARGETS:
        for seed in SEEDS:
            chains.append(Chain(
                f"E2a:{target}", f"E2a:{target}:seed{seed}",
                _py("aim2_loco_transport.py", "train", "--target", target, "--seed", str(seed),
                    "--cap", "8192"),
                e2a_candidates(target.lower(), seed),
                MINUTES[f"E2a:{target}"], env2))

    # Pre-specified size-matched RIH sensitivity arm (+18 fits): a separate
    # model for an already-covered cohort, so it is its own chain family.
    for seed in SEEDS:
        chains.append(Chain(
            "E2a:RIH_sm", f"E2a:RIH_sm:seed{seed}",
            _py("aim2_loco_transport.py", "train", "--target", "RIH", "--seed", str(seed),
                "--cap", "8192", "--size-matched"),
            e2a_candidates("rih" + aim2_loco_transport.SIZE_MATCH_SUFFIX, seed),
            MINUTES["E2a:RIH_sm"], env2))

    for seed in SEEDS:
        chains.append(Chain("E2e", f"E2e:seed{seed}",
                            _py("aim2_metastatic_indomain_bound.py", "train", "--seed", str(seed)),
                            [aim2_metastatic_indomain_bound.train_dir(seed)], MINUTES["E2e"]))

    # --- Aim 3 -------------------------------------------------------------
    for task in aim3_resolution_ladder.TASKS:
        for seed in SEEDS:
            chains.append(Chain("E3", f"E3:{task}:seed{seed}",
                                _py("aim3_resolution_ladder.py", "train", "--task", task, "--seed", str(seed)),
                                [aim3_resolution_ladder.run_dir(task, seed)], MINUTES["E3"]))

    for task in aim3_virchow2_replication.TASKS:
        for seed in SEEDS:
            chains.append(Chain("E3v", f"E3v:{task}:seed{seed}",
                                _py("aim3_virchow2_replication.py", "train", "--task", task, "--seed", str(seed)),
                                [aim3_virchow2_replication.run_dir(task, seed)], MINUTES["E3v"]))

    import aim3_repeated_control_campaign as e3r
    for control in e3r.CONTROLS:
        for draw in e3r.WT_DRAW_SEEDS:
            for seed in e3r.MODEL_SEEDS:
                chains.append(Chain(
                    "E3rep", f"E3rep:{control}:draw{draw}:seed{seed}",
                    _py("aim3_repeated_control_campaign.py", "train",
                        "--output-root", E3_REPEAT_ROOT, "--control", control,
                        "--draw-seed", str(draw), "--model-seed", str(seed),
                        "--jobs", "1", "--apply"),
                    [e3r.run_dir(Path(E3_REPEAT_ROOT), control, draw, seed)],
                    MINUTES["E3rep"]))
    return chains


def pack(durations: list[float], slots: int) -> float:
    """Longest-first wall clock for these durations on `slots` workers."""
    heap = [0.0] * slots
    heapq.heapify(heap)
    for d in sorted(durations, reverse=True):
        heapq.heappush(heap, heapq.heappop(heap) + d)
    return max(heap)


_lock = threading.Lock()


def _record(**kw: object) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with _lock, STATE_PATH.open("a") as fh:
        fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}) + "\n")


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    with _lock:
        print(line, flush=True)


def run_chain(chain: Chain, attempt: int = 1) -> tuple[Chain, bool]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{chain.name.replace(':', '_')}.log"
    t0 = time.time()
    _log(f"START  {chain.name}  (est {chain.minutes:.0f} min, attempt {attempt})")
    with log_path.open("a") as fh:
        fh.write(f"\n=== attempt {attempt} at {time.strftime('%F %T')} ===\n")
        fh.write(" ".join(chain.command) + "\n")
        fh.flush()
        code = subprocess.run(
            chain.command, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
            env={**os.environ, **chain.env, "PYTHONUNBUFFERED": "1"},
        ).returncode
    dt = (time.time() - t0) / 60
    ok = code == 0 and chain.done
    _log(f"{'OK   ' if ok else 'FAIL '}  {chain.name}  {dt:.1f} min"
         + ("" if ok else f"  rc={code} oof={chain.done} -> {log_path}"))
    _record(chain=chain.name, arm=chain.arm, status="ok" if ok else "failed",
            minutes=round(dt, 1), returncode=code, attempt=attempt)
    return chain, ok


def cmd_plan(args: argparse.Namespace) -> int:
    chains = build_chains()
    if args.only:
        wanted = {a.strip() for a in args.only.split(",")}
        chains = [c for c in chains if c.arm.split(":")[0] in wanted or c.arm in wanted]
    pending = chains if args.all else [c for c in chains if not c.done]
    by_arm: dict[str, list[Chain]] = {}
    for c in chains:
        by_arm.setdefault(c.arm.split(":")[0], []).append(c)

    print(f"{'arm':10s} {'chains':>7s} {'fits':>6s} {'done':>6s} {'queued':>7s} "
          f"{'slot-h queued':>14s}")
    tf = 0
    for arm in sorted(by_arm):
        cs = by_arm[arm]
        q = cs if args.all else [c for c in cs if not c.done]
        f = sum(fits_of(c.arm) for c in cs)
        tf += f
        print(f"{arm:10s} {len(cs):7d} {f:6d} {sum(c.done for c in cs):6d} {len(q):7d} "
              f"{sum(c.minutes for c in q)/60:14.1f}")
    total_min = sum(c.minutes for c in pending)
    print(f"{'TOTAL':10s} {len(chains):7d} {tf:6d} {sum(c.done for c in chains):6d} "
          f"{len(pending):7d} {total_min/60:14.1f}")

    if not pending:
        print("\nNothing pending: every chain already has oof_predictions.parquet.")
        return 0
    print(f"\nwall-clock model for the {len(pending)} pending chains:")
    for s in (1, 2, 3, 4, 6, 8):
        glob = pack([c.minutes for c in pending], s)
        seq = sum(pack([c.minutes for c in cs
                        if args.all or not c.done], s) or 0
                  for cs in by_arm.values() if cs)
        mark = "  <- --slots" if s == args.slots else ""
        print(f"  slots {s}: one queue {glob/60:5.1f} h   (arm-by-arm {seq/60:5.1f} h){mark}")
    if args.verbose:
        print("\nqueue order (longest first):")
        for c in sorted(pending, key=lambda c: -c.minutes):
            print(f"  {c.minutes:6.1f} min  {c.name}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not PYTHON.is_file():
        raise SystemExit(f"interpreter not found: {PYTHON}")
    chains = build_chains()
    if args.only:
        wanted = {a.strip() for a in args.only.split(",")}
        chains = [c for c in chains if c.arm.split(":")[0] in wanted or c.arm in wanted]
    pending = sorted((c for c in chains if not c.done), key=lambda c: -c.minutes)
    if not pending:
        print("Nothing to train.")
        return 0

    est = pack([c.minutes for c in pending], args.slots) / 60
    _log(f"=== {len(pending)} pending chains, {args.slots} slots, "
         f"projected {est:.1f} h (longest-first) ===")
    _record(event="start", pending=len(pending), slots=args.slots,
            projected_hours=round(est, 2))
    if args.dry_run:
        for c in pending:
            print(f"{c.minutes:6.1f} min  {c.name}\n    " + " ".join(c.command))
        return 0

    failures: list[str] = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.slots) as pool:
        futures = {pool.submit(run_chain, c): c for c in pending}
        retried: set[str] = set()
        while futures:
            for fut in as_completed(list(futures)):
                chain = futures.pop(fut)
                _, ok = fut.result()
                if not ok and chain.name not in retried and not args.no_retry:
                    retried.add(chain.name)
                    _log(f"RETRY  {chain.name}")
                    futures[pool.submit(run_chain, chain, 2)] = chain
                    continue
                if not ok:
                    failures.append(chain.name)
                done += 1
                elapsed = (time.time() - t0) / 60
                _log(f"       progress {done}/{len(pending)} chains, "
                     f"{elapsed/60:.1f} h elapsed")
                break

    _log(f"=== DONE in {(time.time()-t0)/3600:.1f} h; "
         f"failed: {failures or 'none'} ===")
    _record(event="finish", failed=failures, hours=round((time.time()-t0)/3600, 2))
    return 1 if failures else 0


def cmd_status(_: argparse.Namespace) -> int:
    chains = build_chains()
    done = [c for c in chains if c.done]
    print(f"{len(done)}/{len(chains)} chains complete "
          f"({sum(c.minutes for c in chains if not c.done)/60:.1f} slot-h remaining)")
    if STATE_PATH.is_file():
        rows = [json.loads(x) for x in STATE_PATH.read_text().splitlines() if x.strip()]
        fails = [r for r in rows if r.get("status") == "failed"]
        print(f"state events: {len(rows)}; failed attempts: {len(fails)}")
        for r in fails[-5:]:
            print(f"  {r['at']}  {r['chain']}  rc={r.get('returncode')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("plan", cmd_plan), ("run", cmd_run), ("status", cmd_status)):
        x = sub.add_parser(name)
        x.set_defaults(func=fn)
        if name in ("plan", "run"):
            x.add_argument("--slots", type=int, default=6)
            x.add_argument("--only", help="comma-separated arms, e.g. E3,E3v")
        if name == "plan":
            x.add_argument("--verbose", action="store_true")
            x.add_argument("--all", action="store_true",
                           help="model a full retrain: queue every chain, "
                                "including ones already complete")
        if name == "run":
            x.add_argument("--dry-run", action="store_true")
            x.add_argument("--no-retry", action="store_true")
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
