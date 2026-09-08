#!/usr/bin/env python3
"""Per-epoch training curves — patient AUROC against patient loss.

Both are already logged every epoch by the CSV logger; nothing needed adding.
This just collects `lightning_logs/*/metrics.csv` across arms, seeds and folds
and puts the two curves side by side.

WHY THE PAIR IS WORTH WATCHING. The study monitors `val/patient_auroc` and
deliberately does NOT monitor `val/patient_loss`, because on this cohort the
two diverge: discrimination keeps improving while BCE degrades — the model gets
better at ranking and worse at calibrated probability. The divergence epoch is
the diagnostic. If loss and AUROC were to start peaking together, the argument
for the current criterion would weaken; if loss diverges early and hard, the
downstream Platt calibrator is doing more work than it looks like.

The last row of a fold is usually the restored best checkpoint being
re-evaluated, so it duplicates the best epoch — it is dropped, not plotted.

Usage:
    python tools/visualization/epoch_trends.py --arm 1a_pb_cap4096
    python tools/visualization/epoch_trends.py --arm 1a_pb_cap4096 --seed 42 --fold 0
    python tools/visualization/epoch_trends.py --compare 1a_cap4096 1a_pb_cap4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import paths  # noqa: E402

TRAIN = paths.OUTPUT_ROOT / "train"
COLS = ["val/patient_auroc", "val/patient_loss", "val/auroc", "val/loss", "train/loss_epoch"]


def fold_curve(arm: str, seed: int, fold: int, encoder: str = "univ1") -> pd.DataFrame | None:
    base = TRAIN / arm / encoder / f"seed{seed}" / f"fold_{fold}"
    hits = sorted(base.glob("lightning_logs/*/metrics.csv"))
    if not hits:
        return None
    d = pd.read_csv(hits[-1])
    keep = [c for c in COLS if c in d.columns]
    if not keep or "epoch" not in d.columns:
        return None
    v = d[["epoch", *keep]].dropna(subset=keep, how="all").groupby("epoch").last()
    # Drop the trailing duplicate: at the end of a fold the restored best
    # checkpoint is re-evaluated, so the last row repeats the selected epoch.
    # Compare on the val columns only: the final eval row carries no
    # train/loss_epoch, and a NaN never equals anything, so including it would
    # make the duplicate look distinct and leave a phantom epoch on the curve.
    vcols = [c for c in keep if c.startswith("val/")]
    if len(v) > 1 and vcols:
        last = v.iloc[-1][vcols].astype(float)
        earlier = v.iloc[:-1][vcols].astype(float)
        if bool((earlier.round(6) == last.round(6)).all(axis=1).any()):
            v = v.iloc[:-1]
    return v.reset_index()


def show(arm: str, seeds: list[int], folds: list[int]) -> None:
    print(f"\n{'=' * 96}\n{arm}\n{'=' * 96}")
    peaks = []
    for seed in seeds:
        for fold in folds:
            v = fold_curve(arm, seed, fold)
            if v is None:
                continue
            auroc, loss = v["val/patient_auroc"], v["val/patient_loss"]
            best_auroc, best_loss = int(auroc.idxmax()), int(loss.idxmin())
            peaks.append(
                (
                    seed,
                    fold,
                    int(v.epoch[best_auroc]),
                    float(auroc[best_auroc]),
                    int(v.epoch[best_loss]),
                    float(loss[best_loss]),
                )
            )
            print(
                f"\n  seed {seed} fold {fold}   "
                f"AUROC peaks epoch {int(v.epoch[best_auroc])} ({auroc[best_auroc]:.4f})   "
                f"loss bottoms epoch {int(v.epoch[best_loss])} ({loss[best_loss]:.4f})"
            )
            print(
                f"    {'epoch':>5s} {'pat AUROC':>10s} {'pat loss':>9s} {'train loss':>10s}  curve"
            )
            lo, hi = float(auroc.min()), float(auroc.max())
            for _, r in v.iterrows():
                bar = ""
                if hi > lo:
                    n = int(round(28 * (float(r["val/patient_auroc"]) - lo) / (hi - lo)))
                    bar = "#" * max(n, 1)
                mark = ""
                if int(r.epoch) == int(v.epoch[best_auroc]):
                    mark += "  <- AUROC peak (selected)"
                if int(r.epoch) == int(v.epoch[best_loss]):
                    mark += "  <- loss min"
                tl = r.get("train/loss_epoch", float("nan"))
                print(
                    f"    {int(r.epoch):5d} {r['val/patient_auroc']:10.4f} "
                    f"{r['val/patient_loss']:9.4f} {tl:10.4f}  {bar}{mark}"
                )
    if peaks:
        gap = [p[2] - p[4] for p in peaks]
        print(
            f"\n  SUMMARY over {len(peaks)} folds: AUROC peak epoch median "
            f"{np.median([p[2] for p in peaks]):.0f}, loss-min epoch median "
            f"{np.median([p[4] for p in peaks]):.0f}, "
            f"median gap {np.median(gap):+.0f} epochs"
        )
        print(
            "  A positive gap means discrimination keeps improving after the loss has "
            "already turned —\n  which is exactly why val/patient_auroc is the criterion "
            "and val/patient_loss is not."
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arm", default="1a_pb_cap4096")
    p.add_argument("--compare", nargs="+")
    p.add_argument("--seed", type=int)
    p.add_argument("--fold", type=int)
    a = p.parse_args()
    seeds = [a.seed] if a.seed else [42, 43, 44]
    folds = [a.fold] if a.fold is not None else list(range(paths.N_FOLDS))
    for arm in a.compare or [a.arm]:
        show(arm, seeds, folds)


if __name__ == "__main__":
    main()
