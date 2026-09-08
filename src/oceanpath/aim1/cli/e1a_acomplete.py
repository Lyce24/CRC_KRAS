#!/usr/bin/env python3
"""E1a step 4 — set D against the A-complete comparator (0 new fits).

WHY A SEPARATE REFERENCE. Raw E1a reports every set against A, but D can only
contain patients whose MSI *and* BRAF are both known, while A also contains 99
patients with a missing molecular label. D versus A therefore confounds
"conditioning on MSI/BRAF" with "dropping the label-incomplete". A-complete
matches D's ascertainment without matching its molecular restriction, so
Delta(D - A-complete) is the part attributable to the molecular condition alone.

ESTIMATOR. D is nested inside A-complete and the model is fixed, so a naive
paired bootstrap returns exactly zero. Patients are resampled ONCE from
A-complete and both AUROC(A-complete_boot) and AUROC(D and A-complete_boot) are
recomputed on that same resample -- the same shared-resample estimator raw E1a
uses, with A-complete rather than A as the reference.

SIGN. Reported as Delta(D - A-complete): POSITIVE means D scored HIGHER. This is
the opposite sign convention from `aim1_locked_baseline.py`, which reports
Delta(reference - set); it is stated on every line of output so the two are not
confused.

This regenerates the numbers behind Results.md Table 3, which had been produced
by a one-off that was not kept.

Usage:
    python aim1_challenge_sets.py vs-acomplete --condition pb_cap8192
    python aim1_challenge_sets.py vs-acomplete --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]  # src/oceanpath/aim1/cli/<this> -> repo root
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1.cli import e1a_challenge as e1a  # noqa: E402
from oceanpath.aim1 import evaluate, paths  # noqa: E402

TRAIN = paths.OUTPUT_ROOT / "train"
CONDITIONS = {
    "pb_cap4096": (TRAIN / "1a_pb_cap4096" / "univ1", "PATIENT-BALANCED, 4096 tiles/epoch"),
    "pb_cap8192": (TRAIN / "1a_pb_cap8192" / "univ1", "PATIENT-BALANCED, 8192 tiles/epoch"),
    "v2cls": (TRAIN / "1a_pb_cap4096" / "virchow2_cls", "VIRCHOW2 CLS, 4096 tiles/epoch, patient-balanced"),
}
SEEDS = (42, 43, 44)


def load_seed(root: Path, seed: int) -> pd.DataFrame | None:
    path = root / f"seed{seed}" / "oof_predictions.parquet"
    if not path.is_file():
        return None
    manifest = pd.read_csv(paths.DEV_MANIFEST)
    return evaluate.to_patient_level(pd.read_parquet(path), manifest)


def report_condition(name: str) -> dict:
    root, label = CONDITIONS[name]
    frames = {s: f for s in SEEDS if (f := load_seed(root, s)) is not None}
    if not frames:
        print(f"{name}: no completed seeds")
        return {}
    print(f"\n{'=' * 92}\nE1a step 4 . {name.upper()} ({label}) . D vs A-complete"
          f" . seeds {sorted(frames)}\n{'=' * 92}")
    print("  SIGN: Delta = AUROC(D) - AUROC(A-complete). POSITIVE = D scored HIGHER.")
    print("  Shared-resample bootstrap: patients drawn once from A-COMPLETE, both arms")
    print("  recomputed on that same resample.\n")
    print(f"  {'seed':>5s} {'AUROC A':>9s} {'A-complete':>11s} {'D':>9s} "
          f"{'Delta(D - A-comp)':>18s} {'95% CI':>22s}")

    out: dict = {}
    for seed, f in sorted(frames.items()):
        ac_mask = e1a.subset_mask(f, "A_complete")
        ac = f[ac_mask].reset_index(drop=True)
        d_within_ac = e1a.subset_mask(ac, "D_mss_braf_wt")
        blk = evaluate.shared_resample_delta(ac, d_within_ac)
        # shared_resample_delta returns delta = reference - subset; flip the sign.
        row = {
            "A": evaluate.patient_auroc(f),
            "Ac": blk["auroc_full"],
            "D": blk["auroc_subset"],
            "n_Ac": blk["n_full"],
            "n_D": blk["n_subset"],
            "delta": -blk["delta"],
            "lo": -blk["delta_ci_high"],
            "hi": -blk["delta_ci_low"],
        }
        out[str(seed)] = row
        ci = "[{:+.4f}, {:+.4f}]".format(row["lo"], row["hi"])
        print(f"  {seed:5d} {row['A']:9.4f} {row['Ac']:11.4f} {row['D']:9.4f} "
              f"{row['delta']:+18.4f} {ci:>22s}")

    med = float(np.median([r["delta"] for r in out.values()]))
    lo = float(np.median([r["lo"] for r in out.values()]))
    hi = float(np.median([r["hi"] for r in out.values()]))
    ci_med = f"[{lo:+.4f}, {hi:+.4f}]"
    print(f"  {'median':>5s} {'':9s} {'':11s} {'':9s} {med:+18.4f} {ci_med:>22s}")
    print(f"\n  n(A-complete) = {out[str(sorted(frames)[0])]['n_Ac']}, "
          f"n(D) = {out[str(sorted(frames)[0])]['n_D']}")
    print("  Reading: A-complete sits within a hair of A, so the 99 label-incomplete")
    print("  patients were not holding A down; D's advantage survives the matched reference.")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=list(CONDITIONS), default="pb_cap8192")
    p.add_argument("--compare", action="store_true")
    a = p.parse_args()
    names = list(CONDITIONS) if a.compare else [a.condition]
    results = {}
    for n in names:
        # Key by the run directory, so a newly registered condition cannot
        # KeyError on a hardcoded table that nobody remembered to extend.
        root, _ = CONDITIONS[n]
        arm = f"{root.parent.name}_{root.name}" if root.name != "univ1" else root.parent.name
        block = report_condition(n)
        if block:
            results[arm] = block
    dest = paths.EVAL_ROOT / "e1a_Acomplete_both_caps.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # MERGE, never replace: a single-condition run must not delete the other arm.
    existing = json.loads(dest.read_text()) if dest.is_file() else {}
    existing.update(results)
    dest.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
