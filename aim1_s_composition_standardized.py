#!/usr/bin/env python3
"""E1a-S — composition-standardized challenge-set analysis (cap 4,096, 0 new fits).

WHY THIS EXISTS. Raw E1a compares AUROC(A) with AUROC(D) on two populations that
differ in composition as well as in molecular context: restricting to MSS/pMMR
and BRAF-wild-type does not remove patients uniformly across subcohorts or
across colon/rectum. Some of the apparent "D retains the signal" result could
therefore be a composition shift — D happens to contain relatively more of a
stratum the model does well on — rather than genuine retention after the MSI
and BRAF gradients are removed.

WHAT THIS DOES. Direct standardization. Both A and D are scored WITHIN common
subcohort x site strata, and the stratum AUROCs are combined with ONE FIXED
WEIGHT VECTOR applied identically to both. Any difference that survives is a
difference in discrimination, not in case mix.

The weights are set A's own stratum distribution over the common support, so
the standardized numbers answer: "if D had A's composition, how would it
score?"

INFERENCE. Patients are resampled WITHIN stratum x KRAS cells, so every
resample preserves the composition being standardized to — a plain patient
bootstrap would put composition variance back into the interval this analysis
exists to remove.

The raw E1a result already passes the gate; this is the version the manuscript
claim rests on.

Usage:
    python aim1_s_composition_standardized.py --condition capped
    python aim1_s_composition_standardized.py --condition capped_pb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1.cli import e1a_challenge as e1a  # noqa: E402
from oceanpath.aim1 import evaluate, paths  # noqa: E402

TRAIN = paths.OUTPUT_ROOT / "train"
CONDITIONS = {
    "capped": (TRAIN / "1a_cap4096" / "univ1", "4096 tiles/epoch, slide-uniform (pre-fix)"),
    "pb_cap4096": (TRAIN / "1a_pb_cap4096" / "univ1", "PATIENT-BALANCED, 4096 tiles/epoch"),
    "pb_cap8192": (TRAIN / "1a_pb_cap8192" / "univ1", "PATIENT-BALANCED, 8192 tiles/epoch"),
    "v2cls": (TRAIN / "1a_pb_cap4096" / "virchow2_cls", "VIRCHOW2 CLS, 4096 tiles/epoch, patient-balanced"),
}
SEEDS = (42, 43, 44)
N_BOOT = 2000
MIN_CELL = 10          # patients in a stratum, in BOTH A and D
MIN_PER_CLASS = 3      # mutant and wild-type, in BOTH A and D


def load_seed(root: Path, seed: int) -> pd.DataFrame | None:
    path = root / f"seed{seed}" / "oof_predictions.parquet"
    if not path.is_file():
        return None
    manifest = pd.read_csv(paths.DEV_MANIFEST)
    pat = evaluate.to_patient_level(pd.read_parquet(path), manifest)
    stage = manifest.drop_duplicates("patient_id")[["patient_id", "stage_group_major"]]
    pat = pat.merge(stage, on="patient_id", how="left")
    # Stratum = subcohort x colon/rectum. Sites outside colon/rectum (appendix,
    # other/unknown; 13 patients) cannot be standardized against a colon/rectum
    # weight vector and are dropped from BOTH arms, symmetrically.
    pat["site2"] = pat["tumor_site_group"].where(
        pat["tumor_site_group"].isin(["Colon", "Rectum"])
    )
    pat["stratum"] = pat["subcohort"].astype(str) + " · " + pat["site2"].astype(str)
    return pat[pat["site2"].notna()].copy()


def usable_strata(a: pd.DataFrame, d: pd.DataFrame) -> list[str]:
    """Strata that can carry an AUROC in BOTH arms.

    A stratum with one class present has an undefined AUROC; including it would
    force a choice between dropping it from one arm only (which breaks the
    shared weight vector) or imputing a value. Requiring the same support in
    both arms keeps the two standardized numbers commensurable.
    """
    out = []
    for s in sorted(a["stratum"].unique()):
        ba, bd = a[a.stratum.eq(s)], d[d.stratum.eq(s)]
        ok = (
            len(ba) >= MIN_CELL and len(bd) >= MIN_CELL
            and ba.label.sum() >= MIN_PER_CLASS and (1 - ba.label).sum() >= MIN_PER_CLASS
            and bd.label.sum() >= MIN_PER_CLASS and (1 - bd.label).sum() >= MIN_PER_CLASS
        )
        if ok:
            out.append(s)
    return out


def stratum_auroc(block: pd.DataFrame) -> float:
    y = block["label"].to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, block["prob_raw"].to_numpy()))


def standardized(frame: pd.DataFrame, strata: list[str], weights: dict[str, float]) -> float:
    total, wsum = 0.0, 0.0
    for s in strata:
        v = stratum_auroc(frame[frame.stratum.eq(s)])
        if np.isfinite(v):
            total += weights[s] * v
            wsum += weights[s]
    return total / wsum if wsum else float("nan")


def run_seed(pat: pd.DataFrame, reference: str = "A_all_primary") -> dict:
    a = pat[e1a.subset_mask(pat, reference)] if reference != "A_all_primary" else pat
    d = pat[e1a.subset_mask(pat, "D_mss_braf_wt")]
    if not d.index.isin(a.index).all():
        raise SystemExit(f"D is not nested in {reference}; the shared resample would be invalid")
    strata = usable_strata(a, d)
    # ONE weight vector, from A's composition over the common support.
    counts = a[a.stratum.isin(strata)].groupby("stratum").size()
    weights = (counts / counts.sum()).to_dict()

    per_stratum = {}
    for s in strata:
        ba, bd = a[a.stratum.eq(s)], d[d.stratum.eq(s)]
        per_stratum[s] = {
            "weight": weights[s],
            "n_A": int(len(ba)), "n_D": int(len(bd)),
            "mut_A": int(ba.label.sum()), "mut_D": int(bd.label.sum()),
            "auroc_A": stratum_auroc(ba), "auroc_D": stratum_auroc(bd),
        }

    std_a = standardized(a, strata, weights)
    std_d = standardized(d, strata, weights)

    # Bootstrap WITHIN stratum x KRAS, SHARED between the two arms.
    #
    # D is a SUBSET of A, so the two arms must be resampled together: draw once
    # from A's stratum x label cells, then take D as the subset of that same
    # draw. Resampling them independently would treat two nested, highly
    # correlated samples as independent and inflate the variance of their
    # difference — the same error a naive paired bootstrap makes in raw E1a,
    # arriving from the opposite direction. Composition is held fixed by the
    # weight vector, not by the resampling, so D's cell sizes are free to move.
    rng = np.random.default_rng(paths.BOOTSTRAP_SEED)
    in_d = a.index.isin(d.index)
    d_flag = pd.Series(in_d, index=a.index)
    cells_a = [a[(a.stratum.eq(s)) & (a.label.eq(y))].index.to_numpy()
               for s in strata for y in (0, 1)]
    cells_a = [c for c in cells_a if len(c)]
    da, dd, diffs = [], [], []
    for _ in range(N_BOOT):
        ia = np.concatenate([rng.choice(c, len(c), replace=True) for c in cells_a])
        boot_a = a.loc[ia]
        boot_d = boot_a[d_flag.loc[ia].to_numpy()]
        va = standardized(boot_a, strata, weights)
        vd = standardized(boot_d, strata, weights)
        if np.isfinite(va) and np.isfinite(vd):
            da.append(va); dd.append(vd); diffs.append(vd - va)
    q = lambda v, p: float(np.percentile(v, p)) if v else float("nan")  # noqa: E731
    return {
        "strata_used": strata,
        "n_strata": len(strata),
        "n_A": int(len(a)), "n_D": int(len(d)),
        "per_stratum": per_stratum,
        "crude_A": stratum_auroc(a), "crude_D": stratum_auroc(d),
        "standardized_A": std_a, "standardized_D": std_d,
        "standardized_A_ci": [q(da, 2.5), q(da, 97.5)],
        "standardized_D_ci": [q(dd, 2.5), q(dd, 97.5)],
        "delta_D_minus_A": std_d - std_a,
        "delta_ci": [q(diffs, 2.5), q(diffs, 97.5)],
    }


def summ(v: list[float], nd: int = 4) -> str:
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "pending"
    if len(v) == 1:
        return f"{v[0]:.{nd}f} (1 seed)"
    return f"{np.median(v):.{nd}f} [{min(v):.{nd}f}-{max(v):.{nd}f}]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=list(CONDITIONS), default="capped")
    p.add_argument("--reference", choices=["A_all_primary", "A_complete"],
                   default="A_all_primary",
                   help="A_complete matches D's label ascertainment (MSI AND BRAF known)")
    a = p.parse_args()
    root, label = CONDITIONS[a.condition]
    frames = {s: f for s in SEEDS if (f := load_seed(root, s)) is not None}
    if not frames:
        raise SystemExit(f"{a.condition}: no completed seeds at {root}")
    per_seed = {s: run_seed(f, a.reference) for s, f in sorted(frames.items())}
    ss = sorted(per_seed)
    ref = per_seed[ss[0]]

    print(f"\n{'=' * 104}\nE1a-S · COMPOSITION-STANDARDIZED · {a.condition.upper()} ({label})"
          f" · seeds {ss}\n{'=' * 104}")
    print(f"  strata = subcohort x colon/rectum · {ref['n_strata']} usable cells "
          f"(present with both classes in A AND D)")
    print(f"  weights = set A's composition over the common support, applied IDENTICALLY to A and D")
    print(f"  n_A = {ref['n_A']}   n_D = {ref['n_D']}   "
          f"(patients outside colon/rectum dropped symmetrically from both)\n")

    print(f"  {'stratum':26s} {'weight':>7s} {'n A':>6s} {'n D':>6s} "
          f"{'AUROC A':>20s} {'AUROC D':>20s}")
    for s in ref["strata_used"]:
        wa = ref["per_stratum"][s]
        aus = [per_seed[k]["per_stratum"][s]["auroc_A"] for k in ss]
        dus = [per_seed[k]["per_stratum"][s]["auroc_D"] for k in ss]
        print(f"  {s:26s} {wa['weight']:7.3f} {wa['n_A']:6d} {wa['n_D']:6d} "
              f"{summ(aus):>20s} {summ(dus):>20s}")

    print(f"\n  {'':26s} {'crude':>20s} {'standardized':>22s} {'median 95% CI':>20s}")
    print(f"  {('set ' + a.reference):26s} "
          f"{summ([per_seed[k]['crude_A'] for k in ss]):>20s} "
          f"{summ([per_seed[k]['standardized_A'] for k in ss]):>22s} "
          f"{np.median([per_seed[k]['standardized_A_ci'][0] for k in ss]):.3f}-"
          f"{np.median([per_seed[k]['standardized_A_ci'][1] for k in ss]):.3f}")
    print(f"  {'set D (MSS & BRAF-WT)':26s} "
          f"{summ([per_seed[k]['crude_D'] for k in ss]):>20s} "
          f"{summ([per_seed[k]['standardized_D'] for k in ss]):>22s} "
          f"{np.median([per_seed[k]['standardized_D_ci'][0] for k in ss]):.3f}-"
          f"{np.median([per_seed[k]['standardized_D_ci'][1] for k in ss]):.3f}")

    dl = [per_seed[k]["delta_D_minus_A"] for k in ss]
    lo = np.median([per_seed[k]["delta_ci"][0] for k in ss])
    hi = np.median([per_seed[k]["delta_ci"][1] for k in ss])
    print(f"\n  STANDARDIZED Delta(D - A) = {summ(dl)}   95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"{'   EXCLUDES 0' if (lo > 0 or hi < 0) else '   crosses 0'}")
    crude_delta = [per_seed[k]["crude_D"] - per_seed[k]["crude_A"] for k in ss]
    print(f"  crude       Delta(D - A) = {summ(crude_delta)}")
    print(f"\n  Reading: the standardized delta is the part of D's advantage that is NOT")
    print(f"  explained by case mix. The gap between crude and standardized is the part that IS.")

    tag = "" if a.reference == "A_all_primary" else "_vs_Acomplete"
    dest = paths.EVAL_ROOT / f"e1a_standardized_{a.condition}{tag}.json"
    dest.write_text(json.dumps({"condition": a.condition, "description": label,
                                "per_seed": per_seed}, indent=2, default=str))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
