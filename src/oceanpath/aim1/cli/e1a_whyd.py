#!/usr/bin/env python3
"""Why does set D beat A-complete? (0 new fits)

D (MSS/pMMR & BRAF-WT) scores +0.022 above A-complete. Three candidate causes
remain after the earlier work, and this script separates them:

  A  Is it just smaller n?          random-restriction negative control
  B  WHICH pairs lose the ranking?  pairwise AUC decomposition
  C  Which molecular context is
     scored KRAS-like?              score distributions among KRAS-WT patients

A is the control the whole argument needs: an AUROC computed on 1,129 of 1,387
patients differs from the full-set AUROC for two reasons at once — WHICH
patients left, and simply that fewer remain. Removing 258 patients at random,
holding n and class balance fixed, isolates the second. The stratified variant
additionally holds cohort and colon/rectum composition fixed, leaving molecular
context as essentially the only systematic difference.

Usage:
    python aim1_challenge_sets.py why-d --arm 1a_pb_cap8192
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

from oceanpath.aim1 import evaluate, paths  # noqa: E402

SEEDS = (42, 43, 44)
N_RAND = 10000
N_BOOT = 2000


def load(arm: str, encoder: str = "univ1") -> dict[int, pd.DataFrame]:
    man = pd.read_csv(paths.DEV_MANIFEST)
    out = {}
    for s in SEEDS:
        p = paths.OUTPUT_ROOT / "train" / arm / encoder / f"seed{s}" / "oof_predictions.parquet"
        if not p.is_file():
            continue
        f = evaluate.to_patient_level(pd.read_parquet(p), man)
        f = f[f["msi_dmmr"].isin(["MSS/pMMR", "MSI/dMMR"])
              & f["braf"].isin(["mutant", "wild_type"])].reset_index(drop=True)
        f["D"] = f["msi_dmmr"].eq("MSS/pMMR") & f["braf"].eq("wild_type")
        f["mut"] = f["label"].astype(bool)
        f["site2"] = f["tumor_site_group"].where(f["tumor_site_group"].isin(["Colon", "Rectum"]))
        f["stratum"] = f["subcohort"].astype(str) + "|" + f["site2"].astype(str)
        out[s] = f
    return out


def auc_of_masks(order_pos: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """AUROC for many subsets at once.

    Rank-based rather than pairwise: for a subset, the within-subset rank of the
    element at sorted position j is cumsum(mask)[j], so one cumsum per subset
    replaces n_pos*n_neg comparisons. 10,000 subsets of 1,129 patients would be
    3.2e9 pair comparisons the naive way; this is 1.4e7 cell operations.
    """
    csum = np.cumsum(masks, axis=1)                      # (B, n) within-subset rank
    sel_pos = masks & order_pos[None, :]
    n_pos = sel_pos.sum(axis=1)
    n_neg = masks.sum(axis=1) - n_pos
    rank_sum = np.where(sel_pos, csum, 0).sum(axis=1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / np.maximum(n_pos * n_neg, 1)


def fmt_p(p: float) -> str:
    """Never print more precision than 10,000 resamples can carry."""
    return "<1e-4" if p <= 2 / (N_RAND + 1) else f"{p:.4f}"


def mc_p(draws: np.ndarray, observed: float) -> float:
    """Finite-resampling p-value, (k+1)/(B+1).

    The naive k/B reports 0 whenever no draw reaches the observed value, which
    claims more precision than 10,000 resamples can carry. With B = 10,000 the
    smallest reportable value is 1/10,001 and should be written as "< 1e-4",
    never as "p = 0".
    """
    k = int(np.sum(draws >= observed))
    return (k + 1) / (len(draws) + 1)


def analysis_A(f: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Random restriction at D's exact size and class balance."""
    g = f.sort_values("mean_logit").reset_index(drop=True)     # sort once
    n = len(g)
    is_pos = g["mut"].to_numpy()
    isD = g["D"].to_numpy()
    pos_idx = np.flatnonzero(is_pos)
    neg_idx = np.flatnonzero(~is_pos)
    n_pos_D = int((isD & is_pos).sum())
    n_neg_D = int((isD & ~is_pos).sum())

    auc_full = float(auc_of_masks(is_pos, np.ones((1, n), dtype=bool))[0])
    auc_D = float(auc_of_masks(is_pos, isD[None, :])[0])

    # ── A1: unstratified random restriction ────────────────────────────────
    masks = np.zeros((N_RAND, n), dtype=bool)
    for b in range(N_RAND):
        masks[b, rng.choice(pos_idx, n_pos_D, replace=False)] = True
        masks[b, rng.choice(neg_idx, n_neg_D, replace=False)] = True
    d1 = auc_of_masks(is_pos, masks) - auc_full

    # ── A2: stratified — also fixes cohort x site composition ──────────────
    cells: dict[tuple, np.ndarray] = {}
    want: dict[tuple, int] = {}
    for key, blk in g.groupby(["stratum", "mut"]):
        cells[key] = blk.index.to_numpy()
        want[key] = int(((g["stratum"] == key[0]) & (g["mut"] == key[1]) & isD).sum())
    masks2 = np.zeros((N_RAND, n), dtype=bool)
    for b in range(N_RAND):
        for key, pool in cells.items():
            k = want.get(key, 0)
            if k:
                masks2[b, rng.choice(pool, min(k, len(pool)), replace=False)] = True
    d2 = auc_of_masks(is_pos, masks2) - auc_full

    obs = auc_D - auc_full
    return {
        "auc_A_complete": auc_full, "auc_D": auc_D, "observed_delta": obs,
        "n_pos_D": n_pos_D, "n_neg_D": n_neg_D,
        "random": {"mean": float(d1.mean()), "sd": float(d1.std(ddof=1)),
                   "p2.5": float(np.percentile(d1, 2.5)), "p97.5": float(np.percentile(d1, 97.5)),
                   "p_ge_observed": mc_p(d1, obs), "p_note": "(k+1)/(B+1)"},
        "stratified": {"mean": float(d2.mean()), "sd": float(d2.std(ddof=1)),
                       "p2.5": float(np.percentile(d2, 2.5)),
                       "p97.5": float(np.percentile(d2, 97.5)),
                       "p_ge_observed": mc_p(d2, obs), "p_note": "(k+1)/(B+1)"},
    }


def pair_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties at 0.5."""
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties
    allv = np.concatenate([pos, neg])
    s = pd.Series(allv).rank(method="average").to_numpy()
    rp = s[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def analysis_B(f: pd.DataFrame, rng: np.random.Generator) -> dict:
    z = f["mean_logit"].to_numpy()
    Dp = z[(f["D"] & f["mut"]).to_numpy()]
    Dn = z[(f["D"] & ~f["mut"]).to_numpy()]
    Cp = z[(~f["D"] & f["mut"]).to_numpy()]
    Cn = z[(~f["D"] & ~f["mut"]).to_numpy()]
    groups = {"D+ vs D-": (Dp, Dn), "D+ vs C-": (Dp, Cn),
              "C+ vs D-": (Cp, Dn), "C+ vs C-": (Cp, Cn)}
    tot = (len(Dp) + len(Cp)) * (len(Dn) + len(Cn))
    out = {}
    for k, (p, q) in groups.items():
        boots = []
        for _ in range(N_BOOT):
            boots.append(pair_auc(rng.choice(p, len(p)), rng.choice(q, len(q))))
        boots = np.array([b for b in boots if np.isfinite(b)])
        out[k] = {"auc": pair_auc(p, q), "n_pos": int(len(p)), "n_neg": int(len(q)),
                  "pair_share": float(len(p) * len(q) / tot),
                  "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return out


def analysis_C(f: pd.DataFrame, rng: np.random.Generator) -> dict:
    out: dict = {}
    for kras_lab, sel in (("KRAS-WT", ~f["mut"]), ("KRAS-mutant", f["mut"])):
        blk = f[sel]
        grp = {}
        for (msi, braf), g in blk.groupby(["msi_dmmr", "braf"]):
            z = g["mean_logit"].to_numpy()
            bm = np.array([rng.choice(z, len(z)).mean() for _ in range(N_BOOT)])
            bmed = np.array([np.median(rng.choice(z, len(z))) for _ in range(N_BOOT)])
            grp[f"{msi}|BRAF-{braf}"] = {
                "n": int(len(z)), "mean_logit": float(z.mean()),
                "mean_ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                "median_logit": float(np.median(z)),
                "median_ci": [float(np.percentile(bmed, 2.5)), float(np.percentile(bmed, 97.5))],
                "mean_prob": float(np.mean(1 / (1 + np.exp(-z)))),
            }
        out[kras_lab] = grp
    return out


def analysis_D(f: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Adjusted MSI/BRAF association with the locked KRAS logit, KRAS-WT only.

        logit ~ BRAF + MSI + BRAF:MSI + cohort + site

    The descriptive result (MSS/BRAF-mutant WT tumours score highest) cannot
    separate BRAF from MSI, because the two are correlated and the four cells
    have very different sizes. OLS on the logit with an explicit interaction
    answers three separable questions: is BRAF independently associated, is MSI,
    and does their combination differ from the sum of the parts — each adjusted
    for cohort and colon/rectum.

    Inference is a patient bootstrap of the whole fit, so cohort composition
    uncertainty is inside the interval rather than conditioned away.
    """
    wt = f[~f["mut"]].copy()
    wt = wt[wt["site2"].notna()]
    wt["braf_mut"] = wt["braf"].eq("mutant").astype(float)
    wt["msi"] = wt["msi_dmmr"].eq("MSI/dMMR").astype(float)
    wt["inter"] = wt["braf_mut"] * wt["msi"]

    def design(d: pd.DataFrame):
        X = [np.ones(len(d)), d["braf_mut"].to_numpy(), d["msi"].to_numpy(),
             d["inter"].to_numpy()]
        names = ["intercept", "BRAF_mut", "MSI", "BRAF_mut x MSI"]
        for col, ref in (("cohort", None), ("site2", None)):
            levels = sorted(d[col].astype(str).unique())[1:]
            for lv in levels:
                X.append((d[col].astype(str) == lv).to_numpy(dtype=float))
                names.append(f"{col}={lv}")
        return np.column_stack(X), names

    X, names = design(wt)
    y = wt["mean_logit"].to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    boots = []
    idx = np.arange(len(wt))
    for _ in range(N_BOOT):
        b = rng.choice(idx, len(idx), replace=True)
        Xb, yb = X[b], y[b]
        if np.linalg.matrix_rank(Xb) < Xb.shape[1]:
            continue
        boots.append(np.linalg.lstsq(Xb, yb, rcond=None)[0])
    B = np.vstack(boots)
    out = {}
    for i, nm in enumerate(names):
        lo, hi = np.percentile(B[:, i], [2.5, 97.5])
        out[nm] = {"beta": float(beta[i]), "ci": [float(lo), float(hi)],
                   "excludes_zero": bool(lo > 0 or hi < 0)}
    # marginal (unadjusted) cell means for reference
    cells = {}
    for (msi, braf), g in wt.groupby(["msi_dmmr", "braf"]):
        cells[f"{msi}|BRAF-{braf}"] = {"n": int(len(g)),
                                       "mean_logit": float(g["mean_logit"].mean())}
    out["_cells"] = cells
    # the MSS/BRAF-mut contrast against the D reference, adjusted
    out["_MSS_BRAFmut_vs_reference_adjusted"] = out["BRAF_mut"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="1a_pb_cap8192")
    ap.add_argument("--encoder", default="univ1")
    a = ap.parse_args()
    frames = load(a.arm, a.encoder)
    if not frames:
        raise SystemExit(f"no OOF for {a.arm}/{a.encoder}")
    rng = np.random.default_rng(paths.BOOTSTRAP_SEED)
    rep: dict = {"arm": a.arm, "per_seed": {}}

    print(f"\n{'=' * 100}\nWHY DOES SET D BEAT A-COMPLETE?  arm={a.arm}  seeds={sorted(frames)}"
          f"\n{'=' * 100}")

    # ── A ──────────────────────────────────────────────────────────────────
    print(f"\n{'-' * 100}\nANALYSIS A — random-restriction negative control "
          f"({N_RAND:,} draws per seed)\n{'-' * 100}")
    print("  Removing 258 patients AT RANDOM, holding n and class balance fixed, isolates")
    print("  'fewer patients' from 'which patients'.\n")
    print(f"  {'seed':>5s} {'AUC A-comp':>11s} {'AUC D':>8s} {'observed d':>11s} "
          f"{'random d mean [2.5,97.5]':>30s} {'p':>7s} {'stratified d mean [2.5,97.5]':>32s} {'p':>7s}")
    As = {}
    for s, f in sorted(frames.items()):
        r = analysis_A(f, rng); As[s] = r
        print(f"  {s:5d} {r['auc_A_complete']:11.4f} {r['auc_D']:8.4f} {r['observed_delta']:+11.4f} "
              f"{f'{r[chr(114)+chr(97)+chr(110)+chr(100)+chr(111)+chr(109)][chr(109)+chr(101)+chr(97)+chr(110)]:+.4f} [{r[chr(114)+chr(97)+chr(110)+chr(100)+chr(111)+chr(109)][chr(112)+chr(50)+chr(46)+chr(53)]:+.4f},{r[chr(114)+chr(97)+chr(110)+chr(100)+chr(111)+chr(109)][chr(112)+chr(57)+chr(55)+chr(46)+chr(53)]:+.4f}]':>30s} "
              f"{fmt_p(r['random']['p_ge_observed']):>7s} "
              f"{f'{r[chr(115)+chr(116)+chr(114)+chr(97)+chr(116)+chr(105)+chr(102)+chr(105)+chr(101)+chr(100)][chr(109)+chr(101)+chr(97)+chr(110)]:+.4f} [{r[chr(115)+chr(116)+chr(114)+chr(97)+chr(116)+chr(105)+chr(102)+chr(105)+chr(101)+chr(100)][chr(112)+chr(50)+chr(46)+chr(53)]:+.4f},{r[chr(115)+chr(116)+chr(114)+chr(97)+chr(116)+chr(105)+chr(102)+chr(105)+chr(101)+chr(100)][chr(112)+chr(57)+chr(55)+chr(46)+chr(53)]:+.4f}]':>32s} "
              f"{fmt_p(r['stratified']['p_ge_observed']):>7s}")
    rep["A"] = As

    # ── B ──────────────────────────────────────────────────────────────────
    print(f"\n{'-' * 100}\nANALYSIS B — pairwise AUC decomposition\n{'-' * 100}")
    print("  Every A-complete positive-negative pair falls in exactly one cell. If the drop is")
    print("  driven by context-positive WT tumours scoring KRAS-like, 'D+ vs C-' is the cell.\n")
    Bs = {s: analysis_B(f, rng) for s, f in sorted(frames.items())}
    rep["B"] = Bs
    keys = list(Bs[sorted(Bs)[0]])
    print(f"  {'comparison':12s} {'n+':>5s} {'n-':>5s} {'pairs':>8s} "
          f"{'AUC median [min-max]':>26s} {'median 95% CI':>18s}")
    for k in keys:
        v = [Bs[s][k]["auc"] for s in sorted(Bs)]
        r0 = Bs[sorted(Bs)[0]][k]
        lo = np.median([Bs[s][k]["ci"][0] for s in sorted(Bs)])
        hi = np.median([Bs[s][k]["ci"][1] for s in sorted(Bs)])
        print(f"  {k:12s} {r0['n_pos']:5d} {r0['n_neg']:5d} {r0['pair_share']:8.2%} "
              f"{np.median(v):.4f} [{min(v):.4f},{max(v):.4f}]".rjust(28)
              + f"{f'{lo:.3f}-{hi:.3f}':>18s}")

    # ── C ──────────────────────────────────────────────────────────────────
    print(f"\n{'-' * 100}\nANALYSIS C — score distributions by molecular context\n{'-' * 100}")
    Cs = {s: analysis_C(f, rng) for s, f in sorted(frames.items())}
    rep["C"] = Cs
    for lab in ("KRAS-WT", "KRAS-mutant"):
        print(f"\n  {lab} patients only:")
        print(f"    {'context':28s} {'n':>5s} {'mean logit [95% CI]':>28s} "
              f"{'median logit':>13s} {'mean P(KRAS)':>13s}")
        g0 = Cs[sorted(Cs)[0]][lab]
        for ctx in sorted(g0, key=lambda c: -Cs[sorted(Cs)[0]][lab][c]["mean_logit"]):
            mv = [Cs[s][lab][ctx]["mean_logit"] for s in sorted(Cs)]
            lo = np.median([Cs[s][lab][ctx]["mean_ci"][0] for s in sorted(Cs)])
            hi = np.median([Cs[s][lab][ctx]["mean_ci"][1] for s in sorted(Cs)])
            md = np.median([Cs[s][lab][ctx]["median_logit"] for s in sorted(Cs)])
            pr = np.median([Cs[s][lab][ctx]["mean_prob"] for s in sorted(Cs)])
            tag = "   <- D reference" if ctx.startswith("MSS/pMMR|BRAF-wild") else ""
            print(f"    {ctx:28s} {g0[ctx]['n']:5d} "
                  f"{np.median(mv):+.3f} [{lo:+.3f}, {hi:+.3f}]".rjust(30)
                  + f"{md:13.3f} {pr:13.3f}{tag}")
        if lab == "KRAS-mutant":
            print("    (context-positive mutant cells are n=4/5/32 — DESCRIPTIVE ONLY)")

    # ── D ──────────────────────────────────────────────────────────────────
    print(f"\n{'-' * 100}\nANALYSIS D — adjusted MSI x BRAF model, KRAS-WT patients only"
          f"\n{'-' * 100}")
    print("  outcome = locked KRAS logit;  logit ~ BRAF + MSI + BRAF:MSI + cohort + site")
    print("  patient bootstrap of the whole fit (2,000), so cohort composition uncertainty")
    print("  is inside the interval.\n")
    Ds = {s: analysis_D(f, rng) for s, f in sorted(frames.items())}
    rep["D"] = Ds
    s0 = sorted(Ds)[0]
    terms = [k for k in Ds[s0] if not k.startswith("_")]
    print(f"  {'term':22s} {'beta median [min-max]':>28s} {'median 95% CI':>22s} {'excl 0':>7s}")
    for t in terms:
        bs = [Ds[s][t]["beta"] for s in sorted(Ds)]
        lo = np.median([Ds[s][t]["ci"][0] for s in sorted(Ds)])
        hi = np.median([Ds[s][t]["ci"][1] for s in sorted(Ds)])
        excl = all(Ds[s][t]["excludes_zero"] for s in sorted(Ds))
        star = "  <-" if t in ("BRAF_mut", "MSI", "BRAF_mut x MSI") else ""
        print(f"  {t:22s} {np.median(bs):+.3f} [{min(bs):+.3f},{max(bs):+.3f}]".ljust(52)
              + f"[{lo:+.3f}, {hi:+.3f}]".rjust(22) + f"{str(excl):>7s}{star}")
    print("\n  Reading: BRAF_mut is the adjusted BRAF effect in MSS tumours; MSI is the adjusted")
    print("  MSI effect in BRAF-WT tumours; the interaction is how much their combination")
    print("  departs from the sum. All are on the logit scale of the locked KRAS score.")

    # The filename must carry the ENCODER as well as the arm. Virchow2 and
    # UNI-v1 share the arm directory `1a_pb_cap4096`, so a name built from the
    # arm alone made `--encoder virchow2_cls` silently overwrite the UNI-v1
    # result. The pinned encoder keeps the original name so every existing
    # reference to these files stays valid.
    tag = a.arm if a.encoder == paths.PINNED_ENCODER else f"{a.arm}_{a.encoder}"
    dest = paths.EVAL_ROOT / f"e1_why_D_{tag}.json"
    rep["encoder"] = a.encoder
    dest.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
