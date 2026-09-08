#!/usr/bin/env python3
"""E2d-1 - metastatic-site decomposition (Aim 2, 0 new fits).

THE ONE JOB. E2b observed lower metastatic point estimates and now tests the
combined decrement directly. E2d-1 asks whether performance is heterogeneous
by metastatic site. This is an exploratory decomposition motivated by the
observed E2b data, not a pre-specified independent confirmation; its site-wise
intervals are not multiplicity-adjusted.

If the failure is concentrated in particular metastatic sites, "the model does
not transport to metastases" is the wrong description of it. This is a
decomposition of frozen predictions, not a new model and not an interpretability
analysis - no attention maps, no morphology.

NEVER POOLED. RIH and SurGen come from different source models, different
score scales and different site mixes. A pooled per-site AUROC could manufacture
discrimination from between-cohort offsets, so every table below is within one
cohort. The two cohorts are compared only by whether they AGREE. Failure to
reproduce a site contrast is not evidence that metastatic site is irrelevant;
it only means a common site explanation was not established here.

SITE ASSIGNMENT. `metastatic_site_group` already separates liver / lung /
peritoneum cleanly (verified against `tumor_site_raw`: 19, 8 and 14 distinct raw
strings, all unambiguous). Its residual `other` bucket is what hides the sites
this analysis is about, so ONLY that bucket is decomposed, by explicit keyword
rules on `tumor_site_raw`. Every rule and every unmatched string is printed, so
the mapping is auditable rather than trusted.

Usage:
    python aim2_metastatic_site.py report --cap 8192
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim2_loco_transport  # noqa: E402
from oceanpath.aim1 import lineage, paths  # noqa: E402

COHORTS: tuple[str, ...] = ("RIH", "SurGen")

# Applied IN ORDER to tumor_site_raw, and only to rows the frozen
# metastatic_site_group calls "other". First match wins.
OTHER_RULES: tuple[tuple[str, str], ...] = (
    ("lymph_node", r"lymph\s*node|\bnodes\b|nodal|lymph node metasti"),
    ("ovary", r"\bovary\b|\bovarian\b|adnexa"),
)
SITE_ORDER = ("liver", "lung", "peritoneum", "lymph_node", "ovary", "other")
MIN_N = 8          # at or above this an arm gets a bootstrap CI
DESC_N = 4         # 4..7 patients: AUROC printed as DESCRIPTIVE only, no CI
BOOT = 10_000


def classify(site_group: str, raw: str) -> str:
    if site_group in ("liver", "lung", "peritoneum"):
        return site_group
    text = str(raw).strip().lower()
    for name, pattern in OTHER_RULES:
        if re.search(pattern, text):
            return name
    return "other"


def site_table(cap: int) -> dict[str, pd.DataFrame]:
    """Frozen 3-seed metastatic patient scores, labelled by metastatic site."""
    src = pd.read_csv(paths.LABEL_SOURCE, low_memory=False)
    met = src[src["specimen_role"].eq("metastatic")].copy()
    met["site"] = [classify(g, r) for g, r in
                   zip(met["metastatic_site_group"], met["tumor_site_raw"], strict=True)]
    out: dict[str, pd.DataFrame] = {}
    for target in COHORTS:
        ens, seeds = aim2_loco_transport.seed_ensemble(target, "metastatic", cap)
        if set(seeds) != set(aim2_loco_transport.SEEDS):
            raise RuntimeError(
                f"{target}: incomplete metastatic seed ensemble {sorted(seeds)}; "
                f"expected {list(aim2_loco_transport.SEEDS)}"
            )
        man = pd.read_csv(aim2_loco_transport.target_manifest(target, "metastatic"))
        keep = met[met["patient_uid"].isin(set(man["patient_id"]))]\
            if "patient_uid" in met.columns else met[met["cohort"].eq(target)]
        # one site per PATIENT; a patient with two metastatic slides must not be
        # counted twice, and a patient whose slides disagree on site is reported
        per_pat = keep.groupby(keep["patient_uid"] if "patient_uid" in keep.columns
                               else keep["patient_id"])["site"].agg(
            lambda s: sorted(set(s))[0] if len(set(s)) == 1 else "|".join(sorted(set(s))))
        ens = ens.assign(site=ens["patient_id"].map(per_pat))
        out[target] = ens
    return out


def block(
    frame: pd.DataFrame,
    n_boot: int = BOOT,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = frame["label"].to_numpy()
    eta = frame["mean_logit"].to_numpy()
    p = np.clip(frame["prob_raw"].to_numpy(), 1e-6, 1 - 1e-6)
    res = {"n": int(len(y)), "n_mut": int(y.sum()), "n_wt": int(len(y) - y.sum()),
           "prevalence": float(y.mean()) if len(y) else float("nan")}
    if len(y) < DESC_N or len(np.unique(y)) < 2:
        res["scorable"] = False
        return res
    from sklearn.metrics import roc_auc_score as _auc

    res["auroc"] = float(_auc(y, eta))
    res["auprc"] = float(average_precision_score(y, eta))
    res["brier"] = float(brier_score_loss(y, p))
    # Exact permutation over all label assignments — the only honest test at
    # these counts, and it costs nothing when n is this small.
    if len(y) <= 20:
        import itertools

        k = int(y.sum())
        vals = []
        for pos in itertools.combinations(range(len(y)), k):
            lab = np.zeros(len(y), int)
            lab[list(pos)] = 1
            vals.append(_auc(lab, eta))
        vals = np.array(vals)
        lo_p, hi_p = float((vals <= res["auroc"]).mean()), float((vals >= res["auroc"]).mean())
        # clipped at 1: both tails include the observed value, so when it sits
        # at the permutation median 2*min(...) can exceed 1.
        res["perm_p_two_sided"] = float(min(1.0, 2 * min(lo_p, hi_p)))
        res["perm_n_assignments"] = int(len(vals))
    if len(y) < MIN_N:
        res["scorable"] = "descriptive"      # printed, but no CI and no inference
        return res
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(float(roc_auc_score(y[idx], eta[idx])))
    res["scorable"] = True
    res["auroc_ci"] = ([float(np.percentile(draws, 2.5)),
                        float(np.percentile(draws, 97.5))] if draws else [float("nan")] * 2)
    res["n_bootstrap"] = int(len(draws))
    res["bootstrap_seed"] = int(seed)
    return res


def contrast(
    a: pd.DataFrame,
    b: pd.DataFrame,
    n_boot: int = BOOT,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict:
    """AUROC(a) - AUROC(b), independent resampling: disjoint patient sets."""
    from sklearn.metrics import roc_auc_score

    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    ya, ea = a["label"].to_numpy(), a["mean_logit"].to_numpy()
    yb, eb = b["label"].to_numpy(), b["mean_logit"].to_numpy()
    point = float(roc_auc_score(ya, ea) - roc_auc_score(yb, eb))
    draws = []
    for _ in range(n_boot):
        ia, ib = rng.integers(0, len(ya), len(ya)), rng.integers(0, len(yb), len(yb))
        if len(np.unique(ya[ia])) < 2 or len(np.unique(yb[ib])) < 2:
            continue
        draws.append(float(roc_auc_score(ya[ia], ea[ia]) - roc_auc_score(yb[ib], eb[ib])))
    return {"delta": point,
            "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
            if draws else [float("nan")] * 2,
            "n_bootstrap": int(len(draws)), "bootstrap_seed": int(seed),
            "bootstrap_method": "independent patient resampling of disjoint site arms"}


def cmd_report(args: argparse.Namespace) -> None:
    final_dest = lineage.eval_root() / f"e2d1_metastatic_sites_cap{args.cap}.json"
    lineage.ensure_absent(final_dest)
    tables = site_table(args.cap)
    upstream = lineage.eval_root() / f"e2b_metastatic_cap{args.cap}.json"
    report: dict = {
        "cap": args.cap,
        "lineage": lineage.lineage_name(),
        "cohorts": {},
        "note": "within-cohort only; RIH and SurGen are never pooled",
        "upstream_e2b": lineage.artifact_identity(upstream),
        "inference": {
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "sampling_unit": "patient",
        },
    }

    # audit the mapping before anything is read off it
    src = pd.read_csv(paths.LABEL_SOURCE, low_memory=False)
    met = src[src["specimen_role"].eq("metastatic") & src["cohort"].isin(COHORTS)]
    oth = met[met["metastatic_site_group"].eq("other")]
    reclassified = {}
    for _, row in oth.iterrows():
        s = classify("other", row["tumor_site_raw"])
        reclassified.setdefault(s, []).append(str(row["tumor_site_raw"]).strip().lower())
    print(f"\n{'=' * 112}\nE2d-1 · SITE MAPPING AUDIT — only the frozen 'other' bucket "
          f"is decomposed\n{'=' * 112}")
    for name, _pat in OTHER_RULES:
        vals = sorted(set(reclassified.get(name, [])))
        print(f"  {name:11s} <- {len(reclassified.get(name, []))} rows: "
              f"{'; '.join(vals) if vals else '(none)'}")
    rest = sorted(set(reclassified.get("other", [])))
    print(f"  {'other':11s} <- {len(reclassified.get('other', []))} rows remain "
          f"unmatched ({len(rest)} distinct)")
    print(f"      {'; '.join(rest[:12])}{' ...' if len(rest) > 12 else ''}")

    for target in COHORTS:
        df = tables[target]
        mixed = df[df["site"].astype(str).str.contains(r"\|", na=False)]
        block_out = {"n_total": int(len(df)), "sites": {},
                     "patients_with_discordant_sites": int(len(mixed))}
        print(f"\n{'=' * 112}\nE2d-1 · {target}-M metastatic-site decomposition "
              f"(n = {len(df)} patients)\n{'=' * 112}")
        print(f"  {'site':12s} {'N':>4s} {'mut':>4s} {'WT':>4s} {'prev':>6s} "
              f"{'AUROC':>8s} {'95% CI':>18s} {'AUPRC':>7s} {'Brier':>7s} {'exact p':>9s}")
        for site in SITE_ORDER:
            sub = df[df["site"].eq(site)]
            if not len(sub):
                print(f"  {site:12s} {0:4d} {'—':>4s} {'—':>4s} {'—':>6s} "
                      f"{'—':>8s} {'—':>18s} {'—':>7s} {'—':>7s} {'—':>9s}")
                block_out["sites"][site] = {"n": 0}
                continue
            b = block(sub, n_boot=args.n_bootstrap, seed=args.bootstrap_seed)
            block_out["sites"][site] = b
            pp = (f"p={b['perm_p_two_sided']:.3f}" if "perm_p_two_sided" in b else "—")
            if b.get("scorable") is True:
                ci = b["auroc_ci"]
                print(f"  {site:12s} {b['n']:4d} {b['n_mut']:4d} {b['n_wt']:4d} "
                      f"{b['prevalence']:6.2f} {b['auroc']:8.4f} "
                      f"[{ci[0]:6.3f},{ci[1]:6.3f}] {b['auprc']:7.4f} {b['brier']:7.4f} "
                      f"{pp:>9s}")
            elif b.get("scorable") == "descriptive":
                print(f"  {site:12s} {b['n']:4d} {b['n_mut']:4d} {b['n_wt']:4d} "
                      f"{b['prevalence']:6.2f} {b['auroc']:8.4f} "
                      f"{'descriptive only':>18s} {b['auprc']:7.4f} {b['brier']:7.4f} "
                      f"{pp:>9s}")
            else:
                why = f"n < {DESC_N}" if b["n"] < DESC_N else "single class"
                print(f"  {site:12s} {b['n']:4d} {b['n_mut']:4d} {b['n_wt']:4d} "
                      f"{b['prevalence']:6.2f} {'—':>8s} {'not scorable: ' + why:>18s} "
                      f"{'—':>7s} {'—':>7s} {'—':>9s}")
        if len(mixed):
            print(f"  NOTE {len(mixed)} patient(s) have metastatic slides from more than "
                  f"one site: {sorted(mixed['site'].unique())}")

        # liver vs everything else, within this cohort
        liv = df[df["site"].eq("liver")]
        non = df[~df["site"].eq("liver")]
        if len(liv) >= MIN_N and len(non) >= MIN_N:
            c = contrast(
                liv, non, n_boot=args.n_bootstrap, seed=args.bootstrap_seed
            )
            block_out["liver_vs_non_liver"] = c
            print(f"\n  liver vs non-liver   dAUROC {c['delta']:+.4f} "
                  f"95% CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]   "
                  f"(liver n={len(liv)}, non-liver n={len(non)})")
        per = df[df["site"].eq("peritoneum")]
        rest = df[~df["site"].isin(["liver", "peritoneum"])]
        if len(liv) >= MIN_N and len(per) >= MIN_N:
            c = contrast(
                liv, per, n_boot=args.n_bootstrap, seed=args.bootstrap_seed
            )
            block_out["liver_vs_peritoneum"] = c
            print(f"  liver vs peritoneum  dAUROC {c['delta']:+.4f} "
                  f"95% CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]   "
                  f"(peritoneum n={len(per)})")
        if len(liv) >= MIN_N and len(rest) >= MIN_N:
            c = contrast(
                liv, rest, n_boot=args.n_bootstrap, seed=args.bootstrap_seed
            )
            block_out["liver_vs_non_liver_excl_peritoneum"] = c
            print(f"  liver vs non-liver EXCLUDING peritoneum   dAUROC {c['delta']:+.4f} "
                  f"95% CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]   (n={len(rest)})")
        report["cohorts"][target] = block_out

    # do the two cohorts agree?
    ds = {t: report["cohorts"][t].get("liver_vs_non_liver") for t in COHORTS}
    if all(ds.values()):
        a, b = ds["RIH"], ds["SurGen"]
        excludes = {
            "RIH": bool(a["ci"][0] > 0 or a["ci"][1] < 0),
            "SurGen": bool(b["ci"][0] > 0 or b["ci"][1] < 0),
        }
        both_excl = all(excludes.values())
        same_sign = np.sign(a["delta"]) == np.sign(b["delta"])
        if same_sign and both_excl:
            verdict = "replicated"
        elif any(excludes.values()):
            verdict = "single-cohort finding"
        elif same_sign:
            verdict = "point directions align; effect not established"
        else:
            verdict = "not supported"
        report["concordance"] = {
            "same_sign": bool(same_sign), "both_cis_exclude_zero": bool(both_excl),
            "per_cohort_ci_excludes_zero": excludes,
            "verdict": verdict,
            "point_sign_inferential_role": "none without confidence-interval support",
        }
        print(f"\n{'=' * 112}\nDOES THE ORGAN EFFECT REPLICATE?\n{'=' * 112}")
        for t in COHORTS:
            d = ds[t]
            print(f"  {t:8s} liver − non-liver {d['delta']:+.4f} "
                  f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]   "
                  f"CI excludes 0: {d['ci'][0] > 0 or d['ci'][1] < 0}")
        print(f"\n  -> {report['concordance']['verdict'].upper()}")
        print("  One cohort showing an effect the other does not is a hypothesis, not a")
        print("  result. Site mix differs between the cohorts, and every site arm here")
        print("  is 1-50 patients.")
        n_arms = sum(1 for t in COHORTS for v in report["cohorts"][t]["sites"].values()
                     if v.get("scorable"))
        print(f"\n  MULTIPLICITY: {n_arms} site arms were scored across the two cohorts. "
              f"No p-value\n  below is corrected; at {n_arms} comparisons a nominal 0.05 "
              f"is ~{0.05 / max(n_arms, 1):.3f} after Bonferroni.")

    dest = final_dest
    lineage.write_json_once(dest, report)
    print(f"\nWrote {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("report")
    x.add_argument("--cap", type=int, required=True, choices=aim2_loco_transport.E2A_CAPS)
    x.add_argument("--n-bootstrap", type=int, default=BOOT)
    x.add_argument("--bootstrap-seed", type=int, default=paths.BOOTSTRAP_SEED)
    x.set_defaults(func=cmd_report)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
