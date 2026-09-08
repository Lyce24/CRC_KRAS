#!/usr/bin/env python3
"""E2d-3 - E2b repeated inside Aim 1's Set D (MSS/pMMR AND BRAF-wild-type).

QUESTION. Aim 1's headline is that the KRAS
signal STRENGTHENS in the dependency-robust population: restricting to
MSS/pMMR AND BRAF-wild-type raises AUROC, and E2a showed that strengthening
survives institutional transport in all four LOCO directions. E2b observed
negative primary -> metastatic point contrasts on the full population. The
question is whether Set D changes that gap. The change itself is bootstrapped
directly; subtracting two independently computed confidence intervals is not a
valid test.

IT ALSO REMOVES THE CONFOUND E2d-2 FOUND. Set D drops every BRAF-mutant patient,
and E2d-2 showed the SurGen peritoneal wild-type elevation is concentrated in
BRAF-mutant, KRAS-wild-type cases. If the metastatic penalty is that confound, it
should shrink here. If it persists, it is not.

NO TRAINING, NO ADAPTATION, SAME FROZEN PREDICTIONS. This is a restriction of the
E2b contrast, computed on exactly the arms E2b used:

    RIH      RIH-P inside Set D vs RIH-M inside Set D, with dual-role patients
             excluded from both contrast arms
    SurGen   SR1482-P inside Set D                  vs   SR1482-M inside Set D

SET D IS NOT SPLIT FURTHER. No organ breakdown, no subcohort breakdown - at
48-57 metastatic patients a second split would leave arms too small to read, and
the point of this analysis is a single clean number per cohort.

Usage:
    python aim2_setd_role_contrast.py report --cap 8192
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim2_loco_transport  # noqa: E402
import aim2_metastatic_transport  # noqa: E402
from oceanpath.aim1 import lineage, paths, population  # noqa: E402


def set_d(frame: pd.DataFrame) -> pd.DataFrame:
    """MSS/pMMR AND BRAF-wild-type — Aim 1's decisive population."""
    return frame[frame["msi_dmmr"].eq("MSS/pMMR") & frame["braf"].eq("wild_type")]


def _restriction_change_from_frames(
    primary: pd.DataFrame,
    metastatic: pd.DataFrame,
) -> tuple[float, float, float]:
    """Return (full delta, Set-D delta, Set-D-minus-full change)."""
    primary_d = set_d(primary)
    metastatic_d = set_d(metastatic)
    if primary_d["label"].nunique() < 2 or metastatic_d["label"].nunique() < 2:
        return float("nan"), float("nan"), float("nan")
    delta_full = float(aim2_loco_transport._auroc(metastatic) - aim2_loco_transport._auroc(primary))
    delta_d = float(aim2_loco_transport._auroc(metastatic_d) - aim2_loco_transport._auroc(primary_d))
    return delta_full, delta_d, delta_d - delta_full


def restriction_change_ci(
    primary: pd.DataFrame,
    metastatic: pd.DataFrame,
    n_boot: int = aim2_metastatic_transport.DEFAULT_N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict:
    """Directly bootstrap how Set-D restriction changes the M-minus-P gap.

    Set D is nested inside each full arm.  A valid replicate therefore samples
    each *full* patient arm once and derives both full and Set-D AUROCs from that
    same resample.  Resampling four endpoints independently would discard this
    dependence and cannot support a claim that restriction widens the gap.
    """
    aim2_metastatic_transport._validate_patient_arm(primary, "full primary arm")
    aim2_metastatic_transport._validate_patient_arm(metastatic, "full metastatic arm")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    for name, subset in (("Set-D primary arm", set_d(primary)),
                         ("Set-D metastatic arm", set_d(metastatic))):
        aim2_metastatic_transport._validate_patient_arm(subset, name)

    full_point, setd_point, change_point = _restriction_change_from_frames(
        primary, metastatic
    )
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        primary_b = aim2_metastatic_transport._resample_patient_arm(primary, rng)
        metastatic_b = aim2_metastatic_transport._resample_patient_arm(metastatic, rng)
        _, _, change = _restriction_change_from_frames(primary_b, metastatic_b)
        if np.isfinite(change):
            draws.append(change)
    if not draws:
        raise RuntimeError("No valid Set-D restriction-change bootstrap replicates")
    values = np.asarray(draws, dtype=float)
    ci = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    if ci[1] < 0:
        inference = "Set D makes the metastatic-minus-primary contrast more negative"
    elif ci[0] > 0:
        inference = "Set D makes the metastatic-minus-primary contrast more positive"
    else:
        inference = "change in the metastatic gap is not established"
    return {
        "estimand": "[AUROC(M)-AUROC(P)]_SetD - [AUROC(M)-AUROC(P)]_full",
        "full_delta_auroc": full_point,
        "set_d_delta_auroc": setd_point,
        "change_delta_auroc": change_point,
        "change_delta_auroc_ci": ci,
        "n_bootstrap_requested": int(n_boot),
        "n_bootstrap_valid": int(len(values)),
        "bootstrap_seed": int(seed),
        "bootstrap_fraction_below_zero": float(np.mean(values < 0)),
        "bootstrap_method": (
            "independent outcome-stratified resampling of the disjoint full "
            "primary/metastatic arms; Set D derived inside the same arm resample"
        ),
        "ci_excludes_zero": bool(ci[1] < 0 or ci[0] > 0),
        "inference": inference,
    }


def combined_restriction_change_ci(
    arms: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    n_boot: int = aim2_metastatic_transport.DEFAULT_N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict:
    """Equal-cohort inference for the Set-D-minus-full change in gap."""
    if len(arms) < 2:
        raise ValueError("combined restriction change requires at least two cohorts")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    for cohort, (primary, metastatic) in arms.items():
        aim2_metastatic_transport._validate_patient_arm(primary, f"{cohort} full primary arm")
        aim2_metastatic_transport._validate_patient_arm(metastatic, f"{cohort} full metastatic arm")
        aim2_metastatic_transport._validate_patient_arm(set_d(primary), f"{cohort} Set-D primary arm")
        aim2_metastatic_transport._validate_patient_arm(set_d(metastatic), f"{cohort} Set-D metastatic arm")

    points = {
        cohort: _restriction_change_from_frames(primary, metastatic)[2]
        for cohort, (primary, metastatic) in arms.items()
    }
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        changes = []
        for primary, metastatic in arms.values():
            primary_b = aim2_metastatic_transport._resample_patient_arm(primary, rng)
            metastatic_b = aim2_metastatic_transport._resample_patient_arm(metastatic, rng)
            change = _restriction_change_from_frames(primary_b, metastatic_b)[2]
            if not np.isfinite(change):
                changes = []
                break
            changes.append(change)
        if changes:
            draws.append(float(np.mean(changes)))
    if not draws:
        raise RuntimeError("No valid combined restriction-change bootstrap replicates")
    values = np.asarray(draws, dtype=float)
    ci = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    if ci[1] < 0:
        inference = "Set D makes the metastatic-minus-primary contrast more negative overall"
    elif ci[0] > 0:
        inference = "Set D makes the metastatic-minus-primary contrast more positive overall"
    else:
        inference = "overall change in the metastatic gap is not established"
    return {
        "estimand": "equal-cohort mean of Set-D-minus-full change in AUROC(M)-AUROC(P)",
        "change_delta_auroc": float(np.mean(list(points.values()))),
        "change_delta_auroc_ci": ci,
        "per_cohort_change_delta_auroc": points,
        "n_bootstrap_requested": int(n_boot),
        "n_bootstrap_valid": int(len(values)),
        "bootstrap_seed": int(seed),
        "bootstrap_fraction_below_zero": float(np.mean(values < 0)),
        "bootstrap_method": (
            "nested within-arm Set-D bootstrap in each cohort, with equal cohort weights"
            "; inference is conditional on the frozen fitted ensemble"
        ),
        "ci_excludes_zero": bool(ci[1] < 0 or ci[0] > 0),
        "inference": inference,
    }


def cmd_report(args: argparse.Namespace) -> None:
    cap = args.cap
    final_dest = lineage.eval_root() / f"e2d3_setd_contrast_cap{cap}.json"
    lineage.ensure_absent(final_dest)
    primary, met = aim2_loco_transport.load_primary(), aim2_loco_transport.load_metastatic()
    dual = population.dual_specimen_patients(primary, met)
    upstream = lineage.eval_root() / f"e2b_metastatic_cap{cap}.json"
    report: dict = {
        "cap": cap,
        "lineage": lineage.lineage_name(),
        "population": "Set D = MSS/pMMR AND BRAF-wild-type",
        "upstream_e2b": lineage.artifact_identity(upstream),
        "cohorts": {},
        "inference": {
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "sampling_unit": "patient",
            "training_seed_role": "computational stability only; never an inferential replicate",
            "conditioning": "confidence intervals condition on the frozen fitted ensemble",
        },
    }

    print(f"\n{'=' * 116}\nE2d-3 · E2b repeated inside Set D (MSS/pMMR ∧ BRAF-wild-type) — "
          f"0 new fits\n{'=' * 116}")
    rows = []
    contrast_arms: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for target in aim2_loco_transport.MET_TARGETS:
        ens_p, seed_p = aim2_loco_transport.seed_ensemble(target, "primary", cap)
        ens_m, seed_m = aim2_loco_transport.seed_ensemble(target, "metastatic", cap)
        if set(seed_p) != set(aim2_loco_transport.SEEDS) or set(seed_m) != set(aim2_loco_transport.SEEDS):
            raise RuntimeError(
                f"{target}: incomplete ensemble; primary={sorted(seed_p)}, "
                f"metastatic={sorted(seed_m)}, expected={list(aim2_loco_transport.SEEDS)}"
            )
        if target == "SurGen":
            p_all = ens_p[ens_p["subcohort"].eq(aim2_loco_transport.SURGEN_MET_SUBCOHORT)]
            m_all = ens_m
            note = f"{aim2_loco_transport.SURGEN_MET_SUBCOHORT}-P vs {aim2_loco_transport.SURGEN_MET_SUBCOHORT}-M"
        else:
            p_all = ens_p[~ens_p["patient_id"].isin(dual)]
            m_all = ens_m[~ens_m["patient_id"].isin(dual)]
            note = "RIH-P vs RIH-M (dual-role patients excluded from both arms)"
        p_d, m_d = set_d(p_all), set_d(m_all)

        blk = {"definition": note,
               "full_population": {
                   "primary": aim2_loco_transport.block_metrics(
                       p_all,
                       n_bootstrap=args.n_bootstrap,
                       bootstrap_seed=args.bootstrap_seed,
                   ),
                   "metastatic": aim2_loco_transport.block_metrics(
                       m_all,
                       n_bootstrap=args.n_bootstrap,
                       bootstrap_seed=args.bootstrap_seed,
                   ),
                   **aim2_metastatic_transport.contrast_ci(
                       p_all,
                       m_all,
                       n_boot=args.n_bootstrap,
                       seed=args.bootstrap_seed,
                   )},
               "set_d": {
                   "primary": aim2_loco_transport.block_metrics(
                       p_d,
                       n_bootstrap=args.n_bootstrap,
                       bootstrap_seed=args.bootstrap_seed,
                   ),
                   "metastatic": aim2_loco_transport.block_metrics(
                       m_d,
                       n_bootstrap=args.n_bootstrap,
                       bootstrap_seed=args.bootstrap_seed,
                   ),
                   **aim2_metastatic_transport.contrast_ci(
                       p_d,
                       m_d,
                       n_boot=args.n_bootstrap,
                       seed=args.bootstrap_seed,
                   )},
               "retention": {"primary": f"{len(p_d)}/{len(p_all)}",
                             "metastatic": f"{len(m_d)}/{len(m_all)}"},
               "restriction_change": restriction_change_ci(
                   p_all,
                   m_all,
                   n_boot=args.n_bootstrap,
                   seed=args.bootstrap_seed,
               )}

        # Same-patient per-seed values are retained solely to diagnose numerical
        # / optimisation stability.  They are correlated fits, not six samples.
        blk["set_d"]["per_seed_delta_descriptive"] = {}
        for s in aim2_loco_transport.SEEDS:
            ps, ms = set_d(seed_p[s]), set_d(seed_m[s])
            ps = ps[ps["patient_id"].isin(set(p_d["patient_id"]))]
            ms = ms[ms["patient_id"].isin(set(m_d["patient_id"]))]
            blk["set_d"]["per_seed_delta_descriptive"][s] = (
                aim2_loco_transport._auroc(ms) - aim2_loco_transport._auroc(ps)
            )
        blk["set_d"]["per_seed_inferential_role"] = "none"
        report["cohorts"][target] = blk
        rows.append((target, blk))
        contrast_arms[target] = (p_all, m_all)

    if len(contrast_arms) == len(aim2_loco_transport.MET_TARGETS):
        report["combined_restriction_change"] = combined_restriction_change_ci(
            contrast_arms,
            n_boot=args.n_bootstrap,
            seed=args.bootstrap_seed,
        )

    print(f"\n  {'Target':8s} {'Primary Set D AUROC':>28s} {'Metastatic Set D AUROC':>28s} "
          f"{'Δ M−P':>9s} {'95% CI':>20s}")
    for target, blk in rows:
        d = blk["set_d"]
        pp, mm = d["primary"], d["metastatic"]
        print(f"  {target:8s} {pp['auroc']:9.4f} [{pp['auroc_ci'][0]:.3f},"
              f"{pp['auroc_ci'][1]:.3f}] n={pp['n']:<4d} "
              f"{mm['auroc']:9.4f} [{mm['auroc_ci'][0]:.3f},{mm['auroc_ci'][1]:.3f}] "
              f"n={mm['n']:<4d} {d['delta_auroc']:+9.4f} "
              f"[{d['delta_auroc_ci'][0]:+.3f},{d['delta_auroc_ci'][1]:+.3f}]")

    print(f"\n  {'Target':8s} {'Primary AUPRC':>15s} {'Metastatic AUPRC':>18s} "
          f"{'Δ AUPRC':>10s} {'95% CI':>20s}")
    for target, blk in rows:
        d = blk["set_d"]
        print(f"  {target:8s} {d['primary']['auprc']:15.4f} {d['metastatic']['auprc']:18.4f} "
              f"{d['delta_auprc']:+10.4f} "
              f"[{d['delta_auprc_ci'][0]:+.3f},{d['delta_auprc_ci'][1]:+.3f}]")

    print(f"\n  {'Target':8s} {'retention (P)':>14s} {'retention (M)':>14s} "
          f"{'Δ full pop':>11s} {'Δ Set D':>10s} {'change':>9s}")
    for target, blk in rows:
        f_, d = blk["full_population"], blk["set_d"]
        print(f"  {target:8s} {blk['retention']['primary']:>14s} "
              f"{blk['retention']['metastatic']:>14s} "
              f"{f_['delta_auroc']:+11.4f} {d['delta_auroc']:+10.4f} "
              f"{d['delta_auroc'] - f_['delta_auroc']:+9.4f}")

    print("\n  per-seed Δ inside Set D (computational stability only):")
    for target, blk in rows:
        v = blk["set_d"]["per_seed_delta_descriptive"]
        print(f"    {target:8s} " + "  ".join(f"seed {s}: {x:+.4f}" for s, x in v.items()))
    print("    These fits share evaluation patients and training data; no seed count or "
          "sign pattern is used for inference.")

    print("\n  DIRECT TEST: does Set D change the metastatic gap?")
    for target, blk in rows:
        change = blk["restriction_change"]
        ci = change["change_delta_auroc_ci"]
        print(f"    {target:8s} change {change['change_delta_auroc']:+.4f} "
              f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  -> {change['inference']}")
    combined = report.get("combined_restriction_change")
    if combined:
        ci = combined["change_delta_auroc_ci"]
        print(f"    {'combined':8s} change {combined['change_delta_auroc']:+.4f} "
              f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  -> {combined['inference']}")

    dest = final_dest
    lineage.write_json_once(dest, report)
    print(f"\nWrote {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("report")
    x.add_argument("--cap", type=int, required=True, choices=aim2_loco_transport.E2A_CAPS)
    x.add_argument("--n-bootstrap", type=int, default=aim2_metastatic_transport.DEFAULT_N_BOOTSTRAP)
    x.add_argument("--bootstrap-seed", type=int, default=paths.BOOTSTRAP_SEED)
    x.set_defaults(func=cmd_report)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
