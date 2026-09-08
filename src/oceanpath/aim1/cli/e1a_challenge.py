#!/usr/bin/env python3
"""E1a — dependency-aware challenge sets (Tier 1, ZERO new fits).

Re-analysis of E0's frozen out-of-fold predictions. The model is unchanged, no
retraining and no rethresholding: every set is a RESTRICTION of the evaluation
population, which is what makes this the deployment-relevant question — given a
model trained in the real world, is its discrimination inside a molecularly
homogeneous population still real?

The estimator matters, because the obvious one is wrong. Every set is a subset
of A and the model is fixed, so "AUROC of A restricted to D" is identically
AUROC(D); a naive paired bootstrap returns exactly zero. Inference on
AUROC(A) - AUROC(D) uses a SHARED-RESAMPLE bootstrap: resample patients once
from A, then compute AUROC(A_boot) and AUROC(D and A_boot) on that same
resample.

SCOPE LIMIT, stated because it is easy to overclaim: the per-cohort panels
evaluate the ALL-COHORT model within each cohort. That is cohort-CONDITIONED
evaluation, not cohort-held-out validation, and it cannot support a "not
institution" claim. Institutional transport is established by E2a and by
nothing else. Aim 1 establishes molecular and clinicopathologic specificity.

Usage:
    python aim1_challenge_sets.py challenge-sets report --model 1a --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]  # src/oceanpath/aim1/cli/<this> -> repo root
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import evaluate, paths  # noqa: E402

# Main figure: A -> B -> D, plus E vs F. C, G, H, I-K are supplement.
SETS = {
    "A_all_primary": ("All KRAS primary", "main"),
    "A_complete": ("All primary with MSI AND BRAF known", "main"),
    "B_mss": ("MSS/pMMR only", "main"),
    "C_braf_wt": ("BRAF wild-type only", "supplement"),
    "D_mss_braf_wt": ("MSS/pMMR AND BRAF-WT (decisive)", "main"),
    "E_colon": ("Colon only", "main"),
    "F_rectum": ("Rectum only", "main"),
    "G_stage_known": ("Stage known (derived stage, §0.6)", "supplement"),
    "G_stage_known_frozen": ("Stage known (frozen manifest, pre-§0.6)", "supplement"),
    "H_stage_iv": ("Stage IV only, derived (exploratory)", "supplement"),
    "I_right_proximal": ("Right/proximal colon", "supplement"),
    "J_left_distal": ("Left/distal colorectum", "supplement"),
    "K_transverse": ("Transverse colon (exploratory)", "supplement"),
}

# ── §0.6 covariates, joined rather than baked into the manifest ──────────────
# `aim1_dev.csv` deliberately does NOT carry the derived stage or the derived
# sidedness: its SHA is part of every run's training-identity fingerprint, so
# rewriting it would invalidate the provenance of every completed fit. §0.6
# instructs analyses that want those columns to join them from
# `crc_final_v4.csv` on `patient_uid`, which is what this does — once, cached,
# and only for the evaluation-side masks. Nothing here touches training.
_EXTRA: pd.DataFrame | None = None


def _extra_covariates() -> pd.DataFrame:
    global _EXTRA
    if _EXTRA is None:
        source = pd.read_csv(paths.LABEL_SOURCE, low_memory=False)
        source = source.drop_duplicates("patient_uid")
        _EXTRA = source[["patient_uid", "sidedness", "stage_group_major_filled"]].rename(
            columns={"patient_uid": "patient_id"}
        )
    return _EXTRA


def with_derived_columns(patients: pd.DataFrame) -> pd.DataFrame:
    """Attach `sidedness` and `stage_group_major_filled` if not already present."""
    if {"sidedness", "stage_group_major_filled"} <= set(patients.columns):
        return patients
    return patients.merge(_extra_covariates(), on="patient_id", how="left")


def subset_mask(patients: pd.DataFrame, key: str) -> pd.Series:
    """Boolean mask for one challenge set. Missing context is FALSE, never NA:
    a patient whose stage is unrecorded is outside the stage-restricted set,
    not silently propagated as an unknown."""
    if key in ("G_stage_known", "H_stage_iv", "I_right_proximal",
               "J_left_distal", "K_transverse"):
        patients = with_derived_columns(patients)
    if key == "A_all_primary":
        return pd.Series(True, index=patients.index)
    if key == "A_complete":
        # The label-completeness comparator. Set D can only contain patients
        # whose MSI and BRAF are BOTH known, while set A also contains 99
        # patients with a missing molecular label. Comparing D against raw A
        # therefore confounds "conditioning on MSI/BRAF" with "dropping the
        # label-incomplete". A_complete removes that confound by matching D's
        # ascertainment without matching its molecular restriction.
        return patients["msi_dmmr"].isin(["MSS/pMMR", "MSI/dMMR"]).fillna(False) & patients[
            "braf"
        ].isin(["mutant", "wild_type"]).fillna(False)
    if key == "B_mss":
        return patients["msi_dmmr"].eq("MSS/pMMR").fillna(False)
    if key == "C_braf_wt":
        return patients["braf"].eq("wild_type").fillna(False)
    if key == "D_mss_braf_wt":
        return patients["msi_dmmr"].eq("MSS/pMMR").fillna(False) & patients["braf"].eq(
            "wild_type"
        ).fillna(False)
    if key == "E_colon":
        return patients["tumor_site_group"].eq("Colon").fillna(False)
    if key == "F_rectum":
        return patients["tumor_site_group"].eq("Rectum").fillna(False)
    if key == "G_stage_known":
        # The DERIVED stage of §0.6: the AJCC major group recomputed from TNM
        # for the 297 rows that had none, validated at 99.49% agreement on the
        # 1,366 rows that already carried one. This moves the development
        # stage-known set from 1,105 to 1,270 — the whole of the SR1482 gain.
        return (
            patients["stage_group_major_filled"]
            .astype("string")
            .isin(["I", "II", "III", "IV"])
            .fillna(False)
        )
    if key == "G_stage_known_frozen":
        # The pre-§0.6 set, kept so every number reported before the stage fix
        # stays reproducible and the two are directly comparable.
        return patients["stage_class"].isin(["I-II", "III-IV"]).fillna(False)
    if key == "H_stage_iv":
        return patients["stage_group_major_filled"].astype("string").eq("IV").fillna(False)
    # Sidedness (§0.6). TRANSVERSE IS ITS OWN LEVEL, never folded into right:
    # it straddles the midgut/hindgut watershed and the literature splits on it.
    # Coverage is 71% of development patients and the gap is not random —
    # TCGA-COAD is 1.7% resolved because 450 rows are coded only "Colon" — so
    # these rows are a RESTRICTION with an informative denominator, not a
    # partition of the cohort. Read them against each other, not against A.
    if key == "I_right_proximal":
        return patients["sidedness"].astype("string").eq("right").fillna(False)
    if key == "J_left_distal":
        return patients["sidedness"].astype("string").eq("left").fillna(False)
    if key == "K_transverse":
        return patients["sidedness"].astype("string").eq("transverse").fillna(False)
    raise ValueError(key)


def cmd_report(args: argparse.Namespace) -> None:
    patients = evaluate.load_oof(args.model, args.seed)
    # stage_group_major is needed for set H and is not a patient-level column
    # in the OOF frame; join it back from the frozen manifest.
    manifest = pd.read_csv(paths.DEV_MANIFEST).drop_duplicates("patient_id")
    patients = patients.merge(
        manifest[["patient_id", "stage_group_major"]], on="patient_id", how="left"
    )

    report: dict = {"model": args.model, "seed": args.seed, "sets": {}, "cohorts": {}}
    print(f"E1a · challenge sets · model {args.model} seed {args.seed} · no retraining")
    print("SIGN: dAUROC = AUROC(A) - AUROC(set). POSITIVE = the set scored LOWER than A")
    print("      (signal lost); NEGATIVE = it scored HIGHER (signal retained).\n")
    print(
        f"{'set':26s} {'n':>6s} {'mut':>5s} {'AUROC (95% CI)':>24s} {'AUPRC':>7s} "
        f"{'dAUROC vs A (95% CI)':>26s}"
    )

    for key, (title, tier) in SETS.items():
        mask = subset_mask(patients, key)
        block = patients[mask]
        tierlab = evaluate.support_tier(block["label"].to_numpy())
        delta = evaluate.shared_resample_delta(patients, mask)
        ap = evaluate.auprc_with_ci(block)
        entry = {
            "title": title,
            "figure": tier,
            "support": tierlab,
            "n": int(len(block)),
            "n_mutant": int(block["label"].sum()),
            "auroc": delta["auroc_subset"],
            "auroc_ci": [delta["subset_ci_low"], delta["subset_ci_high"]],
            "auprc": ap["auprc"],
            "auprc_ci": [ap["ci_low"], ap["ci_high"]],
            "auprc_baseline": ap["baseline"],
            "delta_vs_A": delta["delta"],
            "delta_ci": [delta["delta_ci_low"], delta["delta_ci_high"]],
        }
        report["sets"][key] = entry
        dtxt = (
            "—"
            if key == "A_all_primary"
            else f"{entry['delta_vs_A']:+.3f} ({entry['delta_ci'][0]:+.3f},{entry['delta_ci'][1]:+.3f})"
        )
        print(
            f"{key:26s} {entry['n']:6d} {entry['n_mutant']:5d} "
            f"{entry['auroc']:.3f} ({entry['auroc_ci'][0]:.3f}-{entry['auroc_ci'][1]:.3f})  "
            f"{entry['auprc']:7.3f} {dtxt:>26s}"
        )

    print("\nper-cohort (COHORT-CONDITIONED, not held-out — transport is E2a's claim):")
    for cohort, block in patients.groupby("cohort"):
        r = evaluate.bootstrap_auroc(block)
        report["cohorts"][str(cohort)] = r
        print(
            f"  {cohort:8s} n={r['n']:5d} mut={r['n_positive']:4d}  "
            f"{r['auroc']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f})"
        )

    d = patients[subset_mask(patients, "D_mss_braf_wt")]
    mut, wt = d[d.label == 1]["prob_raw"], d[d.label == 0]["prob_raw"]
    report["set_D_score_distribution"] = {
        "mutant_median": float(mut.median()),
        "wild_type_median": float(wt.median()),
        "mutant_iqr": [float(mut.quantile(0.25)), float(mut.quantile(0.75))],
        "wild_type_iqr": [float(wt.quantile(0.25)), float(wt.quantile(0.75))],
    }
    print(
        f"\nset D score distribution: mutant median {mut.median():.3f} "
        f"[{mut.quantile(0.25):.3f}-{mut.quantile(0.75):.3f}]  vs  "
        f"wild-type {wt.median():.3f} [{wt.quantile(0.25):.3f}-{wt.quantile(0.75):.3f}]"
    )

    a = report["sets"]["A_all_primary"]
    dd = report["sets"]["D_mss_braf_wt"]
    print(
        f"\nFALSIFIER CHECK — set D CI lower bound {dd['auroc_ci'][0]:.3f} vs chance 0.500: "
        f"{'SURVIVES' if dd['auroc_ci'][0] > 0.5 else 'THESIS INVERTS'}"
    )
    print(
        f"  A={a['auroc']:.3f} -> D={dd['auroc']:.3f}  "
        f"(delta {dd['delta_vs_A']:+.3f}, CI {dd['delta_ci'][0]:+.3f} to {dd['delta_ci'][1]:+.3f})"
    )

    out = paths.EVAL_ROOT / args.model / f"e1a_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    s = p.add_subparsers(dest="command", required=True)
    r = s.add_parser("report")
    r.add_argument("--model", default="1a")
    r.add_argument("--seed", type=int, default=paths.PRIMARY_SEED)
    r.set_defaults(func=cmd_report)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
