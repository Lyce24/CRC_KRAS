#!/usr/bin/env python3
"""E0 + E1a reporting across seed replicates (cap vs uncap).

Reporting rules, chosen because the obvious summaries mislead at k=3:

* **Median + observed min-max, never a three-draw SD.** An SD from three draws
  is itself so noisy that quoting it implies precision the design cannot
  deliver. The range is what was actually observed.
* **Thresholds are chosen inside each seed's own scores.** Score scales differ
  between seeds, so seed 1's Youden point is meaningless applied to seed 2's
  output; a shared threshold would measure scale drift, not discrimination.
* **Two uncertainties, both quoted.** A patient-bootstrap CI captures sampling
  of patients but treats the trained model as fixed. Seed variance is a second,
  independent source. Total SE = sqrt(SE_patient^2 + (SD_seed/sqrt(k))^2);
  quoting the bootstrap alone understates total uncertainty.
* **E1a deltas are computed WITHIN a seed** (AUROC_A^(s) - AUROC_C^(s)) and
  then summarised across seeds, so seed variance never contaminates the
  restriction effect.

Usage:
    python aim1_locked_baseline.py --condition uncapped
    python aim1_locked_baseline.py --condition capped
    python aim1_locked_baseline.py --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1.cli import e1a_challenge as e1a  # noqa: E402
from oceanpath.aim1 import evaluate, paths  # noqa: E402
from oceanpath.eval.external import logit, sigmoid  # noqa: E402

TRAIN = paths.OUTPUT_ROOT / "train"
CONDITIONS = {
    "uncapped": (TRAIN / "1a" / "univ1", "full bags"),
    "cap2048": (TRAIN / "1a_cap2048" / "univ1", "2048 random tiles/epoch"),
    "capped": (TRAIN / "1a_cap4096" / "univ1", "4096 random tiles/epoch"),
    "cap8192": (TRAIN / "1a_cap8192" / "univ1", "8192 random tiles/epoch"),
    "cap16384": (TRAIN / "1a_cap16384" / "univ1", "16384 random tiles/epoch"),
    "pb_cap4096": (TRAIN / "1a_pb_cap4096" / "univ1", "PATIENT-BALANCED, 4096 tiles/epoch"),
    "pb_cap8192": (TRAIN / "1a_pb_cap8192" / "univ1", "PATIENT-BALANCED, 8192 tiles/epoch"),
    "conch": (TRAIN / "1a_conch" / "conch_v15", "CONCH v1.5, full bags"),
    "v2cls": (TRAIN / "1a_pb_cap4096" / "virchow2_cls", "VIRCHOW2 CLS, 4096 tiles/epoch, patient-balanced"),
}
SEEDS = (42, 43, 44)


def load_seed(root: Path, seed: int) -> pd.DataFrame | None:
    path = root / f"seed{seed}" / "oof_predictions.parquet"
    if not path.is_file():
        return None
    manifest = pd.read_csv(paths.DEV_MANIFEST)
    patients = evaluate.to_patient_level(pd.read_parquet(path), manifest)
    stage = manifest.drop_duplicates("patient_id")[["patient_id", "stage_group_major"]]
    return patients.merge(stage, on="patient_id", how="left")


def youden_within_seed(patients: pd.DataFrame) -> dict:
    """Youden point on THIS seed's own score scale."""
    from sklearn.metrics import roc_curve

    y, p = patients["label"].to_numpy(), patients["prob_raw"].to_numpy()
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr))
    return {
        "threshold": float(thr[i]),
        "sensitivity": float(tpr[i]),
        "specificity": float(1 - fpr[i]),
        "youden_j": float(tpr[i] - fpr[i]),
    }


def per_seed_metrics(patients: pd.DataFrame) -> dict:
    from sklearn.metrics import brier_score_loss

    cal = evaluate.cross_fitted_platt(patients)
    block = evaluate.calibration_block(cal["label"].to_numpy(), cal["prob_cal"].to_numpy())
    out = {
        "n": int(len(patients)),
        "auroc": evaluate.patient_auroc(patients),
        "auprc": evaluate.auprc_with_ci(patients, n_bootstrap=1)["auprc"],
        "brier": float(brier_score_loss(cal["label"], cal["prob_cal"])),
        "calibration_intercept": float(block["calibration_intercept"]),
        "calibration_slope": float(block["calibration_slope"]),
        "youden": youden_within_seed(patients),
        "per_cohort": {},
        "per_fold": [evaluate.patient_auroc(patients[patients.k_fold == f]) for f in range(5)],
    }
    for cohort, blk in patients.groupby("cohort"):
        out["per_cohort"][str(cohort)] = evaluate.patient_auroc(blk)
    return out


def summarise(values: list[float]) -> str:
    v = [x for x in values if np.isfinite(x)]
    if not v:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.4f} (1 seed)"
    return f"{np.median(v):.4f} [{min(v):.4f}-{max(v):.4f}]"


def ensemble(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Mean LOGIT across seeds per patient; the sigmoid comes after."""
    base = None
    stack = []
    reference_ids: list | None = None
    for _seed, f in sorted(frames.items()):
        f = f.sort_values("patient_id").reset_index(drop=True)
        ids = f["patient_id"].tolist()
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(
                "seeds cover different patient sets; stacking their logits would "
                "align scores against the wrong labels"
            )
        if base is None:
            base = f[
                [
                    "patient_id",
                    "label",
                    "cohort",
                    "msi_dmmr",
                    "braf",
                    "tumor_site_group",
                    "stage_class",
                    "stage_group_major",
                    "k_fold",
                ]
            ].copy()
        stack.append(logit(f["prob_raw"].to_numpy()))
    base["prob_raw"] = sigmoid(np.mean(np.vstack(stack), axis=0))
    return base


def report_condition(name: str, quiet: bool = False) -> dict:
    root, label = CONDITIONS[name]
    frames = {s: f for s in SEEDS if (f := load_seed(root, s)) is not None}
    if not frames:
        print(f"{name}: no completed seeds yet")
        return {}
    out: dict = {
        "condition": name,
        "description": label,
        "seeds_complete": sorted(frames),
        "per_seed": {},
    }
    for seed, f in sorted(frames.items()):
        out["per_seed"][seed] = per_seed_metrics(f)

    aurocs = [out["per_seed"][s]["auroc"] for s in sorted(frames)]
    ens = ensemble(frames)
    ens_boot = evaluate.bootstrap_auroc(ens)
    out["ensemble"] = ens_boot

    # TWO SEPARATE QUANTITIES, deliberately NOT combined into a headline number.
    #
    #   patient-bootstrap CI  -> the PRIMARY inferential uncertainty. It is what
    #                            a confidence statement about the population
    #                            means, and it is what should be quoted.
    #   seed range / SD       -> ALGORITHMIC ROBUSTNESS. It says how much the
    #                            answer moves when the optimiser is re-rolled on
    #                            the same patients and the same folds.
    #
    # A combined "total SE" is still emitted below for continuity, but it is NOT
    # the primary uncertainty and must not be presented as one: adding the two in
    # quadrature assumes independent, additive variance components, and no formal
    # variance model here justifies that. The two answer different questions and
    # are reported side by side.
    per_seed_se = []
    for seed in sorted(frames):
        b = evaluate.bootstrap_auroc(frames[seed])
        per_seed_se.append((b["ci_high"] - b["ci_low"]) / (2 * 1.96))
        out["per_seed"][seed]["patient_bootstrap_se"] = per_seed_se[-1]
    se_patient = float(np.median(per_seed_se))
    sd_seed = float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else float("nan")
    se_seed = sd_seed / np.sqrt(len(aurocs)) if len(aurocs) > 1 else float("nan")
    total = float(np.sqrt(se_patient**2 + se_seed**2)) if len(aurocs) > 1 else float("nan")
    out["uncertainty"] = {
        "se_patient_bootstrap": se_patient,
        "se_patient_basis": "median of the per-seed single-model bootstrap SEs",
        "se_patient_ensemble_for_reference": (ens_boot["ci_high"] - ens_boot["ci_low"]) / (2 * 1.96),
        "sd_across_seeds": sd_seed,
        "se_seed_mean": se_seed,
        "se_total": total,
        "se_total_caveat": "NOT the primary uncertainty; quadrature sum without a formal "
                           "variance model. Quote the patient-bootstrap CI and report the "
                           "seed range separately as algorithmic robustness.",
    }

    if quiet:
        return out
    print(f"\n{'=' * 78}\nE0 · {name.upper()} ({label}) · seeds {sorted(frames)}\n{'=' * 78}")
    print(f"{'metric':26s} {'median [min-max] over seeds':>34s}")
    for key in ("auroc", "auprc", "brier", "calibration_intercept", "calibration_slope"):
        print(f"  {key:24s} {summarise([out['per_seed'][s][key] for s in sorted(frames)]):>34s}")
    print(
        f"  {'youden sensitivity':24s} "
        f"{summarise([out['per_seed'][s]['youden']['sensitivity'] for s in sorted(frames)]):>34s}"
    )
    print(
        f"  {'youden specificity':24s} "
        f"{summarise([out['per_seed'][s]['youden']['specificity'] for s in sorted(frames)]):>34s}"
    )
    print("\n  per-cohort AUROC (median [min-max] over seeds):")
    for cohort in sorted(out["per_seed"][sorted(frames)[0]]["per_cohort"]):
        print(
            f"    {cohort:8s} "
            f"{summarise([out['per_seed'][s]['per_cohort'][cohort] for s in sorted(frames)]):>34s}"
        )
    print(
        "\n  per-seed pooled AUROC:",
        {s: round(out["per_seed"][s]["auroc"], 4) for s in sorted(frames)},
    )
    print(
        f"\n  {len(frames)}-seed OOF ENSEMBLE AUROC {ens_boot['auroc']:.4f} "
        f"(patient-bootstrap 95% CI {ens_boot['ci_low']:.4f}-{ens_boot['ci_high']:.4f})"
    )
    print(f"  PRIMARY uncertainty  patient-bootstrap SE {se_patient:.4f} "
          f"(median of the per-seed single-model bootstraps)")
    print(f"  ROBUSTNESS           seed SD {sd_seed:.4f}, observed range across seeds")
    print(f"  [not primary]        quadrature total {total:.4f} — reported for continuity only;"
          " no formal\n                       variance model justifies combining them")
    return out


# ── E1a decision gate ────────────────────────────────────────────────────────
# Operational definition of the design's "meaningfully above chance", written
# down because "at or near chance" cannot be adjudicated.
#
# HONESTY NOTE, and it matters: E0/E1a results were already visible when these
# thresholds were fixed, so this is a POST-HOC FORMALISATION, not a blind
# pre-registration. Its value is that it binds every arm not yet read — the
# remaining cap arms, LOCO-D after E2a, and any re-run — to one rule instead of
# to a judgement call made per table. Say so in the paper; do not describe the
# gate as prespecified.
#
# Thresholds are anchored to the design's own power arithmetic for set D
# (n = 1,129): minimum detectable AUROC 0.548, CI half-width +/-0.031.
GATE = {
    "pass_median_auroc": 0.60,      # comfortably clear of the 0.548 detection floor
    "pass_ci_low_floor": 0.55,      # D must separate from NEAR-chance, not just from 0.5
    "pass_max_drop_vs_A": 0.05,     # D must not be materially below A
    "fail_median_auroc": 0.55,      # at or under the detection floor
}


def evaluate_gate(per_seed: dict) -> dict:
    """PASS -> dependency-robust framing, proceed to E2a.
    FAIL -> fire E1b inside set D and pivot to a molecular-dependency paper.
    INDETERMINATE is treated as FAIL: the fail-safe branch is the cheap one."""
    seeds = sorted(per_seed)
    d = [per_seed[s]["D_mss_braf_wt"]["auroc"] for s in seeds]
    lo = [per_seed[s]["D_mss_braf_wt"]["ci_low"] for s in seeds]
    drop = [per_seed[s]["D_mss_braf_wt"]["delta_vs_A"] for s in seeds]
    median_d = float(np.median(d))
    checks = {
        "median_D_auroc": median_d,
        "median_D_ge_0.60": median_d >= GATE["pass_median_auroc"],
        "all_CI_low_gt_0.55": all(x > GATE["pass_ci_low_floor"] for x in lo),
        "any_CI_includes_0.50": any(x <= 0.50 for x in lo),
        "median_drop_vs_A": float(np.median(drop)),
        "drop_within_0.05": float(np.median(drop)) <= GATE["pass_max_drop_vs_A"],
    }
    if checks["median_D_ge_0.60"] and checks["all_CI_low_gt_0.55"] and checks["drop_within_0.05"]:
        verdict = "PASS"
    elif median_d < GATE["fail_median_auroc"] or checks["any_CI_includes_0.50"]:
        verdict = "FAIL"
    else:
        verdict = "INDETERMINATE (treat as FAIL)"
    return {"verdict": verdict, "thresholds": GATE, "checks": checks,
            "n_seeds": len(seeds), "per_seed_D_auroc": d, "per_seed_D_ci_low": lo}


def report_e1a(name: str) -> dict:
    """Within-repetition shared-resample bootstrap.

    Each repetition gets its OWN shared-resample bootstrap: patients are
    resampled once from set A and both AUROC(A_boot) and AUROC(C n A_boot) are
    recomputed on that same resample, so the nesting-induced correlation is
    carried correctly. Summarising CIs across repetitions afterwards keeps
    seed variance and patient-sampling variance visibly separate, which
    bootstrapping a seed-ensemble would silently merge.
    """
    root, label = CONDITIONS[name]
    frames = {s: f for s in SEEDS if (f := load_seed(root, s)) is not None}
    if not frames:
        return {}
    per_seed: dict[int, dict] = {}
    for seed, f in sorted(frames.items()):
        row = {}
        for key in e1a.SETS:
            mask = e1a.subset_mask(f, key)
            blk = evaluate.shared_resample_delta(f, mask)
            row[key] = {
                "n": int(mask.sum()),
                "n_mutant": int(f.loc[mask, "label"].sum()),
                "auroc": blk["auroc_subset"],
                "ci_low": blk["subset_ci_low"],
                "ci_high": blk["subset_ci_high"],
                "delta_vs_A": blk["delta"],
                "delta_ci_low": blk["delta_ci_low"],
                "delta_ci_high": blk["delta_ci_high"],
            }
        for cohort in sorted(f["cohort"].dropna().unique()):
            mask = f["cohort"].eq(cohort)
            blk = evaluate.shared_resample_delta(f, mask)
            row[f"cohort_{cohort}"] = {
                "n": int(mask.sum()),
                "n_mutant": int(f.loc[mask, "label"].sum()),
                "auroc": blk["auroc_subset"],
                "ci_low": blk["subset_ci_low"],
                "ci_high": blk["subset_ci_high"],
                "delta_vs_A": blk["delta"],
                "delta_ci_low": blk["delta_ci_low"],
                "delta_ci_high": blk["delta_ci_high"],
            }
        per_seed[seed] = row

    seeds = sorted(per_seed)
    keys = list(per_seed[seeds[0]])
    print(f"\n{'=' * 96}\nE1a . {name.upper()} . within-repetition shared-resample bootstrap"
          f" . seeds {seeds}\n{'=' * 96}")
    print(f"{'set':22s} {'n':>6s} {'AUROC median [min-max]':>26s} "
          f"{'median 95% CI':>18s} {'Delta(A-D) med of seedwise diffs':>32s}")
    for k in keys:
        r0 = per_seed[seeds[0]][k]
        au = [per_seed[s][k]["auroc"] for s in seeds]
        dl = [per_seed[s][k]["delta_vs_A"] for s in seeds]
        lo = float(np.median([per_seed[s][k]["ci_low"] for s in seeds]))
        hi = float(np.median([per_seed[s][k]["ci_high"] for s in seeds]))
        print(f"{k:22s} {r0['n']:6d} {summarise(au):>26s} {f'{lo:.3f}-{hi:.3f}':>18s} "
              f"{summarise(dl):>30s}")
    print("\n  SIGN: the column is Delta(A - set), the MEDIAN OF SEEDWISE DIFFERENCES.")
    print("        NEGATIVE = the set scored HIGHER than A (signal retained and then some);")
    print("        POSITIVE = the set scored LOWER (signal lost).")
    print("        It does NOT equal median(A) - median(set): a median of differences is not")
    print("        the difference of medians. Both are correct; they answer different questions.")
    print("  Per-cohort rows are COHORT-CONDITIONED, not held-out - transport is E2a's claim.")

    gate = evaluate_gate(per_seed)
    print(f"\n  DECISION GATE: {gate['verdict']}")
    for k, v in gate["checks"].items():
        print(f"    {k:24s} {v}")
    return {"condition": name, "per_seed": per_seed, "gate": gate}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--condition", choices=list(CONDITIONS), default=None)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--e1a", action="store_true")
    a = p.parse_args()
    results = {}
    names = list(CONDITIONS) if a.compare else [a.condition or "uncapped"]
    for n in names:
        results[n] = report_condition(n)
        if a.e1a:
            results[f"{n}_e1a"] = report_e1a(n)
    dest = paths.EVAL_ROOT / "e0_e1a_seed_report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # MERGE, never replace (see the same note in aim1_clinical_baseline.py).
    existing = json.loads(dest.read_text()) if dest.is_file() else {}
    existing.update({k: v for k, v in results.items() if v})
    dest.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
