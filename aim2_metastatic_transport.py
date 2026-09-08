#!/usr/bin/env python3
"""E2b - primary -> metastatic transport, within RIH and SurGen (Aim 2, 0 new fits).

QUESTION. Within RIH and SurGen, does the SAME source model that E2a scored on
primary tumours also work on metastases?

ZERO NEW FITS, AND THAT IS THE DESIGN. E2b reuses E2a's frozen full-source
refits, E2a's frozen three-seed logit ensemble and E2a's frozen SOURCE-ONLY
calibrator, unchanged. Nothing is retrained, refitted or reselected, so a
primary-to-metastatic difference cannot come from a different model - only from
the slides. The split between E2a and E2b is a split of the EVALUATION, not of
the training.

WHY THIS IS NOT PART OF E2a. E2a asks whether a cohort transports; E2b asks
whether a specimen role transports. Read off one number they are inseparable,
because RIH-M is both a new hospital and a new specimen role at once. Keeping
them apart is what lets Aim 2 say which of the two failed.

WHAT IS SCORED. RIH-M (85 patients, 37 mutant) and SurGen SR1482-M (74
patients, 30 mutant). TCGA's single metastasis is excluded (n = 1). CPTAC has
none. The source pools were primary-only in E2a, so no metastatic morphology
ever entered training.

THE TWO STRUCTURAL RULES, INHERITED FROM E2a AND NOT NEGOTIABLE HERE:

  RIH paired patients   8 patients contribute both a primary and a metastasis.
                        They are RETAINED in standalone RIH-P and RIH-M, and
                        EXCLUDED from the primary-vs-metastatic contrast, where
                        they would make the two arms non-independent.
  SurGen subcohort      SR1482 is the only SurGen subcohort with metastases, so
                        the SurGen contrast is SR1482-P vs SR1482-M. Pooling
                        SR386+SR1482 on the primary side would compare specimen
                        role against a different case mix.

CONCLUSION RULE, fixed before any result is read: claim metastatic transport
only if RIH-M and SR1482-M point estimates BOTH exceed 0.5 AND the equal-cohort
metastatic macro's 95% CI lower bound exceeds 0.5.

THE CONTRAST IS A DEPLOYMENT STRESS TEST, NOT A CAUSAL SPECIMEN-ROLE EFFECT.
The arms differ in patient and organ distribution as well as specimen role, so
a drop is evidence that deployment on metastases is harder - never evidence
that metastatic morphology per se carries less KRAS signal.

Usage:
    python aim2_metastatic_transport.py score  --cap 8192
    python aim2_metastatic_transport.py report --cap 8192
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

import aim2_loco_transport  # noqa: E402
from oceanpath.aim1 import lineage, paths, population  # noqa: E402
from oceanpath.eval.external import sigmoid  # noqa: E402

MET_TARGETS = aim2_loco_transport.MET_TARGETS  # ("RIH", "SurGen")

# Aim-2 contrasts sit close to the decision boundary and the legacy 2,000-draw
# intervals moved visibly across repeated Monte-Carlo runs.  Ten thousand is
# still cheap for these patient-level tables and is now the default for every
# E2b/E2d contrast.  The value and seed are written into every result artifact.
DEFAULT_N_BOOTSTRAP = 10_000


def _auprc(pat: pd.DataFrame) -> float:
    from sklearn.metrics import average_precision_score

    y = pat["label"].to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, pat["mean_logit"].to_numpy()))


def _validate_patient_arm(frame: pd.DataFrame, name: str) -> None:
    """Fail closed before a patient bootstrap is allowed to run."""
    required = {"label", "mean_logit", "prob_raw"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} is empty")
    labels = set(pd.unique(frame["label"]))
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError(f"{name} must contain both binary outcome classes; got {labels}")
    if "patient_id" in frame and frame["patient_id"].duplicated().any():
        dup = frame.loc[frame["patient_id"].duplicated(), "patient_id"].iloc[0]
        raise ValueError(f"{name} is not one-row-per-patient; duplicate {dup!r}")


def _resample_patient_arm(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Outcome-stratified patient resample.

    AUROC is conditional on the two outcome classes.  Resampling mutant and
    wild-type patients separately prevents invalid single-class draws and makes
    the Monte-Carlo error reproducible without changing the estimand.  Arms are
    always sampled independently unless a caller explicitly preserves a nested
    subset relationship (E2d-3).
    """
    chunks = []
    for label in (0, 1):
        arm = frame[frame["label"].eq(label)]
        chunks.append(arm.iloc[rng.integers(0, len(arm), len(arm))])
    return pd.concat(chunks, ignore_index=True)


def contrast_ci(
    p_side: pd.DataFrame,
    m_side: pd.DataFrame,
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap CI for metastatic MINUS primary, resampling the two arms
    INDEPENDENTLY.

    Independent and not paired, because after the dual-role patients are removed
    the arms share no patient — there is nothing to pair on. The CI is therefore
    wide by construction at n = 74-77 metastatic contrast patients, which is the honest
    width: a paired bootstrap here would understate it by pretending a
    correlation that the design deliberately removed.
    """
    _validate_patient_arm(p_side, "primary arm")
    _validate_patient_arm(m_side, "metastatic arm")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    d_auroc, d_auprc = [], []
    for _ in range(n_boot):
        pb = _resample_patient_arm(p_side, rng)
        mb = _resample_patient_arm(m_side, rng)
        d_auroc.append(aim2_loco_transport._auroc(mb) - aim2_loco_transport._auroc(pb))
        d_auprc.append(_auprc(mb) - _auprc(pb))
    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [
            float("nan")] * 2
    return {
        "delta_auroc": aim2_loco_transport._auroc(m_side) - aim2_loco_transport._auroc(p_side),
        "delta_auroc_ci": ci(d_auroc),
        "delta_auprc": _auprc(m_side) - _auprc(p_side),
        "delta_auprc_ci": ci(d_auprc),
        "n_bootstrap": len(d_auroc),
        "bootstrap_seed": int(seed),
        "bootstrap_method": (
            "independent outcome-stratified patient resampling of the disjoint "
            "primary and metastatic arms"
        ),
    }


def combined_decrement_ci(
    arms: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict:
    """Equal-cohort mean of metastatic-minus-primary AUROC contrasts.

    This is the direct overall decrement analysis.  It treats patients—not
    training seeds—as the sampling units, keeps each cohort on its own AUROC
    scale, and gives RIH and SurGen equal weight.  All four disjoint patient
    arms are resampled independently inside each bootstrap replicate.
    """
    if len(arms) < 2:
        raise ValueError("combined decrement requires at least two cohorts")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    for cohort, (primary, metastatic) in arms.items():
        _validate_patient_arm(primary, f"{cohort} primary arm")
        _validate_patient_arm(metastatic, f"{cohort} metastatic arm")

    point_by_cohort = {
        cohort: float(aim2_loco_transport._auroc(metastatic) - aim2_loco_transport._auroc(primary))
        for cohort, (primary, metastatic) in arms.items()
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        deltas = []
        for primary, metastatic in arms.values():
            pb = _resample_patient_arm(primary, rng)
            mb = _resample_patient_arm(metastatic, rng)
            deltas.append(aim2_loco_transport._auroc(mb) - aim2_loco_transport._auroc(pb))
        draws[index] = float(np.mean(deltas))

    point = float(np.mean(list(point_by_cohort.values())))
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    if ci[1] < 0:
        inference = "overall decrement supported"
    elif ci[0] > 0:
        inference = "overall improvement supported"
    else:
        inference = "overall decrement not established"
    return {
        "estimand": "equal-cohort mean of within-cohort AUROC(M)-AUROC(P)",
        "delta_auroc": point,
        "delta_auroc_ci": ci,
        "per_cohort_delta_auroc": point_by_cohort,
        "n_cohorts": int(len(arms)),
        "n_bootstrap": int(n_boot),
        "bootstrap_seed": int(seed),
        "bootstrap_fraction_below_zero": float(np.mean(draws < 0)),
        "bootstrap_method": (
            "independent outcome-stratified patient resampling of every disjoint "
            "cohort-by-role arm; equal weight across cohorts; inference is "
            "conditional on the frozen fitted ensemble"
        ),
        "ci_excludes_zero": bool(ci[1] < 0 or ci[0] > 0),
        "evidence_of_overall_decrement": bool(ci[1] < 0),
        "inference": inference,
    }


def classify(report: dict) -> dict:
    """Summarize E2b without treating point-sign or training seeds as inference.

    The original numbered-outcome classifier used a post-result dead zone and
    promoted two negative point estimates plus six correlated seed fits into a
    decrement claim.  Here point directions are descriptive; the direct
    equal-cohort patient bootstrap is the inferential quantity.
    """
    per: dict = {}
    for target, block in report["targets"].items():
        contrast = block.get("primary_vs_metastatic")
        metastatic = block.get("metastatic_contrast_block") or block.get(
            "metastatic_overall"
        )
        if not contrast or not metastatic:
            continue
        delta = float(contrast["delta_auroc"])
        lo, hi = map(float, contrast["delta_auroc_ci"])
        p_brier = (block.get("primary_contrast_block") or {}).get(
            "brier", float("nan")
        )
        entry = {
            "primary_auroc": float(contrast["primary_auroc"]),
            "metastatic_auroc": float(contrast["metastatic_auroc"]),
            "delta": delta,
            "delta_ci": [lo, hi],
            "point_direction": "down" if delta < 0 else "up" if delta > 0 else "zero",
            "ci_excludes_zero": bool(hi < 0 or lo > 0),
            "brier_primary": p_brier,
            "brier_metastatic": metastatic.get("brier", float("nan")),
            "brier_delta": metastatic.get("brier", float("nan")) - p_brier,
        }
        calibrated = block.get("calibrated_contrast")
        if calibrated and "brier_delta" in calibrated:
            entry.update({
                "brier_delta_raw": entry["brier_delta"],
                "brier_primary": calibrated["brier_primary"],
                "brier_metastatic": calibrated["brier_metastatic"],
                "brier_delta": calibrated["brier_delta"],
                "brier_scale": "source-calibrated",
                "log_loss_delta": calibrated["log_loss_delta"],
            })
        else:
            entry["brier_scale"] = "raw (no calibrator available)"
        organ = block.get("metastatic_by_organ") or {}
        if "liver" in organ and "non_liver" in organ:
            entry["organ_gap_liver_minus_non_liver"] = float(
                organ["liver"].get("auroc", float("nan"))
                - organ["non_liver"].get("auroc", float("nan"))
            )
        per[target] = entry

    combined = report.get("combined_decrement")
    if len(per) < 2 or not combined:
        return {
            "per_target": per,
            "outcome": "incomplete — need both target cohorts and combined decrement",
        }

    directions = {target: value["point_direction"] for target, value in per.items()}
    pattern = ", ".join(f"{target}: {direction}" for target, direction in directions.items())
    return {
        "per_target": per,
        "outcome": combined["inference"],
        "descriptive_point_pattern": pattern,
        "combined_decrement": combined,
        "any_single_cohort_ci_excludes_zero": bool(
            any(value["ci_excludes_zero"] for value in per.values())
        ),
        "seed_results_inferential_role": "none — computational stability diagnostic only",
        "organ_results_inferential_role": (
            "exploratory here; formal within-cohort contrasts belong to E2d-1"
        ),
        "rule": (
            "A decrement is supported only when the 95% patient-bootstrap CI for "
            "the equal-cohort mean of within-cohort AUROC(M)-AUROC(P) lies below "
            "zero. Point-sign concordance and training-seed agreement are "
            "descriptive and do not carry sampling inference. This decrement "
            "analysis is separate from the fixed metastatic-transport rule."
        ),
    }


def cmd_score(args: argparse.Namespace) -> None:
    missing: list[str] = []
    for t in MET_TARGETS:
        for s in aim2_loco_transport.SEEDS:
            try:
                aim2_loco_transport._completed_refit(t, s, args.cap)
            except Exception as exc:
                missing.append(f"{t}/seed{s}: {exc}")
    if missing:
        raise RuntimeError(
            "E2b requires all six completed E2a refits before scoring; "
            + " | ".join(missing)
        )
    for t in MET_TARGETS:
        for s in aim2_loco_transport.SEEDS:
            n = len(aim2_loco_transport.score_one(t, s, "metastatic", args.cap))
            print(f"  cap{args.cap} {t} seed{s} metastatic : {n} slide scores")


def cmd_report(args: argparse.Namespace) -> None:
    from oceanpath.eval.core import compute_calibration_intercept_slope

    final_dest = lineage.eval_root() / f"e2b_metastatic_cap{args.cap}.json"
    lineage.ensure_absent(final_dest)
    for target in MET_TARGETS:
        calibrator = aim2_loco_transport.calibrator_path(target, args.cap)
        if not calibrator.is_file():
            raise FileNotFoundError(
                f"Missing E2a source calibrator for {target}: {calibrator}"
            )
        lineage.ensure_absent(aim2_loco_transport.calibrated_path(target, "metastatic", args.cap))
        lineage.ensure_absent(
            aim2_loco_transport.calibrated_receipt_path(target, "metastatic", args.cap)
        )
        for kind in ("primary", "metastatic"):
            for seed in aim2_loco_transport.SEEDS:
                if aim2_loco_transport._load_valid_score_cache(
                    target, seed, kind, args.cap
                ) is None:
                    raise FileNotFoundError(
                        f"Missing {kind} score cache for {target}/seed{seed}; "
                        "run E2a/E2b score commands before E2b report"
                    )

    primary, met = aim2_loco_transport.load_primary(), aim2_loco_transport.load_metastatic()
    dual = population.dual_specimen_patients(primary, met)
    report: dict = {
        "cap": args.cap,
        "lineage": lineage.lineage_name(),
        "source": "E2a immutable-lineage refits — 0 new fits",
        "targets": {},
        "conclusion": {},
        "inference": {
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "sampling_unit": "patient",
            "training_seed_role": "computational stability only; never an inferential replicate",
            "conditioning": "confidence intervals condition on the frozen fitted ensemble",
        },
    }
    met_frames: dict[str, pd.DataFrame] = {}
    contrast_arms: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for t in MET_TARGETS:
        ens_m, per_seed_m = aim2_loco_transport.seed_ensemble(t, "metastatic", args.cap)
        ens_p, per_seed_p = aim2_loco_transport.seed_ensemble(t, "primary", args.cap)
        if ens_m.empty or ens_p.empty:
            raise RuntimeError(f"{t}: primary or metastatic ensemble is empty")
        expected_seeds = set(aim2_loco_transport.SEEDS)
        if set(per_seed_m) != expected_seeds or set(per_seed_p) != expected_seeds:
            raise RuntimeError(
                f"{t}: incomplete seed ensemble; primary={sorted(per_seed_p)}, "
                f"metastatic={sorted(per_seed_m)}, expected={sorted(expected_seeds)}"
            )
        block: dict = {
            "seeds_complete": sorted(per_seed_m),
            "input_artifacts": {
                "primary_score_receipts": {
                    str(seed): lineage.artifact_identity(
                        aim2_loco_transport.score_receipt_path(t, seed, "primary", args.cap)
                    )
                    for seed in aim2_loco_transport.SEEDS
                },
                "metastatic_score_receipts": {
                    str(seed): lineage.artifact_identity(
                        aim2_loco_transport.score_receipt_path(t, seed, "metastatic", args.cap)
                    )
                    for seed in aim2_loco_transport.SEEDS
                },
                "primary_manifest": lineage.artifact_identity(
                    aim2_loco_transport.target_manifest(t, "primary")
                ),
                "metastatic_manifest": lineage.artifact_identity(
                    aim2_loco_transport.target_manifest(t, "metastatic")
                ),
            },
        }
        block["metastatic_overall"] = aim2_loco_transport.block_metrics(
            ens_m,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )
        block["metastatic_overall_per_seed_auroc"] = {
            s: aim2_loco_transport._auroc(f) for s, f in per_seed_m.items()
        }
        block["metastatic_overall_per_seed_auprc"] = {
            s: _auprc(f) for s, f in per_seed_m.items()
        }
        # The three-seed logit ensemble is the headline; the per-seed spread says
        # whether the ensemble is doing work or the seeds already agree.

        dm = ens_m[ens_m["msi_dmmr"].eq("MSS/pMMR") & ens_m["braf"].eq("wild_type")]
        if len(dm) > 20:
            block["metastatic_D_subset"] = aim2_loco_transport.block_metrics(
                dm,
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.bootstrap_seed,
            )
        met_frames[t] = ens_m

        # ── E2a's frozen SOURCE-ONLY calibrator, applied unchanged ──────────
        # Refitting anything here would make a metastatic calibration failure
        # indistinguishable from a refit artefact. The calibrator saw no target
        # label of either specimen role.
        cal_file = aim2_loco_transport.calibrator_path(t, args.cap)
        if cal_file.is_file():
            block["input_artifacts"]["source_calibrator"] = (
                lineage.artifact_identity(cal_file)
            )
            info = json.loads(cal_file.read_text())
            eta = ens_m["mean_logit"].to_numpy()
            p_cal = sigmoid(info["a"] + info["b"] * eta)
            y = ens_m["label"].to_numpy()
            from sklearn.metrics import brier_score_loss, log_loss

            before = compute_calibration_intercept_slope(
                y, np.clip(ens_m["prob_raw"].to_numpy(), 1e-6, 1 - 1e-6))
            after = compute_calibration_intercept_slope(y, np.clip(p_cal, 1e-6, 1 - 1e-6))
            block["source_calibrated"] = {
                "calibration_intercept_raw": float(before["calibration_intercept"]),
                "calibration_slope_raw": float(before["calibration_slope"]),
                "calibration_intercept": float(after["calibration_intercept"]),
                "calibration_slope": float(after["calibration_slope"]),
                "log_loss_raw": float(log_loss(y, np.clip(
                    ens_m["prob_raw"].to_numpy(), 1e-6, 1 - 1e-6))),
                "log_loss_source_calibrated": float(log_loss(y, np.clip(p_cal, 1e-6, 1 - 1e-6))),
                "brier_source_calibrated": float(brier_score_loss(y, np.clip(
                    p_cal, 1e-6, 1 - 1e-6))),
                "note": "E2a's calibrator, reused verbatim; no metastatic label was fitted",
            }
            # The same map on the PRIMARY side of the contrast, so the Brier /
            # log-loss comparison can be read on the calibrated scale too. The
            # raw-scale comparison is confounded: Table 2.1 shows the raw scale
            # is badly miscalibrated, and each cohort's miscalibration differs.
            block["calibrated_contrast"] = {
                "definition": "E2a source calibrator applied to BOTH arms",
            }
            out = ens_m.assign(prob_source_calibrated=p_cal)
            dest = aim2_loco_transport.calibrated_path(t, "metastatic", args.cap)
            lineage.write_parquet_once(dest, out)
            score_receipts = [
                lineage.artifact_identity(
                    aim2_loco_transport.score_receipt_path(t, seed, "metastatic", args.cap)
                )
                for seed in aim2_loco_transport.SEEDS
            ]
            receipt = aim2_loco_transport.calibrated_receipt_path(t, "metastatic", args.cap)
            lineage.write_json_once(
                receipt,
                {
                    "schema_version": 2,
                    "lineage": lineage.lineage_name(),
                    "calibrator": {"a": info["a"], "b": info["b"]},
                    "calibrator_artifact": lineage.artifact_identity(cal_file),
                    "score_receipts": score_receipts,
                    "artifact": lineage.artifact_identity(dest),
                },
            )
            block["input_artifacts"]["metastatic_calibrated"] = (
                lineage.artifact_identity(receipt)
            )

        # ── the primary side, for the headline table ────────────────────────
        block["primary_overall"] = aim2_loco_transport.block_metrics(
            ens_p,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )
        block["primary_brier"] = block["primary_overall"].get("brier", float("nan"))

        # ── metastatic organ split — outcome 10 (liver vs non-liver) ────────
        # Descriptive only: 50/35 and 35/39 patients, so neither arm can carry a
        # claim on its own. It is here because "liver good, non-liver bad" is one
        # of the pre-listed outcomes and cannot be ruled in or out without it.
        man = pd.read_csv(aim2_loco_transport.target_manifest(t, "metastatic"))
        if "liver_class" in man.columns:
            organ = man.drop_duplicates("patient_id").set_index("patient_id")["liver_class"]
            tagged = ens_m.assign(liver_class=ens_m["patient_id"].map(organ))
            block["metastatic_by_organ"] = {}
            for name, grp in tagged.groupby("liver_class"):
                if len(grp) >= 20 and grp["label"].nunique() > 1:
                    block["metastatic_by_organ"][str(name)] = aim2_loco_transport.block_metrics(
                        grp,
                        n_bootstrap=args.n_bootstrap,
                        bootstrap_seed=args.bootstrap_seed,
                    )

        # ── primary-vs-metastatic, DEPLOYMENT STRESS TEST — not causal ──────
        if t == "SurGen":
            p_side = ens_p[ens_p["subcohort"].eq(aim2_loco_transport.SURGEN_MET_SUBCOHORT)]
            m_side = ens_m
            note = f"{aim2_loco_transport.SURGEN_MET_SUBCOHORT}-P vs {aim2_loco_transport.SURGEN_MET_SUBCOHORT}-M"
        else:
            dual_in_cohort = set(ens_p["patient_id"]).intersection(
                ens_m["patient_id"], dual
            )
            n_dual = len(dual_in_cohort)
            p_side = ens_p[~ens_p["patient_id"].isin(dual)]
            m_side = ens_m[~ens_m["patient_id"].isin(dual)]
            note = (
                f"RIH-P vs RIH-M after excluding {n_dual} dual-role patients "
                "from both contrast arms"
            )
        # The CI must come from the SAME primary population the contrast uses —
        # SR1482-P/M for SurGen, RIH-P/M-minus-dual for RIH — not from the
        # standalone all-metastasis block.
        # Printing a matched point estimate beside an unmatched CI put 0.6271
        # outside its own interval [0.651, 0.727] in the first run of this report.
        p_block = aim2_loco_transport.block_metrics(
            p_side,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )
        m_block = aim2_loco_transport.block_metrics(
            m_side,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )
        block["primary_contrast_block"] = p_block
        block["metastatic_contrast_block"] = m_block
        if "calibrated_contrast" in block:
            from sklearn.metrics import brier_score_loss, log_loss

            cinfo = json.loads(aim2_loco_transport.calibrator_path(t, args.cap).read_text())

            def _cal(frame, ci=cinfo):
                q = np.clip(sigmoid(ci["a"] + ci["b"] * frame["mean_logit"].to_numpy()),
                            1e-6, 1 - 1e-6)
                yy = frame["label"].to_numpy()
                return float(brier_score_loss(yy, q)), float(log_loss(yy, q))

            bp, lp = _cal(p_side)
            bm, lm = _cal(m_side)
            block["calibrated_contrast"].update({
                "brier_primary": bp, "brier_metastatic": bm, "brier_delta": bm - bp,
                "log_loss_primary": lp, "log_loss_metastatic": lm,
                "log_loss_delta": lm - lp,
            })
        # Per-seed on the SAME p_side the contrast uses, or the per-seed spread
        # would describe a different population from the ensemble beside it.
        keep = set(p_side["patient_id"])
        seed_p = {sd: f[f["patient_id"].isin(keep)] for sd, f in per_seed_p.items()}
        keep_m = set(m_side["patient_id"])
        seed_m = {sd: f[f["patient_id"].isin(keep_m)] for sd, f in per_seed_m.items()}
        block["primary_per_seed_auroc"] = {sd: aim2_loco_transport._auroc(f) for sd, f in seed_p.items()}
        block["primary_per_seed_auprc"] = {sd: _auprc(f) for sd, f in seed_p.items()}
        block["metastatic_per_seed_auroc"] = {
            sd: aim2_loco_transport._auroc(f) for sd, f in seed_m.items()
        }
        block["metastatic_per_seed_auprc"] = {
            sd: _auprc(f) for sd, f in seed_m.items()
        }
        block["primary_vs_metastatic"] = {
            "definition": note,
            "primary_auroc": aim2_loco_transport._auroc(p_side), "primary_n": int(len(p_side)),
            "primary_auroc_ci": p_block.get("auroc_ci", [float("nan")] * 2),
            "primary_auprc": _auprc(p_side),
            "metastatic_auroc": aim2_loco_transport._auroc(m_side), "metastatic_n": int(len(m_side)),
            "metastatic_auroc_ci": m_block.get(
                "auroc_ci", [float("nan")] * 2),
            "metastatic_auprc": _auprc(m_side),
            **contrast_ci(
                p_side,
                m_side,
                n_boot=args.n_bootstrap,
                seed=args.bootstrap_seed,
            ),
        }
        contrast_arms[t] = (p_side, m_side)
        # The unmatched SurGen contrast the request asked for, reported next to the
        # matched one so the case-mix cost of pooling SR386+SR1482 is visible.
        if t == "SurGen":
            block["primary_vs_metastatic_unmatched"] = {
                "definition": "all SurGen-P (SR386+SR1482) vs SurGen-M",
                "primary_auroc": aim2_loco_transport._auroc(ens_p), "primary_n": int(len(ens_p)),
                "metastatic_auroc": aim2_loco_transport._auroc(ens_m), "metastatic_n": int(len(ens_m)),
                "delta_auroc": aim2_loco_transport._auroc(ens_m) - aim2_loco_transport._auroc(ens_p),
            }
        report["targets"][t] = block

    # Direct decrement inference is separate from the fixed transport rule.
    # The latter asks whether metastatic discrimination survives above chance;
    # the former asks whether it is demonstrably lower than primary tissue.
    if len(contrast_arms) == len(MET_TARGETS):
        report["combined_decrement"] = combined_decrement_ci(
            contrast_arms,
            n_boot=args.n_bootstrap,
            seed=args.bootstrap_seed,
        )

    # ── metastatic macro + the sufficient-conclusion rule ──────────────────
    if len(met_frames) == 2:
        rng = np.random.default_rng(args.bootstrap_seed)
        points = [aim2_loco_transport._auroc(f) for f in met_frames.values()]
        draws = []
        for _ in range(args.n_bootstrap):
            vals = []
            for f in met_frames.values():
                blk = _resample_patient_arm(f, rng)
                vals.append(aim2_loco_transport._auroc(blk))
            if vals:
                draws.append(float(np.mean(vals)))
        lo = float(np.percentile(draws, 2.5)) if draws else float("nan")
        hi = float(np.percentile(draws, 97.5)) if draws else float("nan")
        rih = report["targets"]["RIH"]["metastatic_overall"]["auroc"]
        sr = report["targets"]["SurGen"]["metastatic_overall"]["auroc"]
        report["conclusion"] = {
            "metastatic_macro_auroc": float(np.mean(points)),
            "metastatic_macro_ci": [lo, hi],
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "bootstrap_method": (
                "outcome-stratified patient bootstrap within each metastatic "
                "cohort; equal cohort weights"
            ),
            "RIH_M_auroc": rih,
            "SR1482_M_auroc": sr,
            "both_point_estimates_above_0.5": bool(rih > 0.5 and sr > 0.5),
            "macro_ci_lower_above_0.5": bool(lo > 0.5),
            "claim_metastatic_transport": bool(rih > 0.5 and sr > 0.5 and lo > 0.5),
            "caveat": (
                "Deployment stress test only. Do NOT claim primary-metastatic "
                "equivalence or a causal specimen-role effect: the arms differ in "
                "patient and organ distribution as well as specimen role."
            ),
        }

    report["classification"] = classify(report)

    dest = final_dest
    lineage.write_json_once(dest, report)
    print_report(report)
    print(f"\nWrote {dest}")


def print_report(report: dict) -> None:
    def line(label, b, indent=4):
        if not b or b.get("degenerate"):
            return
        ci = b.get("auroc_ci", [float("nan")] * 2)
        print(f"{' ' * indent}{label:26s} n={b['n']:5d} prev={b['prevalence']:5.1%}  "
              f"AUROC {b['auroc']:.4f} [{ci[0]:.3f}-{ci[1]:.3f}]  AUPRC {b['auprc']:.4f}  "
              f"Brier {b['brier']:.4f}  logloss {b['log_loss']:.4f}  "
              f"cal {b['calibration_intercept']:+.3f}/{b['calibration_slope']:.3f}")

    for t, b in report["targets"].items():
        print(f"\n{'=' * 118}\nE2b · PRIMARY -> METASTATIC · {t} · seeds "
              f"{b['seeds_complete']}\n{'=' * 118}")
        line("METASTATIC (all)", b.get("metastatic_overall"))
        line("METASTATIC · MSS+BRAF-WT", b.get("metastatic_D_subset"))
        for arm in ("primary", "metastatic"):
            ps = b.get(f"{arm}_per_seed_auroc", {})
            if not ps:
                continue
            vals = [v for v in ps.values() if np.isfinite(v)]
            ens = (b["primary_contrast_block"] if arm == "primary"
                   else b["metastatic_contrast_block"])["auroc"]
            print(f"    {arm:10s} per-seed AUROC "
                  f"{{{', '.join(f'{k}: {v:.4f}' for k, v in ps.items())}}}"
                  f"   mean {np.mean(vals):.4f}   range {max(vals) - min(vals):.4f}"
                  f"   ENSEMBLE {ens:.4f}   gain {ens - np.mean(vals):+.4f}")
        sc = b.get("source_calibrated")
        if sc:
            print(f"    E2a source calibrator, reused: log loss {sc['log_loss_raw']:.4f} -> "
                  f"{sc['log_loss_source_calibrated']:.4f}   cal "
                  f"{sc['calibration_intercept_raw']:+.3f}/{sc['calibration_slope_raw']:.3f}"
                  f" -> {sc['calibration_intercept']:+.3f}/{sc['calibration_slope']:.3f}")
        for name, blk in b.get("metastatic_by_organ", {}).items():
            line(f"  METASTATIC · {name}", blk, indent=6)
        c = b["primary_vs_metastatic"]
        print(f"    primary→metastatic  [{c['definition']}]  "
              f"P {c['primary_auroc']:.4f} (n={c['primary_n']})  →  "
              f"M {c['metastatic_auroc']:.4f} (n={c['metastatic_n']})")
        print(f"      dAUROC {c['delta_auroc']:+.4f} "
              f"95% CI [{c['delta_auroc_ci'][0]:+.4f}, {c['delta_auroc_ci'][1]:+.4f}]   "
              f"dAUPRC {c['delta_auprc']:+.4f} "
              f"95% CI [{c['delta_auprc_ci'][0]:+.4f}, {c['delta_auprc_ci'][1]:+.4f}]")
        u = b.get("primary_vs_metastatic_unmatched")
        if u:
            print(f"      unmatched [{u['definition']}]  P {u['primary_auroc']:.4f} "
                  f"(n={u['primary_n']})  →  M {u['metastatic_auroc']:.4f}  "
                  f"dAUROC {u['delta_auroc']:+.4f}   (case mix differs — the matched "
                  "row above is the one to read)")

    k = report.get("classification", {})
    if k.get("per_target"):
        print(f"\n{'=' * 118}\nE2b HEADLINE — source primary → unseen target primary → "
              f"unseen target metastatic\n{'=' * 118}")
        print(f"  {'Target':8s} {'Primary AUROC':>22s} {'Metastatic AUROC':>24s} "
              f"{'d Meta-Primary':>28s}")
        for t, v in k["per_target"].items():
            b = report["targets"][t]
            c = b["primary_vs_metastatic"]
            pci = c["primary_auroc_ci"]
            mci = c["metastatic_auroc_ci"]
            print(f"  {t:8s} {c['primary_auroc']:8.4f} [{pci[0]:.3f}-{pci[1]:.3f}] "
                  f"{c['metastatic_auroc']:10.4f} [{mci[0]:.3f}-{mci[1]:.3f}] "
                  f"{v['delta']:+14.4f} [{v['delta_ci'][0]:+.3f}, {v['delta_ci'][1]:+.3f}]")
        print(f"  {'':8s} {'AUPRC':>22s}")
        for t in k["per_target"]:
            c = report["targets"][t]["primary_vs_metastatic"]
            print(f"  {t:8s} P {c['primary_auprc']:.4f}   M {c['metastatic_auprc']:.4f}   "
                  f"dAUPRC {c['delta_auprc']:+.4f} "
                  f"[{c['delta_auprc_ci'][0]:+.3f}, {c['delta_auprc_ci'][1]:+.3f}]")
        print(f"\n  DECREMENT INFERENCE: {k['outcome'].upper()}")
        print(f"  descriptive point pattern: {k['descriptive_point_pattern']}")
        print("  training seeds: computational stability diagnostic only; "
              "not independent replicates")
        print(f"  any single-cohort CI excludes zero: "
              f"{k['any_single_cohort_ci_excludes_zero']}")
        for t, v in k["per_target"].items():
            print(f"    {t:8s} Brier P {v['brier_primary']:.4f} -> M "
                  f"{v['brier_metastatic']:.4f}  ({v['brier_delta']:+.4f})"
                  + (f"   liver-vs-non-liver gap "
                     f"{v['organ_gap_liver_minus_non_liver']:+.3f}"
                     if "organ_gap_liver_minus_non_liver" in v else ""))
        print(f"  {k['rule']}")

        combined = k["combined_decrement"]
        ci = combined["delta_auroc_ci"]
        print(f"\n  EQUAL-COHORT COMBINED ΔAUROC(M−P) "
              f"{combined['delta_auroc']:+.4f} "
              f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print(f"  -> {combined['inference'].upper()} "
              f"({combined['n_bootstrap']:,} patient-bootstrap draws)")

    c = report.get("conclusion", {})
    if c:
        print(f"\n{'=' * 118}\nSUFFICIENT-CONCLUSION RULE (metastatic transport)\n{'=' * 118}")
        print(f"  RIH-M AUROC      {c['RIH_M_auroc']:.4f}   > 0.5 ? {c['RIH_M_auroc'] > 0.5}")
        print(f"  SR1482-M AUROC   {c['SR1482_M_auroc']:.4f}   > 0.5 ? {c['SR1482_M_auroc'] > 0.5}")
        print(f"  metastatic macro {c['metastatic_macro_auroc']:.4f} "
              f"95% CI [{c['metastatic_macro_ci'][0]:.4f}, {c['metastatic_macro_ci'][1]:.4f}]"
              f"   lower > 0.5 ? {c['macro_ci_lower_above_0.5']}")
        verdict = "CLAIM SUPPORTED" if c["claim_metastatic_transport"] else "NOT SUPPORTED"
        print(f"\n  -> metastatic transport: {verdict}")
        print(f"  {c['caveat']}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("score", help="score RIH-M and SR1482-M with E2a's frozen refits")
    x.add_argument("--cap", type=int, required=True, choices=aim2_loco_transport.E2A_CAPS)
    x.set_defaults(func=cmd_score)
    x = sub.add_parser("report")
    x.add_argument("--cap", type=int, required=True, choices=aim2_loco_transport.E2A_CAPS)
    x.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    x.add_argument("--bootstrap-seed", type=int, default=paths.BOOTSTRAP_SEED)
    x.set_defaults(func=cmd_report)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
