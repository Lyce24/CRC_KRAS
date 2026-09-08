#!/usr/bin/env python3
"""E2a - zero-shot leave-one-cohort-out transport, PRIMARY targets only (Aim 2).

AIM 2. Determine how a primary-trained KRAS histomorphology model transports
across external cohorts and metastatic tissue, identify whether transport
failure reflects loss of discrimination or miscalibration, and determine how
many - and what type of - local labelled cases are required for reliable
metastatic deployment. E2a is the first of four experiments:

    E2a  zero-shot LOCO           does KRAS ranking/calibration transport to an
                                  unseen cohort?                     <- this file
    E2b  primary -> metastatic    within RIH and SurGen, does the same source
                                  model work on metastases?   aim2_metastatic_transport.py
    E2c  few-shot adaptation      can sparse local labels repair target
                                  metastatic performance?      aim2_head_adaptation_base.py
    E2d  robustness / concordance is the result robust to RIH acquisition batch,
                                  metastatic organ, paired specimens, SurGen
                                  subcohort?              aim2_metastatic_site.py--aim2_paired_specimen_concordance.py

The metastatic-site, peritoneal stability, dependency-restriction, SurGen
subcohort, paired-specimen, and RIH acquisition/processing-regime panels are
implemented.  The latter is explicitly a composite technical-regime
sensitivity—not an identified scanner effect—because repair, resolution,
platform and acquisition era cannot be separated.

THE QUESTION E2a ASKS. If the model has never seen this cohort, can it still
predict KRAS? Nothing more. The answer is reported as discrimination (AUROC,
AUPRC) and calibration (intercept, slope, Brier, log loss) separately, because
the whole point of Aim 2's second clause is that these two can fail apart.

TARGET = PRIMARY TUMOURS ONLY. Metastatic slides are NOT scored here. They are
scored in E2b, by these same frozen models, so that "does this cohort transport"
and "does this specimen role transport" are never read off one number. This
file still BUILDS the metastatic target manifests (E2b consumes them); it never
scores or reports them.

WHAT IS TRANSPORTED, AND THE 72-FIT ACCOUNTING. Per (target, seed):

    5 source-CV fits -> source OOF predictions -> 1 full-source refit -> target
                                                                        prediction

4 targets x 3 seeds x 6 fits = 72. The transported object is the full-source
refit - trained on ALL primary patients of the three non-held-out cohorts, for
a frozen optimizer-step budget, with no validation set and therefore no early
stopping and no model selection. The 5 CV fits exist for ONE reason: a
source-only calibrator needs held-out source predictions, and a refit cannot
produce them. No target label is ever seen by either.

    Target    Train on
    TCGA      SurGen-P + RIH-P + CPTAC-P
    SurGen    TCGA-P + RIH-P + CPTAC-P
    RIH       TCGA-P + SurGen-P + CPTAC-P
    CPTAC     TCGA-P + SurGen-P + RIH-P

SIZE-MATCHED RIH SENSITIVITY (+18 fits), CONDITIONAL. Source pools span
749-1,392 patients, so a comparison of transport MAGNITUDE across targets is
confounded with source size. That comparison is not required to answer E2a's
question, so the arm is not run by default. It fires only if the manuscript
interprets cross-target magnitude differences: RIH's source pool is subsampled
to 749 patients - the smallest, the SurGen-held-out pool - and the same 6-fit
chain is rerun for 3 seeds. `--size-matched` on manifests/train/source-cv.

THE STEP BUDGET IS MATCHED, NOT THE EPOCH COUNT. Source pools span 761-1,544
slides, and at batch_size=1 one optimizer step is one slide - so a fixed epoch
count would hand CPTAC's model twice the gradient updates SurGen's model gets,
and "which cohort transports" would be confounded with "which model trained
longer". Epochs are therefore derived per target as STEP_BUDGET / n_patients.

MODEL. UNIv1 + ABMIL, cap 8,192 tiles/epoch at training, full bags at
inference. Cap 4,096 is retained as the pre-specified sensitivity arm.

WHY NOT REUSE THE E0 MODELS. E0 trained on all four cohorts, so every E0
prediction comes from a model already exposed to the target cohort's
distribution. Any E0-to-target number is supplementary only.

SOURCE IS PRIMARY-ONLY. Including source metastases would put metastatic
morphology into training and turn E2b's specimen-role contrast into a pure
institutional-shift measurement.

Usage:
    python aim2_loco_transport.py plan
    python aim2_loco_transport.py manifests --apply
    python aim2_loco_transport.py train      --cap 8192
    python aim2_loco_transport.py source-cv  --cap 8192
    python aim2_loco_transport.py calibrate  --cap 8192
    python aim2_loco_transport.py score      --cap 8192
    python aim2_loco_transport.py report     --cap 8192
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import (  # noqa: E402
    balance,
    evaluate,
    lineage,
    paths,
    population,
    scoring,
    technical,
)
from oceanpath.eval.external import sigmoid  # noqa: E402

TARGETS: tuple[str, ...] = ("CPTAC", "RIH", "SurGen", "TCGA")
MET_TARGETS: tuple[str, ...] = ("RIH", "SurGen")
SEEDS: tuple[int, ...] = (42, 43, 44)

E2A_CAPS = (4096, 8192)   # both caps, matching the E0 patient-balanced ladder
# PATIENT-BALANCED. One dataset item is a PATIENT: patients drawn uniformly, one
# eligible slide re-drawn per patient per epoch, ordinary UNWEIGHTED BCE. This
# matters MORE here than in E0 because the slide-uniform distortion differed BY
# HELD-OUT TARGET — SR1482 was ~25-30% overweighted when RIH, TCGA or CPTAC was
# held out, and entirely absent when SurGen was. Under the old loader each LOCO
# direction optimised a different estimand, so an apparent RIH-vs-SurGen
# transport difference could have been sampling imbalance, not target difficulty.
#
# One optimizer step is now one PATIENT-VISIT, so the budget is counted in
# patient-visits and epochs derive from the source PATIENT count, not slides.
STEP_BUDGET = 6060        # Aim-1 patient-balanced: 1010 patients x median best epoch 6
MIN_EPOCHS = 3
DEFAULT_N_BOOTSTRAP = 10_000

# ── Source cross-validation, for the source-only calibrator ─────────────────
# The transported object is a full-source refit, but a SOURCE-ONLY calibrator
# needs held-out source predictions, which a refit cannot produce. So each LOCO
# direction also gets an ordinary 5-fold CV over its own source pool, purely to
# emit source OOF logits. No target label is ever seen.
#
# STATED MISMATCH: the CV models train on 4/5 of the source while the refit
# trains on 5/5, so the calibrator is fitted on a slightly different score scale
# from the one it is applied to. Reported, not hidden — it is the reason the
# original E2a design carried two operating rules.
#
# CAP: the source-CV runs at the STUDY cap, 8,192, so the calibrator and the
# scores it is applied to come from the same bag policy. The earlier 54/60 fits
# at 4,096 predate the study-cap decision and are demoted to the sensitivity arm.
SRC_CV_CAP = 8192

# ── Size-matched RIH sensitivity arm (+18 fits), CONDITIONAL ────────────────
# Source pools span 749-1,392 patients. That is irrelevant to "does this cohort
# transport at all", and load-bearing the moment transport MAGNITUDE is compared
# ACROSS targets. So the arm is pre-specified and not run: RIH's source pool is
# subsampled to the smallest pool's size before the same 6-fit chain is repeated.
# 3 seeds x (5 source-CV + 1 refit) = 18 fits. Subsampling is patient-level and
# stratified on cohort x KRAS so the source class balance and cohort mix survive.
SIZE_MATCHED_TARGET = "RIH"
SIZE_MATCH_N = 749  # patients in the smallest source pool (SurGen held out)

LEGACY_E2A_ROOT = paths.OUTPUT_ROOT / "e2a"
FOLD_COLUMNS = ["k_fold"] + [f"val_fold_{i}" for i in range(paths.N_FOLDS)]

# Subcohorts that must be reported separately, never pooled (design §stratification).
SUBCOHORTS = {
    "CPTAC": ["CPTAC-COAD"],
    "RIH": ["RIH-Colon"],
    "SurGen": ["SR386", "SR1482"],
    "TCGA": ["TCGA-COAD", "TCGA-READ"],
}
# The only SurGen subcohort with metastases — so the SurGen primary/metastatic
# contrast is SR1482-P vs SR1482-M, never pooled SR386+SR1482 vs SR1482-M.
SURGEN_MET_SUBCOHORT = "SR1482"


# ── arms ─────────────────────────────────────────────────────────────────────
# An ARM is a held-out cohort plus, optionally, the size-matched suffix. Arms
# are what the directory layout is keyed on; the COHORT is what the target
# manifests and the strata are keyed on. They differ only for RIH_sm, which is a
# second model for the SAME held-out cohort, so it must reuse RIH's target
# manifest and RIH's strata while writing to its own run directory.
SIZE_MATCH_SUFFIX = "_sm"


def matched_arm(target: str) -> str:
    return f"{target}{SIZE_MATCH_SUFFIX}"


def cohort_of(arm: str) -> str:
    return arm[: -len(SIZE_MATCH_SUFFIX)] if arm.endswith(SIZE_MATCH_SUFFIX) else arm


def is_matched(arm: str) -> bool:
    return arm.endswith(SIZE_MATCH_SUFFIX)


# ── paths ────────────────────────────────────────────────────────────────────
def source_stem(target: str) -> str:
    # Folds inside this manifest are IGNORED: refit mode sends train+val+test
    # all to training. One manifest per arm serves every seed.
    return f"aim1_e2a_{target.lower()}_source_seed42"


def source_manifest(target: str) -> Path:
    if is_matched(target) and lineage.lineage_name(required=False) is not None:
        return e2a_root() / "inputs" / "manifests" / f"{source_stem(target)}.csv"
    return paths.MANIFEST_DIR / f"{source_stem(target)}.csv"


def target_manifest(target: str, kind: str) -> Path:
    # Keyed on the COHORT, not the arm: the size-matched arm is scored on exactly
    # the same held-out patients as its unmatched twin, or the comparison it
    # exists to license would not be a comparison.
    return paths.MANIFEST_DIR / f"aim1_e2a_{cohort_of(target).lower()}_{kind}.csv"


def data_name(target: str) -> str:
    return f"e2a_{target.lower()}"


def split_dir(target: str) -> Path:
    if is_matched(target) and lineage.lineage_name(required=False) is not None:
        return (
            e2a_root()
            / "inputs"
            / "splits"
            / f"aim1_{data_name(target)}"
            / paths.SPLIT_NAME
        )
    return REPO / paths.SPLIT_ROOT / f"aim1_{data_name(target)}" / paths.SPLIT_NAME


def split_root(target: str) -> Path:
    """Root passed to Hydra so FoundationPaths resolves ``split_dir``."""

    if is_matched(target) and lineage.lineage_name(required=False) is not None:
        return e2a_root() / "inputs" / "splits"
    return REPO / paths.SPLIT_ROOT


def e2a_root() -> Path:
    """Selected immutable rerun root, or the legacy root for read-only imports."""

    if lineage.lineage_name(required=False) is None:
        return LEGACY_E2A_ROOT
    return lineage.component_root("e2a")


def run_dir(target: str, seed: int, cap: int) -> Path:
    return e2a_root() / "train" / f"pb_cap{cap}" / target.lower() / f"seed{seed}"


def model_ckpt(target: str, seed: int, cap: int) -> Path:
    return run_dir(target, seed, cap) / "final" / "refit" / "model.ckpt"


def source_cv_dir(target: str, seed: int, cap: int = SRC_CV_CAP) -> Path:
    """Read path for source-CV, preferring this lineage then explicit baseline."""

    local = source_cv_output_dir(target, seed, cap)
    if local.is_dir():
        return local
    if lineage.lineage_name(required=False) is None:
        return local
    return lineage.source_cv_input_root() / f"cap{cap}" / target.lower() / f"seed{seed}"


def source_cv_output_dir(target: str, seed: int, cap: int = SRC_CV_CAP) -> Path:
    return e2a_root() / "source_cv" / f"cap{cap}" / target.lower() / f"seed{seed}"


def calibrator_path(target: str, cap: int = SRC_CV_CAP) -> Path:
    return e2a_root() / "calibrators" / f"cap{cap}_{target.lower()}.json"


def calibrated_path(target: str, kind: str, cap: int = SRC_CV_CAP) -> Path:
    return e2a_root() / "calibrated" / f"cap{cap}_{target.lower()}_{kind}.parquet"


def calibrated_receipt_path(target: str, kind: str, cap: int = SRC_CV_CAP) -> Path:
    return calibrated_path(target, kind, cap).with_suffix(".receipt.json")


def scores_path(target: str, seed: int, kind: str, cap: int) -> Path:
    return e2a_root() / "scores" / f"pb_cap{cap}_{target.lower()}_seed{seed}_{kind}.parquet"


def fit_summary_path(target: str, seed: int, cap: int) -> Path:
    return run_dir(target, seed, cap) / "fit_summary.json"


def score_receipt_path(target: str, seed: int, kind: str, cap: int) -> Path:
    return scores_path(target, seed, kind, cap).with_suffix(".receipt.json")


def _completed_refit(target: str, seed: int, cap: int) -> dict:
    """Validate the immutable completion receipt and checkpoint hash."""

    summary_path = fit_summary_path(target, seed, cap)
    checkpoint = model_ckpt(target, seed, cap)
    if not summary_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(
            f"Incomplete refit for {target}/seed{seed}/cap{cap}: expected "
            f"{summary_path} and {checkpoint}"
        )
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "completed":
        raise RuntimeError(f"Refit receipt is not completed: {summary_path}")
    expected = {
        "lineage": lineage.lineage_name(),
        "target": target,
        "seed": int(seed),
        "sampling_seed": int(seed),
        "cap": int(cap),
        "optimizer_step_budget": int(STEP_BUDGET),
        "sampler": "patient_natural",
        "loss_weighting": "none",
    }
    mismatched = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    result = summary.get("result") or {}
    result_expected = {
        "actual_optimizer_steps": int(STEP_BUDGET),
        "refit_max_steps": int(STEP_BUDGET),
        "batch_size": 1,
        "accumulate_grad_batches": 1,
        "seed": int(seed),
        "sampling_seed": int(seed),
        "lr_scheduler": "cosine",
        "lr_scheduler_interval": "step",
        "lr_scheduler_total_steps": int(STEP_BUDGET),
        "train_sampling_strategy": "patient_natural",
        "sample_weight_column": None,
        "class_weights": None,
        "dataset_max_instances": int(cap),
        "max_instances": None,
        "eval_full_bags": True,
    }
    mismatched.update(
        {
            f"result.{key}": {"expected": value, "observed": result.get(key)}
            for key, value in result_expected.items()
            if result.get(key) != value
        }
    )
    final_lrs = result.get("final_learning_rates")
    if not (
        isinstance(final_lrs, list)
        and len(final_lrs) == 1
        and np.isclose(float(final_lrs[0]), 1.0e-6, rtol=1.0e-9, atol=1.0e-12)
    ):
        mismatched["result.final_learning_rates"] = {
            "expected": [1.0e-6],
            "observed": final_lrs,
        }
    if mismatched:
        raise RuntimeError(
            f"Refit completion contract mismatch for {summary_path}: {mismatched}"
        )
    recorded = (summary.get("model") or {}).get("sha256")
    observed = lineage.sha256_file(checkpoint)
    if recorded != observed:
        raise RuntimeError(
            f"Checkpoint hash mismatch for {checkpoint}: receipt={recorded}, observed={observed}"
        )
    return summary


def _score_inputs(target: str, seed: int, kind: str, cap: int) -> dict:
    from oceanpath.workflows.training import _feature_inventory_sha256

    _completed_refit(target, seed, cap)
    manifest_path = target_manifest(target, kind)
    return {
        "checkpoint": lineage.artifact_identity(model_ckpt(target, seed, cap)),
        "manifest": lineage.artifact_identity(manifest_path),
        "feature_store": {
            "path": str(paths.PINNED_FEATURE_DIR.resolve()),
            "selected_inventory_sha256": _feature_inventory_sha256(
                paths.PINNED_FEATURE_DIR,
                manifest_path,
                "slide_id",
            ),
        },
        "kind": kind,
        "target": target,
        "seed": int(seed),
        "cap": int(cap),
    }


def _load_valid_score_cache(target: str, seed: int, kind: str, cap: int) -> pd.DataFrame | None:
    destination = scores_path(target, seed, kind, cap)
    receipt_path = score_receipt_path(target, seed, kind, cap)
    if not destination.exists() and not receipt_path.exists():
        return None
    if not destination.is_file() or not receipt_path.is_file():
        raise RuntimeError(
            f"Partial score cache for {target}/seed{seed}/{kind}; refusing reuse"
        )
    receipt = json.loads(receipt_path.read_text())
    expected = _score_inputs(target, seed, kind, cap)
    if receipt.get("inputs") != expected:
        raise RuntimeError(
            f"Score cache input identity mismatch for {destination}; use a new lineage"
        )
    observed = lineage.artifact_identity(destination)
    if receipt.get("artifact") != observed:
        raise RuntimeError(f"Score cache hash mismatch for {destination}")
    return pd.read_parquet(destination)


def _load_valid_calibrated(target: str, kind: str, cap: int) -> pd.DataFrame | None:
    destination = calibrated_path(target, kind, cap)
    receipt_path = calibrated_receipt_path(target, kind, cap)
    if not destination.exists() and not receipt_path.exists():
        return None
    if not destination.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"Partial calibrated cache for {target}/{kind}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("artifact") != lineage.artifact_identity(destination):
        raise RuntimeError(f"Calibrated artifact hash mismatch for {destination}")
    current_inputs = [
        lineage.artifact_identity(score_receipt_path(target, seed, kind, cap))
        for seed in SEEDS
    ]
    if receipt.get("score_receipts") != current_inputs:
        raise RuntimeError(f"Calibrated score lineage mismatch for {destination}")
    return pd.read_parquet(destination)


# ── population ───────────────────────────────────────────────────────────────
def load_primary() -> pd.DataFrame:
    return population.attach_technical(
        population.add_context_columns(population.eligible()), technical.load()
    )


def load_metastatic() -> pd.DataFrame:
    return population.eligible_metastatic()


def epochs_for(target: str) -> tuple[int, int]:
    """(epoch ceiling, n_patients) containing the exact optimizer-step budget.

    ``training.refit_max_steps`` is authoritative and stops the final partial
    epoch exactly at ``STEP_BUDGET``.  The ceiling here merely gives Lightning
    enough epochs to reach that stop.
    """
    n = pd.read_csv(source_manifest(target))["patient_id"].nunique()
    return max(MIN_EPOCHS, math.ceil(STEP_BUDGET / n)), n


def ensure_splits(target: str) -> Path:
    """predefined_oof_kfold loads a precomputed splits.parquet and fails hard if
    it is missing; the fold columns it reads are unused in refit mode but the
    loader still requires the file."""
    from oceanpath.splitting import SplitConfig, generate_splits

    directory = split_dir(target)
    if (directory / "splits.parquet").is_file():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    generate_splits(
        SplitConfig(
            scheme="predefined_oof_kfold",
            name=paths.SPLIT_NAME,
            csv_path=str(source_manifest(target)),
            output_dir=str(directory),
            filename_column="slide_id",
            label_column="target_label",
            group_column="patient_id",
            fold_column="k_fold",
            n_folds=paths.N_FOLDS,
            seed=paths.PRIMARY_SEED,
        ),
        force=False,
    )
    return directory


# ── commands ─────────────────────────────────────────────────────────────────
def cmd_plan(_: argparse.Namespace) -> None:
    primary, met = load_primary(), load_metastatic()
    print("E2a — leave-one-parent-cohort-out, ONE full-source fit per (target, seed)\n")
    print(f"{'held out':9s} {'source N':>9s} {'src mut/WT':>12s} {'tgt primary':>15s} "
          f"{'tgt met':>13s} {'slides':>7s} {'epochs':>7s}")
    for t in TARGETS:
        src = primary[primary["cohort"].ne(t)]
        c = population.patient_counts(src)
        tp = population.patient_counts(primary[primary["cohort"].eq(t)])
        mt = met[met["cohort"].eq(t)]
        tm = population.patient_counts(mt) if t in MET_TARGETS else {"n": 0, "mutant": 0}
        ep, ns = epochs_for(t) if source_manifest(t).is_file() else (0, 0)
        mut_wt = "{}/{}".format(c["mutant"], c["n"] - c["mutant"])
        tp_txt = "{} ({} mut)".format(tp["n"], tp["mutant"])
        tm_txt = "{} ({} mut)".format(tm["n"], tm["mutant"]) if tm["n"] else "—"
        print(f"{t:9s} {c['n']:9d} {mut_wt:>12s} {tp_txt:>15s} {tm_txt:>13s} "
              f"{ns:7d} {ep:7d}")
    core = len(TARGETS) * len(SEEDS) * (paths.N_FOLDS + 1)
    print(f"\nE2a CORE = {core} fits: {len(TARGETS)} targets x {len(SEEDS)} seeds x "
          f"({paths.N_FOLDS} source-CV + 1 full-source refit)")
    print("  5 source-CV fits -> source OOF predictions -> 1 full-source refit -> "
          "target prediction")
    print(f"  size-matched {SIZE_MATCHED_TARGET} sensitivity = +{len(SEEDS) * (paths.N_FOLDS + 1)} "
          f"fits (source subsampled to N={SIZE_MATCH_N}),")
    print("  run ONLY if cross-target transport MAGNITUDE differences are interpreted.")
    print(f"step budget {STEP_BUDGET} patient-visits (Aim-1 patient-balanced: 1010 patients "
          "x median best epoch 6);\n  epochs derived per target so every model gets the SAME "
          "number of gradient updates.")
    print("sampler: patient_natural — one uniformly drawn slide per patient per epoch; "
          "UNWEIGHTED loss")
    print(f"encoder UNIv1 · ABMIL · cap {SRC_CV_CAP} (study cap); cap 4,096 is the "
          "sensitivity arm")
    print("\nSTRATA reported separately, never pooled:")
    for t, subs in SUBCOHORTS.items():
        if len(subs) > 1:
            print(f"  {t}: {', '.join(subs)}")
    print("\nTARGET = PRIMARY TUMOURS ONLY. The 'tgt met' column above is built into a")
    print("manifest here and scored in E2b — E2a never evaluates a metastatic slide.")
    print(f"  E2b's SurGen contrast will use {SURGEN_MET_SUBCOHORT}-P vs "
          f"{SURGEN_MET_SUBCOHORT}-M only (the only SurGen subcohort with metastases).")
    dual = population.dual_specimen_patients(primary, met)
    rih_dual = {p for p in dual if p in set(primary.loc[primary.cohort.eq("RIH"), "patient_id"])}
    print(f"  {len(rih_dual)} RIH patients appear in BOTH roles — E2b's paired rule, not E2a's.")
    print("  TCGA's single metastasis is excluded (n=1).")


def _add_fold_columns(src: pd.DataFrame) -> pd.DataFrame:
    """Fold + early-stopping columns for the source pool.

    The refit ignores this partition entirely; source-CV is what consumes it.
    Stratified on cohort x KRAS and grouped by patient, so no patient's two
    slides ever straddle a fold boundary.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    src = src.copy()
    pats = src.drop_duplicates("patient_id")
    strata = pats["cohort"].astype(str) + "|" + pats["target_label"].astype(str)
    sp = StratifiedGroupKFold(n_splits=paths.N_FOLDS, shuffle=True, random_state=42)
    fold_of: dict[str, int] = {}
    for f, (_, te) in enumerate(sp.split(pats, strata, groups=pats["patient_id"])):
        for pid in pats.iloc[te]["patient_id"]:
            fold_of[pid] = f
    src["k_fold"] = src["patient_id"].map(fold_of).astype(int)
    for f in range(paths.N_FOLDS):
        pool = src[src["k_fold"] != f]
        val = balance.carve_out_validation(
            pool, population.BALANCE_COLUMNS, pool["patient_id"].unique(),
            paths.ES_VAL_RATIO,
        )
        src[f"val_fold_{f}"] = (
            src["patient_id"].isin(val) & src["k_fold"].ne(f)
        ).astype(int)
    return population.finalize(src, extra=[*population.BALANCE_COLUMNS, *FOLD_COLUMNS])


def _subsample_source(src: pd.DataFrame, n_patients: int, seed: int) -> pd.DataFrame:
    """Patient-level subsample to n_patients, stratified on cohort x KRAS.

    Proportional allocation with largest-remainder rounding, so the source
    cohort mix and class balance are the ones the unmatched arm had - only the
    size changes. Anything else would confound size with composition, which is
    the confound this arm exists to remove.
    """
    pats = src.drop_duplicates("patient_id")[["patient_id", "cohort", "target_label"]]
    key = pats["cohort"].astype(str) + "|" + pats["target_label"].astype(str)
    share = key.value_counts() / len(pats)
    exact = share * n_patients
    take = np.floor(exact).astype(int)
    for k in (exact - take).sort_values(ascending=False).index[: n_patients - int(take.sum())]:
        take[k] += 1
    rng = np.random.default_rng(seed)
    keep: list[str] = []
    for k, n in take.items():
        pool = pats.loc[key.eq(k), "patient_id"].to_numpy()
        keep.extend(rng.choice(pool, size=int(n), replace=False).tolist())
    return src[src["patient_id"].isin(set(keep))].copy()


def cmd_manifests(args: argparse.Namespace) -> None:
    """Source manifests carry fold columns only because the splits loader wants
    them; refit mode ignores the partition entirely."""
    primary, met = load_primary(), load_metastatic()
    for t in TARGETS:
        path = source_manifest(t)
        if not path.is_file():
            out = _add_fold_columns(primary[primary["cohort"].ne(t)])
            if args.apply:
                lineage.write_text_once(path, out.to_csv(index=False))
        print(f"  {t}/source: {'exists' if path.is_file() else 'built'} -> {path.name}")

        tp = population.finalize(
            primary[primary["cohort"].eq(t)], extra=[*population.BALANCE_COLUMNS]
        )
        if args.apply and not target_manifest(t, "primary").is_file():
            lineage.write_text_once(
                target_manifest(t, "primary"), tp.to_csv(index=False)
            )
        print(f"  {t}/target_primary: {tp.patient_id.nunique()} patients")
        if t in MET_TARGETS:
            # Built here, scored NOWHERE in E2a. E2b consumes these.
            rows = met[met["cohort"].eq(t)]
            cols = [c for c in ("liver_class", "met_site_class") if c in rows.columns]
            tm = population.finalize(rows, extra=[*population.BALANCE_COLUMNS, *cols])
            if args.apply and not target_manifest(t, "metastatic").is_file():
                lineage.write_text_once(
                    target_manifest(t, "metastatic"), tm.to_csv(index=False)
                )
            print(f"  {t}/target_metastatic: {tm.patient_id.nunique()} patients "
                  "(for E2b; E2a never scores it)")

    if args.size_matched:
        arm = matched_arm(SIZE_MATCHED_TARGET)
        path = source_manifest(arm)
        full = primary[primary["cohort"].ne(SIZE_MATCHED_TARGET)]
        sub = _subsample_source(full, SIZE_MATCH_N, paths.PRIMARY_SEED)
        out = _add_fold_columns(sub)
        if args.apply and not path.is_file():
            lineage.write_text_once(path, out.to_csv(index=False))
        c_full = population.patient_counts(full)
        c_sub = population.patient_counts(sub)
        print(f"\n  {arm}/source: {c_full['n']} -> {c_sub['n']} patients "
              f"({c_sub['mutant']} mut / {c_sub['n'] - c_sub['mutant']} WT) -> {path.name}")
        print("    " + " · ".join(
            f"{k} {v}" for k, v in sub.drop_duplicates('patient_id')['cohort']
            .value_counts().items()))

    print("\nWrote." if args.apply else "\nDry run — pass --apply to write.")


_REFIT_SOURCE_FILES = (
    "aim2_loco_transport.py",
    "src/oceanpath/aim1/lineage.py",
    "src/oceanpath/aim1/evaluate.py",
    "src/oceanpath/aim1/scoring.py",
    "src/oceanpath/datasets/datamodule.py",
    "src/oceanpath/datasets/__init__.py",
    "src/oceanpath/datasets/packed.py",
    "src/oceanpath/datasets/sampling.py",
    "src/oceanpath/config/__init__.py",
    "src/oceanpath/config/access.py",
    "src/oceanpath/config/paths.py",
    "src/oceanpath/models/__init__.py",
    "src/oceanpath/models/abmil.py",
    "src/oceanpath/models/base.py",
    "src/oceanpath/models/components.py",
    "src/oceanpath/models/mhabmil.py",
    "src/oceanpath/models/static.py",
    "src/oceanpath/models/transmil.py",
    "src/oceanpath/models/wsi_classifier.py",
    "src/oceanpath/training/lightning.py",
    "src/oceanpath/workflows/finalize.py",
    "configs/train.yaml",
    "configs/data/aim1.yaml",
    "configs/encoder/univ1.yaml",
    "configs/model/abmil.yaml",
    "configs/platform/colon_workstation.yaml",
    "configs/splits/aim1_balanced.yaml",
    "configs/training/aim1.yaml",
    "configs/training/default.yaml",
    "pyproject.toml",
    "uv.lock",
)


def _refit_source_files() -> tuple[str, ...]:
    """All local Python/configuration bytes that can affect a refit."""

    files = set(_REFIT_SOURCE_FILES)
    files.update(
        str(path.relative_to(REPO))
        for path in (REPO / "src" / "oceanpath").rglob("*.py")
    )
    return tuple(sorted(files))


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _snapshot_refit_source(directory: Path) -> list[dict]:
    """Copy the exact relevant dirty source into the immutable run directory."""

    snapshot = directory / "source_snapshot"
    entries: list[dict] = []
    for relative in _refit_source_files():
        source = REPO / relative
        if not source.is_file():
            raise FileNotFoundError(f"Required refit source is missing: {source}")
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        lineage.ensure_absent(destination)
        shutil.copy2(source, destination)
        entries.append(
            {
                "relative_path": relative,
                "sha256": lineage.sha256_file(destination),
                "size_bytes": int(destination.stat().st_size),
            }
        )
    return entries


def _verify_source_snapshot(entries: list[dict], directory: Path) -> None:
    """Prove the child process is importing the bytes the parent snapshotted."""

    observed_paths = {str(entry.get("relative_path")) for entry in entries}
    expected_paths = set(_refit_source_files())
    if observed_paths != expected_paths:
        raise RuntimeError(
            "Refit source snapshot inventory mismatch: "
            f"missing={sorted(expected_paths - observed_paths)}, "
            f"unexpected={sorted(observed_paths - expected_paths)}"
        )
    for entry in entries:
        relative = str(entry["relative_path"])
        live = REPO / relative
        frozen = directory / "source_snapshot" / relative
        if not live.is_file():
            raise RuntimeError(f"Snapshotted refit source disappeared: {live}")
        expected_hash = entry.get("sha256")
        live_hash = lineage.sha256_file(live)
        frozen_hash = lineage.sha256_file(frozen)
        if live_hash != expected_hash or frozen_hash != expected_hash:
            raise RuntimeError(
                f"Refit source changed after run request: {relative}; "
                f"recorded={expected_hash}, frozen={frozen_hash}, live={live_hash}"
            )


def _packed_store_identity(manifest_path: Path) -> dict:
    """Bind a run to the packed feature-store metadata and relevant coverage.

    Hashing the 48-GB binary before every fit is unnecessary and would make a
    12-run campaign spend hours rereading immutable bytes.  The pack's content
    contract lives in ``meta.json`` and ``index.parquet``; both are hashed, and
    the two binary payloads are bound by absolute path, byte size and nanosecond
    mtime.  Relevant slide coverage is independently checked against the input
    manifest.
    """

    pack = paths.PACKED_FEATURE_DIR.resolve()
    meta_path = pack / "meta.json"
    index_path = pack / "index.parquet"
    meta = json.loads(meta_path.read_text())
    index = pd.read_parquet(index_path, columns=["slide_id"])
    expected_slides = set(pd.read_csv(manifest_path, usecols=["slide_id"])["slide_id"])
    indexed_slides = set(index["slide_id"].astype(str))
    missing = sorted(expected_slides - indexed_slides)
    if missing:
        raise RuntimeError(
            f"Packed feature store is missing {len(missing)} source slides; "
            f"first={missing[:5]}"
        )

    payloads = {}
    for name in ("features.bin", "coords.bin"):
        payload = pack / name
        stat = payload.stat()
        payloads[name] = {
            "path": str(payload),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return {
        "root": str(pack),
        "meta": lineage.artifact_identity(meta_path),
        "index": lineage.artifact_identity(index_path),
        "payloads": payloads,
        "source_inventory_sha256_recorded": meta.get(
            "source_inventory_sha256"
        ),
        "manifest_slide_count": int(len(expected_slides)),
        "manifest_missing_from_index": 0,
    }


def _run_request(target: str, seed: int, cap: int, epochs: int, directory: Path) -> dict:
    split_file = split_dir(target) / "splits.parquet"
    return {
        "schema_version": 2,
        "status": "requested",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lineage": lineage.lineage_name(),
        "target": target,
        "seed": int(seed),
        "sampling_seed": int(seed),
        "cap": int(cap),
        "optimizer_step_budget": int(STEP_BUDGET),
        "epoch_ceiling": int(epochs),
        "run_dir": str(directory.resolve()),
        "git_head": _git_output("rev-parse", "HEAD"),
        "git_status_porcelain": _git_output("status", "--porcelain=v1"),
        "inputs": {
            "source_manifest": lineage.artifact_identity(source_manifest(target)),
            "splits": lineage.artifact_identity(split_file),
            "packed_store": _packed_store_identity(source_manifest(target)),
        },
        "source_snapshot": _snapshot_refit_source(directory),
    }


def _run_logged(command: list[str], log_path: Path) -> int:
    lineage.ensure_absent(log_path)
    with log_path.open("x", encoding="utf-8", buffering=1) as log_stream:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_stream.write(line)
        return int(process.wait())


def train_one(target: str, seed: int, cap: int, dry_run: bool) -> None:
    """ONE full-source fit: no validation split, no early stopping, no selection."""
    lineage.lineage_name()  # mutating commands must select a new lineage
    directory = run_dir(target, seed, cap)
    if directory.exists():
        try:
            _completed_refit(target, seed, cap)
        except Exception as exc:
            raise RuntimeError(
                f"Refusing to resume or overwrite partial immutable run {directory}; "
                "select a new lineage for the retry"
            ) from exc
        print(f"== {target} seed{seed} cap{cap}: completed and hash-valid — skipping")
        return
    ensure_splits(target)
    epochs, n_pat = epochs_for(target)
    print(f"== {target} seed{seed} cap{cap}: {n_pat} patients, epoch ceiling {epochs}, "
          f"exact optimizer-step budget {STEP_BUDGET}")
    if dry_run:
        return
    directory.mkdir(parents=True, exist_ok=False)
    request = _run_request(target, seed, cap, epochs, directory)
    lineage.write_json_once(directory / "run_request.json", request)
    command = [
        sys.executable, str(REPO / "aim2_loco_transport.py"), "_fit",
        "--target", target, "--seed", str(seed), "--epochs", str(epochs),
        "--cap", str(cap), "--run-dir", str(directory),
    ]
    rc = _run_logged(command, directory / "stdout_stderr.log")
    if rc != 0:
        lineage.write_json_once(
            directory / "failure.json",
            {
                "status": "failed",
                "returncode": rc,
                "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "command": command,
            },
        )
        raise SystemExit(
            f"{target} seed{seed} cap{cap} failed; evidence retained at {directory}. "
            "Use a new lineage for any retry."
        )
    _completed_refit(target, seed, cap)


def cmd_fit(args: argparse.Namespace) -> None:
    """Internal: compose the frozen Aim-1 config and run ONE full-data fit.

    Reuses the pipeline's own refit path rather than a hand-rolled loop, so the
    loss, the per-slide patient weighting, the sampler and the optimizer are
    byte-for-byte the ones Aim 1 used. The fold-metrics argument is synthetic:
    it exists only to pin the epoch count, because there are no folds here.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from oceanpath.workflows.finalize import _run_refit

    lineage.lineage_name()
    target, seed, epochs, cap = args.target, args.seed, args.epochs, args.cap
    directory = Path(args.run_dir).resolve()
    expected_directory = run_dir(target, seed, cap).resolve()
    if directory != expected_directory:
        raise RuntimeError(
            f"Internal refit run-dir mismatch: expected {expected_directory}, got {directory}"
        )
    if not (directory / "run_request.json").is_file():
        raise RuntimeError(f"Missing immutable run request: {directory / 'run_request.json'}")
    request = json.loads((directory / "run_request.json").read_text())
    _verify_source_snapshot(request.get("source_snapshot") or [], directory)
    request_expected = {
        "lineage": lineage.lineage_name(),
        "target": target,
        "seed": int(seed),
        "sampling_seed": int(seed),
        "cap": int(cap),
        "optimizer_step_budget": int(STEP_BUDGET),
        "epoch_ceiling": int(epochs),
        "run_dir": str(directory),
    }
    request_mismatches = {
        key: {"expected": value, "observed": request.get(key)}
        for key, value in request_expected.items()
        if request.get(key) != value
    }
    current_inputs = {
        "source_manifest": lineage.artifact_identity(source_manifest(target)),
        "splits": lineage.artifact_identity(split_dir(target) / "splits.parquet"),
        "packed_store": _packed_store_identity(source_manifest(target)),
    }
    if request.get("inputs") != current_inputs:
        request_mismatches["inputs"] = {
            "expected": current_inputs,
            "observed": request.get("inputs"),
        }
    if request_mismatches:
        raise RuntimeError(
            f"Immutable refit request mismatch for {directory}: {request_mismatches}"
        )
    overrides = [
        "platform=colon_workstation", "data=aim1",
        f"data.aim1_model={data_name(target)}",
        f"data.manifest_stem={source_stem(target)}",
        f"data.csv_path={source_manifest(target)}",
        f"platform.splits_root={split_root(target)}",
        "+data.cohort_column=cohort",
        "encoder=univ1", "splits=aim1_balanced",
        "model=abmil", "model.embed_dim=512", "model.attn_dim=384",
        "model.input_dropout=0.10", f"model.dropout={paths.DROPOUT}",
        "training=aim1", f"training.lr={paths.LR:g}",
        f"training.weight_decay={paths.WEIGHT_DECAY:g}",
        f"training.seed={seed}",
        f"training.dataset_max_instances={cap}",
        "training.eval_full_bags=true",
        "training.refit_epoch_rule=median",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        f"train_dir={run_dir(target, seed, cap)}",
        f"exp_name=e2a_pb_{target.lower()}_c{cap}_s{seed}",
    ]
    with initialize_config_dir(config_dir=str((REPO / "configs").resolve()), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    with open_dict(cfg):
        cfg.training.max_epochs = int(epochs)
        cfg.training.refit_max_steps = int(STEP_BUDGET)
    lineage.write_text_once(
        directory / "resolved_config.yaml", OmegaConf.to_yaml(cfg, resolve=True)
    )
    final_dir = directory / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    result = _run_refit(cfg, final_dir, [{"best_epoch": int(epochs)}])
    checkpoint = final_dir / "refit" / "model.ckpt"
    lineage.write_json_once(
        directory / "fit_summary.json",
        {
            "schema_version": 2,
            "status": "completed",
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "lineage": lineage.lineage_name(),
            "target": target,
            "seed": int(seed),
            "sampling_seed": int(seed),
            "cap": int(cap),
            "epoch_ceiling": int(epochs),
            "optimizer_step_budget": int(STEP_BUDGET),
            "sampler": "patient_natural",
            "loss_weighting": "none",
            "model": lineage.artifact_identity(checkpoint),
            "resolved_config": lineage.artifact_identity(directory / "resolved_config.yaml"),
            "result": result,
        },
    )
    print(f"refit complete: {result}")


def arms_for(args: argparse.Namespace) -> list[str]:
    """The size-matched arm is a separate model for an already-covered cohort,
    so it is never mixed into a default sweep - it is requested or it is absent."""
    if getattr(args, "size_matched", False):
        return [matched_arm(SIZE_MATCHED_TARGET)]
    return [args.target] if args.target else list(TARGETS)


def cmd_train(args: argparse.Namespace) -> None:
    caps = [args.cap] if args.cap else list(E2A_CAPS)
    for t in arms_for(args):
        for s in ([args.seed] if args.seed else list(SEEDS)):
            for c in caps:
                train_one(t, s, c, args.dry_run)


def score_one(target: str, seed: int, kind: str, cap: int) -> pd.DataFrame:
    lineage.lineage_name()
    destination = scores_path(target, seed, kind, cap)
    cached = _load_valid_score_cache(target, seed, kind, cap)
    if cached is not None:
        return cached
    ckpt = model_ckpt(target, seed, cap)
    inputs = _score_inputs(target, seed, kind, cap)
    scores = scoring.score_manifest_with_checkpoints(
        checkpoints=[(seed, 0, ckpt)],
        manifest_csv=target_manifest(target, kind),
        feature_dir=paths.PINNED_FEATURE_DIR,
        num_classes=2,
    )
    if "logit" not in scores.columns or not np.isfinite(scores["logit"]).all():
        raise RuntimeError(f"Scorer did not emit finite native logits for {target}/{kind}")
    lineage.write_parquet_once(destination, scores)
    lineage.write_json_once(
        score_receipt_path(target, seed, kind, cap),
        {
            "schema_version": 2,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "lineage": lineage.lineage_name(),
            "inputs": inputs,
            "artifact": lineage.artifact_identity(destination),
            "n_rows": int(len(scores)),
        },
    )
    return scores


def patients_for(target: str, seed: int, kind: str, cap: int) -> pd.DataFrame:
    """Patient x specimen-role logits for one seed. Each manifest holds exactly
    one role, so grouping by patient within it IS grouping by patient x role."""
    manifest = pd.read_csv(target_manifest(target, kind))
    # to_patient_level groups by patient_id alone, so "patient x role" holds only
    # while a manifest carries ONE role. It does today (primary and metastatic
    # are separate files) — assert it rather than rely on it, because a patient
    # with both roles pooled into one logit would silently destroy the
    # specimen-role contrast this experiment exists to measure.
    roles = set(manifest["specimen_role"].dropna().unique())
    if len(roles) != 1:
        raise SystemExit(
            f"{target}/{kind} manifest mixes specimen roles {sorted(roles)}; patient-level "
            "aggregation would pool them. Split the manifest by role first."
        )
    slides = scoring.ensemble_slide_predictions(score_one(target, seed, kind, cap), manifest)
    pat = evaluate.to_patient_level(slides, manifest)
    keep = [c for c in ("patient_id", "label", "mean_logit", "cohort", "subcohort",
                        "msi_dmmr", "braf") if c in pat.columns]
    return pat[keep].assign(role=kind)


def seed_ensemble(target: str, kind: str, cap: int) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Headline predictor: mean of the three seed-level patient LOGITS, sigmoid
    applied ONCE at the end. Never concatenate seeds as extra patients."""
    per_seed: dict[int, pd.DataFrame] = {}
    missing = [seed for seed in SEEDS if not run_dir(target, seed, cap).is_dir()]
    if missing:
        raise RuntimeError(
            f"{target}/{kind}/cap{cap}: missing refit seeds {missing}; "
            "refusing a partial ensemble"
        )
    for seed in SEEDS:
        _completed_refit(target, seed, cap)
        per_seed[seed] = patients_for(target, seed=seed, kind=kind, cap=cap)
    frames = [f.sort_values("patient_id").reset_index(drop=True) for f in per_seed.values()]
    base = frames[0].copy()
    ids = [tuple(f["patient_id"]) for f in frames]
    if len({tuple(i) for i in ids}) != 1:
        raise SystemExit("seeds cover different patients; cannot ensemble")
    base["mean_logit"] = np.mean([f["mean_logit"].to_numpy() for f in frames], axis=0)
    base["prob_raw"] = sigmoid(base["mean_logit"].to_numpy())
    for f in per_seed.values():
        f["prob_raw"] = sigmoid(f["mean_logit"].to_numpy())
    return base, per_seed


def block_metrics(
    pat: pd.DataFrame,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    bootstrap_seed: int = paths.BOOTSTRAP_SEED,
) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    from oceanpath.eval.core import compute_calibration_intercept_slope

    y = pat["label"].to_numpy()
    p = np.clip(pat["prob_raw"].to_numpy(), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "degenerate": True}
    eta = pat["mean_logit"].to_numpy()
    boot = evaluate.bootstrap_auroc(
        pat,
        score_column="mean_logit",
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    cal = compute_calibration_intercept_slope(y, p)
    return {
        "n": int(len(y)), "n_mutant": int(y.sum()), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, eta)),
        "auroc_ci": [boot["ci_low"], boot["ci_high"]],
        "auprc": float(average_precision_score(y, eta)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "calibration_intercept": float(cal["calibration_intercept"]),
        "calibration_slope": float(cal["calibration_slope"]),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(bootstrap_seed),
    }


def cmd_score(args: argparse.Namespace) -> None:
    """PRIMARY ONLY. Metastatic slides are scored by E2b, from these same
    checkpoints - not here, and not into E2a's report."""
    lineage.lineage_name()
    requested_arms = arms_for(args)
    # Validate the complete requested ensemble before publishing the first
    # score file.  A late missing/corrupt seed must not leave a half-scored
    # lineage that looks successful from the artifacts written before it.
    for t in requested_arms:
        for s in SEEDS:
            _completed_refit(t, s, args.cap)
    for t in requested_arms:
        for s in SEEDS:
            n = len(score_one(t, s, "primary", args.cap))
            print(f"  cap{args.cap} {t} seed{s} primary    : {n} slide scores")


def _auroc_on(pat: pd.DataFrame, column: str) -> float:
    """AUROC against an arbitrary score column, for exact monotonicity audits."""
    from sklearn.metrics import roc_auc_score

    y = pat["label"].to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, pat[column].to_numpy()))


def _auroc(pat: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score

    y = pat["label"].to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, pat["mean_logit"].to_numpy()))


def cmd_source_cv(args: argparse.Namespace) -> None:
    """5-fold CV over each source pool -> patient-level source OOF logits.

    Same recipe as everything else: patient-balanced sampling, unweighted loss,
    full-bag inference. skip_finalize because only the OOF vector is wanted --
    the deployed object is the refit that already exists.
    """
    lineage.lineage_name()
    cap = args.cap
    for t in arms_for(args):
        for seed in ([args.seed] if args.seed else list(SEEDS)):
            d = source_cv_output_dir(t, seed, cap)
            if d.exists():
                from oceanpath.workflows.training import validate_training_run_dir

                try:
                    validate_training_run_dir(d, require_test_predictions=True)
                except Exception as exc:
                    raise RuntimeError(
                        f"Refusing to resume or overwrite partial source-CV run {d}; "
                        "select a new lineage for the retry"
                    ) from exc
                print(f"== src-cv {t} seed{seed} cap{cap}: complete and valid — skipping")
                continue
            ensure_splits(t)
            overrides = [
                "platform=colon_workstation", "data=aim1",
                f"data.aim1_model={data_name(t)}", f"data.manifest_stem={source_stem(t)}",
                f"data.csv_path={source_manifest(t)}",
                f"platform.splits_root={split_root(t)}",
                "+data.cohort_column=cohort", "encoder=univ1",
                "splits=aim1_balanced", f"splits.seed={seed}",
                "model=abmil", "model.embed_dim=512", "model.attn_dim=384",
                "model.input_dropout=0.10", f"model.dropout={paths.DROPOUT}",
                "training=aim1", f"training.lr={paths.LR:g}",
                f"training.weight_decay={paths.WEIGHT_DECAY:g}", f"training.seed={seed}",
                f"training.dataset_max_instances={cap}", "training.eval_full_bags=true",
                "training.train_sampling_strategy=patient_natural",
                "training.sample_weight_column=null",
                "training.skip_finalize=true",
                f"train_dir={d}", f"exp_name=e2a_srccv_{t.lower()}_c{cap}_s{seed}",
                f"hydra.run.dir={e2a_root() / 'hydra_runs' / f'source_cv_{t.lower()}_cap{cap}_seed{seed}'}",
                "hydra.job.chdir=false",
            ]
            cmd = [sys.executable, str(REPO / "tools" / "study_train.py"), "hydra-train", *overrides]
            print(f"== src-cv {t} seed{seed} cap{cap}")
            if args.dry_run:
                print("   " + " ".join(cmd))
                continue
            log_path = (
                e2a_root()
                / "launcher_logs"
                / f"source_cv_{t.lower()}_cap{cap}_seed{seed}.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if _run_logged(cmd, log_path) != 0:
                raise SystemExit(
                    f"src-cv {t} seed{seed} failed; partial evidence retained at {d}. "
                    "Use a new lineage for any retry."
                )
            from oceanpath.workflows.training import validate_training_run_dir

            validate_training_run_dir(d, require_test_predictions=True)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Steps 2-5: average seed OOF logits, fit ONE source-only calibrator per
    LOCO direction, apply to the frozen target ensemble, freeze the result."""
    from sklearn.linear_model import LogisticRegression

    lineage.lineage_name()
    cap = args.cap
    suffix = "_size_matched" if args.size_matched else ""
    final_dest = (
        lineage.eval_root() / f"e2a_source_calibration_cap{cap}{suffix}.json"
    )
    lineage.ensure_absent(final_dest)
    requested_arms = arms_for(args)
    # Calibrate is a pure downstream publication step.  Validate every input
    # and every destination up front so it cannot silently score a missing seed
    # or publish one target before discovering that a later target is partial.
    from oceanpath.workflows.training import validate_training_run_dir

    for t in requested_arms:
        lineage.ensure_absent(calibrator_path(t, cap))
        lineage.ensure_absent(calibrated_path(t, "primary", cap))
        lineage.ensure_absent(calibrated_receipt_path(t, "primary", cap))
        for seed in SEEDS:
            validate_training_run_dir(
                source_cv_dir(t, seed, cap), require_test_predictions=True
            )
            if _load_valid_score_cache(t, seed, "primary", cap) is None:
                raise FileNotFoundError(
                    f"Missing primary score cache for {t}/seed{seed}/cap{cap}; "
                    "run `aim2_loco_transport.py score` before calibration"
                )
    out: dict = {"lineage": lineage.lineage_name(), "cap": cap, "targets": {}}
    print(f"\n{'=' * 96}\nE2a SOURCE-ONLY CALIBRATION · cap {cap}\n{'=' * 96}")
    for t in requested_arms:
        # ── step 2: average the three seed-specific OOF logits per source patient
        frames = {}
        for seed in SEEDS:
            source_run = source_cv_dir(t, seed, cap)
            validate_training_run_dir(source_run, require_test_predictions=True)
            f = source_run / "oof_predictions.parquet"
            man = pd.read_csv(source_manifest(t))
            frames[seed] = evaluate.to_patient_level(pd.read_parquet(f), man)
        base = None
        stack = []
        for seed in sorted(frames):
            g = frames[seed].sort_values("patient_id").reset_index(drop=True)
            if base is None:
                base = g[["patient_id", "label", "cohort", "subcohort"]].copy()
            elif list(g["patient_id"]) != list(base["patient_id"]):
                raise SystemExit(f"{t}: source-CV seeds cover different patients")
            stack.append(g["mean_logit"].to_numpy())
        eta_src = np.mean(np.vstack(stack), axis=0)
        y_src = base["label"].to_numpy()

        # ── step 3: one source-only calibrator per LOCO direction (no target labels)
        lr = LogisticRegression(penalty=None, max_iter=1000)
        lr.fit(eta_src.reshape(-1, 1), y_src)
        a_src, b_src = float(lr.intercept_[0]), float(lr.coef_[0][0])
        pre = _auroc_on(
            pd.DataFrame({"label": y_src, "mean_logit": eta_src}),
            "mean_logit",
        )
        from oceanpath.eval.core import compute_calibration_intercept_slope
        cb = compute_calibration_intercept_slope(y_src, np.clip(sigmoid(eta_src), 1e-6, 1 - 1e-6))
        info = {"a": a_src, "b": b_src, "n_source": int(len(y_src)),
                "source_prevalence": float(y_src.mean()), "source_oof_auroc": pre,
                "source_oof_cal_intercept": float(cb["calibration_intercept"]),
                "source_oof_cal_slope": float(cb["calibration_slope"]),
                "note": "fitted on 3-seed-averaged source OOF logits; no target labels"}
        print(f"\n  {t}: source n={info['n_source']} prev={info['source_prevalence']:.1%} "
              f"OOF AUROC {pre:.4f}  cal(int/slope) {info['source_oof_cal_intercept']:+.3f}/"
              f"{info['source_oof_cal_slope']:.3f}")
        print(f"      calibrator  logit(p) = {a_src:+.4f} + {b_src:.4f} * eta")

        # ── steps 4-5: apply to the frozen target ensemble and freeze
        # PRIMARY ONLY. E2b reloads this same frozen calibrator and applies it to
        # the metastatic ensemble, so the two experiments share one mapping and
        # a metastatic calibration failure cannot be an artefact of refitting.
        for kind in ("primary",):
            ens, _ = seed_ensemble(t, kind, cap)
            if ens.empty:
                continue
            eta_t = ens["mean_logit"].to_numpy()
            ens = ens.assign(
                eta_source_calibrated=a_src + b_src * eta_t,
                prob_source_calibrated=sigmoid(a_src + b_src * eta_t),
            )
            destination = calibrated_path(t, kind, cap)
            lineage.write_parquet_once(destination, ens)
            score_receipts = [
                lineage.artifact_identity(score_receipt_path(t, seed, kind, cap))
                for seed in SEEDS
            ]
            lineage.write_json_once(
                calibrated_receipt_path(t, kind, cap),
                {
                    "schema_version": 2,
                    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "lineage": lineage.lineage_name(),
                    "calibrator": {"a": a_src, "b": b_src},
                    "score_receipts": score_receipts,
                    "artifact": lineage.artifact_identity(destination),
                },
            )
            y = ens["label"].to_numpy()
            b0 = compute_calibration_intercept_slope(
                y, np.clip(sigmoid(eta_t), 1e-6, 1 - 1e-6))
            b1 = compute_calibration_intercept_slope(
                y, np.clip(ens["prob_source_calibrated"].to_numpy(), 1e-6, 1 - 1e-6))
            print(f"      {kind:11s} n={len(ens):4d}  cal intercept "
                  f"{float(b0['calibration_intercept']):+.3f} -> "
                  f"{float(b1['calibration_intercept']):+.3f}   slope "
                  f"{float(b0['calibration_slope']):.3f} -> "
                  f"{float(b1['calibration_slope']):.3f}")
            info.setdefault("applied", {})[kind] = {"n": int(len(ens))}
        out["targets"][t] = info
        info["source_cv_inputs"] = {
            str(seed): lineage.artifact_identity(
                source_cv_dir(t, seed, cap) / "oof_predictions.parquet"
            )
            for seed in SEEDS
        }
        lineage.write_json_once(calibrator_path(t, cap), info)
    dest = final_dest
    lineage.write_json_once(dest, out)
    print("\n  NOTE: the calibrator is fitted on CV models trained on 4/5 of the source and")
    print( "  applied to a refit trained on 5/5. That scale mismatch is real and is why both")
    print( "  the pre- and post-calibration intercepts are printed above.")
    print(f"\nWrote {dest}")


def cmd_report(args: argparse.Namespace) -> None:
    """PRIMARY targets only. The metastatic arm is E2b's report, not this one."""
    lineage.lineage_name()
    suffix = "_size_matched" if args.size_matched else ""
    dest = lineage.eval_root() / f"e2a_transport_pb_cap{args.cap}{suffix}.json"
    lineage.ensure_absent(dest)
    report: dict = {
        "schema_version": 2,
        "lineage": lineage.lineage_name(),
        "scope": "primary targets only; metastatic transport is E2b",
        "cap": args.cap,
        "targets": {},
        "macro": {},
        "diagnosis": {},
        "input_refits": {},
        "inference": {
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "sampling_unit": "patient",
            "conditioning": "confidence intervals condition on the frozen fitted ensemble",
        },
    }
    requested_arms = arms_for(args)
    for t in requested_arms:
        if _load_valid_calibrated(t, "primary", args.cap) is None:
            raise FileNotFoundError(
                f"Missing calibrated primary artifact for {t}; run calibration first"
            )
        for seed in SEEDS:
            if _load_valid_score_cache(t, seed, "primary", args.cap) is None:
                raise FileNotFoundError(
                    f"Missing primary score cache for {t}/seed{seed}/cap{args.cap}"
                )
    if args.size_matched:
        for seed in SEEDS:
            if _load_valid_score_cache(
                SIZE_MATCHED_TARGET, seed, "primary", args.cap
            ) is None:
                raise FileNotFoundError(
                    "Size-matched comparison requires all standard RIH scores"
                )
    strata_auroc: dict[str, dict[str, float]] = {}

    for t in requested_arms:
        ens_p, per_seed_p = seed_ensemble(t, "primary", args.cap)
        if ens_p.empty:
            continue
        block: dict = {"seeds_complete": sorted(per_seed_p)}
        report["input_refits"][t] = {
            str(seed): lineage.artifact_identity(fit_summary_path(t, seed, args.cap))
            for seed in sorted(per_seed_p)
        }
        block["primary_overall"] = block_metrics(
            ens_p,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )
        block["primary_per_seed_auroc"] = {s: _auroc(f) for s, f in per_seed_p.items()}

        # strata — reported separately, never pooled
        block["primary_strata"] = {}
        for sub in SUBCOHORTS[cohort_of(t)]:
            blk = ens_p[ens_p["subcohort"].eq(sub)]
            if len(blk) > 10:
                block["primary_strata"][sub] = block_metrics(
                    blk,
                    n_bootstrap=args.n_bootstrap,
                    bootstrap_seed=args.bootstrap_seed,
                )
        strata_auroc[t] = {
            sub: b["auroc"] for sub, b in block["primary_strata"].items() if "auroc" in b
        }

        # D subset — same predictions, no retraining
        d = ens_p[ens_p["msi_dmmr"].eq("MSS/pMMR") & ens_p["braf"].eq("wild_type")]
        if len(d) > 20:
            block["primary_D_subset"] = block_metrics(
                d,
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.bootstrap_seed,
            )

        # ── discrimination vs calibration: Aim 2's second clause ───────────
        # Transport can fail two ways and they call for different fixes. Ranking
        # loss is a model problem (E2c retrains nothing and cannot repair it);
        # a calibration-only failure is a mapping problem that a source-only or
        # few-shot recalibration CAN repair. Report which one it is, per target.
        cal = _load_valid_calibrated(t, "primary", args.cap)
        if cal is None:
            raise RuntimeError(
                f"Missing or invalid source-calibrated primary artifact for {t}; "
                "run `aim2_loco_transport.py calibrate` before publishing the E2a report"
            )
        if cal is not None:
            from sklearn.metrics import brier_score_loss, log_loss

            from oceanpath.eval.core import compute_calibration_intercept_slope

            y = cal["label"].to_numpy()
            raw = np.clip(cal["prob_raw"].to_numpy(), 1e-6, 1 - 1e-6)
            src = np.clip(cal["prob_source_calibrated"].to_numpy(), 1e-6, 1 - 1e-6)
            after = compute_calibration_intercept_slope(y, src)
            block["source_calibrated"] = {
                "n": int(len(cal)),
                "log_loss_raw": float(log_loss(y, raw)),
                "log_loss_source_calibrated": float(log_loss(y, src)),
                "brier_raw": float(brier_score_loss(y, raw)),
                "brier_source_calibrated": float(brier_score_loss(y, src)),
                "calibration_intercept": float(after["calibration_intercept"]),
                "calibration_slope": float(after["calibration_slope"]),
                # AUDIT ON THE LINEAR PREDICTOR, WHICH IS EXACT.
                # A positive-slope Platt map is monotone, so AUROC must be
                # bit-identical — but only on the logit scale. On the PROBABILITY
                # scale sigmoid compresses, and at |logit| ~ 8 two distinct logits
                # can round to the same float64 probability, creating or removing
                # ties. Auditing there reports a spurious ~1e-4 "change" in a
                # quantity that provably cannot change. The probability-scale
                # shift is reported as a magnitude, not as a pass/fail.
                "auroc_unchanged": bool(
                    _auroc_on(cal, "eta_source_calibrated") == _auroc_on(cal, "mean_logit")
                ),
                "auroc_prob_scale_shift": float(
                    abs(
                        _auroc_on(
                            cal.assign(_source_calibrated_probability=src),
                            "_source_calibrated_probability",
                        )
                        - _auroc_on(cal, "prob_raw")
                    )
                ),
                "auroc_exact_logit_scale": _auroc_on(cal, "mean_logit"),
                "note": ("positive-slope Platt preserves ranking; verified on the "
                         "linear predictor, where it is exact"),
            }
        report["targets"][t] = block

    # The size-matched arm exists specifically to test whether the larger RIH
    # source pool explains its apparent transport advantage.  Both refits score
    # the identical held-out RIH patients, so this is a paired patient bootstrap
    # of matched-source minus standard-source AUROC—not two unrelated CIs.
    if args.size_matched:
        standard, standard_seeds = seed_ensemble(
            SIZE_MATCHED_TARGET, "primary", args.cap
        )
        matched, matched_seeds = seed_ensemble(
            matched_arm(SIZE_MATCHED_TARGET), "primary", args.cap
        )
        comparison = evaluate.compare_auroc_paired(
            standard,
            matched,
            score_column="mean_logit",
            n_bootstrap=args.n_bootstrap,
            seed=args.bootstrap_seed,
        )
        comparison.update(
            {
                "estimand": "AUROC(size-matched source) - AUROC(full source)",
                "left_arm": SIZE_MATCHED_TARGET,
                "right_arm": matched_arm(SIZE_MATCHED_TARGET),
                "n_bootstrap": int(args.n_bootstrap),
                "bootstrap_seed": int(args.bootstrap_seed),
                "bootstrap_method": (
                    "paired patient bootstrap on identical held-out RIH patients"
                ),
                "standard_seeds": sorted(standard_seeds),
                "size_matched_seeds": sorted(matched_seeds),
                "standard_refits": {
                    str(seed): lineage.artifact_identity(
                        fit_summary_path(SIZE_MATCHED_TARGET, seed, args.cap)
                    )
                    for seed in SEEDS
                },
                "size_matched_refits": {
                    str(seed): lineage.artifact_identity(
                        fit_summary_path(
                            matched_arm(SIZE_MATCHED_TARGET), seed, args.cap
                        )
                    )
                    for seed in SEEDS
                },
            }
        )
        report["size_matched_sensitivity"] = comparison

    # ── macro-AUROC: standardize within cohort, then equal-weight ───────────
    per_target_macro = {}
    for t, subs in strata_auroc.items():
        vals = [v for v in subs.values() if np.isfinite(v)]
        if vals:
            per_target_macro[t] = float(np.mean(vals))
    report["macro"]["per_target_standardized"] = per_target_macro
    if len(per_target_macro) == len(TARGETS):
        report["macro"]["four_target_macro_auroc"] = float(np.mean(list(per_target_macro.values())))
    report["macro"]["note"] = (
        "SurGen standardized across SR386/SR1482 and TCGA across COAD/READ before the "
        "equal-weight four-target mean. Raw patient scores are never pooled across cohorts."
    )

    # ── failure-mode diagnosis, four targets ───────────────────────────────
    # Not a conclusion RULE: E2a fixes no operating point and claims no threshold.
    # It classifies each target so E2b/E2c inherit a stated failure mode rather
    # than rediscovering it.
    for t, b in report["targets"].items():
        o = b.get("primary_overall") or {}
        sc = b.get("source_calibrated")
        if not o or o.get("degenerate"):
            continue
        ci_low = o.get("auroc_ci", [float("nan")])[0]
        ranks = bool(np.isfinite(ci_low) and ci_low > 0.5)
        entry = {
            "auroc": o["auroc"], "auroc_ci_low": ci_low,
            "ranking_transports": ranks,
            "calibration_intercept_raw": o["calibration_intercept"],
            "calibration_slope_raw": o["calibration_slope"],
        }
        if sc:
            repaired = sc["log_loss_source_calibrated"] < sc["log_loss_raw"]
            entry["source_calibration_repairs_log_loss"] = bool(repaired)
            entry["log_loss_raw"] = sc["log_loss_raw"]
            entry["log_loss_source_calibrated"] = sc["log_loss_source_calibrated"]
            entry["failure_mode"] = (
                "none" if ranks and not repaired
                else "calibration" if ranks
                else "discrimination"
            )
        else:
            entry["failure_mode"] = "unknown — no source calibrator yet"
        report["diagnosis"][t] = entry

    lineage.write_json_once(dest, report)
    print_report(report)
    print(f"\nWrote {dest}")


def print_report(report: dict) -> None:
    def line(label, b, indent=4):
        if not b or b.get("degenerate"):
            return
        ci = b.get("auroc_ci", [float("nan")] * 2)
        print(f"{' ' * indent}{label:24s} n={b['n']:5d} prev={b['prevalence']:5.1%}  "
              f"AUROC {b['auroc']:.4f} [{ci[0]:.3f}-{ci[1]:.3f}]  AUPRC {b['auprc']:.4f}  "
              f"Brier {b['brier']:.4f}  logloss {b['log_loss']:.4f}  "
              f"cal {b['calibration_intercept']:+.3f}/{b['calibration_slope']:.3f}")

    for t, b in report["targets"].items():
        print(f"\n{'=' * 118}\nE2a · HELD-OUT TARGET = {t} · seeds {b['seeds_complete']}\n{'=' * 118}")
        line("PRIMARY (all)", b.get("primary_overall"))
        for sub, blk in b.get("primary_strata", {}).items():
            line(f"  {sub}", blk, indent=6)
        line("PRIMARY · MSS+BRAF-WT", b.get("primary_D_subset"))
        ps = b.get("primary_per_seed_auroc", {})
        if ps:
            vals = [v for v in ps.values() if np.isfinite(v)]
            print(f"    per-seed AUROC {{{', '.join(f'{k}: {v:.4f}' for k, v in ps.items())}}}"
                  f"   seed range {max(vals) - min(vals):.4f}"
                  f"   (ensemble is the headline; range is NOT a CI)")
        sc = b.get("source_calibrated")
        if sc:
            print(f"    source-calibrated (no target labels): log loss "
                  f"{sc['log_loss_raw']:.4f} -> {sc['log_loss_source_calibrated']:.4f}   "
                  f"Brier {sc['brier_raw']:.4f} -> {sc['brier_source_calibrated']:.4f}   "
                  f"cal {sc['calibration_intercept']:+.3f}/{sc['calibration_slope']:.3f}"
                  f"   AUROC unchanged (logit scale, exact): {sc['auroc_unchanged']}"
                  f"   prob-scale float shift {sc['auroc_prob_scale_shift']:.1e}")

    sensitivity = report.get("size_matched_sensitivity")
    if sensitivity:
        print(
            f"\n{'=' * 118}\nSIZE-MATCHED RIH SENSITIVITY\n{'=' * 118}"
        )
        print(
            "  AUROC(size-matched source) - AUROC(full source) "
            f"= {sensitivity['delta_auroc']:+.4f}  95% CI "
            f"[{sensitivity['ci_low']:+.4f}, {sensitivity['ci_high']:+.4f}] "
            f"(paired, n={sensitivity['n_patients']})"
        )

    m = report.get("macro", {})
    if m.get("per_target_standardized"):
        print(f"\n{'=' * 118}\nMACRO SUMMARY\n{'=' * 118}")
        for t, v in m["per_target_standardized"].items():
            print(f"  {t:9s} standardized AUROC {v:.4f}   "
                  f"(mean over {', '.join(SUBCOHORTS[cohort_of(t)])})")
        if "four_target_macro_auroc" in m:
            print(f"  {'MACRO':9s} four-target equal-weight AUROC "
                  f"{m['four_target_macro_auroc']:.4f}")
        print(f"  {m['note']}")

    d = report.get("diagnosis", {})
    if d:
        print(f"\n{'=' * 118}\nFAILURE-MODE DIAGNOSIS (primary targets; metastatic is E2b)"
              f"\n{'=' * 118}")
        for t, e in d.items():
            print(f"  {t:9s} AUROC {e['auroc']:.4f} (CI low {e['auroc_ci_low']:.3f})  "
                  f"ranking transports: {str(e['ranking_transports']):5s}  "
                  f"raw cal {e['calibration_intercept_raw']:+.3f}/"
                  f"{e['calibration_slope_raw']:.3f}  -> {e['failure_mode']}")
        print("\n  ranking transports = target AUROC 95% CI lower bound > 0.50.")
        print("  failure_mode 'calibration' means the ranking survived and a "
              "SOURCE-ONLY map repaired\n  the probabilities — no target label used. "
              "'discrimination' means recalibration\n  cannot help and E2c must be "
              "read as adaptation, not repair.")
        print("  E2a fixes no operating point and makes no threshold claim.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("plan")
    x.set_defaults(func=cmd_plan)
    x = sub.add_parser("manifests")
    x.add_argument("--apply", action="store_true")
    x.add_argument("--size-matched", action="store_true",
                   help=f"also build the {SIZE_MATCHED_TARGET} source pool subsampled "
                        f"to N={SIZE_MATCH_N}")
    x.set_defaults(func=cmd_manifests)
    x = sub.add_parser("train")
    x.add_argument("--target", choices=TARGETS)
    x.add_argument("--seed", type=int)
    x.add_argument("--cap", type=int, choices=E2A_CAPS)
    x.add_argument("--size-matched", action="store_true")
    x.add_argument("--dry-run", action="store_true")
    x.set_defaults(func=cmd_train)
    x = sub.add_parser("_fit")  # internal single fit
    x.add_argument("--target", required=True)
    x.add_argument("--seed", type=int, required=True)
    x.add_argument("--epochs", type=int, required=True)
    x.add_argument("--cap", type=int, required=True)
    x.add_argument("--run-dir", type=Path, required=True)
    x.set_defaults(func=cmd_fit)
    x = sub.add_parser("score", help="score TARGET PRIMARY only; metastatic is E2b")
    x.add_argument("--target", choices=TARGETS)
    x.add_argument("--size-matched", action="store_true")
    x.add_argument("--cap", type=int, required=True, choices=E2A_CAPS)
    x.set_defaults(func=cmd_score)
    x = sub.add_parser("source-cv", help="5-fold CV over each source pool -> source OOF")
    x.add_argument("--target", choices=TARGETS)
    x.add_argument("--seed", type=int)
    x.add_argument("--cap", type=int, default=SRC_CV_CAP)
    x.add_argument("--size-matched", action="store_true")
    x.add_argument("--dry-run", action="store_true")
    x.set_defaults(func=cmd_source_cv)
    x = sub.add_parser("calibrate", help="steps 2-5: source-only calibrator -> frozen target probs")
    x.add_argument("--target", choices=TARGETS)
    x.add_argument("--size-matched", action="store_true")
    x.add_argument("--cap", type=int, default=SRC_CV_CAP)
    x.set_defaults(func=cmd_calibrate)
    x = sub.add_parser("report")
    x.add_argument("--target", choices=TARGETS)
    x.add_argument("--size-matched", action="store_true")
    x.add_argument("--cap", type=int, required=True, choices=E2A_CAPS)
    x.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    x.add_argument("--bootstrap-seed", type=int, default=paths.BOOTSTRAP_SEED)
    x.set_defaults(func=cmd_report)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
