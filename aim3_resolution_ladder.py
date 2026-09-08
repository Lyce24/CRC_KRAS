#!/usr/bin/env python3
"""E3 - AIM 3, the molecular-resolution ladder (150 fits).

NAMING. The aims split 2026-08-19: this experiment is AIM 3 / E3, and the
morphologic atlas that was "E3b" is now AIM 4 / E4 (`aim4_morphologic_atlas_base.py`). The
file name, the run directories under `outputs/aim1/e3a/`, the manifest stems
`aim1_e3a_*` and `e3a_resolution.json` all still say "e3a" and are NOT being
migrated - the sweep was live when the split was decided, and this module has
already paid once for renaming a live experiment (see `migrate`). See
`Experimental_Setup.md` header for the naming map.

QUESTION. How far below KRAS gene status can H&E resolve molecular differences?
Six rungs of increasing molecular resolution, each scored the same way:

    gene         KRAS mutant vs wild-type        604 / 882   (reuse E0, no new fits)
    codon        G12* vs non-G12 KRAS mutants    420 / 184
    g12d_broad   G12D vs other KRAS mutants      186 / 418
    allele1      G12D vs other G12* mutants      186 / 234
    allele2      G12V vs other G12* mutants      125 / 295
    g12c         G12C vs other G12* mutants       53 / 367

WHY BOTH A BROAD AND A WITHIN-G12 G12D RUNG. Every SUBSTITUTION rung is
G12D/G12V/G12C-WITHIN-G12, not against every other mutant: comparing inside
codon 12 isolates substitution-level information, where the wider contrast
would confound codon differences with allele differences and could look like
allele resolution when it is only codon signal. `g12d_broad` is the one
deliberate exception - it IS the wider contrast - and it is reported BESIDE
`allele1`, never instead of it. The pair is what separates the two: if
`g12d_broad` succeeds where `allele1` fails, the separable part is CODON 12,
not D, and the broad result must not be written up as allele resolution.

THE CONTROL IS THE POINT. "G12C is at chance" is fully explained by n = 53 and
says nothing. So each fine rung gets a MATCHED GENE-LEVEL CONTROL: the same
positive patients, the same class sizes, the same folds and recipe, but the
negatives replaced by cohort-matched wild-type patients - a task known to be
learnable at that exact sample size. Only the comparison of the two is
informative.

    ctrl_codon        G12* (420) vs 184 cohort-matched WT
    ctrl_g12d_broad   G12D (186) vs 418 cohort-matched WT
    ctrl_allele1      G12D (186) vs 234 cohort-matched WT
    ctrl_allele2      G12V (125) vs 295 cohort-matched WT
    ctrl_g12c         G12C  (53) vs 367 cohort-matched WT

Cohort matching mirrors the *aggregate cohort* composition of the class it
replaces.  The frozen controls share every positive patient with their fine
task, but use a separately sampled wild-type negative arm.  Inference from
these controls is therefore conditional on the prespecified wild-type draw;
it does not integrate uncertainty over alternative control draws.

THE HEADLINE IS THE THREE-SEED OOF LOGIT ENSEMBLE. Per-seed pooled OOF AUROC is
retained as the algorithmic-robustness panel, but every conclusion is read off

    z_i(ensemble) = ( z_i,42 + z_i,43 + z_i,44 ) / 3

with the sigmoid applied ONCE afterwards - E2a's ensemble convention. The
reason is asymmetric and load-bearing: ensembling suppresses initialisation
noise and so gives the FINE rung its best reasonable opportunity to find a weak
signal. A ceiling declared against a noisier single-seed fine estimate would be
partly an artefact of variance; a ceiling that survives after the fine model
has been given its best shot, while the matched control succeeds on the same
treatment, is a real one. No new training - a re-read of frozen OOF predictions.

THE DELTA CONTRAST IS ESTIMATED DIRECTLY. Printing two CIs side by side and
eyeballing their overlap is not an inference, so the corrected report
bootstraps

    delta_resolution = AUROC(control) - AUROC(fine)

as one quantity.  The arms are *partially paired*: every positive patient is
shared, while the negative patients differ.  Each replicate therefore draws
the shared positives once (and applies that draw to both models), then draws
the two negative arms independently.  Resampling is stratified by cohort.  A
fully paired or fully unpaired bootstrap is wrong for this design.

CONCLUSION RULE, revised 2026-08-19. The earlier two-condition rule (fine upper
CI < 0.60 AND control lower CI > 0.50 -> ceiling) is superseded by a
three-condition rule that adds the delta requirement, evaluated IN THIS ORDER:

  1. UNDERPOWERED           control lower 95% CI <= 0.50. Checked FIRST and
                            dominates: if the matched easy task cannot be
                            learned at this n, the fine null says nothing.
  2. CEILING                fine upper CI < 0.60 AND control lower CI > 0.50
                            AND delta lower CI > 0.
  3. FINE-RESOLUTION        fine lower CI > 0.50 (AUROC at or above ~0.60 is
                            practically meaningful evidence).
  4. INCONCLUSIVE           everything else.

The third condition is the point of the revision: it makes the resolution
ceiling a POSITIVE empirical result - the control is measurably better than the
fine rung at the same n - rather than a failure to reject chance.

Conditions 2 and 3 can both hold (fine CI entirely inside 0.50-0.60). That is a
real outcome, not a rule conflict, and it is reported as its own verdict -
CEILING_WITH_RESIDUAL_SIGNAL - rather than silently resolved by ordering:
calling it a plain ceiling would hide a fine rung that is demonstrably above
chance, and calling it plain fine-resolution evidence would hide the bound.

SMALL-CLASS RULE. G12C has 53 positives, so a 15% early-stopping carve-out holds
~6-7 - below the floor of 8, and too few for val/patient_auroc to select an
epoch. The pipeline's designed response fires: early stopping is disabled for
that fold and a FIXED 12-epoch budget is trained, deploying the final epoch.
The 12 is the p75 best-epoch across the 60 completed E3a fits - the repo's own
refit convention, not a number chosen here. Among the modelled rungs only g12c
trips the floor; `manifests` prints exactly which (task, fold) pairs will.

Rungs below G12C (G12A 26, G12S 29, G12R 3) are NOT modelled - the minority
class cannot support a fold structure.

Usage:
    python aim3_resolution_ladder.py plan
    python aim3_resolution_ladder.py migrate --apply     # allele -> allele1, once
    python aim3_resolution_ladder.py manifests --apply
    python aim3_resolution_ladder.py train [--task codon] [--seed 42]
    python aim3_resolution_ladder.py report --output-root /new/immutable/result/root
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import balance, evaluate, paths  # noqa: E402
from oceanpath.eval.external import sigmoid  # noqa: E402

SEEDS = (42, 43, 44)
E3A_CAP = 8192
FIXED_EPOCH_BUDGET = 12   # p75 best-epoch over the 60 completed E3a fits
E3A_ROOT = paths.OUTPUT_ROOT / "e3a"
FOLD_COLUMNS = ["k_fold"] + [f"val_fold_{i}" for i in range(paths.N_FOLDS)]

# Verdict thresholds, fixed before any result was read (see the module docstring).
CEILING_BOUND = 0.60
CHANCE = 0.50

# The ladder. Every fine rung is paired with a matched gene-level control at the
# SAME class sizes, folds, recipe and cohort composition, because a fine rung at
# chance is uninterpretable without evidence that the sample size supports
# gene-level learning.
TASKS = {
    "codon":           ("G12* vs non-G12 mutants",        "fine"),
    "g12d_broad":      ("G12D vs other mutants",          "fine"),
    "allele1":         ("G12D vs other G12*",             "fine"),
    "allele2":         ("G12V vs other G12*",             "fine"),
    "g12c":            ("G12C vs other G12*",             "fine"),
    "ctrl_codon":      ("G12* vs cohort-matched WT",      "control"),
    "ctrl_g12d_broad": ("G12D vs cohort-matched WT",      "control"),
    "ctrl_allele1":    ("G12D vs cohort-matched WT",      "control"),
    "ctrl_allele2":    ("G12V vs cohort-matched WT",      "control"),
    "ctrl_g12c":       ("G12C vs cohort-matched WT",      "control"),
}
PAIRS = [
    ("codon", "ctrl_codon"),
    ("g12d_broad", "ctrl_g12d_broad"),
    ("allele1", "ctrl_allele1"),
    ("allele2", "ctrl_allele2"),
    ("g12c", "ctrl_g12c"),
]
# Human-readable rung labels for the headline table.
RUNG_LABEL = {
    "codon": "codon        G12 vs non-G12",
    "g12d_broad": "g12d_broad   G12D vs other mutant",
    "allele1": "allele1      G12D vs other G12",
    "allele2": "allele2      G12V vs other G12",
    "g12c": "g12c         G12C vs other G12",
}

# The 2026-08-19 rename. The original 60 fits ran under `allele`/`ctrl_allele`;
# `migrate` moves the artefacts so a rerun finds them instead of retraining 30
# fits. Nothing about the models changes - only the names.
RENAMES = {"allele": "allele1", "ctrl_allele": "ctrl_allele1"}

# Wild-type draw seeds, one per control, so two controls never share a draw.
CONTROL_WT_SEED = {
    "ctrl_codon": 20260818,
    "ctrl_allele1": 20260819,
    "ctrl_allele2": 20260820,
    "ctrl_g12c": 20260821,
    "ctrl_g12d_broad": 20260822,
}
ALLELE_OF = {"allele1": "G12D", "allele2": "G12V", "g12c": "G12C"}


# ── allele parsing ───────────────────────────────────────────────────────────
def _tokens(value: object) -> list[str]:
    return [t.strip() for t in str(value).split(";") if t.strip()]


def is_g12(value: object) -> bool:
    """Any substitution at codon 12. Multi-substitution patients are KEPT and
    counted by membership, not by string order - dropping them would silently
    change the denominator, and resolving them by 'first token wins' would make
    the label depend on how the lab happened to write the call."""
    return any(t.startswith("G12") for t in _tokens(value))


def has_allele(value: object, allele: str) -> bool:
    return any(t == allele for t in _tokens(value))


def is_g12d(value: object) -> bool:
    return has_allele(value, "G12D")


def population_table() -> pd.DataFrame:
    m = pd.read_csv(paths.DEV_MANIFEST)
    pat = m.drop_duplicates("patient_id").copy()
    sv = pat["kras_subvariant"].fillna("UNKNOWN")
    pat["is_g12"] = sv.map(is_g12)
    pat["is_g12d"] = sv.map(is_g12d)
    return pat


def matched_wt(pat: pd.DataFrame, mirror: pd.DataFrame, seed: int) -> pd.Series:
    """Cohort-matched wild-type controls mirroring ``mirror``'s composition.

    Matching is on cohort because cohort is the axis that carries scanner,
    stain and population differences at once; a control drawn without it could
    be easier or harder than its fine task for reasons that have nothing to do
    with molecular resolution.
    """
    rng = np.random.default_rng(seed)
    wt = pat[pat["kras"].eq("wild_type")]
    want = mirror["cohort"].value_counts()
    picks: list[str] = []
    for cohort, k in want.items():
        pool = wt.loc[wt["cohort"].eq(cohort), "patient_id"].to_numpy()
        if len(pool) < k:
            raise SystemExit(f"cohort {cohort}: need {k} WT, only {len(pool)} available")
        picks.extend(rng.choice(pool, size=int(k), replace=False).tolist())
    return pd.Series(picks, name="patient_id")


def task_patients(task: str) -> pd.DataFrame:
    """(patient_id, label) for one task. Label 1 = the finer/positive class."""
    pat = population_table()
    mut = pat[pat["kras"].eq("mutant")]
    g12 = mut[mut["is_g12"]]

    def _within_g12(allele: str) -> pd.Series:
        return g12["kras_subvariant"].fillna("").map(lambda v, a=allele: has_allele(v, a))

    if task == "codon":
        pos, neg = g12, mut[~mut["is_g12"]]
    elif task == "g12d_broad":
        # The one rung scored against EVERY other mutant, on purpose. See the
        # module docstring: it is read only as a pair with allele1.
        pos, neg = mut[mut["is_g12d"]], mut[~mut["is_g12d"]]
    elif task in ALLELE_OF:
        hit = _within_g12(ALLELE_OF[task])
        pos, neg = g12[hit], g12[~hit]
    elif task == "ctrl_codon":
        pos = g12
        neg = pat[pat["patient_id"].isin(
            matched_wt(pat, mut[~mut["is_g12"]], seed=CONTROL_WT_SEED[task]))]
    elif task == "ctrl_g12d_broad":
        pos = mut[mut["is_g12d"]]
        neg = pat[pat["patient_id"].isin(
            matched_wt(pat, mut[~mut["is_g12d"]], seed=CONTROL_WT_SEED[task]))]
    elif task.startswith("ctrl_") and task.removeprefix("ctrl_") in ALLELE_OF:
        hit = _within_g12(ALLELE_OF[task.removeprefix("ctrl_")])
        pos = g12[hit]
        neg = pat[pat["patient_id"].isin(
            matched_wt(pat, g12[~hit], seed=CONTROL_WT_SEED[task]))]
    else:
        raise ValueError(task)

    out = pd.concat([
        pos[["patient_id"]].assign(e3a_label=1),
        neg[["patient_id"]].assign(e3a_label=0),
    ], ignore_index=True)
    assert out["patient_id"].is_unique, f"{task}: a patient appears in both classes"
    return out


def build_manifest(task: str) -> tuple[pd.DataFrame, list[int]]:
    """Slide-level manifest carrying E0's OUTER folds unchanged.

    'Identical folds' means the outer partition is E0's, so a patient never
    changes fold between rungs and the ladder is paired. The early-stopping
    carve-out must be re-drawn per task, because 15% of a 420-patient pool is
    not the same patients as 15% of a 1,486-patient pool.

    Returns the manifest and the folds whose carve-out cannot hold
    ``MIN_ES_VAL_POSITIVES`` positives - those fire the small-class rule in the
    trainer (early stopping off, fixed epoch budget, final epoch deployed).
    """
    slides = pd.read_csv(paths.DEV_MANIFEST)
    sel = task_patients(task)
    rows = slides.merge(sel, on="patient_id", how="inner").copy()
    rows["target_label"] = rows["e3a_label"].astype(int)
    small_class_folds: list[int] = []
    for f in range(paths.N_FOLDS):
        pool = rows[rows["k_fold"] != f]
        val = balance.carve_out_validation(
            pool, ["cohort", "target_label"], pool["patient_id"].unique(), paths.ES_VAL_RATIO
        )
        rows[f"val_fold_{f}"] = (
            rows["patient_id"].isin(val) & rows["k_fold"].ne(f)
        ).astype(int)
        val_patients = rows.loc[rows[f"val_fold_{f}"] == 1].drop_duplicates("patient_id")
        if int(val_patients["target_label"].sum()) < paths.MIN_ES_VAL_POSITIVES:
            small_class_folds.append(f)
    return rows.drop(columns=["e3a_label"]), small_class_folds


def manifest_path(task: str) -> Path:
    return paths.MANIFEST_DIR / f"aim1_e3a_{task}.csv"


def split_dir(task: str) -> Path:
    return REPO / paths.SPLIT_ROOT / f"aim1_e3a_{task}" / paths.SPLIT_NAME


def task_split_root(task: str) -> Path:
    return REPO / paths.SPLIT_ROOT / f"aim1_e3a_{task}"


def run_dir(task: str, seed: int) -> Path:
    return E3A_ROOT / "train" / task / f"seed{seed}"


# ── statistics ───────────────────────────────────────────────────────────────
def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def load_oof(task: str, seed: int) -> pd.DataFrame | None:
    p = run_dir(task, seed) / "oof_predictions.parquet"
    if not p.is_file():
        return None
    man = pd.read_csv(manifest_path(task))
    return evaluate.to_patient_level(pd.read_parquet(p), man)


def ensemble_oof(task: str) -> tuple[pd.DataFrame | None, dict[int, pd.DataFrame]]:
    """Three-seed OOF logit ensemble, plus the per-seed frames beside it.

    Logits are averaged and the sigmoid applied ONCE - never average
    probabilities, and never concatenate seeds as extra patients. E0's outer
    folds are frozen across seeds, so every seed covers exactly the same OOF
    patients; that is asserted rather than assumed, because a silent mismatch
    would make the ensemble an average over different populations.
    """
    per_seed = {s: f for s in SEEDS if (f := load_oof(task, s)) is not None}
    if not per_seed:
        return None, {}
    frames = {
        s: f.sort_values("patient_id").reset_index(drop=True) for s, f in per_seed.items()
    }
    ids = {s: tuple(f["patient_id"]) for s, f in frames.items()}
    if len(set(ids.values())) != 1:
        raise SystemExit(f"{task}: seeds cover different OOF patients; cannot ensemble")
    base = frames[min(frames)].copy()
    base["mean_logit"] = np.mean([f["mean_logit"].to_numpy() for f in frames.values()], axis=0)
    base["prob_raw"] = sigmoid(base["mean_logit"].to_numpy())
    return base, frames


def gene_ensemble() -> tuple[pd.DataFrame | None, dict[int, pd.DataFrame]]:
    """The gene rung: E0's own OOF predictions, ensembled the same way.

    The top of the ladder costs no fit - it IS E0 - but it must be read through
    the SAME estimator as every rung below it, or the descent from gene to codon
    to substitution would be partly a change of estimator. It has no matched
    control by construction: it is the reference the ladder descends from.
    """
    per_seed: dict[int, pd.DataFrame] = {}
    manifest = pd.read_csv(paths.DEV_MANIFEST)
    for seed in SEEDS:
        p = paths.OUTPUT_ROOT / f"train/1a_pb_cap{E3A_CAP}/univ1" / f"seed{seed}" / (
            "oof_predictions.parquet"
        )
        if p.is_file():
            per_seed[seed] = evaluate.to_patient_level(pd.read_parquet(p), manifest)
    if not per_seed:
        return None, {}
    frames = {
        s: f.sort_values("patient_id").reset_index(drop=True) for s, f in per_seed.items()
    }
    if len({tuple(f["patient_id"]) for f in frames.values()}) != 1:
        raise SystemExit("E0 seeds cover different OOF patients; cannot ensemble")
    base = frames[min(frames)].copy()
    base["mean_logit"] = np.mean([f["mean_logit"].to_numpy() for f in frames.values()], axis=0)
    base["prob_raw"] = sigmoid(base["mean_logit"].to_numpy())
    return base, frames


def delta_bootstrap(
    fine: pd.DataFrame,
    control: pd.DataFrame,
    n_bootstrap: int = paths.N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Partially paired patient bootstrap of AUROC(control) - AUROC(fine).

    The fine/control pair shares every positive patient but has arm-specific
    negatives. Positives are resampled once and their two model scores travel
    together; negative arms are resampled independently. All draws are
    stratified by cohort, preserving the frozen comparison's dependence and
    case-mix structure.
    """
    required = {"patient_id", "label", "cohort"}
    for name, frame in (("fine", fine), ("control", control)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} arm lacks required columns: {sorted(missing)}")
        if frame["patient_id"].duplicated().any():
            raise ValueError(f"{name} arm contains duplicate patients")
        labels = set(pd.to_numeric(frame["label"], errors="raise").astype(int).unique())
        if labels != {0, 1}:
            raise ValueError(f"{name} arm must contain both binary classes")

    score_column = (
        "mean_logit"
        if "mean_logit" in fine.columns and "mean_logit" in control.columns
        else "prob_raw"
    )
    if score_column not in fine.columns or score_column not in control.columns:
        raise ValueError("fine/control arms require native mean_logit (preferred) or prob_raw")

    fine_pos = fine.loc[
        fine["label"].eq(1), ["patient_id", "cohort", score_column]
    ].copy()
    control_pos = control.loc[
        control["label"].eq(1), ["patient_id", "cohort", score_column]
    ].copy()
    shared = fine_pos.merge(
        control_pos,
        on="patient_id",
        how="outer",
        suffixes=("_fine", "_control"),
        indicator=True,
        validate="one_to_one",
    )
    if not shared["_merge"].eq("both").all() or len(shared) != len(fine_pos):
        raise ValueError("fine/control positive patient sets are not identical")
    if not shared["cohort_fine"].eq(shared["cohort_control"]).all():
        raise ValueError("shared positive patients disagree on cohort")
    shared = shared.rename(columns={"cohort_fine": "cohort"}).drop(
        columns=["cohort_control", "_merge"]
    )

    fine_neg = fine.loc[fine["label"].eq(0), ["cohort", score_column]].copy()
    control_neg = control.loc[control["label"].eq(0), ["cohort", score_column]].copy()
    if fine_neg.empty or control_neg.empty or shared.empty:
        raise ValueError("fine/control arms require shared positives and arm-specific negatives")

    def draw_scores(
        frame: pd.DataFrame, column: str, rng: np.random.Generator
    ) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for _, block in frame.groupby("cohort", sort=True, observed=True):
            values = pd.to_numeric(block[column], errors="raise").to_numpy(dtype=float)
            pieces.append(values[rng.integers(0, len(values), len(values))])
        return np.concatenate(pieces)

    def draw_shared_positive_scores(
        frame: pd.DataFrame, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        fine_pieces: list[np.ndarray] = []
        control_pieces: list[np.ndarray] = []
        for _, block in frame.groupby("cohort", sort=True, observed=True):
            index = rng.integers(0, len(block), len(block))
            fine_values = pd.to_numeric(
                block[f"{score_column}_fine"], errors="raise"
            ).to_numpy(dtype=float)
            control_values = pd.to_numeric(
                block[f"{score_column}_control"], errors="raise"
            ).to_numpy(dtype=float)
            fine_pieces.append(fine_values[index])
            control_pieces.append(control_values[index])
        return np.concatenate(fine_pieces), np.concatenate(control_pieces)

    rng = np.random.default_rng(seed)
    yf = pd.to_numeric(fine["label"], errors="raise").to_numpy(dtype=int)
    sf = pd.to_numeric(fine[score_column], errors="raise").to_numpy(dtype=float)
    yc = pd.to_numeric(control["label"], errors="raise").to_numpy(dtype=int)
    sc = pd.to_numeric(control[score_column], errors="raise").to_numpy(dtype=float)
    if not all(np.isfinite(array).all() for array in (sf, sc)):
        raise ValueError("fine/control scores must be finite")
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        fine_positive, control_positive = draw_shared_positive_scores(shared, rng)
        fine_negative = draw_scores(fine_neg, score_column, rng)
        control_negative = draw_scores(control_neg, score_column, rng)
        fine_labels = np.r_[np.ones(len(fine_positive)), np.zeros(len(fine_negative))]
        control_labels = np.r_[np.ones(len(control_positive)), np.zeros(len(control_negative))]
        deltas.append(
            _auroc(control_labels, np.r_[control_positive, control_negative])
            - _auroc(fine_labels, np.r_[fine_positive, fine_negative])
        )
    values = np.asarray(deltas, dtype=float)
    return {
        "delta": _auroc(yc, sc) - _auroc(yf, sf),
        "ci": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        if values.size
        else [float("nan")] * 2,
        "bootstrap_fraction_delta_gt_0": (
            float((values > 0).mean()) if values.size else float("nan")
        ),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap": int(values.size),
        "bootstrap_design": (
            "shared positive patient resample within cohort; arm-specific negative "
            "patient resamples independently within cohort"
        ),
        "score_column": score_column,
        "shared_positive_n": int(len(shared)),
    }


def verdict_for(fine: dict, control: dict, delta: dict) -> tuple[str, str]:
    """The pre-fixed conclusion rule, applied in order. Returns (code, prose)."""
    f_lo, f_hi = fine["ci"]
    c_lo = control["ci"][0]
    d_lo = delta["ci"][0]
    if not np.isfinite(c_lo) or c_lo <= CHANCE:
        return (
            "UNDERPOWERED",
            "the matched control is not learnable at this n, so the fine null says nothing",
        )
    ceiling = f_hi < CEILING_BOUND and c_lo > CHANCE and d_lo > 0.0
    above_chance = f_lo > CHANCE
    if ceiling and above_chance:
        return (
            "CEILING_WITH_RESIDUAL_SIGNAL",
            "bounded below 0.60 with the control measurably better, yet still above chance",
        )
    if ceiling:
        return (
            "CEILING",
            "fine rung bounded below 0.60 while the matched control is learnable and better",
        )
    if above_chance:
        return (
            "FINE_RESOLUTION_EVIDENCE",
            "fine rung is above chance"
            + (" and practically meaningful (>=0.60)" if fine["auroc"] >= CEILING_BOUND else ""),
        )
    return ("INCONCLUSIVE", "neither bounded below 0.60 with a positive delta nor above chance")


# ── commands ─────────────────────────────────────────────────────────────────
def cmd_plan(_: argparse.Namespace) -> None:
    pat = population_table()
    mut = pat[pat["kras"].eq("mutant")]
    print("E3a - molecular-resolution ladder (primary tumours, cap 8192, patient-balanced)\n")
    print(f"  {'rung':16s} {'comparison':30s} {'pos':>5s} {'neg':>5s} {'N':>5s}  fits")
    print(f"  {'gene':16s} {'KRAS mutant vs wild-type':30s} "
          f"{int(pat.kras.eq('mutant').sum()):5d} {int(pat.kras.eq('wild_type').sum()):5d} "
          f"{len(pat):5d}  0 (reuse E0)")
    for task, (desc, _kind) in TASKS.items():
        t = task_patients(task)
        print(f"  {task:16s} {desc:30s} {int(t.e3a_label.sum()):5d} "
              f"{int((1 - t.e3a_label).sum()):5d} {len(t):5d}  {paths.N_FOLDS * len(SEEDS)}")
    total = len(TASKS) * paths.N_FOLDS * len(SEEDS)
    print(f"\n  total fits: {len(TASKS)} tasks x {paths.N_FOLDS} folds x {len(SEEDS)} seeds "
          f"= {total}")
    done = sum(
        1 for t in TASKS for s in SEEDS if (run_dir(t, s) / "oof_predictions.parquet").is_file()
    ) * paths.N_FOLDS
    print(f"  complete: {done}/{total}   (a (task, seed) chain counts as {paths.N_FOLDS} fits)")
    print(f"  small-class rule: any fold whose ES carve-out holds < "
          f"{paths.MIN_ES_VAL_POSITIVES} positives disables early stopping and trains a")
    print(f"  fixed {FIXED_EPOCH_BUDGET}-epoch budget (p75 best-epoch over the 60 completed "
          "E3a fits), deploying the final epoch.")
    print("\n  per-fold class counts (E0's outer folds, unchanged):")
    slides = pd.read_csv(paths.DEV_MANIFEST)
    for task in TASKS:
        rows = slides.merge(task_patients(task), on="patient_id")
        p = rows.drop_duplicates("patient_id")
        counts = p.groupby("k_fold")["e3a_label"].agg(["sum", "size"])
        cells = "  ".join(f"f{i}:{int(r['sum'])}/{int(r['size'])}" for i, r in counts.iterrows())
        print(f"    {task:16s} {cells}")
    sv = mut["kras_subvariant"].fillna("UNKNOWN")
    n_g12c = int(sv.str.contains("G12C", na=False).sum())
    print(f"\n  G12C: {n_g12c} primary patients - a MODELLED rung with its own matched")
    print("        control, and still reported descriptively from the locked KRAS scores.")
    print("        At 53 positives it is underpowered BY DESIGN: its matched control is what")
    print("        makes a null interpretable rather than uninformative.")
    print(f"  multi-substitution patients retained: "
          f"{int(sv.str.contains(';', na=False).sum())} (classified by membership, not string order)")
    print("\n  g12d_broad is the ONE rung scored against every other mutant, and is read only")
    print("  as a pair with allele1: if broad succeeds where allele1 fails, the separable")
    print("  part is CODON 12, not D.")
    stale = [old for old in RENAMES if (E3A_ROOT / "train" / old).is_dir()]
    if stale:
        print(f"\n  ACTION: {stale} still carry the pre-rename names. Run `migrate --apply`")
        print("  before `train`, or 30 completed fits will be recomputed under the new names.")


def cmd_migrate(args: argparse.Namespace) -> None:
    """Rename the 2026-08-18 artefacts to their post-rename task names.

    The 60 completed fits were run as `allele`/`ctrl_allele`. `train` skips a
    (task, seed) whose oof_predictions.parquet exists, so without this move it
    would find nothing and retrain 30 fits for a pure naming change. Nothing
    about the models, manifests or splits changes - only the paths.

    The run directories' own config.yaml / training_identity.json keep the old
    absolute train_dir; that is provenance of where they were written and is
    deliberately left alone.
    """
    moves: list[tuple[Path, Path]] = []
    for old, new in RENAMES.items():
        moves.append((E3A_ROOT / "train" / old, E3A_ROOT / "train" / new))
        moves.append((manifest_path(old), manifest_path(new)))
        moves.append((task_split_root(old), task_split_root(new)))
    todo, skipped, blocked = [], [], []
    for src, dst in moves:
        if not src.exists():
            skipped.append((src, "source absent - already migrated or never built"))
        elif dst.exists():
            blocked.append((src, dst))
        else:
            todo.append((src, dst))
    for src, dst in todo:
        print(f"  MOVE  {src}\n     -> {dst}")
    for src, why in skipped:
        print(f"  skip  {src.name}: {why}")
    for src, dst in blocked:
        print(f"  BLOCKED {src.name}: {dst} already exists - resolve by hand, refusing to merge")
    if blocked:
        raise SystemExit("migration blocked; nothing was moved")
    if not args.apply:
        print("\nDry run - pass --apply to move.")
        return
    for src, dst in todo:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    print(f"\nMoved {len(todo)} artefact(s).")


def cmd_manifests(args: argparse.Namespace) -> None:
    from oceanpath.splitting import SplitConfig, generate_splits

    small_class: dict[str, list[int]] = {}
    for task in TASKS:
        rows, folds = build_manifest(task)
        if folds:
            small_class[task] = folds
        p = manifest_path(task)
        if args.apply:
            rows.to_csv(p, index=False)
        pt = rows.drop_duplicates("patient_id")
        flag = f"  SMALL-CLASS folds {folds}" if folds else ""
        print(f"  {task:16s} {len(rows):5d} slides {len(pt):5d} patients "
              f"pos={int(pt.target_label.sum()):4d}  -> {p.name}{flag}")
        if args.apply:
            d = split_dir(task)
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "splits.parquet").is_file():
                generate_splits(SplitConfig(
                    scheme="predefined_oof_kfold", name=paths.SPLIT_NAME, csv_path=str(p),
                    output_dir=str(d), filename_column="slide_id", label_column="target_label",
                    group_column="patient_id", fold_column="k_fold", n_folds=paths.N_FOLDS,
                    seed=paths.PRIMARY_SEED), force=False)
    if small_class:
        print(f"\n  Small-class rule will fire for {small_class} - early stopping disabled on")
        print(f"  those folds, fixed {FIXED_EPOCH_BUDGET}-epoch budget, final epoch deployed.")
    print("\nWrote." if args.apply else "\nDry run - pass --apply to write.")


def train_command(task: str, seed: int, num_workers: int | None = None) -> list[str]:
    """The frozen recipe for one (task, seed) chain, as a subprocess argv.

    One invocation trains all five outer folds and writes the pooled OOF, so a
    (task, seed) chain is the natural unit of work: it is self-contained, it
    writes only into its own train_dir, and the trainer resumes at FOLD level,
    so an interrupted chain restarts from the fold it was on rather than from
    scratch.
    """
    d = run_dir(task, seed)
    overrides = [
        "platform=colon_workstation", "data=aim1",
        f"data.aim1_model=e3a_{task}", f"data.manifest_stem=aim1_e3a_{task}",
        "+data.cohort_column=cohort", "encoder=univ1",
        "splits=aim1_balanced", f"splits.seed={seed}",
        "model=abmil", "model.embed_dim=512", "model.attn_dim=384",
        "model.input_dropout=0.10", f"model.dropout={paths.DROPOUT}",
        "training=aim1", f"training.lr={paths.LR:g}",
        f"training.weight_decay={paths.WEIGHT_DECAY:g}", f"training.seed={seed}",
        f"training.dataset_max_instances={E3A_CAP}", "training.eval_full_bags=true",
        f"training.fixed_epoch_budget={FIXED_EPOCH_BUDGET}",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        f"train_dir={d}", f"exp_name=e3a_{task}_seed{seed}",
    ]
    if num_workers is not None:
        overrides.append(f"training.num_workers={num_workers}")
    return [sys.executable, str(REPO / "tools" / "study_train.py"), "hydra-train", *overrides]


def pending_chains(tasks: list[str], seeds: list[int]) -> list[tuple[str, int]]:
    """(task, seed) chains with no pooled OOF yet, in a stable order."""
    return [
        (task, seed)
        for task in tasks
        for seed in seeds
        if not (run_dir(task, seed) / "oof_predictions.parquet").is_file()
    ]


def chain_log(task: str, seed: int) -> Path:
    """Per-chain log, deliberately OUTSIDE the train_dir.

    The trainer refuses to resume a train_dir that holds artifacts but no
    training_identity.json, so a log file written next to the checkpoints makes
    every chain unrunnable. Keeping logs in their own tree leaves the identity
    guard doing its job.
    """
    return E3A_ROOT / "logs" / f"{task}_seed{seed}.log"


def _run_chain(task: str, seed: int, num_workers: int | None) -> tuple[str, int, int]:
    """Run one chain with its output captured to its own log.

    Captured, not inherited: N concurrent Lightning progress bars interleaved on
    one terminal are unreadable, and the per-chain log is what makes a failure
    attributable to a chain afterwards.
    """
    log_path = chain_log(task, seed)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        code = subprocess.run(
            train_command(task, seed, num_workers), cwd=REPO,
            stdout=log, stderr=subprocess.STDOUT, check=False,
        ).returncode
    return (task, seed, code)


def cmd_train(args: argparse.Namespace) -> None:
    """Train the pending chains, sequentially or JOBS-wide.

    PARALLELISM. Chains are independent by construction — separate processes,
    separate train_dirs, pre-generated splits, no shared mutable state — so
    running them concurrently cannot change a result, only the wall clock. The
    fit is dataloader-bound (~20% GPU, ~3 cores and ~2.5 GB of VRAM per chain),
    which is exactly the profile that parallelises well: at --jobs 6 the sweep
    uses ~19 of 36 cores and ~15 GB of 24 GB VRAM.

    --jobs is a count of concurrent CHAINS. Each chain also spawns
    `training.num_workers` dataloader workers (4 on this platform), so 6 chains
    is ~30 OS processes; use --num-workers to trade that down if the budget is
    counted in processes rather than jobs.
    """
    stale = [old for old in RENAMES if (E3A_ROOT / "train" / old).is_dir()]
    if stale and not args.allow_stale_names:
        raise SystemExit(
            f"pre-rename run directories still present: {stale}. Run `migrate --apply` first, "
            "or these 30 completed fits will be recomputed. Pass --allow-stale-names to override."
        )
    tasks = [args.task] if args.task else list(TASKS)
    seeds = [args.seed] if args.seed else list(SEEDS)
    for task in tasks:
        if not manifest_path(task).is_file():
            raise SystemExit(f"{manifest_path(task)} missing - run `manifests --apply` first")
        if not (split_dir(task) / "splits.parquet").is_file():
            # Generating splits inside a parallel worker would race; they are
            # written once by `manifests --apply` and only read here.
            raise SystemExit(
                f"{split_dir(task)}/splits.parquet missing - run `manifests --apply` first"
            )

    chains = pending_chains(tasks, seeds)
    done = [(t, s) for t in tasks for s in seeds if (t, s) not in chains]
    for task, seed in done:
        print(f"== {task} seed{seed}: complete - skipping")
    if not chains:
        print("\nNothing to train.")
        return
    print(f"\n{len(chains)} chain(s) pending = {len(chains) * paths.N_FOLDS} fits, "
          f"{args.jobs}-wide")
    if args.dry_run:
        for task, seed in chains:
            print(f"== {task} seed{seed}\n   "
                  + " ".join(train_command(task, seed, args.num_workers)))
        return

    failures: list[tuple[str, int, int]] = []
    if args.jobs == 1:
        for task, seed in chains:
            print(f"== {task} seed{seed}", flush=True)
            code = subprocess.run(
                train_command(task, seed, args.num_workers), cwd=REPO, check=False
            ).returncode
            if code != 0:
                raise SystemExit(f"{task} seed{seed} failed (exit {code})")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(_run_chain, task, seed, args.num_workers): (task, seed)
                for task, seed in chains
            }
            print(f"   logs: {E3A_ROOT / 'logs'}/<task>_seed<seed>.log", flush=True)
            for n, future in enumerate(as_completed(futures), start=1):
                task, seed, code = future.result()
                status = "ok" if code == 0 else f"FAILED (exit {code})"
                print(f"[{n}/{len(chains)}] {task} seed{seed}: {status}", flush=True)
                if code != 0:
                    # One bad chain must not abort its siblings - they are
                    # independent, and the survivors are still reportable.
                    failures.append((task, seed, code))
    if failures:
        raise SystemExit(
            "failed chains: "
            + ", ".join(f"{t} seed{s} (exit {c})" for t, s, c in failures)
            + f" - see {E3A_ROOT / 'logs'}; rerun `train` to resume them at fold level"
        )
    print("\nAll chains complete.")


def _legacy_cmd_report_do_not_use(args: argparse.Namespace) -> None:
    """Historical in-place reporter retained only as an audit trail.

    It used probability-scale task metrics, a wholly unpaired arm bootstrap,
    incomplete-chain tolerance, and an overwrite-prone destination. The public
    ``report`` command below deliberately cannot call it.
    """
    report: dict[str, Any] = {
        "protocol": {
            "headline": "three-seed OOF logit ensemble (mean logit, sigmoid applied once)",
            "delta": "unpaired patient bootstrap of AUROC(control) - AUROC(fine)",
            "n_bootstrap": paths.N_BOOTSTRAP,
            "ceiling_bound": CEILING_BOUND,
            "cap": E3A_CAP,
        },
        "tasks": {},
        "verdict": {},
    }
    ensembles: dict[str, pd.DataFrame] = {}

    for task in ("gene", *TASKS):
        ens, frames = gene_ensemble() if task == "gene" else ensemble_oof(task)
        if ens is None:
            continue
        ensembles[task] = ens
        boot = evaluate.bootstrap_auroc(ens, n_bootstrap=args.n_bootstrap)
        per_seed = {
            s: evaluate.bootstrap_auroc(f, n_bootstrap=args.n_bootstrap)
            for s, f in frames.items()
        }
        report["tasks"][task] = {
            "kind": "reference" if task == "gene" else TASKS[task][1],
            "n": boot["n"],
            "n_positive": boot["n_positive"],
            "n_seeds": len(frames),
            "ensemble": {"auroc": boot["auroc"], "ci": [boot["ci_low"], boot["ci_high"]]},
            "per_seed": {
                str(s): {"auroc": b["auroc"], "ci": [b["ci_low"], b["ci_high"]]}
                for s, b in per_seed.items()
            },
            "seed_range": [
                float(min(b["auroc"] for b in per_seed.values())),
                float(max(b["auroc"] for b in per_seed.values())),
            ],
        }

    print(f"\n{'=' * 100}\nE3a - MOLECULAR-RESOLUTION LADDER")
    print("headline = 3-seed OOF LOGIT ensemble; per-seed min-max is the robustness panel")
    print("=" * 100)
    print(f"  {'task':16s} {'kind':8s} {'n':>5s} {'pos':>5s} {'sd':>3s} "
          f"{'ensemble AUROC [95% CI]':>26s} {'seed min-max':>17s}")
    for task, block in report["tasks"].items():
        e = block["ensemble"]
        lo, hi = block["seed_range"]
        cell = "{:.4f} [{:.4f}, {:.4f}]".format(e["auroc"], e["ci"][0], e["ci"][1])
        span = f"{lo:.4f}-{hi:.4f}"
        print(f"  {task:16s} {block['kind']:8s} {block['n']:5d} {block['n_positive']:5d} "
              f"{block['n_seeds']:3d} {cell:>26s} {span:>17s}")

    print(f"\n{'=' * 100}\nRESOLUTION LADDER - fine vs matched control, with the direct contrast")
    print("=" * 100)
    print(f"  {'rung':30s} {'fine [CI]':>24s} {'control [CI]':>24s} "
          f"{'delta ctrl-fine [CI]':>26s}")
    for fine, ctrl in PAIRS:
        if fine not in report["tasks"] or ctrl not in report["tasks"]:
            print(f"  {RUNG_LABEL[fine]:30s} {'pending':>24s}")
            continue
        f_block, c_block = report["tasks"][fine], report["tasks"][ctrl]
        delta = delta_bootstrap(
            ensembles[fine], ensembles[ctrl], n_bootstrap=args.n_bootstrap
        )
        code, prose = verdict_for(f_block["ensemble"], c_block["ensemble"], delta)
        fe, ce = f_block["ensemble"], c_block["ensemble"]
        fine_cell = "{:.3f} [{:.3f},{:.3f}]".format(fe["auroc"], fe["ci"][0], fe["ci"][1])
        ctrl_cell = "{:.3f} [{:.3f},{:.3f}]".format(ce["auroc"], ce["ci"][0], ce["ci"][1])
        delta_cell = "{:+.3f} [{:+.3f},{:+.3f}]".format(
            delta["delta"], delta["ci"][0], delta["ci"][1]
        )
        print(f"  {RUNG_LABEL[fine]:30s} {fine_cell:>24s} {ctrl_cell:>24s} {delta_cell:>26s}")
        print(f"  {'':30s} -> {code}: {prose}")
        report["verdict"][fine] = {
            "control": ctrl,
            "fine": fe,
            "control_auroc": ce,
            "delta": delta,
            "conditions": {
                "fine_upper_ci_lt_0.60": bool(fe["ci"][1] < CEILING_BOUND),
                "control_lower_ci_gt_0.50": bool(ce["ci"][0] > CHANCE),
                "delta_lower_ci_gt_0": bool(delta["ci"][0] > 0.0),
                "fine_lower_ci_gt_0.50": bool(fe["ci"][0] > CHANCE),
            },
            "verdict": code,
            "prose": prose,
        }

    # G12C, descriptive only, from the locked generic KRAS scores.
    pat = population_table()
    sv = pat["kras_subvariant"].fillna("UNKNOWN")
    g12c = set(pat.loc[sv.str.contains("G12C", na=False), "patient_id"])
    root = paths.OUTPUT_ROOT / f"train/1a_pb_cap{E3A_CAP}/univ1"
    scores = []
    for seed in SEEDS:
        p = root / f"seed{seed}" / "oof_predictions.parquet"
        if p.is_file():
            f = evaluate.to_patient_level(pd.read_parquet(p), pd.read_csv(paths.DEV_MANIFEST))
            scores.append(f.set_index("patient_id")["mean_logit"])
    if scores:
        s = sigmoid(pd.concat(scores, axis=1).mean(axis=1).to_numpy())
        s = pd.Series(s, index=scores[0].index)
        ing, outg = s[s.index.isin(g12c)], s[~s.index.isin(g12c)]
        wt = s[s.index.isin(set(pat.loc[pat.kras.eq("wild_type"), "patient_id"]))]
        print(f"\n  G12C (n={len(ing)}) DESCRIPTIVE from the locked generic KRAS 3-seed ensemble:")
        print(f"    mean P(KRAS) G12C {ing.mean():.4f} | all other patients {outg.mean():.4f} "
              f"| wild-type {wt.mean():.4f}")
        report["g12c_descriptive"] = {"n": int(len(ing)), "mean_score": float(ing.mean()),
                                      "mean_other": float(outg.mean()), "mean_wt": float(wt.mean())}

    missing = [t for t in TASKS if t not in report["tasks"]]
    if "gene" not in report["tasks"]:
        print("\n  gene rung (E0) not found - the ladder has no reference row")
    if missing:
        print(f"\n  NOT YET TRAINED: {missing}")
    dest = paths.EVAL_ROOT / "e3a_resolution.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {dest}")


def cmd_report(args: argparse.Namespace) -> None:
    """Run the fail-closed, native-logit corrected analysis in a new root."""
    if not args.output_root:
        raise SystemExit(
            "Legacy in-place E3 reporting is disabled: it can recreate the known-invalid "
            "unpaired/probability-roundtrip result. Re-run with --output-root pointing to "
            "a NEW directory; the corrected analyzer refuses an existing destination."
        )
    if int(args.n_bootstrap) < 10_000:
        raise SystemExit("Corrected E3 reporting requires at least 10,000 bootstrap draws")
    from aim3_fixed_control_analysis import cmd_run as corrected_run

    corrected_run(
        argparse.Namespace(
            e3_root=str(E3A_ROOT),
            e0_root=str(paths.OUTPUT_ROOT / f"train/1a_pb_cap{E3A_CAP}/univ1"),
            output_root=str(Path(args.output_root).resolve()),
            n_bootstrap=int(args.n_bootstrap),
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("plan")
    x.set_defaults(func=cmd_plan)
    x = sub.add_parser("migrate")
    x.add_argument("--apply", action="store_true")
    x.set_defaults(func=cmd_migrate)
    x = sub.add_parser("manifests")
    x.add_argument("--apply", action="store_true")
    x.set_defaults(func=cmd_manifests)
    x = sub.add_parser("train")
    x.add_argument("--task", choices=list(TASKS))
    x.add_argument("--seed", type=int)
    x.add_argument("--dry-run", action="store_true")
    x.add_argument("--allow-stale-names", action="store_true")
    x.add_argument("--jobs", type=int, default=1,
                   help="concurrent (task, seed) chains; each also spawns "
                        "training.num_workers dataloader workers")
    x.add_argument("--num-workers", type=int,
                   help="override training.num_workers per chain (platform default 4)")
    x.set_defaults(func=cmd_train)
    x = sub.add_parser("report")
    x.add_argument("--n-bootstrap", type=int, default=10_000)
    x.add_argument(
        "--output-root",
        help="required NEW immutable result directory for aim3_fixed_control_analysis.py",
    )
    x.set_defaults(func=cmd_report)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
