#!/usr/bin/env python3
"""E2e - metastatic in-domain training upper bound (Aim 2, final-v4 addition).

THE QUESTION. E2b showed that the four frozen primary-trained models do not
clear the fixed metastatic transport gate, and E2c showed that a sparse
residual linear adapter does not establish repair. Both results leave one
question open: is the KRAS signal ABSENT from metastatic H&E, or PRESENT but
not reachable by a primary-trained representation plus a linear offset?

E2e answers the only version of that question the frozen data can support: it
trains the locked recipe FROM SCRATCH on metastatic slides only, under the same
five-fold patient-grouped OOF protocol as E0, and asks whether an in-domain
model can rank KRAS in metastases at all.

    local OOF AUROC ~ 0.5   ->  no learnable metastatic KRAS signal at this
                                sample size: the transport failure is not a
                                repairable domain shift on this evidence
    local OOF AUROC >> 0.5  ->  signal exists in metastatic tissue; the E2b
                                failure is a representation/domain-shift
                                problem, and adaptation research is warranted

POPULATION. The union of the two frozen E2b metastatic target manifests:
RIH-M (85 patients / 85 slides) and SR1482-M (74 patients / 100 slides);
159 patients, 185 slides, 67 KRAS-mutant. No manifest row is added, removed,
or relabeled; only fold-assignment columns are appended.

DESIGN. Five patient-grouped outer folds stratified by cohort x KRAS, one
frozen layout shared by three model seeds (42/43/44): 15 fits. Inner
early-stopping carve-outs are stratified the same way and sized so every fold
retains >= 8 mutant validation patients (the es_min_val_positives=8 contract).
The recipe is byte-for-byte the E2a source-CV recipe: UNIv1 packed features,
gated ABMIL 512/384, cap 8,192 training bags, full-bag inference,
patient_natural sampling, AdamW 1e-4/1e-5, patient-AUROC checkpoint selection.

POWER, STATED UP FRONT. 159 patients bound the achievable precision
(single-cohort CIs roughly +/-0.1). E2e is an upper-bound/diagnostic
experiment, not a deployment claim: a positive result licenses adaptation
research, a null result is reported as "not learnable AT THIS SAMPLE SIZE".

WHAT E2E MUST NEVER DO. Its scores are OOF within the metastatic population
and are never mixed into E2b's transported scores, the deployment matrix's
transported rows, or any primary-tumour table. The paired local-vs-transported
contrast uses identical patients and shared resamples, per cohort, and is the
only place the two score sets meet.

Usage:
    python aim2_metastatic_indomain_bound.py build              # manifest + frozen splits + receipt
    python aim2_metastatic_indomain_bound.py train --seed 42    # one seed's five folds (one process)
    python aim2_metastatic_indomain_bound.py report             # OOF analysis + paired E2b contrast
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import evaluate, paths  # noqa: E402

# ── frozen identities ────────────────────────────────────────────────────────
ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim2_e2e_met_local_v1_20260820")
E2A_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim2_cap8192_v4_20260819/e2a")
MET_SOURCES = {
    "RIH": paths.MANIFEST_DIR / "aim1_e2a_rih_metastatic.csv",
    "SurGen": paths.MANIFEST_DIR / "aim1_e2a_surgen_metastatic.csv",
}
MANIFEST = paths.MANIFEST_DIR / "aim1_e2e_met.csv"
DATA_NAME = "e2e_met"  # data.name resolves to aim1_e2e_met
SPLIT_ROOT = ROOT / "inputs" / "splits"
SPLIT_DIR = SPLIT_ROOT / f"aim1_{DATA_NAME}" / paths.SPLIT_NAME
SEEDS = (42, 43, 44)
CAP = 8192
N_FOLDS = 5
FOLD_SEED = 42  # deterministic outer-fold deal
CARVE_SEED_BASE = 20260820  # + fold index -> inner-carve RNG
CARVE_FRACTION = 0.20
MIN_VAL_POSITIVES = 9  # one above the es_min_val_positives=8 contract
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260817  # Aim 2 inference convention
# E2b frozen ensemble points (Results.md section 3.2); build-time exactness check.
E2B_EXPECTED = {"RIH": 0.606982, "SurGen": 0.562121}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"append-only: {path} already exists; choose a new lineage")


def train_dir(seed: int) -> Path:
    return ROOT / "train" / f"e2e_met_cap{CAP}_seed{seed}"


# ── build ────────────────────────────────────────────────────────────────────
def _deal_outer_folds(patients: pd.DataFrame) -> pd.Series:
    """Deterministic stratified deal of patients into N_FOLDS outer folds.

    Strata are cohort x KRAS. Within each stratum patients are ordered by id,
    shuffled once with the frozen seed, and dealt round-robin; the starting
    fold rotates with the cumulative dealt count so the four strata do not pile
    their remainders onto fold 0.
    """
    rng = np.random.default_rng(FOLD_SEED)
    fold = pd.Series(index=patients.index, dtype=int)
    dealt = 0
    for _, block in patients.groupby(["cohort", "kras"], sort=True):
        order = block.sort_values("patient_id").index.to_numpy()
        rng.shuffle(order)
        for position, idx in enumerate(order):
            fold[idx] = (dealt + position) % N_FOLDS
        dealt += len(order)
    return fold


def _carve_val(patients: pd.DataFrame, fold: pd.Series, fold_idx: int) -> pd.Series:
    """Stratified early-stopping carve-out from fold ``fold_idx``'s training pool."""
    rng = np.random.default_rng(CARVE_SEED_BASE + fold_idx)
    flags = pd.Series(0, index=patients.index, dtype=int)
    pool = patients[fold != fold_idx]
    for _, block in pool.groupby(["cohort", "kras"], sort=True):
        order = block.sort_values("patient_id").index.to_numpy()
        rng.shuffle(order)
        n_take = int(np.ceil(CARVE_FRACTION * len(order)))
        flags[order[:n_take]] = 1
    n_val_pos = int((flags == 1)[patients["kras"] == "mutant"].sum())
    if n_val_pos < MIN_VAL_POSITIVES:
        raise AssertionError(
            f"fold {fold_idx}: only {n_val_pos} mutant validation patients "
            f"(need >= {MIN_VAL_POSITIVES})"
        )
    return flags


def _check_e2b_reproduction(manifest: pd.DataFrame) -> dict[str, float]:
    """Reproduce the sealed E2b ensemble AUROCs from the frozen slide scores.

    This validates the label join and the patient aggregation used later by
    ``report`` against two published six-decimal values before any training.
    """
    from sklearn.metrics import roc_auc_score

    out: dict[str, float] = {}
    for cohort, target in (("RIH", "rih"), ("SurGen", "surgen")):
        block = manifest[manifest["cohort"] == cohort]
        per_seed = []
        for seed in SEEDS:
            scores = pd.read_parquet(
                E2A_ROOT / "scores" / f"pb_cap{CAP}_{target}_seed{seed}_metastatic.parquet"
            )
            merged = scores.merge(
                block[["slide_id", "patient_id", "target_label"]], on="slide_id", validate="one_to_one"
            )
            if len(merged) != len(block):
                raise AssertionError(f"{cohort} seed{seed}: slide coverage mismatch")
            per_seed.append(merged.groupby("patient_id").agg(
                label=("target_label", "max"), mean_logit=("logit", "mean")
            ))
        ensemble = per_seed[0][["label"]].copy()
        ensemble["eta"] = np.mean([p["mean_logit"] for p in per_seed], axis=0)
        auroc = float(roc_auc_score(ensemble["label"], ensemble["eta"]))
        if abs(auroc - E2B_EXPECTED[cohort]) > 5e-7:
            raise AssertionError(
                f"E2b reproduction failed for {cohort}: {auroc:.6f} != {E2B_EXPECTED[cohort]}"
            )
        out[cohort] = auroc
    return out


def cmd_build(_: argparse.Namespace) -> None:
    ensure_absent(MANIFEST)
    ensure_absent(SPLIT_DIR / "splits.parquet")
    source_hashes = {c: sha256(p) for c, p in MET_SOURCES.items()}
    frames = [pd.read_csv(p) for p in MET_SOURCES.values()]
    manifest = pd.concat(frames, ignore_index=True)
    if manifest["slide_id"].duplicated().any():
        raise AssertionError("duplicate slide_id across metastatic manifests")

    patients = (
        manifest.groupby("patient_id")
        .agg(cohort=("cohort", "first"), kras=("kras", "first"), label=("target_label", "max"))
        .reset_index()
    )
    if len(patients) != 159 or int(patients["label"].sum()) != 67 or len(manifest) != 185:
        raise AssertionError(
            f"population drift: {len(patients)} patients / {int(patients['label'].sum())} "
            f"mutant / {len(manifest)} slides (expected 159/67/185)"
        )

    e2b_check = _check_e2b_reproduction(manifest)
    print(f"E2b ensemble reproduction PASS: {e2b_check}")

    fold = _deal_outer_folds(patients)
    sizes = fold.value_counts().sort_index()
    mutants = patients.groupby(fold)["label"].sum()
    print(f"fold sizes {sizes.tolist()}, mutants/fold {mutants.tolist()}")
    if int(sizes.max() - sizes.min()) > 2:
        raise AssertionError("outer folds unbalanced by more than 2 patients")

    layout = patients[["patient_id"]].copy()
    layout["k_fold"] = fold.to_numpy()
    for fold_idx in range(N_FOLDS):
        layout[f"val_fold_{fold_idx}"] = _carve_val(patients, fold, fold_idx).to_numpy()

    manifest = manifest.merge(layout, on="patient_id", validate="many_to_one")
    manifest.to_csv(MANIFEST, index=False)
    print(f"wrote {MANIFEST} ({len(manifest)} slides)")

    from oceanpath.splitting import SplitConfig, generate_splits

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    generate_splits(
        SplitConfig(
            scheme="predefined_oof_kfold",
            name=paths.SPLIT_NAME,
            csv_path=str(MANIFEST),
            output_dir=str(SPLIT_DIR),
            filename_column="slide_id",
            label_column="target_label",
            group_column="patient_id",
            fold_column="k_fold",
            n_folds=N_FOLDS,
            seed=paths.PRIMARY_SEED,
        ),
        force=False,
    )

    receipt = {
        "experiment": "e2e_metastatic_local_upper_bound",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifests_sha256": source_hashes,
        "manifest_sha256": sha256(MANIFEST),
        "splits_sha256": sha256(SPLIT_DIR / "splits.parquet"),
        "population": {"patients": 159, "mutant": 67, "slides": 185},
        "fold_seed": FOLD_SEED,
        "carve_seed_base": CARVE_SEED_BASE,
        "carve_fraction": CARVE_FRACTION,
        "e2b_ensemble_reproduction": e2b_check,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    receipt_path = ROOT / "build_receipt.json"
    ensure_absent(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"wrote {receipt_path}")


# ── train ────────────────────────────────────────────────────────────────────
def cmd_train(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed must be one of {SEEDS}")
    directory = train_dir(seed)
    if directory.exists():
        from oceanpath.workflows.training import validate_training_run_dir

        validate_training_run_dir(directory, require_test_predictions=True)
        print(f"== e2e seed{seed}: complete and valid — skipping")
        return
    overrides = [
        "platform=colon_workstation",
        "data=aim1",
        f"data.aim1_model={DATA_NAME}",
        "data.manifest_stem=aim1_e2e_met",
        f"data.csv_path={MANIFEST}",
        f"platform.splits_root={SPLIT_ROOT}",
        "+data.cohort_column=cohort",
        "encoder=univ1",
        "splits=aim1_balanced",
        f"splits.seed={seed}",
        "model=abmil",
        "model.embed_dim=512",
        "model.attn_dim=384",
        "model.input_dropout=0.10",
        f"model.dropout={paths.DROPOUT}",
        "training=aim1",
        f"training.lr={paths.LR:g}",
        f"training.weight_decay={paths.WEIGHT_DECAY:g}",
        f"training.seed={seed}",
        f"training.dataset_max_instances={CAP}",
        "training.eval_full_bags=true",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        "training.skip_finalize=true",
        f"train_dir={directory}",
        f"exp_name=e2e_met_c{CAP}_s{seed}",
        f"hydra.run.dir={ROOT / 'hydra_runs' / f'e2e_met_cap{CAP}_seed{seed}'}",
        "hydra.job.chdir=false",
    ]
    cmd = [sys.executable, str(REPO / "tools" / "study_train.py"), "hydra-train", *overrides]
    log_path = ROOT / "launcher_logs" / f"e2e_met_cap{CAP}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"== e2e met-local seed{seed} cap{CAP}\n   log: {log_path}")
    if args.dry_run:
        print("   " + " ".join(cmd))
        return
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f"e2e seed{seed} failed (rc={rc}); see {log_path}")
    from oceanpath.workflows.training import validate_training_run_dir

    validate_training_run_dir(directory, require_test_predictions=True)
    print(f"== e2e seed{seed}: complete")


# ── report ───────────────────────────────────────────────────────────────────
def _local_patients(manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    oof = pd.read_parquet(train_dir(seed) / "oof_predictions.parquet")
    if len(oof) != len(manifest):
        raise AssertionError(f"seed{seed}: OOF covers {len(oof)} slides, expected {len(manifest)}")
    pat = evaluate.to_patient_level(oof, manifest)
    labels = manifest.groupby("patient_id")["target_label"].max()
    if not pat.set_index("patient_id")["label"].eq(labels).all():
        raise AssertionError(f"seed{seed}: OOF labels disagree with manifest")
    return pat


def _transported_patients(manifest: pd.DataFrame) -> pd.DataFrame:
    """Frozen E2b three-seed ensemble patient logits on the same population."""
    per_seed = []
    for seed in SEEDS:
        parts = []
        for cohort, target in (("RIH", "rih"), ("SurGen", "surgen")):
            block = manifest[manifest["cohort"] == cohort]
            scores = pd.read_parquet(
                E2A_ROOT / "scores" / f"pb_cap{CAP}_{target}_seed{seed}_metastatic.parquet"
            )
            merged = scores.merge(
                block[["slide_id", "patient_id", "target_label", "cohort"]],
                on="slide_id",
                validate="one_to_one",
            )
            parts.append(merged)
        both = pd.concat(parts, ignore_index=True)
        per_seed.append(
            both.groupby("patient_id").agg(
                label=("target_label", "max"),
                cohort=("cohort", "first"),
                mean_logit=("logit", "mean"),
            )
        )
    ensemble = per_seed[0][["label", "cohort"]].copy()
    ensemble["eta_transported"] = np.mean([p["mean_logit"] for p in per_seed], axis=0)
    return ensemble.reset_index()


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def _bootstrap_cohort_macro(
    frame: pd.DataFrame, columns: tuple[str, str]
) -> dict[str, list[float] | float]:
    """Per-cohort and equal-cohort paired bootstrap for local vs transported.

    Patients are resampled WITHIN cohort; both score columns are evaluated on
    the same resample, so per-cohort deltas and the equal-cohort macro delta
    inherit the pairing. 10,000 draws, seed 20260817 (Aim 2 convention).
    """
    local_col, transported_col = columns
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cohorts = sorted(frame["cohort"].unique())
    blocks = {
        c: frame[frame["cohort"] == c].reset_index(drop=True) for c in cohorts
    }
    macro_local, macro_transported, macro_delta = [], [], []
    per_cohort: dict[str, dict[str, list[float]]] = {
        c: {"local": [], "transported": [], "delta": []} for c in cohorts
    }
    for _ in range(BOOTSTRAP_DRAWS):
        locals_, transporteds = [], []
        valid = True
        draw: dict[str, tuple[float, float]] = {}
        for c in cohorts:
            block = blocks[c]
            index = rng.integers(0, len(block), len(block))
            y = block["label"].to_numpy()[index]
            if len(np.unique(y)) < 2:
                valid = False
                break
            a_local = _auroc(y, block[local_col].to_numpy()[index])
            a_trans = _auroc(y, block[transported_col].to_numpy()[index])
            draw[c] = (a_local, a_trans)
            locals_.append(a_local)
            transporteds.append(a_trans)
        if not valid:
            continue
        for c, (a_local, a_trans) in draw.items():
            per_cohort[c]["local"].append(a_local)
            per_cohort[c]["transported"].append(a_trans)
            per_cohort[c]["delta"].append(a_local - a_trans)
        macro_local.append(float(np.mean(locals_)))
        macro_transported.append(float(np.mean(transporteds)))
        macro_delta.append(float(np.mean(locals_) - np.mean(transporteds)))

    def ci(vals: list[float]) -> list[float]:
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    out: dict = {"n_valid_draws": len(macro_delta), "per_cohort": {}, "macro": {}}
    for c in cohorts:
        block = blocks[c]
        y = block["label"].to_numpy()
        out["per_cohort"][c] = {
            "n": int(len(block)),
            "n_mutant": int(y.sum()),
            "local_auroc": _auroc(y, block[local_col].to_numpy()),
            "local_ci": ci(per_cohort[c]["local"]),
            "transported_auroc": _auroc(y, block[transported_col].to_numpy()),
            "transported_ci": ci(per_cohort[c]["transported"]),
            "delta_local_minus_transported": _auroc(y, block[local_col].to_numpy())
            - _auroc(y, block[transported_col].to_numpy()),
            "delta_ci": ci(per_cohort[c]["delta"]),
        }
    point_macro_local = float(
        np.mean([out["per_cohort"][c]["local_auroc"] for c in cohorts])
    )
    point_macro_trans = float(
        np.mean([out["per_cohort"][c]["transported_auroc"] for c in cohorts])
    )
    out["macro"] = {
        "local_auroc": point_macro_local,
        "local_ci": ci(macro_local),
        "transported_auroc": point_macro_trans,
        "transported_ci": ci(macro_transported),
        "delta_local_minus_transported": point_macro_local - point_macro_trans,
        "delta_ci": ci(macro_delta),
        "fixed_gate_local": {
            "both_points_above_0p5": all(
                out["per_cohort"][c]["local_auroc"] > 0.5 for c in cohorts
            ),
            "macro_ci_lower_above_0p5": ci(macro_local)[0] > 0.5,
        },
    }
    return out


def _hanley_mcneil_n(auc: float, prevalence: float, alpha: float, power: float) -> int:
    """Patients needed for one-sided H0: AUROC=0.5 at the given design point."""
    from scipy.stats import norm

    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    z_a, z_b = norm.ppf(1 - alpha), norm.ppf(power)
    for n in range(20, 20_000):
        n_pos = max(1, round(n * prevalence))
        n_neg = max(1, n - n_pos)
        var = (
            auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)
        ) / (n_pos * n_neg)
        # Null variance at AUROC 0.5: q1 = q2 = 1/3, so each q - auc^2 term is 1/12.
        var0 = (0.25 + (n_pos - 1 + n_neg - 1) / 12) / (n_pos * n_neg)
        if z_a * np.sqrt(var0) + z_b * np.sqrt(var) <= auc - 0.5:
            return n
    return -1


def cmd_report(_: argparse.Namespace) -> None:
    # analysis/ (v1) is preserved but superseded: its Hanley-McNeil design note
    # used a wrong null-variance constant. Every AUROC/bootstrap value is
    # unchanged in v2; only the power-design numbers differ.
    out_path = ROOT / "analysis_v2" / "results.json"
    ensure_absent(out_path)
    manifest = pd.read_csv(MANIFEST)
    from oceanpath.workflows.training import validate_training_run_dir

    for seed in SEEDS:
        validate_training_run_dir(train_dir(seed), require_test_predictions=True)

    per_seed_frames = {seed: _local_patients(manifest, seed) for seed in SEEDS}
    transported = _transported_patients(manifest)

    per_seed = {}
    for seed, pat in per_seed_frames.items():
        y = pat["label"].to_numpy()
        entry = {
            "pooled_auroc": _auroc(y, pat["mean_logit"].to_numpy()),
            "per_cohort": {
                str(c): _auroc(
                    block["label"].to_numpy(), block["mean_logit"].to_numpy()
                )
                for c, block in pat.groupby("cohort")
            },
        }
        per_seed[str(seed)] = entry

    base = per_seed_frames[SEEDS[0]][["patient_id", "label", "cohort"]].copy()
    stacked = np.stack(
        [
            per_seed_frames[s].set_index("patient_id").loc[base["patient_id"], "mean_logit"]
            for s in SEEDS
        ]
    )
    base["eta_local"] = stacked.mean(axis=0)
    merged = base.merge(
        transported[["patient_id", "eta_transported"]], on="patient_id", validate="one_to_one"
    )
    if len(merged) != 159:
        raise AssertionError(f"paired frame has {len(merged)} patients, expected 159")

    contrast = _bootstrap_cohort_macro(merged, ("eta_local", "eta_transported"))

    pooled_seed_aurocs = [per_seed[str(s)]["pooled_auroc"] for s in SEEDS]
    y_all = merged["label"].to_numpy()
    design = {
        "observed_prevalence": float(y_all.mean()),
        "n_for_auroc_0p60_alpha05_power80": _hanley_mcneil_n(0.60, float(y_all.mean()), 0.05, 0.80),
        "n_for_auroc_0p65_alpha05_power80": _hanley_mcneil_n(0.65, float(y_all.mean()), 0.05, 0.80),
        "note": (
            "patients needed for a one-sided test of AUROC>0.5 at the stated design "
            "point, Hanley-McNeil variance, single cohort; the fixed E2b gate would "
            "additionally require the macro lower bound above 0.50"
        ),
    }

    results = {
        "experiment": "e2e_metastatic_local_upper_bound",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "population": {
            "patients": int(len(merged)),
            "mutant": int(merged["label"].sum()),
            "per_cohort": {
                c: {"n": int(len(b)), "mutant": int(b["label"].sum())}
                for c, b in merged.groupby("cohort")
            },
        },
        "protocol": {
            "folds": N_FOLDS,
            "seeds": list(SEEDS),
            "cap": CAP,
            "recipe": "identical to E2a source-CV (patient_natural, patient-AUROC selection)",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "local_per_seed": per_seed,
        "local_seed_median_pooled_auroc": float(np.median(pooled_seed_aurocs)),
        "local_vs_transported_paired": contrast,
        "design_power_note": design,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(MANIFEST),
        "splits_sha256": sha256(SPLIT_DIR / "splits.parquet"),
        "oof_sha256": {str(s): sha256(train_dir(s) / "oof_predictions.parquet") for s in SEEDS},
        "e2b_score_sha256": {
            f"{t}_seed{s}": sha256(
                E2A_ROOT / "scores" / f"pb_cap{CAP}_{t}_seed{s}_metastatic.parquet"
            )
            for t in ("rih", "surgen")
            for s in SEEDS
        },
        "results_sha256": sha256(out_path),
    }
    receipt["supersedes"] = {
        "path": str(ROOT / "analysis"),
        "reason": (
            "v1 design_power_note used a wrong Hanley-McNeil null-variance "
            "constant; all AUROC and bootstrap values are identical in v2"
        ),
    }
    receipt_path = ROOT / "analysis_v2" / "receipt.json"
    ensure_absent(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(results["local_vs_transported_paired"]["macro"], indent=2))
    print(f"wrote {out_path}\nwrote {receipt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build").set_defaults(func=cmd_build)
    p_train = sub.add_parser("train")
    p_train.add_argument("--seed", required=True, type=int)
    p_train.add_argument("--dry-run", action="store_true")
    p_train.set_defaults(func=cmd_train)
    sub.add_parser("report").set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
