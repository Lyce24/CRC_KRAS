#!/usr/bin/env python3
"""E1e - MSI/dMMR positive-control task (executing the predeclared branch).

WHY. Three of the study's central results are nulls or modest numbers whose
interpretation leans on the pipeline being able to learn when signal exists:
the 0.66 KRAS headline, the Aim 3 ceilings, and the E2e metastatic null. The
field-standard positive control is MSI/dMMR status, which is known to be
strongly predictable from CRC H&E. This experiment runs the UNCHANGED locked
recipe on the frozen, previously built E1e MSI manifest:

    aim1_e1e_msi.csv — 1,433 development patients with observed MSI status
    (171 MSI/dMMR, 1,262 MSS/pMMR; 1,586 slides), carrying the frozen
    k_fold / val_fold_0..4 layout filtered from the balanced design. Every
    inner-validation carve-out holds 19–21 positive patients, so standard
    patient-AUROC selection applies in all folds.

Everything else is byte-identical to the study arm: UNIv1 packed features,
ABMIL 512/384, cap 8,192, patient_natural sampling, AdamW 1e-4/1e-5, three
seeds by five folds. A high MSI AUROC establishes that the modest KRAS
number and the metastatic null are properties of those targets, not of the
pipeline; the MSI score itself is a methods-level control and is never
proposed as an MSI assay.

Usage:
    python aim1_msi_positive_control.py train --seed 42
    python aim1_msi_positive_control.py report
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

SEEDS = (42, 43, 44)
CAP = 8192
MANIFEST = paths.MANIFEST_DIR / "aim1_e1e_msi.csv"
ROOT = paths.OUTPUT_ROOT / "e1e" / f"msi_cap{CAP}" / "univ1"
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20260817
EXPECTED_CENSUS = {"patients": 1433, "positive": 171, "slides": 1586}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_dir(seed: int) -> Path:
    return ROOT / f"seed{seed}"


def _check_census() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    patients = manifest.drop_duplicates("patient_id")
    observed = {
        "patients": int(len(patients)),
        "positive": int(patients["target_label"].sum()),
        "slides": int(len(manifest)),
    }
    if observed != EXPECTED_CENSUS:
        raise AssertionError(f"E1e MSI census drift: {observed} != {EXPECTED_CENSUS}")
    for fold in range(paths.N_FOLDS):
        val_positive = int(
            patients.loc[patients[f"val_fold_{fold}"] == 1, "target_label"].sum()
        )
        if val_positive < 8:
            raise AssertionError(f"fold {fold}: only {val_positive} positive val patients")
    return manifest


def train_command(seed: int) -> list[str]:
    d = train_dir(seed)
    overrides = [
        "platform=colon_workstation",
        "data=aim1",
        "data.aim1_model=e1e_msi",
        "data.manifest_stem=aim1_e1e_msi",
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
        f"train_dir={d}",
        f"exp_name=e1e_msi_cap{CAP}_seed{seed}",
        f"hydra.run.dir={ROOT / 'hydra_runs' / f'seed{seed}'}",
        "hydra.job.chdir=false",
    ]
    return [sys.executable, str(REPO / "tools" / "study_train.py"), "hydra-train", *overrides]


def cmd_train(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed must be one of {SEEDS}")
    _check_census()
    d = train_dir(seed)
    from oceanpath.workflows.training import validate_training_run_dir

    if (d / "oof_predictions.parquet").is_file():
        validate_training_run_dir(d, require_test_predictions=True)
        print(f"== e1e msi seed{seed}: complete and valid — skipping")
        return
    cmd = train_command(seed)
    log_path = ROOT / "launcher_logs" / f"seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"== e1e msi cap{CAP} seed{seed}\n   log: {log_path}")
    if args.dry_run:
        print("   " + " ".join(cmd))
        return
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO).returncode
    if rc != 0:
        raise SystemExit(f"e1e msi seed{seed} failed (rc={rc}); see {log_path}")
    validate_training_run_dir(d, require_test_predictions=True)
    print(f"== e1e msi seed{seed}: complete")


def cmd_report(_: argparse.Namespace) -> None:
    out_path = ROOT / "analysis" / "results.json"
    if out_path.exists():
        raise SystemExit(f"append-only: {out_path} exists")
    from oceanpath.workflows.training import validate_training_run_dir

    manifest = _check_census()
    frames: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        validate_training_run_dir(train_dir(seed), require_test_predictions=True)
        frames[seed] = evaluate.to_patient_level(
            pd.read_parquet(train_dir(seed) / "oof_predictions.parquet"), manifest
        )

    per_seed = {}
    for seed, pat in frames.items():
        per_seed[str(seed)] = {
            "pooled_auroc": evaluate.patient_auroc(pat, "mean_logit"),
            "auprc": float(
                __import__("sklearn.metrics", fromlist=["average_precision_score"])
                .average_precision_score(pat["label"], pat["mean_logit"])
            ),
            "per_cohort_auroc": {
                str(c): evaluate.patient_auroc(block, "mean_logit")
                for c, block in pat.groupby("cohort")
                if block["label"].nunique() > 1
            },
        }

    base = frames[SEEDS[0]][["patient_id", "label", "cohort"]].copy()
    stacked = np.stack(
        [
            frames[s].set_index("patient_id").loc[base["patient_id"], "mean_logit"]
            for s in SEEDS
        ]
    )
    base["mean_logit"] = stacked.mean(axis=0)
    base["prob_raw"] = 1.0 / (1.0 + np.exp(-base["mean_logit"]))
    ensemble = evaluate.bootstrap_auroc(
        base, "mean_logit", n_bootstrap=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )

    results = {
        "experiment": "e1e_msi_positive_control_cap8192",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "census": EXPECTED_CENSUS,
        "prevalence": round(EXPECTED_CENSUS["positive"] / EXPECTED_CENSUS["patients"], 4),
        "per_seed": per_seed,
        "median_pooled_auroc": float(
            np.median([per_seed[str(s)]["pooled_auroc"] for s in SEEDS])
        ),
        "three_seed_ensemble": ensemble,
        "role": (
            "pipeline positive control only; never proposed as an MSI assay and "
            "never mixed into any KRAS table"
        ),
        "conventions": {"bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    receipt = {
        "created_utc": results["created_utc"],
        "manifest_sha256": sha256(MANIFEST),
        "oof_sha256": {str(s): sha256(train_dir(s) / "oof_predictions.parquet") for s in SEEDS},
        "results_sha256": sha256(out_path),
    }
    (ROOT / "analysis" / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"median_pooled_auroc": results["median_pooled_auroc"], "ensemble": ensemble}, indent=1))
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--seed", required=True, type=int)
    p_train.add_argument("--dry-run", action="store_true")
    p_train.set_defaults(func=cmd_train)
    sub.add_parser("report").set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
