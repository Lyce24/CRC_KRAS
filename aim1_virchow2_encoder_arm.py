#!/usr/bin/env python3
"""E1v - Aim 1 Virchow2-CLS encoder sensitivity at the STUDY cap (8,192).

The existing Virchow2-CLS arm ran at cap 4,096 and confirmed the Aim 1
conclusions (A 0.672, D 0.710) but is not cap-matched to the declared
UNIv1 cap-8,192 study arm. This experiment closes that gap: the identical
frozen manifest, folds, seeds and recipe, with only the encoder family and
its tile geometry changed, at the study cap.

    Frozen:  aim1_dev.csv, aim1_balanced5 folds, seeds 42/43/44, ABMIL
             512/384, patient_natural sampling, AdamW 1e-4/1e-5,
             patient-AUROC selection, cap 8,192, full-bag inference.
    Varied:  UNIv1 (256 px, 1,024-d)  ->  Virchow2-CLS (224 px, 1,280-d).

As with the cap-4,096 arm, this is a broader-encoder replication, not a
one-factor ablation: the 224-px tile at 0.5 um/px is a 112-um field versus
UNIv1's 128-um, and the eligible-tile inventory differs. ``skip_finalize``
is true (no refit artifact): only the pooled OOF is evaluated, exactly as in
the sensitivity table. Also serves as the gene-level reference for the E3v
Virchow2 resolution-ladder replication.

Usage:
    python aim1_virchow2_encoder_arm.py train --seed 42
    python aim1_virchow2_encoder_arm.py report
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
ARM_ROOT = paths.OUTPUT_ROOT / "train" / f"1a_pb_cap{CAP}" / "virchow2_cls"
UNIV1_ROOT = paths.OUTPUT_ROOT / "train" / f"1a_pb_cap{CAP}" / "univ1"
V2_COORDS = "20x_224px_0px_overlap_mpp0.5"
V2_PACK = (
    "/mnt/wsl/oceanpath-hot/features/colon_stream/"
    f"{V2_COORDS}/packed_virchow2_cls_1280"
)
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20260817


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_dir(seed: int) -> Path:
    return ARM_ROOT / f"seed{seed}"


def train_command(seed: int) -> list[str]:
    d = train_dir(seed)
    overrides = [
        "platform=colon_workstation",
        "data=aim1",
        "data.aim1_model=1a",
        "data.manifest_stem=aim1_dev",
        "+data.cohort_column=cohort",
        "encoder=virchow2",
        # The sealed Aim 1 arm is the CLS-token variant: the live encoder yaml
        # now declares the 2,560-d full embedding, so pin the frozen 1,280-d.
        "encoder.feature_dim=1280",
        f"extraction.coords_dir={V2_COORDS}",
        f"extraction.coords_subdir={V2_COORDS}",
        "extraction.patch_size=224",
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
        f"training.packed_dir={V2_PACK}",
        f"train_dir={d}",
        f"exp_name=aim1_1a_pb_cap{CAP}_v2cls_seed{seed}",
        f"hydra.run.dir={ARM_ROOT / 'hydra_runs' / f'seed{seed}'}",
        "hydra.job.chdir=false",
    ]
    return [sys.executable, str(REPO / "tools" / "study_train.py"), "hydra-train", *overrides]


def cmd_train(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed must be one of {SEEDS}")
    d = train_dir(seed)
    from oceanpath.workflows.training import validate_training_run_dir

    if (d / "oof_predictions.parquet").is_file():
        validate_training_run_dir(d, require_test_predictions=True)
        print(f"== e1v seed{seed}: complete and valid — skipping")
        return
    cmd = train_command(seed)
    log_path = ARM_ROOT / "launcher_logs" / f"seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"== e1v v2cls cap{CAP} seed{seed}\n   log: {log_path}")
    if args.dry_run:
        print("   " + " ".join(cmd))
        return
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO).returncode
    if rc != 0:
        raise SystemExit(f"e1v seed{seed} failed (rc={rc}); see {log_path}")
    validate_training_run_dir(d, require_test_predictions=True)
    print(f"== e1v seed{seed}: complete")


def _set_d_mask(patients: pd.DataFrame) -> pd.Series:
    return (patients["msi_dmmr"] == "MSS/pMMR") & (patients["braf"] == "wild_type")


def cmd_report(_: argparse.Namespace) -> None:
    out_path = ARM_ROOT / "analysis" / "results.json"
    if out_path.exists():
        raise SystemExit(f"append-only: {out_path} exists")
    from oceanpath.workflows.training import validate_training_run_dir

    manifest = pd.read_csv(paths.DEV_MANIFEST, low_memory=False)
    frames: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        validate_training_run_dir(train_dir(seed), require_test_predictions=True)
        frames[seed] = evaluate.to_patient_level(
            pd.read_parquet(train_dir(seed) / "oof_predictions.parquet"), manifest
        )

    per_seed = {}
    deltas = []
    for seed, pat in frames.items():
        mask_d = _set_d_mask(pat)
        a_complete = (pat["msi_dmmr"] != "unknown") & (pat["braf"] != "unknown")
        shared = evaluate.shared_resample_delta(
            pat[a_complete].reset_index(drop=True),
            _set_d_mask(pat[a_complete].reset_index(drop=True)),
            score_column="mean_logit",
            n_bootstrap=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
        )
        per_seed[str(seed)] = {
            "auroc_A": evaluate.patient_auroc(pat, "mean_logit"),
            "auroc_D": evaluate.patient_auroc(pat[mask_d], "mean_logit"),
            "d_minus_a_complete": -shared["delta"],
            "d_minus_a_complete_ci": [-shared["delta_ci_high"], -shared["delta_ci_low"]],
        }
        deltas.append(per_seed[str(seed)]["d_minus_a_complete"])

    base = frames[SEEDS[0]][["patient_id", "label", "msi_dmmr", "braf"]].copy()
    stacked = np.stack(
        [
            frames[s].set_index("patient_id").loc[base["patient_id"], "mean_logit"]
            for s in SEEDS
        ]
    )
    base["mean_logit"] = stacked.mean(axis=0)
    ensemble = evaluate.bootstrap_auroc(
        base.assign(prob_raw=base["mean_logit"]),
        "mean_logit",
        n_bootstrap=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
    )

    results = {
        "experiment": "e1v_virchow2_cls_cap8192_aim1_sensitivity",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "per_seed": per_seed,
        "median_auroc_A": float(np.median([per_seed[str(s)]["auroc_A"] for s in SEEDS])),
        "median_auroc_D": float(np.median([per_seed[str(s)]["auroc_D"] for s in SEEDS])),
        "median_d_minus_a_complete": float(np.median(deltas)),
        "three_seed_ensemble_A": ensemble,
        "conventions": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "estimands": "same declared conventions as the cap-4096 Virchow2 arm",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    receipt = {
        "created_utc": results["created_utc"],
        "oof_sha256": {str(s): sha256(train_dir(s) / "oof_predictions.parquet") for s in SEEDS},
        "manifest_sha256": sha256(paths.DEV_MANIFEST),
        "results_sha256": sha256(out_path),
    }
    (ARM_ROOT / "analysis" / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k.startswith("median")}, indent=1))
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
