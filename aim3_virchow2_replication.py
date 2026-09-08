#!/usr/bin/env python3
"""E3v - Virchow2 replication of the Aim 3 consensus resolution ceilings.

WHY. Aim 3's three consensus ceilings (codon, G12D-vs-other-mutant,
G12D-within-G12) are the paper's most novel claim, and their one credible
attack is encoder capacity: "UNIv1 simply cannot see allele-level
morphology." The matched learnable controls answer that within-architecture;
this experiment answers it across encoder families. It retrains the three
ceiling rungs and their FROZEN matched controls — identical task manifests,
identical fixed wild-type control draws, identical folds, seeds and recipe,
at the same cap 8,192 — with Virchow2-CLS features in place of UNIv1.

PRE-DECLARED READING, before any E3v result is seen:

    * ceilings replicate under Virchow2  ->  the resolution boundary is
      encoder-robust, not a property of one 2023 encoder;
    * any rung opens (fine upper bound clears the gate) ->  a positive
      discovery: allele-level signal is reachable with a stronger encoder,
      and Aim 3's conclusion must be revised, not defended.

SCOPE. Fine+control for the three consensus-ceiling rungs only (18 chains =
3 rungs x 2 arms x 3 seeds). G12V/G12C are excluded: they are unresolved
under their gates and a second encoder cannot repair power. The three-draw
repeated-control campaign is not repeated: it tested WT-draw sensitivity,
which is orthogonal to encoder choice; the fixed draw is reused verbatim.
The Virchow2 gene-level reference is the cap-matched E1v arm.

STATISTICS. Identical to the sealed corrected analysis, by import: patient
mean native logit per seed, three-seed ensemble, and
``aim3_fixed_control_analysis.partially_paired_bootstrap`` (shared positives,
cohort-stratified independent negatives; 10,000 draws, seed 20260817), with
both the 95% verdict and the fixed one-sided-99% intersection-union gate.
Before reporting any Virchow2 number, the report step recomputes the three
UNIv1 ensemble points from the sealed e3a train dirs and asserts them
against the published values to 1e-6 — anchoring loader and estimator.

Usage:
    python aim3_virchow2_replication.py train --task codon --seed 42
    python aim3_virchow2_replication.py report
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import aim3_resolution_ladder  # noqa: E402  (frozen task registry, manifests, split layout)
from aim3_fixed_control_analysis import partially_paired_bootstrap, verdict_for  # noqa: E402
from oceanpath.aim1 import evaluate, paths  # noqa: E402

SEEDS = (42, 43, 44)
CAP = 8192
ROOT = paths.OUTPUT_ROOT / "reruns" / "aim3_virchow2_cap8192_v1_20260820"
PAIRS = [
    ("codon", "ctrl_codon"),
    ("g12d_broad", "ctrl_g12d_broad"),
    ("allele1", "ctrl_allele1"),
]
TASKS = [t for pair in PAIRS for t in pair]
V2_COORDS = "20x_224px_0px_overlap_mpp0.5"
V2_PACK = (
    "/mnt/wsl/oceanpath-hot/features/colon_stream/"
    f"{V2_COORDS}/packed_virchow2_cls_1280"
)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260817
CEILING_BOUND = aim3_resolution_ladder.CEILING_BOUND
CHANCE = aim3_resolution_ladder.CHANCE
# Sealed UNIv1 three-seed ensemble points (Results.md section 4.1) — anchors.
UNIV1_EXPECTED = {
    "codon": 0.526508,
    "ctrl_codon": 0.658113,
    "g12d_broad": 0.486302,
    "ctrl_g12d_broad": 0.653200,
    "allele1": 0.525216,
    "ctrl_allele1": 0.643932,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(task: str, seed: int) -> Path:
    return ROOT / "train" / task / f"seed{seed}"


def train_command(task: str, seed: int) -> list[str]:
    d = run_dir(task, seed)
    overrides = [
        "platform=colon_workstation",
        "data=aim1",
        f"data.aim1_model=e3a_{task}",
        f"data.manifest_stem=aim1_e3a_{task}",
        "+data.cohort_column=cohort",
        "encoder=virchow2",
        # Sealed Virchow2-CLS variant: pin 1,280-d against the live 2,560-d yaml.
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
        f"training.fixed_epoch_budget={aim3_resolution_ladder.FIXED_EPOCH_BUDGET}",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        "training.skip_finalize=true",
        f"training.packed_dir={V2_PACK}",
        f"train_dir={d}",
        f"exp_name=e3v_{task}_seed{seed}",
        f"hydra.run.dir={ROOT / 'hydra_runs' / f'{task}_seed{seed}'}",
        "hydra.job.chdir=false",
    ]
    return [sys.executable, str(REPO / "tools" / "study_train.py"), "hydra-train", *overrides]


def cmd_train(args: argparse.Namespace) -> None:
    task, seed = args.task, int(args.seed)
    if task not in TASKS:
        raise SystemExit(f"task must be one of {TASKS}")
    if seed not in SEEDS:
        raise SystemExit(f"seed must be one of {SEEDS}")
    if not aim3_resolution_ladder.manifest_path(task).is_file() or not (aim3_resolution_ladder.split_dir(task) / "splits.parquet").is_file():
        raise SystemExit(f"frozen manifest/splits missing for {task}; refusing to rebuild")
    d = run_dir(task, seed)
    from oceanpath.workflows.training import validate_training_run_dir

    if (d / "oof_predictions.parquet").is_file():
        validate_training_run_dir(d, require_test_predictions=True)
        print(f"== e3v {task} seed{seed}: complete and valid — skipping")
        return
    cmd = train_command(task, seed)
    log_path = ROOT / "launcher_logs" / f"{task}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"== e3v {task} seed{seed} cap{CAP}\n   log: {log_path}")
    if args.dry_run:
        print("   " + " ".join(cmd))
        return
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO).returncode
    if rc != 0:
        raise SystemExit(f"e3v {task} seed{seed} failed (rc={rc}); see {log_path}")
    validate_training_run_dir(d, require_test_predictions=True)
    print(f"== e3v {task} seed{seed}: complete")


def _ensemble(task_root: Path, task: str, manifest: pd.DataFrame) -> pd.DataFrame:
    """Three-seed mean-logit ensemble patient frame for one task chain."""
    frames = []
    for seed in SEEDS:
        oof = pd.read_parquet(task_root / f"seed{seed}" / "oof_predictions.parquet")
        pat = evaluate.to_patient_level(oof, manifest).sort_values("patient_id")
        frames.append(pat.reset_index(drop=True))
    reference = frames[0]
    for other in frames[1:]:
        if not reference["patient_id"].equals(other["patient_id"]) or not reference[
            "label"
        ].equals(other["label"]):
            raise AssertionError(f"{task}: seeds do not cover identical patients")
    out = reference[["patient_id", "label", "cohort", "subcohort"]].copy()
    out["mean_logit"] = np.mean(np.stack([f["mean_logit"].to_numpy() for f in frames]), axis=0)
    return out


def _auc(frame: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(frame["label"], frame["mean_logit"]))


def cmd_report(_: argparse.Namespace) -> None:
    out_path = ROOT / "analysis" / "results.json"
    if out_path.exists():
        raise SystemExit(f"append-only: {out_path} exists")
    from oceanpath.workflows.training import validate_training_run_dir

    manifests = {t: pd.read_csv(aim3_resolution_ladder.manifest_path(t), low_memory=False) for t in TASKS}

    anchors = {}
    for task in TASKS:
        univ1 = _ensemble(aim3_resolution_ladder.E3A_ROOT / "train" / task, task, manifests[task])
        observed = _auc(univ1)
        if abs(observed - UNIV1_EXPECTED[task]) > 1e-6:
            raise AssertionError(
                f"UNIv1 anchor failed for {task}: {observed:.6f} != {UNIV1_EXPECTED[task]}"
            )
        anchors[task] = observed
    print(f"UNIv1 anchor PASS: {anchors}")

    for task in TASKS:
        for seed in SEEDS:
            validate_training_run_dir(run_dir(task, seed), require_test_predictions=True)

    rungs = {}
    for fine_task, control_task in PAIRS:
        fine = _ensemble(ROOT / "train" / fine_task, fine_task, manifests[fine_task])
        control = _ensemble(
            ROOT / "train" / control_task, control_task, manifests[control_task]
        )
        boots = partially_paired_bootstrap(
            fine, control, n_bootstrap=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
        )

        def summary(point: float, draws: np.ndarray) -> dict:
            return {
                "auroc": point,
                "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                "one_sided_99_low": float(np.percentile(draws, 1.0)),
                "one_sided_99_high": float(np.percentile(draws, 99.0)),
            }

        fine_summary = summary(_auc(fine), boots["fine_auc"])
        control_summary = summary(_auc(control), boots["control_auc"])
        delta_point = control_summary["auroc"] - fine_summary["auroc"]
        delta_summary = summary(delta_point, boots["delta"])
        gate = {
            "fine_upper99_below_0p60": fine_summary["one_sided_99_high"] < CEILING_BOUND,
            "control_lower99_above_0p50": control_summary["one_sided_99_low"] > CHANCE,
            "delta_lower99_above_zero": delta_summary["one_sided_99_low"] > 0.0,
        }
        gate["ceiling"] = all(gate.values())
        rungs[fine_task] = {
            "n": int(len(fine)),
            "n_positive": int(fine["label"].sum()),
            "fine": fine_summary,
            "control": control_summary,
            "delta_control_minus_fine": delta_summary,
            "fixed_gate_one_sided_99": gate,
            "verdict_ci95": verdict_for(
                {"ci95": fine_summary["ci95"]},
                {"ci95": control_summary["ci95"]},
                {"ci95": delta_summary["ci95"]},
            ),
            "per_seed_fine": {
                str(s): float(
                    _auc(
                        evaluate.to_patient_level(
                            pd.read_parquet(
                                run_dir(fine_task, s) / "oof_predictions.parquet"
                            ),
                            manifests[fine_task],
                        )
                    )
                )
                for s in SEEDS
            },
        }

    results = {
        "experiment": "e3v_virchow2_cap8192_ceiling_replication",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": "virchow2_cls_1280",
        "cap": CAP,
        "univ1_anchor_points": anchors,
        "rungs": rungs,
        "conventions": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "estimator": "three-seed mean-logit ensemble; sealed partially-paired bootstrap",
            "gene_reference": "E1v (Virchow2-CLS cap-8192 Aim 1 arm)",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    receipt = {
        "created_utc": results["created_utc"],
        "manifest_sha256": {t: sha256(aim3_resolution_ladder.manifest_path(t)) for t in TASKS},
        "oof_sha256": {
            f"{t}_seed{s}": sha256(run_dir(t, s) / "oof_predictions.parquet")
            for t in TASKS
            for s in SEEDS
        },
        "results_sha256": sha256(out_path),
    }
    (ROOT / "analysis" / "receipt.json").write_text(json.dumps(receipt, indent=2))
    for fine_task, block in rungs.items():
        print(
            f"{fine_task}: fine {block['fine']['auroc']:.4f} "
            f"ctrl {block['control']['auroc']:.4f} "
            f"gate={'CEILING' if block['fixed_gate_one_sided_99']['ceiling'] else 'no ceiling'}"
        )
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--task", required=True)
    p_train.add_argument("--seed", required=True, type=int)
    p_train.add_argument("--dry-run", action="store_true")
    p_train.set_defaults(func=cmd_train)
    sub.add_parser("report").set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
