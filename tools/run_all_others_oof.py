#!/usr/bin/env python3
"""Prepare and report related 5-fold OOF KRAS binary experiments.

The tasks use the feature-backed v2 DEV population (SR386-P, SR1482-P, and
TCGA-P) and inherit its frozen patient-level seed-42 fold assignment. In
contrast to the mutant-only allele tasks, wild-type samples form part or all
of the negative class:

    p1_all_v2: G12D vs wild-type + every other KRAS mutation
    e2_all_v2: G12D/G12V vs wild-type + every other KRAS mutation
    p1_wt_v2:  G12D vs wild-type only

The training commands intentionally remain separate so the jobs can be
run sequentially and resumed with the generic training workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.kras import study  # noqa: E402
from oceanpath.kras.labels import subvariant_tokens  # noqa: E402
from oceanpath.splitting.core import derive_subset_splits, verify_split_integrity  # noqa: E402

SEED = 42
SPLIT_NAME = study.split_name(SEED)
MASTER_MANIFEST = study.MANIFEST_ROOT / "crc_kras_master_dev_v2.csv"
MASTER_SPLITS = REPO / "outputs/splits/colon_kras_master_v2" / SPLIT_NAME
MANIFEST_ROOT = REPO / "outputs/manifests/colon"
TRAIN_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/train/kras/all_others_oof")
REPORT_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/eval/kras_all_others_oof")

TASKS = {
    "p1_all_v2": {
        "title": "G12D vs all others (wild-type + other KRAS mutations)",
        "positive_tokens": {"G12D"},
        "population": "all",
    },
    "e2_all_v2": {
        "title": "G12D/G12V vs all others (wild-type + other KRAS mutations)",
        "positive_tokens": {"G12D", "G12V"},
        "population": "all",
    },
    "p1_wt_v2": {
        "title": "G12D vs KRAS wild-type",
        "positive_tokens": {"G12D"},
        "population": "g12d_or_wild_type",
    },
}


def manifest_path(task: str) -> Path:
    return MANIFEST_ROOT / f"crc_kras_dev_{task}.csv"


def splits_dir(task: str) -> Path:
    return REPO / "outputs/splits" / f"colon_kras_{task}" / SPLIT_NAME


def train_dir(task: str) -> Path:
    return TRAIN_ROOT / f"{task}_univ1_seed{SEED}"


def prepare() -> None:
    master = pd.read_csv(MASTER_MANIFEST)
    if not set(master["kras"].unique()) <= {"mutant", "wild_type"}:
        raise SystemExit("The v2 master contains a KRAS-unknown row")
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    master_assignment = pd.read_parquet(MASTER_SPLITS / "splits.parquet")
    for task, spec in TASKS.items():
        out = master.copy()
        positive_tokens = spec["positive_tokens"]
        if spec["population"] == "g12d_or_wild_type":
            has_positive_token = out["kras_subvariant"].map(
                lambda value, tokens=positive_tokens: bool(
                    tokens & set(subvariant_tokens(value))
                )
            )
            out = out[(out["kras"] == "wild_type") | has_positive_token].copy()
        out["target_label"] = out["kras_subvariant"].map(
            lambda value, tokens=positive_tokens: int(
                bool(tokens & set(subvariant_tokens(value)))
            )
        )
        out = out[study.MANIFEST_COLUMNS].sort_values("slide_id").reset_index(drop=True)
        study.assert_manifest_invariants(out, task)

        # Explicitly audit the requested negative pool.
        if (out.loc[out["kras"] == "wild_type", "target_label"] != 0).any():
            raise SystemExit(f"{task}: a wild-type row is not negative")
        if (out.loc[out["target_label"] == 1, "kras"] != "mutant").any():
            raise SystemExit(f"{task}: a positive row is not KRAS-mutant")

        path = manifest_path(task)
        out.to_csv(path, index=False)
        derived = derive_subset_splits(
            master_splits_dir=MASTER_SPLITS,
            manifest_csv=path,
            output_dir=splits_dir(task),
            filename_column="slide_id",
        )
        verify_split_integrity(derived.parent, path)

        assignment = pd.read_parquet(derived)
        joined = assignment[["slide_id", "fold"]].merge(
            master_assignment[["slide_id", "fold"]],
            on="slide_id",
            suffixes=("", "_master"),
            validate="one_to_one",
        )
        if len(joined) != len(out) or not (joined["fold"] == joined["fold_master"]).all():
            raise SystemExit(f"{task}: derived folds do not exactly inherit master v2")

        patient_labels = out.groupby("patient_id")["target_label"].first()
        n_wt = out.loc[out["kras"] == "wild_type", "patient_id"].nunique()
        print(
            f"{task}: {len(out)} slides / {len(patient_labels)} patients; "
            f"{int(patient_labels.sum())} positive, "
            f"{len(patient_labels) - int(patient_labels.sum())} negative "
            f"({n_wt} wild-type patients); folds inherited and verified"
        )
        print(f"  manifest: {path}")
        print(f"  splits:   {derived.parent}")


def report_task(task: str) -> dict:
    run = train_dir(task)
    if not study.run_is_complete(run):
        raise SystemExit(f"Incomplete training run: {run}")
    manifest = pd.read_csv(manifest_path(task))
    oof = pd.read_parquet(run / "oof_predictions.parquet")
    if set(oof["slide_id"]) != set(manifest["slide_id"]):
        raise SystemExit(f"{task}: OOF predictions do not exactly cover the manifest")

    fold_aurocs: list[float] = []
    for fold in range(5):
        patients = study.to_patient_logits(oof[oof["fold"] == fold], manifest)
        fold_aurocs.append(study.patient_auroc(patients))

    pooled_patients = study.to_patient_logits(oof.drop(columns=["fold"]), manifest)
    pooled = study.bootstrap_patient_auroc(pooled_patients)
    patient_prob = study.sigmoid(pooled_patients["logit"].to_numpy())
    patient_auprc = float(average_precision_score(pooled_patients["label"], patient_prob))

    per_cohort: dict[str, dict] = {}
    patient_cohort = manifest.groupby("patient_id")["cohort_group"].first()
    for cohort in sorted(patient_cohort.unique()):
        ids = set(patient_cohort[patient_cohort == cohort].index)
        subset = pooled_patients[pooled_patients["patient_id"].isin(ids)]
        per_cohort[str(cohort)] = {
            "auroc": float(roc_auc_score(subset["label"], subset["logit"])),
            "n_patients": int(len(subset)),
            "n_positive": int(subset["label"].sum()),
        }

    summary = json.loads((run / "cv_summary.json").read_text())
    return {
        "task": task,
        "title": TASKS[task]["title"],
        "encoder": "UNIv1",
        "model": "ABMIL",
        "seed": SEED,
        "n_folds": 5,
        "n_slides": int(len(manifest)),
        "n_patients": int(manifest["patient_id"].nunique()),
        "n_positive_patients": int(
            manifest.groupby("patient_id")["target_label"].first().sum()
        ),
        "per_fold_patient_auroc": fold_aurocs,
        "per_fold_patient_auroc_mean": float(np.mean(fold_aurocs)),
        "per_fold_patient_auroc_sd": float(np.std(fold_aurocs)),
        "pooled_oof_patient": {**pooled, "auprc": patient_auprc},
        "per_cohort_oof": per_cohort,
        "pooled_oof_slide": summary["oof_pooled"],
        "best_epochs": [
            int(json.loads((run / f"fold_{fold}/fold_metrics.json").read_text())["best_epoch"])
            for fold in range(5)
        ],
        "run_dir": str(run),
    }


def report() -> None:
    results = {task: report_task(task) for task in TASKS}
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "oof_report.json").write_text(json.dumps(results, indent=2) + "\n")

    lines = [
        "# KRAS binary contrasts — 5-fold OOF CV",
        "",
        "Pinned setup: UNIv1 features, ABMIL, seed 42, shared patient-level folds.",
        "",
        "| Task | Patients (positive) | Fold AUROC mean ± SD | Pooled OOF patient AUROC (95% CI) | Patient AUPRC |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in results.values():
        pooled = item["pooled_oof_patient"]
        lines.append(
            f"| {item['title']} | {item['n_patients']} ({item['n_positive_patients']}) "
            f"| {item['per_fold_patient_auroc_mean']:.3f} ± "
            f"{item['per_fold_patient_auroc_sd']:.3f} | {pooled['auroc']:.3f} "
            f"({pooled['ci_low']:.3f}–{pooled['ci_high']:.3f}) | {pooled['auprc']:.3f} |"
        )
    for item in results.values():
        lines.extend(
            [
                "",
                f"- {item['task']} fold AUROCs: "
                + ", ".join(f"{value:.3f}" for value in item["per_fold_patient_auroc"]),
                f"- {item['task']} best epochs: "
                + ", ".join(str(value) for value in item["best_epochs"]),
                f"- {item['task']} per-cohort: "
                + ", ".join(
                    f"{cohort} {values['auroc']:.3f} "
                    f"(n={values['n_patients']}, pos={values['n_positive']})"
                    for cohort, values in item["per_cohort_oof"].items()
                ),
            ]
        )
    (REPORT_ROOT / "oof_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nJSON: {REPORT_ROOT / 'oof_report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "report"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        report()


if __name__ == "__main__":
    main()
