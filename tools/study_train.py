#!/usr/bin/env python3
"""Aim 1 phase 3 — five-fold development CV for one model.

Trains a model of ``oceanpath.aim1.registry`` under the locked recipe
(``training=aim1``) on the frozen balanced fold layout, and reports the
development numbers of §4.7:

    1. per-fold patient AUROC, mean +/- SD        (stability diagnostic ONLY)
    2. pooled OOF patient AUROC with a patient-clustered bootstrap 95% CI
                                                  (the development headline)
    3. per-subcohort OOF (TCGA-COAD / TCGA-READ / SR386)   (shortcut check)

The frozen predictor is the five fold models averaged as a mean logit — no
refit on the full development set and no best-fold selection (§4.6) — so
training runs with ``skip_finalize=true`` and phase 4 reads the fold
checkpoints directly.

Subcommands:

    train <model|all> [--dry-run] [-o KEY=VALUE ...]
        Run the model's five folds. Completed runs are skipped, so this is
        resumable.

    report <model|all>
        Aggregate the fold runs into dev_report.json / dev_report.md.

    hydra-train [hydra overrides...]
        Internal: one training invocation.

Usage:
    python tools/study_train.py train 1a
    python tools/study_train.py report 1a
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import evaluate, paths, registry  # noqa: E402

# Locked architecture (2026-08-18): feature dropout 0.10 on the frozen UNI-v1
# features -> Linear 1024->512 + ReLU -> gated attention (512->384 Tanh x
# 512->384 Sigmoid, dropout 0.25) -> Linear 384->1 -> softmax over patches ->
# weighted sum of the 512-d patch features -> Linear 512 -> output.
BASE_OVERRIDES = [
    "platform=colon_workstation",
    "data=aim1",
    "encoder=univ1",
    "splits=aim1_balanced",
    "model=abmil",
    "model.embed_dim=512",
    "model.attn_dim=384",
    "model.input_dropout=0.10",
    f"model.dropout={paths.DROPOUT}",
    "training=aim1",
]


def hydra_config_dir() -> Path:
    """Return the repository config root used by the internal Hydra entry point.

    ``hydra.main`` resolves relative ``config_path`` values from this file, not
    from the subprocess working directory.  The project configs live at the
    repository root (``configs/``), one level above this launcher.
    """
    config_dir = (REPO / "configs").resolve()
    if not (config_dir / "train.yaml").is_file():
        raise FileNotFoundError(f"Hydra training config not found: {config_dir / 'train.yaml'}")
    return config_dir


def model_overrides(model_id: str, seed: int) -> list[str]:
    model = registry.MODELS[model_id]
    return [
        *BASE_OVERRIDES,
        f"data.aim1_model={model_id}",
        f"data.manifest_stem={model.manifest_stem}",
        f"splits.seed={seed}",
        f"training.seed={seed}",
        f"training.sample_weight_column={model.weight_column}",
        f"train_dir={model.run_dir(seed)}",
        f"exp_name=aim1_{model_id}_{paths.PINNED_ENCODER}_seed{seed}",
    ]


def run_hydra_train(overrides: list[str]) -> None:
    import hydra
    from omegaconf import DictConfig

    from oceanpath.workflows.training import run_training

    @hydra.main(config_path=str(hydra_config_dir()), config_name="train", version_base="1.3")
    def _main(cfg: DictConfig) -> None:
        print(run_training(cfg).to_json())

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *overrides]
        _main()
    finally:
        sys.argv = original_argv


def selected_models(token: str) -> list[str]:
    if token == "all":
        return list(registry.TRAINING_ORDER)
    if token not in registry.MODELS:
        known = ", ".join([*registry.TRAINING_ORDER, "all"])
        raise SystemExit(f"Unknown model '{token}'. Known: {known}")
    return [token]


def run_is_complete(run_directory: Path) -> bool:
    return (run_directory / "training_summary.json").is_file() and (
        run_directory / "oof_predictions.parquet"
    ).is_file()


def cmd_train(args: argparse.Namespace) -> None:
    for model_id in selected_models(args.model):
        model = registry.MODELS[model_id]
        if not model.manifest_path.is_file():
            raise SystemExit(
                f"{model_id}: manifest {model.manifest_path} missing — build it first "
                "(phase 2 for 1a/1cr, phase 6 for 1cb)"
            )
        run_directory = model.run_dir(args.seed)
        if run_is_complete(run_directory):
            print(f"== {model_id}: already complete at {run_directory} — skipping")
            continue
        overrides = model_overrides(model_id, args.seed) + args.override
        command = [sys.executable, str(Path(__file__).resolve()), "hydra-train", *overrides]
        print(f"== {model_id} seed {args.seed}\n   {' '.join(command)}")
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=REPO, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"{model_id} failed (exit {completed.returncode})")
    if args.dry_run:
        print("dry run — commands printed, nothing executed")


def cmd_report(args: argparse.Namespace) -> None:
    for model_id in selected_models(args.model):
        model = registry.MODELS[model_id]
        run_directory = model.run_dir(args.seed)
        if not run_is_complete(run_directory):
            print(f"== {model_id}: not complete — skipping")
            continue
        report = evaluate.dev_report(model_id, args.seed)
        (run_directory / "dev_report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"\n== {model_id}: {model.title}")
        print(
            f"  per-fold patient AUROC   {report['per_fold_mean']:.3f} "
            f"+/- {report['per_fold_sd']:.3f}   {[round(v, 3) for v in report['per_fold']]}"
        )
        pooled = report["pooled_oof"]
        print(
            f"  pooled OOF patient AUROC {pooled['auroc']:.3f} "
            f"(95% CI {pooled['ci_low']:.3f}-{pooled['ci_high']:.3f}, n={pooled['n']})"
        )
        print("  per-subcohort OOF:")
        for subcohort, block in report["per_subcohort"].items():
            print(
                f"    {subcohort:12s} AUROC {block['auroc']:.3f}  "
                f"n={block['n']} ({block['n_positive']} mutant)"
            )
        print(f"  best epochs per fold: {report['best_epochs']}")
        print(f"  wrote {run_directory / 'dev_report.json'}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "hydra-train":
        run_hydra_train(argv[1:])
        return

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=paths.PRIMARY_SEED)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="run the five folds")
    train.add_argument("model", help="1a | 1cr | 1cb | all")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("-o", "--override", action="append", default=[], help="extra hydra override")
    train.set_defaults(func=cmd_train)

    report = sub.add_parser("report", help="aggregate the development numbers")
    report.add_argument("model", help="1a | 1cr | 1cb | all")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
