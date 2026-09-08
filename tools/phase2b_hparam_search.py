#!/usr/bin/env python3
"""Aim 1 phase 2b — the learning-rate / weight-decay search, run before 1A.

Two axes only. Everything else — optimizer, projection and attention dims,
dropout, schedule, warmup, gradient clipping, max epochs, patience, and bag
handling — is fixed before the first run and recorded in the report. Six
configurations spent on two axes say something about those two axes; six
scattered over six axes say nothing about any.

Grid (6 configurations), centred on the recipe that already works rather than
on a generic default:

    lr             2.5e-5, 1e-4, 4e-4   4x spacing covers 16x in three points,
                                        centred on the frozen 1e-4
    weight_decay   1e-5, 1e-2           effectively-off vs meaningful decay

Patch features enter the aggregator UNCHANGED — no L2 normalization, no
z-scoring. The aggregator keeps its original structure, so the incumbent 1e-4
is directly comparable and belongs at the centre of the grid. The consequence
is that this lr is tuned for UNI-v1's feature scale specifically (mean patch
L2 norm ~38.5) and should NOT be assumed to transfer to Virchow2 or CONCH,
which emit different scales; those encoders need their own 3-point lr sweep at
the locked weight decay.

Selection (§Step 3-4):

    Direction A    train TCGA-COAD+READ (499 patients), select on SR386 (410)
    Direction B    train SR386, select on TCGA

    Configurations are ranked WITHIN each direction by patient-level
    validation AUROC — the same criterion that stops training and selects the
    checkpoint. The winner minimizes the sum of the two ranks, tie-broken
    toward lower lr then higher weight decay.

Ranks, not raw scores, on purpose: taking the maximum of six noisy AUROCs inflates
it, whereas a rank statistic carries no magnitude and so has nothing to
inflate. The search only ORDERS configurations — every reported number comes
later from the independent 5-fold OOF run. Requiring agreement across two
opposite transfer directions also drops configurations that merely suit one
domain's scanner or staining profile.

No external cohort and no outer fold is touched here. The search does see the
labels of all 909 development patients, which makes the eventual development
OOF number mildly optimistic; external estimates are unaffected. That is
recorded in the report and in Aim1_Setup.md rather than left implicit.

Usage:
    python tools/phase2b_hparam_search.py run          # 12 training runs
    python tools/phase2b_hparam_search.py report       # rank and lock the winner
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import paths, registry  # noqa: E402
from oceanpath.splitting import SplitConfig, generate_splits  # noqa: E402

LR_GRID = (2.5e-5, 1e-4, 4e-4)  # x/4, x, 4x around the frozen lr = 1e-4
WD_GRID = (1e-5, 1e-2)

# Direction -> (label, val_filter). The val cohort is both the early-stopping
# monitor and the ranking criterion for that direction.
# Direction -> label. The split profile (configs/splits/aim1_transfer_<d>.yaml)
# carries the cohort filter: Hydra's override grammar cannot take a filter
# expression containing spaces and quotes, and a config file also keeps the
# exact selection rule in version control rather than in a command line.
DIRECTIONS: dict[str, str] = {
    "a": "train TCGA -> select SR386",
    "b": "train SR386 -> select TCGA",
}

TUNING_ROOT = paths.OUTPUT_ROOT / "tuning"
REPORT_PATH = TUNING_ROOT / "tuning_report.json"

# Everything not on the two swept axes, frozen before the first run. These
# mirror the values already in configs/training/aim1.yaml and configs/model —
# the search inherits the recipe, it does not redefine it.
BASE_OVERRIDES = [
    "platform=colon_workstation",
    "data=aim1",
    "data.aim1_model=1a",
    "data.manifest_stem=aim1_dev",
    "encoder=univ1",
    "model=abmil",
    "model.embed_dim=512",
    "model.attn_dim=256",
    f"model.dropout={paths.DROPOUT}",
    "training=aim1",
    "training.sample_weight_column=slide_weight",
    f"training.seed={paths.PRIMARY_SEED}",
    # No finalization during a search: no refit, no exported artifact.
    "training.skip_finalize=true",
]


def config_id(lr: float, weight_decay: float) -> str:
    return f"lr{lr:g}_wd{weight_decay:g}"


def run_dir(direction: str, lr: float, weight_decay: float) -> Path:
    return TUNING_ROOT / f"direction_{direction}" / config_id(lr, weight_decay)


def overrides_for(direction: str, lr: float, weight_decay: float) -> list[str]:
    directory = run_dir(direction, lr, weight_decay)
    return [
        *BASE_OVERRIDES,
        f"splits=aim1_transfer_{direction}",
        f"training.lr={lr:g}",
        f"training.weight_decay={weight_decay:g}",
        f"train_dir={directory}",
        f"exp_name=aim1_tune_{direction}_{config_id(lr, weight_decay)}",
    ]


def run_is_complete(directory: Path) -> bool:
    return (directory / "fold_0" / "fold_metrics.json").is_file()


def selection_score(directory: Path) -> float:
    """Selection criterion: best PATIENT-LEVEL validation AUROC.

    The same quantity that stops training and picks the checkpoint, so the
    search ranks configurations by what the protocol actually optimizes.
    """
    metrics = json.loads((directory / "fold_0" / "fold_metrics.json").read_text())
    return float(metrics[paths.MONITOR_METRIC])


# The cohort filter lives in configs/splits/aim1_transfer_<d>.yaml for the
# trainer; it is repeated here only to materialize the split artifacts the
# trainer then loads. Keep the two in step.
VAL_FILTER = {"a": 'subcohort == "SR386"', "b": 'subcohort != "SR386"'}
NO_TEST_FILTER = 'subcohort == "__none__"'


def ensure_splits(direction: str) -> None:
    """Materialize one transfer split where FoundationPaths will look for it."""
    model = registry.MODELS["1a"]
    output_dir = REPO / paths.SPLIT_ROOT / model.data_name / f"aim1_transfer_{direction}"
    result = generate_splits(
        SplitConfig(
            scheme="custom_holdout",
            name=f"aim1_transfer_{direction}",
            csv_path=str(model.manifest_path),
            output_dir=str(output_dir),
            filename_column="slide_id",
            label_column="target_label",
            group_column="patient_id",
            test_filter=NO_TEST_FILTER,
            val_filter=VAL_FILTER[direction],
            seed=paths.PRIMARY_SEED,
        )
    )
    print(f"   split {direction}: {result.fold_distribution} -> {result.parquet_path}")


def cmd_run(args: argparse.Namespace) -> None:
    grid = list(itertools.product(LR_GRID, WD_GRID))
    print(f"{len(grid)} configurations x {len(DIRECTIONS)} directions = {len(grid) * 2} runs")
    for direction in DIRECTIONS:
        if not args.dry_run:
            ensure_splits(direction)
        for lr, weight_decay in grid:
            directory = run_dir(direction, lr, weight_decay)
            if run_is_complete(directory):
                print(f"== {direction}/{config_id(lr, weight_decay)}: complete — skipping")
                continue
            command = [
                sys.executable,
                str(REPO / "tools" / "study_train.py"),
                "hydra-train",
                *overrides_for(direction, lr, weight_decay),
            ]
            print(f"== {direction}/{config_id(lr, weight_decay)}")
            if args.dry_run:
                print("   " + " ".join(command))
                continue
            completed = subprocess.run(command, cwd=REPO, check=False)
            if completed.returncode != 0:
                raise SystemExit(
                    f"{direction}/{config_id(lr, weight_decay)} failed "
                    f"(exit {completed.returncode})"
                )
    if args.dry_run:
        print("dry run — nothing executed")


def cmd_report(args: argparse.Namespace) -> None:
    grid = list(itertools.product(LR_GRID, WD_GRID))
    scores: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for direction in DIRECTIONS:
        scores[direction] = {}
        for lr, weight_decay in grid:
            directory = run_dir(direction, lr, weight_decay)
            if not run_is_complete(directory):
                missing.append(f"{direction}/{config_id(lr, weight_decay)}")
                continue
            scores[direction][config_id(lr, weight_decay)] = selection_score(directory)
    if missing:
        raise SystemExit(f"{len(missing)} run(s) incomplete: {missing[:6]}")

    # Rank 1 = best within a direction.
    ranks: dict[str, dict[str, int]] = {}
    for direction, block in scores.items():
        ordered = sorted(block.items(), key=lambda item: -item[1])
        ranks[direction] = {name: position + 1 for position, (name, _) in enumerate(ordered)}

    rank_sum = {
        config_id(lr, wd): sum(ranks[d][config_id(lr, wd)] for d in DIRECTIONS) for lr, wd in grid
    }
    # Tie-break: lower learning rate first, then higher weight decay.
    winner = min(grid, key=lambda pair: (rank_sum[config_id(*pair)], pair[0], -pair[1]))
    winner_id = config_id(*winner)

    print(
        f"\n{'config':16s} "
        + " ".join(f"{d.upper() + ' AUROC':>12s} {'rank':>5s}" for d in DIRECTIONS)
        + f" {'sum':>5s}"
    )
    for lr, weight_decay in grid:
        name = config_id(lr, weight_decay)
        cells = " ".join(f"{scores[d][name]:12.4f} {ranks[d][name]:5d}" for d in DIRECTIONS)
        marker = "  <-- winner" if name == winner_id else ""
        print(f"{name:16s} {cells} {rank_sum[name]:5d}{marker}")

    for direction, label in DIRECTIONS.items():
        print(f"  direction {direction}: {label}")

    report = {
        "winner": winner_id,
        "winner_lr": winner[0],
        "winner_weight_decay": winner[1],
        "overrides": [
            f"training.lr={winner[0]:g}",
            f"training.weight_decay={winner[1]:g}",
        ],
        "grid": {"lr": list(LR_GRID), "weight_decay": list(WD_GRID)},
        "directions": dict(DIRECTIONS),
        "selection": "sum of within-direction ranks; ties -> lower lr, then higher wd",
        "criterion": f"{paths.MONITOR_METRIC} (higher is better)",
        "scores": scores,
        "ranks": ranks,
        "rank_sum": rank_sum,
        "fixed": {
            "input_transform": "none (raw UNI-v1 patch features)",
            "embed_dim": 512,
            "attn_dim": 256,
            "dropout": paths.DROPOUT,
            "max_epochs": paths.MAX_EPOCHS,
            "early_stopping_patience": paths.ES_PATIENCE,
            "min_epoch_before_stop": paths.MIN_EPOCH_BEFORE_STOP,
            "monitor": paths.MONITOR_METRIC,
            "oversized_bag_fallback": paths.OVERSIZED_BAG_FALLBACK,
            "seed": paths.PRIMARY_SEED,
        },
        "caveat": (
            "The search sees the labels of all 909 development patients, so the "
            "development OOF AUROC it feeds is mildly optimistic. External "
            "estimates are unaffected: no external cohort or outer fold is read here."
        ),
    }
    if args.apply:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nLocked {REPORT_PATH}")
    else:
        print("\nDry run — pass --apply to lock the winner.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="train the 12 search runs")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="rank the configurations and lock the winner")
    report.add_argument("--apply", action="store_true")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
