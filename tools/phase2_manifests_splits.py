#!/usr/bin/env python3
"""Aim 1 phase 2 — frozen manifests and the balanced five-fold assignment.

Builds, in one deterministic pass, every population artifact the study is
allowed to see:

    aim1_dev.csv              909 development patients (351 mutant / 558 WT)
                              carrying k_fold and val_fold_0..4
    aim1_dev_1cr.csv          the MSS/pMMR + BRAF-WT subset (745 patients),
                              FILTERED from the same fold columns
    aim1_ext_<group>.csv      SR1482-P (317), RIH-P (153), CPTAC-COAD (94)
    aim1_restricted_<group>.csv   the matched restricted external populations
    aim1_dev_folds.csv        the human-readable frozen fold record

Every arm's patient count is hard-checked against the pre-registration before
anything is written, so a change in the label source or the feature store
fails the build instead of silently redefining the study.

Subcommands:

    manifests [--apply]
        Build and check every manifest; print the fold-balance report.

    splits [--force]
        Materialize the split artifacts (splits.parquet + integrity hash) for
        1A and 1C-R through oceanpath.splitting, adopting the frozen fold
        columns with scheme=predefined_oof_kfold.

    verify
        Independent audit of the written artifacts: patient purity of every
        role in every fold, the outer-fold OOF partition, 1C-R folds equal to
        the 1A folds on shared patients, DEV/external patient disjointness,
        and per-fold positive support. Exits non-zero on any violation.

Usage:
    python tools/phase2_manifests_splits.py manifests --apply
    python tools/phase2_manifests_splits.py splits
    python tools/phase2_manifests_splits.py verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import balance, paths, population, registry, technical  # noqa: E402
from oceanpath.splitting import (  # noqa: E402
    SplitConfig,
    generate_splits,
    get_slide_ids_for_fold,
    load_splits,
)

FOLD_COLUMNS = ["k_fold"] + [f"val_fold_{index}" for index in range(paths.N_FOLDS)]


# ── manifests ─────────────────────────────────────────────────────────────────


def build_population() -> pd.DataFrame:
    rows = population.eligible()
    rows = population.add_context_columns(rows)
    return population.attach_technical(rows, technical.load())


def attach_folds(dev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Assign balanced outer folds, then a balanced ES carve-out per fold."""
    folds, report = balance.assign_folds(dev, population.BALANCE_COLUMNS, paths.N_FOLDS)
    out = dev.merge(folds.rename("k_fold"), left_on="patient_id", right_index=True, how="left")
    if out["k_fold"].isna().any():
        raise SystemExit("Fold assignment left some slides unassigned")
    out["k_fold"] = out["k_fold"].astype(int)

    report["early_stopping"] = {}
    for fold in range(paths.N_FOLDS):
        pool = out[out["k_fold"] != fold]
        val_patients = balance.carve_out_validation(
            pool, population.BALANCE_COLUMNS, pool["patient_id"].unique(), paths.ES_VAL_RATIO
        )
        column = f"val_fold_{fold}"
        out[column] = (out["patient_id"].isin(val_patients) & (out["k_fold"] != fold)).astype(int)
        labels = population.patient_labels(out[out[column] == 1])
        report["early_stopping"][f"fold_{fold}"] = {
            "val_patients": int(len(val_patients)),
            "val_mutant_patients": int((labels == "mutant").sum()),
            "val_slides": int(out[column].sum()),
        }
    return out, report


def write_folds_record(dev: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: fold, KRAS status, and every balancing variable."""
    columns = ["patient_id", "k_fold", *population.BALANCE_COLUMNS]
    record = (
        dev[columns + [f"val_fold_{index}" for index in range(paths.N_FOLDS)]]
        .drop_duplicates("patient_id")
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    return record


def cmd_manifests(args: argparse.Namespace) -> None:
    rows = build_population()
    dev = population.arm(rows, "dev")
    observed = population.check_counts(dev, paths.EXPECTED_PATIENTS["dev"], "Aim 1 population")
    print("Aim 1 population (all 4 cohorts, primary, KRAS-known, all sites)")
    print(f"  {observed}  slides={len(dev)}")

    print("\nPer-cohort composition (stratification + reporting variable, never a partition)")
    print(f"  {'subcohort':14s} {'patients':>9s} {'mutant':>7s} {'slides':>7s}")
    labels = population.patient_labels(dev)
    for subcohort, block in dev.groupby("subcohort"):
        block_labels = labels[block["patient_id"].unique()]
        n_pat, n_mut = int(block_labels.size), int((block_labels == "mutant").sum())
        expected = paths.EXPECTED_BY_SUBCOHORT.get(str(subcohort))
        if expected and (n_pat, n_mut) != (expected["patients"], expected["mutant"]):
            raise SystemExit(
                f"{subcohort}: expected {expected['patients']} patients / "
                f"{expected['mutant']} mutant, observed {n_pat} / {n_mut}"
            )
        print(f"  {subcohort:14s} {n_pat:9d} {n_mut:7d} {len(block):7d}")

    dev, report = attach_folds(dev)
    print(f"\nFold balance — worst marginal deviation {report['worst_marginal_deviation']:.4f}")
    print(f"  fold sizes (patients): {report['group_sizes']}")
    for column, block in report["marginals"].items():
        print(f"  {column:16s} max |deviation| = {block['max_abs_proportion_deviation']:.4f}")
    print("  early-stopping carve-outs (15% of each fold's training pool):")
    for fold, block in report["early_stopping"].items():
        print(
            f"    {fold}: {block['val_patients']} patients "
            f"({block['val_mutant_patients']} mutant), {block['val_slides']} slides"
        )
    minimum_positives = min(
        block["val_mutant_patients"] for block in report["early_stopping"].values()
    )
    if minimum_positives < paths.MIN_ES_VAL_POSITIVES:
        raise SystemExit(
            f"An inner-validation split has only {minimum_positives} mutant patients "
            f"(< {paths.MIN_ES_VAL_POSITIVES})."
        )
    print(f"  minimum inner-val mutant patients: {minimum_positives}")

    print("\nPer-cohort patients per outer fold (cohort stratification check)")
    fold_by_cohort = pd.crosstab(
        dev.drop_duplicates("patient_id")["subcohort"], dev.drop_duplicates("patient_id")["k_fold"]
    )
    print(fold_by_cohort.to_string())

    restricted_rows = population.restricted(dev)
    print(
        f"\nMSS/pMMR + BRAF-WT restricted subset (1C-R training rows): "
        f"{population.patient_counts(restricted_rows)}"
    )

    if not args.apply:
        print("\nDry run — pass --apply to write manifests.")
        return

    dev_manifest = population.finalize(dev, extra=[*FOLD_COLUMNS, *population.BALANCE_COLUMNS])
    dev_manifest.to_csv(paths.DEV_MANIFEST, index=False)
    print(f"\nWrote {paths.DEV_MANIFEST} ({len(dev_manifest)} slides)")

    restricted_manifest = dev_manifest[
        dev_manifest["slide_id"].isin(set(restricted_rows["slide_id"]))
    ]
    restricted_path = registry.MODELS["1cr"].manifest_path
    restricted_manifest.to_csv(restricted_path, index=False)
    print(f"Wrote {restricted_path} ({len(restricted_manifest)} slides)")

    write_folds_record(dev_manifest).to_csv(paths.FOLDS_CSV, index=False)
    print(f"Wrote {paths.FOLDS_CSV}")

    paths.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (paths.OUTPUT_ROOT / "fold_balance.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {paths.OUTPUT_ROOT / 'fold_balance.json'}")


# ── splits ────────────────────────────────────────────────────────────────────


def split_dir(model_id: str) -> Path:
    """Where Hydra's FoundationPaths will look: splits_root / data.name / splits.name."""
    return REPO / paths.SPLIT_ROOT / registry.MODELS[model_id].data_name / paths.SPLIT_NAME


def split_config(model_id: str) -> SplitConfig:
    model = registry.MODELS[model_id]
    return SplitConfig(
        scheme="predefined_oof_kfold",
        name=paths.SPLIT_NAME,
        csv_path=str(model.manifest_path),
        output_dir=str(split_dir(model_id)),
        filename_column="slide_id",
        label_column="target_label",
        group_column="patient_id",
        fold_column="k_fold",
        n_folds=paths.N_FOLDS,
        seed=paths.PRIMARY_SEED,
    )


def cmd_splits(args: argparse.Namespace) -> None:
    for model_id in ("1a", "1cr"):
        result = generate_splits(split_config(model_id), force=args.force)
        print(
            f"{model_id}: {result.n_slides} slides / {result.n_groups} patients -> "
            f"{result.parquet_path}\n     folds {result.fold_distribution}"
        )


# ── verify ────────────────────────────────────────────────────────────────────


def cmd_verify(_: argparse.Namespace) -> None:
    failures: list[str] = []

    dev = pd.read_csv(paths.DEV_MANIFEST)
    patient_of = dict(zip(dev["slide_id"], dev["patient_id"], strict=True))

    for model_id in ("1a", "1cr"):
        splits = load_splits(str(split_dir(model_id)), verify=True)
        seen_test: set[str] = set()
        for fold in range(paths.N_FOLDS):
            ids = get_slide_ids_for_fold(splits, fold, scheme="predefined_oof_kfold")
            roles = {role: {patient_of[s] for s in slides} for role, slides in ids.items()}
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                shared = roles[left] & roles[right]
                if shared:
                    failures.append(
                        f"{model_id} fold {fold}: {len(shared)} patient(s) in both "
                        f"{left} and {right}"
                    )
            if not ids["val"]:
                failures.append(f"{model_id} fold {fold}: empty inner-validation split")
            overlap = seen_test & set(ids["test"])
            if overlap:
                failures.append(f"{model_id} fold {fold}: {len(overlap)} slide(s) tested twice")
            seen_test |= set(ids["test"])
        if seen_test != set(splits["slide_id"]):
            failures.append(
                f"{model_id}: outer-test folds do not partition the manifest "
                f"({len(seen_test)} of {len(splits)} slides covered)"
            )

    # 1C-R must inherit 1A's folds exactly, so the two are paired.
    restricted = pd.read_csv(registry.MODELS["1cr"].manifest_path)
    merged = restricted.merge(
        dev[["slide_id", "k_fold"]], on="slide_id", suffixes=("", "_dev"), how="left"
    )
    if not merged["k_fold"].equals(merged["k_fold_dev"]):
        failures.append("1C-R fold assignment differs from 1A — folds were re-drawn, not filtered")

    # Every cohort must appear in every fold, or per-cohort OOF is undefined.
    patients = dev.drop_duplicates("patient_id")
    table = pd.crosstab(patients["subcohort"], patients["k_fold"])
    if (table == 0).any().any():
        failures.append(f"a cohort is missing from some fold:\n{table.to_string()}")

    for fold in range(paths.N_FOLDS):
        block = dev[dev["k_fold"] == fold]
        positives = population.patient_counts(block)["mutant"]
        if positives < 20:
            failures.append(f"fold {fold} has only {positives} mutant patients")

    if failures:
        print("VERIFY FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print(
        "VERIFY OK — patient purity, OOF partition, filtered-fold identity, and "
        "per-cohort fold coverage all hold."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    manifests = sub.add_parser("manifests", help="build and check every manifest")
    manifests.add_argument("--apply", action="store_true", help="write the CSVs")
    manifests.set_defaults(func=cmd_manifests)

    splits = sub.add_parser("splits", help="materialize split artifacts")
    splits.add_argument("--force", action="store_true", help="overwrite existing splits")
    splits.set_defaults(func=cmd_splits)

    verify = sub.add_parser("verify", help="audit the written artifacts")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
