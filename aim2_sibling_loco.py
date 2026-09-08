#!/usr/bin/env python3
"""E2a-D -- final-v8 sibling-stratum leave-one-domain-out transport.

This is an additive runner.  It does not alter or retrain the four frozen
source-family LOCO directions in :mod:`aim2_loco_transport`.  It trains the four
new final-v8 directions whose source pools retain a related acquisition:

    held out       related acquisition retained
    SR386          SR1482
    SR1482         SR386
    TCGA-COAD      TCGA-READ
    TCGA-READ      TCGA-COAD

For each direction and seed, five source-CV fits emit honest source OOF logits
for one source-only Platt map and one full-source refit runs for exactly 6,060
optimizer steps.  The target is never used for fitting, checkpoint selection,
calibration, or ensemble weighting.

All mutable artifacts are lineage-local below
``outputs/aim1/reruns/$OCEANPATH_AIM2_LINEAGE/e2ad``.  The governed label master
and the frozen final-v7 family-LOCO lineage are read-only inputs.

Typical execution::

    export OCEANPATH_AIM2_LINEAGE=aim2_final_v8_e2ad_v1_20260822
    python aim2_sibling_loco.py plan
    python aim2_sibling_loco.py manifests --apply
    python aim2_sibling_loco.py preflight --deep
    python aim2_sibling_loco.py source-cv
    python aim2_sibling_loco.py train
    python aim2_sibling_loco.py score
    python aim2_sibling_loco.py calibrate
    python aim2_sibling_loco.py report

``--arm`` and ``--seed`` restrict the expensive commands for scheduling.  A
partial or failed immutable run is never resumed or overwritten; select a new
lineage for a retry.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

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


@dataclass(frozen=True)
class ArmSpec:
    slug: str
    target_subcohort: str
    related_retained: str
    expected_source_patients: int
    expected_source_mutant: int
    expected_target_patients: int
    expected_target_mutant: int


ARMS: dict[str, ArmSpec] = {
    "sr386": ArmSpec("sr386", "SR386", "SR1482", 1_073, 457, 413, 147),
    "sr1482": ArmSpec("sr1482", "SR1482", "SR386", 1_162, 457, 324, 147),
    "tcga_coad": ArmSpec("tcga_coad", "TCGA-COAD", "TCGA-READ", 1_112, 444, 374, 160),
    "tcga_read": ArmSpec("tcga_read", "TCGA-READ", "TCGA-COAD", 1_358, 557, 128, 47),
}
ARM_NAMES: tuple[str, ...] = tuple(ARMS)
SEEDS: tuple[int, ...] = (42, 43, 44)
CAP = 8_192
STEP_BUDGET = 6_060
MIN_EPOCHS = 3
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_817
GOVERNED_LABEL_SOURCE = paths.MANIFEST_ROOT / "crc_final_v5.csv"
FAMILY_ROOT = paths.OUTPUT_ROOT / "reruns/aim2_cap8192_v4_20260819/e2a"
FAMILY_TARGET_FOR_ARM = {
    "sr386": "surgen",
    "sr1482": "surgen",
    "tcga_coad": "tcga",
    "tcga_read": "tcga",
}
EXPECTED_CONVENTIONAL = {"n": 1_486, "mutant": 604, "wild_type": 882}
FOLD_COLUMNS = ["k_fold", *[f"val_fold_{fold}" for fold in range(paths.N_FOLDS)]]


def component_root() -> Path:
    return lineage.component_root("e2ad")


def manifest_root() -> Path:
    return component_root() / "inputs/manifests"


def source_manifest(arm: str) -> Path:
    return manifest_root() / f"aim1_e2ad_{arm}_source.csv"


def target_manifest(arm: str) -> Path:
    return manifest_root() / f"aim1_e2ad_{arm}_target_primary.csv"


def manifest_contract_path() -> Path:
    return component_root() / "inputs/manifest_contract.json"


def split_root() -> Path:
    return component_root() / "inputs/splits"


def data_name(arm: str) -> str:
    return f"e2ad_{arm}"


def split_dir(arm: str) -> Path:
    return split_root() / f"aim1_{data_name(arm)}" / paths.SPLIT_NAME


def run_dir(arm: str, seed: int) -> Path:
    return component_root() / "train" / f"pb_cap{CAP}" / arm / f"seed{seed}"


def model_ckpt(arm: str, seed: int) -> Path:
    return run_dir(arm, seed) / "final/refit/model.ckpt"


def fit_summary_path(arm: str, seed: int) -> Path:
    return run_dir(arm, seed) / "fit_summary.json"


def source_cv_dir(arm: str, seed: int) -> Path:
    return component_root() / "source_cv" / f"cap{CAP}" / arm / f"seed{seed}"


def source_cv_receipt_path(arm: str, seed: int) -> Path:
    return source_cv_dir(arm, seed) / "e2ad_completion_receipt.json"


def score_path(arm: str, seed: int) -> Path:
    return component_root() / "scores" / f"pb_cap{CAP}_{arm}_seed{seed}_primary.parquet"


def score_receipt_path(arm: str, seed: int) -> Path:
    return score_path(arm, seed).with_suffix(".receipt.json")


def inference_environment_path() -> Path:
    return component_root() / "inputs/inference_environment.json"


def analysis_environment_path() -> Path:
    return component_root() / "inputs/analysis_environment.json"


def calibrator_path(arm: str) -> Path:
    return component_root() / "calibrators" / f"cap{CAP}_{arm}.json"


def calibrated_path(arm: str) -> Path:
    return component_root() / "calibrated" / f"cap{CAP}_{arm}_primary.parquet"


def calibrated_receipt_path(arm: str) -> Path:
    return calibrated_path(arm).with_suffix(".receipt.json")


def report_path() -> Path:
    return component_root() / "eval" / f"e2ad_sibling_loco_cap{CAP}.json"


def report_receipt_path() -> Path:
    return report_path().with_suffix(".receipt.json")


def _artifact(path: Path) -> dict[str, Any]:
    return lineage.artifact_identity(path)


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def _selected_arms(args: argparse.Namespace) -> list[str]:
    return [args.arm] if getattr(args, "arm", None) else list(ARM_NAMES)


def _selected_seeds(args: argparse.Namespace) -> list[int]:
    return [args.seed] if getattr(args, "seed", None) is not None else list(SEEDS)


def load_conventional_primary() -> pd.DataFrame:
    """Load final-v8's governed master and retain the frozen conventional census."""

    source = pd.read_csv(GOVERNED_LABEL_SOURCE, low_memory=False)
    rows = population.eligible(source)
    rows = rows.loc[rows["subcohort"].isin(paths.DEV_SUBCOHORTS)].copy()
    rows = population.attach_technical(population.add_context_columns(rows), technical.load())
    population.check_counts(rows, EXPECTED_CONVENTIONAL, "final-v8 conventional primary")
    if set(rows["subcohort"].unique()) != set(paths.DEV_SUBCOHORTS):
        raise RuntimeError("Conventional acquisition-domain census changed")
    return rows


def build_arm_frames(primary: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = ARMS[arm]
    target = primary.loc[primary["subcohort"].eq(spec.target_subcohort)].copy()
    source = primary.loc[~primary["subcohort"].eq(spec.target_subcohort)].copy()
    expected_source = {
        "n": spec.expected_source_patients,
        "mutant": spec.expected_source_mutant,
        "wild_type": spec.expected_source_patients - spec.expected_source_mutant,
    }
    expected_target = {
        "n": spec.expected_target_patients,
        "mutant": spec.expected_target_mutant,
        "wild_type": spec.expected_target_patients - spec.expected_target_mutant,
    }
    population.check_counts(source, expected_source, f"{arm} source")
    population.check_counts(target, expected_target, f"{arm} target")
    source_patients = set(source["patient_id"].astype(str))
    target_patients = set(target["patient_id"].astype(str))
    overlap = source_patients & target_patients
    if overlap:
        raise RuntimeError(f"{arm}: source-target patient overlap: {sorted(overlap)[:5]}")
    if spec.related_retained not in set(source["subcohort"].astype(str)):
        raise RuntimeError(f"{arm}: related acquisition {spec.related_retained} is absent")
    if spec.target_subcohort in set(source["subcohort"].astype(str)):
        raise RuntimeError(f"{arm}: target acquisition leaked into source")
    return source, target


def add_fold_columns(source: pd.DataFrame) -> pd.DataFrame:
    """Reproduce E2a's deterministic patient-grouped source-CV layout."""

    from sklearn.model_selection import StratifiedGroupKFold

    out = source.copy()
    patients = out.drop_duplicates("patient_id")
    strata = patients["cohort"].astype(str) + "|" + patients["target_label"].astype(str)
    splitter = StratifiedGroupKFold(
        n_splits=paths.N_FOLDS, shuffle=True, random_state=paths.PRIMARY_SEED
    )
    fold_of: dict[str, int] = {}
    for fold, (_, test_indices) in enumerate(
        splitter.split(patients, strata, groups=patients["patient_id"])
    ):
        for patient_id in patients.iloc[test_indices]["patient_id"]:
            fold_of[str(patient_id)] = fold
    out["k_fold"] = out["patient_id"].astype(str).map(fold_of).astype(int)
    for fold in range(paths.N_FOLDS):
        pool = out.loc[out["k_fold"].ne(fold)]
        validation_patients = balance.carve_out_validation(
            pool,
            population.BALANCE_COLUMNS,
            pool["patient_id"].unique(),
            paths.ES_VAL_RATIO,
        )
        out[f"val_fold_{fold}"] = (
            out["patient_id"].isin(validation_patients) & out["k_fold"].ne(fold)
        ).astype(int)
    return population.finalize(out, extra=[*population.BALANCE_COLUMNS, *FOLD_COLUMNS])


def _source_files() -> tuple[str, ...]:
    files = {
        "aim2_sibling_loco.py",
        "pyproject.toml",
        "tools/study_train.py",
        "uv.lock",
    }
    files.update(str(path.relative_to(REPO)) for path in (REPO / "configs").rglob("*.yaml"))
    files.update(str(path.relative_to(REPO)) for path in (REPO / "src/oceanpath").rglob("*.py"))
    return tuple(sorted(files))


def _snapshot_sources(destination: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for relative in _source_files():
        source = REPO / relative
        if not source.is_file():
            raise FileNotFoundError(f"Required E2a-D source is missing: {source}")
        frozen = destination / relative
        frozen.parent.mkdir(parents=True, exist_ok=True)
        lineage.ensure_absent(frozen)
        shutil.copy2(source, frozen)
        identity = _artifact(frozen)
        entries.append(
            {
                "relative_path": relative,
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )
    return entries


def _validate_source_snapshot(entries: Iterable[dict[str, Any]], root: Path) -> None:
    observed = list(entries)
    expected_paths = set(_source_files())
    recorded_paths = {str(entry.get("relative_path")) for entry in observed}
    if recorded_paths != expected_paths:
        raise RuntimeError(
            "E2a-D source snapshot inventory mismatch: "
            f"missing={sorted(expected_paths - recorded_paths)}, "
            f"unexpected={sorted(recorded_paths - expected_paths)}"
        )
    for entry in observed:
        relative = str(entry["relative_path"])
        live = REPO / relative
        frozen = root / relative
        if not live.is_file() or not frozen.is_file():
            raise RuntimeError(f"Missing live or snapshotted source: {relative}")
        expected = {
            "size_bytes": int(entry["size_bytes"]),
            "sha256": str(entry["sha256"]),
        }
        for label, path in (("live", live), ("snapshot", frozen)):
            identity = _artifact(path)
            observed_identity = {
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
            if observed_identity != expected:
                raise RuntimeError(f"E2a-D {label} source changed after seal: {relative}")


def _pack_identity(primary: pd.DataFrame, *, deep: bool) -> dict[str, Any]:
    pack = paths.PACKED_FEATURE_DIR.resolve()
    meta_path = pack / "meta.json"
    index_path = pack / "index.parquet"
    index = pd.read_parquet(index_path, columns=["slide_id"])
    expected = set(primary["slide_id"].astype(str))
    indexed = set(index["slide_id"].astype(str))
    missing = sorted(expected - indexed)
    if missing:
        raise RuntimeError(
            f"UNI-v1 pack is missing {len(missing)} conventional slides: {missing[:5]}"
        )
    payloads: dict[str, Any] = {}
    for name in ("features.bin", "coords.bin"):
        path = pack / name
        stat = path.stat()
        item: dict[str, Any] = {
            "path": str(path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        if deep:
            item["sha256"] = lineage.sha256_file(path)
        payloads[name] = item
    return {
        "root": str(pack),
        "meta": _artifact(meta_path),
        "index": _artifact(index_path),
        "payloads": payloads,
        "conventional_slide_count": len(expected),
        "conventional_missing_from_index": 0,
    }


def _validate_manifest_frame(frame: pd.DataFrame, *, arm: str, role: str) -> None:
    required = {"slide_id", "patient_id", "target_label", "subcohort", "specimen_role"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{arm}/{role}: manifest columns missing {sorted(missing)}")
    if frame["slide_id"].isna().any() or frame["slide_id"].duplicated().any():
        raise RuntimeError(f"{arm}/{role}: slide identifiers are null or duplicated")
    if set(frame["target_label"].astype(int).unique()) != {0, 1}:
        raise RuntimeError(f"{arm}/{role}: binary target labels are incomplete")
    if set(frame["specimen_role"].dropna().astype(str)) != {"primary"}:
        raise RuntimeError(f"{arm}/{role}: manifest is not primary-only")


def _validate_folds(frame: pd.DataFrame, *, arm: str) -> None:
    if not set(FOLD_COLUMNS).issubset(frame.columns):
        raise RuntimeError(f"{arm}: source manifest lacks frozen fold columns")
    if set(frame["k_fold"].astype(int).unique()) != set(range(paths.N_FOLDS)):
        raise RuntimeError(f"{arm}: source folds are not exactly 0..4")
    patient_folds = frame.groupby("patient_id")["k_fold"].nunique()
    if not patient_folds.eq(1).all():
        raise RuntimeError(f"{arm}: patients cross source-CV outer folds")
    for fold in range(paths.N_FOLDS):
        validation = frame[f"val_fold_{fold}"].astype(int)
        if not set(validation.unique()).issubset({0, 1}):
            raise RuntimeError(f"{arm}: val_fold_{fold} is not binary")
        if bool((validation.eq(1) & frame["k_fold"].eq(fold)).any()):
            raise RuntimeError(f"{arm}: fold {fold} test patients enter validation")
        patient_values = frame.groupby("patient_id")[f"val_fold_{fold}"].nunique()
        if not patient_values.eq(1).all():
            raise RuntimeError(f"{arm}: patients cross validation boundary for fold {fold}")


def _validate_contract(*, deep: bool = False) -> dict[str, Any]:
    path = manifest_contract_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing E2a-D manifest contract: {path}")
    contract = json.loads(path.read_text())
    expected_header = {
        "schema_version": 1,
        "experiment": "E2a-D sibling-stratum LOCO",
        "lineage": lineage.lineage_name(),
        "cap": CAP,
        "optimizer_step_budget": STEP_BUDGET,
        "seeds": list(SEEDS),
        "n_folds": paths.N_FOLDS,
        "governed_label_source": _artifact(GOVERNED_LABEL_SOURCE),
    }
    mismatches = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in expected_header.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"E2a-D manifest contract mismatch: {mismatches}")
    source_snapshot = contract.get("source_snapshot") or []
    snapshot_root = component_root() / "inputs/source_snapshot"
    _validate_source_snapshot(source_snapshot, snapshot_root)
    if set(contract.get("arms", {})) != set(ARM_NAMES):
        raise RuntimeError("E2a-D manifest contract has an incomplete arm roster")
    all_target_patients: dict[str, set[str]] = {}
    for arm, spec in ARMS.items():
        entry = contract["arms"][arm]
        for key, file_path in (
            ("source_manifest", source_manifest(arm)),
            ("target_manifest", target_manifest(arm)),
            ("splits", split_dir(arm) / "splits.parquet"),
        ):
            if entry.get(key) != _artifact(file_path):
                raise RuntimeError(f"{arm}: {key} identity changed after seal")
        source = pd.read_csv(source_manifest(arm), low_memory=False)
        target = pd.read_csv(target_manifest(arm), low_memory=False)
        _validate_manifest_frame(source, arm=arm, role="source")
        _validate_manifest_frame(target, arm=arm, role="target")
        _validate_folds(source, arm=arm)
        expected_source = {
            "n": spec.expected_source_patients,
            "mutant": spec.expected_source_mutant,
            "wild_type": spec.expected_source_patients - spec.expected_source_mutant,
        }
        expected_target = {
            "n": spec.expected_target_patients,
            "mutant": spec.expected_target_mutant,
            "wild_type": spec.expected_target_patients - spec.expected_target_mutant,
        }
        population.check_counts(source, expected_source, f"{arm} sealed source")
        population.check_counts(target, expected_target, f"{arm} sealed target")
        source_patients = set(source["patient_id"].astype(str))
        target_patients = set(target["patient_id"].astype(str))
        if source_patients & target_patients:
            raise RuntimeError(f"{arm}: sealed source and target overlap")
        if set(target["subcohort"].astype(str)) != {spec.target_subcohort}:
            raise RuntimeError(f"{arm}: sealed target has the wrong acquisition stratum")
        if spec.target_subcohort in set(source["subcohort"].astype(str)):
            raise RuntimeError(f"{arm}: target stratum leaked into sealed source")
        if spec.related_retained not in set(source["subcohort"].astype(str)):
            raise RuntimeError(f"{arm}: related sibling is absent from sealed source")
        all_target_patients[arm] = target_patients
    if all_target_patients["sr386"] & all_target_patients["sr1482"]:
        raise RuntimeError("SurGen sibling targets overlap by patient")
    if all_target_patients["tcga_coad"] & all_target_patients["tcga_read"]:
        raise RuntimeError("TCGA sibling targets overlap by patient")
    recorded_pack = contract.get("packed_store") or {}
    shallow_pack = _pack_identity(load_conventional_primary(), deep=False)
    for key in (
        "root",
        "meta",
        "index",
        "conventional_slide_count",
        "conventional_missing_from_index",
    ):
        if recorded_pack.get(key) != shallow_pack.get(key):
            raise RuntimeError(f"UNI-v1 packed-store {key} changed after seal")
    for name, item in shallow_pack["payloads"].items():
        recorded = recorded_pack.get("payloads", {}).get(name, {})
        for key in ("path", "size_bytes", "mtime_ns"):
            if recorded.get(key) != item.get(key):
                raise RuntimeError(f"UNI-v1 packed payload changed: {name}/{key}")
        if deep:
            expected_hash = recorded.get("sha256")
            if not expected_hash:
                raise RuntimeError("Deep preflight requires content hashes in the pack seal")
            if lineage.sha256_file(Path(item["path"])) != expected_hash:
                raise RuntimeError(f"UNI-v1 packed payload hash changed: {name}")
    return contract


def cmd_plan(_: argparse.Namespace) -> None:
    primary = load_conventional_primary()
    print("E2a-D -- four final-v8 sibling-stratum LOCO directions\n")
    print(
        f"{'held out':12s} {'source pts':>10s} {'source mut':>10s} "
        f"{'target pts':>10s} {'target mut':>10s} {'retained':>12s} {'refit epochs':>13s}"
    )
    for arm, spec in ARMS.items():
        source, target = build_arm_frames(primary, arm)
        source_counts = population.patient_counts(source)
        target_counts = population.patient_counts(target)
        epochs = epochs_for_n(source_counts["n"])
        print(
            f"{spec.target_subcohort:12s} {source_counts['n']:10d} "
            f"{source_counts['mutant']:10d} {target_counts['n']:10d} "
            f"{target_counts['mutant']:10d} {spec.related_retained:>12s} {epochs:13d}"
        )
    print("\n72 new MIL fits = 4 directions x 3 seeds x (5 source-CV fits + 1 full-source refit).")
    print(f"Every refit uses exactly {STEP_BUDGET:,} optimizer steps at cap {CAP:,}.")


def cmd_manifests(args: argparse.Namespace) -> None:
    lineage.lineage_name()
    primary = load_conventional_primary()
    built: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for arm in ARM_NAMES:
        source, target = build_arm_frames(primary, arm)
        built[arm] = (add_fold_columns(source), population.finalize(target))
        print(
            f"{arm:10s}: source {source.patient_id.nunique():4d} patients / "
            f"target {target.patient_id.nunique():3d} patients"
        )
    if not args.apply:
        print("Dry run -- pass --apply to seal lineage-local manifests and splits.")
        return
    destinations = [manifest_contract_path(), component_root() / "inputs/source_snapshot"]
    for arm in ARM_NAMES:
        destinations.extend([source_manifest(arm), target_manifest(arm), split_dir(arm)])
    occupied = [path for path in destinations if path.exists() or path.is_symlink()]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite a partial/existing E2a-D input seal: "
            + ", ".join(map(str, occupied[:5]))
        )
    for arm, (source, target) in built.items():
        lineage.write_text_once(source_manifest(arm), source.to_csv(index=False))
        lineage.write_text_once(target_manifest(arm), target.to_csv(index=False))
        ensure_splits(arm)
    snapshot_root = component_root() / "inputs/source_snapshot"
    source_snapshot = _snapshot_sources(snapshot_root)
    arms: dict[str, Any] = {}
    for arm, spec in ARMS.items():
        source, target = built[arm]
        arms[arm] = {
            "target_subcohort": spec.target_subcohort,
            "related_retained": spec.related_retained,
            "source": population.summarize(source),
            "target": population.summarize(target),
            "source_manifest": _artifact(source_manifest(arm)),
            "target_manifest": _artifact(target_manifest(arm)),
            "splits": _artifact(split_dir(arm) / "splits.parquet"),
        }
    contract = {
        "schema_version": 1,
        "created_utc": _utcnow(),
        "experiment": "E2a-D sibling-stratum LOCO",
        "lineage": lineage.lineage_name(),
        "cap": CAP,
        "optimizer_step_budget": STEP_BUDGET,
        "seeds": list(SEEDS),
        "n_folds": paths.N_FOLDS,
        "bootstrap": {"n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
        "governed_label_source": _artifact(GOVERNED_LABEL_SOURCE),
        "conventional_population": population.summarize(primary),
        "packed_store": _pack_identity(primary, deep=True),
        "source_snapshot": source_snapshot,
        "git_head": _git_output("rev-parse", "HEAD"),
        "git_status_porcelain": _git_output("status", "--porcelain=v1"),
        "arms": arms,
    }
    lineage.write_json_once(manifest_contract_path(), contract)
    _validate_contract(deep=False)
    print(f"Sealed E2a-D input contract: {manifest_contract_path()}")


def ensure_splits(arm: str) -> Path:
    from oceanpath.splitting import SplitConfig, generate_splits

    directory = split_dir(arm)
    split_file = directory / "splits.parquet"
    if split_file.is_file():
        return directory
    if directory.exists():
        raise RuntimeError(f"Refusing partial split directory: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    generate_splits(
        SplitConfig(
            scheme="predefined_oof_kfold",
            name=paths.SPLIT_NAME,
            csv_path=str(source_manifest(arm)),
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
    if not split_file.is_file():
        raise RuntimeError(f"Split generator did not publish {split_file}")
    return directory


def epochs_for_n(n_patients: int) -> int:
    if n_patients <= 0:
        raise ValueError("n_patients must be positive")
    return max(MIN_EPOCHS, math.ceil(STEP_BUDGET / n_patients))


def cmd_preflight(args: argparse.Namespace) -> None:
    contract = _validate_contract(deep=args.deep)
    print(
        f"PASS: E2a-D input contract is valid for {len(contract['arms'])} arms; "
        f"deep_pack_hashes={args.deep}"
    )


def _run_logged(command: list[str], log_path: Path) -> int:
    lineage.ensure_absent(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
            print(line, end="", flush=True)
            log_stream.write(line)
        return int(process.wait())


def _source_cv_request_path(arm: str, seed: int) -> Path:
    return component_root() / "requests/source_cv" / arm / f"seed{seed}.json"


def _source_cv_log_path(arm: str, seed: int) -> Path:
    return component_root() / "launcher_logs" / f"source_cv_{arm}_cap{CAP}_seed{seed}.log"


def _source_cv_request(arm: str, seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "requested",
        "created_utc": _utcnow(),
        "experiment": "E2a-D source CV",
        "lineage": lineage.lineage_name(),
        "arm": arm,
        "seed": seed,
        "cap": CAP,
        "n_folds": paths.N_FOLDS,
        "purpose": "honest source OOF logits for source-only Platt calibration",
        "inputs": {
            "manifest_contract": _artifact(manifest_contract_path()),
            "source_manifest": _artifact(source_manifest(arm)),
            "splits": _artifact(split_dir(arm) / "splits.parquet"),
        },
        "output_dir": str(source_cv_dir(arm, seed).resolve()),
    }


def _nested_value(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted)
        value = value[key]
    return value


def _source_cv_material_expectations(arm: str, seed: int) -> dict[str, Any]:
    """Material source-CV recipe that every completion identity must prove."""

    return {
        "data.name": f"aim1_{data_name(arm)}",
        "data.aim1_model": data_name(arm),
        "data.csv_path": str(source_manifest(arm).resolve()),
        "data.label_columns": ["target_label"],
        "data.patient_id_column": "patient_id",
        "data.filename_column": "slide_id",
        "data.cohort_column": "cohort",
        "data.num_classes": 2,
        "encoder.name": "uni_v1",
        "encoder.feature_dim": 1_024,
        "splits.scheme": "predefined_oof_kfold",
        "splits.name": paths.SPLIT_NAME,
        "splits.fold_column": "k_fold",
        "splits.group_column": "patient_id",
        "splits.allow_group_overlap": False,
        "splits.n_folds": paths.N_FOLDS,
        "splits.seed": seed,
        "model.name": "abmil",
        "model.arch": "abmil",
        "model.embed_dim": 512,
        "model.attn_dim": 384,
        "model.gate": True,
        "model.dropout": paths.DROPOUT,
        "model.input_dropout": 0.10,
        "training.lr": paths.LR,
        "training.weight_decay": paths.WEIGHT_DECAY,
        "training.lr_scheduler": "cosine",
        "training.final_lr_fraction": 0.01,
        "training.max_epochs": paths.MAX_EPOCHS,
        "training.batch_size": 1,
        "training.accumulate_grad_batches": 1,
        "training.loss_type": "bce",
        "training.class_weights": None,
        "training.training_class_weighted_loss": False,
        "training.validation_loss_weighted": False,
        "training.class_weighted_sampling": False,
        "training.max_instances": None,
        "training.dataset_max_instances": CAP,
        "training.eval_full_bags": True,
        "training.sample_weight_column": None,
        "training.train_sampling_strategy": "patient_natural",
        "training.force_float32": True,
        "training.packed_dir": str(paths.PACKED_FEATURE_DIR.resolve()),
        "training.verify_packed_source": False,
        "training.monitor_metric": paths.MONITOR_METRIC,
        "training.monitor_mode": paths.MONITOR_MODE,
        "training.early_stopping_patience": paths.ES_PATIENCE,
        "training.min_epoch_before_stop": paths.MIN_EPOCH_BEFORE_STOP,
        "training.es_min_val_positives": paths.MIN_ES_VAL_POSITIVES,
        "training.seed": seed,
        "training.skip_finalize": True,
        "training.refit_max_steps": None,
        "platform.precision": "bf16-mixed",
        "platform.accelerator": "auto",
        "platform.devices": 1,
    }


def _validate_source_cv_material_config(
    record: dict[str, Any], *, arm: str, seed: int, context: str
) -> None:
    mismatches: dict[str, dict[str, Any]] = {}
    for dotted, expected in _source_cv_material_expectations(arm, seed).items():
        try:
            observed = _nested_value(record, dotted)
        except KeyError:
            observed = "<missing>"
        if observed != expected:
            mismatches[dotted] = {"expected": expected, "observed": observed}
    if mismatches:
        raise RuntimeError(f"{arm}/seed{seed}: {context} recipe mismatch: {mismatches}")


def _validate_source_cv_training_identity(
    arm: str, seed: int, directory: Path, completion: dict[str, Any]
) -> None:
    """Bind a five-fold completion to the exact requested material recipe."""

    fold_evidence = completion.get("fold_completions")
    if (
        completion.get("n_folds") != paths.N_FOLDS
        or completion.get("skip_finalize") is not True
        or not isinstance(fold_evidence, list)
        or len(fold_evidence) != paths.N_FOLDS
    ):
        raise RuntimeError(
            f"{arm}/seed{seed}: source-CV completion is not exactly five fold-local fits"
        )
    identity_path = directory / "training_identity.json"
    identity = json.loads(identity_path.read_text())
    fingerprint = identity.get("fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != completion.get("training_fingerprint"):
        raise RuntimeError(f"{arm}/seed{seed}: training identity/completion fingerprints differ")
    payload = identity.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{arm}/seed{seed}: training identity payload is malformed")
    material = payload.get("material_config")
    evidence = payload.get("input_evidence")
    if not isinstance(material, dict) or not isinstance(evidence, dict):
        raise RuntimeError(f"{arm}/seed{seed}: training material/input evidence is malformed")
    _validate_source_cv_material_config(material, arm=arm, seed=seed, context="training identity")
    integrity_path = split_dir(arm) / ".integrity_hash"
    expected_evidence = {
        "manifest_sha256": _artifact(source_manifest(arm))["sha256"],
        "split_integrity_sha256": lineage.sha256_file(integrity_path),
    }
    evidence_mismatch = {
        key: {"expected": expected, "observed": evidence.get(key)}
        for key, expected in expected_evidence.items()
        if evidence.get(key) != expected
    }
    if evidence_mismatch:
        raise RuntimeError(
            f"{arm}/seed{seed}: training input evidence mismatch: {evidence_mismatch}"
        )
    for fold in range(paths.N_FOLDS):
        config_path = directory / f"fold_{fold}/config.yaml"
        config = yaml.safe_load(config_path.read_text())
        if not isinstance(config, dict):
            raise RuntimeError(f"{arm}/seed{seed}: malformed fold config: {config_path}")
        _validate_source_cv_material_config(
            config, arm=arm, seed=seed, context=f"fold {fold} resolved config"
        )
        expected_paths = {
            "platform.splits_root": str(split_root().resolve()),
            "train_dir": str(directory.resolve()),
        }
        path_mismatch: dict[str, dict[str, Any]] = {}
        for dotted, expected in expected_paths.items():
            try:
                observed = _nested_value(config, dotted)
            except KeyError:
                observed = "<missing>"
            if observed != expected:
                path_mismatch[dotted] = {"expected": expected, "observed": observed}
        resolved_splits = (
            Path(str(config["platform"]["splits_root"]))
            / str(config["data"]["name"])
            / str(config["splits"]["name"])
        ).resolve()
        if resolved_splits != split_dir(arm).resolve():
            path_mismatch["resolved_splits_dir"] = {
                "expected": str(split_dir(arm).resolve()),
                "observed": str(resolved_splits),
            }
        if path_mismatch:
            raise RuntimeError(
                f"{arm}/seed{seed}: fold {fold} input path mismatch: {path_mismatch}"
            )


def _validate_source_cv(arm: str, seed: int) -> dict[str, Any]:
    from oceanpath.workflows.training import validate_training_run_dir

    directory = source_cv_dir(arm, seed)
    receipt_path = source_cv_receipt_path(arm, seed)
    if not directory.is_dir() or not receipt_path.is_file():
        raise FileNotFoundError(f"Incomplete E2a-D source-CV chain: {directory}")
    completion = validate_training_run_dir(directory, require_test_predictions=True)
    _validate_source_cv_training_identity(arm, seed, directory, completion)
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "schema_version": 1,
        "status": "completed",
        "lineage": lineage.lineage_name(),
        "arm": arm,
        "seed": seed,
        "cap": CAP,
        "n_folds": paths.N_FOLDS,
        "request": _artifact(_source_cv_request_path(arm, seed)),
        "oof_predictions": _artifact(directory / "oof_predictions.parquet"),
        "training_completion": _artifact(directory / "training_completion.json"),
        "training_identity": _artifact(directory / "training_identity.json"),
        "cv_summary": _artifact(directory / "cv_summary.json"),
    }
    mismatches = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{arm}/seed{seed}: source-CV receipt mismatch: {mismatches}")
    manifest = pd.read_csv(source_manifest(arm), low_memory=False)
    oof = pd.read_parquet(directory / "oof_predictions.parquet")
    expected_slides = set(manifest["slide_id"].astype(str))
    observed_slides = set(oof["slide_id"].astype(str))
    if (
        observed_slides != expected_slides
        or oof["slide_id"].astype(str).duplicated().any()
        or len(oof) != len(manifest)
    ):
        raise RuntimeError(f"{arm}/seed{seed}: source OOF slide coverage is incomplete")
    patient = evaluate.to_patient_level(oof, manifest)
    expected_patients = set(manifest["patient_id"].astype(str))
    observed_patients = set(patient["patient_id"].astype(str))
    if observed_patients != expected_patients or len(patient) != len(expected_patients):
        raise RuntimeError(f"{arm}/seed{seed}: source OOF patient coverage is incomplete")
    if not np.isfinite(patient["mean_logit"].to_numpy(float)).all():
        raise RuntimeError(f"{arm}/seed{seed}: source OOF logits are non-finite")
    return receipt


def source_cv_one(arm: str, seed: int, *, dry_run: bool) -> None:
    _validate_contract(deep=False)
    directory = source_cv_dir(arm, seed)
    if directory.exists():
        try:
            _validate_source_cv(arm, seed)
        except Exception as exc:
            raise RuntimeError(
                f"Refusing to resume or overwrite partial immutable source-CV run {directory}; "
                "select a new lineage for the retry"
            ) from exc
        print(f"== source-CV {arm} seed{seed}: completed and hash-valid -- skipping")
        return
    request_path = _source_cv_request_path(arm, seed)
    log_path = _source_cv_log_path(arm, seed)
    if request_path.exists() or log_path.exists():
        raise RuntimeError(
            f"{arm}/seed{seed}: partial source-CV request/log exists; select a new lineage"
        )
    overrides = [
        "platform=colon_workstation",
        "data=aim1",
        f"data.aim1_model={data_name(arm)}",
        f"data.manifest_stem={source_manifest(arm).stem}",
        f"data.csv_path={source_manifest(arm)}",
        f"platform.splits_root={split_root()}",
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
        f"exp_name=e2ad_srccv_{arm}_c{CAP}_s{seed}",
        f"hydra.run.dir={component_root() / 'hydra_runs' / f'source_cv_{arm}_cap{CAP}_seed{seed}'}",
        "hydra.job.chdir=false",
    ]
    command = [
        sys.executable,
        str(REPO / "tools/study_train.py"),
        "hydra-train",
        *overrides,
    ]
    print(f"== source-CV {arm} seed{seed}: 5 fold-local fits")
    if dry_run:
        print("   " + " ".join(command))
        return
    request_path.parent.mkdir(parents=True, exist_ok=True)
    lineage.write_json_once(request_path, _source_cv_request(arm, seed))
    returncode = _run_logged(command, log_path)
    if returncode != 0:
        failure = request_path.with_suffix(".failure.json")
        lineage.write_json_once(
            failure,
            {
                "status": "failed",
                "finished_utc": _utcnow(),
                "returncode": returncode,
                "request": _artifact(request_path),
                "log": _artifact(log_path),
            },
        )
        raise SystemExit(
            f"source-CV {arm}/seed{seed} failed; evidence retained. Use a new lineage for retry."
        )
    from oceanpath.workflows.training import validate_training_run_dir

    validate_training_run_dir(directory, require_test_predictions=True)
    lineage.write_json_once(
        source_cv_receipt_path(arm, seed),
        {
            "schema_version": 1,
            "status": "completed",
            "finished_utc": _utcnow(),
            "lineage": lineage.lineage_name(),
            "arm": arm,
            "seed": seed,
            "cap": CAP,
            "n_folds": paths.N_FOLDS,
            "request": _artifact(request_path),
            "oof_predictions": _artifact(directory / "oof_predictions.parquet"),
            "training_completion": _artifact(directory / "training_completion.json"),
            "training_identity": _artifact(directory / "training_identity.json"),
            "cv_summary": _artifact(directory / "cv_summary.json"),
            "launcher_log": _artifact(log_path),
        },
    )
    _validate_source_cv(arm, seed)


def cmd_source_cv(args: argparse.Namespace) -> None:
    _validate_contract(deep=True)
    for arm in _selected_arms(args):
        for seed in _selected_seeds(args):
            source_cv_one(arm, seed, dry_run=args.dry_run)


def _completed_refit(arm: str, seed: int) -> dict[str, Any]:
    summary_path = fit_summary_path(arm, seed)
    checkpoint = model_ckpt(arm, seed)
    request_path = run_dir(arm, seed) / "run_request.json"
    config_path = run_dir(arm, seed) / "resolved_config.yaml"
    for path in (summary_path, checkpoint, request_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete E2a-D refit: {path}")
    summary = json.loads(summary_path.read_text())
    expected = {
        "schema_version": 1,
        "status": "completed",
        "lineage": lineage.lineage_name(),
        "arm": arm,
        "target_subcohort": ARMS[arm].target_subcohort,
        "seed": seed,
        "sampling_seed": seed,
        "cap": CAP,
        "optimizer_step_budget": STEP_BUDGET,
        "sampler": "patient_natural",
        "loss_weighting": "none",
        "model": _artifact(checkpoint),
        "resolved_config": _artifact(config_path),
        "run_request": _artifact(request_path),
    }
    result = summary.get("result") or {}
    result_expected = {
        "actual_optimizer_steps": STEP_BUDGET,
        "refit_max_steps": STEP_BUDGET,
        "batch_size": 1,
        "accumulate_grad_batches": 1,
        "seed": seed,
        "sampling_seed": seed,
        "lr_scheduler": "cosine",
        "lr_scheduler_interval": "step",
        "lr_scheduler_total_steps": STEP_BUDGET,
        "train_sampling_strategy": "patient_natural",
        "sample_weight_column": None,
        "class_weights": None,
        "dataset_max_instances": CAP,
        "max_instances": None,
        "eval_full_bags": True,
    }
    mismatches = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    mismatches.update(
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
        mismatches["result.final_learning_rates"] = {
            "expected": [1.0e-6],
            "observed": final_lrs,
        }
    if mismatches:
        raise RuntimeError(f"{arm}/seed{seed}: refit contract mismatch: {mismatches}")
    return summary


def _refit_request(arm: str, seed: int, epochs: int) -> dict[str, Any]:
    directory = run_dir(arm, seed)
    return {
        "schema_version": 1,
        "status": "requested",
        "created_utc": _utcnow(),
        "experiment": "E2a-D full-source refit",
        "lineage": lineage.lineage_name(),
        "arm": arm,
        "target_subcohort": ARMS[arm].target_subcohort,
        "seed": seed,
        "sampling_seed": seed,
        "cap": CAP,
        "optimizer_step_budget": STEP_BUDGET,
        "epoch_ceiling": epochs,
        "run_dir": str(directory.resolve()),
        "inputs": {
            "manifest_contract": _artifact(manifest_contract_path()),
            "source_manifest": _artifact(source_manifest(arm)),
            "splits": _artifact(split_dir(arm) / "splits.parquet"),
        },
    }


def train_one(arm: str, seed: int, *, dry_run: bool) -> None:
    _validate_contract(deep=False)
    directory = run_dir(arm, seed)
    if directory.exists():
        try:
            _completed_refit(arm, seed)
        except Exception as exc:
            raise RuntimeError(
                f"Refusing to resume or overwrite partial immutable refit {directory}; "
                "select a new lineage for the retry"
            ) from exc
        print(f"== refit {arm} seed{seed}: completed and hash-valid -- skipping")
        return
    n_patients = pd.read_csv(source_manifest(arm))["patient_id"].nunique()
    epochs = epochs_for_n(int(n_patients))
    print(
        f"== refit {arm} seed{seed}: {n_patients} source patients, "
        f"epoch ceiling {epochs}, exact {STEP_BUDGET} optimizer steps"
    )
    if dry_run:
        return
    directory.mkdir(parents=True, exist_ok=False)
    lineage.write_json_once(directory / "run_request.json", _refit_request(arm, seed, epochs))
    command = [
        sys.executable,
        str(REPO / "aim2_sibling_loco.py"),
        "_fit",
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--run-dir",
        str(directory),
    ]
    returncode = _run_logged(command, directory / "stdout_stderr.log")
    if returncode != 0:
        lineage.write_json_once(
            directory / "failure.json",
            {
                "status": "failed",
                "finished_utc": _utcnow(),
                "returncode": returncode,
                "command": command,
            },
        )
        raise SystemExit(
            f"refit {arm}/seed{seed} failed; evidence retained. Use a new lineage for retry."
        )
    _completed_refit(arm, seed)


def cmd_train(args: argparse.Namespace) -> None:
    _validate_contract(deep=True)
    for arm in _selected_arms(args):
        for seed in _selected_seeds(args):
            train_one(arm, seed, dry_run=args.dry_run)


def cmd_fit(args: argparse.Namespace) -> None:
    """Internal child command for one exact-step full-source refit."""

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from oceanpath.workflows.finalize import _run_refit

    arm, seed, epochs = args.arm, args.seed, args.epochs
    directory = Path(args.run_dir).resolve()
    expected_directory = run_dir(arm, seed).resolve()
    if directory != expected_directory:
        raise RuntimeError(
            f"Internal E2a-D run-dir mismatch: expected {expected_directory}, got {directory}"
        )
    request_path = directory / "run_request.json"
    if not request_path.is_file():
        raise FileNotFoundError(f"Missing immutable refit request: {request_path}")
    contract = _validate_contract(deep=False)
    request = json.loads(request_path.read_text())
    expected_request = _refit_request(arm, seed, epochs)
    expected_request.pop("created_utc")
    mismatches = {
        key: {"expected": value, "observed": request.get(key)}
        for key, value in expected_request.items()
        if request.get(key) != value
    }
    if request.get("inputs", {}).get("manifest_contract") != _artifact(manifest_contract_path()):
        mismatches["inputs.manifest_contract"] = "changed"
    if mismatches:
        raise RuntimeError(f"Immutable E2a-D refit request mismatch: {mismatches}")
    if contract["arms"][arm]["source_manifest"] != _artifact(source_manifest(arm)):
        raise RuntimeError(f"{arm}: source manifest differs from the input seal")
    overrides = [
        "platform=colon_workstation",
        "data=aim1",
        f"data.aim1_model={data_name(arm)}",
        f"data.manifest_stem={source_manifest(arm).stem}",
        f"data.csv_path={source_manifest(arm)}",
        f"platform.splits_root={split_root()}",
        "+data.cohort_column=cohort",
        "encoder=univ1",
        "splits=aim1_balanced",
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
        "training.refit_epoch_rule=median",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        f"train_dir={directory}",
        f"exp_name=e2ad_pb_{arm}_c{CAP}_s{seed}",
    ]
    with initialize_config_dir(config_dir=str((REPO / "configs").resolve()), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    with open_dict(cfg):
        cfg.training.max_epochs = epochs
        cfg.training.refit_max_steps = STEP_BUDGET
    lineage.write_text_once(
        directory / "resolved_config.yaml", OmegaConf.to_yaml(cfg, resolve=True)
    )
    final_dir = directory / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    result = _run_refit(cfg, final_dir, [{"best_epoch": epochs}])
    checkpoint = final_dir / "refit/model.ckpt"
    lineage.write_json_once(
        directory / "fit_summary.json",
        {
            "schema_version": 1,
            "status": "completed",
            "finished_utc": _utcnow(),
            "lineage": lineage.lineage_name(),
            "arm": arm,
            "target_subcohort": ARMS[arm].target_subcohort,
            "seed": seed,
            "sampling_seed": seed,
            "cap": CAP,
            "epoch_ceiling": epochs,
            "optimizer_step_budget": STEP_BUDGET,
            "sampler": "patient_natural",
            "loss_weighting": "none",
            "model": _artifact(checkpoint),
            "resolved_config": _artifact(directory / "resolved_config.yaml"),
            "run_request": _artifact(request_path),
            "result": result,
        },
    )
    print(f"refit complete: {arm}/seed{seed}; {STEP_BUDGET} optimizer steps")


def cmd_smoke(args: argparse.Namespace) -> None:
    """Run a short, explicitly non-governed refit below /tmp.

    The output path and completion marker cannot satisfy `_completed_refit`, so
    this CUDA/configuration check can never enter the 72-fit governed roster.
    """

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from oceanpath.workflows.finalize import _run_refit

    _validate_contract(deep=True)
    _current_inference_environment()
    arm, seed, steps = args.arm, args.seed, args.steps
    if steps <= 0 or steps > 10:
        raise ValueError("Smoke steps must be between 1 and 10")
    if args.output_dir is None:
        if args.dry_run:
            output = Path("/tmp/e2ad_non_governed_smoke_DRY_RUN")
        else:
            output = Path(tempfile.mkdtemp(prefix="e2ad_non_governed_smoke_", dir="/tmp"))
    else:
        output = args.output_dir.resolve()
        tmp_root = Path("/tmp").resolve()
        if not output.is_relative_to(tmp_root) or not output.name.startswith(
            "e2ad_non_governed_smoke_"
        ):
            raise ValueError("Smoke output must be /tmp/e2ad_non_governed_smoke_<name>")
        if output.exists():
            raise FileExistsError(f"Refusing existing smoke output: {output}")
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=False)
    overrides = [
        "platform=colon_workstation",
        "data=aim1",
        f"data.aim1_model={data_name(arm)}",
        f"data.manifest_stem={source_manifest(arm).stem}",
        f"data.csv_path={source_manifest(arm)}",
        f"platform.splits_root={split_root()}",
        "+data.cohort_column=cohort",
        "encoder=univ1",
        "splits=aim1_balanced",
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
        "training.refit_epoch_rule=median",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        f"train_dir={output}",
        f"exp_name=e2ad_NON_GOVERNED_SMOKE_{arm}_s{seed}",
    ]
    with initialize_config_dir(config_dir=str((REPO / "configs").resolve()), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    with open_dict(cfg):
        cfg.training.max_epochs = 1
        cfg.training.refit_max_steps = steps
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print(f"NON-GOVERNED smoke output would be {output}")
        return
    lineage.write_text_once(output / "resolved_config.yaml", OmegaConf.to_yaml(cfg, resolve=True))
    result = _run_refit(cfg, output / "fit", [{"best_epoch": 1}])
    marker = {
        "schema_version": 1,
        "status": "PASS_NON_GOVERNED_SMOKE_ONLY",
        "created_utc": _utcnow(),
        "governed_fit": False,
        "may_enter_e2ad_roster": False,
        "arm": arm,
        "seed": seed,
        "optimizer_steps": steps,
        "result": result,
        "config": _artifact(output / "resolved_config.yaml"),
    }
    lineage.write_json_once(output / "NON_GOVERNED_SMOKE.json", marker)
    print(f"PASS: non-governed {steps}-step CUDA smoke at {output}")


def _current_inference_environment() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Governed E2a-D inference requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Governed E2a-D inference requires CUDA bfloat16 support")
    sources = (
        Path(__file__).resolve(),
        REPO / "src/oceanpath/contracts/__init__.py",
        REPO / "src/oceanpath/contracts/slide_ids.py",
        REPO / "src/oceanpath/datasets/datamodule.py",
        REPO / "src/oceanpath/datasets/packed.py",
        REPO / "src/oceanpath/training/lightning.py",
        REPO / "src/oceanpath/models/__init__.py",
        REPO / "src/oceanpath/models/abmil.py",
        REPO / "src/oceanpath/models/base.py",
        REPO / "src/oceanpath/models/components.py",
        REPO / "src/oceanpath/models/wsi_classifier.py",
    )
    return {
        "device": "cuda",
        "device_index": 0,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "autocast": True,
        "autocast_dtype": "bfloat16",
        "force_float32_before_device_transfer": True,
        "batch_size": 1,
        "evaluation_bag": "full",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "lightning_version": importlib.metadata.version("lightning"),
        "python_version": platform.python_version(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "implementation_sources": [_artifact(path) for path in sources],
    }


def _seal_inference_environment(*, num_workers: int) -> dict[str, Any]:
    environment = _current_inference_environment()
    environment["num_workers"] = int(num_workers)
    path = inference_environment_path()
    if path.exists():
        if not path.is_file():
            raise RuntimeError(f"Invalid inference environment seal: {path}")
        recorded = json.loads(path.read_text())
        if recorded != environment:
            raise RuntimeError("Governed E2a-D inference environment changed after seal")
    else:
        lineage.write_json_once(path, environment)
    return environment


def _score_inputs(arm: str, seed: int) -> dict[str, Any]:
    _completed_refit(arm, seed)
    environment_path = inference_environment_path()
    if not environment_path.is_file():
        raise FileNotFoundError("Inference environment must be sealed before scoring")
    contract = _validate_contract(deep=False)
    return {
        "manifest_contract": _artifact(manifest_contract_path()),
        "checkpoint": _artifact(model_ckpt(arm, seed)),
        "target_manifest": _artifact(target_manifest(arm)),
        "packed_store": contract["packed_store"],
        "inference_environment": _artifact(environment_path),
        "arm": arm,
        "seed": seed,
        "cap": CAP,
        "evaluation_bag": "full",
    }


def _load_valid_score(arm: str, seed: int) -> pd.DataFrame | None:
    destination = score_path(arm, seed)
    receipt_path = score_receipt_path(arm, seed)
    if not destination.exists() and not receipt_path.exists():
        return None
    if not destination.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"Partial immutable score cache: {arm}/seed{seed}")
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "schema_version": 1,
        "lineage": lineage.lineage_name(),
        "inputs": _score_inputs(arm, seed),
        "artifact": _artifact(destination),
    }
    mismatches = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{arm}/seed{seed}: score receipt mismatch: {mismatches}")
    scores = pd.read_parquet(destination)
    manifest = pd.read_csv(target_manifest(arm))
    expected_slides = set(manifest["slide_id"].astype(str))
    if set(scores["slide_id"].astype(str)) != expected_slides:
        raise RuntimeError(f"{arm}/seed{seed}: score slide coverage changed")
    counts = scores.groupby("slide_id").size()
    if not counts.eq(1).all() or len(scores) != len(manifest):
        raise RuntimeError(f"{arm}/seed{seed}: expected one logit per target slide")
    if "logit" not in scores or not np.isfinite(scores["logit"].to_numpy(float)).all():
        raise RuntimeError(f"{arm}/seed{seed}: native logits are missing or non-finite")
    return scores


def _score_packed_checkpoint(arm: str, seed: int, *, num_workers: int) -> pd.DataFrame:
    """One native logit per target slide from the sealed UNI packed store."""

    import torch
    from torch.utils.data import DataLoader

    from oceanpath.datasets.datamodule import SimpleMILCollator, SlideDataset
    from oceanpath.datasets.packed import (
        PackedFeatureStore,
        feature_inventory_sha256,
    )
    from oceanpath.training.lightning import MILTrainModule

    manifest = pd.read_csv(target_manifest(arm), low_memory=False)
    live_inventory = feature_inventory_sha256(paths.PINNED_FEATURE_DIR)
    store = PackedFeatureStore(paths.PACKED_FEATURE_DIR, verify_source=live_inventory)
    slide_ids = manifest["slide_id"].astype(str).tolist()
    labels = dict(
        zip(
            manifest["slide_id"].astype(str),
            manifest["target_label"].astype(int),
            strict=True,
        )
    )
    dataset = SlideDataset(
        feature_dir=str(paths.PINNED_FEATURE_DIR),
        slide_ids=slide_ids,
        labels=labels,
        max_instances=None,
        is_train=False,
        force_float32=True,
        store=store,
    )
    missing = sorted(set(slide_ids) - set(dataset.slide_ids))
    if missing:
        raise RuntimeError(f"{arm}: packed dataset lacks target slides: {missing[:5]}")
    loader_kwargs: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": SimpleMILCollator(max_instances=None),
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs.update({"prefetch_factor": 2, "persistent_workers": True})
    loader = DataLoader(dataset, **loader_kwargs)
    checkpoint = model_ckpt(arm, seed)
    try:
        module = MILTrainModule.load_from_checkpoint(
            str(checkpoint), map_location="cuda", weights_only=False
        )
    except TypeError:
        module = MILTrainModule.load_from_checkpoint(str(checkpoint), map_location="cuda")
    module.eval().to("cuda")
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to("cuda", non_blocking=True)
            mask = (
                batch["mask"].to("cuda", non_blocking=True)
                if batch.get("mask") is not None
                else None
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = module.model(features, mask=mask)
            logits = output.logits.detach().float().cpu().numpy()
            for index, slide_id in enumerate(batch["slide_ids"]):
                values = np.atleast_1d(logits[index]).ravel()
                if values.size != 1:
                    raise RuntimeError(
                        f"{arm}/seed{seed}: expected one KRAS logit, got {values.size}"
                    )
                rows.append(
                    {
                        "slide_id": str(slide_id),
                        "seed": seed,
                        "fold": 0,
                        "logit": float(values[0]),
                    }
                )
    del module
    torch.cuda.empty_cache()
    scores = pd.DataFrame(rows).sort_values("slide_id", kind="stable").reset_index(drop=True)
    if len(scores) != len(manifest) or scores["slide_id"].duplicated().any():
        raise RuntimeError(f"{arm}/seed{seed}: packed inference row count is invalid")
    if not np.isfinite(scores["logit"].to_numpy(float)).all():
        raise RuntimeError(f"{arm}/seed{seed}: packed inference emitted non-finite logits")
    return scores


def score_one(arm: str, seed: int, *, num_workers: int) -> pd.DataFrame:
    cached = _load_valid_score(arm, seed)
    if cached is not None:
        print(f"== score {arm} seed{seed}: completed and hash-valid -- skipping")
        return cached
    destination = score_path(arm, seed)
    receipt_path = score_receipt_path(arm, seed)
    if destination.exists() or receipt_path.exists():
        raise RuntimeError(f"Partial score destination exists: {arm}/seed{seed}")
    inputs = _score_inputs(arm, seed)
    scores = _score_packed_checkpoint(arm, seed, num_workers=num_workers)
    if "logit" not in scores or not np.isfinite(scores["logit"].to_numpy(float)).all():
        raise RuntimeError(f"{arm}/seed{seed}: scorer did not emit finite native logits")
    lineage.write_parquet_once(destination, scores)
    lineage.write_json_once(
        receipt_path,
        {
            "schema_version": 1,
            "created_utc": _utcnow(),
            "lineage": lineage.lineage_name(),
            "inputs": inputs,
            "artifact": _artifact(destination),
            "n_rows": len(scores),
        },
    )
    validated = _load_valid_score(arm, seed)
    assert validated is not None
    return validated


def cmd_score(args: argparse.Namespace) -> None:
    _validate_contract(deep=True)
    _seal_inference_environment(num_workers=args.num_workers)
    selected = [(arm, seed) for arm in _selected_arms(args) for seed in _selected_seeds(args)]
    # Validate all requested refits before publishing the first score.
    for arm, seed in selected:
        _completed_refit(arm, seed)
    for arm, seed in selected:
        scores = score_one(arm, seed, num_workers=args.num_workers)
        print(f"  {arm:10s} seed{seed}: {len(scores)} target slide logits")


def patient_scores(arm: str, seed: int) -> pd.DataFrame:
    scores = _load_valid_score(arm, seed)
    if scores is None:
        raise FileNotFoundError(f"Missing target scores for {arm}/seed{seed}")
    manifest = pd.read_csv(target_manifest(arm), low_memory=False)
    roles = set(manifest["specimen_role"].dropna().astype(str))
    if roles != {"primary"}:
        raise RuntimeError(f"{arm}: target manifest mixes specimen roles: {roles}")
    slides = scoring.ensemble_slide_predictions(scores, manifest)
    patient = evaluate.to_patient_level(slides, manifest)
    keep = [
        column
        for column in (
            "patient_id",
            "label",
            "mean_logit",
            "cohort",
            "subcohort",
            "msi_dmmr",
            "braf",
        )
        if column in patient
    ]
    patient = patient[keep].copy()
    patient["patient_id"] = patient["patient_id"].astype(str)
    if patient["patient_id"].duplicated().any():
        raise RuntimeError(f"{arm}/seed{seed}: patient aggregation is not one row per patient")
    return patient.sort_values("patient_id").reset_index(drop=True)


def seed_ensemble(arm: str) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    per_seed: dict[int, pd.DataFrame] = {seed: patient_scores(arm, seed) for seed in SEEDS}
    frames = [per_seed[seed] for seed in SEEDS]
    base = frames[0].copy()
    for seed, frame in per_seed.items():
        if list(frame["patient_id"]) != list(base["patient_id"]):
            raise RuntimeError(f"{arm}: seed {seed} covers different target patients")
        if not np.array_equal(frame["label"].to_numpy(), base["label"].to_numpy()):
            raise RuntimeError(f"{arm}: seed {seed} target labels differ")
    base["mean_logit"] = np.mean(
        np.vstack([frame["mean_logit"].to_numpy(float) for frame in frames]), axis=0
    )
    base["prob_raw"] = sigmoid(base["mean_logit"].to_numpy(float))
    return base, per_seed


def _load_source_oof_patient(arm: str, seed: int) -> pd.DataFrame:
    _validate_source_cv(arm, seed)
    manifest = pd.read_csv(source_manifest(arm), low_memory=False)
    patient = evaluate.to_patient_level(
        pd.read_parquet(source_cv_dir(arm, seed) / "oof_predictions.parquet"),
        manifest,
    )
    patient["patient_id"] = patient["patient_id"].astype(str)
    return patient.sort_values("patient_id").reset_index(drop=True)


def _validate_calibrated(arm: str) -> tuple[dict[str, Any], pd.DataFrame]:
    cal_path = calibrator_path(arm)
    destination = calibrated_path(arm)
    receipt_path = calibrated_receipt_path(arm)
    if not cal_path.is_file() or not destination.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"Incomplete E2a-D calibration output for {arm}")
    calibrator = json.loads(cal_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    expected_source_inputs = {
        str(seed): _artifact(source_cv_receipt_path(arm, seed)) for seed in SEEDS
    }
    expected_score_inputs = {str(seed): _artifact(score_receipt_path(arm, seed)) for seed in SEEDS}
    expected = {
        "schema_version": 1,
        "lineage": lineage.lineage_name(),
        "arm": arm,
        "calibrator": _artifact(cal_path),
        "source_cv_receipts": expected_source_inputs,
        "score_receipts": expected_score_inputs,
        "artifact": _artifact(destination),
    }
    mismatches = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{arm}: calibrated receipt mismatch: {mismatches}")
    if calibrator.get("source_cv_receipts") != expected_source_inputs:
        raise RuntimeError(f"{arm}: calibrator source-CV lineage changed")
    for key in ("a", "b"):
        if not np.isfinite(float(calibrator.get(key, np.nan))):
            raise RuntimeError(f"{arm}: non-finite calibrator coefficient {key}")
    if float(calibrator["b"]) <= 0:
        raise RuntimeError(f"{arm}: Platt slope is non-positive; ranking would be reversed")
    frame = pd.read_parquet(destination)
    expected_patients = set(pd.read_csv(target_manifest(arm))["patient_id"].astype(str))
    if set(frame["patient_id"].astype(str)) != expected_patients:
        raise RuntimeError(f"{arm}: calibrated target patient coverage changed")
    required = {"mean_logit", "prob_source_calibrated", "eta_source_calibrated"}
    if not required.issubset(frame):
        raise RuntimeError(f"{arm}: calibrated target columns missing {required - set(frame)}")
    return calibrator, frame


def calibrate_one(arm: str) -> None:
    outputs = [calibrator_path(arm), calibrated_path(arm), calibrated_receipt_path(arm)]
    if any(path.exists() for path in outputs):
        if not all(path.is_file() for path in outputs):
            raise RuntimeError(f"{arm}: partial immutable calibration output exists")
        _validate_calibrated(arm)
        print(f"== calibrate {arm}: completed and hash-valid -- skipping")
        return
    source_frames = {seed: _load_source_oof_patient(arm, seed) for seed in SEEDS}
    first = source_frames[SEEDS[0]]
    for seed, frame in source_frames.items():
        if list(frame["patient_id"]) != list(first["patient_id"]):
            raise RuntimeError(f"{arm}: source OOF seed {seed} covers different patients")
        if not np.array_equal(frame["label"].to_numpy(), first["label"].to_numpy()):
            raise RuntimeError(f"{arm}: source OOF seed {seed} labels differ")
    source_logit = np.mean(
        np.vstack([source_frames[seed]["mean_logit"].to_numpy(float) for seed in SEEDS]),
        axis=0,
    )
    source_label = first["label"].to_numpy(int)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    model = LogisticRegression(penalty=None, max_iter=1_000)
    model.fit(source_logit.reshape(-1, 1), source_label)
    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0][0])
    if not np.isfinite([intercept, slope]).all() or slope <= 0:
        raise RuntimeError(f"{arm}: invalid source-only Platt map a={intercept}, b={slope}")
    target, _ = seed_ensemble(arm)
    eta = intercept + slope * target["mean_logit"].to_numpy(float)
    calibrated = target.assign(
        eta_source_calibrated=eta,
        prob_source_calibrated=sigmoid(eta),
    )
    source_receipts = {str(seed): _artifact(source_cv_receipt_path(arm, seed)) for seed in SEEDS}
    calibrator = {
        "schema_version": 1,
        "created_utc": _utcnow(),
        "experiment": "E2a-D source-only Platt calibration",
        "lineage": lineage.lineage_name(),
        "arm": arm,
        "a": intercept,
        "b": slope,
        "n_source": len(first),
        "source_prevalence": float(source_label.mean()),
        "source_oof_auroc": float(roc_auc_score(source_label, source_logit)),
        "fit_input": "three-seed mean honest source OOF native logit",
        "target_labels_used_for_fit": False,
        "source_cv_receipts": source_receipts,
    }
    lineage.write_json_once(calibrator_path(arm), calibrator)
    lineage.write_parquet_once(calibrated_path(arm), calibrated)
    lineage.write_json_once(
        calibrated_receipt_path(arm),
        {
            "schema_version": 1,
            "created_utc": _utcnow(),
            "lineage": lineage.lineage_name(),
            "arm": arm,
            "calibrator": _artifact(calibrator_path(arm)),
            "source_cv_receipts": source_receipts,
            "score_receipts": {
                str(seed): _artifact(score_receipt_path(arm, seed)) for seed in SEEDS
            },
            "artifact": _artifact(calibrated_path(arm)),
        },
    )
    _validate_calibrated(arm)
    print(
        f"== calibrate {arm}: source n={len(first)}, OOF AUROC "
        f"{calibrator['source_oof_auroc']:.4f}, map {intercept:+.4f} + {slope:.4f}*eta"
    )


def cmd_calibrate(args: argparse.Namespace) -> None:
    _validate_contract(deep=False)
    for arm in _selected_arms(args):
        calibrate_one(arm)


def _family_calibrated(target: str, *, family_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = family_root / "calibrated" / f"cap{CAP}_{target}_primary.parquet"
    receipt_path = path.with_suffix(".receipt.json")
    calibrator = family_root / "calibrators" / f"cap{CAP}_{target}.json"
    for required in (path, receipt_path, calibrator):
        if not required.is_file():
            raise FileNotFoundError(f"Missing frozen family-LOCO input: {required}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("artifact") != _artifact(path):
        raise RuntimeError(f"Frozen family-LOCO calibrated artifact changed: {path}")
    score_receipts = receipt.get("score_receipts") or []
    if len(score_receipts) != len(SEEDS):
        raise RuntimeError(f"Frozen family-LOCO score roster is incomplete: {path}")
    for identity in score_receipts:
        score_receipt = Path(str(identity.get("path", "")))
        if identity != _artifact(score_receipt):
            raise RuntimeError(f"Frozen family-LOCO score receipt changed: {score_receipt}")
    frame = pd.read_parquet(path)
    required_columns = {"patient_id", "label", "mean_logit", "subcohort"}
    if not required_columns.issubset(frame):
        raise RuntimeError(f"Family-LOCO frame lacks {required_columns - set(frame)}")
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame["patient_id"].duplicated().any():
        raise RuntimeError(f"Family-LOCO target is not one row per patient: {path}")
    inputs = {
        "calibrated": _artifact(path),
        "calibrated_receipt": _artifact(receipt_path),
        "calibrator": _artifact(calibrator),
        "score_receipts": score_receipts,
    }
    return frame.sort_values("patient_id").reset_index(drop=True), inputs


def _align_pair(
    sibling: pd.DataFrame, family: pd.DataFrame, *, arm: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = sibling.sort_values("patient_id").reset_index(drop=True)
    right = family.sort_values("patient_id").reset_index(drop=True)
    if list(left["patient_id"]) != list(right["patient_id"]):
        missing_left = sorted(set(right["patient_id"]) - set(left["patient_id"]))
        missing_right = sorted(set(left["patient_id"]) - set(right["patient_id"]))
        raise RuntimeError(
            f"{arm}: sibling/family target patients differ; "
            f"missing sibling={missing_left[:5]}, missing family={missing_right[:5]}"
        )
    if not np.array_equal(left["label"].to_numpy(), right["label"].to_numpy()):
        raise RuntimeError(f"{arm}: sibling/family target labels differ")
    return left, right


def stratified_bootstrap_indices(
    labels: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("KRAS-stratified bootstrap requires both binary classes")
    draws = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        draws.append(rng.choice(candidates, size=(n_bootstrap, len(candidates)), replace=True))
    return np.concatenate(draws, axis=1)


def bootstrap_auroc_samples(
    labels: np.ndarray, score: np.ndarray, indices: np.ndarray, *, chunk_size: int = 128
) -> np.ndarray:
    """Tie-correct AUROC on fixed-class-count draws without a multi-GB tensor."""

    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(labels) != len(score) or indices.ndim != 2:
        raise ValueError("AUROC bootstrap inputs have incompatible shapes")
    n_negative = int((labels == 0).sum())
    n_positive = int((labels == 1).sum())
    if n_negative == 0 or n_positive == 0:
        raise ValueError("AUROC requires both classes")
    out = np.empty(len(indices), dtype=float)
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        sampled = score[indices[start:stop]]
        negative = sampled[:, :n_negative]
        positive = sampled[:, n_negative:]
        difference = positive[:, :, None] - negative[:, None, :]
        out[start:stop] = (
            (difference > 0).sum(axis=(1, 2)) + 0.5 * (difference == 0).sum(axis=(1, 2))
        ) / (n_negative * n_positive)
    return out


def _interval(values: np.ndarray) -> list[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def _rank_point(frame: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(frame["label"], frame["mean_logit"]))


def _target_metric_block(
    frame: pd.DataFrame, auroc_samples: np.ndarray, per_seed: dict[int, pd.DataFrame]
) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    from oceanpath.eval.core import compute_calibration_intercept_slope

    labels = frame["label"].to_numpy(int)
    logits = frame["mean_logit"].to_numpy(float)
    probability = np.clip(frame["prob_source_calibrated"].to_numpy(float), 1e-6, 1 - 1e-6)
    calibration = compute_calibration_intercept_slope(labels, probability)
    raw_probability = np.clip(sigmoid(logits), 1e-6, 1 - 1e-6)
    return {
        "n": len(frame),
        "n_mutant": int(labels.sum()),
        "n_wild_type": int(len(labels) - labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, logits)),
        "auroc_ci95": _interval(auroc_samples),
        "auprc": float(average_precision_score(labels, logits)),
        "source_calibrated": {
            "brier": float(brier_score_loss(labels, probability)),
            "log_loss": float(log_loss(labels, probability)),
            "calibration_intercept": float(calibration["calibration_intercept"]),
            "calibration_slope": float(calibration["calibration_slope"]),
        },
        "raw_probability_diagnostic": {
            "brier": float(brier_score_loss(labels, raw_probability)),
            "log_loss": float(log_loss(labels, raw_probability)),
        },
        "per_seed_auroc_descriptive": {
            str(seed): float(roc_auc_score(seed_frame["label"], seed_frame["mean_logit"]))
            for seed, seed_frame in sorted(per_seed.items())
        },
        "directional_gate": {
            "rule": "patient-bootstrap AUROC lower 95% bound > 0.50",
            "passes": bool(_interval(auroc_samples)[0] > 0.50),
        },
    }


def compute_report(
    *,
    family_root: Path,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Recompute every E2a-D primitive, paired delta, and declared macro."""

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    sibling: dict[str, pd.DataFrame] = {}
    per_seed: dict[str, dict[int, pd.DataFrame]] = {}
    calibrator_inputs: dict[str, Any] = {}
    for arm in ARM_NAMES:
        calibrator, frame = _validate_calibrated(arm)
        ensemble, seed_frames = seed_ensemble(arm)
        if list(frame.sort_values("patient_id")["patient_id"]) != list(
            ensemble.sort_values("patient_id")["patient_id"]
        ):
            raise RuntimeError(f"{arm}: calibrated and native ensembles cover different patients")
        sibling[arm] = frame.sort_values("patient_id").reset_index(drop=True)
        per_seed[arm] = seed_frames
        calibrator_inputs[arm] = {
            "calibrator": _artifact(calibrator_path(arm)),
            "calibrated": _artifact(calibrated_path(arm)),
            "calibrated_receipt": _artifact(calibrated_receipt_path(arm)),
            "a": float(calibrator["a"]),
            "b": float(calibrator["b"]),
            "n_source": int(calibrator["n_source"]),
        }
    family: dict[str, pd.DataFrame] = {}
    family_inputs: dict[str, Any] = {}
    for target in ("surgen", "tcga", "rih", "cptac"):
        family[target], family_inputs[target] = _family_calibrated(target, family_root=family_root)
    paired_family: dict[str, pd.DataFrame] = {}
    for arm, spec in ARMS.items():
        target = FAMILY_TARGET_FOR_ARM[arm]
        baseline = family[target].loc[family[target]["subcohort"].eq(spec.target_subcohort)].copy()
        sibling[arm], paired_family[arm] = _align_pair(sibling[arm], baseline, arm=arm)
    rng = np.random.default_rng(bootstrap_seed)
    target_samples: dict[str, np.ndarray] = {}
    family_pair_samples: dict[str, np.ndarray] = {}
    for arm in ARM_NAMES:
        labels = sibling[arm]["label"].to_numpy(int)
        indices = stratified_bootstrap_indices(labels, n_bootstrap=n_bootstrap, rng=rng)
        target_samples[arm] = bootstrap_auroc_samples(
            labels, sibling[arm]["mean_logit"].to_numpy(float), indices
        )
        family_pair_samples[arm] = bootstrap_auroc_samples(
            labels, paired_family[arm]["mean_logit"].to_numpy(float), indices
        )
    standalone_samples: dict[str, np.ndarray] = {}
    for target in ("tcga", "rih", "cptac"):
        labels = family[target]["label"].to_numpy(int)
        indices = stratified_bootstrap_indices(labels, n_bootstrap=n_bootstrap, rng=rng)
        standalone_samples[target] = bootstrap_auroc_samples(
            labels, family[target]["mean_logit"].to_numpy(float), indices
        )
    targets = {
        arm: _target_metric_block(sibling[arm], target_samples[arm], per_seed[arm])
        for arm in ARM_NAMES
    }
    paired: dict[str, Any] = {}
    for arm in ARM_NAMES:
        delta_samples = target_samples[arm] - family_pair_samples[arm]
        paired[arm] = {
            "estimand": (
                f"AUROC(held out {ARMS[arm].target_subcohort}, related sibling retained) "
                "- AUROC(whole source family held out)"
            ),
            "n_paired_patients": len(sibling[arm]),
            "sibling_retained_auroc": _rank_point(sibling[arm]),
            "whole_family_held_out_auroc": _rank_point(paired_family[arm]),
            "delta_auroc": _rank_point(sibling[arm]) - _rank_point(paired_family[arm]),
            "delta_auroc_ci95": _interval(delta_samples),
            "bootstrap": "paired target-patient, KRAS-stratified, shared indices",
        }
    five_samples = np.mean(
        np.vstack(
            [
                standalone_samples["tcga"],
                target_samples["sr386"],
                target_samples["sr1482"],
                standalone_samples["rih"],
                standalone_samples["cptac"],
            ]
        ),
        axis=0,
    )
    six_samples = np.mean(
        np.vstack(
            [
                target_samples["tcga_coad"],
                target_samples["tcga_read"],
                target_samples["sr386"],
                target_samples["sr1482"],
                standalone_samples["rih"],
                standalone_samples["cptac"],
            ]
        ),
        axis=0,
    )
    five_points = {
        "TCGA_whole_family_holdout": _rank_point(family["tcga"]),
        "SR386_sibling_holdout": _rank_point(sibling["sr386"]),
        "SR1482_sibling_holdout": _rank_point(sibling["sr1482"]),
        "RIH_whole_family_holdout": _rank_point(family["rih"]),
        "CPTAC_whole_family_holdout": _rank_point(family["cptac"]),
    }
    six_points = {
        "TCGA_COAD_sibling_holdout": _rank_point(sibling["tcga_coad"]),
        "TCGA_READ_sibling_holdout": _rank_point(sibling["tcga_read"]),
        "SR386_sibling_holdout": _rank_point(sibling["sr386"]),
        "SR1482_sibling_holdout": _rank_point(sibling["sr1482"]),
        "RIH_whole_family_holdout": _rank_point(family["rih"]),
        "CPTAC_whole_family_holdout": _rank_point(family["cptac"]),
    }
    return {
        "schema_version": 1,
        "created_utc": _utcnow(),
        "experiment": "E2a-D sibling-stratum LOCO",
        "lineage": lineage.lineage_name(),
        "analysis_role": "additive final-v8 sibling-stratum transport panel",
        "target_outcomes_used_for_model_selection": False,
        "cap": CAP,
        "seeds": list(SEEDS),
        "inference": {
            "sampling_unit": "patient",
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "stratification": "target x KRAS",
            "paired_model_indices_shared": True,
            "macro_recomputed_each_draw": True,
            "folds_and_seeds_are_not_resampling_units": True,
        },
        "targets": targets,
        "paired_sibling_minus_family": paired,
        "macros": {
            "secondary_five_acquisition_domain": {
                "primitives": five_points,
                "auroc": float(np.mean(list(five_points.values()))),
                "auroc_ci95": _interval(five_samples),
            },
            "descriptive_equal_six_stratum": {
                "primitives": six_points,
                "auroc": float(np.mean(list(six_points.values()))),
                "auroc_ci95": _interval(six_samples),
                "role": "descriptive; gives TCGA and SurGen two votes each",
            },
        },
        "inputs": {
            "manifest_contract": _artifact(manifest_contract_path()),
            "inference_environment": _artifact(inference_environment_path()),
            "analysis_environment": _artifact(analysis_environment_path()),
            "sibling_calibration": calibrator_inputs,
            "frozen_family_root": str(family_root.resolve()),
            "frozen_family": family_inputs,
        },
    }


def _print_report(report: dict[str, Any]) -> None:
    print("\nE2a-D SIBLING-STRATUM LOCO")
    print("=" * 92)
    for arm, block in report["targets"].items():
        interval = block["auroc_ci95"]
        source_calibrated = block["source_calibrated"]
        print(
            f"{ARMS[arm].target_subcohort:12s} n={block['n']:4d} "
            f"AUROC {block['auroc']:.4f} [{interval[0]:.4f}, {interval[1]:.4f}] "
            f"AUPRC {block['auprc']:.4f} gate={block['directional_gate']['passes']} "
            f"Brier/logloss {source_calibrated['brier']:.4f}/"
            f"{source_calibrated['log_loss']:.4f}"
        )
    print("\nPaired sibling-exposure contrasts")
    for arm, block in report["paired_sibling_minus_family"].items():
        interval = block["delta_auroc_ci95"]
        print(
            f"{ARMS[arm].target_subcohort:12s} delta {block['delta_auroc']:+.4f} "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"
        )
    for name, block in report["macros"].items():
        interval = block["auroc_ci95"]
        print(f"{name}: {block['auroc']:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]")


def _current_analysis_environment() -> dict[str, Any]:
    sources = (
        Path(__file__).resolve(),
        REPO / "src/oceanpath/aim1/evaluate.py",
        REPO / "src/oceanpath/eval/core.py",
        REPO / "src/oceanpath/eval/external.py",
    )
    return {
        "python_version": platform.python_version(),
        "numpy_version": importlib.metadata.version("numpy"),
        "pandas_version": importlib.metadata.version("pandas"),
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
        "bootstrap": {
            "n": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
            "sampling_unit": "patient",
            "stratification": "target x KRAS",
        },
        "implementation_sources": [_artifact(path) for path in sources],
    }


def _seal_analysis_environment() -> dict[str, Any]:
    environment = _current_analysis_environment()
    path = analysis_environment_path()
    if path.exists():
        if not path.is_file() or json.loads(path.read_text()) != environment:
            raise RuntimeError("E2a-D analysis environment changed after seal")
    else:
        lineage.write_json_once(path, environment)
    return environment


def cmd_report(args: argparse.Namespace) -> None:
    _validate_contract(deep=False)
    if not inference_environment_path().is_file():
        raise FileNotFoundError("Missing governed inference environment seal")
    recorded_inference = json.loads(inference_environment_path().read_text())
    _seal_inference_environment(num_workers=int(recorded_inference["num_workers"]))
    _seal_analysis_environment()
    destination = report_path()
    receipt_path = report_receipt_path()
    if destination.exists() or receipt_path.exists():
        raise FileExistsError("Refusing to overwrite an E2a-D report or receipt")
    report = compute_report(
        family_root=args.family_root,
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    lineage.write_json_once(destination, report)
    lineage.write_json_once(
        receipt_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "created_utc": _utcnow(),
            "lineage": lineage.lineage_name(),
            "experiment": "E2a-D sibling-stratum LOCO",
            "report": _artifact(destination),
            "manifest_contract": _artifact(manifest_contract_path()),
            "inference_environment": _artifact(inference_environment_path()),
            "analysis_environment": _artifact(analysis_environment_path()),
            "calibrated_receipts": {
                arm: _artifact(calibrated_receipt_path(arm)) for arm in ARM_NAMES
            },
            "family_inputs": report["inputs"]["frozen_family"],
            "required_fit_count": 72,
            "completed_source_cv_fits": 60,
            "completed_refits": 12,
        },
    )
    _print_report(report)
    print(f"\nWrote {destination}")


def cmd_verify(args: argparse.Namespace) -> None:
    _validate_contract(deep=args.deep)
    if not inference_environment_path().is_file():
        raise FileNotFoundError("Missing E2a-D inference environment seal")
    if json.loads(inference_environment_path().read_text()) != {
        **_current_inference_environment(),
        "num_workers": json.loads(inference_environment_path().read_text())["num_workers"],
    }:
        raise RuntimeError("E2a-D inference environment no longer reproduces")
    for arm in ARM_NAMES:
        for seed in SEEDS:
            _validate_source_cv(arm, seed)
            _completed_refit(arm, seed)
            if _load_valid_score(arm, seed) is None:
                raise FileNotFoundError(f"Missing score: {arm}/seed{seed}")
        _validate_calibrated(arm)
    if not report_path().is_file() or not report_receipt_path().is_file():
        raise FileNotFoundError("Missing E2a-D final report or receipt")
    receipt = json.loads(report_receipt_path().read_text())
    if receipt.get("report") != _artifact(report_path()):
        raise RuntimeError("E2a-D report hash differs from its completion receipt")
    if receipt.get("manifest_contract") != _artifact(manifest_contract_path()):
        raise RuntimeError("E2a-D report uses a different manifest contract")
    if receipt.get("required_fit_count") != 72:
        raise RuntimeError("E2a-D completion receipt has the wrong fit roster")
    print("PASS: E2a-D 72-fit lineage, predictions, calibration, report, and receipts verify")


def _add_arm_seed_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm", choices=ARM_NAMES)
    parser.add_argument("--seed", type=int, choices=SEEDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("plan")
    command.set_defaults(func=cmd_plan)

    command = sub.add_parser("manifests")
    command.add_argument("--apply", action="store_true")
    command.set_defaults(func=cmd_manifests)

    command = sub.add_parser("preflight")
    command.add_argument("--deep", action="store_true")
    command.set_defaults(func=cmd_preflight)

    command = sub.add_parser("source-cv")
    _add_arm_seed_options(command)
    command.add_argument("--dry-run", action="store_true")
    command.set_defaults(func=cmd_source_cv)

    command = sub.add_parser("train")
    _add_arm_seed_options(command)
    command.add_argument("--dry-run", action="store_true")
    command.set_defaults(func=cmd_train)

    command = sub.add_parser("smoke", help="1-10-step non-governed CUDA/config smoke below /tmp")
    command.add_argument("--arm", choices=ARM_NAMES, default="sr386")
    command.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    command.add_argument("--steps", type=int, default=2)
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--dry-run", action="store_true")
    command.set_defaults(func=cmd_smoke)

    command = sub.add_parser("_fit")
    command.add_argument("--arm", choices=ARM_NAMES, required=True)
    command.add_argument("--seed", type=int, choices=SEEDS, required=True)
    command.add_argument("--epochs", type=int, required=True)
    command.add_argument("--run-dir", type=Path, required=True)
    command.set_defaults(func=cmd_fit)

    command = sub.add_parser("score")
    _add_arm_seed_options(command)
    command.add_argument("--num-workers", type=int, default=4)
    command.set_defaults(func=cmd_score)

    command = sub.add_parser("calibrate")
    command.add_argument("--arm", choices=ARM_NAMES)
    command.set_defaults(func=cmd_calibrate)

    command = sub.add_parser("report")
    command.add_argument("--family-root", type=Path, default=FAMILY_ROOT)
    command.set_defaults(func=cmd_report)

    command = sub.add_parser("verify")
    command.add_argument("--deep", action="store_true")
    command.set_defaults(func=cmd_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
