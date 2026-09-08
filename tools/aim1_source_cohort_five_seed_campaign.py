#!/usr/bin/env python3
"""Governed five-seed controller for four Aim-1 primary-cohort OOF studies.

The campaign is deliberately additive and contains exactly four arms, five
model seeds, and five predefined OOF folds: 4 * 5 * 5 = 100 fits.  No final
refit is permitted.  The public workflow is::

    uv run python /tmp/aim1_source_cohort_phase1.py plan
    uv run python /tmp/aim1_source_cohort_phase1.py prepare --apply
    uv run python /tmp/aim1_source_cohort_phase1.py preflight --apply
    uv run python /tmp/aim1_source_cohort_phase1.py train --apply --max-workers 6
    uv run python /tmp/aim1_source_cohort_phase1.py validate --seal
    uv run python /tmp/aim1_source_cohort_phase1.py status

``prepare`` subsets the frozen Aim-1 manifest and its already frozen fold
assignment; it never redraws folds.  All persisted control-plane files are
write-once.  A partial training directory is never overwritten.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(
    os.environ.get(
        "OCEANPATH_REPO", "/home/yc_liu/projects/OceanPath-colon-development"
    )
).resolve()
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import paths  # noqa: E402
from oceanpath.splitting.core import derive_subset_splits  # noqa: E402

SCHEMA_VERSION = 1
CAMPAIGN = "aim1_primary_cohort_oof_5seed"
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim1_primary_cohort_5seed_v1_20260824"
)
MASTER_MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/aim1_dev.csv")
MASTER_SPLITS_DIR = REPO / "outputs/splits/aim1_1a/aim1_balanced5"
MASTER_MANIFEST_SHA256 = (
    "d906ed5b61c5d3bbf56da7ec7bad412287307461012deee5e6d98ae97bd1f3d1"
)
MASTER_SPLITS_SHA256 = (
    "3ec0b106f3614ef994efa88c4acdac35591166639517a4c0defa549a476687e8"
)
MASTER_INTEGRITY_FILE_SHA256 = (
    "daa4668cf5364cce61dea5e585ef0b353e6ede2ec932f94322323171bd8ac3c5"
)
MASTER_SUMMARY_SHA256 = (
    "e02c3ddd4a0f712370519c56f7a1e5aba2e20b5c49f8fa8ec5e8ae27887c471a"
)
EXPECTED_PACKED_STORE = {
    "meta.json": {
        "sha256": "44f1f0c80f4b8740c5fc82caafa96a8ead372c561fe53cda896d83972ff3c80b",
    },
    "index.parquet": {
        "sha256": "705464267e35e454e50ea8510bc937e9f8d6a00b8b97bb9cf1e06da5fcfa9556",
    },
    "features.bin": {
        "sha256": "6765d9faf30f40e212075c1a650fd34bc5ef625b3fb8e169dbc14e78c2058bd4",
        "size_bytes": 49_735_520_256,
    },
    "coords.bin": {
        "sha256": "1a90bd95a430e433cff900c0b6b2a4f6fedcae550ae80584ee27ed7f100fe491",
        "size_bytes": 194_279_376,
    },
}
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
N_FOLDS = 5
FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
CAP = 8_192
MAX_WORKERS = 6
MAX_CONCURRENT_GPU_TRAINERS = MAX_WORKERS
TOTAL_JOBS = 20
TOTAL_FITS = 100
GLOBAL_GPU_LOCK = Path("/tmp/oceanpath_gpu0_exclusive.lock")


class ContractError(RuntimeError):
    """Fail-closed violation of the immutable campaign contract."""


@dataclass(frozen=True)
class ArmSpec:
    name: str
    model_name: str
    description: str
    expected_slides: int
    expected_patients: int
    expected_mutant_patients: int
    expected_wildtype_patients: int
    expected_test_patients: tuple[int, int, int, int, int]


ARMS: dict[str, ArmSpec] = {
    spec.name: spec
    for spec in (
        ArmSpec(
            "tcga_primary",
            "primary_tcga",
            "specimen_role=primary and cohort=TCGA",
            508,
            502,
            207,
            295,
            (101, 100, 101, 100, 100),
        ),
        ArmSpec(
            "sr386_primary",
            "primary_sr386",
            "specimen_role=primary and cohort=SurGen and subcohort=SR386",
            413,
            413,
            147,
            266,
            (83, 82, 82, 83, 83),
        ),
        ArmSpec(
            "surgen_primary",
            "primary_surgen",
            "specimen_role=primary and cohort=SurGen and subcohort in {SR386,SR1482}",
            881,
            737,
            294,
            443,
            (147, 147, 147, 148, 148),
        ),
        ArmSpec(
            "tcga_surgen_primary",
            "primary_tcga_surgen",
            "specimen_role=primary and cohort in {TCGA,SurGen}",
            1_389,
            1_239,
            501,
            738,
            (248, 247, 248, 248, 248),
        ),
    )
}
ARM_SPECS = ARMS
EXPECTED_ARM_CENSUS: dict[str, dict[str, int]] = {
    arm: {
        "patients": spec.expected_patients,
        "slides": spec.expected_slides,
        "mutant": spec.expected_mutant_patients,
        "wild_type": spec.expected_wildtype_patients,
    }
    for arm, spec in ARMS.items()
}


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"Required regular artifact missing or symlinked: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON artifact must contain an object: {path}")
    return value


def _write_text_once(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_text(encoding="utf-8") == text:
            return
        raise ContractError(f"Refusing to replace a different sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with temp.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_once(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def assert_safe_output_root(output_root: Path) -> Path:
    raw = Path(output_root).expanduser()
    if not raw.is_absolute():
        raise ContractError("--output-root must be absolute")
    lexical = raw.absolute()
    resolved = raw.resolve(strict=False)
    if lexical != resolved:
        raise ContractError(f"Output root must not traverse symlinks: {raw} -> {resolved}")
    cursor = raw
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise ContractError(f"Output-root path contains a symlink: {cursor}")
        cursor = cursor.parent
    temporary = Path("/tmp").resolve()
    production = DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    if resolved != production and resolved != temporary and not _is_relative_to(resolved, temporary):
        raise ContractError(
            f"Production root must be exactly {production}; tests may use /tmp, got {resolved}"
        )
    for frozen in (MASTER_MANIFEST.resolve(), MASTER_SPLITS_DIR.resolve()):
        if resolved == frozen or _is_relative_to(resolved, frozen) or _is_relative_to(frozen, resolved):
            raise ContractError(f"Output root overlaps frozen input: {frozen}")
    return resolved


def contract_path(root: Path) -> Path:
    return root / "contract.json"


def preflight_path(root: Path) -> Path:
    return root / "receipts/deep_preflight.json"


def training_receipt_path(root: Path) -> Path:
    return root / "receipts/training_complete.json"


def manifest_path(root: Path, arm: str) -> Path:
    return root / f"inputs/manifests/{arm}.csv"


def split_root(root: Path) -> Path:
    return root / "inputs/splits"


def split_dir(root: Path, arm: str) -> Path:
    return split_root(root) / f"aim1_{ARMS[arm].model_name}/aim1_balanced5"


def run_dir(root: Path, arm: str, seed: int) -> Path:
    return root / f"train/source_cv/cap{CAP}/{arm}/seed{seed}"


def job_receipt_path(root: Path, arm: str, seed: int) -> Path:
    return root / f"receipts/source_cv/{arm}/seed{seed}.json"


def request_path(root: Path, arm: str, seed: int) -> Path:
    return root / f"requests/source_cv/{arm}/seed{seed}.json"


def log_path(root: Path, arm: str, seed: int) -> Path:
    return root / f"logs/source_cv/{arm}/seed{seed}.log"


def failure_path(root: Path, arm: str, seed: int) -> Path:
    return root / f"requests/source_cv/{arm}/seed{seed}.failure.json"


def _arm_mask(frame: pd.DataFrame, arm: str) -> pd.Series:
    role = frame["specimen_role"].astype(str).str.strip().str.casefold()
    cohort = frame["cohort"].astype(str).str.strip().str.casefold()
    subcohort = frame["subcohort"].astype(str).str.strip().str.casefold()
    primary = role.eq("primary")
    if arm == "tcga_primary":
        return primary & cohort.eq("tcga")
    if arm == "sr386_primary":
        return primary & cohort.eq("surgen") & subcohort.eq("sr386")
    if arm == "surgen_primary":
        return primary & cohort.eq("surgen") & subcohort.isin({"sr386", "sr1482"})
    if arm == "tcga_surgen_primary":
        return primary & cohort.isin({"tcga", "surgen"})
    raise ContractError(f"Unknown arm: {arm}")


def _patient_labels(frame: pd.DataFrame, *, context: str) -> pd.Series:
    labels = pd.to_numeric(frame["target_label"], errors="raise").astype(int)
    if not labels.isin([0, 1]).all():
        raise ContractError(f"{context}: labels must be binary")
    work = pd.DataFrame(
        {"patient_id": frame["patient_id"].astype(str), "target_label": labels}
    )
    counts = work.groupby("patient_id")["target_label"].nunique()
    if (counts != 1).any():
        raise ContractError(f"{context}: a patient has inconsistent labels")
    return work.groupby("patient_id", sort=True)["target_label"].first()


def _validate_arm_frame(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    spec = ARMS[arm]
    required = {
        "slide_id", "patient_id", "target_label", "specimen_role", "cohort",
        "subcohort", "k_fold", *(f"val_fold_{fold}" for fold in range(N_FOLDS)),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractError(f"{arm}: manifest lacks columns {missing}")
    if frame["slide_id"].astype(str).duplicated().any():
        raise ContractError(f"{arm}: duplicate slide_id")
    if not _arm_mask(frame, arm).all():
        raise ContractError(f"{arm}: row violates the exact arm filter")
    labels = _patient_labels(frame, context=arm)
    observed = {
        "slides": int(len(frame)),
        "patients": int(labels.size),
        "mutant_patients": int(labels.eq(1).sum()),
        "wildtype_patients": int(labels.eq(0).sum()),
    }
    expected = {
        "slides": spec.expected_slides,
        "patients": spec.expected_patients,
        "mutant_patients": spec.expected_mutant_patients,
        "wildtype_patients": spec.expected_wildtype_patients,
    }
    if observed != expected:
        raise ContractError(f"{arm}: census mismatch expected={expected}, observed={observed}")
    fold = pd.to_numeric(frame["k_fold"], errors="raise").astype(int)
    if set(fold) != set(range(N_FOLDS)):
        raise ContractError(f"{arm}: k_fold roster is not exactly 0..4")
    patient_fold = pd.DataFrame(
        {"patient_id": frame["patient_id"].astype(str), "fold": fold}
    ).groupby("patient_id")["fold"]
    if (patient_fold.nunique() != 1).any():
        raise ContractError(f"{arm}: a patient spans OOF test folds")
    test_counts = tuple(
        int(frame.loc[fold.eq(index), "patient_id"].astype(str).nunique())
        for index in range(N_FOLDS)
    )
    if test_counts != spec.expected_test_patients:
        raise ContractError(
            f"{arm}: per-fold test census mismatch expected={spec.expected_test_patients}, "
            f"observed={test_counts}"
        )
    cells: dict[str, Any] = {}
    for index in range(N_FOLDS):
        test = fold.eq(index)
        val = pd.to_numeric(frame[f"val_fold_{index}"], errors="raise").astype(int).eq(1)
        train = ~(test | val)
        if (test & val).any():
            raise ContractError(f"{arm}/fold{index}: validation overlaps test")
        patient_sets = {
            name: set(frame.loc[mask, "patient_id"].astype(str))
            for name, mask in (("train", train), ("val", val), ("test", test))
        }
        if (
            patient_sets["train"] & patient_sets["val"]
            or patient_sets["train"] & patient_sets["test"]
            or patient_sets["val"] & patient_sets["test"]
        ):
            raise ContractError(f"{arm}/fold{index}: train/val/test patient overlap")
        cell = {}
        for name, mask in (("train", train), ("val", val), ("test", test)):
            cell_labels = _patient_labels(frame.loc[mask], context=f"{arm}/fold{index}/{name}")
            if set(cell_labels.unique()) != {0, 1}:
                raise ContractError(f"{arm}/fold{index}/{name}: both labels are required")
            cell[name] = {
                "slides": int(mask.sum()),
                "patients": int(cell_labels.size),
                "mutant_patients": int(cell_labels.eq(1).sum()),
                "wildtype_patients": int(cell_labels.eq(0).sum()),
            }
        cells[str(index)] = cell
    return {**observed, "test_patients_by_fold": list(test_counts), "fold_cells": cells}


def _validate_derived_split(root: Path, arm: str, frame: pd.DataFrame) -> dict[str, Any]:
    directory = split_dir(root, arm)
    parquet = directory / "splits.parquet"
    integrity = directory / ".integrity_hash"
    summary = directory / "summary.json"
    derived = pd.read_parquet(parquet)
    wanted = set(frame["slide_id"].astype(str))
    observed = set(derived["slide_id"].astype(str))
    if len(derived) != len(frame) or observed != wanted or derived["slide_id"].duplicated().any():
        raise ContractError(f"{arm}: derived split roster differs from manifest")
    master = pd.read_parquet(MASTER_SPLITS_DIR / "splits.parquet")
    val_columns = [f"val_fold_{fold}" for fold in range(N_FOLDS)]
    columns = ["slide_id", "fold", *val_columns]
    check = derived[columns].merge(
        master[columns], on="slide_id", how="inner", validate="one_to_one", suffixes=("", "_master")
    )
    if len(check) != len(frame):
        raise ContractError(f"{arm}: master split join is incomplete")
    for column in ["fold", *val_columns]:
        if not pd.to_numeric(check[column]).astype(int).eq(
            pd.to_numeric(check[f"{column}_master"]).astype(int)
        ).all():
            raise ContractError(f"{arm}: derived {column} changed from master")
    manifest_fold = frame[["slide_id", "k_fold", *val_columns]].merge(
        derived[columns], on="slide_id", validate="one_to_one", suffixes=("_manifest", "_split")
    )
    if not pd.to_numeric(manifest_fold["k_fold"]).astype(int).eq(
        pd.to_numeric(manifest_fold["fold"]).astype(int)
    ).all():
        raise ContractError(f"{arm}: manifest k_fold differs from derived split")
    for column in val_columns:
        if not pd.to_numeric(manifest_fold[f"{column}_manifest"]).astype(int).eq(
            pd.to_numeric(manifest_fold[f"{column}_split"]).astype(int)
        ).all():
            raise ContractError(f"{arm}: manifest {column} differs from derived split")
    return {
        "splits": _artifact(parquet),
        "integrity": _artifact(integrity),
        "summary": _artifact(summary),
        "derived_rows": int(len(derived)),
        "fold_distribution": {
            str(key): int(value)
            for key, value in derived["fold"].value_counts().sort_index().items()
        },
    }


def _source_snapshot() -> list[dict[str, Any]]:
    explicit = {
        Path(__file__).resolve(),
        REPO / "tests/test_aim1_source_cohort_five_seed_campaign.py",
        REPO / "tools/study_train.py",
        REPO / "pyproject.toml",
        REPO / "uv.lock",
    }
    sources = explicit | set((REPO / "src/oceanpath").rglob("*.py")) | set(
        (REPO / "configs").rglob("*.yaml")
    )
    return [_artifact(path) for path in sorted(sources, key=lambda item: str(item.resolve()))]


def _master_identities() -> dict[str, Any]:
    identities = {
        "manifest": _artifact(MASTER_MANIFEST),
        "splits": _artifact(MASTER_SPLITS_DIR / "splits.parquet"),
        "integrity": _artifact(MASTER_SPLITS_DIR / ".integrity_hash"),
        "summary": _artifact(MASTER_SPLITS_DIR / "summary.json"),
    }
    expected = {
        "manifest": MASTER_MANIFEST_SHA256,
        "splits": MASTER_SPLITS_SHA256,
        "integrity": MASTER_INTEGRITY_FILE_SHA256,
        "summary": MASTER_SUMMARY_SHA256,
    }
    mismatch = {
        key: {"expected": expected[key], "observed": identities[key]["sha256"]}
        for key in expected
        if identities[key]["sha256"] != expected[key]
    }
    if mismatch:
        raise ContractError(f"Frozen Aim-1 master identities drifted: {mismatch}")
    return identities


def _packed_store_identity(root: Path) -> dict[str, Any]:
    """Authenticate the exact packed UNI-v1 bytes and requested slide coverage."""

    packed = paths.PACKED_FEATURE_DIR
    artifacts: dict[str, Any] = {}
    for name, expected in EXPECTED_PACKED_STORE.items():
        observed = _artifact(packed / name)
        if observed["sha256"] != expected["sha256"]:
            raise ContractError(
                f"Packed UNI-v1 {name} SHA drifted: expected {expected['sha256']}, "
                f"observed {observed['sha256']}"
            )
        expected_size = expected.get("size_bytes")
        if expected_size is not None and observed["size_bytes"] != expected_size:
            raise ContractError(
                f"Packed UNI-v1 {name} size drifted: expected {expected_size}, "
                f"observed {observed['size_bytes']}"
            )
        artifacts[name] = observed

    index = pd.read_parquet(packed / "index.parquet")
    id_column = "slide_id" if "slide_id" in index else "key"
    if id_column not in index or index[id_column].astype(str).duplicated().any():
        raise ContractError("Packed UNI-v1 index lacks a unique slide identifier")
    indexed = set(index[id_column].astype(str))
    coverage: dict[str, Any] = {}
    for arm in ARMS:
        wanted = set(pd.read_csv(manifest_path(root, arm), usecols=["slide_id"])["slide_id"].astype(str))
        missing = sorted(wanted - indexed)
        if missing:
            raise ContractError(
                f"Packed UNI-v1 index lacks {len(missing)} {arm} slides; first={missing[:5]}"
            )
        coverage[arm] = {"requested_slides": len(wanted), "missing_slides": 0}
    return {
        "path": str(packed.resolve()),
        "encoder": "UNI-v1",
        "feature_dim": 1_024,
        "artifacts": artifacts,
        "coverage": coverage,
    }


def _command(root: Path, arm: str, seed: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_train-one",
        "--output-root",
        str(root),
        "--arm",
        arm,
        "--seed",
        str(seed),
    ]


def build_training_jobs(root: Path) -> list[dict[str, Any]]:
    root = assert_safe_output_root(root)
    jobs = []
    for arm, spec in ARMS.items():
        for seed in SEEDS:
            directory = run_dir(root, arm, seed)
            jobs.append(
                {
                    "job_id": f"aim1.primary_source_cv.{arm}.seed{seed}",
                    "arm": arm,
                    "seed": seed,
                    "fit_count": N_FOLDS,
                    "refit_count": 0,
                    "expected_slides": spec.expected_slides,
                    "expected_patients": spec.expected_patients,
                    "output": str(directory),
                    "command": _command(root, arm, seed),
                    "training_command": [
                        sys.executable,
                        str(REPO / "tools/study_train.py"),
                        "hydra-train",
                        *_study_overrides(root, arm, seed, directory),
                    ],
                }
            )
    jobs.sort(key=lambda item: (-int(item["expected_patients"]), int(item["seed"]), str(item["arm"])))
    if (
        len(jobs) != TOTAL_JOBS
        or sum(int(job["fit_count"]) for job in jobs) != TOTAL_FITS
        or sum(int(job["refit_count"]) for job in jobs) != 0
        or {(str(job["arm"]), int(job["seed"])) for job in jobs}
        != {(arm, seed) for arm in ARMS for seed in SEEDS}
    ):
        raise ContractError("Exact 20-job/100-fit/zero-refit inventory drifted")
    return jobs


def build_job_inventory(root: Path = DEFAULT_OUTPUT_ROOT) -> list[dict[str, Any]]:
    """Public deterministic inventory: 20 seed chains and 100 OOF fold fits."""

    return build_training_jobs(Path(root))


def build_contract(
    root: Path = DEFAULT_OUTPUT_ROOT, *, created_utc: str | None = None
) -> dict[str, Any]:
    """Return the filesystem-independent public study contract.

    This dry contract is safe to build before ``prepare``.  The persisted
    contract extends it with content identities for the master and derived
    inputs, feature store, and complete implementation snapshot.
    """

    root = assert_safe_output_root(Path(root))
    jobs = build_job_inventory(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "created_utc": created_utc or _utcnow(),
        "output_root": str(root),
        "arms": list(ARM_SPECS),
        "arm_specs": {
            arm: {
                **dataclasses.asdict(spec),
                "expected_test_patients": list(spec.expected_test_patients),
            }
            for arm, spec in ARM_SPECS.items()
        },
        "expected_arm_census": EXPECTED_ARM_CENSUS,
        "model_seeds": list(SEEDS),
        "folds": list(FOLDS),
        "fit_accounting": {
            "jobs": len(jobs),
            "fold_fits_per_job": len(FOLDS),
            "oof_fits": sum(int(job["fit_count"]) for job in jobs),
            "refits": sum(int(job["refit_count"]) for job in jobs),
            "total_fits": sum(
                int(job["fit_count"]) + int(job["refit_count"]) for job in jobs
            ),
        },
        "execution": {
            "max_concurrent_gpu_trainers": MAX_CONCURRENT_GPU_TRAINERS,
            "scheduler_unit": "one seed-chain process executing five folds",
        },
        "recipe": {
            "encoder": "UNI-v1",
            "model": "gated ABMIL",
            "dataset_max_instances": CAP,
            "patient_natural_sampling": True,
            "native_logits": True,
            "skip_finalize": True,
        },
        "jobs": jobs,
    }


def _contract_semantics(root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "objective": "Aim-1 OOF five-fold CV on four primary-cohort training populations",
        "seeds": list(SEEDS),
        "n_folds": N_FOLDS,
        "arms": list(ARMS),
        "job_count": TOTAL_JOBS,
        "logical_fit_count": TOTAL_FITS,
        "refit_count": 0,
        "max_parallel_training_processes": MAX_WORKERS,
        "split_policy": "immutable row subsets of frozen Aim-1 patient-level folds; never redraw",
        "training_policy": "five OOF folds only; training.skip_finalize=true; no final/refit",
        "prediction_scale": "native logits",
        "aggregation_scope": "patient-level mean of slide native logits; analysis is post-training",
        "jobs": build_training_jobs(root),
    }


def _build_contract(root: Path, arm_records: dict[str, Any]) -> dict[str, Any]:
    return {
        **_contract_semantics(root),
        "created_utc": _utcnow(),
        "master_inputs": _master_identities(),
        "arm_definitions": {
            arm: {**dataclasses.asdict(spec), "expected_test_patients": list(spec.expected_test_patients)}
            for arm, spec in ARMS.items()
        },
        "derived_inputs": arm_records,
        "feature_store": _packed_store_identity(root),
        "material_recipe": {
            "encoder": "UNI-v1",
            "model": "gated ABMIL",
            "embed_dim": 512,
            "attn_dim": 384,
            "input_dropout": 0.10,
            "dropout": float(paths.DROPOUT),
            "lr": float(paths.LR),
            "weight_decay": float(paths.WEIGHT_DECAY),
            "precision": "bf16-mixed",
            "dataset_max_instances": CAP,
            "eval_full_bags": True,
            "train_sampling_strategy": "patient_natural",
            "sample_weight_column": None,
            "skip_finalize": True,
        },
        "implementation_sources": _source_snapshot(),
    }


def _iter_artifacts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            yield value
        else:
            for child in value.values():
                yield from _iter_artifacts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_artifacts(child)


def _validate_artifact(identity: Mapping[str, Any], *, deep: bool) -> None:
    path = Path(str(identity.get("path", "")))
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"Contracted artifact missing or symlinked: {path}")
    if int(path.stat().st_size) != int(identity.get("size_bytes", -1)):
        raise ContractError(f"Contracted artifact size drifted: {path}")
    if deep and _artifact(path) != dict(identity):
        raise ContractError(f"Contracted artifact hash drifted: {path}")


def validate_contract(root: Path, *, deep: bool) -> dict[str, Any]:
    root = assert_safe_output_root(root)
    contract = _read_json(contract_path(root))
    semantics = _contract_semantics(root)
    mismatch = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in semantics.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ContractError(f"Contract semantics drifted: {mismatch}")
    if contract.get("arm_definitions") != {
        arm: {**dataclasses.asdict(spec), "expected_test_patients": list(spec.expected_test_patients)}
        for arm, spec in ARMS.items()
    }:
        raise ContractError("Arm definitions/censuses drifted")
    if contract.get("master_inputs") != _master_identities():
        raise ContractError("Frozen master input identities drifted")
    for identity in _iter_artifacts(contract):
        _validate_artifact(identity, deep=deep)
    for arm in ARMS:
        frame = pd.read_csv(manifest_path(root, arm), low_memory=False)
        census = _validate_arm_frame(frame, arm)
        split = _validate_derived_split(root, arm, frame)
        expected = {
            "manifest": _artifact(manifest_path(root, arm)),
            "census": census,
            "derived_split": split,
        }
        if contract.get("derived_inputs", {}).get(arm) != expected:
            raise ContractError(f"{arm}: derived input contract drifted")
    return contract


def cmd_plan(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    jobs = build_training_jobs(root)
    print(
        json.dumps(
            {
                "status": "PLAN_ONLY_NO_WRITES",
                "output_root": str(root),
                "arms": list(ARMS),
                "seeds": list(SEEDS),
                "folds_per_job": N_FOLDS,
                "jobs": len(jobs),
                "logical_fits": sum(int(job["fit_count"]) for job in jobs),
                "refits": sum(int(job["refit_count"]) for job in jobs),
                "max_parallel": MAX_WORKERS,
                "job_inventory": jobs,
            },
            indent=2,
        )
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    if not args.apply:
        raise ContractError("prepare requires --apply")
    root = assert_safe_output_root(args.output_root)
    if contract_path(root).exists():
        validate_contract(root, deep=True)
        print(f"PASS cached immutable contract: {contract_path(root)}")
        return
    _master_identities()
    master = pd.read_csv(MASTER_MANIFEST, low_memory=False)
    required = {"slide_id", "patient_id", "target_label", "specimen_role", "cohort", "subcohort"}
    if required - set(master.columns):
        raise ContractError(f"Master manifest lacks columns {sorted(required - set(master.columns))}")
    arm_frames: dict[str, pd.DataFrame] = {}
    for arm in ARMS:
        frame = master.loc[_arm_mask(master, arm)].copy().reset_index(drop=True)
        _validate_arm_frame(frame, arm)
        _write_text_once(manifest_path(root, arm), frame.to_csv(index=False))
        arm_frames[arm] = frame
        directory = split_dir(root, arm)
        if directory.exists() and not (directory / "splits.parquet").is_file():
            raise ContractError(f"Refusing incomplete pre-existing derived split directory: {directory}")
        derive_subset_splits(
            MASTER_SPLITS_DIR,
            manifest_path(root, arm),
            directory,
            filename_column="slide_id",
            force=False,
        )
    # Explicitly certify the nesting relationships requested by the design.
    ids = {arm: set(frame["slide_id"].astype(str)) for arm, frame in arm_frames.items()}
    if not ids["sr386_primary"] < ids["surgen_primary"] < ids["tcga_surgen_primary"]:
        raise ContractError("Expected strict SR386 < SurGen < TCGA+SurGen nesting failed")
    if not ids["tcga_primary"] < ids["tcga_surgen_primary"]:
        raise ContractError("Expected strict TCGA < TCGA+SurGen nesting failed")
    if ids["tcga_primary"] & ids["surgen_primary"]:
        raise ContractError("TCGA and SurGen source populations are not disjoint")
    arm_records = {
        arm: {
            "manifest": _artifact(manifest_path(root, arm)),
            "census": _validate_arm_frame(arm_frames[arm], arm),
            "derived_split": _validate_derived_split(root, arm, arm_frames[arm]),
        }
        for arm in ARMS
    }
    _write_json_once(contract_path(root), _build_contract(root, arm_records))
    validate_contract(root, deep=True)
    print(f"PASS sealed contract: 4 arms / 20 jobs / 100 OOF fits / 0 refits at {root}")


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    if not current.exists():
        raise ContractError(f"No existing parent for output path: {path}")
    return current


def _available_ram_gib() -> float:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", maxsplit=1)
        values[key] = int(value.strip().split()[0])
    return float(values.get("MemAvailable", 0)) / 1024**2


def _gpu_resources() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--id=0",
        "--query-gpu=name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise ContractError(f"GPU resource query failed: {result.stderr.strip()}")
    fields = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
    applications = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if applications.returncode != 0:
        raise ContractError(
            f"GPU compute-application query failed: {applications.stderr.strip()}"
        )
    active_applications = [
        line.strip() for line in applications.stdout.splitlines() if line.strip()
    ]
    return {
        "gpu_name": fields[0],
        "total_gpu_mib": int(fields[1]),
        "free_gpu_mib": int(fields[2]),
        "gpu_utilization_percent": int(fields[3]),
        "compute_application_count": len(active_applications),
        "compute_applications": active_applications,
    }


def _resource_preflight(root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(_nearest_existing_parent(root))
    observed = {
        "logical_cpus": int(os.cpu_count() or 0),
        "available_ram_gib": _available_ram_gib(),
        "free_disk_gib": float(disk.free) / 1024**3,
        **_gpu_resources(),
        "packed_feature_dir": str(paths.PACKED_FEATURE_DIR.resolve()),
        "packed_feature_dir_exists": paths.PACKED_FEATURE_DIR.is_dir(),
    }
    minima = {
        "logical_cpus": 36,
        "available_ram_gib": 96.0,
        "free_disk_gib": 100.0,
        "free_gpu_mib": 20_000,
    }
    failed = {
        key: {"minimum": minimum, "observed": observed[key]}
        for key, minimum in minima.items()
        if float(observed[key]) < minimum
    }
    if not observed["packed_feature_dir_exists"]:
        failed["packed_feature_dir_exists"] = {"minimum": True, "observed": False}
    if observed["compute_application_count"] != 0:
        failed["compute_application_count"] = {
            "required": 0,
            "observed": observed["compute_application_count"],
            "applications": observed["compute_applications"],
        }
    # Xwayland/display activity can produce a low nonzero SM reading on this host.
    # Reject real contention while tolerating the audited 16--24% idle-display band.
    if observed["gpu_utilization_percent"] > 25:
        failed["gpu_utilization_percent"] = {
            "maximum": 25,
            "observed": observed["gpu_utilization_percent"],
        }
    if failed:
        raise ContractError(f"Six-worker resource preflight failed: {failed}")
    return observed


def _validate_preflight(root: Path) -> dict[str, Any]:
    receipt = _read_json(preflight_path(root))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "deep_preflight_passed",
        "contract": _artifact(contract_path(root)),
        "job_count": TOTAL_JOBS,
        "logical_fit_count": TOTAL_FITS,
        "refit_count": 0,
        "scheduler_ceiling": MAX_WORKERS,
    }
    mismatch = {key: {"expected": value, "observed": receipt.get(key)} for key, value in expected.items() if receipt.get(key) != value}
    if mismatch:
        raise ContractError(f"Persisted preflight receipt drifted: {mismatch}")
    return receipt


def cmd_preflight(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    validate_contract(root, deep=True)
    if preflight_path(root).exists():
        _validate_preflight(root)
        print(f"PASS cached deep preflight: {preflight_path(root)}")
        return
    resources = _resource_preflight(root)
    if args.apply:
        _write_json_once(
            preflight_path(root),
            {
                "schema_version": SCHEMA_VERSION,
                "status": "deep_preflight_passed",
                "created_utc": _utcnow(),
                "contract": _artifact(contract_path(root)),
                "job_count": TOTAL_JOBS,
                "logical_fit_count": TOTAL_FITS,
                "refit_count": 0,
                "scheduler_ceiling": MAX_WORKERS,
                "resources": resources,
            },
        )
        _validate_preflight(root)
    print(json.dumps({"status": "PASS", "persisted": bool(args.apply), "resources": resources}, indent=2))


def _study_overrides(root: Path, arm: str, seed: int, directory: Path) -> list[str]:
    spec = ARMS[arm]
    return [
        "platform=colon_workstation",
        "data=aim1",
        f"data.aim1_model={spec.model_name}",
        f"data.name=aim1_{spec.model_name}",
        f"data.manifest_stem={manifest_path(root, arm).stem}",
        f"data.csv_path={manifest_path(root, arm)}",
        f"platform.splits_root={split_root(root)}",
        "+data.cohort_column=cohort",
        "encoder=univ1",
        "splits=aim1_balanced",
        f"splits.seed={seed}",
        "model=abmil",
        "model.embed_dim=512",
        "model.attn_dim=384",
        "model.input_dropout=0.10",
        f"model.dropout={float(paths.DROPOUT):g}",
        "training=aim1",
        f"training.lr={float(paths.LR):g}",
        f"training.weight_decay={float(paths.WEIGHT_DECAY):g}",
        f"training.seed={seed}",
        f"training.dataset_max_instances={CAP}",
        "training.eval_full_bags=true",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        "training.skip_finalize=true",
        f"train_dir={directory}",
        f"exp_name=aim1_primary_cohort_oof_{arm}_c{CAP}_s{seed}",
        f"hydra.run.dir={root / 'hydra_runs' / f'{arm}_seed{seed}'}",
        "hydra.job.chdir=false",
    ]


def _nested(record: Mapping[str, Any], dotted: str) -> Any:
    value: Any = record
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return "<missing>"
        value = value[key]
    return value


def _material_expectations(root: Path, arm: str, seed: int) -> dict[str, Any]:
    spec = ARMS[arm]
    return {
        "data.name": f"aim1_{spec.model_name}",
        "data.aim1_model": spec.model_name,
        "data.csv_path": str(manifest_path(root, arm).resolve()),
        "data.label_columns": ["target_label"],
        "data.patient_id_column": "patient_id",
        "data.filename_column": "slide_id",
        "data.cohort_column": "cohort",
        "data.num_classes": 2,
        "encoder.name": "uni_v1",
        "encoder.feature_dim": 1_024,
        "splits.scheme": "predefined_oof_kfold",
        "splits.name": "aim1_balanced5",
        "splits.fold_column": "k_fold",
        "splits.group_column": "patient_id",
        "splits.allow_group_overlap": False,
        "splits.n_folds": N_FOLDS,
        "splits.seed": seed,
        "model.name": "abmil",
        "model.arch": "abmil",
        "model.embed_dim": 512,
        "model.attn_dim": 384,
        "model.gate": True,
        "model.dropout": float(paths.DROPOUT),
        "model.input_dropout": 0.10,
        "training.lr": float(paths.LR),
        "training.weight_decay": float(paths.WEIGHT_DECAY),
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
        "training.seed": seed,
        "training.skip_finalize": True,
        "training.refit_max_steps": None,
        "platform.precision": "bf16-mixed",
        "platform.devices": 1,
    }


def _validate_material_config(config: Mapping[str, Any], root: Path, arm: str, seed: int, *, context: str) -> None:
    mismatch = {
        dotted: {"expected": expected, "observed": _nested(config, dotted)}
        for dotted, expected in _material_expectations(root, arm, seed).items()
        if _nested(config, dotted) != expected
        and not (dotted == "training.refit_max_steps" and expected is None and _nested(config, dotted) == "<missing>")
    }
    if mismatch:
        raise ContractError(f"{arm}/seed{seed}: {context} recipe mismatch: {mismatch}")


def _native_validation(root: Path, arm: str, seed: int) -> dict[str, Any]:
    import yaml

    from oceanpath.workflows.training import validate_training_run_dir

    directory = run_dir(root, arm, seed)
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(f"Missing immutable run directory: {directory}")
    completion = validate_training_run_dir(directory, require_test_predictions=True)
    folds = completion.get("fold_completions")
    expected_paths = [f"fold_{fold}/completion.json" for fold in range(N_FOLDS)]
    observed_paths = [item.get("path") for item in folds] if isinstance(folds, list) else []
    if (
        int(completion.get("n_folds", -1)) != N_FOLDS
        or completion.get("skip_finalize") is not True
        or observed_paths != expected_paths
    ):
        raise ContractError(f"{arm}/seed{seed}: native completion is not exact five-fold/no-refit OOF")
    if (directory / "final").exists() or list(directory.rglob("*refit*")):
        raise ContractError(f"{arm}/seed{seed}: forbidden final/refit artifact exists")
    identity = _read_json(directory / "training_identity.json")
    if identity.get("fingerprint") != completion.get("training_fingerprint"):
        raise ContractError(f"{arm}/seed{seed}: training fingerprint mismatch")
    payload = identity.get("payload") or {}
    material = payload.get("material_config")
    if not isinstance(material, dict):
        raise ContractError(f"{arm}/seed{seed}: missing material training identity")
    _validate_material_config(material, root, arm, seed, context="training identity")
    evidence = payload.get("input_evidence") or {}
    expected_evidence = {
        "manifest_sha256": _sha256(manifest_path(root, arm)),
        "split_integrity_sha256": _sha256(split_dir(root, arm) / ".integrity_hash"),
    }
    if any(evidence.get(key) != value for key, value in expected_evidence.items()):
        raise ContractError(f"{arm}/seed{seed}: manifest/split evidence mismatch")
    manifest = pd.read_csv(manifest_path(root, arm), low_memory=False)
    for fold in range(N_FOLDS):
        config_path = directory / f"fold_{fold}/config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ContractError(f"Malformed resolved config: {config_path}")
        _validate_material_config(config, root, arm, seed, context=f"fold {fold}")
        if Path(str(_nested(config, "platform.splits_root"))).resolve() != split_root(root).resolve():
            raise ContractError(f"{arm}/seed{seed}/fold{fold}: wrong split root")
        masks = {
            "test": pd.to_numeric(manifest["k_fold"]).astype(int).eq(fold),
            "val": pd.to_numeric(manifest[f"val_fold_{fold}"]).astype(int).eq(1),
        }
        for role, mask in masks.items():
            predictions = pd.read_parquet(directory / f"fold_{fold}/preds_{role}.parquet")
            expected = manifest.loc[mask, ["slide_id", "target_label"]].copy()
            joined = predictions.merge(expected, on="slide_id", how="inner", validate="one_to_one")
            if (
                len(predictions) != len(expected)
                or len(joined) != len(expected)
                or set(predictions["slide_id"].astype(str)) != set(expected["slide_id"].astype(str))
                or not pd.to_numeric(joined["label"]).astype(int).eq(pd.to_numeric(joined["target_label"]).astype(int)).all()
                or not np.isfinite(pd.to_numeric(predictions["logit"], errors="coerce")).all()
            ):
                raise ContractError(f"{arm}/seed{seed}/fold{fold}: exact {role} roster/label/logit drift")
    oof = pd.read_parquet(directory / "oof_predictions.parquet")
    expected_slides = set(manifest["slide_id"].astype(str))
    if (
        len(oof) != len(manifest)
        or oof["slide_id"].astype(str).duplicated().any()
        or set(oof["slide_id"].astype(str)) != expected_slides
        or set(pd.to_numeric(oof["fold"], errors="coerce").astype(int)) != set(range(N_FOLDS))
        or not np.isfinite(pd.to_numeric(oof["logit"], errors="coerce")).all()
    ):
        raise ContractError(f"{arm}/seed{seed}: incomplete or invalid OOF roster")
    check = oof.merge(
        manifest[["slide_id", "target_label", "k_fold"]],
        on="slide_id", how="inner", validate="one_to_one",
    )
    if (
        not pd.to_numeric(check["label"]).astype(int).eq(pd.to_numeric(check["target_label"]).astype(int)).all()
        or not pd.to_numeric(check["fold"]).astype(int).eq(pd.to_numeric(check["k_fold"]).astype(int)).all()
    ):
        raise ContractError(f"{arm}/seed{seed}: OOF label or fold mapping drift")
    return {
        "completion": _artifact(directory / "training_completion.json"),
        "identity": _artifact(directory / "training_identity.json"),
        "oof": _artifact(directory / "oof_predictions.parquet"),
        "cv_summary": _artifact(directory / "cv_summary.json"),
        "training_fingerprint": completion.get("training_fingerprint"),
        "fold_count": N_FOLDS,
        "refit_count": 0,
        "oof_rows": int(len(oof)),
    }


def _validate_job(root: Path, arm: str, seed: int) -> dict[str, Any]:
    native = _native_validation(root, arm, seed)
    receipt = _read_json(job_receipt_path(root, arm, seed))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "arm": arm,
        "seed": seed,
        "fit_count": N_FOLDS,
        "refit_count": 0,
        "contract": _artifact(contract_path(root)),
        "artifacts": native,
    }
    mismatch = {key: {"expected": value, "observed": receipt.get(key)} for key, value in expected.items() if receipt.get(key) != value}
    if mismatch:
        raise ContractError(f"{arm}/seed{seed}: job receipt mismatch: {mismatch}")
    return receipt


def _run_logged(command: Sequence[str], output_log: Path) -> int:
    if output_log.exists() or output_log.is_symlink():
        raise ContractError(f"Refusing to overwrite immutable log: {output_log}")
    output_log.parent.mkdir(parents=True, exist_ok=True)
    with output_log.open("x", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen(
            list(command), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        return int(process.wait())


@contextlib.contextmanager
def _kernel_lock(path: Path, *, context: str) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f"Another process owns {context}: {path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, (json.dumps({"pid": os.getpid(), "context": context, "utc": _utcnow()}) + "\n").encode())
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _job_lock_path(root: Path, arm: str, seed: int) -> Path:
    token = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    return Path(f"/tmp/oceanpath_{CAMPAIGN}_{token}_{arm}_s{seed}.lock")


def cmd_internal_train_one(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    if args.arm not in ARMS or args.seed not in SEEDS:
        raise ContractError("Uncontracted arm/seed")
    validate_contract(root, deep=False)
    _validate_preflight(root)
    arm, seed = args.arm, args.seed
    with _kernel_lock(_job_lock_path(root, arm, seed), context=f"{arm}/seed{seed}"):
        if job_receipt_path(root, arm, seed).is_file():
            _validate_job(root, arm, seed)
            print(f"PASS cached: {arm}/seed{seed}")
            return
        directory = run_dir(root, arm, seed)
        request = request_path(root, arm, seed)
        log = log_path(root, arm, seed)
        if directory.exists() or request.exists() or log.exists():
            # A native-complete run can survive termination between fitting and receipt publication.
            try:
                native = _native_validation(root, arm, seed)
            except Exception as exc:
                raise ContractError(
                    f"Refusing to overwrite partial immutable job {arm}/seed{seed}; "
                    "inspect it and use a fresh campaign root"
                ) from exc
            _write_json_once(
                job_receipt_path(root, arm, seed),
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "completed",
                    "finished_utc": _utcnow(),
                    "arm": arm,
                    "seed": seed,
                    "fit_count": N_FOLDS,
                    "refit_count": 0,
                    "contract": _artifact(contract_path(root)),
                    "request": _artifact(request),
                    "log": _artifact(log),
                    "artifacts": native,
                },
            )
            _validate_job(root, arm, seed)
            return
        _write_json_once(
            request,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "requested",
                "created_utc": _utcnow(),
                "arm": arm,
                "seed": seed,
                "fit_count": N_FOLDS,
                "refit_count": 0,
                "manifest": _artifact(manifest_path(root, arm)),
                "split": _artifact(split_dir(root, arm) / "splits.parquet"),
                "contract": _artifact(contract_path(root)),
                "output": str(directory),
            },
        )
        command = [
            sys.executable,
            str(REPO / "tools/study_train.py"),
            "hydra-train",
            *_study_overrides(root, arm, seed, directory),
        ]
        returncode = _run_logged(command, log)
        if returncode:
            _write_json_once(
                failure_path(root, arm, seed),
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "returncode": returncode,
                    "request": _artifact(request),
                    "log": _artifact(log),
                },
            )
            raise SystemExit(returncode)
        native = _native_validation(root, arm, seed)
        _write_json_once(
            job_receipt_path(root, arm, seed),
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "finished_utc": _utcnow(),
                "arm": arm,
                "seed": seed,
                "fit_count": N_FOLDS,
                "refit_count": 0,
                "contract": _artifact(contract_path(root)),
                "request": _artifact(request),
                "log": _artifact(log),
                "artifacts": native,
            },
        )
        _validate_job(root, arm, seed)
        print(f"PASS completed: {arm}/seed{seed} (5 OOF fits, zero refits)")


def _execute_job(job: Mapping[str, Any]) -> dict[str, Any]:
    started_wall = _utcnow()
    started = time.monotonic()
    process = subprocess.Popen(list(job["command"]), cwd=REPO)
    returncode = int(process.wait())
    return {
        "job_id": str(job["job_id"]),
        "pid": int(process.pid),
        "returncode": returncode,
        "started_utc": started_wall,
        "finished_utc": _utcnow(),
        "started_monotonic": started,
        "finished_monotonic": time.monotonic(),
    }


def _peak_parallel(events: Sequence[Mapping[str, Any]]) -> int:
    points = []
    for event in events:
        points.append((float(event["started_monotonic"]), 1))
        points.append((float(event["finished_monotonic"]), -1))
    active = peak = 0
    for _, delta in sorted(points, key=lambda item: (item[0], -item[1])):
        active += delta
        peak = max(peak, active)
    return peak


@contextlib.contextmanager
def _launcher_locks(root: Path) -> Iterable[None]:
    token = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    launcher = Path(f"/tmp/oceanpath_{CAMPAIGN}_{token}_launcher.lock")
    with (
        _kernel_lock(launcher, context="Aim-1 primary-cohort launcher"),
        _kernel_lock(GLOBAL_GPU_LOCK, context="study-wide GPU-0 training"),
    ):
        yield


def cmd_train(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    validate_contract(root, deep=False)
    jobs = build_training_jobs(root)
    if args.arm:
        jobs = [job for job in jobs if job["arm"] == args.arm]
    if args.seed is not None:
        jobs = [job for job in jobs if int(job["seed"]) == args.seed]
    if not args.apply:
        print(
            f"DRY RUN: {len(jobs)} chains / {sum(int(job['fit_count']) for job in jobs)} fits / "
            f"0 refits; max_workers={args.max_workers}"
        )
        for job in jobs:
            print(json.dumps(job, sort_keys=True))
        return
    _validate_preflight(root)
    if not 1 <= args.max_workers <= MAX_WORKERS:
        raise ContractError(f"--max-workers must be 1..{MAX_WORKERS}")
    if root == DEFAULT_OUTPUT_ROOT.resolve() and args.max_workers != MAX_WORKERS:
        raise ContractError(f"Production campaign requires exactly --max-workers {MAX_WORKERS}")
    events = []
    failures = []
    with (
        _launcher_locks(root),
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool,
    ):
        futures = {pool.submit(_execute_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            event = future.result()
            events.append(event)
            print(f"{event['job_id']}: returncode={event['returncode']}", flush=True)
            if event["returncode"]:
                failures.append((event["job_id"], event["returncode"]))
    if failures:
        raise SystemExit(f"Training failures; immutable evidence retained: {failures}")
    # Only the complete unfiltered production launch seals scheduler evidence.
    if not args.arm and args.seed is None and len(jobs) == TOTAL_JOBS:
        peak = _peak_parallel(events)
        if peak > MAX_WORKERS:
            raise ContractError(f"Observed scheduler ceiling violation: peak={peak}")
        if root == DEFAULT_OUTPUT_ROOT.resolve() and peak != MAX_WORKERS:
            raise ContractError(f"Production launch did not demonstrate six-way execution: peak={peak}")
        _write_json_once(
            root / "receipts/scheduler.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed_rc0",
                "created_utc": _utcnow(),
                "configured_max_workers": args.max_workers,
                "observed_peak_parallel_workers": peak,
                "job_count": len(events),
                "logical_fit_count": TOTAL_FITS,
                "refit_count": 0,
                "events": sorted(events, key=lambda item: str(item["job_id"])),
            },
        )


def cmd_validate(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    validate_contract(root, deep=True)
    receipts = []
    fingerprints = set()
    for arm in ARMS:
        for seed in SEEDS:
            receipt = _validate_job(root, arm, seed)
            receipts.append(_artifact(job_receipt_path(root, arm, seed)))
            fingerprints.add(receipt["artifacts"]["training_fingerprint"])
    if len(receipts) != TOTAL_JOBS or len(fingerprints) != TOTAL_JOBS:
        raise ContractError("Expected 20 independently fingerprinted completed jobs")
    scheduler = _read_json(root / "receipts/scheduler.json")
    if (
        scheduler.get("configured_max_workers") != MAX_WORKERS
        or scheduler.get("observed_peak_parallel_workers") != MAX_WORKERS
        or scheduler.get("job_count") != TOTAL_JOBS
        or scheduler.get("logical_fit_count") != TOTAL_FITS
        or scheduler.get("refit_count") != 0
    ):
        raise ContractError("Scheduler receipt does not prove exact six-way 100-fit execution")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_and_certified",
        "created_utc": _utcnow(),
        "contract": _artifact(contract_path(root)),
        "preflight": _artifact(preflight_path(root)),
        "scheduler": _artifact(root / "receipts/scheduler.json"),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "job_count": TOTAL_JOBS,
        "folds_per_job": N_FOLDS,
        "logical_fit_count": TOTAL_FITS,
        "refit_count": 0,
        "job_receipts": receipts,
    }
    if args.seal:
        _write_json_once(training_receipt_path(root), payload)
    print(json.dumps({"status": "PASS", "sealed": bool(args.seal), "fits": TOTAL_FITS, "refits": 0}, indent=2))


def _lock_active(path: Path) -> bool:
    if not path.exists():
        return False
    descriptor = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def cmd_status(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    jobs = build_training_jobs(root)
    rows = []
    for job in jobs:
        arm, seed = str(job["arm"]), int(job["seed"])
        receipt = job_receipt_path(root, arm, seed)
        state = "pending"
        if receipt.is_file():
            try:
                _validate_job(root, arm, seed)
            except Exception as exc:
                state = f"invalid:{type(exc).__name__}"
            else:
                state = "completed"
        elif _lock_active(_job_lock_path(root, arm, seed)):
            state = "running"
        elif failure_path(root, arm, seed).is_file():
            state = "failed"
        elif run_dir(root, arm, seed).exists() or request_path(root, arm, seed).exists():
            state = "partial_or_orphaned"
        rows.append({"job_id": job["job_id"], "arm": arm, "seed": seed, "state": state})
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    complete = counts.get("completed", 0)
    print(
        json.dumps(
            {
                "output_root": str(root),
                "prepared": contract_path(root).is_file(),
                "preflight_passed": preflight_path(root).is_file(),
                "training_sealed": training_receipt_path(root).is_file(),
                "jobs": counts,
                "certified_logical_fits": complete * N_FOLDS,
                "remaining_logical_fits": TOTAL_FITS - complete * N_FOLDS,
                "rows": rows,
            },
            indent=2,
        )
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Fail closed until the separately governed v10.5 analysis is attached."""

    root = assert_safe_output_root(args.output_root)
    if not training_receipt_path(root).is_file():
        raise ContractError(
            "Analysis requires the sealed 100-fit training receipt; run validate --seal first"
        )
    raise ContractError(
        "Phase-1 intentionally stops at certified OOF logits; attach the governed "
        "FINAL-v10.5 analysis/report builder before publishing metrics"
    )


def cmd_verify(args: argparse.Namespace) -> None:
    """Deeply verify all currently sealed campaign stages without mutation."""

    root = assert_safe_output_root(args.output_root)
    validate_contract(root, deep=True)
    stages = {"contract": "PASS"}
    if preflight_path(root).is_file():
        _validate_preflight(root)
        stages["preflight"] = "PASS"
    if training_receipt_path(root).is_file():
        validation_args = argparse.Namespace(output_root=root, seal=False)
        cmd_validate(validation_args)
        stages["training"] = "PASS"
    print(json.dumps({"status": "PASS", "stages": stages}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", parents=[common])
    plan.set_defaults(func=cmd_plan)

    prepare = subparsers.add_parser("prepare", parents=[common])
    prepare.add_argument("--apply", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    preflight = subparsers.add_parser("preflight", parents=[common])
    preflight.add_argument("--apply", action="store_true")
    preflight.set_defaults(func=cmd_preflight)

    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--apply", action="store_true")
    train.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    train.add_argument("--arm", choices=list(ARMS))
    train.add_argument("--seed", type=int, choices=SEEDS)
    train.set_defaults(func=cmd_train)

    validate = subparsers.add_parser("validate", parents=[common])
    validate.add_argument("--seal", action="store_true")
    validate.set_defaults(func=cmd_validate)

    analyze = subparsers.add_parser("analyze", parents=[common])
    analyze.set_defaults(func=cmd_analyze)

    verify = subparsers.add_parser("verify", parents=[common])
    verify.set_defaults(func=cmd_verify)

    status = subparsers.add_parser("status", parents=[common])
    status.set_defaults(func=cmd_status)

    internal = subparsers.add_parser("_train-one", parents=[common])
    internal.add_argument("--arm", required=True, choices=list(ARMS))
    internal.add_argument("--seed", required=True, type=int, choices=SEEDS)
    internal.set_defaults(func=cmd_internal_train_one)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
