#!/usr/bin/env python3
"""Governed Aim-1 TCGA+SurGen five-seed, two-encoder campaign.

This additive campaign adopts the sealed five UNI-v1 OOF chains from the
FINAL-v10.5 source-cohort campaign and creates only the missing artifacts:

* one full-source p75 refit for each adopted UNI-v1 seed (5 fits), and
* five Virchow2-CLS chains, each with five OOF folds and one p75 refit
  (25 OOF fits + 5 refits).

The resulting operational lineage is 25 adopted fits + 35 new fits = 60 fits.
There are exactly ten independently scheduled jobs and at most six may run at
once.  No command retries a partial job.  The public workflow is::

    uv run python tools/aim1_tcga_surgen_two_encoder_campaign.py plan
    uv run python tools/aim1_tcga_surgen_two_encoder_campaign.py prepare --apply
    uv run python tools/aim1_tcga_surgen_two_encoder_campaign.py preflight --apply
    uv run python tools/aim1_tcga_surgen_two_encoder_campaign.py train --apply --max-workers 6
    uv run python tools/aim1_tcga_surgen_two_encoder_campaign.py validate --seal
    uv run python tools/aim1_tcga_surgen_two_encoder_campaign.py verify

``plan`` and a dry ``train`` do not write.  ``prepare`` snapshots the already
sealed TCGA+SurGen manifest/splits byte-for-byte and authenticates every
adopted OOF chain.  All control-plane evidence is write-once.
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
import math
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools import aim1_source_cohort_five_seed_campaign as adopted_campaign  # noqa: E402

SCHEMA_VERSION = 1
CAMPAIGN = "aim1_tcga_surgen_two_encoder_5seed"
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim1_tcga_surgen_two_encoder_v1_20260824"
)
ADOPTED_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim1_primary_cohort_5seed_v1_20260824"
)
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
CAP = 8_192
MAX_WORKERS = 6
TOTAL_JOBS = 10
ADOPTED_OOF_FITS = 25
NEW_OOF_FITS = 25
NEW_REFITS = 10
NEW_FITS = 35
OPERATIONAL_LINEAGE_FITS = 60
GLOBAL_GPU_LOCK = Path("/tmp/oceanpath_gpu0_exclusive.lock")

ADOPTED_CONTROLLER_SHA256 = (
    "242b3be7823046d2e654165676992e59b57d8f246c9637bd28904d816bce5121"
)
ADOPTED_CONTRACT_SHA256 = (
    "fcc5a6744da27b73c5771f71a05511e998d20b3b4d76ecdc8a9c8c4ce34f0973"
)
ADOPTED_TERMINAL_SHA256 = (
    "31cc8005abf32cb1c8958d0b4ecb5ed34e927bcd1665451869d62a913bb75413"
)
ADOPTED_JOB_RECEIPT_SHA256 = {
    42: "6450a83bf13a4fca1912001214045b359e848406bc3e3d6e040729210f9bad94",
    43: "94678c39b303fa458748126d284cebabad36efc0b74bef5a2f250e9c7ed87147",
    44: "561c3a0fd805cd47fe87f0a0f55e0210adcba07f42806c7e9cd7667d49aa36d3",
    45: "454dc92ba6f5f9382829d537bc0f734bfea421a5185841a8ab541378034ec93a",
    46: "654a00ccb3ed7c00b455aedd7ffa5ba86d2b26b15c8b70049f1e43a24f3df62d",
}

SOURCE_MANIFEST = ADOPTED_ROOT / "inputs/manifests/tcga_surgen_primary.csv"
SOURCE_SPLIT_DIR = (
    ADOPTED_ROOT
    / "inputs/splits/aim1_primary_tcga_surgen/aim1_balanced5"
)
EXPECTED_SOURCE_INPUTS = {
    "manifest": {
        "path": SOURCE_MANIFEST,
        "sha256": "d7087a23a84a294670080eb5090f83612f57b3376c7696ed2d0094fb9bcff8a5",
        "size_bytes": 440_060,
    },
    "splits": {
        "path": SOURCE_SPLIT_DIR / "splits.parquet",
        "sha256": "31046858b74a9e435ce7ceac263630e2966ab589c36a8693dcb0fc1060cfa9f4",
        "size_bytes": 114_658,
    },
    "integrity": {
        "path": SOURCE_SPLIT_DIR / ".integrity_hash",
        "sha256": "4d21bd756482ceaaa5a6cb683d4797e49bc710fad0ed8a396e9a0368c27ee2a2",
        "size_bytes": 219,
    },
    "summary": {
        "path": SOURCE_SPLIT_DIR / "summary.json",
        "sha256": "5bc1ec9b2ce07f5af68079ef426e554e930943d23d8eb70a8a7a5358f40d9948",
        "size_bytes": 1_185,
    },
}
EXPECTED_CENSUS = {
    "slides": 1_389,
    "patients": 1_239,
    "mutant_patients": 501,
    "wildtype_patients": 738,
    "test_patients_by_fold": [248, 247, 248, 248, 248],
}

UNI_PACK = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/"
    "20x_256px_0px_overlap_mpp0.5/packed_uni_v1"
)
V2_PACK = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/"
    "20x_224px_0px_overlap_mpp0.5/packed_virchow2_cls_1280"
)
PACKS = {
    "univ1": {
        "path": UNI_PACK,
        "encoder": "UNI-v1",
        "feature_dim": 1_024,
        "patch_profile": "20x_256px_0px_overlap_mpp0.5",
        "artifacts": {
            "meta.json": ("44f1f0c80f4b8740c5fc82caafa96a8ead372c561fe53cda896d83972ff3c80b", 367),
            "index.parquet": ("705464267e35e454e50ea8510bc937e9f8d6a00b8b97bb9cf1e06da5fcfa9556", 66_998),
            "features.bin": ("6765d9faf30f40e212075c1a650fd34bc5ef625b3fb8e169dbc14e78c2058bd4", 49_735_520_256),
            "coords.bin": ("1a90bd95a430e433cff900c0b6b2a4f6fedcae550ae80584ee27ed7f100fe491", 194_279_376),
        },
    },
    "virchow2_cls": {
        "path": V2_PACK,
        "encoder": "Virchow2-CLS",
        "feature_dim": 1_280,
        "patch_profile": "20x_224px_0px_overlap_mpp0.5",
        "artifacts": {
            "meta.json": ("0dd3df5d0b933f40810a75b62d68a6d8297e0e592e98ab8a87847ce7417dd239", 369),
            "index.parquet": ("2b462ac49fce6ba69d70b61b69faedd4f9e68855b8914329536ba1572fa2a87d", 67_586),
            "features.bin": ("1b7adf52e2aae40dd33b9a38bedcb178a0244943218d281429bf50329eae0da7", 81_216_020_480),
            "coords.bin": ("8a9d18e75ee988ad16073cac8c5d8d467dfe01668b681ef62589e3edbbd39a28", 253_800_064),
        },
    },
}


class ContractError(RuntimeError):
    """Fail-closed violation of the immutable campaign contract."""


@dataclass(frozen=True)
class Job:
    kind: str
    encoder: str
    seed: int
    oof_fits: int
    refits: int

    @property
    def total_fits(self) -> int:
        return self.oof_fits + self.refits


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


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _assert_finite(value: Any, *, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{context}: non-finite numeric value")
    if isinstance(value, Mapping):
        for child in value.values():
            _assert_finite(child, context=context)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_finite(child, context=context)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"Missing or symlinked JSON artifact: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"Invalid strict JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON artifact must contain an object: {path}")
    _assert_finite(value, context=str(path))
    return value


def _json_text(value: Mapping[str, Any]) -> str:
    _assert_finite(value, context="JSON payload")
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_text_once(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_text(encoding="utf-8") == text:
            return
        raise ContractError(f"Refusing to replace a different sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_once(path, _json_text(value))


def _copy_once(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ContractError(f"Frozen source is missing or symlinked: {source}")
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and not destination.is_symlink() and _artifact(destination)["sha256"] == _artifact(source)["sha256"]:
            return
        raise ContractError(f"Refusing to replace a different input snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    with source.open("rb") as read, temporary.open("xb") as write:
        shutil.copyfileobj(read, write, length=8 * 1024 * 1024)
        write.flush()
        os.fsync(write.fileno())
    os.replace(temporary, destination)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
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
    for frozen in (ADOPTED_ROOT.resolve(), UNI_PACK.resolve(), V2_PACK.resolve()):
        if resolved == frozen or _is_relative_to(resolved, frozen) or _is_relative_to(frozen, resolved):
            raise ContractError(f"Output root overlaps frozen input: {frozen}")
    return resolved


def _reject_symlinks_below(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink():
        raise ContractError(f"Symlink forbidden in governed tree: {path}")
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_symlink():
                raise ContractError(f"Symlink forbidden in governed tree: {child}")


def contract_path(root: Path) -> Path:
    return root / "contract.json"


def preflight_path(root: Path) -> Path:
    return root / "receipts/deep_preflight.json"


def scheduler_path(root: Path) -> Path:
    return root / "receipts/scheduler.json"


def training_receipt_path(root: Path) -> Path:
    return root / "receipts/training_complete.json"


def manifest_path(root: Path) -> Path:
    return root / "inputs/manifests/tcga_surgen_primary.csv"


def split_dir(root: Path) -> Path:
    return root / "inputs/splits/aim1_primary_tcga_surgen/aim1_balanced5"


def adopted_run_dir(seed: int) -> Path:
    return (
        ADOPTED_ROOT
        / f"train/source_cv/cap{CAP}/tcga_surgen_primary/seed{seed}"
    )


def adopted_job_receipt(seed: int) -> Path:
    return ADOPTED_ROOT / f"receipts/source_cv/tcga_surgen_primary/seed{seed}.json"


def run_dir(root: Path, kind: str, seed: int) -> Path:
    encoder = "univ1" if kind == "univ1_refit" else "virchow2_cls"
    return root / f"train/e0/{encoder}/seed{seed}"


def job_receipt_path(root: Path, kind: str, seed: int) -> Path:
    return root / f"receipts/jobs/{kind}/seed{seed}.json"


def request_path(root: Path, kind: str, seed: int) -> Path:
    return root / f"requests/{kind}/seed{seed}.json"


def failure_path(root: Path, kind: str, seed: int) -> Path:
    return root / f"requests/{kind}/seed{seed}.failure.json"


def log_path(root: Path, kind: str, seed: int) -> Path:
    return root / f"logs/{kind}/seed{seed}.log"


def _job_specs() -> list[Job]:
    # Start the five long V2 chains first so the six-worker launch has a real,
    # auditable peak of six rather than a queue of short refits.
    return [
        *(Job("virchow2_full", "Virchow2-CLS", seed, 5, 1) for seed in SEEDS),
        *(Job("univ1_refit", "UNI-v1", seed, 0, 1) for seed in SEEDS),
    ]


def _worker_command(root: Path, job: Job) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_train-one",
        "--output-root",
        str(root),
        "--kind",
        job.kind,
        "--seed",
        str(job.seed),
    ]


def _v2_overrides(root: Path, seed: int, directory: Path) -> list[str]:
    return [
        "platform=colon_workstation",
        "data=aim1",
        "data.aim1_model=primary_tcga_surgen",
        "data.name=aim1_primary_tcga_surgen",
        f"data.manifest_stem={manifest_path(root).stem}",
        f"data.csv_path={manifest_path(root)}",
        f"platform.splits_root={root / 'inputs/splits'}",
        "+data.cohort_column=cohort",
        "encoder=virchow2",
        "encoder.feature_dim=1280",
        "extraction.coords_dir=20x_224px_0px_overlap_mpp0.5",
        "extraction.coords_subdir=20x_224px_0px_overlap_mpp0.5",
        "extraction.patch_size=224",
        "splits=aim1_balanced",
        f"splits.seed={seed}",
        "model=abmil",
        "model.embed_dim=512",
        "model.attn_dim=384",
        "model.input_dropout=0.10",
        "model.dropout=0.25",
        "training=aim1",
        "training.lr=1e-4",
        "training.weight_decay=1e-5",
        f"training.seed={seed}",
        f"training.dataset_max_instances={CAP}",
        "training.eval_full_bags=true",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        "training.max_epochs=20",
        "training.early_stopping_patience=5",
        "training.min_epoch_before_stop=10",
        "training.monitor_metric=val/patient_auroc",
        "training.monitor_mode=max",
        "training.skip_finalize=false",
        "training.final_strategies=[best_fold,refit]",
        "training.refit_epoch_rule=p75",
        f"training.packed_dir={V2_PACK}",
        "training.verify_packed_source=false",
        f"train_dir={directory}",
        f"exp_name=aim1_e0_tcga_surgen_c{CAP}_virchow2_cls_seed{seed}",
        f"hydra.run.dir={root / 'hydra_runs' / 'virchow2_cls' / f'seed{seed}'}",
        "hydra.job.chdir=false",
    ]


def build_training_jobs(root: Path = DEFAULT_OUTPUT_ROOT) -> list[dict[str, Any]]:
    root = assert_safe_output_root(Path(root))
    jobs = []
    for spec in _job_specs():
        directory = run_dir(root, spec.kind, spec.seed)
        record = {
            **dataclasses.asdict(spec),
            "total_fits": spec.total_fits,
            "job_id": f"aim1.e0.tcga_surgen.{spec.kind}.seed{spec.seed}",
            "output": str(directory),
            "command": _worker_command(root, spec),
        }
        if spec.kind == "virchow2_full":
            record["training_command"] = [
                sys.executable,
                str(REPO / "tools/study_train.py"),
                "hydra-train",
                *_v2_overrides(root, spec.seed, directory),
            ]
        else:
            record["adopted_oof_run"] = str(adopted_run_dir(spec.seed))
            record["refit_entrypoint"] = "oceanpath.workflows.finalize._run_refit"
        jobs.append(record)
    if (
        len(jobs) != TOTAL_JOBS
        or sum(int(item["oof_fits"]) for item in jobs) != NEW_OOF_FITS
        or sum(int(item["refits"]) for item in jobs) != NEW_REFITS
        or sum(int(item["total_fits"]) for item in jobs) != NEW_FITS
        or len({(item["kind"], item["seed"]) for item in jobs}) != TOTAL_JOBS
    ):
        raise ContractError("Exact 10-job / 25-OOF / 10-refit / 35-new-fit inventory drifted")
    return jobs


def build_job_inventory(root: Path = DEFAULT_OUTPUT_ROOT) -> list[dict[str, Any]]:
    """Public deterministic inventory used by tests and downstream receipts."""

    return build_training_jobs(root)


def _fit_accounting() -> dict[str, int]:
    return {
        "adopted_oof_fits": ADOPTED_OOF_FITS,
        "new_oof_fits": NEW_OOF_FITS,
        "new_refits": NEW_REFITS,
        "new_fits": NEW_FITS,
        "operational_lineage_fits": OPERATIONAL_LINEAGE_FITS,
        "hidden_fits": 0,
    }


def build_contract(
    root: Path = DEFAULT_OUTPUT_ROOT, *, created_utc: str | None = None
) -> dict[str, Any]:
    root = assert_safe_output_root(Path(root))
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "created_utc": created_utc or _utcnow(),
        "output_root": str(root),
        "objective": "Aim-1 TCGA+SurGen OOF/refit parity for UNI-v1 and Virchow2-CLS",
        "source_population": "TCGA + SurGen primary (SR386 + SR1482)",
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "encoders": ["UNI-v1", "Virchow2-CLS"],
        "job_count": TOTAL_JOBS,
        "fit_accounting": _fit_accounting(),
        "concurrency": {"maximum": MAX_WORKERS},
        "split_policy": "byte-identical inherited patient folds; never redraw",
        "adoption_policy": "adopt five sealed UNI-v1 OOF chains; add refits only",
        "retry_policy": "zero retries; any partial job requires a fresh campaign root",
        "jobs": build_training_jobs(root),
        "material_recipe": {
            "model": "gated ABMIL",
            "embed_dim": 512,
            "attention_dim": 384,
            "input_dropout": 0.10,
            "dropout": 0.25,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "max_epochs": 20,
            "early_stopping_patience": 5,
            "minimum_epoch_before_stop": 10,
            "selection": "patient AUROC",
            "dataset_max_instances": CAP,
            "eval_full_bags": True,
            "train_sampling_strategy": "patient_natural",
            "sample_weight_column": None,
            "refit_epoch_rule": "p75",
            "UNI-v1": {"patch_px": 256, "feature_dim": 1_024},
            "Virchow2-CLS": {"patch_px": 224, "feature_dim": 1_280},
        },
    }


def _expected_identity(path: Path, sha256: str, size_bytes: int | None = None) -> dict[str, Any]:
    observed = _artifact(path)
    if observed["sha256"] != sha256:
        raise ContractError(
            f"Frozen artifact SHA drifted: {path}; expected={sha256}, observed={observed['sha256']}"
        )
    if size_bytes is not None and observed["size_bytes"] != size_bytes:
        raise ContractError(
            f"Frozen artifact size drifted: {path}; expected={size_bytes}, observed={observed['size_bytes']}"
        )
    return observed


def _source_input_identities(root: Path) -> dict[str, Any]:
    source = {
        key: _expected_identity(
            Path(spec["path"]), str(spec["sha256"]), int(spec["size_bytes"])
        )
        for key, spec in EXPECTED_SOURCE_INPUTS.items()
    }
    snapshot = {
        "manifest": _artifact(manifest_path(root)),
        "splits": _artifact(split_dir(root) / "splits.parquet"),
        "integrity": _artifact(split_dir(root) / ".integrity_hash"),
        "summary": _artifact(split_dir(root) / "summary.json"),
    }
    for key in source:
        if snapshot[key]["sha256"] != source[key]["sha256"] or snapshot[key]["size_bytes"] != source[key]["size_bytes"]:
            raise ContractError(f"{key}: campaign snapshot is not byte-identical to sealed source")
    census = _validate_source_manifest_and_splits(root)
    return {
        "source_manifest": source["manifest"],
        "source_splits": source["splits"],
        "source_integrity": source["integrity"],
        "source_summary": source["summary"],
        "snapshot_manifest": snapshot["manifest"],
        "snapshot_splits": snapshot["splits"],
        "snapshot_integrity": snapshot["integrity"],
        "snapshot_summary": snapshot["summary"],
        "census": census,
    }


def _validate_source_manifest_and_splits(root: Path) -> dict[str, Any]:
    manifest = pd.read_csv(manifest_path(root), low_memory=False)
    required = {
        "slide_id", "patient_id", "target_label", "specimen_role", "cohort",
        "subcohort", "k_fold", *(f"val_fold_{fold}" for fold in FOLDS),
    }
    if required - set(manifest.columns):
        raise ContractError(f"Snapshot manifest lacks columns {sorted(required - set(manifest.columns))}")
    if manifest["slide_id"].astype(str).duplicated().any():
        raise ContractError("Snapshot manifest has duplicate slide_id")
    role = manifest["specimen_role"].astype(str).str.strip().str.casefold()
    cohort = manifest["cohort"].astype(str).str.strip().str.casefold()
    if not role.eq("primary").all() or not cohort.isin({"tcga", "surgen"}).all():
        raise ContractError("Snapshot contains rows outside TCGA+SurGen primary")
    labels = pd.to_numeric(manifest["target_label"], errors="raise").astype(int)
    if not labels.isin([0, 1]).all():
        raise ContractError("Snapshot labels are not binary")
    patients = pd.DataFrame(
        {"patient_id": manifest["patient_id"].astype(str), "label": labels}
    )
    if (patients.groupby("patient_id")["label"].nunique() != 1).any():
        raise ContractError("A patient has inconsistent target labels")
    patient_labels = patients.groupby("patient_id")["label"].first()
    folds = pd.to_numeric(manifest["k_fold"], errors="raise").astype(int)
    if set(folds) != set(FOLDS):
        raise ContractError("Snapshot fold roster is not exactly 0..4")
    patient_fold = pd.DataFrame(
        {"patient_id": manifest["patient_id"].astype(str), "fold": folds}
    ).groupby("patient_id")["fold"]
    if (patient_fold.nunique() != 1).any():
        raise ContractError("A patient spans OOF test folds")
    test_counts = [
        int(manifest.loc[folds.eq(fold), "patient_id"].astype(str).nunique())
        for fold in FOLDS
    ]
    observed = {
        "slides": int(len(manifest)),
        "patients": int(patient_labels.size),
        "mutant_patients": int(patient_labels.eq(1).sum()),
        "wildtype_patients": int(patient_labels.eq(0).sum()),
        "test_patients_by_fold": test_counts,
    }
    if observed != EXPECTED_CENSUS:
        raise ContractError(f"TCGA+SurGen census drifted: expected={EXPECTED_CENSUS}, observed={observed}")
    split = pd.read_parquet(split_dir(root) / "splits.parquet")
    if (
        len(split) != len(manifest)
        or split["slide_id"].astype(str).duplicated().any()
        or set(split["slide_id"].astype(str)) != set(manifest["slide_id"].astype(str))
    ):
        raise ContractError("Snapshot split roster differs from manifest")
    columns = ["slide_id", "fold", *(f"val_fold_{fold}" for fold in FOLDS)]
    joined = manifest[["slide_id", "k_fold", *(f"val_fold_{fold}" for fold in FOLDS)]].merge(
        split[columns], on="slide_id", validate="one_to_one", suffixes=("_manifest", "_split")
    )
    if not pd.to_numeric(joined["k_fold"]).astype(int).eq(pd.to_numeric(joined["fold"]).astype(int)).all():
        raise ContractError("Manifest k_fold differs from inherited splits")
    for fold in FOLDS:
        column = f"val_fold_{fold}"
        if not pd.to_numeric(joined[f"{column}_manifest"]).astype(int).eq(
            pd.to_numeric(joined[f"{column}_split"]).astype(int)
        ).all():
            raise ContractError(f"Manifest {column} differs from inherited splits")
    return observed


def _pack_identity(name: str, root: Path, *, deep: bool) -> dict[str, Any]:
    spec = PACKS[name]
    directory = Path(spec["path"])
    if not directory.is_dir() or directory.is_symlink():
        raise ContractError(f"Packed feature store missing or symlinked: {directory}")
    artifacts = {}
    for filename, (expected_sha, expected_size) in spec["artifacts"].items():
        path = directory / filename
        if not path.is_file() or path.is_symlink() or path.stat().st_size != expected_size:
            raise ContractError(f"{name} packed artifact missing/symlinked/size-drifted: {path}")
        if deep:
            artifacts[filename] = _expected_identity(path, expected_sha, expected_size)
        else:
            artifacts[filename] = {
                "path": str(path.resolve()),
                "sha256": expected_sha,
                "size_bytes": expected_size,
            }
    index = pd.read_parquet(directory / "index.parquet")
    id_column = "slide_id" if "slide_id" in index.columns else "key"
    if id_column not in index.columns or index[id_column].astype(str).duplicated().any():
        raise ContractError(f"{name} pack index has no unique slide identifier")
    wanted = set(pd.read_csv(manifest_path(root), usecols=["slide_id"])["slide_id"].astype(str))
    indexed = set(index[id_column].astype(str))
    missing = sorted(wanted - indexed)
    if missing:
        raise ContractError(f"{name} pack misses {len(missing)} requested slides; first={missing[:5]}")
    meta = _read_json(directory / "meta.json")
    if int(meta.get("feat_dim", -1)) != int(spec["feature_dim"]):
        raise ContractError(f"{name} pack feature dimension drifted")
    return {
        "path": str(directory.resolve()),
        "encoder": spec["encoder"],
        "feature_dim": spec["feature_dim"],
        "patch_profile": spec["patch_profile"],
        "artifacts": artifacts,
        "coverage": {"requested_slides": len(wanted), "missing_slides": 0},
    }


def _strict_scan_json_tree(path: Path) -> None:
    for item in sorted(path.rglob("*.json")):
        _read_json(item)


def _adopted_chain(seed: int, *, deep: bool) -> dict[str, Any]:
    expected_receipt_sha = ADOPTED_JOB_RECEIPT_SHA256[seed]
    receipt_identity = _expected_identity(adopted_job_receipt(seed), expected_receipt_sha)
    if deep:
        receipt = adopted_campaign._validate_job(ADOPTED_ROOT, "tcga_surgen_primary", seed)
        _strict_scan_json_tree(adopted_run_dir(seed))
    else:
        receipt = _read_json(adopted_job_receipt(seed))
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or receipt.get("fit_count") != 5 or receipt.get("refit_count") != 0:
        raise ContractError(f"Adopted UNI-v1 seed{seed} receipt does not prove five OOF fits and zero refits")
    expected_paths = {
        "training_identity": adopted_run_dir(seed) / "training_identity.json",
        "training_completion": adopted_run_dir(seed) / "training_completion.json",
        "oof_predictions": adopted_run_dir(seed) / "oof_predictions.parquet",
        "cv_summary": adopted_run_dir(seed) / "cv_summary.json",
    }
    return {
        "seed": seed,
        "source_job_receipt": receipt_identity,
        **{name: _artifact(path) for name, path in expected_paths.items()},
        "training_fingerprint": artifacts.get("training_fingerprint"),
        "oof_fits": 5,
        "refits": 0,
    }


def _implementation_sources() -> list[dict[str, Any]]:
    explicit = {
        Path(__file__).resolve(),
        REPO / "tests/test_aim1_tcga_surgen_two_encoder_campaign.py",
        REPO / "tools/study_train.py",
        REPO / "tools/aim1_source_cohort_five_seed_campaign.py",
        REPO / "src/oceanpath/workflows/training.py",
        REPO / "src/oceanpath/workflows/finalize.py",
        REPO / "pyproject.toml",
        REPO / "uv.lock",
    }
    sources = explicit | set((REPO / "src/oceanpath").rglob("*.py")) | set(
        (REPO / "configs").rglob("*.yaml")
    )
    return [_artifact(path) for path in sorted(sources, key=lambda path: str(path.resolve()))]


def _iter_artifacts(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
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


def _contract_semantics(root: Path) -> dict[str, Any]:
    dry = build_contract(root, created_utc="<ignored>")
    dry.pop("created_utc")
    return dry


def validate_contract(root: Path, *, deep: bool) -> dict[str, Any]:
    root = assert_safe_output_root(root)
    _reject_symlinks_below(root)
    contract = _read_json(contract_path(root))
    semantics = _contract_semantics(root)
    mismatch = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in semantics.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ContractError(f"Campaign contract semantics drifted: {mismatch}")
    # Large adopted/pack bytes have dedicated validators below.  Avoid hashing
    # 50--81 GB files twice in one stage while still checking their sizes here.
    dedicated_roots = (ADOPTED_ROOT.resolve(), UNI_PACK.resolve(), V2_PACK.resolve())
    for identity in _iter_artifacts(contract):
        identity_path = Path(str(identity["path"])).resolve()
        dedicated = any(
            identity_path == prefix or _is_relative_to(identity_path, prefix)
            for prefix in dedicated_roots
        )
        _validate_artifact(identity, deep=deep and not dedicated)
    if contract.get("source_input") != _source_input_identities(root):
        raise ContractError("Source input snapshot/census drifted")
    if _sha256(REPO / "tools/aim1_source_cohort_five_seed_campaign.py") != ADOPTED_CONTROLLER_SHA256:
        raise ContractError("Adopted controller source SHA drifted")
    _expected_identity(ADOPTED_ROOT / "contract.json", ADOPTED_CONTRACT_SHA256)
    _expected_identity(ADOPTED_ROOT / "receipts/training_complete.json", ADOPTED_TERMINAL_SHA256)
    if deep:
        adopted_campaign.validate_contract(ADOPTED_ROOT, deep=True)
        adopted = [_adopted_chain(seed, deep=True) for seed in SEEDS]
        if contract.get("adopted_univ1_oof_chains") != adopted:
            raise ContractError("Adopted UNI-v1 OOF chain evidence drifted")
        # adopted_campaign already hashes the exact pinned UNI pack.  V2 is
        # new to this campaign and receives its own full byte hash here.
        for name in PACKS:
            observed_pack = _pack_identity(name, root, deep=name == "virchow2_cls")
            if contract.get("feature_stores", {}).get(name) != observed_pack:
                raise ContractError(f"{name} feature-store evidence drifted")
    return contract


def cmd_plan(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    payload = build_contract(root)
    print(json.dumps({"status": "PLAN_ONLY_NO_WRITES", **payload}, indent=2, allow_nan=False))


def cmd_prepare(args: argparse.Namespace) -> None:
    if not args.apply:
        raise ContractError("prepare requires --apply")
    root = assert_safe_output_root(args.output_root)
    if contract_path(root).exists():
        validate_contract(root, deep=True)
        print(f"PASS cached immutable contract: {contract_path(root)}")
        return
    if root.exists() and any(root.iterdir()):
        raise ContractError("Fresh campaign root required before first prepare")
    _expected_identity(REPO / "tools/aim1_source_cohort_five_seed_campaign.py", ADOPTED_CONTROLLER_SHA256)
    _expected_identity(ADOPTED_ROOT / "contract.json", ADOPTED_CONTRACT_SHA256)
    _expected_identity(ADOPTED_ROOT / "receipts/training_complete.json", ADOPTED_TERMINAL_SHA256)
    adopted_campaign.validate_contract(ADOPTED_ROOT, deep=True)
    for key, spec in EXPECTED_SOURCE_INPUTS.items():
        _expected_identity(Path(spec["path"]), str(spec["sha256"]), int(spec["size_bytes"]))
        destination = manifest_path(root) if key == "manifest" else split_dir(root) / Path(spec["path"]).name
        _copy_once(Path(spec["path"]), destination)
    source_input = _source_input_identities(root)
    adopted = [_adopted_chain(seed, deep=True) for seed in SEEDS]
    feature_stores = {
        "univ1": _pack_identity("univ1", root, deep=False),
        "virchow2_cls": _pack_identity("virchow2_cls", root, deep=True),
    }
    persisted = {
        **build_contract(root),
        "source_input": source_input,
        "adopted_campaign": {
            "contract": _artifact(ADOPTED_ROOT / "contract.json"),
            "training_complete": _artifact(ADOPTED_ROOT / "receipts/training_complete.json"),
            "controller": _artifact(REPO / "tools/aim1_source_cohort_five_seed_campaign.py"),
        },
        "adopted_univ1_oof_chains": adopted,
        "feature_stores": feature_stores,
        "implementation_sources": _implementation_sources(),
    }
    _write_json_once(contract_path(root), persisted)
    # All expensive inputs were deeply authenticated immediately above.  This
    # post-write pass proves serialization/semantics without redundant 100+GB
    # hashing; preflight repeats one independent deep pass before training.
    validate_contract(root, deep=False)
    print(f"PASS sealed contract: 25 adopted + 35 new = 60 operational fits at {root}")


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    if not current.exists():
        raise ContractError(f"No existing parent for output path: {path}")
    return current


def _available_ram_gib() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", maxsplit=1)
        values[key] = int(value.strip().split()[0])
    return float(values.get("MemAvailable", 0)) / 1024**2


def _gpu_resources() -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode or not query.stdout.strip():
        raise ContractError(f"GPU query failed: {query.stderr.strip()}")
    values = [part.strip() for part in query.stdout.strip().splitlines()[0].split(",")]
    applications = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if applications.returncode:
        raise ContractError(f"GPU application query failed: {applications.stderr.strip()}")
    active = [line.strip() for line in applications.stdout.splitlines() if line.strip()]
    return {
        "gpu_name": values[0],
        "total_gpu_mib": int(values[1]),
        "free_gpu_mib": int(values[2]),
        "gpu_utilization_percent": int(values[3]),
        "compute_application_count": len(active),
        "compute_applications": active,
    }


def _resource_preflight(root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(_nearest_existing_parent(root))
    observed = {
        "logical_cpus": int(os.cpu_count() or 0),
        "available_ram_gib": _available_ram_gib(),
        "free_disk_gib": float(disk.free) / 1024**3,
        **_gpu_resources(),
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
    if observed["compute_application_count"] != 0:
        failed["compute_application_count"] = {"required": 0, "observed": observed["compute_application_count"], "applications": observed["compute_applications"]}
    if observed["gpu_utilization_percent"] > 25:
        failed["gpu_utilization_percent"] = {"maximum": 25, "observed": observed["gpu_utilization_percent"]}
    if failed:
        raise ContractError(f"Six-worker resource preflight failed: {failed}")
    return observed


def _validate_preflight(root: Path) -> dict[str, Any]:
    receipt = _read_json(preflight_path(root))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "deep_preflight_passed",
        "contract": _artifact(contract_path(root)),
        "job_count": TOTAL_JOBS,
        "new_fits": NEW_FITS,
        "scheduler_ceiling": MAX_WORKERS,
    }
    mismatch = {key: {"expected": value, "observed": receipt.get(key)} for key, value in expected.items() if receipt.get(key) != value}
    if mismatch:
        raise ContractError(f"Deep-preflight receipt drifted: {mismatch}")
    return receipt


def cmd_preflight(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    validate_contract(root, deep=True)
    if preflight_path(root).exists():
        _validate_preflight(root)
        print(f"PASS cached deep preflight: {preflight_path(root)}")
        return
    resources = _resource_preflight(root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "deep_preflight_passed",
        "created_utc": _utcnow(),
        "contract": _artifact(contract_path(root)),
        "job_count": TOTAL_JOBS,
        "new_fits": NEW_FITS,
        "scheduler_ceiling": MAX_WORKERS,
        "resources": resources,
    }
    if args.apply:
        _write_json_once(preflight_path(root), payload)
        _validate_preflight(root)
    print(json.dumps({"status": "PASS", "persisted": bool(args.apply), "resources": resources}, indent=2))


def _nested(record: Mapping[str, Any], dotted: str) -> Any:
    value: Any = record
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return "<missing>"
        value = value[key]
    return value


def _material_expectations(root: Path, seed: int, *, encoder: str, skip_finalize: bool) -> dict[str, Any]:
    expectations = {
        "data.name": "aim1_primary_tcga_surgen",
        "data.aim1_model": "primary_tcga_surgen",
        "data.csv_path": str(manifest_path(root).resolve()),
        "data.label_columns": ["target_label"],
        "data.patient_id_column": "patient_id",
        "data.filename_column": "slide_id",
        "data.cohort_column": "cohort",
        "data.num_classes": 2,
        "splits.scheme": "predefined_oof_kfold",
        "splits.name": "aim1_balanced5",
        "splits.fold_column": "k_fold",
        "splits.group_column": "patient_id",
        "splits.allow_group_overlap": False,
        "splits.n_folds": 5,
        "splits.seed": seed,
        "model.name": "abmil",
        "model.arch": "abmil",
        "model.embed_dim": 512,
        "model.attn_dim": 384,
        "model.gate": True,
        "model.dropout": 0.25,
        "model.input_dropout": 0.10,
        "training.lr": 1e-4,
        "training.weight_decay": 1e-5,
        "training.max_epochs": 20,
        "training.early_stopping_patience": 5,
        "training.min_epoch_before_stop": 10,
        "training.monitor_metric": "val/patient_auroc",
        "training.monitor_mode": "max",
        "training.loss_type": "bce",
        "training.class_weights": None,
        "training.training_class_weighted_loss": False,
        "training.validation_loss_weighted": False,
        "training.class_weighted_sampling": False,
        "training.dataset_max_instances": CAP,
        "training.eval_full_bags": True,
        "training.sample_weight_column": None,
        "training.train_sampling_strategy": "patient_natural",
        "training.seed": seed,
        "training.skip_finalize": skip_finalize,
        "training.refit_epoch_rule": "p75",
        "platform.precision": "bf16-mixed",
        "platform.devices": 1,
    }
    if encoder == "virchow2_cls":
        expectations.update(
            {
                "encoder.name": "virchow2",
                "encoder.feature_dim": 1_280,
                "encoder.patch_size": 224,
                "extraction.patch_size": 224,
                "extraction.coords_dir": "20x_224px_0px_overlap_mpp0.5",
                "extraction.coords_subdir": "20x_224px_0px_overlap_mpp0.5",
                "training.packed_dir": str(V2_PACK),
                "training.verify_packed_source": False,
                "training.final_strategies": ["best_fold", "refit"],
            }
        )
    else:
        expectations.update(
            {
                "encoder.name": "uni_v1",
                "encoder.feature_dim": 1_024,
                "encoder.patch_size": 256,
                "extraction.patch_size": 256,
                "extraction.coords_dir": "20x_256px_0px_overlap_mpp0.5",
                "extraction.coords_subdir": "20x_256px_0px_overlap_mpp0.5",
                "training.packed_dir": str(UNI_PACK),
                "training.verify_packed_source": False,
            }
        )
    return expectations


def _validate_material_config(config: Mapping[str, Any], root: Path, seed: int, *, encoder: str, skip_finalize: bool, context: str) -> None:
    mismatch = {
        dotted: {"expected": expected, "observed": _nested(config, dotted)}
        for dotted, expected in _material_expectations(root, seed, encoder=encoder, skip_finalize=skip_finalize).items()
        if _nested(config, dotted) != expected
    }
    if mismatch:
        raise ContractError(f"{encoder}/seed{seed}: {context} material recipe mismatch: {mismatch}")


def _validate_oof(directory: Path, root: Path, seed: int) -> dict[str, Any]:
    import yaml

    from oceanpath.workflows.training import validate_training_run_dir

    if not directory.is_dir() or directory.is_symlink():
        raise ContractError(f"Missing/symlinked V2 run directory: {directory}")
    completion = validate_training_run_dir(directory, require_test_predictions=True)
    folds = completion.get("fold_completions")
    if (
        completion.get("status") != "completed"
        or int(completion.get("n_folds", -1)) != 5
        or completion.get("skip_finalize") is not False
        or [item.get("path") for item in folds] != [f"fold_{fold}/completion.json" for fold in FOLDS]
    ):
        raise ContractError(f"Virchow2-CLS/seed{seed}: native completion is not exact 5-fold plus finalization")
    identity = _read_json(directory / "training_identity.json")
    if identity.get("fingerprint") != completion.get("training_fingerprint"):
        raise ContractError(f"Virchow2-CLS/seed{seed}: training fingerprint mismatch")
    material = ((identity.get("payload") or {}).get("material_config"))
    if not isinstance(material, Mapping):
        raise ContractError(f"Virchow2-CLS/seed{seed}: missing material identity")
    _validate_material_config(material, root, seed, encoder="virchow2_cls", skip_finalize=False, context="training identity")
    evidence = (identity.get("payload") or {}).get("input_evidence") or {}
    if evidence.get("manifest_sha256") != _sha256(manifest_path(root)) or evidence.get("split_integrity_sha256") != _sha256(split_dir(root) / ".integrity_hash"):
        raise ContractError(f"Virchow2-CLS/seed{seed}: input evidence mismatch")
    manifest = pd.read_csv(manifest_path(root), low_memory=False)
    for fold in FOLDS:
        config_path = directory / f"fold_{fold}/config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise ContractError(f"Malformed fold config: {config_path}")
        _validate_material_config(config, root, seed, encoder="virchow2_cls", skip_finalize=False, context=f"fold{fold}")
        if Path(str(_nested(config, "platform.splits_root"))).resolve() != (root / "inputs/splits").resolve():
            raise ContractError(f"Virchow2-CLS/seed{seed}/fold{fold}: wrong split root")
        fold_column = pd.to_numeric(manifest["k_fold"]).astype(int)
        for role, mask in {
            "test": fold_column.eq(fold),
            "val": pd.to_numeric(manifest[f"val_fold_{fold}"]).astype(int).eq(1),
        }.items():
            predictions = pd.read_parquet(directory / f"fold_{fold}/preds_{role}.parquet")
            expected = manifest.loc[mask, ["slide_id", "target_label"]]
            joined = predictions.merge(expected, on="slide_id", validate="one_to_one")
            logits = pd.to_numeric(predictions["logit"], errors="coerce")
            if (
                len(predictions) != len(expected)
                or len(joined) != len(expected)
                or set(predictions["slide_id"].astype(str)) != set(expected["slide_id"].astype(str))
                or not pd.to_numeric(joined["label"]).astype(int).eq(pd.to_numeric(joined["target_label"]).astype(int)).all()
                or not np.isfinite(logits).all()
            ):
                raise ContractError(f"Virchow2-CLS/seed{seed}/fold{fold}: invalid {role} roster/labels/logits")
    oof = pd.read_parquet(directory / "oof_predictions.parquet")
    check = oof.merge(
        manifest[["slide_id", "target_label", "k_fold"]],
        on="slide_id", validate="one_to_one",
    )
    if (
        len(oof) != len(manifest)
        or oof["slide_id"].astype(str).duplicated().any()
        or set(oof["slide_id"].astype(str)) != set(manifest["slide_id"].astype(str))
        or set(pd.to_numeric(oof["fold"]).astype(int)) != set(FOLDS)
        or not np.isfinite(pd.to_numeric(oof["logit"], errors="coerce")).all()
        or not pd.to_numeric(check["label"]).astype(int).eq(pd.to_numeric(check["target_label"]).astype(int)).all()
        or not pd.to_numeric(check["fold"]).astype(int).eq(pd.to_numeric(check["k_fold"]).astype(int)).all()
    ):
        raise ContractError(f"Virchow2-CLS/seed{seed}: invalid complete OOF roster")
    return {
        "training_completion": _artifact(directory / "training_completion.json"),
        "training_identity": _artifact(directory / "training_identity.json"),
        "oof_predictions": _artifact(directory / "oof_predictions.parquet"),
        "cv_summary": _artifact(directory / "cv_summary.json"),
        "training_fingerprint": completion.get("training_fingerprint"),
        "oof_rows": int(len(oof)),
    }


def _validate_refit(directory: Path, root: Path, seed: int, *, encoder: str) -> dict[str, Any]:
    info_path = directory / "final/refit/info.json"
    checkpoint = directory / "final/refit/model.ckpt"
    info = _read_json(info_path)
    expected_epochs = []
    source = adopted_run_dir(seed) if encoder == "univ1" else directory
    for fold in FOLDS:
        metrics = _read_json(source / f"fold_{fold}/fold_metrics.json")
        epoch = metrics.get("best_epoch")
        if not isinstance(epoch, (int, float)) or not math.isfinite(float(epoch)) or float(epoch) <= 0:
            raise ContractError(f"{encoder}/seed{seed}/fold{fold}: invalid best_epoch")
        expected_epochs.append(epoch)
    expected_refit_epochs = max(1, int(np.ceil(np.percentile(expected_epochs, 75))))
    expected = {
        "strategy": "refit",
        "refit_epochs": expected_refit_epochs,
        "refit_epoch_rule": "p75",
        "refit_max_steps": None,
        "seed": seed,
        "sampling_seed": seed,
        "train_sampling_strategy": "patient_natural",
        "sample_weight_column": None,
        "class_weights": None,
        "dataset_max_instances": CAP,
        "eval_full_bags": True,
        "fold_best_epochs": expected_epochs,
        "n_train_slides": EXPECTED_CENSUS["slides"],
    }
    mismatch = {key: {"expected": value, "observed": info.get(key)} for key, value in expected.items() if info.get(key) != value}
    if mismatch:
        raise ContractError(f"{encoder}/seed{seed}: p75 refit evidence mismatch: {mismatch}")
    if not checkpoint.is_file() or checkpoint.is_symlink() or checkpoint.stat().st_size <= 0:
        raise ContractError(f"{encoder}/seed{seed}: missing/symlinked refit checkpoint")
    if Path(str(info.get("model_path", ""))).resolve() != checkpoint.resolve():
        raise ContractError(f"{encoder}/seed{seed}: refit model_path mismatch")
    return {"info": _artifact(info_path), "checkpoint": _artifact(checkpoint), "refit_epochs": expected_refit_epochs}


def _native_validation(root: Path, kind: str, seed: int) -> dict[str, Any]:
    directory = run_dir(root, kind, seed)
    _reject_symlinks_below(directory)
    if kind == "univ1_refit":
        forbidden = [*directory.glob("fold_*"), *directory.glob("*oof*"), directory / "training_completion.json"]
        if any(path.exists() for path in forbidden):
            raise ContractError(f"UNI-v1/seed{seed}: hidden OOF/training output exists in refit-only job")
        config = _read_json(directory / "refit_provenance.json")
        if config.get("adopted_chain") != _adopted_chain(seed, deep=False):
            raise ContractError(f"UNI-v1/seed{seed}: adopted provenance drifted")
        refit = _validate_refit(directory, root, seed, encoder="univ1")
        return {
            "refit": refit,
            "refit_provenance": _artifact(directory / "refit_provenance.json"),
            "source_config": _artifact(directory / "config.yaml"),
            "oof_fits": 0,
            "refits": 1,
            "total_fits": 1,
        }
    oof = _validate_oof(directory, root, seed)
    refit = _validate_refit(directory, root, seed, encoder="virchow2_cls")
    final_dir = directory / "final"
    allowed = {"best_fold", "refit", "finalize_summary.json"}
    observed = {path.name for path in final_dir.iterdir()}
    if observed - allowed or not {"best_fold", "refit"}.issubset(observed):
        raise ContractError(f"Virchow2-CLS/seed{seed}: unexpected final strategies: {sorted(observed)}")
    if list(directory.rglob("*ensemble*")):
        raise ContractError(f"Virchow2-CLS/seed{seed}: forbidden ensemble artifact")
    return {
        **oof,
        "refit": refit,
        "finalize_summary": _artifact(final_dir / "finalize_summary.json"),
        "best_fold_checkpoint": _artifact(final_dir / "best_fold/model.ckpt"),
        "oof_fits": 5,
        "refits": 1,
        "total_fits": 6,
    }


def _validate_job(root: Path, kind: str, seed: int) -> dict[str, Any]:
    receipt = _read_json(job_receipt_path(root, kind, seed))
    spec = next(job for job in _job_specs() if job.kind == kind and job.seed == seed)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "completed",
        "kind": kind,
        "encoder": spec.encoder,
        "seed": seed,
        "oof_fits": spec.oof_fits,
        "refits": spec.refits,
        "total_fits": spec.total_fits,
        "attempt": 1,
        "contract": _artifact(contract_path(root)),
        "artifacts": _native_validation(root, kind, seed),
    }
    mismatch = {key: {"expected": value, "observed": receipt.get(key)} for key, value in expected.items() if receipt.get(key) != value}
    if mismatch:
        raise ContractError(f"{kind}/seed{seed}: job receipt mismatch: {mismatch}")
    return receipt


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
        os.write(descriptor, (_json_text({"pid": os.getpid(), "context": context, "utc": _utcnow()})).encode())
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _job_lock_path(root: Path, kind: str, seed: int) -> Path:
    token = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    return Path(f"/tmp/oceanpath_{CAMPAIGN}_{token}_{kind}_s{seed}.lock")


def _run_univ1_refit(root: Path, seed: int, directory: Path) -> None:
    from omegaconf import OmegaConf

    from oceanpath.workflows.finalize import _run_refit

    adopted = _adopted_chain(seed, deep=True)
    source_config = adopted_run_dir(seed) / "fold_0/config.yaml"
    _write_text_once(directory / "config.yaml", source_config.read_text(encoding="utf-8"))
    _write_json_once(
        directory / "refit_provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "campaign": CAMPAIGN,
            "kind": "univ1_refit",
            "encoder": "UNI-v1",
            "seed": seed,
            "adopted_chain": adopted,
            "source_config": _artifact(source_config),
            "fold_metrics": [_artifact(adopted_run_dir(seed) / f"fold_{fold}/fold_metrics.json") for fold in FOLDS],
            "refit_entrypoint": "oceanpath.workflows.finalize._run_refit",
        },
    )
    cfg = OmegaConf.load(source_config)
    material = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(material, Mapping):
        raise ContractError("Adopted resolved config is malformed")
    # The adopted config deliberately says skip_finalize=true because no old
    # finalization ran.  Its material fold recipe must otherwise match exactly.
    old_root = ADOPTED_ROOT
    _validate_material_config(material, old_root, seed, encoder="univ1", skip_finalize=True, context="adopted refit source")
    metrics = [_read_json(adopted_run_dir(seed) / f"fold_{fold}/fold_metrics.json") for fold in FOLDS]
    _run_refit(cfg, directory / "final", metrics)


def _run_v2(root: Path, seed: int, directory: Path) -> int:
    command = [
        sys.executable,
        str(REPO / "tools/study_train.py"),
        "hydra-train",
        *_v2_overrides(root, seed, directory),
    ]
    # ``redirect_stdout`` changes Python's stream object, not file descriptor
    # 1 inherited by a child.  Pass the governed log stream explicitly so the
    # complete native training transcript is immutable and receipted.
    return int(
        subprocess.run(
            command,
            cwd=REPO,
            check=False,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode
    )


def cmd_internal_train_one(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    if args.kind not in {job.kind for job in _job_specs()} or args.seed not in SEEDS:
        raise ContractError("Uncontracted job kind/seed")
    validate_contract(root, deep=False)
    _validate_preflight(root)
    spec = next(job for job in _job_specs() if job.kind == args.kind and job.seed == args.seed)
    with _kernel_lock(_job_lock_path(root, spec.kind, spec.seed), context=f"{spec.kind}/seed{spec.seed}"):
        paths = [
            run_dir(root, spec.kind, spec.seed),
            request_path(root, spec.kind, spec.seed),
            log_path(root, spec.kind, spec.seed),
            failure_path(root, spec.kind, spec.seed),
            job_receipt_path(root, spec.kind, spec.seed),
        ]
        if any(path.exists() or path.is_symlink() for path in paths):
            raise ContractError(
                f"Refusing retry/overwrite of {spec.kind}/seed{spec.seed}; use a fresh campaign root"
            )
        _write_json_once(
            request_path(root, spec.kind, spec.seed),
            {
                "schema_version": SCHEMA_VERSION,
                "campaign": CAMPAIGN,
                "status": "requested",
                "created_utc": _utcnow(),
                "kind": spec.kind,
                "encoder": spec.encoder,
                "seed": spec.seed,
                "oof_fits": spec.oof_fits,
                "refits": spec.refits,
                "total_fits": spec.total_fits,
                "attempt": 1,
                "contract": _artifact(contract_path(root)),
                "manifest": _artifact(manifest_path(root)),
                "split": _artifact(split_dir(root) / "splits.parquet"),
                "output": str(run_dir(root, spec.kind, spec.seed)),
            },
        )
        log = log_path(root, spec.kind, spec.seed)
        log.parent.mkdir(parents=True, exist_ok=True)
        returncode = 0
        error: BaseException | None = None
        with log.open("x", encoding="utf-8", buffering=1) as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            try:
                if spec.kind == "univ1_refit":
                    _run_univ1_refit(root, spec.seed, run_dir(root, spec.kind, spec.seed))
                else:
                    returncode = _run_v2(root, spec.seed, run_dir(root, spec.kind, spec.seed))
                    if returncode:
                        raise RuntimeError(f"study_train exited {returncode}")
            except BaseException as exc:  # preserve immutable evidence, including interrupts
                error = exc
                if returncode == 0:
                    returncode = 130 if isinstance(exc, KeyboardInterrupt) else 1
        if error is not None:
            _write_json_once(
                failure_path(root, spec.kind, spec.seed),
                {
                    "schema_version": SCHEMA_VERSION,
                    "campaign": CAMPAIGN,
                    "status": "failed_no_retry",
                    "finished_utc": _utcnow(),
                    "kind": spec.kind,
                    "seed": spec.seed,
                    "attempt": 1,
                    "returncode": returncode,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "request": _artifact(request_path(root, spec.kind, spec.seed)),
                    "log": _artifact(log),
                },
            )
            raise SystemExit(returncode)
        native = _native_validation(root, spec.kind, spec.seed)
        _write_json_once(
            job_receipt_path(root, spec.kind, spec.seed),
            {
                "schema_version": SCHEMA_VERSION,
                "campaign": CAMPAIGN,
                "status": "completed",
                "finished_utc": _utcnow(),
                "kind": spec.kind,
                "encoder": spec.encoder,
                "seed": spec.seed,
                "oof_fits": spec.oof_fits,
                "refits": spec.refits,
                "total_fits": spec.total_fits,
                "attempt": 1,
                "contract": _artifact(contract_path(root)),
                "request": _artifact(request_path(root, spec.kind, spec.seed)),
                "log": _artifact(log),
                "artifacts": native,
            },
        )
        _validate_job(root, spec.kind, spec.seed)
    print(f"PASS completed {spec.kind}/seed{spec.seed}: {spec.total_fits} new fit(s)")


def _execute_job(job: Mapping[str, Any]) -> dict[str, Any]:
    started_utc = _utcnow()
    started = time.monotonic()
    process = subprocess.Popen(list(job["command"]), cwd=REPO)
    returncode = int(process.wait())
    return {
        "job_id": str(job["job_id"]),
        "kind": str(job["kind"]),
        "seed": int(job["seed"]),
        "pid": int(process.pid),
        "attempt": 1,
        "returncode": returncode,
        "started_utc": started_utc,
        "finished_utc": _utcnow(),
        "started_monotonic": started,
        "finished_monotonic": time.monotonic(),
    }


def _peak_parallel(events: Sequence[Mapping[str, Any]]) -> int:
    points: list[tuple[float, int]] = []
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
    with _kernel_lock(launcher, context="Aim-1 two-encoder launcher"), _kernel_lock(GLOBAL_GPU_LOCK, context="study-wide GPU-0 training"):
        yield


def _assert_fresh_training_namespace(root: Path) -> None:
    unexpected = []
    for job in _job_specs():
        for path in (
            run_dir(root, job.kind, job.seed), request_path(root, job.kind, job.seed),
            log_path(root, job.kind, job.seed), failure_path(root, job.kind, job.seed),
            job_receipt_path(root, job.kind, job.seed),
        ):
            if path.exists() or path.is_symlink():
                unexpected.append(str(path))
    if scheduler_path(root).exists() or scheduler_path(root).is_symlink():
        unexpected.append(str(scheduler_path(root)))
    if training_receipt_path(root).exists() or training_receipt_path(root).is_symlink():
        unexpected.append(str(training_receipt_path(root)))
    if unexpected:
        raise ContractError(f"Exactly-once launch requires a fresh training namespace: {unexpected[:10]}")


def cmd_train(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    validate_contract(root, deep=False)
    jobs = build_training_jobs(root)
    if not args.apply:
        print(json.dumps({"status": "DRY_RUN_NO_WRITES", "max_workers": args.max_workers, "jobs": jobs}, indent=2))
        return
    _validate_preflight(root)
    if args.max_workers != MAX_WORKERS:
        raise ContractError(f"Governed campaign requires exactly --max-workers {MAX_WORKERS}")
    _assert_fresh_training_namespace(root)
    events: list[dict[str, Any]] = []
    failures: list[tuple[str, int]] = []
    with _launcher_locks(root), concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_execute_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            event = future.result()
            events.append(event)
            print(f"{event['job_id']}: returncode={event['returncode']}", flush=True)
            if event["returncode"]:
                failures.append((event["job_id"], event["returncode"]))
    if failures:
        raise SystemExit(f"Training failures; zero retries; immutable evidence retained: {failures}")
    peak = _peak_parallel(events)
    if peak != MAX_WORKERS:
        raise ContractError(f"Scheduler did not demonstrate exact six-way execution: observed peak={peak}")
    _write_json_once(
        scheduler_path(root),
        {
            "schema_version": SCHEMA_VERSION,
            "campaign": CAMPAIGN,
            "status": "completed_rc0",
            "created_utc": _utcnow(),
            "configured_max_workers": MAX_WORKERS,
            "observed_peak_parallel_workers": peak,
            "job_count": TOTAL_JOBS,
            "new_oof_fits": NEW_OOF_FITS,
            "new_refits": NEW_REFITS,
            "new_fits": NEW_FITS,
            "retry_count": 0,
            "events": sorted(events, key=lambda item: str(item["job_id"])),
        },
    )


def _validate_scheduler(root: Path) -> dict[str, Any]:
    scheduler = _read_json(scheduler_path(root))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "completed_rc0",
        "configured_max_workers": MAX_WORKERS,
        "observed_peak_parallel_workers": MAX_WORKERS,
        "job_count": TOTAL_JOBS,
        "new_oof_fits": NEW_OOF_FITS,
        "new_refits": NEW_REFITS,
        "new_fits": NEW_FITS,
        "retry_count": 0,
    }
    mismatch = {key: {"expected": value, "observed": scheduler.get(key)} for key, value in expected.items() if scheduler.get(key) != value}
    events = scheduler.get("events")
    if not isinstance(events, list) or len(events) != TOTAL_JOBS:
        mismatch["events"] = {"expected": TOTAL_JOBS, "observed": len(events) if isinstance(events, list) else type(events).__name__}
    else:
        keys = [(event.get("kind"), event.get("seed")) for event in events]
        expected_keys = [(job.kind, job.seed) for job in _job_specs()]
        if sorted(keys) != sorted(expected_keys) or any(event.get("returncode") != 0 or event.get("attempt") != 1 for event in events):
            mismatch["event_roster"] = {"expected": sorted(expected_keys), "observed": sorted(keys)}
    if mismatch:
        raise ContractError(f"Scheduler receipt mismatch: {mismatch}")
    return scheduler


def _validate_output_roster(root: Path) -> None:
    train = root / "train/e0"
    expected_encoders = {"univ1", "virchow2_cls"}
    if {path.name for path in train.iterdir() if path.is_dir()} != expected_encoders:
        raise ContractError("Unexpected/missing encoder output directory")
    for encoder, kind in (("univ1", "univ1_refit"), ("virchow2_cls", "virchow2_full")):
        directory = train / encoder
        expected = {f"seed{seed}" for seed in SEEDS}
        observed = {path.name for path in directory.iterdir() if path.is_dir()}
        if observed != expected:
            raise ContractError(f"{encoder}: unexpected/missing seed directories: {sorted(observed)}")
        for seed in SEEDS:
            _native_validation(root, kind, seed)
    for kind in ("univ1_refit", "virchow2_full"):
        receipt_dir = root / f"receipts/jobs/{kind}"
        expected = {f"seed{seed}.json" for seed in SEEDS}
        observed = {path.name for path in receipt_dir.iterdir() if path.is_file()}
        if observed != expected:
            raise ContractError(f"{kind}: unexpected/missing job receipts: {sorted(observed)}")


def _terminal_payload(root: Path, new_receipts: list[dict[str, Any]]) -> dict[str, Any]:
    adopted = [_adopted_chain(seed, deep=True) for seed in SEEDS]
    chains = []
    for job, receipt in zip(_job_specs(), new_receipts, strict=True):
        chains.append(
            {
                "kind": job.kind,
                "encoder": job.encoder,
                "seed": job.seed,
                "oof_fits": job.oof_fits,
                "refits": job.refits,
                "total_fits": job.total_fits,
                "job_receipt": receipt,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "complete_and_certified",
        "created_utc": _utcnow(),
        "contract": _artifact(contract_path(root)),
        "preflight": _artifact(preflight_path(root)),
        "scheduler": _artifact(scheduler_path(root)),
        "source_population": {
            "name": "TCGA + SurGen primary",
            **EXPECTED_CENSUS,
            "manifest": _artifact(manifest_path(root)),
            "splits": _artifact(split_dir(root) / "splits.parquet"),
        },
        "seeds": list(SEEDS),
        "encoders": ["UNI-v1", "Virchow2-CLS"],
        "job_count": TOTAL_JOBS,
        "fit_accounting": _fit_accounting(),
        "concurrency": {"maximum": MAX_WORKERS, "observed_peak": MAX_WORKERS},
        "adopted_univ1_oof_chains": adopted,
        "new_chains": chains,
        "new_job_receipts": new_receipts,
    }


def cmd_validate(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    validate_contract(root, deep=True)
    _validate_preflight(root)
    _validate_scheduler(root)
    _validate_output_roster(root)
    receipts = [
        _artifact(job_receipt_path(root, job.kind, job.seed))
        for job in _job_specs()
        if _validate_job(root, job.kind, job.seed)
    ]
    if len(receipts) != TOTAL_JOBS or len({item["sha256"] for item in receipts}) != TOTAL_JOBS:
        raise ContractError("Expected ten distinct, authenticated new job receipts")
    payload = _terminal_payload(root, receipts)
    if args.seal:
        if training_receipt_path(root).exists() or training_receipt_path(root).is_symlink():
            raise ContractError("Terminal receipt is exactly-once; refusing to reseal")
        _write_json_once(training_receipt_path(root), payload)
    print(json.dumps({"status": "PASS", "sealed": bool(args.seal), "fit_accounting": _fit_accounting(), "concurrency": payload["concurrency"]}, indent=2))


def _validate_terminal(root: Path) -> dict[str, Any]:
    observed = _read_json(training_receipt_path(root))
    receipts = [_artifact(job_receipt_path(root, job.kind, job.seed)) for job in _job_specs()]
    # created_utc is intentionally the only non-derived field.
    expected = _terminal_payload(root, receipts)
    expected["created_utc"] = observed.get("created_utc")
    if observed != expected:
        raise ContractError("Exactly-once terminal receipt does not match current deep evidence")
    return observed


def cmd_verify(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    validate_contract(root, deep=True)
    stages = {"contract": "PASS"}
    if preflight_path(root).is_file():
        _validate_preflight(root)
        stages["preflight"] = "PASS"
    if scheduler_path(root).is_file():
        _validate_scheduler(root)
        stages["scheduler"] = "PASS"
    if training_receipt_path(root).is_file():
        _validate_output_roster(root)
        for job in _job_specs():
            _validate_job(root, job.kind, job.seed)
        _validate_terminal(root)
        stages["training"] = "PASS"
    print(json.dumps({"status": "PASS", "stages": stages}, indent=2))


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
    rows = []
    for job in _job_specs():
        state = "pending"
        if job_receipt_path(root, job.kind, job.seed).is_file():
            try:
                _validate_job(root, job.kind, job.seed)
            except Exception as exc:
                state = f"invalid:{type(exc).__name__}"
            else:
                state = "completed"
        elif _lock_active(_job_lock_path(root, job.kind, job.seed)):
            state = "running"
        elif failure_path(root, job.kind, job.seed).is_file():
            state = "failed_no_retry"
        elif any(path.exists() for path in (run_dir(root, job.kind, job.seed), request_path(root, job.kind, job.seed), log_path(root, job.kind, job.seed))):
            state = "partial_or_orphaned_no_retry"
        rows.append({"kind": job.kind, "encoder": job.encoder, "seed": job.seed, "new_fits": job.total_fits, "state": state})
    counts: dict[str, int] = {}
    completed_fits = 0
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
        if row["state"] == "completed":
            completed_fits += int(row["new_fits"])
    print(json.dumps({
        "output_root": str(root),
        "prepared": contract_path(root).is_file(),
        "preflight_passed": preflight_path(root).is_file(),
        "training_sealed": training_receipt_path(root).is_file(),
        "jobs": counts,
        "adopted_oof_fits": ADOPTED_OOF_FITS if contract_path(root).is_file() else 0,
        "completed_new_fits": completed_fits,
        "remaining_new_fits": NEW_FITS - completed_fits,
        "rows": rows,
    }, indent=2))


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
    train.set_defaults(func=cmd_train)
    validate = subparsers.add_parser("validate", parents=[common])
    validate.add_argument("--seal", action="store_true")
    validate.set_defaults(func=cmd_validate)
    verify = subparsers.add_parser("verify", parents=[common])
    verify.set_defaults(func=cmd_verify)
    status = subparsers.add_parser("status", parents=[common])
    status.set_defaults(func=cmd_status)
    internal = subparsers.add_parser("_train-one", parents=[common])
    internal.add_argument("--kind", required=True, choices=["univ1_refit", "virchow2_full"])
    internal.add_argument("--seed", required=True, type=int, choices=SEEDS)
    internal.set_defaults(func=cmd_internal_train_one)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
