#!/usr/bin/env python3
"""Governed five-seed extension for every official Aim-3 MIL ladder.

This is an additive lineage.  It authenticates and adopts the completed
seed-42/43/44 runs, then trains *only* seeds 45 and 46 below a new output
root.  Nothing in the sealed Aim-3 trees is modified.

The extension covers four components:

* fixed UNIv1 ladder: ten tasks, five OOF folds and a p75 refit per chain;
* repeated UNIv1 controls: five controls x three frozen WT draws, with the
  same five-fold-plus-refit contract;
* E3v Virchow2-CLS ladder: six tasks, five OOF folds, no refit;
* E1v Virchow2-CLS gene reference: one task, five OOF folds, no refit.

The five-seed estimand is always the patient native-logit ensemble: mean
slide logit within patient and seed, followed by the mean over seeds 42..46.
Seeds are algorithmic ensemble members, not inferential units.  The fixed
and E3v ladders use the predeclared 10,000-draw partially-paired bootstrap;
the repeated-control campaign uses its predeclared 20,000-draw common-random-
number bootstrap and three-draw consensus gate.  E0 is never trained here;
its five-seed gene reference is an explicit dependency on the Aim-1 extension.

Typical workflow::

    python aim3_ladders_five_seed_extension.py plan
    python aim3_ladders_five_seed_extension.py manifest --apply
    python aim3_ladders_five_seed_extension.py preflight --full-pack-check --apply
    python aim3_ladders_five_seed_extension.py train --dry-run
    python aim3_ladders_five_seed_extension.py train --apply --jobs 6
    python aim3_ladders_five_seed_extension.py analyze --apply
    python aim3_ladders_five_seed_extension.py verify

``train`` defaults to six concurrent chains and four DataLoader workers per
chain.  Its public ``job_inventory`` and ``train_command`` functions are also
the integration surface for the final-v9 master scheduler.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import aim1_virchow2_encoder_arm as legacy_e1v  # noqa: E402
import aim3_fixed_control_analysis as fixed_stats  # noqa: E402
import aim3_repeated_control_campaign as repeated_stats  # noqa: E402
import aim3_resolution_ladder as legacy_fixed  # noqa: E402
import aim3_virchow2_replication as legacy_e3v  # noqa: E402
from oceanpath.aim1 import evaluate, paths  # noqa: E402
from oceanpath.datasets.packed import PackedFeatureStore, validate_packed_dir  # noqa: E402
from oceanpath.workflows.training import validate_training_run_dir  # noqa: E402
from tools import aim1_e0_five_seed_extension as aim1_extension  # noqa: E402

SCHEMA_VERSION = 1
OLD_SEEDS = (42, 43, 44)
NEW_SEEDS = (45, 46)
ALL_SEEDS = (*OLD_SEEDS, *NEW_SEEDS)
N_FOLDS = 5
CAP = 8192
MAX_JOBS = 6
DEFAULT_NUM_WORKERS = 4
FIXED_EPOCH_BUDGET = 12
BOOTSTRAP_SEED = 20260817
FIXED_BOOTSTRAPS = 10_000
REPEATED_BOOTSTRAP_SEED = 20260826
REPEATED_BOOTSTRAPS = 20_000
E1V_BOOTSTRAPS = 2_000
CEILING_BOUND = 0.60
CHANCE = 0.50

SHARED_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "final_v9_mil_5seed_expansion_v1_20260823"
)
DEFAULT_ROOT = SHARED_ROOT / "aim3_ladders"
DEFAULT_E0_ROOT = aim1_extension.DEFAULT_CAMPAIGN_ROOT

OLD_FIXED_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/e3a")
OLD_REPEATED_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim3_controls_repeated_v1_20260819"
)
OLD_E3V_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim3_virchow2_cap8192_v1_20260820"
)
OLD_E1V_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/train/1a_pb_cap8192/virchow2_cls"
)
OLD_E0_ROOT = aim1_extension.DEFAULT_LEGACY_ROOT

OLD_FIXED_ANALYSIS = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim3_cap8192_corrected_v1_20260819/analysis/aim3_corrected.json"
)
OLD_REPEATED_ANALYSIS = OLD_REPEATED_ROOT / "analysis/aim3_repeated_control_report.json"
OLD_E3V_ANALYSIS = OLD_E3V_ROOT / "analysis/results.json"
OLD_E1V_ANALYSIS = OLD_E1V_ROOT / "analysis/results.json"
EXPECTED_OLD_ANALYSIS_SHA256 = {
    "fixed": "2622b71dddde795112df65026c30ca226ca9d8afdbe1d052f6be4372a4bbb731",
    "repeated": "809cff9a752319e7924f321f3990dfcc2b2f4a76503099ab83bcdbece8232ee3",
    "e3v": "27b8b0eee8320bb07794a84b6c952d86d46970dc1e2c379a55ebabd436c1a676",
    "e1v": "fa56e3ac3556f9ddc63a75322d48bb5d18c1298e49846810a79b8ed5927bb493",
}

if tuple(aim1_extension.ALL_SEEDS) != ALL_SEEDS:
    raise RuntimeError("Aim1/Aim3 five-seed rosters disagree")

UNI_PACK = Path(paths.PACKED_FEATURE_DIR)
V2_PACK = Path(legacy_e1v.V2_PACK)
UNI_EXPECTED_META_SHA256 = "44f1f0c80f4b8740c5fc82caafa96a8ead372c561fe53cda896d83972ff3c80b"
UNI_EXPECTED_INDEX_SHA256 = "705464267e35e454e50ea8510bc937e9f8d6a00b8b97bb9cf1e06da5fcfa9556"
V2_EXPECTED_META_SHA256 = "0dd3df5d0b933f40810a75b62d68a6d8297e0e592e98ab8a87847ce7417dd239"
V2_EXPECTED_INDEX_SHA256 = "2b462ac49fce6ba69d70b61b69faedd4f9e68855b8914329536ba1572fa2a87d"

FIXED_TASKS = tuple(legacy_fixed.TASKS)
FIXED_PAIRS = tuple(tuple(pair) for pair in legacy_fixed.PAIRS)
REPEATED_CONTROLS = tuple(repeated_stats.CONTROLS)
WT_DRAW_SEEDS = tuple(repeated_stats.WT_DRAW_SEEDS)
E3V_TASKS = tuple(legacy_e3v.TASKS)
FIXED_PAIR_BOOTSTRAP_SEEDS = {
    fine: fixed_stats._stable_seed(BOOTSTRAP_SEED, "primary_pair", fine)
    for fine, _control in FIXED_PAIRS
}
GENE_BOOTSTRAP_SEED = fixed_stats._stable_seed(BOOTSTRAP_SEED, "standalone", "gene")
E3V_PAIR_BOOTSTRAP_SEED = BOOTSTRAP_SEED

Component = Literal["fixed", "repeated", "e3v", "e1v"]


@dataclass(frozen=True, order=True)
class Job:
    """One independent five-fold training-chain job."""

    component: Component
    task: str
    model_seed: int
    draw_seed: int | None = None

    @property
    def key(self) -> str:
        draw = f"__wt{self.draw_seed}" if self.draw_seed is not None else ""
        return f"{self.component}__{self.task}{draw}__seed{self.model_seed}"

    @property
    def skip_finalize(self) -> bool:
        return self.component in {"e3v", "e1v"}

    @property
    def actual_fits(self) -> int:
        return N_FOLDS + (0 if self.skip_finalize else 1)


def job_inventory(*, seeds: Iterable[int] = NEW_SEEDS) -> list[Job]:
    """Stable 64-job inventory consumed by this launcher and the master one."""

    selected = tuple(int(seed) for seed in seeds)
    invalid = sorted(set(selected) - set(ALL_SEEDS))
    if invalid:
        raise ValueError(f"registered model seeds are {ALL_SEEDS}; got {invalid}")
    jobs = [Job("fixed", task, seed) for task in FIXED_TASKS for seed in selected]
    jobs += [
        Job("repeated", control, seed, draw)
        for control in REPEATED_CONTROLS
        for draw in WT_DRAW_SEEDS
        for seed in selected
    ]
    jobs += [Job("e3v", task, seed) for task in E3V_TASKS for seed in selected]
    jobs += [Job("e1v", "gene", seed) for seed in selected]
    return jobs


def job_counts(jobs: Iterable[Job] | None = None) -> dict[str, int]:
    values = list(job_inventory() if jobs is None else jobs)
    return {
        "chains": len(values),
        "oof_folds": len(values) * N_FOLDS,
        "refits": sum(not job.skip_finalize for job in values),
        "actual_mil_fits": sum(job.actual_fits for job in values),
    }


def sha256_file(path: Path, *, limit_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit_bytes
    with path.open("rb") as stream:
        while remaining is None or remaining > 0:
            size = 8 * 1024 * 1024 if remaining is None else min(8 * 1024 * 1024, remaining)
            block = stream.read(size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining not in (None, 0):
        raise RuntimeError(f"{path} is shorter than the authenticated prefix")
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def stat_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _copy_once(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)


def material_source_files() -> tuple[str, ...]:
    fixed = (
        "aim3_ladders_five_seed_extension.py",
        "aim1_virchow2_encoder_arm.py",
        "aim3_fixed_control_analysis.py",
        "aim3_repeated_control_campaign.py",
        "aim3_resolution_ladder.py",
        "aim3_virchow2_replication.py",
        "tools/aim1_e0_five_seed_extension.py",
        "tools/study_train.py",
        "pyproject.toml",
        "uv.lock",
    )
    package = tuple(
        path.relative_to(REPO).as_posix()
        for path in sorted((REPO / "src" / "oceanpath").rglob("*.py"))
    )
    configs = tuple(
        path.relative_to(REPO).as_posix()
        for path in sorted((REPO / "configs").rglob("*.yaml"))
    )
    values = (*fixed, *package, *configs)
    if len(values) != len(set(values)):
        raise AssertionError("duplicate material source path")
    return values


def validate_output_root(root: Path, *, must_exist: bool | None = None) -> Path:
    if not root.is_absolute():
        raise ValueError("--output-root must be absolute")
    resolved = root.resolve(strict=False)
    production = DEFAULT_ROOT.resolve(strict=False)
    temporary = Path("/tmp").resolve()
    if resolved != production and temporary not in resolved.parents:
        raise ValueError(f"root must be {production} (or /tmp for an isolated test)")
    sealed = (OLD_FIXED_ROOT, OLD_REPEATED_ROOT, OLD_E3V_ROOT, OLD_E1V_ROOT, OLD_E0_ROOT)
    if any(resolved == path.resolve(strict=False) or path.resolve(strict=False) in resolved.parents for path in sealed):
        raise ValueError("extension root overlaps a sealed source tree")
    if must_exist is True and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if must_exist is False and (resolved.exists() or resolved.is_symlink()):
        raise FileExistsError(resolved)
    return resolved


def fixed_manifest_source(task: str) -> Path:
    return legacy_fixed.manifest_path(task)


def fixed_split_source(task: str) -> Path:
    return legacy_fixed.split_dir(task)


def repeated_manifest_source(control: str, draw_seed: int) -> Path:
    return repeated_stats.manifest_path(OLD_REPEATED_ROOT, control, draw_seed)


def repeated_split_source(control: str, draw_seed: int) -> Path:
    return repeated_stats.split_dir(OLD_REPEATED_ROOT, control, draw_seed)


def fixed_manifest(root: Path, task: str) -> Path:
    return root / "inputs" / "fixed" / task / "manifest.csv"


def fixed_splits(root: Path, task: str) -> Path:
    return root / "inputs" / "fixed" / task / "splits"


def repeated_manifest(root: Path, control: str, draw_seed: int) -> Path:
    return root / "inputs" / "repeated" / control / f"wt{draw_seed}" / "manifest.csv"


def repeated_splits(root: Path, control: str, draw_seed: int) -> Path:
    return root / "inputs" / "repeated" / control / f"wt{draw_seed}" / "splits"


def e1v_manifest(root: Path) -> Path:
    return root / "inputs" / "e1v" / "manifest.csv"


def e1v_splits(root: Path) -> Path:
    return root / "inputs" / "e1v" / "splits"


def old_run_dir(job: Job) -> Path:
    if job.model_seed not in OLD_SEEDS:
        raise ValueError("old_run_dir accepts seeds 42..44 only")
    if job.component == "fixed":
        return OLD_FIXED_ROOT / "train" / job.task / f"seed{job.model_seed}"
    if job.component == "repeated":
        assert job.draw_seed is not None
        return (
            OLD_REPEATED_ROOT
            / "train"
            / job.task
            / f"wt{job.draw_seed}"
            / f"seed{job.model_seed}"
        )
    if job.component == "e3v":
        return OLD_E3V_ROOT / "train" / job.task / f"seed{job.model_seed}"
    return OLD_E1V_ROOT / f"seed{job.model_seed}"


def run_dir(root: Path, job: Job) -> Path:
    if job.component == "fixed":
        return root / "train" / "fixed_univ1" / job.task / f"seed{job.model_seed}"
    if job.component == "repeated":
        assert job.draw_seed is not None
        return (
            root
            / "train"
            / "repeated_univ1"
            / job.task
            / f"wt{job.draw_seed}"
            / f"seed{job.model_seed}"
        )
    if job.component == "e3v":
        return root / "train" / "e3v_virchow2_cls" / job.task / f"seed{job.model_seed}"
    return root / "train" / "e1v_virchow2_cls" / f"seed{job.model_seed}"


def component_root(output_root: Path = SHARED_ROOT) -> Path:
    """Resolve either the shared final-v9 root or its Aim3 component root."""

    resolved = Path(output_root).resolve(strict=False)
    return resolved if resolved.name == "aim3_ladders" else resolved / "aim3_ladders"


def resolved_run_dir(root: Path, job: Job) -> Path:
    return old_run_dir(job) if job.model_seed in OLD_SEEDS else run_dir(root, job)


def _training_argv(
    root: Path,
    job: Job,
    *,
    num_workers: int,
    attempt_id: str,
) -> list[str]:
    if job.model_seed not in NEW_SEEDS:
        raise ValueError("sealed seeds 42..44 are adoption-only")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    destination = run_dir(root, job)
    hydra_dir = root / "state" / "hydra" / job.key / attempt_id
    common = [
        "platform=colon_workstation",
        "data=aim1",
        "+data.cohort_column=cohort",
        "splits=aim1_balanced",
        f"splits.seed={job.model_seed}",
        "model=abmil",
        "model.embed_dim=512",
        "model.attn_dim=384",
        "model.input_dropout=0.10",
        f"model.dropout={paths.DROPOUT}",
        "training=aim1",
        f"training.lr={paths.LR:g}",
        f"training.weight_decay={paths.WEIGHT_DECAY:g}",
        f"training.seed={job.model_seed}",
        f"training.dataset_max_instances={CAP}",
        "training.eval_full_bags=true",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        f"training.num_workers={num_workers}",
        f"train_dir={destination}",
        f"hydra.run.dir={hydra_dir}",
        "hydra.job.chdir=false",
    ]
    if job.component in {"fixed", "repeated"}:
        common += [
            "encoder=univ1",
            f"training.fixed_epoch_budget={FIXED_EPOCH_BUDGET}",
            "training.skip_finalize=false",
            f"training.packed_dir={UNI_PACK}",
            "training.verify_packed_source=false",
        ]
    else:
        common += [
            "encoder=virchow2",
            "encoder.feature_dim=1280",
            f"extraction.coords_dir={legacy_e1v.V2_COORDS}",
            f"extraction.coords_subdir={legacy_e1v.V2_COORDS}",
            "extraction.patch_size=224",
            "training.skip_finalize=true",
            f"training.packed_dir={V2_PACK}",
            "training.verify_packed_source=false",
        ]
    if job.component == "fixed":
        data_name = f"aim1_e3a_{job.task}"
        common += [
            f"data.aim1_model=e3a_{job.task}",
            f"data.name={data_name}",
            f"data.manifest_stem={data_name}",
            f"data.csv_path={fixed_manifest(root, job.task)}",
            f"+splits.output_dir={fixed_splits(root, job.task)}",
            f"exp_name=aim3_5seed_fixed_{job.task}_seed{job.model_seed}",
        ]
    elif job.component == "repeated":
        assert job.draw_seed is not None
        data_name = repeated_stats.manifest_name(job.task, job.draw_seed)
        common += [
            f"data.aim1_model={data_name.removeprefix('aim1_')}",
            f"data.name={data_name}",
            f"data.manifest_stem={data_name}",
            f"data.csv_path={repeated_manifest(root, job.task, job.draw_seed)}",
            f"+splits.output_dir={repeated_splits(root, job.task, job.draw_seed)}",
            f"exp_name=aim3_5seed_repeated_{job.task}_wt{job.draw_seed}_seed{job.model_seed}",
        ]
    elif job.component == "e3v":
        data_name = f"aim1_e3a_{job.task}"
        common += [
            f"data.aim1_model=e3a_{job.task}",
            f"data.name={data_name}",
            f"data.manifest_stem={data_name}",
            f"data.csv_path={fixed_manifest(root, job.task)}",
            f"+splits.output_dir={fixed_splits(root, job.task)}",
            f"training.fixed_epoch_budget={FIXED_EPOCH_BUDGET}",
            f"exp_name=aim3_5seed_e3v_{job.task}_seed{job.model_seed}",
        ]
    else:
        common += [
            "data.aim1_model=1a",
            "data.name=aim1_1a",
            "data.manifest_stem=aim1_dev",
            f"data.csv_path={e1v_manifest(root)}",
            f"+splits.output_dir={e1v_splits(root)}",
            f"exp_name=aim3_5seed_e1v_gene_seed{job.model_seed}",
        ]
    frozen_repo = root / "source_snapshot"
    executable = frozen_repo / "tools" / "study_train.py"
    return [sys.executable, str(executable), "hydra-train", *common]


def train_command(
    root: Path,
    job: Job,
    num_workers: int = DEFAULT_NUM_WORKERS,
    *,
    attempt_id: str = "scheduler",
) -> list[str]:
    """Public, side-effect-free master-scheduler command builder."""

    root = Path(root).resolve(strict=False)
    if not attempt_id or "/" in attempt_id or "\\" in attempt_id:
        raise ValueError("attempt_id must be one non-empty path component")
    return _training_argv(root, job, num_workers=num_workers, attempt_id=attempt_id)


def build_training_jobs(
    root: Path = SHARED_ROOT,
    *,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> list[dict[str, Any]]:
    """Return the exact one-slot inventory for a study-wide six-slot scheduler.

    The emitted command re-enters this controller for locking, input-contract
    checks, logging, and post-run validation; a master scheduler must not call
    ``study_train.py`` directly and thereby bypass those guards.
    """

    root = component_root(root)
    values = []
    for job in job_inventory():
        external_worker_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "train-one",
            "--output-root",
            str(root),
            "--job-key",
            job.key,
            "--num-workers",
            str(num_workers),
            "--external-scheduler",
        ]
        values.append(
            {
                "job_id": f"aim3.{job.key}",
                "job_key": job.key,
                "component": "aim3_ladders",
                "stage": job.component,
                "task": job.task,
                "draw_seed": job.draw_seed,
                "seed": job.model_seed,
                "scheduler_slots": 1,
                "oof_fold_fits": N_FOLDS,
                "refit_fits": int(not job.skip_finalize),
                "fit_count": job.actual_fits,
                "output": str(run_dir(root, job)),
                "external_worker_command": external_worker_command,
                # Alias retained for simple schedulers; both fields are the
                # guarded one-job controller command, never raw study_train.
                "command": external_worker_command,
            }
        )
    # Launch the two long full-development E1v chains first, then E3v, then
    # the smaller UNIv1 chains.  This reduces the six-worker long tail without
    # changing any model result.
    priority = {"e1v": 0, "e3v": 1, "fixed": 2, "repeated": 3}
    return sorted(values, key=lambda value: (priority[str(value["stage"])], str(value["job_id"])))


def job_from_key(key: str) -> Job:
    matches = [job for job in job_inventory() if job.key == key]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous Aim3 extension job key: {key}")
    return matches[0]


def job_receipt_path(root: Path, job: Job) -> Path:
    return root / "receipts" / "jobs" / f"{job.key}.json"


def _job_receipt_payload(root: Path, job: Job, artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "job": asdict(job) | {"key": job.key},
        "training_process_returncode": 0,
        "contract": identity(root / "inputs" / "experiment_contract.json"),
        "preflight": identity(root / "receipts" / "preflight.json"),
        "artifacts": artifacts,
    }


def _validate_job_receipt(root: Path, job: Job) -> dict[str, Any]:
    path = job_receipt_path(root, job)
    stored = _read_json(path)
    artifacts = validate_job_chain(root, job)
    expected = _job_receipt_payload(root, job, artifacts)
    if stored != expected:
        raise RuntimeError(f"job receipt differs from live native validation: {job.key}")
    return stored


def _ensure_job_receipt(root: Path, job: Job) -> dict[str, Any]:
    artifacts = validate_job_chain(root, job)
    expected = _job_receipt_payload(root, job, artifacts)
    path = job_receipt_path(root, job)
    if path.is_file():
        if _read_json(path) != expected:
            raise RuntimeError(f"refusing to replace divergent job receipt: {job.key}")
    else:
        _write_json_once(path, expected)
    return expected


def _manifest_split_inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in FIXED_TASKS:
        records += [identity(fixed_manifest(root, task))]
        records += [
            identity(fixed_splits(root, task) / name)
            for name in ("splits.parquet", ".integrity_hash")
        ]
    for control in REPEATED_CONTROLS:
        for draw in WT_DRAW_SEEDS:
            records += [identity(repeated_manifest(root, control, draw))]
            records += [
                identity(repeated_splits(root, control, draw) / name)
                for name in ("splits.parquet", ".integrity_hash")
            ]
    records += [identity(e1v_manifest(root))]
    records += [identity(e1v_splits(root) / name) for name in ("splits.parquet", ".integrity_hash")]
    return records


def _validate_refit(directory: Path) -> dict[str, Any]:
    summary_path = directory / "final" / "finalize_summary.json"
    info_path = directory / "final" / "refit" / "info.json"
    checkpoint_path = directory / "final" / "refit" / "model.ckpt"
    summary = _read_json(summary_path)
    info = _read_json(info_path)
    refit = summary.get("refit")
    if not isinstance(refit, dict) or refit != info:
        raise RuntimeError(f"refit summary/info mismatch: {directory}")
    if info.get("strategy") != "refit" or info.get("refit_epoch_rule") != "p75":
        raise RuntimeError(f"refit is not the prespecified p75 strategy: {directory}")
    epochs = info.get("fold_best_epochs")
    if not isinstance(epochs, list) or len(epochs) != N_FOLDS:
        raise RuntimeError(f"refit does not bind exactly five fold epochs: {directory}")
    expected = int(np.ceil(np.percentile(np.asarray(epochs, dtype=float), 75)))
    if int(info.get("refit_epochs", -1)) != expected:
        raise RuntimeError(f"refit epoch is not p75(fold best epochs): {directory}")
    model_path = Path(str(info.get("model_path", "")))
    if model_path.name != "model.ckpt" or model_path.parent.name != "refit":
        raise RuntimeError(f"refit info points to an invalid model path: {directory}")
    return {
        "finalize_summary": identity(summary_path),
        "refit_info": identity(info_path),
        "refit_checkpoint": identity(checkpoint_path),
        "refit_epochs": expected,
    }


def validate_chain(directory: Path, *, skip_finalize: bool) -> dict[str, Any]:
    completion = validate_training_run_dir(directory, require_test_predictions=True)
    if completion.get("n_folds") != N_FOLDS:
        raise RuntimeError(f"chain does not contain exactly five folds: {directory}")
    if completion.get("skip_finalize") is not skip_finalize:
        raise RuntimeError(f"skip_finalize mismatch: {directory}")
    identity_record = _read_json(directory / "training_identity.json")
    payload = identity_record.get("payload")
    fingerprint = identity_record.get("fingerprint")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    if not isinstance(fingerprint, str) or hashlib.sha256(canonical.encode()).hexdigest()[:16] != fingerprint:
        raise RuntimeError(f"invalid training fingerprint: {directory}")
    oof_path = directory / "oof_predictions.parquet"
    oof = repeated_stats._validate_exact_oof_union(directory, oof_path)
    if "logit" not in oof or not np.isfinite(pd.to_numeric(oof["logit"], errors="coerce")).all():
        raise RuntimeError(f"OOF lacks finite native logits: {directory}")
    record: dict[str, Any] = {
        "directory": str(directory.resolve()),
        "training_identity": identity(directory / "training_identity.json"),
        "training_completion": identity(directory / "training_completion.json"),
        "oof_predictions": identity(oof_path),
        "fold_completions": [identity(directory / f"fold_{fold}" / "completion.json") for fold in range(N_FOLDS)],
        "skip_finalize": skip_finalize,
    }
    if skip_finalize:
        if (directory / "final" / "refit" / "model.ckpt").exists():
            raise RuntimeError(f"OOF-only chain unexpectedly contains a refit: {directory}")
    else:
        record["refit"] = _validate_refit(directory)
    return record


def _assert_recipe(directory: Path, job: Job) -> None:
    material = (
        _read_json(directory / "training_identity.json")
        .get("payload", {})
        .get("material_config", {})
    )
    training = material.get("training", {})
    required = {
        "dataset_max_instances": CAP,
        "eval_full_bags": True,
        "train_sampling_strategy": "patient_natural",
        "sample_weight_column": None,
        "lr": paths.LR,
        "weight_decay": paths.WEIGHT_DECAY,
        "seed": job.model_seed,
        "skip_finalize": job.skip_finalize,
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise RuntimeError(f"{job.key}: recipe drift at {key}={training.get(key)!r}")
    expected_budget = FIXED_EPOCH_BUDGET if job.component in {"fixed", "repeated", "e3v"} else None
    observed_budget = training.get("fixed_epoch_budget")
    if job.model_seed in NEW_SEEDS and observed_budget != expected_budget:
        raise RuntimeError(f"{job.key}: fixed-epoch fallback drift")
    if job.model_seed in OLD_SEEDS and observed_budget not in ({None, FIXED_EPOCH_BUDGET} if expected_budget else {None}):
        raise RuntimeError(f"{job.key}: historical fixed-epoch fallback is invalid")
    model = material.get("model", {})
    expected_model = {"arch": "abmil", "embed_dim": 512, "attn_dim": 384, "input_dropout": 0.1, "dropout": paths.DROPOUT}
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise RuntimeError(f"{job.key}: model recipe drift at {key}")
    encoder = material.get("encoder", {})
    expected_encoder = ("uni_v1", 1024) if job.component in {"fixed", "repeated"} else ("virchow2", 1280)
    if (encoder.get("name"), encoder.get("feature_dim")) != expected_encoder:
        raise RuntimeError(f"{job.key}: encoder recipe drift")
    splits = material.get("splits", {})
    if splits.get("scheme") != "predefined_oof_kfold" or splits.get("n_folds") != N_FOLDS:
        raise RuntimeError(f"{job.key}: frozen split recipe drift")


def _job_inputs(root: Path, job: Job, *, adopted: bool) -> tuple[Path, Path]:
    if job.component in {"fixed", "e3v"}:
        return (
            (fixed_manifest_source(job.task), fixed_split_source(job.task))
            if adopted
            else (fixed_manifest(root, job.task), fixed_splits(root, job.task))
        )
    if job.component == "repeated":
        assert job.draw_seed is not None
        return (
            (repeated_manifest_source(job.task, job.draw_seed), repeated_split_source(job.task, job.draw_seed))
            if adopted
            else (repeated_manifest(root, job.task, job.draw_seed), repeated_splits(root, job.task, job.draw_seed))
        )
    source_split = REPO / paths.SPLIT_ROOT / "aim1_1a" / paths.SPLIT_NAME
    return (paths.DEV_MANIFEST, source_split) if adopted else (e1v_manifest(root), e1v_splits(root))


def validate_job_chain(root: Path, job: Job, *, adopted: bool = False) -> dict[str, Any]:
    directory = old_run_dir(job) if adopted else run_dir(root, job)
    record = validate_chain(directory, skip_finalize=job.skip_finalize)
    _assert_recipe(directory, job)
    manifest_path, split_directory = _job_inputs(root, job, adopted=adopted)
    run_identity = _read_json(directory / "training_identity.json")
    evidence = run_identity.get("payload", {}).get("input_evidence", {})
    if evidence.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError(f"{job.key}: training manifest evidence drifted")
    if evidence.get("split_integrity_sha256") != sha256_file(split_directory / ".integrity_hash"):
        raise RuntimeError(f"{job.key}: training split evidence drifted")
    oof = pd.read_parquet(directory / "oof_predictions.parquet")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    if "slide_id" not in oof or oof["slide_id"].duplicated().any():
        raise RuntimeError(f"{job.key}: OOF slide IDs are invalid")
    if set(oof["slide_id"].astype(str)) != set(manifest["slide_id"].astype(str)):
        raise RuntimeError(f"{job.key}: OOF is not exact manifest slide coverage")
    merged = oof.merge(
        manifest[["slide_id", "target_label", "k_fold"]],
        on="slide_id",
        validate="one_to_one",
    )
    if not pd.to_numeric(merged["label"]).astype(int).eq(pd.to_numeric(merged["target_label"]).astype(int)).all():
        raise RuntimeError(f"{job.key}: OOF labels disagree with manifest")
    if not pd.to_numeric(merged["fold"]).astype(int).eq(pd.to_numeric(merged["k_fold"]).astype(int)).all():
        raise RuntimeError(f"{job.key}: OOF folds disagree with manifest")
    record["manifest"] = identity(manifest_path)
    record["splits"] = identity(split_directory / "splits.parquet")
    record["split_integrity"] = identity(split_directory / ".integrity_hash")
    return record


def _adopted_inventory() -> list[dict[str, Any]]:
    records = []
    for job in job_inventory(seeds=OLD_SEEDS):
        record = validate_job_chain(DEFAULT_ROOT, job, adopted=True)
        records.append({"job": asdict(job), "job_key": job.key, "artifacts": record})
    return records


def _validate_adopted_inventory(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-authenticate every artifact in the immutable seed-42/43/44 adoption."""

    jobs = job_inventory(seeds=OLD_SEEDS)
    adopted = contract.get("adopted_old_chains")
    if not isinstance(adopted, list) or len(adopted) != len(jobs):
        raise RuntimeError("old-chain adoption inventory is incomplete")
    for job, stored in zip(jobs, adopted, strict=True):
        if stored.get("job") != asdict(job) or stored.get("job_key") != job.key:
            raise RuntimeError(f"old-chain adoption order drifted at {job.key}")
        observed = validate_job_chain(DEFAULT_ROOT, job, adopted=True)
        if stored.get("artifacts") != observed:
            raise RuntimeError(f"adopted old chain changed: {job.key}")
    return adopted


def _copy_input_snapshots(root: Path) -> None:
    for task in FIXED_TASKS:
        _copy_once(fixed_manifest_source(task), fixed_manifest(root, task))
        for name in ("splits.parquet", ".integrity_hash", "summary.json", "split_metadata.json"):
            source = fixed_split_source(task) / name
            if source.is_file():
                _copy_once(source, fixed_splits(root, task) / name)
    for control in REPEATED_CONTROLS:
        for draw in WT_DRAW_SEEDS:
            _copy_once(repeated_manifest_source(control, draw), repeated_manifest(root, control, draw))
            for name in ("splits.parquet", ".integrity_hash", "summary.json", "split_metadata.json"):
                source = repeated_split_source(control, draw) / name
                if source.is_file():
                    _copy_once(source, repeated_splits(root, control, draw) / name)
    aim1_split = REPO / paths.SPLIT_ROOT / "aim1_1a" / paths.SPLIT_NAME
    _copy_once(paths.DEV_MANIFEST, e1v_manifest(root))
    for name in ("splits.parquet", ".integrity_hash", "summary.json", "split_metadata.json"):
        source = aim1_split / name
        if source.is_file():
            _copy_once(source, e1v_splits(root) / name)


def _snapshot_sources(root: Path) -> list[dict[str, Any]]:
    records = []
    for relative in material_source_files():
        source = REPO / relative
        destination = root / "source_snapshot" / relative
        _copy_once(source, destination)
        records.append({"relative_path": relative, "source": identity(source), "snapshot": identity(destination)})
    return records


def _pack_record(pack: Path, *, expected_dim: int, expected_meta: str, expected_index: str) -> dict[str, Any]:
    meta = validate_packed_dir(pack)
    if int(meta.feat_dim) != expected_dim:
        raise RuntimeError(f"{pack}: dimension {meta.feat_dim} != {expected_dim}")
    meta_identity = identity(pack / "meta.json")
    index_identity = identity(pack / "index.parquet")
    if meta_identity["sha256"] != expected_meta or index_identity["sha256"] != expected_index:
        raise RuntimeError(f"{pack}: expanded-store identity changed")
    return {
        "path": str(pack.resolve()),
        "meta": meta_identity,
        "index": index_identity,
        "features": stat_identity(pack / "features.bin"),
        "coords": stat_identity(pack / "coords.bin"),
        "n_slides": int(meta.n_slides),
        "feat_dim": int(meta.feat_dim),
        "source_dir": str(Path(meta.source_dir).resolve()),
    }


def _old_analysis_inventory() -> dict[str, dict[str, Any]]:
    paths_by_name = {
        "fixed": OLD_FIXED_ANALYSIS,
        "repeated": OLD_REPEATED_ANALYSIS,
        "e3v": OLD_E3V_ANALYSIS,
        "e1v": OLD_E1V_ANALYSIS,
    }
    records = {name: identity(path) for name, path in paths_by_name.items()}
    for name, record in records.items():
        if record["sha256"] != EXPECTED_OLD_ANALYSIS_SHA256[name]:
            raise RuntimeError(f"sealed {name} three-seed analysis identity changed")
    return records


def _selected_slide_ids(root: Path, *, encoder: str) -> list[str]:
    if encoder == "univ1":
        manifests = [fixed_manifest(root, task) for task in FIXED_TASKS]
        manifests += [
            repeated_manifest(root, control, draw)
            for control in REPEATED_CONTROLS
            for draw in WT_DRAW_SEEDS
        ]
    elif encoder == "virchow2":
        manifests = [fixed_manifest(root, task) for task in E3V_TASKS] + [e1v_manifest(root)]
    else:
        raise ValueError(encoder)
    selected: set[str] = set()
    for path in manifests:
        frame = pd.read_csv(path, usecols=["slide_id"])
        selected.update(frame["slide_id"].astype(str))
    return sorted(selected)


def verify_selected_pack_against_h5(pack_dir: Path, slide_ids: Iterable[str]) -> dict[str, Any]:
    """Byte/value-check selected packed slides against their source H5 files.

    This is the safe bridge from the 2,087-slide historical packs to the
    append-expanded 2,128-slide Orion packs.  A whole-store hash is expected to
    differ; every selected conventional slide must remain exactly equal after
    the pack dtype conversion, including coordinates.
    """

    import h5py

    store = PackedFeatureStore(pack_dir)
    selected = tuple(sorted(set(map(str, slide_ids))))
    missing = sorted(set(selected) - set(store.slide_ids))
    if missing:
        raise RuntimeError(f"{pack_dir}: {len(missing)} selected slides are missing")
    source_dir = Path(store.meta.source_dir)
    digest = hashlib.sha256()
    for slide_id in selected:
        source = source_dir / f"{slide_id}.h5"
        if not source.is_file():
            raise FileNotFoundError(source)
        pos = store.position(slide_id)
        packed_features = store.read_features(pos)
        packed_coords = store.read_coords(pos) if store.has_coords else None
        with h5py.File(source, "r") as handle:
            dataset = handle["features"]
            packed_dim = packed_features.shape[1]
            if dataset.ndim == 3 and dataset.shape[0] == 1:
                source_rows, source_dim = dataset.shape[1:]
                source_slice = (0, slice(None), slice(0, packed_dim))
            elif dataset.ndim == 2:
                source_rows, source_dim = dataset.shape
                source_slice = (slice(None), slice(0, packed_dim))
            else:
                raise RuntimeError(f"packed/source feature shape mismatch: {slide_id}")
            if source_rows != packed_features.shape[0]:
                raise RuntimeError(f"packed/source feature shape mismatch: {slide_id}")
            if source_dim < packed_dim:
                raise RuntimeError(f"packed/source feature dimension mismatch: {slide_id}")
            # Packed encoder variants may deliberately retain a leading
            # subvector of the source embedding.  In particular, the
            # Virchow2-CLS pack is the float16 H5 prefix [0:1280] of each
            # 2560-dimensional TRIDENT row.  Reproduce that declared packing
            # transform while remaining an identity transform for UNI-v1.
            # Slice in H5 so the CLS check does not read the unused 1,280
            # Virchow2 dimensions into memory.
            expected_features = np.asarray(dataset[source_slice]).astype(
                packed_features.dtype, copy=False
            )
            if not np.array_equal(packed_features, expected_features):
                raise RuntimeError(f"packed/source feature mismatch: {slide_id}")
            if store.has_coords:
                coords = np.asarray(handle["coords"])
                if coords.ndim == 3 and coords.shape[0] == 1:
                    coords = coords[0]
                if not np.array_equal(packed_coords, coords.astype(np.int32, copy=False)):
                    raise RuntimeError(f"packed/source coordinate mismatch: {slide_id}")
        digest.update(slide_id.encode() + b"\0")
        digest.update(np.ascontiguousarray(packed_features).view(np.uint8))
        if packed_coords is not None:
            digest.update(np.ascontiguousarray(packed_coords).view(np.uint8))
    return {"n_slides": len(selected), "selected_content_sha256": digest.hexdigest()}


def _verify_contract(root: Path, *, verify_live_source: bool = True) -> dict[str, Any]:
    contract_path = root / "inputs" / "experiment_contract.json"
    contract = _read_json(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION or contract.get("status") != "prepared":
        raise RuntimeError("invalid experiment contract")
    if Path(str(contract.get("output_root", ""))).resolve() != root:
        raise RuntimeError("contract output root mismatch")
    expected_jobs = [asdict(job) | {"key": job.key, "actual_fits": job.actual_fits} for job in job_inventory()]
    if contract.get("new_jobs") != expected_jobs:
        raise RuntimeError("contract job inventory drifted")
    for record in contract.get("input_files", []):
        if identity(Path(record["path"])) != record:
            raise RuntimeError(f"input snapshot changed: {record.get('path')}")
    for record in contract.get("sources", []):
        snapshot = record["snapshot"]
        if identity(Path(snapshot["path"])) != snapshot:
            raise RuntimeError(f"source snapshot changed: {snapshot['path']}")
        if verify_live_source and identity(REPO / record["relative_path"])["sha256"] != record["source"]["sha256"]:
            raise RuntimeError(f"live source drifted after manifest: {record['relative_path']}")
    for pack in contract.get("packs", {}).values():
        if identity(Path(pack["meta"]["path"])) != pack["meta"]:
            raise RuntimeError("packed meta changed")
        if identity(Path(pack["index"]["path"])) != pack["index"]:
            raise RuntimeError("packed index changed")
        for name in ("features", "coords"):
            if stat_identity(Path(pack[name]["path"])) != pack[name]:
                raise RuntimeError(f"packed {name} size/mtime changed")
    if contract.get("old_analysis") != _old_analysis_inventory():
        raise RuntimeError("sealed three-seed analysis ledger changed")
    return contract


def cmd_plan(args: argparse.Namespace) -> None:
    root = Path(args.output_root).resolve(strict=False)
    jobs = job_inventory()
    by_component = {
        component: job_counts(job for job in jobs if job.component == component)
        for component in ("fixed", "repeated", "e3v", "e1v")
    }
    payload = {
        "output_root": str(root),
        "e0_dependency": str(Path(args.e0_root).resolve(strict=False)),
        "old_seeds_adopted": list(OLD_SEEDS),
        "new_seeds_trained": list(NEW_SEEDS),
        "scheduler_max_parallel": MAX_JOBS,
        "new": job_counts(jobs),
        "by_component": by_component,
        "five_seed_final": {
            "aim3_chains": 155,
            "aim3_oof_folds": 775,
            "aim3_refits": 125,
            "aim3_actual_mil_fits": 900,
            "e1v_gene_reference_chains": 5,
            "e1v_gene_reference_oof_fits": 25,
        },
        "jobs": [asdict(job) | {"key": job.key, "actual_fits": job.actual_fits} for job in jobs],
    }
    print(json.dumps(payload, indent=2))


def cmd_jobs(args: argparse.Namespace) -> None:
    jobs = build_training_jobs(Path(args.output_root), num_workers=args.num_workers)
    print(
        json.dumps(
            {
                "global_max_concurrent_gpu_trainers": MAX_JOBS,
                "extension_jobs": jobs,
                "fit_census": job_counts(),
            },
            indent=2,
        )
    )


def cmd_manifest(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=False)
    e0_root = Path(args.e0_root).resolve(strict=False)
    if not args.apply:
        print(f"DRY RUN — would exclusively create {root}")
        print("Would hash-adopt 96 old chains and snapshot 26 manifest/split contracts.")
        print("No files were written; pass --apply to prepare the lineage.")
        return
    root.mkdir(parents=True, exist_ok=False)
    try:
        _copy_input_snapshots(root)
        sources = _snapshot_sources(root)
        packs = {
            "univ1": _pack_record(
                UNI_PACK,
                expected_dim=1024,
                expected_meta=UNI_EXPECTED_META_SHA256,
                expected_index=UNI_EXPECTED_INDEX_SHA256,
            ),
            "virchow2_cls": _pack_record(
                V2_PACK,
                expected_dim=1280,
                expected_meta=V2_EXPECTED_META_SHA256,
                expected_index=V2_EXPECTED_INDEX_SHA256,
            ),
        }
        adopted = _adopted_inventory()
        contract = {
            "schema_version": SCHEMA_VERSION,
            "status": "prepared",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(root),
            "e0_dependency_root": str(e0_root),
            "e0_dependency_receipt": str(
                e0_root / "receipts" / "five_seed_training_validation.json"
            ),
            "protocol": {
                "old_seeds": list(OLD_SEEDS),
                "new_seeds": list(NEW_SEEDS),
                "all_seeds": list(ALL_SEEDS),
                "max_parallel_chains": MAX_JOBS,
                "default_num_workers": DEFAULT_NUM_WORKERS,
                "native_logit_ensemble": "mean slides within patient and seed, then mean seeds 42..46",
                "fixed_and_repeated_finalize": "required p75 full-data refit",
                "e3v_and_e1v_finalize": "OOF-only; skip_finalize=true",
            },
            "new_counts": job_counts(),
            "new_jobs": [asdict(job) | {"key": job.key, "actual_fits": job.actual_fits} for job in job_inventory()],
            "input_files": _manifest_split_inventory(root),
            "sources": sources,
            "packs": packs,
            "old_analysis": _old_analysis_inventory(),
            "selected_slides": {
                "univ1": _selected_slide_ids(root, encoder="univ1"),
                "virchow2_cls": _selected_slide_ids(root, encoder="virchow2"),
            },
            "adopted_old_chains": adopted,
        }
        _write_json_once(root / "inputs" / "experiment_contract.json", contract)
        _write_json_once(
            root / "receipts" / "manifest.json",
            {
                "status": "PASS",
                "contract": identity(root / "inputs" / "experiment_contract.json"),
                "adopted_chains": len(adopted),
                "new_chains": len(job_inventory()),
            },
        )
    except Exception:
        # A failed exclusive preparation remains visibly partial.  It is never
        # overwritten or silently repaired; choose a fresh versioned root.
        raise
    print(f"PASS — prepared immutable Aim-3 five-seed extension at {root}")


def _require_e0_dependency(e0_root: Path) -> dict[str, Any]:
    """Authenticate the actual Aim1 controller receipt and all five runs."""

    e0_root = e0_root.resolve(strict=True)
    receipt_path = e0_root / "receipts" / "five_seed_training_validation.json"
    stored = _read_json(receipt_path)
    expected, patients = aim1_extension._validate_five_seed_runs(
        e0_root,
        aim1_extension.DEFAULT_LEGACY_ROOT.resolve(strict=True),
        aim1_extension.DEFAULT_MANIFEST.resolve(strict=True),
        aim1_extension.DEFAULT_SPLIT_DIR.resolve(strict=True),
    )
    if stored != expected:
        raise RuntimeError("Aim1 five-seed training receipt differs from live authentication")
    expected_roots = {
        str(seed): str(
            aim1_extension.resolve_run_dir(
                e0_root, aim1_extension.DEFAULT_LEGACY_ROOT, seed
            ).resolve()
        )
        for seed in ALL_SEEDS
    }
    if (
        stored.get("status") != "complete"
        or stored.get("model_seeds") != list(ALL_SEEDS)
        or stored.get("resolved_roots") != expected_roots
        or set(stored.get("runs", {})) != {str(seed) for seed in ALL_SEEDS}
        or stored.get("fold_layout_shared_across_all_seeds") is not True
        or stored.get("p75_refit_authenticated_for_all_seeds") is not True
    ):
        raise RuntimeError("Aim1 five-seed training receipt contract is incomplete")
    return {"receipt": identity(receipt_path), "payload": stored, "patients": patients}


def cmd_preflight(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    contract = _verify_contract(root)
    selected_receipts: dict[str, Any] = {}
    if args.full_pack_check:
        selected_receipts = {
            "univ1": verify_selected_pack_against_h5(UNI_PACK, contract["selected_slides"]["univ1"]),
            "virchow2_cls": verify_selected_pack_against_h5(V2_PACK, contract["selected_slides"]["virchow2_cls"]),
        }
    if args.apply and not args.full_pack_check:
        raise ValueError("persisted preflight requires --full-pack-check")
    e0_path = Path(args.e0_root).resolve(strict=False)
    e0_evidence: dict[str, Any] | str = "pending; required for analysis/sealing, not training"
    receipt_candidate = e0_path / "receipts" / "five_seed_training_validation.json"
    if receipt_candidate.is_file():
        e0_evidence = _require_e0_dependency(e0_path)["receipt"]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "contract": identity(root / "inputs" / "experiment_contract.json"),
        "e0_dependency": e0_evidence,
        "selected_pack_source_equivalence": selected_receipts or "not requested; use --full-pack-check before training",
        "full_selected_pack_check": bool(args.full_pack_check),
        "packs": contract["packs"],
        "pending_new_chains": sum(not (run_dir(root, job) / "training_completion.json").is_file() for job in job_inventory()),
    }
    if args.apply:
        _write_json_once(root / "receipts" / "preflight.json", payload)
    print(json.dumps(payload, indent=2))


def _verify_preflight_receipt(root: Path) -> dict[str, Any]:
    path = root / "receipts" / "preflight.json"
    receipt = _read_json(path)
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "PASS"
        or receipt.get("full_selected_pack_check") is not True
        or receipt.get("contract") != identity(root / "inputs" / "experiment_contract.json")
    ):
        raise RuntimeError("persisted full selected-pack preflight is missing or invalid")
    selected = receipt.get("selected_pack_source_equivalence")
    if (
        not isinstance(selected, dict)
        or set(selected) != {"univ1", "virchow2_cls"}
        or any(
            not isinstance(value, dict)
            or int(value.get("n_slides", 0)) <= 0
            or len(str(value.get("selected_content_sha256", ""))) != 64
            for value in selected.values()
        )
    ):
        raise RuntimeError("persisted selected-pack equivalence receipt is incomplete")
    contract = _verify_contract(root)
    if receipt.get("packs") != contract.get("packs"):
        raise RuntimeError("preflight packed-store identities differ from the contract")
    return receipt


def _lock_path(root: Path, name: str) -> Path:
    path = root / "state" / "locks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _exclusive_lock(path: Path):
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "a+") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"another launcher holds {path}") from error
            yield
    finally:
        pass


def _run_job(root: Path, job: Job, *, num_workers: int) -> tuple[Job, int, Path]:
    attempt = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}_{uuid.uuid4().hex}"
    log = root / "logs" / job.component / f"{job.key}__{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = train_command(root, job, num_workers, attempt_id=attempt)
    with _exclusive_lock(_lock_path(root, f"chain__{job.key}.lock")):
        with log.open("x", encoding="utf-8") as stream:
            code = subprocess.run(
                command,
                cwd=root / "source_snapshot",
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        if code == 0:
            _ensure_job_receipt(root, job)
    return job, code, log


def _filter_jobs(args: argparse.Namespace) -> list[Job]:
    return [
        job
        for job in job_inventory()
        if (args.component is None or job.component == args.component)
        and (args.task is None or job.task == args.task)
        and (args.model_seed is None or job.model_seed == args.model_seed)
        and (args.draw_seed is None or job.draw_seed == args.draw_seed)
    ]


def cmd_train(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    if not 1 <= args.jobs <= MAX_JOBS:
        raise ValueError(f"--jobs must be 1..{MAX_JOBS}")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    _verify_contract(root)
    if args.apply:
        _verify_preflight_receipt(root)
    if (root / "receipts" / "extension_complete.json").exists():
        raise RuntimeError("extension is sealed; refusing additional training")
    selected = _filter_jobs(args)
    pending: list[Job] = []
    for job in selected:
        completion = run_dir(root, job) / "training_completion.json"
        if completion.is_file():
            _ensure_job_receipt(root, job) if args.apply else validate_job_chain(root, job)
        else:
            pending.append(job)
    print(f"{len(pending)}/{len(selected)} pending chains; jobs={args.jobs}; workers={args.num_workers}")
    if args.dry_run or not args.apply:
        for job in pending:
            print(shlex.join(train_command(root, job, args.num_workers, attempt_id=f"dryrun__{job.key}")))
        print("DRY RUN — no trainer process was launched; pass --apply to execute")
        return
    failures = []
    with (
        _exclusive_lock(_lock_path(root, "campaign_train.lock")),
        ThreadPoolExecutor(max_workers=args.jobs) as pool,
    ):
        futures = {pool.submit(_run_job, root, job, num_workers=args.num_workers): job for job in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            job, code, log = future.result()
            print(f"[{index}/{len(pending)}] {job.key}: {'PASS' if code == 0 else f'FAIL {code}'} ({log})", flush=True)
            if code:
                failures.append({"job": job.key, "exit_code": code, "log": str(log)})
    if failures:
        raise SystemExit(f"failed jobs retained for audit: {failures}")


def cmd_train_one(args: argparse.Namespace) -> None:
    """Run and certify exactly one job for the study-wide max-six scheduler."""

    root = validate_output_root(Path(args.output_root), must_exist=True)
    job = job_from_key(args.job_key)
    _verify_contract(root)
    _verify_preflight_receipt(root)
    if (root / "receipts" / "extension_complete.json").exists():
        raise RuntimeError("extension is sealed; refusing additional training")
    completion = run_dir(root, job) / "training_completion.json"
    if completion.is_file():
        _ensure_job_receipt(root, job)
        print(f"PASS — already complete and revalidated: {job.key}")
        return
    if not args.external_scheduler:
        raise ValueError(
            "train-one is reserved for the study-wide scheduler; pass --external-scheduler "
            "or use `train --jobs 6`"
        )
    completed, code, log = _run_job(root, job, num_workers=args.num_workers)
    if code != 0:
        raise SystemExit(f"{completed.key} failed with rc={code}; see {log}")
    _validate_job_receipt(root, job)
    print(f"PASS — rc=0 and native artifacts certified: {job.key}")


def _manifest_for_job(root: Path, job: Job) -> pd.DataFrame:
    if job.component in {"fixed", "e3v"}:
        return pd.read_csv(fixed_manifest(root, job.task), low_memory=False)
    if job.component == "repeated":
        assert job.draw_seed is not None
        return pd.read_csv(repeated_manifest(root, job.task, job.draw_seed), low_memory=False)
    return pd.read_csv(e1v_manifest(root), low_memory=False)


def _ensemble(root: Path, jobs: list[Job], manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    seed_roster = tuple(job.model_seed for job in jobs)
    if seed_roster not in {OLD_SEEDS, ALL_SEEDS}:
        raise ValueError("ensemble requires ordered seeds 42..44 or 42..46 exactly")
    frames: list[pd.DataFrame] = []
    per_seed: dict[str, float] = {}
    for job in jobs:
        path = resolved_run_dir(root, job) / "oof_predictions.parquet"
        patient = evaluate.to_patient_level(pd.read_parquet(path), manifest).sort_values("patient_id").reset_index(drop=True)
        frames.append(patient)
        per_seed[str(job.model_seed)] = float(fixed_stats._task_point(patient))
    reference = frames[0]
    invariant = [column for column in ("patient_id", "label", "cohort", "subcohort", "k_fold") if column in reference]
    for frame in frames[1:]:
        pd.testing.assert_frame_equal(reference[invariant], frame[invariant], check_dtype=False)
    out = reference.copy()
    out["mean_logit"] = np.mean(np.stack([frame["mean_logit"].to_numpy(dtype=float) for frame in frames]), axis=0)
    return out, per_seed


def _jobs_for(
    component: Component,
    task: str,
    draw: int | None = None,
    *,
    seeds: Iterable[int] = ALL_SEEDS,
) -> list[Job]:
    return [Job(component, task, int(seed), draw) for seed in seeds]


def _assert_anchor(label: str, observed: float, expected: float) -> float:
    if not np.isclose(observed, expected, atol=1e-12, rtol=0):
        raise RuntimeError(
            f"three-seed loader anchor failed for {label}: {observed:.15f} != {expected:.15f}"
        )
    return float(observed)


def _e0_ensemble(
    e0_root: Path,
    manifest: pd.DataFrame,
    *,
    seeds: Iterable[int],
) -> tuple[pd.DataFrame, dict[str, float]]:
    frames = []
    per_seed: dict[str, float] = {}
    selected = tuple(int(seed) for seed in seeds)
    for seed in selected:
        path = (
            aim1_extension.resolve_run_dir(e0_root, OLD_E0_ROOT, seed)
            / "oof_predictions.parquet"
        )
        patient = evaluate.to_patient_level(pd.read_parquet(path), manifest).sort_values(
            "patient_id"
        ).reset_index(drop=True)
        frames.append(patient)
        per_seed[str(seed)] = fixed_stats._task_point(patient)
    for frame in frames[1:]:
        pd.testing.assert_frame_equal(
            frames[0][["patient_id", "label"]],
            frame[["patient_id", "label"]],
            check_dtype=False,
        )
    ensemble = frames[0].copy()
    ensemble["mean_logit"] = np.mean(
        np.stack([frame["mean_logit"].to_numpy(dtype=float) for frame in frames]), axis=0
    )
    return ensemble, per_seed


def _three_seed_replay(root: Path, e0_root: Path) -> dict[str, Any]:
    """Replay every old headline point before opening the five-seed analysis."""

    sealed_fixed = _read_json(OLD_FIXED_ANALYSIS)
    sealed_repeated = _read_json(OLD_REPEATED_ANALYSIS)
    sealed_e3v = _read_json(OLD_E3V_ANALYSIS)
    sealed_e1v = _read_json(OLD_E1V_ANALYSIS)
    fixed_points: dict[str, float] = {}
    for task in FIXED_TASKS:
        manifest = _manifest_for_job(root, Job("fixed", task, OLD_SEEDS[0]))
        ensemble, _ = _ensemble(
            root, _jobs_for("fixed", task, seeds=OLD_SEEDS), manifest
        )
        fixed_points[task] = _assert_anchor(
            f"fixed/{task}",
            fixed_stats._task_point(ensemble),
            float(sealed_fixed["tasks"][task]["auroc"]),
        )
    e0_manifest = pd.read_csv(e1v_manifest(root), low_memory=False)
    e0_ensemble, _ = _e0_ensemble(e0_root, e0_manifest, seeds=OLD_SEEDS)
    fixed_points["gene"] = _assert_anchor(
        "fixed/gene",
        fixed_stats._task_point(e0_ensemble),
        float(sealed_fixed["tasks"]["gene"]["auroc"]),
    )
    repeated_points: dict[str, dict[str, float]] = {}
    for fine_task, control_task in FIXED_PAIRS:
        repeated_points[fine_task] = {}
        for draw in WT_DRAW_SEEDS:
            manifest = _manifest_for_job(
                root, Job("repeated", control_task, OLD_SEEDS[0], draw)
            )
            control, _ = _ensemble(
                root,
                _jobs_for("repeated", control_task, draw, seeds=OLD_SEEDS),
                manifest,
            )
            observed = fixed_stats._task_point(control)
            expected = float(
                sealed_repeated["rungs"][fine_task]["draws"][str(draw)]["control"][
                    "auroc"
                ]
            )
            repeated_points[fine_task][str(draw)] = _assert_anchor(
                f"repeated/{fine_task}/wt{draw}", observed, expected
            )
    e3v_points: dict[str, dict[str, float]] = {}
    for fine_task, control_task in legacy_e3v.PAIRS:
        e3v_points[fine_task] = {}
        for kind, task in (("fine", fine_task), ("control", control_task)):
            manifest = _manifest_for_job(root, Job("e3v", task, OLD_SEEDS[0]))
            ensemble, _ = _ensemble(
                root, _jobs_for("e3v", task, seeds=OLD_SEEDS), manifest
            )
            e3v_points[fine_task][kind] = _assert_anchor(
                f"e3v/{fine_task}/{kind}",
                fixed_stats._task_point(ensemble),
                float(sealed_e3v["rungs"][fine_task][kind]["auroc"]),
            )
    e1v_manifest_frame = pd.read_csv(e1v_manifest(root), low_memory=False)
    e1v_ensemble, _ = _ensemble(
        root, _jobs_for("e1v", "gene", seeds=OLD_SEEDS), e1v_manifest_frame
    )
    e1v_point = _assert_anchor(
        "e1v/gene",
        fixed_stats._task_point(e1v_ensemble),
        float(sealed_e1v["three_seed_ensemble_A"]["auroc"]),
    )
    return {
        "status": "PASS",
        "seeds": list(OLD_SEEDS),
        "source_artifacts": _old_analysis_inventory(),
        "fixed_points": fixed_points,
        "repeated_control_points": repeated_points,
        "e3v_points": e3v_points,
        "e1v_gene_point": e1v_point,
        "tolerance": 1e-12,
    }


def _summary(point: float, values: np.ndarray, *, family_size: int = 5) -> dict[str, Any]:
    one_tail = 0.05 / family_size
    return {
        "estimate": float(point),
        "ci95_two_sided": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "primary_fwer_one_sided": {
            "lower": float(np.quantile(values, one_tail)),
            "upper": float(np.quantile(values, 1 - one_tail)),
            "confidence": float(1 - one_tail),
        },
    }


def _gate(fine: dict[str, Any], control: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    fb = fine["primary_fwer_one_sided"]
    cb = control["primary_fwer_one_sided"]
    db = delta["primary_fwer_one_sided"]
    conditions = {
        "fine_upper_lt_0p60": fb["upper"] < CEILING_BOUND,
        "control_lower_gt_0p50": cb["lower"] > CHANCE,
        "delta_lower_gt_zero": db["lower"] > 0,
    }
    ceiling = all(conditions.values())
    signal = fb["lower"] > CHANCE
    if cb["lower"] <= CHANCE:
        verdict = "UNDERPOWERED"
    elif ceiling and signal:
        verdict = "CEILING_WITH_RESIDUAL_SIGNAL"
    elif ceiling:
        verdict = "CEILING"
    elif signal:
        verdict = "FINE_RESOLUTION_EVIDENCE"
    else:
        verdict = "INCONCLUSIVE"
    return {"conditions": conditions, "ceiling": ceiling, "verdict": verdict}


def _fixed_analysis(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result: dict[str, Any] = {
        "rungs": {},
        "protocol": {
            "n_bootstrap": FIXED_BOOTSTRAPS,
            "base_seed": BOOTSTRAP_SEED,
            "pair_seeds": FIXED_PAIR_BOOTSTRAP_SEEDS,
            "derivation": "sealed _stable_seed(base, 'primary_pair', fine_task)",
        },
    }
    arrays: dict[str, np.ndarray] = {}
    for fine_task, control_task in FIXED_PAIRS:
        fine_manifest_frame = _manifest_for_job(root, Job("fixed", fine_task, 42))
        control_manifest_frame = _manifest_for_job(root, Job("fixed", control_task, 42))
        fine, fine_per_seed = _ensemble(root, _jobs_for("fixed", fine_task), fine_manifest_frame)
        control, control_per_seed = _ensemble(root, _jobs_for("fixed", control_task), control_manifest_frame)
        seed = FIXED_PAIR_BOOTSTRAP_SEEDS[fine_task]
        boot = fixed_stats.partially_paired_bootstrap(fine, control, n_bootstrap=FIXED_BOOTSTRAPS, seed=seed)
        fine_summary = _summary(fixed_stats._task_point(fine), boot["fine_auc"])
        control_summary = _summary(fixed_stats._task_point(control), boot["control_auc"])
        delta_point = control_summary["estimate"] - fine_summary["estimate"]
        delta_summary = _summary(delta_point, boot["delta"])
        prefix = f"fixed__{fine_task}"
        arrays[f"{prefix}__fine"] = boot["fine_auc"]
        arrays[f"{prefix}__control"] = boot["control_auc"]
        arrays[f"{prefix}__delta"] = boot["delta"]
        result["rungs"][fine_task] = {
            "control_task": control_task,
            "fine_per_seed": fine_per_seed,
            "control_per_seed": control_per_seed,
            "fine": fine_summary,
            "control": control_summary,
            "delta_control_minus_fine": delta_summary,
            "gate": _gate(fine_summary, control_summary, delta_summary),
        }
    return result, arrays


def _repeated_analysis(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result: dict[str, Any] = {
        "rungs": {},
        "protocol": {"n_bootstrap": REPEATED_BOOTSTRAPS, "seed": REPEATED_BOOTSTRAP_SEED, "strata": "subcohort x k_fold"},
    }
    arrays: dict[str, np.ndarray] = {}
    for task_index, (fine_task, control_task) in enumerate(FIXED_PAIRS):
        fine_manifest_frame = _manifest_for_job(root, Job("fixed", fine_task, 42))
        fine, fine_per_seed = _ensemble(root, _jobs_for("fixed", fine_task), fine_manifest_frame)
        draws: dict[str, Any] = {}
        verdicts = []
        common_seed = int(np.random.SeedSequence([REPEATED_BOOTSTRAP_SEED, task_index]).generate_state(1)[0])
        for draw in WT_DRAW_SEEDS:
            manifest = _manifest_for_job(root, Job("repeated", control_task, 42, draw))
            control, control_per_seed = _ensemble(root, _jobs_for("repeated", control_task, draw), manifest)
            boot = repeated_stats.partial_paired_bootstrap(fine, control, n_bootstrap=REPEATED_BOOTSTRAPS, seed=common_seed)
            fine_summary = _summary(boot["point"]["fine"], boot["fine_values"])
            control_summary = _summary(boot["point"]["control"], boot["control_values"])
            delta_summary = _summary(boot["point"]["control"] - boot["point"]["fine"], boot["delta_values"])
            gate = _gate(fine_summary, control_summary, delta_summary)
            verdicts.append(gate["verdict"])
            prefix = f"repeated__{fine_task}__wt{draw}"
            arrays[f"{prefix}__fine"] = boot["fine_values"]
            arrays[f"{prefix}__control"] = boot["control_values"]
            arrays[f"{prefix}__delta"] = boot["delta_values"]
            draws[str(draw)] = {
                "control_per_seed": control_per_seed,
                "fine": fine_summary,
                "control": control_summary,
                "delta_control_minus_fine": delta_summary,
                "gate": gate,
            }
        result["rungs"][fine_task] = {
            "control_task": control_task,
            "fine_per_seed": fine_per_seed,
            "draws": draws,
            "consensus_verdict": repeated_stats.consensus_verdict(verdicts),
        }
    return result, arrays


def _e3v_analysis(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result: dict[str, Any] = {
        "rungs": {},
        "protocol": {
            "n_bootstrap": FIXED_BOOTSTRAPS,
            "seed_per_rung": E3V_PAIR_BOOTSTRAP_SEED,
            "derivation": "sealed E3v base seed reused for every rung",
        },
    }
    arrays: dict[str, np.ndarray] = {}
    for fine_task, control_task in legacy_e3v.PAIRS:
        fine_manifest_frame = _manifest_for_job(root, Job("e3v", fine_task, 42))
        control_manifest_frame = _manifest_for_job(root, Job("e3v", control_task, 42))
        fine, fine_per_seed = _ensemble(root, _jobs_for("e3v", fine_task), fine_manifest_frame)
        control, control_per_seed = _ensemble(root, _jobs_for("e3v", control_task), control_manifest_frame)
        # Preserve the sealed E3v derivation exactly: the same declared base
        # stream starts each rung.  Only the model-seed roster changes.
        seed = E3V_PAIR_BOOTSTRAP_SEED
        boot = fixed_stats.partially_paired_bootstrap(fine, control, n_bootstrap=FIXED_BOOTSTRAPS, seed=seed)
        fine_summary = _summary(fixed_stats._task_point(fine), boot["fine_auc"])
        control_summary = _summary(fixed_stats._task_point(control), boot["control_auc"])
        delta_summary = _summary(control_summary["estimate"] - fine_summary["estimate"], boot["delta"])
        prefix = f"e3v__{fine_task}"
        arrays[f"{prefix}__fine"] = boot["fine_auc"]
        arrays[f"{prefix}__control"] = boot["control_auc"]
        arrays[f"{prefix}__delta"] = boot["delta"]
        result["rungs"][fine_task] = {
            "control_task": control_task,
            "fine_per_seed": fine_per_seed,
            "control_per_seed": control_per_seed,
            "fine": fine_summary,
            "control": control_summary,
            "delta_control_minus_fine": delta_summary,
            "gate": _gate(fine_summary, control_summary, delta_summary),
        }
    result["conclusion_rule"] = "any opened rung revises the encoder-robust ceiling conclusion"
    return result, arrays


def _e1v_analysis(root: Path) -> dict[str, Any]:
    manifest = pd.read_csv(e1v_manifest(root), low_memory=False)
    ensemble, per_seed_a = _ensemble(root, _jobs_for("e1v", "gene"), manifest)
    mask_d = legacy_e1v._set_d_mask(ensemble)
    bootstrap_a = evaluate.bootstrap_auroc(
        ensemble.assign(prob_raw=ensemble["mean_logit"]),
        "mean_logit",
        n_bootstrap=E1V_BOOTSTRAPS,
        seed=BOOTSTRAP_SEED,
    )
    return {
        "per_seed_auroc_A": per_seed_a,
        "five_seed_ensemble_A": bootstrap_a,
        "five_seed_ensemble_D_auroc": evaluate.patient_auroc(ensemble[mask_d], "mean_logit"),
        "role": "five-seed Virchow2-CLS gene reference for E3v",
    }


def _e0_analysis(e0_root: Path, manifest: pd.DataFrame) -> dict[str, Any]:
    ensemble, per_seed = _e0_ensemble(e0_root, manifest, seeds=ALL_SEEDS)
    values = fixed_stats.task_bootstrap(
        ensemble,
        n_bootstrap=FIXED_BOOTSTRAPS,
        seed=GENE_BOOTSTRAP_SEED,
    )
    return {"per_seed": per_seed, "five_seed": _summary(fixed_stats._task_point(ensemble), values)}


def _recompute_analysis_payload(
    root: Path,
    e0_root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Deterministically rebuild the complete result semantics and draw arrays."""

    three_seed_replay = _three_seed_replay(root, e0_root)
    fixed, fixed_arrays = _fixed_analysis(root)
    repeated, repeated_arrays = _repeated_analysis(root)
    e3v, e3v_arrays = _e3v_analysis(root)
    e1v = _e1v_analysis(root)
    e0 = _e0_analysis(e0_root, pd.read_csv(e1v_manifest(root), low_memory=False))
    arrays: dict[str, np.ndarray] = {}
    for block in (fixed_arrays, repeated_arrays, e3v_arrays):
        overlap = sorted(set(arrays).intersection(block))
        if overlap:
            raise RuntimeError(f"duplicate bootstrap array keys: {overlap}")
        arrays.update(block)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "estimand": "five-seed patient native-logit ensemble; seeds are not inference units",
        "three_seed_replay_before_extension": three_seed_replay,
        "e0_gene_reference": e0,
        "fixed_univ1": fixed,
        "repeated_univ1": repeated,
        "e3v_virchow2_cls": e3v,
        "e1v_virchow2_cls_gene_reference": e1v,
    }
    return report, arrays


def _verify_analysis_replay(
    report_path: Path,
    distributions_path: Path,
    expected_report: dict[str, Any],
    expected_arrays: dict[str, np.ndarray],
) -> None:
    """Compare a sealed analysis with a fresh deterministic semantic replay."""

    stored_report = _read_json(report_path)
    created_utc = stored_report.pop("created_utc", None)
    if not isinstance(created_utc, str) or not created_utc:
        raise RuntimeError("sealed five-seed result has no creation timestamp")
    stored_semantics = json.dumps(
        stored_report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    expected_semantics = json.dumps(
        expected_report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if stored_semantics != expected_semantics:
        raise RuntimeError("sealed five-seed result semantic replay drifted")

    with np.load(distributions_path, allow_pickle=False) as stored_arrays:
        stored_keys = set(stored_arrays.files)
        expected_keys = set(expected_arrays)
        if stored_keys != expected_keys:
            missing = sorted(expected_keys - stored_keys)
            unexpected = sorted(stored_keys - expected_keys)
            raise RuntimeError(
                "sealed bootstrap array inventory drifted: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for key in sorted(expected_keys):
            stored = stored_arrays[key]
            expected = np.asarray(expected_arrays[key])
            if stored.shape != expected.shape:
                raise RuntimeError(
                    f"sealed bootstrap array shape drifted for {key}: "
                    f"{stored.shape} != {expected.shape}"
                )
            if stored.dtype != expected.dtype:
                raise RuntimeError(
                    f"sealed bootstrap array dtype drifted for {key}: "
                    f"{stored.dtype} != {expected.dtype}"
                )
            if not np.array_equal(stored, expected, equal_nan=True):
                raise RuntimeError(f"sealed bootstrap array values drifted for {key}")


def cmd_analyze(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    contract = _verify_contract(root)
    e0_root = Path(args.e0_root).resolve(strict=True)
    e0_dependency = _require_e0_dependency(e0_root)
    chain_inventory = []
    for job in job_inventory():
        receipt = _validate_job_receipt(root, job)
        chain_inventory.append(
            {
                "job": asdict(job),
                "receipt": identity(job_receipt_path(root, job)),
                "artifacts": receipt["artifacts"],
            }
        )
    _validate_adopted_inventory(contract)
    if not args.apply:
        print(
            f"DRY RUN — validated {len(chain_inventory)}/64 new chains and "
            "96 adopted chains; no analysis written"
        )
        return
    analysis = root / "analysis"
    if analysis.exists() or (root / "receipts" / "extension_complete.json").exists():
        raise FileExistsError("analysis/lineage completion already exists")
    report, bootstrap_arrays = _recompute_analysis_payload(root, e0_root)
    report = {**report, "created_utc": datetime.now(timezone.utc).isoformat()}
    analysis.mkdir(parents=False, exist_ok=False)
    report_path = analysis / "five_seed_results.json"
    _write_json_once(report_path, report)
    distributions_path = analysis / "bootstrap_distributions.npz"
    with distributions_path.open("xb") as stream:
        np.savez_compressed(stream, **bootstrap_arrays)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "contract": identity(root / "inputs" / "experiment_contract.json"),
        "e0_dependency": e0_dependency["receipt"],
        "new_chains": chain_inventory,
        "report": identity(report_path),
        "bootstrap_distributions": identity(distributions_path),
    }
    _write_json_once(analysis / "analysis_audit.json", audit)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "output_root": str(root),
        "counts": job_counts(),
        "artifacts": {
            name: identity(analysis / name)
            for name in ("five_seed_results.json", "bootstrap_distributions.npz", "analysis_audit.json")
        },
    }
    _write_json_once(root / "receipts" / "extension_complete.json", complete)
    print(f"PASS — sealed five-seed Aim-3 results at {root}")


def cmd_verify(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    # Replay executes this live controller and its live imported analysis
    # modules, so all material live-source hashes must match the contract.
    contract = _verify_contract(root)
    e0_root = Path(args.e0_root).resolve(strict=True)
    e0_dependency = _require_e0_dependency(e0_root)
    complete_path = root / "receipts" / "extension_complete.json"
    complete = _read_json(complete_path)
    if complete.get("schema_version") != SCHEMA_VERSION or complete.get("status") != "completed":
        raise RuntimeError("invalid extension completion receipt")
    if complete.get("counts") != job_counts():
        raise RuntimeError("completion accounting drifted")
    artifact_names = {
        "five_seed_results.json",
        "bootstrap_distributions.npz",
        "analysis_audit.json",
    }
    if set(complete.get("artifacts", {})) != artifact_names:
        raise RuntimeError("completion analysis artifact inventory drifted")
    for name, record in complete["artifacts"].items():
        expected = root / "analysis" / name
        if Path(record.get("path", "")).resolve() != expected or identity(expected) != record:
            raise RuntimeError(f"sealed analysis artifact changed: {name}")
    audit = _read_json(root / "analysis" / "analysis_audit.json")
    if audit.get("schema_version") != SCHEMA_VERSION or audit.get("status") != "PASS":
        raise RuntimeError("invalid analysis audit")
    if audit.get("contract") != identity(root / "inputs" / "experiment_contract.json"):
        raise RuntimeError("analysis audit contract changed")
    if audit.get("e0_dependency") != e0_dependency["receipt"]:
        raise RuntimeError("analysis audit Aim-1 E0 dependency changed")
    report_path = root / "analysis" / "five_seed_results.json"
    distributions_path = root / "analysis" / "bootstrap_distributions.npz"
    if audit.get("report") != identity(report_path):
        raise RuntimeError("analysis audit report identity changed")
    if audit.get("bootstrap_distributions") != identity(distributions_path):
        raise RuntimeError("analysis audit bootstrap identity changed")
    chain_inventory = []
    for job in job_inventory():
        receipt = _validate_job_receipt(root, job)
        chain_inventory.append(
            {
                "job": asdict(job),
                "receipt": identity(job_receipt_path(root, job)),
                "artifacts": receipt["artifacts"],
            }
        )
    if audit.get("new_chains") != chain_inventory:
        raise RuntimeError("analysis audit new-chain inventory changed")
    _validate_adopted_inventory(contract)
    expected_report, expected_arrays = _recompute_analysis_payload(root, e0_root)
    _verify_analysis_replay(
        report_path,
        distributions_path,
        expected_report,
        expected_arrays,
    )
    print(
        "PASS — five-seed Aim-3 extension verified: 64 new chains, 320 OOF folds, "
        "50 p75 refits, 370 actual new MIL fits"
    )


def cmd_status(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    _verify_contract(root, verify_live_source=False)
    complete, certified, partial, absent = [], [], [], []
    for job in job_inventory():
        directory = run_dir(root, job)
        if (directory / "training_completion.json").is_file():
            complete.append(job)
            if job_receipt_path(root, job).is_file():
                certified.append(job)
        elif directory.exists():
            partial.append(job)
        else:
            absent.append(job)
    print(
        json.dumps(
            {
                "complete_chains": len(complete),
                "certified_job_receipts": len(certified),
                "partial_chains": len(partial),
                "absent_chains": len(absent),
                "complete_oof_fold_equivalents": len(complete) * N_FOLDS,
                "sealed": (root / "receipts" / "extension_complete.json").is_file(),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--output-root", default=str(DEFAULT_ROOT))
    plan.add_argument("--e0-root", default=str(DEFAULT_E0_ROOT))
    plan.set_defaults(func=cmd_plan)
    jobs = commands.add_parser("jobs")
    jobs.add_argument("--output-root", default=str(DEFAULT_ROOT))
    jobs.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    jobs.set_defaults(func=cmd_jobs)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--output-root", default=str(DEFAULT_ROOT))
    manifest.add_argument("--e0-root", default=str(DEFAULT_E0_ROOT))
    manifest.add_argument("--apply", action="store_true")
    manifest.set_defaults(func=cmd_manifest)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--output-root", default=str(DEFAULT_ROOT))
    preflight.add_argument("--e0-root", default=str(DEFAULT_E0_ROOT))
    preflight.add_argument("--full-pack-check", action="store_true")
    preflight.add_argument("--apply", action="store_true")
    preflight.set_defaults(func=cmd_preflight)
    train = commands.add_parser("train")
    train.add_argument("--output-root", default=str(DEFAULT_ROOT))
    train.add_argument("--e0-root", default=str(DEFAULT_E0_ROOT))
    train.add_argument("--component", choices=("fixed", "repeated", "e3v", "e1v"))
    train.add_argument("--task")
    train.add_argument("--draw-seed", type=int, choices=WT_DRAW_SEEDS)
    train.add_argument("--model-seed", type=int, choices=NEW_SEEDS)
    train.add_argument("--jobs", type=int, default=MAX_JOBS)
    train.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--apply", action="store_true")
    train.set_defaults(func=cmd_train)
    train_one = commands.add_parser("train-one")
    train_one.add_argument("--output-root", default=str(DEFAULT_ROOT))
    train_one.add_argument("--job-key", required=True)
    train_one.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    train_one.add_argument("--external-scheduler", action="store_true")
    train_one.set_defaults(func=cmd_train_one)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--output-root", default=str(DEFAULT_ROOT))
    analyze.add_argument("--e0-root", default=str(DEFAULT_E0_ROOT))
    analyze.add_argument("--apply", action="store_true")
    analyze.set_defaults(func=cmd_analyze)
    verify = commands.add_parser("verify")
    verify.add_argument("--output-root", default=str(DEFAULT_ROOT))
    verify.add_argument("--e0-root", default=str(DEFAULT_E0_ROOT))
    verify.set_defaults(func=cmd_verify)
    status = commands.add_parser("status")
    status.add_argument("--output-root", default=str(DEFAULT_ROOT))
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
