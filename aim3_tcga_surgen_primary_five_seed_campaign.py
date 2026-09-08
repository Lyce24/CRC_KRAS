#!/usr/bin/env python3
"""Governed Aim-3 ladder on TCGA + SurGen primary source patients only.

This additive campaign is deliberately bound to the sealed
``aim1_primary_cohort_oof_5seed`` lineage, specifically its
``tcga_surgen_primary`` arm.  It does not read CPTAC, Orion, RIH, metastatic
SurGen, or any other external-validation cohort.

The exact training inventory is:

* five fine molecular rungs;
* five fixed, size/fold-matched WT controls;
* the same five WT controls under three predeclared repeat draws;
* five model seeds (42..46), five OOF folds, and one p75 refit per chain.

That is 25 task variants x 5 seeds = 125 chains, 625 OOF folds, 125 p75
refits, and 750 physical MIL fits.  At most six chains may execute at once.
Headline analysis averages native slide logits within patient and seed, then
averages the five patient logits.  Fixed and repeated comparisons use a
partially-paired bootstrap; repeated ceiling claims require all three draws.

Typical workflow (``train`` is a dry run unless ``--apply`` is supplied)::

    python aim3_tcga_surgen_primary_five_seed_campaign.py plan
    python aim3_tcga_surgen_primary_five_seed_campaign.py prepare --apply
    python aim3_tcga_surgen_primary_five_seed_campaign.py preflight --apply
    python aim3_tcga_surgen_primary_five_seed_campaign.py train --jobs 6
    python aim3_tcga_surgen_primary_five_seed_campaign.py train --jobs 6 --apply
    python aim3_tcga_surgen_primary_five_seed_campaign.py validate --seal
    python aim3_tcga_surgen_primary_five_seed_campaign.py analyze --apply
    python aim3_tcga_surgen_primary_five_seed_campaign.py verify

Preparation and every published receipt are exclusive-create.  A partial
production root is evidence and is never repaired in place; use a fresh,
versioned root instead.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import aim3_ladders_five_seed_extension as ladder5  # noqa: E402
import aim3_fixed_control_analysis as fixed_stats  # noqa: E402
import aim3_repeated_control_campaign as repeated_stats  # noqa: E402
import aim3_resolution_ladder as legacy_ladder  # noqa: E402
from oceanpath.aim1 import evaluate, paths  # noqa: E402
from oceanpath.splitting.core import derive_subset_splits  # noqa: E402

SCHEMA_VERSION = 1
CAMPAIGN = "aim3_tcga_surgen_primary_univ1_5seed"
SOURCE_CAMPAIGN = "aim1_primary_cohort_oof_5seed"
SOURCE_ARM = "tcga_surgen_primary"
SOURCE_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim1_primary_cohort_5seed_v1_20260824"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim3_tcga_surgen_primary_univ1_5seed_v2_20260827"
)
RERUNS_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns")

SOURCE_MANIFEST = SOURCE_ROOT / "inputs/manifests/tcga_surgen_primary.csv"
SOURCE_SPLITS = (
    SOURCE_ROOT
    / "inputs/splits/aim1_primary_tcga_surgen/aim1_balanced5"
)
SOURCE_CONTRACT = SOURCE_ROOT / "contract.json"
SOURCE_TRAINING_SEAL = SOURCE_ROOT / "receipts/training_complete.json"
SOURCE_EXPECTED_SHA256 = {
    "contract": "fcc5a6744da27b73c5771f71a05511e998d20b3b4d76ecdc8a9c8c4ce34f0973",
    "training_seal": "31cc8005abf32cb1c8958d0b4ecb5ed34e927bcd1665451869d62a913bb75413",
    "manifest": "d7087a23a84a294670080eb5090f83612f57b3376c7696ed2d0094fb9bcff8a5",
    "splits": "31046858b74a9e435ce7ceac263630e2966ab589c36a8693dcb0fc1060cfa9f4",
    "integrity": "4d21bd756482ceaaa5a6cb683d4797e49bc710fad0ed8a396e9a0368c27ee2a2",
    "summary": "5bc1ec9b2ce07f5af68079ef426e554e930943d23d8eb70a8a7a5358f40d9948",
}

MODEL_SEEDS = (42, 43, 44, 45, 46)
N_FOLDS = 5
FOLDS = tuple(range(N_FOLDS))
CAP = 8_192
FIXED_EPOCH_BUDGET = 12
MAX_JOBS = 6
DEFAULT_NUM_WORKERS = 4
FIXED_BOOTSTRAPS = 10_000
REPEATED_BOOTSTRAPS = 20_000
FIXED_BOOTSTRAP_SEED = 20260817
REPEATED_BOOTSTRAP_SEED = 20260826
CHANCE = 0.50
CEILING_BOUND = 0.60

FINE_TASKS = tuple(repeated_stats.FINE_TASKS)
CONTROL_FOR = dict(repeated_stats.CONTROL_FOR)
CONTROLS = tuple(CONTROL_FOR[task] for task in FINE_TASKS)
WT_DRAW_SEEDS = tuple(repeated_stats.WT_DRAW_SEEDS)
FIXED_WT_SEEDS = {
    fine: int(legacy_ladder.CONTROL_WT_SEED[CONTROL_FOR[fine]])
    for fine in FINE_TASKS
}
PAIRS = tuple((fine, CONTROL_FOR[fine]) for fine in FINE_TASKS)

EXPECTED_FINE_PATIENTS = {
    "codon": {"positive": 354, "negative": 147},
    "g12d_broad": {"positive": 154, "negative": 347},
    "allele1": {"positive": 154, "negative": 200},
    "allele2": {"positive": 102, "negative": 252},
    "g12c": {"positive": 46, "negative": 308},
}
EXPECTED_SOURCE = {
    "slides": 1_389,
    "patients": 1_239,
    "mutant_patients": 501,
    "wildtype_patients": 738,
}
ALLOWED_COHORTS = frozenset({"TCGA", "SurGen"})
ALLOWED_SUBCOHORTS = frozenset({"TCGA-COAD", "TCGA-READ", "SR386", "SR1482"})
FORBIDDEN_DEVELOPMENT_COHORTS = (
    "CPTAC",
    "Orion",
    "RIH",
    "SurGen-met",
)

TOTAL_TASK_VARIANTS = len(FINE_TASKS) + len(CONTROLS) + len(CONTROLS) * len(WT_DRAW_SEEDS)
TOTAL_CHAINS = TOTAL_TASK_VARIANTS * len(MODEL_SEEDS)
TOTAL_OOF_FITS = TOTAL_CHAINS * N_FOLDS
TOTAL_REFITS = TOTAL_CHAINS
TOTAL_PHYSICAL_FITS = TOTAL_OOF_FITS + TOTAL_REFITS
FINE_PHASE_CHAINS = len(FINE_TASKS) * len(MODEL_SEEDS)
FINE_PHASE_OOF_FITS = FINE_PHASE_CHAINS * N_FOLDS
FINE_PHASE_REFITS = FINE_PHASE_CHAINS
FINE_PHASE_PHYSICAL_FITS = FINE_PHASE_OOF_FITS + FINE_PHASE_REFITS
CONTROL_PHASE_CHAINS = TOTAL_CHAINS - FINE_PHASE_CHAINS

Kind = Literal["fine", "fixed", "repeated"]


class ContractError(RuntimeError):
    """Fail-closed violation of the governed campaign contract."""


@dataclass(frozen=True, order=True)
class Job:
    """One independent five-fold-plus-refit training chain."""

    kind: Kind
    fine_task: str
    model_seed: int
    draw_seed: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"fine", "fixed", "repeated"}:
            raise ValueError(f"unknown job kind: {self.kind}")
        if self.fine_task not in FINE_TASKS:
            raise ValueError(f"unknown fine task: {self.fine_task}")
        if self.model_seed not in MODEL_SEEDS:
            raise ValueError(f"model seed must be one of {MODEL_SEEDS}")
        if self.kind == "repeated":
            if self.draw_seed not in WT_DRAW_SEEDS:
                raise ValueError(f"repeated draw must be one of {WT_DRAW_SEEDS}")
        elif self.draw_seed is not None:
            raise ValueError("draw_seed is allowed only for repeated controls")

    @property
    def task(self) -> str:
        return self.fine_task if self.kind == "fine" else CONTROL_FOR[self.fine_task]

    @property
    def key(self) -> str:
        draw = f"__wt{self.draw_seed}" if self.draw_seed is not None else ""
        return f"{self.kind}__{self.task}{draw}__seed{self.model_seed}"

    @property
    def actual_fits(self) -> int:
        return N_FOLDS + 1


def job_inventory(*, seeds: Iterable[int] = MODEL_SEEDS) -> list[Job]:
    """Return all chains, with every fixed-ladder chain before repeats."""

    selected = tuple(int(seed) for seed in seeds)
    invalid = sorted(set(selected) - set(MODEL_SEEDS))
    if invalid or len(selected) != len(set(selected)):
        raise ValueError(f"invalid model-seed roster: {selected}")
    jobs = [Job("fine", fine, seed) for fine in FINE_TASKS for seed in selected]
    jobs += [Job("fixed", fine, seed) for fine in FINE_TASKS for seed in selected]
    jobs += [
        Job("repeated", fine, seed, draw)
        for fine in FINE_TASKS
        for draw in WT_DRAW_SEEDS
        for seed in selected
    ]
    return jobs


def fit_accounting(jobs: Iterable[Job] | None = None) -> dict[str, int]:
    values = list(job_inventory() if jobs is None else jobs)
    return {
        "task_variants": len(
            {(job.kind, job.fine_task, job.draw_seed) for job in values}
        ),
        "chains": len(values),
        "oof_folds": len(values) * N_FOLDS,
        "p75_refits": len(values),
        "physical_mil_fits": sum(job.actual_fits for job in values),
    }


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError(f"required regular artifact missing or symlinked: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON artifact must be an object: {path}")
    return value


def _write_bytes_once(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_text_once(path: Path, text: str) -> None:
    _write_bytes_once(path, text.encode("utf-8"))


def _write_json_once(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"
    _write_text_once(path, payload)


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    _write_bytes_once(path, buffer.getvalue())


def _write_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _write_bytes_once(path, buffer.getvalue())


def _copy_once(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_output_root(root: Path, *, must_exist: bool | None = None) -> Path:
    raw = Path(root).expanduser()
    if not raw.is_absolute():
        raise ContractError("--output-root must be absolute")
    lexical = raw.absolute()
    resolved = raw.resolve(strict=False)
    if lexical != resolved:
        raise ContractError(f"output root must not traverse symlinks: {raw}")
    cursor = raw
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise ContractError(f"output-root path contains a symlink: {cursor}")
        cursor = cursor.parent
    temporary = Path("/tmp").resolve()
    production = DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    frozen = SOURCE_ROOT.resolve(strict=False)
    if (
        resolved == frozen
        or _is_relative_to(resolved, frozen)
        or _is_relative_to(frozen, resolved)
    ):
        raise ContractError("output root overlaps the sealed Aim-1 source lineage")
    if resolved != production and not _is_relative_to(resolved, temporary):
        raise ContractError(
            f"production root must be exactly {production}; tests may use /tmp"
        )
    if resolved == production and not _is_relative_to(resolved, RERUNS_ROOT.resolve()):
        raise ContractError("production output must remain below outputs/aim1/reruns")
    if must_exist is True and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if must_exist is False and (resolved.exists() or resolved.is_symlink()):
        raise FileExistsError(resolved)
    return resolved


def contract_path(root: Path) -> Path:
    return root / "contract.json"


def preflight_path(root: Path) -> Path:
    return root / "receipts/preflight.json"


def scheduler_path(root: Path) -> Path:
    return root / "receipts/scheduler.json"


def fine_scheduler_path(root: Path) -> Path:
    return root / "receipts/scheduler_fine.json"


def controls_scheduler_path(root: Path) -> Path:
    return root / "receipts/scheduler_controls.json"


def training_seal_path(root: Path) -> Path:
    return root / "receipts/training_complete.json"


def fine_training_seal_path(root: Path) -> Path:
    return root / "receipts/training_complete_fine.json"


def source_snapshot_manifest(root: Path) -> Path:
    return root / "inputs/source_lineage/tcga_surgen_primary.csv"


def source_snapshot_splits(root: Path) -> Path:
    return root / "inputs/source_lineage/splits"


def variant_name(kind: Kind, fine: str, draw: int | None = None) -> str:
    if kind == "fine":
        return f"fine__{fine}"
    control = CONTROL_FOR[fine]
    if kind == "fixed":
        return f"fixed__{control}"
    if draw not in WT_DRAW_SEEDS:
        raise ValueError("repeated variants require a registered WT draw")
    return f"repeated__{control}__wt{draw}"


def manifest_path(root: Path, kind: Kind, fine: str, draw: int | None = None) -> Path:
    return root / "inputs/manifests" / f"{variant_name(kind, fine, draw)}.csv"


def split_dir(root: Path, kind: Kind, fine: str, draw: int | None = None) -> Path:
    return root / "inputs/splits" / variant_name(kind, fine, draw) / "aim1_balanced5"


def run_dir(root: Path, job: Job) -> Path:
    base = root / "train" / job.kind / job.task
    if job.draw_seed is not None:
        base = base / f"wt{job.draw_seed}"
    return base / f"seed{job.model_seed}"


def job_receipt_path(root: Path, job: Job) -> Path:
    return root / "receipts/jobs" / f"{job.key}.json"


def request_path(root: Path, job: Job) -> Path:
    return root / "requests/jobs" / f"{job.key}.json"


def log_path(root: Path, job: Job) -> Path:
    return root / "logs/jobs" / f"{job.key}.log"


def failure_path(root: Path, job: Job) -> Path:
    return root / "requests/failures" / f"{job.key}.json"


def material_source_files() -> tuple[str, ...]:
    # Root-level experiment controllers import one another transitively (for
    # example Aim-3 -> five-seed extension -> E1v/E3v).  Freeze every regular,
    # non-symlinked root ``*.py`` module rather than maintaining a brittle hand
    # list.  ``tools/**/*.py`` is likewise source-only and bounded: it excludes
    # manifests, outputs, credentials, feature payloads, and arbitrary repo
    # files while making lazy trainer imports reproducible.
    root_python = tuple(
        path.relative_to(REPO).as_posix()
        for path in sorted(REPO.glob("*.py"))
        if path.is_file() and not path.is_symlink()
    )
    tools_python = tuple(
        path.relative_to(REPO).as_posix()
        for path in sorted((REPO / "tools").rglob("*.py"))
        if path.is_file() and not path.is_symlink()
    )
    fixed = (
        "tests/test_aim3_tcga_surgen_primary_five_seed_campaign.py",
        "pyproject.toml",
        "uv.lock",
    )
    package = tuple(
        path.relative_to(REPO).as_posix()
        for path in sorted((REPO / "src/oceanpath").rglob("*.py"))
    )
    configs = tuple(
        path.relative_to(REPO).as_posix()
        for path in sorted((REPO / "configs").rglob("*.yaml"))
    )
    values = (*root_python, *tools_python, *fixed, *package, *configs)
    if len(values) != len(set(values)):
        raise AssertionError("duplicate material source path")
    for relative in values:
        path = REPO / relative
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"material source must be a regular non-symlink file: {path}")
    return values


def _patient_table(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "slide_id",
        "patient_id",
        "target_label",
        "kras",
        "kras_subvariant",
        "cohort",
        "subcohort",
        "specimen_role",
        "k_fold",
        *(f"val_fold_{fold}" for fold in FOLDS),
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ContractError(f"source manifest lacks columns: {missing}")
    if source["slide_id"].astype(str).duplicated().any():
        raise ContractError("source manifest contains duplicate slide_id")
    constant = [
        "target_label",
        "kras",
        "kras_subvariant",
        "cohort",
        "subcohort",
        "specimen_role",
        "k_fold",
        *(f"val_fold_{fold}" for fold in FOLDS),
    ]
    counts = source.groupby("patient_id", sort=False)[constant].nunique(dropna=False)
    if counts.gt(1).any(axis=1).any():
        raise ContractError("source metadata/folds vary within patient")
    patients = source.sort_values("slide_id").drop_duplicates("patient_id").copy()
    return patients


def validate_source_population(source: pd.DataFrame, *, production: bool) -> dict[str, Any]:
    patients = _patient_table(source)
    cohorts = frozenset(source["cohort"].astype(str).unique())
    subcohorts = frozenset(source["subcohort"].astype(str).unique())
    roles = frozenset(source["specimen_role"].astype(str).str.casefold().unique())
    if cohorts != ALLOWED_COHORTS:
        raise ContractError(
            f"source development cohorts must be exactly {sorted(ALLOWED_COHORTS)}; "
            f"observed={sorted(cohorts)}"
        )
    if subcohorts != ALLOWED_SUBCOHORTS:
        raise ContractError("source subcohort boundary drifted")
    if roles != {"primary"}:
        raise ContractError("source campaign permits primary specimens only")
    folds = pd.to_numeric(source["k_fold"], errors="raise").astype(int)
    if set(folds) != set(FOLDS):
        raise ContractError("source outer folds must be exactly 0..4")
    labels = pd.to_numeric(patients["target_label"], errors="raise").astype(int)
    kras = patients["kras"].astype(str)
    observed = {
        "slides": int(len(source)),
        "patients": int(len(patients)),
        "mutant_patients": int(kras.eq("mutant").sum()),
        "wildtype_patients": int(kras.eq("wild_type").sum()),
    }
    if not labels.isin([0, 1]).all():
        raise ContractError("source labels must be binary")
    if production and observed != EXPECTED_SOURCE:
        raise ContractError(
            f"sealed TCGA+SurGen-primary census drifted: {observed} != {EXPECTED_SOURCE}"
        )
    return {
        **observed,
        "cohorts": sorted(cohorts),
        "subcohorts": sorted(subcohorts),
        "specimen_roles": ["primary"],
        "external_development_cohorts": [],
    }


def fine_patient_labels(source: pd.DataFrame, fine: str) -> pd.DataFrame:
    """Population-specific molecular labels on the sealed source arm."""

    try:
        labels = repeated_stats.fine_patient_labels(source, fine).copy()
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    labels["patient_id"] = labels["patient_id"].astype(str)
    return labels


def draw_control_patients(
    source: pd.DataFrame,
    fine: str,
    draw_seed: int,
) -> pd.DataFrame:
    """Replace fine negatives by WT within subcohort x inherited outer fold."""

    patients = _patient_table(source).copy()
    fine_labels = fine_patient_labels(source, fine)
    positive = fine_labels[fine_labels["target_label"].eq(1)].copy()
    replaced = fine_labels[fine_labels["target_label"].eq(0)].copy()
    wild_type = patients[patients["kras"].astype(str).eq("wild_type")].copy()
    rng = np.random.default_rng(int(draw_seed))
    selected: list[pd.DataFrame] = []
    wanted = replaced.groupby(["subcohort", "k_fold"], sort=True).size()
    for (subcohort, fold), count in wanted.items():
        pool = wild_type[
            wild_type["subcohort"].astype(str).eq(str(subcohort))
            & pd.to_numeric(wild_type["k_fold"]).astype(int).eq(int(fold))
        ].sort_values("patient_id")
        if len(pool) < int(count):
            raise ContractError(
                f"{fine}/draw{draw_seed}: {subcohort}/fold{fold} needs {count} WT; "
                f"only {len(pool)} available"
            )
        indices = np.sort(rng.choice(len(pool), size=int(count), replace=False))
        selected.append(pool.iloc[indices])
    columns = ["patient_id", "cohort", "subcohort", "k_fold"]
    negative = pd.concat(selected, ignore_index=True)[columns].assign(target_label=0)
    result = pd.concat(
        [positive[columns + ["target_label"]], negative], ignore_index=True
    )
    if result["patient_id"].astype(str).duplicated().any():
        raise ContractError(f"{fine}/draw{draw_seed}: duplicate or overlapping control patient")
    observed = negative.groupby(["subcohort", "k_fold"], sort=True).size()
    if not observed.equals(wanted):
        raise ContractError(f"{fine}/draw{draw_seed}: WT stratum match failed")
    return result.sort_values(
        ["target_label", "subcohort", "k_fold", "patient_id"]
    ).reset_index(drop=True)


def _manifest_from_labels(source: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    selected = labels[["patient_id", "target_label"]].copy()
    selected["patient_id"] = selected["patient_id"].astype(str)
    work = source.copy()
    work["patient_id"] = work["patient_id"].astype(str)
    work = work.drop(columns=["target_label"]).merge(
        selected,
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )
    columns = list(source.columns)
    return work[columns].sort_values("slide_id").reset_index(drop=True)


def build_task_manifest(
    source: pd.DataFrame,
    kind: Kind,
    fine: str,
    draw_seed: int | None = None,
) -> pd.DataFrame:
    if fine not in FINE_TASKS:
        raise ValueError(fine)
    if kind == "fine":
        if draw_seed is not None:
            raise ValueError("fine tasks do not take a WT draw")
        labels = fine_patient_labels(source, fine)
    elif kind == "fixed":
        if draw_seed is not None:
            raise ValueError("fixed controls use their registered fixed seed")
        labels = draw_control_patients(source, fine, FIXED_WT_SEEDS[fine])
    elif kind == "repeated":
        if draw_seed not in WT_DRAW_SEEDS:
            raise ValueError(f"unregistered repeated WT draw: {draw_seed}")
        labels = draw_control_patients(source, fine, int(draw_seed))
    else:
        raise ValueError(kind)
    frame = _manifest_from_labels(source, labels)
    validate_task_manifest(frame, source, kind, fine, draw_seed)
    return frame


def validate_task_manifest(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    kind: Kind,
    fine: str,
    draw_seed: int | None,
) -> dict[str, Any]:
    source_patients = _patient_table(source)
    patients = _patient_table(frame)
    if not set(frame["slide_id"].astype(str)) < set(source["slide_id"].astype(str)):
        raise ContractError(f"{kind}/{fine}: task must be a strict source-manifest subset")
    if frozenset(frame["cohort"].astype(str).unique()) - ALLOWED_COHORTS:
        raise ContractError(f"{kind}/{fine}: external cohort entered task manifest")
    if set(frame["specimen_role"].astype(str).str.casefold()) != {"primary"}:
        raise ContractError(f"{kind}/{fine}: non-primary specimen entered task manifest")
    lineage_columns = [
        "slide_id",
        "patient_id",
        "k_fold",
        *(f"val_fold_{fold}" for fold in FOLDS),
    ]
    joined = frame[lineage_columns].merge(
        source[lineage_columns],
        on="slide_id",
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    if len(joined) != len(frame):
        raise ContractError(f"{kind}/{fine}: source-lineage join is incomplete")
    for column in ["patient_id", "k_fold", *(f"val_fold_{fold}" for fold in FOLDS)]:
        if not joined[column].astype(str).eq(joined[f"{column}_source"].astype(str)).all():
            raise ContractError(f"{kind}/{fine}: inherited {column} changed")
    labels = pd.to_numeric(patients["target_label"], errors="raise").astype(int)
    if set(labels) != {0, 1}:
        raise ContractError(f"{kind}/{fine}: both classes are required")
    fine_labels = fine_patient_labels(source, fine)
    positive_expected = set(
        fine_labels.loc[fine_labels["target_label"].eq(1), "patient_id"].astype(str)
    )
    positive_observed = set(
        patients.loc[labels.eq(1), "patient_id"].astype(str)
    )
    if positive_observed != positive_expected:
        raise ContractError(f"{kind}/{fine}: positive patient set changed")
    expected_counts = {
        "positive": int(fine_labels["target_label"].eq(1).sum()),
        "negative": int(fine_labels["target_label"].eq(0).sum()),
    }
    observed_counts = {
        "positive": int(labels.eq(1).sum()),
        "negative": int(labels.eq(0).sum()),
    }
    if observed_counts != expected_counts:
        raise ContractError(
            f"{kind}/{fine}: matched class sizes changed: {observed_counts} != {expected_counts}"
        )
    if kind != "fine":
        negative = patients.loc[labels.eq(0)]
        if not negative["kras"].astype(str).eq("wild_type").all():
            raise ContractError(f"{kind}/{fine}: control negatives are not all WT")
        fine_negative = fine_labels[fine_labels["target_label"].eq(0)]
        wanted = fine_negative.groupby(["subcohort", "k_fold"], sort=True).size()
        got = negative.groupby(["subcohort", "k_fold"], sort=True).size()
        if not got.equals(wanted):
            raise ContractError(f"{kind}/{fine}: WT matching strata changed")
    fold_cells: dict[str, Any] = {}
    small_class_folds = []
    for fold in FOLDS:
        test = pd.to_numeric(patients["k_fold"]).astype(int).eq(fold)
        val = pd.to_numeric(patients[f"val_fold_{fold}"]).astype(int).eq(1)
        train = ~(test | val)
        if (test & val).any():
            raise ContractError(f"{kind}/{fine}/fold{fold}: val overlaps test")
        cell: dict[str, Any] = {}
        for role, mask in (("train", train), ("val", val), ("test", test)):
            values = labels.loc[mask]
            if set(values) != {0, 1}:
                raise ContractError(f"{kind}/{fine}/fold{fold}/{role}: both classes required")
            cell[role] = {
                "patients": int(mask.sum()),
                "positive": int(values.eq(1).sum()),
                "negative": int(values.eq(0).sum()),
            }
        if cell["val"]["positive"] < int(paths.MIN_ES_VAL_POSITIVES):
            small_class_folds.append(fold)
        fold_cells[str(fold)] = cell
    if source_patients["patient_id"].astype(str).nunique() < len(patients):
        raise AssertionError("task patient count exceeds source")
    return {
        "slides": int(len(frame)),
        "patients": int(len(patients)),
        **observed_counts,
        "small_class_fixed_epoch_folds": small_class_folds,
        "fold_cells": fold_cells,
    }


def build_all_task_manifests(source: pd.DataFrame) -> dict[tuple[Kind, str, int | None], pd.DataFrame]:
    result: dict[tuple[Kind, str, int | None], pd.DataFrame] = {}
    for fine in FINE_TASKS:
        result[("fine", fine, None)] = build_task_manifest(source, "fine", fine)
        result[("fixed", fine, None)] = build_task_manifest(source, "fixed", fine)
        repeated_negative_rosters = []
        for draw in WT_DRAW_SEEDS:
            frame = build_task_manifest(source, "repeated", fine, draw)
            result[("repeated", fine, draw)] = frame
            patients = frame.sort_values("slide_id").drop_duplicates("patient_id")
            repeated_negative_rosters.append(
                frozenset(
                    patients.loc[patients["target_label"].eq(0), "patient_id"].astype(str)
                )
            )
        if len(set(repeated_negative_rosters)) != len(WT_DRAW_SEEDS):
            raise ContractError(f"{fine}: repeated WT seeds did not produce three distinct draws")
    return result


def _source_paths(source_root: Path) -> dict[str, Path]:
    return {
        "contract": source_root / "contract.json",
        "training_seal": source_root / "receipts/training_complete.json",
        "manifest": source_root / "inputs/manifests/tcga_surgen_primary.csv",
        "splits": source_root
        / "inputs/splits/aim1_primary_tcga_surgen/aim1_balanced5/splits.parquet",
        "integrity": source_root
        / "inputs/splits/aim1_primary_tcga_surgen/aim1_balanced5/.integrity_hash",
        "summary": source_root
        / "inputs/splits/aim1_primary_tcga_surgen/aim1_balanced5/summary.json",
    }


def verify_source_lineage(source_root: Path = SOURCE_ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(source_root).resolve(strict=True)
    paths_by_name = _source_paths(root)
    artifacts = {name: identity(path) for name, path in paths_by_name.items()}
    production = root == SOURCE_ROOT.resolve()
    if production:
        mismatch = {
            name: {"expected": expected, "observed": artifacts[name]["sha256"]}
            for name, expected in SOURCE_EXPECTED_SHA256.items()
            if artifacts[name]["sha256"] != expected
        }
        if mismatch:
            raise ContractError(f"sealed Aim-1 source identities drifted: {mismatch}")
    source_contract = _read_json(paths_by_name["contract"])
    source_seal = _read_json(paths_by_name["training_seal"])
    if (
        source_contract.get("campaign") != SOURCE_CAMPAIGN
        or source_contract.get("seeds") != list(MODEL_SEEDS)
        or source_contract.get("n_folds") != N_FOLDS
        or SOURCE_ARM not in source_contract.get("arms", [])
    ):
        raise ContractError("sealed Aim-1 source contract semantics drifted")
    if (
        source_seal.get("status") != "complete_and_certified"
        or source_seal.get("seeds") != list(MODEL_SEEDS)
        or SOURCE_ARM not in source_seal.get("arms", [])
        or source_seal.get("refit_count") != 0
    ):
        raise ContractError("sealed Aim-1 source training receipt is incomplete")
    source = pd.read_csv(paths_by_name["manifest"], low_memory=False)
    census = validate_source_population(source, production=production)
    split = pd.read_parquet(paths_by_name["splits"])
    columns = ["slide_id", "k_fold", *(f"val_fold_{fold}" for fold in FOLDS)]
    check = source[columns].merge(
        split[["slide_id", "fold", *(f"val_fold_{fold}" for fold in FOLDS)]],
        on="slide_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_manifest", "_split"),
    )
    if len(check) != len(source) or not pd.to_numeric(check["k_fold"]).astype(int).eq(
        pd.to_numeric(check["fold"]).astype(int)
    ).all():
        raise ContractError("sealed source manifest/split outer-fold lineage disagrees")
    for fold in FOLDS:
        column = f"val_fold_{fold}"
        if not pd.to_numeric(check[f"{column}_manifest"]).astype(int).eq(
            pd.to_numeric(check[f"{column}_split"]).astype(int)
        ).all():
            raise ContractError(f"sealed source manifest/split {column} disagrees")
    observed_fine = {}
    for fine in FINE_TASKS:
        labels = fine_patient_labels(source, fine)
        observed_fine[fine] = {
            "positive": int(labels["target_label"].eq(1).sum()),
            "negative": int(labels["target_label"].eq(0).sum()),
        }
    if production and observed_fine != EXPECTED_FINE_PATIENTS:
        raise ContractError(
            f"source fine-task census drifted: {observed_fine} != {EXPECTED_FINE_PATIENTS}"
        )
    return source, {
        "root": str(root),
        "campaign": SOURCE_CAMPAIGN,
        "arm": SOURCE_ARM,
        "artifacts": artifacts,
        "census": census,
        "fine_patient_census": observed_fine,
    }


def _validate_derived_split(
    directory: Path, frame: pd.DataFrame, source_split_path: Path
) -> dict[str, Any]:
    parquet = directory / "splits.parquet"
    integrity = directory / ".integrity_hash"
    summary = directory / "summary.json"
    derived = pd.read_parquet(parquet)
    source = pd.read_parquet(source_split_path)
    wanted = set(frame["slide_id"].astype(str))
    if (
        len(derived) != len(frame)
        or derived["slide_id"].astype(str).duplicated().any()
        or set(derived["slide_id"].astype(str)) != wanted
    ):
        raise ContractError(f"derived split roster differs from task manifest: {directory}")
    columns = ["slide_id", "fold", *(f"val_fold_{fold}" for fold in FOLDS)]
    joined = derived[columns].merge(
        source[columns], on="slide_id", validate="one_to_one", suffixes=("", "_source")
    )
    for column in columns[1:]:
        if not pd.to_numeric(joined[column]).astype(int).eq(
            pd.to_numeric(joined[f"{column}_source"]).astype(int)
        ).all():
            raise ContractError(f"derived split changed inherited {column}: {directory}")
    return {
        "splits": identity(parquet),
        "integrity": identity(integrity),
        "summary": identity(summary),
        "rows": int(len(derived)),
    }


def _snapshot_sources(root: Path) -> list[dict[str, Any]]:
    records = []
    for relative in material_source_files():
        live = REPO / relative
        if not live.is_file():
            raise FileNotFoundError(live)
        frozen = root / "source_snapshot" / relative
        _copy_once(live, frozen)
        live_identity = identity(live)
        frozen_identity = identity(frozen)
        if live_identity["sha256"] != frozen_identity["sha256"]:
            raise ContractError(f"source snapshot copy drifted: {relative}")
        records.append(
            {
                "relative_path": relative,
                "sha256": live_identity["sha256"],
                "size_bytes": live_identity["size_bytes"],
            }
        )
    return records


def smoke_frozen_controller(root: Path) -> dict[str, Any]:
    """Execute the frozen controller's import graph and read-only plan path."""

    controller = root / "source_snapshot/aim3_tcga_surgen_primary_five_seed_campaign.py"
    command = [
        sys.executable,
        str(controller),
        "plan",
        "--output-root",
        str(root),
    ]
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/oceanpath-matplotlib")
    result = subprocess.run(
        command,
        cwd=root / "source_snapshot",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ContractError(
            "frozen controller import/plan smoke failed before training: "
            f"rc={result.returncode}; stderr={result.stderr[-4000:]}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError("frozen controller plan did not emit one JSON object") from exc
    if (
        payload.get("status") != "PLAN_ONLY_NO_WRITES"
        or payload.get("campaign") != CAMPAIGN
        or payload.get("encoder") != "UNI-v1"
        or payload.get("model_seeds") != list(MODEL_SEEDS)
        or payload.get("fit_accounting") != fit_accounting()
        or len(payload.get("jobs", [])) != TOTAL_CHAINS
    ):
        raise ContractError("frozen controller plan semantics drifted")
    return {
        "status": "PASS",
        "controller": identity(controller),
        "command": command,
        "returncode": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "fit_accounting": payload["fit_accounting"],
        "job_count": len(payload["jobs"]),
    }


def _pack_contract(source_contract: Mapping[str, Any], selected_slides: set[str]) -> dict[str, Any]:
    pack = source_contract.get("feature_store")
    if not isinstance(pack, Mapping) or pack.get("encoder") != "UNI-v1" or pack.get("feature_dim") != 1024:
        raise ContractError("Aim-1 source contract lacks the sealed UNI-v1 packed store")
    artifacts = pack.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ContractError("sealed packed-store evidence is malformed")
    for name in ("meta.json", "index.parquet", "features.bin", "coords.bin"):
        evidence = artifacts.get(name)
        if not isinstance(evidence, Mapping):
            raise ContractError(f"sealed packed-store evidence lacks {name}")
        path = Path(str(evidence.get("path", "")))
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"sealed packed-store artifact missing: {path}")
        if int(path.stat().st_size) != int(evidence.get("size_bytes", -1)):
            raise ContractError(f"sealed packed-store size drifted: {path}")
        if name in {"meta.json", "index.parquet"} and sha256_file(path) != evidence.get("sha256"):
            raise ContractError(f"sealed packed-store metadata/index hash drifted: {path}")
    index_path = Path(str(artifacts["index.parquet"]["path"]))
    index = pd.read_parquet(index_path)
    id_column = "slide_id" if "slide_id" in index.columns else "key"
    indexed = set(index[id_column].astype(str))
    missing = sorted(selected_slides - indexed)
    if missing:
        raise ContractError(f"UNI-v1 pack lacks {len(missing)} selected slides; first={missing[:5]}")
    return {
        "encoder": "UNI-v1",
        "feature_dim": 1024,
        "path": str(Path(str(pack["path"])).resolve()),
        "artifacts": {name: dict(artifacts[name]) for name in artifacts},
        "selected_slide_count": len(selected_slides),
        "missing_selected_slides": 0,
        "payload_sha256_authority": "sealed Aim-1 source contract; metadata/index rehashed live",
    }


def contract_semantics(root: Path) -> dict[str, Any]:
    jobs = job_inventory()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "campaign": CAMPAIGN,
        "output_root": str(root.resolve(strict=False)),
        "population": {
            "development_role": "TCGA + SurGen primaries only",
            "source_campaign": SOURCE_CAMPAIGN,
            "source_arm": SOURCE_ARM,
            "allowed_cohorts": sorted(ALLOWED_COHORTS),
            "allowed_specimen_roles": ["primary"],
            "external_validation_cohorts_used_for_development": [],
            "forbidden_development_cohorts": list(FORBIDDEN_DEVELOPMENT_COHORTS),
        },
        "protocol": {
            "encoder": "UNI-v1",
            "model_seeds": list(MODEL_SEEDS),
            "wt_draw_seeds": list(WT_DRAW_SEEDS),
            "fixed_wt_seeds": FIXED_WT_SEEDS,
            "fine_tasks": list(FINE_TASKS),
            "matched_controls": CONTROL_FOR,
            "outer_folds": list(FOLDS),
            "split_policy": "strict subsets of sealed tcga_surgen_primary outer and validation folds; never redraw",
            "native_logit_ensemble": "mean slides within patient and seed, then mean patient logits over seeds 42..46",
            "finalize": "required p75 full-data refit for every chain",
            "max_parallel_chains": MAX_JOBS,
        },
        "fit_accounting": fit_accounting(jobs),
        "jobs": [
            dataclasses.asdict(job)
            | {
                "task": job.task,
                "key": job.key,
                "oof_folds": N_FOLDS,
                "p75_refits": 1,
                "physical_fits": job.actual_fits,
            }
            for job in jobs
        ],
    }


def cmd_prepare(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=False)
    if not args.apply:
        print(f"DRY RUN — would exclusively create {root}")
        print("25 variants / 125 chains / 625 OOF folds / 125 p75 refits / 750 fits")
        return
    source, lineage = verify_source_lineage(Path(args.source_root))
    manifests = build_all_task_manifests(source)
    root.mkdir(parents=True, exist_ok=False)
    source_paths = _source_paths(Path(args.source_root).resolve())
    _copy_once(source_paths["manifest"], source_snapshot_manifest(root))
    frozen_split_dir = source_snapshot_splits(root)
    for name in ("splits.parquet", ".integrity_hash", "summary.json"):
        _copy_once(source_paths["splits"].parent / name, frozen_split_dir / name)
    _copy_once(source_paths["contract"], root / "inputs/source_lineage/source_contract.json")
    _copy_once(source_paths["training_seal"], root / "inputs/source_lineage/source_training_complete.json")
    source_inventory = _snapshot_sources(root)
    frozen_controller_smoke = smoke_frozen_controller(root)
    task_records: dict[str, Any] = {}
    selected_slides: set[str] = set()
    frozen_source = pd.read_csv(source_snapshot_manifest(root), low_memory=False)
    for (kind, fine, draw), frame in manifests.items():
        destination = manifest_path(root, kind, fine, draw)
        _write_text_once(destination, frame.to_csv(index=False))
        directory = split_dir(root, kind, fine, draw)
        derive_subset_splits(
            frozen_split_dir,
            destination,
            directory,
            filename_column="slide_id",
            force=False,
        )
        record = {
            "kind": kind,
            "fine_task": fine,
            "control_task": None if kind == "fine" else CONTROL_FOR[fine],
            "draw_seed": draw if kind == "repeated" else (
                FIXED_WT_SEEDS[fine] if kind == "fixed" else None
            ),
            "manifest": identity(destination),
            "census": validate_task_manifest(frame, frozen_source, kind, fine, draw),
            "split": _validate_derived_split(
                directory, frame, frozen_split_dir / "splits.parquet"
            ),
        }
        task_records[variant_name(kind, fine, draw)] = record
        selected_slides.update(frame["slide_id"].astype(str))
    source_contract = _read_json(root / "inputs/source_lineage/source_contract.json")
    payload = {
        **contract_semantics(root),
        "created_utc": _utcnow(),
        "source_lineage": lineage,
        "frozen_source_lineage": {
            "manifest": identity(source_snapshot_manifest(root)),
            "splits": identity(frozen_split_dir / "splits.parquet"),
            "integrity": identity(frozen_split_dir / ".integrity_hash"),
            "summary": identity(frozen_split_dir / "summary.json"),
            "contract": identity(root / "inputs/source_lineage/source_contract.json"),
            "training_seal": identity(root / "inputs/source_lineage/source_training_complete.json"),
        },
        "task_inputs": task_records,
        "packed_store": _pack_contract(source_contract, selected_slides),
        "material_sources": source_inventory,
        "frozen_controller_smoke": frozen_controller_smoke,
    }
    _write_json_once(contract_path(root), payload)
    verify_contract(root, deep=True)
    print(f"PASS — prepared immutable governed Aim-3 source campaign at {root}")


def verify_contract(root: Path, *, deep: bool) -> dict[str, Any]:
    root = validate_output_root(root, must_exist=True)
    contract = _read_json(contract_path(root))
    expected = contract_semantics(root)
    mismatch = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ContractError(f"campaign contract semantics drifted: {mismatch}")
    source = pd.read_csv(source_snapshot_manifest(root), low_memory=False)
    production_source = Path(
        str(contract.get("source_lineage", {}).get("root", ""))
    ).resolve(strict=False) == SOURCE_ROOT.resolve()
    validate_source_population(source, production=production_source)
    frozen_expected = {
        "manifest": source_snapshot_manifest(root),
        "splits": source_snapshot_splits(root) / "splits.parquet",
        "integrity": source_snapshot_splits(root) / ".integrity_hash",
        "summary": source_snapshot_splits(root) / "summary.json",
        "contract": root / "inputs/source_lineage/source_contract.json",
        "training_seal": root / "inputs/source_lineage/source_training_complete.json",
    }
    recorded_frozen = contract.get("frozen_source_lineage", {})
    for name, path in frozen_expected.items():
        if recorded_frozen.get(name) != identity(path):
            raise ContractError(f"frozen source-lineage identity drifted: {name}")
    if production_source:
        observed_source_hashes = {
            "manifest": identity(frozen_expected["manifest"])["sha256"],
            "splits": identity(frozen_expected["splits"])["sha256"],
            "integrity": identity(frozen_expected["integrity"])["sha256"],
            "summary": identity(frozen_expected["summary"])["sha256"],
            "contract": identity(frozen_expected["contract"])["sha256"],
            "training_seal": identity(frozen_expected["training_seal"])["sha256"],
        }
        if observed_source_hashes != SOURCE_EXPECTED_SHA256:
            raise ContractError("frozen Aim-1 source lineage does not match pinned identities")
    for fine in FINE_TASKS:
        for kind, draws in (("fine", (None,)), ("fixed", (None,)), ("repeated", WT_DRAW_SEEDS)):
            for draw in draws:
                name = variant_name(kind, fine, draw)  # type: ignore[arg-type]
                frame = pd.read_csv(manifest_path(root, kind, fine, draw), low_memory=False)  # type: ignore[arg-type]
                census = validate_task_manifest(frame, source, kind, fine, draw)  # type: ignore[arg-type]
                record = contract.get("task_inputs", {}).get(name, {})
                if record.get("manifest") != identity(manifest_path(root, kind, fine, draw)) or record.get("census") != census:  # type: ignore[arg-type]
                    raise ContractError(f"task input contract drifted: {name}")
                split = _validate_derived_split(
                    split_dir(root, kind, fine, draw),  # type: ignore[arg-type]
                    frame,
                    source_snapshot_splits(root) / "splits.parquet",
                )
                if record.get("split") != split:
                    raise ContractError(f"task split contract drifted: {name}")
    recorded_sources = contract.get("material_sources")
    if not isinstance(recorded_sources, list):
        raise ContractError("contract lacks material source inventory")
    smoke = contract.get("frozen_controller_smoke")
    frozen_controller = (
        root / "source_snapshot/aim3_tcga_surgen_primary_five_seed_campaign.py"
    )
    if (
        not isinstance(smoke, dict)
        or smoke.get("status") != "PASS"
        or smoke.get("controller") != identity(frozen_controller)
        or smoke.get("returncode") != 0
        or smoke.get("fit_accounting") != fit_accounting()
        or smoke.get("job_count") != TOTAL_CHAINS
    ):
        raise ContractError("contract lacks a valid frozen-controller import/plan smoke")
    if deep:
        for record in recorded_sources:
            path = root / "source_snapshot" / str(record["relative_path"])
            if identity(path)["sha256"] != record.get("sha256"):
                raise ContractError(f"frozen source drifted: {record['relative_path']}")
    return contract


def job_from_key(key: str) -> Job:
    matches = [job for job in job_inventory() if job.key == key]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous job key: {key}")
    return matches[0]


def _study_overrides(
    root: Path, job: Job, *, num_workers: int, attempt_id: str
) -> list[str]:
    manifest = manifest_path(root, job.kind, job.fine_task, job.draw_seed)
    splits = split_dir(root, job.kind, job.fine_task, job.draw_seed)
    destination = run_dir(root, job)
    data_name = f"aim1_a3src_{variant_name(job.kind, job.fine_task, job.draw_seed)}"
    return [
        "platform=colon_workstation",
        "data=aim1",
        f"data.aim1_model={data_name.removeprefix('aim1_')}",
        f"data.name={data_name}",
        f"data.manifest_stem={data_name}",
        f"data.csv_path={manifest}",
        "+data.cohort_column=cohort",
        "encoder=univ1",
        "splits=aim1_balanced",
        f"+splits.output_dir={splits}",
        f"splits.seed={job.model_seed}",
        "model=abmil",
        "model.embed_dim=512",
        "model.attn_dim=384",
        "model.input_dropout=0.10",
        f"model.dropout={float(paths.DROPOUT):g}",
        "training=aim1",
        f"training.lr={float(paths.LR):g}",
        f"training.weight_decay={float(paths.WEIGHT_DECAY):g}",
        f"training.seed={job.model_seed}",
        f"training.dataset_max_instances={CAP}",
        "training.eval_full_bags=true",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        f"training.fixed_epoch_budget={FIXED_EPOCH_BUDGET}",
        "training.skip_finalize=false",
        f"training.num_workers={num_workers}",
        f"training.packed_dir={paths.PACKED_FEATURE_DIR}",
        "training.verify_packed_source=false",
        f"train_dir={destination}",
        f"exp_name=aim3_source_{job.key}",
        f"hydra.run.dir={root / 'state/hydra' / job.key / attempt_id}",
        "hydra.job.chdir=false",
    ]


def train_command(
    root: Path,
    job: Job,
    *,
    num_workers: int = DEFAULT_NUM_WORKERS,
    attempt_id: str = "scheduler",
) -> list[str]:
    if num_workers < 0 or not attempt_id or "/" in attempt_id or "\\" in attempt_id:
        raise ValueError("invalid num_workers or attempt_id")
    return [
        sys.executable,
        str(root / "source_snapshot/tools/study_train.py"),
        "hydra-train",
        *_study_overrides(root, job, num_workers=num_workers, attempt_id=attempt_id),
    ]


def build_training_jobs(
    root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> list[dict[str, Any]]:
    root = Path(root).resolve(strict=False)
    controller = root / "source_snapshot/aim3_tcga_surgen_primary_five_seed_campaign.py"
    values = []
    for job in job_inventory():
        command = [
            sys.executable,
            str(controller),
            "_train-one",
            "--output-root",
            str(root),
            "--job-key",
            job.key,
            "--num-workers",
            str(num_workers),
        ]
        values.append(
            {
                "job_id": f"aim3.source.{job.key}",
                "job_key": job.key,
                "kind": job.kind,
                "task": job.task,
                "fine_task": job.fine_task,
                "draw_seed": job.draw_seed,
                "model_seed": job.model_seed,
                "scheduler_slots": 1,
                "oof_folds": N_FOLDS,
                "p75_refits": 1,
                "fit_count": job.actual_fits,
                "output": str(run_dir(root, job)),
                "command": command,
                "training_command": train_command(root, job, num_workers=num_workers),
            }
        )
    return values


def phase_jobs(phase: str) -> list[Job]:
    jobs = job_inventory()
    if phase == "fine":
        return [job for job in jobs if job.kind == "fine"]
    if phase == "controls":
        return [job for job in jobs if job.kind in {"fixed", "repeated"}]
    if phase in {"fixed", "repeated"}:
        return [job for job in jobs if job.kind == phase]
    if phase == "all":
        return jobs
    raise ValueError(f"unknown phase: {phase}")


def phase_accounting(phase: str) -> dict[str, int]:
    return fit_accounting(phase_jobs(phase))


def _control_artifact_paths(root: Path) -> list[Path]:
    paths_found = []
    for job in phase_jobs("controls"):
        candidates = (
            run_dir(root, job),
            request_path(root, job),
            log_path(root, job),
            failure_path(root, job),
            job_receipt_path(root, job),
        )
        paths_found.extend(path for path in candidates if path.exists() or path.is_symlink())
    return paths_found


def assert_no_control_artifacts(root: Path) -> None:
    found = _control_artifact_paths(root)
    if found:
        raise ContractError(
            "fine-only phase forbids fixed/repeated request, run, log, failure, or "
            f"receipt artifacts; first={found[:5]}"
        )


def audit_control_resume_state(root: Path) -> dict[str, list[str]]:
    """Allow certified cached controls plus untouched pending controls only."""

    completed: list[str] = []
    pending: list[str] = []
    for job in phase_jobs("controls"):
        receipt = job_receipt_path(root, job)
        if receipt.is_file() and not receipt.is_symlink():
            validate_job(root, job)
            completed.append(job.key)
            continue
        partial = [
            path
            for path in (
                run_dir(root, job),
                request_path(root, job),
                log_path(root, job),
                failure_path(root, job),
                receipt,
            )
            if path.exists() or path.is_symlink()
        ]
        if partial:
            raise ContractError(
                f"uncertified partial control job cannot be resumed in place: {job.key}; "
                f"artifacts={partial}"
            )
        pending.append(job.key)
    return {"completed": completed, "pending": pending}


def cmd_plan(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root))
    print(
        json.dumps(
            {
                "status": "PLAN_ONLY_NO_WRITES",
                "campaign": CAMPAIGN,
                "output_root": str(root),
                "source_lineage": {"campaign": SOURCE_CAMPAIGN, "arm": SOURCE_ARM},
                "encoder": "UNI-v1",
                "model_seeds": list(MODEL_SEEDS),
                "wt_draw_seeds": list(WT_DRAW_SEEDS),
                "fit_accounting": fit_accounting(),
                "max_parallel_chains": MAX_JOBS,
                "external_development_cohorts": [],
                "jobs": build_training_jobs(root),
            },
            indent=2,
        )
    )


def cmd_preflight(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    contract = verify_contract(root, deep=True)
    selected: set[str] = set()
    for fine in FINE_TASKS:
        selected.update(
            pd.read_csv(manifest_path(root, "fine", fine), usecols=["slide_id"])[
                "slide_id"
            ].astype(str)
        )
        selected.update(
            pd.read_csv(manifest_path(root, "fixed", fine), usecols=["slide_id"])[
                "slide_id"
            ].astype(str)
        )
        for draw in WT_DRAW_SEEDS:
            selected.update(
                pd.read_csv(
                    manifest_path(root, "repeated", fine, draw),
                    usecols=["slide_id"],
                )["slide_id"].astype(str)
            )
    pack = _pack_contract(
        _read_json(root / "inputs/source_lineage/source_contract.json"), selected
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "created_utc": _utcnow(),
        "contract": identity(contract_path(root)),
        "source_training_seal": contract["frozen_source_lineage"]["training_seal"],
        "packed_store": pack,
        "fit_accounting": fit_accounting(),
        "configured_max_parallel_chains": MAX_JOBS,
    }
    if args.apply:
        _write_json_once(preflight_path(root), payload)
    print(json.dumps({**payload, "persisted": bool(args.apply)}, indent=2))


def _verify_preflight(root: Path) -> dict[str, Any]:
    receipt = _read_json(preflight_path(root))
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "PASS"
        or receipt.get("contract") != identity(contract_path(root))
        or receipt.get("fit_accounting") != fit_accounting()
        or receipt.get("configured_max_parallel_chains") != MAX_JOBS
    ):
        raise ContractError("persisted preflight receipt is invalid")
    return receipt


def _assert_recipe(directory: Path, root: Path, job: Job) -> None:
    material = (
        _read_json(directory / "training_identity.json")
        .get("payload", {})
        .get("material_config", {})
    )
    expected_training = {
        "dataset_max_instances": CAP,
        "eval_full_bags": True,
        "train_sampling_strategy": "patient_natural",
        "sample_weight_column": None,
        "fixed_epoch_budget": FIXED_EPOCH_BUDGET,
        "seed": job.model_seed,
        "skip_finalize": False,
    }
    training = material.get("training", {})
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ContractError(f"{job.key}: training recipe drift at {key}")
    encoder = material.get("encoder", {})
    if (encoder.get("name"), encoder.get("feature_dim")) != ("uni_v1", 1024):
        raise ContractError(f"{job.key}: encoder is not UNI-v1/1024")
    model = material.get("model", {})
    for key, expected in {
        "arch": "abmil",
        "embed_dim": 512,
        "attn_dim": 384,
        "input_dropout": 0.1,
        "dropout": float(paths.DROPOUT),
    }.items():
        if model.get(key) != expected:
            raise ContractError(f"{job.key}: model recipe drift at {key}")
    evidence = _read_json(directory / "training_identity.json").get("payload", {}).get(
        "input_evidence", {}
    )
    manifest = manifest_path(root, job.kind, job.fine_task, job.draw_seed)
    splits = split_dir(root, job.kind, job.fine_task, job.draw_seed)
    if (
        evidence.get("manifest_sha256") != sha256_file(manifest)
        or evidence.get("split_integrity_sha256")
        != sha256_file(splits / ".integrity_hash")
    ):
        raise ContractError(f"{job.key}: manifest/split training evidence drifted")


def validate_job(root: Path, job: Job) -> dict[str, Any]:
    directory = run_dir(root, job)
    artifacts = ladder5.validate_chain(directory, skip_finalize=False)
    _assert_recipe(directory, root, job)
    manifest_file = manifest_path(root, job.kind, job.fine_task, job.draw_seed)
    manifest = pd.read_csv(manifest_file, low_memory=False)
    oof = pd.read_parquet(directory / "oof_predictions.parquet")
    joined = oof.merge(
        manifest[["slide_id", "target_label", "k_fold"]],
        on="slide_id",
        how="inner",
        validate="one_to_one",
    )
    if (
        len(joined) != len(manifest)
        or set(oof["slide_id"].astype(str)) != set(manifest["slide_id"].astype(str))
        or not pd.to_numeric(joined["label"]).astype(int).eq(
            pd.to_numeric(joined["target_label"]).astype(int)
        ).all()
        or not pd.to_numeric(joined["fold"]).astype(int).eq(
            pd.to_numeric(joined["k_fold"]).astype(int)
        ).all()
    ):
        raise ContractError(f"{job.key}: exact OOF roster/label/fold validation failed")
    artifacts.update(
        {
            "manifest": identity(manifest_file),
            "splits": identity(
                split_dir(root, job.kind, job.fine_task, job.draw_seed)
                / "splits.parquet"
            ),
        }
    )
    receipt = _read_json(job_receipt_path(root, job))
    expected = {
        "status": "completed",
        "job_key": job.key,
        "oof_folds": N_FOLDS,
        "p75_refits": 1,
        "physical_fits": job.actual_fits,
        "contract": identity(contract_path(root)),
        "artifacts": artifacts,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(f"{job.key}: job receipt drift at {key}")
    return receipt


@contextlib.contextmanager
def _exclusive_lock(path: Path, *, label: str) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f"another process owns {label}: {path}") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _job_lock(root: Path, job: Job) -> Path:
    token = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    return Path(f"/tmp/{CAMPAIGN}_{token}_{job.key}.lock")


def _run_logged(command: Sequence[str], output: Path) -> int:
    if output.exists() or output.is_symlink():
        raise ContractError(f"refusing to overwrite immutable log: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen(
            list(command),
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        return int(process.wait())


def cmd_train_one(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    job = job_from_key(args.job_key)
    verify_contract(root, deep=False)
    _verify_preflight(root)
    with _exclusive_lock(_job_lock(root, job), label=job.key):
        if job_receipt_path(root, job).is_file():
            validate_job(root, job)
            print(f"PASS cached: {job.key}")
            return
        if run_dir(root, job).exists() or request_path(root, job).exists() or log_path(root, job).exists():
            raise ContractError(
                f"partial immutable job exists for {job.key}; use a fresh campaign root"
            )
        _write_json_once(
            request_path(root, job),
            {
                "schema_version": SCHEMA_VERSION,
                "status": "requested",
                "created_utc": _utcnow(),
                "job": dataclasses.asdict(job) | {"key": job.key},
                "contract": identity(contract_path(root)),
            },
        )
        command = train_command(
            root,
            job,
            num_workers=args.num_workers,
            attempt_id=f"pid{os.getpid()}",
        )
        code = _run_logged(command, log_path(root, job))
        if code:
            _write_json_once(
                failure_path(root, job),
                {"status": "failed", "returncode": code, "job_key": job.key},
            )
            raise SystemExit(code)
        artifacts = ladder5.validate_chain(run_dir(root, job), skip_finalize=False)
        _assert_recipe(run_dir(root, job), root, job)
        # Publish only after native completion and p75 refit validation pass.
        manifest_file = manifest_path(root, job.kind, job.fine_task, job.draw_seed)
        artifacts.update(
            {
                "manifest": identity(manifest_file),
                "splits": identity(
                    split_dir(root, job.kind, job.fine_task, job.draw_seed)
                    / "splits.parquet"
                ),
            }
        )
        _write_json_once(
            job_receipt_path(root, job),
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "finished_utc": _utcnow(),
                "job_key": job.key,
                "job": dataclasses.asdict(job),
                "oof_folds": N_FOLDS,
                "p75_refits": 1,
                "physical_fits": job.actual_fits,
                "contract": identity(contract_path(root)),
                "request": identity(request_path(root, job)),
                "log": identity(log_path(root, job)),
                "artifacts": artifacts,
            },
        )
        validate_job(root, job)
        print(f"PASS completed: {job.key}")


def _execute_job(record: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(list(record["command"]), cwd=REPO)
    code = int(process.wait())
    return {
        "job_id": str(record["job_id"]),
        "pid": int(process.pid),
        "returncode": code,
        "started_monotonic": started,
        "finished_monotonic": time.monotonic(),
    }


def _peak_parallel(events: Sequence[Mapping[str, Any]]) -> int:
    points = []
    for event in events:
        points += [
            (float(event["started_monotonic"]), 1),
            (float(event["finished_monotonic"]), -1),
        ]
    active = peak = 0
    for _when, delta in sorted(points, key=lambda item: (item[0], -item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def cmd_train(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    verify_contract(root, deep=False)
    jobs = build_training_jobs(root, num_workers=args.num_workers)
    phase = args.kind or "all"
    if phase == "controls":
        jobs = [job for job in jobs if job["kind"] in {"fixed", "repeated"}]
    elif phase != "all":
        jobs = [job for job in jobs if job["kind"] == phase]
    if args.model_seed is not None:
        jobs = [job for job in jobs if job["model_seed"] == args.model_seed]
    if not args.apply:
        print(
            f"DRY RUN: {len(jobs)} chains / {sum(int(j['fit_count']) for j in jobs)} "
            f"physical fits; max parallel={args.jobs}"
        )
        for job in jobs:
            print(json.dumps(job, sort_keys=True))
        return
    _verify_preflight(root)
    if not 1 <= args.jobs <= MAX_JOBS:
        raise ContractError(f"--jobs must be 1..{MAX_JOBS}")
    if root == DEFAULT_OUTPUT_ROOT.resolve() and args.jobs != MAX_JOBS:
        raise ContractError(f"production requires exactly --jobs {MAX_JOBS}")
    if root == DEFAULT_OUTPUT_ROOT.resolve() and phase not in {"fine", "controls"}:
        raise ContractError(
            "production execution is phased: use --kind fine first, then --kind controls"
        )
    if phase == "fine":
        assert_no_control_artifacts(root)
    if phase == "controls":
        _validate_fine_scheduler(root)
        _validate_fine_training_seal(root)
        _validate_fine_analysis(root)
        audit_control_resume_state(root)
    token = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    events: list[dict[str, Any]] = []
    failures = []
    with (
        _exclusive_lock(Path(f"/tmp/{CAMPAIGN}_{token}_launcher.lock"), label="campaign launcher"),
        ThreadPoolExecutor(max_workers=args.jobs) as pool,
    ):
        futures = {pool.submit(_execute_job, job): job for job in jobs}
        for future in as_completed(futures):
            event = future.result()
            events.append(event)
            if event["returncode"]:
                failures.append((event["job_id"], event["returncode"]))
            print(f"{event['job_id']}: rc={event['returncode']}", flush=True)
    if failures:
        raise SystemExit(f"training failures; immutable evidence retained: {failures}")
    if phase == "fine" and args.model_seed is None and len(jobs) == FINE_PHASE_CHAINS:
        assert_no_control_artifacts(root)
        peak = _peak_parallel(events)
        if peak > MAX_JOBS or (root == DEFAULT_OUTPUT_ROOT.resolve() and peak != MAX_JOBS):
            raise ContractError(f"fine scheduler concurrency evidence invalid: peak={peak}")
        _write_json_once(
            fine_scheduler_path(root),
            {
                "schema_version": SCHEMA_VERSION,
                "status": "fine_phase_completed_rc0",
                "phase": "fine",
                "created_utc": _utcnow(),
                "configured_max_parallel_chains": args.jobs,
                "observed_max_parallel_chains": peak,
                "job_count": len(events),
                "fit_accounting": phase_accounting("fine"),
                "job_keys": sorted(job.key for job in phase_jobs("fine")),
                "control_artifacts_absent": True,
                "events": sorted(events, key=lambda value: str(value["job_id"])),
            },
        )
    elif phase == "controls" and args.model_seed is None and len(jobs) == CONTROL_PHASE_CHAINS:
        peak = _peak_parallel(events)
        if peak > MAX_JOBS or (root == DEFAULT_OUTPUT_ROOT.resolve() and peak != MAX_JOBS):
            raise ContractError(f"control scheduler concurrency evidence invalid: peak={peak}")
        controls_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "control_phase_completed_rc0",
            "phase": "controls",
            "created_utc": _utcnow(),
            "configured_max_parallel_chains": args.jobs,
            "observed_max_parallel_chains": peak,
            "job_count": len(events),
            "fit_accounting": phase_accounting("controls"),
            "job_keys": sorted(job.key for job in phase_jobs("controls")),
            "fixed_scheduled_before_repeated": True,
            "fine_phase_scheduler": identity(fine_scheduler_path(root)),
            "fine_phase_analysis": identity(root / "analysis/fine_analysis_completion.json"),
            "events": sorted(events, key=lambda value: str(value["job_id"])),
        }
        _write_json_once(controls_scheduler_path(root), controls_payload)
        fine_scheduler = _read_json(fine_scheduler_path(root))
        _write_json_once(
            scheduler_path(root),
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed_phased_rc0",
                "created_utc": _utcnow(),
                "configured_max_parallel_chains": max(
                    int(fine_scheduler["configured_max_parallel_chains"]), args.jobs
                ),
                "observed_max_parallel_chains": max(
                    int(fine_scheduler["observed_max_parallel_chains"]), peak
                ),
                "job_count": TOTAL_CHAINS,
                "fit_accounting": fit_accounting(),
                "fixed_ladder_scheduled_before_repeated": True,
                "phase_order": ["fine", "fine_analysis_candidate", "controls"],
                "fine_scheduler": identity(fine_scheduler_path(root)),
                "controls_scheduler": identity(controls_scheduler_path(root)),
                "events": [*fine_scheduler["events"], *controls_payload["events"]],
            },
        )
    elif phase == "all" and args.model_seed is None and len(jobs) == TOTAL_CHAINS:
        peak = _peak_parallel(events)
        if peak > MAX_JOBS or (root == DEFAULT_OUTPUT_ROOT.resolve() and peak != MAX_JOBS):
            raise ContractError(f"scheduler concurrency evidence invalid: peak={peak}")
        _write_json_once(
            scheduler_path(root),
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed_rc0",
                "created_utc": _utcnow(),
                "configured_max_parallel_chains": args.jobs,
                "observed_max_parallel_chains": peak,
                "job_count": len(events),
                "fit_accounting": fit_accounting(),
                "fixed_ladder_scheduled_before_repeated": True,
                "events": sorted(events, key=lambda value: str(value["job_id"])),
            },
        )


def _validate_fine_scheduler(root: Path, *, require_no_controls: bool = False) -> dict[str, Any]:
    receipt = _read_json(fine_scheduler_path(root))
    expected = {
        "status": "fine_phase_completed_rc0",
        "phase": "fine",
        "configured_max_parallel_chains": MAX_JOBS,
        "observed_max_parallel_chains": MAX_JOBS,
        "job_count": FINE_PHASE_CHAINS,
        "fit_accounting": phase_accounting("fine"),
        "job_keys": sorted(job.key for job in phase_jobs("fine")),
        "control_artifacts_absent": True,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(f"fine scheduler receipt drift at {key}")
    events = receipt.get("events")
    if (
        not isinstance(events, list)
        or len(events) != FINE_PHASE_CHAINS
        or any(event.get("returncode") != 0 for event in events)
    ):
        raise ContractError("fine scheduler lacks 25 successful events")
    if require_no_controls:
        assert_no_control_artifacts(root)
    return receipt


def _validate_scheduler(root: Path) -> dict[str, Any]:
    receipt = _read_json(scheduler_path(root))
    expected = {
        "configured_max_parallel_chains": MAX_JOBS,
        "observed_max_parallel_chains": MAX_JOBS,
        "job_count": TOTAL_CHAINS,
        "fit_accounting": fit_accounting(),
        "fixed_ladder_scheduled_before_repeated": True,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(f"scheduler receipt drift at {key}")
    if receipt.get("status") not in {"completed_rc0", "completed_phased_rc0"}:
        raise ContractError("full scheduler status is invalid")
    if receipt.get("status") == "completed_phased_rc0" and (
        receipt.get("phase_order") != ["fine", "fine_analysis_candidate", "controls"]
        or receipt.get("fine_scheduler") != identity(fine_scheduler_path(root))
        or receipt.get("controls_scheduler") != identity(controls_scheduler_path(root))
    ):
        raise ContractError("phased scheduler linkage is invalid")
    events = receipt.get("events")
    if (
        not isinstance(events, list)
        or len(events) != TOTAL_CHAINS
        or any(event.get("returncode") != 0 for event in events)
    ):
        raise ContractError("scheduler receipt lacks 125 successful chain events")
    return receipt


def cmd_validate_fine(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    verify_contract(root, deep=True)
    _verify_preflight(root)
    _validate_fine_scheduler(root, require_no_controls=True)
    receipts = []
    fingerprints = set()
    for job in phase_jobs("fine"):
        receipt = validate_job(root, job)
        receipts.append(identity(job_receipt_path(root, job)))
        fingerprints.add(receipt["artifacts"]["training_identity"]["sha256"])
    if len(receipts) != FINE_PHASE_CHAINS or len(fingerprints) != FINE_PHASE_CHAINS:
        raise ContractError("fine phase requires 25 independently identified chains")
    assert_no_control_artifacts(root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "fine_phase_complete_and_certified",
        "created_utc": _utcnow(),
        "campaign": CAMPAIGN,
        "phase": "fine",
        "population": "tcga_surgen_primary",
        "encoder": "UNI-v1",
        "model_seeds": list(MODEL_SEEDS),
        "controls_status": "not_started",
        "fit_accounting": phase_accounting("fine"),
        "contract": identity(contract_path(root)),
        "preflight": identity(preflight_path(root)),
        "scheduler": identity(fine_scheduler_path(root)),
        "control_artifacts_absent": True,
        "external_development_cohorts": [],
        "job_receipts": receipts,
    }
    if args.seal:
        _write_json_once(fine_training_seal_path(root), payload)
    print(
        json.dumps(
            {"status": "PASS", "sealed": bool(args.seal), **phase_accounting("fine")},
            indent=2,
        )
    )


def _validate_fine_training_seal(root: Path, *, require_no_controls: bool = False) -> dict[str, Any]:
    receipt = _read_json(fine_training_seal_path(root))
    expected = {
        "status": "fine_phase_complete_and_certified",
        "campaign": CAMPAIGN,
        "phase": "fine",
        "population": "tcga_surgen_primary",
        "encoder": "UNI-v1",
        "model_seeds": list(MODEL_SEEDS),
        "controls_status": "not_started",
        "fit_accounting": phase_accounting("fine"),
        "contract": identity(contract_path(root)),
        "preflight": identity(preflight_path(root)),
        "scheduler": identity(fine_scheduler_path(root)),
        "control_artifacts_absent": True,
        "external_development_cohorts": [],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(f"fine training seal drift at {key}")
    if len(receipt.get("job_receipts", [])) != FINE_PHASE_CHAINS:
        raise ContractError("fine training seal does not bind 25 receipts")
    if require_no_controls:
        assert_no_control_artifacts(root)
    return receipt


def cmd_validate(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    verify_contract(root, deep=True)
    _verify_preflight(root)
    _validate_scheduler(root)
    receipts = []
    fingerprints = set()
    for job in job_inventory():
        receipt = validate_job(root, job)
        receipts.append(identity(job_receipt_path(root, job)))
        fingerprints.add(receipt["artifacts"]["training_identity"]["sha256"])
    if len(receipts) != TOTAL_CHAINS or len(fingerprints) != TOTAL_CHAINS:
        raise ContractError("125 independently identified chains are required")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_and_certified",
        "created_utc": _utcnow(),
        "campaign": CAMPAIGN,
        "population": "tcga_surgen_primary",
        "encoder": "UNI-v1",
        "model_seeds": list(MODEL_SEEDS),
        "wt_draw_seeds": list(WT_DRAW_SEEDS),
        "fit_accounting": fit_accounting(),
        "contract": identity(contract_path(root)),
        "preflight": identity(preflight_path(root)),
        "scheduler": identity(scheduler_path(root)),
        "external_development_cohorts": [],
        "job_receipts": receipts,
    }
    if args.seal:
        _write_json_once(training_seal_path(root), payload)
    print(json.dumps({"status": "PASS", "sealed": bool(args.seal), **fit_accounting()}, indent=2))


def _validate_training_seal(root: Path) -> dict[str, Any]:
    receipt = _read_json(training_seal_path(root))
    expected = {
        "status": "complete_and_certified",
        "campaign": CAMPAIGN,
        "population": "tcga_surgen_primary",
        "encoder": "UNI-v1",
        "model_seeds": list(MODEL_SEEDS),
        "wt_draw_seeds": list(WT_DRAW_SEEDS),
        "fit_accounting": fit_accounting(),
        "contract": identity(contract_path(root)),
        "preflight": identity(preflight_path(root)),
        "scheduler": identity(scheduler_path(root)),
        "external_development_cohorts": [],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(f"training seal drift at {key}")
    if len(receipt.get("job_receipts", [])) != TOTAL_CHAINS:
        raise ContractError("training seal does not bind 125 job receipts")
    return receipt


def _ensemble(root: Path, kind: Kind, fine: str, draw: int | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    manifest = pd.read_csv(manifest_path(root, kind, fine, draw), low_memory=False)
    frames = []
    per_seed = {}
    for seed in MODEL_SEEDS:
        job = Job(kind, fine, seed, draw)
        patient = evaluate.to_patient_level(
            pd.read_parquet(run_dir(root, job) / "oof_predictions.parquet"), manifest
        ).sort_values("patient_id").reset_index(drop=True)
        frames.append(patient)
        per_seed[str(seed)] = float(fixed_stats._task_point(patient))
    invariant = [
        column
        for column in ("patient_id", "label", "cohort", "subcohort", "k_fold")
        if column in frames[0]
    ]
    for frame in frames[1:]:
        pd.testing.assert_frame_equal(
            frames[0][invariant], frame[invariant], check_dtype=False
        )
    ensemble = frames[0].copy()
    ensemble["mean_logit"] = np.mean(
        np.stack([frame["mean_logit"].to_numpy(dtype=float) for frame in frames]), axis=0
    )
    ensemble["prob_raw"] = 1.0 / (1.0 + np.exp(-ensemble["mean_logit"].to_numpy()))
    return ensemble, per_seed


def _summary(point: float, values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    one_tail = 0.05 / len(FINE_TASKS)
    return {
        "estimate": float(point),
        "ci95_two_sided": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "primary_fwer_one_sided": {
            "lower": float(np.quantile(array, one_tail)),
            "upper": float(np.quantile(array, 1.0 - one_tail)),
            "confidence": float(1.0 - one_tail),
        },
    }


def _gate(fine: Mapping[str, Any], control: Mapping[str, Any], delta: Mapping[str, Any]) -> dict[str, Any]:
    fine_bounds = fine["primary_fwer_one_sided"]
    control_bounds = control["primary_fwer_one_sided"]
    delta_bounds = delta["primary_fwer_one_sided"]
    conditions = {
        "fine_upper_lt_0p60": fine_bounds["upper"] < CEILING_BOUND,
        "control_lower_gt_0p50": control_bounds["lower"] > CHANCE,
        "delta_lower_gt_zero": delta_bounds["lower"] > 0.0,
    }
    ceiling = all(conditions.values())
    signal = fine_bounds["lower"] > CHANCE
    if control_bounds["lower"] <= CHANCE:
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


def _patient_export(
    frame: pd.DataFrame, *, kind: str, fine: str, task: str, draw: int | None
) -> pd.DataFrame:
    columns = [
        column
        for column in ("patient_id", "label", "mean_logit", "cohort", "subcohort", "k_fold", "n_slides")
        if column in frame
    ]
    out = frame[columns].copy()
    out.insert(0, "draw_seed", draw)
    out.insert(0, "task", task)
    out.insert(0, "rung", fine)
    out.insert(0, "section", kind)
    return out


def compute_fine_analysis(
    root: Path, *, n_bootstrap: int = FIXED_BOOTSTRAPS
) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame]:
    rungs: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    patients = []
    bindings = []
    for fine in FINE_TASKS:
        frame, per_seed = _ensemble(root, "fine", fine)
        seed = fixed_stats._stable_seed(
            FIXED_BOOTSTRAP_SEED, "tcga_surgen_primary_fine", fine
        )
        values = fixed_stats.task_bootstrap(
            frame, n_bootstrap=n_bootstrap, seed=seed
        )
        summary = _summary(fixed_stats._task_point(frame), values)
        arrays[f"fine__{fine}__auroc"] = values
        patients.append(
            _patient_export(frame, kind="fine", fine=fine, task=fine, draw=None)
        )
        rungs[fine] = {
            "per_seed_auroc": per_seed,
            "five_seed_ensemble": summary,
            "patients": int(len(frame)),
            "positive": int(frame["label"].eq(1).sum()),
            "negative": int(frame["label"].eq(0).sum()),
        }
        bindings.append(
            {
                "id": f"aim3_source.fine.{fine}.auroc",
                "section": "fine",
                "rung": fine,
                "draw_seed": None,
                "metric": "fine_auroc",
                "value": summary["estimate"],
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "fine_phase_complete_candidate_unsealed",
        "campaign": CAMPAIGN,
        "phase": "fine",
        "population": "tcga_surgen_primary",
        "population_display": "TCGA + SurGen primaries",
        "encoder": "UNI-v1",
        "model_seeds": list(MODEL_SEEDS),
        "wt_draw_seeds": list(WT_DRAW_SEEDS),
        "controls_status": "not_started",
        "external_development_cohorts": [],
        "aggregation": "patient native-logit five-seed ensemble",
        "fit_accounting": phase_accounting("fine"),
        "fine": {"rungs": rungs},
        "report_bindings": sorted(bindings, key=lambda item: item["id"]),
        "v13_status": "candidate_unsealed_pending_fixed_and_repeated_controls",
    }
    return report, arrays, pd.concat(patients, ignore_index=True)


def cmd_analyze_fine(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    verify_contract(root, deep=False)
    _validate_fine_training_seal(root, require_no_controls=True)
    if not args.apply:
        print("DRY RUN — fine analysis requires --apply; no files were written")
        return
    report, arrays, patients = compute_fine_analysis(root)
    analysis = root / "analysis"
    _write_parquet_once(analysis / "fine_patient_native_logits.parquet", patients)
    _write_npz_once(analysis / "fine_bootstrap_distributions.npz", arrays)
    report.update(
        {
            "created_utc": _utcnow(),
            "contract": identity(contract_path(root)),
            "fine_training_seal": identity(fine_training_seal_path(root)),
            "fine_patient_native_logits": identity(
                analysis / "fine_patient_native_logits.parquet"
            ),
            "fine_bootstrap_distributions": identity(
                analysis / "fine_bootstrap_distributions.npz"
            ),
        }
    )
    _write_json_once(analysis / "fine_results.json", report)
    _write_json_once(
        analysis / "fine_analysis_completion.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "fine_phase_complete_candidate_unsealed",
            "results": identity(analysis / "fine_results.json"),
            "patient_native_logits": identity(
                analysis / "fine_patient_native_logits.parquet"
            ),
            "bootstrap_distributions": identity(
                analysis / "fine_bootstrap_distributions.npz"
            ),
            "report_binding_count": len(report["report_bindings"]),
            "controls_status": "not_started",
        },
    )
    _validate_fine_analysis(root, require_no_controls=True)
    print(f"PASS — fine-only candidate analysis sealed at {analysis}")


def _validate_fine_analysis(root: Path, *, require_no_controls: bool = False) -> dict[str, Any]:
    analysis = root / "analysis"
    receipt = _read_json(analysis / "fine_analysis_completion.json")
    results = _read_json(analysis / "fine_results.json")
    if (
        receipt.get("status") != "fine_phase_complete_candidate_unsealed"
        or receipt.get("results") != identity(analysis / "fine_results.json")
        or receipt.get("patient_native_logits")
        != identity(analysis / "fine_patient_native_logits.parquet")
        or receipt.get("bootstrap_distributions")
        != identity(analysis / "fine_bootstrap_distributions.npz")
        or receipt.get("controls_status") != "not_started"
        or results.get("status") != "fine_phase_complete_candidate_unsealed"
        or results.get("phase") != "fine"
        or results.get("population") != "tcga_surgen_primary"
        or results.get("encoder") != "UNI-v1"
        or results.get("model_seeds") != list(MODEL_SEEDS)
        or results.get("controls_status") != "not_started"
        or results.get("external_development_cohorts") != []
        or results.get("fit_accounting") != phase_accounting("fine")
        or set(results.get("fine", {}).get("rungs", {})) != set(FINE_TASKS)
        or len(results.get("report_bindings", [])) != len(FINE_TASKS)
    ):
        raise ContractError("fine-only analysis receipt/results contract is invalid")
    if require_no_controls:
        assert_no_control_artifacts(root)
    return results


def compute_analysis(
    root: Path,
    *,
    fixed_bootstraps: int = FIXED_BOOTSTRAPS,
    repeated_bootstraps: int = REPEATED_BOOTSTRAPS,
) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame]:
    fixed_result: dict[str, Any] = {"rungs": {}}
    repeated_result: dict[str, Any] = {"rungs": {}}
    arrays: dict[str, np.ndarray] = {}
    patients = []
    bindings: list[dict[str, Any]] = []
    for index, fine in enumerate(FINE_TASKS):
        fine_frame, fine_seeds = _ensemble(root, "fine", fine)
        fixed_frame, fixed_seeds = _ensemble(root, "fixed", fine)
        patients += [
            _patient_export(fine_frame, kind="fine", fine=fine, task=fine, draw=None),
            _patient_export(
                fixed_frame,
                kind="fixed",
                fine=fine,
                task=CONTROL_FOR[fine],
                draw=None,
            ),
        ]
        fixed_seed = fixed_stats._stable_seed(
            FIXED_BOOTSTRAP_SEED, "tcga_surgen_primary_fixed", fine
        )
        boot = repeated_stats.partial_paired_bootstrap(
            fine_frame,
            fixed_frame,
            n_bootstrap=fixed_bootstraps,
            seed=fixed_seed,
        )
        fine_summary = _summary(boot["point"]["fine"], boot["fine_values"])
        control_summary = _summary(boot["point"]["control"], boot["control_values"])
        delta_summary = _summary(
            boot["point"]["control"] - boot["point"]["fine"],
            boot["delta_values"],
        )
        prefix = f"fixed__{fine}"
        arrays[f"{prefix}__fine"] = boot["fine_values"]
        arrays[f"{prefix}__control"] = boot["control_values"]
        arrays[f"{prefix}__delta"] = boot["delta_values"]
        fixed_result["rungs"][fine] = {
            "control_task": CONTROL_FOR[fine],
            "fine_per_seed": fine_seeds,
            "control_per_seed": fixed_seeds,
            "fine": fine_summary,
            "control": control_summary,
            "delta_control_minus_fine": delta_summary,
            "gate": _gate(fine_summary, control_summary, delta_summary),
        }
        for metric, value in (
            ("fine_auroc", fine_summary["estimate"]),
            ("control_auroc", control_summary["estimate"]),
            ("delta_auroc", delta_summary["estimate"]),
        ):
            bindings.append(
                {
                    "id": f"aim3_source.fixed.{fine}.{metric}",
                    "section": "fixed",
                    "rung": fine,
                    "draw_seed": None,
                    "metric": metric,
                    "value": value,
                }
            )
        draws: dict[str, Any] = {}
        verdicts = []
        common_seed = int(
            np.random.SeedSequence([REPEATED_BOOTSTRAP_SEED, index]).generate_state(1)[0]
        )
        for draw in WT_DRAW_SEEDS:
            control_frame, control_seeds = _ensemble(root, "repeated", fine, draw)
            patients.append(
                _patient_export(
                    control_frame,
                    kind="repeated",
                    fine=fine,
                    task=CONTROL_FOR[fine],
                    draw=draw,
                )
            )
            boot = repeated_stats.partial_paired_bootstrap(
                fine_frame,
                control_frame,
                n_bootstrap=repeated_bootstraps,
                seed=common_seed,
            )
            fine_summary = _summary(boot["point"]["fine"], boot["fine_values"])
            control_summary = _summary(
                boot["point"]["control"], boot["control_values"]
            )
            delta_summary = _summary(
                boot["point"]["control"] - boot["point"]["fine"],
                boot["delta_values"],
            )
            gate = _gate(fine_summary, control_summary, delta_summary)
            verdicts.append(gate["verdict"])
            prefix = f"repeated__{fine}__wt{draw}"
            arrays[f"{prefix}__fine"] = boot["fine_values"]
            arrays[f"{prefix}__control"] = boot["control_values"]
            arrays[f"{prefix}__delta"] = boot["delta_values"]
            draws[str(draw)] = {
                "control_per_seed": control_seeds,
                "fine": fine_summary,
                "control": control_summary,
                "delta_control_minus_fine": delta_summary,
                "gate": gate,
            }
            for metric, value in (
                ("fine_auroc", fine_summary["estimate"]),
                ("control_auroc", control_summary["estimate"]),
                ("delta_auroc", delta_summary["estimate"]),
            ):
                bindings.append(
                    {
                        "id": f"aim3_source.repeated.{fine}.wt{draw}.{metric}",
                        "section": "repeated",
                        "rung": fine,
                        "draw_seed": draw,
                        "metric": metric,
                        "value": value,
                    }
                )
        consensus = repeated_stats.consensus_verdict(verdicts)
        repeated_result["rungs"][fine] = {
            "control_task": CONTROL_FOR[fine],
            "fine_per_seed": fine_seeds,
            "draws": draws,
            "consensus_verdict": consensus,
            "all_three_draws_required": True,
        }
        bindings.append(
            {
                "id": f"aim3_source.repeated.{fine}.consensus_verdict",
                "section": "repeated",
                "rung": fine,
                "draw_seed": None,
                "metric": "consensus_verdict",
                "value": consensus,
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "campaign": CAMPAIGN,
        "population": "tcga_surgen_primary",
        "population_display": "TCGA + SurGen primaries",
        "encoder": "UNI-v1",
        "model_seeds": list(MODEL_SEEDS),
        "wt_draw_seeds": list(WT_DRAW_SEEDS),
        "external_development_cohorts": [],
        "aggregation": "patient native-logit five-seed ensemble",
        "fixed": fixed_result,
        "repeated": repeated_result,
        "report_bindings": sorted(bindings, key=lambda item: item["id"]),
    }
    return report, arrays, pd.concat(patients, ignore_index=True)


def cmd_analyze(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    verify_contract(root, deep=False)
    _validate_training_seal(root)
    if not args.apply:
        print("DRY RUN — analysis requires --apply; no files were written")
        return
    report, arrays, patients = compute_analysis(root)
    analysis = root / "analysis"
    _write_parquet_once(analysis / "patient_native_logits.parquet", patients)
    _write_npz_once(analysis / "bootstrap_distributions.npz", arrays)
    report.update(
        {
            "created_utc": _utcnow(),
            "contract": identity(contract_path(root)),
            "training_seal": identity(training_seal_path(root)),
            "patient_native_logits": identity(analysis / "patient_native_logits.parquet"),
            "bootstrap_distributions": identity(analysis / "bootstrap_distributions.npz"),
        }
    )
    _write_json_once(analysis / "results.json", report)
    _write_json_once(
        analysis / "analysis_completion.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "results": identity(analysis / "results.json"),
            "patient_native_logits": identity(analysis / "patient_native_logits.parquet"),
            "bootstrap_distributions": identity(analysis / "bootstrap_distributions.npz"),
            "report_binding_count": len(report["report_bindings"]),
        },
    )
    print(f"PASS — sealed Aim-3 source analysis at {analysis}")


def _validate_analysis(root: Path) -> dict[str, Any]:
    analysis = root / "analysis"
    receipt = _read_json(analysis / "analysis_completion.json")
    results = _read_json(analysis / "results.json")
    if (
        receipt.get("status") != "complete"
        or receipt.get("results") != identity(analysis / "results.json")
        or receipt.get("patient_native_logits")
        != identity(analysis / "patient_native_logits.parquet")
        or receipt.get("bootstrap_distributions")
        != identity(analysis / "bootstrap_distributions.npz")
        or results.get("campaign") != CAMPAIGN
        or results.get("population") != "tcga_surgen_primary"
        or results.get("encoder") != "UNI-v1"
        or results.get("model_seeds") != list(MODEL_SEEDS)
        or results.get("wt_draw_seeds") != list(WT_DRAW_SEEDS)
        or results.get("external_development_cohorts") != []
        or set(results.get("fixed", {}).get("rungs", {})) != set(FINE_TASKS)
        or set(results.get("repeated", {}).get("rungs", {})) != set(FINE_TASKS)
        or not results.get("report_bindings")
    ):
        raise ContractError("analysis receipt/results contract is invalid")
    for fine in FINE_TASKS:
        repeated = results["repeated"]["rungs"][fine]
        if set(repeated.get("draws", {})) != {str(seed) for seed in WT_DRAW_SEEDS}:
            raise ContractError(f"analysis lacks all repeated draws for {fine}")
    return results


def cmd_verify(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    verify_contract(root, deep=True)
    stages = {"contract": "PASS"}
    if preflight_path(root).is_file():
        _verify_preflight(root)
        stages["preflight"] = "PASS"
    if fine_training_seal_path(root).is_file():
        controls_started = bool(_control_artifact_paths(root))
        _validate_fine_scheduler(root, require_no_controls=not controls_started)
        _validate_fine_training_seal(root, require_no_controls=not controls_started)
        for job in phase_jobs("fine"):
            validate_job(root, job)
        stages["fine_training"] = "PASS"
    if (root / "analysis/fine_analysis_completion.json").is_file():
        _validate_fine_analysis(root, require_no_controls=not bool(_control_artifact_paths(root)))
        stages["fine_analysis_candidate"] = "PASS"
    if training_seal_path(root).is_file():
        _validate_scheduler(root)
        _validate_training_seal(root)
        for job in job_inventory():
            validate_job(root, job)
        stages["training"] = "PASS"
    if (root / "analysis/analysis_completion.json").is_file():
        _validate_analysis(root)
        stages["analysis"] = "PASS"
    print(json.dumps({"status": "PASS", "stages": stages}, indent=2))


def cmd_status(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root))
    states = {"completed": 0, "partial": 0, "pending": 0}
    rows = []
    for job in job_inventory():
        if job_receipt_path(root, job).is_file():
            state = "completed"
        elif run_dir(root, job).exists() or request_path(root, job).exists():
            state = "partial"
        else:
            state = "pending"
        states[state] += 1
        rows.append({"job_key": job.key, "state": state})
    print(
        json.dumps(
            {
                "output_root": str(root),
                "prepared": contract_path(root).is_file(),
                "preflight": preflight_path(root).is_file(),
                "fine_scheduler": fine_scheduler_path(root).is_file(),
                "fine_training_sealed": fine_training_seal_path(root).is_file(),
                "fine_analysis_candidate": (
                    root / "analysis/fine_analysis_completion.json"
                ).is_file(),
                "training_sealed": training_seal_path(root).is_file(),
                "analysis_sealed": (root / "analysis/analysis_completion.json").is_file(),
                "states": states,
                "certified_physical_fits": states["completed"] * (N_FOLDS + 1),
                "remaining_physical_fits": (TOTAL_CHAINS - states["completed"]) * (N_FOLDS + 1),
                "rows": rows,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", parents=[common])
    plan.set_defaults(func=cmd_plan)

    prepare = sub.add_parser("prepare", parents=[common])
    prepare.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    prepare.add_argument("--apply", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    preflight = sub.add_parser("preflight", parents=[common])
    preflight.add_argument("--apply", action="store_true")
    preflight.set_defaults(func=cmd_preflight)

    train = sub.add_parser("train", parents=[common])
    train.add_argument("--apply", action="store_true")
    train.add_argument("--jobs", type=int, default=MAX_JOBS)
    train.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    train.add_argument("--kind", choices=("fine", "controls", "fixed", "repeated"))
    train.add_argument("--model-seed", type=int, choices=MODEL_SEEDS)
    train.set_defaults(func=cmd_train)

    validate = sub.add_parser("validate", parents=[common])
    validate.add_argument("--seal", action="store_true")
    validate.set_defaults(func=cmd_validate)

    validate_fine = sub.add_parser("validate-fine", parents=[common])
    validate_fine.add_argument("--seal", action="store_true")
    validate_fine.set_defaults(func=cmd_validate_fine)

    analyze = sub.add_parser("analyze", parents=[common])
    analyze.add_argument("--apply", action="store_true")
    analyze.set_defaults(func=cmd_analyze)

    analyze_fine = sub.add_parser("analyze-fine", parents=[common])
    analyze_fine.add_argument("--apply", action="store_true")
    analyze_fine.set_defaults(func=cmd_analyze_fine)

    verify = sub.add_parser("verify", parents=[common])
    verify.set_defaults(func=cmd_verify)

    status = sub.add_parser("status", parents=[common])
    status.set_defaults(func=cmd_status)

    internal = sub.add_parser("_train-one", parents=[common])
    internal.add_argument("--job-key", required=True)
    internal.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    internal.set_defaults(func=cmd_train_one)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
