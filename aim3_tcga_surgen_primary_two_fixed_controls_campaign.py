#!/usr/bin/env python3
"""Narrow governed Aim-3 continuation for two canonical matched-WT controls.

This append-only continuation answers only the two molecular-resolution
contrasts needed by the Results paragraph:

* codon-12 versus its one predeclared, size/fold/subcohort-matched KRAS-WT
  control; and
* G12D-broad versus its corresponding canonical matched-WT control.

The five completed fine-task chains per rung and every input authority are
read from the certified TCGA+SurGen-primary campaign.  Training is delegated
verbatim to that campaign's frozen ``_train-one`` entry point for exactly ten
fixed-control chains (seeds 42..46).  The frozen entry point necessarily
places its immutable request, log, receipt, and train artifacts in the two
previously reserved ``train/fixed`` namespaces of the certified campaign.
All continuation governance, mirrored receipts, scheduler evidence, patient
ensembles, bootstrap distributions, and results live in a new sibling root.

No repeated-WT job and no other fixed control is schedulable here.  The output
is a conditional canonical matched-control contrast, not a repeated-WT
consensus or a biological/theoretical performance bound.

Production workflow (mutating stages require ``--apply`` or ``--seal``)::

    python aim3_tcga_surgen_primary_two_fixed_controls_campaign.py plan
    python aim3_tcga_surgen_primary_two_fixed_controls_campaign.py prepare --apply
    python aim3_tcga_surgen_primary_two_fixed_controls_campaign.py preflight --apply
    python aim3_tcga_surgen_primary_two_fixed_controls_campaign.py train --jobs 6
    python aim3_tcga_surgen_primary_two_fixed_controls_campaign.py train --jobs 6 --apply
    python aim3_tcga_surgen_primary_two_fixed_controls_campaign.py validate --seal
    python aim3_tcga_surgen_primary_two_fixed_controls_campaign.py analyze --apply
    python aim3_tcga_surgen_primary_two_fixed_controls_campaign.py verify
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
CAMPAIGN = "aim3_tcga_surgen_primary_two_canonical_fixed_wt_controls"
SOURCE_CAMPAIGN = "aim3_tcga_surgen_primary_univ1_5seed"
SOURCE_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim3_tcga_surgen_primary_univ1_5seed_v2_20260827"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim3_tcga_surgen_primary_two_fixed_controls_v1_20260828"
)
RERUNS_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns")

FROZEN_CONTROLLER = SOURCE_ROOT / (
    "source_snapshot/aim3_tcga_surgen_primary_five_seed_campaign.py"
)
FOCUSED_TEST = REPO / "tests/test_aim3_tcga_surgen_primary_two_fixed_controls_campaign.py"

MODEL_SEEDS = (42, 43, 44, 45, 46)
FINE_TASKS = ("codon", "g12d_broad")
CONTROL_FOR = {
    "codon": "ctrl_codon",
    "g12d_broad": "ctrl_g12d_broad",
}
FIXED_WT_DRAW_SEEDS = {
    "codon": 20260818,
    "g12d_broad": 20260822,
}
N_FOLDS = 5
MAX_JOBS = 6
DEFAULT_NUM_WORKERS = 4
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260817
EXPECTED_CHAINS = len(FINE_TASKS) * len(MODEL_SEEDS)
EXPECTED_OOF_FITS = EXPECTED_CHAINS * N_FOLDS
EXPECTED_REFITS = EXPECTED_CHAINS
EXPECTED_PHYSICAL_FITS = EXPECTED_OOF_FITS + EXPECTED_REFITS

EXPECTED_PATIENTS = {
    "codon": {"positive": 354, "negative": 147, "patients": 501},
    "g12d_broad": {"positive": 154, "negative": 347, "patients": 501},
}
ALLOWED_COHORTS = frozenset({"TCGA", "SurGen"})
ALLOWED_SUBCOHORTS = frozenset({"TCGA-COAD", "TCGA-READ", "SR386", "SR1482"})

# These authorities predate and are never rewritten by this continuation.
PINNED_SOURCE_SHA256 = {
    "contract": "690f5714de88801dc341547f3d17013ee98bb258d3a9ae07c06190021c159a41",
    "preflight": "9ada242def0fa4e0cb7bed8270c247c38d80cd6f77b9eeac6e751c68c04b8f82",
    "fine_scheduler": "a4b9eb671823b0faaa84dafce5582b4775fb68fd043b65541f8e796d30538ad1",
    "fine_training_seal": "817aa481d74caed1eb592e672019ab0f0a769dd122ba93912f87e2921f2e77ee",
    "fine_analysis_completion": "8441ec14771186de3d59fe613c9fb1574aaab147b62eab3a30650bdec501a835",
    "fine_results": "477fb3ad25b9358aca879a35244914cc3babb0d95704a8c79b85d89700e6c1c1",
    "fine_patient_native_logits": "8bb9099f1ab38cead16968d7639fae1af7c2ef3d65621375c3309f8bfad0c32f",
    "fine_bootstrap_distributions": "9567da6b14f33fd4404d4b6b65cedbdfe5ffa412f71d06f2d5061c373fdc850f",
    "frozen_controller": "f75e467ea6f7c395c017e79c4e7030b174ab695c24f9094545c6cb98f10d6b31",
    "frozen_fixed_analysis": "c142afb938b01c19071ca200ff33dec27b0457c50ea2c8b9bc3574c9d946f004",
    "frozen_repeated_analysis": "187850f6286963dfebf992e38eededa2f05a349292158db98dc9ebe36d9315d1",
    "fine_manifest_codon": "85851b206e526c71408d0e264759b1ce553c39cdd5f39efd7b9a8ce5b35fb7d2",
    "control_manifest_codon": "03914cd60403f19d7fab45be0648a68891d3a98bb8f712a6091cb2b0327befb1",
    "fine_manifest_g12d_broad": "534f74d8e4b5e53c986130445cc750e1e5ae712cd8d6ee5a5c0982344eba2ca9",
    "control_manifest_g12d_broad": "dbabe63eb7e0ab7aae389cdc56913a01f2ee860c7655e64da3d2c0d5a481fbae",
    "control_split_integrity_codon": "a06b3fac13c9427ea0b665badc02aa94904468ddff2056059815379a3acb0350",
    "control_splits_codon": "6b8c8a5823c487d43f2fb9198c9ef1eaf2c07eac4a119c1f345524e56eaedbf4",
    "control_split_integrity_g12d_broad": "9fd42a96718e19c7ad47f153389052e5639e4170960fe5378dee127eaae0e863",
    "control_splits_g12d_broad": "7ccff1db3a614133a0777a5f1e5d6dd0a1b2b2fbbe1f0889b0fb2232a2a8da66",
}


class ContractError(RuntimeError):
    """A frozen-input, scope, scheduler, or numerical contract failed."""


@dataclass(frozen=True, order=True)
class Job:
    fine_task: str
    model_seed: int

    def __post_init__(self) -> None:
        if self.fine_task not in FINE_TASKS:
            raise ValueError(f"unsupported fine task: {self.fine_task}")
        if self.model_seed not in MODEL_SEEDS:
            raise ValueError(f"unsupported model seed: {self.model_seed}")

    @property
    def control_task(self) -> str:
        return CONTROL_FOR[self.fine_task]

    @property
    def key(self) -> str:
        return f"fixed__{self.control_task}__seed{self.model_seed}"

    @property
    def source_job_id(self) -> str:
        return f"aim3.source.{self.key}"


def job_inventory() -> list[Job]:
    return [Job(task, seed) for task in FINE_TASKS for seed in MODEL_SEEDS]


def fit_accounting() -> dict[str, int]:
    return {
        "task_variants": len(FINE_TASKS),
        "chains": EXPECTED_CHAINS,
        "oof_folds": EXPECTED_OOF_FITS,
        "p75_refits": EXPECTED_REFITS,
        "physical_mil_fits": EXPECTED_PHYSICAL_FITS,
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
    if path.is_symlink():
        raise ContractError(f"required artifact is symlinked: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ContractError(f"required regular artifact missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid strict JSON artifact: {path}") from exc
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


def _write_json_once(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"
    _write_bytes_once(path, payload.encode("utf-8"))


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    _write_bytes_once(path, buffer.getvalue())


def _write_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _write_bytes_once(path, buffer.getvalue())


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_symlink_ancestry(path: Path) -> None:
    cursor = path.absolute()
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise ContractError(f"path contains a symlink component: {cursor}")
        cursor = cursor.parent


def validate_output_root(root: Path, *, must_exist: bool | None = None) -> Path:
    raw = Path(root).expanduser()
    if not raw.is_absolute():
        raise ContractError("--output-root must be absolute")
    _reject_symlink_ancestry(raw)
    lexical = raw.absolute()
    resolved = raw.resolve(strict=False)
    if lexical != resolved:
        raise ContractError("--output-root must not traverse aliases or symlinks")
    source = SOURCE_ROOT.resolve(strict=False)
    if (
        resolved == source
        or _is_relative_to(resolved, source)
        or _is_relative_to(source, resolved)
    ):
        raise ContractError("continuation output must not overlap its certified source root")
    production = DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    temporary = Path("/tmp").resolve()
    if resolved != production and not _is_relative_to(resolved, temporary):
        raise ContractError(f"production output root must be exactly {production}")
    if resolved == production and not _is_relative_to(resolved, RERUNS_ROOT.resolve()):
        raise ContractError("production output must remain below the governed reruns root")
    if must_exist is True and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if must_exist is False and (resolved.exists() or resolved.is_symlink()):
        raise FileExistsError(resolved)
    return resolved


def source_authority_paths(source_root: Path = SOURCE_ROOT) -> dict[str, Path]:
    root = Path(source_root)
    values = {
        "contract": root / "contract.json",
        "preflight": root / "receipts/preflight.json",
        "fine_scheduler": root / "receipts/scheduler_fine.json",
        "fine_training_seal": root / "receipts/training_complete_fine.json",
        "fine_analysis_completion": root / "analysis/fine_analysis_completion.json",
        "fine_results": root / "analysis/fine_results.json",
        "fine_patient_native_logits": root / "analysis/fine_patient_native_logits.parquet",
        "fine_bootstrap_distributions": root / "analysis/fine_bootstrap_distributions.npz",
        "frozen_controller": root / "source_snapshot/aim3_tcga_surgen_primary_five_seed_campaign.py",
        "frozen_fixed_analysis": root / "source_snapshot/e3_fixed_control_analysis.py",
        "frozen_repeated_analysis": root / "source_snapshot/e3_repeated_control_campaign.py",
    }
    for fine in FINE_TASKS:
        values[f"fine_manifest_{fine}"] = root / "inputs/manifests" / f"fine__{fine}.csv"
        values[f"control_manifest_{fine}"] = (
            root / "inputs/manifests" / f"fixed__{CONTROL_FOR[fine]}.csv"
        )
        split = root / "inputs/splits" / f"fixed__{CONTROL_FOR[fine]}" / "aim1_balanced5"
        values[f"control_split_integrity_{fine}"] = split / ".integrity_hash"
        values[f"control_splits_{fine}"] = split / "splits.parquet"
    return values


def _validate_source_authorities(source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    root = Path(source_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ContractError(f"certified source root is invalid: {root}")
    artifacts = {name: identity(path) for name, path in source_authority_paths(root).items()}
    if root == SOURCE_ROOT.resolve():
        observed = {name: record["sha256"] for name, record in artifacts.items()}
        if observed != PINNED_SOURCE_SHA256:
            mismatch = {
                key: {"expected": PINNED_SOURCE_SHA256.get(key), "observed": value}
                for key, value in observed.items()
                if PINNED_SOURCE_SHA256.get(key) != value
            }
            raise ContractError(f"certified source authority identities drifted: {mismatch}")
    contract = _read_json(root / "contract.json")
    fine_seal = _read_json(root / "receipts/training_complete_fine.json")
    fine_analysis = _read_json(root / "analysis/fine_analysis_completion.json")
    if (
        contract.get("campaign") != SOURCE_CAMPAIGN
        or fine_seal.get("status") != "fine_phase_complete_and_certified"
        or fine_seal.get("population") != "tcga_surgen_primary"
        or fine_seal.get("model_seeds") != list(MODEL_SEEDS)
        or fine_seal.get("controls_status") != "not_started"
        or fine_analysis.get("status") != "fine_phase_complete_candidate_unsealed"
        or fine_analysis.get("controls_status") != "not_started"
    ):
        raise ContractError("certified fine-phase authority semantics drifted")
    return artifacts


def _load_frozen_controller(source_root: Path = SOURCE_ROOT) -> ModuleType:
    root = Path(source_root).resolve(strict=True)
    controller = root / "source_snapshot/aim3_tcga_surgen_primary_five_seed_campaign.py"
    if root == SOURCE_ROOT.resolve() and sha256_file(controller) != PINNED_SOURCE_SHA256["frozen_controller"]:
        raise ContractError("frozen source controller identity drifted")
    name = f"_aim3_two_fixed_frozen_{hashlib.sha256(str(root).encode()).hexdigest()[:12]}"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    specification = importlib.util.spec_from_file_location(name, controller)
    if specification is None or specification.loader is None:
        raise ContractError(f"cannot load frozen controller: {controller}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _manifest_path(source_root: Path, fine: str, *, control: bool) -> Path:
    name = f"fixed__{CONTROL_FOR[fine]}" if control else f"fine__{fine}"
    return source_root / "inputs/manifests" / f"{name}.csv"


def _patient_manifest(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = {
        "slide_id", "patient_id", "target_label", "cohort", "subcohort",
        "specimen_role", "k_fold", "kras",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractError(f"{label}: manifest lacks columns {missing}")
    if frame["slide_id"].isna().any() or frame["slide_id"].astype(str).duplicated().any():
        raise ContractError(f"{label}: slide identities are null or duplicated")
    constant = [
        "target_label", "cohort", "subcohort", "specimen_role", "k_fold", "kras"
    ]
    counts = frame.groupby("patient_id", sort=False)[constant].nunique(dropna=False)
    if counts.gt(1).any(axis=1).any():
        raise ContractError(f"{label}: task metadata varies within patient")
    return frame.sort_values("slide_id").drop_duplicates("patient_id").copy()


def validate_pair_manifests(
    fine_frame: pd.DataFrame,
    control_frame: pd.DataFrame,
    fine: str,
    *,
    production: bool,
) -> dict[str, Any]:
    fp = _patient_manifest(fine_frame, label=f"fine/{fine}")
    cp = _patient_manifest(control_frame, label=f"control/{fine}")
    for label, frame in (("fine", fine_frame), ("control", control_frame)):
        if frozenset(frame["cohort"].astype(str).unique()) != ALLOWED_COHORTS:
            raise ContractError(f"{label}/{fine}: development cohorts are not exactly TCGA+SurGen")
        if frozenset(frame["subcohort"].astype(str).unique()) != ALLOWED_SUBCOHORTS:
            raise ContractError(f"{label}/{fine}: source subcohort boundary drifted")
        roles = frozenset(frame["specimen_role"].astype(str).str.casefold().unique())
        if roles != {"primary"}:
            raise ContractError(f"{label}/{fine}: non-primary specimen entered development")
    for label, patients in (("fine", fp), ("control", cp)):
        labels = pd.to_numeric(patients["target_label"], errors="raise").astype(int)
        observed = {
            "positive": int(labels.eq(1).sum()),
            "negative": int(labels.eq(0).sum()),
            "patients": int(len(patients)),
        }
        if set(labels) != {0, 1}:
            raise ContractError(f"{label}/{fine}: binary classes are required")
        if production and observed != EXPECTED_PATIENTS[fine]:
            raise ContractError(f"{label}/{fine}: patient census drifted: {observed}")
    fine_labels = pd.to_numeric(fp["target_label"]).astype(int)
    control_labels = pd.to_numeric(cp["target_label"]).astype(int)
    fine_pos = fp.loc[fine_labels.eq(1)].set_index("patient_id").sort_index()
    control_pos = cp.loc[control_labels.eq(1)].set_index("patient_id").sort_index()
    if fine_pos.index.astype(str).tolist() != control_pos.index.astype(str).tolist():
        raise ContractError(f"{fine}: positive patient roster is not exactly shared")
    if not fine_pos[["subcohort", "k_fold"]].astype(str).equals(
        control_pos[["subcohort", "k_fold"]].astype(str)
    ):
        raise ContractError(f"{fine}: shared-positive strata drifted")
    fine_neg = fp.loc[fine_labels.eq(0)].copy()
    control_neg = cp.loc[control_labels.eq(0)].copy()
    if set(fine_neg["patient_id"].astype(str)) & set(control_neg["patient_id"].astype(str)):
        raise ContractError(f"{fine}: fine and matched-WT negative rosters overlap")
    if not control_neg["kras"].astype(str).eq("wild_type").all():
        raise ContractError(f"{fine}: control negatives are not all KRAS WT")
    wanted = fine_neg.groupby(["subcohort", "k_fold"], sort=True).size()
    observed = control_neg.groupby(["subcohort", "k_fold"], sort=True).size()
    if not wanted.equals(observed):
        raise ContractError(f"{fine}: matched negative subcohort/fold cells drifted")
    return {
        "fine_slides": int(len(fine_frame)),
        "control_slides": int(len(control_frame)),
        "patients": int(len(fp)),
        "positive": int(fine_labels.eq(1).sum()),
        "negative": int(fine_labels.eq(0).sum()),
        "positive_roster_shared": True,
        "negative_rosters_disjoint": True,
        "negative_matching": "subcohort x frozen outer fold",
        "control_negative_kras": "wild_type",
        "development_cohorts": ["SurGen", "TCGA"],
        "specimen_role": "primary",
    }


def validate_selected_manifests(source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    root = Path(source_root).resolve(strict=True)
    production = root == SOURCE_ROOT.resolve()
    return {
        fine: validate_pair_manifests(
            pd.read_csv(_manifest_path(root, fine, control=False), low_memory=False),
            pd.read_csv(_manifest_path(root, fine, control=True), low_memory=False),
            fine,
            production=production,
        )
        for fine in FINE_TASKS
    }


def _source_run_dir(source_root: Path, job: Job) -> Path:
    return source_root / "train/fixed" / job.control_task / f"seed{job.model_seed}"


def _source_job_receipt(source_root: Path, job: Job) -> Path:
    return source_root / "receipts/jobs" / f"{job.key}.json"


def _source_job_artifact_paths(source_root: Path, job: Job) -> tuple[Path, ...]:
    return (
        _source_run_dir(source_root, job),
        source_root / "requests/jobs" / f"{job.key}.json",
        source_root / "logs/jobs" / f"{job.key}.log",
        source_root / "requests/failures" / f"{job.key}.json",
        _source_job_receipt(source_root, job),
        source_root / "state/hydra" / job.key,
    )


def _all_source_control_jobs(source_root: Path = SOURCE_ROOT) -> list[Any]:
    frozen = _load_frozen_controller(source_root)
    return [job for job in frozen.job_inventory() if job.kind in {"fixed", "repeated"}]


def assert_nonselected_source_controls_absent(source_root: Path = SOURCE_ROOT) -> None:
    root = Path(source_root).resolve(strict=True)
    selected = {job.key for job in job_inventory()}
    found: list[str] = []
    for source_job in _all_source_control_jobs(root):
        if source_job.key in selected:
            continue
        candidates = (
            root / "train" / source_job.kind / source_job.task
            / (f"wt{source_job.draw_seed}" if source_job.draw_seed is not None else "")
            / f"seed{source_job.model_seed}",
            root / "requests/jobs" / f"{source_job.key}.json",
            root / "logs/jobs" / f"{source_job.key}.log",
            root / "requests/failures" / f"{source_job.key}.json",
            root / "receipts/jobs" / f"{source_job.key}.json",
            root / "state/hydra" / source_job.key,
        )
        found.extend(str(path) for path in candidates if path.exists() or path.is_symlink())
    if found:
        raise ContractError(
            "nonselected fixed/repeated control artifacts are outside this narrow scope; "
            f"first={found[:5]}"
        )


def assert_selected_source_controls_pristine(source_root: Path = SOURCE_ROOT) -> None:
    root = Path(source_root).resolve(strict=True)
    found = [
        str(path)
        for job in job_inventory()
        for path in _source_job_artifact_paths(root, job)
        if path.exists() or path.is_symlink()
    ]
    if found:
        raise ContractError(
            "selected source fixed-control namespaces are not pristine; "
            f"first={found[:5]}"
        )


def _expected_frozen_records(
    source_root: Path = SOURCE_ROOT, *, num_workers: int = DEFAULT_NUM_WORKERS
) -> list[dict[str, Any]]:
    root = Path(source_root).resolve(strict=True)
    frozen = _load_frozen_controller(root)
    selected = {job.key for job in job_inventory()}
    records = [
        record
        for record in frozen.build_training_jobs(root, num_workers=num_workers)
        if record.get("job_key") in selected
    ]
    if [record.get("job_key") for record in records] != [job.key for job in job_inventory()]:
        raise ContractError("frozen selected-job order or roster drifted")
    controller = root / "source_snapshot/aim3_tcga_surgen_primary_five_seed_campaign.py"
    for job, record in zip(job_inventory(), records, strict=True):
        command = list(record.get("command", []))
        expected = [
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
        if command != expected:
            raise ContractError(f"{job.key}: frozen _train-one command drifted")
        if record.get("kind") != "fixed" or record.get("draw_seed") is not None:
            raise ContractError(f"{job.key}: selected command is not canonical fixed control")
        if Path(str(record.get("output", ""))).resolve(strict=False) != _source_run_dir(root, job):
            raise ContractError(f"{job.key}: frozen output route drifted")
    return records


def build_job_plan(
    source_root: Path = SOURCE_ROOT, *, num_workers: int = DEFAULT_NUM_WORKERS
) -> list[dict[str, Any]]:
    records = _expected_frozen_records(source_root, num_workers=num_workers)
    return [
        {
            "job_id": record["job_id"],
            "job_key": record["job_key"],
            "kind": "fixed",
            "fine_task": record["fine_task"],
            "control_task": record["task"],
            "model_seed": int(record["model_seed"]),
            "canonical_fixed_wt_draw_seed": FIXED_WT_DRAW_SEEDS[str(record["fine_task"])],
            "oof_folds": N_FOLDS,
            "p75_refits": 1,
            "physical_mil_fits": N_FOLDS + 1,
            "source_output": record["output"],
            "command": record["command"],
            "training_command": record["training_command"],
        }
        for record in records
    ]


def contract_path(root: Path) -> Path:
    return root / "contract.json"


def job_plan_path(root: Path) -> Path:
    return root / "jobs/job_plan.json"


def preflight_path(root: Path) -> Path:
    return root / "receipts/preflight.json"


def continuation_job_receipt_path(root: Path, job: Job) -> Path:
    return root / "receipts/jobs" / f"{job.key}.json"


def scheduler_path(root: Path) -> Path:
    return root / "receipts/scheduler.json"


def training_seal_path(root: Path) -> Path:
    return root / "receipts/training_complete.json"


def analysis_root(root: Path) -> Path:
    return root / "analysis"


def _contract_semantics(root: Path, source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "output_root": str(root.resolve(strict=False)),
        "source_campaign": SOURCE_CAMPAIGN,
        "source_root": str(Path(source_root).resolve(strict=True)),
        "population": "tcga_surgen_primary",
        "population_display": "TCGA + SurGen primaries",
        "encoder": "UNI-v1",
        "model_seeds": list(MODEL_SEEDS),
        "fine_tasks": list(FINE_TASKS),
        "control_tasks": [CONTROL_FOR[fine] for fine in FINE_TASKS],
        "control_kind": "canonical_fixed_matched_wt",
        "fixed_wt_draw_seeds": FIXED_WT_DRAW_SEEDS,
        "repeated_wt_controls_scheduled": False,
        "nonselected_fixed_controls_scheduled": False,
        "external_development_cohorts": [],
        "fit_accounting": fit_accounting(),
        "configured_max_parallel_chains": MAX_JOBS,
        "delegation": "exact frozen source-snapshot _train-one commands",
        "source_write_boundary": (
            "append-only selected fixed-control request/log/receipt/train/state namespaces only; "
            "all fine and input authorities remain read-only"
        ),
        "claim_scope": (
            "conditional contrasts against one canonical matched-WT draw per task; "
            "no repeated-WT consensus or performance-bound claim"
        ),
    }


def _prepare_payload(root: Path, source_root: Path = SOURCE_ROOT) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(source_root).resolve(strict=True)
    authorities = _validate_source_authorities(source)
    pair_census = validate_selected_manifests(source)
    assert_nonselected_source_controls_absent(source)
    assert_selected_source_controls_pristine(source)
    plan = build_job_plan(source)
    payload = {
        **_contract_semantics(root, source),
        "created_utc": _utcnow(),
        "source_authorities": authorities,
        "selected_pair_census": pair_census,
        "job_plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "controller": identity(Path(__file__)),
        "focused_test": identity(FOCUSED_TEST),
    }
    return payload, plan


def verify_contract(root: Path, *, deep: bool) -> dict[str, Any]:
    root = validate_output_root(root, must_exist=True)
    contract = _read_json(contract_path(root))
    source = Path(str(contract.get("source_root", ""))).resolve(strict=True)
    expected = _contract_semantics(root, source)
    mismatch = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ContractError(f"continuation contract semantics drifted: {mismatch}")
    if (
        contract.get("controller") != identity(Path(__file__))
        or contract.get("focused_test") != identity(FOCUSED_TEST)
    ):
        raise ContractError("continuation controller/test identities drifted")
    authorities = _validate_source_authorities(source)
    if contract.get("source_authorities") != authorities:
        raise ContractError("continuation source authorities drifted")
    census = validate_selected_manifests(source)
    if contract.get("selected_pair_census") != census:
        raise ContractError("selected pair census drifted")
    plan = _read_json(job_plan_path(root)).get("jobs")
    expected_plan = build_job_plan(source)
    if plan != expected_plan:
        raise ContractError("stored narrow job plan does not replay exactly")
    digest = hashlib.sha256(
        json.dumps(expected_plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if contract.get("job_plan_sha256") != digest:
        raise ContractError("job-plan digest drifted")
    if deep:
        frozen = _load_frozen_controller(source)
        frozen.verify_contract(source, deep=True)
    return contract


def cmd_plan(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root))
    plan = build_job_plan(SOURCE_ROOT, num_workers=args.num_workers)
    print(json.dumps({
        "status": "PLAN_ONLY_NO_WRITES",
        **_contract_semantics(root),
        "jobs": plan,
    }, indent=2, sort_keys=True))


def cmd_prepare(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=False)
    payload, plan = _prepare_payload(root)
    if not args.apply:
        print(json.dumps({"status": "DRY_RUN_NO_WRITES", **payload, "jobs": plan}, indent=2))
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    # ``mkdir(exist_ok=False)`` is the append-only publication boundary.  A
    # failure after this point leaves a partial evidence root to adjudicate;
    # this controller never repairs or replaces it in place.
    root.mkdir(exist_ok=False)
    _write_json_once(root / "jobs/job_plan.json", {"schema_version": SCHEMA_VERSION, "jobs": plan})
    _write_json_once(root / "contract.json", payload)
    verify_contract(root, deep=False)
    print(f"PASS — prepared narrow append-only continuation at {root}")


def cmd_preflight(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    contract = verify_contract(root, deep=True)
    source = Path(contract["source_root"])
    assert_nonselected_source_controls_absent(source)
    assert_selected_source_controls_pristine(source)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "created_utc": _utcnow(),
        "contract": identity(contract_path(root)),
        "job_plan": identity(job_plan_path(root)),
        "source_authorities": contract["source_authorities"],
        "selected_pair_census": contract["selected_pair_census"],
        "selected_source_namespaces_pristine": True,
        "nonselected_source_control_artifacts_absent": True,
        "jobs": EXPECTED_CHAINS,
        "fit_accounting": fit_accounting(),
        "configured_max_parallel_chains": MAX_JOBS,
    }
    if args.apply:
        _write_json_once(preflight_path(root), payload)
    print(json.dumps({**payload, "persisted": bool(args.apply)}, indent=2))


def _verify_preflight(root: Path) -> dict[str, Any]:
    receipt = _read_json(preflight_path(root))
    expected = {
        "status": "PASS",
        "contract": identity(contract_path(root)),
        "job_plan": identity(job_plan_path(root)),
        "selected_source_namespaces_pristine": True,
        "nonselected_source_control_artifacts_absent": True,
        "jobs": EXPECTED_CHAINS,
        "fit_accounting": fit_accounting(),
        "configured_max_parallel_chains": MAX_JOBS,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(f"preflight receipt drifted at {key}")
    return receipt


def _validate_source_job(source_root: Path, job: Job) -> dict[str, Any]:
    frozen = _load_frozen_controller(source_root)
    source_job = frozen.Job("fixed", job.fine_task, job.model_seed)
    return frozen.validate_job(source_root, source_job)


def _continuation_job_payload(root: Path, source_root: Path, job: Job, command: Sequence[str]) -> dict[str, Any]:
    receipt = _validate_source_job(source_root, job)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_and_validated",
        "job_key": job.key,
        "fine_task": job.fine_task,
        "control_task": job.control_task,
        "model_seed": job.model_seed,
        "oof_folds": N_FOLDS,
        "p75_refits": 1,
        "physical_mil_fits": N_FOLDS + 1,
        "command": list(command),
        "continuation_contract": identity(contract_path(root)),
        "source_job_receipt": identity(_source_job_receipt(source_root, job)),
        "source_run_artifacts": receipt["artifacts"],
    }


def _validate_continuation_job(root: Path, source_root: Path, job: Job, command: Sequence[str]) -> dict[str, Any]:
    observed = _read_json(continuation_job_receipt_path(root, job))
    expected = _continuation_job_payload(root, source_root, job, command)
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ContractError(f"{job.key}: continuation receipt drifted at {key}")
    return observed


def _run_one(record: Mapping[str, Any]) -> dict[str, Any]:
    started_wall = _utcnow()
    started = time.monotonic()
    process = subprocess.Popen(list(record["command"]), cwd=REPO)
    code = int(process.wait())
    return {
        "job_id": str(record["job_id"]),
        "job_key": str(record["job_key"]),
        "pid": int(process.pid),
        "returncode": code,
        "started_utc": started_wall,
        "finished_utc": _utcnow(),
        "started_monotonic": started,
        "finished_monotonic": time.monotonic(),
        "command": list(record["command"]),
    }


def _peak_parallel(events: Sequence[Mapping[str, Any]]) -> int:
    points: list[tuple[float, int]] = []
    for event in events:
        points.extend([
            (float(event["started_monotonic"]), 1),
            (float(event["finished_monotonic"]), -1),
        ])
    active = peak = 0
    for _when, delta in sorted(points, key=lambda item: (item[0], -item[1])):
        active += delta
        peak = max(peak, active)
    return peak


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


def _launcher_lock(root: Path) -> Path:
    token = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    return Path(f"/tmp/{CAMPAIGN}_{token}.lock")


def _plan_records(root: Path) -> list[dict[str, Any]]:
    value = _read_json(job_plan_path(root)).get("jobs")
    if not isinstance(value, list):
        raise ContractError("job plan lacks a jobs list")
    return value


def cmd_train(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    contract = verify_contract(root, deep=False)
    _verify_preflight(root)
    source = Path(contract["source_root"])
    records = _plan_records(root)
    if not args.apply:
        print(
            f"DRY RUN: {len(records)} fixed-control chains / {EXPECTED_PHYSICAL_FITS} "
            f"physical MIL fits; max parallel={args.jobs}"
        )
        for record in records:
            print(json.dumps(record, sort_keys=True))
        return
    if not 1 <= args.jobs <= MAX_JOBS:
        raise ContractError(f"--jobs must be in 1..{MAX_JOBS}")
    if root == DEFAULT_OUTPUT_ROOT.resolve() and args.jobs != MAX_JOBS:
        raise ContractError(f"production requires exactly --jobs {MAX_JOBS}")
    if args.num_workers != DEFAULT_NUM_WORKERS:
        raise ContractError(f"this frozen plan requires --num-workers {DEFAULT_NUM_WORKERS}")
    if [record["command"] for record in records] != [
        record["command"] for record in build_job_plan(source, num_workers=args.num_workers)
    ]:
        raise ContractError("scheduler commands differ from the frozen exact plan")
    assert_nonselected_source_controls_absent(source)
    if scheduler_path(root).is_file():
        _validate_scheduler(root)
        print("PASS cached — narrow scheduler already sealed")
        return
    events: list[dict[str, Any]] = []
    failures: list[tuple[str, int]] = []
    with (
        _exclusive_lock(_launcher_lock(root), label="narrow fixed-control launcher"),
        ThreadPoolExecutor(max_workers=args.jobs) as pool,
    ):
        futures = {pool.submit(_run_one, record): record for record in records}
        for future in as_completed(futures):
            event = future.result()
            events.append(event)
            record = futures[future]
            if event["returncode"]:
                failures.append((event["job_key"], int(event["returncode"])))
            else:
                job = next(item for item in job_inventory() if item.key == event["job_key"])
                path = continuation_job_receipt_path(root, job)
                payload = _continuation_job_payload(root, source, job, record["command"])
                if path.is_file():
                    _validate_continuation_job(root, source, job, record["command"])
                else:
                    _write_json_once(path, {**payload, "created_utc": _utcnow()})
                    _validate_continuation_job(root, source, job, record["command"])
            print(f"{event['job_key']}: rc={event['returncode']}", flush=True)
    if failures:
        raise SystemExit(f"fixed-control training failures; evidence retained: {failures}")
    assert_nonselected_source_controls_absent(source)
    peak = _peak_parallel(events)
    if peak > MAX_JOBS or (root == DEFAULT_OUTPUT_ROOT.resolve() and peak != MAX_JOBS):
        raise ContractError(f"scheduler concurrency evidence invalid: peak={peak}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_rc0",
        "created_utc": _utcnow(),
        "contract": identity(contract_path(root)),
        "job_plan": identity(job_plan_path(root)),
        "preflight": identity(preflight_path(root)),
        "configured_max_parallel_chains": args.jobs,
        "observed_max_parallel_chains": peak,
        "job_count": len(events),
        "fit_accounting": fit_accounting(),
        "job_keys": [job.key for job in job_inventory()],
        "commands_exactly_frozen_train_one": True,
        "nonselected_fixed_or_repeated_jobs_dispatched": 0,
        "nonselected_source_control_artifacts_absent_at_completion": True,
        "events": sorted(events, key=lambda value: value["job_key"]),
    }
    _write_json_once(scheduler_path(root), payload)
    _validate_scheduler(root)
    print("PASS — ten canonical fixed-control chains completed")


def _validate_scheduler(root: Path) -> dict[str, Any]:
    contract = verify_contract(root, deep=False)
    source = Path(contract["source_root"])
    receipt = _read_json(scheduler_path(root))
    records = _plan_records(root)
    events = receipt.get("events")
    expected_commands = {record["job_key"]: record["command"] for record in records}
    if (
        receipt.get("status") != "completed_rc0"
        or receipt.get("contract") != identity(contract_path(root))
        or receipt.get("job_plan") != identity(job_plan_path(root))
        or receipt.get("preflight") != identity(preflight_path(root))
        or receipt.get("job_count") != EXPECTED_CHAINS
        or receipt.get("fit_accounting") != fit_accounting()
        or not 1 <= int(receipt.get("configured_max_parallel_chains", 0)) <= MAX_JOBS
        or receipt.get("job_keys") != [job.key for job in job_inventory()]
        or receipt.get("commands_exactly_frozen_train_one") is not True
        or receipt.get("nonselected_fixed_or_repeated_jobs_dispatched") != 0
        or receipt.get("nonselected_source_control_artifacts_absent_at_completion")
        is not True
        or not isinstance(events, list)
        or len(events) != EXPECTED_CHAINS
    ):
        raise ContractError("narrow scheduler receipt semantics are invalid")
    if not 1 <= int(receipt.get("observed_max_parallel_chains", 0)) <= MAX_JOBS:
        raise ContractError("narrow scheduler peak is invalid")
    replayed_peak = _peak_parallel(events)
    if replayed_peak != int(receipt["observed_max_parallel_chains"]):
        raise ContractError("narrow scheduler peak does not replay from event intervals")
    if root == DEFAULT_OUTPUT_ROOT.resolve() and (
        int(receipt["configured_max_parallel_chains"]) != MAX_JOBS
        or replayed_peak != MAX_JOBS
    ):
        raise ContractError("production scheduler did not demonstrate exact six-way execution")
    seen: set[str] = set()
    for event in events:
        key = str(event.get("job_key", ""))
        if key in seen or key not in expected_commands:
            raise ContractError("scheduler event roster contains duplicate/unknown job")
        seen.add(key)
        if event.get("returncode") != 0 or event.get("command") != expected_commands[key]:
            raise ContractError(f"scheduler event failed exact command contract: {key}")
    if seen != set(expected_commands):
        raise ContractError("scheduler event roster is incomplete")
    for job, record in zip(job_inventory(), records, strict=True):
        _validate_continuation_job(root, source, job, record["command"])
    return receipt


def cmd_validate(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    contract = verify_contract(root, deep=True)
    _verify_preflight(root)
    _validate_scheduler(root)
    source = Path(contract["source_root"])
    assert_nonselected_source_controls_absent(source)
    records = _plan_records(root)
    wrappers = []
    source_receipts = []
    fingerprints: set[str] = set()
    for job, record in zip(job_inventory(), records, strict=True):
        wrapper = _validate_continuation_job(root, source, job, record["command"])
        wrappers.append(identity(continuation_job_receipt_path(root, job)))
        source_receipts.append(wrapper["source_job_receipt"])
        fingerprints.add(wrapper["source_run_artifacts"]["training_identity"]["sha256"])
    if len(fingerprints) != EXPECTED_CHAINS:
        raise ContractError("ten independently identified source training chains are required")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_and_certified",
        "created_utc": _utcnow(),
        "campaign": CAMPAIGN,
        "population": "tcga_surgen_primary",
        "encoder": "UNI-v1",
        "model_seeds": list(MODEL_SEEDS),
        "fine_tasks": list(FINE_TASKS),
        "control_kind": "canonical_fixed_matched_wt",
        "repeated_wt_controls_scheduled": False,
        "external_development_cohorts": [],
        "fit_accounting": fit_accounting(),
        "contract": identity(contract_path(root)),
        "preflight": identity(preflight_path(root)),
        "scheduler": identity(scheduler_path(root)),
        "nonselected_source_control_artifacts_absent_at_seal": True,
        "continuation_job_receipts": wrappers,
        "source_job_receipts": source_receipts,
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
        "fine_tasks": list(FINE_TASKS),
        "control_kind": "canonical_fixed_matched_wt",
        "repeated_wt_controls_scheduled": False,
        "external_development_cohorts": [],
        "fit_accounting": fit_accounting(),
        "contract": identity(contract_path(root)),
        "preflight": identity(preflight_path(root)),
        "scheduler": identity(scheduler_path(root)),
        "nonselected_source_control_artifacts_absent_at_seal": True,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(f"training seal drifted at {key}")
    if (
        len(receipt.get("continuation_job_receipts", [])) != EXPECTED_CHAINS
        or len(receipt.get("source_job_receipts", [])) != EXPECTED_CHAINS
    ):
        raise ContractError("training seal does not bind exactly ten job receipts")
    contract = verify_contract(root, deep=False)
    source = Path(contract["source_root"])
    records = _plan_records(root)
    wrappers = []
    source_receipts = []
    for job, record in zip(job_inventory(), records, strict=True):
        wrapper = _validate_continuation_job(root, source, job, record["command"])
        wrappers.append(identity(continuation_job_receipt_path(root, job)))
        source_receipts.append(wrapper["source_job_receipt"])
    if (
        receipt.get("continuation_job_receipts") != wrappers
        or receipt.get("source_job_receipts") != source_receipts
    ):
        raise ContractError("training seal job-receipt identities do not replay")
    return receipt


def _summary(point: float, values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.shape != (N_BOOTSTRAP,) or not np.isfinite(array).all():
        raise ContractError("bootstrap distribution shape/values are invalid")
    return {
        "estimate": float(point),
        "ci95_two_sided": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
    }


def _patient_export(frame: pd.DataFrame, *, section: str, fine: str, task: str) -> pd.DataFrame:
    columns = [
        column for column in (
            "patient_id", "label", "mean_logit", "cohort", "subcohort", "k_fold", "n_slides"
        ) if column in frame
    ]
    out = frame[columns].copy()
    out.insert(0, "task", task)
    out.insert(0, "rung", fine)
    out.insert(0, "section", section)
    return out


def compute_analysis(
    source_root: Path = SOURCE_ROOT,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame]:
    if n_bootstrap != N_BOOTSTRAP:
        raise ContractError(f"governed analysis requires exactly {N_BOOTSTRAP} bootstraps")
    source = Path(source_root).resolve(strict=True)
    frozen = _load_frozen_controller(source)
    fine_results = _read_json(source / "analysis/fine_results.json")
    rungs: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    patients: list[pd.DataFrame] = []
    bindings: list[dict[str, Any]] = []
    for fine in FINE_TASKS:
        fine_frame, fine_seeds = frozen._ensemble(source, "fine", fine)
        control_frame, control_seeds = frozen._ensemble(source, "fixed", fine)
        seed = frozen.fixed_stats._stable_seed(
            BOOTSTRAP_SEED, "tcga_surgen_primary_fixed", fine
        )
        boot = frozen.repeated_stats.partial_paired_bootstrap(
            fine_frame,
            control_frame,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        fine_point = float(boot["point"]["fine"])
        control_point = float(boot["point"]["control"])
        recorded_fine = float(
            fine_results["fine"]["rungs"][fine]["five_seed_ensemble"]["estimate"]
        )
        if fine_point != recorded_fine:
            raise ContractError(f"{fine}: frozen fine ensemble does not replay fine analysis")
        fine_summary = _summary(fine_point, boot["fine_values"])
        control_summary = _summary(control_point, boot["control_values"])
        delta_summary = _summary(control_point - fine_point, boot["delta_values"])
        arrays[f"{fine}__fine_auroc"] = np.asarray(boot["fine_values"], dtype=float)
        arrays[f"{fine}__control_auroc"] = np.asarray(boot["control_values"], dtype=float)
        arrays[f"{fine}__delta_control_minus_fine"] = np.asarray(boot["delta_values"], dtype=float)
        patients.extend([
            _patient_export(fine_frame, section="fine", fine=fine, task=fine),
            _patient_export(
                control_frame,
                section="canonical_fixed_matched_wt",
                fine=fine,
                task=CONTROL_FOR[fine],
            ),
        ])
        rungs[fine] = {
            "fine_task": fine,
            "control_task": CONTROL_FOR[fine],
            "canonical_fixed_wt_draw_seed": FIXED_WT_DRAW_SEEDS[fine],
            "fine_per_seed_auroc": fine_seeds,
            "control_per_seed_auroc": control_seeds,
            "fine_five_seed_ensemble_auroc": fine_summary,
            "control_five_seed_ensemble_auroc": control_summary,
            "delta_control_minus_fine_auroc": delta_summary,
            "patients_per_arm": int(len(fine_frame)),
            "positive_per_arm": int(fine_frame["label"].eq(1).sum()),
            "negative_per_arm": int(fine_frame["label"].eq(0).sum()),
            "bootstrap": {
                "replicates": n_bootstrap,
                "seed": int(seed),
                "design": str(boot["design"]),
            },
            "claim": "conditional canonical fixed matched-WT contrast only",
        }
        for metric, value in (
            ("fine_auroc", fine_point),
            ("control_auroc", control_point),
            ("delta_control_minus_fine_auroc", control_point - fine_point),
        ):
            bindings.append({
                "id": f"aim3_source.canonical_fixed.{fine}.{metric}",
                "rung": fine,
                "metric": metric,
                "value": float(value),
            })
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "campaign": CAMPAIGN,
        "population": "tcga_surgen_primary",
        "population_display": "TCGA + SurGen primaries",
        "encoder": "UNI-v1",
        "model_seeds": list(MODEL_SEEDS),
        "control_kind": "canonical_fixed_matched_wt",
        "fixed_wt_draw_seeds": FIXED_WT_DRAW_SEEDS,
        "repeated_wt_controls_scheduled": False,
        "external_development_cohorts": [],
        "aggregation": "native slide logits averaged within patient, then across five model seeds",
        "fit_accounting": fit_accounting(),
        "canonical_fixed_matched_wt": {"rungs": rungs},
        "report_bindings": sorted(bindings, key=lambda item: item["id"]),
        "interpretation_boundary": (
            "Each result is conditional on one predeclared matched-WT draw; it is not a "
            "repeated-WT consensus or a biological/theoretical performance bound."
        ),
    }
    return report, arrays, pd.concat(patients, ignore_index=True)


def cmd_analyze(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    contract = verify_contract(root, deep=False)
    _validate_training_seal(root)
    source = Path(contract["source_root"])
    if not args.apply:
        print("DRY RUN — analysis requires --apply; no files were written")
        return
    report, arrays, patients = compute_analysis(source)
    directory = analysis_root(root)
    _write_parquet_once(directory / "patient_native_logits.parquet", patients)
    _write_npz_once(directory / "bootstrap_distributions.npz", arrays)
    report.update({
        "created_utc": _utcnow(),
        "contract": identity(contract_path(root)),
        "training_seal": identity(training_seal_path(root)),
        "source_fine_analysis": identity(source / "analysis/fine_analysis_completion.json"),
        "patient_native_logits": identity(directory / "patient_native_logits.parquet"),
        "bootstrap_distributions": identity(directory / "bootstrap_distributions.npz"),
    })
    _write_json_once(directory / "results.json", report)
    _write_json_once(directory / "analysis_completion.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_and_verified",
        "created_utc": _utcnow(),
        "results": identity(directory / "results.json"),
        "patient_native_logits": identity(directory / "patient_native_logits.parquet"),
        "bootstrap_distributions": identity(directory / "bootstrap_distributions.npz"),
        "report_binding_count": len(report["report_bindings"]),
        "repeated_wt_controls_scheduled": False,
    })
    _validate_analysis(root, numerical_replay=True)
    print(f"PASS — narrow canonical fixed-control analysis sealed at {directory}")


def _compare_report_core(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    dynamic = {
        "created_utc", "contract", "training_seal", "source_fine_analysis",
        "patient_native_logits", "bootstrap_distributions",
    }
    left = {key: value for key, value in observed.items() if key not in dynamic}
    right = {key: value for key, value in expected.items() if key not in dynamic}
    if left != right:
        raise ContractError("analysis results do not replay from primitive patient OOF data")


def _validate_analysis(root: Path, *, numerical_replay: bool) -> dict[str, Any]:
    contract = verify_contract(root, deep=False)
    _validate_training_seal(root)
    source = Path(contract["source_root"])
    directory = analysis_root(root)
    completion = _read_json(directory / "analysis_completion.json")
    results = _read_json(directory / "results.json")
    if (
        completion.get("status") != "complete_and_verified"
        or completion.get("results") != identity(directory / "results.json")
        or completion.get("patient_native_logits") != identity(directory / "patient_native_logits.parquet")
        or completion.get("bootstrap_distributions") != identity(directory / "bootstrap_distributions.npz")
        or completion.get("report_binding_count") != len(FINE_TASKS) * 3
        or completion.get("repeated_wt_controls_scheduled") is not False
        or results.get("status") != "complete"
        or results.get("population") != "tcga_surgen_primary"
        or results.get("external_development_cohorts") != []
        or results.get("control_kind") != "canonical_fixed_matched_wt"
        or results.get("repeated_wt_controls_scheduled") is not False
        or results.get("fit_accounting") != fit_accounting()
        or set(results.get("canonical_fixed_matched_wt", {}).get("rungs", {})) != set(FINE_TASKS)
        or len(results.get("report_bindings", [])) != len(FINE_TASKS) * 3
    ):
        raise ContractError("analysis completion/results semantics are invalid")
    forbidden = {"consensus_verdict", "all_three_draws_required", "ceiling"}
    if forbidden & set(_walk_keys(results)):
        raise ContractError("narrow result contains a repeated-control/performance-bound claim")
    if numerical_replay:
        expected, arrays, patients = compute_analysis(source)
        _compare_report_core(results, expected)
        observed_patients = pd.read_parquet(directory / "patient_native_logits.parquet")
        pd.testing.assert_frame_equal(observed_patients, patients, check_exact=True)
        with np.load(directory / "bootstrap_distributions.npz") as observed:
            if set(observed.files) != set(arrays):
                raise ContractError("bootstrap array inventory drifted")
            for key, values in arrays.items():
                if not np.array_equal(observed[key], values, equal_nan=False):
                    raise ContractError(f"bootstrap array does not replay exactly: {key}")
    return results


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).casefold()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def cmd_verify(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    verify_contract(root, deep=True)
    _verify_preflight(root)
    _validate_scheduler(root)
    _validate_training_seal(root)
    results = _validate_analysis(root, numerical_replay=True)
    print(json.dumps({
        "status": "PASS",
        "campaign": CAMPAIGN,
        "output_root": str(root),
        "fit_accounting": fit_accounting(),
        "rungs": {
            fine: {
                "fine_auroc": results["canonical_fixed_matched_wt"]["rungs"][fine]["fine_five_seed_ensemble_auroc"]["estimate"],
                "control_auroc": results["canonical_fixed_matched_wt"]["rungs"][fine]["control_five_seed_ensemble_auroc"]["estimate"],
            }
            for fine in FINE_TASKS
        },
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", parents=[common])
    plan.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    plan.set_defaults(func=cmd_plan)

    prepare = sub.add_parser("prepare", parents=[common])
    prepare.add_argument("--apply", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    preflight = sub.add_parser("preflight", parents=[common])
    preflight.add_argument("--apply", action="store_true")
    preflight.set_defaults(func=cmd_preflight)

    train = sub.add_parser("train", parents=[common])
    train.add_argument("--apply", action="store_true")
    train.add_argument("--jobs", type=int, default=MAX_JOBS)
    train.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    train.set_defaults(func=cmd_train)

    validate = sub.add_parser("validate", parents=[common])
    validate.add_argument("--seal", action="store_true")
    validate.set_defaults(func=cmd_validate)

    analyze = sub.add_parser("analyze", parents=[common])
    analyze.add_argument("--apply", action="store_true")
    analyze.set_defaults(func=cmd_analyze)

    verify = sub.add_parser("verify", parents=[common])
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
