#!/usr/bin/env python3
"""Governed zero-shot external scoring for the five Aim-3 fine tasks.

This additive campaign consumes only the certified fine phase of
``aim3_tcga_surgen_primary_univ1_5seed``.  It never writes to that training
root and performs no fit, refit, calibration, threshold selection, or model
selection.  Its public order is deliberately strict::

    python aim3_tcga_surgen_primary_fine_external_campaign.py plan
    python aim3_tcga_surgen_primary_fine_external_campaign.py prepare --apply
    python aim3_tcga_surgen_primary_fine_external_campaign.py preflight --apply
    python aim3_tcga_surgen_primary_fine_external_campaign.py score --apply --max-workers 6
    python aim3_tcga_surgen_primary_fine_external_campaign.py analyze --apply
    python aim3_tcga_surgen_primary_fine_external_campaign.py verify

``prepare`` snapshots five exact label-blind target rosters.  For each of the
five molecular tasks and five model seeds, ``score`` applies the authenticated
p75 full-source refit to one concatenated 479-slide roster.  It publishes an
exactly-once inference seal over 25 native-logit files (11,975 rows) before any
target outcome source can be opened.  ``analyze`` is the first outcome-opening
stage and derives the molecular task labels from KRAS subvariant tokens.

The external firewall is absolute for these models: CPTAC-primary, Orion,
RIH-primary, RIH-metastatic, and SurGen-metastatic are score-only targets.
They never enter fitting, refitting, adaptation, calibration, thresholding,
hyperparameter selection, checkpoint selection, encoder selection, or model
construction.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import aim3_tcga_surgen_primary_five_seed_campaign as source  # noqa: E402
import aim2_cross_protocol_transfer as cpht  # noqa: E402
import aim3_repeated_control_campaign as molecular  # noqa: E402

SCHEMA_VERSION = 1
CAMPAIGN = "aim3_tcga_surgen_primary_fine_external_univ1_5seed"
MODEL_SEEDS = (42, 43, 44, 45, 46)
TASKS = ("codon", "g12d_broad", "allele1", "allele2", "g12c")
TASK_DISPLAY = {
    "codon": "codon (G12 versus other KRAS mutants)",
    "g12d_broad": "G12D-broad (G12D versus every other KRAS mutant)",
    "allele1": "G12D-within-G12",
    "allele2": "G12V-within-G12",
    "g12c": "G12C-within-G12",
}
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_828
MAX_WORKERS = 6
DEFAULT_NUM_WORKERS = 4

TRAINING_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim3_tcga_surgen_primary_univ1_5seed_v2_20260827"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim3_tcga_surgen_primary_fine_external_v1_20260828"
)
RERUNS_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns")

FORBIDDEN_OUTCOME_COLUMNS = frozenset(
    {
        "label",
        "target_label",
        "kras",
        "kras_subvariant",
        "kras_status",
        "kras_mutant",
        "ras",
        "nras",
        "braf",
        "msi",
        "msi_dmmr",
        "outcome",
    }
)
ALLOWED_LABEL_BLIND_COLUMNS = frozenset(
    {
        "slide_id",
        "patient_id",
        "cohort",
        "subcohort",
        "specimen_role",
        "role",
        "patch_count",
        "mpp",
        "mpp_source",
        "exclude_neoadjuvant",
        "exclude_ambiguous_crc15",
        "liver_class",
    }
)


class ContractError(RuntimeError):
    """Fail-closed violation of the fine-external campaign contract."""


@dataclass(frozen=True)
class TargetSpec:
    key: str
    display: str
    role: str
    blind_source: Path
    blind_sha256: str
    outcome_source: Path
    outcome_sha256: str
    slides: int
    patients: int
    mutant_patients: int


TARGETS: dict[str, TargetSpec] = {
    item.key: item
    for item in (
        TargetSpec(
            "cptac_primary",
            "CPTAC-primary",
            "primary",
            Path(
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "final_v9_mil_5seed_expansion_v1_20260823/aim2_loco/inputs/"
                "label_blind/primary/family_cptac.csv"
            ),
            "135a454edbd94d2c27889946187faedf68312be0cbb6f93786987beae970eb5b",
            Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_cptac_primary.csv"),
            "f134ee9c087a12f6370982aa343778a03811ea077684f797bd8edde7853e69ad",
            98,
            94,
            33,
        ),
        TargetSpec(
            "orion_primary",
            "Orion-primary",
            "primary",
            Path(
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "aim2_final_v8_complete_v1_20260822/cpht/inputs/orion_primary.csv"
            ),
            "6acb9699c53944c1fa87b90a42e773eed830ef71594589de5d71005a1a31d024",
            Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v5.csv"),
            "bc83cd44a59f4ee0dbc672ab23087882449d80ad331c0d4304ecda96d0f79e0d",
            41,
            40,
            15,
        ),
        TargetSpec(
            "rih_primary",
            "RIH-primary",
            "primary",
            Path(
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "final_v9_mil_5seed_expansion_v1_20260823/aim2_loco/inputs/"
                "label_blind/primary/family_rih.csv"
            ),
            "0c33a3562da87a9843a41a34e6693e37825538aa5be0c31971798943f010a06e",
            Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_rih_primary.csv"),
            "0f72f5bc45327aab88368501f667e3f64e89db5066c954a02474ce009601c9c6",
            155,
            153,
            70,
        ),
        TargetSpec(
            "rih_metastatic",
            "RIH-metastatic",
            "metastatic",
            Path(
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "aim2_final_v8_complete_v1_20260822/e2met/inputs/manifests/rih_m.csv"
            ),
            "dec443d99e437e7cb3fc3297a4f98302081ba4455386e3b7d930ff142feb1e5c",
            Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_rih_metastatic.csv"),
            "6aaab722a96c2374296811f7bd4c79f44d842efe75788240274b14f79c3d1303",
            85,
            85,
            37,
        ),
        TargetSpec(
            "surgen_metastatic",
            "SurGen-metastatic",
            "metastatic",
            Path(
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "aim2_final_v8_complete_v1_20260822/e2met/inputs/manifests/sr1482_m.csv"
            ),
            "90eb00ef0baf6a33fc4b423affbce33dde1d38068add59cc136f870f13cf6d15",
            Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_surgen_metastatic.csv"),
            "3e69383c2869b1432fd6a1326c79fd5a6d50ec0c0dee88029c8dd433c9e06135",
            100,
            74,
            30,
        ),
    )
}
TARGET_ORDER = tuple(TARGETS)

RIH_DUAL_PATIENTS = frozenset(
    {
        "RIH:RIH_001216ba7a08c070",
        "RIH:RIH_1845b46a817ef51c",
        "RIH:RIH_24bda4bdbf9140a5",
        "RIH:RIH_28ed5c0131dcfa60",
        "RIH:RIH_3c68f85359d030c4",
        "RIH:RIH_59b36f4590fc4525",
        "RIH:RIH_9db23d204671f3e8",
        "RIH:RIH_c1bac72156d3e1a8",
    }
)

# Exact post-outcome task censes. Values are (positive, negative).
EXPECTED_PRIMITIVE_CENSUS: dict[str, dict[str, tuple[int, int]]] = {
    "codon": {
        "cptac_primary": (19, 14),
        "orion_primary": (12, 3),
        "rih_primary": (47, 23),
        "rih_metastatic": (23, 14),
        "surgen_metastatic": (19, 11),
    },
    "g12d_broad": {
        "cptac_primary": (11, 22),
        "orion_primary": (6, 9),
        "rih_primary": (21, 49),
        "rih_metastatic": (12, 25),
        "surgen_metastatic": (6, 24),
    },
    "allele1": {
        "cptac_primary": (11, 8),
        "orion_primary": (6, 6),
        "rih_primary": (21, 26),
        "rih_metastatic": (12, 11),
        "surgen_metastatic": (6, 13),
    },
    "allele2": {
        "cptac_primary": (6, 13),
        "orion_primary": (2, 10),
        "rih_primary": (17, 30),
        "rih_metastatic": (6, 17),
        "surgen_metastatic": (7, 12),
    },
    "g12c": {
        "cptac_primary": (2, 17),
        "orion_primary": (2, 10),
        "rih_primary": (5, 42),
        "rih_metastatic": (2, 21),
        "surgen_metastatic": (4, 15),
    },
}
EXPECTED_PRIMARY_CENSUS = {
    "codon": (78, 40),
    "g12d_broad": (38, 80),
    "allele1": (38, 40),
    "allele2": (25, 53),
    "g12c": (9, 69),
}
EXPECTED_METASTATIC_CENSUS = {
    "codon": (42, 25),
    "g12d_broad": (18, 49),
    "allele1": (18, 24),
    "allele2": (13, 29),
    "g12c": (6, 36),
}
EXPECTED_STRICT_DISJOINT_CENSUS = {
    "codon": (113, 63),
    "g12d_broad": (52, 124),
    "allele1": (52, 61),
    "allele2": (35, 78),
    "g12c": (15, 98),
}
EXPECTED_ALL_RECORD_CENSUS = {
    "codon": (120, 65),
    "g12d_broad": (56, 129),
    "allele1": (56, 64),
    "allele2": (38, 82),
    "g12c": (15, 105),
}
EXPECTED_ALL_RECORD_UNIQUE_CLUSTERS = {
    "codon": 181,
    "g12d_broad": 181,
    "allele1": 117,
    "allele2": 117,
    "g12c": 117,
}


@dataclass(frozen=True, order=True)
class ScoreJob:
    task: str
    seed: int

    def __post_init__(self) -> None:
        if self.task not in TASKS or self.seed not in MODEL_SEEDS:
            raise ValueError((self.task, self.seed))

    @property
    def key(self) -> str:
        return f"{self.task}__seed{self.seed}"


def score_jobs() -> list[ScoreJob]:
    return [ScoreJob(task, seed) for task in TASKS for seed in MODEL_SEEDS]


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _utcnow_precise() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError(f"required regular artifact missing or symlinked: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _verify_artifact(value: Mapping[str, Any], *, context: str) -> Path:
    path = Path(str(value.get("path", "")))
    if _artifact(path) != dict(value):
        raise ContractError(f"{context} identity drifted: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON artifact is not an object: {path}")
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
    _write_bytes_once(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n").encode(),
    )


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


def validate_output_root(path: Path, *, must_exist: bool | None = None) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ContractError("--output-root must be absolute")
    lexical = raw.absolute()
    resolved = raw.resolve(strict=False)
    if lexical != resolved:
        raise ContractError("output root must be normalized and must not traverse symlinks")
    cursor = raw
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise ContractError(f"output-root path traverses a symlink: {cursor}")
        cursor = cursor.parent
    production = DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    temporary = Path("/tmp").resolve()
    training = TRAINING_ROOT.resolve(strict=False)
    if (
        resolved == training
        or _is_relative_to(resolved, training)
        or _is_relative_to(training, resolved)
    ):
        raise ContractError("external output root must not overlap the Aim-3 training root")
    if resolved != production and not _is_relative_to(resolved, temporary):
        raise ContractError(
            f"production output root must be exactly {production}; tests may use /tmp"
        )
    if resolved == production and not _is_relative_to(resolved, RERUNS_ROOT.resolve()):
        raise ContractError("production output must remain under the governed reruns root")
    if must_exist is True and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if must_exist is False and (resolved.exists() or resolved.is_symlink()):
        raise FileExistsError(resolved)
    return resolved


def _validate_training_output_separation(output: Path, training_root: Path) -> None:
    external = Path(output).resolve(strict=False)
    training = Path(training_root).resolve(strict=True)
    if (
        external == training
        or _is_relative_to(external, training)
        or _is_relative_to(training, external)
    ):
        raise ContractError("external output root must not overlap its resolved training root")


def contract_path(root: Path) -> Path:
    return root / "contract.json"


def blind_path(root: Path, target: str) -> Path:
    return root / f"inputs/label_blind/{target}.csv"


def combined_blind_path(root: Path) -> Path:
    return root / "inputs/label_blind/all_external_479.csv"


def job_plan_path(root: Path) -> Path:
    return root / "jobs/score_jobs.json"


def preflight_path(root: Path) -> Path:
    return root / "receipts/deep_preflight.json"


def environment_path(root: Path) -> Path:
    return root / "inference/environment.json"


def score_path(root: Path, task: str, seed: int) -> Path:
    return root / f"scores/{task}/seed{seed}.parquet"


def score_receipt_path(root: Path, task: str, seed: int) -> Path:
    return score_path(root, task, seed).with_suffix(".receipt.json")


def scoring_completion_path(root: Path) -> Path:
    return root / "receipts/scoring_complete.json"


def inference_seal_path(root: Path) -> Path:
    return root / "inference/inference_seal.json"


def analysis_root(root: Path) -> Path:
    return root / "analysis"


def expected_output_inventory(root: Path) -> set[Path]:
    """Return the complete 67-file terminal namespace."""

    expected = {
        contract_path(root),
        job_plan_path(root),
        preflight_path(root),
        environment_path(root),
        scoring_completion_path(root),
        inference_seal_path(root),
        combined_blind_path(root),
        *(blind_path(root, target) for target in TARGET_ORDER),
        *(score_path(root, job.task, job.seed) for job in score_jobs()),
        *(score_receipt_path(root, job.task, job.seed) for job in score_jobs()),
        analysis_root(root) / "contract.json",
        analysis_root(root) / "patient_native_logits.parquet",
        analysis_root(root) / "bootstrap_distributions.npz",
        analysis_root(root) / "results.json",
        analysis_root(root) / "analysis_completion.json",
    }
    if len(expected) != 67:
        raise AssertionError(f"terminal output inventory is {len(expected)}, expected 67")
    return expected


def _validate_output_namespace(root: Path, *, terminal: str | None = None) -> dict[str, Any]:
    expected = expected_output_inventory(root)
    observed: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ContractError(f"external output namespace contains a symlink: {path}")
        if path.is_file():
            observed.add(path)
        elif not path.is_dir():
            raise ContractError(f"external output namespace contains a special path: {path}")
    unexpected = observed - expected
    if unexpected:
        raise ContractError(
            f"external output namespace has ungoverned files: {sorted(map(str, unexpected))[:5]}"
        )
    prepared = {
        contract_path(root),
        job_plan_path(root),
        combined_blind_path(root),
        *(blind_path(root, target) for target in TARGET_ORDER),
    }
    inference = expected - {
        analysis_root(root) / "contract.json",
        analysis_root(root) / "patient_native_logits.parquet",
        analysis_root(root) / "bootstrap_distributions.npz",
        analysis_root(root) / "results.json",
        analysis_root(root) / "analysis_completion.json",
    }
    required = prepared
    if terminal == "inference":
        required = inference
    elif terminal == "analysis":
        required = expected
    missing = required - observed
    if missing:
        raise ContractError(
            f"external output namespace is incomplete for {terminal or 'prepared'}: "
            f"{sorted(map(str, missing))[:5]}"
        )
    records = []
    for path in sorted(observed):
        identity = _artifact(path)
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": identity["sha256"],
                "size_bytes": identity["size_bytes"],
            }
        )
    tree_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "artifact_count": len(records),
        "total_size_bytes": int(sum(record["size_bytes"] for record in records)),
        "tree_sha256": tree_sha256,
    }


def checkpoint_path(training_root: Path, task: str, seed: int) -> Path:
    job = source.Job("fine", task, seed)
    return source.run_dir(training_root, job) / "final/refit/model.ckpt"


def refit_info_path(training_root: Path, task: str, seed: int) -> Path:
    return checkpoint_path(training_root, task, seed).with_name("info.json")


def _implementation_sources() -> dict[str, dict[str, Any]]:
    paths = {
        "controller": Path(__file__).resolve(),
        "controller_test": REPO / "tests/test_aim3_tcga_surgen_primary_fine_external_campaign.py",
        "source_campaign": REPO / "aim3_tcga_surgen_primary_five_seed_campaign.py",
        "molecular_labels": REPO / "aim3_repeated_control_campaign.py",
        "scoring_runner": REPO / "aim2_cross_protocol_transfer.py",
        "lightning": REPO / "src/oceanpath/training/lightning.py",
        "packed_store": REPO / "src/oceanpath/datasets/packed.py",
        "datamodule": REPO / "src/oceanpath/datasets/datamodule.py",
        "abmil": REPO / "src/oceanpath/models/abmil.py",
    }
    return {name: _artifact(path) for name, path in paths.items()}


def _validate_training_bundle(
    training_root: Path, *, deep: bool, require_no_controls: bool
) -> dict[str, Any]:
    root = Path(training_root).resolve(strict=True)
    if root == TRAINING_ROOT.resolve(strict=False):
        source.validate_output_root(root, must_exist=True)
    source.verify_contract(root, deep=deep)
    source._verify_preflight(root)  # noqa: SLF001
    source._validate_fine_scheduler(  # noqa: SLF001
        root, require_no_controls=require_no_controls
    )
    fine_seal = source._validate_fine_training_seal(  # noqa: SLF001
        root, require_no_controls=require_no_controls
    )
    fine_results = source._validate_fine_analysis(  # noqa: SLF001
        root, require_no_controls=require_no_controls
    )
    if require_no_controls:
        source.assert_no_control_artifacts(root)
    checkpoints: dict[str, Any] = {}
    for job in score_jobs():
        source_job = source.Job("fine", job.task, job.seed)
        if deep:
            source.validate_job(root, source_job)
        checkpoint = checkpoint_path(root, job.task, job.seed)
        info_path = refit_info_path(root, job.task, job.seed)
        info = _read_json(info_path)
        if (
            info.get("strategy") != "refit"
            or info.get("refit_epoch_rule") != "p75"
            or int(info.get("seed", -1)) != job.seed
            or int(info.get("refit_epochs", 0)) <= 0
        ):
            raise ContractError(f"{job.key}: final model is not the governed p75 refit")
        checkpoints[job.key] = {
            "task": job.task,
            "seed": job.seed,
            "checkpoint": _artifact(checkpoint),
            "refit_info": _artifact(info_path),
            "refit_epochs": int(info["refit_epochs"]),
        }
    return {
        "root": str(root),
        "contract": _artifact(source.contract_path(root)),
        "preflight": _artifact(source.preflight_path(root)),
        "fine_scheduler": _artifact(source.fine_scheduler_path(root)),
        "fine_training_seal": _artifact(source.fine_training_seal_path(root)),
        "fine_analysis_completion": _artifact(root / "analysis/fine_analysis_completion.json"),
        "fine_results": _artifact(root / "analysis/fine_results.json"),
        "fine_patient_native_logits": _artifact(
            root / "analysis/fine_patient_native_logits.parquet"
        ),
        "source_manifest": fine_seal["contract"],
        "fine_status": fine_results["status"],
        "controls_status_at_fine_analysis": fine_results["controls_status"],
        "model_seeds": list(MODEL_SEEDS),
        "tasks": list(TASKS),
        "checkpoints": checkpoints,
    }


def _blind_frame(frame: pd.DataFrame, *, spec: TargetSpec, context: str) -> pd.DataFrame:
    overlap = FORBIDDEN_OUTCOME_COLUMNS & set(frame.columns)
    unexpected = set(frame.columns) - ALLOWED_LABEL_BLIND_COLUMNS
    required = {"slide_id", "patient_id"}
    if overlap or unexpected or required - set(frame):
        raise ContractError(
            f"{context}: label-blind schema violation; forbidden={sorted(overlap)}, "
            f"unexpected={sorted(unexpected)}, missing={sorted(required - set(frame))}"
        )
    if frame["slide_id"].isna().any() or frame["patient_id"].isna().any():
        raise ContractError(f"{context}: null slide or patient identifier")
    out = frame.copy()
    out["slide_id"] = out["slide_id"].astype(str)
    out["patient_id"] = out["patient_id"].astype(str)
    if out["slide_id"].str.strip().eq("").any() or out["patient_id"].str.strip().eq("").any():
        raise ContractError(f"{context}: blank slide or patient identifier")
    if out["slide_id"].duplicated().any():
        raise ContractError(f"{context}: duplicate slide identifier")
    if len(out) != spec.slides or out["patient_id"].nunique() != spec.patients:
        raise ContractError(f"{context}: blind census drifted")
    return out.sort_values("slide_id", kind="mergesort").reset_index(drop=True)


def _load_blind_sources() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for key, spec in TARGETS.items():
        if _sha256(spec.blind_source) != spec.blind_sha256:
            raise ContractError(f"{key}: canonical label-blind source drifted")
        frames[key] = _blind_frame(
            pd.read_csv(spec.blind_source, low_memory=False), spec=spec, context=f"{key}/source"
        )
    _validate_blind_relations(frames)
    return frames


def _validate_blind_relations(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    if set(frames) != set(TARGET_ORDER):
        raise ContractError("blind target roster is incomplete")
    all_slides = [slide for key in TARGET_ORDER for slide in frames[key]["slide_id"].astype(str)]
    if len(all_slides) != 479 or len(set(all_slides)) != 479:
        raise ContractError("target slides must be exactly 479 and globally unique")
    patient_sets = {key: set(frame["patient_id"].astype(str)) for key, frame in frames.items()}
    overlaps: dict[str, int] = {}
    for index, left in enumerate(TARGET_ORDER):
        for right in TARGET_ORDER[index + 1 :]:
            observed = patient_sets[left] & patient_sets[right]
            expected = (
                RIH_DUAL_PATIENTS if {left, right} == {"rih_primary", "rih_metastatic"} else set()
            )
            if observed != set(expected):
                raise ContractError(
                    f"target patient overlap drifted: {left}/{right}={len(observed)}"
                )
            overlaps[f"{left}__{right}"] = len(observed)
    unique_patients = set().union(*patient_sets.values())
    if sum(len(value) for value in patient_sets.values()) != 446 or len(unique_patients) != 438:
        raise ContractError("external dataset-patient census drifted")
    return {
        "slides": 479,
        "dataset_patient_records": 446,
        "unique_patient_ids": 438,
        "rih_dual_role_patients": sorted(RIH_DUAL_PATIENTS),
        "pairwise_patient_overlaps": overlaps,
    }


def _combined_blind(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key in TARGET_ORDER:
        block = frames[key][["slide_id", "patient_id"]].copy()
        block["dataset"] = key
        block["role"] = TARGETS[key].role
        rows.append(block)
    combined = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["dataset", "slide_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if list(combined.columns) != ["slide_id", "patient_id", "dataset", "role"]:
        raise AssertionError("combined blind schema drifted")
    _validate_blind_relations(
        {key: combined.loc[combined["dataset"].eq(key)] for key in TARGET_ORDER}
    )
    return combined


def _validate_pack(
    training_root: Path, target_slides: set[str], *, rehash_large_payloads: bool
) -> dict[str, Any]:
    training_contract = _read_json(source.contract_path(training_root))
    upstream = training_contract.get("packed_store")
    if not isinstance(upstream, Mapping) or upstream.get("encoder") != "UNI-v1":
        raise ContractError("Aim-3 source contract lacks the governed UNI-v1 pack")
    artifacts = upstream.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ContractError("packed-store artifact graph is malformed")
    for name in ("features.bin", "coords.bin", "index.parquet", "meta.json"):
        identity = artifacts.get(name)
        if not isinstance(identity, Mapping):
            raise ContractError(f"packed-store graph lacks {name}")
        path = Path(str(identity.get("path", "")))
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(identity.get("size_bytes", -1))
        ):
            raise ContractError(f"packed-store artifact missing/size-drifted: {path}")
        if (name in {"index.parquet", "meta.json"} or rehash_large_payloads) and _sha256(
            path
        ) != identity.get("sha256"):
            raise ContractError(f"packed-store payload drifted: {path}")
    pack_dir = Path(str(upstream["path"])).resolve(strict=True)
    meta = _read_json(Path(str(artifacts["meta.json"]["path"])))
    index = pd.read_parquet(Path(str(artifacts["index.parquet"]["path"])))
    id_column = "slide_id" if "slide_id" in index else "key"
    missing = target_slides - set(index[id_column].astype(str))
    if missing:
        raise ContractError(f"UNI-v1 pack lacks external slides: {sorted(missing)[:5]}")
    selected = index[index[id_column].astype(str).isin(target_slides)]
    patches = int(pd.to_numeric(selected["n_patches"], errors="raise").sum())
    if len(selected) != 479 or patches != 4_915_250:
        raise ContractError(
            f"external packed census drifted: slides={len(selected)}, patches={patches}"
        )
    return {
        "encoder": "UNI-v1",
        "feature_dim": 1024,
        "pack_dir": str(pack_dir),
        "source_dir": str(Path(str(meta["source_dir"])).resolve(strict=True)),
        "upstream_contract": dict(upstream),
        "target_slides": 479,
        "target_patches": patches,
        "missing_target_slides": 0,
        "large_payload_validation": (
            "features.bin and coords.bin SHA-256 reverified at prepare, preflight, parent "
            "score, and terminal deep replay; shallow score workers recheck exact live sizes"
        ),
    }


def _source_target_overlap(
    training_root: Path, frames: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    manifest = pd.read_csv(source.source_snapshot_manifest(training_root), low_memory=False)
    source_slides = set(manifest["slide_id"].astype(str))
    source_patients = set(manifest["patient_id"].astype(str))
    result: dict[str, Any] = {}
    for key, frame in frames.items():
        slide_overlap = source_slides & set(frame["slide_id"].astype(str))
        patient_overlap = source_patients & set(frame["patient_id"].astype(str))
        if slide_overlap or patient_overlap:
            raise ContractError(
                f"{key}: external/source identity overlap; slides={len(slide_overlap)}, "
                f"patients={len(patient_overlap)}"
            )
        result[key] = {"slides": 0, "patients": 0}
    return result


def _job_document(root: Path, training: Mapping[str, Any]) -> dict[str, Any]:
    jobs = []
    for job in score_jobs():
        record = training["checkpoints"][job.key]
        jobs.append(
            {
                "job_id": f"fine_external.score.{job.task}.seed{job.seed}",
                "task": job.task,
                "seed": job.seed,
                "checkpoint": record["checkpoint"],
                "manifest": str(combined_blind_path(root)),
                "output": str(score_path(root, job.task, job.seed)),
                "expected_rows": 479,
                "fit_count": 0,
                "contains_target_outcomes": False,
            }
        )
    return {"jobs": jobs}


def _contract_payload(
    root: Path,
    training_root: Path,
    training: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    pack: Mapping[str, Any],
    *,
    created_utc: str,
) -> dict[str, Any]:
    snapshots = {key: _artifact(blind_path(root, key)) for key in TARGET_ORDER}
    combined = _artifact(combined_blind_path(root))
    relations = _validate_blind_relations(frames)
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "prepared_label_blind_before_outcome_join",
        "created_utc": created_utc,
        "output_root": str(root),
        "training": dict(training),
        "training_root": str(training_root),
        "tasks": list(TASKS),
        "task_display": TASK_DISPLAY,
        "model_seeds": list(MODEL_SEEDS),
        "targets": {
            key: {
                "display": spec.display,
                "role": spec.role,
                "slides": spec.slides,
                "patients": spec.patients,
                "mutant_patients": spec.mutant_patients,
                "canonical_blind_source": {
                    "path": str(spec.blind_source),
                    "expected_sha256": spec.blind_sha256,
                },
                "snapshot": snapshots[key],
                "outcome_source_not_opened": {
                    "path": str(spec.outcome_source),
                    "expected_sha256": spec.outcome_sha256,
                },
            }
            for key, spec in TARGETS.items()
        },
        "combined_label_blind_manifest": combined,
        "blind_roster_relations": relations,
        "source_target_overlap": _source_target_overlap(training_root, frames),
        "feature_store": dict(pack),
        "score_job_plan": _artifact(job_plan_path(root)),
        "score_contract": {
            "jobs": 25,
            "score_artifacts": 25,
            "rows_per_artifact": 479,
            "score_rows": 11_975,
            "factorization": "5 molecular tasks x 5 p75 refit seeds; each scores all 479 blind slides",
            "new_fits": 0,
            "model_slide_forward_passes": 11_975,
            "patch_model_forward_passes": 122_881_250,
            "maximum_parallel_workers": MAX_WORKERS,
            "native_logit_schema": ["slide_id", "seed", "fold", "logit"],
            "refit_fold_marker": 0,
        },
        "artifact_inventory": {
            "terminal_regular_files": 67,
            "prepared_files": 8,
            "score_and_receipt_files": 50,
            "analysis_files": 5,
            "unexpected_files_permitted": False,
        },
        "governance": {
            "external_test_data": [TARGETS[key].display for key in TARGET_ORDER],
            "prepare_reads_target_outcomes": False,
            "preflight_reads_target_outcomes": False,
            "score_reads_target_outcomes": False,
            "analysis_requires_inference_seal": True,
            "target_fitting": False,
            "target_refitting": False,
            "target_adaptation": False,
            "target_calibration": False,
            "target_threshold_selection": False,
            "target_model_or_checkpoint_selection": False,
            "target_encoder_or_construction_selection": False,
            "inference_seal_exactly_once": True,
            "control_artifacts_absent_at_prepare": True,
            "phase_order": [
                "fine_training_and_analysis",
                "external_score_and_analysis",
                "matched_and_repeated_controls",
            ],
        },
        "implementation": _implementation_sources(),
    }


def prepare(root: Path, training_root: Path, *, apply: bool) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=False if apply else None)
    training_path = Path(training_root).resolve(strict=True)
    _validate_training_output_separation(output, training_path)
    training = _validate_training_bundle(
        training_path, deep=True, require_no_controls=True
    )
    frames = _load_blind_sources()
    combined = _combined_blind(frames)
    target_slides = set(combined["slide_id"].astype(str))
    pack = _validate_pack(
        training_path, target_slides, rehash_large_payloads=True
    )
    _source_target_overlap(training_path, frames)
    plan = {
        "status": "DRY_RUN_READY" if not apply else "PREPARING",
        "output_root": str(output),
        "training_fine_phase": "authenticated",
        "tasks": list(TASKS),
        "seeds": list(MODEL_SEEDS),
        "score_jobs": 25,
        "blind_slides_per_job": 479,
        "score_rows": 11_975,
        "target_outcomes_opened": False,
    }
    if not apply:
        return plan
    output.mkdir(parents=True, exist_ok=False)
    for key, spec in TARGETS.items():
        _write_bytes_once(blind_path(output, key), spec.blind_source.read_bytes())
    _write_bytes_once(combined_blind_path(output), combined.to_csv(index=False).encode())
    _write_json_once(job_plan_path(output), _job_document(output, training))
    payload = _contract_payload(
        output,
        training_path,
        training,
        {
            key: _blind_frame(pd.read_csv(blind_path(output, key)), spec=TARGETS[key], context=key)
            for key in TARGET_ORDER
        },
        pack,
        created_utc=_utcnow(),
    )
    _write_json_once(contract_path(output), payload)
    return _load_contract(output, deep=True, require_no_controls=True)


def _load_contract(
    root: Path, *, deep: bool, require_no_controls: bool
) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    stored = _read_json(contract_path(output))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("contract lacks created_utc")
    training_root = Path(str(stored.get("training_root", ""))).resolve(strict=True)
    _validate_training_output_separation(output, training_root)
    training = _validate_training_bundle(
        training_root, deep=deep, require_no_controls=require_no_controls
    )
    frames = {
        key: _blind_frame(
            pd.read_csv(blind_path(output, key), low_memory=False),
            spec=TARGETS[key],
            context=f"{key}/snapshot",
        )
        for key in TARGET_ORDER
    }
    for key, spec in TARGETS.items():
        if _artifact(blind_path(output, key))["sha256"] != spec.blind_sha256:
            raise ContractError(f"{key}: blind snapshot is not the canonical byte copy")
    combined = _combined_blind(frames)
    replay = pd.read_csv(combined_blind_path(output), low_memory=False)
    pd.testing.assert_frame_equal(replay, combined, check_dtype=False)
    pack = _validate_pack(
        training_root,
        set(combined["slide_id"].astype(str)),
        rehash_large_payloads=deep,
    )
    if _read_json(job_plan_path(output)) != _job_document(output, training):
        raise ContractError("score-job plan does not replay")
    expected = _contract_payload(
        output,
        training_root,
        training,
        frames,
        pack,
        created_utc=created,
    )
    if stored != expected:
        raise ContractError("stored external contract does not replay exactly")
    return stored


def preflight(root: Path, *, apply: bool) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    contract = _load_contract(output, deep=True, require_no_controls=True)
    for record in contract["training"]["checkpoints"].values():
        _verify_artifact(record["checkpoint"], context="p75 checkpoint")
        _verify_artifact(record["refit_info"], context="p75 refit info")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_label_blind_inference",
        "created_utc": _utcnow(),
        "contract": _artifact(contract_path(output)),
        "fine_training_seal": contract["training"]["fine_training_seal"],
        "fine_analysis_completion": contract["training"]["fine_analysis_completion"],
        "authenticated_p75_checkpoints": 25,
        "score_jobs": 25,
        "score_rows": 11_975,
        "maximum_parallel_workers": MAX_WORKERS,
        "target_outcomes_opened": False,
        "target_outcomes_present": False,
        "control_artifacts_absent": True,
        "checks": {
            "fine_phase_sealed": True,
            "fine_analysis_sealed": True,
            "label_blind_snapshots_exact": True,
            "pack_covers_all_479_slides": True,
            "source_target_slide_and_patient_overlap_zero": True,
            "rih_primary_metastatic_overlap_exactly_eight": True,
        },
    }
    _validate_output_namespace(output)
    if apply:
        if preflight_path(output).exists():
            old = _read_json(preflight_path(output))
            payload["created_utc"] = old.get("created_utc")
            if old != payload:
                raise ContractError("published preflight receipt drifted")
        else:
            _write_json_once(preflight_path(output), payload)
    return payload


def _load_preflight(
    root: Path, *, require_no_controls: bool = True
) -> dict[str, Any]:
    stored = _read_json(preflight_path(root))
    if (
        stored.get("status") != "ready_for_label_blind_inference"
        or stored.get("contract") != _artifact(contract_path(root))
        or stored.get("authenticated_p75_checkpoints") != 25
        or stored.get("target_outcomes_opened") is not False
        or stored.get("control_artifacts_absent") is not True
    ):
        raise ContractError("persisted preflight receipt is invalid")
    if require_no_controls:
        training_root = Path(str(_read_json(contract_path(root))["training_root"]))
        source.assert_no_control_artifacts(training_root)
    return stored


def _assert_no_control_artifacts_for_external_stage(root: Path) -> Path:
    training_root = Path(str(_read_json(contract_path(root))["training_root"])).resolve(
        strict=True
    )
    source.assert_no_control_artifacts(training_root)
    return training_root


def _score_environment(*, device: str, num_workers: int) -> dict[str, Any]:
    if device != "cuda":
        raise ContractError("governed external scoring requires CUDA")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ContractError("PyTorch is unavailable") from exc
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ContractError("governed external scoring requires CUDA bfloat16")
    return {
        "schema_version": SCHEMA_VERSION,
        "device": "cuda",
        "device_index": 0,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "autocast": True,
        "autocast_dtype": "bfloat16",
        "batch_size": 1,
        "num_workers_per_job": int(num_workers),
        "evaluation_bag": "full",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "target_outcomes_present": False,
    }


def _seal_environment(root: Path, *, device: str, num_workers: int) -> dict[str, Any]:
    value = _score_environment(device=device, num_workers=num_workers)
    path = environment_path(root)
    if path.exists():
        if _read_json(path) != value:
            raise ContractError("scoring environment changed after publication")
    else:
        _write_json_once(path, value)
    return value


def _validate_score_frame(frame: pd.DataFrame, manifest: pd.DataFrame, *, job: ScoreJob) -> None:
    if list(frame.columns) != ["slide_id", "seed", "fold", "logit"]:
        raise ContractError(f"{job.key}: score schema drifted")
    if len(frame) != 479 or set(frame["slide_id"].astype(str)) != set(
        manifest["slide_id"].astype(str)
    ):
        raise ContractError(f"{job.key}: score roster is incomplete")
    if set(pd.to_numeric(frame["seed"], errors="raise").astype(int)) != {job.seed}:
        raise ContractError(f"{job.key}: score seed drifted")
    if set(pd.to_numeric(frame["fold"], errors="raise").astype(int)) != {0}:
        raise ContractError(f"{job.key}: refit fold marker must be zero")
    if frame.duplicated(["slide_id", "seed", "fold"]).any():
        raise ContractError(f"{job.key}: duplicate score row")
    if not np.isfinite(pd.to_numeric(frame["logit"], errors="coerce").to_numpy(float)).all():
        raise ContractError(f"{job.key}: non-finite native logit")


def _validate_inference_event(value: Any, *, job: ScoreJob) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{job.key}: inference execution event is missing")
    event = dict(value)
    expected = {
        "job_id": f"fine_external.score.{job.task}.seed{job.seed}",
        "task": job.task,
        "seed": job.seed,
        "returncode": 0,
        "cache_hit": False,
        "score_rows": 479,
        "execution_role": "label_blind_native_logit_inference",
    }
    timing_keys = {
        "started_utc",
        "completed_utc",
        "started_unix_ns",
        "completed_unix_ns",
    }
    if set(event) != set(expected) | timing_keys or {
        key: event.get(key) for key in expected
    } != expected:
        raise ContractError(f"{job.key}: inference execution event drifted")
    started = event.get("started_unix_ns")
    completed = event.get("completed_unix_ns")
    if (
        not isinstance(started, int)
        or isinstance(started, bool)
        or not isinstance(completed, int)
        or isinstance(completed, bool)
        or started <= 0
        or completed <= started
    ):
        raise ContractError(f"{job.key}: inference execution interval is invalid")
    for key in ("started_utc", "completed_utc"):
        try:
            parsed = dt.datetime.fromisoformat(str(event.get(key, "")))
        except ValueError as exc:
            raise ContractError(f"{job.key}: invalid {key}") from exc
        if parsed.tzinfo is None:
            raise ContractError(f"{job.key}: {key} is not timezone-aware")
    return event


def _validate_cached_score(root: Path, job: ScoreJob) -> pd.DataFrame | None:
    path = score_path(root, job.task, job.seed)
    receipt_path = score_receipt_path(root, job.task, job.seed)
    if not path.exists() and not receipt_path.exists():
        return None
    if (
        not path.is_file()
        or path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
    ):
        raise ContractError(f"{job.key}: partial or symlinked score cache")
    contract = _read_json(contract_path(root))
    checkpoint = contract["training"]["checkpoints"][job.key]["checkpoint"]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "contains_target_outcomes": False,
        "task": job.task,
        "seed": job.seed,
        "manifest": _artifact(combined_blind_path(root)),
        "checkpoint": checkpoint,
        "contract": _artifact(contract_path(root)),
        "environment": _artifact(environment_path(root)),
        "artifact": _artifact(path),
        "n_rows": 479,
    }
    receipt = _read_json(receipt_path)
    if set(receipt) != set(expected) | {"created_utc", "execution"} or {
        key: receipt.get(key) for key in expected
    } != expected:
        raise ContractError(f"{job.key}: score receipt drifted")
    if not isinstance(receipt.get("created_utc"), str) or not receipt["created_utc"]:
        raise ContractError(f"{job.key}: score receipt lacks created_utc")
    _validate_inference_event(receipt.get("execution"), job=job)
    frame = pd.read_parquet(path)
    _validate_score_frame(frame, pd.read_csv(combined_blind_path(root)), job=job)
    return frame


def _score_one(
    root: Path,
    task: str,
    seed: int,
    *,
    device: str,
    num_workers: int,
) -> None:
    output = validate_output_root(root, must_exist=True)
    job = ScoreJob(task, seed)
    contract = _load_contract(output, deep=False, require_no_controls=True)
    _load_preflight(output)
    if inference_seal_path(output).exists():
        verify_inference_seal(output, require_no_controls=True)
        return
    _seal_environment(output, device=device, num_workers=num_workers)
    if _validate_cached_score(output, job) is not None:
        return
    manifest = pd.read_csv(combined_blind_path(output), low_memory=False)
    checkpoint_identity = contract["training"]["checkpoints"][job.key]["checkpoint"]
    checkpoint = _verify_artifact(checkpoint_identity, context=f"{job.key}/checkpoint")
    pack = contract["feature_store"]
    started_unix_ns = time.time_ns()
    started_utc = _utcnow_precise()
    scores = cpht._score_checkpoints(  # noqa: SLF001
        [(seed, 0, checkpoint)],
        manifest,
        feature_dir=Path(str(pack["source_dir"])),
        pack_dir=Path(str(pack["pack_dir"])),
        device=device,
        num_workers=num_workers,
    )
    scores = (
        scores[["slide_id", "seed", "fold", "logit"]]
        .sort_values("slide_id", kind="mergesort")
        .reset_index(drop=True)
    )
    _validate_score_frame(scores, manifest, job=job)
    completed_utc = _utcnow_precise()
    completed_unix_ns = time.time_ns()
    if completed_unix_ns <= started_unix_ns:
        raise ContractError(f"{job.key}: non-positive inference execution interval")
    destination = score_path(output, task, seed)
    _write_parquet_once(destination, scores)
    _write_json_once(
        score_receipt_path(output, task, seed),
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_utc": _utcnow(),
            "contains_target_outcomes": False,
            "task": task,
            "seed": seed,
            "manifest": _artifact(combined_blind_path(output)),
            "checkpoint": checkpoint_identity,
            "contract": _artifact(contract_path(output)),
            "environment": _artifact(environment_path(output)),
            "artifact": _artifact(destination),
            "n_rows": 479,
            "execution": {
                "job_id": f"fine_external.score.{task}.seed{seed}",
                "task": task,
                "seed": seed,
                "started_utc": started_utc,
                "completed_utc": completed_utc,
                "started_unix_ns": started_unix_ns,
                "completed_unix_ns": completed_unix_ns,
                "returncode": 0,
                "cache_hit": False,
                "score_rows": 479,
                "execution_role": "label_blind_native_logit_inference",
            },
        },
    )
    _validate_cached_score(output, job)


def _internal_score_command(
    root: Path, job: ScoreJob, *, device: str, num_workers: int
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_score-one",
        "--output-root",
        str(root),
        "--task",
        job.task,
        "--seed",
        str(job.seed),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
    ]


def _run_score_subprocess(command: Sequence[str]) -> int:
    return int(subprocess.run(list(command), cwd=REPO, check=False).returncode)


def _parallel_peak(
    events: Sequence[Mapping[str, Any]], *, start_key: str, completed_key: str
) -> int:
    points: list[tuple[int, int]] = []
    for event in events:
        started = event.get(start_key)
        completed = event.get(completed_key)
        if (
            not isinstance(started, int)
            or isinstance(started, bool)
            or not isinstance(completed, int)
            or isinstance(completed, bool)
            or started <= 0
            or completed <= started
        ):
            raise ContractError("parallel execution event has an invalid interval")
        points.extend(((started, 1), (completed, -1)))
    active = 0
    peak = 0
    for _timestamp, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        if active < 0:
            raise ContractError("parallel execution intervals are malformed")
        peak = max(peak, active)
    if active != 0:
        raise ContractError("parallel execution intervals did not close")
    return peak


def _validate_scheduler_events(
    root: Path, events: Any, *, environment: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(events, list) or len(events) != 25:
        raise ContractError("scoring scheduler must bind exactly 25 job events")
    by_key: dict[str, dict[str, Any]] = {}
    for value in events:
        if not isinstance(value, Mapping):
            raise ContractError("scoring scheduler event is malformed")
        event = dict(value)
        try:
            job = ScoreJob(str(event.get("task")), int(event.get("seed", -1)))
        except (TypeError, ValueError) as exc:
            raise ContractError("scoring scheduler job identity is malformed") from exc
        expected = {
            "job_id": f"fine_external.score.{job.task}.seed{job.seed}",
            "task": job.task,
            "seed": job.seed,
            "command": _internal_score_command(
                root,
                job,
                device=str(environment["device"]),
                num_workers=int(environment["num_workers_per_job"]),
            ),
            "returncode": 0,
        }
        execution_keys = {
            "cached_before_launch",
            "started_utc",
            "completed_utc",
            "started_unix_ns",
            "completed_unix_ns",
        }
        if set(event) != set(expected) | execution_keys or {
            key: event.get(key) for key in expected
        } != expected:
            raise ContractError(f"{job.key}: scheduler execution event drifted")
        if not isinstance(event.get("cached_before_launch"), bool):
            raise ContractError(f"{job.key}: scheduler cache-state evidence is invalid")
        for key in ("started_utc", "completed_utc"):
            try:
                parsed = dt.datetime.fromisoformat(str(event.get(key, "")))
            except ValueError as exc:
                raise ContractError(f"{job.key}: invalid scheduler {key}") from exc
            if parsed.tzinfo is None:
                raise ContractError(f"{job.key}: scheduler {key} is not timezone-aware")
        if job.key in by_key:
            raise ContractError(f"duplicate scheduler event: {job.key}")
        by_key[job.key] = event
    if set(by_key) != {job.key for job in score_jobs()}:
        raise ContractError("scoring scheduler job roster drifted")
    ordered = [by_key[job.key] for job in score_jobs()]
    if (
        _parallel_peak(
            ordered,
            start_key="started_unix_ns",
            completed_key="completed_unix_ns",
        )
        != MAX_WORKERS
    ):
        raise ContractError("scheduler did not observe exactly six parallel workers")
    return ordered


def _inference_events(root: Path) -> list[dict[str, Any]]:
    events = []
    for job in score_jobs():
        if _validate_cached_score(root, job) is None:
            raise ContractError(f"missing score while collecting inference event: {job.key}")
        receipt = _read_json(score_receipt_path(root, job.task, job.seed))
        events.append(_validate_inference_event(receipt.get("execution"), job=job))
    if (
        _parallel_peak(
            events,
            start_key="started_unix_ns",
            completed_key="completed_unix_ns",
        )
        != MAX_WORKERS
    ):
        raise ContractError("native-logit inference did not reach exactly six parallel workers")
    return events


def _validate_scoring_completion(root: Path) -> dict[str, Any]:
    stored = _read_json(scoring_completion_path(root))
    environment = _read_json(environment_path(root))
    scheduler_events = _validate_scheduler_events(
        root, stored.get("scheduler_events"), environment=environment
    )
    inference_events = _inference_events(root)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_before_outcome_join",
        "contract": _artifact(contract_path(root)),
        "target_outcomes_present": False,
        "target_outcomes_opened": False,
        "control_artifacts_absent": True,
        "score_jobs": 25,
        "score_rows": 11_975,
        "configured_max_workers": MAX_WORKERS,
        "max_observed_parallel_scheduler_workers": MAX_WORKERS,
        "max_observed_parallel_inference_workers": MAX_WORKERS,
        "scheduler_events": scheduler_events,
        "inference_events": inference_events,
    }
    if set(stored) != set(expected) | {"created_utc"} or {
        key: stored.get(key) for key in expected
    } != expected:
        raise ContractError("published scoring completion receipt drifted")
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("scoring completion lacks created_utc")
    return stored


def _scoring_completion(
    root: Path, *, scheduler_events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if scoring_completion_path(root).exists():
        return _validate_scoring_completion(root)
    _assert_no_control_artifacts_for_external_stage(root)
    environment = _read_json(environment_path(root))
    validated_scheduler = _validate_scheduler_events(
        root, list(scheduler_events), environment=environment
    )
    inference_events = _inference_events(root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_before_outcome_join",
        "created_utc": _utcnow(),
        "contract": _artifact(contract_path(root)),
        "target_outcomes_present": False,
        "target_outcomes_opened": False,
        "control_artifacts_absent": True,
        "score_jobs": 25,
        "score_rows": 11_975,
        "configured_max_workers": MAX_WORKERS,
        "max_observed_parallel_scheduler_workers": MAX_WORKERS,
        "max_observed_parallel_inference_workers": MAX_WORKERS,
        "scheduler_events": validated_scheduler,
        "inference_events": inference_events,
    }
    path = scoring_completion_path(root)
    _write_json_once(path, payload)
    return _validate_scoring_completion(root)


def _seal_payload(root: Path, *, require_no_controls: bool) -> dict[str, Any]:
    if require_no_controls:
        _assert_no_control_artifacts_for_external_stage(root)
    records = []
    total_rows = 0
    for job in score_jobs():
        frame = _validate_cached_score(root, job)
        if frame is None:
            raise ContractError(f"cannot seal incomplete inference: {job.key}")
        total_rows += len(frame)
        records.append(
            {
                "task": job.task,
                "seed": job.seed,
                "score": _artifact(score_path(root, job.task, job.seed)),
                "receipt": _artifact(score_receipt_path(root, job.task, job.seed)),
                "rows": len(frame),
            }
        )
    scoring = _validate_scoring_completion(root)
    if len(records) != 25 or total_rows != 11_975:
        raise ContractError("inference score roster drifted")
    if (
        scoring.get("configured_max_workers") != MAX_WORKERS
        or scoring.get("max_observed_parallel_scheduler_workers") != MAX_WORKERS
        or scoring.get("max_observed_parallel_inference_workers") != MAX_WORKERS
    ):
        raise ContractError("scoring concurrency receipt drifted")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_before_outcome_join",
        "created_utc": _utcnow(),
        "target_outcomes_present": False,
        "target_outcomes_opened": False,
        "control_artifacts_absent_at_inference_seal": True,
        "contract": _artifact(contract_path(root)),
        "preflight": _artifact(preflight_path(root)),
        "environment": _artifact(environment_path(root)),
        "scoring_completion": _artifact(scoring_completion_path(root)),
        "tasks": list(TASKS),
        "seeds": list(MODEL_SEEDS),
        "targets": list(TARGET_ORDER),
        "score_artifact_count": 25,
        "score_rows": 11_975,
        "score_artifacts": records,
    }


def seal_inference(root: Path) -> dict[str, Any]:
    path = inference_seal_path(root)
    if path.exists():
        return verify_inference_seal(root, require_no_controls=True)
    payload = _seal_payload(root, require_no_controls=True)
    _write_json_once(path, payload)
    return verify_inference_seal(root, require_no_controls=True)


def verify_inference_seal(
    root: Path, *, require_no_controls: bool = False
) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    stored = _read_json(inference_seal_path(output))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("inference seal lacks created_utc")
    expected = _seal_payload(output, require_no_controls=require_no_controls)
    expected["created_utc"] = created
    if stored != expected:
        raise ContractError("inference seal or sealed score artifact drifted")
    _validate_output_namespace(output, terminal="inference")
    return stored


def score(
    root: Path,
    *,
    apply: bool,
    max_workers: int,
    device: str,
    num_workers: int,
) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    _load_contract(output, deep=True, require_no_controls=True)
    _load_preflight(output)
    if max_workers != MAX_WORKERS:
        raise ContractError("governed scoring requires exactly --max-workers 6")
    if inference_seal_path(output).exists():
        return {
            "status": "already_sealed",
            "seal": verify_inference_seal(output, require_no_controls=True),
        }
    commands = [
        _internal_score_command(output, job, device=device, num_workers=num_workers)
        for job in score_jobs()
    ]
    if not apply:
        return {
            "status": "DRY_RUN",
            "jobs": 25,
            "score_rows": 11_975,
            "max_workers": MAX_WORKERS,
            "target_outcomes_opened": False,
            "commands": commands,
        }
    _seal_environment(output, device=device, num_workers=num_workers)
    if scoring_completion_path(output).exists():
        _validate_scoring_completion(output)
        return {"status": "complete_and_sealed", "seal": seal_inference(output)}
    cached_before = {
        job.key: _validate_cached_score(output, job) is not None for job in score_jobs()
    }

    def run(job: ScoreJob, command: Sequence[str]) -> dict[str, Any]:
        started_unix_ns = time.time_ns()
        started_utc = _utcnow_precise()
        returncode = _run_score_subprocess(command)
        completed_utc = _utcnow_precise()
        completed_unix_ns = time.time_ns()
        return {
            "job_id": f"fine_external.score.{job.task}.seed{job.seed}",
            "task": job.task,
            "seed": job.seed,
            "command": list(command),
            "cached_before_launch": cached_before[job.key],
            "started_utc": started_utc,
            "completed_utc": completed_utc,
            "started_unix_ns": started_unix_ns,
            "completed_unix_ns": completed_unix_ns,
            "returncode": returncode,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run, job, command)
            for job, command in zip(score_jobs(), commands, strict=True)
        ]
        scheduler_events = []
        for future in concurrent.futures.as_completed(futures):
            scheduler_events.append(future.result())
    failures = [event for event in scheduler_events if event["returncode"] != 0]
    if failures:
        raise ContractError(
            "label-blind scoring subprocess failed; immutable successful caches may be "
            f"replayed, first={failures[:3]}"
        )
    _scoring_completion(output, scheduler_events=scheduler_events)
    return {"status": "complete_and_sealed", "seal": seal_inference(output)}


def _open_outcomes_after_seal(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """The sole target-outcome opening point; the first statement verifies the seal."""

    verify_inference_seal(root, require_no_controls=True)
    frames: dict[str, pd.DataFrame] = {}
    identities: dict[str, Any] = {}
    for key, spec in TARGETS.items():
        identity = _artifact(spec.outcome_source)
        if identity["sha256"] != spec.outcome_sha256:
            raise ContractError(f"{key}: outcome source drifted")
        blind = pd.read_csv(blind_path(root, key), low_memory=False)
        raw = pd.read_csv(spec.outcome_source, low_memory=False)
        if key == "orion_primary":
            raw = raw.loc[
                raw["cohort"].eq("Orion")
                & raw["subcohort"].eq("Orion-CRC")
                & raw["specimen_role"].astype(str).str.casefold().eq("primary")
                & raw["output_id"].astype(str).isin(blind["slide_id"].astype(str))
            ].copy()
            raw["slide_id"] = raw["output_id"].astype(str)
            raw["patient_id"] = raw["patient_uid"].astype(str)
        required = {"slide_id", "patient_id", "kras", "kras_subvariant"}
        if required - set(raw):
            raise ContractError(f"{key}: outcome source lacks {sorted(required - set(raw))}")
        normalized = raw[["slide_id", "patient_id", "kras", "kras_subvariant"]].copy()
        normalized["slide_id"] = normalized["slide_id"].astype(str)
        normalized["patient_id"] = normalized["patient_id"].astype(str)
        joined = blind[["slide_id", "patient_id"]].merge(
            normalized,
            on=["slide_id", "patient_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(joined) != spec.slides:
            raise ContractError(f"{key}: outcome join does not cover the exact blind roster")
        consistency = joined.groupby("patient_id")[["kras", "kras_subvariant"]].nunique(
            dropna=False
        )
        if (consistency > 1).any().any():
            raise ContractError(f"{key}: molecular outcome varies within patient")
        patients = joined.sort_values("slide_id").drop_duplicates("patient_id")
        census = (len(patients), int(patients["kras"].astype(str).eq("mutant").sum()))
        if census != (spec.patients, spec.mutant_patients):
            raise ContractError(f"{key}: patient/mutant census drifted: {census}")
        joined["dataset"] = key
        joined["role"] = spec.role
        frames[key] = joined
        identities[key] = identity
    return frames, identities


def derive_task_labels(patients: pd.DataFrame, task: str) -> pd.DataFrame:
    """Apply the source task definition exactly, after the inference seal."""

    if task not in TASKS:
        raise ValueError(task)
    work = patients.copy()
    mutant = work.loc[work["kras"].astype(str).eq("mutant")].copy()
    mutant["is_g12"] = mutant["kras_subvariant"].map(molecular.is_g12)
    if task == "codon":
        selected = mutant
        labels = selected["is_g12"].astype(int)
    elif task == "g12d_broad":
        selected = mutant
        labels = selected["kras_subvariant"].map(molecular.is_g12d).astype(int)
    else:
        selected = mutant.loc[mutant["is_g12"]].copy()
        allele = {"allele1": "G12D", "allele2": "G12V", "g12c": "G12C"}[task]
        labels = (
            selected["kras_subvariant"]
            .map(lambda value: molecular.has_allele(value, allele))
            .astype(int)
        )
    selected = selected.copy()
    selected["label"] = labels.to_numpy(dtype=int)
    if selected["patient_id"].astype(str).duplicated().any() or set(selected["label"]) != {0, 1}:
        raise ContractError(f"{task}: external molecular task is malformed")
    return selected.reset_index(drop=True)


def _patient_scores(
    root: Path, outcome_slides: Mapping[str, pd.DataFrame]
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    result: dict[str, dict[str, pd.DataFrame]] = {task: {} for task in TASKS}
    exports = []
    for task in TASKS:
        seed_scores = {
            seed: pd.read_parquet(score_path(root, task, seed))[["slide_id", "logit"]].rename(
                columns={"logit": f"logit_seed{seed}"}
            )
            for seed in MODEL_SEEDS
        }
        for target in TARGET_ORDER:
            slide = outcome_slides[target].copy()
            for seed in MODEL_SEEDS:
                slide = slide.merge(
                    seed_scores[seed], on="slide_id", how="inner", validate="one_to_one"
                )
            if len(slide) != TARGETS[target].slides:
                raise ContractError(f"{task}/{target}: scored slide roster drifted")
            consistency = slide.groupby("patient_id")[
                ["kras", "kras_subvariant", "dataset", "role"]
            ].nunique(dropna=False)
            if (consistency > 1).any().any():
                raise ContractError(f"{task}/{target}: patient metadata drifted")
            aggregation: dict[str, tuple[str, str]] = {
                "kras": ("kras", "first"),
                "kras_subvariant": ("kras_subvariant", "first"),
                "dataset": ("dataset", "first"),
                "role": ("role", "first"),
                "n_slides": ("slide_id", "size"),
                **{f"logit_seed{seed}": (f"logit_seed{seed}", "mean") for seed in MODEL_SEEDS},
            }
            patient = slide.groupby("patient_id", sort=True).agg(**aggregation).reset_index()
            patient = derive_task_labels(patient, task)
            patient["mean_logit_5seed"] = patient[
                [f"logit_seed{seed}" for seed in MODEL_SEEDS]
            ].mean(axis=1)
            expected = EXPECTED_PRIMITIVE_CENSUS[task][target]
            observed = (int(patient["label"].sum()), int(patient["label"].eq(0).sum()))
            if observed != expected:
                raise ContractError(f"{task}/{target}: task census {observed} != {expected}")
            result[task][target] = patient
            export = patient.copy()
            export.insert(0, "task", task)
            exports.append(export)
    return result, pd.concat(exports, ignore_index=True)


def _support_tier(positive: int, negative: int) -> str:
    smaller = min(positive, negative)
    if smaller < 10:
        return "VERY_SPARSE_LT10"
    if smaller < 20:
        return "EXPLORATORY_10_19"
    return "BETTER_POWERED_GE20"


def _named_rng(stream: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{stream}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))


def _stratified_indices(
    frame: pd.DataFrame, *, n_bootstrap: int, stream: str, strata: Sequence[str]
) -> np.ndarray:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    work = frame.reset_index(drop=True)
    rng = _named_rng(stream)
    pieces = []
    for _key, block in work.groupby(list(strata), sort=True, dropna=False, observed=True):
        indices = block.index.to_numpy(dtype=np.int64)
        pieces.append(rng.choice(indices, size=(n_bootstrap, len(indices)), replace=True))
    if not pieces:
        raise ContractError("bootstrap has no strata")
    return np.concatenate(pieces, axis=1)


def _clustered_patient_role_indices(
    frame: pd.DataFrame, *, n_bootstrap: int, stream: str
) -> np.ndarray:
    """Resample unique patients within their full dataset/label signature."""

    work = frame.reset_index(drop=True)
    patient_rows = {
        str(patient): block.index.to_numpy(dtype=np.int64)
        for patient, block in work.groupby("patient_id", sort=True, observed=True)
    }
    signatures: dict[tuple[tuple[str, int], ...], list[str]] = {}
    for patient, indices in patient_rows.items():
        signature = tuple(
            sorted(
                (str(work.loc[index, "dataset"]), int(work.loc[index, "label"]))
                for index in indices
            )
        )
        signatures.setdefault(signature, []).append(patient)
    rng = _named_rng(stream)
    draws: list[np.ndarray] = []
    for _signature, patients in sorted(signatures.items(), key=lambda item: str(item[0])):
        selected = rng.choice(
            np.asarray(sorted(patients), dtype=object),
            size=(n_bootstrap, len(patients)),
            replace=True,
        )
        rows_per_patient = len(patient_rows[patients[0]])
        if any(len(patient_rows[patient]) != rows_per_patient for patient in patients):
            raise AssertionError("membership signature did not imply constant record multiplicity")
        expanded = np.empty((n_bootstrap, len(patients) * rows_per_patient), dtype=np.int64)
        for draw in range(n_bootstrap):
            expanded[draw] = np.concatenate(
                [patient_rows[str(patient)] for patient in selected[draw]]
            )
        draws.append(expanded)
    if not draws:
        raise ContractError("cluster bootstrap has no patient clusters")
    result = np.concatenate(draws, axis=1)
    if result.shape != (n_bootstrap, len(work)):
        raise ContractError("cluster bootstrap changed record count")
    return result


def _auc_draws(labels: np.ndarray, scores: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = np.empty(len(indices), dtype=np.float64)
    for draw, index in enumerate(indices):
        values[draw] = roc_auc_score(labels[index], scores[index])
    return values


def _seed_summary(frame: pd.DataFrame) -> tuple[dict[str, float], float, float]:
    labels = frame["label"].to_numpy(dtype=int)
    values = {
        str(seed): float(roc_auc_score(labels, frame[f"logit_seed{seed}"].to_numpy(float)))
        for seed in MODEL_SEEDS
    }
    array = np.asarray(list(values.values()), dtype=float)
    return values, float(array.mean()), float(array.std(ddof=1))


def _metric_block(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    stream: str,
    arrays: dict[str, np.ndarray],
    array_key: str,
    clustered: bool = False,
) -> dict[str, Any]:
    work = frame.reset_index(drop=True)
    labels = work["label"].to_numpy(dtype=int)
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    support = _support_tier(positive, negative)
    base: dict[str, Any] = {
        "n_records": int(len(work)),
        "positive": positive,
        "negative": negative,
        "support": support,
        "sparse": support != "BETTER_POWERED_GE20",
    }
    if set(labels) != {0, 1}:
        return {
            **base,
            "status": "NOT_ESTIMABLE_SINGLE_CLASS",
            "per_seed_auroc": None,
            "seed_auroc_mean": None,
            "seed_auroc_sample_sd": None,
            "five_seed_refit_ensemble": {"auroc": None, "ci95": None},
        }
    per_seed, mean, sd = _seed_summary(work)
    scores = work["mean_logit_5seed"].to_numpy(float)
    point = float(roc_auc_score(labels, scores))
    indices = (
        _clustered_patient_role_indices(work, n_bootstrap=n_bootstrap, stream=stream)
        if clustered
        else _stratified_indices(
            work,
            n_bootstrap=n_bootstrap,
            stream=stream,
            strata=("dataset", "label"),
        )
    )
    draws = _auc_draws(labels, scores, indices)
    arrays[array_key] = draws
    return {
        **base,
        "status": "ESTIMABLE",
        "per_seed_auroc": per_seed,
        "seed_auroc_mean": mean,
        "seed_auroc_sample_sd": sd,
        "seed_sd_role": "descriptive across five fixed refit seeds; seeds are not bootstrap units",
        "five_seed_refit_ensemble": {
            "aggregation": "mean slide logit within patient and seed, then mean native patient logits over seeds 42..46",
            "auroc": point,
            "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
            "bootstrap": "10,000 fixed-composition patient resamples"
            if not clustered
            else "10,000 unique-patient cluster resamples within full membership signature",
        },
    }


def _strict_disjoint_blocks(task_frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        target: (
            frame.loc[~frame["patient_id"].astype(str).isin(RIH_DUAL_PATIENTS)].copy()
            if target in {"rih_primary", "rih_metastatic"}
            else frame.copy()
        )
        for target, frame in task_frames.items()
    }


def _macro_block(
    target_frames: Mapping[str, pd.DataFrame],
    *,
    n_bootstrap: int,
    task: str,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    per_target: dict[str, float] = {}
    per_seed: dict[str, float] = {}
    component_support: dict[str, dict[str, Any]] = {}
    target_draws = []
    for target in TARGET_ORDER:
        frame = target_frames[target].reset_index(drop=True)
        labels = frame["label"].to_numpy(dtype=int)
        scores = frame["mean_logit_5seed"].to_numpy(float)
        positive = int(labels.sum())
        negative = int(len(labels) - positive)
        support = _support_tier(positive, negative)
        component_support[target] = {
            "n_records": int(len(frame)),
            "positive": positive,
            "negative": negative,
            "support": support,
            "sparse": support != "BETTER_POWERED_GE20",
        }
        if set(labels) != {0, 1}:
            return {
                "status": "NOT_ESTIMABLE_SINGLE_CLASS",
                "target": target,
                "component_support": component_support,
            }
        per_target[target] = float(roc_auc_score(labels, scores))
        indices = _stratified_indices(
            frame,
            n_bootstrap=n_bootstrap,
            stream=f"{task}:macro:{target}",
            strata=("dataset", "label"),
        )
        target_draws.append(_auc_draws(labels, scores, indices))
    for seed in MODEL_SEEDS:
        per_seed[str(seed)] = float(
            np.mean(
                [
                    roc_auc_score(
                        target_frames[target]["label"].to_numpy(int),
                        target_frames[target][f"logit_seed{seed}"].to_numpy(float),
                    )
                    for target in TARGET_ORDER
                ]
            )
        )
    seed_values = np.asarray(list(per_seed.values()), dtype=float)
    draws = np.mean(np.vstack(target_draws), axis=0)
    arrays[f"external__{task}__equal_cohort_macro_strict_disjoint"] = draws
    point = float(np.mean(list(per_target.values())))
    support_rank = {
        "VERY_SPARSE_LT10": 0,
        "EXPLORATORY_10_19": 1,
        "BETTER_POWERED_GE20": 2,
    }
    worst_support = min(
        (str(block["support"]) for block in component_support.values()),
        key=support_rank.__getitem__,
    )
    positive = int(sum(int(block["positive"]) for block in component_support.values()))
    negative = int(sum(int(block["negative"]) for block in component_support.values()))
    return {
        "status": "ESTIMABLE",
        "role": "secondary equal-cohort macro on strict-disjoint target rosters",
        "cohort_count": len(TARGET_ORDER),
        "n_records": positive + negative,
        "positive": positive,
        "negative": negative,
        "support": worst_support,
        "sparse": worst_support != "BETTER_POWERED_GE20",
        "support_basis": "worst per-cohort class support; pooled counts are descriptive only",
        "component_support": component_support,
        "per_target_ensemble_auroc": per_target,
        "per_seed_macro_auroc": per_seed,
        "seed_auroc_mean": float(seed_values.mean()),
        "seed_auroc_sample_sd": float(seed_values.std(ddof=1)),
        "five_seed_refit_ensemble": {
            "auroc": point,
            "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
            "bootstrap": "independent target-by-label patient resamples, then arithmetic mean of five target AUROCs draw-wise",
        },
    }


def _internal_block(training_root: Path, task: str) -> dict[str, Any]:
    fine_results = source._validate_fine_analysis(  # noqa: SLF001
        training_root, require_no_controls=False
    )
    frame, per_seed = source._ensemble(training_root, "fine", task)  # noqa: SLF001
    values = np.asarray([per_seed[str(seed)] for seed in MODEL_SEEDS], dtype=float)
    fold_values = {
        str(fold): float(
            roc_auc_score(
                frame.loc[frame["k_fold"].eq(fold), "label"],
                frame.loc[frame["k_fold"].eq(fold), "mean_logit"],
            )
        )
        for fold in range(5)
    }
    folds = np.asarray(list(fold_values.values()), dtype=float)
    governed = fine_results["fine"]["rungs"][task]
    if governed["per_seed_auroc"] != per_seed:
        raise ContractError(f"{task}: internal per-seed OOF replay drifted")
    return {
        "design": "honest inherited five-fold OOF on TCGA+SurGen primaries",
        "patients": int(len(frame)),
        "positive": int(frame["label"].sum()),
        "negative": int(frame["label"].eq(0).sum()),
        "per_seed_pooled_oof_auroc": per_seed,
        "seed_auroc_mean": float(values.mean()),
        "seed_auroc_sample_sd": float(values.std(ddof=1)),
        "five_fold_ensemble_auroc": fold_values,
        "fold_auroc_mean": float(folds.mean()),
        "fold_auroc_sample_sd": float(folds.std(ddof=1)),
        "fold_sd_role": "descriptive across the five held-out OOF folds; folds are not inference units",
        "five_seed_oof_ensemble": {
            "auroc": float(governed["five_seed_ensemble"]["estimate"]),
            "ci95": list(governed["five_seed_ensemble"]["ci95_two_sided"]),
            "source": fine_results["fine_bootstrap_distributions"],
        },
    }


def compute_analysis(
    root: Path,
    outcome_slides: Mapping[str, pd.DataFrame],
    *,
    n_bootstrap: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame]:
    task_frames, patient_export = _patient_scores(root, outcome_slides)
    contract = _read_json(contract_path(root))
    training_root = Path(str(contract["training_root"]))
    arrays: dict[str, np.ndarray] = {}
    tasks: dict[str, Any] = {}
    report_rows: list[dict[str, Any]] = []
    for task in TASKS:
        primitive: dict[str, Any] = {}
        for target in TARGET_ORDER:
            primitive[target] = _metric_block(
                task_frames[task][target],
                n_bootstrap=n_bootstrap,
                stream=f"{task}:primitive:{target}",
                arrays=arrays,
                array_key=f"external__{task}__{target}",
            )
        primary_frame = pd.concat(
            [
                task_frames[task][target]
                for target in TARGET_ORDER
                if TARGETS[target].role == "primary"
            ],
            ignore_index=True,
        )
        metastatic_frame = pd.concat(
            [
                task_frames[task][target]
                for target in TARGET_ORDER
                if TARGETS[target].role == "metastatic"
            ],
            ignore_index=True,
        )
        all_record = pd.concat(task_frames[task].values(), ignore_index=True)
        strict_blocks = _strict_disjoint_blocks(task_frames[task])
        strict = pd.concat(strict_blocks.values(), ignore_index=True)
        observed_census = {
            "primary": (int(primary_frame["label"].sum()), int(primary_frame["label"].eq(0).sum())),
            "metastatic": (
                int(metastatic_frame["label"].sum()),
                int(metastatic_frame["label"].eq(0).sum()),
            ),
            "strict": (int(strict["label"].sum()), int(strict["label"].eq(0).sum())),
            "all_record": (int(all_record["label"].sum()), int(all_record["label"].eq(0).sum())),
        }
        expected_census = {
            "primary": EXPECTED_PRIMARY_CENSUS[task],
            "metastatic": EXPECTED_METASTATIC_CENSUS[task],
            "strict": EXPECTED_STRICT_DISJOINT_CENSUS[task],
            "all_record": EXPECTED_ALL_RECORD_CENSUS[task],
        }
        if observed_census != expected_census:
            raise ContractError(f"{task}: aggregate external censes drifted: {observed_census}")
        if all_record["patient_id"].nunique() != EXPECTED_ALL_RECORD_UNIQUE_CLUSTERS[task]:
            raise ContractError(f"{task}: all-record unique-patient cluster census drifted")
        external = {
            "per_cohort": primitive,
            "primary_only": _metric_block(
                primary_frame,
                n_bootstrap=n_bootstrap,
                stream=f"{task}:primary_only",
                arrays=arrays,
                array_key=f"external__{task}__primary_only",
            ),
            "metastatic_only": _metric_block(
                metastatic_frame,
                n_bootstrap=n_bootstrap,
                stream=f"{task}:metastatic_only",
                arrays=arrays,
                array_key=f"external__{task}__metastatic_only",
            ),
            "strict_disjoint_combined": {
                **_metric_block(
                    strict,
                    n_bootstrap=n_bootstrap,
                    stream=f"{task}:strict_disjoint_combined",
                    arrays=arrays,
                    array_key=f"external__{task}__strict_disjoint_combined",
                ),
                "role": "headline patient-pooled AUROC",
                "dual_role_exclusion": sorted(RIH_DUAL_PATIENTS),
                "all_eight_excluded_from_both_rih_roles": True,
            },
            "equal_cohort_macro_strict_disjoint": _macro_block(
                strict_blocks,
                n_bootstrap=n_bootstrap,
                task=task,
                arrays=arrays,
            ),
            "patient_role_pooled_clustered_sensitivity": {
                **_metric_block(
                    all_record,
                    n_bootstrap=n_bootstrap,
                    stream=f"{task}:patient_role_clustered",
                    arrays=arrays,
                    array_key=f"external__{task}__patient_role_clustered",
                    clustered=True,
                ),
                "role": "PATIENT-ROLE-POOLED_CLUSTERED_SENSITIVITY; secondary, never headline",
                "n_unique_patient_clusters": int(all_record["patient_id"].nunique()),
                "point_auroc_weights_patient_role_records": True,
            },
        }
        tasks[task] = {
            "display": TASK_DISPLAY[task],
            "internal_oof": _internal_block(training_root, task),
            "external_refit": external,
        }
        internal = tasks[task]["internal_oof"]
        report_rows.append(
            {
                "task": task,
                "scope": "internal_oof",
                "population": "TCGA+SurGen-primary",
                "n": internal["patients"],
                "positive": internal["positive"],
                "negative": internal["negative"],
                "seed_mean": internal["seed_auroc_mean"],
                "seed_sample_sd": internal["seed_auroc_sample_sd"],
                "fold_mean": internal["fold_auroc_mean"],
                "fold_sample_sd": internal["fold_auroc_sample_sd"],
                "per_fold_auroc": internal["five_fold_ensemble_auroc"],
                "ensemble_auroc": internal["five_seed_oof_ensemble"]["auroc"],
                "ci95": internal["five_seed_oof_ensemble"]["ci95"],
            }
        )
        rows = {
            **primitive,
            **{
                key: external[key]
                for key in (
                    "primary_only",
                    "metastatic_only",
                    "strict_disjoint_combined",
                    "equal_cohort_macro_strict_disjoint",
                    "patient_role_pooled_clustered_sensitivity",
                )
            },
        }
        for population, block in rows.items():
            ensemble = block.get("five_seed_refit_ensemble") or {}
            report_rows.append(
                {
                    "task": task,
                    "scope": "external_refit",
                    "population": population,
                    "n": block.get("n_records"),
                    "positive": block.get("positive"),
                    "negative": block.get("negative"),
                    "seed_mean": block.get("seed_auroc_mean"),
                    "seed_sample_sd": block.get("seed_auroc_sample_sd"),
                    "ensemble_auroc": ensemble.get("auroc"),
                    "ci95": ensemble.get("ci95"),
                    "status": block.get("status"),
                    "support": block.get("support"),
                    "sparse": block.get("sparse"),
                    "support_basis": block.get("support_basis"),
                    "component_support": block.get("component_support"),
                }
            )
    if len(arrays) != 50 or len(report_rows) != 55:
        raise ContractError(
            f"analysis inventory drifted: arrays={len(arrays)}, rows={len(report_rows)}"
        )
    results = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "complete_after_sealed_zero_shot_inference",
        "encoder": "UNI-v1",
        "model_seeds": list(MODEL_SEEDS),
        "tasks": tasks,
        "report_rows": report_rows,
        "estimands": {
            "mean_plus_minus_sd": "arithmetic mean and sample SD (ddof=1) of five seed-specific patient AUROCs",
            "ensemble": "mean slide native logit within patient/seed, then mean native patient logits over seeds 42..46",
            "ci": f"pointwise 95% percentile CI from {n_bootstrap:,} fixed-composition patient bootstrap draws",
            "combined_headline": "strict-disjoint patient-pooled AUROC after excluding all eight RIH dual-role patients from both roles",
            "primary_only": "patient-pooled CPTAC-primary + Orion-primary + RIH-primary",
            "metastatic_only": "patient-pooled RIH-metastatic + SurGen-metastatic",
            "macro": "secondary arithmetic mean of five strict-disjoint target AUROCs",
            "clustered_sensitivity": "secondary all-record pooled AUROC; unique patients resampled within full target/label membership signatures",
        },
        "claim_boundary": {
            "external_target_use": "score-only after model freeze",
            "new_fits": 0,
            "target_calibration": False,
            "target_threshold_selection": False,
            "control_artifacts_absent_through_outcome_join": True,
            "controls_begin_only_after_external_analysis": True,
            "multiplicity_claim": False,
            "sparse_rows": "reported because explicitly requested; VERY_SPARSE_LT10 rows are descriptive",
            "orion": "retrospective model-specific zero-shot test; outcomes were historically accessed by other studies",
            "surgen_metastatic": "patient-disjoint held-out metastatic test with acquisition-family exposure to SurGen primaries",
        },
    }
    return results, arrays, patient_export


def analyze(
    root: Path,
    *,
    apply: bool,
    n_bootstrap: int,
    test_only_noncanonical_parameters: bool = False,
) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    seal = verify_inference_seal(output, require_no_controls=True)
    if n_bootstrap != N_BOOTSTRAP and not test_only_noncanonical_parameters:
        raise ContractError("production analysis requires exactly 10,000 bootstrap draws")
    if not apply:
        return {
            "status": "DRY_RUN_READY_AFTER_INFERENCE_SEAL",
            "inference_seal": _artifact(inference_seal_path(output)),
            "n_bootstrap": n_bootstrap,
            "target_outcomes_opened": False,
        }
    if analysis_root(output).exists() or analysis_root(output).is_symlink():
        return _validate_analysis(output)
    outcome_slides, outcome_identities = _open_outcomes_after_seal(output)
    results, arrays, patients = compute_analysis(output, outcome_slides, n_bootstrap=n_bootstrap)
    _assert_no_control_artifacts_for_external_stage(output)
    directory = analysis_root(output)
    analysis_contract = {
        "schema_version": SCHEMA_VERSION,
        "status": "outcome_join_governed_after_inference_seal",
        "created_utc": _utcnow(),
        "contract": _artifact(contract_path(output)),
        "inference_seal": _artifact(inference_seal_path(output)),
        "seal_status_observed_before_outcome_open": seal["status"],
        "outcome_sources": outcome_identities,
        "bootstrap_draws": n_bootstrap,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "target_outcomes_opened_after_inference_seal": True,
        "control_artifacts_absent_at_outcome_join": True,
    }
    _write_json_once(directory / "contract.json", analysis_contract)
    _write_parquet_once(directory / "patient_native_logits.parquet", patients)
    _write_npz_once(directory / "bootstrap_distributions.npz", arrays)
    results.update(
        {
            "created_utc": _utcnow(),
            "analysis_contract": _artifact(directory / "contract.json"),
            "inference_seal": _artifact(inference_seal_path(output)),
            "patient_native_logits": _artifact(directory / "patient_native_logits.parquet"),
            "bootstrap_distributions": _artifact(directory / "bootstrap_distributions.npz"),
        }
    )
    _write_json_once(directory / "results.json", results)
    _assert_no_control_artifacts_for_external_stage(output)
    _write_json_once(
        directory / "analysis_completion.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_after_sealed_zero_shot_inference",
            "created_utc": _utcnow(),
            "analysis_contract": _artifact(directory / "contract.json"),
            "patient_native_logits": _artifact(directory / "patient_native_logits.parquet"),
            "bootstrap_distributions": _artifact(directory / "bootstrap_distributions.npz"),
            "results": _artifact(directory / "results.json"),
            "bootstrap_array_count": len(arrays),
            "report_row_count": len(results["report_rows"]),
        },
    )
    return _validate_analysis(output)


def _validate_analysis(root: Path) -> dict[str, Any]:
    directory = analysis_root(root)
    completion = _read_json(directory / "analysis_completion.json")
    results = _read_json(directory / "results.json")
    contract = _read_json(directory / "contract.json")
    expected = {
        "analysis_contract": _artifact(directory / "contract.json"),
        "patient_native_logits": _artifact(directory / "patient_native_logits.parquet"),
        "bootstrap_distributions": _artifact(directory / "bootstrap_distributions.npz"),
        "results": _artifact(directory / "results.json"),
    }
    if (
        completion.get("status") != "complete_after_sealed_zero_shot_inference"
        or any(completion.get(key) != value for key, value in expected.items())
        or completion.get("bootstrap_array_count") != 50
        or completion.get("report_row_count") != 55
        or results.get("status") != "complete_after_sealed_zero_shot_inference"
        or results.get("model_seeds") != list(MODEL_SEEDS)
        or set(results.get("tasks", {})) != set(TASKS)
        or len(results.get("report_rows", [])) != 55
        or contract.get("status") != "outcome_join_governed_after_inference_seal"
        or contract.get("contract") != _artifact(contract_path(root))
        or contract.get("inference_seal") != _artifact(inference_seal_path(root))
        or contract.get("seal_status_observed_before_outcome_open") != "sealed_before_outcome_join"
        or contract.get("control_artifacts_absent_at_outcome_join") is not True
        or set(contract.get("outcome_sources", {})) != set(TARGET_ORDER)
    ):
        raise ContractError("external analysis contract/results drifted")
    for target, identity in contract["outcome_sources"].items():
        path = _verify_artifact(identity, context=f"analysis outcome/{target}")
        if (
            path != TARGETS[target].outcome_source.resolve()
            or identity["sha256"] != TARGETS[target].outcome_sha256
        ):
            raise ContractError(f"analysis outcome identity drifted: {target}")
    patients = pd.read_parquet(directory / "patient_native_logits.parquet")
    required = {
        "task",
        "patient_id",
        "dataset",
        "role",
        "label",
        "mean_logit_5seed",
        *(f"logit_seed{seed}" for seed in MODEL_SEEDS),
    }
    if required - set(patients) or set(patients["task"]) != set(TASKS):
        raise ContractError("external patient-native-logit table drifted")
    logits = patients[["mean_logit_5seed", *(f"logit_seed{seed}" for seed in MODEL_SEEDS)]]
    if not np.isfinite(logits.to_numpy(float)).all():
        raise ContractError("external patient-native-logit table has non-finite values")
    try:
        bootstrap_length = int(contract.get("bootstrap_draws", -1))
        if bootstrap_length < 1:
            raise ContractError("analysis bootstrap length is invalid")
        with np.load(directory / "bootstrap_distributions.npz", allow_pickle=False) as archive:
            if len(archive.files) != 50 or any(
                archive[name].dtype != np.float64
                or archive[name].shape != (bootstrap_length,)
                or not np.isfinite(archive[name]).all()
                for name in archive.files
            ):
                raise ContractError("external bootstrap NPZ drifted")
    except (OSError, ValueError) as exc:
        raise ContractError("invalid external bootstrap NPZ") from exc
    _validate_output_namespace(root, terminal="analysis")
    return results


def verify(root: Path) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    _load_contract(output, deep=True, require_no_controls=False)
    stages: dict[str, str] = {"contract": "PASS"}
    if preflight_path(output).is_file():
        _load_preflight(output, require_no_controls=False)
        stages["preflight"] = "PASS"
    if inference_seal_path(output).is_file():
        verify_inference_seal(output)
        stages["inference"] = "PASS"
    if (analysis_root(output) / "analysis_completion.json").is_file():
        results = _validate_analysis(output)
        stages["analysis"] = "PASS"
        source_graph = {
            "controller": _artifact(Path(__file__).resolve()),
            "controller_test": _artifact(
                REPO / "tests/test_aim3_tcga_surgen_primary_fine_external_campaign.py"
            ),
            "external_contract": _artifact(contract_path(output)),
            "fine_training_seal": _read_json(contract_path(output))["training"][
                "fine_training_seal"
            ],
            "fine_analysis_completion": _read_json(contract_path(output))["training"][
                "fine_analysis_completion"
            ],
            "inference_seal": _artifact(inference_seal_path(output)),
            "analysis_contract": results["analysis_contract"],
            "external_results": _artifact(analysis_root(output) / "results.json"),
            "analysis_completion": _artifact(analysis_root(output) / "analysis_completion.json"),
        }
        return {
            "status": "PASS",
            "stages": stages,
            "terminal_artifact_census": _validate_output_namespace(output, terminal="analysis"),
            "direct_source_graph": source_graph,
        }
    _validate_output_namespace(
        output, terminal="inference" if inference_seal_path(output).is_file() else None
    )
    return {"status": "PASS", "stages": stages}


def status(root: Path) -> dict[str, Any]:
    output = validate_output_root(root)
    completed = sum(score_path(output, job.task, job.seed).is_file() for job in score_jobs())
    return {
        "output_root": str(output),
        "prepared": contract_path(output).is_file(),
        "preflight": preflight_path(output).is_file(),
        "scores_complete": completed,
        "scores_expected": 25,
        "inference_sealed": inference_seal_path(output).is_file(),
        "analysis_complete": (analysis_root(output) / "analysis_completion.json").is_file(),
    }


def plan(root: Path) -> dict[str, Any]:
    output = validate_output_root(root)
    return {
        "status": "PLAN_ONLY_NO_WRITES",
        "campaign": CAMPAIGN,
        "output_root": str(output),
        "training_root": str(TRAINING_ROOT),
        "tasks": list(TASKS),
        "model_seeds": list(MODEL_SEEDS),
        "targets": list(TARGET_ORDER),
        "label_blind_slides": 479,
        "score_jobs": 25,
        "score_rows": 11_975,
        "new_fits": 0,
        "max_parallel_workers": MAX_WORKERS,
        "workflow": ["prepare", "preflight", "score", "inference_seal", "analyze", "verify"],
        "launch_gate": "fine training seal and fine analysis completion must both exist and authenticate",
        "phase_order_gate": (
            "no matched-WT or repeated-WT artifact may exist through external outcome analysis"
        ),
    }


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        return child

    common("plan")
    prepare_parser = common("prepare")
    prepare_parser.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
    prepare_parser.add_argument("--apply", action="store_true")
    preflight_parser = common("preflight")
    preflight_parser.add_argument("--apply", action="store_true")
    score_parser = common("score")
    score_parser.add_argument("--apply", action="store_true")
    score_parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    score_parser.add_argument("--device", choices=("cuda",), default="cuda")
    score_parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    analyze_parser = common("analyze")
    analyze_parser.add_argument("--apply", action="store_true")
    analyze_parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    analyze_parser.add_argument("--test-only-noncanonical-parameters", action="store_true")
    common("verify")
    common("status")
    internal = common("_score-one")
    internal.add_argument("--task", choices=TASKS, required=True)
    internal.add_argument("--seed", choices=MODEL_SEEDS, type=int, required=True)
    internal.add_argument("--device", choices=("cuda",), default="cuda")
    internal.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        _print(plan(args.output_root))
    elif args.command == "prepare":
        _print(prepare(args.output_root, args.training_root, apply=args.apply))
    elif args.command == "preflight":
        _print(preflight(args.output_root, apply=args.apply))
    elif args.command == "score":
        _print(
            score(
                args.output_root,
                apply=args.apply,
                max_workers=args.max_workers,
                device=args.device,
                num_workers=args.num_workers,
            )
        )
    elif args.command == "analyze":
        _print(
            analyze(
                args.output_root,
                apply=args.apply,
                n_bootstrap=args.n_bootstrap,
                test_only_noncanonical_parameters=args.test_only_noncanonical_parameters,
            )
        )
    elif args.command == "verify":
        _print(verify(args.output_root))
    elif args.command == "status":
        _print(status(args.output_root))
    elif args.command == "_score-one":
        _score_one(
            args.output_root,
            args.task,
            args.seed,
            device=args.device,
            num_workers=args.num_workers,
        )
    else:  # pragma: no cover
        raise ContractError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
