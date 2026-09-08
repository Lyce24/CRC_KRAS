#!/usr/bin/env python3
"""Governed five-seed extension for every official Aim-2 LOCO MIL arm.

This is an additive overlay.  It never writes into either sealed three-seed
lineage.  Seeds 42--44 are adopted by content identity and seeds 45--46 are
trained from the same frozen source manifests, predefined folds, and material
recipe.  The controlling analysis has eight LOCO directions (four cohort-
family and four sibling-stratum directions); the completed size-matched RIH
arm is retained as a secondary sensitivity because it is also an official MIL
run in the study.

The public workflow is deliberately staged::

    python aim2_loco_five_seed_extension.py plan
    python aim2_loco_five_seed_extension.py manifest --apply
    python aim2_loco_five_seed_extension.py preflight --deep
    python aim2_loco_five_seed_extension.py train --dry-run
    python aim2_loco_five_seed_extension.py train --apply --max-workers 6
    python aim2_loco_five_seed_extension.py score --apply
    python aim2_loco_five_seed_extension.py seal-inference
    python aim2_loco_five_seed_extension.py report --apply
    python aim2_loco_five_seed_extension.py verify --deep

Outcome columns are absent from all inference manifests.  The report command
will not interpret or join target outcomes until the complete inference roster
has been sealed.  A partial immutable job is never resumed or overwritten.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import evaluate, lineage, paths  # noqa: E402
from oceanpath.eval.external import sigmoid  # noqa: E402

SCHEMA_VERSION = 1
CAP = 8_192
N_FOLDS = 5
STEP_BUDGET = 6_060
MIN_EPOCHS = 3
ADOPTED_SEEDS: tuple[int, ...] = (42, 43, 44)
NEW_SEEDS: tuple[int, ...] = (45, 46)
ALL_SEEDS: tuple[int, ...] = (*ADOPTED_SEEDS, *NEW_SEEDS)
DEFAULT_MAX_WORKERS = 6
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_817
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "final_v9_mil_5seed_expansion_v1_20260823"
)
COMPONENT = "aim2_loco"

FAMILY_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim2_cap8192_v4_20260819/e2a"
)
FAMILY_SOURCE_CV_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/e2a/source_cv")
SIBLING_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_final_v8_complete_v1_20260822/e2ad"
)
E2MET_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_final_v8_complete_v1_20260822/e2met"
)
CPHT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_final_v8_complete_v1_20260822/cpht"
)
CPHT_A_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_final_v8_complete_v1_20260822/cpht_a_v2"
)
FAMILY_ORION_ROOT = REPO / "reports/reruns/final_v8_additions_20260822/e2cpht/scores"
SEALED_ROOTS: tuple[Path, ...] = (
    FAMILY_ROOT,
    FAMILY_SOURCE_CV_ROOT,
    SIBLING_ROOT,
    E2MET_ROOT,
    CPHT_ROOT,
    CPHT_A_ROOT,
    FAMILY_ORION_ROOT,
)


class ContractError(RuntimeError):
    """Fail-closed violation of the immutable extension contract."""


@dataclass(frozen=True)
class ArmSpec:
    name: str
    design: str
    slug: str
    target_name: str
    model_name: str
    source_manifest: Path
    split_root: Path
    split_file: Path
    primary_target: Path
    old_root: Path
    old_source_cv_root: Path
    expected_source_patients: int
    expected_source_slides: int
    source_sha256: str
    split_sha256: str
    controlling: bool = True


def _family_spec(
    slug: str,
    target_name: str,
    n_patients: int,
    n_slides: int,
    source_sha256: str,
    split_sha256: str,
    *,
    size_matched: bool = False,
) -> ArmSpec:
    if size_matched:
        source = FAMILY_ROOT / "inputs/manifests/aim1_e2a_rih_sm_source_seed42.csv"
        split_root = FAMILY_ROOT / "inputs/splits"
        split_file = split_root / "aim1_e2a_rih_sm/aim1_balanced5/splits.parquet"
        source_cv_root = FAMILY_ROOT / "source_cv"
        target_slug = "rih"
    else:
        source = Path(f"/mnt/d/YC.Liu/manifests/colon/aim1_e2a_{slug}_source_seed42.csv")
        split_root = REPO / "outputs/splits"
        split_file = split_root / f"aim1_e2a_{slug}/aim1_balanced5/splits.parquet"
        source_cv_root = FAMILY_SOURCE_CV_ROOT
        target_slug = slug
    return ArmSpec(
        name=f"family_{slug}",
        design="family_loco_size_matched" if size_matched else "family_loco",
        slug=slug,
        target_name=target_name,
        model_name=f"e2a_{slug}",
        source_manifest=source,
        split_root=split_root,
        split_file=split_file,
        primary_target=Path(
            f"/mnt/d/YC.Liu/manifests/colon/aim1_e2a_{target_slug}_primary.csv"
        ),
        old_root=FAMILY_ROOT,
        old_source_cv_root=source_cv_root,
        expected_source_patients=n_patients,
        expected_source_slides=n_slides,
        source_sha256=source_sha256,
        split_sha256=split_sha256,
        controlling=not size_matched,
    )


def _sibling_spec(
    slug: str,
    target_name: str,
    n_patients: int,
    n_slides: int,
    source_sha256: str,
    split_sha256: str,
) -> ArmSpec:
    manifest_root = SIBLING_ROOT / "inputs/manifests"
    split_root = SIBLING_ROOT / "inputs/splits"
    return ArmSpec(
        name=f"sibling_{slug}",
        design="sibling_stratum_loco",
        slug=slug,
        target_name=target_name,
        model_name=f"e2ad_{slug}",
        source_manifest=manifest_root / f"aim1_e2ad_{slug}_source.csv",
        split_root=split_root,
        split_file=split_root / f"aim1_e2ad_{slug}/aim1_balanced5/splits.parquet",
        primary_target=manifest_root / f"aim1_e2ad_{slug}_target_primary.csv",
        old_root=SIBLING_ROOT,
        old_source_cv_root=SIBLING_ROOT / "source_cv",
        expected_source_patients=n_patients,
        expected_source_slides=n_slides,
        source_sha256=source_sha256,
        split_sha256=split_sha256,
    )


ARMS: dict[str, ArmSpec] = {
    spec.name: spec
    for spec in (
        _family_spec(
            "cptac", "CPTAC", 1_392, 1_544,
            "56e0a0b671f6502d2983b486bafabb7f64220f2798745c116151fb73f0787e50",
            "956f32867945d24e72720ed71f41daecfd8894be566f29f0f428b4569ecf19c2",
        ),
        _family_spec(
            "rih", "RIH", 1_333, 1_487,
            "8827430d02fdfd03422b31e7b5255c47a695d76f7d1ce1a059c9294a5ecda396",
            "031a9cea3706d656e7fde899fe6da822973eda489d79bb25b3dc0d7e174cc7d4",
        ),
        _family_spec(
            "surgen", "SurGen", 749, 761,
            "0785f536c5a25845b33d99cdb056bc26681d76016aa90b61ab1c274bb02c325e",
            "b0d5c616d3438c796890a0dba89bf1e4cec9dbca71a220e56a022b980077e956",
        ),
        _family_spec(
            "tcga", "TCGA", 984, 1_134,
            "c6920390c847dadf132efad64b02a20a912c5a2cd0d820e9e6a795421f035b70",
            "809f096119607a997080ed690d5c8d368073fae9b5735432207981069ee84340",
        ),
        _family_spec(
            "rih_sm", "RIH size-matched", 749, 819,
            "3c93903c39b8fceef2c1284a0decfe2b9fa9973656d23aac9ecc98b835928961",
            "b4c30d133bda548d12abd08d2a6ac71738ed8e2e48fef6341aa5bec6791b90ef",
            size_matched=True,
        ),
        _sibling_spec(
            "sr386", "SR386", 1_073, 1_229,
            "e25af4c2885f7845e910ba5097282c1b0abb61675fc5c09702ba567514c98041",
            "09da56c81af386d6f423bfb9032acbf62364c9daf2f38159b1ada32f59d79c70",
        ),
        _sibling_spec(
            "sr1482", "SR1482", 1_162, 1_174,
            "8b43741491f27a30e5b0a36bf119f4805241ec69da63c6b9bb0135509e1febb1",
            "59d8fa45027628b17b7d1356a7f4e41d56d24dc41b9f085a9b86e505e7744f76",
        ),
        _sibling_spec(
            "tcga_coad", "TCGA-COAD", 1_112, 1_262,
            "88392c81c29f426cf423adf559bbc786fbb10d56aa8209a94e8db37f35926eb3",
            "d5d4d7b360aead0c1605f4ef6f7030c54ea452ebb477a41074b81c573aec81f7",
        ),
        _sibling_spec(
            "tcga_read", "TCGA-READ", 1_358, 1_514,
            "ab96201b42ab2a6e88348cdd626b5c02327cd2bbaa244b6602aaae99c8761cc6",
            "ebc5ff8cb4f4fd5694378c7b01585f6d1f11254e797033581366528a875c6322",
        ),
    )
}
ARM_NAMES: tuple[str, ...] = tuple(ARMS)
CONTROLLING_ARMS: tuple[str, ...] = tuple(name for name, spec in ARMS.items() if spec.controlling)
MET_TARGETS: tuple[str, ...] = ("rih_m", "sr1482_m")

LABEL_BLIND_ALLOWED: tuple[str, ...] = (
    "slide_id",
    "patient_id",
    "cohort",
    "subcohort",
    "specimen_role",
    "role",
    "liver_class",
    "mpp",
    "mpp_source",
    "patch_count",
    "exclude_neoadjuvant",
    "exclude_ambiguous_crc15",
)
FORBIDDEN_OUTCOME_COLUMNS = {
    "label", "target_label", "kras", "kras_status", "kras_mutant",
    "msi", "msi_dmmr", "braf", "outcome",
}


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _artifact(path: Path) -> dict[str, Any]:
    return lineage.artifact_identity(path)


def component_root(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return Path(output_root).expanduser().resolve() / COMPONENT


def assert_safe_output_root(output_root: Path) -> Path:
    raw = Path(output_root).expanduser()
    if not raw.is_absolute():
        raise ContractError("--output-root must be an absolute path")
    lexical = raw.absolute()
    root = raw.resolve(strict=False)
    if lexical != root:
        raise ContractError(f"Output root must not traverse symlinks: {raw} -> {root}")
    cursor = raw
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise ContractError(f"Output-root path contains a symlink: {cursor}")
        cursor = cursor.parent
    component = root / COMPONENT
    for sealed in SEALED_ROOTS:
        frozen = sealed.resolve()
        if component == frozen or component.is_relative_to(frozen) or frozen.is_relative_to(component):
            raise ContractError(f"Extension output overlaps sealed input root: {frozen}")
    production = DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    temporary = Path("/tmp").resolve()
    if root != production and root != temporary and temporary not in root.parents:
        raise ContractError(
            f"Production output root must be exactly {production}; tests may use /tmp"
        )
    return component


def _write_json_once(path: Path, value: Any) -> None:
    lineage.write_json_once(path, value)


def _write_csv_once(path: Path, frame: pd.DataFrame) -> None:
    lineage.write_text_once(path, frame.to_csv(index=False))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object: {path}")
    return value


def source_cv_dir(output_root: Path, arm: str, seed: int) -> Path:
    return component_root(output_root) / f"train/source_cv/cap{CAP}/{arm}/seed{seed}"


def refit_dir(output_root: Path, arm: str, seed: int) -> Path:
    return component_root(output_root) / f"train/refit/pb_cap{CAP}/{arm}/seed{seed}"


def checkpoint_path(output_root: Path, arm: str, seed: int) -> Path:
    if seed in ADOPTED_SEEDS:
        return old_refit_dir(arm, seed) / "final/refit/model.ckpt"
    if seed not in NEW_SEEDS:
        raise ValueError(f"Uncontracted seed: {seed}")
    return refit_dir(output_root, arm, seed) / "final/refit/model.ckpt"


def old_source_cv_dir(arm: str, seed: int) -> Path:
    spec = ARMS[arm]
    return spec.old_source_cv_root / f"cap{CAP}/{spec.slug}/seed{seed}"


def old_refit_dir(arm: str, seed: int) -> Path:
    spec = ARMS[arm]
    return spec.old_root / f"train/pb_cap{CAP}/{spec.slug}/seed{seed}"


def old_primary_score_path(arm: str, seed: int) -> Path:
    spec = ARMS[arm]
    return spec.old_root / f"scores/pb_cap{CAP}_{spec.slug}_seed{seed}_primary.parquet"


def _met_scorer(arm: str) -> str:
    return f"heldout_{ARMS[arm].slug}"


def old_met_score_path(arm: str, target: str, seed: int) -> Path:
    return E2MET_ROOT / f"scores/{_met_scorer(arm)}_{target}_seed{seed}.parquet"


def old_orion_score_path(arm: str, seed: int) -> Path:
    spec = ARMS[arm]
    root = FAMILY_ORION_ROOT if spec.design.startswith("family_loco") else CPHT_ROOT / "scores"
    return root / f"loco_{spec.slug}_seed{seed}_orion.parquet"


def label_blind_path(output_root: Path, kind: str, name: str) -> Path:
    return component_root(output_root) / f"inputs/label_blind/{kind}/{name}.csv"


def score_path(output_root: Path, arm: str, target: str, seed: int) -> Path:
    if target == "primary":
        return component_root(output_root) / f"scores/primary/{arm}/seed{seed}.parquet"
    if target in MET_TARGETS:
        return component_root(output_root) / f"scores/e2met/{target}/{arm}/seed{seed}.parquet"
    if target == "orion":
        return component_root(output_root) / f"scores/e2cpht/orion/{arm}/seed{seed}.parquet"
    raise ValueError(target)


def contract_path(output_root: Path) -> Path:
    return component_root(output_root) / "contract.json"


def training_inventory_path(output_root: Path) -> Path:
    return component_root(output_root) / "jobs/training_jobs.json"


def score_inventory_path(output_root: Path) -> Path:
    return component_root(output_root) / "jobs/score_jobs.json"


def inference_seal_path(output_root: Path) -> Path:
    return component_root(output_root) / "inference/inference_seal.json"


def preflight_receipt_path(output_root: Path) -> Path:
    return component_root(output_root) / "receipts/deep_preflight.json"


def _receipt(path: Path) -> Path:
    return path.with_suffix(".receipt.json")


def _blind_frame(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    if "slide_id" not in frame or "patient_id" not in frame:
        raise ContractError(f"{context}: inference manifest lacks slide_id/patient_id")
    columns = [column for column in LABEL_BLIND_ALLOWED if column in frame.columns]
    out = frame[columns].copy()
    lower = {str(column).lower() for column in out.columns}
    leaked = sorted(lower & FORBIDDEN_OUTCOME_COLUMNS)
    if leaked:
        raise ContractError(f"{context}: outcome columns survived blinding: {leaked}")
    out["slide_id"] = out["slide_id"].astype(str)
    out["patient_id"] = out["patient_id"].astype(str)
    if out["slide_id"].duplicated().any() or out["slide_id"].isna().any():
        raise ContractError(f"{context}: slide IDs are missing or duplicated")
    return out.sort_values("slide_id", kind="stable").reset_index(drop=True)


def _label_blind_inputs() -> dict[tuple[str, str], tuple[Path, pd.DataFrame]]:
    built: dict[tuple[str, str], tuple[Path, pd.DataFrame]] = {}
    for arm, spec in ARMS.items():
        source = spec.primary_target
        built[("primary", arm)] = (
            source,
            _blind_frame(pd.read_csv(source, low_memory=False), context=f"{arm}/primary"),
        )
    for target in MET_TARGETS:
        source = E2MET_ROOT / f"inputs/manifests/{target}.csv"
        built[("e2met", target)] = (
            source,
            _blind_frame(pd.read_csv(source, low_memory=False), context=f"E2-MET/{target}"),
        )
    source = CPHT_ROOT / "inputs/orion_primary.csv"
    built[("e2cpht", "orion")] = (
        source,
        _blind_frame(pd.read_csv(source, low_memory=False), context="E2-CPHT/Orion"),
    )
    return built


def _adopt_file(path: Path) -> dict[str, Any]:
    return _artifact(path)


def build_adoption_roster() -> list[dict[str, Any]]:
    """Hash-adopt the complete three-seed training and inference roster."""

    records: list[dict[str, Any]] = []
    for arm in ARM_NAMES:
        for seed in ADOPTED_SEEDS:
            cv = old_source_cv_dir(arm, seed)
            refit = old_refit_dir(arm, seed)
            semantic_source_cv = _validate_adopted_source_cv(arm, seed)
            semantic_refit = _validate_adopted_refit(arm, seed)
            summary_path = refit / "fit_summary.json"
            summary = _read_json(summary_path)
            checkpoint = summary.get("model")
            if not isinstance(checkpoint, dict):
                raise ContractError(f"Old refit has no checkpoint identity: {summary_path}")
            if Path(str(checkpoint.get("path", ""))).resolve() != (
                refit / "final/refit/model.ckpt"
            ).resolve():
                raise ContractError(f"Old checkpoint path mismatch: {summary_path}")
            primary = old_primary_score_path(arm, seed)
            records.append(
                {
                    "kind": "training",
                    "arm": arm,
                    "seed": seed,
                    "source_cv": {
                        key: _adopt_file(cv / filename)
                        for key, filename in (
                            ("oof", "oof_predictions.parquet"),
                            ("completion", "training_completion.json"),
                            ("identity", "training_identity.json"),
                            ("summary", "cv_summary.json"),
                        )
                    },
                    "refit_summary": _adopt_file(summary_path),
                    "checkpoint": checkpoint,
                    "refit_semantic_validation": semantic_refit,
                }
            )
            records[-1]["source_cv"]["semantic_validation"] = semantic_source_cv
            records.append(
                {
                    "kind": "score",
                    "scope": "primary",
                    "arm": arm,
                    "target": "primary",
                    "seed": seed,
                    "score": _adopt_file(primary),
                    "receipt": _adopt_file(_receipt(primary)),
                }
            )
            if arm in CONTROLLING_ARMS:
                for target in MET_TARGETS:
                    path = old_met_score_path(arm, target, seed)
                    records.append(
                        {
                            "kind": "score",
                            "scope": "e2met",
                            "arm": arm,
                            "target": target,
                            "seed": seed,
                            "score": _adopt_file(path),
                            "receipt": _adopt_file(_receipt(path)),
                        }
                    )
                path = old_orion_score_path(arm, seed)
                records.append(
                    {
                        "kind": "score",
                        "scope": "e2cpht",
                        "arm": arm,
                        "target": "orion",
                        "seed": seed,
                        "score": _adopt_file(path),
                        "receipt": _adopt_file(_receipt(path)),
                    }
                )
    return records


def _command(output_root: Path, internal: str, arm: str, seed: int, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        internal,
        "--output-root",
        str(Path(output_root).resolve()),
        "--arm",
        arm,
        "--seed",
        str(seed),
        *extra,
    ]


def build_training_jobs(output_root: Path = DEFAULT_OUTPUT_ROOT) -> list[dict[str, Any]]:
    """Return 36 schedulable chains representing exactly 108 new MIL fits."""

    assert_safe_output_root(output_root)
    jobs: list[dict[str, Any]] = []
    for arm, spec in ARMS.items():
        for seed in NEW_SEEDS:
            source_job = {
                    "job_id": f"aim2.source_cv.{arm}.seed{seed}",
                    "stage": "source_cv",
                    "arm": arm,
                    "seed": seed,
                    "fit_count": N_FOLDS,
                    "expected_source_patients": spec.expected_source_patients,
                    "depends_on": ["aim2.manifest_contract"],
                    "output": str(source_cv_dir(output_root, arm, seed)),
                    "command": _command(output_root, "_source-cv", arm, seed),
                }
            source_job["external_worker_command"] = _external_worker_command(
                output_root, str(source_job["job_id"])
            )
            source_job["recovery_command"] = _recovery_command(
                output_root, str(source_job["job_id"])
            )
            jobs.append(source_job)
            refit_job = {
                    "job_id": f"aim2.refit.{arm}.seed{seed}",
                    "stage": "refit",
                    "arm": arm,
                    "seed": seed,
                    "fit_count": 1,
                    "expected_source_patients": spec.expected_source_patients,
                    "depends_on": ["aim2.manifest_contract"],
                    "output": str(refit_dir(output_root, arm, seed)),
                    "command": _command(output_root, "_refit", arm, seed),
                }
            refit_job["external_worker_command"] = _external_worker_command(
                output_root, str(refit_job["job_id"])
            )
            refit_job["recovery_command"] = _recovery_command(
                output_root, str(refit_job["job_id"])
            )
            jobs.append(refit_job)
    # Longest source pools first avoids a long tail with six worker slots.
    return sorted(
        jobs,
        key=lambda job: (
            -int(job["expected_source_patients"]),
            0 if job["stage"] == "source_cv" else 1,
            str(job["job_id"]),
        ),
    )


def _external_worker_command(output_root: Path, job_key: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "train-one",
        "--output-root",
        str(Path(output_root).resolve()),
        "--job-key",
        job_key,
        "--external-scheduler",
        "--recover-orphan",
        "--apply",
    ]


def _recovery_command(output_root: Path, job_key: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "quarantine-failed",
        "--output-root",
        str(Path(output_root).resolve()),
        "--job-key",
        job_key,
        "--apply",
    ]


def build_score_jobs(output_root: Path = DEFAULT_OUTPUT_ROOT) -> list[dict[str, Any]]:
    """Return all 66 new, label-blind downstream inference jobs."""

    assert_safe_output_root(output_root)
    jobs: list[dict[str, Any]] = []
    for arm in ARM_NAMES:
        for seed in NEW_SEEDS:
            jobs.append(_score_job(output_root, arm, "primary", seed))
    for arm in CONTROLLING_ARMS:
        for target in MET_TARGETS:
            for seed in NEW_SEEDS:
                jobs.append(_score_job(output_root, arm, target, seed))
        for seed in NEW_SEEDS:
            jobs.append(_score_job(output_root, arm, "orion", seed))
    return jobs


def _score_job(output_root: Path, arm: str, target: str, seed: int) -> dict[str, Any]:
    return {
        "job_id": f"aim2.score.{target}.{arm}.seed{seed}",
        "stage": "score",
        "arm": arm,
        "target": target,
        "seed": seed,
        "fit_count": 0,
        "contains_target_outcomes": False,
        "depends_on": [f"aim2.refit.{arm}.seed{seed}"],
        "output": str(score_path(output_root, arm, target, seed)),
        "command": _command(output_root, "_score-one", arm, seed, "--target", target),
    }


def _source_snapshot() -> list[dict[str, Any]]:
    # Pin the complete local implementation/dependency surface rather than a
    # brittle curated subset.  Training and deterministic report replay both
    # traverse private helpers, so every OceanPath Python source and Hydra
    # config is material until proven otherwise.
    explicit = {
        Path(__file__).resolve(),
        REPO / "tools/study_train.py",
        REPO / "aim2_cross_protocol_transfer.py",
        REPO / "aim2_primary_to_metastatic_transfer.py",
        REPO / "pyproject.toml",
        REPO / "uv.lock",
    }
    sources = explicit | set((REPO / "src/oceanpath").rglob("*.py")) | set(
        (REPO / "configs").rglob("*.yaml")
    )
    return [_artifact(path) for path in sorted(sources, key=lambda item: str(item.resolve()))]


def _feature_store_contract() -> dict[str, Any]:
    """Adopt the already-governed packed UNI-v1 store without rehashing 50 GB."""

    prior = _read_json(E2MET_ROOT / "inputs/e2met_contract.json")
    store = prior.get("feature_store")
    if not isinstance(store, dict):
        raise ContractError("E2-MET contract lacks its governed feature-store identity")
    expected = {
        "path": str(paths.PACKED_FEATURE_DIR.resolve()),
        "source_dir": str(paths.PINNED_FEATURE_DIR.resolve()),
        "feature_dim": 1_024,
    }
    mismatch = {
        key: {"expected": value, "observed": store.get(key)}
        for key, value in expected.items()
        if store.get(key) != value
    }
    if mismatch:
        raise ContractError(f"Governed UNI-v1 feature-store mismatch: {mismatch}")
    for key in ("features", "coords", "index", "meta"):
        identity = store.get(key)
        if not isinstance(identity, dict):
            raise ContractError(f"Feature-store contract lacks {key} identity")
        _validate_identity(identity, deep=False)
    return store


def _input_identity(spec: ArmSpec) -> dict[str, Any]:
    source = _artifact(spec.source_manifest)
    split = _artifact(spec.split_file)
    if source["sha256"] != spec.source_sha256 or split["sha256"] != spec.split_sha256:
        raise ContractError(
            f"{spec.name}: frozen source/split hash changed; "
            f"source={source['sha256']} split={split['sha256']}"
        )
    frame = pd.read_csv(spec.source_manifest, low_memory=False)
    n_patients = int(frame["patient_id"].astype(str).nunique())
    n_slides = int(frame["slide_id"].astype(str).nunique())
    if (n_patients, n_slides) != (spec.expected_source_patients, spec.expected_source_slides):
        raise ContractError(
            f"{spec.name}: source census changed: {(n_patients, n_slides)}"
        )
    return {
        "source_manifest": source,
        "split_file": split,
        "split_root": str(spec.split_root.resolve()),
        "patients": n_patients,
        "slides": n_slides,
        "fold_manifest_reused_for_every_seed": True,
    }


def _contract_static_semantics(output_root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_extension_contract",
        "experiment": "Aim 2 complete LOCO MIL five-seed extension",
        "output_root": str(Path(output_root).resolve()),
        "component_root": str(component_root(output_root)),
        "seeds": {
            "adopted": list(ADOPTED_SEEDS),
            "new": list(NEW_SEEDS),
            "complete": list(ALL_SEEDS),
        },
        "recipe": {
            "encoder": "UNI-v1",
            "model": "gated ABMIL",
            "embed_dim": 512,
            "attention_dim": 384,
            "input_dropout": 0.10,
            "dropout": float(paths.DROPOUT),
            "learning_rate": float(paths.LR),
            "weight_decay": float(paths.WEIGHT_DECAY),
            "loss": "unweighted BCE",
            "sampler": "patient_natural",
            "train_tile_cap": CAP,
            "inference_bags": "full",
            "source_cv_folds": N_FOLDS,
            "refit_optimizer_steps": STEP_BUDGET,
            "precision": "bf16-mixed",
        },
        "statistics": {
            "bootstrap_unit": "patient",
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "model_seeds_are_not_inference_units": True,
        },
        "split_verdict": {
            "patient_and_slide_folds_change_across_model_seeds": False,
            "policy": "one frozen predefined five-fold split per arm reused for seeds 42-46",
            "model_seed_changes_only_stochastic_training": True,
        },
        "scope": {
            "controlling_arms": list(CONTROLLING_ARMS),
            "secondary_sensitivity": ["family_rih_sm"],
            "raw_all_conventional_cpht_seed_scope": list(ADOPTED_SEEDS),
            "cpht_a_seed_scope": list(ADOPTED_SEEDS),
        },
        "counts": {
            "arms": len(ARMS),
            "controlling_arms": len(CONTROLLING_ARMS),
            "adopted_fits": len(ARMS) * len(ADOPTED_SEEDS) * (N_FOLDS + 1),
            "new_fits": len(ARMS) * len(NEW_SEEDS) * (N_FOLDS + 1),
            "complete_fits": len(ARMS) * len(ALL_SEEDS) * (N_FOLDS + 1),
            "training_chain_jobs": len(build_training_jobs(output_root)),
            "adopted_score_artifacts": 99,
            "new_score_artifacts": len(build_score_jobs(output_root)),
            "complete_score_artifacts": 165,
        },
    }


def build_contract(
    output_root: Path,
    blind_inputs: dict[tuple[str, str], tuple[Path, pd.DataFrame]],
) -> dict[str, Any]:
    met_contract = _read_json(E2MET_ROOT / "inputs/e2met_contract.json")
    met_outcomes = met_contract.get("outcome_sources_not_opened_by_score")
    if not isinstance(met_outcomes, dict):
        raise ContractError("E2-MET contract lacks sealed outcome sources")
    return {
        **_contract_static_semantics(output_root),
        "created_utc": _utcnow(),
        "arms": {
            name: {
                **{key: str(value) if isinstance(value, Path) else value for key, value in asdict(spec).items()},
                "inputs": _input_identity(spec),
            }
            for name, spec in ARMS.items()
        },
        "label_blind_inputs": {
            f"{kind}/{name}": {
                "source_used_only_to_construct_allowlisted_blind_manifest": _artifact(source),
                "sealed_manifest": _artifact(label_blind_path(output_root, kind, name)),
                "rows": int(len(frame)),
                "contains_target_outcomes": False,
                "outcomes_interpreted_or_used_by_inference": False,
            }
            for (kind, name), (source, frame) in blind_inputs.items()
        },
        "feature_store": _feature_store_contract(),
        "outcome_sources_not_interpreted_or_joined_before_inference_seal": {
            "e2met": met_outcomes,
            "orion_patient_labels": _artifact(CPHT_ROOT / "analysis/orion_patient_scores.parquet"),
        },
        "mixed_seed_scope_references": {
            "all_conventional_cpht_three_seed": _artifact(CPHT_ROOT / "analysis/results.json"),
            "cpht_a_three_seed": _artifact(CPHT_A_ROOT / "analysis/results.json"),
        },
        "adoptions": build_adoption_roster(),
        "implementation_sources": _source_snapshot(),
    }


def cmd_plan(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    training = build_training_jobs(args.output_root)
    scores = build_score_jobs(args.output_root)
    print("Aim 2 LOCO five-seed additive extension")
    print(f"overlay: {root}")
    print(f"arms: {len(ARMS)} total = {len(CONTROLLING_ARMS)} controlling + 1 sensitivity")
    print("seeds: hash-adopt 42-44; train 45-46; frozen identical folds for all five")
    print(
        f"training: {len(training)} schedulable chains / "
        f"{sum(int(job['fit_count']) for job in training)} new fits; default concurrency 6"
    )
    print("fit accounting: 162 adopted + 108 new = 270 complete fits")
    print(f"inference: 99 adopted + {len(scores)} new = 165 score artifacts")
    print("all-conventional CPHT and CPHT-A remain explicitly three-seed analyses")


def cmd_manifest(args: argparse.Namespace) -> None:
    root = assert_safe_output_root(args.output_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Refusing existing/partial Aim-2 overlay: {root}")
    blind = _label_blind_inputs()
    if not args.apply:
        print(
            f"Dry run: validated {len(ARMS)} frozen source/split pairs and prepared "
            f"{len(blind)} label-blind manifests; pass --apply to seal {root}"
        )
        return
    for (kind, name), (_, frame) in blind.items():
        _write_csv_once(label_blind_path(args.output_root, kind, name), frame)
    contract = build_contract(args.output_root, blind)
    _write_json_once(contract_path(args.output_root), contract)
    training = build_training_jobs(args.output_root)
    scoring_jobs = build_score_jobs(args.output_root)
    _write_json_once(
        training_inventory_path(args.output_root),
        {
            "schema_version": 1,
            "default_max_workers": DEFAULT_MAX_WORKERS,
            "maximum_supported_workers": DEFAULT_MAX_WORKERS,
            "fit_count": sum(int(job["fit_count"]) for job in training),
            "jobs": training,
        },
    )
    _write_json_once(
        score_inventory_path(args.output_root),
        {
            "schema_version": 1,
            "contains_target_outcomes": False,
            "jobs": scoring_jobs,
        },
    )
    validate_contract(args.output_root, deep=False, adoptions=True)
    print(f"PASS: sealed additive Aim-2 contract and job inventories at {root}")


def _validate_identity(identity: dict[str, Any], *, deep: bool = True) -> None:
    path = Path(str(identity.get("path", "")))
    if not path.is_file():
        raise ContractError(f"Missing adopted artifact: {path}")
    if int(identity.get("size_bytes", -1)) != int(path.stat().st_size):
        raise ContractError(f"Artifact size changed: {path}")
    if deep and _artifact(path) != identity:
        raise ContractError(f"Artifact content changed: {path}")


def _iter_identities(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "size_bytes", "sha256"}.issubset(value):
            yield value
        else:
            for child in value.values():
                yield from _iter_identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_identities(child)


def validate_contract(
    output_root: Path,
    *,
    deep: bool,
    adoptions: bool = True,
) -> dict[str, Any]:
    assert_safe_output_root(output_root)
    path = contract_path(output_root)
    contract = _read_json(path)
    expected = _contract_static_semantics(output_root)
    mismatch = {
        key: {"expected": wanted, "observed": contract.get(key)}
        for key, wanted in expected.items()
        if contract.get(key) != wanted
    }
    if mismatch:
        raise ContractError(f"Aim-2 extension contract semantics mismatch: {mismatch}")
    dynamic_keys = {
        "created_utc",
        "arms",
        "label_blind_inputs",
        "feature_store",
        "outcome_sources_not_interpreted_or_joined_before_inference_seal",
        "mixed_seed_scope_references",
        "adoptions",
        "implementation_sources",
    }
    if set(contract) != set(expected) | dynamic_keys:
        raise ContractError("Aim-2 extension contract top-level schema drifted")
    try:
        dt.datetime.fromisoformat(str(contract["created_utc"]))
    except (TypeError, ValueError) as exc:
        raise ContractError("Aim-2 contract has an invalid creation timestamp") from exc
    expected_arms = {
        name: {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(spec).items()
            },
            "inputs": _input_identity(spec),
        }
        for name, spec in ARMS.items()
    }
    if contract.get("arms") != expected_arms:
        raise ContractError("Contract arm definitions or frozen input identities drifted")

    blind_sources: dict[tuple[str, str], Path] = {
        **{("primary", arm): spec.primary_target for arm, spec in ARMS.items()},
        **{
            ("e2met", target): E2MET_ROOT / f"inputs/manifests/{target}.csv"
            for target in MET_TARGETS
        },
        ("e2cpht", "orion"): CPHT_ROOT / "inputs/orion_primary.csv",
    }
    expected_blind_keys = {f"{kind}/{name}" for kind, name in blind_sources}
    if set(contract.get("label_blind_inputs", {})) != expected_blind_keys:
        raise ContractError("Contract label-blind input roster drifted")
    for key, entry in contract["label_blind_inputs"].items():
        kind, name = key.split("/", maxsplit=1)
        source_identity = entry.get("source_used_only_to_construct_allowlisted_blind_manifest")
        if not isinstance(source_identity, dict):
            raise ContractError(f"Malformed source-manifest identity: {key}")
        _validate_identity(source_identity, deep=True)
        identity = entry.get("sealed_manifest")
        if not isinstance(identity, dict):
            raise ContractError(f"Malformed label-blind identity: {key}")
        _validate_identity(identity, deep=True)
        frame = pd.read_csv(identity["path"], low_memory=False)
        blind = _blind_frame(frame, context=key)
        if set(map(str.lower, frame.columns)) & FORBIDDEN_OUTCOME_COLUMNS:
            raise ContractError(f"Outcome leak in sealed inference manifest: {key}")
        expected_entry = {
            "source_used_only_to_construct_allowlisted_blind_manifest": _artifact(
                blind_sources[(kind, name)]
            ),
            "sealed_manifest": _artifact(label_blind_path(output_root, kind, name)),
            "rows": int(len(blind)),
            "contains_target_outcomes": False,
            "outcomes_interpreted_or_used_by_inference": False,
        }
        if entry != expected_entry:
            raise ContractError(f"Label-blind contract semantics drifted: {key}")
    expected_sources = _source_snapshot()
    if contract.get("implementation_sources") != expected_sources:
        raise ContractError("Material implementation/config/dependency source ledger drifted")
    for identity in expected_sources:
        _validate_identity(identity, deep=True)
    if adoptions:
        for identity in _iter_identities(contract.get("adoptions", [])):
            # Checkpoints can be large; non-deep preflight still binds path+size.
            is_checkpoint = str(identity.get("path", "")).endswith("model.ckpt")
            _validate_identity(identity, deep=deep or not is_checkpoint)
        if deep:
            current_adoptions = build_adoption_roster()
            if contract.get("adoptions") != current_adoptions:
                raise ContractError("Adopted training/scoring roster or semantics drifted")
            adopted_training = {
                (str(record["arm"]), int(record["seed"])): record
                for record in contract.get("adoptions", [])
                if record.get("kind") == "training"
            }
            for arm in ARM_NAMES:
                for seed in ADOPTED_SEEDS:
                    record = adopted_training.get((arm, seed))
                    if not isinstance(record, dict):
                        raise ContractError(f"Missing adopted training record: {arm}/seed{seed}")
                    observed = _validate_adopted_source_cv(arm, seed)
                    if record.get("source_cv", {}).get("semantic_validation") != observed:
                        raise ContractError(
                            f"{arm}/seed{seed}: adopted source-CV semantic evidence drifted"
                        )
                    observed_refit = _validate_adopted_refit(arm, seed)
                    if record.get("refit_semantic_validation") != observed_refit:
                        raise ContractError(
                            f"{arm}/seed{seed}: adopted refit semantic evidence drifted"
                        )
    feature_store = contract.get("feature_store")
    if not isinstance(feature_store, dict):
        raise ContractError("Contract lacks the governed feature store")
    if feature_store != _feature_store_contract():
        raise ContractError("Governed feature-store contract drifted")
    for key in ("features", "coords", "index", "meta"):
        identity = feature_store.get(key)
        if not isinstance(identity, dict):
            raise ContractError(f"Feature-store identity is missing: {key}")
        _validate_identity(identity, deep=deep)
    met_contract = _read_json(E2MET_ROOT / "inputs/e2met_contract.json")
    met_outcomes = met_contract.get("outcome_sources_not_opened_by_score")
    expected_outcomes = {
        "e2met": met_outcomes,
        "orion_patient_labels": _artifact(
            CPHT_ROOT / "analysis/orion_patient_scores.parquet"
        ),
    }
    if (
        not isinstance(met_outcomes, dict)
        or contract.get(
            "outcome_sources_not_interpreted_or_joined_before_inference_seal"
        )
        != expected_outcomes
    ):
        raise ContractError("Sealed downstream outcome-source ledger drifted")
    for identity in _iter_identities(expected_outcomes):
        _validate_identity(identity, deep=True)
    expected_mixed_scope = {
        "all_conventional_cpht_three_seed": _artifact(CPHT_ROOT / "analysis/results.json"),
        "cpht_a_three_seed": _artifact(CPHT_A_ROOT / "analysis/results.json"),
    }
    if contract.get("mixed_seed_scope_references") != expected_mixed_scope:
        raise ContractError("Mixed three-/five-seed reference ledger drifted")
    inventory = _read_json(training_inventory_path(output_root))
    training_jobs = build_training_jobs(output_root)
    expected_inventory = {
        "schema_version": 1,
        "default_max_workers": DEFAULT_MAX_WORKERS,
        "maximum_supported_workers": DEFAULT_MAX_WORKERS,
        "fit_count": sum(int(job["fit_count"]) for job in training_jobs),
        "jobs": training_jobs,
    }
    if inventory != expected_inventory:
        raise ContractError("Training inventory or exact external worker commands drifted")
    score_inventory = _read_json(score_inventory_path(output_root))
    expected_scores = {
        "schema_version": 1,
        "contains_target_outcomes": False,
        "jobs": build_score_jobs(output_root),
    }
    if score_inventory != expected_scores:
        raise ContractError("Exact 66-job label-blind score inventory drifted")
    return contract


def cmd_preflight(args: argparse.Namespace) -> None:
    contract = validate_contract(args.output_root, deep=args.deep, adoptions=True)
    if args.apply and not args.deep:
        raise ContractError("A persisted preflight receipt requires --deep --apply")
    if args.apply:
        path = preflight_receipt_path(args.output_root)
        if path.exists():
            _validate_preflight_receipt(args.output_root)
        else:
            _write_json_once(
                path,
                {
                    "schema_version": 1,
                    "status": "deep_preflight_passed",
                    "created_utc": _utcnow(),
                    "deep": True,
                    "contract": _artifact(contract_path(args.output_root)),
                    "adopted_training_records": 27,
                    "adopted_score_records": 99,
                    "feature_payloads_content_hashed": True,
                    "implementation": _artifact(Path(__file__).resolve()),
                },
            )
        _validate_preflight_receipt(args.output_root)
    print(
        f"PASS: {len(contract['arms'])} arms, frozen identical folds, "
        f"162 adopted fits, 108 planned fits, deep={args.deep}, persisted={args.apply}"
    )


def _validate_preflight_receipt(output_root: Path) -> dict[str, Any]:
    path = preflight_receipt_path(output_root)
    receipt = _read_json(path)
    expected = {
        "schema_version": 1,
        "status": "deep_preflight_passed",
        "deep": True,
        "contract": _artifact(contract_path(output_root)),
        "adopted_training_records": 27,
        "adopted_score_records": 99,
        "feature_payloads_content_hashed": True,
        "implementation": _artifact(Path(__file__).resolve()),
    }
    mismatch = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatch:
        raise ContractError(f"Persisted deep-preflight receipt mismatch: {mismatch}")
    return receipt


def _run_logged(command: Sequence[str], log_path: Path) -> int:
    lineage.ensure_absent(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen(
            list(command), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        return int(process.wait())


def _study_overrides(output_root: Path, arm: str, seed: int, directory: Path) -> list[str]:
    spec = ARMS[arm]
    return [
        "platform=colon_workstation",
        "data=aim1",
        f"data.aim1_model={spec.model_name}",
        f"data.manifest_stem={spec.source_manifest.stem}",
        f"data.csv_path={spec.source_manifest}",
        f"platform.splits_root={spec.split_root}",
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
        f"train_dir={directory}",
    ]


def _nested(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return "<missing>"
        value = value[key]
    return value


def _material_expectations(arm: str, seed: int, *, source_cv: bool) -> dict[str, Any]:
    spec = ARMS[arm]
    values: dict[str, Any] = {
        "data.name": f"aim1_{spec.model_name}",
        "data.aim1_model": spec.model_name,
        "data.csv_path": str(spec.source_manifest.resolve()),
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
        "platform.precision": "bf16-mixed",
        "platform.devices": 1,
    }
    if source_cv:
        values.update(
            {
                "splits.seed": seed,
                "training.skip_finalize": True,
                "training.refit_max_steps": None,
            }
        )
    else:
        values.update(
            {
                "training.refit_max_steps": STEP_BUDGET,
                "training.max_epochs": _epochs(spec),
            }
        )
    return values


def _validate_material_config(
    config: dict[str, Any],
    arm: str,
    seed: int,
    *,
    source_cv: bool,
    context: str,
    allow_legacy_missing_refit_steps: bool = False,
) -> None:
    mismatch: dict[str, dict[str, Any]] = {}
    for dotted, expected in _material_expectations(arm, seed, source_cv=source_cv).items():
        observed = _nested(config, dotted)
        legacy_omission = (
            allow_legacy_missing_refit_steps
            and source_cv
            and dotted == "training.refit_max_steps"
            and observed == "<missing>"
            and expected is None
        )
        if observed != expected and not legacy_omission:
            mismatch[dotted] = {"expected": expected, "observed": observed}
    if mismatch:
        raise ContractError(f"{arm}/seed{seed}: {context} material recipe mismatch: {mismatch}")


def _validate_source_cv(output_root: Path, arm: str, seed: int) -> dict[str, Any]:
    from oceanpath.workflows.training import validate_training_run_dir

    directory = source_cv_dir(output_root, arm, seed)
    receipt_path = component_root(output_root) / f"receipts/source_cv/{arm}/seed{seed}.json"
    if not directory.is_dir() or not receipt_path.is_file():
        raise FileNotFoundError(f"Incomplete source-CV chain: {arm}/seed{seed}")
    completion = validate_training_run_dir(directory, require_test_predictions=True)
    fold_completions = completion.get("fold_completions")
    expected_fold_paths = [f"fold_{fold}/completion.json" for fold in range(N_FOLDS)]
    observed_fold_paths = (
        [item.get("path") for item in fold_completions]
        if isinstance(fold_completions, list)
        else []
    )
    if (
        int(completion.get("n_folds", -1)) != N_FOLDS
        or completion.get("skip_finalize") is not True
        or observed_fold_paths != expected_fold_paths
    ):
        raise ContractError(f"{arm}/seed{seed}: source-CV is not five folds")
    identity = _read_json(directory / "training_identity.json")
    if identity.get("fingerprint") != completion.get("training_fingerprint"):
        raise ContractError(f"{arm}/seed{seed}: training fingerprint differs from completion")
    payload = identity.get("payload") or {}
    material = payload.get("material_config")
    evidence = payload.get("input_evidence") or {}
    if not isinstance(material, dict):
        raise ContractError(f"{arm}/seed{seed}: missing material training identity")
    _validate_material_config(material, arm, seed, source_cv=True, context="training identity")
    integrity_file = ARMS[arm].split_file.parent / ".integrity_hash"
    expected_evidence = {
        "manifest_sha256": ARMS[arm].source_sha256,
        "split_integrity_sha256": lineage.sha256_file(integrity_file),
    }
    evidence_mismatch = {
        key: {"expected": wanted, "observed": evidence.get(key)}
        for key, wanted in expected_evidence.items()
        if evidence.get(key) != wanted
    }
    if evidence_mismatch:
        raise ContractError(f"{arm}/seed{seed}: input evidence mismatch: {evidence_mismatch}")
    import yaml

    manifest = pd.read_csv(ARMS[arm].source_manifest, low_memory=False)
    for fold in range(N_FOLDS):
        config_path = directory / f"fold_{fold}/config.yaml"
        config = yaml.safe_load(config_path.read_text())
        if not isinstance(config, dict):
            raise ContractError(f"Malformed resolved fold config: {config_path}")
        _validate_material_config(config, arm, seed, source_cv=True, context=f"fold {fold}")
        if Path(str(_nested(config, "platform.splits_root"))).resolve() != ARMS[arm].split_root.resolve():
            raise ContractError(f"{arm}/seed{seed}/fold{fold}: wrong frozen split root")
        for role, mask in (
            ("test", pd.to_numeric(manifest["k_fold"]).astype(int).eq(fold)),
            ("val", pd.to_numeric(manifest[f"val_fold_{fold}"]).astype(int).eq(1)),
        ):
            predictions = pd.read_parquet(directory / f"fold_{fold}/preds_{role}.parquet")
            expected_rows = manifest.loc[mask, ["slide_id", "target_label"]].copy()
            joined = predictions.merge(
                expected_rows, on="slide_id", how="inner", validate="one_to_one"
            )
            if (
                len(predictions) != len(expected_rows)
                or len(joined) != len(expected_rows)
                or set(predictions["slide_id"].astype(str))
                != set(expected_rows["slide_id"].astype(str))
                or not pd.to_numeric(joined["label"]).astype(int).eq(
                    pd.to_numeric(joined["target_label"]).astype(int)
                ).all()
                or not np.isfinite(
                    pd.to_numeric(predictions["logit"], errors="coerce")
                ).all()
            ):
                raise ContractError(
                    f"{arm}/seed{seed}/fold{fold}: exact {role} roster/label drift"
                )
    oof = pd.read_parquet(directory / "oof_predictions.parquet")
    if (
        len(oof) != len(manifest)
        or oof["slide_id"].astype(str).duplicated().any()
        or set(oof["slide_id"].astype(str)) != set(manifest["slide_id"].astype(str))
        or set(pd.to_numeric(oof["fold"], errors="coerce").astype(int)) != set(range(N_FOLDS))
        or not np.isfinite(pd.to_numeric(oof["logit"], errors="coerce")).all()
    ):
        raise ContractError(f"{arm}/seed{seed}: invalid source OOF coverage/logits")
    oof_check = oof.merge(
        manifest[["slide_id", "target_label", "k_fold"]],
        on="slide_id",
        how="inner",
        validate="one_to_one",
    )
    if (
        len(oof_check) != len(manifest)
        or not pd.to_numeric(oof_check["label"]).astype(int).eq(
            pd.to_numeric(oof_check["target_label"]).astype(int)
        ).all()
        or not pd.to_numeric(oof_check["fold"]).astype(int).eq(
            pd.to_numeric(oof_check["k_fold"]).astype(int)
        ).all()
    ):
        raise ContractError(f"{arm}/seed{seed}: OOF label or per-slide fold roster drift")
    receipt = _read_json(receipt_path)
    expected = {
        "status": "completed",
        "arm": arm,
        "seed": seed,
        "oof": _artifact(directory / "oof_predictions.parquet"),
        "completion": _artifact(directory / "training_completion.json"),
        "identity": _artifact(directory / "training_identity.json"),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ContractError(f"{arm}/seed{seed}: source-CV receipt mismatch")
    return receipt


def _validate_adopted_source_cv(arm: str, seed: int) -> dict[str, Any]:
    """Semantically authenticate an adopted source-CV run, not only its hashes."""

    from oceanpath.workflows.training import validate_training_run_dir

    if seed not in ADOPTED_SEEDS:
        raise ContractError(f"Not an adopted model seed: {seed}")
    directory = old_source_cv_dir(arm, seed)
    completion = validate_training_run_dir(directory, require_test_predictions=True)
    fold_completions = completion.get("fold_completions")
    expected_fold_paths = [f"fold_{fold}/completion.json" for fold in range(N_FOLDS)]
    observed_fold_paths = (
        [item.get("path") for item in fold_completions]
        if isinstance(fold_completions, list)
        else []
    )
    if (
        int(completion.get("n_folds", -1)) != N_FOLDS
        or completion.get("skip_finalize") is not True
        or observed_fold_paths != expected_fold_paths
    ):
        raise ContractError(f"{arm}/seed{seed}: adopted source-CV is not five folds")
    identity = _read_json(directory / "training_identity.json")
    if identity.get("fingerprint") != completion.get("training_fingerprint"):
        raise ContractError(
            f"{arm}/seed{seed}: adopted training fingerprint differs from completion"
        )
    payload = identity.get("payload") or {}
    material = payload.get("material_config")
    evidence = payload.get("input_evidence") or {}
    if not isinstance(material, dict):
        raise ContractError(f"{arm}/seed{seed}: adopted material identity is missing")
    _validate_material_config(
        material,
        arm,
        seed,
        source_cv=True,
        context="adopted training identity",
        allow_legacy_missing_refit_steps=True,
    )
    integrity_file = ARMS[arm].split_file.parent / ".integrity_hash"
    expected_evidence = {
        "manifest_sha256": ARMS[arm].source_sha256,
        "split_integrity_sha256": lineage.sha256_file(integrity_file),
    }
    evidence_mismatch = {
        key: {"expected": wanted, "observed": evidence.get(key)}
        for key, wanted in expected_evidence.items()
        if evidence.get(key) != wanted
    }
    if evidence_mismatch:
        raise ContractError(
            f"{arm}/seed{seed}: adopted input evidence mismatch: {evidence_mismatch}"
        )

    import yaml

    manifest = pd.read_csv(ARMS[arm].source_manifest, low_memory=False)
    for fold in range(N_FOLDS):
        config_path = directory / f"fold_{fold}/config.yaml"
        config = yaml.safe_load(config_path.read_text())
        if not isinstance(config, dict):
            raise ContractError(f"Malformed adopted fold config: {config_path}")
        _validate_material_config(
            config,
            arm,
            seed,
            source_cv=True,
            context=f"adopted fold {fold}",
            allow_legacy_missing_refit_steps=True,
        )
        if Path(str(_nested(config, "platform.splits_root"))).resolve() != ARMS[
            arm
        ].split_root.resolve():
            raise ContractError(f"{arm}/seed{seed}/fold{fold}: adopted split root drift")
        for role, mask in (
            ("test", pd.to_numeric(manifest["k_fold"]).astype(int).eq(fold)),
            ("val", pd.to_numeric(manifest[f"val_fold_{fold}"]).astype(int).eq(1)),
        ):
            predictions = pd.read_parquet(directory / f"fold_{fold}/preds_{role}.parquet")
            expected_rows = manifest.loc[mask, ["slide_id", "target_label"]].copy()
            joined = predictions.merge(
                expected_rows, on="slide_id", how="inner", validate="one_to_one"
            )
            if (
                len(predictions) != len(expected_rows)
                or len(joined) != len(expected_rows)
                or set(predictions["slide_id"].astype(str))
                != set(expected_rows["slide_id"].astype(str))
                or not pd.to_numeric(joined["label"]).astype(int).eq(
                    pd.to_numeric(joined["target_label"]).astype(int)
                ).all()
                or not np.isfinite(
                    pd.to_numeric(predictions["logit"], errors="coerce")
                ).all()
            ):
                raise ContractError(
                    f"{arm}/seed{seed}/fold{fold}: adopted exact {role} roster/label drift"
                )
    oof = pd.read_parquet(directory / "oof_predictions.parquet")
    if (
        len(oof) != len(manifest)
        or oof["slide_id"].astype(str).duplicated().any()
        or set(oof["slide_id"].astype(str)) != set(manifest["slide_id"].astype(str))
        or set(pd.to_numeric(oof["fold"], errors="coerce").astype(int))
        != set(range(N_FOLDS))
        or not np.isfinite(pd.to_numeric(oof["logit"], errors="coerce")).all()
    ):
        raise ContractError(f"{arm}/seed{seed}: invalid adopted source OOF coverage/logits")
    oof_check = oof.merge(
        manifest[["slide_id", "target_label", "k_fold"]],
        on="slide_id",
        how="inner",
        validate="one_to_one",
    )
    if (
        len(oof_check) != len(manifest)
        or not pd.to_numeric(oof_check["label"]).astype(int).eq(
            pd.to_numeric(oof_check["target_label"]).astype(int)
        ).all()
        or not pd.to_numeric(oof_check["fold"]).astype(int).eq(
            pd.to_numeric(oof_check["k_fold"]).astype(int)
        ).all()
    ):
        raise ContractError(
            f"{arm}/seed{seed}: adopted OOF label or per-slide fold roster drift"
        )
    return {
        "status": "native_completion_and_exact_frozen_rosters_validated",
        "arm": arm,
        "seed": seed,
        "folds": N_FOLDS,
        "slides": int(len(manifest)),
        "patients": int(manifest["patient_id"].astype(str).nunique()),
        "test_roster_exact_for_every_fold": True,
        "validation_roster_exact_for_every_fold": True,
        "oof_label_and_fold_exact_per_slide": True,
        "source_manifest": _artifact(ARMS[arm].source_manifest),
        "split_file": _artifact(ARMS[arm].split_file),
        "completion": _artifact(directory / "training_completion.json"),
        "identity": _artifact(directory / "training_identity.json"),
        "oof": _artifact(directory / "oof_predictions.parquet"),
    }


def _next_quarantine_dir(output_root: Path, stage: str, arm: str, seed: int) -> Path:
    base = component_root(output_root) / f"failures/{stage}/{arm}/seed{seed}"
    base.mkdir(parents=True, exist_ok=True)
    for ordinal in range(1, 10_000):
        destination = base / f"attempt_{ordinal:04d}"
        if not destination.exists() and not destination.is_symlink():
            destination.mkdir(parents=False, exist_ok=False)
            return destination
    raise ContractError(f"Exhausted quarantine attempt numbers: {stage}/{arm}/seed{seed}")


def _directory_artifacts(directory: Path) -> list[dict[str, Any]]:
    return [
        _artifact(path)
        for path in sorted(directory.rglob("*"), key=lambda item: str(item))
        if path.is_file()
    ]


def _quarantine_source_cv(
    output_root: Path, arm: str, seed: int, *, require_failure_marker: bool
) -> Path:
    directory = source_cv_dir(output_root, arm, seed)
    request = component_root(output_root) / f"requests/source_cv/{arm}/seed{seed}.json"
    log = component_root(output_root) / f"logs/source_cv/{arm}/seed{seed}.log"
    failure = request.with_suffix(".failure.json")
    completion_receipt = (
        component_root(output_root) / f"receipts/source_cv/{arm}/seed{seed}.json"
    )
    if require_failure_marker and not failure.is_file():
        raise ContractError(f"Source-CV retry lacks a native failure marker: {failure}")
    existing = [
        path
        for path in (directory, request, log, failure, completion_receipt)
        if path.exists() or path.is_symlink()
    ]
    if not existing:
        raise ContractError(f"No partial source-CV artifacts to quarantine: {arm}/seed{seed}")
    destination = _next_quarantine_dir(output_root, "source_cv", arm, seed)
    moved: list[Path] = []
    for name, path in (
        ("partial_output", directory),
        ("run_request.json", request),
        ("stdout_stderr.log", log),
        ("failure.json", failure),
        ("invalid_completion_receipt.json", completion_receipt),
    ):
        if path.exists() or path.is_symlink():
            target = destination / name
            path.rename(target)
            moved.append(target)
    _write_json_once(
        destination / "quarantine_receipt.json",
        {
            "schema_version": 1,
            "status": "failed_attempt_quarantined_without_overwrite",
            "created_utc": _utcnow(),
            "stage": "source_cv",
            "arm": arm,
            "seed": seed,
            "failure_marker_required": require_failure_marker,
            "artifacts": [_artifact(path) for path in moved if path.is_file()],
            "partial_output_retained": str(destination / "partial_output"),
            "partial_output_artifacts": (
                _directory_artifacts(destination / "partial_output")
                if (destination / "partial_output").is_dir()
                else []
            ),
        },
    )
    return destination


def _quarantine_refit(
    output_root: Path, arm: str, seed: int, *, require_failure_marker: bool
) -> Path:
    directory = refit_dir(output_root, arm, seed)
    failure = directory / "failure.json"
    if require_failure_marker and not failure.is_file():
        raise ContractError(f"Refit retry lacks a native failure marker: {failure}")
    if not directory.exists():
        raise ContractError(f"No partial refit directory to quarantine: {arm}/seed{seed}")
    destination = _next_quarantine_dir(output_root, "refit", arm, seed)
    retained = destination / "partial_output"
    directory.rename(retained)
    _write_json_once(
        destination / "quarantine_receipt.json",
        {
            "schema_version": 1,
            "status": "failed_attempt_quarantined_without_overwrite",
            "created_utc": _utcnow(),
            "stage": "refit",
            "arm": arm,
            "seed": seed,
            "failure_marker_required": require_failure_marker,
            "failure": _artifact(retained / "failure.json") if (retained / "failure.json").is_file() else None,
            "partial_output_retained": str(retained),
            "partial_output_artifacts": _directory_artifacts(retained),
        },
    )
    return destination


def cmd_internal_source_cv(args: argparse.Namespace) -> None:
    arm, seed = args.arm, args.seed
    if seed not in NEW_SEEDS:
        raise ContractError("The extension may train only seeds 45 and 46")
    validate_contract(args.output_root, deep=False, adoptions=False)
    _validate_preflight_receipt(args.output_root)
    directory = source_cv_dir(args.output_root, arm, seed)
    if directory.exists():
        try:
            _validate_source_cv(args.output_root, arm, seed)
        except Exception as exc:
            failure = (
                component_root(args.output_root)
                / f"requests/source_cv/{arm}/seed{seed}.failure.json"
            )
            if not failure.is_file():
                raise ContractError(
                    f"Refusing unmarked partial immutable source-CV directory: {directory}"
                ) from exc
            quarantined = _quarantine_source_cv(
                args.output_root, arm, seed, require_failure_marker=True
            )
            print(f"quarantined failed source-CV attempt: {quarantined}")
        else:
            print(f"PASS cached source-CV: {arm}/seed{seed}")
            return
    request = component_root(args.output_root) / f"requests/source_cv/{arm}/seed{seed}.json"
    log = component_root(args.output_root) / f"logs/source_cv/{arm}/seed{seed}.log"
    if request.exists() or log.exists():
        failure = request.with_suffix(".failure.json")
        if failure.is_file():
            quarantined = _quarantine_source_cv(
                args.output_root, arm, seed, require_failure_marker=True
            )
            print(f"quarantined failed source-CV attempt: {quarantined}")
        else:
            raise ContractError(f"Refusing unmarked partial source-CV request/log: {arm}/seed{seed}")
    _write_json_once(
        request,
        {
            "schema_version": 1,
            "status": "requested",
            "created_utc": _utcnow(),
            "arm": arm,
            "seed": seed,
            "fits": N_FOLDS,
            "source_manifest": _artifact(ARMS[arm].source_manifest),
            "split_file": _artifact(ARMS[arm].split_file),
            "contract": _artifact(contract_path(args.output_root)),
            "output": str(directory),
        },
    )
    overrides = _study_overrides(args.output_root, arm, seed, directory) + [
        "training.skip_finalize=true",
        f"exp_name=aim2_5seed_srccv_{arm}_c{CAP}_s{seed}",
        f"hydra.run.dir={component_root(args.output_root) / 'hydra_runs' / f'source_cv_{arm}_seed{seed}'}",
        "hydra.job.chdir=false",
    ]
    command = [sys.executable, str(REPO / "tools/study_train.py"), "hydra-train", *overrides]
    returncode = _run_logged(command, log)
    if returncode:
        _write_json_once(
            request.with_suffix(".failure.json"),
            {"status": "failed", "returncode": returncode, "request": _artifact(request), "log": _artifact(log)},
        )
        raise SystemExit(returncode)
    receipt = component_root(args.output_root) / f"receipts/source_cv/{arm}/seed{seed}.json"
    _write_json_once(
        receipt,
        {
            "schema_version": 1,
            "status": "completed",
            "finished_utc": _utcnow(),
            "arm": arm,
            "seed": seed,
            "request": _artifact(request),
            "log": _artifact(log),
            "oof": _artifact(directory / "oof_predictions.parquet"),
            "completion": _artifact(directory / "training_completion.json"),
            "identity": _artifact(directory / "training_identity.json"),
            "cv_summary": _artifact(directory / "cv_summary.json"),
        },
    )
    _validate_source_cv(args.output_root, arm, seed)


def _epochs(spec: ArmSpec) -> int:
    return max(MIN_EPOCHS, math.ceil(STEP_BUDGET / spec.expected_source_patients))


def _validate_adopted_refit(arm: str, seed: int) -> dict[str, Any]:
    """Replay the fixed-step recipe and completion semantics of an old refit."""

    if seed not in ADOPTED_SEEDS:
        raise ContractError(f"Not an adopted model seed: {seed}")
    spec = ARMS[arm]
    directory = old_refit_dir(arm, seed)
    summary_path = directory / "fit_summary.json"
    config_path = directory / "resolved_config.yaml"
    checkpoint = directory / "final/refit/model.ckpt"
    if not summary_path.is_file() or not config_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Incomplete adopted refit: {arm}/seed{seed}")
    summary = _read_json(summary_path)
    result = summary.get("result") or {}
    expected_summary = {
        "status": "completed",
        "seed": seed,
        "sampling_seed": seed,
        "cap": CAP,
        "epoch_ceiling": _epochs(spec),
        "optimizer_step_budget": STEP_BUDGET,
        "sampler": "patient_natural",
        "loss_weighting": "none",
        "model": _artifact(checkpoint),
        "resolved_config": _artifact(config_path),
    }
    summary_mismatch = {
        key: {"expected": wanted, "observed": summary.get(key)}
        for key, wanted in expected_summary.items()
        if summary.get(key) != wanted
    }
    if summary_mismatch:
        raise ContractError(
            f"{arm}/seed{seed}: adopted refit summary mismatch: {summary_mismatch}"
        )

    import yaml

    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ContractError(f"{arm}/seed{seed}: malformed adopted refit config")
    _validate_material_config(
        config, arm, seed, source_cv=False, context="adopted refit"
    )
    if Path(str(_nested(config, "platform.splits_root"))).resolve() != spec.split_root.resolve():
        raise ContractError(f"{arm}/seed{seed}: adopted refit split root drift")
    expected_result = {
        "strategy": "refit",
        "refit_epochs": _epochs(spec),
        "refit_epoch_rule": "median",
        "trainer_max_epochs": _epochs(spec),
        "refit_max_steps": STEP_BUDGET,
        "actual_optimizer_steps": STEP_BUDGET,
        "batch_size": 1,
        "accumulate_grad_batches": 1,
        "seed": seed,
        "sampling_seed": seed,
        "train_sampling_strategy": "patient_natural",
        "sample_weight_column": None,
        "class_weights": None,
        "dataset_max_instances": CAP,
        "max_instances": None,
        "eval_full_bags": True,
        "n_train_slides": spec.expected_source_slides,
        "lr_scheduler_total_steps": STEP_BUDGET,
        "model_path": str(checkpoint.resolve()),
    }
    result_mismatch = {
        key: {"expected": wanted, "observed": result.get(key)}
        for key, wanted in expected_result.items()
        if result.get(key) != wanted
    }
    sampling = result.get("training_sampling") or {}
    expected_sampling = {
        "strategy": "patient_natural",
        "seed": seed,
        "n_training_slides": spec.expected_source_slides,
        "n_training_patients": spec.expected_source_patients,
        "target_positive_prevalence": None,
    }
    sampling_mismatch = {
        key: {"expected": wanted, "observed": sampling.get(key)}
        for key, wanted in expected_sampling.items()
        if sampling.get(key) != wanted
    }
    if result_mismatch or sampling_mismatch:
        raise ContractError(
            f"{arm}/seed{seed}: adopted refit result mismatch: "
            f"result={result_mismatch}, sampling={sampling_mismatch}"
        )
    return {
        "status": "fixed_step_refit_recipe_validated",
        "arm": arm,
        "seed": seed,
        "optimizer_steps": STEP_BUDGET,
        "sampler": "patient_natural",
        "loss_weighting": "none",
        "summary": _artifact(summary_path),
        "resolved_config": _artifact(config_path),
        "checkpoint": _artifact(checkpoint),
    }


def _validate_refit(output_root: Path, arm: str, seed: int) -> dict[str, Any]:
    directory = refit_dir(output_root, arm, seed)
    summary_path = directory / "fit_summary.json"
    checkpoint = directory / "final/refit/model.ckpt"
    if not summary_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Incomplete refit: {arm}/seed{seed}")
    summary = _read_json(summary_path)
    result = summary.get("result") or {}
    expected = {
        "status": "completed", "arm": arm, "seed": seed, "sampling_seed": seed,
        "cap": CAP, "optimizer_step_budget": STEP_BUDGET,
        "sampler": "patient_natural", "loss_weighting": "none",
        "model": _artifact(checkpoint),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ContractError(f"{arm}/seed{seed}: refit summary mismatch")
    if int(result.get("actual_optimizer_steps", -1)) != STEP_BUDGET:
        raise ContractError(f"{arm}/seed{seed}: refit did not execute exactly {STEP_BUDGET} steps")
    import yaml

    config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    if not isinstance(config, dict):
        raise ContractError(f"{arm}/seed{seed}: malformed refit config")
    _validate_material_config(config, arm, seed, source_cv=False, context="refit")
    expected_result = {
        "refit_max_steps": STEP_BUDGET,
        "batch_size": 1,
        "accumulate_grad_batches": 1,
        "seed": seed,
        "sampling_seed": seed,
        "train_sampling_strategy": "patient_natural",
        "sample_weight_column": None,
        "class_weights": None,
        "dataset_max_instances": CAP,
        "max_instances": None,
        "eval_full_bags": True,
    }
    mismatch = {
        key: {"expected": wanted, "observed": result.get(key)}
        for key, wanted in expected_result.items()
        if result.get(key) != wanted
    }
    if mismatch:
        raise ContractError(f"{arm}/seed{seed}: refit result recipe mismatch: {mismatch}")
    return summary


def cmd_internal_refit(args: argparse.Namespace) -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from oceanpath.workflows.finalize import _run_refit

    arm, seed = args.arm, args.seed
    if seed not in NEW_SEEDS:
        raise ContractError("The extension may train only seeds 45 and 46")
    validate_contract(args.output_root, deep=False, adoptions=False)
    _validate_preflight_receipt(args.output_root)
    directory = refit_dir(args.output_root, arm, seed)
    if directory.exists():
        try:
            _validate_refit(args.output_root, arm, seed)
        except Exception as exc:
            if not (directory / "failure.json").is_file():
                raise ContractError(
                    f"Refusing unmarked partial immutable refit directory: {directory}"
                ) from exc
            quarantined = _quarantine_refit(
                args.output_root, arm, seed, require_failure_marker=True
            )
            print(f"quarantined failed refit attempt: {quarantined}")
        else:
            print(f"PASS cached refit: {arm}/seed{seed}")
            return
    directory.mkdir(parents=True, exist_ok=False)
    spec = ARMS[arm]
    epochs = _epochs(spec)
    request = directory / "run_request.json"
    _write_json_once(
        request,
        {
            "schema_version": 1,
            "status": "requested",
            "created_utc": _utcnow(),
            "arm": arm,
            "seed": seed,
            "sampling_seed": seed,
            "cap": CAP,
            "optimizer_step_budget": STEP_BUDGET,
            "epoch_ceiling": epochs,
            "source_manifest": _artifact(spec.source_manifest),
            "split_file": _artifact(spec.split_file),
            "contract": _artifact(contract_path(args.output_root)),
        },
    )
    overrides = _study_overrides(args.output_root, arm, seed, directory) + [
        "training.refit_epoch_rule=median",
        f"exp_name=aim2_5seed_refit_{arm}_c{CAP}_s{seed}",
    ]
    with initialize_config_dir(config_dir=str((REPO / "configs").resolve()), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    with open_dict(cfg):
        cfg.training.max_epochs = epochs
        cfg.training.refit_max_steps = STEP_BUDGET
    config_path = directory / "resolved_config.yaml"
    lineage.write_text_once(config_path, OmegaConf.to_yaml(cfg, resolve=True))
    final_dir = directory / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = _run_refit(cfg, final_dir, [{"best_epoch": epochs}])
    except Exception as exc:
        _write_json_once(
            directory / "failure.json",
            {"status": "failed", "finished_utc": _utcnow(), "error": repr(exc), "request": _artifact(request)},
        )
        raise
    checkpoint = final_dir / "refit/model.ckpt"
    _write_json_once(
        directory / "fit_summary.json",
        {
            "schema_version": 1,
            "status": "completed",
            "finished_utc": _utcnow(),
            "arm": arm,
            "seed": seed,
            "sampling_seed": seed,
            "cap": CAP,
            "epoch_ceiling": epochs,
            "optimizer_step_budget": STEP_BUDGET,
            "sampler": "patient_natural",
            "loss_weighting": "none",
            "model": _artifact(checkpoint),
            "resolved_config": _artifact(config_path),
            "request": _artifact(request),
            "result": result,
        },
    )
    _validate_refit(args.output_root, arm, seed)


def _execute_job(job: dict[str, Any]) -> tuple[str, int]:
    process = subprocess.run(job["external_worker_command"], cwd=REPO, check=False)
    return str(job["job_id"]), int(process.returncode)


@contextlib.contextmanager
def _launcher_lock(output_root: Path) -> Iterable[None]:
    """Prevent two six-way Aim-2 launchers from dispatching the same roster."""

    path = component_root(output_root) / "execution/train_launcher.lock"
    with _kernel_file_lock(path, context="Aim-2 launcher"):
        yield


def _scheduler_receipt_path(output_root: Path, job_key: str) -> Path:
    return component_root(output_root) / "receipts/external_scheduler" / f"{job_key}.json"


def _validate_finished_training_job(output_root: Path, job: dict[str, Any]) -> dict[str, Any]:
    if job["stage"] == "source_cv":
        return _validate_source_cv(output_root, str(job["arm"]), int(job["seed"]))
    summary = _validate_refit(output_root, str(job["arm"]), int(job["seed"]))
    return {
        "summary": _artifact(refit_dir(output_root, job["arm"], job["seed"]) / "fit_summary.json"),
        "checkpoint": summary["model"],
    }


@contextlib.contextmanager
def _job_lock(output_root: Path, job_key: str) -> Iterable[None]:
    path = _job_lock_path(output_root, job_key)
    with _kernel_file_lock(path, context=f"training job {job_key}"):
        yield


@contextlib.contextmanager
def _kernel_file_lock(path: Path, *, context: str) -> Iterable[None]:
    """Kernel-owned flock: process death releases it even if the file remains."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f"Another process owns {context}: {path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            (
                json.dumps(
                    {"pid": os.getpid(), "context": context, "created_utc": _utcnow()}
                )
                + "\n"
            ).encode(),
        )
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _job_lock_path(output_root: Path, job_key: str) -> Path:
    return component_root(output_root) / "execution/job_locks" / f"{job_key}.lock"


def _job_lock_is_active(output_root: Path, job_key: str) -> bool:
    path = _job_lock_path(output_root, job_key)
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


def _validate_scheduler_receipt(
    output_root: Path, job: dict[str, Any]
) -> dict[str, Any] | None:
    path = _scheduler_receipt_path(output_root, str(job["job_id"]))
    if not path.exists():
        return None
    receipt = _read_json(path)
    artifacts = _validate_finished_training_job(output_root, job)
    expected = {
        "schema_version": 1,
        "status": "completed_rc0",
        "job_key": job["job_id"],
        "returncode": 0,
        "contract": _artifact(contract_path(output_root)),
        "artifacts": artifacts,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ContractError(f"External-scheduler receipt mismatch: {path}")
    return receipt


def _recover_orphaned_job(output_root: Path, job: dict[str, Any]) -> Path | None:
    """Quarantine an inactive partial attempt while holding its kernel job lock."""

    try:
        _validate_finished_training_job(output_root, job)
    except Exception:
        pass
    else:
        # Native completion can survive a kill between the worker and its
        # scheduler receipt.  cmd_train_one will authenticate and receipt it.
        return None
    arm, seed = str(job["arm"]), int(job["seed"])
    if job["stage"] == "source_cv":
        directory = source_cv_dir(output_root, arm, seed)
        request = component_root(output_root) / f"requests/source_cv/{arm}/seed{seed}.json"
        log = component_root(output_root) / f"logs/source_cv/{arm}/seed{seed}.log"
        failure = request.with_suffix(".failure.json")
        receipt = component_root(output_root) / f"receipts/source_cv/{arm}/seed{seed}.json"
        if not any(
            path.exists() or path.is_symlink()
            for path in (directory, request, log, failure, receipt)
        ):
            return None
        return _quarantine_source_cv(
            output_root, arm, seed, require_failure_marker=False
        )
    directory = refit_dir(output_root, arm, seed)
    if not (directory.exists() or directory.is_symlink()):
        return None
    return _quarantine_refit(output_root, arm, seed, require_failure_marker=False)


def cmd_train_one(args: argparse.Namespace) -> None:
    """Execute exactly one inventory job under the study-wide max-six scheduler."""

    if not args.external_scheduler:
        raise ContractError("train-one is reserved for the study-wide external scheduler")
    if not args.apply:
        raise ContractError("train-one requires --apply")
    validate_contract(args.output_root, deep=False, adoptions=False)
    _validate_preflight_receipt(args.output_root)
    jobs = {str(job["job_id"]): job for job in build_training_jobs(args.output_root)}
    if args.job_key not in jobs:
        raise ContractError(f"Unknown governed Aim-2 job key: {args.job_key}")
    job = jobs[args.job_key]
    cached = _validate_scheduler_receipt(args.output_root, job)
    if cached is not None:
        print(f"PASS cached external-scheduler job: {args.job_key}")
        return
    with _job_lock(args.output_root, args.job_key):
        # Re-check after lock acquisition in case another worker just finished.
        cached = _validate_scheduler_receipt(args.output_root, job)
        if cached is not None:
            print(f"PASS cached external-scheduler job: {args.job_key}")
            return
        if args.recover_orphan:
            recovered = _recover_orphaned_job(args.output_root, job)
            if recovered is not None:
                print(f"quarantined inactive orphan before retry: {recovered}")
        result = subprocess.run(job["command"], cwd=REPO, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"{args.job_key} failed rc={result.returncode}; immutable job evidence retained"
            )
        artifacts = _validate_finished_training_job(args.output_root, job)
        _write_json_once(
            _scheduler_receipt_path(args.output_root, args.job_key),
            {
                "schema_version": 1,
                "status": "completed_rc0",
                "finished_utc": _utcnow(),
                "external_scheduler": True,
                "scheduler_slots": 1,
                "job_key": args.job_key,
                "returncode": 0,
                "contract": _artifact(contract_path(args.output_root)),
                "artifacts": artifacts,
            },
        )
    _validate_scheduler_receipt(args.output_root, job)
    print(f"PASS external-scheduler job rc0 + native receipts: {args.job_key}")


def cmd_quarantine_failed(args: argparse.Namespace) -> None:
    """Governed recovery for an orphaned partial attempt with no native marker."""

    validate_contract(args.output_root, deep=False, adoptions=False)
    _validate_preflight_receipt(args.output_root)
    jobs = {str(job["job_id"]): job for job in build_training_jobs(args.output_root)}
    if args.job_key not in jobs:
        raise ContractError(f"Unknown governed Aim-2 job key: {args.job_key}")
    job = jobs[args.job_key]
    if _scheduler_receipt_path(args.output_root, args.job_key).exists():
        _validate_scheduler_receipt(args.output_root, job)
        raise ContractError(f"Refusing to quarantine completed job: {args.job_key}")
    if _job_lock_is_active(args.output_root, args.job_key):
        raise ContractError(f"Refusing to quarantine a job with an active lock: {args.job_key}")
    try:
        _validate_finished_training_job(args.output_root, job)
    except Exception:
        pass
    else:
        raise ContractError(f"Refusing to quarantine valid completed artifacts: {args.job_key}")
    if not args.apply:
        print(f"DRY RUN: would quarantine incomplete governed attempt {args.job_key}")
        return
    if job["stage"] == "source_cv":
        destination = _quarantine_source_cv(
            args.output_root, str(job["arm"]), int(job["seed"]), require_failure_marker=False
        )
    else:
        destination = _quarantine_refit(
            args.output_root, str(job["arm"]), int(job["seed"]), require_failure_marker=False
        )
    print(
        f"PASS: quarantined immutable failed/orphan attempt at {destination}; "
        f"rerun {args.job_key} to create a new attempt"
    )


def cmd_train(args: argparse.Namespace) -> None:
    validate_contract(args.output_root, deep=False, adoptions=False)
    if not 1 <= args.max_workers <= DEFAULT_MAX_WORKERS:
        raise ValueError(f"--max-workers must be 1..{DEFAULT_MAX_WORKERS}")
    jobs = build_training_jobs(args.output_root)
    if args.arm:
        jobs = [job for job in jobs if job["arm"] == args.arm]
    if args.seed:
        jobs = [job for job in jobs if int(job["seed"]) == args.seed]
    if args.stage:
        jobs = [job for job in jobs if job["stage"] == args.stage]
    if args.dry_run or not args.apply:
        print(
            f"DRY RUN: {len(jobs)} chains / {sum(int(job['fit_count']) for job in jobs)} fits; "
            f"max_workers={args.max_workers}"
        )
        for job in jobs:
            print(json.dumps(job, sort_keys=True))
        if not args.dry_run and not args.apply:
            print("No jobs launched; pass --apply explicitly.")
        return
    _validate_preflight_receipt(args.output_root)
    failures: list[tuple[str, int]] = []
    with (
        _launcher_lock(args.output_root),
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool,
    ):
        futures = {pool.submit(_execute_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job_id, returncode = future.result()
            print(f"{job_id}: returncode={returncode}", flush=True)
            if returncode:
                failures.append((job_id, returncode))
    if failures:
        raise SystemExit(f"Training failures (immutable evidence retained): {failures}")


def _inference_environment() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ContractError("Governed Aim-2 inference requires CUDA with bfloat16 support")
    return {
        "device": "cuda",
        "device_index": 0,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "autocast": True,
        "autocast_dtype": "bfloat16",
        "evaluation_bag": "full",
        "batch_size": 1,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "lightning_version": importlib.metadata.version("lightning"),
        "python_version": platform.python_version(),
        "implementation": [
            _artifact(REPO / "aim2_cross_protocol_transfer.py"),
            _artifact(REPO / "src/oceanpath/datasets/datamodule.py"),
            _artifact(REPO / "src/oceanpath/datasets/packed.py"),
            _artifact(REPO / "src/oceanpath/training/lightning.py"),
            _artifact(REPO / "src/oceanpath/models/abmil.py"),
        ],
    }


def _seal_inference_environment(output_root: Path, *, num_workers: int) -> dict[str, Any]:
    environment = _inference_environment()
    environment["num_workers"] = int(num_workers)
    path = component_root(output_root) / "inference/environment.json"
    if path.exists():
        recorded = _read_json(path)
        if recorded != environment:
            raise ContractError("Inference environment changed after it was sealed")
    else:
        _write_json_once(path, environment)
    return environment


def _blind_manifest_for_score(output_root: Path, arm: str, target: str) -> Path:
    if target == "primary":
        return label_blind_path(output_root, "primary", arm)
    if target in MET_TARGETS:
        return label_blind_path(output_root, "e2met", target)
    if target == "orion":
        return label_blind_path(output_root, "e2cpht", "orion")
    raise ValueError(target)


def _validate_score_frame(
    scores: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    seed: int,
    context: str,
) -> None:
    if list(scores.columns) != ["slide_id", "seed", "fold", "logit"]:
        raise ContractError(f"{context}: wrong native-logit schema: {list(scores.columns)}")
    if set(scores["seed"].astype(int)) != {seed} or set(scores["fold"].astype(int)) != {0}:
        raise ContractError(f"{context}: wrong seed/fold identity")
    if scores.duplicated(["slide_id", "seed", "fold"]).any():
        raise ContractError(f"{context}: duplicated slide/model score")
    expected = set(manifest["slide_id"].astype(str))
    if len(scores) != len(manifest) or set(scores["slide_id"].astype(str)) != expected:
        raise ContractError(f"{context}: incomplete slide roster")
    if not np.isfinite(pd.to_numeric(scores["logit"], errors="coerce")).all():
        raise ContractError(f"{context}: non-finite native logits")


def _validate_new_score(
    output_root: Path,
    arm: str,
    target: str,
    seed: int,
) -> pd.DataFrame | None:
    path = score_path(output_root, arm, target, seed)
    receipt_path = _receipt(path)
    if not path.exists() and not receipt_path.exists():
        return None
    if not path.is_file() or not receipt_path.is_file():
        raise ContractError(f"Partial immutable score cache: {path}")
    manifest_path = _blind_manifest_for_score(output_root, arm, target)
    checkpoint = checkpoint_path(output_root, arm, seed)
    receipt = _read_json(receipt_path)
    expected = {
        "schema_version": 1,
        "contains_target_outcomes": False,
        "arm": arm,
        "target": target,
        "seed": seed,
        "manifest": _artifact(manifest_path),
        "checkpoint": _artifact(checkpoint),
        "contract": _artifact(contract_path(output_root)),
        "environment": _artifact(component_root(output_root) / "inference/environment.json"),
        "artifact": _artifact(path),
    }
    mismatch = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatch:
        raise ContractError(f"Invalid score receipt {receipt_path}: {mismatch}")
    frame = pd.read_parquet(path)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    _validate_score_frame(frame, manifest, seed=seed, context=f"{arm}/{target}/seed{seed}")
    return frame


def _score_one(
    output_root: Path,
    arm: str,
    target: str,
    seed: int,
    *,
    num_workers: int,
) -> None:
    if seed not in NEW_SEEDS:
        raise ContractError("The extension may infer only new seeds 45 and 46")
    cached = _validate_new_score(output_root, arm, target, seed)
    if cached is not None:
        print(f"PASS cached score: {arm}/{target}/seed{seed}")
        return
    from aim2_cross_protocol_transfer import _score_checkpoints

    manifest_path = _blind_manifest_for_score(output_root, arm, target)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    _blind_frame(manifest, context=f"{arm}/{target}")
    checkpoint = checkpoint_path(output_root, arm, seed)
    _validate_refit(output_root, arm, seed)
    contract = validate_contract(output_root, deep=False, adoptions=False)
    store = contract["feature_store"]
    scores = _score_checkpoints(
        [(seed, 0, checkpoint)],
        manifest,
        feature_dir=Path(store["source_dir"]),
        pack_dir=Path(store["path"]),
        device="cuda",
        num_workers=num_workers,
    )
    _validate_score_frame(scores, manifest, seed=seed, context=f"{arm}/{target}/seed{seed}")
    destination = score_path(output_root, arm, target, seed)
    lineage.write_parquet_once(destination, scores)
    _write_json_once(
        _receipt(destination),
        {
            "schema_version": 1,
            "created_utc": _utcnow(),
            "contains_target_outcomes": False,
            "arm": arm,
            "target": target,
            "seed": seed,
            "manifest": _artifact(manifest_path),
            "checkpoint": _artifact(checkpoint),
            "contract": _artifact(contract_path(output_root)),
            "environment": _artifact(component_root(output_root) / "inference/environment.json"),
            "artifact": _artifact(destination),
            "n_rows": int(len(scores)),
        },
    )
    _validate_new_score(output_root, arm, target, seed)


def cmd_internal_score(args: argparse.Namespace) -> None:
    validate_contract(args.output_root, deep=False, adoptions=False)
    _seal_inference_environment(args.output_root, num_workers=args.num_workers)
    _score_one(
        args.output_root,
        args.arm,
        args.target,
        args.seed,
        num_workers=args.num_workers,
    )


def cmd_score(args: argparse.Namespace) -> None:
    validate_contract(args.output_root, deep=False, adoptions=False)
    jobs = build_score_jobs(args.output_root)
    if args.arm:
        jobs = [job for job in jobs if job["arm"] == args.arm]
    if args.seed:
        jobs = [job for job in jobs if int(job["seed"]) == args.seed]
    if args.target:
        jobs = [job for job in jobs if job["target"] == args.target]
    if not args.apply:
        print(f"DRY RUN: {len(jobs)} label-blind score jobs")
        for job in jobs:
            print(json.dumps(job, sort_keys=True))
        print("No inference launched; pass --apply explicitly.")
        return
    # No target outcome has been opened.  Validate every requested checkpoint
    # before publishing the first prediction in this invocation.
    for job in jobs:
        _validate_refit(args.output_root, str(job["arm"]), int(job["seed"]))
    _seal_inference_environment(args.output_root, num_workers=args.num_workers)
    for job in jobs:
        _score_one(
            args.output_root,
            str(job["arm"]),
            str(job["target"]),
            int(job["seed"]),
            num_workers=args.num_workers,
        )


def _adopted_score_records(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [record for record in contract["adoptions"] if record.get("kind") == "score"]


def verify_inference_seal(output_root: Path, *, deep: bool = False) -> dict[str, Any]:
    contract = validate_contract(output_root, deep=deep, adoptions=True)
    seal = _read_json(inference_seal_path(output_root))
    adopted = _adopted_score_records(contract)
    jobs = build_score_jobs(output_root)
    expected = {
        "schema_version": 1,
        "status": "sealed_before_outcome_join",
        "target_outcomes_present": False,
        "contract": _artifact(contract_path(output_root)),
        "adopted_score_count": 99,
        "new_score_count": 66,
        "complete_score_count": 165,
        "five_seed_loco_complete": True,
        "all_conventional_cpht_scope": list(ADOPTED_SEEDS),
        "cpht_a_scope": list(ADOPTED_SEEDS),
    }
    mismatch = {
        key: {"expected": value, "observed": seal.get(key)}
        for key, value in expected.items()
        if seal.get(key) != value
    }
    if mismatch or len(adopted) != 99:
        raise ContractError(f"Inference seal header/old roster mismatch: {mismatch}")
    records = seal.get("new_scores") or []
    if len(records) != len(jobs):
        raise ContractError("Inference seal has an incomplete new-score roster")
    for record, job in zip(records, jobs, strict=True):
        path = score_path(output_root, job["arm"], job["target"], job["seed"])
        wanted = {"score": _artifact(path), "receipt": _artifact(_receipt(path))}
        if record != wanted:
            raise ContractError(f"Score changed after inference seal: {path}")
        _validate_new_score(output_root, job["arm"], job["target"], job["seed"])
    return seal


def cmd_seal_inference(args: argparse.Namespace) -> None:
    contract = validate_contract(args.output_root, deep=False, adoptions=True)
    path = inference_seal_path(args.output_root)
    if path.exists():
        verify_inference_seal(args.output_root, deep=args.deep)
        print(f"PASS: existing inference seal is valid: {path}")
        return
    jobs = build_score_jobs(args.output_root)
    records: list[dict[str, Any]] = []
    for job in jobs:
        frame = _validate_new_score(
            args.output_root, job["arm"], job["target"], job["seed"]
        )
        if frame is None:
            raise ContractError(
                f"Cannot seal incomplete inference: {job['arm']}/{job['target']}/seed{job['seed']}"
            )
        score = score_path(args.output_root, job["arm"], job["target"], job["seed"])
        records.append({"score": _artifact(score), "receipt": _artifact(_receipt(score))})
    adopted = _adopted_score_records(contract)
    if len(adopted) != 99:
        raise ContractError(f"Expected 99 adopted scores, found {len(adopted)}")
    _write_json_once(
        path,
        {
            "schema_version": 1,
            "created_utc": _utcnow(),
            "status": "sealed_before_outcome_join",
            "target_outcomes_present": False,
            "contract": _artifact(contract_path(args.output_root)),
            "adopted_score_count": len(adopted),
            "new_score_count": len(records),
            "complete_score_count": len(adopted) + len(records),
            "five_seed_loco_complete": True,
            "all_conventional_cpht_scope": list(ADOPTED_SEEDS),
            "cpht_a_scope": list(ADOPTED_SEEDS),
            "new_scores": records,
        },
    )
    verify_inference_seal(args.output_root, deep=args.deep)
    print(f"PASS: inference sealed before any outcome join: {path}")


def _source_oof_path(output_root: Path, arm: str, seed: int) -> Path:
    if seed in ADOPTED_SEEDS:
        return old_source_cv_dir(arm, seed) / "oof_predictions.parquet"
    return source_cv_dir(output_root, arm, seed) / "oof_predictions.parquet"


def _resolved_score_path(output_root: Path, arm: str, target: str, seed: int) -> Path:
    if seed in NEW_SEEDS:
        return score_path(output_root, arm, target, seed)
    if target == "primary":
        return old_primary_score_path(arm, seed)
    if target in MET_TARGETS:
        return old_met_score_path(arm, target, seed)
    if target == "orion":
        return old_orion_score_path(arm, seed)
    raise ValueError(target)


def _five_seed_source_oof(output_root: Path, arm: str) -> pd.DataFrame:
    manifest = pd.read_csv(ARMS[arm].source_manifest, low_memory=False)
    frames: list[pd.DataFrame] = []
    for seed in ALL_SEEDS:
        if seed in ADOPTED_SEEDS:
            _validate_adopted_source_cv(arm, seed)
        else:
            _validate_source_cv(output_root, arm, seed)
        oof = pd.read_parquet(_source_oof_path(output_root, arm, seed))
        patient = evaluate.to_patient_level(oof, manifest)
        keep = patient[["patient_id", "label", "mean_logit"]].copy()
        keep["seed"] = seed
        frames.append(keep)
    stacked = pd.concat(frames, ignore_index=True)
    counts = stacked.groupby("patient_id", sort=False)["seed"].nunique()
    if not counts.eq(len(ALL_SEEDS)).all() or set(stacked["seed"].astype(int)) != set(ALL_SEEDS):
        raise ContractError(f"{arm}: incomplete five-seed source OOF ensemble")
    label_counts = stacked.groupby("patient_id", sort=False)["label"].nunique()
    if not label_counts.eq(1).all():
        raise ContractError(f"{arm}: source OOF labels differ across seeds")
    patient = (
        stacked.groupby("patient_id", sort=True)
        .agg(label=("label", "first"), mean_logit=("mean_logit", "mean"))
        .reset_index()
    )
    patient["label"] = patient["label"].astype(int)
    patient["prob_raw"] = sigmoid(patient["mean_logit"].to_numpy(float))
    patient["arm"] = arm
    return patient


def _fit_five_seed_calibrator(source_oof: pd.DataFrame, arm: str) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    labels = source_oof["label"].to_numpy(int)
    logits = source_oof[["mean_logit"]].to_numpy(float)
    if set(labels) != {0, 1} or not np.isfinite(logits).all():
        raise ContractError(f"{arm}: invalid source OOF data for Platt calibration")
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10_000)
    model.fit(logits, labels)
    return {
        "schema_version": 1,
        "method": "Platt logistic regression on five-seed mean native OOF logits",
        "arm": arm,
        "seeds": list(ALL_SEEDS),
        "n_source_patients": int(len(source_oof)),
        "n_source_mutant": int(labels.sum()),
        "a": float(model.intercept_[0]),
        "b": float(model.coef_[0, 0]),
        "converged": bool(model.n_iter_[0] < model.max_iter),
        "target_outcomes_used": False,
    }


def _outcome_frame_for_target(
    contract: dict[str, Any], arm: str, target: str
) -> pd.DataFrame:
    if target == "primary":
        identity = contract["label_blind_inputs"][f"primary/{arm}"][
            "source_used_only_to_construct_allowlisted_blind_manifest"
        ]
        _validate_identity(identity, deep=True)
        if Path(identity["path"]).resolve() != ARMS[arm].primary_target.resolve():
            raise ContractError(f"{arm}: sealed primary outcome path changed")
        return pd.read_csv(identity["path"], low_memory=False)
    if target in MET_TARGETS:
        source = contract[
            "outcome_sources_not_interpreted_or_joined_before_inference_seal"
        ]["e2met"][target]
        identity = source["artifact"]
        _validate_identity(identity, deep=True)
        frame = pd.read_csv(identity["path"], low_memory=False)
        if source.get("filter") == "subcohort == SR1482":
            frame = frame.loc[frame["subcohort"].eq("SR1482")].copy()
        return frame
    if target == "orion":
        identity = contract[
            "outcome_sources_not_interpreted_or_joined_before_inference_seal"
        ][
            "orion_patient_labels"
        ]
        _validate_identity(identity, deep=True)
        frame = pd.read_parquet(identity["path"])
        labels = frame[["patient_id", "label"]].drop_duplicates()
        if labels.groupby("patient_id")["label"].nunique().max() != 1:
            raise ContractError("Orion patient labels differ across prior scorers")
        return labels.drop_duplicates("patient_id")
    raise ValueError(target)


def _patient_from_score_frames(
    score_frames: Sequence[pd.DataFrame],
    blind_manifest: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    scores = pd.concat(score_frames, ignore_index=True)
    if scores.duplicated(["slide_id", "seed", "fold"]).any():
        raise ContractError("Duplicate slide/model logits before patient aggregation")
    expected_models = {(int(seed), 0) for seed in expected_seeds}
    observed_models = set(
        zip(scores["seed"].astype(int), scores["fold"].astype(int), strict=True)
    )
    if observed_models != expected_models:
        raise ContractError(f"Wrong model ensemble: {sorted(observed_models)}")
    counts = scores.groupby("slide_id", sort=False).size()
    if (
        len(counts) != len(blind_manifest)
        or not counts.eq(len(expected_seeds)).all()
        or set(counts.index.astype(str)) != set(blind_manifest["slide_id"].astype(str))
    ):
        raise ContractError("Every target slide must have exactly one logit per requested seed")
    slide = scores.groupby("slide_id", sort=False)["logit"].mean().rename("slide_logit")
    merged = blind_manifest.merge(
        slide, left_on="slide_id", right_index=True, how="left", validate="one_to_one"
    )
    if merged["slide_logit"].isna().any():
        raise ContractError("Missing slide logits after ensemble aggregation")
    aggregation: dict[str, tuple[str, str]] = {
        "mean_logit": ("slide_logit", "mean"),
        "n_slides": ("slide_id", "nunique"),
    }
    for column in ("cohort", "subcohort", "specimen_role", "liver_class"):
        if column in merged:
            aggregation[column] = (column, "first")
    for column in ("exclude_neoadjuvant", "exclude_ambiguous_crc15"):
        if column in merged:
            aggregation[column] = (column, "max")
    patient = merged.groupby("patient_id", sort=True).agg(**aggregation).reset_index()
    if "target_label" in outcomes:
        labels = outcomes[["patient_id", "target_label"]].rename(columns={"target_label": "label"})
    elif "label" in outcomes:
        labels = outcomes[["patient_id", "label"]].copy()
    else:
        raise ContractError("Outcome source lacks a KRAS label column")
    labels["patient_id"] = labels["patient_id"].astype(str)
    label_nunique = labels.groupby("patient_id", sort=False)["label"].nunique()
    if not label_nunique.eq(1).all():
        raise ContractError("Outcome source has inconsistent patient labels")
    labels = labels.drop_duplicates("patient_id")
    patient = patient.merge(labels, on="patient_id", how="left", validate="one_to_one")
    if patient["label"].isna().any() or set(patient["label"].astype(int)) != {0, 1}:
        raise ContractError("Missing or non-binary KRAS outcomes after sealed inference join")
    patient["label"] = patient["label"].astype(int)
    patient["prob_raw"] = sigmoid(patient["mean_logit"].to_numpy(float))
    return patient


def _five_seed_patients(
    output_root: Path,
    contract: dict[str, Any],
    arm: str,
    target: str,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    manifest_path = _blind_manifest_for_score(output_root, arm, target)
    blind = pd.read_csv(manifest_path, low_memory=False)
    outcomes = _outcome_frame_for_target(contract, arm, target)
    frames: dict[int, pd.DataFrame] = {
        seed: pd.read_parquet(_resolved_score_path(output_root, arm, target, seed))
        for seed in ALL_SEEDS
    }
    ensemble = _patient_from_score_frames(
        list(frames.values()), blind, outcomes, expected_seeds=ALL_SEEDS
    )
    per_seed = {
        seed: _patient_from_score_frames([frame], blind, outcomes, expected_seeds=[seed])
        for seed, frame in frames.items()
    }
    return ensemble, per_seed


def _metric_block(
    patients: pd.DataFrame,
    calibrator: dict[str, Any],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    from oceanpath.eval.core import compute_calibration_intercept_slope

    labels = patients["label"].to_numpy(int)
    logits = patients["mean_logit"].to_numpy(float)
    if set(labels) != {0, 1}:
        return {"n": int(len(labels)), "degenerate": True}
    raw = sigmoid(logits)
    calibrated = sigmoid(float(calibrator["a"]) + float(calibrator["b"]) * logits)
    rng = np.random.default_rng(bootstrap_seed)
    by_class = [np.flatnonzero(labels == value) for value in (0, 1)]
    bootstrap: dict[str, list[float]] = {
        "auroc": [], "auprc": [], "brier_raw": [], "brier_cal": [],
        "log_loss_raw": [], "log_loss_cal": [],
    }
    for _ in range(n_bootstrap):
        index = np.concatenate(
            [rng.choice(candidates, size=len(candidates), replace=True) for candidates in by_class]
        )
        y = labels[index]
        eta = logits[index]
        p_raw = np.clip(raw[index], 1e-6, 1 - 1e-6)
        p_cal = np.clip(calibrated[index], 1e-6, 1 - 1e-6)
        bootstrap["auroc"].append(float(roc_auc_score(y, eta)))
        bootstrap["auprc"].append(float(average_precision_score(y, eta)))
        bootstrap["brier_raw"].append(float(brier_score_loss(y, p_raw)))
        bootstrap["brier_cal"].append(float(brier_score_loss(y, p_cal)))
        bootstrap["log_loss_raw"].append(float(log_loss(y, p_raw, labels=[0, 1])))
        bootstrap["log_loss_cal"].append(float(log_loss(y, p_cal, labels=[0, 1])))

    def interval(key: str) -> list[float]:
        return [
            float(np.percentile(bootstrap[key], 2.5)),
            float(np.percentile(bootstrap[key], 97.5)),
        ]

    raw_cal = compute_calibration_intercept_slope(labels, np.clip(raw, 1e-6, 1 - 1e-6))
    source_cal = compute_calibration_intercept_slope(
        labels, np.clip(calibrated, 1e-6, 1 - 1e-6)
    )
    return {
        "n": int(len(labels)),
        "n_mutant": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, logits)),
        "auroc_ci95": interval("auroc"),
        "auprc": float(average_precision_score(labels, logits)),
        "auprc_ci95": interval("auprc"),
        "auprc_baseline": float(labels.mean()),
        "raw": {
            "brier": float(brier_score_loss(labels, raw)),
            "brier_ci95": interval("brier_raw"),
            "log_loss": float(log_loss(labels, np.clip(raw, 1e-6, 1 - 1e-6), labels=[0, 1])),
            "log_loss_ci95": interval("log_loss_raw"),
            "calibration_intercept": float(raw_cal["calibration_intercept"]),
            "calibration_slope": float(raw_cal["calibration_slope"]),
        },
        "source_calibrated": {
            "brier": float(brier_score_loss(labels, calibrated)),
            "brier_ci95": interval("brier_cal"),
            "log_loss": float(
                log_loss(labels, np.clip(calibrated, 1e-6, 1 - 1e-6), labels=[0, 1])
            ),
            "log_loss_ci95": interval("log_loss_cal"),
            "calibration_intercept": float(source_cal["calibration_intercept"]),
            "calibration_slope": float(source_cal["calibration_slope"]),
        },
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_unit": "patient",
    }


def _with_calibrated_probability(
    patients: pd.DataFrame, calibrator: dict[str, Any]
) -> pd.DataFrame:
    out = patients.copy()
    out["prob_source_calibrated"] = sigmoid(
        float(calibrator["a"]) + float(calibrator["b"]) * out["mean_logit"].to_numpy(float)
    )
    return out


def _per_seed_aurocs(per_seed: dict[int, pd.DataFrame]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    return {
        str(seed): float(roc_auc_score(frame["label"], frame["mean_logit"]))
        for seed, frame in per_seed.items()
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _report_paths(output_root: Path) -> dict[str, Path]:
    root = component_root(output_root) / "analysis"
    return {
        "source_oof": root / "source_oof_five_seed.parquet",
        "calibrators": root / "source_calibrators_five_seed.json",
        "primary_patients": root / "primary_patient_scores_five_seed.parquet",
        "met_patients": root / "e2met_patient_scores_five_seed.parquet",
        "orion_patients": root / "e2cpht_orion_patient_scores_five_seed.parquet",
        "results": root / "results_five_seed.json",
        "table": root / "results_five_seed.csv",
        "receipt": root / "results_five_seed.receipt.json",
    }


def _validate_report(output_root: Path) -> dict[str, Any]:
    files = _report_paths(output_root)
    if not all(path.is_file() for path in files.values()):
        raise FileNotFoundError("Five-seed Aim-2 report is incomplete")
    receipt = _read_json(files["receipt"])
    artifacts = {key: _artifact(path) for key, path in files.items() if key != "receipt"}
    expected = {
        "schema_version": 1,
        "status": "sealed_five_seed_results",
        "contract": _artifact(contract_path(output_root)),
        "inference_seal": _artifact(inference_seal_path(output_root)),
        "artifacts": artifacts,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ContractError("Five-seed report receipt mismatch")
    results = _read_json(files["results"])
    _replay_report(output_root, results)
    return results


def _replay_report(output_root: Path, results: dict[str, Any]) -> None:
    """Deterministically replay load-bearing results from sealed patient tables."""

    from sklearn.metrics import roc_auc_score

    files = _report_paths(output_root)
    if results.get("seeds") != list(ALL_SEEDS) or results.get("bootstrap") != {
        "unit": "patient",
        "n": N_BOOTSTRAP,
        "seed": BOOTSTRAP_SEED,
    }:
        raise ContractError("Reported seed/bootstrap contract drifted")
    expected_header = {
        "schema_version": 1,
        "experiment": "Aim 2 complete LOCO MIL five-seed results",
        "ensemble_rule": "mean native logits across seeds, then one sigmoid",
        "calibration": (
            "one source-only Platt map per arm fitted on five-seed mean source OOF native logits"
        ),
    }
    if any(results.get(key) != value for key, value in expected_header.items()):
        raise ContractError("Reported five-seed analysis header drifted")
    source = pd.read_parquet(files["source_oof"])
    calibrators = _read_json(files["calibrators"])
    contract = validate_contract(output_root, deep=False, adoptions=True)
    expected_mixed_scope = {
        "five_seed": [
            "all LOCO primary",
            "E2-MET complete LOCO matrix",
            "Orion LOCO sensitivity",
        ],
        "three_seed_unchanged": [
            "raw all-conventional CPHT",
            "CPHT-A residual adaptation",
        ],
        "references": contract["mixed_seed_scope_references"],
    }
    if results.get("mixed_seed_scope") != expected_mixed_scope:
        raise ContractError("Mixed five-/three-seed scope declaration drifted")
    for arm in ARM_NAMES:
        live_source = _five_seed_source_oof(output_root, arm).sort_values("patient_id").reset_index(
            drop=True
        )
        sealed_source = (
            source.loc[source["arm"].eq(arm)]
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            live_source[["patient_id", "label", "mean_logit"]],
            sealed_source[["patient_id", "label", "mean_logit"]],
            check_dtype=False,
            obj=f"{arm} source OOF replay",
        )
        replayed = _fit_five_seed_calibrator(source.loc[source["arm"].eq(arm)].copy(), arm)
        if _json_safe(replayed) != calibrators[arm]:
            raise ContractError(f"{arm}: source calibrator fails deterministic replay")

    primary_table = pd.read_parquet(files["primary_patients"])
    primary = {arm: primary_table.loc[primary_table["arm"].eq(arm)].copy() for arm in ARM_NAMES}
    for ordinal, (arm, frame) in enumerate(primary.items()):
        live, per_seed = _five_seed_patients(output_root, contract, arm, "primary")
        live = _with_calibrated_probability(live, calibrators[arm])
        _assert_patient_replay(live, frame, context=f"{arm}/primary")
        replayed_block = {
            **_metric_block(
                live,
                calibrators[arm],
                n_bootstrap=N_BOOTSTRAP,
                bootstrap_seed=BOOTSTRAP_SEED + ordinal,
            ),
            "per_seed_auroc": _per_seed_aurocs(per_seed),
        }
        if _json_safe(replayed_block) != results["primary"][arm]:
            raise ContractError(f"{arm}: full primary metric block fails replay")
        expected_probability = sigmoid(
            float(calibrators[arm]["a"])
            + float(calibrators[arm]["b"]) * frame["mean_logit"].to_numpy(float)
        )
        if not np.allclose(frame["prob_source_calibrated"], expected_probability):
            raise ContractError(f"{arm}: calibrated probabilities fail replay")
    replayed_macro = _family_standardized_macro(
        results["primary"],
        primary,
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    if _json_safe(replayed_macro) != results["family_loco_standardized_macro"]:
        raise ContractError("Family standardized macro fails deterministic replay")
    replayed_sibling_gate = _sibling_directional_gate(results["primary"])
    if _json_safe(replayed_sibling_gate) != results["sibling_loco_directional_gate"]:
        raise ContractError("Sibling 4/4 directional gate fails deterministic replay")

    sibling_pairs = {
        "sibling_sr386": ("family_surgen", "SR386"),
        "sibling_sr1482": ("family_surgen", "SR1482"),
        "sibling_tcga_coad": ("family_tcga", "TCGA-COAD"),
        "sibling_tcga_read": ("family_tcga", "TCGA-READ"),
    }
    replayed_contrasts: dict[str, Any] = {}
    for ordinal, (sibling, (family, subcohort)) in enumerate(sibling_pairs.items()):
        family_frame = primary[family].loc[primary[family]["subcohort"].eq(subcohort)].copy()
        replayed_contrasts[f"{sibling}_minus_{family}_{subcohort}"] = evaluate.compare_auroc_paired(
            family_frame,
            primary[sibling],
            score_column="mean_logit",
            n_bootstrap=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED + 300 + ordinal,
        )
    replayed_contrasts["family_rih_sm_minus_family_rih"] = evaluate.compare_auroc_paired(
        primary["family_rih"],
        primary["family_rih_sm"],
        score_column="mean_logit",
        n_bootstrap=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED + 310,
    )
    if _json_safe(replayed_contrasts) != results["paired_sibling_and_size_matched_contrasts"]:
        raise ContractError("Sibling/size-matched paired contrasts fail deterministic replay")

    met_table = pd.read_parquet(files["met_patients"])
    met = {
        (arm, target): met_table.loc[
            met_table["arm"].eq(arm) & met_table["target"].eq(target)
        ].copy()
        for arm in CONTROLLING_ARMS
        for target in MET_TARGETS
    }
    matrix = results["e2met_complete_eight_by_two_matrix"]
    dual = set(matrix["rih_dual_role_patient_ids"])
    for (arm, target), frame in met.items():
        evaluated = (
            frame.loc[~frame["patient_id"].astype(str).isin(dual)].copy()
            if target == "rih_m"
            else frame
        )
        block = matrix["targets"][target]["metrics"][arm]
        if not np.isclose(
            roc_auc_score(evaluated["label"], evaluated["mean_logit"]), block["auroc"]
        ):
            raise ContractError(f"{arm}/{target}: E2-MET AUROC fails replay")
    replayed_matrix = _official_e2met_matrix(
        contract,
        met,
        calibrators,
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    if _json_safe(replayed_matrix) != matrix:
        raise ContractError("Official E2-MET matrix/paired contrasts fail deterministic replay")
    _replay_confirmatory_met(results, met, calibrators)

    orion_table = pd.read_parquet(files["orion_patients"])
    for ordinal, arm in enumerate(CONTROLLING_ARMS, start=200):
        frame = orion_table.loc[orion_table["arm"].eq(arm)].copy()
        live, per_seed = _five_seed_patients(output_root, contract, arm, "orion")
        live = _with_calibrated_probability(live, calibrators[arm])
        _assert_patient_replay(live, frame, context=f"{arm}/orion")
        populations = {"all": live}
        for column in ("exclude_neoadjuvant", "exclude_ambiguous_crc15"):
            if column in live:
                values = live[column]
                excluded = (
                    values.astype(str).str.lower().isin({"true", "1", "yes"})
                    if values.dtype != bool
                    else values
                )
                populations[column] = live.loc[~excluded].copy()
        replayed_orion = {
            "populations": {
                name: _metric_block(
                    population,
                    calibrators[arm],
                    n_bootstrap=N_BOOTSTRAP,
                    bootstrap_seed=BOOTSTRAP_SEED + ordinal + index,
                )
                for index, (name, population) in enumerate(populations.items())
            },
            "per_seed_auroc_all": _per_seed_aurocs(per_seed),
        }
        if _json_safe(replayed_orion) != results[
            "e2cpht_orion_complete_eight_loco_sensitivity"
        ][arm]:
            raise ContractError(f"{arm}: full Orion metric block fails replay")

    for (arm, target), frame in met.items():
        live, _ = _five_seed_patients(output_root, contract, arm, target)
        live = _with_calibrated_probability(live, calibrators[arm])
        _assert_patient_replay(live, frame, context=f"{arm}/{target}")

    expected_rows: list[dict[str, Any]] = []
    for arm, block in results["primary"].items():
        expected_rows.append(
            {"scope": "primary", "arm": arm, "target": "primary", **{
                key: block.get(key) for key in ("n", "n_mutant", "auroc", "auprc")
            }}
        )
    for target, target_block in matrix["targets"].items():
        for arm, block in target_block["metrics"].items():
            expected_rows.append(
                {"scope": "e2met", "arm": arm, "target": target, **{
                    key: block.get(key) for key in ("n", "n_mutant", "auroc", "auprc")
                }}
            )
    for arm, block in results["e2cpht_orion_complete_eight_loco_sensitivity"].items():
        cell = block["populations"]["all"]
        expected_rows.append(
            {"scope": "e2cpht", "arm": arm, "target": "orion", **{
                key: cell.get(key) for key in ("n", "n_mutant", "auroc", "auprc")
            }}
        )
    observed_table = pd.read_csv(files["table"])
    expected_table = pd.DataFrame(expected_rows)
    order = ["scope", "arm", "target"]
    pd.testing.assert_frame_equal(
        observed_table.sort_values(order).reset_index(drop=True),
        expected_table.sort_values(order).reset_index(drop=True),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
        obj="Aim-2 results CSV semantic replay",
    )


def _assert_patient_replay(live: pd.DataFrame, sealed: pd.DataFrame, *, context: str) -> None:
    columns = ["patient_id", "label", "mean_logit", "prob_raw", "prob_source_calibrated"]
    left = live[columns].sort_values("patient_id").reset_index(drop=True)
    right = sealed[columns].sort_values("patient_id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left, right, check_dtype=False, rtol=1e-12, atol=1e-12, obj=context
        )
    except AssertionError as exc:
        raise ContractError(f"{context}: sealed patient scores fail deterministic replay") from exc


def _replay_confirmatory_met(
    results: dict[str, Any],
    met: dict[tuple[str, str], pd.DataFrame],
    calibrators: dict[str, dict[str, Any]],
) -> None:
    replayed_targets, replayed_conclusion = _official_confirmatory_met(
        met,
        calibrators,
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    if _json_safe(replayed_targets) != results["e2met_confirmatory_family_naive"]:
        raise ContractError("Confirmatory E2-MET target metrics fail deterministic replay")
    if _json_safe(replayed_conclusion) != results["e2met_confirmatory"]:
        raise ContractError("Confirmatory E2-MET macro/gate fails deterministic replay")


def _family_standardized_macro(
    primary_results: dict[str, Any],
    primary_patients: dict[str, pd.DataFrame],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Frozen target-x-KRAS bootstrap for the nested four-family macro."""

    from sklearn.metrics import roc_auc_score

    surgen = primary_patients["family_surgen"]
    directional_frames = {
        "CPTAC": primary_patients["family_cptac"],
        "RIH": primary_patients["family_rih"],
        "SR386_given_whole_SurGen_holdout": surgen.loc[surgen["subcohort"].eq("SR386")],
        "SR1482_given_whole_SurGen_holdout": surgen.loc[surgen["subcohort"].eq("SR1482")],
        "TCGA_pooled_COAD_READ": primary_patients["family_tcga"],
    }
    directional_points = {
        name: float(roc_auc_score(frame["label"], frame["mean_logit"]))
        for name, frame in directional_frames.items()
    }
    rng = np.random.default_rng(bootstrap_seed)
    directional_draws = {
        name: _stratified_auroc_draws(frame, n_bootstrap, rng)
        for name, frame in directional_frames.items()
    }
    macro_draws = np.mean(
        np.vstack(
            [
                directional_draws["CPTAC"],
                directional_draws["RIH"],
                np.mean(
                    np.vstack(
                        [
                            directional_draws["SR386_given_whole_SurGen_holdout"],
                            directional_draws["SR1482_given_whole_SurGen_holdout"],
                        ]
                    ),
                    axis=0,
                ),
                directional_draws["TCGA_pooled_COAD_READ"],
            ]
        ),
        axis=0,
    )
    components = {
        "CPTAC": directional_points["CPTAC"],
        "RIH": directional_points["RIH"],
        "SurGen_mean_SR386_SR1482": float(
            np.mean(
                [
                    directional_points["SR386_given_whole_SurGen_holdout"],
                    directional_points["SR1482_given_whole_SurGen_holdout"],
                ]
            )
        ),
        "TCGA_pooled_COAD_READ": directional_points["TCGA_pooled_COAD_READ"],
    }
    macro_ci = [float(np.percentile(macro_draws, 2.5)), float(np.percentile(macro_draws, 97.5))]
    direction_results = {
        name: {
            "auroc": point,
            "auroc_ci95": [
                float(np.percentile(directional_draws[name], 2.5)),
                float(np.percentile(directional_draws[name], 97.5)),
            ],
            "point_above_0p5": bool(point > 0.5),
        }
        for name, point in directional_points.items()
    }
    passed = sum(int(block["point_above_0p5"]) for block in direction_results.values())
    all_directions = bool(passed == 5)
    macro_lower = bool(macro_ci[0] > 0.5)
    return {
        "formula": "(CPTAC + RIH + mean(SR386, SR1482) + pooled_TCGA) / 4",
        "components": components,
        "macro_auroc": float(np.mean(list(components.values()))),
        "macro_auroc_ci95": macro_ci,
        "stratification": "target_x_KRAS",
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(bootstrap_seed),
        "directional_results": direction_results,
        "directional_gate": {
            "criterion": "all five target AUROC point estimates > 0.5",
            "passed_directions": int(passed),
            "required_directions": 5,
            "all_5_points_above_0p5": all_directions,
        },
        "macro_lower_bound_above_0p5": macro_lower,
        "claim_family_loco_transport": bool(all_directions and macro_lower),
    }


def _sibling_directional_gate(primary_results: dict[str, Any]) -> dict[str, Any]:
    arms = ("sibling_sr386", "sibling_sr1482", "sibling_tcga_coad", "sibling_tcga_read")
    directions = {
        arm: {
            "auroc": float(primary_results[arm]["auroc"]),
            "auroc_ci95": primary_results[arm]["auroc_ci95"],
            "point_above_0p5": bool(primary_results[arm]["auroc"] > 0.5),
            "lower_ci_above_0p5": bool(primary_results[arm]["auroc_ci95"][0] > 0.5),
        }
        for arm in arms
    }
    passed = sum(int(block["point_above_0p5"]) for block in directions.values())
    return {
        "criterion": "all four sibling-stratum LOCO AUROC point estimates > 0.5",
        "directions": directions,
        "passed_directions": int(passed),
        "required_directions": 4,
        "all_4_points_above_0p5": bool(passed == 4),
    }


def _stratified_auroc_draws(
    patients: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    from sklearn.metrics import roc_auc_score

    labels = patients["label"].to_numpy(int)
    logits = patients["mean_logit"].to_numpy(float)
    candidates = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(len(items) == 0 for items in candidates):
        raise ContractError("Confirmatory metastatic target lacks both KRAS classes")
    values = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        draw = np.concatenate(
            [rng.choice(items, size=len(items), replace=True) for items in candidates]
        )
        values[index] = roc_auc_score(labels[draw], logits[draw])
    return values


def _confirmatory_met_conclusion(
    met_results: dict[str, Any],
    met_patients: dict[tuple[str, str], pd.DataFrame],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    point_by_target = {
        "rih_m": float(met_results["family_rih"]["rih_m"]["auroc"]),
        "sr1482_m": float(met_results["family_surgen"]["sr1482_m"]["auroc"]),
    }
    rng = np.random.default_rng(bootstrap_seed)
    draws = [
        _stratified_auroc_draws(
            met_patients[(arm, target)], n_bootstrap=n_bootstrap, rng=rng
        )
        for arm, target in (("family_rih", "rih_m"), ("family_surgen", "sr1482_m"))
    ]
    macro_draws = np.mean(np.vstack(draws), axis=0)
    macro_ci = [
        float(np.percentile(macro_draws, 2.5)),
        float(np.percentile(macro_draws, 97.5)),
    ]
    both_points = bool(all(value > 0.5 for value in point_by_target.values()))
    lower_bound = bool(macro_ci[0] > 0.5)
    return {
        "target_aurocs": point_by_target,
        "equal_cohort_metastatic_macro_auroc": float(np.mean(list(point_by_target.values()))),
        "macro_auroc_ci95": macro_ci,
        "both_target_points_above_0p5": both_points,
        "macro_lower_bound_above_0p5": lower_bound,
        "claim_metastatic_transport": bool(both_points and lower_bound),
        "gate": "both target AUROC points > 0.5 AND macro AUROC CI95 lower bound > 0.5",
        "bootstrap_unit": "patient within target; equal-cohort macro per replicate",
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(bootstrap_seed),
    }


def _dual_rih_patient_roster(
    contract: dict[str, Any], rih_metastatic: pd.DataFrame
) -> list[str]:
    source = contract[
        "outcome_sources_not_interpreted_or_joined_before_inference_seal"
    ]["e2met"]["rih_primary"]
    identity = source["artifact"]
    _validate_identity(identity, deep=True)
    primary = pd.read_csv(identity["path"], low_memory=False)
    dual = sorted(
        set(primary["patient_id"].astype(str))
        & set(rih_metastatic["patient_id"].astype(str))
    )
    if len(dual) != 8:
        raise ContractError(f"RIH primary/metastatic dual-role roster changed: n={len(dual)}")
    return dual


def _official_e2met_matrix(
    contract: dict[str, Any],
    met_patients: dict[tuple[str, str], pd.DataFrame],
    calibrators: dict[str, dict[str, Any]],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Eight scorers x two targets with one shared roster/bootstrap per target."""

    import aim2_cross_protocol_transfer as cpht
    import aim2_primary_to_metastatic_transfer as e2met

    dual = _dual_rih_patient_roster(contract, met_patients[("family_rih", "rih_m")])
    common_ids = {
        "rih_m": sorted(
            set(met_patients[("family_rih", "rih_m")]["patient_id"].astype(str)) - set(dual)
        ),
        "sr1482_m": sorted(
            set(met_patients[("family_surgen", "sr1482_m")]["patient_id"].astype(str))
        ),
    }
    rng = np.random.default_rng(bootstrap_seed)
    targets: dict[str, Any] = {}
    for target in MET_TARGETS:
        ordered = common_ids[target]
        aligned: dict[str, pd.DataFrame] = {}
        for arm in CONTROLLING_ARMS:
            frame = met_patients[(arm, target)].copy()
            frame["patient_id"] = frame["patient_id"].astype(str)
            frame = frame.set_index("patient_id").loc[ordered].reset_index()
            if frame["patient_id"].tolist() != ordered:
                raise ContractError(f"{arm}/{target}: common metastatic roster mismatch")
            aligned[arm] = frame
        base = aligned[CONTROLLING_ARMS[0]]
        for arm, frame in aligned.items():
            if not np.array_equal(frame["label"].to_numpy(int), base["label"].to_numpy(int)):
                raise ContractError(f"{arm}/{target}: common metastatic labels differ")
        indices = e2met._shared_stratified_indices(  # noqa: SLF001
            base["label"].to_numpy(int), n_bootstrap, rng
        )
        metrics: dict[str, Any] = {}
        samples: dict[str, dict[str, np.ndarray]] = {}
        for arm, frame in aligned.items():
            labels = frame["label"].to_numpy(int)
            logits = frame["mean_logit"].to_numpy(float)
            probability = sigmoid(
                float(calibrators[arm]["a"]) + float(calibrators[arm]["b"]) * logits
            )
            sample = cpht.bootstrap_metric_samples(
                labels, logits, indices, probability=probability
            )
            block = cpht._rank_metric_block(labels, logits, sample)  # noqa: SLF001
            block["source_calibrated"] = cpht._probability_metric_block(  # noqa: SLF001
                labels, probability, sample
            )
            metrics[arm] = block
            samples[arm] = sample
        paired: dict[str, Any] = {}
        for left_index, left in enumerate(CONTROLLING_ARMS):
            for right in CONTROLLING_ARMS[left_index + 1 :]:
                draws = samples[right]["auroc"] - samples[left]["auroc"]
                paired[f"{right}_minus_{left}"] = {
                    "delta_auroc": float(metrics[right]["auroc"] - metrics[left]["auroc"]),
                    "delta_auroc_ci95": [
                        float(np.percentile(draws, 2.5)),
                        float(np.percentile(draws, 97.5)),
                    ],
                    "paired_shared_patient_bootstrap": True,
                }
        targets[target] = {
            "n_common": int(len(base)),
            "n_mutant": int(base["label"].sum()),
            "metrics": metrics,
            "paired_auroc_contrasts": paired,
        }
    if (targets["rih_m"]["n_common"], targets["rih_m"]["n_mutant"]) != (77, 33):
        raise ContractError("Leakage-free RIH-M matrix census changed from 77/33")
    if targets["sr1482_m"]["n_common"] != 74:
        raise ContractError("SR1482-M matrix census changed from 74")
    return {
        "status": "complete eight-heldout-scorer x two-target matrix",
        "complete_eight_model_matrix": True,
        "scorers": list(CONTROLLING_ARMS),
        "targets": targets,
        "rih_population_rule": (
            "all matrix cells and every paired model contrast use the same 77 patients "
            "after excluding eight dual-role patients"
        ),
        "rih_dual_role_patient_ids": dual,
        "bootstrap": {
            "unit": "patient",
            "stratification": "target_x_KRAS",
            "indices_shared_across_all_eight_scorers": True,
            "n": int(n_bootstrap),
            "seed": int(bootstrap_seed),
        },
    }


def _official_confirmatory_met(
    met_patients: dict[tuple[str, str], pd.DataFrame],
    calibrators: dict[str, dict[str, Any]],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Family-naive RIH(all 85) and SR1482(74), plus their equal-target macro gate."""

    import aim2_cross_protocol_transfer as cpht
    import aim2_primary_to_metastatic_transfer as e2met

    mappings = {"rih_m": "family_rih", "sr1482_m": "family_surgen"}
    rng = np.random.default_rng(bootstrap_seed)
    targets: dict[str, Any] = {}
    samples: dict[str, np.ndarray] = {}
    for target, arm in mappings.items():
        frame = met_patients[(arm, target)].sort_values("patient_id").reset_index(drop=True)
        labels = frame["label"].to_numpy(int)
        logits = frame["mean_logit"].to_numpy(float)
        probability = sigmoid(
            float(calibrators[arm]["a"]) + float(calibrators[arm]["b"]) * logits
        )
        indices = e2met._shared_stratified_indices(  # noqa: SLF001
            labels, n_bootstrap, rng
        )
        metric_samples = cpht.bootstrap_metric_samples(
            labels, logits, indices, probability=probability
        )
        block = cpht._rank_metric_block(labels, logits, metric_samples)  # noqa: SLF001
        block["source_calibrated"] = cpht._probability_metric_block(  # noqa: SLF001
            labels, probability, metric_samples
        )
        block["scorer"] = arm
        block["exposure"] = "family-naive"
        targets[target] = block
        samples[target] = metric_samples["auroc"]
    macro_draws = np.mean(np.vstack([samples[target] for target in MET_TARGETS]), axis=0)
    macro_ci = [
        float(np.percentile(macro_draws, 2.5)),
        float(np.percentile(macro_draws, 97.5)),
    ]
    both_points = bool(all(targets[target]["auroc"] > 0.5 for target in MET_TARGETS))
    macro_lower = bool(macro_ci[0] > 0.5)
    conclusion = {
        "equal_cohort_metastatic_macro_auroc": float(
            np.mean([targets[target]["auroc"] for target in MET_TARGETS])
        ),
        "macro_auroc_ci95": macro_ci,
        "both_target_points_above_0p5": both_points,
        "macro_lower_bound_above_0p5": macro_lower,
        "claim_metastatic_transport": bool(both_points and macro_lower),
        "gate": "both target AUROC points > 0.5 AND macro AUROC CI95 lower bound > 0.5",
        "rih_confirmatory_population": "all 85 RIH-M patients including dual-role patients",
        "sr1482_confirmatory_population": "all 74 SR1482-M patients",
        "bootstrap_unit": "patient within target; equal-cohort macro per replicate",
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(bootstrap_seed),
    }
    return targets, conclusion


def cmd_report(args: argparse.Namespace) -> None:
    # This is the load-bearing ordering guarantee: target outcomes are not
    # interpreted or joined to predictions until all 165 inference artifacts
    # verify.  Source CSV bytes were previously read only to construct an
    # explicit allowlisted, outcome-free inference manifest.
    verify_inference_seal(args.output_root, deep=False)
    report_files = _report_paths(args.output_root)
    if report_files["results"].exists():
        _validate_report(args.output_root)
        print(f"PASS: existing five-seed Aim-2 report verifies: {report_files['results']}")
        return
    if not args.apply:
        print("Inference seal verified. Report dry run only; pass --apply to join outcomes.")
        return
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive")
    if (args.n_bootstrap, args.bootstrap_seed) != (N_BOOTSTRAP, BOOTSTRAP_SEED):
        raise ContractError(
            f"Frozen Aim-2 bootstrap is n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}"
        )
    contract = validate_contract(args.output_root, deep=False, adoptions=True)

    source_oof: dict[str, pd.DataFrame] = {}
    calibrators: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        source_oof[arm] = _five_seed_source_oof(args.output_root, arm)
        calibrators[arm] = _fit_five_seed_calibrator(source_oof[arm], arm)

    primary_patients: dict[str, pd.DataFrame] = {}
    primary_per_seed: dict[str, dict[int, pd.DataFrame]] = {}
    primary_results: dict[str, Any] = {}
    for ordinal, arm in enumerate(ARM_NAMES):
        patients, per_seed = _five_seed_patients(args.output_root, contract, arm, "primary")
        patients = _with_calibrated_probability(patients, calibrators[arm])
        patients["arm"] = arm
        patients["target"] = "primary"
        primary_patients[arm] = patients
        primary_per_seed[arm] = per_seed
        primary_results[arm] = {
            **_metric_block(
                patients,
                calibrators[arm],
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.bootstrap_seed + ordinal,
            ),
            "per_seed_auroc": _per_seed_aurocs(per_seed),
        }

    met_patients: dict[tuple[str, str], pd.DataFrame] = {}
    for arm in CONTROLLING_ARMS:
        for target in MET_TARGETS:
            patients, _ = _five_seed_patients(args.output_root, contract, arm, target)
            patients = _with_calibrated_probability(patients, calibrators[arm])
            patients["arm"] = arm
            patients["target"] = target
            met_patients[(arm, target)] = patients

    orion_patients: dict[str, pd.DataFrame] = {}
    orion_results: dict[str, Any] = {}
    for ordinal, arm in enumerate(CONTROLLING_ARMS, start=200):
        patients, per_seed = _five_seed_patients(args.output_root, contract, arm, "orion")
        patients = _with_calibrated_probability(patients, calibrators[arm])
        patients["arm"] = arm
        patients["target"] = "orion"
        orion_patients[arm] = patients
        populations = {"all": patients}
        for column in ("exclude_neoadjuvant", "exclude_ambiguous_crc15"):
            if column in patients:
                values = patients[column]
                excluded = (
                    values.astype(str).str.lower().isin({"true", "1", "yes"})
                    if values.dtype != bool
                    else values
                )
                populations[column] = patients.loc[~excluded].copy()
        orion_results[arm] = {
            "populations": {
                name: _metric_block(
                    frame,
                    calibrators[arm],
                    n_bootstrap=args.n_bootstrap,
                    bootstrap_seed=args.bootstrap_seed + ordinal + index,
                )
                for index, (name, frame) in enumerate(populations.items())
            },
            "per_seed_auroc_all": _per_seed_aurocs(per_seed),
        }

    family_macro = _family_standardized_macro(
        primary_results,
        primary_patients,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )

    sibling_pairs = {
        "sibling_sr386": ("family_surgen", "SR386"),
        "sibling_sr1482": ("family_surgen", "SR1482"),
        "sibling_tcga_coad": ("family_tcga", "TCGA-COAD"),
        "sibling_tcga_read": ("family_tcga", "TCGA-READ"),
    }
    paired_contrasts: dict[str, Any] = {}
    for ordinal, (sibling, (family, subcohort)) in enumerate(sibling_pairs.items()):
        family_frame = primary_patients[family].loc[
            primary_patients[family]["subcohort"].eq(subcohort)
        ].copy()
        paired_contrasts[f"{sibling}_minus_{family}_{subcohort}"] = evaluate.compare_auroc_paired(
            family_frame,
            primary_patients[sibling],
            score_column="mean_logit",
            n_bootstrap=args.n_bootstrap,
            seed=args.bootstrap_seed + 300 + ordinal,
        )
    paired_contrasts["family_rih_sm_minus_family_rih"] = evaluate.compare_auroc_paired(
        primary_patients["family_rih"],
        primary_patients["family_rih_sm"],
        score_column="mean_logit",
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed + 310,
    )

    confirmatory_targets, confirmatory_met = _official_confirmatory_met(
        met_patients,
        calibrators,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    official_met_matrix = _official_e2met_matrix(
        contract,
        met_patients,
        calibrators,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    results = {
        "schema_version": 1,
        "experiment": "Aim 2 complete LOCO MIL five-seed results",
        "seeds": list(ALL_SEEDS),
        "ensemble_rule": "mean native logits across seeds, then one sigmoid",
        "calibration": "one source-only Platt map per arm fitted on five-seed mean source OOF native logits",
        "bootstrap": {"unit": "patient", "n": args.n_bootstrap, "seed": args.bootstrap_seed},
        "primary": primary_results,
        "family_loco_standardized_macro": family_macro,
        "sibling_loco_directional_gate": _sibling_directional_gate(primary_results),
        "paired_sibling_and_size_matched_contrasts": paired_contrasts,
        "e2met_complete_eight_by_two_matrix": official_met_matrix,
        "e2met_confirmatory_family_naive": confirmatory_targets,
        "e2met_confirmatory": confirmatory_met,
        "e2cpht_orion_complete_eight_loco_sensitivity": orion_results,
        "mixed_seed_scope": {
            "five_seed": ["all LOCO primary", "E2-MET complete LOCO matrix", "Orion LOCO sensitivity"],
            "three_seed_unchanged": ["raw all-conventional CPHT", "CPHT-A residual adaptation"],
            "references": contract["mixed_seed_scope_references"],
        },
    }

    source_table = pd.concat(source_oof.values(), ignore_index=True)
    primary_table = pd.concat(primary_patients.values(), ignore_index=True)
    met_table = pd.concat(met_patients.values(), ignore_index=True)
    orion_table = pd.concat(orion_patients.values(), ignore_index=True)
    rows: list[dict[str, Any]] = []
    for arm, block in primary_results.items():
        rows.append({"scope": "primary", "arm": arm, "target": "primary", **{key: block.get(key) for key in ("n", "n_mutant", "auroc", "auprc")}})
    for target, target_block in official_met_matrix["targets"].items():
        for arm, block in target_block["metrics"].items():
            rows.append({"scope": "e2met", "arm": arm, "target": target, **{key: block.get(key) for key in ("n", "n_mutant", "auroc", "auprc")}})
    for arm, block in orion_results.items():
        all_block = block["populations"]["all"]
        rows.append({"scope": "e2cpht", "arm": arm, "target": "orion", **{key: all_block.get(key) for key in ("n", "n_mutant", "auroc", "auprc")}})

    lineage.write_parquet_once(report_files["source_oof"], source_table)
    _write_json_once(report_files["calibrators"], _json_safe(calibrators))
    lineage.write_parquet_once(report_files["primary_patients"], primary_table)
    lineage.write_parquet_once(report_files["met_patients"], met_table)
    lineage.write_parquet_once(report_files["orion_patients"], orion_table)
    _write_json_once(report_files["results"], _json_safe(results))
    _write_csv_once(report_files["table"], pd.DataFrame(rows))
    artifacts = {
        key: _artifact(path) for key, path in report_files.items() if key != "receipt"
    }
    _write_json_once(
        report_files["receipt"],
        {
            "schema_version": 1,
            "status": "sealed_five_seed_results",
            "created_utc": _utcnow(),
            "contract": _artifact(contract_path(args.output_root)),
            "inference_seal": _artifact(inference_seal_path(args.output_root)),
            "artifacts": artifacts,
            "target_outcomes_opened_only_after_inference_seal": True,
            "bootstrap_unit": "patient",
            "model_seeds_are_not_inference_units": True,
        },
    )
    _validate_report(args.output_root)
    print(f"PASS: sealed five-seed Aim-2 results: {report_files['results']}")


def cmd_verify(args: argparse.Namespace) -> None:
    contract = validate_contract(args.output_root, deep=args.deep, adoptions=True)
    for arm in ARM_NAMES:
        for seed in NEW_SEEDS:
            _validate_source_cv(args.output_root, arm, seed)
            _validate_refit(args.output_root, arm, seed)
    verify_inference_seal(args.output_root, deep=args.deep)
    _validate_report(args.output_root)
    for receipt_path in component_root(args.output_root).glob(
        "failures/*/*/seed*/attempt_*/quarantine_receipt.json"
    ):
        receipt = _read_json(receipt_path)
        if receipt.get("status") != "failed_attempt_quarantined_without_overwrite":
            raise ContractError(f"Invalid quarantine receipt: {receipt_path}")
        for identity in _iter_identities(receipt):
            _validate_identity(identity, deep=True)
    if contract["counts"] != {
        "arms": 9,
        "controlling_arms": 8,
        "adopted_fits": 162,
        "new_fits": 108,
        "complete_fits": 270,
        "training_chain_jobs": 36,
        "adopted_score_artifacts": 99,
        "new_score_artifacts": 66,
        "complete_score_artifacts": 165,
    }:
        raise ContractError("Final fit/score accounting changed")
    print(
        "PASS: Aim-2 five-seed extension verifies: 270 fits, 165 downstream score "
        "artifacts, source-only five-seed calibration, sealed results and receipts"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("plan", parents=[common])
    command.set_defaults(func=cmd_plan)
    command = sub.add_parser("manifest", parents=[common])
    command.add_argument("--apply", action="store_true")
    command.set_defaults(func=cmd_manifest)
    command = sub.add_parser("preflight", parents=[common])
    command.add_argument("--deep", action="store_true")
    command.add_argument("--apply", action="store_true")
    command.set_defaults(func=cmd_preflight)
    command = sub.add_parser("train", parents=[common])
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--apply", action="store_true")
    command.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    command.add_argument("--arm", choices=ARM_NAMES)
    command.add_argument("--seed", type=int, choices=NEW_SEEDS)
    command.add_argument("--stage", choices=("source_cv", "refit"))
    command.set_defaults(func=cmd_train)
    command = sub.add_parser("train-one", parents=[common])
    command.add_argument("--job-key", required=True)
    command.add_argument("--external-scheduler", action="store_true")
    command.add_argument("--recover-orphan", action="store_true")
    command.add_argument("--apply", action="store_true")
    command.set_defaults(func=cmd_train_one)
    command = sub.add_parser("quarantine-failed", parents=[common])
    command.add_argument("--job-key", required=True)
    command.add_argument("--apply", action="store_true")
    command.set_defaults(func=cmd_quarantine_failed)

    command = sub.add_parser("_source-cv", parents=[common])
    command.add_argument("--arm", choices=ARM_NAMES, required=True)
    command.add_argument("--seed", type=int, choices=NEW_SEEDS, required=True)
    command.set_defaults(func=cmd_internal_source_cv)
    command = sub.add_parser("_refit", parents=[common])
    command.add_argument("--arm", choices=ARM_NAMES, required=True)
    command.add_argument("--seed", type=int, choices=NEW_SEEDS, required=True)
    command.set_defaults(func=cmd_internal_refit)
    command = sub.add_parser("_score-one", parents=[common])
    command.add_argument("--arm", choices=ARM_NAMES, required=True)
    command.add_argument("--seed", type=int, choices=NEW_SEEDS, required=True)
    command.add_argument("--target", choices=("primary", *MET_TARGETS, "orion"), required=True)
    command.add_argument("--num-workers", type=int, default=4)
    command.set_defaults(func=cmd_internal_score)
    command = sub.add_parser("score", parents=[common])
    command.add_argument("--apply", action="store_true")
    command.add_argument("--arm", choices=ARM_NAMES)
    command.add_argument("--seed", type=int, choices=NEW_SEEDS)
    command.add_argument("--target", choices=("primary", *MET_TARGETS, "orion"))
    command.add_argument("--num-workers", type=int, default=4)
    command.set_defaults(func=cmd_score)
    command = sub.add_parser("seal-inference", parents=[common])
    command.add_argument("--deep", action="store_true")
    command.set_defaults(func=cmd_seal_inference)
    command = sub.add_parser("report", parents=[common])
    command.add_argument("--apply", action="store_true")
    command.set_defaults(
        func=cmd_report, n_bootstrap=N_BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED
    )
    command = sub.add_parser("verify", parents=[common])
    command.add_argument("--deep", action="store_true")
    command.set_defaults(func=cmd_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
