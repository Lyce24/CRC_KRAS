#!/usr/bin/env python3
"""Governed source-anchored metastatic adaptation and local-bound campaign.

This append-only campaign is deliberately separate from the Aim-1 model and
from every zero-shot external score namespace.  It binds the five frozen
TCGA+SurGen-primary UNI-v1 p75 refits, exports label-blind RIH-M and SurGen-M
native logits plus 512-dimensional slide embeddings, and seals those bytes
before target labels can be opened.

After the seal, every result is explicitly target-internal and is therefore
*not* external validation.  Three prespecified analyses are supported:

* exactly two metastatic-only few-shot methods at 2, 4, and 8 patients/class:
  a source-anchored residual linear probe and a pure ridge linear probe;
* full-label, honestly cross-fitted residual adaptation (a sample-bounded
  empirical adapter bound), with a two-parameter Platt comparator;
* a sample/recipe-bounded empirical target-internal local-MIL OOF bound trained
  from scratch on the pooled metastatic population.

The source encoder, attention module, classifier head, refit checkpoints, and
source manifests are read-only.  CPTAC, Orion, RIH-primary, and every primary
support pool are forbidden.  The local-MIL arm is a target-internal comparator,
not a source-anchored adaptation arm.

Production order::

    plan
    prepare --apply
    preflight --apply
    score-source --apply --max-workers 6
    open-targets --apply
    adapt --apply
    train-local --apply --max-workers 6
    analyze --apply
    verify

No command writes into the source/main-model root.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import hashlib
import io
import json
import math
import os
import platform
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
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import aim2_head_adaptation_base as e2c_stats  # noqa: E402
import aim2_confirmatory_transfer as embedding_runner  # noqa: E402
import aim2_metastatic_indomain_bound as legacy_local  # noqa: E402
import aim2_v3_fulllabel_residual_adaptation as adapter  # noqa: E402

SCHEMA_VERSION = 1
CAMPAIGN = "aim2_tcga_surgen_source_anchored_met_adaptation_univ1_5seed"
MODEL_SEEDS = (42, 43, 44, 45, 46)
FOLDS = (0, 1, 2, 3, 4)
OUTER_LAYOUT_SEEDS = (20260817, 20260818, 20260819, 20260820, 20260821)
PRIMARY_OUTER_SEED = 20260821
SUPPORT_PER_CLASS = (2, 4, 8)
SUPPORT_TOTAL = tuple(2 * value for value in SUPPORT_PER_CLASS)
SUPPORT_DRAWS_PER_LAYOUT = 20
SUPPORT_PROCEDURES_PER_CELL = len(OUTER_LAYOUT_SEEDS) * SUPPORT_DRAWS_PER_LAYOUT
SUPPORT_REGIMES = ("RIH_ONLY", "SURGEN_ONLY", "COMBINED")
LAMBDA_GRID = adapter.LAMBDA_GRID
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260817
MAX_WORKERS = 6
DEFAULT_NUM_WORKERS = 4
CAP = 8192
EMBED_DIM = 512

INTERNAL_ROLE = "TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION"
SOURCE_ANCHORED_ROLE = (
    "TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION; frozen TCGA+SurGen-primary "
    "UNI-v1 source anchor"
)
LOCAL_ROLE = (
    "TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION; local metastatic MIL comparator "
    "trained from scratch, not source-anchored"
)

SOURCE_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim1_tcga_surgen_two_encoder_v1_20260824"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_tcga_surgen_source_anchored_met_adaptation_univ1_5seed_v1_20260828"
)
RERUNS_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns")
SOURCE_CONTRACT_SHA256 = "0aa475fa30123b5f2e44c6d3b54de5ade5ebcadeb5a2aa08224558bdc41509d9"
SOURCE_RECOVERY_SHA256 = "fa41798f4266ebcb6bfde7ccae7ad4990e8de965c2cdacca4be1276989b43d08"
SOURCE_MANIFEST_SHA256 = "d7087a23a84a294670080eb5090f83612f57b3376c7696ed2d0094fb9bcff8a5"
SOURCE_SPLITS_SHA256 = "31046858b74a9e435ce7ceac263630e2966ab589c36a8693dcb0fc1060cfa9f4"
SOURCE_INFERENCE_SEAL_SHA256 = "d7a435ddae080fd8fed814a1d9e659844a83cf140565ed1e57972a3be4b24f45"
SOURCE_ANALYSIS_COMPLETION_SHA256 = "639696aaa4bdfafa887fc67cc541b976d6c3bf6113fc3909e2f133823df1b125"
EXPECTED_NATIVE = {
    "RIH": {
        "auroc": 0.6255630630630631,
        "ci95": [0.49774774774774777, 0.7483108108108107],
    },
    "SurGen": {
        "auroc": 0.5863636363636364,
        "ci95": [0.45, 0.7242424242424242],
    },
}

SOURCE_CHECKPOINT_SHA256 = {
    42: "8471a88630ca70145b476d8c07537317390f0e511609fed05b3962e27ed3613d",
    43: "1e38d78301a56460590591888072e3a65e5041a46700352e943488969da65063",
    44: "abcc02304dd56386e2556ca638e4fe8a6ba3750f4835dcd92323cd66b36caa23",
    45: "eacd57af5bbba3d91af30929b0f5e97d1b646e09ae15f8da061c83f406a9e18a",
    46: "4945084c09d9a6bebcf66d2527832e531578a8562d5790755c31c3fefb4a6bdb",
}
SOURCE_JOB_RECEIPT_SHA256 = {
    42: "01144a33811406c6e73bda0d3fff7df18ac68dcb7ca4557a6bd9dc641ca93887",
    43: "79b75870a950805eb22919a35fdd4e36afa4589feb5c9d7c2e14c86e34fa14c1",
    44: "d5a75830f56ef563f6c7d951da7d9e659572b3a4701b6c9e22bdee18d6c28cd6",
    45: "2b6d532a104683f73406810d83b754019ddb628c761ba2c0dea0a50d7d8498a9",
    46: "580cca1e2a3af4e7ff34d2be27e2887019666bbe9b38335961fb4463aa8d3361",
}
SOURCE_OOF_CONFIG_SHA256 = {
    42: "a4577d5a7fe6d619eef8ad717421e39698bff5e6fc53c613cdacac78d52772c0",
    43: "0bcb3a62d87f1bc353dcf88b5d75936eb3c34d1fd5c918135288a0f5032f2868",
    44: "545068d108191a020faae4d88fe328ad5cd9685e4eced4631db75b5f57faa70a",
    45: "f161e911c9564ab3cde3abe415590fe0a88158399a3a7518b14718e3a042d84e",
    46: "0b2549e19664175d90a97c904ab6bfbbf887c8fd41641f8a914ccefef92812b7",
}
PACK_ARTIFACTS = {
    "features.bin": ("6765d9faf30f40e212075c1a650fd34bc5ef625b3fb8e169dbc14e78c2058bd4", 49_735_520_256),
    "coords.bin": ("1a90bd95a430e433cff900c0b6b2a4f6fedcae550ae80584ee27ed7f100fe491", 194_279_376),
    "index.parquet": ("705464267e35e454e50ea8510bc937e9f8d6a00b8b97bb9cf1e06da5fcfa9556", 66_998),
    "meta.json": ("44f1f0c80f4b8740c5fc82caafa96a8ead372c561fe53cda896d83972ff3c80b", 367),
}


class ContractError(RuntimeError):
    """Fail-closed campaign contract violation."""


@dataclass(frozen=True)
class TargetSpec:
    key: str
    cohort: str
    blind_source: Path
    blind_sha256: str
    outcome_source: Path
    outcome_sha256: str
    prior_score_key: str
    slides: int
    patients: int
    mutant: int
    source_family_exposed: bool


TARGETS: dict[str, TargetSpec] = {
    item.key: item
    for item in (
        TargetSpec(
            "rih_metastatic",
            "RIH",
            SOURCE_ROOT / "downstream/inputs/label_blind/rih_metastatic.csv",
            "dec443d99e437e7cb3fc3297a4f98302081ba4455386e3b7d930ff142feb1e5c",
            Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_rih_metastatic.csv"),
            "6aaab722a96c2374296811f7bd4c79f44d842efe75788240274b14f79c3d1303",
            "rih_metastatic",
            85,
            85,
            37,
            False,
        ),
        TargetSpec(
            "surgen_metastatic",
            "SurGen",
            SOURCE_ROOT / "downstream/inputs/label_blind/sr1482_metastatic.csv",
            "90eb00ef0baf6a33fc4b423affbce33dde1d38068add59cc136f870f13cf6d15",
            Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_surgen_metastatic.csv"),
            "3e69383c2869b1432fd6a1326c79fd5a6d50ec0c0dee88029c8dd433c9e06135",
            "sr1482_metastatic",
            100,
            74,
            30,
            True,
        ),
    )
}
TARGET_ORDER = tuple(TARGETS)
FORBIDDEN_SUPPORT_ROSTERS = (
    "CPTAC",
    "Orion",
    "RIH-primary",
    "SurGen-primary",
    "TCGA",
)
FORBIDDEN_BLIND_COLUMNS = frozenset(
    {
        "label",
        "target_label",
        "kras",
        "kras_status",
        "kras_mutant",
        "outcome",
        "ras",
        "nras",
        "braf",
        "msi",
        "msi_dmmr",
    }
)
EXACT_BLIND_COLUMNS = (
    "slide_id",
    "patient_id",
    "specimen_role",
    "subcohort",
    "liver_class",
)


@dataclass(frozen=True, order=True)
class ScoreJob:
    target: str
    seed: int

    def __post_init__(self) -> None:
        if self.target not in TARGETS or self.seed not in MODEL_SEEDS:
            raise ValueError((self.target, self.seed))

    @property
    def key(self) -> str:
        return f"{self.target}__seed{self.seed}"


@dataclass(frozen=True, order=True)
class LocalJob:
    seed: int
    fold: int

    def __post_init__(self) -> None:
        if self.seed not in MODEL_SEEDS or self.fold not in FOLDS:
            raise ValueError((self.seed, self.fold))

    @property
    def key(self) -> str:
        return f"seed{self.seed}__fold{self.fold}"


@dataclass(frozen=True, order=True)
class AdaptJob:
    kind: str
    layout_seed: int
    support_regime: str | None = None
    cohort: str | None = None

    def __post_init__(self) -> None:
        if self.layout_seed not in OUTER_LAYOUT_SEEDS:
            raise ValueError(self.layout_seed)
        if self.kind == "few":
            if self.support_regime not in SUPPORT_REGIMES or self.cohort is not None:
                raise ValueError(self)
        elif self.kind == "full":
            if self.cohort not in {"RIH", "SurGen"} or self.support_regime is not None:
                raise ValueError(self)
        else:
            raise ValueError(self.kind)

    @property
    def key(self) -> str:
        suffix = self.support_regime if self.kind == "few" else self.cohort
        return f"{self.kind}__layout{self.layout_seed}__{suffix}"


def score_jobs() -> list[ScoreJob]:
    return [ScoreJob(target, seed) for target in TARGET_ORDER for seed in MODEL_SEEDS]


def local_jobs() -> list[LocalJob]:
    return [LocalJob(seed, fold) for seed in MODEL_SEEDS for fold in FOLDS]


def adapt_jobs() -> list[AdaptJob]:
    return [
        *(AdaptJob("few", layout, support_regime=regime)
          for layout in OUTER_LAYOUT_SEEDS for regime in SUPPORT_REGIMES),
        *(AdaptJob("full", layout, cohort=cohort)
          for layout in OUTER_LAYOUT_SEEDS for cohort in ("RIH", "SurGen")),
    ]


def fit_accounting() -> dict[str, int]:
    return {
        "source_main_model_fits": 0,
        "label_blind_embedding_jobs": 10,
        "few_shot_final_residual_decisions": (
            len(SUPPORT_PER_CLASS)
            * len(OUTER_LAYOUT_SEEDS)
            * SUPPORT_DRAWS_PER_LAYOUT
            * len(SUPPORT_REGIMES)
            * len(FOLDS)
            * len(MODEL_SEEDS)
        ),
        "few_shot_final_pure_ridge_decisions": (
            len(SUPPORT_PER_CLASS)
            * len(OUTER_LAYOUT_SEEDS)
            * SUPPORT_DRAWS_PER_LAYOUT
            * len(SUPPORT_REGIMES)
            * len(FOLDS)
            * len(MODEL_SEEDS)
        ),
        "few_shot_solver_calls_per_method": 3_382_500,
        "few_shot_solver_calls_both_methods": 6_765_000,
        "few_shot_unique_support_procedures_by_fold": 4_500,
        "few_shot_final_head_decisions_both_methods": 45_000,
        "few_shot_fit_to_test_cohort_applications": 90_000,
        "full_label_final_residual_decisions": (
            len(OUTER_LAYOUT_SEEDS) * len(TARGETS) * len(FOLDS) * len(MODEL_SEEDS)
        ),
        "full_label_platt_fits": len(OUTER_LAYOUT_SEEDS) * len(TARGETS) * len(FOLDS),
        "local_mil_oof_fits": len(local_jobs()),
        "local_mil_refits": 0,
    }


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
    raw = Path(path)
    lexical = raw.absolute()
    cursor = lexical
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise ContractError(f"artifact path traverses a symlink: {cursor}")
        cursor = cursor.parent
    resolved = raw.resolve(strict=True)
    if raw.is_symlink() or not resolved.is_file():
        raise ContractError(f"required regular artifact missing or symlinked: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _pinned_artifact(path: Path, sha256: str, size_bytes: int | None = None) -> dict[str, Any]:
    observed = _artifact(path)
    if observed["sha256"] != sha256 or (
        size_bytes is not None and observed["size_bytes"] != size_bytes
    ):
        raise ContractError(f"pinned artifact drifted: {path}")
    return observed


def _read_json(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ContractError(f"non-finite JSON constant {value}: {path}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
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
    payload = json.dumps(
        adapter.json_ready(value), indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"
    _write_bytes_once(path, payload.encode("utf-8"))


def _write_jsonl_once(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(
        (
            json.dumps(adapter.json_ready(dict(row)), sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    _write_bytes_once(path, payload)


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    _write_bytes_once(path, buffer.getvalue())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"invalid JSONL artifact: {path}") from exc
    for index, line in enumerate(lines, start=1):
        if not line:
            raise ContractError(f"blank JSONL record at {path}:{index}")
        try:
            value = json.loads(
                line,
                object_pairs_hook=lambda pairs: _strict_pairs(pairs, path=path),
                parse_constant=lambda token, line_index=index: (_ for _ in ()).throw(
                    ContractError(f"non-finite JSON constant {token}: {path}:{line_index}")
                ),
            )
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSONL at {path}:{index}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"JSONL record is not an object at {path}:{index}")
        rows.append(value)
    return rows


def _strict_pairs(pairs: list[tuple[str, Any]], *, path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}: {path}")
        result[key] = value
    return result


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
            raise ContractError(f"output-root path traverses symlink: {cursor}")
        cursor = cursor.parent
    source = SOURCE_ROOT.resolve(strict=False)
    if resolved == source or _is_relative_to(resolved, source) or _is_relative_to(source, resolved):
        raise ContractError("adaptation output must never overlap the source/main-model root")
    production = DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    temporary = Path("/tmp").resolve()
    if resolved != production and not _is_relative_to(resolved, temporary):
        raise ContractError(f"production root must be exactly {production}; tests may use /tmp")
    if resolved == production and not _is_relative_to(resolved, RERUNS_ROOT.resolve()):
        raise ContractError("production root escaped governed reruns")
    if must_exist is True and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if must_exist is False and (resolved.exists() or resolved.is_symlink()):
        raise FileExistsError(resolved)
    return resolved


def contract_path(root: Path) -> Path:
    return root / "contract.json"


def blind_path(root: Path, target: str) -> Path:
    return root / f"inputs/label_blind/{target}.csv"


def score_path(root: Path, job: ScoreJob) -> Path:
    return root / f"source_inference/embeddings/{job.target}/seed{job.seed}.parquet"


def score_receipt_path(root: Path, job: ScoreJob) -> Path:
    return score_path(root, job).with_suffix(".receipt.json")


def source_scheduler_path(root: Path) -> Path:
    return root / "receipts/source_embedding_scheduler.json"


def inference_seal_path(root: Path) -> Path:
    return root / "source_inference/inference_seal.json"


def target_open_path(root: Path) -> Path:
    return root / "receipts/target_internal_open.json"


def labeled_manifest_path(root: Path) -> Path:
    return root / "inputs/target_internal/labeled_metastatic.csv"


def split_dir(root: Path) -> Path:
    return root / "inputs/splits/aim1_e2e_met/aim1_balanced5"


def adapter_root(root: Path) -> Path:
    return root / "adaptation"


def adapt_shard_dir(root: Path, job: AdaptJob) -> Path:
    return adapter_root(root) / f"shards/{job.key}"


def adapt_shard_receipt_path(root: Path, job: AdaptJob) -> Path:
    return adapt_shard_dir(root, job) / "completion.json"


def adapt_scheduler_path(root: Path) -> Path:
    return root / "receipts/adaptation_scheduler.json"


def local_run_dir(root: Path, job: LocalJob) -> Path:
    return root / f"local_mil/jobs/seed{job.seed}/fold{job.fold}"


def local_receipt_path(root: Path, job: LocalJob) -> Path:
    return root / f"receipts/local_jobs/seed{job.seed}_fold{job.fold}.json"


def local_scheduler_path(root: Path) -> Path:
    return root / "receipts/local_mil_scheduler.json"


def local_pack_gate_path(root: Path, phase: str) -> Path:
    if phase not in {"pre", "post"}:
        raise ValueError(phase)
    return root / f"receipts/local_mil_pack_{phase}.json"


def local_oof_path(root: Path, seed: int) -> Path:
    return root / f"local_mil/oof/seed{seed}.parquet"


def analysis_root(root: Path) -> Path:
    return root / "analysis"


def source_checkpoint(seed: int) -> Path:
    return SOURCE_ROOT / f"train/e0/univ1/seed{seed}/final/refit/model.ckpt"


def source_job_receipt(seed: int) -> Path:
    return SOURCE_ROOT / f"receipts/jobs/univ1_refit/seed{seed}.json"


def source_oof_config(seed: int) -> Path:
    return SOURCE_ROOT / f"train/e0/univ1/seed{seed}/config.yaml"


def prior_score_path(job: ScoreJob) -> Path:
    return (
        SOURCE_ROOT
        / "downstream_v2/continuation_v3/scores/univ1"
        / TARGETS[job.target].prior_score_key
        / f"seed{job.seed}.parquet"
    )


def _implementation_sources() -> dict[str, Any]:
    paths = {
        "controller": Path(__file__).resolve(),
        "controller_test": REPO
        / "tests/test_aim2_tcga_surgen_source_anchored_met_adaptation_campaign.py",
        "embedding_runner": REPO / "aim2_confirmatory_transfer.py",
        "few_shot_statistics": REPO / "aim2_head_adaptation_base.py",
        "residual_adapter": REPO / "aim2_v3_fulllabel_residual_adaptation.py",
        "local_reference": REPO / "aim2_metastatic_indomain_bound.py",
        "training_workflow": REPO / "src/oceanpath/workflows/training.py",
        "splitting_core": REPO / "src/oceanpath/splitting/core.py",
        "mil_datamodule": REPO / "src/oceanpath/datasets/datamodule.py",
        "packed_dataset": REPO / "src/oceanpath/datasets/packed.py",
        "train_config": REPO / "configs/train.yaml",
        "aim1_data_config": REPO / "configs/data/aim1.yaml",
        "aim1_splits_config": REPO / "configs/splits/aim1_balanced.yaml",
        "aim1_training_config": REPO / "configs/training/aim1.yaml",
        "default_training_config": REPO / "configs/training/default.yaml",
        "abmil_config": REPO / "configs/model/abmil.yaml",
        "univ1_config": REPO / "configs/encoder/univ1.yaml",
        "platform_config": REPO / "configs/platform/colon_workstation.yaml",
    }
    return {name: _artifact(path) for name, path in paths.items()}


def _pack_stat_snapshot(pack: Mapping[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name in PACK_ARTIFACTS:
        path = Path(str(pack["artifacts"][name]["path"]))
        stat = path.stat(follow_symlinks=False)
        snapshot[name] = {
            "path": str(path.resolve(strict=True)),
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return snapshot


def _validate_source_manifest(path: Path) -> dict[str, Any]:
    identity = _pinned_artifact(path, SOURCE_MANIFEST_SHA256, 440_060)
    frame = pd.read_csv(path, low_memory=False)
    if (
        len(frame) != 1_389
        or frame["patient_id"].astype(str).nunique() != 1_239
        or set(frame["cohort"].astype(str)) != {"TCGA", "SurGen"}
        or set(frame["subcohort"].astype(str))
        != {"TCGA-COAD", "TCGA-READ", "SR386", "SR1482"}
        or set(frame["specimen_role"].astype(str).str.casefold()) != {"primary"}
    ):
        raise ContractError("source development manifest is not TCGA+SurGen primaries only")
    patients = frame.sort_values("slide_id").drop_duplicates("patient_id")
    if int(patients["target_label"].sum()) != 501:
        raise ContractError("source patient KRAS census drifted")
    return {**identity, "slides": 1_389, "patients": 1_239, "mutant_patients": 501}


def validate_source(*, deep_pack: bool) -> dict[str, Any]:
    contract_file = SOURCE_ROOT / "contract.json"
    recovery_file = SOURCE_ROOT / "recovery_v2/receipts/training_complete_scoped.json"
    contract_identity = _pinned_artifact(contract_file, SOURCE_CONTRACT_SHA256, 69_114)
    recovery_identity = _pinned_artifact(recovery_file, SOURCE_RECOVERY_SHA256, 6_267)
    contract = _read_json(contract_file)
    recovery = _read_json(recovery_file)
    if (
        contract.get("campaign") != "aim1_tcga_surgen_two_encoder_5seed"
        or contract.get("source_population") != "TCGA + SurGen primary (SR386 + SR1482)"
        or contract.get("seeds") != list(MODEL_SEEDS)
        or recovery.get("status") != "complete_and_certified_via_scoped_census_erratum"
        or recovery.get("fit_accounting", {}).get("hidden_fits") != 0
    ):
        raise ContractError("source campaign/recovery semantics drifted")
    manifest = _validate_source_manifest(SOURCE_ROOT / "inputs/manifests/tcga_surgen_primary.csv")
    splits = _pinned_artifact(
        SOURCE_ROOT / "inputs/splits/aim1_primary_tcga_surgen/aim1_balanced5/splits.parquet",
        SOURCE_SPLITS_SHA256,
        114_658,
    )
    checkpoints: dict[str, Any] = {}
    for seed in MODEL_SEEDS:
        receipt_path = source_job_receipt(seed)
        receipt_identity = _pinned_artifact(
            receipt_path, SOURCE_JOB_RECEIPT_SHA256[seed]
        )
        receipt = _read_json(receipt_path)
        checkpoint = _pinned_artifact(
            source_checkpoint(seed), SOURCE_CHECKPOINT_SHA256[seed], 11_052_075
        )
        oof_config = _pinned_artifact(
            source_oof_config(seed), SOURCE_OOF_CONFIG_SHA256[seed], 5_345
        )
        if (
            receipt.get("status") != "completed"
            or receipt.get("campaign") != "aim1_tcga_surgen_two_encoder_5seed"
            or receipt.get("encoder") != "UNI-v1"
            or receipt.get("kind") != "univ1_refit"
            or receipt.get("seed") != seed
            or receipt.get("oof_fits") != 0
            or receipt.get("refits") != 1
            or receipt.get("artifacts", {}).get("refit", {}).get("checkpoint") != checkpoint
            or receipt.get("artifacts", {}).get("source_config") != oof_config
        ):
            raise ContractError(f"seed{seed}: source p75 refit receipt drifted")
        checkpoints[str(seed)] = {
            "checkpoint": checkpoint,
            "job_receipt": receipt_identity,
            "refit_info": receipt["artifacts"]["refit"]["info"],
            "source_oof_resolved_config": oof_config,
        }
    pack = contract.get("feature_stores", {}).get("univ1")
    if (
        not isinstance(pack, Mapping)
        or pack.get("encoder") != "UNI-v1"
        or pack.get("feature_dim") != 1024
    ):
        raise ContractError("source contract lacks the governed UNI-v1 pack")
    for name, (sha256, size_bytes) in PACK_ARTIFACTS.items():
        record = pack.get("artifacts", {}).get(name)
        path = Path(str(record.get("path", ""))) if isinstance(record, Mapping) else Path("")
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != sha256
            or record.get("size_bytes") != size_bytes
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != size_bytes
        ):
            raise ContractError(f"source UNI-v1 pack {name} drifted")
        if (deep_pack or name in {"index.parquet", "meta.json"}) and _sha256(path) != sha256:
            raise ContractError(f"source UNI-v1 pack {name} hash drifted")
    inference = _pinned_artifact(
        SOURCE_ROOT / "downstream_v2/continuation_v3/inference/inference_seal.json",
        SOURCE_INFERENCE_SEAL_SHA256,
        39_086,
    )
    inference_value = _read_json(Path(inference["path"]))
    if (
        inference_value.get("status") != "sealed_before_outcome_join"
        or inference_value.get("encoders") != ["univ1", "virchow2_cls"]
        or inference_value.get("seeds") != list(MODEL_SEEDS)
        or inference_value.get("score_artifact_count") != 50
        or inference_value.get("score_slide_rows") != 4_790
        or inference_value.get("target_outcome_files_opened") is not False
        or inference_value.get("target_outcomes_present") is not False
        or set(inference_value.get("targets", []))
        != {
            "cptac_primary",
            "rih_primary",
            "orion_cpht",
            "rih_metastatic",
            "sr1482_metastatic",
        }
    ):
        raise ContractError("pinned source-only inference seal semantics drifted")
    return {
        "root": str(SOURCE_ROOT.resolve()),
        "contract": contract_identity,
        "training_recovery": recovery_identity,
        "source_manifest": manifest,
        "source_splits": splits,
        "checkpoints": checkpoints,
        "feature_store": dict(pack),
        "prior_label_blind_inference_seal": inference,
        "development_population": "TCGA+SurGen primaries only",
        "encoder": "UNI-v1",
        "model_seeds": list(MODEL_SEEDS),
    }


def _validate_blind(frame: pd.DataFrame, spec: TargetSpec, *, context: str) -> pd.DataFrame:
    if list(frame.columns) != list(EXACT_BLIND_COLUMNS):
        raise ContractError(f"{context}: blind roster header is not the exact allowlist")
    if FORBIDDEN_BLIND_COLUMNS & set(frame):
        raise ContractError(f"{context}: target outcome leaked into blind roster")
    if {"slide_id", "patient_id"} - set(frame):
        raise ContractError(f"{context}: blind identity schema incomplete")
    out = frame.copy()
    if out[["slide_id", "patient_id"]].isna().any().any():
        raise ContractError(f"{context}: null blind identity")
    if not out["slide_id"].map(lambda value: isinstance(value, str)).all() or not out[
        "patient_id"
    ].map(lambda value: isinstance(value, str)).all():
        raise ContractError(f"{context}: blind identifiers must be strings before coercion")
    out["slide_id"] = out["slide_id"].astype(str)
    out["patient_id"] = out["patient_id"].astype(str)
    if (
        out["slide_id"].str.strip().eq("").any()
        or out["patient_id"].str.strip().eq("").any()
        or out["slide_id"].duplicated().any()
        or len(out) != spec.slides
        or out["patient_id"].nunique() != spec.patients
        or set(out["specimen_role"].astype(str)) != {"metastatic"}
        or set(out["subcohort"].astype(str))
        != ({"RIH-Colon"} if spec.key == "rih_metastatic" else {"SR1482"})
        or set(out["liver_class"].astype(str)) != {"liver", "non_liver"}
    ):
        raise ContractError(f"{context}: blind roster census/identity drifted")
    return out.sort_values("slide_id", kind="mergesort").reset_index(drop=True)


def _contract_payload(
    root: Path, source: Mapping[str, Any], *, created_utc: str
) -> dict[str, Any]:
    blind = {}
    all_slides: set[str] = set()
    all_patients: set[str] = set()
    for target, spec in TARGETS.items():
        frame = _validate_blind(pd.read_csv(blind_path(root, target)), spec, context=target)
        slides = set(frame["slide_id"])
        patients = set(frame["patient_id"])
        if all_slides & slides or all_patients & patients:
            raise ContractError("target blind rosters overlap")
        all_slides |= slides
        all_patients |= patients
        blind[target] = {
            "artifact": _artifact(blind_path(root, target)),
            "slides": spec.slides,
            "patients": spec.patients,
            "outcome_source_not_opened": {
                "path": str(spec.outcome_source),
                "expected_sha256": spec.outcome_sha256,
            },
            "source_family_exposed": spec.source_family_exposed,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "PREPARED_LABEL_BLIND_SOURCE_ANCHORED",
        "created_utc": created_utc,
        "output_root": str(root),
        "source": dict(source),
        "targets": blind,
        "target_census": {"slides": 185, "patients": 159, "mutant_after_open": 67},
        "protocol": {
            "source_model_seeds": list(MODEL_SEEDS),
            "outer_layout_seeds": list(OUTER_LAYOUT_SEEDS),
            "outer_folds": list(FOLDS),
            "few_shot_support_per_class": list(SUPPORT_PER_CLASS),
            "few_shot_draws_per_layout": SUPPORT_DRAWS_PER_LAYOUT,
            "few_shot_procedures_per_cell": SUPPORT_PROCEDURES_PER_CELL,
            "lambda_grid": [adapter.lambda_key(value) for value in LAMBDA_GRID],
            "few_shot_methods": {
                "source_anchored_residual_linear_probe": {
                    "formula": "eta_source + H @ delta_w + delta_b",
                    "anchor": "frozen source native logit",
                },
                "pure_ridge_linear_probe": {
                    "formula": "H @ w + b",
                    "anchor": "zero logit; infinity is no-information",
                },
            },
            "few_shot_representation": (
                "raw frozen 512-dimensional patient-mean MIL embeddings; "
                "no target-wide scaling or preprocessing"
            ),
            "few_shot_objective": (
                "mean binary logistic loss + lambda/2*(||w||^2 + b^2); "
                "bias is penalized"
            ),
            "lambda_selected_separately_by_method": True,
            "lambda_selection_data": "support patients only; identical LOO split layout",
            "support_regimes": list(SUPPORT_REGIMES),
            "test_cohorts_for_every_fitted_head": ["RIH-M", "SurGen-M"],
            "combined_support_balance": (
                "k patients/class total, exactly k/2 from each cohort within each class"
            ),
            "one_head_reused_for_both_test_cohorts": True,
            "bootstrap_draws": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "maximum_parallel_workers": MAX_WORKERS,
            "few_shot_unique_supports": 4_500,
            "few_shot_final_head_decisions_per_method": 22_500,
            "few_shot_fit_to_test_cohort_applications": 90_000,
            "few_shot_inner_plus_final_solver_calls": 6_765_000,
        },
        "fit_accounting": fit_accounting(),
        "firewall": {
            "result_role": INTERNAL_ROLE,
            "target_labels_open_only_after_source_inference_seal": True,
            "external_validation_claim_permitted": False,
            "primary_support_permitted": False,
            "forbidden_support_rosters": list(FORBIDDEN_SUPPORT_ROSTERS),
            "allowed_target_label_rosters": ["RIH-M", "SurGen-M"],
            "source_or_main_model_mutation": False,
            "source_checkpoint_selection_from_target": False,
            "support_budget_selection_from_target_test": False,
            "all_prespecified_support_budgets_reported": True,
            "surgen_metastatic_source_family_exposed": True,
        },
        "implementation": _implementation_sources(),
    }


def prepare(root: Path, *, apply: bool) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=False if apply else None)
    source = validate_source(deep_pack=True)
    plan = {
        "status": "DRY_RUN_READY" if not apply else "PREPARING",
        "output_root": str(output),
        "source": "authenticated TCGA+SurGen-primary UNI-v1 p75 refits",
        "blind_embedding_jobs": len(score_jobs()),
        "target_outcomes_opened": False,
        "fit_accounting": fit_accounting(),
    }
    if not apply:
        return plan
    # Validate every external byte and schema before exclusively creating the
    # production namespace; a bad second roster must not strand a partial v1.
    blind_payloads: dict[str, bytes] = {}
    validated_frames: dict[str, pd.DataFrame] = {}
    for target, spec in TARGETS.items():
        _pinned_artifact(spec.blind_source, spec.blind_sha256)
        payload = spec.blind_source.read_bytes()
        validated_frames[target] = _validate_blind(
            pd.read_csv(io.BytesIO(payload), low_memory=False), spec, context=f"{target}/source"
        )
        blind_payloads[target] = payload
    slide_sets = {
        target: set(frame["slide_id"].astype(str)) for target, frame in validated_frames.items()
    }
    patient_sets = {
        target: set(frame["patient_id"].astype(str)) for target, frame in validated_frames.items()
    }
    if (
        slide_sets[TARGET_ORDER[0]] & slide_sets[TARGET_ORDER[1]]
        or patient_sets[TARGET_ORDER[0]] & patient_sets[TARGET_ORDER[1]]
    ):
        raise ContractError("blind metastatic source rosters overlap")
    pack = source["feature_store"]
    index = pd.read_parquet(Path(pack["artifacts"]["index.parquet"]["path"]))
    id_column = "slide_id" if "slide_id" in index else "key"
    wanted = set().union(*slide_sets.values())
    selected = index[index[id_column].astype(str).isin(wanted)]
    if (
        index[id_column].astype(str).duplicated().any()
        or len(wanted) != 185
        or len(selected) != 185
        or set(selected[id_column].astype(str)) != wanted
    ):
        raise ContractError("source pack lacks exact blind target coverage before prepare")
    output.mkdir(parents=True, exist_ok=False)
    for target in TARGET_ORDER:
        _write_bytes_once(blind_path(output, target), blind_payloads[target])
    payload = _contract_payload(output, source, created_utc=_utcnow())
    _write_json_once(contract_path(output), payload)
    return load_contract(output, deep_pack=True)


def load_contract(root: Path, *, deep_pack: bool) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    stored = _read_json(contract_path(output))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("campaign contract lacks created_utc")
    source = validate_source(deep_pack=deep_pack)
    expected = _contract_payload(output, source, created_utc=created)
    if stored != expected:
        raise ContractError("campaign contract does not replay exactly")
    return stored


def preflight(root: Path, *, apply: bool, deep_pack: bool = True) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    contract = load_contract(output, deep_pack=deep_pack)
    pack = contract["source"]["feature_store"]
    index = pd.read_parquet(Path(pack["artifacts"]["index.parquet"]["path"]))
    id_column = "slide_id" if "slide_id" in index else "key"
    wanted = set().union(
        *(
            set(pd.read_csv(blind_path(output, target))["slide_id"].astype(str))
            for target in TARGET_ORDER
        )
    )
    selected = index[index[id_column].astype(str).isin(wanted)]
    if (
        index[id_column].astype(str).duplicated().any()
        or len(wanted) != 185
        or len(selected) != 185
        or set(selected[id_column].astype(str)) != wanted
    ):
        raise ContractError("UNI-v1 pack does not cover exact metastatic blind roster")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY_FOR_LABEL_BLIND_SOURCE_INFERENCE",
        "created_utc": _utcnow(),
        "contract": _artifact(contract_path(output)),
        "authenticated_source_checkpoints": 5,
        "blind_embedding_jobs": 10,
        "blind_slides": 185,
        "target_outcomes_opened": False,
        "target_labels_present": False,
        "external_validation_claim_permitted": False,
        "source_main_model_writes": 0,
        "pack_stat_snapshot": _pack_stat_snapshot(contract["source"]["feature_store"]),
    }
    path = output / "receipts/deep_preflight.json"
    if apply:
        if path.exists():
            old = _read_json(path)
            payload["created_utc"] = old.get("created_utc")
            if old != payload:
                raise ContractError("persisted preflight drifted")
        else:
            _write_json_once(path, payload)
    return payload


def verify_preflight(root: Path, *, deep_pack: bool = False) -> dict[str, Any]:
    """Replay every preflight field while preserving only its creation time."""

    output = validate_output_root(root, must_exist=True)
    stored = _read_json(output / "receipts/deep_preflight.json")
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("deep preflight lacks created_utc")
    expected = preflight(output, apply=False, deep_pack=deep_pack)
    expected["created_utc"] = created
    if stored != expected:
        raise ContractError("deep preflight does not replay exactly")
    return stored


def _source_inference_environment(*, num_workers: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ContractError("source embedding inference requires CUDA bfloat16")
    return {
        "device": "cuda",
        "device_index": 0,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "autocast": True,
        "autocast_dtype": "bfloat16",
        "full_bag": True,
        "batch_size": 1,
        "num_workers_per_job": int(num_workers),
        "embedding_dimensions": EMBED_DIM,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "target_labels_present": False,
    }


def _prior_score_identity(job: ScoreJob) -> dict[str, Any]:
    seal_path = SOURCE_ROOT / "downstream_v2/continuation_v3/inference/inference_seal.json"
    _pinned_artifact(seal_path, SOURCE_INFERENCE_SEAL_SHA256, 39_086)
    seal = _read_json(seal_path)
    expected_id = (
        f"final_v11.continuation_v3.score.univ1."
        f"{TARGETS[job.target].prior_score_key}.seed{job.seed}"
    )
    matches = [
        value
        for value in seal.get("score_artifacts", [])
        if isinstance(value, Mapping) and value.get("job_id") == expected_id
    ]
    if len(matches) != 1:
        raise ContractError(f"{job.key}: prior native score is absent/nonunique in seal")
    record = dict(matches[0])
    expected_path = prior_score_path(job).resolve()
    score = record.get("score")
    receipt = record.get("receipt")
    if (
        record.get("rows") != TARGETS[job.target].slides
        or not isinstance(score, Mapping)
        or Path(str(score.get("path", ""))).resolve() != expected_path
        or dict(score) != _artifact(expected_path)
        or not isinstance(receipt, Mapping)
        or dict(receipt) != _artifact(Path(str(receipt.get("path", ""))))
    ):
        raise ContractError(f"{job.key}: prior native score identity drifted")
    return {"score": dict(score), "receipt": dict(receipt), "job_id": expected_id}


def _validate_embedding_frame(frame: pd.DataFrame, manifest: pd.DataFrame, job: ScoreJob) -> None:
    embedding_columns = [f"e{index}" for index in range(EMBED_DIM)]
    required = ["slide_id", "seed", "fold", "logit", *embedding_columns]
    if list(frame.columns) != required:
        raise ContractError(f"{job.key}: embedding score schema drifted")
    if (
        len(frame) != TARGETS[job.target].slides
        or set(frame["slide_id"].astype(str)) != set(manifest["slide_id"].astype(str))
        or frame["slide_id"].astype(str).duplicated().any()
        or set(pd.to_numeric(frame["seed"], errors="raise").astype(int)) != {job.seed}
        or set(pd.to_numeric(frame["fold"], errors="raise").astype(int)) != {0}
        or not np.isfinite(frame[["logit", *embedding_columns]].to_numpy(float)).all()
    ):
        raise ContractError(f"{job.key}: embedding score roster/value drifted")


def _bit_exact_native_reconciliation(frame: pd.DataFrame, job: ScoreJob) -> dict[str, Any]:
    prior_identity = _prior_score_identity(job)
    prior = pd.read_parquet(prior_score_path(job))
    columns = ["slide_id", "seed", "fold", "logit"]
    if list(prior.columns) != columns:
        raise ContractError(f"{job.key}: prior native score schema drifted")
    left = frame[columns].sort_values("slide_id", kind="mergesort").reset_index(drop=True)
    right = prior[columns].sort_values("slide_id", kind="mergesort").reset_index(drop=True)
    if not left[["slide_id", "seed", "fold"]].equals(right[["slide_id", "seed", "fold"]]):
        raise ContractError(f"{job.key}: native score identity columns disagree")
    left_bits = left["logit"].to_numpy(np.float64).view(np.uint64)
    right_bits = right["logit"].to_numpy(np.float64).view(np.uint64)
    if not np.array_equal(left_bits, right_bits):
        difference = float(
            np.max(np.abs(left["logit"].to_numpy(float) - right["logit"].to_numpy(float)))
        )
        raise ContractError(
            f"{job.key}: exported native logits are not bit-exact; max_abs={difference}"
        )
    return {
        "status": "BIT_EXACT_PASS",
        "rows": len(left),
        "prior": prior_identity,
        "float64_bit_mismatches": 0,
        "max_absolute_difference": 0.0,
    }


def _validate_cached_embedding(root: Path, job: ScoreJob) -> pd.DataFrame | None:
    path = score_path(root, job)
    receipt_file = score_receipt_path(root, job)
    if not path.exists() and not receipt_file.exists():
        return None
    if (
        not path.is_file()
        or path.is_symlink()
        or not receipt_file.is_file()
        or receipt_file.is_symlink()
    ):
        raise ContractError(f"{job.key}: partial/symlinked immutable embedding cache")
    contract = load_contract(root, deep_pack=False)
    manifest = pd.read_csv(blind_path(root, job.target), low_memory=False)
    frame = pd.read_parquet(path)
    _validate_embedding_frame(frame, manifest, job)
    reconciliation = _bit_exact_native_reconciliation(frame, job)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "LABEL_BLIND_SOURCE_EMBEDDING_COMPLETE",
        "target": job.target,
        "seed": job.seed,
        "contains_target_labels": False,
        "result_role": INTERNAL_ROLE,
        "contract": _artifact(contract_path(root)),
        "manifest": _artifact(blind_path(root, job.target)),
        "checkpoint": contract["source"]["checkpoints"][str(job.seed)]["checkpoint"],
        "feature_store": contract["source"]["feature_store"],
        "artifact": _artifact(path),
        "rows": TARGETS[job.target].slides,
        "embedding_dimensions": EMBED_DIM,
        "native_reconciliation": reconciliation,
    }
    receipt = _read_json(receipt_file)
    if set(receipt) != set(expected) | {"created_utc", "execution"} or {
        key: receipt.get(key) for key in expected
    } != expected:
        raise ContractError(f"{job.key}: embedding receipt drifted")
    event = receipt.get("execution")
    if (
        not isinstance(event, Mapping)
        or event.get("job_id") != f"source_embedding.{job.target}.seed{job.seed}"
        or event.get("returncode") != 0
        or event.get("target_labels_present") is not False
        or not isinstance(event.get("started_unix_ns"), int)
        or not isinstance(event.get("completed_unix_ns"), int)
        or event["completed_unix_ns"] <= event["started_unix_ns"]
    ):
        raise ContractError(f"{job.key}: embedding execution event drifted")
    return frame


def _score_one(root: Path, job: ScoreJob, *, num_workers: int) -> None:
    output = validate_output_root(root, must_exist=True)
    contract = load_contract(output, deep_pack=False)
    verify_preflight(output, deep_pack=False)
    if num_workers != DEFAULT_NUM_WORKERS:
        raise ContractError("governed source inference requires exactly --num-workers 4")
    if target_open_path(output).exists() or labeled_manifest_path(output).exists():
        raise ContractError("target outcomes opened before label-blind source inference completed")
    if inference_seal_path(output).exists():
        verify_inference_seal(output)
        return
    if _validate_cached_embedding(output, job) is not None:
        return
    manifest = _validate_blind(
        pd.read_csv(blind_path(output, job.target), low_memory=False),
        TARGETS[job.target],
        context=job.key,
    )
    checkpoint = Path(contract["source"]["checkpoints"][str(job.seed)]["checkpoint"]["path"])
    pack = contract["source"]["feature_store"]
    meta = _read_json(Path(pack["artifacts"]["meta.json"]["path"]))
    feature_dir = Path(str(meta.get("source_dir", ""))).resolve(strict=True)
    started_unix_ns = time.time_ns()
    started_utc = _utcnow_precise()
    frame = embedding_runner.score_checkpoints_with_embeddings(
        [(job.seed, checkpoint)],
        manifest,
        feature_dir=feature_dir,
        pack_dir=Path(pack["path"]),
        device="cuda",
        num_workers=num_workers,
        export_embeddings=True,
    )
    frame = frame[["slide_id", "seed", "fold", "logit", *(f"e{i}" for i in range(EMBED_DIM))]]
    frame = frame.sort_values("slide_id", kind="mergesort").reset_index(drop=True)
    _validate_embedding_frame(frame, manifest, job)
    reconciliation = _bit_exact_native_reconciliation(frame, job)
    completed_utc = _utcnow_precise()
    completed_unix_ns = time.time_ns()
    if completed_unix_ns <= started_unix_ns:
        raise ContractError(f"{job.key}: invalid embedding execution interval")
    destination = score_path(output, job)
    _write_parquet_once(destination, frame)
    _write_json_once(
        score_receipt_path(output, job),
        {
            "schema_version": SCHEMA_VERSION,
            "status": "LABEL_BLIND_SOURCE_EMBEDDING_COMPLETE",
            "created_utc": _utcnow(),
            "target": job.target,
            "seed": job.seed,
            "contains_target_labels": False,
            "result_role": INTERNAL_ROLE,
            "contract": _artifact(contract_path(output)),
            "manifest": _artifact(blind_path(output, job.target)),
            "checkpoint": contract["source"]["checkpoints"][str(job.seed)]["checkpoint"],
            "feature_store": pack,
            "artifact": _artifact(destination),
            "rows": TARGETS[job.target].slides,
            "embedding_dimensions": EMBED_DIM,
            "native_reconciliation": reconciliation,
            "execution": {
                "job_id": f"source_embedding.{job.target}.seed{job.seed}",
                "started_utc": started_utc,
                "completed_utc": completed_utc,
                "started_unix_ns": started_unix_ns,
                "completed_unix_ns": completed_unix_ns,
                "returncode": 0,
                "target_labels_present": False,
            },
        },
    )
    _validate_cached_embedding(output, job)


def _source_score_command(root: Path, job: ScoreJob, *, num_workers: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_score-one",
        "--output-root",
        str(root),
        "--target",
        job.target,
        "--seed",
        str(job.seed),
        "--num-workers",
        str(num_workers),
    ]


def _parallel_peak(events: Sequence[Mapping[str, Any]]) -> int:
    points: list[tuple[int, int]] = []
    for event in events:
        started = event.get("started_unix_ns")
        completed = event.get("completed_unix_ns")
        if (
            not isinstance(started, int)
            or not isinstance(completed, int)
            or started <= 0
            or completed <= started
        ):
            raise ContractError("scheduler event interval is invalid")
        points.extend(((started, 1), (completed, -1)))
    active = 0
    peak = 0
    for _timestamp, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        if active < 0:
            raise ContractError("scheduler intervals are malformed")
        peak = max(peak, active)
    if active:
        raise ContractError("scheduler intervals did not close")
    return peak


def _parse_utc(value: Any, *, context: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ContractError(f"{context}: UTC timestamp is not a string")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ContractError(f"{context}: invalid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ContractError(f"{context}: timestamp is not timezone-aware UTC")
    return parsed


SOURCE_SCHEDULER_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "created_utc",
        "contract",
        "environment",
        "configured_max_workers",
        "observed_peak_workers",
        "job_count",
        "score_rows",
        "target_outcomes_opened",
        "events",
    }
)
SCHEDULER_EVENT_KEYS = frozenset(
    {
        "job_key",
        "command",
        "started_utc",
        "completed_utc",
        "started_unix_ns",
        "completed_unix_ns",
        "returncode",
    }
)


def _validate_source_scheduler(root: Path) -> dict[str, Any]:
    value = _read_json(source_scheduler_path(root))
    events = value.get("events")
    if (
        set(value) != SOURCE_SCHEDULER_KEYS
        or value.get("schema_version") != SCHEMA_VERSION
        or
        value.get("status") != "COMPLETE_LABEL_BLIND_SOURCE_EMBEDDINGS"
        or not isinstance(value.get("created_utc"), str)
        or value.get("contract") != _artifact(contract_path(root))
        or value.get("environment")
        != _artifact(root / "source_inference/environment.json")
        or value.get("configured_max_workers") != MAX_WORKERS
        or value.get("observed_peak_workers") != MAX_WORKERS
        or value.get("job_count") != 10
        or value.get("score_rows") != 925
        or value.get("target_outcomes_opened") is not False
        or not isinstance(events, list)
        or len(events) != 10
        or _parallel_peak(events) != MAX_WORKERS
    ):
        raise ContractError("source embedding scheduler receipt drifted")
    _parse_utc(value["created_utc"], context="source scheduler created_utc")
    by_key: dict[str, Mapping[str, Any]] = {}
    for raw in events:
        if not isinstance(raw, Mapping) or set(raw) != SCHEDULER_EVENT_KEYS:
            raise ContractError("source scheduler event schema drifted")
        key = raw.get("job_key")
        if not isinstance(key, str) or key in by_key:
            raise ContractError("source scheduler job keys are missing/duplicated")
        by_key[key] = raw
    if set(by_key) != {job.key for job in score_jobs()}:
        raise ContractError("source scheduler job roster drifted")
    for job in score_jobs():
        event = by_key[job.key]
        if (
            event.get("command")
            != _source_score_command(root, job, num_workers=DEFAULT_NUM_WORKERS)
            or event.get("returncode") != 0
        ):
            raise ContractError(f"{job.key}: source scheduler command/result drifted")
        started = _parse_utc(event["started_utc"], context=f"{job.key}/started")
        completed = _parse_utc(event["completed_utc"], context=f"{job.key}/completed")
        if completed <= started:
            raise ContractError(f"{job.key}: source scheduler wall interval drifted")
        _validate_cached_embedding(root, job)
        child = _read_json(score_receipt_path(root, job))["execution"]
        if (
            event["started_unix_ns"] > child["started_unix_ns"]
            or child["completed_unix_ns"] > event["completed_unix_ns"]
            or _parse_utc(child["started_utc"], context=f"{job.key}/child-start") < started
            or _parse_utc(child["completed_utc"], context=f"{job.key}/child-complete")
            > completed
        ):
            raise ContractError(f"{job.key}: child inference escaped scheduler interval")
    return value


def _inference_seal_payload(root: Path, *, created_utc: str) -> dict[str, Any]:
    scheduler = _validate_source_scheduler(root)
    records = []
    for job in score_jobs():
        _validate_cached_embedding(root, job)
        records.append(
            {
                "job_key": job.key,
                "score": _artifact(score_path(root, job)),
                "receipt": _artifact(score_receipt_path(root, job)),
                "rows": TARGETS[job.target].slides,
                "native_reconciliation": "BIT_EXACT_PASS",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED_LABEL_BLIND_BEFORE_TARGET_OUTCOME_OPEN",
        "created_utc": created_utc,
        "contract": _artifact(contract_path(root)),
        "preflight": _artifact(root / "receipts/deep_preflight.json"),
        "scheduler": _artifact(source_scheduler_path(root)),
        "configured_max_workers": scheduler["configured_max_workers"],
        "observed_peak_workers": scheduler["observed_peak_workers"],
        "score_artifact_count": 10,
        "score_rows": 925,
        "embedding_dimensions": EMBED_DIM,
        "all_native_logits_bit_exact_to_continuation_v3": True,
        "target_outcome_files_opened": False,
        "target_labels_present": False,
        "score_artifacts": records,
    }


def verify_inference_seal(root: Path) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    stored = _read_json(inference_seal_path(output))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("source inference seal lacks created_utc")
    expected = _inference_seal_payload(output, created_utc=created)
    if stored != expected:
        raise ContractError("source inference seal does not replay")
    return stored


def score_source(
    root: Path,
    *,
    apply: bool,
    max_workers: int,
    num_workers: int,
) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    contract = load_contract(output, deep_pack=True)
    preflight_value = verify_preflight(output, deep_pack=False)
    initial_pack_stats = _pack_stat_snapshot(contract["source"]["feature_store"])
    if preflight_value.get("pack_stat_snapshot") != initial_pack_stats:
        raise ContractError("source pack stat identity changed after preflight")
    if max_workers != MAX_WORKERS:
        raise ContractError("governed source inference requires exactly --max-workers 6")
    if num_workers != DEFAULT_NUM_WORKERS:
        raise ContractError("governed source inference requires exactly --num-workers 4")
    if target_open_path(output).exists() or labeled_manifest_path(output).exists():
        raise ContractError("target labels opened before source inference")
    commands = [
        _source_score_command(output, job, num_workers=num_workers) for job in score_jobs()
    ]
    if not apply:
        return {
            "status": "DRY_RUN_LABEL_BLIND",
            "jobs": 10,
            "max_workers": MAX_WORKERS,
            "score_rows": 925,
            "target_outcomes_opened": False,
            "commands": commands,
        }
    environment = _source_inference_environment(num_workers=num_workers)
    environment_path = output / "source_inference/environment.json"
    if environment_path.exists():
        if _read_json(environment_path) != environment:
            raise ContractError("source inference environment drifted")
    else:
        _write_json_once(environment_path, environment)
    if inference_seal_path(output).exists():
        return {"status": "ALREADY_SEALED", "seal": verify_inference_seal(output)}
    if source_scheduler_path(output).exists():
        _validate_source_scheduler(output)
        terminal_source = validate_source(deep_pack=True)
        if (
            terminal_source["feature_store"] != contract["source"]["feature_store"]
            or _pack_stat_snapshot(terminal_source["feature_store"]) != initial_pack_stats
        ):
            raise ContractError("source pack changed after completed label-blind scheduler")
        seal = _inference_seal_payload(output, created_utc=_utcnow())
        _write_json_once(inference_seal_path(output), seal)
        return {"status": "SEALED_FROM_VALID_SCHEDULER", "seal": verify_inference_seal(output)}

    def run(job: ScoreJob, command: Sequence[str]) -> dict[str, Any]:
        started_unix_ns = time.time_ns()
        started_utc = _utcnow_precise()
        returncode = int(subprocess.run(list(command), cwd=REPO, check=False).returncode)
        completed_utc = _utcnow_precise()
        completed_unix_ns = time.time_ns()
        return {
            "job_key": job.key,
            "command": list(command),
            "started_utc": started_utc,
            "completed_utc": completed_utc,
            "started_unix_ns": started_unix_ns,
            "completed_unix_ns": completed_unix_ns,
            "returncode": returncode,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(run, job, command)
            for job, command in zip(score_jobs(), commands, strict=True)
        ]
        events = [future.result() for future in concurrent.futures.as_completed(futures)]
    if any(event["returncode"] for event in events):
        raise ContractError("one or more label-blind embedding jobs failed")
    # Rehash every pack byte and compare inode/mtime/size after all ten jobs;
    # this narrows same-size mutation races around inference.
    terminal_source = validate_source(deep_pack=True)
    if (
        terminal_source["feature_store"] != contract["source"]["feature_store"]
        or _pack_stat_snapshot(terminal_source["feature_store"]) != initial_pack_stats
    ):
        raise ContractError("source pack changed during label-blind inference")
    peak = _parallel_peak(events)
    if peak != MAX_WORKERS:
        raise ContractError(f"source embedding concurrency was {peak}, required 6")
    _write_json_once(
        source_scheduler_path(output),
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE_LABEL_BLIND_SOURCE_EMBEDDINGS",
            "created_utc": _utcnow(),
            "contract": _artifact(contract_path(output)),
            "environment": _artifact(environment_path),
            "configured_max_workers": MAX_WORKERS,
            "observed_peak_workers": peak,
            "job_count": 10,
            "score_rows": 925,
            "target_outcomes_opened": False,
            "events": sorted(events, key=lambda value: value["job_key"]),
        },
    )
    _validate_source_scheduler(output)
    seal = _inference_seal_payload(output, created_utc=_utcnow())
    _write_json_once(inference_seal_path(output), seal)
    return {"status": "SEALED", "seal": verify_inference_seal(output)}


def _open_target_outcomes_after_seal(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The sole target-label opening function; its first action verifies the seal."""

    verify_inference_seal(root)
    blocks: list[pd.DataFrame] = []
    identities: dict[str, Any] = {}
    for target, spec in TARGETS.items():
        identity = _pinned_artifact(spec.outcome_source, spec.outcome_sha256)
        blind = _validate_blind(
            pd.read_csv(blind_path(root, target), low_memory=False), spec, context=target
        )
        raw = pd.read_csv(spec.outcome_source, low_memory=False)
        required = {"slide_id", "patient_id", "target_label", "kras"}
        if required - set(raw):
            raise ContractError(f"{target}: outcome source lacks {sorted(required - set(raw))}")
        if raw[["slide_id", "patient_id", "target_label"]].isna().any().any():
            raise ContractError(f"{target}: outcome source contains null required values")
        raw = raw.copy()
        raw["slide_id"] = raw["slide_id"].astype(str)
        raw["patient_id"] = raw["patient_id"].astype(str)
        selected = blind[["slide_id", "patient_id"]].merge(
            raw,
            on=["slide_id", "patient_id"],
            how="inner",
            validate="one_to_one",
        )
        numeric_labels = pd.to_numeric(selected["target_label"], errors="raise")
        if (
            not np.isfinite(numeric_labels.to_numpy(float)).all()
            or not numeric_labels.isin([0, 1]).all()
        ):
            raise ContractError(f"{target}: target labels must be finite exact binary values")
        labels = numeric_labels.astype(int)
        kras = selected["kras"].astype("string")
        if (
            len(selected) != spec.slides
            or set(labels) != {0, 1}
            or selected.groupby("patient_id")["target_label"].nunique().ne(1).any()
            or kras.isna().any()
            or set(kras.astype(str)) != {"mutant", "wild_type"}
            or selected.assign(_kras=kras.astype(str))
            .groupby("patient_id")["_kras"]
            .nunique()
            .ne(1)
            .any()
            or not np.array_equal(
                kras.astype(str).eq("mutant").to_numpy(dtype=int), labels.to_numpy(int)
            )
        ):
            raise ContractError(f"{target}: sealed blind/outcome join drifted")
        selected["target_label"] = labels
        selected["cohort"] = spec.cohort
        selected["source_family_exposed"] = spec.source_family_exposed
        patient = selected.sort_values("slide_id").drop_duplicates("patient_id")
        if len(patient) != spec.patients or int(patient["target_label"].sum()) != spec.mutant:
            raise ContractError(f"{target}: target patient/mutant census drifted")
        blocks.append(selected)
        identities[target] = identity
    combined = pd.concat(blocks, ignore_index=True)
    if (
        len(combined) != 185
        or combined["slide_id"].astype(str).nunique() != 185
        or combined["patient_id"].astype(str).nunique() != 159
        or int(
            combined.sort_values("slide_id")
            .drop_duplicates("patient_id")["target_label"]
            .sum()
        )
        != 67
    ):
        raise ContractError("combined target-internal population drifted")
    return combined, identities


def _target_fold_layout(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    patient = (
        frame.groupby("patient_id", sort=True)
        .agg(
            cohort=("cohort", "first"),
            kras=("kras", "first"),
            label=("target_label", "first"),
        )
        .reset_index()
    )
    if len(patient) != 159 or int(patient["label"].sum()) != 67:
        raise ContractError("target-internal patient fold population drifted")
    fold = legacy_local._deal_outer_folds(patients=patient)  # noqa: SLF001
    layout = patient[["patient_id"]].copy()
    layout["k_fold"] = fold.to_numpy(dtype=int)
    validation_counts: dict[str, int] = {}
    for fold_index in FOLDS:
        flags = legacy_local._carve_val(patient, fold, fold_index)  # noqa: SLF001
        layout[f"val_fold_{fold_index}"] = flags.to_numpy(dtype=int)
        validation_counts[str(fold_index)] = int(
            patient.loc[flags.astype(bool), "label"].sum()
        )
    sizes = layout["k_fold"].value_counts().sort_index()
    mutants = patient.groupby(layout["k_fold"].to_numpy())["label"].sum().sort_index()
    if (
        set(layout["k_fold"]) != set(FOLDS)
        or int(sizes.max() - sizes.min()) > 2
        or int(mutants.max() - mutants.min()) > 2
        or min(validation_counts.values()) < legacy_local.MIN_VAL_POSITIVES
    ):
        raise ContractError("fixed cohort-by-KRAS local OOF layout is unbalanced")
    return layout, {
        "layout_seed": legacy_local.FOLD_SEED,
        "carve_seed_base": legacy_local.CARVE_SEED_BASE,
        "fold_patients": {str(key): int(value) for key, value in sizes.items()},
        "fold_mutants": {str(key): int(value) for key, value in mutants.items()},
        "validation_mutants": validation_counts,
        "shared_across_model_seeds": True,
    }


def _validate_target_open(root: Path) -> dict[str, Any]:
    verify_inference_seal(root)
    receipt = _read_json(target_open_path(root))
    manifest = pd.read_csv(labeled_manifest_path(root), low_memory=False)
    required = {
        "slide_id",
        "patient_id",
        "target_label",
        "kras",
        "cohort",
        "source_family_exposed",
        "k_fold",
        *(f"val_fold_{fold}" for fold in FOLDS),
    }
    if required - set(manifest):
        raise ContractError("target-internal labeled manifest schema drifted")
    numeric_labels = pd.to_numeric(manifest["target_label"], errors="raise")
    if (
        not np.isfinite(numeric_labels.to_numpy(float)).all()
        or not numeric_labels.isin([0, 1]).all()
    ):
        raise ContractError("target-internal manifest labels are not exact binary values")
    manifest = manifest.copy()
    manifest["target_label"] = numeric_labels.astype(int)
    kras = manifest["kras"].astype("string")
    if (
        kras.isna().any()
        or set(kras.astype(str)) != {"mutant", "wild_type"}
        or manifest.assign(_kras=kras.astype(str))
        .groupby("patient_id")["_kras"]
        .nunique()
        .ne(1)
        .any()
        or not np.array_equal(
            kras.astype(str).eq("mutant").to_numpy(dtype=int),
            manifest["target_label"].to_numpy(int),
        )
        or manifest.groupby("patient_id")["target_label"].nunique().ne(1).any()
        or manifest.groupby("patient_id")["cohort"].nunique().ne(1).any()
        or set(manifest["cohort"].astype(str)) != {"RIH", "SurGen"}
    ):
        raise ContractError("target-internal KRAS/label patient semantics drifted")
    layout, layout_receipt = _target_fold_layout(manifest)
    manifest_layout = (
        manifest[["patient_id", "k_fold", *(f"val_fold_{fold}" for fold in FOLDS)]]
        .drop_duplicates()
        .sort_values("patient_id", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_layout = layout.sort_values("patient_id", kind="mergesort").reset_index(drop=True)
    for column in ["k_fold", *(f"val_fold_{fold}" for fold in FOLDS)]:
        manifest_layout[column] = pd.to_numeric(
            manifest_layout[column], errors="raise"
        ).astype(int)
    if not manifest_layout.equals(expected_layout):
        raise ContractError("target-internal fixed fold/validation layout drifted")
    if (
        len(manifest) != 185
        or manifest["slide_id"].astype(str).nunique() != 185
        or manifest["patient_id"].astype(str).nunique() != 159
        or int(
            manifest.sort_values("slide_id")
            .drop_duplicates("patient_id")["target_label"]
            .sum()
        )
        != 67
    ):
        raise ContractError("target-internal opening receipt/manifest drifted")
    expected_split_artifacts = {
        name: _artifact(split_dir(root) / name)
        for name in ("splits.parquet", ".integrity_hash", "summary.json")
    }
    from oceanpath.splitting import verify_split_integrity

    verify_split_integrity(split_dir(root), labeled_manifest_path(root))
    splits = pd.read_parquet(split_dir(root) / "splits.parquet")
    expected_columns = [
        "slide_id",
        "patient_id",
        "target_label",
        "k_fold",
        *(f"val_fold_{fold}" for fold in FOLDS),
    ]
    source_membership = manifest[expected_columns].copy()
    split_membership = splits[
        [
            "slide_id",
            "group_id",
            "target_label",
            "fold",
            *(f"val_fold_{fold}" for fold in FOLDS),
        ]
    ].rename(
        columns={
            "group_id": "patient_id",
            "fold": "k_fold",
        }
    )
    for frame in (source_membership, split_membership):
        frame["slide_id"] = frame["slide_id"].astype(str)
        frame["patient_id"] = frame["patient_id"].astype(str)
        for column in ["target_label", "k_fold", *(f"val_fold_{fold}" for fold in FOLDS)]:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        frame.sort_values("slide_id", kind="mergesort", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if not source_membership.equals(split_membership):
        raise ContractError("target-internal split membership drifted from labeled manifest")
    source_manifest = pd.read_csv(
        SOURCE_ROOT / "inputs/manifests/tcga_surgen_primary.csv", low_memory=False
    )
    if (
        set(manifest["slide_id"].astype(str)) & set(source_manifest["slide_id"].astype(str))
        or set(manifest["patient_id"].astype(str))
        & set(source_manifest["patient_id"].astype(str))
    ):
        raise ContractError("target-internal manifest overlaps source development")
    outcome_identities = {
        target: _pinned_artifact(spec.outcome_source, spec.outcome_sha256)
        for target, spec in TARGETS.items()
    }
    created = receipt.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("target opening receipt lacks created_utc")
    _parse_utc(created, context="target opening created_utc")
    expected_receipt = _target_open_payload(
        root,
        created_utc=created,
        outcome_identities=outcome_identities,
        layout_receipt=layout_receipt,
        split_artifacts=expected_split_artifacts,
    )
    if receipt != expected_receipt:
        raise ContractError("target-internal opening receipt does not replay exactly")
    return receipt


def _target_open_payload(
    root: Path,
    *,
    created_utc: str,
    outcome_identities: Mapping[str, Any],
    layout_receipt: Mapping[str, Any],
    split_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "TARGET_LABELS_OPENED_FOR_INTERNAL_ANALYSIS_ONLY",
        "created_utc": created_utc,
        "result_role": INTERNAL_ROLE,
        "source_inference_seal": _artifact(inference_seal_path(root)),
        "seal_status_observed_before_outcome_open": (
            "SEALED_LABEL_BLIND_BEFORE_TARGET_OUTCOME_OPEN"
        ),
        "outcome_sources": dict(outcome_identities),
        "labeled_manifest": _artifact(labeled_manifest_path(root)),
        "splits": dict(split_artifacts),
        "population": {"slides": 185, "patients": 159, "mutant": 67},
        "local_oof_layout": dict(layout_receipt),
        "primary_support_used": False,
        "allowed_target_labels": ["RIH-M", "SurGen-M"],
        "external_validation_claim_permitted": False,
        "surgen_metastatic_source_family_exposed": True,
    }


def open_targets(root: Path, *, apply: bool) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    seal = verify_inference_seal(output)
    if not apply:
        return {
            "status": "DRY_RUN_READY_TO_OPEN_TARGET_LABELS",
            "source_inference_seal": _artifact(inference_seal_path(output)),
            "target_outcomes_opened": False,
            "result_role_after_open": INTERNAL_ROLE,
        }
    if target_open_path(output).exists():
        return _validate_target_open(output)
    if labeled_manifest_path(output).exists() or split_dir(output).exists():
        raise ContractError("partial target-label opening namespace exists")
    combined, outcome_identities = _open_target_outcomes_after_seal(output)
    source_manifest = pd.read_csv(
        SOURCE_ROOT / "inputs/manifests/tcga_surgen_primary.csv", low_memory=False
    )
    if (
        set(combined["slide_id"].astype(str)) & set(source_manifest["slide_id"].astype(str))
        or set(combined["patient_id"].astype(str))
        & set(source_manifest["patient_id"].astype(str))
    ):
        raise ContractError("target-internal patients/slides overlap source development")
    layout, layout_receipt = _target_fold_layout(combined)
    old_fold_columns = {"k_fold", *(f"val_fold_{fold}" for fold in FOLDS)}
    combined = combined.drop(columns=[name for name in old_fold_columns if name in combined])
    combined = combined.merge(layout, on="patient_id", validate="many_to_one")
    combined = combined.sort_values(["cohort", "patient_id", "slide_id"], kind="mergesort")
    combined = combined.reset_index(drop=True)
    _write_bytes_once(labeled_manifest_path(output), combined.to_csv(index=False).encode())
    from oceanpath.splitting import SplitConfig, generate_splits

    generate_splits(
        SplitConfig(
            scheme="predefined_oof_kfold",
            name="aim1_balanced5",
            csv_path=str(labeled_manifest_path(output)),
            output_dir=str(split_dir(output)),
            filename_column="slide_id",
            label_column="target_label",
            group_column="patient_id",
            fold_column="k_fold",
            n_folds=len(FOLDS),
            seed=42,
        ),
        force=False,
    )
    split_artifacts = {
        name: _artifact(split_dir(output) / name)
        for name in ("splits.parquet", ".integrity_hash", "summary.json")
    }
    if seal["status"] != "SEALED_LABEL_BLIND_BEFORE_TARGET_OUTCOME_OPEN":
        raise ContractError("unexpected source-inference seal status")
    _write_json_once(
        target_open_path(output),
        _target_open_payload(
            output,
            created_utc=_utcnow(),
            outcome_identities=outcome_identities,
            layout_receipt=layout_receipt,
            split_artifacts=split_artifacts,
        ),
    )
    return _validate_target_open(output)


def _source_native_analysis_authority() -> dict[str, Any]:
    completion_path = (
        SOURCE_ROOT
        / "downstream_v2/continuation_v3/analysis/analysis_completion_receipt.json"
    )
    completion_identity = _pinned_artifact(
        completion_path, SOURCE_ANALYSIS_COMPLETION_SHA256, 2_070
    )
    completion = _read_json(completion_path)
    results_identity = completion.get("artifacts", {}).get("results")
    if not isinstance(results_identity, Mapping):
        # The receipt schema names direct artifacts at top level in some
        # continuation revisions; accept only that one alternate exact slot.
        results_identity = completion.get("results")
    if not isinstance(results_identity, Mapping):
        raise ContractError("source continuation completion lacks results identity")
    results_path = Path(str(results_identity.get("path", "")))
    if dict(results_identity) != _artifact(results_path):
        raise ContractError("source continuation results identity drifted")
    results = _read_json(results_path)
    for target, cohort in (("rih_metastatic", "RIH"), ("sr1482_metastatic", "SurGen")):
        block = (
            results.get("aim2", {})
            .get("targets", {})
            .get(target, {})
            .get("encoders", {})
            .get("univ1", {})
        )
        expected = EXPECTED_NATIVE[cohort]
        if (
            not math.isclose(float(block.get("auroc", -1)), expected["auroc"], abs_tol=1e-15)
            or block.get("auroc_ci95") != expected["ci95"]
        ):
            raise ContractError(f"source continuation native gate drifted: {cohort}")
    return {"completion": completion_identity, "results": dict(results_identity)}


def _patient_adapter_data(root: Path) -> dict[str, dict[int, pd.DataFrame]]:
    _validate_target_open(root)
    manifest = pd.read_csv(labeled_manifest_path(root), low_memory=False)
    data: dict[str, dict[int, pd.DataFrame]] = {}
    embedding_columns = [f"e{index}" for index in range(EMBED_DIM)]
    for target, spec in TARGETS.items():
        target_manifest = manifest.loc[
            manifest["cohort"].astype(str).eq(spec.cohort)
        ][["slide_id", "patient_id", "target_label"]]
        per_seed: dict[int, pd.DataFrame] = {}
        for seed in MODEL_SEEDS:
            job = ScoreJob(target, seed)
            scores = _validate_cached_embedding(root, job)
            assert scores is not None
            joined = scores.merge(
                target_manifest,
                on="slide_id",
                validate="one_to_one",
            )
            if len(joined) != spec.slides:
                raise ContractError(f"{job.key}: labeled embedding roster is incomplete")
            patient = (
                joined.groupby("patient_id", sort=True)
                .agg(
                    label=("target_label", "first"),
                    eta_native=("logit", "mean"),
                    **{
                        column: (column, "mean") for column in embedding_columns
                    },
                )
            )
            patient["label"] = patient["label"].astype(int)
            if (
                len(patient) != spec.patients
                or int(patient["label"].sum()) != spec.mutant
                or not np.isfinite(
                    patient[["eta_native", *embedding_columns]].to_numpy(float)
                ).all()
            ):
                raise ContractError(f"{job.key}: patient embedding census/value drifted")
            per_seed[seed] = patient
        reference = per_seed[MODEL_SEEDS[0]]
        for seed in MODEL_SEEDS[1:]:
            if (
                not reference.index.equals(per_seed[seed].index)
                or not np.array_equal(reference["label"], per_seed[seed]["label"])
            ):
                raise ContractError(f"{spec.cohort}: patient/label order differs by seed")
        native = np.mean(
            np.vstack([per_seed[seed]["eta_native"].to_numpy(float) for seed in MODEL_SEEDS]),
            axis=0,
        )
        point = float(roc_auc_score(reference["label"].to_numpy(int), native))
        if not math.isclose(point, EXPECTED_NATIVE[spec.cohort]["auroc"], abs_tol=1e-15):
            raise ContractError(
                f"{spec.cohort}: new patient native AUROC does not reproduce source authority"
            )
        data[spec.cohort] = per_seed
    return data


@contextlib.contextmanager
def _adapter_protocol(*, primary_outer_seed: int) -> Iterable[None]:
    """Temporarily parameterize the immutable E2f implementation for five seeds."""

    names = (
        "SEEDS",
        "PRIMARY_OUTER_SEED",
        "OUTER_SENSITIVITY_SEEDS",
        "SUPPORT_SIZES",
        "SUPPORT_DRAWS",
        "BOOTSTRAP_DRAWS",
        "BOOTSTRAP_SEED",
    )
    old = {name: getattr(adapter, name) for name in names}
    if old["SEEDS"] != (42, 43, 44):
        raise ContractError("inherited adapter seed authority changed unexpectedly")
    try:
        adapter.SEEDS = MODEL_SEEDS
        adapter.PRIMARY_OUTER_SEED = primary_outer_seed
        adapter.OUTER_SENSITIVITY_SEEDS = OUTER_LAYOUT_SEEDS
        adapter.SUPPORT_SIZES = SUPPORT_TOTAL
        adapter.SUPPORT_DRAWS = SUPPORT_DRAWS_PER_LAYOUT
        adapter.BOOTSTRAP_DRAWS = N_BOOTSTRAP
        adapter.BOOTSTRAP_SEED = BOOTSTRAP_SEED
        yield
    finally:
        for name, value in old.items():
            setattr(adapter, name, value)


def analytical_power_precision_table() -> dict[str, Any]:
    rows = []
    designs = (("RIH-M", 85, 37), ("SurGen-M", 74, 30), ("pooled", 159, 67))
    for design, current_n, n_positive in designs:
        prevalence = n_positive / current_n
        for alternative in (0.60, 0.65, 0.70):
            required = int(
                math.ceil(
                    legacy_local._hanley_mcneil_n(  # noqa: SLF001
                        alternative, prevalence, 0.05, 0.80
                    )
                )
            )
            n_negative = current_n - n_positive
            q1 = alternative / (2 - alternative)
            q2 = 2 * alternative * alternative / (1 + alternative)
            variance = (
                alternative * (1 - alternative)
                + (n_positive - 1) * (q1 - alternative * alternative)
                + (n_negative - 1) * (q2 - alternative * alternative)
            ) / (n_positive * n_negative)
            rows.append(
                {
                    "design": design,
                    "assumed_true_auroc": alternative,
                    "patients_required_80pct_one_sided_power_alpha_0p05": required,
                    "current_n": current_n,
                    "current_mutant": n_positive,
                    "current_design_approx_se": float(math.sqrt(variance)),
                    "current_design_approx_two_sided_95pct_half_width": float(
                        1.96 * math.sqrt(variance)
                    ),
                }
            )
    return {
        "role": "ANALYTICAL_THEORETICAL_POWER_PRECISION_ONLY; not observed performance",
        "method": "Hanley-McNeil normal approximation",
        "equal_cohort_macro_note": (
            "macro precision is obtained empirically by the prespecified paired "
            "within-cohort bootstrap, not by treating 159 patients as one cohort"
        ),
        "rows": rows,
    }


def _local_overrides(root: Path, job: LocalJob, *, num_workers: int) -> list[str]:
    """Exact source-recipe overrides for one genuinely fold-scoped local fit."""

    if num_workers != DEFAULT_NUM_WORKERS:
        raise ContractError("governed local MIL requires exactly --num-workers 4")
    return [
        "platform=colon_workstation",
        "data=aim1",
        "data.aim1_model=e2e_met",
        f"data.manifest_stem={labeled_manifest_path(root).stem}",
        f"data.csv_path={labeled_manifest_path(root)}",
        f"platform.splits_root={root / 'inputs/splits'}",
        "+data.cohort_column=cohort",
        "encoder=univ1",
        "splits=aim1_balanced",
        f"splits.seed={job.seed}",
        "model=abmil",
        "model.embed_dim=512",
        "model.attn_dim=384",
        "model.input_dropout=0.10",
        "model.dropout=0.25",
        "training=aim1",
        "training.lr=0.0001",
        "training.weight_decay=0.00001",
        f"training.seed={job.seed}",
        f"training.dataset_max_instances={CAP}",
        "training.eval_full_bags=true",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        "training.skip_finalize=true",
        f"training.num_workers={num_workers}",
        f"train_dir={local_run_dir(root, job)}",
        f"exp_name=aim2_met_local_bound_univ1_seed{job.seed}_fold{job.fold}",
        f"hydra.run.dir={root / 'local_mil/hydra_runs' / job.key}",
        "hydra.job.chdir=false",
    ]


def _local_config(root: Path, job: LocalJob, *, num_workers: int):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=_local_overrides(root, job, num_workers=num_workers),
        )
    from omegaconf import OmegaConf

    expected = {
        "data.name": "aim1_e2e_met",
        "data.csv_path": str(labeled_manifest_path(root)),
        "data.patient_id_column": "patient_id",
        "splits.scheme": "predefined_oof_kfold",
        "splits.n_folds": 5,
        "splits.allow_group_overlap": False,
        "encoder.name": "uni_v1",
        "encoder.feature_dim": 1024,
        "model.arch": "abmil",
        "model.embed_dim": 512,
        "model.attn_dim": 384,
        "model.input_dropout": 0.1,
        "model.dropout": 0.25,
        "training.lr": 1e-4,
        "training.weight_decay": 1e-5,
        "training.dataset_max_instances": CAP,
        "training.eval_full_bags": True,
        "training.train_sampling_strategy": "patient_natural",
        "training.sample_weight_column": None,
        "training.loss_type": "bce",
        "training.training_class_weighted_loss": False,
        "training.monitor_metric": "val/patient_auroc",
        "training.monitor_mode": "max",
        "training.skip_finalize": True,
        "training.seed": job.seed,
        "training.num_workers": DEFAULT_NUM_WORKERS,
    }
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise ContractError(f"{job.key}: governed local recipe drifted at {key}")
    pack_path = Path(str(OmegaConf.select(cfg, "training.packed_dir"))).resolve()
    contract = load_contract(root, deep_pack=False)
    if pack_path != Path(str(contract["source"]["feature_store"]["path"])).resolve():
        raise ContractError(f"{job.key}: local recipe is not bound to source UNI-v1 pack")
    _pinned_artifact(source_oof_config(job.seed), SOURCE_OOF_CONFIG_SHA256[job.seed], 5_345)
    source_cfg = OmegaConf.load(source_oof_config(job.seed))
    material_keys = (
        "platform.precision",
        "platform.strategy",
        "platform.accelerator",
        "platform.devices",
        "platform.num_gpus",
        "platform.prefetch_factor",
        "encoder.name",
        "encoder.feature_dim",
        "model.name",
        "model.arch",
        "model.embed_dim",
        "model.num_fc_layers",
        "model.attn_dim",
        "model.gate",
        "model.dropout",
        "model.gradient_checkpointing",
        "model.input_dropout",
        "splits.scheme",
        "splits.name",
        "splits.fold_column",
        "splits.label_column",
        "splits.group_column",
        "splits.allow_group_overlap",
        "splits.n_folds",
        "training.lr",
        "training.weight_decay",
        "training.lr_scheduler",
        "training.final_lr_fraction",
        "training.warmup_epochs",
        "training.max_epochs",
        "training.batch_size",
        "training.adam_betas",
        "training.adam_eps",
        "training.loss_type",
        "training.class_weights",
        "training.training_class_weighted_loss",
        "training.validation_loss_weighted",
        "training.max_instances",
        "training.dataset_max_instances",
        "training.eval_full_bags",
        "training.sample_weight_column",
        "training.class_weighted_sampling",
        "training.train_sampling_strategy",
        "training.use_preallocated_collator",
        "training.return_coords",
        "training.verify_splits",
        "training.drop_last",
        "training.force_float32",
        "training.verify_packed_source",
        "training.resident_device",
        "training.fixed_bag_size",
        "training.eval_fixed_bag_size",
        "training.eval_n_crops",
        "training.instance_dropout",
        "training.feature_noise_std",
        "training.bag_curriculum",
        "training.collect_embeddings",
        "training.monitor_metric",
        "training.monitor_mode",
        "training.early_stopping_patience",
        "training.early_stopping_min_delta",
        "training.min_epoch_before_stop",
        "training.es_min_val_positives",
        "training.fixed_epoch_budget",
        "training.save_top_k",
        "training.save_last",
        "training.cap_strategy",
        "training.cap_grid_size",
        "training.compile_model",
        "training.freeze_aggregator",
        "training.deterministic",
        "training.gradient_clip_val",
        "training.accumulate_grad_batches",
        "training.skip_finalize",
    )
    for key in material_keys:
        if OmegaConf.select(cfg, key) != OmegaConf.select(source_cfg, key):
            raise ContractError(f"{job.key}: local/source material recipe differs at {key}")
    source_workers = OmegaConf.select(source_cfg, "training.num_workers")
    source_effective_workers = (
        OmegaConf.select(source_cfg, "platform.num_workers")
        if source_workers is None
        else source_workers
    )
    if int(source_effective_workers) != DEFAULT_NUM_WORKERS:
        raise ContractError("source effective worker recipe drifted")
    local_sections = OmegaConf.to_container(cfg, resolve=True)
    source_sections = OmegaConf.to_container(source_cfg, resolve=True)
    assert isinstance(local_sections, dict) and isinstance(source_sections, dict)
    local_training = dict(local_sections["training"])
    source_training = dict(source_sections["training"])
    local_training["num_workers"] = DEFAULT_NUM_WORKERS
    source_training["num_workers"] = DEFAULT_NUM_WORKERS
    for section, local_value, source_value in (
        ("encoder", local_sections["encoder"], source_sections["encoder"]),
        ("model", local_sections["model"], source_sections["model"]),
        ("training", local_training, source_training),
    ):
        if local_value != source_value:
            raise ContractError(f"{job.key}: canonical local/source {section} recipe differs")
    return cfg


def _expected_local_partition(root: Path, job: LocalJob) -> dict[str, set[str]]:
    from oceanpath.splitting import get_slide_ids_for_fold

    manifest = pd.read_csv(labeled_manifest_path(root), low_memory=False)
    splits = pd.read_parquet(split_dir(root) / "splits.parquet")
    ids = get_slide_ids_for_fold(splits, job.fold, scheme="predefined_oof_kfold")
    partition = {name: set(map(str, values)) for name, values in ids.items()}
    if set().union(*partition.values()) != set(manifest["slide_id"].astype(str)):
        raise ContractError(f"{job.key}: local split does not cover the full target roster")
    if any(
        partition[left] & partition[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ContractError(f"{job.key}: slide leakage across local partitions")
    patient_by_slide = manifest.set_index(manifest["slide_id"].astype(str))["patient_id"].astype(str)
    patient_sets = {
        name: set(patient_by_slide.loc[sorted(slides)]) for name, slides in partition.items()
    }
    if any(
        patient_sets[left] & patient_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ContractError(f"{job.key}: patient leakage across local partitions")
    expected_test = set(
        manifest.loc[pd.to_numeric(manifest["k_fold"]).eq(job.fold), "slide_id"].astype(str)
    )
    expected_val = set(
        manifest.loc[
            pd.to_numeric(manifest[f"val_fold_{job.fold}"]).eq(1)
            & ~pd.to_numeric(manifest["k_fold"]).eq(job.fold),
            "slide_id",
        ].astype(str)
    )
    expected_train = set(manifest["slide_id"].astype(str)) - expected_test - expected_val
    if partition != {"train": expected_train, "val": expected_val, "test": expected_test}:
        raise ContractError(f"{job.key}: exact train/val/test membership drifted")
    return partition


def _fold_artifacts(run_dir: Path, fold: int) -> tuple[dict[str, Any], dict[str, Any]]:
    fold_dir = run_dir / f"fold_{fold}"
    completion = _read_json(fold_dir / "completion.json")
    artifacts = completion.get("artifacts")
    required = {"config", "metrics", "val_predictions", "best_checkpoint", "test_predictions"}
    if (
        completion.get("status") != "completed"
        or completion.get("fold") != fold
        or not isinstance(completion.get("training_fingerprint"), str)
        or not isinstance(artifacts, Mapping)
        or required - set(artifacts)
        or set(artifacts) - (required | {"sampling_plan"})
    ):
        raise ContractError(f"local fold{fold}: workflow completion schema drifted")
    resolved: dict[str, Any] = {}
    for name, evidence in artifacts.items():
        if not isinstance(evidence, Mapping) or set(evidence) != {"path", "size_bytes", "sha256"}:
            raise ContractError(f"local fold{fold}: malformed {name} evidence")
        path = fold_dir / str(evidence["path"])
        identity = _artifact(path)
        relative = path.resolve().relative_to(fold_dir.resolve()).as_posix()
        if (
            relative != evidence["path"]
            or identity["size_bytes"] != evidence["size_bytes"]
            or identity["sha256"] != evidence["sha256"]
        ):
            raise ContractError(f"local fold{fold}: {name} evidence drifted")
        resolved[name] = identity
    return completion, resolved


def _local_job_payload(
    root: Path,
    job: LocalJob,
    *,
    created_utc: str,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = local_run_dir(root, job)
    completion, artifacts = _fold_artifacts(run_dir, job.fold)
    partition = _expected_local_partition(root, job)
    test = pd.read_parquet(Path(artifacts["test_predictions"]["path"]))
    manifest = pd.read_csv(labeled_manifest_path(root), low_memory=False)
    expected = manifest.loc[
        manifest["slide_id"].astype(str).isin(partition["test"]),
        ["slide_id", "target_label"],
    ].copy()
    observed = test[["slide_id", "label"]].copy()
    expected["slide_id"] = expected["slide_id"].astype(str)
    observed["slide_id"] = observed["slide_id"].astype(str)
    expected["target_label"] = pd.to_numeric(expected["target_label"], errors="raise").astype(int)
    observed["label"] = pd.to_numeric(observed["label"], errors="raise").astype(int)
    expected.sort_values("slide_id", inplace=True, kind="mergesort")
    observed.sort_values("slide_id", inplace=True, kind="mergesort")
    if (
        list(test.columns) != ["slide_id", "label", "prob_1", "logit"]
        or test["slide_id"].astype(str).duplicated().any()
        or not expected.reset_index(drop=True).rename(columns={"target_label": "label"}).equals(
            observed.reset_index(drop=True)
        )
        or not np.isfinite(test[["prob_1", "logit"]].to_numpy(float)).all()
    ):
        raise ContractError(f"{job.key}: local held-out predictions drifted")
    from omegaconf import OmegaConf

    from oceanpath.workflows.training import training_run_fingerprint

    cfg = _local_config(root, job, num_workers=DEFAULT_NUM_WORKERS)
    config_path = Path(artifacts["config"]["path"])
    if config_path.read_text(encoding="utf-8") != OmegaConf.to_yaml(cfg, resolve=True):
        raise ContractError(f"{job.key}: resolved local training config drifted")
    expected_fingerprint = training_run_fingerprint(cfg)
    _assert_local_training_fingerprint(
        completion["training_fingerprint"], expected_fingerprint, context=job.key
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_TARGET_INTERNAL_LOCAL_MIL_FOLD",
        "created_utc": created_utc,
        "job_key": job.key,
        "seed": job.seed,
        "fold": job.fold,
        "result_role": LOCAL_ROLE,
        "bound_label": "sample/recipe-bounded empirical local OOF bound",
        "target_internal_not_external_validation": True,
        "source_anchored": False,
        "source_or_main_model_modified": False,
        "refit_performed": False,
        "command": _local_command(root, job, num_workers=DEFAULT_NUM_WORKERS),
        "contract": _artifact(contract_path(root)),
        "target_open_receipt": _artifact(target_open_path(root)),
        "labeled_manifest": _artifact(labeled_manifest_path(root)),
        "splits": _artifact(split_dir(root) / "splits.parquet"),
        "training_workflow": _artifact(REPO / "src/oceanpath/workflows/training.py"),
        "workflow_completion": _artifact(run_dir / f"fold_{job.fold}/completion.json"),
        "training_fingerprint": completion["training_fingerprint"],
        "artifacts": artifacts,
        "partition_slides": {name: len(values) for name, values in partition.items()},
        "execution": dict(execution),
    }


def _assert_local_training_fingerprint(
    observed: Any, expected: str, *, context: str
) -> None:
    if not isinstance(observed, str) or observed != expected:
        raise ContractError(f"{context}: local training fingerprint does not recompute")


def _validate_local_job(root: Path, job: LocalJob) -> dict[str, Any] | None:
    receipt_path = local_receipt_path(root, job)
    run_dir = local_run_dir(root, job)
    if not receipt_path.exists() and not run_dir.exists():
        return None
    if not receipt_path.is_file() or receipt_path.is_symlink() or not run_dir.is_dir():
        raise ContractError(f"{job.key}: partial/symlinked local fold job")
    stored = _read_json(receipt_path)
    created = stored.get("created_utc")
    execution = stored.get("execution")
    if not isinstance(created, str) or not isinstance(execution, Mapping):
        raise ContractError(f"{job.key}: local fold receipt timestamps are malformed")
    _parse_utc(created, context=f"{job.key}/created")
    exact_execution_keys = {
        "job_id",
        "started_utc",
        "completed_utc",
        "started_unix_ns",
        "completed_unix_ns",
        "returncode",
        "physical_fit_count",
    }
    if (
        set(execution) != exact_execution_keys
        or execution.get("job_id") != f"local_mil.seed{job.seed}.fold{job.fold}"
        or execution.get("returncode") != 0
        or execution.get("physical_fit_count") != 1
        or execution.get("completed_unix_ns", 0) <= execution.get("started_unix_ns", 0)
        or _parse_utc(execution.get("completed_utc"), context=f"{job.key}/complete")
        <= _parse_utc(execution.get("started_utc"), context=f"{job.key}/start")
    ):
        raise ContractError(f"{job.key}: local fold execution receipt drifted")
    expected = _local_job_payload(
        root, job, created_utc=created, execution=dict(execution)
    )
    if stored != expected:
        raise ContractError(f"{job.key}: local fold receipt does not replay exactly")
    return stored


def _local_command(root: Path, job: LocalJob, *, num_workers: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_train-local-one",
        "--output-root",
        str(root),
        "--seed",
        str(job.seed),
        "--fold",
        str(job.fold),
        "--num-workers",
        str(num_workers),
    ]


def _train_local_one(root: Path, job: LocalJob, *, num_workers: int) -> None:
    output = validate_output_root(root, must_exist=True)
    _validate_target_open(output)
    if num_workers != DEFAULT_NUM_WORKERS:
        raise ContractError("governed local MIL requires exactly --num-workers 4")
    if _validate_local_job(output, job) is not None:
        return
    if local_run_dir(output, job).exists() or local_receipt_path(output, job).exists():
        raise ContractError(f"{job.key}: partial local job requires a fresh campaign root")
    cfg = _local_config(output, job, num_workers=num_workers)
    started_unix_ns = time.time_ns()
    started_utc = _utcnow_precise()
    from oceanpath.workflows.training import fold_context, run_fold

    with fold_context(job.fold):
        run_fold(cfg=cfg, fold_idx=job.fold, output_dir=local_run_dir(output, job))
    completed_utc = _utcnow_precise()
    completed_unix_ns = time.time_ns()
    execution = {
        "job_id": f"local_mil.seed{job.seed}.fold{job.fold}",
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": completed_unix_ns,
        "returncode": 0,
        "physical_fit_count": 1,
    }
    payload = _local_job_payload(
        output, job, created_utc=_utcnow(), execution=execution
    )
    _write_json_once(local_receipt_path(output, job), payload)
    _validate_local_job(output, job)


def _assemble_local_oof(root: Path, seed: int) -> pd.DataFrame:
    manifest = pd.read_csv(labeled_manifest_path(root), low_memory=False)
    blocks: list[pd.DataFrame] = []
    for fold in FOLDS:
        job = LocalJob(seed, fold)
        receipt = _validate_local_job(root, job)
        assert receipt is not None
        path = Path(receipt["artifacts"]["test_predictions"]["path"])
        block = pd.read_parquet(path).copy()
        block["fold"] = fold
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    joined = frame.merge(
        manifest[["slide_id", "patient_id", "target_label", "cohort", "k_fold"]],
        on="slide_id",
        validate="one_to_one",
    )
    if (
        len(joined) != 185
        or joined["slide_id"].astype(str).nunique() != 185
        or joined["patient_id"].astype(str).nunique() != 159
        or not np.array_equal(
            pd.to_numeric(joined["label"]).to_numpy(int),
            pd.to_numeric(joined["target_label"]).to_numpy(int),
        )
        or not np.array_equal(
            pd.to_numeric(joined["fold"]).to_numpy(int),
            pd.to_numeric(joined["k_fold"]).to_numpy(int),
        )
    ):
        raise ContractError(f"seed{seed}: local OOF partition/label census drifted")
    return joined[
        ["slide_id", "patient_id", "cohort", "target_label", "fold", "logit", "prob_1"]
    ].sort_values("slide_id", kind="mergesort").reset_index(drop=True)


def _local_scheduler_payload(root: Path, *, created_utc: str) -> dict[str, Any]:
    events = []
    for job in local_jobs():
        receipt = _validate_local_job(root, job)
        if receipt is None:
            raise ContractError(f"{job.key}: local fold is incomplete")
        events.append(
            {
                "job_key": job.key,
                "command": receipt["command"],
                **receipt["execution"],
            }
        )
    events.sort(key=lambda value: value["job_key"])
    peak = _parallel_peak(events)
    if peak != MAX_WORKERS:
        raise ContractError(f"local MIL observed physical-fit peak was {peak}, required 6")
    oof = {str(seed): _artifact(local_oof_path(root, seed)) for seed in MODEL_SEEDS}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_TARGET_INTERNAL_LOCAL_MIL_OOF",
        "created_utc": created_utc,
        "result_role": LOCAL_ROLE,
        "bound_label": "sample/recipe-bounded empirical local OOF bound",
        "configured_max_workers": MAX_WORKERS,
        "observed_peak_workers": peak,
        "num_workers_per_fit": DEFAULT_NUM_WORKERS,
        "fold_scoped_physical_fits": 25,
        "refits": 0,
        "external_validation_claim_permitted": False,
        "source_pack_pre_gate": _artifact(local_pack_gate_path(root, "pre")),
        "source_pack_post_gate": _artifact(local_pack_gate_path(root, "post")),
        "events": events,
        "oof_artifacts": oof,
    }


def _local_pack_gate_payload(root: Path, *, phase: str, created_utc: str) -> dict[str, Any]:
    source = validate_source(deep_pack=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": f"LOCAL_MIL_SOURCE_PACK_{phase.upper()}_DEEP_GATE_PASS",
        "created_utc": created_utc,
        "phase": phase,
        "source_feature_store": source["feature_store"],
        "pack_stat_snapshot": _pack_stat_snapshot(source["feature_store"]),
        "source_main_model_writes": 0,
    }


def _verify_local_pack_gate(root: Path, phase: str) -> dict[str, Any]:
    stored = _read_json(local_pack_gate_path(root, phase))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError(f"local pack {phase} gate lacks created_utc")
    expected = _local_pack_gate_payload(root, phase=phase, created_utc=created)
    if stored != expected:
        raise ContractError(f"local pack {phase} gate does not replay")
    return stored


def _validate_local_scheduler(root: Path) -> dict[str, Any]:
    pre = _verify_local_pack_gate(root, "pre")
    post = _verify_local_pack_gate(root, "post")
    _assert_local_pack_gates_match(pre, post)
    stored = _read_json(local_scheduler_path(root))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("local scheduler receipt lacks created_utc")
    expected = _local_scheduler_payload(root, created_utc=created)
    if stored != expected:
        raise ContractError("local MIL scheduler does not replay exactly")
    return stored


def _assert_local_pack_gates_match(
    pre: Mapping[str, Any], post: Mapping[str, Any]
) -> None:
    if (
        pre.get("source_feature_store") != post.get("source_feature_store")
        or pre.get("pack_stat_snapshot") != post.get("pack_stat_snapshot")
    ):
        raise ContractError("local MIL source pack pre/post gates disagree")


def train_local(
    root: Path,
    *,
    apply: bool,
    max_workers: int,
    num_workers: int,
) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    _validate_target_open(output)
    if max_workers != MAX_WORKERS or num_workers != DEFAULT_NUM_WORKERS:
        raise ContractError("local MIL requires --max-workers 6 --num-workers 4")
    commands = [_local_command(output, job, num_workers=num_workers) for job in local_jobs()]
    if not apply:
        return {
            "status": "DRY_RUN_TARGET_INTERNAL_LOCAL_MIL",
            "jobs": 25,
            "fold_scoped_physical_fits": 25,
            "refits": 0,
            "max_workers": MAX_WORKERS,
            "commands": commands,
        }
    if local_scheduler_path(output).exists():
        return _validate_local_scheduler(output)
    pre_gate = local_pack_gate_path(output, "pre")
    if pre_gate.exists():
        _verify_local_pack_gate(output, "pre")
    else:
        _write_json_once(
            pre_gate,
            _local_pack_gate_payload(output, phase="pre", created_utc=_utcnow()),
        )

    def run(command: Sequence[str]) -> int:
        return int(subprocess.run(list(command), cwd=REPO, check=False).returncode)

    pending = [
        (job, command)
        for job, command in zip(local_jobs(), commands, strict=True)
        if _validate_local_job(output, job) is None
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run, command) for _job, command in pending]
        returncodes = [future.result() for future in concurrent.futures.as_completed(futures)]
    if any(returncodes):
        raise ContractError("one or more fold-scoped local MIL fits failed")
    for seed in MODEL_SEEDS:
        frame = _assemble_local_oof(output, seed)
        path = local_oof_path(output, seed)
        if path.exists():
            observed = pd.read_parquet(path)
            if not observed.equals(frame):
                raise ContractError(f"seed{seed}: persisted local OOF drifted")
        else:
            _write_parquet_once(path, frame)
    post_gate = local_pack_gate_path(output, "post")
    if post_gate.exists():
        _verify_local_pack_gate(output, "post")
    else:
        _write_json_once(
            post_gate,
            _local_pack_gate_payload(output, phase="post", created_utc=_utcnow()),
        )
    pre_value = _read_json(pre_gate)
    post_value = _read_json(post_gate)
    _assert_local_pack_gates_match(pre_value, post_value)
    payload = _local_scheduler_payload(output, created_utc=_utcnow())
    _write_json_once(local_scheduler_path(output), payload)
    return _validate_local_scheduler(output)


# ---------------------------------------------------------------------------
# Governed sharded two-method adaptation implementation.  These definitions
# intentionally supersede the initial single-process implementation above.


def _support_for_procedure(
    data: Mapping[str, Mapping[int, pd.DataFrame]],
    *,
    layout_seed: int,
    fold: int,
    support_regime: str,
    support_per_class: int,
    draw: int,
) -> list[tuple[str, str, int]]:
    if support_regime not in SUPPORT_REGIMES or support_per_class not in SUPPORT_PER_CLASS:
        raise ContractError("unknown support regime/budget")
    reference = {cohort: data[cohort][MODEL_SEEDS[0]] for cohort in ("RIH", "SurGen")}
    folds = {
        cohort: adapter.stratified_folds(frame["label"].to_numpy(int), layout_seed)
        for cohort, frame in reference.items()
    }
    support: list[tuple[str, str, int]] = []
    if support_regime in {"RIH_ONLY", "SURGEN_ONLY"}:
        cohort = "RIH" if support_regime == "RIH_ONLY" else "SurGen"
        frame = reference[cohort]
        labels = frame["label"].to_numpy(int)
        train = np.flatnonzero(folds[cohort] != fold)
        support_seed = adapter.stable_seed(
            "aim2_v1_support",
            layout_seed,
            fold,
            support_regime,
            support_per_class,
            draw,
        )
        indices = adapter.draw_exact_support(
            labels, train, 2 * support_per_class, seed=support_seed
        )
        support.extend(
            (cohort, str(frame.index[index]), int(labels[index])) for index in indices
        )
    else:
        if support_per_class % 2:
            raise ContractError("combined support requires an even k per class")
        per_cohort_class = support_per_class // 2
        for cohort in ("RIH", "SurGen"):
            frame = reference[cohort]
            labels = frame["label"].to_numpy(int)
            train = np.flatnonzero(folds[cohort] != fold)
            rng = np.random.default_rng(
                adapter.stable_seed(
                    "aim2_v1_combined_support",
                    layout_seed,
                    fold,
                    support_per_class,
                    draw,
                    cohort,
                )
            )
            for value in (0, 1):
                candidates = train[labels[train] == value]
                if len(candidates) < per_cohort_class:
                    raise ContractError("combined exact cohort-by-class support is infeasible")
                indices = rng.choice(candidates, per_cohort_class, replace=False)
                support.extend(
                    (cohort, str(frame.index[index]), int(value)) for index in indices
                )
    if (
        len(support) != 2 * support_per_class
        or len({(cohort, patient) for cohort, patient, _label in support}) != len(support)
        or [label for _cohort, _patient, label in support].count(0) != support_per_class
        or [label for _cohort, _patient, label in support].count(1) != support_per_class
    ):
        raise ContractError("exact support size/class contract failed")
    if support_regime == "COMBINED":
        for cohort in ("RIH", "SurGen"):
            block = [label for source, _patient, label in support if source == cohort]
            if block.count(0) != support_per_class // 2 or block.count(1) != support_per_class // 2:
                raise ContractError("combined support is not exactly cohort-by-class balanced")
    for cohort, patient, _label in support:
        reference_index = list(reference[cohort].index.astype(str))
        position = reference_index.index(patient)
        if int(folds[cohort][position]) == fold:
            raise ContractError("support/test leakage")
    return support


def _fit_probe_method(
    data: Mapping[str, Mapping[int, pd.DataFrame]],
    *,
    model_seed: int,
    support: Sequence[tuple[str, str, int]],
    method: str,
    inner_seed: int,
    ledger: Any,
    context: Mapping[str, Any],
) -> tuple[np.ndarray, float, dict[str, Any], dict[str, Any]]:
    if method not in {"source_anchored_residual_linear_probe", "pure_ridge_linear_probe"}:
        raise ContractError(f"unknown few-shot method: {method}")
    embedding_columns = [f"e{index}" for index in range(EMBED_DIM)]
    features = np.vstack(
        [
            data[cohort][model_seed].loc[patient, embedding_columns].to_numpy(float)
            for cohort, patient, _label in support
        ]
    )
    labels = np.asarray([label for _cohort, _patient, label in support], dtype=int)
    native = np.asarray(
        [
            float(data[cohort][model_seed].loc[patient, "eta_native"])
            for cohort, patient, _label in support
        ],
        dtype=float,
    )
    offset = native if method == "source_anchored_residual_linear_probe" else np.zeros_like(native)
    selected, selection = adapter.select_lambda(
        features,
        offset,
        labels,
        inner_seed=inner_seed,
        ledger=ledger,
        context={**dict(context), "method": method},
    )
    weights, bias, diagnostic = adapter.fit_residual(
        features,
        offset,
        labels,
        selected,
        ledger=ledger,
        context={**dict(context), "method": method, "fit_role": "outer_refit"},
    )
    return weights, bias, selection, diagnostic


def _run_few_shard(
    data: Mapping[str, Mapping[int, pd.DataFrame]], job: AdaptJob
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    assert job.kind == "few" and job.support_regime is not None
    methods = ("source_anchored_residual_linear_probe", "pure_ridge_linear_probe")
    ledgers = {method: adapter.SolverLedger() for method in methods}
    rows: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    support_records: list[dict[str, Any]] = []
    embedding_columns = [f"e{index}" for index in range(EMBED_DIM)]
    reference = {cohort: data[cohort][MODEL_SEEDS[0]] for cohort in ("RIH", "SurGen")}
    fold_layout = {
        cohort: adapter.stratified_folds(frame["label"].to_numpy(int), job.layout_seed)
        for cohort, frame in reference.items()
    }
    for support_per_class in SUPPORT_PER_CLASS:
        for draw in range(SUPPORT_DRAWS_PER_LAYOUT):
            prediction = {
                method: {
                    cohort: {
                        seed: np.full(len(reference[cohort]), np.nan, dtype=float)
                        for seed in MODEL_SEEDS
                    }
                    for cohort in ("RIH", "SurGen")
                }
                for method in methods
            }
            for fold in FOLDS:
                support = _support_for_procedure(
                    data,
                    layout_seed=job.layout_seed,
                    fold=fold,
                    support_regime=job.support_regime,
                    support_per_class=support_per_class,
                    draw=draw,
                )
                support_keys = [f"{cohort}::{patient}" for cohort, patient, _ in support]
                support_labels = [label for _cohort, _patient, label in support]
                test_ids = {
                    cohort: [
                        str(reference[cohort].index[index])
                        for index in np.flatnonzero(fold_layout[cohort] == fold)
                    ]
                    for cohort in ("RIH", "SurGen")
                }
                if any(set(support_keys) & {f"{cohort}::{pid}" for pid in ids}
                       for cohort, ids in test_ids.items()):
                    raise ContractError("few-shot support overlaps outer test")
                support_records.append(
                    {
                        "layout_seed": job.layout_seed,
                        "outer_fold": fold,
                        "support_regime": job.support_regime,
                        "support_per_class": support_per_class,
                        "support_total": 2 * support_per_class,
                        "draw": draw,
                        "support_patient_keys": support_keys,
                        "support_labels": support_labels,
                        "test_patient_ids_by_cohort": test_ids,
                        "shared_across_methods": True,
                        "shared_across_model_seeds": True,
                    }
                )
                for model_seed in MODEL_SEEDS:
                    for method in methods:
                        context = {
                            "phase": "few_shot",
                            "cohort": job.support_regime,
                            "outer_seed": job.layout_seed,
                            "outer_fold": fold,
                            "model_seed": model_seed,
                            "support": 2 * support_per_class,
                            "draw": draw,
                        }
                        weights, bias, selection, diagnostic = _fit_probe_method(
                            data,
                            model_seed=model_seed,
                            support=support,
                            method=method,
                            inner_seed=job.layout_seed + fold,
                            ledger=ledgers[method],
                            context=context,
                        )
                        for test_cohort in ("RIH", "SurGen"):
                            test_index = np.flatnonzero(fold_layout[test_cohort] == fold)
                            test_frame = data[test_cohort][model_seed]
                            features = test_frame.iloc[test_index][embedding_columns].to_numpy(float)
                            native = test_frame.iloc[test_index]["eta_native"].to_numpy(float)
                            eta = features @ weights + bias
                            if method == "source_anchored_residual_linear_probe":
                                eta = native + eta
                            prediction[method][test_cohort][model_seed][test_index] = eta
                        fit_records.append(
                            {
                                "phase": "few_shot",
                                "method": method,
                                "model_kind": method,
                                "anchor_kind": (
                                    "frozen_native_logit"
                                    if method == "source_anchored_residual_linear_probe"
                                    else "zero_logit"
                                ),
                                "infinite_lambda_semantics": (
                                    "frozen_native_predictor"
                                    if method == "source_anchored_residual_linear_probe"
                                    else "zero-logit no-information pure probe"
                                ),
                                "coefficient_order": "e0..e511",
                                "layout_seed": job.layout_seed,
                                "outer_fold": fold,
                                "model_seed": model_seed,
                                "support_regime": job.support_regime,
                                "support_per_class": support_per_class,
                                "support_total": 2 * support_per_class,
                                "draw": draw,
                                "fit_patient_keys": support_keys,
                                "fit_labels": support_labels,
                                "test_patient_ids_by_cohort": test_ids,
                                "selected_lambda": selection["selected_lambda"],
                                "inner_selection": selection,
                                "coefficients": weights.tolist(),
                                "bias": bias,
                                "solver_diagnostic": diagnostic,
                            }
                        )
            for cohort in ("RIH", "SurGen"):
                native_by_seed = {
                    seed: data[cohort][seed]["eta_native"].to_numpy(float)
                    for seed in MODEL_SEEDS
                }
                for method in methods:
                    if any(
                        not np.isfinite(prediction[method][cohort][seed]).all()
                        for seed in MODEL_SEEDS
                    ):
                        raise ContractError("few-shot OOF prediction coverage is incomplete")
                for index, patient_id in enumerate(reference[cohort].index.astype(str)):
                    rows.append(
                        {
                            "phase": "few_shot",
                            "layout_seed": job.layout_seed,
                            "support_regime": job.support_regime,
                            "support_per_class": support_per_class,
                            "support_total": 2 * support_per_class,
                            "draw": draw,
                            "test_cohort": cohort,
                            "patient_id": patient_id,
                            "label": int(reference[cohort].iloc[index]["label"]),
                            "fold": int(fold_layout[cohort][index]),
                            "eta_native": float(np.mean([native_by_seed[s][index] for s in MODEL_SEEDS])),
                            "eta_source_anchored_residual_linear_probe": float(
                                np.mean([prediction[methods[0]][cohort][s][index] for s in MODEL_SEEDS])
                            ),
                            "eta_pure_ridge_linear_probe": float(
                                np.mean([prediction[methods[1]][cohort][s][index] for s in MODEL_SEEDS])
                            ),
                            **{
                                f"eta_native_seed{seed}": float(native_by_seed[seed][index])
                                for seed in MODEL_SEEDS
                            },
                            **{
                                f"eta_{method}_seed{seed}": float(
                                    prediction[method][cohort][seed][index]
                                )
                                for method in methods
                                for seed in MODEL_SEEDS
                            },
                        }
                    )
    expected_rows = SUPPORT_DRAWS_PER_LAYOUT * len(SUPPORT_PER_CLASS) * 159
    if (
        len(rows) != expected_rows
        or len(fit_records) != 3_000
        or len(support_records) != 300
        or {record["method"] for record in fit_records}
        != {"source_anchored_residual_linear_probe", "pure_ridge_linear_probe"}
    ):
        raise ContractError(f"{job.key}: few-shot shard census drifted")
    solver = {
        "by_method": {
            method: {
                **ledger.summary(),
                "infinite_lambda_semantics": (
                    "frozen_native_predictor"
                    if method == "source_anchored_residual_linear_probe"
                    else "zero-logit no-information pure probe"
                ),
            }
            for method, ledger in ledgers.items()
        }
    }
    solver["total_solver_calls"] = int(
        sum(
            block["n_finite_calls"] + block["n_native_exact_calls"]
            for block in solver["by_method"].values()
        )
    )
    if any(block.get("n_unaccepted") != 0 for block in solver["by_method"].values()):
        raise ContractError(f"{job.key}: unaccepted solver result")
    return pd.DataFrame(rows), fit_records, support_records, solver


def _run_full_shard(
    data: Mapping[str, Mapping[int, pd.DataFrame]], job: AdaptJob
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    assert job.kind == "full" and job.cohort is not None
    ledger = adapter.SolverLedger()
    with _adapter_protocol(primary_outer_seed=PRIMARY_OUTER_SEED):
        _blocks, rows, fits = adapter.run_full_layout(
            {job.cohort: data[job.cohort]}, outer_seed=job.layout_seed, ledger=ledger
        )
        adapter.assert_full_label_lambda_grid_closed(fits)
    if (
        len(rows) != len(data[job.cohort][MODEL_SEEDS[0]])
        or sum(record["model_kind"] == "residual_adapter" for record in fits) != 25
        or sum(record["model_kind"] == "platt" for record in fits) != 5
    ):
        raise ContractError(f"{job.key}: full-label shard census drifted")
    solver = ledger.summary()
    if solver.get("n_unaccepted") != 0:
        raise ContractError(f"{job.key}: unaccepted full-label solver result")
    return pd.DataFrame(rows), fits, [], solver


def _adapt_command(root: Path, job: AdaptJob) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_adapt-one",
        "--output-root",
        str(root),
        "--kind",
        job.kind,
        "--layout-seed",
        str(job.layout_seed),
    ]
    if job.kind == "few":
        command.extend(["--support-regime", str(job.support_regime)])
    else:
        command.extend(["--cohort", str(job.cohort)])
    return command


def _adapt_shard_payload(
    root: Path, job: AdaptJob, *, created_utc: str, execution: Mapping[str, Any]
) -> dict[str, Any]:
    directory = adapt_shard_dir(root, job)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_TARGET_INTERNAL_ADAPTATION_SHARD",
        "created_utc": created_utc,
        "job": adapter.json_ready(job.__dict__),
        "command": _adapt_command(root, job),
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "contract": _artifact(contract_path(root)),
        "target_open_receipt": _artifact(target_open_path(root)),
        "oof": _artifact(directory / "oof.parquet"),
        "fit_ledger": _artifact(directory / "fits.jsonl"),
        "support_ledger": _artifact(directory / "supports.jsonl"),
        "solver": _artifact(directory / "solver.json"),
        "execution_record": _artifact(directory / "execution.json"),
        "execution": dict(execution),
    }


def _validate_selected_lambda(record: Mapping[str, Any]) -> None:
    selection = record.get("inner_selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "scheme",
        "inner_seed",
        "n_pool",
        "n_splits",
        "losses_by_lambda",
        "mean_loss_by_lambda",
        "solver_summary_by_lambda",
        "selected_lambda",
    }:
        raise ContractError("few-shot inner-selection schema drifted")
    n_pool = int(record["support_total"])
    lambda_keys = [adapter.lambda_key(value) for value in LAMBDA_GRID]
    losses = selection["losses_by_lambda"]
    means = selection["mean_loss_by_lambda"]
    if (
        selection["scheme"] != "leave_one_out"
        or selection["inner_seed"] != record["layout_seed"] + record["outer_fold"]
        or selection["n_pool"] != n_pool
        or selection["n_splits"] != n_pool
        or set(losses) != set(lambda_keys)
        or set(means) != set(lambda_keys)
        or set(selection["solver_summary_by_lambda"]) != set(lambda_keys)
        or any(len(losses[key]) != n_pool for key in lambda_keys)
        or any(
            not math.isclose(float(means[key]), float(np.mean(losses[key])), abs_tol=1e-15)
            for key in lambda_keys
        )
    ):
        raise ContractError("few-shot inner-selection layout/loss replay failed")
    best = min(float(means[key]) for key in lambda_keys)
    selected = next(key for key in lambda_keys if float(means[key]) <= best + 1e-12)
    if selection["selected_lambda"] != selected or record["selected_lambda"] != selected:
        raise ContractError("few-shot selected lambda is not the prespecified first minimum")
    diagnostic = record.get("solver_diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise ContractError("few-shot final solver diagnostic missing")
    coefficients = np.asarray(record["coefficients"], dtype=float)
    if selected == "infinity":
        if (
            diagnostic.get("status") != "native_exact"
            or not np.array_equal(coefficients, np.zeros(EMBED_DIM))
            or float(record["bias"]) != 0.0
        ):
            raise ContractError("infinite-lambda exact anchor replay failed")
    elif (
        diagnostic.get("status") != "finite_optimum"
        or not math.isclose(float(diagnostic.get("lambda")), float(selected), abs_tol=0.0)
        or diagnostic.get("objective_decrease", -1.0) < -adapter.OBJECTIVE_TOL
    ):
        raise ContractError("finite selected-lambda diagnostic replay failed")


def _validate_few_shard_science(
    root: Path,
    job: AdaptJob,
    frame: pd.DataFrame,
    fits: Sequence[Mapping[str, Any]],
    supports: Sequence[Mapping[str, Any]],
) -> None:
    assert job.kind == "few" and job.support_regime is not None
    methods = ("source_anchored_residual_linear_probe", "pure_ridge_linear_probe")
    expected_frame_columns = [
        "phase",
        "layout_seed",
        "support_regime",
        "support_per_class",
        "support_total",
        "draw",
        "test_cohort",
        "patient_id",
        "label",
        "fold",
        "eta_native",
        "eta_source_anchored_residual_linear_probe",
        "eta_pure_ridge_linear_probe",
        *(f"eta_native_seed{seed}" for seed in MODEL_SEEDS),
        *(
            f"eta_{method}_seed{seed}"
            for method in methods
            for seed in MODEL_SEEDS
        ),
    ]
    if list(frame.columns) != expected_frame_columns:
        raise ContractError(f"{job.key}: few-shot OOF exact schema drifted")
    numeric_columns = [name for name in frame if name.startswith("eta_")]
    if (
        set(frame["phase"].astype(str)) != {"few_shot"}
        or set(pd.to_numeric(frame["layout_seed"])) != {job.layout_seed}
        or set(frame["support_regime"].astype(str)) != {job.support_regime}
        or set(pd.to_numeric(frame["support_per_class"]).astype(int)) != set(SUPPORT_PER_CLASS)
        or set(frame["test_cohort"].astype(str)) != {"RIH", "SurGen"}
        or not np.isfinite(frame[numeric_columns].to_numpy(float)).all()
    ):
        raise ContractError(f"{job.key}: few-shot OOF values/identities drifted")
    data = _patient_adapter_data(root)
    support_key_fields = (
        "layout_seed",
        "outer_fold",
        "support_regime",
        "support_per_class",
        "support_total",
        "draw",
        "support_patient_keys",
        "support_labels",
        "test_patient_ids_by_cohort",
        "shared_across_methods",
        "shared_across_model_seeds",
    )
    support_map: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    for record in supports:
        if set(record) != set(support_key_fields):
            raise ContractError(f"{job.key}: support-ledger schema drifted")
        key = (
            int(record["support_per_class"]), int(record["draw"]), int(record["outer_fold"])
        )
        if key in support_map:
            raise ContractError(f"{job.key}: duplicated support procedure/fold")
        expected_support = _support_for_procedure(
            data,
            layout_seed=job.layout_seed,
            fold=key[2],
            support_regime=job.support_regime,
            support_per_class=key[0],
            draw=key[1],
        )
        expected_keys = [f"{cohort}::{patient}" for cohort, patient, _ in expected_support]
        expected_labels = [label for _cohort, _patient, label in expected_support]
        expected_tests = {
            cohort: [
                str(data[cohort][MODEL_SEEDS[0]].index[index])
                for index in np.flatnonzero(
                    adapter.stratified_folds(
                        data[cohort][MODEL_SEEDS[0]]["label"].to_numpy(int), job.layout_seed
                    )
                    == key[2]
                )
            ]
            for cohort in ("RIH", "SurGen")
        }
        if (
            record["layout_seed"] != job.layout_seed
            or record["support_regime"] != job.support_regime
            or record["support_total"] != 2 * key[0]
            or record["support_patient_keys"] != expected_keys
            or record["support_labels"] != expected_labels
            or record["test_patient_ids_by_cohort"] != expected_tests
            or record["shared_across_methods"] is not True
            or record["shared_across_model_seeds"] is not True
            or any(
                set(expected_keys) & {f"{cohort}::{patient}" for patient in patients}
                for cohort, patients in expected_tests.items()
            )
        ):
            raise ContractError(f"{job.key}: deterministic support/test replay failed at {key}")
        support_map[key] = record
    if set(support_map) != {
        (support, draw, fold)
        for support in SUPPORT_PER_CLASS
        for draw in range(SUPPORT_DRAWS_PER_LAYOUT)
        for fold in FOLDS
    }:
        raise ContractError(f"{job.key}: support procedure roster drifted")
    fit_map: dict[tuple[int, int, int, int, str], Mapping[str, Any]] = {}
    exact_fit_keys = {
        "phase",
        "method",
        "model_kind",
        "anchor_kind",
        "infinite_lambda_semantics",
        "coefficient_order",
        "layout_seed",
        "outer_fold",
        "model_seed",
        "support_regime",
        "support_per_class",
        "support_total",
        "draw",
        "fit_patient_keys",
        "fit_labels",
        "test_patient_ids_by_cohort",
        "selected_lambda",
        "inner_selection",
        "coefficients",
        "bias",
        "solver_diagnostic",
    }
    for record in fits:
        if set(record) != exact_fit_keys:
            raise ContractError(f"{job.key}: fit-ledger exact schema drifted")
        key = (
            int(record["support_per_class"]),
            int(record["draw"]),
            int(record["outer_fold"]),
            int(record["model_seed"]),
            str(record["method"]),
        )
        if key in fit_map:
            raise ContractError(f"{job.key}: duplicate fitted head")
        support = support_map[key[:3]]
        expected_anchor = (
            ("frozen_native_logit", "frozen_native_predictor")
            if key[4] == methods[0]
            else ("zero_logit", "zero-logit no-information pure probe")
        )
        if (
            key[3] not in MODEL_SEEDS
            or key[4] not in methods
            or record["phase"] != "few_shot"
            or record["model_kind"] != key[4]
            or (record["anchor_kind"], record["infinite_lambda_semantics"])
            != expected_anchor
            or record["coefficient_order"] != "e0..e511"
            or record["layout_seed"] != job.layout_seed
            or record["support_regime"] != job.support_regime
            or record["support_total"] != 2 * key[0]
            or record["fit_patient_keys"] != support["support_patient_keys"]
            or record["fit_labels"] != support["support_labels"]
            or record["test_patient_ids_by_cohort"] != support["test_patient_ids_by_cohort"]
            or len(record["coefficients"]) != EMBED_DIM
            or not np.isfinite(np.asarray([*record["coefficients"], record["bias"]], float)).all()
        ):
            raise ContractError(f"{job.key}: fitted-head support/method replay failed at {key}")
        _validate_selected_lambda(record)
        if record["selected_lambda"] != "infinity":
            embedding_columns = [f"e{index}" for index in range(EMBED_DIM)]
            parsed = [item.split("::", 1) for item in record["fit_patient_keys"]]
            features = np.vstack(
                [
                    data[cohort][key[3]].loc[patient, embedding_columns].to_numpy(float)
                    for cohort, patient in parsed
                ]
            )
            labels = np.asarray(record["fit_labels"], dtype=int)
            native = np.asarray(
                [
                    data[cohort][key[3]].loc[patient, "eta_native"]
                    for cohort, patient in parsed
                ],
                dtype=float,
            )
            offset = native if key[4] == methods[0] else np.zeros_like(native)
            theta = np.asarray([*record["coefficients"], record["bias"]], dtype=float)
            value, gradient = adapter.residual_objective_and_gradient(
                theta, features, offset, labels, float(record["selected_lambda"])
            )
            zero, _ = adapter.residual_objective_and_gradient(
                np.zeros_like(theta), features, offset, labels, float(record["selected_lambda"])
            )
            diagnostic = record["solver_diagnostic"]
            gradient_inf = float(np.max(np.abs(gradient)))
            if (
                not math.isclose(value, float(diagnostic["objective_at_fit"]), abs_tol=1e-12)
                or not math.isclose(zero, float(diagnostic["objective_at_zero"]), abs_tol=1e-12)
                or not math.isclose(
                    zero - value, float(diagnostic["objective_decrease"]), abs_tol=1e-12
                )
                or not math.isclose(
                    gradient_inf, float(diagnostic["gradient_inf_norm"]), abs_tol=1e-10
                )
                or (not diagnostic.get("scipy_success") and gradient_inf > adapter.SOLVER_GRAD_TOL)
            ):
                raise ContractError(f"{job.key}: final optimizer diagnostic does not replay")
        fit_map[key] = record
    expected_fit_keys = {
        (support, draw, fold, seed, method)
        for support in SUPPORT_PER_CLASS
        for draw in range(SUPPORT_DRAWS_PER_LAYOUT)
        for fold in FOLDS
        for seed in MODEL_SEEDS
        for method in methods
    }
    if set(fit_map) != expected_fit_keys:
        raise ContractError(f"{job.key}: exact fitted-head roster drifted")
    embedding_columns = [f"e{index}" for index in range(EMBED_DIM)]
    for row in frame.to_dict(orient="records"):
        support = int(row["support_per_class"])
        draw = int(row["draw"])
        cohort = str(row["test_cohort"])
        patient = str(row["patient_id"])
        reference = data[cohort][MODEL_SEEDS[0]]
        if patient not in reference.index:
            raise ContractError(f"{job.key}: prediction patient is not in target roster")
        label = int(reference.loc[patient, "label"])
        fold = int(adapter.stratified_folds(reference["label"].to_numpy(int), job.layout_seed)[
            list(reference.index.astype(str)).index(patient)
        ])
        if row["label"] != label or row["fold"] != fold:
            raise ContractError(f"{job.key}: prediction label/fold drifted")
        native_values = []
        for seed in MODEL_SEEDS:
            seed_frame = data[cohort][seed]
            native = float(seed_frame.loc[patient, "eta_native"])
            native_values.append(native)
            if not math.isclose(float(row[f"eta_native_seed{seed}"]), native, abs_tol=0.0):
                raise ContractError(f"{job.key}: per-seed native logit drifted")
            features = seed_frame.loc[patient, embedding_columns].to_numpy(float)
            for method in methods:
                record = fit_map[(support, draw, fold, seed, method)]
                expected_eta = float(features @ np.asarray(record["coefficients"], float) + record["bias"])
                if method == methods[0]:
                    expected_eta += native
                if not math.isclose(
                    float(row[f"eta_{method}_seed{seed}"]), expected_eta, rel_tol=0, abs_tol=1e-12
                ):
                    raise ContractError(f"{job.key}: fitted-head test application drifted")
        if not math.isclose(float(row["eta_native"]), float(np.mean(native_values)), abs_tol=1e-15):
            raise ContractError(f"{job.key}: native ensemble drifted")
        for method in methods:
            expected_ensemble = float(
                np.mean([row[f"eta_{method}_seed{seed}"] for seed in MODEL_SEEDS])
            )
            if not math.isclose(float(row[f"eta_{method}"]), expected_ensemble, abs_tol=1e-15):
                raise ContractError(f"{job.key}: five-seed adapted ensemble drifted")


def _validate_full_selection(record: Mapping[str, Any]) -> None:
    selection = record.get("inner_selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "scheme",
        "inner_seed",
        "n_pool",
        "n_splits",
        "losses_by_lambda",
        "mean_loss_by_lambda",
        "solver_summary_by_lambda",
        "selected_lambda",
    }:
        raise ContractError("full-label inner-selection schema drifted")
    lambda_keys = [adapter.lambda_key(value) for value in LAMBDA_GRID]
    losses = selection["losses_by_lambda"]
    means = selection["mean_loss_by_lambda"]
    if (
        selection["scheme"] != "five_fold"
        or selection["n_splits"] != len(FOLDS)
        or set(losses) != set(lambda_keys)
        or set(means) != set(lambda_keys)
        or set(selection["solver_summary_by_lambda"]) != set(lambda_keys)
        or any(len(losses[key]) != len(FOLDS) for key in lambda_keys)
        or any(
            not math.isclose(float(means[key]), float(np.mean(losses[key])), abs_tol=1e-15)
            for key in lambda_keys
        )
    ):
        raise ContractError("full-label inner-selection loss/grid replay failed")
    best = min(float(means[key]) for key in lambda_keys)
    selected = next(key for key in lambda_keys if float(means[key]) <= best + 1e-12)
    if selection["selected_lambda"] != selected or record["selected_lambda"] != selected:
        raise ContractError("full-label selected lambda is not the first grid minimum")


def _platt_objective_gradient(
    theta: np.ndarray, native: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray]:
    eta = float(theta[0]) + float(theta[1]) * native
    probability = adapter.expit(eta)
    residual_value = probability - labels
    return (
        float(np.mean(np.logaddexp(0.0, eta) - labels * eta)),
        np.asarray(
            [
                float(np.mean(residual_value)),
                float(np.mean(residual_value * native)),
            ]
        ),
    )


def _validate_full_shard_science(
    root: Path,
    job: AdaptJob,
    frame: pd.DataFrame,
    fits: Sequence[Mapping[str, Any]],
) -> None:
    assert job.kind == "full" and job.cohort is not None
    expected_columns = [
        "phase",
        "outer_seed",
        "is_primary",
        "support_requested",
        "draw",
        "cohort",
        "patient_id",
        "label",
        "fold",
        "eta_native",
        "eta_adapted",
        "eta_platt",
        *(f"eta_native_seed{seed}" for seed in MODEL_SEEDS),
        *(f"eta_adapted_seed{seed}" for seed in MODEL_SEEDS),
    ]
    if list(frame.columns) != expected_columns:
        raise ContractError(f"{job.key}: full-label OOF exact schema drifted")
    data = _patient_adapter_data(root)
    reference = data[job.cohort][MODEL_SEEDS[0]]
    patient_ids = list(reference.index.astype(str))
    labels = reference["label"].to_numpy(int)
    folds = adapter.stratified_folds(labels, job.layout_seed)
    if (
        len(frame) != len(reference)
        or frame["patient_id"].astype(str).duplicated().any()
        or set(frame["patient_id"].astype(str)) != set(patient_ids)
        or set(frame["cohort"].astype(str)) != {job.cohort}
        or set(pd.to_numeric(frame["outer_seed"])) != {job.layout_seed}
        or set(frame["phase"].astype(str)) != {"full_label"}
        or set(frame["is_primary"].astype(bool))
        != {job.layout_seed == PRIMARY_OUTER_SEED}
        or not frame["support_requested"].isna().all()
        or not frame["draw"].isna().all()
        or not np.isfinite(
            frame[[name for name in frame if name.startswith("eta_")]].to_numpy(float)
        ).all()
    ):
        raise ContractError(f"{job.key}: full-label OOF roster/value drifted")
    residual = [record for record in fits if record.get("model_kind") == "residual_adapter"]
    platt = [record for record in fits if record.get("model_kind") == "platt"]
    if len(residual) != 25 or len(platt) != 5:
        raise ContractError(f"{job.key}: full-label fit kind census drifted")
    exact_fit_keys = {
        "phase",
        "model_kind",
        "cohort",
        "outer_seed",
        "outer_fold",
        "model_seed",
        "support_requested",
        "support_realized",
        "draw",
        "support_seed",
        "fit_patient_ids",
        "fit_labels",
        "n_fit",
        "n_fit_class0",
        "n_fit_class1",
        "test_patient_ids",
        "n_test",
        "selected_lambda",
        "inner_selection",
        "coefficient_order",
        "coefficients",
        "bias",
        "solver_diagnostic",
    }
    if any(set(record) != exact_fit_keys for record in fits):
        raise ContractError(f"{job.key}: full-label fit-record schema drifted")
    residual_map: dict[tuple[int, int], Mapping[str, Any]] = {}
    for record in residual:
        key = (int(record["outer_fold"]), int(record["model_seed"]))
        if key in residual_map or key[0] not in FOLDS or key[1] not in MODEL_SEEDS:
            raise ContractError(f"{job.key}: residual fitted-head key drifted")
        train = np.flatnonzero(folds != key[0])
        test = np.flatnonzero(folds == key[0])
        expected_train_ids = [patient_ids[index] for index in train]
        expected_test_ids = [patient_ids[index] for index in test]
        if (
            record["phase"] != "full_label"
            or record["cohort"] != job.cohort
            or record["outer_seed"] != job.layout_seed
            or record["fit_patient_ids"] != expected_train_ids
            or record["fit_labels"] != labels[train].astype(int).tolist()
            or record["test_patient_ids"] != expected_test_ids
            or set(record["fit_patient_ids"]) & set(record["test_patient_ids"])
            or record["n_fit"] != len(train)
            or record["n_fit_class0"] != int((labels[train] == 0).sum())
            or record["n_fit_class1"] != int((labels[train] == 1).sum())
            or record["n_test"] != len(test)
            or record["support_requested"] is not None
            or record["support_realized"] is not None
            or record["draw"] is not None
            or record["support_seed"] is not None
            or record["coefficient_order"] != "e0..e511"
            or len(record["coefficients"]) != EMBED_DIM
            or record["inner_selection"]["n_pool"] != len(train)
            or record["inner_selection"]["inner_seed"] != job.layout_seed + key[0]
        ):
            raise ContractError(f"{job.key}: residual train/test/label replay failed")
        _validate_full_selection(record)
        if record["selected_lambda"] != "infinity":
            embedding_columns = [f"e{index}" for index in range(EMBED_DIM)]
            seed_frame = data[job.cohort][key[1]]
            features = seed_frame.iloc[train][embedding_columns].to_numpy(float)
            offset = seed_frame.iloc[train]["eta_native"].to_numpy(float)
            theta = np.asarray([*record["coefficients"], record["bias"]], dtype=float)
            value, gradient = adapter.residual_objective_and_gradient(
                theta,
                features,
                offset,
                labels[train],
                float(record["selected_lambda"]),
            )
            zero, _ = adapter.residual_objective_and_gradient(
                np.zeros_like(theta),
                features,
                offset,
                labels[train],
                float(record["selected_lambda"]),
            )
            diagnostic = record["solver_diagnostic"]
            if (
                not math.isclose(value, float(diagnostic["objective_at_fit"]), abs_tol=1e-12)
                or not math.isclose(zero, float(diagnostic["objective_at_zero"]), abs_tol=1e-12)
                or not math.isclose(
                    float(np.max(np.abs(gradient))),
                    float(diagnostic["gradient_inf_norm"]),
                    abs_tol=1e-10,
                )
            ):
                raise ContractError(f"{job.key}: full residual optimizer replay failed")
        residual_map[key] = record
    if set(residual_map) != {(fold, seed) for fold in FOLDS for seed in MODEL_SEEDS}:
        raise ContractError(f"{job.key}: full residual head roster drifted")
    platt_map: dict[int, Mapping[str, Any]] = {}
    for record in platt:
        fold = int(record["outer_fold"])
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        if (
            fold in platt_map
            or fold not in FOLDS
            or record["model_seed"] is not None
            or record["cohort"] != job.cohort
            or record["outer_seed"] != job.layout_seed
            or record["fit_patient_ids"] != [patient_ids[index] for index in train]
            or record["fit_labels"] != labels[train].astype(int).tolist()
            or record["test_patient_ids"] != [patient_ids[index] for index in test]
            or record["coefficient_order"] != ["intercept", "slope"]
            or len(record["coefficients"]) != 2
            or record["inner_selection"] is not None
            or record["selected_lambda"] is not None
            or record["support_requested"] is not None
            or record["support_realized"] is not None
            or record["draw"] is not None
            or record["support_seed"] is not None
            or record["n_fit"] != len(train)
            or record["n_fit_class0"] != int((labels[train] == 0).sum())
            or record["n_fit_class1"] != int((labels[train] == 1).sum())
            or record["n_test"] != len(test)
        ):
            raise ContractError(f"{job.key}: Platt train/test replay failed")
        intercept, slope = map(float, record["coefficients"])
        if not math.isclose(float(record["bias"]), intercept, abs_tol=0.0):
            raise ContractError(f"{job.key}: Platt bias/intercept identity drifted")
        native_train = np.mean(
            np.vstack(
                [
                    data[job.cohort][seed].iloc[train]["eta_native"].to_numpy(float)
                    for seed in MODEL_SEEDS
                ]
            ),
            axis=0,
        )

        value, gradient = _platt_objective_gradient(
            np.asarray([intercept, slope]), native_train, labels[train]
        )
        zero, _ = _platt_objective_gradient(
            np.asarray([0.0, 1.0]), native_train, labels[train]
        )
        diagnostic = record["solver_diagnostic"]
        gradient_inf = float(np.max(np.abs(gradient)))
        if (
            diagnostic.get("status") != "finite_optimum"
            or not math.isclose(value, float(diagnostic["objective_at_fit"]), abs_tol=1e-12)
            or not math.isclose(zero, float(diagnostic["objective_at_zero"]), abs_tol=1e-12)
            or not math.isclose(
                zero - value, float(diagnostic["objective_decrease"]), abs_tol=1e-12
            )
            or not math.isclose(
                gradient_inf, float(diagnostic["gradient_inf_norm"]), abs_tol=1e-10
            )
            or (not diagnostic.get("scipy_success") and gradient_inf > adapter.SOLVER_GRAD_TOL)
        ):
            raise ContractError(f"{job.key}: Platt optimizer diagnostic does not replay")
        platt_map[fold] = record
    if set(platt_map) != set(FOLDS):
        raise ContractError(f"{job.key}: Platt fold roster drifted")
    embedding_columns = [f"e{index}" for index in range(EMBED_DIM)]
    frame_by_patient = frame.assign(patient_id=frame["patient_id"].astype(str)).set_index("patient_id")
    for position, patient in enumerate(patient_ids):
        row = frame_by_patient.loc[patient]
        fold = int(folds[position])
        if int(row["label"]) != int(labels[position]) or int(row["fold"]) != fold:
            raise ContractError(f"{job.key}: full-label row label/fold drifted")
        native_values = []
        adapted_values = []
        for seed in MODEL_SEEDS:
            seed_frame = data[job.cohort][seed]
            native = float(seed_frame.loc[patient, "eta_native"])
            features = seed_frame.loc[patient, embedding_columns].to_numpy(float)
            record = residual_map[(fold, seed)]
            adapted = native + float(
                features @ np.asarray(record["coefficients"], float) + record["bias"]
            )
            native_values.append(native)
            adapted_values.append(adapted)
            if (
                not math.isclose(float(row[f"eta_native_seed{seed}"]), native, abs_tol=0.0)
                or not math.isclose(
                    float(row[f"eta_adapted_seed{seed}"]), adapted, rel_tol=0, abs_tol=1e-12
                )
            ):
                raise ContractError(f"{job.key}: full per-seed prediction replay failed")
        native_ensemble = float(np.mean(native_values))
        adapted_ensemble = float(np.mean(adapted_values))
        platt_record = platt_map[fold]
        intercept, slope = map(float, platt_record["coefficients"])
        expected_platt = intercept + slope * native_ensemble
        if (
            not math.isclose(float(row["eta_native"]), native_ensemble, abs_tol=1e-15)
            or not math.isclose(float(row["eta_adapted"]), adapted_ensemble, abs_tol=1e-15)
            or not math.isclose(float(row["eta_platt"]), expected_platt, abs_tol=1e-12)
        ):
            raise ContractError(f"{job.key}: full ensemble/Platt replay failed")
    point = float(roc_auc_score(labels, frame_by_patient.loc[patient_ids, "eta_native"]))
    if not math.isclose(point, EXPECTED_NATIVE[job.cohort]["auroc"], abs_tol=1e-15):
        raise ContractError(f"{job.key}: full-label native authority drifted")
    with _adapter_protocol(primary_outer_seed=PRIMARY_OUTER_SEED):
        adapter.assert_full_label_lambda_grid_closed(list(fits))


def _validate_adapt_shard(root: Path, job: AdaptJob) -> dict[str, Any] | None:
    directory = adapt_shard_dir(root, job)
    completion_path = adapt_shard_receipt_path(root, job)
    if not directory.exists():
        return None
    required_files = {
        "oof.parquet",
        "fits.jsonl",
        "supports.jsonl",
        "solver.json",
        "execution.json",
        "completion.json",
    }
    if (
        not directory.is_dir()
        or directory.is_symlink()
        or set(path.name for path in directory.iterdir()) != required_files
        or any(not (directory / name).is_file() or (directory / name).is_symlink() for name in required_files)
    ):
        raise ContractError(f"{job.key}: partial adaptation shard")
    execution = _read_json(directory / "execution.json")
    execution_keys = {
        "job_id",
        "started_utc",
        "completed_utc",
        "started_unix_ns",
        "completed_unix_ns",
        "returncode",
        "configured_workers",
    }
    if (
        set(execution) != execution_keys
        or execution.get("job_id") != f"adaptation.{job.key}"
        or execution.get("returncode") != 0
        or execution.get("configured_workers") != 1
        or execution.get("completed_unix_ns", 0) <= execution.get("started_unix_ns", 0)
        or _parse_utc(execution.get("completed_utc"), context=f"{job.key}/completed")
        <= _parse_utc(execution.get("started_utc"), context=f"{job.key}/started")
    ):
        raise ContractError(f"{job.key}: adaptation execution record drifted")
    stored = _read_json(completion_path)
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError(f"{job.key}: adaptation shard lacks creation time")
    expected = _adapt_shard_payload(root, job, created_utc=created, execution=execution)
    if stored != expected:
        raise ContractError(f"{job.key}: adaptation shard receipt does not replay")
    frame = pd.read_parquet(directory / "oof.parquet")
    fits = _read_jsonl(directory / "fits.jsonl")
    supports = _read_jsonl(directory / "supports.jsonl")
    solver = _read_json(directory / "solver.json")
    if job.kind == "few":
        methods = {"source_anchored_residual_linear_probe", "pure_ridge_linear_probe"}
        key_columns = [
            "layout_seed", "support_regime", "support_per_class", "draw",
            "test_cohort", "patient_id",
        ]
        if (
            len(frame) != SUPPORT_DRAWS_PER_LAYOUT * len(SUPPORT_PER_CLASS) * 159
            or frame.duplicated(key_columns).any()
            or len(fits) != 3_000
            or len(supports) != 300
            or {record.get("method") for record in fits} != methods
            or any(record.get("coefficient_order") != "e0..e511" for record in fits)
            or any(len(record.get("coefficients", [])) != EMBED_DIM for record in fits)
            or any(record.get("selected_lambda") not in {adapter.lambda_key(x) for x in LAMBDA_GRID} for record in fits)
        ):
            raise ContractError(f"{job.key}: few-shot shard replay failed")
        _validate_few_shard_science(root, job, frame, fits, supports)
        paired: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
        for record in fits:
            key = (
                record["layout_seed"], record["outer_fold"], record["model_seed"],
                record["support_regime"], record["support_per_class"], record["draw"],
            )
            paired.setdefault(key, {})[record["method"]] = record
        if len(paired) != 1_500:
            raise ContractError(f"{job.key}: method-paired fit census drifted")
        for key, values in paired.items():
            if set(values) != methods:
                raise ContractError(f"{job.key}: method pair missing at {key}")
            residual, pure = values["source_anchored_residual_linear_probe"], values["pure_ridge_linear_probe"]
            if (
                residual["fit_patient_keys"] != pure["fit_patient_keys"]
                or residual["fit_labels"] != pure["fit_labels"]
                or residual["test_patient_ids_by_cohort"] != pure["test_patient_ids_by_cohort"]
                or residual["inner_selection"]["scheme"] != pure["inner_selection"]["scheme"]
                or residual["inner_selection"]["inner_seed"] != pure["inner_selection"]["inner_seed"]
            ):
                raise ContractError(f"{job.key}: methods are not paired on support/splits")
    else:
        if len(frame) != TARGETS[
            "rih_metastatic" if job.cohort == "RIH" else "surgen_metastatic"
        ].patients or len(fits) != 30 or supports:
            raise ContractError(f"{job.key}: full-label shard replay failed")
        _validate_full_shard_science(root, job, frame, fits)
    if job.kind == "few":
        if (
            set(solver) != {"by_method", "total_solver_calls"}
            or solver.get("total_solver_calls") != 451_000
            or set(solver.get("by_method", {}))
            != {"source_anchored_residual_linear_probe", "pure_ridge_linear_probe"}
            or any(
                block.get("n_unaccepted") != 0
                or block.get("n_finite_calls", 0) + block.get("n_native_exact_calls", 0)
                != 225_500
                for block in solver["by_method"].values()
            )
        ):
            raise ContractError(f"{job.key}: few-shot solver accounting drifted")
    elif (
        solver.get("n_unaccepted") != 0
        or solver.get("n_finite_calls", 0) + solver.get("n_native_exact_calls", 0)
        != 2_030
    ):
        raise ContractError(f"{job.key}: full-label solver accounting drifted")
    return stored


def _adapt_one(root: Path, job: AdaptJob) -> None:
    output = validate_output_root(root, must_exist=True)
    _validate_target_open(output)
    if _validate_adapt_shard(output, job) is not None:
        return
    directory = adapt_shard_dir(output, job)
    if directory.exists() or directory.is_symlink():
        raise ContractError(f"{job.key}: partial adaptation shard")
    data = _patient_adapter_data(output)
    started_unix_ns = time.time_ns()
    started_utc = _utcnow_precise()
    if job.kind == "few":
        frame, fits, supports, solver = _run_few_shard(data, job)
    else:
        frame, fits, supports, solver = _run_full_shard(data, job)
    directory.mkdir(parents=True, exist_ok=False)
    _write_parquet_once(directory / "oof.parquet", frame)
    _write_jsonl_once(directory / "fits.jsonl", fits)
    _write_jsonl_once(directory / "supports.jsonl", supports)
    _write_json_once(directory / "solver.json", solver)
    completed_utc = _utcnow_precise()
    completed_unix_ns = time.time_ns()
    execution = {
        "job_id": f"adaptation.{job.key}",
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": completed_unix_ns,
        "returncode": 0,
        "configured_workers": 1,
    }
    _write_json_once(directory / "execution.json", execution)
    _write_json_once(
        adapt_shard_receipt_path(output, job),
        _adapt_shard_payload(output, job, created_utc=_utcnow(), execution=execution),
    )
    _validate_adapt_shard(output, job)


def _procedure_matrix_summary(
    frame: pd.DataFrame,
    *,
    procedure_columns: Sequence[str],
    cohort_column: str,
    score_columns: Mapping[str, str],
    expected_procedures: int,
    seed_score_columns: Mapping[str, str] | None,
    random_seed: int,
) -> dict[str, Any]:
    """Paired patient+procedure bootstrap without averaging predictions first."""

    methods = tuple(score_columns)
    if methods[0] != "native" or len(methods) < 2:
        raise ContractError("procedure summary requires native first and at least one comparator")
    working = frame.copy()
    working["_procedure"] = working[list(procedure_columns)].astype(str).agg(":".join, axis=1)
    cohorts = ("RIH", "SurGen")
    matrices: dict[str, dict[str, np.ndarray]] = {}
    labels_by_cohort: dict[str, np.ndarray] = {}
    procedures_ref: list[str] | None = None
    for cohort in cohorts:
        block = working.loc[working[cohort_column].astype(str).eq(cohort)].copy()
        patients = sorted(block["patient_id"].astype(str).unique())
        procedures = sorted(block["_procedure"].astype(str).unique())
        if procedures_ref is None:
            procedures_ref = procedures
        elif procedures != procedures_ref:
            raise ContractError("cohort procedure rosters are not paired")
        label_table = block[["patient_id", "label"]].drop_duplicates()
        if label_table["patient_id"].astype(str).duplicated().any():
            raise ContractError("procedure summary patient labels drifted")
        label_table["patient_id"] = label_table["patient_id"].astype(str)
        labels = label_table.set_index("patient_id").loc[patients, "label"].to_numpy(int)
        labels_by_cohort[cohort] = labels
        matrices[cohort] = {}
        for method, column in score_columns.items():
            pivot = block.pivot(index="_procedure", columns="patient_id", values=column)
            matrix = pivot.loc[procedures, patients].to_numpy(float)
            if matrix.shape != (len(procedures), len(patients)) or not np.isfinite(matrix).all():
                raise ContractError("procedure prediction matrix is incomplete/nonfinite")
            matrices[cohort][method] = matrix
    assert procedures_ref is not None
    n_procedures = len(procedures_ref)
    if n_procedures != expected_procedures:
        raise ContractError(
            f"procedure census is {n_procedures}, expected {expected_procedures}"
        )
    for cohort in cohorts:
        if not np.allclose(matrices[cohort]["native"], matrices[cohort]["native"][:1], rtol=0, atol=0):
            raise ContractError("frozen native predictions changed across procedures")
    metric_names = ("auroc", "log_loss", "brier")

    def row_metrics(y: np.ndarray, eta: np.ndarray) -> dict[str, np.ndarray]:
        auroc = e2c_stats._auroc_rows(y, eta)  # noqa: SLF001
        probability = np.clip(adapter.expit(eta), adapter.PROB_EPS, 1 - adapter.PROB_EPS)
        yy = y[None, :]
        return {
            "auroc": np.asarray(auroc, dtype=float),
            "log_loss": np.mean(
                -(yy * np.log(probability) + (1 - yy) * np.log(1 - probability)), axis=1
            ),
            "brier": np.mean((probability - yy) ** 2, axis=1),
        }

    points: dict[str, dict[str, dict[str, float]]] = {}
    per_procedure: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for cohort in cohorts:
        per_procedure[cohort] = {
            method: row_metrics(labels_by_cohort[cohort], matrices[cohort][method])
            for method in methods
        }
        points[cohort] = {
            method: {
                metric: float(np.mean(per_procedure[cohort][method][metric]))
                for metric in metric_names
            }
            for method in methods
        }
    points["macro"] = {
        method: {
            metric: float(np.mean([points[c][method][metric] for c in cohorts]))
            for metric in metric_names
        }
        for method in methods
    }
    for cohort in cohorts:
        expected_native = EXPECTED_NATIVE[cohort]["auroc"]
        if not math.isclose(points[cohort]["native"]["auroc"], expected_native, abs_tol=1e-15):
            raise ContractError(f"{cohort}: procedure summary native authority drifted")
    scopes = (*cohorts, "macro")
    draws: dict[str, dict[str, list[float]]] = {
        scope: {
            **{f"method:{method}:{metric}": [] for method in methods for metric in metric_names},
            **{
                f"contrast:{method}_minus_native:{metric}": []
                for method in methods[1:]
                for metric in metric_names
            },
            **(
                {
                    f"contrast:{methods[1]}_minus_{methods[2]}:{metric}": []
                    for metric in metric_names
                }
                if len(methods) == 3
                else {}
            ),
        }
        for scope in scopes
    }
    rng = np.random.default_rng(random_seed)
    for _ in range(N_BOOTSTRAP):
        boot_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for cohort in cohorts:
            y = labels_by_cohort[cohort]
            patient_index = e2c_stats._bootstrap_patient_indices(y, rng)  # noqa: SLF001
            boot_metrics[cohort] = {
                method: row_metrics(
                    y[patient_index], matrices[cohort][method][:, patient_index]
                )
                for method in methods
            }
        procedure_index = rng.integers(0, n_procedures, n_procedures)
        for scope in scopes:
            for method in methods:
                for metric in metric_names:
                    if scope == "macro":
                        value = float(
                            np.mean(
                                [
                                    np.mean(boot_metrics[c][method][metric][procedure_index])
                                    for c in cohorts
                                ]
                            )
                        )
                    else:
                        value = float(
                            np.mean(boot_metrics[scope][method][metric][procedure_index])
                        )
                    draws[scope][f"method:{method}:{metric}"].append(value)
            for method in methods[1:]:
                for metric in metric_names:
                    a = draws[scope][f"method:{method}:{metric}"][-1]
                    b = draws[scope][f"method:native:{metric}"][-1]
                    draws[scope][f"contrast:{method}_minus_native:{metric}"].append(a - b)
            if len(methods) == 3:
                for metric in metric_names:
                    a = draws[scope][f"method:{methods[1]}:{metric}"][-1]
                    b = draws[scope][f"method:{methods[2]}:{metric}"][-1]
                    draws[scope][f"contrast:{methods[1]}_minus_{methods[2]}:{metric}"].append(a - b)

    def interval(values: Sequence[float]) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    output: dict[str, Any] = {
        "estimand": "mean of per-procedure metrics; predictions never averaged across support procedures",
        "procedures": n_procedures,
        "bootstrap_draws": N_BOOTSTRAP,
        "patient_resampling": "within cohort",
        "procedure_resampling": "paired across methods and cohorts",
        "contrast_sign": (
            "A_minus_B for all metrics; positive AUROC favors A, while negative "
            "log-loss/Brier favors A"
        ),
        "per_cohort": {},
        "equal_cohort_macro": {},
    }
    for scope in scopes:
        target: dict[str, Any] = {"methods": {}, "contrasts": {}}
        for method in methods:
            target["methods"][method] = {
                metric: {
                    "point": points[scope][method][metric],
                    "ci95": interval(draws[scope][f"method:{method}:{metric}"]),
                }
                for metric in metric_names
            }
            if scope != "macro":
                target["methods"][method]["auroc_procedure_sample_sd"] = float(
                    np.std(per_procedure[scope][method]["auroc"], ddof=1)
                )
        for method in methods[1:]:
            name = f"{method}_minus_native"
            target["contrasts"][name] = {
                metric: {
                    "point": points[scope][method][metric] - points[scope]["native"][metric],
                    "ci95": interval(draws[scope][f"contrast:{name}:{metric}"]),
                }
                for metric in metric_names
            }
        if len(methods) == 3:
            name = f"{methods[1]}_minus_{methods[2]}"
            target["contrasts"][name] = {
                metric: {
                    "point": points[scope][methods[1]][metric] - points[scope][methods[2]][metric],
                    "ci95": interval(draws[scope][f"contrast:{name}:{metric}"]),
                }
                for metric in metric_names
            }
        if scope == "macro":
            output["equal_cohort_macro"] = target
        else:
            labels = labels_by_cohort[scope]
            target["census"] = {
                "patients": int(len(labels)), "mutant": int(labels.sum())
            }
            output["per_cohort"][scope] = target
    if seed_score_columns:
        seed_values: dict[str, dict[str, dict[str, float]]] = {
            scope: {} for scope in scopes
        }
        for method, template in seed_score_columns.items():
            per_seed_by_cohort: dict[str, dict[str, float]] = {cohort: {} for cohort in cohorts}
            for cohort in cohorts:
                block = working.loc[working[cohort_column].astype(str).eq(cohort)].copy()
                patients = sorted(block["patient_id"].astype(str).unique())
                procedures = sorted(block["_procedure"].astype(str).unique())
                labels = labels_by_cohort[cohort]
                for seed in MODEL_SEEDS:
                    column = template.format(seed=seed)
                    if column not in block:
                        raise ContractError(f"missing per-source-seed score column: {column}")
                    matrix = (
                        block.pivot(index="_procedure", columns="patient_id", values=column)
                        .loc[procedures, patients]
                        .to_numpy(float)
                    )
                    per_seed_by_cohort[cohort][str(seed)] = float(
                        np.mean(e2c_stats._auroc_rows(labels, matrix))  # noqa: SLF001
                    )
                values = list(per_seed_by_cohort[cohort].values())
                seed_values[cohort][method] = {
                    "by_seed": per_seed_by_cohort[cohort],
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)),
                }
            macro_by_seed = {
                str(seed): float(
                    np.mean(
                        [per_seed_by_cohort[cohort][str(seed)] for cohort in cohorts]
                    )
                )
                for seed in MODEL_SEEDS
            }
            seed_values["macro"][method] = {
                "by_seed": macro_by_seed,
                "mean": float(np.mean(list(macro_by_seed.values()))),
                "sample_sd": float(np.std(list(macro_by_seed.values()), ddof=1)),
            }
        for cohort in cohorts:
            output["per_cohort"][cohort]["per_source_seed_expected_auroc"] = seed_values[cohort]
        output["equal_cohort_macro"]["per_source_seed_expected_auroc"] = seed_values["macro"]
    return output


def _adapter_aggregate_paths(root: Path) -> dict[str, Path]:
    return {
        "few_shot_oof": adapter_root(root) / "few_shot_oof.parquet",
        "full_label_oof": adapter_root(root) / "full_label_oof.parquet",
        "results": adapter_root(root) / "results.json",
        "completion": adapter_root(root) / "completion.json",
    }


def _write_or_reconcile_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        if not pd.read_parquet(path).equals(frame):
            raise ContractError(f"immutable aggregate parquet drifted: {path}")
    else:
        _write_parquet_once(path, frame)


def _collect_adaptation_shards(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    few_frames: list[pd.DataFrame] = []
    full_frames: list[pd.DataFrame] = []
    for job in adapt_jobs():
        if _validate_adapt_shard(root, job) is None:
            raise ContractError(f"{job.key}: adaptation shard missing")
        frame = pd.read_parquet(adapt_shard_dir(root, job) / "oof.parquet")
        (few_frames if job.kind == "few" else full_frames).append(frame)
    few = pd.concat(few_frames, ignore_index=True).sort_values(
        ["support_per_class", "support_regime", "layout_seed", "draw", "test_cohort", "patient_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    full = pd.concat(full_frames, ignore_index=True).sort_values(
        ["outer_seed", "cohort", "patient_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(few) != 143_100 or len(full) != 795:
        raise ContractError(f"aggregate adaptation OOF census drifted: {len(few)}/{len(full)}")
    return few, full


def _adapt_scheduler_payload(root: Path, *, created_utc: str) -> dict[str, Any]:
    events = []
    for job in adapt_jobs():
        receipt = _validate_adapt_shard(root, job)
        if receipt is None:
            raise ContractError(f"{job.key}: missing adaptation shard")
        events.append(
            {"job_key": job.key, "command": receipt["command"], **receipt["execution"]}
        )
    events.sort(key=lambda event: event["job_key"])
    peak = _parallel_peak(events)
    if peak != MAX_WORKERS:
        raise ContractError(f"adaptation observed process peak was {peak}, required 6")
    paths = _adapter_aggregate_paths(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_TARGET_INTERNAL_TWO_METHOD_ADAPTATION_SCHEDULER",
        "created_utc": created_utc,
        "configured_max_workers": MAX_WORKERS,
        "observed_peak_workers": peak,
        "shard_jobs": len(adapt_jobs()),
        "few_shot_final_decisions_per_method": 22_500,
        "few_shot_final_decisions_both_methods": 45_000,
        "few_shot_solver_calls_per_method": 3_382_500,
        "full_label_residual_decisions": 250,
        "full_label_platt_calibration_fits": 50,
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "events": events,
        "few_shot_oof": _artifact(paths["few_shot_oof"]),
        "full_label_oof": _artifact(paths["full_label_oof"]),
    }


def _validate_adapt_scheduler(root: Path) -> dict[str, Any]:
    stored = _read_json(adapt_scheduler_path(root))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("adaptation scheduler lacks created_utc")
    expected = _adapt_scheduler_payload(root, created_utc=created)
    if stored != expected:
        raise ContractError("adaptation scheduler does not replay exactly")
    return stored


def _build_adapter_results(root: Path) -> dict[str, Any]:
    paths = _adapter_aggregate_paths(root)
    few = pd.read_parquet(paths["few_shot_oof"])
    full = pd.read_parquet(paths["full_label_oof"])
    if len(few) != 143_100 or len(full) != 795:
        raise ContractError("adaptation aggregate row census drifted")
    few_summary: dict[str, Any] = {}
    for support_per_class in SUPPORT_PER_CLASS:
        few_summary[str(support_per_class)] = {}
        for regime in SUPPORT_REGIMES:
            block = few.loc[
                pd.to_numeric(few["support_per_class"]).eq(support_per_class)
                & few["support_regime"].astype(str).eq(regime)
            ].copy()
            few_summary[str(support_per_class)][regime] = _procedure_matrix_summary(
                block,
                procedure_columns=("layout_seed", "draw"),
                cohort_column="test_cohort",
                score_columns={
                    "native": "eta_native",
                    "source_anchored_residual_linear_probe": "eta_source_anchored_residual_linear_probe",
                    "pure_ridge_linear_probe": "eta_pure_ridge_linear_probe",
                },
                expected_procedures=SUPPORT_PROCEDURES_PER_CELL,
                seed_score_columns={
                    "native": "eta_native_seed{seed}",
                    "source_anchored_residual_linear_probe": (
                        "eta_source_anchored_residual_linear_probe_seed{seed}"
                    ),
                    "pure_ridge_linear_probe": "eta_pure_ridge_linear_probe_seed{seed}",
                },
                random_seed=adapter.stable_seed(
                    "two_method_summary", support_per_class, regime, BOOTSTRAP_SEED
                ),
            )
    full_summary = _procedure_matrix_summary(
        full,
        procedure_columns=("outer_seed",),
        cohort_column="cohort",
        score_columns={"native": "eta_native", "source_anchored_residual_linear_probe": "eta_adapted", "platt": "eta_platt"},
        expected_procedures=len(OUTER_LAYOUT_SEEDS),
        seed_score_columns={
            "native": "eta_native_seed{seed}",
            "source_anchored_residual_linear_probe": "eta_adapted_seed{seed}",
        },
        random_seed=adapter.stable_seed("full_label_summary", BOOTSTRAP_SEED),
    )
    full_layout_descriptive: dict[str, Any] = {}
    for layout_seed in OUTER_LAYOUT_SEEDS:
        layout = full.loc[pd.to_numeric(full["outer_seed"]).eq(layout_seed)]
        per_cohort: dict[str, Any] = {}
        for cohort in ("RIH", "SurGen"):
            block = layout.loc[layout["cohort"].astype(str).eq(cohort)]
            labels = block["label"].to_numpy(int)
            per_cohort[cohort] = {
                method: adapter.point_metrics(labels, block[column].to_numpy(float))
                for method, column in {
                    "native": "eta_native",
                    "source_anchored_residual_linear_probe": "eta_adapted",
                    "platt": "eta_platt",
                }.items()
            }
        full_layout_descriptive[str(layout_seed)] = {
            "per_cohort": per_cohort,
            "equal_cohort_macro": {
                method: {
                    metric: float(
                        np.mean([per_cohort[c][method][metric] for c in ("RIH", "SurGen")])
                    )
                    for metric in ("auroc", "log_loss", "brier")
                }
                for method in ("native", "source_anchored_residual_linear_probe", "platt")
            },
        }
    native_authority = _source_native_analysis_authority()
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "COMPLETE_TARGET_INTERNAL_TWO_METHOD_ADAPTATION",
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "source_anchor": {
            "development_population": "TCGA+SurGen primaries only",
            "encoder": "UNI-v1",
            "model_seeds": list(MODEL_SEEDS),
            "encoder_mil_and_source_classifier_frozen": True,
            "native_authority": native_authority,
            "native_exact_gate": EXPECTED_NATIVE,
        },
        "few_shot": {
            "methods": ["source_anchored_residual_linear_probe", "pure_ridge_linear_probe"],
            "support_regimes": list(SUPPORT_REGIMES),
            "test_cohorts": ["RIH", "SurGen"],
            "support_per_class": list(SUPPORT_PER_CLASS),
            "all_methods_regimes_budgets_reported": True,
            "method_regime_or_budget_selected_on_target_results": False,
            "positive_repair_claim_permitted": False,
            "reason_positive_claim_withheld": (
                "prespecified multi-dose/multi-regime target-internal analysis; descriptive only"
            ),
            "cells": few_summary,
        },
        "full_label_empirical_adapter_bound": {
            "role": (
                "sample-bounded empirical cross-fitted full-label residual-adapter bound; "
                "target-internal, not external validation"
            ),
            "summary_across_five_layouts": full_summary,
            "per_layout_descriptive": full_layout_descriptive,
            "platt_role": "calibration comparator only; headline log-loss/Brier; never a ceiling",
        },
        "analytical_power_precision": analytical_power_precision_table(),
        "fit_and_solver_accounting": {
            "few_shot_unique_support_procedures_by_fold": 4_500,
            "few_shot_final_residual_decisions": 22_500,
            "few_shot_final_pure_ridge_decisions": 22_500,
            "few_shot_final_head_decisions_both_methods": 45_000,
            "few_shot_fit_to_test_cohort_applications": 90_000,
            "few_shot_solver_calls_per_method": 3_382_500,
            "few_shot_solver_calls_both_methods": 6_765_000,
            "full_label_final_residual_decisions": 250,
            "full_label_platt_calibration_fits": 50,
            "full_label_patient_procedure_rows": 795,
            "full_label_method_cohort_evaluations": 30,
        },
        "firewall": {
            "primary_support_used": False,
            "allowed_target_labels": ["RIH-M", "SurGen-M"],
            "source_checkpoint_encoder_mil_or_classifier_modified": False,
            "external_validation_claim_permitted": False,
            "surgen_metastatic_source_family_exposed": True,
            "prior_zero_shot_authority_sealed_before_target_labels": True,
            "adaptation_feedback_to_source_model_selection_or_external_claims": False,
        },
    }


def _adapter_completion_payload(root: Path, *, created_utc: str) -> dict[str, Any]:
    paths = _adapter_aggregate_paths(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_TARGET_INTERNAL_TWO_METHOD_ADAPTATION",
        "created_utc": created_utc,
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "target_open_receipt": _artifact(target_open_path(root)),
        "scheduler": _artifact(adapt_scheduler_path(root)),
        "few_shot_oof": _artifact(paths["few_shot_oof"]),
        "full_label_oof": _artifact(paths["full_label_oof"]),
        "results": _artifact(paths["results"]),
        "few_shot_rows": 143_100,
        "full_label_rows": 795,
        "few_shot_final_decisions": {
            "source_anchored_residual_linear_probe": 22_500,
            "pure_ridge_linear_probe": 22_500,
        },
    }


def _validate_adapter_completion(root: Path) -> dict[str, Any]:
    paths = _adapter_aggregate_paths(root)
    _validate_adapt_scheduler(root)
    stored_results = _read_json(paths["results"])
    expected_results = _build_adapter_results(root)
    if stored_results != expected_results:
        raise ContractError("adaptation results do not numerically replay from OOF/shards")
    stored = _read_json(paths["completion"])
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise ContractError("adaptation completion lacks created_utc")
    expected = _adapter_completion_payload(root, created_utc=created)
    if stored != expected:
        raise ContractError("adaptation completion does not replay exactly")
    return stored_results


def run_adaptation(
    root: Path, *, apply: bool, max_workers: int = MAX_WORKERS
) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    _validate_target_open(output)
    if max_workers != MAX_WORKERS:
        raise ContractError("governed adaptation requires exactly --max-workers 6")
    commands = [_adapt_command(output, job) for job in adapt_jobs()]
    if not apply:
        return {
            "status": "DRY_RUN_TARGET_INTERNAL_TWO_METHOD_ADAPTATION",
            "shards": len(adapt_jobs()),
            "max_workers": MAX_WORKERS,
            "final_decisions_per_method": 22_500,
            "solver_calls_per_method": 3_382_500,
            "commands": commands,
        }
    paths = _adapter_aggregate_paths(output)
    if paths["completion"].exists():
        return _validate_adapter_completion(output)

    def run(command: Sequence[str]) -> int:
        return int(subprocess.run(list(command), cwd=REPO, check=False).returncode)

    pending = [
        (job, command)
        for job, command in zip(adapt_jobs(), commands, strict=True)
        if _validate_adapt_shard(output, job) is None
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run, command) for _job, command in pending]
        returncodes = [future.result() for future in concurrent.futures.as_completed(futures)]
    if any(returncodes):
        raise ContractError("one or more deterministic adaptation shards failed")
    few, full = _collect_adaptation_shards(output)
    _write_or_reconcile_parquet(paths["few_shot_oof"], few)
    _write_or_reconcile_parquet(paths["full_label_oof"], full)
    if not adapt_scheduler_path(output).exists():
        _write_json_once(
            adapt_scheduler_path(output),
            _adapt_scheduler_payload(output, created_utc=_utcnow()),
        )
    _validate_adapt_scheduler(output)
    results = _build_adapter_results(output)
    if paths["results"].exists():
        if _read_json(paths["results"]) != results:
            raise ContractError("persisted adaptation results drifted")
    else:
        _write_json_once(paths["results"], results)
    if not paths["completion"].exists():
        _write_json_once(
            paths["completion"],
            _adapter_completion_payload(output, created_utc=_utcnow()),
        )
    return _validate_adapter_completion(output)


def _local_mil_analysis(root: Path) -> dict[str, Any]:
    _validate_local_scheduler(root)
    source_data = _patient_adapter_data(root)
    per_seed: dict[int, pd.DataFrame] = {}
    for seed in MODEL_SEEDS:
        slides = pd.read_parquet(local_oof_path(root, seed))
        if (
            slides.groupby("patient_id")["cohort"].nunique().ne(1).any()
            or slides.groupby("patient_id")["target_label"].nunique().ne(1).any()
        ):
            raise ContractError(f"seed{seed}: local OOF has patient cohort/label discordance")
        patient = (
            slides.groupby("patient_id", sort=True)
            .agg(
                cohort=("cohort", "first"),
                label=("target_label", "first"),
                eta_local=("logit", "mean"),
            )
        )
        native_parts = []
        for cohort in ("RIH", "SurGen"):
            block = source_data[cohort][seed][["label", "eta_native"]].copy()
            block["cohort"] = cohort
            native_parts.append(block)
        native = pd.concat(native_parts).sort_index()
        joined = patient.join(
            native[["label", "eta_native"]].rename(columns={"label": "source_label"}),
            how="inner",
            validate="one_to_one",
        )
        if (
            len(joined) != 159
            or int(joined["label"].sum()) != 67
            or not np.array_equal(
                joined["label"].to_numpy(int), joined["source_label"].to_numpy(int)
            )
            or not np.isfinite(joined[["eta_local", "eta_native"]].to_numpy(float)).all()
        ):
            raise ContractError(f"seed{seed}: local/native patient alignment drifted")
        per_seed[seed] = joined.drop(columns="source_label")
    reference = per_seed[MODEL_SEEDS[0]][["cohort", "label"]].copy()
    for seed in MODEL_SEEDS[1:]:
        if (
            not reference.index.equals(per_seed[seed].index)
            or not reference.equals(per_seed[seed][["cohort", "label"]])
        ):
            raise ContractError("local MIL patient roster differs across training seeds")
    combined = reference.copy()
    combined["eta_local"] = np.mean(
        np.vstack([per_seed[seed]["eta_local"].to_numpy(float) for seed in MODEL_SEEDS]), axis=0
    )
    combined["eta_native"] = np.mean(
        np.vstack([per_seed[seed]["eta_native"].to_numpy(float) for seed in MODEL_SEEDS]), axis=0
    )

    def auc(block: pd.DataFrame, column: str) -> float:
        return float(roc_auc_score(block["label"].to_numpy(int), block[column].to_numpy(float)))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scopes = {
        "RIH": combined.loc[combined["cohort"].eq("RIH")],
        "SurGen": combined.loc[combined["cohort"].eq("SurGen")],
        "pooled": combined,
    }
    draws = {
        scope: {"local": [], "native": [], "delta": []} for scope in (*scopes, "macro")
    }
    for _ in range(N_BOOTSTRAP):
        current: dict[str, tuple[float, float]] = {}
        for cohort in ("RIH", "SurGen"):
            block = scopes[cohort].reset_index(drop=True)
            index = e2c_stats._bootstrap_patient_indices(block["label"].to_numpy(int), rng)  # noqa: SLF001
            y = block["label"].to_numpy(int)[index]
            current[cohort] = (
                float(roc_auc_score(y, block["eta_local"].to_numpy(float)[index])),
                float(roc_auc_score(y, block["eta_native"].to_numpy(float)[index])),
            )
        block = scopes["pooled"].reset_index(drop=True)
        index = e2c_stats._bootstrap_patient_indices(block["label"].to_numpy(int), rng)  # noqa: SLF001
        y = block["label"].to_numpy(int)[index]
        current["pooled"] = (
            float(roc_auc_score(y, block["eta_local"].to_numpy(float)[index])),
            float(roc_auc_score(y, block["eta_native"].to_numpy(float)[index])),
        )
        current["macro"] = (
            float(np.mean([current[c][0] for c in ("RIH", "SurGen")])),
            float(np.mean([current[c][1] for c in ("RIH", "SurGen")])),
        )
        for scope, (local_value, native_value) in current.items():
            draws[scope]["local"].append(local_value)
            draws[scope]["native"].append(native_value)
            draws[scope]["delta"].append(local_value - native_value)

    def ci(values: Sequence[float]) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    result_scopes: dict[str, Any] = {}
    point_blocks = {**scopes, "macro": None}
    for scope, block in point_blocks.items():
        if scope == "macro":
            local_point = float(np.mean([auc(scopes[c], "eta_local") for c in ("RIH", "SurGen")]))
            native_point = float(np.mean([auc(scopes[c], "eta_native") for c in ("RIH", "SurGen")]))
            seed_aurocs = [
                float(
                    np.mean(
                        [
                            auc(per_seed[seed].loc[per_seed[seed]["cohort"].eq(c)], "eta_local")
                            for c in ("RIH", "SurGen")
                        ]
                    )
                )
                for seed in MODEL_SEEDS
            ]
            census = {"patients": 159, "mutant": 67, "weighting": "equal cohort"}
        else:
            assert block is not None
            local_point, native_point = auc(block, "eta_local"), auc(block, "eta_native")
            seed_aurocs = [
                auc(
                    per_seed[seed]
                    if scope == "pooled"
                    else per_seed[seed].loc[per_seed[seed]["cohort"].eq(scope)],
                    "eta_local",
                )
                for seed in MODEL_SEEDS
            ]
            census = {"patients": int(len(block)), "mutant": int(block["label"].sum())}
        result_scopes[scope] = {
            "census": census,
            "local_five_seed_mean_logit_ensemble": {
                "auroc": local_point, "ci95": ci(draws[scope]["local"])
            },
            "frozen_source_native_five_seed_ensemble": {
                "auroc": native_point,
                "ci95": ci(draws[scope]["native"]),
                "ci_role": "campaign paired-comparator bootstrap; not sealed zero-shot CI",
            },
            "local_minus_native": {
                "auroc_delta": local_point - native_point,
                "ci95": ci(draws[scope]["delta"]),
            },
            "per_training_seed_auroc": {
                "by_seed": {str(seed): value for seed, value in zip(MODEL_SEEDS, seed_aurocs, strict=True)},
                "mean": float(np.mean(seed_aurocs)),
                "sample_sd": float(np.std(seed_aurocs, ddof=1)),
            },
        }
    for cohort in ("RIH", "SurGen"):
        if not math.isclose(
            result_scopes[cohort]["frozen_source_native_five_seed_ensemble"]["auroc"],
            EXPECTED_NATIVE[cohort]["auroc"],
            abs_tol=1e-15,
        ):
            raise ContractError(f"{cohort}: local analysis native authority drifted")
    return {
        "result_role": LOCAL_ROLE,
        "bound_label": "sample/recipe-bounded empirical local OOF bound",
        "external_validation_claim_permitted": False,
        "training_seeds": list(MODEL_SEEDS),
        "fold_scoped_fits": 25,
        "refits": 0,
        "bootstrap_draws": N_BOOTSTRAP,
        "scopes": result_scopes,
    }


def _analysis_results(root: Path) -> dict[str, Any]:
    adaptation = _validate_adapter_completion(root)
    local = _local_mil_analysis(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "COMPLETE_AIM2_PRIORITY4_TARGET_INTERNAL_ANALYSIS",
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "source_development_population": "TCGA+SurGen primaries only",
        "zero_shot_external_results_modified": False,
        "adaptation": adaptation,
        "local_mil_empirical_bound": local,
    }


def _analysis_completion_payload(root: Path, *, created_utc: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_AIM2_PRIORITY4_TARGET_INTERNAL_ANALYSIS",
        "created_utc": created_utc,
        "result_role": INTERNAL_ROLE,
        "results": _artifact(analysis_root(root) / "results.json"),
        "adaptation_completion": _artifact(adapter_root(root) / "completion.json"),
        "local_scheduler": _artifact(local_scheduler_path(root)),
        "external_validation_claim_permitted": False,
    }


def analyze_campaign(root: Path, *, apply: bool) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    if not apply:
        return {
            "status": "DRY_RUN_ANALYSIS",
            "requires": ["adaptation completion", "local MIL scheduler"],
            "result_role": INTERNAL_ROLE,
        }
    results = _analysis_results(output)
    results_path = analysis_root(output) / "results.json"
    completion_path = analysis_root(output) / "completion.json"
    if results_path.exists():
        if _read_json(results_path) != results:
            raise ContractError("persisted terminal analysis results drifted")
    else:
        _write_json_once(results_path, results)
    if completion_path.exists():
        stored = _read_json(completion_path)
        created = stored.get("created_utc")
        if not isinstance(created, str) or stored != _analysis_completion_payload(
            output, created_utc=created
        ):
            raise ContractError("terminal analysis completion does not replay")
    else:
        _write_json_once(
            completion_path, _analysis_completion_payload(output, created_utc=_utcnow())
        )
    return results


def campaign_plan(root: Path) -> dict[str, Any]:
    output = validate_output_root(root)
    return {
        "campaign": CAMPAIGN,
        "output_root": str(output),
        "source_root_read_only": str(SOURCE_ROOT),
        "result_role": INTERNAL_ROLE,
        "production_sequence": [
            "prepare --apply",
            "preflight --apply",
            "score-source --apply --max-workers 6 --num-workers 4",
            "open-targets --apply",
            "adapt --apply --max-workers 6",
            "train-local --apply --max-workers 6 --num-workers 4",
            "analyze --apply",
            "verify",
        ],
        "fit_accounting": fit_accounting(),
        "adaptation": {
            "methods": [
                "source_anchored_residual_linear_probe",
                "pure_ridge_linear_probe",
            ],
            "support_regimes": list(SUPPORT_REGIMES),
            "test_cohorts_per_head": ["RIH-M", "SurGen-M"],
            "shards": len(adapt_jobs()),
            "max_workers": MAX_WORKERS,
        },
        "production_launch_authorized_by_this_command": False,
    }


def campaign_status(root: Path) -> dict[str, Any]:
    output = validate_output_root(root)
    stages = {
        "prepared": contract_path(output),
        "preflight": output / "receipts/deep_preflight.json",
        "source_inference_sealed": inference_seal_path(output),
        "target_internal_open": target_open_path(output),
        "adaptation_complete": adapter_root(output) / "completion.json",
        "local_mil_complete": local_scheduler_path(output),
        "analysis_complete": analysis_root(output) / "completion.json",
    }
    return {
        "output_root": str(output),
        "exists": output.is_dir(),
        "stages": {name: path.is_file() and not path.is_symlink() for name, path in stages.items()},
        "production_root_absent": not output.exists(),
    }


def verify_campaign(root: Path) -> dict[str, Any]:
    output = validate_output_root(root, must_exist=True)
    load_contract(output, deep_pack=True)
    verify_preflight(output, deep_pack=False)
    verify_inference_seal(output)
    _validate_target_open(output)
    _validate_adapter_completion(output)
    _validate_local_scheduler(output)
    results = _analysis_results(output)
    stored_results = _read_json(analysis_root(output) / "results.json")
    if stored_results != results:
        raise ContractError("terminal analysis does not numerically replay")
    completion = _read_json(analysis_root(output) / "completion.json")
    created = completion.get("created_utc")
    if not isinstance(created, str) or completion != _analysis_completion_payload(
        output, created_utc=created
    ):
        raise ContractError("terminal analysis completion does not replay")
    return {
        "status": "VERIFIED_COMPLETE",
        "campaign": CAMPAIGN,
        "result_role": INTERNAL_ROLE,
        "external_validation_claim_permitted": False,
        "analysis": _artifact(analysis_root(output) / "results.json"),
    }


def _print_json(value: Any) -> None:
    print(json.dumps(adapter.json_ready(value), indent=2, sort_keys=True, allow_nan=False))


def _add_output_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare", "preflight", "score-source", "open-targets", "adapt", "train-local", "analyze", "verify", "status"):
        command_parser = subparsers.add_parser(name)
        _add_output_root(command_parser)
        if name in {"prepare", "preflight", "score-source", "open-targets", "adapt", "train-local", "analyze"}:
            command_parser.add_argument("--apply", action="store_true")
        if name in {"score-source", "adapt", "train-local"}:
            command_parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
        if name in {"score-source", "train-local"}:
            command_parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    score_one = subparsers.add_parser("_score-one")
    _add_output_root(score_one)
    score_one.add_argument("--target", choices=TARGET_ORDER, required=True)
    score_one.add_argument("--seed", type=int, choices=MODEL_SEEDS, required=True)
    score_one.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    local_one = subparsers.add_parser("_train-local-one")
    _add_output_root(local_one)
    local_one.add_argument("--seed", type=int, choices=MODEL_SEEDS, required=True)
    local_one.add_argument("--fold", type=int, choices=FOLDS, required=True)
    local_one.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    adapt_one = subparsers.add_parser("_adapt-one")
    _add_output_root(adapt_one)
    adapt_one.add_argument("--kind", choices=("few", "full"), required=True)
    adapt_one.add_argument("--layout-seed", type=int, choices=OUTER_LAYOUT_SEEDS, required=True)
    adapt_one.add_argument("--support-regime", choices=SUPPORT_REGIMES)
    adapt_one.add_argument("--cohort", choices=("RIH", "SurGen"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.output_root
    if args.command == "plan":
        value = campaign_plan(root)
    elif args.command == "status":
        value = campaign_status(root)
    elif args.command == "prepare":
        value = prepare(root, apply=args.apply)
    elif args.command == "preflight":
        value = preflight(root, apply=args.apply, deep_pack=True)
    elif args.command == "score-source":
        value = score_source(
            root,
            apply=args.apply,
            max_workers=args.max_workers,
            num_workers=args.num_workers,
        )
    elif args.command == "open-targets":
        value = open_targets(root, apply=args.apply)
    elif args.command == "adapt":
        value = run_adaptation(root, apply=args.apply, max_workers=args.max_workers)
    elif args.command == "train-local":
        value = train_local(
            root,
            apply=args.apply,
            max_workers=args.max_workers,
            num_workers=args.num_workers,
        )
    elif args.command == "analyze":
        value = analyze_campaign(root, apply=args.apply)
    elif args.command == "verify":
        value = verify_campaign(root)
    elif args.command == "_score-one":
        _score_one(root, ScoreJob(args.target, args.seed), num_workers=args.num_workers)
        value = {"status": "COMPLETE", "job": f"{args.target}__seed{args.seed}"}
    elif args.command == "_train-local-one":
        _train_local_one(root, LocalJob(args.seed, args.fold), num_workers=args.num_workers)
        value = {"status": "COMPLETE", "job": f"seed{args.seed}__fold{args.fold}"}
    elif args.command == "_adapt-one":
        job = AdaptJob(
            args.kind,
            args.layout_seed,
            support_regime=args.support_regime,
            cohort=args.cohort,
        )
        _adapt_one(root, job)
        value = {"status": "COMPLETE", "job": job.key}
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _print_json(value)


if __name__ == "__main__":
    main()
