#!/usr/bin/env python3
"""Governed additive extension of canonical Aim-1 E0 from three to five seeds.

The historical seed-42/43/44 run tree is immutable.  This controller trains only
seeds 45 and 46 below a new overlay lineage, authenticates the three inherited
runs by hash, and resolves the final five-seed ensemble across the two roots.

No split is generated here.  All five seeds consume the same frozen
``predefined_oof_kfold`` artifact and its frozen ``k_fold``/``val_fold_*``
columns.  Each new chain is the exact canonical E0 recipe: five OOF fits followed
by the governed p75 full-source refit.

Typical production flow::

    python tools/aim1_e0_five_seed_extension.py plan
    python tools/aim1_e0_five_seed_extension.py prepare
    python tools/aim1_e0_five_seed_extension.py preflight
    python tools/aim1_e0_five_seed_extension.py train --max-workers 2
    python tools/aim1_e0_five_seed_extension.py validate
    python tools/aim1_e0_five_seed_extension.py analyze
    python tools/aim1_e0_five_seed_extension.py verify

``jobs`` emits two one-slot worker commands for integration into the study-wide
six-trainer scheduler.  The controller's own scheduler is deliberately capped at
two because this extension contains only two independent seed chains.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.workflows.training import (  # noqa: E402
    _feature_inventory_sha256,
    validate_training_run_dir,
)


class ContractError(RuntimeError):
    """The sealed extension contract or one of its inputs is invalid."""


SCHEMA_VERSION = 1
ALL_SEEDS = (42, 43, 44, 45, 46)
INHERITED_SEEDS = (42, 43, 44)
NEW_SEEDS = (45, 46)
N_FOLDS = 5
DOMAINS = ("TCGA", "SR386", "SR1482", "RIH", "CPTAC")
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20_260_817
MAX_OWN_TRAINERS = 2
EXTERNAL_GLOBAL_CAP = 6

DEFAULT_CAMPAIGN_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/final_v9_mil_5seed_expansion_v1_20260823/aim1_e0"
)
DEFAULT_LEGACY_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/train/1a_pb_cap8192/univ1")
DEFAULT_MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/aim1_dev.csv")
DEFAULT_SPLIT_DIR = REPO / "outputs/splits/aim1_1a/aim1_balanced5"
DEFAULT_PACKED_DIR = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/20x_256px_0px_overlap_mpp0.5/packed_uni_v1"
)
DEFAULT_FEATURE_DIR = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/20x_256px_0px_overlap_mpp0.5/features_uni_v1"
)
PYTHON = REPO / ".venv/bin/python"

EXPECTED_INPUT_HASHES = {
    "manifest": "d906ed5b61c5d3bbf56da7ec7bad412287307461012deee5e6d98ae97bd1f3d1",
    "splits": "3ec0b106f3614ef994efa88c4acdac35591166639517a4c0defa549a476687e8",
    "split_integrity": "daa4668cf5364cce61dea5e585ef0b353e6ede2ec932f94322323171bd8ac3c5",
    "split_summary": "e02c3ddd4a0f712370519c56f7a1e5aba2e20b5c49f8fa8ec5e8ae27887c471a",
}
EXPECTED_FEATURE_INVENTORY = "84fa286bca2c94b8488c6d4b1a72b9b2a0b756053d422658e1951a4552722c5d"
EXPECTED_PACK = {
    "meta.json": "44f1f0c80f4b8740c5fc82caafa96a8ead372c561fe53cda896d83972ff3c80b",
    "index.parquet": "705464267e35e454e50ea8510bc937e9f8d6a00b8b97bb9cf1e06da5fcfa9556",
    "features.bin": "6765d9faf30f40e212075c1a650fd34bc5ef625b3fb8e169dbc14e78c2058bd4",
    "coords.bin": "1a90bd95a430e433cff900c0b6b2a4f6fedcae550ae80584ee27ed7f100fe491",
    "features.bin_size": 49_735_520_256,
    "coords.bin_size": 194_279_376,
}

# These are the material implementation/config files used by the frozen command.
# The new controller is bound dynamically in the contract because this file did
# not exist in the historical lineage; all pre-existing training code is pinned.
EXPECTED_CODE_HASHES = {
    "scripts/train.py": "7d288862c791f0dd7358057619be06824d3169571a4c13744a826eadfee695ea",
    "configs/train.yaml": "b87cf8beff74e074537baa34c37f8b4f832eed34b700e0601a7ce95a7807d0d4",
    "configs/platform/colon_workstation.yaml": (
        "7cc47ba497ccf585a12a7fdfcd8859763a2f18b80937046047f04a9b8962c03e"
    ),
    "configs/runtime/default.yaml": (
        "c9b2ad2a7255faeaffb1681a61196be8003d90f09ce6f02ce31b0f79dbf3b824"
    ),
    "configs/data/aim1.yaml": "1e3d9a23ab051e28990090c3b3a36541f7bc163ca851992cf24ed88b3b4f5282",
    "configs/encoder/univ1.yaml": (
        "1adc3d9045ab3bc4bd019c710bb35169ca907fa5993e21523203d92a92ee75d7"
    ),
    "configs/extraction/default.yaml": (
        "9dada4023b2d926e64b2fd6cc707499e241e683d799ce0a7bb0d5dc55cb33411"
    ),
    "configs/splits/aim1_balanced.yaml": (
        "67785ee111dd8811f88a85bf7178252e34477c0c5043efe49c66de735db855eb"
    ),
    "configs/model/abmil.yaml": (
        "9f9320c107d40bc9931d80970e599e1d9d42eb2261711cf1c4bc743138a5e741"
    ),
    "configs/training/default.yaml": (
        "ac948d025a3922031395a7cb8926012337ce262577e21f6ac2482b5a942557ed"
    ),
    "configs/training/aim1.yaml": (
        "96b9d4100f53868527a5dab2c05bf403dcc69960482ebc77e20ee54bc26c1f70"
    ),
    "configs/experiment/supervised.yaml": (
        "28ec147dc1fd5153025eb9dc87bd21405d53c34fc71bf1a86ffc91b9255238ff"
    ),
    "src/oceanpath/workflows/training.py": (
        "a22cd0579533cbd43941d338a15d0eb6b972f8c440f4a622361ca22de8e939b6"
    ),
    "src/oceanpath/workflows/finalize.py": (
        "8143ad8e60957d8c34b0e1d29ce319425e723a33f69387bc48aa220a60815951"
    ),
    "src/oceanpath/training/folds.py": (
        "5822d765f77eb4c00c2f5410e9485aaeb5d0e096e234f9e1e1f2c74aea87b1f0"
    ),
    "src/oceanpath/training/lightning.py": (
        "89bf0d9acfa1024eb3ed95711b9710d2d06026f0f292463eb9f593709295ecd6"
    ),
    "src/oceanpath/datasets/datamodule.py": (
        "348d3265f77fba70bc368edc698a69f9ee05c54b4260cea0f7eb6c43bd1f2192"
    ),
    "src/oceanpath/datasets/sampling.py": (
        "7bc3d17cdf167f3c5b3096a64bfb1d797746f27b42b9c59009a50602e60ede54"
    ),
    "src/oceanpath/datasets/packed.py": (
        "bcfc22a887e5b02b423512b72405343988e4abd79709f278769ae5b8a57d6a3f"
    ),
    "src/oceanpath/models/abmil.py": (
        "e72deb0418b04b605babdc64bbffe5acf870bee8a8f731dec0d9735b51a709d6"
    ),
}

EXPECTED_INHERITED = {
    42: {
        "fingerprint": "62b05269c4c9ff7e",
        "config": "5c1241e8536322cbeb6e78d91e72cd8ee68706b5fe9e5e68dc90eced066d34dd",
        "oof": "fc61f04175817214614f0526677e1baedaead7e0d21ebab8bd66f55a517c3be8",
        "identity": "d65015c07a976b5ae8edd83c0e38e47e680b42044c690908800cdcd91df845cd",
        "completion": "abe8f8ab24dafad8cc9a08dcf51ab8cb93d385bd4b82f3cc585c2dc344c29725",
        "refit": "fbf7ff9720467005e7cd1273be6a680bd6b2d39a7ddf850b3de371c5bf930f36",
        "refit_info": ("b7a56bce7090f37c136844207d1d57ee60447da72c480d256134dec392dc97b3"),
    },
    43: {
        "fingerprint": "6286bc5e58afd3a4",
        "config": "3708faf23232632e7f65ed3fa2b1f100d11a43222aebd0cd09841065433fcf89",
        "oof": "ce725874d940e0e329dc29d928c68aa2b5be4486689667210105cbd716e0a1f6",
        "identity": "c8dc90325f38015ceeb4f89ba3f6a8040ea2c497eca6fca25209a422e3c831b7",
        "completion": "52b0b5884bfc879161d804ca7bad79817ffa8b35a303012877c58665dba27bfd",
        "refit": "3594fa7a4379b86e3eaa4728dd66774e1fb219e3e00033b56d937ea94619cc2d",
        "refit_info": ("8977f99ff3b22160ccc8637775d91d1042c74d0e3cc00e1604b8a203fff3258c"),
    },
    44: {
        "fingerprint": "9d1475d20430dd3e",
        "config": "339c44fe06e18aad1734bedd369d04fff074732844333c1f61ac9d12ae63c687",
        "oof": "653b4c0d69bd75a8824bbeae4e14c23b7e37f476ae3871d232cb54a02db84646",
        "identity": "a3d5bca8bcf8806b49706dfa05873d0583fa7259c4691055275117773cbb4177",
        "completion": "b08bb8b3351c26b20cb9aa2991918c5e12a963a31e00574ed18b13ddada8dcf8",
        "refit": "1bab8487190376a4e2a397871d3b2b383ef5445fb83fe0353ffb028511700c67",
        "refit_info": ("e3248ecfa2761a4a838bcd289c2b39bcf675b46d95ebbc0a4d7173250c8261fc"),
    },
}

EXPECTED_NEW_FINGERPRINTS = {45: "3d720b1d3de792b1", 46: "e298710fc2edddf3"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"Required artifact is missing: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON artifact must contain an object: {path}")
    return value


def _publish_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ContractError(f"Refusing to replace a different sealed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _guard_campaign_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == DEFAULT_CAMPAIGN_ROOT.resolve() or _is_relative_to(resolved, Path("/tmp")):
        return resolved
    raise ContractError(
        f"Production campaign root must be {DEFAULT_CAMPAIGN_ROOT}; "
        f"only /tmp alternatives are accepted for tests, got {resolved}"
    )


def _contract_path(campaign_root: Path) -> Path:
    return campaign_root / "contract/campaign_contract.json"


def _training_receipt_path(campaign_root: Path) -> Path:
    return campaign_root / "receipts/five_seed_training_validation.json"


def _results_path(campaign_root: Path) -> Path:
    return campaign_root / "analysis/five_seed_results.json"


def _patient_scores_path(campaign_root: Path) -> Path:
    return campaign_root / "analysis/five_seed_patient_native_logits.parquet"


def _analysis_receipt_path(campaign_root: Path) -> Path:
    return campaign_root / "analysis/receipt.json"


def _overlay_run_dir(campaign_root: Path, seed: int) -> Path:
    if seed not in NEW_SEEDS:
        raise ContractError(f"Seed {seed} is not an extension seed")
    return campaign_root / f"train/1a_pb_cap8192/univ1/seed{seed}"


def resolve_run_dir(campaign_root: Path, legacy_root: Path, seed: int) -> Path:
    """Resolve inherited seeds from the old tree and new seeds from the overlay."""

    if seed in INHERITED_SEEDS:
        return legacy_root / f"seed{seed}"
    if seed in NEW_SEEDS:
        return _overlay_run_dir(campaign_root, seed)
    raise ContractError(f"Seed {seed} is outside the frozen five-seed roster")


def _training_command(campaign_root: Path, seed: int) -> list[str]:
    run_dir = _overlay_run_dir(campaign_root, seed)
    hydra_dir = campaign_root / f"hydra/seed{seed}"
    return [
        str(PYTHON),
        str(REPO / "scripts/train.py"),
        "platform=colon_workstation",
        "data=aim1",
        "data.aim1_model=1a",
        "data.manifest_stem=aim1_dev",
        "+data.cohort_column=cohort",
        "encoder=univ1",
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
        "training.dataset_max_instances=8192",
        "training.eval_full_bags=true",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        f"train_dir={run_dir}",
        f"exp_name=aim1_e0_5seed_extension_v1_seed{seed}",
        f"hydra.run.dir={hydra_dir}",
        "hydra.job.chdir=false",
    ]


def _controller_worker_command(campaign_root: Path, seed: int) -> list[str]:
    return [
        str(PYTHON),
        str(Path(__file__).resolve()),
        "--campaign-root",
        str(campaign_root),
        "train-one",
        "--seed",
        str(seed),
        "--external-scheduler",
    ]


def build_job_inventory(campaign_root: Path = DEFAULT_CAMPAIGN_ROOT) -> list[dict[str, Any]]:
    """Return the two one-slot jobs consumed by the study-wide max-six scheduler.

    This deliberately performs no filesystem write and no GPU query, so a master
    scheduler can import it while composing the cross-aim queue.  ``prepare``
    remains responsible for sealing/authenticating the same commands before any
    worker is launched.
    """

    root = _guard_campaign_root(campaign_root)
    return [
        {
            "key": f"aim1_e0_seed{seed}",
            "component": "aim1_e0",
            "seed": seed,
            "scheduler_slots": 1,
            "run_dir": str(_overlay_run_dir(root, seed).resolve()),
            "expected_training_fingerprint": EXPECTED_NEW_FINGERPRINTS[seed],
            "training_command": _training_command(root, seed),
            "external_worker_command": _controller_worker_command(root, seed),
        }
        for seed in NEW_SEEDS
    ]


def _authenticate_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    observed = _artifact(path)
    if observed["sha256"] != expected:
        raise ContractError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed['sha256']}"
        )
    return observed


def _authenticate_code() -> dict[str, dict[str, Any]]:
    return {
        relative: _authenticate_hash(REPO / relative, expected, relative)
        for relative, expected in EXPECTED_CODE_HASHES.items()
    }


def _load_manifest_and_splits(
    manifest_path: Path, split_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    inputs = {
        "manifest": _authenticate_hash(
            manifest_path, EXPECTED_INPUT_HASHES["manifest"], "Aim1 manifest"
        ),
        "splits": _authenticate_hash(
            split_dir / "splits.parquet", EXPECTED_INPUT_HASHES["splits"], "Aim1 splits"
        ),
        "split_integrity": _authenticate_hash(
            split_dir / ".integrity_hash",
            EXPECTED_INPUT_HASHES["split_integrity"],
            "Aim1 split integrity",
        ),
        "split_summary": _authenticate_hash(
            split_dir / "summary.json",
            EXPECTED_INPUT_HASHES["split_summary"],
            "Aim1 split summary",
        ),
    }
    manifest = pd.read_csv(manifest_path)
    splits = pd.read_parquet(split_dir / "splits.parquet")
    required = {
        "slide_id",
        "patient_id",
        "target_label",
        "cohort",
        "subcohort",
        "k_fold",
        *{f"val_fold_{fold}" for fold in range(N_FOLDS)},
    }
    if not required <= set(manifest) or not required <= set(splits):
        raise ContractError(f"Frozen manifest/splits lack required columns: {sorted(required)}")
    if len(manifest) != 1_642 or manifest["slide_id"].nunique() != 1_642:
        raise ContractError("Aim1 manifest must contain exactly 1,642 unique slides")
    if manifest["patient_id"].nunique() != 1_486:
        raise ContractError("Aim1 manifest must contain exactly 1,486 patients")
    if manifest.groupby("patient_id")["k_fold"].nunique().max() != 1:
        raise ContractError("A patient crosses outer folds in the frozen manifest")
    left = manifest.sort_values("slide_id").reset_index(drop=True)
    right = splits.sort_values("slide_id").reset_index(drop=True)
    compare = ["slide_id", "patient_id", "target_label", "k_fold"] + [
        f"val_fold_{fold}" for fold in range(N_FOLDS)
    ]
    mismatches = {
        column: int((left[column].astype(str) != right[column].astype(str)).sum())
        for column in compare
    }
    if any(mismatches.values()):
        raise ContractError(f"Frozen manifest and split artifact disagree: {mismatches}")
    patient = manifest.drop_duplicates("patient_id")
    counts = patient["target_label"].value_counts().to_dict()
    if counts != {0: 882, 1: 604}:
        raise ContractError(f"Unexpected patient labels: {counts}")
    return manifest, splits, inputs


def _authenticate_feature_inputs(
    manifest_path: Path,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    packed_dir: Path = DEFAULT_PACKED_DIR,
) -> dict[str, Any]:
    """Authenticate selected H5 identities plus the indexed packed store."""

    inventory = _feature_inventory_sha256(feature_dir, manifest_path, "slide_id")
    if inventory != EXPECTED_FEATURE_INVENTORY:
        raise ContractError(
            "Aim1 selected-feature inventory changed: "
            f"expected {EXPECTED_FEATURE_INVENTORY}, observed {inventory}"
        )
    pack: dict[str, Any] = {
        "meta": _authenticate_hash(
            packed_dir / "meta.json", EXPECTED_PACK["meta.json"], "packed feature metadata"
        ),
        "index": _authenticate_hash(
            packed_dir / "index.parquet",
            EXPECTED_PACK["index.parquet"],
            "packed feature index",
        ),
    }
    for name in ("features.bin", "coords.bin"):
        path = packed_dir / name
        if not path.is_file():
            raise ContractError(f"Packed feature payload is missing: {path}")
        observed_size = path.stat().st_size
        expected_size = EXPECTED_PACK[f"{name}_size"]
        if observed_size != expected_size:
            raise ContractError(
                f"Packed {name} size changed: expected {expected_size}, observed {observed_size}"
            )
        pack[name] = _authenticate_hash(path, EXPECTED_PACK[name], f"packed {name}")
    index = pd.read_parquet(packed_dir / "index.parquet")
    manifest_ids = set(pd.read_csv(manifest_path, usecols=["slide_id"])["slide_id"].astype(str))
    index_column = "slide_id" if "slide_id" in index else "key"
    if index_column not in index:
        raise ContractError("Packed feature index lacks a slide identifier column")
    missing = manifest_ids - set(index[index_column].astype(str))
    if missing:
        raise ContractError(f"Packed feature index lacks {len(missing)} Aim1 slides")
    return {"selected_h5_inventory_sha256": inventory, "packed_store": pack}


def _validate_refit(run_dir: Path) -> dict[str, Any]:
    info_path = run_dir / "final/refit/info.json"
    model_path = run_dir / "final/refit/model.ckpt"
    summary_path = run_dir / "final/finalize_summary.json"
    info = _read_json(info_path)
    summary = _read_json(summary_path)
    epochs = info.get("fold_best_epochs")
    expected_epoch = (
        int(math.ceil(float(np.percentile(epochs, 75))))
        if isinstance(epochs, list) and len(epochs) == N_FOLDS
        else None
    )
    required = {
        "strategy": "refit",
        "refit_epoch_rule": "p75",
        "n_train_slides": 1_642,
        "label_counts": {"0": 966, "1": 676},
    }
    mismatch = {
        key: (value, info.get(key)) for key, value in required.items() if info.get(key) != value
    }
    if mismatch or expected_epoch is None or info.get("refit_epochs") != expected_epoch:
        raise ContractError(f"Invalid p75 refit at {run_dir}: {mismatch}")
    if (summary.get("refit") or {}).get("refit_epoch_rule") != "p75":
        raise ContractError(f"Finalize summary does not authenticate a p75 refit: {summary_path}")
    return {
        "checkpoint": _artifact(model_path),
        "info": _artifact(info_path),
        "finalize_summary": _artifact(summary_path),
        "refit_epochs": expected_epoch,
        "fold_best_epochs": [int(value) for value in epochs],
    }


def _validate_oof_and_fold_rosters(
    run_dir: Path, manifest: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    oof_path = run_dir / "oof_predictions.parquet"
    oof = pd.read_parquet(oof_path)
    required = {"slide_id", "label", "logit", "fold"}
    if not required <= set(oof):
        raise ContractError(f"Seed {seed} OOF lacks native-logit columns: {sorted(required)}")
    if len(oof) != 1_642 or oof["slide_id"].nunique() != 1_642:
        raise ContractError(f"Seed {seed} OOF does not cover exactly 1,642 slides")
    if not np.isfinite(pd.to_numeric(oof["logit"], errors="coerce")).all():
        raise ContractError(f"Seed {seed} OOF contains non-finite native logits")
    expected = manifest[["slide_id", "target_label", "k_fold"]].sort_values("slide_id")
    observed = oof[["slide_id", "label", "fold"]].sort_values("slide_id")
    if expected["slide_id"].tolist() != observed["slide_id"].tolist():
        raise ContractError(f"Seed {seed} OOF slide roster differs from the manifest")
    if expected["target_label"].astype(int).tolist() != observed["label"].astype(int).tolist():
        raise ContractError(f"Seed {seed} OOF labels differ from the manifest")
    if expected["k_fold"].astype(int).tolist() != observed["fold"].astype(int).tolist():
        raise ContractError(f"Seed {seed} OOF folds differ from the manifest")

    folds: dict[str, Any] = {}
    for fold in range(N_FOLDS):
        test = pd.read_parquet(run_dir / f"fold_{fold}/preds_test.parquet")
        val = pd.read_parquet(run_dir / f"fold_{fold}/preds_val.parquet")
        expected_test = set(manifest.loc[manifest["k_fold"].eq(fold), "slide_id"].astype(str))
        expected_val = set(manifest.loc[manifest[f"val_fold_{fold}"].eq(1), "slide_id"].astype(str))
        if set(test["slide_id"].astype(str)) != expected_test:
            raise ContractError(f"Seed {seed} fold {fold} test roster changed")
        if set(val["slide_id"].astype(str)) != expected_val:
            raise ContractError(f"Seed {seed} fold {fold} validation roster changed")
        folds[str(fold)] = {
            "n_test_slides": len(expected_test),
            "n_val_slides": len(expected_val),
            "completion": _artifact(run_dir / f"fold_{fold}/completion.json"),
        }
    return oof, {"oof": _artifact(oof_path), "folds": folds}


def _validate_run(
    run_dir: Path,
    manifest: pd.DataFrame,
    seed: int,
    expected_fingerprint: str,
    inherited_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    validate_training_run_dir(
        run_dir,
        expected_fingerprint=expected_fingerprint,
        require_test_predictions=True,
    )
    identity = _read_json(run_dir / "training_identity.json")
    evidence = (identity.get("payload") or {}).get("input_evidence") or {}
    expected_evidence = {
        "manifest_sha256": EXPECTED_INPUT_HASHES["manifest"],
        "split_integrity_sha256": EXPECTED_INPUT_HASHES["split_integrity"],
        "feature_inventory_sha256": EXPECTED_FEATURE_INVENTORY,
        "encoder_checkpoint_sha256": "default",
        "aggregator_checkpoint_sha256": "none",
    }
    if identity.get("fingerprint") != expected_fingerprint or evidence != expected_evidence:
        raise ContractError(f"Seed {seed} training identity differs from the frozen E0 recipe")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    checks = {
        "splits.scheme": (config.get("splits") or {}).get("scheme"),
        "splits.name": (config.get("splits") or {}).get("name"),
        "splits.seed": (config.get("splits") or {}).get("seed"),
        "training.seed": (config.get("training") or {}).get("seed"),
        "training.dataset_max_instances": (config.get("training") or {}).get(
            "dataset_max_instances"
        ),
        "training.train_sampling_strategy": (config.get("training") or {}).get(
            "train_sampling_strategy"
        ),
        "training.refit_epoch_rule": (config.get("training") or {}).get("refit_epoch_rule"),
    }
    expected_checks = {
        "splits.scheme": "predefined_oof_kfold",
        "splits.name": "aim1_balanced5",
        "splits.seed": seed,
        "training.seed": seed,
        "training.dataset_max_instances": 8_192,
        "training.train_sampling_strategy": "patient_natural",
        "training.refit_epoch_rule": "p75",
    }
    if checks != expected_checks:
        raise ContractError(f"Seed {seed} resolved config mismatch: {checks}")

    oof, roster = _validate_oof_and_fold_rosters(run_dir, manifest, seed)
    refit = _validate_refit(run_dir)
    artifacts = {
        "config": _artifact(run_dir / "config.yaml"),
        "identity": _artifact(run_dir / "training_identity.json"),
        "completion": _artifact(run_dir / "training_completion.json"),
        **roster,
        "refit": refit,
    }
    if inherited_hashes:
        observed = {
            "config": artifacts["config"]["sha256"],
            "oof": artifacts["oof"]["sha256"],
            "identity": artifacts["identity"]["sha256"],
            "completion": artifacts["completion"]["sha256"],
            "refit": artifacts["refit"]["checkpoint"]["sha256"],
            "refit_info": artifacts["refit"]["info"]["sha256"],
        }
        wanted = {key: value for key, value in inherited_hashes.items() if key != "fingerprint"}
        if observed != wanted:
            raise ContractError(f"Inherited seed {seed} artifact hashes changed: {observed}")
    return {"seed": seed, "run_dir": str(run_dir.resolve()), "artifacts": artifacts, "oof": oof}


def _validate_partial_new_run(run_dir: Path, seed: int) -> None:
    if not run_dir.exists():
        return
    identity_path = run_dir / "training_identity.json"
    if not identity_path.is_file():
        raise ContractError(
            f"Seed {seed} output exists without an authenticated identity; refusing reuse: {run_dir}"
        )
    identity = _read_json(identity_path)
    if identity.get("fingerprint") != EXPECTED_NEW_FINGERPRINTS[seed]:
        raise ContractError(f"Seed {seed} partial run has a foreign training fingerprint")


def _quarantine_incomplete_new_run(campaign_root: Path, seed: int) -> Path | None:
    """Preserve, then clear, a prior attempt that never published completion."""

    run_dir = _overlay_run_dir(campaign_root, seed)
    if not run_dir.exists() or (run_dir / "training_completion.json").is_file():
        return None
    root = campaign_root / "quarantine" / f"seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (
        (root / f"attempt-{attempt:03d}").exists()
        or (root / f"attempt-{attempt:03d}.receipt.json").exists()
    ):
        attempt += 1
    destination = root / f"attempt-{attempt:03d}"
    if any(path.is_symlink() for path in run_dir.rglob("*")):
        raise ContractError(f"Incomplete seed {seed} attempt contains a symlink")
    os.replace(run_dir, destination)
    evidence = _quarantine_evidence(campaign_root, seed, destination)
    _publish_json(root / f"attempt-{attempt:03d}.receipt.json", evidence)
    return destination


def _quarantine_evidence(
    campaign_root: Path,
    seed: int,
    destination: Path,
    *,
    recovered_after_interrupted_publication: bool = False,
) -> dict[str, Any]:
    if any(path.is_symlink() for path in destination.rglob("*")):
        raise ContractError(f"Quarantined seed {seed} attempt contains a symlink")
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "incomplete_attempt_quarantined_before_retry",
        "seed": seed,
        "source": str(_overlay_run_dir(campaign_root, seed)),
        "destination": str(destination),
        "files": [
            {
                "relative_path": str(path.relative_to(destination)),
                **_artifact(path),
            }
            for path in sorted(destination.rglob("*"))
            if path.is_file() and not path.is_symlink()
        ],
    }
    if recovered_after_interrupted_publication:
        evidence["recovered_after_interrupted_receipt_publication"] = True
    identity = destination / "training_identity.json"
    if identity.is_file():
        evidence["training_identity"] = _artifact(identity)
    return evidence


def _recover_unreceipted_quarantine_attempts(campaign_root: Path, seed: int) -> None:
    """Close the move→receipt crash window before a governed retry starts."""

    root = campaign_root / "quarantine" / f"seed{seed}"
    if not root.exists():
        return
    for destination in sorted(path for path in root.glob("attempt-*") if path.is_dir()):
        receipt_path = destination.with_suffix(".receipt.json")
        if receipt_path.exists():
            continue
        _publish_json(
            receipt_path,
            _quarantine_evidence(
                campaign_root,
                seed,
                destination,
                recovered_after_interrupted_publication=True,
            ),
        )


def _validate_quarantine_receipts(campaign_root: Path) -> list[dict[str, Any]]:
    """Authenticate every preserved failed attempt without accepting it as training."""

    identities: list[dict[str, Any]] = []
    quarantine_root = campaign_root / "quarantine"
    if not quarantine_root.exists():
        return identities
    for receipt_path in sorted(quarantine_root.glob("seed*/attempt-*.receipt.json")):
        receipt = _read_json(receipt_path)
        seed = int(receipt.get("seed", -1))
        destination = Path(str(receipt.get("destination", "")))
        if seed not in NEW_SEEDS or destination.parent != quarantine_root / f"seed{seed}":
            raise ContractError(f"Malformed Aim1 quarantine receipt: {receipt_path}")
        expected = _quarantine_evidence(
            campaign_root,
            seed,
            destination,
            recovered_after_interrupted_publication=bool(
                receipt.get("recovered_after_interrupted_receipt_publication", False)
            ),
        )
        if receipt != expected:
            raise ContractError(f"Aim1 quarantine evidence changed: {receipt_path}")
        identities.append(_artifact(receipt_path))
    attempt_directories = {
        path.resolve()
        for path in quarantine_root.glob("seed*/attempt-*")
        if path.is_dir()
    }
    receipted_directories = {
        Path(_read_json(path)["destination"]).resolve()
        for path in quarantine_root.glob("seed*/attempt-*.receipt.json")
    }
    if attempt_directories != receipted_directories:
        raise ContractError("Aim1 quarantine attempt/receipt census is not one-to-one")
    return identities


def _build_contract(
    campaign_root: Path,
    legacy_root: Path,
    manifest_path: Path,
    split_dir: Path,
    *,
    created_utc: str,
) -> dict[str, Any]:
    manifest, _splits, inputs = _load_manifest_and_splits(manifest_path, split_dir)
    inputs["features"] = _authenticate_feature_inputs(manifest_path)
    code = _authenticate_code()
    inherited: dict[str, Any] = {}
    for seed in INHERITED_SEEDS:
        record = _validate_run(
            legacy_root / f"seed{seed}",
            manifest,
            seed,
            EXPECTED_INHERITED[seed]["fingerprint"],
            EXPECTED_INHERITED[seed],
        )
        inherited[str(seed)] = {
            "run_dir": record["run_dir"],
            "fingerprint": EXPECTED_INHERITED[seed]["fingerprint"],
            "artifacts": record["artifacts"],
        }
    jobs = build_job_inventory(campaign_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created_utc,
        "status": "sealed before extension training",
        "experiment": "Aim1 E0 canonical five-seed extension",
        "scientific_change": {
            "old_model_seeds": list(INHERITED_SEEDS),
            "new_model_seeds": list(ALL_SEEDS),
            "added_model_seeds": list(NEW_SEEDS),
            "split_layout_changed": False,
            "model_recipe_changed": False,
            "inherited_artifacts_modified": False,
        },
        "roots": {
            "campaign_overlay": str(campaign_root.resolve()),
            "inherited_three_seed": str(legacy_root.resolve()),
        },
        "inputs": inputs,
        "material_code_and_configs": code,
        "extension_controller": _artifact(Path(__file__).resolve()),
        "inherited_runs": inherited,
        "jobs": jobs,
        "scheduler_contract": {
            "own_max_concurrent_trainers": MAX_OWN_TRAINERS,
            "external_study_max_concurrent_trainers": EXTERNAL_GLOBAL_CAP,
            "each_job_consumes_one_trainer_slot": True,
            "external_scheduler_must_enforce_global_cap": True,
        },
        "fit_census": {
            "inherited_folds": 15,
            "inherited_p75_refits": 3,
            "new_folds": 10,
            "new_p75_refits": 2,
            "final_folds": 25,
            "final_p75_refits": 5,
            "new_fits": 12,
            "final_fits": 30,
        },
        "analysis_contract": {
            "score": "stored native slide logit",
            "patient_aggregation": "mean native logit across slides",
            "seed_ensemble": "mean native patient logit across five seeds",
            "primary": "median across five seed-specific equal-five-domain AUROC macros",
            "bootstrap": {
                "unit": "patient",
                "strata": "acquisition_domain_x_KRAS",
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "shared_indices_across_model_seeds": True,
            },
            "probability_roundtrip": False,
        },
        "target_outcomes_opened_for_training": False,
    }


def _load_contract(
    campaign_root: Path, legacy_root: Path, manifest_path: Path, split_dir: Path
) -> dict[str, Any]:
    path = _contract_path(campaign_root)
    observed = _read_json(path)
    created = observed.get("created_utc")
    if not isinstance(created, str):
        raise ContractError("Campaign contract lacks a creation timestamp")
    expected = _build_contract(
        campaign_root,
        legacy_root,
        manifest_path,
        split_dir,
        created_utc=created,
    )
    if observed != expected:
        raise ContractError("Campaign contract differs from authenticated live inputs")
    return observed


def _available_ram_gib() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", maxsplit=1)
        values[key] = int(value.strip().split()[0])
    return values.get("MemAvailable", 0) / 1024**2


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise ContractError(f"No existing parent for output path {path}")
    return candidate


def _gpu_resources() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ContractError(f"GPU resource query failed: {result.stderr.strip()}")
    fields = [int(value.strip()) for value in result.stdout.strip().splitlines()[0].split(",")]
    return {"total_gpu_mib": fields[0], "free_gpu_mib": fields[1], "gpu_utilization": fields[2]}


def _resource_preflight(campaign_root: Path, packed_dir: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(_nearest_existing_parent(campaign_root))
    observed: dict[str, Any] = {
        "logical_cpus": int(os.cpu_count() or 0),
        "available_ram_gib": _available_ram_gib(),
        "free_disk_gib": disk.free / 1024**3,
        **_gpu_resources(),
        "packed_dir_exists": packed_dir.is_dir(),
    }
    minima = {
        "logical_cpus": 12,
        "available_ram_gib": 24.0,
        "free_disk_gib": 2.0,
        "free_gpu_mib": 8_000,
    }
    failed = {
        key: {"minimum": minimum, "observed": observed[key]}
        for key, minimum in minima.items()
        if observed[key] < minimum
    }
    if not observed["packed_dir_exists"]:
        failed["packed_dir_exists"] = {"minimum": True, "observed": False}
    if failed:
        raise ContractError(f"Aim1 E0 extension resource preflight failed: {failed}")
    return observed


def _root_lock_path(campaign_root: Path) -> Path:
    token = hashlib.sha256(str(campaign_root.resolve()).encode()).hexdigest()[:12]
    return Path(f"/tmp/oceanpath_aim1_e0_5seed_{token}.lock")


def _job_lock_path(campaign_root: Path, seed: int) -> Path:
    return _root_lock_path(campaign_root).with_name(
        f"{_root_lock_path(campaign_root).stem}_s{seed}.lock"
    )


@contextlib.contextmanager
def _exclusive_lock(path: Path, label: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f"Another process holds the {label} lock: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _terminate_process(
    process: subprocess.Popen[str], *, external_scheduler: bool = False
) -> None:
    if process.poll() is not None:
        return
    try:
        if external_scheduler:
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is not None:
            return
        if external_scheduler:
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _run_one(
    campaign_root: Path,
    seed: int,
    *,
    external_scheduler: bool = False,
) -> None:
    run_dir = _overlay_run_dir(campaign_root, seed)
    log_path = campaign_root / f"logs/seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(_job_lock_path(campaign_root, seed), f"seed {seed}"):
        _recover_unreceipted_quarantine_attempts(campaign_root, seed)
        _quarantine_incomplete_new_run(campaign_root, seed)
        _validate_partial_new_run(run_dir, seed)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== seed {seed} extension attempt {_utc_now()} ===\n")
            log.write(" ".join(_training_command(campaign_root, seed)) + "\n")
            log.flush()
            process = subprocess.Popen(
                _training_command(campaign_root, seed),
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                # A standalone Aim-1 launcher owns a child process group so it can
                # reap the full native training tree.  Under the study-wide
                # scheduler, remain in the authorized controller's process group:
                # its parent-death cleanup can then terminate this trainer and all
                # DataLoader descendants without leaving an orphan on GPU 0.
                start_new_session=not external_scheduler,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "0",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "PYTHONUNBUFFERED": "1",
                },
            )
            try:
                returncode = process.wait()
            except BaseException:
                _terminate_process(process, external_scheduler=external_scheduler)
                raise
    if returncode != 0:
        raise RuntimeError(f"Aim1 E0 seed {seed} failed with rc={returncode}; see {log_path}")


def _patient_native_logits(oof: pd.DataFrame, manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    joined = manifest[
        ["slide_id", "patient_id", "target_label", "cohort", "subcohort", "k_fold"]
    ].merge(oof[["slide_id", "logit"]], on="slide_id", how="left", validate="one_to_one")
    if joined["logit"].isna().any():
        raise ContractError(f"Seed {seed} lacks native logits for some manifest slides")
    consistency = joined.groupby("patient_id").agg(
        n_label=("target_label", "nunique"),
        n_cohort=("cohort", "nunique"),
        n_subcohort=("subcohort", "nunique"),
        n_fold=("k_fold", "nunique"),
    )
    if (consistency > 1).any().any():
        raise ContractError(f"Seed {seed} patient metadata is not internally consistent")
    patient = (
        joined.groupby("patient_id", as_index=False)
        .agg(
            label=("target_label", "first"),
            cohort=("cohort", "first"),
            subcohort=("subcohort", "first"),
            k_fold=("k_fold", "first"),
            **{f"logit_seed{seed}": ("logit", "mean")},
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    patient["domain"] = patient["cohort"].astype(str)
    patient.loc[patient["subcohort"].eq("SR386"), "domain"] = "SR386"
    patient.loc[patient["subcohort"].eq("SR1482"), "domain"] = "SR1482"
    if set(patient["domain"]) != set(DOMAINS):
        raise ContractError(
            f"Unexpected Aim1 acquisition domains: {sorted(patient['domain'].unique())}"
        )
    return patient


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) != 2:
        raise ContractError("AUROC cell lacks both KRAS classes")
    return float(roc_auc_score(labels, scores))


def _domain_aurocs(frame: pd.DataFrame, score: str) -> dict[str, float]:
    return {
        domain: _auc(
            frame.loc[frame["domain"].eq(domain), "label"].to_numpy(),
            frame.loc[frame["domain"].eq(domain), score].to_numpy(),
        )
        for domain in DOMAINS
    }


def _macro5(values: Mapping[str, float]) -> float:
    return float(np.mean([values[domain] for domain in DOMAINS]))


def _macro4(values: Mapping[str, float]) -> float:
    surgen = (values["SR386"] + values["SR1482"]) / 2
    return float(np.mean([values["TCGA"], surgen, values["RIH"], values["CPTAC"]]))


def _weighted_macro(values: Mapping[str, float], counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    return float(sum(counts[key] * values[key] for key in DOMAINS) / total)


def _interval(values: Sequence[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def build_five_seed_results(
    patient: pd.DataFrame,
    *,
    n_bootstrap: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute the governed E0 result only from stored native logits."""

    seed_columns = [f"logit_seed{seed}" for seed in ALL_SEEDS]
    if not set(seed_columns) <= set(patient):
        raise ContractError("Five-seed patient table is incomplete")
    if len(patient) != 1_486 or patient["patient_id"].nunique() != 1_486:
        raise ContractError("Five-seed result must contain exactly 1,486 patients")
    if not np.isfinite(patient[seed_columns].to_numpy(dtype=float)).all():
        raise ContractError("Five-seed patient table contains non-finite native logits")
    frame = patient.copy()
    frame["mean_logit_5seed"] = frame[seed_columns].mean(axis=1)
    counts = {domain: int(frame["domain"].eq(domain).sum()) for domain in DOMAINS}
    per_seed_domain: dict[str, dict[str, float]] = {}
    per_seed_macro5: dict[str, float] = {}
    per_seed_macro4: dict[str, float] = {}
    per_seed_weighted: dict[str, float] = {}
    pooled: dict[str, float] = {}
    for seed, column in zip(ALL_SEEDS, seed_columns, strict=True):
        domains = _domain_aurocs(frame, column)
        per_seed_domain[str(seed)] = domains
        per_seed_macro5[str(seed)] = _macro5(domains)
        per_seed_macro4[str(seed)] = _macro4(domains)
        per_seed_weighted[str(seed)] = _weighted_macro(domains, counts)
        pooled[str(seed)] = _auc(frame["label"].to_numpy(), frame[column].to_numpy())
    ensemble_domain = _domain_aurocs(frame, "mean_logit_5seed")

    rng = np.random.default_rng(bootstrap_seed)
    cells = [
        frame.index[frame["domain"].eq(domain) & frame["label"].eq(label)].to_numpy()
        for domain in DOMAINS
        for label in (0, 1)
    ]
    if any(len(cell) == 0 for cell in cells):
        raise ContractError("A domain-by-KRAS bootstrap cell is empty")
    primary_draws: list[float] = []
    ensemble_draws: list[float] = []
    macro4_draws: list[float] = []
    weighted_draws: list[float] = []
    primitive_draws: dict[str, list[float]] = {domain: [] for domain in DOMAINS}
    for _ in range(n_bootstrap):
        index = np.concatenate([rng.choice(cell, len(cell), replace=True) for cell in cells])
        boot = frame.loc[index]
        seed_domains = [_domain_aurocs(boot, column) for column in seed_columns]
        primary_draws.append(float(np.median([_macro5(value) for value in seed_domains])))
        macro4_draws.append(float(np.median([_macro4(value) for value in seed_domains])))
        weighted_draws.append(
            float(np.median([_weighted_macro(value, counts) for value in seed_domains]))
        )
        for domain in DOMAINS:
            primitive_draws[domain].append(
                float(np.median([value[domain] for value in seed_domains]))
            )
        ensemble_draws.append(_macro5(_domain_aurocs(boot, "mean_logit_5seed")))

    label = frame["label"].to_numpy()
    ensemble_score = frame["mean_logit_5seed"].to_numpy()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "Aim1 E0 canonical five-seed extension",
        "score_scale": "native_logit_only",
        "model_seeds": list(ALL_SEEDS),
        "population": {
            "patients": len(frame),
            "mutant": int(frame["label"].sum()),
            "wild_type": int((1 - frame["label"]).sum()),
            "domain_counts": counts,
        },
        "per_seed_domain_auroc": per_seed_domain,
        "per_seed_macro5": per_seed_macro5,
        "primary_median_seed_macro5": {
            "point": float(np.median(list(per_seed_macro5.values()))),
            "ci95": _interval(primary_draws),
        },
        "primitive_median_seed_auroc": {
            domain: {
                "point": float(
                    np.median([per_seed_domain[str(seed)][domain] for seed in ALL_SEEDS])
                ),
                "ci95": _interval(primitive_draws[domain]),
            }
            for domain in DOMAINS
        },
        "five_seed_ensemble": {
            "macro5_auroc": _macro5(ensemble_domain),
            "macro5_ci95": _interval(ensemble_draws),
            "domain_auroc": ensemble_domain,
            "pooled_auroc": _auc(label, ensemble_score),
            "pooled_auprc": float(average_precision_score(label, ensemble_score)),
        },
        "continuity": {
            "per_seed_pooled_auroc": pooled,
            "median_seed_pooled_auroc": float(np.median(list(pooled.values()))),
            "seed_macro5_range": [min(per_seed_macro5.values()), max(per_seed_macro5.values())],
            "seed_macro5_sd": float(np.std(list(per_seed_macro5.values()), ddof=1)),
        },
        "family_macro4_median_seed": {
            "point": float(np.median(list(per_seed_macro4.values()))),
            "ci95": _interval(macro4_draws),
        },
        "patient_count_weighted_macro_median_seed": {
            "point": float(np.median(list(per_seed_weighted.values()))),
            "ci95": _interval(weighted_draws),
        },
        "inference": {
            "bootstrap_draws": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "unit": "patient",
            "stratification": "acquisition_domain_x_KRAS",
            "shared_indices_across_model_seeds": True,
            "model_seeds_are_inferential_units": False,
        },
    }


def _validate_five_seed_runs(
    campaign_root: Path,
    legacy_root: Path,
    manifest_path: Path,
    split_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = _load_contract(campaign_root, legacy_root, manifest_path, split_dir)
    manifest, _splits, _inputs = _load_manifest_and_splits(manifest_path, split_dir)
    records: dict[str, Any] = {}
    patients: pd.DataFrame | None = None
    for seed in ALL_SEEDS:
        run_dir = resolve_run_dir(campaign_root, legacy_root, seed)
        expected = (
            EXPECTED_INHERITED[seed]["fingerprint"]
            if seed in INHERITED_SEEDS
            else EXPECTED_NEW_FINGERPRINTS[seed]
        )
        record = _validate_run(
            run_dir,
            manifest,
            seed,
            expected,
            EXPECTED_INHERITED.get(seed),
        )
        patient = _patient_native_logits(record.pop("oof"), manifest, seed)
        records[str(seed)] = record
        if patients is None:
            patients = patient
        else:
            metadata = ["patient_id", "label", "cohort", "subcohort", "k_fold", "domain"]
            if not patients[metadata].equals(patient[metadata]):
                raise ContractError(f"Patient/fold roster differs for seed {seed}")
            patients[f"logit_seed{seed}"] = patient[f"logit_seed{seed}"].to_numpy()
    if patients is None:
        raise ContractError("No five-seed patient records were resolved")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "campaign_contract": _artifact(_contract_path(campaign_root)),
        "model_seeds": list(ALL_SEEDS),
        "resolved_roots": {
            str(seed): str(resolve_run_dir(campaign_root, legacy_root, seed).resolve())
            for seed in ALL_SEEDS
        },
        "runs": records,
        "fit_census": contract["fit_census"],
        "fold_layout_shared_across_all_seeds": True,
        "p75_refit_authenticated_for_all_seeds": True,
        "quarantined_incomplete_attempt_receipts": _validate_quarantine_receipts(
            campaign_root
        ),
    }
    return receipt, patients


def _args_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    return (
        _guard_campaign_root(args.campaign_root),
        args.legacy_root.resolve(),
        args.manifest.resolve(),
        args.split_dir.resolve(),
    )


def cmd_plan(args: argparse.Namespace) -> None:
    root, legacy, manifest, splits = _args_paths(args)
    contract = _build_contract(root, legacy, manifest, splits, created_utc="UNSEALED_PLAN")
    print(json.dumps({"jobs": contract["jobs"], "fit_census": contract["fit_census"]}, indent=2))


def cmd_prepare(args: argparse.Namespace) -> None:
    root, legacy, manifest, splits = _args_paths(args)
    if _contract_path(root).is_file():
        contract = _load_contract(root, legacy, manifest, splits)
        print(f"Contract already sealed: {_contract_path(root)} ({len(contract['jobs'])} jobs)")
        return
    for seed in NEW_SEEDS:
        run_dir = _overlay_run_dir(root, seed)
        if run_dir.exists():
            raise ContractError(f"Fresh extension output already exists: {run_dir}")
    contract = _build_contract(root, legacy, manifest, splits, created_utc=_utc_now())
    _publish_json(_contract_path(root), contract)
    print(f"Sealed Aim1 E0 five-seed extension contract: {_contract_path(root)}")


def cmd_preflight(args: argparse.Namespace) -> None:
    root, legacy, manifest, splits = _args_paths(args)
    _load_contract(root, legacy, manifest, splits)
    for seed in NEW_SEEDS:
        _validate_partial_new_run(_overlay_run_dir(root, seed), seed)
    resources = _resource_preflight(root, args.packed_dir.resolve())
    print(json.dumps({"status": "PASS", "resources": resources}, indent=2))


def cmd_jobs(args: argparse.Namespace) -> None:
    root, legacy, manifest, splits = _args_paths(args)
    contract = _load_contract(root, legacy, manifest, splits)
    print(
        json.dumps(
            {
                "global_max_concurrent_gpu_trainers": EXTERNAL_GLOBAL_CAP,
                "extension_jobs": contract["jobs"],
            },
            indent=2,
        )
    )


def cmd_train_one(args: argparse.Namespace) -> None:
    root, legacy, manifest, splits = _args_paths(args)
    _load_contract(root, legacy, manifest, splits)
    if not args.external_scheduler:
        _resource_preflight(root, args.packed_dir.resolve())
    _run_one(root, args.seed, external_scheduler=args.external_scheduler)
    print(f"Seed {args.seed} training process completed; run validate to authenticate artifacts")


def cmd_train(args: argparse.Namespace) -> None:
    root, legacy, manifest, splits = _args_paths(args)
    _load_contract(root, legacy, manifest, splits)
    if not 1 <= args.max_workers <= MAX_OWN_TRAINERS:
        raise ContractError(f"--max-workers must be 1..{MAX_OWN_TRAINERS}")
    _resource_preflight(root, args.packed_dir.resolve())
    for seed in NEW_SEEDS:
        _validate_partial_new_run(_overlay_run_dir(root, seed), seed)
    failures: dict[int, str] = {}
    with (
        _exclusive_lock(_root_lock_path(root), "campaign orchestrator"),
        ThreadPoolExecutor(max_workers=args.max_workers) as pool,
    ):
        futures = {pool.submit(_run_one, root, seed): seed for seed in NEW_SEEDS}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - preserve both job failures
                failures[seed] = repr(exc)
    if failures:
        raise RuntimeError(f"Aim1 E0 extension training failures: {failures}")
    print("Both extension seed processes completed; run validate to authenticate artifacts")


def cmd_validate(args: argparse.Namespace) -> None:
    root, legacy, manifest, splits = _args_paths(args)
    receipt, _patients = _validate_five_seed_runs(root, legacy, manifest, splits)
    _publish_json(_training_receipt_path(root), receipt)
    print(f"Five-seed training validation PASS: {_training_receipt_path(root)}")


def cmd_analyze(args: argparse.Namespace) -> None:
    root, legacy, manifest, splits = _args_paths(args)
    receipt, patients = _validate_five_seed_runs(root, legacy, manifest, splits)
    _publish_json(_training_receipt_path(root), receipt)
    results = build_five_seed_results(patients)
    patient_output = patients.copy()
    patient_output["mean_logit_5seed"] = patient_output[
        [f"logit_seed{seed}" for seed in ALL_SEEDS]
    ].mean(axis=1)
    patient_path = _patient_scores_path(root)
    patient_path.parent.mkdir(parents=True, exist_ok=True)
    if patient_path.exists():
        existing = pd.read_parquet(patient_path)
        if not existing.equals(patient_output):
            raise ContractError(f"Refusing to replace different patient scores: {patient_path}")
    else:
        patient_output.to_parquet(patient_path, index=False)
    results["inputs"] = {
        "campaign_contract": _artifact(_contract_path(root)),
        "training_validation": _artifact(_training_receipt_path(root)),
        "patient_native_logits": _artifact(patient_path),
    }
    _publish_json(_results_path(root), results)
    analysis_receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "results": _artifact(_results_path(root)),
        "patient_native_logits": _artifact(patient_path),
        "training_validation": _artifact(_training_receipt_path(root)),
        "probability_roundtrip_used": False,
    }
    _publish_json(_analysis_receipt_path(root), analysis_receipt)
    primary = results["primary_median_seed_macro5"]
    print(
        "Five-seed Aim1 E0 primary macro "
        f"{primary['point']:.4f} [{primary['ci95'][0]:.4f}, {primary['ci95'][1]:.4f}]"
    )


def cmd_verify(args: argparse.Namespace) -> None:
    root, legacy, manifest, splits = _args_paths(args)
    receipt, patients = _validate_five_seed_runs(root, legacy, manifest, splits)
    stored_training = _read_json(_training_receipt_path(root))
    if receipt != stored_training:
        raise ContractError("Stored five-seed training receipt differs from live authentication")
    scores = pd.read_parquet(_patient_scores_path(root))
    if not scores.equals(
        patients.assign(
            mean_logit_5seed=patients[[f"logit_seed{seed}" for seed in ALL_SEEDS]].mean(axis=1)
        )
    ):
        raise ContractError("Stored five-seed patient-native-logit table changed")
    recomputed = build_five_seed_results(patients)
    stored = _read_json(_results_path(root))
    stored_core = {key: value for key, value in stored.items() if key != "inputs"}
    if stored_core != recomputed:
        raise ContractError("Stored five-seed result does not replay exactly")
    analysis_receipt = _read_json(_analysis_receipt_path(root))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "results": _artifact(_results_path(root)),
        "patient_native_logits": _artifact(_patient_scores_path(root)),
        "training_validation": _artifact(_training_receipt_path(root)),
        "probability_roundtrip_used": False,
    }
    if analysis_receipt != expected:
        raise ContractError("Analysis receipt differs from authenticated artifacts")
    print("Aim1 E0 five-seed extension verification PASS")


def cmd_status(args: argparse.Namespace) -> None:
    root, _legacy, _manifest, _splits = _args_paths(args)
    rows = []
    for seed in NEW_SEEDS:
        directory = _overlay_run_dir(root, seed)
        rows.append(
            {
                "seed": seed,
                "directory_exists": directory.exists(),
                "fold_receipts": sum(
                    (directory / f"fold_{fold}/completion.json").is_file()
                    for fold in range(N_FOLDS)
                ),
                "oof_exists": (directory / "oof_predictions.parquet").is_file(),
                "refit_exists": (directory / "final/refit/model.ckpt").is_file(),
            }
        )
    print(json.dumps({"contract": _contract_path(root).is_file(), "new_seeds": rows}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--packed-dir", type=Path, default=DEFAULT_PACKED_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    handlers = {
        "plan": cmd_plan,
        "prepare": cmd_prepare,
        "preflight": cmd_preflight,
        "jobs": cmd_jobs,
        "validate": cmd_validate,
        "analyze": cmd_analyze,
        "verify": cmd_verify,
        "status": cmd_status,
    }
    for name, handler in handlers.items():
        command = sub.add_parser(name)
        command.set_defaults(handler=handler)
    one = sub.add_parser("train-one")
    one.add_argument("--seed", type=int, choices=NEW_SEEDS, required=True)
    one.add_argument("--external-scheduler", action="store_true")
    one.set_defaults(handler=cmd_train_one)
    train = sub.add_parser("train")
    train.add_argument("--max-workers", type=int, default=MAX_OWN_TRAINERS)
    train.set_defaults(handler=cmd_train)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except (ContractError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
