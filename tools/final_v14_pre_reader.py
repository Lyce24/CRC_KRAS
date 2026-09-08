#!/usr/bin/env python3
"""Governed FINAL-v14 E4V computation through the blinded reader handoff.

The command is deliberately stage based.  Source data are authenticated before
any fit, every expensive unit is checkpointed, and the public reader directory
is created only after the canonical all-tile assignment gate is complete.

Typical production use::

    python tools/final_v14_pre_reader.py prepare
    python tools/final_v14_pre_reader.py preflight --deep-hash
    python tools/final_v14_pre_reader.py fit-vocabularies
    python tools/final_v14_pre_reader.py assign-profiles --max-workers 4
    python tools/final_v14_pre_reader.py score-teachers
    python tools/final_v14_pre_reader.py build-reader-package
    python tools/final_v14_pre_reader.py verify

Long-running commands belong in named tmux sessions.  This program itself does
not daemonize so that the exact command and its log remain visible to the
coordinator.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from oceanpath.aim1.v14_concepts import (  # noqa: E402
    V14Vocabulary,
    blinded_occurrences,
    draw_collision_free_hmac_salt,
    equal_slide_patient_profiles,
    hierarchical_sample_plan,
    hmac_blinding_table,
    l2_normalize_rows,
    lloyd_kmeans_parameters,
    map_vocabulary_to_reference,
    pca64_parameters,
    pcg64_rng,
    salt_sha256,
    select_duplicate_montages,
    select_montage_tiles,
    slide_abundance,
)
from oceanpath.datasets.packed import (  # noqa: E402
    PackedFeatureStore,
    feature_inventory_sha256,
)
from oceanpath.splitting.core import derive_subset_splits  # noqa: E402

SCHEMA_VERSION = 1
COMPONENT = "final_v14_e4v_pre_reader"
TODAY = "20260903"
DEFAULT_RUN_ROOT = REPO / f"reports/reruns/final_v14_additions_{TODAY}"
DEFAULT_OUTPUT_ROOT = DEFAULT_RUN_ROOT / "e4v_pre_reader"
DEFAULT_REVIEW_ROOT = REPO / "reviews/v14"
FINAL_V13 = REPO / "reports/final_v13"
FINAL_V13_SNAPSHOT = REPO / f"reports/snapshots/final_v13_pre_v14_{TODAY}"
PREREGISTRATION = REPO / "reports/final_v14_PREREGISTRATION.md"
PREREGISTRATION_SHA256 = (
    "677aad233a2e5f7c1d2975b5436269ed5060b19596904930ae49ae8c1cff3908"
)

HOT_ROOT = Path("/mnt/wsl/oceanpath-hot")
FEATURE_PROFILE_ROOT = (
    HOT_ROOT / "features/colon_stream/20x_256px_0px_overlap_mpp0.5"
)
PACK_ROOT = FEATURE_PROFILE_ROOT / "packed_uni_v1"
PATCH_ROOT = FEATURE_PROFILE_ROOT / "patches"
FEATURE_ROOT = FEATURE_PROFILE_ROOT / "features_uni_v1"
V13_CAMPAIGN_ROOT = (
    HOT_ROOT
    / "outputs/aim1/reruns/aim1_primary_cohort_5seed_v1_20260824"
)
V13_SOURCE_MANIFEST = V13_CAMPAIGN_ROOT / "inputs/manifests/tcga_surgen_primary.csv"
V13_SOURCE_SPLITS = (
    V13_CAMPAIGN_ROOT
    / "inputs/splits/aim1_primary_tcga_surgen/aim1_balanced5/splits.parquet"
)
V13_TEACHER_ROOT = (
    V13_CAMPAIGN_ROOT / "train/source_cv/cap8192/tcga_surgen_primary"
)
V13_PATIENT_LOGITS = V13_CAMPAIGN_ROOT / "analysis/patient_native_logits.parquet"
V13_TRAINING_RECEIPT = V13_CAMPAIGN_ROOT / "receipts/training_complete.json"
V13_CONTRACT = V13_CAMPAIGN_ROOT / "contract.json"
MASTER_MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/aim1_dev.csv")
MASTER_SPLITS = REPO / "outputs/splits/aim1_1a/aim1_balanced5"
SLIDE_ROOT = Path("/mnt/d/YC.Liu/slides/colon")
READY_INVENTORY = Path("/mnt/d/YC.Liu/manifests/colon/colon_ready_inventory.csv")

SOURCE_MANIFEST_SHA256 = (
    "d7087a23a84a294670080eb5090f83612f57b3376c7696ed2d0094fb9bcff8a5"
)
SOURCE_SPLITS_SHA256 = (
    "31046858b74a9e435ce7ceac263630e2966ab589c36a8693dcb0fc1060cfa9f4"
)
V13_PATIENT_LOGITS_SHA256 = (
    "b5ce00ffad68b26954a88bd909a53512ea25bbcaaf2f07847ce889c3f7fdd5c3"
)
V13_TRAINING_RECEIPT_SHA256 = (
    "31cc8005abf32cb1c8958d0b4ecb5ed34e927bcd1665451869d62a913bb75413"
)
V13_CONTRACT_SHA256 = (
    "fcc5a6744da27b73c5771f71a05511e998d20b3b4d76ecdc8a9c8c4ce34f0973"
)
V13_SOURCE_CV_RECEIPT_SHA256 = {
    42: "6450a83bf13a4fca1912001214045b359e848406bc3e3d6e040729210f9bad94",
    43: "94678c39b303fa458748126d284cebabad36efc0b74bef5a2f250e9c7ed87147",
    44: "561c3a0fd805cd47fe87f0a0f55e0210adcba07f42806c7e9cd7667d49aa36d3",
    45: "454dc92ba6f5f9382829d537bc0f734bfea421a5185841a8ab541378034ec93a",
    46: "654a00ccb3ed7c00b455aedd7ffa5ba86d2b26b15c8b70049f1e43a24f3df62d",
}
MASTER_MANIFEST_SHA256 = (
    "d906ed5b61c5d3bbf56da7ec7bad412287307461012deee5e6d98ae97bd1f3d1"
)
MASTER_SPLITS_SHA256 = (
    "3ec0b106f3614ef994efa88c4acdac35591166639517a4c0defa549a476687e8"
)
READY_INVENTORY_SHA256 = (
    "cf95d757cabd677ac03c612f59e7be1b3724875bd233acd2988735646fea79b7"
)
READY_INVENTORY_SIZE = 913_715

PACK_FILES: dict[str, dict[str, Any]] = {
    "meta.json": {
        "sha256": "44f1f0c80f4b8740c5fc82caafa96a8ead372c561fe53cda896d83972ff3c80b",
        "size_bytes": 367,
    },
    "index.parquet": {
        "sha256": "705464267e35e454e50ea8510bc937e9f8d6a00b8b97bb9cf1e06da5fcfa9556",
        "size_bytes": 66_998,
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

SOURCE_SLIDES = 1_389
SOURCE_PATIENTS = 1_239
SOURCE_MUTANT = 501
SOURCE_WILD_TYPE = 738
SOURCE_TILES = 16_711_039
SOURCE_SUBCOHORTS = ("SR1482", "SR386", "TCGA-COAD", "TCGA-READ")
FOLDS = (0, 1, 2, 3, 4)
MODEL_SEEDS = (42, 43, 44, 45, 46)
VOCAB_SEED = 20_260_819
SAMPLE_TILES = 400_000
PCA_COMPONENTS = 64
CANONICAL_K = 32
VARIANT_K = (24, 32, 40)
VARIANT_SEEDS = (20_260_819, 20_260_820, 20_260_821)
MAX_PROFILE_WORKERS = 4
FINE_TASKS = ("codon", "g12d_broad", "allele1", "allele2", "g12c")
WT_DRAW_SEEDS = (20_260_823, 20_260_824, 20_260_825)

LEGACY_INPUTS: dict[str, tuple[Path, str]] = {
    "legacy_vocab_npz": (
        HOT_ROOT / "outputs/aim1/e3b/vocabulary/vocab_k32.npz",
        "14496f3c629c4e8bcb809a3ab8bc4648d50a157c7c778e12150b0880333cac32",
    ),
    "legacy_vocab_json": (
        HOT_ROOT / "outputs/aim1/e3b/vocabulary/vocab_k32.json",
        "2e7f2bcfc41ef63af06eeb768daaa8417576c6f8740c52421670f93668ae2220",
    ),
    "legacy_human_results": (
        REPO
        / "reports/reruns/final_v5_additions_20260820/"
        "aim4_human_read_legacy_v2/results.json",
        "cfec16a15b5dbe9de367eaa5cda25bb97f5c8e25a26dd31b41f5cc5cb8b55d99",
    ),
    "legacy_human_receipt": (
        REPO
        / "reports/reruns/final_v5_additions_20260820/"
        "aim4_human_read_legacy_v2/receipt.json",
        "9b2571b24a541f9b3931ff05e0d08b1c2d1cea4b696f338bf36751234c8e93af",
    ),
}

ONTOLOGY = (
    "malignant gland-forming epithelium/gland–lumen",
    "malignant solid or poorly differentiated epithelium",
    "extracellular mucin/mucinous pattern",
    "normal or benign colonic epithelium",
    "desmoplastic/fibrous stroma",
    "smooth muscle",
    "lymphoid/inflammatory tissue",
    "necrosis/debris",
    "adipose",
    "blood/vessel",
    "mixed/other interpretable",
)


class ContractError(RuntimeError):
    """A fail-closed violation of the preregistered E4V contract."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, *, hash_content: bool = True) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_content:
        result["sha256"] = sha256_file(resolved)
    return result


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow")
    return buffer.getvalue()


def write_once(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    """Create immutable-by-convention bytes, accepting only an exact replay."""

    if path.exists() or path.is_symlink():
        if not path.is_file() or path.read_bytes() != payload:
            raise ContractError(f"Refusing to overwrite nonidentical artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    stage = Path(raw)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link is an atomic, same-filesystem publication that cannot
            # replace a path another process created after the initial check.
            os.link(stage, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise ContractError(
                    f"Refusing to overwrite nonidentical artifact: {path}"
                ) from None
    finally:
        with contextlib.suppress(FileNotFoundError):
            stage.unlink()


def atomic_write(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    """Publish a checkpoint atomically; an existing different file is fatal."""

    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise ContractError(f"Refusing to replace existing checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    stage = Path(raw)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(stage, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise ContractError(
                    f"Refusing to replace existing checkpoint: {path}"
                ) from None
    finally:
        with contextlib.suppress(FileNotFoundError):
            stage.unlink()


@contextlib.contextmanager
def exclusive_lock(path: Path, context: str) -> Iterable[None]:
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
            json_bytes({"pid": os.getpid(), "context": context, "utc": utc_now()}),
        )
        os.fsync(descriptor)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def require_hash(path: Path, expected: str, context: str) -> dict[str, Any]:
    observed = identity(path)
    if observed["sha256"] != expected:
        raise ContractError(
            f"{context} SHA-256 drift: expected {expected}, got {observed['sha256']}"
        )
    return observed


def sorted_utf8(values: Iterable[Any]) -> list[str]:
    normalized = {
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    }
    return sorted(normalized, key=lambda value: value.encode("utf-8"))


def source_frame_from_master() -> pd.DataFrame:
    require_hash(MASTER_MANIFEST, MASTER_MANIFEST_SHA256, "Aim-1 master manifest")
    master = pd.read_csv(MASTER_MANIFEST, low_memory=False)
    required = {
        "slide_id",
        "patient_id",
        "target_label",
        "cohort",
        "subcohort",
        "specimen_role",
        "k_fold",
        *(f"val_fold_{fold}" for fold in FOLDS),
    }
    if missing := required - set(master):
        raise ContractError(f"Master manifest lacks columns: {sorted(missing)}")
    role = master["specimen_role"].astype(str).str.strip().str.casefold()
    cohort = master["cohort"].astype(str).str.strip().str.casefold()
    frame = master.loc[role.eq("primary") & cohort.isin({"tcga", "surgen"})].copy()
    frame = frame.reset_index(drop=True)
    if frame["slide_id"].astype(str).duplicated().any():
        raise ContractError("Source manifest contains duplicate slide IDs")
    patients = frame.drop_duplicates("patient_id")
    labels = pd.to_numeric(patients["target_label"], errors="raise").astype(int)
    folds = pd.to_numeric(patients["k_fold"], errors="raise").astype(int)
    if (
        len(frame) != SOURCE_SLIDES
        or len(patients) != SOURCE_PATIENTS
        or int(labels.sum()) != SOURCE_MUTANT
        or int((labels == 0).sum()) != SOURCE_WILD_TYPE
        or set(frame["subcohort"].astype(str)) != set(SOURCE_SUBCOHORTS)
        or set(folds) != set(FOLDS)
    ):
        raise ContractError("The reconstructed FINAL-v13 source census drifted")
    patient_fold_counts = patients["k_fold"].value_counts().sort_index().tolist()
    if patient_fold_counts != [248, 247, 248, 248, 248]:
        raise ContractError(f"Source held-out fold census drifted: {patient_fold_counts}")
    raw = frame.to_csv(index=False).encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise ContractError("Source manifest is not byte-identical to FINAL-v13")
    return frame


def _label_blind_source(frame: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    allowed = [
        "slide_id",
        "patient_id",
        "cohort",
        "subcohort",
        "specimen_role",
        "native_mpp",
        "mpp_bin",
        "patch_count",
    ]
    blind = frame[[column for column in allowed if column in frame]].copy()
    fold_table = splits[["slide_id", "fold"]].copy()
    blind = blind.merge(fold_table, on="slide_id", validate="one_to_one")
    blind["fold"] = pd.to_numeric(blind["fold"], errors="raise").astype(int)
    forbidden = {
        "target_label",
        "label",
        "kras",
        "ras",
        "nras",
        "braf",
        "msi_dmmr",
        "outcome",
    }
    if forbidden & set(blind):
        raise ContractError("Outcome column entered the label-blind source roster")
    return blind.sort_values(
        ["subcohort", "patient_id", "slide_id"], kind="mergesort"
    ).reset_index(drop=True)


def build_analysis_dictionary(frame: pd.DataFrame) -> dict[str, Any]:
    patients = frame.sort_values("slide_id").drop_duplicates("patient_id")
    return {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "created_utc": utc_now(),
        "population": {
            "name": "tcga_surgen_primary",
            "slides": SOURCE_SLIDES,
            "patients": SOURCE_PATIENTS,
            "mutant": SOURCE_MUTANT,
            "wild_type": SOURCE_WILD_TYPE,
            "source_subcohort_order": list(SOURCE_SUBCOHORTS),
        },
        "ordered_indices": {
            "outer_folds": list(FOLDS),
            "model_seeds": list(MODEL_SEEDS),
            "fine_tasks": list(FINE_TASKS),
            "wt_draw_indices": [0, 1, 2, 3],
            "repeated_wt_draw_seeds": list(WT_DRAW_SEEDS),
            "prototype_ids": list(range(CANONICAL_K)),
            "full_source_grid": [
                {"k": k, "kmeans_seed": seed}
                for k in VARIANT_K
                for seed in VARIANT_SEEDS
            ],
        },
        "variables": {
            "kras": {
                "type": "binary",
                "contrast": "mutant-minus-wild_type",
                "levels": ["wild_type", "mutant"],
                "reference": "wild_type",
            },
            "msi_dmmr": {
                "type": "binary",
                "contrast": "MSI-H/dMMR-minus-MSS/pMMR",
                "levels": ["MSS/pMMR", "MSI-H/dMMR"],
                "reference": "MSS/pMMR",
            },
            "braf": {
                "type": "binary",
                "contrast": "mutant-minus-wild_type",
                "levels": ["wild_type", "mutant"],
                "reference": "wild_type",
            },
            "sex": {
                "type": "binary",
                "contrast": "male-minus-female",
                "levels": ["female", "male"],
                "reference": "female",
            },
            "age_at_diagnosis": {"type": "continuous", "unit": "10 years"},
            "tumor_site_group": {
                "type": "categorical",
                "levels": sorted_utf8(patients.get("tumor_site_group", [])),
            },
            "stage_group_major": {
                "type": "categorical",
                "levels": sorted_utf8(patients.get("stage_group_major", [])),
            },
            "source_subcohort": {
                "type": "categorical",
                "levels": list(SOURCE_SUBCOHORTS),
                "reference": SOURCE_SUBCOHORTS[0],
            },
        },
        "clinical_encoder": {
            "scope": "fit within each training fold/split only",
            "age": "training-fold median, scale, and missing indicator",
            "categorical": (
                "UTF-8-byte sorted training levels; first dropped; explicit unknown"
            ),
            "feature_order": [
                "age_at_diagnosis",
                "age_at_diagnosis_missing",
                "sex",
                "tumor_site_group",
                "stage_group_major",
            ],
        },
        "vocabulary": {
            "sample_tiles": SAMPLE_TILES,
            "normalize": "l2",
            "pca_components": PCA_COMPONENTS,
            "pca_whiten": False,
            "pca_solver": "randomized",
            "pca_seed": VOCAB_SEED,
            "kmeans_algorithm": "lloyd",
            "kmeans_n_init": 10,
            "kmeans_max_iter": 300,
            "kmeans_tol": 1e-4,
            "canonical_k": CANONICAL_K,
            "canonical_seed": VOCAB_SEED,
            "mapping_cosine_threshold": 0.80,
        },
        "statuses": {
            "legacy_correspondence": "PENDING_PIN_PREFLIGHT",
            "name_gate": "PENDING_READER_RETURN",
            "grid_offset": "PENDING_NONBLOCKING_SENSITIVITY",
            "module_i": "NOT_RUN_BEFORE_READER_RETURN",
            "module_ii": "NOT_RUN_BEFORE_SOURCE_FREEZE",
            "module_iii": "NOT_RUN_BEFORE_READER_RETURN",
        },
    }


def implementation_contract() -> dict[str, Any]:
    """Resolve preregistration details that must not drift during execution."""

    return {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "created_utc": utc_now(),
        "randomness": {
            "numpy_generator": "numpy.random.Generator(PCG64)",
            "global_seed": VOCAB_SEED,
            "within_slide_draw": (
                "one PCG64 stream, seeded once, consumed in sorted "
                "subcohort/patient/slide order"
            ),
        },
        "tile_identifier": {
            "definition": "zero-based row in the hash-pinned packed store",
            "sort": "ascending integer; equivalent zero-padded lexical identifier",
            "coordinate_binding": "packed coords.bin row at the same global offset",
        },
        "allocation": {
            "levels": ["subcohort", "patient_id", "slide_id"],
            "order": "UTF-8 byte lexical at every string level",
            "remainders": "sorted round-robin among nonexhausted units at same level",
        },
        "montage": {
            "prototype_index": "zero-based rank in sorted prototype IDs 0..31",
            "distance_quantiles": (
                "numpy quantile method=linear over every source tile assigned to "
                "that prototype; patient candidate is its closest assigned tile"
            ),
            "expansion": ["distance<=q10", "distance<=q25", "all assigned patients"],
            "tile_px": 256,
            "canvas": "four columns by three rows for 12 tiles; white unused cells",
            "format": "RGB JPEG",
            "jpeg": {
                "quality": 92,
                "subsampling": 0,
                "optimize": False,
                "progressive": False,
            },
        },
        "compute_limits": {
            "resident_pca_kmeans_fits": 1,
            "gpu_jobs": 1,
            "profile_workers": MAX_PROFILE_WORKERS,
        },
    }


def _seal(path: Path, *, role: str) -> dict[str, Any]:
    seal_path = path.with_suffix(path.suffix + ".seal.json")
    if seal_path.is_file():
        record = json.loads(seal_path.read_text(encoding="utf-8"))
        if (
            record.get("role") != role
            or record.get("artifact", {}).get("sha256") != sha256_file(path)
        ):
            raise ContractError(f"Existing seal does not bind {path}: {seal_path}")
        return record
    record = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "role": role,
        "sealed_utc": utc_now(),
        "artifact": identity(path),
    }
    write_once(seal_path, json_bytes(record))
    return record


def prepare(output: Path) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require_hash(PREREGISTRATION, PREREGISTRATION_SHA256, "FINAL-v14 preregistration")
    if not FINAL_V13_SNAPSHOT.is_dir():
        raise ContractError(f"FINAL-v13 snapshot is missing: {FINAL_V13_SNAPSHOT}")
    # Byte-level tree equality is checked without writing either tree.
    left = {
        path.relative_to(FINAL_V13): sha256_file(path)
        for path in sorted(FINAL_V13.rglob("*"))
        if path.is_file()
    }
    right = {
        path.relative_to(FINAL_V13_SNAPSHOT): sha256_file(path)
        for path in sorted(FINAL_V13_SNAPSHOT.rglob("*"))
        if path.is_file()
    }
    if left != right:
        raise ContractError("FINAL-v13 snapshot is not byte-identical")

    frame = source_frame_from_master()
    inputs = output / "inputs"
    source_path = inputs / "tcga_surgen_primary.csv"
    write_once(source_path, frame.to_csv(index=False).encode("utf-8"))
    require_hash(source_path, SOURCE_MANIFEST_SHA256, "reproduced FINAL-v13 source manifest")

    split_dir = inputs / "splits"
    split_artifacts = ("splits.parquet", ".integrity_hash", "summary.json")
    if not all((split_dir / name).is_file() for name in split_artifacts):
        split_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=".splits.", dir=split_dir.parent)
        )
        try:
            derive_subset_splits(
                MASTER_SPLITS,
                source_path,
                stage,
                filename_column="slide_id",
                force=False,
            )
            for name in split_artifacts:
                write_once(split_dir / name, (stage / name).read_bytes())
        finally:
            shutil.rmtree(stage)
    split_path = split_dir / "splits.parquet"
    require_hash(split_path, SOURCE_SPLITS_SHA256, "reproduced FINAL-v13 source splits")
    splits = pd.read_parquet(split_path)
    blind = _label_blind_source(frame, splits)
    blind_path = inputs / "source_roster_label_blind.csv"
    write_once(blind_path, blind.to_csv(index=False).encode("utf-8"))

    dictionary_path = output / "analysis_dictionary.json"
    if dictionary_path.exists():
        dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    else:
        dictionary = build_analysis_dictionary(frame)
        write_once(dictionary_path, json_bytes(dictionary))
    dictionary_seal = _seal(dictionary_path, role="analysis_dictionary_pre_model_freeze")

    implementation_path = output / "implementation_contract.json"
    if implementation_path.exists():
        implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    else:
        implementation = implementation_contract()
        write_once(implementation_path, json_bytes(implementation))
    implementation_seal = _seal(
        implementation_path, role="pre_reader_implementation_freeze"
    )

    receipt_path = output / "receipts/prepare.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") != "PREPARED"
            or existing.get("inputs", {}).get("source_manifest", {}).get("sha256")
            != SOURCE_MANIFEST_SHA256
            or existing.get("inputs", {}).get("source_splits", {}).get("sha256")
            != SOURCE_SPLITS_SHA256
        ):
            raise ContractError("Existing prepare receipt fails replay")
        return existing

    result = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": "PREPARED",
        "created_utc": utc_now(),
        "governance": {
            "preregistration": identity(PREREGISTRATION),
            "final_v13_snapshot": str(FINAL_V13_SNAPSHOT.resolve()),
            "snapshot_diff": "PASS_BYTE_IDENTICAL",
        },
        "inputs": {
            "source_manifest": identity(source_path),
            "source_splits": identity(split_path),
            "label_blind_roster": identity(blind_path),
        },
        "analysis_dictionary": dictionary_seal,
        "implementation_contract": implementation_seal,
    }
    write_once(receipt_path, json_bytes(result))
    return result


def _validate_source_against_live_copy(output: Path) -> dict[str, Any]:
    reproduced = output / "inputs/tcga_surgen_primary.csv"
    derived = output / "inputs/splits/splits.parquet"
    records = {
        "reproduced_manifest": require_hash(
            reproduced, SOURCE_MANIFEST_SHA256, "reproduced source manifest"
        ),
        "reproduced_splits": require_hash(
            derived, SOURCE_SPLITS_SHA256, "reproduced source splits"
        ),
    }
    if V13_SOURCE_MANIFEST.is_file():
        records["live_manifest"] = require_hash(
            V13_SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256, "live FINAL-v13 source manifest"
        )
        if V13_SOURCE_MANIFEST.read_bytes() != reproduced.read_bytes():
            raise ContractError("Live and reproduced source manifests differ byte-wise")
    if V13_SOURCE_SPLITS.is_file():
        records["live_splits"] = require_hash(
            V13_SOURCE_SPLITS, SOURCE_SPLITS_SHA256, "live FINAL-v13 source splits"
        )
        if V13_SOURCE_SPLITS.read_bytes() != derived.read_bytes():
            raise ContractError("Live and reproduced source split files differ byte-wise")
    return records


def _validate_teacher_inventory() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    contract = require_hash(V13_CONTRACT, V13_CONTRACT_SHA256, "FINAL-v13 contract")
    seed_receipts: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        seed_receipt_path = (
            V13_CAMPAIGN_ROOT
            / f"receipts/source_cv/tcga_surgen_primary/seed{seed}.json"
        )
        seed_receipts.append(
            {
                "seed": seed,
                "artifact": require_hash(
                    seed_receipt_path,
                    V13_SOURCE_CV_RECEIPT_SHA256[seed],
                    f"FINAL-v13 source-CV seed {seed} receipt",
                ),
            }
        )
        for fold in FOLDS:
            fold_dir = V13_TEACHER_ROOT / f"seed{seed}/fold_{fold}"
            metrics_path = fold_dir / "fold_metrics.json"
            completion_path = fold_dir / "completion.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            raw_checkpoint = Path(str(metrics.get("best_checkpoint", "")))
            if raw_checkpoint.is_file():
                checkpoint = raw_checkpoint
            else:
                try:
                    relative = raw_checkpoint.relative_to(V13_CAMPAIGN_ROOT)
                except ValueError as exc:
                    raise ContractError(
                        f"seed{seed}/fold{fold}: checkpoint is outside v13 campaign"
                    ) from exc
                checkpoint = V13_CAMPAIGN_ROOT / relative
            if not checkpoint.is_file():
                raise ContractError(
                    f"seed{seed}/fold{fold}: exact metrics checkpoint is missing: {checkpoint}"
                )
            checkpoint_record = identity(checkpoint)
            completion_checkpoint = completion.get("artifacts", {}).get(
                "best_checkpoint", {}
            )
            completion_checkpoint_path = (
                fold_dir / str(completion_checkpoint.get("path", ""))
            ).resolve()
            completion_metrics = completion.get("artifacts", {}).get("metrics", {})
            if (
                completion_checkpoint_path != checkpoint.resolve()
                or completion_checkpoint.get("sha256") != checkpoint_record["sha256"]
                or int(completion_checkpoint.get("size_bytes", -1))
                != checkpoint_record["size_bytes"]
                or completion_metrics.get("sha256") != sha256_file(metrics_path)
                or int(completion_metrics.get("size_bytes", -1))
                != metrics_path.stat().st_size
            ):
                raise ContractError(
                    f"seed{seed}/fold{fold}: completion receipt does not bind "
                    "fold_metrics best checkpoint"
                )
            records.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "metrics": identity(metrics_path),
                    "completion": identity(completion_path),
                    "checkpoint": checkpoint_record,
                }
            )
    if len(records) != 25 or len({(r["seed"], r["fold"]) for r in records}) != 25:
        raise ContractError("The FINAL-v13 teacher inventory is not exactly 25 heads")
    return {
        "count": 25,
        "contract": contract,
        "seed_receipts": seed_receipts,
        "heads": records,
    }


def preflight(output: Path, *, deep_hash: bool) -> dict[str, Any]:
    destination = output / (
        "receipts/preflight.json"
        if deep_hash
        else "receipts/preflight_shallow.json"
    )
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        feature_record = (
            existing.get("pack", {}).get("artifacts", {}).get("features.bin", {})
        )
        if existing.get("status") != "PASS":
            raise ContractError("Existing preflight receipt is not a PASS")
        if deep_hash and not feature_record.get("hash_verified", False):
            raise ContractError("Production preflight receipt lacks its deep hash")
        if feature_record.get("expected_sha256") != PACK_FILES["features.bin"]["sha256"]:
            raise ContractError("Existing preflight receipt does not bind the governed pack")
        return existing
    if not (output / "receipts/prepare.json").is_file():
        raise ContractError("Run prepare before preflight")
    if not HOT_ROOT.is_dir():
        raise ContractError(
            "Governed hot volume is not mounted at /mnt/wsl/oceanpath-hot; "
            "native Windows VHDX attach is required"
        )
    source = _validate_source_against_live_copy(output)
    store = PackedFeatureStore(PACK_ROOT)
    if (
        store.meta.total_patches != 24_284_922
        or store.meta.feat_dim != 1_024
        or store.meta.feat_dtype != "float16"
        or not store.meta.has_coords
    ):
        raise ContractError(f"Packed UNI-v1 metadata drifted: {store.meta}")
    if not FEATURE_ROOT.is_dir():
        raise ContractError(f"UNI-v1 per-slide feature root is missing: {FEATURE_ROOT}")
    live_feature_inventory = feature_inventory_sha256(FEATURE_ROOT)
    if live_feature_inventory != store.meta.source_inventory_sha256:
        raise ContractError("Packed UNI-v1 source inventory no longer matches its H5 source")
    pack_records: dict[str, Any] = {}
    for name, expected in PACK_FILES.items():
        path = PACK_ROOT / name
        stat = path.stat()
        if int(stat.st_size) != int(expected["size_bytes"]):
            raise ContractError(f"Packed UNI-v1 {name} size drifted")
        record = identity(path, hash_content=deep_hash or name != "features.bin")
        if "sha256" in record and record["sha256"] != expected["sha256"]:
            raise ContractError(f"Packed UNI-v1 {name} SHA-256 drifted")
        record["expected_sha256"] = expected["sha256"]
        record["hash_verified"] = "sha256" in record
        pack_records[name] = record
    manifest = pd.read_csv(output / "inputs/source_roster_label_blind.csv")
    missing = sorted(set(manifest["slide_id"].astype(str)) - set(store.slide_ids))
    if missing:
        raise ContractError(f"Packed UNI-v1 lacks {len(missing)} source slides: {missing[:5]}")
    if not PATCH_ROOT.is_dir():
        raise ContractError(f"Patch-coordinate root is missing: {PATCH_ROOT}")
    missing_patches = [
        slide_id
        for slide_id in manifest["slide_id"].astype(str)
        if not (PATCH_ROOT / f"{slide_id}_patches.h5").is_file()
    ]
    if missing_patches:
        raise ContractError(
            f"Patch-coordinate root lacks {len(missing_patches)} source slides: "
            f"{missing_patches[:5]}"
        )
    reader_sources = _validate_reader_source_preflight(output, store)
    teacher_inventory = _validate_teacher_inventory()
    logits = require_hash(
        V13_PATIENT_LOGITS,
        V13_PATIENT_LOGITS_SHA256,
        "FINAL-v13 patient-native logits",
    )
    training = require_hash(
        V13_TRAINING_RECEIPT,
        V13_TRAINING_RECEIPT_SHA256,
        "FINAL-v13 training receipt",
    )
    legacy: dict[str, Any] = {}
    geometry_ok = True
    for name, (path, expected) in LEGACY_INPUTS.items():
        if not path.is_file():
            legacy[name] = {"path": str(path), "status": "MISSING"}
            geometry_ok = False
            continue
        try:
            legacy[name] = {"status": "PASS", **require_hash(path, expected, name)}
        except ContractError:
            legacy[name] = {
                "path": str(path.resolve(strict=False)),
                "status": "DIGEST_MISMATCH",
                "expected_sha256": expected,
                "observed_sha256": sha256_file(path),
            }
            geometry_ok = False
    correspondence = (
        "PINS_AVAILABLE_PENDING_NAMES"
        if geometry_ok
        else "CORRESPONDENCE_NOT_EVALUABLE"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": "PASS",
        "created_utc": utc_now(),
        "deep_hash": deep_hash,
        "source": source,
        "pack": {
            "root": str(PACK_ROOT.resolve()),
            "slides": len(store),
            "tiles": store.meta.total_patches,
            "feature_dim": store.meta.feat_dim,
            "dtype": store.meta.feat_dtype,
            "artifacts": pack_records,
            "live_source_inventory_sha256": live_feature_inventory,
        },
        "patch_coordinate_files": SOURCE_SLIDES,
        "reader_sources": reader_sources,
        "teachers": teacher_inventory,
        "v13_patient_logits": logits,
        "v13_training_receipt": training,
        "legacy_inputs": legacy,
        "legacy_correspondence_status": correspondence,
    }
    write_once(destination, json_bytes(result))
    return result


def _require_stage_receipt(
    output: Path, name: str, *, expected_status: str = "PASS"
) -> dict[str, Any]:
    path = output / f"receipts/{name}.json"
    if not path.is_file():
        raise ContractError(f"Required stage receipt is missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != expected_status:
        raise ContractError(
            f"Required stage {name!r} has status {result.get('status')!r}, "
            f"expected {expected_status!r}"
        )
    return result


def _require_production_preflight(output: Path) -> dict[str, Any]:
    receipt = _require_stage_receipt(output, "preflight")
    prepare_receipt = _require_stage_receipt(
        output, "prepare", expected_status="PREPARED"
    )
    feature_record = receipt.get("pack", {}).get("artifacts", {}).get("features.bin", {})
    if not receipt.get("deep_hash") or not feature_record.get("hash_verified"):
        raise ContractError(
            "Production computation requires `preflight --deep-hash` on the exact pack"
        )
    if feature_record.get("sha256") != PACK_FILES["features.bin"]["sha256"]:
        raise ContractError("Production preflight does not authenticate features.bin")
    require_hash(PREREGISTRATION, PREREGISTRATION_SHA256, "FINAL-v14 preregistration")
    _validate_artifact_tree(prepare_receipt.get("inputs", {}), "prepared source inputs")
    _validate_artifact_tree(
        prepare_receipt.get("analysis_dictionary", {}), "analysis dictionary seal"
    )
    _validate_artifact_tree(
        prepare_receipt.get("implementation_contract", {}),
        "implementation contract seal",
    )
    _validate_artifact_tree(receipt.get("source", {}), "preflight source")
    pack_artifacts = receipt.get("pack", {}).get("artifacts", {})
    for name in ("meta.json", "index.parquet", "coords.bin"):
        record = pack_artifacts.get(name, {})
        if record.get("sha256") != PACK_FILES[name]["sha256"]:
            raise ContractError(f"Production preflight does not authenticate {name}")
        _validate_identity_record(record, f"packed UNI-v1 {name}")
    _validate_identity_record(
        receipt.get("v13_patient_logits", {}), "FINAL-v13 patient logits"
    )
    _validate_identity_record(
        receipt.get("v13_training_receipt", {}), "FINAL-v13 training receipt"
    )
    _validate_artifact_tree(receipt.get("teachers", {}), "FINAL-v13 teachers")
    _validate_artifact_tree(receipt.get("legacy_inputs", {}), "legacy inputs")
    _validate_artifact_tree(receipt.get("reader_sources", {}), "reader sources")
    live = identity(PACK_ROOT / "features.bin", hash_content=False)
    if any(
        live.get(key) != feature_record.get(key)
        for key in ("size_bytes", "mtime_ns")
    ):
        raise ContractError("features.bin changed after its deep-hash preflight")
    return receipt


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def _encoded_json_array(value: Any) -> np.ndarray:
    return np.frombuffer(json_bytes(value), dtype=np.uint8).copy()


def _decoded_json_array(value: np.ndarray) -> dict[str, Any]:
    return json.loads(np.asarray(value, dtype=np.uint8).tobytes().decode("utf-8"))


def _load_json_receipt_if_valid(
    path: Path, *, status: str, artifact_keys: Sequence[str] = ()
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != status:
        raise ContractError(f"Existing receipt has unexpected status: {path}")
    artifacts = record.get("artifacts", {})
    for key in artifact_keys:
        artifact = artifacts.get(key)
        if not isinstance(artifact, dict) or "path" not in artifact or "sha256" not in artifact:
            raise ContractError(f"Existing receipt lacks artifact {key!r}: {path}")
        candidate = Path(artifact["path"])
        if not candidate.is_file() or sha256_file(candidate) != artifact["sha256"]:
            raise ContractError(f"Existing receipt artifact fails replay: {candidate}")
    return record


def _validate_identity_record(record: Mapping[str, Any], context: str) -> None:
    if "path" not in record or "sha256" not in record:
        raise ContractError(f"Missing path/digest in {context}")
    path = Path(str(record["path"]))
    if not path.is_file():
        raise ContractError(f"Missing artifact in {context}: {path}")
    if int(path.stat().st_size) != int(record.get("size_bytes", -1)):
        raise ContractError(f"Artifact size drift in {context}: {path}")
    if sha256_file(path) != record["sha256"]:
        raise ContractError(f"Artifact digest drift in {context}: {path}")


def _validate_artifact_tree(value: Any, context: str) -> None:
    """Validate every file identity nested below an artifact/dependency record."""

    if isinstance(value, Mapping):
        if {"path", "size_bytes", "sha256"}.issubset(value):
            _validate_identity_record(value, context)
            return
        for key, child in value.items():
            _validate_artifact_tree(child, f"{context}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_artifact_tree(child, f"{context}[{index}]")


def _scope_directory(output: Path, fold: int | None) -> Path:
    if fold is None:
        return output / "vocabularies/reference"
    return output / f"vocabularies/outer_fold_{fold}"


def _vocabulary_path(
    output: Path, *, fold: int | None, k: int = CANONICAL_K, seed: int = VOCAB_SEED
) -> Path:
    scope = _scope_directory(output, fold)
    if fold is None:
        return scope / "variants" / f"k{k}_seed{seed}" / "vocabulary.npz"
    if k != CANONICAL_K or seed != VOCAB_SEED:
        raise ContractError("Outer vocabularies have only the canonical k and seed")
    return scope / "vocabulary.npz"


def _canonical_vocabulary_path(output: Path) -> Path:
    return _vocabulary_path(output, fold=None, k=CANONICAL_K, seed=VOCAB_SEED)


def _load_vocabulary(path: Path) -> V14Vocabulary:
    if not path.is_file():
        raise ContractError(f"Vocabulary is missing: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        required = {"centroids", "pca_mean", "pca_components", "config_json"}
        if missing := required - set(bundle.files):
            raise ContractError(f"Vocabulary {path} lacks arrays: {sorted(missing)}")
        vocabulary = V14Vocabulary(
            centroids=np.asarray(bundle["centroids"], dtype=np.float32),
            pca_mean=np.asarray(bundle["pca_mean"], dtype=np.float32),
            pca_components=np.asarray(bundle["pca_components"], dtype=np.float32),
            config=_decoded_json_array(bundle["config_json"]),
        )
    if not all(
        np.isfinite(array).all()
        for array in (
            vocabulary.centroids,
            vocabulary.pca_mean,
            vocabulary.pca_components,
        )
    ):
        raise ContractError(f"Vocabulary contains nonfinite arrays: {path}")
    return vocabulary


def _publish_vocabulary(path: Path, vocabulary: V14Vocabulary) -> dict[str, Any]:
    if path.is_file():
        replay = _load_vocabulary(path)
        if (
            replay.config != vocabulary.config
            or not np.array_equal(replay.centroids, vocabulary.centroids)
            or not np.array_equal(replay.pca_mean, vocabulary.pca_mean)
            or not np.array_equal(replay.pca_components, vocabulary.pca_components)
        ):
            raise ContractError(f"Existing vocabulary differs from replay: {path}")
        return identity(path)
    atomic_write(
        path,
        _npz_bytes(
            centroids=np.asarray(vocabulary.centroids, dtype=np.float32),
            pca_mean=np.asarray(vocabulary.pca_mean, dtype=np.float32),
            pca_components=np.asarray(vocabulary.pca_components, dtype=np.float32),
            config_json=_encoded_json_array(dict(vocabulary.config)),
        ),
    )
    return identity(path)


def _blind_roster_with_counts(output: Path, store: PackedFeatureStore) -> pd.DataFrame:
    roster = pd.read_csv(output / "inputs/source_roster_label_blind.csv", low_memory=False)
    required = {"slide_id", "patient_id", "subcohort", "fold"}
    if missing := required - set(roster):
        raise ContractError(f"Label-blind source roster lacks columns: {sorted(missing)}")
    roster = roster.copy()
    for column in ("slide_id", "patient_id", "subcohort"):
        roster[column] = roster[column].astype(str)
    roster["fold"] = pd.to_numeric(roster["fold"], errors="raise").astype(int)
    roster["n_tiles"] = roster["slide_id"].map(store.length_of).astype(np.int64)
    declared_counts = pd.to_numeric(roster["patch_count"], errors="raise").astype(np.int64)
    if not np.array_equal(declared_counts.to_numpy(), roster["n_tiles"].to_numpy()):
        raise ContractError("Source manifest patch counts do not match the governed pack")
    pack_index = pd.read_parquet(PACK_ROOT / "index.parquet", columns=["slide_id", "offset"])
    pack_index["slide_id"] = pack_index["slide_id"].astype(str)
    offset_by_slide = pack_index.set_index("slide_id")["offset"].astype(np.int64)
    roster["global_offset"] = roster["slide_id"].map(offset_by_slide)
    if roster["global_offset"].isna().any():
        raise ContractError("A source slide has no global packed-store offset")
    roster["global_offset"] = roster["global_offset"].astype(np.int64)
    if len(roster) != SOURCE_SLIDES or roster["patient_id"].nunique() != SOURCE_PATIENTS:
        raise ContractError("Label-blind source roster census drifted")
    if int(roster["n_tiles"].sum()) != SOURCE_TILES:
        raise ContractError("Source tile census drifted from 16,711,039")
    if roster.groupby("patient_id")["fold"].nunique().max() != 1:
        raise ContractError("A source patient crosses outer folds")
    return roster.sort_values(
        ["subcohort", "patient_id", "slide_id"], kind="mergesort"
    ).reset_index(drop=True)


def _materialize_sample_ids(
    output: Path,
    roster: pd.DataFrame,
    *,
    fold: int | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    scope = _scope_directory(output, fold)
    eligible = roster if fold is None else roster.loc[roster["fold"].ne(int(fold))]
    if fold is not None:
        heldout_patients = set(roster.loc[roster["fold"].eq(int(fold)), "patient_id"])
        if heldout_patients & set(eligible["patient_id"]):
            raise ContractError(f"Outer-training vocabulary {fold} contains a held-out patient")
    plan = hierarchical_sample_plan(
        eligible[["subcohort", "patient_id", "slide_id", "n_tiles"]],
        cap=SAMPLE_TILES,
    )
    if int(plan["n_sample"].sum()) != SAMPLE_TILES:
        raise ContractError(f"Vocabulary scope {fold=} cannot supply {SAMPLE_TILES} tiles")
    if plan.groupby("subcohort")["n_sample"].sum().to_dict() != {
        subcohort: 100_000 for subcohort in SOURCE_SUBCOHORTS
    }:
        raise ContractError(f"Vocabulary scope {fold=} is not balanced 100k/subcohort")
    if fold is not None:
        expected_patients = (991, 992, 991, 991, 991)
        expected_slides = (1111, 1112, 1112, 1110, 1111)
        if (
            eligible["patient_id"].nunique() != expected_patients[fold]
            or len(eligible) != expected_slides[fold]
        ):
            raise ContractError(f"Outer-training census drifted for fold {fold}")
    plan_path = scope / "sample_plan.parquet"
    write_once(plan_path, parquet_bytes(plan))

    sample_path = scope / "sample_ids.parquet"
    rng = pcg64_rng(VOCAB_SEED)
    blocks: list[pd.DataFrame] = []
    global_offsets = roster.set_index("slide_id")["global_offset"].astype(np.int64)
    for row in plan.itertuples(index=False):
        n_tiles = int(row.n_tiles)
        n_sample = int(row.n_sample)
        drawn = np.asarray(
            rng.choice(n_tiles, size=n_sample, replace=False), dtype=np.int64
        )
        order = np.argsort(drawn, kind="stable")
        positions = drawn[order]
        draw_indices = np.arange(n_sample, dtype=np.int64)[order]
        blocks.append(
            pd.DataFrame(
                {
                    "subcohort": str(row.subcohort),
                    "patient_id": str(row.patient_id),
                    "slide_id": str(row.slide_id),
                    "tile_id": [f"{value:012d}" for value in positions],
                    "tile_index": positions,
                    "global_tile_index": (
                        positions + int(global_offsets.loc[str(row.slide_id)])
                    ),
                    "within_slide_draw_index": draw_indices,
                }
            )
        )
    sample_ids = pd.concat(blocks, ignore_index=True)
    sample_ids["sample_index"] = np.arange(len(sample_ids), dtype=np.int64)
    write_once(sample_path, parquet_bytes(sample_ids))

    required = {
        "subcohort",
        "patient_id",
        "slide_id",
        "tile_id",
        "tile_index",
        "global_tile_index",
        "within_slide_draw_index",
        "sample_index",
    }
    if set(sample_ids) != required or len(sample_ids) != SAMPLE_TILES:
        raise ContractError(f"Sample-ID census is malformed: {sample_path}")
    if sample_ids.duplicated(["slide_id", "tile_index"]).any():
        raise ContractError(f"Sample-ID census repeats a tile: {sample_path}")
    if sample_ids["sample_index"].tolist() != list(range(SAMPLE_TILES)):
        raise ContractError(f"Sample-ID census has noncanonical row order: {sample_path}")
    if not np.array_equal(
        sample_ids["tile_id"].to_numpy(),
        sample_ids["tile_index"].map(lambda value: f"{int(value):012d}").to_numpy(),
    ):
        raise ContractError(f"Sample tile identifiers do not bind local rows: {sample_path}")
    expected_global = sample_ids["slide_id"].map(global_offsets).to_numpy(dtype=np.int64)
    expected_global += sample_ids["tile_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        expected_global, sample_ids["global_tile_index"].to_numpy(dtype=np.int64)
    ):
        raise ContractError(f"Sample global row binding failed: {sample_path}")
    expected_per_slide = plan.set_index("slide_id")["n_sample"].astype(int)
    observed_per_slide = sample_ids.groupby("slide_id", sort=False).size()
    if not observed_per_slide.equals(expected_per_slide.loc[observed_per_slide.index]):
        raise ContractError(f"Sample-ID census does not replay its plan: {sample_path}")
    return sample_ids, identity(plan_path), identity(sample_path)


def _load_sample_matrix(
    store: PackedFeatureStore, sample_ids: pd.DataFrame
) -> np.ndarray:
    sample = np.empty((len(sample_ids), store.feat_dim), dtype=np.float32)
    for slide_id, rows in sample_ids.groupby("slide_id", sort=False):
        indices = rows["tile_index"].to_numpy(dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= store.length_of(str(slide_id))):
            raise ContractError(f"Sample tile index lies outside slide {slide_id}")
        features = store.read_features(store.position(str(slide_id)), rows=indices)
        destinations = rows["sample_index"].to_numpy(dtype=np.int64)
        sample[destinations] = np.asarray(features, dtype=np.float32)
    if not np.isfinite(sample).all():
        raise ContractError("Vocabulary sample contains nonfinite embeddings")
    return sample


def _basis_path(output: Path, fold: int | None) -> Path:
    return _scope_directory(output, fold) / "pca_basis.npz"


def _projection_path(output: Path, fold: int | None) -> Path:
    return _scope_directory(output, fold) / "sample_pca64.float32.npy"


def _checkpoint_receipt_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.receipt.json")


def _fit_source_identities() -> list[dict[str, Any]]:
    return [
        identity(Path(__file__).resolve()),
        identity(REPO / "src/oceanpath/aim1/v14_concepts.py"),
        identity(REPO / "pyproject.toml"),
        identity(REPO / "uv.lock"),
    ]


def _freeze_checkpoint_receipt(
    path: Path,
    *,
    role: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    dependencies: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = _checkpoint_receipt_path(path)
    core = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": "PASS",
        "role": role,
        "artifacts": dict(artifacts),
        "dependencies": dict(dependencies),
        "metadata": dict(metadata),
    }
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if {key: value for key, value in receipt.items() if key != "created_utc"} != core:
            raise ContractError(f"Checkpoint receipt dependencies drifted: {receipt_path}")
        _validate_artifact_tree(receipt["artifacts"], f"{role} checkpoint")
    else:
        receipt = {**core, "created_utc": utc_now()}
        write_once(receipt_path, json_bytes(receipt))
    return identity(receipt_path)


def _freeze_first_fit_benchmark(
    output: Path,
    *,
    job_class: str,
    benchmark: Mapping[str, Any],
    checkpoint_receipt: Mapping[str, Any],
    projected_units: int,
) -> dict[str, Any]:
    path = output / f"benchmarks/{job_class}_first_completed.json"
    if not path.is_file():
        elapsed = float(benchmark["elapsed_seconds"])
        write_once(
            path,
            json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "component": COMPONENT,
                    "status": "PASS",
                    "created_utc": utc_now(),
                    **dict(benchmark),
                    "projected_units": int(projected_units),
                    "projected_seconds": elapsed * int(projected_units),
                    "checkpoint_receipt": dict(checkpoint_receipt),
                }
            ),
        )
    return identity(path)


def _fit_or_load_basis(
    output: Path,
    store: PackedFeatureStore,
    roster: pd.DataFrame,
    *,
    fold: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    sample_ids, plan_record, sample_record = _materialize_sample_ids(
        output, roster, fold=fold
    )
    software = _software_versions()
    implementation_sources = _fit_source_identities()
    scope_name = "full_source" if fold is None else f"outer_training_fold_{fold}"
    path = _basis_path(output, fold)
    projection_path = _projection_path(output, fold)
    benchmarks: list[dict[str, Any]] = []
    started = time.monotonic()
    if path.is_file():
        if not projection_path.is_file():
            raise ContractError(
                f"PCA basis exists without its byte-frozen coordinates: {path}"
            )
        with np.load(path, allow_pickle=False) as bundle:
            if set(bundle.files) != {"pca_mean", "pca_components", "config_json"}:
                raise ContractError(f"Existing PCA basis schema drifted: {path}")
            mean = np.asarray(bundle["pca_mean"], dtype=np.float32)
            components = np.asarray(bundle["pca_components"], dtype=np.float32)
            config = _decoded_json_array(bundle["config_json"])
        expected_config = {
            "schema_version": SCHEMA_VERSION,
            "scope": scope_name,
            "heldout_fold": fold,
            "sample_ids_sha256": sample_record["sha256"],
            "sample_plan_sha256": plan_record["sha256"],
            "n_sample_tiles": SAMPLE_TILES,
            "parameters": pca64_parameters(seed=VOCAB_SEED),
            "software": software,
            "implementation_sources": implementation_sources,
        }
        if any(config.get(key) != value for key, value in expected_config.items()):
            raise ContractError(f"Existing PCA basis contract drifted: {path}")
        if config.get("projected_coordinates_sha256") != sha256_file(projection_path):
            raise ContractError(f"PCA coordinates fail their basis binding: {projection_path}")
        projected = np.load(projection_path, allow_pickle=False, mmap_mode="r")
        original_elapsed = float(config.get("fit_elapsed_seconds", math.nan))
        if not math.isfinite(original_elapsed) or original_elapsed <= 0:
            raise ContractError(f"PCA basis lacks a valid fit benchmark: {path}")
        action = "replayed"
    else:
        fit_started = time.monotonic()
        sample = _load_sample_matrix(store, sample_ids)
        normalized = l2_normalize_rows(sample)
        del sample
        with threadpool_limits(limits=1):
            pca = PCA(**pca64_parameters(seed=VOCAB_SEED))
            pca.fit(normalized)
            projected = np.asarray(pca.transform(normalized), dtype=np.float32)
        del normalized
        mean = np.asarray(pca.mean_, dtype=np.float32)
        components = np.asarray(pca.components_, dtype=np.float32)
        original_elapsed = time.monotonic() - fit_started
        atomic_write(projection_path, _npy_bytes(projected))
        config = {
            "schema_version": SCHEMA_VERSION,
            "scope": scope_name,
            "heldout_fold": fold,
            "sample_ids_sha256": sample_record["sha256"],
            "sample_plan_sha256": plan_record["sha256"],
            "projected_coordinates_sha256": sha256_file(projection_path),
            "n_sample_tiles": SAMPLE_TILES,
            "parameters": pca64_parameters(seed=VOCAB_SEED),
            "software": software,
            "implementation_sources": implementation_sources,
            "fit_elapsed_seconds": original_elapsed,
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        }
        atomic_write(
            path,
            _npz_bytes(
                pca_mean=mean,
                pca_components=components,
                config_json=_encoded_json_array(config),
            ),
        )
        action = "fit"
    if projected.shape != (SAMPLE_TILES, PCA_COMPONENTS) or not np.isfinite(projected).all():
        raise ContractError(f"PCA projection has invalid shape or values: {projected.shape}")
    if mean.shape != (store.feat_dim,) or components.shape != (
        PCA_COMPONENTS,
        store.feat_dim,
    ):
        raise ContractError(f"PCA basis arrays have invalid shapes: {path}")
    if not np.isfinite(mean).all() or not np.isfinite(components).all():
        raise ContractError(f"PCA basis arrays contain nonfinite values: {path}")
    basis_checkpoint_receipt = _freeze_checkpoint_receipt(
        path,
        role="pca_basis_and_coordinates",
        artifacts={
            "basis": identity(path),
            "projected_coordinates": identity(projection_path),
            "sample_plan": plan_record,
            "sample_ids": sample_record,
        },
        dependencies={
            "software": software,
            "implementation_sources": implementation_sources,
            "parameters": pca64_parameters(seed=VOCAB_SEED),
        },
        metadata={
            "scope": scope_name,
            "heldout_fold": fold,
            "fit_elapsed_seconds": original_elapsed,
        },
    )
    elapsed = time.monotonic() - started
    benchmark = {
        "job_class": "pca64",
        "scope": config["scope"],
        "action": action,
        "elapsed_seconds": original_elapsed,
        "tiles_per_second": SAMPLE_TILES / max(original_elapsed, 1e-12),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "replay_validation_seconds": elapsed if action == "replayed" else 0.0,
    }
    benchmarks.append(benchmark)
    _freeze_first_fit_benchmark(
        output,
        job_class="pca64",
        benchmark=benchmark,
        checkpoint_receipt=basis_checkpoint_receipt,
        projected_units=6,
    )
    return projected, mean, components, config, benchmarks


def _fit_scope_vocabularies(
    output: Path,
    store: PackedFeatureStore,
    roster: pd.DataFrame,
    *,
    fold: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    projected, mean, components, basis_config, benchmarks = _fit_or_load_basis(
        output, store, roster, fold=fold
    )
    if fold is None:
        canonical = (CANONICAL_K, VOCAB_SEED)
        grid = [(k, seed) for k in VARIANT_K for seed in VARIANT_SEEDS]
        specifications = [canonical, *[item for item in grid if item != canonical]]
    else:
        specifications = [(CANONICAL_K, VOCAB_SEED)]

    artifacts: list[dict[str, Any]] = []
    for k, seed in specifications:
        path = _vocabulary_path(output, fold=fold, k=k, seed=seed)
        if path.is_file():
            vocabulary = _load_vocabulary(path)
            expected_config = {
                "schema_version": SCHEMA_VERSION,
                "scope": basis_config["scope"],
                "heldout_fold": fold,
                "n_prototypes": k,
                "kmeans_seed": seed,
                "normalize": "l2",
                "pca": pca64_parameters(seed=VOCAB_SEED),
                "kmeans": lloyd_kmeans_parameters(k, seed=seed),
                "sample_ids_sha256": basis_config["sample_ids_sha256"],
                "sample_plan_sha256": basis_config["sample_plan_sha256"],
                "n_sample_tiles": SAMPLE_TILES,
                "pca_basis_sha256": sha256_file(_basis_path(output, fold)),
                "projected_coordinates_sha256": sha256_file(
                    _projection_path(output, fold)
                ),
                "software": basis_config["software"],
                "implementation_sources": basis_config["implementation_sources"],
            }
            if (
                vocabulary.n_prototypes != k
                or any(
                    vocabulary.config.get(key) != value
                    for key, value in expected_config.items()
                )
                or not np.array_equal(vocabulary.pca_mean, mean)
                or not np.array_equal(vocabulary.pca_components, components)
            ):
                raise ContractError(f"Existing vocabulary has incompatible metadata: {path}")
            original_elapsed = float(
                vocabulary.config.get("fit_elapsed_seconds", math.nan)
            )
            if not math.isfinite(original_elapsed) or original_elapsed <= 0:
                raise ContractError(f"Vocabulary lacks a valid fit benchmark: {path}")
            action = "replayed"
            elapsed = 0.0
        else:
            started = time.monotonic()
            with threadpool_limits(limits=1):
                model = KMeans(**lloyd_kmeans_parameters(k, seed=seed)).fit(projected)
            elapsed = time.monotonic() - started
            original_elapsed = elapsed
            config = {
                "schema_version": SCHEMA_VERSION,
                "scope": basis_config["scope"],
                "heldout_fold": fold,
                "n_prototypes": k,
                "kmeans_seed": seed,
                "normalize": "l2",
                "pca": pca64_parameters(seed=VOCAB_SEED),
                "kmeans": lloyd_kmeans_parameters(k, seed=seed),
                "sample_ids_sha256": basis_config["sample_ids_sha256"],
                "sample_plan_sha256": basis_config["sample_plan_sha256"],
                "n_sample_tiles": SAMPLE_TILES,
                "pca_basis_sha256": sha256_file(_basis_path(output, fold)),
                "projected_coordinates_sha256": sha256_file(
                    _projection_path(output, fold)
                ),
                "software": basis_config["software"],
                "implementation_sources": basis_config["implementation_sources"],
                "fit_elapsed_seconds": original_elapsed,
                "explained_variance_ratio": basis_config[
                    "explained_variance_ratio"
                ],
                "inertia": float(model.inertia_),
                "n_iter": int(model.n_iter_),
                "cluster_sizes": np.bincount(model.labels_, minlength=k)
                .astype(int)
                .tolist(),
            }
            vocabulary = V14Vocabulary(
                centroids=np.asarray(model.cluster_centers_, dtype=np.float32),
                pca_mean=mean,
                pca_components=components,
                config=config,
            )
            _publish_vocabulary(path, vocabulary)
            action = "fit"
        checkpoint_receipt = _freeze_checkpoint_receipt(
            path,
            role="lloyd_kmeans_vocabulary",
            artifacts={
                "vocabulary": identity(path),
                "pca_basis": identity(_basis_path(output, fold)),
                "projected_coordinates": identity(_projection_path(output, fold)),
            },
            dependencies={
                "sample_ids_sha256": basis_config["sample_ids_sha256"],
                "sample_plan_sha256": basis_config["sample_plan_sha256"],
                "software": basis_config["software"],
                "implementation_sources": basis_config["implementation_sources"],
                "pca": pca64_parameters(seed=VOCAB_SEED),
                "kmeans": lloyd_kmeans_parameters(k, seed=seed),
            },
            metadata={
                "scope": basis_config["scope"],
                "heldout_fold": fold,
                "k": k,
                "kmeans_seed": seed,
                "fit_elapsed_seconds": original_elapsed,
            },
        )
        record = {
            "scope": vocabulary.config["scope"],
            "heldout_fold": fold,
            "k": k,
            "kmeans_seed": seed,
            "artifact": identity(path),
            "checkpoint_receipt": checkpoint_receipt,
            "inertia": float(vocabulary.config["inertia"]),
            "n_iter": int(vocabulary.config["n_iter"]),
            "cluster_sizes": vocabulary.config["cluster_sizes"],
        }
        artifacts.append(record)
        benchmark = {
            "job_class": "lloyd_kmeans",
            "scope": vocabulary.config["scope"],
            "k": k,
            "seed": seed,
            "action": action,
            "elapsed_seconds": original_elapsed,
            "tiles_per_second": SAMPLE_TILES / max(original_elapsed, 1e-12),
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "replay_validation_seconds": elapsed if action == "replayed" else 0.0,
        }
        benchmarks.append(benchmark)
        _freeze_first_fit_benchmark(
            output,
            job_class="lloyd_kmeans",
            benchmark=benchmark,
            checkpoint_receipt=checkpoint_receipt,
            projected_units=14,
        )
    return artifacts, benchmarks


def _software_versions() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def _threadpool_runtime() -> list[dict[str, Any]]:
    from threadpoolctl import threadpool_info

    return [
        {
            key: value
            for key, value in record.items()
            if key
            in {
                "user_api",
                "internal_api",
                "num_threads",
                "prefix",
                "version",
                "threading_layer",
                "architecture",
            }
        }
        for record in threadpool_info()
    ]


def _freeze_outer_mappings(output: Path) -> dict[str, Any]:
    receipt_path = output / "receipts/mappings.json"
    replay = _load_json_receipt_if_valid(
        receipt_path, status="PASS", artifact_keys=("fold_to_reference",)
    )
    if replay is not None:
        dependencies = replay.get("vocabulary_sha256", {})
        current_dependencies = {
            "reference": sha256_file(_canonical_vocabulary_path(output)),
            **{
                f"outer_fold_{fold}": sha256_file(_vocabulary_path(output, fold=fold))
                for fold in FOLDS
            },
        }
        if dependencies != current_dependencies:
            raise ContractError("Mapping receipt vocabulary dependencies drifted")
        table = pd.read_csv(replay["artifacts"]["fold_to_reference"]["path"])
        required = {
            "outer_fold",
            "source_prototype_id",
            "reference_prototype_id",
            "cosine_similarity",
            "name_mappable",
        }
        if set(table) != required or len(table) != len(FOLDS) * CANONICAL_K:
            raise ContractError("Mapping replay table has an invalid schema or census")
        for fold in FOLDS:
            block = table.loc[table["outer_fold"].eq(fold)]
            if (
                sorted(block["source_prototype_id"].astype(int))
                != list(range(CANONICAL_K))
                or block["reference_prototype_id"].nunique() != CANONICAL_K
                or not np.isfinite(block["cosine_similarity"]).all()
            ):
                raise ContractError(f"Mapping replay failed bijection gate for fold {fold}")
        return replay
    reference = _load_vocabulary(_canonical_vocabulary_path(output))
    blocks: list[pd.DataFrame] = []
    for fold in FOLDS:
        vocabulary = _load_vocabulary(_vocabulary_path(output, fold=fold))
        mapping = map_vocabulary_to_reference(vocabulary, reference)
        mapping.insert(0, "outer_fold", fold)
        blocks.append(mapping)
    table = pd.concat(blocks, ignore_index=True)
    if len(table) != len(FOLDS) * CANONICAL_K:
        raise ContractError("Fold-to-reference mapping census is incomplete")
    destination = output / "mappings/outer_to_reference.csv"
    write_once(destination, table.to_csv(index=False).encode("utf-8"))
    result = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": "PASS",
        "created_utc": utc_now(),
        "threshold": 0.80,
        "artifacts": {"fold_to_reference": identity(destination)},
        "vocabulary_sha256": {
            "reference": sha256_file(_canonical_vocabulary_path(output)),
            **{
                f"outer_fold_{fold}": sha256_file(_vocabulary_path(output, fold=fold))
                for fold in FOLDS
            },
        },
        "fold_summary": [
            {
                "outer_fold": fold,
                "name_mappable": int(
                    table.loc[table["outer_fold"].eq(fold), "name_mappable"].sum()
                ),
                "minimum_cosine": float(
                    table.loc[table["outer_fold"].eq(fold), "cosine_similarity"].min()
                ),
            }
            for fold in FOLDS
        ],
        "all32_invariance": "MAPPING_DOES_NOT_REINDEX_OR_DROP_TECHNICAL_PREDICTORS",
    }
    write_once(receipt_path, json_bytes(result))
    return result


def fit_vocabularies(output: Path) -> dict[str, Any]:
    """Fit and freeze six PCA bases and fourteen preregistered dictionaries."""

    _require_production_preflight(output)
    receipt_path = output / "receipts/vocabularies.json"
    replay = _load_json_receipt_if_valid(
        receipt_path, status="PASS", artifact_keys=("inventory",)
    )
    if replay is not None:
        artifacts = replay.get("artifacts", {})
        bases = artifacts.get("pca_bases", [])
        projections = artifacts.get("projected_coordinates", [])
        vocabularies = artifacts.get("vocabularies", [])
        plans = artifacts.get("sample_plans", [])
        samples = artifacts.get("sample_ids", [])
        basis_receipts = artifacts.get("pca_checkpoint_receipts", [])
        fit_benchmarks = artifacts.get("first_completed_benchmarks", [])
        if (
            len(bases) != 6
            or len(projections) != 6
            or len(vocabularies) != 14
            or len(plans) != 6
            or len(samples) != 6
            or len(basis_receipts) != 6
            or len(fit_benchmarks) != 2
        ):
            raise ContractError("Vocabulary receipt artifact census drifted")
        _validate_artifact_tree(artifacts, "vocabulary receipt")
        store = PackedFeatureStore(PACK_ROOT)
        roster = _blind_roster_with_counts(output, store)
        _fit_scope_vocabularies(output, store, roster, fold=None)
        for fold in FOLDS:
            _fit_scope_vocabularies(output, store, roster, fold=fold)
        _freeze_outer_mappings(output)
        return replay
    with exclusive_lock(output / "locks/vocabularies.lock", "vocabulary fitting"):
        store = PackedFeatureStore(PACK_ROOT)
        roster = _blind_roster_with_counts(output, store)
        artifacts: list[dict[str, Any]] = []
        benchmarks: list[dict[str, Any]] = []
        scope_artifacts, scope_benchmarks = _fit_scope_vocabularies(
            output, store, roster, fold=None
        )
        artifacts.extend(scope_artifacts)
        benchmarks.extend(scope_benchmarks)
        for fold in FOLDS:
            scope_artifacts, scope_benchmarks = _fit_scope_vocabularies(
                output, store, roster, fold=fold
            )
            artifacts.extend(scope_artifacts)
            benchmarks.extend(scope_benchmarks)
        if len(artifacts) != 14:
            raise ContractError(f"Expected 14 vocabulary artifacts, got {len(artifacts)}")
        basis_paths = [_basis_path(output, None), *[_basis_path(output, fold) for fold in FOLDS]]
        projection_paths = [
            _projection_path(output, None),
            *[_projection_path(output, fold) for fold in FOLDS],
        ]
        scope_directories = [
            _scope_directory(output, None),
            *[_scope_directory(output, fold) for fold in FOLDS],
        ]
        plan_paths = [directory / "sample_plan.parquet" for directory in scope_directories]
        sample_paths = [directory / "sample_ids.parquet" for directory in scope_directories]
        if len(basis_paths) != 6 or not all(path.is_file() for path in basis_paths):
            raise ContractError("Expected six completed PCA bases")
        if not all(path.is_file() for path in projection_paths):
            raise ContractError("Expected six byte-frozen PCA coordinate matrices")
        inventory = pd.DataFrame(
            [
                {
                    "scope": row["scope"],
                    "heldout_fold": row["heldout_fold"],
                    "k": row["k"],
                    "kmeans_seed": row["kmeans_seed"],
                    "path": row["artifact"]["path"],
                    "sha256": row["artifact"]["sha256"],
                    "size_bytes": row["artifact"]["size_bytes"],
                }
                for row in artifacts
            ]
        )
        inventory_path = output / "vocabularies/inventory.csv"
        write_once(inventory_path, inventory.to_csv(index=False).encode("utf-8"))
        result = {
            "schema_version": SCHEMA_VERSION,
            "component": COMPONENT,
            "status": "PASS",
            "created_utc": utc_now(),
            "fit_census": {"pca_fits": 6, "kmeans_fits": 14, "kmeans_starts": 140},
            "sample_tiles_per_fit": SAMPLE_TILES,
            "software": _software_versions(),
            "threadpool_runtime_before_limits": _threadpool_runtime(),
            "compute_limits": {
                "resident_pca_kmeans_fits": 1,
                "blas_threads": 1,
            },
            "artifacts": {
                "inventory": identity(inventory_path),
                "pca_bases": [identity(path) for path in basis_paths],
                "projected_coordinates": [identity(path) for path in projection_paths],
                "sample_plans": [identity(path) for path in plan_paths],
                "sample_ids": [identity(path) for path in sample_paths],
                "pca_checkpoint_receipts": [
                    identity(_checkpoint_receipt_path(path)) for path in basis_paths
                ],
                "first_completed_benchmarks": [
                    identity(output / "benchmarks/pca64_first_completed.json"),
                    identity(
                        output / "benchmarks/lloyd_kmeans_first_completed.json"
                    ),
                ],
                "vocabularies": artifacts,
            },
            "benchmarks": benchmarks,
        }
        write_once(receipt_path, json_bytes(result))
    _freeze_outer_mappings(output)
    return result


def _validate_vocabulary_stage(
    output: Path, receipt: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    record = (
        dict(receipt)
        if receipt is not None
        else _require_stage_receipt(output, "vocabularies")
    )
    if record.get("fit_census") != {
        "pca_fits": 6,
        "kmeans_fits": 14,
        "kmeans_starts": 140,
    }:
        raise ContractError("Vocabulary fit census drifted")
    artifacts = record.get("artifacts", {})
    _validate_artifact_tree(artifacts, "vocabulary stage")
    current_software = _software_versions()
    current_sources = _fit_source_identities()
    scopes = [None, *FOLDS]
    for fold in scopes:
        basis_path = _basis_path(output, fold)
        with np.load(basis_path, allow_pickle=False) as bundle:
            config = _decoded_json_array(bundle["config_json"])
        scope_name = (
            "full_source" if fold is None else f"outer_training_fold_{fold}"
        )
        plan_path = _scope_directory(output, fold) / "sample_plan.parquet"
        sample_path = _scope_directory(output, fold) / "sample_ids.parquet"
        projection_path = _projection_path(output, fold)
        expected = {
            "schema_version": SCHEMA_VERSION,
            "scope": scope_name,
            "heldout_fold": fold,
            "sample_ids_sha256": sha256_file(sample_path),
            "sample_plan_sha256": sha256_file(plan_path),
            "projected_coordinates_sha256": sha256_file(projection_path),
            "n_sample_tiles": SAMPLE_TILES,
            "parameters": pca64_parameters(seed=VOCAB_SEED),
            "software": current_software,
            "implementation_sources": current_sources,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise ContractError(f"PCA basis stage contract drifted: {basis_path}")
    vocabulary_records = artifacts.get("vocabularies", [])
    if len(vocabulary_records) != 14:
        raise ContractError("Vocabulary stage does not contain 14 dictionaries")
    for item in vocabulary_records:
        path = Path(str(item["artifact"]["path"]))
        vocabulary = _load_vocabulary(path)
        fold = item.get("heldout_fold")
        fold = None if fold is None else int(fold)
        k = int(item["k"])
        seed = int(item["kmeans_seed"])
        basis_path = _basis_path(output, fold)
        projection_path = _projection_path(output, fold)
        expected = {
            "schema_version": SCHEMA_VERSION,
            "scope": (
                "full_source"
                if fold is None
                else f"outer_training_fold_{fold}"
            ),
            "heldout_fold": fold,
            "n_prototypes": k,
            "kmeans_seed": seed,
            "normalize": "l2",
            "pca": pca64_parameters(seed=VOCAB_SEED),
            "kmeans": lloyd_kmeans_parameters(k, seed=seed),
            "n_sample_tiles": SAMPLE_TILES,
            "pca_basis_sha256": sha256_file(basis_path),
            "projected_coordinates_sha256": sha256_file(projection_path),
            "software": current_software,
            "implementation_sources": current_sources,
        }
        if any(vocabulary.config.get(key) != value for key, value in expected.items()):
            raise ContractError(f"Vocabulary stage contract drifted: {path}")
    _freeze_outer_mappings(output)
    return record


_ASSIGNMENT_STORE: PackedFeatureStore | None = None
_ASSIGNMENT_VOCABULARIES: dict[str, V14Vocabulary] = {}
_ASSIGNMENT_VOCABULARY_DIGESTS: dict[str, str] = {}
_ASSIGNMENT_DEPENDENCIES: dict[str, Any] = {}
_ASSIGNMENT_SHARD_ROOT: Path | None = None


def _assignment_vocabulary_paths(output: Path) -> dict[str, Path]:
    return {
        "reference": _canonical_vocabulary_path(output),
        **{
            f"outer_fold_{fold}": _vocabulary_path(output, fold=fold)
            for fold in FOLDS
        },
    }


def _assignment_dependency_contract(
    output: Path, vocabulary_digests: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pack_sha256": {
            name: record["sha256"] for name, record in PACK_FILES.items()
        },
        "source_roster_sha256": sha256_file(
            output / "inputs/source_roster_label_blind.csv"
        ),
        "implementation_contract_sha256": sha256_file(
            output / "implementation_contract.json"
        ),
        "vocabulary_sha256": dict(vocabulary_digests),
        "implementation_sources": [
            identity(Path(__file__).resolve()),
            identity(REPO / "src/oceanpath/aim1/v14_concepts.py"),
            identity(REPO / "src/oceanpath/datasets/packed.py"),
        ],
    }


def _assignment_shard_receipt_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.receipt.json")


def _assignment_worker_initialize(
    pack_root: str,
    vocabulary_paths: Mapping[str, str],
    vocabulary_digests: Mapping[str, str],
    dependencies: Mapping[str, Any],
    shard_root: str,
) -> None:
    global _ASSIGNMENT_SHARD_ROOT
    global _ASSIGNMENT_STORE
    global _ASSIGNMENT_VOCABULARIES
    global _ASSIGNMENT_VOCABULARY_DIGESTS
    global _ASSIGNMENT_DEPENDENCIES

    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    _ASSIGNMENT_STORE = PackedFeatureStore(Path(pack_root))
    _ASSIGNMENT_VOCABULARIES = {
        key: _load_vocabulary(Path(path)) for key, path in vocabulary_paths.items()
    }
    _ASSIGNMENT_VOCABULARY_DIGESTS = dict(vocabulary_digests)
    _ASSIGNMENT_DEPENDENCIES = dict(dependencies)
    _ASSIGNMENT_SHARD_ROOT = Path(shard_root)


def _assignment_shard_path(shard_root: Path, slide_id: str) -> Path:
    if Path(slide_id).name != slide_id or "/" in slide_id or "\\" in slide_id:
        raise ContractError(f"Slide ID is unsafe as a checkpoint name: {slide_id!r}")
    return shard_root / f"{slide_id}.npz"


def _validate_assignment_shard(
    path: Path,
    *,
    slide_id: str,
    n_tiles: int,
    vocabulary_digests: Mapping[str, str],
    dependencies: Mapping[str, Any],
) -> None:
    receipt_path = _assignment_shard_receipt_path(path)
    if not receipt_path.is_file():
        raise ContractError(f"Assignment checkpoint receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_dependency_digest = hashlib.sha256(json_bytes(dependencies)).hexdigest()
    if (
        receipt.get("status") != "PASS"
        or receipt.get("slide_id") != slide_id
        or int(receipt.get("n_tiles", -1)) != int(n_tiles)
        or receipt.get("dependencies") != dict(dependencies)
        or receipt.get("dependency_contract_sha256") != expected_dependency_digest
    ):
        raise ContractError(f"Assignment checkpoint receipt drifted: {receipt_path}")
    _validate_identity_record(
        receipt.get("artifact", {}), f"assignment checkpoint {slide_id}"
    )
    try:
        with np.load(path, allow_pickle=False) as bundle:
            metadata = _decoded_json_array(bundle["metadata_json"])
            if (
                metadata.get("slide_id") != slide_id
                or int(metadata.get("n_tiles", -1)) != int(n_tiles)
                or metadata.get("vocabulary_sha256") != dict(vocabulary_digests)
                or metadata.get("dependency_contract_sha256")
                != expected_dependency_digest
            ):
                raise ContractError(f"Assignment checkpoint metadata drifted: {path}")
            expected_arrays = {
                "reference_labels",
                "reference_distances",
                *(f"outer_fold_{fold}_labels" for fold in FOLDS),
                "metadata_json",
            }
            if set(bundle.files) != expected_arrays:
                raise ContractError(f"Assignment checkpoint schema drifted: {path}")
            for key in ["reference", *(f"outer_fold_{fold}" for fold in FOLDS)]:
                labels = np.asarray(bundle[f"{key}_labels"])
                if (
                    labels.shape != (n_tiles,)
                    or not np.issubdtype(labels.dtype, np.integer)
                    or (labels < 0).any()
                    or (labels >= CANONICAL_K).any()
                ):
                    raise ContractError(f"Invalid {key} labels in {path}")
            distances = np.asarray(bundle["reference_distances"])
            if distances.shape != (n_tiles,) or not np.isfinite(distances).all():
                raise ContractError(f"Invalid reference distances in {path}")
            if (distances < 0).any():
                raise ContractError(f"Negative reference distances in {path}")
    except (KeyError, OSError, ValueError) as exc:
        raise ContractError(f"Cannot replay assignment checkpoint {path}: {exc}") from exc


def _assign_one_slide(slide_id: str) -> dict[str, Any]:
    from threadpoolctl import threadpool_limits

    if _ASSIGNMENT_STORE is None or _ASSIGNMENT_SHARD_ROOT is None:
        raise RuntimeError("Assignment worker was not initialized")
    store = _ASSIGNMENT_STORE
    n_tiles = store.length_of(slide_id)
    destination = _assignment_shard_path(_ASSIGNMENT_SHARD_ROOT, slide_id)
    receipt_path = _assignment_shard_receipt_path(destination)
    if receipt_path.is_file() and not destination.is_file():
        raise ContractError(
            f"Assignment receipt exists without its checkpoint: {receipt_path}"
        )
    if destination.is_file() and receipt_path.is_file():
        _validate_assignment_shard(
            destination,
            slide_id=slide_id,
            n_tiles=n_tiles,
            vocabulary_digests=_ASSIGNMENT_VOCABULARY_DIGESTS,
            dependencies=_ASSIGNMENT_DEPENDENCIES,
        )
        return {
            "slide_id": slide_id,
            "n_tiles": n_tiles,
            "elapsed_seconds": 0.0,
            "original_elapsed_seconds": float(
                json.loads(receipt_path.read_text(encoding="utf-8"))[
                    "elapsed_seconds"
                ]
            ),
            "action": "replayed",
            "path": str(destination.resolve()),
            "receipt_path": str(receipt_path.resolve()),
            "peak_rss_kib": int(
                json.loads(receipt_path.read_text(encoding="utf-8"))["peak_rss_kib"]
            ),
        }

    started = time.monotonic()
    features = store.read_features(store.position(slide_id))
    arrays: dict[str, np.ndarray] = {}
    with threadpool_limits(limits=1):
        for key, vocabulary in _ASSIGNMENT_VOCABULARIES.items():
            labels, distances = vocabulary.assign_with_distances(features)
            arrays[f"{key}_labels"] = np.asarray(labels, dtype=np.int16)
            if key == "reference":
                arrays["reference_distances"] = np.asarray(distances, dtype=np.float32)
    del features
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "slide_id": slide_id,
        "n_tiles": n_tiles,
        "vocabulary_sha256": _ASSIGNMENT_VOCABULARY_DIGESTS,
        "dependency_contract_sha256": hashlib.sha256(
            json_bytes(_ASSIGNMENT_DEPENDENCIES)
        ).hexdigest(),
        "tile_order": "zero-based packed-store slide-local row",
    }
    arrays["metadata_json"] = _encoded_json_array(metadata)
    if destination.is_file():
        # The only valid orphan window is a fully published NPZ followed by a
        # crash before its receipt.  Recompute and compare arrays, since NPZ
        # container timestamps need not be byte-reproducible.
        try:
            with np.load(destination, allow_pickle=False) as orphan:
                if set(orphan.files) != set(arrays) or any(
                    not np.array_equal(np.asarray(orphan[key]), value)
                    for key, value in arrays.items()
                ):
                    raise ContractError(
                        f"Unreceipted assignment checkpoint differs from replay: {destination}"
                    )
        except (KeyError, OSError, ValueError) as exc:
            raise ContractError(
                f"Cannot recover unreceipted assignment checkpoint {destination}: {exc}"
            ) from exc
    else:
        atomic_write(destination, _npz_bytes(**arrays))
    elapsed = time.monotonic() - started
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": "PASS",
        "created_utc": utc_now(),
        "slide_id": slide_id,
        "n_tiles": n_tiles,
        "dependencies": _ASSIGNMENT_DEPENDENCIES,
        "dependency_contract_sha256": metadata["dependency_contract_sha256"],
        "elapsed_seconds": elapsed,
        "tiles_per_second": n_tiles / max(elapsed, 1e-12),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "artifact": identity(destination),
    }
    write_once(receipt_path, json_bytes(receipt))
    _validate_assignment_shard(
        destination,
        slide_id=slide_id,
        n_tiles=n_tiles,
        vocabulary_digests=_ASSIGNMENT_VOCABULARY_DIGESTS,
        dependencies=_ASSIGNMENT_DEPENDENCIES,
    )
    return {
        "slide_id": slide_id,
        "n_tiles": n_tiles,
        "elapsed_seconds": elapsed,
        "action": "computed",
        "path": str(destination.resolve()),
        "receipt_path": str(receipt_path.resolve()),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _profile_frame(
    roster: pd.DataFrame,
    profiles: np.ndarray,
    *,
    vocabulary_id: str,
) -> pd.DataFrame:
    columns = [f"prototype_{prototype:02d}" for prototype in range(profiles.shape[1])]
    result = roster[["slide_id", "patient_id", "subcohort", "fold", "n_tiles"]].copy()
    result.insert(0, "vocabulary_id", vocabulary_id)
    for index, column in enumerate(columns):
        result[column] = profiles[:, index]
    return result


def _freeze_first_assignment_benchmark(
    output: Path, result: Mapping[str, Any]
) -> None:
    path = output / "benchmarks/assignment_first_completed_slide.json"
    if path.is_file():
        return
    elapsed = float(
        result.get("elapsed_seconds")
        or result.get("original_elapsed_seconds")
        or 0.0
    )
    if elapsed <= 0:
        return
    n_tiles = int(result["n_tiles"])
    rate = n_tiles / elapsed
    write_once(
        path,
        json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "component": COMPONENT,
                "status": "PASS",
                "created_utc": utc_now(),
                "slide_id": str(result["slide_id"]),
                "n_tiles": n_tiles,
                "elapsed_seconds": elapsed,
                "tiles_per_second": rate,
                "projected_serial_seconds_for_all_source_slides": SOURCE_TILES
                / rate,
                "peak_rss_kib": int(result["peak_rss_kib"]),
                "assignment_receipt": identity(
                    Path(str(result["receipt_path"]))
                ),
            }
        ),
    )


def _patient_profile_frame(
    slide_frame: pd.DataFrame,
    *,
    vocabulary_id: str,
) -> pd.DataFrame:
    prototype_columns = [column for column in slide_frame if column.startswith("prototype_")]
    patients = equal_slide_patient_profiles(
        slide_frame[prototype_columns].to_numpy(dtype=np.float64),
        slide_frame["patient_id"].astype(str).tolist(),
    )
    metadata = (
        slide_frame.sort_values(["patient_id", "slide_id"], kind="mergesort")
        .drop_duplicates("patient_id")
        .set_index("patient_id")
        .loc[patients.patient_ids]
    )
    result = pd.DataFrame(patients.profiles, columns=prototype_columns)
    result.insert(0, "n_slides", patients.n_slides)
    result.insert(0, "fold", metadata["fold"].to_numpy(dtype=np.int64))
    result.insert(0, "subcohort", metadata["subcohort"].astype(str).to_numpy())
    result.insert(0, "patient_id", patients.patient_ids)
    result.insert(0, "vocabulary_id", vocabulary_id)
    return result


def _aggregate_assignment_shards(
    output: Path,
    roster: pd.DataFrame,
    vocabulary_digests: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    vocabulary_keys = ["reference", *(f"outer_fold_{fold}" for fold in FOLDS)]
    profile_rows = {
        key: np.empty((len(roster), CANONICAL_K), dtype=np.float64)
        for key in vocabulary_keys
    }
    all_reference_labels: list[np.ndarray] = []
    all_reference_distances: list[np.ndarray] = []
    candidate_rows: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    shard_root = output / "assignments/source"
    store = PackedFeatureStore(PACK_ROOT)
    dependencies = _assignment_dependency_contract(output, vocabulary_digests)

    for row_index, row in enumerate(roster.itertuples(index=False)):
        slide_id = str(row.slide_id)
        n_tiles = int(row.n_tiles)
        path = _assignment_shard_path(shard_root, slide_id)
        _validate_assignment_shard(
            path,
            slide_id=slide_id,
            n_tiles=n_tiles,
            vocabulary_digests=vocabulary_digests,
            dependencies=dependencies,
        )
        shard_records.append(
            {
                "slide_id": slide_id,
                **identity(path),
                "receipt_path": str(_assignment_shard_receipt_path(path).resolve()),
                "receipt_size_bytes": _assignment_shard_receipt_path(path).stat().st_size,
                "receipt_sha256": sha256_file(_assignment_shard_receipt_path(path)),
            }
        )
        with np.load(path, allow_pickle=False) as bundle:
            for key in vocabulary_keys:
                labels = np.asarray(bundle[f"{key}_labels"], dtype=np.int16)
                profile_rows[key][row_index] = slide_abundance(labels, CANONICAL_K)
            labels = np.asarray(bundle["reference_labels"], dtype=np.int16)
            distances = np.asarray(bundle["reference_distances"], dtype=np.float32)
        all_reference_labels.append(labels)
        all_reference_distances.append(distances)
        unique_prototypes = np.unique(labels)
        selected_indices: list[int] = []
        selected_prototypes: list[int] = []
        for prototype in unique_prototypes:
            eligible = np.flatnonzero(labels == prototype)
            selected_indices.append(int(eligible[int(np.argmin(distances[eligible]))]))
            selected_prototypes.append(int(prototype))
        coords = store.read_coords(
            store.position(slide_id),
            rows=np.asarray(selected_indices, dtype=np.int64),
        )
        for prototype, tile_index, coordinate in zip(
            selected_prototypes, selected_indices, coords, strict=True
        ):
            candidate_rows.append(
                {
                    "prototype_id": prototype,
                    "patient_id": str(row.patient_id),
                    "subcohort": str(row.subcohort),
                    "slide_id": slide_id,
                    "tile_id": f"{tile_index:012d}",
                    "tile_index": tile_index,
                    "x": int(coordinate[0]),
                    "y": int(coordinate[1]),
                    "distance": float(distances[tile_index]),
                }
            )

    labels_all = np.concatenate(all_reference_labels)
    distances_all = np.concatenate(all_reference_distances)
    if len(labels_all) != int(roster["n_tiles"].sum()):
        raise ContractError("Canonical all-tile assignment census is incomplete")
    if not np.isfinite(distances_all).all():
        raise ContractError("Canonical all-tile distance census contains nonfinite values")

    candidates = pd.DataFrame.from_records(candidate_rows)
    candidates = candidates.sort_values(
        ["prototype_id", "patient_id", "distance", "slide_id", "tile_id"],
        kind="mergesort",
    ).drop_duplicates(["prototype_id", "patient_id"], keep="first")
    distance_records: list[dict[str, Any]] = []
    support_records: list[dict[str, Any]] = []
    candidate_blocks: list[pd.DataFrame] = []
    for prototype in range(CANONICAL_K):
        values = distances_all[labels_all == prototype]
        if len(values) == 0:
            q10 = q25 = q99 = math.nan
        else:
            q10, q25, q99 = np.quantile(
                values, [0.10, 0.25, 0.99], method="linear"
            ).tolist()
        block = candidates.loc[candidates["prototype_id"].eq(prototype)].copy()
        block["prototype_q10_distance"] = q10
        block["prototype_q25_distance"] = q25
        block["eligibility_tier"] = np.select(
            [block["distance"] <= q10, block["distance"] <= q25],
            [0, 1],
            default=2,
        ).astype(np.int8)
        block["eligibility_stage"] = block["eligibility_tier"].map(
            {0: "decile", 1: "quartile", 2: "all"}
        )
        candidate_blocks.append(block)
        support = int(block["patient_id"].nunique())
        distance_records.append(
            {
                "prototype_id": prototype,
                "n_assigned_tiles": int(len(values)),
                "q10_distance": q10,
                "q25_distance": q25,
                "q99_distance": q99,
            }
        )
        support_records.append(
            {
                "prototype_id": prototype,
                "distinct_patient_support": support,
                "decile_patient_support": int((block["eligibility_tier"] == 0).sum()),
                "quartile_patient_support": int((block["eligibility_tier"] <= 1).sum()),
                "montage_support_status": (
                    "MONTAGE_SUPPORT_SUFFICIENT"
                    if support >= 12
                    else "MONTAGE_SUPPORT_INSUFFICIENT"
                ),
            }
        )
    candidates = pd.concat(candidate_blocks, ignore_index=True)
    distances = pd.DataFrame.from_records(distance_records)
    support = pd.DataFrame.from_records(support_records)

    profile_artifacts: dict[str, Any] = {}
    patient_frames: dict[str, pd.DataFrame] = {}
    for key in vocabulary_keys:
        if not np.allclose(profile_rows[key].sum(axis=1), 1.0, rtol=0, atol=1e-12):
            raise ContractError(f"Slide profile masses do not sum to one: {key}")
        slide_frame = _profile_frame(roster, profile_rows[key], vocabulary_id=key)
        patient_frame = _patient_profile_frame(slide_frame, vocabulary_id=key)
        prototype_columns = [
            column for column in patient_frame if column.startswith("prototype_")
        ]
        if len(patient_frame) != SOURCE_PATIENTS or not np.allclose(
            patient_frame[prototype_columns].sum(axis=1), 1.0, rtol=0, atol=1e-12
        ):
            raise ContractError(f"Patient profile gate failed: {key}")
        directory = output / "profiles" / key
        slide_path = directory / "slide_profiles.parquet"
        patient_path = directory / "patient_profiles.parquet"
        write_once(slide_path, parquet_bytes(slide_frame))
        write_once(patient_path, parquet_bytes(patient_frame))
        patient_frames[key] = patient_frame
        profile_artifacts[key] = {
            "slide_profiles": identity(slide_path),
            "patient_profiles": identity(patient_path),
        }

    oof_blocks = [
        patient_frames[f"outer_fold_{fold}"].loc[
            patient_frames[f"outer_fold_{fold}"]["fold"].eq(fold)
        ]
        for fold in FOLDS
    ]
    oof = pd.concat(oof_blocks, ignore_index=True).sort_values(
        "patient_id", kind="mergesort"
    )
    if len(oof) != SOURCE_PATIENTS or oof["patient_id"].duplicated().any():
        raise ContractError("OOF patient profile is incomplete or duplicated")
    oof_path = output / "profiles/oof_patient_profiles.parquet"
    write_once(oof_path, parquet_bytes(oof))

    shard_manifest_path = output / "assignments/source_manifest.parquet"
    candidates_path = output / "montage/canonical_candidates.parquet"
    distances_path = output / "profiles/reference_distance_quantiles.csv"
    support_path = output / "montage/support_roster.csv"
    write_once(shard_manifest_path, parquet_bytes(pd.DataFrame(shard_records)))
    write_once(candidates_path, parquet_bytes(candidates))
    write_once(distances_path, distances.to_csv(index=False).encode("utf-8"))
    write_once(support_path, support.to_csv(index=False).encode("utf-8"))
    assignment_artifacts = {
        "shard_manifest": identity(shard_manifest_path),
        "canonical_candidates": identity(candidates_path),
        "reference_distance_quantiles": identity(distances_path),
        "montage_support_roster": identity(support_path),
    }
    profile_artifacts["oof_patient_profiles"] = identity(oof_path)
    return assignment_artifacts, profile_artifacts


def assign_profiles(output: Path, *, max_workers: int) -> dict[str, Any]:
    """Assign every source tile under the reference and five outer dictionaries."""

    _require_production_preflight(output)
    vocabulary_receipt = _require_stage_receipt(output, "vocabularies")
    _validate_vocabulary_stage(output, vocabulary_receipt)
    if not 1 <= int(max_workers) <= MAX_PROFILE_WORKERS:
        raise ContractError(f"max_workers must be between 1 and {MAX_PROFILE_WORKERS}")
    receipt_path = output / "receipts/profiles.json"
    replay = _load_json_receipt_if_valid(
        receipt_path,
        status="PASS",
        artifact_keys=("shard_manifest", "oof_patient_profiles"),
    )
    if replay is not None:
        _validate_artifact_tree(replay.get("artifacts", {}), "profile receipt")
        _validate_identity_record(
            replay.get("first_completed_benchmark", {}),
            "assignment first-slide benchmark",
        )
        _validate_assignment_manifest(output)
        _validate_profile_replay(output)
        return replay

    with exclusive_lock(output / "locks/profiles.lock", "source profile assignment"):
        store = PackedFeatureStore(PACK_ROOT)
        roster = _blind_roster_with_counts(output, store)
        vocabulary_paths = _assignment_vocabulary_paths(output)
        vocabulary_digests = {
            key: sha256_file(path) for key, path in vocabulary_paths.items()
        }
        assignment_dependencies = _assignment_dependency_contract(
            output, vocabulary_digests
        )
        shard_root = output / "assignments/source"
        shard_root.mkdir(parents=True, exist_ok=True)
        initializer_arguments = (
            str(PACK_ROOT),
            {key: str(path) for key, path in vocabulary_paths.items()},
            vocabulary_digests,
            assignment_dependencies,
            str(shard_root),
        )
        slides = roster["slide_id"].astype(str).tolist()
        worker_results: list[dict[str, Any]] = []
        if max_workers == 1:
            _assignment_worker_initialize(*initializer_arguments)
            for position, slide_id in enumerate(slides, start=1):
                result = _assign_one_slide(slide_id)
                worker_results.append(result)
                _freeze_first_assignment_benchmark(output, result)
                if position == 1 or position % 25 == 0 or position == len(slides):
                    print(
                        f"assign-profiles {position}/{len(slides)}: {slide_id}",
                        flush=True,
                    )
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_assignment_worker_initialize,
                initargs=initializer_arguments,
            ) as executor:
                futures = {
                    executor.submit(_assign_one_slide, slide_id): slide_id
                    for slide_id in slides
                }
                for position, future in enumerate(
                    concurrent.futures.as_completed(futures), start=1
                ):
                    result = future.result()
                    worker_results.append(result)
                    _freeze_first_assignment_benchmark(output, result)
                    if position == 1 or position % 25 == 0 or position == len(slides):
                        print(
                            f"assign-profiles {position}/{len(slides)}: "
                            f"{result['slide_id']}",
                            flush=True,
                        )
        assignment_artifacts, profile_artifacts = _aggregate_assignment_shards(
            output, roster, vocabulary_digests
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "component": COMPONENT,
            "status": "PASS",
            "created_utc": utc_now(),
            "source_slides": len(slides),
            "source_patients": int(roster["patient_id"].nunique()),
            "source_tiles": int(roster["n_tiles"].sum()),
            "vocabularies_applied": len(vocabulary_paths),
            "max_workers": int(max_workers),
            "artifacts": {**assignment_artifacts, **profile_artifacts},
            "first_completed_benchmark": identity(
                output / "benchmarks/assignment_first_completed_slide.json"
            ),
        }
        write_once(receipt_path, json_bytes(result))
    return result


def _teacher_source_identities() -> list[dict[str, Any]]:
    paths = [
        Path(__file__).resolve(),
        REPO / "src/oceanpath/training/lightning.py",
        REPO / "src/oceanpath/models/__init__.py",
        REPO / "src/oceanpath/models/base.py",
        REPO / "src/oceanpath/models/components.py",
        REPO / "src/oceanpath/models/abmil.py",
        REPO / "src/oceanpath/models/wsi_classifier.py",
        REPO / "src/oceanpath/datasets/datamodule.py",
        REPO / "src/oceanpath/datasets/packed.py",
        REPO / "src/oceanpath/aim1/v14_concepts.py",
        REPO / "src/oceanpath/contracts/__init__.py",
        REPO / "src/oceanpath/contracts/slide_ids.py",
        REPO / "pyproject.toml",
        REPO / "uv.lock",
    ]
    return [identity(path) for path in paths]


def _teacher_environment(torch: Any) -> dict[str, Any]:
    import lightning

    device = torch.cuda.current_device()
    return {
        **_software_versions(),
        "torch": torch.__version__,
        "lightning": lightning.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": int(torch.backends.cudnn.version()),
        "device_index": int(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _freeze_teacher_execution_contract(
    output: Path,
    *,
    num_workers: int,
    environment: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    vocabulary_digests: Mapping[str, str],
    assignment_dependencies: Mapping[str, Any],
) -> dict[str, Any]:
    path = output / "teachers/execution_contract.json"
    core = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": "FROZEN_BEFORE_HEAD_1",
        "inference_contract": {
            "device": "cuda",
            "autocast": "bfloat16",
            "batch_size": 1,
            "full_bags": True,
            "instance_cap": None,
            "augmentation": False,
            "model_mode": "eval",
            "float32_matmul_precision": "high",
            "num_loader_workers": int(num_workers),
        },
        "environment": dict(environment),
        "implementation_sources": _teacher_source_identities(),
        "pack_dependencies": preflight_receipt["pack"],
        "checkpoint_dependencies": preflight_receipt["teachers"],
        "v13_patient_logits": preflight_receipt["v13_patient_logits"],
        "profile_receipt": identity(output / "receipts/profiles.json"),
        "assignment_manifest": identity(
            output / "assignments/source_manifest.parquet"
        ),
        "assignment_vocabulary_sha256": dict(vocabulary_digests),
        "assignment_dependencies": dict(assignment_dependencies),
    }
    if path.is_file():
        contract = json.loads(path.read_text(encoding="utf-8"))
        if {key: value for key, value in contract.items() if key != "created_utc"} != core:
            raise ContractError("Teacher execution contract changed across restart")
    else:
        contract = {**core, "created_utc": utc_now()}
        write_once(path, json_bytes(contract))
    return identity(path)


def _validate_teacher_module(module: Any, checkpoint: Path) -> dict[str, Any]:
    hparams = dict(module.hparams)
    required = {
        "arch": "abmil",
        "in_dim": 1024,
        "num_classes": 1,
        "loss_type": "bce",
        "compile_model": False,
    }
    for key, expected in required.items():
        if hparams.get(key) != expected:
            raise ContractError(
                f"Teacher {checkpoint} has {key}={hparams.get(key)!r}, expected {expected!r}"
            )
    model_cfg = dict(hparams.get("model_cfg") or {})
    expected_model_cfg = {
        "embed_dim": 512,
        "num_fc_layers": 1,
        "attn_dim": 384,
        "gate": True,
        "dropout": 0.25,
        "input_dropout": 0.10,
        "gradient_checkpointing": False,
    }
    for key, expected in expected_model_cfg.items():
        if model_cfg.get(key) != expected:
            raise ContractError(
                f"Teacher {checkpoint} has model_cfg.{key}={model_cfg.get(key)!r}, "
                f"expected {expected!r}"
            )
    parameter_count = sum(parameter.numel() for parameter in module.model.parameters())
    if parameter_count != 919_682:
        raise ContractError(
            f"Teacher {checkpoint} has {parameter_count:,} parameters, expected 919,682"
        )
    expected_state_keys = {
        "model.aggregator.patch_embed.0.weight",
        "model.aggregator.patch_embed.0.bias",
        "model.aggregator.global_attn.attention_a.0.weight",
        "model.aggregator.global_attn.attention_a.0.bias",
        "model.aggregator.global_attn.attention_b.0.weight",
        "model.aggregator.global_attn.attention_b.0.bias",
        "model.aggregator.global_attn.attention_c.weight",
        "model.aggregator.global_attn.attention_c.bias",
        "model.head.weight",
        "model.head.bias",
    }
    model_keys = {key for key in module.state_dict() if key.startswith("model.")}
    if model_keys != expected_state_keys:
        raise ContractError(f"Teacher state schema drifted for {checkpoint}: {model_keys}")
    if module.training or any(
        child.training for child in module.modules() if child.__class__.__name__ == "Dropout"
    ):
        raise ContractError(f"Teacher did not enter evaluation mode: {checkpoint}")
    return {
        "hparams": {
            "arch": hparams["arch"],
            "in_dim": int(hparams["in_dim"]),
            "num_classes": int(hparams["num_classes"]),
            "loss_type": hparams.get("loss_type"),
            "model_cfg": model_cfg,
            "compile_model": hparams.get("compile_model"),
        },
        "parameter_count": parameter_count,
        "state_keys": sorted(model_keys),
    }


def _sealed_v13_logits() -> pd.DataFrame:
    columns = [
        "arm",
        "patient_id",
        "k_fold",
        *(f"logit_seed{seed}" for seed in MODEL_SEEDS),
        "mean_logit_5seed",
    ]
    frame = pd.read_parquet(V13_PATIENT_LOGITS, columns=columns)
    frame = frame.loc[frame["arm"].eq("tcga_surgen_primary")].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["k_fold"] = pd.to_numeric(frame["k_fold"], errors="raise").astype(int)
    if (
        len(frame) != SOURCE_PATIENTS
        or frame["patient_id"].duplicated().any()
        or not np.isfinite(
            frame[[*(f"logit_seed{seed}" for seed in MODEL_SEEDS), "mean_logit_5seed"]]
            .to_numpy(dtype=np.float64)
        ).all()
    ):
        raise ContractError("Sealed FINAL-v13 patient logits have an invalid source census")
    return frame.sort_values("patient_id", kind="mergesort").reset_index(drop=True)


def _head_output_directory(output: Path, seed: int, fold: int) -> Path:
    return output / f"teachers/seed{seed}/fold_{fold}"


def _load_valid_head_receipt(
    output: Path,
    *,
    seed: int,
    fold: int,
    checkpoint_record: Mapping[str, Any],
    execution_contract_record: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = _head_output_directory(output, seed, fold) / "receipt.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "PASS":
        raise ContractError(f"Teacher head has an immutable non-PASS receipt: {path}")
    if record.get("seed") != seed or record.get("fold") != fold:
        raise ContractError(f"Teacher head receipt identity drifted: {path}")
    if record.get("checkpoint", {}).get("sha256") != checkpoint_record.get("sha256"):
        raise ContractError(f"Teacher head checkpoint dependency drifted: {path}")
    if record.get("execution_contract") != dict(execution_contract_record):
        raise ContractError(f"Teacher head execution contract drifted: {path}")
    if record.get("v13_patient_logits") != identity(V13_PATIENT_LOGITS):
        raise ContractError(f"Teacher head FINAL-v13 logit dependency drifted: {path}")
    if record.get("assignment_manifest") != identity(
        output / "assignments/source_manifest.parquet"
    ):
        raise ContractError(f"Teacher head assignment manifest drifted: {path}")
    current_vocabulary_digests = {
        key: sha256_file(value)
        for key, value in _assignment_vocabulary_paths(output).items()
    }
    if record.get("assignment_vocabulary_sha256") != current_vocabulary_digests:
        raise ContractError(f"Teacher head assignment vocabulary drifted: {path}")
    for key in (
        "slide_logits",
        "patient_logits",
        "heldout_attention_slides",
        "heldout_attention_patients",
    ):
        _validate_identity_record(record["artifacts"][key], f"teacher {seed}/{fold} {key}")
    return record


def _attention_patient_frame(
    slide_attention: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    prototype_columns = [f"prototype_{index:02d}" for index in range(CANONICAL_K)]
    patient_profiles = equal_slide_patient_profiles(
        slide_attention[prototype_columns].to_numpy(dtype=np.float64),
        slide_attention["patient_id"].astype(str).tolist(),
    )
    metadata = (
        roster.loc[roster["fold"].eq(fold)]
        .sort_values(["patient_id", "slide_id"], kind="mergesort")
        .drop_duplicates("patient_id")
        .set_index("patient_id")
        .loc[patient_profiles.patient_ids]
    )
    result = pd.DataFrame(patient_profiles.profiles, columns=prototype_columns)
    result.insert(0, "n_slides", patient_profiles.n_slides)
    result.insert(0, "fold", fold)
    result.insert(0, "seed", seed)
    result.insert(0, "subcohort", metadata["subcohort"].astype(str).to_numpy())
    result.insert(0, "patient_id", patient_profiles.patient_ids)
    if not np.allclose(
        result[prototype_columns].sum(axis=1), 1.0, rtol=0, atol=2e-6
    ):
        raise ContractError(f"Heldout attention patient masses fail for seed{seed}/fold{fold}")
    return result


def _score_one_teacher_head(
    output: Path,
    *,
    seed: int,
    fold: int,
    checkpoint_record: Mapping[str, Any],
    execution_contract_record: Mapping[str, Any],
    loader: Any,
    roster: pd.DataFrame,
    sealed_logits: pd.DataFrame,
    heldout_label_cache: dict[str, np.ndarray],
    assignment_vocabulary_digests: Mapping[str, str],
    assignment_dependencies: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from oceanpath.training.lightning import MILTrainModule

    existing = _load_valid_head_receipt(
        output,
        seed=seed,
        fold=fold,
        checkpoint_record=checkpoint_record,
        execution_contract_record=execution_contract_record,
    )
    if existing is not None:
        return existing
    checkpoint = Path(str(checkpoint_record["path"]))
    current_checkpoint = identity(checkpoint)
    if any(
        current_checkpoint.get(key) != checkpoint_record.get(key)
        for key in ("sha256", "size_bytes", "mtime_ns")
    ):
        raise ContractError(f"Teacher checkpoint changed after preflight: {checkpoint}")
    try:
        module = MILTrainModule.load_from_checkpoint(
            str(checkpoint), map_location="cuda", weights_only=False
        )
    except TypeError:
        module = MILTrainModule.load_from_checkpoint(str(checkpoint), map_location="cuda")
    module.eval().to("cuda")
    schema = _validate_teacher_module(module, checkpoint)

    fold_by_slide = roster.set_index("slide_id")["fold"].astype(int).to_dict()
    patient_by_slide = roster.set_index("slide_id")["patient_id"].astype(str).to_dict()
    subcohort_by_slide = roster.set_index("slide_id")["subcohort"].astype(str).to_dict()
    slide_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        for position, batch in enumerate(loader, start=1):
            slide_id = str(batch["slide_ids"][0])
            is_heldout = int(fold_by_slide[slide_id]) == fold
            features = batch["features"].to("cuda", non_blocking=True)
            mask = (
                batch["mask"].to("cuda", non_blocking=True)
                if batch.get("mask") is not None
                else None
            )
            if mask is not None:
                raise ContractError("Batch-size-one full-bag teacher inference produced a mask")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                model_output = module.model(
                    features,
                    mask=None,
                    return_attention=is_heldout,
                )
            values = model_output.logits.detach().float().cpu().numpy().reshape(-1)
            if values.size != 1 or not np.isfinite(values[0]):
                raise ContractError(f"Teacher emitted an invalid logit for {slide_id}")
            slide_rows.append(
                {
                    "slide_id": slide_id,
                    "patient_id": patient_by_slide[slide_id],
                    "subcohort": subcohort_by_slide[slide_id],
                    "patient_fold": fold_by_slide[slide_id],
                    "seed": seed,
                    "head_fold": fold,
                    "logit": float(values[0]),
                }
            )
            if is_heldout:
                raw_attention = model_output.extras.get("attention_weights")
                if raw_attention is None:
                    raise ContractError(f"Teacher returned no attention for {slide_id}")
                weights = (
                    torch.softmax(raw_attention.float(), dim=-1)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                if slide_id not in heldout_label_cache:
                    shard = _assignment_shard_path(
                        output / "assignments/source", slide_id
                    )
                    _validate_assignment_shard(
                        shard,
                        slide_id=slide_id,
                        n_tiles=len(weights),
                        vocabulary_digests=assignment_vocabulary_digests,
                        dependencies=assignment_dependencies,
                    )
                    with np.load(shard, allow_pickle=False) as bundle:
                        heldout_label_cache[slide_id] = np.asarray(
                            bundle[f"outer_fold_{fold}_labels"], dtype=np.int16
                        )
                labels = heldout_label_cache[slide_id]
                if labels.shape != weights.shape:
                    raise ContractError(
                        f"Attention/assignment tile-order mismatch for {slide_id}"
                    )
                masses = np.bincount(
                    labels.astype(np.int64),
                    weights=weights.astype(np.float64),
                    minlength=CANONICAL_K,
                )
                if not np.isclose(masses.sum(), 1.0, rtol=0, atol=2e-6):
                    raise ContractError(f"Attention masses do not sum to one: {slide_id}")
                row = {
                    "slide_id": slide_id,
                    "patient_id": patient_by_slide[slide_id],
                    "subcohort": subcohort_by_slide[slide_id],
                    "seed": seed,
                    "fold": fold,
                    "n_tiles": len(labels),
                }
                row.update(
                    {
                        f"prototype_{prototype:02d}": float(masses[prototype])
                        for prototype in range(CANONICAL_K)
                    }
                )
                attention_rows.append(row)
            if position % 100 == 0 or position == SOURCE_SLIDES:
                print(
                    f"score-teachers seed{seed}/fold{fold}: "
                    f"{position}/{SOURCE_SLIDES}",
                    flush=True,
                )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started

    slide_logits = pd.DataFrame.from_records(slide_rows).sort_values(
        "slide_id", kind="mergesort"
    )
    if len(slide_logits) != SOURCE_SLIDES or slide_logits["slide_id"].duplicated().any():
        raise ContractError(f"Teacher slide-logit census failed for seed{seed}/fold{fold}")
    patient_logits = (
        slide_logits.groupby("patient_id", sort=True, as_index=False)["logit"].mean()
    )
    patient_metadata = (
        roster.sort_values(["patient_id", "slide_id"], kind="mergesort")
        .drop_duplicates("patient_id")
        [["patient_id", "subcohort", "fold"]]
    )
    patient_logits = patient_metadata.merge(
        patient_logits, on="patient_id", validate="one_to_one"
    )
    patient_logits.insert(3, "seed", seed)
    patient_logits.insert(4, "head_fold", fold)
    heldout_recomputed = patient_logits.loc[
        patient_logits["fold"].eq(fold), ["patient_id", "logit"]
    ]
    heldout_expected = sealed_logits.loc[
        sealed_logits["k_fold"].eq(fold), ["patient_id", f"logit_seed{seed}"]
    ]
    comparison = heldout_expected.merge(
        heldout_recomputed, on="patient_id", how="outer", validate="one_to_one", indicator=True
    )
    expected_count = (248, 247, 248, 248, 248)[fold]
    if len(comparison) != expected_count or not comparison["_merge"].eq("both").all():
        raise ContractError(f"Teacher heldout patient roster drifted for seed{seed}/fold{fold}")
    deltas = np.abs(
        comparison[f"logit_seed{seed}"].to_numpy(dtype=np.float64)
        - comparison["logit"].to_numpy(dtype=np.float64)
    )
    max_delta = float(deltas.max(initial=0.0))

    attention_slides = pd.DataFrame.from_records(attention_rows).sort_values(
        "slide_id", kind="mergesort"
    )
    if len(attention_slides) != int(roster["fold"].eq(fold).sum()):
        raise ContractError(f"Teacher attention slide census failed for seed{seed}/fold{fold}")
    attention_patients = _attention_patient_frame(
        attention_slides, roster, seed=seed, fold=fold
    )
    if len(attention_patients) != expected_count:
        raise ContractError(f"Teacher attention patient census failed for seed{seed}/fold{fold}")

    directory = _head_output_directory(output, seed, fold)
    slide_path = directory / "slide_logits.parquet"
    patient_path = directory / "patient_logits.parquet"
    attention_slide_path = directory / "heldout_attention_slides.parquet"
    attention_patient_path = directory / "heldout_attention_patients.parquet"
    write_once(slide_path, parquet_bytes(slide_logits))
    write_once(patient_path, parquet_bytes(patient_logits))
    write_once(attention_slide_path, parquet_bytes(attention_slides))
    write_once(attention_patient_path, parquet_bytes(attention_patients))
    result = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": "PASS" if max_delta <= 1e-6 else "BLOCKED_V13_REPRODUCTION",
        "created_utc": utc_now(),
        "seed": seed,
        "fold": fold,
        "checkpoint": current_checkpoint,
        "execution_contract": dict(execution_contract_record),
        "v13_patient_logits": identity(V13_PATIENT_LOGITS),
        "assignment_manifest": identity(
            output / "assignments/source_manifest.parquet"
        ),
        "assignment_vocabulary_sha256": dict(assignment_vocabulary_digests),
        "model_schema": schema,
        "heldout_patients": expected_count,
        "maximum_absolute_v13_logit_delta": max_delta,
        "tolerance": 1e-6,
        "elapsed_seconds": elapsed,
        "slides_per_second": SOURCE_SLIDES / max(elapsed, 1e-12),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "artifacts": {
            "slide_logits": identity(slide_path),
            "patient_logits": identity(patient_path),
            "heldout_attention_slides": identity(attention_slide_path),
            "heldout_attention_patients": identity(attention_patient_path),
        },
    }
    write_once(directory / "receipt.json", json_bytes(result))
    del module
    torch.cuda.empty_cache()
    if result["status"] != "PASS":
        raise ContractError(
            f"Teacher seed{seed}/fold{fold} failed FINAL-v13 replay: {max_delta:.9g}"
        )
    return result


def _combine_teacher_outputs(
    output: Path,
    roster: pd.DataFrame,
    sealed_logits: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    prototype_columns = [f"prototype_{index:02d}" for index in range(CANONICAL_K)]
    seed_logit_frames: list[pd.DataFrame] = []
    seed_attention_frames: list[pd.DataFrame] = []
    artifacts: dict[str, dict[str, Any]] = {}
    for seed in MODEL_SEEDS:
        logit_blocks: list[pd.DataFrame] = []
        attention_blocks: list[pd.DataFrame] = []
        for fold in FOLDS:
            directory = _head_output_directory(output, seed, fold)
            patient_logits = pd.read_parquet(directory / "patient_logits.parquet")
            logit_blocks.append(
                patient_logits.loc[
                    patient_logits["fold"].eq(fold), ["patient_id", "fold", "logit"]
                ]
            )
            attention_blocks.append(
                pd.read_parquet(directory / "heldout_attention_patients.parquet")
            )
        seed_logits = pd.concat(logit_blocks, ignore_index=True).sort_values(
            "patient_id", kind="mergesort"
        )
        seed_attention = pd.concat(attention_blocks, ignore_index=True).sort_values(
            "patient_id", kind="mergesort"
        )
        if (
            len(seed_logits) != SOURCE_PATIENTS
            or seed_logits["patient_id"].duplicated().any()
            or len(seed_attention) != SOURCE_PATIENTS
            or seed_attention["patient_id"].duplicated().any()
        ):
            raise ContractError(f"Incomplete OOF teacher output for seed {seed}")
        seed_logits = seed_logits.rename(columns={"logit": f"logit_seed{seed}"})
        seed_logit_frames.append(seed_logits.drop(columns="fold"))
        seed_attention_frames.append(seed_attention)
        attention_path = output / f"teachers/seed{seed}_oof_attention_patients.parquet"
        write_once(attention_path, parquet_bytes(seed_attention))
        artifacts[f"seed{seed}_oof_attention"] = identity(attention_path)

    native = seed_logit_frames[0]
    for frame in seed_logit_frames[1:]:
        native = native.merge(frame, on="patient_id", validate="one_to_one")
    native["mean_logit_5seed"] = native[
        [f"logit_seed{seed}" for seed in MODEL_SEEDS]
    ].mean(axis=1)
    comparison = sealed_logits.merge(
        native, on="patient_id", validate="one_to_one", suffixes=("_sealed", "_recomputed")
    )
    mean_delta = np.abs(
        comparison["mean_logit_5seed_sealed"].to_numpy(dtype=np.float64)
        - comparison["mean_logit_5seed_recomputed"].to_numpy(dtype=np.float64)
    )
    maximum_mean_delta = float(mean_delta.max(initial=0.0))
    if maximum_mean_delta > 1e-6:
        raise ContractError(
            "Five-seed recomputed mean logits fail FINAL-v13 replay: "
            f"{maximum_mean_delta:.9g}"
        )
    native_path = output / "teachers/oof_native_logits_recomputed.parquet"
    write_once(native_path, parquet_bytes(native))
    artifacts["oof_native_logits"] = identity(native_path)

    stacked_attention = pd.concat(seed_attention_frames, ignore_index=True)
    counts = stacked_attention.groupby("patient_id").size()
    if not counts.eq(len(MODEL_SEEDS)).all() or len(counts) != SOURCE_PATIENTS:
        raise ContractError("Attention mean would use a partial seed set")
    attention_mean = (
        stacked_attention.groupby("patient_id", sort=True, as_index=False)[prototype_columns]
        .mean()
    )
    metadata = (
        roster.sort_values(["patient_id", "slide_id"], kind="mergesort")
        .drop_duplicates("patient_id")
        [["patient_id", "subcohort", "fold"]]
    )
    attention_mean = metadata.merge(
        attention_mean, on="patient_id", validate="one_to_one"
    )
    if not np.allclose(
        attention_mean[prototype_columns].sum(axis=1), 1.0, rtol=0, atol=2e-6
    ):
        raise ContractError("Five-seed attention means do not sum to one")
    attention_mean_path = output / "teachers/oof_attention_patients_5seed_mean.parquet"
    write_once(attention_mean_path, parquet_bytes(attention_mean))
    artifacts["oof_attention_5seed_mean"] = identity(attention_mean_path)
    return artifacts, {"maximum_absolute_mean_logit_delta": maximum_mean_delta}


def _validate_teacher_receipt(receipt: Mapping[str, Any], output: Path) -> None:
    """Replay the immutable 25-head and combined-output evidence."""

    if receipt.get("status") != "PASS" or receipt.get("head_census") != 25:
        raise ContractError("Teacher receipt does not bind a complete 25-head PASS")
    _validate_artifact_tree(receipt.get("artifacts", {}), "teacher combined artifacts")
    _validate_artifact_tree(
        receipt.get("checkpoint_dependencies", {}), "teacher checkpoint dependencies"
    )
    execution_record = receipt.get("execution_contract", {})
    _validate_identity_record(execution_record, "teacher execution contract")
    _validate_identity_record(
        receipt.get("first_completed_benchmark", {}),
        "teacher first-head benchmark",
    )
    execution = json.loads(
        Path(str(execution_record["path"])).read_text(encoding="utf-8")
    )
    if (
        execution.get("status") != "FROZEN_BEFORE_HEAD_1"
        or execution.get("implementation_sources") != _teacher_source_identities()
        or execution.get("profile_receipt")
        != identity(output / "receipts/profiles.json")
        or execution.get("assignment_manifest")
        != identity(output / "assignments/source_manifest.parquet")
        or execution.get("v13_patient_logits") != identity(V13_PATIENT_LOGITS)
    ):
        raise ContractError("Teacher execution contract no longer replays")
    if (
        receipt.get("v13_patient_logits") != execution["v13_patient_logits"]
        or receipt.get("profile_receipt") != execution["profile_receipt"]
        or receipt.get("assignment_manifest") != execution["assignment_manifest"]
    ):
        raise ContractError("Teacher aggregate dependency binding drifted")
    current_vocabulary_digests = {
        key: sha256_file(path)
        for key, path in _assignment_vocabulary_paths(output).items()
    }
    if (
        receipt.get("assignment_vocabulary_sha256") != current_vocabulary_digests
        or execution.get("assignment_vocabulary_sha256")
        != current_vocabulary_digests
    ):
        raise ContractError("Teacher assignment vocabulary dependency drifted")
    head_receipts = receipt.get("head_receipts", [])
    benchmarks = receipt.get("head_benchmarks", [])
    if len(head_receipts) != 25 or len(benchmarks) != 25:
        raise ContractError("Teacher receipt does not contain exactly 25 heads")
    expected_grid = {(seed, fold) for seed in MODEL_SEEDS for fold in FOLDS}
    observed_grid = {
        (int(row.get("seed", -1)), int(row.get("fold", -1))) for row in benchmarks
    }
    if observed_grid != expected_grid:
        raise ContractError("Teacher benchmark grid is not the governed 5x5 grid")
    if any(
        float(row.get("maximum_absolute_v13_logit_delta", math.inf)) > 1e-6
        for row in benchmarks
    ):
        raise ContractError("A teacher benchmark fails FINAL-v13 reproduction")
    observed_heads: set[tuple[int, int]] = set()
    checkpoint_lookup = {
        (int(item["seed"]), int(item["fold"])): item["checkpoint"]
        for item in execution["checkpoint_dependencies"]["heads"]
    }
    if set(checkpoint_lookup) != expected_grid:
        raise ContractError("Teacher execution contract checkpoint grid drifted")
    for index, record in enumerate(head_receipts):
        _validate_identity_record(record, f"teacher head receipt {index}")
        head = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        if head.get("status") != "PASS":
            raise ContractError(f"Teacher head receipt {index} is not a PASS")
        if head.get("execution_contract") != execution_record:
            raise ContractError(f"Teacher head {index} uses another execution contract")
        if (
            head.get("v13_patient_logits") != execution["v13_patient_logits"]
            or head.get("assignment_manifest") != execution["assignment_manifest"]
            or head.get("assignment_vocabulary_sha256")
            != execution["assignment_vocabulary_sha256"]
        ):
            raise ContractError(f"Teacher head {index} dependency binding drifted")
        head_key = (int(head.get("seed", -1)), int(head.get("fold", -1)))
        if head_key not in expected_grid or head_key in observed_heads:
            raise ContractError(f"Teacher head receipt {index} has an invalid seed/fold")
        if head.get("checkpoint") != checkpoint_lookup[head_key]:
            raise ContractError(f"Teacher head {index} checkpoint binding drifted")
        observed_heads.add(head_key)
        _validate_artifact_tree(head.get("artifacts", {}), f"teacher head {index}")
    if observed_heads != expected_grid:
        raise ContractError("Teacher head receipts are not the governed 5x5 grid")


def score_teachers(output: Path, *, num_workers: int = 4) -> dict[str, Any]:
    """Replay all 25 FINAL-v13 heads and emit native logits plus heldout attention."""

    preflight_receipt = _require_production_preflight(output)
    profile_receipt = _require_stage_receipt(output, "profiles")
    _validate_artifact_tree(
        profile_receipt.get("artifacts", {}), "teacher profile inputs"
    )
    _validate_assignment_manifest(output)
    _validate_profile_replay(output)
    if not 0 <= int(num_workers) <= 4:
        raise ContractError("Teacher DataLoader workers must be between zero and four")
    receipt_path = output / "receipts/teachers.json"
    replay = _load_json_receipt_if_valid(
        receipt_path,
        status="PASS",
        artifact_keys=("oof_native_logits", "oof_attention_5seed_mean"),
    )
    if replay is not None:
        _validate_teacher_receipt(replay, output)
        return replay

    import torch
    from torch.utils.data import DataLoader

    from oceanpath.datasets.datamodule import SimpleMILCollator, SlideDataset

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ContractError("Governed teacher replay requires CUDA with bfloat16 support")
    torch.set_float32_matmul_precision("high")
    if torch.get_float32_matmul_precision() != "high":
        raise ContractError("Could not establish FINAL-v13 high matmul precision contract")

    with exclusive_lock(output / "locks/teachers.lock", "frozen-head GPU scoring"):
        store = PackedFeatureStore(PACK_ROOT)
        roster = _blind_roster_with_counts(output, store)
        slides = roster["slide_id"].astype(str).tolist()
        dataset = SlideDataset(
            feature_dir=str(FEATURE_ROOT),
            slide_ids=slides,
            labels=dict.fromkeys(slides, 0),
            max_instances=None,
            is_train=False,
            force_float32=True,
            store=store,
        )
        if dataset.slide_ids != slides:
            raise ContractError("Teacher dataset changed the sealed source slide order")
        loader_arguments: dict[str, Any] = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": int(num_workers),
            "collate_fn": SimpleMILCollator(max_instances=None),
            "pin_memory": True,
        }
        if num_workers > 0:
            loader_arguments.update({"prefetch_factor": 2, "persistent_workers": True})
        loader = DataLoader(dataset, **loader_arguments)
        sealed_logits = _sealed_v13_logits()
        vocabulary_paths = _assignment_vocabulary_paths(output)
        vocabulary_digests = {
            key: sha256_file(path) for key, path in vocabulary_paths.items()
        }
        assignment_dependencies = _assignment_dependency_contract(
            output, vocabulary_digests
        )
        teacher_environment = _teacher_environment(torch)
        execution_contract_record = _freeze_teacher_execution_contract(
            output,
            num_workers=int(num_workers),
            environment=teacher_environment,
            preflight_receipt=preflight_receipt,
            vocabulary_digests=vocabulary_digests,
            assignment_dependencies=assignment_dependencies,
        )
        inventory = preflight_receipt["teachers"]["heads"]
        lookup = {(int(row["seed"]), int(row["fold"])): row for row in inventory}
        if set(lookup) != {(seed, fold) for seed in MODEL_SEEDS for fold in FOLDS}:
            raise ContractError("Preflight teacher inventory is not the exact 5x5 grid")
        heldout_label_cache: dict[str, np.ndarray] = {}
        head_results: list[dict[str, Any]] = []
        for ordinal, (seed, fold) in enumerate(
            ((seed, fold) for seed in MODEL_SEEDS for fold in FOLDS), start=1
        ):
            print(
                f"score-teachers head {ordinal}/25: seed{seed}/fold{fold}", flush=True
            )
            head_result = _score_one_teacher_head(
                output,
                seed=seed,
                fold=fold,
                checkpoint_record=lookup[(seed, fold)]["checkpoint"],
                execution_contract_record=execution_contract_record,
                loader=loader,
                roster=roster,
                sealed_logits=sealed_logits,
                heldout_label_cache=heldout_label_cache,
                assignment_vocabulary_digests=vocabulary_digests,
                assignment_dependencies=assignment_dependencies,
            )
            head_results.append(head_result)
            first_head_benchmark_path = (
                output / "benchmarks/teacher_first_completed_head.json"
            )
            if ordinal == 1 and not first_head_benchmark_path.is_file():
                write_once(
                    first_head_benchmark_path,
                    json_bytes(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "component": COMPONENT,
                            "status": "PASS",
                            "created_utc": utc_now(),
                            "seed": seed,
                            "fold": fold,
                            "elapsed_seconds": float(head_result["elapsed_seconds"]),
                            "projected_seconds_for_25_heads": (
                                25.0 * float(head_result["elapsed_seconds"])
                            ),
                            "slides_per_second": float(
                                head_result["slides_per_second"]
                            ),
                            "peak_cuda_memory_bytes": int(
                                head_result["peak_cuda_memory_bytes"]
                            ),
                            "peak_rss_kib": int(head_result["peak_rss_kib"]),
                            "head_receipt": identity(
                                _head_output_directory(output, seed, fold)
                                / "receipt.json"
                            ),
                        }
                    ),
                )
        combined_artifacts, combined_checks = _combine_teacher_outputs(
            output, roster, sealed_logits
        )
        head_receipt_records = [
            identity(_head_output_directory(output, seed, fold) / "receipt.json")
            for seed in MODEL_SEEDS
            for fold in FOLDS
        ]
        result = {
            "schema_version": SCHEMA_VERSION,
            "component": COMPONENT,
            "status": "PASS",
            "created_utc": utc_now(),
            "head_census": 25,
            "inference_contract": {
                "device": "cuda",
                "autocast": "bfloat16",
                "batch_size": 1,
                "full_bags": True,
                "instance_cap": None,
                "augmentation": False,
                "model_mode": "eval",
                "float32_matmul_precision": "high",
                "patient_logit_aggregation": "equal-slide arithmetic mean",
                "attention_aggregation": "FP32 softmax then equal-slide and five-seed means",
                "num_loader_workers": int(num_workers),
            },
            "environment": teacher_environment,
            "execution_contract": execution_contract_record,
            "implementation_sources": _teacher_source_identities(),
            "checkpoint_dependencies": preflight_receipt["teachers"],
            "v13_patient_logits": identity(V13_PATIENT_LOGITS),
            "profile_receipt": identity(output / "receipts/profiles.json"),
            "assignment_manifest": identity(
                output / "assignments/source_manifest.parquet"
            ),
            "assignment_vocabulary_sha256": vocabulary_digests,
            "head_receipts": head_receipt_records,
            "first_completed_benchmark": identity(
                output / "benchmarks/teacher_first_completed_head.json"
            ),
            "head_benchmarks": [
                {
                    key: row[key]
                    for key in (
                        "seed",
                        "fold",
                        "elapsed_seconds",
                        "slides_per_second",
                        "peak_cuda_memory_bytes",
                        "maximum_absolute_v13_logit_delta",
                    )
                }
                for row in head_results
            ],
            "checks": combined_checks,
            "artifacts": combined_artifacts,
        }
        write_once(receipt_path, json_bytes(result))
    return result


def _reader_duplicate_selection(support: pd.DataFrame) -> tuple[list[int], str]:
    support_by_prototype = (
        support.set_index("prototype_id")["distinct_patient_support"].astype(int)
    )
    if sorted(support_by_prototype.index.astype(int)) != list(range(CANONICAL_K)):
        raise ContractError("Montage support roster is not the canonical 32 prototypes")
    eligible = sorted(
        int(prototype)
        for prototype, count in support_by_prototype.items()
        if int(count) >= 12
    )
    if len(eligible) >= 8:
        chosen = pcg64_rng(VOCAB_SEED).choice(
            np.asarray(eligible, dtype=np.int64), size=8, replace=False
        )
        return sorted(int(value) for value in chosen), "DUPLICATE_PREFLIGHT_PASS"
    remaining = sorted(
        (int(prototype) for prototype in support_by_prototype.index if prototype not in eligible),
        key=lambda prototype: (-int(support_by_prototype.loc[prototype]), prototype),
    )
    chosen = [*eligible, *remaining[: 8 - len(eligible)]]
    return sorted(chosen), "NAME_GATE_FAIL_INSUFFICIENT_DUPLICATE_SUPPORT"


def _reader_selections(
    candidates: pd.DataFrame, duplicated: Sequence[int]
) -> tuple[dict[tuple[int, int], Any], pd.DataFrame]:
    selections: dict[tuple[int, int], Any] = {}
    audits: list[dict[str, Any]] = []
    duplicate_set = set(int(value) for value in duplicated)
    for prototype in range(CANONICAL_K):
        block = candidates.loc[candidates["prototype_id"].eq(prototype)].copy()
        if prototype in duplicate_set:
            original, duplicate = select_duplicate_montages(
                block,
                prototype_index=prototype,
                subcohort_order=SOURCE_SUBCOHORTS,
            )
            selections[(prototype, 0)] = original
            selections[(prototype, 1)] = duplicate
        else:
            selections[(prototype, 0)] = select_montage_tiles(
                block,
                prototype_index=prototype,
                occurrence=0,
                subcohort_order=SOURCE_SUBCOHORTS,
            )
        for occurrence in (0, 1):
            key = (prototype, occurrence)
            if key not in selections:
                continue
            selection = selections[key]
            if selection.tiles["patient_id"].duplicated().any():
                raise ContractError(f"Montage {key} repeats a patient")
            audits.append(
                {
                    "prototype_id": prototype,
                    "occurrence": occurrence,
                    "selection_seed": selection.seed,
                    "selection_stage": selection.stage,
                    "support_status": selection.support_status,
                    "n_available_patients": selection.n_available_patients,
                    "n_eligible_patients": selection.n_eligible_patients,
                    "n_displayed_tiles": len(selection.tiles),
                    "patient_overlap_with_controlling_read": selection.overlap_count,
                }
            )
    if len(selections) != 40:
        raise ContractError(f"Reader selection census is {len(selections)}, expected 40")
    return selections, pd.DataFrame.from_records(audits)


def _resolved_source_wsi_inventory(roster: pd.DataFrame) -> pd.DataFrame:
    record = require_hash(
        READY_INVENTORY, READY_INVENTORY_SHA256, "label-free source WSI inventory"
    )
    if record["size_bytes"] != READY_INVENTORY_SIZE:
        raise ContractError("Label-free source WSI inventory size drifted")
    columns = ["output_id", "wsi", "mpp", "status", "size_bytes", "mtime_ns"]
    inventory = pd.read_csv(READY_INVENTORY, usecols=columns, low_memory=False)
    inventory["output_id"] = inventory["output_id"].astype(str)
    source_ids = set(roster["slide_id"].astype(str))
    inventory = inventory.loc[inventory["output_id"].isin(source_ids)].copy()
    if (
        len(inventory) != SOURCE_SLIDES
        or inventory["output_id"].duplicated().any()
        or set(inventory["output_id"]) != source_ids
        or not inventory["status"].astype(str).str.casefold().eq("ready").all()
    ):
        raise ContractError("Label-free WSI inventory does not map source slides one-to-one")
    root = SLIDE_ROOT.resolve(strict=True)
    resolved_paths: list[str] = []
    for row in inventory.itertuples(index=False):
        relative = Path(str(row.wsi))
        if relative.is_absolute():
            raise ContractError(f"WSI inventory contains an absolute path: {relative}")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ContractError(f"WSI path escapes the source root: {relative}")
        if path.suffix.casefold() not in {".svs", ".tif", ".tiff"}:
            raise ContractError(f"Unexpected WSI extension: {path}")
        stat = path.stat()
        if int(row.size_bytes) != stat.st_size or int(row.mtime_ns) != stat.st_mtime_ns:
            raise ContractError(f"WSI identity changed after inventory: {path}")
        resolved_paths.append(str(path))
    inventory["resolved_path"] = resolved_paths
    return inventory.sort_values("output_id", kind="mergesort").reset_index(drop=True)


def _validate_reader_source_preflight(
    output: Path, store: PackedFeatureStore
) -> dict[str, Any]:
    """Open and schema-check every raw/coordinate dependency before long compute."""

    import h5py
    import openslide

    from oceanpath.extraction.mpp_sampling import validate_exact_mpp_coordinate_file

    roster = _blind_roster_with_counts(output, store)
    inventory = _resolved_source_wsi_inventory(roster)
    lookup = inventory.set_index("output_id")
    for position, row in enumerate(roster.itertuples(index=False), start=1):
        slide_id = str(row.slide_id)
        wsi = lookup.loc[slide_id]
        patch_path = PATCH_ROOT / f"{slide_id}_patches.h5"
        feature_path = FEATURE_ROOT / f"{slide_id}.h5"
        attributes = validate_exact_mpp_coordinate_file(
            patch_path,
            target_mpp=0.5,
            source_mpp=float(wsi["mpp"]),
            patch_size=256,
            overlap=0,
            min_tissue_proportion=0.5,
        )
        if (
            str(attributes.get("coordinate_units")) != "level0_pixels"
            or int(attributes.get("read_level", -1)) != 0
            or int(attributes.get("patch_size_level0", -1)) <= 0
        ):
            raise ContractError(f"Exact-MPP geometry drifted for {slide_id}")
        with h5py.File(patch_path, "r") as handle:
            if "coords" not in handle or int(handle["coords"].shape[-2]) != int(row.n_tiles):
                raise ContractError(f"Patch-coordinate census drifted for {slide_id}")
        with h5py.File(feature_path, "r") as handle:
            if "coords" not in handle or int(handle["coords"].shape[-2]) != int(row.n_tiles):
                raise ContractError(f"Feature-coordinate census drifted for {slide_id}")
        with openslide.OpenSlide(str(wsi["resolved_path"])) as slide:
            if (
                slide.level_count < 1
                or slide.dimensions[0] <= 0
                or slide.dimensions[1] <= 0
            ):
                raise ContractError(f"Raw WSI geometry is invalid: {slide_id}")
        if position % 250 == 0:
            print(
                f"preflight reader sources: {position}/{SOURCE_SLIDES}",
                flush=True,
            )
    return {
        "slides": SOURCE_SLIDES,
        "patch_coordinate_files": SOURCE_SLIDES,
        "feature_coordinate_files": SOURCE_SLIDES,
        "raw_wsi_opened": SOURCE_SLIDES,
        "ready_inventory": identity(READY_INVENTORY),
        "environment": _reader_environment(),
    }


def _bind_reader_tile_provenance(
    output: Path,
    selections: Mapping[tuple[int, int], Any],
    blinding: pd.DataFrame,
    roster: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    import h5py

    from oceanpath.extraction.mpp_sampling import validate_exact_mpp_coordinate_file

    key = blinding.set_index(["prototype_id", "occurrence"])
    blocks: list[pd.DataFrame] = []
    for occurrence, selection in selections.items():
        prototype, repetition = occurrence
        block = selection.tiles.copy()
        block["prototype_id"] = prototype
        block["occurrence"] = repetition
        block["code"] = str(key.loc[occurrence, "code"])
        block["presentation_order"] = int(key.loc[occurrence, "presentation_order"])
        block["is_controlling_read"] = bool(key.loc[occurrence, "is_controlling_read"])
        blocks.append(block)
    provenance = pd.concat(blocks, ignore_index=True)
    if not set(provenance["patient_id"].astype(str)).issubset(
        set(roster["patient_id"].astype(str))
    ):
        raise ContractError("Reader package includes a non-source patient")
    wsi_inventory = _resolved_source_wsi_inventory(roster)
    wsi_lookup = wsi_inventory.set_index("output_id")
    store = PackedFeatureStore(PACK_ROOT)
    geometry: dict[str, dict[str, Any]] = {}
    evidence_rows: list[dict[str, Any]] = []
    for slide_id, rows in provenance.groupby("slide_id", sort=True):
        slide_id = str(slide_id)
        indices = sorted(set(rows["tile_index"].astype(int)))
        position = store.position(slide_id)
        n_tiles = store.length_of(slide_id)
        if min(indices) < 0 or max(indices) >= n_tiles:
            raise ContractError(f"Selected tile is outside source slide {slide_id}")
        wsi = wsi_lookup.loc[slide_id]
        patch_path = PATCH_ROOT / f"{slide_id}_patches.h5"
        feature_path = FEATURE_ROOT / f"{slide_id}.h5"
        attributes = validate_exact_mpp_coordinate_file(
            patch_path,
            target_mpp=0.5,
            source_mpp=float(wsi["mpp"]),
            patch_size=256,
            overlap=0,
            min_tissue_proportion=0.5,
        )
        if (
            str(attributes.get("coordinate_units")) != "level0_pixels"
            or int(attributes.get("read_level", -1)) != 0
            or int(attributes.get("patch_size_level0", -1)) <= 0
        ):
            raise ContractError(f"Exact-MPP geometry drifted for {slide_id}")
        with h5py.File(patch_path, "r") as handle:
            patch_coordinates = np.asarray(handle["coords"][:], dtype=np.int64)
        with h5py.File(feature_path, "r") as handle:
            if "coords" not in handle:
                raise ContractError(f"Feature H5 has no coordinates: {feature_path}")
            feature_coordinates = np.asarray(handle["coords"][:], dtype=np.int64)
        if feature_coordinates.ndim == 3 and feature_coordinates.shape[0] == 1:
            feature_coordinates = feature_coordinates[0]
        if len(patch_coordinates) != n_tiles or len(feature_coordinates) != n_tiles:
            raise ContractError(f"Coordinate row census differs from pack for {slide_id}")
        packed_coordinates = store.read_coords(
            position, rows=np.asarray(indices, dtype=np.int64)
        ).astype(np.int64)
        if not np.array_equal(packed_coordinates, patch_coordinates[indices]) or not np.array_equal(
            packed_coordinates, feature_coordinates[indices]
        ):
            raise ContractError(f"Packed/patch/feature coordinate binding failed: {slide_id}")
        assignment_path = _assignment_shard_path(
            output / "assignments/source", slide_id
        )
        with np.load(assignment_path, allow_pickle=False) as bundle:
            labels = np.asarray(bundle["reference_labels"], dtype=np.int16)
            distances = np.asarray(bundle["reference_distances"], dtype=np.float32)
        row_lookup = rows.set_index("tile_index")
        for packed_row, tile_index in enumerate(indices):
            occurrences = row_lookup.loc[[tile_index]]
            for selected in occurrences.itertuples(index=False):
                if (
                    int(labels[tile_index]) != int(selected.prototype_id)
                    or float(distances[tile_index]) != float(selected.distance)
                    or not np.array_equal(
                        packed_coordinates[packed_row],
                        np.asarray([selected.x, selected.y], dtype=np.int64),
                    )
                ):
                    raise ContractError(
                        f"Selected candidate no longer binds assignment row {slide_id}/{tile_index}"
                    )
                evidence_rows.append(
                    {
                        "slide_id": slide_id,
                        "tile_index": tile_index,
                        "prototype_id": int(selected.prototype_id),
                        "occurrence": int(selected.occurrence),
                        "code": str(selected.code),
                        "packed_patch_feature_coordinate_match": True,
                    }
                )
        geometry[slide_id] = {
            "wsi_path": str(wsi["resolved_path"]),
            "wsi_size_bytes": int(wsi["size_bytes"]),
            "wsi_mtime_ns": int(wsi["mtime_ns"]),
            "source_mpp": float(wsi["mpp"]),
            "patch_size_level0": int(attributes["patch_size_level0"]),
            "effective_target_mpp": float(attributes["effective_target_mpp"]),
            "coordinate_units": str(attributes["coordinate_units"]),
        }
    return provenance, pd.DataFrame.from_records(evidence_rows), geometry


def _render_reader_montages(
    provenance: pd.DataFrame,
    geometry: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    import h5py
    import openslide
    import PIL
    from PIL import Image, features

    rendered: dict[str, bytes] = {}
    for code, rows in provenance.groupby("code", sort=True):
        rows = rows.sort_values("montage_slot", kind="mergesort")
        if len(rows) > 12:
            raise ContractError(f"Montage {code} has more than 12 tiles")
        canvas = Image.new("RGB", (4 * 256, 3 * 256), "white")
        for row in rows.itertuples(index=False):
            specification = geometry[str(row.slide_id)]
            footprint = int(specification["patch_size_level0"])
            with openslide.OpenSlide(str(specification["wsi_path"])) as slide:
                if int(row.x) + footprint > slide.dimensions[0] or int(row.y) + footprint > slide.dimensions[1]:
                    raise ContractError(
                        f"Selected tile crosses the WSI boundary: {row.slide_id}/{row.tile_index}"
                    )
                tile = slide.read_region(
                    (int(row.x), int(row.y)), 0, (footprint, footprint)
                ).convert("RGB")
            tile = tile.resize((256, 256), Image.Resampling.LANCZOS)
            slot = int(row.montage_slot)
            canvas.paste(tile, ((slot % 4) * 256, (slot // 4) * 256))
        buffer = io.BytesIO()
        canvas.save(
            buffer,
            format="JPEG",
            quality=92,
            subsampling=0,
            optimize=False,
            progressive=False,
            exif=b"",
        )
        payload = buffer.getvalue()
        with Image.open(io.BytesIO(payload)) as check:
            if check.mode != "RGB" or check.size != (1024, 768) or len(check.getexif()) != 0:
                raise ContractError(f"Montage JPEG contract failed: {code}")
            if any(name in check.info for name in ("exif", "icc_profile", "comment")):
                raise ContractError(f"Montage JPEG contains forbidden metadata: {code}")
        rendered[str(code)] = payload
    if len(rendered) != 40:
        raise ContractError(f"Rendered {len(rendered)} montages, expected 40")
    environment = {
        "pillow": PIL.__version__,
        "jpeg_codec": features.version_codec("jpg"),
        "libjpeg_turbo": features.version_feature("libjpeg_turbo"),
        "openslide_python": getattr(openslide, "__version__", "unknown"),
        "openslide_native": getattr(openslide, "__library_version__", "unknown"),
        "h5py": h5py.__version__,
        "canvas_pixels": [1024, 768],
        "tile_pixels": [256, 256],
        "jpeg_parameters": {
            "quality": 92,
            "subsampling": 0,
            "optimize": False,
            "progressive": False,
        },
    }
    return rendered, environment


def _reader_instructions() -> str:
    return """# Blinded morphology naming session

Please review all 40 montages once, in the row order of `review_form.csv`, during one approximately two-hour session. Each montage contains up to twelve 256-pixel tissue fields on a fixed 4-by-3 canvas; unused cells are white. The nominal field width is approximately 128 micrometres.

For every blinded code, set `review_status` to exactly `complete`; select exactly one primary category; optionally select up to two distinct secondary categories; add a short morphology-only description; record confidence as an integer from 1 (very low) to 5 (very high); and set `artifact_uninterpretable` to exactly `yes` or `no`. Copy category text exactly from `RUBRIC.md`, leave unused secondary-category cells blank, and do not repeat the primary category as a secondary category. Do not describe what a pattern might predict.

Before beginning, make a copy of `review_form.csv` named `completed_review_form.csv`. Complete that copy only. Enter the same reviewer identifier in every row, enter the review date in `YYYY-MM-DD` format, and set `blinding_attestation` to exactly `confirmed_no_key_access` in every row to confirm that you did not access any hidden key or outcome information before finishing the session.

Return only `completed_review_form.csv` to the study coordinator. Do not rename, reorder, add, or remove rows, and do not open any material outside this folder.
"""


def _reader_rubric() -> str:
    categories = "\n".join(f"- {category}" for category in ONTOLOGY)
    return f"""# Fixed morphology rubric

Choose one primary category and no more than two secondary categories from this closed list:

{categories}

Use `mixed/other interpretable` only when the montage is interpretable but none of the more specific categories is adequate. If artifact makes interpretation unreliable, mark the artifact field `yes`; otherwise mark it `no`. Confidence must be an integer from 1 through 5.

Some montages were independently resampled for a prespecified consistency check, but their identities are blinded. After return, two readings count as exact agreement when their primary categories match, partial agreement when either primary appears in the other's secondary list, and different otherwise. Free text cannot change that scoring rule. A missing response or an artifact/uninterpretable flag counts as no agreement.
"""


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"Reader package contains a symlink: {path}")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def _publish_directory(stage: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if not destination.is_dir() or _tree_inventory(stage) != _tree_inventory(destination):
            raise ContractError(f"Refusing to replace nonidentical directory: {destination}")
        shutil.rmtree(stage)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, destination)


def _build_public_reader_directory(
    destination: Path,
    blinding: pd.DataFrame,
    provenance: pd.DataFrame,
    rendered: Mapping[str, bytes],
    public_salt_sha256: str,
    sealed_utc: str | None = None,
) -> None:
    tile_counts = provenance.groupby("code").size().astype(int)
    form = blinding.sort_values("presentation_order", kind="mergesort")[
        ["presentation_order", "code"]
    ].copy()
    form["presentation_order"] = form["presentation_order"].astype(int) + 1
    form = form.rename(columns={"code": "blinded_code"})
    form["n_tiles"] = form["blinded_code"].map(tile_counts).astype(int)
    for column in (
        "review_status",
        "primary_category",
        "secondary_category_1",
        "secondary_category_2",
        "free_text_description",
        "confidence_1_to_5",
        "artifact_uninterpretable",
        "reviewer_id",
        "review_date",
        "blinding_attestation",
    ):
        form[column] = ""
    expected_static = {
        "INSTRUCTIONS.md": _reader_instructions().encode("utf-8"),
        "RUBRIC.md": _reader_rubric().encode("utf-8"),
        "PUBLIC_SALT_SHA256.txt": f"{public_salt_sha256}\n".encode("ascii"),
        "review_form.csv": form.to_csv(index=False).encode("utf-8"),
    }
    if destination.is_dir():
        for name, payload in expected_static.items():
            path = destination / name
            if not path.is_file() or path.read_bytes() != payload:
                raise ContractError(f"Existing public reader artifact differs: {path}")
        actual_montages = sorted((destination / "montages").glob("*.jpg"))
        if {path.stem for path in actual_montages} != set(rendered):
            raise ContractError("Existing public montage inventory differs")
        for path in actual_montages:
            if path.read_bytes() != rendered[path.stem]:
                raise ContractError(f"Existing public montage bytes differ: {path}")
        manifest_path = destination / "HANDOFF_MANIFEST.json"
        sidecar_path = destination / "HANDOFF_MANIFEST.sha256"
        if not manifest_path.is_file() or not sidecar_path.is_file():
            raise ContractError("Existing public reader package is incompletely sealed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sealed_utc is not None and manifest.get("sealed_utc") != sealed_utc:
            raise ContractError("Existing public handoff timestamp differs")
        for record in manifest.get("files", []):
            _validate_identity_record(
                {**record, "path": str(destination / record["path"])},
                "existing public handoff manifest",
            )
        if sidecar_path.read_text(encoding="ascii") != (
            f"{sha256_file(manifest_path)}  HANDOFF_MANIFEST.json\n"
        ):
            raise ContractError("Existing public handoff sidecar differs")
        return
    if destination.exists() or destination.is_symlink():
        raise ContractError(f"Public reader destination is not a directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        montages = stage / "montages"
        montages.mkdir()
        for code, payload in rendered.items():
            atomic_write(montages / f"{code}.jpg", payload)
        instructions_path = stage / "INSTRUCTIONS.md"
        rubric_path = stage / "RUBRIC.md"
        salt_path = stage / "PUBLIC_SALT_SHA256.txt"
        write_once(instructions_path, expected_static["INSTRUCTIONS.md"])
        write_once(rubric_path, expected_static["RUBRIC.md"])
        write_once(salt_path, expected_static["PUBLIC_SALT_SHA256.txt"])
        form_path = stage / "review_form.csv"
        write_once(form_path, expected_static["review_form.csv"])
        included = [
            path
            for path in sorted(stage.rglob("*"))
            if path.is_file() and path.name not in {"HANDOFF_MANIFEST.json", "HANDOFF_MANIFEST.sha256"}
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "package": "FINAL-v14 blinded morphology naming session",
            "sealed_utc": sealed_utc or utc_now(),
            "n_montages": 40,
            "public_salt_sha256": public_salt_sha256,
            "files": [
                {
                    "path": path.relative_to(stage).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in included
            ],
        }
        manifest_path = stage / "HANDOFF_MANIFEST.json"
        write_once(manifest_path, json_bytes(manifest))
        write_once(
            stage / "HANDOFF_MANIFEST.sha256",
            f"{sha256_file(manifest_path)}  HANDOFF_MANIFEST.json\n".encode("ascii"),
        )
        _publish_directory(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _validate_public_reader_directory(
    public: Path, roster: pd.DataFrame
) -> dict[str, Any]:
    import re

    from PIL import Image

    expected_top = {
        "INSTRUCTIONS.md",
        "RUBRIC.md",
        "review_form.csv",
        "PUBLIC_SALT_SHA256.txt",
        "HANDOFF_MANIFEST.json",
        "HANDOFF_MANIFEST.sha256",
        "montages",
    }
    if {path.name for path in public.iterdir()} != expected_top:
        raise ContractError("Public reader package contains unexpected top-level material")
    montage_paths = sorted((public / "montages").glob("*.jpg"))
    form = pd.read_csv(public / "review_form.csv", keep_default_na=False)
    if len(montage_paths) != 40 or len(form) != 40:
        raise ContractError("Public reader package is not exactly 40 montages and rows")
    response_columns = [
        "review_status",
        "primary_category",
        "secondary_category_1",
        "secondary_category_2",
        "free_text_description",
        "confidence_1_to_5",
        "artifact_uninterpretable",
        "reviewer_id",
        "review_date",
        "blinding_attestation",
    ]
    expected_columns = [
        "presentation_order",
        "blinded_code",
        "n_tiles",
        *response_columns,
    ]
    if form.columns.tolist() != expected_columns:
        raise ContractError("Public review form schema drifted")
    if any(not form[column].eq("").all() for column in response_columns):
        raise ContractError("Public review form is not an uncompleted blank master")
    tile_counts = pd.to_numeric(form["n_tiles"], errors="coerce")
    if tile_counts.isna().any() or not tile_counts.between(1, 12).all():
        raise ContractError("Public review form has invalid montage tile counts")
    codes = form["blinded_code"].astype(str)
    if not codes.is_unique or not codes.str.fullmatch(r"[A-Z2-7]{6}").all():
        raise ContractError("Public blinded codes are invalid or duplicated")
    if [path.stem for path in montage_paths] != sorted(codes.tolist()):
        raise ContractError("Public montage files do not match form codes")
    if form["presentation_order"].tolist() != list(range(1, 41)):
        raise ContractError("Public form order is not the frozen presentation order")
    for path in montage_paths:
        if path.is_symlink():
            raise ContractError(f"Public montage is a symlink: {path}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (1024, 768) or len(image.getexif()) != 0:
                raise ContractError(f"Public montage image contract failed: {path}")
            if any(name in image.info for name in ("exif", "icc_profile", "comment")):
                raise ContractError(f"Public montage leaks metadata: {path}")
    manifest_path = public / "HANDOFF_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_keys = {
        "schema_version",
        "package",
        "sealed_utc",
        "n_montages",
        "public_salt_sha256",
        "files",
    }
    public_salt = (public / "PUBLIC_SALT_SHA256.txt").read_text(
        encoding="ascii"
    ).strip()
    try:
        sealed_time = dt.datetime.fromisoformat(str(manifest.get("sealed_utc", "")))
    except ValueError as exc:
        raise ContractError("Public handoff timestamp is malformed") from exc
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("package") != "FINAL-v14 blinded morphology naming session"
        or manifest.get("n_montages") != 40
        or manifest.get("public_salt_sha256") != public_salt
        or sealed_time.tzinfo is None
    ):
        raise ContractError("Public handoff manifest contract drifted")
    expected_files = {
        "INSTRUCTIONS.md",
        "RUBRIC.md",
        "review_form.csv",
        "PUBLIC_SALT_SHA256.txt",
        "HANDOFF_MANIFEST.json",
        "HANDOFF_MANIFEST.sha256",
        *(f"montages/{code}.jpg" for code in codes),
    }
    observed_files = {
        path.relative_to(public).as_posix()
        for path in public.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise ContractError("Public reader package contains an unexpected file")
    manifest_files = {str(record["path"]) for record in manifest.get("files", [])}
    if manifest_files != expected_files - {
        "HANDOFF_MANIFEST.json",
        "HANDOFF_MANIFEST.sha256",
    }:
        raise ContractError("Public handoff manifest file census drifted")
    for record in manifest.get("files", []):
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError("Public manifest contains an unsafe path")
        _validate_identity_record(
            {**record, "path": str(public / relative)}, "public handoff manifest"
        )
    sidecar = (public / "HANDOFF_MANIFEST.sha256").read_text(encoding="ascii")
    if sidecar != f"{sha256_file(manifest_path)}  HANDOFF_MANIFEST.json\n":
        raise ContractError("Public handoff manifest sidecar failed")
    public_text_paths = [
        public / "INSTRUCTIONS.md",
        public / "RUBRIC.md",
        public / "review_form.csv",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in public_text_paths)
    if (
        (public / "INSTRUCTIONS.md").read_text(encoding="utf-8")
        != _reader_instructions()
        or (public / "RUBRIC.md").read_text(encoding="utf-8")
        != _reader_rubric()
    ):
        raise ContractError("Public reader instructions or rubric drifted")
    forbidden_terms = {
        "patient_id",
        "slide_id",
        "prototype_id",
        "subcohort",
        "target_label",
        "kras",
        "outer_fold",
        "centroid_distance",
        "model_score",
        "association_statistic",
        *SOURCE_SUBCOHORTS,
    }
    lowered = text.casefold()
    leaks = sorted(term for term in forbidden_terms if term.casefold() in lowered)
    if leaks:
        raise ContractError(f"Public reader text contains forbidden terms: {leaks}")
    known_identifiers = {
        str(value)
        for column in ("patient_id", "slide_id")
        for value in roster[column]
        if len(str(value)) >= 8
    }
    if any(identifier in text for identifier in known_identifiers):
        raise ContractError("Public reader text contains a known source identifier")
    if any(path.is_symlink() for path in public.rglob("*")):
        raise ContractError("Public reader package contains a symlink")
    if not re.fullmatch(r"[0-9a-f]{64}\n", (public / "PUBLIC_SALT_SHA256.txt").read_text()):
        raise ContractError("Public salt digest file is malformed")
    return {
        "status": "PASS",
        "montages": 40,
        "form_rows": 40,
        "tree_inventory": _tree_inventory(public),
    }


def _reader_environment() -> dict[str, Any]:
    import h5py
    import openslide
    import PIL
    from PIL import features

    return {
        "pillow": PIL.__version__,
        "jpeg_codec": features.version_codec("jpg"),
        "libjpeg_turbo": features.version_feature("libjpeg_turbo"),
        "openslide_python": getattr(openslide, "__version__", "unknown"),
        "openslide_native": getattr(openslide, "__library_version__", "unknown"),
        "h5py": h5py.__version__,
    }


def _coordinator_instructions() -> str:
    return """# FINAL-v14 reader handoff

Give the pathologist only the `FOR_PATHOLOGIST` directory. Keep all files under the governed rerun's `reader_package/embargoed` directory inaccessible until the completed form has been returned and SHA-256 sealed.

On return, copy the completed form into a new governed post-reader stage, hash-seal it immediately, and only then open the embargoed key. Do not place the completed form over the blank master.
"""


def _validate_review_root(review_root: Path, roster: pd.DataFrame) -> None:
    expected = {"FOR_PATHOLOGIST", "README_COORDINATOR.md"}
    if (review_root / "PRE_READER_STATUS.json").is_file():
        expected.add("PRE_READER_STATUS.json")
    if (
        not review_root.is_dir()
        or {path.name for path in review_root.iterdir()} != expected
        or any(path.is_symlink() for path in review_root.rglob("*"))
    ):
        raise ContractError("reviews/v14 root census or symlink policy drifted")
    if (
        (review_root / "README_COORDINATOR.md").read_text(encoding="utf-8")
        != _coordinator_instructions()
    ):
        raise ContractError("Coordinator handoff instructions drifted")
    _validate_public_reader_directory(review_root / "FOR_PATHOLOGIST", roster)
    if "PRE_READER_STATUS.json" in expected:
        status = json.loads(
            (review_root / "PRE_READER_STATUS.json").read_text(encoding="utf-8")
        )
        if status.get("status") != "READY_FOR_PATHOLOGIST":
            raise ContractError("Existing pre-reader coordinator status drifted")


def _reader_dependencies(output: Path) -> dict[str, Any]:
    return {
        "profile_receipt": identity(output / "receipts/profiles.json"),
        "assignment_manifest": identity(
            output / "assignments/source_manifest.parquet"
        ),
        "canonical_candidates": identity(
            output / "montage/canonical_candidates.parquet"
        ),
        "montage_support_roster": identity(
            output / "montage/support_roster.csv"
        ),
        "canonical_vocabulary": identity(_canonical_vocabulary_path(output)),
        "implementation_sources": [
            identity(Path(__file__).resolve()),
            identity(REPO / "src/oceanpath/aim1/v14_concepts.py"),
        ],
    }


def _validate_embargoed_reader_package(
    output: Path, review_root: Path, reader_receipt: Mapping[str, Any]
) -> None:
    embargoed = output / "reader_package/embargoed"
    expected_files = {
        "secret_salt.bin",
        "handoff_sealed_utc.txt",
        "occurrence_key.csv",
        "tile_provenance.parquet",
        "coordinate_binding_evidence.parquet",
        "render_geometry.json",
        "render_environment.json",
    }
    if (
        not embargoed.is_dir()
        or {path.name for path in embargoed.iterdir()} != expected_files
        or embargoed.stat().st_mode & 0o777 != 0o700
    ):
        raise ContractError("Embargoed reader directory census or permissions drifted")
    for path in embargoed.iterdir():
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_mode & 0o777 != 0o600
        ):
            raise ContractError(f"Embargoed reader artifact is not private: {path}")
    salt_path = embargoed / "secret_salt.bin"
    salt = salt_path.read_bytes()
    if len(salt) != 32:
        raise ContractError("Embargoed HMAC salt is not 256 bits")
    public_salt = (review_root / "FOR_PATHOLOGIST/PUBLIC_SALT_SHA256.txt").read_text(
        encoding="ascii"
    ).strip()
    if salt_sha256(salt) != public_salt or reader_receipt.get(
        "public_salt_sha256"
    ) != public_salt:
        raise ContractError("Embargoed salt does not bind the public salt digest")
    key = pd.read_csv(embargoed / "occurrence_key.csv")
    required = {
        "presentation_order",
        "prototype_id",
        "occurrence",
        "code",
        "code_window_index",
        "is_controlling_read",
        "order_digest_hex",
    }
    if not required.issubset(key):
        raise ContractError("Embargoed occurrence key schema drifted")
    occurrences = [
        (int(row.prototype_id), int(row.occurrence))
        for row in key[["prototype_id", "occurrence"]].itertuples(index=False)
    ]
    replay = hmac_blinding_table(occurrences, salt)
    observed = key[list(replay.columns)].sort_values(
        "presentation_order", kind="mergesort"
    ).reset_index(drop=True)
    expected = replay.sort_values(
        "presentation_order", kind="mergesort"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
    except AssertionError as exc:
        raise ContractError("Embargoed HMAC occurrence key does not replay") from exc
    timestamp = (embargoed / "handoff_sealed_utc.txt").read_text(
        encoding="ascii"
    ).strip()
    manifest = json.loads(
        (
            review_root / "FOR_PATHOLOGIST/HANDOFF_MANIFEST.json"
        ).read_text(encoding="utf-8")
    )
    if manifest.get("sealed_utc") != timestamp:
        raise ContractError("Embargoed handoff timestamp does not bind the manifest")
    if reader_receipt.get("dependencies") != _reader_dependencies(output):
        raise ContractError("Reader-package upstream dependencies drifted")


def build_reader_package(output: Path, review_root: Path) -> dict[str, Any]:
    """Create and seal the one blinded 40-montage pathologist handoff."""

    _require_production_preflight(output)
    profile_receipt = _require_stage_receipt(output, "profiles")
    _validate_artifact_tree(
        profile_receipt.get("artifacts", {}), "reader profile inputs"
    )
    _validate_assignment_manifest(output)
    _validate_profile_replay(output)
    receipt_path = output / "receipts/reader_package.json"
    replay = _load_json_receipt_if_valid(
        receipt_path,
        status="PASS",
        artifact_keys=(
            "handoff_manifest",
            "embargoed_occurrence_key",
            "embargoed_tile_provenance",
        ),
    )
    if replay is not None:
        _validate_artifact_tree(replay.get("artifacts", {}), "reader receipt")
        roster = _blind_roster_with_counts(output, PackedFeatureStore(PACK_ROOT))
        _validate_review_root(review_root, roster)
        _validate_embargoed_reader_package(output, review_root, replay)
        return replay

    with exclusive_lock(output / "locks/reader_package.lock", "reader package"):
        roster = _blind_roster_with_counts(output, PackedFeatureStore(PACK_ROOT))
        candidates = pd.read_parquet(output / "montage/canonical_candidates.parquet")
        support = pd.read_csv(output / "montage/support_roster.csv")
        duplicated, duplicate_preflight = _reader_duplicate_selection(support)
        selections, selection_audit = _reader_selections(candidates, duplicated)
        occurrences = blinded_occurrences(list(range(CANONICAL_K)), duplicated)

        embargoed = output / "reader_package/embargoed"
        embargoed.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(embargoed, 0o700)
        salt_path = embargoed / "secret_salt.bin"
        if salt_path.is_file():
            salt = salt_path.read_bytes()
            blinding = hmac_blinding_table(occurrences, salt)
        else:
            salt, blinding = draw_collision_free_hmac_salt(occurrences)
            atomic_write(salt_path, salt, mode=0o600)
        os.chmod(salt_path, 0o600)
        public_salt = salt_sha256(salt)
        sealed_utc_path = embargoed / "handoff_sealed_utc.txt"
        if sealed_utc_path.is_file():
            handoff_sealed_utc = sealed_utc_path.read_text(encoding="ascii").strip()
            if not handoff_sealed_utc:
                raise ContractError("Frozen handoff timestamp is empty")
        else:
            handoff_sealed_utc = utc_now()
            write_once(
                sealed_utc_path,
                f"{handoff_sealed_utc}\n".encode("ascii"),
                mode=0o600,
            )
        os.chmod(sealed_utc_path, 0o600)

        occurrence_key = blinding.merge(
            selection_audit, on=["prototype_id", "occurrence"], validate="one_to_one"
        ).sort_values("presentation_order", kind="mergesort")
        key_path = embargoed / "occurrence_key.csv"
        write_once(key_path, occurrence_key.to_csv(index=False).encode("utf-8"), mode=0o600)
        os.chmod(key_path, 0o600)

        provenance, binding_evidence, geometry = _bind_reader_tile_provenance(
            output, selections, blinding, roster
        )
        provenance_path = embargoed / "tile_provenance.parquet"
        evidence_path = embargoed / "coordinate_binding_evidence.parquet"
        geometry_path = embargoed / "render_geometry.json"
        write_once(provenance_path, parquet_bytes(provenance), mode=0o600)
        write_once(evidence_path, parquet_bytes(binding_evidence), mode=0o600)
        write_once(geometry_path, json_bytes(geometry), mode=0o600)
        for path in (provenance_path, evidence_path, geometry_path):
            os.chmod(path, 0o600)

        rendered, render_environment = _render_reader_montages(provenance, geometry)
        environment_path = embargoed / "render_environment.json"
        write_once(environment_path, json_bytes(render_environment), mode=0o600)
        os.chmod(environment_path, 0o600)

        canonical_public = output / "reader_package/public"
        _build_public_reader_directory(
            canonical_public,
            blinding,
            provenance,
            rendered,
            public_salt,
            sealed_utc=handoff_sealed_utc,
        )
        public_check = _validate_public_reader_directory(canonical_public, roster)

        review_root.parent.mkdir(parents=True, exist_ok=True)
        review_stage = Path(
            tempfile.mkdtemp(prefix=f".{review_root.name}.", dir=review_root.parent)
        )
        try:
            shutil.copytree(canonical_public, review_stage / "FOR_PATHOLOGIST")
            write_once(
                review_stage / "README_COORDINATOR.md",
                _coordinator_instructions().encode("utf-8"),
            )
            _publish_directory(review_stage, review_root)
        finally:
            if review_stage.exists():
                shutil.rmtree(review_stage)
        if _tree_inventory(review_root / "FOR_PATHOLOGIST") != _tree_inventory(
            canonical_public
        ):
            raise ContractError("reviews/v14 delivery mirror differs from sealed public package")
        _validate_review_root(review_root, roster)

        result = {
            "schema_version": SCHEMA_VERSION,
            "component": COMPONENT,
            "status": "PASS",
            "created_utc": utc_now(),
            "duplicate_preflight_status": duplicate_preflight,
            "duplicated_prototype_count": len(duplicated),
            "public_salt_sha256": public_salt,
            "public_check": public_check,
            "render_environment": _reader_environment(),
            "dependencies": _reader_dependencies(output),
            "artifacts": {
                "handoff_manifest": identity(
                    review_root / "FOR_PATHOLOGIST/HANDOFF_MANIFEST.json"
                ),
                "handoff_manifest_sha256": identity(
                    review_root / "FOR_PATHOLOGIST/HANDOFF_MANIFEST.sha256"
                ),
                "embargoed_salt": identity(salt_path),
                "embargoed_occurrence_key": identity(key_path),
                "embargoed_tile_provenance": identity(provenance_path),
                "coordinate_binding_evidence": identity(evidence_path),
                "render_geometry": identity(geometry_path),
                "render_environment": identity(environment_path),
                "handoff_sealed_utc": identity(sealed_utc_path),
                "canonical_handoff_manifest": identity(
                    canonical_public / "HANDOFF_MANIFEST.json"
                ),
                "coordinator_readme": identity(
                    review_root / "README_COORDINATOR.md"
                ),
            },
        }
        write_once(receipt_path, json_bytes(result))
        _validate_embargoed_reader_package(output, review_root, result)
    return result


def _validate_profile_replay(output: Path) -> dict[str, Any]:
    roster = _blind_roster_with_counts(output, PackedFeatureStore(PACK_ROOT))
    prototype_columns = [f"prototype_{index:02d}" for index in range(CANONICAL_K)]
    matrices: dict[str, pd.DataFrame] = {}
    for key in ["reference", *(f"outer_fold_{fold}" for fold in FOLDS)]:
        path = output / f"profiles/{key}/patient_profiles.parquet"
        frame = pd.read_parquet(path)
        if (
            len(frame) != SOURCE_PATIENTS
            or frame["patient_id"].astype(str).duplicated().any()
            or prototype_columns != [
                column for column in frame if column.startswith("prototype_")
            ]
            or not np.allclose(
                frame[prototype_columns].sum(axis=1), 1.0, rtol=0, atol=1e-12
            )
        ):
            raise ContractError(f"Profile replay failed for {key}")
        matrices[key] = frame
    oof = pd.read_parquet(output / "profiles/oof_patient_profiles.parquet")
    expected_blocks = [
        matrices[f"outer_fold_{fold}"].loc[
            matrices[f"outer_fold_{fold}"]["fold"].eq(fold)
        ]
        for fold in FOLDS
    ]
    expected = pd.concat(expected_blocks, ignore_index=True).sort_values(
        "patient_id", kind="mergesort"
    ).reset_index(drop=True)
    observed = oof.sort_values("patient_id", kind="mergesort").reset_index(drop=True)
    if not observed.equals(expected):
        raise ContractError("OOF profile does not replay fold-excluded patient rows exactly")
    patient_fold = (
        roster.sort_values("slide_id").drop_duplicates("patient_id").set_index("patient_id")["fold"]
    )
    if not np.array_equal(
        observed["fold"].to_numpy(dtype=np.int64),
        observed["patient_id"].map(patient_fold).to_numpy(dtype=np.int64),
    ):
        raise ContractError("OOF profile contains a patient from the wrong vocabulary fold")
    return {"patient_matrices": 6, "oof_patients": len(observed)}


def _validate_assignment_manifest(output: Path) -> dict[str, Any]:
    manifest = pd.read_parquet(output / "assignments/source_manifest.parquet")
    shard_root = output / "assignments/source"
    actual = sorted(shard_root.glob("*.npz"))
    actual_receipts = sorted(shard_root.glob("*.npz.receipt.json"))
    if (
        len(manifest) != SOURCE_SLIDES
        or len(actual) != SOURCE_SLIDES
        or len(actual_receipts) != SOURCE_SLIDES
    ):
        raise ContractError("Assignment shard census is not exactly 1,389")
    if set(manifest["path"].map(lambda value: str(Path(value).resolve()))) != {
        str(path.resolve()) for path in actual
    }:
        raise ContractError("Assignment manifest and shard directory differ")
    if set(manifest["receipt_path"].map(lambda value: str(Path(value).resolve()))) != {
        str(path.resolve()) for path in actual_receipts
    }:
        raise ContractError("Assignment manifest and shard-receipt directory differ")
    vocabulary_digests = {
        key: sha256_file(path)
        for key, path in _assignment_vocabulary_paths(output).items()
    }
    dependencies = _assignment_dependency_contract(output, vocabulary_digests)
    store = PackedFeatureStore(PACK_ROOT)
    for row in manifest.itertuples(index=False):
        record = {
            "path": row.path,
            "size_bytes": row.size_bytes,
            "sha256": row.sha256,
        }
        _validate_identity_record(record, f"assignment shard {row.slide_id}")
        receipt_record = {
            "path": row.receipt_path,
            "size_bytes": row.receipt_size_bytes,
            "sha256": row.receipt_sha256,
        }
        _validate_identity_record(
            receipt_record, f"assignment shard receipt {row.slide_id}"
        )
        _validate_assignment_shard(
            Path(str(row.path)),
            slide_id=str(row.slide_id),
            n_tiles=store.length_of(str(row.slide_id)),
            vocabulary_digests=vocabulary_digests,
            dependencies=dependencies,
        )
    return {"shards": len(actual)}


def _validate_vocabulary_samples(output: Path) -> dict[str, Any]:
    roster = _blind_roster_with_counts(output, PackedFeatureStore(PACK_ROOT))
    fold_by_patient = (
        roster.drop_duplicates("patient_id").set_index("patient_id")["fold"].astype(int)
    )
    scopes = [(None, SOURCE_PATIENTS)] + [
        (fold, (991, 992, 991, 991, 991)[fold]) for fold in FOLDS
    ]
    for fold, expected_patients in scopes:
        sample, _, _ = _materialize_sample_ids(output, roster, fold=fold)
        if (
            len(sample) != SAMPLE_TILES
            or sample["patient_id"].nunique() != expected_patients
        ):
            raise ContractError(f"Vocabulary sample replay failed for {fold=}")
        if fold is not None and sample["patient_id"].map(fold_by_patient).eq(fold).any():
            raise ContractError(f"Outer vocabulary sample includes heldout fold {fold}")
        if sample.groupby("subcohort").size().to_dict() != {
            subcohort: 100_000 for subcohort in SOURCE_SUBCOHORTS
        }:
            raise ContractError(f"Vocabulary sample lost subcohort balance for {fold=}")
    return {"sample_censuses": 6, "tiles_per_sample": SAMPLE_TILES}


def verify(output: Path, review_root: Path) -> dict[str, Any]:
    """Replay all pre-reader gates and seal the coordinator-ready status."""

    receipt_path = output / "receipts/verification.json"
    existing: dict[str, Any] | None = None
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("status") != "READY_FOR_PATHOLOGIST":
            raise ContractError("Existing verification receipt is not reader-ready")
    prepare_receipt = _require_stage_receipt(output, "prepare", expected_status="PREPARED")
    preflight_receipt = _require_production_preflight(output)
    vocabulary_receipt = _require_stage_receipt(output, "vocabularies")
    _require_stage_receipt(output, "mappings")
    profile_receipt = _require_stage_receipt(output, "profiles")
    teacher_receipt = _require_stage_receipt(output, "teachers")
    reader_receipt = _require_stage_receipt(output, "reader_package")
    del prepare_receipt, preflight_receipt
    _freeze_outer_mappings(output)

    if vocabulary_receipt.get("fit_census") != {
        "pca_fits": 6,
        "kmeans_fits": 14,
        "kmeans_starts": 140,
    }:
        raise ContractError("Vocabulary receipt fit census drifted")
    if teacher_receipt.get("head_census") != 25:
        raise ContractError("Teacher receipt head census drifted")
    _validate_vocabulary_stage(output, vocabulary_receipt)
    _validate_artifact_tree(profile_receipt.get("artifacts", {}), "profile artifacts")
    _validate_identity_record(
        profile_receipt.get("first_completed_benchmark", {}),
        "assignment first-slide benchmark",
    )
    _validate_artifact_tree(reader_receipt.get("artifacts", {}), "reader artifacts")
    _validate_teacher_receipt(teacher_receipt, output)
    sample_check = _validate_vocabulary_samples(output)
    assignment_check = _validate_assignment_manifest(output)
    profile_check = _validate_profile_replay(output)
    roster = _blind_roster_with_counts(output, PackedFeatureStore(PACK_ROOT))
    public_check = _validate_public_reader_directory(
        review_root / "FOR_PATHOLOGIST", roster
    )
    _validate_review_root(review_root, roster)
    _validate_embargoed_reader_package(output, review_root, reader_receipt)
    if _tree_inventory(review_root / "FOR_PATHOLOGIST") != _tree_inventory(
        output / "reader_package/public"
    ):
        raise ContractError("Reader delivery mirror differs from the canonical public package")
    if reader_receipt.get("public_salt_sha256") != (
        review_root / "FOR_PATHOLOGIST/PUBLIC_SALT_SHA256.txt"
    ).read_text(encoding="ascii").strip():
        raise ContractError("Public salt digest drifted from reader receipt")
    embargoed = output / "reader_package/embargoed"
    for path in (
        embargoed,
        embargoed / "secret_salt.bin",
        embargoed / "occurrence_key.csv",
        embargoed / "tile_provenance.parquet",
    ):
        expected_mode = 0o700 if path.is_dir() else 0o600
        if path.stat().st_mode & 0o777 != expected_mode:
            raise ContractError(f"Embargoed artifact permissions are not {oct(expected_mode)}: {path}")
    source_drift = []
    for record in teacher_receipt.get("implementation_sources", []):
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            source_drift.append(str(path))
    if source_drift:
        raise ContractError(f"Teacher replay implementation changed before verification: {source_drift}")

    if existing is not None:
        _validate_identity_record(
            existing.get("artifacts", {}).get("pre_reader_status", {}),
            "pre-reader status",
        )
        _validate_identity_record(
            existing.get("artifacts", {}).get("handoff_manifest", {}),
            "verified handoff manifest",
        )
        expected_checks = {
            "vocabulary_samples": sample_check,
            "assignment_shards": assignment_check,
            "profiles": profile_check,
            "public_reader_package": public_check,
            "teacher_heads": 25,
            "v13_teacher_reproduction_maximum_absolute_delta": max(
                float(row["maximum_absolute_v13_logit_delta"])
                for row in teacher_receipt["head_benchmarks"]
            ),
        }
        if existing.get("checks") != expected_checks:
            raise ContractError("Existing verification checks do not replay exactly")
        return existing

    status_core = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": "READY_FOR_PATHOLOGIST",
        "give_to_pathologist": "FOR_PATHOLOGIST only",
        "do_not_release": str(embargoed.resolve()),
        "next_action": (
            "Return only completed_review_form.csv; hash-seal it before opening "
            "the embargoed key or running current association analyses."
        ),
        "completed_gates": {
            "pca_fits": 6,
            "kmeans_fits": 14,
            "source_profile_matrices": 6,
            "teacher_heads": 25,
            "reader_montages": 40,
        },
    }
    status_path = review_root / "PRE_READER_STATUS.json"
    if status_path.is_file():
        status_record = json.loads(status_path.read_text(encoding="utf-8"))
        if {
            key: value for key, value in status_record.items() if key != "created_utc"
        } != status_core or not status_record.get("created_utc"):
            raise ContractError("Existing pre-reader status does not replay exactly")
    else:
        status_record = {**status_core, "created_utc": utc_now()}
        write_once(status_path, json_bytes(status_record))
    _validate_review_root(review_root, roster)
    result = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": "READY_FOR_PATHOLOGIST",
        "created_utc": status_record["created_utc"],
        "checks": {
            "vocabulary_samples": sample_check,
            "assignment_shards": assignment_check,
            "profiles": profile_check,
            "public_reader_package": public_check,
            "teacher_heads": 25,
            "v13_teacher_reproduction_maximum_absolute_delta": max(
                float(row["maximum_absolute_v13_logit_delta"])
                for row in teacher_receipt["head_benchmarks"]
            ),
        },
        "artifacts": {
            "pre_reader_status": identity(status_path),
            "handoff_manifest": identity(
                review_root / "FOR_PATHOLOGIST/HANDOFF_MANIFEST.json"
            ),
        },
    }
    write_once(receipt_path, json_bytes(result))
    return result


def run_all(
    output: Path,
    review_root: Path,
    *,
    max_workers: int,
    teacher_workers: int,
) -> dict[str, Any]:
    prepare(output)
    preflight(output, deep_hash=True)
    fit_vocabularies(output)
    assign_profiles(output, max_workers=max_workers)
    score_teachers(output, num_workers=teacher_workers)
    build_reader_package(output, review_root)
    verify(output, review_root)
    return status(output, review_root)


def status(output: Path, review_root: Path) -> dict[str, Any]:
    expected = {
        "prepare": output / "receipts/prepare.json",
        "preflight": output / "receipts/preflight.json",
        "vocabularies": output / "receipts/vocabularies.json",
        "mappings": output / "receipts/mappings.json",
        "profiles": output / "receipts/profiles.json",
        "teachers": output / "receipts/teachers.json",
        "reader_package": output / "receipts/reader_package.json",
        "verification": output / "receipts/verification.json",
    }
    expected_statuses = {
        "prepare": "PREPARED",
        "preflight": "PASS",
        "vocabularies": "PASS",
        "mappings": "PASS",
        "profiles": "PASS",
        "teachers": "PASS",
        "reader_package": "PASS",
        "verification": "READY_FOR_PATHOLOGIST",
    }
    stages: dict[str, dict[str, Any]] = {}
    parsed: dict[str, dict[str, Any]] = {}
    for name, path in expected.items():
        state: dict[str, Any] = {"complete": False, "path": str(path)}
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                parsed[name] = payload
                state.update(
                    {
                        "complete": payload.get("status")
                        == expected_statuses[name],
                        "status": payload.get("status"),
                        "sha256": sha256_file(path),
                    }
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                state["error"] = f"{type(exc).__name__}: {exc}"
        stages[name] = state
    handoff_ready = False
    verification = parsed.get("verification", {})
    if stages["verification"]["complete"]:
        try:
            _validate_identity_record(
                verification.get("artifacts", {}).get("pre_reader_status", {}),
                "status pre-reader receipt",
            )
            _validate_identity_record(
                verification.get("artifacts", {}).get("handoff_manifest", {}),
                "status handoff manifest",
            )
            handoff_ready = True
        except (ContractError, OSError, KeyError, TypeError, ValueError):
            handoff_ready = False
    return {
        "component": COMPONENT,
        "output_root": str(output.resolve()),
        "review_root": str(review_root.resolve()),
        "stages": stages,
        "reader_handoff_ready": handoff_ready,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--deep-hash", action="store_true")
    commands.add_parser("status")
    # The compute commands are registered below as their implementations are
    # intentionally kept independent and directly unit-testable.
    commands.add_parser("fit-vocabularies")
    assign = commands.add_parser("assign-profiles")
    assign.add_argument("--max-workers", type=int, default=MAX_PROFILE_WORKERS)
    teachers = commands.add_parser("score-teachers")
    teachers.add_argument("--num-workers", type=int, default=4)
    commands.add_parser("build-reader-package")
    commands.add_parser("verify")
    all_parser = commands.add_parser("run-all")
    all_parser.add_argument("--max-workers", type=int, default=MAX_PROFILE_WORKERS)
    all_parser.add_argument("--teacher-workers", type=int, default=4)
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    review_root = args.review_root.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "preflight":
        result = preflight(output, deep_hash=bool(args.deep_hash))
    elif args.command == "status":
        result = status(output, review_root)
    elif args.command == "fit-vocabularies":
        result = fit_vocabularies(output)
    elif args.command == "assign-profiles":
        result = assign_profiles(output, max_workers=int(args.max_workers))
    elif args.command == "score-teachers":
        result = score_teachers(output, num_workers=int(args.num_workers))
    elif args.command == "build-reader-package":
        result = build_reader_package(output, review_root)
    elif args.command == "verify":
        result = verify(output, review_root)
    elif args.command == "run-all":
        result = run_all(
            output,
            review_root,
            max_workers=int(args.max_workers),
            teacher_workers=int(args.teacher_workers),
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
