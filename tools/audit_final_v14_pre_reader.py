#!/usr/bin/env python3
"""Independent, read-only audit of the sealed FINAL-v14 pre-reader run.

This program deliberately does not import ``final_v14_pre_reader`` or the
FINAL-v14 concept helpers.  It re-expresses the frozen contract, reconstructs
derived artifacts from their nearest upstream records, and writes a single
atomic audit receipt outside both frozen artifact trees.

The default audit is intentionally practical: it does not refit PCA/k-means,
rerun a teacher, reassign packed embeddings, or rerender WSIs.  It does hash
the pinned packed feature store and all material receipt/checkpoint artifacts.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import hmac
import json
import math
import os
import re
import stat
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

REPOSITORY = Path(__file__).resolve().parents[1]
AUDITOR = Path(__file__).resolve()
RUNNER = REPOSITORY / "tools/final_v14_pre_reader.py"
FROZEN_ROOT = REPOSITORY / "reports/reruns/final_v14_additions_20260903/e4v_pre_reader"
REVIEW_ROOT = REPOSITORY / "reviews/v14"
AUDIT_ROOT = (
    REPOSITORY / "reports/reruns/final_v14_additions_20260903/e4v_pre_reader_independent_audit"
)
AUDIT_RECEIPT = AUDIT_ROOT / "audit_receipt.json"

HOT_ROOT = Path("/mnt/wsl/oceanpath-hot")
PACK_ROOT = HOT_ROOT / "features/colon_stream/20x_256px_0px_overlap_mpp0.5/packed_uni_v1"
V13_ROOT = HOT_ROOT / "outputs/aim1/reruns/aim1_primary_cohort_5seed_v1_20260824"
V13_TEACHERS = V13_ROOT / "train/source_cv/cap8192/tcga_surgen_primary"
V13_LOGITS = V13_ROOT / "analysis/patient_native_logits.parquet"
V13_TRAINING_RECEIPT = V13_ROOT / "receipts/training_complete.json"
V13_CONTRACT = V13_ROOT / "contract.json"
FINAL_V13 = REPOSITORY / "reports/final_v13"
FINAL_V13_SNAPSHOT = REPOSITORY / "reports/snapshots/final_v13_pre_v14_20260903"

SCHEMA_VERSION = 1
COMPONENT = "final_v14_e4v_pre_reader_independent_audit"
SOURCE_SLIDES = 1_389
SOURCE_PATIENTS = 1_239
SOURCE_TILES = 16_711_039
SOURCE_MUTANT = 501
SOURCE_WILD_TYPE = 738
SOURCE_SUBCOHORTS = ("SR1482", "SR386", "TCGA-COAD", "TCGA-READ")
FOLDS = (0, 1, 2, 3, 4)
MODEL_SEEDS = (42, 43, 44, 45, 46)
FOLD_PATIENTS = (248, 247, 248, 248, 248)
FOLD_TRAINING_PATIENTS = (991, 992, 991, 991, 991)
FOLD_TRAINING_SLIDES = (1111, 1112, 1112, 1110, 1111)
VOCAB_SEED = 20_260_819
SAMPLE_TILES = 400_000
PCA_COMPONENTS = 64
FEATURE_DIM = 1_024
CANONICAL_K = 32
VARIANT_K = (24, 32, 40)
VARIANT_SEEDS = (20_260_819, 20_260_820, 20_260_821)
PROFILE_KEYS = ("reference", *(f"outer_fold_{fold}" for fold in FOLDS))
PROTOTYPE_COLUMNS = tuple(f"prototype_{index:02d}" for index in range(CANONICAL_K))

PINNED_RUNNER_SHA256 = "af206c4ea5204606f540df3e58c0089e9d5e12f6d5b18de570d3cc60f680ca61"
PINNED_HANDOFF_MANIFEST_SHA256 = "1b6f7af9f953649b81dc70756cd6981b69bd6f45fe66fdea80c40570f4fe3830"
PINNED_CORE_SHA256 = {
    "preregistration": "677aad233a2e5f7c1d2975b5436269ed5060b19596904930ae49ae8c1cff3908",
    "source_manifest": "d7087a23a84a294670080eb5090f83612f57b3376c7696ed2d0094fb9bcff8a5",
    "source_splits": "31046858b74a9e435ce7ceac263630e2966ab589c36a8693dcb0fc1060cfa9f4",
    "v13_logits": "b5ce00ffad68b26954a88bd909a53512ea25bbcaaf2f07847ce889c3f7fdd5c3",
    "v13_training_receipt": "31cc8005abf32cb1c8958d0b4ecb5ed34e927bcd1665451869d62a913bb75413",
    "v13_contract": "fcc5a6744da27b73c5771f71a05511e998d20b3b4d76ecdc8a9c8c4ce34f0973",
    "analysis_dictionary": "2914ac614a642e6076219a7cf2b966f7a0c1dd890b443d0fdd879a84f25e939e",
    "implementation_contract": "2e5e658c2c27063c490e6737fee0a036cf73f14d439ea0cc77632b33a923b470",
}
PINNED_PACK = {
    "meta.json": ("44f1f0c80f4b8740c5fc82caafa96a8ead372c561fe53cda896d83972ff3c80b", 367),
    "index.parquet": ("705464267e35e454e50ea8510bc937e9f8d6a00b8b97bb9cf1e06da5fcfa9556", 66_998),
    "features.bin": (
        "6765d9faf30f40e212075c1a650fd34bc5ef625b3fb8e169dbc14e78c2058bd4",
        49_735_520_256,
    ),
    "coords.bin": ("1a90bd95a430e433cff900c0b6b2a4f6fedcae550ae80584ee27ed7f100fe491", 194_279_376),
}
PINNED_V13_SEED_RECEIPTS = {
    42: "6450a83bf13a4fca1912001214045b359e848406bc3e3d6e040729210f9bad94",
    43: "94678c39b303fa458748126d284cebabad36efc0b74bef5a2f250e9c7ed87147",
    44: "561c3a0fd805cd47fe87f0a0f55e0210adcba07f42806c7e9cd7667d49aa36d3",
    45: "454dc92ba6f5f9382829d537bc0f734bfea421a5185841a8ab541378034ec93a",
    46: "654a00ccb3ed7c00b455aedd7ffa5ba86d2b26b15c8b70049f1e43a24f3df62d",
}
PINNED_STAGE_RECEIPTS = {
    "prepare": "18e82a78790c97a0582512fa279f28cd3c9b3d98ac9a504249fdd77f7a932a6b",
    "preflight": "5db760b8ff6bb6359dd86eb7062044b6948eb98a7640326dd3dd698ffe348263",
    "vocabularies": "483ab61ec3b02a31c2c5cb42e148c557773d151d938ae5617a5e14ce91a359b8",
    "mappings": "2a71dc5ed62e245d4c6a027ec0f095c99e60fb7399265dc0702aa0d6ca2a4b75",
    "profiles": "65a8c9d5c62ddc7eebd261d9780f34c450fff87c07c33699d4ac96e35e32d990",
    "teachers": "f57df251a67fca80436870c77429773f78dd1cf13a558b08c76ba361c8f681b5",
    "reader_package": "d607cea7b7e9ee56faf75ae46bbf24d8ce329b4922ea0b9fe63ca7b1615607fd",
    "verification": "e05617a3a9f5a63ee6a76b52f5247090d80d7264d8bd19a08f4d735b1e411d15",
}


class AuditError(RuntimeError):
    """A fail-closed independent audit violation."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _regular_file(path: Path, context: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise AuditError(f"{context}: required file is missing") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise AuditError(f"{context}: artifact is not a regular non-symlink file")


def _identity(path: Path) -> dict[str, Any]:
    _regular_file(path, "identity")
    metadata = path.stat()
    return {
        "path": str(path.resolve(strict=True)),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def _require_sha(path: Path, expected: str, context: str) -> dict[str, Any]:
    record = _identity(path)
    if record["sha256"] != expected:
        raise AuditError(f"{context}: hard-pinned SHA-256 mismatch")
    return record


def _validate_identity(
    record: Mapping[str, Any],
    context: str,
    *,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    if not {"path", "size_bytes", "sha256"}.issubset(record):
        raise AuditError(f"{context}: incomplete identity record")
    path = Path(str(record["path"]))
    if expected_path is not None and path.resolve(strict=False) != expected_path.resolve(
        strict=False
    ):
        raise AuditError(f"{context}: identity points to an unexpected path")
    observed = _identity(path)
    keys = ["size_bytes", "sha256"]
    if "mtime_ns" in record:
        keys.append("mtime_ns")
    if any(observed[key] != record.get(key) for key in keys):
        raise AuditError(f"{context}: identity no longer matches its artifact")
    return observed


def _validate_identity_tree(value: Any, context: str) -> int:
    validated = 0
    if isinstance(value, Mapping):
        if {"path", "size_bytes", "sha256"}.issubset(value):
            _validate_identity(value, context)
            return 1
        for key, child in value.items():
            validated += _validate_identity_tree(child, f"{context}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            validated += _validate_identity_tree(child, f"{context}[{index}]")
    return validated


def _load_json(path: Path, context: str) -> dict[str, Any]:
    _regular_file(path, context)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{context}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{context}: JSON root is not an object")
    return value


def _decode_json_array(value: np.ndarray, context: str) -> dict[str, Any]:
    try:
        result = json.loads(np.asarray(value, dtype=np.uint8).tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AuditError(f"{context}: invalid embedded JSON") from exc
    if not isinstance(result, dict):
        raise AuditError(f"{context}: embedded JSON is not an object")
    return result


def _assert_frame_exact(observed: pd.DataFrame, expected: pd.DataFrame, context: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise AuditError(f"{context}: exact table reconstruction failed") from exc


def _assert_numeric_close(
    observed: np.ndarray,
    expected: np.ndarray,
    context: str,
    *,
    atol: float,
) -> float:
    left = np.asarray(observed, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AuditError(f"{context}: numeric shape or finiteness failure")
    residual = float(np.max(np.abs(left - right), initial=0.0))
    if residual > atol:
        raise AuditError(f"{context}: numeric reconstruction exceeds tolerance")
    return residual


def _tree_digest(root: Path) -> tuple[str, int]:
    records: list[dict[str, Any]] = []
    for directory, names, files in os.walk(root, followlinks=False):
        names.sort()
        files.sort()
        base = Path(directory)
        for name in [*names, *files]:
            candidate = base / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise AuditError("tree integrity: symlink encountered")
            if name.startswith(".") and name != ".integrity_hash":
                raise AuditError("tree integrity: temporary or hidden orphan encountered")
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise AuditError("tree integrity: special filesystem object encountered")
        for name in files:
            path = base / name
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(records)


def _assert_exact_tree(root: Path, expected_files: set[str], context: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise AuditError(f"{context}: tree root is missing, linked, or not a directory")
    actual_files: set[str] = set()
    actual_directories = {"."}
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        relative_directory = base.relative_to(root).as_posix()
        actual_directories.add(relative_directory)
        for name in [*names, *files]:
            path = base / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise AuditError(f"{context}: symlink is forbidden")
            if name.startswith(".") and name != ".integrity_hash":
                raise AuditError(f"{context}: temporary/hidden artifact is forbidden")
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise AuditError(f"{context}: special filesystem object is forbidden")
        actual_files.update((base / name).relative_to(root).as_posix() for name in files)
    expected_directories = {"."}
    for item in expected_files:
        parent = Path(item).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_files != expected_files or actual_directories != expected_directories:
        raise AuditError(f"{context}: unexpected, missing, temporary, or orphan artifact")


def _atomic_receipt(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    stage = Path(raw)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if stage.exists():
            stage.unlink()


def balanced_integer_quotas(capacities: Mapping[str, int], budget: int) -> dict[str, int]:
    normalized = {str(key): int(value) for key, value in capacities.items()}
    if budget < 0 or any(value < 0 for value in normalized.values()):
        raise AuditError("sample plan: negative budget or capacity")
    keys = sorted(normalized)
    if not keys:
        return {}
    target = min(int(budget), sum(normalized.values()))
    quotient, remainder = divmod(target, len(keys))
    allocation = {
        key: min(normalized[key], quotient + int(index < remainder))
        for index, key in enumerate(keys)
    }
    shortfall = target - sum(allocation.values())
    active = [key for key in keys if allocation[key] < normalized[key]]
    while shortfall:
        next_active: list[str] = []
        for position, key in enumerate(active):
            if shortfall == 0:
                next_active.extend(
                    candidate
                    for candidate in active[position:]
                    if allocation[candidate] < normalized[candidate]
                )
                break
            allocation[key] += 1
            shortfall -= 1
            if allocation[key] < normalized[key]:
                next_active.append(key)
        active = next_active
    return allocation


def hierarchical_sample_plan(slides: pd.DataFrame, cap: int = SAMPLE_TILES) -> pd.DataFrame:
    required = ["subcohort", "patient_id", "slide_id", "n_tiles"]
    if not set(required).issubset(slides):
        raise AuditError("sample plan: roster schema is incomplete")
    frame = slides[required].copy()
    for column in required[:3]:
        frame[column] = frame[column].astype(str)
    frame["n_tiles"] = pd.to_numeric(frame["n_tiles"], errors="raise").astype(np.int64)
    if frame["slide_id"].duplicated().any() or (frame["n_tiles"] < 0).any():
        raise AuditError("sample plan: duplicate slide or negative capacity")
    frame = frame.sort_values(required[:3], kind="stable").reset_index(drop=True)
    target = min(int(cap), int(frame["n_tiles"].sum()))
    sub_capacity = frame.groupby("subcohort", sort=True)["n_tiles"].sum().to_dict()
    sub_quota = balanced_integer_quotas(sub_capacity, target)
    slide_quota: dict[str, int] = {}
    for subcohort in sorted(sub_quota):
        block = frame.loc[frame["subcohort"].eq(subcohort)]
        patient_capacity = block.groupby("patient_id", sort=True)["n_tiles"].sum().to_dict()
        patient_quota = balanced_integer_quotas(patient_capacity, sub_quota[subcohort])
        for patient in sorted(patient_quota):
            patient_rows = block.loc[block["patient_id"].eq(patient)]
            capacities = dict(
                zip(
                    patient_rows["slide_id"],
                    patient_rows["n_tiles"].astype(int),
                    strict=True,
                )
            )
            slide_quota.update(balanced_integer_quotas(capacities, patient_quota[patient]))
    frame["n_sample"] = frame["slide_id"].map(slide_quota).fillna(0).astype(np.int64)
    if int(frame["n_sample"].sum()) != target:
        raise AuditError("sample plan: quota did not exhaust the budget")
    return frame


def _scope_directory(root: Path, fold: int | None) -> Path:
    return root / "vocabularies" / ("reference" if fold is None else f"outer_fold_{fold}")


def _vocabulary_path(root: Path, fold: int | None, k: int, seed: int) -> Path:
    scope = _scope_directory(root, fold)
    if fold is None:
        return scope / "variants" / f"k{k}_seed{seed}" / "vocabulary.npz"
    return scope / "vocabulary.npz"


def _load_npz_vocabulary(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    _regular_file(path, "vocabulary")
    try:
        with np.load(path, allow_pickle=False) as bundle:
            if set(bundle.files) != {"centroids", "pca_mean", "pca_components", "config_json"}:
                raise AuditError("vocabulary: invalid archive schema")
            centroids = np.asarray(bundle["centroids"], dtype=np.float32)
            mean = np.asarray(bundle["pca_mean"], dtype=np.float32)
            components = np.asarray(bundle["pca_components"], dtype=np.float32)
            config = _decode_json_array(bundle["config_json"], "vocabulary")
    except (OSError, ValueError, KeyError) as exc:
        raise AuditError("vocabulary: unreadable archive") from exc
    if not all(np.isfinite(value).all() for value in (centroids, mean, components)):
        raise AuditError("vocabulary: nonfinite array")
    return centroids, mean, components, config


def _pca_parameters() -> dict[str, Any]:
    return {
        "n_components": PCA_COMPONENTS,
        "whiten": False,
        "svd_solver": "randomized",
        "random_state": VOCAB_SEED,
    }


def _kmeans_parameters(k: int, seed: int) -> dict[str, Any]:
    return {
        "n_clusters": k,
        "n_init": 10,
        "max_iter": 300,
        "tol": 1e-4,
        "algorithm": "lloyd",
        "random_state": seed,
    }


def _replay_sample_ids(
    plan: pd.DataFrame,
    sample: pd.DataFrame,
    offset_by_slide: Mapping[str, int],
) -> None:
    required = [
        "subcohort",
        "patient_id",
        "slide_id",
        "tile_id",
        "tile_index",
        "global_tile_index",
        "within_slide_draw_index",
        "sample_index",
    ]
    if sample.columns.tolist() != required or len(sample) != SAMPLE_TILES:
        raise AuditError("vocabulary sample: schema or census mismatch")
    rng = np.random.Generator(np.random.PCG64(VOCAB_SEED))
    cursor = 0
    for row in plan.itertuples(index=False):
        n_tiles = int(row.n_tiles)
        n_sample = int(row.n_sample)
        block = sample.iloc[cursor : cursor + n_sample]
        drawn = np.asarray(rng.choice(n_tiles, size=n_sample, replace=False), dtype=np.int64)
        order = np.argsort(drawn, kind="stable")
        expected_positions = drawn[order]
        expected_draw_indices = np.arange(n_sample, dtype=np.int64)[order]
        if len(block) != n_sample:
            raise AuditError("vocabulary sample: truncated slide block")
        metadata_ok = (
            block["subcohort"].astype(str).eq(str(row.subcohort)).all()
            and block["patient_id"].astype(str).eq(str(row.patient_id)).all()
            and block["slide_id"].astype(str).eq(str(row.slide_id)).all()
        )
        if not metadata_ok:
            raise AuditError("vocabulary sample: noncanonical hierarchy order")
        observed_positions = block["tile_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed_positions, expected_positions):
            raise AuditError("vocabulary sample: PCG64 tile draw does not replay")
        if not np.array_equal(
            block["within_slide_draw_index"].to_numpy(dtype=np.int64),
            expected_draw_indices,
        ):
            raise AuditError("vocabulary sample: draw-index binding mismatch")
        expected_global = expected_positions + int(offset_by_slide[str(row.slide_id)])
        if not np.array_equal(block["global_tile_index"].to_numpy(dtype=np.int64), expected_global):
            raise AuditError("vocabulary sample: global packed row binding mismatch")
        expected_ids = np.asarray([f"{value:012d}" for value in expected_positions])
        if not np.array_equal(block["tile_id"].astype(str).to_numpy(), expected_ids):
            raise AuditError("vocabulary sample: tile identifier binding mismatch")
        cursor += n_sample
    if cursor != SAMPLE_TILES or not np.array_equal(
        sample["sample_index"].to_numpy(dtype=np.int64), np.arange(SAMPLE_TILES)
    ):
        raise AuditError("vocabulary sample: global sample order mismatch")


def _nearest_centroid_summary(
    projected: np.ndarray, centroids: np.ndarray, *, batch_size: int = 8_192
) -> tuple[np.ndarray, float]:
    counts = np.zeros(len(centroids), dtype=np.int64)
    inertia = 0.0
    center_norms = np.sum(centroids * centroids, axis=1)
    for start in range(0, len(projected), batch_size):
        block = np.asarray(projected[start : start + batch_size], dtype=np.float32)
        squared = (
            np.sum(block * block, axis=1, keepdims=True)
            + center_norms[None, :]
            - 2.0 * block @ centroids.T
        )
        np.maximum(squared, 0.0, out=squared)
        labels = np.argmin(squared, axis=1)
        counts += np.bincount(labels, minlength=len(centroids))
        inertia += float(np.sum(squared[np.arange(len(block)), labels], dtype=np.float64))
    return counts, inertia


def _backprojected_centroids(
    centroids: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    restored = centroids @ components + mean[None, :]
    norms = np.linalg.norm(restored.astype(np.float32), axis=1, keepdims=True)
    return restored.astype(np.float32) / np.maximum(norms, np.float32(1e-12))


def _recompute_mapping(source: np.ndarray, reference: np.ndarray) -> pd.DataFrame:
    # The governed mapper normalizes the inverse-PCA rows and its generic
    # cosine helper independently normalizes them once more.
    source_norm = source / np.maximum(
        np.linalg.norm(source.astype(np.float32), axis=1, keepdims=True),
        np.float32(1e-12),
    )
    reference_norm = reference / np.maximum(
        np.linalg.norm(reference.astype(np.float32), axis=1, keepdims=True),
        np.float32(1e-12),
    )
    similarities = np.asarray(source_norm @ reference_norm.T, dtype=np.float64)
    rows, columns = linear_sum_assignment(-similarities)
    lookup = {int(row): int(column) for row, column in zip(rows, columns, strict=True)}
    records = []
    for source_id in range(len(source)):
        reference_id = lookup[source_id]
        cosine = float(similarities[source_id, reference_id])
        records.append(
            {
                "source_prototype_id": source_id,
                "reference_prototype_id": reference_id,
                "cosine_similarity": cosine,
                "name_mappable": bool(cosine >= 0.80),
            }
        )
    return pd.DataFrame.from_records(records)


def _audit_core(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if root.resolve(strict=False) != FROZEN_ROOT.resolve(strict=False):
        raise AuditError("core identities: audit root is not the hard-pinned production root")
    _require_sha(RUNNER, PINNED_RUNNER_SHA256, "frozen runner")
    _require_sha(
        REPOSITORY / "reports/final_v14_PREREGISTRATION.md",
        PINNED_CORE_SHA256["preregistration"],
        "preregistration",
    )
    _require_sha(
        root / "analysis_dictionary.json",
        PINNED_CORE_SHA256["analysis_dictionary"],
        "analysis dictionary",
    )
    _require_sha(
        root / "implementation_contract.json",
        PINNED_CORE_SHA256["implementation_contract"],
        "implementation contract",
    )

    receipts: dict[str, dict[str, Any]] = {}
    for name, expected_hash in PINNED_STAGE_RECEIPTS.items():
        path = root / f"receipts/{name}.json"
        _require_sha(path, expected_hash, f"{name} stage receipt")
        receipts[name] = _load_json(path, f"{name} stage receipt")
    expected_status = {
        "prepare": "PREPARED",
        "preflight": "PASS",
        "vocabularies": "PASS",
        "mappings": "PASS",
        "profiles": "PASS",
        "teachers": "PASS",
        "reader_package": "PASS",
        "verification": "READY_FOR_PATHOLOGIST",
    }
    for name, status_value in expected_status.items():
        if (
            receipts[name].get("schema_version") != SCHEMA_VERSION
            or receipts[name].get("component") != "final_v14_e4v_pre_reader"
            or receipts[name].get("status") != status_value
        ):
            raise AuditError(f"{name} stage receipt: frozen status contract mismatch")

    for artifact_name, role in (
        ("analysis_dictionary.json", "analysis_dictionary_pre_model_freeze"),
        ("implementation_contract.json", "pre_reader_implementation_freeze"),
    ):
        seal = _load_json(root / f"{artifact_name}.seal.json", f"{artifact_name} seal")
        if seal.get("role") != role:
            raise AuditError(f"{artifact_name} seal: role mismatch")
        _validate_identity(
            seal.get("artifact", {}),
            f"{artifact_name} seal",
            expected_path=root / artifact_name,
        )

    dictionary = _load_json(root / "analysis_dictionary.json", "analysis dictionary")
    population = dictionary.get("population", {})
    if population != {
        "name": "tcga_surgen_primary",
        "slides": SOURCE_SLIDES,
        "patients": SOURCE_PATIENTS,
        "mutant": SOURCE_MUTANT,
        "wild_type": SOURCE_WILD_TYPE,
        "source_subcohort_order": list(SOURCE_SUBCOHORTS),
    }:
        raise AuditError("analysis dictionary: source population mismatch")
    ordered = dictionary.get("ordered_indices", {})
    if (
        ordered.get("outer_folds") != list(FOLDS)
        or ordered.get("model_seeds") != list(MODEL_SEEDS)
        or ordered.get("prototype_ids") != list(range(CANONICAL_K))
    ):
        raise AuditError("analysis dictionary: governed index order mismatch")
    vocabulary_contract = dictionary.get("vocabulary", {})
    if (
        vocabulary_contract.get("sample_tiles") != SAMPLE_TILES
        or vocabulary_contract.get("pca_components") != PCA_COMPONENTS
        or vocabulary_contract.get("pca_seed") != VOCAB_SEED
        or vocabulary_contract.get("canonical_k") != CANONICAL_K
        or vocabulary_contract.get("canonical_seed") != VOCAB_SEED
        or vocabulary_contract.get("mapping_cosine_threshold") != 0.80
    ):
        raise AuditError("analysis dictionary: vocabulary contract mismatch")

    implementation = _load_json(root / "implementation_contract.json", "implementation contract")
    if (
        implementation.get("randomness", {}).get("numpy_generator")
        != "numpy.random.Generator(PCG64)"
        or implementation.get("randomness", {}).get("global_seed") != VOCAB_SEED
        or implementation.get("compute_limits", {}).get("profile_workers") != 4
        or implementation.get("montage", {}).get("tile_px") != 256
        or implementation.get("montage", {}).get("canvas")
        != "four columns by three rows for 12 tiles; white unused cells"
    ):
        raise AuditError("implementation contract: frozen computation policy mismatch")

    left_digest, left_count = _tree_digest(FINAL_V13)
    right_digest, right_count = _tree_digest(FINAL_V13_SNAPSHOT)
    if (left_digest, left_count) != (right_digest, right_count):
        raise AuditError("FINAL-v13 snapshot: byte tree differs from FINAL-v13")

    preflight = receipts["preflight"]
    if not preflight.get("deep_hash"):
        raise AuditError("preflight: production deep hash was not asserted")
    pack = preflight.get("pack", {})
    if (
        pack.get("slides") != 2_128
        or pack.get("tiles") != 24_284_922
        or pack.get("feature_dim") != FEATURE_DIM
        or pack.get("dtype") != "float16"
    ):
        raise AuditError("preflight: packed store census mismatch")
    for name, (expected_hash, expected_size) in PINNED_PACK.items():
        path = PACK_ROOT / name
        if path.stat().st_size != expected_size:
            raise AuditError(f"packed store {name}: hard-pinned size mismatch")
        _require_sha(path, expected_hash, f"packed store {name}")
        record = pack.get("artifacts", {}).get(name, {})
        if (
            record.get("sha256") != expected_hash
            or record.get("expected_sha256") != expected_hash
            or record.get("size_bytes") != expected_size
            or not record.get("hash_verified")
        ):
            raise AuditError(f"preflight: packed store {name} receipt mismatch")
        _validate_identity(record, f"preflight packed store {name}", expected_path=path)

    prepare = receipts["prepare"]
    _validate_identity_tree(prepare.get("inputs", {}), "prepare inputs")
    if prepare.get("governance", {}).get("snapshot_diff") != "PASS_BYTE_IDENTICAL":
        raise AuditError("prepare: FINAL-v13 snapshot status mismatch")
    if (
        prepare.get("inputs", {}).get("source_manifest", {}).get("sha256")
        != PINNED_CORE_SHA256["source_manifest"]
        or prepare.get("inputs", {}).get("source_splits", {}).get("sha256")
        != PINNED_CORE_SHA256["source_splits"]
    ):
        raise AuditError("prepare: source pins mismatch")

    return receipts, {
        "pinned_stage_receipts": len(receipts),
        "pinned_pack_artifacts": len(PINNED_PACK),
        "final_v13_snapshot_files": left_count,
        "packed_feature_bytes_deep_hashed": PINNED_PACK["features.bin"][1],
    }


def _audit_source_and_pack(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_path = root / "inputs/tcga_surgen_primary.csv"
    splits_path = root / "inputs/splits/splits.parquet"
    blind_path = root / "inputs/source_roster_label_blind.csv"
    _require_sha(source_path, PINNED_CORE_SHA256["source_manifest"], "source manifest")
    _require_sha(splits_path, PINNED_CORE_SHA256["source_splits"], "source splits")
    source = pd.read_csv(source_path, low_memory=False)
    splits = pd.read_parquet(splits_path)
    blind = pd.read_csv(blind_path, low_memory=False)
    required_source = {
        "slide_id",
        "patient_id",
        "target_label",
        "cohort",
        "subcohort",
        "specimen_role",
        "k_fold",
    }
    if not required_source.issubset(source) or source["slide_id"].astype(str).duplicated().any():
        raise AuditError("source roster: schema or slide uniqueness mismatch")
    patients = source.sort_values("slide_id").drop_duplicates("patient_id")
    labels = pd.to_numeric(patients["target_label"], errors="raise").astype(int)
    folds = pd.to_numeric(patients["k_fold"], errors="raise").astype(int)
    if (
        len(source) != SOURCE_SLIDES
        or len(patients) != SOURCE_PATIENTS
        or int(labels.sum()) != SOURCE_MUTANT
        or int((labels == 0).sum()) != SOURCE_WILD_TYPE
        or folds.value_counts().sort_index().tolist() != list(FOLD_PATIENTS)
        or set(source["subcohort"].astype(str)) != set(SOURCE_SUBCOHORTS)
    ):
        raise AuditError("source roster: governed census mismatch")
    expected_blind_columns = [
        "slide_id",
        "patient_id",
        "cohort",
        "subcohort",
        "specimen_role",
        "native_mpp",
        "mpp_bin",
        "patch_count",
        "fold",
    ]
    if blind.columns.tolist() != expected_blind_columns:
        raise AuditError("label-blind roster: exact schema mismatch")
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
    if forbidden & {column.casefold() for column in blind.columns}:
        raise AuditError("label-blind roster: outcome leakage")
    expected_blind = source[[column for column in expected_blind_columns if column != "fold"]]
    expected_blind = expected_blind.merge(
        splits[["slide_id", "fold"]], on="slide_id", validate="one_to_one"
    )
    expected_blind["fold"] = pd.to_numeric(expected_blind["fold"], errors="raise").astype(int)
    expected_blind = expected_blind.sort_values(
        ["subcohort", "patient_id", "slide_id"], kind="mergesort"
    ).reset_index(drop=True)
    _assert_frame_exact(blind, expected_blind, "label-blind roster")
    if blind.groupby("patient_id")["fold"].nunique().max() != 1:
        raise AuditError("label-blind roster: patient crosses outer folds")

    pack_index = pd.read_parquet(PACK_ROOT / "index.parquet")
    if pack_index.columns.tolist() != ["slide_id", "offset", "n_patches"]:
        raise AuditError("packed index: schema mismatch")
    if (
        len(pack_index) != 2_128
        or int(pack_index["n_patches"].sum()) != 24_284_922
        or pack_index["slide_id"].astype(str).duplicated().any()
    ):
        raise AuditError("packed index: census mismatch")
    indexed = pack_index.set_index(pack_index["slide_id"].astype(str))
    if not set(blind["slide_id"].astype(str)).issubset(indexed.index):
        raise AuditError("packed index: source slide missing")
    n_tiles = blind["slide_id"].astype(str).map(indexed["n_patches"]).astype(np.int64)
    if not np.array_equal(n_tiles, blind["patch_count"].to_numpy(dtype=np.int64)):
        raise AuditError("packed index: source patch counts mismatch")
    if int(n_tiles.sum()) != SOURCE_TILES:
        raise AuditError("packed index: source tile census mismatch")
    roster = blind.copy()
    roster["n_tiles"] = n_tiles
    roster["global_offset"] = roster["slide_id"].astype(str).map(indexed["offset"]).astype(np.int64)
    return (
        roster,
        pack_index,
        {
            "slides": len(roster),
            "patients": roster["patient_id"].nunique(),
            "tiles": int(roster["n_tiles"].sum()),
            "fold_patient_census": list(FOLD_PATIENTS),
        },
    )


def _resolve_record_path(record: Mapping[str, Any], base: Path, context: str) -> Path:
    raw = Path(str(record.get("path", "")))
    path = raw if raw.is_absolute() else base / raw
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(base.resolve(strict=True)):
        raise AuditError(f"{context}: relative identity escapes its governed root")
    return resolved


def _validate_relative_identity(record: Mapping[str, Any], base: Path, context: str) -> Path:
    path = _resolve_record_path(record, base, context)
    _validate_identity({**record, "path": str(path)}, context, expected_path=path)
    return path


def _audit_v13_chain(preflight: Mapping[str, Any]) -> dict[str, Any]:
    _require_sha(V13_CONTRACT, PINNED_CORE_SHA256["v13_contract"], "FINAL-v13 contract")
    _require_sha(
        V13_TRAINING_RECEIPT,
        PINNED_CORE_SHA256["v13_training_receipt"],
        "FINAL-v13 training receipt",
    )
    _require_sha(V13_LOGITS, PINNED_CORE_SHA256["v13_logits"], "FINAL-v13 logits")
    contract = _load_json(V13_CONTRACT, "FINAL-v13 contract")
    arm = contract.get("arm_definitions", {}).get("tcga_surgen_primary", {})
    if (
        contract.get("campaign") != "aim1_primary_cohort_oof_5seed"
        or contract.get("seeds") != list(MODEL_SEEDS)
        or contract.get("n_folds") != 5
        or contract.get("refit_count") != 0
        or contract.get("prediction_scale") != "native logits"
        or arm.get("expected_patients") != SOURCE_PATIENTS
        or arm.get("expected_slides") != SOURCE_SLIDES
        or arm.get("expected_test_patients") != list(FOLD_PATIENTS)
    ):
        raise AuditError("FINAL-v13 contract: relevant training policy mismatch")
    feature_artifacts = contract.get("feature_store", {}).get("artifacts", {})
    for name, (expected_hash, expected_size) in PINNED_PACK.items():
        record = feature_artifacts.get(name, {})
        if record.get("sha256") != expected_hash or record.get("size_bytes") != expected_size:
            raise AuditError("FINAL-v13 contract: packed feature identity mismatch")

    training = _load_json(V13_TRAINING_RECEIPT, "FINAL-v13 training receipt")
    if (
        training.get("status") != "complete_and_certified"
        or training.get("seeds") != list(MODEL_SEEDS)
        or training.get("folds_per_job") != 5
        or training.get("refit_count") != 0
    ):
        raise AuditError("FINAL-v13 training receipt: certification mismatch")
    _validate_identity(
        training.get("contract", {}), "FINAL-v13 training contract", expected_path=V13_CONTRACT
    )
    _validate_identity(training.get("preflight", {}), "FINAL-v13 training preflight")
    _validate_identity(training.get("scheduler", {}), "FINAL-v13 training scheduler")
    job_records = training.get("job_receipts", [])
    selected_job_records = {
        int(re.search(r"seed(\d+)\.json$", str(record.get("path", ""))).group(1)): record
        for record in job_records
        if "/tcga_surgen_primary/seed" in str(record.get("path", ""))
        and re.search(r"seed(\d+)\.json$", str(record.get("path", "")))
    }
    if set(selected_job_records) != set(MODEL_SEEDS):
        raise AuditError("FINAL-v13 training receipt: source-arm seed grid mismatch")

    preflight_teachers = preflight.get("teachers", {})
    if preflight_teachers.get("count") != 25:
        raise AuditError("FINAL-v14 preflight: teacher census mismatch")
    preflight_heads = {
        (int(record["seed"]), int(record["fold"])): record
        for record in preflight_teachers.get("heads", [])
    }
    expected_grid = {(seed, fold) for seed in MODEL_SEEDS for fold in FOLDS}
    if set(preflight_heads) != expected_grid:
        raise AuditError("FINAL-v14 preflight: teacher grid mismatch")

    checkpoint_bytes = 0
    artifacts_validated = 0
    for seed in MODEL_SEEDS:
        seed_receipt_path = V13_ROOT / f"receipts/source_cv/tcga_surgen_primary/seed{seed}.json"
        seed_identity = _require_sha(
            seed_receipt_path,
            PINNED_V13_SEED_RECEIPTS[seed],
            "FINAL-v13 source-CV seed receipt",
        )
        listed = selected_job_records[seed]
        if (
            listed.get("sha256") != seed_identity["sha256"]
            or listed.get("size_bytes") != seed_identity["size_bytes"]
        ):
            raise AuditError("FINAL-v13 training receipt: seed receipt binding mismatch")
        seed_receipt = _load_json(seed_receipt_path, "FINAL-v13 source-CV seed receipt")
        if (
            seed_receipt.get("status") != "completed"
            or seed_receipt.get("arm") != "tcga_surgen_primary"
            or seed_receipt.get("seed") != seed
            or seed_receipt.get("fit_count") != 5
            or seed_receipt.get("refit_count") != 0
        ):
            raise AuditError("FINAL-v13 seed receipt: training census mismatch")
        artifacts_validated += _validate_identity_tree(seed_receipt, "FINAL-v13 seed receipt")
        seed_dir = V13_TEACHERS / f"seed{seed}"
        completion_record = seed_receipt.get("artifacts", {}).get("completion", {})
        completion_path = _validate_identity(
            completion_record,
            "FINAL-v13 seed training completion",
            expected_path=seed_dir / "training_completion.json",
        )["path"]
        completion = _load_json(Path(completion_path), "FINAL-v13 seed training completion")
        if (
            completion.get("status") != "completed"
            or completion.get("n_folds") != 5
            or completion.get("skip_finalize") is not True
            or len(completion.get("fold_completions", [])) != 5
        ):
            raise AuditError("FINAL-v13 seed completion: fold policy mismatch")
        fold_records = {
            int(
                re.search(r"fold_(\d+)/completion\.json$", str(item.get("path", ""))).group(1)
            ): item
            for item in completion["fold_completions"]
            if re.search(r"fold_(\d+)/completion\.json$", str(item.get("path", "")))
        }
        if set(fold_records) != set(FOLDS):
            raise AuditError("FINAL-v13 seed completion: fold grid mismatch")
        for fold in FOLDS:
            fold_dir = seed_dir / f"fold_{fold}"
            fold_completion_path = _validate_relative_identity(
                fold_records[fold], seed_dir, "FINAL-v13 fold completion"
            )
            fold_completion = _load_json(fold_completion_path, "FINAL-v13 fold completion")
            if (
                fold_completion.get("status") != "completed"
                or fold_completion.get("fold") != fold
                or fold_completion.get("training_fingerprint")
                != completion.get("training_fingerprint")
            ):
                raise AuditError("FINAL-v13 fold completion: identity mismatch")
            artifacts = fold_completion.get("artifacts", {})
            if set(artifacts) != {
                "best_checkpoint",
                "config",
                "metrics",
                "sampling_plan",
                "test_predictions",
                "val_predictions",
            }:
                raise AuditError("FINAL-v13 fold completion: artifact schema mismatch")
            paths = {
                name: _validate_relative_identity(record, fold_dir, "FINAL-v13 fold artifact")
                for name, record in artifacts.items()
            }
            metrics = _load_json(paths["metrics"], "FINAL-v13 fold metrics")
            checkpoint = Path(str(metrics.get("best_checkpoint", ""))).resolve(strict=False)
            if checkpoint != paths["best_checkpoint"] or metrics.get("fold") != fold:
                raise AuditError("FINAL-v13 fold metrics: best-checkpoint binding mismatch")
            preflight_head = preflight_heads[(seed, fold)]
            for name, path in (
                ("metrics", paths["metrics"]),
                ("completion", fold_completion_path),
                ("checkpoint", checkpoint),
            ):
                _validate_identity(
                    preflight_head.get(name, {}),
                    "FINAL-v14 preflight teacher dependency",
                    expected_path=path,
                )
            checkpoint_bytes += checkpoint.stat().st_size
            artifacts_validated += len(artifacts) + 1
    return {
        "seed_receipts": len(MODEL_SEEDS),
        "teacher_heads": len(preflight_heads),
        "checkpoint_bytes_hashed": checkpoint_bytes,
        "chain_artifacts_validated": artifacts_validated,
    }


def _audit_vocabularies(
    root: Path,
    roster: pd.DataFrame,
    pack_index: pd.DataFrame,
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    vocabulary_receipt = receipts["vocabularies"]
    if (
        vocabulary_receipt.get("fit_census")
        != {
            "pca_fits": 6,
            "kmeans_fits": 14,
            "kmeans_starts": 140,
        }
        or vocabulary_receipt.get("sample_tiles_per_fit") != SAMPLE_TILES
    ):
        raise AuditError("vocabularies: fit census mismatch")
    _validate_identity_tree(vocabulary_receipt.get("artifacts", {}), "vocabulary receipt")
    offset_by_slide = (
        pack_index.assign(slide_id=pack_index["slide_id"].astype(str))
        .set_index("slide_id")["offset"]
        .astype(np.int64)
        .to_dict()
    )
    fold_by_patient = (
        roster.drop_duplicates("patient_id").set_index("patient_id")["fold"].astype(int)
    )
    canonical_backprojected: dict[str, np.ndarray] = {}
    vocabulary_hashes: dict[str, str] = {}
    maximum_orthogonality_residual = 0.0
    maximum_relative_inertia_residual = 0.0
    noncanonical_boundary_rows = 0
    for fold in (None, *FOLDS):
        scope = _scope_directory(root, fold)
        eligible = roster if fold is None else roster.loc[roster["fold"].ne(fold)]
        plan_path = scope / "sample_plan.parquet"
        sample_path = scope / "sample_ids.parquet"
        projection_path = scope / "sample_pca64.float32.npy"
        basis_path = scope / "pca_basis.npz"
        plan = pd.read_parquet(plan_path)
        expected_plan = hierarchical_sample_plan(eligible)
        _assert_frame_exact(plan, expected_plan, "vocabulary sample plan")
        if plan.groupby("subcohort")["n_sample"].sum().to_dict() != {
            subcohort: 100_000 for subcohort in SOURCE_SUBCOHORTS
        }:
            raise AuditError("vocabulary sample plan: subcohort balance mismatch")
        expected_patients = SOURCE_PATIENTS if fold is None else FOLD_TRAINING_PATIENTS[fold]
        expected_slides = SOURCE_SLIDES if fold is None else FOLD_TRAINING_SLIDES[fold]
        if plan["patient_id"].nunique() != expected_patients or len(plan) != expected_slides:
            raise AuditError("vocabulary sample plan: scope census mismatch")
        if fold is not None and plan["patient_id"].map(fold_by_patient).eq(fold).any():
            raise AuditError("vocabulary sample plan: heldout-patient leakage")
        sample = pd.read_parquet(sample_path)
        _replay_sample_ids(plan, sample, offset_by_slide)
        if sample["patient_id"].nunique() != expected_patients:
            raise AuditError("vocabulary sample: patient census mismatch")

        try:
            projection = np.load(projection_path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise AuditError("vocabulary projection: unreadable array") from exc
        if (
            projection.shape != (SAMPLE_TILES, PCA_COMPONENTS)
            or projection.dtype != np.dtype("float32")
            or not np.isfinite(projection).all()
        ):
            raise AuditError("vocabulary projection: shape, dtype, or finiteness mismatch")
        with np.load(basis_path, allow_pickle=False) as bundle:
            if set(bundle.files) != {"pca_mean", "pca_components", "config_json"}:
                raise AuditError("PCA basis: archive schema mismatch")
            mean = np.asarray(bundle["pca_mean"], dtype=np.float32)
            components = np.asarray(bundle["pca_components"], dtype=np.float32)
            basis_config = _decode_json_array(bundle["config_json"], "PCA basis")
        scope_name = "full_source" if fold is None else f"outer_training_fold_{fold}"
        if (
            mean.shape != (FEATURE_DIM,)
            or components.shape != (PCA_COMPONENTS, FEATURE_DIM)
            or not np.isfinite(mean).all()
            or not np.isfinite(components).all()
            or basis_config.get("scope") != scope_name
            or basis_config.get("heldout_fold") != fold
            or basis_config.get("n_sample_tiles") != SAMPLE_TILES
            or basis_config.get("parameters") != _pca_parameters()
            or basis_config.get("sample_ids_sha256") != sha256_file(sample_path)
            or basis_config.get("sample_plan_sha256") != sha256_file(plan_path)
            or basis_config.get("projected_coordinates_sha256") != sha256_file(projection_path)
        ):
            raise AuditError("PCA basis: semantic or dependency contract mismatch")
        gram = components @ components.T
        orthogonality = float(np.max(np.abs(gram - np.eye(PCA_COMPONENTS)), initial=0.0))
        maximum_orthogonality_residual = max(maximum_orthogonality_residual, orthogonality)
        if orthogonality > 2e-5:
            raise AuditError("PCA basis: components are not numerically orthonormal")

        specifications = (
            [(k, seed) for k in VARIANT_K for seed in VARIANT_SEEDS]
            if fold is None
            else [(CANONICAL_K, VOCAB_SEED)]
        )
        for k, seed in specifications:
            vocabulary_path = _vocabulary_path(root, fold, k, seed)
            centroids, vocabulary_mean, vocabulary_components, config = _load_npz_vocabulary(
                vocabulary_path
            )
            if (
                centroids.shape != (k, PCA_COMPONENTS)
                or not np.array_equal(vocabulary_mean, mean)
                or not np.array_equal(vocabulary_components, components)
                or config.get("scope") != scope_name
                or config.get("heldout_fold") != fold
                or config.get("n_prototypes") != k
                or config.get("kmeans_seed") != seed
                or config.get("normalize") != "l2"
                or config.get("pca") != _pca_parameters()
                or config.get("kmeans") != _kmeans_parameters(k, seed)
                or config.get("n_sample_tiles") != SAMPLE_TILES
                or config.get("sample_ids_sha256") != sha256_file(sample_path)
                or config.get("sample_plan_sha256") != sha256_file(plan_path)
                or config.get("pca_basis_sha256") != sha256_file(basis_path)
                or config.get("projected_coordinates_sha256") != sha256_file(projection_path)
            ):
                raise AuditError("vocabulary: semantic or dependency contract mismatch")
            declared_counts = np.asarray(config.get("cluster_sizes", []), dtype=np.int64)
            if (
                declared_counts.shape != (k,)
                or (declared_counts <= 0).any()
                or declared_counts.sum() != SAMPLE_TILES
            ):
                raise AuditError("vocabulary: cluster-size census mismatch")
            predicted_counts, predicted_inertia = _nearest_centroid_summary(projection, centroids)
            count_delta = np.abs(predicted_counts - declared_counts)
            canonical = k == CANONICAL_K and seed == VOCAB_SEED
            if canonical and count_delta.sum() != 0:
                raise AuditError("canonical vocabulary: saved-centroid assignments drifted")
            if not canonical:
                # A converged training label can differ from predict at a single
                # float32 Voronoi-boundary row.  It is diagnostics-only for
                # nonoperative sensitivity dictionaries, never silently broad.
                if count_delta.sum() > 2 or count_delta.max(initial=0) > 1:
                    raise AuditError("sensitivity vocabulary: boundary discrepancy is excessive")
                noncanonical_boundary_rows += int(count_delta.sum() // 2)
            declared_inertia = float(config.get("inertia", math.nan))
            relative_inertia = abs(predicted_inertia - declared_inertia) / max(
                abs(declared_inertia), 1.0
            )
            maximum_relative_inertia_residual = max(
                maximum_relative_inertia_residual, relative_inertia
            )
            # sklearn's training inertia is accumulated in float32/OpenMP
            # chunks; the independent streaming reduction is float64 and may
            # differ by a few parts in 100,000 solely from reduction order.
            if not math.isfinite(declared_inertia) or relative_inertia > 3e-5:
                raise AuditError("vocabulary: saved-centroid inertia is inconsistent")
            if canonical:
                key = "reference" if fold is None else f"outer_fold_{fold}"
                canonical_backprojected[key] = _backprojected_centroids(centroids, mean, components)
                vocabulary_hashes[key] = sha256_file(vocabulary_path)

    inventory = pd.read_csv(root / "vocabularies/inventory.csv")
    if len(inventory) != 14 or set(inventory["sha256"].astype(str)) != {
        item.get("artifact", {}).get("sha256")
        for item in vocabulary_receipt.get("artifacts", {}).get("vocabularies", [])
    }:
        raise AuditError("vocabulary inventory: exact artifact census mismatch")

    mapping_receipt = receipts["mappings"]
    mapping_path = root / "mappings/outer_to_reference.csv"
    mapping = pd.read_csv(mapping_path)
    expected_mapping_blocks = []
    for fold in FOLDS:
        block = _recompute_mapping(
            canonical_backprojected[f"outer_fold_{fold}"],
            canonical_backprojected["reference"],
        )
        block.insert(0, "outer_fold", fold)
        expected_mapping_blocks.append(block)
    expected_mapping = pd.concat(expected_mapping_blocks, ignore_index=True)
    mapping_metadata = [
        "outer_fold",
        "source_prototype_id",
        "reference_prototype_id",
        "name_mappable",
    ]
    if not mapping[mapping_metadata].equals(expected_mapping[mapping_metadata]):
        raise AuditError("fold-to-reference mapping: discrete reconstruction failed")
    mapping_residual = _assert_numeric_close(
        mapping[["cosine_similarity"]].to_numpy(),
        expected_mapping[["cosine_similarity"]].to_numpy(),
        "fold-to-reference mapping",
        atol=5e-15,
    )
    if (
        mapping_receipt.get("threshold") != 0.80
        or mapping_receipt.get("all32_invariance")
        != "MAPPING_DOES_NOT_REINDEX_OR_DROP_TECHNICAL_PREDICTORS"
        or mapping_receipt.get("vocabulary_sha256") != vocabulary_hashes
    ):
        raise AuditError("fold-to-reference mapping: receipt dependency mismatch")
    for fold in FOLDS:
        block = mapping.loc[mapping["outer_fold"].eq(fold)]
        if (
            block["source_prototype_id"].astype(int).tolist() != list(range(CANONICAL_K))
            or sorted(block["reference_prototype_id"].astype(int)) != list(range(CANONICAL_K))
            or not np.array_equal(
                block["name_mappable"].astype(bool),
                block["cosine_similarity"].ge(0.80),
            )
        ):
            raise AuditError("fold-to-reference mapping: bijection or threshold mismatch")
    return {
        "sample_censuses": 6,
        "sampled_tiles": 6 * SAMPLE_TILES,
        "pca_bases": 6,
        "vocabularies": 14,
        "canonical_vocabularies_exact": 6,
        "mapping_rows": len(mapping),
        "maximum_mapping_cosine_residual": mapping_residual,
        "noncanonical_boundary_rows": noncanonical_boundary_rows,
        "maximum_pca_orthogonality_residual": maximum_orthogonality_residual,
        "maximum_relative_saved_centroid_inertia_residual": maximum_relative_inertia_residual,
    }


def _assignment_vocabulary_hashes(root: Path) -> dict[str, str]:
    return {
        "reference": sha256_file(_vocabulary_path(root, None, CANONICAL_K, VOCAB_SEED)),
        **{
            f"outer_fold_{fold}": sha256_file(_vocabulary_path(root, fold, CANONICAL_K, VOCAB_SEED))
            for fold in FOLDS
        },
    }


def _assignment_dependencies(root: Path, vocabularies: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pack_sha256": {name: digest for name, (digest, _size) in PINNED_PACK.items()},
        "source_roster_sha256": sha256_file(root / "inputs/source_roster_label_blind.csv"),
        "implementation_contract_sha256": sha256_file(root / "implementation_contract.json"),
        "vocabulary_sha256": dict(vocabularies),
        "implementation_sources": [
            _identity(RUNNER),
            _identity(REPOSITORY / "src/oceanpath/aim1/v14_concepts.py"),
            _identity(REPOSITORY / "src/oceanpath/datasets/packed.py"),
        ],
    }


def _expected_patient_profile(slide: pd.DataFrame, vocabulary_id: str) -> pd.DataFrame:
    patients = sorted(set(slide["patient_id"].astype(str)))
    values = slide[list(PROTOTYPE_COLUMNS)].to_numpy(dtype=np.float64)
    patient_ids = slide["patient_id"].astype(str).to_numpy()
    profiles = np.empty((len(patients), CANONICAL_K), dtype=np.float64)
    counts = np.empty(len(patients), dtype=np.int64)
    for index, patient in enumerate(patients):
        mask = patient_ids == patient
        profiles[index] = values[mask].mean(axis=0)
        counts[index] = int(mask.sum())
    metadata = (
        slide.sort_values(["patient_id", "slide_id"], kind="mergesort")
        .drop_duplicates("patient_id")
        .set_index("patient_id")
        .loc[patients]
    )
    result = pd.DataFrame(profiles, columns=PROTOTYPE_COLUMNS)
    result.insert(0, "n_slides", counts)
    result.insert(0, "fold", metadata["fold"].to_numpy(dtype=np.int64))
    result.insert(0, "subcohort", metadata["subcohort"].astype(str).to_numpy())
    result.insert(0, "patient_id", patients)
    result.insert(0, "vocabulary_id", vocabulary_id)
    return result


def _profile_residual(observed: pd.DataFrame, expected: pd.DataFrame, context: str) -> float:
    expected_columns = [
        "vocabulary_id",
        *(["slide_id"] if "slide_id" in expected else []),
        "patient_id",
        "subcohort",
        "fold",
        *(["n_tiles"] if "n_tiles" in expected else ["n_slides"]),
        *PROTOTYPE_COLUMNS,
    ]
    if observed.columns.tolist() != expected_columns:
        raise AuditError(f"{context}: schema mismatch")
    metadata_columns = [column for column in expected_columns if column not in PROTOTYPE_COLUMNS]
    if not observed[metadata_columns].equals(expected[metadata_columns]):
        raise AuditError(f"{context}: metadata reconstruction mismatch")
    return _assert_numeric_close(
        observed[list(PROTOTYPE_COLUMNS)].to_numpy(),
        expected[list(PROTOTYPE_COLUMNS)].to_numpy(),
        context,
        atol=0.0,
    )


def _audit_assignments_and_profiles(
    root: Path,
    roster: pd.DataFrame,
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    profile_receipt = receipts["profiles"]
    if (
        profile_receipt.get("source_slides") != SOURCE_SLIDES
        or profile_receipt.get("source_patients") != SOURCE_PATIENTS
        or profile_receipt.get("source_tiles") != SOURCE_TILES
        or profile_receipt.get("vocabularies_applied") != 6
        or not 1 <= int(profile_receipt.get("max_workers", 0)) <= 4
    ):
        raise AuditError("profiles receipt: census or worker contract mismatch")
    _validate_identity_tree(profile_receipt.get("artifacts", {}), "profiles receipt")
    manifest_path = root / "assignments/source_manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    expected_manifest_columns = [
        "slide_id",
        "path",
        "size_bytes",
        "mtime_ns",
        "sha256",
        "receipt_path",
        "receipt_size_bytes",
        "receipt_sha256",
    ]
    if manifest.columns.tolist() != expected_manifest_columns or len(manifest) != SOURCE_SLIDES:
        raise AuditError("assignment manifest: schema or census mismatch")
    if manifest["slide_id"].astype(str).tolist() != roster["slide_id"].astype(str).tolist():
        raise AuditError("assignment manifest: source slide order mismatch")
    shard_root = root / "assignments/source"
    actual_shards = sorted(shard_root.glob("*.npz"))
    actual_receipts = sorted(shard_root.glob("*.npz.receipt.json"))
    expected_shards = {f"{slide_id}.npz" for slide_id in roster["slide_id"].astype(str)}
    if {path.name for path in actual_shards} != expected_shards or {
        path.name for path in actual_receipts
    } != {f"{name}.receipt.json" for name in expected_shards}:
        raise AuditError("assignment shards: missing or orphan checkpoint")

    vocabularies = _assignment_vocabulary_hashes(root)
    dependencies = _assignment_dependencies(root, vocabularies)
    dependency_digest = hashlib.sha256(json_bytes(dependencies)).hexdigest()
    profile_arrays = {
        key: np.empty((SOURCE_SLIDES, CANONICAL_K), dtype=np.float64) for key in PROFILE_KEYS
    }
    distances_by_prototype: list[list[np.ndarray]] = [[] for _ in range(CANONICAL_K)]
    candidate_rows: list[dict[str, Any]] = []
    coords = np.memmap(PACK_ROOT / "coords.bin", dtype=np.int32, mode="r").reshape(-1, 2)
    total_tiles = 0
    for row_index, (roster_row, manifest_row) in enumerate(
        zip(roster.itertuples(index=False), manifest.itertuples(index=False), strict=True)
    ):
        slide_id = str(roster_row.slide_id)
        n_tiles = int(roster_row.n_tiles)
        path = shard_root / f"{slide_id}.npz"
        receipt_path = shard_root / f"{slide_id}.npz.receipt.json"
        observed = _identity(path)
        if (
            Path(str(manifest_row.path)).resolve(strict=False) != path.resolve()
            or observed["size_bytes"] != int(manifest_row.size_bytes)
            or observed["mtime_ns"] != int(manifest_row.mtime_ns)
            or observed["sha256"] != str(manifest_row.sha256)
        ):
            raise AuditError("assignment manifest: shard identity mismatch")
        receipt_identity = _identity(receipt_path)
        if (
            Path(str(manifest_row.receipt_path)).resolve(strict=False) != receipt_path.resolve()
            or receipt_identity["size_bytes"] != int(manifest_row.receipt_size_bytes)
            or receipt_identity["sha256"] != str(manifest_row.receipt_sha256)
        ):
            raise AuditError("assignment manifest: shard receipt identity mismatch")
        receipt = _load_json(receipt_path, "assignment shard receipt")
        if (
            receipt.get("status") != "PASS"
            or receipt.get("slide_id") != slide_id
            or receipt.get("n_tiles") != n_tiles
            or receipt.get("dependencies") != dependencies
            or receipt.get("dependency_contract_sha256") != dependency_digest
            or receipt.get("artifact") != observed
        ):
            raise AuditError("assignment shard receipt: contract mismatch")
        try:
            with np.load(path, allow_pickle=False) as bundle:
                expected_arrays = {
                    "reference_labels",
                    "reference_distances",
                    *(f"outer_fold_{fold}_labels" for fold in FOLDS),
                    "metadata_json",
                }
                if set(bundle.files) != expected_arrays:
                    raise AuditError("assignment shard: archive schema mismatch")
                metadata = _decode_json_array(bundle["metadata_json"], "assignment shard")
                if (
                    metadata.get("slide_id") != slide_id
                    or metadata.get("n_tiles") != n_tiles
                    or metadata.get("vocabulary_sha256") != vocabularies
                    or metadata.get("dependency_contract_sha256") != dependency_digest
                    or metadata.get("tile_order") != "zero-based packed-store slide-local row"
                ):
                    raise AuditError("assignment shard: embedded contract mismatch")
                loaded_labels: dict[str, np.ndarray] = {}
                for key in PROFILE_KEYS:
                    labels = np.asarray(bundle[f"{key}_labels"])
                    if (
                        labels.shape != (n_tiles,)
                        or not np.issubdtype(labels.dtype, np.integer)
                        or (labels < 0).any()
                        or (labels >= CANONICAL_K).any()
                    ):
                        raise AuditError("assignment shard: invalid label vector")
                    loaded_labels[key] = labels.astype(np.int16, copy=False)
                    profile_arrays[key][row_index] = (
                        np.bincount(labels.astype(np.int64), minlength=CANONICAL_K).astype(
                            np.float64
                        )
                        / n_tiles
                    )
                distances = np.asarray(bundle["reference_distances"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as exc:
            raise AuditError("assignment shard: unreadable archive") from exc
        if (
            distances.shape != (n_tiles,)
            or not np.isfinite(distances).all()
            or (distances < 0).any()
        ):
            raise AuditError("assignment shard: invalid reference distance vector")
        reference_labels = loaded_labels["reference"]
        for prototype in range(CANONICAL_K):
            mask = reference_labels == prototype
            if mask.any():
                distances_by_prototype[prototype].append(distances[mask].copy())
        for prototype in np.unique(reference_labels):
            eligible = np.flatnonzero(reference_labels == prototype)
            tile_index = int(eligible[int(np.argmin(distances[eligible]))])
            coordinate = coords[int(roster_row.global_offset) + tile_index]
            candidate_rows.append(
                {
                    "prototype_id": int(prototype),
                    "patient_id": str(roster_row.patient_id),
                    "subcohort": str(roster_row.subcohort),
                    "slide_id": slide_id,
                    "tile_id": f"{tile_index:012d}",
                    "tile_index": tile_index,
                    "x": int(coordinate[0]),
                    "y": int(coordinate[1]),
                    "distance": float(distances[tile_index]),
                }
            )
        total_tiles += n_tiles
    del coords
    if total_tiles != SOURCE_TILES:
        raise AuditError("assignment shards: aggregate tile census mismatch")

    maximum_profile_residual = 0.0
    patient_frames: dict[str, pd.DataFrame] = {}
    for key in PROFILE_KEYS:
        if not np.allclose(profile_arrays[key].sum(axis=1), 1.0, rtol=0, atol=1e-12):
            raise AuditError("slide profiles: probability mass mismatch")
        expected_slide = roster[["slide_id", "patient_id", "subcohort", "fold", "n_tiles"]].copy()
        expected_slide.insert(0, "vocabulary_id", key)
        for index, column in enumerate(PROTOTYPE_COLUMNS):
            expected_slide[column] = profile_arrays[key][:, index]
        observed_slide = pd.read_parquet(root / f"profiles/{key}/slide_profiles.parquet")
        maximum_profile_residual = max(
            maximum_profile_residual,
            _profile_residual(observed_slide, expected_slide, "slide profiles"),
        )
        expected_patient = _expected_patient_profile(expected_slide, key)
        observed_patient = pd.read_parquet(root / f"profiles/{key}/patient_profiles.parquet")
        maximum_profile_residual = max(
            maximum_profile_residual,
            _profile_residual(observed_patient, expected_patient, "patient profiles"),
        )
        patient_frames[key] = expected_patient
    expected_oof = (
        pd.concat(
            [
                patient_frames[f"outer_fold_{fold}"].loc[
                    patient_frames[f"outer_fold_{fold}"]["fold"].eq(fold)
                ]
                for fold in FOLDS
            ],
            ignore_index=True,
        )
        .sort_values("patient_id", kind="mergesort")
        .reset_index(drop=True)
    )
    observed_oof = pd.read_parquet(root / "profiles/oof_patient_profiles.parquet")
    maximum_profile_residual = max(
        maximum_profile_residual,
        _profile_residual(observed_oof, expected_oof, "OOF patient profiles"),
    )

    candidates = (
        pd.DataFrame.from_records(candidate_rows)
        .sort_values(
            ["prototype_id", "patient_id", "distance", "slide_id", "tile_id"],
            kind="mergesort",
        )
        .drop_duplicates(["prototype_id", "patient_id"], keep="first")
    )
    candidate_blocks = []
    quantile_rows = []
    support_rows = []
    for prototype in range(CANONICAL_K):
        values = np.concatenate(distances_by_prototype[prototype])
        q10, q25, q99 = np.quantile(values, [0.10, 0.25, 0.99], method="linear")
        block = candidates.loc[candidates["prototype_id"].eq(prototype)].copy()
        block["prototype_q10_distance"] = float(q10)
        block["prototype_q25_distance"] = float(q25)
        block["eligibility_tier"] = np.select(
            [block["distance"] <= q10, block["distance"] <= q25],
            [0, 1],
            default=2,
        ).astype(np.int8)
        block["eligibility_stage"] = block["eligibility_tier"].map(
            {0: "decile", 1: "quartile", 2: "all"}
        )
        candidate_blocks.append(block)
        quantile_rows.append(
            {
                "prototype_id": prototype,
                "n_assigned_tiles": len(values),
                "q10_distance": float(q10),
                "q25_distance": float(q25),
                "q99_distance": float(q99),
            }
        )
        support = len(block)
        support_rows.append(
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
    expected_candidates = pd.concat(candidate_blocks, ignore_index=True)
    observed_candidates = pd.read_parquet(root / "montage/canonical_candidates.parquet")
    _assert_frame_exact(observed_candidates, expected_candidates, "canonical montage candidates")
    expected_quantiles = pd.DataFrame.from_records(quantile_rows)
    observed_quantiles = pd.read_csv(root / "profiles/reference_distance_quantiles.csv")
    if not observed_quantiles[["prototype_id", "n_assigned_tiles"]].equals(
        expected_quantiles[["prototype_id", "n_assigned_tiles"]]
    ):
        raise AuditError("reference distance quantiles: discrete reconstruction failed")
    _assert_numeric_close(
        observed_quantiles[["q10_distance", "q25_distance", "q99_distance"]].to_numpy(),
        expected_quantiles[["q10_distance", "q25_distance", "q99_distance"]].to_numpy(),
        "reference distance quantiles",
        atol=5e-15,
    )
    expected_support = pd.DataFrame.from_records(support_rows)
    observed_support = pd.read_csv(root / "montage/support_roster.csv")
    _assert_frame_exact(observed_support, expected_support, "montage support roster")
    return {
        "assignment_shards": SOURCE_SLIDES,
        "assigned_tiles": total_tiles,
        "slide_profile_matrices_reconstructed": 6,
        "patient_profile_matrices_reconstructed": 6,
        "oof_patients": len(expected_oof),
        "canonical_candidates": len(expected_candidates),
        "maximum_profile_residual": maximum_profile_residual,
    }


def _equal_slide_attention(
    slide_attention: pd.DataFrame,
    roster: pd.DataFrame,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    patients = sorted(set(slide_attention["patient_id"].astype(str)))
    ids = slide_attention["patient_id"].astype(str).to_numpy()
    values = slide_attention[list(PROTOTYPE_COLUMNS)].to_numpy(dtype=np.float64)
    profiles = np.empty((len(patients), CANONICAL_K), dtype=np.float64)
    counts = np.empty(len(patients), dtype=np.int64)
    for index, patient in enumerate(patients):
        mask = ids == patient
        profiles[index] = values[mask].mean(axis=0)
        counts[index] = int(mask.sum())
    metadata = (
        roster.loc[roster["fold"].eq(fold)]
        .sort_values(["patient_id", "slide_id"], kind="mergesort")
        .drop_duplicates("patient_id")
        .set_index("patient_id")
        .loc[patients]
    )
    result = pd.DataFrame(profiles, columns=PROTOTYPE_COLUMNS)
    result.insert(0, "n_slides", counts)
    result.insert(0, "fold", fold)
    result.insert(0, "seed", seed)
    result.insert(0, "subcohort", metadata["subcohort"].astype(str).to_numpy())
    result.insert(0, "patient_id", patients)
    return result


def _assert_table_numeric_exact(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    numeric_columns: Sequence[str],
    context: str,
) -> float:
    if observed.columns.tolist() != expected.columns.tolist() or len(observed) != len(expected):
        raise AuditError(f"{context}: schema or row census mismatch")
    metadata = [column for column in expected if column not in numeric_columns]
    if (
        not observed[metadata]
        .reset_index(drop=True)
        .equals(expected[metadata].reset_index(drop=True))
    ):
        raise AuditError(f"{context}: metadata reconstruction mismatch")
    return _assert_numeric_close(
        observed[list(numeric_columns)].to_numpy(),
        expected[list(numeric_columns)].to_numpy(),
        context,
        atol=0.0,
    )


def _audit_teachers(
    root: Path,
    roster: pd.DataFrame,
    preflight: Mapping[str, Any],
    teacher_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if teacher_receipt.get("head_census") != 25:
        raise AuditError("teacher receipt: head census mismatch")
    inference = teacher_receipt.get("inference_contract", {})
    required_inference = {
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
    }
    if any(inference.get(key) != value for key, value in required_inference.items()):
        raise AuditError("teacher receipt: inference contract mismatch")
    execution_path = root / "teachers/execution_contract.json"
    execution_identity = _identity(execution_path)
    if teacher_receipt.get("execution_contract") != execution_identity:
        raise AuditError("teacher receipt: execution-contract identity mismatch")
    execution = _load_json(execution_path, "teacher execution contract")
    if (
        execution.get("status") != "FROZEN_BEFORE_HEAD_1"
        or execution.get("inference_contract", {}).get("device") != "cuda"
        or execution.get("inference_contract", {}).get("autocast") != "bfloat16"
        or execution.get("inference_contract", {}).get("full_bags") is not True
        or execution.get("inference_contract", {}).get("instance_cap") is not None
        or execution.get("checkpoint_dependencies") != preflight.get("teachers")
        or execution.get("assignment_vocabulary_sha256") != _assignment_vocabulary_hashes(root)
    ):
        raise AuditError("teacher execution contract: frozen dependency mismatch")
    # The 49.7-GB pack was independently deep-hashed by the core gate.  Avoid
    # hashing it again through the nested execution-contract copy while still
    # replaying every other material binding here.
    _validate_identity_tree(
        execution.get("implementation_sources", []),
        "teacher execution implementation sources",
    )
    for name in ("profile_receipt", "assignment_manifest", "v13_patient_logits"):
        _validate_identity(execution.get(name, {}), f"teacher execution contract {name}")

    listed_head_receipts = teacher_receipt.get("head_receipts", [])
    if len(listed_head_receipts) != 25:
        raise AuditError("teacher receipt: head receipt census mismatch")
    listed_lookup: dict[tuple[int, int], Mapping[str, Any]] = {}
    for record in listed_head_receipts:
        match = re.search(r"seed(\d+)/fold_(\d+)/receipt\.json$", str(record.get("path", "")))
        if match is None:
            raise AuditError("teacher receipt: malformed head receipt path")
        key = (int(match.group(1)), int(match.group(2)))
        if key in listed_lookup:
            raise AuditError("teacher receipt: duplicate head receipt")
        listed_lookup[key] = record
    expected_grid = {(seed, fold) for seed in MODEL_SEEDS for fold in FOLDS}
    if set(listed_lookup) != expected_grid:
        raise AuditError("teacher receipt: incomplete seed/fold grid")
    checkpoint_lookup = {
        (int(item["seed"]), int(item["fold"])): item["checkpoint"]
        for item in preflight.get("teachers", {}).get("heads", [])
    }

    sealed_columns = [
        "arm",
        "patient_id",
        "k_fold",
        *(f"logit_seed{seed}" for seed in MODEL_SEEDS),
        "mean_logit_5seed",
    ]
    sealed = pd.read_parquet(V13_LOGITS, columns=sealed_columns)
    sealed = sealed.loc[sealed["arm"].eq("tcga_surgen_primary")].copy()
    sealed["patient_id"] = sealed["patient_id"].astype(str)
    sealed = sealed.sort_values("patient_id", kind="mergesort").reset_index(drop=True)
    if len(sealed) != SOURCE_PATIENTS or sealed["patient_id"].duplicated().any():
        raise AuditError("sealed FINAL-v13 logits: patient census mismatch")
    patient_metadata = roster.sort_values(
        ["patient_id", "slide_id"], kind="mergesort"
    ).drop_duplicates("patient_id")[["patient_id", "subcohort", "fold"]]
    roster_by_slide = roster.set_index("slide_id")
    seed_logit_frames: list[pd.DataFrame] = []
    seed_attention_frames: list[pd.DataFrame] = []
    maximum_logit_residual = 0.0
    maximum_attention_residual = 0.0
    maximum_attention_mass_residual = 0.0
    expected_model_schema = {
        "arch": "abmil",
        "in_dim": FEATURE_DIM,
        "num_classes": 1,
        "loss_type": "bce",
        "compile_model": False,
    }
    for seed in MODEL_SEEDS:
        heldout_logit_blocks: list[pd.DataFrame] = []
        heldout_attention_blocks: list[pd.DataFrame] = []
        for fold in FOLDS:
            directory = root / f"teachers/seed{seed}/fold_{fold}"
            head_receipt_path = directory / "receipt.json"
            listed = listed_lookup[(seed, fold)]
            _validate_identity(listed, "teacher head receipt", expected_path=head_receipt_path)
            head = _load_json(head_receipt_path, "teacher head receipt")
            if (
                head.get("status") != "PASS"
                or head.get("seed") != seed
                or head.get("fold") != fold
                or head.get("checkpoint") != checkpoint_lookup[(seed, fold)]
                or head.get("execution_contract") != execution_identity
                or head.get("assignment_vocabulary_sha256") != _assignment_vocabulary_hashes(root)
                or float(head.get("tolerance", math.nan)) != 1e-6
            ):
                raise AuditError("teacher head receipt: frozen contract mismatch")
            _validate_identity_tree(head.get("artifacts", {}), "teacher head artifacts")
            model_schema = head.get("model_schema", {})
            hparams = model_schema.get("hparams", {})
            if (
                any(hparams.get(key) != value for key, value in expected_model_schema.items())
                or hparams.get("model_cfg", {}).get("embed_dim") != 512
                or hparams.get("model_cfg", {}).get("attn_dim") != 384
                or hparams.get("model_cfg", {}).get("gate") is not True
                or hparams.get("model_cfg", {}).get("dropout") != 0.25
                or hparams.get("model_cfg", {}).get("input_dropout") != 0.10
                or model_schema.get("parameter_count") != 919_682
                or len(model_schema.get("state_keys", [])) != 10
            ):
                raise AuditError("teacher head receipt: model schema mismatch")

            slide = pd.read_parquet(directory / "slide_logits.parquet")
            if (
                slide.columns.tolist()
                != [
                    "slide_id",
                    "patient_id",
                    "subcohort",
                    "patient_fold",
                    "seed",
                    "head_fold",
                    "logit",
                ]
                or len(slide) != SOURCE_SLIDES
            ):
                raise AuditError("teacher slide logits: schema or census mismatch")
            if slide["slide_id"].astype(str).duplicated().any() or set(
                slide["slide_id"].astype(str)
            ) != set(roster_by_slide.index.astype(str)):
                raise AuditError("teacher slide logits: source roster mismatch")
            expected_slide_metadata = roster_by_slide.loc[slide["slide_id"].astype(str)]
            if (
                not np.array_equal(
                    slide["patient_id"].astype(str),
                    expected_slide_metadata["patient_id"].astype(str),
                )
                or not np.array_equal(
                    slide["subcohort"].astype(str), expected_slide_metadata["subcohort"].astype(str)
                )
                or not np.array_equal(
                    slide["patient_fold"].astype(int), expected_slide_metadata["fold"].astype(int)
                )
                or not slide["seed"].eq(seed).all()
                or not slide["head_fold"].eq(fold).all()
                or not np.isfinite(slide["logit"].to_numpy(dtype=np.float64)).all()
            ):
                raise AuditError("teacher slide logits: metadata or numeric mismatch")
            expected_patient = slide.groupby("patient_id", sort=True, as_index=False)[
                "logit"
            ].mean()
            expected_patient = patient_metadata.merge(
                expected_patient, on="patient_id", validate="one_to_one"
            )
            expected_patient.insert(3, "seed", seed)
            expected_patient.insert(4, "head_fold", fold)
            observed_patient = pd.read_parquet(directory / "patient_logits.parquet")
            maximum_logit_residual = max(
                maximum_logit_residual,
                _assert_table_numeric_exact(
                    observed_patient, expected_patient, ["logit"], "teacher patient logits"
                ),
            )
            heldout = expected_patient.loc[
                expected_patient["fold"].eq(fold), ["patient_id", "fold", "logit"]
            ].copy()
            expected_sealed = sealed.loc[
                sealed["k_fold"].eq(fold), ["patient_id", f"logit_seed{seed}"]
            ]
            comparison = expected_sealed.merge(
                heldout[["patient_id", "logit"]],
                on="patient_id",
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
            if len(comparison) != FOLD_PATIENTS[fold] or not comparison["_merge"].eq("both").all():
                raise AuditError("teacher heldout logits: patient roster mismatch")
            residual = float(
                np.max(
                    np.abs(
                        comparison[f"logit_seed{seed}"].to_numpy(dtype=np.float64)
                        - comparison["logit"].to_numpy(dtype=np.float64)
                    ),
                    initial=0.0,
                )
            )
            maximum_logit_residual = max(maximum_logit_residual, residual)
            if (
                residual > 1e-6
                or float(head.get("maximum_absolute_v13_logit_delta", math.inf)) != residual
            ):
                raise AuditError("teacher heldout logits: sealed FINAL-v13 replay mismatch")
            heldout_logit_blocks.append(heldout)

            attention_slide = pd.read_parquet(directory / "heldout_attention_slides.parquet")
            expected_attention_columns = [
                "slide_id",
                "patient_id",
                "subcohort",
                "seed",
                "fold",
                "n_tiles",
                *PROTOTYPE_COLUMNS,
            ]
            heldout_roster = roster.loc[roster["fold"].eq(fold)].set_index("slide_id")
            if (
                attention_slide.columns.tolist() != expected_attention_columns
                or len(attention_slide) != len(heldout_roster)
                or attention_slide["slide_id"].astype(str).duplicated().any()
                or set(attention_slide["slide_id"].astype(str))
                != set(heldout_roster.index.astype(str))
                or not attention_slide["seed"].eq(seed).all()
                or not attention_slide["fold"].eq(fold).all()
            ):
                raise AuditError("teacher slide attention: schema, census, or fold mismatch")
            expected_attention_metadata = heldout_roster.loc[
                attention_slide["slide_id"].astype(str)
            ]
            if (
                not np.array_equal(
                    attention_slide["patient_id"].astype(str),
                    expected_attention_metadata["patient_id"].astype(str),
                )
                or not np.array_equal(
                    attention_slide["subcohort"].astype(str),
                    expected_attention_metadata["subcohort"].astype(str),
                )
                or not np.array_equal(
                    attention_slide["n_tiles"].astype(int),
                    expected_attention_metadata["n_tiles"].astype(int),
                )
            ):
                raise AuditError("teacher slide attention: source metadata mismatch")
            masses = attention_slide[list(PROTOTYPE_COLUMNS)].to_numpy(dtype=np.float64)
            if not np.isfinite(masses).all() or (masses < 0).any():
                raise AuditError("teacher slide attention: invalid probability mass")
            maximum_attention_mass_residual = max(
                maximum_attention_mass_residual,
                float(np.max(np.abs(masses.sum(axis=1) - 1.0), initial=0.0)),
            )
            if maximum_attention_mass_residual > 2e-6:
                raise AuditError("teacher slide attention: probability mass exceeds tolerance")
            expected_attention_patient = _equal_slide_attention(attention_slide, roster, seed, fold)
            observed_attention_patient = pd.read_parquet(
                directory / "heldout_attention_patients.parquet"
            )
            maximum_attention_residual = max(
                maximum_attention_residual,
                _assert_table_numeric_exact(
                    observed_attention_patient,
                    expected_attention_patient,
                    PROTOTYPE_COLUMNS,
                    "teacher patient attention",
                ),
            )
            heldout_attention_blocks.append(expected_attention_patient)

        seed_logits = pd.concat(heldout_logit_blocks, ignore_index=True).sort_values(
            "patient_id", kind="mergesort"
        )
        if len(seed_logits) != SOURCE_PATIENTS or seed_logits["patient_id"].duplicated().any():
            raise AuditError("teacher seed OOF logits: patient census mismatch")
        seed_logits = seed_logits.rename(columns={"logit": f"logit_seed{seed}"})
        seed_logit_frames.append(seed_logits[["patient_id", f"logit_seed{seed}"]])
        seed_attention = pd.concat(heldout_attention_blocks, ignore_index=True).sort_values(
            "patient_id", kind="mergesort"
        )
        observed_seed_attention = pd.read_parquet(
            root / f"teachers/seed{seed}_oof_attention_patients.parquet"
        )
        maximum_attention_residual = max(
            maximum_attention_residual,
            _assert_table_numeric_exact(
                observed_seed_attention,
                seed_attention,
                PROTOTYPE_COLUMNS,
                "teacher seed OOF attention",
            ),
        )
        seed_attention_frames.append(seed_attention)

    expected_native = seed_logit_frames[0]
    for frame in seed_logit_frames[1:]:
        expected_native = expected_native.merge(frame, on="patient_id", validate="one_to_one")
    logit_columns = [f"logit_seed{seed}" for seed in MODEL_SEEDS]
    expected_native["mean_logit_5seed"] = expected_native[logit_columns].mean(axis=1)
    observed_native = pd.read_parquet(root / "teachers/oof_native_logits_recomputed.parquet")
    maximum_logit_residual = max(
        maximum_logit_residual,
        _assert_table_numeric_exact(
            observed_native,
            expected_native,
            [*logit_columns, "mean_logit_5seed"],
            "teacher OOF native logits",
        ),
    )
    sealed_comparison = sealed.merge(
        expected_native, on="patient_id", validate="one_to_one", suffixes=("_sealed", "_recomputed")
    )
    mean_residual = float(
        np.max(
            np.abs(
                sealed_comparison["mean_logit_5seed_sealed"].to_numpy(dtype=np.float64)
                - sealed_comparison["mean_logit_5seed_recomputed"].to_numpy(dtype=np.float64)
            ),
            initial=0.0,
        )
    )
    maximum_logit_residual = max(maximum_logit_residual, mean_residual)
    if mean_residual > 1e-6:
        raise AuditError("teacher five-seed logits: sealed mean replay mismatch")

    stacked_attention = pd.concat(seed_attention_frames, ignore_index=True)
    if not stacked_attention.groupby("patient_id").size().eq(5).all():
        raise AuditError("teacher five-seed attention: partial seed set")
    expected_attention_mean = stacked_attention.groupby("patient_id", sort=True, as_index=False)[
        list(PROTOTYPE_COLUMNS)
    ].mean()
    expected_attention_mean = patient_metadata.merge(
        expected_attention_mean, on="patient_id", validate="one_to_one"
    )
    observed_attention_mean = pd.read_parquet(
        root / "teachers/oof_attention_patients_5seed_mean.parquet"
    )
    maximum_attention_residual = max(
        maximum_attention_residual,
        _assert_table_numeric_exact(
            observed_attention_mean,
            expected_attention_mean,
            PROTOTYPE_COLUMNS,
            "teacher five-seed attention mean",
        ),
    )
    final_mass = observed_attention_mean[list(PROTOTYPE_COLUMNS)].sum(axis=1).to_numpy()
    maximum_attention_mass_residual = max(
        maximum_attention_mass_residual,
        float(np.max(np.abs(final_mass - 1.0), initial=0.0)),
    )
    if maximum_attention_mass_residual > 2e-6:
        raise AuditError("teacher aggregate attention: probability mass exceeds tolerance")
    _validate_identity_tree(teacher_receipt.get("artifacts", {}), "teacher aggregate artifacts")
    if (
        float(teacher_receipt.get("checks", {}).get("maximum_absolute_mean_logit_delta", math.inf))
        != mean_residual
    ):
        raise AuditError("teacher receipt: aggregate logit check mismatch")
    return {
        "heads": 25,
        "slides_scored_per_head": SOURCE_SLIDES,
        "oof_patients": SOURCE_PATIENTS,
        "maximum_absolute_sealed_logit_residual": maximum_logit_residual,
        "maximum_attention_aggregation_residual": maximum_attention_residual,
        "maximum_attention_mass_residual": maximum_attention_mass_residual,
    }


def _montage_quotas(
    capacities: Mapping[str, int], total: int, existing: Mapping[str, int] | None = None
) -> dict[str, int]:
    allocated = {group: 0 for group in SOURCE_SUBCOHORTS}
    current = {group: int((existing or {}).get(group, 0)) for group in SOURCE_SUBCOHORTS}
    remaining = min(total, sum(int(capacities.get(group, 0)) for group in SOURCE_SUBCOHORTS))
    for group in SOURCE_SUBCOHORTS:
        add = min(max(0, 3 - current[group]), int(capacities.get(group, 0)), remaining)
        allocated[group] += add
        remaining -= add
    while remaining:
        available = {
            group: int(capacities.get(group, 0)) - allocated[group]
            for group in SOURCE_SUBCOHORTS
            if allocated[group] < int(capacities.get(group, 0))
        }
        largest = max(available.values())
        group = next(group for group in SOURCE_SUBCOHORTS if available.get(group) == largest)
        allocated[group] += 1
        remaining -= 1
    return allocated


def _sample_candidate_rows(
    pool: pd.DataFrame,
    quotas: Mapping[str, int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    blocks = []
    for group in SOURCE_SUBCOHORTS:
        count = int(quotas.get(group, 0))
        if not count:
            continue
        candidates = pool.loc[pool["subcohort"].eq(group)].sort_values(
            ["distance", "patient_id", "slide_id", "tile_id"], kind="stable"
        )
        positions = np.asarray(rng.choice(len(candidates), size=count, replace=False))
        blocks.append(candidates.iloc[positions].copy())
    chosen = pd.concat(blocks, ignore_index=True)
    return chosen.iloc[np.asarray(rng.permutation(len(chosen)), dtype=np.int64)].reset_index(
        drop=True
    )


def _select_montage(
    candidates: pd.DataFrame,
    prototype: int,
    occurrence: int,
    avoid: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    avoid = avoid or set()
    chosen_tier = 2
    for tier in range(3):
        pool = candidates.loc[candidates["eligibility_tier"].le(tier)]
        available = pool.loc[~pool["patient_id"].astype(str).isin(avoid)]
        if len(available) >= 12:
            chosen_tier = tier
            break
    pool = candidates.loc[candidates["eligibility_tier"].le(chosen_tier)].copy()
    target = min(12, len(pool))
    nonoverlap = pool.loc[~pool["patient_id"].astype(str).isin(avoid)].copy()
    overlap = pool.loc[pool["patient_id"].astype(str).isin(avoid)].copy()
    seed = VOCAB_SEED + 100 * prototype + occurrence
    rng = np.random.Generator(np.random.PCG64(seed))
    if len(nonoverlap) >= target:
        quotas = _montage_quotas(nonoverlap["subcohort"].value_counts().to_dict(), target)
        selected = _sample_candidate_rows(nonoverlap, quotas, rng)
    else:
        selected_nonoverlap = nonoverlap.copy()
        current = (
            selected_nonoverlap["subcohort"]
            .value_counts()
            .reindex(SOURCE_SUBCOHORTS, fill_value=0)
            .astype(int)
            .to_dict()
        )
        overlap_quotas = _montage_quotas(
            overlap["subcohort"].value_counts().to_dict(),
            target - len(selected_nonoverlap),
            current,
        )
        selected_overlap = _sample_candidate_rows(overlap, overlap_quotas, rng)
        selected = pd.concat([selected_nonoverlap, selected_overlap], ignore_index=True)
        selected = selected.iloc[
            np.asarray(rng.permutation(len(selected)), dtype=np.int64)
        ].reset_index(drop=True)
    selected["montage_slot"] = np.arange(len(selected), dtype=np.int64)
    selected["prototype_index"] = prototype
    selected["occurrence"] = occurrence
    selected["selection_seed"] = seed
    selected["selection_stage"] = ("decile", "quartile", "all")[chosen_tier]
    selected["overlaps_avoidance_set"] = selected["patient_id"].astype(str).isin(avoid)
    overlap_count = int(selected["overlaps_avoidance_set"].sum())
    audit = {
        "selection_seed": seed,
        "selection_stage": ("decile", "quartile", "all")[chosen_tier],
        "support_status": (
            "MONTAGE_SUPPORT_SUFFICIENT"
            if len(candidates) >= 12
            else "MONTAGE_SUPPORT_INSUFFICIENT"
        ),
        "n_available_patients": len(candidates),
        "n_eligible_patients": len(pool),
        "n_displayed_tiles": len(selected),
        "patient_overlap_with_controlling_read": overlap_count,
    }
    return selected, audit


def _hmac_blinding_table(occurrences: Sequence[tuple[int, int]], salt: bytes) -> pd.DataFrame:
    used: set[str] = set()
    records: list[dict[str, Any]] = []
    for prototype, occurrence in sorted(occurrences):
        code_digest = hmac.new(
            salt, f"code|{prototype}|{occurrence}".encode(), hashlib.sha256
        ).digest()
        encoded = base64.b32encode(code_digest).decode("ascii").rstrip("=")
        code = ""
        window_index = -1
        for candidate_window in range(8):
            candidate = encoded[6 * candidate_window : 6 * candidate_window + 6]
            if len(candidate) == 6 and candidate not in used:
                code, window_index = candidate, candidate_window
                break
        if not code:
            raise AuditError("reader blinding: HMAC code windows exhausted")
        used.add(code)
        order_digest = hmac.new(
            salt, f"order|{prototype}|{occurrence}".encode(), hashlib.sha256
        ).digest()
        records.append(
            {
                "prototype_id": prototype,
                "occurrence": occurrence,
                "code": code,
                "code_window_index": window_index,
                "order_digest_hex": order_digest.hex(),
                "is_controlling_read": occurrence == 0,
                "_order": order_digest,
            }
        )
    records.sort(key=lambda row: (row["_order"], row["prototype_id"], row["occurrence"]))
    for position, record in enumerate(records):
        record["presentation_order"] = position
        del record["_order"]
    columns = [
        "presentation_order",
        "prototype_id",
        "occurrence",
        "code",
        "code_window_index",
        "is_controlling_read",
        "order_digest_hex",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _inventory_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AuditError("public package: symlink is forbidden")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def _audit_public_directory(public: Path, roster: pd.DataFrame) -> dict[str, Any]:
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
        raise AuditError("public package: top-level census mismatch")
    manifest_path = public / "HANDOFF_MANIFEST.json"
    _require_sha(manifest_path, PINNED_HANDOFF_MANIFEST_SHA256, "public handoff manifest")
    manifest = _load_json(manifest_path, "public handoff manifest")
    public_salt = (public / "PUBLIC_SALT_SHA256.txt").read_text(encoding="ascii")
    if not re.fullmatch(r"[0-9a-f]{64}\n", public_salt):
        raise AuditError("public package: salt digest format mismatch")
    try:
        sealed_time = dt.datetime.fromisoformat(str(manifest.get("sealed_utc", "")))
    except ValueError as exc:
        raise AuditError("public package: malformed seal timestamp") from exc
    if (
        set(manifest)
        != {
            "schema_version",
            "package",
            "sealed_utc",
            "n_montages",
            "public_salt_sha256",
            "files",
        }
        or manifest.get("schema_version") != 1
        or manifest.get("package") != "FINAL-v14 blinded morphology naming session"
        or manifest.get("n_montages") != 40
        or manifest.get("public_salt_sha256") != public_salt.strip()
        or sealed_time.tzinfo is None
    ):
        raise AuditError("public package: manifest contract mismatch")
    sidecar = (public / "HANDOFF_MANIFEST.sha256").read_text(encoding="ascii")
    if sidecar != f"{PINNED_HANDOFF_MANIFEST_SHA256}  HANDOFF_MANIFEST.json\n":
        raise AuditError("public package: manifest sidecar mismatch")
    form = pd.read_csv(public / "review_form.csv", keep_default_na=False)
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
    expected_columns = ["presentation_order", "blinded_code", "n_tiles", *response_columns]
    if (
        form.columns.tolist() != expected_columns
        or len(form) != 40
        or form["presentation_order"].tolist() != list(range(1, 41))
        or not form["blinded_code"].astype(str).str.fullmatch(r"[A-Z2-7]{6}").all()
        or not form["blinded_code"].is_unique
        or not form["n_tiles"].eq(12).all()
        or any(not form[column].eq("").all() for column in response_columns)
    ):
        raise AuditError("public package: blank review form contract mismatch")
    montage_paths = sorted((public / "montages").glob("*.jpg"))
    if len(montage_paths) != 40 or {path.stem for path in montage_paths} != set(
        form["blinded_code"].astype(str)
    ):
        raise AuditError("public package: montage/form code census mismatch")
    for path in montage_paths:
        _regular_file(path, "public montage")
        try:
            with Image.open(path) as image:
                image.load()
                if (
                    image.format != "JPEG"
                    or image.mode != "RGB"
                    or image.size != (1024, 768)
                    or len(image.getexif()) != 0
                    or any(name in image.info for name in ("exif", "icc_profile", "comment"))
                ):
                    raise AuditError("public montage: image contract mismatch")
        except (OSError, ValueError) as exc:
            raise AuditError("public montage: unreadable JPEG") from exc
    actual_files = {
        path.relative_to(public).as_posix() for path in public.rglob("*") if path.is_file()
    }
    expected_files = {
        "INSTRUCTIONS.md",
        "RUBRIC.md",
        "review_form.csv",
        "PUBLIC_SALT_SHA256.txt",
        "HANDOFF_MANIFEST.json",
        "HANDOFF_MANIFEST.sha256",
        *(f"montages/{code}.jpg" for code in form["blinded_code"].astype(str)),
    }
    if actual_files != expected_files:
        raise AuditError("public package: missing or orphan file")
    manifest_records = manifest.get("files", [])
    if {str(record.get("path")) for record in manifest_records} != expected_files - {
        "HANDOFF_MANIFEST.json",
        "HANDOFF_MANIFEST.sha256",
    }:
        raise AuditError("public package: manifest file census mismatch")
    for record in manifest_records:
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise AuditError("public package: unsafe manifest path")
        _validate_identity(
            {**record, "path": str(public / relative)},
            "public package manifest",
            expected_path=public / relative,
        )
    text = "\n".join(
        (public / name).read_text(encoding="utf-8")
        for name in ("INSTRUCTIONS.md", "RUBRIC.md", "review_form.csv")
    )
    required_literals = {
        "approximately two-hour session",
        "set `review_status` to exactly `complete`",
        "up to two distinct secondary categories",
        "integer from 1 (very low) to 5 (very high)",
        "set `artifact_uninterpretable` to exactly `yes` or `no`",
        "confirmed_no_key_access",
        "Return only `completed_review_form.csv`",
    }
    if any(literal not in text for literal in required_literals):
        raise AuditError("public package: reader task instructions are incomplete")
    forbidden = {
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
    if any(term.casefold() in lowered for term in forbidden):
        raise AuditError("public package: forbidden technical identifier or term leaked")
    known_ids = {
        str(value)
        for column in ("patient_id", "slide_id")
        for value in roster[column]
        if len(str(value)) >= 8
    }
    if any(identifier in text for identifier in known_ids):
        raise AuditError("public package: known source identifier leaked")
    return {
        "manifest_sha256": PINNED_HANDOFF_MANIFEST_SHA256,
        "form_rows": len(form),
        "montages": len(montage_paths),
        "fields": int(form["n_tiles"].sum()),
        "codes": form["blinded_code"].astype(str).tolist(),
    }


def _audit_reader_package(
    root: Path,
    review_root: Path,
    roster: pd.DataFrame,
    reader_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = root / "reader_package/public"
    delivery = review_root / "FOR_PATHOLOGIST"
    canonical_check = _audit_public_directory(canonical, roster)
    _audit_public_directory(delivery, roster)
    if _inventory_records(canonical) != _inventory_records(delivery):
        raise AuditError("reader package: delivery mirror differs from canonical public tree")
    if review_root.stat().st_mode & 0o777 != 0o700 or delivery.stat().st_mode & 0o777 != 0o700:
        raise AuditError("reader package: review privacy boundary is not mode 0700")
    if canonical.stat().st_mode & 0o777 != 0o700:
        raise AuditError("reader package: canonical public privacy boundary is not mode 0700")
    embargoed = root / "reader_package/embargoed"
    expected_embargoed = {
        "secret_salt.bin",
        "handoff_sealed_utc.txt",
        "occurrence_key.csv",
        "tile_provenance.parquet",
        "coordinate_binding_evidence.parquet",
        "render_geometry.json",
        "render_environment.json",
    }
    if (
        embargoed.stat().st_mode & 0o777 != 0o700
        or {path.name for path in embargoed.iterdir()} != expected_embargoed
    ):
        raise AuditError("reader package: embargoed directory census or permissions mismatch")
    for path in embargoed.iterdir():
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o777 != 0o600:
            raise AuditError("reader package: embargoed artifact is not private")
    salt = (embargoed / "secret_salt.bin").read_bytes()
    if (
        len(salt) != 32
        or hashlib.sha256(salt).hexdigest()
        != (delivery / "PUBLIC_SALT_SHA256.txt").read_text(encoding="ascii").strip()
    ):
        raise AuditError("reader package: secret/public salt binding mismatch")
    if reader_receipt.get("public_salt_sha256") != hashlib.sha256(salt).hexdigest():
        raise AuditError("reader package receipt: public salt binding mismatch")

    key = pd.read_csv(embargoed / "occurrence_key.csv")
    if len(key) != 40 or key["prototype_id"].nunique() != CANONICAL_K:
        raise AuditError("reader blinding: occurrence census mismatch")
    originals = key.loc[key["occurrence"].eq(0)]
    duplicates = key.loc[key["occurrence"].eq(1)]
    if (
        len(originals) != CANONICAL_K
        or len(duplicates) != 8
        or set(duplicates["prototype_id"].astype(int)) - set(originals["prototype_id"].astype(int))
        or not originals["is_controlling_read"].astype(bool).all()
        or duplicates["is_controlling_read"].astype(bool).any()
    ):
        raise AuditError("reader blinding: duplicate/control contract mismatch")
    occurrences = [
        (int(row.prototype_id), int(row.occurrence))
        for row in key[["prototype_id", "occurrence"]].itertuples(index=False)
    ]
    replay_blinding = _hmac_blinding_table(occurrences, salt)
    observed_blinding = (
        key[list(replay_blinding.columns)]
        .sort_values("presentation_order", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_blinding = replay_blinding.sort_values(
        "presentation_order", kind="mergesort"
    ).reset_index(drop=True)
    _assert_frame_exact(observed_blinding, expected_blinding, "reader HMAC blinding")
    form = pd.read_csv(delivery / "review_form.csv", keep_default_na=False)
    ordered_key = key.sort_values("presentation_order", kind="mergesort")
    if not np.array_equal(form["blinded_code"].astype(str), ordered_key["code"].astype(str)):
        raise AuditError("reader package: blinded form order mismatch")

    support = pd.read_csv(root / "montage/support_roster.csv")
    support_by_prototype = support.set_index("prototype_id")["distinct_patient_support"].astype(int)
    eligible = sorted(int(index) for index, count in support_by_prototype.items() if count >= 12)
    expected_duplicates = sorted(
        int(value)
        for value in np.random.Generator(np.random.PCG64(VOCAB_SEED)).choice(
            np.asarray(eligible, dtype=np.int64), size=8, replace=False
        )
    )
    if sorted(duplicates["prototype_id"].astype(int)) != expected_duplicates:
        raise AuditError("reader package: deterministic duplicate selection mismatch")
    candidates = pd.read_parquet(root / "montage/canonical_candidates.parquet")
    provenance = pd.read_parquet(embargoed / "tile_provenance.parquet")
    if len(provenance) != 480:
        raise AuditError("reader package: tile provenance census mismatch")
    selection_columns = [
        *candidates.columns,
        "montage_slot",
        "prototype_index",
        "occurrence",
        "selection_seed",
        "selection_stage",
        "overlaps_avoidance_set",
    ]
    key_lookup = key.set_index(["prototype_id", "occurrence"])
    for prototype in range(CANONICAL_K):
        block = candidates.loc[candidates["prototype_id"].eq(prototype)].copy()
        first, first_audit = _select_montage(block, prototype, 0)
        selections = [(0, first, first_audit)]
        if prototype in expected_duplicates:
            second, second_audit = _select_montage(
                block, prototype, 1, set(first["patient_id"].astype(str))
            )
            selections.append((1, second, second_audit))
        for occurrence, expected_selection, expected_audit in selections:
            observed = (
                provenance.loc[
                    provenance["prototype_id"].eq(prototype)
                    & provenance["occurrence"].eq(occurrence),
                    selection_columns,
                ]
                .sort_values("montage_slot", kind="mergesort")
                .reset_index(drop=True)
            )
            _assert_frame_exact(
                observed,
                expected_selection[selection_columns],
                "reader deterministic montage selection",
            )
            key_row = key_lookup.loc[(prototype, occurrence)]
            if any(key_row[name] != value for name, value in expected_audit.items()):
                raise AuditError("reader package: selection audit metadata mismatch")
    if (
        provenance.groupby("code").size().ne(12).any()
        or provenance.groupby("code")["patient_id"].nunique().ne(12).any()
    ):
        raise AuditError("reader package: montage size or patient uniqueness mismatch")
    if not np.array_equal(
        form.set_index("blinded_code").loc[provenance["code"].unique(), "n_tiles"].to_numpy(),
        provenance.groupby("code", sort=False).size().to_numpy(),
    ):
        raise AuditError("reader package: form/provenance tile counts mismatch")

    evidence = pd.read_parquet(embargoed / "coordinate_binding_evidence.parquet")
    evidence_keys = ["slide_id", "tile_index", "prototype_id", "occurrence", "code"]
    if (
        len(evidence) != len(provenance)
        or not evidence["packed_patch_feature_coordinate_match"].astype(bool).all()
        or evidence[evidence_keys].value_counts().sort_index().to_dict()
        != provenance[evidence_keys].value_counts().sort_index().to_dict()
    ):
        raise AuditError("reader package: coordinate evidence/provenance mismatch")
    geometry = _load_json(embargoed / "render_geometry.json", "reader render geometry")
    if set(geometry) != set(provenance["slide_id"].astype(str)):
        raise AuditError("reader package: render geometry slide census mismatch")
    for record in geometry.values():
        wsi = Path(str(record.get("wsi_path", "")))
        _regular_file(wsi, "reader render source")
        metadata = wsi.stat()
        if (
            metadata.st_size != int(record.get("wsi_size_bytes", -1))
            or metadata.st_mtime_ns != int(record.get("wsi_mtime_ns", -1))
            or record.get("coordinate_units") != "level0_pixels"
            or int(record.get("patch_size_level0", 0)) <= 0
        ):
            raise AuditError("reader package: render source identity/geometry mismatch")
    sealed_utc = (embargoed / "handoff_sealed_utc.txt").read_text(encoding="ascii").strip()
    manifest = _load_json(delivery / "HANDOFF_MANIFEST.json", "reader manifest")
    if manifest.get("sealed_utc") != sealed_utc:
        raise AuditError("reader package: embargoed/public timestamp mismatch")
    _validate_identity_tree(reader_receipt.get("artifacts", {}), "reader receipt")
    return {
        "public_manifest_sha256": canonical_check["manifest_sha256"],
        "montages": canonical_check["montages"],
        "fields": canonical_check["fields"],
        "duplicated_concepts": len(duplicates),
        "provenance_rows": len(provenance),
        "coordinate_evidence_rows": len(evidence),
        "source_wsis": len(geometry),
        "blinding_replay": "PASS",
    }


def _expected_frozen_files(roster: pd.DataFrame, codes: Iterable[str]) -> set[str]:
    files = {
        "analysis_dictionary.json",
        "analysis_dictionary.json.seal.json",
        "implementation_contract.json",
        "implementation_contract.json.seal.json",
        "assignments/source_manifest.parquet",
        "benchmarks/assignment_first_completed_slide.json",
        "benchmarks/lloyd_kmeans_first_completed.json",
        "benchmarks/pca64_first_completed.json",
        "benchmarks/teacher_first_completed_head.json",
        "inputs/source_roster_label_blind.csv",
        "inputs/tcga_surgen_primary.csv",
        "inputs/splits/.integrity_hash",
        "inputs/splits/splits.parquet",
        "inputs/splits/summary.json",
        "locks/profiles.lock",
        "locks/reader_package.lock",
        "locks/teachers.lock",
        "locks/vocabularies.lock",
        "mappings/outer_to_reference.csv",
        "montage/canonical_candidates.parquet",
        "montage/support_roster.csv",
        "profiles/oof_patient_profiles.parquet",
        "profiles/reference_distance_quantiles.csv",
        "reader_package/embargoed/coordinate_binding_evidence.parquet",
        "reader_package/embargoed/handoff_sealed_utc.txt",
        "reader_package/embargoed/occurrence_key.csv",
        "reader_package/embargoed/render_environment.json",
        "reader_package/embargoed/render_geometry.json",
        "reader_package/embargoed/secret_salt.bin",
        "reader_package/embargoed/tile_provenance.parquet",
        "teachers/execution_contract.json",
        "teachers/oof_attention_patients_5seed_mean.parquet",
        "teachers/oof_native_logits_recomputed.parquet",
        "vocabularies/inventory.csv",
        *(f"receipts/{name}.json" for name in PINNED_STAGE_RECEIPTS),
    }
    for slide_id in roster["slide_id"].astype(str):
        files.add(f"assignments/source/{slide_id}.npz")
        files.add(f"assignments/source/{slide_id}.npz.receipt.json")
    for key in PROFILE_KEYS:
        files.add(f"profiles/{key}/slide_profiles.parquet")
        files.add(f"profiles/{key}/patient_profiles.parquet")
    for fold in (None, *FOLDS):
        scope = "reference" if fold is None else f"outer_fold_{fold}"
        for name in (
            "pca_basis.npz",
            "pca_basis.npz.receipt.json",
            "sample_ids.parquet",
            "sample_pca64.float32.npy",
            "sample_plan.parquet",
        ):
            files.add(f"vocabularies/{scope}/{name}")
        if fold is not None:
            files.add(f"vocabularies/{scope}/vocabulary.npz")
            files.add(f"vocabularies/{scope}/vocabulary.npz.receipt.json")
    for k in VARIANT_K:
        for seed in VARIANT_SEEDS:
            base = f"vocabularies/reference/variants/k{k}_seed{seed}"
            files.add(f"{base}/vocabulary.npz")
            files.add(f"{base}/vocabulary.npz.receipt.json")
    for seed in MODEL_SEEDS:
        files.add(f"teachers/seed{seed}_oof_attention_patients.parquet")
        for fold in FOLDS:
            base = f"teachers/seed{seed}/fold_{fold}"
            for name in (
                "heldout_attention_patients.parquet",
                "heldout_attention_slides.parquet",
                "patient_logits.parquet",
                "receipt.json",
                "slide_logits.parquet",
            ):
                files.add(f"{base}/{name}")
    public_static = {
        "HANDOFF_MANIFEST.json",
        "HANDOFF_MANIFEST.sha256",
        "INSTRUCTIONS.md",
        "PUBLIC_SALT_SHA256.txt",
        "RUBRIC.md",
        "review_form.csv",
    }
    files.update(f"reader_package/public/{name}" for name in public_static)
    files.update(f"reader_package/public/montages/{code}.jpg" for code in codes)
    return files


def _audit_tree_hygiene(
    root: Path, review_root: Path, roster: pd.DataFrame, codes: Sequence[str]
) -> dict[str, Any]:
    expected_frozen = _expected_frozen_files(roster, codes)
    _assert_exact_tree(root, expected_frozen, "frozen output tree")
    public_static = {
        "HANDOFF_MANIFEST.json",
        "HANDOFF_MANIFEST.sha256",
        "INSTRUCTIONS.md",
        "PUBLIC_SALT_SHA256.txt",
        "RUBRIC.md",
        "review_form.csv",
    }
    expected_review = {
        "README_COORDINATOR.md",
        "PRE_READER_STATUS.json",
        *(f"FOR_PATHOLOGIST/{name}" for name in public_static),
        *(f"FOR_PATHOLOGIST/montages/{code}.jpg" for code in codes),
    }
    _assert_exact_tree(review_root, expected_review, "reviews/v14 tree")
    coordinator_status = _load_json(
        review_root / "PRE_READER_STATUS.json", "coordinator pre-reader status"
    )
    completed = coordinator_status.get("completed_gates", {})
    if coordinator_status.get("status") != "READY_FOR_PATHOLOGIST" or completed != {
        "kmeans_fits": 14,
        "pca_fits": 6,
        "reader_montages": 40,
        "source_profile_matrices": 6,
        "teacher_heads": 25,
    }:
        raise AuditError("coordinator pre-reader status: not handoff-ready")
    return {
        "frozen_files": len(expected_frozen),
        "review_files": len(expected_review),
        "symlinks": 0,
        "temporary_or_orphan_files": 0,
    }


def audit(
    *,
    root: Path = FROZEN_ROOT,
    review_root: Path = REVIEW_ROOT,
    receipt_path: Path = AUDIT_RECEIPT,
    write_receipt: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    root = root.resolve(strict=True)
    review_root = review_root.resolve(strict=True)
    if root == receipt_path.resolve(strict=False) or receipt_path.is_relative_to(root):
        raise AuditError("audit receipt: destination must be outside the frozen output")
    if receipt_path.is_relative_to(review_root):
        raise AuditError("audit receipt: destination must be outside reviews/v14")
    before_runner = sha256_file(RUNNER)
    before_auditor = sha256_file(AUDITOR)
    before_output_tree, _ = _tree_digest(root)
    before_review_tree, _ = _tree_digest(review_root)

    gates: dict[str, dict[str, Any]] = {}
    receipts, core_metrics = _audit_core(root)
    gates["core_identities"] = {"status": "PASS", **core_metrics}
    roster, pack_index, source_metrics = _audit_source_and_pack(root)
    gates["source_and_pack"] = {"status": "PASS", **source_metrics}
    v13_metrics = _audit_v13_chain(receipts["preflight"])
    gates["final_v13_chain"] = {"status": "PASS", **v13_metrics}
    vocabulary_metrics = _audit_vocabularies(root, roster, pack_index, receipts)
    gates["vocabularies_and_mappings"] = {"status": "PASS", **vocabulary_metrics}
    profile_metrics = _audit_assignments_and_profiles(root, roster, receipts)
    gates["assignments_and_profiles"] = {"status": "PASS", **profile_metrics}
    teacher_metrics = _audit_teachers(root, roster, receipts["preflight"], receipts["teachers"])
    gates["teachers"] = {"status": "PASS", **teacher_metrics}
    reader_metrics = _audit_reader_package(root, review_root, roster, receipts["reader_package"])
    gates["reader_package"] = {"status": "PASS", **reader_metrics}
    form = pd.read_csv(review_root / "FOR_PATHOLOGIST/review_form.csv")
    hygiene_metrics = _audit_tree_hygiene(
        root, review_root, roster, form["blinded_code"].astype(str).tolist()
    )
    gates["tree_hygiene"] = {"status": "PASS", **hygiene_metrics}

    after_runner = sha256_file(RUNNER)
    after_auditor = sha256_file(AUDITOR)
    after_output_tree, _ = _tree_digest(root)
    after_review_tree, _ = _tree_digest(review_root)
    if (
        before_runner != after_runner
        or before_auditor != after_auditor
        or before_output_tree != after_output_tree
        or before_review_tree != after_review_tree
    ):
        raise AuditError("read-only guarantee: a frozen input tree changed during audit")
    gates["read_only_preservation"] = {
        "status": "PASS",
        "auditor_unchanged": True,
        "runner_unchanged": True,
        "frozen_output_tree_unchanged": True,
        "review_tree_unchanged": True,
    }
    handoff_ready = all(gate.get("status") == "PASS" for gate in gates.values())
    if not handoff_ready:
        raise AuditError("handoff readiness: one or more independent gates failed")
    result = {
        "schema_version": SCHEMA_VERSION,
        "component": COMPONENT,
        "status": "PASS",
        "handoff_ready": handoff_ready,
        "created_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "auditor": {
            "path": str(AUDITOR),
            "sha256": after_auditor,
        },
        "frozen_inputs": {
            "runner_sha256": after_runner,
            "output_tree_sha256": after_output_tree,
            "review_tree_sha256": after_review_tree,
            "handoff_manifest_sha256": PINNED_HANDOFF_MANIFEST_SHA256,
        },
        "gates": gates,
        "disclosure": {
            "prototype_code_mapping_in_receipt": False,
            "embargoed_identifiers_in_receipt": False,
        },
    }
    if write_receipt:
        if receipt_path.parent.resolve(strict=False) != AUDIT_ROOT.resolve(strict=False):
            raise AuditError("audit receipt: production receipt directory is hard-pinned")
        _atomic_receipt(receipt_path, json_bytes(result))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="run every check without publishing the independent receipt",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    try:
        result = audit(write_receipt=not arguments.no_write)
    except AuditError as exc:
        raise SystemExit(f"INDEPENDENT_AUDIT_FAIL: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
