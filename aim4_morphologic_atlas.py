#!/usr/bin/env python3
"""Append-only Aim 4 cap=8192 correction and verification runner.

The legacy Aim 4 analysis under ``outputs/aim1/e3b`` is an input only.  This
runner never changes it.  A run is built in a new, explicit absolute output
root and binds every reused artifact by size and SHA256.

Corrections enforced here:

* the frozen, label-blind k=32 vocabulary and the three E0 OOF-attention files
  are imported byte-for-byte from the legacy E4 root;
* all twelve RIH/SR1482 attention files are re-exported from the validated Aim
  2 v4 cap=8192 full-source refits, not the superseded checkpoint tree;
* slide/patient profiles and tile candidates are recomputed from those inputs;
* every inferential panel uses one fixed family of exactly 32 prototypes.
  Structurally non-estimable rows remain in that family with p=1 and can never
  become significant;
* every arm, cohort, prototype, quantity, attention seed, and primary-versus-
  metastatic contrast is checked for exact coverage before a report is made;
* the corrected readout creates fresh, separate blinded base and attention-
  addendum packets.  Their unblinding keys live outside both packets.

No MIL model is trained by this runner.  Attention export is inference through
already-frozen checkpoints.  ``preflight`` is read-only; construction commands
require ``--apply``.  Full numeric/final verification exclusively creates its
corresponding immutable verification addendum after every check passes.

Typical sequence::

    python aim4_morphologic_atlas.py preflight --input-e4-root /abs/legacy/e3b \
      --aim2-root /abs/aim2_v4 --output-root /abs/new/e4_corrected
    python aim4_morphologic_atlas.py prepare ... --apply
    python aim4_morphologic_atlas.py attention ... --apply
    python aim4_morphologic_atlas.py assign ... --apply
    python aim4_morphologic_atlas.py analyze ... --n-bootstrap 2000 --apply
    python aim4_morphologic_atlas.py review-packets ... --apply
    python aim4_morphologic_atlas.py finalize-numeric ... --apply
    python aim4_morphologic_atlas.py verify-numeric ...
    python aim4_morphologic_atlas.py import-reviews ... --base-form /abs/base.csv \
      --attention-addendum-form /abs/addendum.csv --apply
    python aim4_morphologic_atlas.py finalize ... --apply
    python aim4_morphologic_atlas.py verify-output ...
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim4_morphologic_atlas_base as legacy  # noqa: E402
from oceanpath.aim1 import atlas, attention, paths  # noqa: E402
from oceanpath.extraction.mpp_sampling import validate_exact_mpp_coordinate_file  # noqa: E402

CAP = 8192
K = 32
SEEDS: tuple[int, ...] = (42, 43, 44)
TARGET_ARMS: tuple[str, ...] = (
    "rih_primary",
    "rih_metastatic",
    "sr1482_primary",
    "sr1482_metastatic",
)
ALL_ARMS: tuple[str, ...] = ("e0", *TARGET_ARMS)
ARM_TARGET = {
    "rih_primary": "RIH",
    "rih_metastatic": "RIH",
    "sr1482_primary": "SurGen",
    "sr1482_metastatic": "SurGen",
}
TARGET_SLUG = {"RIH": "rih", "SurGen": "surgen"}
QUANTITIES: tuple[str, ...] = ("abundance", "attn_mass_mean")
DEFAULT_BOOTSTRAP = 2_000
BOOTSTRAP_SEED = atlas.VOCAB_SEED
FDR_ALPHA = 0.05
NUMERIC_COMPLETION_NAME = "numeric_complete.json"
NUMERIC_VERIFICATION_NAME = "numeric_verification.json"
VERIFICATION_NAME = "verification.json"
REVIEW_TRANSACTION_RECEIPT = "_bundle_receipt_DO_NOT_SHARE.json"
REVIEW_IMPORT_TRANSACTION_RECEIPT = "_import_receipt_DO_NOT_SHARE.json"
ARCHIVAL_INPUT_COMMANDS = frozenset(
    {"verify-numeric", "import-reviews", "finalize", "verify-output"}
)
EXPECTED_COUNTS = {
    "e0": (1642, 1486, 604),
    "rih_primary": (155, 153, 70),
    "rih_metastatic": (85, 85, 37),
    "sr1482_primary": (468, 324, 147),
    "sr1482_metastatic": (100, 74, 30),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"expected a file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _stat_identity(path: Path) -> dict[str, Any]:
    """A clearly labelled metadata identity for the 92-GB unpacked store."""
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _absolute_dir(path: Path, label: str, *, exists: bool) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an explicit absolute path: {path}")
    resolved = path.resolve(strict=exists)
    if exists and not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    return resolved


def _validate_roots(
    input_e4_root: Path,
    aim2_root: Path,
    output_root: Path,
    *,
    output_exists: bool,
    input_exists: bool = True,
) -> tuple[Path, Path, Path]:
    input_root = _absolute_dir(input_e4_root, "--input-e4-root", exists=input_exists)
    if not input_exists and input_root.exists() and not input_root.is_dir():
        raise NotADirectoryError(f"--input-e4-root is not a directory: {input_root}")
    aim2 = _absolute_dir(aim2_root, "--aim2-root", exists=True)
    output = _absolute_dir(output_root, "--output-root", exists=output_exists)
    roots = (input_root, aim2, output)
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(f"input/output roots must be distinct and non-nested: {left}, {right}")
    return input_root, aim2, output


def _write_bytes_once(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _write_json_once(path: Path, value: Any) -> None:
    _write_bytes_once(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def _write_csv_once(path: Path, frame: pd.DataFrame) -> None:
    _write_bytes_once(path, frame.to_csv(index=False).encode())


def _copy_once(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        os.link(temporary, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    source_id = _identity(source)
    destination_id = _identity(destination)
    if (source_id["size_bytes"], source_id["sha256"]) != (
        destination_id["size_bytes"],
        destination_id["sha256"],
    ):
        raise RuntimeError(f"copy verification failed: {source} -> {destination}")
    return {"source": source_id, "imported": destination_id}


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sanitize_json(value: Any) -> Any:
    """Convert numpy values and structural non-finites to strict-JSON values."""
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _source_files() -> list[Path]:
    fixed = [REPO / "aim4_morphologic_atlas.py", REPO / "aim4_morphologic_atlas_base.py", REPO / "aim2_loco_transport.py",
             REPO / "pyproject.toml", REPO / "uv.lock"]
    dynamic = sorted((REPO / "src" / "oceanpath").rglob("*.py"))
    configs = sorted((REPO / "configs").rglob("*.yaml"))
    files = [path for path in (*fixed, *dynamic, *configs) if path.is_file()]
    if REPO / "aim4_morphologic_atlas.py" not in files:
        raise RuntimeError("live corrected source is missing")
    return files


def _snapshot_sources(output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in _source_files():
        relative = source.relative_to(REPO)
        destination = output / "source_snapshot" / relative
        evidence = _copy_once(source, destination)
        rows.append({"relative_path": str(relative), **evidence})
    receipt = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "files": rows,
    }
    _write_json_once(output / "receipts" / "source_snapshot.json", receipt)
    return receipt


def _checkpoint_path(aim2_root: Path, target: str, seed: int) -> Path:
    return (
        aim2_root / "e2a" / "train" / f"pb_cap{CAP}" / TARGET_SLUG[target]
        / f"seed{seed}" / "final" / "refit" / "model.ckpt"
    )


def _arm_manifests() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name in ALL_ARMS:
        frame = legacy.arm_manifest(legacy.ARMS[name]).copy()
        required = {"slide_id", "patient_id", "target_label", "specimen_role", "cohort"}
        if missing := required - set(frame):
            raise RuntimeError(f"{name}: manifest columns missing: {sorted(missing)}")
        if frame["slide_id"].isna().any() or frame["slide_id"].duplicated().any():
            raise RuntimeError(f"{name}: slide_id must be non-null and unique")
        if frame["patient_id"].isna().any():
            raise RuntimeError(f"{name}: patient_id is null")
        labels = pd.to_numeric(frame["target_label"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(labels).all() or set(np.unique(labels)) != {0.0, 1.0}:
            raise RuntimeError(f"{name}: target labels must contain both binary classes")
        varying = frame.groupby("patient_id")["target_label"].nunique(dropna=False)
        if (varying != 1).any():
            raise RuntimeError(f"{name}: target label varies within patient")
        observed = (
            len(frame),
            int(frame["patient_id"].nunique()),
            int(frame.drop_duplicates("patient_id")["target_label"].sum()),
        )
        if observed != EXPECTED_COUNTS[name]:
            raise RuntimeError(
                f"{name}: frozen count mismatch; expected {EXPECTED_COUNTS[name]}, got {observed}"
            )
        frames[name] = frame.reset_index(drop=True)
    return frames


def _validate_vocabulary(input_root: Path, manifests: dict[str, pd.DataFrame]) -> dict[str, Any]:
    vocabulary = input_root / "vocabulary" / f"vocab_k{K}.npz"
    metadata = vocabulary.with_suffix(".json")
    plan_path = input_root / "vocabulary" / f"sample_plan_k{K}.parquet"
    for path in (vocabulary, metadata, plan_path):
        if not path.is_file():
            raise RuntimeError(f"legacy model-independent vocabulary artifact missing: {path}")
    blob = np.load(vocabulary)
    exact_arrays = {"centroids", "pca_mean", "pca_components"}
    if set(blob.files) != exact_arrays:
        raise RuntimeError(f"vocabulary array inventory mismatch: {blob.files}")
    if blob["centroids"].shape[0] != K or blob["pca_components"].shape[0] != blob["centroids"].shape[1]:
        raise RuntimeError("vocabulary dimensions are inconsistent with k=32")
    if blob["pca_mean"].ndim != 1 or blob["pca_components"].shape[1] != len(blob["pca_mean"]):
        raise RuntimeError("vocabulary PCA dimensions are inconsistent")
    if not all(np.isfinite(blob[key]).all() for key in blob.files):
        raise RuntimeError("vocabulary contains non-finite arrays")
    meta = _json_object(metadata)
    if int(meta.get("n_prototypes", -1)) != K or not bool(meta.get("label_blind")):
        raise RuntimeError("vocabulary is not the frozen label-blind k=32 object")
    plan = pd.read_parquet(plan_path)
    required = {"slide_id", "patient_id", "atlas_group", "n_tiles", "n_sample"}
    if set(plan.columns) != required or plan["slide_id"].duplicated().any():
        raise RuntimeError("vocabulary sample plan schema/identity is invalid")
    union = set(pd.concat([frame[["slide_id"]] for frame in manifests.values()])["slide_id"])
    if set(plan["slide_id"].astype(str)) != set(map(str, union)):
        raise RuntimeError("vocabulary plan does not cover the exact evaluated slide union")
    if (pd.to_numeric(plan["n_sample"], errors="coerce") <= 0).any():
        raise RuntimeError("vocabulary sample plan contains an empty allocation")
    return {
        "vocab_npz": _identity(vocabulary),
        "vocab_json": _identity(metadata),
        "sample_plan": _identity(plan_path),
        "n_plan_slides": int(len(plan)),
        "n_sample_tiles": int(plan["n_sample"].sum()),
    }


def _feature_inventory(manifests: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], dict[str, Path]]:
    slides = sorted(
        set(
            pd.concat([frame[["slide_id"]] for frame in manifests.values()])[
                "slide_id"
            ].astype(str)
        )
    )
    files: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for slide_id in slides:
        path = paths.PINNED_FEATURE_DIR / f"{slide_id}.h5"
        if not path.is_file():
            raise RuntimeError(f"feature file missing: {path}")
        with h5py.File(path, "r") as handle:
            if set(handle.keys()) != {"coords", "features"}:
                raise RuntimeError(f"{slide_id}: unexpected feature HDF5 keys")
            features = handle["features"]
            coords = handle["coords"]
            if features.ndim != 2 or features.shape[1] != 1024:
                raise RuntimeError(f"{slide_id}: expected 1,024-dimensional UNIv1 features")
            if coords.shape != (features.shape[0], 2):
                raise RuntimeError(f"{slide_id}: feature/coordinate dimensions differ")
            if str(features.dtype) != "float32" or not np.issubdtype(coords.dtype, np.integer):
                raise RuntimeError(f"{slide_id}: unexpected feature/coordinate dtype")
            shape = [int(features.shape[0]), int(features.shape[1])]
        evidence = _stat_identity(path)
        row = {"slide_id": slide_id, "shape": shape, **evidence}
        rows.append(row)
        files[slide_id] = path
    # The payload is 92 GB.  We bind every file's size, mtime, shape, and path,
    # then validate the exact coordinates against each attention artifact.  We
    # deliberately call this an inventory fingerprint rather than a content SHA.
    fingerprint = hashlib.sha256(_canonical(rows).encode()).hexdigest()
    return {
        "root": str(paths.PINNED_FEATURE_DIR.resolve(strict=True)),
        "integrity_scope": "metadata inventory plus exact per-slide HDF5 schema/attention-coordinate equality",
        "n_files": len(rows),
        "inventory_sha256": fingerprint,
        "files": rows,
    }, files


def _validate_attention_file(
    path: Path,
    manifest: pd.DataFrame,
    *,
    seed: int,
    feature_files: dict[str, Path],
    expected_checkpoint: Path | None,
    deep: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"attention artifact missing: {path}")
    expected = set(manifest["slide_id"].astype(str))
    oof_fold_counts: dict[str, int] = {}
    oof_checkpoints: dict[str, set[str]] = {}
    with h5py.File(path, "r") as handle:
        actual = set(map(str, handle.keys()))
        if actual != expected:
            raise RuntimeError(
                f"{path.name}: attention slide coverage mismatch; missing={len(expected-actual)}, "
                f"unexpected={len(actual-expected)}"
            )
        for slide_id in sorted(expected):
            group = handle[slide_id]
            if set(group.keys()) != {"attention", "coords", "top_decile"}:
                raise RuntimeError(f"{path.name}/{slide_id}: dataset inventory mismatch")
            n = int(group.attrs.get("n_tiles", -1))
            if n <= 0:
                raise RuntimeError(f"{path.name}/{slide_id}: non-positive tile count")
            if group["attention"].shape != (n,) or group["top_decile"].shape != (n,):
                raise RuntimeError(f"{path.name}/{slide_id}: attention dimensions differ")
            if group["coords"].shape != (n, 2):
                raise RuntimeError(f"{path.name}/{slide_id}: coordinate dimensions differ")
            if "seed" not in group.attrs or int(group.attrs["seed"]) != seed:
                raise RuntimeError(f"{path.name}/{slide_id}: seed attribute mismatch")
            if expected_checkpoint is not None:
                observed = Path(str(group.attrs.get("checkpoint", ""))).resolve(strict=True)
                if observed != expected_checkpoint.resolve(strict=True):
                    raise RuntimeError(f"{path.name}/{slide_id}: checkpoint entitlement mismatch")
                if str(group.attrs.get("mode", "")) != "refit":
                    raise RuntimeError(f"{path.name}/{slide_id}: target attention is not refit mode")
            else:
                try:
                    fold = int(group.attrs["fold"])
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"{path.name}/{slide_id}: OOF fold entitlement is missing"
                    ) from error
                checkpoint_value = str(group.attrs.get("checkpoint", ""))
                checkpoint_path = Path(checkpoint_value)
                if fold not in range(paths.N_FOLDS) or not checkpoint_path.is_absolute():
                    raise RuntimeError(
                        f"{path.name}/{slide_id}: malformed OOF entitlement"
                    )
                oof_fold_counts[str(fold)] = oof_fold_counts.get(str(fold), 0) + 1
                oof_checkpoints.setdefault(str(fold), set()).add(checkpoint_value)
            if deep:
                weights = group["attention"][:].astype(float)
                coords = group["coords"][:]
                if not np.isfinite(weights).all() or (weights < 0).any():
                    raise RuntimeError(f"{path.name}/{slide_id}: invalid attention weights")
                if not np.isclose(weights.sum(), 1.0, atol=2e-5, rtol=2e-5):
                    raise RuntimeError(f"{path.name}/{slide_id}: attention does not sum to one")
                top_decile = group["top_decile"][:]
                if not np.issubdtype(top_decile.dtype, np.bool_):
                    raise RuntimeError(f"{path.name}/{slide_id}: top-decile mask is inconsistent")
                mask = top_decile.astype(bool)
                minimum_count = max(1, int(np.ceil(0.10 * n)))
                if mask.sum() < minimum_count or (
                    (~mask).any()
                    and float(weights[mask].min()) < float(weights[~mask].max()) - 1e-12
                ):
                    raise RuntimeError(f"{path.name}/{slide_id}: top-decile mask is inconsistent")
                with h5py.File(feature_files[slide_id], "r") as features:
                    if features["features"].shape[0] != n or not np.array_equal(
                        coords, features["coords"][:]
                    ):
                        raise RuntimeError(
                            f"{path.name}/{slide_id}: attention and feature tile order differ"
                        )
    summary_path = path.with_suffix(".json")
    if not summary_path.is_file():
        raise RuntimeError(f"attention summary missing: {summary_path}")
    summary = _json_object(summary_path)
    expected_mode = "out_of_fold" if expected_checkpoint is None else "refit"
    if summary.get("mode") != expected_mode or int(summary.get("n_slides", -1)) != len(expected):
        raise RuntimeError(f"{summary_path}: mode/count mismatch")
    if not np.isclose(float(summary.get("top_fraction", np.nan)), 0.10, atol=1e-12):
        raise RuntimeError(f"{summary_path}: top-attention fraction mismatch")
    if expected_checkpoint is not None:
        if int(summary.get("seed", -1)) != seed:
            raise RuntimeError(f"{summary_path}: seed mismatch")
        observed = Path(str(summary.get("checkpoint", ""))).resolve(strict=True)
        if observed != expected_checkpoint.resolve(strict=True):
            raise RuntimeError(f"{summary_path}: checkpoint mismatch")
    else:
        expected_per_fold = {
            str(key): int(value) for key, value in summary.get("per_fold", {}).items()
        }
        if expected_per_fold != oof_fold_counts or set(oof_fold_counts) != {
            str(value) for value in range(paths.N_FOLDS)
        }:
            raise RuntimeError(f"{summary_path}: OOF fold coverage differs from HDF5")
        if any(len(values) != 1 for values in oof_checkpoints.values()):
            raise RuntimeError(f"{path.name}: an OOF fold names multiple checkpoints")
    return {
        "h5": _identity(path),
        "summary": _identity(summary_path),
        "n_slides": len(expected),
        "deep_validation": bool(deep),
        "oof_fold_counts": oof_fold_counts if expected_checkpoint is None else None,
    }


def _validate_checkpoint(aim2_root: Path, target: str, seed: int) -> dict[str, Any]:
    checkpoint = _checkpoint_path(aim2_root, target, seed)
    run_root = checkpoint.parents[2]
    summary_path = run_root / "fit_summary.json"
    info_path = checkpoint.parent / "info.json"
    config_path = run_root / "resolved_config.yaml"
    request_path = run_root / "run_request.json"
    for path in (checkpoint, summary_path, info_path, config_path, request_path):
        if not path.is_file():
            raise RuntimeError(f"Aim 2 v4 refit artifact missing: {path}")
    summary = _json_object(summary_path)
    info = _json_object(info_path)
    request = _json_object(request_path)
    expected = {
        "status": "completed",
        "target": target,
        "seed": seed,
        "sampling_seed": seed,
        "cap": CAP,
        "sampler": "patient_natural",
        "loss_weighting": "none",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"{summary_path}: expected {key}={value!r}, got {summary.get(key)!r}")
    lineage_name = str(summary.get("lineage", ""))
    if not lineage_name.startswith("aim2_cap8192_v4_"):
        raise RuntimeError(f"{summary_path}: refit is not from an Aim 2 cap8192 v4 lineage")
    result = summary.get("result", {})
    info_required = {
        "strategy": "refit",
        "seed": seed,
        "sampling_seed": seed,
        "train_sampling_strategy": "patient_natural",
        "sample_weight_column": None,
        "class_weights": None,
        "dataset_max_instances": CAP,
        "max_instances": None,
        "eval_full_bags": True,
    }
    for key, value in info_required.items():
        if result.get(key) != value or info.get(key) != value:
            raise RuntimeError(f"{run_root}: corrected refit contract mismatch for {key}")
    if (
        request.get("target") != target
        or request.get("seed") != seed
        or request.get("cap") != CAP
        or request.get("lineage") != lineage_name
    ):
        raise RuntimeError(f"{request_path}: request target/seed/cap mismatch")
    model_id = _identity(checkpoint)
    recorded = summary.get("model", {})
    if recorded.get("sha256") != model_id["sha256"] or int(
        recorded.get("size_bytes", -1)
    ) != model_id["size_bytes"]:
        raise RuntimeError(f"{summary_path}: checkpoint identity does not match fit receipt")
    return {
        "target": target,
        "seed": seed,
        "checkpoint": model_id,
        "fit_summary": _identity(summary_path),
        "info": _identity(info_path),
        "resolved_config": _identity(config_path),
        "run_request": _identity(request_path),
        "contract": info_required,
    }


@dataclass
class InputBundle:
    input_root: Path
    aim2_root: Path
    manifests: dict[str, pd.DataFrame]
    feature_files: dict[str, Path]
    checkpoints: dict[tuple[str, int], Path]
    receipt: dict[str, Any]


def validate_inputs(input_e4_root: Path, aim2_root: Path, *, deep_attention: bool = True) -> InputBundle:
    input_root = _absolute_dir(input_e4_root, "--input-e4-root", exists=True)
    aim2 = _absolute_dir(aim2_root, "--aim2-root", exists=True)
    if input_root == aim2 or input_root in aim2.parents or aim2 in input_root.parents:
        raise ValueError("legacy E4 and Aim 2 input roots must be distinct and non-nested")
    manifests = _arm_manifests()
    manifest_files: dict[str, dict[str, Any]] = {"e0": _identity(paths.DEV_MANIFEST)}
    for name in TARGET_ARMS:
        arm = legacy.ARMS[name]
        manifest_files[name] = _identity(legacy.e2a.target_manifest(arm.target, arm.role))
    vocabulary = _validate_vocabulary(input_root, manifests)
    feature_inventory, feature_files = _feature_inventory(manifests)
    e0_attention: dict[str, Any] = {}
    for seed in SEEDS:
        path = input_root / "attention" / f"e0_seed{seed}.h5"
        e0_attention[str(seed)] = _validate_attention_file(
            path,
            manifests["e0"],
            seed=seed,
            feature_files=feature_files,
            expected_checkpoint=None,
            deep=deep_attention,
        )
    checkpoints: dict[tuple[str, int], Path] = {}
    refits: dict[str, Any] = {}
    for target in ("RIH", "SurGen"):
        for seed in SEEDS:
            evidence = _validate_checkpoint(aim2, target, seed)
            checkpoints[(target, seed)] = Path(evidence["checkpoint"]["path"])
            refits[f"{target}/seed{seed}"] = evidence
    lineage = aim2 / "lineage_start.json"
    if not lineage.is_file():
        raise RuntimeError(f"Aim 2 root has no lineage_start.json: {aim2}")
    lineage_payload = _json_object(lineage)
    lineage_name = str(lineage_payload.get("lineage", ""))
    if (
        lineage_payload.get("status") != "started"
        or not lineage_name.startswith("aim2_cap8192_v4_")
        or Path(str(lineage_payload.get("lineage_root", ""))).resolve(strict=True) != aim2
    ):
        raise RuntimeError("--aim2-root is not a self-identifying Aim 2 cap8192 v4 lineage")
    refit_lineages = {
        str(evidence["fit_summary"]["path"]): _json_object(
            Path(str(evidence["fit_summary"]["path"]))
        ).get("lineage")
        for evidence in refits.values()
    }
    if set(refit_lineages.values()) != {lineage_name}:
        raise RuntimeError("Aim 2 refits do not all belong to the supplied v4 lineage")
    receipt = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "input_e4_root": str(input_root),
        "aim2_root": str(aim2),
        "protocol": {
            "cap": CAP,
            "k": K,
            "seeds": list(SEEDS),
            "target_attention_files_to_reexport": len(TARGET_ARMS) * len(SEEDS),
            "reused_e0_attention_files": len(SEEDS),
        },
        "manifest_files": manifest_files,
        "manifest_counts": {
            name: {
                "slides": len(frame),
                "patients": int(frame["patient_id"].nunique()),
                "mutant_patients": int(frame.drop_duplicates("patient_id")["target_label"].sum()),
            }
            for name, frame in manifests.items()
        },
        "vocabulary": vocabulary,
        "legacy_e0_attention": e0_attention,
        "aim2_lineage_start": _identity(lineage),
        "aim2_refits": refits,
        "feature_store": feature_inventory,
        "validation": {
            "status": "PASS",
            "deep_attention_validation": bool(deep_attention),
            "vocabulary_is_label_blind_k32": True,
            "target_refits_are_cap8192_patient_natural_unweighted_full_bag": True,
            "exact_arm_and_slide_coverage": True,
        },
    }
    return InputBundle(input_root, aim2, manifests, feature_files, checkpoints, receipt)


def _same_content_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        int(left.get("size_bytes", -1)), str(left.get("sha256", ""))
    ) == (
        int(right.get("size_bytes", -1)), str(right.get("sha256", ""))
    )


def validate_archival_inputs(
    input_e4_root: Path,
    aim2_root: Path,
    output: Path,
    *,
    deep_attention: bool = True,
) -> InputBundle:
    """Validate live non-imported inputs and frozen local copies after sealing.

    The legacy E4 root supplied on the command line is authenticated by the
    prepared receipt but need not retain its future bytes: its only consumed
    artifacts (vocabulary and E0 attention) were imported into ``output``.
    Aim2 checkpoints, manifests, and the feature store were not imported and
    therefore remain live-identity checked.
    """
    recorded = _json_object(output / "receipts" / "inputs.json")
    recorded_input = Path(str(recorded.get("input_e4_root", ""))).resolve(strict=False)
    if recorded_input != input_e4_root.resolve(strict=False):
        raise RuntimeError("--input-e4-root differs from the prepared lineage receipt")
    candidate = validate_inputs(output, aim2_root, deep_attention=deep_attention)
    for key in (
        "schema_version", "component", "aim2_root", "protocol", "manifest_files",
        "manifest_counts", "aim2_lineage_start", "aim2_refits", "feature_store",
        "validation",
    ):
        if _canonical(candidate.receipt.get(key)) != _canonical(recorded.get(key)):
            raise RuntimeError(f"archival non-imported input identity differs: {key}")
    candidate_vocabulary = candidate.receipt["vocabulary"]
    recorded_vocabulary = recorded.get("vocabulary", {})
    for key in ("vocab_npz", "vocab_json", "sample_plan"):
        if not _same_content_identity(candidate_vocabulary.get(key, {}), recorded_vocabulary.get(key, {})):
            raise RuntimeError(f"frozen imported vocabulary differs: {key}")
    for key in ("n_plan_slides", "n_sample_tiles"):
        if candidate_vocabulary.get(key) != recorded_vocabulary.get(key):
            raise RuntimeError(f"frozen imported vocabulary metadata differs: {key}")
    for seed in map(str, SEEDS):
        candidate_attention = candidate.receipt["legacy_e0_attention"][seed]
        recorded_attention = recorded.get("legacy_e0_attention", {}).get(seed, {})
        for key in ("h5", "summary"):
            if not _same_content_identity(
                candidate_attention.get(key, {}), recorded_attention.get(key, {})
            ):
                raise RuntimeError(f"frozen imported E0 attention differs: seed{seed}/{key}")
        if candidate_attention.get("n_slides") != recorded_attention.get("n_slides"):
            raise RuntimeError(f"frozen imported E0 attention count differs: seed{seed}")
    return InputBundle(
        input_root=recorded_input,
        aim2_root=candidate.aim2_root,
        manifests=candidate.manifests,
        feature_files=candidate.feature_files,
        checkpoints=candidate.checkpoints,
        receipt=recorded,
    )


def _validate_recorded_identity(recorded: dict[str, Any]) -> None:
    if not isinstance(recorded, dict) or set(recorded) != {"path", "size_bytes", "sha256"}:
        raise RuntimeError("malformed recorded artifact identity")
    actual = _identity(Path(str(recorded["path"])))
    if actual != recorded:
        raise RuntimeError(f"recorded artifact changed: {recorded.get('path')}")


def _validate_source_snapshot(output: Path, *, require_live_source: bool) -> None:
    receipt = _json_object(output / "receipts" / "source_snapshot.json")
    rows = receipt.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("source snapshot receipt has no file inventory")
    recorded_relative = {str(row.get("relative_path", "")) for row in rows}
    if not recorded_relative or "aim4_morphologic_atlas.py" not in recorded_relative:
        raise RuntimeError("source snapshot inventory is incomplete")
    if require_live_source:
        expected_relative = {str(path.relative_to(REPO)) for path in _source_files()}
        if recorded_relative != expected_relative:
            raise RuntimeError("live/snapshotted source inventory differs")
    for row in rows:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe source snapshot relative path: {relative}")
        snapshot = _identity(output / "source_snapshot" / relative)
        recorded_live = row.get("source")
        recorded_snapshot = row.get("imported")
        if snapshot != recorded_snapshot:
            raise RuntimeError(f"frozen source snapshot changed: {relative}")
        if not isinstance(recorded_live, dict) or (
            recorded_live.get("size_bytes"), recorded_live.get("sha256")
        ) != (snapshot["size_bytes"], snapshot["sha256"]):
            raise RuntimeError(f"source receipt does not authenticate snapshot: {relative}")
        if require_live_source:
            live = _identity(REPO / relative)
            if live != recorded_live:
                raise RuntimeError(f"source changed after run preparation: {relative}")


def _configure_legacy(
    output: Path,
    bundle: InputBundle,
    *,
    eval_root: Path | None = None,
    profile_root: Path | None = None,
) -> None:
    legacy.E3B_ROOT = output
    legacy.ATTN_DIR = output / "attention"
    legacy.VOCAB_DIR = output / "vocabulary"
    legacy.PROFILE_DIR = profile_root or output / "profiles"
    legacy.MONTAGE_DIR = output / "unblinding_keys"
    legacy.REVIEW_ROOT = output / "review_packets"
    legacy.paths.EVAL_ROOT = eval_root or output / "analysis"

    def corrected_checkpoint(target: str, seed: int) -> Path:
        try:
            return bundle.checkpoints[(target, int(seed))]
        except KeyError as error:
            raise SystemExit(f"unbound corrected checkpoint: {target} seed{seed}") from error

    legacy.refit_checkpoint = corrected_checkpoint
    legacy._effect_table = fixed_effect_table
    legacy._auc_difference_table = fixed_auc_difference_table


@contextlib.contextmanager
def _legacy_configuration(
    output: Path,
    bundle: InputBundle,
    *,
    eval_root: Path | None = None,
    profile_root: Path | None = None,
) -> Iterable[None]:
    module_attributes = {
        name: getattr(legacy, name)
        for name in (
            "E3B_ROOT",
            "ATTN_DIR",
            "VOCAB_DIR",
            "PROFILE_DIR",
            "MONTAGE_DIR",
            "REVIEW_ROOT",
            "refit_checkpoint",
            "_effect_table",
            "_auc_difference_table",
        )
    }
    old_eval_root = legacy.paths.EVAL_ROOT
    _configure_legacy(
        output,
        bundle,
        eval_root=eval_root,
        profile_root=profile_root,
    )
    try:
        yield
    finally:
        for name, value in module_attributes.items():
            setattr(legacy, name, value)
        legacy.paths.EVAL_ROOT = old_eval_root


def _validate_prepared(
    output: Path, bundle: InputBundle, *, require_live_source: bool = True
) -> dict[str, Any]:
    start_path = output / "lineage_start.json"
    inputs_path = output / "receipts" / "inputs.json"
    imports_path = output / "receipts" / "imports.json"
    for path in (start_path, inputs_path, imports_path):
        if not path.is_file():
            raise RuntimeError(f"output root is not prepared: missing {path}")
    start = _json_object(start_path)
    if start.get("status") != "prepared" or start.get("output_root") != str(output):
        raise RuntimeError("invalid prepared lineage receipt")
    if _canonical(_json_object(inputs_path)) != _canonical(bundle.receipt):
        raise RuntimeError("current input identities differ from the prepared run")
    _validate_source_snapshot(output, require_live_source=require_live_source)
    imports = _json_object(imports_path)
    for evidence in imports.get("files", {}).values():
        # Mutable upstream copies are required to retain their prepared bytes
        # while a lineage is being built.  Once sealed, the imported copy and
        # its source receipt are the archive: verification must not depend on
        # the legacy tree continuing to exist forever.
        if require_live_source:
            _validate_recorded_identity(evidence["source"])
        _validate_recorded_identity(evidence["imported"])
        if (
            evidence["source"]["size_bytes"], evidence["source"]["sha256"]
        ) != (
            evidence["imported"]["size_bytes"], evidence["imported"]["sha256"]
        ):
            raise RuntimeError("imported frozen artifact no longer matches its source")
    return start


@contextlib.contextmanager
def _run_lock(output: Path, *, exclusive: bool) -> Iterable[None]:
    """Serialize stage mutation and prevent verification during a half-written stage."""
    lock_path = output / ".aim4_corrected.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _assert_not_completed(output: Path) -> None:
    if (output / "lineage_complete.json").exists():
        raise RuntimeError(
            f"completed Aim 4 lineage is immutable; use a new output root: {output}"
        )


def _assert_numeric_not_completed(output: Path) -> None:
    _assert_not_completed(output)
    if (output / NUMERIC_COMPLETION_NAME).exists():
        raise RuntimeError(
            f"numeric Aim 4 lineage is immutable; use a new output root: {output}"
        )


def prepare_run(bundle: InputBundle, output_root: Path) -> Path:
    output = _absolute_dir(output_root, "--output-root", exists=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output root: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent must already exist: {output.parent}")
    output.mkdir(mode=0o755, exist_ok=False)
    _write_bytes_once(output / ".aim4_corrected.lock", b"")
    _snapshot_sources(output)
    _write_json_once(output / "receipts" / "inputs.json", bundle.receipt)
    imports: dict[str, Any] = {}
    reusable = {
        f"vocabulary/vocab_k{K}.npz": bundle.input_root / "vocabulary" / f"vocab_k{K}.npz",
        f"vocabulary/vocab_k{K}.json": bundle.input_root / "vocabulary" / f"vocab_k{K}.json",
        f"vocabulary/sample_plan_k{K}.parquet": (
            bundle.input_root / "vocabulary" / f"sample_plan_k{K}.parquet"
        ),
    }
    for seed in SEEDS:
        reusable[f"attention/e0_seed{seed}.h5"] = bundle.input_root / "attention" / f"e0_seed{seed}.h5"
        reusable[f"attention/e0_seed{seed}.json"] = bundle.input_root / "attention" / f"e0_seed{seed}.json"
    for relative, source in reusable.items():
        imports[relative] = _copy_once(source, output / relative)
    _write_json_once(
        output / "receipts" / "imports.json",
        {
            "schema_version": 1,
            "component": "aim4_corrected_cap8192",
            "policy": "byte-for-byte verified import; legacy source remains unchanged",
            "files": imports,
        },
    )
    _write_json_once(
        output / "lineage_start.json",
        {
            "schema_version": 1,
            "component": "aim4_corrected_cap8192",
            "status": "prepared",
            "created_at_utc": _utc_now(),
            "output_root": str(output),
            "input_e4_root": str(bundle.input_root),
            "aim2_root": str(bundle.aim2_root),
            "source_snapshot_receipt": _identity(output / "receipts" / "source_snapshot.json"),
        },
    )
    return output


def _validate_resumable_target_attention_stage(
    path: Path,
    manifest: pd.DataFrame,
    *,
    seed: int,
    feature_files: dict[str, Path],
    expected_checkpoint: Path,
) -> None:
    """Accept only complete per-slide groups before resuming an HDF5 append."""
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("staged target attention is not a regular file")
    expected = set(manifest["slide_id"].astype(str))
    with h5py.File(path, "r") as handle:
        actual = set(map(str, handle.keys()))
        if not actual <= expected:
            raise RuntimeError("staged target attention contains unexpected slides")
        for slide_id in sorted(actual):
            group = handle[slide_id]
            if set(group.keys()) != {"attention", "coords", "top_decile"}:
                raise RuntimeError(f"{slide_id}: interrupted attention group is incomplete")
            n = int(group.attrs.get("n_tiles", -1))
            if (
                n <= 0
                or group["attention"].shape != (n,)
                or group["coords"].shape != (n, 2)
                or group["top_decile"].shape != (n,)
                or int(group.attrs.get("seed", -1)) != int(seed)
                or str(group.attrs.get("mode", "")) != "refit"
                or Path(str(group.attrs.get("checkpoint", ""))).resolve(strict=True)
                != expected_checkpoint.resolve(strict=True)
            ):
                raise RuntimeError(f"{slide_id}: interrupted attention group contract differs")
            weights = group["attention"][:].astype(float)
            coords = group["coords"][:]
            mask = group["top_decile"][:]
            if (
                not np.isfinite(weights).all()
                or (weights < 0).any()
                or not np.isclose(weights.sum(), 1.0, atol=2e-5, rtol=2e-5)
                or not np.issubdtype(mask.dtype, np.bool_)
            ):
                raise RuntimeError(f"{slide_id}: interrupted attention values are invalid")
            boolean_mask = mask.astype(bool)
            minimum_count = max(1, int(np.ceil(0.10 * n)))
            if boolean_mask.sum() < minimum_count or (
                (~boolean_mask).any()
                and float(weights[boolean_mask].min())
                < float(weights[~boolean_mask].max()) - 1e-12
            ):
                raise RuntimeError(f"{slide_id}: interrupted top-decile mask is invalid")
            with h5py.File(feature_files[slide_id], "r") as features:
                if features["features"].shape[0] != n or not np.array_equal(
                    coords, features["coords"][:]
                ):
                    raise RuntimeError(
                        f"{slide_id}: interrupted attention tile order differs"
                    )


def _archive_interrupted_attention_stage(
    stage: Path, *, include_h5: bool = True
) -> None:
    """Preserve, rather than delete or replace, an unusable staged pair."""
    summary = stage.with_suffix(".json")
    candidates = (stage, summary) if include_h5 else (summary,)
    existing = [path for path in candidates if path.exists() or path.is_symlink()]
    if not existing:
        return
    archive = stage.parent / "interrupted" / uuid.uuid4().hex
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        os.rename(path, archive / path.name)


def export_target_attention(
    bundle: InputBundle,
    output: Path,
    *,
    arms: Iterable[str] = TARGET_ARMS,
    seeds: Iterable[int] = SEEDS,
    apply: bool,
) -> dict[str, Any]:
    _validate_prepared(output, bundle)
    _assert_numeric_not_completed(output)
    selected_arms = tuple(arms)
    selected_seeds = tuple(int(seed) for seed in seeds)
    if set(selected_arms) - set(TARGET_ARMS) or set(selected_seeds) - set(SEEDS):
        raise ValueError("attention export is restricted to the four target arms and seeds 42-44")
    plan: list[dict[str, Any]] = []
    for arm_name in selected_arms:
        target = ARM_TARGET[arm_name]
        for seed in selected_seeds:
            destination = output / "attention" / f"{arm_name}_seed{seed}.h5"
            plan.append({
                "arm": arm_name,
                "seed": seed,
                "slides": len(bundle.manifests[arm_name]),
                "checkpoint": str(bundle.checkpoints[(target, seed)]),
                "destination": str(destination),
            })
    if not apply:
        return {"status": "DRY_RUN", "exports": plan}
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "corrected target-attention export requires an available CUDA GPU; "
            "refusing an accidental multi-hour CPU fallback"
        )
    for item in plan:
        arm_name = str(item["arm"])
        seed = int(item["seed"])
        target = ARM_TARGET[arm_name]
        checkpoint = bundle.checkpoints[(target, seed)]
        destination = Path(str(item["destination"]))
        summary_destination = destination.with_suffix(".json")
        stage_dir = output / "_staging" / "attention"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage = stage_dir / destination.name
        stage_summary = stage.with_suffix(".json")
        # Recover a crash between the two canonical renames. Neither canonical
        # member is replaced; the already-validated staged mate is promoted.
        if destination.is_file() and not summary_destination.exists() and stage_summary.is_file():
            os.rename(stage_summary, summary_destination)
        if summary_destination.is_file() and not destination.exists() and stage.is_file():
            os.rename(stage, destination)
        if destination.is_file() and summary_destination.is_file():
            _validate_attention_file(
                destination,
                bundle.manifests[arm_name],
                seed=seed,
                feature_files=bundle.feature_files,
                expected_checkpoint=checkpoint,
                deep=True,
            )
            print(f"{arm_name} seed{seed}: complete and verified; skipping")
            continue
        if destination.exists() or summary_destination.exists():
            raise RuntimeError(f"unrecoverable half-committed attention pair: {destination}")
        if stage.is_file() and stage_summary.is_file():
            try:
                _validate_attention_file(
                    stage,
                    bundle.manifests[arm_name],
                    seed=seed,
                    feature_files=bundle.feature_files,
                    expected_checkpoint=checkpoint,
                    deep=True,
                )
            except Exception:  # noqa: BLE001
                pass
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.rename(stage, destination)
                os.rename(stage_summary, summary_destination)
                print(f"{arm_name} seed{seed}: recovered complete staged export")
                continue
        if stage.exists() or stage.is_symlink():
            try:
                _validate_resumable_target_attention_stage(
                    stage,
                    bundle.manifests[arm_name],
                    seed=seed,
                    feature_files=bundle.feature_files,
                    expected_checkpoint=checkpoint,
                )
            except Exception as error:  # noqa: BLE001
                _archive_interrupted_attention_stage(stage)
                print(
                    f"{arm_name} seed{seed}: preserved unusable interrupted stage "
                    f"and restarting it ({error})"
                )
        # The exporter regenerates its summary after a resumable HDF5 append.
        # Preserve any older staged summary so no staging file is overwritten.
        if stage_summary.exists() or stage_summary.is_symlink():
            _archive_interrupted_attention_stage(stage, include_h5=False)
        print(f"{arm_name} seed{seed}: exporting {len(bundle.manifests[arm_name])} slides")
        attention.export_attention_refit(
            checkpoint=checkpoint,
            slide_ids=bundle.manifests[arm_name]["slide_id"].astype(str).tolist(),
            destination=stage,
            feature_dir=paths.PINNED_FEATURE_DIR,
            top_fraction=0.10,
            seed=seed,
        )
        _validate_attention_file(
            stage,
            bundle.manifests[arm_name],
            seed=seed,
            feature_files=bundle.feature_files,
            expected_checkpoint=checkpoint,
            deep=True,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or summary_destination.exists():
            raise FileExistsError(f"canonical attention destination appeared during export: {destination}")
        os.rename(stage, destination)
        os.rename(stage.with_suffix(".json"), summary_destination)
        print(f"{arm_name} seed{seed}: verified -> {destination}")
    receipt_path = output / "receipts" / "target_attention.json"
    if not receipt_path.exists():
        all_evidence: dict[str, Any] = {}
        complete = True
        for arm_name in TARGET_ARMS:
            for seed in SEEDS:
                path = output / "attention" / f"{arm_name}_seed{seed}.h5"
                if not path.is_file():
                    complete = False
                    continue
                checkpoint = bundle.checkpoints[(ARM_TARGET[arm_name], seed)]
                all_evidence[f"{arm_name}/seed{seed}"] = _validate_attention_file(
                    path,
                    bundle.manifests[arm_name],
                    seed=seed,
                    feature_files=bundle.feature_files,
                    expected_checkpoint=checkpoint,
                    deep=True,
                )
        if complete:
            _write_json_once(
                receipt_path,
                {
                    "schema_version": 1,
                    "component": "aim4_corrected_cap8192",
                    "status": "PASS",
                    "n_target_attention_files": 12,
                    "artifacts": all_evidence,
                },
            )
    return {"status": "PASS", "exports": plan}


def _assert_exact_prototype_rows(
    frame: pd.DataFrame,
    *,
    unit: str,
    unit_columns: list[str],
    expected_units: set[tuple[Any, ...]] | None = None,
) -> None:
    required = {*unit_columns, "prototype", *QUANTITIES}
    if missing := required - set(frame):
        raise RuntimeError(f"{unit}: profile columns missing: {sorted(missing)}")
    if frame.duplicated([*unit_columns, "prototype"]).any():
        raise RuntimeError(f"{unit}: duplicate unit/prototype rows")
    prototypes = pd.to_numeric(frame["prototype"], errors="coerce")
    if prototypes.isna().any() or set(prototypes.astype(int).unique()) != set(range(K)):
        raise RuntimeError(f"{unit}: global prototype inventory is not exactly 0..31")
    grouped = frame.groupby(unit_columns, sort=False)["prototype"]
    sizes = grouped.size()
    distinct = grouped.nunique()
    if (sizes != K).any() or (distinct != K).any():
        raise RuntimeError(f"{unit}: at least one unit lacks an exact 32-prototype block")
    if expected_units is not None:
        actual_units = set(map(tuple, frame[unit_columns].drop_duplicates().itertuples(index=False, name=None)))
        if actual_units != expected_units:
            raise RuntimeError(
                f"{unit}: unit coverage mismatch; missing={len(expected_units-actual_units)}, "
                f"unexpected={len(actual_units-expected_units)}"
            )
    for quantity in QUANTITIES:
        values = pd.to_numeric(frame[quantity], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{unit}: {quantity} contains non-finite values")


def validate_profiles(
    output: Path,
    bundle: InputBundle,
    *,
    deep_candidates: bool,
    profile_root: Path | None = None,
) -> dict[str, Any]:
    profiles = profile_root or output / "profiles"
    slide_path = profiles / f"slide_profiles_k{K}.parquet"
    patient_path = profiles / f"patient_profiles_k{K}.parquet"
    candidate_path = profiles / f"tile_candidates_k{K}.parquet"
    for path in (slide_path, patient_path, candidate_path):
        if not path.is_file():
            raise RuntimeError(f"profile artifact missing: {path}")
    slides = pd.read_parquet(slide_path)
    patients = pd.read_parquet(patient_path)
    candidates = pd.read_parquet(candidate_path)
    expected_slides = {
        (arm, str(slide_id))
        for arm, manifest in bundle.manifests.items()
        for slide_id in manifest["slide_id"]
    }
    expected_patients = {
        (arm, str(patient_id))
        for arm, manifest in bundle.manifests.items()
        for patient_id in manifest["patient_id"].unique()
    }
    _assert_exact_prototype_rows(
        slides, unit="slide", unit_columns=["arm", "slide_id"], expected_units=expected_slides
    )
    _assert_exact_prototype_rows(
        patients,
        unit="patient",
        unit_columns=["arm", "patient_id"],
        expected_units=expected_patients,
    )
    manifest_rows: list[pd.DataFrame] = []
    patient_manifest_rows: list[pd.DataFrame] = []
    for arm, manifest in bundle.manifests.items():
        context_columns = [
            column for column in (
                "slide_id", "patient_id", "target_label", "specimen_role", "cohort",
                "subcohort",
            ) if column in manifest
        ]
        context = manifest[context_columns].copy()
        context.insert(0, "arm", arm)
        context = context.rename(columns={"target_label": "expected_label",
                                          "specimen_role": "expected_role"})
        manifest_rows.append(context)
        patient_core = context.drop(columns=["slide_id"]).drop_duplicates()
        if patient_core["patient_id"].duplicated().any():
            raise RuntimeError(f"{arm}: core profile metadata varies within patient")
        slide_counts = (
            context.groupby(["arm", "patient_id"], as_index=False)["slide_id"]
            .count().rename(columns={"slide_id": "expected_n_slides"})
        )
        patient_manifest_rows.append(
            patient_core.merge(slide_counts, on=["arm", "patient_id"], validate="one_to_one")
        )
    manifest_context = pd.concat(manifest_rows, ignore_index=True)
    observed_slide_context = slides[
        ["arm", "slide_id", "patient_id", "label", "role", "cohort", "subcohort"]
    ].drop_duplicates()
    if len(observed_slide_context) != len(expected_slides):
        raise RuntimeError("slide profile context is not constant over prototypes")
    checked_slides = observed_slide_context.merge(
        manifest_context, on=["arm", "slide_id"], how="left", validate="one_to_one"
    )
    if checked_slides["expected_label"].isna().any() or not (
        checked_slides["patient_id_x"].astype(str).eq(checked_slides["patient_id_y"].astype(str)).all()
        and checked_slides["label"].astype(int).eq(checked_slides["expected_label"].astype(int)).all()
        and checked_slides["role"].astype(str).eq(checked_slides["expected_role"].astype(str)).all()
        and checked_slides["cohort_x"].astype(str).eq(checked_slides["cohort_y"].astype(str)).all()
        and checked_slides["subcohort_x"].astype(str).eq(checked_slides["subcohort_y"].astype(str)).all()
    ):
        raise RuntimeError("slide profile metadata differs from the frozen manifests")
    expected_patient_context = pd.concat(patient_manifest_rows, ignore_index=True)
    observed_patient_context = patients[
        ["arm", "patient_id", "label", "role", "cohort", "subcohort", "n_slides"]
    ].drop_duplicates()
    if len(observed_patient_context) != len(expected_patients):
        raise RuntimeError("patient profile context is not constant over prototypes")
    checked_patients = observed_patient_context.merge(
        expected_patient_context, on=["arm", "patient_id"], how="left", validate="one_to_one"
    )
    if checked_patients["expected_label"].isna().any() or not (
        checked_patients["label"].astype(int).eq(checked_patients["expected_label"].astype(int)).all()
        and checked_patients["role"].astype(str).eq(checked_patients["expected_role"].astype(str)).all()
        and checked_patients["cohort_x"].astype(str).eq(checked_patients["cohort_y"].astype(str)).all()
        and checked_patients["subcohort_x"].astype(str).eq(checked_patients["subcohort_y"].astype(str)).all()
        and checked_patients["n_slides"].astype(int).eq(
            checked_patients["expected_n_slides"].astype(int)
        ).all()
    ):
        raise RuntimeError("patient profile metadata differs from the frozen manifests")
    expected_patient_profiles = legacy._to_patient_level(slides)
    profile_keys = ["arm", "patient_id", "prototype"]
    if set(expected_patient_profiles) != set(patients):
        raise RuntimeError("patient profile schema differs from deterministic slide aggregation")
    observed_sorted = patients.sort_values(profile_keys).reset_index(drop=True)
    expected_sorted = expected_patient_profiles.sort_values(profile_keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            observed_sorted[sorted(observed_sorted.columns)],
            expected_sorted[sorted(expected_sorted.columns)],
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise RuntimeError(
            "patient profiles differ from deterministic equal-slide aggregation"
        ) from error
    slide_keys = ["arm", "slide_id"]
    for column in ("n_tiles_prototype", "n_tiles_slide"):
        numeric = pd.to_numeric(slides[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any() or not np.equal(
            numeric, np.floor(numeric)
        ).all():
            raise RuntimeError(f"slide {column} is not finite non-negative integer data")
    if (pd.to_numeric(slides["n_tiles_slide"], errors="raise") <= 0).any():
        raise RuntimeError("slide tile totals must be positive")
    if (slides.groupby(slide_keys)["n_tiles_slide"].nunique() != 1).any() or (
        pd.to_numeric(slides["n_tiles_prototype"], errors="raise")
        > pd.to_numeric(slides["n_tiles_slide"], errors="raise")
    ).any():
        raise RuntimeError("slide tile totals are inconsistent across prototype rows")
    tile_sums = slides.groupby(slide_keys)["n_tiles_prototype"].sum()
    tile_totals = slides.groupby(slide_keys)["n_tiles_slide"].first()
    if not np.array_equal(tile_sums.to_numpy(dtype=int), tile_totals.to_numpy(dtype=int)):
        raise RuntimeError("slide prototype tile counts do not sum to slide tile totals")
    expected_abundance = (
        slides["n_tiles_prototype"].to_numpy(dtype=float)
        / slides["n_tiles_slide"].to_numpy(dtype=float)
    )
    if not np.allclose(slides["abundance"].to_numpy(dtype=float), expected_abundance):
        raise RuntimeError("slide abundance is not prototype tiles divided by slide tiles")
    attention_columns = [f"attn_mass_seed{seed}" for seed in SEEDS]
    for label, frame, keys in (
        ("slide", slides, ["arm", "slide_id"]),
        ("patient", patients, ["arm", "patient_id"]),
    ):
        if set(attention_columns) - set(frame):
            raise RuntimeError(f"{label}: missing per-seed attention mass columns")
        if not (pd.to_numeric(frame["n_attention_seeds"], errors="coerce") == len(SEEDS)).all():
            raise RuntimeError(f"{label}: not every row averages exactly three attention seeds")
        for quantity in ["abundance", "attn_mass_mean", *attention_columns]:
            sums = frame.groupby(keys, sort=False)[quantity].sum().to_numpy(dtype=float)
            if not np.allclose(sums, 1.0, atol=2e-5, rtol=2e-5):
                raise RuntimeError(f"{label}: {quantity} does not sum to one within unit")
        seed_mean = frame[attention_columns].mean(axis=1).to_numpy(dtype=float)
        if not np.allclose(frame["attn_mass_mean"].to_numpy(dtype=float), seed_mean):
            raise RuntimeError(f"{label}: attention mean is not the exact three-seed mean")
    required_candidates = {
        "arm", "role", "slide_id", "patient_id", "cohort", "subcohort", "label",
        "prototype", "kind", "tile_index", "x", "y", "distance_to_centroid",
        "attention",
    }
    if missing := required_candidates - set(candidates):
        raise RuntimeError(f"candidate columns missing: {sorted(missing)}")
    key = ["arm", "slide_id", "prototype", "kind"]
    if candidates.duplicated(key).any() or set(candidates["kind"]) != {"medoid", "top_attention"}:
        raise RuntimeError("candidate keys/kinds are not exact")
    numeric_candidates = candidates[
        ["prototype", "tile_index", "x", "y", "distance_to_centroid", "attention"]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric_candidates).all():
        raise RuntimeError("candidate numeric fields contain non-finite values")
    for column in ("prototype", "tile_index", "x", "y"):
        values = pd.to_numeric(candidates[column], errors="raise").to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise RuntimeError(f"candidate {column} is not integer data")
    if (
        (pd.to_numeric(candidates["distance_to_centroid"], errors="raise") < 0).any()
        or not pd.to_numeric(candidates["attention"], errors="raise").between(0, 1).all()
    ):
        raise RuntimeError("candidate distance/attention is outside its valid range")
    candidate_context = candidates[
        ["arm", "slide_id", "patient_id", "role", "cohort", "subcohort", "label"]
    ].drop_duplicates()
    checked_candidates = candidate_context.merge(
        manifest_context, on=["arm", "slide_id"], how="left", validate="one_to_one"
    )
    if checked_candidates["expected_label"].isna().any() or not (
        checked_candidates["patient_id_x"].astype(str).eq(
            checked_candidates["patient_id_y"].astype(str)
        ).all()
        and checked_candidates["label"].astype(int).eq(
            checked_candidates["expected_label"].astype(int)
        ).all()
        and checked_candidates["role"].astype(str).eq(
            checked_candidates["expected_role"].astype(str)
        ).all()
        and checked_candidates["cohort_x"].astype(str).eq(
            checked_candidates["cohort_y"].astype(str)
        ).all()
        and checked_candidates["subcohort_x"].astype(str).eq(
            checked_candidates["subcohort_y"].astype(str)
        ).all()
    ):
        raise RuntimeError("candidate metadata differs from the frozen manifests")
    positive = slides.loc[slides["n_tiles_prototype"].gt(0), ["arm", "slide_id", "prototype"]]
    expected_candidates = {
        (*row, kind)
        for row in positive.itertuples(index=False, name=None)
        for kind in ("medoid", "top_attention")
    }
    actual_candidates = set(candidates[key].itertuples(index=False, name=None))
    if actual_candidates != expected_candidates:
        raise RuntimeError(
            "candidate coverage is not exactly two kinds for every non-empty slide/prototype"
        )
    if deep_candidates:
        for (arm_name, slide_id), block in candidates.groupby(
            ["arm", "slide_id"], sort=True
        ):
            with h5py.File(bundle.feature_files[str(slide_id)], "r") as handle:
                coords = handle["coords"][:]
            indices = pd.to_numeric(block["tile_index"], errors="coerce").to_numpy(dtype=int)
            if (indices < 0).any() or (indices >= len(coords)).any():
                raise RuntimeError(f"{slide_id}: candidate tile index is out of range")
            observed = block[["x", "y"]].to_numpy(dtype=int)
            if not np.array_equal(observed, coords[indices].astype(int)):
                raise RuntimeError(f"{slide_id}: candidate coordinates differ from feature order")
            attention_values: list[np.ndarray] = []
            for seed in SEEDS:
                with h5py.File(
                    output / "attention" / f"{arm_name}_seed{seed}.h5", "r"
                ) as handle:
                    attention_values.append(
                        handle[str(slide_id)]["attention"][:].astype(float)[indices]
                    )
            expected_attention = np.mean(attention_values, axis=0)
            if not np.allclose(
                block["attention"].to_numpy(dtype=float),
                expected_attention,
                rtol=1e-6,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"{arm_name}/{slide_id}: candidate attention differs from exact three-seed mean"
                )
    return {
        "status": "PASS",
        "slide_rows": len(slides),
        "patient_rows": len(patients),
        "candidate_rows": len(candidates),
        "slides": len(expected_slides),
        "arm_patients": len(expected_patients),
        "prototype_family": list(range(K)),
        "quantities": list(QUANTITIES),
        "attention_seeds": list(SEEDS),
        "deep_candidate_validation": bool(deep_candidates),
        "artifacts": {
            "slide_profiles": _identity(slide_path),
            "patient_profiles": _identity(patient_path),
            "tile_candidates": _identity(candidate_path),
        },
    }


def assign_profiles(bundle: InputBundle, output: Path, *, apply: bool) -> dict[str, Any]:
    _validate_prepared(output, bundle)
    _assert_numeric_not_completed(output)
    for arm_name in ALL_ARMS:
        for seed in SEEDS:
            path = output / "attention" / f"{arm_name}_seed{seed}.h5"
            checkpoint = None if arm_name == "e0" else bundle.checkpoints[(ARM_TARGET[arm_name], seed)]
            _validate_attention_file(
                path,
                bundle.manifests[arm_name],
                seed=seed,
                feature_files=bundle.feature_files,
                expected_checkpoint=checkpoint,
                deep=True,
            )
    expected = {
        "slide_rows": sum(len(frame) for frame in bundle.manifests.values()) * K,
        "patient_rows": sum(frame["patient_id"].nunique() for frame in bundle.manifests.values()) * K,
    }
    if not apply:
        return {"status": "DRY_RUN", **expected}
    canonical_profiles = output / "profiles"
    receipt_path = output / "receipts" / "profiles.json"
    # Recover the only permitted interruption boundary: the complete staged
    # directory was atomically promoted, but its small external receipt was not
    # yet published.  A partial canonical directory is never accepted.
    if canonical_profiles.exists() or receipt_path.exists():
        if not canonical_profiles.is_dir():
            raise RuntimeError("unrecoverable partial corrected profile campaign")
        validation = validate_profiles(output, bundle, deep_candidates=True)
        if receipt_path.is_file():
            if _canonical(_json_object(receipt_path)) != _canonical(validation):
                raise RuntimeError("canonical profiles and profile receipt disagree")
        else:
            _write_json_once(receipt_path, validation)
        return validation
    stage_parent = output / "_staging" / "profile_campaigns" / f"k{K}.{uuid.uuid4().hex}"
    stage_profiles = stage_parent / "profiles"
    with _legacy_configuration(output, bundle, profile_root=stage_profiles):
        legacy.cmd_assign(SimpleNamespace(
            k=K,
            arm=None,
            limit=None,
            seeds=None,
            allow_missing_attention=False,
        ))
    validate_profiles(
        output,
        bundle,
        deep_candidates=True,
        profile_root=stage_profiles,
    )
    if canonical_profiles.exists():
        raise FileExistsError("canonical profile directory appeared during assignment")
    os.rename(stage_profiles, canonical_profiles)
    validation = validate_profiles(output, bundle, deep_candidates=True)
    _write_json_once(receipt_path, validation)
    return validation


def _validate_effect_input(frame: pd.DataFrame, positive_column: str, quantity: str) -> None:
    required = {"patient_id", "prototype", positive_column, quantity}
    if missing := required - set(frame):
        raise RuntimeError(f"effect table input columns missing: {sorted(missing)}")
    if frame.duplicated(["patient_id", "prototype"]).any():
        raise RuntimeError("effect table input repeats patient/prototype rows")
    if not frame.empty:
        prototypes = set(pd.to_numeric(frame["prototype"], errors="raise").astype(int))
        if prototypes != set(range(K)):
            raise RuntimeError(
                f"effect table prototype family is incomplete: expected 0..31, got {sorted(prototypes)}"
            )
        by_prototype = frame.groupby("prototype")["patient_id"].agg(list)
        reference = set(map(str, by_prototype.iloc[0]))
        if any(set(map(str, values)) != reference for values in by_prototype.iloc[1:]):
            raise RuntimeError("effect table prototypes do not cover identical patients")
        labels = pd.to_numeric(frame[positive_column], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(frame[quantity], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
            raise RuntimeError("effect labels are not finite and binary")
        if not np.isfinite(values).all():
            raise RuntimeError(f"effect quantity {quantity} is non-finite")


def fixed_effect_table(
    frame: pd.DataFrame,
    positive_column: str,
    quantity: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Exactly 32 BH hypotheses; structural non-estimability receives p=1."""
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be positive")
    _validate_effect_input(frame, positive_column, quantity)
    rows: list[dict[str, Any]] = []
    for prototype in range(K):
        block = frame[frame["prototype"].eq(prototype)]
        labels = pd.to_numeric(block[positive_column], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(block[quantity], errors="coerce").to_numpy(dtype=float)
        estimable = bool(
            len(block) > 0
            and np.isfinite(labels).all()
            and np.isfinite(values).all()
            and set(np.unique(labels)) == {0.0, 1.0}
        )
        if estimable:
            stats = atlas.bootstrap_auc_effect(
                values, labels, n_bootstrap=int(n_bootstrap), seed=int(seed) + prototype
            )
            p_value = atlas.mannwhitney_p(values, labels)
            estimable = bool(np.isfinite([
                stats["auc"], stats["ci_low"], stats["ci_high"], p_value
            ]).all())
        else:
            stats = {
                "auc": float("nan"), "delta": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n": int(len(block)), "n_positive": int(np.nansum(labels)) if len(labels) else 0,
            }
            p_value = 1.0
        if not estimable:
            p_value = 1.0
        rows.append({
            **stats,
            "prototype": prototype,
            "p": float(p_value),
            "estimable": estimable,
            "structural_p_policy": None if estimable else "p=1_in_fixed_32_family",
        })
    out = pd.DataFrame(rows).sort_values("prototype").reset_index(drop=True)
    out["q"] = atlas.benjamini_hochberg(out["p"].to_numpy(dtype=float))
    if len(out) != K or not np.isfinite(out[["p", "q"]].to_numpy(dtype=float)).all():
        raise RuntimeError("fixed 32-prototype BH family was not formed")
    out["significant"] = out["estimable"] & (out["q"] < FDR_ALPHA) & (
        (out["ci_low"] > 0.5) | (out["ci_high"] < 0.5)
    )
    return out


def fixed_auc_difference_table(
    primary: pd.DataFrame,
    metastatic: pd.DataFrame,
    quantity: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Exactly 32 direct P-v-M hypotheses with structural p=1 rows retained."""
    _validate_effect_input(primary, "label", quantity)
    _validate_effect_input(metastatic, "label", quantity)
    rows: list[dict[str, Any]] = []
    for prototype in range(K):
        p_block = primary[primary["prototype"].eq(prototype)]
        m_block = metastatic[metastatic["prototype"].eq(prototype)]
        p_labels = p_block["label"].to_numpy(dtype=int)
        m_labels = m_block["label"].to_numpy(dtype=int)
        estimable = bool(
            len(p_block) and len(m_block)
            and set(np.unique(p_labels)) == {0, 1}
            and set(np.unique(m_labels)) == {0, 1}
        )
        if estimable:
            row = legacy._auc_difference(
                p_block[quantity].to_numpy(dtype=float),
                p_labels,
                m_block[quantity].to_numpy(dtype=float),
                m_labels,
                n_bootstrap=int(n_bootstrap),
                seed=int(seed) + prototype,
            )
            estimable = bool(np.isfinite([
                row["delta_auc_metastatic_minus_primary"],
                row["delta_ci_low"], row["delta_ci_high"], row["delta_p"],
            ]).all())
        else:
            row = {
                "delta_auc_metastatic_minus_primary": float("nan"),
                "delta_ci_low": float("nan"),
                "delta_ci_high": float("nan"),
                "delta_se": float("nan"),
                "delta_p": 1.0,
                "n_bootstrap_valid": 0,
            }
        if not estimable:
            row["delta_p"] = 1.0
        rows.append({
            **row,
            "prototype": prototype,
            "estimable": estimable,
            "structural_p_policy": None if estimable else "p=1_in_fixed_32_family",
        })
    out = pd.DataFrame(rows).sort_values("prototype").reset_index(drop=True)
    out["delta_q"] = atlas.benjamini_hochberg(out["delta_p"].to_numpy(dtype=float))
    if len(out) != K or not np.isfinite(out[["delta_p", "delta_q"]].to_numpy()).all():
        raise RuntimeError("fixed 32-prototype direct-contrast family was not formed")
    out["changed"] = out["estimable"] & (out["delta_q"] < FDR_ALPHA) & (
        (out["delta_ci_low"] > 0) | (out["delta_ci_high"] < 0)
    )
    return out


def _assert_effect_records(records: Any, label: str, *, delta: bool = False) -> None:
    if not isinstance(records, list) or len(records) != K:
        raise RuntimeError(f"{label}: expected exactly 32 effect rows")
    prototypes = [int(row.get("prototype", -1)) for row in records]
    if prototypes != list(range(K)):
        raise RuntimeError(f"{label}: prototype row order/inventory mismatch")
    p_key, q_key = ("delta_p", "delta_q") if delta else ("p", "q")
    for row in records:
        if p_key not in row or q_key not in row or "estimable" not in row:
            raise RuntimeError(f"{label}: missing multiplicity/estimability fields")
        if not np.isfinite(float(row[p_key])) or not np.isfinite(float(row[q_key])):
            raise RuntimeError(f"{label}: p/q is non-finite")
        if not row["estimable"] and (
            float(row[p_key]) != 1.0 or bool(row.get("significant", row.get("changed", False)))
        ):
            raise RuntimeError(f"{label}: structural row was not forced to p=1/non-significant")


def _assert_merged_transport_records(records: Any, label: str) -> None:
    """Validate the two single-arm families and direct-contrast family after merge."""
    if not isinstance(records, list) or len(records) != K:
        raise RuntimeError(f"{label}: expected exactly 32 merged transport rows")
    if [int(row.get("prototype", -1)) for row in records] != list(range(K)):
        raise RuntimeError(f"{label}: merged prototype row order/inventory mismatch")
    for side in ("primary", "metastatic"):
        for row in records:
            required = {f"p_{side}", f"q_{side}", f"estimable_{side}", f"significant_{side}"}
            if missing := required - set(row):
                raise RuntimeError(f"{label}/{side}: missing fields {sorted(missing)}")
            p_value = float(row[f"p_{side}"])
            q_value = float(row[f"q_{side}"])
            if not np.isfinite([p_value, q_value]).all():
                raise RuntimeError(f"{label}/{side}: p/q is non-finite")
            if not row[f"estimable_{side}"] and (
                p_value != 1.0 or bool(row[f"significant_{side}"])
            ):
                raise RuntimeError(f"{label}/{side}: structural row is not p=1/non-significant")
    _assert_effect_records(records, f"{label}/direct_delta", delta=True)


def validate_analysis(spec: dict[str, Any], transport: dict[str, Any]) -> dict[str, Any]:
    if (
        spec.get("component") != "aim4_corrected_cap8192"
        or transport.get("component") != "aim4_corrected_cap8192"
        or int(spec.get("k", -1)) != K
        or int(transport.get("k", -1)) != K
        or set(spec.get("prototypes", {})) != set(map(str, range(K)))
    ):
        raise RuntimeError("specificity prototype inventory mismatch")
    protocol = spec.get("protocol")
    if not isinstance(protocol, dict) or _canonical(protocol) != _canonical(
        transport.get("protocol")
    ):
        raise RuntimeError("specificity/transport protocol mismatch")
    expected_protocol = {
        "cap": CAP,
        "k": K,
        "seeds": list(SEEDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "inference_unit": "patient",
        "multiplicity": "BH independently within each quantity x population/arm/contrast",
        "bh_family": "fixed prototypes 0..31, including structural p=1 rows",
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()) or int(
        protocol.get("n_bootstrap", -1)
    ) < DEFAULT_BOOTSTRAP:
        raise RuntimeError("corrected analysis protocol is not the locked cap8192/k32 contract")
    for quantity in QUANTITIES:
        for panel in ("A", "D", "context_in_wt"):
            records = [spec["prototypes"][str(p)][quantity][panel] for p in range(K)]
            _assert_effect_records(records, f"specificity/{quantity}/{panel}")
        cohort_names = set(spec["prototypes"]["0"]["by_cohort"][quantity])
        if cohort_names != {"CPTAC", "RIH", "SurGen", "TCGA"}:
            raise RuntimeError(f"specificity/{quantity}: per-cohort inventory mismatch")
        for cohort in sorted(cohort_names):
            records = [spec["prototypes"][str(p)]["by_cohort"][quantity][cohort] for p in range(K)]
            _assert_effect_records(records, f"specificity/{quantity}/{cohort}")
    expected_transport_arms = set(TARGET_ARMS)
    if set(transport.get("arms", {})) != expected_transport_arms:
        raise RuntimeError("transport standalone-arm inventory mismatch")
    for arm in TARGET_ARMS:
        for quantity in QUANTITIES:
            _assert_effect_records(
                transport["arms"][arm]["effects"][quantity], f"transport/{arm}/{quantity}"
            )
    if set(transport.get("conservation", {})) != {"RIH", "SR1482"}:
        raise RuntimeError("transport conservation-cohort inventory mismatch")
    for cohort in ("RIH", "SR1482"):
        for quantity in QUANTITIES:
            records = transport["conservation"][cohort]["by_quantity"][quantity]
            _assert_merged_transport_records(records, f"transport/{cohort}/{quantity}")
    return {
        "status": "PASS",
        "fixed_bh_family_size": K,
        "specificity_panels": 14,
        "transport_effect_panels": 16,
        "transport_direct_contrast_panels": 4,
        "structural_policy": "retain row; p=1; q includes all 32; never significant",
    }


def _report_table(
    readout: dict[int, dict[str, Any]], selected_for_review: set[int]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prototype in range(K):
        row = readout[prototype]
        abundance = row["specificity_effects"]["abundance"]
        attn = row["specificity_effects"]["attn_mass_mean"]
        rows.append({
            "prototype": prototype,
            "group": row["group"],
            "specificity_bucket": row["specificity_bucket"],
            "abundance_auc_A": abundance["A"].get("auc"),
            "abundance_ci_low_A": abundance["A"].get("ci_low"),
            "abundance_ci_high_A": abundance["A"].get("ci_high"),
            "abundance_q_A": abundance["A"].get("q"),
            "abundance_significant_A": abundance["A"].get("significant"),
            "abundance_auc_D": abundance["D"].get("auc"),
            "abundance_ci_low_D": abundance["D"].get("ci_low"),
            "abundance_ci_high_D": abundance["D"].get("ci_high"),
            "abundance_q_D": abundance["D"].get("q"),
            "abundance_significant_D": abundance["D"].get("significant"),
            "attention_auc_A": attn["A"].get("auc"),
            "attention_q_A": attn["A"].get("q"),
            "attention_significant_A": attn["A"].get("significant"),
            "persists_in_D": row["persists_in_D"],
            "transport_state": row["conservation"]["state"],
            "shortcut_flags": "; ".join(row["shortcut_flags"]),
            "organ_flags_metastatic_only": "; ".join(row["organ_flags"]),
            "pathology_status": (
                "pending_new_corrected_packet"
                if prototype in selected_for_review
                else "not_selected"
            ),
        })
    return pd.DataFrame(rows)


def _corrected_review_selection(
    readout: dict[int, dict[str, Any]],
) -> dict[str, list[int]]:
    """Return selection from an explicitly numeric-ordered k32 readout.

    This normalization is intentionally defensive: analysis holds integer keys
    in numeric order, while strict JSON reloads can expose lexicographically
    ordered string keys before :func:`legacy.final_readout` rebuilds the rows.
    Equal shortcut-dominance scores must never make packet membership depend on
    either representation's insertion order.
    """
    if set(readout) != set(range(K)):
        raise RuntimeError("review selection requires exactly prototypes 0..31")
    ordered = {prototype: readout[prototype] for prototype in range(K)}
    return legacy.review_prototypes(ordered)


def validate_corrected_report(
    output: Path,
    spec: dict[str, Any],
    transport: dict[str, Any],
    *,
    require_receipt: bool = True,
) -> dict[str, Any]:
    report_path = output / "analysis" / f"aim4_corrected_k{K}.json"
    table_path = output / "analysis" / f"prototypes_k{K}.csv"
    receipt_path = output / "receipts" / "analysis.json"
    report = _json_object(report_path)
    if (
        report.get("component") != "aim4_corrected_cap8192"
        or report.get("status") != "NUMERIC_ANALYSIS_COMPLETE_PATHOLOGY_REVIEW_PENDING"
        or _canonical(report.get("protocol")) != _canonical(spec.get("protocol"))
    ):
        raise RuntimeError("corrected report status/protocol mismatch")
    rebuilt = legacy.final_readout(
        _restore_structural_nan(spec), _restore_structural_nan(transport)
    )
    if _canonical(report.get("readout")) != _canonical(_sanitize_json(rebuilt)):
        raise RuntimeError("corrected report readout differs from inferential artifacts")
    selection = _corrected_review_selection(rebuilt)
    expected_base = sorted(
        set(selection["claimed"])
        | set(selection["suppressed"])
        | set(selection["technical"])
    )
    expected_addendum = sorted(selection["attention_followup"])
    recorded_selection = report.get("review_selection", {})
    for key, expected in {
        **selection,
        "base_packet": expected_base,
        "attention_addendum_packet": expected_addendum,
    }.items():
        if list(recorded_selection.get(key, [])) != list(expected):
            raise RuntimeError(f"corrected report review selection differs for {key}")
    if set(expected_base) & set(expected_addendum):
        raise RuntimeError("corrected report base/addendum selection overlaps")
    expected_table = _report_table(rebuilt, set(expected_base) | set(expected_addendum))
    if table_path.read_bytes() != expected_table.to_csv(index=False).encode():
        raise RuntimeError("corrected prototype table differs from deterministic readout")
    expected_paths = {
        "specificity": output / "analysis" / f"specificity_k{K}.json",
        "transport": output / "analysis" / f"transport_k{K}.json",
        "report": report_path,
        "prototype_table": table_path,
    }
    result = {
        "status": "PASS",
        "base_packet_prototypes": expected_base,
        "attention_addendum_prototypes": expected_addendum,
        "report": _identity(report_path),
        "prototype_table": _identity(table_path),
    }
    if require_receipt:
        receipt = _json_object(receipt_path)
        if (
            receipt.get("status") != "PASS"
            or receipt.get("component") != "aim4_corrected_cap8192"
        ):
            raise RuntimeError("corrected analysis receipt is invalid")
        if set(receipt.get("artifacts", {})) != set(expected_paths):
            raise RuntimeError("corrected analysis receipt artifact inventory differs")
        for key, path in expected_paths.items():
            if receipt["artifacts"][key] != _identity(path):
                raise RuntimeError(f"corrected analysis receipt identity differs for {key}")
        result["receipt"] = _identity(receipt_path)
    return result


def _analysis_artifact_paths(output: Path) -> dict[str, Path]:
    return {
        "specificity": output / "analysis" / f"specificity_k{K}.json",
        "transport": output / "analysis" / f"transport_k{K}.json",
        "report": output / "analysis" / f"aim4_corrected_k{K}.json",
        "prototype_table": output / "analysis" / f"prototypes_k{K}.csv",
    }


def _analysis_receipt(
    output: Path,
    *,
    validation: dict[str, Any],
    profile_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "PASS",
        "validation": validation,
        "profiles": profile_validation["artifacts"],
        "artifacts": {
            key: _identity(path) for key, path in _analysis_artifact_paths(output).items()
        },
    }


def analyze(bundle: InputBundle, output: Path, *, n_bootstrap: int, apply: bool) -> dict[str, Any]:
    _validate_prepared(output, bundle)
    _assert_numeric_not_completed(output)
    if int(n_bootstrap) < DEFAULT_BOOTSTRAP:
        raise ValueError(f"corrected Aim 4 requires at least {DEFAULT_BOOTSTRAP} bootstrap draws")
    profile_validation = validate_profiles(output, bundle, deep_candidates=False)
    if not apply:
        return {
            "status": "DRY_RUN",
            "n_bootstrap": int(n_bootstrap),
            "fixed_bh_family_size": K,
            "profiles": profile_validation,
        }
    canonical_analysis = output / "analysis"
    receipt_path = output / "receipts" / "analysis.json"
    if canonical_analysis.exists() or receipt_path.exists():
        if not canonical_analysis.is_dir():
            raise RuntimeError("unrecoverable partial corrected analysis campaign")
        spec = _json_object(canonical_analysis / f"specificity_k{K}.json")
        transport = _json_object(canonical_analysis / f"transport_k{K}.json")
        if int(spec.get("protocol", {}).get("n_bootstrap", -1)) != int(n_bootstrap):
            raise RuntimeError("existing analysis used a different bootstrap count")
        validation = validate_analysis(spec, transport)
        validate_corrected_report(
            output, spec, transport, require_receipt=False
        )
        expected_receipt = _analysis_receipt(
            output,
            validation=validation,
            profile_validation=profile_validation,
        )
        if receipt_path.is_file():
            if _canonical(_json_object(receipt_path)) != _canonical(expected_receipt):
                raise RuntimeError("canonical analysis and analysis receipt disagree")
        else:
            _write_json_once(receipt_path, expected_receipt)
        validate_corrected_report(output, spec, transport)
        return expected_receipt
    stage_parent = output / "_staging" / "analysis_campaigns" / uuid.uuid4().hex
    stage_eval = stage_parent / "legacy_raw"
    stage_analysis = stage_parent / "analysis"
    stage_eval.mkdir(parents=True, exist_ok=False)
    stage_analysis.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(k=K, n_bootstrap=int(n_bootstrap), seed=BOOTSTRAP_SEED)
    with _legacy_configuration(output, bundle, eval_root=stage_eval):
        legacy.cmd_specificity(args)
        legacy.cmd_transport(args)
    stage_spec = stage_eval / f"e3b_specificity_k{K}.json"
    stage_transport = stage_eval / f"e3b_transport_k{K}.json"
    spec = _json_object(stage_spec)
    transport = _json_object(stage_transport)
    protocol = {
        "cap": CAP,
        "k": K,
        "seeds": list(SEEDS),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "inference_unit": "patient",
        "multiplicity": "BH independently within each quantity x population/arm/contrast",
        "bh_family": "fixed prototypes 0..31, including structural p=1 rows",
        "attention_entitlement": (
            "E0 OOF attention imported byte-for-byte; target attention re-exported from "
            "Aim2 v4 cap8192 full-source refits"
        ),
    }
    spec["component"] = "aim4_corrected_cap8192"
    spec["protocol"] = protocol
    transport["component"] = "aim4_corrected_cap8192"
    transport["protocol"] = protocol
    validation = validate_analysis(spec, transport)
    readout = legacy.final_readout(spec, transport)
    if set(readout) != set(range(K)):
        raise RuntimeError("final readout does not contain exactly 32 prototypes")
    selection = _corrected_review_selection(readout)
    base = sorted(set(selection["claimed"]) | set(selection["suppressed"]) | set(selection["technical"]))
    addendum = sorted(selection["attention_followup"])
    if set(base) & set(addendum):
        raise RuntimeError("base and attention-addendum review selections overlap")
    selected = set(base) | set(addendum)
    required_claims = {
        p for p, row in readout.items()
        if row["group"] in legacy.CLAIMED_GROUPS
        or (row["group"] == "shortcut_technical" and (
            row["kras_association"]["significant"] or row["model_attends"]
        ))
        or row["model_attends"]
    }
    if not required_claims <= selected:
        raise RuntimeError(f"review selection omits claim-bearing prototypes: {sorted(required_claims-selected)}")
    table = _report_table(readout, selected)
    report = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "NUMERIC_ANALYSIS_COMPLETE_PATHOLOGY_REVIEW_PENDING",
        "protocol": protocol,
        "populations": spec["populations"],
        "groups": {
            group: sorted(p for p, row in readout.items() if row["group"] == group)
            for group in legacy.GROUPS
        },
        "readout": readout,
        "review_selection": {
            **selection,
            "base_packet": base,
            "attention_addendum_packet": addendum,
            "selection_rule": (
                "base = claimed + shortcut-suppressed + technical sample; "
                "attention addendum = attention-only prototypes not already in base"
            ),
            "legacy_review_reuse": False,
            "legacy_review_reuse_reason": (
                "target-arm attention and resulting candidates were recomputed from corrected "
                "checkpoints; fresh blinded packets are required"
            ),
        },
        "claim_limits": [
            "Attention is region prioritisation, not causal explanation.",
            "Abundance and attention mass are separate claims.",
            "Set-D survival is robustness only to measured MSI/BRAF dependencies.",
            "Inconclusive or underpowered transport is not evidence of change.",
            "A direct P-v-M contrast is required for a changed call.",
            "Pathology descriptions remain pending until the new corrected packets are reviewed.",
        ],
    }
    spec_path = stage_analysis / f"specificity_k{K}.json"
    transport_path = stage_analysis / f"transport_k{K}.json"
    report_path = stage_analysis / f"aim4_corrected_k{K}.json"
    table_path = stage_analysis / f"prototypes_k{K}.csv"
    _write_json_once(spec_path, _sanitize_json(spec))
    _write_json_once(transport_path, _sanitize_json(transport))
    _write_json_once(report_path, _sanitize_json(report))
    _write_csv_once(table_path, table)
    if canonical_analysis.exists():
        raise FileExistsError("canonical analysis directory appeared during analysis")
    os.rename(stage_analysis, canonical_analysis)
    receipt = _analysis_receipt(
        output,
        validation=validation,
        profile_validation=profile_validation,
    )
    _write_json_once(receipt_path, receipt)
    validate_corrected_report(output, spec, transport)
    return receipt


def _selection_from_report(output: Path) -> tuple[dict[str, Any], dict[str, list[int]]]:
    report = _json_object(output / "analysis" / f"aim4_corrected_k{K}.json")
    selection = report.get("review_selection", {})
    required = {"base_packet", "attention_addendum_packet", "claimed", "suppressed", "technical"}
    if missing := required - set(selection):
        raise RuntimeError(f"corrected report lacks review selection: {sorted(missing)}")
    parsed = {
        key: [int(value) for value in selection[key]]
        for key in ("base_packet", "attention_addendum_packet", "claimed", "suppressed", "technical")
    }
    return report, parsed


PRESENCE_REVIEW_FIELDS: tuple[str, ...] = (
    "mucin",
    "dirty_necrosis",
    "desmoplasia_stroma",
    "budding_invasion",
    "immune_infiltration",
    "normal_organ_tissue",
    "artifact",
)
REVIEW_COLUMNS: tuple[str, ...] = (
    "montage_id",
    "n_tiles",
    "review_status",
    "interpretable",
    "dominant_pattern",
    "heterogeneity",
    "architecture",
    *PRESENCE_REVIEW_FIELDS,
    "differentiation",
    "confidence",
    "reviewer_id",
    "review_date",
    "blinding_attestation",
    "free_text",
)


def _review_form(ids: list[str], tile_counts: list[int]) -> pd.DataFrame:
    pending = {column: "pending" for column in REVIEW_COLUMNS[2:]}
    return pd.DataFrame(
        [
            {
                "montage_id": montage_id,
                "n_tiles": n_tiles,
                **pending,
            }
            for montage_id, n_tiles in zip(ids, tile_counts, strict=True)
        ],
        columns=REVIEW_COLUMNS,
    )


def _packet_instructions(
    packet_name: str, n_montages: int, tiles_per_montage: int, rendered_tile_px: int
) -> str:
    return f"""# Aim 4 corrected blinded morphology review — {packet_name}

This fresh corrected packet contains {n_montages} montages. Each montage targets
one frozen k=32 morphology prototype and contains {tiles_per_montage} tiles when
all required slide files are readable. The vocabulary was discovered without
KRAS, MSI, BRAF, cohort, or specimen-role labels.

You are blinded. Montage IDs carry no biological, molecular, cohort, specimen-
role, or statistical meaning. The unblinding key is stored outside this packet.
Do not open any key until every row in `review_form.csv` is complete.

The packet in the sealed result root is the immutable blank master. Before
entering any assessment, copy `review_form.csv` to a separate working/submission
location and edit only that copy. Return the two completed CSV copies to the
study team; do not edit, rename, or add files inside the sealed packet itself.

## Scale

Each displayed cell is {rendered_tile_px} x {rendered_tile_px} pixels. The source
patch contract is 256 samples at target MPP 0.5, a nominal tissue field of
128 x 128 micrometres. Scanner sampling makes the recorded effective field vary
slightly; `scale_info.csv` gives the minimum, median, and maximum field width for
each montage. Coordinates are level-0 pixels and every field is checked against
the patcher's recorded geometry before rendering.

## Required completion semantics

Every generated row starts with `review_status=pending`. Never use a blank cell:
a blank is indistinguishable from an omitted assessment. When a row is done:

* set `review_status` to `complete`;
* set `interpretable` to exactly `yes`, `partial`, or `no`;
* fill `dominant_pattern` with a concise description, or `not_assessable`;
* set `heterogeneity` to `homogeneous`, `mixed`, or `not_assessable`;
* set `architecture` to `glandular`, `cribriform`, `solid`, `papillary`,
  `mixed`, `non_tumour`, `artifact`, or `not_assessable`;
* set each of `{', '.join(PRESENCE_REVIEW_FIELDS)}` to exactly `present`,
  `absent`, or `not_assessable`;
* set `differentiation` to `well`, `moderate`, `poor`, `mixed`, or
  `not_assessable`, and `confidence` to `low`, `moderate`, `high`, or
  `not_assessable`;
* when `interpretable=no`, set every morphology field from `dominant_pattern`
  through `confidence` to `not_assessable`;
* record `reviewer_id`, an ISO date in `review_date`, and set
  `blinding_attestation=confirmed_no_key_access`;
* fill `free_text`; write `none` if there is nothing additional to report.

Describe morphology, not what it may predict. `not_assessable` is the required
answer when the montage cannot support a field; it is not a missing value.
"""


def _tile_scale(slide_id: str) -> dict[str, Any]:
    patch_path = paths.PINNED_FEATURE_DIR.parent / "patches" / f"{slide_id}_patches.h5"
    # This public validator checks the complete exact-MPP schema: native
    # level-0 coordinate/read units, unsimplified geometry, source-MPP-derived
    # read footprint, and consistency of effective_target_mpp.
    attrs = validate_exact_mpp_coordinate_file(
        patch_path, target_mpp=0.5, patch_size=256
    )
    patch_size = int(attrs.get("patch_size", -1))
    target_mpp = float(attrs.get("target_mpp", np.nan))
    effective_mpp = float(attrs.get("effective_target_mpp", np.nan))
    coordinate_units = str(attrs.get("coordinate_units", ""))
    if (
        patch_size != 256
        or not np.isclose(target_mpp, 0.5, atol=1e-9)
        or not np.isfinite(effective_mpp)
        or coordinate_units != "level0_pixels"
    ):
        raise RuntimeError(f"{slide_id}: patch scale/coordinate contract mismatch")
    return {
        "patch_size_samples": patch_size,
        "target_mpp": target_mpp,
        "effective_target_mpp": effective_mpp,
        "field_width_um": patch_size * effective_mpp,
        "patch_size_level0": int(attrs.get("patch_size_level0", -1)),
        "coordinate_units": coordinate_units,
    }


def _validate_patch_coordinate(slide_id: str, tile_index: int, x: int, y: int) -> None:
    """Bind a selected feature row back to the patcher's exact coordinate row."""
    patch_path = paths.PINNED_FEATURE_DIR.parent / "patches" / f"{slide_id}_patches.h5"
    with h5py.File(patch_path, "r") as handle:
        if "coords" not in handle:
            raise RuntimeError(f"{slide_id}: patch coordinate dataset is missing")
        coords = handle["coords"]
        if tile_index < 0 or tile_index >= len(coords):
            raise RuntimeError(f"{slide_id}: montage tile index is outside patch coordinates")
        recorded = np.asarray(coords[int(tile_index)], dtype=np.int64)
    if recorded.shape != (2,) or not np.array_equal(
        recorded, np.asarray([int(x), int(y)], dtype=np.int64)
    ):
        raise RuntimeError(
            f"{slide_id}: montage coordinate differs from patch row {tile_index}"
        )


def _diverse_patient_picks(
    pool: pd.DataFrame, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Round-robin over blinded strata while using each patient and slide once."""
    if pool.empty or n <= 0:
        return pool.head(0).copy()
    required = {"cohort", "role", "kind", "patient_id", "slide_id", "x", "y"}
    if missing := required - set(pool):
        raise RuntimeError(f"montage candidate pool is missing columns: {sorted(missing)}")
    work = pool.reset_index(drop=True).copy()
    work["_stratum"] = (
        work["cohort"].astype(str)
        + "|"
        + work["role"].astype(str)
        + "|"
        + work["kind"].astype(str)
    )
    queues: dict[str, list[int]] = {}
    for stratum in sorted(work["_stratum"].unique()):
        indices = work.index[work["_stratum"].eq(stratum)].to_numpy()
        queues[str(stratum)] = [int(value) for value in rng.permutation(indices)]
    chosen: list[int] = []
    used_patients: set[str] = set()
    used_slides: set[str] = set()
    used_sites: set[tuple[str, int, int]] = set()
    while len(chosen) < n:
        progressed = False
        for stratum in sorted(queues):
            queue = queues[stratum]
            while queue:
                index = queue.pop()
                row = work.loc[index]
                patient = str(row["patient_id"])
                slide = str(row["slide_id"])
                site = (slide, int(row["x"]), int(row["y"]))
                if (
                    patient in used_patients
                    or slide in used_slides
                    or site in used_sites
                ):
                    continue
                chosen.append(index)
                used_patients.add(patient)
                used_slides.add(slide)
                used_sites.add(site)
                progressed = True
                break
            if len(chosen) == n:
                break
        if not progressed:
            break
    return work.loc[chosen].drop(columns=["_stratum"]).reset_index(drop=True)


def _load_slide_index(output: Path, slide_root: Path) -> dict[str, str]:
    """Create one root/content-bound WSI index and reject stale precedence."""
    index_path = output / "support" / "slide_path_index.json"
    resolved_root = slide_root.resolve(strict=True)
    temporary_cache = (
        output / "_staging" / "slide_indexes" / f"{uuid.uuid4().hex}.json"
    )
    generated = legacy.slide_path_index(resolved_root, temporary_cache)

    def normalized(raw: dict[str, Any]) -> dict[str, str]:
        checked: dict[str, str] = {}
        for slide_id, value in raw.items():
            candidate = Path(str(value)).resolve(strict=True)
            try:
                candidate.relative_to(resolved_root)
            except ValueError as error:
                raise RuntimeError(
                    f"{slide_id}: cached WSI is outside the requested root"
                ) from error
            if candidate.suffix.lower() not in legacy._WSI_SUFFIXES:
                raise RuntimeError(f"{slide_id}: cached WSI has an unsupported suffix")
            checked[str(slide_id)] = str(candidate)
        return checked

    current_index = normalized({str(key): str(value) for key, value in generated.items()})
    current_inventory = {
        slide_id: _stat_identity(Path(value))
        for slide_id, value in sorted(current_index.items())
    }
    if index_path.is_file():
        payload = _json_object(index_path)
        if (
            payload.get("schema_version") != 2
            or payload.get("slide_root") != str(resolved_root)
            or not isinstance(payload.get("index"), dict)
            or not isinstance(payload.get("wsi_inventory"), dict)
        ):
            raise RuntimeError("cached slide index is not bound to the requested slide root")
        recorded_index = normalized(payload["index"])
        if recorded_index != current_index:
            raise RuntimeError(
                "cached slide index differs from current WSI precedence/inventory"
            )
        if _canonical(payload["wsi_inventory"]) != _canonical(current_inventory):
            raise RuntimeError("cached WSI size/mtime inventory changed")
    else:
        _write_json_once(
            index_path,
            {
                "schema_version": 2,
                "slide_root": str(resolved_root),
                "index": current_index,
                "wsi_inventory": current_inventory,
            },
        )
    for slide_id, value in current_index.items():
        candidate = Path(value).resolve(strict=True)
        current_index[slide_id] = str(candidate)
    return current_index


def _render_packet(
    *,
    output: Path,
    candidates: pd.DataFrame,
    prototypes: list[int],
    packet_slug: str,
    id_prefix: str,
    selection_strata: dict[int, str],
    slide_root: Path,
    tiles_per_montage: int,
    tile_px: int,
    seed: int,
    bundle_parent: Path | None = None,
) -> dict[str, Any]:
    from PIL import Image

    try:
        import openslide  # noqa: F401
    except ImportError as error:
        raise RuntimeError("review packets require openslide-python") from error
    canonical_bundle = (
        bundle_parent if bundle_parent is not None
        else output / "review_bundles" / f"k{K}"
    ) / packet_slug
    if canonical_bundle.exists() or canonical_bundle.is_symlink():
        raise FileExistsError(f"refusing to overwrite review bundle: {canonical_bundle}")
    stage_bundle = (
        output / "_staging" / "review_bundles" / f"k{K}"
        / f"{packet_slug}.{uuid.uuid4().hex}"
    )
    packet = stage_bundle / "packet"
    key_path = stage_bundle / "unblinding_key_DO_NOT_SHARE.csv"
    slide_index = _load_slide_index(output, slide_root)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(prototypes)) if prototypes else np.asarray([], dtype=int)
    ordered = [prototypes[int(index)] for index in order]
    key_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    montage_ids: list[str] = []
    tile_counts: list[int] = []
    for position, prototype in enumerate(ordered, start=1):
        montage_id = f"{id_prefix}{position:02d}"
        pool = candidates[candidates["prototype"].eq(prototype)]
        picks = _diverse_patient_picks(pool, tiles_per_montage, rng)
        site = ["slide_id", "x", "y"]
        if len(picks) != tiles_per_montage:
            raise RuntimeError(
                f"{packet_slug}/{montage_id}: selected {len(picks)} of {tiles_per_montage} tiles"
            )
        if picks.duplicated(site).any() or picks["slide_id"].nunique() != len(picks)\
                or picks["patient_id"].nunique() != len(picks):
            raise RuntimeError(f"{packet_slug}/{montage_id}: montage diversity contract failed")
        images: list[Any] = []
        provenance: list[Any] = []
        scales: list[dict[str, Any]] = []
        for pick in picks.itertuples(index=False):
            wsi_path = slide_index.get(str(pick.slide_id))
            if wsi_path is None:
                raise RuntimeError(f"no WSI path for required montage slide {pick.slide_id}")
            try:
                scale = _tile_scale(str(pick.slide_id))
                _validate_patch_coordinate(
                    str(pick.slide_id),
                    int(pick.tile_index),
                    int(pick.x),
                    int(pick.y),
                )
                image = legacy._read_tile(
                    Image, wsi_path, pick.slide_id, int(pick.x), int(pick.y), tile_px
                )
            except Exception as error:  # noqa: BLE001
                raise RuntimeError(f"could not render required tile {pick.slide_id}: {error}") from error
            images.append(image)
            provenance.append(pick)
            scales.append(scale)
        montage_path = packet / "montages" / f"{montage_id}.jpg"
        montage_path.parent.mkdir(parents=True, exist_ok=True)
        if montage_path.exists():
            raise FileExistsError(f"refusing to overwrite montage: {montage_path}")
        legacy._grid(Image, images, tile_px).save(montage_path, quality=92)
        montage_ids.append(montage_id)
        tile_counts.append(len(images))
        widths = np.asarray([scale["field_width_um"] for scale in scales], dtype=float)
        scale_rows.append({
            "montage_id": montage_id,
            "n_tiles": len(scales),
            "nominal_target_mpp": 0.5,
            "nominal_field_width_um": 128.0,
            "min_recorded_field_width_um": float(widths.min()),
            "median_recorded_field_width_um": float(np.median(widths)),
            "max_recorded_field_width_um": float(widths.max()),
            "coordinate_units": "level0_pixels",
        })
        for slot, (pick, scale) in enumerate(zip(provenance, scales, strict=True)):
            key_rows.append({
                "montage_id": montage_id,
                "slot": slot,
                "prototype": prototype,
                "selection_stratum": selection_strata[prototype],
                **scale,
                **pick._asdict(),
            })
    form = _review_form(montage_ids, tile_counts)
    _write_csv_once(packet / "review_form.csv", form)
    scale_columns = [
        "montage_id", "n_tiles", "nominal_target_mpp", "nominal_field_width_um",
        "min_recorded_field_width_um", "median_recorded_field_width_um",
        "max_recorded_field_width_um", "coordinate_units",
    ]
    _write_csv_once(packet / "scale_info.csv", pd.DataFrame(scale_rows, columns=scale_columns))
    _write_bytes_once(
        packet / "README.md",
        _packet_instructions(
            packet_slug, len(montage_ids), tiles_per_montage, tile_px
        ).encode(),
    )
    key_columns = [
        "montage_id", "slot", "prototype", "selection_stratum", "arm", "role",
        "slide_id", "patient_id", "cohort", "subcohort", "label", "kind",
        "tile_index", "x", "y", "distance_to_centroid", "attention",
        "patch_size_samples", "target_mpp", "effective_target_mpp", "field_width_um",
        "patch_size_level0", "coordinate_units",
    ]
    key = pd.DataFrame(key_rows)
    if key.empty:
        key = pd.DataFrame(columns=key_columns)
    if prototypes:
        if key["prototype"].nunique() != len(prototypes):
            raise RuntimeError(f"{packet_slug}: key does not cover every selected prototype")
        if set(key["montage_id"]) != set(montage_ids):
            raise RuntimeError(f"{packet_slug}: key/form montage identities differ")
    _write_csv_once(key_path, key)
    # A whole review bundle (packet + sibling external key) becomes canonical
    # in one directory rename. A rendering failure can only strand a uniquely
    # named staging directory and can never leave a canonical partial packet.
    canonical_bundle.parent.mkdir(parents=True, exist_ok=True)
    if canonical_bundle.exists():
        raise FileExistsError(f"canonical review bundle appeared during render: {canonical_bundle}")
    os.rename(stage_bundle, canonical_bundle)
    packet = canonical_bundle / "packet"
    key_path = canonical_bundle / "unblinding_key_DO_NOT_SHARE.csv"
    return {
        "bundle": str(canonical_bundle),
        "packet": str(packet),
        "packet_slug": packet_slug,
        "n_montages": len(montage_ids),
        "prototype_count": len(prototypes),
        "form": _identity(packet / "review_form.csv"),
        "scale_info": _identity(packet / "scale_info.csv"),
        "instructions": _identity(packet / "README.md"),
        "key": _identity(key_path),
        "montages": {
            montage_id: _identity(packet / "montages" / f"{montage_id}.jpg")
            for montage_id in montage_ids
        },
    }


def _rebase_review_block(
    block: dict[str, Any], *, staged_parent: Path, canonical_parent: Path
) -> dict[str, Any]:
    """Point staged receipt identities at their future atomic destination."""
    rebased = dict(block)

    def future_path(value: str) -> str:
        relative = Path(value).resolve(strict=True).relative_to(
            staged_parent.resolve(strict=True)
        )
        return str(canonical_parent / relative)

    for key in ("bundle", "packet"):
        rebased[key] = future_path(str(block[key]))
    for key in ("form", "scale_info", "instructions", "key"):
        rebased[key] = {**block[key], "path": future_path(str(block[key]["path"]))}
    rebased["montages"] = {
        montage_id: {**evidence, "path": future_path(str(evidence["path"]))}
        for montage_id, evidence in block["montages"].items()
    }
    return rebased


def make_review_packets(
    bundle: InputBundle,
    output: Path,
    *,
    slide_root: Path,
    tiles_per_montage: int,
    tile_px: int,
    seed: int,
    apply: bool,
) -> dict[str, Any]:
    _validate_prepared(output, bundle)
    _assert_numeric_not_completed(output)
    validate_profiles(output, bundle, deep_candidates=False)
    _report, selection = _selection_from_report(output)
    if int(tiles_per_montage) != 12 or int(tile_px) != 256 or int(seed) != BOOTSTRAP_SEED:
        raise ValueError(
            "corrected review packets lock 12 tiles/prototype, 256 display pixels, "
            f"and seed {BOOTSTRAP_SEED}"
        )
    if not slide_root.is_absolute() or not slide_root.is_dir():
        raise ValueError(f"--slide-root must be an existing absolute directory: {slide_root}")
    base = selection["base_packet"]
    addendum = selection["attention_addendum_packet"]
    if set(base) & set(addendum):
        raise RuntimeError("base/addendum selections overlap")
    plan = {
        "status": "DRY_RUN" if not apply else "PASS",
        "base_packet_prototypes": base,
        "attention_addendum_prototypes": addendum,
        "separate_keys": True,
        "tiles_per_montage": int(tiles_per_montage),
    }
    if not apply:
        return plan
    canonical_parent = output / "review_bundles" / f"k{K}"
    receipt_path = output / "receipts" / "review_packets.json"
    transaction_receipt = canonical_parent / REVIEW_TRANSACTION_RECEIPT
    # Recover a process interruption after the all-packet atomic rename but
    # before publication of its external receipt.  No packet is regenerated or
    # overwritten; the receipt staged inside the bundle is copied verbatim.
    if canonical_parent.exists() or receipt_path.exists():
        if canonical_parent.is_dir() and transaction_receipt.is_file():
            transaction = _json_object(transaction_receipt)
            if receipt_path.is_file():
                if _canonical(_json_object(receipt_path)) != _canonical(transaction):
                    raise RuntimeError("canonical review bundle and receipt disagree")
            else:
                _copy_once(transaction_receipt, receipt_path)
            validate_review_packets(output)
            return transaction
        raise RuntimeError("unrecoverable partial corrected review campaign")
    candidates = pd.read_parquet(output / "profiles" / f"tile_candidates_k{K}.parquet")
    strata: dict[int, str] = {}
    for key in ("claimed", "suppressed", "technical"):
        for prototype in selection[key]:
            if prototype in strata:
                raise RuntimeError(f"prototype {prototype} is repeated across base strata")
            strata[prototype] = key
    if set(strata) != set(base):
        raise RuntimeError("base selection is not exactly claimed/suppressed/technical")
    for prototype in addendum:
        strata[prototype] = "attention_followup"
    staged_parent = (
        output / "_staging" / "review_campaigns" / f"k{K}.{uuid.uuid4().hex}"
    )
    base_staged = _render_packet(
        output=output,
        candidates=candidates,
        prototypes=base,
        packet_slug="base",
        id_prefix="M",
        selection_strata=strata,
        slide_root=slide_root,
        tiles_per_montage=tiles_per_montage,
        tile_px=tile_px,
        seed=seed,
        bundle_parent=staged_parent,
    )
    addendum_staged = _render_packet(
        output=output,
        candidates=candidates,
        prototypes=addendum,
        packet_slug="attention_addendum",
        id_prefix="A",
        selection_strata=strata,
        slide_root=slide_root,
        tiles_per_montage=tiles_per_montage,
        tile_px=tile_px,
        seed=seed + 1,
        bundle_parent=staged_parent,
    )
    base_receipt = _rebase_review_block(
        base_staged, staged_parent=staged_parent, canonical_parent=canonical_parent
    )
    addendum_receipt = _rebase_review_block(
        addendum_staged, staged_parent=staged_parent, canonical_parent=canonical_parent
    )
    receipt = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "PASS",
        "selection": plan,
        "base": base_receipt,
        "attention_addendum": addendum_receipt,
        "blinding": "prototype IDs occur only in external keys, never packet files",
    }
    # The internal transaction receipt deliberately sits beside (not inside)
    # reviewer packets and is marked DO_NOT_SHARE because it binds their keys.
    # Both packets and their separate keys become canonical in one rename.
    _write_json_once(staged_parent / REVIEW_TRANSACTION_RECEIPT, receipt)
    canonical_parent.parent.mkdir(parents=True, exist_ok=True)
    if canonical_parent.exists() or receipt_path.exists():
        raise FileExistsError("canonical review campaign appeared during rendering")
    os.rename(staged_parent, canonical_parent)
    _copy_once(canonical_parent / REVIEW_TRANSACTION_RECEIPT, receipt_path)
    return receipt


def validate_review_packets(output: Path) -> dict[str, Any]:
    receipt_path = output / "receipts" / "review_packets.json"
    receipt = _json_object(receipt_path)
    campaign_root = output / "review_bundles" / f"k{K}"
    transaction_path = campaign_root / REVIEW_TRANSACTION_RECEIPT
    transaction = _json_object(transaction_path)
    if not _same_content_identity(_identity(transaction_path), _identity(receipt_path)) or (
        _canonical(transaction) != _canonical(receipt)
    ):
        raise RuntimeError("review transaction receipt differs from its external copy")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("component") != "aim4_corrected_cap8192"
        or receipt.get("status") != "PASS"
    ):
        raise RuntimeError("fresh corrected review packets do not have PASS status")
    _report, selection = _selection_from_report(output)
    expected_by_packet = {
        "base": selection["base_packet"],
        "attention_addendum": selection["attention_addendum_packet"],
    }
    expected_selection = {
        "status": "PASS",
        "base_packet_prototypes": selection["base_packet"],
        "attention_addendum_prototypes": selection["attention_addendum_packet"],
        "separate_keys": True,
        "tiles_per_montage": 12,
    }
    if receipt.get("selection") != expected_selection or receipt.get("blinding") != (
        "prototype IDs occur only in external keys, never packet files"
    ):
        raise RuntimeError("review receipt selection/blinding contract differs")
    montage_ids_across_packets: set[str] = set()
    expected_campaign_files = {REVIEW_TRANSACTION_RECEIPT}
    summaries: dict[str, Any] = {}
    for packet_slug, expected_prototypes in expected_by_packet.items():
        block = receipt.get(packet_slug)
        if not isinstance(block, dict):
            raise RuntimeError(f"review receipt lacks {packet_slug} block")
        bundle_path = Path(str(block.get("bundle", ""))).resolve(strict=True)
        expected_bundle = (output / "review_bundles" / f"k{K}" / packet_slug).resolve(strict=True)
        if bundle_path != expected_bundle:
            raise RuntimeError(f"{packet_slug}: bundle path mismatch")
        packet = Path(str(block.get("packet", ""))).resolve(strict=True)
        expected_packet = (expected_bundle / "packet").resolve(strict=True)
        if packet != expected_packet:
            raise RuntimeError(f"{packet_slug}: packet path mismatch")
        exact_artifact_paths = {
            "form": packet / "review_form.csv",
            "scale_info": packet / "scale_info.csv",
            "instructions": packet / "README.md",
            "key": expected_bundle / "unblinding_key_DO_NOT_SHARE.csv",
        }
        for key, expected_path in exact_artifact_paths.items():
            _validate_recorded_identity(block[key])
            if Path(str(block[key]["path"])).resolve(strict=True) != expected_path:
                raise RuntimeError(f"{packet_slug}: {key} path mismatch")
        key_path = Path(str(block["key"]["path"])).resolve(strict=True)
        try:
            key_path.relative_to(packet)
        except ValueError:
            pass
        else:
            raise RuntimeError(f"{packet_slug}: unblinding key is inside reviewer packet")
        form = pd.read_csv(Path(str(block["form"]["path"])), dtype=str, keep_default_na=False)
        scale_info = pd.read_csv(Path(str(block["scale_info"]["path"])))
        key_frame = pd.read_csv(key_path, dtype={"montage_id": str, "prototype": int})
        if tuple(form.columns) != REVIEW_COLUMNS:
            raise RuntimeError(f"{packet_slug}: reviewer form schema differs from locked schema")
        if {"prototype", "selection_stratum"} & set(form):
            raise RuntimeError(f"{packet_slug}: reviewer form leaks unblinding information")
        if not form.empty and not (form[list(REVIEW_COLUMNS[2:])] == "pending").all().all():
            raise RuntimeError(f"{packet_slug}: generated form is not in explicit pending state")
        required_key_columns = {
            "montage_id", "prototype", "selection_stratum", "slot",
            "patient_id", "slide_id", "x", "y",
        }
        if missing := required_key_columns - set(key_frame):
            raise RuntimeError(f"{packet_slug}: key schema is incomplete: {sorted(missing)}")
        id_prefix = "M" if packet_slug == "base" else "A"
        expected_ids = [
            f"{id_prefix}{position:02d}"
            for position in range(1, len(expected_prototypes) + 1)
        ]
        if form["montage_id"].astype(str).tolist() != expected_ids:
            raise RuntimeError(f"{packet_slug}: montage ID inventory/order differs")
        form_ids = set(expected_ids)
        key_ids = set(key_frame["montage_id"].astype(str))
        if set(scale_info["montage_id"].astype(str)) != form_ids:
            raise RuntimeError(f"{packet_slug}: scale/form montage identities differ")
        if not scale_info.empty and not (
            np.isclose(scale_info["nominal_target_mpp"].to_numpy(dtype=float), 0.5).all()
            and np.isclose(scale_info["nominal_field_width_um"].to_numpy(dtype=float), 128.0).all()
            and (scale_info["coordinate_units"].astype(str) == "level0_pixels").all()
        ):
            raise RuntimeError(f"{packet_slug}: physical scale contract differs")
        if form_ids != key_ids or form_ids & montage_ids_across_packets:
            raise RuntimeError(f"{packet_slug}: form/key IDs mismatch or cross-packet ID reuse")
        montage_ids_across_packets |= form_ids
        keyed_prototypes = set(pd.to_numeric(key_frame["prototype"], errors="raise").astype(int))
        if keyed_prototypes != set(expected_prototypes):
            raise RuntimeError(f"{packet_slug}: key prototype selection differs from report")
        if not key_frame.empty:
            per_id_prototypes = key_frame.groupby("montage_id")["prototype"].nunique()
            if (per_id_prototypes != 1).any():
                raise RuntimeError(f"{packet_slug}: a montage maps to multiple prototypes")
            keyed_counts = key_frame.groupby("montage_id").size().to_dict()
            declared = pd.to_numeric(form.set_index("montage_id")["n_tiles"], errors="raise")
            if any(int(declared[montage_id]) != int(count) for montage_id, count in keyed_counts.items()):
                raise RuntimeError(f"{packet_slug}: form/key tile count differs")
            if set(map(int, declared)) != {12} or set(map(int, keyed_counts.values())) != {12}:
                raise RuntimeError(f"{packet_slug}: montage does not contain exactly 12 tiles")
            if key_frame.duplicated(["montage_id", "slot"]).any():
                raise RuntimeError(f"{packet_slug}: duplicate montage slot")
            expected_slots = set(range(12))
            if any(
                set(pd.to_numeric(block_rows["slot"], errors="raise").astype(int))
                != expected_slots
                for _montage_id, block_rows in key_frame.groupby("montage_id")
            ):
                raise RuntimeError(f"{packet_slug}: montage slot inventory differs")
            for montage_id, block_rows in key_frame.groupby("montage_id"):
                if (
                    block_rows["patient_id"].astype(str).nunique() != 12
                    or block_rows["slide_id"].astype(str).nunique() != 12
                    or block_rows.duplicated(["slide_id", "x", "y"]).any()
                ):
                    raise RuntimeError(
                        f"{packet_slug}/{montage_id}: sealed montage diversity differs"
                    )
            expected_strata = {
                **{int(value): "claimed" for value in selection["claimed"]},
                **{int(value): "suppressed" for value in selection["suppressed"]},
                **{int(value): "technical" for value in selection["technical"]},
                **{
                    int(value): "attention_followup"
                    for value in selection["attention_addendum_packet"]
                },
            }
            if any(
                str(row.selection_stratum) != expected_strata[int(row.prototype)]
                for row in key_frame.itertuples(index=False)
            ):
                raise RuntimeError(f"{packet_slug}: key selection stratum differs")
        montage_evidence = block.get("montages", {})
        if set(montage_evidence) != form_ids:
            raise RuntimeError(f"{packet_slug}: montage receipt inventory differs from form")
        for montage_id, evidence in montage_evidence.items():
            _validate_recorded_identity(evidence)
            image = Path(str(evidence["path"])).resolve(strict=True)
            if image != packet / "montages" / f"{montage_id}.jpg":
                raise RuntimeError(f"{packet_slug}: montage image path differs")
        if (
            block.get("packet_slug") != packet_slug
            or int(block.get("n_montages", -1)) != len(expected_prototypes)
            or int(block.get("prototype_count", -1)) != len(expected_prototypes)
        ):
            raise RuntimeError(f"{packet_slug}: receipt counts/slug differ")
        expected_bundle_files = {
            "packet/review_form.csv",
            "packet/scale_info.csv",
            "packet/README.md",
            "unblinding_key_DO_NOT_SHARE.csv",
            *{f"packet/montages/{montage_id}.jpg" for montage_id in form_ids},
        }
        actual_bundle_files = {
            str(path.relative_to(expected_bundle))
            for path in expected_bundle.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if actual_bundle_files != expected_bundle_files:
            raise RuntimeError(f"{packet_slug}: exact blinded bundle inventory differs")
        expected_campaign_files.update(
            f"{packet_slug}/{relative}" for relative in expected_bundle_files
        )
        summaries[packet_slug] = {
            "n_montages": len(form_ids),
            "prototypes": sorted(keyed_prototypes),
            "key_external_to_packet": True,
        }
    if set(expected_by_packet["base"]) & set(expected_by_packet["attention_addendum"]):
        raise RuntimeError("review base and attention-addendum selections overlap")
    actual_campaign_files = {
        str(path.relative_to(campaign_root))
        for path in campaign_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_campaign_files != expected_campaign_files:
        raise RuntimeError("exact review campaign inventory differs")
    return {"status": "PASS", "packets": summaries, "receipt": _identity(receipt_path)}


def _completed_review_form(source: Path, generated: Path, *, label: str) -> pd.DataFrame:
    if not source.is_absolute() or not source.is_file():
        raise ValueError(f"{label} must be an existing absolute CSV path: {source}")
    submitted = pd.read_csv(source, dtype=str, keep_default_na=False)
    locked = pd.read_csv(generated, dtype=str, keep_default_na=False)
    if tuple(submitted.columns) != REVIEW_COLUMNS or tuple(locked.columns) != REVIEW_COLUMNS:
        raise RuntimeError(f"{label}: completed/generated form schema differs from locked schema")
    if submitted["montage_id"].duplicated().any():
        raise RuntimeError(f"{label}: duplicate montage ID")
    if set(submitted["montage_id"]) != set(locked["montage_id"]):
        raise RuntimeError(f"{label}: completed form montage inventory differs from generated form")
    submitted = submitted.set_index("montage_id").loc[locked["montage_id"]].reset_index()
    submitted_counts = pd.to_numeric(submitted["n_tiles"], errors="raise").to_numpy(dtype=int)
    locked_counts = pd.to_numeric(locked["n_tiles"], errors="raise").to_numpy(dtype=int)
    if not np.array_equal(submitted_counts, locked_counts):
        raise RuntimeError(f"{label}: completed form changed locked tile counts")
    if submitted.empty:
        return submitted
    assessed = submitted[list(REVIEW_COLUMNS[2:])].apply(
        lambda column: column.astype(str).str.strip()
    )
    if (assessed == "").any().any() or (assessed == "pending").any().any():
        raise RuntimeError(f"{label}: blank or pending fields remain")
    submitted.loc[:, list(REVIEW_COLUMNS[2:])] = assessed
    allowed = {
        "review_status": {"complete"},
        "interpretable": {"yes", "partial", "no"},
        "heterogeneity": {"homogeneous", "mixed", "not_assessable"},
        "architecture": {
            "glandular", "cribriform", "solid", "papillary", "mixed",
            "non_tumour", "artifact", "not_assessable",
        },
        "differentiation": {"well", "moderate", "poor", "mixed", "not_assessable"},
        "confidence": {"low", "moderate", "high", "not_assessable"},
        "blinding_attestation": {"confirmed_no_key_access"},
        **{field: {"present", "absent", "not_assessable"}
           for field in PRESENCE_REVIEW_FIELDS},
    }
    for column, choices in allowed.items():
        unexpected = set(submitted[column]) - choices
        if unexpected:
            raise RuntimeError(f"{label}: invalid {column} values: {sorted(unexpected)}")
    for row in submitted.itertuples(index=False):
        if row.interpretable == "no":
            not_assessable_fields = (
                "dominant_pattern", "heterogeneity", "architecture",
                *PRESENCE_REVIEW_FIELDS, "differentiation", "confidence",
            )
            if any(getattr(row, field) != "not_assessable" for field in not_assessable_fields):
                raise RuntimeError(
                    f"{label}/{row.montage_id}: non-interpretable morphology fields "
                    "must all be not_assessable"
                )
        if row.interpretable in {"yes", "partial"} and row.dominant_pattern == "not_assessable":
            raise RuntimeError(f"{label}/{row.montage_id}: interpretable row lacks dominant pattern")
        if row.reviewer_id in {"none", "not_assessable"}:
            raise RuntimeError(f"{label}/{row.montage_id}: reviewer_id is not identified")
        try:
            date.fromisoformat(str(row.review_date))
        except ValueError as error:
            raise RuntimeError(
                f"{label}/{row.montage_id}: review_date is not ISO YYYY-MM-DD"
            ) from error
    return submitted


def _review_bundle_paths(output: Path, packet_slug: str) -> tuple[Path, Path]:
    bundle = output / "review_bundles" / f"k{K}" / packet_slug
    return bundle / "packet" / "review_form.csv", bundle / "unblinding_key_DO_NOT_SHARE.csv"


def _rebase_identity(
    evidence: dict[str, Any], *, staged_parent: Path, canonical_parent: Path
) -> dict[str, Any]:
    source = Path(str(evidence["path"])).resolve(strict=True)
    relative = source.relative_to(staged_parent.resolve(strict=True))
    return {**evidence, "path": str(canonical_parent / relative)}


def import_completed_reviews(
    bundle: InputBundle,
    output: Path,
    *,
    base_form: Path,
    attention_addendum_form: Path,
    apply: bool,
) -> dict[str, Any]:
    _validate_prepared(output, bundle, require_live_source=False)
    _assert_not_completed(output)
    # Pathology is an addendum to a sealed numeric analysis, never a way to
    # mutate or complete an unverified numeric campaign.
    validate_numeric_verification(output)
    validate_review_packets(output)
    canonical_bundle = output / "review_completion" / f"k{K}"
    receipt_path = output / "receipts" / "review_import.json"
    transaction_path = canonical_bundle / REVIEW_IMPORT_TRANSACTION_RECEIPT
    # Recover a crash after the complete review import was promoted but before
    # its external receipt was copied.  The original submission paths are no
    # longer needed for this recovery because their immutable copies are in the
    # promoted bundle.
    if apply and (canonical_bundle.exists() or receipt_path.exists()):
        if canonical_bundle.is_dir() and transaction_path.is_file():
            transaction = _json_object(transaction_path)
            if receipt_path.is_file():
                if _canonical(_json_object(receipt_path)) != _canonical(transaction):
                    raise RuntimeError("completed review bundle and receipt disagree")
            else:
                _copy_once(transaction_path, receipt_path)
            validate_imported_reviews(output)
            return transaction
        raise RuntimeError("unrecoverable partial completed-review campaign")
    if not base_form.is_absolute() or not attention_addendum_form.is_absolute():
        raise ValueError("completed review forms must be explicit absolute paths")
    source_forms = {
        "base": base_form.resolve(strict=True),
        "attention_addendum": attention_addendum_form.resolve(strict=True),
    }
    if source_forms["base"] == source_forms["attention_addendum"]:
        raise ValueError("base and attention-addendum completed forms must be separate files")
    validated: dict[str, pd.DataFrame] = {}
    for packet_slug, source in source_forms.items():
        generated, _key = _review_bundle_paths(output, packet_slug)
        if source == generated.resolve(strict=True):
            raise ValueError(
                f"completed-{packet_slug} must be a copy outside the sealed packet"
            )
        validated[packet_slug] = _completed_review_form(
            source, generated, label=f"completed-{packet_slug}"
        )
    plan = {
        "status": "DRY_RUN" if not apply else "PASS",
        "base_rows": len(validated["base"]),
        "attention_addendum_rows": len(validated["attention_addendum"]),
        "all_rows_explicitly_complete": True,
    }
    if not apply:
        return plan
    stage_bundle = (
        output / "_staging" / "review_imports" / f"k{K}.{uuid.uuid4().hex}"
    )
    staged_destinations = {
        packet_slug: stage_bundle / "submissions" / f"{packet_slug}_completed.csv"
        for packet_slug in source_forms
    }
    staged_imports = {
        packet_slug: _copy_once(source_forms[packet_slug], staged_destinations[packet_slug])
        for packet_slug in source_forms
    }
    imports = {
        packet_slug: {
            "source": evidence["source"],
            "imported": _rebase_identity(
                evidence["imported"],
                staged_parent=stage_bundle,
                canonical_parent=canonical_bundle,
            ),
        }
        for packet_slug, evidence in staged_imports.items()
    }
    annotations: dict[str, Any] = {}
    montage_ids: set[str] = set()
    for packet_slug, destination in staged_destinations.items():
        generated, key_path = _review_bundle_paths(output, packet_slug)
        submitted = _completed_review_form(
            destination, generated, label=f"imported-{packet_slug}"
        )
        key = pd.read_csv(key_path, dtype={"montage_id": str, "prototype": int})
        mapping = (
            key[["montage_id", "prototype"]].drop_duplicates()
            .set_index("montage_id")["prototype"].to_dict()
        )
        if set(submitted["montage_id"]) != set(mapping):
            raise RuntimeError(f"{packet_slug}: completed form/key montage coverage differs")
        if montage_ids & set(mapping):
            raise RuntimeError("completed base/addendum forms reuse montage IDs")
        montage_ids |= set(mapping)
        for record in submitted.to_dict("records"):
            montage_id = str(record["montage_id"])
            prototype = int(mapping[montage_id])
            if str(prototype) in annotations:
                raise RuntimeError(f"prototype {prototype} has multiple completed assessments")
            annotations[str(prototype)] = {
                "prototype": prototype,
                "packet": packet_slug,
                "montage_id": montage_id,
                "assessment": record,
            }
    numeric_report = _json_object(output / "analysis" / f"aim4_corrected_k{K}.json")
    expected = {
        int(value)
        for value in (
            numeric_report["review_selection"]["base_packet"]
            + numeric_report["review_selection"]["attention_addendum_packet"]
        )
    }
    if set(map(int, annotations)) != expected:
        raise RuntimeError("completed review annotations do not cover exact locked selection")
    pathology = {
        "schema_version": 1,
        "component": "aim4_corrected_pathology_review",
        "status": "complete",
        "structured_not_concatenated": True,
        "annotations": annotations,
        "submission_imports": imports,
    }
    pathology_path = stage_bundle / f"pathology_review_k{K}.json"
    reviewed_path = stage_bundle / f"aim4_corrected_reviewed_k{K}.json"
    _write_json_once(pathology_path, pathology)
    completed_claim_limits = [
        value for value in numeric_report.get("claim_limits", [])
        if not str(value).startswith("Pathology descriptions remain pending")
    ]
    completed_claim_limits.append(
        "Pathology descriptions are the imported structured assessments from the fresh "
        "corrected blinded packets."
    )
    reviewed = {
        **numeric_report,
        "status": "ANALYSIS_AND_PATHOLOGY_REVIEW_COMPLETE",
        "claim_limits": completed_claim_limits,
        "pathology_review": pathology,
    }
    _write_json_once(reviewed_path, reviewed)
    receipt = {
        "schema_version": 1,
        "component": "aim4_corrected_pathology_review",
        "status": "PASS",
        "forms": imports,
        "pathology": _rebase_identity(
            _identity(pathology_path),
            staged_parent=stage_bundle,
            canonical_parent=canonical_bundle,
        ),
        "reviewed_report": _rebase_identity(
            _identity(reviewed_path),
            staged_parent=stage_bundle,
            canonical_parent=canonical_bundle,
        ),
        "n_completed_prototypes": len(annotations),
    }
    _write_json_once(stage_bundle / REVIEW_IMPORT_TRANSACTION_RECEIPT, receipt)
    canonical_bundle.parent.mkdir(parents=True, exist_ok=True)
    if canonical_bundle.exists() or receipt_path.exists():
        raise FileExistsError("completed-review campaign appeared during import")
    os.rename(stage_bundle, canonical_bundle)
    _copy_once(canonical_bundle / REVIEW_IMPORT_TRANSACTION_RECEIPT, receipt_path)
    validate_imported_reviews(output)
    return receipt


def validate_imported_reviews(output: Path) -> dict[str, Any]:
    receipt_path = output / "receipts" / "review_import.json"
    receipt = _json_object(receipt_path)
    canonical_bundle = output / "review_completion" / f"k{K}"
    transaction_path = canonical_bundle / REVIEW_IMPORT_TRANSACTION_RECEIPT
    transaction = _json_object(transaction_path)
    if not _same_content_identity(_identity(transaction_path), _identity(receipt_path)) or (
        _canonical(transaction) != _canonical(receipt)
    ):
        raise RuntimeError("completed-review transaction receipt differs from external copy")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("component") != "aim4_corrected_pathology_review"
        or receipt.get("status") != "PASS"
    ):
        raise RuntimeError("completed review import lacks PASS status")
    exact_paths = {
        "base": canonical_bundle / "submissions" / "base_completed.csv",
        "attention_addendum": (
            canonical_bundle / "submissions" / "attention_addendum_completed.csv"
        ),
    }
    annotations: dict[str, Any] = {}
    completed_montage_ids: set[str] = set()
    for packet_slug in ("base", "attention_addendum"):
        import_evidence = receipt["forms"][packet_slug]
        _validate_recorded_identity(import_evidence["imported"])
        if Path(str(import_evidence["imported"]["path"])).resolve(strict=True) != (
            exact_paths[packet_slug].resolve(strict=True)
        ):
            raise RuntimeError(f"sealed {packet_slug} review path differs")
        if (
            import_evidence["source"]["size_bytes"],
            import_evidence["source"]["sha256"],
        ) != (
            import_evidence["imported"]["size_bytes"],
            import_evidence["imported"]["sha256"],
        ):
            raise RuntimeError(f"sealed {packet_slug} review differs from imported source receipt")
        generated, key_path = _review_bundle_paths(output, packet_slug)
        submitted = _completed_review_form(
            Path(str(import_evidence["imported"]["path"])),
            generated,
            label=f"sealed-{packet_slug}",
        )
        key = pd.read_csv(key_path, dtype={"montage_id": str, "prototype": int})
        mapping = key[["montage_id", "prototype"]].drop_duplicates().set_index(
            "montage_id"
        )["prototype"].to_dict()
        if set(submitted["montage_id"].astype(str)) != set(map(str, mapping)):
            raise RuntimeError(f"sealed {packet_slug} review/key montage coverage differs")
        if completed_montage_ids & set(map(str, mapping)):
            raise RuntimeError("sealed base/addendum reviews reuse montage IDs")
        completed_montage_ids |= set(map(str, mapping))
        for record in submitted.to_dict("records"):
            montage_id = str(record["montage_id"])
            prototype = int(mapping[montage_id])
            if str(prototype) in annotations:
                raise RuntimeError(
                    f"prototype {prototype} has multiple sealed completed assessments"
                )
            annotations[str(prototype)] = {
                "prototype": prototype,
                "packet": packet_slug,
                "montage_id": montage_id,
                "assessment": record,
            }
    exact_paths.update({
        "pathology": canonical_bundle / f"pathology_review_k{K}.json",
        "reviewed_report": canonical_bundle / f"aim4_corrected_reviewed_k{K}.json",
    })
    for key in ("pathology", "reviewed_report"):
        _validate_recorded_identity(receipt[key])
        if Path(str(receipt[key]["path"])).resolve(strict=True) != exact_paths[key].resolve(
            strict=True
        ):
            raise RuntimeError(f"sealed {key} path differs")
    pathology = _json_object(Path(str(receipt["pathology"]["path"])))
    reviewed = _json_object(Path(str(receipt["reviewed_report"]["path"])))
    numeric = _json_object(output / "analysis" / f"aim4_corrected_k{K}.json")
    expected_prototypes = {
        int(value)
        for value in (
            numeric["review_selection"]["base_packet"]
            + numeric["review_selection"]["attention_addendum_packet"]
        )
    }
    if set(map(int, annotations)) != expected_prototypes:
        raise RuntimeError("sealed reviews do not cover the exact locked selection")
    expected_claim_limits = [
        value for value in numeric.get("claim_limits", [])
        if not str(value).startswith("Pathology descriptions remain pending")
    ]
    expected_claim_limits.append(
        "Pathology descriptions are the imported structured assessments from the fresh "
        "corrected blinded packets."
    )
    expected_pathology = {
        "schema_version": 1,
        "component": "aim4_corrected_pathology_review",
        "status": "complete",
        "structured_not_concatenated": True,
        "annotations": annotations,
        "submission_imports": receipt["forms"],
    }
    expected_reviewed = {
        **numeric,
        "status": "ANALYSIS_AND_PATHOLOGY_REVIEW_COMPLETE",
        "claim_limits": expected_claim_limits,
        "pathology_review": expected_pathology,
    }
    if (
        _canonical(pathology) != _canonical(expected_pathology)
        or _canonical(reviewed) != _canonical(expected_reviewed)
        or int(receipt.get("n_completed_prototypes", -1)) != len(annotations)
    ):
        raise RuntimeError("sealed structured pathology review differs from completed forms")
    expected_bundle_files = {
        REVIEW_IMPORT_TRANSACTION_RECEIPT,
        "submissions/base_completed.csv",
        "submissions/attention_addendum_completed.csv",
        f"pathology_review_k{K}.json",
        f"aim4_corrected_reviewed_k{K}.json",
    }
    actual_bundle_files = {
        str(path.relative_to(canonical_bundle))
        for path in canonical_bundle.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_bundle_files != expected_bundle_files:
        raise RuntimeError("exact completed-review bundle inventory differs")
    return {
        "status": "PASS",
        "n_completed_prototypes": len(annotations),
        "receipt": _identity(receipt_path),
    }


def _manifest_files(output: Path, *, phase: str) -> dict[str, dict[str, Any]]:
    if phase == "numeric":
        excluded = {
            NUMERIC_COMPLETION_NAME, NUMERIC_VERIFICATION_NAME,
            "lineage_complete.json", VERIFICATION_NAME,
        }
    elif phase == "final":
        excluded = {"lineage_complete.json", VERIFICATION_NAME}
    else:
        raise ValueError(f"unknown manifest phase: {phase}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output.rglob("*")):
        relative_path = path.relative_to(output)
        if (
            not path.is_file()
            or (len(relative_path.parts) == 1 and path.name in excluded)
            or "_staging" in relative_path.parts
        ):
            continue
        relative = str(relative_path)
        evidence = _identity(path)
        evidence["path"] = relative
        files[relative] = evidence
    return files


def _manifest_inventory(output: Path, *, phase: str) -> set[str]:
    if phase == "numeric":
        excluded = {
            NUMERIC_COMPLETION_NAME, NUMERIC_VERIFICATION_NAME,
            "lineage_complete.json", VERIFICATION_NAME,
        }
    elif phase == "final":
        excluded = {"lineage_complete.json", VERIFICATION_NAME}
    else:
        raise ValueError(f"unknown manifest phase: {phase}")
    inventory: set[str] = set()
    for path in output.rglob("*"):
        relative = path.relative_to(output)
        if "_staging" in relative.parts:
            continue
        if len(relative.parts) == 1 and path.name in excluded:
            continue
        if path.is_file() or path.is_symlink():
            inventory.add(str(relative))
    return inventory


def _numeric_addendum_paths() -> set[str]:
    completion = Path("review_completion") / f"k{K}"
    return {
        "receipts/review_import.json",
        str(completion / REVIEW_IMPORT_TRANSACTION_RECEIPT),
        str(completion / "submissions" / "base_completed.csv"),
        str(completion / "submissions" / "attention_addendum_completed.csv"),
        str(completion / f"pathology_review_k{K}.json"),
        str(completion / f"aim4_corrected_reviewed_k{K}.json"),
    }


def _validate_manifest(output: Path, files: dict[str, Any], *, phase: str) -> None:
    if not isinstance(files, dict):
        raise RuntimeError("completion manifest artifact inventory is malformed")
    recorded_inventory = set(files)
    actual_inventory = _manifest_inventory(output, phase=phase)
    missing = recorded_inventory - actual_inventory
    unexpected = actual_inventory - recorded_inventory
    if missing:
        raise RuntimeError(f"completed artifact is missing: {sorted(missing)[0]}")
    if phase == "final" and unexpected:
        raise RuntimeError(f"unexpected artifact after final seal: {sorted(unexpected)[0]}")
    if phase == "numeric" and unexpected - _numeric_addendum_paths():
        raise RuntimeError(
            f"unexpected artifact after numeric seal: "
            f"{sorted(unexpected - _numeric_addendum_paths())[0]}"
        )
    for relative, recorded in files.items():
        relative_path = Path(str(relative))
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not isinstance(recorded, dict)
            or set(recorded) != {"path", "size_bytes", "sha256"}
            or recorded.get("path") != relative
        ):
            raise RuntimeError(f"malformed completion artifact identity: {relative}")
        path = output / relative
        actual = _identity(path)
        if actual["size_bytes"] != int(recorded["size_bytes"]) or actual["sha256"] != recorded["sha256"]:
            raise RuntimeError(f"completed artifact identity changed: {relative}")


def validate_numeric_completion(output: Path) -> dict[str, Any]:
    path = output / NUMERIC_COMPLETION_NAME
    payload = _json_object(path)
    if (
        payload.get("component") != "aim4_corrected_cap8192"
        or payload.get("status") != "numeric_completed_pathology_review_pending"
        or payload.get("output_root") != str(output)
        or not isinstance(payload.get("artifacts"), dict)
    ):
        raise RuntimeError("invalid numeric completion receipt")
    _validate_manifest(output, payload["artifacts"], phase="numeric")
    aggregate = hashlib.sha256(_canonical(payload["artifacts"]).encode()).hexdigest()
    if aggregate != payload.get("aggregate_sha256"):
        raise RuntimeError("numeric completion aggregate digest mismatch")
    return payload


def validate_numeric_verification(output: Path) -> dict[str, Any]:
    completion_path = output / NUMERIC_COMPLETION_NAME
    completion = validate_numeric_completion(output)
    path = output / NUMERIC_VERIFICATION_NAME
    receipt = _json_object(path)
    if (
        receipt.get("component") != "aim4_corrected_cap8192"
        or receipt.get("status") != "PASS"
        or receipt.get("receipt_role") != "immutable_numeric_verification_addendum"
        or receipt.get("output_root") != str(output)
        or receipt.get("numeric_completion") != _identity(completion_path)
        or receipt.get("numeric_aggregate_sha256") != completion["aggregate_sha256"]
        or receipt.get("replay", {}).get("status") != "PASS"
        or receipt.get("checks", {}).get("deterministic_statistical_replay") != "PASS"
    ):
        raise RuntimeError("numeric verification receipt does not bind a full-replay PASS")
    return {
        "status": "PASS",
        "numeric_completion": _identity(completion_path),
        "numeric_verification": _identity(path),
        "numeric_aggregate_sha256": completion["aggregate_sha256"],
    }


def finalize_numeric(bundle: InputBundle, output: Path, *, apply: bool) -> dict[str, Any]:
    """Seal numeric analysis and blinded pending packets before pathology review."""
    _validate_prepared(output, bundle)
    _assert_numeric_not_completed(output)
    profiles = validate_profiles(output, bundle, deep_candidates=True)
    spec = _json_object(output / "analysis" / f"specificity_k{K}.json")
    transport = _json_object(output / "analysis" / f"transport_k{K}.json")
    analysis_validation = validate_analysis(spec, transport)
    report_validation = validate_corrected_report(output, spec, transport)
    review_validation = validate_review_packets(output)
    result = {
        "status": "DRY_RUN" if not apply else "PASS",
        "profiles": profiles,
        "analysis": analysis_validation,
        "corrected_report": report_validation,
        "review_packets": review_validation,
        "pathology_review": "PENDING_EXPLICIT_COMPLETED_FORMS",
    }
    if not apply:
        return result
    completion_path = output / NUMERIC_COMPLETION_NAME
    if completion_path.exists() or completion_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite numeric completion: {completion_path}")
    files = _manifest_files(output, phase="numeric")
    payload = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "numeric_completed_pathology_review_pending",
        "created_at_utc": _utc_now(),
        "output_root": str(output),
        "artifacts": files,
        "aggregate_sha256": hashlib.sha256(_canonical(files).encode()).hexdigest(),
        "validation": result,
        "pathology_status": "fresh corrected blinded packets sealed; completed review pending",
    }
    _write_json_once(completion_path, payload)
    return payload


def finalize(bundle: InputBundle, output: Path, *, apply: bool) -> dict[str, Any]:
    _validate_prepared(output, bundle, require_live_source=False)
    _assert_not_completed(output)
    numeric_validation = validate_numeric_verification(output)
    profiles = validate_profiles(output, bundle, deep_candidates=True)
    spec = _json_object(output / "analysis" / f"specificity_k{K}.json")
    transport = _json_object(output / "analysis" / f"transport_k{K}.json")
    analysis_validation = validate_analysis(spec, transport)
    report_validation = validate_corrected_report(output, spec, transport)
    review_validation = validate_review_packets(output)
    completed_review_validation = validate_imported_reviews(output)
    result = {
        "status": "DRY_RUN" if not apply else "PASS",
        "profiles": profiles,
        "analysis": analysis_validation,
        "corrected_report": report_validation,
        "review_packets": review_validation,
        "numeric_seal": numeric_validation,
        "completed_pathology_review": completed_review_validation,
    }
    if not apply:
        return result
    completion = output / "lineage_complete.json"
    if completion.exists():
        raise FileExistsError(f"refusing to overwrite completion receipt: {completion}")
    files = _manifest_files(output, phase="final")
    payload = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "completed",
        "created_at_utc": _utc_now(),
        "output_root": str(output),
        "artifacts": files,
        "aggregate_sha256": hashlib.sha256(_canonical(files).encode()).hexdigest(),
        "validation": result,
        "pathology_status": "completed structured blinded pathology review imported and sealed",
    }
    _write_json_once(completion, payload)
    return payload


def _restore_structural_nan(value: Any, *, field: str | None = None) -> Any:
    """Restore only statistical nulls needed by legacy classification helpers."""
    if isinstance(value, dict):
        restored = {
            key: _restore_structural_nan(item, field=str(key))
            for key, item in value.items()
        }
        if restored.get("estimable") is False:
            for key in (
                "auc", "delta", "ci_low", "ci_high", "delta_auc_metastatic_minus_primary",
                "delta_ci_low", "delta_ci_high", "delta_se",
            ):
                if key in restored and restored[key] is None:
                    restored[key] = float("nan")
        return restored
    if isinstance(value, list):
        return [_restore_structural_nan(item, field=field) for item in value]
    # atlas.concentration deliberately uses NaN for an empty prototype and
    # final_readout calls np.isfinite(top_share). Strict JSON represents that
    # structural NaN as null, so restore this one concentration field too.
    if value is None and field == "top_share":
        return float("nan")
    return value


def _compare_records(actual: list[dict[str, Any]], expected: pd.DataFrame, label: str) -> None:
    expected_records = _sanitize_json(expected.to_dict("records"))
    if _canonical(actual) != _canonical(expected_records):
        raise RuntimeError(f"deterministic statistical replay differs: {label}")


def replay_statistics(output: Path) -> dict[str, Any]:
    """Replay the complete specificity/transport objects from sealed profiles."""
    spec = _json_object(output / "analysis" / f"specificity_k{K}.json")
    transport = _json_object(output / "analysis" / f"transport_k{K}.json")
    protocol = spec["protocol"]
    n_bootstrap = int(protocol["n_bootstrap"])
    seed = int(protocol["bootstrap_seed"])
    old_profile_root = legacy.PROFILE_DIR
    old_eval_root = legacy.paths.EVAL_ROOT
    old_effect = legacy._effect_table
    old_difference = legacy._auc_difference_table
    try:
        with tempfile.TemporaryDirectory(prefix="aim4-corrected-replay-") as temporary:
            replay_root = Path(temporary)
            legacy.PROFILE_DIR = output / "profiles"
            legacy.paths.EVAL_ROOT = replay_root
            legacy._effect_table = fixed_effect_table
            legacy._auc_difference_table = fixed_auc_difference_table
            args = SimpleNamespace(k=K, n_bootstrap=n_bootstrap, seed=seed)
            # The legacy commands print human-readable tables; verification
            # compares their complete machine objects and keeps stdout concise.
            with contextlib.redirect_stdout(io.StringIO()):
                legacy.cmd_specificity(args)
                legacy.cmd_transport(args)
            rebuilt_spec = _json_object(replay_root / f"e3b_specificity_k{K}.json")
            rebuilt_transport = _json_object(replay_root / f"e3b_transport_k{K}.json")
    finally:
        legacy.PROFILE_DIR = old_profile_root
        legacy.paths.EVAL_ROOT = old_eval_root
        legacy._effect_table = old_effect
        legacy._auc_difference_table = old_difference
    for rebuilt in (rebuilt_spec, rebuilt_transport):
        rebuilt["component"] = "aim4_corrected_cap8192"
        rebuilt["protocol"] = protocol
    if _canonical(spec) != _canonical(_sanitize_json(rebuilt_spec)):
        raise RuntimeError("deterministic full specificity replay differs")
    if _canonical(transport) != _canonical(_sanitize_json(rebuilt_transport)):
        raise RuntimeError("deterministic full transport replay differs")
    report = _json_object(output / "analysis" / f"aim4_corrected_k{K}.json")
    replay_readout = legacy.final_readout(
        _restore_structural_nan(spec), _restore_structural_nan(transport)
    )
    if _canonical(report["readout"]) != _canonical(_sanitize_json(replay_readout)):
        raise RuntimeError("final readout algebra differs from specificity/transport")
    return {
        "status": "PASS",
        "replayed_complete_analysis_objects": 2,
        "n_bootstrap": n_bootstrap,
    }


def _deep_numeric_checks(
    bundle: InputBundle, output: Path, *, replay: bool
) -> dict[str, Any]:
    for arm_name in ALL_ARMS:
        for seed in SEEDS:
            path = output / "attention" / f"{arm_name}_seed{seed}.h5"
            checkpoint = None if arm_name == "e0" else bundle.checkpoints[(ARM_TARGET[arm_name], seed)]
            _validate_attention_file(
                path,
                bundle.manifests[arm_name],
                seed=seed,
                feature_files=bundle.feature_files,
                expected_checkpoint=checkpoint,
                deep=True,
            )
    profiles = validate_profiles(output, bundle, deep_candidates=True)
    spec = _json_object(output / "analysis" / f"specificity_k{K}.json")
    transport = _json_object(output / "analysis" / f"transport_k{K}.json")
    analysis_validation = validate_analysis(spec, transport)
    report_validation = validate_corrected_report(output, spec, transport)
    review_validation = validate_review_packets(output)
    replay_result = replay_statistics(output) if replay else {"status": "SKIPPED_BY_FLAG"}
    return {
        "profiles": profiles,
        "analysis": analysis_validation,
        "corrected_report": report_validation,
        "review_packets": review_validation,
        "replay": replay_result,
    }


def verify_numeric(bundle: InputBundle, output: Path) -> dict[str, Any]:
    """Full archival verification of sealed numeric results and pending packets."""
    _validate_prepared(output, bundle, require_live_source=False)
    completion_path = output / NUMERIC_COMPLETION_NAME
    completion = validate_numeric_completion(output)
    checked = _deep_numeric_checks(bundle, output, replay=True)
    return {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "PASS",
        "receipt_role": "immutable_numeric_verification_addendum",
        "output_root": str(output),
        "numeric_completion": _identity(completion_path),
        "numeric_aggregate_sha256": completion["aggregate_sha256"],
        "checks": {
            "frozen_source_snapshot": "PASS",
            "numeric_completion_inventory_and_hashes": "PASS",
            "all_15_attention_files_and_tile_order": "PASS",
            "profile_and_candidate_exact_coverage": "PASS",
            "fixed_32_prototype_families": "PASS",
            "deterministic_statistical_replay": "PASS",
            "fresh_separate_pending_review_packets": "PASS",
        },
        **checked,
    }


def verify_output(bundle: InputBundle, output: Path, *, replay: bool) -> dict[str, Any]:
    # Archival verification authenticates the frozen snapshot. It deliberately
    # does not require a future mutable worktree to retain the same bytes.
    _validate_prepared(output, bundle, require_live_source=False)
    numeric_validation = validate_numeric_verification(output)
    completion_path = output / "lineage_complete.json"
    if not completion_path.is_file():
        raise RuntimeError("output has no completion receipt")
    completion = _json_object(completion_path)
    if completion.get("status") != "completed" or completion.get("component") != "aim4_corrected_cap8192":
        raise RuntimeError("invalid completion receipt")
    _validate_manifest(output, completion.get("artifacts", {}), phase="final")
    aggregate = hashlib.sha256(_canonical(completion["artifacts"]).encode()).hexdigest()
    if aggregate != completion.get("aggregate_sha256"):
        raise RuntimeError("completion aggregate digest mismatch")
    checked = _deep_numeric_checks(bundle, output, replay=replay)
    completed_review_validation = validate_imported_reviews(output)
    replay_result = checked["replay"]
    result = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "PASS",
        "receipt_role": "immutable_completion_verification_addendum",
        "output_root": str(output),
        "completion_aggregate_sha256": completion["aggregate_sha256"],
        "checks": {
            "upstream_identities": "PASS",
            "completion_inventory_and_hashes": "PASS",
            "all_15_attention_files_and_tile_order": "PASS",
            "profile_and_candidate_exact_coverage": "PASS",
            "fixed_32_prototype_families": "PASS",
            "deterministic_statistical_replay": replay_result["status"],
            "fresh_separate_review_packets": "PASS",
        },
        "numeric_seal": numeric_validation,
        "profiles": checked["profiles"],
        "analysis": checked["analysis"],
        "review_packets": checked["review_packets"],
        "completed_pathology_review": completed_review_validation,
        "replay": replay_result,
        "completion": _identity(completion_path),
    }
    return result


def _seal_full_verification(
    output: Path, result: dict[str, Any], *, name: str, receipt_role: str
) -> dict[str, Any]:
    if (
        result.get("status") != "PASS"
        or result.get("receipt_role") != receipt_role
        or result.get("replay", {}).get("status") != "PASS"
        or result.get("checks", {}).get("deterministic_statistical_replay") != "PASS"
    ):
        raise RuntimeError("only a full deterministic verification PASS may be sealed")
    path = output / name
    if path.exists() or path.is_symlink():
        existing = _json_object(path)
        if _canonical(existing) != _canonical(result):
            raise RuntimeError("immutable verification receipt differs from the current full replay")
    else:
        _write_json_once(path, result)
    return _identity(path)


def seal_numeric_verification(output: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Exclusively seal the numeric full-replay PASS while pathology is pending."""
    return _seal_full_verification(
        output,
        result,
        name=NUMERIC_VERIFICATION_NAME,
        receipt_role="immutable_numeric_verification_addendum",
    )


def seal_verification(output: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Exclusively seal a final full-replay PASS after completed pathology."""
    return _seal_full_verification(
        output,
        result,
        name=VERIFICATION_NAME,
        receipt_role="immutable_completion_verification_addendum",
    )


def _add_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-e4-root", type=Path, required=True)
    parser.add_argument("--aim2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "preflight", "prepare", "attention", "assign", "analyze", "review-packets",
        "finalize-numeric", "verify-numeric", "import-reviews", "finalize", "verify-output",
    ):
        command = commands.add_parser(name)
        _add_roots(command)
        if name in {
            "prepare", "attention", "assign", "analyze", "review-packets",
            "finalize-numeric", "import-reviews", "finalize",
        }:
            command.add_argument("--apply", action="store_true")
        if name == "attention":
            command.add_argument("--arm", action="append", choices=TARGET_ARMS)
            command.add_argument("--seed", action="append", type=int, choices=SEEDS)
        if name == "analyze":
            command.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
        if name == "review-packets":
            command.add_argument("--slide-root", type=Path, default=Path("/mnt/d/YC.Liu/slides/colon"))
            command.add_argument("--tiles-per-montage", type=int, default=12)
            command.add_argument("--tile-px", type=int, default=256)
            command.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
        if name == "import-reviews":
            command.add_argument("--base-form", type=Path, required=True)
            command.add_argument("--attention-addendum-form", type=Path, required=True)
        if name == "verify-output":
            command.add_argument("--skip-statistical-replay", action="store_true")
    args = parser.parse_args()
    output_exists = args.command not in {"preflight", "prepare"}
    archival_verification = args.command in ARCHIVAL_INPUT_COMMANDS
    input_root, aim2_root, output = _validate_roots(
        args.input_e4_root,
        args.aim2_root,
        args.output_root,
        output_exists=output_exists,
        input_exists=not archival_verification,
    )
    bundle = (
        validate_archival_inputs(input_root, aim2_root, output, deep_attention=True)
        if archival_verification
        else validate_inputs(input_root, aim2_root, deep_attention=True)
    )
    lock = (
        # Full verification seals a new immutable receipt, so it also takes an
        # exclusive campaign lock.  A skip-replay diagnostic shares the same
        # conservative lock discipline to avoid observing a concurrent write.
        _run_lock(output, exclusive=True)
        if args.command not in {"preflight", "prepare"}
        else contextlib.nullcontext()
    )
    with lock:
        if args.command == "preflight":
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"proposed output root already exists: {output}")
            if not output.parent.is_dir() or not os.access(output.parent, os.W_OK):
                raise RuntimeError(
                    f"proposed output parent must exist and be writable: {output.parent}"
                )
            result = {
                "status": "PASS",
                "output_root_absent": True,
                "output_parent_free_bytes": int(shutil.disk_usage(output.parent).free),
                "cuda_required_for_attention_apply": True,
                **bundle.receipt["validation"],
            }
        elif args.command == "prepare":
            if not args.apply:
                result = {"status": "DRY_RUN", "would_create": str(output)}
            else:
                result = {"status": "PASS", "output_root": str(prepare_run(bundle, output))}
        elif args.command == "attention":
            result = export_target_attention(
                bundle, output,
                arms=args.arm or TARGET_ARMS,
                seeds=args.seed or SEEDS,
                apply=args.apply,
            )
        elif args.command == "assign":
            result = assign_profiles(bundle, output, apply=args.apply)
        elif args.command == "analyze":
            result = analyze(bundle, output, n_bootstrap=args.n_bootstrap, apply=args.apply)
        elif args.command == "review-packets":
            result = make_review_packets(
                bundle,
                output,
                slide_root=args.slide_root,
                tiles_per_montage=args.tiles_per_montage,
                tile_px=args.tile_px,
                seed=args.seed,
                apply=args.apply,
            )
        elif args.command == "finalize-numeric":
            result = finalize_numeric(bundle, output, apply=args.apply)
        elif args.command == "verify-numeric":
            result = verify_numeric(bundle, output)
            numeric_identity = seal_numeric_verification(output, result)
            result = {**result, "sealed_verification": numeric_identity}
        elif args.command == "import-reviews":
            result = import_completed_reviews(
                bundle,
                output,
                base_form=args.base_form,
                attention_addendum_form=args.attention_addendum_form,
                apply=args.apply,
            )
        elif args.command == "finalize":
            result = finalize(bundle, output, apply=args.apply)
        else:
            full_replay = not args.skip_statistical_replay
            result = verify_output(bundle, output, replay=full_replay)
            if full_replay:
                verification_identity = seal_verification(output, result)
                result = {**result, "sealed_verification": verification_identity}
    print(json.dumps(_sanitize_json(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
