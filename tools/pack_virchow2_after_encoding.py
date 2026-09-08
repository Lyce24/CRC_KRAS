#!/usr/bin/env python3
"""Wait for the frozen Virchow2 queue, then publish two verified packs.

Outputs (both float16 and both retaining coordinates):

* ``packed_virchow2_full_2560``: the complete TRIDENT Virchow2 embedding.
* ``packed_virchow2_cls_1280``: columns ``[0:1280]``, the class token.

The production TRIDENT Virchow2 adapter returns
``concat(class_token, mean(non-register patch tokens))``.  Its class token is
1280-dimensional, so the first half of each saved 2560-dimensional row is the
CLS representation.  The CLS pack is published first directly from validated
H5 columns ``[0:1280]``.  The full pack is then materialised and its float16
prefix is proven byte-identical to CLS before the completion receipt is issued.

This program is deliberately a completion gate, not an encoder controller. It
never starts or stops the live encoder and cannot publish a pack while the
queue still has pending, processing, retry, failed, or blocked rows.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from oceanpath.datasets.packed import (
    COORDS_FILE,
    FEATURES_FILE,
    INDEX_FILE,
    META_FILE,
    PackedMeta,
    feature_inventory_sha256,
    pack_features,
    validate_packed_dir,
)

REPO = Path(__file__).resolve().parents[1]
QUEUE_DB = Path("/home/yc_liu/projects/OceanPath-colon/outputs/colon_stream/queue.sqlite")
FEATURE_DIR = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/"
    "20x_224px_0px_overlap_mpp0.5/features_virchow2"
)
FULL_PACK_DIR = FEATURE_DIR.with_name("packed_virchow2_full_2560")
CLS_PACK_DIR = FEATURE_DIR.with_name("packed_virchow2_cls_1280")
RECEIPT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/_stream_receipts"
)
RUN_DIR = REPO / "outputs" / "virchow2_packing"
STATE_PATH = RUN_DIR / "state.json"
COMPLETION_PATH = RUN_DIR / "packing_completion.json"

# Cohort size this completion gate must see before it will publish. Defaults to
# the 2,087-slide four-cohort study; raise it with --expected-slides when a new
# cohort has been added (Orion took it to 2,128). Passing the default reproduces
# the sealed run exactly.
EXPECTED_SLIDES = 2_087
SOURCE_DIM = 2_560
CLS_DIM = 1_280
CHECKPOINT_SHA256 = "14244fbaa5409452f6a6ae01b5d2dc452f2399e2210e696d0f7ef04bb4838666"
IMPLEMENTATION_SHA256 = "c186b18b06c755fead1c3f6ea9a7e8a9a2ce167e6b2a8c04525599d79fd482de"
TRIDENT_LAYOUT = "concat(class_token, mean(non-register patch_tokens))"
COPY_CHUNK_ROWS = 8_192
_SHA256_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}


def _stat_identity(path: Path) -> tuple[str, int, int, int, int, int]:
    stat = path.stat()
    return (
        str(path.resolve()),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _sha256(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    before = _stat_identity(path)
    cached = _SHA256_CACHE.get(before)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    after = _stat_identity(path)
    if after != before:
        raise RuntimeError(f"Artifact changed while hashing: {path}")
    result = digest.hexdigest()
    _SHA256_CACHE[before] = result
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_state(stage: str, **details: Any) -> None:
    payload = {
        "schema_version": 1,
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **details,
    }
    _atomic_json(STATE_PATH, payload)
    print(f"STATE {stage}: {json.dumps(details, sort_keys=True)}", flush=True)


def _queue_rows() -> tuple[dict[str, int], list[str]]:
    if not QUEUE_DB.is_file():
        raise FileNotFoundError(QUEUE_DB)
    uri = f"file:{QUEUE_DB}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=30) as connection:
        counts = {
            str(status): int(count)
            for status, count in connection.execute(
                "SELECT status, COUNT(*) FROM stream_jobs GROUP BY status"
            )
        }
        output_ids = [
            str(row[0])
            for row in connection.execute("SELECT output_id FROM stream_jobs ORDER BY output_id")
        ]
    for status in ("pending", "processing", "complete", "retry", "failed", "blocked"):
        counts.setdefault(status, 0)
    counts["total"] = len(output_ids)
    return counts, output_ids


def _queue_snapshot_sha256() -> str:
    """Hash the immutable scientific identity and completion evidence of the queue."""

    uri = f"file:{QUEUE_DB}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=30) as connection:
        rows = connection.execute(
            """
            SELECT output_id, source_key, config_key, status, receipt_path, result_json
            FROM stream_jobs
            ORDER BY output_id
            """
        ).fetchall()
    payload = [
        {
            "output_id": str(output_id),
            "source_key": str(source_key),
            "config_key": str(config_key),
            "status": str(status),
            "receipt_path": str(receipt_path or ""),
            "result_json": str(result_json or ""),
        }
        for output_id, source_key, config_key, status, receipt_path, result_json in rows
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _queue_is_complete(counts: dict[str, int]) -> bool:
    return (
        counts.get("total") == EXPECTED_SLIDES
        and counts.get("complete") == EXPECTED_SLIDES
        and all(
            counts.get(status, 0) == 0
            for status in ("pending", "processing", "retry", "failed", "blocked")
        )
    )


def _wait_for_queue(poll_seconds: float) -> tuple[dict[str, int], list[str]]:
    previous: dict[str, int] | None = None
    while True:
        counts, output_ids = _queue_rows()
        if counts["total"] != EXPECTED_SLIDES or len(set(output_ids)) != EXPECTED_SLIDES:
            raise RuntimeError(
                f"Expected {EXPECTED_SLIDES} unique queue rows, got "
                f"total={counts['total']} unique={len(set(output_ids))}"
            )
        if counts["failed"] or counts["blocked"]:
            _write_state("blocked", queue=counts)
            raise RuntimeError(f"Virchow2 queue contains failed/blocked jobs: {counts}")
        if _queue_is_complete(counts):
            _write_state("queue_complete", queue=counts)
            return counts, output_ids
        if counts != previous:
            _write_state("waiting_for_encoding", queue=counts)
            previous = counts
        time.sleep(poll_seconds)


def _validate_source(output_ids: list[str]) -> tuple[str, dict[str, int]]:
    expected = set(output_ids)
    actual = {path.stem for path in FEATURE_DIR.glob("*.h5")}
    if actual != expected:
        raise RuntimeError(
            "Virchow2 H5 inventory does not equal the frozen queue: "
            f"expected={len(expected)} actual={len(actual)} "
            f"missing={sorted(expected - actual)[:5]} extra={sorted(actual - expected)[:5]}"
        )

    implementation_hashes: set[str] = set()
    patch_counts: dict[str, int] = {}
    for index, slide_id in enumerate(output_ids):
        receipt_path = RECEIPT_ROOT / slide_id / "virchow2_feat.json"
        try:
            receipt = json.loads(receipt_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Missing or malformed receipt: {receipt_path}") from exc
        if receipt.get("stage") != "virchow2_feat":
            raise RuntimeError(f"Wrong stage in {receipt_path}")
        source = receipt.get("source", {})
        validation = receipt.get("validation", {})
        checkpoints = receipt.get("checkpoint_hashes", {})
        if source.get("output_id") != slide_id:
            raise RuntimeError(f"Receipt identity mismatch: {receipt_path}")
        if int(validation.get("feature_dim", -1)) != SOURCE_DIM:
            raise RuntimeError(f"Receipt feature dimension mismatch: {receipt_path}")
        if checkpoints.get("virchow2") != CHECKPOINT_SHA256:
            raise RuntimeError(f"Receipt checkpoint mismatch: {receipt_path}")
        implementation_hashes.add(str(receipt.get("implementation_hash", "")))

        h5_path = FEATURE_DIR / f"{slide_id}.h5"
        with h5py.File(h5_path, "r") as handle:
            if "features" not in handle or "coords" not in handle:
                raise RuntimeError(f"{h5_path} lacks features or coords")
            features = handle["features"]
            coords = handle["coords"]
            encoder = features.attrs.get("encoder", "")
            if isinstance(encoder, bytes):
                encoder = encoder.decode()
            if np.dtype(features.dtype) != np.dtype("float32"):
                raise RuntimeError(
                    f"Expected float32 Virchow2 features in {h5_path}, got {features.dtype}"
                )
            if np.dtype(coords.dtype) != np.dtype("int64"):
                raise RuntimeError(
                    f"Expected int64 Virchow2 coordinates in {h5_path}, got {coords.dtype}"
                )
            if str(encoder) != "virchow2":
                raise RuntimeError(
                    f"Expected encoder='virchow2' in {h5_path}, got {encoder!r}"
                )
            name = features.attrs.get("name", "")
            if isinstance(name, bytes):
                name = name.decode()
            if str(name) != slide_id:
                raise RuntimeError(
                    f"Expected feature name={slide_id!r} in {h5_path}, got {name!r}"
                )
            shape = tuple(int(value) for value in features.shape)
            if len(shape) == 3 and shape[0] == 1:
                n_patches, feature_dim = shape[1], shape[2]
            elif len(shape) == 2:
                n_patches, feature_dim = shape
            else:
                raise RuntimeError(f"Unexpected feature shape in {h5_path}: {shape}")
            coord_shape = tuple(int(value) for value in coords.shape)
            if len(coord_shape) == 3 and coord_shape[0] == 1:
                coord_rows, coord_dim = coord_shape[1], coord_shape[2]
            elif len(coord_shape) == 2:
                coord_rows, coord_dim = coord_shape
            else:
                raise RuntimeError(f"Unexpected coordinate shape in {h5_path}: {coord_shape}")
            if (
                n_patches <= 0
                or feature_dim != SOURCE_DIM
                or coord_rows != n_patches
                or coord_dim != 2
            ):
                raise RuntimeError(
                    f"Invalid H5 structure in {h5_path}: features={shape}, coords={coord_shape}"
                )
            if int(validation.get("patch_count", -1)) != n_patches:
                raise RuntimeError(f"Receipt patch count mismatch: {receipt_path}")
            coord_min = int(np.min(coords))
            coord_max = int(np.max(coords))
            int32 = np.iinfo(np.int32)
            if coord_min < int32.min or coord_max > int32.max:
                raise RuntimeError(
                    f"Coordinates exceed packed int32 range in {h5_path}: "
                    f"[{coord_min}, {coord_max}]"
                )
            patch_counts[slide_id] = n_patches
        if (index + 1) % 250 == 0:
            print(f"validated {index + 1}/{len(output_ids)} Virchow2 H5 receipts", flush=True)

    if implementation_hashes != {IMPLEMENTATION_SHA256}:
        raise RuntimeError(
            "Virchow2 extraction implementation hash mismatch: "
            f"expected={IMPLEMENTATION_SHA256}, got={sorted(implementation_hashes)}"
        )
    return feature_inventory_sha256(FEATURE_DIR), patch_counts


def _validate_pack(
    pack_dir: Path,
    *,
    expected_dim: int,
    source_hash: str,
    expected_patch_counts: dict[str, int],
) -> PackedMeta:
    meta = validate_packed_dir(pack_dir, verify_source=source_hash)
    if (
        meta.n_slides != EXPECTED_SLIDES
        or meta.feat_dim != expected_dim
        or meta.feat_dtype != "float16"
        or not meta.has_coords
        or meta.coord_dim != 2
        or meta.total_patches != sum(expected_patch_counts.values())
    ):
        raise RuntimeError(f"Unexpected packed metadata at {pack_dir}: {meta}")
    index = pd.read_parquet(pack_dir / INDEX_FILE, columns=["slide_id", "n_patches"])
    actual_ids = set(index["slide_id"].astype(str))
    expected_ids = set(expected_patch_counts)
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"Packed index mismatch at {pack_dir}: "
            f"missing={sorted(expected_ids - actual_ids)[:5]} "
            f"extra={sorted(actual_ids - expected_ids)[:5]}"
        )
    actual_patch_counts = {
        str(row.slide_id): int(row.n_patches) for row in index.itertuples(index=False)
    }
    if actual_patch_counts != expected_patch_counts:
        mismatched = sorted(
            slide_id
            for slide_id in expected_ids
            if actual_patch_counts.get(slide_id) != expected_patch_counts[slide_id]
        )
        raise RuntimeError(
            f"Packed per-slide patch counts mismatch at {pack_dir}: {mismatched[:5]}"
        )
    return meta


def _build_or_validate_full(
    source_hash: str, expected_patch_counts: dict[str, int]
) -> PackedMeta:
    if FULL_PACK_DIR.exists():
        print(f"validating existing full pack: {FULL_PACK_DIR}", flush=True)
        return _validate_pack(
            FULL_PACK_DIR,
            expected_dim=SOURCE_DIM,
            source_hash=source_hash,
            expected_patch_counts=expected_patch_counts,
        )
    _write_state("packing_full_2560", destination=str(FULL_PACK_DIR))
    pack_features(
        feature_dir=FEATURE_DIR,
        pack_dir=FULL_PACK_DIR,
        feat_dtype="float16",
        include_coords=True,
        overwrite=False,
        stream_chunk_size=16_384,
        verify_source_unchanged=True,
    )
    return _validate_pack(
        FULL_PACK_DIR,
        expected_dim=SOURCE_DIM,
        source_hash=source_hash,
        expected_patch_counts=expected_patch_counts,
    )


def _representation_payload(
    source_hash: str, *, features_sha256: str | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "encoder": "virchow2",
        "representation": "class_token",
        "source_feature_dim": SOURCE_DIM,
        "output_feature_dim": CLS_DIM,
        "selected_columns": {"start_inclusive": 0, "stop_exclusive": CLS_DIM},
        "source_layout": TRIDENT_LAYOUT,
        "register_tokens_excluded_from_mean": 4,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_dir": str(FEATURE_DIR),
        "companion_full_pack": str(FULL_PACK_DIR),
        "source_inventory_sha256": source_hash,
        "derivation": "float16 cast of validated source H5 columns [0:1280]",
        "relationship_contract": (
            "final completion requires byte-identical equality to the float16 "
            "full-pack prefix"
        ),
    }
    if features_sha256 is not None:
        payload["features_sha256"] = features_sha256
        payload["full_prefix_sha256"] = features_sha256
    return payload


def _packed_slide_order(pack_dir: Path) -> list[str]:
    index = pd.read_parquet(
        pack_dir / INDEX_FILE,
        columns=["slide_id", "offset", "n_patches"],
    )
    if index["slide_id"].duplicated().any():
        raise RuntimeError(f"Duplicate slide IDs in packed index: {pack_dir / INDEX_FILE}")
    return index["slide_id"].astype(str).tolist()


def _packed_spans(pack_dir: Path) -> dict[str, tuple[int, int]]:
    index = pd.read_parquet(
        pack_dir / INDEX_FILE,
        columns=["slide_id", "offset", "n_patches"],
    )
    if index["slide_id"].duplicated().any():
        raise RuntimeError(f"Duplicate slide IDs in packed index: {pack_dir / INDEX_FILE}")
    return {
        str(row.slide_id): (int(row.offset), int(row.n_patches))
        for row in index.itertuples(index=False)
    }


def _full_prefix_sha256(full_meta: PackedMeta, *, slide_order: list[str]) -> str:
    """Hash the full-pack CLS prefix in a caller-specified logical slide order.

    Packed stores are addressed through ``index.parquet``; physical block order is
    not part of their scientific identity.  Hashing in the CLS pack's slide order
    proves row-wise equality even when two valid packers chose different global
    block orders.
    """

    spans = _packed_spans(FULL_PACK_DIR)
    if set(slide_order) != set(spans):
        raise RuntimeError("Full and CLS packed slide-ID sets differ")
    source_features = np.memmap(
        FULL_PACK_DIR / FEATURES_FILE,
        dtype=np.dtype(full_meta.feat_dtype),
        mode="r",
        shape=(full_meta.total_patches, SOURCE_DIM),
    )
    digest = hashlib.sha256()
    try:
        for slide_id in slide_order:
            offset, n_patches = spans[slide_id]
            for relative_start in range(0, n_patches, COPY_CHUNK_ROWS):
                start = offset + relative_start
                stop = offset + min(relative_start + COPY_CHUNK_ROWS, n_patches)
                block = np.ascontiguousarray(source_features[start:stop, :CLS_DIM])
                digest.update(memoryview(block).cast("B"))
    finally:
        del source_features
    return digest.hexdigest()


def _full_coords_sha256(full_meta: PackedMeta, *, slide_order: list[str]) -> str:
    """Hash full-pack coordinates in the CLS pack's logical slide order."""

    spans = _packed_spans(FULL_PACK_DIR)
    if set(slide_order) != set(spans):
        raise RuntimeError("Full and CLS packed slide-ID sets differ")
    source_coords = np.memmap(
        FULL_PACK_DIR / COORDS_FILE,
        dtype=np.int32,
        mode="r",
        shape=(full_meta.total_patches, full_meta.coord_dim),
    )
    digest = hashlib.sha256()
    try:
        for slide_id in slide_order:
            offset, n_patches = spans[slide_id]
            for relative_start in range(0, n_patches, COPY_CHUNK_ROWS):
                start = offset + relative_start
                stop = offset + min(relative_start + COPY_CHUNK_ROWS, n_patches)
                block = np.ascontiguousarray(source_coords[start:stop])
                digest.update(memoryview(block).cast("B"))
    finally:
        del source_coords
    return digest.hexdigest()


def _validate_cls_relationship(
    full_meta: PackedMeta,
    source_hash: str,
    expected_patch_counts: dict[str, int],
    *,
    verify_full_prefix: bool,
) -> PackedMeta:
    meta = _validate_pack(
        CLS_PACK_DIR,
        expected_dim=CLS_DIM,
        source_hash=source_hash,
        expected_patch_counts=expected_patch_counts,
    )
    representation_path = CLS_PACK_DIR / "representation.json"
    try:
        representation = json.loads(representation_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Missing or malformed CLS representation provenance: {representation_path}"
        ) from exc
    expected = _representation_payload(source_hash)
    for key, value in expected.items():
        if representation.get(key) != value:
            raise RuntimeError(
                f"CLS representation provenance mismatch for {key!r}: "
                f"{representation.get(key)!r} != {value!r}"
            )
    declared = str(representation.get("features_sha256", ""))
    if len(declared) != 64 or representation.get("full_prefix_sha256") != declared:
        raise RuntimeError("CLS representation lacks matching feature/prefix SHA-256 evidence")
    actual = _sha256(CLS_PACK_DIR / FEATURES_FILE)
    if actual != declared:
        raise RuntimeError("CLS features do not match their recorded SHA-256")
    if verify_full_prefix:
        cls_order = _packed_slide_order(CLS_PACK_DIR)
        if _full_prefix_sha256(full_meta, slide_order=cls_order) != declared:
            raise RuntimeError(
                "CLS features are not the exact per-slide [0:1280] prefix of the full pack"
            )
        if _full_coords_sha256(full_meta, slide_order=cls_order) != _sha256(
            CLS_PACK_DIR / COORDS_FILE
        ):
            raise RuntimeError("Full and CLS packed coordinates differ by slide ID")
    return meta


def _validate_cls_source_pack(
    source_hash: str,
    expected_patch_counts: dict[str, int],
) -> PackedMeta:
    meta = _validate_pack(
        CLS_PACK_DIR,
        expected_dim=CLS_DIM,
        source_hash=source_hash,
        expected_patch_counts=expected_patch_counts,
    )
    representation_path = CLS_PACK_DIR / "representation.json"
    try:
        representation = json.loads(representation_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Missing or malformed CLS representation provenance: {representation_path}"
        ) from exc
    for key, value in _representation_payload(source_hash).items():
        if representation.get(key) != value:
            raise RuntimeError(
                f"CLS representation provenance mismatch for {key!r}: "
                f"{representation.get(key)!r} != {value!r}"
            )
    declared = str(representation.get("features_sha256", ""))
    if len(declared) != 64 or representation.get("full_prefix_sha256") != declared:
        raise RuntimeError("CLS representation lacks matching feature/prefix SHA-256 evidence")
    if _sha256(CLS_PACK_DIR / FEATURES_FILE) != declared:
        raise RuntimeError("CLS features do not match their recorded SHA-256")
    return meta


def _read_h5_rows(dataset: h5py.Dataset, start: int, stop: int) -> np.ndarray:
    if dataset.ndim == 3:
        return np.asarray(dataset[0, start:stop, :])
    return np.asarray(dataset[start:stop, :])


def _build_cls_pack(
    source_hash: str,
    expected_patch_counts: dict[str, int],
) -> PackedMeta:
    if CLS_PACK_DIR.exists():
        print(f"validating existing CLS pack: {CLS_PACK_DIR}", flush=True)
        return _validate_cls_source_pack(source_hash, expected_patch_counts)

    _write_state("packing_cls_1280", destination=str(CLS_PACK_DIR))
    staging = Path(
        tempfile.mkdtemp(prefix=f".{CLS_PACK_DIR.name}.tmp-", dir=str(CLS_PACK_DIR.parent))
    )
    try:
        prefix_digest = hashlib.sha256()
        rows: list[dict[str, int | str]] = []
        offset = 0
        feat_max = float(np.finfo(np.float16).max)
        with (
            (staging / FEATURES_FILE).open("wb") as feature_output,
            (staging / COORDS_FILE).open("wb") as coord_output,
        ):
            for slide_index, (slide_id, n_patches) in enumerate(
                expected_patch_counts.items()
            ):
                with h5py.File(FEATURE_DIR / f"{slide_id}.h5", "r") as handle:
                    feature_dataset = handle["features"]
                    coord_dataset = handle["coords"]
                    for start in range(0, n_patches, COPY_CHUNK_ROWS):
                        stop = min(start + COPY_CHUNK_ROWS, n_patches)
                        source_block = _read_h5_rows(feature_dataset, start, stop)
                        if not np.isfinite(source_block).all():
                            raise RuntimeError(
                                f"Non-finite Virchow2 values in {slide_id}.h5"
                            )
                        block = source_block[:, :CLS_DIM]
                        if np.any(np.abs(block) > feat_max):
                            block = np.clip(block, -feat_max, feat_max)
                        converted = np.ascontiguousarray(block, dtype=np.float16)
                        prefix_digest.update(memoryview(converted).cast("B"))
                        converted.tofile(feature_output)

                        if coord_dataset.ndim == 3:
                            coords = np.asarray(coord_dataset[0, start:stop, :])
                        else:
                            coords = np.asarray(coord_dataset[start:stop, :])
                        if not np.isfinite(coords).all():
                            raise RuntimeError(
                                f"Non-finite Virchow2 coordinates in {slide_id}.h5"
                            )
                        np.ascontiguousarray(coords, dtype=np.int32).tofile(coord_output)
                rows.append(
                    {"slide_id": slide_id, "offset": offset, "n_patches": n_patches}
                )
                offset += n_patches
                if (slide_index + 1) % 50 == 0:
                    print(
                        f"CLS pack: {slide_index + 1}/{len(expected_patch_counts)} "
                        f"slides, {offset} patch rows",
                        flush=True,
                    )
            feature_output.flush()
            coord_output.flush()
            os.fsync(feature_output.fileno())
            os.fsync(coord_output.fileno())

        prefix_sha256 = prefix_digest.hexdigest()
        if _sha256(staging / FEATURES_FILE) != prefix_sha256:
            raise RuntimeError("CLS staging bytes do not match their streaming digest")

        pd.DataFrame(rows).to_parquet(staging / INDEX_FILE, index=False)
        cls_meta = PackedMeta(
            schema_version=1,
            feat_dim=CLS_DIM,
            feat_dtype="float16",
            coord_dim=2,
            n_slides=len(rows),
            total_patches=offset,
            source_dir=str(FEATURE_DIR),
            source_inventory_sha256=source_hash,
            has_coords=True,
        )
        (staging / META_FILE).write_text(json.dumps(asdict(cls_meta), indent=2, sort_keys=True))
        (staging / "representation.json").write_text(
            json.dumps(
                _representation_payload(source_hash, features_sha256=prefix_sha256),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        validate_packed_dir(staging, verify_source=source_hash)
        if feature_inventory_sha256(FEATURE_DIR) != source_hash:
            raise RuntimeError("Source H5 inventory changed while CLS was being packed")

        staging.rename(CLS_PACK_DIR)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return _validate_cls_source_pack(source_hash, expected_patch_counts)


def _cleanup_stale_staging(target: Path) -> None:
    for path in sorted(target.parent.glob(f".{target.name}.tmp-*")):
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"Refusing unsafe stale staging cleanup: {path}")
        print(f"removing stale interrupted staging directory: {path}", flush=True)
        shutil.rmtree(path)


def _artifact_summary(pack_dir: Path, meta: PackedMeta) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(pack_dir),
        "meta": asdict(meta),
        "meta_sha256": _sha256(pack_dir / META_FILE),
        "index_sha256": _sha256(pack_dir / INDEX_FILE),
        "features_sha256": _sha256(pack_dir / FEATURES_FILE),
        "features_size_bytes": (pack_dir / FEATURES_FILE).stat().st_size,
        "coords_size_bytes": (pack_dir / COORDS_FILE).stat().st_size if meta.has_coords else 0,
    }
    if meta.has_coords:
        summary["coords_sha256"] = _sha256(pack_dir / COORDS_FILE)
    representation = pack_dir / "representation.json"
    if representation.is_file():
        summary["representation_sha256"] = _sha256(representation)
    return summary


def _validate_completion_contract(prior: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "status": "complete",
        "source_dir": str(FEATURE_DIR),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "implementation_sha256": IMPLEMENTATION_SHA256,
    }
    for key, value in expected.items():
        if prior.get(key) != value:
            raise RuntimeError(
                f"Completion receipt contract mismatch for {key!r}: "
                f"{prior.get(key)!r} != {value!r}"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if COMPLETION_PATH.is_file():
        prior = json.loads(COMPLETION_PATH.read_text())
        _validate_completion_contract(prior)
        counts, output_ids = _queue_rows()
        if not _queue_is_complete(counts):
            raise RuntimeError("Completion receipt exists but the queue is no longer complete")
        source_hash, patch_counts = _validate_source(output_ids)
        if prior.get("source_inventory_sha256") != source_hash:
            raise RuntimeError("Completion receipt is stale relative to the H5 inventory")
        if prior.get("queue_snapshot_sha256") != _queue_snapshot_sha256():
            raise RuntimeError("Completion receipt is stale relative to the queue snapshot")
        full_meta = _validate_pack(
            FULL_PACK_DIR,
            expected_dim=SOURCE_DIM,
            source_hash=source_hash,
            expected_patch_counts=patch_counts,
        )
        cls_meta = _validate_cls_relationship(
            full_meta,
            source_hash,
            patch_counts,
            verify_full_prefix=False,
        )
        current_packs = {
            "full_2560": _artifact_summary(FULL_PACK_DIR, full_meta),
            "cls_1280": _artifact_summary(CLS_PACK_DIR, cls_meta),
        }
        if prior.get("packs") != current_packs:
            raise RuntimeError("Packed artifacts do not match the completion receipt")
        _write_state("complete", resumed_from_receipt=True)
        return prior

    queue, output_ids = _wait_for_queue(args.poll_seconds)
    _write_state("validating_source", expected_slides=EXPECTED_SLIDES)
    source_hash, patch_counts = _validate_source(output_ids)

    # A second gate protects against an unexpected external queue/source edit
    # between the completion observation and the expensive pack publication.
    time.sleep(10)
    stable_queue, stable_ids = _queue_rows()
    if not _queue_is_complete(stable_queue) or stable_ids != output_ids:
        raise RuntimeError("Virchow2 queue changed during the stability window")
    stable_queue_snapshot = _queue_snapshot_sha256()
    if feature_inventory_sha256(FEATURE_DIR) != source_hash:
        raise RuntimeError("Virchow2 H5 inventory changed during the stability window")

    _cleanup_stale_staging(CLS_PACK_DIR)
    _cleanup_stale_staging(FULL_PACK_DIR)

    cls_meta = _build_cls_pack(source_hash, patch_counts)
    if feature_inventory_sha256(FEATURE_DIR) != source_hash:
        raise RuntimeError("Source H5 inventory changed after CLS packing")
    full_meta = _build_or_validate_full(source_hash, patch_counts)
    if feature_inventory_sha256(FEATURE_DIR) != source_hash:
        raise RuntimeError("Source H5 inventory changed after full packing")

    # This is the expensive exhaustive relationship check. It runs once before
    # publication; later idempotent receipt checks rely on the bound file hashes.
    _validate_cls_relationship(
        full_meta,
        source_hash,
        patch_counts,
        verify_full_prefix=True,
    )

    pack_summaries = {
        "full_2560": _artifact_summary(FULL_PACK_DIR, full_meta),
        "cls_1280": _artifact_summary(CLS_PACK_DIR, cls_meta),
    }
    if _queue_snapshot_sha256() != stable_queue_snapshot:
        raise RuntimeError("Virchow2 queue changed while packs were being built or hashed")
    if feature_inventory_sha256(FEATURE_DIR) != source_hash:
        raise RuntimeError("Source H5 inventory changed while packs were being hashed")

    completion = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "queue": queue,
        "queue_snapshot_sha256": stable_queue_snapshot,
        "source_dir": str(FEATURE_DIR),
        "source_inventory_sha256": source_hash,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "script_sha256": _sha256(Path(__file__)),
        "packs": pack_summaries,
    }
    _atomic_json(COMPLETION_PATH, completion)
    _write_state("complete", completion_path=str(COMPLETION_PATH))
    return completion


def status() -> dict[str, Any]:
    counts, _ = _queue_rows()
    state: dict[str, Any]
    try:
        state = json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        state = {"stage": "not_started"}
    return {
        "queue": counts,
        "state": state,
        "completion_receipt": COMPLETION_PATH.is_file(),
        "full_pack_exists": FULL_PACK_DIR.exists(),
        "cls_pack_exists": CLS_PACK_DIR.exists(),
        "full_pack_path": str(FULL_PACK_DIR),
        "cls_pack_path": str(CLS_PACK_DIR),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--poll-seconds", type=float, default=60.0)
    run_parser.add_argument("--expected-slides", type=int, default=EXPECTED_SLIDES)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--expected-slides", type=int, default=EXPECTED_SLIDES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The gate compares against one cohort size in six places; bind it once.
    global EXPECTED_SLIDES
    EXPECTED_SLIDES = args.expected_slides
    if args.command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
        return
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = RUN_DIR / "packing.lock"
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Virchow2 packing process holds the lock") from exc
        try:
            result = run(args)
        except BaseException as exc:
            _write_state("failed", error=f"{type(exc).__name__}: {exc}")
            raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
