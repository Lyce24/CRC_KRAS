#!/usr/bin/env python3
"""Validate and migrate production colon receipts to remount-stable source keys.

This is an intentionally strict one-time migration.  It recognizes only the
known production implementation, recomputes every legacy key, verifies the
current source snapshot and all committed artifacts, and writes the completion
receipt last.  Without ``--apply`` it performs a read-only dry run.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from oceanpath.extraction.stream_encoder import (
    SlideEncoder,
    SlideEncoderConfig,
    SourceSnapshot,
    _atomic_json,
    _fingerprint,
    _load_json,
)
from oceanpath.workflows.streaming import DEFAULT_CONCH, DEFAULT_HEST, DEFAULT_UNI

EXPECTED_LEGACY_IMPLEMENTATION = "a9063a40c31940dc4fa511158ef0f5bc1f2cdbf30a53ce22003b41dd47eb4899"
STAGES = (
    "seg",
    "uni_v1_coords",
    "uni_v1_feat",
    "conch_v15_coords",
    "conch_v15_feat",
)


@dataclass(frozen=True, slots=True)
class _ReceiptSlide:
    wsi: str
    path: Path
    output_id: str
    cohort: str
    mpp: float


@dataclass(frozen=True, slots=True)
class _Candidate:
    snapshot: SourceSnapshot
    completion: dict[str, Any]
    stage_receipts: dict[str, dict[str, Any]]
    new_keys: dict[str, str]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("/mnt/wsl/oceanpath-hot/features/colon_stream"),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "colon_stream",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/mnt/wsl/oceanpath-hot/scratch/colon_stream"),
    )
    parser.add_argument("--hest-checkpoint", type=Path, default=DEFAULT_HEST)
    parser.add_argument("--uni-checkpoint", type=Path, default=DEFAULT_UNI)
    parser.add_argument("--conch-v15-checkpoint", type=Path, default=DEFAULT_CONCH)
    parser.add_argument("--apply", action="store_true")
    return parser


def _legacy_stage_keys(
    encoder: SlideEncoder,
    *,
    source_key: str,
    implementation_hash: str,
) -> dict[str, str]:
    hashes = encoder.checkpoint_hashes
    seg = _fingerprint(
        {
            "source": source_key,
            "implementation": implementation_hash,
            "checkpoint": hashes["hest"],
            "policy": encoder.segmentation_policy,
        }
    )
    keys = {"seg": seg}
    for spec in encoder.cfg.encoders:
        coords = _fingerprint(
            {
                "seg": seg,
                "mpp": None,
                "target_mpp": encoder.cfg.target_mpp,
                "target_mag": encoder.cfg.target_mag,
                "patch_size": spec.patch_size,
                "overlap": encoder.cfg.overlap,
                "min_tissue_proportion": encoder.cfg.min_tissue_proportion,
            }
        )
        keys[f"{spec.name}_coords"] = coords
        keys[f"{spec.name}_feat"] = _fingerprint(
            {
                "coords": coords,
                "encoder": spec.name,
                "feature_dim": spec.feature_dim,
                "checkpoint": hashes[spec.name],
            }
        )
    return keys


def _require_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def _validate_candidate(
    encoder: SlideEncoder,
    completion_path: Path,
) -> _Candidate:
    completion = _load_json(completion_path)
    if not completion:
        raise ValueError("completion receipt is unreadable")
    _require_equal(
        completion.get("implementation_hash"),
        EXPECTED_LEGACY_IMPLEMENTATION,
        "not the expected production implementation",
    )
    _require_equal(
        completion.get("checkpoint_hashes"),
        encoder.checkpoint_hashes,
        "checkpoint hashes changed",
    )
    _require_equal(
        completion.get("segmentation_policy"),
        encoder.segmentation_policy,
        "segmentation policy changed",
    )

    source = completion.get("source")
    if not isinstance(source, dict):
        raise ValueError("missing legacy source snapshot")
    legacy_source_key = _fingerprint(source)
    _require_equal(
        completion.get("source_key"), legacy_source_key, "legacy source key is inconsistent"
    )
    mpp = float(source["mpp"])
    if not math.isfinite(mpp) or mpp <= 0:
        raise ValueError("invalid source MPP")
    slide = _ReceiptSlide(
        wsi=str(source["wsi"]),
        path=Path(str(source["path"])),
        output_id=str(source["output_id"]),
        cohort=str(source["cohort"]),
        mpp=mpp,
    )
    snapshot = SourceSnapshot.capture(slide)
    for field in ("wsi", "path", "output_id", "cohort", "mpp", "size", "mtime_ns"):
        observed = getattr(snapshot, field)
        expected = Path(str(source[field])) if field == "path" else source[field]
        _require_equal(observed, expected, f"current source {field} changed")

    legacy_keys = _legacy_stage_keys(
        encoder,
        source_key=legacy_source_key,
        implementation_hash=EXPECTED_LEGACY_IMPLEMENTATION,
    )
    # MPP is a source property in the coordinate lineage.
    for spec in encoder.cfg.encoders:
        coords_stage = f"{spec.name}_coords"
        coords = _fingerprint(
            {
                "seg": legacy_keys["seg"],
                "mpp": snapshot.mpp,
                "target_mpp": encoder.cfg.target_mpp,
                "target_mag": encoder.cfg.target_mag,
                "patch_size": spec.patch_size,
                "overlap": encoder.cfg.overlap,
                "min_tissue_proportion": encoder.cfg.min_tissue_proportion,
            }
        )
        legacy_keys[coords_stage] = coords
        legacy_keys[f"{spec.name}_feat"] = _fingerprint(
            {
                "coords": coords,
                "encoder": spec.name,
                "feature_dim": spec.feature_dim,
                "checkpoint": encoder.checkpoint_hashes[spec.name],
            }
        )
    _require_equal(completion.get("stage_keys"), legacy_keys, "legacy stage keys disagree")
    _require_equal(
        completion.get("config_key"),
        _fingerprint(legacy_keys),
        "legacy config key disagrees",
    )

    stage_receipts: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        payload = _load_json(encoder._receipt_path(snapshot.output_id, stage))
        if not payload:
            raise ValueError(f"missing {stage} receipt")
        _require_equal(payload.get("stage"), stage, f"wrong {stage} receipt name")
        _require_equal(payload.get("source_key"), legacy_source_key, f"wrong {stage} source key")
        _require_equal(payload.get("stage_key"), legacy_keys[stage], f"wrong {stage} stage key")
        _require_equal(
            payload.get("implementation_hash"),
            EXPECTED_LEGACY_IMPLEMENTATION,
            f"wrong {stage} implementation",
        )
        _require_equal(
            payload.get("checkpoint_hashes"),
            encoder.checkpoint_hashes,
            f"wrong {stage} checkpoint hashes",
        )
        stage_receipts[stage] = payload

    encoder._validate_segmentation(snapshot.output_id)
    for spec in encoder.cfg.encoders:
        encoder._validate_coordinates(snapshot, spec)
        encoder._validate_features(snapshot, spec)
    snapshot.assert_unchanged()
    return _Candidate(snapshot, completion, stage_receipts, encoder.stage_keys(snapshot))


def _migrate(encoder: SlideEncoder, candidate: _Candidate) -> None:
    snapshot = candidate.snapshot
    source = asdict(snapshot)
    for stage in STAGES:
        payload = dict(candidate.stage_receipts[stage])
        payload.update(
            {
                "schema_version": 2,
                "source": source,
                "source_key": snapshot.key,
                "stage_key": candidate.new_keys[stage],
                "implementation_hash": encoder.implementation_hash,
                "migrated_from": {
                    "implementation_hash": EXPECTED_LEGACY_IMPLEMENTATION,
                    "source_key": candidate.completion["source_key"],
                },
            }
        )
        _atomic_json(encoder._receipt_path(snapshot.output_id, stage), payload)

    completion = dict(candidate.completion)
    completion.update(
        {
            "schema_version": 2,
            "source": source,
            "source_key": snapshot.key,
            "config_key": encoder.config_key(snapshot),
            "stage_keys": candidate.new_keys,
            "implementation_hash": encoder.implementation_hash,
            "migrated_from": {
                "implementation_hash": EXPECTED_LEGACY_IMPLEMENTATION,
                "source_key": candidate.completion["source_key"],
            },
        }
    )
    _atomic_json(encoder._completion_path(snapshot.output_id), completion)


def main() -> int:
    args = _parser().parse_args()
    encoder = SlideEncoder(
        SlideEncoderConfig(
            output_root=args.feature_root.expanduser().resolve(strict=True),
            state_root=args.state_root.expanduser().resolve(strict=True),
            scratch_root=args.scratch_root.expanduser().resolve(strict=True),
            hest_checkpoint_path=args.hest_checkpoint.expanduser().resolve(strict=True),
            uni_checkpoint_path=args.uni_checkpoint.expanduser().resolve(strict=True),
            conch_v15_checkpoint_path=args.conch_v15_checkpoint.expanduser().resolve(strict=True),
            segmentation_batch_size=64,
            uni_batch_size=256,
            conch_v15_batch_size=128,
            max_workers=10,
            remove_holes=True,
        )
    )
    completion_paths = sorted(
        (encoder.cfg.output_root / "_stream_receipts").glob("*/complete.json")
    )
    candidates: list[_Candidate] = []
    ignored = 0
    errors: list[str] = []
    for index, path in enumerate(completion_paths, start=1):
        receipt = _load_json(path)
        if not receipt or receipt.get("implementation_hash") != EXPECTED_LEGACY_IMPLEMENTATION:
            ignored += 1
            continue
        try:
            candidates.append(_validate_candidate(encoder, path))
        except Exception as exc:
            errors.append(f"{path.parent.name}: {type(exc).__name__}: {exc}")
        if index % 25 == 0:
            print(f"validated={len(candidates)} ignored={ignored} errors={len(errors)}", flush=True)

    print(
        f"production={len(candidates)} ignored_legacy={ignored} errors={len(errors)} "
        f"mode={'apply' if args.apply else 'dry-run'}",
        flush=True,
    )
    if errors:
        for error in errors:
            print(error)
        return 1
    if not args.apply:
        return 0

    for candidate in candidates:
        _migrate(encoder, candidate)
    failed = [
        item.snapshot.output_id
        for item in candidates
        if not encoder.validate_complete(
            _ReceiptSlide(
                wsi=item.snapshot.wsi,
                path=item.snapshot.path,
                output_id=item.snapshot.output_id,
                cohort=item.snapshot.cohort,
                mpp=item.snapshot.mpp,
            ),
            deep=False,
        )
    ]
    if failed:
        print(f"post-migration validation failed: {', '.join(failed)}")
        return 1
    print(f"migrated={len(candidates)} post_validation=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
