#!/usr/bin/env python3
"""E2-CPHT existing-model transfer to post-Orion H&E (Aim 2, final-v8).

This runner performs the two frozen-model Orion analyses that can be executed
before the new confirmatory all-conventional refits are trained:

* the rank-only ensemble of the 15 selected Aim-1 outer-fold checkpoints;
* the four available family-LOCO three-seed ensembles (CPTAC, RIH, SurGen,
  and TCGA held out), each reported separately.

These are sensitivities, not the confirmatory CPHT estimator.  Final-v8 makes
the confirmatory estimator a new three-seed, 6,060-step all-conventional
refit.  The historical Aim-1 ``final/refit`` checkpoints do not satisfy that
contract and are deliberately not resolved anywhere in this file.  The four
sibling-stratum LOCO ensembles are also still pending, so this runner calls
the available source-composition result a four-family partial matrix, never a
complete eight-model matrix.

The command boundary enforces pre-outcome sealing:

1. ``manifest`` writes a label-blind 41-slide inference manifest and a
   checkpoint/configuration contract.
2. ``score`` performs full-bag inference from the validated packed UNIv1
   store and seals native logits without reading or writing KRAS outcomes.
3. ``report`` refuses to run without the complete inference seal; only then
   does it join the frozen v5 labels, average C33's two slide logits, and run
   the pre-registered patient analysis.

Usage::

    python aim2_cross_protocol_transfer.py preflight
    python aim2_cross_protocol_transfer.py manifest
    python aim2_cross_protocol_transfer.py score
    python aim2_cross_protocol_transfer.py report
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import lineage  # noqa: E402
from oceanpath.datasets.packed import (  # noqa: E402
    PackedFeatureStore,
    feature_inventory_sha256,
    validate_packed_dir,
)
from oceanpath.eval.core import compute_calibration_intercept_slope  # noqa: E402
from oceanpath.eval.external import sigmoid  # noqa: E402

LABEL_SOURCE = Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v5.csv")
FEATURE_DIR = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/"
    "20x_256px_0px_overlap_mpp0.5/features_uni_v1"
)
PACK_DIR = FEATURE_DIR.parent / "packed_uni_v1"
AIM1_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/train/1a_pb_cap8192/univ1"
)
LOCO_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_cap8192_v4_20260819/e2a"
)
DEFAULT_OUTPUT_ROOT = (
    REPO / "reports/reruns/final_v8_additions_20260822/e2cpht"
)

SEEDS: tuple[int, ...] = (42, 43, 44)
FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
FAMILY_LOCO_TARGETS: tuple[str, ...] = ("CPTAC", "RIH", "SurGen", "TCGA")
PENDING_SIBLING_TARGETS: tuple[str, ...] = ("SR386", "SR1482", "TCGA-COAD", "TCGA-READ")

EXPECTED_SLIDES = 41
EXPECTED_PATIENTS = 40
EXPECTED_MUTANT_PATIENTS = 15
EXPECTED_WILD_TYPE_PATIENTS = 25
EXPECTED_PACK_SLIDES = 2_128
EXPECTED_FEATURE_DIM = 1_024
EXPECTED_TREATMENT_PATIENTS = 6
EXPECTED_AMBIGUOUS_PATIENT = "ORION:C15"

CAP = 8_192
LOCO_STEP_BUDGET = 6_060
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_817
INFERENCE_DEVICE = "cuda"
INFERENCE_AUTOCAST_DTYPE = "bfloat16"
INFERENCE_NUM_WORKERS = 4

FORBIDDEN_PREOUTCOME_COLUMNS = frozenset(
    {
        "target_label",
        "label",
        "kras",
        "kras_subvariant",
        "ras",
        "nras",
        "braf",
        "braf_subvariant",
        "msi_dmmr",
    }
)


class ContractError(RuntimeError):
    """A frozen input, model, or output violates the final-v8 contract."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"


def _publish_text(path: Path, text: str) -> None:
    """Publish once; an exact existing artifact is a safe resumable no-op."""

    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to replace non-identical artifact: {path}")
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _publish_json(path: Path, value: Any) -> None:
    _publish_text(path, _canonical_json(value))


def _publish_csv(path: Path, frame: pd.DataFrame) -> None:
    _publish_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _publish_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Atomically publish parquet; validate an existing cache by its receipt."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_parquet(temporary, index=False)
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    return _sha256_bytes(frame.to_csv(index=False, lineterminator="\n").encode())


def _artifact(path: Path) -> dict[str, Any]:
    return lineage.artifact_identity(path)


def _assert_exact_keys(record: dict[str, Any], expected: dict[str, Any], context: Path) -> None:
    mismatch = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatch:
        raise ContractError(f"Contract mismatch in {context}: {mismatch}")


def _nested_value(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ContractError(f"Missing governed configuration field {dotted!r}")
        value = value[key]
    return value


def _assert_recipe(
    record: dict[str, Any], expected: dict[str, Any], context: Path
) -> None:
    mismatched = {
        dotted: {"expected": value, "observed": _nested_value(record, dotted)}
        for dotted, value in expected.items()
        if _nested_value(record, dotted) != value
    }
    if mismatched:
        raise ContractError(f"Locked model recipe mismatch in {context}: {mismatched}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object: {path}")
    return value


def _orion_rows(master: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cohort",
        "subcohort",
        "specimen_role",
        "output_id",
        "patient_uid",
        "include",
        "available",
        "used_kras",
        "qc_slides",
        "qc_flags",
        "mpp",
        "mpp_source",
    }
    missing = sorted(required - set(master.columns))
    if missing:
        raise ContractError(f"v5 label master lacks columns: {missing}")
    rows = master.loc[
        master["cohort"].eq("Orion")
        & master["include"].astype(str).str.lower().eq("yes")
        & master["available"].astype(str).str.lower().eq("yes")
        & master["used_kras"].astype(str).str.lower().eq("yes")
        & master["qc_slides"].astype(str).str.lower().eq("pass")
    ].copy()
    if len(rows) != EXPECTED_SLIDES:
        raise ContractError(f"Orion census is {len(rows)} slides, expected {EXPECTED_SLIDES}")
    if rows["patient_uid"].nunique() != EXPECTED_PATIENTS:
        raise ContractError(
            f"Orion census is {rows['patient_uid'].nunique()} patients, "
            f"expected {EXPECTED_PATIENTS}"
        )
    if rows["output_id"].duplicated().any():
        raise ContractError("Orion output_id values are not unique")
    if set(rows["subcohort"].dropna().astype(str)) != {"Orion-CRC"}:
        raise ContractError("Unexpected Orion subcohort")
    if set(rows["specimen_role"].dropna().astype(str)) != {"primary"}:
        raise ContractError("Orion manifest contains a non-primary specimen")
    mpp = pd.to_numeric(rows["mpp"], errors="raise").to_numpy(dtype=float)
    if not np.allclose(mpp, 0.325, rtol=0.0, atol=1e-12):
        raise ContractError("Orion MPP is not uniformly 0.325")
    if set(rows["mpp_source"].dropna().astype(str)) != {"ome_xml"}:
        raise ContractError("Orion MPP source is not uniformly ome_xml")
    return rows.sort_values("output_id", kind="stable").reset_index(drop=True)


def build_label_blind_manifest(master: pd.DataFrame, pack_index: pd.DataFrame) -> pd.DataFrame:
    """Derive the governed Orion inference rows without touching molecular columns."""

    rows = _orion_rows(master)
    required_index = {"slide_id", "n_patches"}
    missing_index = sorted(required_index - set(pack_index.columns))
    if missing_index:
        raise ContractError(f"Packed index lacks columns: {missing_index}")
    patch_counts = dict(
        zip(
            pack_index["slide_id"].astype(str),
            pd.to_numeric(pack_index["n_patches"], errors="raise").astype(int),
            strict=True,
        )
    )
    missing_slides = sorted(set(rows["output_id"].astype(str)) - set(patch_counts))
    if missing_slides:
        raise ContractError(f"Packed store lacks Orion slides: {missing_slides[:5]}")

    manifest = pd.DataFrame(
        {
            "slide_id": rows["output_id"].astype(str),
            "patient_id": rows["patient_uid"].astype(str),
            "cohort": rows["cohort"].astype(str),
            "subcohort": rows["subcohort"].astype(str),
            "specimen_role": rows["specimen_role"].astype(str),
            "mpp": pd.to_numeric(rows["mpp"], errors="raise"),
            "mpp_source": rows["mpp_source"].astype(str),
            "patch_count": rows["output_id"].astype(str).map(patch_counts).astype(int),
            "exclude_neoadjuvant": rows["qc_flags"].fillna("").astype(str).eq("treatment"),
            "exclude_ambiguous_crc15": rows["patient_uid"].astype(str).eq(
                EXPECTED_AMBIGUOUS_PATIENT
            ),
        }
    )
    overlap = FORBIDDEN_PREOUTCOME_COLUMNS & set(manifest.columns)
    if overlap:
        raise ContractError(f"Label-blind manifest leaks outcome columns: {sorted(overlap)}")
    treatment_n = manifest.loc[manifest["exclude_neoadjuvant"], "patient_id"].nunique()
    if treatment_n != EXPECTED_TREATMENT_PATIENTS:
        raise ContractError(
            f"Found {treatment_n} neoadjuvant patients, expected {EXPECTED_TREATMENT_PATIENTS}"
        )
    ambiguous = manifest.loc[manifest["exclude_ambiguous_crc15"], "patient_id"].unique()
    if list(ambiguous) != [EXPECTED_AMBIGUOUS_PATIENT]:
        raise ContractError(f"Ambiguous-specimen sensitivity mismatch: {ambiguous.tolist()}")
    c33 = manifest[manifest["patient_id"].eq("ORION:C33")]
    if len(c33) != 2 or set(c33["slide_id"]) != {"CRC33_01", "CRC33_02"}:
        raise ContractError("C33 must contain exactly CRC33_01 and CRC33_02")
    return manifest


def _validate_pack(pack_dir: Path, feature_dir: Path) -> dict[str, Any]:
    live_inventory = feature_inventory_sha256(feature_dir)
    meta = validate_packed_dir(pack_dir, verify_source=live_inventory)
    if meta.n_slides != EXPECTED_PACK_SLIDES or meta.feat_dim != EXPECTED_FEATURE_DIM:
        raise ContractError(
            f"Packed UNIv1 shape is {meta.n_slides} x {meta.feat_dim}; expected "
            f"{EXPECTED_PACK_SLIDES} x {EXPECTED_FEATURE_DIM}"
        )
    return {
        "path": str(pack_dir.resolve()),
        "meta": _artifact(pack_dir / "meta.json"),
        "index": _artifact(pack_dir / "index.parquet"),
        # The inventory hash binds H5 names/sizes/mtimes, while these payload
        # hashes bind the exact packed bytes consumed by inference. Structural
        # size checks alone cannot detect same-size corruption.
        "features": _artifact(pack_dir / "features.bin"),
        "coords": _artifact(pack_dir / "coords.bin"),
        "n_slides": int(meta.n_slides),
        "total_patches": int(meta.total_patches),
        "feature_dim": int(meta.feat_dim),
        "feature_dtype": meta.feat_dtype,
        "source_dir": str(feature_dir.resolve()),
        "source_inventory_sha256": live_inventory,
    }


def inference_environment() -> dict[str, Any]:
    """Exact governed CUDA/BF16 inference implementation and runtime identity."""

    import torch

    if not torch.cuda.is_available():
        raise ContractError("Governed E2-CPHT inference requires CUDA; no CUDA device is available")
    if not torch.cuda.is_bf16_supported():
        raise ContractError("Governed E2-CPHT inference requires CUDA bfloat16 autocast")
    sources = (
        Path(__file__).resolve(),
        REPO / "src/oceanpath/training/lightning.py",
        REPO / "src/oceanpath/models/__init__.py",
        REPO / "src/oceanpath/models/base.py",
        REPO / "src/oceanpath/models/components.py",
        REPO / "src/oceanpath/models/abmil.py",
        REPO / "src/oceanpath/models/wsi_classifier.py",
        REPO / "src/oceanpath/contracts/__init__.py",
        REPO / "src/oceanpath/contracts/slide_ids.py",
        REPO / "src/oceanpath/datasets/datamodule.py",
        REPO / "src/oceanpath/datasets/packed.py",
    )
    return {
        "device": INFERENCE_DEVICE,
        "device_index": 0,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "autocast": True,
        "autocast_dtype": INFERENCE_AUTOCAST_DTYPE,
        "force_float32_before_device_transfer": True,
        "batch_size": 1,
        "num_workers": INFERENCE_NUM_WORKERS,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "lightning_version": importlib.metadata.version("lightning"),
        "python_version": platform.python_version(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "implementation_sources": [_artifact(path) for path in sources],
    }


def analysis_environment() -> dict[str, Any]:
    """Exact implementation and numerical runtime used after the outcome join."""

    import scipy
    import sklearn

    sources = (
        Path(__file__).resolve(),
        REPO / "src/oceanpath/eval/core.py",
        REPO / "src/oceanpath/eval/external.py",
    )
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "implementation_sources": [_artifact(path) for path in sources],
    }


def _resolve_aim1_checkpoints(root: Path = AIM1_ROOT) -> dict[int, list[dict[str, Any]]]:
    resolved: dict[int, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        seed_root = root / f"seed{seed}"
        identity = _read_json(seed_root / "training_identity.json")
        fingerprint = str(identity.get("fingerprint", ""))
        _assert_recipe(
            identity,
            {
                "schema_version": 2,
                "payload.schema_version": 2,
                "payload.material_config.data.aim1_model": "1a",
                "payload.material_config.data.csv_path": (
                    "/mnt/d/YC.Liu/manifests/colon/aim1_dev.csv"
                ),
                "payload.material_config.data.num_classes": 2,
                "payload.material_config.encoder.name": "uni_v1",
                "payload.material_config.encoder.feature_dim": EXPECTED_FEATURE_DIM,
                "payload.material_config.model.arch": "abmil",
                "payload.material_config.model.embed_dim": 512,
                "payload.material_config.model.attn_dim": 384,
                "payload.material_config.model.gate": True,
                "payload.material_config.model.dropout": 0.25,
                "payload.material_config.model.input_dropout": 0.1,
                "payload.material_config.training.loss_type": "bce",
                "payload.material_config.training.training_class_weighted_loss": False,
                "payload.material_config.training.sample_weight_column": None,
                "payload.material_config.training.train_sampling_strategy": "patient_natural",
                "payload.material_config.training.dataset_max_instances": CAP,
                "payload.material_config.training.eval_full_bags": True,
                "payload.material_config.training.force_float32": True,
                "payload.material_config.training.seed": seed,
                "payload.material_config.splits.n_folds": len(FOLDS),
                "payload.material_config.splits.group_column": "patient_id",
                "payload.material_config.platform.precision": "bf16-mixed",
            },
            seed_root / "training_identity.json",
        )
        completion = _read_json(seed_root / "training_completion.json")
        _assert_exact_keys(
            completion,
            {"status": "completed", "n_folds": len(FOLDS), "training_fingerprint": fingerprint},
            seed_root / "training_completion.json",
        )
        completion_entries = completion.get("fold_completions")
        if not isinstance(completion_entries, list) or len(completion_entries) != len(FOLDS):
            raise ContractError(f"Incomplete fold receipt roster for Aim1 seed {seed}")
        receipt_by_path = {str(entry.get("path")): entry for entry in completion_entries}

        models: list[dict[str, Any]] = []
        for fold in FOLDS:
            fold_root = seed_root / f"fold_{fold}"
            receipt_path = fold_root / "completion.json"
            relative_receipt = str(receipt_path.relative_to(seed_root))
            aggregate_entry = receipt_by_path.get(relative_receipt)
            if aggregate_entry is None or aggregate_entry.get("sha256") != lineage.sha256_file(
                receipt_path
            ):
                raise ContractError(f"Aim1 aggregate receipt does not bind {receipt_path}")
            receipt = _read_json(receipt_path)
            _assert_exact_keys(
                receipt,
                {"status": "completed", "fold": fold, "training_fingerprint": fingerprint},
                receipt_path,
            )
            metrics_path = fold_root / "fold_metrics.json"
            metrics = _read_json(metrics_path)
            checkpoint = Path(str(metrics.get("best_checkpoint", ""))).resolve()
            recorded = (receipt.get("artifacts") or {}).get("best_checkpoint") or {}
            recorded_path = (fold_root / str(recorded.get("path", ""))).resolve()
            if checkpoint != recorded_path:
                raise ContractError(
                    f"Aim1 fold metrics and completion select different checkpoints: {fold_root}"
                )
            observed = _artifact(checkpoint)
            if observed.get("sha256") != recorded.get("sha256") or observed.get(
                "size_bytes"
            ) != recorded.get("size_bytes"):
                raise ContractError(f"Aim1 selected checkpoint hash mismatch: {checkpoint}")
            recorded_metrics = (receipt.get("artifacts") or {}).get("metrics") or {}
            if _artifact(metrics_path) != {
                "path": str(metrics_path.resolve()),
                "size_bytes": recorded_metrics.get("size_bytes"),
                "sha256": recorded_metrics.get("sha256"),
            }:
                raise ContractError(f"Aim1 fold metrics receipt mismatch: {metrics_path}")
            models.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "checkpoint": observed,
                    "fold_completion": _artifact(receipt_path),
                    "fold_metrics": _artifact(metrics_path),
                    "training_identity": _artifact(seed_root / "training_identity.json"),
                    "training_completion": _artifact(seed_root / "training_completion.json"),
                    "resolved_config": _artifact(seed_root / "config.yaml"),
                    "training_fingerprint": fingerprint,
                }
            )
        resolved[seed] = models
    return resolved


def _resolve_loco_models(root: Path = LOCO_ROOT) -> dict[str, dict[str, dict[str, Any]]]:
    resolved: dict[str, dict[str, dict[str, Any]]] = {}
    expected_source_n = {"CPTAC": 1_392, "RIH": 1_333, "SurGen": 749, "TCGA": 984}
    for target in FAMILY_LOCO_TARGETS:
        # Keep seed keys as strings because each target mapping also carries
        # the named ``calibrator`` entry. Mixed int/string keys cannot be
        # canonicalized with json.dumps(sort_keys=True).
        per_seed: dict[str, dict[str, Any]] = {}
        for seed in SEEDS:
            run = root / "train/pb_cap8192" / target.lower() / f"seed{seed}"
            summary_path = run / "fit_summary.json"
            summary = _read_json(summary_path)
            _assert_exact_keys(
                summary,
                {
                    "status": "completed",
                    "lineage": "aim2_cap8192_v4_20260819",
                    "target": target,
                    "seed": seed,
                    "sampling_seed": seed,
                    "cap": CAP,
                    "optimizer_step_budget": LOCO_STEP_BUDGET,
                    "sampler": "patient_natural",
                    "loss_weighting": "none",
                },
                summary_path,
            )
            result = summary.get("result") or {}
            _assert_exact_keys(
                result,
                {
                    "actual_optimizer_steps": LOCO_STEP_BUDGET,
                    "refit_max_steps": LOCO_STEP_BUDGET,
                    "seed": seed,
                    "sampling_seed": seed,
                    "train_sampling_strategy": "patient_natural",
                    "dataset_max_instances": CAP,
                    "eval_full_bags": True,
                },
                summary_path,
            )
            checkpoint = run / "final/refit/model.ckpt"
            config = run / "resolved_config.yaml"
            request_path = run / "run_request.json"
            request = _read_json(request_path)
            _assert_exact_keys(
                request,
                {
                    "schema_version": 2,
                    "status": "requested",
                    "lineage": "aim2_cap8192_v4_20260819",
                    "target": target,
                    "seed": seed,
                    "sampling_seed": seed,
                    "cap": CAP,
                    "optimizer_step_budget": LOCO_STEP_BUDGET,
                },
                request_path,
            )
            request_inputs = request.get("inputs") or {}
            source_manifest = (request_inputs.get("source_manifest") or {}).get("path")
            if not source_manifest or request_inputs["source_manifest"] != _artifact(
                Path(source_manifest)
            ):
                raise ContractError(f"LOCO source manifest identity mismatch: {request_path}")
            split_path = (request_inputs.get("splits") or {}).get("path")
            if not split_path or request_inputs["splits"] != _artifact(Path(split_path)):
                raise ContractError(f"LOCO split identity mismatch: {request_path}")
            for source in request.get("source_snapshot") or []:
                frozen = run / "source_snapshot" / str(source["relative_path"])
                observed_source = _artifact(frozen)
                if observed_source["sha256"] != source.get("sha256") or observed_source[
                    "size_bytes"
                ] != source.get("size_bytes"):
                    raise ContractError(f"LOCO source snapshot mismatch: {frozen}")

            config_value = yaml.safe_load(config.read_text(encoding="utf-8"))
            if not isinstance(config_value, dict):
                raise ContractError(f"Malformed LOCO resolved configuration: {config}")
            _assert_recipe(
                config_value,
                {
                    "data.csv_path": str(Path(source_manifest).resolve()),
                    "data.num_classes": 2,
                    "encoder.name": "uni_v1",
                    "encoder.feature_dim": EXPECTED_FEATURE_DIM,
                    "model.arch": "abmil",
                    "model.embed_dim": 512,
                    "model.attn_dim": 384,
                    "model.gate": True,
                    "model.dropout": 0.25,
                    "model.input_dropout": 0.1,
                    "training.loss_type": "bce",
                    "training.training_class_weighted_loss": False,
                    "training.class_weights": None,
                    "training.sample_weight_column": None,
                    "training.train_sampling_strategy": "patient_natural",
                    "training.dataset_max_instances": CAP,
                    "training.eval_full_bags": True,
                    "training.force_float32": True,
                    "training.refit_max_steps": LOCO_STEP_BUDGET,
                    "training.seed": seed,
                    "platform.precision": "bf16-mixed",
                },
                config,
            )
            checkpoint_identity = _artifact(checkpoint)
            config_identity = _artifact(config)
            if checkpoint_identity != summary.get("model"):
                raise ContractError(f"LOCO checkpoint identity mismatch: {checkpoint}")
            if config_identity != summary.get("resolved_config"):
                raise ContractError(f"LOCO config identity mismatch: {config}")
            per_seed[str(seed)] = {
                "seed": seed,
                "checkpoint": checkpoint_identity,
                "resolved_config": config_identity,
                "fit_summary": _artifact(summary_path),
                "run_request": _artifact(request_path),
                "source_manifest": request_inputs["source_manifest"],
                "splits": request_inputs["splits"],
                "optimizer_steps": LOCO_STEP_BUDGET,
            }
        calibrator_path = root / "calibrators" / f"cap8192_{target.lower()}.json"
        calibrator = _read_json(calibrator_path)
        if not all(np.isfinite(float(calibrator[key])) for key in ("a", "b")):
            raise ContractError(f"Non-finite source calibrator: {calibrator_path}")
        if int(calibrator.get("n_source", -1)) != expected_source_n[target]:
            raise ContractError(f"Unexpected source calibrator census: {calibrator_path}")
        source_cv_inputs = calibrator.get("source_cv_inputs") or {}
        if set(source_cv_inputs) != {str(seed) for seed in SEEDS}:
            raise ContractError(f"Incomplete source-CV calibrator roster: {calibrator_path}")
        for seed in SEEDS:
            identity = source_cv_inputs[str(seed)]
            if identity != _artifact(Path(identity["path"])):
                raise ContractError(f"Source-CV calibrator input changed: {identity['path']}")
        resolved[target] = {
            **per_seed,
            "calibrator": {
                "artifact": _artifact(calibrator_path),
                "a": float(calibrator["a"]),
                "b": float(calibrator["b"]),
                "n_source": int(calibrator["n_source"]),
                "source_cv_inputs": source_cv_inputs,
            },
        }
    return resolved


def build_contract(
    manifest: pd.DataFrame,
    *,
    label_source: Path = LABEL_SOURCE,
    feature_dir: Path = FEATURE_DIR,
    pack_dir: Path = PACK_DIR,
    aim1_root: Path = AIM1_ROOT,
    loco_root: Path = LOCO_ROOT,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "E2-CPHT existing-model sensitivities",
        "analysis_role": "preliminary frozen-model sensitivities; not confirmatory CPHT",
        "preoutcome": True,
        "label_join_permitted": False,
        "orion_manifest": {
            "n_slides": int(len(manifest)),
            "n_patients": int(manifest["patient_id"].nunique()),
            "columns": list(manifest.columns),
            "csv_sha256": _frame_sha256(manifest),
            "patient_aggregation": "mean native slide logit; C33 has two equal-weight slides",
        },
        "label_source_identity_not_opened_by_score": _artifact(label_source),
        "feature_store": _validate_pack(pack_dir, feature_dir),
        "aim1_outer15": {
            "role": "rank-only sensitivity",
            "aggregation": "mean native logit over 3 seeds x 5 selected outer folds",
            "models_by_seed": _resolve_aim1_checkpoints(aim1_root),
            "probability_calibration": None,
        },
        "family_loco": {
            "role": "four-family partial source-composition matrix",
            "complete_eight_model_matrix": False,
            "available_held_out_families": list(FAMILY_LOCO_TARGETS),
            "pending_sibling_strata": list(PENDING_SIBLING_TARGETS),
            "models": _resolve_loco_models(loco_root),
            "aggregation": "mean native logit over seeds 42, 43, 44 within each held-out family",
        },
        "inference": {
            "encoder": "UNIv1",
            "feature_dim": EXPECTED_FEATURE_DIM,
            "training_cap": CAP,
            "evaluation_bag": "full",
            "force_float32": True,
            "model_selection_from_orion": False,
            "environment": inference_environment(),
        },
        "statistics": {
            "sampling_unit": "patient",
            "bootstrap": "KRAS-stratified percentile bootstrap",
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "shared_indices_across_models": True,
            "primary_population": "all 40 governed patients",
            "sensitivities": ["exclude_neoadjuvant", "exclude_ambiguous_crc15"],
            "analysis_environment": analysis_environment(),
        },
        "excluded_models": {
            "historical_aim1_final_refits": (
                "not used: historical p75 refits do not satisfy the final-v8 "
                "6,060-step confirmatory contract"
            ),
            "confirmatory_all_conventional_refits": "pending new training",
        },
    }


def _manifest_path(root: Path) -> Path:
    return root / "inputs/orion_primary.csv"


def _contract_path(root: Path) -> Path:
    return root / "inputs/preoutcome_contract.json"


def _score_path(root: Path, kind: str, seed: int, target: str | None = None) -> Path:
    if kind == "aim1":
        name = f"aim1_seed{seed}_orion.parquet"
    elif kind == "loco" and target is not None:
        name = f"loco_{target.lower()}_seed{seed}_orion.parquet"
    else:
        raise ValueError(f"Unknown scorer: kind={kind!r}, target={target!r}")
    return root / "scores" / name


def _receipt_path(score_path: Path) -> Path:
    return score_path.with_suffix(".receipt.json")


def _load_sealed_inputs(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = _manifest_path(root)
    contract_path = _contract_path(root)
    if not manifest_path.is_file() or not contract_path.is_file():
        raise ContractError("Run `manifest` before scoring")
    manifest = pd.read_csv(manifest_path)
    contract = _read_json(contract_path)
    if _frame_sha256(manifest) != contract["orion_manifest"]["csv_sha256"]:
        raise ContractError("Sealed Orion manifest does not match its contract")
    overlap = FORBIDDEN_PREOUTCOME_COLUMNS & set(manifest.columns)
    if overlap:
        raise ContractError(f"Sealed inference manifest leaks outcomes: {sorted(overlap)}")
    return manifest, contract


def _verify_sealed_feature_store(contract: dict[str, Any]) -> None:
    expected = contract["feature_store"]
    observed = _validate_pack(Path(expected["path"]), Path(expected["source_dir"]))
    if observed != expected:
        raise ContractError("Packed feature store changed after the pre-outcome contract seal")


def _checkpoint_path(identity: dict[str, Any]) -> Path:
    path = Path(str(identity["path"]))
    if _artifact(path) != identity:
        raise ContractError(f"Checkpoint changed after contract seal: {path}")
    return path


def _score_checkpoints(
    checkpoints: Sequence[tuple[int, int, Path]],
    manifest: pd.DataFrame,
    *,
    feature_dir: Path,
    pack_dir: Path,
    device: str,
    num_workers: int,
) -> pd.DataFrame:
    """Split-free full-bag packed inference retaining one native logit per model/slide."""

    import torch
    from torch.utils.data import DataLoader

    from oceanpath.datasets.datamodule import SimpleMILCollator, SlideDataset
    from oceanpath.training.lightning import MILTrainModule

    live_inventory = feature_inventory_sha256(feature_dir)
    store = PackedFeatureStore(pack_dir, verify_source=live_inventory)
    slide_ids = manifest["slide_id"].astype(str).tolist()
    dataset = SlideDataset(
        feature_dir=str(feature_dir),
        slide_ids=slide_ids,
        labels=dict.fromkeys(slide_ids, 0),
        max_instances=None,
        is_train=False,
        force_float32=True,
        store=store,
    )
    missing = sorted(set(slide_ids) - set(dataset.slide_ids))
    if missing:
        raise ContractError(f"Packed dataset lacks Orion slides: {missing[:5]}")
    loader_kwargs: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": SimpleMILCollator(max_instances=None),
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs.update({"prefetch_factor": 2, "persistent_workers": True})
    loader = DataLoader(dataset, **loader_kwargs)

    if device != INFERENCE_DEVICE:
        raise ContractError(f"Governed inference device must be {INFERENCE_DEVICE}, got {device}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ContractError("Governed inference requires CUDA with bfloat16 support")
    resolved_device = device
    rows: list[dict[str, Any]] = []
    for ordinal, (seed, fold, checkpoint) in enumerate(checkpoints, start=1):
        print(
            f"  model {ordinal}/{len(checkpoints)} seed{seed} fold{fold}: {checkpoint.name}",
            flush=True,
        )
        try:
            module = MILTrainModule.load_from_checkpoint(
                str(checkpoint), map_location=resolved_device, weights_only=False
            )
        except TypeError:
            module = MILTrainModule.load_from_checkpoint(
                str(checkpoint), map_location=resolved_device
            )
        module.eval().to(resolved_device)
        with torch.inference_mode():
            for batch in loader:
                features = batch["features"].to(resolved_device, non_blocking=True)
                mask = (
                    batch["mask"].to(resolved_device, non_blocking=True)
                    if batch.get("mask") is not None
                    else None
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    output = module.model(features, mask=mask)
                logits = output.logits.detach().float().cpu().numpy()
                for index, slide_id in enumerate(batch["slide_ids"]):
                    values = np.atleast_1d(logits[index]).ravel()
                    if values.size != 1:
                        raise ContractError(
                            f"Expected one KRAS logit, got {values.size} from {checkpoint}"
                        )
                    rows.append(
                        {
                            "slide_id": str(slide_id),
                            "seed": int(seed),
                            "fold": int(fold),
                            "logit": float(values[0]),
                        }
                    )
        del module
        if resolved_device == "cuda":
            torch.cuda.empty_cache()

    scores = pd.DataFrame(rows)
    expected = len(checkpoints) * len(manifest)
    if len(scores) != expected:
        raise ContractError(f"Inference emitted {len(scores)} rows, expected {expected}")
    if scores.duplicated(["slide_id", "seed", "fold"]).any():
        raise ContractError("Inference emitted duplicate slide/model rows")
    if not np.isfinite(scores["logit"].to_numpy()).all():
        raise ContractError("Inference emitted non-finite native logits")
    return scores.sort_values(["seed", "fold", "slide_id"], kind="stable").reset_index(
        drop=True
    )


def _score_inputs_for_seed(
    contract: dict[str, Any], kind: str, seed: int, target: str | None
) -> list[dict[str, Any]]:
    if kind == "aim1":
        entries = contract["aim1_outer15"]["models_by_seed"][str(seed)]
        return [
            {"seed": seed, "fold": int(entry["fold"]), "checkpoint": entry["checkpoint"]}
            for entry in entries
        ]
    if kind == "loco" and target is not None:
        return [
            {
                "seed": seed,
                "fold": 0,
                "checkpoint": contract["family_loco"]["models"][target][str(seed)][
                    "checkpoint"
                ],
            }
        ]
    raise ValueError(f"Unknown scorer: {kind}/{target}")


def _validate_score_frame(
    scores: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    expected_seed: int,
    expected_folds: Sequence[int],
    context: Path | str,
) -> None:
    expected_columns = ["slide_id", "seed", "fold", "logit"]
    if list(scores.columns) != expected_columns:
        raise ContractError(
            f"Score schema mismatch for {context}: {list(scores.columns)} != {expected_columns}"
        )
    if FORBIDDEN_PREOUTCOME_COLUMNS & set(scores.columns):
        raise ContractError(f"Outcome column leaked into pre-outcome scores: {context}")
    if not np.isfinite(pd.to_numeric(scores["logit"], errors="coerce").to_numpy()).all():
        raise ContractError(f"Non-finite native logits: {context}")
    if set(pd.to_numeric(scores["seed"], errors="raise").astype(int)) != {expected_seed}:
        raise ContractError(f"Wrong model seed roster: {context}")
    expected_fold_set = {int(fold) for fold in expected_folds}
    if set(pd.to_numeric(scores["fold"], errors="raise").astype(int)) != expected_fold_set:
        raise ContractError(f"Wrong model fold roster: {context}")
    if scores.duplicated(["slide_id", "seed", "fold"]).any():
        raise ContractError(f"Duplicate slide/model score rows: {context}")
    slide_ids = set(manifest["slide_id"].astype(str))
    if set(scores["slide_id"].astype(str)) != slide_ids:
        raise ContractError(f"Score slide roster differs from the sealed manifest: {context}")
    expected_rows = len(slide_ids) * len(expected_fold_set)
    if len(scores) != expected_rows:
        raise ContractError(f"Score row count is {len(scores)}, expected {expected_rows}: {context}")
    per_model = scores.groupby(["seed", "fold"])["slide_id"].agg(list)
    for model_slides in per_model:
        if set(map(str, model_slides)) != slide_ids or len(model_slides) != len(slide_ids):
            raise ContractError(f"A model does not cover every Orion slide exactly once: {context}")


def _validate_score_receipt(
    receipt: dict[str, Any],
    path: Path,
    *,
    manifest: pd.DataFrame,
    manifest_identity: dict[str, Any],
    contract_identity: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    environment: dict[str, Any],
) -> None:
    expected_rows = len(manifest) * len({int(entry["fold"]) for entry in checkpoints})
    _assert_exact_keys(
        receipt,
        {
            "schema_version": 1,
            "preoutcome": True,
            "contains_target_outcomes": False,
            "n_rows": expected_rows,
            "n_slides": int(len(manifest)),
        },
        _receipt_path(path),
    )
    expected_inputs = {
        "manifest": manifest_identity,
        "contract": contract_identity,
        "checkpoints": checkpoints,
        "inference_environment": environment,
    }
    if receipt.get("inputs") != expected_inputs:
        raise ContractError(f"Score input identity mismatch: {path}")
    if receipt.get("artifact") != _artifact(path):
        raise ContractError(f"Score artifact identity mismatch: {path}")


def _validate_cached_score(
    path: Path,
    *,
    manifest: pd.DataFrame,
    manifest_identity: dict[str, Any],
    contract_identity: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    environment: dict[str, Any],
) -> bool:
    receipt_path = _receipt_path(path)
    if not path.exists() and not receipt_path.exists():
        return False
    if not path.is_file() or not receipt_path.is_file():
        raise ContractError(f"Partial score cache: {path}")
    receipt = _read_json(receipt_path)
    _validate_score_receipt(
        receipt,
        path,
        manifest=manifest,
        manifest_identity=manifest_identity,
        contract_identity=contract_identity,
        checkpoints=checkpoints,
        environment=environment,
    )
    scores = pd.read_parquet(path)
    _validate_score_frame(
        scores,
        manifest,
        expected_seed=int(checkpoints[0]["seed"]),
        expected_folds=[int(entry["fold"]) for entry in checkpoints],
        context=path,
    )
    return True


def _score_one_seed(
    root: Path,
    manifest: pd.DataFrame,
    contract: dict[str, Any],
    *,
    kind: str,
    seed: int,
    target: str | None,
    device: str,
    num_workers: int,
) -> None:
    path = _score_path(root, kind, seed, target)
    checkpoints = _score_inputs_for_seed(contract, kind, seed, target)
    manifest_identity = _artifact(_manifest_path(root))
    contract_identity = _artifact(_contract_path(root))
    if _validate_cached_score(
        path,
        manifest=manifest,
        manifest_identity=manifest_identity,
        contract_identity=contract_identity,
        checkpoints=checkpoints,
        environment=contract["inference"]["environment"],
    ):
        print(f"  cached and verified: {path.name}")
        return
    score_checkpoints = [
        (int(entry["seed"]), int(entry["fold"]), _checkpoint_path(entry["checkpoint"]))
        for entry in checkpoints
    ]
    scores = _score_checkpoints(
        score_checkpoints,
        manifest,
        feature_dir=Path(contract["feature_store"]["source_dir"]),
        pack_dir=Path(contract["feature_store"]["path"]),
        device=device,
        num_workers=num_workers,
    )
    _validate_score_frame(
        scores,
        manifest,
        expected_seed=seed,
        expected_folds=[int(entry["fold"]) for entry in checkpoints],
        context=path,
    )
    _publish_parquet(path, scores)
    _publish_json(
        _receipt_path(path),
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preoutcome": True,
            "contains_target_outcomes": False,
            "inputs": {
                "manifest": manifest_identity,
                "contract": contract_identity,
                "checkpoints": checkpoints,
                "inference_environment": contract["inference"]["environment"],
            },
            "artifact": _artifact(path),
            "n_rows": int(len(scores)),
            "n_slides": int(scores["slide_id"].nunique()),
        },
    )
    print(f"  sealed {len(scores)} label-blind rows: {path}")


def expected_score_paths(root: Path) -> list[Path]:
    paths = [_score_path(root, "aim1", seed) for seed in SEEDS]
    paths.extend(
        _score_path(root, "loco", seed, target)
        for target in FAMILY_LOCO_TARGETS
        for seed in SEEDS
    )
    return paths


def _seal_inference(root: Path) -> dict[str, Any]:
    seal_path = root / "inference_seal.json"
    if seal_path.is_file():
        return _verify_inference_seal(root)
    if seal_path.exists() or seal_path.is_symlink():
        raise ContractError(f"Invalid inference seal path: {seal_path}")
    manifest, contract = _load_sealed_inputs(root)
    manifest_identity = _artifact(_manifest_path(root))
    contract_identity = _artifact(_contract_path(root))
    artifacts: list[dict[str, Any]] = []
    jobs = [("aim1", seed, None) for seed in SEEDS]
    jobs.extend(
        ("loco", seed, target) for target in FAMILY_LOCO_TARGETS for seed in SEEDS
    )
    for kind, seed, target in jobs:
        path = _score_path(root, kind, seed, target)
        if not path.is_file() or not _receipt_path(path).is_file():
            raise ContractError(f"Cannot seal incomplete inference; missing {path}")
        receipt = _read_json(_receipt_path(path))
        expected_checkpoints = _score_inputs_for_seed(contract, kind, seed, target)
        _validate_score_receipt(
            receipt,
            path,
            manifest=manifest,
            manifest_identity=manifest_identity,
            contract_identity=contract_identity,
            checkpoints=expected_checkpoints,
            environment=contract["inference"]["environment"],
        )
        frame = pd.read_parquet(path)
        _validate_score_frame(
            frame,
            manifest,
            expected_seed=seed,
            expected_folds=[int(entry["fold"]) for entry in expected_checkpoints],
            context=path,
        )
        artifacts.append(
            {"score": _artifact(path), "receipt": _artifact(_receipt_path(path))}
        )
    seal = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "sealed_before_outcome_join",
        "preoutcome": True,
        "manifest": manifest_identity,
        "contract": contract_identity,
        "n_slides": int(len(manifest)),
        "n_patients": int(manifest["patient_id"].nunique()),
        "score_artifacts": artifacts,
        "score_artifact_count": len(artifacts),
        "model_forward_passes_per_slide": 15 + 4 * 3,
        "target_outcomes_present": False,
        "inference_environment": contract["inference"]["environment"],
        "confirmatory_cpht_model_present": False,
        "complete_eight_model_matrix": contract["family_loco"][
            "complete_eight_model_matrix"
        ],
    }
    _publish_json(seal_path, seal)
    return _read_json(seal_path)


def _verify_inference_seal(root: Path) -> dict[str, Any]:
    seal_path = root / "inference_seal.json"
    seal = _read_json(seal_path)
    _assert_exact_keys(
        seal,
        {
            "schema_version": 1,
            "status": "sealed_before_outcome_join",
            "preoutcome": True,
            "score_artifact_count": 15,
            "n_slides": EXPECTED_SLIDES,
            "n_patients": EXPECTED_PATIENTS,
            "model_forward_passes_per_slide": 27,
            "target_outcomes_present": False,
            "confirmatory_cpht_model_present": False,
            "complete_eight_model_matrix": False,
        },
        seal_path,
    )
    if seal.get("manifest") != _artifact(_manifest_path(root)):
        raise ContractError("Inference seal manifest identity mismatch")
    if seal.get("contract") != _artifact(_contract_path(root)):
        raise ContractError("Inference seal contract identity mismatch")
    contract = _read_json(_contract_path(root))
    if seal.get("inference_environment") != contract["inference"]["environment"]:
        raise ContractError("Inference seal environment identity mismatch")
    expected = expected_score_paths(root)
    records = seal.get("score_artifacts") or []
    if len(records) != len(expected):
        raise ContractError("Inference seal has the wrong prediction roster")
    for path, record in zip(expected, records, strict=True):
        if record.get("score") != _artifact(path):
            raise ContractError(f"Inference score changed after seal: {path}")
        if record.get("receipt") != _artifact(_receipt_path(path)):
            raise ContractError(f"Inference receipt changed after seal: {path}")
    return seal


def _load_model_scores(root: Path) -> dict[str, pd.DataFrame]:
    scores: dict[str, pd.DataFrame] = {}
    aim1 = pd.concat(
        [pd.read_parquet(_score_path(root, "aim1", seed)) for seed in SEEDS],
        ignore_index=True,
    )
    scores["aim1_outer15"] = aim1
    for target in FAMILY_LOCO_TARGETS:
        scores[f"loco_heldout_{target.lower()}"] = pd.concat(
            [
                pd.read_parquet(_score_path(root, "loco", seed, target))
                for seed in SEEDS
            ],
            ignore_index=True,
        )
    return scores


def aggregate_patient_logits(
    scores: pd.DataFrame, manifest: pd.DataFrame, *, expected_models: int
) -> pd.DataFrame:
    """Mean model logits per slide, then mean slide logits per patient."""

    required = {"slide_id", "seed", "fold", "logit"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ContractError(f"Score table lacks columns: {missing}")
    if scores.duplicated(["slide_id", "seed", "fold"]).any():
        raise ContractError("Score table contains duplicate slide/model rows")
    counts = scores.groupby("slide_id", sort=False).size()
    if not counts.eq(expected_models).all() or len(counts) != len(manifest):
        raise ContractError(
            f"Every Orion slide must have {expected_models} model logits; "
            f"observed counts={sorted(counts.unique().tolist())}"
        )
    slide = scores.groupby("slide_id", as_index=False, sort=False)["logit"].mean()
    slide = slide.rename(columns={"logit": "slide_mean_logit"}).merge(
        manifest,
        on="slide_id",
        how="inner",
        validate="one_to_one",
    )
    if len(slide) != len(manifest):
        raise ContractError("Scores and Orion inference manifest cover different slides")
    patient = (
        slide.sort_values("slide_id", kind="stable")
        .groupby("patient_id", as_index=False, sort=True)
        .agg(
            mean_logit=("slide_mean_logit", "mean"),
            n_slides=("slide_id", "count"),
            exclude_neoadjuvant=("exclude_neoadjuvant", "max"),
            exclude_ambiguous_crc15=("exclude_ambiguous_crc15", "max"),
        )
    )
    if len(patient) != EXPECTED_PATIENTS:
        raise ContractError(f"Patient aggregation produced {len(patient)} rows")
    c33 = patient[patient["patient_id"].eq("ORION:C33")]
    if len(c33) != 1 or int(c33.iloc[0]["n_slides"]) != 2:
        raise ContractError("C33 was not aggregated from exactly two slides")
    if not patient.loc[~patient["patient_id"].eq("ORION:C33"), "n_slides"].eq(1).all():
        raise ContractError("An unexpected Orion patient has multiple slides")
    return patient


def _patient_labels(master: pd.DataFrame) -> pd.DataFrame:
    rows = _orion_rows(master)
    if "kras" not in rows.columns:
        raise ContractError("v5 label master lacks KRAS outcomes")
    known = rows["kras"].astype(str)
    if not known.isin(["mutant", "wild_type"]).all():
        raise ContractError("Orion includes unknown/non-binary KRAS values")
    labels = rows.assign(target_label=known.eq("mutant").astype(int))[
        ["patient_uid", "target_label"]
    ]
    consistency = labels.groupby("patient_uid")["target_label"].nunique()
    if not consistency.eq(1).all():
        raise ContractError("Orion patient slides disagree on KRAS label")
    patients = (
        labels.drop_duplicates("patient_uid")
        .rename(columns={"patient_uid": "patient_id", "target_label": "label"})
        .sort_values("patient_id", kind="stable")
        .reset_index(drop=True)
    )
    n_mutant = int(patients["label"].sum())
    if len(patients) != EXPECTED_PATIENTS or n_mutant != EXPECTED_MUTANT_PATIENTS:
        raise ContractError(
            f"Orion outcome census is n={len(patients)}, mutant={n_mutant}; "
            f"expected {EXPECTED_PATIENTS}/{EXPECTED_MUTANT_PATIENTS}"
        )
    return patients


def stratified_bootstrap_indices(
    labels: np.ndarray, *, n_bootstrap: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED
) -> np.ndarray:
    """Shared fixed-class-count patient bootstrap indices."""

    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("KRAS-stratified bootstrap requires both binary classes")
    rng = np.random.default_rng(seed)
    per_class = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        per_class.append(rng.choice(candidates, size=(n_bootstrap, len(candidates)), replace=True))
    return np.concatenate(per_class, axis=1)


def bootstrap_metric_samples(
    labels: np.ndarray,
    score: np.ndarray,
    indices: np.ndarray,
    probability: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Vectorized rank/probability metrics on shared patient resamples."""

    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(labels) != len(score) or indices.ndim != 2:
        raise ValueError("Metric inputs have incompatible shapes")
    point: dict[str, float] = {
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
    }
    draw_labels = labels[indices]
    draw_scores = score[indices]
    order = np.argsort(-draw_scores, axis=1, kind="stable")
    ranked_labels = np.take_along_axis(draw_labels, order, axis=1)
    ranked_scores = np.take_along_axis(draw_scores, order, axis=1)
    cumulative = np.cumsum(ranked_labels, axis=1)
    # Average precision evaluates precision only after each complete score-tie
    # block. Bootstrap resampling necessarily duplicates patients/scores, so a
    # naive stable-rank formula would depend on the arbitrary order within a
    # tie and would not reproduce sklearn's threshold-based definition.
    tie_end = np.ones_like(ranked_labels, dtype=bool)
    tie_end[:, :-1] = ranked_scores[:, :-1] != ranked_scores[:, 1:]
    ap = np.zeros(len(indices), dtype=float)
    previous_tp = np.zeros(len(indices), dtype=float)
    n_positive = ranked_labels.sum(axis=1)
    for column in range(ranked_labels.shape[1]):
        end = tie_end[:, column]
        current_tp = cumulative[:, column].astype(float)
        increment = current_tp - previous_tp
        precision = current_tp / float(column + 1)
        ap += np.where(end, increment / n_positive * precision, 0.0)
        previous_tp = np.where(end, current_tp, previous_tp)

    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    negative_width = len(negative)
    # stratified_bootstrap_indices concatenates the class-0 and class-1 draws.
    sampled_negative = draw_scores[:, :negative_width]
    sampled_positive = draw_scores[:, negative_width:]
    pairwise = sampled_positive[:, :, None] - sampled_negative[:, None, :]
    auc = (
        (pairwise > 0).sum(axis=(1, 2))
        + 0.5 * (pairwise == 0).sum(axis=(1, 2))
    ) / (len(negative) * len(positive))
    samples: dict[str, np.ndarray] = {"auroc": auc.astype(float), "auprc": ap.astype(float)}
    samples["point"] = np.asarray([point["auroc"], point["auprc"]], dtype=float)

    if probability is not None:
        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if len(probability) != len(labels):
            raise ValueError("Probability and label lengths differ")
        draw_probability = probability[indices]
        samples["brier"] = np.mean((draw_probability - draw_labels) ** 2, axis=1)
        samples["log_loss"] = -np.mean(
            draw_labels * np.log(draw_probability)
            + (1 - draw_labels) * np.log(1 - draw_probability),
            axis=1,
        )
    return samples


def _interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _rank_metric_block(
    labels: np.ndarray, score: np.ndarray, samples: dict[str, np.ndarray]
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    auroc = float(roc_auc_score(labels, score))
    return {
        "n": int(len(labels)),
        "n_mutant": int(labels.sum()),
        "n_wild_type": int(len(labels) - labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": auroc,
        "auroc_ci95": _interval(samples["auroc"]),
        "auprc": float(average_precision_score(labels, score)),
        "auprc_ci95": _interval(samples["auprc"]),
        "auprc_baseline": float(labels.mean()),
        "lower_auroc_bound_above_0p5": bool(np.percentile(samples["auroc"], 2.5) > 0.5),
    }


def _probability_metric_block(
    labels: np.ndarray, probability: np.ndarray, samples: dict[str, np.ndarray]
) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss, log_loss

    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    calibration = compute_calibration_intercept_slope(labels, probability)
    keep = (
        "calibration_intercept",
        "calibration_slope",
        "slope_model_intercept",
        "slope_converged",
    )
    return {
        "brier": float(brier_score_loss(labels, probability)),
        "brier_ci95": _interval(samples["brier"]),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "log_loss_ci95": _interval(samples["log_loss"]),
        **{key: calibration.get(key) for key in keep},
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


def analyse_scores(
    model_scores: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    contract: dict[str, Any],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    patient_frames: dict[str, pd.DataFrame] = {}
    for scorer, frame in model_scores.items():
        expected_models = 15 if scorer == "aim1_outer15" else 3
        patient = aggregate_patient_logits(frame, manifest, expected_models=expected_models)
        patient = patient.merge(labels, on="patient_id", how="left", validate="one_to_one")
        if patient["label"].isna().any():
            raise ContractError(f"Missing Orion labels after sealed inference for {scorer}")
        patient["label"] = patient["label"].astype(int)
        patient["prob_raw"] = sigmoid(patient["mean_logit"].to_numpy())
        patient["scorer"] = scorer
        if scorer.startswith("loco_heldout_"):
            target = scorer.removeprefix("loco_heldout_")
            canonical = next(t for t in FAMILY_LOCO_TARGETS if t.lower() == target)
            calibration = contract["family_loco"]["models"][canonical]["calibrator"]
            # The fitted Platt map is a + b * native ensemble logit. Applying
            # it through probability space would invoke EPS clipping at
            # |eta| ~= 13.8 and erase genuine scale shift in this extreme
            # target domain.
            patient["prob_source_calibrated"] = sigmoid(
                float(calibration["a"])
                + float(calibration["b"]) * patient["mean_logit"].to_numpy()
            )
        else:
            patient["prob_source_calibrated"] = np.nan
        patient_frames[scorer] = patient.sort_values("patient_id", kind="stable").reset_index(
            drop=True
        )

    scorer_order = ["aim1_outer15"] + [
        f"loco_heldout_{target.lower()}" for target in FAMILY_LOCO_TARGETS
    ]
    reference_ids = patient_frames[scorer_order[0]]["patient_id"].tolist()
    reference_labels = patient_frames[scorer_order[0]]["label"].tolist()
    for scorer in scorer_order[1:]:
        if patient_frames[scorer]["patient_id"].tolist() != reference_ids:
            raise ContractError(f"Scorers cover different Orion patients: {scorer}")
        if patient_frames[scorer]["label"].tolist() != reference_labels:
            raise ContractError(f"Scorers carry different labels: {scorer}")

    populations = {
        "all_40": np.ones(EXPECTED_PATIENTS, dtype=bool),
        "exclude_neoadjuvant": ~patient_frames[scorer_order[0]][
            "exclude_neoadjuvant"
        ].to_numpy(dtype=bool),
        "exclude_ambiguous_crc15": ~patient_frames[scorer_order[0]][
            "exclude_ambiguous_crc15"
        ].to_numpy(dtype=bool),
    }
    population_results: dict[str, Any] = {}
    for population_name, mask in populations.items():
        base = patient_frames[scorer_order[0]].loc[mask].reset_index(drop=True)
        y = base["label"].to_numpy(dtype=int)
        indices = stratified_bootstrap_indices(
            y, n_bootstrap=n_bootstrap, seed=bootstrap_seed
        )
        metric_samples: dict[str, dict[str, np.ndarray]] = {}
        metrics: dict[str, Any] = {}
        for scorer in scorer_order:
            patient = patient_frames[scorer].loc[mask].reset_index(drop=True)
            probability = (
                None
                if scorer == "aim1_outer15"
                else patient["prob_source_calibrated"].to_numpy(dtype=float)
            )
            samples = bootstrap_metric_samples(
                y,
                patient["mean_logit"].to_numpy(dtype=float),
                indices,
                probability=probability,
            )
            metric_samples[scorer] = samples
            block = _rank_metric_block(
                y, patient["mean_logit"].to_numpy(dtype=float), samples
            )
            if probability is not None:
                block["source_calibrated"] = _probability_metric_block(
                    y, probability, samples
                )
            else:
                block["probability_metrics"] = (
                    "not inferred: final-v8 defines the Aim1 outer15 ensemble as rank-only"
                )
            metrics[scorer] = block

        contrasts: dict[str, Any] = {}
        for left_index, left in enumerate(scorer_order):
            for right in scorer_order[left_index + 1 :]:
                point = metrics[right]["auroc"] - metrics[left]["auroc"]
                delta = metric_samples[right]["auroc"] - metric_samples[left]["auroc"]
                contrasts[f"{right}_minus_{left}"] = {
                    "delta_auroc": float(point),
                    "delta_auroc_ci95": _interval(delta),
                    "paired_shared_patient_bootstrap": True,
                    "inferential_role": "descriptive sensitivity; no model selection",
                }
        population_results[population_name] = {
            "n": int(len(base)),
            "n_mutant": int(y.sum()),
            "n_wild_type": int(len(y) - y.sum()),
            "bootstrap_indices_sha256": _sha256_bytes(indices.astype("<i8").tobytes()),
            "metrics": metrics,
            "paired_auroc_contrasts": contrasts,
        }

    patient_table = pd.concat(
        [patient_frames[scorer] for scorer in scorer_order], ignore_index=True
    )
    column_order = [
        "scorer",
        "patient_id",
        "label",
        "mean_logit",
        "prob_raw",
        "prob_source_calibrated",
        "n_slides",
        "exclude_neoadjuvant",
        "exclude_ambiguous_crc15",
    ]
    results = {
        "schema_version": 1,
        "experiment": "E2-CPHT existing-model Orion sensitivities",
        "analysis_role": "preliminary sensitivities; not confirmatory CPHT",
        "confirmatory_cpht_completed": False,
        "confirmatory_all_conventional_refits": "pending",
        "aim1_outer15_role": "rank-only sensitivity",
        "family_loco_matrix": {
            "status": "four-family partial",
            "complete_eight_model_matrix": False,
            "available": list(FAMILY_LOCO_TARGETS),
            "pending": list(PENDING_SIBLING_TARGETS),
        },
        "population": {
            "slides": EXPECTED_SLIDES,
            "patients": EXPECTED_PATIENTS,
            "mutant": EXPECTED_MUTANT_PATIENTS,
            "wild_type": EXPECTED_WILD_TYPE_PATIENTS,
            "multislide_patient": "ORION:C33 (CRC33_01 and CRC33_02)",
        },
        "statistics": {
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_method": "KRAS-stratified patient percentile bootstrap",
            "shared_indices_across_models": True,
        },
        "populations": population_results,
        "reading_rule": (
            "No Orion result selects, reweights, or promotes a checkpoint/model. "
            "Only the pending new all-conventional refit can populate the confirmatory CPHT row."
        ),
    }
    return patient_table[column_order], _json_safe(results)


def _compact_results(results: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population, population_block in results["populations"].items():
        for scorer, metrics in population_block["metrics"].items():
            calibrated = metrics.get("source_calibrated") or {}
            rows.append(
                {
                    "population": population,
                    "scorer": scorer,
                    "n": metrics["n"],
                    "n_mutant": metrics["n_mutant"],
                    "auroc": metrics["auroc"],
                    "auroc_ci_low": metrics["auroc_ci95"][0],
                    "auroc_ci_high": metrics["auroc_ci95"][1],
                    "auprc": metrics["auprc"],
                    "auprc_ci_low": metrics["auprc_ci95"][0],
                    "auprc_ci_high": metrics["auprc_ci95"][1],
                    "brier_source_calibrated": calibrated.get("brier"),
                    "log_loss_source_calibrated": calibrated.get("log_loss"),
                    "calibration_intercept": calibrated.get("calibration_intercept"),
                    "calibration_slope": calibrated.get("calibration_slope"),
                }
            )
    return pd.DataFrame(rows)


def cmd_preflight(args: argparse.Namespace) -> None:
    master = pd.read_csv(args.label_source)
    pack = _validate_pack(args.pack_dir, args.feature_dir)
    pack_index = pd.read_parquet(args.pack_dir / "index.parquet")
    manifest = build_label_blind_manifest(master, pack_index)
    aim1 = _resolve_aim1_checkpoints(args.aim1_root)
    loco = _resolve_loco_models(args.loco_root)
    environment = inference_environment()
    print(
        f"READY: {len(manifest)} Orion slides / {manifest.patient_id.nunique()} patients; "
        f"pack {pack['n_slides']} slides x {pack['feature_dim']}"
    )
    print(f"  Aim1 selected folds: {sum(len(value) for value in aim1.values())}/15")
    print(f"  family-LOCO refits: {sum(3 for _ in loco)}/12")
    print(
        f"  governed inference: {environment['device_name']} / "
        f"CUDA {environment['torch_cuda_version']} / {environment['autocast_dtype']}"
    )
    print("  sibling-stratum LOCO ensembles pending: " + ", ".join(PENDING_SIBLING_TARGETS))
    print("  confirmatory all-conventional three-seed refit pending")


def cmd_manifest(args: argparse.Namespace) -> None:
    master = pd.read_csv(args.label_source)
    pack_index = pd.read_parquet(args.pack_dir / "index.parquet")
    manifest = build_label_blind_manifest(master, pack_index)
    contract = build_contract(
        manifest,
        label_source=args.label_source,
        feature_dir=args.feature_dir,
        pack_dir=args.pack_dir,
        aim1_root=args.aim1_root,
        loco_root=args.loco_root,
    )
    _publish_csv(_manifest_path(args.output_root), manifest)
    _publish_json(_contract_path(args.output_root), contract)
    print(f"sealed label-blind manifest: {_manifest_path(args.output_root)}")
    print(f"sealed pre-outcome contract: {_contract_path(args.output_root)}")


def cmd_score(args: argparse.Namespace) -> None:
    manifest, contract = _load_sealed_inputs(args.output_root)
    _verify_sealed_feature_store(contract)
    if inference_environment() != contract["inference"]["environment"]:
        raise ContractError("Inference implementation/runtime changed after contract seal")
    if args.device != contract["inference"]["environment"]["device"]:
        raise ContractError("CLI inference device differs from the governed contract")
    if args.num_workers != contract["inference"]["environment"]["num_workers"]:
        raise ContractError("CLI worker count differs from the governed contract")
    if args.arm in ("all", "aim1"):
        print("Aim1 outer15 rank sensitivity", flush=True)
        for seed in SEEDS:
            _score_one_seed(
                args.output_root,
                manifest,
                contract,
                kind="aim1",
                seed=seed,
                target=None,
                device=args.device,
                num_workers=args.num_workers,
            )
    if args.arm in ("all", "family-loco"):
        for target in FAMILY_LOCO_TARGETS:
            print(f"family-LOCO held out {target}", flush=True)
            for seed in SEEDS:
                _score_one_seed(
                    args.output_root,
                    manifest,
                    contract,
                    kind="loco",
                    seed=seed,
                    target=target,
                    device=args.device,
                    num_workers=args.num_workers,
                )
    try:
        seal = _seal_inference(args.output_root)
    except ContractError as exc:
        if args.arm == "all":
            raise
        print(f"partial scoring complete; inference not yet sealable: {exc}")
    else:
        print(
            f"sealed {seal['score_artifact_count']} label-blind score artifacts: "
            f"{args.output_root / 'inference_seal.json'}"
        )


def cmd_report(args: argparse.Namespace) -> None:
    seal = _verify_inference_seal(args.output_root)
    manifest, contract = _load_sealed_inputs(args.output_root)
    if analysis_environment() != contract["statistics"]["analysis_environment"]:
        raise ContractError("Analysis implementation/runtime changed after contract seal")
    if _artifact(args.label_source) != contract["label_source_identity_not_opened_by_score"]:
        raise ContractError("Report label source differs from the identity frozen pre-outcome")
    master = pd.read_csv(args.label_source)
    labels = _patient_labels(master)
    patient_scores, results = analyse_scores(
        _load_model_scores(args.output_root), manifest, labels, contract
    )
    analysis_dir = args.output_root / "analysis"
    patient_path = analysis_dir / "orion_patient_scores.parquet"
    results_path = analysis_dir / "results.json"
    table_path = analysis_dir / "results.csv"
    receipt_path = analysis_dir / "receipt.json"
    if any(path.exists() for path in (patient_path, results_path, table_path, receipt_path)):
        raise FileExistsError(
            f"Analysis output already exists below {analysis_dir}; immutable reruns need a new root"
        )
    _publish_parquet(patient_path, patient_scores)
    _publish_json(results_path, results)
    _publish_csv(table_path, _compact_results(results))
    _publish_json(
        receipt_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "analysis_complete_pending_independent_verification",
            "inference_seal": _artifact(args.output_root / "inference_seal.json"),
            "label_source_joined_after_seal": _artifact(args.label_source),
            "outputs": {
                "patient_scores": _artifact(patient_path),
                "results": _artifact(results_path),
                "compact_table": _artifact(table_path),
            },
            "inference_seal_created_utc": seal["created_utc"],
            "outcome_join_performed_by": "report subcommand only",
        },
    )
    full = results["populations"]["all_40"]["metrics"]
    print("\nOrion all-40 patient AUROC (10,000 stratified bootstrap draws)")
    for scorer in ["aim1_outer15"] + [
        f"loco_heldout_{target.lower()}" for target in FAMILY_LOCO_TARGETS
    ]:
        block = full[scorer]
        lo, hi = block["auroc_ci95"]
        print(f"  {scorer:24s} {block['auroc']:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"\nanalysis written: {analysis_dir}")
    print("status: preliminary sensitivities; confirmatory all-conventional CPHT remains pending")


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--label-source", type=Path, default=LABEL_SOURCE)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    parser.add_argument("--aim1-root", type=Path, default=AIM1_ROOT)
    parser.add_argument("--loco-root", type=Path, default=LOCO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="validate packs, census, and model receipts")
    _add_common_paths(preflight)
    preflight.set_defaults(func=cmd_preflight)

    manifest = sub.add_parser("manifest", help="seal label-blind inputs and model contract")
    _add_common_paths(manifest)
    manifest.set_defaults(func=cmd_manifest)

    score = sub.add_parser("score", help="run packed full-bag label-blind inference")
    _add_common_paths(score)
    score.add_argument("--arm", choices=("all", "aim1", "family-loco"), default="all")
    score.add_argument("--device", choices=("cuda",), default=INFERENCE_DEVICE)
    score.add_argument("--num-workers", type=int, default=INFERENCE_NUM_WORKERS)
    score.set_defaults(func=cmd_score)

    report = sub.add_parser("report", help="join outcomes after the inference seal and evaluate")
    _add_common_paths(report)
    report.set_defaults(func=cmd_report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (ContractError, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"E2-CPHT FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
