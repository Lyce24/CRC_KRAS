#!/usr/bin/env python3
"""E2-MET - final-v8 primary-to-metastatic transfer and model hierarchy.

This is an additive final-v8 runner.  It never trains a metastatic or
organ-specific MIL model and it never writes into the frozen final-v7 family
LOCO lineage.  Instead it consumes three explicit, read-only model roots:

* four frozen family-held-out ensembles (TCGA, SurGen, RIH, CPTAC);
* four new sibling-stratum ensembles (SR386, SR1482, COAD, READ); and
* the new all-conventional deployment refit used by E2-CPHT.

The held-out ensembles form the required 8-scorer x 2-metastatic-target
source-composition matrix.  The all-conventional refit is deliberately kept
outside that matrix: it is target-primary-exposed deployment sensitivity, not
external transport.  RIH's family-naive scorer is reported on all 85
metastatic patients, but every model comparison is made on the same 77
patients after excluding the eight patients whose RIH primaries could have
entered another scorer's training pool.

Command boundary::

    python aim2_primary_to_metastatic_transfer.py preflight \
      --sibling-root <final-v8-lineage>/e2ad \
      --all-conventional-root <final-v8-lineage>/cpht
    python aim2_primary_to_metastatic_transfer.py manifest  ...
    python aim2_primary_to_metastatic_transfer.py score     ...
    python aim2_primary_to_metastatic_transfer.py seal      ...
    python aim2_primary_to_metastatic_transfer.py report    ...

``manifest`` seals label-blind inference manifests and every checkpoint,
configuration and source-only calibrator identity.  ``score`` reads only
those label-blind manifests.  ``report`` is the first command that joins KRAS
outcomes.  All generated artifacts live below ``--output-root``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim2_cross_protocol_transfer as cpht  # noqa: E402
from oceanpath.aim1 import lineage  # noqa: E402
from oceanpath.eval.external import sigmoid  # noqa: E402


class ContractError(RuntimeError):
    """A governed E2-MET input or output violates the final-v8 contract."""


SEEDS: tuple[int, ...] = (42, 43, 44)
CAP = 8_192
STEP_BUDGET = 6_060
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_817

DEFAULT_FAMILY_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim2_cap8192_v4_20260819/e2a"
)
DEFAULT_FEATURE_DIR = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/20x_256px_0px_overlap_mpp0.5/features_uni_v1"
)
DEFAULT_PACK_DIR = DEFAULT_FEATURE_DIR.parent / "packed_uni_v1"
DEFAULT_OUTPUT_ROOT = REPO / "reports/reruns/final_v8_additions_20260822/e2met"
MANIFEST_ROOT = Path("/mnt/d/YC.Liu/manifests/colon")

# Canonical scorer order is frozen because it controls table order and the
# deterministic paired-contrast roster.  The value is the model-directory and
# calibrator slug below the corresponding explicit read-only root.
FAMILY_SCORERS: dict[str, str] = {
    "heldout_tcga": "tcga",
    "heldout_surgen": "surgen",
    "heldout_rih": "rih",
    "heldout_cptac": "cptac",
}
SIBLING_SCORERS: dict[str, str] = {
    "heldout_sr386": "sr386",
    "heldout_sr1482": "sr1482",
    "heldout_tcga_coad": "tcga_coad",
    "heldout_tcga_read": "tcga_read",
}
HELDOUT_SCORERS: tuple[str, ...] = tuple(FAMILY_SCORERS) + tuple(SIBLING_SCORERS)
ALL_CONVENTIONAL_SCORER = "all_conventional"
ALL_SCORERS: tuple[str, ...] = HELDOUT_SCORERS + (ALL_CONVENTIONAL_SCORER,)

EXPECTED_SOURCE_PATIENTS: dict[str, int] = {
    "heldout_tcga": 984,
    "heldout_surgen": 749,
    "heldout_rih": 1_333,
    "heldout_cptac": 1_392,
    "heldout_sr386": 1_073,
    "heldout_sr1482": 1_162,
    "heldout_tcga_coad": 1_112,
    "heldout_tcga_read": 1_358,
    ALL_CONVENTIONAL_SCORER: 1_486,
}
EXPECTED_SOURCE_MUTANTS: dict[str, int] = {
    "heldout_tcga": 397,
    "heldout_surgen": 310,
    "heldout_rih": 534,
    "heldout_cptac": 571,
    "heldout_sr386": 457,
    "heldout_sr1482": 457,
    "heldout_tcga_coad": 444,
    "heldout_tcga_read": 557,
    ALL_CONVENTIONAL_SCORER: 604,
}

FAMILY_TARGET_VALUES: dict[str, str] = {
    "heldout_tcga": "TCGA",
    "heldout_surgen": "SurGen",
    "heldout_rih": "RIH",
    "heldout_cptac": "CPTAC",
}
SIBLING_TARGET_VALUES: dict[str, tuple[str, str]] = {
    "heldout_sr386": ("SR386", "SR1482"),
    "heldout_sr1482": ("SR1482", "SR386"),
    "heldout_tcga_coad": ("TCGA-COAD", "TCGA-READ"),
    "heldout_tcga_read": ("TCGA-READ", "TCGA-COAD"),
}

TARGET_SOURCES: dict[str, Path] = {
    "rih_primary": MANIFEST_ROOT / "aim1_e2a_rih_primary.csv",
    "rih_m": MANIFEST_ROOT / "aim1_e2a_rih_metastatic.csv",
    "sr1482_primary": MANIFEST_ROOT / "aim1_e2a_surgen_primary.csv",
    "sr1482_m": MANIFEST_ROOT / "aim1_e2a_surgen_metastatic.csv",
}
TARGET_EXPECTED: dict[str, tuple[int, int]] = {
    "rih_primary": (153, 70),
    "rih_m": (85, 37),
    "sr1482_primary": (324, 147),
    "sr1482_m": (74, 30),
}
MET_TARGETS: tuple[str, ...] = ("rih_m", "sr1482_m")

# Only these three scorer/target pairs are permitted to enter an honest role
# contrast.  In particular, an all-conventional primary-vs-metastatic delta is
# forbidden because the primary cohort was part of model fitting.
ROLE_JOBS: tuple[tuple[str, str], ...] = (
    ("heldout_rih", "rih_primary"),
    ("heldout_surgen", "sr1482_primary"),
    ("heldout_sr1482", "sr1482_primary"),
)

EXPECTED_DUAL_RIH = 8
EXPECTED_RIH_COMMON = 77


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"


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


def _publish_text(path: Path, text: str) -> None:
    """Publish once; an exact existing artifact is a resumable no-op."""

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
    _publish_text(path, _canonical_json(_json_safe(value)))


def _publish_csv(path: Path, frame: pd.DataFrame) -> None:
    _publish_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _publish_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        frame.to_parquet(temporary, index=False)
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_parquet_resumable(path: Path, frame: pd.DataFrame) -> None:
    """Publish once, accepting an exact table left by an interrupted report."""

    if path.is_file():
        existing = pd.read_parquet(path)
        try:
            pd.testing.assert_frame_equal(existing, frame, check_exact=True)
        except AssertionError as exc:
            raise FileExistsError(f"Refusing to replace non-identical artifact: {path}") from exc
        return
    _publish_parquet(path, frame)


def _artifact(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], lineage.artifact_identity(path))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"Expected JSON object: {path}")
    return value


def _manifest_path(output_root: Path, target: str) -> Path:
    return output_root / "inputs/manifests" / f"{target}.csv"


def _contract_path(output_root: Path) -> Path:
    return output_root / "inputs/e2met_contract.json"


def _score_path(output_root: Path, scorer: str, target: str, seed: int) -> Path:
    return output_root / "scores" / f"{scorer}_{target}_seed{seed}.parquet"


def _receipt_path(path: Path) -> Path:
    return path.with_suffix(".receipt.json")


def _seal_path(output_root: Path) -> Path:
    return output_root / "inference_seal.json"


def _nested(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ContractError(f"Missing configuration field {dotted!r}")
        value = value[key]
    return value


def _validate_locked_config(path: Path, seed: int, source_manifest: Path) -> None:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "data.csv_path": str(source_manifest.resolve()),
        "data.num_classes": 2,
        "encoder.name": "uni_v1",
        "encoder.feature_dim": 1_024,
        "model.name": "abmil",
        "model.arch": "abmil",
        "model.embed_dim": 512,
        "model.attn_dim": 384,
        "model.gate": True,
        "model.dropout": 0.25,
        "model.input_dropout": 0.1,
        "training.lr": 1e-4,
        "training.weight_decay": 1e-5,
        "training.lr_scheduler": "cosine",
        "training.final_lr_fraction": 0.01,
        "training.loss_type": "bce",
        "training.training_class_weighted_loss": False,
        "training.class_weights": None,
        "training.dataset_max_instances": CAP,
        "training.eval_full_bags": True,
        "training.force_float32": True,
        "training.train_sampling_strategy": "patient_natural",
        "training.sample_weight_column": None,
        "training.refit_max_steps": STEP_BUDGET,
        "training.seed": seed,
        "platform.precision": "bf16-mixed",
    }
    mismatch = {
        key: {"expected": wanted, "observed": _nested(value, key)}
        for key, wanted in expected.items()
        if _nested(value, key) != wanted
    }
    if mismatch:
        raise ContractError(f"Locked model recipe mismatch in {path}: {mismatch}")


def exposure_class(scorer: str, target: str) -> str:
    """Return the frozen family/sibling/target-primary exposure label."""

    if target == "rih_m":
        return "family-naive" if scorer == "heldout_rih" else "target-primary-exposed"
    if target == "sr1482_m":
        if scorer == "heldout_surgen":
            return "family-naive"
        if scorer == "heldout_sr1482":
            return "sibling-exposed"
        return "target-primary-exposed"
    raise ValueError(f"Unknown metastatic target: {target}")


def _root_and_slug(
    scorer: str, family_root: Path, sibling_root: Path, all_conventional_root: Path
) -> tuple[Path, str]:
    if scorer in FAMILY_SCORERS:
        return family_root, FAMILY_SCORERS[scorer]
    if scorer in SIBLING_SCORERS:
        return sibling_root, SIBLING_SCORERS[scorer]
    if scorer == ALL_CONVENTIONAL_SCORER:
        return all_conventional_root, ALL_CONVENTIONAL_SCORER
    raise ValueError(f"Unknown scorer: {scorer}")


def _verified_artifact_path(record: Any, context: str) -> Path:
    """Fail closed on a complete, live artifact identity and return its path."""

    if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
        raise ContractError(f"Malformed artifact identity: {context}")
    path = Path(str(record["path"]))
    if _artifact(path) != record:
        raise ContractError(f"Recorded artifact changed: {context}: {path}")
    return path


def _source_manifest_from_request(
    request: dict[str, Any], context: str
) -> tuple[Path, dict[str, Any]]:
    record = (request.get("inputs") or {}).get("source_manifest") or request.get("source_manifest")
    return _verified_artifact_path(record, f"{context}/source_manifest"), cast(
        dict[str, Any], record
    )


def _validate_source_population(path: Path, scorer: str) -> None:
    """Verify the exact primary-only training population implied by a scorer name."""

    frame = pd.read_csv(path, low_memory=False)
    required = {"slide_id", "patient_id", "target_label", "cohort", "subcohort", "specimen_role"}
    missing = required - set(frame)
    if missing:
        raise ContractError(f"Source manifest for {scorer} lacks {sorted(missing)}: {path}")
    if frame["slide_id"].astype(str).duplicated().any():
        raise ContractError(f"Source manifest for {scorer} has duplicate slide IDs")
    if set(frame["specimen_role"].dropna().astype(str)) != {"primary"}:
        raise ContractError(f"Source manifest for {scorer} is not primary-only")
    labels = frame.groupby(frame["patient_id"].astype(str))["target_label"].nunique()
    if not labels.eq(1).all():
        raise ContractError(f"Source manifest for {scorer} has within-patient label disagreement")
    patient = frame.drop_duplicates("patient_id")
    observed = (len(patient), int(pd.to_numeric(patient["target_label"], errors="raise").sum()))
    expected = (EXPECTED_SOURCE_PATIENTS[scorer], EXPECTED_SOURCE_MUTANTS[scorer])
    if observed != expected:
        raise ContractError(f"Source population for {scorer} is {observed}, expected {expected}")

    if scorer in FAMILY_TARGET_VALUES:
        held_out = FAMILY_TARGET_VALUES[scorer]
        if held_out in set(frame["cohort"].astype(str)):
            raise ContractError(f"Held-out family {held_out} leaked into {scorer} source")
    elif scorer in SIBLING_TARGET_VALUES:
        held_out, retained = SIBLING_TARGET_VALUES[scorer]
        values = set(frame["subcohort"].astype(str))
        if held_out in values or retained not in values:
            raise ContractError(
                f"Sibling source roster for {scorer} must exclude {held_out} and retain {retained}"
            )
    elif set(frame["cohort"].astype(str)) != {"TCGA", "SurGen", "RIH", "CPTAC"}:
        raise ContractError("All-conventional source does not contain all four source families")


def _validate_completed_result(summary: dict[str, Any], scorer: str, seed: int) -> None:
    result = summary.get("result") or {}
    expected = {
        "actual_optimizer_steps": STEP_BUDGET,
        "refit_max_steps": STEP_BUDGET,
        "batch_size": 1,
        "accumulate_grad_batches": 1,
        "seed": seed,
        "sampling_seed": seed,
        "lr_scheduler": "cosine",
        "lr_scheduler_interval": "step",
        "lr_scheduler_total_steps": STEP_BUDGET,
        "train_sampling_strategy": "patient_natural",
        "sample_weight_column": None,
        "class_weights": None,
        "dataset_max_instances": CAP,
        "max_instances": None,
        "eval_full_bags": True,
    }
    mismatch = {
        key: {"expected": wanted, "observed": result.get(key)}
        for key, wanted in expected.items()
        if result.get(key) != wanted
    }
    if mismatch:
        raise ContractError(f"Incomplete/noncanonical refit result {scorer}/seed{seed}: {mismatch}")


def resolve_models(
    family_root: Path,
    sibling_root: Path,
    all_conventional_root: Path,
) -> dict[str, Any]:
    """Resolve and validate all nine read-only three-seed ensembles."""

    resolved: dict[str, Any] = {}
    for scorer in ALL_SCORERS:
        root, slug = _root_and_slug(scorer, family_root, sibling_root, all_conventional_root)
        seed_records: dict[str, Any] = {}
        for seed in SEEDS:
            run = root / "train" / f"pb_cap{CAP}" / slug / f"seed{seed}"
            checkpoint = run / "final/refit/model.ckpt"
            summary_path = run / "fit_summary.json"
            request_path = run / "run_request.json"
            for required in (checkpoint, summary_path, request_path):
                if not required.is_file():
                    raise FileNotFoundError(required)
            summary = _read_json(summary_path)
            expected_summary = {
                "status": "completed",
                "seed": seed,
                "sampling_seed": seed,
                "cap": CAP,
                "optimizer_step_budget": STEP_BUDGET,
            }
            mismatch = {
                key: {"expected": wanted, "observed": summary.get(key)}
                for key, wanted in expected_summary.items()
                if summary.get(key) != wanted
            }
            if mismatch:
                raise ContractError(f"Incomplete/noncanonical refit {run}: {mismatch}")
            if (
                scorer in FAMILY_TARGET_VALUES
                and summary.get("target") != FAMILY_TARGET_VALUES[scorer]
            ):
                raise ContractError(f"Wrong family holdout identity in {summary_path}")
            if scorer in SIBLING_TARGET_VALUES and summary.get("arm") != slug:
                raise ContractError(f"Wrong sibling holdout identity in {summary_path}")
            if scorer == ALL_CONVENTIONAL_SCORER and summary.get("experiment") != (
                "E2-CPHT raw all-conventional refit"
            ):
                raise ContractError(f"Wrong all-conventional refit identity in {summary_path}")
            if summary.get("model") != _artifact(checkpoint):
                raise ContractError(f"Fit summary does not bind checkpoint: {checkpoint}")
            config = _verified_artifact_path(
                summary.get("resolved_config"), f"{scorer}/seed{seed}/resolved_config"
            )
            _validate_completed_result(summary, scorer, seed)
            request = _read_json(request_path)
            expected_request = {
                "status": "requested",
                "seed": seed,
                "sampling_seed": seed,
                "cap": CAP,
                "optimizer_step_budget": STEP_BUDGET,
            }
            request_mismatch = {
                key: {"expected": wanted, "observed": request.get(key)}
                for key, wanted in expected_request.items()
                if request.get(key) != wanted
            }
            if request_mismatch:
                raise ContractError(f"Noncanonical run request {request_path}: {request_mismatch}")
            if summary.get("run_request") is not None and summary["run_request"] != _artifact(
                request_path
            ):
                raise ContractError(f"Fit summary does not bind run request: {request_path}")
            if request.get("resolved_config") is not None and request[
                "resolved_config"
            ] != _artifact(config):
                raise ContractError(f"Run request does not bind resolved config: {request_path}")
            frozen_config = request.get("frozen_input_config")
            if frozen_config is not None:
                _verified_artifact_path(frozen_config, f"{scorer}/seed{seed}/frozen_input_config")
                if (
                    frozen_config["sha256"],
                    frozen_config["size_bytes"],
                ) != (_artifact(config)["sha256"], _artifact(config)["size_bytes"]):
                    raise ContractError(
                        f"Run-local and frozen input configs differ for {scorer}/seed{seed}"
                    )
            for name, identity in (
                ("manifest_contract", (request.get("inputs") or {}).get("manifest_contract")),
                ("splits", (request.get("inputs") or {}).get("splits") or request.get("splits")),
                ("execution_contract", request.get("execution_contract")),
            ):
                if identity is not None:
                    _verified_artifact_path(identity, f"{scorer}/seed{seed}/{name}")
            source_manifest, source_identity = _source_manifest_from_request(
                request, f"{scorer}/seed{seed}"
            )
            _validate_source_population(source_manifest, scorer)
            _validate_locked_config(config, seed, source_manifest)
            if seed > SEEDS[0]:
                first_source = seed_records[str(SEEDS[0])]["source_manifest"]
                if source_identity != first_source:
                    raise ContractError(f"Source manifest changes across seeds for {scorer}")
            seed_records[str(seed)] = {
                "checkpoint": _artifact(checkpoint),
                "resolved_config": _artifact(config),
                "fit_summary": _artifact(summary_path),
                "run_request": _artifact(request_path),
                "source_manifest": source_identity,
            }

        calibrator_path = root / "calibrators" / f"cap{CAP}_{slug}.json"
        calibrator = _read_json(calibrator_path)
        if not all(np.isfinite(float(calibrator.get(key, np.nan))) for key in ("a", "b")):
            raise ContractError(f"Non-finite source calibrator: {calibrator_path}")
        if float(calibrator["b"]) <= 0:
            raise ContractError(f"Non-monotone source calibrator: {calibrator_path}")
        for flag in ("target_labels_used_for_fit", "target_labels_used"):
            if calibrator.get(flag) not in (None, False):
                raise ContractError(f"Target-informed source calibrator: {calibrator_path}")
        if scorer in SIBLING_TARGET_VALUES and calibrator.get("arm") != slug:
            raise ContractError(f"Wrong sibling calibrator identity: {calibrator_path}")
        expected_n = EXPECTED_SOURCE_PATIENTS[scorer]
        if int(calibrator.get("n_source", -1)) != expected_n:
            raise ContractError(
                f"Source calibrator census for {scorer} is {calibrator.get('n_source')}, "
                f"expected {expected_n}: {calibrator_path}"
            )
        source_calibration_inputs = (
            calibrator.get("source_cv_inputs")
            or calibrator.get("source_cv_receipts")
            or calibrator.get("source_oof_inputs")
            or {}
        )
        if set(source_calibration_inputs) != {str(seed) for seed in SEEDS}:
            raise ContractError(
                f"Incomplete three-seed calibration lineage for {scorer}: "
                f"{sorted(source_calibration_inputs)}"
            )
        for name, identity in source_calibration_inputs.items():
            _verified_artifact_path(identity, f"{scorer} source OOF seed {name}")
        if calibrator.get("source_manifest") is not None:
            calibrator_source = _verified_artifact_path(
                calibrator["source_manifest"], f"{scorer}/calibrator/source_manifest"
            )
            if calibrator_source != Path(seed_records[str(SEEDS[0])]["source_manifest"]["path"]):
                raise ContractError(
                    f"Calibrator and refits use different source manifests for {scorer}"
                )
        resolved[scorer] = {
            "model_root": str(root.resolve()),
            "model_slug": slug,
            "seeds": seed_records,
            "calibrator": {
                "artifact": _artifact(calibrator_path),
                "a": float(calibrator["a"]),
                "b": float(calibrator["b"]),
                "n_source": int(calibrator["n_source"]),
                "source_calibration_inputs": source_calibration_inputs,
            },
        }
    return resolved


def build_input_manifests() -> tuple[dict[str, pd.DataFrame], dict[str, Any], set[str]]:
    """Build the four label-blind inference manifests and audit the outcome sources."""

    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, Any] = {}
    outcome_frames: dict[str, pd.DataFrame] = {}
    required = {"slide_id", "patient_id", "target_label", "specimen_role", "subcohort"}
    for target, source in TARGET_SOURCES.items():
        raw = pd.read_csv(source, low_memory=False)
        missing = required - set(raw.columns)
        if missing:
            raise ContractError(f"{source} lacks required columns: {sorted(missing)}")
        if target.startswith("sr1482_"):
            raw = raw[raw["subcohort"].eq("SR1482")].copy()
        expected_role = "metastatic" if target.endswith("_m") else "primary"
        if set(raw["specimen_role"].dropna().astype(str)) != {expected_role}:
            raise ContractError(f"{target} mixes specimen roles")
        if raw["slide_id"].duplicated().any():
            raise ContractError(f"{target} contains duplicate slide IDs")
        patient_labels = raw.groupby("patient_id")["target_label"].nunique()
        if not patient_labels.eq(1).all():
            raise ContractError(f"{target} has within-patient KRAS disagreement")
        patient = raw.drop_duplicates("patient_id")
        observed = (int(len(patient)), int(patient["target_label"].sum()))
        if observed != TARGET_EXPECTED[target]:
            raise ContractError(
                f"{target} patient census {observed}, expected {TARGET_EXPECTED[target]}"
            )
        if target.endswith("_m"):
            if "liver_class" not in raw.columns:
                raise ContractError(f"{target} lacks governed liver_class")
            organ = patient.groupby(["liver_class", "target_label"]).size().to_dict()
            expected_organ = (
                {("liver", 0): 28, ("liver", 1): 22, ("non_liver", 0): 20, ("non_liver", 1): 15}
                if target == "rih_m"
                else {
                    ("liver", 0): 19,
                    ("liver", 1): 16,
                    ("non_liver", 0): 25,
                    ("non_liver", 1): 14,
                }
            )
            if organ != expected_organ:
                raise ContractError(f"{target} organ/KRAS census changed: {organ}")
        outcome_frames[target] = raw
        keep = ["slide_id", "patient_id", "specimen_role", "subcohort"]
        if "liver_class" in raw.columns:
            keep.append("liver_class")
        frame = raw[keep].copy().sort_values("slide_id", kind="stable").reset_index(drop=True)
        frames[target] = frame
        sources[target] = {
            "artifact": _artifact(source),
            "filter": "subcohort == SR1482" if target.startswith("sr1482_") else None,
            "n_slides": int(len(raw)),
            "n_patients": observed[0],
            "n_mutant": observed[1],
        }

    dual = set(outcome_frames["rih_primary"]["patient_id"]) & set(
        outcome_frames["rih_m"]["patient_id"]
    )
    if len(dual) != EXPECTED_DUAL_RIH:
        raise ContractError(f"RIH dual-role census is {len(dual)}, expected 8")
    return frames, sources, dual


def build_contract(
    output_root: Path,
    *,
    family_root: Path,
    sibling_root: Path,
    all_conventional_root: Path,
    feature_dir: Path,
    pack_dir: Path,
) -> dict[str, Any]:
    models = resolve_models(family_root, sibling_root, all_conventional_root)
    manifests, outcome_sources, dual = build_input_manifests()
    manifest_records = {
        target: {
            "artifact": _artifact(_manifest_path(output_root, target)),
            "n_slides": int(len(frame)),
            "n_patients": int(frame["patient_id"].nunique()),
            "contains_kras_outcomes": False,
        }
        for target, frame in manifests.items()
    }
    feature_store = cpht._validate_pack(pack_dir, feature_dir)  # noqa: SLF001
    return {
        "schema_version": 1,
        "experiment": "E2-MET final-v8 primary-to-metastatic transfer",
        "analysis_role": "confirmatory hierarchy plus prespecified sensitivities",
        "implementation": _artifact(Path(__file__).resolve()),
        "models_are_read_only": True,
        "training_performed_by_this_runner": False,
        "heldout_scorers": list(HELDOUT_SCORERS),
        "complete_eight_model_matrix": len(models) == 9,
        "all_conventional_outside_heldout_matrix": True,
        "models": models,
        "label_blind_inference_manifests": manifest_records,
        "outcome_sources_not_opened_by_score": outcome_sources,
        "rih_dual_role_patients": sorted(dual),
        "rih_pairwise_population": {
            "all_metastatic_for_family_naive_standalone": 85,
            "common_leakage_free_comparisons": 77,
        },
        "feature_store": feature_store,
        "inference": {
            "encoder": "UNIv1",
            "training_cap": CAP,
            "evaluation_bag": "full",
            "environment": cpht.inference_environment(),
        },
        "statistics": {
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "sampling_unit": "patient",
            "target_metrics": "within target x KRAS",
            "role_contrasts": "independent role x KRAS resampling",
            "organ_analysis": "within cohort x organ-bin x KRAS; shared across scorers",
            "paired_model_contrasts": True,
            "analysis_environment": cpht.analysis_environment(),
        },
    }


def _load_contract(output_root: Path) -> dict[str, Any]:
    contract = _read_json(_contract_path(output_root))
    if contract.get("implementation") != _artifact(Path(__file__).resolve()):
        raise ContractError("E2-MET implementation changed after contract seal")
    live_inference_environment = cpht.inference_environment()
    if contract.get("inference", {}).get("environment") != live_inference_environment:
        raise ContractError("E2-MET inference environment changed after contract seal")
    live_analysis_environment = cpht.analysis_environment()
    if contract.get("statistics", {}).get("analysis_environment") != live_analysis_environment:
        raise ContractError("E2-MET analysis environment changed after contract seal")
    for target, record in contract["label_blind_inference_manifests"].items():
        if record["artifact"] != _artifact(_manifest_path(output_root, target)):
            raise ContractError(f"Sealed inference manifest changed: {target}")
    for scorer, model in contract["models"].items():
        for seed in SEEDS:
            for key in ("checkpoint", "resolved_config", "fit_summary", "run_request"):
                _verified_artifact_path(model["seeds"][str(seed)][key], f"{scorer}/{seed}/{key}")
            _verified_artifact_path(
                model["seeds"][str(seed)]["source_manifest"],
                f"{scorer}/{seed}/source_manifest",
            )
        _verified_artifact_path(model["calibrator"]["artifact"], f"{scorer}/calibrator")
        for name, identity in model["calibrator"]["source_calibration_inputs"].items():
            _verified_artifact_path(identity, f"{scorer}/calibrator/source_oof/{name}")
    observed_pack = cpht._validate_pack(  # noqa: SLF001
        Path(contract["feature_store"]["path"]),
        Path(contract["feature_store"]["source_dir"]),
    )
    if observed_pack != contract["feature_store"]:
        raise ContractError("Packed UNIv1 store changed after E2-MET contract seal")
    return contract


def score_jobs() -> list[tuple[str, str, int]]:
    jobs = [
        (scorer, target, seed) for scorer in ALL_SCORERS for target in MET_TARGETS for seed in SEEDS
    ]
    jobs.extend((scorer, target, seed) for scorer, target in ROLE_JOBS for seed in SEEDS)
    return jobs


def _validate_score_frame(
    scores: pd.DataFrame, manifest: pd.DataFrame, *, seed: int, context: Path | str
) -> None:
    expected_columns = ["slide_id", "seed", "fold", "logit"]
    if list(scores.columns) != expected_columns:
        raise ContractError(f"Score schema mismatch for {context}: {list(scores.columns)}")
    if set(scores["seed"].astype(int)) != {seed} or set(scores["fold"].astype(int)) != {0}:
        raise ContractError(f"Wrong seed/fold in {context}")
    if scores.duplicated(["slide_id", "seed", "fold"]).any():
        raise ContractError(f"Duplicate slide/model rows in {context}")
    expected_slides = set(manifest["slide_id"].astype(str))
    if set(scores["slide_id"].astype(str)) != expected_slides or len(scores) != len(manifest):
        raise ContractError(f"Incomplete slide roster in {context}")
    if not np.isfinite(pd.to_numeric(scores["logit"], errors="coerce")).all():
        raise ContractError(f"Non-finite native logits in {context}")


def _checkpoint_identity(contract: dict[str, Any], scorer: str, seed: int) -> dict[str, Any]:
    value = contract["models"][scorer]["seeds"][str(seed)]["checkpoint"]
    if not isinstance(value, dict):
        raise ContractError(f"Invalid checkpoint identity for {scorer}/seed{seed}")
    return cast(dict[str, Any], value)


def _validate_cached_score(
    output_root: Path,
    contract: dict[str, Any],
    scorer: str,
    target: str,
    seed: int,
) -> bool:
    path = _score_path(output_root, scorer, target, seed)
    receipt_path = _receipt_path(path)
    if not path.exists() and not receipt_path.exists():
        return False
    if not path.is_file() or not receipt_path.is_file():
        raise ContractError(f"Partial E2-MET score cache: {path}")
    receipt = _read_json(receipt_path)
    expected = {
        "schema_version": 1,
        "contains_target_outcomes": False,
        "scorer": scorer,
        "target": target,
        "seed": seed,
        "manifest": _artifact(_manifest_path(output_root, target)),
        "contract": _artifact(_contract_path(output_root)),
        "checkpoint": _checkpoint_identity(contract, scorer, seed),
        "inference_environment": contract["inference"]["environment"],
    }
    mismatch = {
        key: (wanted, receipt.get(key))
        for key, wanted in expected.items()
        if receipt.get(key) != wanted
    }
    if mismatch or receipt.get("artifact") != _artifact(path):
        raise ContractError(f"Invalid E2-MET score receipt {receipt_path}: {mismatch}")
    frame = pd.read_parquet(path)
    manifest = pd.read_csv(_manifest_path(output_root, target))
    _validate_score_frame(frame, manifest, seed=seed, context=path)
    return True


def _score_one(
    output_root: Path,
    contract: dict[str, Any],
    scorer: str,
    target: str,
    seed: int,
    *,
    device: str,
    num_workers: int,
) -> None:
    if _validate_cached_score(output_root, contract, scorer, target, seed):
        print(f"  cached and verified: {scorer}/{target}/seed{seed}")
        return
    manifest = pd.read_csv(_manifest_path(output_root, target))
    checkpoint_identity = _checkpoint_identity(contract, scorer, seed)
    checkpoint = Path(checkpoint_identity["path"])
    if _artifact(checkpoint) != checkpoint_identity:
        raise ContractError(f"Checkpoint changed before inference: {checkpoint}")
    scores = cpht._score_checkpoints(  # noqa: SLF001
        [(seed, 0, checkpoint)],
        manifest,
        feature_dir=Path(contract["feature_store"]["source_dir"]),
        pack_dir=Path(contract["feature_store"]["path"]),
        device=device,
        num_workers=num_workers,
    )
    path = _score_path(output_root, scorer, target, seed)
    _validate_score_frame(scores, manifest, seed=seed, context=path)
    _publish_parquet(path, scores)
    _publish_json(
        _receipt_path(path),
        {
            "schema_version": 1,
            "contains_target_outcomes": False,
            "scorer": scorer,
            "target": target,
            "seed": seed,
            "manifest": _artifact(_manifest_path(output_root, target)),
            "contract": _artifact(_contract_path(output_root)),
            "checkpoint": checkpoint_identity,
            "inference_environment": contract["inference"]["environment"],
            "artifact": _artifact(path),
            "n_rows": int(len(scores)),
        },
    )
    print(f"  sealed {scorer}/{target}/seed{seed}: {path}")


def seal_inference(output_root: Path) -> dict[str, Any]:
    seal_path = _seal_path(output_root)
    if seal_path.is_file():
        return verify_inference_seal(output_root)
    contract = _load_contract(output_root)
    artifacts = []
    for scorer, target, seed in score_jobs():
        if not _validate_cached_score(output_root, contract, scorer, target, seed):
            raise ContractError(f"Cannot seal incomplete inference: {scorer}/{target}/seed{seed}")
        path = _score_path(output_root, scorer, target, seed)
        artifacts.append({"score": _artifact(path), "receipt": _artifact(_receipt_path(path))})
    seal = {
        "schema_version": 1,
        "status": "sealed_before_outcome_join",
        "target_outcomes_present": False,
        "contract": _artifact(_contract_path(output_root)),
        "score_artifact_count": len(artifacts),
        "expected_score_artifact_count": len(score_jobs()),
        "complete_eight_model_matrix": True,
        "all_conventional_scored_separately": True,
        "score_artifacts": artifacts,
    }
    _publish_json(seal_path, seal)
    return verify_inference_seal(output_root)


def verify_inference_seal(output_root: Path) -> dict[str, Any]:
    seal = _read_json(_seal_path(output_root))
    expected_count = len(score_jobs())
    expected = {
        "schema_version": 1,
        "status": "sealed_before_outcome_join",
        "target_outcomes_present": False,
        "contract": _artifact(_contract_path(output_root)),
        "score_artifact_count": expected_count,
        "expected_score_artifact_count": expected_count,
        "complete_eight_model_matrix": True,
        "all_conventional_scored_separately": True,
    }
    mismatch = {
        key: (wanted, seal.get(key)) for key, wanted in expected.items() if seal.get(key) != wanted
    }
    if mismatch:
        raise ContractError(f"Invalid inference seal: {mismatch}")
    contract = _load_contract(output_root)
    records = seal.get("score_artifacts") or []
    if len(records) != expected_count:
        raise ContractError("Inference seal score roster is incomplete")
    for record, (scorer, target, seed) in zip(records, score_jobs(), strict=True):
        path = _score_path(output_root, scorer, target, seed)
        if record != {"score": _artifact(path), "receipt": _artifact(_receipt_path(path))}:
            raise ContractError(f"Score changed after seal: {path}")
        _validate_cached_score(output_root, contract, scorer, target, seed)
    return seal


def _outcome_manifest(contract: dict[str, Any], target: str) -> pd.DataFrame:
    source = contract["outcome_sources_not_opened_by_score"][target]
    path = Path(source["artifact"]["path"])
    if _artifact(path) != source["artifact"]:
        raise ContractError(f"Outcome source changed after contract seal: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if source.get("filter") == "subcohort == SR1482":
        frame = frame[frame["subcohort"].eq("SR1482")].copy()
    return frame


def aggregate_patient_scores(
    score_frames: Sequence[pd.DataFrame],
    label_blind_manifest: pd.DataFrame,
    outcome_manifest: pd.DataFrame,
    *,
    calibrator: dict[str, Any],
) -> pd.DataFrame:
    """Three seed slide logits -> one native-logit and probability row/patient."""

    scores = pd.concat(score_frames, ignore_index=True)
    expected_models = {(seed, 0) for seed in SEEDS}
    observed_models = set(zip(scores["seed"].astype(int), scores["fold"].astype(int), strict=True))
    if observed_models != expected_models:
        raise ContractError(f"Incomplete seed ensemble: {sorted(observed_models)}")
    if scores.duplicated(["slide_id", "seed", "fold"]).any():
        raise ContractError("Duplicate score rows before patient aggregation")
    expected_slides = set(label_blind_manifest["slide_id"].astype(str))
    if set(scores["slide_id"].astype(str)) != expected_slides:
        raise ContractError("Score and inference-manifest slide rosters differ")
    slide = scores.groupby("slide_id", sort=False)["logit"].mean().rename("slide_logit")
    merged = label_blind_manifest.merge(
        slide, left_on="slide_id", right_index=True, how="left", validate="one_to_one"
    )
    if merged["slide_logit"].isna().any():
        raise ContractError("Missing slide logits after aggregation")
    labels = outcome_manifest[["patient_id", "target_label"]].drop_duplicates()
    if labels["patient_id"].duplicated().any():
        raise ContractError("Outcome manifest has inconsistent patient rows")
    patient = (
        merged.groupby("patient_id", sort=False)
        .agg(
            mean_logit=("slide_logit", "mean"),
            n_slides=("slide_id", "nunique"),
            specimen_role=("specimen_role", "first"),
            subcohort=("subcohort", "first"),
            **(
                {"liver_class": ("liver_class", "first")} if "liver_class" in merged.columns else {}
            ),
        )
        .reset_index()
        .merge(
            labels.rename(columns={"target_label": "label"}),
            on="patient_id",
            how="left",
            validate="one_to_one",
        )
    )
    if patient["label"].isna().any() or set(patient["label"].astype(int)) != {0, 1}:
        raise ContractError("Missing/non-binary KRAS outcomes after inference join")
    patient["label"] = patient["label"].astype(int)
    patient["prob_raw"] = sigmoid(patient["mean_logit"].to_numpy(dtype=float))
    patient["prob_source_calibrated"] = sigmoid(
        float(calibrator["a"])
        + float(calibrator["b"]) * patient["mean_logit"].to_numpy(dtype=float)
    )
    return patient.sort_values("patient_id", kind="stable").reset_index(drop=True)


def _shared_stratified_indices(
    labels: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ContractError("Every inferential arm must contain both KRAS classes")
    pieces = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        pieces.append(rng.choice(candidates, (n_bootstrap, len(candidates)), replace=True))
    return cast(np.ndarray, np.concatenate(pieces, axis=1))


def _metric_block(
    frame: pd.DataFrame, indices: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels = frame["label"].to_numpy(dtype=int)
    logits = frame["mean_logit"].to_numpy(dtype=float)
    probability = frame["prob_source_calibrated"].to_numpy(dtype=float)
    samples = cpht.bootstrap_metric_samples(labels, logits, indices, probability=probability)
    block = cpht._rank_metric_block(labels, logits, samples)  # noqa: SLF001
    block["source_calibrated"] = cpht._probability_metric_block(  # noqa: SLF001
        labels, probability, samples
    )
    return block, samples


def _align_frames(
    frames: dict[str, pd.DataFrame], patient_ids: Iterable[str]
) -> dict[str, pd.DataFrame]:
    ordered = sorted(set(map(str, patient_ids)))
    aligned: dict[str, pd.DataFrame] = {}
    for scorer, frame in frames.items():
        part = frame[frame["patient_id"].astype(str).isin(ordered)].copy()
        part["patient_id"] = part["patient_id"].astype(str)
        part = part.set_index("patient_id").loc[ordered].reset_index()
        if part["patient_id"].tolist() != ordered:
            raise ContractError(f"{scorer} does not cover the common patient roster")
        aligned[scorer] = part
    return aligned


def _paired_contrasts(
    metrics: dict[str, dict[str, Any]],
    samples: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    order = list(metrics)
    contrasts: dict[str, Any] = {}
    for left_index, left in enumerate(order):
        for right in order[left_index + 1 :]:
            draws = samples[right]["auroc"] - samples[left]["auroc"]
            contrasts[f"{right}_minus_{left}"] = {
                "delta_auroc": float(metrics[right]["auroc"] - metrics[left]["auroc"]),
                "delta_auroc_ci95": cpht._interval(draws),  # noqa: SLF001
                "paired_shared_patient_bootstrap": True,
            }
    return contrasts


def _role_contrast(
    primary: pd.DataFrame,
    metastatic: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    primary_indices = _shared_stratified_indices(primary["label"].to_numpy(), n_bootstrap, rng)
    met_indices = _shared_stratified_indices(metastatic["label"].to_numpy(), n_bootstrap, rng)
    primary_block, primary_samples = _metric_block(primary, primary_indices)
    met_block, met_samples = _metric_block(metastatic, met_indices)
    delta_auc = met_samples["auroc"] - primary_samples["auroc"]
    delta_ap = met_samples["auprc"] - primary_samples["auprc"]
    delta_brier = met_samples["brier"] - primary_samples["brier"]
    delta_logloss = met_samples["log_loss"] - primary_samples["log_loss"]
    pcal = primary_block["source_calibrated"]
    mcal = met_block["source_calibrated"]
    return {
        "primary": primary_block,
        "metastatic": met_block,
        "delta_metastatic_minus_primary": {
            "auroc": float(met_block["auroc"] - primary_block["auroc"]),
            "auroc_ci95": cpht._interval(delta_auc),  # noqa: SLF001
            "auprc": float(met_block["auprc"] - primary_block["auprc"]),
            "auprc_ci95": cpht._interval(delta_ap),  # noqa: SLF001
            "brier_source_calibrated": float(mcal["brier"] - pcal["brier"]),
            "brier_ci95": cpht._interval(delta_brier),  # noqa: SLF001
            "log_loss_source_calibrated": float(mcal["log_loss"] - pcal["log_loss"]),
            "log_loss_ci95": cpht._interval(delta_logloss),  # noqa: SLF001
            "calibration_intercept": float(
                mcal["calibration_intercept"] - pcal["calibration_intercept"]
            ),
            "calibration_slope": float(mcal["calibration_slope"] - pcal["calibration_slope"]),
        },
        "bootstrap_method": "independent KRAS-stratified patient resampling by role",
    }


def _organ_analysis(
    patient_scores: dict[str, dict[str, pd.DataFrame]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    scorer_sets = {
        "confirmatory_family_naive": {
            "rih_m": "heldout_rih",
            "sr1482_m": "heldout_surgen",
        },
        "sr1482_sibling_exposed_sensitivity": {
            "rih_m": "heldout_rih",
            "sr1482_m": "heldout_sr1482",
        },
    }
    rng = np.random.default_rng(seed)
    # One index tensor per cohort x organ bin, reused by both scorer sets.
    indices: dict[tuple[str, str], np.ndarray] = {}
    organ_rosters: dict[tuple[str, str], list[str]] = {}
    for target in MET_TARGETS:
        base = patient_scores["heldout_rih" if target == "rih_m" else "heldout_surgen"][target]
        for organ in ("liver", "non_liver"):
            frame = (
                base[base["liver_class"].eq(organ)]
                .sort_values("patient_id", kind="stable")
                .reset_index(drop=True)
            )
            organ_rosters[(target, organ)] = frame["patient_id"].astype(str).tolist()
            indices[(target, organ)] = _shared_stratified_indices(
                frame["label"].to_numpy(), n_bootstrap, rng
            )

    analyses: dict[str, Any] = {}
    raw_samples: dict[str, dict[str, np.ndarray]] = {}
    for name, mappings in scorer_sets.items():
        cohorts: dict[str, Any] = {}
        cell_samples: dict[str, np.ndarray] = {}
        for target, scorer in mappings.items():
            frame = patient_scores[scorer][target]
            cohort: dict[str, Any] = {"scorer": scorer}
            for organ in ("liver", "non_liver"):
                part = (
                    frame[frame["liver_class"].eq(organ)]
                    .sort_values("patient_id", kind="stable")
                    .reset_index(drop=True)
                )
                if part["patient_id"].astype(str).tolist() != organ_rosters[(target, organ)]:
                    raise ContractError(
                        f"Organ scorer {scorer}/{target}/{organ} does not cover "
                        "the shared patient roster"
                    )
                block, samples = _metric_block(part, indices[(target, organ)])
                cohort[organ] = block
                cell_samples[f"{target}_{organ}"] = samples["auroc"]
            gap_draws = cell_samples[f"{target}_liver"] - cell_samples[f"{target}_non_liver"]
            cohort["liver_minus_non_liver"] = {
                "delta_auroc": float(cohort["liver"]["auroc"] - cohort["non_liver"]["auroc"]),
                "delta_auroc_ci95": cpht._interval(gap_draws),  # noqa: SLF001
            }
            cohorts[target] = cohort
        liver_draws = np.mean([cell_samples[f"{target}_liver"] for target in MET_TARGETS], axis=0)
        non_draws = np.mean([cell_samples[f"{target}_non_liver"] for target in MET_TARGETS], axis=0)
        gap_draws = liver_draws - non_draws
        macro = {
            "liver_auroc": float(np.mean([cohorts[t]["liver"]["auroc"] for t in MET_TARGETS])),
            "liver_auroc_ci95": cpht._interval(liver_draws),  # noqa: SLF001
            "non_liver_auroc": float(
                np.mean([cohorts[t]["non_liver"]["auroc"] for t in MET_TARGETS])
            ),
            "non_liver_auroc_ci95": cpht._interval(non_draws),  # noqa: SLF001
            "liver_minus_non_liver": float(
                np.mean([cohorts[t]["liver_minus_non_liver"]["delta_auroc"] for t in MET_TARGETS])
            ),
            "liver_minus_non_liver_ci95": cpht._interval(gap_draws),  # noqa: SLF001
        }
        analyses[name] = {"cohorts": cohorts, "equal_cohort_macro": macro}
        raw_samples[name] = {"liver": liver_draws, "non_liver": non_draws, "gap": gap_draws}

    sensitivity = analyses["sr1482_sibling_exposed_sensitivity"]["equal_cohort_macro"]
    confirmatory = analyses["confirmatory_family_naive"]["equal_cohort_macro"]
    paired = {}
    for metric, point_key in (
        ("liver", "liver_auroc"),
        ("non_liver", "non_liver_auroc"),
        ("gap", "liver_minus_non_liver"),
    ):
        draws = (
            raw_samples["sr1482_sibling_exposed_sensitivity"][metric]
            - raw_samples["confirmatory_family_naive"][metric]
        )
        paired[f"sibling_exposed_minus_family_naive_{metric}"] = {
            "delta": float(sensitivity[point_key] - confirmatory[point_key]),
            "ci95": cpht._interval(draws),  # noqa: SLF001
            "shared_cohort_organ_kras_bootstrap": True,
        }
    return {
        **analyses,
        "paired_sensitivity_contrasts": paired,
        "bootstrap_method": (
            "patient resampling within cohort x liver/non-liver x KRAS; "
            "indices shared across scorers; equal cohort weights"
        ),
        "individual_non_liver_organs": "descriptive inherited E2d analysis; not re-modelled here",
    }


def analyse_patient_scores(
    patient_scores: dict[str, dict[str, pd.DataFrame]],
    primary_scores: dict[tuple[str, str], pd.DataFrame],
    dual_rih_patients: set[str],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    enforce_live_census: bool = True,
) -> dict[str, Any]:
    """Pure final-v8 analysis from already aggregated patient-score tables."""

    if set(patient_scores) != set(ALL_SCORERS):
        raise ContractError(
            f"Scorer roster is {sorted(patient_scores)}, expected {sorted(ALL_SCORERS)}"
        )
    for scorer in ALL_SCORERS:
        if set(patient_scores[scorer]) != set(MET_TARGETS):
            raise ContractError(f"{scorer} does not cover both metastatic targets")
    if len(dual_rih_patients) != (
        EXPECTED_DUAL_RIH if enforce_live_census else len(dual_rih_patients)
    ):
        raise ContractError("RIH dual-role patient roster changed")

    # Align every matrix scorer before any shared bootstrap.  RIH's matrix and
    # all model contrasts use the 77-patient leakage-free roster.  The 85-row
    # family-naive result is retained separately below.
    rih_all = patient_scores["heldout_rih"]["rih_m"]
    rih_common_ids = set(rih_all["patient_id"].astype(str)) - set(dual_rih_patients)
    sr_ids = set(patient_scores["heldout_surgen"]["sr1482_m"]["patient_id"].astype(str))
    if enforce_live_census and (
        len(rih_all) != 85 or len(rih_common_ids) != EXPECTED_RIH_COMMON or len(sr_ids) != 74
    ):
        raise ContractError("Live E2-MET target census is not 85/77/74")
    aligned_by_target = {
        "rih_m": _align_frames(
            {scorer: patient_scores[scorer]["rih_m"] for scorer in ALL_SCORERS},
            rih_common_ids,
        ),
        "sr1482_m": _align_frames(
            {scorer: patient_scores[scorer]["sr1482_m"] for scorer in ALL_SCORERS},
            sr_ids,
        ),
    }

    # Keep the confirmatory bootstrap stream independent of the size/order of
    # the exploratory model matrix. Adding a descriptive scorer must not move a
    # primary interval merely by consuming earlier pseudorandom draws.
    matrix_rng = np.random.default_rng(bootstrap_seed)
    heldout_matrix: dict[str, Any] = {}
    allconv: dict[str, Any] = {}
    for target in MET_TARGETS:
        base = aligned_by_target[target][HELDOUT_SCORERS[0]]
        indices = _shared_stratified_indices(base["label"].to_numpy(), n_bootstrap, matrix_rng)
        metrics: dict[str, Any] = {}
        samples: dict[str, dict[str, np.ndarray]] = {}
        for scorer in ALL_SCORERS:
            block, sample = _metric_block(aligned_by_target[target][scorer], indices)
            block["exposure"] = exposure_class(scorer, target)
            metrics[scorer] = block
            samples[scorer] = sample
        heldout_metrics = {scorer: metrics[scorer] for scorer in HELDOUT_SCORERS}
        heldout_matrix[target] = {
            "n_common": int(len(base)),
            "metrics": heldout_metrics,
            "paired_auroc_contrasts": _paired_contrasts(
                heldout_metrics, {s: samples[s] for s in HELDOUT_SCORERS}
            ),
        }
        allconv[target] = {
            **metrics[ALL_CONVENTIONAL_SCORER],
            "family_naive_comparator": ("heldout_rih" if target == "rih_m" else "heldout_surgen"),
        }
        comparator = allconv[target]["family_naive_comparator"]
        draws = samples[ALL_CONVENTIONAL_SCORER]["auroc"] - samples[comparator]["auroc"]
        allconv[target]["paired_minus_family_naive"] = {
            "delta_auroc": float(
                metrics[ALL_CONVENTIONAL_SCORER]["auroc"] - metrics[comparator]["auroc"]
            ),
            "delta_auroc_ci95": cpht._interval(draws),  # noqa: SLF001
            "common_patients": int(len(base)),
        }
    # Standalone family-naive metastatic performance: RIH legitimately uses
    # all 85 here; SR1482 uses all 74.  A separate index tensor is required for
    # the 85-patient RIH estimate.
    confirmatory_frames = {
        "rih_m": rih_all.sort_values("patient_id", kind="stable").reset_index(drop=True),
        "sr1482_m": patient_scores["heldout_surgen"]["sr1482_m"]
        .sort_values("patient_id", kind="stable")
        .reset_index(drop=True),
    }
    confirmatory: dict[str, Any] = {}
    confirmatory_samples: dict[str, dict[str, np.ndarray]] = {}
    confirmatory_rng = np.random.default_rng(bootstrap_seed)
    for target in MET_TARGETS:
        frame = confirmatory_frames[target]
        indices = _shared_stratified_indices(
            frame["label"].to_numpy(), n_bootstrap, confirmatory_rng
        )
        block, metric_samples = _metric_block(frame, indices)
        block["scorer"] = "heldout_rih" if target == "rih_m" else "heldout_surgen"
        block["exposure"] = "family-naive"
        confirmatory[target] = block
        confirmatory_samples[target] = metric_samples
    macro_draws = np.mean([confirmatory_samples[target]["auroc"] for target in MET_TARGETS], axis=0)
    macro_point = float(np.mean([confirmatory[target]["auroc"] for target in MET_TARGETS]))
    macro_ci = cpht._interval(macro_draws)  # noqa: SLF001
    conclusion = {
        "equal_cohort_metastatic_macro_auroc": macro_point,
        "macro_auroc_ci95": macro_ci,
        "both_target_points_above_0p5": bool(
            all(confirmatory[target]["auroc"] > 0.5 for target in MET_TARGETS)
        ),
        "macro_lower_bound_above_0p5": bool(macro_ci[0] > 0.5),
    }
    conclusion["claim_metastatic_transport"] = bool(
        conclusion["both_target_points_above_0p5"] and conclusion["macro_lower_bound_above_0p5"]
    )

    required_primary = set(ROLE_JOBS)
    if set(primary_scores) != required_primary:
        raise ContractError(
            f"Primary-role scorer roster is {sorted(primary_scores)}, "
            f"expected {sorted(required_primary)}"
        )
    role: dict[str, Any] = {}
    rih_primary = primary_scores[("heldout_rih", "rih_primary")]
    rih_primary = rih_primary[~rih_primary["patient_id"].isin(dual_rih_patients)]
    rih_met = aligned_by_target["rih_m"]["heldout_rih"]
    role["rih_family_naive"] = _role_contrast(
        rih_primary, rih_met, n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )
    for scorer, name in (
        ("heldout_surgen", "sr1482_family_naive"),
        ("heldout_sr1482", "sr1482_sibling_exposed"),
    ):
        role[name] = _role_contrast(
            primary_scores[(scorer, "sr1482_primary")],
            patient_scores[scorer]["sr1482_m"],
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
    role["all_conventional_role_delta"] = (
        "not estimated: target primaries entered model fitting; metastatic performance only"
    )

    organ = _organ_analysis(patient_scores, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
    return cast(
        dict[str, Any],
        _json_safe(
            {
                "schema_version": 1,
                "experiment": "E2-MET final-v8 primary-to-metastatic transfer",
                "confirmatory_family_naive": confirmatory,
                "confirmatory_conclusion": conclusion,
                "heldout_source_composition_matrix": {
                    "status": "complete eight-heldout-scorer x two-target matrix",
                    "complete_eight_model_matrix": True,
                    "scorers": list(HELDOUT_SCORERS),
                    "targets": heldout_matrix,
                    "rih_population_rule": (
                        "all matrix cells and every model contrast use the same 77 "
                        "patients after excluding eight dual-role patients"
                    ),
                },
                "all_conventional_deployment_sensitivity": allconv,
                "primary_to_metastatic_role_contrasts": role,
                "liver_non_liver": organ,
                "statistics": {
                    "n_bootstrap": n_bootstrap,
                    "bootstrap_seed": bootstrap_seed,
                    "sampling_unit": "patient",
                    "training_seeds_are_inferential_units": False,
                },
                "reading_rule": (
                    "No metastatic result selects a model. Family-naive mappings are "
                    "confirmatory; sibling and all-conventional estimates are declared "
                    "sensitivities; the eight-model matrix is descriptive."
                ),
            },
        ),
    )


def _load_patient_inputs(
    output_root: Path, contract: dict[str, Any]
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[tuple[str, str], pd.DataFrame]]:
    patient_scores: dict[str, dict[str, pd.DataFrame]] = {}
    primary_scores: dict[tuple[str, str], pd.DataFrame] = {}
    for scorer in ALL_SCORERS:
        patient_scores[scorer] = {}
        for target in MET_TARGETS:
            frames = [
                pd.read_parquet(_score_path(output_root, scorer, target, seed)) for seed in SEEDS
            ]
            patient_scores[scorer][target] = aggregate_patient_scores(
                frames,
                pd.read_csv(_manifest_path(output_root, target)),
                _outcome_manifest(contract, target),
                calibrator=contract["models"][scorer]["calibrator"],
            )
    for scorer, target in ROLE_JOBS:
        frames = [pd.read_parquet(_score_path(output_root, scorer, target, seed)) for seed in SEEDS]
        primary_scores[(scorer, target)] = aggregate_patient_scores(
            frames,
            pd.read_csv(_manifest_path(output_root, target)),
            _outcome_manifest(contract, target),
            calibrator=contract["models"][scorer]["calibrator"],
        )
    return patient_scores, primary_scores


def cmd_preflight(args: argparse.Namespace) -> None:
    models = resolve_models(args.family_root, args.sibling_root, args.all_conventional_root)
    manifests, _, dual = build_input_manifests()
    pack = cpht._validate_pack(args.pack_dir, args.feature_dir)  # noqa: SLF001
    print(
        f"READY: {len(models)} three-seed ensembles; "
        f"{len(HELDOUT_SCORERS)} heldout matrix scorers + all-conventional sensitivity"
    )
    print(
        f"  targets: RIH-M {manifests['rih_m'].patient_id.nunique()} patients; "
        f"SR1482-M {manifests['sr1482_m'].patient_id.nunique()} patients"
    )
    print(f"  RIH dual-role exclusions: {len(dual)}; paired-model population: 77")
    print(f"  UNIv1 pack: {pack['n_slides']} slides x {pack['feature_dim']}")


def cmd_manifest(args: argparse.Namespace) -> None:
    frames, _, _ = build_input_manifests()
    for target, frame in frames.items():
        _publish_csv(_manifest_path(args.output_root, target), frame)
    contract = build_contract(
        args.output_root,
        family_root=args.family_root,
        sibling_root=args.sibling_root,
        all_conventional_root=args.all_conventional_root,
        feature_dir=args.feature_dir,
        pack_dir=args.pack_dir,
    )
    _publish_json(_contract_path(args.output_root), contract)
    print(f"sealed four label-blind manifests and model contract under {args.output_root}")


def cmd_score(args: argparse.Namespace) -> None:
    contract = _load_contract(args.output_root)
    sealed_workers = int(contract["inference"]["environment"]["num_workers"])
    if args.num_workers != sealed_workers:
        raise ContractError(
            f"Governed inference num_workers is {sealed_workers}, got {args.num_workers}"
        )
    selected = score_jobs()
    if args.scorer:
        selected = [job for job in selected if job[0] == args.scorer]
    if args.target:
        selected = [job for job in selected if job[1] == args.target]
    if args.seed:
        selected = [job for job in selected if job[2] == args.seed]
    if not selected:
        raise ContractError("Score filters selected no governed job")
    for scorer, target, seed in selected:
        print(f"E2-MET score: {scorer}/{target}/seed{seed}", flush=True)
        _score_one(
            args.output_root,
            contract,
            scorer,
            target,
            seed,
            device=args.device,
            num_workers=args.num_workers,
        )
    if not (args.scorer or args.target or args.seed):
        seal_inference(args.output_root)


def cmd_seal(args: argparse.Namespace) -> None:
    seal = seal_inference(args.output_root)
    print(
        f"sealed {seal['score_artifact_count']} outcome-free score artifacts: "
        f"{_seal_path(args.output_root)}"
    )


def cmd_report(args: argparse.Namespace) -> None:
    verify_inference_seal(args.output_root)
    contract = _load_contract(args.output_root)
    governed_bootstrap = contract["statistics"]
    if args.n_bootstrap != int(governed_bootstrap["n_bootstrap"]) or args.bootstrap_seed != int(
        governed_bootstrap["bootstrap_seed"]
    ):
        raise ContractError(
            "Report bootstrap must use the sealed 10,000-draw/seed-20260817 contract"
        )
    patient_scores, primary_scores = _load_patient_inputs(args.output_root, contract)
    results = analyse_patient_scores(
        patient_scores,
        primary_scores,
        set(contract["rih_dual_role_patients"]),
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        enforce_live_census=True,
    )
    results_path = args.output_root / "analysis/results.json"
    patients_path = args.output_root / "analysis/patient_scores.parquet"
    receipt_path = args.output_root / "analysis/report_receipt.json"
    rows = []
    for scorer, targets in patient_scores.items():
        for target, frame in targets.items():
            rows.append(frame.assign(scorer=scorer, target=target))
    for (scorer, target), frame in primary_scores.items():
        rows.append(frame.assign(scorer=scorer, target=target))
    patient_table = pd.concat(rows, ignore_index=True)
    _publish_parquet_resumable(patients_path, patient_table)
    results["inputs"] = {
        "inference_seal": _artifact(_seal_path(args.output_root)),
        "contract": _artifact(_contract_path(args.output_root)),
        "patient_scores": _artifact(patients_path),
        "analysis_environment": contract["statistics"]["analysis_environment"],
    }
    _publish_json(results_path, results)
    _publish_json(
        receipt_path,
        {
            "schema_version": 1,
            "status": "analysis_complete",
            "inference_seal": _artifact(_seal_path(args.output_root)),
            "contract": _artifact(_contract_path(args.output_root)),
            "analysis_environment": contract["statistics"]["analysis_environment"],
            "outputs": {
                "patient_scores": _artifact(patients_path),
                "results": _artifact(results_path),
            },
        },
    )
    c = results["confirmatory_conclusion"]
    ci = c["macro_auroc_ci95"]
    print(
        f"E2-MET family-naive macro AUROC {c['equal_cohort_metastatic_macro_auroc']:.4f} "
        f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    )
    print(f"metastatic transport gate: {c['claim_metastatic_transport']}")
    print("complete 8-heldout-scorer x 2-target matrix: True")
    print(f"wrote {results_path}")


def _add_common_roots(parser: argparse.ArgumentParser, *, require_new: bool) -> None:
    parser.add_argument("--family-root", type=Path, default=DEFAULT_FAMILY_ROOT)
    parser.add_argument("--sibling-root", type=Path, required=require_new)
    parser.add_argument("--all-conventional-root", type=Path, required=require_new)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command, function in (("preflight", cmd_preflight), ("manifest", cmd_manifest)):
        child = sub.add_parser(command)
        _add_common_roots(child, require_new=True)
        child.set_defaults(func=function)
    child = sub.add_parser("score")
    child.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    child.add_argument("--scorer", choices=ALL_SCORERS)
    child.add_argument("--target", choices=tuple(TARGET_SOURCES))
    child.add_argument("--seed", type=int, choices=SEEDS)
    child.add_argument("--device", default="cuda", choices=("cuda",))
    child.add_argument("--num-workers", type=int, default=4)
    child.set_defaults(func=cmd_score)
    child = sub.add_parser("seal")
    child.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    child.set_defaults(func=cmd_seal)
    child = sub.add_parser("report")
    child.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    child.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    child.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    child.set_defaults(func=cmd_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
