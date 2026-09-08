#!/usr/bin/env python3
"""Additively certify the completed Aim-1 two-encoder campaign after a validator defect.

The frozen campaign controller completed every native Virchow2-CLS training
run, then rejected the result in its post-fit validator.  Its validator applied
three extraction expectations to the schema-v2 ``training_identity`` material
payload even though that schema intentionally contains only data, encoder,
splits, model, training, and a reduced platform section.

This tool does not train, retry, repair, or mutate any campaign artifact.  It
accepts exactly those three absent identity keys only after proving their values
from the immutable contracted launch command, the resolved root config, every
fold config, the exact packed feature store, and the recomputed native identity.
It then publishes a distinct, atomic ``recovery_v1`` evidence namespace.  It
never writes the stock scheduler, Virchow2 job-receipt, or terminal paths.

Production workflow::

    uv run python tools/aim1_tcga_surgen_two_encoder_recovery.py audit
    uv run python tools/aim1_tcga_surgen_two_encoder_recovery.py certify --apply
    uv run python tools/aim1_tcga_surgen_two_encoder_recovery.py verify

``plan`` and ``audit`` are read-only.  ``certify`` requires ``--apply`` and is
exactly once.  The production campaign is deliberately pinned to one absolute
root and to the exact frozen controller, test, training implementation,
contract, and deep-preflight bytes involved in the incident.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tools import aim1_tcga_surgen_two_encoder_campaign as campaign  # noqa: E402

ContractError = campaign.ContractError

SCHEMA_VERSION = 1
RECOVERY = "aim1_tcga_surgen_two_encoder_validator_recovery_v1"
DEFAULT_OUTPUT_ROOT = campaign.DEFAULT_OUTPUT_ROOT
RECOVERY_DIRNAME = "recovery_v1"
RECOVERY_STATUS = "complete_and_certified_via_bounded_erratum"

PINNED_BASE_ARTIFACTS = {
    "controller": {
        "path": REPO / "tools/aim1_tcga_surgen_two_encoder_campaign.py",
        "sha256": "4635042e76c3ad3f8fc9aea46e478881681d4a1dfeaef046e0968424d7e2bcc7",
        "size_bytes": 75_355,
    },
    "controller_test": {
        "path": REPO / "tests/test_aim1_tcga_surgen_two_encoder_campaign.py",
        "sha256": "9609411e1bff2feb7517a23931a1e3907eb4984bee90da5d7cdd95c750ae3a16",
        "size_bytes": 8_636,
    },
    "training_implementation": {
        "path": REPO / "src/oceanpath/workflows/training.py",
        "sha256": "a22cd0579533cbd43941d338a15d0eb6b972f8c440f4a622361ca22de8e939b6",
        "size_bytes": 68_350,
    },
    "base_contract": {
        "path": DEFAULT_OUTPUT_ROOT / "contract.json",
        "sha256": "0aa475fa30123b5f2e44c6d3b54de5ade5ebcadeb5a2aa08224558bdc41509d9",
        "size_bytes": 69_114,
    },
    "base_preflight": {
        "path": DEFAULT_OUTPUT_ROOT / "receipts/deep_preflight.json",
        "sha256": "b6643d123470030d7ea237d83bdcc7eabbf6446ddb3fa24f7430a357c2ed973d",
        "size_bytes": 786,
    },
}

IDENTITY_SCHEMA_VERSION = 2
IDENTITY_MATERIAL_SECTIONS = (
    "data",
    "encoder",
    "model",
    "platform",
    "splits",
    "training",
)
BOUNDED_ABSENT_IDENTITY_KEYS = (
    "extraction.coords_dir",
    "extraction.coords_subdir",
    "extraction.patch_size",
)
BOUNDED_EXTRACTION_VALUES = {
    "extraction.coords_dir": "20x_224px_0px_overlap_mpp0.5",
    "extraction.coords_subdir": "20x_224px_0px_overlap_mpp0.5",
    "extraction.patch_size": 224,
}

FIT_ACCOUNTING = {
    "adopted_oof_fits": 25,
    "new_oof_fits": 25,
    "new_refits": 10,
    "new_fits": 35,
    "physical_new_fits": 35,
    "operational_lineage_fits": 60,
    "recovery_new_fits": 0,
    "hidden_fits": 0,
}
EXECUTION_ACCOUNTING = {
    "job_count": 10,
    "attempts_per_job": 1,
    "total_attempts": 10,
    "retries": 0,
}


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _canonical_sha256(value: Any) -> str:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return campaign._artifact(path)


def _logical_artifact(physical_path: Path, logical_path: Path) -> dict[str, Any]:
    observed = _artifact(physical_path)
    return {
        "path": str(logical_path.resolve(strict=False)),
        "sha256": observed["sha256"],
        "size_bytes": observed["size_bytes"],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return campaign._read_json(path)


def recovery_dir(root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return Path(root) / RECOVERY_DIRNAME


def erratum_contract_path(root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return recovery_dir(root) / "contract_erratum.json"


def recovered_job_receipt_path(root: Path, seed: int) -> Path:
    return recovery_dir(root) / f"receipts/jobs/virchow2_full/seed{seed}.json"


def scheduler_recovery_path(root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return recovery_dir(root) / "receipts/scheduler_recovery.json"


def adjudication_path(root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return recovery_dir(root) / "receipts/validator_adjudication.json"


def recovered_terminal_path(root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return recovery_dir(root) / "receipts/training_complete_recovered.json"


def recovery_write_inventory(root: Path = DEFAULT_OUTPUT_ROOT) -> tuple[Path, ...]:
    return (
        erratum_contract_path(root),
        *(recovered_job_receipt_path(root, seed) for seed in campaign.SEEDS),
        scheduler_recovery_path(root),
        adjudication_path(root),
        recovered_terminal_path(root),
    )


def _assert_production_root(root: Path) -> Path:
    candidate = campaign.assert_safe_output_root(Path(root))
    if candidate.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise ContractError(
            "Recovery is pinned to the single affected production root; "
            f"expected={DEFAULT_OUTPUT_ROOT}, observed={candidate}"
        )
    return candidate


def _expected_identity(path: Path, expected_sha256: str, expected_size: int) -> dict[str, Any]:
    return campaign._expected_identity(path, expected_sha256, expected_size)


def _pin_base(root: Path) -> dict[str, dict[str, Any]]:
    root = _assert_production_root(root)
    identities: dict[str, dict[str, Any]] = {}
    for name, spec in PINNED_BASE_ARTIFACTS.items():
        expected_path = Path(spec["path"])
        if name in {"base_contract", "base_preflight"}:
            relative = (
                "contract.json" if name == "base_contract" else "receipts/deep_preflight.json"
            )
            expected_path = root / relative
        identities[name] = _expected_identity(
            expected_path,
            str(spec["sha256"]),
            int(spec["size_bytes"]),
        )
    campaign.validate_contract(root, deep=False)
    preflight = campaign._validate_preflight(root)
    if preflight.get("status") != "deep_preflight_passed":
        raise ContractError("Pinned campaign preflight is not passed")
    return identities


def _material_mismatches(
    config: Mapping[str, Any], expectations: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        dotted: {
            "expected": expected,
            "observed": campaign._nested(config, dotted),
        }
        for dotted, expected in expectations.items()
        if campaign._nested(config, dotted) != expected
    }


def _validate_bounded_identity_material(
    material: Mapping[str, Any], root: Path, seed: int
) -> dict[str, Any]:
    if len(material) != len(IDENTITY_MATERIAL_SECTIONS) or set(material) != set(
        IDENTITY_MATERIAL_SECTIONS
    ):
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: schema-v2 material sections drifted; "
            f"expected={IDENTITY_MATERIAL_SECTIONS}, "
            f"observed={tuple(sorted(material.keys()))}"
        )
    expectations = campaign._material_expectations(
        root,
        seed,
        encoder="virchow2_cls",
        skip_finalize=False,
    )
    mismatches = _material_mismatches(material, expectations)
    missing = tuple(
        sorted(
            dotted for dotted, mismatch in mismatches.items() if mismatch["observed"] == "<missing>"
        )
    )
    other = {key: value for key, value in mismatches.items() if key not in missing}
    if missing != BOUNDED_ABSENT_IDENTITY_KEYS or other:
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: bounded identity exception is not exact; "
            f"missing={missing}, other={other}"
        )
    observed_values = {key: expectations[key] for key in missing}
    if observed_values != BOUNDED_EXTRACTION_VALUES:
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: bounded extraction values drifted: {observed_values}"
        )
    return {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "material_sections": list(IDENTITY_MATERIAL_SECTIONS),
        "accepted_absent_keys_only": list(missing),
        "values_proven_outside_reduced_identity": observed_values,
        "other_material_mismatches": 0,
    }


def _validate_full_material_config(
    config: Mapping[str, Any], root: Path, seed: int, *, context: str
) -> dict[str, Any]:
    campaign._validate_material_config(
        config,
        root,
        seed,
        encoder="virchow2_cls",
        skip_finalize=False,
        context=context,
    )
    extraction = {key: campaign._nested(config, key) for key in BOUNDED_ABSENT_IDENTITY_KEYS}
    if extraction != BOUNDED_EXTRACTION_VALUES:
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: {context} extraction proof drifted: {extraction}"
        )
    return extraction


def _contract_job(contract: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    jobs = contract.get("jobs")
    if not isinstance(jobs, list):
        raise ContractError("Pinned base contract has no job list")
    matches = [
        job
        for job in jobs
        if isinstance(job, Mapping)
        and job.get("kind") == "virchow2_full"
        and job.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ContractError(
            f"Expected one contracted Virchow2-CLS job for seed{seed}; got {len(matches)}"
        )
    return matches[0]


def _validate_command_proof(contract: Mapping[str, Any], root: Path, seed: int) -> dict[str, Any]:
    job = _contract_job(contract, seed)
    expected_job = next(
        candidate
        for candidate in campaign.build_training_jobs(root)
        if candidate["kind"] == "virchow2_full" and candidate["seed"] == seed
    )
    if job != expected_job:
        raise ContractError(f"Virchow2-CLS/seed{seed}: immutable contract job drifted")
    command = job.get("training_command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ContractError(f"Virchow2-CLS/seed{seed}: malformed training command")
    observed: dict[str, Any] = {}
    for dotted, expected in BOUNDED_EXTRACTION_VALUES.items():
        token = f"{dotted}={expected}"
        if command.count(token) != 1:
            raise ContractError(
                f"Virchow2-CLS/seed{seed}: contracted command does not contain "
                f"exactly one {token!r}"
            )
        observed[dotted] = expected
    return {
        "contract_job_sha256": _canonical_sha256(job),
        "training_command_sha256": hashlib.sha256("\0".join(command).encode()).hexdigest(),
        "training_command_argc": len(command),
        "extraction_overrides": observed,
    }


def _validate_request(
    root: Path, seed: int, contract_identity: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = campaign.request_path(root, "virchow2_full", seed)
    request = _read_json(path)
    created = request.get("created_utc")
    _parse_utc(created, context=f"Virchow2-CLS/seed{seed} request")
    expected = {
        "schema_version": campaign.SCHEMA_VERSION,
        "campaign": campaign.CAMPAIGN,
        "status": "requested",
        "created_utc": created,
        "kind": "virchow2_full",
        "encoder": "Virchow2-CLS",
        "seed": seed,
        "oof_fits": 5,
        "refits": 1,
        "total_fits": 6,
        "attempt": 1,
        "contract": dict(contract_identity),
        "manifest": _artifact(campaign.manifest_path(root)),
        "split": _artifact(campaign.split_dir(root) / "splits.parquet"),
        "output": str(campaign.run_dir(root, "virchow2_full", seed)),
    }
    if request != expected:
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: immutable request is not exact; "
            f"expected_keys={sorted(expected)}, observed_keys={sorted(request)}"
        )
    return request, _artifact(path)


def _parse_utc(value: Any, *, context: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ContractError(f"{context}: missing UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ContractError(f"{context}: invalid timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError(f"{context}: timestamp is not timezone-aware")
    return parsed.astimezone(dt.timezone.utc)


def _validate_native_status(directory: Path, seed: int) -> dict[str, Any]:
    status_path = directory / "_run/status.json"
    status = _read_json(status_path)
    if (
        status.get("stage") != "train_model"
        or status.get("status") != "completed"
        or not isinstance(status.get("duration_seconds"), (int, float))
        or not math.isfinite(float(status["duration_seconds"]))
        or float(status["duration_seconds"]) <= 0
    ):
        raise ContractError(f"Virchow2-CLS/seed{seed}: native study_train status is not completed")
    started = _parse_utc(status.get("started_at"), context=f"Virchow2-CLS/seed{seed} native status")
    finished = started + dt.timedelta(seconds=float(status["duration_seconds"]))
    return {
        "artifact": _artifact(status_path),
        "stage": "train_model",
        "status": "completed",
        "started_utc": started.isoformat(),
        "duration_seconds": float(status["duration_seconds"]),
        "derived_finished_utc": finished.isoformat(),
        "native_study_train_returncode": 0,
    }


def _validate_identity_and_configs(directory: Path, root: Path, seed: int) -> dict[str, Any]:
    from oceanpath.workflows.training import (
        training_identity_payload,
        training_run_fingerprint,
    )

    identity_path = directory / "training_identity.json"
    identity = _read_json(identity_path)
    if identity.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise ContractError(f"Virchow2-CLS/seed{seed}: training identity schema is not v2")
    payload = identity.get("payload")
    if not isinstance(payload, Mapping) or payload.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise ContractError(f"Virchow2-CLS/seed{seed}: malformed schema-v2 training payload")
    material = payload.get("material_config")
    if not isinstance(material, Mapping):
        raise ContractError(f"Virchow2-CLS/seed{seed}: missing material identity")
    bridge = _validate_bounded_identity_material(material, root, seed)

    root_config_path = directory / "config.yaml"
    root_cfg = OmegaConf.load(root_config_path)
    root_plain = OmegaConf.to_container(root_cfg, resolve=True)
    if not isinstance(root_plain, Mapping):
        raise ContractError(f"Virchow2-CLS/seed{seed}: malformed root config")
    root_extraction = _validate_full_material_config(
        root_plain,
        root,
        seed,
        context="resolved root config",
    )
    recomputed_payload = training_identity_payload(root_cfg)
    recomputed_fingerprint = training_run_fingerprint(root_cfg)
    if dict(payload) != recomputed_payload:
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: schema-v2 identity payload does not "
            "exactly recompute from the resolved root config and current inputs"
        )
    if identity.get("fingerprint") != recomputed_fingerprint:
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: native training fingerprint does not recompute"
        )

    fold_configs = []
    for fold in campaign.FOLDS:
        path = directory / f"fold_{fold}/config.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise ContractError(f"Virchow2-CLS/seed{seed}/fold{fold}: malformed fold config")
        extraction = _validate_full_material_config(
            config,
            root,
            seed,
            context=f"fold{fold} config",
        )
        if extraction != root_extraction:
            raise ContractError(f"Virchow2-CLS/seed{seed}/fold{fold}: extraction differs from root")
        fold_configs.append({"fold": fold, "config": _artifact(path), "extraction": extraction})

    evidence = payload.get("input_evidence")
    if not isinstance(evidence, Mapping):
        raise ContractError(f"Virchow2-CLS/seed{seed}: missing input evidence")
    expected_input = {
        "manifest_sha256": _artifact(campaign.manifest_path(root))["sha256"],
        "split_integrity_sha256": _artifact(campaign.split_dir(root) / ".integrity_hash")["sha256"],
        "feature_inventory_sha256": recomputed_payload["input_evidence"][
            "feature_inventory_sha256"
        ],
        "encoder_checkpoint_sha256": "default",
        "aggregator_checkpoint_sha256": "none",
    }
    if dict(evidence) != expected_input:
        raise ContractError(f"Virchow2-CLS/seed{seed}: exact native input evidence drifted")
    return {
        "training_identity": _artifact(identity_path),
        "fingerprint": recomputed_fingerprint,
        "payload_sha256": _canonical_sha256(payload),
        "bridge": bridge,
        "root_config": _artifact(root_config_path),
        "root_extraction": root_extraction,
        "fold_configs": fold_configs,
        "input_evidence": dict(evidence),
    }


def _validate_prediction_table(
    path: Path,
    expected: pd.DataFrame,
    *,
    seed: int,
    fold: int | None,
    role: str,
) -> dict[str, Any]:
    predictions = pd.read_parquet(path)
    required = {"slide_id", "label", "logit"}
    if required - set(predictions.columns):
        raise ContractError(f"Virchow2-CLS/seed{seed}/{role}: missing prediction columns")
    joined = predictions.merge(expected, on="slide_id", validate="one_to_one")
    logits = pd.to_numeric(predictions["logit"], errors="coerce")
    if (
        len(predictions) != len(expected)
        or predictions["slide_id"].astype(str).duplicated().any()
        or set(predictions["slide_id"].astype(str)) != set(expected["slide_id"].astype(str))
        or len(joined) != len(expected)
        or not pd.to_numeric(joined["label"])
        .astype(int)
        .eq(pd.to_numeric(joined["target_label"]).astype(int))
        .all()
        or not np.isfinite(logits).all()
    ):
        fold_text = "" if fold is None else f"/fold{fold}"
        raise ContractError(
            f"Virchow2-CLS/seed{seed}{fold_text}: invalid {role} roster/labels/logits"
        )
    return {
        "artifact": _artifact(path),
        "rows": int(len(predictions)),
        "unique_slides": int(predictions["slide_id"].astype(str).nunique()),
        "finite_logits": True,
    }


def _validate_checkpoint_topology(
    directory: Path,
    completion: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    expected: set[Path] = {
        (directory / "final/best_fold/model.ckpt").resolve(),
        (directory / "final/refit/model.ckpt").resolve(),
    }
    fold_records = completion.get("fold_completions")
    if not isinstance(fold_records, list):
        raise ContractError(f"Virchow2-CLS/seed{seed}: malformed fold completions")
    for fold in campaign.FOLDS:
        fold_record = _read_json(directory / f"fold_{fold}/completion.json")
        artifacts = fold_record.get("artifacts")
        checkpoint = artifacts.get("best_checkpoint") if isinstance(artifacts, Mapping) else None
        relative = checkpoint.get("path") if isinstance(checkpoint, Mapping) else None
        if not isinstance(relative, str):
            raise ContractError(f"Virchow2-CLS/seed{seed}/fold{fold}: no completed checkpoint")
        path = (directory / f"fold_{fold}" / relative).resolve()
        fold_root = (directory / f"fold_{fold}").resolve()
        try:
            path.relative_to(fold_root)
        except ValueError as exc:
            raise ContractError(
                f"Virchow2-CLS/seed{seed}/fold{fold}: checkpoint escapes fold"
            ) from exc
        expected.add(path)
        last = (directory / f"fold_{fold}/checkpoints/last.ckpt").resolve()
        if not last.is_file() or last.is_symlink() or last.stat().st_size <= 0:
            raise ContractError(
                f"Virchow2-CLS/seed{seed}/fold{fold}: missing/symlinked last checkpoint"
            )
        expected.add(last)
    observed = {path.resolve() for path in directory.rglob("*.ckpt")}
    if observed != expected:
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: hidden/missing checkpoint topology; "
            f"expected={sorted(map(str, expected))}, observed={sorted(map(str, observed))}"
        )
    return {
        "fold_fits": 5,
        "fold_checkpoint_artifacts": 10,
        "refit_fits": 1,
        "refit_checkpoint_artifacts": 1,
        "derived_best_fold_checkpoint_copies": 1,
        "total_checkpoint_artifacts": 12,
        "all_checkpoint_artifacts": [_artifact(path) for path in sorted(observed, key=str)],
    }


def _validate_v2_native(root: Path, seed: int) -> dict[str, Any]:
    from oceanpath.workflows.training import validate_training_run_dir

    directory = campaign.run_dir(root, "virchow2_full", seed)
    if not directory.is_dir() or directory.is_symlink():
        raise ContractError(f"Missing/symlinked Virchow2-CLS native output: {directory}")
    campaign._reject_symlinks_below(directory)
    completion = validate_training_run_dir(
        directory,
        require_test_predictions=True,
    )
    if (
        completion.get("status") != "completed"
        or completion.get("n_folds") != 5
        or completion.get("skip_finalize") is not False
        or [item.get("path") for item in completion.get("fold_completions", [])]
        != [f"fold_{fold}/completion.json" for fold in campaign.FOLDS]
    ):
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: native completion is not exact five-fold plus finalization"
        )
    identity = _validate_identity_and_configs(directory, root, seed)
    if identity["fingerprint"] != completion.get("training_fingerprint"):
        raise ContractError(f"Virchow2-CLS/seed{seed}: completion fingerprint mismatch")

    manifest = pd.read_csv(campaign.manifest_path(root), low_memory=False)
    fold_column = pd.to_numeric(manifest["k_fold"], errors="raise").astype(int)
    prediction_evidence = []
    for fold in campaign.FOLDS:
        for role, mask in {
            "test": fold_column.eq(fold),
            "val": pd.to_numeric(manifest[f"val_fold_{fold}"], errors="raise").astype(int).eq(1),
        }.items():
            expected = manifest.loc[mask, ["slide_id", "target_label"]]
            prediction_evidence.append(
                {
                    "fold": fold,
                    "role": role,
                    **_validate_prediction_table(
                        directory / f"fold_{fold}/preds_{role}.parquet",
                        expected,
                        seed=seed,
                        fold=fold,
                        role=role,
                    ),
                }
            )

    oof_path = directory / "oof_predictions.parquet"
    oof = pd.read_parquet(oof_path)
    required_oof = {"slide_id", "label", "logit", "fold"}
    if required_oof - set(oof.columns):
        raise ContractError(f"Virchow2-CLS/seed{seed}: malformed OOF table")
    check = oof.merge(
        manifest[["slide_id", "target_label", "k_fold"]],
        on="slide_id",
        validate="one_to_one",
    )
    if (
        len(oof) != len(manifest)
        or oof["slide_id"].astype(str).duplicated().any()
        or set(oof["slide_id"].astype(str)) != set(manifest["slide_id"].astype(str))
        or set(pd.to_numeric(oof["fold"], errors="raise").astype(int)) != set(campaign.FOLDS)
        or not np.isfinite(pd.to_numeric(oof["logit"], errors="coerce")).all()
        or not pd.to_numeric(check["label"])
        .astype(int)
        .eq(pd.to_numeric(check["target_label"]).astype(int))
        .all()
        or not pd.to_numeric(check["fold"])
        .astype(int)
        .eq(pd.to_numeric(check["k_fold"]).astype(int))
        .all()
    ):
        raise ContractError(f"Virchow2-CLS/seed{seed}: invalid complete OOF roster/labels/logits")

    refit = campaign._validate_refit(
        directory,
        root,
        seed,
        encoder="virchow2_cls",
    )
    final_dir = directory / "final"
    allowed = {"best_fold", "refit", "finalize_summary.json"}
    observed_final = {path.name for path in final_dir.iterdir()}
    if observed_final != allowed:
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: unexpected final strategies: {sorted(observed_final)}"
        )
    if list(directory.rglob("*ensemble*")):
        raise ContractError(f"Virchow2-CLS/seed{seed}: forbidden ensemble artifact")
    fold_dirs = {
        path.name for path in directory.iterdir() if path.is_dir() and path.name.startswith("fold_")
    }
    if fold_dirs != {f"fold_{fold}" for fold in campaign.FOLDS}:
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: hidden/missing fold output: {sorted(fold_dirs)}"
        )
    checkpoints = _validate_checkpoint_topology(
        directory,
        completion,
        seed=seed,
    )
    campaign._strict_scan_json_tree(directory)
    status = _validate_native_status(directory, seed)
    return {
        "directory": str(directory.resolve()),
        "training_completion": _artifact(directory / "training_completion.json"),
        "training_identity": identity,
        "oof_predictions": _artifact(oof_path),
        "oof_rows": int(len(oof)),
        "cv_summary": _artifact(directory / "cv_summary.json"),
        "fold_prediction_tables": prediction_evidence,
        "refit": refit,
        "finalize_summary": _artifact(final_dir / "finalize_summary.json"),
        "best_fold_checkpoint": _artifact(final_dir / "best_fold/model.ckpt"),
        "checkpoint_topology": checkpoints,
        "native_status": status,
        "oof_fits": 5,
        "refits": 1,
        "total_fits": 6,
        "hidden_fits": 0,
    }


def _reproduce_original_defect(root: Path, seed: int) -> dict[str, Any]:
    expectations = campaign._material_expectations(
        root,
        seed,
        encoder="virchow2_cls",
        skip_finalize=False,
    )
    identity = _read_json(campaign.run_dir(root, "virchow2_full", seed) / "training_identity.json")
    material = (identity.get("payload") or {}).get("material_config")
    if not isinstance(material, Mapping):
        raise ContractError(f"Virchow2-CLS/seed{seed}: missing material identity")
    mismatch = _material_mismatches(material, expectations)
    expected_mismatch = {
        key: {"expected": BOUNDED_EXTRACTION_VALUES[key], "observed": "<missing>"}
        for key in expectations
        if key in BOUNDED_ABSENT_IDENTITY_KEYS
    }
    if mismatch != expected_mismatch:
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: defect reproduction is not bounded: {mismatch}"
        )
    expected_message = (
        f"virchow2_cls/seed{seed}: training identity material recipe mismatch: {expected_mismatch}"
    )
    try:
        campaign._native_validation(root, "virchow2_full", seed)
    except campaign.ContractError as exc:
        if str(exc) != expected_message:
            raise ContractError(
                f"Virchow2-CLS/seed{seed}: frozen validator failed differently: {exc}"
            ) from exc
    else:
        raise ContractError(
            f"Virchow2-CLS/seed{seed}: frozen validator defect no longer reproduces"
        )
    return {
        "seed": seed,
        "affected_validator": "_validate_oof -> _validate_material_config",
        "failure_phase": "post_fit_native_validator",
        "native_study_train_returncode": 0,
        "original_wrapper_returncode": 1,
        "wrapper_returncode_evidence": (
            "deterministic replay of the pinned uncaught post-fit ContractError"
        ),
        "exception_type": "ContractError",
        "exception_sha256": hashlib.sha256(expected_message.encode()).hexdigest(),
        "missing_keys_only": list(BOUNDED_ABSENT_IDENTITY_KEYS),
        "other_mismatches": 0,
    }


def _validate_stock_univ1(root: Path, seed: int) -> dict[str, Any]:
    receipt = campaign._validate_job(root, "univ1_refit", seed)
    path = campaign.job_receipt_path(root, "univ1_refit", seed)
    request_path = campaign.request_path(root, "univ1_refit", seed)
    log_path = campaign.log_path(root, "univ1_refit", seed)
    request = _read_json(request_path)
    request_identity = _artifact(request_path)
    log_identity = _artifact(log_path)
    request_created = request.get("created_utc")
    _parse_utc(request_created, context=f"UNI-v1/seed{seed} request")
    expected_request = {
        "schema_version": campaign.SCHEMA_VERSION,
        "campaign": campaign.CAMPAIGN,
        "status": "requested",
        "created_utc": request_created,
        "kind": "univ1_refit",
        "encoder": "UNI-v1",
        "seed": seed,
        "oof_fits": 0,
        "refits": 1,
        "total_fits": 1,
        "attempt": 1,
        "contract": _artifact(campaign.contract_path(root)),
        "manifest": _artifact(campaign.manifest_path(root)),
        "split": _artifact(campaign.split_dir(root) / "splits.parquet"),
        "output": str(campaign.run_dir(root, "univ1_refit", seed)),
    }
    if request != expected_request:
        raise ContractError(
            f"UNI-v1/seed{seed}: immutable request is not exact; "
            f"expected_keys={sorted(expected_request)}, observed_keys={sorted(request)}"
        )
    if (
        receipt.get("status") != "completed"
        or receipt.get("attempt") != 1
        or receipt.get("oof_fits") != 0
        or receipt.get("refits") != 1
        or receipt.get("total_fits") != 1
    ):
        raise ContractError(f"UNI-v1/seed{seed}: stock refit receipt drifted")
    if receipt.get("request") != request_identity or receipt.get("log") != log_identity:
        raise ContractError(f"UNI-v1/seed{seed}: stock receipt request/log identity drifted")
    finished_utc = receipt.get("finished_utc")
    _parse_utc(finished_utc, context=f"UNI-v1/seed{seed} stock receipt")
    directory = campaign.run_dir(root, "univ1_refit", seed)
    checkpoints = list(directory.rglob("*.ckpt"))
    expected = [directory / "final/refit/model.ckpt"]
    if [path.resolve() for path in checkpoints] != [path.resolve() for path in expected]:
        raise ContractError(f"UNI-v1/seed{seed}: hidden/missing refit checkpoint topology")
    return {
        "seed": seed,
        "job_receipt": _artifact(path),
        "request": request_identity,
        "request_created_utc": request_created,
        "log": log_identity,
        "finished_utc": finished_utc,
        "artifacts": receipt.get("artifacts"),
        "oof_fits": 0,
        "refits": 1,
        "total_fits": 1,
        "attempt": 1,
        "retries": 0,
    }


def _validate_incident_roster(root: Path) -> dict[str, Any]:
    train_root = root / "train/e0"
    observed_encoders = {path.name for path in train_root.iterdir() if path.is_dir()}
    if observed_encoders != {"univ1", "virchow2_cls"}:
        raise ContractError(
            f"Unexpected/missing training encoder output: {sorted(observed_encoders)}"
        )
    expected_seeds = {f"seed{seed}" for seed in campaign.SEEDS}
    for encoder in ("univ1", "virchow2_cls"):
        directory = train_root / encoder
        observed = {path.name for path in directory.iterdir() if path.is_dir()}
        if observed != expected_seeds:
            raise ContractError(f"{encoder}: unexpected/missing seed output: {sorted(observed)}")

    for kind in ("univ1_refit", "virchow2_full"):
        expected = {f"seed{seed}.json" for seed in campaign.SEEDS}
        for category, accessor in (
            ("requests", campaign.request_path),
            ("logs", campaign.log_path),
        ):
            directory = root / category / kind
            observed = {path.name for path in directory.iterdir() if path.is_file()}
            expected_names = (
                expected
                if category == "requests"
                else {f"seed{seed}.log" for seed in campaign.SEEDS}
            )
            if observed != expected_names:
                raise ContractError(f"{kind}: unexpected/missing {category}: {sorted(observed)}")
            for seed in campaign.SEEDS:
                accessor(root, kind, seed)

    stock_v2_receipt_dir = root / "receipts/jobs/virchow2_full"
    stock_v2_receipts = (
        list(stock_v2_receipt_dir.iterdir()) if stock_v2_receipt_dir.is_dir() else []
    )
    forbidden = [
        campaign.scheduler_path(root),
        campaign.training_receipt_path(root),
        *(campaign.failure_path(root, "virchow2_full", seed) for seed in campaign.SEEDS),
        *(campaign.failure_path(root, "univ1_refit", seed) for seed in campaign.SEEDS),
    ]
    unexpected = [str(path) for path in forbidden if path.exists() or path.is_symlink()]
    unexpected.extend(str(path) for path in stock_v2_receipts)
    if unexpected:
        raise ContractError(
            "Incident recovery requires absent stock scheduler/terminal/V2 receipt/"
            f"failure artifacts; observed={unexpected}"
        )
    stock_uni = root / "receipts/jobs/univ1_refit"
    observed_uni = {path.name for path in stock_uni.iterdir() if path.is_file()}
    expected_uni = {f"seed{seed}.json" for seed in campaign.SEEDS}
    if observed_uni != expected_uni:
        raise ContractError(f"Unexpected/missing stock UNI receipts: {sorted(observed_uni)}")
    return {
        "train_encoder_directories": ["univ1", "virchow2_cls"],
        "seed_directories_per_encoder": sorted(expected_seeds),
        "stock_univ1_job_receipts": 5,
        "stock_virchow2_job_receipts": 0,
        "stock_scheduler_receipts": 0,
        "stock_terminal_receipts": 0,
        "failure_receipts": 0,
    }


def _raw_artifact_census(root: Path) -> dict[str, Any]:
    records = []
    recovery = recovery_dir(root).resolve(strict=False)
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(recovery)
        except ValueError:
            pass
        else:
            continue
        if path.is_symlink():
            raise ContractError(f"Raw campaign artifact is symlinked: {path}")
        if not path.is_file():
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": campaign._sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    if not records:
        raise ContractError("Raw campaign artifact census is empty")
    return {
        "root": str(root.resolve()),
        "excluded_prefix": f"{RECOVERY_DIRNAME}/",
        "artifact_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "tree_sha256": _canonical_sha256(records),
        "artifacts": records,
    }


def _iter_artifact_identities(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            yield value
        else:
            for child in value.values():
                yield from _iter_artifact_identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_artifact_identities(child)


def _assert_evidence_matches_census(
    root: Path, evidence: Mapping[str, Any], census: Mapping[str, Any]
) -> None:
    lookup = {
        str((root / record["path"]).resolve()): record
        for record in census.get("artifacts", [])
        if isinstance(record, Mapping)
    }
    recovery = recovery_dir(root).resolve(strict=False)
    for identity in _iter_artifact_identities(evidence):
        path = Path(str(identity["path"])).resolve(strict=False)
        try:
            path.relative_to(recovery)
        except ValueError:
            pass
        else:
            continue
        try:
            path.relative_to(root.resolve())
        except ValueError:
            continue
        record = lookup.get(str(path))
        if (
            record is None
            or record.get("sha256") != identity.get("sha256")
            or record.get("size_bytes") != identity.get("size_bytes")
        ):
            raise ContractError(f"Validated evidence changed before recovery publication: {path}")


def _concurrency_witness(
    v2_records: Sequence[Mapping[str, Any]],
    univ1_records: Sequence[Mapping[str, Any]],
    root: Path,
) -> dict[str, Any]:
    if len(v2_records) != 5 or len(univ1_records) != 5:
        raise ContractError("Concurrency witness requires five V2 and five UNI jobs")
    v2_intervals = []
    for record in v2_records:
        seed = int(record["seed"])
        request = _read_json(campaign.request_path(root, "virchow2_full", seed))
        status = record["native"]["native_status"]
        start = _parse_utc(status["started_utc"], context=f"V2 seed{seed} start")
        finish = _parse_utc(status["derived_finished_utc"], context=f"V2 seed{seed} finish")
        request_time = _parse_utc(request.get("created_utc"), context=f"V2 seed{seed} request")
        if request_time > start or finish <= start:
            raise ContractError(f"V2 seed{seed}: invalid request/native interval")
        v2_intervals.append((seed, start, finish))
    witness = max(start for _, start, _ in v2_intervals)
    active_v2 = [
        f"virchow2_full.seed{seed}"
        for seed, start, finish in v2_intervals
        if start <= witness < finish
    ]

    uni42 = next(record for record in univ1_records if record["seed"] == 42)
    uni_request = _read_json(campaign.request_path(root, "univ1_refit", 42))
    uni_start = _parse_utc(uni_request.get("created_utc"), context="UNI seed42 request")
    uni_finish = _parse_utc(uni42.get("finished_utc"), context="UNI seed42 stock receipt")
    if not (uni_start <= witness < uni_finish):
        raise ContractError(
            "Five simultaneous V2 native intervals do not overlap the initial UNI refit"
        )
    active = [*active_v2, "univ1_refit.seed42"]
    if len(active) != campaign.MAX_WORKERS:
        raise ContractError(f"Recovered execution does not prove exact six-way peak: {active}")
    return {
        "configured_max_workers": campaign.MAX_WORKERS,
        "observed_peak_parallel_workers": campaign.MAX_WORKERS,
        "witness_utc": witness.isoformat(),
        "active_jobs": sorted(active),
        "lower_bound_proof": "five native V2 intervals plus active UNI seed42",
        "upper_bound_proof": (
            "pinned ThreadPoolExecutor(max_workers=6) campaign controller and contract"
        ),
    }


def _audit_base(
    root: Path, *, deep_pack: bool, allow_published_recovery: bool = False
) -> dict[str, Any]:
    root = _assert_production_root(root)
    if not allow_published_recovery and (
        recovery_dir(root).exists() or recovery_dir(root).is_symlink()
    ):
        raise ContractError("Fresh recovery_v1 namespace required for pre-publication audit")
    pinned = _pin_base(root)
    roster = _validate_incident_roster(root)
    contract = _read_json(campaign.contract_path(root))
    pack = campaign._pack_identity("virchow2_cls", root, deep=deep_pack)
    if contract.get("feature_stores", {}).get("virchow2_cls") != pack:
        raise ContractError("Exact Virchow2-CLS pack identity/inventory drifted")

    adopted_oof = [campaign._adopted_chain(seed, deep=True) for seed in campaign.SEEDS]
    univ1 = [_validate_stock_univ1(root, seed) for seed in campaign.SEEDS]
    v2 = []
    defect = []
    for seed in campaign.SEEDS:
        request, request_identity = _validate_request(
            root,
            seed,
            pinned["base_contract"],
        )
        command = _validate_command_proof(contract, root, seed)
        native = _validate_v2_native(root, seed)
        defect_record = _reproduce_original_defect(root, seed)
        log = _artifact(campaign.log_path(root, "virchow2_full", seed))
        v2.append(
            {
                "seed": seed,
                "request": request_identity,
                "request_created_utc": request["created_utc"],
                "log": log,
                "contracted_command": command,
                "native": native,
                "attempt": 1,
                "retries": 0,
            }
        )
        defect.append(defect_record)

    concurrency = _concurrency_witness(v2, univ1, root)
    census = _raw_artifact_census(root)
    evidence = {
        "pinned": pinned,
        "roster": roster,
        "pack": pack,
        "adopted_oof": adopted_oof,
        "univ1": univ1,
        "v2": v2,
        "defect": defect,
        "concurrency": concurrency,
        "raw_census": census,
    }
    _assert_evidence_matches_census(root, evidence, census)
    return evidence


def _recovery_implementation() -> dict[str, dict[str, Any]]:
    return {
        "controller": _artifact(Path(__file__).resolve()),
        "controller_test": _artifact(REPO / "tests/test_aim1_tcga_surgen_two_encoder_recovery.py"),
    }


def _erratum_payload(evidence: Mapping[str, Any], *, created_utc: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "recovery": RECOVERY,
        "status": "bounded_validator_erratum_contract",
        "created_utc": created_utc,
        "base_campaign": campaign.CAMPAIGN,
        "base_contract": evidence["pinned"]["base_contract"],
        "base_preflight": evidence["pinned"]["base_preflight"],
        "frozen_implementation": {
            "controller": evidence["pinned"]["controller"],
            "controller_test": evidence["pinned"]["controller_test"],
            "training_implementation": evidence["pinned"]["training_implementation"],
        },
        "recovery_implementation": _recovery_implementation(),
        "defect": {
            "affected_kind": "virchow2_full",
            "affected_seeds": list(campaign.SEEDS),
            "failure_phase": "post_fit_native_validator",
            "training_identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "training_identity_material_sections": list(IDENTITY_MATERIAL_SECTIONS),
            "accepted_absent_keys_only": list(BOUNDED_ABSENT_IDENTITY_KEYS),
            "required_values": BOUNDED_EXTRACTION_VALUES,
            "other_missing_keys_allowed": 0,
            "other_material_mismatches_allowed": 0,
            "native_output_mismatches_allowed": 0,
            "cause": (
                "frozen campaign validator applied full resolved-config extraction "
                "expectations to a deliberately reduced schema-v2 identity payload"
            ),
        },
        "proof_requirements": [
            "immutable contracted study_train command",
            "resolved root config",
            "each of five fold configs",
            "exact Virchow2-CLS packed store and source inventory",
            "exact recomputation of schema-v2 payload and fingerprint",
            "native completion hashes and every held-out prediction roster/logit",
            "OOF roster, refit, finalization, and checkpoint topology",
        ],
        "mutation_policy": {
            "raw_campaign_writes": 0,
            "training_or_refit_runs": 0,
            "retries": 0,
            "stock_receipts_created": 0,
            "only_write_prefix": f"{RECOVERY_DIRNAME}/",
            "publication": "atomic_exactly_once_directory_rename",
        },
        "fit_accounting": FIT_ACCOUNTING,
        "execution_accounting": EXECUTION_ACCOUNTING,
    }


def _adjudication_payload(
    evidence: Mapping[str, Any],
    *,
    created_utc: str,
    erratum_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "recovery": RECOVERY,
        "status": "native_outputs_certified_by_bounded_validator_adjudication",
        "created_utc": created_utc,
        "base_contract": evidence["pinned"]["base_contract"],
        "base_preflight": evidence["pinned"]["base_preflight"],
        "erratum_contract": dict(erratum_identity),
        "pack_and_inventory": evidence["pack"],
        "incident_roster": evidence["roster"],
        "defect_reproduction": evidence["defect"],
        "virchow2_native_validations": evidence["v2"],
        "stock_univ1_validations": evidence["univ1"],
        "adopted_univ1_oof_validations": evidence["adopted_oof"],
        "raw_artifact_hash_census": {
            "before": evidence["raw_census"],
            "after_expected_identical": evidence["raw_census"],
            "publication_exclusion": f"{RECOVERY_DIRNAME}/",
        },
        "fit_accounting": FIT_ACCOUNTING,
        "execution_accounting": EXECUTION_ACCOUNTING,
        "concurrency": evidence["concurrency"],
        "adjudication_boundary": (
            "validation schema bridge only; model bytes, predictions, rosters, "
            "labels, logits, checkpoints, refits, and fit counts are unchanged"
        ),
    }


def _recovered_v2_job_payload(
    evidence: Mapping[str, Any],
    seed: int,
    *,
    created_utc: str,
    erratum_identity: Mapping[str, Any],
    adjudication_identity: Mapping[str, Any],
) -> dict[str, Any]:
    record = next(item for item in evidence["v2"] if item["seed"] == seed)
    defect = next(item for item in evidence["defect"] if item["seed"] == seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "recovery": RECOVERY,
        "base_campaign": campaign.CAMPAIGN,
        "status": "recovered_complete_via_bounded_erratum",
        "created_utc": created_utc,
        "kind": "virchow2_full",
        "encoder": "Virchow2-CLS",
        "seed": seed,
        "attempt": 1,
        "retries": 0,
        "oof_fits": 5,
        "refits": 1,
        "total_fits": 6,
        "hidden_fits": 0,
        "native_study_train_returncode": 0,
        "original_wrapper_returncode": 1,
        "failure_phase": "post_fit_native_validator",
        "returncode_adjudication": defect["wrapper_returncode_evidence"],
        "base_contract": evidence["pinned"]["base_contract"],
        "base_preflight": evidence["pinned"]["base_preflight"],
        "request": record["request"],
        "log": record["log"],
        "erratum_contract": dict(erratum_identity),
        "validator_adjudication": dict(adjudication_identity),
        "contracted_command": record["contracted_command"],
        "identity_schema_bridge": record["native"]["training_identity"]["bridge"],
        "artifacts": record["native"],
    }


def _scheduler_payload(
    evidence: Mapping[str, Any],
    recovered_receipts: Mapping[int, Mapping[str, Any]],
    *,
    created_utc: str,
    erratum_identity: Mapping[str, Any],
    adjudication_identity: Mapping[str, Any],
) -> dict[str, Any]:
    events = []
    for record in evidence["v2"]:
        seed = int(record["seed"])
        status = record["native"]["native_status"]
        events.append(
            {
                "job_id": f"aim1.e0.tcga_surgen.virchow2_full.seed{seed}",
                "kind": "virchow2_full",
                "seed": seed,
                "attempt": 1,
                "retries": 0,
                "request_created_utc": record["request_created_utc"],
                "native_started_utc": status["started_utc"],
                "native_derived_finished_utc": status["derived_finished_utc"],
                "native_study_train_returncode": 0,
                "original_wrapper_returncode": 1,
                "failure_phase": "post_fit_native_validator",
                "job_receipt": dict(recovered_receipts[seed]),
            }
        )
    for record in evidence["univ1"]:
        seed = int(record["seed"])
        events.append(
            {
                "job_id": f"aim1.e0.tcga_surgen.univ1_refit.seed{seed}",
                "kind": "univ1_refit",
                "seed": seed,
                "attempt": 1,
                "retries": 0,
                "request_created_utc": record["request_created_utc"],
                "wrapper_finished_utc": record["finished_utc"],
                "native_refit_returncode": 0,
                "original_wrapper_returncode": 0,
                "job_receipt": record["job_receipt"],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "recovery": RECOVERY,
        "status": "execution_recovered_without_rerun",
        "created_utc": created_utc,
        "base_contract": evidence["pinned"]["base_contract"],
        "base_preflight": evidence["pinned"]["base_preflight"],
        "erratum_contract": dict(erratum_identity),
        "validator_adjudication": dict(adjudication_identity),
        "configured_max_workers": 6,
        "observed_peak_parallel_workers": 6,
        "job_count": 10,
        "attempts_per_job": 1,
        "retry_count": 0,
        "physical_new_fits": 35,
        "operational_lineage_fits": 60,
        "events": sorted(events, key=lambda item: item["job_id"]),
        "concurrency_witness": evidence["concurrency"],
        "returncode_semantics": {
            "univ1_refit": "native refit rc0; frozen wrapper rc0",
            "virchow2_full": (
                "native study_train rc0; frozen wrapper rc1 only in uncaught post-fit validator"
            ),
        },
        "stock_scheduler_created": False,
    }


def _terminal_payload(
    evidence: Mapping[str, Any],
    recovered_receipts: Mapping[int, Mapping[str, Any]],
    *,
    created_utc: str,
    erratum_identity: Mapping[str, Any],
    adjudication_identity: Mapping[str, Any],
    scheduler_identity: Mapping[str, Any],
) -> dict[str, Any]:
    new_chains = []
    job_receipts = []
    for record in evidence["univ1"]:
        identity = record["job_receipt"]
        job_receipts.append(identity)
        new_chains.append(
            {
                "kind": "univ1_refit",
                "encoder": "UNI-v1",
                "seed": record["seed"],
                "oof_fits": 0,
                "refits": 1,
                "total_fits": 1,
                "receipt_namespace": "immutable_stock",
                "job_receipt": identity,
            }
        )
    for seed in campaign.SEEDS:
        identity = dict(recovered_receipts[seed])
        job_receipts.append(identity)
        new_chains.append(
            {
                "kind": "virchow2_full",
                "encoder": "Virchow2-CLS",
                "seed": seed,
                "oof_fits": 5,
                "refits": 1,
                "total_fits": 6,
                "receipt_namespace": RECOVERY_DIRNAME,
                "job_receipt": identity,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "recovery": RECOVERY,
        "base_campaign": campaign.CAMPAIGN,
        "status": RECOVERY_STATUS,
        "created_utc": created_utc,
        "base_contract": evidence["pinned"]["base_contract"],
        "base_preflight": evidence["pinned"]["base_preflight"],
        "erratum_contract": dict(erratum_identity),
        "scheduler_recovery": dict(scheduler_identity),
        "validator_adjudication": dict(adjudication_identity),
        "recovery_implementation": _recovery_implementation(),
        "source_population": {
            "name": "TCGA + SurGen primary",
            **campaign.EXPECTED_CENSUS,
            "manifest": _artifact(campaign.manifest_path(DEFAULT_OUTPUT_ROOT)),
            "splits": _artifact(campaign.split_dir(DEFAULT_OUTPUT_ROOT) / "splits.parquet"),
        },
        "seeds": list(campaign.SEEDS),
        "encoders": ["UNI-v1", "Virchow2-CLS"],
        "job_count": 10,
        "fit_accounting": FIT_ACCOUNTING,
        "execution_accounting": EXECUTION_ACCOUNTING,
        "concurrency": {
            "maximum": 6,
            "observed_peak": 6,
            "witness_utc": evidence["concurrency"]["witness_utc"],
        },
        "adopted_univ1_oof_chains": evidence["adopted_oof"],
        "stock_univ1_job_receipts": [record["job_receipt"] for record in evidence["univ1"]],
        "recovered_virchow2_job_receipts": [
            dict(recovered_receipts[seed]) for seed in campaign.SEEDS
        ],
        "new_chains": sorted(
            new_chains,
            key=lambda item: (item["kind"], item["seed"]),
        ),
        "new_job_receipts": sorted(
            job_receipts,
            key=lambda item: item["path"],
        ),
        "raw_artifact_hash_census": {
            "artifact_count": evidence["raw_census"]["artifact_count"],
            "total_size_bytes": evidence["raw_census"]["total_size_bytes"],
            "tree_sha256_before": evidence["raw_census"]["tree_sha256"],
            "tree_sha256_after": evidence["raw_census"]["tree_sha256"],
            "unchanged": True,
        },
        "certification_boundary": (
            "bounded schema-v2 validator erratum; no fit, refit, retry, score, "
            "calibration, selection, or raw artifact mutation"
        ),
        "stock_receipts_fabricated": False,
    }


def _write_staged_json(path: Path, payload: Mapping[str, Any]) -> None:
    campaign._write_json_once(path, payload)


def _materialize_recovery(
    root: Path,
    stage: Path,
    evidence: Mapping[str, Any],
    *,
    created_utc: str,
) -> None:
    logical = recovery_dir(root)
    erratum = _erratum_payload(evidence, created_utc=created_utc)
    staged_erratum = stage / "contract_erratum.json"
    _write_staged_json(staged_erratum, erratum)
    erratum_identity = _logical_artifact(
        staged_erratum,
        erratum_contract_path(root),
    )

    adjudication = _adjudication_payload(
        evidence,
        created_utc=created_utc,
        erratum_identity=erratum_identity,
    )
    staged_adjudication = stage / "receipts/validator_adjudication.json"
    _write_staged_json(staged_adjudication, adjudication)
    adjudication_identity = _logical_artifact(
        staged_adjudication,
        adjudication_path(root),
    )

    recovered_receipts = {}
    for seed in campaign.SEEDS:
        payload = _recovered_v2_job_payload(
            evidence,
            seed,
            created_utc=created_utc,
            erratum_identity=erratum_identity,
            adjudication_identity=adjudication_identity,
        )
        staged_path = stage / f"receipts/jobs/virchow2_full/seed{seed}.json"
        _write_staged_json(staged_path, payload)
        recovered_receipts[seed] = _logical_artifact(
            staged_path,
            recovered_job_receipt_path(root, seed),
        )

    scheduler = _scheduler_payload(
        evidence,
        recovered_receipts,
        created_utc=created_utc,
        erratum_identity=erratum_identity,
        adjudication_identity=adjudication_identity,
    )
    staged_scheduler = stage / "receipts/scheduler_recovery.json"
    _write_staged_json(staged_scheduler, scheduler)
    scheduler_identity = _logical_artifact(
        staged_scheduler,
        scheduler_recovery_path(root),
    )

    terminal = _terminal_payload(
        evidence,
        recovered_receipts,
        created_utc=created_utc,
        erratum_identity=erratum_identity,
        adjudication_identity=adjudication_identity,
        scheduler_identity=scheduler_identity,
    )
    _write_staged_json(
        stage / "receipts/training_complete_recovered.json",
        terminal,
    )
    expected = {path.relative_to(logical).as_posix() for path in recovery_write_inventory(root)}
    observed = {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}
    if observed != expected:
        raise ContractError(
            f"Recovery staged write roster drifted: expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )


def _publish_atomic(root: Path, evidence: Mapping[str, Any]) -> None:
    destination = recovery_dir(root)
    if destination.exists() or destination.is_symlink():
        raise ContractError(
            f"Recovery publication is exactly once; refusing existing {destination}"
        )
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{root.name}.recovery-v1-stage-",
            dir=root.parent,
        )
    )
    published = False
    try:
        _materialize_recovery(
            root,
            stage,
            evidence,
            created_utc=_utcnow(),
        )
        before = evidence["raw_census"]
        current = _raw_artifact_census(root)
        if current != before:
            raise ContractError("Raw campaign bytes changed during recovery certification")
        os.rename(stage, destination)
        published = True
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        after = _raw_artifact_census(root)
        if after != before:
            raise ContractError("Raw campaign bytes changed across additive recovery publication")
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


def _verify_recovery_files(root: Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    observed_paths = {
        path.relative_to(recovery_dir(root)).as_posix()
        for path in recovery_dir(root).rglob("*")
        if path.is_file()
    }
    expected_paths = {
        path.relative_to(recovery_dir(root)).as_posix() for path in recovery_write_inventory(root)
    }
    if observed_paths != expected_paths:
        raise ContractError(
            f"Recovery output roster drifted: expected={sorted(expected_paths)}, "
            f"observed={sorted(observed_paths)}"
        )
    campaign._reject_symlinks_below(recovery_dir(root))
    terminal = _read_json(recovered_terminal_path(root))
    created = terminal.get("created_utc")
    _parse_utc(created, context="recovered terminal")

    erratum = _read_json(erratum_contract_path(root))
    expected_erratum = _erratum_payload(evidence, created_utc=created)
    if erratum != expected_erratum:
        raise ContractError("Recovery erratum contract does not replay")
    erratum_identity = _artifact(erratum_contract_path(root))

    adjudication = _read_json(adjudication_path(root))
    expected_adjudication = _adjudication_payload(
        evidence,
        created_utc=created,
        erratum_identity=erratum_identity,
    )
    if adjudication != expected_adjudication:
        raise ContractError("Validator adjudication receipt does not replay")
    adjudication_identity = _artifact(adjudication_path(root))

    receipt_identities = {}
    for seed in campaign.SEEDS:
        receipt = _read_json(recovered_job_receipt_path(root, seed))
        expected = _recovered_v2_job_payload(
            evidence,
            seed,
            created_utc=created,
            erratum_identity=erratum_identity,
            adjudication_identity=adjudication_identity,
        )
        if receipt != expected:
            raise ContractError(f"Recovered Virchow2-CLS seed{seed} receipt does not replay")
        receipt_identities[seed] = _artifact(recovered_job_receipt_path(root, seed))

    scheduler = _read_json(scheduler_recovery_path(root))
    expected_scheduler = _scheduler_payload(
        evidence,
        receipt_identities,
        created_utc=created,
        erratum_identity=erratum_identity,
        adjudication_identity=adjudication_identity,
    )
    if scheduler != expected_scheduler:
        raise ContractError("Recovery scheduler receipt does not replay")
    scheduler_identity = _artifact(scheduler_recovery_path(root))

    expected_terminal = _terminal_payload(
        evidence,
        receipt_identities,
        created_utc=created,
        erratum_identity=erratum_identity,
        adjudication_identity=adjudication_identity,
        scheduler_identity=scheduler_identity,
    )
    if terminal != expected_terminal:
        raise ContractError("Recovered terminal receipt does not replay")
    return terminal


def validate_recovered_terminal(
    root: Path = DEFAULT_OUTPUT_ROOT, *, deep_pack: bool = True
) -> dict[str, Any]:
    """Replay the bounded recovery and return its authenticated terminal receipt."""

    root = _assert_production_root(root)
    if not recovery_dir(root).is_dir() or recovery_dir(root).is_symlink():
        raise ContractError("Missing or symlinked recovery_v1 namespace")
    evidence = _audit_base(
        root,
        deep_pack=deep_pack,
        allow_published_recovery=True,
    )
    return _verify_recovery_files(root, evidence)


def cmd_plan(args: argparse.Namespace) -> None:
    root = _assert_production_root(args.output_root)
    payload = {
        "status": "PLAN_ONLY_NO_WRITES",
        "recovery": RECOVERY,
        "output_root": str(root),
        "recovery_root": str(recovery_dir(root)),
        "write_inventory": [str(path) for path in recovery_write_inventory(root)],
        "accepted_absent_identity_keys_only": list(BOUNDED_ABSENT_IDENTITY_KEYS),
        "fit_accounting": FIT_ACCOUNTING,
        "execution_accounting": EXECUTION_ACCOUNTING,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def cmd_audit(args: argparse.Namespace) -> None:
    root = _assert_production_root(args.output_root)
    evidence = _audit_base(root, deep_pack=True)
    print(
        json.dumps(
            {
                "status": "PASS_READ_ONLY_RECOVERY_AUDIT",
                "recovery": RECOVERY,
                "fit_accounting": FIT_ACCOUNTING,
                "execution_accounting": EXECUTION_ACCOUNTING,
                "concurrency": evidence["concurrency"],
                "raw_artifact_hash_census": {
                    key: evidence["raw_census"][key]
                    for key in ("artifact_count", "total_size_bytes", "tree_sha256")
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def cmd_certify(args: argparse.Namespace) -> None:
    if not args.apply:
        raise ContractError("certify requires --apply")
    root = _assert_production_root(args.output_root)
    evidence = _audit_base(root, deep_pack=True)
    _publish_atomic(root, evidence)
    print(
        json.dumps(
            {
                "status": RECOVERY_STATUS,
                "terminal": _artifact(recovered_terminal_path(root)),
                "fit_accounting": FIT_ACCOUNTING,
                "execution_accounting": EXECUTION_ACCOUNTING,
                "concurrency": {"maximum": 6, "observed_peak": 6},
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def cmd_verify(args: argparse.Namespace) -> None:
    root = _assert_production_root(args.output_root)
    terminal = validate_recovered_terminal(root, deep_pack=True)
    print(
        json.dumps(
            {
                "status": "PASS",
                "terminal_status": terminal["status"],
                "terminal": _artifact(recovered_terminal_path(root)),
                "fit_accounting": terminal["fit_accounting"],
                "execution_accounting": terminal["execution_accounting"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def cmd_status(args: argparse.Namespace) -> None:
    root = _assert_production_root(args.output_root)
    states = {}
    for seed in campaign.SEEDS:
        directory = campaign.run_dir(root, "virchow2_full", seed)
        status_path = directory / "_run/status.json"
        states[str(seed)] = (
            _read_json(status_path).get("status") if status_path.is_file() else "incomplete"
        )
    terminal = recovered_terminal_path(root)
    print(
        json.dumps(
            {
                "status": "RECOVERY_STATUS_READ_ONLY",
                "virchow2_native_runs": states,
                "recovery_published": terminal.is_file() and not terminal.is_symlink(),
                "terminal": str(terminal),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", parents=[common]).set_defaults(func=cmd_plan)
    commands.add_parser("audit", parents=[common]).set_defaults(func=cmd_audit)
    certify = commands.add_parser("certify", parents=[common])
    certify.add_argument("--apply", action="store_true")
    certify.set_defaults(func=cmd_certify)
    commands.add_parser("verify", parents=[common]).set_defaults(func=cmd_verify)
    commands.add_parser("status", parents=[common]).set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
