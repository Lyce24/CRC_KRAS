#!/usr/bin/env python3
"""Governed phase-2 analysis for the Aim-1 four-cohort five-seed OOF study.

This tool is additive to FINAL-v10 and to the phase-1 training campaign.  It
never trains or refits a model.  Its formal estimand is the mean native slide
logit within patient and model seed, followed by the mean patient logit across
seeds 42--46.  Patients are the only inferential units.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import functools
import hashlib
import io
import json
import os
import sys
import threading
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import aim1_source_cohort_five_seed_campaign as training  # noqa: E402

SCHEMA_VERSION = 1
EXPERIMENT = "aim1_primary_cohort_oof_5seed_analysis"
SEEDS = tuple(training.SEEDS)
ARMS = tuple(training.ARM_SPECS)
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260824
ANALYSIS_FILES = (
    "contract.json",
    "patient_native_logits.parquet",
    "results.json",
    "bootstrap_distributions.npz",
    "analysis_completion_receipt.json",
)
DEFAULT_PARENT_FINAL_V10_RECEIPT = REPO / "reports/final_v10/report_bundle_receipt.json"
EXPECTED_PARENT_FINAL_V10_RECEIPT_SHA256 = (
    "7d3ac2b82f71dd956c0e2b5ad7c951474ce7310c63c421f1a347b5e0c55f43e2"
)
EXPECTED_PARENT_FINAL_V10_STATUS = "SEALED_COMPLETED_RESULTS_WITH_DECLARED_NOT_RUN_ARM"
ANALYSIS_TEST = REPO / "tests/test_aim1_source_cohort_five_seed_analysis.py"


class AnalysisError(RuntimeError):
    """Fail-closed violation of the governed phase-2 analysis contract."""


@dataclass(frozen=True)
class ContrastSpec:
    key: str
    larger_arm: str
    smaller_arm: str
    evaluation_arm: str
    expected_patients: int
    expected_mutant: int
    expected_wild_type: int


CONTRASTS = (
    ContrastSpec(
        "surgen_minus_sr386_on_sr386",
        "surgen_primary",
        "sr386_primary",
        "sr386_primary",
        413,
        147,
        266,
    ),
    ContrastSpec(
        "tcga_surgen_minus_surgen_on_surgen",
        "tcga_surgen_primary",
        "surgen_primary",
        "surgen_primary",
        737,
        294,
        443,
    ),
    ContrastSpec(
        "tcga_surgen_minus_tcga_on_tcga",
        "tcga_surgen_primary",
        "tcga_primary",
        "tcga_primary",
        502,
        207,
        295,
    ),
)
BOOTSTRAP_ARRAY_NAMES = tuple(
    sorted(
        [f"arm__{arm}__ensemble__{metric}" for arm in ARMS for metric in ("auroc", "auprc")]
        + [
            f"contrast__{spec.key}__{role}__{metric}"
            for spec in CONTRASTS
            for role in ("larger", "smaller", "delta")
            for metric in ("auroc", "auprc")
        ]
    )
)


@dataclass(frozen=True)
class SourceBundle:
    campaign_root: Path
    identities: dict[str, Any]
    manifests: dict[str, pd.DataFrame]
    oof_by_arm_seed: dict[tuple[str, int], pd.DataFrame]


@dataclass(frozen=True)
class AnalysisProduct:
    patient_scores: pd.DataFrame
    results: dict[str, Any]
    bootstraps: dict[str, np.ndarray]


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AnalysisError(f"Required regular artifact missing or symlinked: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _identity_for_bytes(path: Path, data: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=False)),
        "sha256": _sha256_bytes(data),
        "size_bytes": len(data),
    }


def _validate_artifact(identity: Mapping[str, Any], *, expected_path: Path | None = None) -> None:
    path = Path(str(identity.get("path", "")))
    if expected_path is not None and path != expected_path.resolve(strict=False):
        raise AnalysisError(
            f"Artifact path drifted: expected {expected_path.resolve(strict=False)}, got {path}"
        )
    if _artifact(path) != dict(identity):
        raise AnalysisError(f"Artifact identity drifted: {path}")


def _iter_artifacts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            yield value
        else:
            for child in value.values():
                yield from _iter_artifacts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_artifacts(child)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise AnalysisError(f"Non-finite JSON constant: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AnalysisError(f"Required JSON missing or symlinked: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON artifact must contain an object: {path}")
    return value


def _strict_read_json_tree(
    path: Path, *, _visited: set[Path] | None = None
) -> dict[str, Any]:
    """Reject ambiguous JSON throughout an identity-linked receipt tree."""
    path = Path(path)
    resolved = path.resolve(strict=False)
    visited = set() if _visited is None else _visited
    if resolved in visited:
        return _read_json(path)
    value = _read_json(path)
    visited.add(resolved)
    for identity in _iter_artifacts(value):
        child = Path(str(identity["path"]))
        if not child.is_absolute():
            child = path.parent / child
        if child.suffix.casefold() == ".json":
            _strict_read_json_tree(child, _visited=visited)
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AnalysisError("Analysis JSON is not finite and serializable") from exc


def _write_bytes_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary_inode: tuple[int, int] | None = None
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            observed = os.fstat(stream.fileno())
            temporary_inode = (observed.st_dev, observed.st_ino)
        try:
            os.link(temporary, path)
            linked = True
        except FileExistsError as exc:
            if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
                return
            raise AnalysisError(f"Refusing to replace a different sealed artifact: {path}") from exc
        _fsync_directory(path.parent)
    except BaseException:
        if linked and temporary_inode is not None:
            try:
                destination_stat = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if temporary_inode == (destination_stat.st_dev, destination_stat.st_ino):
                    with contextlib.suppress(FileNotFoundError):
                        path.unlink()
                    with contextlib.suppress(OSError):
                        _fsync_directory(path.parent)
        raise
    finally:
        if temporary_inode is not None:
            try:
                temporary_stat = temporary.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if temporary_inode == (temporary_stat.st_dev, temporary_stat.st_ino):
                    with contextlib.suppress(FileNotFoundError):
                        temporary.unlink()
                    with contextlib.suppress(OSError):
                        _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def analysis_root(campaign_root: Path) -> Path:
    return Path(campaign_root) / "analysis"


def _guard_roots(campaign_root: Path, output_root: Path | None = None) -> tuple[Path, Path]:
    campaign = training.assert_safe_output_root(Path(campaign_root))
    output = analysis_root(campaign) if output_root is None else Path(output_root)
    if not output.is_absolute():
        raise AnalysisError("Analysis output root must be absolute")
    lexical = output.absolute()
    resolved = output.resolve(strict=False)
    if lexical != resolved:
        raise AnalysisError(f"Analysis output root traverses a symlink: {output}")
    cursor = output
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise AnalysisError(f"Analysis output root contains a symlink: {cursor}")
        cursor = cursor.parent
    if resolved != analysis_root(campaign).resolve(strict=False):
        raise AnalysisError(
            f"Analysis root must be exactly {analysis_root(campaign).resolve(strict=False)}"
        )
    return campaign, resolved


@functools.lru_cache(maxsize=1)
def _live_parent_verification(path_text: str, expected_sha256: str) -> dict[str, Any]:
    from tools import final_v10_bundle_receipt

    path = Path(path_text)
    if path.resolve() != DEFAULT_PARENT_FINAL_V10_RECEIPT.resolve():
        raise AnalysisError("Live FINAL-v10 replay is restricted to the canonical parent receipt")
    try:
        published = final_v10_bundle_receipt.verify_published_receipt()
    except Exception as exc:
        raise AnalysisError("Live sealed FINAL-v10 verification failed") from exc
    if (
        published.get("status") != EXPECTED_PARENT_FINAL_V10_STATUS
        or _sha256(path) != expected_sha256
    ):
        raise AnalysisError("Live FINAL-v10 replay returned a different sealed parent")
    return {
        "status": "PASS",
        "published_status": EXPECTED_PARENT_FINAL_V10_STATUS,
        "verifier": _artifact(Path(final_v10_bundle_receipt.__file__).resolve()),
    }


def _validate_parent_receipt(
    path: Path, *, expected_sha256: str = EXPECTED_PARENT_FINAL_V10_RECEIPT_SHA256
) -> dict[str, Any]:
    identity = _artifact(path)
    if identity["sha256"] != expected_sha256:
        raise AnalysisError(
            "Sealed FINAL-v10 parent receipt identity drifted: "
            f"expected {expected_sha256}, observed {identity['sha256']}"
        )
    receipt = _read_json(path)
    if receipt.get("status") != EXPECTED_PARENT_FINAL_V10_STATUS:
        raise AnalysisError("FINAL-v10 parent receipt is not terminal-success evidence")
    return {
        "receipt": identity,
        "live_verification": _live_parent_verification(str(path.resolve()), expected_sha256),
    }


def _exact_job_pairs() -> list[tuple[str, int]]:
    return [(arm, seed) for arm in ARMS for seed in SEEDS]


def load_training_source(
    campaign_root: Path,
    *,
    parent_final_v10_receipt: Path = DEFAULT_PARENT_FINAL_V10_RECEIPT,
    expected_parent_sha256: str = EXPECTED_PARENT_FINAL_V10_RECEIPT_SHA256,
) -> SourceBundle:
    """Deeply authenticate the terminal 20-chain/100-fold source campaign."""
    campaign, _output = _guard_roots(campaign_root)
    parent_identity = _validate_parent_receipt(
        Path(parent_final_v10_receipt), expected_sha256=expected_parent_sha256
    )
    _strict_read_json_tree(training.contract_path(campaign))
    _strict_read_json_tree(training.preflight_path(campaign))
    _strict_read_json_tree(campaign / "receipts/scheduler.json")
    try:
        training.validate_contract(campaign, deep=True)
    except Exception as exc:
        raise AnalysisError("Phase-1 campaign contract does not replay deeply") from exc

    terminal_path = training.training_receipt_path(campaign)
    terminal = _read_json(terminal_path)
    required_keys = {
        "schema_version",
        "status",
        "created_utc",
        "contract",
        "preflight",
        "scheduler",
        "arms",
        "seeds",
        "job_count",
        "folds_per_job",
        "logical_fit_count",
        "refit_count",
        "job_receipts",
    }
    if set(terminal) != required_keys:
        raise AnalysisError("Training terminal receipt field roster changed")
    expected_semantics = {
        "schema_version": training.SCHEMA_VERSION,
        "status": "complete_and_certified",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "job_count": 20,
        "folds_per_job": 5,
        "logical_fit_count": 100,
        "refit_count": 0,
    }
    mismatch = {
        key: {"expected": value, "observed": terminal.get(key)}
        for key, value in expected_semantics.items()
        if terminal.get(key) != value
    }
    if mismatch:
        raise AnalysisError(f"Training terminal semantics drifted: {mismatch}")

    expected_refs = {
        "contract": _artifact(training.contract_path(campaign)),
        "preflight": _artifact(training.preflight_path(campaign)),
        "scheduler": _artifact(campaign / "receipts/scheduler.json"),
    }
    if any(terminal.get(key) != value for key, value in expected_refs.items()):
        raise AnalysisError("Training terminal source identities drifted")
    for identity in expected_refs.values():
        _validate_artifact(identity)

    pairs = _exact_job_pairs()
    expected_job_paths = [training.job_receipt_path(campaign, arm, seed) for arm, seed in pairs]
    expected_job_identities = [_artifact(path) for path in expected_job_paths]
    if terminal["job_receipts"] != expected_job_identities:
        raise AnalysisError("Training terminal job-receipt roster or identity drifted")
    receipt_directory = campaign / "receipts/source_cv"
    actual_job_paths = (
        sorted(receipt_directory.rglob("*.json")) if receipt_directory.is_dir() else []
    )
    if set(actual_job_paths) != set(expected_job_paths):
        raise AnalysisError("Training job receipt inventory contains missing or extra files")

    manifests: dict[str, pd.DataFrame] = {}
    oof_by_arm_seed: dict[tuple[str, int], pd.DataFrame] = {}
    job_identities: list[dict[str, Any]] = []
    for arm in ARMS:
        manifest = pd.read_csv(training.manifest_path(campaign, arm), low_memory=False)
        try:
            training._validate_arm_frame(manifest, arm)  # noqa: SLF001
        except Exception as exc:
            raise AnalysisError(f"Derived manifest fails replay for {arm}") from exc
        manifests[arm] = manifest
        for seed in SEEDS:
            receipt_path = training.job_receipt_path(campaign, arm, seed)
            try:
                job = training._validate_job(campaign, arm, seed)  # noqa: SLF001
            except Exception as exc:
                raise AnalysisError(
                    f"Training job is not terminal-valid: {arm}/seed{seed}"
                ) from exc
            strict_job = _strict_read_json_tree(receipt_path)
            if strict_job != job:
                raise AnalysisError(f"Training job JSON replay drifted: {arm}/seed{seed}")
            if (
                job.get("status") != "completed"
                or job.get("arm") != arm
                or job.get("seed") != seed
                or job.get("fit_count") != 5
                or job.get("refit_count") != 0
            ):
                raise AnalysisError(f"Training job semantics drifted: {arm}/seed{seed}")
            artifacts = job.get("artifacts") or {}
            if (
                artifacts.get("fold_count") != 5
                or artifacts.get("refit_count") != 0
                or artifacts.get("oof_rows") != len(manifest)
            ):
                raise AnalysisError(f"Training fold/refit/OOF census drifted: {arm}/seed{seed}")
            oof_path = training.run_dir(campaign, arm, seed) / "oof_predictions.parquet"
            if artifacts.get("oof") != _artifact(oof_path):
                raise AnalysisError(f"OOF identity drifted: {arm}/seed{seed}")
            for identity in _iter_artifacts(job):
                _validate_artifact(identity)
            oof_by_arm_seed[(arm, seed)] = pd.read_parquet(oof_path)
            job_identities.append(_artifact(receipt_path))

    identities = {
        "parent_final_v10_receipt": parent_identity,
        "campaign_contract": expected_refs["contract"],
        "training_terminal_receipt": _artifact(terminal_path),
        "training_preflight": expected_refs["preflight"],
        "training_scheduler": expected_refs["scheduler"],
        "job_receipts": job_identities,
        "campaign_controller": _artifact(Path(training.__file__).resolve()),
        "analysis_implementation": _artifact(Path(__file__).resolve()),
        "analysis_test": _artifact(ANALYSIS_TEST),
    }
    return SourceBundle(campaign, identities, manifests, oof_by_arm_seed)


def _binary_labels(values: pd.Series, *, context: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise AnalysisError(f"{context}: labels must be finite binary integers")
    labels = numeric.astype(int)
    if set(labels) != {0, 1}:
        raise AnalysisError(f"{context}: both binary labels are required")
    return labels


def aggregate_patient_scores(source: SourceBundle) -> pd.DataFrame:
    """Join exact OOF rosters and compute patient-native logits for all seeds."""
    arm_tables: list[pd.DataFrame] = []
    required_manifest = {
        "slide_id",
        "patient_id",
        "target_label",
        "cohort",
        "subcohort",
        "specimen_role",
        "k_fold",
    }
    for arm in ARMS:
        manifest = source.manifests.get(arm)
        if manifest is None or required_manifest - set(manifest):
            raise AnalysisError(f"{arm}: manifest metadata is incomplete")
        manifest = manifest.copy()
        if manifest["slide_id"].astype(str).duplicated().any():
            raise AnalysisError(f"{arm}: duplicate slide IDs in manifest")
        metadata_columns = [
            "slide_id",
            "patient_id",
            "target_label",
            "cohort",
            "subcohort",
            "specimen_role",
            "k_fold",
        ]
        patient: pd.DataFrame | None = None
        for seed in SEEDS:
            oof = source.oof_by_arm_seed.get((arm, seed))
            if oof is None:
                raise AnalysisError(f"Missing exact OOF table: {arm}/seed{seed}")
            required_oof = {"slide_id", "label", "logit", "fold"}
            if required_oof - set(oof):
                raise AnalysisError(f"{arm}/seed{seed}: OOF columns are incomplete")
            if len(oof) != len(manifest) or oof["slide_id"].astype(str).duplicated().any():
                raise AnalysisError(f"{arm}/seed{seed}: OOF roster size or uniqueness changed")
            joined = oof[["slide_id", "label", "logit", "fold"]].merge(
                manifest[metadata_columns],
                on="slide_id",
                how="inner",
                validate="one_to_one",
            )
            if len(joined) != len(manifest):
                raise AnalysisError(f"{arm}/seed{seed}: OOF roster is not the exact manifest")
            labels = pd.to_numeric(joined["label"], errors="raise").astype(int)
            target = pd.to_numeric(joined["target_label"], errors="raise").astype(int)
            folds = pd.to_numeric(joined["fold"], errors="raise").astype(int)
            expected_folds = pd.to_numeric(joined["k_fold"], errors="raise").astype(int)
            logits = pd.to_numeric(joined["logit"], errors="coerce").to_numpy(float)
            if (
                not labels.equals(target)
                or not folds.equals(expected_folds)
                or not np.isfinite(logits).all()
            ):
                raise AnalysisError(f"{arm}/seed{seed}: OOF label/fold/native-logit drift")
            grouped = joined.assign(logit=logits).groupby("patient_id", sort=True)
            consistency = grouped[
                ["target_label", "cohort", "subcohort", "specimen_role", "k_fold"]
            ].nunique(dropna=False)
            if (consistency != 1).any().any():
                raise AnalysisError(f"{arm}/seed{seed}: patient metadata is inconsistent")
            current = grouped.agg(
                target_label=("target_label", "first"),
                cohort=("cohort", "first"),
                subcohort=("subcohort", "first"),
                specimen_role=("specimen_role", "first"),
                k_fold=("k_fold", "first"),
                n_slides=("slide_id", "size"),
                slide_roster_sha256=(
                    "slide_id",
                    lambda values: hashlib.sha256(
                        "\n".join(sorted(values.astype(str))).encode("utf-8")
                    ).hexdigest(),
                ),
                **{f"logit_seed{seed}": ("logit", "mean")},
            ).reset_index()
            current["target_label"] = pd.to_numeric(current["target_label"], errors="raise").astype(
                int
            )
            current["k_fold"] = pd.to_numeric(current["k_fold"], errors="raise").astype(int)
            if patient is None:
                patient = current
            else:
                metadata = [
                    "patient_id",
                    "target_label",
                    "cohort",
                    "subcohort",
                    "specimen_role",
                    "k_fold",
                    "n_slides",
                    "slide_roster_sha256",
                ]
                if not patient[metadata].equals(current[metadata]):
                    raise AnalysisError(f"{arm}: patient/slide/fold roster differs by seed")
                patient[f"logit_seed{seed}"] = current[f"logit_seed{seed}"].to_numpy(float)
        if patient is None:
            raise AnalysisError(f"No patient scores resolved for {arm}")
        seed_columns = [f"logit_seed{seed}" for seed in SEEDS]
        patient["mean_logit_5seed"] = patient[seed_columns].mean(axis=1)
        patient.insert(0, "arm", arm)
        _binary_labels(patient["target_label"], context=arm)
        arm_tables.append(patient)
    combined = pd.concat(arm_tables, ignore_index=True)
    combined = combined.sort_values(["arm", "patient_id"], kind="mergesort").reset_index(drop=True)
    _validate_patient_table(combined)
    return combined


def _validate_patient_table(patient_scores: pd.DataFrame) -> None:
    expected_columns = {
        "arm",
        "patient_id",
        "target_label",
        "cohort",
        "subcohort",
        "specimen_role",
        "k_fold",
        "n_slides",
        "slide_roster_sha256",
        "mean_logit_5seed",
        *(f"logit_seed{seed}" for seed in SEEDS),
    }
    if set(patient_scores) != expected_columns:
        raise AnalysisError("Patient score column contract changed")
    if patient_scores[["arm", "patient_id"]].duplicated().any():
        raise AnalysisError("Patient score arm/patient roster is not unique")
    if set(patient_scores["arm"].astype(str)) != set(ARMS):
        raise AnalysisError("Patient score arm roster changed")
    for arm in ARMS:
        frame = patient_scores.loc[patient_scores["arm"].eq(arm)]
        expected = training.EXPECTED_ARM_CENSUS[arm]
        labels = _binary_labels(frame["target_label"], context=arm)
        if (
            len(frame) != expected["patients"]
            or int(labels.sum()) != expected["mutant"]
            or int((labels == 0).sum()) != expected["wild_type"]
        ):
            raise AnalysisError(f"{arm}: patient score census changed")
        if not frame["specimen_role"].astype(str).str.casefold().eq("primary").all():
            raise AnalysisError(f"{arm}: nonprimary patient entered analysis")
        values = frame[[*(f"logit_seed{seed}" for seed in SEEDS), "mean_logit_5seed"]]
        if not np.isfinite(values.to_numpy(float)).all():
            raise AnalysisError(f"{arm}: non-finite native patient logits")
        expected_mean = values[[f"logit_seed{seed}" for seed in SEEDS]].mean(axis=1)
        if not np.array_equal(
            expected_mean.to_numpy(float), frame["mean_logit_5seed"].to_numpy(float)
        ):
            raise AnalysisError(f"{arm}: five-seed native-logit mean changed")
    rosters = {
        arm: set(patient_scores.loc[patient_scores["arm"].eq(arm), "patient_id"].astype(str))
        for arm in ARMS
    }
    if not rosters["sr386_primary"] < rosters["surgen_primary"] < rosters["tcga_surgen_primary"]:
        raise AnalysisError("SR386 < SurGen < combined patient nesting changed")
    if not rosters["tcga_primary"] < rosters["tcga_surgen_primary"]:
        raise AnalysisError("TCGA < combined patient nesting changed")
    if rosters["tcga_primary"] & rosters["surgen_primary"]:
        raise AnalysisError("TCGA and SurGen patient populations are not disjoint")


def _metric_points(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if not np.isfinite(scores).all() or set(labels) != {0, 1}:
        raise AnalysisError("Binary metrics require finite scores and both labels")
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def _named_rng(seed: int, name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    child = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(child)


def stratified_bootstrap_indices(
    labels: np.ndarray,
    subcohorts: Sequence[str],
    *,
    n_bootstrap: int,
    seed: int,
    stream: str,
) -> np.ndarray:
    if n_bootstrap < 1:
        raise AnalysisError("n_bootstrap must be positive")
    labels = np.asarray(labels)
    subcohorts = np.asarray(subcohorts, dtype=str)
    if len(labels) != len(subcohorts) or set(labels) != {0, 1}:
        raise AnalysisError("Bootstrap strata require aligned binary patient labels")
    keys = np.asarray(
        [
            f"{subcohort}\x1f{int(label)}"
            for subcohort, label in zip(subcohorts, labels, strict=True)
        ],
        dtype=str,
    )
    rng = _named_rng(seed, stream)
    groups = [np.flatnonzero(keys == key) for key in sorted(set(keys))]
    if any(len(group) == 0 for group in groups):
        raise AnalysisError("Bootstrap contains an empty stratum")
    sampled = [rng.choice(group, size=(n_bootstrap, len(group)), replace=True) for group in groups]
    return np.concatenate(sampled, axis=1).astype(np.int64, copy=False)


def bootstrap_metric_samples(
    labels: np.ndarray, scores: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    auroc = np.empty(len(indices), dtype=np.float64)
    auprc = np.empty(len(indices), dtype=np.float64)
    for draw, row in enumerate(indices):
        auroc[draw], auprc[draw] = _metric_points(labels[row], scores[row])
    return auroc, auprc


def _interval(samples: np.ndarray) -> list[float]:
    values = np.quantile(np.asarray(samples, dtype=float), [0.025, 0.975])
    return [float(values[0]), float(values[1])]


def _aligned_pair(
    patient_scores: pd.DataFrame, spec: ContrastSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = (
        patient_scores.loc[patient_scores["arm"].eq(spec.evaluation_arm)]
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    larger = (
        patient_scores.loc[
            patient_scores["arm"].eq(spec.larger_arm)
            & patient_scores["patient_id"].isin(evaluation["patient_id"])
        ]
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    smaller = (
        patient_scores.loc[
            patient_scores["arm"].eq(spec.smaller_arm)
            & patient_scores["patient_id"].isin(evaluation["patient_id"])
        ]
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    metadata = [
        "patient_id",
        "target_label",
        "cohort",
        "subcohort",
        "k_fold",
        "n_slides",
        "slide_roster_sha256",
    ]
    if len(larger) != len(evaluation) or len(smaller) != len(evaluation):
        raise AnalysisError(f"{spec.key}: exact shared-patient roster is incomplete")
    if not larger[metadata].equals(evaluation[metadata]) or not smaller[metadata].equals(
        evaluation[metadata]
    ):
        raise AnalysisError(f"{spec.key}: paired patient metadata/slide roster drifted")
    labels = _binary_labels(evaluation["target_label"], context=spec.key)
    if (
        len(evaluation) != spec.expected_patients
        or int(labels.sum()) != spec.expected_mutant
        or int((labels == 0).sum()) != spec.expected_wild_type
    ):
        raise AnalysisError(f"{spec.key}: paired evaluation census changed")
    return larger, smaller


def build_analysis(
    patient_scores: pd.DataFrame,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    inference_unit: str = "patient",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute deterministic patient-level points, intervals, and paired contrasts."""
    if inference_unit != "patient":
        raise AnalysisError("Model seeds and folds are not permitted as inference units")
    patient_scores = patient_scores.sort_values(
        ["arm", "patient_id"], kind="mergesort"
    ).reset_index(drop=True)
    _validate_patient_table(patient_scores)
    arrays: dict[str, np.ndarray] = {}
    arm_performance: dict[str, Any] = {}
    for arm in ARMS:
        frame = (
            patient_scores.loc[patient_scores["arm"].eq(arm)]
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
        labels = _binary_labels(frame["target_label"], context=arm)
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            auroc, auprc = _metric_points(labels, frame[f"logit_seed{seed}"].to_numpy(float))
            per_seed[str(seed)] = {"auroc": auroc, "auprc": auprc}
        indices = stratified_bootstrap_indices(
            labels,
            frame["subcohort"].astype(str),
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
            stream=f"arm:{arm}",
        )
        ensemble_scores = frame["mean_logit_5seed"].to_numpy(float)
        point_auroc, point_auprc = _metric_points(labels, ensemble_scores)
        draws_auroc, draws_auprc = bootstrap_metric_samples(labels, ensemble_scores, indices)
        arrays[f"arm__{arm}__ensemble__auroc"] = draws_auroc
        arrays[f"arm__{arm}__ensemble__auprc"] = draws_auprc
        arm_performance[arm] = {
            "population": {
                "patients": int(len(frame)),
                "mutant": int(labels.sum()),
                "wild_type": int((labels == 0).sum()),
            },
            "per_seed_descriptive": per_seed,
            "five_seed_mean_native_logit": {
                "auroc": point_auroc,
                "auroc_ci95": _interval(draws_auroc),
                "auprc": point_auprc,
                "auprc_ci95": _interval(draws_auprc),
            },
        }

    paired: dict[str, Any] = {}
    for spec in CONTRASTS:
        larger, smaller = _aligned_pair(patient_scores, spec)
        labels = _binary_labels(larger["target_label"], context=spec.key)
        indices = stratified_bootstrap_indices(
            labels,
            larger["subcohort"].astype(str),
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
            stream=f"contrast:{spec.key}",
        )
        larger_scores = larger["mean_logit_5seed"].to_numpy(float)
        smaller_scores = smaller["mean_logit_5seed"].to_numpy(float)
        larger_points = _metric_points(labels, larger_scores)
        smaller_points = _metric_points(labels, smaller_scores)
        larger_auroc, larger_auprc = bootstrap_metric_samples(labels, larger_scores, indices)
        smaller_auroc, smaller_auprc = bootstrap_metric_samples(labels, smaller_scores, indices)
        delta_auroc = larger_auroc - smaller_auroc
        delta_auprc = larger_auprc - smaller_auprc
        prefix = f"contrast__{spec.key}"
        arrays[f"{prefix}__larger__auroc"] = larger_auroc
        arrays[f"{prefix}__smaller__auroc"] = smaller_auroc
        arrays[f"{prefix}__delta__auroc"] = delta_auroc
        arrays[f"{prefix}__larger__auprc"] = larger_auprc
        arrays[f"{prefix}__smaller__auprc"] = smaller_auprc
        arrays[f"{prefix}__delta__auprc"] = delta_auprc
        paired[spec.key] = {
            "larger_arm": spec.larger_arm,
            "smaller_arm": spec.smaller_arm,
            "evaluation_population": spec.evaluation_arm,
            "patients": int(len(larger)),
            "mutant": int(labels.sum()),
            "wild_type": int((labels == 0).sum()),
            "paired_indices_shared": True,
            "auroc": {
                "larger": larger_points[0],
                "smaller": smaller_points[0],
                "delta_larger_minus_smaller": larger_points[0] - smaller_points[0],
                "delta_ci95": _interval(delta_auroc),
            },
            "auprc": {
                "larger": larger_points[1],
                "smaller": smaller_points[1],
                "delta_larger_minus_smaller": larger_points[1] - smaller_points[1],
                "delta_ci95": _interval(delta_auprc),
            },
        }

    ranking = {
        metric: sorted(
            (
                {
                    "arm": arm,
                    metric: arm_performance[arm]["five_seed_mean_native_logit"][metric],
                }
                for arm in ARMS
            ),
            key=lambda item: (-float(item[metric]), str(item["arm"])),
        )
        for metric in ("auroc", "auprc")
    }
    results = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "experiment": EXPERIMENT,
        "design_status": "additive_FINAL_v10_5_source_cohort_OOF_analysis",
        "score_contract": {
            "slide_to_patient": "arithmetic mean of native slide logits within patient and seed",
            "five_seed_ensemble": "arithmetic mean of five patient native logits",
            "probability_roundtrip_used": False,
            "model_seeds": list(SEEDS),
        },
        "inference": {
            "unit": "patient",
            "bootstrap_draws": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "stratification": "subcohort_x_KRAS_label",
            "interval": "percentile_95",
            "model_seeds_are_inference_units": False,
            "folds_are_inference_units": False,
            "paired_indices_shared": True,
            "bootstrap_array_count": len(BOOTSTRAP_ARRAY_NAMES),
        },
        "arm_performance": arm_performance,
        "paired_common_patient_contrasts": paired,
        "cross_population_ranking": {
            "role": "descriptive_only",
            "no_cross_population_inference_or_transport_claim": True,
            **ranking,
        },
        "scope_boundary": {
            "append_only_to_final_v10": True,
            "supersedes_parent_fields": [],
            "canonical_final_v10_e0_unchanged": True,
        },
    }
    if set(arrays) != set(BOOTSTRAP_ARRAY_NAMES):
        raise AnalysisError("Bootstrap array name roster changed")
    if any(value.dtype != np.float64 or value.shape != (n_bootstrap,) for value in arrays.values()):
        raise AnalysisError("Bootstrap arrays must be exact float64 vectors")
    return results, {key: arrays[key] for key in BOOTSTRAP_ARRAY_NAMES}


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for key in sorted(arrays):
            array = np.asarray(arrays[key], dtype=np.float64)
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    frame.to_parquet(output, index=False)
    return output.getvalue()


def _analysis_contract(
    source: SourceBundle,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    test_only_noncanonical_parameters: bool,
    created_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_before_analysis_compute",
        "created_utc": created_utc,
        "experiment": EXPERIMENT,
        "campaign_root": str(source.campaign_root),
        "source_inputs": source.identities,
        "population_contract": {
            "arms": list(ARMS),
            "expected_arm_census": training.EXPECTED_ARM_CENSUS,
            "strict_nesting": [
                "sr386_primary < surgen_primary < tcga_surgen_primary",
                "tcga_primary < tcga_surgen_primary",
            ],
        },
        "fit_contract": {
            "chains": 20,
            "folds_per_chain": 5,
            "oof_fits": 100,
            "refits": 0,
            "seeds": list(SEEDS),
        },
        "analysis_contract": {
            "inference_unit": "patient",
            "model_seeds_are_inference_units": False,
            "folds_are_inference_units": False,
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "test_only_noncanonical_parameters": test_only_noncanonical_parameters,
            "stratification": "subcohort_x_KRAS_label",
            "paired_contrasts": [dataclasses.asdict(spec) for spec in CONTRASTS],
            "cross_population_ranking": "descriptive_only",
            "bootstrap_arrays": {
                "names": list(BOOTSTRAP_ARRAY_NAMES),
                "count": len(BOOTSTRAP_ARRAY_NAMES),
                "dtype": "float64",
                "length": n_bootstrap,
            },
        },
        "output_inventory": list(ANALYSIS_FILES),
    }


def _load_or_publish_contract(
    source: SourceBundle,
    root: Path,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    test_only_noncanonical_parameters: bool,
) -> dict[str, Any]:
    path = root / "contract.json"
    if path.exists() or path.is_symlink():
        stored = _read_json(path)
        created = stored.get("created_utc")
        if not isinstance(created, str) or not created:
            raise AnalysisError("Analysis contract lacks a sealed creation timestamp")
        expected = _analysis_contract(
            source,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            test_only_noncanonical_parameters=test_only_noncanonical_parameters,
            created_utc=created,
        )
        if stored != expected:
            raise AnalysisError("Stored analysis contract does not replay exactly")
        return stored
    root.mkdir(parents=True, exist_ok=False)
    contract = _analysis_contract(
        source,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        test_only_noncanonical_parameters=test_only_noncanonical_parameters,
        created_utc=_utcnow(),
    )
    _write_bytes_once(path, _json_bytes(contract))
    return contract


def _build_product(source: SourceBundle, contract: Mapping[str, Any]) -> AnalysisProduct:
    patients = aggregate_patient_scores(source)
    settings = contract["analysis_contract"]
    results, arrays = build_analysis(
        patients,
        n_bootstrap=int(settings["n_bootstrap"]),
        bootstrap_seed=int(settings["bootstrap_seed"]),
        inference_unit=str(settings["inference_unit"]),
    )
    results["inputs"] = source.identities
    return AnalysisProduct(patients, results, arrays)


def _validate_analysis_parameters(
    campaign: Path,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    allow_test_parameters: bool,
) -> bool:
    canonical = n_bootstrap == N_BOOTSTRAP and bootstrap_seed == BOOTSTRAP_SEED
    if canonical:
        if allow_test_parameters:
            raise AnalysisError(
                "The test-only parameter gate is invalid for canonical production values"
            )
        return False
    if not allow_test_parameters:
        raise AnalysisError(
            f"Production analysis requires exactly {N_BOOTSTRAP} draws and bootstrap seed "
            f"{BOOTSTRAP_SEED}"
        )
    temporary = Path("/tmp").resolve()
    try:
        campaign.resolve().relative_to(temporary)
    except ValueError as exc:
        raise AnalysisError(
            "Noncanonical analysis parameters are restricted to /tmp tests"
        ) from exc
    return True


def analyze(
    campaign_root: Path,
    *,
    parent_final_v10_receipt: Path = DEFAULT_PARENT_FINAL_V10_RECEIPT,
    expected_parent_sha256: str = EXPECTED_PARENT_FINAL_V10_RECEIPT_SHA256,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    _allow_test_parameters: bool = False,
) -> dict[str, Any]:
    """Publish the immutable analysis component; existing complete output is verify-only."""
    campaign, root = _guard_roots(campaign_root)
    test_only_parameters = _validate_analysis_parameters(
        campaign,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        allow_test_parameters=_allow_test_parameters,
    )
    source = load_training_source(
        campaign,
        parent_final_v10_receipt=parent_final_v10_receipt,
        expected_parent_sha256=expected_parent_sha256,
    )
    if (root / "analysis_completion_receipt.json").is_file():
        return verify(
            campaign,
            parent_final_v10_receipt=parent_final_v10_receipt,
            expected_parent_sha256=expected_parent_sha256,
        )
    if root.exists() and set(path.name for path in root.iterdir()) != {"contract.json"}:
        raise AnalysisError("Refusing to repair or overwrite a partial analysis component")
    contract = _load_or_publish_contract(
        source,
        root,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        test_only_noncanonical_parameters=test_only_parameters,
    )
    if set(path.name for path in root.iterdir()) != {"contract.json"}:
        raise AnalysisError("Analysis directory changed after contract publication")
    product = _build_product(source, contract)
    patient_path = root / "patient_native_logits.parquet"
    result_path = root / "results.json"
    bootstrap_path = root / "bootstrap_distributions.npz"
    _write_bytes_once(patient_path, _parquet_bytes(product.patient_scores))
    _write_bytes_once(result_path, _json_bytes(product.results))
    _write_bytes_once(bootstrap_path, _npz_bytes(product.bootstraps))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_and_certified",
        "created_utc": _utcnow(),
        "experiment": EXPERIMENT,
        "analysis_contract": _artifact(root / "contract.json"),
        "source_inputs": source.identities,
        "artifacts": {
            "patient_native_logits": _artifact(patient_path),
            "results": _artifact(result_path),
            "bootstrap_distributions": _artifact(bootstrap_path),
        },
        "inference": {
            "unit": "patient",
            "model_seeds_are_inference_units": False,
            "folds_are_inference_units": False,
            "paired_indices_shared": True,
            "bootstrap_arrays": {
                "names": list(BOOTSTRAP_ARRAY_NAMES),
                "count": len(BOOTSTRAP_ARRAY_NAMES),
                "dtype": "float64",
                "length": int(contract["analysis_contract"]["n_bootstrap"]),
            },
        },
        "output_inventory": list(ANALYSIS_FILES),
    }
    _write_bytes_once(root / "analysis_completion_receipt.json", _json_bytes(receipt))
    return verify(
        campaign,
        parent_final_v10_receipt=parent_final_v10_receipt,
        expected_parent_sha256=expected_parent_sha256,
    )


def verify(
    campaign_root: Path,
    *,
    parent_final_v10_receipt: Path = DEFAULT_PARENT_FINAL_V10_RECEIPT,
    expected_parent_sha256: str = EXPECTED_PARENT_FINAL_V10_RECEIPT_SHA256,
) -> dict[str, Any]:
    """Read-only full replay of source, patient aggregation, inference, and receipt."""
    campaign, root = _guard_roots(campaign_root)
    if not root.is_dir() or root.is_symlink():
        raise AnalysisError(f"Analysis component missing or symlinked: {root}")
    if set(path.name for path in root.iterdir()) != set(ANALYSIS_FILES):
        raise AnalysisError("Analysis component inventory contains missing or extra files")
    source = load_training_source(
        campaign,
        parent_final_v10_receipt=parent_final_v10_receipt,
        expected_parent_sha256=expected_parent_sha256,
    )
    contract = _read_json(root / "contract.json")
    created = contract.get("created_utc")
    if not isinstance(created, str) or not created:
        raise AnalysisError("Analysis contract timestamp is invalid")
    expected_contract = _analysis_contract(
        source,
        n_bootstrap=int((contract.get("analysis_contract") or {}).get("n_bootstrap", -1)),
        bootstrap_seed=int((contract.get("analysis_contract") or {}).get("bootstrap_seed", -1)),
        test_only_noncanonical_parameters=bool(
            (contract.get("analysis_contract") or {}).get(
                "test_only_noncanonical_parameters", False
            )
        ),
        created_utc=created,
    )
    settings = contract.get("analysis_contract") or {}
    observed_test_only = _validate_analysis_parameters(
        campaign,
        n_bootstrap=int(settings.get("n_bootstrap", -1)),
        bootstrap_seed=int(settings.get("bootstrap_seed", -1)),
        allow_test_parameters=bool(settings.get("test_only_noncanonical_parameters", False)),
    )
    if observed_test_only != bool(settings.get("test_only_noncanonical_parameters", False)):
        raise AnalysisError("Analysis parameter mode does not replay exactly")
    if contract != expected_contract:
        raise AnalysisError("Analysis contract does not replay exactly")
    product = _build_product(source, contract)
    stored_patients = pd.read_parquet(root / "patient_native_logits.parquet")
    try:
        pd.testing.assert_frame_equal(
            stored_patients, product.patient_scores, check_exact=True, check_dtype=True
        )
    except AssertionError as exc:
        raise AnalysisError("Stored patient native logits do not replay exactly") from exc
    stored_results = _read_json(root / "results.json")
    if stored_results != product.results:
        raise AnalysisError("Stored analysis results do not replay exactly")
    if (root / "results.json").read_bytes() != _json_bytes(product.results):
        raise AnalysisError("Stored analysis result bytes are not canonical")
    if (root / "patient_native_logits.parquet").read_bytes() != _parquet_bytes(
        product.patient_scores
    ):
        raise AnalysisError("Stored patient native-logit Parquet bytes are not canonical")
    if (root / "bootstrap_distributions.npz").read_bytes() != _npz_bytes(product.bootstraps):
        raise AnalysisError("Stored deterministic bootstrap NPZ does not replay exactly")

    receipt = _read_json(root / "analysis_completion_receipt.json")
    required_receipt_keys = {
        "schema_version",
        "status",
        "created_utc",
        "experiment",
        "analysis_contract",
        "source_inputs",
        "artifacts",
        "inference",
        "output_inventory",
    }
    if set(receipt) != required_receipt_keys:
        raise AnalysisError("Analysis completion receipt field roster changed")
    expected_receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_and_certified",
        "created_utc": receipt.get("created_utc"),
        "experiment": EXPERIMENT,
        "analysis_contract": _artifact(root / "contract.json"),
        "source_inputs": source.identities,
        "artifacts": {
            "patient_native_logits": _artifact(root / "patient_native_logits.parquet"),
            "results": _artifact(root / "results.json"),
            "bootstrap_distributions": _artifact(root / "bootstrap_distributions.npz"),
        },
        "inference": {
            "unit": "patient",
            "model_seeds_are_inference_units": False,
            "folds_are_inference_units": False,
            "paired_indices_shared": True,
            "bootstrap_arrays": {
                "names": list(BOOTSTRAP_ARRAY_NAMES),
                "count": len(BOOTSTRAP_ARRAY_NAMES),
                "dtype": "float64",
                "length": int(contract["analysis_contract"]["n_bootstrap"]),
            },
        },
        "output_inventory": list(ANALYSIS_FILES),
    }
    if not isinstance(receipt.get("created_utc"), str) or receipt != expected_receipt:
        raise AnalysisError("Analysis completion receipt does not replay exactly")
    return stored_results


def status(campaign_root: Path) -> dict[str, Any]:
    campaign, root = _guard_roots(campaign_root)
    if not training.training_receipt_path(campaign).is_file():
        return {"status": "waiting_for_training", "analysis_root": str(root)}
    if not root.exists():
        return {"status": "ready", "analysis_root": str(root)}
    if (root / "analysis_completion_receipt.json").is_file():
        try:
            verify(campaign)
        except Exception as exc:  # noqa: BLE001 - status reports invalid state
            return {
                "status": "partial_or_invalid",
                "analysis_root": str(root),
                "error": type(exc).__name__,
            }
        return {"status": "complete_and_certified", "analysis_root": str(root)}
    return {"status": "partial_or_invalid", "analysis_root": str(root)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=training.DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--parent-final-v10-receipt",
        type=Path,
        default=DEFAULT_PARENT_FINAL_V10_RECEIPT,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.set_defaults(func=cmd_plan)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--apply", action="store_true")
    analyze_parser.set_defaults(func=cmd_analyze)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.set_defaults(func=cmd_verify)
    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(func=cmd_status)
    return parser


def cmd_plan(args: argparse.Namespace) -> None:
    campaign, root = _guard_roots(args.campaign_root)
    print(
        json.dumps(
            {
                "status": "PLAN_ONLY_NO_WRITES",
                "campaign_root": str(campaign),
                "analysis_root": str(root),
                "output_inventory": list(ANALYSIS_FILES),
                "model_seeds": list(SEEDS),
                "inference_unit": "patient",
                "paired_contrasts": [spec.key for spec in CONTRASTS],
            },
            indent=2,
        )
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    if not args.apply:
        load_training_source(
            args.campaign_root,
            parent_final_v10_receipt=args.parent_final_v10_receipt,
        )
        print("DRY RUN PASS: sealed training source is ready; no analysis files written")
        return
    result = analyze(
        args.campaign_root,
        parent_final_v10_receipt=args.parent_final_v10_receipt,
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    print(json.dumps({"status": "PASS", "experiment": result["experiment"]}, indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    result = verify(
        args.campaign_root,
        parent_final_v10_receipt=args.parent_final_v10_receipt,
    )
    print(json.dumps({"status": "PASS", "experiment": result["experiment"]}, indent=2))


def cmd_status(args: argparse.Namespace) -> None:
    print(json.dumps(status(args.campaign_root), indent=2))


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
