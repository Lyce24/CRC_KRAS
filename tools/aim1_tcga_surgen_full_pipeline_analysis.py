#!/usr/bin/env python3
"""Governed FINAL-v11 scoring and analysis after TCGA+SurGen training.

This controller is additive.  It consumes the terminal receipt produced by
``aim1_tcga_surgen_two_encoder_campaign.py`` and never trains, refits, selects,
or calibrates on a target cohort.  Its public stages are deliberately ordered::

    prepare --apply
    preflight --apply
    score --apply --max-workers 6
    analyze --apply
    verify
    status

``prepare`` snapshots only label-blind, allowlisted target manifests.  The
score stage publishes fifty immutable native-logit files (two encoders, five
seeds, five targets) and then publishes one exactly-once inference seal.  No
outcome file is opened before that seal.  ``analyze`` is the first stage that
authenticates and joins target outcomes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import threading
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools import aim1_tcga_surgen_two_encoder_recovery as training_recovery  # noqa: E402

SCHEMA_VERSION = 1
EXPERIMENT = "final_v11_tcga_surgen_two_encoder_full_pipeline"
SEEDS = (42, 43, 44, 45, 46)
ENCODERS = ("univ1", "virchow2_cls")
TARGET_ORDER = (
    "cptac_primary",
    "rih_primary",
    "rih_metastatic",
    "sr1482_metastatic",
    "orion_cpht",
)
DEFAULT_MAX_WORKERS = 6
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_824
REPORT_PERFORMANCE_CLAIM_COUNT = 78
REPORT_CONTRAST_CLAIM_COUNT = 48
WHY_D_EVIDENCE_RECORD_COUNT = 2
DEFAULT_CAMPAIGN_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim1_tcga_surgen_two_encoder_v1_20260824"
)
TRAINING_CONTROLLER = REPO / "tools/aim1_tcga_surgen_two_encoder_campaign.py"
TRAINING_RECOVERY_CONTROLLER = REPO / "tools/aim1_tcga_surgen_two_encoder_recovery.py"
TRAINING_RECOVERY_TEST = REPO / "tests/test_aim1_tcga_surgen_two_encoder_recovery.py"
TRAINING_RECOVERY_CONTROLLER_SHA256 = (
    "d47cbd54b584fb3d0dd5160ae0bdd1fe2aca79b53207f8c1b63e366957b66ae5"
)
TRAINING_RECOVERY_CONTROLLER_SIZE = 66_818
TRAINING_RECOVERY_TEST_SHA256 = "2cf182aab9cb8a92d4416402024251a7b450adc11e0be7da4f24d0bfac6c2f24"
TRAINING_RECOVERY_TEST_SIZE = 27_047
TRAINING_RECOVERY_TERMINAL = Path("recovery_v1/receipts/training_complete_recovered.json")
ANALYSIS_TEST = REPO / "tests/test_aim1_tcga_surgen_full_pipeline_analysis.py"

OLD_UNIV1_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim1_primary_cohort_5seed_v1_20260824/train/source_cv/cap8192/"
    "tcga_surgen_primary"
)
SOURCE_MANIFEST_SHA256 = "d7087a23a84a294670080eb5090f83612f57b3376c7696ed2d0094fb9bcff8a5"
SOURCE_SLIDES = 1_389
SOURCE_PATIENTS = 1_239
SOURCE_MUTANT = 501

CANONICAL_MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/aim1_dev.csv")
CANONICAL_MANIFEST_SHA256 = "d906ed5b61c5d3bbf56da7ec7bad412287307461012deee5e6d98ae97bd1f3d1"
CANONICAL_SLIDES = 1_642
CANONICAL_PATIENTS = 1_486
CANONICAL_MUTANT = 604
FINAL_V10_RECEIPT = REPO / "reports/final_v10/report_bundle_receipt.json"
FINAL_V10_RECEIPT_SHA256 = "7d3ac2b82f71dd956c0e2b5ad7c951474ce7310c63c421f1a347b5e0c55f43e2"
CANONICAL_UNI_VALIDATION = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "final_v9_mil_5seed_expansion_v1_20260823/aim1_e0/receipts/"
    "five_seed_training_validation.json"
)
CANONICAL_UNI_VALIDATION_SHA256 = "2fc59d7cc4b34e2e737b58cb77197e7b3fc910405713e1eacbab969797c38f92"
CANONICAL_V2_CONTRACT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "final_v9_mil_5seed_expansion_v1_20260823/aim3_ladders/inputs/experiment_contract.json"
)
CANONICAL_V2_CONTRACT_SHA256 = "4fad1be62564f9091747365939da76d4dc2cc44c7bec3499c6115ceebd7bed17"
CANONICAL_V2_AUDIT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "final_v9_mil_5seed_expansion_v1_20260823/aim3_ladders/analysis/analysis_audit.json"
)
CANONICAL_V2_AUDIT_SHA256 = "550b03cfb0d9cd3a2b1ad42dce0618931c2b0b40606fb25789e2a1505a2afc2c"

# Evaluation-only clinicopathologic covariates used by the governed E1a/E1d
# restrictions.  This frozen source is never part of model fitting, scoring,
# calibration, or target selection.  It is opened only after the label-blind
# inference seal and is already bound by the inherited Aim-1 sealed replay.
DERIVED_COVARIATE_SOURCE = Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v4.csv")
DERIVED_COVARIATE_SOURCE_SHA256 = "312438c12aaa25b4376cc55b70ebc3a23763f149c7413f1fde7c3ce71c21596c"
DERIVED_COVARIATE_SOURCE_SIZE = 1_431_625

UNI_PACK = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/20x_256px_0px_overlap_mpp0.5/packed_uni_v1"
)
V2_PACK = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/"
    "20x_224px_0px_overlap_mpp0.5/packed_virchow2_cls_1280"
)

LABEL_BLIND_ALLOWED = frozenset(
    {
        "slide_id",
        "patient_id",
        "cohort",
        "subcohort",
        "specimen_role",
        "role",
        "liver_class",
        "mpp",
        "mpp_source",
        "patch_count",
        "exclude_neoadjuvant",
        "exclude_ambiguous_crc15",
    }
)
FORBIDDEN_OUTCOME_COLUMNS = frozenset(
    {
        "label",
        "target_label",
        "kras",
        "kras_status",
        "kras_mutant",
        "ras",
        "nras",
        "braf",
        "msi",
        "msi_dmmr",
        "outcome",
    }
)

ANALYSIS_FILES = (
    "contract.json",
    "patient_native_logits.parquet",
    "bootstrap_distributions.npz",
    "results.json",
    "analysis_completion_receipt.json",
)


class GovernanceError(RuntimeError):
    """Fail-closed violation of the FINAL-v11 downstream contract."""


@dataclass(frozen=True)
class TargetSpec:
    key: str
    source_blind: Path
    source_blind_sha256: str
    outcome_source: Path
    outcome_sha256: str
    slides: int
    patients: int
    mutant: int
    family_exposure: str
    inferential_role: str


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    pack_dir: Path
    feature_dim: int


TARGETS: dict[str, TargetSpec] = {
    spec.key: spec
    for spec in (
        TargetSpec(
            "cptac_primary",
            Path(
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "final_v9_mil_5seed_expansion_v1_20260823/aim2_loco/inputs/"
                "label_blind/primary/family_cptac.csv"
            ),
            "135a454edbd94d2c27889946187faedf68312be0cbb6f93786987beae970eb5b",
            Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_cptac_primary.csv"),
            "f134ee9c087a12f6370982aa343778a03811ea077684f797bd8edde7853e69ad",
            98,
            94,
            33,
            "family_naive",
            "zero_shot_external_primary",
        ),
        TargetSpec(
            "rih_primary",
            Path(
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "final_v9_mil_5seed_expansion_v1_20260823/aim2_loco/inputs/"
                "label_blind/primary/family_rih.csv"
            ),
            "0c33a3562da87a9843a41a34e6693e37825538aa5be0c31971798943f010a06e",
            Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_rih_primary.csv"),
            "0f72f5bc45327aab88368501f667e3f64e89db5066c954a02474ce009601c9c6",
            155,
            153,
            70,
            "family_naive",
            "zero_shot_external_primary",
        ),
        TargetSpec(
            "rih_metastatic",
            Path(
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "aim2_final_v8_complete_v1_20260822/e2met/inputs/manifests/rih_m.csv"
            ),
            "dec443d99e437e7cb3fc3297a4f98302081ba4455386e3b7d930ff142feb1e5c",
            Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_rih_metastatic.csv"),
            "6aaab722a96c2374296811f7bd4c79f44d842efe75788240274b14f79c3d1303",
            85,
            85,
            37,
            "family_naive",
            "standalone_metastatic_sensitivity",
        ),
        TargetSpec(
            "sr1482_metastatic",
            Path(
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "aim2_final_v8_complete_v1_20260822/e2met/inputs/manifests/sr1482_m.csv"
            ),
            "90eb00ef0baf6a33fc4b423affbce33dde1d38068add59cc136f870f13cf6d15",
            Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_surgen_metastatic.csv"),
            "3e69383c2869b1432fd6a1326c79fd5a6d50ec0c0dee88029c8dd433c9e06135",
            100,
            74,
            30,
            "source_exposed_SR1482_primary",
            "source_exposed_metastatic_sensitivity",
        ),
        TargetSpec(
            "orion_cpht",
            Path(
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "aim2_final_v8_complete_v1_20260822/cpht/inputs/orion_primary.csv"
            ),
            "6acb9699c53944c1fa87b90a42e773eed830ef71594589de5d71005a1a31d024",
            Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v5.csv"),
            "bc83cd44a59f4ee0dbc672ab23087882449d80ad331c0d4304ecda96d0f79e0d",
            41,
            40,
            15,
            "family_naive_but_outcomes_historically_accessed",
            "retrospective_raw_CPHT_sensitivity",
        ),
    )
}

ENCODER_SPECS = {
    "univ1": EncoderSpec("univ1", UNI_PACK, 1_024),
    "virchow2_cls": EncoderSpec("virchow2_cls", V2_PACK, 1_280),
}

EXPECTED_FIT_ACCOUNTING = {
    "adopted_oof_fits": 25,
    "new_oof_fits": 25,
    "new_refits": 10,
    "new_fits": 35,
    "physical_new_fits": 35,
    "operational_lineage_fits": 60,
    "recovery_new_fits": 0,
    "hidden_fits": 0,
}
EXPECTED_EXECUTION_ACCOUNTING = {
    "job_count": 10,
    "attempts_per_job": 1,
    "total_attempts": 10,
    "retries": 0,
}
EXPECTED_TRAINING_TOP_LEVEL = {
    "schema_version",
    "recovery",
    "base_campaign",
    "status",
    "created_utc",
    "base_contract",
    "base_preflight",
    "erratum_contract",
    "scheduler_recovery",
    "validator_adjudication",
    "recovery_implementation",
    "source_population",
    "seeds",
    "encoders",
    "job_count",
    "fit_accounting",
    "execution_accounting",
    "concurrency",
    "adopted_univ1_oof_chains",
    "stock_univ1_job_receipts",
    "recovered_virchow2_job_receipts",
    "new_chains",
    "new_job_receipts",
    "raw_artifact_hash_census",
    "certification_boundary",
    "stock_receipts_fabricated",
}


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


def _regular_file(path: Path, *, context: str = "artifact") -> Path:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise GovernanceError(f"Required regular {context} missing or symlinked: {path}")
    return path


def _artifact(path: Path) -> dict[str, Any]:
    path = _regular_file(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _validate_identity(identity: Mapping[str, Any], *, expected_path: Path | None = None) -> None:
    if set(identity) != {"path", "sha256", "size_bytes"}:
        raise GovernanceError(f"Malformed artifact identity: {identity}")
    path = Path(str(identity["path"]))
    if expected_path is not None and path != expected_path.resolve(strict=False):
        raise GovernanceError(f"Artifact path drift: expected {expected_path}, got {path}")
    _validate_identity_at_path(identity, path)


def _validate_identity_at_path(identity: Mapping[str, Any], path: Path) -> None:
    """Validate identity bytes at an already context-resolved artifact path."""
    if set(identity) != {"path", "sha256", "size_bytes"}:
        raise GovernanceError(f"Malformed artifact identity: {identity}")
    path = Path(path)
    _regular_file(path)
    expected_size = int(identity["size_bytes"])
    expected_sha = str(identity["sha256"])
    if int(path.stat().st_size) != expected_size:
        raise GovernanceError(f"Artifact size drifted: {path}")
    # The training contract pins the 49/81-GB packed matrices by an upstream
    # authenticated digest.  Rehashing both on every downstream status check
    # would make the control plane unusable; their regular-file path and size
    # are replayed here while pack meta/index/source-inventory identities are
    # independently authenticated by this controller.
    if expected_size <= 2_000_000_000:
        if _sha256(path) != expected_sha:
            raise GovernanceError(f"Artifact SHA drifted: {path}")
    elif len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise GovernanceError(f"Malformed upstream large-artifact digest: {path}")


def _iter_identities(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if set(value) == {"path", "sha256", "size_bytes"}:
            yield value
        else:
            for child in value.values():
                yield from _iter_identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_identities(child)


def _contains_identity(value: Any, wanted: Mapping[str, Any]) -> bool:
    return any(identity == dict(wanted) for identity in _iter_identities(value))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GovernanceError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise GovernanceError(f"Non-finite JSON constant: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    _regular_file(path, context="JSON")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError(f"Invalid strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GovernanceError(f"JSON root must be an object: {path}")
    return value


_RAW_CENSUS_KEYS = {
    "root",
    "excluded_prefix",
    "artifact_count",
    "total_size_bytes",
    "tree_sha256",
    "artifacts",
}


def _safe_identity_base(raw_base: Any, *, context: str) -> Path:
    if not isinstance(raw_base, str) or not raw_base:
        raise GovernanceError(f"{context}: artifact-identity base must be a non-empty string")
    base = Path(raw_base)
    if not base.is_absolute():
        raise GovernanceError(f"{context}: artifact-identity base is ambiguous/non-absolute")
    resolved = base.resolve(strict=False)
    if base != resolved:
        raise GovernanceError(f"{context}: artifact-identity base is non-normalized or symlinked")
    if not base.is_dir() or base.is_symlink():
        raise GovernanceError(f"{context}: artifact-identity base is missing or symlinked")
    return resolved


def _iter_identity_bindings(
    value: Any, *, default_base: Path, context: str
) -> Iterable[tuple[dict[str, Any], Path]]:
    """Yield identities with the directory that owns any relative path.

    Native completion JSON stores paths relative to the completion artifact's
    directory.  The bounded-recovery raw census is the one deliberate
    exception: its records are relative to the exact authenticated ``root``
    carried by that census object.
    """
    if isinstance(value, dict):
        if set(value) == {"path", "sha256", "size_bytes"}:
            yield value, default_base
            return
        if set(value) == _RAW_CENSUS_KEYS:
            artifacts = value["artifacts"]
            if not isinstance(artifacts, list) or any(
                not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}
                for record in artifacts
            ):
                raise GovernanceError(f"{context}: malformed raw artifact census records")
            if any(Path(str(record["path"])).is_absolute() for record in artifacts):
                raise GovernanceError(f"{context}: raw artifact census paths must be relative")
            if value.get("artifact_count") != len(artifacts):
                raise GovernanceError(f"{context}: raw artifact census count drifted")
            census_base = _safe_identity_base(value["root"], context=f"{context}.root")
            for record in artifacts:
                yield record, census_base
            return
        if "root" in value and "artifacts" in value:
            raise GovernanceError(
                f"{context}: ambiguous artifact-identity base in census-like object"
            )
        for key, child in value.items():
            yield from _iter_identity_bindings(
                child,
                default_base=default_base,
                context=f"{context}.{key}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_identity_bindings(
                child,
                default_base=default_base,
                context=f"{context}[{index}]",
            )


def _resolve_identity_path(identity: Mapping[str, Any], *, base: Path, context: str) -> Path:
    raw_value = identity.get("path")
    if not isinstance(raw_value, str) or not raw_value:
        raise GovernanceError(f"{context}: artifact identity path must be a non-empty string")
    raw_path = Path(raw_value)
    if raw_path.is_absolute():
        resolved = raw_path.resolve(strict=False)
        if raw_path.as_posix() != raw_value or raw_path != resolved:
            raise GovernanceError(
                f"{context}: absolute artifact path is non-normalized or symlinked"
            )
        return resolved
    if (
        raw_path.as_posix() != raw_value
        or any(part in {".", ".."} for part in raw_path.parts)
        or "\\" in raw_value
    ):
        raise GovernanceError(f"{context}: relative artifact path is non-normalized/traversing")
    safe_base = _safe_identity_base(str(base), context=f"{context}.base")
    candidate = safe_base / raw_path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(safe_base)
    except ValueError as exc:
        raise GovernanceError(f"{context}: relative artifact path escapes its owning base") from exc
    if candidate != resolved:
        raise GovernanceError(f"{context}: relative artifact path traverses a symlink")
    return resolved


def _strict_json_tree(path: Path, *, visited: set[Path] | None = None) -> dict[str, Any]:
    seen = set() if visited is None else visited
    source = _regular_file(Path(path), context="JSON")
    resolved = source.resolve(strict=False)
    value = _read_json(source)
    if resolved in seen:
        return value
    seen.add(resolved)
    for index, (identity, base) in enumerate(
        _iter_identity_bindings(value, default_base=resolved.parent, context="$")
    ):
        child = _resolve_identity_path(identity, base=base, context=f"identity[{index}]")
        _validate_identity_at_path(identity, child)
        if child.suffix.casefold() == ".json":
            _strict_json_tree(child, visited=seen)
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    except (TypeError, ValueError) as exc:
        raise GovernanceError("JSON payload is not finite and serializable") from exc


def _write_bytes_once(path: Path, data: bytes) -> None:
    """Atomically publish immutable bytes; identical replay is idempotent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
            return
        raise GovernanceError(f"Refusing to overwrite immutable artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            linked = True
        except FileExistsError as exc:
            if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
                return
            raise GovernanceError(f"Concurrent immutable publication conflict: {path}") from exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    if not linked:
        raise GovernanceError(f"Failed to publish immutable artifact: {path}")


def _write_json_once(path: Path, value: Any) -> None:
    _write_bytes_once(path, _json_bytes(value))


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    _write_bytes_once(path, _parquet_bytes(frame))


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in sorted(arrays):
            array = np.asarray(arrays[name], dtype=np.float64)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise GovernanceError(f"Bootstrap array must be a finite vector: {name}")
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, payload.getvalue())
    return output.getvalue()


def _safe_campaign_root(campaign_root: Path) -> Path:
    raw = Path(campaign_root).expanduser()
    if not raw.is_absolute():
        raise GovernanceError("Campaign root must be absolute")
    lexical = raw.absolute()
    resolved = raw.resolve(strict=False)
    if lexical != resolved:
        raise GovernanceError(f"Campaign root traverses a symlink: {raw} -> {resolved}")
    cursor = raw
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise GovernanceError(f"Campaign-root path contains a symlink: {cursor}")
        cursor = cursor.parent
    default = DEFAULT_CAMPAIGN_ROOT.resolve(strict=False)
    tmp = Path("/tmp").resolve()
    if resolved != default and resolved != tmp and tmp not in resolved.parents:
        raise GovernanceError(f"Production root must be exactly {default}; tests may use /tmp")
    return resolved


def downstream_root(campaign_root: Path) -> Path:
    return _safe_campaign_root(campaign_root) / "downstream"


def contract_path(campaign_root: Path) -> Path:
    return downstream_root(campaign_root) / "contract.json"


def blind_path(campaign_root: Path, target: str) -> Path:
    return downstream_root(campaign_root) / f"inputs/label_blind/{target}.csv"


def score_path(campaign_root: Path, encoder: str, target: str, seed: int) -> Path:
    return downstream_root(campaign_root) / f"scores/{encoder}/{target}/seed{seed}.parquet"


def score_receipt_path(campaign_root: Path, encoder: str, target: str, seed: int) -> Path:
    return score_path(campaign_root, encoder, target, seed).with_suffix(".receipt.json")


def preflight_path(campaign_root: Path) -> Path:
    return downstream_root(campaign_root) / "receipts/deep_preflight.json"


def scoring_receipt_path(campaign_root: Path) -> Path:
    return downstream_root(campaign_root) / "receipts/scoring_complete.json"


def environment_path(campaign_root: Path) -> Path:
    return downstream_root(campaign_root) / "inference/environment.json"


def seal_path(campaign_root: Path) -> Path:
    return downstream_root(campaign_root) / "inference/inference_seal.json"


def analysis_root(campaign_root: Path) -> Path:
    return _safe_campaign_root(campaign_root) / "analysis"


def checkpoint_path(campaign_root: Path, encoder: str, seed: int) -> Path:
    return _safe_campaign_root(campaign_root) / (
        f"train/e0/{encoder}/seed{seed}/final/refit/model.ckpt"
    )


def refit_info_path(campaign_root: Path, encoder: str, seed: int) -> Path:
    return checkpoint_path(campaign_root, encoder, seed).with_name("info.json")


def oof_path(campaign_root: Path, encoder: str, seed: int) -> Path:
    if encoder == "univ1":
        return OLD_UNIV1_ROOT / f"seed{seed}/oof_predictions.parquet"
    if encoder == "virchow2_cls":
        return _safe_campaign_root(campaign_root) / (
            f"train/e0/virchow2_cls/seed{seed}/oof_predictions.parquet"
        )
    raise GovernanceError(f"Unknown encoder: {encoder}")


def adopted_pointer_path(campaign_root: Path, seed: int) -> Path:
    return _safe_campaign_root(campaign_root) / f"inputs/adopted_runs/univ1_seed{seed}.json"


def _score_jobs(campaign_root: Path) -> list[dict[str, Any]]:
    root = _safe_campaign_root(campaign_root)
    jobs: list[dict[str, Any]] = []
    for encoder in ENCODERS:
        for target in TARGET_ORDER:
            for seed in SEEDS:
                jobs.append(
                    {
                        "job_id": f"final_v11.score.{encoder}.{target}.seed{seed}",
                        "encoder": encoder,
                        "target": target,
                        "seed": seed,
                        "fit_count": 0,
                        "contains_target_outcomes": False,
                        "expected_rows": TARGETS[target].slides,
                        "checkpoint": str(checkpoint_path(root, encoder, seed)),
                        "manifest": str(blind_path(root, target)),
                        "output": str(score_path(root, encoder, target, seed)),
                    }
                )
    return jobs


def _blind_frame(frame: pd.DataFrame, *, spec: TargetSpec, context: str) -> pd.DataFrame:
    columns = [str(column) for column in frame.columns]
    lower = {column.casefold() for column in columns}
    leaked = sorted(lower & FORBIDDEN_OUTCOME_COLUMNS)
    unallowed = sorted(set(columns) - LABEL_BLIND_ALLOWED)
    if leaked or unallowed:
        raise GovernanceError(
            f"{context}: label-blind schema violation; leaked={leaked}, unallowed={unallowed}"
        )
    if "slide_id" not in frame or "patient_id" not in frame:
        raise GovernanceError(f"{context}: slide_id and patient_id are required")
    if frame.isna()[["slide_id", "patient_id"]].any().any():
        raise GovernanceError(f"{context}: null slide/patient identifier")
    out = frame.copy()
    out["slide_id"] = out["slide_id"].astype(str)
    out["patient_id"] = out["patient_id"].astype(str)
    if out["slide_id"].duplicated().any():
        raise GovernanceError(f"{context}: duplicate slide identifiers")
    census = (len(out), out["patient_id"].nunique())
    if census != (spec.slides, spec.patients):
        raise GovernanceError(
            f"{context}: target census drifted; expected {(spec.slides, spec.patients)}, got {census}"
        )
    return out


def _validate_source_manifest(path: Path) -> pd.DataFrame:
    identity = _artifact(path)
    if identity["sha256"] != SOURCE_MANIFEST_SHA256:
        raise GovernanceError(f"TCGA+SurGen source-manifest SHA drifted: {identity['sha256']}")
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "slide_id",
        "patient_id",
        "target_label",
        "cohort",
        "subcohort",
        "specimen_role",
        "k_fold",
        "msi_dmmr",
        "braf",
        "tumor_site_group",
        "stage_class",
        "age_at_diagnosis",
        "sex",
    }
    if required - set(frame):
        raise GovernanceError(f"Source manifest lacks columns: {sorted(required - set(frame))}")
    frame = frame.copy()
    frame["slide_id"] = frame["slide_id"].astype(str)
    frame["patient_id"] = frame["patient_id"].astype(str)
    labels = pd.to_numeric(frame["target_label"], errors="raise").astype(int)
    census = (
        len(frame),
        frame["patient_id"].nunique(),
        int(frame.groupby("patient_id")["target_label"].first().sum()),
    )
    if census != (SOURCE_SLIDES, SOURCE_PATIENTS, SOURCE_MUTANT):
        raise GovernanceError(f"TCGA+SurGen source census drifted: {census}")
    if set(labels) != {0, 1} or frame["slide_id"].duplicated().any():
        raise GovernanceError("Source labels/slide IDs are not exact binary unique data")
    if not frame["specimen_role"].astype(str).str.casefold().eq("primary").all():
        raise GovernanceError("Nonprimary specimen entered TCGA+SurGen training source")
    if set(frame["cohort"].astype(str)) != {"TCGA", "SurGen"}:
        raise GovernanceError("TCGA+SurGen source cohort roster drifted")
    return frame


def _training_source_manifest(campaign_root: Path, training_contract: dict[str, Any]) -> Path:
    source_input = training_contract.get("source_input")
    if not isinstance(source_input, dict):
        raise GovernanceError("Training contract lacks source_input")
    identity = source_input.get("snapshot_manifest")
    if not isinstance(identity, dict):
        raise GovernanceError("Training contract lacks source_input.snapshot_manifest identity")
    _validate_identity(identity)
    return Path(str(identity["path"]))


def _validate_oof_frame(path: Path, manifest: pd.DataFrame, *, context: str) -> pd.DataFrame:
    frame = pd.read_parquet(_regular_file(path, context="OOF parquet"))
    required = {"slide_id", "label", "logit", "fold"}
    if required - set(frame):
        raise GovernanceError(f"{context}: OOF schema lacks {sorted(required - set(frame))}")
    if len(frame) != len(manifest) or frame["slide_id"].astype(str).duplicated().any():
        raise GovernanceError(f"{context}: OOF roster size/uniqueness drifted")
    joined = frame[["slide_id", "label", "logit", "fold"]].merge(
        manifest[["slide_id", "target_label", "k_fold"]],
        on="slide_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(manifest):
        raise GovernanceError(f"{context}: OOF does not exactly cover source slides")
    labels = pd.to_numeric(joined["label"], errors="raise").astype(int)
    expected_labels = pd.to_numeric(joined["target_label"], errors="raise").astype(int)
    folds = pd.to_numeric(joined["fold"], errors="raise").astype(int)
    expected_folds = pd.to_numeric(joined["k_fold"], errors="raise").astype(int)
    logits = pd.to_numeric(joined["logit"], errors="coerce").to_numpy(float)
    if not labels.equals(expected_labels) or not folds.equals(expected_folds):
        raise GovernanceError(f"{context}: OOF label/fold identity drifted")
    if not np.isfinite(logits).all() or set(folds) != {0, 1, 2, 3, 4}:
        raise GovernanceError(f"{context}: invalid native logits/fold roster")
    return frame


def _validate_pack(spec: EncoderSpec, target_slides: set[str]) -> dict[str, Any]:
    meta_path = spec.pack_dir / "meta.json"
    index_path = spec.pack_dir / "index.parquet"
    meta = _read_json(meta_path)
    expected_meta = {
        "schema_version": 1,
        "feat_dim": spec.feature_dim,
        "n_slides": 2_128,
    }
    mismatch = {
        key: {"expected": value, "observed": meta.get(key)}
        for key, value in expected_meta.items()
        if meta.get(key) != value
    }
    if mismatch:
        raise GovernanceError(f"{spec.key}: packed-feature metadata drifted: {mismatch}")
    source_dir = Path(str(meta.get("source_dir", "")))
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise GovernanceError(f"{spec.key}: packed source directory missing/symlinked")
    index = pd.read_parquet(_regular_file(index_path, context="packed index"))
    if "slide_id" not in index or index["slide_id"].astype(str).duplicated().any():
        raise GovernanceError(f"{spec.key}: invalid packed index")
    packed = set(index["slide_id"].astype(str))
    missing = sorted(target_slides - packed)
    if missing:
        raise GovernanceError(f"{spec.key}: target slides absent from pack: {missing[:5]}")
    large_files: dict[str, Any] = {}
    for name in ("features.bin", "coords.bin"):
        path = _regular_file(spec.pack_dir / name, context="packed binary")
        large_files[name] = {
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
        }
    record: dict[str, Any] = {
        "pack_dir": str(spec.pack_dir.resolve()),
        "feature_dim": spec.feature_dim,
        "source_dir": str(source_dir.resolve()),
        "meta": _artifact(meta_path),
        "index": _artifact(index_path),
        "large_file_stat_seals": large_files,
        "source_inventory_sha256": meta.get("source_inventory_sha256"),
        "n_slides": int(len(index)),
    }
    representation = spec.pack_dir / "representation.json"
    if representation.is_file() and not representation.is_symlink():
        record["representation"] = _artifact(representation)
    return record


def _forbidden_stock_training_artifacts(root: Path) -> list[Path]:
    forbidden = [
        root / "receipts/training_complete.json",
        root / "receipts/scheduler.json",
        *(root / f"requests/virchow2_full/seed{seed}.failure.json" for seed in SEEDS),
        *(root / f"requests/univ1_refit/seed{seed}.failure.json" for seed in SEEDS),
    ]
    observed = [path for path in forbidden if path.exists() or path.is_symlink()]
    stock_v2_dir = root / "receipts/jobs/virchow2_full"
    if stock_v2_dir.is_dir():
        observed.extend(stock_v2_dir.iterdir())
    elif stock_v2_dir.exists() or stock_v2_dir.is_symlink():
        observed.append(stock_v2_dir)
    return sorted(observed, key=lambda path: path.as_posix())


def _validate_training_bundle(campaign_root: Path, *, deep: bool = True) -> dict[str, Any]:
    """Authenticate the terminal 60-fit lineage and every inference checkpoint."""
    root = _safe_campaign_root(campaign_root)
    recovery_sources = {
        "controller": {
            "path": TRAINING_RECOVERY_CONTROLLER,
            "sha256": TRAINING_RECOVERY_CONTROLLER_SHA256,
            "size_bytes": TRAINING_RECOVERY_CONTROLLER_SIZE,
        },
        "controller_test": {
            "path": TRAINING_RECOVERY_TEST,
            "sha256": TRAINING_RECOVERY_TEST_SHA256,
            "size_bytes": TRAINING_RECOVERY_TEST_SIZE,
        },
    }
    for name, expected_identity in recovery_sources.items():
        observed_identity = _artifact(Path(expected_identity["path"]))
        if observed_identity != {
            "path": str(Path(expected_identity["path"]).resolve()),
            "sha256": expected_identity["sha256"],
            "size_bytes": expected_identity["size_bytes"],
        }:
            raise GovernanceError(f"Frozen training-recovery {name} bytes drifted")

    unexpected_stock = [str(path) for path in _forbidden_stock_training_artifacts(root)]
    if unexpected_stock:
        raise GovernanceError(
            "Bounded recovery requires absent stock terminal/scheduler/V2/failure "
            f"receipts: {unexpected_stock}"
        )

    terminal_path = root / TRAINING_RECOVERY_TERMINAL
    try:
        terminal = training_recovery.validate_recovered_terminal(root, deep_pack=deep)
    except training_recovery.ContractError as exc:
        raise GovernanceError("Bounded training-recovery replay failed") from exc
    stored_terminal = _strict_json_tree(terminal_path) if deep else _read_json(terminal_path)
    if terminal != stored_terminal:
        raise GovernanceError("Recovered training terminal differs from replayed evidence")
    if set(terminal) != EXPECTED_TRAINING_TOP_LEVEL:
        raise GovernanceError(
            "Training terminal top-level field roster drifted: "
            f"{sorted(set(terminal) ^ EXPECTED_TRAINING_TOP_LEVEL)}"
        )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "recovery": training_recovery.RECOVERY,
        "base_campaign": training_recovery.campaign.CAMPAIGN,
        "status": training_recovery.RECOVERY_STATUS,
        "seeds": list(SEEDS),
        "encoders": ["UNI-v1", "Virchow2-CLS"],
        "job_count": 10,
        "fit_accounting": EXPECTED_FIT_ACCOUNTING,
        "execution_accounting": EXPECTED_EXECUTION_ACCOUNTING,
        "stock_receipts_fabricated": False,
        "certification_boundary": (
            "bounded schema-v2 validator erratum; no fit, refit, retry, score, "
            "calibration, selection, or raw artifact mutation"
        ),
    }
    mismatch = {
        key: {"expected": value, "observed": terminal.get(key)}
        for key, value in expected.items()
        if terminal.get(key) != value
    }
    if mismatch:
        raise GovernanceError(f"Training terminal semantics drifted: {mismatch}")
    concurrency = terminal.get("concurrency")
    if (
        not isinstance(concurrency, dict)
        or set(concurrency) != {"maximum", "observed_peak", "witness_utc"}
        or concurrency.get("maximum") != 6
        or concurrency.get("observed_peak") != 6
        or not isinstance(concurrency.get("witness_utc"), str)
        or not concurrency["witness_utc"]
    ):
        raise GovernanceError("Recovered training concurrency witness drifted")
    raw_census = terminal.get("raw_artifact_hash_census")
    if (
        not isinstance(raw_census, dict)
        or set(raw_census)
        != {
            "artifact_count",
            "total_size_bytes",
            "tree_sha256_before",
            "tree_sha256_after",
            "unchanged",
        }
        or raw_census.get("unchanged") is not True
        or raw_census.get("tree_sha256_before") != raw_census.get("tree_sha256_after")
    ):
        raise GovernanceError("Recovered training raw-artifact immutability proof drifted")
    for key, relative in (
        ("base_contract", "contract.json"),
        ("base_preflight", "receipts/deep_preflight.json"),
        ("erratum_contract", "recovery_v1/contract_erratum.json"),
        ("scheduler_recovery", "recovery_v1/receipts/scheduler_recovery.json"),
        (
            "validator_adjudication",
            "recovery_v1/receipts/validator_adjudication.json",
        ),
    ):
        wanted = _artifact(root / relative)
        if terminal.get(key) != wanted:
            raise GovernanceError(f"Training terminal {key} identity drifted")
    if terminal.get("recovery_implementation") != {
        name: _artifact(Path(record["path"])) for name, record in recovery_sources.items()
    }:
        raise GovernanceError("Recovered terminal implementation identity drifted")
    training_contract = (
        _strict_json_tree(root / "contract.json") if deep else _read_json(root / "contract.json")
    )
    source_manifest_path = _training_source_manifest(root, training_contract)
    source_manifest = _validate_source_manifest(source_manifest_path)

    adopted = terminal.get("adopted_univ1_oof_chains")
    if not isinstance(adopted, list) or len(adopted) != 5:
        raise GovernanceError("Training terminal must contain five adopted UNI OOF chains")
    if sorted(int(item.get("seed", -1)) for item in adopted if isinstance(item, dict)) != list(
        SEEDS
    ):
        raise GovernanceError("Adopted UNI seed roster drifted")
    if any(
        not isinstance(item, dict) or item.get("oof_fits") != 5 or item.get("refits") != 0
        for item in adopted
    ):
        raise GovernanceError("Adopted UNI fit semantics drifted")
    new_chains = terminal.get("new_chains")
    if not isinstance(new_chains, list) or len(new_chains) != 10:
        raise GovernanceError("Training terminal must contain ten new chain records")
    kinds = sorted(str(item.get("kind")) for item in new_chains if isinstance(item, dict))
    if kinds != sorted(["univ1_refit"] * 5 + ["virchow2_full"] * 5):
        raise GovernanceError("New training-chain kind roster drifted")
    new_receipts = terminal.get("new_job_receipts")
    if not isinstance(new_receipts, list) or len(new_receipts) != 10:
        raise GovernanceError("Training terminal new-job receipt roster drifted")
    stock_uni_receipts = terminal.get("stock_univ1_job_receipts")
    recovered_v2_receipts = terminal.get("recovered_virchow2_job_receipts")
    if not isinstance(stock_uni_receipts, list) or len(stock_uni_receipts) != 5:
        raise GovernanceError("Recovered terminal stock UNI receipt roster drifted")
    if not isinstance(recovered_v2_receipts, list) or len(recovered_v2_receipts) != 5:
        raise GovernanceError("Recovered terminal V2 receipt roster drifted")
    expected_uni_paths = {
        str((root / f"receipts/jobs/univ1_refit/seed{seed}.json").resolve()) for seed in SEEDS
    }
    expected_v2_paths = {
        str((root / f"recovery_v1/receipts/jobs/virchow2_full/seed{seed}.json").resolve())
        for seed in SEEDS
    }
    if {
        str(item.get("path")) for item in stock_uni_receipts if isinstance(item, dict)
    } != expected_uni_paths:
        raise GovernanceError("Recovered terminal stock UNI receipt paths drifted")
    if {
        str(item.get("path")) for item in recovered_v2_receipts if isinstance(item, dict)
    } != expected_v2_paths:
        raise GovernanceError("Recovered terminal V2 receipt paths drifted")
    expected_new_receipts = sorted(
        [*stock_uni_receipts, *recovered_v2_receipts], key=lambda item: str(item["path"])
    )
    if new_receipts != expected_new_receipts:
        raise GovernanceError("Recovered terminal combined receipt graph drifted")
    expected_chain_keys = {
        "kind",
        "encoder",
        "seed",
        "oof_fits",
        "refits",
        "total_fits",
        "receipt_namespace",
        "job_receipt",
    }
    expected_chain_semantics = {
        (kind, seed): {
            "encoder": "UNI-v1" if kind == "univ1_refit" else "Virchow2-CLS",
            "oof_fits": 0 if kind == "univ1_refit" else 5,
            "refits": 1,
            "total_fits": 1 if kind == "univ1_refit" else 6,
        }
        for kind in ("univ1_refit", "virchow2_full")
        for seed in SEEDS
    }
    for chain in new_chains:
        if not isinstance(chain, dict) or set(chain) != expected_chain_keys:
            raise GovernanceError("New training-chain record schema drifted")
        key = (str(chain["kind"]), int(chain["seed"]))
        semantics = expected_chain_semantics.get(key)
        if semantics is None or any(chain.get(name) != value for name, value in semantics.items()):
            raise GovernanceError(f"New training-chain semantics drifted: {key}")
        expected_namespace = "immutable_stock" if key[0] == "univ1_refit" else "recovery_v1"
        if chain.get("receipt_namespace") != expected_namespace:
            raise GovernanceError(f"New training-chain namespace drifted: {key}")
        if chain.get("job_receipt") not in new_receipts:
            raise GovernanceError(f"New training-chain receipt is not in terminal roster: {key}")
    receipt_values: list[dict[str, Any]] = []
    for identity in new_receipts:
        if not isinstance(identity, dict):
            raise GovernanceError("Malformed training job-receipt identity")
        _validate_identity(identity)
        receipt_path = Path(str(identity["path"]))
        receipt_values.append(_strict_json_tree(receipt_path) if deep else _read_json(receipt_path))

    lineages: dict[str, Any] = {}
    for encoder in ENCODERS:
        lineages[encoder] = {}
        for seed in SEEDS:
            oof = oof_path(root, encoder, seed)
            checkpoint = checkpoint_path(root, encoder, seed)
            info = refit_info_path(root, encoder, seed)
            oof_identity = _artifact(oof)
            checkpoint_identity = _artifact(checkpoint)
            info_identity = _artifact(info)
            _validate_oof_frame(oof, source_manifest, context=f"{encoder}/seed{seed}")
            if encoder == "univ1":
                record = next(item for item in adopted if int(item.get("seed", -1)) == seed)
                if record.get("oof_predictions") != oof_identity:
                    raise GovernanceError(f"UNI seed{seed}: adopted terminal OOF drifted")
            else:
                completion_path = (
                    root / f"train/e0/virchow2_cls/seed{seed}/training_completion.json"
                )
                identity_path = root / f"train/e0/virchow2_cls/seed{seed}/training_identity.json"
                completion = (
                    _strict_json_tree(completion_path) if deep else _read_json(completion_path)
                )
                identity = _strict_json_tree(identity_path) if deep else _read_json(identity_path)
                completion_artifact = _artifact(
                    root / f"train/e0/virchow2_cls/seed{seed}/training_completion.json"
                )
                # A terminal may bind completion through its immutable job receipt.
                if not _contains_identity(terminal, completion_artifact) and not any(
                    _contains_identity(value, completion_artifact) for value in receipt_values
                ):
                    raise GovernanceError(
                        f"V2 seed{seed}: completion is absent from terminal graph"
                    )
                del completion, identity
            if not any(
                _contains_identity(value, checkpoint_identity)
                and _contains_identity(value, info_identity)
                for value in receipt_values
            ):
                raise GovernanceError(
                    f"{encoder}/seed{seed}: refit absent from terminal job receipts"
                )
            lineages[encoder][str(seed)] = {
                "oof": oof_identity,
                "checkpoint": checkpoint_identity,
                "refit_info": info_identity,
            }
    return {
        "terminal": _artifact(terminal_path),
        "contract": terminal["base_contract"],
        "preflight": terminal["base_preflight"],
        "scheduler": terminal["scheduler_recovery"],
        "erratum_contract": terminal["erratum_contract"],
        "validator_adjudication": terminal["validator_adjudication"],
        "recovery_implementation": terminal["recovery_implementation"],
        "certification_boundary": terminal["certification_boundary"],
        "raw_artifact_hash_census": terminal["raw_artifact_hash_census"],
        "source_manifest": _artifact(source_manifest_path),
        "source_manifest_path": str(source_manifest_path.resolve()),
        "lineages": lineages,
        "fit_accounting": EXPECTED_FIT_ACCOUNTING,
        "execution_accounting": EXPECTED_EXECUTION_ACCOUNTING,
        "concurrency": concurrency,
    }


def _target_source_records() -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    records: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    for key in TARGET_ORDER:
        spec = TARGETS[key]
        identity = _artifact(spec.source_blind)
        if identity["sha256"] != spec.source_blind_sha256:
            raise GovernanceError(f"{key}: frozen label-blind source SHA drifted")
        frame = _blind_frame(
            pd.read_csv(spec.source_blind, low_memory=False),
            spec=spec,
            context=f"{key}/source_blind",
        )
        records[key] = {
            "source_blind": identity,
            "expected_outcome_locator_after_seal": {
                "path": str(spec.outcome_source.resolve(strict=False)),
                "expected_sha256": spec.outcome_sha256,
            },
            "slides": spec.slides,
            "patients": spec.patients,
            "family_exposure": spec.family_exposure,
            "inferential_role": spec.inferential_role,
        }
        frames[key] = frame
    return records, frames


def _implementation_sources() -> list[dict[str, Any]]:
    required = [
        Path(__file__).resolve(),
        TRAINING_CONTROLLER,
        TRAINING_RECOVERY_CONTROLLER,
        TRAINING_RECOVERY_TEST,
        ANALYSIS_TEST,
        REPO / "aim2_cross_protocol_transfer.py",
        REPO / "aim1_clinical_baseline.py",
        REPO / "src/oceanpath/aim1/cli/e1a_whyd.py",
        REPO / "src/oceanpath/aim1/baselines.py",
        REPO / "src/oceanpath/eval/core.py",
        REPO / "src/oceanpath/datasets/datamodule.py",
        REPO / "src/oceanpath/datasets/packed.py",
        REPO / "src/oceanpath/training/lightning.py",
        REPO / "src/oceanpath/models/abmil.py",
        REPO / "pyproject.toml",
        REPO / "uv.lock",
    ]
    return [_artifact(path) for path in required]


def _label_blind_roster_relations(
    campaign_root: Path, training: Mapping[str, Any]
) -> dict[str, Any]:
    source = pd.read_csv(
        Path(str(training["source_manifest_path"])),
        usecols=["slide_id", "patient_id"],
    )
    source_slides = set(source["slide_id"].astype(str))
    source_patients = set(source["patient_id"].astype(str))
    frames = {
        key: pd.read_csv(blind_path(campaign_root, key), usecols=["slide_id", "patient_id"])
        for key in TARGET_ORDER
    }
    source_overlap: dict[str, Any] = {}
    for key, frame in frames.items():
        slide_overlap = source_slides & set(frame["slide_id"].astype(str))
        patient_overlap = source_patients & set(frame["patient_id"].astype(str))
        if slide_overlap or patient_overlap:
            raise GovernanceError(
                f"{key}: target/source identity overlap; "
                f"slides={len(slide_overlap)}, patients={len(patient_overlap)}"
            )
        source_overlap[key] = {"slides": 0, "patients": 0}
    all_target_slides = [
        slide for frame in frames.values() for slide in frame["slide_id"].astype(str).tolist()
    ]
    if len(all_target_slides) != 479 or len(set(all_target_slides)) != 479:
        raise GovernanceError("Five target manifests must contain 479 globally unique slides")
    patient_overlaps: dict[str, int] = {}
    for left_index, left in enumerate(TARGET_ORDER):
        for right in TARGET_ORDER[left_index + 1 :]:
            count = len(
                set(frames[left]["patient_id"].astype(str))
                & set(frames[right]["patient_id"].astype(str))
            )
            patient_overlaps[f"{left}__{right}"] = count
            expected = 8 if {left, right} == {"rih_primary", "rih_metastatic"} else 0
            if count != expected:
                raise GovernanceError(
                    f"Label-blind target patient-overlap drifted: {left}/{right}={count}"
                )
    return {
        "source_target_overlap": source_overlap,
        "target_slides_globally_unique": 479,
        "target_patient_pair_overlaps": patient_overlaps,
        "rih_primary_metastatic_overlap_patients": 8,
    }


def _contract_payload(
    campaign_root: Path,
    training: dict[str, Any],
    target_records: dict[str, Any],
    snapshots: dict[str, Any],
    feature_stores: dict[str, Any],
    *,
    created_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "prepared_label_blind_before_outcome_join",
        "created_utc": created_utc,
        "campaign_root": str(_safe_campaign_root(campaign_root)),
        "training": training,
        "seeds": list(SEEDS),
        "encoders": list(ENCODERS),
        "targets": target_records,
        "label_blind_snapshots": snapshots,
        "label_blind_roster_relations": _label_blind_roster_relations(campaign_root, training),
        "feature_stores": feature_stores,
        "score_contract": {
            "jobs": 50,
            "score_files": 50,
            "slide_rows": 4_790,
            "factorization": "2 encoders x 5 seeds x 5 targets",
            "native_logit_schema": ["slide_id", "seed", "fold", "logit"],
            "fold_value_for_refit_inference": 0,
            "maximum_parallel_scoring_workers": 6,
            "target_outcomes_present": False,
        },
        "governance": {
            "prepare_reads_target_outcomes": False,
            "preflight_reads_target_outcomes": False,
            "target_calibration": False,
            "target_model_selection": False,
            "inference_seal_exactly_once": True,
            "analysis_requires_inference_seal": True,
        },
        "implementation": _implementation_sources(),
        "analysis_output_inventory": list(ANALYSIS_FILES),
    }


def _load_contract(campaign_root: Path, *, deep: bool = True) -> dict[str, Any]:
    root = _safe_campaign_root(campaign_root)
    stored = _read_json(contract_path(root))
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise GovernanceError("Downstream contract lacks creation timestamp")
    training = _validate_training_bundle(root, deep=deep)
    target_records, frames = _target_source_records()
    snapshots: dict[str, Any] = {}
    target_slides: set[str] = set()
    for key, frame in frames.items():
        snapshot = blind_path(root, key)
        snapshot_identity = _artifact(snapshot)
        if snapshot_identity["sha256"] != TARGETS[key].source_blind_sha256:
            raise GovernanceError(f"{key}: label-blind snapshot is not an exact byte copy")
        replay = _blind_frame(
            pd.read_csv(snapshot, low_memory=False), spec=TARGETS[key], context=f"{key}/snapshot"
        )
        if not replay.equals(frame):
            raise GovernanceError(f"{key}: label-blind snapshot frame drifted")
        snapshots[key] = snapshot_identity
        target_slides.update(replay["slide_id"].astype(str))
    stores = {
        encoder: _validate_pack(ENCODER_SPECS[encoder], target_slides) for encoder in ENCODERS
    }
    expected = _contract_payload(
        root,
        training,
        target_records,
        snapshots,
        stores,
        created_utc=created,
    )
    if stored != expected:
        raise GovernanceError("Stored downstream contract does not replay exactly")
    if deep:
        _strict_json_tree(contract_path(root))
    return stored


def prepare(campaign_root: Path, *, apply: bool) -> dict[str, Any]:
    root = _safe_campaign_root(campaign_root)
    training = _validate_training_bundle(root)
    target_records, frames = _target_source_records()
    plan = {
        "campaign_root": str(root),
        "label_blind_targets": list(TARGET_ORDER),
        "score_jobs": len(_score_jobs(root)),
        "score_slide_rows": sum(TARGETS[job["target"]].slides for job in _score_jobs(root)),
        "outcome_files_opened": False,
    }
    if not apply:
        return plan
    for key in TARGET_ORDER:
        _write_bytes_once(blind_path(root, key), TARGETS[key].source_blind.read_bytes())
    snapshots = {key: _artifact(blind_path(root, key)) for key in TARGET_ORDER}
    target_slides = {slide for frame in frames.values() for slide in frame["slide_id"].astype(str)}
    stores = {
        encoder: _validate_pack(ENCODER_SPECS[encoder], target_slides) for encoder in ENCODERS
    }
    path = contract_path(root)
    created = _read_json(path)["created_utc"] if path.exists() else _utcnow()
    payload = _contract_payload(
        root,
        training,
        target_records,
        snapshots,
        stores,
        created_utc=created,
    )
    _write_json_once(path, payload)
    _write_json_once(downstream_root(root) / "jobs/score_jobs.json", {"jobs": _score_jobs(root)})
    _load_contract(root, deep=True)
    return payload


def preflight(campaign_root: Path, *, apply: bool) -> dict[str, Any]:
    root = _safe_campaign_root(campaign_root)
    contract = _load_contract(root, deep=True)
    jobs = _score_jobs(root)
    checkpoints = {(job["encoder"], int(job["seed"])) for job in jobs}
    for encoder, seed in checkpoints:
        _regular_file(checkpoint_path(root, encoder, seed), context="inference checkpoint")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_label_blind_inference",
        "created_utc": _utcnow(),
        "contract": _artifact(contract_path(root)),
        "training_terminal": contract["training"]["terminal"],
        "score_jobs": 50,
        "score_slide_rows": 4_790,
        "target_outcomes_present": False,
        "target_outcome_files_opened": False,
        "target_calibration": False,
        "target_model_selection": False,
        "maximum_parallel_scoring_workers": 6,
        "checks": {
            "training_terminal_authenticated": True,
            "ten_refit_checkpoints_authenticated": True,
            "five_label_blind_snapshots_exact": True,
            "both_packed_stores_cover_all_479_target_slides": True,
            "source_target_slide_and_patient_overlap_zero": True,
            "rih_primary_metastatic_overlap_exactly_eight": True,
        },
    }
    if apply:
        path = preflight_path(root)
        if path.exists():
            old = _read_json(path)
            receipt["created_utc"] = old.get("created_utc")
        _write_json_once(path, receipt)
    return receipt


def _score_environment(*, device: str, num_workers: int) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "device": device,
        "num_workers_per_score_job": int(num_workers),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "native_logits": True,
        "autocast_dtype": "bfloat16" if device == "cuda" else "none",
        "evaluation_bag": "full",
        "target_outcomes_present": False,
    }
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise GovernanceError("Governed production inference requires CUDA bfloat16")
            value.update(
                {
                    "torch_version": torch.__version__,
                    "torch_cuda_version": torch.version.cuda,
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    "cuda_capability": list(torch.cuda.get_device_capability(0)),
                }
            )
        except ImportError as exc:
            raise GovernanceError("PyTorch is unavailable for governed inference") from exc
    return value


def _seal_environment(campaign_root: Path, *, device: str, num_workers: int) -> dict[str, Any]:
    environment = _score_environment(device=device, num_workers=num_workers)
    path = environment_path(campaign_root)
    if path.exists():
        if _read_json(path) != environment:
            raise GovernanceError("Inference environment changed after publication")
    else:
        _write_json_once(path, environment)
    return environment


def _validate_score_frame(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    seed: int,
    context: str,
) -> None:
    if list(frame.columns) != ["slide_id", "seed", "fold", "logit"]:
        raise GovernanceError(f"{context}: score schema drifted: {list(frame.columns)}")
    if len(frame) != len(manifest):
        raise GovernanceError(f"{context}: score-row census drifted")
    if set(pd.to_numeric(frame["seed"], errors="raise").astype(int)) != {seed}:
        raise GovernanceError(f"{context}: score seed drifted")
    if set(pd.to_numeric(frame["fold"], errors="raise").astype(int)) != {0}:
        raise GovernanceError(f"{context}: refit inference fold marker must be zero")
    if frame.duplicated(["slide_id", "seed", "fold"]).any():
        raise GovernanceError(f"{context}: duplicate slide/model score")
    observed = set(frame["slide_id"].astype(str))
    expected = set(manifest["slide_id"].astype(str))
    logits = pd.to_numeric(frame["logit"], errors="coerce").to_numpy(float)
    if observed != expected or not np.isfinite(logits).all():
        raise GovernanceError(f"{context}: incomplete roster or non-finite logits")


def _validate_cached_score(
    campaign_root: Path,
    encoder: str,
    target: str,
    seed: int,
) -> pd.DataFrame | None:
    path = score_path(campaign_root, encoder, target, seed)
    receipt_path = score_receipt_path(campaign_root, encoder, target, seed)
    if not path.exists() and not receipt_path.exists():
        return None
    if (
        not path.is_file()
        or path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
    ):
        raise GovernanceError(f"Partial/symlinked immutable score cache: {path}")
    manifest_path = blind_path(campaign_root, target)
    receipt = _read_json(receipt_path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "contains_target_outcomes": False,
        "encoder": encoder,
        "target": target,
        "seed": seed,
        "manifest": _artifact(manifest_path),
        "checkpoint": _artifact(checkpoint_path(campaign_root, encoder, seed)),
        "contract": _artifact(contract_path(campaign_root)),
        "environment": _artifact(environment_path(campaign_root)),
        "artifact": _artifact(path),
        "n_rows": TARGETS[target].slides,
    }
    observed = {key: receipt.get(key) for key in expected}
    if observed != expected:
        raise GovernanceError(f"Score receipt drifted: {receipt_path}")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    frame = pd.read_parquet(path)
    _validate_score_frame(
        frame,
        manifest,
        seed=seed,
        context=f"{encoder}/{target}/seed{seed}",
    )
    return frame


def _score_one(
    campaign_root: Path,
    encoder: str,
    target: str,
    seed: int,
    *,
    device: str,
    num_workers: int,
) -> None:
    root = _safe_campaign_root(campaign_root)
    if encoder not in ENCODERS or target not in TARGET_ORDER or seed not in SEEDS:
        raise GovernanceError("Uncontracted internal score job")
    _load_contract(root, deep=False)
    _regular_file(preflight_path(root), context="downstream preflight receipt")
    _seal_environment(root, device=device, num_workers=num_workers)
    cached = _validate_cached_score(root, encoder, target, seed)
    if cached is not None:
        return
    if seal_path(root).exists():
        raise GovernanceError("Inference is already sealed; no new score may be created")
    manifest_path = blind_path(root, target)
    manifest = _blind_frame(
        pd.read_csv(manifest_path, low_memory=False),
        spec=TARGETS[target],
        context=f"{encoder}/{target}/seed{seed}",
    )
    checkpoint = checkpoint_path(root, encoder, seed)
    _regular_file(checkpoint, context="inference checkpoint")
    pack_record = _read_json(contract_path(root))["feature_stores"][encoder]
    from aim2_cross_protocol_transfer import _score_checkpoints

    scores = _score_checkpoints(
        [(seed, 0, checkpoint)],
        manifest,
        feature_dir=Path(str(pack_record["source_dir"])),
        pack_dir=Path(str(pack_record["pack_dir"])),
        device=device,
        num_workers=num_workers,
    )
    scores = (
        scores[["slide_id", "seed", "fold", "logit"]]
        .sort_values(["slide_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    _validate_score_frame(
        scores,
        manifest,
        seed=seed,
        context=f"{encoder}/{target}/seed{seed}",
    )
    destination = score_path(root, encoder, target, seed)
    _write_parquet_once(destination, scores)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": _utcnow(),
        "contains_target_outcomes": False,
        "encoder": encoder,
        "target": target,
        "seed": seed,
        "manifest": _artifact(manifest_path),
        "checkpoint": _artifact(checkpoint),
        "contract": _artifact(contract_path(root)),
        "environment": _artifact(environment_path(root)),
        "artifact": _artifact(destination),
        "n_rows": int(len(scores)),
    }
    _write_json_once(score_receipt_path(root, encoder, target, seed), receipt)
    _validate_cached_score(root, encoder, target, seed)


def _internal_score_command(
    campaign_root: Path,
    job: Mapping[str, Any],
    *,
    device: str,
    num_workers: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_score-one",
        "--campaign-root",
        str(_safe_campaign_root(campaign_root)),
        "--encoder",
        str(job["encoder"]),
        "--target",
        str(job["target"]),
        "--seed",
        str(job["seed"]),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
    ]


def _run_score_subprocess(command: Sequence[str]) -> None:
    subprocess.run(list(command), cwd=REPO, check=True)


def _seal_payload(campaign_root: Path) -> dict[str, Any]:
    root = _safe_campaign_root(campaign_root)
    records: list[dict[str, Any]] = []
    total_rows = 0
    for job in _score_jobs(root):
        encoder = str(job["encoder"])
        target = str(job["target"])
        seed = int(job["seed"])
        frame = _validate_cached_score(root, encoder, target, seed)
        if frame is None:
            raise GovernanceError(f"Cannot seal incomplete score job: {job['job_id']}")
        total_rows += len(frame)
        records.append(
            {
                "job_id": job["job_id"],
                "score": _artifact(score_path(root, encoder, target, seed)),
                "receipt": _artifact(score_receipt_path(root, encoder, target, seed)),
                "rows": len(frame),
            }
        )
    if len(records) != 50 or total_rows != 4_790:
        raise GovernanceError(f"Inference roster drifted: {len(records)} files/{total_rows} rows")
    scoring = _read_json(scoring_receipt_path(root))
    observed_peak = int(scoring.get("max_observed_parallel_workers", -1))
    if scoring.get("configured_max_workers") != 6 or not 1 <= observed_peak <= 6:
        raise GovernanceError("Scoring concurrency receipt violates the six-worker ceiling")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_before_outcome_join",
        "created_utc": _utcnow(),
        "target_outcomes_present": False,
        "target_outcome_files_opened": False,
        "contract": _artifact(contract_path(root)),
        "preflight": _artifact(preflight_path(root)),
        "scoring_completion": _artifact(scoring_receipt_path(root)),
        "encoders": list(ENCODERS),
        "seeds": list(SEEDS),
        "targets": list(TARGET_ORDER),
        "score_artifact_count": 50,
        "score_slide_rows": 4_790,
        "max_scoring_concurrency": 6,
        "score_artifacts": records,
    }


def seal_inference(campaign_root: Path) -> dict[str, Any]:
    root = _safe_campaign_root(campaign_root)
    path = seal_path(root)
    if path.exists():
        return verify_inference_seal(root)
    payload = _seal_payload(root)
    _write_json_once(path, payload)
    return verify_inference_seal(root)


def verify_inference_seal(campaign_root: Path) -> dict[str, Any]:
    root = _safe_campaign_root(campaign_root)
    path = seal_path(root)
    stored = _read_json(path)
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise GovernanceError("Inference seal lacks an immutable creation timestamp")
    expected = _seal_payload(root)
    expected["created_utc"] = created
    if stored != expected:
        raise GovernanceError("Inference seal or a sealed score artifact drifted")
    return stored


def score(
    campaign_root: Path,
    *,
    apply: bool,
    max_workers: int,
    device: str,
    num_workers: int,
) -> dict[str, Any]:
    root = _safe_campaign_root(campaign_root)
    _load_contract(root, deep=False)
    receipt = _read_json(preflight_path(root))
    if receipt.get("status") != "ready_for_label_blind_inference":
        raise GovernanceError("Downstream preflight is not terminal-ready")
    jobs = _score_jobs(root)
    if max_workers != DEFAULT_MAX_WORKERS:
        raise GovernanceError("Governed scoring requires exactly --max-workers 6")
    if seal_path(root).exists():
        seal = verify_inference_seal(root)
        return {"status": "already_sealed", "seal": seal}
    if not apply:
        return {
            "status": "dry_run",
            "jobs": len(jobs),
            "slide_rows": sum(int(job["expected_rows"]) for job in jobs),
            "max_workers": max_workers,
            "commands": [
                _internal_score_command(root, job, device=device, num_workers=num_workers)
                for job in jobs
            ],
        }
    _seal_environment(root, device=device, num_workers=num_workers)
    active = 0
    peak = 0
    lock = threading.Lock()

    def run(job: Mapping[str, Any]) -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            _run_score_subprocess(
                _internal_score_command(root, job, device=device, num_workers=num_workers)
            )
        finally:
            with lock:
                active -= 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    if not 1 <= peak <= 6:
        raise GovernanceError(f"Scoring scheduler observed invalid peak concurrency: {peak}")
    complete = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_before_outcome_join",
        "created_utc": _utcnow(),
        "contract": _artifact(contract_path(root)),
        "target_outcomes_present": False,
        "score_jobs": 50,
        "score_slide_rows": 4_790,
        "configured_max_workers": 6,
        "max_observed_parallel_workers": peak,
    }
    if scoring_receipt_path(root).exists():
        previous = _read_json(scoring_receipt_path(root))
        complete["created_utc"] = previous.get("created_utc")
        complete["max_observed_parallel_workers"] = previous.get("max_observed_parallel_workers")
    _write_json_once(scoring_receipt_path(root), complete)
    return {"status": "complete_and_sealed", "seal": seal_inference(root)}


def _canonical_oof_path(encoder: str, seed: int) -> Path:
    if encoder == "univ1":
        root = (
            Path("/mnt/wsl/oceanpath-hot/outputs/aim1/train/1a_pb_cap8192/univ1")
            if seed <= 44
            else Path(
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "final_v9_mil_5seed_expansion_v1_20260823/aim1_e0/train/"
                "1a_pb_cap8192/univ1"
            )
        )
    elif encoder == "virchow2_cls":
        root = (
            Path("/mnt/wsl/oceanpath-hot/outputs/aim1/train/1a_pb_cap8192/virchow2_cls")
            if seed <= 44
            else Path(
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "final_v9_mil_5seed_expansion_v1_20260823/aim3_ladders/"
                "train/e1v_virchow2_cls"
            )
        )
    else:
        raise GovernanceError(f"Unknown canonical encoder: {encoder}")
    return root / f"seed{seed}/oof_predictions.parquet"


def _validate_canonical_inherited_bundle() -> dict[str, Any]:
    parent = _artifact(FINAL_V10_RECEIPT)
    if parent["sha256"] != FINAL_V10_RECEIPT_SHA256:
        raise GovernanceError("Inherited FINAL-v10 parent receipt drifted")
    parent_value = _read_json(FINAL_V10_RECEIPT)
    if parent_value.get("status") != "SEALED_COMPLETED_RESULTS_WITH_DECLARED_NOT_RUN_ARM":
        raise GovernanceError("Inherited FINAL-v10 parent is not terminal-sealed")
    manifest_identity = _artifact(CANONICAL_MANIFEST)
    if manifest_identity["sha256"] != CANONICAL_MANIFEST_SHA256:
        raise GovernanceError("Canonical all-primary manifest drifted")
    manifest = pd.read_csv(CANONICAL_MANIFEST, low_memory=False)
    census = (
        len(manifest),
        manifest["patient_id"].astype(str).nunique(),
        int(manifest.groupby("patient_id")["target_label"].first().sum()),
    )
    if census != (CANONICAL_SLIDES, CANONICAL_PATIENTS, CANONICAL_MUTANT):
        raise GovernanceError(f"Canonical all-primary manifest census drifted: {census}")
    uni_identity = _artifact(CANONICAL_UNI_VALIDATION)
    if uni_identity["sha256"] != CANONICAL_UNI_VALIDATION_SHA256:
        raise GovernanceError("Canonical UNI validation receipt drifted")
    uni = _read_json(CANONICAL_UNI_VALIDATION)
    v2_contract_identity = _artifact(CANONICAL_V2_CONTRACT)
    v2_audit_identity = _artifact(CANONICAL_V2_AUDIT)
    if v2_contract_identity["sha256"] != CANONICAL_V2_CONTRACT_SHA256:
        raise GovernanceError("Canonical V2 contract drifted")
    if v2_audit_identity["sha256"] != CANONICAL_V2_AUDIT_SHA256:
        raise GovernanceError("Canonical V2 analysis audit drifted")
    v2_contract = _read_json(CANONICAL_V2_CONTRACT)
    v2_audit = _read_json(CANONICAL_V2_AUDIT)
    lineages: dict[str, Any] = {encoder: {} for encoder in ENCODERS}
    for encoder in ENCODERS:
        for seed in SEEDS:
            path = _canonical_oof_path(encoder, seed)
            identity = _artifact(path)
            if encoder == "univ1":
                run = (uni.get("runs") or {}).get(str(seed))
                if not isinstance(run, dict) or (run.get("artifacts") or {}).get("oof") != identity:
                    raise GovernanceError(f"Canonical UNI seed{seed} OOF is unauthenticated")
            elif seed <= 44:
                if not _contains_identity(v2_contract, identity):
                    raise GovernanceError(f"Canonical V2 seed{seed} OOF is unauthenticated")
            elif not _contains_identity(v2_audit, identity):
                raise GovernanceError(f"Canonical V2 seed{seed} OOF is unauthenticated")
            lineages[encoder][str(seed)] = identity
    return {
        "parent_final_v10": parent,
        "manifest": manifest_identity,
        "univ1_validation": uni_identity,
        "virchow2_contract": v2_contract_identity,
        "virchow2_audit": v2_audit_identity,
        "lineages": lineages,
    }


PATIENT_METADATA = (
    "cohort",
    "subcohort",
    "specimen_role",
    "k_fold",
    "msi_dmmr",
    "braf",
    "tumor_site_group",
    "site_class",
    "stage_group_major",
    "stage_class",
    "age_at_diagnosis",
    "sex",
)
DERIVED_SOURCE_METADATA = ("stage_group_major_filled", "sidedness")


def _aggregate_oof_patients(
    manifest: pd.DataFrame,
    oof_paths: Mapping[int, Path],
    *,
    encoder: str,
    analysis_family: str,
    dataset: str,
) -> pd.DataFrame:
    required = {"slide_id", "patient_id", "target_label", *PATIENT_METADATA}
    if required - set(manifest):
        raise GovernanceError(f"{dataset}: manifest lacks patient metadata")
    manifest = manifest.copy()
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest["patient_id"] = manifest["patient_id"].astype(str)
    patient: pd.DataFrame | None = None
    metadata = ["slide_id", "patient_id", "target_label", *PATIENT_METADATA]
    for seed in SEEDS:
        path = oof_paths[seed]
        oof = _validate_oof_frame(path, manifest, context=f"{dataset}/{encoder}/seed{seed}")
        joined = oof[["slide_id", "logit"]].merge(
            manifest[metadata], on="slide_id", how="inner", validate="one_to_one"
        )
        joined["logit"] = pd.to_numeric(joined["logit"], errors="raise").astype(float)
        grouped = joined.groupby("patient_id", sort=True)
        consistency = grouped[["target_label", *PATIENT_METADATA]].nunique(dropna=False)
        if (consistency != 1).any().any():
            raise GovernanceError(f"{dataset}/{encoder}: inconsistent patient metadata")
        current = grouped.agg(
            label=("target_label", "first"),
            **{column: (column, "first") for column in PATIENT_METADATA},
            n_slides=("slide_id", "size"),
            slide_roster_sha256=(
                "slide_id",
                lambda values: hashlib.sha256(
                    "\n".join(sorted(values.astype(str))).encode()
                ).hexdigest(),
            ),
            **{f"logit_seed{seed}": ("logit", "mean")},
        ).reset_index()
        if patient is None:
            patient = current
        else:
            stable = [
                "patient_id",
                "label",
                *PATIENT_METADATA,
                "n_slides",
                "slide_roster_sha256",
            ]
            if not patient[stable].equals(current[stable]):
                raise GovernanceError(f"{dataset}/{encoder}: roster changed across seeds")
            patient[f"logit_seed{seed}"] = current[f"logit_seed{seed}"].to_numpy(float)
    if patient is None:
        raise GovernanceError(f"{dataset}/{encoder}: no OOF patient table")
    seed_columns = [f"logit_seed{seed}" for seed in SEEDS]
    patient["mean_logit_5seed"] = patient[seed_columns].mean(axis=1)
    patient.insert(0, "dataset", dataset)
    patient.insert(0, "encoder", encoder)
    patient.insert(0, "analysis_family", analysis_family)
    patient["source_platt_probability"] = np.nan
    patient["exclude_neoadjuvant"] = False
    patient["exclude_ambiguous_crc15"] = False
    return patient


def _load_source_restricted_patients(campaign_root: Path) -> dict[str, pd.DataFrame]:
    training = _validate_training_bundle(campaign_root)
    manifest = pd.read_csv(Path(training["source_manifest_path"]), low_memory=False)
    return {
        encoder: _aggregate_oof_patients(
            manifest,
            {seed: oof_path(campaign_root, encoder, seed) for seed in SEEDS},
            encoder=encoder,
            analysis_family="source_restricted_tcga_surgen_oof",
            dataset="tcga_surgen_source",
        )
        for encoder in ENCODERS
    }


def _load_canonical_patients() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    identities = _validate_canonical_inherited_bundle()
    manifest = pd.read_csv(CANONICAL_MANIFEST, low_memory=False)
    frames = {
        encoder: _aggregate_oof_patients(
            manifest,
            {seed: _canonical_oof_path(encoder, seed) for seed in SEEDS},
            encoder=encoder,
            analysis_family="inherited_canonical_all_primary_oof",
            dataset="all_primary_canonical",
        )
        for encoder in ENCODERS
    }
    return frames, identities


def _open_target_outcomes_after_seal(
    campaign_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """First and only target-outcome opening point; caller must verify seal first."""
    verify_inference_seal(campaign_root)
    outcomes: dict[str, pd.DataFrame] = {}
    identities: dict[str, Any] = {}
    for key in TARGET_ORDER:
        spec = TARGETS[key]
        identity = _artifact(spec.outcome_source)
        if identity["sha256"] != spec.outcome_sha256:
            raise GovernanceError(f"{key}: target outcome source SHA drifted")
        blind = pd.read_csv(blind_path(campaign_root, key), low_memory=False)
        if key == "orion_cpht":
            master = pd.read_csv(spec.outcome_source, low_memory=False)
            rows = master.loc[
                master["cohort"].eq("Orion")
                & master["subcohort"].eq("Orion-CRC")
                & master["specimen_role"].eq("primary")
                & master["output_id"].astype(str).isin(blind["slide_id"].astype(str))
            ].copy()
            rows["slide_id"] = rows["output_id"].astype(str)
            rows["patient_id"] = rows["patient_uid"].astype(str)
            rows["target_label"] = rows["kras"].astype(str).map({"wild_type": 0, "mutant": 1})
            raw = rows
        else:
            raw = pd.read_csv(spec.outcome_source, low_memory=False)
        required = {"slide_id", "patient_id", "target_label"}
        if required - set(raw):
            raise GovernanceError(f"{key}: outcome manifest lacks exact join columns")
        joined = blind.merge(
            raw,
            on=["slide_id", "patient_id"],
            how="inner",
            suffixes=("_blind", ""),
            validate="one_to_one",
        )
        if len(joined) != spec.slides:
            raise GovernanceError(f"{key}: outcome join does not cover exact blind roster")
        labels = pd.to_numeric(joined["target_label"], errors="raise").astype(int)
        if set(labels) != {0, 1}:
            raise GovernanceError(f"{key}: target outcome is not binary")
        joined["target_label"] = labels
        consistency = joined.groupby("patient_id")["target_label"].nunique()
        patient_labels = joined.groupby("patient_id")["target_label"].first()
        census = (len(patient_labels), int(patient_labels.sum()))
        if not consistency.eq(1).all() or census != (spec.patients, spec.mutant):
            raise GovernanceError(f"{key}: target patient outcome census drifted: {census}")
        outcomes[key] = joined
        identities[key] = identity
    return outcomes, identities


def _open_source_derived_covariates_after_seal(
    campaign_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Open the frozen evaluation-only stage/sidedness join after inference.

    ``crc_final_v4.csv`` contains outcome-bearing study data, so even though
    only two clinicopathologic covariates are selected here, the file is not
    opened until the target inference seal has replayed successfully.
    """

    root = _safe_campaign_root(campaign_root)
    verify_inference_seal(root)
    identity = _artifact(DERIVED_COVARIATE_SOURCE)
    if identity != {
        "path": str(DERIVED_COVARIATE_SOURCE.resolve()),
        "sha256": DERIVED_COVARIATE_SOURCE_SHA256,
        "size_bytes": DERIVED_COVARIATE_SOURCE_SIZE,
    }:
        raise GovernanceError("Frozen E1a/E1d derived-covariate source drifted")
    downstream_contract = _load_contract(root, deep=False)
    source = _validate_source_manifest(
        Path(str(downstream_contract["training"]["source_manifest_path"]))
    )
    source_ids = set(source["patient_id"].astype(str))
    master = pd.read_csv(DERIVED_COVARIATE_SOURCE, low_memory=False)
    required = {
        "patient_uid",
        "specimen_role",
        "stage_group_major_filled",
        "sidedness",
    }
    if required - set(master):
        raise GovernanceError("Frozen derived-covariate source lacks required columns")
    master = master.copy()
    master["patient_uid"] = master["patient_uid"].astype(str)
    block = master.loc[
        master["patient_uid"].isin(source_ids)
        & master["specimen_role"].astype(str).str.casefold().eq("primary"),
        ["patient_uid", "stage_group_major_filled", "sidedness"],
    ].copy()
    consistency = block.groupby("patient_uid")[["stage_group_major_filled", "sidedness"]].nunique(
        dropna=False
    )
    if (consistency > 1).any().any():
        raise GovernanceError("Derived stage/sidedness varies within a source patient")
    covariates = (
        block.sort_values("patient_uid", kind="mergesort")
        .drop_duplicates("patient_uid")
        .rename(columns={"patient_uid": "patient_id"})
        .reset_index(drop=True)
    )
    if len(covariates) != SOURCE_PATIENTS or set(covariates["patient_id"]) != source_ids:
        raise GovernanceError("Derived stage/sidedness does not cover exact source patients")
    check = (
        source.sort_values("slide_id", kind="mergesort")
        .drop_duplicates("patient_id")
        .merge(covariates, on="patient_id", how="inner", validate="one_to_one")
    )
    labels = pd.to_numeric(check["target_label"], errors="raise").astype(int)
    census = {
        "G_stage_known_derived": check["stage_group_major_filled"]
        .astype(str)
        .isin(["I", "II", "III", "IV"]),
        "G_stage_known_frozen": check["stage_group_major"]
        .astype(str)
        .isin(["I", "II", "III", "IV"]),
        "H_stage_iv": check["stage_group_major_filled"].astype(str).eq("IV"),
        "I_right_proximal": check["sidedness"].astype(str).eq("right"),
        "J_left_distal": check["sidedness"].astype(str).eq("left"),
        "K_transverse": check["sidedness"].astype(str).eq("transverse"),
    }
    expected = {
        "G_stage_known_derived": [1_060, 422],
        "G_stage_known_frozen": [895, 344],
        "H_stage_iv": [120, 54],
        "I_right_proximal": [212, 99],
        "J_left_distal": [573, 212],
        "K_transverse": [42, 10],
    }
    observed = {key: [int(mask.sum()), int(labels.loc[mask].sum())] for key, mask in census.items()}
    if observed != expected:
        raise GovernanceError(f"Derived E1a covariate census drifted: {observed}")
    return covariates, {"artifact": identity, "source_patient_census": observed}


def _aggregate_target_patients(
    campaign_root: Path,
    outcomes: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, pd.DataFrame]]:
    result: dict[str, dict[str, pd.DataFrame]] = {encoder: {} for encoder in ENCODERS}
    for encoder in ENCODERS:
        for target in TARGET_ORDER:
            metadata = outcomes[target].copy()
            seed_slide: list[pd.DataFrame] = []
            for seed in SEEDS:
                scores = _validate_cached_score(campaign_root, encoder, target, seed)
                if scores is None:
                    raise GovernanceError(
                        f"Missing sealed target score: {encoder}/{target}/seed{seed}"
                    )
                seed_slide.append(
                    scores[["slide_id", "logit"]].rename(columns={"logit": f"logit_seed{seed}"})
                )
            slide = seed_slide[0]
            for current in seed_slide[1:]:
                slide = slide.merge(current, on="slide_id", how="inner", validate="one_to_one")
            if len(slide) != TARGETS[target].slides:
                raise GovernanceError(f"{encoder}/{target}: five-seed slide roster drifted")
            slide["mean_logit_5seed"] = slide[[f"logit_seed{seed}" for seed in SEEDS]].mean(axis=1)
            joined = slide.merge(metadata, on="slide_id", how="inner", validate="one_to_one")
            aggregation: dict[str, tuple[str, str]] = {
                "label": ("target_label", "first"),
                "n_slides": ("slide_id", "size"),
                "slide_roster_sha256": (
                    "slide_id",
                    lambda values: hashlib.sha256(
                        "\n".join(sorted(values.astype(str))).encode()
                    ).hexdigest(),
                ),
                **{f"logit_seed{seed}": (f"logit_seed{seed}", "mean") for seed in SEEDS},
                "mean_logit_5seed": ("mean_logit_5seed", "mean"),
            }
            for column in PATIENT_METADATA:
                if column in joined:
                    aggregation[column] = (column, "first")
            for column in ("exclude_neoadjuvant", "exclude_ambiguous_crc15"):
                if column in joined:
                    aggregation[column] = (column, "max")
            patient = joined.groupby("patient_id", sort=True).agg(**aggregation).reset_index()
            if len(patient) != TARGETS[target].patients:
                raise GovernanceError(f"{encoder}/{target}: target patient census drifted")
            for column in PATIENT_METADATA:
                if column not in patient:
                    patient[column] = -1 if column == "k_fold" else pd.NA
            for column in ("exclude_neoadjuvant", "exclude_ambiguous_crc15"):
                if column not in patient:
                    patient[column] = False
            patient.insert(0, "dataset", target)
            patient.insert(0, "encoder", encoder)
            patient.insert(0, "analysis_family", "target_refit_zero_shot_or_sensitivity")
            patient["source_platt_probability"] = np.nan
            result[encoder][target] = patient
    return result


def _fit_source_platt(frame: pd.DataFrame) -> dict[str, float]:
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int).to_numpy()
    logits = frame["mean_logit_5seed"].to_numpy(float)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10_000)
    model.fit(logits.reshape(-1, 1), labels)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def _apply_platt(logits: np.ndarray, calibration: Mapping[str, float]) -> np.ndarray:
    linear = float(calibration["intercept"]) + float(calibration["slope"]) * np.asarray(
        logits, dtype=float
    )
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -50.0, 50.0)))


def _validate_patient_output_table(frame: pd.DataFrame) -> None:
    key_columns = ["analysis_family", "encoder", "dataset", "patient_id"]
    if frame[key_columns].duplicated().any() or set(frame["encoder"].astype(str)) != set(ENCODERS):
        raise GovernanceError("Patient output key/encoder roster drifted")
    expected: dict[tuple[str, str], tuple[int, int]] = {
        ("inherited_canonical_all_primary_oof", "all_primary_canonical"): (
            CANONICAL_PATIENTS,
            CANONICAL_MUTANT,
        ),
        ("source_restricted_tcga_surgen_oof", "tcga_surgen_source"): (
            SOURCE_PATIENTS,
            SOURCE_MUTANT,
        ),
        **{
            ("target_refit_zero_shot_or_sensitivity", key): (
                TARGETS[key].patients,
                TARGETS[key].mutant,
            )
            for key in TARGET_ORDER
        },
    }
    for encoder in ENCODERS:
        for (family, dataset), (patients, mutant) in expected.items():
            block = frame.loc[
                frame["encoder"].eq(encoder)
                & frame["analysis_family"].eq(family)
                & frame["dataset"].eq(dataset)
            ]
            labels = pd.to_numeric(block["label"], errors="raise").astype(int)
            if len(block) != patients or int(labels.sum()) != mutant or set(labels) != {0, 1}:
                raise GovernanceError(
                    f"Patient output census drifted: {encoder}/{family}/{dataset}"
                )
        source = frame.loc[
            frame["encoder"].eq(encoder)
            & frame["analysis_family"].eq("source_restricted_tcga_surgen_oof")
        ].copy()
        observed_restrictions = {
            key: (
                int(_e1a_mask(source, key).sum()),
                int(source.loc[_e1a_mask(source, key), "label"].astype(int).sum()),
            )
            for key in E1A_SETS
        }
        if observed_restrictions != E1A_EXPECTED_CENSUS:
            raise GovernanceError(
                f"Patient output E1a covariate census drifted: {encoder}/{observed_restrictions}"
            )
    seed_columns = [f"logit_seed{seed}" for seed in SEEDS]
    observed_mean = frame[seed_columns].mean(axis=1).to_numpy(float)
    if not np.array_equal(observed_mean, frame["mean_logit_5seed"].to_numpy(float)):
        raise GovernanceError("Patient five-seed native-logit mean drifted")
    primary = ~frame["dataset"].isin(["rih_metastatic", "sr1482_metastatic"])
    if (
        not frame.loc[primary, "specimen_role"].astype(str).str.casefold().eq("primary").all()
        or not frame.loc[~primary, "specimen_role"]
        .astype(str)
        .str.casefold()
        .eq("metastatic")
        .all()
    ):
        raise GovernanceError("Patient specimen-role boundary drifted")


def build_patient_table(
    campaign_root: Path,
    outcomes: Mapping[str, pd.DataFrame],
    source_derived_covariates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, pd.DataFrame]]]:
    source = _load_source_restricted_patients(campaign_root)
    canonical, inherited = _load_canonical_patients()
    targets = _aggregate_target_patients(campaign_root, outcomes)
    expected_covariate_columns = ["patient_id", *DERIVED_SOURCE_METADATA]
    if (
        list(source_derived_covariates.columns) != expected_covariate_columns
        or len(source_derived_covariates) != SOURCE_PATIENTS
        or source_derived_covariates["patient_id"].astype(str).duplicated().any()
    ):
        raise GovernanceError("Source derived-covariate join schema/census drifted")
    for encoder in ENCODERS:
        source[encoder] = source[encoder].merge(
            source_derived_covariates,
            on="patient_id",
            how="inner",
            validate="one_to_one",
        )
        if len(source[encoder]) != SOURCE_PATIENTS:
            raise GovernanceError(f"{encoder}: derived-covariate join lost source patients")
    calibrations: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []
    for encoder in ENCODERS:
        calibration = _fit_source_platt(source[encoder])
        calibrations[encoder] = {
            **calibration,
            "fit_population": "TCGA+SurGen primary OOF only",
            "target_outcomes_used": False,
            "model_selection_used": False,
        }
        source[encoder]["source_platt_probability"] = _apply_platt(
            source[encoder]["mean_logit_5seed"].to_numpy(float), calibration
        )
        for target in TARGET_ORDER:
            targets[encoder][target]["source_platt_probability"] = _apply_platt(
                targets[encoder][target]["mean_logit_5seed"].to_numpy(float), calibration
            )
        frames.extend([canonical[encoder], source[encoder]])
        frames.extend(targets[encoder][target] for target in TARGET_ORDER)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    ordered = [
        "analysis_family",
        "encoder",
        "dataset",
        "patient_id",
        "label",
        *PATIENT_METADATA,
        *DERIVED_SOURCE_METADATA,
        "n_slides",
        "slide_roster_sha256",
        *(f"logit_seed{seed}" for seed in SEEDS),
        "mean_logit_5seed",
        "source_platt_probability",
        "exclude_neoadjuvant",
        "exclude_ambiguous_crc15",
    ]
    combined = (
        combined[ordered]
        .sort_values(["analysis_family", "encoder", "dataset", "patient_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    expected_rows = 2 * (
        CANONICAL_PATIENTS + SOURCE_PATIENTS + sum(TARGETS[key].patients for key in TARGET_ORDER)
    )
    if len(combined) != expected_rows or expected_rows != 6_342:
        raise GovernanceError(f"Patient-native-logit row census drifted: {len(combined)}")
    logits = combined[[*(f"logit_seed{seed}" for seed in SEEDS), "mean_logit_5seed"]]
    if not np.isfinite(logits.to_numpy(float)).all():
        raise GovernanceError("Patient table contains non-finite native logits")
    canonical_mask = combined["analysis_family"].eq("inherited_canonical_all_primary_oof")
    probabilities = pd.to_numeric(combined["source_platt_probability"], errors="coerce")
    if (
        probabilities.loc[canonical_mask].notna().any()
        or not np.isfinite(probabilities.loc[~canonical_mask].to_numpy(float)).all()
    ):
        raise GovernanceError("Source-Platt applicability/null boundary drifted")
    _validate_patient_output_table(combined)
    return combined, {"source_platt": calibrations, "inherited_canonical": inherited}, targets


def _binary(values: Sequence[Any], *, context: str) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise GovernanceError(f"{context}: labels must be finite binary integers")
    labels = numeric.astype(int)
    if set(labels) != {0, 1}:
        raise GovernanceError(f"{context}: both outcome classes are required")
    return labels


def _rank_points(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _probability_points(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
    }


def _named_rng(name: str, *, seed: int = BOOTSTRAP_SEED) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))


def stratified_bootstrap_indices(
    labels: np.ndarray,
    strata: Sequence[str],
    *,
    n_bootstrap: int,
    stream: str,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    strata = np.asarray(strata, dtype=str)
    if n_bootstrap < 1 or len(labels) != len(strata) or set(labels) != {0, 1}:
        raise GovernanceError("Invalid patient-stratified bootstrap request")
    keys = np.asarray(
        [f"{stratum}\x1f{label}" for stratum, label in zip(strata, labels, strict=True)]
    )
    rng = _named_rng(stream)
    groups = [np.flatnonzero(keys == key) for key in sorted(set(keys))]
    return np.concatenate(
        [rng.choice(group, size=(n_bootstrap, len(group)), replace=True) for group in groups],
        axis=1,
    ).astype(np.int64, copy=False)


def _rank_draws(
    labels: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    auroc = np.empty(len(indices), dtype=np.float64)
    auprc = np.empty(len(indices), dtype=np.float64)
    for draw, index in enumerate(indices):
        auroc[draw] = roc_auc_score(labels[index], scores[index])
        auprc[draw] = average_precision_score(labels[index], scores[index])
    return auroc, auprc


def _probability_draws(
    labels: np.ndarray,
    probabilities: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    brier = np.mean((probabilities[indices] - labels[indices]) ** 2, axis=1)
    losses = -(
        labels[indices] * np.log(probabilities[indices])
        + (1 - labels[indices]) * np.log(1 - probabilities[indices])
    )
    return brier.astype(np.float64), losses.mean(axis=1).astype(np.float64)


def _ci(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _aligned_encoder_frames(
    frames: Mapping[str, pd.DataFrame], *, context: str
) -> dict[str, pd.DataFrame]:
    aligned = {
        encoder: frames[encoder].sort_values("patient_id", kind="mergesort").reset_index(drop=True)
        for encoder in ENCODERS
    }
    keys = ["patient_id", "label", "cohort", "subcohort", "specimen_role"]
    if not aligned[ENCODERS[0]][keys].equals(aligned[ENCODERS[1]][keys]):
        raise GovernanceError(f"{context}: encoder patient rosters are not exactly aligned")
    return aligned


def _performance_panel(
    frames: Mapping[str, pd.DataFrame],
    populations: Mapping[str, np.ndarray],
    *,
    prefix: str,
    n_bootstrap: int,
    arrays: dict[str, np.ndarray],
    include_probability_metrics: bool,
) -> dict[str, Any]:
    aligned = _aligned_encoder_frames(frames, context=prefix)
    result: dict[str, Any] = {}
    base = aligned[ENCODERS[0]]
    index_cache: dict[bytes, np.ndarray] = {}
    for population, mask_value in populations.items():
        mask = np.asarray(mask_value, dtype=bool)
        if len(mask) != len(base) or not mask.any():
            raise GovernanceError(f"{prefix}/{population}: invalid population mask")
        reference = base.loc[mask].reset_index(drop=True)
        labels = _binary(reference["label"], context=f"{prefix}/{population}")
        strata = reference["subcohort"].fillna("unknown").astype(str).to_numpy()
        mask_key = mask.tobytes()
        if mask_key not in index_cache:
            index_cache[mask_key] = stratified_bootstrap_indices(
                labels,
                strata,
                n_bootstrap=n_bootstrap,
                stream=f"{prefix}:{population}",
            )
        indices = index_cache[mask_key]
        result[population] = {}
        for encoder in ENCODERS:
            frame = aligned[encoder].loc[mask].reset_index(drop=True)
            if not frame["patient_id"].equals(reference["patient_id"]):
                raise GovernanceError(f"{prefix}/{population}: shared encoder roster drifted")
            scores = frame["mean_logit_5seed"].to_numpy(float)
            points = _rank_points(labels, scores)
            draw_auroc, draw_auprc = _rank_draws(labels, scores, indices)
            key = f"{prefix}__{population}__{encoder}"
            arrays[f"{key}__auroc"] = draw_auroc
            arrays[f"{key}__auprc"] = draw_auprc
            block: dict[str, Any] = {
                "patients": int(len(frame)),
                "mutant": int(labels.sum()),
                "wild_type": int((labels == 0).sum()),
                "auroc": points["auroc"],
                "auroc_ci95": _ci(draw_auroc),
                "auprc": points["auprc"],
                "auprc_ci95": _ci(draw_auprc),
                "per_seed_descriptive": {
                    str(seed): _rank_points(labels, frame[f"logit_seed{seed}"].to_numpy(float))
                    for seed in SEEDS
                },
                "shared_encoder_bootstrap_indices": True,
            }
            if include_probability_metrics:
                probabilities = frame["source_platt_probability"].to_numpy(float)
                if not np.isfinite(probabilities).all():
                    raise GovernanceError(f"{prefix}/{population}: missing source-Platt scores")
                probability_points = _probability_points(labels, probabilities)
                brier, loss = _probability_draws(labels, probabilities, indices)
                arrays[f"{key}__brier"] = brier
                arrays[f"{key}__log_loss"] = loss
                block.update(
                    {
                        **probability_points,
                        "brier_ci95": _ci(brier),
                        "log_loss_ci95": _ci(loss),
                    }
                )
            result[population][encoder] = block
    return result


def _source_encoder_contrast_panel(
    source_performance: Mapping[str, Any],
    *,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Paired Virchow2-minus-UNI contrasts on identical source patients.

    The component draw arrays were produced from the same patient bootstrap
    indices by ``_performance_panel``.  Subtracting them draw-by-draw retains
    the encoder correlation and never treats model seeds as inference units.
    """

    result: dict[str, Any] = {}
    for population, encoders in source_performance.items():
        uni = encoders["univ1"]
        v2 = encoders["virchow2_cls"]
        if (uni["patients"], uni["mutant"]) != (v2["patients"], v2["mutant"]):
            raise GovernanceError(f"Source E0 encoder roster drifted: {population}")
        metrics: dict[str, Any] = {}
        for metric in ("auroc", "auprc"):
            left = arrays[f"aim1_source_restricted_e0__{population}__univ1__{metric}"]
            right = arrays[f"aim1_source_restricted_e0__{population}__virchow2_cls__{metric}"]
            delta = right - left
            arrays[
                f"aim1_source_restricted_e0__{population}__encoder_delta_v2_minus_uni__{metric}"
            ] = delta
            metrics[metric] = {
                "univ1": float(uni[metric]),
                "univ1_ci95": [float(value) for value in uni[f"{metric}_ci95"]],
                "virchow2_cls": float(v2[metric]),
                "virchow2_cls_ci95": [float(value) for value in v2[f"{metric}_ci95"]],
                "delta_virchow2_cls_minus_univ1": float(v2[metric] - uni[metric]),
                "delta_ci95": _ci(delta),
            }
        result[population] = {
            "patients": int(uni["patients"]),
            "mutant": int(uni["mutant"]),
            "comparison": "virchow2_cls minus univ1",
            "shared_patient_bootstrap": True,
            "metrics": metrics,
        }
    expected = ["pooled_source", "tcga", "sr386", "sr1482", "surgen", "tcga_surgen"]
    if list(result) != expected:
        raise GovernanceError(
            f"Source E0 encoder-contrast population roster drifted: {list(result)}"
        )
    return result


def _canonical_population_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    cohort = frame["cohort"].astype(str)
    subcohort = frame["subcohort"].astype(str)
    return {
        "pooled_all_primary": np.ones(len(frame), dtype=bool),
        "tcga": cohort.eq("TCGA").to_numpy(),
        "sr386": subcohort.eq("SR386").to_numpy(),
        "sr1482": subcohort.eq("SR1482").to_numpy(),
        "surgen": cohort.eq("SurGen").to_numpy(),
        "tcga_surgen": cohort.isin(["TCGA", "SurGen"]).to_numpy(),
    }


def _source_population_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    cohort = frame["cohort"].astype(str)
    subcohort = frame["subcohort"].astype(str)
    all_rows = np.ones(len(frame), dtype=bool)
    return {
        "pooled_source": all_rows,
        "tcga": cohort.eq("TCGA").to_numpy(),
        "sr386": subcohort.eq("SR386").to_numpy(),
        "sr1482": subcohort.eq("SR1482").to_numpy(),
        "surgen": cohort.eq("SurGen").to_numpy(),
        "tcga_surgen": all_rows.copy(),
    }


def _target_performance(
    target_frames: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    n_bootstrap: int,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in TARGET_ORDER:
        frames = {encoder: target_frames[encoder][target] for encoder in ENCODERS}
        panel = _performance_panel(
            frames,
            {target: np.ones(len(frames[ENCODERS[0]]), dtype=bool)},
            prefix="aim2",
            n_bootstrap=n_bootstrap,
            arrays=arrays,
            include_probability_metrics=True,
        )[target]
        result[target] = {
            "family_exposure": TARGETS[target].family_exposure,
            "inferential_role": TARGETS[target].inferential_role,
            "target_calibration": False,
            "target_model_selection": False,
            "encoders": panel,
        }
    return result


def _rih_disjoint_role_contrast(
    target_frames: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    n_bootstrap: int,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    primary_ref = target_frames[ENCODERS[0]]["rih_primary"]
    metastatic_ref = target_frames[ENCODERS[0]]["rih_metastatic"]
    overlap = sorted(set(primary_ref["patient_id"]) & set(metastatic_ref["patient_id"]))
    if len(overlap) != 8:
        raise GovernanceError(f"RIH primary/metastatic overlap drifted: {len(overlap)}")
    keep_p = ~primary_ref["patient_id"].isin(overlap)
    keep_m = ~metastatic_ref["patient_id"].isin(overlap)
    p_ref = primary_ref.loc[keep_p].sort_values("patient_id").reset_index(drop=True)
    m_ref = metastatic_ref.loc[keep_m].sort_values("patient_id").reset_index(drop=True)
    yp = _binary(p_ref["label"], context="RIH disjoint primary")
    ym = _binary(m_ref["label"], context="RIH disjoint metastatic")
    if (len(p_ref), int(yp.sum()), len(m_ref), int(ym.sum())) != (145, 65, 77, 33):
        raise GovernanceError("RIH disjoint role census drifted")
    ip = stratified_bootstrap_indices(
        yp,
        p_ref["subcohort"].fillna("RIH").astype(str),
        n_bootstrap=n_bootstrap,
        stream="aim2:rih_disjoint:primary",
    )
    im = stratified_bootstrap_indices(
        ym,
        m_ref["subcohort"].fillna("RIH").astype(str),
        n_bootstrap=n_bootstrap,
        stream="aim2:rih_disjoint:metastatic",
    )
    encoders: dict[str, Any] = {}
    for encoder in ENCODERS:
        primary = target_frames[encoder]["rih_primary"].loc[keep_p].sort_values("patient_id")
        metastatic = target_frames[encoder]["rih_metastatic"].loc[keep_m].sort_values("patient_id")
        sp = primary["mean_logit_5seed"].to_numpy(float)
        sm = metastatic["mean_logit_5seed"].to_numpy(float)
        p_auc, p_ap = _rank_draws(yp, sp, ip)
        m_auc, m_ap = _rank_draws(ym, sm, im)
        delta_auc = m_auc - p_auc
        delta_ap = m_ap - p_ap
        key = f"aim2__rih_disjoint_role__{encoder}"
        arrays[f"{key}__primary__auroc"] = p_auc
        arrays[f"{key}__primary__auprc"] = p_ap
        arrays[f"{key}__metastatic__auroc"] = m_auc
        arrays[f"{key}__metastatic__auprc"] = m_ap
        arrays[f"{key}__delta_auroc"] = delta_auc
        arrays[f"{key}__delta_auprc"] = delta_ap
        point_p = _rank_points(yp, sp)
        point_m = _rank_points(ym, sm)
        encoders[encoder] = {
            "primary": {
                "patients": 145,
                "mutant": 65,
                **point_p,
                "auroc_ci95": _ci(p_auc),
                "auprc_ci95": _ci(p_ap),
            },
            "metastatic": {
                "patients": 77,
                "mutant": 33,
                **point_m,
                "auroc_ci95": _ci(m_auc),
                "auprc_ci95": _ci(m_ap),
            },
            "delta_metastatic_minus_primary": {
                "auroc": point_m["auroc"] - point_p["auroc"],
                "auroc_ci95": _ci(delta_auc),
                "auprc": point_m["auprc"] - point_p["auprc"],
                "auprc_ci95": _ci(delta_ap),
            },
        }
    return {
        "overlap_patients_excluded_from_both_roles": 8,
        "primary_population": {"patients": 145, "mutant": 65},
        "metastatic_population": {"patients": 77, "mutant": 33},
        "roles_are_disjoint_not_paired": True,
        "family_exposure": "family_naive",
        "encoders": encoders,
    }


E1A_SETS = (
    "A_all_primary",
    "A_complete",
    "B_mss",
    "C_braf_wt",
    "D_mss_braf_wt",
    "E_colon",
    "F_rectum",
    "G_stage_known_derived",
    "G_stage_known_frozen",
    "H_stage_iv",
    "I_right_proximal",
    "J_left_distal",
    "K_transverse",
)
E1A_EXPECTED_CENSUS = {
    "A_all_primary": (1_239, 501),
    "A_complete": (1_158, 459),
    "B_mss": (1_071, 452),
    "C_braf_wt": (1_039, 467),
    "D_mss_braf_wt": (955, 424),
    "E_colon": (829, 343),
    "F_rectum": (397, 151),
    "G_stage_known_derived": (1_060, 422),
    "G_stage_known_frozen": (895, 344),
    "H_stage_iv": (120, 54),
    "I_right_proximal": (212, 99),
    "J_left_distal": (573, 212),
    "K_transverse": (42, 10),
}


def _e1a_mask(frame: pd.DataFrame, key: str) -> np.ndarray:
    if key == "A_all_primary":
        return np.ones(len(frame), dtype=bool)
    if key == "A_complete":
        return (
            frame["msi_dmmr"].isin(["MSS/pMMR", "MSI/dMMR"])
            & frame["braf"].isin(["mutant", "wild_type"])
        ).to_numpy()
    if key == "B_mss":
        return frame["msi_dmmr"].eq("MSS/pMMR").fillna(False).to_numpy()
    if key == "C_braf_wt":
        return frame["braf"].eq("wild_type").fillna(False).to_numpy()
    if key == "D_mss_braf_wt":
        return (
            (frame["msi_dmmr"].eq("MSS/pMMR") & frame["braf"].eq("wild_type"))
            .fillna(False)
            .to_numpy()
        )
    if key == "E_colon":
        return frame["tumor_site_group"].eq("Colon").fillna(False).to_numpy()
    if key == "F_rectum":
        return frame["tumor_site_group"].eq("Rectum").fillna(False).to_numpy()
    if key == "G_stage_known_derived":
        return (
            frame["stage_group_major_filled"].astype(str).isin(["I", "II", "III", "IV"]).to_numpy()
        )
    if key == "G_stage_known_frozen":
        return frame["stage_group_major"].astype(str).isin(["I", "II", "III", "IV"]).to_numpy()
    if key == "H_stage_iv":
        return frame["stage_group_major_filled"].astype(str).eq("IV").to_numpy()
    if key == "I_right_proximal":
        return frame["sidedness"].astype(str).eq("right").to_numpy()
    if key == "J_left_distal":
        return frame["sidedness"].astype(str).eq("left").to_numpy()
    if key == "K_transverse":
        return frame["sidedness"].astype(str).eq("transverse").to_numpy()
    raise GovernanceError(f"Unknown E1a restriction: {key}")


def _restricted_rank_draws(
    labels: np.ndarray,
    scores: np.ndarray,
    base_indices: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    auroc = np.empty(len(base_indices), dtype=np.float64)
    auprc = np.empty(len(base_indices), dtype=np.float64)
    for draw, index in enumerate(base_indices):
        selected = index[mask[index]]
        if len(selected) == 0 or len(np.unique(labels[selected])) < 2:
            raise GovernanceError("E1a shared resample produced an undefined restriction metric")
        auroc[draw] = roc_auc_score(labels[selected], scores[selected])
        auprc[draw] = average_precision_score(labels[selected], scores[selected])
    return auroc, auprc


def _e1a_panel(
    source_frames: Mapping[str, pd.DataFrame],
    *,
    n_bootstrap: int,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    aligned = _aligned_encoder_frames(source_frames, context="aim1/e1a")
    base = aligned[ENCODERS[0]]
    labels = _binary(base["label"], context="Aim1 E1a")
    indices = stratified_bootstrap_indices(
        labels,
        base["subcohort"].astype(str),
        n_bootstrap=n_bootstrap,
        stream="aim1:e1a:shared_A",
    )
    masks = {key: _e1a_mask(base, key) for key in E1A_SETS}
    if not all(mask.sum() > 1 for mask in masks.values()):
        raise GovernanceError("E1a contains an empty restriction")
    observed_census = {
        key: (int(mask.sum()), int(labels[mask].sum())) for key, mask in masks.items()
    }
    if observed_census != E1A_EXPECTED_CENSUS:
        raise GovernanceError(f"E1a restriction census drifted: {observed_census}")
    result: dict[str, Any] = {
        "design": "fixed-model restrictions of TCGA+SurGen source OOF; no refit/rethreshold",
        "delta_sign": "A_all_primary minus restriction; positive means restriction scored lower",
        "shared_resample_from_A": True,
        "population_census": observed_census,
        "encoders": {},
    }
    for encoder in ENCODERS:
        frame = aligned[encoder]
        scores = frame["mean_logit_5seed"].to_numpy(float)
        full_auc, full_ap = _restricted_rank_draws(labels, scores, indices, masks["A_all_primary"])
        encoder_sets: dict[str, Any] = {}
        for key in E1A_SETS:
            mask = masks[key]
            subset_labels = labels[mask]
            if set(subset_labels) != {0, 1}:
                raise GovernanceError(f"E1a {key} lacks both classes")
            subset_scores = scores[mask]
            draws_auc, draws_ap = _restricted_rank_draws(labels, scores, indices, mask)
            delta_auc = full_auc - draws_auc
            delta_ap = full_ap - draws_ap
            prefix = f"aim1__e1a__{key}__{encoder}"
            arrays[f"{prefix}__auroc"] = draws_auc
            arrays[f"{prefix}__auprc"] = draws_ap
            arrays[f"{prefix}__delta_A_minus_subset_auroc"] = delta_auc
            arrays[f"{prefix}__delta_A_minus_subset_auprc"] = delta_ap
            point = _rank_points(subset_labels, subset_scores)
            full_point = _rank_points(labels, scores)
            encoder_sets[key] = {
                "patients": int(mask.sum()),
                "mutant": int(subset_labels.sum()),
                "auroc": point["auroc"],
                "auroc_ci95": _ci(draws_auc),
                "auprc": point["auprc"],
                "auprc_ci95": _ci(draws_ap),
                "delta_A_minus_subset_auroc": full_point["auroc"] - point["auroc"],
                "delta_A_minus_subset_auroc_ci95": _ci(delta_auc),
                "delta_A_minus_subset_auprc": full_point["auprc"] - point["auprc"],
                "delta_A_minus_subset_auprc_ci95": _ci(delta_ap),
            }
        result["encoders"][encoder] = encoder_sets
    return result


def _standardized_auroc(
    frame: pd.DataFrame,
    strata: Sequence[str],
    weights: Mapping[str, float],
    *,
    score_column: str = "mean_logit_5seed",
) -> float:
    total = 0.0
    weight_sum = 0.0
    for stratum in strata:
        block = frame.loc[frame["stratum"].eq(stratum)]
        labels = block["label"].to_numpy(int)
        if len(block) and len(np.unique(labels)) == 2:
            total += float(weights[stratum]) * roc_auc_score(labels, block[score_column])
            weight_sum += float(weights[stratum])
    return total / weight_sum if weight_sum else float("nan")


def _usable_standardization_strata(a: pd.DataFrame, d: pd.DataFrame) -> list[str]:
    strata: list[str] = []
    for stratum in sorted(set(a["stratum"].astype(str))):
        ba = a.loc[a["stratum"].eq(stratum)]
        bd = d.loc[d["stratum"].eq(stratum)]
        ok = (
            len(ba) >= 10
            and len(bd) >= 10
            and int(ba["label"].sum()) >= 3
            and int((ba["label"] == 0).sum()) >= 3
            and int(bd["label"].sum()) >= 3
            and int((bd["label"] == 0).sum()) >= 3
        )
        if ok:
            strata.append(stratum)
    if not strata:
        raise GovernanceError("E1a-S has no common estimable strata")
    return strata


def _e1a_standardized(
    source_frames: Mapping[str, pd.DataFrame],
    *,
    n_bootstrap: int,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    aligned = _aligned_encoder_frames(source_frames, context="aim1/e1a_s")
    base = aligned[ENCODERS[0]].copy()
    base["site2"] = base["tumor_site_group"].where(
        base["tumor_site_group"].isin(["Colon", "Rectum"])
    )
    base = base.loc[base["site2"].notna()].copy()
    base["stratum"] = base["subcohort"].astype(str) + "|" + base["site2"].astype(str)
    result: dict[str, Any] = {
        "standardization": "direct, common subcohort x colon/rectum support",
        "weights": "reference A composition over common support, identical for A and D",
        "bootstrap": "shared A resample within stratum x label; D remains nested",
        "minimum_cell": 10,
        "minimum_per_class": 3,
        "references": {},
    }
    for reference in ("A_all_primary", "A_complete"):
        mask_a = _e1a_mask(base, reference)
        mask_d = _e1a_mask(base, "D_mss_braf_wt")
        if np.any(mask_d & ~mask_a):
            raise GovernanceError(f"E1a-S D is not nested within {reference}")
        a_ref = base.loc[mask_a].copy()
        d_ref = base.loc[mask_d].copy()
        strata = _usable_standardization_strata(a_ref, d_ref)
        counts = a_ref.loc[a_ref["stratum"].isin(strata)].groupby("stratum").size()
        weights = (counts / counts.sum()).to_dict()
        ref_result: dict[str, Any] = {"strata": strata, "weights": weights, "encoders": {}}
        for encoder in ENCODERS:
            frame = aligned[encoder].loc[base.index].copy()
            frame["site2"] = base["site2"]
            frame["stratum"] = base["stratum"]
            a = frame.loc[mask_a].copy()
            d = frame.loc[mask_d].copy()
            std_a = _standardized_auroc(a, strata, weights)
            std_d = _standardized_auroc(d, strata, weights)
            d_members = pd.Series(a.index.isin(d.index), index=a.index)
            cells = [
                a.loc[a["stratum"].eq(stratum) & a["label"].eq(label)].index.to_numpy()
                for stratum in strata
                for label in (0, 1)
            ]
            rng = _named_rng(f"aim1:e1a_s:{reference}")
            draw_a = np.empty(n_bootstrap, dtype=np.float64)
            draw_d = np.empty(n_bootstrap, dtype=np.float64)
            for draw in range(n_bootstrap):
                sampled = np.concatenate(
                    [rng.choice(cell, len(cell), replace=True) for cell in cells if len(cell)]
                )
                boot_a = a.loc[sampled]
                boot_d = boot_a.loc[d_members.loc[sampled].to_numpy()]
                draw_a[draw] = _standardized_auroc(boot_a, strata, weights)
                draw_d[draw] = _standardized_auroc(boot_d, strata, weights)
            if not np.isfinite(draw_a).all() or not np.isfinite(draw_d).all():
                raise GovernanceError(f"E1a-S {reference}/{encoder}: undefined bootstrap")
            delta = draw_d - draw_a
            prefix = f"aim1__e1a_s__{reference}__{encoder}"
            arrays[f"{prefix}__standardized_A_auroc"] = draw_a
            arrays[f"{prefix}__standardized_D_auroc"] = draw_d
            arrays[f"{prefix}__delta_D_minus_A_auroc"] = delta
            ref_result["encoders"][encoder] = {
                "patients_A": int(len(a)),
                "mutant_A": int(a["label"].sum()),
                "patients_D": int(len(d)),
                "mutant_D": int(d["label"].sum()),
                "standardized_A_auroc": std_a,
                "standardized_A_auroc_ci95": _ci(draw_a),
                "standardized_D_auroc": std_d,
                "standardized_D_auroc_ci95": _ci(draw_d),
                "delta_D_minus_A_auroc": std_d - std_a,
                "delta_D_minus_A_auroc_ci95": _ci(delta),
            }
        result["references"][reference] = ref_result
    return result


def _why_d(source_frames: Mapping[str, pd.DataFrame], *, n_bootstrap: int) -> dict[str, Any]:
    from oceanpath.aim1.cli import e1a_whyd as whyd

    original_rand, original_boot = whyd.N_RAND, whyd.N_BOOT
    whyd.N_RAND = n_bootstrap
    whyd.N_BOOT = n_bootstrap
    try:
        result: dict[str, Any] = {
            "status": "DERIVED_WHERE_VALID",
            "random_restriction_draws": n_bootstrap,
            "patient_bootstrap_draws": n_bootstrap,
            "encoders": {},
        }
        for encoder in ENCODERS:
            frame = source_frames[encoder].copy()
            complete = _e1a_mask(frame, "A_complete")
            frame = frame.loc[complete].copy()
            frame["mean_logit"] = frame["mean_logit_5seed"].astype(float)
            frame["D"] = _e1a_mask(frame, "D_mss_braf_wt")
            frame["mut"] = frame["label"].astype(bool)
            frame["site2"] = frame["tumor_site_group"].where(
                frame["tumor_site_group"].isin(["Colon", "Rectum"])
            )
            frame["stratum"] = frame["subcohort"].astype(str) + "|" + frame["site2"].astype(str)
            if frame["site2"].notna().sum() < 10 or not frame["D"].any():
                result["encoders"][encoder] = {
                    "status": "NOT_DERIVED",
                    "reason": "required molecular/site cells unavailable",
                }
                continue
            rng = _named_rng(f"aim1:why_d:{encoder}")
            result["encoders"][encoder] = {
                "status": "DERIVED",
                "A_random_restriction_negative_control": whyd.analysis_A(frame, rng),
                "B_pairwise_auc_decomposition": whyd.analysis_B(frame, rng),
                "C_molecular_score_distributions": whyd.analysis_C(frame, rng),
                "D_adjusted_molecular_association": whyd.analysis_D(frame, rng),
            }
        _json_bytes(result)
        return result
    finally:
        whyd.N_RAND, whyd.N_BOOT = original_rand, original_boot


def _e1d_clinical(
    source_frames: Mapping[str, pd.DataFrame],
    *,
    n_bootstrap: int,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    import aim1_clinical_baseline as clinical

    aligned = _aligned_encoder_frames(source_frames, context="aim1/e1d")
    base = aligned[ENCODERS[0]].copy()
    base["label"] = pd.to_numeric(base["label"], errors="raise").astype(int)
    base["k_fold"] = pd.to_numeric(base["k_fold"], errors="raise").astype(int)
    labels_all = _binary(base["label"], context="Aim1 E1d")

    # Every score and calibrator is cross-fitted once on the full source
    # population.  The stage-known arm below is a fixed evaluation restriction
    # of those already OOF scores, never a clinical/stacking refit.
    clinical_raw = clinical.cross_fitted_scores(base, clinical.NUMERIC, clinical.CATEGORICAL)
    clinical_cal = clinical.platt(base, clinical_raw)
    model_scores: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for encoder in ENCODERS:
        frame = aligned[encoder].copy()
        raw_logit = frame["mean_logit_5seed"].to_numpy(float)
        wsi_raw = 1.0 / (1.0 + np.exp(-np.clip(raw_logit, -50.0, 50.0)))
        wsi_cal = clinical.platt(frame, wsi_raw)
        fusion_frame = frame.assign(wsi_logit=raw_logit)
        fusion_raw = clinical.cross_fitted_scores(
            fusion_frame, [*clinical.NUMERIC, "wsi_logit"], clinical.CATEGORICAL
        )
        fusion_cal = clinical.platt(frame, fusion_raw)
        model_scores[encoder] = {
            "wsi": (wsi_raw, wsi_cal),
            "fusion": (fusion_raw, fusion_cal),
        }

    population_masks = {
        "A_all_primary": np.ones(len(base), dtype=bool),
        "G_stage_known_derived": _e1a_mask(base, "G_stage_known_derived"),
    }
    observed = {
        key: (int(mask.sum()), int(labels_all[mask].sum()))
        for key, mask in population_masks.items()
    }
    if observed != {
        "A_all_primary": (1_239, 501),
        "G_stage_known_derived": (1_060, 422),
    }:
        raise GovernanceError(f"E1d population census drifted: {observed}")
    result: dict[str, Any] = {
        "design": "E0-fold cross-fitted clinical comparator and clinical+WSI stack",
        "score_generation_population": "A_all_primary TCGA+SurGen primary",
        "stage_known_is_fixed_evaluation_subset_not_refit": True,
        "clinical_variables": {
            "numeric": list(clinical.NUMERIC),
            "categorical": list(clinical.CATEGORICAL),
            "missingness": "training-fold imputation/explicit unknown levels",
        },
        "populations": {},
    }
    for population, mask in population_masks.items():
        labels = labels_all[mask]
        patient = base.loc[mask].reset_index(drop=True)
        indices = stratified_bootstrap_indices(
            labels,
            patient["subcohort"].astype(str),
            n_bootstrap=n_bootstrap,
            stream=f"aim1:e1d:{population}:shared",
        )

        def model_block(
            raw_all: np.ndarray,
            calibrated_all: np.ndarray,
            *,
            model_key: str,
            population_name: str = population,
            population_mask: np.ndarray = mask,
            population_labels: np.ndarray = labels,
            population_indices: np.ndarray = indices,
            population_patient: pd.DataFrame = patient,
        ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
            raw = np.asarray(raw_all, dtype=float)[population_mask]
            calibrated = np.asarray(calibrated_all, dtype=float)[population_mask]
            point = clinical.metrics(population_labels, raw, calibrated)
            auc, ap = _rank_draws(population_labels, raw, population_indices)
            brier, loss = _probability_draws(population_labels, calibrated, population_indices)
            draws = {"auroc": auc, "auprc": ap, "brier": brier, "log_loss": loss}
            for metric, values in draws.items():
                arrays[f"aim1__e1d__{population_name}__{model_key}__{metric}"] = values
            return (
                {
                    "patients": int(len(population_patient)),
                    "mutant": int(population_labels.sum()),
                    **point,
                    "auroc_ci95": _ci(auc),
                    "auprc_ci95": _ci(ap),
                    "brier_ci95": _ci(brier),
                    "log_loss_ci95": _ci(loss),
                },
                draws,
            )

        clinical_point, clinical_draws = model_block(
            clinical_raw, clinical_cal, model_key="clinical"
        )
        population_result: dict[str, Any] = {
            "population": {
                "patients": int(len(patient)),
                "mutant": int(labels.sum()),
                "stage_unknown_frozen": int(patient["stage_class"].astype(str).eq("unknown").sum()),
                "age_missing": int(
                    pd.to_numeric(patient["age_at_diagnosis"], errors="coerce").isna().sum()
                ),
            },
            "clinical": clinical_point,
            "encoders": {},
        }
        for encoder in ENCODERS:
            wsi_point, wsi_draws = model_block(
                *model_scores[encoder]["wsi"], model_key=f"{encoder}__wsi"
            )
            fusion_point, fusion_draws = model_block(
                *model_scores[encoder]["fusion"], model_key=f"{encoder}__fusion"
            )
            comparisons: dict[str, Any] = {}
            for name, left_point, left_draws, right_point, right_draws in (
                ("wsi_minus_clinical", wsi_point, wsi_draws, clinical_point, clinical_draws),
                (
                    "fusion_minus_clinical",
                    fusion_point,
                    fusion_draws,
                    clinical_point,
                    clinical_draws,
                ),
                ("fusion_minus_wsi", fusion_point, fusion_draws, wsi_point, wsi_draws),
            ):
                metric_blocks: dict[str, Any] = {}
                for metric in ("auroc", "auprc", "brier", "log_loss"):
                    delta = left_draws[metric] - right_draws[metric]
                    arrays[f"aim1__e1d__{population}__{encoder}__{name}__{metric}"] = delta
                    metric_blocks[metric] = {
                        "left": float(left_point[metric]),
                        "right": float(right_point[metric]),
                        "delta": float(left_point[metric] - right_point[metric]),
                        "delta_ci95": _ci(delta),
                        "lower_is_better": metric in {"brier", "log_loss"},
                    }
                comparisons[name] = metric_blocks
            population_result["encoders"][encoder] = {
                "wsi": wsi_point,
                "fusion": {
                    **fusion_point,
                    "stacking_caveat": "WSI input is OOF and coefficients are fitted off-fold",
                },
                "incremental_contrasts": comparisons,
                "paired_patient_bootstrap_shared_across_models": True,
            }
        result["populations"][population] = population_result
    return result


def _orion_sensitivity_populations(
    target_frames: Mapping[str, Mapping[str, pd.DataFrame]],
    overall: Mapping[str, Any],
    *,
    n_bootstrap: int,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    reference = target_frames[ENCODERS[0]]["orion_cpht"]
    populations = {
        "all_40": np.ones(len(reference), dtype=bool),
        "exclude_neoadjuvant": ~reference["exclude_neoadjuvant"].astype(bool).to_numpy(),
        "exclude_ambiguous_crc15": ~reference["exclude_ambiguous_crc15"].astype(bool).to_numpy(),
    }
    expected = {
        "all_40": (40, 15),
        "exclude_neoadjuvant": (34, 11),
        "exclude_ambiguous_crc15": (39, 15),
    }
    for name, mask in populations.items():
        labels = reference.loc[mask, "label"].astype(int)
        if (len(labels), int(labels.sum())) != expected[name]:
            raise GovernanceError(f"Orion sensitivity population {name} census drifted")
    exclusions = _performance_panel(
        {encoder: target_frames[encoder]["orion_cpht"] for encoder in ENCODERS},
        {key: populations[key] for key in ("exclude_neoadjuvant", "exclude_ambiguous_crc15")},
        prefix="aim2_cpht_raw",
        n_bootstrap=n_bootstrap,
        arrays=arrays,
        include_probability_metrics=True,
    )
    if any(
        (int(overall[encoder]["patients"]), int(overall[encoder]["mutant"])) != (40, 15)
        for encoder in ENCODERS
    ):
        raise GovernanceError("Orion overall claim does not bind the exact all-40 population")
    return {
        "role": "retrospective raw cross-protocol sensitivity",
        "not_confirmatory": True,
        "not_a_replacement_for_existing_all_conventional_CPHT": True,
        "populations": {
            # This is an exact alias of the already bound overall Orion target
            # row, avoiding a second bootstrap stream for the same estimand.
            "all_40": {encoder: dict(overall[encoder]) for encoder in ENCODERS},
            **exclusions,
        },
        "cpht_a": {
            "status": "INHERITED_NO_NEW_RUN",
            "role": "separate target-internal residual adaptation; unchanged",
        },
        "cpht_r": {
            "status": "NOT_RUN",
            "role": "pixel normalization/attenuation arm",
        },
    }


def _claim_row(
    *,
    row_id: str,
    section: str,
    training_population: str,
    evaluation_population: str,
    encoder: str,
    block: Mapping[str, Any],
    evidence_state: str,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "section": section,
        "training_population": training_population,
        "evaluation_population": evaluation_population,
        "encoder": encoder,
        "n_patients": int(block["patients"]),
        "n_mutant": int(block["mutant"]),
        "auroc": float(block["auroc"]),
        "auroc_ci95": [float(value) for value in block["auroc_ci95"]],
        "auprc": float(block["auprc"]),
        "auprc_ci95": [float(value) for value in block["auprc_ci95"]],
        "evidence_state": evidence_state,
    }


def _report_claim_rows(
    canonical_e0: Mapping[str, Any],
    source_e0: Mapping[str, Any],
    aim2_targets: Mapping[str, Any],
    e1a: Mapping[str, Any],
    e1d: Mapping[str, Any],
    rih_role: Mapping[str, Any],
    cpht: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for population, encoders in canonical_e0.items():
        for encoder in ENCODERS:
            rows.append(
                _claim_row(
                    row_id=f"aim1.canonical_e0.{population}.{encoder}",
                    section="Aim1/canonical_E0",
                    training_population="TCGA+SurGen+RIH+CPTAC primary",
                    evaluation_population=population,
                    encoder=encoder,
                    block=encoders[encoder],
                    evidence_state="INHERITED_GOVERNED_FIVE_SEED_OOF",
                )
            )
    for population, encoders in source_e0.items():
        for encoder in ENCODERS:
            rows.append(
                _claim_row(
                    row_id=f"aim1.source_restricted_e0.{population}.{encoder}",
                    section="Aim1/source_restricted_E0",
                    training_population="TCGA+SurGen primary",
                    evaluation_population=population,
                    encoder=encoder,
                    block=encoders[encoder],
                    evidence_state="NEW_GOVERNED_FIVE_SEED_OOF",
                )
            )
    aim2_states = {
        "cptac_primary": "FAMILY_NAIVE_ZERO_SHOT",
        "rih_primary": "FAMILY_NAIVE_ZERO_SHOT",
        "rih_metastatic": "FAMILY_NAIVE_RETROSPECTIVE_SENSITIVITY",
        "sr1482_metastatic": "SOURCE_EXPOSED_RETROSPECTIVE_SENSITIVITY",
        "orion_cpht": "RETROSPECTIVE_RAW_CPHT_SENSITIVITY",
    }
    aim2_sections = {
        "cptac_primary": "Aim2/zero_shot_primary",
        "rih_primary": "Aim2/zero_shot_primary",
        "rih_metastatic": "Aim2/metastatic_sensitivity",
        "sr1482_metastatic": "Aim2/metastatic_sensitivity",
        "orion_cpht": "Aim2/CPHT_raw_sensitivity",
    }
    for target in TARGET_ORDER:
        for encoder in ENCODERS:
            rows.append(
                _claim_row(
                    row_id=f"aim2.{target}.{encoder}",
                    section=aim2_sections[target],
                    training_population="TCGA+SurGen primary",
                    evaluation_population=target,
                    encoder=encoder,
                    block=aim2_targets[target]["encoders"][encoder],
                    evidence_state=aim2_states[target],
                )
            )
    for key in E1A_SETS:
        for encoder in ENCODERS:
            rows.append(
                _claim_row(
                    row_id=f"aim1.e1a.{key}.{encoder}",
                    section="Aim1/E1a_controlled_challenges",
                    training_population="TCGA+SurGen primary",
                    evaluation_population=key,
                    encoder=encoder,
                    block=e1a["encoders"][encoder][key],
                    evidence_state="NEW_GOVERNED_FIXED_MODEL_RESTRICTION",
                )
            )
    for population in ("A_all_primary", "G_stage_known_derived"):
        block = e1d["populations"][population]
        rows.append(
            _claim_row(
                row_id=f"aim1.e1d.{population}.clinical",
                section="Aim1/E1d_clinical_improvements",
                training_population="TCGA+SurGen primary",
                evaluation_population=population,
                encoder="clinical",
                block=block["clinical"],
                evidence_state="NEW_GOVERNED_CROSSFITTED_CLINICAL_COMPARISON",
            )
        )
        for encoder in ENCODERS:
            for model in ("wsi", "fusion"):
                rows.append(
                    _claim_row(
                        row_id=f"aim1.e1d.{population}.{encoder}.{model}",
                        section="Aim1/E1d_clinical_improvements",
                        training_population="TCGA+SurGen primary",
                        evaluation_population=population,
                        encoder=encoder,
                        block=block["encoders"][encoder][model],
                        evidence_state="NEW_GOVERNED_CROSSFITTED_CLINICAL_COMPARISON",
                    )
                )
    for role in ("primary", "metastatic"):
        for encoder in ENCODERS:
            rows.append(
                _claim_row(
                    row_id=f"aim2.rih_disjoint_role.{role}.{encoder}",
                    section="Aim2/RIH_disjoint_role",
                    training_population="TCGA+SurGen primary",
                    evaluation_population=f"rih_disjoint_{role}",
                    encoder=encoder,
                    block=rih_role["encoders"][encoder][role],
                    evidence_state="FAMILY_NAIVE_DISJOINT_ROLE_SENSITIVITY",
                )
            )
    for population in ("exclude_neoadjuvant", "exclude_ambiguous_crc15"):
        for encoder in ENCODERS:
            rows.append(
                _claim_row(
                    row_id=f"aim2.cpht_raw.{population}.{encoder}",
                    section="Aim2/CPHT_raw_sensitivity",
                    training_population="TCGA+SurGen primary",
                    evaluation_population=population,
                    encoder=encoder,
                    block=cpht["populations"][population][encoder],
                    evidence_state="RETROSPECTIVE_RAW_CPHT_RESTRICTION_SENSITIVITY",
                )
            )
    expected_ids = _expected_performance_claim_ids()
    if (
        len(rows) != REPORT_PERFORMANCE_CLAIM_COUNT
        or len({row["row_id"] for row in rows}) != REPORT_PERFORMANCE_CLAIM_COUNT
        or {row["row_id"] for row in rows} != expected_ids
    ):
        raise GovernanceError("FINAL-v11 report claim-row roster drifted")
    return rows


def _expected_performance_claim_ids() -> set[str]:
    canonical = {
        f"aim1.canonical_e0.{population}.{encoder}"
        for population in (
            "pooled_all_primary",
            "tcga",
            "sr386",
            "sr1482",
            "surgen",
            "tcga_surgen",
        )
        for encoder in ENCODERS
    }
    source = {
        f"aim1.source_restricted_e0.{population}.{encoder}"
        for population in ("pooled_source", "tcga", "sr386", "sr1482", "surgen", "tcga_surgen")
        for encoder in ENCODERS
    }
    aim2 = {f"aim2.{target}.{encoder}" for target in TARGET_ORDER for encoder in ENCODERS}
    e1a = {f"aim1.e1a.{key}.{encoder}" for key in E1A_SETS for encoder in ENCODERS}
    e1d = {
        *{
            f"aim1.e1d.{population}.clinical"
            for population in ("A_all_primary", "G_stage_known_derived")
        },
        *{
            f"aim1.e1d.{population}.{encoder}.{model}"
            for population in ("A_all_primary", "G_stage_known_derived")
            for encoder in ENCODERS
            for model in ("wsi", "fusion")
        },
    }
    rih = {
        f"aim2.rih_disjoint_role.{role}.{encoder}"
        for role in ("primary", "metastatic")
        for encoder in ENCODERS
    }
    cpht = {
        f"aim2.cpht_raw.{population}.{encoder}"
        for population in ("exclude_neoadjuvant", "exclude_ambiguous_crc15")
        for encoder in ENCODERS
    }
    ids = canonical | source | aim2 | e1a | e1d | rih | cpht
    if len(ids) != REPORT_PERFORMANCE_CLAIM_COUNT:
        raise GovernanceError("Internal performance-claim ID inventory drifted")
    return ids


def _population_arm(name: str, patients: int, mutant: int) -> dict[str, Any]:
    return {"name": name, "n_patients": int(patients), "n_mutant": int(mutant)}


def _contrast_metric(
    reference: float,
    reference_ci95: Sequence[float],
    comparison: float,
    comparison_ci95: Sequence[float],
    delta: float,
    delta_ci95: Sequence[float],
    *,
    lower_is_better: bool,
) -> dict[str, Any]:
    values = [reference, *reference_ci95, comparison, *comparison_ci95, delta, *delta_ci95]
    if (
        len(reference_ci95) != 2
        or len(comparison_ci95) != 2
        or len(delta_ci95) != 2
        or not np.isfinite(np.asarray(values, dtype=float)).all()
    ):
        raise GovernanceError("Non-finite or malformed report contrast metric")
    return {
        "reference": float(reference),
        "reference_ci95": [float(value) for value in reference_ci95],
        "comparison": float(comparison),
        "comparison_ci95": [float(value) for value in comparison_ci95],
        "delta_comparison_minus_reference": float(delta),
        "delta_ci95": [float(value) for value in delta_ci95],
        "lower_is_better": bool(lower_is_better),
    }


def _contrast_row(
    *,
    row_id: str,
    section: str,
    training_population: str,
    encoder: str,
    reference: Mapping[str, Any],
    comparison: Mapping[str, Any],
    metrics: Mapping[str, Any],
    evidence_state: str,
) -> dict[str, Any]:
    if not metrics:
        raise GovernanceError(f"Contrast row has no metrics: {row_id}")
    return {
        "row_id": row_id,
        "section": section,
        "training_population": training_population,
        "encoder": encoder,
        "reference": dict(reference),
        "comparison": dict(comparison),
        "contrast_definition": "comparison minus reference",
        "metrics": dict(metrics),
        "evidence_state": evidence_state,
    }


def _report_contrast_rows(
    source_encoder_contrasts: Mapping[str, Any],
    e1a: Mapping[str, Any],
    e1a_s: Mapping[str, Any],
    e1d: Mapping[str, Any],
    rih_role: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for population, block in source_encoder_contrasts.items():
        rows.append(
            _contrast_row(
                row_id=(
                    f"aim1.source_restricted_e0_encoder_delta.{population}.virchow2_cls_minus_univ1"
                ),
                section="Aim1/source_restricted_E0_encoder_contrast",
                training_population="TCGA+SurGen primary",
                encoder="virchow2_cls_minus_univ1",
                reference=_population_arm(
                    f"{population}:univ1", block["patients"], block["mutant"]
                ),
                comparison=_population_arm(
                    f"{population}:virchow2_cls", block["patients"], block["mutant"]
                ),
                metrics={
                    metric: _contrast_metric(
                        values["univ1"],
                        values["univ1_ci95"],
                        values["virchow2_cls"],
                        values["virchow2_cls_ci95"],
                        values["delta_virchow2_cls_minus_univ1"],
                        values["delta_ci95"],
                        lower_is_better=False,
                    )
                    for metric, values in block["metrics"].items()
                },
                evidence_state="NEW_GOVERNED_SHARED_PATIENT_ENCODER_CONTRAST",
            )
        )
    a_sets = e1a["encoders"]
    for key in E1A_SETS:
        if key == "A_all_primary":
            continue
        for encoder in ENCODERS:
            full = a_sets[encoder]["A_all_primary"]
            subset = a_sets[encoder][key]
            rows.append(
                _contrast_row(
                    row_id=f"aim1.e1a_delta_A_minus_set.{key}.{encoder}",
                    section="Aim1/E1a_controlled_challenge_contrast",
                    training_population="TCGA+SurGen primary",
                    encoder=encoder,
                    reference=_population_arm(key, subset["patients"], subset["mutant"]),
                    comparison=_population_arm("A_all_primary", full["patients"], full["mutant"]),
                    metrics={
                        "auroc": _contrast_metric(
                            subset["auroc"],
                            subset["auroc_ci95"],
                            full["auroc"],
                            full["auroc_ci95"],
                            subset["delta_A_minus_subset_auroc"],
                            subset["delta_A_minus_subset_auroc_ci95"],
                            lower_is_better=False,
                        ),
                        "auprc": _contrast_metric(
                            subset["auprc"],
                            subset["auprc_ci95"],
                            full["auprc"],
                            full["auprc_ci95"],
                            subset["delta_A_minus_subset_auprc"],
                            subset["delta_A_minus_subset_auprc_ci95"],
                            lower_is_better=False,
                        ),
                    },
                    evidence_state="NEW_GOVERNED_NESTED_SHARED_RESAMPLE_CONTRAST",
                )
            )
    for reference_name, reference_block in e1a_s["references"].items():
        for encoder in ENCODERS:
            block = reference_block["encoders"][encoder]
            rows.append(
                _contrast_row(
                    row_id=(f"aim1.e1a_s_delta_D_minus_reference.{reference_name}.{encoder}"),
                    section="Aim1/E1a_S_standardized_contrast",
                    training_population="TCGA+SurGen primary",
                    encoder=encoder,
                    reference=_population_arm(
                        reference_name, block["patients_A"], block["mutant_A"]
                    ),
                    comparison=_population_arm(
                        "D_mss_braf_wt", block["patients_D"], block["mutant_D"]
                    ),
                    metrics={
                        "standardized_auroc": _contrast_metric(
                            block["standardized_A_auroc"],
                            block["standardized_A_auroc_ci95"],
                            block["standardized_D_auroc"],
                            block["standardized_D_auroc_ci95"],
                            block["delta_D_minus_A_auroc"],
                            block["delta_D_minus_A_auroc_ci95"],
                            lower_is_better=False,
                        )
                    },
                    evidence_state="NEW_GOVERNED_STANDARDIZED_SHARED_RESAMPLE_CONTRAST",
                )
            )
    comparison_models = {
        "wsi_minus_clinical": ("clinical", "wsi"),
        "fusion_minus_clinical": ("clinical", "fusion"),
        "fusion_minus_wsi": ("wsi", "fusion"),
    }
    for population, population_block in e1d["populations"].items():
        census = population_block["population"]
        for encoder in ENCODERS:
            for name, (reference_model, comparison_model) in comparison_models.items():
                metrics = population_block["encoders"][encoder]["incremental_contrasts"][name]
                reference_result = (
                    population_block["clinical"]
                    if reference_model == "clinical"
                    else population_block["encoders"][encoder][reference_model]
                )
                comparison_result = population_block["encoders"][encoder][comparison_model]
                rows.append(
                    _contrast_row(
                        row_id=f"aim1.e1d_delta.{population}.{encoder}.{name}",
                        section="Aim1/E1d_clinical_increment_contrast",
                        training_population="TCGA+SurGen primary",
                        encoder=encoder,
                        reference=_population_arm(
                            f"{population}:{reference_model}",
                            census["patients"],
                            census["mutant"],
                        ),
                        comparison=_population_arm(
                            f"{population}:{comparison_model}",
                            census["patients"],
                            census["mutant"],
                        ),
                        metrics={
                            metric: _contrast_metric(
                                value["right"],
                                reference_result[f"{metric}_ci95"],
                                value["left"],
                                comparison_result[f"{metric}_ci95"],
                                value["delta"],
                                value["delta_ci95"],
                                lower_is_better=value["lower_is_better"],
                            )
                            for metric, value in metrics.items()
                        },
                        evidence_state="NEW_GOVERNED_SHARED_PATIENT_MODEL_CONTRAST",
                    )
                )
    for encoder in ENCODERS:
        block = rih_role["encoders"][encoder]
        primary = block["primary"]
        metastatic = block["metastatic"]
        delta = block["delta_metastatic_minus_primary"]
        rows.append(
            _contrast_row(
                row_id=f"aim2.rih_disjoint_role.delta_metastatic_minus_primary.{encoder}",
                section="Aim2/RIH_disjoint_role_contrast",
                training_population="TCGA+SurGen primary",
                encoder=encoder,
                reference=_population_arm("rih_disjoint_primary", 145, 65),
                comparison=_population_arm("rih_disjoint_metastatic", 77, 33),
                metrics={
                    "auroc": _contrast_metric(
                        primary["auroc"],
                        primary["auroc_ci95"],
                        metastatic["auroc"],
                        metastatic["auroc_ci95"],
                        delta["auroc"],
                        delta["auroc_ci95"],
                        lower_is_better=False,
                    ),
                    "auprc": _contrast_metric(
                        primary["auprc"],
                        primary["auprc_ci95"],
                        metastatic["auprc"],
                        metastatic["auprc_ci95"],
                        delta["auprc"],
                        delta["auprc_ci95"],
                        lower_is_better=False,
                    ),
                },
                evidence_state="FAMILY_NAIVE_DISJOINT_ROLE_CONTRAST",
            )
        )
    expected_ids = _expected_contrast_claim_ids()
    if (
        len(rows) != REPORT_CONTRAST_CLAIM_COUNT
        or len({row["row_id"] for row in rows}) != REPORT_CONTRAST_CLAIM_COUNT
        or {row["row_id"] for row in rows} != expected_ids
    ):
        raise GovernanceError("FINAL-v11 report contrast-row roster drifted")
    return rows


def _expected_contrast_claim_ids() -> set[str]:
    source = {
        f"aim1.source_restricted_e0_encoder_delta.{population}.virchow2_cls_minus_univ1"
        for population in ("pooled_source", "tcga", "sr386", "sr1482", "surgen", "tcga_surgen")
    }
    e1a = {
        f"aim1.e1a_delta_A_minus_set.{key}.{encoder}"
        for key in E1A_SETS
        if key != "A_all_primary"
        for encoder in ENCODERS
    }
    e1a_s = {
        f"aim1.e1a_s_delta_D_minus_reference.{reference}.{encoder}"
        for reference in ("A_all_primary", "A_complete")
        for encoder in ENCODERS
    }
    e1d = {
        f"aim1.e1d_delta.{population}.{encoder}.{comparison}"
        for population in ("A_all_primary", "G_stage_known_derived")
        for encoder in ENCODERS
        for comparison in (
            "wsi_minus_clinical",
            "fusion_minus_clinical",
            "fusion_minus_wsi",
        )
    }
    rih = {
        f"aim2.rih_disjoint_role.delta_metastatic_minus_primary.{encoder}" for encoder in ENCODERS
    }
    ids = source | e1a | e1a_s | e1d | rih
    if len(ids) != REPORT_CONTRAST_CLAIM_COUNT:
        raise GovernanceError("Internal contrast-claim ID inventory drifted")
    return ids


def _why_d_evidence_records(why_d: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for encoder in ENCODERS:
        block = why_d["encoders"][encoder]
        if block.get("status") != "DERIVED":
            raise GovernanceError(f"Why-D must be derived for governed source data: {encoder}")
        records.append(
            {
                "record_id": f"aim1.why_d.{encoder}",
                "section": "Aim1/Why_D_explanatory",
                "training_population": "TCGA+SurGen primary",
                "evaluation_population": "A_complete molecular/site-valid subset",
                "encoder": encoder,
                "n_patients": E1A_EXPECTED_CENSUS["A_complete"][0],
                "n_mutant": E1A_EXPECTED_CENSUS["A_complete"][1],
                "random_restriction_draws": int(why_d["random_restriction_draws"]),
                "patient_bootstrap_draws": int(why_d["patient_bootstrap_draws"]),
                "analysis_a_random_restriction": block["A_random_restriction_negative_control"],
                "analysis_b_pairwise_auc_decomposition": block["B_pairwise_auc_decomposition"],
                "analysis_c_molecular_score_distributions": block[
                    "C_molecular_score_distributions"
                ],
                "analysis_d_adjusted_molecular_association": block[
                    "D_adjusted_molecular_association"
                ],
                "result_path": f"/aim1/why_d/encoders/{encoder}",
                "evidence_state": "NEW_GOVERNED_EXPLANATORY_DIAGNOSTIC",
            }
        )
    if (
        len(records) != WHY_D_EVIDENCE_RECORD_COUNT
        or len({record["record_id"] for record in records}) != WHY_D_EVIDENCE_RECORD_COUNT
    ):
        raise GovernanceError("Why-D evidence-record roster drifted")
    _json_bytes(records)
    return records


def build_analysis_results(
    patient_table: pd.DataFrame,
    provenance: Mapping[str, Any],
    target_frames: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    n_bootstrap: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    canonical_frames = {
        encoder: patient_table.loc[
            patient_table["analysis_family"].eq("inherited_canonical_all_primary_oof")
            & patient_table["encoder"].eq(encoder)
        ].copy()
        for encoder in ENCODERS
    }
    source_frames = {
        encoder: patient_table.loc[
            patient_table["analysis_family"].eq("source_restricted_tcga_surgen_oof")
            & patient_table["encoder"].eq(encoder)
        ].copy()
        for encoder in ENCODERS
    }
    canonical_e0 = _performance_panel(
        canonical_frames,
        _canonical_population_masks(canonical_frames[ENCODERS[0]]),
        prefix="aim1_canonical_e0",
        n_bootstrap=n_bootstrap,
        arrays=arrays,
        include_probability_metrics=False,
    )
    source_e0 = _performance_panel(
        source_frames,
        _source_population_masks(source_frames[ENCODERS[0]]),
        prefix="aim1_source_restricted_e0",
        n_bootstrap=n_bootstrap,
        arrays=arrays,
        include_probability_metrics=True,
    )
    source_e0_encoder_contrasts = _source_encoder_contrast_panel(
        source_e0,
        arrays=arrays,
    )
    e1a = _e1a_panel(source_frames, n_bootstrap=n_bootstrap, arrays=arrays)
    e1a_s = _e1a_standardized(source_frames, n_bootstrap=n_bootstrap, arrays=arrays)
    why_d = _why_d(source_frames, n_bootstrap=n_bootstrap)
    e1d = _e1d_clinical(source_frames, n_bootstrap=n_bootstrap, arrays=arrays)
    aim2_targets = _target_performance(target_frames, n_bootstrap=n_bootstrap, arrays=arrays)
    rih_role = _rih_disjoint_role_contrast(target_frames, n_bootstrap=n_bootstrap, arrays=arrays)
    cpht = _orion_sensitivity_populations(
        target_frames,
        aim2_targets["orion_cpht"]["encoders"],
        n_bootstrap=n_bootstrap,
        arrays=arrays,
    )
    claim_rows = _report_claim_rows(
        canonical_e0,
        source_e0,
        aim2_targets,
        e1a,
        e1d,
        rih_role,
        cpht,
    )
    contrast_rows = _report_contrast_rows(
        source_e0_encoder_contrasts,
        e1a,
        e1a_s,
        e1d,
        rih_role,
    )
    why_d_records = _why_d_evidence_records(why_d)
    results = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "experiment": EXPERIMENT,
        "score_contract": {
            "slide_ensemble": "arithmetic mean of five native slide logits",
            "patient_aggregation": "arithmetic mean of ensemble slide logits within patient",
            "arithmetic_commutation_note": "equals patient-within-seed then seed mean on fixed slide rosters",
            "probability_roundtrip_used_for_rank_metrics": False,
            "seeds": list(SEEDS),
            "encoders": list(ENCODERS),
        },
        "inference": {
            "unit": "patient",
            "bootstrap_draws": n_bootstrap,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "stratification": "subcohort_x_KRAS_label; fixed-class target equivalent",
            "interval": "percentile_95",
            "shared_encoder_draws": True,
            "model_seeds_are_inference_units": False,
            "folds_are_inference_units": False,
            "bootstrap_array_count": len(arrays),
        },
        "calibration": {
            **provenance["source_platt"],
            "method": "one unregularized Platt logistic map per encoder",
            "fit_data": "five-seed mean TCGA+SurGen OOF patient native logits only",
            "target_labels_used": False,
            "target_refitting": False,
        },
        "aim1": {
            "canonical_all_primary_e0": {
                "training_population": "TCGA+SurGen+RIH+CPTAC primary",
                "evaluation_design": "inherited governed five-seed all-primary OOF",
                "performance": canonical_e0,
            },
            "source_restricted_tcga_surgen_e0": {
                "training_population": "TCGA+SurGen primary",
                "evaluation_design": "new governed five-seed source OOF",
                "performance": source_e0,
                "paired_encoder_contrasts": source_e0_encoder_contrasts,
            },
            "e1a": e1a,
            "e1a_s": e1a_s,
            "why_d": why_d,
            "e1d": e1d,
        },
        "aim2": {
            "targets": aim2_targets,
            "rih_disjoint_role_contrast": rih_role,
            "cpht_raw": cpht,
            "boundaries": {
                "cptac_primary": "family-naive zero-shot external primary",
                "rih_primary": "family-naive zero-shot external primary",
                "rih_metastatic": "family-naive standalone retrospective sensitivity",
                "sr1482_metastatic": "source-exposed sensitivity; no external-transport claim",
                "orion_cpht": "retrospective raw cross-protocol sensitivity",
                "no_confirmatory_metastatic_macro": True,
                "no_sr1482_primary_metastatic_role_delta": True,
            },
        },
        "report_claim_rows": claim_rows,
        "report_contrast_rows": contrast_rows,
        "why_d_evidence_records": why_d_records,
        "estimand_boundaries": {
            "canonical_e0_evaluation_is_not_source_restricted_training": True,
            "source_restricted_oof_is_not_external_transport": True,
            "zero_shot_targets_absent_from_training_selection": True,
            "model_seeds_are_not_inference_units": True,
            "folds_are_not_inference_units": True,
            "cpht_raw_role": "retrospective_sensitivity",
            "cpht_a_status": "INHERITED_NO_NEW_RUN",
            "cpht_r_status": "NOT_RUN",
            "whole_section_pathology_status": "GENERATED_UNREAD",
        },
    }
    _json_bytes(results)
    if not arrays or any(
        value.dtype != np.float64 or value.shape != (n_bootstrap,) or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise GovernanceError("Bootstrap array contract is incomplete or non-finite")
    return results, {key: arrays[key] for key in sorted(arrays)}


def _analysis_contract_payload(
    campaign_root: Path,
    *,
    inference_seal: Mapping[str, Any],
    outcome_identities: Mapping[str, Any],
    derived_covariate_source: Mapping[str, Any],
    inherited: Mapping[str, Any],
    n_bootstrap: int,
    array_names: Sequence[str],
    created_utc: str,
    test_only_noncanonical_parameters: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "analysis_governed_after_inference_seal",
        "created_utc": created_utc,
        "experiment": EXPERIMENT,
        "campaign_root": str(_safe_campaign_root(campaign_root)),
        "downstream_contract": _artifact(contract_path(campaign_root)),
        "inference_seal": _artifact(seal_path(campaign_root)),
        "seal_status_observed_before_outcome_open": inference_seal["status"],
        "outcomes_opened_only_after_seal": True,
        "outcome_sources": dict(outcome_identities),
        "derived_covariate_source": dict(derived_covariate_source),
        "inherited_canonical_e0": dict(inherited),
        "analysis": {
            "inference_unit": "patient",
            "bootstrap_draws": n_bootstrap,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "shared_encoder_draws": True,
            "target_calibration": False,
            "target_model_selection": False,
            "test_only_noncanonical_parameters": test_only_noncanonical_parameters,
            "bootstrap_arrays": {
                "names": list(array_names),
                "count": len(array_names),
                "dtype": "float64",
                "length": n_bootstrap,
            },
        },
        "patient_table_rows": 6_342,
        "report_claim_row_count": REPORT_PERFORMANCE_CLAIM_COUNT,
        "report_contrast_row_count": REPORT_CONTRAST_CLAIM_COUNT,
        "why_d_evidence_record_count": WHY_D_EVIDENCE_RECORD_COUNT,
        "output_inventory": list(ANALYSIS_FILES),
        "implementation": _implementation_sources(),
    }


def analyze(
    campaign_root: Path,
    *,
    apply: bool,
    n_bootstrap: int = N_BOOTSTRAP,
    test_only_noncanonical_parameters: bool = False,
) -> dict[str, Any]:
    root = _safe_campaign_root(campaign_root)
    seal = verify_inference_seal(root)
    if n_bootstrap != N_BOOTSTRAP and (
        not test_only_noncanonical_parameters or not str(root).startswith("/tmp/")
    ):
        raise GovernanceError("Canonical analysis requires exactly 10,000 bootstrap draws")
    destination = analysis_root(root)
    completion = destination / "analysis_completion_receipt.json"
    if completion.exists():
        verified = verify_analysis(root)
        return {"status": "already_complete", "verification": verified}
    existing = list(destination.iterdir()) if destination.is_dir() else []
    if existing:
        raise GovernanceError(
            "Partial immutable analysis exists; quarantine it explicitly before replay"
        )
    if not apply:
        return {
            "status": "dry_run_ready_after_inference_seal",
            "analysis_files": list(ANALYSIS_FILES),
            "bootstrap_draws": n_bootstrap,
            "target_outcomes_opened": False,
        }
    outcomes, outcome_identities = _open_target_outcomes_after_seal(root)
    source_covariates, derived_covariate_source = _open_source_derived_covariates_after_seal(root)
    patient_table, provenance, target_frames = build_patient_table(
        root,
        outcomes,
        source_covariates,
    )
    results, arrays = build_analysis_results(
        patient_table,
        provenance,
        target_frames,
        n_bootstrap=n_bootstrap,
    )
    contract = _analysis_contract_payload(
        root,
        inference_seal=seal,
        outcome_identities=outcome_identities,
        derived_covariate_source=derived_covariate_source,
        inherited=provenance["inherited_canonical"],
        n_bootstrap=n_bootstrap,
        array_names=list(arrays),
        created_utc=_utcnow(),
        test_only_noncanonical_parameters=test_only_noncanonical_parameters,
    )
    destination.mkdir(parents=True, exist_ok=False)
    _write_json_once(destination / "contract.json", contract)
    _write_parquet_once(destination / "patient_native_logits.parquet", patient_table)
    _write_bytes_once(destination / "bootstrap_distributions.npz", _npz_bytes(arrays))
    results["analysis_contract"] = _artifact(destination / "contract.json")
    _write_json_once(destination / "results.json", results)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_and_verified",
        "created_utc": _utcnow(),
        "experiment": EXPERIMENT,
        "seal_before_outcomes": _artifact(seal_path(root)),
        "artifacts": {
            "contract": _artifact(destination / "contract.json"),
            "patient_native_logits": _artifact(destination / "patient_native_logits.parquet"),
            "bootstrap_distributions": _artifact(destination / "bootstrap_distributions.npz"),
            "results": _artifact(destination / "results.json"),
        },
        "analysis_artifact_count": 5,
        "patient_rows": 6_342,
        "bootstrap_array_count": len(arrays),
        "bootstrap_draws": n_bootstrap,
        "report_claim_row_count": REPORT_PERFORMANCE_CLAIM_COUNT,
        "report_contrast_row_count": REPORT_CONTRAST_CLAIM_COUNT,
        "why_d_evidence_record_count": WHY_D_EVIDENCE_RECORD_COUNT,
        "outcomes_opened_after_inference_seal": True,
        "target_refits": 0,
        "target_calibrations": 0,
        "target_model_selections": 0,
    }
    _write_json_once(completion, receipt)
    return verify_analysis(root)


def _validate_npz(path: Path, names: Sequence[str], *, length: int) -> None:
    _regular_file(path, context="bootstrap NPZ")
    try:
        with zipfile.ZipFile(path) as zf:
            members = zf.namelist()
            if len(members) != len(set(members)) or any(
                name.startswith("/") or ".." in Path(name).parts for name in members
            ):
                raise GovernanceError("Unsafe or duplicate NPZ members")
        with np.load(path, allow_pickle=False) as archive:
            if sorted(archive.files) != sorted(names):
                raise GovernanceError("Bootstrap NPZ key roster drifted")
            for name in names:
                value = archive[name]
                if (
                    value.dtype != np.float64
                    or value.shape != (length,)
                    or not np.isfinite(value).all()
                ):
                    raise GovernanceError(f"Bootstrap array drifted: {name}")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise GovernanceError(f"Invalid strict bootstrap NPZ: {path}") from exc


def verify_analysis(campaign_root: Path) -> dict[str, Any]:
    root = _safe_campaign_root(campaign_root)
    verify_inference_seal(root)
    directory = analysis_root(root)
    if not directory.is_dir() or directory.is_symlink():
        raise GovernanceError("Analysis directory is missing or symlinked")
    actual = sorted(path.name for path in directory.iterdir())
    if actual != sorted(ANALYSIS_FILES):
        raise GovernanceError(f"Analysis inventory must be exactly five artifacts: {actual}")
    contract = _read_json(directory / "contract.json")
    expected_contract_keys = {
        "schema_version",
        "status",
        "created_utc",
        "experiment",
        "campaign_root",
        "downstream_contract",
        "inference_seal",
        "seal_status_observed_before_outcome_open",
        "outcomes_opened_only_after_seal",
        "outcome_sources",
        "derived_covariate_source",
        "inherited_canonical_e0",
        "analysis",
        "patient_table_rows",
        "report_claim_row_count",
        "report_contrast_row_count",
        "why_d_evidence_record_count",
        "output_inventory",
        "implementation",
    }
    if set(contract) != expected_contract_keys:
        raise GovernanceError("Analysis contract top-level schema drifted")
    if (
        contract.get("status") != "analysis_governed_after_inference_seal"
        or contract.get("outcomes_opened_only_after_seal") is not True
        or contract.get("inference_seal") != _artifact(seal_path(root))
        or contract.get("output_inventory") != list(ANALYSIS_FILES)
        or contract.get("report_claim_row_count") != REPORT_PERFORMANCE_CLAIM_COUNT
        or contract.get("report_contrast_row_count") != REPORT_CONTRAST_CLAIM_COUNT
        or contract.get("why_d_evidence_record_count") != WHY_D_EVIDENCE_RECORD_COUNT
    ):
        raise GovernanceError("Analysis contract ordering/inventory drifted")
    for identity in contract.get("outcome_sources", {}).values():
        _validate_identity(identity)
    derived = contract.get("derived_covariate_source") or {}
    _validate_identity(derived.get("artifact") or {})
    if (derived.get("artifact") or {}).get(
        "sha256"
    ) != DERIVED_COVARIATE_SOURCE_SHA256 or derived.get("source_patient_census") != {
        "G_stage_known_derived": [1_060, 422],
        "G_stage_known_frozen": [895, 344],
        "H_stage_iv": [120, 54],
        "I_right_proximal": [212, 99],
        "J_left_distal": [573, 212],
        "K_transverse": [42, 10],
    }:
        raise GovernanceError("Analysis derived-covariate source identity drifted")
    analysis_spec = contract.get("analysis") or {}
    names = (analysis_spec.get("bootstrap_arrays") or {}).get("names")
    length = int(analysis_spec.get("bootstrap_draws", -1))
    if not isinstance(names, list) or len(names) != len(set(names)) or length < 1:
        raise GovernanceError("Analysis bootstrap contract is malformed")
    _validate_npz(directory / "bootstrap_distributions.npz", names, length=length)
    patient = pd.read_parquet(_regular_file(directory / "patient_native_logits.parquet"))
    if len(patient) != 6_342 or list(patient.columns) != [
        "analysis_family",
        "encoder",
        "dataset",
        "patient_id",
        "label",
        *PATIENT_METADATA,
        *DERIVED_SOURCE_METADATA,
        "n_slides",
        "slide_roster_sha256",
        *(f"logit_seed{seed}" for seed in SEEDS),
        "mean_logit_5seed",
        "source_platt_probability",
        "exclude_neoadjuvant",
        "exclude_ambiguous_crc15",
    ]:
        raise GovernanceError("Patient-native-logit parquet schema/census drifted")
    logits = patient[[*(f"logit_seed{seed}" for seed in SEEDS), "mean_logit_5seed"]]
    if not np.isfinite(logits.to_numpy(float)).all():
        raise GovernanceError("Patient-native-logit parquet contains non-finite scores")
    canonical_mask = patient["analysis_family"].eq("inherited_canonical_all_primary_oof")
    probabilities = pd.to_numeric(patient["source_platt_probability"], errors="coerce")
    if (
        probabilities.loc[canonical_mask].notna().any()
        or not np.isfinite(probabilities.loc[~canonical_mask].to_numpy(float)).all()
    ):
        raise GovernanceError("Patient source-Platt applicability boundary drifted")
    _validate_patient_output_table(patient)
    results = _read_json(directory / "results.json")
    required_results = {
        "schema_version",
        "status",
        "experiment",
        "score_contract",
        "inference",
        "calibration",
        "aim1",
        "aim2",
        "report_claim_rows",
        "report_contrast_rows",
        "why_d_evidence_records",
        "estimand_boundaries",
        "analysis_contract",
    }
    if set(results) != required_results or results.get("status") != "complete":
        raise GovernanceError("Analysis results top-level schema/status drifted")
    claim_rows = results.get("report_claim_rows")
    claim_keys = {
        "row_id",
        "section",
        "training_population",
        "evaluation_population",
        "encoder",
        "n_patients",
        "n_mutant",
        "auroc",
        "auroc_ci95",
        "auprc",
        "auprc_ci95",
        "evidence_state",
    }
    if (
        not isinstance(claim_rows, list)
        or len(claim_rows) != REPORT_PERFORMANCE_CLAIM_COUNT
        or len({row.get("row_id") for row in claim_rows if isinstance(row, dict)})
        != REPORT_PERFORMANCE_CLAIM_COUNT
        or {row.get("row_id") for row in claim_rows if isinstance(row, dict)}
        != _expected_performance_claim_ids()
        or any(set(row) != claim_keys for row in claim_rows if isinstance(row, dict))
    ):
        raise GovernanceError("FINAL-v11 report claim-row roster/schema drifted")
    contrast_rows = results.get("report_contrast_rows")
    contrast_keys = {
        "row_id",
        "section",
        "training_population",
        "encoder",
        "reference",
        "comparison",
        "contrast_definition",
        "metrics",
        "evidence_state",
    }
    metric_keys = {
        "reference",
        "reference_ci95",
        "comparison",
        "comparison_ci95",
        "delta_comparison_minus_reference",
        "delta_ci95",
        "lower_is_better",
    }
    if (
        not isinstance(contrast_rows, list)
        or len(contrast_rows) != REPORT_CONTRAST_CLAIM_COUNT
        or len({row.get("row_id") for row in contrast_rows if isinstance(row, dict)})
        != REPORT_CONTRAST_CLAIM_COUNT
        or {row.get("row_id") for row in contrast_rows if isinstance(row, dict)}
        != _expected_contrast_claim_ids()
        or any(set(row) != contrast_keys for row in contrast_rows if isinstance(row, dict))
        or any(
            set(metric) != metric_keys
            for row in contrast_rows
            if isinstance(row, dict)
            for metric in (row.get("metrics") or {}).values()
        )
    ):
        raise GovernanceError("FINAL-v11 report contrast-row roster/schema drifted")
    why_d_records = results.get("why_d_evidence_records")
    why_d_keys = {
        "record_id",
        "section",
        "training_population",
        "evaluation_population",
        "encoder",
        "n_patients",
        "n_mutant",
        "random_restriction_draws",
        "patient_bootstrap_draws",
        "analysis_a_random_restriction",
        "analysis_b_pairwise_auc_decomposition",
        "analysis_c_molecular_score_distributions",
        "analysis_d_adjusted_molecular_association",
        "result_path",
        "evidence_state",
    }
    if (
        not isinstance(why_d_records, list)
        or len(why_d_records) != WHY_D_EVIDENCE_RECORD_COUNT
        or {record.get("record_id") for record in why_d_records if isinstance(record, dict)}
        != {f"aim1.why_d.{encoder}" for encoder in ENCODERS}
        or any(set(record) != why_d_keys for record in why_d_records if isinstance(record, dict))
    ):
        raise GovernanceError("FINAL-v11 Why-D evidence-record roster/schema drifted")
    aim1 = results["aim1"]
    aim2 = results["aim2"]
    replayed_claims = _report_claim_rows(
        aim1["canonical_all_primary_e0"]["performance"],
        aim1["source_restricted_tcga_surgen_e0"]["performance"],
        aim2["targets"],
        aim1["e1a"],
        aim1["e1d"],
        aim2["rih_disjoint_role_contrast"],
        aim2["cpht_raw"],
    )
    replayed_contrasts = _report_contrast_rows(
        aim1["source_restricted_tcga_surgen_e0"]["paired_encoder_contrasts"],
        aim1["e1a"],
        aim1["e1a_s"],
        aim1["e1d"],
        aim2["rih_disjoint_role_contrast"],
    )
    replayed_why_d = _why_d_evidence_records(aim1["why_d"])
    if (
        claim_rows != replayed_claims
        or contrast_rows != replayed_contrasts
        or why_d_records != replayed_why_d
    ):
        raise GovernanceError("Report bindings do not replay from governed nested results")
    expected_boundaries = {
        "canonical_e0_evaluation_is_not_source_restricted_training": True,
        "source_restricted_oof_is_not_external_transport": True,
        "zero_shot_targets_absent_from_training_selection": True,
        "model_seeds_are_not_inference_units": True,
        "folds_are_not_inference_units": True,
        "cpht_raw_role": "retrospective_sensitivity",
        "cpht_a_status": "INHERITED_NO_NEW_RUN",
        "cpht_r_status": "NOT_RUN",
        "whole_section_pathology_status": "GENERATED_UNREAD",
    }
    if results.get("estimand_boundaries") != expected_boundaries:
        raise GovernanceError("Estimand-boundary block drifted")
    receipt = _read_json(directory / "analysis_completion_receipt.json")
    expected_artifacts = {
        "contract": _artifact(directory / "contract.json"),
        "patient_native_logits": _artifact(directory / "patient_native_logits.parquet"),
        "bootstrap_distributions": _artifact(directory / "bootstrap_distributions.npz"),
        "results": _artifact(directory / "results.json"),
    }
    if (
        receipt.get("status") != "complete_and_verified"
        or receipt.get("artifacts") != expected_artifacts
        or receipt.get("seal_before_outcomes") != _artifact(seal_path(root))
        or receipt.get("analysis_artifact_count") != 5
        or receipt.get("patient_rows") != 6_342
        or receipt.get("report_claim_row_count") != REPORT_PERFORMANCE_CLAIM_COUNT
        or receipt.get("report_contrast_row_count") != REPORT_CONTRAST_CLAIM_COUNT
        or receipt.get("why_d_evidence_record_count") != WHY_D_EVIDENCE_RECORD_COUNT
    ):
        raise GovernanceError("Analysis completion receipt drifted")
    return {
        "status": "PASS",
        "inference_seal": _artifact(seal_path(root)),
        "analysis_completion": _artifact(directory / "analysis_completion_receipt.json"),
        "analysis_artifacts": 5,
        "patient_rows": 6_342,
        "bootstrap_arrays": len(names),
        "bootstrap_draws": length,
        "report_claim_rows": REPORT_PERFORMANCE_CLAIM_COUNT,
        "report_contrast_rows": REPORT_CONTRAST_CLAIM_COUNT,
        "why_d_evidence_records": WHY_D_EVIDENCE_RECORD_COUNT,
    }


def verify(campaign_root: Path) -> dict[str, Any]:
    root = _safe_campaign_root(campaign_root)
    _load_contract(root, deep=True)
    seal = verify_inference_seal(root)
    analysis = verify_analysis(root)
    return {"status": "PASS", "inference": seal["status"], "analysis": analysis}


def status(campaign_root: Path) -> dict[str, Any]:
    root = _safe_campaign_root(campaign_root)
    forbidden_stock = _forbidden_stock_training_artifacts(root)
    stages = {
        "training_complete": (root / TRAINING_RECOVERY_TERMINAL).is_file(),
        "forbidden_stock_receipts": len(forbidden_stock),
        "prepared": contract_path(root).is_file(),
        "preflight": preflight_path(root).is_file(),
        "scores_complete": sum(
            score_path(root, encoder, target, seed).is_file()
            for encoder in ENCODERS
            for target in TARGET_ORDER
            for seed in SEEDS
        ),
        "inference_sealed": seal_path(root).is_file(),
        "analysis_complete": (analysis_root(root) / "analysis_completion_receipt.json").is_file(),
    }
    if forbidden_stock:
        state = "INVALID_FORBIDDEN_STOCK_RECEIPTS"
    elif stages["analysis_complete"]:
        state = "COMPLETE"
    elif stages["inference_sealed"]:
        state = "READY_TO_ANALYZE"
    elif stages["scores_complete"]:
        state = "SCORING_OR_READY_TO_SEAL"
    elif stages["preflight"]:
        state = "READY_TO_SCORE"
    elif stages["prepared"]:
        state = "READY_FOR_PREFLIGHT"
    elif stages["training_complete"]:
        state = "READY_TO_PREPARE"
    else:
        state = "WAITING_FOR_TRAINING"
    return {"campaign_root": str(root), "state": state, "stages": stages}


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
        return child

    prepare_parser = common("prepare")
    prepare_parser.add_argument("--apply", action="store_true")
    preflight_parser = common("preflight")
    preflight_parser.add_argument("--apply", action="store_true")
    score_parser = common("score")
    score_parser.add_argument("--apply", action="store_true")
    score_parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    score_parser.add_argument("--device", choices=("cuda",), default="cuda")
    score_parser.add_argument("--num-workers", type=int, default=4)
    analyze_parser = common("analyze")
    analyze_parser.add_argument("--apply", action="store_true")
    analyze_parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    analyze_parser.add_argument("--test-only-noncanonical-parameters", action="store_true")
    common("verify")
    common("status")
    internal = common("_score-one")
    internal.add_argument("--encoder", choices=ENCODERS, required=True)
    internal.add_argument("--target", choices=TARGET_ORDER, required=True)
    internal.add_argument("--seed", choices=SEEDS, type=int, required=True)
    internal.add_argument("--device", choices=("cuda",), default="cuda")
    internal.add_argument("--num-workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        _print(prepare(args.campaign_root, apply=args.apply))
    elif args.command == "preflight":
        _print(preflight(args.campaign_root, apply=args.apply))
    elif args.command == "score":
        _print(
            score(
                args.campaign_root,
                apply=args.apply,
                max_workers=args.max_workers,
                device=args.device,
                num_workers=args.num_workers,
            )
        )
    elif args.command == "analyze":
        _print(
            analyze(
                args.campaign_root,
                apply=args.apply,
                n_bootstrap=args.n_bootstrap,
                test_only_noncanonical_parameters=args.test_only_noncanonical_parameters,
            )
        )
    elif args.command == "verify":
        _print(verify(args.campaign_root))
    elif args.command == "status":
        _print(status(args.campaign_root))
    elif args.command == "_score-one":
        _score_one(
            args.campaign_root,
            args.encoder,
            args.target,
            args.seed,
            device=args.device,
            num_workers=args.num_workers,
        )
    else:  # pragma: no cover
        raise GovernanceError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
