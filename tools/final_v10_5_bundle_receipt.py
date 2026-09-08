#!/usr/bin/env python3
"""Stage, verify, and exactly-once seal the additive FINAL-v10.5 bundle.

FINAL-v10.5 is an append-only Aim-1 source-cohort OOF extension over the
already sealed FINAL-v10 bundle.  Draft status recursively verifies the sealed
parent and directly rehashes all 81 parent source records.  New records remain
pending until their exact regular-file bytes exist.  Candidate construction is
fail closed until all 13 extension records, terminal receipt pins, final
document pins, and the publication timestamp are frozen.

Only --refresh-manifest mutates the draft manifest.  Only --seal can publish
report_bundle_receipt.json, using non-overwriting atomic publication.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import aim1_source_cohort_five_seed_campaign as campaign  # noqa: E402
from tools import final_v10_bundle_receipt as parent  # noqa: E402

FINAL_V10_5 = REPO / "reports" / "final_v10_5"
BASE_FINAL_V10 = REPO / "reports" / "final_v10"
BASE_RECEIPT = BASE_FINAL_V10 / "report_bundle_receipt.json"
BASE_MANIFEST = BASE_FINAL_V10 / "source_manifest.json"
BASE_VERIFIER = REPO / "tools" / "final_v10_bundle_receipt.py"
BASE_TEST = REPO / "tests" / "test_final_v10_bundle_receipt.py"
CAMPAIGN_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim1_primary_cohort_5seed_v1_20260824"
)
ANALYSIS_IMPLEMENTATION = REPO / "tools" / "aim1_source_cohort_five_seed_analysis.py"
ANALYSIS_TEST = REPO / "tests" / "test_aim1_source_cohort_five_seed_analysis.py"

REPORT_DOCUMENTS = ("Experimental_Setup.md", "Results.md", "Audit.md")
SOURCE_MANIFEST_NAME = "source_manifest.json"
FINAL_RECEIPT_NAME = "report_bundle_receipt.json"
SEALED_STATUS = parent.SEALED_STATUS
DRAFT_STATUS = "draft_awaiting_aim1_source_cohort_training_and_analysis"
CANDIDATE_STATUS = "candidate_ready_for_final_v10_5_verification"
EXPECTED_BASE_SOURCE_COUNT = 81
EXPECTED_NEW_SOURCE_COUNT = 13
EXPECTED_COMPLETE_SOURCE_COUNT = EXPECTED_BASE_SOURCE_COUNT + EXPECTED_NEW_SOURCE_COUNT
EXPECTED_ARMS = (
    "tcga_primary",
    "sr386_primary",
    "surgen_primary",
    "tcga_surgen_primary",
)
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_ARM_CENSUS = {
    "tcga_primary": (508, 502, 207, 295),
    "sr386_primary": (413, 413, 147, 266),
    "surgen_primary": (881, 737, 294, 443),
    "tcga_surgen_primary": (1389, 1239, 501, 738),
}

EXPECTED_FROZEN_DEPENDENCY_SHA256 = {
    "base_verifier": "e55eb82f7daf5f61838cd9ffeb49e185d45ab61cbc719061007938a055a6ac66",
    "base_test": "e78ef29a53a105fc4825adc261f1cd2a5e0629d992bb9e78b9673c58f1be3255",
    "base_manifest": "8f52982ea3ce2381d753de9bd6289dfd76228b069914f6f2157a7e5bc69a4b13",
    "base_receipt": "7d3ac2b82f71dd956c0e2b5ad7c951474ce7310c63c421f1a347b5e0c55f43e2",
    "campaign_controller": ("242b3be7823046d2e654165676992e59b57d8f246c9637bd28904d816bce5121"),
    "campaign_test": "cb92df888ead95c6b6b991376d2517917f799a56a9505b1a0249efe144636451",
    "analysis_implementation": ("196cb858f5158f83e45631d9bdfd1d64a09909ab605b415d7d4b77e90ee47193"),
    "analysis_test": "31131595449baffee7c05281dd10f889bf05bf9bdea5291ad84978cd5a8a7782",
}

_UNFROZEN = "REPLACE_AFTER_FINAL_V10_5_RECONCILIATION"
EXPECTED_FINAL_DOCUMENT_SHA256 = {
    "Experimental_Setup.md": "96716b973ddfeae881629bbad65e1d5456839c44f9141e2c2e8b6fc347295c9e",
    "Results.md": "5ba978a67cd2b2f7eb1a11994700d215c89b267c20d80ecf070f370df073e9ba",
    "Audit.md": "e308690c076bfc032f0a8aea572fcd9f6ff3c57da03eb1c9b081bc8b81b36565",
}
EXPECTED_TERMINAL_RECEIPT_SHA256 = {
    "aim1-source-cohort-five-seed-training-completion": "31cc8005abf32cb1c8958d0b4ecb5ed34e927bcd1665451869d62a913bb75413",
    "aim1-source-cohort-five-seed-analysis-completion": "82c33cf2d1f0a09338ce42371b4fc8dfe74696c53736a8cb10433591fd280543",
}
EXPECTED_RECEIPT_CREATED_UTC = "2026-08-24T22:25:00+00:00"

FINAL_STATE_STATUS_PARAGRAPH = (
    "FINAL-v10.5 evidence is complete: the sealed FINAL-v10 parent and all 81 parent "
    "sources were directly verified, all 13 governed extension sources are materialized "
    "with zero unresolved records, and the 100-fit within-source OOF extension passed "
    "deterministic analysis replay before exactly-once receipt publication."
)

_DRAFT_KEYS = {
    "schema_version",
    "bundle",
    "status",
    "base_bundle_receipt",
    "base_source_manifest",
    "artifacts",
    "pending_artifacts",
}
_FLAT_KEYS = {
    "schema_version",
    "bundle",
    "status",
    "artifacts",
    "pending_artifacts",
}
_ARTIFACT_KEYS = {
    "id",
    "aims",
    "experiments",
    "role",
    "path",
    "size_bytes",
    "sha256",
}
_PENDING_KEYS = {"id", "aims", "experiments", "role", "path"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SOURCE_ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_CANDIDATE_BLOCKERS = (
    re.compile(r"\b(?:draft|pending|awaiting|in progress|unsealed)\b", re.IGNORECASE),
    re.compile(r"\bno new (?:auroc|result|receipt)\b", re.IGNORECASE),
    re.compile(r"\b(?:target|planned)\s+(?:only|logical fits|oof fits)\b", re.IGNORECASE),
)
_CANDIDATE_CONTRADICTIONS = (
    re.compile(
        r"\bexternal[- ]transport(?:\s+(?:claim|gate))?\s*(?:is|=|:)?\s*"
        r"(?:established|supported|confirmed|true|passed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bconfirmatory\s+gate\s*(?:is|=|:)?\s*(?:true|passed|met)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcanonical\s+(?:final-v10\s+)?e0\s*(?:is|was|=|:)?\s*"
        r"(?:replaced|superseded|changed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmodel\s+seeds?\s+(?:are|were|as)\s+(?:independent\s+)?inference\s+units?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bfolds?\s+(?:are|were|as)\s+inference\s+units?\b", re.IGNORECASE),
)


class BundleVerificationError(RuntimeError):
    """A fail-closed FINAL-v10.5 staging, source, report, or seal error."""


BaseValidator = Callable[["BundlePaths"], dict[str, Any]]
StageValidator = Callable[["BundlePaths"], dict[str, Any] | None]


@dataclass(frozen=True)
class BundlePaths:
    """Filesystem locations and frozen candidate handoff values."""

    repo: Path
    final_v10_5: Path
    destination: Path
    verifier_code: Path
    verifier_test: Path
    base_final_v10: Path
    base_receipt: Path
    base_manifest: Path
    base_verifier: Path
    base_test: Path
    campaign_root: Path
    campaign_controller: Path
    campaign_test: Path
    analysis_implementation: Path
    analysis_test: Path
    expected_dependency_sha256: dict[str, str]
    expected_document_sha256: dict[str, str] | None = None
    expected_terminal_receipt_sha256: dict[str, str] | None = None
    expected_created_utc: str | None = None
    expected_base_source_count: int = EXPECTED_BASE_SOURCE_COUNT
    base_validator: BaseValidator | None = None
    campaign_validator: StageValidator | None = None
    analysis_validator: StageValidator | None = None


def default_paths() -> BundlePaths:
    """Return production FINAL-v10.5 paths."""

    return BundlePaths(
        repo=REPO,
        final_v10_5=FINAL_V10_5,
        destination=FINAL_V10_5 / FINAL_RECEIPT_NAME,
        verifier_code=Path(__file__).resolve(),
        verifier_test=REPO / "tests" / "test_final_v10_5_bundle_receipt.py",
        base_final_v10=BASE_FINAL_V10,
        base_receipt=BASE_RECEIPT,
        base_manifest=BASE_MANIFEST,
        base_verifier=BASE_VERIFIER,
        base_test=BASE_TEST,
        campaign_root=CAMPAIGN_ROOT,
        campaign_controller=REPO / "tools" / "aim1_source_cohort_five_seed_campaign.py",
        campaign_test=REPO / "tests" / "test_aim1_source_cohort_five_seed_campaign.py",
        analysis_implementation=ANALYSIS_IMPLEMENTATION,
        analysis_test=ANALYSIS_TEST,
        expected_dependency_sha256=dict(EXPECTED_FROZEN_DEPENDENCY_SHA256),
        expected_document_sha256=dict(EXPECTED_FINAL_DOCUMENT_SHA256),
        expected_terminal_receipt_sha256=dict(EXPECTED_TERMINAL_RECEIPT_SHA256),
        expected_created_utc=EXPECTED_RECEIPT_CREATED_UTC,
    )


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_symlink_chain(path: Path, *, context: str) -> None:
    candidate = path if path.is_absolute() else path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise BundleVerificationError(f"{context} contains a symlink component: {component}")


def identity(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    """Return an exact regular non-symlink file identity."""

    _reject_symlink_chain(path, context="artifact path")
    if not path.is_file():
        raise BundleVerificationError(f"expected a regular file: {path}")
    return {
        "path": str(path if display_path is None else display_path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise BundleVerificationError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _reject_nonfinite_json_constant(value: str) -> None:
    raise BundleVerificationError(f"non-finite JSON constant: {value}")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _reject_symlink_chain(path, context=label)
    if not path.is_file():
        raise BundleVerificationError(f"{label} must be a regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleVerificationError(f"{label} is not readable canonical JSON") from exc
    if not isinstance(value, dict):
        raise BundleVerificationError(f"{label} must contain a JSON object")
    return value


def _display(path: Path, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def _lexical_path(raw_path: str, paths: BundlePaths) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = paths.repo / candidate
        try:
            candidate.absolute().relative_to(paths.repo.absolute())
        except ValueError as exc:
            raise BundleVerificationError(
                f"relative source escapes repository: {raw_path}"
            ) from exc
    _reject_symlink_chain(candidate, context=f"declared source {raw_path}")
    return candidate


def _record_identity(record: Mapping[str, Any], paths: BundlePaths) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise BundleVerificationError("source record path is invalid")
    path = _lexical_path(raw_path, paths)
    actual = identity(path, display_path=raw_path)
    if actual["size_bytes"] != record.get("size_bytes") or actual["sha256"] != record.get("sha256"):
        raise BundleVerificationError(f"source identity drift: {record.get('id', raw_path)}")
    return path


def _source_specs(paths: BundlePaths) -> dict[str, dict[str, Any]]:
    experiment = ["Aim 1 source-cohort OOF extension"]
    common = {"aims": ["Aim 1"], "experiments": experiment}
    root = paths.campaign_root
    return {
        "aim1-source-cohort-five-seed-campaign-controller": {
            **common,
            "role": "governed_aim1_source_cohort_campaign_controller",
            "path": _display(paths.campaign_controller, paths.repo),
        },
        "aim1-source-cohort-five-seed-campaign-test": {
            **common,
            "role": "focused_campaign_controller_validation",
            "path": _display(paths.campaign_test, paths.repo),
        },
        "aim1-source-cohort-five-seed-campaign-contract": {
            **common,
            "role": "immutable_four_arm_five_seed_oof_campaign_contract",
            "path": str(root / "contract.json"),
        },
        "aim1-source-cohort-five-seed-deep-preflight": {
            **common,
            "role": "deep_input_roster_split_pack_and_resource_preflight",
            "path": str(root / "receipts/deep_preflight.json"),
        },
        "aim1-source-cohort-five-seed-scheduler": {
            **common,
            "role": "six_worker_execution_receipt",
            "path": str(root / "receipts/scheduler.json"),
        },
        "aim1-source-cohort-five-seed-training-completion": {
            **common,
            "role": "terminal_twenty_job_one_hundred_fit_zero_refit_training_receipt",
            "path": str(root / "receipts/training_complete.json"),
        },
        "aim1-source-cohort-five-seed-analysis-implementation": {
            **common,
            "role": "governed_patient_native_logit_and_paired_bootstrap_analysis",
            "path": _display(paths.analysis_implementation, paths.repo),
        },
        "aim1-source-cohort-five-seed-analysis-test": {
            **common,
            "role": "focused_analysis_validation",
            "path": _display(paths.analysis_test, paths.repo),
        },
        "aim1-source-cohort-five-seed-analysis-contract": {
            **common,
            "role": "immutable_patient_native_logit_analysis_contract",
            "path": str(root / "analysis/contract.json"),
        },
        "aim1-source-cohort-five-seed-patient-native-logits": {
            **common,
            "role": "governed_patient_native_logit_table",
            "path": str(root / "analysis/patient_native_logits.parquet"),
        },
        "aim1-source-cohort-five-seed-results": {
            **common,
            "role": "descriptive_within_source_oof_results",
            "path": str(root / "analysis/results.json"),
        },
        "aim1-source-cohort-five-seed-bootstrap": {
            **common,
            "role": "stratified_patient_bootstrap_distributions",
            "path": str(root / "analysis/bootstrap_distributions.npz"),
        },
        "aim1-source-cohort-five-seed-analysis-completion": {
            **common,
            "role": "terminal_deterministic_analysis_replay_receipt",
            "path": str(root / "analysis/analysis_completion_receipt.json"),
        },
    }


def _validate_dependency_pins(paths: BundlePaths) -> None:
    expected_paths = {
        "base_verifier": paths.base_verifier,
        "base_test": paths.base_test,
        "base_manifest": paths.base_manifest,
        "base_receipt": paths.base_receipt,
        "campaign_controller": paths.campaign_controller,
        "campaign_test": paths.campaign_test,
        "analysis_implementation": paths.analysis_implementation,
        "analysis_test": paths.analysis_test,
    }
    if set(paths.expected_dependency_sha256) != set(expected_paths):
        raise BundleVerificationError("frozen dependency pin roster changed")
    if paths.repo.resolve() == REPO.resolve() and (
        paths.expected_dependency_sha256 != EXPECTED_FROZEN_DEPENDENCY_SHA256
        or paths.base_validator is not None
        or paths.campaign_validator is not None
        or paths.analysis_validator is not None
    ):
        raise BundleVerificationError("production validation callbacks or dependency pins changed")
    for key, path in expected_paths.items():
        if identity(path)["sha256"] != paths.expected_dependency_sha256[key]:
            raise BundleVerificationError(f"frozen dependency identity drift: {key}")


def _base_parent_paths(paths: BundlePaths) -> parent.BundlePaths:
    return parent.BundlePaths(
        repo=paths.repo,
        final_v10=paths.base_final_v10,
        destination=paths.base_receipt,
        verifier_code=paths.base_verifier,
        verifier_test=paths.base_test,
        base_manifest=paths.repo / "reports/final_v9/source_manifest.json",
        core_verifier=paths.repo / "tools/final_v9_bundle_receipt.py",
        campaign_root=parent.core.FIVE_SEED_CAMPAIGN_ROOT,
        adjudication_root=parent.core.FIVE_SEED_ADJUDICATION_ROOT,
        expected_document_sha256=dict(parent.EXPECTED_FINAL_DOCUMENT_SHA256),
        expected_terminal_receipt_sha256=dict(parent.EXPECTED_TERMINAL_VERIFICATION_RECEIPT_SHA256),
        expected_created_utc=parent.EXPECTED_RECEIPT_CREATED_UTC,
    )


def _verify_base(paths: BundlePaths) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_dependency_pins(paths)
    if paths.base_validator is None:
        try:
            receipt = parent.verify_published_receipt(_base_parent_paths(paths))
        except parent.BundleVerificationError as exc:
            raise BundleVerificationError(f"sealed FINAL-v10 parent failed replay: {exc}") from exc
    else:
        receipt = paths.base_validator(paths)
    if not isinstance(receipt, dict) or receipt.get("status") != SEALED_STATUS:
        raise BundleVerificationError("base validator did not return a sealed FINAL-v10 receipt")
    sources = receipt.get("authoritative_sources")
    if not isinstance(sources, list) or len(sources) != paths.expected_base_source_count:
        raise BundleVerificationError("sealed FINAL-v10 parent source census changed")
    source_ids: set[str] = set()
    source_paths: set[Path] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or set(source) != _ARTIFACT_KEYS:
            raise BundleVerificationError(f"base source {index} has invalid shape")
        source_id = source.get("id")
        if not isinstance(source_id, str) or _SOURCE_ID_RE.fullmatch(source_id) is None:
            raise BundleVerificationError(f"base source {index} has invalid ID")
        if source_id in source_ids:
            raise BundleVerificationError(f"duplicate base source ID: {source_id}")
        source_ids.add(source_id)
        source_paths.add(_record_identity(source, paths).resolve())
    if len(source_paths) != len(sources):
        raise BundleVerificationError("base source ledger contains duplicate paths")
    manifest = _load_json(paths.base_manifest, label="sealed FINAL-v10 source manifest")
    manifest_sources = manifest.get("artifacts")
    if not isinstance(manifest_sources, list):
        raise BundleVerificationError("sealed FINAL-v10 manifest source ledger is invalid")
    manifest_by_id = {
        str(source.get("id")): source for source in manifest_sources if isinstance(source, dict)
    }
    receipt_by_id = {str(source["id"]): source for source in sources}
    if (
        manifest.get("status") != parent.CANDIDATE_STATUS
        or manifest.get("pending_artifacts") != []
        or len(manifest_by_id) != len(manifest_sources)
        or manifest_by_id != receipt_by_id
    ):
        raise BundleVerificationError("sealed FINAL-v10 manifest and receipt source ledgers differ")
    expected_manifest = receipt.get("source_manifest")
    expected_receipt = identity(
        paths.base_receipt, display_path=_display(paths.base_receipt, paths.repo)
    )
    if expected_manifest != identity(
        paths.base_manifest, display_path=_display(paths.base_manifest, paths.repo)
    ):
        raise BundleVerificationError("sealed FINAL-v10 receipt does not bind its live manifest")
    if expected_receipt["sha256"] != paths.expected_dependency_sha256["base_receipt"]:
        raise BundleVerificationError("sealed FINAL-v10 receipt pin drift")
    return receipt, sources


def _validate_source_record(
    record: Any,
    *,
    specs: Mapping[str, Mapping[str, Any]],
    pending: bool,
    location: str,
) -> dict[str, Any]:
    expected_keys = _PENDING_KEYS if pending else _ARTIFACT_KEYS
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise BundleVerificationError(f"{location} keys are not exact")
    source_id = record.get("id")
    if source_id not in specs:
        raise BundleVerificationError(f"{location} has an unexpected source ID: {source_id}")
    expected = {"id": source_id, **specs[str(source_id)]}
    for key, value in expected.items():
        if record.get(key) != value:
            raise BundleVerificationError(f"{location}.{key} differs from frozen metadata")
    if not pending and (
        isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or int(record["size_bytes"]) < 0
        or not isinstance(record.get("sha256"), str)
        or _SHA256_RE.fullmatch(str(record["sha256"])) is None
    ):
        raise BundleVerificationError(f"{location} has an invalid byte identity")
    return record


def _manifest_identity_record(path: Path, paths: BundlePaths) -> dict[str, Any]:
    return identity(path, display_path=_display(path, paths.repo))


def _validate_manifest(
    paths: BundlePaths, *, require_candidate: bool
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base_receipt, base_sources = _verify_base(paths)
    manifest_path = paths.final_v10_5 / SOURCE_MANIFEST_NAME
    manifest = _load_json(manifest_path, label="FINAL-v10.5 source manifest")
    flat = set(manifest) == _FLAT_KEYS
    if not flat and set(manifest) != _DRAFT_KEYS:
        raise BundleVerificationError("source manifest shape is neither exact draft nor exact flat")
    if manifest.get("schema_version") not in {1, 2} or manifest.get("bundle") != "final_v10_5":
        raise BundleVerificationError("source manifest header is invalid")
    if flat and manifest.get("schema_version") != 2:
        raise BundleVerificationError("flat candidate manifest must use schema version 2")
    if not flat:
        if manifest["base_bundle_receipt"] != _manifest_identity_record(paths.base_receipt, paths):
            raise BundleVerificationError("draft base FINAL-v10 receipt identity drift")
        if manifest["base_source_manifest"] != _manifest_identity_record(
            paths.base_manifest, paths
        ):
            raise BundleVerificationError("draft base FINAL-v10 manifest identity drift")
        if base_receipt.get("authoritative_sources") != base_sources:
            raise BundleVerificationError("base source replay returned inconsistent records")

    artifacts = manifest.get("artifacts")
    pending = manifest.get("pending_artifacts")
    if not isinstance(artifacts, list) or not isinstance(pending, list):
        raise BundleVerificationError("manifest artifacts and pending_artifacts must be lists")
    specs = _source_specs(paths)
    if len(specs) != EXPECTED_NEW_SOURCE_COUNT:
        raise BundleVerificationError("internal extension source roster is not exact")

    if flat:
        by_id = {str(item.get("id")): item for item in artifacts if isinstance(item, dict)}
        if len(by_id) != len(artifacts):
            raise BundleVerificationError("flat manifest has duplicate or invalid source IDs")
        base_by_id = {str(item["id"]): item for item in base_sources}
        if set(by_id) != set(base_by_id) | set(specs):
            raise BundleVerificationError("flat manifest source ID roster is not exact")
        for source_id, base_record in base_by_id.items():
            if by_id[source_id] != base_record:
                raise BundleVerificationError(f"adopted base source record drift: {source_id}")
        new_artifacts = [
            _validate_source_record(
                by_id[source_id],
                specs=specs,
                pending=False,
                location=f"artifacts[{source_id}]",
            )
            for source_id in specs
        ]
    else:
        new_artifacts = [
            _validate_source_record(
                item, specs=specs, pending=False, location=f"artifacts[{index}]"
            )
            for index, item in enumerate(artifacts)
        ]
        pending = [
            _validate_source_record(
                item, specs=specs, pending=True, location=f"pending_artifacts[{index}]"
            )
            for index, item in enumerate(pending)
        ]
        observed_ids = [str(item["id"]) for item in [*new_artifacts, *pending]]
        if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(specs):
            raise BundleVerificationError("draft extension source roster is not exact")

    for source in new_artifacts:
        _record_identity(source, paths)
    all_paths = [
        _lexical_path(str(source["path"]), paths).absolute()
        for source in [*base_sources, *new_artifacts, *pending]
    ]
    if len(all_paths) != len(set(all_paths)):
        raise BundleVerificationError("source ledger declares a path more than once")

    if flat:
        if pending or len(artifacts) != paths.expected_base_source_count + len(specs):
            raise BundleVerificationError("flat candidate must have 94 sources and zero pending")
        if manifest.get("status") != CANDIDATE_STATUS:
            raise BundleVerificationError("flat complete manifest is not candidate-ready")
    else:
        if manifest.get("status") != DRAFT_STATUS:
            raise BundleVerificationError("layered manifest must remain in draft status")
        if not pending:
            raise BundleVerificationError("complete layered manifest must be flattened atomically")
        if require_candidate:
            raise BundleVerificationError(
                f"FINAL-v10.5 candidate is blocked by {len(pending)} pending sources"
            )
    return manifest, base_sources, new_artifacts, pending


def _document_suffix(paths: BundlePaths, name: str) -> str:
    base = paths.base_final_v10 / name
    extension = paths.final_v10_5 / name
    _reject_symlink_chain(base, context=f"base document {name}")
    _reject_symlink_chain(extension, context=f"extension document {name}")
    if not base.is_file() or not extension.is_file():
        raise BundleVerificationError(f"missing report document: {name}")
    base_bytes = base.read_bytes()
    extension_bytes = extension.read_bytes()
    if not extension_bytes.startswith(base_bytes):
        raise BundleVerificationError(f"FINAL-v10 continuity prefix drift: {name}")
    suffix = extension_bytes[len(base_bytes) :]
    try:
        return suffix.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleVerificationError(f"extension document is not UTF-8: {name}") from exc


def _validate_document_pins(paths: BundlePaths) -> None:
    pins = paths.expected_document_sha256
    if not isinstance(pins, dict) or set(pins) != set(REPORT_DOCUMENTS):
        raise BundleVerificationError("FINAL-v10.5 document pin roster is not frozen")
    if paths.repo.resolve() == REPO.resolve() and pins != EXPECTED_FINAL_DOCUMENT_SHA256:
        raise BundleVerificationError("production FINAL-v10.5 document pins were overridden")
    for name, expected in pins.items():
        if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
            raise BundleVerificationError("FINAL-v10.5 document pins remain unfrozen")
        if sha256_file(paths.final_v10_5 / name) != expected:
            raise BundleVerificationError(f"FINAL-v10.5 document identity drift: {name}")


def _validate_documents(paths: BundlePaths, *, require_candidate: bool) -> dict[str, Any]:
    suffixes = {name: _document_suffix(paths, name) for name in REPORT_DOCUMENTS}
    required = {
        "Experimental_Setup.md": (
            "Additive FINAL-v10.5 Aim 1 within-source primary-cohort OOF extension",
            "20 chains",
            "exactly 100",
            "not an external-transport experiment",
            "not a confirmatory gate",
        ),
        "Results.md": (
            "Additive FINAL-v10.5 Aim 1 within-source primary-cohort OOF extension",
            "descriptive within source",
            "neither an external-transport estimand",
        ),
        "Audit.md": (
            "FINAL-v10.5 additive audit state",
            "20 arm-seed chains",
            "100 logical fold fits",
            "zero refits",
            "extension is descriptive within source",
        ),
    }
    for name, phrases in required.items():
        missing = [phrase for phrase in phrases if phrase not in suffixes[name]]
        if missing:
            raise BundleVerificationError(
                f"{name} extension contract text is incomplete: {missing}"
            )
    combined = "\n".join(suffixes.values())
    if require_candidate:
        _validate_document_pins(paths)
        for pattern in _CANDIDATE_BLOCKERS:
            if pattern.search(combined):
                raise BundleVerificationError(
                    f"candidate documents retain a draft-state marker: {pattern.pattern}"
                )
        for pattern in _CANDIDATE_CONTRADICTIONS:
            if pattern.search(combined):
                raise BundleVerificationError(
                    f"candidate documents contain a scope contradiction: {pattern.pattern}"
                )
        if sum(FINAL_STATE_STATUS_PARAGRAPH in text for text in suffixes.values()) != 3:
            raise BundleVerificationError(
                "each candidate document must contain the exact completion paragraph"
            )
    return {
        name: identity(
            paths.final_v10_5 / name,
            display_path=_display(paths.final_v10_5 / name, paths.repo),
        )
        for name in REPORT_DOCUMENTS
    }


def _source_map(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(source["id"]): source for source in sources}


def _source_json(
    by_id: Mapping[str, Mapping[str, Any]], source_id: str, paths: BundlePaths
) -> dict[str, Any]:
    source = by_id.get(source_id)
    if source is None:
        raise BundleVerificationError(f"missing source: {source_id}")
    return _load_json(_record_identity(source, paths), label=source_id)


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleVerificationError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise BundleVerificationError(f"{context} must be finite")
    return number


def _metric_cell(point: Any, interval: Any, *, context: str, signed: bool = False) -> str:
    value = _finite_float(point, context=f"{context}.point")
    if not isinstance(interval, list) or len(interval) != 2:
        raise BundleVerificationError(f"{context}.ci95 must contain two bounds")
    lower = _finite_float(interval[0], context=f"{context}.ci95[0]")
    upper = _finite_float(interval[1], context=f"{context}.ci95[1]")
    if lower > value or value > upper:
        raise BundleVerificationError(f"{context} point is outside its ordered interval")
    formatter = "+.4f" if signed else ".4f"
    return f"{format(value, formatter)} [{format(lower, formatter)}, {format(upper, formatter)}]"


def _markdown_section(text: str, heading: str) -> str:
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines) if line.strip() == heading]
    if len(matches) != 1:
        raise BundleVerificationError(f"expected exactly one Markdown section: {heading}")
    start = matches[0]
    level = len(heading) - len(heading.lstrip("#"))
    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = re.match(r"^(#+)\s", lines[index])
        if match is not None and len(match.group(1)) <= level:
            end = index
            break
    return "\n".join(lines[start:end])


def _require_markdown_row(section: str, label: str, expected_cells: list[str]) -> None:
    matches = []
    for line in section.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == label:
            matches.append(cells)
    if matches != [expected_cells]:
        raise BundleVerificationError(f"Markdown row is missing, duplicated, or drifted: {label}")


def _validate_candidate_claims(paths: BundlePaths, new_sources: list[dict[str, Any]]) -> None:
    by_id = _source_map(new_sources)
    results = _source_json(by_id, "aim1-source-cohort-five-seed-results", paths)
    setup = _document_suffix(paths, "Experimental_Setup.md")
    result_text = _document_suffix(paths, "Results.md")
    audit = _document_suffix(paths, "Audit.md")
    setup_section = _markdown_section(
        setup,
        "## Additive FINAL-v10.5 Aim 1 within-source primary-cohort OOF extension",
    )
    setup_rows = {
        "tcga_primary": [
            "tcga_primary",
            "TCGA conventional-primary only",
            "508",
            "502",
            "207",
            "295",
        ],
        "sr386_primary": [
            "sr386_primary",
            "SurGen SR386 primary only",
            "413",
            "413",
            "147",
            "266",
        ],
        "surgen_primary": [
            "surgen_primary",
            "SurGen SR386 + SR1482 primary",
            "881",
            "737",
            "294",
            "443",
        ],
        "tcga_surgen_primary": [
            "tcga_surgen_primary",
            "TCGA + SurGen primary",
            "1,389",
            "1,239",
            "501",
            "738",
        ],
    }
    for label, cells in setup_rows.items():
        _require_markdown_row(setup_section, label, cells)

    arm_section = _markdown_section(
        result_text, "### Governed FINAL-v10.5 within-source OOF arm results"
    )
    arm_performance = results["arm_performance"]
    for arm in EXPECTED_ARMS:
        record = arm_performance[arm]
        population = record.get("population")
        metrics = record.get("five_seed_mean_native_logit")
        if not isinstance(population, dict) or not isinstance(metrics, dict):
            raise BundleVerificationError(f"analysis arm record is incomplete: {arm}")
        expected_population = EXPECTED_ARM_CENSUS[arm]
        observed_population = (
            population.get("patients"),
            population.get("mutant"),
            population.get("wild_type"),
        )
        if observed_population != expected_population[1:]:
            raise BundleVerificationError(f"analysis arm population drift: {arm}")
        _require_markdown_row(
            arm_section,
            arm,
            [
                arm,
                str(population["patients"]),
                str(population["mutant"]),
                str(population["wild_type"]),
                _metric_cell(
                    metrics.get("auroc"),
                    metrics.get("auroc_ci95"),
                    context=f"arm_performance.{arm}.auroc",
                ),
                _metric_cell(
                    metrics.get("auprc"),
                    metrics.get("auprc_ci95"),
                    context=f"arm_performance.{arm}.auprc",
                ),
            ],
        )

    contrast_section = _markdown_section(
        result_text, "### Governed FINAL-v10.5 paired common-patient contrasts"
    )
    contrasts = results["paired_common_patient_contrasts"]
    for key, record in contrasts.items():
        auroc = record.get("auroc")
        auprc = record.get("auprc")
        if not isinstance(auroc, dict) or not isinstance(auprc, dict):
            raise BundleVerificationError(f"analysis contrast metrics are incomplete: {key}")
        _require_markdown_row(
            contrast_section,
            key,
            [
                key,
                str(record["evaluation_population"]),
                str(record["patients"]),
                str(record["mutant"]),
                str(record["wild_type"]),
                _metric_cell(
                    auroc.get("delta_larger_minus_smaller"),
                    auroc.get("delta_ci95"),
                    context=f"paired.{key}.auroc",
                    signed=True,
                ),
                _metric_cell(
                    auprc.get("delta_larger_minus_smaller"),
                    auprc.get("delta_ci95"),
                    context=f"paired.{key}.auprc",
                    signed=True,
                ),
            ],
        )

    identity_section = _markdown_section(
        audit, "### Governed FINAL-v10.5 extension source identities"
    )
    for source in sorted(new_sources, key=lambda item: str(item["id"])):
        _require_markdown_row(
            identity_section,
            str(source["id"]),
            [str(source["id"]), str(source["sha256"])],
        )


def _validate_campaign_contract(
    paths: BundlePaths, new_sources: list[dict[str, Any]], *, candidate: bool
) -> None:
    by_id = _source_map(new_sources)
    contract = _source_json(by_id, "aim1-source-cohort-five-seed-campaign-contract", paths)
    expected_header = {
        "campaign": "aim1_primary_cohort_oof_5seed",
        "arms": list(EXPECTED_ARMS),
        "seeds": list(EXPECTED_SEEDS),
        "n_folds": 5,
        "job_count": 20,
        "logical_fit_count": 100,
        "refit_count": 0,
        "max_parallel_training_processes": 6,
        "training_policy": "five OOF folds only; training.skip_finalize=true; no final/refit",
    }
    drift = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in expected_header.items()
        if contract.get(key) != value
    }
    if drift or contract.get("material_recipe", {}).get("skip_finalize") is not True:
        raise BundleVerificationError(f"Aim-1 source-cohort contract drift: {drift}")
    definitions = contract.get("arm_definitions")
    if not isinstance(definitions, dict) or set(definitions) != set(EXPECTED_ARMS):
        raise BundleVerificationError("campaign arm-definition roster is not exact")
    for arm, census in EXPECTED_ARM_CENSUS.items():
        observed = definitions[arm]
        keys = (
            "expected_slides",
            "expected_patients",
            "expected_mutant_patients",
            "expected_wildtype_patients",
        )
        if tuple(observed.get(key) for key in keys) != census:
            raise BundleVerificationError(f"campaign census drift: {arm}")
    jobs = contract.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 20:
        raise BundleVerificationError("campaign contract does not contain exactly 20 jobs")
    observed_jobs = set()
    expected_job_ids = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise BundleVerificationError("campaign job is not an object")
        arm, seed = job.get("arm"), job.get("seed")
        observed_jobs.add((arm, seed))
        expected_job_ids.add(job.get("job_id"))
        training_command = job.get("training_command")
        if (
            arm not in EXPECTED_ARMS
            or seed not in EXPECTED_SEEDS
            or job.get("fit_count") != 5
            or job.get("refit_count") != 0
            or not isinstance(training_command, list)
            or "training.skip_finalize=true" not in training_command
            or "training.skip_finalize=false" in training_command
            or f"splits.seed={seed}" not in training_command
            or f"training.seed={seed}" not in training_command
        ):
            raise BundleVerificationError(f"campaign job contract drift: {job.get('job_id')}")
    if observed_jobs != {(arm, seed) for arm in EXPECTED_ARMS for seed in EXPECTED_SEEDS}:
        raise BundleVerificationError("campaign arm-seed matrix is not exact")

    preflight = _source_json(by_id, "aim1-source-cohort-five-seed-deep-preflight", paths)
    contract_record = by_id["aim1-source-cohort-five-seed-campaign-contract"]
    expected_contract_identity = {
        key: contract_record[key] for key in ("path", "size_bytes", "sha256")
    }
    if (
        preflight.get("status") != "deep_preflight_passed"
        or preflight.get("contract") != expected_contract_identity
        or preflight.get("job_count") != 20
        or preflight.get("logical_fit_count") != 100
        or preflight.get("refit_count") != 0
        or preflight.get("scheduler_ceiling") != 6
    ):
        raise BundleVerificationError("campaign deep-preflight receipt drift")
    if not candidate:
        return
    scheduler = _source_json(by_id, "aim1-source-cohort-five-seed-scheduler", paths)
    if (
        scheduler.get("status") != "completed_rc0"
        or scheduler.get("configured_max_workers") != 6
        or scheduler.get("observed_peak_parallel_workers") != 6
        or scheduler.get("job_count") != 20
        or scheduler.get("logical_fit_count") != 100
        or scheduler.get("refit_count") != 0
    ):
        raise BundleVerificationError("scheduler receipt does not prove exact six-way execution")
    events = scheduler.get("events")
    event_keys = {
        "job_id",
        "pid",
        "returncode",
        "started_utc",
        "finished_utc",
        "started_monotonic",
        "finished_monotonic",
    }
    if not isinstance(events, list) or len(events) != 20:
        raise BundleVerificationError("scheduler receipt event census is not exact")
    event_ids = set()
    points: list[tuple[float, int]] = []
    for event in events:
        if not isinstance(event, dict) or set(event) != event_keys:
            raise BundleVerificationError("scheduler event field roster is not exact")
        started = event.get("started_monotonic")
        finished = event.get("finished_monotonic")
        if (
            not isinstance(event.get("job_id"), str)
            or isinstance(event.get("pid"), bool)
            or not isinstance(event.get("pid"), int)
            or event["pid"] <= 0
            or event.get("returncode") != 0
            or not isinstance(started, (int, float))
            or isinstance(started, bool)
            or not isinstance(finished, (int, float))
            or isinstance(finished, bool)
            or not math.isfinite(float(started))
            or not math.isfinite(float(finished))
            or float(finished) < float(started)
        ):
            raise BundleVerificationError("scheduler event is not a successful finite interval")
        event_ids.add(event["job_id"])
        points.extend(((float(started), 1), (float(finished), -1)))
    active = peak = 0
    for _, delta in sorted(points, key=lambda item: (item[0], -item[1])):
        active += delta
        peak = max(peak, active)
    if (
        event_ids != expected_job_ids
        or len(event_ids) != 20
        or len(expected_job_ids) != 20
        or peak != 6
    ):
        raise BundleVerificationError("scheduler events do not prove the exact 20-job six-way run")
    training = _source_json(by_id, "aim1-source-cohort-five-seed-training-completion", paths)
    expected_training_keys = {
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
    if (
        set(training) != expected_training_keys
        or training.get("status") != "complete_and_certified"
        or training.get("arms") != list(EXPECTED_ARMS)
        or training.get("seeds") != list(EXPECTED_SEEDS)
        or training.get("job_count") != 20
        or training.get("folds_per_job") != 5
        or training.get("logical_fit_count") != 100
        or training.get("refit_count") != 0
    ):
        raise BundleVerificationError("terminal training receipt census drift")
    preflight_record = by_id["aim1-source-cohort-five-seed-deep-preflight"]
    scheduler_record = by_id["aim1-source-cohort-five-seed-scheduler"]

    def exact_identity(record: Mapping[str, Any]) -> dict[str, Any]:
        return {key: record[key] for key in ("path", "size_bytes", "sha256")}

    if (
        training.get("contract") != exact_identity(contract_record)
        or training.get("preflight") != exact_identity(preflight_record)
        or training.get("scheduler") != exact_identity(scheduler_record)
    ):
        raise BundleVerificationError("terminal training receipt input identity drift")
    job_receipts = training.get("job_receipts")
    if not isinstance(job_receipts, list) or len(job_receipts) != 20:
        raise BundleVerificationError("terminal training receipt lacks 20 job identities")
    observed_receipt_paths = set()
    for record in job_receipts:
        if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
            raise BundleVerificationError("job receipt identity field roster drift")
        observed_receipt_paths.add(str(_record_identity(record, paths).resolve()))
    expected_receipt_paths = {
        str(campaign.job_receipt_path(paths.campaign_root, arm, seed).resolve())
        for arm in EXPECTED_ARMS
        for seed in EXPECTED_SEEDS
    }
    if observed_receipt_paths != expected_receipt_paths:
        raise BundleVerificationError("terminal training job-receipt roster drift")
    if paths.campaign_validator is not None:
        paths.campaign_validator(paths)
    else:
        try:
            campaign.validate_contract(paths.campaign_root, deep=True)
            campaign._validate_preflight(paths.campaign_root)  # noqa: SLF001
            with contextlib.redirect_stdout(io.StringIO()):
                campaign.cmd_validate(
                    argparse.Namespace(output_root=paths.campaign_root, seal=False)
                )
        except (campaign.ContractError, OSError, ValueError, SystemExit) as exc:
            raise BundleVerificationError(f"full 100-fit campaign replay failed: {exc}") from exc


def _validate_terminal_pins(paths: BundlePaths, new_sources: list[dict[str, Any]]) -> None:
    pins = paths.expected_terminal_receipt_sha256
    if not isinstance(pins, dict) or set(pins) != set(EXPECTED_TERMINAL_RECEIPT_SHA256):
        raise BundleVerificationError("terminal receipt pin roster is not frozen")
    if paths.repo.resolve() == REPO.resolve() and pins != EXPECTED_TERMINAL_RECEIPT_SHA256:
        raise BundleVerificationError("production terminal receipt pins were overridden")
    by_id = _source_map(new_sources)
    for source_id, expected in pins.items():
        if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
            raise BundleVerificationError("terminal receipt pins remain unfrozen")
        if by_id.get(source_id, {}).get("sha256") != expected:
            raise BundleVerificationError(f"terminal receipt pin drift: {source_id}")


def _load_analysis_module(paths: BundlePaths) -> Any:
    spec = importlib.util.spec_from_file_location(
        "_governed_aim1_source_cohort_analysis", paths.analysis_implementation
    )
    if spec is None or spec.loader is None:
        raise BundleVerificationError("cannot import governed analysis implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise BundleVerificationError("governed analysis implementation import failed") from exc
    return module


def _validate_bootstrap_archive(paths: BundlePaths) -> None:
    arm_names = {
        f"arm__{arm}__ensemble__{metric}" for arm in EXPECTED_ARMS for metric in ("auroc", "auprc")
    }
    contrast_keys = (
        "surgen_minus_sr386_on_sr386",
        "tcga_surgen_minus_surgen_on_surgen",
        "tcga_surgen_minus_tcga_on_tcga",
    )
    contrast_names = {
        f"contrast__{key}__{component}__{metric}"
        for key in contrast_keys
        for component in ("larger", "smaller", "delta")
        for metric in ("auroc", "auprc")
    }
    expected = arm_names | contrast_names
    archive_path = paths.campaign_root / "analysis/bootstrap_distributions.npz"
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != expected or len(archive.files) != 26:
                raise BundleVerificationError("analysis bootstrap archive must contain 26 arrays")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise BundleVerificationError("analysis bootstrap archive is unreadable") from exc
    for name, values in arrays.items():
        if values.dtype != np.float64 or values.shape != (10_000,) or not np.isfinite(values).all():
            raise BundleVerificationError(f"analysis bootstrap array contract drift: {name}")
    for key in contrast_keys:
        for metric in ("auroc", "auprc"):
            prefix = f"contrast__{key}"
            if not np.array_equal(
                arrays[f"{prefix}__delta__{metric}"],
                arrays[f"{prefix}__larger__{metric}"] - arrays[f"{prefix}__smaller__{metric}"],
            ):
                raise BundleVerificationError(
                    f"analysis paired bootstrap delta identity drift: {key}/{metric}"
                )


def _validate_analysis(paths: BundlePaths, new_sources: list[dict[str, Any]]) -> None:
    by_id = _source_map(new_sources)
    results = _source_json(by_id, "aim1-source-cohort-five-seed-results", paths)
    required_result_keys = {
        "schema_version",
        "status",
        "experiment",
        "design_status",
        "score_contract",
        "inference",
        "arm_performance",
        "paired_common_patient_contrasts",
        "cross_population_ranking",
        "scope_boundary",
        "inputs",
    }
    if (
        set(results) != required_result_keys
        or results.get("status") != "complete"
        or results.get("experiment") != "aim1_primary_cohort_oof_5seed_analysis"
        or results.get("design_status") != "additive_FINAL_v10_5_source_cohort_OOF_analysis"
    ):
        raise BundleVerificationError("analysis result schema is not exact")
    if not isinstance(results.get("arm_performance"), dict) or set(
        results["arm_performance"]
    ) != set(EXPECTED_ARMS):
        raise BundleVerificationError("analysis arm result roster is not exact")
    inference = results.get("inference")
    if not isinstance(inference, dict):
        raise BundleVerificationError("analysis inference contract is missing")
    if (
        inference.get("unit") != "patient"
        or inference.get("bootstrap_draws") != 10_000
        or inference.get("bootstrap_seed") != 20260824
        or inference.get("stratification") != "subcohort_x_KRAS_label"
        or inference.get("model_seeds_are_inference_units") is not False
        or inference.get("folds_are_inference_units") is not False
        or inference.get("paired_indices_shared") is not True
        or inference.get("bootstrap_array_count") != 26
    ):
        raise BundleVerificationError("analysis promotes seeds/folds or breaks paired resampling")
    score_contract = results.get("score_contract")
    if (
        not isinstance(score_contract, dict)
        or score_contract.get("probability_roundtrip_used") is not False
        or score_contract.get("model_seeds") != list(EXPECTED_SEEDS)
    ):
        raise BundleVerificationError("analysis native-logit score contract drift")
    expected_contrasts = {
        "surgen_minus_sr386_on_sr386": (
            "surgen_primary",
            "sr386_primary",
            "sr386_primary",
            413,
            147,
            266,
        ),
        "tcga_surgen_minus_surgen_on_surgen": (
            "tcga_surgen_primary",
            "surgen_primary",
            "surgen_primary",
            737,
            294,
            443,
        ),
        "tcga_surgen_minus_tcga_on_tcga": (
            "tcga_surgen_primary",
            "tcga_primary",
            "tcga_primary",
            502,
            207,
            295,
        ),
    }
    contrasts = results.get("paired_common_patient_contrasts")
    if not isinstance(contrasts, dict) or set(contrasts) != set(expected_contrasts):
        raise BundleVerificationError("analysis paired-contrast roster is not exact")
    for key, expected in expected_contrasts.items():
        contrast = contrasts[key]
        observed = (
            contrast.get("larger_arm"),
            contrast.get("smaller_arm"),
            contrast.get("evaluation_population"),
            contrast.get("patients"),
            contrast.get("mutant"),
            contrast.get("wild_type"),
        )
        if observed != expected or contrast.get("paired_indices_shared") is not True:
            raise BundleVerificationError(f"analysis paired-contrast drift: {key}")
    ranking = results.get("cross_population_ranking")
    if (
        not isinstance(ranking, dict)
        or ranking.get("role") != "descriptive_only"
        or ranking.get("no_cross_population_inference_or_transport_claim") is not True
    ):
        raise BundleVerificationError("cross-population ranking was promoted beyond descriptive")
    scope = results.get("scope_boundary")
    if not isinstance(scope, dict) or (
        scope.get("canonical_final_v10_e0_unchanged") is not True
        or scope.get("append_only_to_final_v10") is not True
        or scope.get("supersedes_parent_fields") != []
    ):
        raise BundleVerificationError("analysis scope boundary does not preserve FINAL-v10")
    if paths.analysis_validator is not None:
        result = paths.analysis_validator(paths)
    else:
        module = _load_analysis_module(paths)
        verifier = getattr(module, "verify", None)
        if not callable(verifier):
            raise BundleVerificationError("governed analysis exposes no read-only verify API")
        try:
            result = verifier(
                paths.campaign_root,
                parent_final_v10_receipt=paths.base_receipt,
                expected_parent_sha256=paths.expected_dependency_sha256["base_receipt"],
            )
        except Exception as exc:
            raise BundleVerificationError(f"deterministic analysis replay failed: {exc}") from exc
        _validate_bootstrap_archive(paths)
    if result != results or str(result.get("status", "")).casefold() not in {
        "pass",
        "complete",
        "complete_and_certified",
        "completed_and_verified",
    }:
        raise BundleVerificationError("analysis verifier did not return terminal PASS")


def draft_status(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Recursively validate the parent and report the unsealed staging state."""

    selected = default_paths() if paths is None else paths
    manifest, base_sources, new_sources, pending = _validate_manifest(
        selected, require_candidate=False
    )
    flat = set(manifest) == _FLAT_KEYS
    _validate_documents(selected, require_candidate=flat)
    if flat:
        _validate_terminal_pins(selected, new_sources)
        _validate_campaign_contract(selected, new_sources, candidate=True)
        _validate_analysis(selected, new_sources)
        _validate_candidate_claims(selected, new_sources)
    else:
        _validate_campaign_contract(selected, new_sources, candidate=False)
    ready = []
    missing = []
    for source in pending:
        path = _lexical_path(str(source["path"]), selected)
        (ready if path.is_file() else missing).append(str(source["path"]))
    return {
        "bundle": "reports/final_v10_5",
        "status": manifest["status"],
        "published_receipt_present": selected.destination.exists(),
        "sealed_parent_source_count": len(base_sources),
        "materialized_extension_source_count": len(new_sources),
        "pending_extension_source_count": len(pending),
        "pending_now_materialized_count": len(ready),
        "missing_source_count": len(missing),
        "pending_now_materialized": ready,
        "missing_sources": missing,
        "authenticated_fit_census": 1225,
        "pending_complete_fit_census": 1325,
        "parent_recursive_verification": "PASS",
        "parent_direct_81_source_rehash": "PASS",
    }


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _atomic_replace(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def refresh_manifest(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Promote available pending sources without publishing a receipt."""

    selected = default_paths() if paths is None else paths
    if selected.destination.exists() or selected.destination.is_symlink():
        raise BundleVerificationError("refusing manifest refresh after receipt publication")
    manifest, base_sources, new_sources, pending = _validate_manifest(
        selected, require_candidate=False
    )
    if set(manifest) == _FLAT_KEYS:
        return draft_status(selected)
    promoted = []
    remaining = []
    for source in pending:
        path = _lexical_path(str(source["path"]), selected)
        if path.is_file():
            promoted.append(
                {
                    **source,
                    **identity(path, display_path=str(source["path"])),
                }
            )
        else:
            remaining.append(source)
    if not promoted:
        return draft_status(selected)
    new_sources = [*new_sources, *promoted]
    if remaining:
        updated = {
            **manifest,
            "artifacts": new_sources,
            "pending_artifacts": remaining,
            "status": DRAFT_STATUS,
        }
    else:
        updated = {
            "schema_version": 2,
            "bundle": "final_v10_5",
            "status": CANDIDATE_STATUS,
            "artifacts": sorted([*base_sources, *new_sources], key=lambda item: str(item["id"])),
            "pending_artifacts": [],
        }
        _validate_documents(selected, require_candidate=True)
        _validate_terminal_pins(selected, new_sources)
        _validate_campaign_contract(selected, new_sources, candidate=True)
        _validate_analysis(selected, new_sources)
        _validate_candidate_claims(selected, new_sources)
    _atomic_replace(selected.final_v10_5 / SOURCE_MANIFEST_NAME, _manifest_bytes(updated))
    return draft_status(selected)


def _validated_created_utc(paths: BundlePaths) -> str:
    value = paths.expected_created_utc
    if paths.repo.resolve() == REPO.resolve() and value != EXPECTED_RECEIPT_CREATED_UTC:
        raise BundleVerificationError("production receipt timestamp was overridden")
    if not isinstance(value, str) or value == _UNFROZEN:
        raise BundleVerificationError("FINAL-v10.5 receipt timestamp remains unfrozen")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise BundleVerificationError("receipt timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise BundleVerificationError("receipt timestamp must be canonical UTC")
    if parsed.isoformat() != value:
        raise BundleVerificationError("receipt timestamp must use +00:00 canonical form")
    return value


def build_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Build, but never publish, the fully validated candidate receipt."""

    selected = default_paths() if paths is None else paths
    manifest, base_sources, new_sources, pending = _validate_manifest(
        selected, require_candidate=True
    )
    if pending:
        raise BundleVerificationError("candidate retained pending sources")
    documents = _validate_documents(selected, require_candidate=True)
    _validate_terminal_pins(selected, new_sources)
    _validate_campaign_contract(selected, new_sources, candidate=True)
    _validate_analysis(selected, new_sources)
    _validate_candidate_claims(selected, new_sources)
    all_sources = sorted([*base_sources, *new_sources], key=lambda item: str(item["id"]))
    expected_complete = selected.expected_base_source_count + EXPECTED_NEW_SOURCE_COUNT
    if len(all_sources) != expected_complete:
        raise BundleVerificationError("complete source census is not exact")
    return {
        "schema_version": 1,
        "bundle": "reports/final_v10_5",
        "status": SEALED_STATUS,
        "created_utc": _validated_created_utc(selected),
        "organization": "append_only_aim1_source_cohort_oof_extension",
        "base_final_v10": {
            "receipt": _manifest_identity_record(selected.base_receipt, selected),
            "source_manifest": _manifest_identity_record(selected.base_manifest, selected),
            "source_count": selected.expected_base_source_count,
            "recursive_verification": "PASS",
            "direct_source_rehash": "PASS",
        },
        "scope": {
            "aim": "Aim 1",
            "design": "descriptive_within_source_oof",
            "external_transport_claim": False,
            "confirmatory_gate": False,
            "canonical_final_v10_e0_unchanged": True,
            "arms": list(EXPECTED_ARMS),
            "model_seeds": list(EXPECTED_SEEDS),
            "model_seeds_are_inference_units": False,
            "folds_are_inference_units": False,
        },
        "fit_census": {
            "sealed_final_v10": 1225,
            "new_oof_fits": 100,
            "new_refits": 0,
            "complete": 1325,
            "new_training_jobs": 20,
            "maximum_concurrent_gpu_trainers": 6,
            "observed_peak_concurrent_gpu_trainers": 6,
        },
        "documents": documents,
        "source_manifest": identity(
            selected.final_v10_5 / SOURCE_MANIFEST_NAME,
            display_path=_display(selected.final_v10_5 / SOURCE_MANIFEST_NAME, selected.repo),
        ),
        "authoritative_sources": all_sources,
        "verification": {
            "verifier": identity(
                selected.verifier_code,
                display_path=_display(selected.verifier_code, selected.repo),
            ),
            "tests": identity(
                selected.verifier_test,
                display_path=_display(selected.verifier_test, selected.repo),
            ),
            "frozen_final_v10_verifier": identity(
                selected.base_verifier,
                display_path=_display(selected.base_verifier, selected.repo),
            ),
            "campaign_controller": identity(
                selected.campaign_controller,
                display_path=_display(selected.campaign_controller, selected.repo),
            ),
            "analysis_implementation": identity(
                selected.analysis_implementation,
                display_path=_display(selected.analysis_implementation, selected.repo),
            ),
        },
        "checks": {
            "sealed_final_v10_recursive_verification": "PASS",
            "direct_81_parent_source_rehash": "PASS",
            "verbatim_final_v10_document_continuity": "PASS",
            "exact_13_source_extension_inventory": "PASS",
            "direct_extension_source_rehash": "PASS",
            "exact_20_job_100_fit_zero_refit_campaign": "PASS",
            "five_inherited_patient_folds_across_seeds": "PASS",
            "patient_native_logit_analysis_replay": "PASS",
            "paired_shared_population_patient_bootstrap": "PASS",
            "descriptive_nontransport_scope": "PASS",
            "final_document_byte_pins": "PASS",
            "terminal_receipt_byte_pins": "PASS",
        },
    }


def _receipt_bytes(receipt: dict[str, Any]) -> bytes:
    return (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_published_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Verify the exact bytes of an already published receipt."""

    selected = default_paths() if paths is None else paths
    _reject_symlink_chain(selected.destination, context="published receipt")
    if not selected.destination.is_file():
        raise BundleVerificationError("published FINAL-v10.5 receipt is absent")
    published = _load_json(selected.destination, label="published FINAL-v10.5 receipt")
    expected = build_receipt(selected)
    if published != expected or selected.destination.read_bytes() != _receipt_bytes(expected):
        raise BundleVerificationError("published FINAL-v10.5 receipt byte identity drift")
    return published


def seal(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Publish the candidate receipt atomically and exactly once."""

    selected = default_paths() if paths is None else paths
    if selected.destination.exists() or selected.destination.is_symlink():
        raise BundleVerificationError("refusing to overwrite FINAL-v10.5 receipt")
    receipt = build_receipt(selected)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{selected.destination.name}.",
        suffix=".tmp",
        dir=selected.destination.parent,
    )
    temporary = Path(temporary_name)
    temporary_inode: tuple[int, int] | None = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_receipt_bytes(receipt))
            handle.flush()
            os.fsync(handle.fileno())
            stat = os.fstat(handle.fileno())
            temporary_inode = (stat.st_dev, stat.st_ino)
        try:
            os.link(temporary, selected.destination)
        except FileExistsError as exc:
            raise BundleVerificationError("receipt was concurrently published") from exc
        _fsync_directory(selected.destination.parent)
        temporary.unlink(missing_ok=True)
        return verify_published_receipt(selected)
    except BaseException:
        try:
            observed = selected.destination.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if temporary_inode == (observed.st_dev, observed.st_ino):
                with contextlib.suppress(FileNotFoundError):
                    selected.destination.unlink()
                with contextlib.suppress(OSError):
                    _fsync_directory(selected.destination.parent)
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--refresh-manifest", action="store_true")
    modes.add_argument("--check-candidate", action="store_true")
    modes.add_argument("--seal", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.status:
            value = draft_status()
        elif args.refresh_manifest:
            value = refresh_manifest()
        elif args.check_candidate:
            value = build_receipt()
        elif args.seal:
            value = seal()
        else:
            value = verify_published_receipt()
    except BundleVerificationError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
