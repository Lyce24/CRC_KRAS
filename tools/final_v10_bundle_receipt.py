#!/usr/bin/env python3
"""Stage, verify, and seal the additive FINAL-v10 report bundle.

FINAL-v10 adopts the FINAL-v9 source ledger by exact manifest identity and adds
the governed five-seed campaign plus corrected Aim-2 adjudication.  A draft
manifest may list not-yet-materialized files in ``pending_artifacts``.  Those
records never count as evidence and make candidate construction and sealing
fail closed.

``--refresh-manifest`` is the only mode that edits the manifest.  It moves a
pending record to the authenticated artifact list only when the declared path
is a regular file, recording its live byte size and SHA-256.  It never writes
the final report receipt.  ``--seal`` validates the complete source graph and
publishes ``report_bundle_receipt.json`` atomically and exactly once.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import final_v9_bundle_receipt as core  # noqa: E402

FINAL_V10 = REPO / "reports" / "final_v10"
BASE_MANIFEST = REPO / "reports" / "final_v9" / core.SOURCE_MANIFEST_NAME
CORE_VERIFIER = REPO / "tools" / "final_v9_bundle_receipt.py"
EXPECTED_CORE_VERIFIER_SHA256 = "3d7a0b5efbd746ee4e5766202a50b215eb3d68a6e122c0a413e484831b362c7b"
EXPECTED_BASE_MANIFEST_SHA256 = "b36ccce52784457d73229cc91f49530c0c861a1652a181ecd26069b9aee594d0"
REPORT_DOCUMENTS = core.REPORT_DOCUMENTS
SOURCE_MANIFEST_NAME = core.SOURCE_MANIFEST_NAME
FINAL_RECEIPT_NAME = core.FINAL_RECEIPT_NAME
SEALED_STATUS = core.SEALED_STATUS
DRAFT_STATUS = "draft_awaiting_five_seed_postprocessing_and_adjudication"
CANDIDATE_STATUS = "candidate_ready_for_final_v10_verification"
MODEL_SEEDS = core.MODEL_SEEDS
ADOPTED_MODEL_SEEDS = core.ADOPTED_MODEL_SEEDS
NEW_MODEL_SEEDS = core.NEW_MODEL_SEEDS
OFFICIAL_CPHT_NAME = core.OFFICIAL_CPHT_NAME

# FINALIZATION HANDOFF: replace every sentinel below with the direct SHA-256 of
# the reconciled, independently reviewed final document or terminal verification
# receipt. Candidate validation intentionally remains impossible until then.
_UNFROZEN_RECEIPT_CREATED_UTC = "REPLACE_IMMEDIATELY_BEFORE_FINAL_V10_SEAL"
EXPECTED_RECEIPT_CREATED_UTC = "2026-08-24T10:27:00+00:00"
EXPECTED_FINAL_DOCUMENT_SHA256 = {
    "Experimental_Setup.md": "bc70ccc275d84840d7f22d838cc4b18df3cc812769a429df5afc182fa82a9c25",
    "Results.md": "8e567f451a643ac650f862deeed2d96ca28911616490f902e37635ab9259d523",
    "Audit.md": "cca885b4658a7d2219cc3fec9332101c8108919fe604c14b7c7b91cce6e4b0c2",
}
EXPECTED_TERMINAL_VERIFICATION_RECEIPT_SHA256 = {
    "final-v9-five-seed-campaign-results-completion": "a827c7e01e1b8f23121bfab41371b0f9a6e95597e98d1002bafe18039ac9c2e0",
    "aim2-e2a-five-seed-adjudication-receipt": "ce1667aae03c146b04c8327c0ed27341f2a140cb4d381f34aed4201eeb579d67",
}

_DRAFT_MANIFEST_KEYS = {
    "schema_version",
    "bundle",
    "status",
    "base_manifest",
    "role_overrides",
    "artifacts",
    "pending_artifacts",
}
_FLAT_MANIFEST_KEYS = {
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
_CANDIDATE_FORBIDDEN_TEXT = (
    "DRAFT_UNSEALED",
    "descriptive peek",
    "not yet materialized",
    "not yet available",
    "await governed postprocessing",
    "remain in progress",
    "corrected adjudication pending",
    "No v10 transport decision yet",
    "pending v10 reconciliation",
    "pending sensitivity",
    "preparation state",
    "v10 receipt is intentionally absent",
    "is in progress",
    "remains blocked",
    "active seal blockers",
    "absent by design",
    "remain pending",
)
_CANDIDATE_FINALIZATION_STATE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bworking\s+(?:copy|document|bundle)\b",
        r"\bdraft\s+manifest\b",
        r"\bdocument\s+reconciliation\b",
        r"\b(?:must\s+)?remain(?:s)?\s+absent(?:\s+until\b)?",
        r"\bbefore\s+sealing\b",
        r"\bcurrent\s+staged\s+manifest\b",
        r"\bfinal\s+materialization\s+must\b",
        r"\bsource[- ]manifest\s+materialization\b",
        r"\bfinal\s+reconciliation\b",
        r"\bseal\s+blockers?\b",
        r"\bawaiting\b",
        r"\b(?:is|are|remain|remains)\s+(?:still\s+)?in\s+progress\b",
    )
)
_CANDIDATE_SCIENTIFIC_CONTRADICTION_PATTERNS = (
    (
        "Orion included in canonical Aim 1",
        re.compile(
            r"(?:\borion\b[^.!?\n]{0,100}\b(?:included|part\s+of)\b[^.!?\n]{0,60}"
            r"\bcanonical\s+aim\s*1\b|\bcanonical\s+aim\s*1\b[^.!?\n]{0,100}"
            r"\b(?:includes?|contains?)\b[^.!?\n]{0,60}\borion\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "Orion LOCO promoted to controlling or confirmatory evidence",
        re.compile(
            r"\borion\s+loco\b[^.!?\n]{0,120}\b(?:controll\w*|confirm\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "E2-CPHT-R reported complete or passed",
        re.compile(
            r"\be2[- ]cpht-r\b[^.!?\n]{0,100}\b(?:complete(?:d)?|pass(?:ed|es)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "whole-section pathology validation reported read or confirmed",
        re.compile(
            r"\bwhole[- ]section\s+pathology\s+validation\b[^.!?\n]{0,120}"
            r"\b(?:read|confirmed|validated|passed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "model seeds treated as independent inference units",
        re.compile(
            r"(?:\bmodel\s+seeds?\b[^.!?\n]{0,120}\bindependent\s+inference\s+units?\b|"
            r"\bindependent\s+inference\s+units?\b[^.!?\n]{0,120}\bmodel\s+seeds?\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "patient or slide folds reported to change across model seeds",
        re.compile(
            r"(?:\b(?:patient|slide|patient/slide|patient\s+and\s+slide)\s+folds?\b"
            r"[^.!?\n]{0,100}\b(?:differ\w*|chang\w*|var(?:y|ied))\b[^.!?\n]{0,80}"
            r"\b(?:across|between)\b[^.!?\n]{0,30}\bmodel\s+seeds?\b|"
            r"\b(?:across|between)\b[^.!?\n]{0,30}\bmodel\s+seeds?\b[^.!?\n]{0,100}"
            r"\b(?:patient|slide|patient/slide|patient\s+and\s+slide)\s+folds?\b"
            r"[^.!?\n]{0,80}\b(?:differ\w*|chang\w*|var(?:y|ied))\b)",
            re.IGNORECASE,
        ),
    ),
)
FINAL_STATE_STATUS_PARAGRAPH = (
    "FINAL-v10 evidence is complete: all 29 governed new sources are materialized, "
    "the flat source manifest contains exactly 81 authenticated artifacts with zero "
    "pending records, and candidate verification precedes exactly-once receipt publication."
)
_EXPECTED_ROLE_OVERRIDES = {
    "aim1-sealed-replay-results": "adopted_three_seed_continuity_and_unaffected_aim1_results",
    "aim1-encoder-and-control-results": ("adopted_three_seed_continuity_and_unchanged_e1e_results"),
    "aim2-e2a-family-results": ("adopted_three_seed_continuity_superseded_by_five_seed_results"),
    "aim2-e2ad-results": ("adopted_three_seed_continuity_superseded_by_five_seed_results"),
    "aim2-e2met-results": "adopted_three_seed_detailed_role_organ_and_continuity_results",
    "aim3-fixed-control-results": ("adopted_three_seed_continuity_superseded_by_five_seed_results"),
    "aim3-repeated-control-results": (
        "adopted_three_seed_continuity_superseded_by_five_seed_results"
    ),
    "aim3-e3v-results": ("adopted_three_seed_continuity_superseded_by_five_seed_results"),
    "aim3-actionability-results": "legacy_three_seed_actionability_continuity_only",
}
_EXPECTED_BASE_SOURCE_IDS = frozenset(
    {
        "final-v8-preregistration",
        "final-v8-available-results-receipt",
        "aim1-sealed-replay-results",
        "aim1-sealed-replay-receipt",
        "aim1-encoder-and-control-results",
        "aim1-worklist-results",
        "aim1-ras-composite-results",
        "aim1-decision-curve-results",
        "aim2-e2a-family-results",
        "aim2-e2ad-results",
        "aim2-e2ad-receipt",
        "aim2-between-slide-results",
        "aim2-between-slide-receipt",
        "aim2-e2met-results",
        "aim2-e2met-report-receipt",
        "aim2-e2met-inference-seal",
        "aim2-e2f-v3-results",
        "aim2-e2f-v3-receipt",
        "aim2-e2e-results",
        "aim2-e2cpht-results",
        "aim2-e2cpht-analysis-receipt",
        "aim2-e2cpht-inference-seal",
        "aim2-e2cpht-a-v2-results",
        "aim2-e2cpht-a-v2-analysis-receipt",
        "aim2-e2cpht-a-v2-run-receipt",
        "aim2-e2cpht-a-v2-execution-contract",
        "aim2-aim3-aim4-claim-source-map",
        "aim3-fixed-control-results",
        "aim3-repeated-control-results",
        "aim3-repeated-control-bootstrap",
        "aim3-repeated-control-audit",
        "aim3-repeated-control-lineage-completion",
        "aim3-e3v-results",
        "aim3-actionability-results",
        "aim4-corrected-k32-results",
        "aim4-specificity-results",
        "aim4-transport-results",
        "aim4-prototype-table",
        "aim4-numeric-completion",
        "aim4-numeric-verification",
        "aim4-vocabulary-stability-results",
        "aim4-vocabulary-stability-completion",
        "aim4-vocabulary-stability-verification",
        "aim4-score-compressibility-results",
        "aim4-machine-montage-read-results",
        "aim4-human-montage-read-results",
        "aim4-human-machine-concordance",
        "aim4-pathway-context-weld-results",
        "aim4-whole-section-packet-receipt",
        "aim4-pre-read-nuisance-receipt",
        "aim1-exploratory-all-primary-orion-results",
        "aim1-exploratory-all-primary-orion-parallel-receipt",
    }
)
_AIM1_AIM3_SOURCE_IDS = frozenset(
    {
        "aim3-ladders-five-seed-contract",
        "aim3-ladders-five-seed-results",
        "aim3-ladders-five-seed-bootstrap",
        "aim3-ladders-five-seed-analysis-audit",
        "aim3-ladders-five-seed-completion",
    }
)
_CAMPAIGN_EXPERIMENTS = ("Study-wide MIL five-seed expansion",)
_AIM1_E0_EXPERIMENTS = ("E0 five-seed extension",)
_AIM2_LOCO_EXPERIMENTS = (
    "E2a-F",
    "E2a-D",
    "E2-MET LOCO sensitivity",
    "Orion LOCO sensitivity",
    "RIH size-matched sensitivity",
)
_AIM2_ADJUDICATION_EXPERIMENTS = (
    "E2a-F corrected adjudication",
    "E2a-D corrected adjudication",
    "E2-MET LOCO sensitivity",
    "RIH size-matched sensitivity",
)
_AIM3_LADDER_EXPERIMENTS = ("E3 fixed", "E3 repeated controls", "E3v", "E1v")
_EXPECTED_NEW_SOURCE_METADATA: dict[str, tuple[tuple[str, ...], str]] = {
    "final-v9-five-seed-campaign-contract": (
        _CAMPAIGN_EXPERIMENTS,
        "governed_five_seed_campaign_contract",
    ),
    "final-v9-five-seed-campaign-deep-preflight": (
        _CAMPAIGN_EXPERIMENTS,
        "governed_deep_preflight_receipt",
    ),
    "final-v9-five-seed-campaign-training-completion": (
        _CAMPAIGN_EXPERIMENTS,
        "governed_training_completion_receipt",
    ),
    "final-v9-five-seed-campaign-results-completion": (
        _CAMPAIGN_EXPERIMENTS,
        "governed_results_completion_receipt",
    ),
    "aim1-e0-five-seed-contract": (
        _AIM1_E0_EXPERIMENTS,
        "governed_five_seed_component_contract",
    ),
    "aim1-e0-five-seed-training-validation": (
        _AIM1_E0_EXPERIMENTS,
        "governed_training_validation_receipt",
    ),
    "aim1-e0-five-seed-results": (
        _AIM1_E0_EXPERIMENTS,
        "controlling_five_seed_results",
    ),
    "aim1-e0-five-seed-patient-logits": (
        _AIM1_E0_EXPERIMENTS,
        "controlling_patient_native_logits",
    ),
    "aim1-e0-five-seed-analysis-receipt": (
        _AIM1_E0_EXPERIMENTS,
        "governed_analysis_receipt",
    ),
    "aim2-loco-five-seed-contract": (
        _AIM2_LOCO_EXPERIMENTS,
        "governed_five_seed_component_contract",
    ),
    "aim2-loco-five-seed-inference-seal": (
        ("Five-seed LOCO inference",),
        "label_blind_inference_seal",
    ),
    "aim2-loco-five-seed-source-oof": (
        ("Five-seed LOCO source OOF",),
        "governed_source_oof_native_logits",
    ),
    "aim2-loco-five-seed-calibrators": (
        ("Five-seed LOCO source calibration",),
        "governed_source_only_calibrators",
    ),
    "aim2-loco-five-seed-primary-patients": (
        ("E2a-F", "E2a-D", "RIH size-matched sensitivity"),
        "controlling_five_seed_primary_patient_scores",
    ),
    "aim2-loco-five-seed-metastatic-patients": (
        ("E2-MET LOCO sensitivity",),
        "controlling_five_seed_metastatic_sensitivity_scores",
    ),
    "aim2-loco-five-seed-orion-patients": (
        ("Orion LOCO sensitivity",),
        "governed_five_seed_orion_loco_sensitivity_scores",
    ),
    "aim2-loco-five-seed-results": (
        _AIM2_LOCO_EXPERIMENTS,
        "controlling_five_seed_results_with_field_scoped_adjudication",
    ),
    "aim2-loco-five-seed-table": (
        ("Five-seed LOCO results",),
        "governed_result_table",
    ),
    "aim2-loco-five-seed-report-receipt": (
        ("Five-seed LOCO results",),
        "governed_analysis_receipt",
    ),
    "aim2-e2a-five-seed-adjudication-implementation": (
        _AIM2_ADJUDICATION_EXPERIMENTS,
        "governed_adjudication_implementation",
    ),
    "aim2-e2a-five-seed-adjudication-test": (
        _AIM2_ADJUDICATION_EXPERIMENTS,
        "focused_adjudication_verification_test",
    ),
    "aim2-e2a-five-seed-adjudication-result": (
        _AIM2_ADJUDICATION_EXPERIMENTS,
        "controlling_field_scoped_adjudication_result",
    ),
    "aim2-e2a-five-seed-adjudication-bootstrap": (
        _AIM2_ADJUDICATION_EXPERIMENTS,
        "governed_adjudication_bootstrap_distributions",
    ),
    "aim2-e2a-five-seed-adjudication-receipt": (
        _AIM2_ADJUDICATION_EXPERIMENTS,
        "governed_adjudication_receipt",
    ),
    "aim3-ladders-five-seed-contract": (
        _AIM3_LADDER_EXPERIMENTS,
        "governed_five_seed_component_contract",
    ),
    "aim3-ladders-five-seed-results": (
        _AIM3_LADDER_EXPERIMENTS,
        "controlling_five_seed_results",
    ),
    "aim3-ladders-five-seed-bootstrap": (
        _AIM3_LADDER_EXPERIMENTS,
        "governed_bootstrap_distributions",
    ),
    "aim3-ladders-five-seed-analysis-audit": (
        _AIM3_LADDER_EXPERIMENTS,
        "governed_analysis_audit",
    ),
    "aim3-ladders-five-seed-completion": (
        _AIM3_LADDER_EXPERIMENTS,
        "governed_extension_completion_receipt",
    ),
}


class BundleVerificationError(core.BundleVerificationError):
    """A fail-closed FINAL-v10 staging, report, source, or receipt error."""


@dataclass(frozen=True)
class BundlePaths:
    """Filesystem locations used by the FINAL-v10 verifier."""

    repo: Path
    final_v10: Path
    destination: Path
    verifier_code: Path
    verifier_test: Path
    base_manifest: Path = BASE_MANIFEST
    core_verifier: Path = CORE_VERIFIER
    campaign_root: Path = core.FIVE_SEED_CAMPAIGN_ROOT
    adjudication_root: Path = core.FIVE_SEED_ADJUDICATION_ROOT
    expected_document_sha256: dict[str, str] | None = None
    expected_terminal_receipt_sha256: dict[str, str] | None = None
    expected_created_utc: str | None = None


def default_paths() -> BundlePaths:
    """Return production FINAL-v10 paths."""

    return BundlePaths(
        repo=REPO,
        final_v10=FINAL_V10,
        destination=FINAL_V10 / FINAL_RECEIPT_NAME,
        verifier_code=Path(__file__).resolve(),
        verifier_test=REPO / "tests" / "test_final_v10_bundle_receipt.py",
        expected_document_sha256=dict(EXPECTED_FINAL_DOCUMENT_SHA256),
        expected_terminal_receipt_sha256=dict(EXPECTED_TERMINAL_VERIFICATION_RECEIPT_SHA256),
        expected_created_utc=EXPECTED_RECEIPT_CREATED_UTC,
    )


def _core_paths(paths: BundlePaths) -> core.BundlePaths:
    """Adapt v10 paths to the audited five-seed semantic verifier."""

    return core.BundlePaths(
        repo=paths.repo,
        final_v9=paths.final_v10,
        destination=paths.destination,
        verifier_code=paths.verifier_code,
        verifier_test=paths.verifier_test,
        campaign_root=paths.campaign_root,
        adjudication_root=paths.adjudication_root,
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    """Return a regular file's path, byte size, and SHA-256 identity."""

    if path.is_symlink() or not path.is_file():
        raise BundleVerificationError(f"expected a regular non-symlink file: {path}")
    return {
        "path": str(path if display_path is None else display_path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_frozen_dependencies(paths: BundlePaths) -> None:
    """Pin the imported semantic core and the unsealed adopted source ledger."""

    core_identity = identity(paths.core_verifier)
    base_identity = identity(paths.base_manifest)
    if core_identity["sha256"] != EXPECTED_CORE_VERIFIER_SHA256:
        raise BundleVerificationError("frozen FINAL-v9 semantic verifier identity drift")
    if base_identity["sha256"] != EXPECTED_BASE_MANIFEST_SHA256:
        raise BundleVerificationError("adopted FINAL-v9 source-manifest identity drift")


def _validate_sha256_pins(pins: Any, expected_keys: set[str], *, context: str) -> dict[str, str]:
    if not isinstance(pins, dict) or set(pins) != expected_keys:
        raise BundleVerificationError(f"{context} SHA-256 pin roster is not frozen")
    if any(
        not isinstance(value, str) or core._SHA256_RE.fullmatch(value) is None  # noqa: SLF001
        for value in pins.values()
    ):
        raise BundleVerificationError(
            f"{context} SHA-256 pins must be replaced after final reconciliation"
        )
    return pins


def _validate_frozen_document_bytes(paths: BundlePaths) -> None:
    """Reject any candidate report byte not explicitly frozen after final review."""

    pins = _validate_sha256_pins(
        paths.expected_document_sha256,
        set(REPORT_DOCUMENTS),
        context="FINAL-v10 document",
    )
    if paths.repo.resolve() == REPO.resolve() and pins != EXPECTED_FINAL_DOCUMENT_SHA256:
        raise BundleVerificationError(
            "production FINAL-v10 documents must use the frozen verifier constants"
        )
    for filename in REPORT_DOCUMENTS:
        path = paths.final_v10 / filename
        if sha256_file(path) != pins[filename]:
            raise BundleVerificationError(f"frozen FINAL-v10 document identity drift: {filename}")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BundleVerificationError(f"{label} must be a regular non-symlink file")
    try:
        return core._load_json(path, label=label)  # noqa: SLF001
    except core.BundleVerificationError as exc:
        raise BundleVerificationError(str(exc)) from exc


def _validate_recorded_identity(record: Any, paths: BundlePaths, *, context: str) -> Path:
    """Rehash one exact path/size/SHA identity and return its resolved path."""

    if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
        raise BundleVerificationError(f"{context} must be an exact artifact identity")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise BundleVerificationError(f"{context} path is invalid")
    lexical_path = _lexical_source_path(raw_path, paths)
    if lexical_path.is_symlink():
        raise BundleVerificationError(f"{context} declared path must not be a symlink")
    resolved = _resolve_source_path(raw_path, paths)
    if identity(resolved, display_path=raw_path) != record:
        raise BundleVerificationError(f"{context} identity drift")
    return resolved


def _relative_display(path: Path, repo: Path) -> str:
    return core._relative_display(path, repo)  # noqa: SLF001


def _lexical_source_path(raw_path: str, paths: BundlePaths) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else paths.repo / candidate


def _resolve_source_path(raw_path: str, paths: BundlePaths) -> Path:
    lexical_path = _lexical_source_path(raw_path, paths)
    if lexical_path.is_symlink():
        raise BundleVerificationError(
            f"declared source path must be a regular non-symlink file: {raw_path}"
        )
    try:
        return core._resolve_source_path(raw_path, _core_paths(paths))  # noqa: SLF001
    except core.BundleVerificationError as exc:
        raise BundleVerificationError(str(exc)) from exc


def _required_sources(paths: BundlePaths) -> dict[str, tuple[Path, str]]:
    return core._required_five_seed_sources(_core_paths(paths))  # noqa: SLF001


def _expected_source_aims(source_id: str, core_aim: str) -> list[str]:
    """Return v10 scientific scope, including E1v's Aim-1/Aim-3 dual role."""

    if source_id in _AIM1_AIM3_SOURCE_IDS:
        return ["Aim 1", "Aim 3"]
    return [core_aim]


def _core_compatible_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt dual-scoped E1v lineage metadata to the frozen v9 core API."""

    return [
        {**source, "aims": ["Aim 3"]} if str(source.get("id")) in _AIM1_AIM3_SOURCE_IDS else source
        for source in sources
    ]


def _validate_string_list(value: Any, *, location: str) -> list[str]:
    try:
        return core._require_string_list(value, location=location)  # noqa: SLF001
    except core.BundleVerificationError as exc:
        raise BundleVerificationError(str(exc)) from exc


def _validate_metadata(record: Any, *, location: str, pending: bool) -> dict[str, Any]:
    expected = _PENDING_KEYS if pending else _ARTIFACT_KEYS
    if not isinstance(record, dict) or set(record) != expected:
        raise BundleVerificationError(
            f"{location} keys must be exactly {', '.join(sorted(expected))}"
        )
    source_id = record["id"]
    if not isinstance(source_id, str) or core._SOURCE_ID_RE.fullmatch(source_id) is None:  # noqa: SLF001
        raise BundleVerificationError(f"{location}.id is not a lowercase source ID")
    aims = _validate_string_list(record["aims"], location=f"{location}.aims")
    invalid_aims = set(aims) - set(core.REQUIRED_AIMS) - {"Shared"}
    if invalid_aims:
        raise BundleVerificationError(
            f"{location}.aims contains invalid values: {sorted(invalid_aims)}"
        )
    experiments = _validate_string_list(record["experiments"], location=f"{location}.experiments")
    role = record["role"]
    raw_path = record["path"]
    if not isinstance(role, str) or not role.strip():
        raise BundleVerificationError(f"{location}.role must be a non-empty string")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise BundleVerificationError(f"{location}.path must be a non-empty string")
    normalized = {
        "id": source_id,
        "aims": aims,
        "experiments": experiments,
        "role": role,
        "path": raw_path,
    }
    if not pending:
        size_bytes = record["size_bytes"]
        sha256 = record["sha256"]
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise BundleVerificationError(f"{location}.size_bytes must be non-negative")
        if not isinstance(sha256, str) or core._SHA256_RE.fullmatch(sha256) is None:  # noqa: SLF001
            raise BundleVerificationError(f"{location}.sha256 must be lowercase SHA-256")
        normalized.update(size_bytes=size_bytes, sha256=sha256)
    return normalized


def _validate_base_manifest_record(record: Any, paths: BundlePaths) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
        raise BundleVerificationError("base_manifest must be an exact artifact identity")
    expected_path = paths.base_manifest.resolve()
    actual_path = _resolve_source_path(str(record["path"]), paths)
    if actual_path != expected_path:
        raise BundleVerificationError(
            "base_manifest must point to reports/final_v9/source_manifest.json"
        )
    actual = identity(expected_path, display_path=str(record["path"]))
    if record != actual:
        raise BundleVerificationError("adopted FINAL-v9 source-manifest identity drift")
    return actual


def _validate_base_sources(paths: BundlePaths, role_overrides: Any) -> list[dict[str, Any]]:
    if role_overrides != _EXPECTED_ROLE_OVERRIDES:
        raise BundleVerificationError("historical five-seed precedence role overrides changed")

    manifest = _load_json(paths.base_manifest, label="adopted FINAL-v9 source manifest")
    if set(manifest) != {"schema_version", "bundle", "artifacts"}:
        raise BundleVerificationError("adopted FINAL-v9 source manifest schema changed")
    if manifest["schema_version"] != 1 or manifest["bundle"] != "final_v9":
        raise BundleVerificationError("adopted source manifest is not FINAL-v9 schema 1")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise BundleVerificationError("adopted FINAL-v9 source manifest has no artifacts")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(artifacts):
        source = _validate_metadata(raw, location=f"base artifact {index}", pending=False)
        source_id = str(source["id"])
        if source_id in seen:
            raise BundleVerificationError(f"duplicate adopted source ID: {source_id}")
        seen.add(source_id)
        resolved = _resolve_source_path(str(source["path"]), paths)
        actual = identity(resolved, display_path=str(source["path"]))
        if actual["size_bytes"] != source["size_bytes"] or actual["sha256"] != source["sha256"]:
            raise BundleVerificationError(f"adopted source identity drift for {source_id}")
        if source_id in role_overrides:
            source["role"] = role_overrides[source_id]
        normalized.append(source)

    if seen != _EXPECTED_BASE_SOURCE_IDS or len(normalized) != 52:
        raise BundleVerificationError("adopted source inventory is not the exact 52-source roster")

    for source_id in core.SUPERSEDED_BY_FIVE_SEED_SOURCE_IDS:
        role = str(role_overrides[source_id]).casefold()
        if not any(marker in role for marker in core._NONCONTROLLING_ROLE_MARKERS):  # noqa: SLF001
            raise BundleVerificationError(
                f"role override for {source_id} does not declare adopted/continuity precedence"
            )
    return normalized


def _validate_terminal_verification_receipts(
    paths: BundlePaths, sources: list[dict[str, Any]]
) -> None:
    """Authenticate terminal receipts produced only after full component verification."""

    expected_ids = set(EXPECTED_TERMINAL_VERIFICATION_RECEIPT_SHA256)
    pins = _validate_sha256_pins(
        paths.expected_terminal_receipt_sha256,
        expected_ids,
        context="terminal full-component verification receipt",
    )
    if (
        paths.repo.resolve() == REPO.resolve()
        and pins != EXPECTED_TERMINAL_VERIFICATION_RECEIPT_SHA256
    ):
        raise BundleVerificationError(
            "production terminal verification receipts must use the frozen verifier constants"
        )
    by_id = core._source_map(sources)  # noqa: SLF001
    for source_id in sorted(expected_ids):
        source = by_id.get(source_id)
        if not isinstance(source, dict) or source.get("sha256") != pins[source_id]:
            raise BundleVerificationError(
                f"pinned full-component verification receipt changed: {source_id}"
            )
        receipt_path = _validate_recorded_identity(
            {key: source[key] for key in ("path", "size_bytes", "sha256")},
            paths,
            context=f"terminal full-component verification receipt {source_id}",
        )
        if sha256_file(receipt_path) != pins[source_id]:
            raise BundleVerificationError(
                f"terminal full-component verification receipt byte drift: {source_id}"
            )


def _validate_complete_source_graph(
    paths: BundlePaths, sources: list[dict[str, Any]], *, validate_claims: bool
) -> None:
    """Replay every v10 semantic gate before accepting a flat source graph."""

    try:
        core._validate_five_seed_source_inventory(  # noqa: SLF001
            _core_paths(paths), _core_compatible_sources(sources)
        )
    except core.BundleVerificationError as exc:
        raise BundleVerificationError(str(exc)) from exc
    _validate_terminal_verification_receipts(paths, sources)
    _validate_aim1_result_semantics(paths, sources)
    _validate_aim2_split_membership(paths, sources)
    _validate_aim3_result_semantics(paths, sources)
    _validate_aim3_split_membership(paths, sources)
    _validate_aim2_adjudication_additions(paths, sources)
    if validate_claims:
        _validate_report_claims(paths, sources)
        _validate_frozen_document_bytes(paths)


def _validate_manifest(
    paths: BundlePaths, *, require_candidate: bool
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_frozen_dependencies(paths)
    manifest_path = paths.final_v10 / SOURCE_MANIFEST_NAME
    manifest = _load_json(manifest_path, label="FINAL-v10 source manifest")
    if manifest.get("schema_version") != 2 or manifest.get("bundle") != "final_v10":
        raise BundleVerificationError(
            "source manifest must declare schema_version=2 and bundle=final_v10"
        )

    layered = set(manifest) == _DRAFT_MANIFEST_KEYS
    flat = set(manifest) == _FLAT_MANIFEST_KEYS
    if not layered and not flat:
        raise BundleVerificationError(
            "FINAL-v10 source manifest must use the exact layered-draft or flat-candidate schema"
        )
    expected_base = _validate_base_sources(paths, _EXPECTED_ROLE_OVERRIDES)
    if layered:
        if manifest["status"] != DRAFT_STATUS:
            raise BundleVerificationError("the layered source manifest is draft-only")
        _validate_base_manifest_record(manifest["base_manifest"], paths)
        if manifest["role_overrides"] != _EXPECTED_ROLE_OVERRIDES:
            raise BundleVerificationError("draft precedence role overrides changed")
        base_sources = expected_base
    else:
        if manifest["status"] != CANDIDATE_STATUS:
            raise BundleVerificationError("the flat source manifest must be candidate-ready")
        base_sources = []

    raw_artifacts = manifest["artifacts"]
    raw_pending = manifest["pending_artifacts"]
    if not isinstance(raw_artifacts, list) or not isinstance(raw_pending, list):
        raise BundleVerificationError("artifacts and pending_artifacts must be arrays")
    artifacts = [
        _validate_metadata(item, location=f"artifact {index}", pending=False)
        for index, item in enumerate(raw_artifacts)
    ]
    pending = [
        _validate_metadata(item, location=f"pending artifact {index}", pending=True)
        for index, item in enumerate(raw_pending)
    ]

    expected_new = set(_required_sources(paths))
    if set(_EXPECTED_NEW_SOURCE_METADATA) != expected_new:
        raise BundleVerificationError("frozen new-source metadata roster is incomplete")
    expected = expected_new if layered else (_EXPECTED_BASE_SOURCE_IDS | expected_new)
    ids = [str(item["id"]) for item in [*artifacts, *pending]]
    if len(ids) != len(set(ids)):
        raise BundleVerificationError("duplicate source ID across artifacts and pending_artifacts")
    if set(ids) != expected:
        missing = sorted(expected - set(ids))
        unexpected = sorted(set(ids) - expected)
        raise BundleVerificationError(
            f"source roster mismatch; missing={missing}, unexpected={unexpected}"
        )

    base_ids = {str(item["id"]) for item in base_sources}
    if base_ids & set(ids):
        raise BundleVerificationError("new source ID collides with adopted FINAL-v9 source ID")

    seen_paths: set[Path] = {
        _resolve_source_path(str(item["path"]), paths) for item in base_sources
    }
    requirements = _required_sources(paths)
    for source in [*artifacts, *pending]:
        source_id = str(source["id"])
        resolved = _resolve_source_path(str(source["path"]), paths)
        if source_id in requirements:
            expected_path, expected_aim = requirements[source_id]
            if resolved != expected_path.resolve(strict=False):
                raise BundleVerificationError(f"required source path mismatch for {source_id}")
            expected_aims = _expected_source_aims(source_id, expected_aim)
            if source["aims"] != expected_aims:
                raise BundleVerificationError(
                    f"required source {source_id} must declare aims={expected_aims!r}"
                )
            expected_experiments, expected_role = _EXPECTED_NEW_SOURCE_METADATA[source_id]
            if (
                source["experiments"] != list(expected_experiments)
                or source["role"] != expected_role
            ):
                raise BundleVerificationError(
                    f"required source {source_id} must declare exact experiments/role metadata"
                )
        try:
            resolved.relative_to(paths.final_v10.resolve())
        except ValueError:
            pass
        else:
            raise BundleVerificationError(f"authoritative source is inside FINAL-v10: {source_id}")
        if resolved in seen_paths:
            raise BundleVerificationError(f"source file is declared more than once: {resolved}")
        seen_paths.add(resolved)

    for source in artifacts:
        resolved = _resolve_source_path(str(source["path"]), paths)
        actual = identity(resolved, display_path=str(source["path"]))
        if actual["size_bytes"] != source["size_bytes"] or actual["sha256"] != source["sha256"]:
            raise BundleVerificationError(f"authoritative source identity drift for {source['id']}")

    if flat:
        expected_base_by_id = {str(source["id"]): source for source in expected_base}
        observed_by_id = {str(source["id"]): source for source in artifacts}
        for source_id in _EXPECTED_BASE_SOURCE_IDS:
            if observed_by_id[source_id] != expected_base_by_id[source_id]:
                raise BundleVerificationError(
                    f"flat adopted source record differs from authenticated ledger: {source_id}"
                )
        if len(artifacts) != 81 or pending:
            raise BundleVerificationError(
                "candidate manifest must contain exactly 81 flat artifacts and no pending records"
            )

    if pending:
        if manifest["status"] != DRAFT_STATUS:
            raise BundleVerificationError(
                "a manifest with pending artifacts must have draft status"
            )
        if require_candidate:
            raise BundleVerificationError(
                f"FINAL-v10 candidate is blocked by {len(pending)} pending artifacts"
            )
    elif manifest["status"] != CANDIDATE_STATUS:
        raise BundleVerificationError("a complete source roster must have candidate-ready status")

    combined = [*base_sources, *artifacts]
    combined.sort(key=lambda item: str(item["id"]))
    if require_candidate or flat:
        _validate_complete_source_graph(paths, combined, validate_claims=True)
    elif not flat:
        _validate_training_stage(paths, combined)
        combined_ids = {str(source["id"]) for source in combined}
        if {
            "aim1-e0-five-seed-results",
            "aim1-e0-five-seed-training-validation",
        }.issubset(combined_ids):
            _validate_aim1_result_semantics(paths, combined)
        if {
            "aim2-loco-five-seed-contract",
            "final-v9-five-seed-campaign-training-completion",
        }.issubset(combined_ids):
            _validate_aim2_split_membership(paths, combined)
        if "aim3-ladders-five-seed-results" in combined_ids:
            _validate_aim3_result_semantics(paths, combined)
        if {
            "aim3-ladders-five-seed-contract",
            "aim3-ladders-five-seed-analysis-audit",
        }.issubset(combined_ids):
            _validate_aim3_split_membership(paths, combined)
    return manifest, combined, pending


def _validate_training_stage(paths: BundlePaths, sources: list[dict[str, Any]]) -> None:
    """Replay the available contract/preflight/training graph in draft state."""

    by_id = core._source_map(sources)  # noqa: SLF001
    required = {
        "final-v9-five-seed-campaign-contract",
        "final-v9-five-seed-campaign-deep-preflight",
        "final-v9-five-seed-campaign-training-completion",
        "aim1-e0-five-seed-contract",
        "aim2-loco-five-seed-contract",
        "aim3-ladders-five-seed-contract",
    }
    if not required.issubset(by_id):
        raise BundleVerificationError("draft lacks the complete training-stage evidence graph")
    adapted = _core_paths(paths)
    try:
        contract = core._source_json(  # noqa: SLF001
            by_id, "final-v9-five-seed-campaign-contract", adapted
        )
        if (
            contract.get("status") != "sealed_before_new_fit"
            or contract.get("experiment") != "final-v9 study-wide MIL five-seed expansion"
            or contract.get("model_seeds")
            != {
                "adopted": ADOPTED_MODEL_SEEDS,
                "new": NEW_MODEL_SEEDS,
                "complete": MODEL_SEEDS,
            }
        ):
            raise BundleVerificationError(
                "five-seed campaign contract status or seed scope changed"
            )
        split = contract.get("split_policy", {})
        execution = contract.get("execution", {})
        counts = contract.get("counts", {})
        if (
            split.get("outer_and_inner_membership_changes_across_model_seeds") is not False
            or execution.get("max_concurrent_gpu_trainers") != 6
            or execution.get("fresh_process_per_chain") is not True
            or counts.get("jobs") != 102
            or counts.get("new_mil_fits") != 490
        ):
            raise BundleVerificationError("five-seed split, concurrency, or fit census changed")
        jobs = core._validate_campaign_job_inventory(  # noqa: SLF001
            contract.get("jobs"), counts.get("by_component")
        )
        core._validate_campaign_component_job_rosters(adapted, by_id, jobs)  # noqa: SLF001

        preflight = core._source_json(  # noqa: SLF001
            by_id, "final-v9-five-seed-campaign-deep-preflight", adapted
        )
        if (
            preflight.get("status") != "deep_preflight_passed"
            or preflight.get("maximum_concurrent_gpu_trainers") != 6
        ):
            raise BundleVerificationError("five-seed deep preflight did not pass")
        core._assert_source_id(  # noqa: SLF001
            preflight.get("contract"),
            by_id,
            "final-v9-five-seed-campaign-contract",
            adapted,
            context="campaign preflight contract",
        )

        training = core._source_json(  # noqa: SLF001
            by_id, "final-v9-five-seed-campaign-training-completion", adapted
        )
        if (
            training.get("status") != "490 new MIL fits completed and certified"
            or training.get("new_mil_fits") != 490
            or training.get("new_training_jobs") != 102
            or training.get("maximum_concurrent_gpu_trainers") != 6
            or training.get("observed_peak_concurrent_gpu_trainers") != 6
            or training.get("reconstructed_peak_from_job_exit_intervals") != 6
        ):
            raise BundleVerificationError("training completion census or observed peak changed")
        core._assert_source_id(  # noqa: SLF001
            training.get("contract"),
            by_id,
            "final-v9-five-seed-campaign-contract",
            adapted,
            context="campaign training contract",
        )
        core._assert_source_id(  # noqa: SLF001
            training.get("preflight"),
            by_id,
            "final-v9-five-seed-campaign-deep-preflight",
            adapted,
            context="campaign training preflight",
        )
        core._validate_campaign_job_receipts(  # noqa: SLF001
            adapted, jobs, training.get("job_receipts")
        )
    except core.BundleVerificationError as exc:
        raise BundleVerificationError(str(exc)) from exc


def _finite_number(
    value: Any, *, context: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleVerificationError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise BundleVerificationError(f"{context} must be finite")
    if minimum is not None and number < minimum:
        raise BundleVerificationError(f"{context} is below {minimum}")
    if maximum is not None and number > maximum:
        raise BundleVerificationError(f"{context} is above {maximum}")
    return number


def _ordered_interval(
    value: Any,
    *,
    point: float,
    context: str,
    minimum: float,
    maximum: float,
) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise BundleVerificationError(f"{context} must be a two-value interval")
    lower = _finite_number(value[0], context=f"{context}[0]", minimum=minimum, maximum=maximum)
    upper = _finite_number(value[1], context=f"{context}[1]", minimum=minimum, maximum=maximum)
    if not lower <= point <= upper:
        raise BundleVerificationError(f"{context} must be ordered around its point estimate")
    return lower, upper


def _validate_seed_map(value: Any, *, context: str) -> None:
    if not isinstance(value, dict) or set(value) != {"42", "43", "44", "45", "46"}:
        raise BundleVerificationError(f"{context} must contain exact model seeds 42-46")
    for seed, point in value.items():
        _finite_number(point, context=f"{context}/{seed}", minimum=0.0, maximum=1.0)


def _validate_ladder_metric(value: Any, *, context: str, delta: bool) -> tuple[float, float, float]:
    if not isinstance(value, dict):
        raise BundleVerificationError(f"{context} must be an object")
    minimum, maximum = (-1.0, 1.0) if delta else (0.0, 1.0)
    point = _finite_number(
        value.get("estimate"), context=f"{context}/estimate", minimum=minimum, maximum=maximum
    )
    _ordered_interval(
        value.get("ci95_two_sided"),
        point=point,
        context=f"{context}/ci95_two_sided",
        minimum=minimum,
        maximum=maximum,
    )
    one_sided = value.get("primary_fwer_one_sided")
    if not isinstance(one_sided, dict) or one_sided.get("confidence") != 0.99:
        raise BundleVerificationError(f"{context} must declare the one-sided 99% bounds")
    lower, upper = _ordered_interval(
        [one_sided.get("lower"), one_sided.get("upper")],
        point=point,
        context=f"{context}/primary_fwer_one_sided",
        minimum=minimum,
        maximum=maximum,
    )
    return point, lower, upper


_LADDER_CONTROL_TASKS = {
    "codon": "ctrl_codon",
    "g12d_broad": "ctrl_g12d_broad",
    "allele1": "ctrl_allele1",
    "allele2": "ctrl_allele2",
    "g12c": "ctrl_g12c",
}


def _validate_ladder_rung(
    value: Any, *, task: str, context: str, require_control_task: bool = True
) -> bool:
    if not isinstance(value, dict):
        raise BundleVerificationError(f"{context} must be an object")
    if require_control_task and value.get("control_task") != _LADDER_CONTROL_TASKS[task]:
        raise BundleVerificationError(f"{context} has the wrong matched-control task")
    if not require_control_task and "control_task" in value:
        raise BundleVerificationError(f"{context} duplicates its parent control task")
    fine, _fine_lower, fine_upper = _validate_ladder_metric(
        value.get("fine"), context=f"{context}/fine", delta=False
    )
    control, control_lower, _control_upper = _validate_ladder_metric(
        value.get("control"), context=f"{context}/control", delta=False
    )
    delta, delta_lower, _delta_upper = _validate_ladder_metric(
        value.get("delta_control_minus_fine"),
        context=f"{context}/delta_control_minus_fine",
        delta=True,
    )
    if not math.isclose(delta, control - fine, rel_tol=0.0, abs_tol=1e-12):
        raise BundleVerificationError(f"{context} delta is not control minus fine")
    if "fine_per_seed" in value:
        _validate_seed_map(value.get("fine_per_seed"), context=f"{context}/fine_per_seed")
    if "control_per_seed" in value:
        _validate_seed_map(value.get("control_per_seed"), context=f"{context}/control_per_seed")

    expected_conditions = {
        "fine_upper_lt_0p60": fine_upper < 0.60,
        "control_lower_gt_0p50": control_lower > 0.50,
        "delta_lower_gt_zero": delta_lower > 0.0,
    }
    gate = value.get("gate")
    if not isinstance(gate, dict) or gate.get("conditions") != expected_conditions:
        raise BundleVerificationError(f"{context} gate conditions do not replay from bounds")
    ceiling = all(expected_conditions.values())
    if gate.get("ceiling") is not ceiling:
        raise BundleVerificationError(f"{context} ceiling boolean does not replay")
    expected_verdict = (
        "CEILING"
        if ceiling
        else (
            "UNDERPOWERED" if not expected_conditions["control_lower_gt_0p50"] else "INCONCLUSIVE"
        )
    )
    if gate.get("verdict") != expected_verdict:
        raise BundleVerificationError(f"{context} verdict does not replay from gate conditions")
    return ceiling


def _validate_aim3_result_semantics(paths: BundlePaths, sources: list[dict[str, Any]]) -> None:
    by_id = core._source_map(sources)  # noqa: SLF001
    result = core._source_json(  # noqa: SLF001
        by_id, "aim3-ladders-five-seed-results", _core_paths(paths)
    )
    fixed = result.get("fixed_univ1", {}).get("rungs")
    if not isinstance(fixed, dict) or set(fixed) != set(_LADDER_CONTROL_TASKS):
        raise BundleVerificationError("Aim-3 fixed rung roster changed")
    for task, rung in fixed.items():
        _validate_ladder_rung(rung, task=task, context=f"Aim-3/fixed/{task}")

    e3v = result.get("e3v_virchow2_cls", {}).get("rungs")
    if not isinstance(e3v, dict) or set(e3v) != {"codon", "g12d_broad", "allele1"}:
        raise BundleVerificationError("Aim-3 E3v rung roster changed")
    for task, rung in e3v.items():
        _validate_ladder_rung(rung, task=task, context=f"Aim-3/E3v/{task}")

    repeated = result.get("repeated_univ1", {}).get("rungs")
    if not isinstance(repeated, dict) or set(repeated) != set(_LADDER_CONTROL_TASKS):
        raise BundleVerificationError("Aim-3 repeated-control rung roster changed")
    for task, rung in repeated.items():
        if not isinstance(rung, dict) or rung.get("control_task") != _LADDER_CONTROL_TASKS[task]:
            raise BundleVerificationError(f"Aim-3/repeated/{task} control task changed")
        _validate_seed_map(
            rung.get("fine_per_seed"), context=f"Aim-3/repeated/{task}/fine_per_seed"
        )
        draws = rung.get("draws")
        if not isinstance(draws, dict) or set(draws) != {"20260823", "20260824", "20260825"}:
            raise BundleVerificationError(f"Aim-3/repeated/{task} draw roster changed")
        ceilings = [
            _validate_ladder_rung(
                draw,
                task=task,
                context=f"Aim-3/repeated/{task}/{draw_seed}",
                require_control_task=False,
            )
            for draw_seed, draw in draws.items()
        ]
        expected_consensus = "CONSENSUS_CEILING" if all(ceilings) else "NO_CEILING_CONSENSUS"
        if rung.get("consensus_verdict") != expected_consensus:
            raise BundleVerificationError(
                f"Aim-3/repeated/{task} consensus does not replay from all three draws"
            )

    e1v = result.get("e1v_virchow2_cls_gene_reference", {})
    ensemble = e1v.get("five_seed_ensemble_A", {})
    point = _finite_number(
        ensemble.get("auroc"), context="Aim-1/E1v/A/auroc", minimum=0.0, maximum=1.0
    )
    _ordered_interval(
        [ensemble.get("ci_low"), ensemble.get("ci_high")],
        point=point,
        context="Aim-1/E1v/A/ci95",
        minimum=0.0,
        maximum=1.0,
    )
    if ensemble.get("n") != 1486 or ensemble.get("n_positive") != 604:
        raise BundleVerificationError("Aim-1/E1v patient census changed")
    _validate_seed_map(e1v.get("per_seed_auroc_A"), context="Aim-1/E1v/per_seed_auroc_A")
    _finite_number(
        e1v.get("five_seed_ensemble_D_auroc"),
        context="Aim-1/E1v/D/auroc",
        minimum=0.0,
        maximum=1.0,
    )


def _validate_aim1_result_semantics(paths: BundlePaths, sources: list[dict[str, Any]]) -> None:
    by_id = core._source_map(sources)  # noqa: SLF001
    adapted = _core_paths(paths)
    contract = core._source_json(  # noqa: SLF001
        by_id, "aim1-e0-five-seed-contract", adapted
    )
    contract_inputs = contract.get("inputs")
    if not isinstance(contract_inputs, dict):
        raise BundleVerificationError("Aim-1 E0 contract input evidence is missing")
    for input_name in ("manifest", "splits", "split_integrity", "split_summary"):
        _validate_recorded_identity(
            contract_inputs.get(input_name),
            paths,
            context=f"Aim-1 E0 contract {input_name}",
        )
    _validate_split_integrity_content(
        contract_inputs["manifest"],
        contract_inputs["splits"],
        contract_inputs["split_integrity"],
        paths,
        context="Aim-1 E0 frozen membership",
    )
    split_summary_path = _resolve_source_path(str(contract_inputs["split_summary"]["path"]), paths)
    split_summary = _load_json(split_summary_path, label="Aim-1 E0 split summary")
    if (
        split_summary.get("n_slides") != 1642
        or split_summary.get("n_groups") != 1486
        or split_summary.get("n_folds") != 5
        or split_summary.get("seed") != 42
        or split_summary.get("identity", {}).get("payload", {}).get("manifest_sha256")
        != contract_inputs["manifest"]["sha256"]
    ):
        raise BundleVerificationError("Aim-1 E0 frozen split summary changed")
    result = core._source_json(by_id, "aim1-e0-five-seed-results", adapted)  # noqa: SLF001
    population = result.get("population", {})
    expected_domains = {"CPTAC", "RIH", "SR1482", "SR386", "TCGA"}
    domain_counts = population.get("domain_counts")
    if (
        population.get("patients") != 1486
        or population.get("mutant") != 604
        or population.get("wild_type") != 882
        or not isinstance(domain_counts, dict)
        or set(domain_counts) != expected_domains
        or sum(domain_counts.values()) != 1486
    ):
        raise BundleVerificationError("Aim-1 E0 patient/domain census changed")

    per_seed = result.get("per_seed_macro5")
    _validate_seed_map(per_seed, context="Aim-1/E0/per_seed_macro5")
    per_seed_domains = result.get("per_seed_domain_auroc")
    if not isinstance(per_seed_domains, dict) or set(per_seed_domains) != set(per_seed):
        raise BundleVerificationError("Aim-1 E0 per-seed domain roster changed")
    for seed, domains in per_seed_domains.items():
        if not isinstance(domains, dict) or set(domains) != expected_domains:
            raise BundleVerificationError(f"Aim-1 E0 domain roster changed for seed {seed}")
        values = [
            _finite_number(
                domains[domain],
                context=f"Aim-1/E0/per_seed_domain_auroc/{seed}/{domain}",
                minimum=0.0,
                maximum=1.0,
            )
            for domain in sorted(expected_domains)
        ]
        if not math.isclose(
            float(per_seed[seed]), sum(values) / len(values), rel_tol=0.0, abs_tol=1e-12
        ):
            raise BundleVerificationError(f"Aim-1 E0 macro does not replay for seed {seed}")

    primary = result.get("primary_median_seed_macro5", {})
    primary_point = _finite_number(
        primary.get("point"), context="Aim-1/E0/primary", minimum=0.0, maximum=1.0
    )
    _ordered_interval(
        primary.get("ci95"),
        point=primary_point,
        context="Aim-1/E0/primary_ci95",
        minimum=0.0,
        maximum=1.0,
    )
    sorted_seed_points = sorted(float(value) for value in per_seed.values())
    if not math.isclose(primary_point, sorted_seed_points[2], rel_tol=0.0, abs_tol=1e-12):
        raise BundleVerificationError("Aim-1 E0 primary is not the median seed macro")

    primitives = result.get("primitive_median_seed_auroc")
    if not isinstance(primitives, dict) or set(primitives) != expected_domains:
        raise BundleVerificationError("Aim-1 E0 primitive roster changed")
    for domain, block in primitives.items():
        point = _finite_number(
            block.get("point"),
            context=f"Aim-1/E0/primitive/{domain}",
            minimum=0.0,
            maximum=1.0,
        )
        _ordered_interval(
            block.get("ci95"),
            point=point,
            context=f"Aim-1/E0/primitive/{domain}/ci95",
            minimum=0.0,
            maximum=1.0,
        )
        expected = sorted(float(per_seed_domains[seed][domain]) for seed in per_seed_domains)[2]
        if not math.isclose(point, expected, rel_tol=0.0, abs_tol=1e-12):
            raise BundleVerificationError(
                f"Aim-1 E0 primitive is not the median seed value: {domain}"
            )

    ensemble = result.get("five_seed_ensemble", {})
    ensemble_domains = ensemble.get("domain_auroc")
    if not isinstance(ensemble_domains, dict) or set(ensemble_domains) != expected_domains:
        raise BundleVerificationError("Aim-1 E0 ensemble domain roster changed")
    ensemble_values = [
        _finite_number(
            ensemble_domains[domain],
            context=f"Aim-1/E0/ensemble/{domain}",
            minimum=0.0,
            maximum=1.0,
        )
        for domain in sorted(expected_domains)
    ]
    ensemble_macro = _finite_number(
        ensemble.get("macro5_auroc"),
        context="Aim-1/E0/ensemble_macro",
        minimum=0.0,
        maximum=1.0,
    )
    if not math.isclose(
        ensemble_macro, sum(ensemble_values) / len(ensemble_values), rel_tol=0.0, abs_tol=1e-12
    ):
        raise BundleVerificationError("Aim-1 E0 ensemble macro is not the domain mean")
    _ordered_interval(
        ensemble.get("macro5_ci95"),
        point=ensemble_macro,
        context="Aim-1/E0/ensemble_macro_ci95",
        minimum=0.0,
        maximum=1.0,
    )

    training = core._source_json(  # noqa: SLF001
        by_id, "aim1-e0-five-seed-training-validation", adapted
    )
    if (
        training.get("model_seeds") != [42, 43, 44, 45, 46]
        or training.get("fold_layout_shared_across_all_seeds") is not True
        or training.get("fit_census")
        != {
            "final_fits": 30,
            "final_folds": 25,
            "final_p75_refits": 5,
            "inherited_folds": 15,
            "inherited_p75_refits": 3,
            "new_fits": 12,
            "new_folds": 10,
            "new_p75_refits": 2,
        }
    ):
        raise BundleVerificationError("Aim-1 E0 training/fold census changed")
    runs = training.get("runs")
    if not isinstance(runs, dict) or set(runs) != {"42", "43", "44", "45", "46"}:
        raise BundleVerificationError("Aim-1 E0 validated-run roster changed")
    fold_counts: list[tuple[int, ...]] = []
    split_evidence: list[tuple[str, str]] = []
    for seed, run in runs.items():
        identity_record = run.get("artifacts", {}).get("identity")
        if not isinstance(identity_record, dict):
            raise BundleVerificationError(f"Aim-1 E0 training identity missing for seed {seed}")
        identity_path = Path(str(identity_record.get("path", "")))
        if identity(identity_path, display_path=str(identity_path)) != identity_record:
            raise BundleVerificationError(f"Aim-1 E0 training identity drift for seed {seed}")
        training_identity = _load_json(
            identity_path, label=f"Aim-1 E0 training identity seed {seed}"
        )
        evidence = training_identity.get("payload", {}).get("input_evidence", {})
        manifest_sha = evidence.get("manifest_sha256")
        split_sha = evidence.get("split_integrity_sha256")
        if not isinstance(manifest_sha, str) or not isinstance(split_sha, str):
            raise BundleVerificationError(f"Aim-1 E0 split evidence missing for seed {seed}")
        split_evidence.append((manifest_sha, split_sha))
        folds = run.get("artifacts", {}).get("folds", {})
        if not isinstance(folds, dict) or set(folds) != {"0", "1", "2", "3", "4"}:
            raise BundleVerificationError(f"Aim-1 E0 fold roster changed for seed {seed}")
        counts = tuple(int(folds[str(index)].get("n_test_slides")) for index in range(5))
        if sum(counts) != 1642:
            raise BundleVerificationError(f"Aim-1 E0 slide census changed for seed {seed}")
        fold_counts.append(counts)
    if len(set(fold_counts)) != 1:
        raise BundleVerificationError("Aim-1 E0 fold slide counts differ across model seeds")
    if len(set(split_evidence)) != 1:
        raise BundleVerificationError("Aim-1 E0 manifest/split membership differs across seeds")
    expected_membership = (
        contract_inputs["manifest"]["sha256"],
        contract_inputs["split_integrity"]["sha256"],
    )
    if split_evidence[0] != expected_membership:
        raise BundleVerificationError(
            "Aim-1 E0 training membership evidence differs from frozen contract inputs"
        )


def _training_input_evidence(record: Any, paths: BundlePaths, *, context: str) -> tuple[str, str]:
    """Rehash a training identity and return its manifest/split-integrity hashes."""

    identity_path = _validate_recorded_identity(record, paths, context=context)
    training_identity = _load_json(identity_path, label=context)
    evidence = training_identity.get("payload", {}).get("input_evidence", {})
    manifest_sha = evidence.get("manifest_sha256")
    split_sha = evidence.get("split_integrity_sha256")
    if (
        not isinstance(manifest_sha, str)
        or core._SHA256_RE.fullmatch(manifest_sha) is None  # noqa: SLF001
        or not isinstance(split_sha, str)
        or core._SHA256_RE.fullmatch(split_sha) is None  # noqa: SLF001
    ):
        raise BundleVerificationError(f"{context} lacks manifest/split membership evidence")
    return manifest_sha, split_sha


def _validate_split_integrity_content(
    manifest_record: Any,
    splits_record: Any,
    integrity_record: Any,
    paths: BundlePaths,
    *,
    context: str,
) -> None:
    """Replay the split-integrity digest from manifest bytes followed by split bytes."""

    manifest_path = _validate_recorded_identity(
        manifest_record, paths, context=f"{context}/manifest"
    )
    splits_path = _validate_recorded_identity(splits_record, paths, context=f"{context}/splits")
    integrity_path = _validate_recorded_identity(
        integrity_record, paths, context=f"{context}/split integrity"
    )
    integrity_value = _load_json(integrity_path, label=f"{context} split integrity")
    if set(integrity_value) != {"hash", "csv_path"}:
        raise BundleVerificationError(f"{context} split-integrity schema changed")
    recorded_hash = integrity_value.get("hash")
    csv_path = integrity_value.get("csv_path")
    if (
        not isinstance(recorded_hash, str)
        or core._SHA256_RE.fullmatch(recorded_hash) is None  # noqa: SLF001
        or not isinstance(csv_path, str)
        or not csv_path
    ):
        raise BundleVerificationError(f"{context} split-integrity content is invalid")
    original_manifest = _resolve_source_path(csv_path, paths)
    if original_manifest.is_file() and sha256_file(original_manifest) != str(
        manifest_record.get("sha256")
    ):
        raise BundleVerificationError(
            f"{context} split-integrity manifest pointer differs from governed manifest"
        )
    digest = hashlib.sha256()
    for input_path in (manifest_path, splits_path):
        with input_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    if digest.hexdigest() != recorded_hash:
        raise BundleVerificationError(f"{context} split-integrity digest does not replay")


def _validate_aim2_split_membership(paths: BundlePaths, sources: list[dict[str, Any]]) -> None:
    """Prove identical source-CV membership across all five seeds in every LOCO arm."""

    by_id = core._source_map(sources)  # noqa: SLF001
    adapted = _core_paths(paths)
    contract = core._source_json(  # noqa: SLF001
        by_id, "aim2-loco-five-seed-contract", adapted
    )
    training = core._source_json(  # noqa: SLF001
        by_id, "final-v9-five-seed-campaign-training-completion", adapted
    )
    arms = contract.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(core.AIM2_ALL_ARMS):
        raise BundleVerificationError("Aim-2 split-proof arm roster changed")

    training_adoptions = [
        record
        for record in contract.get("adoptions", [])
        if isinstance(record, dict) and record.get("kind") == "training"
    ]
    adopted_records = {
        (record.get("arm"), record.get("seed")): record for record in training_adoptions
    }
    if len(training_adoptions) != 27 or len(adopted_records) != 27:
        raise BundleVerificationError("Aim-2 adopted split-proof chains are duplicated")
    if set(adopted_records) != {
        (arm, seed) for arm in core.AIM2_ALL_ARMS for seed in ADOPTED_MODEL_SEEDS
    }:
        raise BundleVerificationError("Aim-2 adopted split-proof chain roster changed")
    top_receipts = training.get("job_receipts")
    if not isinstance(top_receipts, dict):
        raise BundleVerificationError("Aim-2 split proof lacks campaign job receipts")

    for arm in core.AIM2_ALL_ARMS:
        arm_record = arms[arm]
        inputs = arm_record.get("inputs") if isinstance(arm_record, dict) else None
        if (
            not isinstance(inputs, dict)
            or inputs.get("fold_manifest_reused_for_every_seed") is not True
        ):
            raise BundleVerificationError(f"Aim-2 {arm} does not declare shared folds")
        source_manifest = inputs.get("source_manifest")
        split_file = inputs.get("split_file")
        _validate_recorded_identity(source_manifest, paths, context=f"Aim-2/{arm}/source manifest")
        _validate_recorded_identity(split_file, paths, context=f"Aim-2/{arm}/split file")
        if arm_record.get("source_sha256") != source_manifest.get("sha256") or arm_record.get(
            "split_sha256"
        ) != split_file.get("sha256"):
            raise BundleVerificationError(f"Aim-2 {arm} contract split identities disagree")

        evidence_by_seed: dict[int, tuple[str, str]] = {}
        for seed in ADOPTED_MODEL_SEEDS:
            adopted = adopted_records[(arm, seed)]
            source_cv = adopted.get("source_cv")
            semantic = source_cv.get("semantic_validation") if isinstance(source_cv, dict) else None
            if (
                not isinstance(semantic, dict)
                or semantic.get("source_manifest") != source_manifest
                or semantic.get("split_file") != split_file
                or semantic.get("test_roster_exact_for_every_fold") is not True
                or semantic.get("validation_roster_exact_for_every_fold") is not True
                or semantic.get("oof_label_and_fold_exact_per_slide") is not True
            ):
                raise BundleVerificationError(
                    f"Aim-2 {arm} seed {seed} adopted fold membership was not exact"
                )
            evidence_by_seed[seed] = _training_input_evidence(
                source_cv.get("identity"),
                paths,
                context=f"Aim-2/{arm}/seed{seed}/training identity",
            )

        for seed in NEW_MODEL_SEEDS:
            job_id = f"aim2.source_cv.{arm}.seed{seed}"
            top_record = top_receipts.get(job_id)
            top_path = _validate_recorded_identity(
                top_record, paths, context=f"Aim-2/{arm}/seed{seed}/top receipt"
            )
            top = _load_json(top_path, label=f"Aim-2 {arm} seed {seed} top receipt")
            if top.get("job_id") != job_id or top.get("status") != "completed_rc0":
                raise BundleVerificationError(f"Aim-2 {arm} seed {seed} top receipt changed")
            component_record = top.get("component_evidence", {}).get("component_receipt")
            component_path = _validate_recorded_identity(
                component_record,
                paths,
                context=f"Aim-2/{arm}/seed{seed}/component receipt",
            )
            component = _load_json(
                component_path, label=f"Aim-2 {arm} seed {seed} component receipt"
            )
            if component.get("job_key") != job_id or component.get("status") != "completed_rc0":
                raise BundleVerificationError(f"Aim-2 {arm} seed {seed} component receipt changed")
            artifacts = component.get("artifacts")
            if (
                not isinstance(artifacts, dict)
                or artifacts.get("arm") != arm
                or artifacts.get("seed") != seed
            ):
                raise BundleVerificationError(
                    f"Aim-2 {arm} seed {seed} component semantics changed"
                )
            evidence_by_seed[seed] = _training_input_evidence(
                artifacts.get("identity"),
                paths,
                context=f"Aim-2/{arm}/seed{seed}/training identity",
            )

        if set(evidence_by_seed) != set(MODEL_SEEDS) or len(set(evidence_by_seed.values())) != 1:
            raise BundleVerificationError(
                f"Aim-2 {arm} manifest/split membership differs across model seeds"
            )
        manifest_sha, split_integrity_sha = next(iter(evidence_by_seed.values()))
        if manifest_sha != source_manifest.get("sha256"):
            raise BundleVerificationError(f"Aim-2 {arm} training manifest differs from contract")
        split_path = _resolve_source_path(str(split_file["path"]), paths)
        integrity_path = split_path.parent / ".integrity_hash"
        integrity_record = identity(integrity_path, display_path=str(integrity_path))
        if integrity_record["sha256"] != split_integrity_sha:
            raise BundleVerificationError(
                f"Aim-2 {arm} training identity differs from live split-integrity artifact"
            )
        _validate_split_integrity_content(
            source_manifest,
            split_file,
            integrity_record,
            paths,
            context=f"Aim-2/{arm}/frozen membership",
        )


def _validate_aim3_split_membership(paths: BundlePaths, sources: list[dict[str, Any]]) -> None:
    """Prove identical patient/slide inputs across seeds for every Aim-3 task/draw."""

    by_id = core._source_map(sources)  # noqa: SLF001
    adapted = _core_paths(paths)
    contract = core._source_json(  # noqa: SLF001
        by_id, "aim3-ladders-five-seed-contract", adapted
    )
    audit = core._source_json(  # noqa: SLF001
        by_id, "aim3-ladders-five-seed-analysis-audit", adapted
    )
    adopted = contract.get("adopted_old_chains")
    new = audit.get("new_chains")
    if not isinstance(adopted, list) or len(adopted) != 96:
        raise BundleVerificationError("Aim-3 adopted split-proof chain census changed")
    if not isinstance(new, list) or len(new) != 64:
        raise BundleVerificationError("Aim-3 new split-proof chain census changed")

    groups: dict[tuple[str, str, int | None], dict[int, tuple[str, str, str]]] = {}
    for index, record in enumerate([*adopted, *new]):
        if not isinstance(record, dict):
            raise BundleVerificationError(f"Aim-3 split-proof chain {index} is malformed")
        job = record.get("job")
        artifacts = record.get("artifacts")
        if not isinstance(job, dict) or not isinstance(artifacts, dict):
            raise BundleVerificationError(f"Aim-3 split-proof chain {index} lacks semantics")
        component = job.get("component")
        task = job.get("task")
        draw_seed = job.get("draw_seed")
        seed = job.get("model_seed")
        key = (component, task, draw_seed)
        if key not in core._aim3_logical_roster() or seed not in MODEL_SEEDS:  # noqa: SLF001
            raise BundleVerificationError(f"Aim-3 split-proof chain {index} is out of scope")

        hashes: list[str] = []
        for artifact_name in ("manifest", "splits", "split_integrity"):
            artifact = artifacts.get(artifact_name)
            _validate_recorded_identity(
                artifact,
                paths,
                context=(f"Aim-3/{component}/{task}/{draw_seed}/seed{seed}/{artifact_name}"),
            )
            hashes.append(str(artifact.get("sha256")))
        identity_evidence = _training_input_evidence(
            artifacts.get("training_identity"),
            paths,
            context=f"Aim-3/{component}/{task}/{draw_seed}/seed{seed}/training identity",
        )
        if identity_evidence != (hashes[0], hashes[2]):
            raise BundleVerificationError(
                f"Aim-3 {component}/{task}/{draw_seed} seed {seed} input evidence disagrees"
            )
        _validate_split_integrity_content(
            artifacts["manifest"],
            artifacts["splits"],
            artifacts["split_integrity"],
            paths,
            context=f"Aim-3/{component}/{task}/{draw_seed}/seed{seed}",
        )
        seed_map = groups.setdefault(key, {})
        if seed in seed_map:
            raise BundleVerificationError(
                f"Aim-3 {component}/{task}/{draw_seed} duplicates seed {seed}"
            )
        seed_map[seed] = (hashes[0], hashes[1], hashes[2])

    if set(groups) != core._aim3_logical_roster():  # noqa: SLF001
        raise BundleVerificationError("Aim-3 split-proof task/draw roster changed")
    for key, seed_map in groups.items():
        if set(seed_map) != set(MODEL_SEEDS) or len(set(seed_map.values())) != 1:
            raise BundleVerificationError(
                f"Aim-3 {key[0]}/{key[1]}/{key[2]} membership differs across seeds"
            )


def _validate_delta_contrast(
    block: Any, *, left_key: str, right_key: str, context: str
) -> tuple[float, float, float, int]:
    if not isinstance(block, dict):
        raise BundleVerificationError(f"{context} must be an object")
    left = _finite_number(
        block.get(left_key), context=f"{context}/{left_key}", minimum=0.0, maximum=1.0
    )
    right = _finite_number(
        block.get(right_key), context=f"{context}/{right_key}", minimum=0.0, maximum=1.0
    )
    delta = _finite_number(
        block.get("delta_auroc"), context=f"{context}/delta_auroc", minimum=-1.0, maximum=1.0
    )
    if not math.isclose(delta, right - left, rel_tol=0.0, abs_tol=1e-12):
        raise BundleVerificationError(f"{context} delta is not right minus left")
    _ordered_interval(
        block.get("delta_auroc_ci95"),
        point=delta,
        context=f"{context}/delta_auroc_ci95",
        minimum=-1.0,
        maximum=1.0,
    )
    n = block.get("n_paired_patients")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise BundleVerificationError(f"{context}/n_paired_patients must be positive")
    return left, right, delta, n


def _validate_aim2_adjudication_additions(
    paths: BundlePaths, sources: list[dict[str, Any]]
) -> None:
    by_id = core._source_map(sources)  # noqa: SLF001
    adapted = _core_paths(paths)
    adjudication = core._source_json(  # noqa: SLF001
        by_id, "aim2-e2a-five-seed-adjudication-result", adapted
    )
    source = core._source_json(by_id, "aim2-loco-five-seed-results", adapted)  # noqa: SLF001
    if adjudication.get("schema_version") != 1:
        raise BundleVerificationError("Aim-2 adjudication schema version changed")
    precedence = adjudication.get("precedence", {})
    if precedence.get("retained_source_paired_delta_replay") != (
        "/e2a_d/source_report_paired_point_confirmation"
    ):
        raise BundleVerificationError("Aim-2 retained paired-point replay pointer changed")
    family_confirmation = adjudication.get("e2a_f", {}).get("source_report_confirmation")
    if family_confirmation != {
        "source_report_macro_point_and_interval_unchanged": True,
        "source_report_direction_points_and_intervals_unchanged": True,
        "only_gate_interpretation_superseded": True,
    }:
        raise BundleVerificationError("Aim-2 E2a-F source-value confirmation changed")
    inputs = adjudication.get("inputs")
    expected_input_keys = {
        "final_v8_preregistration",
        "frozen_aim2_five_seed_extension",
        "frozen_five_seed_campaign_controller",
        "frozen_final_v8_e2ad_topology",
        "campaign_contract",
        "campaign_final_receipt",
        "aim2_contract",
        "aim2_inference_seal",
        "aim2_results",
        "aim2_report_receipt",
        "aim2_primary_patient_table",
        "aim2_metastatic_patient_table",
        "raw_cpht_three_seed_result",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_input_keys:
        raise BundleVerificationError("Aim-2 adjudication input identity roster changed")
    for name, recorded in inputs.items():
        _validate_recorded_identity(recorded, paths, context=f"Aim-2 adjudication input {name}")
    e2a_d = adjudication.get("e2a_d", {})
    paired = e2a_d.get("paired_sibling_minus_family")
    slugs = ("sr386", "sr1482", "tcga_coad", "tcga_read")
    if not isinstance(paired, dict) or set(paired) != set(slugs):
        raise BundleVerificationError("Aim-2 sibling paired-contrast roster changed")
    source_keys = (
        "sibling_sr386_minus_family_surgen_SR386",
        "sibling_sr1482_minus_family_surgen_SR1482",
        "sibling_tcga_coad_minus_family_tcga_TCGA-COAD",
        "sibling_tcga_read_minus_family_tcga_TCGA-READ",
        "family_rih_sm_minus_family_rih",
    )
    source_paired = source.get("paired_sibling_and_size_matched_contrasts")
    if not isinstance(source_paired, dict) or set(source_paired) != set(source_keys):
        raise BundleVerificationError("Aim-2 retained source paired-point roster changed")
    confirmations = e2a_d.get("source_report_paired_point_confirmation", {}).get("comparisons")
    if not isinstance(confirmations, dict) or set(confirmations) != set(source_keys):
        raise BundleVerificationError("Aim-2 retained paired-point confirmation roster changed")

    observed: dict[str, tuple[float, float, float, int]] = {}
    for slug, source_key in zip(slugs, source_keys[:4], strict=True):
        observed[source_key] = _validate_delta_contrast(
            paired[slug],
            left_key="whole_family_held_out_auroc",
            right_key="sibling_retained_auroc",
            context=f"Aim-2/paired/{slug}",
        )
    size = e2a_d.get("size_matched_rih_sensitivity")
    observed[source_keys[4]] = _validate_delta_contrast(
        size,
        left_key="full_source_auroc",
        right_key="size_matched_auroc",
        context="Aim-2/paired/rih_size_matched",
    )
    for source_key, (left, right, delta, n) in observed.items():
        recorded = source_paired[source_key]
        confirmation = confirmations[source_key]
        if not isinstance(recorded, dict) or not isinstance(confirmation, dict):
            raise BundleVerificationError(
                f"Aim-2 paired source/confirmation is malformed: {source_key}"
            )
        expected = {
            "delta_auroc": delta,
            "auroc_left": left,
            "auroc_right": right,
            "n_patients": n,
        }
        for field, value in expected.items():
            if isinstance(value, float):
                minimum, maximum = (-1.0, 1.0) if field == "delta_auroc" else (0.0, 1.0)
                recorded_value = _finite_number(
                    recorded.get(field),
                    context=f"Aim-2/source/{source_key}/{field}",
                    minimum=minimum,
                    maximum=maximum,
                )
                confirmation_value = _finite_number(
                    confirmation.get(field),
                    context=f"Aim-2/confirmation/{source_key}/{field}",
                    minimum=minimum,
                    maximum=maximum,
                )
                if not math.isclose(recorded_value, value, rel_tol=0.0, abs_tol=1e-12):
                    raise BundleVerificationError(
                        f"Aim-2 retained source paired point changed: {source_key}/{field}"
                    )
                if not math.isclose(confirmation_value, value, rel_tol=0.0, abs_tol=1e-12):
                    raise BundleVerificationError(
                        f"Aim-2 paired confirmation changed: {source_key}/{field}"
                    )
            elif recorded.get(field) != value or confirmation.get(field) != value:
                raise BundleVerificationError(
                    f"Aim-2 retained paired census changed: {source_key}/{field}"
                )
        if confirmation.get("matches") is not True:
            raise BundleVerificationError(f"Aim-2 paired confirmation failed: {source_key}")

    met = adjudication.get("e2met_gate_confirmation")
    if not isinstance(met, dict) or set(met.get("targets", {})) != {"rih_m", "sr1482_m"}:
        raise BundleVerificationError("Aim-2 E2-MET confirmation target roster changed")
    target_points: list[float] = []
    for target, block in met["targets"].items():
        point = _finite_number(
            block.get("auroc"), context=f"Aim-2/E2-MET/{target}/auroc", minimum=0.0, maximum=1.0
        )
        _ordered_interval(
            block.get("auroc_ci95"),
            point=point,
            context=f"Aim-2/E2-MET/{target}/auroc_ci95",
            minimum=0.0,
            maximum=1.0,
        )
        target_points.append(point)
    macro = _finite_number(
        met.get("equal_cohort_metastatic_macro_auroc"),
        context="Aim-2/E2-MET/macro",
        minimum=0.0,
        maximum=1.0,
    )
    macro_low, _macro_high = _ordered_interval(
        met.get("macro_auroc_ci95"),
        point=macro,
        context="Aim-2/E2-MET/macro_ci95",
        minimum=0.0,
        maximum=1.0,
    )
    if not math.isclose(macro, sum(target_points) / 2.0, rel_tol=0.0, abs_tol=1e-12):
        raise BundleVerificationError("Aim-2 E2-MET macro is not the equal-target mean")
    both = all(point > 0.5 for point in target_points)
    lower = macro_low > 0.5
    if (
        met.get("both_target_points_above_0p5") is not both
        or met.get("macro_lower_bound_above_0p5") is not lower
        or met.get("claim_metastatic_transport") is not (both and lower)
        or met.get("matches_and_remains_authoritative") is not True
    ):
        raise BundleVerificationError("Aim-2 E2-MET gate does not replay")
    source_met = source.get("e2met_confirmatory")
    source_targets = source.get("e2met_confirmatory_family_naive")
    if not isinstance(source_met, dict) or not isinstance(source_targets, dict):
        raise BundleVerificationError("Aim-2 source report lacks E2-MET confirmation")
    for field in (
        "equal_cohort_metastatic_macro_auroc",
        "both_target_points_above_0p5",
        "macro_lower_bound_above_0p5",
        "claim_metastatic_transport",
        "gate",
    ):
        observed = met.get(field)
        recorded = source_met.get(field)
        if isinstance(observed, float):
            if not math.isclose(float(recorded), observed, rel_tol=0.0, abs_tol=1e-12):
                raise BundleVerificationError(f"Aim-2 source E2-MET field changed: {field}")
        elif recorded != observed:
            raise BundleVerificationError(f"Aim-2 source E2-MET field changed: {field}")
    if source_met.get("macro_auroc_ci95") != met.get("macro_auroc_ci95"):
        raise BundleVerificationError("Aim-2 source E2-MET macro interval changed")
    for target in ("rih_m", "sr1482_m"):
        source_block = source_targets.get(target, {})
        target_block = met["targets"][target]
        if source_block.get("auroc") != target_block.get("auroc") or source_block.get(
            "auroc_ci95"
        ) != target_block.get("auroc_ci95"):
            raise BundleVerificationError(f"Aim-2 source E2-MET target changed: {target}")


def _metric_ci_text(point: Any, interval: Any) -> str:
    if (
        not isinstance(point, (int, float))
        or isinstance(point, bool)
        or not isinstance(interval, list)
        or len(interval) != 2
        or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in interval)
    ):
        raise BundleVerificationError("governed report metric lacks a numeric point/interval")
    return f"{float(point):.4f} [{float(interval[0]):.4f}, {float(interval[1]):.4f}]"


def _require_claim_text(text: str, expected: str, *, label: str) -> None:
    if not expected or expected not in text:
        raise BundleVerificationError(
            f"report document does not bind governed {label} value {expected!r}"
        )


def _markdown_section(text: str, heading: str) -> str:
    """Return one exact Markdown heading section through the next peer/parent heading."""

    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines) if line == heading]
    if len(matches) != 1 or not heading.startswith("#"):
        raise BundleVerificationError(f"report must contain one exact section {heading!r}")
    start = matches[0]
    level = len(heading) - len(heading.lstrip("#"))
    stop = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("#"):
            next_level = len(line) - len(line.lstrip("#"))
            if next_level <= level:
                stop = index
                break
    return "\n".join(lines[start:stop])


def _require_markdown_row(
    text: str, row_label: str, expected_values: tuple[str, ...], *, label: str
) -> None:
    rows: list[tuple[str, ...]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = tuple(cell.strip() for cell in stripped[1:-1].split("|"))
        if cells and cells[0] == row_label:
            rows.append(cells)
    if len(rows) != 1 or not all(value in rows[0][1:] for value in expected_values):
        raise BundleVerificationError(
            f"Results.md does not map governed {label} values to row {row_label!r}: "
            f"{expected_values}"
        )


def _validate_audit_source_identity_table(
    audit_text: str, by_id: dict[str, dict[str, Any]]
) -> None:
    """Bind every new source ID to its one governed digest in a canonical table."""

    section = _markdown_section(audit_text, "### Governed five-seed source identities")
    header = "| Source ID | SHA-256 |"
    separator = "|---|---|"
    lines = [line.strip() for line in section.splitlines()]
    if lines.count(header) != 1 or lines.count(separator) != 1:
        raise BundleVerificationError(
            "Audit source identity table must contain one canonical header"
        )
    observed: list[tuple[str, str]] = []
    for line in lines:
        if line in {header, separator} or not line.startswith("|"):
            continue
        if not line.endswith("|"):
            raise BundleVerificationError("Audit source identity table has a malformed row")
        cells = tuple(cell.strip() for cell in line[1:-1].split("|"))
        if len(cells) != 2:
            raise BundleVerificationError("Audit source identity table has a noncanonical row")
        observed.append((cells[0], cells[1]))
    expected = {
        (f"`{source_id}`", f"`{by_id[source_id]['sha256']}`")
        for source_id in _EXPECTED_NEW_SOURCE_METADATA
    }
    if len(observed) != len(expected) or set(observed) != expected:
        raise BundleVerificationError(
            "Audit source identity table does not contain exactly one canonical ID-to-SHA row "
            "for each of the 29 governed new sources"
        )


def _validate_report_claims(paths: BundlePaths, sources: list[dict[str, Any]]) -> None:
    """Bind upgraded and retained headline numbers to governed JSON fields."""

    by_id = core._source_map(sources)  # noqa: SLF001
    adapted = _core_paths(paths)
    text = (paths.final_v10 / "Results.md").read_text(encoding="utf-8")
    audit_text = (paths.final_v10 / "Audit.md").read_text(encoding="utf-8")
    setup_text = (paths.final_v10 / "Experimental_Setup.md").read_text(encoding="utf-8")
    summary_section = _markdown_section(text, "## Result status and interpretation rule")
    e0_section = _markdown_section(text, "### Five-seed E0 governed result")
    e1v_section = _markdown_section(text, "### Five-seed E1v encoder-sensitivity reference")
    aim2_section = _markdown_section(
        text, "## Aim 2 — transfer, adaptation, and deployment boundaries"
    )
    e2a_f_section = _markdown_section(text, "### Five-seed E2a-F family directions")
    e2a_d_section = _markdown_section(text, "### Five-seed E2a-D sibling directions")
    paired_section = _markdown_section(text, "### Corrected paired contrasts")
    e2met_section = _markdown_section(text, "### E2-MET conformant gate")
    fixed_section = _markdown_section(text, "### Five-seed fixed UNI-v1 ladder")
    repeated_section = _markdown_section(text, "### Five-seed repeated-control consensus")
    e3v_section = _markdown_section(text, "### Five-seed Virchow2-CLS replication")

    aim1 = core._source_json(by_id, "aim1-e0-five-seed-results", adapted)  # noqa: SLF001
    primary = aim1.get("primary_median_seed_macro5", {})
    _require_markdown_row(
        e0_section,
        "**Primary: median of five seed-specific equal-five-domain macros**",
        (_metric_ci_text(primary.get("point"), primary.get("ci95")),),
        label="Aim-1 primary macro",
    )
    ensemble = aim1.get("five_seed_ensemble", {})
    _require_markdown_row(
        e0_section,
        "Five-seed mean-native-logit equal-five-domain macro sensitivity",
        (_metric_ci_text(ensemble.get("macro5_auroc"), ensemble.get("macro5_ci95")),),
        label="Aim-1 ensemble macro sensitivity",
    )
    _require_markdown_row(
        e0_section,
        "Five-seed pooled continuity sensitivity",
        (
            f"{float(ensemble.get('pooled_auroc')):.4f}",
            f"{float(ensemble.get('pooled_auprc')):.4f}",
        ),
        label="Aim-1 pooled AUROC/AUPRC sensitivity",
    )
    for key, row_label, label in (
        (
            "family_macro4_median_seed",
            "Median-seed equal-four-family sensitivity",
            "Aim-1 equal-four-family sensitivity",
        ),
        (
            "patient_count_weighted_macro_median_seed",
            "Median-seed patient-count-weighted sensitivity",
            "Aim-1 weighted sensitivity",
        ),
    ):
        block = aim1.get(key, {})
        _require_markdown_row(
            e0_section,
            row_label,
            (_metric_ci_text(block.get("point"), block.get("ci95")),),
            label=label,
        )
    cptac = aim1.get("primitive_median_seed_auroc", {}).get("CPTAC", {})
    _require_claim_text(
        e0_section,
        _metric_ci_text(cptac.get("point"), cptac.get("ci95")),
        label="Aim-1 CPTAC primitive",
    )
    _require_claim_text(
        e0_section,
        "1,486 patients and 1,642 slides, including 604 KRAS-mutant and 882 wild-type patients",
        label="Aim-1 patient/slide census",
    )

    cpht = core._source_json(by_id, "aim2-e2cpht-results", adapted)  # noqa: SLF001
    raw_cpht = (
        cpht.get("populations", {}).get("all_40", {}).get("metrics", {}).get("all_conventional", {})
    )
    _require_claim_text(
        aim2_section,
        _metric_ci_text(raw_cpht.get("auroc"), raw_cpht.get("auroc_ci95")),
        label="unchanged raw E2-CPHT AUROC",
    )

    adjudication = core._source_json(  # noqa: SLF001
        by_id, "aim2-e2a-five-seed-adjudication-result", adapted
    )
    e2a_f = adjudication.get("e2a_f", {})
    for name, block in e2a_f.get("directions", {}).items():
        _require_markdown_row(
            e2a_f_section,
            name,
            (
                _metric_ci_text(block.get("auroc"), block.get("auroc_ci95")),
                f"`{str(block.get('directional_gate', {}).get('passes')).upper()}`",
            ),
            label=f"E2a-F {name}",
        )
    family_macro = e2a_f.get("nested_four_family_macro", {})
    _require_markdown_row(
        e2a_f_section,
        "Nested four-family macro",
        (_metric_ci_text(family_macro.get("auroc"), family_macro.get("auroc_ci95")),),
        label="E2a-F nested macro",
    )
    _require_markdown_row(
        e2a_f_section,
        "Family LOCO transport claim",
        (f"`{str(e2a_f.get('adjudication', {}).get('claim_family_loco_transport')).upper()}`",),
        label="E2a-F family transport claim",
    )
    e2a_d = adjudication.get("e2a_d", {})
    for name, block in e2a_d.get("targets", {}).items():
        _require_markdown_row(
            e2a_d_section,
            name,
            (
                _metric_ci_text(block.get("auroc"), block.get("auroc_ci95")),
                f"`{str(block.get('directional_gate', {}).get('passes')).upper()}`",
            ),
            label=f"E2a-D {name}",
        )
    for name, block in e2a_d.get("macros", {}).items():
        _require_markdown_row(
            e2a_d_section,
            name,
            (_metric_ci_text(block.get("auroc"), block.get("auroc_ci95")),),
            label=f"E2a-D {name} macro",
        )
    _require_markdown_row(
        e2a_d_section,
        "Sibling-stratum transport claim",
        (f"`{str(e2a_d.get('adjudication', {}).get('claim_sibling_stratum_transport')).upper()}`",),
        label="E2a-D sibling-stratum transport claim",
    )
    for slug, block in e2a_d.get("paired_sibling_minus_family", {}).items():
        _require_markdown_row(
            paired_section,
            f"{slug} sibling minus family",
            (
                f"{float(block.get('delta_auroc')):+.4f} "
                f"[{float(block.get('delta_auroc_ci95')[0]):.4f}, "
                f"{float(block.get('delta_auroc_ci95')[1]):.4f}]",
            ),
            label=f"E2a-D paired {slug}",
        )
    size_matched = e2a_d.get("size_matched_rih_sensitivity", {})
    _require_markdown_row(
        paired_section,
        "RIH size-matched minus full-source",
        (
            f"{float(size_matched.get('delta_auroc')):+.4f} "
            f"[{float(size_matched.get('delta_auroc_ci95')[0]):.4f}, "
            f"{float(size_matched.get('delta_auroc_ci95')[1]):.4f}]",
        ),
        label="E2a-D RIH size-matched paired sensitivity",
    )
    e2met = adjudication.get("e2met_gate_confirmation", {})
    for target, block in e2met.get("targets", {}).items():
        _require_markdown_row(
            e2met_section,
            target,
            (_metric_ci_text(block.get("auroc"), block.get("auroc_ci95")),),
            label=f"E2-MET {target}",
        )
    _require_markdown_row(
        e2met_section,
        "Equal-cohort metastatic macro",
        (
            _metric_ci_text(
                e2met.get("equal_cohort_metastatic_macro_auroc"),
                e2met.get("macro_auroc_ci95"),
            ),
        ),
        label="E2-MET equal-cohort macro",
    )
    _require_markdown_row(
        e2met_section,
        "Metastatic transport claim",
        (f"`{str(e2met.get('claim_metastatic_transport')).upper()}`",),
        label="E2-MET conformant claim",
    )
    _require_markdown_row(
        summary_section,
        "Aim 2 E2a",
        (
            "Family macro "
            f"{_metric_ci_text(family_macro.get('auroc'), family_macro.get('auroc_ci95'))}; "
            "all four sibling directions pass",
            "E2a-F "
            f"`{str(e2a_f.get('adjudication', {}).get('claim_family_loco_transport')).upper()}`; "
            "E2a-D "
            f"`{str(e2a_d.get('adjudication', {}).get('claim_sibling_stratum_transport')).upper()}`",
        ),
        label="Aim-2 E2a summary",
    )
    _require_markdown_row(
        summary_section,
        "Aim 2 E2-MET",
        (
            "Five-seed conformant macro "
            f"{_metric_ci_text(e2met.get('equal_cohort_metastatic_macro_auroc'), e2met.get('macro_auroc_ci95'))}",
            "Metastatic transport claim "
            f"`{str(e2met.get('claim_metastatic_transport')).upper()}` "
            "within the declared LOCO-sensitivity scope",
        ),
        label="Aim-2 E2-MET summary",
    )
    historical_met_section = _markdown_section(
        text, "### Three-seed E2-MET continuity and unchanged detailed analyses"
    )
    _require_claim_text(
        historical_met_section,
        "The former three-seed family-naive gate is retained as historical continuity, "
        "not as the controlling FINAL-v10 gate",
        label="historical E2-MET precedence",
    )
    folded_results = text.casefold()
    for contradiction in (
        "remain the scientific e2-met gate record",
        "remains the scientific e2-met gate record",
        "metastatic transport was not established",
    ):
        if contradiction in folded_results:
            raise BundleVerificationError(
                f"Results.md retains contradictory Aim-2 prose: {contradiction!r}"
            )

    aim3 = core._source_json(by_id, "aim3-ladders-five-seed-results", adapted)  # noqa: SLF001
    task_labels = {
        "codon": "G12 vs non-G12 KRAS",
        "g12d_broad": "G12D vs other KRAS",
        "allele1": "G12D vs other G12",
        "allele2": "G12V vs other G12",
        "g12c": "G12C vs other G12",
    }
    section_text = {
        "fixed_univ1": fixed_section,
        "e3v_virchow2_cls": e3v_section,
    }
    for section_name in ("fixed_univ1", "e3v_virchow2_cls"):
        for task, rung in aim3.get(section_name, {}).get("rungs", {}).items():
            verdict = str(rung.get("gate", {}).get("verdict", ""))
            _require_markdown_row(
                section_text[section_name],
                task_labels.get(task, task),
                (
                    f"{float(rung.get('fine', {}).get('estimate')):.4f}",
                    f"{float(rung.get('control', {}).get('estimate')):.4f}",
                    f"{float(rung.get('delta_control_minus_fine', {}).get('estimate')):+.4f}",
                    f"`{verdict}`",
                ),
                label=f"Aim-3 {section_name}/{task} verdict",
            )
    for task, rung in aim3.get("repeated_univ1", {}).get("rungs", {}).items():
        verdict = str(rung.get("consensus_verdict", ""))
        _require_markdown_row(
            repeated_section,
            task_labels.get(task, task),
            (f"`{verdict}`",),
            label=f"Aim-3 repeated/{task} consensus",
        )
    e1v_root = aim3.get("e1v_virchow2_cls_gene_reference", {})
    e1v = e1v_root.get("five_seed_ensemble_A", {})
    _require_claim_text(
        e1v_section,
        _metric_ci_text(e1v.get("auroc"), [e1v.get("ci_low"), e1v.get("ci_high")]),
        label="Aim-1/E1v five-seed gene reference",
    )
    _require_claim_text(
        e1v_section,
        f"AUROC {float(e1v_root.get('five_seed_ensemble_D_auroc')):.4f} in Set D",
        label="Aim-1/E1v Set-D gene reference",
    )
    _require_claim_text(
        e1v_section,
        f"all {int(e1v.get('n')):,} patients, including {int(e1v.get('n_positive'))} mutants",
        label="Aim-1/E1v population",
    )

    fixed_rungs = aim3.get("fixed_univ1", {}).get("rungs", {})
    ceiling_order = ("codon", "g12d_broad", "allele1")
    fine_bounds = ", ".join(
        f"{float(fixed_rungs[task]['fine']['primary_fwer_one_sided']['upper']):.4f}"
        for task in ceiling_order
    )
    control_bounds = ", ".join(
        f"{float(fixed_rungs[task]['control']['primary_fwer_one_sided']['lower']):.4f}"
        for task in ceiling_order
    )
    delta_bounds = ", ".join(
        f"{float(fixed_rungs[task]['delta_control_minus_fine']['primary_fwer_one_sided']['lower']):.4f}"
        for task in ceiling_order
    )
    _require_claim_text(
        fixed_section,
        (
            "For the three ceiling rungs, the one-sided familywise fine upper bounds were "
            f"{fine_bounds}; control lower bounds were {control_bounds}; and delta lower "
            f"bounds were {delta_bounds}."
        ),
        label="Aim-3 fixed one-sided bounds",
    )
    _require_claim_text(
        fixed_section,
        "G12V missed the fine-ceiling and delta conditions. G12C did not establish a learnable control contrast.",
        label="Aim-3 unresolved fixed-gate conditions",
    )

    e3v_rungs = aim3.get("e3v_virchow2_cls", {}).get("rungs", {})
    worst_fine = max(
        float(rung["fine"]["primary_fwer_one_sided"]["upper"]) for rung in e3v_rungs.values()
    )
    worst_control = min(
        float(rung["control"]["primary_fwer_one_sided"]["lower"]) for rung in e3v_rungs.values()
    )
    worst_delta = min(
        float(rung["delta_control_minus_fine"]["primary_fwer_one_sided"]["lower"])
        for rung in e3v_rungs.values()
    )
    _require_claim_text(
        e3v_section,
        (
            f"The worst fine upper bound was {worst_fine:.4f}, the worst control lower bound "
            f"was {worst_control:.4f}, and the worst delta lower bound was {worst_delta:.4f}."
        ),
        label="Aim-3 E3v bound extrema",
    )

    _require_claim_text(
        setup_text,
        "The study-wide MIL census is 1,225 fits: 735 adopted three-seed fits plus 490 new fits for seeds 45 and 46. The new-fit allocation is Aim 1, 12; Aim 2, 108; and Aim 3, 370.",
        label="Experimental Setup fit census",
    )
    _require_claim_text(
        setup_text,
        "Canonical Aim 1 contains 1,642 conventional-primary slides from 1,486 patients, including 604 KRAS-mutant and 882 wild-type patients",
        label="Experimental Setup Aim-1 census",
    )
    _require_claim_text(
        audit_text,
        _metric_ci_text(primary.get("point"), primary.get("ci95")),
        label="Audit Aim-1 primary",
    )
    _require_claim_text(
        audit_text,
        "The canonical patient roster remains 1,486 patients and 1,642 slides, with 604 KRAS-mutant and 882 wild-type patients.",
        label="Audit Aim-1 census",
    )
    _validate_audit_source_identity_table(audit_text, by_id)


def _validate_documents(paths: BundlePaths, *, require_candidate: bool) -> dict[str, Any]:
    documents: dict[str, Any] = {}
    for filename in REPORT_DOCUMENTS:
        path = paths.final_v10 / filename
        document_identity = identity(path, display_path=_relative_display(path, paths.repo))
        try:
            core._validate_report_document(path)  # noqa: SLF001
        except core.BundleVerificationError as exc:
            raise BundleVerificationError(str(exc)) from exc
        text = path.read_text(encoding="utf-8")
        if not text.startswith("# FINAL-v10"):
            raise BundleVerificationError(f"{filename} must identify FINAL-v10 on its first line")
        if require_candidate:
            folded_text = text.casefold()
            marker = next(
                (value for value in _CANDIDATE_FORBIDDEN_TEXT if value.casefold() in folded_text),
                None,
            )
            if marker is None:
                marker = next(
                    (
                        match.group(0)
                        for pattern in _CANDIDATE_FINALIZATION_STATE_PATTERNS
                        if (match := pattern.search(text)) is not None
                    ),
                    None,
                )
            if marker is not None:
                raise BundleVerificationError(
                    f"candidate document {filename} still contains draft marker {marker!r}"
                )
            if FINAL_STATE_STATUS_PARAGRAPH not in text:
                raise BundleVerificationError(
                    f"candidate document {filename} must contain the exact FINAL-v10 "
                    "completion-status paragraph"
                )
            contradiction = next(
                (
                    (label, match.group(0))
                    for label, pattern in _CANDIDATE_SCIENTIFIC_CONTRADICTION_PATTERNS
                    if (match := pattern.search(text)) is not None
                ),
                None,
            )
            if contradiction is not None:
                label, matched_text = contradiction
                raise BundleVerificationError(
                    f"candidate document {filename} contains contradictory scientific claim "
                    f"{label!r}: {matched_text!r}"
                )
        documents[filename] = document_identity
    setup = (paths.final_v10 / "Experimental_Setup.md").read_text(encoding="utf-8")
    if "max_concurrent_gpu_trainers=6" not in setup:
        raise BundleVerificationError("Experimental_Setup.md must declare six-way training")
    if core._ORION_EXCLUSION_RE.search(setup) is None:  # noqa: SLF001
        raise BundleVerificationError(
            "Experimental_Setup.md must exclude Orion from canonical Aim 1"
        )
    try:
        core._validate_five_seed_audit_scope(paths.final_v10 / "Audit.md")  # noqa: SLF001
    except core.BundleVerificationError as exc:
        raise BundleVerificationError(str(exc)) from exc
    audit = (paths.final_v10 / "Audit.md").read_text(encoding="utf-8")
    results = (paths.final_v10 / "Results.md").read_text(encoding="utf-8")
    five_seed_scope = core._audit_scope_paragraph(audit, "Five-seed scope")  # noqa: SLF001
    _require_claim_text(
        five_seed_scope,
        "eight controlling arms plus the RIH size-matched sensitivity",
        label="Aim-2 eight-controlling-plus-one-sensitivity scope",
    )
    _require_claim_text(
        setup,
        "eight controlling LOCO score ensembles plus the RIH size-matched sensitivity ensemble",
        label="Experimental Setup Aim-2 eight-controlling-plus-one-sensitivity scope",
    )
    _require_claim_text(
        audit,
        "Five-seed analysis controls eight LOCO arms; the ninth ensemble, RIH size-matched, remains a prespecified sensitivity.",
        label="Audit Aim-2 eight-controlling-plus-one-sensitivity scope",
    )
    for document_name, document_text in {
        "Experimental_Setup.md": setup,
        "Audit.md": audit,
        "Results.md": results,
    }.items():
        folded = document_text.casefold()
        nine_loco_controlling = re.search(
            r"\b(?:all\s+)?nine\s+loco"
            r"(?:\s+(?!are\b|controlling\b)[a-z0-9_-]+){0,4}"
            r"\s+(?:are\s+)?controlling\b",
            folded,
        )
        rih_size_matched_controlling = re.search(
            r"\brih\s+size[- ]matched\b[^.!?\n]{0,100}\bcontroll[a-z]*\b",
            folded,
        )
        if (
            "nine controlling" in folded
            or "controls the nine loco" in folded
            or nine_loco_controlling is not None
            or rih_size_matched_controlling is not None
        ):
            raise BundleVerificationError(
                f"{document_name} incorrectly promotes the RIH size-matched sensitivity"
            )
    return documents


def draft_status(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Validate available evidence and return the unsealed staging state."""

    selected = default_paths() if paths is None else paths
    _validate_documents(selected, require_candidate=False)
    manifest, sources, pending = _validate_manifest(selected, require_candidate=False)
    missing_paths = [
        str(item["path"])
        for item in pending
        if not _resolve_source_path(str(item["path"]), selected).is_file()
    ]
    ready_paths = [
        str(item["path"])
        for item in pending
        if _resolve_source_path(str(item["path"]), selected).is_file()
    ]
    new_ids = set(_required_sources(selected))
    materialized_new = sum(str(source["id"]) in new_ids for source in sources)
    return {
        "bundle": "reports/final_v10",
        "status": manifest["status"],
        "published_receipt_present": selected.destination.exists(),
        "adopted_source_count": sum(
            str(source["id"]) in _EXPECTED_BASE_SOURCE_IDS for source in sources
        ),
        "materialized_five_seed_source_count": materialized_new,
        "pending_source_count": len(pending),
        "pending_now_materialized_count": len(ready_paths),
        "missing_source_count": len(missing_paths),
        "pending_now_materialized": ready_paths,
        "missing_sources": missing_paths,
        "training_stage_replay": "PASS",
    }


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _atomic_replace_bytes(path: Path, content: bytes) -> None:
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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def refresh_manifest(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Authenticate currently available pending sources without publishing a receipt."""

    selected = default_paths() if paths is None else paths
    if selected.destination.exists():
        raise BundleVerificationError("refusing to refresh a manifest after FINAL-v10 was sealed")
    manifest, sources, pending = _validate_manifest(selected, require_candidate=False)
    if set(manifest) == _FLAT_MANIFEST_KEYS:
        return draft_status(selected)
    remaining: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    for source in pending:
        path = _resolve_source_path(str(source["path"]), selected)
        if path.is_file() and not path.is_symlink():
            promoted.append({**source, **identity(path, display_path=str(source["path"]))})
        else:
            remaining.append(source)
    if promoted:
        manifest["artifacts"] = [*manifest["artifacts"], *promoted]
        manifest["pending_artifacts"] = remaining
        if remaining:
            manifest["status"] = DRAFT_STATUS
        else:
            adopted = [
                source for source in sources if str(source["id"]) in _EXPECTED_BASE_SOURCE_IDS
            ]
            manifest = {
                "schema_version": 2,
                "bundle": "final_v10",
                "status": CANDIDATE_STATUS,
                "artifacts": [*adopted, *manifest["artifacts"]],
                "pending_artifacts": [],
            }
            candidate_sources = sorted(manifest["artifacts"], key=lambda item: str(item["id"]))
            try:
                _validate_documents(selected, require_candidate=True)
                _validate_complete_source_graph(selected, candidate_sources, validate_claims=True)
            except BundleVerificationError as exc:
                raise BundleVerificationError(
                    f"refusing to flatten an invalid FINAL-v10 candidate: {exc}"
                ) from exc
        _atomic_replace_bytes(selected.final_v10 / SOURCE_MANIFEST_NAME, _manifest_bytes(manifest))
    return draft_status(selected)


def _verification_identities(paths: BundlePaths) -> dict[str, Any]:
    return {
        "verifier": identity(
            paths.verifier_code,
            display_path=_relative_display(paths.verifier_code, paths.repo),
        ),
        "tests": identity(
            paths.verifier_test,
            display_path=_relative_display(paths.verifier_test, paths.repo),
        ),
        "frozen_semantic_core": identity(
            paths.core_verifier,
            display_path=_relative_display(paths.core_verifier, paths.repo),
        ),
    }


def _validated_receipt_created_utc(paths: BundlePaths) -> str:
    """Return the one frozen, canonical UTC publication timestamp."""

    value = paths.expected_created_utc
    if paths.repo.resolve() == REPO.resolve() and value != EXPECTED_RECEIPT_CREATED_UTC:
        raise BundleVerificationError(
            "production FINAL-v10 receipt must use the frozen verifier timestamp constant"
        )
    if not isinstance(value, str) or value == _UNFROZEN_RECEIPT_CREATED_UTC:
        raise BundleVerificationError(
            "FINAL-v10 receipt timestamp must be frozen immediately before sealing"
        )
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise BundleVerificationError("frozen FINAL-v10 receipt timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise BundleVerificationError("frozen FINAL-v10 receipt timestamp must be UTC")
    if parsed.isoformat() != value:
        raise BundleVerificationError(
            "frozen FINAL-v10 receipt timestamp must use canonical ISO-8601 +00:00 form"
        )
    return value


def build_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Build but do not publish a fully validated FINAL-v10 receipt."""

    selected = default_paths() if paths is None else paths
    documents = _validate_documents(selected, require_candidate=True)
    manifest, sources, pending = _validate_manifest(selected, require_candidate=True)
    if pending:
        raise BundleVerificationError(
            "internal error: candidate validation retained pending sources"
        )
    manifest_path = selected.final_v10 / SOURCE_MANIFEST_NAME
    return {
        "schema_version": 1,
        "bundle": "reports/final_v10",
        "status": SEALED_STATUS,
        "created_utc": _validated_receipt_created_utc(selected),
        "organization": "aim_focused_additive_five_seed_successor",
        "post_outcome_official_specification": True,
        "not_preregistration": True,
        "append_only_five_seed_extension": True,
        "model_seed_scope": {
            "adopted": ADOPTED_MODEL_SEEDS,
            "new": NEW_MODEL_SEEDS,
            "complete": MODEL_SEEDS,
            "patient_and_slide_fold_membership_changes_across_model_seeds": False,
            "model_seeds_are_inference_units": False,
        },
        "study_wide_mil_census": {
            "adopted_fits": 735,
            "new_fits": 490,
            "complete_fits": 1225,
            "new_training_chain_jobs": 102,
            "maximum_concurrent_gpu_trainers": 6,
            "observed_peak_concurrent_gpu_trainers": 6,
        },
        "mixed_seed_scope": {
            "five_seed": [
                "Aim 1 E0",
                "Aim 1 E1v",
                "Aim 2 all nine LOCO primary score ensembles (eight controlling plus RIH size-matched sensitivity)",
                "Aim 2 E2-MET LOCO sensitivity",
                "Aim 2 Orion LOCO sensitivity",
                "Aim 3 fixed controls",
                "Aim 3 repeated controls",
                "Aim 3 E3v",
            ],
            "three_seed_unchanged": [
                "raw E2-CPHT",
                "E2-CPHT-A",
                "15-fold Orion sensitivity",
                "E2e",
                "E2f-v3",
                "between-slide analysis",
                "detailed E2-MET role/organ analyses",
                "unaffected Aim 1 analyses",
            ],
        },
        "declared_scientific_states": {
            "canonical_aim1_orion_included": False,
            "cpht_r": "NOT_RUN",
            "aim4_whole_section_pathology_validation": "GENERATED_UNREAD",
        },
        "documents": documents,
        "source_manifest": identity(
            manifest_path, display_path=_relative_display(manifest_path, selected.repo)
        ),
        "adopted_base_manifest": identity(
            selected.base_manifest,
            display_path=_relative_display(selected.base_manifest, selected.repo),
        ),
        "authoritative_sources": sources,
        "verification": _verification_identities(selected),
        "checks": {
            "ordered_aim_documents": "PASS",
            "no_draft_or_peek_markers": "PASS",
            "exact_flat_81_source_inventory": "PASS",
            "direct_source_rehash": "PASS",
            "complete_five_seed_source_inventory": "PASS",
            "upgraded_and_controlling_headline_validation": "PASS",
            "unchanged_continuity_claims_independently_reviewed_and_final_document_pinned": (
                "PASS"
            ),
            "direct_json_and_split_semantic_replay": "PASS",
            "pinned_full_component_verification_receipts": "PASS",
            "mixed_three_five_seed_scope": "PASS",
            "unchanged_patient_slide_folds": "PASS",
            "fit_and_concurrency_census": "PASS",
            "declared_not_run_and_unread_states": "PASS",
        },
    }


def verify_published_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Verify an existing FINAL-v10 receipt and every identity it seals."""

    selected = default_paths() if paths is None else paths
    if selected.destination.is_symlink() or not selected.destination.is_file():
        raise BundleVerificationError(
            "published FINAL-v10 receipt must be a regular non-symlink file"
        )
    published = _load_json(selected.destination, label="FINAL-v10 bundle receipt")
    created = published.get("created_utc")
    if not isinstance(created, str):
        raise BundleVerificationError("published receipt has no created_utc timestamp")
    try:
        parsed = dt.datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BundleVerificationError("published receipt created_utc is invalid") from exc
    if parsed.tzinfo is None:
        raise BundleVerificationError("published receipt created_utc must be timezone-aware")
    current = build_receipt(selected)
    if published != current or selected.destination.read_bytes() != _receipt_bytes(current):
        raise BundleVerificationError(
            "published FINAL-v10 receipt identity drift: bytes do not match the frozen "
            "candidate receipt"
        )
    return published


def _receipt_bytes(receipt: dict[str, Any]) -> bytes:
    return (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Validate and atomically publish the FINAL-v10 receipt exactly once."""

    selected = default_paths() if paths is None else paths
    if selected.destination.exists():
        raise BundleVerificationError(
            f"refusing to overwrite existing FINAL-v10 receipt: {selected.destination}"
        )
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
            file_stat = os.fstat(handle.fileno())
            temporary_inode = (file_stat.st_dev, file_stat.st_ino)
        try:
            os.link(temporary, selected.destination)
        except FileExistsError as exc:
            raise BundleVerificationError(
                f"refusing to overwrite existing FINAL-v10 receipt: {selected.destination}"
            ) from exc
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
                try:
                    selected.destination.unlink()
                except FileNotFoundError:
                    pass
                else:
                    # Preserve the original publication failure. The receipt
                    # itself is already gone, so a retry cannot authenticate
                    # poisoned bytes even if the cleanup fsync also fails.
                    with contextlib.suppress(OSError):
                        _fsync_directory(selected.destination.parent)
        # Best effort only: a private temporary file cannot be mistaken for the
        # exactly-once public receipt.
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--status", action="store_true", help="validate and print draft state")
    modes.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="authenticate available pending sources without sealing",
    )
    modes.add_argument(
        "--check-candidate",
        action="store_true",
        help="fully validate a candidate without publishing a receipt",
    )
    modes.add_argument(
        "--seal",
        action="store_true",
        help="validate and publish report_bundle_receipt.json exactly once",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.status:
            payload = draft_status()
        elif args.refresh_manifest:
            payload = refresh_manifest()
        elif args.check_candidate:
            payload = build_receipt()
        elif args.seal:
            payload = seal()
        else:
            payload = verify_published_receipt()
    except BundleVerificationError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
