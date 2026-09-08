#!/usr/bin/env python3
"""Verify and exactly-once seal the standalone FINAL-v12.1 report bundle.

FINAL-v12.1 is a fully integrated successor to sealed FINAL-v12.
This verifier recursively replays and pins FINAL-v12, requires exact reuse of
its 139-source ledger, directly rehashes every inherited and promoted source,
and checks that the new evidence is integrated into the complete standalone
paper-selection report rather than presented as an addendum.

Running the command without an action is read-only.  Only the explicit
``--seal`` action may create ``report_bundle_receipt.json``.  Publication is
deterministic, atomic, and exactly once: an existing file or symlink is never
overwritten.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FINAL_V12_1 = REPO / "reports" / "final_v12_1"
VERIFIER_CODE = Path(__file__).resolve()
VERIFIER_TEST = REPO / "tests" / "test_final_v12_1_bundle_receipt.py"
PARENT_DIR = REPO / "reports" / "final_v12"
PARENT_VERIFIER = REPO / "tools" / "final_v12_bundle_receipt.py"
PARENT_TEST = REPO / "tests" / "test_final_v12_bundle_receipt.py"

REPORT_DOCUMENTS = ("Experimental_Setup.md", "Results.md", "Audit.md")
SOURCE_MANIFEST_NAME = "source_manifest.json"
FINAL_RECEIPT_NAME = "report_bundle_receipt.json"

PARENT_SEALED_STATUS = "SEALED_FINAL_V12_INTEGRATED_PAPER_SELECTION"
PARENT_MANIFEST_STATUS = "sealed_final_v11_source_whitelist_reused_for_integrated_selection"
FINAL_MANIFEST_STATUS = "candidate_ready_for_final_v12_1_verification"
FINAL_SEALED_STATUS = "SEALED_FINAL_V12_1_INTEGRATED_PAPER_SELECTION"
EXPECTED_PARENT_SOURCE_COUNT = 139
GOVERNED_RESULTS_SOURCE_ID = "aim1-tcga-surgen-two-encoder-downstream-v3-results"
PROMOTED_RESULTS_SOURCE_ID = "aim2-refit-vs-fold5-results"

# Frozen production source IDs and document hashes. Synthetic tests override
# these values through BundlePaths.
_UNFROZEN = "REPLACE_AFTER_FINAL_V12_1_RECONCILIATION"
EXPECTED_EXTENSION_SOURCE_IDS = (
    "aim2-refit-vs-fold5-posthoc-pooled-results",
    "aim2-refit-vs-fold5-promotion-receipt",
    "aim2-refit-vs-fold5-results",
)

EXPECTED_PARENT_RECEIPT_SHA256 = "b5773af4fe5cb50269e391aaee04ec33f98f05548881c1aa042e4433d355fa38"
EXPECTED_PARENT_MANIFEST_SHA256 = "f9adb645a9dfbb7b887100cf2f9050f1971849103c481c0f5bfa7b27b477284e"
EXPECTED_PARENT_VERIFIER_SHA256 = "d536c1e246618e94a8a58d2d3039e0216b2c9bdbd43c7d3ad66406706235b1a9"
EXPECTED_PARENT_TEST_SHA256 = "ad8bbc92dccaab15882e9eca69e1508121a8ad916ac3e81829710c833c22bfda"
EXPECTED_PARENT_DOCUMENT_SHA256 = {
    "Experimental_Setup.md": "130f5a54e624e87424d5185d71a354d99344c569870cf720c771eb2773d7a117",
    "Results.md": "7f07d4bf576ee8256ead04c7dee01433d8da8b35f30cb2329e7d86fccf01bc61",
    "Audit.md": "f299c1eedf0ba477adf4d882589cf965f07d4ca596f222cb6e3a7129629da713",
}
EXPECTED_FINAL_DOCUMENT_SHA256 = {
    "Experimental_Setup.md": "3b9218a581f9277323a7e7cbd15280b8cb3831fa9073de0b0e0e235ca11e09f9",
    "Results.md": "9120c1f3ad64d3ff3279eb05f24dfd9362842ca06e8cfb48193a8e083170ddb7",
    "Audit.md": "eaea6be69e0cf8ff00c165cc1589f973cafd16622222a9be94165d75e677d8d3",
}

_MANIFEST_KEYS = {
    "schema_version",
    "bundle",
    "status",
    "artifacts",
    "pending_artifacts",
}
_SOURCE_KEYS = {"id", "aims", "experiments", "role", "path", "size_bytes", "sha256"}
_IDENTITY_KEYS = {"path", "size_bytes", "sha256"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SOURCE_ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_ROLE_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


class BundleVerificationError(RuntimeError):
    """A fail-closed FINAL-v12.1 parent, source, document, or receipt error."""


@dataclass(frozen=True)
class BundlePaths:
    """Filesystem locations and immutable parent expectations."""

    repo: Path
    final_v12_1: Path
    destination: Path
    verifier_code: Path
    verifier_test: Path
    parent_dir: Path
    parent_receipt: Path
    parent_manifest: Path
    parent_verifier: Path
    parent_test: Path
    expected_parent_receipt_sha256: str
    expected_parent_manifest_sha256: str
    expected_parent_verifier_sha256: str
    expected_parent_test_sha256: str
    expected_parent_document_sha256: Mapping[str, str]
    expected_final_document_sha256: Mapping[str, str]
    expected_extension_source_ids: Sequence[str]
    expected_parent_source_count: int = EXPECTED_PARENT_SOURCE_COUNT
    expected_parent_status: str = PARENT_SEALED_STATUS
    expected_parent_manifest_status: str = PARENT_MANIFEST_STATUS
    expected_final_manifest_status: str = FINAL_MANIFEST_STATUS
    replay_parent_verifier: bool = True


def default_paths() -> BundlePaths:
    """Return production FINAL-v12.1 and sealed FINAL-v12 paths."""

    return BundlePaths(
        repo=REPO,
        final_v12_1=FINAL_V12_1,
        destination=FINAL_V12_1 / FINAL_RECEIPT_NAME,
        verifier_code=VERIFIER_CODE,
        verifier_test=VERIFIER_TEST,
        parent_dir=PARENT_DIR,
        parent_receipt=PARENT_DIR / FINAL_RECEIPT_NAME,
        parent_manifest=PARENT_DIR / SOURCE_MANIFEST_NAME,
        parent_verifier=PARENT_VERIFIER,
        parent_test=PARENT_TEST,
        expected_parent_receipt_sha256=EXPECTED_PARENT_RECEIPT_SHA256,
        expected_parent_manifest_sha256=EXPECTED_PARENT_MANIFEST_SHA256,
        expected_parent_verifier_sha256=EXPECTED_PARENT_VERIFIER_SHA256,
        expected_parent_test_sha256=EXPECTED_PARENT_TEST_SHA256,
        expected_parent_document_sha256=dict(EXPECTED_PARENT_DOCUMENT_SHA256),
        expected_final_document_sha256=dict(EXPECTED_FINAL_DOCUMENT_SHA256),
        expected_extension_source_ids=EXPECTED_EXTENSION_SOURCE_IDS,
    )


def sha256_file(path: Path) -> str:
    """Hash a file in bounded-memory chunks."""

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


def _display(path: Path, repo: Path) -> str:
    absolute = path.absolute()
    try:
        return str(absolute.relative_to(repo.absolute()))
    except ValueError:
        return str(absolute)


def identity(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    """Return a stable identity after rejecting symlinks and concurrent drift."""

    _reject_symlink_chain(path, context="artifact path")
    if not path.is_file():
        raise BundleVerificationError(f"expected a regular file: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    before_key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_key != after_key:
        raise BundleVerificationError(f"file changed while it was being hashed: {path}")
    return {
        "path": str(path if display_path is None else display_path),
        "size_bytes": after.st_size,
        "sha256": digest,
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise BundleVerificationError(f"JSON contains duplicate key {key!r}")
        parsed[key] = value
    return parsed


def _reject_nonfinite_constant(value: str) -> None:
    raise BundleVerificationError(f"JSON contains non-finite constant {value}")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    """Load strict UTF-8 JSON, rejecting duplicate keys and NaN/Infinity."""

    _reject_symlink_chain(path, context=label)
    if not path.is_file():
        raise BundleVerificationError(f"{label} is missing: {path}")
    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleVerificationError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise BundleVerificationError(f"{label} must be a JSON object")
    return parsed


def _load_stable_json(
    path: Path,
    *,
    label: str,
    display_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strict-load JSON and prove its identity was stable around the read."""

    before = identity(path, display_path=display_path)
    parsed = _load_json(path, label=label)
    after = identity(path, display_path=display_path)
    if before != after:
        raise BundleVerificationError(f"{label} changed while it was being parsed")
    return parsed, after


def _require_exact_identity(
    record: Any,
    path: Path,
    *,
    display_path: str,
    context: str,
) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != _IDENTITY_KEYS:
        raise BundleVerificationError(f"{context} identity schema is not exact")
    actual = identity(path, display_path=display_path)
    if record != actual:
        raise BundleVerificationError(f"{context} identity drift")
    return actual


def _lexical_source_path(raw_path: str, paths: BundlePaths) -> Path:
    if "\x00" in raw_path:
        raise BundleVerificationError("source path contains a NUL byte")
    lexical = Path(raw_path)
    if not raw_path or any(part == ".." for part in lexical.parts):
        raise BundleVerificationError(f"source path is empty or traverses a parent: {raw_path!r}")
    return lexical if lexical.is_absolute() else paths.repo / lexical


def _validate_source_metadata(record: Any, *, index: int) -> dict[str, Any]:
    context = f"source record {index}"
    if not isinstance(record, dict) or set(record) != _SOURCE_KEYS:
        raise BundleVerificationError(f"{context} schema is not exact")
    source_id = record["id"]
    if not isinstance(source_id, str) or _SOURCE_ID_RE.fullmatch(source_id) is None:
        raise BundleVerificationError(f"{context} has an invalid id")
    role = record["role"]
    if not isinstance(role, str) or _ROLE_RE.fullmatch(role) is None:
        raise BundleVerificationError(f"{context} has an invalid role")
    for field in ("aims", "experiments"):
        values = record[field]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(values) != len(set(values))
        ):
            raise BundleVerificationError(f"{context} has an invalid {field} roster")
    raw_path = record["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise BundleVerificationError(f"{context} has an invalid path")
    size = record["size_bytes"]
    if type(size) is not int or size < 0:
        raise BundleVerificationError(f"{context} has an invalid byte size")
    digest = record["sha256"]
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise BundleVerificationError(f"{context} has an invalid SHA-256")
    return record


def _validate_source_roster(
    sources: Any,
    paths: BundlePaths,
    *,
    expected_count: int,
    rehash: bool,
) -> list[dict[str, Any]]:
    if not isinstance(sources, list) or len(sources) != expected_count:
        raise BundleVerificationError(
            f"source roster must contain exactly {expected_count} records"
        )
    validated = [
        _validate_source_metadata(record, index=index) for index, record in enumerate(sources)
    ]
    ids = [str(record["id"]) for record in validated]
    raw_paths = [str(record["path"]) for record in validated]
    if len(ids) != len(set(ids)):
        raise BundleVerificationError("source roster contains duplicate IDs")
    if len(raw_paths) != len(set(raw_paths)):
        raise BundleVerificationError("source roster contains duplicate paths")
    if ids != sorted(ids):
        raise BundleVerificationError("source roster is not sorted by ID")
    if rehash:
        for record in validated:
            source_path = _lexical_source_path(str(record["path"]), paths)
            actual = identity(source_path, display_path=str(record["path"]))
            if actual["size_bytes"] != record["size_bytes"] or actual["sha256"] != record["sha256"]:
                raise BundleVerificationError(f"source identity drift: {record['id']}")
    return validated


def _validate_parent(
    paths: BundlePaths,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Authenticate and, in production, recursively replay sealed FINAL-v12."""

    receipt, receipt_identity = _load_stable_json(
        paths.parent_receipt,
        label="sealed FINAL-v12 receipt",
        display_path=_display(paths.parent_receipt, paths.repo),
    )
    if receipt_identity["sha256"] != paths.expected_parent_receipt_sha256:
        raise BundleVerificationError("sealed FINAL-v12 receipt SHA-256 drift")
    if receipt.get("schema_version") != 1 or receipt.get("bundle") != "reports/final_v12":
        raise BundleVerificationError("sealed FINAL-v12 receipt identity fields are invalid")
    if receipt.get("status") != paths.expected_parent_status:
        raise BundleVerificationError("sealed FINAL-v12 receipt status is not authoritative")
    checks = receipt.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(value != "PASS" for value in checks.values())
    ):
        raise BundleVerificationError("sealed FINAL-v12 receipt does not have all checks PASS")

    if paths.replay_parent_verifier:
        verifier_identity = identity(
            paths.parent_verifier,
            display_path=_display(paths.parent_verifier, paths.repo),
        )
        test_identity = identity(
            paths.parent_test,
            display_path=_display(paths.parent_test, paths.repo),
        )
        if verifier_identity["sha256"] != paths.expected_parent_verifier_sha256:
            raise BundleVerificationError("frozen FINAL-v12 verifier SHA-256 drift")
        if test_identity["sha256"] != paths.expected_parent_test_sha256:
            raise BundleVerificationError("frozen FINAL-v12 verifier test SHA-256 drift")
        from tools import final_v12_bundle_receipt as parent

        replayed = parent.verify_published_receipt(parent.default_paths())
        if replayed != receipt:
            raise BundleVerificationError("recursive FINAL-v12 receipt replay differs")
    else:
        verifier_identity = {
            "path": _display(paths.parent_verifier, paths.repo),
            "size_bytes": 0,
            "sha256": paths.expected_parent_verifier_sha256,
        }
        test_identity = {
            "path": _display(paths.parent_test, paths.repo),
            "size_bytes": 0,
            "sha256": paths.expected_parent_test_sha256,
        }

    parent_manifest, manifest_identity = _load_stable_json(
        paths.parent_manifest,
        label="sealed FINAL-v12 source manifest",
        display_path=_display(paths.parent_manifest, paths.repo),
    )
    if manifest_identity["sha256"] != paths.expected_parent_manifest_sha256:
        raise BundleVerificationError("sealed FINAL-v12 manifest SHA-256 drift")
    _require_exact_identity(
        receipt.get("source_manifest"),
        paths.parent_manifest,
        display_path=_display(paths.parent_manifest, paths.repo),
        context="sealed FINAL-v12 receipt source_manifest",
    )
    if set(parent_manifest) != _MANIFEST_KEYS:
        raise BundleVerificationError("sealed FINAL-v12 source manifest schema is not exact")
    if parent_manifest["schema_version"] != 2 or parent_manifest["bundle"] != "final_v12":
        raise BundleVerificationError("sealed FINAL-v12 source manifest identity is invalid")
    if parent_manifest["status"] != paths.expected_parent_manifest_status:
        raise BundleVerificationError("sealed FINAL-v12 source manifest status drift")
    if parent_manifest["pending_artifacts"] != []:
        raise BundleVerificationError("sealed FINAL-v12 source manifest has pending artifacts")
    parent_sources = _validate_source_roster(
        parent_manifest["artifacts"],
        paths,
        expected_count=paths.expected_parent_source_count,
        rehash=False,
    )
    if receipt.get("authoritative_sources") != parent_sources:
        raise BundleVerificationError(
            "sealed FINAL-v12 receipt and source-manifest artifact rosters differ"
        )

    expected_document_pins = dict(paths.expected_parent_document_sha256)
    if set(expected_document_pins) != set(REPORT_DOCUMENTS):
        raise BundleVerificationError("sealed FINAL-v12 document pin roster is not exact")
    receipt_documents = receipt.get("documents")
    if not isinstance(receipt_documents, dict) or set(receipt_documents) != set(REPORT_DOCUMENTS):
        raise BundleVerificationError("sealed FINAL-v12 receipt document roster is not exact")
    parent_documents: dict[str, Any] = {}
    for name in REPORT_DOCUMENTS:
        document_path = paths.parent_dir / name
        actual = _require_exact_identity(
            receipt_documents[name],
            document_path,
            display_path=_display(document_path, paths.repo),
            context=f"sealed FINAL-v12 {name}",
        )
        if actual["sha256"] != expected_document_pins[name]:
            raise BundleVerificationError(f"sealed FINAL-v12 {name} SHA-256 drift")
        parent_documents[name] = actual

    return (
        receipt,
        parent_manifest,
        parent_sources,
        {
            "receipt": receipt_identity,
            "source_manifest": manifest_identity,
            "documents": parent_documents,
            "verifier": verifier_identity,
            "verifier_test": test_identity,
        },
    )


def _validate_final_manifest(
    paths: BundlePaths,
    parent_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = paths.final_v12_1 / SOURCE_MANIFEST_NAME
    manifest, manifest_identity = _load_stable_json(
        manifest_path,
        label="FINAL-v12.1 source manifest",
        display_path=_display(manifest_path, paths.repo),
    )
    if set(manifest) != _MANIFEST_KEYS:
        raise BundleVerificationError("FINAL-v12.1 source manifest schema is not exact")
    if manifest["schema_version"] != 2 or manifest["bundle"] != "final_v12_1":
        raise BundleVerificationError("FINAL-v12.1 source manifest identity is invalid")
    if manifest["status"] != paths.expected_final_manifest_status:
        raise BundleVerificationError("FINAL-v12.1 source manifest status drift")
    if manifest["pending_artifacts"] != []:
        raise BundleVerificationError("FINAL-v12.1 source manifest has pending artifacts")

    expected_extension_ids = tuple(paths.expected_extension_source_ids)
    if (
        not expected_extension_ids
        or any(
            not isinstance(source_id, str)
            or _SOURCE_ID_RE.fullmatch(source_id) is None
            for source_id in expected_extension_ids
        )
        or len(expected_extension_ids) != len(set(expected_extension_ids))
        or tuple(sorted(expected_extension_ids)) != expected_extension_ids
        or _UNFROZEN in expected_extension_ids
    ):
        raise BundleVerificationError("FINAL-v12.1 promoted source-ID roster is not frozen")

    expected_count = paths.expected_parent_source_count + len(expected_extension_ids)
    sources = _validate_source_roster(
        manifest["artifacts"],
        paths,
        expected_count=expected_count,
        rehash=True,
    )
    parent_sources = parent_manifest["artifacts"]
    parent_by_id = {str(record["id"]): record for record in parent_sources}
    source_by_id = {str(record["id"]): record for record in sources}
    if set(parent_by_id) & set(expected_extension_ids):
        raise BundleVerificationError("FINAL-v12.1 promoted source ID collides with its parent")
    if set(source_by_id) != set(parent_by_id) | set(expected_extension_ids):
        raise BundleVerificationError("FINAL-v12.1 source roster is not parent plus exact extension")
    for source_id, parent_record in parent_by_id.items():
        if source_by_id[source_id] != parent_record:
            raise BundleVerificationError(
                f"FINAL-v12.1 inherited source record drift: {source_id}"
            )
    return manifest, sources, manifest_identity


def _require_one(pattern: str, text: str, *, context: str, flags: int = 0) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags))
    if len(matches) != 1:
        raise BundleVerificationError(f"Results.md must contain exactly one {context}")
    return matches[0]


def _require_ordered_patterns(
    text: str,
    requirements: Sequence[tuple[str, str]],
    *,
    context: str,
) -> None:
    positions: list[int] = []
    for label, pattern in requirements:
        match = _require_one(pattern, text, context=label, flags=re.MULTILINE | re.IGNORECASE)
        positions.append(match.start())
    if positions != sorted(positions):
        raise BundleVerificationError(f"Results.md {context} section order is invalid")


_AIM1_HEADINGS = (
    ("Aim-1 E0 heading", r"^### E0\s+—"),
    ("Aim-1 E1a heading", r"^### E1a\s+—"),
    ("Aim-1 E1a-S heading", r"^### E1a-S\s+—"),
    ("Aim-1 Why-D heading", r"^### Why-D$"),
    ("Aim-1 E1d heading", r"^### E1d\s+—"),
    ("Aim-1 E1v heading", r"^### E1v\b"),
    ("Aim-1 E1e heading", r"^### E1e\s+—"),
    ("Aim-1 worklist heading", r"^### Worklist enrichment\b"),
    ("Aim-1 decision-curve heading", r"^### Decision-curve analysis$"),
    ("Aim-1 RAS/MAPK heading", r"^### Extended-RAS, MAPK\b"),
)
_AIM2_HEADINGS = (
    ("Aim-2 acquisition heading", r"^### Label-blind acquisition and composition evidence$"),
    ("Aim-2 source scoring heading", r"^### TCGA\+SurGen source-restricted target scoring$"),
    ("Aim-2 E2a-F heading", r"^### E2a-F\s+—"),
    ("Aim-2 E2a-D heading", r"^### E2a-D\s+—"),
    ("Aim-2 between-slide heading", r"^### Between-slide sampling\b"),
    ("Aim-2 E2-MET heading", r"^### E2-MET\s+—"),
    ("Aim-2 E2c heading", r"^### E2c\s+—"),
    ("Aim-2 E2d family heading", r"^### E2d1[–-]E2d6\s+—"),
    ("Aim-2 E2e heading", r"^### E2e\s+—"),
    ("Aim-2 E2f-v3 heading", r"^### E2f-v3\s+—"),
    ("Aim-2 E2-CPHT heading", r"^### E2-CPHT\s+—"),
    ("Aim-2 E2-CPHT-A heading", r"^### E2-CPHT-A\s+—"),
    ("Aim-2 E2-CPHT-R heading", r"^### E2-CPHT-R\s+—"),
)
_AIM3_HEADINGS = (
    ("Aim-3 E3 fixed heading", r"^### E3 fixed\b"),
    ("Aim-3 E3 repeated heading", r"^### E3 repeated\b"),
    ("Aim-3 E3v heading", r"^### E3v\b"),
)
_AIM4_HEADINGS = (
    ("Aim-4 E4 atlas heading", r"^### E4 canonical\b"),
    ("Aim-4 E4 transport heading", r"^### E4 primary-to-metastatic\b"),
    ("Aim-4 E4 vocabulary heading", r"^### E4 vocabulary-size\b"),
    ("Aim-4 E4 compressibility heading", r"^### E4 score compressibility$"),
    ("Aim-4 E4 montage heading", r"^### E4 adopted montage\b"),
    ("Aim-4 E4 pathway heading", r"^### E4 pathway-context weld$"),
    ("Aim-4 whole-section heading", r"^### Whole-section pathology state$"),
)

_STALE_PUBLICATION_PATTERNS = (
    ("candidate staging declaration", r"\b(?:the|this) candidate\b"),
    ("candidate manifest declaration", r"\b(?:flat )?candidate manifest\b"),
    ("candidate-ready status", r"\bcandidate[_ -]ready\b"),
    (
        "unpublished parent receipt declaration",
        r"\bno\s+FINAL[- ]v(?:11|12) receipt has been published\b",
    ),
    (
        "unpublished parent receipt declaration",
        r"\b(?:unpublished\s+FINAL[- ]v(?:11|12) receipt|FINAL[- ]v(?:11|12) receipt.{0,80}unpublished)\b",
    ),
    (
        "awaiting publication declaration",
        r"\bawait(?:ing)?\b.{0,80}\b(?:publication|verification|seal(?:ing)?)\b",
    ),
)


def _aim_blocks(results: str) -> dict[int, str]:
    matches: list[tuple[int, re.Match[str]]] = []
    for aim in range(1, 5):
        match = _require_one(
            rf"^## Aim {aim}(?:\s+—.*)?$",
            results,
            context=f"Aim {aim} section",
            flags=re.MULTILINE,
        )
        matches.append((aim, match))
    positions = [match.start() for _, match in matches]
    if positions != sorted(positions):
        raise BundleVerificationError("Results.md Aim 1--4 sections are out of order")
    blocks: dict[int, str] = {}
    for index, (aim, match) in enumerate(matches):
        end = matches[index + 1][1].start() if index + 1 < len(matches) else len(results)
        blocks[aim] = results[match.start() : end]
    return blocks


def _validate_results(results: str) -> None:
    lines = results.splitlines()
    expected_title = "# FINAL-v12.1 integrated results for paper selection"
    if not lines or lines[0] != expected_title:
        raise BundleVerificationError("Results.md integrated-results title is not exact")
    h1_lines = [line for line in lines if re.fullmatch(r"#\s+.+", line)]
    if h1_lines != [expected_title]:
        raise BundleVerificationError("Results.md must be one standalone report, with one H1")
    if re.search(r"(?im)^#{1,6}\s+.*\baddend(?:um|a)\b", results):
        raise BundleVerificationError("Results.md uses addendum-form heading topology")
    if re.search(
        r"(?im)^\s*(?:this report|FINAL[- ]v12(?:\.1)?)\s+(?:is\s+)?an?\s+addendum\b",
        results,
    ):
        raise BundleVerificationError("Results.md declares itself to be an addendum")

    blocks = _aim_blocks(results)
    _require_ordered_patterns(blocks[1], _AIM1_HEADINGS, context="Aim 1")
    _require_ordered_patterns(blocks[2], _AIM2_HEADINGS, context="Aim 2")
    _require_ordered_patterns(blocks[3], _AIM3_HEADINGS, context="Aim 3")
    _require_ordered_patterns(blocks[4], _AIM4_HEADINGS, context="Aim 4")

    _require_ordered_patterns(
        blocks[1],
        (
            ("E0 Design A heading", r"^#### Design A: canonical ALL-primary OOF model\b"),
            (
                "E0 Design B heading",
                r"^#### Design B: TCGA\+SurGen source-restricted OOF training$",
            ),
            ("E0 Design C heading", r"^#### Design C: separately trained within-source OOF arms$"),
        ),
        context="E0 training-design",
    )
    _require_ordered_patterns(
        blocks[1],
        (
            (
                "E1a ALL-primary population panel",
                r"^#### ALL-primary inherited challenge panel$",
            ),
            (
                "E1a TCGA+SurGen population panel",
                r"^#### TCGA\+SurGen two-encoder challenge panel$",
            ),
            (
                "E1a-S TCGA+SurGen population panel",
                r"^#### TCGA\+SurGen two-encoder standardization$",
            ),
            (
                "E1a-S ALL-primary population panel",
                r"^#### ALL-primary inherited standardization$",
            ),
            ("Why-D A panel", r"^#### Why-D A\s+—"),
            ("Why-D B panel", r"^#### Why-D B\s+—"),
            ("Why-D C panel", r"^#### Why-D C\s+—"),
            ("Why-D D panel", r"^#### Why-D D\s+—"),
        ),
        context="Aim-1 population/diagnostic",
    )

    for label in ("ALL primary", "TCGA", "SurGen", "TCGA+SurGen"):
        pattern = rf"^\| \*\*{re.escape(label)}\*\* \|"
        _require_one(
            pattern,
            results,
            context=f"cohort-taxonomy row {label}",
            flags=re.MULTILINE,
        )
    population_bindings = (
        "aim1.canonical_e0.pooled_all_primary.",
        "aim1.canonical_e0.tcga.",
        "aim1.canonical_e0.surgen.",
        "aim1.canonical_e0.tcga_surgen.",
        "aim1.source_restricted_e0.pooled_source.",
        "aim1.source_restricted_e0.tcga.",
        "aim1.source_restricted_e0.surgen.",
        "aim1.source_restricted_e0.tcga_surgen.",
    )
    for binding in population_bindings:
        if binding not in blocks[1]:
            raise BundleVerificationError(
                f"Results.md lacks required E0 population binding: {binding}"
            )
    for within_source_row in (
        r"^\| TCGA \| TCGA \| univ1 \|",
        r"^\| SurGen \| SurGen \| univ1 \|",
        r"^\| TCGA\+SurGen \| TCGA\+SurGen \| univ1 \|",
    ):
        if re.search(within_source_row, blocks[1], re.MULTILINE) is None:
            raise BundleVerificationError(
                "Results.md lacks a required within-source E0 population row"
            )

    _require_one(
        r"^### Cohort taxonomy used throughout$",
        results,
        context="cohort taxonomy section",
        flags=re.MULTILINE,
    )
    _require_one(
        r"^### Evidence tiers and inclusion rule$",
        results,
        context="evidence-tier and inclusion section",
        flags=re.MULTILINE,
    )
    matrix_match = _require_one(
        r"^## Integrated experiment and population availability matrix$",
        results,
        context="paper-selection availability matrix",
        flags=re.MULTILINE,
    )
    next_h2 = re.search(r"^## ", results[matrix_match.end() :], re.MULTILINE)
    matrix_end = matrix_match.end() + next_h2.start() if next_h2 else len(results)
    matrix = results[matrix_match.start() : matrix_end]
    if "paper selection" not in matrix.lower():
        raise BundleVerificationError("Results.md availability matrix lacks paper-selection scope")
    for aim in range(1, 5):
        if re.search(rf"^\| {aim} \|", matrix, re.MULTILINE) is None:
            raise BundleVerificationError(f"Results.md selection matrix lacks Aim {aim}")
    for marker in ("E0 canonical", "E1a", "E2d1–E2d6", "E3 fixed", "E4 atlas"):
        if marker not in matrix:
            raise BundleVerificationError(f"Results.md selection matrix lacks {marker}")
    _require_one(
        r"^## Paper-building candidate map$",
        results,
        context="paper-building candidate map",
        flags=re.MULTILINE,
    )

    e2d_positions: list[int] = []
    for index in range(1, 7):
        match = _require_one(
            rf"^#### E2d{index}\s+—",
            blocks[2],
            context=f"explicit E2d{index} section",
            flags=re.MULTILINE | re.IGNORECASE,
        )
        e2d_positions.append(match.start())
    if e2d_positions != sorted(e2d_positions):
        raise BundleVerificationError("Results.md E2d1--E2d6 sections are out of order")

    if "Excluded from selectable results" not in results:
        raise BundleVerificationError("Results.md lacks the selectable-result exclusion rule")
    exclusion_match = _require_one(
        r"^## (?:Excluded|Historical).*(?:superseded|unavailable).*ledger$",
        results,
        context="exclusion/supersession ledger",
        flags=re.MULTILINE | re.IGNORECASE,
    )
    exclusion_text = results[exclusion_match.start() :]
    for marker in ("NOT_RUN", "GENERATED_UNREAD", "audit-only", "supersed"):
        if marker.lower() not in exclusion_text.lower():
            raise BundleVerificationError(f"Results.md exclusion ledger lacks {marker}")

    comparison_match = _require_one(
        r"^#### Per-seed refit versus within-seed five-fold ensemble$",
        blocks[2],
        context="refit-versus-five-fold-ensemble section",
        flags=re.MULTILINE,
    )
    next_heading = re.search(r"^#{3,4} ", blocks[2][comparison_match.end() :], re.MULTILINE)
    comparison_end = (
        comparison_match.end() + next_heading.start()
        if next_heading is not None
        else len(blocks[2])
    )
    comparison = blocks[2][comparison_match.start() : comparison_end]
    normalized_comparison = " ".join(comparison.split())
    for marker in (
        "Refit AUROCs",
        "Refit mean ± SD",
        "Ensemble AUROCs",
        "Ensemble mean ± SD",
        "RIH-Pri + CPTAC",
        "All-Met",
        "sample SD",
        "patient-pooled",
        "not confidence intervals",
    ):
        if marker.lower() not in normalized_comparison.lower():
            raise BundleVerificationError(
                f"Results.md refit-versus-five-fold section lacks {marker}"
            )
    for encoder in ("UNI-v1", "Virchow2-CLS"):
        if encoder not in comparison:
            raise BundleVerificationError(
                f"Results.md refit-versus-five-fold section lacks {encoder}"
            )


def _validate_governed_record_bindings(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
    results: str,
) -> None:
    """Require every governed 78/48/2 report identifier exactly once."""

    matches = [source for source in sources if source["id"] == GOVERNED_RESULTS_SOURCE_ID]
    if not matches:
        # Small synthetic unit-test manifests do not model the scientific graph.
        if paths.expected_parent_source_count == EXPECTED_PARENT_SOURCE_COUNT:
            raise BundleVerificationError("governed analysis result source is missing")
        return
    if len(matches) != 1:
        raise BundleVerificationError("governed analysis result source is not unique")
    governed_path = _lexical_source_path(str(matches[0]["path"]), paths)
    governed = _load_json(governed_path, label="governed 78/48/2 analysis result")
    expected_groups = (
        ("report_claim_rows", 78, "row_id"),
        ("report_contrast_rows", 48, "row_id"),
        ("why_d_evidence_records", 2, "record_id"),
    )
    record_ids: list[str] = []
    for field, expected_count, id_field in expected_groups:
        records = governed.get(field)
        if not isinstance(records, list) or len(records) != expected_count:
            raise BundleVerificationError(
                f"governed analysis {field} must contain exactly {expected_count} records"
            )
        for record in records:
            if not isinstance(record, dict):
                raise BundleVerificationError(f"governed analysis {field} has a non-object record")
            record_id = record.get(id_field)
            if not isinstance(record_id, str) or not record_id:
                raise BundleVerificationError(f"governed analysis {field} has an invalid ID")
            record_ids.append(record_id)
    if len(record_ids) != len(set(record_ids)):
        raise BundleVerificationError("governed 78/48/2 analysis IDs are not unique")
    incorrect = [(record_id, results.count(record_id)) for record_id in record_ids]
    incorrect = [(record_id, count) for record_id, count in incorrect if count != 1]
    if incorrect:
        preview = ", ".join(f"{record_id}={count}" for record_id, count in incorrect[:5])
        raise BundleVerificationError(
            f"Results.md governed 78/48/2 IDs are not exact-once bound: {preview}"
        )


def _promoted_source_json(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
    source_id: str,
) -> dict[str, Any]:
    matches = [source for source in sources if source["id"] == source_id]
    if len(matches) != 1:
        raise BundleVerificationError(f"promoted source is not unique: {source_id}")
    return _load_json(
        _lexical_source_path(str(matches[0]["path"]), paths),
        label=f"promoted source {source_id}",
    )


def _require_finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleVerificationError(f"{context} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise BundleVerificationError(f"{context} must be finite")
    return parsed


def _strategy_summary(
    strategy: Mapping[str, Any],
    *,
    context: str,
) -> tuple[list[float], float, float]:
    per_seed = strategy.get("per_seed_descriptive")
    expected_seed_keys = ("42", "43", "44", "45", "46")
    if not isinstance(per_seed, dict) or tuple(per_seed) != expected_seed_keys:
        raise BundleVerificationError(f"{context} seed roster is not exact")
    values: list[float] = []
    for seed in expected_seed_keys:
        record = per_seed[seed]
        if not isinstance(record, dict):
            raise BundleVerificationError(f"{context} seed {seed} record is invalid")
        values.append(
            _require_finite_number(record.get("auroc"), context=f"{context} seed {seed} AUROC")
        )
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return values, mean, sd


def _pooled_strategy_summary(
    strategy: Mapping[str, Any],
    *,
    context: str,
) -> tuple[list[float], float, float]:
    per_seed = strategy.get("per_seed_auroc")
    expected_seed_keys = ("42", "43", "44", "45", "46")
    if not isinstance(per_seed, dict) or tuple(per_seed) != expected_seed_keys:
        raise BundleVerificationError(f"{context} pooled seed roster is not exact")
    values = [
        _require_finite_number(per_seed[seed], context=f"{context} pooled seed {seed} AUROC")
        for seed in expected_seed_keys
    ]
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    declared_mean = _require_finite_number(
        strategy.get("mean_per_seed_auroc"), context=f"{context} declared mean"
    )
    declared_sd = _require_finite_number(
        strategy.get("sample_sd_per_seed_auroc_ddof1"), context=f"{context} declared SD"
    )
    if not math.isclose(mean, declared_mean, rel_tol=0.0, abs_tol=1e-15):
        raise BundleVerificationError(f"{context} pooled mean does not replay")
    if not math.isclose(sd, declared_sd, rel_tol=0.0, abs_tol=1e-15):
        raise BundleVerificationError(f"{context} pooled sample SD does not replay")
    return values, mean, sd


def _parse_auroc_vector(cell: str, *, context: str) -> list[float]:
    cleaned = cell.replace("**", "").replace("`", "").strip()
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != 5:
        raise BundleVerificationError(f"{context} must contain five AUROCs")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise BundleVerificationError(f"{context} contains a nonnumeric AUROC") from exc
    if any(not math.isfinite(value) for value in values):
        raise BundleVerificationError(f"{context} contains a nonfinite AUROC")
    return values


def _parse_mean_sd(cell: str, *, context: str) -> tuple[float, float]:
    cleaned = cell.replace("**", "").replace("`", "").strip()
    parts = [part.strip() for part in cleaned.split("±")]
    if len(parts) != 2:
        raise BundleVerificationError(f"{context} must use mean ± SD")
    try:
        values = (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise BundleVerificationError(f"{context} contains a nonnumeric mean or SD") from exc
    if any(not math.isfinite(value) for value in values):
        raise BundleVerificationError(f"{context} contains a nonfinite mean or SD")
    return values


def _validate_promoted_table_bindings(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
    results_text: str,
) -> None:
    """Replay promoted summaries and bind all 28 strategy/population table cells."""

    required_source_ids = {
        PROMOTED_RESULTS_SOURCE_ID,
        "aim2-refit-vs-fold5-posthoc-pooled-results",
        "aim2-refit-vs-fold5-promotion-receipt",
    }
    observed_source_ids = {str(source["id"]) for source in sources}
    if not required_source_ids <= observed_source_ids:
        if paths.expected_parent_source_count == EXPECTED_PARENT_SOURCE_COUNT:
            raise BundleVerificationError("promoted FINAL-v12.1 source trio is incomplete")
        return

    primitive = _promoted_source_json(paths, sources, PROMOTED_RESULTS_SOURCE_ID)
    pooled = _promoted_source_json(
        paths,
        sources,
        "aim2-refit-vs-fold5-posthoc-pooled-results",
    )
    promotion = _promoted_source_json(
        paths,
        sources,
        "aim2-refit-vs-fold5-promotion-receipt",
    )
    if (
        primitive.get("schema_version") != 1
        or primitive.get("status") != "complete"
        or primitive.get("experiment")
        != "aim2_tcga_surgen_refit5_vs_hierarchical_fold5_zero_shot_scratch"
    ):
        raise BundleVerificationError("promoted primitive result identity is invalid")
    semantics = primitive.get("strategy_semantics")
    if (
        not isinstance(semantics, dict)
        or semantics.get("folds_or_seeds_as_n") is not False
        or semantics.get("patient_unit") is not True
        or semantics.get("target_training_selection_or_calibration") is not False
    ):
        raise BundleVerificationError("promoted primitive strategy boundary is invalid")
    if (
        pooled.get("schema_version") != 1
        or pooled.get("status") != "complete"
        or pooled.get("new_scoring_for_this_derivation") != 0
        or pooled.get("new_training_fits") != 0
        or pooled.get("patient_is_inference_unit") is not True
        or pooled.get("folds_or_seeds_are_inference_units") is not False
        or pooled.get("scoring_process_was_label_blind") is not True
        or pooled.get("not_retrospectively_preregistered") is not True
    ):
        raise BundleVerificationError("promoted pooled-result boundary is invalid")
    seed_summary = pooled.get("seed_summary")
    if (
        not isinstance(seed_summary, dict)
        or seed_summary.get("seeds") != [42, 43, 44, 45, 46]
        or seed_summary.get("ddof") != 1
    ):
        raise BundleVerificationError("promoted pooled seed summary is invalid")
    boundaries = promotion.get("boundaries")
    if (
        promotion.get("schema_version") != 1
        or promotion.get("status") != "complete"
        or promotion.get("source_files_all_present_and_byte_identical") is not True
        or promotion.get("source_root_left_unmodified") is not True
        or not isinstance(boundaries, dict)
        or boundaries.get("new_scoring_during_promotion") != 0
        or boundaries.get("new_training_fits") != 0
        or boundaries.get("primitive_results_modified") is not False
        or boundaries.get("posthoc_pooled_rows_are_descriptive_only") is not True
    ):
        raise BundleVerificationError("promoted receipt boundary is invalid")

    targets = primitive.get("targets")
    target_map = {
        "RIH-Pri": "rih_primary",
        "CPTAC": "cptac_primary",
        "RIH-Met": "rih_metastatic",
        "SurGen-Met (SR1482-M)": "sr1482_metastatic",
        "Orion": "orion_cpht",
    }
    if not isinstance(targets, dict) or set(targets) != set(target_map.values()):
        raise BundleVerificationError("promoted primitive target roster is invalid")
    pooled_records = pooled.get("pools")
    if not isinstance(pooled_records, dict) or set(pooled_records) != {
        "pooled_primary",
        "pooled_metastatic",
    }:
        raise BundleVerificationError("promoted pooled target roster is invalid")

    heading = _require_one(
        r"^#### Per-seed refit versus within-seed five-fold ensemble$",
        results_text,
        context="promoted refit-versus-five-fold section",
        flags=re.MULTILINE,
    )
    next_h4_or_h3 = re.search(r"^#{3,4} ", results_text[heading.end() :], re.MULTILINE)
    end = heading.end() + next_h4_or_h3.start() if next_h4_or_h3 else len(results_text)
    section = results_text[heading.start() : end]
    encoder_headings = list(
        re.finditer(
            r"^(?:##### |\*\*)(UNI-v1|Virchow2-CLS)(?:\*\*)?$",
            section,
            flags=re.MULTILINE,
        )
    )
    if [match.group(1) for match in encoder_headings] != ["UNI-v1", "Virchow2-CLS"]:
        raise BundleVerificationError("promoted comparison encoder panels are not exact")

    expected_by_encoder: dict[str, dict[str, tuple[list[float], float, float, list[float], float, float]]] = {}
    for encoder_key, encoder_display in (("univ1", "UNI-v1"), ("virchow2_cls", "Virchow2-CLS")):
        expected_rows: dict[
            str,
            tuple[list[float], float, float, list[float], float, float],
        ] = {}
        for display, target_id in target_map.items():
            target = targets[target_id]
            encoders = target.get("encoders") if isinstance(target, dict) else None
            encoder = encoders.get(encoder_key) if isinstance(encoders, dict) else None
            strategies = encoder.get("strategies") if isinstance(encoder, dict) else None
            if not isinstance(strategies, dict):
                raise BundleVerificationError(
                    f"promoted strategy record is missing: {target_id}/{encoder_key}"
                )
            refit_values, refit_mean, refit_sd = _strategy_summary(
                strategies["refit5"], context=f"{target_id}/{encoder_key}/refit5"
            )
            fold_values, fold_mean, fold_sd = _strategy_summary(
                strategies["fold25_hierarchical"],
                context=f"{target_id}/{encoder_key}/fold25_hierarchical",
            )
            expected_rows[display] = (
                refit_values,
                refit_mean,
                refit_sd,
                fold_values,
                fold_mean,
                fold_sd,
            )
        for display, pool_id in (
            ("RIH-Pri + CPTAC", "pooled_primary"),
            ("All-Met", "pooled_metastatic"),
        ):
            pool = pooled_records[pool_id]
            encoders = pool.get("encoders") if isinstance(pool, dict) else None
            encoder = encoders.get(encoder_key) if isinstance(encoders, dict) else None
            if not isinstance(encoder, dict):
                raise BundleVerificationError(
                    f"promoted pooled strategy record is missing: {pool_id}/{encoder_key}"
                )
            refit_values, refit_mean, refit_sd = _pooled_strategy_summary(
                encoder["full_source_single_refit"],
                context=f"{pool_id}/{encoder_key}/refit",
            )
            fold_values, fold_mean, fold_sd = _pooled_strategy_summary(
                encoder["within_seed_five_fold_checkpoint_ensemble"],
                context=f"{pool_id}/{encoder_key}/fold5",
            )
            expected_rows[display] = (
                refit_values,
                refit_mean,
                refit_sd,
                fold_values,
                fold_mean,
                fold_sd,
            )
        expected_by_encoder[encoder_display] = expected_rows

    for index, heading_match in enumerate(encoder_headings):
        encoder_display = heading_match.group(1)
        panel_end = (
            encoder_headings[index + 1].start()
            if index + 1 < len(encoder_headings)
            else len(section)
        )
        panel = section[heading_match.end() : panel_end]
        observed_rows: dict[str, list[str]] = {}
        for line in panel.splitlines():
            if not line.startswith("|") or line.startswith("|---"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not cells or "population" in cells[0].lower():
                continue
            if len(cells) == 6:
                value_cells = cells[2:]
            elif len(cells) == 5:
                value_cells = cells[1:]
            else:
                continue
            label = cells[0].replace("**", "").replace("†", "").strip()
            if label in observed_rows:
                raise BundleVerificationError(
                    f"promoted comparison table duplicates {encoder_display}/{label}"
                )
            observed_rows[label] = value_cells
        expected_rows = expected_by_encoder[encoder_display]
        if set(observed_rows) != set(expected_rows):
            raise BundleVerificationError(
                f"promoted comparison row roster is invalid for {encoder_display}"
            )
        for label, expected in expected_rows.items():
            cells = observed_rows[label]
            observed_refit = _parse_auroc_vector(
                cells[0], context=f"{encoder_display}/{label} refit vector"
            )
            observed_refit_mean, observed_refit_sd = _parse_mean_sd(
                cells[1], context=f"{encoder_display}/{label} refit summary"
            )
            observed_fold = _parse_auroc_vector(
                cells[2], context=f"{encoder_display}/{label} ensemble vector"
            )
            observed_fold_mean, observed_fold_sd = _parse_mean_sd(
                cells[3], context=f"{encoder_display}/{label} ensemble summary"
            )
            observed = (
                *observed_refit,
                observed_refit_mean,
                observed_refit_sd,
                *observed_fold,
                observed_fold_mean,
                observed_fold_sd,
            )
            expected_flat = (
                *expected[0],
                expected[1],
                expected[2],
                *expected[3],
                expected[4],
                expected[5],
            )
            if any(
                not math.isclose(observed_value, expected_value, rel_tol=0.0, abs_tol=5e-5)
                for observed_value, expected_value in zip(observed, expected_flat, strict=True)
            ):
                raise BundleVerificationError(
                    f"promoted comparison numeric drift: {encoder_display}/{label}"
                )


def _validate_document_topology(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    texts: dict[str, str] = {}
    identities: dict[str, Any] = {}
    expected_titles = {
        "Experimental_Setup.md": "# FINAL-v12.1 integrated experimental setup",
        "Results.md": "# FINAL-v12.1 integrated results for paper selection",
        "Audit.md": "# FINAL-v12.1 evidence audit and selection record",
    }
    document_pins = dict(paths.expected_final_document_sha256)
    if (
        set(document_pins) != set(REPORT_DOCUMENTS)
        or any(
            not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or digest == _UNFROZEN
            for digest in document_pins.values()
        )
    ):
        raise BundleVerificationError("FINAL-v12.1 document hash roster is not frozen")
    for name in REPORT_DOCUMENTS:
        document_path = paths.final_v12_1 / name
        document_identity = identity(
            document_path,
            display_path=_display(document_path, paths.repo),
        )
        try:
            text = document_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise BundleVerificationError(f"FINAL-v12.1 {name} is not readable UTF-8: {exc}") from exc
        if not text.startswith(expected_titles[name] + "\n"):
            raise BundleVerificationError(f"FINAL-v12.1 {name} has an invalid title")
        if (
            identity(
                document_path,
                display_path=_display(document_path, paths.repo),
            )
            != document_identity
        ):
            raise BundleVerificationError(f"FINAL-v12.1 {name} changed while it was being read")
        texts[name] = text
        identities[name] = document_identity

    for name, text in texts.items():
        for label, pattern in _STALE_PUBLICATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                raise BundleVerificationError(f"{name} retains stale {label}")

    _validate_results(texts["Results.md"])
    _validate_governed_record_bindings(paths, sources, texts["Results.md"])
    _validate_promoted_table_bindings(paths, sources, texts["Results.md"])
    for name in ("Experimental_Setup.md", "Audit.md"):
        positions: list[int] = []
        for aim in range(1, 5):
            match = _require_one(
                rf"^## Aim {aim}$",
                texts[name],
                context=f"{name} Aim {aim} section",
                flags=re.MULTILINE,
            )
            positions.append(match.start())
        if positions != sorted(positions):
            raise BundleVerificationError(f"{name} Aim 1--4 sections are out of order")

    audit_rows = re.findall(
        r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$",
        texts["Audit.md"],
        flags=re.MULTILINE,
    )
    observed = {source_id: digest for source_id, digest in audit_rows}
    expected = {str(source["id"]): str(source["sha256"]) for source in sources}
    if len(audit_rows) != len(observed) or observed != expected:
        raise BundleVerificationError("Audit.md governed source index is not one-to-one and exact")
    for name in REPORT_DOCUMENTS:
        if identities[name]["sha256"] != document_pins[name]:
            raise BundleVerificationError(f"FINAL-v12.1 {name} SHA-256 drift")
    return identities


def _validate_bundle(paths: BundlePaths) -> dict[str, Any]:
    parent_receipt, parent_manifest, _, parent_identities = _validate_parent(paths)
    manifest, sources, manifest_identity = _validate_final_manifest(paths, parent_manifest)
    documents = _validate_document_topology(paths, sources)
    return {
        "parent_receipt": parent_receipt,
        "parent_identities": parent_identities,
        "manifest": manifest,
        "manifest_identity": manifest_identity,
        "sources": sources,
        "documents": documents,
    }


def build_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Build the deterministic receipt object without publishing it."""

    selected = default_paths() if paths is None else paths
    validated = _validate_bundle(selected)
    parent = validated["parent_identities"]
    sources = validated["sources"]
    return {
        "schema_version": 1,
        "bundle": "reports/final_v12_1",
        "status": FINAL_SEALED_STATUS,
        "organization": "standalone_integrated_results_for_paper_selection",
        "parent_final_v12": {
            "status": validated["parent_receipt"]["status"],
            "receipt": parent["receipt"],
            "source_manifest": parent["source_manifest"],
            "documents": parent["documents"],
            "verifier": parent["verifier"],
            "verifier_test": parent["verifier_test"],
            "authoritative_source_count": selected.expected_parent_source_count,
        },
        "documents": validated["documents"],
        "source_manifest": validated["manifest_identity"],
        "authoritative_sources": sources,
        "verification": {
            "verifier": identity(
                selected.verifier_code,
                display_path=_display(selected.verifier_code, selected.repo),
            ),
            "tests": identity(
                selected.verifier_test,
                display_path=_display(selected.verifier_test, selected.repo),
            ),
        },
        "source_count": len(sources),
        "extension_source_count": len(selected.expected_extension_source_ids),
        "extension_source_ids": list(selected.expected_extension_source_ids),
        "declared_exclusions": {
            "not_run": ["separately_trained_missing_E0_cells", "E2-CPHT-R"],
            "generated_unread": ["whole_section_pathology"],
            "not_selectable": [
                "superseded_headline_fields",
                "failed_closed_or_unverified_attempts",
                "audit_only_noncontrolling_results",
            ],
        },
        "checks": {
            "strict_parent_receipt_and_manifest_json": "PASS",
            "recursive_sealed_final_v12_replay": "PASS",
            "sealed_final_v12_receipt_manifest_and_document_identities": "PASS",
            "exact_139_record_parent_reuse_plus_frozen_extension": "PASS",
            "direct_complete_source_rehash": "PASS",
            "strict_json_and_no_symlinks": "PASS",
            "standalone_non_addendum_results_topology": "PASS",
            "complete_aim_experiment_population_topology": "PASS",
            "refit_vs_within_seed_five_fold_table_topology": "PASS",
            "explicit_e2d1_through_e2d6_sections": "PASS",
            "paper_selection_matrix_and_exclusions": "PASS",
            "governed_78_48_2_record_ids_exactly_once": "PASS",
            "one_to_one_audit_source_index": "PASS",
            "no_stale_candidate_or_unpublished_parent_wording": "PASS",
            "final_v12_1_document_and_manifest_hashes": "PASS",
            "final_v12_1_verifier_and_test_identities": "PASS",
        },
    }


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def check_bundle(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Run the complete read-only verification and report seal readiness."""

    selected = default_paths() if paths is None else paths
    receipt = build_receipt(selected)
    return {
        "bundle": receipt["bundle"],
        "status": "READY_TO_SEAL",
        "published_receipt_present": selected.destination.exists()
        or selected.destination.is_symlink(),
        "parent_status": receipt["parent_final_v12"]["status"],
        "source_count": receipt["source_count"],
        "source_manifest": receipt["source_manifest"],
        "documents": receipt["documents"],
        "verification": receipt["verification"],
        "checks": receipt["checks"],
    }


def verify_published_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Rebuild and byte-compare an already published FINAL-v12.1 receipt."""

    selected = default_paths() if paths is None else paths
    _reject_symlink_chain(selected.destination, context="published FINAL-v12.1 receipt")
    if not selected.destination.is_file():
        raise BundleVerificationError("published FINAL-v12.1 receipt is absent")
    published, _ = _load_stable_json(
        selected.destination,
        label="published FINAL-v12.1 receipt",
        display_path=_display(selected.destination, selected.repo),
    )
    expected = build_receipt(selected)
    if published != expected or selected.destination.read_bytes() != _receipt_bytes(expected):
        raise BundleVerificationError("published FINAL-v12.1 receipt byte identity drift")
    return published


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Publish the deterministic FINAL-v12.1 receipt, refusing every overwrite."""

    selected = default_paths() if paths is None else paths
    if selected.destination.exists() or selected.destination.is_symlink():
        raise BundleVerificationError("refusing to overwrite FINAL-v12.1 receipt")
    _reject_symlink_chain(selected.destination.parent, context="FINAL-v12.1 receipt directory")
    receipt = build_receipt(selected)
    content = _receipt_bytes(receipt)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{selected.destination.name}.",
        suffix=".tmp",
        dir=selected.destination.parent,
    )
    temporary = Path(temporary_name)
    temporary_inode: tuple[int, int] | None = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            stat = os.fstat(handle.fileno())
            temporary_inode = (stat.st_dev, stat.st_ino)
        try:
            os.link(temporary, selected.destination)
        except FileExistsError as exc:
            raise BundleVerificationError("FINAL-v12.1 receipt was concurrently published") from exc
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
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true", help="run read-only verification")
    actions.add_argument("--seal", action="store_true", help="create the receipt exactly once")
    actions.add_argument(
        "--verify-published",
        action="store_true",
        help="rebuild and compare an existing receipt",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.seal:
            result = seal()
        elif args.verify_published:
            result = verify_published_receipt()
        else:
            result = check_bundle()
    except BundleVerificationError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
