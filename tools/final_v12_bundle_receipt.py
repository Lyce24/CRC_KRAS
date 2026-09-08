#!/usr/bin/env python3
"""Verify and exactly-once seal the FINAL-v12 integrated report bundle.

FINAL-v12 is a presentation-only integration of the sealed FINAL-v11 evidence
graph.  This verifier pins and strict-parses the FINAL-v11 receipt and source
manifest, authenticates the parent documents, requires byte-for-byte equality
of the 139 source records in the FINAL-v11 and FINAL-v12 manifests, and then
directly rehashes every listed source.  It also checks that Results.md is a
standalone paper-selection report with complete aim, experiment, population,
E2d1--E2d6, selection-matrix, and exclusion topology.

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
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FINAL_V12 = REPO / "reports" / "final_v12"
PARENT_DIR = REPO / "reports" / "final_v11"

REPORT_DOCUMENTS = ("Experimental_Setup.md", "Results.md", "Audit.md")
SOURCE_MANIFEST_NAME = "source_manifest.json"
FINAL_RECEIPT_NAME = "report_bundle_receipt.json"

PARENT_SEALED_STATUS = "SEALED_COMPLETED_RESULTS_WITH_DECLARED_NOT_RUN_ARM"
PARENT_MANIFEST_STATUS = "candidate_ready_for_final_v11_verification"
FINAL_MANIFEST_STATUS = "sealed_final_v11_source_whitelist_reused_for_integrated_selection"
FINAL_SEALED_STATUS = "SEALED_FINAL_V12_INTEGRATED_PAPER_SELECTION"
EXPECTED_SOURCE_COUNT = 139
GOVERNED_RESULTS_SOURCE_ID = "aim1-tcga-surgen-two-encoder-downstream-v3-results"

EXPECTED_PARENT_RECEIPT_SHA256 = "1a06b6ecfcdcaba83b2a5380e0f61a06380647ae787cd1d2777006f5a027a840"
EXPECTED_PARENT_MANIFEST_SHA256 = "ba6037a7b2d385cb278d5fa7666094e152774535044fec6077867c9fa41a2b3a"
EXPECTED_PARENT_DOCUMENT_SHA256 = {
    "Experimental_Setup.md": "85a50335db2eb0546f4bbffae8302cfb4cadf7ef86d84debebb52e469f6d05e5",
    "Results.md": "9c737cdb52be2bdcf518a81c758b241e018ff9621db5505ac299abafbe713d1a",
    "Audit.md": "db7f4e733067a4e36c84521b5156bfd6046f5e4ebc8c1ab17f57dbe754a37b1a",
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
    """A fail-closed FINAL-v12 parent, source, document, or receipt error."""


@dataclass(frozen=True)
class BundlePaths:
    """Filesystem locations and immutable parent expectations."""

    repo: Path
    final_v12: Path
    destination: Path
    parent_dir: Path
    parent_receipt: Path
    parent_manifest: Path
    expected_parent_receipt_sha256: str
    expected_parent_manifest_sha256: str
    expected_parent_document_sha256: Mapping[str, str]
    expected_source_count: int = EXPECTED_SOURCE_COUNT
    expected_parent_status: str = PARENT_SEALED_STATUS
    expected_parent_manifest_status: str = PARENT_MANIFEST_STATUS
    expected_final_manifest_status: str = FINAL_MANIFEST_STATUS


def default_paths() -> BundlePaths:
    """Return production FINAL-v12 and sealed FINAL-v11 paths."""

    return BundlePaths(
        repo=REPO,
        final_v12=FINAL_V12,
        destination=FINAL_V12 / FINAL_RECEIPT_NAME,
        parent_dir=PARENT_DIR,
        parent_receipt=PARENT_DIR / FINAL_RECEIPT_NAME,
        parent_manifest=PARENT_DIR / SOURCE_MANIFEST_NAME,
        expected_parent_receipt_sha256=EXPECTED_PARENT_RECEIPT_SHA256,
        expected_parent_manifest_sha256=EXPECTED_PARENT_MANIFEST_SHA256,
        expected_parent_document_sha256=dict(EXPECTED_PARENT_DOCUMENT_SHA256),
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
    rehash: bool,
) -> list[dict[str, Any]]:
    if not isinstance(sources, list) or len(sources) != paths.expected_source_count:
        raise BundleVerificationError(
            f"source roster must contain exactly {paths.expected_source_count} records"
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
    """Authenticate the sealed FINAL-v11 receipt, manifest, and documents."""

    receipt, receipt_identity = _load_stable_json(
        paths.parent_receipt,
        label="sealed FINAL-v11 receipt",
        display_path=_display(paths.parent_receipt, paths.repo),
    )
    if receipt_identity["sha256"] != paths.expected_parent_receipt_sha256:
        raise BundleVerificationError("sealed FINAL-v11 receipt SHA-256 drift")
    if receipt.get("schema_version") != 1 or receipt.get("bundle") != "reports/final_v11":
        raise BundleVerificationError("sealed FINAL-v11 receipt identity fields are invalid")
    if receipt.get("status") != paths.expected_parent_status:
        raise BundleVerificationError("sealed FINAL-v11 receipt status is not authoritative")
    checks = receipt.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(value != "PASS" for value in checks.values())
    ):
        raise BundleVerificationError("sealed FINAL-v11 receipt does not have all checks PASS")

    parent_manifest, manifest_identity = _load_stable_json(
        paths.parent_manifest,
        label="sealed FINAL-v11 source manifest",
        display_path=_display(paths.parent_manifest, paths.repo),
    )
    if manifest_identity["sha256"] != paths.expected_parent_manifest_sha256:
        raise BundleVerificationError("sealed FINAL-v11 manifest SHA-256 drift")
    _require_exact_identity(
        receipt.get("source_manifest"),
        paths.parent_manifest,
        display_path=_display(paths.parent_manifest, paths.repo),
        context="sealed FINAL-v11 receipt source_manifest",
    )
    if set(parent_manifest) != _MANIFEST_KEYS:
        raise BundleVerificationError("sealed FINAL-v11 source manifest schema is not exact")
    if parent_manifest["schema_version"] != 2 or parent_manifest["bundle"] != "final_v11":
        raise BundleVerificationError("sealed FINAL-v11 source manifest identity is invalid")
    if parent_manifest["status"] != paths.expected_parent_manifest_status:
        raise BundleVerificationError("sealed FINAL-v11 source manifest status drift")
    if parent_manifest["pending_artifacts"] != []:
        raise BundleVerificationError("sealed FINAL-v11 source manifest has pending artifacts")
    parent_sources = _validate_source_roster(parent_manifest["artifacts"], paths, rehash=False)
    if receipt.get("authoritative_sources") != parent_sources:
        raise BundleVerificationError(
            "sealed FINAL-v11 receipt and source-manifest artifact rosters differ"
        )

    expected_document_pins = dict(paths.expected_parent_document_sha256)
    if set(expected_document_pins) != set(REPORT_DOCUMENTS):
        raise BundleVerificationError("sealed FINAL-v11 document pin roster is not exact")
    receipt_documents = receipt.get("documents")
    if not isinstance(receipt_documents, dict) or set(receipt_documents) != set(REPORT_DOCUMENTS):
        raise BundleVerificationError("sealed FINAL-v11 receipt document roster is not exact")
    parent_documents: dict[str, Any] = {}
    for name in REPORT_DOCUMENTS:
        document_path = paths.parent_dir / name
        actual = _require_exact_identity(
            receipt_documents[name],
            document_path,
            display_path=_display(document_path, paths.repo),
            context=f"sealed FINAL-v11 {name}",
        )
        if actual["sha256"] != expected_document_pins[name]:
            raise BundleVerificationError(f"sealed FINAL-v11 {name} SHA-256 drift")
        parent_documents[name] = actual

    return (
        receipt,
        parent_manifest,
        parent_sources,
        {
            "receipt": receipt_identity,
            "source_manifest": manifest_identity,
            "documents": parent_documents,
        },
    )


def _validate_final_manifest(
    paths: BundlePaths,
    parent_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = paths.final_v12 / SOURCE_MANIFEST_NAME
    manifest, manifest_identity = _load_stable_json(
        manifest_path,
        label="FINAL-v12 source manifest",
        display_path=_display(manifest_path, paths.repo),
    )
    if set(manifest) != _MANIFEST_KEYS:
        raise BundleVerificationError("FINAL-v12 source manifest schema is not exact")
    if manifest["schema_version"] != 2 or manifest["bundle"] != "final_v12":
        raise BundleVerificationError("FINAL-v12 source manifest identity is invalid")
    if manifest["status"] != paths.expected_final_manifest_status:
        raise BundleVerificationError("FINAL-v12 source manifest status drift")
    if manifest["pending_artifacts"] != []:
        raise BundleVerificationError("FINAL-v12 source manifest has pending artifacts")

    parent_comparable = {
        key: value for key, value in parent_manifest.items() if key not in {"bundle", "status"}
    }
    final_comparable = {
        key: value for key, value in manifest.items() if key not in {"bundle", "status"}
    }
    if final_comparable != parent_comparable:
        raise BundleVerificationError(
            "FINAL-v12 manifest differs from FINAL-v11 beyond bundle/status metadata"
        )
    sources = _validate_source_roster(manifest["artifacts"], paths, rehash=True)
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
        "unpublished FINAL-v11 receipt declaration",
        r"\bno\s+FINAL[- ]v11 receipt has been published\b",
    ),
    (
        "unpublished FINAL-v11 receipt declaration",
        r"\b(?:unpublished\s+FINAL[- ]v11 receipt|FINAL[- ]v11 receipt.{0,80}unpublished)\b",
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
    expected_title = "# FINAL-v12 integrated results for paper selection"
    if not lines or lines[0] != expected_title:
        raise BundleVerificationError("Results.md integrated-results title is not exact")
    h1_lines = [line for line in lines if re.fullmatch(r"#\s+.+", line)]
    if h1_lines != [expected_title]:
        raise BundleVerificationError("Results.md must be one standalone report, with one H1")
    if re.search(r"(?im)^#{1,6}\s+.*\baddend(?:um|a)\b", results):
        raise BundleVerificationError("Results.md uses addendum-form heading topology")
    if re.search(
        r"(?im)^\s*(?:this report|FINAL[- ]v12)\s+(?:is\s+)?an?\s+addendum\b",
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


def _validate_governed_record_bindings(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
    results: str,
) -> None:
    """Require every governed 78/48/2 report identifier exactly once."""

    matches = [source for source in sources if source["id"] == GOVERNED_RESULTS_SOURCE_ID]
    if not matches:
        # Small synthetic unit-test manifests do not model the scientific graph.
        if paths.expected_source_count == EXPECTED_SOURCE_COUNT:
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


def _validate_document_topology(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    texts: dict[str, str] = {}
    identities: dict[str, Any] = {}
    for name in REPORT_DOCUMENTS:
        document_path = paths.final_v12 / name
        document_identity = identity(
            document_path,
            display_path=_display(document_path, paths.repo),
        )
        try:
            text = document_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise BundleVerificationError(f"FINAL-v12 {name} is not readable UTF-8: {exc}") from exc
        if not text.startswith("# FINAL-v12 "):
            raise BundleVerificationError(f"FINAL-v12 {name} has an invalid title")
        if (
            identity(
                document_path,
                display_path=_display(document_path, paths.repo),
            )
            != document_identity
        ):
            raise BundleVerificationError(f"FINAL-v12 {name} changed while it was being read")
        texts[name] = text
        identities[name] = document_identity

    for name, text in texts.items():
        for label, pattern in _STALE_PUBLICATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                raise BundleVerificationError(f"{name} retains stale {label}")

    _validate_results(texts["Results.md"])
    _validate_governed_record_bindings(paths, sources, texts["Results.md"])
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
        "bundle": "reports/final_v12",
        "status": FINAL_SEALED_STATUS,
        "organization": "standalone_integrated_results_for_paper_selection",
        "parent_final_v11": {
            "status": validated["parent_receipt"]["status"],
            "receipt": parent["receipt"],
            "source_manifest": parent["source_manifest"],
            "documents": parent["documents"],
            "authoritative_source_count": len(sources),
        },
        "documents": validated["documents"],
        "source_manifest": validated["manifest_identity"],
        "authoritative_sources": sources,
        "source_count": len(sources),
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
            "sealed_final_v11_receipt_sha256_and_status": "PASS",
            "sealed_final_v11_document_identities": "PASS",
            "exact_139_record_manifest_reuse": "PASS",
            "direct_139_source_rehash": "PASS",
            "strict_json_and_no_symlinks": "PASS",
            "standalone_non_addendum_results_topology": "PASS",
            "complete_aim_experiment_population_topology": "PASS",
            "explicit_e2d1_through_e2d6_sections": "PASS",
            "paper_selection_matrix_and_exclusions": "PASS",
            "governed_78_48_2_record_ids_exactly_once": "PASS",
            "one_to_one_audit_source_index": "PASS",
            "no_stale_candidate_or_unpublished_parent_wording": "PASS",
            "final_v12_document_and_manifest_hashes": "PASS",
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
        "parent_status": receipt["parent_final_v11"]["status"],
        "source_count": receipt["source_count"],
        "source_manifest": receipt["source_manifest"],
        "documents": receipt["documents"],
        "checks": receipt["checks"],
    }


def verify_published_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Rebuild and byte-compare an already published FINAL-v12 receipt."""

    selected = default_paths() if paths is None else paths
    _reject_symlink_chain(selected.destination, context="published FINAL-v12 receipt")
    if not selected.destination.is_file():
        raise BundleVerificationError("published FINAL-v12 receipt is absent")
    published, _ = _load_stable_json(
        selected.destination,
        label="published FINAL-v12 receipt",
        display_path=_display(selected.destination, selected.repo),
    )
    expected = build_receipt(selected)
    if published != expected or selected.destination.read_bytes() != _receipt_bytes(expected):
        raise BundleVerificationError("published FINAL-v12 receipt byte identity drift")
    return published


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Publish the deterministic receipt atomically, refusing every overwrite."""

    selected = default_paths() if paths is None else paths
    if selected.destination.exists() or selected.destination.is_symlink():
        raise BundleVerificationError("refusing to overwrite FINAL-v12 receipt")
    _reject_symlink_chain(selected.destination.parent, context="FINAL-v12 receipt directory")
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
            raise BundleVerificationError("FINAL-v12 receipt was concurrently published") from exc
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
