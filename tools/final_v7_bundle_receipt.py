#!/usr/bin/env python3
"""Verify and optionally seal the append-only final-v7 report bundle.

This verifier is intentionally independent of the E2f-v3 analysis and the
reviews/v5 packet builder.  Its default mode is read-only.  ``--seal`` writes
``report_bundle_receipt.json`` exactly once, after every check has passed,
using atomic exclusive publication.

The final-v7 integration receipt is a small, explicit contract.  It must be
append-only PASS, declare non-empty ``inputs``, ``outputs`` (or ``artifacts``),
and ``code`` identity inventories, and contain exactly these components::

    {
      "components": {
        "e2f_v3": {
          "integrity_status": "PASS",
          "scientific_status": "EXECUTED",
          "receipt": {"path": ..., "size_bytes": ..., "sha256": ...}
        },
        "reviews_v5": {
          "integrity_status": "PASS",
          "scientific_status": "GENERATED_UNREAD",
          "analysis_executed": false,
          "unblinding_performed": false,
          "analysis_result": null,
          "review_root": ...,
          "receipt": { ... },
          "reader_scoring_form": { ... },
          "reviewer_info_form": { ... }
        }
      }
    }

Every declared identity requires both SHA-256 and byte size.  Receipt-shaped
JSON files are followed recursively, and every identity in every visited
receipt is independently rehashed.  Thus a PASS receipt that merely lists
counts, paths, or hashes without sizes cannot satisfy this seal.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FINAL_V6 = REPO / "reports" / "final_v6"
SNAPSHOT_ROOT = REPO / "reports" / "snapshots" / "final_v6_pre_v7_20260821"
SNAPSHOT_RECEIPT = (
    REPO / "reports" / "snapshots" / "final_v6_pre_v7_20260821_receipt.json"
)
FINAL_V7 = REPO / "reports" / "final_v7"
FINAL_V7_ADDITIONS = REPO / "reports" / "reruns" / "final_v7_additions_20260821"
INTEGRATION_RECEIPT = FINAL_V7_ADDITIONS / "integration" / "receipt.json"
REVIEWS_V5 = REPO / "reviews" / "v5"
V5_PRE_READ_ADDENDUM = (
    REPO / "reviews" / "v5_pre_read_nuisance_addendum_20260821"
)
V5_PRE_READ_ADDENDUM_RECEIPT = V5_PRE_READ_ADDENDUM / "ADDENDUM_RECEIPT.json"

REPORT_DOCUMENTS = ("Experimental_Setup.md", "Results.md", "Audit.md")
SNAPSHOT_DOCUMENTS = (
    *REPORT_DOCUMENTS,
    "parent_final_v5_receipt.json",
    "report_bundle_receipt.json",
)
PARENT_COPY_NAME = "parent_final_v6_receipt.json"
INHERITED_PARENT_NAME = "parent_final_v5_receipt.json"
FINAL_RECEIPT_NAME = "report_bundle_receipt.json"
COMPONENT_NAMES = ("e2f_v3", "reviews_v5")
FOLLOW_RECEIPT_NAMES = {
    "receipt.json",
    "input_receipt.json",
    "inputs_receipt.json",
    "output_receipt.json",
    "completion_receipt.json",
    "verification_receipt.json",
    "packet_receipt.json",
    "report_bundle_receipt.json",
}

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_PLACEHOLDER_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])(?:TODO|TBD|TK)(?![A-Za-z0-9_])", re.IGNORECASE),
    re.compile(r"<\s*(?:PLACEHOLDER|INSERT(?:\s+[^>]*)?|FILL(?:\s+[^>]*)?)\s*>", re.IGNORECASE),
    re.compile(r"\{\{[^{}]+\}\}"),
    re.compile(r"\[\[\s*(?:PLACEHOLDER|TODO|TBD|TK)[^\]]*\]\]", re.IGNORECASE),
    re.compile(
        r"\b(?:REPLACE_ME|FILL_ME_IN|SHA256_HERE|HASH_HERE|P_VALUE_HERE|"
        r"RESULT_HERE|INSERT_(?:VALUE|HASH|RESULT|TEXT))\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![0-9a-fA-F])0{64}(?![0-9a-fA-F])"),
)
_REPORT_CONTRACT_START = "<!-- FINAL_V7_REPORT_CONTRACT_START -->\n```json\n"
_REPORT_CONTRACT_END = "\n```\n<!-- FINAL_V7_REPORT_CONTRACT_END -->"
_REQUIRED_PREFIX_TOKENS = (
    "FINAL-v7 CONTROLLING STATUS",
    "INTERPRETIVE PRECEDENCE",
    "`EXECUTED` with integrity `PASS`",
    "`GENERATED_UNREAD`",
    "`analysis_executed=false`",
    "`unblinding_performed=false`",
    "`analysis_result=null`",
    "inherited statements",
    "superseded",
)
_REQUIRED_AUDIT_CORRECTIONS = (
    "No equivalence margin, equivalence test or noninferiority test was performed.",
    "establishes no mechanism.",
    "cannot assign the cause of every KRAS estimate",
    "neither mediation nor an attributable fraction",
    "do not prove specificity.",
    "does not establish biological absence.",
    "Neither impossibility nor deployment readiness is established.",
    "with no reader score, unblinding, analysis or scientific result.",
)
_REQUIRED_ADDENDUM_RESULTS_TOKENS = (
    "GENERATED_UNREAD_SECONDARY_ADDENDUM",
    "NONE_SECONDARY_ROBUSTNESS_ONLY",
    "no result",
)
_REQUIRED_ADDENDUM_AUDIT_TOKENS = (
    "GENERATED_UNREAD_SECONDARY_ADDENDUM",
    "analysis_executed=false",
    "unblinding_performed=false",
    "analysis_result=null",
    "NONE_SECONDARY_ROBUSTNESS_ONLY",
    "parent_primary_unchanged=true",
)
_SCORING_IDENTIFIER_COLUMNS = {
    "case_id",
    "packet_case_id",
    "item_id",
    "image_id",
    "field_id",
    "slide_id",
    "display_order",
}
_FORBIDDEN_REVIEW_RESULT_NAMES = {
    "analysis_result.json",
    "analysis_results.json",
    "results.json",
    "scientific_results.json",
    "unblinded_results.json",
    "completed_scoring_form.csv",
    "scoring_form_completed.csv",
}
_FORBIDDEN_REVIEW_RESULT_DIRS = {
    "analysis_results",
    "completed_results",
    "unblinded_analysis",
}
_LINEAGE_ONLY_TRAIL_TOKENS = {
    "legacy_receipts",
    "legacy_lineage",
    "non_authoritative",
    "preserved",
    "preserved_receipts",
    "superseded",
    "supersedes",
}


class BundleVerificationError(RuntimeError):
    """A fail-closed final-v7 bundle or provenance verification failure."""


@dataclass(frozen=True)
class BundlePaths:
    final_v6: Path
    snapshot_root: Path
    snapshot_receipt: Path
    final_v7: Path
    integration_receipt: Path
    reviews_v5: Path
    destination: Path
    v5_pre_read_addendum: Path = V5_PRE_READ_ADDENDUM
    verifier_code: Path | None = None
    verifier_test: Path | None = None


@dataclass(frozen=True)
class DeclaredIdentity:
    source_receipt: Path
    trail: tuple[str, ...]
    path: Path
    size_bytes: int
    sha256: str


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    """Return the canonical absolute-path, size, and SHA-256 identity."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise BundleVerificationError(f"missing required file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleVerificationError(f"invalid JSON receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleVerificationError(f"receipt must contain a JSON object: {path}")
    return value


def _resolve_declared_path(raw: str, receipt: Path) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = receipt.parent / candidate
    return candidate.resolve()


def _inferred_relative_path(trail: tuple[str, ...]) -> str | None:
    if not trail:
        return None
    candidate = trail[-1]
    if "/" in candidate or "\\" in candidate or Path(candidate).suffix:
        return candidate
    return None


def _declared_identities(
    value: Any,
    receipt: Path,
    trail: tuple[str, ...] = (),
) -> Iterator[DeclaredIdentity]:
    """Yield complete identities and reject partial hash/size declarations."""
    if isinstance(value, dict):
        has_sha = "sha256" in value
        has_size = "size_bytes" in value
        has_path = any(key in value for key in ("path", "relative_path", "file"))
        if has_sha or (has_size and has_path):
            location = f"{receipt}:{'/'.join(trail) or '<root>'}"
            if not has_sha or not has_size:
                raise BundleVerificationError(
                    f"identity must declare both sha256 and size_bytes at {location}"
                )
            raw_path = next(
                (
                    value[key]
                    for key in ("path", "relative_path", "file")
                    if isinstance(value.get(key), str)
                ),
                None,
            )
            if raw_path is None:
                raw_path = _inferred_relative_path(trail)
            if raw_path is None:
                raise BundleVerificationError(f"identity has no file path at {location}")
            digest = value["sha256"]
            size = value["size_bytes"]
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise BundleVerificationError(f"invalid SHA-256 at {location}")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise BundleVerificationError(f"invalid size_bytes at {location}")
            yield DeclaredIdentity(
                source_receipt=receipt.resolve(),
                trail=trail,
                path=_resolve_declared_path(raw_path, receipt),
                size_bytes=size,
                sha256=digest.lower(),
            )
        for key, item in value.items():
            yield from _declared_identities(item, receipt, (*trail, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _declared_identities(item, receipt, (*trail, str(index)))


def _verify_declared_identity(declared: DeclaredIdentity) -> dict[str, Any]:
    observed = identity(declared.path)
    if observed["size_bytes"] != declared.size_bytes:
        raise BundleVerificationError(
            f"size mismatch for {declared.path}: declared {declared.size_bytes}, "
            f"observed {observed['size_bytes']}"
        )
    if observed["sha256"] != declared.sha256:
        raise BundleVerificationError(
            f"SHA-256 mismatch for {declared.path}: declared {declared.sha256}, "
            f"observed {observed['sha256']}"
        )
    return observed


def _looks_like_receipt(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".json" and (
        name in FOLLOW_RECEIPT_NAMES or "receipt" in name
    )


def _is_lineage_only_receipt_reference(declared: DeclaredIdentity) -> bool:
    """Identify an explicitly non-authoritative legacy receipt reference.

    The legacy receipt file is still size-checked and rehashed.  Only descent
    into its historical inventory is skipped.  Current integration, E2f-v3,
    and reviews/v5 receipts cannot use this exemption because their semantic
    component records are separately required to appear in the recursively
    visited set.
    """
    return any(
        token.casefold() in _LINEAGE_ONLY_TRAIL_TOKENS
        or token.casefold().startswith("legacy_")
        for token in declared.trail
    )


def verify_receipt_tree(receipt_path: Path) -> dict[str, Any]:
    """Recursively rehash all complete identities reachable from a receipt."""
    root_receipt = receipt_path.resolve()
    root = root_receipt.parent
    if not root_receipt.is_file():
        raise BundleVerificationError(f"missing integration receipt: {root_receipt}")
    top = _load_json(root_receipt)
    top_status = str(top.get("status", "")).upper()
    if top_status not in {"PASS", "COMPLETE"}:
        raise BundleVerificationError(
            f"top receipt is not PASS/COMPLETE: {root_receipt}"
        )
    if top.get("append_only") is not True:
        raise BundleVerificationError(
            f"top receipt must declare append_only true: {root_receipt}"
        )

    pending = [root_receipt]
    visited_receipts: set[Path] = set()
    verified: dict[tuple[Path, str, int], dict[str, Any]] = {}
    declaration_count = 0
    lineage_only_receipts: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in visited_receipts:
            continue
        visited_receipts.add(current)
        payload = _load_json(current)
        nested_status = payload.get("status")
        if nested_status is not None and str(nested_status).upper() not in {
            "PASS",
            "COMPLETE",
        }:
            raise BundleVerificationError(
                f"nested receipt has non-PASS status {nested_status!r}: {current}"
            )
        if payload.get("append_only") is False:
            raise BundleVerificationError(
                f"receipt explicitly is not append-only: {current}"
            )
        for declared in _declared_identities(payload, current):
            declaration_count += 1
            key = (declared.path, declared.sha256, declared.size_bytes)
            if key not in verified:
                verified[key] = _verify_declared_identity(declared)
            if _looks_like_receipt(declared.path):
                if _is_lineage_only_receipt_reference(declared):
                    lineage_only_receipts.add(declared.path)
                elif declared.path not in visited_receipts:
                    pending.append(declared.path)

    local_files = [
        item for item in verified.values() if Path(item["path"]).is_relative_to(root)
    ]
    if not local_files:
        raise BundleVerificationError(
            f"integration receipt declares no integration-local artifacts: {root_receipt}"
        )
    verified_files = sorted(verified.values(), key=lambda item: item["path"])
    return {
        "status": "PASS",
        "declared_component_status": top_status,
        "component_root": str(root),
        "receipt": identity(root_receipt),
        "append_only_declared": True,
        "declared_identity_references": declaration_count,
        "unique_declared_files": len(verified_files),
        "component_local_files": len(local_files),
        "recursively_verified_receipts": [
            str(path) for path in sorted(visited_receipts)
        ],
        "lineage_only_receipts_directly_rehashed_not_recursed": [
            str(path) for path in sorted(lineage_only_receipts)
        ],
        "verified_files": verified_files,
        "rehash_mismatches": 0,
    }


def _regular_file_inventory(root: Path) -> dict[str, Path]:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise BundleVerificationError(f"missing required directory: {resolved}")
    inventory: dict[str, Path] = {}
    for item in sorted(resolved.rglob("*")):
        if item.is_symlink():
            raise BundleVerificationError(f"snapshot trees may not contain symlinks: {item}")
        if item.is_file():
            inventory[item.relative_to(resolved).as_posix()] = item
    return inventory


def _root_reference_matches(
    raw: Any,
    expected: Path,
    receipt: Path,
    repository_root: Path,
) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return False
    candidate = Path(raw)
    if candidate.is_absolute():
        candidates = {candidate.resolve()}
    else:
        candidates = {
            (receipt.parent / candidate).resolve(),
            (repository_root / candidate).resolve(),
            (repository_root / "reports" / candidate).resolve(),
        }
    return expected.resolve() in candidates


def _snapshot_inventory(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("documents", "identities"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    raise BundleVerificationError("snapshot receipt has no documents/identities inventory")


def _snapshot_record(record: Any, relative_path: str) -> tuple[str, int]:
    if not isinstance(record, dict):
        raise BundleVerificationError(
            f"invalid snapshot identity record for {relative_path}"
        )
    digest = record.get("sha256")
    size = record.get("size_bytes", record.get("bytes"))
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise BundleVerificationError(f"invalid snapshot SHA-256 for {relative_path}")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise BundleVerificationError(f"invalid snapshot byte size for {relative_path}")
    return digest.lower(), size


def _verify_final_v6_receipt(final_v6: Path) -> dict[str, Any]:
    receipt_path = (final_v6 / FINAL_RECEIPT_NAME).resolve()
    payload = _load_json(receipt_path)
    if str(payload.get("status", "")).upper() != "PASS":
        raise BundleVerificationError("authoritative final-v6 receipt is not PASS")
    problems = payload.get("problems")
    if problems not in (None, []):
        raise BundleVerificationError("authoritative final-v6 receipt records problems")

    verification = payload.get("verification")
    markdown = verification.get("markdown") if isinstance(verification, dict) else None
    if not isinstance(markdown, dict):
        raise BundleVerificationError(
            "authoritative final-v6 receipt omits its Markdown identity inventory"
        )
    checked: dict[str, Any] = {}
    for filename in REPORT_DOCUMENTS:
        record = markdown.get(filename)
        if not isinstance(record, dict):
            raise BundleVerificationError(
                f"authoritative final-v6 receipt omits {filename}"
            )
        expected_sha = record.get("sha256")
        expected_size = record.get("size_bytes", record.get("bytes"))
        observed = identity(final_v6 / filename)
        if (
            observed["sha256"] != expected_sha
            or observed["size_bytes"] != expected_size
        ):
            raise BundleVerificationError(
                f"authoritative final-v6 Markdown identity mismatch for {filename}"
            )
        checked[filename] = observed
    return {
        "status": "PASS",
        "receipt": identity(receipt_path),
        "markdown_rehashed": checked,
    }


def verify_parent_and_snapshot(paths: BundlePaths) -> dict[str, Any]:
    final_v6 = paths.final_v6.resolve()
    snapshot_root = paths.snapshot_root.resolve()
    snapshot_receipt_path = paths.snapshot_receipt.resolve()
    final_v7 = paths.final_v7.resolve()
    repository_root = final_v6.parents[1]

    authoritative_check = _verify_final_v6_receipt(final_v6)
    authoritative = final_v6 / FINAL_RECEIPT_NAME
    copied = final_v7 / PARENT_COPY_NAME
    if not copied.is_file() or copied.read_bytes() != authoritative.read_bytes():
        raise BundleVerificationError(
            "final-v7 parent_final_v6_receipt.json is not byte-identical to "
            "the authoritative final-v6 receipt"
        )

    snapshot = _load_json(snapshot_receipt_path)
    status = snapshot.get("status")
    if status is not None:
        if str(status).upper() != "PASS" or snapshot.get("append_only") is False:
            raise BundleVerificationError("pre-v7 snapshot receipt is not PASS")
        if snapshot.get("problems") not in (None, []):
            raise BundleVerificationError("pre-v7 snapshot receipt records problems")
    elif snapshot.get("diff_rq_clean") is not True:
        raise BundleVerificationError(
            "legacy pre-v7 snapshot receipt must declare diff_rq_clean true"
        )

    source_reference = snapshot.get("source_root", snapshot.get("source"))
    snapshot_reference = snapshot.get("snapshot_root", snapshot.get("snapshot"))
    if not _root_reference_matches(
        source_reference, final_v6, snapshot_receipt_path, repository_root
    ):
        raise BundleVerificationError("snapshot receipt source does not identify final-v6")
    if not _root_reference_matches(
        snapshot_reference, snapshot_root, snapshot_receipt_path, repository_root
    ):
        raise BundleVerificationError(
            "snapshot receipt snapshot root does not match the configured snapshot"
        )

    source_files = _regular_file_inventory(final_v6)
    snapshot_files = _regular_file_inventory(snapshot_root)
    if set(source_files) != set(snapshot_files):
        missing = sorted(set(source_files) - set(snapshot_files))
        extra = sorted(set(snapshot_files) - set(source_files))
        raise BundleVerificationError(
            f"final-v6/snapshot file-set mismatch; missing={missing}, extra={extra}"
        )
    missing_required = set(SNAPSHOT_DOCUMENTS) - set(source_files)
    if missing_required:
        raise BundleVerificationError(
            f"final-v6 omits required snapshot documents: {sorted(missing_required)}"
        )

    inventory = _snapshot_inventory(snapshot)
    if set(inventory) != set(source_files):
        missing = sorted(set(source_files) - set(inventory))
        extra = sorted(set(inventory) - set(source_files))
        raise BundleVerificationError(
            f"snapshot receipt inventory mismatch; missing={missing}, extra={extra}"
        )

    pairs: dict[str, Any] = {}
    for relative_path, source_path in source_files.items():
        snapshot_path = snapshot_files[relative_path]
        expected_sha, expected_size = _snapshot_record(
            inventory[relative_path], relative_path
        )
        source_identity = identity(source_path)
        snapshot_identity = identity(snapshot_path)
        for label, observed in (
            ("source", source_identity),
            ("snapshot", snapshot_identity),
        ):
            if (
                observed["sha256"] != expected_sha
                or observed["size_bytes"] != expected_size
            ):
                raise BundleVerificationError(
                    f"{label} identity mismatch for snapshot document {relative_path}"
                )
        if source_path.read_bytes() != snapshot_path.read_bytes():
            raise BundleVerificationError(
                f"source/snapshot byte mismatch for {relative_path}"
            )
        pairs[relative_path] = {
            "sha256": expected_sha,
            "size_bytes": expected_size,
            "source_path": str(source_path),
            "snapshot_path": str(snapshot_path),
            "byte_identical": True,
        }

    inherited_source = final_v6 / INHERITED_PARENT_NAME
    inherited_copy = final_v7 / INHERITED_PARENT_NAME
    if (
        not inherited_copy.is_file()
        or inherited_copy.read_bytes() != inherited_source.read_bytes()
    ):
        raise BundleVerificationError(
            "final-v7 inherited parent_final_v5_receipt.json is not byte-identical "
            "to the final-v6 copy"
        )

    return {
        "status": "PASS",
        "authoritative_final_v6": authoritative_check,
        "copied_parent_receipt": identity(copied),
        "parent_copy_byte_identical": True,
        "inherited_parent_final_v5_receipt": identity(inherited_copy),
        "inherited_parent_final_v5_byte_identical": True,
        "snapshot_receipt": identity(snapshot_receipt_path),
        "source_snapshot_pairs": pairs,
        "source_snapshot_file_sets_identical": True,
        "rehash_mismatches": 0,
    }


def _single_identity_record(
    record: Any,
    receipt: Path,
    label: str,
) -> DeclaredIdentity:
    identities = list(
        _declared_identities({label: record}, receipt, ("components",))
    )
    if len(identities) != 1:
        raise BundleVerificationError(
            f"{label} must contain exactly one path/size/SHA-256 identity"
        )
    return identities[0]


def _integration_report_payload(
    integration_receipt: Path,
    category_details: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the uniquely declared integration results used by the report."""
    output_files = category_details.get("outputs", {}).get("files", [])
    candidates = [
        record
        for record in output_files
        if isinstance(record, dict)
        and Path(str(record.get("path", ""))).resolve().parent
        == integration_receipt.parent.resolve()
        and Path(str(record.get("path", ""))).name == "results.json"
    ]
    if len(candidates) != 1:
        raise BundleVerificationError(
            "integration outputs must declare exactly one sibling results.json"
        )
    results_identity = candidates[0]
    results_path = Path(str(results_identity["path"])).resolve()
    payload = _load_json(results_path)
    if str(payload.get("status", "")).upper() != "PASS":
        raise BundleVerificationError("integration results.json status is not PASS")
    report_contract = payload.get("report_contract")
    if not isinstance(report_contract, dict) or not report_contract:
        raise BundleVerificationError(
            "integration results.json omits a non-empty report_contract"
        )
    for section in ("claim_boundaries", "report_ready_sentences"):
        value = payload.get(section)
        if (
            not isinstance(value, dict)
            or not value
            or any(not isinstance(item, str) or not item for item in value.values())
        ):
            raise BundleVerificationError(
                f"integration results.json has malformed {section}"
            )
    return payload, results_identity


def _identity_section(
    payload: Mapping[str, Any],
    receipt: Path,
    canonical_name: str,
    aliases: Sequence[str],
) -> tuple[str, list[DeclaredIdentity]]:
    present = [name for name in aliases if name in payload]
    if len(present) != 1:
        raise BundleVerificationError(
            f"integration receipt must declare exactly one {canonical_name} section "
            f"from {tuple(aliases)}"
        )
    name = present[0]
    identities = list(_declared_identities(payload[name], receipt, (name,)))
    if not identities:
        raise BundleVerificationError(
            f"integration receipt {name} section declares no complete identities"
        )
    return name, identities


def _assert_component_receipt(
    component: Mapping[str, Any],
    integration_receipt: Path,
    label: str,
    recursively_verified: set[Path],
) -> tuple[DeclaredIdentity, dict[str, Any]]:
    declared = _single_identity_record(
        component.get("receipt"), integration_receipt, f"{label}.receipt"
    )
    _verify_declared_identity(declared)
    if declared.path not in recursively_verified:
        raise BundleVerificationError(
            f"{label} receipt was not followed recursively: {declared.path}"
        )
    payload = _load_json(declared.path)
    if str(payload.get("status", "")).upper() != "PASS":
        raise BundleVerificationError(f"{label} nested receipt status is not PASS")
    return declared, payload


def _csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise BundleVerificationError(f"invalid reader CSV {path}: {exc}") from exc
    if not rows or not rows[0] or any(not cell.strip() for cell in rows[0]):
        raise BundleVerificationError(f"reader CSV has an invalid header: {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows[1:]):
        raise BundleVerificationError(f"reader CSV has ragged rows: {path}")
    return [cell.strip() for cell in rows[0]], rows[1:]


def _verify_blank_reader_forms(scoring_form: Path, reviewer_info: Path) -> dict[str, Any]:
    scoring_header, scoring_rows = _csv_rows(scoring_form)
    if not scoring_rows:
        raise BundleVerificationError("reader scoring form has no case rows")
    normalized = [cell.casefold() for cell in scoring_header]
    identifiers = {
        index
        for index, name in enumerate(normalized)
        if name in _SCORING_IDENTIFIER_COLUMNS
    }
    if not identifiers:
        raise BundleVerificationError(
            "reader scoring form has no recognized prefilled identifier column"
        )
    case_values: list[str] = []
    for row_number, row in enumerate(scoring_rows, start=2):
        if any(row[index].strip() for index in range(len(row)) if index not in identifiers):
            raise BundleVerificationError(
                f"reader scoring form is not blank at data row {row_number}"
            )
        identifier_values = [row[index].strip() for index in identifiers]
        if not any(identifier_values):
            raise BundleVerificationError(
                f"reader scoring form lacks an identifier at data row {row_number}"
            )
        case_values.append("|".join(identifier_values))
    if len(case_values) != len(set(case_values)):
        raise BundleVerificationError("reader scoring form contains duplicate identifiers")

    _, reviewer_rows = _csv_rows(reviewer_info)
    if not reviewer_rows:
        raise BundleVerificationError("reviewer information form has no blank entry row")
    for row_number, row in enumerate(reviewer_rows, start=2):
        if any(cell.strip() for cell in row):
            raise BundleVerificationError(
                f"reviewer information form is not blank at data row {row_number}"
            )
    return {
        "status": "PASS",
        "scoring_form": identity(scoring_form),
        "case_rows": len(scoring_rows),
        "scored_cells_nonblank": 0,
        "reviewer_info_form": identity(reviewer_info),
        "reviewer_cells_nonblank": 0,
    }


def _verify_no_review_analysis(review_root: Path) -> dict[str, Any]:
    if not review_root.resolve().is_dir():
        raise BundleVerificationError(f"missing reviews/v5 root: {review_root.resolve()}")
    forbidden: list[str] = []
    files_checked = 0
    for path in sorted(review_root.resolve().rglob("*")):
        if not path.is_file():
            continue
        files_checked += 1
        relative = path.relative_to(review_root.resolve())
        lowered_parts = {part.casefold() for part in relative.parts[:-1]}
        if (
            path.name.casefold() in _FORBIDDEN_REVIEW_RESULT_NAMES
            or lowered_parts & _FORBIDDEN_REVIEW_RESULT_DIRS
        ):
            forbidden.append(relative.as_posix())
    if forbidden:
        raise BundleVerificationError(
            f"reviews/v5 contains an analysis result despite GENERATED_UNREAD: {forbidden}"
        )
    return {
        "status": "PASS",
        "files_checked": files_checked,
        "forbidden_analysis_results": 0,
    }


def verify_integration_component(paths: BundlePaths) -> dict[str, Any]:
    receipt_path = paths.integration_receipt.resolve()
    payload = _load_json(receipt_path)
    if str(payload.get("status", "")).upper() != "PASS":
        raise BundleVerificationError("final-v7 integration receipt status is not PASS")
    if payload.get("append_only") is not True:
        raise BundleVerificationError(
            "final-v7 integration receipt must declare append_only true"
        )

    category_details: dict[str, Any] = {}
    for canonical, aliases in (
        ("inputs", ("inputs",)),
        ("outputs", ("outputs", "artifacts")),
        ("code", ("code", "implementation")),
    ):
        section_name, declared = _identity_section(
            payload, receipt_path, canonical, aliases
        )
        category_details[canonical] = {
            "declared_section": section_name,
            "identity_count": len(declared),
            "files": [_verify_declared_identity(item) for item in declared],
        }
    integration_results, integration_results_identity = _integration_report_payload(
        receipt_path, category_details
    )

    components = payload.get("components")
    if not isinstance(components, dict) or set(components) != set(COMPONENT_NAMES):
        raise BundleVerificationError(
            f"integration components must be exactly {COMPONENT_NAMES}"
        )
    e2f = components["e2f_v3"]
    review = components["reviews_v5"]
    if not isinstance(e2f, dict) or not isinstance(review, dict):
        raise BundleVerificationError("integration component metadata must be objects")

    tree = verify_receipt_tree(receipt_path)
    recursively_verified = {
        Path(path).resolve() for path in tree["recursively_verified_receipts"]
    }


    if str(e2f.get("integrity_status", "")).upper() != "PASS":
        raise BundleVerificationError("E2f-v3 integrity_status must be PASS")
    if str(e2f.get("scientific_status", "")).upper() != "EXECUTED":
        raise BundleVerificationError("E2f-v3 scientific_status must be EXECUTED")
    e2f_receipt, _ = _assert_component_receipt(
        e2f, receipt_path, "E2f-v3", recursively_verified
    )

    if str(review.get("integrity_status", "")).upper() != "PASS":
        raise BundleVerificationError("reviews/v5 integrity_status must be PASS")
    if str(review.get("scientific_status", "")).upper() != "GENERATED_UNREAD":
        raise BundleVerificationError(
            "reviews/v5 scientific_status must be GENERATED_UNREAD"
        )
    if review.get("analysis_executed") is not False:
        raise BundleVerificationError("reviews/v5 analysis_executed must be false")
    if review.get("unblinding_performed") is not False:
        raise BundleVerificationError("reviews/v5 unblinding_performed must be false")
    if "analysis_result" not in review or review.get("analysis_result") is not None:
        raise BundleVerificationError("reviews/v5 analysis_result must be explicit null")

    review_root_raw = review.get("review_root")
    repository_root = paths.final_v6.resolve().parents[1]
    if not _root_reference_matches(
        review_root_raw, paths.reviews_v5, receipt_path, repository_root
    ):
        raise BundleVerificationError(
            "reviews/v5 review_root does not match the configured review root"
        )
    review_root = paths.reviews_v5.resolve()
    review_receipt, _ = _assert_component_receipt(
        review, receipt_path, "reviews/v5", recursively_verified
    )
    scoring = _single_identity_record(
        review.get("reader_scoring_form"), receipt_path, "reviews_v5.reader_scoring_form"
    )
    reviewer = _single_identity_record(
        review.get("reviewer_info_form"), receipt_path, "reviews_v5.reviewer_info_form"
    )
    for label, declared in (("reader scoring", scoring), ("reviewer info", reviewer)):
        _verify_declared_identity(declared)
        if not declared.path.is_relative_to(review_root):
            raise BundleVerificationError(
                f"{label} form is outside configured reviews/v5 root: {declared.path}"
            )

    blank_forms = _verify_blank_reader_forms(scoring.path, reviewer.path)
    no_analysis = _verify_no_review_analysis(review_root)
    expected_component_states = {
        "e2f_v3": {
            "integrity_status": "PASS",
            "scientific_status": "EXECUTED",
        },
        "reviews_v5": {
            "integrity_status": "PASS",
            "scientific_status": "GENERATED_UNREAD",
            "analysis_executed": False,
            "unblinding_performed": False,
            "analysis_result": None,
        },
    }
    if integration_results.get("component_states") != expected_component_states:
        raise BundleVerificationError(
            "integration results component_states disagree with the sealed receipt"
        )
    report_contract = integration_results["report_contract"]
    if report_contract.get("component_states") != expected_component_states:
        raise BundleVerificationError(
            "integration report_contract component_states disagree with the sealed receipt"
        )
    return {
        **tree,
        "identity_categories": category_details,
        "integration_results": integration_results_identity,
        "report_contract": report_contract,
        "claim_boundaries": integration_results["claim_boundaries"],
        "report_ready_sentences": integration_results["report_ready_sentences"],
        "scientific_states": {
            "e2f_v3": {
                "integrity_status": "PASS",
                "scientific_status": "EXECUTED",
                "receipt": identity(e2f_receipt.path),
            },
            "reviews_v5": {
                "integrity_status": "PASS",
                "scientific_status": "GENERATED_UNREAD",
                "analysis_executed": False,
                "unblinding_performed": False,
                "analysis_result": None,
                "receipt": identity(review_receipt.path),
                "blank_reader_forms": blank_forms,
                "no_analysis_result": no_analysis,
            },
        },
    }


def verify_v5_pre_read_addendum(
    paths: BundlePaths,
    integration: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the unread secondary addendum without treating it as a result."""
    root = paths.v5_pre_read_addendum.resolve()
    receipt_path = root / V5_PRE_READ_ADDENDUM_RECEIPT.name
    if not root.is_dir() or not receipt_path.is_file():
        raise BundleVerificationError(
            f"missing v5 pre-read nuisance addendum or receipt: {root}"
        )
    payload = _load_json(receipt_path)
    if payload.get("schema_version") != 1:
        raise BundleVerificationError("v5 pre-read addendum schema_version must be 1")
    expected_state = {
        "status": "PASS",
        "problems": [],
        "scientific_status": "GENERATED_UNREAD_SECONDARY_ADDENDUM",
        "analysis_executed": False,
        "unblinding_performed": False,
        "analysis_result": None,
        "confirmatory_role": "NONE_SECONDARY_ROBUSTNESS_ONLY",
        "parent_primary_unchanged": True,
    }
    for key, expected in expected_state.items():
        if payload.get(key) != expected:
            raise BundleVerificationError(
                f"v5 pre-read addendum {key} differs from {expected!r}"
            )
    if payload.get("addendum_id") != root.name:
        raise BundleVerificationError("v5 pre-read addendum ID/root mismatch")

    outputs_raw = payload.get("outputs")
    if not isinstance(outputs_raw, list) or len(outputs_raw) != 6:
        raise BundleVerificationError(
            "v5 pre-read addendum must declare exactly six frozen outputs"
        )
    output_records = [
        _single_identity_record(record, receipt_path, f"addendum.outputs[{index}]")
        for index, record in enumerate(outputs_raw)
    ]
    output_identities = [_verify_declared_identity(record) for record in output_records]
    expected_files = {receipt_path.resolve(), *(record.path for record in output_records)}
    observed_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if observed_files != expected_files:
        raise BundleVerificationError(
            "v5 pre-read addendum file set differs from receipt outputs plus receipt"
        )
    if any(path.is_symlink() for path in root.rglob("*")):
        raise BundleVerificationError("v5 pre-read addendum contains a symlink")
    if any(record.path.parent != root for record in output_records):
        raise BundleVerificationError(
            "v5 pre-read addendum output is outside its sealed root"
        )

    named_outputs = {
        "analyzer": "analyze_nuisance.py",
        "sealer": "seal_addendum.py",
        "plan": "ANALYSIS_PLAN.md",
        "readme": "README.md",
        "covariate_manifest": "covariate_manifest.csv",
        "design_constants": "design_constants.json",
    }
    by_path = {record.path: record for record in output_records}
    for field, filename in named_outputs.items():
        declared = _single_identity_record(
            payload.get(field), receipt_path, f"addendum.{field}"
        )
        _verify_declared_identity(declared)
        if declared.path != (root / filename).resolve() or declared.path not in by_path:
            raise BundleVerificationError(
                f"v5 pre-read addendum {field} does not match its frozen output"
            )

    parent = _single_identity_record(
        payload.get("parent_v5_receipt"), receipt_path, "addendum.parent_v5_receipt"
    )
    _verify_declared_identity(parent)
    integrated_parent = integration["scientific_states"]["reviews_v5"]["receipt"]
    if identity(parent.path) != integrated_parent:
        raise BundleVerificationError(
            "v5 pre-read addendum parent receipt differs from integrated reviews/v5"
        )

    frozen_raw = payload.get("frozen_inputs")
    if not isinstance(frozen_raw, list) or len(frozen_raw) != 4:
        raise BundleVerificationError(
            "v5 pre-read addendum must declare exactly four frozen inputs"
        )
    frozen = [
        _single_identity_record(record, receipt_path, f"addendum.frozen_inputs[{index}]")
        for index, record in enumerate(frozen_raw)
    ]
    frozen_identities = [_verify_declared_identity(record) for record in frozen]
    if len({record.path for record in frozen}) != 4:
        raise BundleVerificationError(
            "v5 pre-read addendum frozen inputs are not four unique files"
        )
    named_frozen_fields = (
        "parent_v5_receipt",
        "parent_v5_frozen_analyzer",
        "v5_case_key",
        "development_manifest",
    )
    named_frozen = {
        field: _single_identity_record(
            payload.get(field), receipt_path, f"addendum.{field}"
        )
        for field in named_frozen_fields
    }
    for record in named_frozen.values():
        _verify_declared_identity(record)
    def frozen_key(record: DeclaredIdentity) -> tuple[Path, int, str]:
        return record.path, record.size_bytes, record.sha256

    if {frozen_key(record) for record in named_frozen.values()} != {
        frozen_key(record) for record in frozen
    }:
        raise BundleVerificationError(
            "v5 pre-read addendum named frozen inputs differ from frozen_inputs"
        )
    repository_root = paths.final_v6.resolve().parents[1]
    expected_named_paths = {
        "parent_v5_receipt": parent.path,
        "parent_v5_frozen_analyzer": (
            repository_root / "tools" / "analyze_reviews_v5.py"
        ).resolve(),
        "v5_case_key": (
            paths.reviews_v5 / "KEYS_DO_NOT_DISTRIBUTE" / "case_key.csv"
        ).resolve(),
    }
    for field, expected_path in expected_named_paths.items():
        if named_frozen[field].path != expected_path:
            raise BundleVerificationError(
                f"v5 pre-read addendum {field} path differs from the frozen parent"
            )
    tests_raw = payload.get("tests")
    if not isinstance(tests_raw, list) or len(tests_raw) != 1:
        raise BundleVerificationError(
            "v5 pre-read addendum must bind exactly one focused test file"
        )
    test_record = _single_identity_record(
        tests_raw[0], receipt_path, "addendum.tests[0]"
    )
    test_identity = _verify_declared_identity(test_record)

    design_constants_path = by_path[(root / "design_constants.json").resolve()].path
    design_constants = _load_json(design_constants_path)
    expected_blocks = [
        f"{cohort}|{kras}"
        for cohort in ("CPTAC", "RIH", "SurGen", "TCGA")
        for kras in ("mutant", "wild_type")
    ]
    sampling_counts = design_constants.get("sampling_cell_counts")
    if (
        design_constants.get("schema_version") != 1
        or design_constants.get("addendum_id") != root.name
        or design_constants.get("case_count") != 60
        or design_constants.get("analysis_blocks") != expected_blocks
        or design_constants.get("reference_block") != expected_blocks[0]
        or not isinstance(sampling_counts, dict)
        or len(sampling_counts) != 24
        or not all(isinstance(value, int) for value in sampling_counts.values())
        or sum(sampling_counts.values()) != 60
        or set(sampling_counts.values()) != {2, 3}
        or design_constants.get("area_rank_tertile_counts")
        != {"largest_20": 20, "middle_20": 20, "smallest_20": 20}
    ):
        raise BundleVerificationError(
            "v5 pre-read addendum design_constants census/design mismatch"
        )
    scaling = design_constants.get("scaling")
    if not isinstance(scaling, dict) or not scaling:
        raise BundleVerificationError(
            "v5 pre-read addendum design_constants scaling is absent"
        )
    for key, value in scaling.items():
        numeric = float(value)
        if not math.isfinite(numeric) or ("sd_" in key and numeric <= 0):
            raise BundleVerificationError(
                f"v5 pre-read addendum invalid frozen scaling constant: {key}"
            )

    bootstrap = payload.get("bootstrap_contract")
    missingness = payload.get("missingness_contract")
    interpretation = payload.get("interpretation_contract")
    model = payload.get("model_contract")
    contract_pairs = {
        "model": (model, design_constants.get("model")),
        "bootstrap": (bootstrap, design_constants.get("bootstrap")),
        "missingness": (missingness, design_constants.get("missingness")),
        "interpretation": (interpretation, design_constants.get("interpretation")),
    }
    for label, (declared, frozen_value) in contract_pairs.items():
        if not isinstance(declared, dict) or declared != frozen_value:
            raise BundleVerificationError(
                f"v5 pre-read addendum {label} contract differs from design_constants"
            )
    if (
        bootstrap.get("draws"),
        bootstrap.get("minimum_valid_draws"),
        bootstrap.get("seed"),
    ) != (2000, 1900, 20260829):
        raise BundleVerificationError("v5 pre-read addendum bootstrap contract mismatch")
    if interpretation.get("confirmatory_gate") is not False:
        raise BundleVerificationError(
            "v5 pre-read addendum must declare no confirmatory gate"
        )
    if model.get("link") != "cumulative_logit_proportional_odds":
        raise BundleVerificationError("v5 pre-read addendum model contract mismatch")

    return {
        "status": "PASS",
        "scientific_status": expected_state["scientific_status"],
        "analysis_executed": False,
        "unblinding_performed": False,
        "analysis_result": None,
        "confirmatory_role": expected_state["confirmatory_role"],
        "parent_primary_unchanged": True,
        "receipt": identity(receipt_path),
        "outputs": output_identities,
        "frozen_inputs": frozen_identities,
        "focused_test": test_identity,
        "bootstrap_contract": bootstrap,
        "missingness_contract": missingness,
        "model_contract": model,
        "interpretation_contract": interpretation,
        "design_constants": identity(design_constants_path),
    }



def _markdown_identity_without_placeholders(
    path: Path,
    authoritative_final_v6_path: Path,
) -> dict[str, Any]:
    observed = identity(path)
    if observed["size_bytes"] == 0:
        raise BundleVerificationError(f"empty final-v7 Markdown document: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise BundleVerificationError(
            f"final-v7 Markdown is not UTF-8: {path}"
        ) from exc
    authoritative_bytes = authoritative_final_v6_path.resolve().read_bytes()
    occurrence_count = path.resolve().read_bytes().count(authoritative_bytes)
    if occurrence_count != 1:
        raise BundleVerificationError(
            f"{path} must contain the complete authoritative final-v6 Markdown "
            f"byte sequence unchanged exactly once; observed {occurrence_count}"
        )
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern in _PLACEHOLDER_PATTERNS:
            match = pattern.search(line)
            if match is not None:
                raise BundleVerificationError(
                    f"placeholder {match.group(0)!r} in {path}:{line_number}"
                )
    return {
        **observed,
        "authoritative_final_v6_document": identity(authoritative_final_v6_path),
        "authoritative_final_v6_byte_sequence_occurrences": occurrence_count,
        "authoritative_final_v6_byte_sequence_unchanged": True,
    }


def _document_regions(path: Path, authoritative_final_v6_path: Path) -> tuple[str, str]:
    """Return UTF-8 text before and after the unique inherited final-v6 bytes."""
    payload = path.resolve().read_bytes()
    inherited = authoritative_final_v6_path.resolve().read_bytes()
    offset = payload.find(inherited)
    if offset < 0 or payload.find(inherited, offset + 1) >= 0:
        raise BundleVerificationError(
            f"cannot split {path} around a unique inherited final-v6 document"
        )
    try:
        return (
            payload[:offset].decode("utf-8"),
            payload[offset + len(inherited) :].decode("utf-8"),
        )
    except UnicodeDecodeError as exc:
        raise BundleVerificationError(f"non-UTF-8 final-v7 document region: {path}") from exc


def _canonical_report_contract(contract: Mapping[str, Any]) -> str:
    return json.dumps(contract, indent=2, sort_keys=True, allow_nan=False)


def _format_report_interval(record: Mapping[str, Any], *, signed: bool = False) -> str:
    point = float(record["point"])
    interval = record["ci95"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise BundleVerificationError("report-contract metric has an invalid ci95")
    low, high = (float(interval[0]), float(interval[1]))
    if not all(math.isfinite(value) for value in (point, low, high)):
        raise BundleVerificationError("report-contract metric is non-finite")
    spec = "+.4f" if signed else ".4f"
    return f"{point:{spec}} [{low:{spec}}, {high:{spec}}]"


def _expected_numeric_fragments(contract: Mapping[str, Any]) -> list[str]:
    """Return the human-table fragments that must match the machine contract."""
    try:
        e2f = contract["e2f_v3"]
        primary = e2f["primary_metrics"]
        fragments: list[str] = []
        for population in (*sorted(primary["per_cohort"]), "macro"):
            block = (
                primary["macro"]
                if population == "macro"
                else primary["per_cohort"][population]
            )
            procedures = block.get("procedures", block)
            contrasts = block.get("contrasts", block)
            for procedure in ("native", "adapted", "platt"):
                fragments.append(_format_report_interval(procedures[procedure]["auroc"]))
            fragments.append(
                _format_report_interval(
                    contrasts["adapted_minus_native"]["auroc"], signed=True
                )
            )

        macro = primary["macro"]
        macro_procedures = macro["procedures"]
        macro_contrasts = macro["contrasts"]
        for metric in ("log_loss", "brier"):
            for procedure in ("native", "adapted", "platt"):
                fragments.append(
                    _format_report_interval(macro_procedures[procedure][metric])
                )
            for contrast in ("adapted_minus_native", "platt_minus_native"):
                fragments.append(
                    _format_report_interval(macro_contrasts[contrast][metric], signed=True)
                )
        for cohort in sorted(primary["per_cohort"]):
            block = primary["per_cohort"][cohort]["adapted_minus_native"]
            for metric in ("log_loss", "brier"):
                fragments.append(_format_report_interval(block[metric], signed=True))

        for seed in sorted(e2f["outer_fold_layouts"], key=int):
            layout = e2f["outer_fold_layouts"][seed]
            fragments.append(_format_report_interval(layout["macro_adapted_auroc"]))
            fragments.append(
                _format_report_interval(
                    layout["macro_adapted_minus_native_auroc"], signed=True
                )
            )

        for support in sorted(e2f["label_efficiency_curve"], key=int):
            row = e2f["label_efficiency_curve"][support]
            cohort_delta = row["per_cohort_delta_auroc_mean"]
            log_loss = row["per_cohort_log_loss_mean"]
            brier = row["per_cohort_brier_mean"]
            fragments.append(
                " | ".join(
                    [
                        str(int(row["requested_support"])),
                        f"{float(cohort_delta['RIH']):+.4f}",
                        f"{float(cohort_delta['SurGen']):+.4f}",
                        f"{float(row['macro_delta_auroc_mean']):+.4f} "
                        f"({float(row['macro_delta_auroc_sd']):.4f})",
                        f"{float(log_loss['RIH']['adapted']):.4f} / "
                        f"{float(brier['RIH']['adapted']):.4f}",
                        f"{float(log_loss['SurGen']['adapted']):.4f} / "
                        f"{float(brier['SurGen']['adapted']):.4f}",
                    ]
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleVerificationError(
            f"report_contract lacks the report-facing metric schema: {exc}"
        ) from exc
    return fragments


def _verify_report_binding(
    paths: BundlePaths,
    integration: Mapping[str, Any],
    addendum: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind report states, numeric tables, and claim boundaries to integration."""
    contract = integration.get("report_contract")
    if not isinstance(contract, dict):
        raise BundleVerificationError("integration verification lacks report_contract")
    try:
        primary = contract["e2f_v3"]["primary_metrics"]
        summary = contract["e2f_v3"]["outer_fold_sensitivity_summary"]
        layouts = contract["e2f_v3"]["outer_fold_layouts"]
    except (KeyError, TypeError) as exc:
        raise BundleVerificationError(f"malformed report_contract state schema: {exc}") from exc
    n_layouts = int(summary["n_layouts"])
    n_gate_pass = int(summary["n_gate_pass"])
    incremental_passes = sum(
        bool(layout["incremental_improvement_established"]["pass"])
        for layout in layouts.values()
    )
    fixed_pass = bool(primary["fixed_gate_adapted"]["pass"])
    incremental_pass = bool(primary["incremental_improvement_established"]["pass"])
    dynamic_prefix_tokens = (
        "passed the prespecified absolute discrimination gate"
        if fixed_pass
        else "did not pass the prespecified absolute discrimination gate",
        "incremental improvement over the native model was established"
        if incremental_pass
        else "incremental improvement over the native model was not established",
        f"absolute gate passed in {n_gate_pass} of {n_layouts} declared fold layouts",
        (
            f"incremental improvement passed in {incremental_passes} of {n_layouts} "
            "declared fold layouts"
            if incremental_passes
            else f"incremental improvement failed in all {n_layouts} declared fold layouts"
        ),
    )

    regions: dict[str, tuple[str, str]] = {}
    for filename in REPORT_DOCUMENTS:
        prefix, appendix = _document_regions(
            paths.final_v7 / filename, paths.final_v6 / filename
        )
        for token in (*_REQUIRED_PREFIX_TOKENS, *dynamic_prefix_tokens):
            if token not in prefix:
                raise BundleVerificationError(
                    f"final-v7 controlling prefix for {filename} lacks {token!r}"
                )
        regions[filename] = (prefix, appendix)

    results_appendix = regions["Results.md"][1]
    if results_appendix.count(_REPORT_CONTRACT_START) != 1 or results_appendix.count(
        _REPORT_CONTRACT_END
    ) != 1:
        raise BundleVerificationError(
            "Results.md must contain exactly one canonical FINAL_V7_REPORT_CONTRACT block"
        )
    contract_text = results_appendix.split(_REPORT_CONTRACT_START, 1)[1].split(
        _REPORT_CONTRACT_END, 1
    )[0]
    canonical = _canonical_report_contract(contract)
    if contract_text != canonical:
        raise BundleVerificationError(
            "Results.md FINAL_V7_REPORT_CONTRACT does not exactly match integration"
        )
    try:
        parsed_contract = json.loads(contract_text)
    except json.JSONDecodeError as exc:
        raise BundleVerificationError(
            f"Results.md FINAL_V7_REPORT_CONTRACT is invalid JSON: {exc}"
        ) from exc
    if parsed_contract != contract:
        raise BundleVerificationError(
            "Results.md parsed report contract differs from integration"
        )

    required_prose = [
        *integration.get("claim_boundaries", {}).values(),
        *integration.get("report_ready_sentences", {}).values(),
        *_expected_numeric_fragments(contract),
    ]
    for fragment in required_prose:
        if fragment not in results_appendix:
            raise BundleVerificationError(
                f"Results.md appendix is not bound to integration fragment {fragment!r}"
            )
    for token in _REQUIRED_ADDENDUM_RESULTS_TOKENS:
        if token not in results_appendix:
            raise BundleVerificationError(
                f"Results.md appendix lacks unread addendum state {token!r}"
            )

    audit_appendix = regions["Audit.md"][1]
    for correction in _REQUIRED_AUDIT_CORRECTIONS:
        if correction not in audit_appendix:
            raise BundleVerificationError(
                f"Audit.md appendix lacks controlling correction {correction!r}"
            )
    addendum_receipt_path = str(paths.v5_pre_read_addendum.resolve() / "ADDENDUM_RECEIPT.json")
    for token in (*_REQUIRED_ADDENDUM_AUDIT_TOKENS, addendum_receipt_path):
        if token not in audit_appendix:
            raise BundleVerificationError(
                f"Audit.md appendix lacks unread addendum binding {token!r}"
            )

    receipt_identities = {
        "E2f-v3 receipt": integration["scientific_states"]["e2f_v3"]["receipt"],
        "reviews/v5 receipt": integration["scientific_states"]["reviews_v5"]["receipt"],
        "reviews/v5 pre-read nuisance addendum receipt": addendum["receipt"],
        "integration receipt": identity(paths.integration_receipt),
        "integration results": integration["integration_results"],
    }
    output_files = integration["identity_categories"]["outputs"]["files"]
    verification = [
        item for item in output_files if Path(str(item.get("path", ""))).name == "verification.json"
    ]
    if len(verification) != 1:
        raise BundleVerificationError(
            "integration outputs must declare exactly one verification.json"
        )
    receipt_identities["integration verification"] = verification[0]
    for label, record in receipt_identities.items():
        digest = str(record["sha256"])
        size = int(record["size_bytes"])
        if digest not in audit_appendix or (
            str(size) not in audit_appendix and f"{size:,}" not in audit_appendix
        ):
            raise BundleVerificationError(
                f"Audit.md does not record the sealed {label} identity"
            )

    contract_sha256 = hashlib.sha256((canonical + "\n").encode("utf-8")).hexdigest()
    return {
        "status": "PASS",
        "controlling_prefixes": "3/3 PASS",
        "report_contract": {
            "sha256": contract_sha256,
            "canonical_json_size_bytes": len((canonical + "\n").encode("utf-8")),
            "exact_match": True,
        },
        "claim_boundary_strings": len(integration["claim_boundaries"]),
        "report_ready_sentences": len(integration["report_ready_sentences"]),
        "numeric_fragments": len(_expected_numeric_fragments(contract)),
        "audit_semantic_corrections": len(_REQUIRED_AUDIT_CORRECTIONS),
        "addendum_state_tokens": len(_REQUIRED_ADDENDUM_AUDIT_TOKENS),
        "audit_component_identities": len(receipt_identities),
    }


def _validate_locations(paths: BundlePaths) -> None:
    final_v6 = paths.final_v6.resolve()
    snapshot = paths.snapshot_root.resolve()
    final_v7 = paths.final_v7.resolve()
    if len({final_v6, snapshot, final_v7}) != 3:
        raise BundleVerificationError(
            "final-v6, snapshot, and final-v7 roots must be distinct"
        )
    destination = paths.destination.resolve()
    if destination.parent != final_v7 or destination.name != FINAL_RECEIPT_NAME:
        raise BundleVerificationError(
            "final receipt destination must be report_bundle_receipt.json directly under final-v7"
        )
    if not _looks_like_receipt(paths.integration_receipt.resolve()):
        raise BundleVerificationError("unrecognized integration receipt filename")


def verify_bundle(paths: BundlePaths) -> dict[str, Any]:
    """Verify the complete final-v7 seal without writing any file."""
    _validate_locations(paths)
    parent_snapshot = verify_parent_and_snapshot(paths)
    integration = verify_integration_component(paths)
    addendum = verify_v5_pre_read_addendum(paths, integration)

    documents = {
        filename: _markdown_identity_without_placeholders(
            paths.final_v7 / filename,
            paths.final_v6 / filename,
        )
        for filename in REPORT_DOCUMENTS
    }
    report_binding = _verify_report_binding(paths, integration, addendum)
    implementation: dict[str, Any] = {}
    if paths.verifier_code is not None:
        implementation["verifier_code"] = identity(paths.verifier_code)
    if paths.verifier_test is not None:
        implementation["focused_tests"] = identity(paths.verifier_test)

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    return {
        "schema_version": 1,
        "status": "PASS",
        "bundle_status": (
            "FINAL_V7_REPORTS_REHASHED__E2F_V3_EXECUTED_PASS__"
            "REVIEWS_V5_INTEGRITY_PASS_SCIENTIFIC_GENERATED_UNREAD__"
            "V5_PRE_READ_SECONDARY_ADDENDUM_PASS_UNREAD__"
            "PARENT_V6_AND_SNAPSHOT_IDENTICAL"
        ),
        "created_at_utc": now,
        "bundle_root": str(paths.final_v7.resolve()),
        "append_only": True,
        "documents": documents,
        "report_binding": report_binding,
        "parent_final_v6_and_snapshot": parent_snapshot,
        "integration_component": integration,
        "v5_pre_read_nuisance_addendum": addendum,
        "bound_receipts": {
            "authoritative_final_v6": identity(
                paths.final_v6 / FINAL_RECEIPT_NAME
            ),
            "parent_final_v6_copy": identity(
                paths.final_v7 / PARENT_COPY_NAME
            ),
            "inherited_parent_final_v5": identity(
                paths.final_v7 / INHERITED_PARENT_NAME
            ),
            "pre_v7_snapshot": identity(paths.snapshot_receipt),
            "final_v7_integration": identity(paths.integration_receipt),
            "e2f_v3": integration["scientific_states"]["e2f_v3"]["receipt"],
            "reviews_v5": integration["scientific_states"]["reviews_v5"]["receipt"],
            "reviews_v5_pre_read_nuisance_addendum": addendum["receipt"],
        },
        "verification_implementation": implementation,
        "independent_checks": {
            "final_v6_markdown_rehash": "3/3 PASS",
            "final_v6_parent_copy_byte_identity": "PASS",
            "pre_v7_snapshot_full_tree_byte_identity": "PASS",
            "integration_inputs_outputs_code_recursive_rehash": "PASS",
            "e2f_v3_executed_integrity": "PASS",
            "reviews_v5_packet_integrity": "PASS",
            "reviews_v5_scientific_status": "GENERATED_UNREAD",
            "reviews_v5_blank_reader_forms": "PASS",
            "reviews_v5_analysis_result_absent": "PASS",
            "reviews_v5_pre_read_nuisance_addendum_integrity": "PASS",
            "reviews_v5_pre_read_nuisance_addendum_status": (
                "GENERATED_UNREAD_SECONDARY_ADDENDUM"
            ),
            "reviews_v5_pre_read_nuisance_addendum_confirmatory_role": (
                "NONE_SECONDARY_ROBUSTNESS_ONLY"
            ),
            "final_v7_document_hashing_and_placeholder_scan": "3/3 PASS",
            "final_v6_markdown_byte_sequence_inherited_exactly_once": "3/3 PASS",
            "final_v7_report_contract_exact_match": "PASS",
            "final_v7_claim_and_numeric_binding": "PASS",
            "final_v7_semantic_correction_binding": "PASS",
            "rehash_mismatches": 0,
        },
        "seal_protocol": {
            "receipt_written_last": True,
            "publication": "exclusive atomic hard-link; overwrite refused",
            "self_hash_excluded": True,
        },
        "immutability_note": (
            "This receipt does not hash itself. Any later byte change to a bound "
            "report, parent/snapshot receipt, integration receipt, or recursively "
            "declared input/output/code artifact, or pre-read addendum invalidates "
            "final-v7 and requires "
            "a new append-only report directory. GENERATED_UNREAD is packet integrity, "
            "not a pathology result."
        ),
    }


def write_json_once_atomic(destination: Path, payload: Mapping[str, Any]) -> None:
    """Publish the receipt once and atomically; never expose an overwrite path."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite final receipt: {destination}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(raw_temporary)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def default_paths(args: argparse.Namespace) -> BundlePaths:
    return BundlePaths(
        final_v6=args.final_v6,
        snapshot_root=args.snapshot_root,
        snapshot_receipt=args.snapshot_receipt,
        final_v7=args.final_v7,
        integration_receipt=args.integration_receipt,
        reviews_v5=args.reviews_v5,
        v5_pre_read_addendum=args.v5_pre_read_addendum,
        destination=args.destination,
        verifier_code=Path(__file__).resolve(),
        verifier_test=REPO / "tests" / "test_final_v7_bundle_receipt.py",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-v6", type=Path, default=FINAL_V6)
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_ROOT)
    parser.add_argument("--snapshot-receipt", type=Path, default=SNAPSHOT_RECEIPT)
    parser.add_argument("--final-v7", type=Path, default=FINAL_V7)
    parser.add_argument(
        "--integration-receipt", type=Path, default=INTEGRATION_RECEIPT
    )
    parser.add_argument("--reviews-v5", type=Path, default=REVIEWS_V5)
    parser.add_argument(
        "--v5-pre-read-addendum", type=Path, default=V5_PRE_READ_ADDENDUM
    )
    parser.add_argument(
        "--destination", type=Path, default=FINAL_V7 / FINAL_RECEIPT_NAME
    )
    parser.add_argument(
        "--seal",
        action="store_true",
        help="write the receipt once after verification; omit for a read-only check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = default_paths(args)
    if args.seal and paths.destination.exists():
        raise FileExistsError(
            f"refusing to overwrite final receipt: {paths.destination.resolve()}"
        )
    payload = verify_bundle(paths)
    if args.seal:
        write_json_once_atomic(paths.destination, payload)
        print(f"PASS: sealed {paths.destination}")
    else:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "read-only; not sealed",
                    "destination": str(paths.destination),
                    "e2f_v3": "EXECUTED / integrity PASS",
                    "reviews_v5": "GENERATED_UNREAD / integrity PASS",
                    "recursively_verified_files": integration_count(payload),
                },
                indent=2,
            )
        )


def integration_count(payload: Mapping[str, Any]) -> int:
    """Return a compact CLI count without weakening the stored receipt detail."""
    integration = payload.get("integration_component")
    if not isinstance(integration, Mapping):
        return 0
    count = integration.get("unique_declared_files")
    return count if isinstance(count, int) else 0


if __name__ == "__main__":
    main()
