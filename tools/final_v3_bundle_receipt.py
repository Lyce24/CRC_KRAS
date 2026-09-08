#!/usr/bin/env python3
"""Verify and optionally seal the append-only final-v3 report bundle.

The verifier is deliberately independent of the component analysis launchers.
It understands their heterogeneous receipt layouts by recursively locating
declared SHA-256 identities, rehashing every declared file, and following JSON
receipt artifacts.  It also proves that the authoritative final-v2 bundle, its
copy in final-v3, and the pre-v3 byte snapshot are identical.

Running without ``--seal`` is read-only.  ``--seal`` writes exactly one
``report_bundle_receipt.json`` with exclusive, atomic creation and refuses an
existing destination.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FINAL_V2 = REPO / "reports" / "final_v2"
SNAPSHOT_ROOT = REPO / "reports" / "snapshots" / "final_v2_pre_v3_20260820"
SNAPSHOT_RECEIPT = (
    REPO / "reports" / "snapshots" / "final_v2_pre_v3_20260820.receipt.json"
)
FINAL_V3 = REPO / "reports" / "final_v3"
ADDITIONS = REPO / "reports" / "reruns" / "final_v3_additions_20260820"
REPORT_DOCUMENTS = ("Experimental_Setup.md", "Results.md", "Audit.md")
SNAPSHOT_DOCUMENTS = (*REPORT_DOCUMENTS, "report_bundle_receipt.json")
COMPONENT_NAMES = (
    "aim1_worklist",
    "aim2_operational",
    "aim3_actionability",
    "aim4_compressibility",
)
FOLLOW_RECEIPT_NAMES = {
    "receipt.json",
    "input_receipt.json",
    "inputs_receipt.json",
    "output_receipt.json",
    "completion_receipt.json",
    "verification_receipt.json",
}


class BundleVerificationError(RuntimeError):
    """A fail-closed bundle or lineage verification failure."""


@dataclass(frozen=True)
class BundlePaths:
    final_v2: Path
    snapshot_root: Path
    snapshot_receipt: Path
    final_v3: Path
    component_receipts: Mapping[str, Path]
    destination: Path
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
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
    # Supports keyed inventories such as {"result.json": {sha256, size_bytes}}
    # without mistaking semantic keys such as "fixed" or "source" for files.
    if "/" in candidate or "\\" in candidate or Path(candidate).suffix:
        return candidate
    return None


def _declared_identities(
    value: Any,
    receipt: Path,
    trail: tuple[str, ...] = (),
) -> Iterator[DeclaredIdentity]:
    if isinstance(value, dict):
        if "sha256" in value and "size_bytes" in value:
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
            if raw_path is not None:
                digest = value["sha256"]
                size = value["size_bytes"]
                if not isinstance(digest, str) or len(digest) != 64:
                    raise BundleVerificationError(
                        f"invalid SHA-256 at {receipt}:{'/'.join(trail)}"
                    )
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    raise BundleVerificationError(
                        f"invalid size_bytes at {receipt}:{'/'.join(trail)}"
                    )
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
    """Return true for preserved/superseded receipts that are not authoritative trees.

    Their receipt file identity is still rehashed.  We intentionally do not
    recurse through their obsolete path inventories, which may record staging
    locations and are retained only to document append-only lineage.
    """
    lineage_tokens = {"supersedes", "superseded", "non_authoritative", "preserved"}
    return any(token.lower() in lineage_tokens for token in declared.trail)


def verify_receipt_tree(receipt_path: Path) -> dict[str, Any]:
    """Recursively rehash all identities declared by one component receipt."""
    root_receipt = receipt_path.resolve()
    root = root_receipt.parent
    if not root_receipt.is_file():
        raise BundleVerificationError(f"missing component receipt: {root_receipt}")
    top = _load_json(root_receipt)
    component_status = str(top.get("status", "")).upper()
    if component_status not in {"PASS", "COMPLETE"}:
        raise BundleVerificationError(
            f"component receipt is not PASS/COMPLETE: {root_receipt}"
        )
    if top.get("append_only") is False:
        raise BundleVerificationError(f"component explicitly is not append-only: {root_receipt}")

    pending = [root_receipt]
    visited_receipts: set[Path] = set()
    verified: dict[tuple[Path, str, int], dict[str, Any]] = {}
    trails: list[dict[str, Any]] = []
    while pending:
        current = pending.pop()
        if current in visited_receipts:
            continue
        visited_receipts.add(current)
        payload = _load_json(current)
        for declared in _declared_identities(payload, current):
            key = (declared.path, declared.sha256, declared.size_bytes)
            if key not in verified:
                verified[key] = _verify_declared_identity(declared)
            trails.append(
                {
                    "source_receipt": str(current),
                    "trail": "/".join(declared.trail),
                    **verified[key],
                }
            )
            if (
                _looks_like_receipt(declared.path)
                and not _is_lineage_only_receipt_reference(declared)
                and declared.path not in visited_receipts
            ):
                pending.append(declared.path)

    files_inside_root = [
        item for item in verified.values() if Path(item["path"]).is_relative_to(root)
    ]
    if not files_inside_root:
        raise BundleVerificationError(
            f"component receipt declares no component-local artifacts: {root_receipt}"
        )
    return {
        "status": "PASS",
        "declared_component_status": component_status,
        "component_root": str(root),
        "receipt": identity(root_receipt),
        "append_only_declared": top.get("append_only"),
        "declared_identity_references": len(trails),
        "unique_declared_files": len(verified),
        "component_local_files": len(files_inside_root),
        "recursively_verified_receipts": [str(path) for path in sorted(visited_receipts)],
        "rehash_mismatches": 0,
    }


def verify_parent_and_snapshot(paths: BundlePaths) -> dict[str, Any]:
    authoritative = (paths.final_v2 / "report_bundle_receipt.json").resolve()
    copied = (paths.final_v3 / "parent_final_v2_receipt.json").resolve()
    snapshot_receipt_path = paths.snapshot_receipt.resolve()
    snapshot_root = paths.snapshot_root.resolve()

    parent = _load_json(authoritative)
    if str(parent.get("status", "")).upper() != "PASS" or parent.get("append_only") is not True:
        raise BundleVerificationError("authoritative final-v2 receipt is not append-only PASS")
    declared_parent_root = Path(str(parent.get("bundle_root", ""))).resolve()
    if declared_parent_root != paths.final_v2.resolve():
        raise BundleVerificationError(
            f"final-v2 receipt root mismatch: {declared_parent_root} != {paths.final_v2.resolve()}"
        )

    parent_identities = list(_declared_identities(parent, authoritative))
    declared_documents = {
        declared.path.name: declared
        for declared in parent_identities
        if declared.path.parent == paths.final_v2.resolve()
    }
    missing_parent_docs = set(REPORT_DOCUMENTS) - set(declared_documents)
    if missing_parent_docs:
        raise BundleVerificationError(
            f"final-v2 receipt omits documents: {sorted(missing_parent_docs)}"
        )
    for document in REPORT_DOCUMENTS:
        _verify_declared_identity(declared_documents[document])

    if not copied.is_file() or copied.read_bytes() != authoritative.read_bytes():
        raise BundleVerificationError(
            "final-v3 parent_final_v2_receipt.json is not byte-identical to final-v2 receipt"
        )

    snapshot = _load_json(snapshot_receipt_path)
    if str(snapshot.get("status", "")).upper() != "PASS" or snapshot.get("append_only") is not True:
        raise BundleVerificationError("pre-v3 snapshot receipt is not append-only PASS")
    if Path(str(snapshot.get("source_root", ""))).resolve() != paths.final_v2.resolve():
        raise BundleVerificationError("snapshot source_root does not identify final-v2")
    if Path(str(snapshot.get("snapshot_root", ""))).resolve() != snapshot_root:
        raise BundleVerificationError("snapshot_root does not match configured snapshot")
    documents = snapshot.get("documents")
    if not isinstance(documents, dict):
        raise BundleVerificationError("snapshot receipt has no documents inventory")
    missing_snapshot_docs = set(SNAPSHOT_DOCUMENTS) - set(documents)
    if missing_snapshot_docs:
        raise BundleVerificationError(
            f"snapshot receipt omits documents: {sorted(missing_snapshot_docs)}"
        )

    pairs: dict[str, Any] = {}
    for filename in SNAPSHOT_DOCUMENTS:
        declared = documents[filename]
        if not isinstance(declared, dict):
            raise BundleVerificationError(f"invalid snapshot identity for {filename}")
        expected_sha = declared.get("sha256")
        expected_size = declared.get("size_bytes")
        source_path = paths.final_v2 / filename
        snapshot_path = snapshot_root / filename
        source_identity = identity(source_path)
        snapshot_identity = identity(snapshot_path)
        for label, observed in (("source", source_identity), ("snapshot", snapshot_identity)):
            if observed["sha256"] != expected_sha or observed["size_bytes"] != expected_size:
                raise BundleVerificationError(
                    f"{label} identity mismatch for snapshot document {filename}"
                )
        if source_path.read_bytes() != snapshot_path.read_bytes():
            raise BundleVerificationError(f"source/snapshot byte mismatch for {filename}")
        pairs[filename] = {
            "sha256": expected_sha,
            "size_bytes": expected_size,
            "source_path": str(source_path.resolve()),
            "snapshot_path": str(snapshot_path.resolve()),
            "byte_identical": True,
        }
    return {
        "status": "PASS",
        "authoritative_receipt": identity(authoritative),
        "copied_parent_receipt": identity(copied),
        "parent_copy_byte_identical": True,
        "snapshot_receipt": identity(snapshot_receipt_path),
        "source_snapshot_pairs": pairs,
        "rehash_mismatches": 0,
    }


def _validate_component_locations(paths: BundlePaths) -> None:
    if set(paths.component_receipts) != set(COMPONENT_NAMES):
        raise BundleVerificationError(
            f"component set must be exactly {COMPONENT_NAMES}; got {tuple(paths.component_receipts)}"
        )
    roots = [path.resolve().parent for path in paths.component_receipts.values()]
    if len(set(roots)) != len(roots):
        raise BundleVerificationError("component roots must be distinct")
    final_root = paths.final_v3.resolve()
    if paths.destination.resolve().parent != final_root:
        raise BundleVerificationError("final receipt destination must be directly under final-v3")
    for name, receipt in paths.component_receipts.items():
        if receipt.name not in FOLLOW_RECEIPT_NAMES and "receipt" not in receipt.name.lower():
            raise BundleVerificationError(f"{name}: unrecognized component receipt filename")


def verify_bundle(paths: BundlePaths) -> dict[str, Any]:
    _validate_component_locations(paths)
    parent_snapshot = verify_parent_and_snapshot(paths)
    components = {
        name: verify_receipt_tree(receipt)
        for name, receipt in sorted(paths.component_receipts.items())
    }

    documents: dict[str, Any] = {}
    for filename in REPORT_DOCUMENTS:
        path = paths.final_v3 / filename
        observed = identity(path)
        if observed["size_bytes"] == 0:
            raise BundleVerificationError(f"empty final-v3 document: {path}")
        # Fail on binary/corrupt Markdown before sealing.
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise BundleVerificationError(f"final-v3 Markdown is not UTF-8: {path}") from exc
        documents[filename] = observed

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
            "FINAL_V3_REPORTS_AND_FOUR_ADDITIONS_REHASHED__PARENT_V2_AND_SNAPSHOT_IDENTICAL"
        ),
        "created_at_utc": now,
        "bundle_root": str(paths.final_v3.resolve()),
        "append_only": True,
        "documents": documents,
        "parent_final_v2_and_snapshot": parent_snapshot,
        "components": components,
        "bound_receipts": {
            "parent_final_v2": identity(paths.final_v3 / "parent_final_v2_receipt.json"),
            "pre_v3_snapshot": identity(paths.snapshot_receipt),
            **{
                name: identity(receipt)
                for name, receipt in sorted(paths.component_receipts.items())
            },
        },
        "verification_implementation": implementation,
        "independent_checks": {
            "final_v2_receipt_rehash": "PASS",
            "final_v2_parent_copy_byte_identity": "PASS",
            "pre_v3_snapshot_byte_identity": "PASS",
            "component_receipt_recursive_rehash": "4/4 PASS",
            "final_v3_document_hashing": "3/3 PASS",
            "rehash_mismatches": 0,
        },
        "immutability_note": (
            "This receipt does not hash itself. Any later byte change to a bound report, "
            "parent/snapshot receipt, component receipt, or recursively declared component "
            "artifact invalidates this bundle and requires a new append-only report directory."
        ),
    }


def write_json_once_atomic(destination: Path, payload: dict[str, Any]) -> None:
    """Atomically publish JSON without any overwrite path."""
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
        # Hard-link publication is atomic and fails with FileExistsError if a
        # concurrent process sealed the destination after the preflight check.
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
    components = {
        "aim1_worklist": args.aim1_receipt,
        "aim2_operational": args.aim2_receipt,
        "aim3_actionability": args.aim3_receipt,
        "aim4_compressibility": args.aim4_receipt,
    }
    return BundlePaths(
        final_v2=args.final_v2,
        snapshot_root=args.snapshot_root,
        snapshot_receipt=args.snapshot_receipt,
        final_v3=args.final_v3,
        component_receipts=components,
        destination=args.destination,
        verifier_code=Path(__file__).resolve(),
        verifier_test=REPO / "tests" / "test_final_v3_bundle_receipt.py",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-v2", type=Path, default=FINAL_V2)
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_ROOT)
    parser.add_argument("--snapshot-receipt", type=Path, default=SNAPSHOT_RECEIPT)
    parser.add_argument("--final-v3", type=Path, default=FINAL_V3)
    parser.add_argument(
        "--aim1-receipt", type=Path, default=ADDITIONS / "aim1_worklist" / "receipt.json"
    )
    parser.add_argument(
        "--aim2-receipt",
        type=Path,
        default=ADDITIONS / "aim2_operational_v3" / "receipt.json",
    )
    parser.add_argument(
        "--aim3-receipt", type=Path, default=ADDITIONS / "aim3_actionability" / "receipt.json"
    )
    parser.add_argument(
        "--aim4-receipt",
        type=Path,
        default=ADDITIONS / "aim4_compressibility_v2" / "completion_receipt.json",
    )
    parser.add_argument(
        "--destination", type=Path, default=FINAL_V3 / "report_bundle_receipt.json"
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
        raise FileExistsError(f"refusing to overwrite final receipt: {paths.destination}")
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
                    "components": sorted(payload["components"]),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
