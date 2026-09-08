#!/usr/bin/env python3
"""Scope-correct the immutable Aim-1 training recovery after downstream preparation.

Recovery-v1 correctly certified the completed training lineage, but its read-only
validator later recomputed a census of every non-recovery file below the campaign
root.  Seven governed, label-blind downstream preparation files therefore made
that validator fail even though none of the original 441 training artifacts had
changed.  This additive recovery-v2 certificate narrows the immutability claim to
the exact original training roster.

The tool never trains, scores, edits recovery-v1, or edits downstream artifacts.
It authenticates the exact recovery-v1 implementation and nine-file namespace,
extracts the original census from the pinned adjudication, rehashes every one of
the 441 records, and rejects every ungoverned file outside four exact namespaces.
The seven already-published downstream preparation files are immutable baseline
records.  Future regular, non-symlink files may be added only under
``downstream_v2`` and are deliberately outside the training claim.

Production workflow::

    .venv/bin/python tools/aim1_tcga_surgen_two_encoder_recovery_v2.py audit
    .venv/bin/python tools/aim1_tcga_surgen_two_encoder_recovery_v2.py certify --apply
    .venv/bin/python tools/aim1_tcga_surgen_two_encoder_recovery_v2.py verify

``plan`` and ``audit`` are read-only.  ``certify`` is an atomic, exactly-once
publication of the three-file ``recovery_v2`` namespace.  Callers that launch
work must use deep validation at a control-plane boundary.  The shallow API is
only for workers already covered by a fresh deep preflight receipt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import aim1_tcga_surgen_two_encoder_recovery as recovery_v1  # noqa: E402

campaign = recovery_v1.campaign
ContractError = campaign.ContractError

SCHEMA_VERSION = 1
RECOVERY = "aim1_tcga_surgen_two_encoder_scope_recovery_v2"
RECOVERY_DIRNAME = "recovery_v2"
RECOVERY_STATUS = "complete_and_certified_via_scoped_census_erratum"
DEFAULT_OUTPUT_ROOT = recovery_v1.DEFAULT_OUTPUT_ROOT
RECOVERY_V2_TERMINAL = Path("recovery_v2/receipts/training_complete_scoped.json")

AUTHORIZED_EXCLUDED_NAMESPACES = (
    "recovery_v1",
    "recovery_v2",
    "downstream",
    "downstream_v2",
)
MUTABLE_DELEGATED_NAMESPACES = ("downstream_v2",)

ORIGINAL_CENSUS = {
    "artifact_count": 441,
    "total_size_bytes": 989_858_771,
    "tree_sha256": "a2ffd5b61eaf65dc8d0c5df6a1b29867ffabc57f0af869120603ea37591ac261",
}
PREPARED_BASELINE = {
    "artifact_count": 7,
    "total_size_bytes": 95_802,
    "tree_sha256": "d6400d6004ad2c1a850c022db1cb694f08b0fac445cdbbc3d94400f1fc995621",
}

PINNED_V1_IMPLEMENTATION = {
    "controller": {
        "path": REPO / "tools/aim1_tcga_surgen_two_encoder_recovery.py",
        "sha256": "d47cbd54b584fb3d0dd5160ae0bdd1fe2aca79b53207f8c1b63e366957b66ae5",
        "size_bytes": 66_818,
    },
    "controller_test": {
        "path": REPO / "tests/test_aim1_tcga_surgen_two_encoder_recovery.py",
        "sha256": "2cf182aab9cb8a92d4416402024251a7b450adc11e0be7da4f24d0bfac6c2f24",
        "size_bytes": 27_047,
    },
}

PINNED_V1_NAMESPACE = {
    "contract_erratum.json": (
        "bf1d81786590707399f5a9ba59c84e70ee6e527704f6874b3925d030630434f0",
        4_080,
    ),
    "receipts/jobs/virchow2_full/seed42.json": (
        "5b5a72e6b9d7a5fafb3c939e82f0af82fed80d22fbc0abdada34793ff0da45c9",
        19_847,
    ),
    "receipts/jobs/virchow2_full/seed43.json": (
        "d0a6c1a2cd91721c92e31eac9872f7f6d769c41ff2bacef7179f4b2ed930bf76",
        19_848,
    ),
    "receipts/jobs/virchow2_full/seed44.json": (
        "8d5ae9b5b864da08689182c2c9c379090bfe48cc675cfec19969ae96a123074c",
        19_848,
    ),
    "receipts/jobs/virchow2_full/seed45.json": (
        "5111e1c959f832398a1b2825436661bba5c6ffd64918092f8474c353c2c52734",
        19_847,
    ),
    "receipts/jobs/virchow2_full/seed46.json": (
        "3bda142c272e387363be9f0abdb7b43859b66c459ccb78114ca0c52cbad467c7",
        19_846,
    ),
    "receipts/scheduler_recovery.json": (
        "98f38c4e9968da9ab19923748d4592f2c97415ad553d8adf1cf8576b4e0c2bb8",
        9_371,
    ),
    "receipts/training_complete_recovered.json": (
        "4c3a2f4626b8a66e37b37afd20b211bb2de8fa32901db49a86d123a572b13c86",
        23_089,
    ),
    "receipts/validator_adjudication.json": (
        "dbe3515f1c958627f3230827b7c8c9a2c0f30ee5306495ad44bc204a3be6b1f1",
        314_976,
    ),
}

PINNED_PREPARED_BASELINE = {
    "downstream/contract.json": (
        "a3a7329c7ae57c6d3c5b17cd275d196539559a01dab1e307f3cfe65dfd5361a9",
        27_484,
    ),
    "downstream/inputs/label_blind/cptac_primary.csv": (
        "135a454edbd94d2c27889946187faedf68312be0cbb6f93786987beae970eb5b",
        8_203,
    ),
    "downstream/inputs/label_blind/orion_cpht.csv": (
        "6acb9699c53944c1fa87b90a42e773eed830ef71594589de5d71005a1a31d024",
        3_059,
    ),
    "downstream/inputs/label_blind/rih_metastatic.csv": (
        "dec443d99e437e7cb3fc3297a4f98302081ba4455386e3b7d930ff142feb1e5c",
        5_177,
    ),
    "downstream/inputs/label_blind/rih_primary.csv": (
        "0c33a3562da87a9843a41a34e6693e37825538aa5be0c31971798943f010a06e",
        9_268,
    ),
    "downstream/inputs/label_blind/sr1482_metastatic.csv": (
        "90eb00ef0baf6a33fc4b423affbce33dde1d38068add59cc136f870f13cf6d15",
        6_772,
    ),
    "downstream/jobs/score_jobs.json": (
        "2385c72511069f6c1db7da2cd08a3b9b941500fc0cb2aa0232d922019d8bc7d9",
        35_839,
    ),
}

FIT_ACCOUNTING = dict(recovery_v1.FIT_ACCOUNTING)
EXECUTION_ACCOUNTING = dict(recovery_v1.EXECUTION_ACCOUNTING)
CONCURRENCY = {
    "maximum": 6,
    "observed_peak": 6,
    "witness_utc": "2026-08-25T01:10:34.730410+00:00",
}

TERMINAL_FIELDS = {
    "schema_version",
    "recovery",
    "status",
    "created_utc",
    "base_campaign",
    "scope_contract",
    "scope_adjudication",
    "recovery_implementation",
    "predecessor_v1_terminal",
    "training_scoped_census",
    "prepared_downstream_baseline",
    "namespace_policy",
    "fit_accounting",
    "execution_accounting",
    "concurrency",
    "certification_boundary",
}


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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


def recovery_v2_dir(root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return Path(root) / RECOVERY_DIRNAME


def scope_contract_path(root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return recovery_v2_dir(root) / "contract_scope_erratum.json"


def scope_adjudication_path(root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return recovery_v2_dir(root) / "receipts/scope_adjudication.json"


def scoped_terminal_path(root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return Path(root) / RECOVERY_V2_TERMINAL


def recovery_v2_write_inventory(root: Path = DEFAULT_OUTPUT_ROOT) -> tuple[Path, ...]:
    return (scope_contract_path(root), scope_adjudication_path(root), scoped_terminal_path(root))


def _assert_root(root: Path) -> Path:
    candidate = campaign.assert_safe_output_root(Path(root))
    if candidate.resolve() != Path(DEFAULT_OUTPUT_ROOT).resolve():
        raise ContractError(
            "Recovery-v2 is pinned to the affected production root; "
            f"expected={DEFAULT_OUTPUT_ROOT}, observed={candidate}"
        )
    return candidate


def _assert_path_no_symlink(path: Path, *, context: str) -> Path:
    path = Path(path)
    lexical = path.absolute()
    resolved = path.resolve(strict=False)
    if lexical != resolved:
        raise ContractError(f"{context} is non-normalized or traverses a symlink: {path}")
    cursor = path
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise ContractError(f"{context} contains a symlink: {cursor}")
        cursor = cursor.parent
    return resolved


def _expected_identity(path: Path, sha256: str, size_bytes: int) -> dict[str, Any]:
    _assert_path_no_symlink(path, context="Pinned artifact path")
    return campaign._expected_identity(path, sha256, size_bytes)


def _implementation() -> dict[str, dict[str, Any]]:
    return {
        "controller": _artifact(Path(__file__).resolve()),
        "controller_test": _artifact(
            REPO / "tests/test_aim1_tcga_surgen_two_encoder_recovery_v2.py"
        ),
    }


def _walk_regular_files(root: Path, *, excluded_top: Sequence[str] = ()) -> list[Path]:
    """Walk without following symlinks and prune only exact root-level names."""

    root = _assert_path_no_symlink(root, context="Governed directory")
    if not root.is_dir() or root.is_symlink():
        raise ContractError(f"Governed directory is missing or symlinked: {root}")
    excluded = set(excluded_top)
    observed: list[Path] = []

    def fail(error: OSError) -> None:
        raise ContractError(f"Governed tree walk failed: {error}")

    for current_raw, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False, onerror=fail
    ):
        current = Path(current_raw)
        directory_names.sort()
        file_names.sort()
        retained = []
        for name in directory_names:
            child = current / name
            if child.is_symlink() or not child.is_dir():
                raise ContractError(f"Governed tree contains symlink/special directory: {child}")
            if current == root and name in excluded:
                continue
            retained.append(name)
        directory_names[:] = retained
        for name in file_names:
            child = current / name
            if child.is_symlink() or not child.is_file():
                raise ContractError(f"Governed tree contains symlink/special file: {child}")
            observed.append(child)
    return sorted(observed, key=lambda item: item.relative_to(root).as_posix())


def _pinned_v1_namespace(root: Path) -> dict[str, Any]:
    implementation = {
        name: _expected_identity(Path(spec["path"]), str(spec["sha256"]), int(spec["size_bytes"]))
        for name, spec in PINNED_V1_IMPLEMENTATION.items()
    }
    namespace = root / recovery_v1.RECOVERY_DIRNAME
    files = _walk_regular_files(namespace)
    observed_roster = {path.relative_to(namespace).as_posix() for path in files}
    if observed_roster != set(PINNED_V1_NAMESPACE):
        raise ContractError(
            "Recovery-v1 namespace roster drifted: "
            f"{sorted(observed_roster ^ set(PINNED_V1_NAMESPACE))}"
        )
    artifacts = {
        relative: _expected_identity(namespace / relative, sha256, size_bytes)
        for relative, (sha256, size_bytes) in sorted(PINNED_V1_NAMESPACE.items())
    }
    terminal = _read_json(recovery_v1.recovered_terminal_path(root))
    if (
        terminal.get("status") != recovery_v1.RECOVERY_STATUS
        or terminal.get("recovery") != recovery_v1.RECOVERY
        or terminal.get("recovery_implementation") != implementation
        or terminal.get("fit_accounting") != FIT_ACCOUNTING
        or terminal.get("execution_accounting") != EXECUTION_ACCOUNTING
    ):
        raise ContractError("Pinned recovery-v1 terminal semantics drifted")
    adjudication = _read_json(recovery_v1.adjudication_path(root))
    if terminal.get("validator_adjudication") != artifacts["receipts/validator_adjudication.json"]:
        raise ContractError("Recovery-v1 terminal does not bind the pinned adjudication")
    return {
        "implementation": implementation,
        "artifacts": artifacts,
        "terminal": terminal,
        "adjudication": adjudication,
    }


def _validate_record(record: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
        raise ContractError(f"Original census record {index} has malformed schema")
    raw_path = record["path"]
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ContractError(f"Original census record {index} has malformed path")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != raw_path
        or not relative.parts
        or any(part in {".", ".."} for part in relative.parts)
        or relative.parts[0] in AUTHORIZED_EXCLUDED_NAMESPACES
    ):
        raise ContractError(f"Original census record {index} escapes its immutable scope")
    sha256 = record["sha256"]
    size = record["size_bytes"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(char not in "0123456789abcdef" for char in sha256)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
    ):
        raise ContractError(f"Original census record {index} has malformed digest/size")
    return {"path": raw_path, "sha256": sha256, "size_bytes": size}


def _original_census(v1: Mapping[str, Any], root: Path) -> dict[str, Any]:
    adjudication = v1["adjudication"]
    raw = adjudication.get("raw_artifact_hash_census")
    if not isinstance(raw, dict) or set(raw) != {
        "before",
        "after_expected_identical",
        "publication_exclusion",
    }:
        raise ContractError("Recovery-v1 raw census wrapper drifted")
    if raw["before"] != raw["after_expected_identical"]:
        raise ContractError("Recovery-v1 before/after census records are not identical")
    if raw["publication_exclusion"] != "recovery_v1/":
        raise ContractError("Recovery-v1 publication exclusion drifted")
    census = raw["before"]
    if not isinstance(census, dict) or set(census) != {
        "root",
        "excluded_prefix",
        "artifact_count",
        "total_size_bytes",
        "tree_sha256",
        "artifacts",
    }:
        raise ContractError("Recovery-v1 original census schema drifted")
    if census.get("root") != str(root.resolve()) or census.get("excluded_prefix") != "recovery_v1/":
        raise ContractError("Recovery-v1 original census root/exclusion drifted")
    raw_records = census.get("artifacts")
    if not isinstance(raw_records, list):
        raise ContractError("Recovery-v1 original census records are missing")
    records = [_validate_record(record, index=index) for index, record in enumerate(raw_records)]
    paths = [record["path"] for record in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ContractError("Recovery-v1 original census paths are unsorted or duplicated")
    observed = {
        "artifact_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "tree_sha256": _canonical_sha256(records),
    }
    if observed != ORIGINAL_CENSUS or any(
        census.get(key) != value for key, value in observed.items()
    ):
        raise ContractError(f"Recovery-v1 original census scalar/digest drifted: {observed}")
    return {**census, "artifacts": records}


def _current_scoped_census(root: Path) -> dict[str, Any]:
    files = _walk_regular_files(root, excluded_top=AUTHORIZED_EXCLUDED_NAMESPACES)
    records = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "sha256": campaign._sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return {
        "artifact_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "tree_sha256": _canonical_sha256(records),
        "artifacts": records,
    }


def _validate_downstream_baseline(
    root: Path, *, require_continuation_absent: bool
) -> dict[str, Any]:
    downstream = root / "downstream"
    files = _walk_regular_files(downstream)
    observed_paths = {path.relative_to(root).as_posix() for path in files}
    if observed_paths != set(PINNED_PREPARED_BASELINE):
        raise ContractError(
            "Legacy downstream namespace must remain exactly the immutable seven-file "
            f"preparation baseline; drift={sorted(observed_paths ^ set(PINNED_PREPARED_BASELINE))}"
        )
    relative_records = [
        {
            "path": relative,
            **{
                key: value
                for key, value in _expected_identity(root / relative, sha256, size_bytes).items()
                if key != "path"
            },
        }
        for relative, (sha256, size_bytes) in sorted(PINNED_PREPARED_BASELINE.items())
    ]
    artifacts = [
        _expected_identity(root / relative, sha256, size_bytes)
        for relative, (sha256, size_bytes) in sorted(PINNED_PREPARED_BASELINE.items())
    ]
    summary = {
        "artifact_count": len(relative_records),
        "total_size_bytes": sum(record["size_bytes"] for record in relative_records),
        "tree_sha256": _canonical_sha256(relative_records),
    }
    if summary != PREPARED_BASELINE:
        raise ContractError(f"Prepared downstream baseline summary drifted: {summary}")
    continuation = root / "downstream_v2"
    if require_continuation_absent and (continuation.exists() or continuation.is_symlink()):
        raise ContractError("Fresh recovery-v2 certification requires absent downstream_v2")
    if not require_continuation_absent and (continuation.exists() or continuation.is_symlink()):
        _walk_regular_files(continuation)
    return {
        **summary,
        "root": str(root.resolve()),
        "canonical_record_schema": "sorted campaign-relative {path,sha256,size_bytes}",
        "artifacts": artifacts,
    }


def _reconstruct_and_verify_v1(root: Path, original_census: Mapping[str, Any]) -> dict[str, Any]:
    """Replay recovery-v1 with its immutable original census, without monkeypatching."""

    pinned = recovery_v1._pin_base(root)
    roster = recovery_v1._validate_incident_roster(root)
    contract = _read_json(campaign.contract_path(root))
    pack = campaign._pack_identity("virchow2_cls", root, deep=False)
    if contract.get("feature_stores", {}).get("virchow2_cls") != pack:
        raise ContractError("Exact Virchow2-CLS pack identity/inventory drifted")
    adopted_oof = [campaign._adopted_chain(seed, deep=True) for seed in campaign.SEEDS]
    univ1 = [recovery_v1._validate_stock_univ1(root, seed) for seed in campaign.SEEDS]
    v2 = []
    defect = []
    for seed in campaign.SEEDS:
        request, request_identity = recovery_v1._validate_request(
            root, seed, pinned["base_contract"]
        )
        command = recovery_v1._validate_command_proof(contract, root, seed)
        native = recovery_v1._validate_v2_native(root, seed)
        defect_record = recovery_v1._reproduce_original_defect(root, seed)
        v2.append(
            {
                "seed": seed,
                "request": request_identity,
                "request_created_utc": request["created_utc"],
                "log": _artifact(campaign.log_path(root, "virchow2_full", seed)),
                "contracted_command": command,
                "native": native,
                "attempt": 1,
                "retries": 0,
            }
        )
        defect.append(defect_record)
    concurrency = recovery_v1._concurrency_witness(v2, univ1, root)
    evidence = {
        "pinned": pinned,
        "roster": roster,
        "pack": pack,
        "adopted_oof": adopted_oof,
        "univ1": univ1,
        "v2": v2,
        "defect": defect,
        "concurrency": concurrency,
        "raw_census": dict(original_census),
    }
    recovery_v1._assert_evidence_matches_census(root, evidence, original_census)
    return recovery_v1._verify_recovery_files(root, evidence)


def _scope_summary(original: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root": original["root"],
        **{key: original[key] for key in ("artifact_count", "total_size_bytes", "tree_sha256")},
        "original_census_source": _artifact(
            Path(original["root"]) / "recovery_v1/receipts/validator_adjudication.json"
        ),
        "original_census_json_pointer": "/raw_artifact_hash_census/before",
        "all_original_records_rehashed_at_certification": True,
        "closed_roster_outside_exclusions": True,
    }


def _validate_namespace_boundaries(
    root: Path, original: Mapping[str, Any], *, require_v2_absent: bool
) -> None:
    forbidden_analysis = root / "analysis"
    if forbidden_analysis.exists() or forbidden_analysis.is_symlink():
        raise ContractError(
            "Root analysis/ is outside the delegated scope; continuation analysis must live "
            "under downstream_v2/analysis"
        )
    immutable_top = {PurePosixPath(record["path"]).parts[0] for record in original["artifacts"]}
    allowed_top = immutable_top | set(AUTHORIZED_EXCLUDED_NAMESPACES)
    observed_top = set()
    for path in root.iterdir():
        if path.is_symlink():
            raise ContractError(f"Campaign top-level namespace is symlinked: {path}")
        observed_top.add(path.name)
    unexpected = observed_top - allowed_top
    if unexpected:
        raise ContractError(f"Campaign top-level namespace roster is open: {sorted(unexpected)}")
    required = immutable_top | {"recovery_v1", "downstream"}
    if not required.issubset(observed_top):
        raise ContractError(
            f"Campaign top-level namespaces are missing: {sorted(required - observed_top)}"
        )
    if require_v2_absent and "recovery_v2" in observed_top:
        raise ContractError("Fresh recovery_v2 namespace required")
    if require_v2_absent and "downstream_v2" in observed_top:
        raise ContractError("Fresh recovery-v2 certification requires absent downstream_v2")


def _audit_scope(root: Path, *, deep_scope: bool, require_v2_absent: bool) -> dict[str, Any]:
    root = _assert_root(root)
    destination = recovery_v2_dir(root)
    if require_v2_absent and (destination.exists() or destination.is_symlink()):
        raise ContractError("Fresh recovery_v2 namespace required")
    v1 = _pinned_v1_namespace(root)
    original = _original_census(v1, root)
    _validate_namespace_boundaries(root, original, require_v2_absent=require_v2_absent)
    baseline = _validate_downstream_baseline(root, require_continuation_absent=require_v2_absent)
    if deep_scope:
        current = _current_scoped_census(root)
        if current["artifacts"] != original["artifacts"]:
            original_by_path = {item["path"]: item for item in original["artifacts"]}
            current_by_path = {item["path"]: item for item in current["artifacts"]}
            missing = sorted(set(original_by_path) - set(current_by_path))
            added = sorted(set(current_by_path) - set(original_by_path))
            changed = sorted(
                path
                for path in set(original_by_path) & set(current_by_path)
                if original_by_path[path] != current_by_path[path]
            )
            raise ContractError(
                "Immutable training scope drifted: "
                f"missing={missing[:10]}, added={added[:10]}, changed={changed[:10]}"
            )
        if {key: current[key] for key in ORIGINAL_CENSUS} != ORIGINAL_CENSUS:
            raise ContractError("Immutable training scope census summary drifted")
        replayed = _reconstruct_and_verify_v1(root, original)
        if replayed != v1["terminal"]:
            raise ContractError("Recovery-v1 graph replay differs from pinned terminal")
    return {
        "root": root,
        "v1": v1,
        "original": original,
        "scope": _scope_summary(original),
        "baseline": baseline,
    }


def _namespace_policy() -> dict[str, Any]:
    return {
        "immutable_training_scope": "exact original 441-file campaign census",
        "excluded_root_prefixes": [f"{name}/" for name in AUTHORIZED_EXCLUDED_NAMESPACES],
        "pinned_predecessor_namespace": "recovery_v1/ (exact nine-file graph)",
        "current_certificate_namespace": "recovery_v2/ (exact three-file graph)",
        "prepared_baseline_namespace": "downstream/ (seven named bytes remain immutable)",
        "delegated_growth_namespaces": [f"{name}/" for name in MUTABLE_DELEGATED_NAMESPACES],
        "delegated_growth_policy": (
            "regular non-symlink contents wholly delegated to the downstream_v2 controller; "
            "not training-authenticated"
        ),
        "root_analysis_namespace_authorized": False,
    }


def _scope_contract_payload(evidence: Mapping[str, Any], *, created_utc: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "recovery": RECOVERY,
        "status": "bounded_training_scope_erratum_contract",
        "created_utc": created_utc,
        "base_campaign": campaign.CAMPAIGN,
        "incident": {
            "predecessor_status": recovery_v1.RECOVERY_STATUS,
            "failure_phase": "post_certification_consumer_validation",
            "cause": (
                "recovery_v1 recomputed every non-recovery root artifact after seven "
                "authorized label-blind downstream files were published"
            ),
            "training_artifacts_missing": 0,
            "training_artifacts_changed": 0,
            "authorized_downstream_files_added": 7,
        },
        "predecessor_v1": {
            "implementation": evidence["v1"]["implementation"],
            "terminal": evidence["v1"]["artifacts"]["receipts/training_complete_recovered.json"],
            "namespace_artifact_count": len(PINNED_V1_NAMESPACE),
            "namespace_artifacts": [
                evidence["v1"]["artifacts"][relative]
                for relative in sorted(evidence["v1"]["artifacts"])
            ],
        },
        "recovery_implementation": _implementation(),
        "training_scoped_census": evidence["scope"],
        "prepared_downstream_baseline": evidence["baseline"],
        "namespace_policy": _namespace_policy(),
        "fit_accounting": FIT_ACCOUNTING,
        "execution_accounting": EXECUTION_ACCOUNTING,
        "mutation_policy": {
            "training_fits": 0,
            "training_refits": 0,
            "training_retries": 0,
            "scores": 0,
            "allowed_write_prefix": "recovery_v2/",
            "publication": "atomic exactly-once sibling-directory rename",
        },
    }


def _scope_adjudication_payload(
    evidence: Mapping[str, Any], *, created_utc: str, scope_contract: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "recovery": RECOVERY,
        "status": "scope_adjudicated_original_training_bytes_unchanged",
        "created_utc": created_utc,
        "base_campaign": campaign.CAMPAIGN,
        "scope_contract": dict(scope_contract),
        "predecessor_v1_terminal": evidence["v1"]["artifacts"][
            "receipts/training_complete_recovered.json"
        ],
        "predecessor_v1_adjudication": evidence["v1"]["artifacts"][
            "receipts/validator_adjudication.json"
        ],
        "training_scoped_census": evidence["scope"],
        "prepared_downstream_baseline": evidence["baseline"],
        "namespace_policy": _namespace_policy(),
        "validation": {
            "original_record_schema_exact": True,
            "original_record_order_unique": True,
            "original_record_count": ORIGINAL_CENSUS["artifact_count"],
            "original_records_individually_rehashed": True,
            "original_roster_exact": True,
            "extra_files_outside_exclusions": 0,
            "v1_receipt_graph_replayed_without_monkeypatch": True,
            "prepared_baseline_records_rehashed": PREPARED_BASELINE["artifact_count"],
            "recovery_v2_new_fits": 0,
        },
    }


def _terminal_payload(
    evidence: Mapping[str, Any],
    *,
    created_utc: str,
    scope_contract: Mapping[str, Any],
    scope_adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "recovery": RECOVERY,
        "status": RECOVERY_STATUS,
        "created_utc": created_utc,
        "base_campaign": campaign.CAMPAIGN,
        "scope_contract": dict(scope_contract),
        "scope_adjudication": dict(scope_adjudication),
        "recovery_implementation": _implementation(),
        "predecessor_v1_terminal": evidence["v1"]["artifacts"][
            "receipts/training_complete_recovered.json"
        ],
        "training_scoped_census": evidence["scope"],
        "prepared_downstream_baseline": evidence["baseline"],
        "namespace_policy": _namespace_policy(),
        "fit_accounting": FIT_ACCOUNTING,
        "execution_accounting": EXECUTION_ACCOUNTING,
        "concurrency": CONCURRENCY,
        "certification_boundary": (
            "training-only immutability over the exact original 441 records; recovery_v1 "
            "and recovery_v2 are pinned receipt namespaces; seven prepared downstream "
            "records and their namespace roster remain immutable; future downstream_v2 "
            "contents are delegated and are not authenticated as training evidence"
        ),
    }
    if set(terminal) != TERMINAL_FIELDS:
        raise ContractError("Internal recovery-v2 terminal field roster drifted")
    return terminal


def _write_staged_json(path: Path, payload: Mapping[str, Any]) -> None:
    campaign._write_json_once(path, payload)


def _verify_staged(root: Path, stage: Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    expected_roster = {
        path.relative_to(recovery_v2_dir(root)).as_posix()
        for path in recovery_v2_write_inventory(root)
    }
    observed_roster = {path.relative_to(stage).as_posix() for path in _walk_regular_files(stage)}
    if observed_roster != expected_roster:
        raise ContractError(
            f"Recovery-v2 staged roster drifted: {sorted(observed_roster ^ expected_roster)}"
        )
    terminal_path = stage / "receipts/training_complete_scoped.json"
    terminal = _read_json(terminal_path)
    created_utc = terminal.get("created_utc")
    recovery_v1._parse_utc(created_utc, context="staged recovery-v2 created_utc")
    contract_path = stage / "contract_scope_erratum.json"
    contract = _read_json(contract_path)
    if contract != _scope_contract_payload(evidence, created_utc=created_utc):
        raise ContractError("Staged recovery-v2 scope contract does not replay")
    contract_identity = _logical_artifact(contract_path, scope_contract_path(root))
    adjudication_path = stage / "receipts/scope_adjudication.json"
    adjudication = _read_json(adjudication_path)
    if adjudication != _scope_adjudication_payload(
        evidence, created_utc=created_utc, scope_contract=contract_identity
    ):
        raise ContractError("Staged recovery-v2 scope adjudication does not replay")
    adjudication_identity = _logical_artifact(adjudication_path, scope_adjudication_path(root))
    if terminal != _terminal_payload(
        evidence,
        created_utc=created_utc,
        scope_contract=contract_identity,
        scope_adjudication=adjudication_identity,
    ):
        raise ContractError("Staged recovery-v2 terminal does not replay")
    return terminal


def _materialize(root: Path, stage: Path, evidence: Mapping[str, Any], *, created_utc: str) -> None:
    logical = recovery_v2_dir(root)
    contract_payload = _scope_contract_payload(evidence, created_utc=created_utc)
    staged_contract = stage / "contract_scope_erratum.json"
    _write_staged_json(staged_contract, contract_payload)
    contract_identity = _logical_artifact(staged_contract, scope_contract_path(root))

    adjudication_payload = _scope_adjudication_payload(
        evidence,
        created_utc=created_utc,
        scope_contract=contract_identity,
    )
    staged_adjudication = stage / "receipts/scope_adjudication.json"
    _write_staged_json(staged_adjudication, adjudication_payload)
    adjudication_identity = _logical_artifact(staged_adjudication, scope_adjudication_path(root))

    terminal_payload = _terminal_payload(
        evidence,
        created_utc=created_utc,
        scope_contract=contract_identity,
        scope_adjudication=adjudication_identity,
    )
    _write_staged_json(stage / "receipts/training_complete_scoped.json", terminal_payload)
    expected = {path.relative_to(logical).as_posix() for path in recovery_v2_write_inventory(root)}
    observed = {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}
    if observed != expected:
        raise ContractError(
            f"Recovery-v2 staged roster drifted: expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )
    _verify_staged(root, stage, evidence)


def _publish_atomic(root: Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    destination = recovery_v2_dir(root)
    if destination.exists() or destination.is_symlink():
        raise ContractError(f"Recovery-v2 is exactly once; refusing existing {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.recovery-v2-stage-", dir=root.parent))
    published = False
    try:
        _materialize(root, stage, evidence, created_utc=_utcnow())
        current = _audit_scope(root, deep_scope=True, require_v2_absent=True)
        if (
            current["scope"] != evidence["scope"]
            or current["baseline"] != evidence["baseline"]
            or current["v1"]["artifacts"] != evidence["v1"]["artifacts"]
        ):
            raise ContractError("Governed evidence changed during recovery-v2 staging")
        os.rename(stage, destination)
        published = True
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        post = _audit_scope(root, deep_scope=True, require_v2_absent=False)
        if (
            post["scope"] != evidence["scope"]
            or post["baseline"] != evidence["baseline"]
            or post["v1"]["artifacts"] != evidence["v1"]["artifacts"]
        ):
            raise ContractError("Governed evidence changed across recovery-v2 publication")
        return _verify_v2_files(root, post)
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


def _verify_v2_files(root: Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    namespace = recovery_v2_dir(root)
    files = _walk_regular_files(namespace)
    observed = {path.relative_to(namespace).as_posix() for path in files}
    expected = {
        path.relative_to(namespace).as_posix() for path in recovery_v2_write_inventory(root)
    }
    if observed != expected:
        raise ContractError(f"Recovery-v2 namespace roster drifted: {sorted(observed ^ expected)}")
    terminal = _read_json(scoped_terminal_path(root))
    if set(terminal) != TERMINAL_FIELDS:
        raise ContractError("Recovery-v2 terminal field roster drifted")
    created_utc = terminal.get("created_utc")
    recovery_v1._parse_utc(created_utc, context="recovery-v2 terminal created_utc")

    contract = _read_json(scope_contract_path(root))
    if contract != _scope_contract_payload(evidence, created_utc=created_utc):
        raise ContractError("Recovery-v2 scope contract does not replay")
    contract_identity = _artifact(scope_contract_path(root))
    adjudication = _read_json(scope_adjudication_path(root))
    if adjudication != _scope_adjudication_payload(
        evidence, created_utc=created_utc, scope_contract=contract_identity
    ):
        raise ContractError("Recovery-v2 scope adjudication does not replay")
    adjudication_identity = _artifact(scope_adjudication_path(root))
    expected_terminal = _terminal_payload(
        evidence,
        created_utc=created_utc,
        scope_contract=contract_identity,
        scope_adjudication=adjudication_identity,
    )
    if terminal != expected_terminal:
        raise ContractError("Recovery-v2 terminal does not replay")
    return terminal


def validate_scoped_terminal(
    root: Path = DEFAULT_OUTPUT_ROOT, *, deep_scope: bool = True
) -> dict[str, Any]:
    """Validate and return the scoped recovery-v2 terminal.

    ``deep_scope=True`` rehashes every immutable training artifact and replays
    recovery-v1.  ``False`` authenticates all pinned receipt/implementation and
    seven baseline bytes, but is only safe within a run already gated by a fresh
    deep validation receipt.
    """

    root = _assert_root(root)
    if not recovery_v2_dir(root).is_dir() or recovery_v2_dir(root).is_symlink():
        raise ContractError("Missing or symlinked recovery_v2 namespace")
    evidence = _audit_scope(root, deep_scope=deep_scope, require_v2_absent=False)
    return _verify_v2_files(root, evidence)


def validate_recovered_terminal_v2(
    root: Path = DEFAULT_OUTPUT_ROOT, *, deep_pack: bool = True
) -> dict[str, Any]:
    """Compatibility alias for downstream consumers."""

    return validate_scoped_terminal(root, deep_scope=deep_pack)


def cmd_plan(args: argparse.Namespace) -> None:
    root = _assert_root(args.output_root)
    print(
        json.dumps(
            {
                "status": "PLAN_ONLY_NO_WRITES",
                "recovery": RECOVERY,
                "output_root": str(root),
                "write_inventory": [str(path) for path in recovery_v2_write_inventory(root)],
                "original_training_census": ORIGINAL_CENSUS,
                "prepared_downstream_baseline": PREPARED_BASELINE,
                "namespace_policy": _namespace_policy(),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def cmd_audit(args: argparse.Namespace) -> None:
    root = _assert_root(args.output_root)
    evidence = _audit_scope(root, deep_scope=True, require_v2_absent=True)
    print(
        json.dumps(
            {
                "status": "PASS_READ_ONLY_RECOVERY_V2_AUDIT",
                "recovery": RECOVERY,
                "training_scoped_census": evidence["scope"],
                "prepared_downstream_baseline": evidence["baseline"],
                "fit_accounting": FIT_ACCOUNTING,
                "execution_accounting": EXECUTION_ACCOUNTING,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def cmd_certify(args: argparse.Namespace) -> None:
    if not args.apply:
        raise ContractError("certify requires --apply")
    root = _assert_root(args.output_root)
    evidence = _audit_scope(root, deep_scope=True, require_v2_absent=True)
    _publish_atomic(root, evidence)
    print(
        json.dumps(
            {
                "status": RECOVERY_STATUS,
                "terminal": _artifact(scoped_terminal_path(root)),
                "fit_accounting": FIT_ACCOUNTING,
                "execution_accounting": EXECUTION_ACCOUNTING,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def cmd_verify(args: argparse.Namespace) -> None:
    root = _assert_root(args.output_root)
    terminal = validate_scoped_terminal(root, deep_scope=True)
    print(
        json.dumps(
            {
                "status": "PASS",
                "terminal_status": terminal["status"],
                "terminal": _artifact(scoped_terminal_path(root)),
                "training_scoped_census": terminal["training_scoped_census"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def cmd_status(args: argparse.Namespace) -> None:
    root = _assert_root(args.output_root)
    terminal = scoped_terminal_path(root)
    print(
        json.dumps(
            {
                "status": "RECOVERY_V2_STATUS_READ_ONLY",
                "predecessor_v1_terminal": _artifact(recovery_v1.recovered_terminal_path(root)),
                "recovery_v2_published": terminal.is_file() and not terminal.is_symlink(),
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
