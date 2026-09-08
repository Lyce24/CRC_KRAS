#!/usr/bin/env python3
"""Fail-closed consolidated verifier for corrected Aim 2, Aim 3, and Aim 4.

This layer does not replace any component verifier.  It delegates the existing
corrected Aim-2/Aim-3 verifier (including the 45-chain/225-fold repeated-control
campaign), the corrected Aim-4 verifier archived inside its output root, and
the vocabulary-stability verifier archived inside its output root.  It then
checks cross-root bindings and recomputes the central claim-state algebra.

The default remains a fresh strict replay of the archived Aim-4 verifiers.  An
explicit fast mode skips only those two expensive replays and instead verifies
the immutable full-replay receipt/hash chains before applying the same
lightweight protocol and claim-algebra checks.

Seven explicit, absolute, mutually non-nested roots are required.  No campaign
root is modified.  An optional consolidated JSON receipt is atomically and
exclusively created at a path outside every supplied root.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import verify_corrected_aim2_aim3 as aim23  # noqa: E402

SCHEMA_VERSION = 1
FULL_REPLAY_MODE = "full-replay"
FAST_RECEIPT_MODE = "sealed-full-replay-receipts-fast"
CAP = 8192
AIM23_BOOTSTRAP = 10_000
REPEATED_BOOTSTRAP = 20_000
E4_BOOTSTRAP = 2_000
STABILITY_BOOTSTRAP = 2_000
STABILITY_SCHEMA_VERSION = 2
K = 32
FDR_ALPHA = 0.05
STABILITY_K_VALUES = (24, 32, 40)
STABILITY_SEEDS = (20260819, 20260820, 20260821)
STABILITY_CANONICAL_SAMPLE_SHA256 = (
    "d0ac16fb094b0adc368b325a714e6f4ad08ad10d534ddfe52b70c63d4a4c5e5d"
)
STABILITY_MAX_CLUSTER_MASS_TOTAL_VARIATION = 5e-4
STABILITY_MAX_SINGLE_CLUSTER_MASS_DELTA = 1e-4
STABILITY_INERTIA_CONTRACT = {
    "model_inertia": "scikit_learn_float32_fit_diagnostic_finite_positive_only",
    "replay_inertia_float64": (
        "numpy_sum_float32_squared_residuals_with_float64_accumulator"
    ),
    "replay_verification": "exact_equality_from_sealed_sample_assignments_centroids",
}
STABILITY_VARIANTS = tuple(
    f"k{k}_seed{seed}" for k in STABILITY_K_VALUES for seed in STABILITY_SEEDS
)
ANCHORS = {17: "M04", 28: "M07"}


class FullVerificationError(RuntimeError):
    """A component, lineage, cross-root, or claim-algebra contract failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullVerificationError(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise FullVerificationError(f"Required artifact does not resolve: {path}") from error
    _require(resolved.is_file(), f"Required artifact is not a file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FullVerificationError(f"Cannot read JSON object {path}: {error}") from error
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _absolute_root(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    _require(path.is_absolute(), f"{label} must be an explicit absolute path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise FullVerificationError(f"{label} does not resolve: {path}") from error
    _require(resolved.is_dir(), f"{label} is not a directory: {resolved}")
    return resolved


def _validate_distinct_roots(roots: dict[str, Path]) -> None:
    _require(len(set(roots.values())) == len(roots), "All seven roots must be distinct")
    ordered = list(roots.items())
    for index, (left_name, left) in enumerate(ordered):
        for right_name, right in ordered[index + 1 :]:
            _require(
                left not in right.parents and right not in left.parents,
                f"Roots must not be nested: {left_name}={left}, {right_name}={right}",
            )


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise FullVerificationError(f"{label} is not numeric: {value!r}") from error
    _require(math.isfinite(number), f"{label} is non-finite")
    return number


def _require_files(root: Path, relatives: tuple[str, ...], label: str) -> None:
    missing = [relative for relative in relatives if not (root / relative).is_file()]
    _require(
        not missing,
        f"{label} root is partial/incomplete before delegation; missing {missing}",
    )


def _root_file(root: Path, relative: str, label: str) -> Path:
    """Resolve one receipt-controlled relative path without permitting escape."""

    relative_path = Path(relative)
    _require(
        relative and not relative_path.is_absolute() and ".." not in relative_path.parts,
        f"{label} has an unsafe relative path: {relative!r}",
    )
    try:
        path = (root / relative_path).resolve(strict=True)
    except OSError as error:
        raise FullVerificationError(
            f"{label} does not resolve inside its root: {relative}"
        ) from error
    _require(root in path.parents, f"{label} escapes its root: {relative}")
    _require(path.is_file(), f"{label} is not a file: {path}")
    return path


def _require_manifest_artifact(
    root: Path,
    artifacts: dict[str, Any],
    relative: str,
    label: str,
) -> dict[str, Any]:
    """Bind a critical current file to an E4 dictionary manifest record."""

    _require(isinstance(relative, str) and relative, f"{label} has an invalid path")
    record = artifacts.get(relative)
    _require(isinstance(record, dict), f"{label} lacks {relative}")
    path = _root_file(root, relative, label)
    observed = _identity(path)
    _require(
        record.get("path") == relative
        and record.get("size_bytes") == observed["size_bytes"]
        and record.get("sha256") == observed["sha256"],
        f"{label} artifact hash differs: {relative}",
    )
    return observed


def _inventory_by_path(rows: object, label: str) -> dict[str, dict[str, Any]]:
    """Parse a stability payload inventory while rejecting partial/duplicate rows."""

    _require(isinstance(rows, list) and bool(rows), f"{label} is missing or empty")
    inventory: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"{label}[{index}] is not an object")
        relative = row.get("path")
        _require(isinstance(relative, str) and relative, f"{label}[{index}] lacks path")
        relative_path = Path(relative)
        _require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"{label}[{index}] has an unsafe path: {relative}",
        )
        _require(relative not in inventory, f"{label} duplicates {relative}")
        inventory[relative] = row
    return inventory


def _require_inventory_artifact(
    root: Path,
    inventory: dict[str, dict[str, Any]],
    relative: str,
    label: str,
) -> dict[str, Any]:
    """Bind a critical current file to a stability list-inventory record."""

    record = inventory.get(relative)
    _require(isinstance(record, dict), f"{label} lacks {relative}")
    path = _root_file(root, relative, label)
    observed = _identity(path)
    _require(
        record.get("size_bytes") == observed["size_bytes"]
        and record.get("sha256") == observed["sha256"],
        f"{label} artifact hash differs: {relative}",
    )
    return observed


def _all_checks_pass(value: object, required: set[str]) -> bool:
    return (
        isinstance(value, dict)
        and required <= set(value)
        and all(value.get(label) == "PASS" for label in required)
    )


def _e4_manifest_paths(root: Path, *, phase: str) -> set[str]:
    """Rebuild the corrected-E4 file inventory using its sealing exclusions."""

    if phase == "numeric":
        excluded_names = {
            "numeric_complete.json",
            "numeric_verification.json",
            "lineage_complete.json",
            "verification.json",
        }
    elif phase == "final":
        excluded_names = {"lineage_complete.json", "verification.json"}
    else:
        raise ValueError(f"unknown corrected-E4 manifest phase: {phase}")
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in excluded_names
        and "_staging" not in path.relative_to(root).parts
    }


def _precheck_roots(roots: dict[str, Path], *, pathology_state: str) -> None:
    """Reject partial roots before any expensive component delegate is called."""

    required = {
        "aim2_core": ("lineage_start.json",),
        "e2c_offset": (
            "lineage_complete.json",
            f"analysis/e2c_native_logit_offset_cap{CAP}.json",
        ),
        "aim2_e2d_refresh": ("refresh_receipt.json", "verification_receipt.json"),
        "aim3_corrected": (
            "lineage_start.json",
            "lineage_complete.json",
            "analysis/aim3_corrected.json",
        ),
        "aim3_repeated_controls": (
            "lineage_start.json",
            "lineage_complete.json",
            "analysis/aim3_repeated_control_report.json",
            "analysis/bootstrap_distributions.npz",
            "analysis/analysis_audit.json",
        ),
        "aim4_corrected": (
            "lineage_start.json",
            "numeric_complete.json",
            "numeric_verification.json",
            "profiles/patient_profiles_k32.parquet",
            "analysis/specificity_k32.json",
            "analysis/aim4_corrected_k32.json",
            "receipts/inputs.json",
            "receipts/review_packets.json",
            "receipts/source_snapshot.json",
            "source_snapshot/aim4_morphologic_atlas.py",
        ),
        "aim4_vocab_stability": (
            "completion_receipt.json",
            "verification_receipt.json",
            "results.json",
            "input_receipt.json",
            "development_patient_abundance.parquet",
            "association_effects.csv",
            "all_anchor_correspondence.csv",
            "source_snapshot/manifest.json",
            "source_snapshot/tools/aim4_vocab_stability.py",
        ),
    }
    for name, relatives in required.items():
        _require_files(roots[name], relatives, name)

    final_files = (
        "lineage_complete.json",
        "verification.json",
        "receipts/review_import.json",
    )
    review_completion = roots["aim4_corrected"] / "review_completion" / f"k{K}"
    retired_files = (
        "analysis/pathology_review_k32.json",
        "analysis/aim4_corrected_reviewed_k32.json",
    )
    present = [
        relative
        for relative in (*final_files, *retired_files)
        if (roots["aim4_corrected"] / relative).is_file()
    ]
    if review_completion.exists() or review_completion.is_symlink():
        present.append(f"review_completion/k{K}")
    if pathology_state == "pending":
        _require(
            not present,
            "Aim4 was declared pathology-pending but final/import artifacts are present: "
            f"{present}",
        )
    else:
        missing = [relative for relative in final_files if relative not in present]
        if f"review_completion/k{K}" not in present:
            missing.append(f"review_completion/k{K}")
        _require(
            not missing,
            f"Aim4 was declared final-complete but final artifacts are missing: {missing}",
        )
        stale = [relative for relative in retired_files if relative in present]
        _require(
            not stale,
            f"Aim4 final root contains retired non-atomic pathology artifacts: {stale}",
        )


def _authenticate_e4_snapshot(root: Path) -> dict[str, Any]:
    receipt = _read_json(root / "receipts" / "source_snapshot.json")
    _require(
        receipt.get("schema_version") == 1 and receipt.get("component") == "aim4_corrected_cap8192",
        "Aim4 archived source receipt is invalid",
    )
    matches = [
        row for row in receipt.get("files", []) if row.get("relative_path") == "aim4_morphologic_atlas.py"
    ]
    _require(len(matches) == 1, "Aim4 source receipt does not uniquely bind aim4_morphologic_atlas.py")
    snapshot = root / "source_snapshot" / "aim4_morphologic_atlas.py"
    _require(
        matches[0].get("imported") == _identity(snapshot),
        "Aim4 archived verifier identity differs from its source receipt",
    )
    return _identity(snapshot)


def _authenticate_stability_snapshot(root: Path) -> dict[str, Any]:
    manifest_path = root / "source_snapshot" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FullVerificationError(f"Cannot read stability source manifest: {error}") from error
    _require(isinstance(manifest, list), "Stability source manifest is not a list")
    matches = [row for row in manifest if row.get("path") == "tools/aim4_vocab_stability.py"]
    _require(len(matches) == 1, "Stability source manifest does not uniquely bind its verifier")
    snapshot = root / "source_snapshot" / "tools" / "aim4_vocab_stability.py"
    observed = _identity(snapshot)
    _require(
        matches[0].get("size_bytes") == observed["size_bytes"]
        and matches[0].get("sha256") == observed["sha256"],
        "Stability archived verifier identity differs from its source manifest",
    )
    return observed


_ARCHIVED_BRIDGE = r"""
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

mode = sys.argv[1]
snapshot = Path(sys.argv[2]).resolve(strict=True)
with contextlib.redirect_stdout(sys.stderr):
    if mode in {"e4_numeric", "e4_final"}:
        source_root = snapshot.parent
        sys.path.insert(0, str(source_root))
        spec = importlib.util.spec_from_file_location("archived_e4_corrected", snapshot)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        legacy_input = Path(sys.argv[3])
        aim2_root = Path(sys.argv[4])
        output_root = Path(sys.argv[5])
        bundle = module.validate_archival_inputs(
            legacy_input, aim2_root, output_root, deep_attention=True
        )
        if mode == "e4_numeric":
            result = module.verify_numeric(bundle, output_root)
        else:
            result = module.verify_output(bundle, output_root, replay=True)
        result = module._sanitize_json(result)
    elif mode == "stability":
        spec = importlib.util.spec_from_file_location(
            "archived_aim4_vocab_stability", snapshot
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result = module.verify_output(Path(sys.argv[3]), replay_input=True)
        result = module._sanitize_json(result)
    else:
        raise RuntimeError(f"unknown archived verifier mode: {mode}")
sys.stdout.write(json.dumps(result, sort_keys=True, allow_nan=False))
"""


def _run_archived(mode: str, snapshot: Path, arguments: tuple[Path, ...]) -> dict[str, Any]:
    command = [
        sys.executable,
        "-c",
        _ARCHIVED_BRIDGE,
        mode,
        str(snapshot),
        *(str(path) for path in arguments),
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    # Archived verification is read-only: never deposit __pycache__ files in
    # an immutable campaign's source snapshot while importing its verifier.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=str(snapshot.parent),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    _require(
        completed.returncode == 0,
        f"Archived {mode} verifier failed (exit {completed.returncode}): "
        f"{completed.stderr[-8000:]}",
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise FullVerificationError(
            f"Archived {mode} verifier did not return one JSON object: {completed.stdout[-2000:]}"
        ) from error
    _require(isinstance(result, dict), f"Archived {mode} verifier returned a non-object")
    return result


def _delegate_e4(
    root: Path,
    aim2_root: Path,
    *,
    pathology_state: str,
    execute_replay: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    verifier = _authenticate_e4_snapshot(root)
    inputs = _read_json(root / "receipts" / "inputs.json")
    _require(
        inputs.get("schema_version") == 1 and inputs.get("component") == "aim4_corrected_cap8192",
        "Corrected Aim4 prepared input receipt is invalid",
    )
    _require(
        Path(str(inputs.get("aim2_root", ""))).resolve(strict=True) == aim2_root,
        "Corrected Aim4 is not bound to the explicit Aim2 core root",
    )
    _require(
        inputs.get("aim2_lineage_start") == _identity(aim2_root / "lineage_start.json"),
        "Corrected Aim4 input receipt does not hash-bind the explicit Aim2 lineage",
    )
    legacy_input = Path(str(inputs.get("input_e4_root", ""))).resolve(strict=False)
    _require(legacy_input.is_absolute(), "Corrected Aim4 lacks an absolute legacy E4 input root")
    snapshot = Path(verifier["path"])
    sealed_numeric = _read_json(root / "numeric_verification.json")
    if execute_replay:
        numeric = _run_archived("e4_numeric", snapshot, (legacy_input, aim2_root, root))
        _require(
            _canonical(numeric) == _canonical(sealed_numeric),
            "Archived Aim4 numeric replay differs from numeric_verification.json",
        )
    else:
        numeric = sealed_numeric
    final: dict[str, Any] | None = None
    if pathology_state == "final":
        sealed_final = _read_json(root / "verification.json")
        if execute_replay:
            final = _run_archived("e4_final", snapshot, (legacy_input, aim2_root, root))
            _require(
                _canonical(final) == _canonical(sealed_final),
                "Archived Aim4 final replay differs from verification.json",
            )
        else:
            final = sealed_final
    return numeric, final, verifier


def _delegate_stability(
    root: Path, *, execute_replay: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    verifier = _authenticate_stability_snapshot(root)
    sealed = _read_json(root / "verification_receipt.json")
    if execute_replay:
        replay = _run_archived("stability", Path(verifier["path"]), (root,))
        replay_without_time = dict(replay)
        sealed_without_time = dict(sealed)
        replay_without_time.pop("verified_at_utc", None)
        sealed_without_time.pop("verified_at_utc", None)
        _require(
            _canonical(replay_without_time) == _canonical(sealed_without_time),
            "Archived stability full replay differs from verification_receipt.json",
        )
    else:
        replay = sealed
    return replay, verifier


def _verify_aim23_delegate(result: dict[str, Any]) -> dict[str, Any]:
    _require(
        result.get("schema_version") == 2 and result.get("status") == "PASS",
        "Existing Aim2+Aim3 consolidated verifier did not PASS",
    )
    repeated = result.get("checks", {}).get(
        "aim3_repeated_control_component_and_headline_verifier", {}
    )
    _require(
        repeated.get("status") == "PASS"
        and repeated.get("control_chains") == 45
        and repeated.get("control_folds") == 225
        and repeated.get("n_bootstrap") == REPEATED_BOOTSTRAP,
        "Aim3 repeated-control 45-chain/225-fold/full-bootstrap contract failed",
    )
    aim3 = result.get("checks", {}).get("aim3_native_logit_and_headline_algebra", {})
    _require(aim3.get("status") == "PASS", "Aim3 primary claim algebra did not PASS")
    return {
        "status": "PASS",
        "primary_bootstrap": AIM23_BOOTSTRAP,
        "repeated_bootstrap": REPEATED_BOOTSTRAP,
        "repeated_control_chains": 45,
        "repeated_control_folds": 225,
        "aim3_familywise_rungs_passing": aim3.get("familywise_rungs_passing"),
        "aim3_repeated_consensus_verdicts": repeated.get("consensus_verdicts"),
    }


def _verify_aim23_root_budgets(roots: dict[str, Path]) -> dict[str, Any]:
    """Bind canonical result bytes and independently restate full budgets."""

    e2c_path = roots["e2c_offset"] / "analysis" / f"e2c_native_logit_offset_cap{CAP}.json"
    refresh_path = roots["aim2_e2d_refresh"] / "refresh_receipt.json"
    aim3_path = roots["aim3_corrected"] / "analysis" / "aim3_corrected.json"
    repeated_path = (
        roots["aim3_repeated_controls"] / "analysis" / "aim3_repeated_control_report.json"
    )
    e2c = _read_json(e2c_path)
    refresh = _read_json(refresh_path)
    aim3 = _read_json(aim3_path)
    repeated = _read_json(repeated_path)
    _require(
        e2c.get("cap") == CAP
        and e2c.get("n_bootstrap") == AIM23_BOOTSTRAP
        and e2c.get("reps") == 100,
        "Aim2 E2c cap/full-bootstrap/procedure budget differs",
    )
    _require(
        refresh.get("cap") == CAP
        and refresh.get("n_bootstrap") == AIM23_BOOTSTRAP
        and refresh.get("analysis_grade") == "full_10000_bootstrap",
        "Aim2 E2d refresh cap/full-bootstrap grade differs",
    )
    aim3_protocol = aim3.get("protocol", {})
    _require(
        aim3_protocol.get("cap") == CAP and aim3_protocol.get("n_bootstrap") == AIM23_BOOTSTRAP,
        "Aim3 primary cap/full-bootstrap budget differs",
    )
    repeated_protocol = repeated.get("protocol", {})
    _require(
        repeated_protocol.get("n_bootstrap") == REPEATED_BOOTSTRAP
        and repeated.get("accounting")
        == {"fine_folds_reused": 75, "control_chains": 45, "control_folds": 225},
        "Aim3 repeated full-bootstrap/45-chain/225-fold accounting differs",
    )
    return {
        "status": "PASS",
        "artifacts": {
            "e2c_result": _identity(e2c_path),
            "aim2_refresh_receipt": _identity(refresh_path),
            "aim2_refresh_verification": _identity(
                roots["aim2_e2d_refresh"] / "verification_receipt.json"
            ),
            "aim3_result": _identity(aim3_path),
            "aim3_completion": _identity(roots["aim3_corrected"] / "lineage_complete.json"),
            "aim3_repeated_result": _identity(repeated_path),
            "aim3_repeated_completion": _identity(
                roots["aim3_repeated_controls"] / "lineage_complete.json"
            ),
        },
    }


def _verify_packet_key_separation(root: Path) -> dict[str, Any]:
    receipt_path = root / "receipts" / "review_packets.json"
    receipt = _read_json(receipt_path)
    transaction_path = root / "review_bundles" / f"k{K}" / "_bundle_receipt_DO_NOT_SHARE.json"
    transaction = _read_json(transaction_path)
    _require(
        _canonical(transaction) == _canonical(receipt)
        and transaction_path.stat().st_size == receipt_path.stat().st_size
        and _sha256_file(transaction_path) == _sha256_file(receipt_path),
        "Corrected Aim4 review transaction receipt differs from its external copy",
    )
    selection = receipt.get("selection", {})
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("component") == "aim4_corrected_cap8192"
        and receipt.get("status") == "PASS"
        and selection.get("separate_keys") is True
        and selection.get("tiles_per_montage") == 12
        and receipt.get("blinding")
        == "prototype IDs occur only in external keys, never packet files",
        "Corrected Aim4 packet blinding/separate-key declaration differs",
    )
    packets: dict[str, Any] = {}
    prototype_sets: list[set[int]] = []
    key_paths: list[Path] = []
    packet_paths: list[Path] = []
    for slug in ("base", "attention_addendum"):
        block = receipt.get(slug, {})
        bundle = Path(str(block.get("bundle", ""))).resolve(strict=True)
        packet = Path(str(block.get("packet", ""))).resolve(strict=True)
        key = Path(str(block.get("key", {}).get("path", ""))).resolve(strict=True)
        expected_bundle = (root / "review_bundles" / f"k{K}" / slug).resolve(strict=True)
        _require(bundle == expected_bundle, f"Aim4 {slug} bundle path mismatch")
        _require(packet == bundle / "packet", f"Aim4 {slug} reviewer packet path mismatch")
        _require(
            key == bundle / "unblinding_key_DO_NOT_SHARE.csv", f"Aim4 {slug} key path mismatch"
        )
        _require(
            key not in packet.parents and packet not in key.parents,
            f"Aim4 {slug} key/packet nesting",
        )
        _require(block.get("key") == _identity(key), f"Aim4 {slug} key identity drifted")
        selection_key = (
            "base_packet_prototypes" if slug == "base" else "attention_addendum_prototypes"
        )
        prototypes = set(map(int, selection.get(selection_key, [])))
        prototype_sets.append(prototypes)
        key_paths.append(key)
        packet_paths.append(packet)
        packets[slug] = {
            "packet": str(packet),
            "key": _identity(key),
            "prototypes": sorted(prototypes),
        }
    _require(key_paths[0] != key_paths[1], "Aim4 base/addendum reuse one unblinding key")
    _require(packet_paths[0] != packet_paths[1], "Aim4 base/addendum reuse one reviewer packet")
    _require(not prototype_sets[0] & prototype_sets[1], "Aim4 packet prototype selections overlap")
    return {
        "status": "PASS",
        "packets": packets,
        "receipt": _identity(receipt_path),
        "transaction_receipt": _identity(transaction_path),
    }


def _verify_completed_review_import(
    root: Path,
    final_artifacts: dict[str, Any],
    numeric_report: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate the atomic post-pathology bundle through its copied receipt."""

    receipt_relative = "receipts/review_import.json"
    receipt_path = root / receipt_relative
    _require_manifest_artifact(
        root,
        final_artifacts,
        receipt_relative,
        "Corrected Aim4 final completion",
    )
    receipt = _read_json(receipt_path)
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("component") == "aim4_corrected_pathology_review"
        and receipt.get("status") == "PASS",
        "Corrected Aim4 completed-review receipt is invalid",
    )

    bundle_relative = f"review_completion/k{K}"
    transaction_relative = f"{bundle_relative}/_import_receipt_DO_NOT_SHARE.json"
    transaction_path = root / transaction_relative
    _require_manifest_artifact(
        root,
        final_artifacts,
        transaction_relative,
        "Corrected Aim4 final completion",
    )
    _require(
        _canonical(_read_json(transaction_path)) == _canonical(receipt),
        "Corrected Aim4 copied review-import receipt differs from its atomic bundle",
    )

    forms = receipt.get("forms")
    _require(
        isinstance(forms, dict) and set(forms) == {"base", "attention_addendum"},
        "Corrected Aim4 completed-review receipt has the wrong form inventory",
    )
    imported_forms: dict[str, dict[str, Any]] = {}
    for slug in ("base", "attention_addendum"):
        evidence = forms[slug]
        _require(
            isinstance(evidence, dict)
            and isinstance(evidence.get("source"), dict)
            and isinstance(evidence.get("imported"), dict),
            f"Corrected Aim4 {slug} review import evidence is malformed",
        )
        relative = f"{bundle_relative}/submissions/{slug}_completed.csv"
        expected_path = (root / relative).resolve(strict=True)
        imported = evidence["imported"]
        source = evidence["source"]
        _require(
            imported == _identity(expected_path)
            and set(source) == {"path", "size_bytes", "sha256"}
            and Path(str(source.get("path", ""))).is_absolute()
            and source.get("size_bytes") == imported.get("size_bytes")
            and source.get("sha256") == imported.get("sha256"),
            f"Corrected Aim4 {slug} sealed review differs from its import receipt",
        )
        imported_forms[slug] = _require_manifest_artifact(
            root,
            final_artifacts,
            relative,
            "Corrected Aim4 final completion",
        )

    pathology_relative = f"{bundle_relative}/pathology_review_k{K}.json"
    reviewed_relative = f"{bundle_relative}/aim4_corrected_reviewed_k{K}.json"
    pathology_identity = _require_manifest_artifact(
        root,
        final_artifacts,
        pathology_relative,
        "Corrected Aim4 final completion",
    )
    reviewed_identity = _require_manifest_artifact(
        root,
        final_artifacts,
        reviewed_relative,
        "Corrected Aim4 final completion",
    )
    _require(
        receipt.get("pathology") == pathology_identity
        and receipt.get("reviewed_report") == reviewed_identity,
        "Corrected Aim4 review-import receipt does not bind its canonical reports",
    )

    selection = numeric_report.get("review_selection", {})
    base = selection.get("base_packet")
    addendum = selection.get("attention_addendum_packet")
    _require(
        isinstance(base, list)
        and isinstance(addendum, list)
        and all(type(value) is int and 0 <= value < K for value in (*base, *addendum))
        and len(base) == len(set(base))
        and len(addendum) == len(set(addendum))
        and not set(base) & set(addendum),
        "Corrected Aim4 numeric report has an invalid final-review selection",
    )
    expected_prototypes = set(base) | set(addendum)
    expected_packet = {
        **{prototype: "base" for prototype in base},
        **{prototype: "attention_addendum" for prototype in addendum},
    }

    pathology = _read_json(root / pathology_relative)
    annotations = pathology.get("annotations")
    _require(
        pathology.get("schema_version") == 1
        and pathology.get("component") == "aim4_corrected_pathology_review"
        and pathology.get("status") == "complete"
        and pathology.get("structured_not_concatenated") is True
        and pathology.get("submission_imports") == forms
        and isinstance(annotations, dict)
        and set(annotations) == {str(value) for value in expected_prototypes}
        and type(receipt.get("n_completed_prototypes")) is int
        and receipt.get("n_completed_prototypes") == len(expected_prototypes),
        "Corrected Aim4 structured pathology coverage is incomplete or inconsistent",
    )
    for key, annotation in annotations.items():
        prototype = int(key)
        _require(
            isinstance(annotation, dict)
            and type(annotation.get("prototype")) is int
            and annotation.get("prototype") == prototype
            and annotation.get("packet") == expected_packet[prototype]
            and isinstance(annotation.get("montage_id"), str)
            and bool(annotation["montage_id"])
            and isinstance(annotation.get("assessment"), dict)
            and annotation["assessment"].get("montage_id") == annotation["montage_id"]
            and annotation["assessment"].get("review_status") == "complete"
            and annotation["assessment"].get("blinding_attestation") == "confirmed_no_key_access",
            f"Corrected Aim4 pathology annotation is malformed: p{prototype}",
        )

    reviewed = _read_json(root / reviewed_relative)
    numeric_claim_limits = numeric_report.get("claim_limits")
    _require(
        isinstance(numeric_claim_limits, list),
        "Corrected Aim4 numeric report lacks a claim-limit inventory",
    )
    expected_claim_limits = [
        value
        for value in numeric_claim_limits
        if not str(value).startswith("Pathology descriptions remain pending")
    ]
    expected_claim_limits.append(
        "Pathology descriptions are the imported structured assessments from the fresh "
        "corrected blinded packets."
    )
    expected_reviewed = {
        **numeric_report,
        "status": "ANALYSIS_AND_PATHOLOGY_REVIEW_COMPLETE",
        "claim_limits": expected_claim_limits,
        "pathology_review": pathology,
    }
    _require(
        _canonical(reviewed) == _canonical(expected_reviewed),
        "Corrected Aim4 reviewed report differs from its numeric report and sealed review",
    )
    return {
        "status": "PASS",
        "n_completed_prototypes": len(expected_prototypes),
        "receipt": _identity(receipt_path),
        "transaction_receipt": _identity(transaction_path),
        "forms": imported_forms,
        "pathology": pathology_identity,
        "reviewed_report": reviewed_identity,
    }


def _verify_e4_contract(
    root: Path,
    *,
    pathology_state: str,
    numeric_replay: dict[str, Any],
    final_replay: dict[str, Any] | None,
) -> dict[str, Any]:
    completion_path = root / "numeric_complete.json"
    verification_path = root / "numeric_verification.json"
    completion = _read_json(completion_path)
    verification = _read_json(verification_path)
    artifacts = completion.get("artifacts")
    _require(
        completion.get("schema_version") == 1
        and completion.get("component") == "aim4_corrected_cap8192"
        and completion.get("status") == "numeric_completed_pathology_review_pending"
        and isinstance(artifacts, dict)
        and bool(artifacts)
        and completion.get("aggregate_sha256")
        == hashlib.sha256(_canonical(artifacts).encode()).hexdigest()
        and completion.get("validation", {}).get("status") == "PASS"
        and completion.get("validation", {}).get("pathology_review")
        == "PENDING_EXPLICIT_COMPLETED_FORMS"
        and completion.get("pathology_status")
        == "fresh corrected blinded packets sealed; completed review pending"
        and Path(str(completion.get("output_root", ""))).resolve(strict=True) == root,
        "Corrected Aim4 numeric completion contract failed",
    )
    numeric_artifacts = {
        str(relative): _require_manifest_artifact(
            root, artifacts, relative, "Corrected Aim4 numeric completion"
        )
        for relative in artifacts
    }
    required_numeric_artifacts = {
        "lineage_start.json",
        "receipts/inputs.json",
        "receipts/review_packets.json",
        "receipts/source_snapshot.json",
        "source_snapshot/aim4_morphologic_atlas.py",
        "profiles/patient_profiles_k32.parquet",
        "analysis/specificity_k32.json",
        "analysis/aim4_corrected_k32.json",
    }
    _require(
        required_numeric_artifacts <= set(numeric_artifacts),
        "Corrected Aim4 numeric completion lacks a critical result/input artifact",
    )
    if pathology_state == "pending":
        _require(
            set(numeric_artifacts) == _e4_manifest_paths(root, phase="numeric"),
            "Corrected Aim4 numeric completion inventory is not exclusive",
        )
    expected_numeric_checks = {
        "frozen_source_snapshot",
        "numeric_completion_inventory_and_hashes",
        "all_15_attention_files_and_tile_order",
        "profile_and_candidate_exact_coverage",
        "fixed_32_prototype_families",
        "deterministic_statistical_replay",
        "fresh_separate_pending_review_packets",
    }
    _require(
        verification.get("schema_version") == 1
        and verification.get("component") == "aim4_corrected_cap8192"
        and verification.get("status") == "PASS"
        and verification.get("receipt_role") == "immutable_numeric_verification_addendum"
        and Path(str(verification.get("output_root", ""))).resolve(strict=True) == root
        and verification.get("numeric_completion") == _identity(completion_path)
        and verification.get("numeric_aggregate_sha256") == completion.get("aggregate_sha256")
        and verification.get("replay", {}).get("status") == "PASS"
        and _all_checks_pass(verification.get("checks"), expected_numeric_checks),
        "Corrected Aim4 numeric verification is not a bound full-replay PASS",
    )
    _require(
        _canonical(numeric_replay) == _canonical(verification)
        and numeric_replay.get("analysis", {}).get("fixed_bh_family_size") == K,
        "Corrected Aim4 archived replay did not validate fixed m=32 families",
    )
    specificity_path = root / "analysis" / "specificity_k32.json"
    report_path = root / "analysis" / "aim4_corrected_k32.json"
    specificity = _read_json(specificity_path)
    report = _read_json(report_path)
    protocol = specificity.get("protocol", {})
    expected_protocol = {
        "cap": CAP,
        "k": K,
        "seeds": [42, 43, 44],
        "n_bootstrap": E4_BOOTSTRAP,
        "bh_family": "fixed prototypes 0..31, including structural p=1 rows",
    }
    for key, expected in expected_protocol.items():
        _require(protocol.get(key) == expected, f"Corrected Aim4 protocol mismatch: {key}")
    _require(
        report.get("status") == "NUMERIC_ANALYSIS_COMPLETE_PATHOLOGY_REVIEW_PENDING"
        and report.get("protocol") == protocol,
        "Corrected Aim4 numeric report does not declare pathology pending",
    )
    _require(
        set(specificity.get("prototypes", {})) == set(map(str, range(K))),
        "Corrected Aim4 specificity does not contain prototypes 0..31",
    )
    for prototype in range(K):
        for population in ("A", "D"):
            effect = specificity["prototypes"][str(prototype)]["abundance"][population]
            p_value = _finite(effect.get("p"), f"Aim4 p{prototype}/{population} p")
            q_value = _finite(effect.get("q"), f"Aim4 p{prototype}/{population} q")
            _require(0 <= p_value <= 1 and 0 <= q_value <= 1, "Aim4 p/q outside [0,1]")
            _require(isinstance(effect.get("estimable"), bool), "Aim4 estimability is not Boolean")
            if not effect["estimable"]:
                _require(
                    effect.get("auc") is None
                    and p_value == 1.0
                    and effect.get("significant") is False,
                    f"Aim4 p{prototype}/{population} structural policy differs",
                )

    packets = _verify_packet_key_separation(root)
    completed_review: dict[str, Any] | None = None
    if pathology_state == "pending":
        _require(final_replay is None, "Unexpected final Aim4 delegate in pending mode")
        pathology = {
            "state": "PENDING",
            "numeric_results_valid": True,
            "pathology_claims_final": False,
        }
    else:
        final_completion_path = root / "lineage_complete.json"
        final_verification_path = root / "verification.json"
        final_completion = _read_json(final_completion_path)
        final_verification = _read_json(final_verification_path)
        final_artifacts = final_completion.get("artifacts")
        _require(
            final_completion.get("schema_version") == 1
            and final_completion.get("component") == "aim4_corrected_cap8192"
            and final_completion.get("status") == "completed"
            and isinstance(final_artifacts, dict)
            and bool(final_artifacts)
            and final_completion.get("aggregate_sha256")
            == hashlib.sha256(_canonical(final_artifacts).encode()).hexdigest()
            and Path(str(final_completion.get("output_root", ""))).resolve(strict=True) == root,
            "Corrected Aim4 final completion hash contract failed",
        )
        validated_final_artifacts = {
            str(relative): _require_manifest_artifact(
                root,
                final_artifacts,
                relative,
                "Corrected Aim4 final completion",
            )
            for relative in final_artifacts
        }
        _require(
            set(validated_final_artifacts) == _e4_manifest_paths(root, phase="final"),
            "Corrected Aim4 final completion inventory is not exclusive",
        )
        required_final_artifacts = {
            "numeric_verification.json",
            "receipts/review_import.json",
            f"review_completion/k{K}/_import_receipt_DO_NOT_SHARE.json",
            f"review_completion/k{K}/submissions/base_completed.csv",
            f"review_completion/k{K}/submissions/attention_addendum_completed.csv",
            f"review_completion/k{K}/pathology_review_k{K}.json",
            f"review_completion/k{K}/aim4_corrected_reviewed_k{K}.json",
        }
        _require(
            required_final_artifacts <= set(validated_final_artifacts),
            "Corrected Aim4 final completion lacks a critical result/input artifact",
        )
        completed_review = _verify_completed_review_import(
            root,
            final_artifacts,
            report,
        )
        expected_final_checks = {
            "upstream_identities",
            "completion_inventory_and_hashes",
            "all_15_attention_files_and_tile_order",
            "profile_and_candidate_exact_coverage",
            "fixed_32_prototype_families",
            "deterministic_statistical_replay",
            "fresh_separate_review_packets",
        }
        numeric_seal = final_verification.get("numeric_seal", {})
        _require(
            final_replay is not None
            and _canonical(final_replay) == _canonical(final_verification)
            and final_verification.get("schema_version") == 1
            and final_verification.get("component") == "aim4_corrected_cap8192"
            and final_replay.get("status") == "PASS"
            and final_verification.get("receipt_role")
            == "immutable_completion_verification_addendum"
            and Path(str(final_verification.get("output_root", ""))).resolve(strict=True) == root
            and final_verification.get("completion") == _identity(final_completion_path)
            and final_verification.get("completion_aggregate_sha256")
            == final_completion.get("aggregate_sha256")
            and final_verification.get("replay", {}).get("status") == "PASS"
            and _all_checks_pass(final_verification.get("checks"), expected_final_checks)
            and numeric_seal.get("status") == "PASS"
            and numeric_seal.get("numeric_completion") == _identity(completion_path)
            and numeric_seal.get("numeric_verification") == _identity(verification_path)
            and numeric_seal.get("numeric_aggregate_sha256") == completion.get("aggregate_sha256")
            and final_replay.get("completed_pathology_review", {}).get("status") == "PASS",
            "Corrected Aim4 final pathology replay did not PASS",
        )
        _require(
            final_completion.get("status") == "completed"
            and final_completion.get("pathology_status")
            == "completed structured blinded pathology review imported and sealed",
            "Corrected Aim4 final completion pathology semantics differ",
        )
        pathology = {
            "state": "FINAL_COMPLETE",
            "numeric_results_valid": True,
            "pathology_claims_final": True,
        }
    return {
        "status": "PASS",
        "pathology": pathology,
        "fixed_bh_family_size": K,
        "n_bootstrap": E4_BOOTSTRAP,
        "packets": packets,
        "artifacts": {
            "numeric_completion": _identity(completion_path),
            "numeric_verification": _identity(verification_path),
            "prepared_inputs": numeric_artifacts["receipts/inputs.json"],
            "patient_profiles": _identity(root / "profiles" / "patient_profiles_k32.parquet"),
            "specificity": _identity(specificity_path),
            "numeric_report": _identity(report_path),
            **(
                {
                    "final_completion": _identity(root / "lineage_complete.json"),
                    "final_verification": _identity(root / "verification.json"),
                    "completed_review_import": completed_review,
                }
                if pathology_state == "final"
                else {}
            ),
        },
    }


def _effect_gate(row: dict[str, Any], population: str, label: str) -> bool:
    estimable = row.get(f"estimable_{population}")
    significant = row.get(f"significant_{population}")
    _require(isinstance(estimable, bool), f"{label} estimability is not Boolean")
    _require(isinstance(significant, bool), f"{label} significance is not Boolean")
    p_value = _finite(row.get(f"p_{population}"), f"{label} p")
    q_value = _finite(row.get(f"q_{population}"), f"{label} q")
    _require(0 <= p_value <= 1 and 0 <= q_value <= 1, f"{label} p/q outside [0,1]")
    if not estimable:
        _require(
            row.get(f"auc_{population}") is None
            and row.get(f"ci_low_{population}") is None
            and row.get(f"ci_high_{population}") is None
            and p_value == 1.0
            and significant is False,
            f"{label} structural/non-estimable encoding differs",
        )
        return False
    auc = _finite(row.get(f"auc_{population}"), f"{label} AUC")
    low = _finite(row.get(f"ci_low_{population}"), f"{label} CI low")
    high = _finite(row.get(f"ci_high_{population}"), f"{label} CI high")
    _require(low <= high, f"{label} CI is reversed")
    expected_significant = bool(q_value < FDR_ALPHA and (low > 0.5 or high < 0.5))
    _require(significant is expected_significant, f"{label} significant flag algebra mismatch")
    return bool(auc > 0.5 and low > 0.5 and q_value < FDR_ALPHA and significant)


def _verify_stability_contract(
    root: Path,
    e4_root: Path,
    *,
    delegated_replay: dict[str, Any],
) -> dict[str, Any]:
    completion_path = root / "completion_receipt.json"
    verification_path = root / "verification_receipt.json"
    results_path = root / "results.json"
    input_path = root / "input_receipt.json"
    completion = _read_json(completion_path)
    verification = _read_json(verification_path)
    results = _read_json(results_path)
    inputs = _read_json(input_path)
    payload_inventory = _inventory_by_path(
        completion.get("payload_inventory"), "Aim4 stability payload inventory"
    )
    _require(
        completion.get("schema_version") == STABILITY_SCHEMA_VERSION
        and completion.get("status") == "COMPLETE"
        and completion.get("results_sha256") == _sha256_file(results_path)
        and completion.get("input_receipt_sha256") == _sha256_file(input_path)
        and Path(str(completion.get("output_root", ""))).resolve(strict=True) == root,
        "Aim4 stability completion contract failed",
    )
    validated_payload = {
        relative: _require_inventory_artifact(
            root,
            payload_inventory,
            relative,
            "Aim4 stability payload inventory",
        )
        for relative in payload_inventory
    }
    observed_payload_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix()
        not in {"completion_receipt.json", "verification_receipt.json"}
    }
    _require(
        set(validated_payload) == observed_payload_paths,
        "Aim4 stability current payload inventory is partial or has unsealed extras",
    )
    required_payload = {
        "results.json",
        "input_receipt.json",
        "feature_inventory.json",
        "development_patient_abundance.parquet",
        "association_effects.csv",
        "all_anchor_correspondence.csv",
        "source_snapshot/manifest.json",
        "source_snapshot/tools/aim4_vocab_stability.py",
    }
    _require(
        required_payload <= set(validated_payload),
        "Aim4 stability payload lacks a critical result/input artifact",
    )
    required_replay_checks = {
        "exclusive_payload_inventory",
        "input_hashes",
        "source_snapshot",
        "assignment_and_centroid_shapes",
        "statistical_recomputation",
        "full_m32_A_D_family_recomputation",
        "canonical_reconstruction_control",
        "exact_feature_and_assignment_replay",
        "deterministic_inertia_replay",
    }
    _require(
        verification.get("schema_version") == STABILITY_SCHEMA_VERSION
        and verification.get("status") == "PASS"
        and Path(str(verification.get("output_root", ""))).resolve(strict=True) == root
        and verification.get("replay_input") is True
        and verification.get("completion_sha256") == _sha256_file(completion_path)
        and verification.get("results_sha256") == _sha256_file(results_path)
        and _all_checks_pass(verification.get("checks"), required_replay_checks)
        and verification.get("checks", {}).get("pathology_label_transfer")
        == "PROHIBITED_WITHOUT_NEW_BLINDED_REVIEW",
        "Aim4 stability receipt is not a bound full-input replay PASS",
    )
    delegated_without_time = dict(delegated_replay)
    sealed_without_time = dict(verification)
    delegated_without_time.pop("verified_at_utc", None)
    sealed_without_time.pop("verified_at_utc", None)
    _require(
        delegated_replay.get("status") == "PASS"
        and _canonical(delegated_without_time) == _canonical(sealed_without_time),
        "Archived stability replay evidence differs from its sealed receipt",
    )
    _require(
        inputs.get("schema_version") == STABILITY_SCHEMA_VERSION
        and Path(str(inputs.get("corrected_aim4_root", ""))).resolve(strict=True)
        == e4_root,
        "Aim4 stability is not bound to the explicit corrected Aim4 root",
    )
    identities = inputs.get("input_identities", {})
    _require(
        isinstance(identities, dict) and bool(identities),
        "Aim4 stability input receipt lacks hashed input identities",
    )
    for label, record in identities.items():
        _require(
            isinstance(record, dict) and isinstance(record.get("path"), str),
            f"Stability input identity is malformed: {label}",
        )
        _require(
            record == _identity(Path(record["path"])),
            f"Stability input identity drifted: {label}",
        )
    expected_e4_inputs = {
        "corrected_aim4_numeric_completion": e4_root / "numeric_complete.json",
        "corrected_aim4_numeric_verification": e4_root / "numeric_verification.json",
        "canonical_patient_profiles": e4_root / "profiles" / "patient_profiles_k32.parquet",
        "canonical_specificity": e4_root / "analysis" / "specificity_k32.json",
    }
    for label, path in expected_e4_inputs.items():
        _require(
            identities.get(label) == _identity(path), f"Stability input binding differs: {label}"
        )

    source_manifest = inputs.get("source_snapshot")
    _require(
        isinstance(source_manifest, list) and bool(source_manifest),
        "Aim4 stability input receipt lacks its source snapshot manifest",
    )
    try:
        sealed_source_manifest = json.loads(
            (root / "source_snapshot" / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FullVerificationError(
            f"Cannot read Aim4 stability source manifest: {error}"
        ) from error
    _require(
        _canonical(sealed_source_manifest) == _canonical(source_manifest),
        "Aim4 stability source manifest differs from its input receipt",
    )
    for index, record in enumerate(source_manifest):
        _require(isinstance(record, dict), f"Stability source row {index} is invalid")
        relative = record.get("path")
        _require(isinstance(relative, str), f"Stability source row {index} lacks path")
        observed = _identity(
            _root_file(root / "source_snapshot", relative, "Stability source snapshot")
        )
        _require(
            record.get("size_bytes") == observed["size_bytes"]
            and record.get("sha256") == observed["sha256"],
            f"Stability source snapshot hash differs: {relative}",
        )
    _require(
        inputs.get("feature_inventory_sha256") == _sha256_file(root / "feature_inventory.json"),
        "Aim4 stability feature-inventory receipt hash differs",
    )

    protocol = results.get("protocol", {})
    _require(
        results.get("schema_version") == STABILITY_SCHEMA_VERSION
        and results.get("status") == "COMPLETE"
        and protocol.get("k_values") == list(STABILITY_K_VALUES)
        and protocol.get("cluster_seeds") == list(STABILITY_SEEDS)
        and protocol.get("canonical_k") == K
        and protocol.get("n_init") == 10
        and protocol.get("association_bootstrap_replicates") == STABILITY_BOOTSTRAP
        and protocol.get("association_fdr_alpha") == FDR_ALPHA
        and protocol.get("association_family_size") == K
        and protocol.get("inertia_contract") == STABILITY_INERTIA_CONTRACT,
        "Aim4 stability protocol/capacity/full-bootstrap contract failed",
    )
    variant_metrics = results.get("variant_metrics")
    _require(
        isinstance(variant_metrics, list)
        and len(variant_metrics) == len(STABILITY_VARIANTS),
        "Aim4 stability variant-metric inventory differs",
    )
    metric_variants: set[str] = set()
    for row in variant_metrics:
        _require(isinstance(row, dict), "Aim4 stability variant metric is malformed")
        variant = str(row.get("variant", ""))
        _require(
            variant in STABILITY_VARIANTS and variant not in metric_variants,
            f"Aim4 stability variant metric is unexpected or duplicated: {variant}",
        )
        expected_k, expected_seed = variant.removeprefix("k").split("_seed", 1)
        model_inertia = _finite(
            row.get("model_inertia"), f"{variant} model inertia"
        )
        replay_inertia = _finite(
            row.get("replay_inertia_float64"), f"{variant} replay inertia"
        )
        _require(
            int(row.get("k", -1)) == int(expected_k)
            and int(row.get("seed", -1)) == int(expected_seed)
            and int(row.get("n_iter", -1)) > 0
            and model_inertia > 0.0
            and replay_inertia >= 0.0
            and "inertia" not in row,
            f"Aim4 stability inertia contract failed: {variant}",
        )
        metric_variants.add(variant)
    _require(
        metric_variants == set(STABILITY_VARIANTS),
        "Aim4 stability variant-metric grid is incomplete",
    )
    control_protocol = protocol.get("canonical_reconstruction_control", {})
    control = results.get("canonical_reconstruction_control", {})
    _require(
        control_protocol.get("sample_values_sha256")
        == STABILITY_CANONICAL_SAMPLE_SHA256
        and control_protocol.get("maximum_cluster_mass_total_variation")
        == STABILITY_MAX_CLUSTER_MASS_TOTAL_VARIATION
        and control_protocol.get("maximum_single_cluster_mass_delta")
        == STABILITY_MAX_SINGLE_CLUSTER_MASS_DELTA
        and control_protocol.get("same_seed_refit_role")
        == "sensitivity_variant_not_an_identity_control",
        "Aim4 stability canonical reconstruction protocol differs",
    )
    training_sizes = control.get("training_cluster_sizes")
    observed_sizes = control.get("reprojected_cluster_sizes")
    absolute_differences = control.get("absolute_cluster_size_differences")
    sampled_tiles = int(results.get("counts", {}).get("sampled_tiles", -1))
    _require(
        isinstance(training_sizes, list)
        and isinstance(observed_sizes, list)
        and isinstance(absolute_differences, list)
        and len(training_sizes) == len(observed_sizes) == len(absolute_differences) == K
        and all(isinstance(value, int) and value > 0 for value in training_sizes)
        and all(isinstance(value, int) and value > 0 for value in observed_sizes)
        and sum(training_sizes) == sum(observed_sizes) == sampled_tiles
        and absolute_differences
        == [
            abs(observed - training)
            for training, observed in zip(training_sizes, observed_sizes, strict=True)
        ],
        "Aim4 stability canonical reconstruction cluster-mass inventory differs",
    )
    expected_tv = sum(absolute_differences) / (2.0 * sampled_tiles)
    expected_maximum = max(absolute_differences) / sampled_tiles
    same_seed = control.get("same_seed_refit", {})
    _require(
        control.get("name") == "exact_sample_and_frozen_centroid_reprojection"
        and control.get("sample_values_sha256_expected")
        == STABILITY_CANONICAL_SAMPLE_SHA256
        and control.get("sample_values_sha256_observed")
        == STABILITY_CANONICAL_SAMPLE_SHA256
        and control.get("sample_values_sha256_exact") is True
        and control.get("maximum_cluster_mass_total_variation_allowed")
        == STABILITY_MAX_CLUSTER_MASS_TOTAL_VARIATION
        and control.get("maximum_single_cluster_mass_delta_allowed")
        == STABILITY_MAX_SINGLE_CLUSTER_MASS_DELTA
        and math.isclose(
            _finite(control.get("cluster_mass_total_variation"), "cluster-mass TV"),
            expected_tv,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            _finite(control.get("maximum_cluster_mass_delta"), "maximum cluster-mass delta"),
            expected_maximum,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and expected_tv <= STABILITY_MAX_CLUSTER_MASS_TOTAL_VARIATION
        and expected_maximum <= STABILITY_MAX_SINGLE_CLUSTER_MASS_DELTA
        and same_seed.get("variant") == "k32_seed20260819"
        and same_seed.get("role") == "sensitivity_variant_not_an_identity_control"
        and same_seed.get("included_in_all_nine_biological_gate") is True
        and control.get("pass") is True,
        "Aim4 stability canonical reconstruction control failed",
    )
    reconstruction = results.get("canonical_patient_profile_reconstruction", {})
    effect_reconstruction = results.get("downstream_abundance_claim_stability", {}).get(
        "canonical_effect_reconstruction", {}
    )
    _require(
        reconstruction.get("pass") is True
        and _finite(
            reconstruction.get("maximum_absolute_abundance_difference"), "profile reconstruction"
        )
        <= 1e-12
        and effect_reconstruction.get("pass") is True
        and _finite(
            effect_reconstruction.get("maximum_absolute_auc_difference"), "AUC reconstruction"
        )
        <= 1e-12
        and _finite(effect_reconstruction.get("maximum_absolute_p_difference"), "p reconstruction")
        <= 1e-12
        and _finite(effect_reconstruction.get("maximum_absolute_q_difference"), "q reconstruction")
        <= 1e-12,
        "Aim4 stability canonical k32 numeric reconstruction failed",
    )

    downstream = results.get("downstream_abundance_claim_stability", {})
    rows = downstream.get("association_effects")
    _require(
        isinstance(rows, list) and len(rows) == 10 * K,
        "Stability association family inventory differs",
    )
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    expected_all_variants = {"canonical_k32", *STABILITY_VARIANTS}
    for row in rows:
        variant = str(row.get("variant"))
        anchor = int(row.get("anchor_prototype", -1))
        key = (variant, anchor)
        _require(
            variant in expected_all_variants and 0 <= anchor < K, f"Unexpected stability row {key}"
        )
        _require(key not in by_key, f"Duplicate stability row {key}")
        _require(
            row.get("family_size_A") == K and row.get("family_size_D") == K,
            f"Stability row {key} is not in fixed m=32 A/D families",
        )
        pass_a = _effect_gate(row, "A", f"{variant}/p{anchor}/A")
        pass_d = _effect_gate(row, "D", f"{variant}/p{anchor}/D")
        expected_joint = bool(pass_a and pass_d)
        _require(
            row.get("positive_significant_A_and_D") is expected_joint,
            f"Stability row {key} A-and-D gate algebra mismatch",
        )
        by_key[key] = row
    _require(
        set(by_key)
        == {(variant, anchor) for variant in expected_all_variants for anchor in range(K)},
        "Stability association rows do not form exact 10 x 32 inventory",
    )

    summaries = downstream.get("claim_stability", {})
    claim_verdicts: dict[str, str] = {}
    for anchor, montage in ANCHORS.items():
        passing = [
            variant
            for variant in STABILITY_VARIANTS
            if by_key[(variant, anchor)]["positive_significant_A_and_D"]
        ]
        failing = [variant for variant in STABILITY_VARIANTS if variant not in passing]
        all_pass = not failing
        expected_verdict = (
            "ROBUST_ACROSS_PREDECLARED_K_SEED_GRID"
            if all_pass
            else "CONDITIONAL_K32_NOT_STABLE_ACROSS_FULL_GRID"
        )
        summary = summaries.get(str(anchor), {})
        _require(
            summary.get("montage_id") == montage
            and summary.get("n_variants") == 9
            and summary.get("passing_variants") == passing
            and summary.get("failing_variants") == failing
            and summary.get("all_variants_pass") is all_pass
            and summary.get("verdict") == expected_verdict,
            f"Stability p{anchor}/{montage} claim-verdict algebra mismatch",
        )
        _require(
            "pathology-unlabeled" in str(summary.get("claim_limit", "")),
            f"Stability p{anchor}/{montage} lacks pathology non-transfer guardrail",
        )
        claim_verdicts[str(anchor)] = expected_verdict

    scope_limit = str(results.get("scope", {}).get("claim_limit", ""))
    _require(
        "do not transfer M04/M07 pathology identities" in scope_limit,
        "Stability result does not prohibit pathology label transfer",
    )
    return {
        "status": "PASS",
        "fixed_bh_family_size": K,
        "n_bootstrap": STABILITY_BOOTSTRAP,
        "inertia_contract": STABILITY_INERTIA_CONTRACT,
        "claim_verdicts": claim_verdicts,
        "pathology_label_transfer": "PROHIBITED_WITHOUT_NEW_BLINDED_REVIEW",
        "artifacts": {
            "completion": _identity(completion_path),
            "full_replay_verification": _identity(verification_path),
            "results": validated_payload["results.json"],
            "inputs": validated_payload["input_receipt.json"],
            "patient_profiles": _identity(root / "development_patient_abundance.parquet"),
            "association_effects": _identity(root / "association_effects.csv"),
            "all_anchor_correspondence": _identity(root / "all_anchor_correspondence.csv"),
        },
    }


def verify(
    *,
    aim2_core_root: Path,
    e2c_root: Path,
    aim2_refresh_root: Path,
    aim3_root: Path,
    aim3_repeated_root: Path,
    aim4_root: Path,
    aim4_stability_root: Path,
    aim4_pathology_state: str,
    fast_sealed_receipts: bool = False,
) -> dict[str, Any]:
    _require(
        aim4_pathology_state in {"pending", "final"},
        "--aim4-pathology-state must be pending or final",
    )
    roots = {
        "aim2_core": _absolute_root(aim2_core_root, "--aim2-core-root"),
        "e2c_offset": _absolute_root(e2c_root, "--e2c-root"),
        "aim2_e2d_refresh": _absolute_root(aim2_refresh_root, "--aim2-refresh-root"),
        "aim3_corrected": _absolute_root(aim3_root, "--aim3-root"),
        "aim3_repeated_controls": _absolute_root(aim3_repeated_root, "--aim3-repeated-root"),
        "aim4_corrected": _absolute_root(aim4_root, "--aim4-root"),
        "aim4_vocab_stability": _absolute_root(aim4_stability_root, "--aim4-stability-root"),
    }
    _validate_distinct_roots(roots)
    _precheck_roots(roots, pathology_state=aim4_pathology_state)
    verification_mode = FAST_RECEIPT_MODE if fast_sealed_receipts else FULL_REPLAY_MODE

    aim23_result = aim23.verify(
        aim2_core_root=roots["aim2_core"],
        e2c_root=roots["e2c_offset"],
        aim2_refresh_root=roots["aim2_e2d_refresh"],
        aim3_root=roots["aim3_corrected"],
        aim3_repeated_root=roots["aim3_repeated_controls"],
    )
    aim23_summary = _verify_aim23_delegate(aim23_result)
    aim23_roots = _verify_aim23_root_budgets(roots)

    numeric_replay, final_replay, e4_verifier = _delegate_e4(
        roots["aim4_corrected"],
        roots["aim2_core"],
        pathology_state=aim4_pathology_state,
        execute_replay=not fast_sealed_receipts,
    )
    e4_summary = _verify_e4_contract(
        roots["aim4_corrected"],
        pathology_state=aim4_pathology_state,
        numeric_replay=numeric_replay,
        final_replay=final_replay,
    )
    e4_summary["verification_evidence"] = verification_mode

    stability_replay, stability_verifier = _delegate_stability(
        roots["aim4_vocab_stability"],
        execute_replay=not fast_sealed_receipts,
    )
    stability_summary = _verify_stability_contract(
        roots["aim4_vocab_stability"],
        roots["aim4_corrected"],
        delegated_replay=stability_replay,
    )
    stability_summary["verification_evidence"] = verification_mode

    if fast_sealed_receipts:
        aim4_component_results = {
            "aim4_numeric_full_replay_receipt": numeric_replay,
            **(
                {"aim4_final_full_replay_receipt": final_replay} if final_replay is not None else {}
            ),
            "aim4_stability_full_replay_receipt": stability_replay,
        }
    else:
        aim4_component_results = {
            "aim4_numeric_replay": numeric_replay,
            **({"aim4_final_replay": final_replay} if final_replay is not None else {}),
            "aim4_stability_replay": stability_replay,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "scope": "corrected Aim2 + Aim3 + Aim4, cap 8192",
        "roots": {name: str(path) for name, path in roots.items()},
        "pathology_state": aim4_pathology_state,
        "verification_mode": verification_mode,
        "fresh_aim4_replay_executed": not fast_sealed_receipts,
        "checks": {
            "aim2_aim3_existing_consolidated_delegate": aim23_summary,
            "aim2_aim3_result_hashes_and_budgets": aim23_roots,
            "aim4_corrected_archived_numeric_delegate": e4_summary,
            "aim4_vocab_stability_archived_full_replay_delegate": stability_summary,
        },
        "component_results": {
            "aim2_aim3": aim23_result,
            **aim4_component_results,
        },
        "central_claims": {
            "aim3_primary_familywise_rungs_passing": aim23_summary["aim3_familywise_rungs_passing"],
            "aim3_repeated_consensus_verdicts": aim23_summary["aim3_repeated_consensus_verdicts"],
            "aim4_p17_p28_abundance_stability": stability_summary["claim_verdicts"],
            "aim4_pathology": e4_summary["pathology"],
            "alternative_cluster_pathology_identity": "NOT_TRANSFERRED",
        },
        "verifiers": {
            "aim2_aim3": _identity(REPO / "tools" / "verify_corrected_aim2_aim3.py"),
            "aim4_corrected_archived": e4_verifier,
            "aim4_stability_archived": stability_verifier,
            "this_consolidated_verifier": _identity(Path(__file__)),
        },
    }


def _receipt_destination(value: str | Path, roots: tuple[Path, ...]) -> Path:
    path = Path(value).expanduser()
    _require(path.is_absolute(), f"--write-receipt must be an absolute path: {path}")
    resolved = path.resolve(strict=False)
    _require(resolved.parent.is_dir(), f"Receipt parent does not exist: {resolved.parent}")
    for root in roots:
        _require(
            resolved != root and root not in resolved.parents,
            f"Receipt must be outside immutable root: {root}",
        )
    return resolved


def _write_json_once_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite consolidated receipt: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aim2-core-root", required=True)
    parser.add_argument("--e2c-root", required=True)
    parser.add_argument("--aim2-refresh-root", required=True)
    parser.add_argument("--aim3-root", required=True)
    parser.add_argument("--aim3-repeated-root", required=True)
    parser.add_argument("--aim4-root", required=True)
    parser.add_argument("--aim4-stability-root", required=True)
    parser.add_argument(
        "--aim4-pathology-state",
        choices=("pending", "final"),
        required=True,
        help="Explicitly distinguish numeric-complete/review-pending from final pathology",
    )
    parser.add_argument(
        "--fast-sealed-receipts",
        action="store_true",
        help=(
            "skip fresh E4/stability computation only after validating their "
            "sealed full-replay PASS receipt and hash chains"
        ),
    )
    parser.add_argument("--write-receipt", metavar="ABS_JSON")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        roots = tuple(
            _absolute_root(value, label)
            for value, label in (
                (args.aim2_core_root, "--aim2-core-root"),
                (args.e2c_root, "--e2c-root"),
                (args.aim2_refresh_root, "--aim2-refresh-root"),
                (args.aim3_root, "--aim3-root"),
                (args.aim3_repeated_root, "--aim3-repeated-root"),
                (args.aim4_root, "--aim4-root"),
                (args.aim4_stability_root, "--aim4-stability-root"),
            )
        )
        result = verify(
            aim2_core_root=roots[0],
            e2c_root=roots[1],
            aim2_refresh_root=roots[2],
            aim3_root=roots[3],
            aim3_repeated_root=roots[4],
            aim4_root=roots[5],
            aim4_stability_root=roots[6],
            aim4_pathology_state=args.aim4_pathology_state,
            fast_sealed_receipts=args.fast_sealed_receipts,
        )
        if args.write_receipt:
            destination = _receipt_destination(args.write_receipt, roots)
            _write_json_once_atomic(destination, result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (
        FullVerificationError,
        aim23.ConsolidatedVerificationError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"FAIL — {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
