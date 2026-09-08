#!/usr/bin/env python3
"""Versioned v14 correspondence resume; missing archive access is operational.

Old preliminary missing-input reports are preserved. Each dependency state and
ready geometry run has a new content-addressed directory. No geometry is opened
while the archive root is absent. The original four pins, all nine variants,
direct joint anchor assignment, and frozen naming decision remain controlling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import final_v14_post_reader as original  # noqa: E402

OUT = REPO / "reports/reruns/final_v14_additions_20260903/e4v_correspondence_replay"
PRE = original.PRE
POST = original.POST
HOT_ROOT = original.pre.HOT_ROOT
LEGACY_INPUTS = original.pre.LEGACY_INPUTS
AXES = {"p17": "extracellular mucin/mucinous pattern",
        "p28": "malignant gland-forming epithelium/gland–lumen"}


class PendingDependency(RuntimeError):
    """No scientific geometry decision has been made because inputs are unavailable."""


def content_id(value: dict) -> str:
    return hashlib.sha256(original.json_bytes(value)).hexdigest()


def publish_version(path: Path, payload: dict) -> dict:
    """Idempotently retain the first timestamp for unchanged immutable content."""
    if path.exists():
        old = original.verify_seal(path)
        check = dict(old)
        check.pop("created_utc", None)
        if check != payload:
            raise original.PostReaderError(f"Versioned payload conflict: {path}")
        return old
    result = {**payload, "created_utc": original.now()}
    original.publish(path, result)
    return result


def pinned_file(path: Path, expected_digest: str) -> dict:
    """Follow an existing archive link only when its resolved bytes match the pin."""
    record = original.identity(path.resolve(strict=True))
    if record["sha256"] != expected_digest:
        raise original.PostReaderError(f"Preregistered legacy digest mismatch: {path}")
    return record


def preflight() -> tuple[dict, Path]:
    original.intake()
    naming_path = POST / "naming/naming_freeze.json"
    names = original.verify_seal(naming_path)
    gate = names["name_gate_status"]
    if gate not in ("NAME_GATE_PASS", "NAME_GATE_FAIL"):
        raise original.PostReaderError("Missing frozen name gate")
    issues, legacy = [], {}
    if not HOT_ROOT.is_dir():
        issues.append({"path": str(HOT_ROOT), "reason": "ARCHIVE_ROOT_UNAVAILABLE"})
    for name, (path, expected) in LEGACY_INPUTS.items():
        if not path.is_file():
            issues.append({"path": str(path), "input": name, "reason": "MISSING"})
        elif HOT_ROOT.is_dir():
            try:
                legacy[name] = pinned_file(path, expected)
            except original.PostReaderError as error:
                issues.append({"path": str(path), "input": name, "reason": "DIGEST_MISMATCH", "detail": str(error)})
    variants = []
    # Local array bytes are authenticated only once the archive dependency is
    # restored. This preflight never loads centroids or computes any mapping.
    if not issues:
        for k in original.pre.VARIANT_K:
            for seed in original.pre.VARIANT_SEEDS:
                path = original.pre._vocabulary_path(PRE, fold=None, k=k, seed=seed)
                receipt_path = path.with_name(path.name + ".receipt.json")
                receipt = original.read_json(receipt_path)
                original.check_identity(receipt["artifacts"]["vocabulary"], path)
                basis = receipt["artifacts"]["pca_basis"]
                original.check_identity(basis)
                variants.append({"k": k, "seed": seed, "vocabulary": original.identity(path),
                                 "receipt": original.identity(receipt_path), "pca_basis": original.identity(Path(basis["path"]))})
    prior = [POST / "correspondence" / name for name in (
        "results.json", "resolved_legacy_path_amendment.json", "geometry_availability_correction.json")]
    payload = {"schema_version": 1, "component": "final_v14_correspondence_resume_preflight",
               "status": "BLOCKED_OPERATIONAL_DEPENDENCY" if issues else "READY_FOR_GEOMETRY",
               "scientific_status": "PENDING_NOT_TESTED", "geometry_computed": False,
               "name_gate_status": gate,
               "name_dependent_criteria_status": "CORRESPONDENCE_NOT_EVALUABLE_NAME_GATE_FAIL" if gate == "NAME_GATE_FAIL" else "PENDING_GEOMETRY",
               "naming_freeze": original.identity(naming_path),
               "literal_intake": original.identity(POST / "literal_intake/receipt.json"),
               "runner": original.identity(Path(__file__)), "imported_runner": original.identity(Path(original.__file__)),
               "archive_root": str(HOT_ROOT), "issues": issues, "legacy_inputs": legacy, "variants": variants,
               "historical_reports_preserved": [original.identity(p) for p in prior if p.is_file()],
               "scope": "Archive absence is pending operational access, not a scientific correspondence failure. Frozen NAME_GATE_FAIL independently precludes named criteria and spotlights."}
    path = OUT / "dependency_preflight" / (content_id(payload) + ".json")
    return publish_version(path, payload), path


def normalized_centroids(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        mean = np.asarray(archive["pca_mean"], dtype=np.float64)
        components = np.asarray(archive["pca_components"], dtype=np.float64)
        centers = np.asarray(archive["centroids"], dtype=np.float64)
    if mean.ndim != 1 or components.ndim != 2 or centers.ndim != 2 or components.shape[1] != len(mean) or centers.shape[1] != components.shape[0]:
        raise original.PostReaderError(f"Required PCA geometry arrays malformed: {path}")
    vectors = mean[None, :] + centers @ components
    length = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(length <= 0):
        raise original.PostReaderError(f"Nonfinite/zero legacy geometry: {path}")
    return vectors / length


def geometry_table(anchors: np.ndarray, variants: list[tuple[int, int, np.ndarray]]) -> list[dict]:
    rows = []
    for k, seed, vectors in variants:
        if len(vectors) != k or anchors.shape != (2, vectors.shape[1]):
            raise original.PostReaderError("Legacy-to-variant geometry dimension mismatch")
        cosine = anchors @ vectors.T
        assigned = original.joint_anchor_assignment(cosine)
        rows.append({"k": k, "seed": seed, "canonical": k == 32 and seed == 20260819,
                     "joint_cosine_objective": float(cosine[0, assigned[0]] + cosine[1, assigned[1]]),
                     **{axis: {"prototype_id": assigned[i], "cosine": float(cosine[i, assigned[i]]),
                               "geometry_pass": bool(cosine[i, assigned[i]] >= 0.8)}
                        for i, axis in enumerate(AXES)}})
    return rows


def axis_statuses(names: dict, rows: list[dict]) -> dict:
    axes = {}
    for axis, category in AXES.items():
        canonical = next(row[axis] for row in rows if row["canonical"])
        passes = sum(row[axis]["geometry_pass"] for row in rows)
        compatible = None
        if names["name_gate_status"] == "NAME_GATE_FAIL":
            status = "CORRESPONDENCE_NOT_EVALUABLE"
        else:
            controlling = next(row["response"] for row in names["controlling_reads"] if row["prototype_id"] == canonical["prototype_id"])
            compatible = "exact" if controlling["primary_category"] == category else "partial" if category in (controlling["secondary_category_1"], controlling["secondary_category_2"]) else "different"
            status = "CORRESPONDENCE_CRITERIA_MET" if canonical["geometry_pass"] and passes >= 7 and compatible in ("exact", "partial") else "CORRESPONDENCE_CRITERIA_NOT_MET"
        axes[axis] = {"status": status, "legacy_category": category,
                      "canonical_geometry": canonical, "geometry_passes": passes, "variant_denominator": 9,
                      "controlling_category_compatibility": compatible, "name_gate_status": names["name_gate_status"]}
    return axes


def run() -> tuple[dict, Path]:
    dependency, dependency_path = preflight()
    if dependency["status"] != "READY_FOR_GEOMETRY":
        raise PendingDependency(f"Archive inputs unavailable; scientific geometry remains pending. Dependency receipt: {dependency_path}")
    names = original.verify_seal(Path(dependency["naming_freeze"]["path"]))
    contract_payload = {"schema_version": 1, "status": "CORRESPONDENCE_GEOMETRY_CONTRACT_SEALED",
                        "dependency_receipt": original.identity(dependency_path),
                        "legacy_inputs": dependency["legacy_inputs"], "variants": dependency["variants"],
                        "name_gate_status": names["name_gate_status"], "runner": dependency["runner"],
                        "mapping": "Direct legacy p17/p28 to each of nine variants; maximum joint cosine under distinct destination constraint; ties within1e-12 choose lexical pair; cosine>=0.80; canonical plus>=7/9 required.",
                        "geometry_precision": "float64 backprojection and L2 normalization",
                        "historical_reports_preserved": dependency["historical_reports_preserved"]}
    run_root = OUT / "runs" / content_id(contract_payload)
    contract_path = run_root / "contract.json"
    publish_version(contract_path, contract_payload)
    result_path = run_root / "results.json"
    if result_path.exists():
        result = original.verify_seal(result_path)
        original.check_identity(result["contract"])
        return result, result_path
    legacy = normalized_centroids(Path(dependency["legacy_inputs"]["legacy_vocab_npz"]["path"]))
    if len(legacy) <= 28:
        raise original.PostReaderError("Pinned legacy vocabulary lacks p17/p28")
    variants = [(item["k"], item["seed"], normalized_centroids(Path(item["vocabulary"]["path"]))) for item in dependency["variants"]]
    if len(variants) != 9:
        raise original.PostReaderError("All nine variants are required")
    rows = geometry_table(legacy[[17, 28]], variants)
    result_payload = {"schema_version": 1, "status": "CORRESPONDENCE_GEOMETRY_COMPLETE",
                      "contract": original.identity(contract_path), "legacy_input_pins": "ALL_FOUR_PASS",
                      "variants": rows, "axes": axis_statuses(names, rows), "name_gate_status": names["name_gate_status"],
                      "claim_limit": "Legacy vocabulary was target-inclusive/transductive. Geometric correspondence is descriptive, cannot choose features, and cannot rescue a failed naming gate."}
    return publish_version(result_path, result_payload), result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run"))
    arguments = parser.parse_args()
    try:
        result, path = preflight() if arguments.stage == "preflight" else run()
    except PendingDependency as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps({"status": result["status"], "scientific_status": result.get("scientific_status"), "artifact": str(path)}, indent=2))
