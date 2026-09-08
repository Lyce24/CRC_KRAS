#!/usr/bin/env python3
"""Append-only pinned-geometry replay accepting the governed legacy archive links.

The initial intake helper rejected symlinks before testing legacy file digests.
The preregistration requires the four byte identities, not a regular-file path
topology. This replay preserves the first report and checks resolved files at
the exact original SHA-256 pins. It cannot change the frozen naming gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import final_v14_post_reader as post  # noqa: E402


def resolved_pin(path: Path, expected: str) -> dict:
    record = post.identity(path.resolve(strict=True))
    if record["sha256"] != expected:
        raise post.PostReaderError(f"Legacy pinned digest mismatch: {path}")
    return record


def run() -> dict:
    names = post.naming()
    root = post.POST / "correspondence"
    destination = root / "replayed_results.json"
    if destination.exists():
        result = post.verify_seal(destination)
        for record in result["source_identities"].values():
            post.check_identity(record)
        return result
    correction_path = root / "resolved_legacy_path_amendment.json"
    correction = {
        "status": "PRE_GEOMETRY_IMPLEMENTATION_CORRECTION", "created_utc": post.now(),
        "prior_report": post.identity(root / "results.json"),
        "reason": "The initial local regular-file guard rejected governed legacy archive symlinks before comparing the preregistered digests. The prior artifact is preserved; this replay follows each existing path to its read-only archive file and enforces the same four original SHA-256 pins.",
        "method_changes": "None: names, geometry, thresholds, variants, and matching rule are unchanged.",
        "name_gate_status": names["name_gate_status"], "runner": post.identity(Path(__file__)),
    }
    post.publish(correction_path, correction)
    records = {"naming_freeze": post.identity(post.POST / "naming/naming_freeze.json"),
               "implementation_correction": post.identity(correction_path)}
    requested_paths = {}
    for name, (path, digest) in post.pre.LEGACY_INPUTS.items():
        records[name] = resolved_pin(path, digest)
        requested_paths[name] = str(path)
    with np.load(records["legacy_vocab_npz"]["path"], allow_pickle=False) as bundle:
        anchors = np.asarray(bundle["pca_mean"], dtype=float) + np.asarray(bundle["centroids"][[17, 28]], dtype=float) @ np.asarray(bundle["pca_components"], dtype=float)
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    variants = []
    for k in post.pre.VARIANT_K:
        for seed in post.pre.VARIANT_SEEDS:
            path = post.pre._vocabulary_path(post.PRE, fold=None, k=k, seed=seed)
            receipt = post.read_json(path.with_name(path.name + ".receipt.json"))
            post.pre._validate_artifact_tree(receipt, f"k{k}/seed{seed}")
            vocabulary = post.pre._load_vocabulary(path)
            vectors = np.asarray(vocabulary.pca_mean, dtype=float) + np.asarray(vocabulary.centroids, dtype=float) @ np.asarray(vocabulary.pca_components, dtype=float)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            cosines = anchors @ vectors.T
            pair = post.joint_anchor_assignment(cosines)
            records[f"k{k}_seed{seed}"] = post.identity(path)
            variants.append({"k": k, "seed": seed, "canonical": k == 32 and seed == post.pre.VOCAB_SEED,
                             **{axis: {"prototype_id": pair[i], "cosine": float(cosines[i, pair[i]]),
                                       "geometry_pass": bool(cosines[i, pair[i]] >= 0.8)}
                                for i, axis in enumerate(("p17", "p28"))}})
    axes = {}
    for axis in ("p17", "p28"):
        axes[axis] = {"status": "CORRESPONDENCE_NOT_EVALUABLE", "reason": names["name_gate_status"],
                      "geometry_passes": sum(v[axis]["geometry_pass"] for v in variants), "variant_denominator": 9,
                      "canonical_geometry": next(v[axis] for v in variants if v["canonical"]),
                      "controlling_category_compatibility": None}
    if names["name_gate_status"] != "NAME_GATE_FAIL":
        raise post.PostReaderError("This authorized replay is scoped to the frozen failed name gate")
    result = {"schema_version": 1, "component": "final_v14_legacy_geometry_replay", "status": "GEOMETRY_COMPLETE_NAME_DEPENDENT_CORRESPONDENCE_NOT_EVALUABLE",
              "created_utc": post.now(), "source_identities": records, "requested_legacy_paths": requested_paths,
              "variants": variants, "axes": axes, "legacy_input_pins": "ALL_FOUR_PASS",
              "supersedes": "results.json metadata-path guard failures only; original preserved",
              "claim_limit": "Direct geometric correspondence is descriptive; no named correspondence or spotlight under NAME_GATE_FAIL."}
    post.publish(destination, result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "axes": result["axes"]}, indent=2))
