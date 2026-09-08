#!/usr/bin/env python3
"""Independent numeric replay of the newly restored legacy geometry."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "reports/reruns/final_v14_additions_20260903"
LEGACY_HASHES = {
    "legacy_vocab_npz": "14496f3c629c4e8bcb809a3ab8bc4648d50a157c7c778e12150b0880333cac32",
    "legacy_vocab_json": "2e7f2bcfc41ef63af06eeb768daaa8417576c6f8740c52421670f93668ae2220",
    "legacy_human_results": "cfec16a15b5dbe9de367eaa5cda25bb97f5c8e25a26dd31b41f5cc5cb8b55d99",
    "legacy_human_receipt": "9b2571b24a541f9b3931ff05e0d08b1c2d1cea4b696f338bf36751234c8e93af",
}


def identity(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


def verify(item: dict) -> Path:
    path = Path(item["path"])
    observed = identity(path)
    assert all(observed[k] == item[k] for k in observed)
    return path


def sealed(path: Path) -> dict:
    seal = json.loads(Path(str(path) + ".seal.json").read_text())
    assert identity(path)["sha256"] == seal.get("sha256", seal.get("artifact", {}).get("sha256"))
    return json.loads(path.read_text())


def vectors(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        v = np.einsum("ij,jk->ik", z["centroids"].astype(np.float64), z["pca_components"].astype(np.float64))
        v += z["pca_mean"].astype(np.float64)
    assert np.isfinite(v).all()
    return v / np.sqrt(np.einsum("ij,ij->i", v, v))[:, None]


def audit(path: Path) -> dict:
    result = sealed(path)
    contract = sealed(verify(result["contract"]))
    dependency = sealed(verify(contract["dependency_receipt"]))
    naming = sealed(verify(dependency["naming_freeze"]))
    assert result["status"] == "CORRESPONDENCE_GEOMETRY_COMPLETE"
    assert result["name_gate_status"] == naming["name_gate_status"] == "NAME_GATE_FAIL"
    assert set(contract["legacy_inputs"]) == set(LEGACY_HASHES)
    for name, item in contract["legacy_inputs"].items():
        verify(item)
        assert item["sha256"] == LEGACY_HASHES[name]
    for item in contract["historical_reports_preserved"]:
        verify(item)
    frozen_inventory = json.loads((RUN / "e4v_pre_reader/receipts/vocabularies.json").read_text())
    frozen_variants = {r["artifact"]["path"]: r for r in frozen_inventory["artifacts"]["vocabularies"]}
    expected_grid = {(k, seed) for k in (24, 32, 40) for seed in (20260819, 20260820, 20260821)}
    assert len(result["variants"]) == len(contract["variants"]) == 9
    assert {(v["k"], v["seed"]) for v in result["variants"]} == expected_grid
    assert {(v["k"], v["seed"]) for v in contract["variants"]} == expected_grid
    assert sum(v["canonical"] for v in result["variants"]) == 1
    anchors = vectors(Path(contract["legacy_inputs"]["legacy_vocab_npz"]["path"]))[[17, 28]]
    maximum_error = 0.0
    rows = []
    for item in contract["variants"]:
        for key in ("vocabulary", "receipt", "pca_basis"):
            verify(item[key])
        original = frozen_variants[item["vocabulary"]["path"]]
        assert item["vocabulary"]["sha256"] == original["artifact"]["sha256"]
        assert item["receipt"]["sha256"] == original["checkpoint_receipt"]["sha256"]
        saved = next(r for r in result["variants"] if (r["k"], r["seed"]) == (item["k"], item["seed"]))
        cosine = anchors @ vectors(Path(item["vocabulary"]["path"])).T
        # Independent rectangular assignment solver, then vectorized lexical tie rule.
        a, b = linear_sum_assignment(cosine, maximize=True)
        optimum = cosine[a, b].sum()
        joint = cosine[0, :, None] + cosine[1, None, :]
        np.fill_diagonal(joint, -np.inf)
        assert abs(np.max(joint) - optimum) < 1e-12
        chosen = np.argwhere(joint >= optimum - 1e-12)[0]
        assert saved["canonical"] == (item["k"] == 32 and item["seed"] == 20260819)
        observed = (saved["p17"]["prototype_id"], saved["p28"]["prototype_id"])
        assert tuple(chosen) == observed and observed[0] != observed[1]
        for j, axis in enumerate(("p17", "p28")):
            error = abs(float(cosine[j, chosen[j]]) - saved[axis]["cosine"])
            maximum_error = max(maximum_error, error)
            assert error < 1e-12
            assert saved[axis]["geometry_pass"] == bool(cosine[j, chosen[j]] >= .8)
        assert abs(saved["joint_cosine_objective"] - joint[tuple(chosen)]) < 1e-12
        rows.append({"k": item["k"], "seed": item["seed"], "pair": list(observed)})
    canonical = next(r for r in result["variants"] if r["canonical"])
    for axis in ("p17", "p28"):
        summary = result["axes"][axis]
        assert summary["geometry_passes"] == sum(r[axis]["geometry_pass"] for r in result["variants"]) == 9
        assert summary["canonical_geometry"] == canonical[axis]
        assert summary["status"] == "CORRESPONDENCE_NOT_EVALUABLE"
        assert summary["controlling_category_compatibility"] is None
    return {"status": "PASS_INDEPENDENT_CORRESPONDENCE_REPLAY", "result": identity(path),
            "audit_code": identity(Path(__file__)), "four_legacy_pins_verified": True,
            "all_nine_variants_match_original_pre_reader_receipt": True,
            "independent_method": "einsum float64 backprojection; scipy rectangular maximum assignment; vectorized lexical tie check",
            "max_absolute_cosine_error": maximum_error, "pairs_verified": rows,
            "geometry_pass_counts": {axis: 9 for axis in ("p17", "p28")},
            "canonical": {axis: canonical[axis] for axis in ("p17", "p28")},
            "named_correspondence_status": "CORRESPONDENCE_NOT_EVALUABLE_NAME_GATE_FAIL",
            "historical_missing_input_reports_unchanged": True,
            "target_outcomes_opened": False, "model_refits_performed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    path = parser.parse_args().result.resolve()
    checked = audit(path)
    output = path.with_name("independent_audit.json")
    if output.exists():
        previous = sealed(output)
        assert {k: v for k, v in previous.items() if k != "created_utc"} == checked
    else:
        checked["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        output.write_text(json.dumps(checked, indent=2, sort_keys=True) + "\n")
        Path(str(output) + ".seal.json").write_text(json.dumps(identity(output), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": checked["status"], "receipt": str(output), "max_absolute_cosine_error": checked["max_absolute_cosine_error"]}, indent=2))


if __name__ == "__main__":
    main()
