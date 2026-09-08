"""Read-only independent numerical audit of the sealed nine-variant mapping."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports/reruns/final_v14_additions_20260903"
OUT = RUN / "e4v_variant_mapping"


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(item: dict) -> Path:
    path = Path(item["path"])
    actual = identity(path)
    assert all(actual[key] == item[key] for key in ["sha256", "size_bytes"]), path
    return path


def sealed(path: Path) -> dict:
    seal = json.loads(Path(str(path)+".seal.json").read_text())
    verify(seal.get("artifact", seal))
    return json.loads(path.read_text())


def embedded_identities(value):
    if isinstance(value, dict):
        if {"path", "size_bytes", "sha256"} <= set(value):
            yield value
        for child in value.values():
            yield from embedded_identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from embedded_identities(child)


def inverse_centers(vocab) -> np.ndarray:
    x = vocab["centroids"].astype(float) @ vocab["pca_components"].astype(float) + vocab["pca_mean"].astype(float)
    return x / np.sqrt(np.einsum("ij,ij->i", x, x))[:, None]


def main() -> None:
    result = sealed(OUT / "results.json")
    contract = sealed(verify(result["contract"]))
    assert result["status"] == "CANONICAL_NINE_VARIANT_MAPPING_COMPLETE"
    assert result["canonical_rows"] == 288 and result["models_or_names_changed"] is False
    assert contract["created_utc"] < result["created_utc"]
    for item in result["artifacts"] + contract["implementation"] + [contract["preregistration"]]:
        verify(item)
    top = json.loads(verify(contract["vocabulary_receipt"]).read_text())
    pins = {(item["path"], item["sha256"]) for item in embedded_identities(top)}
    frame = pd.read_csv(OUT / "canonical_to_variant.csv", float_precision="round_trip")
    summaries = pd.read_csv(OUT / "variant_summary.csv", float_precision="round_trip").set_index(["k", "seed"])
    assert len(frame) == 288 and not frame.duplicated(["k", "seed", "canonical_prototype_id"]).any()
    variants = contract["variants"]
    assert {(item["k"], item["seed"]) for item in variants} == {(k, seed) for k in [24,32,40] for seed in range(20260819,20260822)}
    canonical = np.load(next(item["vocabulary"]["path"] for item in variants if item["k"] == 32 and item["seed"] == 20260819))
    c = inverse_centers(canonical)
    checks, hashed = [], set()
    for item in variants:
        for key in ["checkpoint", "vocabulary"]:
            assert (item[key]["path"], item[key]["sha256"]) in pins
            verify(item[key])
        checkpoint = json.loads(Path(item["checkpoint"]["path"]).read_text())
        assert checkpoint["status"] == "PASS"
        assert checkpoint["metadata"]["k"] == item["k"] and checkpoint["metadata"]["kmeans_seed"] == item["seed"]
        for key in ["pca_basis", "projected_coordinates"]:
            pin = checkpoint["artifacts"][key]
            assert pin["sha256"] == contract["shared_sample_and_pca"][key]
            if pin["sha256"] not in hashed:
                verify(pin)
                hashed.add(pin["sha256"])
        for key, name in [("sample_ids_sha256", "sample_ids.parquet"), ("sample_plan_sha256", "sample_plan.parquet")]:
            assert checkpoint["dependencies"][key] == contract["shared_sample_and_pca"][key]
            assert identity(RUN / "e4v_pre_reader/vocabularies/reference" / name)["sha256"] == checkpoint["dependencies"][key]
        other = np.load(item["vocabulary"]["path"])
        assert np.array_equal(canonical["pca_mean"], other["pca_mean"])
        assert np.array_equal(canonical["pca_components"], other["pca_components"])
        similarities = c @ inverse_centers(other).T
        nearest = similarities.argmax(axis=1)
        if item["k"] == 32:
            rows, cols = linear_sum_assignment(similarities, maximize=True)
            pairs = dict(zip(rows, cols, strict=True))
            method = "HUNGARIAN_MAXIMUM_COSINE"
        else:
            reverse = similarities.argmax(axis=0)
            pairs = {i: int(j) for i,j in enumerate(nearest) if reverse[j] == i}
            method = "MUTUAL_NEAREST_COSINE"
        block = frame[(frame.k == item["k"]) & (frame.seed == item["seed"])].sort_values("canonical_prototype_id")
        assert block.canonical_prototype_id.tolist() == list(range(32))
        assert block.method.eq(method).all()
        assert np.array_equal(nearest, block.nearest_variant_prototype_id)
        errors = []
        for row in block.itertuples(index=False):
            i = row.canonical_prototype_id
            expected = pairs.get(i)
            if expected is None:
                assert pd.isna(row.variant_prototype_id) and pd.isna(row.cosine_similarity)
                assert not row.geometrically_mappable
            else:
                assert row.variant_prototype_id == expected
                errors.append(abs(row.cosine_similarity - similarities[i, expected]))
                assert row.geometrically_mappable == (similarities[i, expected] >= .8)
            assert abs(row.nearest_cosine_similarity - similarities[i, nearest[i]]) < 1e-6
        assert max(errors) < 1e-6
        summary = summaries.loc[(item["k"], item["seed"])]
        assert summary.matched_pairs == len(pairs)
        assert summary.pairs_cosine_ge_0_80 == int(block.geometrically_mappable.sum())
        assert summary.unmatched_canonical == 32-len(pairs) and summary.unmatched_variant == item["k"]-len(pairs)
        assert summary.matched_cosine_min == block.cosine_similarity.min()
        assert summary.matched_cosine_median == block.cosine_similarity.median()
        checks.append({"k": item["k"], "seed": item["seed"], "status": "PASS", "method": method,
                       "pairs": len(pairs), "pairs_cosine_ge_0_80": int(block.geometrically_mappable.sum()),
                       "max_float32_vs_float64_cosine_error": max(errors)})
    destination = OUT / "independent_audit.json"
    assert not destination.exists()
    audit = {"status": "INDEPENDENT_NINE_VARIANT_MAPPING_AUDIT_PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
             "implementation": identity(Path(__file__)), "results": identity(OUT / "results.json"),
             "contract": identity(OUT / "contract.json"), "canonical_rows_checked": 288, "variants": checks,
             "numerical_method": "Independent float64 inverse-PCA normalization and cosine; scipy maximize assignment for equal k, explicit reciprocal argmax for unequal k",
             "maximum_cosine_absolute_tolerance": 1e-6, "refits_performed": 0,
             "review": "All nine PCA/sample/projected-coordinate pins and all mapping/census rows verified; geometry is descriptive and cannot change models or names."}
    destination.write_text(json.dumps(audit, indent=2, sort_keys=True)+"\n")
    Path(str(destination)+".seal.json").write_text(json.dumps(identity(destination), indent=2)+"\n")
    print(json.dumps({"status": audit["status"], "output": str(destination), "variants": checks}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
