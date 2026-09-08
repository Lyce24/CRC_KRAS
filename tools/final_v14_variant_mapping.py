"""Prespecified canonical-to-nine-variant geometry; no refit or model selection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from oceanpath.aim1 import v14_concepts as geometry  # noqa: E402
from tools import final_v14_module2 as io  # noqa: E402

PRE = io.RUN / "e4v_pre_reader"
OUT = io.RUN / "e4v_variant_mapping"


def match_matrix(similarities: np.ndarray) -> list[dict]:
    matrix = np.asarray(similarities, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) == 0 or not np.isfinite(matrix).all():
        raise ValueError("A nonempty finite canonical-by-variant cosine matrix is required")
    nearest = np.argmax(matrix, axis=1)
    if matrix.shape[0] == matrix.shape[1]:
        rows, columns = linear_sum_assignment(-matrix)
        pairs = dict(zip(rows.tolist(), columns.tolist(), strict=True))
        method = "HUNGARIAN_MAXIMUM_COSINE"
    else:
        reverse = np.argmax(matrix, axis=0)
        pairs = {i: int(j) for i, j in enumerate(nearest) if reverse[j] == i}
        method = "MUTUAL_NEAREST_COSINE"
    return [{"canonical_prototype_id": i, "variant_prototype_id": pairs.get(i),
             "cosine_similarity": float(matrix[i, pairs[i]]) if i in pairs else None,
             "geometrically_mappable": bool(i in pairs and matrix[i, pairs[i]] >= .8),
             "nearest_variant_prototype_id": int(nearest[i]),
             "nearest_cosine_similarity": float(matrix[i, nearest[i]]), "method": method}
            for i in range(len(matrix))]


def vocabulary(path: Path) -> geometry.V14Vocabulary:
    with np.load(path, allow_pickle=False) as a:
        return geometry.V14Vocabulary(a["centroids"], a["pca_mean"], a["pca_components"])


def run() -> dict:
    final = OUT / "results.json"
    if final.exists():
        result = io.read_sealed(final)
        for item in result["artifacts"] + [result["contract"]]:
            io.verify_identity(item)
        return result
    receipt_path = PRE / "receipts/vocabularies.json"
    top = json.loads(receipt_path.read_text())
    shared, variants, checked = None, [], set()
    for k in (24, 32, 40):
        for seed in (20260819, 20260820, 20260821):
            path = PRE / f"vocabularies/reference/variants/k{k}_seed{seed}/vocabulary.npz"
            checkpoint = Path(str(path) + ".receipt.json")
            io._verify_bound(top, path)
            io._verify_bound(top, checkpoint)
            record = json.loads(checkpoint.read_text())
            if record["status"] != "PASS" or record["metadata"]["k"] != k or record["metadata"]["kmeans_seed"] != seed:
                raise ValueError("Variant checkpoint identity/census mismatch")
            fields = {key: record["artifacts"][key]["sha256"] for key in ("pca_basis", "projected_coordinates")}
            fields.update({key: record["dependencies"][key] for key in ("sample_ids_sha256", "sample_plan_sha256")})
            if shared is not None and fields != shared:
                raise ValueError("Variants must use byte-identical source sample, PCA and projected coordinates")
            shared = fields
            for item in record["artifacts"].values():
                if item["path"] not in checked:
                    io.verify_identity(item)
                    checked.add(item["path"])
            variants.append({"k": k, "seed": seed, "vocabulary": io.identity(path), "checkpoint": io.identity(checkpoint)})
    for key, path in (("sample_ids_sha256", PRE / "vocabularies/reference/sample_ids.parquet"),
                      ("sample_plan_sha256", PRE / "vocabularies/reference/sample_plan.parquet")):
        if io.digest(path) != shared[key]:
            raise ValueError("Shared source sample identity mismatch")
    contract = {"status": "CANONICAL_VARIANT_GEOMETRY_CONTRACT_SEALED", "created_utc": io.now(),
                "preregistration": io.identity(REPO / "reports/final_v14_PREREGISTRATION.md"),
                "vocabulary_receipt": io.identity(receipt_path), "variants": variants,
                "shared_sample_and_pca": shared, "canonical": {"k": 32, "seed": 20260819},
                "implementation": [io.identity(Path(__file__)), io.identity(Path(geometry.__file__)),
                                   io.identity(REPO / "tests/test_final_v14_variant_mapping.py")],
                "runtime": io.runtime(), "mapping": "Same frozen float32 inverse-PCA/renormalization/cosine helper as outer mapping; Hungarian for equal k, mutual-nearest for unequal k; argmax ties choose lowest zero-based ID; cosine>=0.80.",
                "quarantine": "Descriptive §3.1 sensitivity; never changes any coordinate, name gate, penalty, coefficient, prediction or primary result. Computed after sealed main analyses without their data as inputs."}
    contract_path = OUT / "contract.json"
    if contract_path.exists():
        previous = io.read_sealed(contract_path)
        contract["created_utc"] = previous["created_utc"]
    io.write_json(contract_path, contract)
    io.seal(contract_path)
    canonical_path = PRE / "vocabularies/reference/variants/k32_seed20260819/vocabulary.npz"
    canonical = vocabulary(canonical_path)
    rows, summaries = [], []
    with threadpool_limits(limits=1):
        for item in variants:
            other = vocabulary(Path(item["vocabulary"]["path"]))
            if not np.array_equal(canonical.pca_mean, other.pca_mean) or not np.array_equal(canonical.pca_components, other.pca_components):
                raise ValueError("Variant vocabulary embeds a different PCA basis")
            matrix = geometry.cosine_similarity_matrix(canonical.backprojected_centroids(), other.backprojected_centroids())
            block = [{"k": item["k"], "seed": item["seed"], **r} for r in match_matrix(matrix)]
            rows.extend(block)
            matched = [r for r in block if r["variant_prototype_id"] is not None]
            summaries.append({"k": item["k"], "seed": item["seed"], "method": block[0]["method"],
                              "canonical_prototypes": 32, "matched_pairs": len(matched),
                              "pairs_cosine_ge_0_80": sum(r["geometrically_mappable"] for r in block),
                              "unmatched_canonical": 32-len(matched), "unmatched_variant": item["k"]-len(matched),
                              "matched_cosine_min": min(r["cosine_similarity"] for r in matched),
                              "matched_cosine_median": float(np.median([r["cosine_similarity"] for r in matched]))})
    table = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    artifacts = []
    for name, frame in (("canonical_to_variant.csv", table), ("variant_summary.csv", summary)):
        path = OUT / name
        io.write_once(path, frame.to_csv(index=False, float_format="%.17g").encode())
        artifacts.append(io.identity(path))
    result = {"status": "CANONICAL_NINE_VARIANT_MAPPING_COMPLETE", "created_utc": io.now(),
              "contract": io.identity(contract_path), "artifacts": artifacts, "variants": summaries,
              "canonical_rows": len(rows), "models_or_names_changed": False,
              "claim_scope": "Descriptive geometric dictionary stability; no named morphology inference or alternative-model selection. Float32 cosine roundoff may slightly exceed 1."}
    io.write_json(final, result)
    io.seal(final)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "variants": result["variants"]}, indent=2))
