"""Governed FINAL-v14 Module II source freeze and label-blind target scoring.

The freeze snapshots its complete scoring implementation. Target execution must
use that snapshot; later analysis/report code cannot change the target scorer.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1.v14_concepts import V14Vocabulary  # noqa: E402
from tools.final_v14_modeling_v2 import fit_logistic  # noqa: E402

RUN = REPO / "reports/reruns/final_v14_additions_20260903"
TARGETS = ("cptac_primary", "rih_primary", "orion_cpht", "rih_metastatic", "sr1482_metastatic")
CENSUS = {"cptac_primary": (98, 94), "rih_primary": (155, 153),
          "orion_cpht": (41, 40), "rih_metastatic": (85, 85), "sr1482_metastatic": (100, 74)}
COLS = [f"prototype_{j:02d}" for j in range(32)]
PACK_ROOT = Path("/mnt/wsl/oceanpath-hot/features/colon_stream/20x_256px_0px_overlap_mpp0.5/packed_uni_v1")
CODE_FILES = ("tools/final_v14_module2.py", "tools/final_v14_modeling_v2.py",
              "src/oceanpath/__init__.py", "src/oceanpath/aim1/__init__.py",
              "src/oceanpath/aim1/v14_concepts.py")
ALLOWED_ROSTER = {"slide_id", "patient_id", "cohort", "subcohort", "specimen_role", "role",
                  "liver_class", "mpp", "mpp_source", "patch_count", "exclude_neoadjuvant",
                  "exclude_ambiguous_crc15"}


class ContractError(RuntimeError):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": digest(path), "size_bytes": path.stat().st_size}


def verify_identity(item: dict) -> Path:
    p = Path(item["path"])
    if p.stat().st_size != item["size_bytes"] or digest(p) != item["sha256"]:
        raise ContractError(f"Sealed input drift: {p}")
    return p


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ContractError(f"Refusing to overwrite: {path}")
        return
    with path.open("xb") as f:
        f.write(payload)


def write_json(path: Path, data: Any) -> None:
    write_once(path, (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def seal(path: Path) -> None:
    write_json(Path(str(path) + ".seal.json"), identity(path))


def read_sealed(path: Path) -> dict:
    record = json.loads(Path(str(path) + ".seal.json").read_text())
    expected = record.get("sha256", record.get("artifact", {}).get("sha256"))
    if not expected or digest(path) != expected:
        raise ContractError(f"Invalid or missing seal: {path}")
    return json.loads(path.read_text())


def runtime() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "threadpoolctl")}}


def source_penalty(outer_folds: dict) -> float | None:
    if set(outer_folds) != {str(f) for f in range(5)}:
        return None
    values = []
    for fold in range(5):
        row = outer_folds[str(fold)]
        if row.get("status") != "ESTIMABLE":
            return None
        c = float(row["selected_penalty"])
        if c not in tuple(10.0**p for p in range(-4, 5)):
            raise ContractError("Selected C outside the preregistered grid")
        values.append(c)
    return float(sorted(values)[2])


def score_matrix(model: dict, X: np.ndarray) -> np.ndarray:
    if model.get("status") != "ESTIMABLE":
        raise ContractError("No score exists for unavailable refit")
    X = np.asarray(X, dtype=np.float64)
    result = ((X - np.asarray(model["scaler_mean"], dtype=np.float64)) /
              np.asarray(model["scaler_scale"], dtype=np.float64)) @ np.asarray(model["coef"], dtype=np.float64) + float(model["intercept"])
    if not np.isfinite(result).all():
        raise ContractError("Nonfinite frozen score")
    return result


def _verify_bound(receipt: dict, path: Path) -> None:
    """Find the exact pre-reader identity without opening any other data path."""
    matches = []
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("path") == str(path.resolve()) and "sha256" in value:
                matches.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(receipt)
    if not matches or any(digest(path) != r["sha256"] for r in matches):
        raise ContractError(f"Missing/drifted pre-reader binding: {path}")


def freeze_source(pre: Path, summary_path: Path, naming_path: Path, output: Path) -> dict:
    final = output / "source_scoring_bundle.json"
    if final.exists():
        return verify_bundle(output, require_snapshot=False)
    summary = read_sealed(summary_path)
    if summary.get("status") != "SOURCE_MODELS_SEALED":
        raise ContractError("Module-I source models are not sealed")
    for representation in summary["representations"].values():
        for fold in representation.get("outer_folds", {}).values():
            if fold.get("status") == "ESTIMABLE":
                checkpoint = verify_identity(fold["checkpoint"])
                read_sealed(checkpoint)
    verify_identity(summary["contract"])
    naming = read_sealed(naming_path)
    gate = naming.get("name_gate_status", naming.get("status"))
    if gate not in {"NAME_GATE_PASS", "NAME_GATE_FAIL"}:
        raise ContractError("Naming must be frozen before Module II")
    profiles_path = pre / "profiles/reference/patient_profiles.parquet"
    quantile_path = pre / "profiles/reference_distance_quantiles.csv"
    vocabulary_path = pre / "vocabularies/reference/variants/k32_seed20260819/vocabulary.npz"
    source_path = pre / "inputs/tcga_surgen_primary.csv"
    profile_receipt = json.loads((pre / "receipts/profiles.json").read_text())
    vocabulary_receipt = json.loads((pre / "receipts/vocabularies.json").read_text())
    preflight = json.loads((pre / "receipts/preflight.json").read_text())
    for p in (profiles_path, quantile_path):
        _verify_bound(profile_receipt, p)
    _verify_bound(vocabulary_receipt, vocabulary_path)
    _verify_bound(preflight, source_path)
    profiles = pd.read_parquet(profiles_path).sort_values("patient_id").reset_index(drop=True)
    # Source labels are an inherited input; no target data enters this stage.
    source = pd.read_csv(source_path, usecols=["patient_id", "target_label"])
    if source.groupby("patient_id").target_label.nunique().max() != 1:
        raise ContractError("Inconsistent source labels")
    source = source.drop_duplicates("patient_id").set_index("patient_id")
    if profiles.patient_id.duplicated().any() or set(profiles.patient_id) != set(source.index):
        raise ContractError("Source matrix/label roster drift")
    y = source.loc[profiles.patient_id, "target_label"].to_numpy(int)
    if len(profiles) != 1239 or int(y.sum()) != 501:
        raise ContractError("Frozen source population drift")
    X = profiles[COLS].to_numpy(np.float64)
    if not np.allclose(X.sum(axis=1), 1, atol=1e-12, rtol=0) or (X < 0).any():
        raise ContractError("Reference abundance matrix invalid")
    representations = {"ALL32": list(range(32))}
    if gate == "NAME_GATE_PASS":
        coordinates = naming.get("NAMED_REF", naming.get("named_ref"))
        if not coordinates or len(coordinates) < 16 or sorted(set(coordinates)) != coordinates:
            raise ContractError("Invalid NAMED_REF under name gate pass")
        representations["NAMED_REF"] = coordinates
    models, dry = {}, {}
    source_predictions = profiles[["patient_id"]].copy()
    from threadpoolctl import threadpool_limits
    for rep, coordinates in representations.items():
        m1rep = "NAMED_OOF" if rep == "NAMED_REF" else rep
        folds = summary["representations"].get(m1rep, {}).get("outer_folds", {})
        C = source_penalty(folds)
        if C is None:
            models[rep] = {"status": "NOT_ESTIMABLE", "reason": "not all five Module-I outer fits estimable", "coordinates": coordinates}
            continue
        with threadpool_limits(limits=1):
            fitted = fit_logistic(X[:, coordinates], y, C)
            expected = np.asarray(fitted.pop("predictions", []), dtype=np.float64)
            fitted["coordinates"] = coordinates
            models[rep] = fitted
            if fitted["status"] == "ESTIMABLE":
                actual = score_matrix(fitted, X[:, coordinates])
                error = float(np.max(np.abs(expected - actual)))
                if error > 1e-12:
                    raise ContractError("Source scoring dry run exceeds 1e-12")
                source_predictions[f"logit_{rep}"] = actual
                dry[rep] = {"status": "PASS", "patients": len(X), "max_absolute_logit_error": error, "tolerance": 1e-12}
    # Expected target identities come from already sealed metadata; no target path is opened.
    parent_manifest_path = REPO / "reports/final_v13/source_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    rosters = {}
    for cohort in TARGETS:
        role = f"label_blind_{cohort}_scoring_roster"
        found = [r for r in parent_manifest["artifacts"] if r.get("role") == role]
        if len(found) != 1:
            raise ContractError(f"Missing unique inherited roster identity: {role}")
        rosters[cohort] = {k: found[0][k] for k in ("path", "sha256", "size_bytes")}
    frozen_code = output / "frozen_code"
    code = []
    for relative in CODE_FILES + ("uv.lock", "pyproject.toml"):
        dest = frozen_code / relative
        write_once(dest, (REPO / relative).read_bytes())
        code.append(identity(dest))
    source_logits_path = output / "source_dry_run_logits.csv"
    write_once(source_logits_path, source_predictions.to_csv(index=False, float_format="%.17g").encode())
    model_path = output / "source_models.json"
    write_json(model_path, models)
    seal(model_path)
    request_path = output / "source_replay_request.json"
    write_json(request_path, {"models": identity(model_path), "source_profiles": identity(profiles_path),
                              "expected_logits": identity(source_logits_path)})
    seal(request_path)
    subprocess.run([sys.executable, str(frozen_code / "tools/final_v14_module2.py"),
                    "source-dry-run", "--output", str(output.resolve())], check=True,
                   capture_output=True, text=True)
    replay_path = output / "source_snapshot_replay.json"
    replay = read_sealed(replay_path)
    dry_path = output / "source_dry_run_receipt.json"
    write_json(dry_path, {"status": "SOURCE_DRY_RUN_SEALED", "created_utc": now(), "representations": dry,
                          "logits": identity(source_logits_path), "serialized_snapshot_replay": identity(replay_path),
                          "snapshot_replay_status": replay["status"], "target_feature_paths_read": 0})
    seal(dry_path)
    bundle = {"schema_version": 1, "status": "SOURCE_SCORING_BUNDLE_SEALED", "created_utc": now(),
              "name_gate_status": gate, "models": models,
              "source_inputs": [identity(p) for p in (profiles_path, source_path, quantile_path, vocabulary_path,
                                                        summary_path, naming_path, parent_manifest_path)],
              "reference_vocabulary": identity(vocabulary_path), "source_quantiles": identity(quantile_path),
              "source_profiles": identity(profiles_path), "source_labels": identity(source_path),
              "dry_run": identity(dry_path), "code": code,
              "code_tree_sha256": hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(), "runtime": runtime(),
              "target_rosters": rosters, "pack_root": str(PACK_ROOT), "pack_preflight": preflight["pack"],
              "patient_aggregation": "Arithmetic mean of slide tile-assignment fractions; equal slide weights",
              "precision": {"embedding_projection_and_distance": "float32", "abundance_and_logit": "float64"},
              "input_schema": {"roster_allowed_columns": sorted(ALLOWED_ROSTER), "required": ["slide_id", "patient_id"], "coordinates": COLS},
              "output_schema": {"profiles": ["cohort", "patient_id", "n_slides", *COLS], "scores": ["cohort", "patient_id", *[f"logit_{r}" for r in models if models[r]["status"] == "ESTIMABLE"]],
                                "ood": ["cohort", "patient_id", "prototype_id", "assigned_tiles", "nonempty_slides", "fraction_beyond_source_q99", "median_distance"]},
              "target_entry_command": [sys.executable, str(frozen_code / "tools/final_v14_module2.py"), "score-targets", "--output", str(output.resolve())],
              "claim_scope": "Locked source-frozen secondary analysis of previously opened archived cohorts"}
    write_json(final, bundle)
    seal(final)
    return bundle


def verify_bundle(output: Path, *, require_snapshot: bool = True) -> dict:
    bundle = read_sealed(output / "source_scoring_bundle.json")
    for item in bundle["code"] + bundle["source_inputs"] + [bundle["dry_run"]]:
        verify_identity(item)
    dry = read_sealed(Path(bundle["dry_run"]["path"]))
    verify_identity(dry["logits"])
    replay = read_sealed(verify_identity(dry["serialized_snapshot_replay"]))
    for item in replay["inputs"]:
        verify_identity(item)
    if bundle["runtime"] != runtime():
        raise ContractError("Frozen scoring runtime changed")
    if require_snapshot and Path(__file__).resolve() != Path(bundle["target_entry_command"][1]).resolve():
        raise ContractError("Run the exact target_entry_command from the sealed source bundle")
    return bundle


def source_snapshot_replay(output: Path) -> dict:
    request = read_sealed(output / "source_replay_request.json")
    for item in request.values():
        verify_identity(item)
    models = read_sealed(Path(request["models"]["path"]))
    profiles = pd.read_parquet(request["source_profiles"]["path"]).sort_values("patient_id")
    expected = pd.read_csv(request["expected_logits"]["path"], float_precision="round_trip").set_index("patient_id")
    results = {}
    from threadpoolctl import threadpool_limits
    for rep, model in models.items():
        if model["status"] != "ESTIMABLE":
            continue
        X = profiles[[COLS[j] for j in model["coordinates"]]].to_numpy(np.float64)
        with threadpool_limits(limits=1):
            actual = score_matrix(model, X)
        difference = float(np.max(np.abs(actual - expected.loc[profiles.patient_id, f"logit_{rep}"].to_numpy(float))))
        if difference > 1e-12:
            raise ContractError("Serialized snapshot fails source replay tolerance")
        results[rep] = {"max_absolute_error": difference, "tolerance": 1e-12}
    result = {"status": "SERIALIZED_SOURCE_SNAPSHOT_REPLAY_PASS", "inputs": list(request.values()),
              "representations": results, "target_feature_paths_read": 0}
    path = output / "source_snapshot_replay.json"
    write_json(path, result)
    seal(path)
    return result


def slide_diagnostics(labels: np.ndarray, distances: np.ndarray, q99: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    labels = np.asarray(labels, dtype=int)
    distances = np.asarray(distances, dtype=float)
    if not len(labels) or labels.shape != distances.shape or not np.isfinite(distances).all():
        raise ContractError("Empty or invalid slide assignments")
    counts = np.bincount(labels, minlength=len(q99))
    if len(counts) != len(q99):
        raise ContractError("Assignment outside frozen coordinate set")
    rows = []
    for j, count in enumerate(counts):
        d = distances[labels == j]
        rows.append({"prototype_id": j, "assigned_tiles": int(count), "nonempty_slides": int(count > 0),
                     "fraction_beyond_source_q99": float(np.mean(d > q99[j])) if count else np.nan,
                     "median_distance": float(np.median(d)) if count else np.nan})
    return counts.astype(np.float64) / len(labels), rows


def aggregate_slides(slides: pd.DataFrame, diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["cohort", "patient_id"]
    profiles = slides.groupby(keys, sort=True)[COLS].mean().reset_index()
    counts = slides.groupby(keys, sort=True).size().rename("n_slides").reset_index()
    profiles = counts.merge(profiles, on=keys, validate="one_to_one")
    ood = diagnostics.groupby([*keys, "prototype_id"], sort=True).agg(
        assigned_tiles=("assigned_tiles", "sum"), nonempty_slides=("nonempty_slides", "sum"),
        fraction_beyond_source_q99=("fraction_beyond_source_q99", "mean"), median_distance=("median_distance", "mean")).reset_index()
    return profiles, ood


def score_targets(output: Path) -> dict:
    bundle = verify_bundle(output)
    final = output / "target_score_seal.json"
    if final.exists():
        result = read_sealed(final)
        for item in result["artifacts"]:
            verify_identity(item)
        return result
    pack = Path(bundle["pack_root"])
    # Nothing touching the packed store precedes the bundle/dry-run verification.
    if not (pack / "features.bin").is_file():
        raise FileNotFoundError(f"Target packed feature storage unavailable: {pack}")
    meta = json.loads((pack / "meta.json").read_text())
    index = pd.read_parquet(pack / "index.parquet").set_index("slide_id")
    for item in _pack_identities(bundle["pack_preflight"]):
        p = Path(item["path"])
        # The sealed preflight content digest is preserved; cheap mtime/size checks
        # protect the immutable 50-GB pack without rereading source/target tiles.
        st = p.stat()
        if st.st_size != item["size_bytes"] or st.st_mtime_ns != item["mtime_ns"]:
            raise ContractError(f"Packed-store identity changed: {p}")
    features = np.memmap(pack / "features.bin", mode="r", dtype=np.dtype(meta["feat_dtype"]),
                         shape=(int(meta["total_patches"]), int(meta["feat_dim"])))
    with np.load(verify_identity(bundle["reference_vocabulary"]), allow_pickle=False) as z:
        vocab = V14Vocabulary(z["centroids"], z["pca_mean"], z["pca_components"])
    quantiles = pd.read_csv(verify_identity(bundle["source_quantiles"])).sort_values("prototype_id")
    q99 = quantiles.q99_distance.to_numpy(float)
    if len(q99) != 32 or not np.isfinite(q99).all():
        raise ContractError("Source all-tile q99 census unavailable")
    slides, cells, roster_relations = [], [], {}
    from threadpoolctl import threadpool_limits
    for cohort in TARGETS:
        roster = pd.read_csv(verify_identity(bundle["target_rosters"][cohort]), dtype={"slide_id": str, "patient_id": str})
        if set(roster) - ALLOWED_ROSTER or not {"slide_id", "patient_id"} <= set(roster):
            raise ContractError(f"Outcome-bearing or invalid target roster: {cohort}")
        if roster.slide_id.duplicated().any() or (len(roster), roster.patient_id.nunique()) != CENSUS[cohort]:
            raise ContractError(f"Frozen target roster census drift: {cohort}")
        roster_relations[cohort] = set(roster.patient_id)
        for row in roster.sort_values(["patient_id", "slide_id"]).itertuples(index=False):
            shard = output / "target_shards" / cohort / f"{hashlib.sha256(row.slide_id.encode()).hexdigest()}.json"
            if shard.exists():
                data = read_sealed(shard)
                if data["bundle_sha256"] != digest(output / "source_scoring_bundle.json"):
                    raise ContractError("Target shard source-bundle drift")
            else:
                ix = index.loc[row.slide_id]
                start, count = int(ix["offset"]), int(ix["n_patches"])
                with threadpool_limits(limits=1):
                    labels, distances = vocab.assign_with_distances(np.asarray(features[start:start + count]))
                abundance, diag = slide_diagnostics(labels, distances, q99)
                # JSON null encodes the preregistered undefined no-support cell.
                for item in diag:
                    for k in ("fraction_beyond_source_q99", "median_distance"):
                        if not np.isfinite(item[k]):
                            item[k] = None
                data = {"bundle_sha256": digest(output / "source_scoring_bundle.json"), "cohort": cohort,
                        "patient_id": row.patient_id, "slide_id": row.slide_id,
                        "abundance": abundance.tolist(), "diagnostics": diag}
                write_json(shard, data)
                seal(shard)
            slides.append({"cohort": cohort, "patient_id": row.patient_id, "slide_id": row.slide_id,
                           **dict(zip(COLS, data["abundance"], strict=True))})
            cells.extend({"cohort": cohort, "patient_id": row.patient_id, **d} for d in data["diagnostics"])
        print(json.dumps({"cohort": cohort, "status": "LABEL_BLIND_ASSIGNMENTS_COMPLETE"}), flush=True)
    if len(roster_relations["rih_primary"] & roster_relations["rih_metastatic"]) != 8:
        raise ContractError("RIH dual-role roster drift")
    source_ids = set(pd.read_parquet(verify_identity(bundle["source_profiles"]), columns=["patient_id"]).patient_id)
    if any(source_ids & ids for ids in roster_relations.values()):
        raise ContractError("Target/source patient overlap")
    profiles, ood = aggregate_slides(pd.DataFrame(slides), pd.DataFrame(cells))
    scores = profiles[["cohort", "patient_id"]].copy()
    for rep, model in bundle["models"].items():
        if model["status"] == "ESTIMABLE":
            with threadpool_limits(limits=1):
                scores[f"logit_{rep}"] = score_matrix(model, profiles[[COLS[j] for j in model["coordinates"]]].to_numpy(float))
    artifacts = []
    for name, frame in (("target_profiles", profiles), ("target_logits", scores), ("target_ood", ood)):
        path = output / f"{name}.csv"
        write_once(path, frame.to_csv(index=False, float_format="%.17g").encode())
        artifacts.append(identity(path))
    summary = ood.groupby(["cohort", "prototype_id"], sort=True).agg(
        patients=("patient_id", "size"), supported_patients=("nonempty_slides", lambda x: int((x > 0).sum())),
        mean_fraction_beyond_source_q99=("fraction_beyond_source_q99", "mean"), mean_median_distance=("median_distance", "mean")).reset_index()
    p = output / "target_ood_before_outcome_join.csv"
    write_once(p, summary.to_csv(index=False, float_format="%.17g").encode())
    artifacts.append(identity(p))
    result = {"status": "TARGET_PROFILES_AND_LOGITS_SEALED_BEFORE_OUTCOME_JOIN", "created_utc": now(),
              "source_bundle": identity(output / "source_scoring_bundle.json"), "artifacts": artifacts,
              "outcome_columns_read": 0, "patients_by_cohort": {c: len(ids) for c, ids in roster_relations.items()}}
    write_json(final, result)
    seal(final)
    return result


def _pack_identities(value: Any) -> list[dict]:
    if isinstance(value, dict):
        if {"path", "sha256", "size_bytes", "mtime_ns"} <= set(value):
            return [value]
        return [r for v in value.values() for r in _pack_identities(v)]
    if isinstance(value, list):
        return [r for v in value for r in _pack_identities(v)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "verify", "source-dry-run", "score-targets"])
    parser.add_argument("--output", type=Path, default=RUN / "e4m2_source_frozen")
    parser.add_argument("--pre-reader", type=Path, default=RUN / "e4v_pre_reader")
    parser.add_argument("--module1-summary", type=Path, default=RUN / "e4m1_source/source_fit_summary.json")
    parser.add_argument("--naming", type=Path, default=RUN / "e4v_post_reader_xlsx/naming/naming_freeze.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if not any(p.name.startswith("final_v14_additions_") and p.parent.name == "reruns" and p.parent.parent.name == "reports" for p in output.parents):
        raise ContractError("All Module-II outputs must be below a governed final_v14 rerun")
    if args.stage == "freeze":
        result = freeze_source(args.pre_reader, args.module1_summary, args.naming, output)
    elif args.stage == "source-dry-run":
        result = source_snapshot_replay(output)
    elif args.stage == "verify":
        result = verify_bundle(output, require_snapshot=False)
    else:
        result = score_targets(output)
    print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
