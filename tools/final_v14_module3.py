#!/usr/bin/env python3
"""Append-only FINAL-v14 Module III source fits and separately invoked inference.

Production commands require a sealed, completed naming decision. ``prepare``
authenticates inherited source task rosters and records the analysis program;
``fit-source`` checkpoints each outer fit; ``evaluate`` alone joins the frozen
FINAL-v13 source predictions. No command reads target cohorts or fits MIL.
Long-running execution belongs in a coordinator-managed tmux session.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
FINE_TASKS = ("codon", "g12d_broad", "allele1", "allele2", "g12c")
WT_SEEDS = (20260823, 20260824, 20260825)
FOLDS = tuple(range(5))
PROTOTYPES = [f"prototype_{i:02d}" for i in range(32)]
PREREG = REPO / "reports/final_v14_PREREGISTRATION.md"
PREREG_SHA256 = "677aad233a2e5f7c1d2975b5436269ed5060b19596904930ae49ae8c1cff3908"
V13_MANIFEST = REPO / "reports/final_v13/source_manifest.json"
V13_MANIFEST_SHA256 = "71b118c57e4f02ce80246a0aa1444d919ac9de7db6a1316ced22122682e464d0"
V13_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim3_tcga_surgen_primary_univ1_5seed_v2_20260827")
V13_LOGITS_SHA256 = "b124eaede8934cb814a276225e7e94c16deb019ec66d0d2e3321ffa7db21ce4f"
DEFAULT_PRE_READER = REPO / "reports/reruns/final_v14_additions_20260903/e4v_pre_reader"


class ContractError(RuntimeError):
    """An authenticated input or the preregistered analysis contract drifted."""


def identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def require_identity(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    actual = identity(path)
    if actual["sha256"] != record.get("sha256") or (
        "size_bytes" in record and actual["size_bytes"] != record["size_bytes"]
    ):
        raise ContractError(f"Identity mismatch: {path}")
    return actual


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"Expected JSON object: {path}")
    return value


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_once(path: Path, payload: bytes) -> None:
    """Atomically publish a file, accepting only byte-identical replay."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ContractError(f"Refusing to replace immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ContractError(f"Concurrent artifact conflict: {path}") from None
    finally:
        os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    write_once(path, json_bytes(value))


def seal_file(path: Path) -> None:
    write_json(path.with_suffix(path.suffix + ".seal.json"), {"status": "SEALED", "artifact": identity(path)})


def verify_seal(path: Path) -> None:
    seal = read_json(path.with_suffix(path.suffix + ".seal.json"))
    if seal.get("status") != "SEALED":
        raise ContractError(f"Missing completed seal: {path}")
    require_identity(path, seal["artifact"])


def write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    write_once(path, buffer.getvalue())


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    write_once(path, buffer.getvalue())


@contextlib.contextmanager
def exclusive_lock(root: Path, stage: str) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f".{stage}.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ContractError(f"Another {stage} process owns {root}") from None
        yield


def validate_output_root(root: Path) -> Path:
    root = root.resolve()
    allowed = (REPO / "reports/reruns").resolve()
    if not root.is_relative_to(allowed) or root == allowed:
        raise ContractError("Module III output must be a new directory below reports/reruns")
    if root.is_relative_to(DEFAULT_PRE_READER):
        raise ContractError("Do not write into the sealed pre-reader tree")
    return root


def variant(task: str, kind: str, seed: int | None = None) -> str:
    if task not in FINE_TASKS:
        raise ContractError(f"Unregistered task: {task}")
    if kind == "fine" and seed is None:
        return f"fine__{task}"
    if kind == "fixed" and seed is None:
        return f"fixed__ctrl_{task}"
    if kind == "repeated" and seed in WT_SEEDS:
        return f"repeated__ctrl_{task}__wt{seed}"
    raise ContractError("Unregistered task kind or WT draw")


def variants() -> Iterator[tuple[str, str, int | None]]:
    for task in FINE_TASKS:
        yield task, "fine", None
        yield task, "fixed", None
        for seed in WT_SEEDS:
            yield task, "repeated", seed


def inner_seed(task: str, kind: str, seed: int | None, fold: int) -> int:
    variant(task, kind, seed)
    if fold not in FOLDS:
        raise ContractError("Unknown outer fold")
    draw_index = 0 if seed is None else WT_SEEDS.index(seed) + 1
    return 20260819 + 1000 * FINE_TASKS.index(task) + 10 * draw_index + fold


def make_inner_splits(labels: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        return []
    count = min(4, int(np.bincount(labels, minlength=2).min()))
    if count < 2:
        return []
    return list(StratifiedKFold(count, shuffle=True, random_state=seed).split(
        np.zeros(len(labels)), labels))


def patient_roster(manifest: pd.DataFrame) -> pd.DataFrame:
    required = ["patient_id", "subcohort", "k_fold", "target_label"]
    if set(required) - set(manifest) or manifest[required].isna().any().any():
        raise ContractError("Task manifest lacks patient labels/folds/subcohorts")
    for column in required[1:]:
        if manifest.groupby("patient_id")[column].nunique().gt(1).any():
            raise ContractError(f"Within-patient disagreement in {column}")
    result = manifest[required].drop_duplicates("patient_id").rename(columns={"target_label": "label"})
    result["patient_id"] = result["patient_id"].astype(str)
    result[["k_fold", "label"]] = result[["k_fold", "label"]].astype(int)
    if set(result.label) != {0, 1} or set(result.k_fold) != set(FOLDS):
        raise ContractError("Task must retain both classes and all five frozen folds")
    return result.sort_values("patient_id").reset_index(drop=True)


def validate_name_freeze(path: Path) -> dict[str, Any]:
    seal = read_json(path.with_suffix(path.suffix + ".seal.json"))
    require_identity(path, seal.get("artifact", {}))
    names = read_json(path)
    names["status"] = names.get("name_gate_status", names.get("status"))
    names["named_ref"] = names.get("NAMED_REF", names.get("named_ref", []))
    names["named_oof"] = names.get("NAMED_OOF", names.get("named_oof", {}))
    if names.get("status") not in {"NAME_GATE_PASS", "NAME_GATE_FAIL"}:
        raise ContractError("Completed sealed naming decision is required before modeling")
    if names["status"] == "NAME_GATE_PASS":
        if set(names.get("named_oof", {})) != set(map(str, FOLDS)):
            raise ContractError("Named set must be resolved separately for all five folds")
        for coordinates in [names.get("named_ref", []), *names["named_oof"].values()]:
            if coordinates != sorted(set(coordinates)) or not set(coordinates).issubset(range(32)):
                raise ContractError("Named coordinates must be sorted unique zero-based indices")
    return names


def dependency_preflight(root: Path, pre_reader: Path, name_freeze: Path,
                         aim3_root: Path) -> dict[str, Any]:
    """Record an operational dependency block without adjudicating any H3 test."""
    root = validate_output_root(root)
    require_identity(PREREG, {"sha256": PREREG_SHA256})
    require_identity(V13_MANIFEST, {"sha256": V13_MANIFEST_SHA256})
    names = validate_name_freeze(name_freeze)
    receipt = read_json(pre_reader / "receipts/profiles.json")
    if receipt.get("status") != "PASS":
        raise ContractError("Source profiles are incomplete")
    profile_inputs = []
    for fold in FOLDS:
        key = f"outer_fold_{fold}"
        path = pre_reader / "profiles" / key / "patient_profiles.parquet"
        profile_inputs.append(require_identity(path, receipt["artifacts"][key]["patient_profiles"]))
    required = [aim3_root / "contract.json", aim3_root / "analysis/patient_native_logits.parquet",
                aim3_root / "analysis/results.json"]
    for task, kind, seed in variants():
        key = variant(task, kind, seed)
        required.extend([aim3_root / f"inputs/manifests/{key}.csv",
                         aim3_root / f"inputs/splits/{key}/aim1_balanced5/splits.parquet"])
    missing = [str(path) for path in required if not path.is_file()]
    result = {"schema_version": 1, "component": "final_v14_module3_dependency_preflight",
              "status": "BLOCKED_MISSING_SEALED_AIM3_INPUTS" if missing else "DEPENDENCIES_PRESENT_PENDING_AUTHENTICATION",
              "scientific_status": "PENDING_NOT_TESTED", "scientific_gate_adjudicated": False,
              "name_gate_status": names["status"], "named_fits_authorized": names["status"] == "NAME_GATE_PASS",
              "name_freeze": identity(name_freeze), "source_profiles": profile_inputs,
              "aim3_root": str(aim3_root), "aim3_root_present": aim3_root.is_dir(),
              "required_path_count": len(required), "missing_paths": missing,
              "reason": ("Sealed source task manifests, splits, and parent score artifacts unavailable; restore their existing mount or byte-identical archive." if missing else None),
              "parent_verdicts_unchanged": True, "actual_model_fits_run": 0}
    # Content-addressed receipts preserve both a missing-mount observation and
    # a later restored-input observation without replacing historical evidence.
    digest = hashlib.sha256(json_bytes(result)).hexdigest()
    path = root / "dependency_preflight" / f"{digest}.json"
    write_json(path, result)
    write_json(path.with_suffix(".json.seal.json"), {"status": "SEALED", "artifact": identity(path)})
    print(f"Dependency receipt: {path}", flush=True)
    return result


def _relocated(record: Mapping[str, Any], aim3_root: Path) -> Path:
    original = Path(str(record["path"]))
    if not original.is_relative_to(V13_ROOT):
        raise ContractError("Inherited Aim3 input escaped its sealed source campaign")
    return aim3_root / original.relative_to(V13_ROOT)


def prepare(root: Path, pre_reader: Path, name_freeze: Path, aim3_root: Path) -> dict[str, Any]:
    """Seal source inputs without opening parent result values or target data."""
    root = validate_output_root(root)
    with exclusive_lock(root, "prepare"):
        if (root / "contract.json.seal.json").exists():
            return verify_contract(root)
        require_identity(PREREG, {"sha256": PREREG_SHA256})
        require_identity(V13_MANIFEST, {"sha256": V13_MANIFEST_SHA256})
        names = validate_name_freeze(name_freeze)
        source_records = {r["id"]: r for r in read_json(V13_MANIFEST)["artifacts"]}
        inherited_path = aim3_root / "contract.json"
        require_identity(inherited_path, source_records["aim3-source-primary-campaign-contract"])
        inherited = read_json(inherited_path)
        if inherited["protocol"]["fine_tasks"] != list(FINE_TASKS):
            raise ContractError("Inherited task index order drifted")
        profile_receipt = read_json(pre_reader / "receipts/profiles.json")
        if profile_receipt.get("status") != "PASS":
            raise ContractError("Source profiles are incomplete")
        dictionary = read_json(pre_reader / "analysis_dictionary.json")
        dictionary_seal = read_json(pre_reader / "analysis_dictionary.json.seal.json")
        require_identity(pre_reader / "analysis_dictionary.json", dictionary_seal["artifact"])
        if dictionary["ordered_indices"]["fine_tasks"] != list(FINE_TASKS):
            raise ContractError("Analysis dictionary task indices drifted")
        inputs: dict[str, Any] = {}

        def bind(key: str, source: Path, destination: str) -> Path:
            target = root / destination
            write_once(target, source.read_bytes())
            inputs[key] = {"source": identity(source), "frozen": identity(target)}
            return target

        bind("preregistration", PREREG, "inputs/preregistration.md")
        bind("name_freeze", name_freeze, "inputs/name_freeze.json")
        bind("name_freeze_seal", name_freeze.with_suffix(name_freeze.suffix + ".seal.json"), "inputs/name_freeze.json.seal.json")
        bind("dictionary", pre_reader / "analysis_dictionary.json", "inputs/analysis_dictionary.json")
        bind("dictionary_seal", pre_reader / "analysis_dictionary.json.seal.json", "inputs/analysis_dictionary.json.seal.json")
        bind("profile_receipt", pre_reader / "receipts/profiles.json", "inputs/profile_receipt.json")
        bind("parent_manifest", V13_MANIFEST, "inputs/v13_source_manifest.json")
        bind("aim3_contract", inherited_path, "inputs/aim3_contract.json")
        source_path = pre_reader / "inputs/tcga_surgen_primary.csv"
        require_identity(source_path, inherited["frozen_source_lineage"]["manifest"])
        source_roster = patient_roster(pd.read_csv(source_path))
        bind("source_manifest", source_path, "inputs/source_manifest.csv")
        for fold in FOLDS:
            key = f"outer_fold_{fold}"
            path = pre_reader / "profiles" / key / "patient_profiles.parquet"
            require_identity(path, profile_receipt["artifacts"][key]["patient_profiles"])
            frozen = bind(key, path, f"inputs/profiles/{key}.parquet")
            frame = pd.read_parquet(frozen)
            if frame.patient_id.duplicated().any() or set(frame.patient_id) != set(source_roster.patient_id):
                raise ContractError("Outer vocabulary matrix source roster drifted")
            if not frame.vocabulary_id.eq(key).all():
                raise ContractError("Outer vocabulary matrix uses incorrect dictionary")
            mass = frame[PROTOTYPES].to_numpy(float)
            if not np.isfinite(mass).all() or (mass < 0).any() or not np.allclose(mass.sum(1), 1, rtol=0, atol=1e-12):
                raise ContractError("Invalid abundance profile")
            joined = source_roster.merge(frame, on="patient_id", suffixes=("", "_profile"), validate="one_to_one")
            if not joined.k_fold.eq(joined.fold).all() or not joined.subcohort.eq(joined.subcohort_profile).all():
                raise ContractError("Profile outer fold/subcohort disagrees with frozen source")
        task_inputs = {}
        for task, kind, seed in variants():
            key = variant(task, kind, seed)
            record = inherited["task_inputs"][key]
            path = _relocated(record["manifest"], aim3_root)
            require_identity(path, record["manifest"])
            task_manifest = pd.read_csv(path)
            roster = patient_roster(task_manifest)
            source_check = roster.merge(source_roster, on="patient_id", suffixes=("", "_source"), validate="one_to_one")
            if len(source_check) != len(roster) or not source_check.k_fold.eq(source_check.k_fold_source).all():
                raise ContractError("Task roster is not a frozen source subset")
            frozen = bind(key, path, f"inputs/manifests/{key}.csv")
            split_path = _relocated(record["split"]["splits"], aim3_root)
            require_identity(split_path, record["split"]["splits"])
            bind(f"{key}_splits", split_path, f"inputs/splits/{key}.parquet")
            splits = pd.read_parquet(split_path)
            check = task_manifest[["slide_id", "k_fold"]].merge(splits[["slide_id", "fold"]], on="slide_id", validate="one_to_one")
            if len(check) != len(task_manifest) or len(check) != len(splits) or not check.k_fold.eq(check.fold).all():
                raise ContractError("Task split differs from inherited manifest")
            task_inputs[key] = {"manifest": identity(frozen), "patients": len(roster),
                                "positive": int(roster.label.sum())}
        implementation = []
        for relative in ("tools/final_v14_module3.py", "tools/final_v14_modeling_v2.py", "tests/test_final_v14_module3.py", "tests/test_final_v14_modeling_v2.py", "uv.lock"):
            path = REPO / relative
            frozen = bind(relative, path, f"source_snapshot/{relative}")
            implementation.append(identity(frozen))
        contract = {
            "schema_version": 1, "component": "final_v14_module3", "status": "SOURCE_INPUTS_SEALED",
            "inputs": inputs, "task_inputs": task_inputs, "implementation": implementation,
            "runtime": {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scikit-learn", "scipy")},
            "representations": ["ALL32"] + (["NAMED_OOF"] if names["status"] == "NAME_GATE_PASS" else []),
            "fine_tasks": list(FINE_TASKS), "wt_seeds": list(WT_SEEDS),
            "folds": list(FOLDS), "canonical_bootstraps": 10000, "repeated_bootstraps": 20000,
            "fitting": {"penalty": "L2", "C_grid": [10.0 ** i for i in range(-4, 5)],
                        "solver": "lbfgs", "tol": 1e-8, "max_iter": 10000,
                        "class_weight": None, "intercept_penalized": False,
                        "inner_folds": "min(4, outer-training minority class count), minimum 2",
                        "inner_seed": "20260819 + 1000*task_index + 10*draw_index + outer_fold",
                        "selection": "mean inner-fold log-loss; ties within 1e-12 choose smaller C",
                        "scaler": "fit on relevant training patients only; population standard deviation",
                        "zero_variance": "scale 1, coefficient 0",
                        "failure": "all inner fits required; failed selected refit has no fallback"},
            "repeated_seed": 20260826, "repeated_chunk_size": 256,
            "repeated_sampling": "one PCG64 task stream; shared positives and fine negatives; independent sequential WT negative draws",
            "formal_gates": "ALL32 only", "parent_estimates_unchanged": True,
            "aim3_root": str(aim3_root.resolve()), "target_access": "forbidden",
        }
        write_json(root / "contract.json", contract)
        seal_file(root / "contract.json")
        return contract


def verify_contract(root: Path) -> dict[str, Any]:
    verify_seal(root / "contract.json")
    contract = read_json(root / "contract.json")
    for record in contract["inputs"].values():
        require_identity(Path(record["frozen"]["path"]), record["frozen"])
    for relative in ("tools/final_v14_module3.py", "tools/final_v14_modeling_v2.py"):
        require_identity(REPO / relative, contract["inputs"][relative]["source"])
    actual = {package: importlib.metadata.version(package) for package in contract["runtime"]}
    if actual != contract["runtime"]:
        raise ContractError("Modeling runtime versions drifted from source freeze")
    validate_name_freeze(root / "inputs/name_freeze.json")
    return contract


def fit_outer(roster: pd.DataFrame, profiles: pd.DataFrame, coordinates: list[int],
              task: str, kind: str, seed: int | None, fold: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit a single patient-inductive readout; held-out labels are not passed."""
    from tools.final_v14_modeling_v2 import nested_logistic

    roster = roster.sort_values("patient_id").reset_index(drop=True)
    frame = roster.merge(profiles[["patient_id", *PROTOTYPES]], on="patient_id", validate="one_to_one", how="left")
    if frame[PROTOTYPES].isna().any().any():
        raise ContractError("Missing patient abundance vector")
    train = frame.k_fold.ne(fold).to_numpy()
    test = ~train
    columns = [PROTOTYPES[c] for c in coordinates]
    labels = frame.loc[train, "label"].to_numpy(dtype=int)
    random_state = inner_seed(task, kind, seed, fold)
    splits = make_inner_splits(labels, random_state)
    audit = {"task": task, "kind": kind, "draw_seed": seed, "fold": fold,
             "coordinates": coordinates, "n_train": int(train.sum()), "n_test": int(test.sum()),
             "inner_seed": random_state, "n_inner_folds": len(splits),
             "training_patient_ids": frame.loc[train, "patient_id"].tolist(),
             "inner_validation_patient_ids": [frame.loc[train, "patient_id"].iloc[v].tolist() for _, v in splits]}
    predictions = frame.loc[test, ["patient_id", "label", "subcohort", "k_fold"]].copy()
    if not coordinates or not splits:
        fit = {"status": "NOT_ESTIMABLE", "reason": "zero predictors" if not coordinates else "minority count < 2"}
    else:
        fit = nested_logistic(frame.loc[train, columns].to_numpy(float), labels,
                              frame.loc[test, columns].to_numpy(float), splits)
    predictions["mean_logit"] = fit.get("predictions", np.full(int(test.sum()), np.nan))
    audit.update({key: value for key, value in fit.items() if key not in {
        "predictions", "scaler_mean", "scaler_scale", "coef", "intercept"}})
    parameters = {key: fit[key] for key in ("scaler_mean", "scaler_scale", "coef", "intercept") if key in fit}
    return {"audit": audit, "parameters": parameters}, predictions


def fit_source(root: Path) -> dict[str, Any]:
    root = validate_output_root(root)
    with exclusive_lock(root, "fit-source"), threadpool_limits(limits=1):
        contract = verify_contract(root)
        names = validate_name_freeze(root / "inputs/name_freeze.json")
        profiles = {fold: pd.read_parquet(root / f"inputs/profiles/outer_fold_{fold}.parquet") for fold in FOLDS}
        outputs = []
        for representation in contract["representations"]:
            for task, kind, seed in variants():
                key = variant(task, kind, seed)
                roster = patient_roster(pd.read_csv(root / f"inputs/manifests/{key}.csv"))
                for fold in FOLDS:
                    directory = root / "fits" / representation / key / f"fold_{fold}"
                    receipt_path = directory / "receipt.json"
                    if receipt_path.exists():
                        receipt = read_json(receipt_path)
                        for record in receipt["artifacts"].values():
                            require_identity(Path(record["path"]), record)
                    else:
                        coordinates = list(range(32)) if representation == "ALL32" else names["named_oof"][str(fold)]
                        result, predictions = fit_outer(roster, profiles[fold], coordinates, task, kind, seed, fold)
                        write_parquet(directory / "predictions.parquet", predictions)
                        write_npz(directory / "parameters.npz", result["parameters"])
                        write_json(directory / "audit.json", result["audit"])
                        receipt = {"status": result["audit"]["status"], "artifacts": {
                            name: identity(directory / name) for name in ("predictions.parquet", "parameters.npz", "audit.json")}}
                        write_json(receipt_path, receipt)
                        print(f"{representation} {key} fold {fold}: {receipt['status']}", flush=True)
                    outputs.append(identity(receipt_path))
        completion = {"status": "SOURCE_FITS_COMPLETE", "contract": identity(root / "contract.json"),
                      "fold_receipts": outputs, "outer_fit_count": len(outputs), "target_access": False}
        write_json(root / "source_completion.json", completion)
        seal_file(root / "source_completion.json")
        return completion


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Inherited tie-correct Mann–Whitney AUROC on native logits."""
    if not len(positive) or not len(negative) or not np.isfinite(positive).all() or not np.isfinite(negative).all():
        return float("nan")
    ordered = np.sort(negative)
    left = np.searchsorted(ordered, positive, side="left")
    right = np.searchsorted(ordered, positive, side="right")
    return float(np.sum(left + 0.5 * (right - left)) / (len(positive) * len(negative)))


def _groups(frame: pd.DataFrame) -> list[np.ndarray]:
    return [np.asarray(indices) for indices in frame.groupby(["subcohort", "k_fold"], sort=True).indices.values()]


def _draw_indices(frame: pd.DataFrame, count: int, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([indices[rng.integers(0, len(indices), size=(count, len(indices)))]
                           for indices in _groups(frame)], axis=1)


def _validate_bootstrap_frame(frame: pd.DataFrame) -> pd.DataFrame:
    needed = {"patient_id", "label", "mean_logit", "subcohort", "k_fold"}
    if needed - set(frame) or frame.patient_id.duplicated().any() or set(frame.label) != {0, 1}:
        raise ContractError("Invalid unique binary patient OOF vector")
    if frame[list(needed - {"mean_logit"})].isna().any().any():
        raise ContractError("Missing patient inference strata")
    return frame.sort_values("patient_id").reset_index(drop=True)


def partially_paired_bootstrap(fine: pd.DataFrame, controls: Mapping[str, pd.DataFrame], *,
                              n_bootstrap: int, seed: int, chunk_size: int = 256) -> dict[str, np.ndarray]:
    """Draw shared patient indices once, with independent WT pools per draw.

    A single-control call reproduces the inherited canonical algorithm. For
    repeated controls, consume all three independent negative-pool draws in
    fixed WT-seed order in each chunk. ``mil_logit`` is evaluated on exactly
    the same fine patient indices. Non-estimable logits remain NaN, retaining
    the full roster and RNG consumption for other estimable quantities.
    """
    if n_bootstrap < 1 or chunk_size < 1:
        raise ValueError("Positive bootstrap and chunk counts required")
    fine = _validate_bootstrap_frame(fine)
    positive = fine[fine.label.eq(1)].reset_index(drop=True)
    negative = fine[fine.label.eq(0)].reset_index(drop=True)
    controls = {str(key): _validate_bootstrap_frame(frame) for key, frame in controls.items()}
    negative_counts = negative.groupby(["subcohort", "k_fold"], sort=True).size()
    control_parts = {}
    for key, frame in controls.items():
        cp, cn = (frame[frame.label.eq(label)].reset_index(drop=True) for label in (1, 0))
        if not positive[["patient_id", "subcohort", "k_fold"]].equals(cp[["patient_id", "subcohort", "k_fold"]]):
            raise ContractError("Fine and control positives must share identical patients and strata")
        if set(negative.patient_id) & set(cn.patient_id):
            raise ContractError("Fine-negative and WT-negative patient pools overlap")
        if not negative_counts.equals(cn.groupby(["subcohort", "k_fold"], sort=True).size()):
            raise ContractError("Control negatives are not matched by source subcohort and outer fold")
        control_parts[key] = cp, cn
    values = {"fine": np.empty(n_bootstrap), **{f"control_{key}": np.empty(n_bootstrap) for key in controls}}
    if "mil_logit" in fine:
        values["mil"] = np.empty(n_bootstrap)
    rng = np.random.default_rng(seed)
    for cursor in range(0, n_bootstrap, chunk_size):
        count = min(chunk_size, n_bootstrap - cursor)
        pidx = _draw_indices(positive, count, rng)
        nidx = _draw_indices(negative, count, rng)
        control_indices = {key: _draw_indices(cn, count, rng) for key, (_, cn) in control_parts.items()}
        for row in range(count):
            dest = cursor + row
            values["fine"][dest] = auc(positive.mean_logit.to_numpy()[pidx[row]], negative.mean_logit.to_numpy()[nidx[row]])
            if "mil" in values:
                values["mil"][dest] = auc(positive.mil_logit.to_numpy()[pidx[row]], negative.mil_logit.to_numpy()[nidx[row]])
            for key, (cp, cn) in control_parts.items():
                values[f"control_{key}"][dest] = auc(cp.mean_logit.to_numpy()[pidx[row]], cn.mean_logit.to_numpy()[control_indices[key][row]])
    for key in controls:
        values[f"control_minus_fine_{key}"] = values[f"control_{key}"] - values["fine"]
    if "mil" in values:
        values["concept_minus_mil"] = values["fine"] - values["mil"]
    return values


def canonical_seed(task: str) -> int:
    token = f"20260817\0tcga_surgen_primary_fixed\0{task}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 2**32


def repeated_seed(task: str) -> int:
    return int(np.random.SeedSequence([20260826, FINE_TASKS.index(task)]).generate_state(1)[0])


def summarize(point: float, samples: np.ndarray, *, governed: bool = True) -> dict[str, Any]:
    if not np.isfinite(point) or not np.isfinite(samples).all():
        return {"status": "NOT_ESTIMABLE", "estimate": None, "n_bootstrap": len(samples),
                "finite_draws": int(np.isfinite(samples).sum())}
    result = {"status": "ESTIMABLE", "estimate": float(point), "n_bootstrap": len(samples),
              "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}
    if governed:
        result["fwer_lower99"], result["fwer_upper99"] = np.quantile(samples, [0.01, 0.99]).tolist()
    return result


def draw_verdict(fine: Mapping[str, Any], control: Mapping[str, Any], delta: Mapping[str, Any]) -> str:
    if any(value.get("status") != "ESTIMABLE" for value in (fine, control, delta)):
        return "NOT_EVALUABLE"
    if control["fwer_lower99"] <= 0.5:
        return "UNDERPOWERED"
    ceiling = fine["fwer_upper99"] < 0.60 and delta["fwer_lower99"] > 0
    signal = fine["fwer_lower99"] > 0.5
    if ceiling:
        return "CEILING_WITH_RESIDUAL_SIGNAL" if signal else "CEILING"
    return "FINE_RESOLUTION_EVIDENCE" if signal else "INCONCLUSIVE"


def task_statuses(fine: Mapping[str, Any], paired: Mapping[str, Any],
                  draws: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(draws) != set(map(str, WT_SEEDS)):
        raise ContractError("Task requires all three prespecified WT draws")
    fine_ok = fine.get("status") == "ESTIMABLE"
    control_ok = all(draw["control"].get("status") == "ESTIMABLE" for draw in draws.values())
    adequate = control_ok and all(draw["control"]["fwer_lower99"] > 0.5 for draw in draws.values())
    signal = fine_ok and fine["fwer_lower99"] > 0.5
    control_status = ("CONCEPT_CONTROL_NOT_EVALUABLE" if not control_ok else
                      "CONCEPT_CONTROL_ADEQUATE" if adequate else "CONCEPT_CONTROL_INADEQUATE")
    boundary = "CONCEPT_BOUNDARY_NOT_EVALUABLE"
    if fine_ok and adequate:
        consistent = all(draw["verdict"] in {"CEILING", "CEILING_WITH_RESIDUAL_SIGNAL"} for draw in draws.values())
        boundary = "CONCEPT_NONRESOLUTION_CONSISTENT" if consistent else "CONCEPT_NONRESOLUTION_NOT_ESTABLISHED"
    superiority = "CONCEPT_FINE_RANKING_SUPERIORITY_NOT_EVALUABLE"
    if fine_ok and control_ok and paired.get("status") == "ESTIMABLE":
        passed = adequate and signal and paired["fwer_lower99"] > 0
        superiority = "CONCEPT_FINE_RANKING_SUPERIOR_TO_MIL" if passed else "CONCEPT_FINE_RANKING_SUPERIORITY_NOT_ESTABLISHED"
    return {"control": control_status, "boundary": boundary,
            "fine_signal": ("CONCEPT_FINE_SIGNAL_NOT_EVALUABLE" if not fine_ok else
                            "CONCEPT_FINE_SIGNAL_SUPPORTED" if signal else "CONCEPT_FINE_SIGNAL_NOT_ESTABLISHED"),
            "superiority": superiority,
            "point_fine_auroc_at_least_060": bool(fine_ok and fine["estimate"] >= 0.60)}


def _point(frame: pd.DataFrame, column: str = "mean_logit") -> float:
    return auc(frame.loc[frame.label.eq(1), column].to_numpy(float), frame.loc[frame.label.eq(0), column].to_numpy(float))


def _oof(root: Path, representation: str, key: str) -> pd.DataFrame:
    blocks = []
    for fold in FOLDS:
        directory = root / "fits" / representation / key / f"fold_{fold}"
        receipt = read_json(directory / "receipt.json")
        for record in receipt["artifacts"].values():
            require_identity(Path(record["path"]), record)
        frame = pd.read_parquet(directory / "predictions.parquet")
        if not frame.k_fold.eq(fold).all():
            raise ContractError("Prediction scored outside its held-out fold")
        if receipt["status"] != "ESTIMABLE":
            frame["mean_logit"] = np.nan
        blocks.append(frame)
    result = pd.concat(blocks, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
    expected = patient_roster(pd.read_csv(root / f"inputs/manifests/{key}.csv"))
    pd.testing.assert_frame_equal(result[expected.columns], expected, check_dtype=False)
    return result


def evaluate(root: Path) -> dict[str, Any]:
    root = validate_output_root(root)
    with exclusive_lock(root, "evaluate"):
        contract = verify_contract(root)
        verify_seal(root / "source_completion.json")
        completion = read_json(root / "source_completion.json")
        expected_receipts = {str(root / "fits" / representation / variant(task, kind, seed) / f"fold_{fold}" / "receipt.json")
                             for representation in contract["representations"]
                             for task, kind, seed in variants() for fold in FOLDS}
        if (completion.get("status") != "SOURCE_FITS_COMPLETE"
                or completion.get("outer_fit_count") != len(expected_receipts)
                or len(completion["fold_receipts"]) != len(expected_receipts)
                or {record["path"] for record in completion["fold_receipts"]} != expected_receipts):
            raise ContractError("Source completion does not contain all required outer fits")
        require_identity(root / "contract.json", completion["contract"])
        for record in completion["fold_receipts"]:
            require_identity(Path(record["path"]), record)
        aim3_root = Path(contract["aim3_root"])
        logits_path = aim3_root / "analysis/patient_native_logits.parquet"
        require_identity(logits_path, {"sha256": V13_LOGITS_SHA256})
        source_records = {r["id"]: r for r in read_json(root / "inputs/v13_source_manifest.json")["artifacts"]}
        results_path = aim3_root / "analysis/results.json"
        require_identity(results_path, source_records["aim3-source-primary-results"])
        mil = pd.read_parquet(logits_path)
        inherited = read_json(results_path)
        results = {"schema_version": 1, "component": "final_v14_module3", "status": "COMPLETE",
                   "source_completion": identity(root / "source_completion.json"),
                   "v13_comparators": {"patient_logits": identity(logits_path), "results": identity(results_path)},
                   "representations": {}, "parent_estimates_unchanged": True,
                   "claim_ceiling": "Related UNI-derived representation/readout consistency only; no information-absence or causal claim."}
        for representation in contract["representations"]:
            task_results = {}
            for task in FINE_TASKS:
                directory = root / "analysis" / representation / task
                result_path = directory / "results.json"
                if result_path.with_suffix(".json.seal.json").exists():
                    verify_seal(result_path)
                    row = read_json(result_path)
                    require_identity(directory / "bootstrap.npz", row["bootstrap"])
                    task_results[task] = row
                    continue
                fine = _oof(root, representation, variant(task, "fine"))
                comparator = mil.loc[mil.section.eq("fine") & mil.rung.eq(task),
                                     ["patient_id", "label", "k_fold", "subcohort", "mean_logit"]]
                if comparator.patient_id.duplicated().any():
                    raise ContractError("MIL comparator duplicates fine-task patients")
                joined = fine.merge(comparator, on=["patient_id", "label", "k_fold", "subcohort"],
                                    suffixes=("", "_mil"), validate="one_to_one")
                if len(joined) != len(fine) or len(comparator) != len(fine):
                    raise ContractError("Paired MIL comparator is not the exact fine-task roster")
                fine = joined.rename(columns={"mean_logit_mil": "mil_logit"})
                controls = {str(seed): _oof(root, representation, variant(task, "repeated", seed)) for seed in WT_SEEDS}
                repeated = partially_paired_bootstrap(fine, controls, n_bootstrap=20000, seed=repeated_seed(task))
                fine_summary = summarize(_point(fine), repeated["fine"])
                paired = summarize(_point(fine) - _point(fine, "mil_logit"), repeated["concept_minus_mil"])
                draw_rows = {}
                for seed in WT_SEEDS:
                    key = str(seed)
                    control = summarize(_point(controls[key]), repeated[f"control_{key}"])
                    delta = summarize(_point(controls[key]) - _point(fine), repeated[f"control_minus_fine_{key}"])
                    draw_rows[key] = {"fine": fine_summary, "control": control,
                                      "control_minus_fine": delta, "verdict": draw_verdict(fine_summary, control, delta)}
                fixed = _oof(root, representation, variant(task, "fixed"))
                canonical = partially_paired_bootstrap(fine, {"canonical": fixed}, n_bootstrap=10000, seed=canonical_seed(task))
                canonical_row = {"fine": summarize(_point(fine), canonical["fine"], governed=False),
                                 "control": summarize(_point(fixed), canonical["control_canonical"], governed=False),
                                 "control_minus_fine": summarize(_point(fixed) - _point(fine), canonical["control_minus_fine_canonical"], governed=False),
                                 "role": "descriptive only; does not determine any status"}
                arrays = {f"repeated__{key}": value for key, value in repeated.items()}
                arrays.update({f"canonical__{key}": value for key, value in canonical.items()})
                write_npz(directory / "bootstrap.npz", arrays)
                row = {"task": task, "representation": representation,
                       "inference_role": "formal" if representation == "ALL32" else "secondary name-dependent sensitivity; no formal H3 gate",
                       "inherited_v13_consensus": inherited["repeated"]["rungs"][task]["consensus_verdict"],
                       "inherited_v13_repeated_unchanged": inherited["repeated"]["rungs"][task],
                       "inherited_v13_canonical_unchanged": inherited["fixed"]["rungs"][task],
                       "fine": fine_summary, "mil_fine": summarize(_point(fine, "mil_logit"), repeated["mil"]),
                       "mil_fine_interval_scope": "Comparator under the v14 paired patient draws; inherited v13 intervals retained unchanged separately.",
                       "concept_minus_mil": paired, "repeated_draws": draw_rows,
                       "canonical": canonical_row, "bootstrap": identity(directory / "bootstrap.npz")}
                if representation == "ALL32":
                    row["statuses"] = task_statuses(fine_summary, paired, draw_rows)
                write_json(result_path, row)
                seal_file(result_path)
                task_results[task] = row
                print(f"Evaluated {representation}/{task}", flush=True)
            results["representations"][representation] = task_results
        write_json(root / "results.json", results)
        seal_file(root / "results.json")
        return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "prepare", "fit-source", "evaluate", "verify"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pre-reader-root", type=Path, default=DEFAULT_PRE_READER)
    parser.add_argument("--name-freeze", type=Path)
    parser.add_argument("--aim3-root", type=Path, default=V13_ROOT)
    args = parser.parse_args(argv)
    if args.command in {"prepare", "preflight"}:
        if args.name_freeze is None:
            parser.error("prepare requires --name-freeze with an adjacent artifact SHA256 seal")
        action = prepare if args.command == "prepare" else dependency_preflight
        result = action(args.output_root, args.pre_reader_root, args.name_freeze, args.aim3_root)
    elif args.command == "fit-source":
        result = fit_source(args.output_root)
    elif args.command == "evaluate":
        result = evaluate(args.output_root)
    else:
        result = verify_contract(args.output_root)
    print(json.dumps({"status": result["status"], "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
