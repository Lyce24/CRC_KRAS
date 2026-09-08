#!/usr/bin/env python3
"""E2f-v3: audited full-label metastatic residual adaptation.

This is a new append-only lineage.  It does not read or modify E2f-v1/v2.
It fixes the v2 tuning-boundary and auditability defects while preserving the
same patient-level estimand:

    eta_adapted = eta_native + H @ delta_w + delta_b

The residual head minimizes mean logistic loss plus an L2 penalty.  Lambda is
selected strictly inside each outer training pool.  The primary result uses
outer-fold seed 20260821; four consecutive neighbouring layouts are repeated
as procedure sensitivity.  A two-parameter, outer-cross-fitted Platt model is
reported as a low-capacity local-calibration comparator.

IMPORTANT SCOPE: this experiment uses metastatic target labels internally for
cross-fitting.  Held-out patients receive honest OOF predictions, but this is
not independent deployment validation, prospective validation, or evidence of
clinical readiness.

Usage:
    python aim2_v3_fulllabel_residual_adaptation.py run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
UPSTREAM_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim2_cap8192_v4_20260819"
)
EMBEDDING_DIR = UPSTREAM_ROOT / "e2c" / "embeddings"
MANIFEST_DIR = Path("/mnt/d/YC.Liu/manifests/colon")
OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_e2f_fulllabel_adapter_v3_20260821"
)
V1_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_e2f_fulllabel_adapter_v1_20260821"
)
V2_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_e2f_fulllabel_adapter_v2_20260821"
)

COHORTS: dict[str, str] = {"RIH": "rih", "SurGen": "surgen"}
SEEDS: tuple[int, ...] = (42, 43, 44)
CAP = 8192
N_FOLDS = 5
PRIMARY_OUTER_SEED = 20260821
OUTER_SENSITIVITY_SEEDS: tuple[int, ...] = tuple(range(20260817, 20260822))
LAMBDA_GRID: tuple[float, ...] = (
    np.inf,
    10000.0,
    3000.0,
    1000.0,
    300.0,
    100.0,
    30.0,
    10.0,
    3.0,
    1.0,
    0.3,
    0.1,
    0.03,
    0.01,
    0.003,
    0.001,
)
SUPPORT_SIZES: tuple[int, ...] = (8, 16, 32, 48)
SUPPORT_DRAWS = 20
LOO_POOL_MAX = 20
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260817
SOLVER_GRAD_TOL = 1e-6
OBJECTIVE_TOL = 1e-9
PROB_EPS = 1e-9
E2B_EXPECTED = {"RIH": 0.606982, "SurGen": 0.562121}
SCOPE_WARNING = (
    "Target-label internal metastatic cross-fitting; not independent deployment "
    "validation, prospective validation, or clinical-readiness evidence."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"expected file artifact: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value):
            raise ValueError("NaN is not allowed in sealed JSON")
        if np.isposinf(value):
            return "infinity"
        if np.isneginf(value):
            return "-infinity"
    return value


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(json_ready(value), indent=2, allow_nan=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(json_ready(row), allow_nan=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def lambda_key(value: float) -> str:
    return "infinity" if not np.isfinite(value) else format(float(value), ".12g")


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def git_state() -> dict[str, Any]:
    """Record, but do not require, the repository state around untracked analysis code."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", SCRIPT.name],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
        ).returncode == 0
        return {
            "head": head,
            "dirty": bool(status),
            "status_entry_count": len(status.splitlines()),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "generating_code_tracked": tracked,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def manifest_path(cohort: str) -> Path:
    return MANIFEST_DIR / f"aim1_e2a_{COHORTS[cohort]}_metastatic.csv"


def embedding_path(cohort: str) -> Path:
    return EMBEDDING_DIR / f"cap{CAP}_{COHORTS[cohort]}_metastatic.parquet"


def embedding_receipt_path(cohort: str) -> Path:
    return embedding_path(cohort).with_suffix(".receipt.json")


def validate_inputs() -> tuple[dict[str, Any], dict[str, dict[int, pd.DataFrame]]]:
    """Validate and load every consumed upstream file."""
    identities: dict[str, Any] = {
        "upstream_lineage_start": artifact_identity(UPSTREAM_ROOT / "lineage_start.json"),
        "superseded_lineages": {
            "v1": {
                "results": artifact_identity(V1_ROOT / "analysis" / "results.json"),
                "receipt": artifact_identity(V1_ROOT / "analysis" / "receipt.json"),
            },
            "v2": {
                "results": artifact_identity(V2_ROOT / "analysis" / "results.json"),
                "receipt": artifact_identity(V2_ROOT / "analysis" / "receipt.json"),
            },
        },
        "cohorts": {},
    }
    data: dict[str, dict[int, pd.DataFrame]] = {}
    for cohort in COHORTS:
        emb = embedding_path(cohort)
        manifest_file = manifest_path(cohort)
        emb_receipt_file = embedding_receipt_path(cohort)
        emb_identity = artifact_identity(emb)
        manifest_identity = artifact_identity(manifest_file)
        emb_receipt_identity = artifact_identity(emb_receipt_file)
        receipt = json.loads(emb_receipt_file.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != 2:
            raise RuntimeError(f"{cohort}: upstream embedding receipt schema is not 2")
        if receipt.get("artifact", {}).get("sha256") != emb_identity["sha256"]:
            raise RuntimeError(f"{cohort}: upstream embedding hash mismatch")
        if receipt.get("inputs", {}).get("manifest", {}).get("sha256") != manifest_identity["sha256"]:
            raise RuntimeError(f"{cohort}: manifest hash differs from upstream receipt")
        upstream_code_path = Path(receipt["inputs"]["code"]["path"])
        upstream_code_identity = artifact_identity(upstream_code_path)
        if upstream_code_identity["sha256"] != receipt["inputs"]["code"]["sha256"]:
            raise RuntimeError(f"{cohort}: current upstream code does not match its receipt")

        manifest = pd.read_csv(manifest_file, low_memory=False)
        required_manifest = {"slide_id", "patient_id", "target_label"}
        if not required_manifest.issubset(manifest.columns):
            raise RuntimeError(f"{cohort}: malformed manifest")
        if manifest["slide_id"].duplicated().any():
            raise RuntimeError(f"{cohort}: duplicate slide IDs in manifest")
        if manifest[["slide_id", "patient_id", "target_label"]].isna().any().any():
            raise RuntimeError(f"{cohort}: missing manifest identity or label")
        if not set(manifest["target_label"].astype(int).unique()).issubset({0, 1}):
            raise RuntimeError(f"{cohort}: non-binary target label")
        discordance = manifest.groupby("patient_id")["target_label"].nunique()
        if int((discordance != 1).sum()) != 0:
            raise RuntimeError(f"{cohort}: slide labels disagree within patient")

        raw = pd.read_parquet(emb)
        required_raw = {"slide_id", "seed", "logit"}
        if not required_raw.issubset(raw.columns):
            raise RuntimeError(f"{cohort}: malformed embedding parquet")
        if set(map(int, raw["seed"].unique())) != set(SEEDS):
            raise RuntimeError(f"{cohort}: unexpected model-seed inventory")
        ecols = sorted(
            [name for name in raw.columns if name.startswith("e") and name[1:].isdigit()],
            key=lambda name: int(name[1:]),
        )
        if ecols != [f"e{i}" for i in range(512)]:
            raise RuntimeError(f"{cohort}: expected exactly e0..e511")

        slide_map = manifest[["slide_id", "patient_id", "target_label"]]
        per_seed: dict[int, pd.DataFrame] = {}
        for seed in SEEDS:
            seed_raw = raw.loc[raw["seed"] == seed]
            block = seed_raw.merge(slide_map, on="slide_id", validate="one_to_one")
            if len(block) != len(manifest) or block["slide_id"].nunique() != len(manifest):
                raise RuntimeError(f"{cohort} seed{seed}: incomplete slide coverage")
            if not np.isfinite(block[["logit", *ecols]].to_numpy(dtype=float)).all():
                raise RuntimeError(f"{cohort} seed{seed}: non-finite model value")
            grouped = block.groupby("patient_id", sort=True)
            if int((grouped["target_label"].nunique() != 1).sum()) != 0:
                raise RuntimeError(f"{cohort} seed{seed}: patient label discordance")
            patient = grouped.agg(
                label=("target_label", "first"),
                eta_native=("logit", "mean"),
                **{column: (column, "mean") for column in ecols},
            )
            patient["label"] = patient["label"].astype(int)
            per_seed[seed] = patient

        reference = per_seed[SEEDS[0]]
        for seed in SEEDS[1:]:
            other = per_seed[seed]
            if not reference.index.equals(other.index):
                raise RuntimeError(f"{cohort}: patient order differs across model seeds")
            if not np.array_equal(reference["label"].to_numpy(), other["label"].to_numpy()):
                raise RuntimeError(f"{cohort}: labels differ across model seeds")

        data[cohort] = per_seed
        identities["cohorts"][cohort] = {
            "embedding": emb_identity,
            "embedding_receipt": emb_receipt_identity,
            "manifest": manifest_identity,
            "upstream_generator_code": upstream_code_identity,
            "census": {
                "n_slides": int(len(manifest)),
                "n_patients": int(len(reference)),
                "n_mutant": int(reference["label"].sum()),
                "n_multislide_patients": int(
                    (manifest.groupby("patient_id").size() > 1).sum()
                ),
                "n_embedding_dimensions": len(ecols),
                "model_seeds": list(SEEDS),
                "label_discordances": 0,
                "nonfinite_values": 0,
            },
        }
    return identities, data


def stratified_folds(labels: np.ndarray, seed: int) -> np.ndarray:
    """The deterministic five-fold layout used by E2f-v2."""
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=int)
    dealt = 0
    for value in (1, 0):
        index = np.flatnonzero(labels == value)
        rng.shuffle(index)
        for position, item in enumerate(index):
            folds[item] = (dealt + position) % N_FOLDS
        dealt += len(index)
    if sorted(np.unique(folds).tolist()) != list(range(N_FOLDS)):
        raise RuntimeError("fold construction did not populate all five folds")
    return folds


@dataclass
class SolverLedger:
    n_finite_calls: int = 0
    n_native_exact_calls: int = 0
    n_scipy_success: int = 0
    n_accepted_small_gradient: int = 0
    max_gradient_inf_norm: float = 0.0
    min_objective_decrease: float = float("inf")
    status_counts: Counter[str] = field(default_factory=Counter)
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def record(self, diagnostic: dict[str, Any], context: dict[str, Any]) -> None:
        if diagnostic["status"] == "native_exact":
            self.n_native_exact_calls += 1
            return
        self.n_finite_calls += 1
        self.status_counts[str(diagnostic["scipy_status"])] += 1
        self.max_gradient_inf_norm = max(
            self.max_gradient_inf_norm, float(diagnostic["gradient_inf_norm"])
        )
        self.min_objective_decrease = min(
            self.min_objective_decrease, float(diagnostic["objective_decrease"])
        )
        if diagnostic["scipy_success"]:
            self.n_scipy_success += 1
        else:
            self.n_accepted_small_gradient += 1
            self.warnings.append({"context": context, "diagnostic": diagnostic})

    def summary(self) -> dict[str, Any]:
        return {
            "acceptance_rule": (
                f"SciPy success OR gradient_inf_norm <= {SOLVER_GRAD_TOL:g}; "
                f"objective_decrease >= {-OBJECTIVE_TOL:g}"
            ),
            "n_finite_calls": self.n_finite_calls,
            "n_native_exact_calls": self.n_native_exact_calls,
            "n_scipy_success": self.n_scipy_success,
            "n_accepted_small_gradient": self.n_accepted_small_gradient,
            "max_gradient_inf_norm": self.max_gradient_inf_norm,
            "min_objective_decrease": (
                self.min_objective_decrease if self.n_finite_calls else None
            ),
            "scipy_status_counts": dict(sorted(self.status_counts.items())),
            "n_unaccepted": 0,
        }


def residual_objective_and_gradient(
    theta: np.ndarray,
    features: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    lam: float,
) -> tuple[float, np.ndarray]:
    """Mean logistic residual objective and its exact analytic gradient."""
    features = np.asarray(features, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    dimension = features.shape[1]
    if theta.shape != (dimension + 1,):
        raise ValueError("theta has the wrong residual-head dimension")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("objective helper requires a positive finite lambda")
    weights, bias = theta[:dimension], float(theta[dimension])
    eta = offset + features @ weights + bias
    value = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    value += 0.5 * lam * float(weights @ weights + bias * bias)
    residual = (expit(eta) - y) / len(y)
    gradient = np.empty(dimension + 1, dtype=np.float64)
    gradient[:dimension] = features.T @ residual + lam * weights
    gradient[dimension] = float(residual.sum()) + lam * bias
    return value, gradient


def fit_residual(
    features: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    lam: float,
    *,
    ledger: SolverLedger,
    context: dict[str, Any],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Fit mean-loss ridge residual and assert an acceptable optimum."""
    features = np.asarray(features, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if features.ndim != 2 or offset.shape != (len(y),) or len(features) != len(y):
        raise ValueError("residual fit dimensions disagree")
    if not np.isfinite(features).all() or not np.isfinite(offset).all() or not np.isfinite(y).all():
        raise ValueError("residual fit inputs must be finite")
    dimension = features.shape[1]
    if not np.isfinite(lam):
        diagnostic = {
            "status": "native_exact",
            "lambda": "infinity",
            "scipy_success": True,
            "gradient_inf_norm": None,
            "objective_at_zero": float(
                np.mean(np.logaddexp(0.0, offset) - y * offset)
            ),
            "objective_at_fit": float(
                np.mean(np.logaddexp(0.0, offset) - y * offset)
            ),
            "objective_decrease": 0.0,
            "coefficient_l2": 0.0,
            "bias": 0.0,
        }
        ledger.record(diagnostic, context)
        return np.zeros(dimension, dtype=np.float64), 0.0, diagnostic
    if lam <= 0:
        raise ValueError("finite lambda must be positive")

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return residual_objective_and_gradient(theta, features, offset, y, lam)

    theta0 = np.zeros(dimension + 1, dtype=np.float64)
    objective0, _ = objective(theta0)
    result = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
    )
    objective_fit, gradient = objective(np.asarray(result.x, dtype=np.float64))
    if (
        not np.isfinite(np.asarray(result.x, dtype=float)).all()
        or not np.isfinite(objective_fit)
        or not np.isfinite(gradient).all()
    ):
        raise RuntimeError("residual solver returned non-finite solution diagnostics")
    gradient_inf = float(np.max(np.abs(gradient)))
    objective_decrease = float(objective0 - objective_fit)
    acceptable = bool(result.success) or gradient_inf <= SOLVER_GRAD_TOL
    if not acceptable:
        raise RuntimeError(
            f"unaccepted residual solver result: {result.message}; |g|inf={gradient_inf:.3e}"
        )
    if objective_decrease < -OBJECTIVE_TOL:
        raise RuntimeError(
            f"residual fit increased objective by {-objective_decrease:.3e}"
        )
    weights = np.asarray(result.x[:dimension], dtype=np.float64)
    bias = float(result.x[dimension])
    diagnostic = {
        "status": "finite_optimum",
        "lambda": float(lam),
        "scipy_success": bool(result.success),
        "scipy_status": int(result.status),
        "scipy_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "gradient_inf_norm": gradient_inf,
        "objective_at_zero": float(objective0),
        "objective_at_fit": float(objective_fit),
        "objective_decrease": objective_decrease,
        "coefficient_l2": float(np.linalg.norm(weights)),
        "bias": bias,
        "accepted_by": "scipy_success" if result.success else "small_gradient",
    }
    ledger.record(diagnostic, context)
    return weights, bias, diagnostic


def inner_splits(labels: np.ndarray, seed: int) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    n = len(labels)
    if n <= LOO_POOL_MAX:
        splits = [(np.arange(n) != index, np.arange(n) == index) for index in range(n)]
        return "leave_one_out", splits
    folds = stratified_folds(labels, seed)
    return "five_fold", [
        (folds != fold, folds == fold) for fold in range(N_FOLDS)
    ]


def select_lambda(
    features: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    *,
    inner_seed: int,
    ledger: SolverLedger,
    context: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    scheme, splits = inner_splits(y, inner_seed)
    losses: dict[str, list[float]] = {lambda_key(lam): [] for lam in LAMBDA_GRID}
    solver_by_lambda: dict[str, dict[str, Any]] = {
        lambda_key(lam): {
            "n_fits": 0,
            "n_native_exact": 0,
            "n_scipy_success": 0,
            "n_accepted_small_gradient": 0,
            "max_gradient_inf_norm": None,
            "min_objective_decrease": None,
        }
        for lam in LAMBDA_GRID
    }
    for split_index, (train, test) in enumerate(splits):
        if len(np.unique(y[train])) < 2:
            continue
        for lam in LAMBDA_GRID:
            fit_context = {
                **context,
                "fit_role": "inner_selection",
                "inner_split": split_index,
                "lambda": lambda_key(lam),
            }
            weights, bias, diagnostic = fit_residual(
                features[train], offset[train], y[train], lam,
                ledger=ledger, context=fit_context,
            )
            summary = solver_by_lambda[lambda_key(lam)]
            summary["n_fits"] += 1
            if diagnostic["status"] == "native_exact":
                summary["n_native_exact"] += 1
            else:
                if diagnostic["scipy_success"]:
                    summary["n_scipy_success"] += 1
                else:
                    summary["n_accepted_small_gradient"] += 1
                gradient = float(diagnostic["gradient_inf_norm"])
                decrease = float(diagnostic["objective_decrease"])
                summary["max_gradient_inf_norm"] = (
                    gradient
                    if summary["max_gradient_inf_norm"] is None
                    else max(float(summary["max_gradient_inf_norm"]), gradient)
                )
                summary["min_objective_decrease"] = (
                    decrease
                    if summary["min_objective_decrease"] is None
                    else min(float(summary["min_objective_decrease"]), decrease)
                )
            eta = offset[test] + features[test] @ weights + bias
            probability = np.clip(expit(eta), PROB_EPS, 1 - PROB_EPS)
            losses[lambda_key(lam)].append(
                float(log_loss(y[test], probability, labels=[0, 1]))
            )
    means = {
        key: (float(np.mean(values)) if values else float("inf"))
        for key, values in losses.items()
    }
    best = min(means.values())
    selected = LAMBDA_GRID[0]
    for lam in LAMBDA_GRID:
        if means[lambda_key(lam)] <= best + 1e-12:
            selected = lam
            break
    return selected, {
        "scheme": scheme,
        "inner_seed": inner_seed,
        "n_pool": int(len(y)),
        "n_splits": len(splits),
        "losses_by_lambda": losses,
        "mean_loss_by_lambda": means,
        "solver_summary_by_lambda": solver_by_lambda,
        "selected_lambda": lambda_key(selected),
    }


def fit_platt(
    native_eta: np.ndarray,
    y: np.ndarray,
    *,
    ledger: SolverLedger,
    context: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    """Unpenalized two-parameter logistic calibration on an outer train pool."""
    native_eta = np.asarray(native_eta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, slope = map(float, theta)
        eta = intercept + slope * native_eta
        probability = expit(eta)
        value = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
        residual = probability - y
        gradient = np.array(
            [float(np.mean(residual)), float(np.mean(residual * native_eta))],
            dtype=np.float64,
        )
        return value, gradient

    theta0 = np.array([0.0, 1.0], dtype=np.float64)
    objective0, _ = objective(theta0)
    result = minimize(
        objective,
        theta0,
        jac=True,
        method="BFGS",
        options={"maxiter": 1000, "gtol": 1e-10},
    )
    objective_fit, gradient = objective(np.asarray(result.x, dtype=np.float64))
    if (
        not np.isfinite(np.asarray(result.x, dtype=float)).all()
        or not np.isfinite(objective_fit)
        or not np.isfinite(gradient).all()
    ):
        raise RuntimeError("Platt solver returned non-finite solution diagnostics")
    gradient_inf = float(np.max(np.abs(gradient)))
    objective_decrease = float(objective0 - objective_fit)
    acceptable = bool(result.success) or gradient_inf <= SOLVER_GRAD_TOL
    if not acceptable:
        raise RuntimeError(
            f"unaccepted Platt solver result: {result.message}; |g|inf={gradient_inf:.3e}"
        )
    if objective_decrease < -OBJECTIVE_TOL:
        raise RuntimeError(f"Platt fit increased objective by {-objective_decrease:.3e}")
    intercept, slope = map(float, result.x)
    diagnostic = {
        "status": "finite_optimum",
        "lambda": 0.0,
        "scipy_success": bool(result.success),
        "scipy_status": int(result.status),
        "scipy_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "gradient_inf_norm": gradient_inf,
        "objective_at_zero": float(objective0),
        "objective_at_fit": float(objective_fit),
        "objective_decrease": objective_decrease,
        "coefficient_l2": float(np.linalg.norm(result.x)),
        "bias": intercept,
        "accepted_by": "scipy_success" if result.success else "small_gradient",
    }
    ledger.record(diagnostic, {**context, "fit_role": "platt_outer_refit"})
    return intercept, slope, diagnostic


def point_metrics(labels: np.ndarray, eta: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    eta = np.asarray(eta, dtype=float)
    probability = np.clip(expit(eta), PROB_EPS, 1 - PROB_EPS)
    return {
        "auroc": float(roc_auc_score(labels, eta)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "brier": float(np.mean((probability - labels) ** 2)),
    }


def derive_incremental_improvement(
    macro_delta_ci: list[float], cohort_delta_points: list[float]
) -> dict[str, bool]:
    """Fail-closed semantic claim: interval and both cohort directions must agree."""
    verdict = {
        "macro_auroc_delta_ci_lower_above_zero": float(macro_delta_ci[0]) > 0,
        "both_cohort_auroc_delta_points_above_zero": all(
            float(value) > 0 for value in cohort_delta_points
        ),
    }
    verdict["pass"] = all(verdict.values())
    return verdict


def make_fit_record(
    *,
    phase: str,
    model_kind: str,
    cohort: str,
    outer_seed: int,
    outer_fold: int,
    model_seed: int | None,
    fit_ids: list[str],
    fit_labels: list[int],
    test_ids: list[str],
    coefficients: np.ndarray,
    bias: float,
    diagnostic: dict[str, Any],
    selection: dict[str, Any] | None,
    support: int | None = None,
    draw: int | None = None,
    support_seed: int | None = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "model_kind": model_kind,
        "cohort": cohort,
        "outer_seed": outer_seed,
        "outer_fold": outer_fold,
        "model_seed": model_seed,
        "support_requested": support,
        "support_realized": len(fit_ids) if support is not None else None,
        "draw": draw,
        "support_seed": support_seed,
        "fit_patient_ids": fit_ids,
        "fit_labels": fit_labels,
        "n_fit": len(fit_ids),
        "n_fit_class0": int(fit_labels.count(0)),
        "n_fit_class1": int(fit_labels.count(1)),
        "test_patient_ids": test_ids,
        "n_test": len(test_ids),
        "selected_lambda": selection["selected_lambda"] if selection else None,
        "inner_selection": selection,
        "coefficient_order": (
            "e0..e511" if model_kind == "residual_adapter" else ["intercept", "slope"]
        ),
        "coefficients": coefficients.tolist(),
        "bias": bias,
        "solver_diagnostic": diagnostic,
    }


def run_full_layout(
    data: dict[str, dict[int, pd.DataFrame]],
    *,
    outer_seed: int,
    ledger: SolverLedger,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    blocks: dict[str, pd.DataFrame] = {}
    oof_rows: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    for cohort, seed_frames in data.items():
        reference = seed_frames[SEEDS[0]]
        patient_ids = [str(item) for item in reference.index.tolist()]
        labels = reference["label"].to_numpy(dtype=int)
        folds = stratified_folds(labels, outer_seed)
        ecols = [f"e{i}" for i in range(512)]
        native_by_seed: dict[int, np.ndarray] = {}
        adapted_by_seed: dict[int, np.ndarray] = {}
        for model_seed in SEEDS:
            frame = seed_frames[model_seed]
            features = frame[ecols].to_numpy(dtype=float)
            native = frame["eta_native"].to_numpy(dtype=float)
            native_by_seed[model_seed] = native
            adapted = native.copy()
            for fold in range(N_FOLDS):
                train = np.flatnonzero(folds != fold)
                test = np.flatnonzero(folds == fold)
                context = {
                    "phase": "full_label",
                    "cohort": cohort,
                    "outer_seed": outer_seed,
                    "outer_fold": fold,
                    "model_seed": model_seed,
                }
                selected, selection = select_lambda(
                    features[train], native[train], labels[train],
                    inner_seed=outer_seed + fold,
                    ledger=ledger,
                    context=context,
                )
                weights, bias, diagnostic = fit_residual(
                    features[train], native[train], labels[train], selected,
                    ledger=ledger,
                    context={**context, "fit_role": "outer_refit"},
                )
                adapted[test] = native[test] + features[test] @ weights + bias
                fit_records.append(
                    make_fit_record(
                        phase="full_label",
                        model_kind="residual_adapter",
                        cohort=cohort,
                        outer_seed=outer_seed,
                        outer_fold=fold,
                        model_seed=model_seed,
                        fit_ids=[patient_ids[index] for index in train],
                        fit_labels=labels[train].astype(int).tolist(),
                        test_ids=[patient_ids[index] for index in test],
                        coefficients=weights,
                        bias=bias,
                        diagnostic=diagnostic,
                        selection=selection,
                    )
                )
            adapted_by_seed[model_seed] = adapted

        native_ensemble = np.mean(
            np.vstack([native_by_seed[seed] for seed in SEEDS]), axis=0
        )
        adapted_ensemble = np.mean(
            np.vstack([adapted_by_seed[seed] for seed in SEEDS]), axis=0
        )
        platt = np.empty_like(native_ensemble)
        for fold in range(N_FOLDS):
            train = np.flatnonzero(folds != fold)
            test = np.flatnonzero(folds == fold)
            context = {
                "phase": "full_label",
                "cohort": cohort,
                "outer_seed": outer_seed,
                "outer_fold": fold,
                "model_seed": None,
            }
            intercept, slope, diagnostic = fit_platt(
                native_ensemble[train], labels[train], ledger=ledger, context=context
            )
            platt[test] = intercept + slope * native_ensemble[test]
            fit_records.append(
                make_fit_record(
                    phase="full_label",
                    model_kind="platt",
                    cohort=cohort,
                    outer_seed=outer_seed,
                    outer_fold=fold,
                    model_seed=None,
                    fit_ids=[patient_ids[index] for index in train],
                    fit_labels=labels[train].astype(int).tolist(),
                    test_ids=[patient_ids[index] for index in test],
                    coefficients=np.array([intercept, slope], dtype=float),
                    bias=intercept,
                    diagnostic=diagnostic,
                    selection=None,
                )
            )

        block = pd.DataFrame(
            {
                "patient_id": patient_ids,
                "label": labels,
                "fold": folds,
                "eta_native": native_ensemble,
                "eta_adapted": adapted_ensemble,
                "eta_platt": platt,
                **{
                    f"eta_native_seed{seed}": native_by_seed[seed] for seed in SEEDS
                },
                **{
                    f"eta_adapted_seed{seed}": adapted_by_seed[seed] for seed in SEEDS
                },
            }
        )
        blocks[cohort] = block
        for row in block.to_dict(orient="records"):
            oof_rows.append(
                {
                    "phase": "full_label",
                    "outer_seed": outer_seed,
                    "is_primary": outer_seed == PRIMARY_OUTER_SEED,
                    "support_requested": None,
                    "draw": None,
                    "cohort": cohort,
                    **row,
                }
            )
    return blocks, oof_rows, fit_records


def bootstrap_layout(
    blocks: dict[str, pd.DataFrame], *, bootstrap_seed: int
) -> dict[str, Any]:
    """Patient bootstrap, paired within cohort and equal-weighted across cohorts."""
    procedures = ("native", "adapted", "platt")
    metrics = ("auroc", "log_loss", "brier")
    cohorts = sorted(blocks)
    point: dict[str, Any] = {"per_cohort": {}}
    for cohort in cohorts:
        block = blocks[cohort]
        y = block["label"].to_numpy(dtype=int)
        point["per_cohort"][cohort] = {
            "n": int(len(block)),
            "n_mutant": int(y.sum()),
            "procedures": {
                procedure: point_metrics(y, block[f"eta_{procedure}"].to_numpy(dtype=float))
                for procedure in procedures
            },
        }
    point["macro"] = {
        procedure: {
            metric: float(
                np.mean(
                    [
                        point["per_cohort"][cohort]["procedures"][procedure][metric]
                        for cohort in cohorts
                    ]
                )
            )
            for metric in metrics
        }
        for procedure in procedures
    }

    draws: dict[str, dict[str, list[float]]] = {
        "macro": {
            f"{procedure}:{metric}": []
            for procedure in procedures
            for metric in metrics
        }
    }
    for cohort in cohorts:
        draws[cohort] = {
            f"{procedure}:{metric}": []
            for procedure in procedures
            for metric in metrics
        }
    for procedure in ("adapted", "platt"):
        for metric in metrics:
            draws["macro"][f"{procedure}_minus_native:{metric}"] = []
    for cohort in cohorts:
        for procedure in ("adapted", "platt"):
            for metric in metrics:
                draws[cohort][f"{procedure}_minus_native:{metric}"] = []

    rng = np.random.default_rng(bootstrap_seed)
    valid = 0
    for _ in range(BOOTSTRAP_DRAWS):
        current: dict[str, dict[str, dict[str, float]]] = {}
        okay = True
        for cohort in cohorts:
            block = blocks[cohort]
            index = rng.integers(0, len(block), len(block))
            y = block["label"].to_numpy(dtype=int)[index]
            if len(np.unique(y)) < 2:
                okay = False
                break
            current[cohort] = {}
            for procedure in procedures:
                eta = block[f"eta_{procedure}"].to_numpy(dtype=float)[index]
                current[cohort][procedure] = point_metrics(y, eta)
        if not okay:
            continue
        valid += 1
        for cohort in cohorts:
            for procedure in procedures:
                for metric in metrics:
                    draws[cohort][f"{procedure}:{metric}"].append(
                        current[cohort][procedure][metric]
                    )
            for procedure in ("adapted", "platt"):
                for metric in metrics:
                    draws[cohort][f"{procedure}_minus_native:{metric}"].append(
                        current[cohort][procedure][metric]
                        - current[cohort]["native"][metric]
                    )
        for procedure in procedures:
            for metric in metrics:
                draws["macro"][f"{procedure}:{metric}"].append(
                    float(np.mean([current[c][procedure][metric] for c in cohorts]))
                )
        for procedure in ("adapted", "platt"):
            for metric in metrics:
                draws["macro"][f"{procedure}_minus_native:{metric}"].append(
                    float(
                        np.mean(
                            [
                                current[c][procedure][metric]
                                - current[c]["native"][metric]
                                for c in cohorts
                            ]
                        )
                    )
                )

    def interval(values: list[float]) -> list[float]:
        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]

    out: dict[str, Any] = {
        "bootstrap_draws_requested": BOOTSTRAP_DRAWS,
        "bootstrap_draws_valid": valid,
        "bootstrap_seed": bootstrap_seed,
        "per_cohort": {},
        "macro": {"procedures": {}, "contrasts": {}},
    }
    for cohort in cohorts:
        out["per_cohort"][cohort] = {
            "n": point["per_cohort"][cohort]["n"],
            "n_mutant": point["per_cohort"][cohort]["n_mutant"],
            "procedures": {},
            "contrasts": {},
        }
        for procedure in procedures:
            out["per_cohort"][cohort]["procedures"][procedure] = {
                metric: {
                    "point": point["per_cohort"][cohort]["procedures"][procedure][metric],
                    "ci95": interval(draws[cohort][f"{procedure}:{metric}"]),
                }
                for metric in metrics
            }
        for procedure in ("adapted", "platt"):
            key = f"{procedure}_minus_native"
            out["per_cohort"][cohort]["contrasts"][key] = {
                metric: {
                    "point": (
                        point["per_cohort"][cohort]["procedures"][procedure][metric]
                        - point["per_cohort"][cohort]["procedures"]["native"][metric]
                    ),
                    "ci95": interval(draws[cohort][f"{key}:{metric}"]),
                }
                for metric in metrics
            }
    for procedure in procedures:
        out["macro"]["procedures"][procedure] = {
            metric: {
                "point": point["macro"][procedure][metric],
                "ci95": interval(draws["macro"][f"{procedure}:{metric}"]),
            }
            for metric in metrics
        }
    for procedure in ("adapted", "platt"):
        key = f"{procedure}_minus_native"
        out["macro"]["contrasts"][key] = {
            metric: {
                "point": point["macro"][procedure][metric]
                - point["macro"]["native"][metric],
                "ci95": interval(draws["macro"][f"{key}:{metric}"]),
            }
            for metric in metrics
        }
    out["contrast_sign_convention"] = (
        "procedure minus native for every metric: positive favors the procedure for AUROC; "
        "negative favors the procedure for log-loss and Brier"
    )
    adapted_macro = out["macro"]["procedures"]["adapted"]["auroc"]
    out["fixed_gate_adapted"] = {
        "both_cohort_points_above_0p5": all(
            out["per_cohort"][cohort]["procedures"]["adapted"]["auroc"]["point"] > 0.5
            for cohort in cohorts
        ),
        "macro_ci_lower_above_0p5": adapted_macro["ci95"][0] > 0.5,
    }
    out["fixed_gate_adapted"]["pass"] = all(out["fixed_gate_adapted"].values())
    out["incremental_improvement_established"] = derive_incremental_improvement(
        out["macro"]["contrasts"]["adapted_minus_native"]["auroc"]["ci95"],
        [
            out["per_cohort"][cohort]["contrasts"]["adapted_minus_native"]["auroc"]["point"]
            for cohort in cohorts
        ],
    )
    return out


def draw_exact_support(
    labels: np.ndarray,
    train: np.ndarray,
    requested: int,
    *,
    seed: int,
) -> np.ndarray:
    if requested % 2:
        raise ValueError("support must be even")
    per_class = requested // 2
    rng = np.random.default_rng(seed)
    picks: list[np.ndarray] = []
    for value in (0, 1):
        candidates = train[labels[train] == value]
        if len(candidates) < per_class:
            raise RuntimeError(
                f"exact support {requested} infeasible: class {value} has {len(candidates)}"
            )
        picks.append(rng.choice(candidates, per_class, replace=False))
    support = np.concatenate(picks)
    if (
        len(support) != requested
        or len(np.unique(support)) != requested
        or int((labels[support] == 0).sum()) != per_class
        or int((labels[support] == 1).sum()) != per_class
    ):
        raise RuntimeError("exact balanced support contract failed")
    return support


def run_support_curve(
    data: dict[str, dict[int, pd.DataFrame]],
    *,
    ledger: SolverLedger,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: dict[str, Any] = {}
    oof_rows: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    draw_records: list[dict[str, Any]] = []
    for requested in SUPPORT_SIZES:
        per_support_draws: list[dict[str, Any]] = []
        for draw in range(SUPPORT_DRAWS):
            cohort_metrics: dict[str, Any] = {}
            for cohort, seed_frames in data.items():
                reference = seed_frames[SEEDS[0]]
                patient_ids = [str(item) for item in reference.index.tolist()]
                labels = reference["label"].to_numpy(dtype=int)
                folds = stratified_folds(labels, PRIMARY_OUTER_SEED)
                ecols = [f"e{i}" for i in range(512)]
                native_by_seed: dict[int, np.ndarray] = {}
                adapted_by_seed: dict[int, np.ndarray] = {}
                supports: dict[int, tuple[np.ndarray, int]] = {}
                for fold in range(N_FOLDS):
                    train = np.flatnonzero(folds != fold)
                    seed = stable_seed(
                        "e2f_v3_support", PRIMARY_OUTER_SEED, cohort, requested, draw, fold
                    )
                    support = draw_exact_support(labels, train, requested, seed=seed)
                    test = np.flatnonzero(folds == fold)
                    if set(support.tolist()) & set(test.tolist()):
                        raise RuntimeError("support/test leakage")
                    supports[fold] = (support, seed)

                for model_seed in SEEDS:
                    frame = seed_frames[model_seed]
                    features = frame[ecols].to_numpy(dtype=float)
                    native = frame["eta_native"].to_numpy(dtype=float)
                    native_by_seed[model_seed] = native
                    adapted = native.copy()
                    for fold in range(N_FOLDS):
                        support, support_seed = supports[fold]
                        test = np.flatnonzero(folds == fold)
                        context = {
                            "phase": "support_curve",
                            "cohort": cohort,
                            "outer_seed": PRIMARY_OUTER_SEED,
                            "outer_fold": fold,
                            "model_seed": model_seed,
                            "support": requested,
                            "draw": draw,
                        }
                        selected, selection = select_lambda(
                            features[support], native[support], labels[support],
                            inner_seed=PRIMARY_OUTER_SEED + fold,
                            ledger=ledger,
                            context=context,
                        )
                        weights, bias, diagnostic = fit_residual(
                            features[support], native[support], labels[support], selected,
                            ledger=ledger,
                            context={**context, "fit_role": "outer_refit"},
                        )
                        adapted[test] = native[test] + features[test] @ weights + bias
                        fit_records.append(
                            make_fit_record(
                                phase="support_curve",
                                model_kind="residual_adapter",
                                cohort=cohort,
                                outer_seed=PRIMARY_OUTER_SEED,
                                outer_fold=fold,
                                model_seed=model_seed,
                                fit_ids=[patient_ids[index] for index in support],
                                fit_labels=labels[support].astype(int).tolist(),
                                test_ids=[patient_ids[index] for index in test],
                                coefficients=weights,
                                bias=bias,
                                diagnostic=diagnostic,
                                selection=selection,
                                support=requested,
                                draw=draw,
                                support_seed=support_seed,
                            )
                        )
                    adapted_by_seed[model_seed] = adapted

                native_ensemble = np.mean(
                    np.vstack([native_by_seed[seed] for seed in SEEDS]), axis=0
                )
                adapted_ensemble = np.mean(
                    np.vstack([adapted_by_seed[seed] for seed in SEEDS]), axis=0
                )
                native_metrics = point_metrics(labels, native_ensemble)
                adapted_metrics = point_metrics(labels, adapted_ensemble)
                cohort_metrics[cohort] = {
                    "native": native_metrics,
                    "adapted": adapted_metrics,
                    "delta_auroc": adapted_metrics["auroc"] - native_metrics["auroc"],
                    "realized_support_by_fold": {
                        str(fold): int(len(supports[fold][0])) for fold in range(N_FOLDS)
                    },
                    "class_counts_by_fold": {
                        str(fold): {
                            "class0": int((labels[supports[fold][0]] == 0).sum()),
                            "class1": int((labels[supports[fold][0]] == 1).sum()),
                        }
                        for fold in range(N_FOLDS)
                    },
                }
                for index, patient_id in enumerate(patient_ids):
                    oof_rows.append(
                        {
                            "phase": "support_curve",
                            "outer_seed": PRIMARY_OUTER_SEED,
                            "is_primary": False,
                            "support_requested": requested,
                            "draw": draw,
                            "cohort": cohort,
                            "patient_id": patient_id,
                            "label": int(labels[index]),
                            "fold": int(folds[index]),
                            "eta_native": float(native_ensemble[index]),
                            "eta_adapted": float(adapted_ensemble[index]),
                            "eta_platt": None,
                            **{
                                f"eta_native_seed{seed}": float(native_by_seed[seed][index])
                                for seed in SEEDS
                            },
                            **{
                                f"eta_adapted_seed{seed}": float(adapted_by_seed[seed][index])
                                for seed in SEEDS
                            },
                        }
                    )
            macro_delta = float(
                np.mean([cohort_metrics[c]["delta_auroc"] for c in COHORTS])
            )
            record = {
                "support_requested": requested,
                "draw": draw,
                "cohorts": cohort_metrics,
                "macro_delta_auroc": macro_delta,
            }
            per_support_draws.append(record)
            draw_records.append(record)

        macro_values = [row["macro_delta_auroc"] for row in per_support_draws]
        summaries[str(requested)] = {
            "requested_support": requested,
            "realized_support_min": requested,
            "realized_support_max": requested,
            "exact_balanced_contract": True,
            "n_draws": SUPPORT_DRAWS,
            "macro_delta_auroc_mean": float(np.mean(macro_values)),
            "macro_delta_auroc_sd": float(np.std(macro_values, ddof=1)),
            "per_cohort_delta_auroc_mean": {
                cohort: float(
                    np.mean([row["cohorts"][cohort]["delta_auroc"] for row in per_support_draws])
                )
                for cohort in COHORTS
            },
            "per_cohort_log_loss_mean": {
                cohort: {
                    procedure: float(
                        np.mean(
                            [
                                row["cohorts"][cohort][procedure]["log_loss"]
                                for row in per_support_draws
                            ]
                        )
                    )
                    for procedure in ("native", "adapted")
                }
                for cohort in COHORTS
            },
            "per_cohort_brier_mean": {
                cohort: {
                    procedure: float(
                        np.mean(
                            [
                                row["cohorts"][cohort][procedure]["brier"]
                                for row in per_support_draws
                            ]
                        )
                    )
                    for procedure in ("native", "adapted")
                }
                for cohort in COHORTS
            },
        }
    return summaries, oof_rows, fit_records, draw_records


def lambda_inventory(
    fit_records: list[dict[str, Any]], *, phase: str, outer_seed: int | None = None
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    records = [
        row
        for row in fit_records
        if row["phase"] == phase and row["model_kind"] == "residual_adapter"
        and (outer_seed is None or row["outer_seed"] == outer_seed)
    ]
    for cohort in COHORTS:
        selected = [row["selected_lambda"] for row in records if row["cohort"] == cohort]
        output[cohort] = {
            "n": len(selected),
            "counts": dict(sorted(Counter(selected).items())),
            "fraction_infinity": float(np.mean([item == "infinity" for item in selected])),
            "fraction_at_lower_boundary_0p001": float(
                np.mean([item == "0.001" for item in selected])
            ),
        }
    return output


def assert_full_label_lambda_grid_closed(fit_records: list[dict[str, Any]]) -> None:
    """Fail if any full-label layout selects the newly expanded lower boundary."""
    lower_boundary = lambda_key(LAMBDA_GRID[-1])
    hits = [
        row
        for row in fit_records
        if row.get("phase") == "full_label"
        and row.get("model_kind") == "residual_adapter"
        and row.get("selected_lambda") == lower_boundary
    ]
    if hits:
        raise RuntimeError(
            f"full-label lambda search remains truncated: {len(hits)} fits selected "
            f"lower boundary {lower_boundary}"
        )


def cmd_run(_: argparse.Namespace) -> None:
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(
            f"append-only destination already exists; refusing any write: {OUTPUT_ROOT}"
        )
    source_start = artifact_identity(SCRIPT)
    repository_state = git_state()
    input_identities, data = validate_inputs()

    ledger = SolverLedger()
    all_oof_rows: list[dict[str, Any]] = []
    all_fit_records: list[dict[str, Any]] = []
    layout_blocks: dict[int, dict[str, pd.DataFrame]] = {}
    print("Running full-label outer-fold layouts", flush=True)
    for outer_seed in OUTER_SENSITIVITY_SEEDS:
        blocks, oof_rows, fit_records = run_full_layout(
            data, outer_seed=outer_seed, ledger=ledger
        )
        layout_blocks[outer_seed] = blocks
        all_oof_rows.extend(oof_rows)
        all_fit_records.extend(fit_records)
        macro_delta = np.mean(
            [
                point_metrics(block["label"], block["eta_adapted"])["auroc"]
                - point_metrics(block["label"], block["eta_native"])["auroc"]
                for block in blocks.values()
            ]
        )
        print(f"  outer seed {outer_seed}: macro delta {macro_delta:+.6f}", flush=True)

    print("Running exact support curve", flush=True)
    support_summary, support_oof, support_fits, support_draws = run_support_curve(
        data, ledger=ledger
    )
    all_oof_rows.extend(support_oof)
    all_fit_records.extend(support_fits)

    assert_full_label_lambda_grid_closed(all_fit_records)

    print("Bootstrapping primary and fold-layout sensitivity", flush=True)
    layout_results: dict[str, Any] = {}
    for outer_seed in OUTER_SENSITIVITY_SEEDS:
        layout_results[str(outer_seed)] = bootstrap_layout(
            layout_blocks[outer_seed], bootstrap_seed=BOOTSTRAP_SEED
        )
        gate = layout_results[str(outer_seed)]["fixed_gate_adapted"]["pass"]
        lower = layout_results[str(outer_seed)]["macro"]["procedures"]["adapted"]["auroc"]["ci95"][0]
        print(
            f"  outer seed {outer_seed}: gate={gate}, macro adapted lower={lower:.6f}",
            flush=True,
        )

    primary = layout_results[str(PRIMARY_OUTER_SEED)]
    for cohort, expected in E2B_EXPECTED.items():
        observed = primary["per_cohort"][cohort]["procedures"]["native"]["auroc"]["point"]
        if abs(observed - expected) >= 5e-7:
            raise RuntimeError(f"{cohort}: failed frozen E2b reproduction")

    primary_delta = primary["macro"]["contrasts"]["adapted_minus_native"]["auroc"]
    primary_gate = primary["fixed_gate_adapted"]["pass"]
    interpretation = (
        "In target-label internal cross-fitting, the expanded-grid full-label residual "
        f"adapter {'passed' if primary_gate else 'did not pass'} the fixed metastatic "
        "performance gate. It did not establish consistent cross-cohort improvement "
        "over native unless the paired macro delta interval excludes zero and cohort "
        "directions agree. This is adaptation-feasibility evidence, not independent "
        "deployment validation; no 'not repairable' or clinical-readiness claim is licensed."
    )

    oof = pd.DataFrame(all_oof_rows)
    full_oof = oof[oof["phase"] == "full_label"]
    support_oof_frame = oof[oof["phase"] == "support_curve"]
    expected_full_rows = len(OUTER_SENSITIVITY_SEEDS) * sum(
        len(data[c][SEEDS[0]]) for c in COHORTS
    )
    expected_support_rows = len(SUPPORT_SIZES) * SUPPORT_DRAWS * sum(
        len(data[c][SEEDS[0]]) for c in COHORTS
    )
    if len(full_oof) != expected_full_rows or len(support_oof_frame) != expected_support_rows:
        raise RuntimeError("OOF artifact census failed")
    if full_oof.duplicated(["outer_seed", "cohort", "patient_id"]).any():
        raise RuntimeError("duplicate full-label OOF patient")
    if support_oof_frame.duplicated(
        ["support_requested", "draw", "cohort", "patient_id"]
    ).any():
        raise RuntimeError("duplicate support-curve OOF patient")

    expected_full_adapter_fits = len(OUTER_SENSITIVITY_SEEDS) * len(COHORTS) * len(SEEDS) * N_FOLDS
    expected_platt_fits = len(OUTER_SENSITIVITY_SEEDS) * len(COHORTS) * N_FOLDS
    expected_support_fits = len(SUPPORT_SIZES) * SUPPORT_DRAWS * len(COHORTS) * len(SEEDS) * N_FOLDS
    fit_counter = Counter((row["phase"], row["model_kind"]) for row in all_fit_records)
    if fit_counter[("full_label", "residual_adapter")] != expected_full_adapter_fits:
        raise RuntimeError("full-label adapter-fit census failed")
    if fit_counter[("full_label", "platt")] != expected_platt_fits:
        raise RuntimeError("Platt-fit census failed")
    if fit_counter[("support_curve", "residual_adapter")] != expected_support_fits:
        raise RuntimeError("support-fit census failed")

    created = utc_now()
    results = {
        "schema_version": 1,
        "experiment": "e2f_v3_full_label_residual_adapter",
        "created_utc": created,
        "status": "PASS",
        "problems": [],
        "scope_warning": SCOPE_WARNING,
        "design": {
            "estimand": "eta_adapted = eta_native + H @ delta_w + delta_b",
            "residual_parameters": 513,
            "objective": "MEAN logistic loss + lambda/2*(||delta_w||^2 + delta_b^2)",
            "lambda_grid": [lambda_key(value) for value in LAMBDA_GRID],
            "lambda_selection": (
                "training pool only: leave-one-out for n<=20, otherwise inner five-fold; "
                "ties prefer earlier/stronger grid entries, beginning with exact native infinity"
            ),
            "primary_outer_seed": PRIMARY_OUTER_SEED,
            "outer_sensitivity_seeds": list(OUTER_SENSITIVITY_SEEDS),
            "outer_folds": N_FOLDS,
            "model_seeds": list(SEEDS),
            "cohorts_fit_separately": True,
            "platt_comparator": (
                "two-parameter intercept+slope calibration of ensemble native logit, "
                "fit on each outer training pool and predicted on its held-out fold"
            ),
            "support_sizes_exact_total": list(SUPPORT_SIZES),
            "support_balance": "exactly half class 0 and half class 1 in every outer fold",
            "support_draws": SUPPORT_DRAWS,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "uncertainty": (
                "paired within-cohort patient bootstrap, equal-cohort macro; primary interval "
                "conditions on one fold layout, with five declared layouts reported separately"
            ),
            "capacity_wording": (
                "513 raw coefficients per cohort-specific seed/fold head; ridge regularization "
                "controls effective complexity. No pooled patients-per-parameter claim is made."
            ),
        },
        "input_census": {
            cohort: input_identities["cohorts"][cohort]["census"] for cohort in COHORTS
        },
        "frozen_baseline_checks": {
            cohort: {
                "observed": primary["per_cohort"][cohort]["procedures"]["native"]["auroc"]["point"],
                "expected": E2B_EXPECTED[cohort],
                "pass": True,
            }
            for cohort in COHORTS
        },
        "primary": primary,
        "primary_macro_delta_auroc": primary_delta,
        "primary_incremental_improvement_established": primary[
            "incremental_improvement_established"
        ],
        "outer_fold_sensitivity": layout_results,
        "outer_fold_sensitivity_summary": {
            "n_layouts": len(OUTER_SENSITIVITY_SEEDS),
            "n_gate_pass": int(
                sum(row["fixed_gate_adapted"]["pass"] for row in layout_results.values())
            ),
            "macro_adapted_auroc_range": [
                float(
                    min(
                        row["macro"]["procedures"]["adapted"]["auroc"]["point"]
                        for row in layout_results.values()
                    )
                ),
                float(
                    max(
                        row["macro"]["procedures"]["adapted"]["auroc"]["point"]
                        for row in layout_results.values()
                    )
                ),
            ],
            "macro_delta_auroc_range": [
                float(
                    min(
                        row["macro"]["contrasts"]["adapted_minus_native"]["auroc"]["point"]
                        for row in layout_results.values()
                    )
                ),
                float(
                    max(
                        row["macro"]["contrasts"]["adapted_minus_native"]["auroc"]["point"]
                        for row in layout_results.values()
                    )
                ),
            ],
        },
        "label_efficiency_curve": support_summary,
        "lambda_inventory_primary": lambda_inventory(
            all_fit_records, phase="full_label", outer_seed=PRIMARY_OUTER_SEED
        ),
        "lambda_inventory_full_label_all_layouts": lambda_inventory(
            all_fit_records, phase="full_label"
        ),
        "lambda_inventory_support_curve": lambda_inventory(
            all_fit_records, phase="support_curve"
        ),
        "solver": ledger.summary(),
        "artifact_census": {
            "oof_rows_total": int(len(oof)),
            "oof_rows_full_label": int(len(full_oof)),
            "oof_rows_primary_full_label": int(
                len(full_oof[full_oof["outer_seed"] == PRIMARY_OUTER_SEED])
            ),
            "oof_rows_support_curve": int(len(support_oof_frame)),
            "fit_records_total": len(all_fit_records),
            "fit_records_full_label_adapter": expected_full_adapter_fits,
            "fit_records_full_label_platt": expected_platt_fits,
            "fit_records_support_adapter": expected_support_fits,
            "support_draw_records": len(support_draws),
            "solver_warning_records": len(ledger.warnings),
        },
        "interpretation": interpretation,
        "supersedes_for_authoritative_use": {
            "v1": (
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "aim2_e2f_fulllabel_adapter_v1_20260821"
            ),
            "v2": (
                "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
                "aim2_e2f_fulllabel_adapter_v2_20260821"
            ),
            "preservation": "v1 and v2 remain byte-preserved; v3 uses a new root",
            "identities_bound_in_receipt": True,
        },
    }

    if artifact_identity(SCRIPT)["sha256"] != source_start["sha256"]:
        raise RuntimeError("generating source changed during execution")

    analysis = OUTPUT_ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=False)
    paths = {
        "results": analysis / "results.json",
        "oof_predictions": analysis / "oof_predictions.csv",
        "fits": analysis / "fits.jsonl",
        "support_draws": analysis / "support_draws.jsonl",
        "solver_warnings": analysis / "solver_warnings.jsonl",
    }
    write_once(paths["results"], json_bytes(results))
    write_once(paths["oof_predictions"], oof.to_csv(index=False).encode("utf-8"))
    write_once(paths["fits"], jsonl_bytes(all_fit_records))
    write_once(paths["support_draws"], jsonl_bytes(support_draws))
    write_once(paths["solver_warnings"], jsonl_bytes(ledger.warnings))

    output_identities = {
        name: artifact_identity(path) for name, path in paths.items()
    }
    receipt = {
        "schema_version": 1,
        "experiment": "e2f_v3_full_label_residual_adapter",
        "created_utc": created,
        "status": "PASS",
        "problems": [],
        "scope_warning": SCOPE_WARNING,
        "append_only": {
            "output_root": str(OUTPUT_ROOT),
            "root_was_absent_at_start": True,
            "v1_v2_modified": False,
        },
        "generating_code": source_start,
        "git": repository_state,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "inputs": input_identities,
        "outputs": output_identities,
        "census": results["artifact_census"],
        "solver": results["solver"],
        "fixed_gate_primary": primary["fixed_gate_adapted"],
        "outer_fold_sensitivity_summary": results["outer_fold_sensitivity_summary"],
    }
    write_once(analysis / "receipt.json", json_bytes(receipt))
    print(f"Wrote sealed v3 lineage: {OUTPUT_ROOT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run").set_defaults(function=cmd_run)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
