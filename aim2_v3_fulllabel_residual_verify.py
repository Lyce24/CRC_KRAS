#!/usr/bin/env python3
"""Independent read-only verifier for the sealed E2f-v3 adapter lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import log_loss, roc_auc_score

DEFAULT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_e2f_fulllabel_adapter_v3_20260821"
)
EXPECTED_OUTER_SEEDS = tuple(range(20260817, 20260822))
EXPECTED_SUPPORTS = (8, 16, 32, 48)
EXPECTED_DRAWS = 20
EXPECTED_COHORT_N = {"RIH": 85, "SurGen": 74}
PRIMARY_SEED = 20260821


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_identity(identity: dict[str, Any], problems: list[str], label: str) -> None:
    try:
        path = Path(identity["path"]).resolve(strict=True)
        if not path.is_file():
            problems.append(f"{label}: not a file")
            return
        if path.stat().st_size != int(identity["size_bytes"]):
            problems.append(f"{label}: size mismatch")
        if sha256_file(path) != str(identity["sha256"]):
            problems.append(f"{label}: SHA256 mismatch")
    except Exception as exc:  # verifier must report all recoverable problems
        problems.append(f"{label}: {exc}")


def point_metrics(labels: np.ndarray, eta: np.ndarray) -> dict[str, float]:
    probability = np.clip(expit(eta), 1e-9, 1 - 1e-9)
    return {
        "auroc": float(roc_auc_score(labels, eta)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "brier": float(np.mean((probability - labels) ** 2)),
    }


def close(observed: float, expected: float, tolerance: float = 1e-12) -> bool:
    return abs(float(observed) - float(expected)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    analysis = root / "analysis"
    problems: list[str] = []

    receipt = json.loads((analysis / "receipt.json").read_text(encoding="utf-8"))
    results = json.loads((analysis / "results.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or receipt.get("problems") != []:
        problems.append("receipt is not PASS with an empty problems list")
    if results.get("status") != "PASS" or results.get("problems") != []:
        problems.append("results are not PASS with an empty problems list")
    if receipt.get("append_only", {}).get("output_root") != str(root):
        problems.append("receipt output root mismatch")

    verify_identity(receipt.get("generating_code", {}), problems, "generating code")
    verify_identity(
        receipt.get("inputs", {}).get("upstream_lineage_start", {}),
        problems,
        "upstream lineage start",
    )
    for version in ("v1", "v2"):
        block = receipt.get("inputs", {}).get("superseded_lineages", {}).get(version, {})
        for name in ("results", "receipt"):
            verify_identity(block.get(name, {}), problems, f"superseded {version} {name}")
    for cohort in EXPECTED_COHORT_N:
        block = receipt.get("inputs", {}).get("cohorts", {}).get(cohort, {})
        for name in ("embedding", "embedding_receipt", "manifest", "upstream_generator_code"):
            verify_identity(block.get(name, {}), problems, f"{cohort} {name}")
    for name, identity in receipt.get("outputs", {}).items():
        verify_identity(identity, problems, f"output {name}")

    oof = pd.read_csv(analysis / "oof_predictions.csv", low_memory=False)
    full = oof[oof["phase"] == "full_label"].copy()
    support = oof[oof["phase"] == "support_curve"].copy()
    expected_full = len(EXPECTED_OUTER_SEEDS) * sum(EXPECTED_COHORT_N.values())
    expected_support = len(EXPECTED_SUPPORTS) * EXPECTED_DRAWS * sum(
        EXPECTED_COHORT_N.values()
    )
    if len(full) != expected_full:
        problems.append(f"full OOF rows {len(full)} != {expected_full}")
    if len(support) != expected_support:
        problems.append(f"support OOF rows {len(support)} != {expected_support}")
    if full.duplicated(["outer_seed", "cohort", "patient_id"]).any():
        problems.append("duplicate full-label OOF patient key")
    if support.duplicated(["support_requested", "draw", "cohort", "patient_id"]).any():
        problems.append("duplicate support-curve OOF patient key")
    if set(full["outer_seed"].astype(int).unique()) != set(EXPECTED_OUTER_SEEDS):
        problems.append("outer sensitivity seed inventory mismatch")
    if set(support["support_requested"].astype(int).unique()) != set(EXPECTED_SUPPORTS):
        problems.append("support inventory mismatch")
    if set(support["draw"].astype(int).unique()) != set(range(EXPECTED_DRAWS)):
        problems.append("support draw inventory mismatch")
    for phase_name, frame, key_columns in (
        ("full", full, ["outer_seed", "cohort"]),
        ("support", support, ["support_requested", "draw", "cohort"]),
    ):
        for key, group in frame.groupby(key_columns):
            cohort = key[-1] if isinstance(key, tuple) else key
            if len(group) != EXPECTED_COHORT_N[str(cohort)]:
                problems.append(f"{phase_name} OOF group {key}: patient census mismatch")
            if set(group["fold"].astype(int).unique()) != set(range(5)):
                problems.append(f"{phase_name} OOF group {key}: fold coverage mismatch")

    fits: list[dict[str, Any]] = []
    with (analysis / "fits.jsonl").open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                fits.append(json.loads(line))
            except Exception as exc:
                problems.append(f"fits line {line_number}: {exc}")
    fit_counts = Counter((row.get("phase"), row.get("model_kind")) for row in fits)
    expected_counts = {
        ("full_label", "residual_adapter"): 5 * 2 * 3 * 5,
        ("full_label", "platt"): 5 * 2 * 5,
        ("support_curve", "residual_adapter"): 4 * 20 * 2 * 3 * 5,
    }
    if dict(fit_counts) != expected_counts:
        problems.append(f"fit census mismatch: {dict(fit_counts)}")

    lambda_grid = results.get("design", {}).get("lambda_grid", [])
    expected_grid = [
        "infinity", "10000", "3000", "1000", "300", "100", "30", "10",
        "3", "1", "0.3", "0.1", "0.03", "0.01", "0.003", "0.001",
    ]
    if lambda_grid != expected_grid:
        problems.append("expanded lambda grid mismatch")

    for index, row in enumerate(fits):
        fit_ids = list(map(str, row.get("fit_patient_ids", [])))
        test_ids = list(map(str, row.get("test_patient_ids", [])))
        if len(fit_ids) != len(set(fit_ids)):
            problems.append(f"fit {index}: duplicate fit patient")
        if set(fit_ids) & set(test_ids):
            problems.append(f"fit {index}: fit/test leakage")
        labels = list(map(int, row.get("fit_labels", [])))
        if len(labels) != len(fit_ids):
            problems.append(f"fit {index}: fit label census mismatch")
        diagnostic = row.get("solver_diagnostic", {})
        if diagnostic.get("status") == "finite_optimum":
            acceptable = bool(diagnostic.get("scipy_success")) or float(
                diagnostic.get("gradient_inf_norm", np.inf)
            ) <= 1e-6
            if not acceptable:
                problems.append(f"fit {index}: unaccepted solver result")
            if float(diagnostic.get("objective_decrease", -np.inf)) < -1e-9:
                problems.append(f"fit {index}: objective increased")
        if row.get("model_kind") == "residual_adapter":
            if len(row.get("coefficients", [])) != 512:
                problems.append(f"fit {index}: residual coefficient census mismatch")
            selection = row.get("inner_selection", {})
            losses = selection.get("losses_by_lambda", {})
            means = selection.get("mean_loss_by_lambda", {})
            if set(losses) != set(expected_grid) or set(means) != set(expected_grid):
                problems.append(f"fit {index}: inner lambda-loss inventory mismatch")
            if row.get("selected_lambda") not in expected_grid:
                problems.append(f"fit {index}: selected lambda outside grid")
        elif row.get("model_kind") == "platt":
            if len(row.get("coefficients", [])) != 2:
                problems.append(f"fit {index}: Platt coefficient census mismatch")
        if row.get("phase") == "support_curve":
            requested = int(row.get("support_requested", -1))
            realized = int(row.get("support_realized", -1))
            if requested not in EXPECTED_SUPPORTS or realized != requested:
                problems.append(f"fit {index}: exact support count failed")
            if labels.count(0) != requested // 2 or labels.count(1) != requested // 2:
                problems.append(f"fit {index}: exact support balance failed")

    draw_rows = [
        json.loads(line)
        for line in (analysis / "support_draws.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(draw_rows) != len(EXPECTED_SUPPORTS) * EXPECTED_DRAWS:
        problems.append("support draw-record census mismatch")
    warning_rows = [
        json.loads(line)
        for line in (analysis / "solver_warnings.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(warning_rows) != int(results["artifact_census"]["solver_warning_records"]):
        problems.append("solver warning-record census mismatch")

    primary = full[full["outer_seed"].astype(int) == PRIMARY_SEED]
    for cohort, group in primary.groupby("cohort"):
        y = group["label"].to_numpy(dtype=int)
        for procedure in ("native", "adapted", "platt"):
            observed = point_metrics(y, group[f"eta_{procedure}"].to_numpy(dtype=float))
            recorded = results["primary"]["per_cohort"][cohort]["procedures"][procedure]
            for metric, value in observed.items():
                if not close(value, recorded[metric]["point"]):
                    problems.append(
                        f"primary {cohort}/{procedure}/{metric}: metric mismatch"
                    )
    if int(results["artifact_census"]["oof_rows_total"]) != len(oof):
        problems.append("results OOF census mismatch")
    if int(results["artifact_census"]["fit_records_total"]) != len(fits):
        problems.append("results fit census mismatch")
    if receipt.get("census") != results.get("artifact_census"):
        problems.append("receipt/results census mismatch")
    primary_inventory = results.get("lambda_inventory_primary", {})
    all_layout_inventory = results.get("lambda_inventory_full_label_all_layouts", {})
    if any(
        int(primary_inventory.get(cohort, {}).get("counts", {}).get("0.001", 0))
        for cohort in EXPECTED_COHORT_N
    ):
        problems.append("primary full-label fit selected lower lambda boundary")
    if any(
        int(all_layout_inventory.get(cohort, {}).get("counts", {}).get("0.001", 0))
        for cohort in EXPECTED_COHORT_N
    ):
        problems.append("sensitivity full-label fit selected lower lambda boundary")
    derived = results.get("primary", {}).get("incremental_improvement_established", {})
    macro_lower = results["primary"]["macro"]["contrasts"]["adapted_minus_native"]["auroc"]["ci95"][0]
    cohort_positive = all(
        results["primary"]["per_cohort"][cohort]["contrasts"]["adapted_minus_native"]["auroc"]["point"] > 0
        for cohort in EXPECTED_COHORT_N
    )
    if derived.get("macro_auroc_delta_ci_lower_above_zero") != (macro_lower > 0):
        problems.append("incremental macro-delta flag is not mechanically derived")
    if derived.get("both_cohort_auroc_delta_points_above_zero") != cohort_positive:
        problems.append("incremental cohort-direction flag is not mechanically derived")
    if derived.get("pass") != ((macro_lower > 0) and cohort_positive):
        problems.append("incremental-improvement verdict is not mechanically derived")

    verdict = {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "root": str(root),
        "verified": {
            "outputs_hashed": len(receipt.get("outputs", {})),
            "oof_rows": len(oof),
            "fit_records": len(fits),
            "support_draw_records": len(draw_rows),
            "solver_warning_records": len(warning_rows),
        },
    }
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
