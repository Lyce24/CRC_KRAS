#!/usr/bin/env python3
"""Decision-curve (net-benefit) analysis for the v4 KRAS report.

WHAT IT ADDS. The v3 worklist result showed capacity-constrained enrichment;
decision-curve analysis is the complementary TRIPOD+AI-aligned utility
grammar: across a RANGE of hypothetical threshold probabilities, does using
the WSI score to flag patients for molecular-testing prioritization yield
higher net benefit than the two label-free defaults, prioritize-everyone and
prioritize-no-one? Reporting a threshold RANGE keeps faith with the study's
standing rule that no single clinical cutoff is validated.

DECISION MODEL. "Treatment" is prioritization of a case for (earlier)
confirmatory KRAS/RAS testing. At threshold probability pt,

    NB(pt) = TP/n - FP/n * pt / (1 - pt)

in units of net true positives per patient (multiplied by 100 in the report).
Prioritize-everyone has NB_all(pt) = prevalence - (1 - prevalence) * pt/(1-pt);
prioritize-no-one has NB 0. The examined grid is 0.05-0.60 in steps of 0.05,
bracketing the development prevalence (0.41) and the plausible triage regime.

PROBABILITIES, NOT RANKS. Net benefit requires calibrated probabilities.
Development uses the study's cross-fitted Platt maps per seed (declared
median-of-seeds summary); held-out targets use the frozen SOURCE-ONLY Platt
probabilities exactly as deployed by E2a — no target label touches any
mapping. Calibration intercept/slope are reported next to every curve because
miscalibration shifts the effective threshold; this is a descriptive caveat,
not a repair.

Inputs are frozen (E0 OOF logits, E2a calibrated target parquets); nothing is
trained and no upstream artifact is modified. The development clinical
comparator is the existing cross-fitted E1d logistic model, evaluated on the
same draws.

CLAIM BOUNDARY. Retrospective net benefit under a hypothetical threshold
range. It does not select or validate an operating threshold, estimate
turnaround-time or cost effects, or authorize omission of molecular testing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import evaluate  # noqa: E402
from oceanpath.eval.core import compute_calibration_intercept_slope  # noqa: E402
from tools import final_v3_aim1_worklist as worklist  # noqa: E402

SEEDS = (42, 43, 44)
TARGETS = ("CPTAC", "RIH", "SurGen", "TCGA")
THRESHOLDS = tuple(round(0.05 * k, 2) for k in range(1, 13))  # 0.05 .. 0.60
DEV_DRAWS = 2_000
HELDOUT_DRAWS = 10_000
BOOTSTRAP_SEED = 20260817
DEFAULT_OUTPUT = (
    REPO / "reports" / "reruns" / "final_v4_additions_20260820" / "decision_curve"
)
E2A_ROOT = worklist.E2A_ROOT


def net_benefit_curves(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_draws: int,
    seed: int,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> list[dict[str, Any]]:
    """Point and patient-bootstrap net benefit at each threshold.

    The bootstrap reuses one multinomial count matrix across thresholds and
    comparators, so model-minus-default deltas are paired within draw.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_draws).astype(float)
    prevalence_draws = counts @ y / n

    rows: list[dict[str, Any]] = []
    for pt in thresholds:
        odds = pt / (1.0 - pt)
        flagged = (p >= pt).astype(float)
        tp_vec = y * flagged
        fp_vec = (1.0 - y) * flagged
        nb_point = float(tp_vec.mean() - fp_vec.mean() * odds)
        nb_draws = (counts @ tp_vec) / n - (counts @ fp_vec) / n * odds
        nb_all_point = float(y.mean() - (1.0 - y.mean()) * odds)
        nb_all_draws = prevalence_draws - (1.0 - prevalence_draws) * odds
        best_default_point = max(nb_all_point, 0.0)
        best_default_draws = np.maximum(nb_all_draws, 0.0)
        delta_draws = nb_draws - best_default_draws

        def pctl(a: np.ndarray) -> list[float]:
            return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

        rows.append(
            {
                "threshold": pt,
                "net_benefit": nb_point,
                "net_benefit_ci": pctl(nb_draws),
                "net_benefit_all": nb_all_point,
                "net_benefit_all_ci": pctl(nb_all_draws),
                "net_benefit_none": 0.0,
                "delta_vs_best_default": nb_point - best_default_point,
                "delta_vs_best_default_ci": pctl(delta_draws),
                "flagged_fraction": float(flagged.mean()),
            }
        )
    return rows


def summarize_range(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive threshold range where the model beats both defaults."""
    positive = [r["threshold"] for r in rows if r["delta_vs_best_default"] > 0]
    strict = [
        r["threshold"] for r in rows if r["delta_vs_best_default_ci"][0] > 0
    ]
    return {
        "thresholds_model_above_best_default_point": positive,
        "thresholds_model_above_best_default_ci_lower": strict,
    }


def median_of_seed_curves(per_seed: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """E1a convention: medians of the three seed-specific points/endpoints."""
    out: list[dict[str, Any]] = []
    for idx in range(len(per_seed[0])):
        merged = dict(per_seed[0][idx])
        for key in ("net_benefit", "net_benefit_all", "delta_vs_best_default", "flagged_fraction"):
            merged[key] = float(np.median([curve[idx][key] for curve in per_seed]))
        for key in ("net_benefit_ci", "net_benefit_all_ci", "delta_vs_best_default_ci"):
            merged[key] = [
                float(np.median([curve[idx][key][0] for curve in per_seed])),
                float(np.median([curve[idx][key][1] for curve in per_seed])),
            ]
        out.append(merged)
    return out


def calibration_of(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    fit = compute_calibration_intercept_slope(np.asarray(y), np.asarray(p))
    return {
        "calibration_intercept": float(fit["calibration_intercept"]),
        "calibration_slope": float(fit["calibration_slope"]),
    }


def run_analysis(n_dev: int, n_heldout: int) -> tuple[dict[str, Any], list[Path]]:
    base, checks, inputs = worklist.load_e0()

    dev_blocks: dict[str, Any] = {}
    per_seed_curves: list[list[dict[str, Any]]] = []
    per_seed_calibration: dict[str, Any] = {}
    y_dev = base["label"].to_numpy(dtype=float)
    for seed in SEEDS:
        frame = base[["patient_id", "label", "k_fold"]].copy()
        frame["mean_logit"] = base[f"wsi_seed{seed}"]
        frame["prob_raw"] = 1.0 / (1.0 + np.exp(-frame["mean_logit"]))
        calibrated = evaluate.cross_fitted_platt(frame, fold_column="k_fold")
        p = calibrated["prob_cal"].to_numpy(dtype=float)
        per_seed_curves.append(
            net_benefit_curves(y_dev, p, n_draws=n_dev, seed=BOOTSTRAP_SEED)
        )
        per_seed_calibration[str(seed)] = calibration_of(y_dev, p)
    median_curve = median_of_seed_curves(per_seed_curves)
    dev_blocks["wsi_declared_seed_median"] = {
        "curve": median_curve,
        "range_summary": summarize_range(median_curve),
        "per_seed_calibration": per_seed_calibration,
        "probability": "cross-fitted Platt per seed (E0 convention)",
    }

    clinical_p = base["clinical_oof"].to_numpy(dtype=float)
    clinical_curve = net_benefit_curves(
        y_dev, clinical_p, n_draws=n_dev, seed=BOOTSTRAP_SEED
    )
    dev_blocks["clinical_cross_fitted"] = {
        "curve": clinical_curve,
        "range_summary": summarize_range(clinical_curve),
        "calibration": calibration_of(y_dev, clinical_p),
        "probability": "cross-fitted E1d logistic probability",
    }

    heldout_blocks: dict[str, Any] = {}
    for target in TARGETS:
        path = E2A_ROOT / "calibrated" / f"cap{8192}_{target.lower()}_primary.parquet"
        inputs.append(path)
        frame = pd.read_parquet(path)
        y = frame["label"].to_numpy(dtype=float)
        p = frame["prob_source_calibrated"].to_numpy(dtype=float)
        curve = net_benefit_curves(
            y, p, n_draws=n_heldout, seed=worklist.stable_seed(f"dca_{target}")
        )
        heldout_blocks[target] = {
            "n": int(len(frame)),
            "n_mutant": int(y.sum()),
            "prevalence": float(y.mean()),
            "curve": curve,
            "range_summary": summarize_range(curve),
            "calibration": calibration_of(y, p),
            "probability": "frozen source-only Platt (E2a deployment object)",
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "component": "decision_curve",
        "decision": (
            "flag a resected primary CRC case for molecular-testing prioritization "
            "when predicted mutation probability >= threshold"
        ),
        "thresholds": list(THRESHOLDS),
        "development": dev_blocks,
        "heldout": heldout_blocks,
        "conventions": {
            "dev_draws": n_dev,
            "heldout_draws": n_heldout,
            "bootstrap_seed_dev": BOOTSTRAP_SEED,
            "paired": "model-minus-default deltas share multinomial draws",
            "seed_summary": "median of the three seed-specific points and interval endpoints",
        },
        "validation": checks,
    }
    return payload, inputs


def render_design(payload: dict[str, Any]) -> str:
    lines = [
        "# Final-v4 addition — decision-curve (net-benefit) analysis",
        "",
        f"Generated {payload['created_utc']}. Frozen probabilities; no threshold selected.",
        "",
        "Net benefit is reported per 100 patients. `delta` is model minus the best",
        "label-free default (prioritize-everyone or prioritize-no-one) at that threshold.",
        "",
    ]

    def table(title: str, rows: list[dict[str, Any]], calibration: dict | None) -> None:
        lines.append(f"## {title}")
        if calibration:
            lines.append(
                f"\nCalibration intercept {calibration['calibration_intercept']:+.3f}, "
                f"slope {calibration['calibration_slope']:.3f}.\n"
            )
        lines.append("| pt | NB model | 95% CI | NB all | delta vs best default | 95% CI |")
        lines.append("| ---: | ---: | --- | ---: | ---: | --- |")
        for row in rows:
            lines.append(
                f"| {row['threshold']:.2f} | {100 * row['net_benefit']:.2f} | "
                f"[{100 * row['net_benefit_ci'][0]:.2f}, {100 * row['net_benefit_ci'][1]:.2f}] | "
                f"{100 * row['net_benefit_all']:.2f} | "
                f"{100 * row['delta_vs_best_default']:.2f} | "
                f"[{100 * row['delta_vs_best_default_ci'][0]:.2f}, "
                f"{100 * row['delta_vs_best_default_ci'][1]:.2f}] |"
            )
        lines.append("")

    dev = payload["development"]
    table(
        "Development — WSI (declared seed median, cross-fitted Platt)",
        dev["wsi_declared_seed_median"]["curve"],
        None,
    )
    table(
        "Development — clinical comparator",
        dev["clinical_cross_fitted"]["curve"],
        dev["clinical_cross_fitted"]["calibration"],
    )
    for target, block in payload["heldout"].items():
        table(
            f"Held-out {target} (source-only Platt, n={block['n']}, prevalence "
            f"{block['prevalence']:.3f})",
            block["curve"],
            block["calibration"],
        )
    lines += [
        "Boundary: retrospective net benefit across a hypothetical threshold range;",
        "no operating threshold is validated and molecular testing is not replaced.",
        "",
    ]
    return "\n".join(lines)


def write_once(output: Path, payload: dict[str, Any], inputs: list[Path]) -> None:
    if output.exists():
        raise FileExistsError(f"append-only destination already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    unique_inputs = sorted({p.resolve() for p in inputs})
    (output / "input_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": payload["created_utc"],
                "inputs": [worklist.identity(p) for p in unique_inputs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "results.json").write_text(
        json.dumps(worklist._json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    flat = []
    for name, block in (
        ("dev_wsi", payload["development"]["wsi_declared_seed_median"]),
        ("dev_clinical", payload["development"]["clinical_cross_fitted"]),
        *((f"heldout_{t}", payload["heldout"][t]) for t in TARGETS),
    ):
        for row in block["curve"]:
            flat.append({"population": name, **row})
    pd.DataFrame(flat).to_csv(output / "decision_curves.csv", index=False)
    (output / "DESIGN_AND_RESULTS.md").write_text(render_design(payload))
    produced = [
        output / "input_receipt.json",
        output / "results.json",
        output / "decision_curves.csv",
        output / "DESIGN_AND_RESULTS.md",
    ]
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "append_only": True,
                "created_utc": payload["created_utc"],
                "output_root": str(output.resolve()),
                "artifacts": [worklist.identity(p) for p in produced],
                "promise": "No upstream result root or prior report bundle was modified.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-dev", type=int, default=DEV_DRAWS)
    parser.add_argument("--n-heldout", type=int, default=HELDOUT_DRAWS)
    args = parser.parse_args()
    payload, inputs = run_analysis(args.n_dev, args.n_heldout)
    failed = [
        key
        for key, value in payload["validation"].items()
        if isinstance(value, dict) and value.get("pass") is False
    ]
    if failed:
        raise SystemExit(f"sealed-identity validation failed: {failed}")
    write_once(args.output, payload, inputs)
    print(f"PASS — wrote {args.output}")


if __name__ == "__main__":
    main()
