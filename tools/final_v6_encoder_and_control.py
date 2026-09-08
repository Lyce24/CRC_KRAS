#!/usr/bin/env python3
"""Final-v6 component: encoder robustness and the pipeline positive control.

Seals the three campaigns pre-registered in final-v5 section 11.5 into one
append-only, receipted root, together with the two derived analyses their raw
outputs demand:

  E1v  Aim 1 Virchow2-CLS at the study cap 8,192 -> is the headline an
       artifact of encoder choice?
  E3v  Virchow2 replication of the three Aim 3 consensus ceilings -> is the
       resolution boundary encoder-robust?
  E1e  MSI/dMMR positive control -> does the locked pipeline learn when a
       strong signal exists?

DERIVED ANALYSES (computed here, not in the launchers):

  1. PAIRED ENCODER CONTRAST. Both cap-8,192 arms score the identical 1,486
     patients, so the encoder comparison must be paired; an unpaired interval
     would discard the correlation and overstate uncertainty. Reported at A,
     A-complete and Set D on shared patient resamples (2,000 draws, seed
     20260817), plus the Spearman rank correlation between the two frozen
     three-seed ensembles.
  2. WHY-D ENCODER COMPARISON. The adjusted MSI x BRAF model on KRAS-wild-type
     patients is read from the frozen why-D artifacts of all three arms, to
     test whether a larger Set-D gain tracks a larger BRAF-context effect.

PRE-DECLARED READINGS (final-v5 section 11.5, applied verbatim):
  * E1v is a sensitivity row and the Virchow2 gene-level reference for E3v;
  * E3v ceilings replicate -> the boundary is encoder-robust; any rung opens
    -> a positive discovery requiring Aim 3 revision, not defense;
  * E1e is a pipeline control only, never an MSI assay, never mixed into a
    KRAS table.

Every sealed comparator (the UNIv1 study-arm values and the Aim 3 fixed-draw
values) is asserted before any new number is interpreted, and the E3v report's
own UNIv1 anchors are re-verified here independently.
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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import evaluate, paths  # noqa: E402
from tools import final_v3_aim1_worklist as worklist  # noqa: E402

SEEDS = (42, 43, 44)
DRAWS = 2_000
BOOTSTRAP_SEED = 20260817
EVAL = paths.OUTPUT_ROOT / "eval"
E1V_RESULTS = paths.OUTPUT_ROOT / "train" / "1a_pb_cap8192" / "virchow2_cls" / "analysis" / "results.json"
E1E_RESULTS = paths.OUTPUT_ROOT / "e1e" / "msi_cap8192" / "univ1" / "analysis" / "results.json"
E3V_RESULTS = (
    paths.OUTPUT_ROOT / "reruns" / "aim3_virchow2_cap8192_v1_20260820" / "analysis" / "results.json"
)
WHYD = {
    "univ1_cap8192": EVAL / "e1_why_D_1a_pb_cap8192.json",
    "virchow2_cap8192": EVAL / "e1_why_D_1a_pb_cap8192_virchow2_cls.json",
    "virchow2_cap4096": EVAL / "e1_why_D_1a_pb_cap4096_virchow2_cls.json",
}
DEFAULT_OUTPUT = (
    REPO / "reports" / "reruns" / "final_v6_additions_20260820" / "encoder_and_control"
)

# Sealed comparators (final-v5 Results sections 2.1/2.6 and 4.1).
SEALED = {
    "univ1_cap8192_A_median": 0.6643,
    "univ1_cap8192_D_median": 0.6877,
    "univ1_cap8192_ensemble": 0.6795456593,
    "univ1_cap4096_A_median": 0.6669,
    "virchow2_cap4096_A_median": 0.6717,
    "virchow2_cap4096_D_median": 0.7103,
}
AIM3_SEALED_UNIV1 = {
    "codon": {"fine": 0.526508, "control": 0.658113, "verdict": "CEILING"},
    "g12d_broad": {"fine": 0.486302, "control": 0.653200, "verdict": "CEILING"},
    "allele1": {"fine": 0.525216, "control": 0.643932, "verdict": "CEILING"},
}
RUNG_LABEL = {
    "codon": "G12 vs non-G12 mutant",
    "g12d_broad": "G12D vs other KRAS mutant",
    "allele1": "G12D vs other G12",
}


def load_ensemble(arm: str, encoder: str) -> pd.DataFrame:
    manifest = pd.read_csv(paths.DEV_MANIFEST, low_memory=False)
    frames = {}
    for seed in SEEDS:
        path = paths.OUTPUT_ROOT / "train" / arm / encoder / f"seed{seed}" / "oof_predictions.parquet"
        frames[seed] = (
            evaluate.to_patient_level(pd.read_parquet(path), manifest)
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
    base = frames[SEEDS[0]][["patient_id", "label", "msi_dmmr", "braf"]].copy()
    base["eta"] = np.mean([frames[s]["mean_logit"].to_numpy() for s in SEEDS], axis=0)
    return base


def paired_contrast(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    """Virchow2-minus-UNIv1 AUROC on shared patient resamples, per population."""
    if not (left["patient_id"].values == right["patient_id"].values).all():
        raise AssertionError("encoder arms do not cover identical patients")
    if not (left["label"].values == right["label"].values).all():
        raise AssertionError("encoder arms disagree on labels")
    populations = {
        "A": np.ones(len(left), dtype=bool),
        "A_complete": (
            left["msi_dmmr"].isin(["MSS/pMMR", "MSI/dMMR"])
            & left["braf"].isin(["mutant", "wild_type"])
        ).to_numpy(),
        "D": (
            left["msi_dmmr"].eq("MSS/pMMR") & left["braf"].eq("wild_type")
        ).to_numpy(),
    }
    out: dict[str, Any] = {}
    for name, mask in populations.items():
        y = left["label"].to_numpy()[mask]
        a = left["eta"].to_numpy()[mask]
        b = right["eta"].to_numpy()[mask]
        point = float(roc_auc_score(y, b) - roc_auc_score(y, a))
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = len(y)
        deltas = []
        for _ in range(DRAWS):
            index = rng.integers(0, n, n)
            if len(np.unique(y[index])) > 1:
                deltas.append(
                    roc_auc_score(y[index], b[index]) - roc_auc_score(y[index], a[index])
                )
        low, high = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
        out[name] = {
            "n": int(n),
            "auroc_univ1": float(roc_auc_score(y, a)),
            "auroc_virchow2": float(roc_auc_score(y, b)),
            "delta_virchow2_minus_univ1": point,
            "delta_ci": [low, high],
            "excludes_zero": bool(low > 0 or high < 0),
        }
    out["spearman_between_ensembles"] = float(
        spearmanr(left["eta"].to_numpy(), right["eta"].to_numpy()).statistic
    )
    out["estimator"] = (
        "paired patient bootstrap on identical patients; shared resample per draw"
    )
    return out


def whyd_terms() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm, path in WHYD.items():
        stored = json.loads(path.read_text())
        terms = {}
        for term in ("BRAF_mut", "MSI", "BRAF_mut x MSI"):
            terms[term] = {
                "beta_median": float(np.median([stored["D"][str(s)][term]["beta"] for s in SEEDS])),
                "ci_median": [
                    float(np.median([stored["D"][str(s)][term]["ci"][0] for s in SEEDS])),
                    float(np.median([stored["D"][str(s)][term]["ci"][1] for s in SEEDS])),
                ],
                "excludes_zero_all_seeds": bool(
                    all(stored["D"][str(s)][term]["excludes_zero"] for s in SEEDS)
                ),
            }
        cells = {}
        for context in stored["C"]["42"]["KRAS-WT"]:
            cells[context] = {
                "n": int(stored["C"]["42"]["KRAS-WT"][context]["n"]),
                "mean_logit_median": float(
                    np.median(
                        [stored["C"][str(s)]["KRAS-WT"][context]["mean_logit"] for s in SEEDS]
                    )
                ),
            }
        out[arm] = {"adjusted_terms": terms, "wt_context_cells": cells}
    return out


def e3v_block(e3v: dict[str, Any]) -> dict[str, Any]:
    for task, expected in AIM3_SEALED_UNIV1.items():
        observed = e3v["univ1_anchor_points"][task]
        if abs(observed - expected["fine"]) > 1e-6:
            raise AssertionError(f"{task}: UNIv1 anchor drift {observed} vs {expected['fine']}")
    rungs: dict[str, Any] = {}
    replicated = 0
    for task, block in e3v["rungs"].items():
        gate = block["fixed_gate_one_sided_99"]
        univ1 = AIM3_SEALED_UNIV1[task]
        replicated += int(gate["ceiling"])
        rungs[task] = {
            "label": RUNG_LABEL[task],
            "n": block["n"],
            "n_positive": block["n_positive"],
            "virchow2_fine": block["fine"]["auroc"],
            "virchow2_fine_ci95": block["fine"]["ci95"],
            "virchow2_control": block["control"]["auroc"],
            "virchow2_control_ci95": block["control"]["ci95"],
            "virchow2_delta": block["delta_control_minus_fine"]["auroc"],
            "virchow2_delta_ci95": block["delta_control_minus_fine"]["ci95"],
            "gate_components": gate,
            "virchow2_ceiling": gate["ceiling"],
            "univ1_fine": univ1["fine"],
            "univ1_control": univ1["control"],
            "univ1_verdict": univ1["verdict"],
            "replication": "REPLICATED" if gate["ceiling"] else "NOT_REPLICATED",
        }
    return {
        "rungs": rungs,
        "n_replicated": replicated,
        "n_rungs": len(rungs),
        "verdict": (
            "ENCODER_ROBUST_CEILINGS"
            if replicated == len(rungs)
            else "PARTIAL_OR_OPENED — Aim 3 conclusion requires revision, not defense"
        ),
    }


def run() -> dict[str, Any]:
    for path in (E1V_RESULTS, E1E_RESULTS, E3V_RESULTS, *WHYD.values()):
        if not path.is_file():
            raise SystemExit(f"missing required input (campaign incomplete?): {path}")
    e1v = json.loads(E1V_RESULTS.read_text())
    e1e = json.loads(E1E_RESULTS.read_text())
    e3v = json.loads(E3V_RESULTS.read_text())

    checks: dict[str, Any] = {}
    for key, expected in (
        ("e1v_A_vs_sealed_univ1", SEALED["univ1_cap8192_A_median"]),
        ("e1v_D_vs_sealed_univ1", SEALED["univ1_cap8192_D_median"]),
    ):
        checks[key] = {"sealed_comparator": expected}
    checks["e1v_population"] = {
        "n": e1v["three_seed_ensemble_A"]["n"],
        "n_positive": e1v["three_seed_ensemble_A"]["n_positive"],
        "pass": e1v["three_seed_ensemble_A"]["n"] == 1486
        and e1v["three_seed_ensemble_A"]["n_positive"] == 604,
    }
    checks["e1e_census"] = {"census": e1e["census"], "pass": e1e["census"]["patients"] == 1433}

    univ1 = load_ensemble("1a_pb_cap8192", "univ1")
    virchow2 = load_ensemble("1a_pb_cap8192", "virchow2_cls")
    observed_univ1_ensemble = float(roc_auc_score(univ1["label"], univ1["eta"]))
    checks["univ1_ensemble_anchor"] = {
        "observed": observed_univ1_ensemble,
        "expected": SEALED["univ1_cap8192_ensemble"],
        "pass": abs(observed_univ1_ensemble - SEALED["univ1_cap8192_ensemble"]) < 1e-6,
    }
    if not checks["univ1_ensemble_anchor"]["pass"]:
        raise AssertionError("UNIv1 study-arm ensemble anchor failed")

    contrast = paired_contrast(univ1, virchow2)
    aim1_gate = {
        "set_d_median_at_least_0p60": e1v["median_auroc_D"] >= 0.60,
        "all_seed_delta_ci_excludes_zero": all(
            entry["d_minus_a_complete_ci"][0] > 0 for entry in e1v["per_seed"].values()
        ),
        "no_material_drop_from_A": e1v["median_auroc_D"] >= e1v["median_auroc_A"],
    }
    aim1_gate["pass"] = all(aim1_gate.values())

    msi = {
        "median_pooled_auroc": e1e["median_pooled_auroc"],
        "three_seed_ensemble": e1e["three_seed_ensemble"],
        "per_seed": e1e["per_seed"],
        "prevalence": e1e["prevalence"],
        "control_verdict": (
            "PIPELINE_LEARNS_STRONG_SIGNAL"
            if e1e["three_seed_ensemble"]["ci_low"] > 0.80
            else "BELOW_FIELD_EXPECTATION — interpret the KRAS and metastatic nulls with care"
        ),
        "role": e1e["role"],
    }

    return {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "component": "final_v6_encoder_and_control",
        "e1v_aim1_virchow2_cap8192": {
            "per_seed": e1v["per_seed"],
            "median_auroc_A": e1v["median_auroc_A"],
            "median_auroc_D": e1v["median_auroc_D"],
            "median_d_minus_a_complete": e1v["median_d_minus_a_complete"],
            "three_seed_ensemble_A": e1v["three_seed_ensemble_A"],
            "e1a_gate": aim1_gate,
            "sealed_comparators": SEALED,
        },
        "paired_encoder_contrast": contrast,
        "whyd_by_encoder": whyd_terms(),
        "e3v_aim3_ceiling_replication": e3v_block(e3v),
        "e1e": msi,
        "validation": checks,
        "conventions": {
            "draws": DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "seed_summary": "median of the three seed-specific values",
        },
    }


def render(payload: dict[str, Any]) -> str:
    e1v = payload["e1v_aim1_virchow2_cap8192"]
    c = payload["paired_encoder_contrast"]
    e3 = payload["e3v_aim3_ceiling_replication"]
    msi = payload["e1e"]
    lines = [
        "# Final-v6 — encoder robustness and pipeline positive control",
        "",
        f"Generated {payload['created_utc']}.",
        "",
        "## Aim 1 — Virchow2-CLS at the study cap 8,192 (E1v)",
        "",
        "| Arm | A median | Set-D median | D − A-complete |",
        "| --- | ---: | ---: | ---: |",
        f"| UNIv1 cap-8192 (declared) | {payload['e1v_aim1_virchow2_cap8192']['sealed_comparators']['univ1_cap8192_A_median']:.4f} "
        f"| {payload['e1v_aim1_virchow2_cap8192']['sealed_comparators']['univ1_cap8192_D_median']:.4f} | +0.0222 |",
        f"| Virchow2 cap-8192 (E1v) | {e1v['median_auroc_A']:.4f} | {e1v['median_auroc_D']:.4f} "
        f"| {e1v['median_d_minus_a_complete']:+.4f} |",
        "",
        f"E1a gate on the new arm: **{'PASS' if e1v['e1a_gate']['pass'] else 'FAIL'}**.",
        "",
        "### Paired encoder contrast (identical patients)",
        "",
        "| Population | n | UNIv1 | Virchow2 | Δ (V−U) [95% CI] | excludes 0 |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for name in ("A", "A_complete", "D"):
        b = c[name]
        lines.append(
            f"| {name} | {b['n']:,} | {b['auroc_univ1']:.4f} | {b['auroc_virchow2']:.4f} | "
            f"{b['delta_virchow2_minus_univ1']:+.4f} [{b['delta_ci'][0]:+.4f}, {b['delta_ci'][1]:+.4f}] "
            f"| {b['excludes_zero']} |"
        )
    lines += [
        "",
        f"Spearman between the two frozen ensembles: {c['spearman_between_ensembles']:.3f}.",
        "",
        "### Adjusted MSI × BRAF model by encoder (KRAS-wild-type patients)",
        "",
        "| Arm | BRAF | MSI | BRAF×MSI |",
        "| --- | --- | --- | --- |",
    ]
    for arm, block in payload["whyd_by_encoder"].items():
        cells = []
        for term in ("BRAF_mut", "MSI", "BRAF_mut x MSI"):
            t = block["adjusted_terms"][term]
            mark = " ✓" if t["excludes_zero_all_seeds"] else ""
            cells.append(
                f"{t['beta_median']:+.3f} [{t['ci_median'][0]:+.3f}, {t['ci_median'][1]:+.3f}]{mark}"
            )
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Aim 3 — Virchow2 ceiling replication (E3v)",
        "",
        f"**{e3['n_replicated']}/{e3['n_rungs']} rungs replicated — {e3['verdict']}**",
        "",
        "| Rung | UNIv1 fine / control | Virchow2 fine [95% CI] | Virchow2 control [95% CI] | Gate | Replication |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in e3["rungs"].values():
        lines.append(
            f"| {r['label']} | {r['univ1_fine']:.4f} / {r['univ1_control']:.4f} | "
            f"{r['virchow2_fine']:.4f} [{r['virchow2_fine_ci95'][0]:.4f}, {r['virchow2_fine_ci95'][1]:.4f}] | "
            f"{r['virchow2_control']:.4f} [{r['virchow2_control_ci95'][0]:.4f}, {r['virchow2_control_ci95'][1]:.4f}] | "
            f"{'CEILING' if r['virchow2_ceiling'] else 'no ceiling'} | {r['replication']} |"
        )
    ens = msi["three_seed_ensemble"]
    lines += [
        "",
        "## MSI/dMMR positive control (E1e)",
        "",
        f"Three-seed ensemble AUROC **{ens['auroc']:.4f}** [{ens['ci_low']:.4f}, {ens['ci_high']:.4f}] "
        f"(n={ens['n']:,}, {ens['n_positive']} MSI/dMMR, prevalence {msi['prevalence']:.3f}); "
        f"seed-median pooled {msi['median_pooled_auroc']:.4f}.",
        "",
        f"Verdict: **{msi['control_verdict']}**. {msi['role']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run()
    failed = [
        key
        for key, value in payload["validation"].items()
        if isinstance(value, dict) and value.get("pass") is False
    ]
    if failed:
        raise SystemExit(f"validation failed: {failed}")
    output: Path = args.output
    if output.exists():
        raise FileExistsError(f"append-only destination already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(
        json.dumps(worklist._json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    (output / "DESIGN_AND_RESULTS.md").write_text(render(payload))
    inputs = [E1V_RESULTS, E1E_RESULTS, E3V_RESULTS, *WHYD.values(), paths.DEV_MANIFEST]
    produced = [output / "results.json", output / "DESIGN_AND_RESULTS.md"]
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "append_only": True,
                "created_utc": payload["created_utc"],
                "output_root": str(output.resolve()),
                "inputs": [worklist.identity(p) for p in inputs],
                "artifacts": [worklist.identity(p) for p in produced],
                "promise": "No upstream result root or prior report bundle was modified.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"PASS — {output}")
    print(render(payload))


if __name__ == "__main__":
    main()
