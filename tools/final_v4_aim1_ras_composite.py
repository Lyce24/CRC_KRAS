#!/usr/bin/env python3
"""Extended-RAS / MAPK composite-endpoint re-scoring for the v4 KRAS report.

MOTIVATION. Why-D established that BRAF-mutant MSS wild-types are scored more
KRAS-like (+1.68 logits), i.e. the image signal tracks a shared MAPK context
rather than the KRAS locus alone. The frozen manifests carry ``nras`` and the
extended-RAS composite ``ras`` for 1,483 of 1,486 development patients, so the
clinically decisional endpoint — anti-EGFR eligibility requires extended-RAS
(and practically BRAF) wild-type status — is evaluable read-only against the
same frozen scores. This tool asks three questions:

    1. CONTAMINATION. Do the KRAS-wild-type NRAS-mutant patients — currently
       counted as negatives — score KRAS-like, as the MAPK interpretation
       predicts?
    2. ENDPOINT. Does relabeling the frozen scores from KRAS to extended-RAS
       (``ras``) and to RAS-or-BRAF ("MAPK", the anti-EGFR-ineligibility
       surrogate) change discrimination, on shared patient resamples?
    3. OPERATION. What does the fixed top-30% worklist capture under the
       composite endpoints, in development and in the four held-out targets?

NO model is trained, NO threshold is chosen, and NO upstream file is modified.
Scores are the frozen E0 OOF native logits and the corrected E2a held-out
three-seed ensembles; both loaders re-verify their sealed identities before
any new number is computed. Endpoint labels come only from the frozen
manifests; the composite ``ras`` column is re-derived from ``kras``/``nras``
row-wise and the tool fails closed on any disagreement.

CLAIM BOUNDARY. Composite-endpoint discrimination and enrichment are
retrospective ranking statements. They do not validate anti-EGFR treatment
selection, establish NRAS-specific detectability (n is small), or replace
molecular testing. The MAPK composite is a surrogate for "molecular anti-EGFR
ineligibility" limited to the measured KRAS/NRAS/BRAF labels; it ignores
amplifications, fusions, and any unmeasured resistance mechanism.
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
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools import final_v3_aim1_worklist as worklist  # noqa: E402

SEEDS = (42, 43, 44)
TARGETS = ("CPTAC", "RIH", "SurGen", "TCGA")
DEV_MANIFEST = worklist.DEV_MANIFEST
AUROC_DRAWS = 2_000
AUROC_SEED = 20260817  # E0/E1a interval convention
HELDOUT_DRAWS = 10_000  # E2a inference convention
WORKLIST_DRAWS = 10_000
WORKLIST_SEED = worklist.BOOTSTRAP_SEED  # 20260820
PRIMARY_CAPACITY = worklist.PRIMARY_CAPACITY
DEFAULT_OUTPUT = (
    REPO / "reports" / "reruns" / "final_v4_additions_20260820" / "aim1_ras_composite"
)

ENDPOINTS = ("kras", "ras", "mapk")


# ── endpoint derivation ──────────────────────────────────────────────────────
def derive_endpoints(patients: pd.DataFrame) -> pd.DataFrame:
    """Attach kras/ras/mapk 0-1 endpoint columns (NaN where undefined).

    ``ras`` is re-derived from kras/nras and cross-checked against the
    manifest's composite column: mutant if either gene is mutant, wild_type if
    both observed wild_type, otherwise unknown. ``mapk`` extends ras with BRAF:
    one observed mutation suffices for positivity even when the other genes
    are unknown; negativity requires all three observed wild_type.
    """
    frame = patients.copy()
    for column in ("kras", "nras", "braf", "ras"):
        if column not in frame.columns:
            raise ValueError(f"missing molecular column '{column}'")
        frame[column] = frame[column].fillna("unknown").astype(str)

    kras_m = frame["kras"] == "mutant"
    nras_m = frame["nras"] == "mutant"
    braf_m = frame["braf"] == "mutant"
    kras_w = frame["kras"] == "wild_type"
    nras_w = frame["nras"] == "wild_type"
    braf_w = frame["braf"] == "wild_type"

    ras = np.where(kras_m | nras_m, 1.0, np.where(kras_w & nras_w, 0.0, np.nan))
    manifest_ras = frame["ras"].map({"mutant": 1.0, "wild_type": 0.0}).astype(float)
    derived = pd.Series(ras, index=frame.index)
    if not derived.fillna(-1).eq(manifest_ras.fillna(-1)).all():
        bad = frame.loc[derived.fillna(-1) != manifest_ras.fillna(-1), "patient_id"]
        raise AssertionError(
            f"manifest 'ras' disagrees with kras|nras derivation for {len(bad)} "
            f"patients, e.g. {bad.head().tolist()}"
        )
    frame["endpoint_kras"] = frame["label"].astype(float)
    if not frame["endpoint_kras"].eq(kras_m.astype(float)).all():
        raise AssertionError("target label disagrees with manifest kras column")
    frame["endpoint_ras"] = derived
    frame["endpoint_mapk"] = np.where(
        kras_m | nras_m | braf_m, 1.0, np.where(kras_w & nras_w & braf_w, 0.0, np.nan)
    )
    # Pathway-context groups among KRAS-wild-type patients (contamination cells).
    frame["wt_context"] = np.where(
        ~kras_w,
        "kras_mutant",
        np.select(
            [
                nras_m,
                braf_m & nras_w,
                nras_w & braf_w,
            ],
            ["nras_mutant", "braf_mutant_nras_wt", "pathway_quiet"],
            default="incomplete_labels",
        ),
    )
    return frame


# ── bootstrap helpers ────────────────────────────────────────────────────────
def _pctl(values: list[float] | np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def endpoint_auroc_block(
    frame: pd.DataFrame,
    endpoint: str,
    score_columns: dict[str, str],
    *,
    n_draws: int,
    seed: int,
) -> dict[str, Any]:
    """AUROC per score column on the endpoint population, plus the paired
    endpoint-relabeling delta versus KRAS on shared patient resamples."""
    population = frame[np.isfinite(frame[f"endpoint_{endpoint}"])].reset_index(drop=True)
    y = population[f"endpoint_{endpoint}"].to_numpy()
    y_kras = population["endpoint_kras"].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(population)
    indices = rng.integers(0, n, size=(n_draws, n))

    out: dict[str, Any] = {
        "endpoint": endpoint,
        "n": int(n),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "n_draws": n_draws,
        "bootstrap_seed": seed,
        "scores": {},
    }
    for name, column in score_columns.items():
        s = population[column].to_numpy()
        point = float(roc_auc_score(y, s))
        boots, deltas = [], []
        for idx in indices:
            yi, si = y[idx], s[idx]
            if len(np.unique(yi)) < 2:
                continue
            a = roc_auc_score(yi, si)
            boots.append(a)
            yk = y_kras[idx]
            if endpoint != "kras" and len(np.unique(yk)) > 1:
                deltas.append(a - roc_auc_score(yk, si))
        entry: dict[str, Any] = {"auroc": point, "auroc_ci": _pctl(boots)}
        if endpoint != "kras":
            entry["delta_vs_kras_on_shared_population"] = float(
                point - roc_auc_score(y_kras, s)
            )
            entry["delta_vs_kras_ci"] = _pctl(deltas)
        out["scores"][name] = entry
    return out


def median_of_endpoints(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """E1a convention: the point and each interval endpoint are medians of the
    three seed-specific values (descriptive endpoint summary, not n=3
    inference)."""

    def med(path) -> float:
        return float(np.median([path(entry) for entry in per_seed]))

    out = {
        "auroc": med(lambda e: e["auroc"]),
        "auroc_ci": [med(lambda e: e["auroc_ci"][0]), med(lambda e: e["auroc_ci"][1])],
    }
    if "delta_vs_kras_on_shared_population" in per_seed[0]:
        out["delta_vs_kras_on_shared_population"] = med(
            lambda e: e["delta_vs_kras_on_shared_population"]
        )
        out["delta_vs_kras_ci"] = [
            med(lambda e: e["delta_vs_kras_ci"][0]),
            med(lambda e: e["delta_vs_kras_ci"][1]),
        ]
    return out


def context_cells(frame: pd.DataFrame) -> dict[str, Any]:
    """NRAS/BRAF contamination cells among KRAS-wild-type patients.

    Mirrors the Why-D presentation: ensemble native-logit location per cell,
    a patient bootstrap for the mean, and cell-versus-quiet Mann-Whitney
    (whose U/n1n2 is exactly the AUROC of the score for separating the cell
    from the quiet group)."""
    wt = frame[frame["wt_context"] != "kras_mutant"]
    quiet = wt[wt["wt_context"] == "pathway_quiet"]
    rng = np.random.default_rng(AUROC_SEED)
    cells: dict[str, Any] = {}
    for name, block in wt.groupby("wt_context"):
        eta = block["wsi_three_seed_ensemble"].to_numpy()
        boot_means = [
            float(np.mean(eta[rng.integers(0, len(eta), len(eta))])) for _ in range(AUROC_DRAWS)
        ]
        cell: dict[str, Any] = {
            "n": int(len(block)),
            "mean_logit": float(np.mean(eta)),
            "mean_logit_ci": _pctl(boot_means),
            "median_logit": float(np.median(eta)),
            "mean_probability": float(np.mean(1 / (1 + np.exp(-eta)))),
            "per_seed_mean_logit": {
                str(s): float(block[f"wsi_seed{s}"].mean()) for s in SEEDS
            },
        }
        if name not in ("pathway_quiet", "incomplete_labels") and len(quiet) and len(block):
            quiet_eta = quiet["wsi_three_seed_ensemble"].to_numpy()
            stat = mannwhitneyu(eta, quiet_eta, alternative="two-sided")
            cell["vs_pathway_quiet"] = {
                "mean_logit_difference": float(np.mean(eta) - np.mean(quiet_eta)),
                "auroc_cell_vs_quiet": float(stat.statistic / (len(eta) * len(quiet_eta))),
                "mannwhitney_p_two_sided": float(stat.pvalue),
            }
        cells[str(name)] = cell
    return cells


# ── worklist blocks ──────────────────────────────────────────────────────────
def dev_worklist(frame: pd.DataFrame, endpoint: str, n_bootstrap: int) -> list[dict[str, Any]]:
    population = frame[np.isfinite(frame[f"endpoint_{endpoint}"])].reset_index(drop=True)
    y = population[f"endpoint_{endpoint}"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []

    per_seed_rows = []
    for seed in SEEDS:
        seed_rows, _ = worklist.summarize_method(
            analysis="e0_dev_composite",
            population=f"{endpoint}_label_complete",
            method=f"wsi_seed{seed}",
            y=y,
            score=population[f"wsi_seed{seed}"].to_numpy(),
            n_bootstrap=n_bootstrap,
            bootstrap_seed=WORKLIST_SEED,
            status="per_seed",
        )
        per_seed_rows.append(seed_rows)
    for capacity_index in range(len(per_seed_rows[0])):
        merged = dict(per_seed_rows[0][capacity_index])
        merged["method"] = "wsi_declared_seed_median"
        merged["status"] = "primary"
        for metric in worklist.METRICS:
            for suffix in ("", "_ci_low", "_ci_high"):
                merged[f"{metric}{suffix}"] = float(
                    np.median(
                        [seed_rows[capacity_index][f"{metric}{suffix}"] for seed_rows in per_seed_rows]
                    )
                )
        rows.append(merged)

    ensemble_rows, _ = worklist.summarize_method(
        analysis="e0_dev_composite",
        population=f"{endpoint}_label_complete",
        method="wsi_three_seed_ensemble",
        y=y,
        score=population["wsi_three_seed_ensemble"].to_numpy(),
        n_bootstrap=n_bootstrap,
        bootstrap_seed=WORKLIST_SEED,
        status="reference",
    )
    random_rows, _ = worklist.summarize_method(
        analysis="e0_dev_composite",
        population=f"{endpoint}_label_complete",
        method="random_expected",
        y=y,
        score=None,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=WORKLIST_SEED,
        status="comparator",
    )
    rows.extend(ensemble_rows)
    rows.extend(random_rows)
    for row in rows:
        row["endpoint"] = endpoint
    return rows


def heldout_worklist(
    frames: dict[str, pd.DataFrame], endpoint: str, n_bootstrap: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    macro_boot: list[np.ndarray] = []
    macro_points: list[np.ndarray] = []
    for target in TARGETS:
        population = frames[target]
        population = population[np.isfinite(population[f"endpoint_{endpoint}"])]
        y = population[f"endpoint_{endpoint}"].to_numpy(dtype=float)
        target_rows, boot = worklist.summarize_method(
            analysis="e2a_heldout_composite",
            population=f"{target}_{endpoint}_label_complete",
            method="wsi_three_seed_ensemble",
            y=y,
            score=population["mean_logit"].to_numpy(),
            n_bootstrap=n_bootstrap,
            bootstrap_seed=worklist.stable_seed(f"heldout_{endpoint}_{target}"),
            status="primary",
        )
        rows.extend(target_rows)
        macro_boot.append(boot)
        macro_points.append(
            np.asarray(
                [
                    [row[metric] for metric in worklist.METRICS]
                    for row in target_rows
                ]
            )
        )
    stacked = np.stack(macro_boot)  # (targets, draws, capacities, metrics)
    macro_draws = np.nanmean(stacked, axis=0)
    point = np.mean(np.stack(macro_points), axis=0)
    for capacity_index, capacity in enumerate(worklist.CAPACITIES):
        row: dict[str, Any] = {
            "analysis": "e2a_heldout_composite",
            "population": f"equal_target_macro_{endpoint}",
            "method": "wsi_three_seed_ensemble",
            "status": "primary",
            "capacity_nominal": capacity,
            "endpoint": endpoint,
        }
        for metric_index, metric in enumerate(worklist.METRICS):
            draws = macro_draws[:, capacity_index, metric_index]
            row[metric] = float(point[capacity_index, metric_index])
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = _pctl(draws)
        rows.append(row)
    for row in rows:
        row["endpoint"] = endpoint
    return rows


# ── main analysis ────────────────────────────────────────────────────────────
def run_analysis(n_auroc: int, n_heldout: int, n_worklist: int) -> tuple[dict[str, Any], list[Path]]:
    base, checks, inputs = worklist.load_e0()
    manifest = pd.read_csv(DEV_MANIFEST, low_memory=False)
    molecular = manifest.drop_duplicates("patient_id")[
        ["patient_id", "kras", "nras", "braf", "ras"]
    ]
    base = base.drop(columns=[c for c in ("kras", "nras", "braf", "ras") if c in base.columns])
    base = base.merge(molecular, on="patient_id", validate="one_to_one")
    base = derive_endpoints(base)

    counts = {
        "dev_patients": int(len(base)),
        "kras": {"positive": int(base["endpoint_kras"].sum()), "n": int(len(base))},
        "ras": {
            "positive": int(np.nansum(base["endpoint_ras"])),
            "n": int(np.isfinite(base["endpoint_ras"]).sum()),
        },
        "mapk": {
            "positive": int(np.nansum(base["endpoint_mapk"])),
            "n": int(np.isfinite(base["endpoint_mapk"]).sum()),
        },
        "wt_context": base.loc[base["wt_context"] != "kras_mutant", "wt_context"]
        .value_counts()
        .to_dict(),
    }
    if counts["ras"] != {"positive": 668, "n": 1483}:
        raise AssertionError(f"extended-RAS census drift: {counts['ras']}")

    score_columns = {f"seed{s}": f"wsi_seed{s}" for s in SEEDS}
    dev_blocks: dict[str, Any] = {}
    for endpoint in ENDPOINTS:
        per_seed_block = endpoint_auroc_block(
            base, endpoint, score_columns, n_draws=n_auroc, seed=AUROC_SEED
        )
        per_seed_entries = [per_seed_block["scores"][f"seed{s}"] for s in SEEDS]
        ensemble_block = endpoint_auroc_block(
            base,
            endpoint,
            {"ensemble": "wsi_three_seed_ensemble"},
            n_draws=n_auroc,
            seed=AUROC_SEED,
        )
        dev_blocks[endpoint] = {
            "population": {
                key: per_seed_block[key] for key in ("n", "n_positive", "prevalence")
            },
            "per_seed": per_seed_block["scores"],
            "declared_seed_median": median_of_endpoints(per_seed_entries),
            "three_seed_ensemble": ensemble_block["scores"]["ensemble"],
        }

    # KRAS with pathway-clean negatives: mutants versus pathway-quiet WT only.
    clean = base[
        (base["endpoint_kras"] == 1.0) | (base["wt_context"] == "pathway_quiet")
    ].copy()
    clean_entries = []
    for seed in SEEDS:
        y = clean["endpoint_kras"].to_numpy()
        s = clean[f"wsi_seed{seed}"].to_numpy()
        rng = np.random.default_rng(AUROC_SEED)
        boots = []
        for _ in range(n_auroc):
            idx = rng.integers(0, len(clean), len(clean))
            if len(np.unique(y[idx])) > 1:
                boots.append(roc_auc_score(y[idx], s[idx]))
        clean_entries.append({"auroc": float(roc_auc_score(y, s)), "auroc_ci": _pctl(boots)})
    dev_blocks["kras_vs_pathway_quiet_wt"] = {
        "population": {
            "n": int(len(clean)),
            "n_positive": int(clean["endpoint_kras"].sum()),
            "note": "KRAS mutants versus KRAS/NRAS/BRAF-wild-type negatives only",
        },
        "declared_seed_median": median_of_endpoints(clean_entries),
    }

    heldout_frames: dict[str, pd.DataFrame] = {}
    heldout_blocks: dict[str, Any] = {}
    for target in TARGETS:
        frame, check, target_inputs = worklist.load_e2a_target(target)
        checks[f"e2a_{target}"] = check
        inputs.extend(target_inputs)
        target_molecular = (
            pd.read_csv(worklist.target_manifest(target), low_memory=False)
            .drop_duplicates("patient_id")[["patient_id", "kras", "nras", "braf", "ras"]]
        )
        frame = frame.drop(
            columns=[c for c in ("kras", "nras", "braf", "ras") if c in frame.columns]
        )
        frame = frame.merge(target_molecular, on="patient_id", validate="one_to_one")
        frame = derive_endpoints(frame)
        heldout_frames[target] = frame
        heldout_blocks[target] = {
            endpoint: endpoint_auroc_block(
                frame,
                endpoint,
                {"ensemble": "mean_logit"},
                n_draws=n_heldout,
                seed=AUROC_SEED,
            )
            for endpoint in ENDPOINTS
        }

    macro: dict[str, Any] = {}
    for endpoint in ENDPOINTS:
        aurocs = [
            heldout_blocks[t][endpoint]["scores"]["ensemble"]["auroc"] for t in TARGETS
        ]
        macro[endpoint] = {"equal_target_macro_auroc": float(np.mean(aurocs))}

    worklist_rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        worklist_rows.extend(dev_worklist(base, endpoint, n_worklist))
        worklist_rows.extend(heldout_worklist(heldout_frames, endpoint, n_worklist))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "component": "aim1_ras_composite",
        "endpoint_definitions": {
            "kras": "KRAS mutant versus KRAS wild-type (study reference)",
            "ras": "extended RAS: KRAS or NRAS mutant versus both observed wild-type",
            "mapk": (
                "RAS or BRAF mutant versus KRAS/NRAS/BRAF all observed wild-type "
                "(anti-EGFR molecular-ineligibility surrogate)"
            ),
        },
        "census": counts,
        "development": dev_blocks,
        "wt_context_cells": context_cells(base),
        "heldout": heldout_blocks,
        "heldout_equal_target": macro,
        "worklist_rows": worklist_rows,
        "conventions": {
            "dev_auroc_draws": n_auroc,
            "heldout_auroc_draws": n_heldout,
            "worklist_draws": n_worklist,
            "auroc_bootstrap_seed": AUROC_SEED,
            "worklist_bootstrap_seed": WORKLIST_SEED,
            "seed_summary": "median of the three seed-specific points and interval endpoints",
            "delta_estimator": (
                "endpoint-relabeling delta on the shared label-complete population "
                "with shared patient resamples"
            ),
        },
        "validation": checks,
    }
    return payload, inputs


# ── report rendering ─────────────────────────────────────────────────────────
def render_design(payload: dict[str, Any]) -> str:
    dev = payload["development"]
    cells = payload["wt_context_cells"]
    lines = [
        "# Aim 1 final-v4 addition — extended-RAS / MAPK composite endpoints",
        "",
        f"Generated {payload['created_utc']}. Frozen scores; no training, no threshold.",
        "",
        "## Development (E0 OOF, declared seed-median convention)",
        "",
        "| Endpoint | n (positive) | AUROC [95% CI] | Delta vs KRAS [95% CI] |",
        "| --- | ---: | --- | --- |",
    ]
    for endpoint in ("kras", "ras", "mapk"):
        block = dev[endpoint]
        med = block["declared_seed_median"]
        delta = (
            f"{med['delta_vs_kras_on_shared_population']:+.4f} "
            f"[{med['delta_vs_kras_ci'][0]:+.4f}, {med['delta_vs_kras_ci'][1]:+.4f}]"
            if "delta_vs_kras_on_shared_population" in med
            else "—"
        )
        lines.append(
            f"| {endpoint} | {block['population']['n']:,} ({block['population']['n_positive']:,}) "
            f"| {med['auroc']:.4f} [{med['auroc_ci'][0]:.4f}, {med['auroc_ci'][1]:.4f}] | {delta} |"
        )
    clean = dev["kras_vs_pathway_quiet_wt"]
    med = clean["declared_seed_median"]
    lines += [
        f"| kras vs quiet-WT | {clean['population']['n']:,} "
        f"({clean['population']['n_positive']:,}) | "
        f"{med['auroc']:.4f} [{med['auroc_ci'][0]:.4f}, {med['auroc_ci'][1]:.4f}] | — |",
        "",
        "## KRAS-wild-type pathway-context cells (three-seed ensemble logit)",
        "",
        "| Cell | n | Mean logit [95% CI] | AUROC vs quiet | MW p |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for name, cell in sorted(cells.items()):
        contrast = cell.get("vs_pathway_quiet", {})
        lines.append(
            f"| {name} | {cell['n']} | {cell['mean_logit']:+.3f} "
            f"[{cell['mean_logit_ci'][0]:+.3f}, {cell['mean_logit_ci'][1]:+.3f}] | "
            f"{contrast.get('auroc_cell_vs_quiet', float('nan')):.3f} | "
            f"{contrast.get('mannwhitney_p_two_sided', float('nan')):.2e} |"
        )
    lines += ["", "## Held-out targets (E2a three-seed ensemble)", ""]
    lines += ["| Target | Endpoint | n (positive) | AUROC [95% CI] | Delta vs KRAS [95% CI] |",
              "| --- | --- | ---: | --- | --- |"]
    for target in TARGETS:
        for endpoint in ("kras", "ras", "mapk"):
            block = payload["heldout"][target][endpoint]
            entry = block["scores"]["ensemble"]
            delta = (
                f"{entry['delta_vs_kras_on_shared_population']:+.4f} "
                f"[{entry['delta_vs_kras_ci'][0]:+.4f}, {entry['delta_vs_kras_ci'][1]:+.4f}]"
                if "delta_vs_kras_on_shared_population" in entry
                else "—"
            )
            lines.append(
                f"| {target} | {endpoint} | {block['n']:,} ({block['n_positive']:,}) | "
                f"{entry['auroc']:.4f} [{entry['auroc_ci'][0]:.4f}, {entry['auroc_ci'][1]:.4f}] "
                f"| {delta} |"
            )
    lines += [
        "",
        "## Top-30% worklist (primary capacity)",
        "",
        "| Analysis | Endpoint | Method | Capture [95% CI] | PPV | Enrichment |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in payload["worklist_rows"]:
        if abs(row["capacity_nominal"] - PRIMARY_CAPACITY) > 1e-9:
            continue
        if row["method"] not in (
            "wsi_declared_seed_median",
            "wsi_three_seed_ensemble",
            "random_expected",
        ):
            continue
        lines.append(
            f"| {row['analysis']}/{row['population']} | {row['endpoint']} | {row['method']} | "
            f"{row['capture']:.3f} [{row['capture_ci_low']:.3f}, {row['capture_ci_high']:.3f}] | "
            f"{row['worklist_ppv']:.3f} | {row['enrichment']:.3f} |"
        )
    lines += [
        "",
        "Boundary: retrospective ranking and enrichment only; no anti-EGFR",
        "treatment-selection validation; NRAS-level detectability is not claimed.",
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
    pd.DataFrame(payload["worklist_rows"]).to_csv(output / "worklist_rows.csv", index=False)
    (output / "DESIGN_AND_RESULTS.md").write_text(render_design(payload))
    produced = [
        output / "input_receipt.json",
        output / "results.json",
        output / "worklist_rows.csv",
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
    parser.add_argument("--n-auroc", type=int, default=AUROC_DRAWS)
    parser.add_argument("--n-heldout", type=int, default=HELDOUT_DRAWS)
    parser.add_argument("--n-worklist", type=int, default=WORKLIST_DRAWS)
    args = parser.parse_args()
    payload, inputs = run_analysis(args.n_auroc, args.n_heldout, args.n_worklist)
    failed = [k for k, v in payload["validation"].items() if isinstance(v, dict) and v.get("pass") is False]
    if failed:
        raise SystemExit(f"sealed-identity validation failed: {failed}")
    write_once(args.output, payload, inputs)
    print(f"PASS — wrote {args.output}")


if __name__ == "__main__":
    main()
