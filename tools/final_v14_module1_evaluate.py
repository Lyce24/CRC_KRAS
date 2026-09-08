#!/usr/bin/env python3
"""Evaluate sealed v14 Module-I predictions with fixed paired patient resamples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import final_v14_module1 as source  # noqa: E402

N_BOOT = 10000
SEED = 20260827
OUT = source.ROOT / "e4m1_results"


def auc(y: np.ndarray, score: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0 or not np.isfinite(score).all():
        return float("nan")
    return float((rankdata(score)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fidelity(y: np.ndarray, actual: np.ndarray, prediction: np.ndarray) -> dict:
    full_auc, rec_auc = auc(y, actual), auc(y, prediction)
    a, p = actual - actual.mean(), prediction - prediction.mean()
    av, pv = float(a @ a), float(p @ p)
    pearson = float(a @ p / np.sqrt(av * pv)) if av > 0 and pv > 0 else np.nan
    ar, pr = rankdata(actual), rankdata(prediction)
    ar -= ar.mean()
    pr -= pr.mean()
    denom = np.sqrt((ar @ ar) * (pr @ pr))
    slope = float(a @ p / pv) if pv > 0 else np.nan
    return {
        "r2_oof": float(1 - np.sum((actual - prediction) ** 2) / av) if av > 0 else np.nan,
        "pearson": pearson, "spearman": float(ar @ pr / denom) if denom > 0 else np.nan,
        "calibration_slope": slope,
        "calibration_intercept": float(actual.mean() - slope * prediction.mean()),
        "reconstructed_auroc": rec_auc, "full_auroc": full_auc,
        "reconstructed_minus_full_auroc": rec_auc - full_auc,
        "retention_ratio": (rec_auc - .5) / (full_auc - .5) if full_auc > .5 else np.nan,
    }


def interval(point: float, draws: list | np.ndarray, *, ratio: bool = False) -> dict:
    values = np.asarray(draws, float)
    finite = values[np.isfinite(values)]
    estimable = len(finite) >= (9500 if ratio else 1)
    ci = np.quantile(finite, [.025, .975]).tolist() if estimable else [None, None]
    return {"estimate": float(point) if np.isfinite(point) else None,
            "ci95": ci, "finite_draws": len(finite), "undefined_draws": len(values) - len(finite),
            "status": "ESTIMABLE" if estimable else ("RATIO_CI_NOT_ESTIMABLE" if ratio else "NOT_ESTIMABLE")}


def metrics(frame: pd.DataFrame, rep: str, idx: np.ndarray) -> dict:
    data = frame.iloc[idx]
    y = data.label.to_numpy(int)
    full = data.full_logit.to_numpy(float)
    out = {}
    for subset in ["all", "stage_known"]:
        mask = np.ones(len(data), bool) if subset == "all" else data.stage_known_derived.to_numpy(bool)
        yy = y[mask]
        aa = {}
        for model in ["concept", "joint", "clinical"]:
            score = data[f"{rep}__all__{model}"].to_numpy(float)[mask]
            aa[model] = auc(yy, score)
            out[f"{subset}/{model}/auroc"] = aa[model]
            out[f"{subset}/{model}/auprc"] = float(average_precision_score(yy, score)) if np.isfinite(score).all() and len(np.unique(yy)) == 2 else np.nan
        aa["full"] = auc(yy, full[mask])
        out[f"{subset}/full/auroc"] = aa["full"]
        for left, right in [("concept", "clinical"), ("joint", "clinical"), ("joint", "concept"), ("concept", "full")]:
            out[f"{subset}/{left}_minus_{right}/auroc"] = aa[left] - aa[right]
    pred = data[f"{rep}__all__ridge_mean"].to_numpy(float)
    if np.isfinite(pred).all():
        out.update({f"fidelity/{key}": value for key, value in fidelity(y, full, pred).items()})
    restricted = data.msi_dmmr.eq("MSS/pMMR").to_numpy() & data.braf.eq("wild_type").to_numpy()
    score = data[f"{rep}__all__concept"].to_numpy(float)
    restricted_auc = auc(y[restricted], score[restricted])
    out["restriction/restricted_auroc"] = restricted_auc
    out["restriction/restricted_minus_all_auroc"] = restricted_auc - auc(y, score)
    return out


def bootstrap_main(frame: pd.DataFrame, rep: str, n_boot: int = N_BOOT) -> tuple[dict, dict]:
    y = frame.label.to_numpy(int)
    cohort = frame.subcohort.to_numpy(str)
    cells = [np.flatnonzero((cohort == c) & (y == k)) for c in sorted(set(cohort)) for k in [0, 1]]
    points = metrics(frame, rep, np.arange(len(frame)))
    draws = {key: np.empty(n_boot) for key in points}
    cohorts = sorted(set(cohort))
    cohort_points = {c: fidelity(y[cohort == c], frame.full_logit.to_numpy()[cohort == c],
                                frame[f"{rep}__all__ridge_mean"].to_numpy()[cohort == c]) for c in cohorts}
    cohort_draws = {c: {key: np.empty(n_boot) for key in cohort_points[c]} for c in cohorts}
    rng = np.random.default_rng(SEED)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(cell, len(cell), replace=True) for cell in cells if len(cell)])
        values = metrics(frame, rep, idx)
        for key in draws:
            draws[key][b] = values[key]
        for c in cohorts:
            ci = idx[cohort[idx] == c]
            f = fidelity(y[ci], frame.full_logit.to_numpy()[ci], frame[f"{rep}__all__ridge_mean"].to_numpy()[ci])
            for key in f:
                cohort_draws[c][key][b] = f[key]
        if (b + 1) % 1000 == 0:
            print(json.dumps({"representation": rep, "bootstrap_draws_complete": b + 1}), flush=True)
    result = {key: interval(point, draws[key], ratio=key.endswith("retention_ratio")) for key, point in points.items()}
    subcohorts = {c: {key: interval(point, cohort_draws[c][key], ratio=key == "retention_ratio")
                     for key, point in cohort_points[c].items()} for c in cohorts}
    arrays = {**draws, **{f"subcohort/{c}/{k}": a for c in cohorts for k, a in cohort_draws[c].items()}}
    return {"pooled": result, "subcohort_fidelity": subcohorts}, arrays


def standardized_restriction(frame: pd.DataFrame, rep: str, n_boot: int = N_BOOT) -> tuple[dict, dict]:
    """Inherited E1a-S common support and fixed A-composition weighting."""
    raw = frame.loc[frame.tumor_site_group.isin(["Colon", "Rectum"])].copy()
    raw["stratum"] = raw.subcohort.astype(str) + "|" + raw.tumor_site_group.astype(str)
    raw["restricted"] = raw.msi_dmmr.eq("MSS/pMMR") & raw.braf.eq("wild_type")
    result, arrays = {}, {}
    for reference in ["A_all_primary", "A_complete"]:
        a = raw if reference == "A_all_primary" else raw.loc[raw.msi_dmmr.isin(["MSS/pMMR", "MSI/dMMR", "MSI-H/dMMR"]) & raw.braf.isin(["mutant", "wild_type"])]
        a = a.reset_index(drop=True)
        groups = []
        for key, block in a.groupby("stratum", sort=True):
            d = block.loc[block.restricted]
            if len(block) >= 10 and len(d) >= 10 and min(block.label.value_counts().reindex([0, 1], fill_value=0)) >= 3 and min(d.label.value_counts().reindex([0, 1], fill_value=0)) >= 3:
                groups.append(key)
        if not groups:
            result[reference] = {"status": "NOT_ESTIMABLE", "reason": "no common support"}
            continue
        counts = a.loc[a.stratum.isin(groups)].groupby("stratum").size()
        weights = (counts / counts.sum()).to_dict()
        y, s = a.label.to_numpy(int), a[f"{rep}__all__concept"].to_numpy(float)
        r, strata = a.restricted.to_numpy(), a.stratum.to_numpy(str)
        def estimate(idx, groups=groups, strata=strata, r=r, y=y, s=s, weights=weights):
            totals = [0.0, 0.0]
            ws = [0.0, 0.0]
            for group in groups:
                ai = idx[strata[idx] == group]
                for j, chosen in enumerate([ai, ai[r[ai]]]):
                    val = auc(y[chosen], s[chosen])
                    if np.isfinite(val):
                        totals[j] += weights[group] * val
                        ws[j] += weights[group]
            return np.array([totals[j] / ws[j] if ws[j] else np.nan for j in range(2)])
        point = estimate(np.arange(len(a)))
        cells = [np.flatnonzero((strata == g) & (y == k)) for g in groups for k in [0, 1]]
        rng = np.random.default_rng(SEED)
        draw = np.empty((n_boot, 2))
        for b in range(n_boot):
            idx = np.concatenate([rng.choice(cell, len(cell), replace=True) for cell in cells])
            draw[b] = estimate(idx)
        result[reference] = {"status": "ESTIMABLE", "strata": groups, "weights": weights,
                             "patients_A": len(a), "patients_D": int(r.sum()),
                             "standardized_A_auroc": interval(point[0], draw[:, 0]),
                             "standardized_D_auroc": interval(point[1], draw[:, 1]),
                             "standardized_D_minus_A": interval(point[1] - point[0], draw[:, 1] - draw[:, 0])}
        arrays[f"{reference}/A"] = draw[:, 0]
        arrays[f"{reference}/D"] = draw[:, 1]
        arrays[f"{reference}/D_minus_A"] = draw[:, 1] - draw[:, 0]
    return result, arrays


def evaluate(source_dir: Path = source.OUT, out: Path = OUT, module3_seal: Path | None = None) -> dict:
    summary = source.read_sealed(source_dir / "source_fit_summary.json")
    source.verify_record(summary["predictions"])
    if module3_seal is None:
        # Do not silently ignore the preregistered joint source-freeze boundary.
        raise ValueError("Module-III source seal or explicit dependency-status receipt is required before result inspection")
    dependency = source.read_sealed(module3_seal)
    stage_path = Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v4.csv")
    stage_sha = "312438c12aaa25b4376cc55b70ebc3a23763f149c7413f1fde7c3ce71c21596c"
    if source.io.sha256_file(stage_path) != stage_sha:
        raise ValueError("inherited derived-stage source digest mismatch")
    contract = {"created_utc": source.io.utc_now(), "source_summary": source.io.identity(source_dir / "source_fit_summary.json"),
                "module3_dependency": source.io.identity(module3_seal), "bootstrap_draws": N_BOOT,
                "bootstrap_seed": SEED, "code": source.io.identity(Path(__file__)),
                "derived_stage_source": source.io.identity(stage_path),
                "derived_stage_read_columns": ["patient_uid", "specimen_role", "stage_group_major_filled"],
                "dependency_ordering_amendment": "If Module III inputs are operationally unavailable, Module I can be evaluated after its own full source seal; all Module III methods remain unchanged and later source fits cannot be selected using Module I results.",
                "module3_status": dependency.get("scientific_status", dependency.get("status")),
                "interpretation": "unsupervised concept space; intervals conditional on fitted models"}
    source.seal_json(out / "evaluation_contract.json", contract)
    frame = pd.read_parquet(summary["predictions"]["path"])
    stage = pd.read_csv(stage_path, usecols=["patient_uid", "specimen_role", "stage_group_major_filled"])
    stage = stage.loc[stage.patient_uid.isin(frame.patient_id) & stage.specimen_role.str.casefold().eq("primary")]
    if (stage.groupby("patient_uid").stage_group_major_filled.nunique(dropna=False) > 1).any():
        raise ValueError("conflicting inherited derived stage")
    stage = stage.drop_duplicates("patient_uid").set_index("patient_uid")
    if set(stage.index) != set(frame.patient_id):
        raise ValueError("derived stage does not cover exact source roster")
    frame["stage_known_derived"] = stage.loc[frame.patient_id, "stage_group_major_filled"].isin(["I", "II", "III", "IV"]).to_numpy()
    if (int(frame.stage_known_derived.sum()), int(frame.loc[frame.stage_known_derived, "label"].sum())) != (1060, 422):
        raise ValueError("inherited derived-stage census drifted")
    results, all_arrays = {}, {}
    for rep in summary["representations"]:
        estimates, arrays = bootstrap_main(frame, rep)
        restriction, r_arrays = standardized_restriction(frame, rep)
        estimates["standardized_restriction"] = restriction
        results[rep] = estimates
        all_arrays.update({f"{rep}/{k}": a for k, a in arrays.items()})
        all_arrays.update({f"{rep}/standardized/{k}": a for k, a in r_arrays.items()})
    all32 = results["ALL32"]
    direct = all32["pooled"]["all/concept/auroc"]
    standardized = all32["standardized_restriction"]["A_all_primary"]
    delta = standardized.get("standardized_D_minus_A", {})
    def gate(estimate, threshold, label):
        if estimate.get("estimate") is None or estimate.get("ci95", [None])[0] is None:
            return label + "_NOT_EVALUABLE"
        return label + ("_SUPPORTED" if estimate["ci95"][0] > threshold else "_NOT_ESTABLISHED")
    results["gates"] = {"H1.1": gate(direct, .5, "H1.1_CONCEPT_DISCRIMINATION"),
                        "H1.3": gate(delta, 0, "H1.3_RESTRICTION_PERSISTENCE")}
    arrays_path = out / "bootstrap_arrays.npz"
    np.savez_compressed(arrays_path, **all_arrays)
    arrays_path.chmod(0o400)
    results["bootstrap"] = {"draws": N_BOOT, "seed": SEED, "arrays": source.io.identity(arrays_path)}
    results["source_summary"] = contract["source_summary"]
    source.seal_json(out / "results.json", results)
    rows = [{"representation": rep, "estimand": key, **v, "ci_low": v["ci95"][0], "ci_high": v["ci95"][1]}
            for rep in summary["representations"] for key, v in results[rep]["pooled"].items()]
    pd.DataFrame(rows).drop(columns="ci95").to_csv(out / "performance_and_fidelity.csv", index=False)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=source.OUT)
    p.add_argument("--output-root", type=Path, default=OUT)
    p.add_argument("--module3-source-seal", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(evaluate(a.source_dir, a.output_root, a.module3_source_seal)["gates"], indent=2))
