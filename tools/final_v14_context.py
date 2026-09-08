#!/usr/bin/env python3
"""FINAL-v14 fold-mapped abundance/context and attention associations.

The contract is sealed before association computation. Source model checkpoints
must be sealed first; a missing Module-III mount requires a separately sealed
dependency-block receipt. No feature selection, named claim, or performance gate
is supplied by these descriptive association panels.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import final_v14_module1 as m1  # noqa: E402
from tools import final_v14_post_reader as io_tools  # noqa: E402

ROOT = REPO / "reports/reruns/final_v14_additions_20260903"
PRE = ROOT / "e4v_pre_reader"
OUT = ROOT / "e4m1_context"
CONTRACT_PATH = OUT / "analysis_contract_v2.json"
NAMING = ROOT / "e4v_post_reader_xlsx/naming/naming_freeze.json"
COVARIATES = ("oof_score", "kras", "msi_dmmr", "braf", "sex", "age_at_diagnosis",
              "tumor_site_group", "stage_group_major", "source_subcohort")
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260819


def sealed(path: Path) -> dict:
    return m1.read_sealed(path)


def verify_pin(record: dict) -> None:
    io_tools.check_identity(record)


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("BH requires finite p-values within [0,1]")
    order = np.argsort(p, kind="mergesort")
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def hc3_multioutput(X: np.ndarray, Y: np.ndarray, tested: list[int]) -> dict:
    """OLS with independent per-outcome HC3 covariance, equivalent to scalar fits."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0] or not tested:
        raise ValueError("Invalid OLS design/outcome/contrast shapes")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("OLS inputs must be complete and finite")
    if np.linalg.matrix_rank(X) < X.shape[1] or len(X) <= X.shape[1]:
        raise ValueError("OLS design lacks full column rank or residual degrees of freedom")
    pinv = np.linalg.pinv(X)
    coefficient = pinv @ Y
    residual = Y - X @ coefficient
    leverage = np.einsum("np,pn->n", X, pinv)
    if np.any(1 - leverage <= 1e-12):
        raise ValueError("HC3 undefined at unit leverage")
    variances = (residual / (1 - leverage[:, None])) ** 2
    covariance = np.einsum("in,nk,jn->kij", pinv, variances, pinv)
    selected = coefficient[tested]
    tests, pvalues = [], []
    for outcome in range(Y.shape[1]):
        cov = covariance[outcome][np.ix_(tested, tested)]
        b = selected[:, outcome]
        if np.linalg.matrix_rank(cov) < len(tested):
            tests.append(np.nan)
            pvalues.append(np.nan)
        elif len(tested) == 1:
            z = float(b[0] / np.sqrt(cov[0, 0]))
            tests.append(z)
            pvalues.append(float(2 * norm.sf(abs(z))))
        else:
            wald = float(b @ np.linalg.solve(cov, b))
            tests.append(wald)
            pvalues.append(float(chi2.sf(wald, len(tested))))
    reduced = np.delete(X, tested, axis=1)
    reduced_residual = Y - reduced @ np.linalg.lstsq(reduced, Y, rcond=None)[0]
    sse_reduced = np.sum(reduced_residual**2, axis=0)
    partial_r2 = np.divide(sse_reduced - np.sum(residual**2, axis=0), sse_reduced,
                           out=np.full(Y.shape[1], np.nan), where=sse_reduced > 0)
    sd = np.std(Y, axis=0, ddof=1)
    standardized = np.divide(selected, sd[None, :], out=np.full_like(selected, np.nan), where=sd[None, :] > 0)
    return {"coefficient": coefficient, "covariance": covariance, "contrast": selected,
            "standardized_contrast": standardized, "outcome_sd": sd,
            "statistic": np.asarray(tests), "pvalue": np.asarray(pvalues),
            "partial_r2": partial_r2, "leverage": leverage}


def stratified_bootstrap_indices(strata: list[tuple], n_bootstrap: int,
                                 seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    labels = sorted(set(strata))
    groups = [np.array([i for i, key in enumerate(strata) if key == label], dtype=int) for label in labels]
    rng = np.random.default_rng(seed)
    result = np.empty((n_bootstrap, len(strata)), dtype=np.int32)
    for draw in range(n_bootstrap):
        result[draw] = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
    return result


def bootstrap_ols(X: np.ndarray, Y: np.ndarray, tested: list[int],
                  indices: np.ndarray) -> dict[str, np.ndarray]:
    shape = (len(indices), len(tested), Y.shape[1])
    raw, standardized = np.full(shape, np.nan), np.full(shape, np.nan)
    partial_r2 = np.full((len(indices), Y.shape[1]), np.nan)
    reduced = np.delete(X, tested, axis=1)
    for draw, sampled in enumerate(indices):
        design, outcomes = X[sampled], Y[sampled]
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        coefficient = np.linalg.lstsq(design, outcomes, rcond=None)[0]
        raw[draw] = coefficient[tested]
        sd = np.std(outcomes, axis=0, ddof=1)
        standardized[draw] = np.divide(raw[draw], sd[None, :], out=np.full_like(raw[draw], np.nan), where=sd[None, :] > 0)
        full_error = outcomes - design @ coefficient
        base = reduced[sampled]
        reduced_error = outcomes - base @ np.linalg.lstsq(base, outcomes, rcond=None)[0]
        denominator = np.sum(reduced_error**2, axis=0)
        partial_r2[draw] = np.divide(denominator - np.sum(full_error**2, axis=0), denominator,
                                     out=np.full(Y.shape[1], np.nan), where=denominator > 0)
    return {"raw_contrasts": raw, "standardized_contrasts": standardized, "partial_r2": partial_r2}


def string_values(series: pd.Series) -> pd.Series:
    return series.map(lambda x: None if pd.isna(x) else unicodedata.normalize("NFC", str(x)).strip())


def association_design(frame: pd.DataFrame, variable: str, dictionary: dict) -> dict:
    """Training-free evaluation encoding from the sealed association dictionary."""
    source_levels = dictionary["variables"]["source_subcohort"]["levels"]
    source = string_values(frame["subcohort"])
    source_valid = source.isin(source_levels).to_numpy()
    if variable == "oof_score":
        values = frame["full_logit"].to_numpy(float)
        score_sd = np.std(values, ddof=1)
        if not np.isfinite(score_sd) or score_sd == 0:
            raise ValueError("OOF score cannot be standardized")
        values = (values - np.mean(values)) / score_sd
        levels = None
        kind = "continuous"
    elif variable == "age_at_diagnosis":
        values = pd.to_numeric(frame[variable], errors="coerce").to_numpy(float) / 10.0
        levels = None
        kind = "continuous"
    else:
        spec = dictionary["variables"][variable]
        levels = spec["levels"]
        raw = source if variable == "source_subcohort" else string_values(frame["kras"] if variable == "kras" else frame[variable])
        if variable == "msi_dmmr":
            raw = raw.replace({"MSI/dMMR": "MSI-H/dMMR"})
        kind = spec["type"]
        if kind == "binary":
            values = raw.map({levels[0]: 0.0, levels[1]: 1.0}).to_numpy(float)
        else:
            values = raw.where(raw.isin(levels), None).to_numpy(object)
    valid = source_valid & (pd.notna(values) if levels is not None and kind == "categorical" else np.isfinite(values))
    kept = np.flatnonzero(valid)
    source_kept = source.iloc[kept]
    nuisance_levels = [level for level in source_levels if source_kept.eq(level).any()]
    columns = [np.ones(len(kept))]
    names = ["intercept"]
    if variable != "source_subcohort":
        for level in nuisance_levels[1:]:
            columns.append(source_kept.eq(level).to_numpy(float))
            names.append(f"source_subcohort={level}")
    tested = []
    contrast_names = []
    if kind == "categorical":
        for level in levels[1:]:
            tested.append(len(columns))
            columns.append(np.asarray(values[kept] == level, float))
            contrast_names.append(f"{level} minus {levels[0]}")
            names.append(f"{variable}={level}")
    else:
        tested.append(len(columns))
        columns.append(values[kept].astype(float))
        contrast_names.append(f"{levels[1]} minus {levels[0]}" if kind == "binary" else "per 10 years" if variable == "age_at_diagnosis" else "per OOF-score SD")
        names.append(variable)
    return {"X": np.column_stack(columns), "kept": kept, "tested": tested,
            "feature_names": names, "contrast_names": contrast_names, "kind": kind,
            "reference_level": levels[0] if levels else None,
            "nuisance_source_reference": nuisance_levels[0] if variable != "source_subcohort" and nuisance_levels else None,
            "nuisance_source_levels": nuisance_levels if variable != "source_subcohort" else []}


def aligned_context(frame: pd.DataFrame, mapping: pd.DataFrame,
                    local: pd.DataFrame, attributable: list[int], *,
                    boundary_tolerance: float = 1e-12) -> np.ndarray:
    table = local.set_index("patient_id")
    if not table.index.is_unique or set(table.index) != set(frame.patient_id):
        raise ValueError("Context profile patient identities differ")
    table = table.loc[frame.patient_id]
    if not np.array_equal(table["fold"].to_numpy(int), frame.k_fold.to_numpy(int)):
        raise ValueError("Context profile folds differ")
    matrix = np.full((len(frame), len(attributable)), np.nan)
    for fold in range(5):
        mask = frame.k_fold.eq(fold).to_numpy()
        block = mapping.loc[mapping.outer_fold.eq(fold)].set_index("reference_prototype_id")
        for column, prototype in enumerate(attributable):
            match = block.loc[prototype]
            if float(match.cosine_similarity) < 0.8:
                raise ValueError("Attributable coordinate lacks a qualifying fold match")
            source_coordinate = int(match.source_prototype_id)
            matrix[mask, column] = table[f"prototype_{source_coordinate:02d}"].to_numpy(float)[mask]
    if not np.isfinite(matrix).all() or np.min(matrix) < -boundary_tolerance or np.max(matrix) > 1 + boundary_tolerance:
        raise ValueError("Mapped context proportions invalid")
    return np.arcsin(np.sqrt(np.clip(matrix, 0, 1)))


def finite_interval(values: np.ndarray) -> dict:
    finite = np.asarray(values)[np.isfinite(values)]
    return {"status": "ESTIMABLE" if len(finite) else "CI_NOT_ESTIMABLE",
            "finite_draws": len(finite), "undefined_draws": len(values) - len(finite),
            "ci95": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None}


def source_gate(module3_receipt: Path) -> dict:
    summary_path = ROOT / "e4m1_source/source_fit_summary.json"
    summary = sealed(summary_path)
    if summary["status"] != "SOURCE_MODELS_SEALED":
        raise ValueError("Module-I source predictions are not sealed")
    verify_pin(summary["predictions"])
    third = sealed(module3_receipt)
    if third.get("status") not in {"SOURCE_MODELS_SEALED", "SOURCE_FITS_SEALED", "SOURCE_FITS_COMPLETE", "MODULE_III_DEPENDENCY_BLOCKED", "BLOCKED_MISSING_SEALED_AIM3_INPUTS"}:
        raise ValueError("Require sealed Module-III source fits or explicit dependency-block status")
    if third.get("status") == "BLOCKED_MISSING_SEALED_AIM3_INPUTS":
        if third.get("scientific_status") != "PENDING_NOT_TESTED" or third.get("scientific_gate_adjudicated") is not False:
            raise ValueError("Module-III dependency status must preserve scientifically untested status")
        evaluation_contract = sealed(ROOT / "e4m1_results/evaluation_contract.json")
        if not evaluation_contract.get("dependency_ordering_amendment"):
            raise ValueError("Independent Module-I evaluation lacks a sealed ordering amendment")
    return {"module_i_summary": io_tools.identity(summary_path),
            "module_iii_receipt": io_tools.identity(module3_receipt),
            "module_iii_status": third["status"]}


def prepare(module3_receipt: Path) -> dict:
    path = CONTRACT_PATH
    if path.exists():
        contract = sealed(path)
        for record in contract["input_identities"] + contract["code_identities"]:
            verify_pin(record)
        source_gate(module3_receipt)
        return contract
    gates = source_gate(module3_receipt)
    names = sealed(NAMING)
    source_contract = sealed(ROOT / "e4m1_source/source_fit_contract.json")
    normalization = source_contract["supplement_to_analysis_dictionary"]
    if normalization["msi_normalization"] != {"MSI/dMMR": "MSI-H/dMMR"}:
        raise ValueError("MSI alias normalization is not source-frozen")
    inputs = [NAMING, PRE / "analysis_dictionary.json", PRE / "inputs/tcga_surgen_primary.csv",
              PRE / "mappings/outer_to_reference.csv", PRE / "profiles/oof_patient_profiles.parquet",
              PRE / "teachers/oof_attention_patients_5seed_mean.parquet",
              PRE / "teachers/oof_native_logits_recomputed.parquet", ROOT / "e4m1_source/source_fit_contract.json",
              Path(gates["module_i_summary"]["path"]), module3_receipt,
              ROOT / "e4m1_results/evaluation_contract.json"]
    inputs += [ROOT / f"e4m1_source/fits/ALL32/all/fold_{fold}.json" for fold in range(5)]
    contract = {"status": "CONTEXT_ANALYSIS_CONTRACT_SEALED", "created_utc": io_tools.now(),
                "source_gates": gates, "input_identities": [io_tools.identity(p) for p in inputs],
                "code_identities": [io_tools.identity(Path(__file__)), io_tools.identity(Path(m1.__file__)),
                                    io_tools.identity(Path(io_tools.__file__)), io_tools.identity(REPO / "uv.lock")],
                "bootstrap_draws": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap": "2000 patient resamples within source-subcohort x KRAS complete-case strata, each OLS refitted; common indices across all prototypes for each covariate, same indices for abundance/attention when covariate rosters match; no redraw of non-estimable samples.",
                "response_transform": "asin(sqrt(equal-slide patient proportion)); fold-specific heldout profiles reindexed to qualifying reference coordinates only",
                "covariates": list(COVARIATES), "attention_covariates": ["oof_score", "kras"],
                "ATTRIBUTABLE_REF": names["ATTRIBUTABLE_REF"], "name_gate_status": names["name_gate_status"],
                "test": "OLS HC3 Wald z (single coefficient) or HC3 joint Wald chi-square (multilevel); BH separately by covariate and representation across attributable prototypes",
                "scaling": "OOF score standardized once using complete full-source sample SD; age per 10 years; adjusted effects divided by transformed-outcome sample SD; bootstrap outcome SD recomputed per draw",
                "nuisance_rank_policy": "Within complete cases, remove absent source levels and drop the first observed source category. This spans the same source fixed-effect nuisance space; effective nuisance reference is disclosed. Tested covariate contrasts retain frozen dictionary references.",
                "pre_association_numerical_amendment": {
                    "prior_contract": io_tools.identity(OUT / "analysis_contract.json"),
                    "archived_initial_runner": io_tools.identity(OUT / "source_archive/initial_runner.py"),
                    "prior_execution": "Stopped at input validation before any association fit or bootstrap; one frozen attention mass was 1.000000006046733.",
                    "policy": "Attention uses the pre-reader numerical tolerance 2e-6 and clips boundary roundoff to [0,1] before asin(sqrt). Abundance retains 1e-12. No profile renormalization, data selection, or scientific model change.",
                },
                "nonestimability": "Rank-deficient design or undefined HC3 receives explicit NOT_ESTIMABLE; non-estimable prototype p=1 occupies the fixed BH family, with no effect imputation. Bootstrap rank-deficient samples remain undefined and counts are reported.",
                "missingness": "Complete cases per covariate; missing/unrecognized values remain missing; source-frozen MSI alias conversion only; other strings Unicode NFC and whitespace trim.",
                "named_claims": "NONE for this failed-name-gate campaign; attention is always reported under prototype codes."}
    io_tools.publish(path, contract)
    return contract


def coefficient_stability(mapping: pd.DataFrame, prototypes: list[int]) -> list[dict]:
    values = {prototype: {"ridge": [], "logistic": []} for prototype in prototypes}
    for fold in range(5):
        checkpoint = sealed(ROOT / f"e4m1_source/fits/ALL32/all/fold_{fold}.json")
        coordinates = checkpoint["coordinates"]
        block = mapping.loc[mapping.outer_fold.eq(fold)].set_index("reference_prototype_id")
        for prototype in prototypes:
            position = coordinates.index(int(block.loc[prototype, "source_prototype_id"]))
            for model_name, model in checkpoint["models"].items():
                family = "logistic" if model_name == "concept" else "ridge" if model_name.startswith("ridge_seed") else None
                if family is not None and model["status"] == "ESTIMABLE":
                    values[prototype][family].append(float(model["coef"][position]))
    result = []
    for prototype, families in values.items():
        record = {"prototype_id": prototype}
        for family, coefficients in families.items():
            record[family] = {"median_standardized_coefficient": float(np.median(coefficients)) if coefficients else None,
                              "positive": sum(x > 0 for x in coefficients), "negative": sum(x < 0 for x in coefficients),
                              "zero": sum(x == 0 for x in coefficients), "available": len(coefficients),
                              "expected": 25 if family == "ridge" else 5, "coefficients": coefficients}
        result.append(record)
    return result


def run(module3_receipt: Path) -> dict:
    contract = prepare(module3_receipt)
    summary_path = OUT / "results.json"
    if summary_path.exists():
        summary = sealed(summary_path)
        for record in summary["artifacts"].values():
            verify_pin(record)
        return summary
    frame = m1.patient_frame(PRE)
    full = pd.read_parquet(PRE / "teachers/oof_native_logits_recomputed.parquet").set_index("patient_id")
    frame["full_logit"] = full.loc[frame.patient_id, "mean_logit_5seed"].to_numpy(float)
    mapping = pd.read_csv(PRE / "mappings/outer_to_reference.csv")
    dictionary = sealed(PRE / "analysis_dictionary.json")
    prototypes = contract["ATTRIBUTABLE_REF"]
    transformed = {
        "abundance": aligned_context(frame, mapping, pd.read_parquet(PRE / "profiles/oof_patient_profiles.parquet"), prototypes),
        "attention": aligned_context(frame, mapping, pd.read_parquet(PRE / "teachers/oof_attention_patients_5seed_mean.parquet"), prototypes, boundary_tolerance=2e-6),
    }
    artifacts, rows = {}, []
    with threadpool_limits(limits=1):
        for representation in ("abundance", "attention"):
            for variable in COVARIATES if representation == "abundance" else COVARIATES[:2]:
                start = time.monotonic()
                checkpoint = OUT / "panels" / f"{representation}__{variable}.json"
                if checkpoint.exists():
                    panel = sealed(checkpoint)
                    if panel["contract_sha256"] != io_tools.identity(CONTRACT_PATH)["sha256"]:
                        raise ValueError("Panel contract drift")
                    verify_pin(panel["bootstrap_artifact"])
                else:
                    design = association_design(frame, variable, dictionary)
                    X, kept, tested = design.pop("X"), design.pop("kept"), design.pop("tested")
                    Y = transformed[representation][kept]
                    strata = list(zip(frame.iloc[kept].subcohort.astype(str), frame.iloc[kept].label.astype(int)))
                    indices = stratified_bootstrap_indices(strata, N_BOOTSTRAP)
                    boots = bootstrap_ols(X, Y, tested, indices)
                    arrays = {**boots, "patient_indices": kept[indices], "prototype_ids": np.asarray(prototypes),
                              "complete_case_patient_ids": frame.iloc[kept].patient_id.to_numpy(str)}
                    buffer = io.BytesIO()
                    np.savez_compressed(buffer, **arrays)
                    bootstrap_path = checkpoint.with_suffix(".bootstrap.npz")
                    io_tools.write_once(bootstrap_path, buffer.getvalue())
                    try:
                        fitted = hc3_multioutput(X, Y, tested)
                        fit_error = None
                    except ValueError as error:
                        fitted, fit_error = None, str(error)
                    pvalues = np.ones(len(prototypes)) if fitted is None else np.where(np.isfinite(fitted["pvalue"]), fitted["pvalue"], 1.0)
                    qvalues = bh_adjust(pvalues)
                    panel_rows = []
                    for index, prototype in enumerate(prototypes):
                        estimable = fitted is not None and np.isfinite(fitted["pvalue"][index]) and fitted["outcome_sd"][index] > 0
                        contrasts = []
                        for c, name in enumerate(design["contrast_names"]):
                            contrasts.append({"contrast": name,
                                "adjusted_coefficient": float(fitted["contrast"][c, index]) if estimable else None,
                                "standardized_effect": float(fitted["standardized_contrast"][c, index]) if estimable else None,
                                "adjusted_coefficient_bootstrap": finite_interval(boots["raw_contrasts"][:, c, index]),
                                "standardized_effect_bootstrap": finite_interval(boots["standardized_contrasts"][:, c, index])})
                        panel_rows.append({"representation": representation, "covariate": variable, "prototype_id": prototype,
                                           "status": "ESTIMABLE" if estimable else "NOT_ESTIMABLE", "not_estimable_reason": fit_error if fitted is None else None,
                                           "n_complete_cases": len(kept), "n_missing": len(frame) - len(kept),
                                           "hc3_statistic": float(fitted["statistic"][index]) if estimable else None,
                                           "hc3_distribution": "standard_normal_two_sided" if len(tested) == 1 else "chi_square",
                                           "tested_degrees_of_freedom": len(tested), "p_value": float(pvalues[index]), "q_value_bh": float(qvalues[index]),
                                           "bh_family_size": len(prototypes), "contrasts": contrasts,
                                           "partial_r2": float(fitted["partial_r2"][index]) if estimable else None,
                                           "partial_r2_bootstrap": finite_interval(boots["partial_r2"][:, index])})
                    panel = {"status": "CONTEXT_PANEL_COMPLETE", "created_utc": io_tools.now(),
                             "contract_sha256": io_tools.identity(CONTRACT_PATH)["sha256"],
                             "representation": representation, "covariate": variable, "design": design,
                             "n_complete_cases": len(kept), "rows": panel_rows,
                             "bootstrap_artifact": io_tools.identity(bootstrap_path), "elapsed_seconds": time.monotonic() - start}
                    io_tools.publish(checkpoint, panel)
                artifacts[f"{representation}__{variable}"] = io_tools.identity(checkpoint)
                artifacts[f"{representation}__{variable}__bootstrap"] = panel["bootstrap_artifact"]
                rows.extend(panel["rows"])
                print(json.dumps({"representation": representation, "covariate": variable, "seconds": time.monotonic() - start}), flush=True)
    stability_path = OUT / "coefficient_stability.json"
    io_tools.publish(stability_path, {"status": "DESCRIPTIVE_STABILITY_COMPLETE", "rows": coefficient_stability(mapping, prototypes),
                                    "interpretation": "Descriptive standardized coefficient signs/medians; cannot select predictors or determine gates."})
    artifacts["coefficient_stability"] = io_tools.identity(stability_path)
    table = pd.DataFrame([{k: v for k, v in row.items() if k not in ("contrasts", "partial_r2_bootstrap")} for row in rows])
    table_path = OUT / "association_tests.csv"
    io_tools.write_once(table_path, table.to_csv(index=False).encode())
    artifacts["association_tests"] = io_tools.identity(table_path)
    summary = {"status": "CONTEXT_ASSOCIATIONS_COMPLETE", "created_utc": io_tools.now(),
               "analysis_contract": io_tools.identity(CONTRACT_PATH), "artifacts": artifacts,
               "prototypes": prototypes, "n_covariates_abundance": len(COVARIATES), "n_covariates_attention": 2,
               "n_test_rows": len(rows), "bootstrap_draws_per_panel": N_BOOTSTRAP,
               "name_gate_status": contract["name_gate_status"], "source_gates": contract["source_gates"],
               "claim_limit": "Source-transductive semantic context associations and coefficient stability, not mechanism, confounding explanation, or feature selection. All outputs remain under prototype codes; no attention naming claims."}
    io_tools.publish(summary_path, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--module3-receipt", type=Path, required=True)
    arguments = parser.parse_args()
    result = prepare(arguments.module3_receipt) if arguments.stage == "prepare" else run(arguments.module3_receipt)
    print(json.dumps({"status": result["status"], "output_root": str(OUT)}, indent=2))
