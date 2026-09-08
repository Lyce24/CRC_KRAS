"""Preregistered paired target analysis for FINAL-v14 Module II.

`prepare` seals analysis code before target scoring. `analyze` will not open
outcome-bearing input until every label-blind target output is sealed.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import final_v14_module2 as scoring  # noqa: E402

N_BOOT = 10000
PERFORMANCE_SEED = 20260828
INTERACTION_SEED = 20260829
CONTROLLING = ("cptac_primary", "rih_primary")
FULL_SCORES = {
    "path": "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim1_tcga_surgen_two_encoder_v1_20260824/downstream_v2/continuation_v3/analysis/patient_native_logits.parquet",
    "size_bytes": 311419,
    "sha256": "02f4a5bf0c2cd909c5ecb3f9edc7c05ae725649199a015cf0a1d0cb544df8d09",
}
CLAIM_SCOPE = "Locked source-frozen secondary analysis of previously opened archived cohorts; no fresh external validation, causal claim, or AUROC decomposition."


def auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(auc_batch(np.asarray(y)[None, :], np.asarray(score)[None, :])[0])


def auc_batch(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if y.shape != score.shape or y.ndim != 2 or not np.isfinite(score).all():
        raise ValueError("Aligned finite two-dimensional label/score arrays required")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Binary labels required")
    n1, n0 = y.sum(axis=1), (1 - y).sum(axis=1)
    if (n1 == 0).any() or (n0 == 0).any():
        raise ValueError("AUROC requires both classes")
    ranks = rankdata(score, method="average", axis=1)
    return ((ranks * y).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def strata_indices(y: np.ndarray, strata: np.ndarray | None = None) -> list[np.ndarray]:
    y = np.asarray(y, dtype=int)
    strata = np.repeat("all", len(y)) if strata is None else np.asarray(strata, dtype=str)
    if len(strata) != len(y):
        raise ValueError("Strata/label length mismatch")
    return [np.flatnonzero((strata == s) & (y == k)) for s in sorted(set(strata)) for k in [0, 1]
            if np.any((strata == s) & (y == k))]


def bootstrap_auc_vectors(y: np.ndarray, scores: dict[str, np.ndarray], *, rng: np.random.Generator,
                          n_boot: int = N_BOOT, strata: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """One patient resample is reused for every concept and full score."""
    y = np.asarray(y, dtype=int)
    arrays = {key: np.asarray(value, dtype=float) for key, value in scores.items()}
    if any(value.shape != y.shape for value in arrays.values()):
        raise ValueError("Score vectors must have identical patient rosters")
    groups = strata_indices(y, strata)
    result = {key: np.empty(n_boot) for key in arrays}
    for start in range(0, n_boot, 250):
        size = min(250, n_boot - start)
        indices = np.concatenate([rng.choice(g, size=(size, len(g)), replace=True) for g in groups], axis=1)
        for key, score in arrays.items():
            result[key][start:start + size] = auc_batch(y[indices], score[indices])
    return result


def interval(values: np.ndarray) -> list[float]:
    if not np.isfinite(values).all():
        raise ValueError("Unexpected nonfinite bootstrap value")
    return np.quantile(values, [0.025, 0.975], method="linear").astype(float).tolist()


def ratio_summary(concept: float, full: float, boot_concept: np.ndarray, boot_full: np.ndarray) -> dict:
    denominator = np.asarray(boot_full) - .5
    ratios = np.divide(np.asarray(boot_concept) - .5, denominator,
                       out=np.full_like(denominator, np.nan), where=denominator > 0)
    finite = ratios[np.isfinite(ratios)]
    count = len(finite)
    # The prespecified floor is an absolute 9,500 of 10,000, never redraw.
    estimable = len(ratios) == N_BOOT and count >= 9500
    return {"point": (concept - .5) / (full - .5) if full > .5 else None,
            "finite_draws": count, "undefined_draws": len(ratios) - count,
            "ci95": interval(finite) if estimable else None,
            "status": "ESTIMABLE" if estimable else "RATIO_CI_NOT_ESTIMABLE"}


def summarize_pair(y: np.ndarray, concept: np.ndarray, full: np.ndarray,
                   boot_concept: np.ndarray, boot_full: np.ndarray) -> dict:
    ac, af = auc(y, concept), auc(y, full)
    return {"patients": len(y), "mutant": int(np.sum(y)), "wild_type": int(len(y) - np.sum(y)),
            "concept_auroc": ac, "concept_ci95": interval(boot_concept),
            "concept_one_sided_97_5_lower": float(np.quantile(boot_concept, .025)),
            "full_auroc": af, "full_ci95": interval(boot_full), "concept_minus_full": ac - af,
            "concept_minus_full_ci95": interval(boot_concept - boot_full),
            "retention_ratio": ratio_summary(ac, af, boot_concept, boot_full)}


def hierarchical_gates(results: dict, name_gate: str) -> dict:
    all_rows = results.get("ALL32", {}).get("cohorts", {})
    if not all(c in all_rows for c in CONTROLLING):
        all_status = "ALL32_CROSS_COHORT_DISCRIMINATION_NOT_EVALUABLE"
    elif all(all_rows[c]["concept_one_sided_97_5_lower"] > .5 for c in CONTROLLING):
        all_status = "ALL32_CROSS_COHORT_DISCRIMINATION_SUPPORTED"
    else:
        all_status = "ALL32_CROSS_COHORT_DISCRIMINATION_NOT_ESTABLISHED"
    named_status = "NAMED_CROSS_COHORT_DISCRIMINATION_NOT_EVALUABLE"
    named_rows = results.get("NAMED_REF", {}).get("cohorts", {})
    if (all_status == "ALL32_CROSS_COHORT_DISCRIMINATION_SUPPORTED" and name_gate == "NAME_GATE_PASS"
            and all(c in named_rows for c in CONTROLLING)):
        named_status = ("NAMED_CROSS_COHORT_DISCRIMINATION_SUPPORTED" if all(
            named_rows[c]["concept_one_sided_97_5_lower"] > .5 for c in CONTROLLING)
                        else "NAMED_CROSS_COHORT_DISCRIMINATION_NOT_ESTABLISHED")
    return {"ALL32": all_status, "NAMED_REF": named_status}


def bh(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.minimum.accumulate((pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1))[::-1])[::-1]
    result = np.empty(len(pvalues))
    result[order] = np.minimum(1, adjusted)
    return result


def interaction_panel(frame: pd.DataFrame, model: dict, *, rng: np.random.Generator,
                      n_boot: int = N_BOOT, n_perm: int = N_BOOT,
                      draw_sink: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """All raw-abundance coordinates, class-stratified roles, no target refit."""
    if frame.patient_id.duplicated().any():
        raise ValueError("Interaction populations must be strictly patient-disjoint")
    X, y, role = frame[scoring.COLS].to_numpy(float), frame.label.to_numpy(int), frame.role.to_numpy(str)
    if not np.isfinite(X).all() or set(role) != {"primary", "metastatic"} or set(y) != {0, 1}:
        raise ValueError("Invalid role/class interaction panel")
    group_order = [("primary", 0), ("primary", 1), ("metastatic", 0), ("metastatic", 1)]
    groups = [np.flatnonzero((role == r) & (y == k)) for r, k in group_order]
    if any(not len(g) for g in groups):
        raise ValueError("Each tissue-role/KRAS cell requires support")
    means = np.stack([X[g].mean(axis=0) for g in groups])
    shifts = np.stack([means[2] - means[0], means[3] - means[1]])
    d = shifts[1] - shifts[0]
    boot_means = np.empty((n_boot, 4, 32))
    for start in range(0, n_boot, 250):
        size = min(250, n_boot - start)
        for j, group in enumerate(groups):
            indices = rng.choice(group, size=(size, len(group)), replace=True)
            boot_means[start:start + size, j] = X[indices].mean(axis=1)
    boot_shifts = np.stack([boot_means[:, 2] - boot_means[:, 0], boot_means[:, 3] - boot_means[:, 1]], axis=1)
    boot_d = boot_shifts[:, 1] - boot_shifts[:, 0]
    perm_exceed = np.zeros(32, dtype=int)
    class_groups = [np.flatnonzero(y == k) for k in [0, 1]]
    n_met = [int(np.sum((y == k) & (role == "metastatic"))) for k in [0, 1]]
    for _ in range(n_perm):
        pshifts = []
        for indices, nm in zip(class_groups, n_met, strict=True):
            shuffled = rng.permutation(indices)
            pshifts.append(X[shuffled[:nm]].mean(axis=0) - X[shuffled[nm:]].mean(axis=0))
        perm_exceed += np.abs(pshifts[1] - pshifts[0]) >= np.abs(d)
    pvalues = (1 + perm_exceed) / (n_perm + 1)
    qvalues = bh(pvalues)
    weights = np.zeros(32)
    for j, beta, scale in zip(model["coordinates"], model["coef"], model["scaler_scale"], strict=True):
        weights[j] = beta / scale
    logit = scoring.score_matrix(model, X[:, model["coordinates"]])
    separation = ((logit[groups[3]].mean() - logit[groups[2]].mean()) -
                  (logit[groups[1]].mean() - logit[groups[0]].mean()))
    contributions = weights * d
    if not np.isclose(contributions.sum(), separation, atol=1e-12, rtol=0):
        raise ValueError("Fixed-coefficient separation decomposition does not reproduce direct logits")
    if draw_sink is not None:
        draw_sink.update({"bootstrap_cell_means": boot_means, "bootstrap_class_role_shifts": boot_shifts,
                          "bootstrap_interactions": boot_d, "bootstrap_logit_separation_change": boot_d @ weights,
                          "permutation_exceedance_counts": perm_exceed, "fixed_logit_weights": weights})
    records = []
    for j in range(32):
        row = {"prototype_id": j, "interaction": float(d[j]), "permutation_p": float(pvalues[j]), "bh_q": float(qvalues[j]),
               "interaction_ci95_low": interval(boot_d[:, j])[0], "interaction_ci95_high": interval(boot_d[:, j])[1],
               "fixed_logit_contribution": float(contributions[j]),
               "logit_contribution_ci95_low": interval(boot_d[:, j] * weights[j])[0],
               "logit_contribution_ci95_high": interval(boot_d[:, j] * weights[j])[1]}
        for cell, (r, k) in enumerate(group_order):
            key = f"{r}_{'mutant' if k else 'WT'}"
            ci = interval(boot_means[:, cell, j])
            row.update({f"{key}_n": len(groups[cell]), f"{key}_mean": float(means[cell, j]),
                        f"{key}_ci95_low": ci[0], f"{key}_ci95_high": ci[1]})
        for k in [0, 1]:
            key = f"role_shift_{'mutant' if k else 'WT'}"
            ci = interval(boot_shifts[:, k, j])
            row.update({key: float(shifts[k, j]), f"{key}_ci95_low": ci[0], f"{key}_ci95_high": ci[1]})
        records.append(row)
    summary = {"patients": len(X), "bootstrap_draws": n_boot, "permutation_draws": n_perm,
               "interaction_family_size": 32, "multiplicity": "BH within cohort family, all canonical prototypes",
               "direct_logit_separation_change": float(separation), "sum_fixed_contributions": float(contributions.sum()),
               "logit_separation_change_ci95": interval(boot_d @ weights),
               "interpretation": "Exact decomposition of the linear mean-logit contrast; not a decomposition of AUROC or a causal explanation."}
    return pd.DataFrame(records), summary


def prepare(output: Path) -> dict:
    bundle = scoring.verify_bundle(output, require_snapshot=False)
    path = output / "target_analysis_contract.json"
    if path.exists():
        contract = scoring.read_sealed(path)
        for item in contract["implementation"]:
            scoring.verify_identity(item)
        return contract
    if (output / "target_shards").exists() or (output / "target_score_seal.json").exists():
        raise scoring.ContractError("Cannot first seal analysis rules after target scoring began")
    contract = {"status": "TARGET_ANALYSIS_CONTRACT_SEALED", "created_utc": scoring.now(),
                "source_bundle": scoring.identity(output / "source_scoring_bundle.json"),
                "implementation": [scoring.identity(Path(__file__)), scoring.identity(Path(scoring.__file__)),
                                   scoring.identity(REPO / "tests/test_final_v14_module2_analysis.py")],
                "runtime": scoring.runtime(), "full_scores": FULL_SCORES,
                "name_gate_status": bundle["name_gate_status"], "cohort_order": list(scoring.TARGETS),
                "controlling_cohorts": list(CONTROLLING), "macro_weights": [.5, .5],
                "performance": {"draws": N_BOOT, "seed": PERFORMANCE_SEED, "strata": "cohort x KRAS", "resampling_unit": "patient", "paired_scores": True},
                "interaction": {"bootstrap_draws": N_BOOT, "permutations": N_BOOT, "seed": INTERACTION_SEED,
                                "families": ["RIH", "SR1482"], "strata": "cohort family x role x KRAS",
                                "permutation": "role labels within KRAS class", "bh_family_size": 32,
                                "RIH_dual_role_removed_from_both_arms": 8},
                "gate": "Both conventional-primary cohort one-sided97.5% bootstrap lower AUROC bounds >0.5; ALL32 first, then eligible NAMED_REF",
                "ratio_finite_floor": 9500, "claim_scope": CLAIM_SCOPE,
                "target_adaptation": "FORBIDDEN", "outcome_input_opened": False,
                "entry_command": [sys.executable, str(Path(__file__).resolve()), "analyze", "--output", str(output.resolve())]}
    scoring.write_json(path, contract)
    scoring.seal(path)
    return contract


def _joined_input(output: Path, contract: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    bundle = scoring.verify_bundle(output, require_snapshot=False)
    seal_path = output / "target_score_seal.json"
    sealed = scoring.read_sealed(seal_path)
    if sealed["status"] != "TARGET_PROFILES_AND_LOGITS_SEALED_BEFORE_OUTCOME_JOIN" or sealed["outcome_columns_read"] != 0:
        raise scoring.ContractError("Target scoring/outcome firewall not sealed")
    scoring.verify_identity(sealed["source_bundle"])
    if sealed["source_bundle"]["sha256"] != scoring.digest(output / "source_scoring_bundle.json"):
        raise scoring.ContractError("Target scores came from a different source bundle")
    for item in sealed["artifacts"]:
        scoring.verify_identity(item)
    expected_names = {"target_profiles.csv", "target_logits.csv", "target_ood.csv", "target_ood_before_outcome_join.csv"}
    if {Path(r["path"]).name for r in sealed["artifacts"]} != expected_names:
        raise scoring.ContractError("Incomplete sealed target output set")
    if any(Path(r["path"]).resolve() != (output / Path(r["path"]).name).resolve() for r in sealed["artifacts"]):
        raise scoring.ContractError("Target output artifact path escaped its sealed output directory")
    profiles = pd.read_csv(output / "target_profiles.csv", float_precision="round_trip")
    logits = pd.read_csv(output / "target_logits.csv", float_precision="round_trip")
    ood = pd.read_csv(output / "target_ood.csv", float_precision="round_trip")
    # This is the first opening of the inherited outcome-bearing score artifact.
    full = pd.read_parquet(scoring.verify_identity(contract["full_scores"]))
    full = full[(full.analysis_family == "target_refit_zero_shot_or_sensitivity") & (full.encoder == "univ1") & full.dataset.isin(scoring.TARGETS)].copy()
    seed_columns = [f"logit_seed{s}" for s in range(42, 47)]
    if not np.isfinite(full[seed_columns + ["mean_logit_5seed"]].to_numpy(float)).all():
        raise scoring.ContractError("Frozen full-score artifact contains nonfinite scores")
    if not np.allclose(full.mean_logit_5seed, full[seed_columns].mean(axis=1), atol=1e-15, rtol=0):
        raise scoring.ContractError("Frozen five-refit mean score drift")
    full = full[["dataset", "patient_id", "label", "mean_logit_5seed"]].rename(columns={"dataset": "cohort", "mean_logit_5seed": "full_logit"})
    if not np.isin(full.label.to_numpy(), [0, 1]).all():
        raise scoring.ContractError("Nonbinary inherited target labels")
    full["label"] = full.label.astype(int)
    keys = ["cohort", "patient_id"]
    if full.duplicated(keys).any():
        raise scoring.ContractError("Duplicate paired full-score patient")
    before = len(profiles)
    joined = profiles.merge(logits, on=keys, validate="one_to_one").merge(full, on=keys, validate="one_to_one")
    if len(joined) != before or len(full) != before:
        raise scoring.ContractError("Target paired complete-roster join lost patients")
    expected_mutants = {"cptac_primary": 33, "rih_primary": 70, "orion_cpht": 15, "rih_metastatic": 37, "sr1482_metastatic": 30}
    for c, (_, n) in scoring.CENSUS.items():
        block = joined[joined.cohort == c]
        if len(block) != n or int(block.label.sum()) != expected_mutants[c]:
            raise scoring.ContractError(f"Full-score target outcome census drift: {c}")
    return joined.sort_values(keys).reset_index(drop=True), ood, bundle


def analyze(output: Path) -> dict:
    contract = scoring.read_sealed(output / "target_analysis_contract.json")
    for item in contract["implementation"] + [contract["source_bundle"]]:
        scoring.verify_identity(item)
    if contract["runtime"] != scoring.runtime():
        raise scoring.ContractError("Frozen analysis runtime changed")
    final = output / "target_results.json"
    if final.exists():
        result = scoring.read_sealed(final)
        for item in result["artifacts"]:
            scoring.verify_identity(item)
        return result
    joined, ood, bundle = _joined_input(output, contract)
    reps = [r for r, model in bundle["models"].items() if model["status"] == "ESTIMABLE"]
    if bundle["name_gate_status"] != "NAME_GATE_PASS" and "NAMED_REF" in reps:
        raise scoring.ContractError("Named score forbidden when naming gate fails")
    results = {r: {"cohorts": {}} for r in reps}
    bootstrap = {}
    draw_archive = {}
    rng = np.random.Generator(np.random.PCG64(PERFORMANCE_SEED))
    for c in scoring.TARGETS:
        block = joined[joined.cohort == c]
        y, full = block.label.to_numpy(int), block.full_logit.to_numpy(float)
        scores = {r: block[f"logit_{r}"].to_numpy(float) for r in reps}
        boot = bootstrap_auc_vectors(y, {**scores, "FULL": full}, rng=rng)
        bootstrap[c] = boot
        draw_archive.update({f"{c}__{key}__auroc": value for key, value in boot.items()})
        for r in reps:
            results[r]["cohorts"][c] = summarize_pair(y, scores[r], full, boot[r], boot["FULL"])
    for r in reps:
        a, b = (results[r]["cohorts"][c] for c in CONTROLLING)
        bc = .5 * (bootstrap[CONTROLLING[0]][r] + bootstrap[CONTROLLING[1]][r])
        bf = .5 * (bootstrap[CONTROLLING[0]]["FULL"] + bootstrap[CONTROLLING[1]]["FULL"])
        draw_archive.update({f"equal_cohort_macro__{r}__auroc": bc,
                             "equal_cohort_macro__FULL__auroc": bf,
                             f"equal_cohort_macro__{r}__concept_minus_full": bc - bf})
        results[r]["equal_cohort_macro"] = {"weights": [.5, .5], "concept_auroc": .5 * (a["concept_auroc"] + b["concept_auroc"]),
                                              "full_auroc": .5 * (a["full_auroc"] + b["full_auroc"]),
                                              "concept_ci95": interval(bc), "full_ci95": interval(bf),
                                              "concept_minus_full": .5 * (a["concept_minus_full"] + b["concept_minus_full"]),
                                              "concept_minus_full_ci95": interval(bc - bf)}
    pool = joined[joined.cohort.isin(CONTROLLING)]
    pool_scores = {r: pool[f"logit_{r}"].to_numpy(float) for r in reps}
    pool_boot = bootstrap_auc_vectors(pool.label.to_numpy(int), {**pool_scores, "FULL": pool.full_logit.to_numpy(float)},
                                      rng=rng, strata=pool.cohort.to_numpy(str))
    draw_archive.update({f"concatenated_247__{key}__auroc": value for key, value in pool_boot.items()})
    for r in reps:
        results[r]["concatenated_247_continuity_only"] = summarize_pair(pool.label.to_numpy(int), pool_scores[r],
                                                                           pool.full_logit.to_numpy(float), pool_boot[r], pool_boot["FULL"])
    artifacts = []
    def publish(name: str, frame: pd.DataFrame) -> None:
        path = output / name
        scoring.write_once(path, frame.to_csv(index=False, float_format="%.17g").encode())
        artifacts.append(scoring.identity(path))
    def publish_draws(name: str, arrays: dict) -> None:
        path = output / name
        if path.exists():
            with np.load(path, allow_pickle=False) as stored:
                if set(stored.files) != set(arrays) or any(not np.array_equal(stored[key], value) for key, value in arrays.items()):
                    raise scoring.ContractError(f"Bootstrap replay arrays changed: {path}")
        else:
            stream = io.BytesIO()
            np.savez_compressed(stream, **arrays)
            scoring.write_once(path, stream.getvalue())
        artifacts.append(scoring.identity(path))
    publish_draws("performance_bootstrap_draws.npz", draw_archive)
    publish("target_outcome_join.csv", joined)
    labeled_ood = ood.merge(joined[["cohort", "patient_id", "label"]], on=["cohort", "patient_id"], validate="many_to_one")
    ood_summary = labeled_ood.groupby(["cohort", "label", "prototype_id"], sort=True).agg(
        patients=("patient_id", "size"), supported_patients=("nonempty_slides", lambda x: int((x > 0).sum())),
        mean_fraction_beyond_source_q99=("fraction_beyond_source_q99", "mean"), mean_median_distance=("median_distance", "mean")).reset_index()
    ood_summary["unsupported_patients"] = ood_summary.patients - ood_summary.supported_patients
    publish("target_ood_by_kras.csv", ood_summary)
    abundance = joined.melt(id_vars=["cohort", "patient_id", "label"], value_vars=scoring.COLS, var_name="prototype", value_name="abundance")
    distribution = abundance.groupby(["cohort", "label", "prototype"], sort=True).abundance.agg(
        ["size", "mean", "median", "std", "min", "max"]).reset_index()
    publish("target_prototype_distributions_by_kras.csv", distribution)
    interactions = {}
    if "ALL32" in reps:
        rng = np.random.Generator(np.random.PCG64(INTERACTION_SEED))
        rih = joined[joined.cohort.isin(["rih_primary", "rih_metastatic"])].copy()
        dual = set(rih[rih.cohort == "rih_primary"].patient_id) & set(rih[rih.cohort == "rih_metastatic"].patient_id)
        if len(dual) != 8:
            raise scoring.ContractError("RIH strict-disjoint population drift")
        rih = rih[~rih.patient_id.isin(dual)].copy()
        rih["role"] = np.where(rih.cohort == "rih_primary", "primary", "metastatic")
        source = pd.read_parquet(scoring.verify_identity(bundle["source_profiles"]))
        source = source[source.subcohort == "SR1482"].copy()
        labels = pd.read_csv(scoring.verify_identity(bundle["source_labels"]), usecols=["patient_id", "target_label"])
        if (not np.isin(labels.target_label.to_numpy(), [0, 1]).all()
                or labels.groupby("patient_id").target_label.nunique().max() != 1):
            raise scoring.ContractError("Invalid source labels for SR1482 characterization")
        labels = labels.drop_duplicates("patient_id")
        labels["target_label"] = labels.target_label.astype(int)
        if len(labels) != 1239 or int(labels.target_label.sum()) != 501:
            raise scoring.ContractError("Source-label census drift during metastatic characterization")
        source = source.merge(labels.rename(columns={"target_label": "label"}), on="patient_id", validate="one_to_one")
        source["role"] = "primary"
        srmet = joined[joined.cohort == "sr1482_metastatic"].copy()
        srmet["role"] = "metastatic"
        sr = pd.concat([source, srmet], ignore_index=True)
        for family, panel in (("RIH", rih), ("SR1482", sr)):
            interaction_draws = {}
            interaction, summary = interaction_panel(panel.sort_values("patient_id"), bundle["models"]["ALL32"], rng=rng,
                                                       draw_sink=interaction_draws)
            publish_draws(f"metastatic_interaction_draws_{family}.npz", interaction_draws)
            interactions[family] = summary
            interactions[family]["source_family_exposed"] = family == "SR1482"
            interactions[family]["dual_role_removed_from_both_arms"] = len(dual) if family == "RIH" else 0
            publish(f"metastatic_interactions_{family}.csv", interaction)
    result = {"status": "MODULE_II_ANALYSIS_COMPLETE", "created_utc": scoring.now(), "claim_scope": CLAIM_SCOPE,
              "name_gate_status": bundle["name_gate_status"], "representations": results,
              "gates": hierarchical_gates(results, bundle["name_gate_status"]), "interactions": interactions,
              "bootstrap_draws": N_BOOT, "performance_seed": PERFORMANCE_SEED, "interaction_seed": INTERACTION_SEED,
              "conditioning": "Intervals resample fixed patient predictions and condition on source-selected models/vocabulary",
              "ood_missingness": "No assigned tiles: undefined distance/fraction, zero support; cohort and class means use supported patients only and disclose denominators",
              "named_spotlights": "NOT_EVALUABLE" if bundle["name_gate_status"] != "NAME_GATE_PASS" else "REQUIRES_SEPARATELY_SEALED_CORRESPONDENCE_CRITERIA",
              "artifacts": artifacts, "source_contract": scoring.identity(output / "target_analysis_contract.json"),
              "label_blind_score_seal": scoring.identity(output / "target_score_seal.json"),
              "inherited_full_scores": FULL_SCORES}
    scoring.write_json(final, result)
    scoring.seal(final)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "analyze"])
    parser.add_argument("--output", type=Path, default=scoring.RUN / "e4m2_source_frozen")
    args = parser.parse_args()
    result = (prepare if args.stage == "prepare" else analyze)(args.output.resolve())
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
