"""Independent numerical replay of completed, sealed Module II artifacts.

This audit never calls the production scoring or statistical functions.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import false_discovery_control
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports/reruns/final_v14_additions_20260903"
OUT = RUN / "e4m2_source_frozen"
AUDIT = RUN / "e4m2_independent_audit"
COLS = [f"prototype_{j:02}" for j in range(32)]
COHORTS = ["cptac_primary", "rih_primary", "orion_cpht", "rih_metastatic", "sr1482_metastatic"]
CHECKS: dict[str, dict] = {}


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(item: dict) -> Path:
    path = Path(item["path"])
    actual = identity(path)
    assert all(actual[k] == item[k] for k in ["size_bytes", "sha256"]), path
    return path


def sealed(path: Path) -> dict:
    seal = json.loads(Path(str(path) + ".seal.json").read_text())
    verify(seal.get("artifact", seal))
    return json.loads(path.read_text())


def close(name: str, actual, expected, tolerance=1e-12) -> None:
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.allclose(a, b, rtol=0, atol=tolerance, equal_nan=True), name
    finite = np.isfinite(a) & np.isfinite(b)
    CHECKS[name] = {"status": "PASS", "values": int(a.size), "absolute_tolerance": tolerance,
                    "max_absolute_error": float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0}


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name, float_precision="round_trip")


def quantile(x) -> np.ndarray:
    return np.quantile(x, [.025, .975], axis=0)


def replay_auc(frame: pd.DataFrame, rng: np.random.Generator, stratify: bool) -> dict:
    """Weighted pair comparisons, independent of production rank-sum AUROC."""
    y = frame.label.to_numpy(int)
    strata = frame.cohort.to_numpy() if stratify else np.repeat("all", len(y))
    groups = [np.flatnonzero((strata == s) & (y == k)) for s in sorted(set(strata)) for k in [0, 1]]
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    kernels = {}
    for key, col in [("ALL32", "logit_ALL32"), ("FULL", "full_logit")]:
        values = frame[col].to_numpy(float)
        delta = values[positive, None] - values[None, negative]
        kernels[key] = (delta > 0).astype(float) + .5 * (delta == 0)
    result = {k: np.empty(10000) for k in kernels}
    for start in range(0, 10000, 250):
        draws = np.concatenate([rng.choice(g, size=(250, len(g))) for g in groups], axis=1)
        counts = np.stack([np.bincount(row, minlength=len(y)) for row in draws])
        for key, kernel in kernels.items():
            result[key][start:start + 250] = np.einsum(
                "bi,ij,bj->b", counts[:, positive], kernel, counts[:, negative], optimize=True) / (len(positive) * len(negative))
    return result


def check_pair(name: str, frame: pd.DataFrame, result: dict, draws: dict) -> None:
    ac = roc_auc_score(frame.label, frame.logit_ALL32)
    af = roc_auc_score(frame.label, frame.full_logit)
    close(name + "/point", [ac, af, ac-af], [result[k] for k in ["concept_auroc", "full_auroc", "concept_minus_full"]])
    bc, bf = draws["ALL32"], draws["FULL"]
    for metric, values in [("concept", bc), ("full", bf), ("concept_minus_full", bc-bf)]:
        close(name + "/" + metric + "_ci", quantile(values), result[metric + "_ci95"])
    ratio = result["retention_ratio"]
    finite = bf > .5
    assert ratio["finite_draws"] == int(finite.sum())
    assert ratio["undefined_draws"] == int((~finite).sum())
    close(name + "/ratio_point", (ac-.5)/(af-.5), ratio["point"])
    if finite.sum() >= 9500:
        assert ratio["status"] == "ESTIMABLE"
        close(name + "/ratio_ci", quantile((bc[finite]-.5)/(bf[finite]-.5)), ratio["ci95"])
    else:
        assert ratio["status"] == "RATIO_CI_NOT_ESTIMABLE" and ratio["ci95"] is None


def main() -> None:
    bundle = sealed(OUT / "source_scoring_bundle.json")
    contract = sealed(OUT / "target_analysis_contract.json")
    score = sealed(OUT / "target_score_seal.json")
    result = sealed(OUT / "target_results.json")
    for item in bundle["code"] + bundle["source_inputs"] + contract["implementation"] + score["artifacts"] + result["artifacts"]:
        verify(item)
    assert score["outcome_columns_read"] == 0
    assert bundle["created_utc"] < score["created_utc"] < result["created_utc"]
    assert set(bundle["models"]) == {"ALL32"} and result["name_gate_status"] == "NAME_GATE_FAIL"
    model = bundle["models"]["ALL32"]
    profiles, logits, joined, ood = (csv(f) for f in ["target_profiles.csv", "target_logits.csv", "target_outcome_join.csv", "target_ood.csv"])
    replay = np.einsum("ij,j->i", (profiles[COLS].to_numpy()-model["scaler_mean"])/model["scaler_scale"], model["coef"]) + model["intercept"]
    close("target_logits", replay, logits.logit_ALL32)
    assert len(joined) == 446 and not joined.duplicated(["cohort", "patient_id"]).any()
    full = pd.read_parquet(verify(result["inherited_full_scores"]))
    full = full[(full.analysis_family == "target_refit_zero_shot_or_sensitivity") & (full.encoder == "univ1") & full.dataset.isin(COHORTS)].sort_values(["dataset", "patient_id"])
    assert np.array_equal(joined.label, full.label) and np.array_equal(joined.patient_id, full.patient_id)
    close("full_five_seed_mean", full[[f"logit_seed{s}" for s in range(42,47)]].mean(axis=1), joined.full_logit)
    shards = [sealed(p) for p in sorted((OUT / "target_shards").rglob("*.json")) if not p.name.endswith(".seal.json")]
    assert len(shards) == 479
    by_patient = {}
    for shard in shards:
        assert shard["bundle_sha256"] == identity(OUT / "source_scoring_bundle.json")["sha256"]
        by_patient.setdefault((shard["cohort"], shard["patient_id"]), []).append(shard)
    aggregated, diagnostic = [], []
    for row in profiles.itertuples(index=False):
        slides = by_patient[(row.cohort, row.patient_id)]
        assert len(slides) == row.n_slides
        aggregated.append(np.mean([s["abundance"] for s in slides], axis=0))
        for j in range(32):
            cells = [s["diagnostics"][j] for s in slides]
            supported = [c for c in cells if c["assigned_tiles"]]
            diagnostic.append([sum(c["assigned_tiles"] for c in cells), len(supported),
                               np.mean([c["fraction_beyond_source_q99"] for c in supported]) if supported else np.nan,
                               np.mean([c["median_distance"] for c in supported]) if supported else np.nan])
    close("equal_slide_profiles", aggregated, profiles[COLS])
    close("patient_ood", diagnostic, ood[["assigned_tiles", "nonempty_slides", "fraction_beyond_source_q99", "median_distance"]])
    labeled = ood.merge(joined[["cohort", "patient_id", "label"]], on=["cohort", "patient_id"], validate="many_to_one")
    summary = csv("target_ood_by_kras.csv").set_index(["cohort", "label", "prototype_id"])
    for key, block in labeled.groupby(["cohort", "label", "prototype_id"]):
        row = summary.loc[key]
        supported = block[block.nonempty_slides > 0]
        close("ood_class/" + str(key), [len(block), len(supported), len(block)-len(supported), supported.fraction_beyond_source_q99.mean(), supported.median_distance.mean()], row[["patients", "supported_patients", "unsupported_patients", "mean_fraction_beyond_source_q99", "mean_median_distance"]])
    # Independent direct Euclidean assignment check on first patient/slide in each cohort.
    pack = Path(bundle["pack_root"])
    index = pd.read_parquet(pack / "index.parquet").set_index("slide_id")
    meta = json.loads((pack / "meta.json").read_text())
    features = np.memmap(pack / "features.bin", mode="r", dtype=meta["feat_dtype"], shape=(meta["total_patches"], meta["feat_dim"]))
    vocab = np.load(verify(bundle["reference_vocabulary"]))
    q99 = pd.read_csv(verify(bundle["source_quantiles"])).sort_values("prototype_id").q99_distance.to_numpy()
    tile_checks = []
    for cohort in COHORTS:
        roster = pd.read_csv(verify(bundle["target_rosters"][cohort])).sort_values(["patient_id", "slide_id"])
        actual = {(s["slide_id"], s["patient_id"]) for s in shards if s["cohort"] == cohort}
        assert actual == set(zip(roster.slide_id, roster.patient_id))
        row = roster.iloc[0]
        shard = next(s for s in shards if s["cohort"] == cohort and s["slide_id"] == row.slide_id)
        entry = index.loc[row.slide_id]
        x = np.array(features[int(entry.offset):int(entry.offset + entry.n_patches)], dtype=np.float32)
        z = (x / np.linalg.norm(x, axis=1)[:, None] - vocab["pca_mean"]) @ vocab["pca_components"].T
        distances = cdist(z.astype(float), vocab["centroids"].astype(float))
        ids = distances.argmin(axis=1)
        minimum = distances[np.arange(len(ids)), ids]
        counts = np.bincount(ids, minlength=32)
        close("tile_assignment/"+cohort, counts/len(ids), shard["abundance"], 0)
        for j in range(32):
            if counts[j]:
                values = minimum[ids == j]
                close(f"tile_distance/{cohort}/{j}", np.median(values), shard["diagnostics"][j]["median_distance"], 2e-6)
                close(f"tile_q99/{cohort}/{j}", np.mean(values > q99[j]), shard["diagnostics"][j]["fraction_beyond_source_q99"], 0)
        tile_checks.append({"cohort": cohort, "slide_id": row.slide_id, "tiles": len(ids)})
    rng = np.random.Generator(np.random.PCG64(20260828))
    stored = np.load(OUT / "performance_bootstrap_draws.npz")
    reports = result["representations"]["ALL32"]
    for name in COHORTS + ["concatenated_247"]:
        frame = joined[joined.cohort.isin(COHORTS[:2])] if name == "concatenated_247" else joined[joined.cohort == name]
        draws = replay_auc(frame, rng, name == "concatenated_247")
        for rep, values in draws.items():
            close(f"bootstrap/{name}/{rep}", values, stored[f"{name}__{rep}__auroc"], 0)
        report = reports["concatenated_247_continuity_only"] if name == "concatenated_247" else reports["cohorts"][name]
        check_pair(name, frame, report, draws)
    for rep, metric in [("ALL32", "concept"), ("FULL", "full")]:
        values = .5*(stored[f"cptac_primary__{rep}__auroc"]+stored[f"rih_primary__{rep}__auroc"])
        close("macro/"+rep, values, stored[f"equal_cohort_macro__{rep}__auroc"], 0)
        close("macro_ci/"+rep, quantile(values), reports["equal_cohort_macro"][metric+"_ci95"])
    macro_gap = stored["equal_cohort_macro__ALL32__auroc"] - stored["equal_cohort_macro__FULL__auroc"]
    close("macro_gap", macro_gap, stored["equal_cohort_macro__ALL32__concept_minus_full"], 0)
    close("macro_gap_ci", quantile(macro_gap), reports["equal_cohort_macro"]["concept_minus_full_ci95"])
    assert not all(reports["cohorts"][c]["concept_ci95"][0] > .5 for c in COHORTS[:2])
    assert result["gates"]["ALL32"] == "ALL32_CROSS_COHORT_DISCRIMINATION_NOT_ESTABLISHED"
    assert result["gates"]["NAMED_REF"] == "NAMED_CROSS_COHORT_DISCRIMINATION_NOT_EVALUABLE"
    primary = pd.read_parquet(verify(bundle["source_profiles"]))
    labels = pd.read_csv(verify(bundle["source_labels"])).drop_duplicates("patient_id")
    primary = primary[primary.subcohort == "SR1482"].merge(labels[["patient_id", "target_label"]], on="patient_id").rename(columns={"target_label": "label"})
    primary["role"] = "primary"
    rih = joined[joined.cohort.isin(["rih_primary", "rih_metastatic"])].copy()
    dual = set(rih[rih.cohort == "rih_primary"].patient_id) & set(rih[rih.cohort == "rih_metastatic"].patient_id)
    assert len(dual) == 8
    rih = rih[~rih.patient_id.isin(dual)].copy()
    rih["role"] = np.where(rih.cohort == "rih_primary", "primary", "metastatic")
    met = joined[joined.cohort == "sr1482_metastatic"].copy()
    met["role"] = "metastatic"
    weights = np.array(model["coef"])/model["scaler_scale"]
    rng = np.random.Generator(np.random.PCG64(20260829))
    interaction_summary = {}
    for family, panel in [("RIH", rih), ("SR1482", pd.concat([primary, met], ignore_index=True))]:
        panel = panel.sort_values("patient_id")
        assert not panel.patient_id.duplicated().any()
        x, y, role = panel[COLS].to_numpy(), panel.label.to_numpy(int), panel.role.to_numpy()
        groups = [np.flatnonzero((role == r) & (y == k)) for r in ["primary", "metastatic"] for k in [0,1]]
        means = np.stack([x[g].mean(axis=0) for g in groups])
        d = means[3]-means[1]-means[2]+means[0]
        draws = np.load(OUT / f"metastatic_interaction_draws_{family}.npz")
        report = csv(f"metastatic_interactions_{family}.csv").sort_values("prototype_id")
        close(family+"/means", means.T, report[["primary_WT_mean", "primary_mutant_mean", "metastatic_WT_mean", "metastatic_mutant_mean"]])
        close(family+"/interaction", d, report.interaction)
        close(family+"/contribution", d*weights, report.fixed_logit_contribution)
        boot = np.empty((10000,4,32))
        for start in range(0,10000,250):
            for j,g in enumerate(groups):
                sampled = rng.choice(g, size=(250,len(g)))
                counts = np.stack([np.bincount(s, minlength=len(x)) for s in sampled])
                boot[start:start+250,j] = counts @ x / len(g)
        close(family+"/bootstrap_means", boot, draws["bootstrap_cell_means"])
        bd = boot[:,3]-boot[:,1]-boot[:,2]+boot[:,0]
        close(family+"/bootstrap_interactions", bd, draws["bootstrap_interactions"])
        close(family+"/bootstrap_logit", np.einsum("ij,j->i", bd, weights), draws["bootstrap_logit_separation_change"])
        close(family+"/interaction_ci", quantile(bd).T, report[["interaction_ci95_low", "interaction_ci95_high"]])
        close(family+"/logit_ci", quantile(bd @ weights), result["interactions"][family]["logit_separation_change_ci95"])
        exceed = np.zeros(32, int)
        for _ in range(10000):
            shifts = []
            for k in [0,1]:
                permuted = rng.permutation(np.flatnonzero(y == k))
                n = int(np.sum((role == "metastatic") & (y == k)))
                shifts.append(x[permuted[:n]].mean(axis=0)-x[permuted[n:]].mean(axis=0))
            # Reproduce arithmetic grouping of the preregistered observed statistic.
            observed = (means[3]-means[1])-(means[2]-means[0])
            exceed += np.abs(shifts[1]-shifts[0]) >= np.abs(observed)
        close(family+"/permutation_counts", exceed, draws["permutation_exceedance_counts"], 0)
        pvalues = (1+exceed)/10001
        close(family+"/permutation_p", pvalues, report.permutation_p, 0)
        close(family+"/BH32", false_discovery_control(pvalues, method="bh"), report.bh_q)
        interaction_summary[family] = {"patients": len(panel), "role_class_cell_sizes": [len(g) for g in groups],
                                       "bh_q_below_0_05": int((report.bh_q < .05).sum())}
    AUDIT.mkdir(exist_ok=True)
    receipt = {"status": "INDEPENDENT_MODULE_II_NUMERICAL_AUDIT_PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
               "implementation": identity(Path(__file__)), "results": identity(OUT / "target_results.json"),
               "score_seal": identity(OUT / "target_score_seal.json"), "checks": CHECKS,
               "independent_auc_method": "Weighted positive-negative pair comparisons; sklearn point estimates",
               "independent_BH_method": "scipy.stats.false_discovery_control(method=bh)",
               "independent_assignment_method": "Direct float64 Euclidean distances after frozen float32 PCA, first listed slide per cohort",
               "assignment_sample": tile_checks, "interaction_summary": interaction_summary,
               "limitations": "Tile distance replay samples five slides; all 479 slide checkpoints, all 446 patient profiles/logits, every saved performance draw and every interaction resample/permutation were checked."}
    path = AUDIT / "numerical_audit_receipt.json"
    assert not path.exists(), "Audit already published; preserve original receipt"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    Path(str(path)+".seal.json").write_text(json.dumps(identity(path), indent=2)+"\n")
    print(json.dumps({"status": receipt["status"], "checks": len(CHECKS), "output": str(path), "interactions": interaction_summary}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
