#!/usr/bin/env python3
"""Frozen-score worklist-enrichment analysis for the v3 KRAS report.

This add-on trains no image model and never chooses a target-specific cutoff.
It reads the three frozen E0 OOF native-logit files and the authoritative E2a
held-out primary target scores, then asks how many KRAS-mutant patients occur
in worklists with fixed capacities of 10, 20, 30, 40 and 50 percent.

The default destination is append-only.  Existing v2 reports and upstream
artifacts are inputs and are never opened for writing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import baselines, evaluate  # noqa: E402
from oceanpath.eval.external import logit, sigmoid  # noqa: E402


SEEDS = (42, 43, 44)
CAPACITIES = (0.10, 0.20, 0.30, 0.40, 0.50)
PRIMARY_CAPACITY = 0.30
BOOTSTRAP_SEED = 20260820
DEFAULT_N_BOOTSTRAP = 10_000

DEV_MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/aim1_dev.csv")
LABEL_SOURCE = Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v4.csv")
E0_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/train/1a_pb_cap8192/univ1")
E1D_RESULT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/eval/e1d.json")
E2A_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim2_cap8192_v4_20260819/e2a"
)
E2A_SOURCE_CV_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/e2a/source_cv/cap8192")
FINAL_V2 = REPO / "reports" / "final_v2"
DEFAULT_OUTPUT = (
    REPO / "reports" / "reruns" / "final_v3_additions_20260820" / "aim1_worklist"
)
TARGETS = ("CPTAC", "RIH", "SurGen", "TCGA")

NUMERIC = list(baselines.CLINICAL_NUMERIC)
CATEGORICAL = list(baselines.CLINICAL_CATEGORICAL) + [baselines.STAGE_COLUMN]
METRICS = ("capture", "worklist_ppv", "enrichment", "cases_per_mutant")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def stable_seed(name: str) -> int:
    suffix = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 1_000_000
    return BOOTSTRAP_SEED + suffix


def _align(matrix: np.ndarray, names: list[str], expected: list[str]) -> np.ndarray:
    index = {name: position for position, name in enumerate(names)}
    return np.column_stack(
        [matrix[:, index[name]] if name in index else np.zeros(len(matrix)) for name in expected]
    )


@dataclass
class UnweightedLogistic:
    """Exact E1d preprocessing with an unweighted L2 logistic estimator."""

    numeric: list[str]
    categorical: list[str]
    model: LogisticRegression | None = None
    names: list[str] | None = None
    levels: dict[str, list[str]] | None = None
    medians: dict[str, float] | None = None
    scales: dict[str, float] | None = None

    def fit(self, frame: pd.DataFrame) -> "UnweightedLogistic":
        matrix, names, levels, medians, scales = baselines._design(
            frame, self.numeric, self.categorical
        )
        model = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
        model.fit(matrix, frame["label"].to_numpy(dtype=int))
        self.model = model
        self.names = names
        self.levels = levels
        self.medians = medians
        self.scales = scales
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.names is None:
            raise RuntimeError("model is not fitted")
        matrix, names, _, _, _ = baselines._design(
            frame,
            self.numeric,
            self.categorical,
            self.levels,
            self.medians,
            self.scales,
        )
        return self.model.predict_proba(_align(matrix, names, self.names))[:, 1]


def cross_fitted_score(
    frame: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> np.ndarray:
    out = np.full(len(frame), np.nan)
    folds = frame["k_fold"].to_numpy()
    for fold in sorted(pd.unique(folds)):
        held_out = folds == fold
        model = UnweightedLogistic(numeric, categorical).fit(frame.loc[~held_out])
        out[held_out] = model.predict(frame.loc[held_out])
    if not np.isfinite(out).all():
        raise RuntimeError("cross-fitting left patients unscored")
    return out


def selected_n(n: int, capacity: float) -> int:
    """Largest integer worklist that does not exceed its nominal capacity."""
    return max(1, int(math.floor(capacity * n + 1e-12)))


def _tie_groups(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    sorted_score = np.asarray(score, dtype=float)[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    return order, starts


def _weighted_rank_metrics(
    y: np.ndarray,
    score: np.ndarray,
    weights: np.ndarray,
    capacities: Iterable[float] = CAPACITIES,
) -> np.ndarray:
    """Metrics with fractional allocation when a capacity cuts a score tie."""
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = int(round(float(weights.sum())))
    total_mutants = float(np.dot(weights, y))
    order, starts = _tie_groups(score)
    group_n = np.add.reduceat(weights[order], starts)
    group_y = np.add.reduceat((weights * y)[order], starts)
    cumulative_n = np.cumsum(group_n)
    cumulative_y = np.cumsum(group_y)
    prevalence = total_mutants / n if n and total_mutants else np.nan
    rows: list[list[float]] = []
    for capacity in capacities:
        k = selected_n(n, capacity)
        boundary = int(np.searchsorted(cumulative_n, k, side="left"))
        n_above = float(cumulative_n[boundary - 1]) if boundary else 0.0
        y_above = float(cumulative_y[boundary - 1]) if boundary else 0.0
        fraction = (k - n_above) / float(group_n[boundary])
        selected_mutants = y_above + fraction * float(group_y[boundary])
        capture = selected_mutants / total_mutants if total_mutants else np.nan
        ppv = selected_mutants / k
        enrichment = ppv / prevalence if prevalence else np.nan
        cases_per_mutant = k / selected_mutants if selected_mutants else np.nan
        rows.append([capture, ppv, enrichment, cases_per_mutant])
    return np.asarray(rows, dtype=float)


def rank_point(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    return _weighted_rank_metrics(y, score, np.ones(len(y), dtype=float))


def bootstrap_rank(
    y: np.ndarray,
    score: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 250,
) -> np.ndarray:
    """Ordinary patient bootstrap, vectorized over fixed score-tie groups."""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    n = len(y)
    order, starts = _tie_groups(score)
    y_sorted = y[order]
    probabilities = np.full(n, 1.0 / n)
    out = np.full((n_bootstrap, len(CAPACITIES), len(METRICS)), np.nan)
    rng = np.random.default_rng(seed)
    offset = 0
    while offset < n_bootstrap:
        size = min(batch_size, n_bootstrap - offset)
        counts = rng.multinomial(n, probabilities, size=size)
        counts = counts[:, order]
        group_n = np.add.reduceat(counts, starts, axis=1)
        group_y = np.add.reduceat(counts * y_sorted[None, :], starts, axis=1)
        cumulative_n = np.cumsum(group_n, axis=1)
        cumulative_y = np.cumsum(group_y, axis=1)
        total_y = cumulative_y[:, -1]
        prevalence = total_y / n
        for capacity_index, capacity in enumerate(CAPACITIES):
            k = selected_n(n, capacity)
            boundary = np.argmax(cumulative_n >= k, axis=1)
            row = np.arange(size)
            has_above = boundary > 0
            n_above = np.zeros(size)
            y_above = np.zeros(size)
            n_above[has_above] = cumulative_n[row[has_above], boundary[has_above] - 1]
            y_above[has_above] = cumulative_y[row[has_above], boundary[has_above] - 1]
            boundary_n = group_n[row, boundary]
            boundary_y = group_y[row, boundary]
            fraction = (k - n_above) / boundary_n
            selected_y = y_above + fraction * boundary_y
            capture = np.divide(
                selected_y, total_y, out=np.full(size, np.nan), where=total_y > 0
            )
            ppv = selected_y / k
            enrichment = np.divide(
                ppv, prevalence, out=np.full(size, np.nan), where=prevalence > 0
            )
            cases = np.divide(
                k, selected_y, out=np.full(size, np.nan), where=selected_y > 0
            )
            out[offset : offset + size, capacity_index, :] = np.column_stack(
                [capture, ppv, enrichment, cases]
            )
        offset += size
    return out


def random_point_and_bootstrap(
    y: np.ndarray, *, n_bootstrap: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Expected random ordering; the diagonal is analytic, not one permutation."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    prevalence = float(y.mean())
    point = []
    for capacity in CAPACITIES:
        realized = selected_n(n, capacity) / n
        point.append([realized, prevalence, 1.0, 1.0 / prevalence])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_bootstrap)
    prevalence_boot = counts @ y / n
    boot = np.full((n_bootstrap, len(CAPACITIES), len(METRICS)), np.nan)
    for index, capacity in enumerate(CAPACITIES):
        realized = selected_n(n, capacity) / n
        boot[:, index, 0] = realized
        boot[:, index, 1] = prevalence_boot
        boot[:, index, 2] = 1.0
        boot[:, index, 3] = np.divide(
            1.0,
            prevalence_boot,
            out=np.full(n_bootstrap, np.nan),
            where=prevalence_boot > 0,
        )
    return np.asarray(point), boot


def summarize_method(
    *,
    analysis: str,
    population: str,
    method: str,
    y: np.ndarray,
    score: np.ndarray | None,
    n_bootstrap: int,
    bootstrap_seed: int,
    status: str = "primary",
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if method == "random_expected":
        point, boot = random_point_and_bootstrap(
            y, n_bootstrap=n_bootstrap, seed=bootstrap_seed
        )
    else:
        if score is None or not np.isfinite(score).all():
            raise ValueError(f"{analysis}/{population}/{method}: invalid score")
        point = rank_point(y, score)
        boot = bootstrap_rank(
            y, score, n_bootstrap=n_bootstrap, seed=bootstrap_seed
        )
    rows: list[dict[str, Any]] = []
    n = len(y)
    for capacity_index, capacity in enumerate(CAPACITIES):
        row: dict[str, Any] = {
            "analysis": analysis,
            "population": population,
            "method": method,
            "status": status,
            "n": n,
            "n_mutant": int(y.sum()),
            "prevalence": float(y.mean()),
            "capacity_nominal": capacity,
            "selected_n": selected_n(n, capacity),
            "capacity_realized": selected_n(n, capacity) / n,
            "bootstrap_seed": bootstrap_seed,
            "n_bootstrap": n_bootstrap,
        }
        for metric_index, metric in enumerate(METRICS):
            values = boot[:, capacity_index, metric_index]
            finite = values[np.isfinite(values)]
            row[metric] = float(point[capacity_index, metric_index])
            row[f"{metric}_ci_low"] = (
                float(np.percentile(finite, 2.5)) if len(finite) else None
            )
            row[f"{metric}_ci_high"] = (
                float(np.percentile(finite, 97.5)) if len(finite) else None
            )
            row[f"{metric}_bootstrap_valid"] = int(len(finite))
        rows.append(row)
    return rows, boot


def _patient_context(manifest: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "patient_id",
        "target_label",
        "age_at_diagnosis",
        "sex",
        "site_class",
        "stage_class",
        "cohort",
        "subcohort",
        "msi_dmmr",
        "braf",
        "tumor_site_group",
        "k_fold",
    ]
    available = [column for column in columns if column in manifest.columns]
    patients = manifest.sort_values("slide_id").drop_duplicates("patient_id")[available].copy()
    return patients.rename(columns={"target_label": "label"})


def load_e0() -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[Path]]:
    manifest = pd.read_csv(DEV_MANIFEST, low_memory=False)
    base: pd.DataFrame | None = None
    inputs = [DEV_MANIFEST, LABEL_SOURCE, E1D_RESULT]
    for seed in SEEDS:
        path = E0_ROOT / f"seed{seed}" / "oof_predictions.parquet"
        inputs.append(path)
        patient = evaluate.to_patient_level(pd.read_parquet(path), manifest)
        patient = patient.rename(columns={"mean_logit": f"wsi_seed{seed}"})
        keep = ["patient_id", "label", f"wsi_seed{seed}"]
        if base is None:
            context = [
                column
                for column in (
                    "k_fold",
                    "age_at_diagnosis",
                    "sex",
                    "site_class",
                    "stage_class",
                    "cohort",
                    "subcohort",
                    "msi_dmmr",
                    "braf",
                    "tumor_site_group",
                )
                if column in patient.columns
            ]
            base = patient[[*keep, *context]].copy()
        else:
            base = base.merge(patient[keep], on=["patient_id", "label"], validate="one_to_one")
    assert base is not None
    extra = pd.read_csv(LABEL_SOURCE, low_memory=False).drop_duplicates("patient_uid")
    base = base.merge(
        extra[["patient_uid", "stage_group_major_filled"]].rename(
            columns={"patient_uid": "patient_id"}
        ),
        on="patient_id",
        how="left",
        validate="one_to_one",
    )
    base = base.sort_values("patient_id").reset_index(drop=True)

    base["clinical_oof"] = cross_fitted_score(base, NUMERIC, CATEGORICAL)
    for seed in SEEDS:
        fusion_frame = base.copy()
        # Reproduce the already-published E1d stack exactly.  E1d transformed
        # its stored probability back through the shared clipped-logit helper;
        # WSI worklist ranking itself always uses the native logit below.
        fusion_frame["wsi_logit"] = logit(
            sigmoid(fusion_frame[f"wsi_seed{seed}"].to_numpy())
        )
        base[f"fusion_seed{seed}"] = cross_fitted_score(
            fusion_frame, [*NUMERIC, "wsi_logit"], CATEGORICAL
        )
    base["wsi_three_seed_ensemble"] = base[[f"wsi_seed{s}" for s in SEEDS]].mean(axis=1)

    frozen = json.loads(E1D_RESULT.read_text())["pb_cap8192__A"]["per_seed"]
    checks: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        wsi_auc = float(roc_auc_score(base["label"], base[f"wsi_seed{seed}"]))
        fusion_auc = float(roc_auc_score(base["label"], base[f"fusion_seed{seed}"]))
        expected_wsi = float(frozen[str(seed)]["wsi"]["auroc"])
        expected_fusion = float(frozen[str(seed)]["fusion"]["auroc"])
        checks[f"e0_seed{seed}_wsi_auroc"] = {
            "observed": wsi_auc,
            "expected": expected_wsi,
            "absolute_error": abs(wsi_auc - expected_wsi),
            "note": "native-logit rank versus legacy probability-rank E1d identity",
            "pass": round(wsi_auc, 4) == round(expected_wsi, 4),
        }
        checks[f"e0_seed{seed}_fusion_auroc"] = {
            "observed": fusion_auc,
            "expected": expected_fusion,
            "absolute_error": abs(fusion_auc - expected_fusion),
            "pass": abs(fusion_auc - expected_fusion) < 1e-5,
        }
    clinical_auc = float(roc_auc_score(base["label"], base["clinical_oof"]))
    expected_clinical = float(frozen["42"]["clinical"]["auroc"])
    checks["e0_clinical_auroc"] = {
        "observed": clinical_auc,
        "expected": expected_clinical,
        "absolute_error": abs(clinical_auc - expected_clinical),
        "pass": abs(clinical_auc - expected_clinical) < 1e-12,
    }
    checks["e0_population"] = {
        "observed": [len(base), int(base["label"].sum())],
        "expected": [1486, 604],
        "pass": len(base) == 1486 and int(base["label"].sum()) == 604,
    }
    return base, checks, inputs


def target_manifest(target: str) -> Path:
    return Path(f"/mnt/d/YC.Liu/manifests/colon/aim1_e2a_{target.lower()}_primary.csv")


def source_manifest(target: str) -> Path:
    return Path(
        f"/mnt/d/YC.Liu/manifests/colon/aim1_e2a_{target.lower()}_source_seed42.csv"
    )


def load_e2a_target(
    target: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    target_lower = target.lower()
    target_score_path = E2A_ROOT / "calibrated" / f"cap8192_{target_lower}_primary.parquet"
    target_receipt = target_score_path.with_suffix(".receipt.json")
    target_manifest_path = target_manifest(target)
    source_manifest_path = source_manifest(target)
    inputs = [target_score_path, target_receipt, target_manifest_path, source_manifest_path]

    target_scores = pd.read_parquet(target_score_path)
    target_context = _patient_context(pd.read_csv(target_manifest_path, low_memory=False))
    target_frame = target_scores[["patient_id", "label", "mean_logit", "subcohort"]].merge(
        target_context.drop(columns=["subcohort"], errors="ignore"),
        on=["patient_id", "label"],
        validate="one_to_one",
    )
    target_frame = target_frame.sort_values("patient_id").reset_index(drop=True)

    source_frames: list[pd.DataFrame] = []
    source_manifest_frame = pd.read_csv(source_manifest_path, low_memory=False)
    for seed in SEEDS:
        score_path = E2A_SOURCE_CV_ROOT / target_lower / f"seed{seed}" / "oof_predictions.parquet"
        inputs.append(score_path)
        patient = evaluate.to_patient_level(pd.read_parquet(score_path), source_manifest_frame)
        source_frames.append(
            patient[["patient_id", "label", "mean_logit"]].rename(
                columns={"mean_logit": f"wsi_seed{seed}"}
            )
        )
    source = _patient_context(source_manifest_frame)
    for frame in source_frames:
        source = source.merge(frame, on=["patient_id", "label"], validate="one_to_one")
    source["wsi_logit"] = source[[f"wsi_seed{s}" for s in SEEDS]].mean(axis=1)
    source = source.sort_values("patient_id").reset_index(drop=True)

    if target in set(source["cohort"].astype(str)):
        raise RuntimeError(f"{target}: held-out target appears in source pool")
    if set(target_frame["cohort"].astype(str)) != {target}:
        raise RuntimeError(f"{target}: target manifest contains another cohort")

    clinical = UnweightedLogistic(NUMERIC, CATEGORICAL).fit(source)
    fusion = UnweightedLogistic([*NUMERIC, "wsi_logit"], CATEGORICAL).fit(source)
    target_frame["clinical_source_only"] = clinical.predict(target_frame)
    target_frame["wsi_logit"] = target_frame["mean_logit"]
    target_frame["fusion_source_only"] = fusion.predict(target_frame)

    expected = json.loads((E2A_ROOT.parent / "eval" / "e2a_transport_pb_cap8192.json").read_text())
    expected_auc = float(expected["targets"][target]["primary_overall"]["auroc"])
    observed_auc = float(roc_auc_score(target_frame["label"], target_frame["mean_logit"]))
    check = {
        "n": len(target_frame),
        "n_mutant": int(target_frame["label"].sum()),
        "source_n": len(source),
        "source_cohorts": sorted(source["cohort"].astype(str).unique().tolist()),
        "target_auc_observed": observed_auc,
        "target_auc_expected": expected_auc,
        "absolute_error": abs(observed_auc - expected_auc),
        "pass": abs(observed_auc - expected_auc) < 1e-12,
        "clinical_and_fusion_note": (
            "post-hoc source-only comparators; coefficients use source labels and source OOF "
            "WSI logits only; target labels are evaluation-only"
        ),
    }
    inputs.append(E2A_ROOT.parent / "eval" / "e2a_transport_pb_cap8192.json")
    return target_frame, check, inputs


def _add_comparisons(
    rows: list[dict[str, Any]],
    boot_by_method: dict[str, np.ndarray],
    analysis: str,
    population: str,
    references: Iterable[str],
    candidate: str,
) -> list[dict[str, Any]]:
    point_lookup = {
        (row["method"], row["capacity_nominal"]): row
        for row in rows
        if row["analysis"] == analysis and row["population"] == population
    }
    comparisons = []
    for reference in references:
        if reference not in boot_by_method or candidate not in boot_by_method:
            continue
        delta = boot_by_method[candidate] - boot_by_method[reference]
        for capacity_index, capacity in enumerate(CAPACITIES):
            candidate_point = point_lookup[(candidate, capacity)]
            reference_point = point_lookup[(reference, capacity)]
            entry: dict[str, Any] = {
                "analysis": analysis,
                "population": population,
                "candidate": candidate,
                "reference": reference,
                "capacity_nominal": capacity,
            }
            for metric_index, metric in enumerate(METRICS):
                values = delta[:, capacity_index, metric_index]
                finite = values[np.isfinite(values)]
                entry[f"delta_{metric}"] = (
                    candidate_point[metric] - reference_point[metric]
                )
                entry[f"delta_{metric}_ci_low"] = (
                    float(np.percentile(finite, 2.5)) if len(finite) else None
                )
                entry[f"delta_{metric}_ci_high"] = (
                    float(np.percentile(finite, 97.5)) if len(finite) else None
                )
            comparisons.append(entry)
    return comparisons


def run_analysis(n_bootstrap: int) -> tuple[dict[str, Any], list[Path]]:
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    all_inputs: list[Path] = [
        FINAL_V2 / "Experimental_Setup.md",
        FINAL_V2 / "Results.md",
        FINAL_V2 / "Audit.md",
        FINAL_V2 / "report_bundle_receipt.json",
        Path(__file__).resolve(),
    ]

    e0, checks, inputs = load_e0()
    validation.update(checks)
    all_inputs.extend(inputs)
    populations = {
        "A_all_primary": np.ones(len(e0), dtype=bool),
        "D_MSS_pMMR_BRAF_WT": (
            e0["msi_dmmr"].eq("MSS/pMMR") & e0["braf"].eq("wild_type")
        ).to_numpy(),
        "H_stage_IV_primary_exploratory": e0["stage_group_major_filled"]
        .astype("string")
        .eq("IV")
        .fillna(False)
        .to_numpy(),
    }
    expected_populations = {
        "A_all_primary": (1486, 604),
        "D_MSS_pMMR_BRAF_WT": (1129, 514),
        "H_stage_IV_primary_exploratory": (156, 74),
    }
    for population, mask in populations.items():
        frame = e0.loc[mask].reset_index(drop=True)
        observed = (len(frame), int(frame["label"].sum()))
        validation[f"population_{population}"] = {
            "observed": list(observed),
            "expected": list(expected_populations[population]),
            "pass": observed == expected_populations[population],
        }
        y = frame["label"].to_numpy(dtype=int)
        bootstrap_seed = stable_seed(f"e0::{population}")
        method_scores: dict[str, np.ndarray | None] = {
            "random_expected": None,
            "clinical_oof": frame["clinical_oof"].to_numpy(),
            **{f"wsi_seed{s}": frame[f"wsi_seed{s}"].to_numpy() for s in SEEDS},
            **{f"fusion_seed{s}": frame[f"fusion_seed{s}"].to_numpy() for s in SEEDS},
            "wsi_three_seed_ensemble_sensitivity": frame[
                "wsi_three_seed_ensemble"
            ].to_numpy(),
        }
        boot_by_method: dict[str, np.ndarray] = {}
        for method, score in method_scores.items():
            status = "primary" if population == "A_all_primary" else "exploratory_or_sensitivity"
            method_rows, boot = summarize_method(
                analysis="E0_frozen_OOF",
                population=population,
                method=method,
                y=y,
                score=score,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
                status=status,
            )
            rows.extend(method_rows)
            boot_by_method[method] = boot

        # Preserve E0's declared median-over-seeds estimand; the ensemble remains sensitivity.
        for family in ("wsi", "fusion"):
            members = [f"{family}_seed{s}" for s in SEEDS]
            member_rows = [
                row
                for row in rows
                if row["analysis"] == "E0_frozen_OOF"
                and row["population"] == population
                and row["method"] in members
            ]
            for capacity in CAPACITIES:
                blocks = [row for row in member_rows if row["capacity_nominal"] == capacity]
                summary = dict(blocks[0])
                summary["method"] = f"{family}_declared_seed_median"
                summary["status"] = (
                    "primary" if population == "A_all_primary" else "exploratory_or_sensitivity"
                )
                summary["summary_rule"] = (
                    "median of seed-specific points and percentile endpoints; "
                    "seeds are not inferential units"
                )
                for metric in METRICS:
                    summary[metric] = float(np.median([row[metric] for row in blocks]))
                    summary[f"{metric}_ci_low"] = float(
                        np.median([row[f"{metric}_ci_low"] for row in blocks])
                    )
                    summary[f"{metric}_ci_high"] = float(
                        np.median([row[f"{metric}_ci_high"] for row in blocks])
                    )
                    summary[f"{metric}_bootstrap_valid"] = min(
                        row[f"{metric}_bootstrap_valid"] for row in blocks
                    )
                rows.append(summary)
            boot_by_method[f"{family}_declared_seed_median"] = np.median(
                np.stack([boot_by_method[member] for member in members]), axis=0
            )

        comparisons.extend(
            _add_comparisons(
                rows,
                boot_by_method,
                "E0_frozen_OOF",
                population,
                ("random_expected", "clinical_oof", "fusion_declared_seed_median"),
                "wsi_declared_seed_median",
            )
        )

    target_boot: dict[str, dict[str, np.ndarray]] = {}
    for target in TARGETS:
        frame, check, inputs = load_e2a_target(target)
        validation[f"e2a_{target}"] = check
        all_inputs.extend(inputs)
        y = frame["label"].to_numpy(dtype=int)
        bootstrap_seed = stable_seed(f"e2a::{target}")
        method_scores = {
            "random_expected": None,
            "clinical_source_only": frame["clinical_source_only"].to_numpy(),
            "wsi_heldout_ensemble": frame["mean_logit"].to_numpy(),
            "fusion_source_only": frame["fusion_source_only"].to_numpy(),
        }
        target_boot[target] = {}
        for method, score in method_scores.items():
            method_rows, boot = summarize_method(
                analysis="E2a_heldout_primary",
                population=target,
                method=method,
                y=y,
                score=score,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
                status="heldout_primary",
            )
            rows.extend(method_rows)
            target_boot[target][method] = boot
        comparisons.extend(
            _add_comparisons(
                rows,
                target_boot[target],
                "E2a_heldout_primary",
                target,
                ("random_expected", "clinical_source_only", "fusion_source_only"),
                "wsi_heldout_ensemble",
            )
        )

    # Equal-target macro summary: no pooling patients or prevalence across cohorts.
    external_methods = tuple(target_boot[TARGETS[0]])
    for method in external_methods:
        target_rows = [
            row
            for row in rows
            if row["analysis"] == "E2a_heldout_primary"
            and row["population"] in TARGETS
            and row["method"] == method
        ]
        macro_boot = np.mean(np.stack([target_boot[t][method] for t in TARGETS]), axis=0)
        for capacity_index, capacity in enumerate(CAPACITIES):
            blocks = [row for row in target_rows if row["capacity_nominal"] == capacity]
            row = {
                "analysis": "E2a_heldout_primary",
                "population": "equal_target_macro",
                "method": method,
                "status": "descriptive_equal_target_macro",
                "n": sum(block["n"] for block in blocks),
                "n_mutant": sum(block["n_mutant"] for block in blocks),
                "prevalence": float(np.mean([block["prevalence"] for block in blocks])),
                "capacity_nominal": capacity,
                "selected_n": sum(block["selected_n"] for block in blocks),
                "capacity_realized": float(
                    np.mean([block["capacity_realized"] for block in blocks])
                ),
                "bootstrap_seed": None,
                "n_bootstrap": n_bootstrap,
                "summary_rule": "equal arithmetic mean over four target-specific metrics",
            }
            for metric_index, metric in enumerate(METRICS):
                values = macro_boot[:, capacity_index, metric_index]
                finite = values[np.isfinite(values)]
                row[metric] = float(np.mean([block[metric] for block in blocks]))
                row[f"{metric}_ci_low"] = float(np.percentile(finite, 2.5))
                row[f"{metric}_ci_high"] = float(np.percentile(finite, 97.5))
                row[f"{metric}_bootstrap_valid"] = int(len(finite))
            rows.append(row)

    # Point-level mathematical and monotonicity checks.
    frame_rows = pd.DataFrame(rows)
    for key, block in frame_rows.groupby(["analysis", "population", "method"], dropna=False):
        block = block.sort_values("capacity_nominal")
        ok_monotone = bool(np.all(np.diff(block["capture"].to_numpy()) >= -1e-12))
        algebra = block["capture"].to_numpy() / block["capacity_realized"].to_numpy()
        # Equal-target macros average each target's enrichment and realized
        # capacity separately; the pooled algebraic identity is not defined.
        ok_algebra = bool(
            key[1] == "equal_target_macro"
            or np.allclose(algebra, block["enrichment"], atol=1e-12, rtol=0)
        )
        validation[f"curve::{key[0]}::{key[1]}::{key[2]}"] = {
            "capture_non_decreasing": ok_monotone,
            "enrichment_identity": ok_algebra,
            "pass": ok_monotone and ok_algebra,
        }

    validation["overall"] = {
        "status": "PASS" if all(v.get("pass", True) for v in validation.values()) else "FAIL"
    }
    if validation["overall"]["status"] != "PASS":
        failed = [key for key, value in validation.items() if not value.get("pass", True)]
        raise RuntimeError(f"validation failed: {failed}")

    payload = {
        "schema_version": 1,
        "analysis": "final_v3_worklist_enrichment",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "design": {
            "image_training": "none",
            "primary_capacity": PRIMARY_CAPACITY,
            "capacity_grid": list(CAPACITIES),
            "capacity_integer_rule": "floor(q*n), minimum one; realized capacity reported",
            "tie_rule": "fractional allocation across all patients tied at the cutoff",
            "uncertainty": (
                "ordinary patient bootstrap with replacement; paired resamples within population; "
                "percentile 95% intervals"
            ),
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed_base": BOOTSTRAP_SEED,
            "score_scale": "native patient mean logit; monotone probabilities not required",
            "random_comparator": "analytic expected random ordering (diagonal)",
            "target_threshold_optimization": False,
            "interpretation_boundary": (
                "retrospective worklist ordering/enrichment only; not test avoidance, treatment "
                "selection, utility, cost-effectiveness, or prospective validation"
            ),
        },
        "rows": rows,
        "comparisons": comparisons,
        "validation": validation,
    }
    # Deduplicate paths without losing deterministic order.
    unique_inputs = list(dict.fromkeys(path.resolve() for path in all_inputs))
    return payload, unique_inputs


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _fmt_ci(row: pd.Series, metric: str, digits: int = 3) -> str:
    return (
        f"{row[metric]:.{digits}f} "
        f"[{row[f'{metric}_ci_low']:.{digits}f}, {row[f'{metric}_ci_high']:.{digits}f}]"
    )


def render_design(payload: dict[str, Any]) -> str:
    frame = pd.DataFrame(payload["rows"])
    top30 = frame[np.isclose(frame["capacity_nominal"], PRIMARY_CAPACITY)]
    e0 = top30[
        (top30["analysis"] == "E0_frozen_OOF")
        & (top30["population"] == "A_all_primary")
        & top30["method"].isin(
            [
                "random_expected",
                "clinical_oof",
                "wsi_declared_seed_median",
                "fusion_declared_seed_median",
            ]
        )
    ]
    ext = top30[
        (top30["analysis"] == "E2a_heldout_primary")
        & top30["population"].isin(TARGETS)
        & (top30["method"] == "wsi_heldout_ensemble")
    ]
    d_h = top30[
        (top30["analysis"] == "E0_frozen_OOF")
        & top30["population"].isin(
            ["D_MSS_pMMR_BRAF_WT", "H_stage_IV_primary_exploratory"]
        )
        & (top30["method"] == "wsi_declared_seed_median")
    ]
    lines = [
        "# Final-v3 add-on: KRAS worklist enrichment",
        "",
        "## Design",
        "",
        "No image model was trained. Patients were ranked by frozen E0 OOF native logits or by "
        "the corrected E2a three-seed held-out primary ensemble. Worklist capacities were fixed "
        "at 10%, 20%, 30%, 40% and 50%; top 30% is the sole headline capacity. The selected "
        "count is `floor(q*n)`, score ties crossing the cutoff receive fractional allocation, "
        "and intervals use 10,000 ordinary patient bootstraps with paired resamples. Random is "
        "the analytic expected ordering, not a favorable random permutation.",
        "",
        "Clinical and fusion comparators in E0 reproduce the existing five-fold cross-fitting. "
        "For E2a, they are post-hoc but strictly source-only: coefficients use source labels, and "
        "fusion uses source OOF WSI logits; no held-out target label fits or selects a model.",
        "",
        "These are retrospective worklist-ordering results. They do not show that molecular tests "
        "can be omitted, that turnaround time or costs improve, or that any treatment decision can "
        "be made from H&E.",
        "",
        "## Top-30% results",
        "",
        "### E0 development OOF",
        "",
        "| Ranking | Patients selected | Mutant capture (95% CI) | Enrichment (95% CI) | Cases per mutant |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    order = {
        "random_expected": 0,
        "clinical_oof": 1,
        "wsi_declared_seed_median": 2,
        "fusion_declared_seed_median": 3,
    }
    for _, row in e0.assign(_order=e0["method"].map(order)).sort_values("_order").iterrows():
        lines.append(
            f"| {row['method']} | {int(row['selected_n'])}/{int(row['n'])} | "
            f"{_fmt_ci(row, 'capture')} | {_fmt_ci(row, 'enrichment')} | "
            f"{row['cases_per_mutant']:.2f} |"
        )
    lines += [
        "",
        "### Held-out E2a primary targets",
        "",
        "| Target | Patients selected | Mutant capture (95% CI) | Enrichment (95% CI) | Cases per mutant |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in ext.iterrows():
        lines.append(
            f"| {row['population']} | {int(row['selected_n'])}/{int(row['n'])} | "
            f"{_fmt_ci(row, 'capture')} | {_fmt_ci(row, 'enrichment')} | "
            f"{row['cases_per_mutant']:.2f} |"
        )
    lines += [
        "",
        "### Molecular and stage sensitivities",
        "",
        "| Population | Patients selected | Mutant capture (95% CI) | Enrichment (95% CI) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in d_h.iterrows():
        lines.append(
            f"| {row['population']} | {int(row['selected_n'])}/{int(row['n'])} | "
            f"{_fmt_ci(row, 'capture')} | {_fmt_ci(row, 'enrichment')} |"
        )
    lines += [
        "",
        "Full curves, comparator results, paired differences and machine-readable definitions are "
        "in `worklist_results.csv`, `comparisons.csv` and `worklist_results.json`.",
        "",
    ]
    return "\n".join(lines)


def write_once(output: Path, payload: dict[str, Any], inputs: list[Path]) -> None:
    if output.exists():
        raise FileExistsError(f"append-only destination already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    input_receipt = {
        "schema_version": 1,
        "created_utc": payload["created_utc"],
        "inputs": [identity(path) for path in inputs],
        "final_v2_immutable": True,
    }
    (output / "input_receipt.json").write_text(
        json.dumps(_json_safe(input_receipt), indent=2, sort_keys=True) + "\n"
    )
    (output / "worklist_results.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(payload["rows"]).to_csv(output / "worklist_results.csv", index=False)
    pd.DataFrame(payload["comparisons"]).to_csv(output / "comparisons.csv", index=False)
    (output / "validation.json").write_text(
        json.dumps(_json_safe(payload["validation"]), indent=2, sort_keys=True) + "\n"
    )
    (output / "DESIGN_AND_RESULTS.md").write_text(render_design(payload))

    produced = [
        output / "input_receipt.json",
        output / "worklist_results.json",
        output / "worklist_results.csv",
        output / "comparisons.csv",
        output / "validation.json",
        output / "DESIGN_AND_RESULTS.md",
    ]
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "append_only": True,
        "created_utc": payload["created_utc"],
        "output_root": str(output.resolve()),
        "artifacts": [identity(path) for path in produced],
        "promise": "No file under reports/final_v2 or any upstream result root was modified.",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and validate but do not create the append-only result root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    payload, inputs = run_analysis(args.n_bootstrap)
    if args.dry_run:
        print(json.dumps({"status": "PASS", "rows": len(payload["rows"])}, indent=2))
        return
    write_once(args.output, payload, inputs)
    print(f"PASS: wrote append-only worklist analysis to {args.output}")


if __name__ == "__main__":
    main()
