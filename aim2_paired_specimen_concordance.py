#!/usr/bin/env python3
"""E2d-5 - paired primary/metastatic specimen concordance (0 new fits).

This panel asks a deliberately narrow question: when the same patient has a
primary and a metastatic specimen, how stable is the *frozen* LOCO model score?
It is not a primary-versus-metastatic AUROC analysis.  There are only nine
eligible dual-role patients (eight RIH and one TCGA), one RIH pair has
discordant KRAS labels across specimens, and the TCGA pair is scored by a
different held-out-target model from the RIH pairs.  Accordingly:

* all nine verified pairs are retained in the auditable pair table;
* paired shift/correlation/concordance inference is headline only within RIH,
  where all eight pairs share one frozen RIH-held-out ensemble;
* the single TCGA pair is descriptive, while an all-nine bootstrap is emitted
  only as a heterogeneous-model sensitivity;
* no AUROC, equivalence, non-inferiority, or causal specimen-role claim is made.

The E2a native-logit score caches are reused for RIH primary/metastatic and
TCGA primary specimens.  E2a intentionally did not score TCGA's single
metastasis for E2b, so ``score`` applies the already-frozen TCGA-held-out E2a
checkpoints to that one feature-backed slide and records immutable receipts.
No model is trained or adapted here.

Usage:
    python aim2_paired_specimen_concordance.py verify
    python aim2_paired_specimen_concordance.py score  --cap 8192
    python aim2_paired_specimen_concordance.py report --cap 8192 --n-bootstrap 10000
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim2_loco_transport  # noqa: E402
from oceanpath.aim1 import evaluate, lineage, paths, population, scoring  # noqa: E402

PAIR_COHORT_COUNTS: dict[str, int] = {"RIH": 8, "TCGA": 1}
EXPECTED_LABEL_DISCORDANT = 1
DEFAULT_N_BOOTSTRAP = 10_000


def e2d5_root() -> Path:
    return lineage.component_root("e2d5")


def tcga_metastatic_manifest_path(cap: int) -> Path:
    return e2d5_root() / "inputs" / f"tcga_paired_metastatic_cap{cap}.csv"


def tcga_metastatic_score_path(seed: int, cap: int) -> Path:
    return e2d5_root() / "scores" / f"tcga_seed{seed}_metastatic_cap{cap}.parquet"


def tcga_metastatic_receipt_path(seed: int, cap: int) -> Path:
    return tcga_metastatic_score_path(seed, cap).with_suffix(".receipt.json")


def paired_table_path(cap: int) -> Path:
    return lineage.eval_root() / f"e2d5_paired_specimens_cap{cap}_patients.parquet"


def paired_table_receipt_path(cap: int) -> Path:
    return paired_table_path(cap).with_suffix(".receipt.json")


def report_path(cap: int) -> Path:
    return lineage.eval_root() / f"e2d5_paired_specimens_cap{cap}.json"


def verified_pair_population() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return the frozen feature-backed dual-role population, or fail closed."""

    primary = population.add_context_columns(population.eligible())
    metastatic = population.eligible_metastatic()
    pair_ids = population.dual_specimen_patients(primary, metastatic)
    primary = primary[primary["patient_id"].isin(pair_ids)].copy()
    metastatic = metastatic[metastatic["patient_id"].isin(pair_ids)].copy()

    primary_ids = set(primary["patient_id"])
    metastatic_ids = set(metastatic["patient_id"])
    if primary_ids != metastatic_ids:
        raise RuntimeError("Primary and metastatic dual-role patient sets differ")

    per_patient = primary.drop_duplicates("patient_id")[["patient_id", "cohort"]]
    observed = per_patient.groupby("cohort")["patient_id"].nunique().to_dict()
    observed = {str(key): int(value) for key, value in observed.items()}
    if observed != PAIR_COHORT_COUNTS:
        raise RuntimeError(
            "Feature-backed dual-role population changed: "
            f"expected {PAIR_COHORT_COUNTS}, observed {observed}"
        )

    for role, frame in (("primary", primary), ("metastatic", metastatic)):
        label_counts = frame.groupby("patient_id")["target_label"].nunique()
        if not label_counts.eq(1).all():
            bad = label_counts[label_counts.ne(1)].index.tolist()
            raise RuntimeError(f"{role} has within-specimen label conflicts for {bad}")

    p_labels = primary.groupby("patient_id")["target_label"].first()
    m_labels = metastatic.groupby("patient_id")["target_label"].first()
    discordant = sorted(p_labels.index[p_labels.ne(m_labels)].tolist())
    if len(discordant) != EXPECTED_LABEL_DISCORDANT:
        raise RuntimeError(
            "Dual-role label concordance changed: expected "
            f"{EXPECTED_LABEL_DISCORDANT} discordant pair, observed {len(discordant)}"
        )

    audit = {
        "n_pairs": int(len(primary_ids)),
        "cohort_counts": observed,
        "n_primary_slides": int(len(primary)),
        "n_metastatic_slides": int(len(metastatic)),
        "n_label_concordant": int(len(primary_ids) - len(discordant)),
        "n_label_discordant": int(len(discordant)),
        "label_discordant_patient_ids": discordant,
        "label_caveat": (
            "KRAS status differs between the primary and metastasis for one pair. "
            "This may be biological heterogeneity or a source-data discrepancy; "
            "the score-concordance panel does not adjudicate it."
        ),
    }
    return primary, metastatic, audit


def tcga_metastatic_manifest() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the one-patient TCGA metastatic scoring manifest in memory."""

    _, metastatic, audit = verified_pair_population()
    rows = metastatic[metastatic["cohort"].eq("TCGA")].copy()
    if rows["patient_id"].nunique() != 1:
        raise RuntimeError("Expected exactly one feature-backed TCGA dual-role metastasis")
    manifest = population.finalize(rows, extra=["met_site_class", "liver_class"])
    return manifest, audit


def _tcga_score_inputs(seed: int, cap: int, manifest_path: Path) -> dict[str, Any]:
    from oceanpath.workflows.training import _feature_inventory_sha256

    aim2_loco_transport._completed_refit("TCGA", seed, cap)
    return {
        "lineage": lineage.lineage_name(),
        "target": "TCGA",
        "specimen_role": "metastatic",
        "seed": int(seed),
        "cap": int(cap),
        "aggregation_contract": "native slide logits -> mean slide logit per patient",
        "checkpoint": lineage.artifact_identity(aim2_loco_transport.model_ckpt("TCGA", seed, cap)),
        "fit_receipt": lineage.artifact_identity(aim2_loco_transport.fit_summary_path("TCGA", seed, cap)),
        "manifest": lineage.artifact_identity(manifest_path),
        "feature_store": {
            "path": str(paths.PINNED_FEATURE_DIR.resolve()),
            "selected_inventory_sha256": _feature_inventory_sha256(
                paths.PINNED_FEATURE_DIR, manifest_path, "slide_id"
            ),
        },
        "analysis_code": lineage.artifact_identity(REPO / "aim2_paired_specimen_concordance.py"),
    }


def _validate_native_scores(
    scores: pd.DataFrame, manifest: pd.DataFrame, *, seed: int
) -> None:
    required = {"slide_id", "seed", "logit"}
    missing = required - set(scores.columns)
    if missing:
        raise RuntimeError(f"TCGA metastatic scores lack columns {sorted(missing)}")
    if len(scores) != len(manifest):
        raise RuntimeError(
            f"TCGA metastatic score count mismatch: {len(scores)} vs {len(manifest)}"
        )
    if set(scores["slide_id"]) != set(manifest["slide_id"]):
        raise RuntimeError("TCGA metastatic scores do not cover the frozen manifest")
    if set(pd.to_numeric(scores["seed"], errors="raise").astype(int)) != {int(seed)}:
        raise RuntimeError(f"TCGA metastatic score file is not exclusively seed {seed}")
    if not np.isfinite(pd.to_numeric(scores["logit"], errors="coerce")).all():
        raise RuntimeError("TCGA metastatic scorer did not emit finite native logits")


def cmd_verify(_: argparse.Namespace) -> None:
    primary, metastatic, audit = verified_pair_population()
    labels = (
        primary.groupby("patient_id")["target_label"]
        .first()
        .rename("primary_label")
        .to_frame()
        .join(
            metastatic.groupby("patient_id")["target_label"]
            .first()
            .rename("metastatic_label")
        )
        .join(primary.groupby("patient_id")["cohort"].first())
        .reset_index()
        .sort_values(["cohort", "patient_id"])
    )
    print("E2d-5 verified feature-backed dual-role population")
    print(json.dumps(audit, indent=2))
    print(labels.to_string(index=False))
    print("\nRead-only verification: no artifact was written.")


def cmd_score(args: argparse.Namespace) -> None:
    """Score TCGA's single metastasis with its already-frozen E2a ensemble."""

    lineage.lineage_name()
    manifest_path = tcga_metastatic_manifest_path(args.cap)
    destinations = [manifest_path]
    for seed in aim2_loco_transport.SEEDS:
        destinations.extend(
            [
                tcga_metastatic_score_path(seed, args.cap),
                tcga_metastatic_receipt_path(seed, args.cap),
            ]
        )
    for destination in destinations:
        lineage.ensure_absent(destination)

    manifest, audit = tcga_metastatic_manifest()
    # Validate all immutable upstream fits before publishing the first E2d-5
    # artifact.  A scorer/runtime failure can still leave useful partial
    # evidence, but a known-missing checkpoint should not consume a lineage.
    for seed in aim2_loco_transport.SEEDS:
        aim2_loco_transport._completed_refit("TCGA", seed, args.cap)
    lineage.write_text_once(manifest_path, manifest.to_csv(index=False))
    for seed in aim2_loco_transport.SEEDS:
        score_path = tcga_metastatic_score_path(seed, args.cap)
        receipt_path = tcga_metastatic_receipt_path(seed, args.cap)
        inputs = _tcga_score_inputs(seed, args.cap, manifest_path)
        scores = scoring.score_manifest_with_checkpoints(
            checkpoints=[(seed, 0, aim2_loco_transport.model_ckpt("TCGA", seed, args.cap))],
            manifest_csv=manifest_path,
            feature_dir=paths.PINNED_FEATURE_DIR,
            num_classes=2,
        )
        _validate_native_scores(scores, manifest, seed=seed)
        lineage.write_parquet_once(score_path, scores)
        lineage.write_json_once(
            receipt_path,
            {
                "schema_version": 1,
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "inputs": inputs,
                "population_audit": audit,
                "artifact": lineage.artifact_identity(score_path),
                "n_rows": int(len(scores)),
            },
        )
        print(f"  TCGA paired metastasis seed {seed}: {len(scores)} native-logit score(s)")
    print(f"Wrote immutable E2d-5 score inputs under {e2d5_root()}")


def _load_tcga_metastatic_seed(seed: int, cap: int) -> pd.DataFrame:
    manifest_path = tcga_metastatic_manifest_path(cap)
    score_path = tcga_metastatic_score_path(seed, cap)
    receipt_path = tcga_metastatic_receipt_path(seed, cap)
    for path in (manifest_path, score_path, receipt_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing E2d-5 score input {path}; run `aim2_paired_specimen_concordance.py score`")
    manifest = pd.read_csv(manifest_path)
    scores = pd.read_parquet(score_path)
    _validate_native_scores(scores, manifest, seed=seed)
    receipt = json.loads(receipt_path.read_text())
    expected = _tcga_score_inputs(seed, cap, manifest_path)
    if receipt.get("inputs") != expected:
        raise RuntimeError(f"TCGA metastatic score input identity mismatch: {score_path}")
    if receipt.get("artifact") != lineage.artifact_identity(score_path):
        raise RuntimeError(f"TCGA metastatic score hash mismatch: {score_path}")
    slides = scoring.ensemble_slide_predictions(scores, manifest)
    return evaluate.to_patient_level(slides, manifest)


def _require_e2a_score_cache(target: str, kind: str, cap: int) -> None:
    missing: list[str] = []
    for seed in aim2_loco_transport.SEEDS:
        for path in (
            aim2_loco_transport.scores_path(target, seed, kind, cap),
            aim2_loco_transport.score_receipt_path(target, seed, kind, cap),
        ):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"Missing frozen E2a {target}/{kind} score cache(s); run E2a/E2b scoring first: "
            + ", ".join(missing)
        )


def _frozen_e2a_ensemble(
    target: str, kind: str, cap: int
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    _require_e2a_score_cache(target, kind, cap)
    ensemble, per_seed = aim2_loco_transport.seed_ensemble(target, kind, cap)
    if set(per_seed) != set(aim2_loco_transport.SEEDS):
        raise RuntimeError(
            f"Incomplete E2a {target}/{kind} ensemble: {sorted(per_seed)}; "
            f"expected {list(aim2_loco_transport.SEEDS)}"
        )
    return ensemble, per_seed


def _subset_exact(frame: pd.DataFrame, patient_ids: set[str], what: str) -> pd.DataFrame:
    subset = frame[frame["patient_id"].isin(patient_ids)].copy()
    if subset["patient_id"].duplicated().any():
        raise RuntimeError(f"{what} is not one row per patient")
    observed = set(subset["patient_id"])
    if observed != patient_ids:
        raise RuntimeError(
            f"{what} pair coverage mismatch; missing={sorted(patient_ids - observed)}, "
            f"extra={sorted(observed - patient_ids)}"
        )
    return subset


def make_pair_table(
    primary: pd.DataFrame, metastatic: pd.DataFrame, *, cohort: str
) -> pd.DataFrame:
    """Join one-row-per-patient native-logit role ensembles."""

    required = {"patient_id", "label", "mean_logit", "prob_raw"}
    for role, frame in (("primary", primary), ("metastatic", metastatic)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{role} frame lacks {sorted(missing)}")
        if frame["patient_id"].duplicated().any():
            raise ValueError(f"{role} frame is not one row per patient")
        if not np.isfinite(pd.to_numeric(frame["mean_logit"], errors="coerce")).all():
            raise ValueError(f"{role} frame has non-finite native logits")

    keep = ["patient_id", "label", "mean_logit", "prob_raw"]
    pairs = primary[keep].merge(
        metastatic[keep],
        on="patient_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_primary", "_metastatic"),
    )
    if len(pairs) != len(primary) or len(pairs) != len(metastatic):
        raise ValueError("Primary and metastatic patient sets do not match exactly")
    pairs.insert(1, "cohort", cohort)
    pairs["delta_logit_metastatic_minus_primary"] = (
        pairs["mean_logit_metastatic"] - pairs["mean_logit_primary"]
    )
    pairs["label_concordant"] = pairs["label_primary"].eq(pairs["label_metastatic"])
    return pairs.sort_values("patient_id").reset_index(drop=True)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearson(xr, yr)


def lins_concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's CCC: agreement, not merely correlation up to a shift/scale."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx = float(np.mean((x - mx) ** 2))
    vy = float(np.mean((y - my) ** 2))
    covariance = float(np.mean((x - mx) * (y - my)))
    denominator = vx + vy + (mx - my) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else float("nan")


def exact_sign_test(differences: np.ndarray) -> dict[str, Any]:
    """Two-sided exact binomial sign test; exact zero shifts are ties."""

    differences = np.asarray(differences, dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Sign test received non-finite paired differences")
    positive = int(np.sum(differences > 0))
    negative = int(np.sum(differences < 0))
    ties = int(np.sum(differences == 0))
    n = positive + negative
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, index) for index in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / (2**n))
    return {
        "positive": positive,
        "negative": negative,
        "ties_excluded": ties,
        "n_nonzero": n,
        "two_sided_exact_p": float(p_value),
        "null": "positive and negative paired shifts are equally likely",
    }


def paired_point_summary(pairs: pd.DataFrame) -> dict[str, float | int]:
    _validate_pair_frame(pairs)
    primary = pairs["mean_logit_primary"].to_numpy(dtype=float)
    metastatic = pairs["mean_logit_metastatic"].to_numpy(dtype=float)
    delta = metastatic - primary
    return {
        "n_pairs": int(len(pairs)),
        "mean_logit_shift": float(np.mean(delta)),
        "median_logit_shift": float(np.median(delta)),
        "mean_absolute_logit_shift": float(np.mean(np.abs(delta))),
        "pearson_r": _pearson(primary, metastatic),
        "spearman_rho": _spearman(primary, metastatic),
        "lins_ccc": lins_concordance_correlation(primary, metastatic),
    }


def _validate_pair_frame(pairs: pd.DataFrame) -> None:
    required = {
        "patient_id",
        "mean_logit_primary",
        "mean_logit_metastatic",
        "delta_logit_metastatic_minus_primary",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Paired frame lacks {sorted(missing)}")
    if pairs.empty:
        raise ValueError("Paired frame is empty")
    if pairs["patient_id"].duplicated().any():
        raise ValueError("Paired frame is not one row per patient")
    values = pairs[["mean_logit_primary", "mean_logit_metastatic"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Paired frame has non-finite native logits")
    observed_delta = values[:, 1] - values[:, 0]
    recorded_delta = pairs["delta_logit_metastatic_minus_primary"].to_numpy(dtype=float)
    if not np.allclose(observed_delta, recorded_delta, rtol=0, atol=1e-12):
        raise ValueError("Recorded paired logit shifts do not match role logits")


def paired_bootstrap(
    pairs: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Patient-pair bootstrap for shift and descriptive agreement metrics."""

    _validate_pair_frame(pairs)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    point = paired_point_summary(pairs)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {
        key: []
        for key in (
            "mean_logit_shift",
            "median_logit_shift",
            "mean_absolute_logit_shift",
            "pearson_r",
            "spearman_rho",
            "lins_ccc",
        )
    }
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(pairs), len(pairs))
        sample = pairs.iloc[indices]
        summary = paired_point_summary(sample.assign(
            patient_id=[f"bootstrap-{index}" for index in range(len(sample))]
        ))
        for key in draws:
            value = float(summary[key])
            if np.isfinite(value):
                draws[key].append(value)

    intervals: dict[str, dict[str, Any]] = {}
    for key, values in draws.items():
        if values:
            low, high = np.percentile(np.asarray(values), [2.5, 97.5])
            ci = [float(low), float(high)]
        else:
            ci = [float("nan"), float("nan")]
        intervals[key] = {"ci_95_percentile": ci, "n_bootstrap_valid": len(values)}

    return {
        **point,
        "bootstrap": {
            "method": "nonparametric patient-pair resampling",
            "sampling_unit": "patient pair",
            "n_bootstrap_requested": int(n_bootstrap),
            "bootstrap_seed": int(seed),
            "intervals": intervals,
            "conditioning": "conditional on the frozen three-seed logit ensemble",
        },
        "exact_sign_test": exact_sign_test(
            pairs["delta_logit_metastatic_minus_primary"].to_numpy(dtype=float)
        ),
    }


def _input_artifacts(cap: int) -> dict[str, Any]:
    e2a_receipts: dict[str, dict[str, dict[str, Any]]] = {}
    for target, kinds in (("RIH", ("primary", "metastatic")), ("TCGA", ("primary",))):
        e2a_receipts[target] = {}
        for kind in kinds:
            e2a_receipts[target][kind] = {
                str(seed): lineage.artifact_identity(
                    aim2_loco_transport.score_receipt_path(target, seed, kind, cap)
                )
                for seed in aim2_loco_transport.SEEDS
            }
    return {
        "label_source": lineage.artifact_identity(paths.LABEL_SOURCE),
        "analysis_code": lineage.artifact_identity(REPO / "aim2_paired_specimen_concordance.py"),
        "e2a_score_receipts": e2a_receipts,
        "tcga_metastatic_manifest": lineage.artifact_identity(
            tcga_metastatic_manifest_path(cap)
        ),
        "tcga_metastatic_score_receipts": {
            str(seed): lineage.artifact_identity(tcga_metastatic_receipt_path(seed, cap))
            for seed in aim2_loco_transport.SEEDS
        },
    }


def _assemble_pairs(
    cap: int,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[str, Any]]:
    primary_source, metastatic_source, audit = verified_pair_population()
    ids_by_cohort = {
        cohort: set(primary_source.loc[primary_source["cohort"].eq(cohort), "patient_id"])
        for cohort in PAIR_COHORT_COUNTS
    }

    rih_p, rih_p_seed = _frozen_e2a_ensemble("RIH", "primary", cap)
    rih_m, rih_m_seed = _frozen_e2a_ensemble("RIH", "metastatic", cap)
    tcga_p, tcga_p_seed = _frozen_e2a_ensemble("TCGA", "primary", cap)
    tcga_m_seed = {seed: _load_tcga_metastatic_seed(seed, cap) for seed in aim2_loco_transport.SEEDS}
    tcga_m_frames = [
        tcga_m_seed[seed].sort_values("patient_id").reset_index(drop=True)
        for seed in aim2_loco_transport.SEEDS
    ]
    tcga_m = tcga_m_frames[0].copy()
    if len({tuple(frame["patient_id"]) for frame in tcga_m_frames}) != 1:
        raise RuntimeError("TCGA metastatic seeds cover different paired patients")
    tcga_m["mean_logit"] = np.mean(
        [frame["mean_logit"].to_numpy(dtype=float) for frame in tcga_m_frames], axis=0
    )
    from oceanpath.eval.external import sigmoid

    tcga_m["prob_raw"] = sigmoid(tcga_m["mean_logit"].to_numpy())

    ensemble_pairs = pd.concat(
        [
            make_pair_table(
                _subset_exact(rih_p, ids_by_cohort["RIH"], "RIH primary ensemble"),
                _subset_exact(rih_m, ids_by_cohort["RIH"], "RIH metastatic ensemble"),
                cohort="RIH",
            ),
            make_pair_table(
                _subset_exact(tcga_p, ids_by_cohort["TCGA"], "TCGA primary ensemble"),
                _subset_exact(tcga_m, ids_by_cohort["TCGA"], "TCGA metastatic ensemble"),
                cohort="TCGA",
            ),
        ],
        ignore_index=True,
    ).sort_values(["cohort", "patient_id"]).reset_index(drop=True)

    per_seed_pairs: dict[int, pd.DataFrame] = {}
    for seed in aim2_loco_transport.SEEDS:
        frame = pd.concat(
            [
                make_pair_table(
                    _subset_exact(
                        rih_p_seed[seed], ids_by_cohort["RIH"], f"RIH primary seed {seed}"
                    ),
                    _subset_exact(
                        rih_m_seed[seed], ids_by_cohort["RIH"],
                        f"RIH metastatic seed {seed}"
                    ),
                    cohort="RIH",
                ),
                make_pair_table(
                    _subset_exact(
                        tcga_p_seed[seed], ids_by_cohort["TCGA"],
                        f"TCGA primary seed {seed}"
                    ),
                    _subset_exact(
                        tcga_m_seed[seed], ids_by_cohort["TCGA"],
                        f"TCGA metastatic seed {seed}"
                    ),
                    cohort="TCGA",
                ),
            ],
            ignore_index=True,
        ).sort_values(["cohort", "patient_id"]).reset_index(drop=True)
        if not frame[["patient_id", "cohort"]].equals(
            ensemble_pairs[["patient_id", "cohort"]]
        ):
            raise RuntimeError(f"Seed {seed} pair identity differs from ensemble")
        per_seed_pairs[seed] = frame
        for role in ("primary", "metastatic"):
            ensemble_pairs[f"mean_logit_{role}_seed{seed}"] = frame[
                f"mean_logit_{role}"
            ].to_numpy()
        ensemble_pairs[f"delta_logit_seed{seed}"] = frame[
            "delta_logit_metastatic_minus_primary"
        ].to_numpy()

    if len(ensemble_pairs) != sum(PAIR_COHORT_COUNTS.values()):
        raise RuntimeError("Assembled pair table does not contain all nine verified patients")
    expected_primary_labels = primary_source.groupby("patient_id")["target_label"].first()
    expected_metastatic_labels = metastatic_source.groupby("patient_id")["target_label"].first()
    assembled = ensemble_pairs.set_index("patient_id")
    if not assembled["label_primary"].astype(int).equals(
        expected_primary_labels.loc[assembled.index].astype(int)
    ):
        raise RuntimeError("Primary score-manifest labels differ from the frozen label source")
    if not assembled["label_metastatic"].astype(int).equals(
        expected_metastatic_labels.loc[assembled.index].astype(int)
    ):
        raise RuntimeError("Metastatic score-manifest labels differ from the frozen label source")
    if int((~ensemble_pairs["label_concordant"]).sum()) != EXPECTED_LABEL_DISCORDANT:
        raise RuntimeError("Assembled pair table does not preserve the label-discordant pair")
    return ensemble_pairs, per_seed_pairs, audit


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def cmd_report(args: argparse.Namespace) -> None:
    lineage.lineage_name()
    destinations = [
        report_path(args.cap),
        paired_table_path(args.cap),
        paired_table_receipt_path(args.cap),
    ]
    for destination in destinations:
        lineage.ensure_absent(destination)

    pairs, per_seed, audit = _assemble_pairs(args.cap)
    inputs = _input_artifacts(args.cap)
    rih = pairs[pairs["cohort"].eq("RIH")].copy()
    tcga = pairs[pairs["cohort"].eq("TCGA")].copy()
    if len(rih) != 8 or len(tcga) != 1:
        raise RuntimeError("Unexpected RIH/TCGA pair split after assembly")

    headline = paired_bootstrap(
        rih, n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed
    )
    sensitivity = paired_bootstrap(
        pairs, n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed
    )
    per_seed_diagnostic = {
        str(seed): {
            "RIH": paired_point_summary(frame[frame["cohort"].eq("RIH")]),
            "all_nine_heterogeneous_models": paired_point_summary(frame),
        }
        for seed, frame in per_seed.items()
    }

    lineage.write_parquet_once(paired_table_path(args.cap), pairs)
    lineage.write_json_once(
        paired_table_receipt_path(args.cap),
        {
            "schema_version": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "lineage": lineage.lineage_name(),
            "cap": int(args.cap),
            "inputs": inputs,
            "population_audit": audit,
            "artifact": lineage.artifact_identity(paired_table_path(args.cap)),
        },
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "lineage": lineage.lineage_name(),
        "cap": int(args.cap),
        "question": (
            "How stable are frozen LOCO native logits between paired primary and "
            "metastatic specimens from the same patient?"
        ),
        "population_audit": audit,
        "input_artifacts": inputs,
        "paired_table": {
            "artifact": lineage.artifact_identity(paired_table_path(args.cap)),
            "receipt": lineage.artifact_identity(paired_table_receipt_path(args.cap)),
            "rows": _json_records(pairs),
        },
        "headline_RIH_same_model": {
            **headline,
            "scope": (
                "Eight RIH pairs scored by one frozen RIH-held-out three-seed ensemble"
            ),
            "inferential_status": (
                "exploratory small-n paired inference; correlation and CCC intervals "
                "are descriptive and cannot establish equivalence"
            ),
        },
        "TCGA_single_pair": {
            "scope": "one pair under the frozen TCGA-held-out ensemble; descriptive only",
            "row": _json_records(tcga)[0],
        },
        "all_nine_heterogeneous_model_sensitivity": {
            **sensitivity,
            "scope": (
                "All verified pairs retained, but RIH and TCGA use different held-out-target "
                "models; this is not the headline concordance analysis"
            ),
            "inferential_status": "sensitivity only; no pooled-model claim",
        },
        "per_training_seed_diagnostic": {
            "inferential_role": (
                "none; seeds are correlated computational stability checks, not patients"
            ),
            "summaries": per_seed_diagnostic,
        },
        "guardrails": {
            "auroc_computed": False,
            "equivalence_claim_allowed": False,
            "noninferiority_claim_allowed": False,
            "causal_specimen_role_claim_allowed": False,
            "label_discordance": (
                "One pair changes recorded KRAS status across roles, so label-conditioned "
                "performance is not inferred from this panel."
            ),
            "interpretation": (
                "A confidence interval crossing zero is not evidence of no shift, and a high "
                "correlation is not agreement: CCC and the paired shift must be read together."
            ),
        },
    }
    lineage.write_json_once(report_path(args.cap), report)

    mean_ci = headline["bootstrap"]["intervals"]["mean_logit_shift"]["ci_95_percentile"]
    print("E2d-5 paired-specimen concordance")
    print(
        f"  verified pairs: {audit['n_pairs']} "
        f"(RIH {PAIR_COHORT_COUNTS['RIH']}, TCGA {PAIR_COHORT_COUNTS['TCGA']}); "
        f"label-discordant: {audit['n_label_discordant']}"
    )
    print(
        f"  RIH mean metastatic-primary logit shift {headline['mean_logit_shift']:+.4f} "
        f"(95% patient-pair bootstrap [{mean_ci[0]:+.4f}, {mean_ci[1]:+.4f}])"
    )
    print(
        f"  RIH Pearson {headline['pearson_r']:+.3f}; Spearman "
        f"{headline['spearman_rho']:+.3f}; Lin CCC {headline['lins_ccc']:+.3f}"
    )
    print("  TCGA's one pair is descriptive; no AUROC/equivalence/causal claim is made.")
    print(f"Wrote {report_path(args.cap)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="read-only verification of the nine pairs")
    verify.set_defaults(func=cmd_verify)
    score = commands.add_parser("score", help="score TCGA's one metastasis with frozen E2a")
    score.add_argument("--cap", type=int, default=8192, choices=aim2_loco_transport.E2A_CAPS)
    score.set_defaults(func=cmd_score)
    report = commands.add_parser("report", help="paired concordance report")
    report.add_argument("--cap", type=int, default=8192, choices=aim2_loco_transport.E2A_CAPS)
    report.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    report.add_argument("--bootstrap-seed", type=int, default=paths.BOOTSTRAP_SEED)
    report.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
