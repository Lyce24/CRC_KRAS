#!/usr/bin/env python3
"""Frozen analysis for the blinded reviews/v5 whole-section panel study.

This script is deliberately independent of the packet builder.  It accepts a
returned scoring form, the returned reviewer-information form, and the sealed
case key.  It refuses to emit a confirmatory result unless all 60 blinded cases
have a syntactically complete disposition and the reviewer attests that the key
and source material were unavailable during reading.

Primary: extracellular-mucin score versus continuous frozen p17 abundance.
The estimand is a restricted-pair Kendall tau-b: concordance and both tie
denominators are accumulated only within cohort x KRAS blocks, then combined.
Inference uses 20,000 within-block permutations and the finite-simulation (+1)
one-sided p-value. Confidence intervals use a patient bootstrap stratified by
the 24 cohort x KRAS x p17 sampling cells.

Exploratory only: gland-formation score versus continuous frozen p28
abundance.  It receives an effect estimate and bootstrap interval, but no
confirmatory test or claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_KEY = REPO / "reviews" / "v5" / "KEYS_DO_NOT_DISTRIBUTE" / "case_key.csv"
DEFAULT_PACKET_RECEIPT = REPO / "reviews" / "v5" / "KEYS_DO_NOT_DISTRIBUTE" / "packet_receipt.json"
EXPECTED_CASES = 60
N_PERMUTATIONS = 20_000
PERMUTATION_SEED = 20260824
N_BOOTSTRAP = 2_000
BOOTSTRAP_SEED = 20260825
ATTESTATION = "confirmed_no_key_or_source_access"
MIN_CLINICALLY_MEANINGFUL_TAU = 0.20
MAX_NON_ASSESSABLE_FRACTION = 0.10

MUCIN_CODES = {
    "none": 0,
    "focal_lt10": 1,
    "moderate_10_50": 2,
    "extensive_gt50": 3,
}
GLAND_CODES = {"lt50": 1, "pct50_95": 2, "gt95": 3}
P17_GROUPS = ("absent", "positive_low", "positive_high")
SCORE_COLUMNS = (
    "case_id",
    "assessable",
    "extracellular_mucin_extent",
    "gland_formation",
    "note_if_unusual",
)
REVIEWER_COLUMNS = (
    "reviewer_id",
    "review_date",
    "viewer_software",
    "years_experience_gi_pathology",
    "elapsed_minutes",
    "blinding_attestation",
)


class ValidationError(ValueError):
    """Returned packet is not eligible for confirmatory analysis."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def verify_file_identity(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValidationError(f"sealed {label} is missing: {path}")
    expected_size = record.get("size_bytes")
    expected_sha = record.get("sha256")
    if expected_size is None or expected_sha is None:
        raise ValidationError(f"packet receipt lacks size/hash for {label}")
    if path.stat().st_size != int(expected_size) or sha256(path) != str(expected_sha):
        raise ValidationError(f"sealed {label} identity does not match packet receipt: {path}")


def verify_packet_receipt(receipt_path: Path, key_path: Path) -> dict[str, Any]:
    """Fail closed on any break in the pre-read packet freeze chain."""
    if not receipt_path.is_file():
        raise ValidationError(f"sealed packet receipt does not exist: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"sealed packet receipt is unreadable: {exc}") from exc
    if receipt.get("status") != "PASS" or receipt.get("problems") not in ([], None):
        raise ValidationError("sealed packet receipt is not PASS with zero problems")
    if (
        receipt.get("scientific_status") != "GENERATED_UNREAD"
        or receipt.get("analysis_executed") is not False
        or receipt.get("unblinding_performed") is not False
        or receipt.get("analysis_result") is not None
    ):
        raise ValidationError("packet receipt does not preserve the GENERATED_UNREAD boundary")

    analyzer_record = receipt.get("frozen_analyzer")
    builder_record = receipt.get("builder")
    if not isinstance(analyzer_record, dict) or not isinstance(builder_record, dict):
        raise ValidationError("packet receipt lacks builder or frozen-analyzer identity")
    verify_file_identity(Path(__file__).resolve(), analyzer_record, "frozen analyzer")
    builder_path = Path(str(builder_record.get("path", "")))
    verify_file_identity(builder_path, builder_record, "packet builder")

    packet_root = receipt_path.resolve().parents[1]

    def declared_path(record: dict[str, Any]) -> Path:
        path = Path(str(record.get("path", "")))
        return path if path.is_absolute() else packet_root / path

    outputs = receipt.get("outputs")
    if not isinstance(outputs, list):
        raise ValidationError("packet receipt lacks output identities")
    key_records = [
        record
        for record in outputs
        if isinstance(record, dict) and declared_path(record).resolve() == key_path.resolve()
    ]
    if len(key_records) != 1:
        raise ValidationError("packet receipt does not identify exactly one sealed case key")
    verify_file_identity(key_path.resolve(), key_records[0], "case key")

    manifests = receipt.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValidationError("packet receipt lacks sealed manifest identities")
    for record in manifests:
        if not isinstance(record, dict) or not record.get("path"):
            raise ValidationError("packet receipt contains a malformed manifest identity")
        manifest_path = declared_path(record)
        verify_file_identity(manifest_path, record, f"manifest {record['path']}")
    return receipt


def read_returned_forms(
    scores_path: Path, reviewer_path: Path, key_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate a complete 60-case return before exposing any outcome result."""
    for path in (scores_path, reviewer_path, key_path):
        if not path.is_file():
            raise ValidationError(f"required file does not exist: {path}")

    scores = pd.read_csv(scores_path, dtype=str, keep_default_na=False)
    reviewer = pd.read_csv(reviewer_path, dtype=str, keep_default_na=False)
    key = pd.read_csv(key_path, low_memory=False)

    missing_score_columns = sorted(set(SCORE_COLUMNS) - set(scores.columns))
    if missing_score_columns:
        raise ValidationError(f"scoring form lacks columns: {missing_score_columns}")
    missing_key_columns = sorted(
        {
            "case_id",
            "cohort",
            "kras",
            "p17_group",
            "p17_abundance",
            "p28_abundance",
        }
        - set(key.columns)
    )
    if missing_key_columns:
        raise ValidationError(f"case key lacks columns: {missing_key_columns}")

    scores = scores.loc[:, SCORE_COLUMNS].copy()
    for column in SCORE_COLUMNS:
        scores[column] = scores[column].map(clean_text)
    key["case_id"] = key["case_id"].astype(str).str.strip()

    problems: list[str] = []
    if len(key) != EXPECTED_CASES:
        problems.append(f"case key has {len(key)} rows, expected {EXPECTED_CASES}")
    if len(scores) != EXPECTED_CASES:
        problems.append(f"scoring form has {len(scores)} rows, expected {EXPECTED_CASES}")
    if key["case_id"].duplicated().any():
        problems.append("case key contains duplicate case_id values")
    if scores["case_id"].duplicated().any():
        problems.append("scoring form contains duplicate case_id values")
    expected = set(key["case_id"])
    observed = set(scores["case_id"])
    if expected != observed:
        problems.append(
            "case-id mismatch: missing="
            + repr(sorted(expected - observed))
            + ", unexpected="
            + repr(sorted(observed - expected))
        )

    for row in scores.itertuples(index=False):
        case_id = row.case_id or "<blank>"
        if row.assessable not in {"yes", "no"}:
            problems.append(f"{case_id}: assessable must be yes or no")
            continue
        if row.assessable == "yes":
            if row.extracellular_mucin_extent not in MUCIN_CODES:
                problems.append(f"{case_id}: invalid or blank mucin score")
            if row.gland_formation not in GLAND_CODES:
                problems.append(f"{case_id}: invalid or blank gland score")
        else:
            if row.extracellular_mucin_extent or row.gland_formation:
                problems.append(f"{case_id}: non-assessable case must have blank scores")

    missing_reviewer_columns = sorted(set(REVIEWER_COLUMNS) - set(reviewer.columns))
    if missing_reviewer_columns:
        problems.append(f"reviewer form lacks columns: {missing_reviewer_columns}")
    elif len(reviewer) != 1:
        problems.append(f"reviewer form has {len(reviewer)} rows, expected 1")
    else:
        info = {column: clean_text(reviewer.iloc[0][column]) for column in REVIEWER_COLUMNS}
        for column in ("reviewer_id", "viewer_software"):
            if not info[column]:
                problems.append(f"reviewer form: {column} is blank")
        try:
            date.fromisoformat(info["review_date"])
        except ValueError:
            problems.append("reviewer form: review_date must be ISO YYYY-MM-DD")
        for column, allow_zero in (
            ("years_experience_gi_pathology", True),
            ("elapsed_minutes", False),
        ):
            try:
                value = float(info[column])
                if not math.isfinite(value) or value < 0 or (not allow_zero and value <= 0):
                    raise ValueError
            except ValueError:
                qualifier = "non-negative" if allow_zero else "positive"
                problems.append(f"reviewer form: {column} must be a {qualifier} number")
        if info["blinding_attestation"] != ATTESTATION:
            problems.append("reviewer form: blinding_attestation must be exactly " + ATTESTATION)

    if problems:
        raise ValidationError(
            "CONFIRMATORY ANALYSIS REFUSED. All 60 cases and the blinding record "
            "must be complete before unblinding:\n- " + "\n- ".join(problems)
        )

    merged = key.merge(scores, on="case_id", validate="one_to_one", how="left")
    merged["mucin_score"] = merged["extracellular_mucin_extent"].map(MUCIN_CODES)
    merged["gland_score"] = merged["gland_formation"].map(GLAND_CODES)
    merged["primary_complete"] = (merged["assessable"] == "yes") & merged["mucin_score"].notna()
    merged["exploratory_complete"] = (merged["assessable"] == "yes") & merged["gland_score"].notna()
    merged["analysis_block"] = merged["cohort"].astype(str) + "|" + merged["kras"].astype(str)
    merged["sampling_cell"] = (
        merged["cohort"].astype(str)
        + "|"
        + merged["kras"].astype(str)
        + "|"
        + merged["p17_group"].astype(str)
    )
    return merged, reviewer


def kendall_components(x: Iterable[float], y: Iterable[float]) -> tuple[float, int, int]:
    """Return S=C-D and the two non-tied-pair denominator components."""
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    if xa.size < 2 or ya.size != xa.size:
        return 0.0, 0, 0
    i, j = np.triu_indices(xa.size, k=1)
    dx = np.sign(xa[i] - xa[j])
    dy = np.sign(ya[i] - ya[j])
    return float(np.sum(dx * dy)), int(np.count_nonzero(dx)), int(np.count_nonzero(dy))


def kendall_tau_b(x: Iterable[float], y: Iterable[float]) -> tuple[float, bool]:
    """Small-sample Kendall tau-b, returning whether a block is informative."""
    score, non_tied_x, non_tied_y = kendall_components(x, y)
    denominator = math.sqrt(float(non_tied_x) * float(non_tied_y))
    if denominator == 0:
        return float("nan"), False
    return score / denominator, True


def restricted_pair_tau_b(
    blocks: Iterable[tuple[Iterable[float], Iterable[float]]],
) -> tuple[float, dict[str, Any]]:
    """Combine within-block Kendall components without any cross-block pair."""
    score_total = 0.0
    non_tied_x_total = 0
    non_tied_y_total = 0
    block_count = 0
    for exposure, outcome in blocks:
        score, non_tied_x, non_tied_y = kendall_components(exposure, outcome)
        score_total += score
        non_tied_x_total += non_tied_x
        non_tied_y_total += non_tied_y
        block_count += 1
    denominator = math.sqrt(float(non_tied_x_total) * float(non_tied_y_total))
    statistic = score_total / denominator if denominator else float("nan")
    return statistic, {
        "concordance_score_c_minus_d": score_total,
        "non_tied_exposure_pairs": non_tied_x_total,
        "non_tied_outcome_pairs": non_tied_y_total,
        "denominator": denominator,
        "blocks": block_count,
    }


def fixed_block_sizes(frame: pd.DataFrame) -> dict[str, int]:
    return frame.groupby("analysis_block", sort=True).size().astype(int).to_dict()


def stratified_kendall_tau_b(
    frame: pd.DataFrame,
    exposure: str,
    outcome: str,
    complete: str,
    design_sizes: dict[str, int],
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    """Restricted-pair tau-b aggregated only over cohort x KRAS pairs.

    The numerator and both tie denominators are summed across blocks before the
    ratio is formed. A constant-outcome block therefore contributes its
    non-tied exposure pairs to the denominator instead of being dropped.
    """
    rows: list[dict[str, Any]] = []
    block_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for block in sorted(design_sizes):
        sub = frame[(frame["analysis_block"] == block) & frame[complete]].copy()
        score, non_tied_x, non_tied_y = kendall_components(sub[exposure], sub[outcome])
        tau, informative = kendall_tau_b(sub[exposure], sub[outcome])
        rows.append(
            {
                "analysis_block": block,
                "design_n": int(design_sizes[block]),
                "complete_n": int(len(sub)),
                "tau_b": None if not informative else tau,
                "informative": bool(informative),
                "concordance_score_c_minus_d": score,
                "non_tied_exposure_pairs": non_tied_x,
                "non_tied_outcome_pairs": non_tied_y,
            }
        )
        block_arrays.append(
            (sub[exposure].to_numpy(dtype=float), sub[outcome].to_numpy(dtype=float))
        )
    statistic, components = restricted_pair_tau_b(block_arrays)
    return statistic, rows, components


def adjusted_absent_high_contrast(
    frame: pd.DataFrame,
    outcome: str,
    complete: str,
    design_sizes: dict[str, int],
) -> tuple[float, list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    weighted_numerator = 0.0
    included_weight = 0.0
    for block in sorted(design_sizes):
        sub = frame[(frame["analysis_block"] == block) & frame[complete]]
        absent = sub.loc[sub["p17_group"] == "absent", outcome].astype(float)
        high = sub.loc[sub["p17_group"] == "positive_high", outcome].astype(float)
        informative = bool(len(absent) and len(high))
        difference = float(high.mean() - absent.mean()) if informative else float("nan")
        weight = float(design_sizes[block])
        rows.append(
            {
                "analysis_block": block,
                "design_n": int(design_sizes[block]),
                "absent_complete_n": int(len(absent)),
                "positive_high_complete_n": int(len(high)),
                "mean_score_difference_high_minus_absent": (difference if informative else None),
                "informative": informative,
                "fixed_weight": weight,
            }
        )
        if informative:
            weighted_numerator += weight * difference
            included_weight += weight
    if included_weight == 0:
        return float("nan"), rows, 0.0
    for row in rows:
        row["normalized_weight"] = (
            row["fixed_weight"] / included_weight if row["informative"] else 0.0
        )
    coverage = included_weight / float(sum(design_sizes.values()))
    return weighted_numerator / included_weight, rows, coverage


def permutation_p_value(
    frame: pd.DataFrame,
    observed: float,
    design_sizes: dict[str, int],
) -> tuple[float, np.ndarray]:
    if not math.isfinite(observed):
        raise ValidationError("primary stratified tau-b is not estimable")
    rng = np.random.default_rng(PERMUTATION_SEED)
    complete = frame[frame["primary_complete"]].copy()
    pieces: list[tuple[str, np.ndarray, np.ndarray]] = []
    for block in sorted(design_sizes):
        sub = complete[complete["analysis_block"] == block]
        pieces.append(
            (
                block,
                sub["p17_abundance"].to_numpy(dtype=float),
                sub["mucin_score"].to_numpy(dtype=float),
            )
        )
    null = np.empty(N_PERMUTATIONS, dtype=float)
    for iteration in range(N_PERMUTATIONS):
        null[iteration], _ = restricted_pair_tau_b(
            (exposure, rng.permutation(outcome)) for _block, exposure, outcome in pieces
        )
    valid = null[np.isfinite(null)]
    if len(valid) != N_PERMUTATIONS:
        raise ValidationError("one or more permutation statistics were not estimable")
    p_value = (1.0 + float(np.count_nonzero(valid >= observed))) / (float(N_PERMUTATIONS) + 1.0)
    return p_value, null


def stratified_bootstrap(frame: pd.DataFrame, design_sizes: dict[str, int]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cells = [sub.index.to_numpy() for _, sub in frame.groupby("sampling_cell", sort=True)]
    rows: list[dict[str, Any]] = []
    for draw in range(N_BOOTSTRAP):
        sampled_indices = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in cells]
        )
        sampled = frame.loc[sampled_indices].reset_index(drop=True)
        primary_tau, _, _ = stratified_kendall_tau_b(
            sampled,
            "p17_abundance",
            "mucin_score",
            "primary_complete",
            design_sizes,
        )
        contrast, _, _ = adjusted_absent_high_contrast(
            sampled, "mucin_score", "primary_complete", design_sizes
        )
        exploratory_tau, _, _ = stratified_kendall_tau_b(
            sampled,
            "p28_abundance",
            "gland_score",
            "exploratory_complete",
            design_sizes,
        )
        rows.append(
            {
                "draw": draw + 1,
                "primary_stratified_tau_b": primary_tau,
                "primary_adjusted_high_minus_absent": contrast,
                "exploratory_p28_gland_stratified_tau_b": exploratory_tau,
            }
        )
    return pd.DataFrame(rows)


def percentile_ci(values: pd.Series) -> dict[str, Any]:
    valid = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(valid):
        return {"low": None, "high": None, "valid_draws": 0, "method": "percentile"}
    low, high = np.quantile(valid, [0.025, 0.975], method="linear")
    return {
        "low": float(low),
        "high": float(high),
        "valid_draws": int(len(valid)),
        "method": "sampling-cell-stratified patient percentile bootstrap",
    }


def missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in P17_GROUPS:
        sub = frame[frame["p17_group"] == group]
        assessable = int((sub["assessable"] == "yes").sum())
        total = int(len(sub))
        rows.append(
            {
                "p17_group": group,
                "n_total": total,
                "n_assessable": assessable,
                "n_non_assessable": total - assessable,
                "assessable_fraction": assessable / total if total else None,
                "n_primary_complete": int(sub["primary_complete"].sum()),
                "n_exploratory_complete": int(sub["exploratory_complete"].sum()),
            }
        )
    return pd.DataFrame(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    frame: pd.DataFrame,
    scores_path: Path,
    reviewer_path: Path,
    key_path: Path,
    packet_receipt_path: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise ValidationError(f"append-only analysis output already exists: {output_dir}")

    design_sizes = fixed_block_sizes(frame)
    primary_tau, primary_blocks, primary_components = stratified_kendall_tau_b(
        frame,
        "p17_abundance",
        "mucin_score",
        "primary_complete",
        design_sizes,
    )
    exploratory_tau, exploratory_blocks, exploratory_components = stratified_kendall_tau_b(
        frame,
        "p28_abundance",
        "gland_score",
        "exploratory_complete",
        design_sizes,
    )
    contrast, contrast_blocks, contrast_weight_coverage = adjusted_absent_high_contrast(
        frame, "mucin_score", "primary_complete", design_sizes
    )
    p_value, null = permutation_p_value(frame, primary_tau, design_sizes)
    boot = stratified_bootstrap(frame, design_sizes)
    missing = missingness_table(frame)

    primary_ci = percentile_ci(boot["primary_stratified_tau_b"])
    contrast_ci = percentile_ci(boot["primary_adjusted_high_minus_absent"])
    exploratory_ci = percentile_ci(boot["exploratory_p28_gland_stratified_tau_b"])
    non_assessable = int((frame["assessable"] == "no").sum())
    non_assessable_fraction = non_assessable / EXPECTED_CASES
    evidence_gates = {
        "clinically_meaningful_magnitude": bool(primary_tau >= MIN_CLINICALLY_MEANINGFUL_TAU),
        "one_sided_permutation_p_lt_0_05": bool(p_value < 0.05),
        "bootstrap_ci_lower_gt_0": bool(primary_ci["low"] is not None and primary_ci["low"] > 0),
        "non_assessable_fraction_le_0_10": bool(
            non_assessable_fraction <= MAX_NON_ASSESSABLE_FRACTION
        ),
    }
    if not evidence_gates["non_assessable_fraction_le_0_10"]:
        scientific_status = "INCONCLUSIVE_MISSINGNESS"
    elif all(evidence_gates.values()):
        scientific_status = "SUPPORTS_PRIMARY_CLAIM"
    else:
        scientific_status = "DOES_NOT_SUPPORT_PRIMARY_CLAIM"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent))
    try:
        write_csv(staging / "primary_block_effects.csv", primary_blocks)
        write_csv(staging / "primary_absent_high_block_contrasts.csv", contrast_blocks)
        write_csv(staging / "exploratory_block_effects.csv", exploratory_blocks)
        missing.to_csv(staging / "missingness_by_p17_group.csv", index=False)
        boot.to_csv(staging / "bootstrap_statistics.csv", index=False)
        pd.DataFrame(
            {"permutation": np.arange(1, N_PERMUTATIONS + 1), "stratified_tau_b": null}
        ).to_csv(staging / "primary_permutation_null.csv", index=False)

        result = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ANALYSIS_COMPLETE",
            "scientific_status": scientific_status,
            "cases_required": EXPECTED_CASES,
            "cases_returned": int(len(frame)),
            "primary_complete_cases": int(frame["primary_complete"].sum()),
            "non_assessable_cases": non_assessable,
            "non_assessable_fraction": non_assessable_fraction,
            "primary": {
                "name": "extracellular mucin versus continuous frozen p17 abundance",
                "direction": "higher p17 predicts higher mucin score",
                "stratified_restricted_pair_tau_b": float(primary_tau),
                "stratified_tau_components": primary_components,
                "tau_b_ci_95": primary_ci,
                "permutation_p_one_sided": float(p_value),
                "permutations": N_PERMUTATIONS,
                "permutation_seed": PERMUTATION_SEED,
                "p_value_formula": "(1 + count(null >= observed)) / (20000 + 1)",
                "adjusted_mean_score_difference_positive_high_minus_absent": float(contrast),
                "contrast_ci_95": contrast_ci,
                "contrast_fixed_design_weight_coverage": contrast_weight_coverage,
                "minimum_clinically_meaningful_tau_b": MIN_CLINICALLY_MEANINGFUL_TAU,
                "evidence_gates": evidence_gates,
                "evidence_rule": (
                    "SUPPORTS_PRIMARY_CLAIM only when stratified tau-b >= 0.20, the "
                    "one-sided blocked-permutation p-value is <0.05, the 95% bootstrap "
                    "CI lower bound is >0, and no more than 10% of cases are "
                    "non-assessable. More than 10% non-assessable forces "
                    "INCONCLUSIVE_MISSINGNESS; otherwise failure of any evidence gate "
                    "yields DOES_NOT_SUPPORT_PRIMARY_CLAIM. Estimates are always "
                    "reported."
                ),
            },
            "exploratory": {
                "name": "gland formation versus continuous frozen p28 abundance",
                "status": "exploratory; no confirmatory test or error-controlled claim",
                "stratified_restricted_pair_tau_b": float(exploratory_tau),
                "stratified_tau_components": exploratory_components,
                "tau_b_ci_95": exploratory_ci,
            },
            "contrast_fixed_design_weights": {
                block: count / EXPECTED_CASES for block, count in design_sizes.items()
            },
            "block_rule": (
                "cohort x KRAS block-specific tau-b values are descriptive; the primary "
                "restricted-pair statistic sums C-D and both non-tied-pair denominator "
                "components across all blocks before forming tau-b"
            ),
            "missing_data_rule": (
                "complete-case analysis only; no imputation; non-assessability reported "
                "by p17 exposure group"
            ),
            "bootstrap": {
                "draws": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "unit": "patient",
                "strata": "cohort x KRAS x p17_group sampling cell",
            },
            "score_codes": {"mucin": MUCIN_CODES, "gland": GLAND_CODES},
        }
        result_path = staging / "analysis_result.json"
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

        output_files = sorted(path for path in staging.iterdir() if path.is_file())
        analyzer_path = Path(__file__).resolve()
        receipt = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "problems": [],
            "scientific_status": scientific_status,
            "frozen_analyzer": {
                "path": str(analyzer_path),
                "size_bytes": analyzer_path.stat().st_size,
                "sha256": sha256(analyzer_path),
            },
            "inputs": [
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in (scores_path, reviewer_path, key_path, packet_receipt_path)
            ],
            "outputs": [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in output_files
            ],
            "write_policy": "write-once staging directory atomically renamed after success",
        }
        (staging / "analysis_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path, help="returned scoring_form.csv")
    parser.add_argument(
        "--reviewer-info", required=True, type=Path, help="returned reviewer_info.csv"
    )
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY, help="sealed case key")
    parser.add_argument(
        "--packet-receipt",
        type=Path,
        default=DEFAULT_PACKET_RECEIPT,
        help="sealed v5 packet receipt used to verify the analyzer, builder, key, and manifests",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="new append-only directory for analysis outputs"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the return and print PASS without unblinded effect estimates",
    )
    args = parser.parse_args()
    try:
        verify_packet_receipt(args.packet_receipt, args.key)
        frame, _reviewer = read_returned_forms(args.scores, args.reviewer_info, args.key)
        if args.validate_only:
            print(json.dumps({"status": "PASS", "complete_cases": int(len(frame))}, indent=2))
            return
        if args.output_dir is None:
            raise ValidationError("--output-dir is required unless --validate-only is used")
        run_analysis(
            frame,
            args.scores,
            args.reviewer_info,
            args.key,
            args.packet_receipt,
            args.output_dir,
        )
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
