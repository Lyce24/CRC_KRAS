#!/usr/bin/env python3
"""Independently verify and integrate the sealed final-v7 additions.

The integration has two deliberately different scientific states:

* E2f-v3 is an executed, target-label-internal cross-fitting experiment.  This
  tool rehashes its seal and independently audits the result, OOF, and fit
  artifacts before mechanically deriving the fixed-gate and incremental-value
  verdicts.
* reviews/v5 is a generated-but-unread pathology packet.  This tool verifies
  packet integrity, exhaustive whole-section panel structure, blinding, and
  blank reader forms.  It never treats the packet as a pathology result.

Running without ``--seal`` is read-only.  ``--seal`` stages results.json and
verification.json, writes receipt.json last, fsyncs the files, and atomically
renames the new directory into its append-only destination.  Existing output
or staging directories are always refused.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
E2F_ANALYSIS = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_e2f_fulllabel_adapter_v3_20260821/analysis"
)
REVIEWS_V5 = REPO / "reviews" / "v5"
OUTPUT_ROOT = (
    REPO
    / "reports"
    / "reruns"
    / "final_v7_additions_20260821"
    / "integration"
)

EXPECTED_LAMBDA_GRID = (
    "infinity",
    "10000",
    "3000",
    "1000",
    "300",
    "100",
    "30",
    "10",
    "3",
    "1",
    "0.3",
    "0.1",
    "0.03",
    "0.01",
    "0.003",
    "0.001",
)
EXPECTED_FORM_COLUMNS = (
    "case_id",
    "assessable",
    "extracellular_mucin_extent",
    "gland_formation",
    "note_if_unusual",
)
EXPECTED_REVIEWER_COLUMNS = (
    "reviewer_id",
    "review_date",
    "viewer_software",
    "years_experience_gi_pathology",
    "elapsed_minutes",
    "blinding_attestation",
)
E2F_OUTPUT_NAMES = (
    "results",
    "oof_predictions",
    "fits",
    "support_draws",
    "solver_warnings",
)
REVIEW_MANIFEST_PATHS = (
    "KEYS_DO_NOT_DISTRIBUTE/image_manifest.csv",
    "KEYS_DO_NOT_DISTRIBUTE/panel_manifest.csv",
    "KEYS_DO_NOT_DISTRIBUTE/source_identity_manifest.csv",
    "KEYS_DO_NOT_DISTRIBUTE/case_key.csv",
    "KEYS_DO_NOT_DISTRIBUTE/selection_audit.csv",
    "KEYS_DO_NOT_DISTRIBUTE/excluded_previously_exposed_patients.csv",
    "FOR_PATHOLOGIST/HANDOFF_MANIFEST.sha256",
)
FORBIDDEN_REVIEW_RESULT_NAMES = {
    "analysis_result.json",
    "analysis_results.json",
    "results.json",
    "scientific_results.json",
    "unblinded_results.json",
    "completed_scoring_form.csv",
    "scoring_form_completed.csv",
}
FORBIDDEN_REVIEW_RESULT_DIRS = {
    "analysis_results",
    "completed_results",
    "unblinded_analysis",
}
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
OPAQUE_CASE_RE = re.compile(r"Q\d{5}\Z")


class IntegrationError(RuntimeError):
    """A fail-closed component or integration verification error."""


@dataclass(frozen=True)
class CohortContract:
    n: int
    n_mutant: int
    e2b_native_auroc: float


@dataclass(frozen=True)
class StudyContract:
    cohorts: Mapping[str, CohortContract]
    outer_seeds: tuple[int, ...]
    primary_outer_seed: int
    folds: int
    model_seeds: tuple[int, ...]
    support_sizes: tuple[int, ...]
    support_draws: int
    lambda_grid: tuple[str, ...]
    bootstrap_draws: int
    bootstrap_seed: int
    embedding_dimensions: int
    review_cases: int
    review_panels_per_case: int
    review_prior_exposed: int
    review_p17_counts: Mapping[str, int]
    review_kras_counts: Mapping[str, int]

    @property
    def patient_total(self) -> int:
        return sum(item.n for item in self.cohorts.values())

    @property
    def expected_oof_full(self) -> int:
        return len(self.outer_seeds) * self.patient_total

    @property
    def expected_oof_support(self) -> int:
        return len(self.support_sizes) * self.support_draws * self.patient_total

    @property
    def expected_full_adapter_fits(self) -> int:
        return (
            len(self.outer_seeds)
            * len(self.cohorts)
            * len(self.model_seeds)
            * self.folds
        )

    @property
    def expected_full_platt_fits(self) -> int:
        return len(self.outer_seeds) * len(self.cohorts) * self.folds

    @property
    def expected_support_adapter_fits(self) -> int:
        return (
            len(self.support_sizes)
            * self.support_draws
            * len(self.cohorts)
            * len(self.model_seeds)
            * self.folds
        )

    @property
    def expected_fit_total(self) -> int:
        return (
            self.expected_full_adapter_fits
            + self.expected_full_platt_fits
            + self.expected_support_adapter_fits
        )


PRODUCTION_CONTRACT = StudyContract(
    cohorts={
        "RIH": CohortContract(n=85, n_mutant=37, e2b_native_auroc=0.606982),
        "SurGen": CohortContract(n=74, n_mutant=30, e2b_native_auroc=0.562121),
    },
    outer_seeds=tuple(range(20260817, 20260822)),
    primary_outer_seed=20260821,
    folds=5,
    model_seeds=(42, 43, 44),
    support_sizes=(8, 16, 32, 48),
    support_draws=20,
    lambda_grid=EXPECTED_LAMBDA_GRID,
    bootstrap_draws=10_000,
    bootstrap_seed=20260817,
    embedding_dimensions=512,
    review_cases=60,
    review_panels_per_case=6,
    review_prior_exposed=191,
    review_p17_counts={"absent": 20, "positive_low": 20, "positive_high": 20},
    review_kras_counts={"mutant": 30, "wild_type": 30},
)


@dataclass(frozen=True)
class IntegrationPaths:
    e2f_analysis: Path
    reviews_v5: Path
    output_root: Path
    integration_code: Path
    integration_test: Path
    contract: StudyContract = PRODUCTION_CONTRACT

    @property
    def e2f_receipt(self) -> Path:
        return self.e2f_analysis / "receipt.json"

    @property
    def e2f_results(self) -> Path:
        return self.e2f_analysis / "results.json"

    @property
    def e2f_oof(self) -> Path:
        return self.e2f_analysis / "oof_predictions.csv"

    @property
    def e2f_fits(self) -> Path:
        return self.e2f_analysis / "fits.jsonl"

    @property
    def review_receipt(self) -> Path:
        return self.reviews_v5 / "KEYS_DO_NOT_DISTRIBUTE" / "packet_receipt.json"

    @property
    def review_summary(self) -> Path:
        return self.reviews_v5 / "build_summary.json"

    @property
    def scoring_form(self) -> Path:
        return self.reviews_v5 / "FOR_PATHOLOGIST" / "scoring_form.csv"

    @property
    def reviewer_info(self) -> Path:
        return self.reviews_v5 / "FOR_PATHOLOGIST" / "reviewer_info.csv"


@dataclass(frozen=True)
class IntegrationProduct:
    results: dict[str, Any]
    verification: dict[str, Any]
    input_identities: tuple[dict[str, Any], ...]
    e2f_receipt_identity: dict[str, Any]
    review_receipt_identity: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise IntegrationError(f"missing required file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-standard/non-finite JSON constant {token}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise IntegrationError(f"invalid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrationError(f"JSON artifact must contain an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise IntegrationError(f"blank JSONL record at {path}:{line_number}")
                value = json.loads(line, parse_constant=_reject_constant)
                if not isinstance(value, dict):
                    raise IntegrationError(
                        f"JSONL record is not an object at {path}:{line_number}"
                    )
                rows.append(value)
    except IntegrationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise IntegrationError(f"invalid JSONL artifact {path}: {exc}") from exc
    return rows


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrationError(message)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise IntegrationError(f"{label} must be a finite number")
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise IntegrationError(f"{label} must be a finite number") from exc
    if not math.isfinite(observed):
        raise IntegrationError(f"{label} must be finite")
    return observed


def _integer(value: Any, label: str) -> int:
    observed = _finite(value, label)
    if not observed.is_integer():
        raise IntegrationError(f"{label} must be an integer")
    return int(observed)


def _optional_integer(value: Any, label: str) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _integer(value, label)


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise IntegrationError(f"{label} must be boolean")


def _close(observed: Any, expected: Any, label: str, tolerance: float = 1e-10) -> None:
    left = _finite(observed, label)
    right = _finite(expected, f"{label} expected")
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise IntegrationError(f"{label} mismatch: observed {left}, expected {right}")


def _identity_records(
    value: Any,
    source: Path,
    trail: tuple[str, ...] = (),
) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if "sha256" in value:
            location = f"{source}:{'/'.join(trail) or '<root>'}"
            digest = value.get("sha256")
            size = value.get("size_bytes")
            raw_path = value.get("path")
            if (
                not isinstance(digest, str)
                or SHA256_RE.fullmatch(digest) is None
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or not isinstance(raw_path, str)
                or not raw_path
            ):
                raise IntegrationError(
                    f"incomplete path/size/SHA-256 identity at {location}"
                )
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = source.parent / candidate
            yield {
                "path": str(candidate.resolve()),
                "size_bytes": size,
                "sha256": digest.lower(),
            }
        for key, item in value.items():
            yield from _identity_records(item, source, (*trail, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _identity_records(item, source, (*trail, str(index)))


def verify_identity_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = Path(str(record.get("path", ""))).resolve()
    observed = identity(path)
    if observed["size_bytes"] != record.get("size_bytes"):
        raise IntegrationError(f"{label} size mismatch: {path}")
    if observed["sha256"] != str(record.get("sha256", "")).casefold():
        raise IntegrationError(f"{label} SHA-256 mismatch: {path}")
    return observed


def verify_all_identities(payload: Mapping[str, Any], source: Path) -> list[dict[str, Any]]:
    unique: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in _identity_records(payload, source):
        observed = verify_identity_record(record, f"declared artifact in {source}")
        key = (observed["path"], observed["size_bytes"], observed["sha256"])
        unique[key] = observed
    if not unique:
        raise IntegrationError(f"receipt declares no complete identities: {source}")
    return sorted(unique.values(), key=lambda item: item["path"])


def _deduplicate_identities(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    unique: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record["path"]),
            int(record["size_bytes"]),
            str(record["sha256"]),
        )
        unique[key] = dict(record)
    return tuple(sorted(unique.values(), key=lambda item: item["path"]))


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise IntegrationError(f"invalid or duplicate CSV header: {path}")
            rows = [dict(row) for row in reader]
            return list(reader.fieldnames), rows
    except IntegrationError:
        raise
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise IntegrationError(f"invalid CSV artifact {path}: {exc}") from exc


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise IntegrationError("AUROC inputs are empty or have unequal lengths")
    pairs = sorted(
        ((_finite(score, "AUROC score"), int(label), index) for index, (label, score) in enumerate(zip(labels, scores, strict=True))),
        key=lambda item: item[0],
    )
    ranks = [0.0] * len(pairs)
    cursor = 0
    while cursor < len(pairs):
        end = cursor + 1
        while end < len(pairs) and pairs[end][0] == pairs[cursor][0]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[position] = average_rank
        cursor = end
    positives = sum(item[1] == 1 for item in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        raise IntegrationError("AUROC requires both classes")
    positive_rank_sum = sum(rank for rank, item in zip(ranks, pairs, strict=True) if item[1] == 1)
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def point_metrics(rows: Sequence[Mapping[str, Any]], procedure: str) -> dict[str, float]:
    labels = [int(row["label"]) for row in rows]
    eta = [_finite(row[f"eta_{procedure}"], f"eta_{procedure}") for row in rows]
    probability = [min(max(_sigmoid(value), 1e-9), 1 - 1e-9) for value in eta]
    log_loss = -sum(
        label * math.log(probability_value)
        + (1 - label) * math.log(1 - probability_value)
        for label, probability_value in zip(labels, probability, strict=True)
    ) / len(labels)
    brier = sum(
        (probability_value - label) ** 2
        for label, probability_value in zip(labels, probability, strict=True)
    ) / len(labels)
    return {"auroc": auroc(labels, eta), "log_loss": log_loss, "brier": brier}


def _interval(record: Any, label: str) -> tuple[float, float]:
    if not isinstance(record, list) or len(record) != 2:
        raise IntegrationError(f"{label} must be a two-value interval")
    low = _finite(record[0], f"{label} lower")
    high = _finite(record[1], f"{label} upper")
    if low > high:
        raise IntegrationError(f"{label} interval is reversed")
    return low, high


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise IntegrationError("cannot calculate an empty mean")
    return sum(materialized) / len(materialized)


def verify_e2f_receipt(paths: IntegrationPaths) -> dict[str, Any]:
    receipt_path = paths.e2f_receipt.resolve()
    receipt = load_json(receipt_path)
    _require(receipt.get("schema_version") == 1, "E2f-v3 receipt schema must be 1")
    _require(receipt.get("status") == "PASS", "E2f-v3 receipt status is not PASS")
    _require(receipt.get("problems") == [], "E2f-v3 receipt records problems")
    _require(
        receipt.get("experiment") == "e2f_v3_full_label_residual_adapter",
        "unexpected E2f-v3 experiment identity",
    )
    generating_code = receipt.get("generating_code")
    _require(isinstance(generating_code, dict), "E2f-v3 receipt lacks generating code")
    verify_identity_record(generating_code, "E2f-v3 generating code")

    outputs = receipt.get("outputs")
    _require(isinstance(outputs, dict), "E2f-v3 receipt lacks outputs mapping")
    _require(set(outputs) == set(E2F_OUTPUT_NAMES), "E2f-v3 output inventory is not exact")
    expected_paths = {
        "results": paths.e2f_results,
        "oof_predictions": paths.e2f_oof,
        "fits": paths.e2f_fits,
        "support_draws": paths.e2f_analysis / "support_draws.jsonl",
        "solver_warnings": paths.e2f_analysis / "solver_warnings.jsonl",
    }
    verified_outputs: dict[str, Any] = {}
    for name, expected_path in expected_paths.items():
        record = outputs[name]
        _require(isinstance(record, dict), f"E2f-v3 output {name} identity is malformed")
        _require(
            Path(str(record.get("path", ""))).resolve() == expected_path.resolve(),
            f"E2f-v3 output {name} path does not match the sealed analysis root",
        )
        verified_outputs[name] = verify_identity_record(record, f"E2f-v3 {name}")

    all_identities = verify_all_identities(receipt, receipt_path)
    return {
        "payload": receipt,
        "receipt_identity": identity(receipt_path),
        "outputs": verified_outputs,
        "all_identities": all_identities,
    }


def read_oof_rows(path: Path, contract: StudyContract) -> list[dict[str, Any]]:
    header, raw_rows = read_csv_rows(path)
    required = {
        "phase",
        "outer_seed",
        "is_primary",
        "support_requested",
        "draw",
        "cohort",
        "patient_id",
        "label",
        "fold",
        "eta_native",
        "eta_adapted",
        "eta_platt",
        *{f"eta_native_seed{seed}" for seed in contract.model_seeds},
        *{f"eta_adapted_seed{seed}" for seed in contract.model_seeds},
    }
    _require(required.issubset(header), f"E2f-v3 OOF CSV lacks columns: {sorted(required - set(header))}")
    rows: list[dict[str, Any]] = []
    for row_number, raw in enumerate(raw_rows, start=2):
        phase = str(raw["phase"])
        _require(phase in {"full_label", "support_curve"}, f"invalid OOF phase at row {row_number}")
        cohort = str(raw["cohort"])
        _require(cohort in contract.cohorts, f"invalid OOF cohort at row {row_number}")
        patient_id = str(raw["patient_id"]).strip()
        _require(bool(patient_id), f"blank OOF patient_id at row {row_number}")
        label = _integer(raw["label"], f"OOF label row {row_number}")
        _require(label in {0, 1}, f"non-binary OOF label at row {row_number}")
        fold = _integer(raw["fold"], f"OOF fold row {row_number}")
        _require(0 <= fold < contract.folds, f"OOF fold out of range at row {row_number}")
        parsed: dict[str, Any] = {
            "phase": phase,
            "outer_seed": _integer(raw["outer_seed"], f"OOF outer seed row {row_number}"),
            "is_primary": _boolean(raw["is_primary"], f"OOF primary flag row {row_number}"),
            "support_requested": _optional_integer(
                raw["support_requested"], f"OOF support row {row_number}"
            ),
            "draw": _optional_integer(raw["draw"], f"OOF draw row {row_number}"),
            "cohort": cohort,
            "patient_id": patient_id,
            "label": label,
            "fold": fold,
        }
        for name in ("eta_native", "eta_adapted"):
            parsed[name] = _finite(raw[name], f"OOF {name} row {row_number}")
        if phase == "full_label":
            parsed["eta_platt"] = _finite(
                raw["eta_platt"], f"OOF eta_platt row {row_number}"
            )
            _require(
                parsed["support_requested"] is None and parsed["draw"] is None,
                f"full-label OOF row has support metadata at row {row_number}",
            )
        else:
            _require(
                raw["eta_platt"] is None or not str(raw["eta_platt"]).strip(),
                f"support-curve OOF row unexpectedly contains Platt output at row {row_number}",
            )
            parsed["eta_platt"] = None
            _require(
                parsed["support_requested"] in contract.support_sizes
                and parsed["draw"] is not None
                and 0 <= parsed["draw"] < contract.support_draws,
                f"invalid support/draw metadata at OOF row {row_number}",
            )
        for seed in contract.model_seeds:
            for prefix in ("eta_native_seed", "eta_adapted_seed"):
                name = f"{prefix}{seed}"
                parsed[name] = _finite(raw[name], f"OOF {name} row {row_number}")
        rows.append(parsed)
    return rows


def verify_oof_census(
    rows: Sequence[Mapping[str, Any]], contract: StudyContract
) -> dict[str, Any]:
    full = [row for row in rows if row["phase"] == "full_label"]
    support = [row for row in rows if row["phase"] == "support_curve"]
    _require(len(full) == contract.expected_oof_full, "full-label OOF row census mismatch")
    _require(
        len(support) == contract.expected_oof_support,
        "support-curve OOF row census mismatch",
    )
    _require(
        len(rows) == contract.expected_oof_full + contract.expected_oof_support,
        "total OOF row census mismatch",
    )

    full_groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    support_groups: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    label_by_patient: dict[tuple[str, str], int] = {}
    for row in rows:
        patient_key = (str(row["cohort"]), str(row["patient_id"]))
        if patient_key in label_by_patient:
            _require(
                label_by_patient[patient_key] == row["label"],
                f"OOF label changes across repeats for {patient_key}",
            )
        label_by_patient[patient_key] = int(row["label"])
        if row["phase"] == "full_label":
            full_groups[(int(row["outer_seed"]), str(row["cohort"]))].append(row)
        else:
            support_groups[
                (
                    int(row["support_requested"]),
                    int(row["draw"]),
                    str(row["cohort"]),
                )
            ].append(row)

    expected_full_keys = {
        (seed, cohort) for seed in contract.outer_seeds for cohort in contract.cohorts
    }
    _require(set(full_groups) == expected_full_keys, "full-label OOF group inventory mismatch")
    patient_sets: dict[str, set[str]] = {}
    for (seed, cohort), group in full_groups.items():
        expected = contract.cohorts[cohort]
        ids = [str(row["patient_id"]) for row in group]
        _require(
            len(group) == expected.n and len(ids) == len(set(ids)),
            f"full-label OOF patient census/uniqueness failed for {seed}/{cohort}",
        )
        _require(
            sum(int(row["label"]) for row in group) == expected.n_mutant,
            f"full-label mutant census failed for {seed}/{cohort}",
        )
        _require(
            {int(row["fold"]) for row in group} == set(range(contract.folds)),
            f"full-label fold census failed for {seed}/{cohort}",
        )
        _require(
            all(bool(row["is_primary"]) == (seed == contract.primary_outer_seed) for row in group),
            f"full-label primary flag mismatch for {seed}/{cohort}",
        )
        current = set(ids)
        if cohort in patient_sets:
            _require(
                patient_sets[cohort] == current,
                f"full-label patient set changes across layouts for {cohort}",
            )
        patient_sets[cohort] = current

    expected_support_keys = {
        (support_size, draw, cohort)
        for support_size in contract.support_sizes
        for draw in range(contract.support_draws)
        for cohort in contract.cohorts
    }
    _require(set(support_groups) == expected_support_keys, "support OOF group inventory mismatch")
    for (support_size, draw, cohort), group in support_groups.items():
        expected = contract.cohorts[cohort]
        ids = [str(row["patient_id"]) for row in group]
        _require(
            len(group) == expected.n and len(ids) == len(set(ids)),
            f"support OOF patient census/uniqueness failed for {support_size}/{draw}/{cohort}",
        )
        _require(set(ids) == patient_sets[cohort], f"support OOF patient set mismatch for {cohort}")
        _require(
            all(int(row["outer_seed"]) == contract.primary_outer_seed for row in group)
            and not any(bool(row["is_primary"]) for row in group),
            f"support OOF seed/primary flag mismatch for {support_size}/{draw}/{cohort}",
        )
    return {
        "full_rows": len(full),
        "primary_full_rows": sum(
            int(row["outer_seed"]) == contract.primary_outer_seed for row in full
        ),
        "support_rows": len(support),
        "total_rows": len(rows),
        "full_groups": full_groups,
        "support_groups": support_groups,
        "patient_sets": patient_sets,
    }


def _validate_metric_record(
    record: Any,
    expected_point: float,
    label: str,
) -> tuple[float, float]:
    _require(isinstance(record, dict), f"{label} metric record is malformed")
    _close(record.get("point"), expected_point, f"{label} point")
    return _interval(record.get("ci95"), f"{label} ci95")


def validate_layout_result(
    layout: Mapping[str, Any],
    cohort_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: StudyContract,
    label: str,
) -> dict[str, Any]:
    procedures = ("native", "adapted", "platt")
    metrics = ("auroc", "log_loss", "brier")
    per_cohort = layout.get("per_cohort")
    macro = layout.get("macro")
    _require(isinstance(per_cohort, dict) and set(per_cohort) == set(contract.cohorts), f"{label} cohort metric inventory mismatch")
    _require(isinstance(macro, dict), f"{label} macro metrics are malformed")
    calculated: dict[str, dict[str, dict[str, float]]] = {}
    for cohort, rows in cohort_rows.items():
        block = per_cohort[cohort]
        _require(isinstance(block, dict), f"{label}/{cohort} metric block is malformed")
        _require(block.get("n") == contract.cohorts[cohort].n, f"{label}/{cohort} n mismatch")
        _require(
            block.get("n_mutant") == contract.cohorts[cohort].n_mutant,
            f"{label}/{cohort} mutant n mismatch",
        )
        calculated[cohort] = {procedure: point_metrics(rows, procedure) for procedure in procedures}
        procedure_records = block.get("procedures")
        contrast_records = block.get("contrasts")
        _require(isinstance(procedure_records, dict), f"{label}/{cohort} procedures malformed")
        _require(isinstance(contrast_records, dict), f"{label}/{cohort} contrasts malformed")
        for procedure in procedures:
            _require(procedure in procedure_records, f"{label}/{cohort} lacks {procedure}")
            for metric in metrics:
                _validate_metric_record(
                    procedure_records[procedure].get(metric),
                    calculated[cohort][procedure][metric],
                    f"{label}/{cohort}/{procedure}/{metric}",
                )
        for procedure in ("adapted", "platt"):
            contrast_name = f"{procedure}_minus_native"
            _require(contrast_name in contrast_records, f"{label}/{cohort} lacks {contrast_name}")
            for metric in metrics:
                expected = (
                    calculated[cohort][procedure][metric]
                    - calculated[cohort]["native"][metric]
                )
                _validate_metric_record(
                    contrast_records[contrast_name].get(metric),
                    expected,
                    f"{label}/{cohort}/{contrast_name}/{metric}",
                )

    macro_procedures = macro.get("procedures")
    macro_contrasts = macro.get("contrasts")
    _require(isinstance(macro_procedures, dict), f"{label} macro procedures malformed")
    _require(isinstance(macro_contrasts, dict), f"{label} macro contrasts malformed")
    for procedure in procedures:
        for metric in metrics:
            expected = _mean(
                calculated[cohort][procedure][metric] for cohort in contract.cohorts
            )
            _validate_metric_record(
                macro_procedures[procedure].get(metric),
                expected,
                f"{label}/macro/{procedure}/{metric}",
            )
    for procedure in ("adapted", "platt"):
        contrast_name = f"{procedure}_minus_native"
        for metric in metrics:
            expected = _mean(
                calculated[cohort][procedure][metric]
                - calculated[cohort]["native"][metric]
                for cohort in contract.cohorts
            )
            _validate_metric_record(
                macro_contrasts[contrast_name].get(metric),
                expected,
                f"{label}/macro/{contrast_name}/{metric}",
            )

    adapted_macro_ci = _interval(
        macro_procedures["adapted"]["auroc"].get("ci95"),
        f"{label} macro adapted AUROC",
    )
    derived_gate = {
        "both_cohort_points_above_0p5": all(
            calculated[cohort]["adapted"]["auroc"] > 0.5
            for cohort in contract.cohorts
        ),
        "macro_ci_lower_above_0p5": adapted_macro_ci[0] > 0.5,
    }
    derived_gate["pass"] = all(derived_gate.values())
    _require(
        layout.get("fixed_gate_adapted") == derived_gate,
        f"{label} fixed gate is not mechanically derived",
    )

    delta_ci = _interval(
        macro_contrasts["adapted_minus_native"]["auroc"].get("ci95"),
        f"{label} macro adapted-minus-native AUROC",
    )
    derived_incremental = {
        "macro_auroc_delta_ci_lower_above_zero": delta_ci[0] > 0,
        "both_cohort_auroc_delta_points_above_zero": all(
            calculated[cohort]["adapted"]["auroc"]
            - calculated[cohort]["native"]["auroc"]
            > 0
            for cohort in contract.cohorts
        ),
    }
    derived_incremental["pass"] = all(derived_incremental.values())
    _require(
        layout.get("incremental_improvement_established") == derived_incremental,
        f"{label} incremental verdict is not mechanically derived",
    )
    requested_draws = _integer(
        layout.get("bootstrap_draws_requested"),
        f"{label} requested bootstrap draws",
    )
    _require(
        requested_draws == contract.bootstrap_draws,
        f"{label} bootstrap draw contract mismatch",
    )
    _require(
        layout.get("bootstrap_seed") == contract.bootstrap_seed,
        f"{label} bootstrap seed mismatch",
    )
    valid_draws = _integer(
        layout.get("bootstrap_draws_valid"),
        f"{label} valid bootstrap draws",
    )
    _require(
        valid_draws == requested_draws,
        f"{label} valid bootstrap draw count mismatch",
    )
    return {
        "calculated_points": calculated,
        "fixed_gate": derived_gate,
        "incremental_improvement": derived_incremental,
    }


def _validate_inner_selection(
    selection: Any,
    selected_lambda: str,
    n_fit: int,
    contract: StudyContract,
    label: str,
) -> None:
    _require(isinstance(selection, dict), f"{label} lacks inner selection audit")
    _require(selection.get("selected_lambda") == selected_lambda, f"{label} selected lambda disagrees with inner audit")
    _require(selection.get("n_pool") == n_fit, f"{label} inner selection pool size mismatch")
    _require(
        selection.get("scheme") in {"leave_one_out", "five_fold"},
        f"{label} inner selection scheme is invalid",
    )
    for key in ("losses_by_lambda", "mean_loss_by_lambda", "solver_summary_by_lambda"):
        inventory = selection.get(key)
        _require(isinstance(inventory, dict), f"{label} lacks {key}")
        _require(set(inventory) == set(contract.lambda_grid), f"{label} {key} grid is incomplete")
    selected_loss = selection["mean_loss_by_lambda"].get(selected_lambda)
    _finite(selected_loss, f"{label} selected-lambda mean loss")


def _validate_solver_diagnostic(
    diagnostic: Any,
    coefficients: Sequence[Any],
    bias: Any,
    selected_lambda: str | None,
    label: str,
) -> None:
    _require(isinstance(diagnostic, dict), f"{label} solver diagnostic is malformed")
    status = diagnostic.get("status")
    _require(status in {"native_exact", "finite_optimum"}, f"{label} solver status is invalid")
    coefficient_values = [_finite(value, f"{label} coefficient") for value in coefficients]
    bias_value = _finite(bias, f"{label} bias")
    objective_zero = _finite(diagnostic.get("objective_at_zero"), f"{label} objective_at_zero")
    objective_fit = _finite(diagnostic.get("objective_at_fit"), f"{label} objective_at_fit")
    decrease = _finite(diagnostic.get("objective_decrease"), f"{label} objective_decrease")
    _require(
        decrease >= -1e-9 and math.isclose(objective_zero - objective_fit, decrease, rel_tol=1e-8, abs_tol=1e-8),
        f"{label} solver objective/decrease audit failed",
    )
    if status == "native_exact":
        _require(selected_lambda == "infinity", f"{label} native-exact fit lacks infinite lambda")
        _require(
            all(value == 0 for value in coefficient_values) and bias_value == 0,
            f"{label} native-exact fit is not an exact zero residual",
        )
    else:
        gradient = _finite(diagnostic.get("gradient_inf_norm"), f"{label} gradient")
        accepted_by = diagnostic.get("accepted_by")
        _require(
            accepted_by in {"scipy_success", "small_gradient"},
            f"{label} solver acceptance mode is invalid",
        )
        if accepted_by == "small_gradient":
            _require(gradient <= 1e-6, f"{label} small-gradient acceptance exceeds tolerance")


def _derive_lambda_inventory(
    rows: Sequence[Mapping[str, Any]],
    contract: StudyContract,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for cohort in contract.cohorts:
        selected = [str(row["selected_lambda"]) for row in rows if row["cohort"] == cohort]
        counts = dict(sorted(Counter(selected).items()))
        output[cohort] = {
            "n": len(selected),
            "counts": counts,
            "fraction_infinity": sum(item == "infinity" for item in selected) / len(selected),
            "fraction_at_lower_boundary_0p001": sum(item == contract.lambda_grid[-1] for item in selected) / len(selected),
        }
    return output


def _compare_lambda_inventory(observed: Any, expected: Mapping[str, Any], label: str) -> None:
    _require(isinstance(observed, dict) and set(observed) == set(expected), f"{label} cohort inventory mismatch")
    for cohort, expected_record in expected.items():
        record = observed[cohort]
        _require(isinstance(record, dict), f"{label}/{cohort} is malformed")
        _require(record.get("n") == expected_record["n"], f"{label}/{cohort} n mismatch")
        _require(record.get("counts") == expected_record["counts"], f"{label}/{cohort} counts mismatch")
        _close(record.get("fraction_infinity"), expected_record["fraction_infinity"], f"{label}/{cohort} infinity fraction")
        _close(
            record.get("fraction_at_lower_boundary_0p001"),
            expected_record["fraction_at_lower_boundary_0p001"],
            f"{label}/{cohort} lower-boundary fraction",
        )


def verify_fit_records(
    records: Sequence[Mapping[str, Any]],
    oof_audit: Mapping[str, Any],
    results: Mapping[str, Any],
    contract: StudyContract,
) -> dict[str, Any]:
    _require(len(records) == contract.expected_fit_total, "E2f-v3 fit-record total mismatch")
    counter = Counter((row.get("phase"), row.get("model_kind")) for row in records)
    expected_counter = {
        ("full_label", "residual_adapter"): contract.expected_full_adapter_fits,
        ("full_label", "platt"): contract.expected_full_platt_fits,
        ("support_curve", "residual_adapter"): contract.expected_support_adapter_fits,
    }
    _require(counter == expected_counter, f"E2f-v3 adapter/Platt fit census mismatch: {counter}")

    full_groups = oof_audit["full_groups"]
    support_groups = oof_audit["support_groups"]
    patient_sets = oof_audit["patient_sets"]
    full_seen: set[tuple[Any, ...]] = set()
    support_seen: set[tuple[Any, ...]] = set()
    support_fit_sets: dict[tuple[int, int, str, int], tuple[frozenset[str], int]] = {}
    full_adapter_rows: list[Mapping[str, Any]] = []
    support_adapter_rows: list[Mapping[str, Any]] = []

    for index, row in enumerate(records, start=1):
        label = f"fit record {index}"
        phase = row.get("phase")
        model_kind = row.get("model_kind")
        cohort = row.get("cohort")
        _require(cohort in contract.cohorts, f"{label} cohort is invalid")
        outer_seed = _integer(row.get("outer_seed"), f"{label} outer seed")
        fold = _integer(row.get("outer_fold"), f"{label} outer fold")
        _require(0 <= fold < contract.folds, f"{label} fold is out of range")
        fit_ids_raw = row.get("fit_patient_ids")
        test_ids_raw = row.get("test_patient_ids")
        fit_labels_raw = row.get("fit_labels")
        _require(isinstance(fit_ids_raw, list) and isinstance(test_ids_raw, list), f"{label} lacks fit/test IDs")
        _require(isinstance(fit_labels_raw, list), f"{label} lacks fit labels")
        fit_ids = [str(value) for value in fit_ids_raw]
        test_ids = [str(value) for value in test_ids_raw]
        fit_labels = [_integer(value, f"{label} fit label") for value in fit_labels_raw]
        _require(set(fit_labels).issubset({0, 1}), f"{label} has a non-binary fit label")
        _require(len(fit_ids) == len(set(fit_ids)) and len(test_ids) == len(set(test_ids)), f"{label} has duplicate fit/test IDs")
        _require(not (set(fit_ids) & set(test_ids)), f"{label} has fit/test leakage")
        _require(len(fit_ids) == len(fit_labels) == row.get("n_fit"), f"{label} fit census mismatch")
        _require(len(test_ids) == row.get("n_test"), f"{label} test census mismatch")
        _require(fit_labels.count(0) == row.get("n_fit_class0"), f"{label} class-0 census mismatch")
        _require(fit_labels.count(1) == row.get("n_fit_class1"), f"{label} class-1 census mismatch")
        coefficients = row.get("coefficients")
        _require(isinstance(coefficients, list), f"{label} coefficients are malformed")
        expected_dimension = contract.embedding_dimensions if model_kind == "residual_adapter" else 2
        _require(len(coefficients) == expected_dimension, f"{label} coefficient dimension mismatch")
        selected_lambda = row.get("selected_lambda")
        selected_text = None if selected_lambda is None else str(selected_lambda)
        _validate_solver_diagnostic(
            row.get("solver_diagnostic"), coefficients, row.get("bias"), selected_text, label
        )

        if phase == "full_label":
            _require(outer_seed in contract.outer_seeds, f"{label} full-label seed is invalid")
            expected_test = {
                str(item["patient_id"])
                for item in full_groups[(outer_seed, str(cohort))]
                if int(item["fold"]) == fold
            }
            _require(set(test_ids) == expected_test, f"{label} test IDs disagree with OOF fold")
            _require(
                set(fit_ids) == set(patient_sets[str(cohort)]) - expected_test,
                f"{label} full-label training pool is not exact",
            )
            if model_kind == "residual_adapter":
                model_seed = _integer(row.get("model_seed"), f"{label} model seed")
                _require(model_seed in contract.model_seeds, f"{label} model seed is invalid")
                key = (outer_seed, cohort, model_seed, fold)
                _require(key not in full_seen, f"duplicate full-label adapter fit {key}")
                full_seen.add(key)
                _require(selected_text in contract.lambda_grid, f"{label} selected lambda is outside grid")
                _require(
                    selected_text != contract.lambda_grid[-1],
                    f"{label} selected expanded-grid lower boundary {contract.lambda_grid[-1]}",
                )
                _validate_inner_selection(
                    row.get("inner_selection"), selected_text, len(fit_ids), contract, label
                )
                full_adapter_rows.append(row)
            else:
                _require(model_kind == "platt", f"{label} full-label model kind is invalid")
                _require(row.get("model_seed") is None and selected_lambda is None, f"{label} Platt metadata is invalid")
                key = (outer_seed, cohort, "platt", fold)
                _require(key not in full_seen, f"duplicate full-label Platt fit {key}")
                full_seen.add(key)
        else:
            _require(phase == "support_curve" and model_kind == "residual_adapter", f"{label} phase/model combination is invalid")
            _require(outer_seed == contract.primary_outer_seed, f"{label} support outer seed mismatch")
            support = _integer(row.get("support_requested"), f"{label} support")
            draw = _integer(row.get("draw"), f"{label} support draw")
            model_seed = _integer(row.get("model_seed"), f"{label} model seed")
            support_seed = _integer(row.get("support_seed"), f"{label} support seed")
            _require(support in contract.support_sizes, f"{label} support is invalid")
            _require(0 <= draw < contract.support_draws, f"{label} draw is invalid")
            _require(model_seed in contract.model_seeds, f"{label} model seed is invalid")
            _require(row.get("support_realized") == support == len(fit_ids), f"{label} exact support realization failed")
            _require(
                fit_labels.count(0) == fit_labels.count(1) == support // 2,
                f"{label} support is not exactly balanced",
            )
            expected_test = {
                str(item["patient_id"])
                for item in support_groups[(support, draw, str(cohort))]
                if int(item["fold"]) == fold
            }
            _require(set(test_ids) == expected_test, f"{label} support test IDs disagree with OOF fold")
            key = (support, draw, cohort, model_seed, fold)
            _require(key not in support_seen, f"duplicate support adapter fit {key}")
            support_seen.add(key)
            shared_key = (support, draw, str(cohort), fold)
            shared_value = (frozenset(fit_ids), support_seed)
            if shared_key in support_fit_sets:
                _require(
                    support_fit_sets[shared_key] == shared_value,
                    f"{label} support patients/seed differ across model seeds",
                )
            support_fit_sets[shared_key] = shared_value
            _require(selected_text in contract.lambda_grid, f"{label} selected lambda is outside grid")
            _validate_inner_selection(
                row.get("inner_selection"), selected_text, len(fit_ids), contract, label
            )
            support_adapter_rows.append(row)

    _require(len(full_seen) == contract.expected_full_adapter_fits + contract.expected_full_platt_fits, "full-label fit key census mismatch")
    _require(len(support_seen) == contract.expected_support_adapter_fits, "support fit key census mismatch")

    primary_rows = [
        row for row in full_adapter_rows if row.get("outer_seed") == contract.primary_outer_seed
    ]
    derived_primary = _derive_lambda_inventory(primary_rows, contract)
    derived_all = _derive_lambda_inventory(full_adapter_rows, contract)
    derived_support = _derive_lambda_inventory(support_adapter_rows, contract)
    _compare_lambda_inventory(results.get("lambda_inventory_primary"), derived_primary, "primary lambda inventory")
    _compare_lambda_inventory(results.get("lambda_inventory_full_label_all_layouts"), derived_all, "all-layout lambda inventory")
    _compare_lambda_inventory(results.get("lambda_inventory_support_curve"), derived_support, "support lambda inventory")
    _require(
        all(record["fraction_at_lower_boundary_0p001"] == 0 for record in derived_all.values()),
        "expanded full-label grid remains boundary-truncated",
    )
    return {
        "fit_records_total": len(records),
        "full_label_adapter": counter[("full_label", "residual_adapter")],
        "full_label_platt": counter[("full_label", "platt")],
        "support_adapter": counter[("support_curve", "residual_adapter")],
        "fit_test_intersections": 0,
        "exact_support_and_balance_failures": 0,
        "full_label_lower_boundary_hits": 0,
        "lambda_inventory_primary": derived_primary,
        "lambda_inventory_full_label_all_layouts": derived_all,
        "lambda_inventory_support_curve": derived_support,
    }


def _validate_input_census(results: Mapping[str, Any], contract: StudyContract) -> None:
    census = results.get("input_census")
    _require(isinstance(census, dict) and set(census) == set(contract.cohorts), "E2f-v3 input cohort census mismatch")
    for cohort, expected in contract.cohorts.items():
        record = census[cohort]
        _require(isinstance(record, dict), f"E2f-v3 {cohort} input census is malformed")
        _require(record.get("n_patients") == expected.n, f"E2f-v3 {cohort} patient census mismatch")
        _require(record.get("n_mutant") == expected.n_mutant, f"E2f-v3 {cohort} mutant census mismatch")
        _require(record.get("n_embedding_dimensions") == contract.embedding_dimensions, f"E2f-v3 {cohort} embedding dimension mismatch")
        _require(record.get("model_seeds") == list(contract.model_seeds), f"E2f-v3 {cohort} seed census mismatch")
        _require(record.get("label_discordances") == 0, f"E2f-v3 {cohort} has label discordance")
        _require(record.get("nonfinite_values") == 0, f"E2f-v3 {cohort} has non-finite inputs")


def _recompute_support_curve(
    support_groups: Mapping[tuple[int, int, str], Sequence[Mapping[str, Any]]],
    contract: StudyContract,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for support in contract.support_sizes:
        macro_delta: list[float] = []
        cohort_delta: dict[str, list[float]] = {cohort: [] for cohort in contract.cohorts}
        log_loss: dict[str, dict[str, list[float]]] = {
            cohort: {"native": [], "adapted": []} for cohort in contract.cohorts
        }
        brier: dict[str, dict[str, list[float]]] = {
            cohort: {"native": [], "adapted": []} for cohort in contract.cohorts
        }
        for draw in range(contract.support_draws):
            current_delta: list[float] = []
            for cohort in contract.cohorts:
                rows = support_groups[(support, draw, cohort)]
                native = point_metrics(rows, "native")
                adapted = point_metrics(rows, "adapted")
                delta = adapted["auroc"] - native["auroc"]
                cohort_delta[cohort].append(delta)
                current_delta.append(delta)
                for procedure, metrics in (("native", native), ("adapted", adapted)):
                    log_loss[cohort][procedure].append(metrics["log_loss"])
                    brier[cohort][procedure].append(metrics["brier"])
            macro_delta.append(_mean(current_delta))
        output[str(support)] = {
            "requested_support": support,
            "realized_support_min": support,
            "realized_support_max": support,
            "exact_balanced_contract": True,
            "n_draws": contract.support_draws,
            "macro_delta_auroc_mean": _mean(macro_delta),
            "macro_delta_auroc_sd": statistics.stdev(macro_delta),
            "per_cohort_delta_auroc_mean": {
                cohort: _mean(values) for cohort, values in cohort_delta.items()
            },
            "per_cohort_log_loss_mean": {
                cohort: {
                    procedure: _mean(values)
                    for procedure, values in procedures.items()
                }
                for cohort, procedures in log_loss.items()
            },
            "per_cohort_brier_mean": {
                cohort: {
                    procedure: _mean(values)
                    for procedure, values in procedures.items()
                }
                for cohort, procedures in brier.items()
            },
        }
    return output


def _compare_nested_numbers(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        _require(isinstance(observed, dict) and set(observed) == set(expected), f"{label} key inventory mismatch")
        for key, value in expected.items():
            _compare_nested_numbers(observed[key], value, f"{label}/{key}")
    elif isinstance(expected, bool):
        _require(observed is expected, f"{label} boolean mismatch")
    elif isinstance(expected, int):
        _require(observed == expected, f"{label} integer mismatch")
    elif isinstance(expected, float):
        _close(observed, expected, label)
    else:
        _require(observed == expected, f"{label} value mismatch")


def verify_e2f_results(
    paths: IntegrationPaths,
    receipt_audit: Mapping[str, Any],
) -> dict[str, Any]:
    contract = paths.contract
    results = load_json(paths.e2f_results)
    _require(results.get("schema_version") == 1, "E2f-v3 results schema must be 1")
    _require(results.get("status") == "PASS", "E2f-v3 results status is not PASS")
    _require(results.get("problems") == [], "E2f-v3 results record problems")
    _require(
        results.get("experiment") == "e2f_v3_full_label_residual_adapter",
        "unexpected E2f-v3 results experiment identity",
    )
    scope_warning = str(results.get("scope_warning", ""))
    _require(
        "target-label" in scope_warning.casefold()
        and "not independent deployment" in scope_warning.casefold()
        and "clinical-readiness" in scope_warning.casefold(),
        "E2f-v3 scope warning omits the internal-label/deployment boundary",
    )
    design = results.get("design")
    _require(isinstance(design, dict), "E2f-v3 design block is malformed")
    _require(design.get("residual_parameters") == contract.embedding_dimensions + 1, "E2f-v3 residual parameter census mismatch")
    _require(tuple(design.get("lambda_grid", ())) == contract.lambda_grid, "E2f-v3 expanded lambda grid mismatch")
    _require(design.get("primary_outer_seed") == contract.primary_outer_seed, "E2f-v3 primary outer seed mismatch")
    _require(tuple(design.get("outer_sensitivity_seeds", ())) == contract.outer_seeds, "E2f-v3 outer-layout inventory mismatch")
    _require(design.get("outer_folds") == contract.folds, "E2f-v3 fold count mismatch")
    _require(tuple(design.get("model_seeds", ())) == contract.model_seeds, "E2f-v3 model seed inventory mismatch")
    _require(design.get("cohorts_fit_separately") is True, "E2f-v3 cohorts were not fit separately")
    _require(tuple(design.get("support_sizes_exact_total", ())) == contract.support_sizes, "E2f-v3 support-size inventory mismatch")
    _require(design.get("support_draws") == contract.support_draws, "E2f-v3 support draw count mismatch")
    _require(design.get("bootstrap_draws") == contract.bootstrap_draws, "E2f-v3 bootstrap draw count mismatch")
    _require(design.get("bootstrap_seed") == contract.bootstrap_seed, "E2f-v3 bootstrap seed mismatch")
    _require("exactly half" in str(design.get("support_balance", "")), "E2f-v3 exact balanced support contract is missing")
    _validate_input_census(results, contract)

    oof_rows = read_oof_rows(paths.e2f_oof, contract)
    oof_audit = verify_oof_census(oof_rows, contract)
    layouts = results.get("outer_fold_sensitivity")
    _require(isinstance(layouts, dict) and set(layouts) == {str(seed) for seed in contract.outer_seeds}, "E2f-v3 layout result inventory mismatch")
    layout_audits: dict[str, Any] = {}
    for seed in contract.outer_seeds:
        cohort_rows = {
            cohort: oof_audit["full_groups"][(seed, cohort)]
            for cohort in contract.cohorts
        }
        layout_audits[str(seed)] = validate_layout_result(
            layouts[str(seed)], cohort_rows, contract, f"outer layout {seed}"
        )
    primary = results.get("primary")
    _require(isinstance(primary, dict), "E2f-v3 primary result is malformed")
    _require(primary == layouts[str(contract.primary_outer_seed)], "E2f-v3 primary result differs from its declared layout")
    primary_audit = layout_audits[str(contract.primary_outer_seed)]

    baseline = results.get("frozen_baseline_checks")
    _require(isinstance(baseline, dict) and set(baseline) == set(contract.cohorts), "E2f-v3 E2b baseline inventory mismatch")
    for cohort, expected in contract.cohorts.items():
        record = baseline[cohort]
        _require(isinstance(record, dict) and record.get("pass") is True, f"E2f-v3 {cohort} E2b reproduction did not pass")
        _close(record.get("expected"), expected.e2b_native_auroc, f"E2f-v3 {cohort} frozen E2b expected", tolerance=5e-7)
        recomputed = primary_audit["calculated_points"][cohort]["native"]["auroc"]
        _close(record.get("observed"), recomputed, f"E2f-v3 {cohort} E2b observed")
        _require(abs(recomputed - expected.e2b_native_auroc) < 5e-7, f"E2f-v3 {cohort} failed independent E2b reproduction")

    primary_delta = primary["macro"]["contrasts"]["adapted_minus_native"]["auroc"]
    _require(results.get("primary_macro_delta_auroc") == primary_delta, "E2f-v3 primary macro delta alias mismatch")
    _require(
        results.get("primary_incremental_improvement_established")
        == primary_audit["incremental_improvement"],
        "E2f-v3 primary incremental alias is not mechanical",
    )

    summary = results.get("outer_fold_sensitivity_summary")
    _require(isinstance(summary, dict), "E2f-v3 outer-layout summary is malformed")
    expected_gate_count = sum(audit["fixed_gate"]["pass"] for audit in layout_audits.values())
    adapted_points = [
        layouts[str(seed)]["macro"]["procedures"]["adapted"]["auroc"]["point"]
        for seed in contract.outer_seeds
    ]
    delta_points = [
        layouts[str(seed)]["macro"]["contrasts"]["adapted_minus_native"]["auroc"]["point"]
        for seed in contract.outer_seeds
    ]
    _require(summary.get("n_layouts") == len(contract.outer_seeds), "E2f-v3 layout summary count mismatch")
    _require(summary.get("n_gate_pass") == expected_gate_count, "E2f-v3 layout gate count mismatch")
    for observed_range, values, label in (
        (summary.get("macro_adapted_auroc_range"), adapted_points, "adapted AUROC range"),
        (summary.get("macro_delta_auroc_range"), delta_points, "delta AUROC range"),
    ):
        low, high = _interval(observed_range, f"E2f-v3 {label}")
        _close(low, min(values), f"E2f-v3 {label} lower")
        _close(high, max(values), f"E2f-v3 {label} upper")

    support_recomputed = _recompute_support_curve(oof_audit["support_groups"], contract)
    _compare_nested_numbers(
        results.get("label_efficiency_curve"),
        support_recomputed,
        "E2f-v3 label-efficiency curve",
    )

    fits = load_jsonl(paths.e2f_fits)
    fit_audit = verify_fit_records(fits, oof_audit, results, contract)
    artifact_census = results.get("artifact_census")
    _require(isinstance(artifact_census, dict), "E2f-v3 artifact census is malformed")
    expected_artifact_values = {
        "oof_rows_total": len(oof_rows),
        "oof_rows_full_label": contract.expected_oof_full,
        "oof_rows_primary_full_label": contract.patient_total,
        "oof_rows_support_curve": contract.expected_oof_support,
        "fit_records_total": len(fits),
        "fit_records_full_label_adapter": contract.expected_full_adapter_fits,
        "fit_records_full_label_platt": contract.expected_full_platt_fits,
        "fit_records_support_adapter": contract.expected_support_adapter_fits,
        "support_draw_records": len(contract.support_sizes) * contract.support_draws,
    }
    for key, expected in expected_artifact_values.items():
        _require(artifact_census.get(key) == expected, f"E2f-v3 artifact census mismatch for {key}")
    solver = results.get("solver")
    _require(isinstance(solver, dict), "E2f-v3 solver summary is malformed")
    _require(solver.get("n_unaccepted") == 0, "E2f-v3 solver records unaccepted fits")
    _require(_integer(solver.get("n_finite_calls"), "E2f-v3 finite solver calls") > 0, "E2f-v3 has no finite solver calls")
    _finite(solver.get("max_gradient_inf_norm"), "E2f-v3 maximum gradient")
    _finite(solver.get("min_objective_decrease"), "E2f-v3 minimum objective decrease")

    receipt = receipt_audit["payload"]
    _require(receipt.get("census") == artifact_census, "E2f-v3 receipt/results census mismatch")
    _require(receipt.get("solver") == solver, "E2f-v3 receipt/results solver summary mismatch")
    _require(
        receipt.get("fixed_gate_primary") == primary_audit["fixed_gate"],
        "E2f-v3 receipt fixed gate is not mechanical",
    )
    _require(
        receipt.get("outer_fold_sensitivity_summary") == summary,
        "E2f-v3 receipt outer-layout summary mismatch",
    )
    return {
        "payload": results,
        "oof": oof_audit,
        "fits": fit_audit,
        "layouts": layout_audits,
        "primary": primary_audit,
        "label_efficiency": support_recomputed,
        "artifact_census": artifact_census,
        "solver": solver,
    }


def _verify_generated_unread(payload: Mapping[str, Any], label: str) -> None:
    _require(payload.get("scientific_status") == "GENERATED_UNREAD", f"{label} scientific_status is not GENERATED_UNREAD")
    _require(payload.get("analysis_executed") is False, f"{label} analysis_executed is not false")
    _require(payload.get("unblinding_performed") is False, f"{label} unblinding_performed is not false")
    _require("analysis_result" in payload and payload.get("analysis_result") is None, f"{label} analysis_result is not explicit null")


def _output_record_map(receipt: Mapping[str, Any], review_root: Path) -> dict[Path, dict[str, Any]]:
    outputs = receipt.get("outputs")
    _require(isinstance(outputs, list) and outputs, "reviews/v5 receipt lacks output identities")
    mapped: dict[Path, dict[str, Any]] = {}
    for index, record in enumerate(outputs):
        _require(isinstance(record, dict), f"reviews/v5 output identity {index} is malformed")
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = review_root / path
        path = path.resolve()
        _require(path.is_relative_to(review_root.resolve()), f"reviews/v5 output escapes packet root: {path}")
        _require(path not in mapped, f"duplicate reviews/v5 output identity: {path}")
        mapped[path] = dict(record)
    return mapped


def _verify_blank_forms(paths: IntegrationPaths, case_ids: set[str]) -> dict[str, Any]:
    score_header, scores = read_csv_rows(paths.scoring_form)
    reviewer_header, reviewers = read_csv_rows(paths.reviewer_info)
    _require(tuple(score_header) == EXPECTED_FORM_COLUMNS, "reviews/v5 scoring-form columns mismatch")
    _require(tuple(reviewer_header) == EXPECTED_REVIEWER_COLUMNS, "reviews/v5 reviewer-form columns mismatch")
    _require(len(scores) == paths.contract.review_cases, "reviews/v5 scoring-form row count mismatch")
    observed_ids: list[str] = []
    for row_number, row in enumerate(scores, start=2):
        case_id = str(row.get("case_id", "")).strip()
        _require(case_id, f"reviews/v5 scoring form lacks case ID at row {row_number}")
        _require(
            not any(str(row.get(column, "")).strip() for column in EXPECTED_FORM_COLUMNS if column != "case_id"),
            f"reviews/v5 scoring form is not blank at row {row_number}",
        )
        observed_ids.append(case_id)
    _require(len(observed_ids) == len(set(observed_ids)), "reviews/v5 scoring form has duplicate IDs")
    _require(set(observed_ids) == case_ids, "reviews/v5 scoring form case IDs differ from packet")
    _require(len(reviewers) == 1, "reviews/v5 reviewer form must contain one blank row")
    _require(
        not any(str(value).strip() for value in reviewers[0].values()),
        "reviews/v5 reviewer information form is not blank",
    )

    score_template = paths.reviews_v5 / "KEYS_DO_NOT_DISTRIBUTE" / "scoring_form_TEMPLATE.csv"
    reviewer_template = paths.reviews_v5 / "KEYS_DO_NOT_DISTRIBUTE" / "reviewer_info_TEMPLATE.csv"
    _require(
        score_template.read_bytes() == paths.scoring_form.read_bytes(),
        "reviews/v5 sealed scoring template differs from reader form",
    )
    _require(
        reviewer_template.read_bytes() == paths.reviewer_info.read_bytes(),
        "reviews/v5 sealed reviewer template differs from reader form",
    )
    return {
        "case_rows": len(scores),
        "scored_cells_nonblank": 0,
        "reviewer_rows": len(reviewers),
        "reviewer_cells_nonblank": 0,
        "template_copies_byte_identical": True,
    }


def _rectangles_partition(rows: Sequence[Mapping[str, str]], width: int, height: int) -> bool:
    area = 0
    rectangles: list[tuple[int, int, int, int]] = []
    for row in rows:
        x0 = _integer(row.get("level0_x0"), "panel x0")
        y0 = _integer(row.get("level0_y0"), "panel y0")
        x1 = _integer(row.get("level0_x1"), "panel x1")
        y1 = _integer(row.get("level0_y1"), "panel y1")
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            return False
        area += (x1 - x0) * (y1 - y0)
        rectangles.append((x0, y0, x1, y1))
    if area != width * height:
        return False
    for index, first in enumerate(rectangles):
        for second in rectangles[index + 1 :]:
            overlap_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
            overlap_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
            if overlap_width * overlap_height:
                return False
    return True


def _verify_handoff_manifest(reader_root: Path) -> None:
    manifest = reader_root / "HANDOFF_MANIFEST.sha256"
    _require(manifest.is_file(), "reviews/v5 reader handoff manifest is missing")
    records: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        _require(match is not None, f"malformed reader handoff manifest line {line_number}")
        assert match is not None
        digest, relative = match.groups()
        _require(relative not in records, f"duplicate reader handoff path: {relative}")
        records[relative] = digest
    expected = {
        path.relative_to(reader_root).as_posix(): path
        for path in reader_root.rglob("*")
        if path.is_file() and path != manifest
    }
    _require(set(records) == set(expected), "reader handoff manifest file inventory mismatch")
    for relative, path in expected.items():
        _require(sha256_file(path) == records[relative], f"reader handoff hash mismatch: {relative}")


def _verify_no_analysis_result(review_root: Path) -> int:
    checked = 0
    forbidden: list[str] = []
    for path in sorted(review_root.rglob("*")):
        if not path.is_file():
            continue
        checked += 1
        relative = path.relative_to(review_root)
        directory_tokens = {part.casefold() for part in relative.parts[:-1]}
        if (
            path.name.casefold() in FORBIDDEN_REVIEW_RESULT_NAMES
            or directory_tokens & FORBIDDEN_REVIEW_RESULT_DIRS
        ):
            forbidden.append(relative.as_posix())
    _require(not forbidden, f"reviews/v5 contains a scientific result despite GENERATED_UNREAD: {forbidden}")
    return checked


def verify_reviews_v5(paths: IntegrationPaths) -> dict[str, Any]:
    contract = paths.contract
    root = paths.reviews_v5.resolve()
    _require(root.is_dir(), f"reviews/v5 root is missing: {root}")
    receipt = load_json(paths.review_receipt)
    summary = load_json(paths.review_summary)
    _require(receipt.get("schema_version") == 2, "reviews/v5 receipt schema must be 2")
    _require(receipt.get("status") == "PASS", "reviews/v5 receipt status is not PASS")
    _require(receipt.get("problems") == [], "reviews/v5 receipt records problems")
    _verify_generated_unread(receipt, "reviews/v5 receipt")
    _require(summary.get("status") == "PASS", "reviews/v5 build summary status is not PASS")
    _require(summary.get("problems") == [], "reviews/v5 build summary records problems")
    _verify_generated_unread(summary, "reviews/v5 build summary")

    all_identities = verify_all_identities(receipt, paths.review_receipt)
    outputs = _output_record_map(receipt, root)
    receipt_path = paths.review_receipt.resolve()
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file() and path.resolve() != receipt_path}
    _require(set(outputs) == actual_files, "reviews/v5 receipt output inventory does not exactly cover packet files")
    for path, record in outputs.items():
        verify_identity_record(record, f"reviews/v5 output {path.relative_to(root)}")

    manifest_records = receipt.get("manifests")
    _require(isinstance(manifest_records, list), "reviews/v5 receipt lacks manifest identities")
    manifest_paths = {
        Path(str(record.get("path", ""))).resolve()
        for record in manifest_records
        if isinstance(record, dict)
    }
    expected_manifests = {(root / relative).resolve() for relative in REVIEW_MANIFEST_PATHS}
    _require(manifest_paths == expected_manifests, "reviews/v5 manifest identity inventory mismatch")
    for record in manifest_records:
        _require(isinstance(record, dict), "reviews/v5 manifest identity is malformed")
        verify_identity_record(record, "reviews/v5 manifest")

    census = receipt.get("census")
    expected_images = contract.review_cases * (contract.review_panels_per_case + 1)
    expected_census = {
        "cases": contract.review_cases,
        "images": expected_images,
        "overviews": contract.review_cases,
        "panels": contract.review_cases * contract.review_panels_per_case,
        "exact_images_per_case": contract.review_panels_per_case + 1,
    }
    _require(census == expected_census, "reviews/v5 receipt census is not exact")
    for key, value in (
        ("cases", contract.review_cases),
        ("images", expected_images),
        ("overviews", contract.review_cases),
        ("panels", contract.review_cases * contract.review_panels_per_case),
        ("images_per_case", contract.review_panels_per_case + 1),
        ("prior_exposed_union", contract.review_prior_exposed),
        ("selected_prior_exposure_overlap", 0),
    ):
        _require(summary.get(key) == value, f"reviews/v5 build-summary {key} mismatch")
    _require(summary.get("by_p17_group") == dict(contract.review_p17_counts), "reviews/v5 p17-group census mismatch")
    _require(summary.get("by_kras") == dict(contract.review_kras_counts), "reviews/v5 KRAS census mismatch")
    _require(sum(summary.get("by_cohort", {}).values()) == contract.review_cases, "reviews/v5 cohort census mismatch")

    panel_design = str(receipt.get("panel_design", "")).casefold()
    blinding = str(receipt.get("blinding", "")).casefold()
    _require(all(token in panel_design for token in ("disjoint", "partition", "full main-image")), "reviews/v5 receipt lacks whole-section partition declaration")
    _require(all(token in blinding for token in ("opaque", "fresh rgb jpeg", "blank forms", "no raw wsi/path/key")), "reviews/v5 receipt lacks structural blinding declaration")

    case_header, case_rows = read_csv_rows(root / "KEYS_DO_NOT_DISTRIBUTE" / "case_key.csv")
    required_case = {"case_id", "patient_id", "slide_id", "cohort", "kras", "p17_group"}
    _require(required_case.issubset(case_header), "reviews/v5 case key lacks required fields")
    _require(len(case_rows) == contract.review_cases, "reviews/v5 case-key row census mismatch")
    case_ids = [str(row["case_id"]).strip() for row in case_rows]
    _require(len(case_ids) == len(set(case_ids)) and all(OPAQUE_CASE_RE.fullmatch(case_id) for case_id in case_ids), "reviews/v5 case IDs are not unique opaque Qxxxxx tokens")
    case_id_set = set(case_ids)
    case_by_cohort = dict(sorted(Counter(row["cohort"] for row in case_rows).items()))
    case_by_kras = dict(sorted(Counter(row["kras"] for row in case_rows).items()))
    case_by_p17_group = dict(
        sorted(Counter(row["p17_group"] for row in case_rows).items())
    )
    _require(
        case_by_p17_group == dict(contract.review_p17_counts),
        "reviews/v5 case-key p17 census mismatch",
    )
    _require(
        case_by_kras == dict(contract.review_kras_counts),
        "reviews/v5 case-key KRAS census mismatch",
    )
    _require(
        summary.get("by_cohort") == case_by_cohort,
        "reviews/v5 cohort census differs between build summary and case key",
    )

    blank_forms = _verify_blank_forms(paths, case_id_set)
    image_header, image_rows = read_csv_rows(root / "KEYS_DO_NOT_DISTRIBUTE" / "image_manifest.csv")
    panel_header, panel_rows = read_csv_rows(root / "KEYS_DO_NOT_DISTRIBUTE" / "panel_manifest.csv")
    source_header, source_rows = read_csv_rows(root / "KEYS_DO_NOT_DISTRIBUTE" / "source_identity_manifest.csv")
    _require({"file", "case_id", "image_role", "width_px", "height_px", "size_bytes", "sha256"}.issubset(image_header), "reviews/v5 image manifest lacks fields")
    _require({"case_id", "panel_number", "image_file", "level0_x0", "level0_y0", "level0_x1", "level0_y1", "actual_output_mpp_x", "actual_output_mpp_y", "size_bytes", "sha256"}.issubset(panel_header), "reviews/v5 panel manifest lacks fields")
    _require({"case_id", "patient_id", "slide_id", "source_path", "source_size_bytes", "source_sha256", "openslide_quickhash1", "level0_width_px", "level0_height_px"}.issubset(source_header), "reviews/v5 source manifest lacks fields")
    _require(len(image_rows) == expected_images, "reviews/v5 image-manifest row census mismatch")
    _require(len(panel_rows) == contract.review_cases * contract.review_panels_per_case, "reviews/v5 panel-manifest row census mismatch")
    _require(len(source_rows) == contract.review_cases, "reviews/v5 source-manifest row census mismatch")

    image_dir = root / "FOR_PATHOLOGIST" / "images"
    jpeg_files = sorted(image_dir.glob("*.jpg"))
    _require(len(jpeg_files) == expected_images, "reviews/v5 filesystem JPEG census mismatch")
    _require(not [path for path in image_dir.iterdir() if path.is_file() and path.suffix.casefold() != ".jpg"], "reviews/v5 reader image directory contains non-JPEG files")
    images_by_case: dict[str, set[str]] = defaultdict(set)
    for path in jpeg_files:
        overview_match = re.fullmatch(r"(Q\d{5})_overview\.jpg", path.name)
        panel_match = re.fullmatch(r"(Q\d{5})_panel([1-9]\d*)\.jpg", path.name)
        _require(overview_match is not None or panel_match is not None, f"reviews/v5 non-opaque JPEG filename: {path.name}")
        if overview_match is not None:
            images_by_case[overview_match.group(1)].add("overview")
        else:
            assert panel_match is not None
            number = _integer(panel_match.group(2), f"reviews/v5 panel number {path.name}")
            _require(1 <= number <= contract.review_panels_per_case, f"reviews/v5 panel number out of range: {path.name}")
            images_by_case[panel_match.group(1)].add(f"panel{number}")
    expected_roles = {"overview", *{f"panel{number}" for number in range(1, contract.review_panels_per_case + 1)}}
    _require(set(images_by_case) == case_id_set, "reviews/v5 JPEG case IDs differ from case key")
    _require(all(roles == expected_roles for roles in images_by_case.values()), "reviews/v5 is not cases x (overview + exact panels)")

    image_by_name = {str(row["file"]): row for row in image_rows}
    _require(len(image_by_name) == len(image_rows), "reviews/v5 image manifest has duplicate files")
    _require(set(image_by_name) == {path.name for path in jpeg_files}, "reviews/v5 image manifest/filesystem mismatch")
    _require(len({row["sha256"] for row in image_rows}) == len(image_rows), "reviews/v5 contains duplicate JPEG hashes")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a project dependency
        raise IntegrationError("Pillow is required for structural JPEG verification") from exc
    Image.MAX_IMAGE_PIXELS = None
    allowed_info = {"jfif", "jfif_version", "jfif_unit", "jfif_density"}
    for path in jpeg_files:
        row = image_by_name[path.name]
        _require(Path(str(outputs[path.resolve()]["path"])).resolve() == path.resolve(), f"reviews/v5 JPEG lacks exact receipt identity: {path.name}")
        _require(path.stat().st_size == _integer(row["size_bytes"], f"{path.name} manifest size"), f"reviews/v5 JPEG size differs from manifest: {path.name}")
        _require(sha256_file(path) == str(row["sha256"]), f"reviews/v5 JPEG hash differs from manifest: {path.name}")
        try:
            with Image.open(path) as image:
                _require(image.format == "JPEG" and image.mode == "RGB", f"reviews/v5 image is not RGB JPEG: {path.name}")
                _require(image.size == (_integer(row["width_px"], f"{path.name} width"), _integer(row["height_px"], f"{path.name} height")), f"reviews/v5 JPEG dimensions differ from manifest: {path.name}")
                _require(len(image.getexif()) == 0, f"reviews/v5 JPEG contains EXIF: {path.name}")
                _require(not (set(image.info) - allowed_info), f"reviews/v5 JPEG contains unexpected metadata: {path.name}")
                image.verify()
        except IntegrationError:
            raise
        except Exception as exc:
            raise IntegrationError(f"reviews/v5 JPEG decode failed: {path.name}: {exc}") from exc

    source_by_case = {str(row["case_id"]): row for row in source_rows}
    _require(set(source_by_case) == case_id_set, "reviews/v5 source manifest case inventory mismatch")
    receipt_sources = receipt.get("source_wsi_inputs")
    _require(isinstance(receipt_sources, list) and len(receipt_sources) == contract.review_cases, "reviews/v5 source-WSI receipt census mismatch")
    receipt_source_map = {Path(str(record.get("path", ""))).resolve(): record for record in receipt_sources if isinstance(record, dict)}
    _require(len(receipt_source_map) == contract.review_cases, "reviews/v5 source-WSI paths are not unique")
    for case_id, row in source_by_case.items():
        source_path = Path(str(row["source_path"])).resolve()
        record = receipt_source_map.get(source_path)
        _require(record is not None, f"reviews/v5 source WSI is absent from receipt: {case_id}")
        _require(
            str(record.get("openslide_quickhash1", ""))
            == str(row["openslide_quickhash1"]),
            f"reviews/v5 source WSI OpenSlide quickhash differs from manifest: {case_id}",
        )
        _require(str(record.get("sha256")) == row["source_sha256"], f"reviews/v5 source WSI hash differs from manifest: {case_id}")
        _require(record.get("size_bytes") == _integer(row["source_size_bytes"], f"{case_id} source size"), f"reviews/v5 source WSI size differs from manifest: {case_id}")

    panels_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in panel_rows:
        panels_by_case[str(row["case_id"])].append(row)
    _require(set(panels_by_case) == case_id_set, "reviews/v5 panel manifest case inventory mismatch")
    for case_id, rows in panels_by_case.items():
        _require(len(rows) == contract.review_panels_per_case, f"reviews/v5 {case_id} panel count mismatch")
        _require(
            sorted(_integer(row["panel_number"], f"{case_id} panel number") for row in rows)
            == list(range(1, contract.review_panels_per_case + 1)),
            f"reviews/v5 {case_id} panel numbering mismatch",
        )
        source = source_by_case[case_id]
        _require(
            _rectangles_partition(
                rows,
                _integer(source["level0_width_px"], f"{case_id} source width"),
                _integer(source["level0_height_px"], f"{case_id} source height"),
            ),
            f"reviews/v5 {case_id} panels do not exactly partition the WSI canvas",
        )
        for row in rows:
            _require(abs(_finite(row["actual_output_mpp_x"], f"{case_id} panel mpp-x") - 2.0) <= 0.01, f"reviews/v5 {case_id} panel mpp-x outside tolerance")
            _require(abs(_finite(row["actual_output_mpp_y"], f"{case_id} panel mpp-y") - 2.0) <= 0.01, f"reviews/v5 {case_id} panel mpp-y outside tolerance")

    reader_root = root / "FOR_PATHOLOGIST"
    _verify_handoff_manifest(reader_root)
    reader_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in reader_root.rglob("*")
        if path.is_file() and path.suffix.casefold() != ".jpg"
    )
    forbidden_values = {
        str(row[field])
        for row in case_rows
        for field in ("patient_id", "slide_id")
        if str(row.get(field, ""))
    } | {str(row["source_path"]) for row in source_rows}
    leaked = sorted(value for value in forbidden_values if value and value in reader_text)
    _require(not leaked, f"reviews/v5 reader payload leaks source identity/path: {leaked[:3]}")
    files_checked = _verify_no_analysis_result(root)

    return {
        "payload": receipt,
        "summary": summary,
        "receipt_identity": identity(paths.review_receipt),
        "all_identities": all_identities,
        "output_files_verified": len(outputs),
        "source_wsi_inputs_verified": len(receipt_source_map),
        "census": expected_census,
        "sampling_strata": {
            "by_cohort": case_by_cohort,
            "by_kras": case_by_kras,
            "by_p17_group": case_by_p17_group,
            "prior_exposed_union": summary["prior_exposed_union"],
            "selected_prior_exposure_overlap": summary[
                "selected_prior_exposure_overlap"
            ],
        },
        "blank_forms": blank_forms,
        "whole_section_partitions_verified": len(panels_by_case),
        "structural_blinding": {
            "opaque_case_ids": True,
            "fresh_rgb_jpegs_without_exif_or_unexpected_metadata": True,
            "reader_source_identity_or_path_leaks": 0,
            "reader_handoff_hashes": "PASS",
        },
        "scientific_status": "GENERATED_UNREAD",
        "analysis_executed": False,
        "unblinding_performed": False,
        "analysis_result": None,
        "files_scanned_for_absent_result": files_checked,
    }


def _report_metric_extract(primary: Mapping[str, Any], contract: StudyContract) -> dict[str, Any]:
    per_cohort: dict[str, Any] = {}
    for cohort in contract.cohorts:
        block = primary["per_cohort"][cohort]
        per_cohort[cohort] = {
            "n": block["n"],
            "n_mutant": block["n_mutant"],
            "native": block["procedures"]["native"],
            "adapted": block["procedures"]["adapted"],
            "platt": block["procedures"]["platt"],
            "adapted_minus_native": block["contrasts"]["adapted_minus_native"],
            "platt_minus_native": block["contrasts"]["platt_minus_native"],
        }
    return {
        "primary_outer_seed": contract.primary_outer_seed,
        "contrast_sign_convention": primary["contrast_sign_convention"],
        "per_cohort": per_cohort,
        "macro": primary["macro"],
        "fixed_gate_adapted": primary["fixed_gate_adapted"],
        "incremental_improvement_established": primary[
            "incremental_improvement_established"
        ],
        "bootstrap_draws_requested": primary["bootstrap_draws_requested"],
        "bootstrap_draws_valid": primary["bootstrap_draws_valid"],
        "bootstrap_seed": primary["bootstrap_seed"],
    }


def _report_layout_extract(
    layouts: Mapping[str, Any], contract: StudyContract
) -> dict[str, Any]:
    """Return the complete, deterministic report-facing fold-layout subset."""
    extracted: dict[str, Any] = {}
    for seed in contract.outer_seeds:
        key = str(seed)
        layout = layouts[key]
        extracted[key] = {
            "outer_seed": seed,
            "is_primary": seed == contract.primary_outer_seed,
            "macro_adapted_auroc": layout["macro"]["procedures"]["adapted"][
                "auroc"
            ],
            "macro_adapted_minus_native_auroc": layout["macro"]["contrasts"][
                "adapted_minus_native"
            ]["auroc"],
            "fixed_gate_adapted": layout["fixed_gate_adapted"],
            "incremental_improvement_established": layout[
                "incremental_improvement_established"
            ],
            "bootstrap_draws_requested": layout["bootstrap_draws_requested"],
            "bootstrap_draws_valid": layout["bootstrap_draws_valid"],
        }
    return extracted


def _format_interval(record: Mapping[str, Any]) -> str:
    point = _finite(record["point"], "report-ready point")
    low, high = _interval(record["ci95"], "report-ready interval")
    return f"{point:.3f} [{low:.3f}, {high:.3f}]"


def _claim_boundaries(primary: Mapping[str, Any]) -> dict[str, str]:
    fixed_pass = bool(primary["fixed_gate_adapted"]["pass"])
    incremental_pass = bool(primary["incremental_improvement_established"]["pass"])
    fixed_phrase = "passed" if fixed_pass else "did not pass"
    incremental_phrase = "was established" if incremental_pass else "was not established"
    return {
        "e2f_v3": (
            "In target-label-internal metastatic cross-fitting, the expanded-grid "
            f"full-label residual adapter {fixed_phrase} the prespecified fixed "
            f"discrimination gate; consistent incremental improvement over the native "
            f"model {incremental_phrase}. This is adaptation-feasibility evidence, not "
            "independent deployment validation, prospective validation, or evidence of "
            "clinical readiness."
        ),
        "reviews_v5": (
            "reviews/v5 is a sealed, structurally blinded whole-section packet in "
            "GENERATED_UNREAD state. It contributes no reader score, association estimate, "
            "pathology validation result, or scientific support for the central claim."
        ),
        "central_claim_boundary": (
            "Keep fixed-threshold feasibility separate from incremental value. E2f-v3 "
            "uses target labels internally and therefore cannot convert internal OOF "
            "performance into external clinical utility. The unread pathology packet is "
            "protocol/integrity evidence only."
        ),
    }


def build_integration(paths: IntegrationPaths) -> IntegrationProduct:
    """Read and verify both sealed components without writing anything."""
    e2f_receipt = verify_e2f_receipt(paths)
    e2f = verify_e2f_results(paths, e2f_receipt)
    review = verify_reviews_v5(paths)
    primary = e2f["payload"]["primary"]
    metric_extract = _report_metric_extract(primary, paths.contract)
    layout_extract = _report_layout_extract(
        e2f["payload"]["outer_fold_sensitivity"], paths.contract
    )
    claim_boundaries = _claim_boundaries(primary)
    macro = primary["macro"]
    report_ready_sentences = {
        "macro_native_auroc": (
            "Primary-layout macro native AUROC: "
            + _format_interval(macro["procedures"]["native"]["auroc"])
            + "."
        ),
        "macro_adapted_auroc": (
            "Primary-layout macro adapted AUROC: "
            + _format_interval(macro["procedures"]["adapted"]["auroc"])
            + "."
        ),
        "macro_adapted_minus_native_auroc": (
            "Primary-layout paired macro adapted-minus-native AUROC: "
            + _format_interval(macro["contrasts"]["adapted_minus_native"]["auroc"])
            + "."
        ),
        "verdict": claim_boundaries["e2f_v3"],
        "pathology_state": claim_boundaries["reviews_v5"],
    }
    created = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    component_states = {
        "e2f_v3": {
            "integrity_status": "PASS",
            "scientific_status": "EXECUTED",
        },
        "reviews_v5": {
            "integrity_status": "PASS",
            "scientific_status": "GENERATED_UNREAD",
            "analysis_executed": False,
            "unblinding_performed": False,
            "analysis_result": None,
        },
    }
    report_contract = {
        "schema_version": 1,
        "component_states": component_states,
        "e2f_v3": {
            "primary_metrics": metric_extract,
            "outer_fold_layouts": layout_extract,
            "outer_fold_sensitivity_summary": e2f["payload"][
                "outer_fold_sensitivity_summary"
            ],
            "label_efficiency_curve": e2f["label_efficiency"],
        },
        "reviews_v5": {
            **component_states["reviews_v5"],
            "census": review["census"],
            "sampling_strata": review["sampling_strata"],
        },
    }
    results = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": created,
        "integration": "final_v7_e2f_v3_and_reviews_v5",
        "component_states": component_states,
        "e2f_v3": {
            "scope_warning": e2f["payload"]["scope_warning"],
            "primary_metrics": metric_extract,
            "outer_fold_layouts": layout_extract,
            "outer_fold_sensitivity_summary": e2f["payload"][
                "outer_fold_sensitivity_summary"
            ],
            "label_efficiency_curve": e2f["label_efficiency"],
            "lambda_inventory_primary": e2f["fits"]["lambda_inventory_primary"],
            "lambda_inventory_full_label_all_layouts": e2f["fits"][
                "lambda_inventory_full_label_all_layouts"
            ],
            "solver": e2f["solver"],
        },
        "reviews_v5": {
            "integrity_status": "PASS",
            "scientific_status": "GENERATED_UNREAD",
            "census": review["census"],
            "sampling_strata": review["sampling_strata"],
            "blank_reader_forms": review["blank_forms"],
            "structural_blinding": review["structural_blinding"],
            "scientific_result": None,
        },
        "report_contract": report_contract,
        "report_ready_sentences": report_ready_sentences,
        "claim_boundaries": claim_boundaries,
    }
    verification = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": created,
        "checks": {
            "e2f_v3_receipt_all_declared_identities_rehashed": "PASS",
            "e2f_v3_adapter_and_oof_census": "PASS",
            "e2f_v3_fit_test_leakage": "0 intersections / PASS",
            "e2f_v3_exact_supports_and_class_balance": "PASS",
            "e2f_v3_finite_solver_and_objective": "PASS",
            "e2f_v3_expanded_grid_full_label_nonboundary": "PASS",
            "e2f_v3_e2b_reproduction": "2/2 PASS",
            "e2f_v3_gate_mechanically_derived": "PASS",
            "e2f_v3_incremental_verdict_mechanically_derived": "PASS",
            "e2f_v3_label_efficiency_recomputed_from_oof": "PASS",
            "reviews_v5_receipt_all_declared_identities_rehashed": "PASS",
            "reviews_v5_60_by_overview_plus_6": "PASS",
            "reviews_v5_structural_blinding": "PASS",
            "reviews_v5_blank_reader_forms": "PASS",
            "reviews_v5_generated_unread_no_result": "PASS",
        },
        "e2f_v3": {
            "receipt": e2f_receipt["receipt_identity"],
            "declared_files_rehashed": len(e2f_receipt["all_identities"]),
            "artifact_census": e2f["artifact_census"],
            "fit_audit": e2f["fits"],
            "derived_primary_fixed_gate": e2f["primary"]["fixed_gate"],
            "derived_primary_incremental_improvement": e2f["primary"][
                "incremental_improvement"
            ],
        },
        "reviews_v5": {
            "receipt": review["receipt_identity"],
            "declared_files_rehashed": len(review["all_identities"]),
            "output_files_verified": review["output_files_verified"],
            "source_wsi_inputs_verified": review["source_wsi_inputs_verified"],
            "whole_section_partitions_verified": review[
                "whole_section_partitions_verified"
            ],
            "scientific_status": "GENERATED_UNREAD",
            "analysis_result": None,
        },
        "claim_boundary_enforcement": claim_boundaries,
    }
    input_identities = _deduplicate_identities(
        [
            identity(paths.e2f_receipt),
            *e2f_receipt["all_identities"],
            identity(paths.review_receipt),
            *review["all_identities"],
        ]
    )
    return IntegrationProduct(
        results=results,
        verification=verification,
        input_identities=input_identities,
        e2f_receipt_identity=e2f_receipt["receipt_identity"],
        review_receipt_identity=review["receipt_identity"],
    )


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_once_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _staged_identity(path: Path, staging: Path, final_root: Path) -> dict[str, Any]:
    return {
        "path": str((final_root / path.relative_to(staging)).resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def seal_integration(paths: IntegrationPaths) -> Path:
    """Verify first, then atomically publish a new write-once integration root."""
    output = paths.output_root.resolve()
    staging = output.parent / f".{output.name}.building"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"append-only destination already exists: {output}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"stale integration staging directory exists: {staging}")
    product = build_integration(paths)
    code_identities = (
        identity(paths.integration_code),
        identity(paths.integration_test),
    )
    results_bytes = json_bytes(product.results)
    verification_bytes = json_bytes(product.verification)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(exist_ok=False)
    results_path = staging / "results.json"
    verification_path = staging / "verification.json"
    receipt_path = staging / "receipt.json"
    _write_once_fsync(results_path, results_bytes)
    _write_once_fsync(verification_path, verification_bytes)
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": product.results["created_utc"],
        "append_only": True,
        "inputs": list(product.input_identities),
        "outputs": [
            _staged_identity(results_path, staging, output),
            _staged_identity(verification_path, staging, output),
        ],
        "code": list(code_identities),
        "components": {
            "e2f_v3": {
                "integrity_status": "PASS",
                "scientific_status": "EXECUTED",
                "receipt": product.e2f_receipt_identity,
            },
            "reviews_v5": {
                "integrity_status": "PASS",
                "scientific_status": "GENERATED_UNREAD",
                "analysis_executed": False,
                "unblinding_performed": False,
                "analysis_result": None,
                "review_root": str(paths.reviews_v5.resolve()),
                "receipt": product.review_receipt_identity,
                "reader_scoring_form": identity(paths.scoring_form),
                "reviewer_info_form": identity(paths.reviewer_info),
            },
        },
        "write_policy": (
            "results.json and verification.json written and fsynced first; "
            "receipt.json written and fsynced last; staged directory atomically renamed; "
            "existing destination and staging refused"
        ),
        "scientific_boundary": (
            "E2f-v3 is executed target-label-internal evidence. reviews/v5 is "
            "GENERATED_UNREAD packet-integrity evidence with no scientific result."
        ),
    }
    _write_once_fsync(receipt_path, json_bytes(receipt))
    directory_fd = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"append-only destination appeared during staging: {output}")
    staging.rename(output)
    parent_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return output


def default_paths(args: argparse.Namespace) -> IntegrationPaths:
    return IntegrationPaths(
        e2f_analysis=args.e2f_analysis,
        reviews_v5=args.reviews_v5,
        output_root=args.output_root,
        integration_code=Path(__file__).resolve(),
        integration_test=REPO / "tests" / "test_final_v7_integration.py",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2f-analysis", type=Path, default=E2F_ANALYSIS)
    parser.add_argument("--reviews-v5", type=Path, default=REVIEWS_V5)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--seal",
        action="store_true",
        help="atomically write the append-only integration; omit for read-only verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = default_paths(args)
    if args.seal:
        output = seal_integration(paths)
        print(f"PASS: sealed final-v7 integration at {output}")
    else:
        product = build_integration(paths)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "read-only; integration not written",
                    "e2f_v3": product.results["component_states"]["e2f_v3"],
                    "reviews_v5": product.results["component_states"]["reviews_v5"],
                    "output_root": str(paths.output_root),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
