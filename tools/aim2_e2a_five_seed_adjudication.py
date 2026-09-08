#!/usr/bin/env python3
"""Governed FINAL-v9 adjudication of five-seed Aim-2 E2a results.

This is a post-outcome robustness extension, not a retrospective
preregistration.  It consumes the *completed and sealed* study-wide five-seed
campaign and reapplies the unchanged FINAL-v8 E2a decision rules:

* every E2a-F family direction passes only when its patient-bootstrap AUROC
  lower 95% confidence bound exceeds 0.50;
* every E2a-D sibling-stratum direction uses the same lower-bound gate;
* a macro can summarize a panel but cannot rescue a failed direction;
* 10,000 target-by-KRAS patient-bootstrap draws use seed 20260817;
* paired scorers use exactly the same target-patient resamples; and
* the secondary five-acquisition-domain and descriptive six-stratum macros
  are recomputed in every draw.

The source campaign is always read-only.  ``adjudicate`` writes only beneath a
separate ``final_v9_adjudication`` root and only with explicit ``--apply``.
``verify`` is deterministic and read-only: it rehashes all bound inputs and
outputs and independently recomputes every load-bearing value and bootstrap
distribution.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
import shutil
import sys
import uuid
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.dont_write_bytecode = True

_LIBC = ctypes.CDLL(None, use_errno=True)
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


SCHEMA_VERSION = 1
ALL_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_817

REPO = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPO / "reports/final_v8/PREREGISTRATION.md"
FROZEN_AIM2_EXTENSION = REPO / "aim2_loco_five_seed_extension.py"
FROZEN_CAMPAIGN_CONTROLLER = REPO / "aim1_mil_five_seed_expansion.py"
FROZEN_E2AD_TOPOLOGY = REPO / "aim2_sibling_loco.py"
EXPECTED_PREREGISTRATION_SHA256 = "8637bbfa13509b67cce98b3c7e902842a8e2ba8cdd049a5f682ac36756ae463a"
EXPECTED_FROZEN_AIM2_EXTENSION_SHA256 = (
    "35359fb0730614ab0c291d2f50af7e20af101d98804f49b1fdf98d119a50f243"
)
EXPECTED_FROZEN_CAMPAIGN_CONTROLLER_SHA256 = (
    "6251f7e2224ce0dc0f687784b310cfb0165d1fbe2aefdfedbd71640f704c8d41"
)
EXPECTED_FROZEN_E2AD_TOPOLOGY_SHA256 = (
    "9da3a45b2e484f28dbbbf1b11b779bde88b0458d3b023cd9e836887e39a6abfb"
)
EXPECTED_RAW_CPHT_SHA256 = "814329ed7afb78aa6888fdd3d53e6a97d863c4233590460364938dbb00b08ae3"

DEFAULT_CAMPAIGN_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/final_v9_mil_5seed_expansion_v1_20260823"
)
DEFAULT_ADJUDICATION_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/final_v9_adjudication")
COMPONENT = "aim2_e2a_five_seed"
IMPLEMENTATION = Path(__file__).resolve()
FOCUSED_TEST = REPO / "tests/test_aim2_e2a_five_seed_adjudication.py"

FAMILY_ARMS: tuple[str, ...] = (
    "family_cptac",
    "family_rih",
    "family_surgen",
    "family_tcga",
)
SIBLING_ARMS: tuple[str, ...] = (
    "sibling_sr386",
    "sibling_sr1482",
    "sibling_tcga_coad",
    "sibling_tcga_read",
)
ALL_PRIMARY_ARMS: tuple[str, ...] = (
    *FAMILY_ARMS,
    "family_rih_sm",
    *SIBLING_ARMS,
)
MET_TARGETS: tuple[str, ...] = ("rih_m", "sr1482_m")
EXPECTED_PRIMARY_CENSUS: Mapping[str, tuple[int, int]] = {
    "family_cptac": (94, 33),
    "family_rih": (153, 70),
    "family_surgen": (737, 294),
    "family_tcga": (502, 207),
    "family_rih_sm": (153, 70),
    "sibling_sr386": (413, 147),
    "sibling_sr1482": (324, 147),
    "sibling_tcga_coad": (374, 160),
    "sibling_tcga_read": (128, 47),
}
EXPECTED_CONFIRMATORY_MET_CENSUS: Mapping[tuple[str, str], tuple[int, int]] = {
    ("family_rih", "rih_m"): (85, 37),
    ("family_surgen", "sr1482_m"): (74, 30),
}

SIBLING_PAIRINGS: Mapping[str, tuple[str, str, str]] = {
    "sibling_sr386": ("family_surgen", "SR386", "sr386"),
    "sibling_sr1482": ("family_surgen", "SR1482", "sr1482"),
    "sibling_tcga_coad": ("family_tcga", "TCGA-COAD", "tcga_coad"),
    "sibling_tcga_read": ("family_tcga", "TCGA-READ", "tcga_read"),
}

# The source report stores the first two fields below with scientifically
# incorrect point-estimate gates, and its generic paired bootstrap is not the
# frozen FINAL-v8 target-by-KRAS design.  These exact JSON pointers make the
# precedence boundary machine-readable rather than relying on prose.
SUPERSEDED_SIBLING_PRIMARY_CI_POINTERS: tuple[str, ...] = tuple(
    f"/primary/{arm}/auroc_ci95" for arm in SIBLING_ARMS
)
SUPERSEDED_SOURCE_POINTERS: tuple[str, ...] = (
    "/family_loco_standardized_macro/directional_gate",
    "/family_loco_standardized_macro/claim_family_loco_transport",
    "/sibling_loco_directional_gate",
    *SUPERSEDED_SIBLING_PRIMARY_CI_POINTERS,
)
PAIRED_SOURCE_KEYS: tuple[str, ...] = (
    "sibling_sr386_minus_family_surgen_SR386",
    "sibling_sr1482_minus_family_surgen_SR1482",
    "sibling_tcga_coad_minus_family_tcga_TCGA-COAD",
    "sibling_tcga_read_minus_family_tcga_TCGA-READ",
    "family_rih_sm_minus_family_rih",
)
NONAUTHORITATIVE_PAIRED_CI_POINTERS: tuple[str, ...] = tuple(
    f"/paired_sibling_and_size_matched_contrasts/{key}/{field}"
    for key in PAIRED_SOURCE_KEYS
    for field in ("ci_low", "ci_high")
)
RETAINED_PAIRED_POINT_POINTERS: tuple[str, ...] = tuple(
    f"/paired_sibling_and_size_matched_contrasts/{key}/delta_auroc" for key in PAIRED_SOURCE_KEYS
)


class AdjudicationError(RuntimeError):
    """Fail-closed violation of the governed adjudication contract."""


@dataclass(frozen=True)
class SourcePaths:
    campaign_root: Path
    campaign_contract: Path
    campaign_final_receipt: Path
    aim2_contract: Path
    aim2_inference_seal: Path
    aim2_results: Path
    aim2_report_receipt: Path
    primary_patients: Path
    metastatic_patients: Path


@dataclass(frozen=True)
class SourceBundle:
    paths: SourcePaths
    identities: dict[str, dict[str, Any]]
    campaign_contract: dict[str, Any]
    campaign_final_receipt: dict[str, Any]
    aim2_contract: dict[str, Any]
    aim2_results: dict[str, Any]
    aim2_report_receipt: dict[str, Any]
    primary_patients: pd.DataFrame
    metastatic_patients: pd.DataFrame
    raw_cpht: dict[str, Any]


@dataclass(frozen=True)
class AdjudicationProduct:
    result: dict[str, Any]
    bootstraps: dict[str, np.ndarray]


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdjudicationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_strict_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise AdjudicationError(f"Cannot read governed JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdjudicationError(f"Expected a JSON object: {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AdjudicationError(f"Value is not strict JSON: {exc}") from exc


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(chunk_size):
                digest.update(block)
    except OSError as exc:
        raise AdjudicationError(f"Cannot hash governed artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    raw = Path(path)
    if raw.is_symlink():
        raise AdjudicationError(f"Governed file may not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise AdjudicationError(f"Missing governed artifact: {raw}") from exc
    if not resolved.is_file():
        raise AdjudicationError(f"Expected a regular file artifact: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _validate_identity(record: Any, path: Path, *, context: str) -> dict[str, Any]:
    observed = _artifact(path)
    if record != observed:
        raise AdjudicationError(
            f"{context} identity mismatch: expected {observed}, recorded {record}"
        )
    return observed


def _stable_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _artifact(path)
    payload = _read_json(path)
    after = _artifact(path)
    if before != after:
        raise AdjudicationError(f"Governed JSON changed while being read: {path}")
    return payload, before


def _stable_parquet(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = _artifact(path)
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - engine details vary
        raise AdjudicationError(f"Cannot read governed Parquet {path}: {exc}") from exc
    after = _artifact(path)
    if before != after:
        raise AdjudicationError(f"Governed Parquet changed while being read: {path}")
    return frame, before


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_campaign_root(path: Path) -> Path:
    raw = Path(path)
    resolved = raw.resolve(strict=False)
    production = DEFAULT_CAMPAIGN_ROOT.resolve(strict=False)
    if resolved != production and not _is_under(resolved, Path("/tmp").resolve()):
        raise AdjudicationError(f"Campaign root must be exactly {production}; tests may use /tmp")
    if raw.is_symlink() or (resolved.exists() and resolved.is_symlink()):
        raise AdjudicationError(f"Campaign root may not be a symlink: {raw}")
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def _validate_adjudication_root(path: Path, campaign_root: Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise AdjudicationError("Adjudication root must be absolute")
    resolved = raw.resolve(strict=False)
    production = DEFAULT_ADJUDICATION_ROOT.resolve(strict=False)
    if resolved != production and not _is_under(resolved, Path("/tmp").resolve()):
        raise AdjudicationError(
            f"Adjudication root must be exactly {production}; tests may use /tmp"
        )
    if raw.is_symlink() or (resolved.exists() and resolved.is_symlink()):
        raise AdjudicationError(f"Adjudication root may not be a symlink: {raw}")
    if _is_under(resolved, campaign_root) or _is_under(campaign_root, resolved):
        raise AdjudicationError(
            "Adjudication and source campaign roots must be separate and non-nested"
        )
    return resolved


def _source_paths(root: Path) -> SourcePaths:
    analysis = root / "aim2_loco/analysis"
    return SourcePaths(
        campaign_root=root,
        campaign_contract=root / "campaign/experiment_contract.json",
        campaign_final_receipt=root / "campaign/receipts/five_seed_results_complete.json",
        aim2_contract=root / "aim2_loco/contract.json",
        aim2_inference_seal=root / "aim2_loco/inference/inference_seal.json",
        aim2_results=analysis / "results_five_seed.json",
        aim2_report_receipt=analysis / "results_five_seed.receipt.json",
        primary_patients=analysis / "primary_patient_scores_five_seed.parquet",
        metastatic_patients=analysis / "e2met_patient_scores_five_seed.parquet",
    )


def component_root(adjudication_root: Path) -> Path:
    return Path(adjudication_root).resolve(strict=False) / COMPONENT


def _output_paths(adjudication_root: Path) -> dict[str, Path]:
    root = component_root(adjudication_root)
    return {
        "root": root,
        "result": root / "result.json",
        "bootstrap": root / "bootstrap_distributions.npz",
        "receipt": root / "receipt.json",
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdjudicationError(message)


def _validated_binary_labels(values: pd.Series, *, context: str) -> pd.Series:
    """Validate numeric binary values before any integer conversion."""

    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise AdjudicationError(f"{context} labels are not numeric binary values") from exc
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or not numeric.isin([0, 1]).all():
        raise AdjudicationError(
            f"{context} labels must be exactly 0 or 1 before integer conversion"
        )
    return numeric.astype(int)


def _validate_primary_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "arm", "label", "mean_logit", "subcohort"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AdjudicationError(f"Primary patient table lacks columns: {missing}")
    table = frame.copy()
    table["patient_id"] = table["patient_id"].astype(str)
    table["arm"] = table["arm"].astype(str)
    table["label"] = _validated_binary_labels(table["label"], context="Primary")
    table["mean_logit"] = pd.to_numeric(table["mean_logit"], errors="raise")
    if "target" in table and set(table["target"].astype(str)) != {"primary"}:
        raise AdjudicationError("Primary patient table contains a non-primary target")
    if set(table["arm"]) != set(ALL_PRIMARY_ARMS):
        raise AdjudicationError(f"Primary arm roster changed: {sorted(set(table['arm']))}")
    if table.duplicated(["arm", "patient_id"]).any():
        raise AdjudicationError("Primary table has duplicate arm/patient rows")
    if set(table["label"]) != {0, 1} or not np.isfinite(table["mean_logit"]).all():
        raise AdjudicationError("Primary labels/logits are not finite binary analysis data")
    for arm in ALL_PRIMARY_ARMS:
        block = table.loc[table["arm"].eq(arm)]
        labels = set(block["label"])
        if labels != {0, 1}:
            raise AdjudicationError(f"{arm}: target lacks both KRAS classes")
        census = (int(len(block)), int(block["label"].sum()))
        if census != EXPECTED_PRIMARY_CENSUS[arm]:
            raise AdjudicationError(f"{arm}: primary patient census changed: {census}")
    return table


def _validate_metastatic_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "arm", "target", "label", "mean_logit"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AdjudicationError(f"Metastatic patient table lacks columns: {missing}")
    table = frame.copy()
    table["patient_id"] = table["patient_id"].astype(str)
    table["arm"] = table["arm"].astype(str)
    table["target"] = table["target"].astype(str)
    table["label"] = _validated_binary_labels(table["label"], context="Metastatic")
    table["mean_logit"] = pd.to_numeric(table["mean_logit"], errors="raise")
    if table.duplicated(["arm", "target", "patient_id"]).any():
        raise AdjudicationError("Metastatic table has duplicate arm/target/patient rows")
    if not np.isfinite(table["mean_logit"]).all():
        raise AdjudicationError("Metastatic logits are non-finite")
    for target, arm in (("rih_m", "family_rih"), ("sr1482_m", "family_surgen")):
        block = table.loc[table["target"].eq(target) & table["arm"].eq(arm)]
        if block.empty or set(block["label"]) != {0, 1}:
            raise AdjudicationError(f"Missing binary confirmatory E2-MET cell {arm}/{target}")
        census = (int(len(block)), int(block["label"].sum()))
        if census != EXPECTED_CONFIRMATORY_MET_CENSUS[(arm, target)]:
            raise AdjudicationError(f"{arm}/{target}: confirmatory E2-MET census changed: {census}")
    return table


def _finite_json_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(np.isfinite(float(value)))
    )


def _validate_source_paired_points(results: dict[str, Any]) -> None:
    paired = results.get("paired_sibling_and_size_matched_contrasts")
    _require(isinstance(paired, dict), "Aim-2 report lacks paired sibling/size-matched contrasts")
    _require(
        set(paired) == set(PAIRED_SOURCE_KEYS),
        f"Aim-2 paired contrast key roster changed: {sorted(paired)}",
    )
    numeric_fields = ("delta_auroc", "auroc_left", "auroc_right", "ci_low", "ci_high")
    for key in PAIRED_SOURCE_KEYS:
        block = paired[key]
        _require(isinstance(block, dict), f"Aim-2 paired contrast is not an object: {key}")
        for field in numeric_fields:
            _require(
                field in block and _finite_json_number(block[field]),
                f"Aim-2 paired contrast lacks finite {field}: {key}",
            )
        n_patients = block.get("n_patients")
        _require(
            isinstance(n_patients, int) and not isinstance(n_patients, bool) and n_patients > 0,
            f"Aim-2 paired contrast lacks a positive integer n_patients: {key}",
        )
        _require(
            float(block["ci_low"]) <= float(block["ci_high"]),
            f"Aim-2 paired contrast interval is reversed: {key}",
        )
        _require(
            _numeric_close(
                block["delta_auroc"],
                float(block["auroc_right"]) - float(block["auroc_left"]),
            ),
            f"Aim-2 paired contrast delta is not right-minus-left: {key}",
        )


def _validate_source_results(results: dict[str, Any]) -> None:
    _require(results.get("schema_version") == 1, "Aim-2 report schema changed")
    _require(
        results.get("experiment") == "Aim 2 complete LOCO MIL five-seed results",
        "Aim-2 report experiment changed",
    )
    _require(results.get("seeds") == list(ALL_SEEDS), "Aim-2 report is not five-seed")
    _require(
        results.get("bootstrap") == {"unit": "patient", "n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
        "Aim-2 report bootstrap declaration changed",
    )
    _require(
        isinstance(results.get("primary"), dict)
        and set(results["primary"]) == set(ALL_PRIMARY_ARMS),
        "Aim-2 primary result arm roster changed",
    )
    for arm in SIBLING_ARMS:
        block = results["primary"].get(arm)
        interval = block.get("auroc_ci95") if isinstance(block, dict) else None
        _require(
            isinstance(interval, list)
            and len(interval) == 2
            and all(_finite_json_number(value) for value in interval),
            f"Aim-2 report lacks the superseded sibling interval: /primary/{arm}/auroc_ci95",
        )
    for pointer in ("family_loco_standardized_macro", "sibling_loco_directional_gate"):
        _require(pointer in results, f"Aim-2 report lacks /{pointer}")
    _validate_source_paired_points(results)
    _require(
        isinstance(results.get("e2met_confirmatory"), dict),
        "Aim-2 report lacks the confirmatory E2-MET gate",
    )
    _require(
        isinstance(results.get("e2met_confirmatory_family_naive"), dict),
        "Aim-2 report lacks confirmatory E2-MET target metrics",
    )


def _load_sources(campaign_root: Path, preregistration: Path) -> SourceBundle:
    root = _validate_campaign_root(campaign_root)
    paths = _source_paths(root)

    prereg_identity = _artifact(preregistration)
    _require(
        prereg_identity["sha256"] == EXPECTED_PREREGISTRATION_SHA256,
        "FINAL-v8 preregistration bytes changed",
    )
    frozen_extension_identity = _artifact(FROZEN_AIM2_EXTENSION)
    _require(
        frozen_extension_identity["sha256"] == EXPECTED_FROZEN_AIM2_EXTENSION_SHA256,
        "Frozen five-seed Aim-2 extension bytes changed",
    )
    frozen_campaign_identity = _artifact(FROZEN_CAMPAIGN_CONTROLLER)
    _require(
        frozen_campaign_identity["sha256"] == EXPECTED_FROZEN_CAMPAIGN_CONTROLLER_SHA256,
        "Frozen five-seed campaign controller bytes changed",
    )
    frozen_e2ad_identity = _artifact(FROZEN_E2AD_TOPOLOGY)
    _require(
        frozen_e2ad_identity["sha256"] == EXPECTED_FROZEN_E2AD_TOPOLOGY_SHA256,
        "Frozen FINAL-v8 E2a-D topology implementation bytes changed",
    )

    campaign_contract, campaign_contract_identity = _stable_json(paths.campaign_contract)
    campaign_final, campaign_final_identity = _stable_json(paths.campaign_final_receipt)
    aim2_contract, aim2_contract_identity = _stable_json(paths.aim2_contract)
    inference_seal, inference_seal_identity = _stable_json(paths.aim2_inference_seal)
    results, results_identity = _stable_json(paths.aim2_results)
    report_receipt, report_receipt_identity = _stable_json(paths.aim2_report_receipt)
    primary, primary_identity = _stable_parquet(paths.primary_patients)
    metastatic, metastatic_identity = _stable_parquet(paths.metastatic_patients)

    _require(
        campaign_contract.get("status") == "sealed_before_new_fit"
        and campaign_contract.get("experiment") == "final-v9 study-wide MIL five-seed expansion",
        "Campaign contract status/experiment changed",
    )
    _require(
        campaign_contract.get("model_seeds", {}).get("complete") == list(ALL_SEEDS),
        "Campaign contract is not the five-seed expansion",
    )
    _require(
        campaign_contract.get("execution", {}).get("max_concurrent_gpu_trainers") == 6,
        "Campaign contract no longer binds six-way training",
    )
    _require(
        campaign_contract.get("controllers", {}).get("campaign") == frozen_campaign_identity,
        "Campaign contract/controller identity disagrees",
    )
    _require(
        campaign_contract.get("controllers", {}).get("aim2_loco") == frozen_extension_identity,
        "Campaign contract/Aim-2 controller identity disagrees",
    )
    _validate_identity(
        campaign_contract.get("component_contracts", {}).get("aim2_loco"),
        paths.aim2_contract,
        context="Campaign Aim-2 component contract",
    )

    _require(
        campaign_final.get("status") == "five_seed_results_complete",
        "Final campaign receipt is absent or incomplete",
    )
    _require(campaign_final.get("seeds") == list(ALL_SEEDS), "Final campaign seed roster changed")
    _require(
        campaign_final.get("model_seeds_are_not_inference_units") is True,
        "Final campaign receipt treats model seeds as inference units",
    )
    _validate_identity(
        campaign_final.get("contract"),
        paths.campaign_contract,
        context="Final campaign contract",
    )
    aim2_component = campaign_final.get("components", {}).get("aim2_loco")
    _require(isinstance(aim2_component, dict), "Final campaign receipt lacks Aim-2")
    for key, path in (
        ("inference_seal", paths.aim2_inference_seal),
        ("analysis_results", paths.aim2_results),
        ("analysis_receipt", paths.aim2_report_receipt),
        ("analysis_primary_patients", paths.primary_patients),
        ("analysis_met_patients", paths.metastatic_patients),
    ):
        _validate_identity(aim2_component.get(key), path, context=f"Campaign {key}")

    _require(
        report_receipt.get("status") == "sealed_five_seed_results",
        "Aim-2 report receipt is not sealed",
    )
    _require(
        inference_seal.get("status") == "sealed_before_outcome_join"
        and inference_seal.get("five_seed_loco_complete") is True,
        "Aim-2 inference seal does not establish complete pre-outcome scoring",
    )
    _validate_identity(
        report_receipt.get("contract"), paths.aim2_contract, context="Aim-2 report contract"
    )
    _validate_identity(
        report_receipt.get("inference_seal"),
        paths.aim2_inference_seal,
        context="Aim-2 report inference seal",
    )
    artifacts = report_receipt.get("artifacts")
    _require(isinstance(artifacts, dict), "Aim-2 report receipt lacks artifacts")
    for key, path in (
        ("results", paths.aim2_results),
        ("primary_patients", paths.primary_patients),
        ("met_patients", paths.metastatic_patients),
    ):
        _validate_identity(artifacts.get(key), path, context=f"Aim-2 report {key}")

    _validate_source_results(results)
    mixed = results.get("mixed_seed_scope", {})
    _require(
        mixed.get("three_seed_unchanged")
        == ["raw all-conventional CPHT", "CPHT-A residual adaptation"],
        "Aim-2 mixed-seed scope changed",
    )
    raw_cpht = mixed.get("references", {}).get("all_conventional_cpht_three_seed")
    _require(isinstance(raw_cpht, dict), "Aim-2 report lacks raw CPHT identity")
    _require(
        raw_cpht.get("sha256") == EXPECTED_RAW_CPHT_SHA256,
        "Raw CPHT result identity changed from the sealed three-seed result",
    )
    raw_cpht_path = Path(str(raw_cpht.get("path", "")))
    _validate_identity(raw_cpht, raw_cpht_path, context="Raw three-seed CPHT result")
    _require(
        aim2_contract.get("mixed_seed_scope_references", {}).get("all_conventional_cpht_three_seed")
        == raw_cpht,
        "Aim-2 contract/report disagree on raw CPHT identity",
    )

    primary = _validate_primary_table(primary)
    metastatic = _validate_metastatic_table(metastatic)
    identities = {
        "final_v8_preregistration": prereg_identity,
        "frozen_aim2_five_seed_extension": frozen_extension_identity,
        "frozen_five_seed_campaign_controller": frozen_campaign_identity,
        "frozen_final_v8_e2ad_topology": frozen_e2ad_identity,
        "campaign_contract": campaign_contract_identity,
        "campaign_final_receipt": campaign_final_identity,
        "aim2_contract": aim2_contract_identity,
        "aim2_inference_seal": inference_seal_identity,
        "aim2_results": results_identity,
        "aim2_report_receipt": report_receipt_identity,
        "aim2_primary_patient_table": primary_identity,
        "aim2_metastatic_patient_table": metastatic_identity,
        "raw_cpht_three_seed_result": raw_cpht,
    }
    return SourceBundle(
        paths=paths,
        identities=identities,
        campaign_contract=campaign_contract,
        campaign_final_receipt=campaign_final,
        aim2_contract=aim2_contract,
        aim2_results=results,
        aim2_report_receipt=report_receipt,
        primary_patients=primary,
        metastatic_patients=metastatic,
        raw_cpht=raw_cpht,
    )


def _revalidate_source_identities(source: SourceBundle) -> None:
    """Close the multi-file read/recompute window with a second full rehash."""

    for name, recorded in source.identities.items():
        path = Path(str(recorded.get("path", "")))
        observed = _artifact(path)
        if observed != recorded:
            raise AdjudicationError(f"Source identity changed during adjudication replay: {name}")


def stratified_bootstrap_indices(
    labels: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw fixed-class-count patient indices in FINAL-v8 class order 0, 1."""

    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("KRAS-stratified bootstrap requires both binary classes")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    draws = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        draws.append(rng.choice(candidates, size=(n_bootstrap, len(candidates)), replace=True))
    return np.concatenate(draws, axis=1)


def final_v8_family_bootstrap_indices(
    labels: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    """Reproduce E2a-F's historical replicate-major RNG call ordering.

    FINAL-v8's family replay drew class 0 and class 1 inside each replicate,
    whereas the paired E2a-D/E2-MET implementations draw one complete class
    matrix at a time. Both are fixed-class bootstraps; preserving this ordering
    reproduces the sealed E2a-F directional and nested-macro intervals.
    """

    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("KRAS-stratified bootstrap requires both binary classes")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    indices = np.empty((n_bootstrap, len(labels)), dtype=np.int64)
    for draw in range(n_bootstrap):
        indices[draw, : len(negative)] = rng.choice(negative, size=len(negative), replace=True)
        indices[draw, len(negative) :] = rng.choice(positive, size=len(positive), replace=True)
    return indices


def bootstrap_auroc_samples(
    labels: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
    *,
    chunk_size: int = 64,
) -> np.ndarray:
    """Tie-correct AUROC on fixed-class draws without a multi-gigabyte tensor."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    indices = np.asarray(indices)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("AUROC bootstrap labels/scores are incompatible")
    if indices.ndim != 2 or indices.shape[1] != len(labels):
        raise ValueError("AUROC bootstrap indices have the wrong shape")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("AUROC bootstrap indices must be integers")
    if indices.size and (indices.min() < 0 or indices.max() >= len(labels)):
        raise ValueError("AUROC bootstrap index is out of bounds")
    n_negative = int((labels == 0).sum())
    n_positive = int((labels == 1).sum())
    if n_negative == 0 or n_positive == 0:
        raise ValueError("AUROC requires both classes")
    output = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        sampled = scores[indices[start:stop]]
        negative = sampled[:, :n_negative]
        positive = sampled[:, n_negative:]
        difference = positive[:, :, None] - negative[:, None, :]
        output[start:stop] = (
            (difference > 0).sum(axis=(1, 2)) + 0.5 * (difference == 0).sum(axis=(1, 2))
        ) / (n_negative * n_positive)
    return output


def _interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise AdjudicationError("Bootstrap distribution is empty or non-finite")
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def _rank_point(frame: pd.DataFrame) -> float:
    return float(roc_auc_score(frame["label"], frame["mean_logit"]))


def _direction_block(frame: pd.DataFrame, draws: np.ndarray) -> dict[str, Any]:
    interval = _interval(draws)
    return {
        "n": int(len(frame)),
        "n_mutant": int(frame["label"].sum()),
        "auroc": _rank_point(frame),
        "auroc_ci95": interval,
        "directional_gate": {
            "rule": "patient-bootstrap AUROC lower 95% bound > 0.50",
            "lower_ci_above_0p5": bool(interval[0] > 0.50),
            "passes": bool(interval[0] > 0.50),
        },
    }


def _arm(table: pd.DataFrame, arm: str) -> pd.DataFrame:
    block = table.loc[table["arm"].eq(arm)].copy()
    if block.empty:
        raise AdjudicationError(f"Missing primary arm: {arm}")
    return block.sort_values("patient_id").reset_index(drop=True)


def _subcohort(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    block = frame.loc[frame["subcohort"].astype(str).eq(value)].copy()
    if block.empty:
        raise AdjudicationError(f"Missing target subcohort: {value}")
    return block.sort_values("patient_id").reset_index(drop=True)


def _align_pair(
    left: pd.DataFrame, right: pd.DataFrame, *, context: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = left.sort_values("patient_id").reset_index(drop=True)
    right = right.sort_values("patient_id").reset_index(drop=True)
    if left["patient_id"].tolist() != right["patient_id"].tolist():
        raise AdjudicationError(f"{context}: paired target patient rosters differ")
    if not np.array_equal(left["label"].to_numpy(), right["label"].to_numpy()):
        raise AdjudicationError(f"{context}: paired target KRAS labels differ")
    return left, right


def _compute_e2a_f(
    table: pd.DataFrame, *, n_bootstrap: int, bootstrap_seed: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    family = {
        "CPTAC": _arm(table, "family_cptac"),
        "RIH": _arm(table, "family_rih"),
        "SR386_given_whole_SurGen_holdout": _subcohort(_arm(table, "family_surgen"), "SR386"),
        "SR1482_given_whole_SurGen_holdout": _subcohort(_arm(table, "family_surgen"), "SR1482"),
        "TCGA_pooled_COAD_READ": _arm(table, "family_tcga"),
    }
    # This order is the original FINAL-v8 E2a-F replay order and is therefore
    # part of the deterministic bootstrap contract.
    rng = np.random.default_rng(bootstrap_seed)
    samples: dict[str, np.ndarray] = {}
    for name, frame in family.items():
        labels = frame["label"].to_numpy(int)
        indices = final_v8_family_bootstrap_indices(labels, n_bootstrap=n_bootstrap, rng=rng)
        samples[name] = bootstrap_auroc_samples(
            labels, frame["mean_logit"].to_numpy(float), indices
        )
    macro_draws = np.mean(
        np.vstack(
            [
                samples["CPTAC"],
                samples["RIH"],
                np.mean(
                    np.vstack(
                        [
                            samples["SR386_given_whole_SurGen_holdout"],
                            samples["SR1482_given_whole_SurGen_holdout"],
                        ]
                    ),
                    axis=0,
                ),
                samples["TCGA_pooled_COAD_READ"],
            ]
        ),
        axis=0,
    )
    directions = {name: _direction_block(frame, samples[name]) for name, frame in family.items()}
    components = {
        "CPTAC": directions["CPTAC"]["auroc"],
        "RIH": directions["RIH"]["auroc"],
        "SurGen_mean_SR386_SR1482": float(
            np.mean(
                [
                    directions["SR386_given_whole_SurGen_holdout"]["auroc"],
                    directions["SR1482_given_whole_SurGen_holdout"]["auroc"],
                ]
            )
        ),
        "TCGA_pooled_COAD_READ": directions["TCGA_pooled_COAD_READ"]["auroc"],
    }
    passed = [name for name, block in directions.items() if block["directional_gate"]["passes"]]
    result = {
        "role": "primary four-source-family leave-one-family-out transport",
        "directions": directions,
        "nested_four_family_macro": {
            "formula": "(CPTAC + RIH + mean(SR386, SR1482) + pooled_TCGA) / 4",
            "components": components,
            "auroc": float(np.mean(list(components.values()))),
            "auroc_ci95": _interval(macro_draws),
            "role": "panel summary; cannot rescue a failed direction",
        },
        "adjudication": {
            "rule": (
                "all five target-specific patient-bootstrap AUROC lower 95% bounds "
                "must exceed 0.50; the macro cannot rescue a failed direction"
            ),
            "passed_directions": passed,
            "required_directions": list(directions),
            "all_directional_lower_bounds_above_0p5": bool(len(passed) == 5),
            "claim_family_loco_transport": bool(len(passed) == 5),
        },
    }
    arrays = {f"e2a_f__{name}": draws for name, draws in samples.items()}
    arrays["e2a_f__nested_four_family_macro"] = macro_draws
    return result, arrays


def _confirm_e2a_f_source_values(
    adjudicated: dict[str, Any], source_results: dict[str, Any]
) -> dict[str, Any]:
    """Confirm that only the source gate logic—not its estimates—needed repair."""

    source = source_results["family_loco_standardized_macro"]
    macro = adjudicated["nested_four_family_macro"]
    if not _numeric_close(macro["auroc"], source.get("macro_auroc")):
        raise AdjudicationError("E2a-F source macro point changed during adjudication")
    recorded_ci = source.get("macro_auroc_ci95")
    if not (
        isinstance(recorded_ci, list)
        and len(recorded_ci) == 2
        and all(
            _numeric_close(left, right)
            for left, right in zip(macro["auroc_ci95"], recorded_ci, strict=True)
        )
    ):
        raise AdjudicationError("E2a-F source macro interval changed during adjudication")
    recorded_directions = source.get("directional_results")
    if not isinstance(recorded_directions, dict):
        raise AdjudicationError("E2a-F source report lacks directional estimates")
    for name, block in adjudicated["directions"].items():
        recorded = recorded_directions.get(name)
        if not isinstance(recorded, dict) or not _numeric_close(
            block["auroc"], recorded.get("auroc")
        ):
            raise AdjudicationError(f"E2a-F source direction point changed: {name}")
        recorded_ci = recorded.get("auroc_ci95")
        if not (
            isinstance(recorded_ci, list)
            and len(recorded_ci) == 2
            and all(
                _numeric_close(left, right)
                for left, right in zip(block["auroc_ci95"], recorded_ci, strict=True)
            )
        ):
            raise AdjudicationError(f"E2a-F source direction interval changed: {name}")
    return {
        "source_report_macro_point_and_interval_unchanged": True,
        "source_report_direction_points_and_intervals_unchanged": True,
        "only_gate_interpretation_superseded": True,
    }


def _compute_e2a_d(
    table: pd.DataFrame, *, n_bootstrap: int, bootstrap_seed: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    siblings = {arm: _arm(table, arm) for arm in SIBLING_ARMS}
    family_pairs: dict[str, pd.DataFrame] = {}
    for sibling, (family_arm, subcohort, _slug) in SIBLING_PAIRINGS.items():
        siblings[sibling], family_pairs[sibling] = _align_pair(
            siblings[sibling],
            _subcohort(_arm(table, family_arm), subcohort),
            context=sibling,
        )

    # This is the original governed aim2_sibling_loco.compute_report topology:
    # the four paired sibling cells first, then pooled TCGA, RIH, and CPTAC.
    rng = np.random.default_rng(bootstrap_seed)
    sibling_samples: dict[str, np.ndarray] = {}
    family_pair_samples: dict[str, np.ndarray] = {}
    for sibling in SIBLING_ARMS:
        labels = siblings[sibling]["label"].to_numpy(int)
        indices = stratified_bootstrap_indices(labels, n_bootstrap=n_bootstrap, rng=rng)
        sibling_samples[sibling] = bootstrap_auroc_samples(
            labels, siblings[sibling]["mean_logit"].to_numpy(float), indices
        )
        family_pair_samples[sibling] = bootstrap_auroc_samples(
            labels, family_pairs[sibling]["mean_logit"].to_numpy(float), indices
        )

    standalone_frames = {
        "tcga": _arm(table, "family_tcga"),
        "rih": _arm(table, "family_rih"),
        "cptac": _arm(table, "family_cptac"),
    }
    standalone_samples: dict[str, np.ndarray] = {}
    rih_indices: np.ndarray | None = None
    for name, frame in standalone_frames.items():
        labels = frame["label"].to_numpy(int)
        indices = stratified_bootstrap_indices(labels, n_bootstrap=n_bootstrap, rng=rng)
        if name == "rih":
            rih_indices = indices
        standalone_samples[name] = bootstrap_auroc_samples(
            labels, frame["mean_logit"].to_numpy(float), indices
        )

    targets = {
        SIBLING_PAIRINGS[arm][2]: _direction_block(siblings[arm], sibling_samples[arm])
        for arm in SIBLING_ARMS
    }
    paired: dict[str, Any] = {}
    paired_draws: dict[str, np.ndarray] = {}
    for sibling in SIBLING_ARMS:
        slug = SIBLING_PAIRINGS[sibling][2]
        delta = sibling_samples[sibling] - family_pair_samples[sibling]
        paired_draws[slug] = delta
        paired[slug] = {
            "estimand": "AUROC(sibling retained) - AUROC(whole family held out)",
            "n_paired_patients": int(len(siblings[sibling])),
            "sibling_retained_auroc": _rank_point(siblings[sibling]),
            "whole_family_held_out_auroc": _rank_point(family_pairs[sibling]),
            "delta_auroc": float(
                _rank_point(siblings[sibling]) - _rank_point(family_pairs[sibling])
            ),
            "delta_auroc_ci95": _interval(delta),
            "bootstrap": (
                "target-by-KRAS-stratified patient bootstrap; identical indices "
                "shared by both scorers"
            ),
        }

    five_points = {
        "TCGA_whole_family_holdout": _rank_point(standalone_frames["tcga"]),
        "SR386_sibling_holdout": _rank_point(siblings["sibling_sr386"]),
        "SR1482_sibling_holdout": _rank_point(siblings["sibling_sr1482"]),
        "RIH_whole_family_holdout": _rank_point(standalone_frames["rih"]),
        "CPTAC_whole_family_holdout": _rank_point(standalone_frames["cptac"]),
    }
    five_draws = np.mean(
        np.vstack(
            [
                standalone_samples["tcga"],
                sibling_samples["sibling_sr386"],
                sibling_samples["sibling_sr1482"],
                standalone_samples["rih"],
                standalone_samples["cptac"],
            ]
        ),
        axis=0,
    )
    six_points = {
        "TCGA_COAD_sibling_holdout": _rank_point(siblings["sibling_tcga_coad"]),
        "TCGA_READ_sibling_holdout": _rank_point(siblings["sibling_tcga_read"]),
        "SR386_sibling_holdout": _rank_point(siblings["sibling_sr386"]),
        "SR1482_sibling_holdout": _rank_point(siblings["sibling_sr1482"]),
        "RIH_whole_family_holdout": _rank_point(standalone_frames["rih"]),
        "CPTAC_whole_family_holdout": _rank_point(standalone_frames["cptac"]),
    }
    six_draws = np.mean(
        np.vstack(
            [
                sibling_samples["sibling_tcga_coad"],
                sibling_samples["sibling_tcga_read"],
                sibling_samples["sibling_sr386"],
                sibling_samples["sibling_sr1482"],
                standalone_samples["rih"],
                standalone_samples["cptac"],
            ]
        ),
        axis=0,
    )

    baseline_rih, size_matched_rih = _align_pair(
        standalone_frames["rih"],
        _arm(table, "family_rih_sm"),
        context="family_rih_size_matched",
    )
    if rih_indices is None:  # pragma: no cover - fixed standalone roster
        raise AdjudicationError("RIH bootstrap indices were not constructed")
    rih_size_samples = bootstrap_auroc_samples(
        size_matched_rih["label"].to_numpy(int),
        size_matched_rih["mean_logit"].to_numpy(float),
        rih_indices,
    )
    rih_baseline_samples = bootstrap_auroc_samples(
        baseline_rih["label"].to_numpy(int),
        baseline_rih["mean_logit"].to_numpy(float),
        rih_indices,
    )
    rih_size_delta = rih_size_samples - rih_baseline_samples

    passed = [name for name, block in targets.items() if block["directional_gate"]["passes"]]
    result = {
        "role": "sibling-stratum/project leave-one-domain-out transport",
        "targets": targets,
        "paired_sibling_minus_family": paired,
        "macros": {
            "secondary_five_acquisition_domain": {
                "primitives": five_points,
                "auroc": float(np.mean(list(five_points.values()))),
                "auroc_ci95": _interval(five_draws),
                "role": "secondary panel summary; cannot rescue a failed direction",
            },
            "descriptive_equal_six_stratum": {
                "primitives": six_points,
                "auroc": float(np.mean(list(six_points.values()))),
                "auroc_ci95": _interval(six_draws),
                "role": "descriptive; gives TCGA and SurGen two votes each",
            },
        },
        "adjudication": {
            "rule": (
                "all four sibling-stratum patient-bootstrap AUROC lower 95% bounds "
                "must exceed 0.50; neither macro can rescue a failed direction"
            ),
            "passed_directions": passed,
            "required_directions": [SIBLING_PAIRINGS[arm][2] for arm in SIBLING_ARMS],
            "all_directional_lower_bounds_above_0p5": bool(len(passed) == 4),
            "claim_sibling_stratum_transport": bool(len(passed) == 4),
        },
        "size_matched_rih_sensitivity": {
            "estimand": "AUROC(size-matched RIH holdout) - AUROC(full-source RIH holdout)",
            "n_paired_patients": int(len(baseline_rih)),
            "size_matched_auroc": _rank_point(size_matched_rih),
            "full_source_auroc": _rank_point(baseline_rih),
            "delta_auroc": float(_rank_point(size_matched_rih) - _rank_point(baseline_rih)),
            "delta_auroc_ci95": _interval(rih_size_delta),
            "bootstrap": (
                "RIH-by-KRAS-stratified patient bootstrap; identical indices shared by both scorers"
            ),
            "role": "secondary sensitivity; not an E2a-D directional gate",
        },
    }
    arrays: dict[str, np.ndarray] = {}
    for sibling in SIBLING_ARMS:
        slug = SIBLING_PAIRINGS[sibling][2]
        arrays[f"e2a_d__{slug}__sibling"] = sibling_samples[sibling]
        arrays[f"e2a_d__{slug}__family"] = family_pair_samples[sibling]
        arrays[f"e2a_d__{slug}__delta"] = paired_draws[slug]
    for name, draws in standalone_samples.items():
        arrays[f"e2a_d__{name}__whole_family"] = draws
    arrays["e2a_d__secondary_five_acquisition_domain"] = five_draws
    arrays["e2a_d__descriptive_equal_six_stratum"] = six_draws
    arrays["e2a_d__rih_size_matched"] = rih_size_samples
    arrays["e2a_d__rih_size_matched_minus_full"] = rih_size_delta
    return result, arrays


def _numeric_close(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol))
    except (TypeError, ValueError):
        return False


def _confirm_retained_source_paired_points(
    e2a_d: dict[str, Any], source_results: dict[str, Any]
) -> dict[str, Any]:
    source = source_results["paired_sibling_and_size_matched_contrasts"]
    expected: dict[str, dict[str, Any]] = {}
    for key, slug in zip(
        PAIRED_SOURCE_KEYS[:4], ("sr386", "sr1482", "tcga_coad", "tcga_read"), strict=True
    ):
        block = e2a_d["paired_sibling_minus_family"][slug]
        expected[key] = {
            "delta_auroc": block["delta_auroc"],
            "auroc_left": block["whole_family_held_out_auroc"],
            "auroc_right": block["sibling_retained_auroc"],
            "n_patients": block["n_paired_patients"],
            "adjudication_json_pointer": f"/e2a_d/paired_sibling_minus_family/{slug}",
        }
    size_matched = e2a_d["size_matched_rih_sensitivity"]
    expected[PAIRED_SOURCE_KEYS[4]] = {
        "delta_auroc": size_matched["delta_auroc"],
        "auroc_left": size_matched["full_source_auroc"],
        "auroc_right": size_matched["size_matched_auroc"],
        "n_patients": size_matched["n_paired_patients"],
        "adjudication_json_pointer": "/e2a_d/size_matched_rih_sensitivity",
    }

    confirmations: dict[str, Any] = {}
    for key in PAIRED_SOURCE_KEYS:
        recorded = source[key]
        replayed = expected[key]
        for field in ("delta_auroc", "auroc_left", "auroc_right"):
            if not _numeric_close(recorded[field], replayed[field]):
                raise AdjudicationError(f"Retained source paired point fails replay: {key}/{field}")
        if recorded["n_patients"] != replayed["n_patients"]:
            raise AdjudicationError(f"Retained source paired census fails replay: {key}/n_patients")
        confirmations[key] = {
            "source_json_pointer": (
                f"/paired_sibling_and_size_matched_contrasts/{key}/delta_auroc"
            ),
            "adjudication_json_pointer": replayed["adjudication_json_pointer"],
            "delta_auroc": replayed["delta_auroc"],
            "auroc_left": replayed["auroc_left"],
            "auroc_right": replayed["auroc_right"],
            "n_patients": replayed["n_patients"],
            "matches": True,
        }
    return {
        "all_five_retained_delta_points_recomputed_and_match": True,
        "comparisons": confirmations,
    }


def _compute_e2met_confirmation(
    table: pd.DataFrame,
    source_results: dict[str, Any],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    mappings = {"rih_m": "family_rih", "sr1482_m": "family_surgen"}
    rng = np.random.default_rng(bootstrap_seed)
    targets: dict[str, Any] = {}
    samples: dict[str, np.ndarray] = {}
    for target, arm in mappings.items():
        frame = (
            table.loc[table["target"].eq(target) & table["arm"].eq(arm)]
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
        labels = frame["label"].to_numpy(int)
        indices = stratified_bootstrap_indices(labels, n_bootstrap=n_bootstrap, rng=rng)
        draws = bootstrap_auroc_samples(labels, frame["mean_logit"].to_numpy(float), indices)
        samples[target] = draws
        targets[target] = {
            "scorer": arm,
            "n": int(len(frame)),
            "n_mutant": int(labels.sum()),
            "auroc": _rank_point(frame),
            "auroc_ci95": _interval(draws),
        }
    macro_draws = np.mean(np.vstack([samples[target] for target in MET_TARGETS]), axis=0)
    macro_ci = _interval(macro_draws)
    both_points = bool(all(targets[target]["auroc"] > 0.5 for target in MET_TARGETS))
    lower = bool(macro_ci[0] > 0.5)
    recomputed = {
        "targets": targets,
        "equal_cohort_metastatic_macro_auroc": float(
            np.mean([targets[target]["auroc"] for target in MET_TARGETS])
        ),
        "macro_auroc_ci95": macro_ci,
        "both_target_points_above_0p5": both_points,
        "macro_lower_bound_above_0p5": lower,
        "claim_metastatic_transport": bool(both_points and lower),
        "gate": "both target AUROC points > 0.5 AND macro AUROC CI95 lower bound > 0.5",
    }
    source = source_results["e2met_confirmatory"]
    for key in (
        "equal_cohort_metastatic_macro_auroc",
        "macro_auroc_ci95",
        "both_target_points_above_0p5",
        "macro_lower_bound_above_0p5",
        "claim_metastatic_transport",
        "gate",
    ):
        observed = recomputed[key]
        recorded = source.get(key)
        if key == "macro_auroc_ci95":
            matches = (
                isinstance(recorded, list)
                and len(recorded) == 2
                and all(_numeric_close(x, y) for x, y in zip(observed, recorded, strict=True))
            )
        elif isinstance(observed, float):
            matches = _numeric_close(observed, recorded)
        else:
            matches = observed == recorded
        if not matches:
            raise AdjudicationError(f"Conformant E2-MET gate changed at /e2met_confirmatory/{key}")
    source_targets = source_results["e2met_confirmatory_family_naive"]
    for target in MET_TARGETS:
        recorded = source_targets.get(target)
        if not isinstance(recorded, dict):
            raise AdjudicationError(f"Conformant E2-MET target block is missing: {target}")
        if not _numeric_close(targets[target]["auroc"], recorded.get("auroc")):
            raise AdjudicationError(f"Conformant E2-MET target point changed: {target}")
        recorded_ci = recorded.get("auroc_ci95")
        if not (
            isinstance(recorded_ci, list)
            and len(recorded_ci) == 2
            and all(
                _numeric_close(left, right)
                for left, right in zip(targets[target]["auroc_ci95"], recorded_ci, strict=True)
            )
        ):
            raise AdjudicationError(f"Conformant E2-MET target interval changed: {target}")
    recomputed["source_report_pointer"] = "/e2met_confirmatory"
    recomputed["matches_and_remains_authoritative"] = True
    arrays = {f"e2met__{target}": draws for target, draws in samples.items()}
    arrays["e2met__equal_cohort_macro"] = macro_draws
    return recomputed, arrays


def _build_product(
    source: SourceBundle,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> AdjudicationProduct:
    if (n_bootstrap, bootstrap_seed) != (N_BOOTSTRAP, BOOTSTRAP_SEED):
        raise AdjudicationError(f"Frozen E2a inference is n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}")
    e2a_f, f_arrays = _compute_e2a_f(
        source.primary_patients,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    e2a_f["source_report_confirmation"] = _confirm_e2a_f_source_values(e2a_f, source.aim2_results)
    e2a_d, d_arrays = _compute_e2a_d(
        source.primary_patients,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    e2a_d["source_report_paired_point_confirmation"] = _confirm_retained_source_paired_points(
        e2a_d, source.aim2_results
    )
    e2met, met_arrays = _compute_e2met_confirmation(
        source.metastatic_patients,
        source.aim2_results,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    bootstraps = {**f_arrays, **d_arrays, **met_arrays}
    if len(bootstraps) != len(f_arrays) + len(d_arrays) + len(met_arrays):
        raise AdjudicationError("Bootstrap array names collided")
    if any(
        values.shape != (n_bootstrap,) or not np.isfinite(values).all()
        for values in bootstraps.values()
    ):
        raise AdjudicationError("A bootstrap output is incomplete or non-finite")

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "governed_five_seed_adjudication",
        "experiment": "Aim 2 E2a five-seed governed adjudication",
        "design_status": (
            "post-outcome robustness extension applying inherited FINAL-v8 gates; "
            "not retrospectively preregistered"
        ),
        "model_seeds": list(ALL_SEEDS),
        "ensemble_rule": "mean native logits across five model seeds",
        "inference": {
            "unit": "patient",
            "stratification": "target_x_KRAS",
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "paired_scorer_indices_shared": True,
            "macro_recomputed_each_draw": True,
            "folds_and_model_seeds_are_not_resampling_units": True,
            "independent_named_streams": {
                "e2a_f": "seed 20260817; FINAL-v8 family direction order",
                "e2a_d": "seed 20260817; original e2ad sibling-then-standalone order",
                "e2met_confirmation": "seed 20260817; RIH-M then SR1482-M",
            },
        },
        "precedence": {
            "source_report": source.identities["aim2_results"],
            "superseded_source_report_json_pointers": list(SUPERSEDED_SOURCE_POINTERS),
            "non_authoritative_source_paired_ci_json_pointers": list(
                NONAUTHORITATIVE_PAIRED_CI_POINTERS
            ),
            "retained_source_paired_delta_point_json_pointers": list(
                RETAINED_PAIRED_POINT_POINTERS
            ),
            "authoritative_replacements": {
                "E2a-F gates and claim": "/e2a_f/adjudication",
                "E2a-D gates and claim": "/e2a_d/adjudication",
                "sibling paired intervals": "/e2a_d/paired_sibling_minus_family",
                "RIH size-matched paired interval": "/e2a_d/size_matched_rih_sensitivity",
            },
            "superseded_sibling_primary_ci_replacements": {
                f"/primary/{arm}/auroc_ci95": (
                    f"/e2a_d/targets/{SIBLING_PAIRINGS[arm][2]}/auroc_ci95"
                )
                for arm in SIBLING_ARMS
            },
            "retained_source_paired_delta_replay": (
                "/e2a_d/source_report_paired_point_confirmation"
            ),
            "confirmed_unchanged_source_report_json_pointers": [
                "/family_loco_standardized_macro/directional_results",
                "/family_loco_standardized_macro/macro_auroc",
                "/family_loco_standardized_macro/macro_auroc_ci95",
                "/e2met_confirmatory",
            ],
            "supersession_is_field_scoped": True,
            "all_unlisted_source_report_fields_remain_authoritative": True,
            "macro_rescue_prohibited": True,
        },
        "e2a_f": e2a_f,
        "e2a_d": e2a_d,
        "e2met_gate_confirmation": e2met,
        "scope_boundary": {
            "raw_cpht": {
                "status": "unchanged sealed three-seed analysis",
                "model_seeds": [42, 43, 44],
                "result": source.raw_cpht,
                "five_seed_adjudication_applied": False,
            },
            "cpht_a": {
                "status": "unchanged three-seed analysis; outside this adjudication",
                "model_seeds": [42, 43, 44],
            },
            "orion_loco_sensitivity": {
                "status": "five-seed exploratory LOCO sensitivity only",
                "is_confirmatory_e2_cpht": False,
            },
            "five_seed_e2met_role_and_organ_analyses": {
                "status": "not included in this focused E2a gate adjudication",
                "required_to_confirm_existing_e2met_gate": False,
            },
        },
        "inputs": source.identities,
    }
    _revalidate_source_identities(source)
    return AdjudicationProduct(result=result, bootstraps=bootstraps)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer, np.ascontiguousarray(array), allow_pickle=False, version=(1, 0)
    )
    return buffer.getvalue()


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Create deterministic, uncompressed NPZ bytes with fixed ZIP metadata."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name or name.endswith(".npy"):
                raise AdjudicationError(f"Unsafe bootstrap array name: {name!r}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(np.asarray(arrays[name], dtype=np.float64)))
    return buffer.getvalue()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = list(archive.files)
            if len(names) != len(set(names)):
                raise AdjudicationError("Bootstrap archive has duplicate names")
            arrays = {name: np.asarray(archive[name]) for name in names}
    except Exception as exc:
        if isinstance(exc, AdjudicationError):
            raise
        raise AdjudicationError(f"Cannot read bootstrap archive {path}: {exc}") from exc
    return arrays


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any existing target."""

    renameat2 = getattr(_LIBC, "renameat2", None)
    if renameat2 is None:  # pragma: no cover - governed platform is Linux/WSL
        raise AdjudicationError("Atomic no-replace publication requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise AdjudicationError(f"Concurrent adjudication publication collision: {destination}")
        raise AdjudicationError(f"Atomic adjudication publication failed: {os.strerror(error)}")


def _receipt_payload(
    *,
    source: SourceBundle,
    result_identity: dict[str, Any],
    bootstrap_identity: dict[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_governed_five_seed_adjudication",
        "experiment": "Aim 2 E2a five-seed governed adjudication",
        "design_status": "post-outcome robustness extension; not preregistration",
        "source_inputs": source.identities,
        "outputs": {
            "result": result_identity,
            "bootstrap_distributions": bootstrap_identity,
        },
        "implementation": {
            "tool": _artifact(IMPLEMENTATION),
            "focused_test": _artifact(FOCUSED_TEST),
        },
        "bootstrap_inventory": {
            "names": sorted(arrays),
            "arrays": len(arrays),
            "draws_per_array": N_BOOTSTRAP,
            "dtype": "float64",
        },
        "verification_contract": {
            "deterministic_full_recomputation": True,
            "verify_is_read_only": True,
            "receipt_written_last": True,
            "source_campaign_never_written": True,
        },
    }


def _publish(adjudication_root: Path, source: SourceBundle, product: AdjudicationProduct) -> Path:
    outputs = _output_paths(adjudication_root)
    final = outputs["root"]
    if final.exists():
        raise AdjudicationError(f"Refusing to overwrite existing adjudication: {final}")
    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{final.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    staging.mkdir(mode=0o755)
    try:
        result = staging / outputs["result"].name
        bootstrap = staging / outputs["bootstrap"].name
        receipt = staging / outputs["receipt"].name
        _write_exclusive(result, _json_bytes(product.result))
        _write_exclusive(bootstrap, _npz_bytes(product.bootstraps))
        result_identity = _artifact(result)
        bootstrap_identity = _artifact(bootstrap)
        # Identities in a durable receipt describe final paths, not staging
        # paths; their content hashes and sizes are already fixed above.
        result_identity["path"] = str(outputs["result"])
        bootstrap_identity["path"] = str(outputs["bootstrap"])
        receipt_payload = _receipt_payload(
            source=source,
            result_identity=result_identity,
            bootstrap_identity=bootstrap_identity,
            arrays=product.bootstraps,
        )
        _write_exclusive(receipt, _json_bytes(receipt_payload))
        _fsync_directory(staging)
        _rename_noreplace(staging, final)
        _fsync_directory(parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final


def _assert_json_equal(observed: Any, expected: Any, *, context: str) -> None:
    if _json_bytes(observed) != _json_bytes(expected):
        raise AdjudicationError(f"{context} fails deterministic semantic replay")


def verify(
    *, campaign_root: Path, adjudication_root: Path, preregistration: Path
) -> dict[str, Any]:
    """Read-only verification by identity checks and full statistical replay."""

    campaign = _validate_campaign_root(campaign_root)
    adjudication = _validate_adjudication_root(adjudication_root, campaign)
    outputs = _output_paths(adjudication)
    if outputs["root"].is_symlink():
        raise AdjudicationError(f"Adjudication component may not be a symlink: {outputs['root']}")
    if not outputs["root"].is_dir():
        raise FileNotFoundError(outputs["root"])
    entries = list(outputs["root"].iterdir())
    actual_files = {path.name for path in entries}
    expected_files = {
        outputs["result"].name,
        outputs["bootstrap"].name,
        outputs["receipt"].name,
    }
    if actual_files != expected_files or any(
        not path.is_file() or path.is_symlink() for path in entries
    ):
        raise AdjudicationError(f"Adjudication inventory changed: {sorted(actual_files)}")

    receipt, _receipt_identity = _stable_json(outputs["receipt"])
    result, result_identity = _stable_json(outputs["result"])
    before_bootstrap = _artifact(outputs["bootstrap"])
    observed_arrays = _load_npz(outputs["bootstrap"])
    after_bootstrap = _artifact(outputs["bootstrap"])
    if before_bootstrap != after_bootstrap:
        raise AdjudicationError("Bootstrap archive changed while being read")

    source = _load_sources(campaign, preregistration)
    product = _build_product(source)
    _assert_json_equal(result, product.result, context="Adjudication result")
    if set(observed_arrays) != set(product.bootstraps):
        raise AdjudicationError("Bootstrap array inventory fails replay")
    for name, expected in product.bootstraps.items():
        observed = observed_arrays[name]
        if observed.dtype != np.float64 or not np.array_equal(observed, expected):
            raise AdjudicationError(f"Bootstrap array fails exact replay: {name}")

    expected_receipt = _receipt_payload(
        source=source,
        result_identity=result_identity,
        bootstrap_identity=before_bootstrap,
        arrays=product.bootstraps,
    )
    _assert_json_equal(receipt, expected_receipt, context="Adjudication receipt")
    return result


def cmd_plan(args: argparse.Namespace) -> None:
    campaign = Path(args.campaign_root).resolve(strict=False)
    adjudication = Path(args.adjudication_root).resolve(strict=False)
    print(
        json.dumps(
            {
                "source_campaign": str(campaign),
                "source_mode": "read_only",
                "adjudication_root": str(component_root(adjudication)),
                "writes_without_apply": False,
                "model_seeds": list(ALL_SEEDS),
                "bootstrap": {
                    "unit": "patient",
                    "stratification": "target_x_KRAS",
                    "n": N_BOOTSTRAP,
                    "seed": BOOTSTRAP_SEED,
                    "paired_indices_shared": True,
                },
                "design_status": (
                    "post-outcome robustness extension applying inherited FINAL-v8 gates; "
                    "not retrospectively preregistered"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


def cmd_adjudicate(args: argparse.Namespace) -> None:
    campaign = _validate_campaign_root(args.campaign_root)
    adjudication = _validate_adjudication_root(args.adjudication_root, campaign)
    outputs = _output_paths(adjudication)
    if outputs["root"].exists():
        result = verify(
            campaign_root=campaign,
            adjudication_root=adjudication,
            preregistration=args.preregistration,
        )
        print(
            "PASS: existing immutable Aim-2 adjudication verifies: "
            f"{outputs['receipt']} (E2a-F={result['e2a_f']['adjudication']['claim_family_loco_transport']}, "
            f"E2a-D={result['e2a_d']['adjudication']['claim_sibling_stratum_transport']})"
        )
        return
    source = _load_sources(campaign, args.preregistration)
    product = _build_product(source)
    if not args.apply:
        print(
            json.dumps(
                {
                    "status": "validated_and_recomputed_dry_run",
                    "writes": 0,
                    "would_publish": str(outputs["root"]),
                    "e2a_f_claim": product.result["e2a_f"]["adjudication"][
                        "claim_family_loco_transport"
                    ],
                    "e2a_d_claim": product.result["e2a_d"]["adjudication"][
                        "claim_sibling_stratum_transport"
                    ],
                    "e2met_gate_unchanged": product.result["e2met_gate_confirmation"][
                        "matches_and_remains_authoritative"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    _publish(adjudication, source, product)
    result = verify(
        campaign_root=campaign,
        adjudication_root=adjudication,
        preregistration=args.preregistration,
    )
    print(
        "PASS: sealed governed Aim-2 E2a five-seed adjudication: "
        f"{outputs['receipt']} (E2a-F={result['e2a_f']['adjudication']['claim_family_loco_transport']}, "
        f"E2a-D={result['e2a_d']['adjudication']['claim_sibling_stratum_transport']})"
    )


def cmd_verify(args: argparse.Namespace) -> None:
    result = verify(
        campaign_root=args.campaign_root,
        adjudication_root=args.adjudication_root,
        preregistration=args.preregistration,
    )
    print(
        "PASS: deterministic read-only Aim-2 adjudication verification; "
        f"E2a-F={result['e2a_f']['adjudication']['claim_family_loco_transport']}, "
        f"E2a-D={result['e2a_d']['adjudication']['claim_sibling_stratum_transport']}, "
        "E2-MET=unchanged"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    common.add_argument("--adjudication-root", type=Path, default=DEFAULT_ADJUDICATION_ROOT)
    common.add_argument("--preregistration", type=Path, default=PREREGISTRATION)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("plan", parents=[common])
    command.set_defaults(func=cmd_plan)
    command = commands.add_parser("adjudicate", parents=[common])
    command.add_argument("--apply", action="store_true")
    command.set_defaults(func=cmd_adjudicate)
    command = commands.add_parser("verify", parents=[common])
    command.set_defaults(func=cmd_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
