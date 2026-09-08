#!/usr/bin/env python3
"""Fail-closed verification of the corrected cap-8192 Aim-2 campaign.

The original Aim-2 lineage and the downstream E2c native-logit recovery are
deliberately separate immutable roots.  This verifier takes both roots
explicitly, validates every recorded file identity it encounters, and derives
the semantic claim states from confidence intervals instead of trusting prose.

It is read-only unless ``--write-receipt`` is supplied.  A receipt is published
atomically and exclusively; an existing destination is never replaced.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

CAP = 8192
BOOTSTRAP_REPS = 10_000
OPTIMIZER_STEP_BUDGET = 6_060
FINAL_LEARNING_RATE = 1.0e-6
SEEDS = (42, 43, 44)
STANDARD_TARGETS = ("CPTAC", "RIH", "SurGen", "TCGA")
MATCHED_TARGET = "RIH_sm"
METASTATIC_TARGETS = ("RIH", "SurGen")
E2C_ARMS = tuple(f"{arm}_k{k}" for arm in ("S1", "S2") for k in (2, 4, 8))

# Post-audit reporting thresholds documented by the corrected E2d4 source.
# They are interpretation guardrails, not hypothesis-test thresholds.
OVERLAP_THRESHOLDS = {
    "max_control_fraction_clipped": 0.10,
    "max_control_fraction_outside_empirical_common_support": 0.20,
    "min_control_ess_fraction": 0.25,
    "max_control_normalized_weight": 10.0,
}

EXPECTED_CORE_RESULTS = {
    "e2a_calibration": "e2a_source_calibration_cap{cap}.json",
    "e2a_calibration_matched": "e2a_source_calibration_cap{cap}_size_matched.json",
    "e2a": "e2a_transport_pb_cap{cap}.json",
    "e2a_matched": "e2a_transport_pb_cap{cap}_size_matched.json",
    "e2b": "e2b_metastatic_cap{cap}.json",
    "e2d1": "e2d1_metastatic_sites_cap{cap}.json",
    "e2d2": "e2d2_peritoneal_audit_cap{cap}.json",
    "e2d3": "e2d3_setd_contrast_cap{cap}.json",
    "e2d4": "e2d4_surgen_gap_cap{cap}.json",
    "e2d5": "e2d5_paired_specimens_cap{cap}.json",
    "e2d6": "e2d6_rih_acquisition_regime_cap{cap}.json",
}


class VerificationError(RuntimeError):
    """An artifact or semantic contract failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file(), f"Expected a file artifact: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _absolute_root(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    _require(path.is_absolute(), f"{label} must be an explicit absolute path: {path}")
    resolved = path.resolve(strict=True)
    _require(resolved.is_dir(), f"{label} is not a directory: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise VerificationError(f"Cannot read required JSON {path}: {exc}") from exc
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Malformed JSON object {path}: {exc}") from exc
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _same_number(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label} is not numeric: {value!r}") from exc
    _require(math.isfinite(result), f"{label} is not finite: {result}")
    return result


def _ci(value: object, label: str, *, unit_interval: bool = False) -> tuple[float, float]:
    _require(isinstance(value, list) and len(value) == 2, f"{label} is not a 2-value CI")
    low, high = _finite(value[0], f"{label}[0]"), _finite(value[1], f"{label}[1]")
    _require(low <= high, f"{label} is reversed: {value}")
    if unit_interval:
        _require(0.0 <= low <= high <= 1.0, f"{label} lies outside [0, 1]")
    return low, high


def _contrast_state(ci: object, label: str, *, null: float = 0.0) -> str:
    low, high = _ci(ci, label)
    if low > null:
        return "above_null_established"
    if high < null:
        return "below_null_established"
    return "not_established"


def _require_bootstrap(block: dict[str, Any], label: str, field: str = "n_bootstrap") -> None:
    _require(
        block.get(field) == BOOTSTRAP_REPS,
        f"{label} must record {BOOTSTRAP_REPS:,} requested bootstrap draws in {field}",
    )


class IdentityAuditor:
    """Validate and cache every recorded ``path/size/SHA256`` identity."""

    _required = frozenset({"path", "size_bytes", "sha256"})

    def __init__(
        self,
        *,
        live_source_root: Path | None = None,
        frozen_source_root: Path | None = None,
        allow_known_documentation_exception: bool = False,
    ) -> None:
        self.occurrences = 0
        self._cache: dict[Path, tuple[int, int, str]] = {}
        self.live_source_root = live_source_root
        self.frozen_source_root = frozen_source_root
        self.allow_known_documentation_exception = allow_known_documentation_exception
        self.documentation_exceptions: list[dict[str, Any]] = []

    def _is_known_documentation_exception(
        self, *, label: str, raw_path: Path, redirected: bool
    ) -> bool:
        if (
            not self.allow_known_documentation_exception
            or redirected
            or self.live_source_root is None
        ):
            return False
        expected = self.live_source_root / "reports" / "Experimental_Setup.md"
        return (
            label
            == "core_results.e2d6.input_artifacts.mapping_sources.prespecified_protocol"
            and raw_path.resolve(strict=True) == expected.resolve(strict=True)
        )

    def validate(
        self,
        recorded: object,
        label: str,
        *,
        actual_path: Path | None = None,
        allow_relative_record: bool = False,
    ) -> dict[str, Any]:
        _require(isinstance(recorded, dict), f"{label}: identity is not an object")
        _require(self._required.issubset(recorded), f"{label}: incomplete artifact identity")
        raw_path = Path(str(recorded["path"]))
        if not allow_relative_record:
            _require(raw_path.is_absolute(), f"{label}: recorded path is not absolute: {raw_path}")
        redirected = False
        path = actual_path if actual_path is not None else raw_path
        if (
            actual_path is None
            and self.live_source_root is not None
            and self.frozen_source_root is not None
        ):
            try:
                relative_source = raw_path.relative_to(self.live_source_root)
            except ValueError:
                pass
            else:
                frozen = self.frozen_source_root / relative_source
                if frozen.is_file():
                    path = frozen
                    redirected = True
        resolved = path.resolve(strict=True)
        _require(resolved.is_file(), f"{label}: recorded artifact is not a file: {resolved}")
        if actual_path is None and not redirected:
            _require(
                raw_path.resolve(strict=True) == resolved,
                f"{label}: recorded path does not resolve to its artifact",
            )
        stat = resolved.stat()
        cached = self._cache.get(resolved)
        if cached is None or cached[:2] != (stat.st_size, stat.st_mtime_ns):
            cached = (int(stat.st_size), int(stat.st_mtime_ns), _sha256_file(resolved))
            self._cache[resolved] = cached
        size, _mtime, digest = cached
        _require(
            isinstance(recorded["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", str(recorded["sha256"])) is not None,
            f"{label}: malformed SHA256",
        )
        size_matches = int(recorded["size_bytes"]) == size
        digest_matches = recorded["sha256"] == digest
        if not size_matches or not digest_matches:
            if self._is_known_documentation_exception(
                label=label, raw_path=raw_path, redirected=redirected
            ):
                self.documentation_exceptions.append(
                    {
                        "scope": "E2d6 protocol-context documentation only",
                        "label": label,
                        "recorded": {
                            "path": str(raw_path),
                            "size_bytes": int(recorded["size_bytes"]),
                            "sha256": str(recorded["sha256"]),
                        },
                        "current": {
                            "path": str(resolved),
                            "size_bytes": size,
                            "sha256": digest,
                        },
                        "reason": (
                            "The protocol Markdown changed after E2d6 and its exact old bytes "
                            "were not included in the v4 source snapshot. This exception does "
                            "not apply to numeric results, code, models, manifests, scores, "
                            "tables, or any other result input."
                        ),
                    }
                )
            elif not size_matches:
                raise VerificationError(f"{label}: size mismatch for {resolved}")
            else:
                raise VerificationError(f"{label}: SHA256 mismatch for {resolved}")
        self.occurrences += 1
        return {"path": str(resolved), "size_bytes": size, "sha256": digest}

    def scan(self, value: object, label: str) -> None:
        if isinstance(value, dict):
            if self._required.issubset(value):
                self.validate(value, label)
            for key, child in value.items():
                self.scan(child, f"{label}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self.scan(child, f"{label}[{index}]")

    def assert_stable(self) -> None:
        for path, (size, mtime, digest) in self._cache.items():
            stat = path.stat()
            _require(
                (stat.st_size, stat.st_mtime_ns) == (size, mtime),
                f"Artifact changed during verification: {path}",
            )
            _require(_sha256_file(path) == digest, f"Artifact changed during verification: {path}")

    @property
    def unique_files(self) -> int:
        return len(self._cache)


def _validate_core_start(
    core_root: Path, auditor: IdentityAuditor
) -> tuple[dict[str, Any], str]:
    start_path = core_root / "lineage_start.json"
    start = _read_json(start_path)
    _require(start.get("schema_version") == 1, "Core lineage_start schema is not 1")
    _require(start.get("status") == "started", "Core lineage is not a started immutable lineage")
    lineage_name = str(start.get("lineage", ""))
    _require(bool(lineage_name), "Core lineage_start has no lineage name")
    _require(
        Path(str(start.get("lineage_root", ""))).resolve(strict=True) == core_root,
        "--core-root does not match lineage_start.lineage_root",
    )

    source_rows = start.get("source_snapshot_inventory")
    live_rows = start.get("live_code_inventory")
    legacy_rows = start.get("legacy_inventory")
    for rows, label in (
        (source_rows, "source_snapshot_inventory"),
        (live_rows, "live_code_inventory"),
        (legacy_rows, "legacy_inventory"),
    ):
        _require(isinstance(rows, list), f"Core lineage_start.{label} is not a list")

    source_by_path: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(source_rows):
        rel = Path(str(row.get("path", ""))) if isinstance(row, dict) else Path("")
        _require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe snapshot inventory path")
        auditor.validate(
            row,
            f"core.lineage_start.source_snapshot_inventory[{index}]",
            actual_path=core_root / rel,
            allow_relative_record=True,
        )
        source_by_path[str(rel)] = row

    for index, row in enumerate(live_rows):
        rel = Path(str(row.get("path", ""))) if isinstance(row, dict) else Path("")
        _require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe live-code path")
        frozen = core_root / "source_snapshot" / rel
        auditor.validate(
            row,
            f"core.lineage_start.live_code_inventory[{index}]",
            actual_path=frozen,
            allow_relative_record=True,
        )
        snapshot_row = source_by_path.get(str(Path("source_snapshot") / rel))
        _require(snapshot_row is not None, f"Live source has no frozen snapshot inventory row: {rel}")
        _require(
            row["size_bytes"] == snapshot_row["size_bytes"]
            and row["sha256"] == snapshot_row["sha256"],
            f"Frozen source differs from the lineage-start live source: {rel}",
        )

    legacy_base = core_root.parent.parent
    for index, row in enumerate(legacy_rows):
        rel = Path(str(row.get("path", ""))) if isinstance(row, dict) else Path("")
        _require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe legacy path")
        auditor.validate(
            row,
            f"core.lineage_start.legacy_inventory[{index}]",
            actual_path=legacy_base / rel,
            allow_relative_record=True,
        )

    _require(start.get("source_file_count") == len(live_rows), "Core source-file count mismatch")
    _require(start.get("legacy_file_count") == len(legacy_rows), "Core legacy-file count mismatch")
    return start, lineage_name


def _load_core_results(
    core_root: Path,
    lineage_name: str,
    cap: int,
    auditor: IdentityAuditor,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    reports: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for key, template in EXPECTED_CORE_RESULTS.items():
        path = core_root / "eval" / template.format(cap=cap)
        _require(path.is_file(), f"Missing required corrected Aim-2 result: {path}")
        report = _read_json(path)
        _require(report.get("cap") == cap, f"{key}: cap mismatch")
        _require(report.get("lineage") == lineage_name, f"{key}: lineage mismatch")
        auditor.scan(report, f"core_results.{key}")
        reports[key] = report
        identities[key] = _artifact_identity(path)
    return reports, identities


def _validate_refit_receipt(
    record: object,
    *,
    target: str,
    seed: int,
    cap: int,
    lineage_name: str,
    core_root: Path,
    auditor: IdentityAuditor,
) -> dict[str, Any]:
    expected = (
        core_root
        / "e2a"
        / "train"
        / f"pb_cap{cap}"
        / target.lower()
        / f"seed{seed}"
        / "fit_summary.json"
    )
    identity = auditor.validate(
        record,
        f"refit {target}/seed{seed}",
    )
    _require(Path(identity["path"]) == expected, "Refit path is outside the selected core root")
    receipt = _read_json(expected)
    auditor.scan(receipt, f"refit_receipts.{target}.{seed}")
    _require(receipt.get("schema_version") == 2, f"{target}/seed{seed}: receipt schema")
    _require(receipt.get("status") == "completed", f"{target}/seed{seed}: refit incomplete")
    for key, expected_value in (
        ("lineage", lineage_name),
        ("target", target),
        ("seed", seed),
        ("sampling_seed", seed),
        ("cap", cap),
        ("sampler", "patient_natural"),
        ("loss_weighting", "none"),
    ):
        _require(receipt.get(key) == expected_value, f"{target}/seed{seed}: {key} mismatch")
    result = receipt.get("result")
    _require(isinstance(result, dict), f"{target}/seed{seed}: missing refit result")
    step_budget = receipt.get("optimizer_step_budget")
    _require(
        step_budget == OPTIMIZER_STEP_BUDGET,
        f"Optimizer budget must be the fixed {OPTIMIZER_STEP_BUDGET:,}-step contract",
    )
    contracts = {
        "strategy": "refit",
        "refit_max_steps": step_budget,
        "actual_optimizer_steps": step_budget,
        "batch_size": 1,
        "accumulate_grad_batches": 1,
        "seed": seed,
        "sampling_seed": seed,
        "train_sampling_strategy": "patient_natural",
        "sample_weight_column": None,
        "class_weights": None,
        "dataset_max_instances": cap,
        "max_instances": None,
        "eval_full_bags": True,
        "lr_scheduler": "cosine",
        "lr_scheduler_interval": "step",
        "lr_scheduler_total_steps": step_budget,
    }
    for key, expected_value in contracts.items():
        _require(result.get(key) == expected_value, f"{target}/seed{seed}: bad {key}")
    sampling = result.get("training_sampling")
    _require(isinstance(sampling, dict), f"{target}/seed{seed}: sampling audit missing")
    _require(
        sampling.get("strategy") == "patient_natural" and sampling.get("seed") == seed,
        f"{target}/seed{seed}: sampling audit mismatch",
    )
    _require(
        isinstance(sampling.get("samples_per_epoch"), int)
        and sampling["samples_per_epoch"] > 0,
        f"{target}/seed{seed}: invalid patient visits per epoch",
    )
    learning_rates = result.get("final_learning_rates")
    _require(
        isinstance(learning_rates, list)
        and len(learning_rates) == 1
        and _same_number(
            _finite(learning_rates[0], "final learning rate"),
            FINAL_LEARNING_RATE,
            tolerance=1e-12,
        ),
        f"{target}/seed{seed}: invalid final learning rate",
    )
    return identity


def _validate_refits(
    standard: dict[str, Any],
    matched: dict[str, Any],
    *,
    core_root: Path,
    lineage_name: str,
    cap: int,
    auditor: IdentityAuditor,
) -> dict[str, Any]:
    standard_map = standard.get("input_refits")
    matched_map = matched.get("input_refits")
    _require(isinstance(standard_map, dict), "E2a standard input_refits missing")
    _require(set(standard_map) == set(STANDARD_TARGETS), "E2a does not have exactly 12 refits")
    _require(isinstance(matched_map, dict), "E2a matched input_refits missing")
    _require(set(matched_map) == {MATCHED_TARGET}, "E2a does not have exactly 3 matched refits")
    seen: list[dict[str, Any]] = []
    for target in STANDARD_TARGETS:
        seed_map = standard_map[target]
        _require(set(map(str, seed_map)) == set(map(str, SEEDS)), f"{target}: incomplete seeds")
        for seed in SEEDS:
            seen.append(
                _validate_refit_receipt(
                    seed_map[str(seed)],
                    target=target,
                    seed=seed,
                    cap=cap,
                    lineage_name=lineage_name,
                    core_root=core_root,
                    auditor=auditor,
                )
            )
    seed_map = matched_map[MATCHED_TARGET]
    _require(set(map(str, seed_map)) == set(map(str, SEEDS)), "RIH_sm: incomplete seeds")
    for seed in SEEDS:
        seen.append(
            _validate_refit_receipt(
                seed_map[str(seed)],
                target=MATCHED_TARGET,
                seed=seed,
                cap=cap,
                lineage_name=lineage_name,
                core_root=core_root,
                auditor=auditor,
            )
        )
    _require(len({item["path"] for item in seen}) == 15, "Refit receipts are not 15 distinct files")
    return {"standard_completed": 12, "size_matched_completed": 3}


def _validate_calibration_reports(
    standard: dict[str, Any], matched: dict[str, Any], e2a: dict[str, Any]
) -> dict[str, Any]:
    for report, targets, label in (
        (standard, STANDARD_TARGETS, "standard"),
        (matched, (MATCHED_TARGET,), "size-matched"),
    ):
        blocks = report.get("targets")
        _require(isinstance(blocks, dict) and set(blocks) == set(targets), f"{label} calibration targets")
        for target in targets:
            block = blocks[target]
            _require(_finite(block.get("b"), f"{target} calibrator slope") > 0, "Non-positive calibrator")
            _require(int(block.get("n_source", 0)) > 0, f"{target}: empty calibration source")
            inputs = block.get("source_cv_inputs")
            _require(
                isinstance(inputs, dict) and set(map(str, inputs)) == set(map(str, SEEDS)),
                f"{target}: incomplete source-CV inputs",
            )
            note = str(block.get("note", "")).lower()
            _require("no target labels" in note, f"{target}: calibration leakage guardrail missing")
    for target in STANDARD_TARGETS:
        expected_n = e2a["targets"][target]["primary_overall"]["n"]
        _require(
            standard["targets"][target]["applied"]["primary"]["n"] == expected_n,
            f"{target}: calibrated/result patient count mismatch",
        )
    return {"positive_slope_source_only_calibrators": 5, "source_cv_seeds_per_target": 3}


def _validate_e2a(
    report: dict[str, Any], matched: dict[str, Any]
) -> dict[str, Any]:
    _require(report.get("schema_version") == 2, "E2a schema mismatch")
    _require(matched.get("schema_version") == 2, "E2a size-matched schema mismatch")
    _require(set(report.get("targets", {})) == set(STANDARD_TARGETS), "E2a targets mismatch")
    _require(set(matched.get("targets", {})) == {MATCHED_TARGET}, "E2a matched target mismatch")
    for item, label in ((report, "E2a"), (matched, "E2a matched")):
        inference = item.get("inference")
        _require(isinstance(inference, dict), f"{label}: missing inference metadata")
        _require_bootstrap(inference, f"{label} inference")
        _require(inference.get("sampling_unit") == "patient", f"{label}: wrong sampling unit")

    states: dict[str, str] = {}
    diagnosis = report.get("diagnosis")
    _require(isinstance(diagnosis, dict) and set(diagnosis) == set(STANDARD_TARGETS), "E2a diagnosis mismatch")
    for target in STANDARD_TARGETS:
        block = report["targets"][target]
        _require(block.get("seeds_complete") == list(SEEDS), f"{target}: incomplete E2a seeds")
        overall = block.get("primary_overall")
        _require(isinstance(overall, dict), f"{target}: missing primary result")
        _require_bootstrap(overall, f"{target} primary")
        auc = _finite(overall.get("auroc"), f"{target} AUROC")
        low, high = _ci(overall.get("auroc_ci"), f"{target} AUROC CI", unit_interval=True)
        _require(low <= auc <= high, f"{target}: AUROC is outside its interval")
        state = _contrast_state(overall["auroc_ci"], f"{target} transport CI", null=0.5)
        states[target] = state
        claimed = diagnosis[target].get("ranking_transports")
        _require(claimed is (state == "above_null_established"), f"{target}: transport claim/CI mismatch")
        _require(_same_number(diagnosis[target].get("auroc"), auc), f"{target}: diagnosis AUROC mismatch")
        _require(_same_number(diagnosis[target].get("auroc_ci_low"), low), f"{target}: diagnosis CI mismatch")
        source_cal = block.get("source_calibrated")
        _require(isinstance(source_cal, dict), f"{target}: source calibration missing")
        _require(source_cal.get("auroc_unchanged") is True, f"{target}: calibration changed ranking")
        _require(
            _same_number(source_cal.get("auroc_exact_logit_scale"), auc),
            f"{target}: calibrated AUROC does not equal raw AUROC",
        )

    matched_block = matched["targets"][MATCHED_TARGET]
    _require(matched_block.get("seeds_complete") == list(SEEDS), "RIH_sm seeds incomplete")
    _require_bootstrap(matched_block["primary_overall"], "RIH_sm primary")
    sensitivity = matched.get("size_matched_sensitivity")
    _require(isinstance(sensitivity, dict), "Missing size-matched sensitivity")
    _require_bootstrap(sensitivity, "Size-matched sensitivity")
    left = report["targets"]["RIH"]["primary_overall"]["auroc"]
    right = matched_block["primary_overall"]["auroc"]
    _require(_same_number(sensitivity.get("auroc_left"), left), "Size sensitivity left AUROC mismatch")
    _require(_same_number(sensitivity.get("auroc_right"), right), "Size sensitivity right AUROC mismatch")
    _require(_same_number(sensitivity.get("delta_auroc"), right - left), "Size sensitivity delta mismatch")
    _require(
        sensitivity.get("standard_refits") == report.get("input_refits", {}).get("RIH"),
        "Size sensitivity standard-refit identities differ from E2a",
    )
    _require(
        sensitivity.get("size_matched_refits")
        == matched.get("input_refits", {}).get(MATCHED_TARGET),
        "Size sensitivity matched-refit identities differ from E2a",
    )
    size_state = _contrast_state(
        [sensitivity.get("ci_low"), sensitivity.get("ci_high")], "Size sensitivity CI"
    )
    return {"ranking_transport": states, "size_matched_minus_full": size_state}


def _validate_e2b(report: dict[str, Any]) -> dict[str, Any]:
    _require(set(report.get("targets", {})) == set(METASTATIC_TARGETS), "E2b targets mismatch")
    inference = report.get("inference")
    _require(isinstance(inference, dict), "E2b inference metadata missing")
    _require_bootstrap(inference, "E2b inference")
    _require(inference.get("sampling_unit") == "patient", "E2b sampling unit is not patient")

    target_deltas: dict[str, float] = {}
    metastatic_aucs: list[float] = []
    for target in METASTATIC_TARGETS:
        block = report["targets"][target]
        _require(block.get("seeds_complete") == list(SEEDS), f"{target}: incomplete E2b seeds")
        _require_bootstrap(block["metastatic_overall"], f"{target} metastatic")
        contrast = block.get("primary_vs_metastatic")
        _require(isinstance(contrast, dict), f"{target}: missing primary/metastatic contrast")
        _require_bootstrap(contrast, f"{target} primary/metastatic contrast")
        expected_delta = _finite(contrast["metastatic_auroc"], "metastatic AUROC") - _finite(
            contrast["primary_auroc"], "primary AUROC"
        )
        _require(_same_number(contrast.get("delta_auroc"), expected_delta), f"{target}: delta mismatch")
        target_deltas[target] = expected_delta
        metastatic_aucs.append(_finite(block["metastatic_overall"]["auroc"], "met AUROC"))
    rih_definition = str(report["targets"]["RIH"]["primary_vs_metastatic"].get("definition", "")).lower()
    _require(
        "exclud" in rih_definition and "both" in rih_definition,
        "E2b RIH contrast does not document dual-role exclusion from both arms",
    )

    conclusion = report.get("conclusion")
    _require(isinstance(conclusion, dict), "E2b fixed transport conclusion missing")
    _require_bootstrap(conclusion, "E2b fixed metastatic-transport rule")
    macro = float(sum(metastatic_aucs) / len(metastatic_aucs))
    _require(_same_number(conclusion.get("metastatic_macro_auroc"), macro), "E2b macro AUROC mismatch")
    macro_ci = _ci(conclusion.get("metastatic_macro_ci"), "E2b metastatic macro CI")
    fixed_transport = macro_ci[0] > 0.5
    _require(
        conclusion.get("claim_metastatic_transport") is fixed_transport,
        "E2b fixed metastatic-transport claim disagrees with its CI rule",
    )
    _require(
        conclusion.get("both_point_estimates_above_0.5")
        is all(value > 0.5 for value in metastatic_aucs),
        "E2b point-estimate diagnostic mismatch",
    )
    caveat = str(conclusion.get("caveat", "")).lower()
    _require("equivalence" in caveat and "causal" in caveat, "E2b caveat guardrail missing")

    combined = report.get("combined_decrement")
    _require(isinstance(combined, dict), "E2b combined decrement missing")
    _require_bootstrap(combined, "E2b combined decrement")
    expected_combined = sum(target_deltas.values()) / len(target_deltas)
    _require(_same_number(combined.get("delta_auroc"), expected_combined), "E2b combined delta mismatch")
    combined_state = _contrast_state(combined.get("delta_auroc_ci"), "E2b combined decrement CI")
    excludes = combined_state != "not_established"
    decrement = combined_state == "below_null_established"
    _require(combined.get("ci_excludes_zero") is excludes, "E2b combined CI flag mismatch")
    _require(
        combined.get("evidence_of_overall_decrement") is decrement,
        "E2b decrement claim disagrees with its CI",
    )
    if not decrement:
        _require("not established" in str(combined.get("inference", "")).lower(), "E2b overclaim")
    classification = report.get("classification")
    _require(isinstance(classification, dict), "E2b classification missing")
    _require(classification.get("outcome") == combined.get("inference"), "E2b outcome mismatch")
    return {
        "fixed_metastatic_transport": (
            "established" if fixed_transport else "not_established"
        ),
        "overall_primary_to_metastatic_change": combined_state,
    }


def _validate_e2d1(report: dict[str, Any]) -> dict[str, Any]:
    _require(set(report.get("cohorts", {})) == set(METASTATIC_TARGETS), "E2d1 cohorts mismatch")
    _require_bootstrap(report.get("inference", {}), "E2d1 inference")
    contrasts: dict[str, dict[str, Any]] = {}
    for cohort in METASTATIC_TARGETS:
        contrast = report["cohorts"][cohort].get("liver_vs_non_liver")
        _require(isinstance(contrast, dict), f"E2d1 {cohort} contrast missing")
        _require_bootstrap(contrast, f"E2d1 {cohort} contrast")
        _ci(contrast.get("ci"), f"E2d1 {cohort} CI")
        contrasts[cohort] = contrast
    deltas = [float(contrasts[name]["delta"]) for name in METASTATIC_TARGETS]
    same_sign = (deltas[0] > 0) == (deltas[1] > 0) and all(value != 0 for value in deltas)
    excludes = {
        name: _contrast_state(contrasts[name]["ci"], f"E2d1 {name} CI")
        != "not_established"
        for name in METASTATIC_TARGETS
    }
    both = all(excludes.values())
    expected_verdict = (
        "replicated"
        if same_sign and both
        else "single-cohort finding"
        if any(excludes.values())
        else "point directions align; effect not established"
        if same_sign
        else "not supported"
    )
    recorded = report.get("concordance")
    _require(isinstance(recorded, dict), "E2d1 concordance block missing")
    _require(recorded.get("same_sign") is same_sign, "E2d1 sign concordance mismatch")
    _require(recorded.get("both_cis_exclude_zero") is both, "E2d1 replication flag mismatch")
    _require(recorded.get("per_cohort_ci_excludes_zero") == excludes, "E2d1 CI flags mismatch")
    _require(recorded.get("verdict") == expected_verdict, "E2d1 verdict mismatch")
    _require(
        str(recorded.get("point_sign_inferential_role", "")).lower().startswith("none"),
        "E2d1 point signs are incorrectly treated as inference",
    )
    return {"organ_effect_replication": "established" if expected_verdict == "replicated" else expected_verdict}


def _validate_e2d2(report: dict[str, Any], e2d1: dict[str, Any]) -> dict[str, Any]:
    peritoneal = e2d1["cohorts"]["SurGen"]["sites"]["peritoneum"]
    _require(report.get("n") == peritoneal.get("n"), "E2d2/E2d1 peritoneal n mismatch")
    _require(
        _same_number(report.get("ensemble_auroc"), peritoneal.get("auroc")),
        "E2d2/E2d1 peritoneal AUROC mismatch",
    )
    _require(
        str(report.get("seed_results_inferential_role", "")).lower().startswith("none"),
        "E2d2 training seeds are incorrectly treated as replicates",
    )
    _require(isinstance(report.get("loo"), dict), "E2d2 leave-one-out audit missing")
    return {
        "peritoneal_result": "single_cohort_exploratory",
        "training_seeds_inferential": False,
    }


def _validate_e2d3(report: dict[str, Any]) -> dict[str, Any]:
    _require(set(report.get("cohorts", {})) == set(METASTATIC_TARGETS), "E2d3 cohorts mismatch")
    _require_bootstrap(report.get("inference", {}), "E2d3 inference")
    states: dict[str, str] = {}
    changes: list[float] = []
    for cohort in METASTATIC_TARGETS:
        block = report["cohorts"][cohort]
        for population in ("full_population", "set_d"):
            _require_bootstrap(block[population], f"E2d3 {cohort} {population}")
        change = block.get("restriction_change")
        _require(isinstance(change, dict), f"E2d3 {cohort} restriction change missing")
        _require_bootstrap(change, f"E2d3 {cohort} restriction change", "n_bootstrap_requested")
        state = _contrast_state(change.get("change_delta_auroc_ci"), f"E2d3 {cohort} change CI")
        states[cohort] = state
        changes.append(float(change["change_delta_auroc"]))
        _require(change.get("ci_excludes_zero") is (state != "not_established"), "E2d3 CI flag")
        if state == "not_established":
            _require("not established" in str(change.get("inference", "")).lower(), "E2d3 overclaim")
    combined = report.get("combined_restriction_change")
    _require(isinstance(combined, dict), "E2d3 combined restriction change missing")
    _require_bootstrap(combined, "E2d3 combined change", "n_bootstrap_requested")
    _require(
        _same_number(combined.get("change_delta_auroc"), sum(changes) / len(changes)),
        "E2d3 combined change is not the equal-cohort mean",
    )
    combined_state = _contrast_state(combined.get("change_delta_auroc_ci"), "E2d3 combined CI")
    _require(
        combined.get("ci_excludes_zero") is (combined_state != "not_established"),
        "E2d3 combined CI flag mismatch",
    )
    if combined_state == "not_established":
        _require("not established" in str(combined.get("inference", "")).lower(), "E2d3 overclaim")
    states["combined"] = combined_state
    return {"set_d_minus_full_change": states, "widening_established": combined_state == "below_null_established"}


def _derive_overlap_gate(block: dict[str, Any], label: str) -> dict[str, Any]:
    diagnostics = block.get("diagnostics")
    _require(isinstance(diagnostics, dict), f"{label}: overlap diagnostics missing")
    groups = diagnostics.get("propensity_by_subcohort")
    _require(isinstance(groups, dict) and "SR1482" in groups, f"{label}: SR1482 diagnostics missing")
    control = groups["SR1482"]
    observed = {
        "control_fraction_clipped": _finite(control.get("fraction_clipped"), f"{label} clipping"),
        "control_fraction_outside_empirical_common_support": _finite(
            control.get("fraction_outside_empirical_common_support"), f"{label} support"
        ),
        "control_ess_fraction": _finite(
            diagnostics.get("SR1482_ess_fraction"), f"{label} ESS fraction"
        ),
        "control_max_normalized_weight": _finite(
            diagnostics.get("SR1482_max_weight"), f"{label} max weight"
        ),
    }
    checks = {
        "clipping": observed["control_fraction_clipped"]
        <= OVERLAP_THRESHOLDS["max_control_fraction_clipped"],
        "common_support": observed["control_fraction_outside_empirical_common_support"]
        <= OVERLAP_THRESHOLDS[
            "max_control_fraction_outside_empirical_common_support"
        ],
        "effective_sample_size": observed["control_ess_fraction"]
        >= OVERLAP_THRESHOLDS["min_control_ess_fraction"],
        "maximum_weight": observed["control_max_normalized_weight"]
        <= OVERLAP_THRESHOLDS["max_control_normalized_weight"],
    }
    adequate = all(checks.values())
    derived = {
        "overlap_adequate": adequate,
        "estimable_for_inference": adequate,
        "status": (
            "diagnostically adequate overlap"
            if adequate
            else "limited overlap; positivity diagnostic only"
        ),
        "observed": observed,
        "thresholds": dict(OVERLAP_THRESHOLDS),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "provenance": "post-audit reporting guardrail; not a pre-registered hypothesis test",
        "claim_guardrail": (
            "A non-estimable arm cannot support a claim that adjustment explains, "
            "or fails to explain, the subcohort gap."
        ),
    }
    recorded = block.get("overlap_assessment")
    if recorded is not None:
        for key in ("overlap_adequate", "estimable_for_inference", "observed", "thresholds", "checks", "failed_checks"):
            _require(recorded.get(key) == derived[key], f"{label}: recorded overlap gate mismatch")
    return derived


def _validate_e2d4(report: dict[str, Any]) -> dict[str, Any]:
    _require(report.get("schema_version") == 2, "E2d4 must use corrected schema 2")
    _require_bootstrap(report.get("inference", {}), "E2d4 inference")
    crude = report.get("delta_crude")
    _require(isinstance(crude, dict), "E2d4 crude contrast missing")
    _require_bootstrap(crude, "E2d4 crude contrast")
    crude_expected = float(report["SR386"]["auroc"]) - float(report["SR1482"]["auroc"])
    _require(_same_number(crude.get("delta"), crude_expected), "E2d4 crude delta mismatch")
    crude_state = _contrast_state(
        [crude.get("ci_low"), crude.get("ci_high")], "E2d4 crude CI"
    )
    d_subset = report.get("D_subset", {}).get("delta")
    _require(isinstance(d_subset, dict), "E2d4 Set-D contrast missing")
    _require_bootstrap(d_subset, "E2d4 Set-D contrast")

    adjustment_keys = (
        "ipw_age + site",
        "ipw_+ MSI/BRAF",
        "ipw_+ slide count & size",
        "ipw_+ sex",
        "ipw_+ filled stage",
    )
    gates: dict[str, Any] = {}
    for key in adjustment_keys:
        block = report.get(key)
        _require(isinstance(block, dict), f"E2d4 adjustment missing: {key}")
        _require_bootstrap(block, f"E2d4 {key}")
        _require(block.get("n_propensity_fits") == BOOTSTRAP_REPS + 1, f"{key}: propensity fits")
        expected = float(block["auroc_SR386"]) - float(block["auroc_SR1482_reweighted"])
        _require(_same_number(block.get("delta"), expected), f"{key}: adjusted delta mismatch")
        state = _contrast_state(block.get("delta_ci"), f"E2d4 {key} CI")
        gate = _derive_overlap_gate(block, f"E2d4 {key}")
        gate["contrast_state"] = state
        gate["positive_gap_established"] = gate["estimable_for_inference"] and state == "above_null_established"
        gates[key] = gate
    return {
        "crude_gap": crude_state,
        "overlap_guardrail": gates,
        "non_estimable_adjustments_are_diagnostics_only": True,
    }


def _validate_e2d5(report: dict[str, Any]) -> dict[str, Any]:
    population = report.get("population_audit")
    table = report.get("paired_table")
    headline = report.get("headline_RIH_same_model")
    _require(isinstance(population, dict), "E2d5 population audit missing")
    _require(isinstance(table, dict) and isinstance(table.get("rows"), list), "E2d5 table missing")
    _require(isinstance(headline, dict), "E2d5 RIH headline missing")
    _require(population.get("n_pairs") == len(table["rows"]), "E2d5 pair count mismatch")
    _require(
        headline.get("n_pairs") == population.get("cohort_counts", {}).get("RIH"),
        "E2d5 RIH pair count mismatch",
    )
    bootstrap = headline.get("bootstrap")
    _require(isinstance(bootstrap, dict), "E2d5 bootstrap missing")
    _require_bootstrap(bootstrap, "E2d5 paired bootstrap", "n_bootstrap_requested")
    intervals = bootstrap.get("intervals")
    _require(isinstance(intervals, dict) and bool(intervals), "E2d5 intervals missing")
    for name, interval in intervals.items():
        _require(
            interval.get("n_bootstrap_valid") == BOOTSTRAP_REPS,
            f"E2d5 {name}: incomplete bootstrap interval",
        )
    shift_ci = bootstrap.get("intervals", {}).get("mean_logit_shift", {}).get("ci_95_percentile")
    shift_state = _contrast_state(shift_ci, "E2d5 mean paired shift CI")
    tcga = report.get("TCGA_single_pair")
    _require(isinstance(tcga, dict) and isinstance(tcga.get("row"), dict), "E2d5 TCGA pair missing")
    _require("descriptive only" in str(tcga.get("scope", "")).lower(), "E2d5 TCGA guardrail missing")
    guardrails = report.get("guardrails")
    required_false = (
        "auroc_computed",
        "equivalence_claim_allowed",
        "noninferiority_claim_allowed",
        "causal_specimen_role_claim_allowed",
    )
    _require(isinstance(guardrails, dict), "E2d5 guardrails missing")
    for key in required_false:
        _require(guardrails.get(key) is False, f"E2d5 guardrail failed: {key}")
    _require("not evidence" in str(guardrails.get("interpretation", "")).lower(), "E2d5 CI guardrail missing")
    return {"paired_logit_shift": shift_state, "equivalence_claim_allowed": False}


def _validate_e2d6(report: dict[str, Any]) -> dict[str, Any]:
    _require(report.get("schema_version") == 1, "E2d6 schema mismatch")
    _require(report.get("seeds_complete") == list(SEEDS), "E2d6 incomplete model seeds")
    _require_bootstrap(report.get("inference", {}), "E2d6 inference")
    mapping = report.get("mapping_audit")
    _require(isinstance(mapping, dict), "E2d6 mapping audit missing")
    _require(
        mapping.get("repair_ledger_membership_matches_repaired_regime") is True,
        "E2d6 repair-ledger mapping mismatch",
    )
    aligned = report.get("e2b_aligned_primary_to_metastatic")
    _require(isinstance(aligned, dict), "E2d6 aligned contrasts missing")
    rule = str(aligned.get("dual_role_rule", "")).lower()
    _require("exclud" in rule and "both" in rule, "E2d6 dual-role exclusion guardrail missing")
    within = aligned.get("within_regime")
    expected_regimes = {"aperio_native", "repaired_converted_technical_regime"}
    _require(isinstance(within, dict) and set(within) == expected_regimes, "E2d6 regime arms mismatch")
    regime_deltas: dict[str, float] = {}
    for name in expected_regimes:
        block = within[name]
        _require_bootstrap(block, f"E2d6 {name}")
        expected = float(block["metastatic"]["auroc"]) - float(block["primary"]["auroc"])
        _require(_same_number(block.get("delta_auroc"), expected), f"E2d6 {name} delta mismatch")
        regime_deltas[name] = expected
    interaction = aligned.get("four_arm_interaction")
    _require(isinstance(interaction, dict), "E2d6 interaction missing")
    _require_bootstrap(interaction, "E2d6 interaction")
    expected_interaction = (
        regime_deltas["repaired_converted_technical_regime"]
        - regime_deltas["aperio_native"]
    )
    _require(
        _same_number(interaction.get("interaction_delta_auroc"), expected_interaction),
        "E2d6 interaction delta mismatch",
    )
    interaction_state = _contrast_state(
        interaction.get("interaction_delta_auroc_ci"), "E2d6 interaction CI"
    )
    _require(
        interaction.get("ci_excludes_zero") is (interaction_state != "not_established"),
        "E2d6 interaction CI flag mismatch",
    )
    _require(
        interaction.get("four_arms_pairwise_patient_disjoint") is True,
        "E2d6 interaction arms are not patient-disjoint",
    )
    if interaction_state == "not_established":
        _require("not established" in str(interaction.get("inference", "")).lower(), "E2d6 overclaim")
    versa = aligned.get("versa")
    _require(
        isinstance(versa, dict) and "descriptive only" in str(versa.get("status", "")).lower(),
        "E2d6 Versa must be descriptive only",
    )
    guardrails = report.get("guardrails")
    _require(isinstance(guardrails, dict), "E2d6 guardrails missing")
    for key in (
        "scanner_claim_allowed",
        "causal_specimen_role_claim_allowed",
        "equivalence_or_invariance_claim_allowed",
        "versa_inferential_claim_allowed",
    ):
        _require(guardrails.get(key) is False, f"E2d6 guardrail failed: {key}")
    _require("confounded" in str(guardrails.get("technical_regime_interpretation", "")).lower(), "E2d6 confounding guardrail missing")
    return {"technical_regime_interaction": interaction_state, "scanner_effect_claim_allowed": False}


def _claim_is_affirmative(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    if any(
        phrase in lowered
        for phrase in (
            "not established",
            "no evidence",
            "not supported",
            "cannot establish",
            "does not establish",
        )
    ):
        return False
    return bool(
        re.search(
            r"\b(superior|superiority established|improvement established|benefit established)\b",
            lowered,
        )
    )


def _assert_no_unsupported_positive_claims(value: object, label: str = "e2c") -> None:
    """Reject affirmative superiority language without a positive supporting CI."""
    if isinstance(value, dict):
        local_ci = value.get("ci")
        local_support = (
            _contrast_state(local_ci, f"{label}.ci") == "above_null_established"
            if local_ci is not None
            else False
        )
        for key, child in value.items():
            lowered_key = key.lower()
            if isinstance(child, bool) and child and any(
                token in lowered_key for token in ("superior", "improvement", "benefit")
            ):
                _require(local_support, f"{label}.{key}: unsupported positive claim")
            if isinstance(child, str) and lowered_key in {
                "claim",
                "conclusion",
                "inference",
                "verdict",
                "interpretation",
                "status",
            }:
                _require(
                    not _claim_is_affirmative(child) or local_support,
                    f"{label}.{key}: unsupported positive claim",
                )
            _assert_no_unsupported_positive_claims(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_unsupported_positive_claims(child, f"{label}[{index}]")


def _expected_e2c_contrasts() -> set[str]:
    out: set[str] = set()
    for k in (2, 4, 8):
        out.add(f"S2_minus_S1_k{k}")
        out.add(f"S1_k{k}_minus_S0")
        out.add(f"S2_k{k}_minus_S0")
    return out


def _validate_e2c(
    e2c_root: Path,
    core_root: Path,
    lineage_name: str,
    cap: int,
    e2b: dict[str, Any],
    auditor: IdentityAuditor,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    paths = {
        "lineage_start": e2c_root / "lineage_start.json",
        "upstream_imports": e2c_root / "receipts" / "upstream_imports.json",
        "analysis": e2c_root / "analysis" / f"e2c_native_logit_offset_cap{cap}.json",
        "analysis_audit": e2c_root / "receipts" / "analysis_audit.json",
        "lineage_complete": e2c_root / "lineage_complete.json",
    }
    for label, path in paths.items():
        _require(path.is_file(), f"E2c is partial: missing {label} at {path}")
    docs = {label: _read_json(path) for label, path in paths.items()}
    identities = {label: _artifact_identity(path) for label, path in paths.items()}
    for label, doc in docs.items():
        auditor.scan(doc, f"e2c.{label}")

    start = docs["lineage_start"]
    _require(start.get("status") == "started", "E2c lineage did not start cleanly")
    _require(start.get("component") == "e2c_native_logit_residual_offset", "Wrong E2c component")
    _require(Path(str(start.get("output_root", ""))).resolve(strict=True) == e2c_root, "E2c output root mismatch")
    _require(Path(str(start.get("input_lineage_root", ""))).resolve(strict=True) == core_root, "E2c input root mismatch")
    _require(start.get("input_lineage") == lineage_name, "E2c input lineage mismatch")
    _require(start.get("cap") == cap, "E2c cap mismatch")
    _require(start.get("reps") == 100, "E2c must complete exactly 100 procedure repetitions")
    _require(start.get("n_bootstrap") == BOOTSTRAP_REPS, "E2c start bootstrap count mismatch")

    complete = docs["lineage_complete"]
    _require(complete.get("status") == "completed", "E2c lineage is not completed")
    _require(complete.get("input_lineage") == lineage_name, "E2c completion input mismatch")
    expected_artifacts = {"lineage_start", "upstream_imports", "analysis", "analysis_audit", "source_snapshot"}
    _require(set(complete.get("artifacts", {})) == expected_artifacts, "E2c completion inventory mismatch")
    for label in ("lineage_start", "upstream_imports", "analysis", "analysis_audit"):
        recorded = complete["artifacts"][label]
        _require(Path(str(recorded.get("path", ""))).resolve(strict=True) == paths[label], f"E2c {label} path mismatch")

    imports = docs["upstream_imports"]
    _require(imports.get("input_lineage") == lineage_name, "E2c import lineage mismatch")
    _require(Path(str(imports.get("input_lineage_root", ""))).resolve(strict=True) == core_root, "E2c import root mismatch")
    _require(
        imports.get("canonical_e2c_result", {}).get("consumed") is False,
        "E2c recovery consumed the failed canonical E2c result",
    )

    report = docs["analysis"]
    _require(report.get("component") == "e2c_native_logit_residual_offset", "Wrong E2c result")
    _require(report.get("input_lineage") == lineage_name, "E2c analysis input mismatch")
    _require(Path(str(report.get("input_lineage_root", ""))).resolve(strict=True) == core_root, "E2c analysis root mismatch")
    _require(report.get("cap") == cap, "E2c analysis cap mismatch")
    _require(report.get("reps") == 100, "E2c analysis repetition count mismatch")
    _require(report.get("n_bootstrap") == BOOTSTRAP_REPS, "E2c analysis bootstrap mismatch")
    design = report.get("design")
    _require(isinstance(design, dict), "E2c design block missing")
    _require(design.get("n_bootstrap") == BOOTSTRAP_REPS, "E2c design bootstrap mismatch")
    _require("bit-exact native S0" in str(design.get("lambda_infinity", "")), "E2c S0 continuity contract missing")

    audit = docs["analysis_audit"]
    _require(audit.get("status") == "PASS", "E2c analysis audit did not pass")
    _require(audit.get("upstream_unchanged_during_run") == "PASS", "E2c upstream changed")
    _require(audit.get("checks", {}).get("n_bootstrap") == BOOTSTRAP_REPS, "E2c audit bootstrap mismatch")

    _require(set(report.get("cohorts", {})) == set(METASTATIC_TARGETS), "E2c cohorts mismatch")
    contrast_states: dict[str, dict[str, str]] = {}
    for target in METASTATIC_TARGETS:
        block = report["cohorts"][target]
        _require(block.get("reps") == 100, f"E2c {target}: repetitions mismatch")
        _require(set(block.get("arms", {})) == set(E2C_ARMS), f"E2c {target}: arm inventory mismatch")
        _require(
            set(block.get("contrasts", {})) == _expected_e2c_contrasts(),
            f"E2c {target}: contrast inventory mismatch",
        )
        s0 = _finite(block.get("S0", {}).get("auroc"), f"E2c {target} S0 AUROC")
        e2b_s0 = _finite(e2b["targets"][target]["metastatic_overall"]["auroc"], "E2b S0")
        _require(_same_number(s0, e2b_s0, tolerance=0.0), f"E2c {target}: S0 != E2b")
        target_audit = audit.get("checks", {}).get(target)
        _require(isinstance(target_audit, dict), f"E2c {target}: audit block missing")
        _require(
            target_audit.get("S0_native_exact") == "PASS"
            and target_audit.get("S0_equals_E2b") == "PASS",
            f"E2c {target}: S0 equality audit failed",
        )
        for arm in E2C_ARMS:
            arm_block = block["arms"][arm]
            _require(arm_block.get("reps_completed") == 100, f"E2c {target}/{arm}: incomplete")
            _require(arm_block.get("support_failures") == 0, f"E2c {target}/{arm}: support failure")
            _require(arm_block.get("performance", {}).get("n_procedure_draws") == 100, f"E2c {target}/{arm}: draw count")
            _require(arm_block.get("solver_audit", {}).get("status") == "PASS", f"E2c {target}/{arm}: solver audit")
            arm_audit = target_audit.get("arms", {}).get(arm, {})
            for key in (
                "predictions_complete",
                "supports_exact_and_leak_free",
                "lambda_infinity_continuity",
                "finite_solver_kkt_and_objective",
            ):
                _require(arm_audit.get(key) == "PASS", f"E2c {target}/{arm}: {key} audit")

        states: dict[str, str] = {}
        for name, contrast in block["contrasts"].items():
            state = _contrast_state(contrast.get("ci"), f"E2c {target}/{name} CI")
            if name.startswith("S2_minus_S1_"):
                k = name.rsplit("_k", 1)[1]
                expected = (
                    float(block["arms"][f"S2_k{k}"]["performance"]["metrics"]["auroc"])
                    - float(block["arms"][f"S1_k{k}"]["performance"]["metrics"]["auroc"])
                )
            else:
                arm = name.removesuffix("_minus_S0")
                expected = float(block["arms"][arm]["performance"]["metrics"]["auroc"]) - s0
            _require(_same_number(contrast.get("delta"), expected), f"E2c {target}/{name}: delta mismatch")
            states[name] = state
        contrast_states[target] = states

    _assert_no_unsupported_positive_claims(report)
    return (
        {
            "status": "completed_and_audited",
            "S0_native_equals_E2b": True,
            "contrast_states": contrast_states,
            "interpretation_guardrail": (
                "Only a CI wholly above zero is called superiority; a CI crossing zero is "
                "not established and is never converted into equivalence or no benefit."
            ),
        },
        identities,
    )


def verify(
    core_root: Path,
    e2c_root: Path,
    cap: int = CAP,
    *,
    allow_known_documentation_exception: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable PASS receipt or raise ``VerificationError``."""
    _require(cap == CAP, f"This verifier is intentionally scoped to cap={CAP}; got {cap}")
    _require(core_root != e2c_root, "Core and E2c roots must be distinct")
    _require(
        core_root not in e2c_root.parents and e2c_root not in core_root.parents,
        "Core and E2c roots must not overlap",
    )
    repository = Path(__file__).resolve().parents[1]
    auditor = IdentityAuditor(
        live_source_root=repository,
        frozen_source_root=core_root / "source_snapshot",
        allow_known_documentation_exception=allow_known_documentation_exception,
    )
    start, lineage_name = _validate_core_start(core_root, auditor)
    reports, core_result_identities = _load_core_results(
        core_root, lineage_name, cap, auditor
    )

    refits = _validate_refits(
        reports["e2a"],
        reports["e2a_matched"],
        core_root=core_root,
        lineage_name=lineage_name,
        cap=cap,
        auditor=auditor,
    )
    calibration = _validate_calibration_reports(
        reports["e2a_calibration"], reports["e2a_calibration_matched"], reports["e2a"]
    )
    semantic = {
        "E2a": _validate_e2a(reports["e2a"], reports["e2a_matched"]),
        "E2b": _validate_e2b(reports["e2b"]),
        "E2d1": _validate_e2d1(reports["e2d1"]),
        "E2d2": _validate_e2d2(reports["e2d2"], reports["e2d1"]),
        "E2d3": _validate_e2d3(reports["e2d3"]),
        "E2d4": _validate_e2d4(reports["e2d4"]),
        "E2d5": _validate_e2d5(reports["e2d5"]),
        "E2d6": _validate_e2d6(reports["e2d6"]),
    }
    e2c_check, e2c_identities = _validate_e2c(
        e2c_root,
        core_root,
        lineage_name,
        cap,
        reports["e2b"],
        auditor,
    )
    semantic["E2c"] = e2c_check
    auditor.assert_stable()

    exception_used = bool(auditor.documentation_exceptions)
    overall_status = (
        "NUMERIC_AND_CODE_PASS_WITH_DOCUMENTATION_PROVENANCE_EXCEPTION"
        if exception_used
        else "PASS"
    )
    return {
        "schema_version": 1,
        "status": overall_status,
        "scientific_and_numeric_checks": "PASS",
        "code_model_data_identity_checks": "PASS",
        "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "corrected Aim 2, cap 8192",
        "cap": cap,
        "core_lineage": lineage_name,
        "core_root": str(core_root),
        "core_lineage_state": start.get("status"),
        "e2c_root": str(e2c_root),
        "required_core_results": core_result_identities,
        "required_e2c_artifacts": e2c_identities,
        "identity_audit": {
            "status": "DOCUMENTATION_EXCEPTION" if exception_used else "PASS",
            "recorded_identity_occurrences": auditor.occurrences,
            "unique_files_hashed": auditor.unique_files,
            "paths_sizes_sha256_live_except_listed_documentation": True,
            "stable_during_verification": True,
        },
        "documentation_provenance": {
            "status": "EXCEPTION" if exception_used else "PASS",
            "exceptions": auditor.documentation_exceptions,
        },
        "refits": refits,
        "calibration": calibration,
        "semantic_checks": semantic,
        "interpretation_guardrail": (
            "Every claim state is derived mechanically from its recorded confidence interval. "
            "An interval crossing its null means not established; it never means equivalence, "
            "invariance, no effect, or no benefit."
        ),
        "verifier": _artifact_identity(Path(__file__)),
    }


def _write_json_once_atomic(path: Path, value: dict[str, Any]) -> None:
    _require(path.is_absolute(), f"--write-receipt must be an absolute path: {path}")
    _require(path.parent.is_dir(), f"Receipt parent does not exist: {path.parent}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite verification receipt: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _receipt_destination(value: str | Path, *, forbidden_roots: tuple[Path, ...]) -> Path:
    path = Path(value).expanduser()
    _require(path.is_absolute(), f"--write-receipt must be an absolute path: {path}")
    resolved = path.resolve(strict=False)
    for root in forbidden_roots:
        _require(
            resolved != root and root not in resolved.parents,
            f"Receipt must not modify immutable lineage root: {root}",
        )
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", required=True, help="absolute corrected core lineage root")
    parser.add_argument("--e2c-root", required=True, help="absolute completed E2c recovery root")
    parser.add_argument("--cap", type=int, default=CAP)
    parser.add_argument(
        "--write-receipt",
        metavar="ABS_JSON",
        help="atomically create this new receipt; existing files are refused",
    )
    parser.add_argument(
        "--allow-known-documentation-exception",
        action="store_true",
        help=(
            "allow only the known E2d6 Experimental_Setup.md provenance gap; all "
            "numeric, code, model and data identities remain fail-closed"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        core_root = _absolute_root(args.core_root, "--core-root")
        e2c_root = _absolute_root(args.e2c_root, "--e2c-root")
        receipt = verify(
            core_root,
            e2c_root,
            args.cap,
            allow_known_documentation_exception=args.allow_known_documentation_exception,
        )
        if args.write_receipt:
            destination = _receipt_destination(
                args.write_receipt, forbidden_roots=(core_root, e2c_root)
            )
            _write_json_once_atomic(destination, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    except (VerificationError, FileExistsError, FileNotFoundError, OSError) as exc:
        print(f"FAIL — {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
