#!/usr/bin/env python3
"""Append-only refresh of corrected Aim-2 exploratory panels E2d2/E2d4/E2d6.

The expensive cap-8192 refits and score tables are immutable inputs.  This
runner takes their root explicitly, computes all three downstream panels in a
private staging directory, verifies their numerical/provenance contracts, and
only then claims and publishes a wholly new output root with exclusive-create
writes.  It never writes into the frozen input lineage.

The refresh closes three post-audit gaps:

* E2d2 records patient-level slide bytes as the sum over metastatic slides.
* E2d4 embeds machine-readable overlap/positivity claim guardrails.
* E2d6 binds exact snapshots of the current protocol and historical design
  archive, eliminating reliance on mutable documentation bytes.

Example (the output root must not already exist)::

    uv run python tools/aim2_exploratory_refresh.py run \
      --input-root /abs/path/to/aim2_cap8192_v4_20260819 \
      --output-root /abs/path/to/aim2_cap8192_e2d_refresh_v1_20260819

    uv run python tools/aim2_exploratory_refresh.py verify \
      --output-root /abs/path/to/aim2_cap8192_e2d_refresh_v1_20260819
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from argparse import Namespace
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import aim2_loco_transport  # noqa: E402
import aim2_peritoneal_stability  # noqa: E402
import aim2_surgen_subcohort_gap  # noqa: E402
import aim2_rih_acquisition_regime  # noqa: E402
from oceanpath.aim1 import lineage, paths  # noqa: E402

CAP = 8192
FULL_BOOTSTRAP_REPS = 10_000
EXPECTED_RESULTS = (
    "eval/e2d2_peritoneal_audit_cap8192.json",
    "eval/e2d4_surgen_gap_cap8192.json",
    "eval/e2d6_rih_acquisition_regime_cap8192.json",
    "e2d6/tables/rih_regime_patients_cap8192.parquet",
    "e2d6/tables/rih_regime_patients_cap8192.receipt.json",
)
RECEIPT_NAME = "refresh_receipt.json"
VERIFICATION_NAME = "verification_receipt.json"
_LINEAGE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RefreshError(RuntimeError):
    """The append-only refresh or one of its verification contracts failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RefreshError(message)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file(), f"Expected file input: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RefreshError(f"Cannot read JSON object {path}: {exc}") from exc
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _absolute_existing_root(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    _require(path.is_absolute(), f"{label} must be an explicit absolute path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RefreshError(f"{label} does not resolve: {path}: {exc}") from exc
    _require(resolved.is_dir(), f"{label} is not a directory: {resolved}")
    return resolved


def _new_output_root(value: str | Path, input_root: Path) -> Path:
    path = Path(value).expanduser()
    _require(path.is_absolute(), f"output root must be an explicit absolute path: {path}")
    _require(not path.exists() and not path.is_symlink(), f"output root already exists: {path}")
    parent = path.parent.resolve(strict=True)
    output = parent / path.name
    _require(_LINEAGE_SLUG.fullmatch(output.name) is not None, "invalid output lineage name")
    _require(output != input_root, "input and output roots must differ")
    _require(input_root not in output.parents, "output root may not be inside input root")
    _require(output not in input_root.parents, "input root may not be inside output root")
    return output


def _source_lineage(input_root: Path) -> str:
    started = _read_json(input_root / "lineage_start.json")
    name = started.get("lineage")
    _require(isinstance(name, str) and bool(name), "input lineage_start lacks lineage")
    _require(_LINEAGE_SLUG.fullmatch(name) is not None, "input lineage name is invalid")
    _require(input_root.name == name, "input root name and lineage_start lineage differ")
    return name


def _required_input_paths(input_root: Path, cap: int) -> dict[str, Path]:
    _require(cap == CAP, f"This refresh is locked to cap={CAP}")
    required: dict[str, Path] = {
        "lineage_start": input_root / "lineage_start.json",
        "upstream_e2b": input_root / "eval" / f"e2b_metastatic_cap{cap}.json",
        "upstream_e2d1": input_root
        / "eval"
        / f"e2d1_metastatic_sites_cap{cap}.json",
        "development_manifest": paths.DEV_MANIFEST,
        "label_source": paths.LABEL_SOURCE,
        "ready_inventory": aim2_rih_acquisition_regime.INVENTORY_PATH,
        "rih_repair_state": aim2_rih_acquisition_regime.RIH_REPAIR_STATE_PATH,
        "current_protocol": aim2_rih_acquisition_regime.PROTOCOL_PATH,
        "historical_design_archive": aim2_rih_acquisition_regime.DESIGN_ARCHIVE_PATH,
    }
    roles_by_target = {
        "RIH": ("primary", "metastatic"),
        "SurGen": ("primary", "metastatic"),
    }
    for target, roles in roles_by_target.items():
        target_key = target.lower()
        for role in roles:
            required[f"manifest/{target_key}/{role}"] = aim2_loco_transport.target_manifest(target, role)
        for seed in aim2_loco_transport.SEEDS:
            run = (
                input_root
                / "e2a"
                / "train"
                / f"pb_cap{cap}"
                / target_key
                / f"seed{seed}"
            )
            required[f"fit/{target_key}/seed{seed}"] = run / "fit_summary.json"
            required[f"checkpoint/{target_key}/seed{seed}"] = (
                run / "final" / "refit" / "model.ckpt"
            )
            for role in roles:
                stem = f"pb_cap{cap}_{target_key}_seed{seed}_{role}"
                score = input_root / "e2a" / "scores" / f"{stem}.parquet"
                required[f"score/{target_key}/seed{seed}/{role}"] = score
                required[f"score_receipt/{target_key}/seed{seed}/{role}"] = (
                    score.with_suffix(".receipt.json")
                )
    return required


def _inventory_named(files: Mapping[str, Path]) -> list[dict[str, Any]]:
    return [
        {"label": label, **_identity(path)}
        for label, path in sorted(files.items())
    ]


def _source_files() -> list[Path]:
    files = [
        *(REPO / name for name in ("aim2_loco_transport.py", "aim2_metastatic_transport.py", "aim2_metastatic_site.py", "aim2_peritoneal_stability.py", "aim2_surgen_subcohort_gap.py", "aim2_rih_acquisition_regime.py")),
        Path(__file__).resolve(),
        REPO / "pyproject.toml",
        REPO / "uv.lock",
    ]
    files.extend((REPO / "src" / "oceanpath").rglob("*.py"))
    unique = sorted(set(path.resolve() for path in files))
    missing = [path for path in unique if not path.is_file()]
    _require(not missing, f"Missing source files for snapshot: {missing}")
    return unique


def _relative_inventory(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative in excluded:
            continue
        rows.append(
            {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _snapshot_files(stage_root: Path, sources: list[Path]) -> list[dict[str, Any]]:
    snapshot = stage_root / "source_snapshot"
    for source in sources:
        destination = snapshot / source.relative_to(REPO)
        lineage.write_bytes_once(destination, source.read_bytes())
    return _relative_inventory(snapshot)


def _snapshot_context(stage_root: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    context_root = stage_root / "context_snapshot" / "reports"
    protocol = context_root / "Experimental_Setup.md"
    archive = context_root / "Experimental_Setup_DESIGN_ARCHIVE.md"
    lineage.write_bytes_once(protocol, aim2_rih_acquisition_regime.PROTOCOL_PATH.read_bytes())
    lineage.write_bytes_once(archive, aim2_rih_acquisition_regime.DESIGN_ARCHIVE_PATH.read_bytes())
    return protocol, archive, _relative_inventory(stage_root / "context_snapshot")


@contextlib.contextmanager
def _explicit_analysis_context(
    *,
    input_root: Path,
    stage_root: Path,
    final_root: Path,
    source_lineage: str,
    output_lineage: str,
    protocol_snapshot: Path,
    archive_snapshot: Path,
) -> Iterator[None]:
    """Route model reads to v4 and all newly written artifacts to staging."""

    environment_key = lineage.AIM2_LINEAGE_ENV
    previous_environment = os.environ.get(environment_key)
    original_aim2_root = lineage.aim2_root
    original_identity = lineage.artifact_identity
    original_e2a_root = aim2_loco_transport.e2a_root
    original_completed_refit = aim2_loco_transport._completed_refit
    original_protocol = aim2_rih_acquisition_regime.PROTOCOL_PATH
    original_archive = aim2_rih_acquisition_regime.DESIGN_ARCHIVE_PATH
    stage_resolved = stage_root.resolve()

    def staged_root(*, required: bool = True) -> Path:  # noqa: ARG001
        return stage_root

    def frozen_e2a_root() -> Path:
        return input_root / "e2a"

    def completed_from_source(target: str, seed: int, cap: int) -> dict:
        active = os.environ.get(environment_key)
        os.environ[environment_key] = source_lineage
        try:
            return original_completed_refit(target, seed, cap)
        finally:
            if active is None:
                os.environ.pop(environment_key, None)
            else:
                os.environ[environment_key] = active

    def canonical_identity(path: Path) -> dict[str, Any]:
        result = original_identity(path)
        resolved = Path(result["path"])
        try:
            relative = resolved.relative_to(stage_resolved)
        except ValueError:
            return result
        result["path"] = str(final_root / relative)
        return result

    os.environ[environment_key] = output_lineage
    lineage.aim2_root = staged_root
    lineage.artifact_identity = canonical_identity
    aim2_loco_transport.e2a_root = frozen_e2a_root
    aim2_loco_transport._completed_refit = completed_from_source
    aim2_rih_acquisition_regime.PROTOCOL_PATH = protocol_snapshot
    aim2_rih_acquisition_regime.DESIGN_ARCHIVE_PATH = archive_snapshot
    try:
        yield
    finally:
        lineage.aim2_root = original_aim2_root
        lineage.artifact_identity = original_identity
        aim2_loco_transport.e2a_root = original_e2a_root
        aim2_loco_transport._completed_refit = original_completed_refit
        aim2_rih_acquisition_regime.PROTOCOL_PATH = original_protocol
        aim2_rih_acquisition_regime.DESIGN_ARCHIVE_PATH = original_archive
        if previous_environment is None:
            os.environ.pop(environment_key, None)
        else:
            os.environ[environment_key] = previous_environment


def _run_components(
    *,
    input_root: Path,
    stage_root: Path,
    final_root: Path,
    source_lineage: str,
    n_bootstrap: int,
    bootstrap_seed: int,
    protocol_snapshot: Path,
    archive_snapshot: Path,
) -> None:
    with _explicit_analysis_context(
        input_root=input_root,
        stage_root=stage_root,
        final_root=final_root,
        source_lineage=source_lineage,
        output_lineage=final_root.name,
        protocol_snapshot=protocol_snapshot,
        archive_snapshot=archive_snapshot,
    ):
        shared = {
            "cap": CAP,
            "input_eval_root": input_root / "eval",
            "output_eval_root": stage_root / "eval",
        }
        aim2_peritoneal_stability.cmd_report(Namespace(**shared))
        aim2_surgen_subcohort_gap.cmd_report(
            Namespace(
                **shared,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
            )
        )
        aim2_rih_acquisition_regime.cmd_report(
            Namespace(
                cap=CAP,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
            )
        )


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RefreshError(f"{label} is not numeric: {value!r}") from exc
    _require(math.isfinite(number), f"{label} is not finite: {number}")
    return number


def _interval(value: object, label: str) -> tuple[float, float]:
    _require(isinstance(value, list) and len(value) == 2, f"{label} is not a 2-value CI")
    low = _finite(value[0], f"{label}[0]")
    high = _finite(value[1], f"{label}[1]")
    _require(low <= high, f"{label} is reversed")
    return low, high


def _actual_recorded_path(path: Path, actual_root: Path, recorded_root: Path) -> Path:
    try:
        relative = path.relative_to(recorded_root)
    except ValueError:
        return path
    return actual_root / relative


def _validate_identity(
    recorded: Mapping[str, Any],
    *,
    actual_root: Path,
    recorded_root: Path,
    label: str,
) -> None:
    _require({"path", "size_bytes", "sha256"}.issubset(recorded), f"{label}: bad identity")
    raw = Path(str(recorded["path"]))
    _require(raw.is_absolute(), f"{label}: identity path is not absolute")
    actual = _actual_recorded_path(raw, actual_root, recorded_root).resolve(strict=True)
    _require(actual.is_file(), f"{label}: identity target is not a file: {actual}")
    _require(actual.stat().st_size == int(recorded["size_bytes"]), f"{label}: size mismatch")
    _require(_sha256_file(actual) == recorded["sha256"], f"{label}: SHA256 mismatch")


def _validate_recorded_identities(
    value: object,
    *,
    actual_root: Path,
    recorded_root: Path,
    label: str,
) -> int:
    if isinstance(value, dict):
        if {"path", "size_bytes", "sha256"}.issubset(value):
            _validate_identity(
                value,
                actual_root=actual_root,
                recorded_root=recorded_root,
                label=label,
            )
            return 1
        return sum(
            _validate_recorded_identities(
                child,
                actual_root=actual_root,
                recorded_root=recorded_root,
                label=f"{label}.{key}",
            )
            for key, child in value.items()
        )
    if isinstance(value, list):
        return sum(
            _validate_recorded_identities(
                child,
                actual_root=actual_root,
                recorded_root=recorded_root,
                label=f"{label}[{index}]",
            )
            for index, child in enumerate(value)
        )
    return 0


def _verify_e2d2(report: dict[str, Any]) -> dict[str, Any]:
    _require(report.get("schema_version") == 2, "E2d2 schema version is not 2")
    _require(report.get("cap") == CAP, "E2d2 cap mismatch")
    _require(report.get("n") == 14 and report.get("n_mut") == 4, "E2d2 population changed")
    _require(report.get("metadata_aggregation", {}).get("slide_size_bytes") == (
        "sum across metastatic slides"
    ), "E2d2 does not declare summed slide-byte aggregation")
    patients = report.get("patients")
    _require(isinstance(patients, list) and len(patients) == report["n"], "E2d2 patient rows mismatch")
    source_identity = report.get("inputs", {}).get("label_source")
    _require(isinstance(source_identity, dict), "E2d2 lacks label-source identity")
    source = pd.read_csv(Path(source_identity["path"]), low_memory=False)
    source = source[source["specimen_role"].eq("metastatic")].copy()
    seen: set[str] = set()
    for row in patients:
        patient = str(row.get("patient"))
        _require(patient not in seen, f"E2d2 duplicates patient {patient}")
        seen.add(patient)
        block = source[source["patient_uid"].astype(str).eq(patient)]
        _require(not block.empty, f"E2d2 metadata source lacks patient {patient}")
        _require(row.get("metadata_slide_count") == len(block), f"E2d2 slide count mismatch for {patient}")
        slide_bytes = pd.to_numeric(block["slide_size_bytes"], errors="coerce")
        expected_bytes = float(slide_bytes.sum(min_count=1))
        _require(
            math.isclose(float(row["slide_size_bytes"]), expected_bytes, rel_tol=0.0, abs_tol=0.5),
            f"E2d2 summed slide bytes mismatch for {patient}",
        )
        mpp = pd.to_numeric(block["mpp"], errors="coerce")
        _require(
            math.isclose(float(row["mpp"]), float(mpp.median()), rel_tol=0.0, abs_tol=1e-12),
            f"E2d2 median MPP mismatch for {patient}",
        )
    auc = _finite(report.get("ensemble_auroc"), "E2d2 ensemble AUROC")
    _require(0.0 <= auc <= 1.0, "E2d2 ensemble AUROC outside [0,1]")
    _require(set(map(str, report.get("per_seed_auroc", {}))) == {"42", "43", "44"}, "E2d2 seed set mismatch")
    return {"patients_checked": len(patients), "summed_slide_bytes_verified": True}


def _verify_e2d4(report: dict[str, Any], n_bootstrap: int) -> dict[str, Any]:
    _require(report.get("schema_version") == 2, "E2d4 schema version is not 2")
    _require(report.get("cap") == CAP, "E2d4 cap mismatch")
    checked = 0
    for name, _covariates in aim2_surgen_subcohort_gap.standardization_specs():
        key = f"ipw_{name}"
        block = report.get(key)
        _require(isinstance(block, dict), f"E2d4 lacks {key}")
        _require(block.get("n_bootstrap") == n_bootstrap, f"{key} bootstrap count mismatch")
        _interval(block.get("delta_ci"), f"{key}.delta_ci")
        target = _finite(block.get("auroc_SR386"), f"{key}.auroc_SR386")
        control = _finite(
            block.get("auroc_SR1482_reweighted"),
            f"{key}.auroc_SR1482_reweighted",
        )
        delta = _finite(block.get("delta"), f"{key}.delta")
        _require(math.isclose(delta, target - control, abs_tol=1e-12), f"{key} delta mismatch")
        assessment = block.get("overlap_assessment")
        _require(isinstance(assessment, dict), f"{key} lacks overlap assessment")
        observed = assessment.get("observed", {})
        thresholds = assessment.get("thresholds", {})
        expected_checks = {
            "clipping": observed.get("control_fraction_clipped")
            <= thresholds.get("max_control_fraction_clipped"),
            "common_support": observed.get(
                "control_fraction_outside_empirical_common_support"
            )
            <= thresholds.get("max_control_fraction_outside_empirical_common_support"),
            "effective_sample_size": observed.get("control_ess_fraction")
            >= thresholds.get("min_control_ess_fraction"),
            "maximum_weight": observed.get("control_max_normalized_weight")
            <= thresholds.get("max_control_normalized_weight"),
        }
        _require(assessment.get("checks") == expected_checks, f"{key} overlap checks mismatch")
        adequate = all(expected_checks.values())
        _require(assessment.get("estimable_for_inference") is adequate, f"{key} estimability mismatch")
        _require(
            set(assessment.get("failed_checks", []))
            == {check for check, passed in expected_checks.items() if not passed},
            f"{key} failed overlap checks mismatch",
        )
        checked += 1
    return {"standardizations_checked": checked, "overlap_guardrails_verified": True}


def _verify_e2d6(
    report: dict[str, Any],
    *,
    actual_root: Path,
    recorded_root: Path,
    n_bootstrap: int,
) -> dict[str, Any]:
    _require(report.get("schema_version") == 1, "E2d6 schema version mismatch")
    _require(report.get("cap") == CAP, "E2d6 cap mismatch")
    inference = report.get("inference", {})
    _require(inference.get("n_bootstrap") == n_bootstrap, "E2d6 bootstrap count mismatch")
    aligned = report.get("e2b_aligned_primary_to_metastatic", {})
    interaction = aligned.get("four_arm_interaction", {})
    _require(interaction.get("n_bootstrap") == n_bootstrap, "E2d6 interaction bootstrap mismatch")
    _require(interaction.get("four_arms_pairwise_patient_disjoint") is True, "E2d6 arms overlap")
    _interval(interaction.get("interaction_delta_auroc_ci"), "E2d6 interaction CI")
    per_regime = interaction.get("per_regime_delta_auroc", {})
    expected = _finite(per_regime.get(aim2_rih_acquisition_regime.REGIME_REPAIRED), "E2d6 repaired delta") - _finite(
        per_regime.get(aim2_rih_acquisition_regime.REGIME_APERIO), "E2d6 native delta"
    )
    observed = _finite(interaction.get("interaction_delta_auroc"), "E2d6 interaction")
    _require(math.isclose(observed, expected, abs_tol=1e-12), "E2d6 interaction algebra mismatch")
    guardrails = report.get("guardrails", {})
    for key in (
        "confirmatory_claim_allowed",
        "scanner_claim_allowed",
        "causal_specimen_role_claim_allowed",
        "equivalence_or_invariance_claim_allowed",
        "versa_inferential_claim_allowed",
    ):
        _require(guardrails.get(key) is False, f"E2d6 guardrail {key} is not closed")
    sources = report.get("input_artifacts", {}).get("mapping_sources", {})
    for key, filename in (
        ("current_protocol_context", "Experimental_Setup.md"),
        ("historical_design_archive", "Experimental_Setup_DESIGN_ARCHIVE.md"),
    ):
        identity = sources.get(key, {})
        expected_path = recorded_root / "context_snapshot" / "reports" / filename
        _require(Path(str(identity.get("path"))) == expected_path, f"E2d6 {key} is not snapshotted")
    table_path = actual_root / "e2d6" / "tables" / "rih_regime_patients_cap8192.parquet"
    table = pd.read_parquet(table_path)
    _require(len(table) == 238, "E2d6 patient-table row count changed")
    _require(not table.duplicated(["role", "patient_id"]).any(), "E2d6 patient table duplicates units")
    _require(np.isfinite(table[["mean_logit", "prob_raw"]].to_numpy(dtype=float)).all(), "E2d6 patient table is non-finite")
    return {
        "patient_rows_checked": int(len(table)),
        "four_arm_interaction_verified": True,
        "documentation_snapshots_verified": True,
    }


def _verify_generated(
    actual_root: Path,
    *,
    recorded_root: Path,
    output_lineage: str,
    n_bootstrap: int,
) -> dict[str, Any]:
    reports = {
        "e2d2": _read_json(actual_root / EXPECTED_RESULTS[0]),
        "e2d4": _read_json(actual_root / EXPECTED_RESULTS[1]),
        "e2d6": _read_json(actual_root / EXPECTED_RESULTS[2]),
    }
    for name, report in reports.items():
        _require(report.get("lineage") == output_lineage, f"{name} lineage mismatch")
    identities = sum(
        _validate_recorded_identities(
            report,
            actual_root=actual_root,
            recorded_root=recorded_root,
            label=name,
        )
        for name, report in reports.items()
    )
    table_receipt = _read_json(actual_root / EXPECTED_RESULTS[4])
    identities += _validate_recorded_identities(
        table_receipt,
        actual_root=actual_root,
        recorded_root=recorded_root,
        label="e2d6_table_receipt",
    )
    return {
        "recorded_identities_checked": identities,
        "e2d2": _verify_e2d2(reports["e2d2"]),
        "e2d4": _verify_e2d4(reports["e2d4"], n_bootstrap),
        "e2d6": _verify_e2d6(
            reports["e2d6"],
            actual_root=actual_root,
            recorded_root=recorded_root,
            n_bootstrap=n_bootstrap,
        ),
    }


def _publish_once(stage_root: Path, output_root: Path) -> None:
    _require(not output_root.exists() and not output_root.is_symlink(), f"output exists: {output_root}")
    try:
        output_root.mkdir()
    except FileExistsError as exc:
        raise RefreshError(f"Concurrent writer claimed output root: {output_root}") from exc
    for source in sorted(path for path in stage_root.rglob("*") if path.is_file()):
        destination = output_root / source.relative_to(stage_root)
        lineage.write_bytes_once(destination, source.read_bytes())


def _verify_inventory(root: Path, recorded: list[dict[str, Any]]) -> None:
    observed = _relative_inventory(
        root,
        exclude={RECEIPT_NAME, VERIFICATION_NAME},
    )
    _require(observed == recorded, "Published artifact inventory differs from refresh receipt")


def verify_refresh(output_root: Path, *, check_live_inputs: bool = True) -> dict[str, Any]:
    root = _absolute_existing_root(output_root, "output root")
    receipt = _read_json(root / RECEIPT_NAME)
    _require(receipt.get("status") == "completed", "refresh receipt is not completed")
    _require(receipt.get("output_root") == str(root), "refresh receipt root mismatch")
    _require(receipt.get("lineage") == root.name, "refresh receipt lineage mismatch")
    _require(receipt.get("cap") == CAP, "refresh receipt cap mismatch")
    n_bootstrap = int(receipt.get("n_bootstrap", 0))
    _require(n_bootstrap > 0, "refresh receipt has invalid bootstrap count")
    _verify_inventory(root, receipt.get("artifact_inventory", []))
    if check_live_inputs:
        for row in receipt.get("input_inventory", []):
            _validate_identity(row, actual_root=root, recorded_root=root, label=row.get("label", "input"))
        for row in receipt.get("live_source_inventory", []):
            _validate_identity(row, actual_root=root, recorded_root=root, label="live_source")
    semantic = _verify_generated(
        root,
        recorded_root=root,
        output_lineage=root.name,
        n_bootstrap=n_bootstrap,
    )
    verification_path = root / VERIFICATION_NAME
    if verification_path.is_file():
        verification = _read_json(verification_path)
        _require(verification.get("status") == "pass", "verification receipt is not pass")
        _require(verification.get("refresh_receipt") == _identity(root / RECEIPT_NAME), "verification receipt points to different refresh receipt")
    return {
        "status": "PASS",
        "output_root": str(root),
        "analysis_grade": receipt.get("analysis_grade"),
        "n_bootstrap": n_bootstrap,
        **semantic,
    }


def cmd_run(args: argparse.Namespace) -> None:
    input_root = _absolute_existing_root(args.input_root, "input root")
    output_root = _new_output_root(args.output_root, input_root)
    _require(args.n_bootstrap > 0, "n-bootstrap must be positive")
    source_lineage = _source_lineage(input_root)
    required_inputs = _required_input_paths(input_root, CAP)
    input_inventory_before = _inventory_named(required_inputs)
    source_files = _source_files()
    live_source_before = _inventory_named(
        {str(path.relative_to(REPO)): path for path in source_files}
    )

    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.stage-", dir="/tmp") as temporary:
        stage_root = Path(temporary)
        source_snapshot = _snapshot_files(stage_root, source_files)
        protocol_snapshot, archive_snapshot, context_snapshot = _snapshot_context(stage_root)
        run_request = {
            "schema_version": 1,
            "created_utc": _utc_now(),
            "input_root": str(input_root),
            "input_lineage": source_lineage,
            "output_root": str(output_root),
            "lineage": output_root.name,
            "cap": CAP,
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "publication": "stage, verify, then exclusive-create into a wholly new root",
        }
        lineage.write_json_once(stage_root / "run_request.json", run_request)
        _run_components(
            input_root=input_root,
            stage_root=stage_root,
            final_root=output_root,
            source_lineage=source_lineage,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
            protocol_snapshot=protocol_snapshot,
            archive_snapshot=archive_snapshot,
        )
        input_inventory_after = _inventory_named(required_inputs)
        live_source_after = _inventory_named(
            {str(path.relative_to(REPO)): path for path in source_files}
        )
        _require(input_inventory_after == input_inventory_before, "material inputs changed during refresh")
        _require(live_source_after == live_source_before, "source code changed during refresh")
        for relative in EXPECTED_RESULTS:
            _require((stage_root / relative).is_file(), f"Missing staged result: {relative}")
        staged_verification = _verify_generated(
            stage_root,
            recorded_root=output_root,
            output_lineage=output_root.name,
            n_bootstrap=args.n_bootstrap,
        )
        receipt = {
            "schema_version": 1,
            "status": "completed",
            "completed_utc": _utc_now(),
            "lineage": output_root.name,
            "output_root": str(output_root),
            "input_root": str(input_root),
            "input_lineage": source_lineage,
            "cap": CAP,
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "analysis_grade": (
                "full_10000_bootstrap"
                if args.n_bootstrap == FULL_BOOTSTRAP_REPS
                else "noncanonical_smoke_test"
            ),
            "input_inventory": input_inventory_before,
            "live_source_inventory": live_source_before,
            "source_snapshot_inventory": source_snapshot,
            "context_snapshot_inventory": context_snapshot,
            "staged_semantic_verification": staged_verification,
            "artifact_inventory": _relative_inventory(stage_root),
            "immutability": {
                "frozen_input_unchanged": True,
                "source_unchanged_during_run": True,
                "exclusive_output_writes": True,
            },
        }
        lineage.write_json_once(stage_root / RECEIPT_NAME, receipt)
        _publish_once(stage_root, output_root)

    verification = verify_refresh(output_root)
    lineage.write_json_once(
        output_root / VERIFICATION_NAME,
        {
            "schema_version": 1,
            "status": "pass",
            "verified_utc": _utc_now(),
            "refresh_receipt": _identity(output_root / RECEIPT_NAME),
            "verification": verification,
        },
    )
    final = verify_refresh(output_root)
    print(json.dumps(final, indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    print(json.dumps(verify_refresh(Path(args.output_root)), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="compute, verify and exclusively publish")
    run.add_argument("--input-root", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--n-bootstrap", type=int, default=FULL_BOOTSTRAP_REPS)
    run.add_argument("--bootstrap-seed", type=int, default=paths.BOOTSTRAP_SEED)
    run.set_defaults(func=cmd_run)
    verify = commands.add_parser("verify", help="read-only verification of a published root")
    verify.add_argument("--output-root", required=True)
    verify.set_defaults(func=cmd_verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
