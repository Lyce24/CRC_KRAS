#!/usr/bin/env python3
"""Read-only consolidated verifier for corrected cap-8192 Aim 2 and Aim 3.

Five explicit immutable roots are required:

* the Aim-2 v4 core lineage;
* the native-logit E2c residual-offset recovery;
* the append-only E2d2/E2d4/E2d6 refresh;
* the corrected Aim-3 molecular-resolution lineage;
* the three-draw Aim-3 repeated-control robustness lineage.

The verifier delegates each component's full lineage audit to its frozen
component verifier, then enforces cross-root binding, canonical-result
supersession, full (not smoke) bootstrap budgets, native-logit-only scoring,
and headline verdict algebra.  It never modifies any experiment root.  An
optional consolidated receipt is created atomically and exclusively at an
absolute path outside all five roots.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import aim3_fixed_control_analysis  # noqa: E402
import aim3_repeated_control_campaign  # noqa: E402
from tools import (
    aim2_exploratory_refresh,
    verify_aim2_corrected,
)  # noqa: E402

CAP = 8192
FULL_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260817
REPEATED_BOOTSTRAP = 20_000
REPEATED_BOOTSTRAP_SEED = 20260826
REPEATED_DRAW_SEEDS = (20260823, 20260824, 20260825)
REPEATED_MODEL_SEEDS = (42, 43, 44)
REPEATED_FINE_TASKS = ("codon", "g12d_broad", "allele1", "allele2", "g12c")
REPEATED_CONTROLS = {
    "codon": "ctrl_codon",
    "g12d_broad": "ctrl_g12d_broad",
    "allele1": "ctrl_allele1",
    "allele2": "ctrl_allele2",
    "g12c": "ctrl_g12c",
}
REPEATED_CHAINS = 45
REPEATED_CONTROL_FOLDS = 225
REPEATED_FINE_FOLDS = 75
REPEATED_SCHEMA_VERSION = 2


class ConsolidatedVerificationError(RuntimeError):
    """A lineage, numerical, or claim contract failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConsolidatedVerificationError(message)


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file(), f"Expected file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConsolidatedVerificationError(f"Cannot read JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _absolute_root(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    _require(path.is_absolute(), f"{label} must be an explicit absolute path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ConsolidatedVerificationError(f"{label} does not resolve: {path}") from error
    _require(resolved.is_dir(), f"{label} is not a directory: {resolved}")
    return resolved


def _validate_distinct_roots(roots: dict[str, Path]) -> None:
    _require(len(set(roots.values())) == len(roots), "All roots must be distinct")
    for left_name, left in roots.items():
        for right_name, right in roots.items():
            if left_name >= right_name:
                continue
            _require(
                left not in right.parents and right not in left.parents,
                f"Roots must not be nested: {left_name}={left}, {right_name}={right}",
            )


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ConsolidatedVerificationError(f"{label} is not numeric: {value!r}") from error
    _require(math.isfinite(number), f"{label} is non-finite")
    return number


def _ci(value: object, label: str) -> tuple[float, float]:
    _require(isinstance(value, list) and len(value) == 2, f"{label} is not a two-value CI")
    low, high = _finite(value[0], f"{label}[0]"), _finite(value[1], f"{label}[1]")
    _require(low <= high, f"{label} is reversed")
    return low, high


def _contrast_state(value: object, *, null: float = 0.0) -> str:
    low, high = _ci(value, "contrast CI")
    if low > null:
        return "positive"
    if high < null:
        return "negative"
    return "not_established"


def _walk_forbidden_roundtrip(value: object, *, location: str = "root") -> list[str]:
    """Find structural markers of a probability-to-logit analysis path.

    Warnings that merely discuss the historical problem are not rejected.  We
    reject machine fields or affirmative method values that designate a
    reconstructed/round-tripped predictor as an active input.
    """
    forbidden_key_fragments = (
        "probability_to_logit",
        "probability_roundtrip",
        "probability_derived_logit",
        "reconstructed_logit",
        "logit_from_probability",
    )
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_").replace(" ", "_")
            if any(fragment in normalized for fragment in forbidden_key_fragments):
                findings.append(f"{location}.{key}")
            findings.extend(_walk_forbidden_roundtrip(child, location=f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_walk_forbidden_roundtrip(child, location=f"{location}[{index}]"))
    elif isinstance(value, str):
        normalized = " ".join(value.lower().replace("-", " ").split())
        affirmative = (
            "scores reconstructed from probability",
            "logits reconstructed from probability",
            "probability derived logits used",
            "probability to logit roundtrip used",
        )
        if any(phrase in normalized for phrase in affirmative):
            findings.append(location)
    return findings


def _verify_e2c_native_contract(e2c_root: Path, core_root: Path) -> dict[str, Any]:
    result_path = e2c_root / "analysis" / f"e2c_native_logit_offset_cap{CAP}.json"
    report = _read_json(result_path)
    _require(report.get("component") == "e2c_native_logit_residual_offset", "Wrong E2c component")
    _require(report.get("cap") == CAP, "E2c cap mismatch")
    _require(report.get("reps") == 100, "E2c must contain 100 procedure repetitions")
    _require(report.get("n_bootstrap") == FULL_BOOTSTRAP, "E2c is not the full 10k analysis")
    _require(Path(str(report.get("input_lineage_root", ""))).resolve(strict=True) == core_root,
             "E2c is not bound to the explicit Aim2 core root")
    design = report.get("design", {})
    _require(design.get("model") == "eta_native + H @ delta_w + delta_b", "E2c model is not native-offset")
    _require("native logits" in str(design.get("frozen", "")), "E2c does not freeze native logits")
    _require(design.get("n_bootstrap") == FULL_BOOTSTRAP, "E2c design bootstrap mismatch")
    _require(design.get("bootstrap_seed") == BOOTSTRAP_SEED, "E2c bootstrap seed mismatch")
    _require(not _walk_forbidden_roundtrip(report), "E2c contains an active probability-roundtrip marker")

    contrasts: dict[str, str] = {}
    expected_arms = {f"{arm}_k{k}" for arm in ("S1", "S2") for k in (2, 4, 8)}
    for cohort, block in report.get("cohorts", {}).items():
        _require(set(block.get("arms", {})) == expected_arms, f"E2c {cohort} arm inventory mismatch")
        _require(block.get("reps") == 100 and block.get("folds") == 5, f"E2c {cohort} procedure incomplete")
        s0 = _finite(block.get("S0", {}).get("auroc"), f"E2c {cohort} S0 AUROC")
        _require(0.0 <= s0 <= 1.0, f"E2c {cohort} S0 AUROC outside [0,1]")
        arm_points: dict[str, float] = {}
        for name, arm in block["arms"].items():
            performance = arm.get("performance", {})
            _require(performance.get("n_procedure_draws") == 100, f"E2c {cohort}/{name} draws incomplete")
            point = _finite(performance.get("metrics", {}).get("auroc"), f"E2c {cohort}/{name} AUROC")
            draws = performance.get("per_draw_metrics", [])
            _require(len(draws) == 100, f"E2c {cohort}/{name} per-draw metrics incomplete")
            mean_draw = sum(_finite(row.get("auroc"), f"E2c {cohort}/{name} draw") for row in draws) / 100
            _require(math.isclose(point, mean_draw, rel_tol=0.0, abs_tol=1e-12),
                     f"E2c {cohort}/{name} headline is not mean procedure AUROC")
            _ci(performance.get("expected_auroc_ci"), f"E2c {cohort}/{name} CI")
            arm_points[name] = point
        for name, contrast in block.get("contrasts", {}).items():
            delta = _finite(contrast.get("delta"), f"E2c {cohort}/{name} delta")
            if name.startswith("S2_minus_S1_k"):
                k = name.rsplit("k", 1)[1]
                expected = arm_points[f"S2_k{k}"] - arm_points[f"S1_k{k}"]
            elif name.endswith("_minus_S0"):
                expected = arm_points[name.removesuffix("_minus_S0")] - s0
            else:
                raise ConsolidatedVerificationError(f"Unexpected E2c contrast: {cohort}/{name}")
            _require(math.isclose(delta, expected, rel_tol=0.0, abs_tol=1e-12),
                     f"E2c {cohort}/{name} point contrast algebra mismatch")
            contrasts[f"{cohort}/{name}"] = _contrast_state(contrast.get("ci"))
    _require(set(report.get("cohorts", {})) == {"RIH", "SurGen"}, "E2c cohort inventory mismatch")
    return {
        "status": "PASS",
        "result": _identity(result_path),
        "native_logit_residual_model": True,
        "probability_roundtrip_rejected": True,
        "headline_contrast_states": contrasts,
    }


def _expected_aim3_verdict(fine: dict[str, Any], control: dict[str, Any], delta: dict[str, Any]) -> str:
    return aim3_fixed_control_analysis.verdict_for(fine, control, delta)


def _verify_aim3_headlines(aim3_root: Path) -> dict[str, Any]:
    result_path = aim3_root / "analysis" / "aim3_corrected.json"
    report = _read_json(result_path)
    _require(report.get("component") == "aim3_corrected_cap8192", "Wrong Aim3 component")
    protocol = report.get("protocol", {})
    _require(protocol.get("cap") == CAP, "Aim3 cap mismatch")
    _require(protocol.get("seeds") == [42, 43, 44], "Aim3 seed inventory mismatch")
    _require(protocol.get("n_bootstrap") == FULL_BOOTSTRAP, "Aim3 is not the full 10k analysis")
    _require(protocol.get("bootstrap_seed") == BOOTSTRAP_SEED, "Aim3 bootstrap seed mismatch")
    _require(
        protocol.get("patient_aggregation")
        == "mean native slide logit, then mean across exactly three seeds",
        "Aim3 does not declare native-logit patient/seed aggregation",
    )
    _require("partially paired" in str(protocol.get("pair_contrast", "")),
             "Aim3 primary pair contrast is not partially paired")
    _require("cohort x label" in str(protocol.get("task_ci", "")),
             "Aim3 standalone intervals are not cohort x label stratified")
    fwer = protocol.get("familywise_sensitivity", {})
    _require(fwer.get("rungs") == 5, "Aim3 familywise rung count mismatch")
    _require(math.isclose(float(fwer.get("bonferroni_one_sided_alpha_per_rung", -1)), 0.01),
             "Aim3 familywise alpha is not 0.01 per rung")
    _require(not _walk_forbidden_roundtrip(report), "Aim3 contains an active probability-roundtrip marker")

    expected_tasks = set(aim3_fixed_control_analysis.ALL_TASKS)
    _require(set(report.get("tasks", {})) == expected_tasks, "Aim3 task inventory mismatch")
    _require(set(report.get("pairs", {})) == set(aim3_fixed_control_analysis.FINE_TASKS),
             "Aim3 pair inventory mismatch")
    verdicts: dict[str, Any] = {}
    familywise_pass: list[str] = []
    for fine, control in aim3_fixed_control_analysis.PAIRS:
        fine_block = report["tasks"][fine]
        control_block = report["tasks"][control]
        pair = report["pairs"][fine]
        delta = pair.get("primary_delta_control_minus_fine", {})
        expected_delta = _finite(control_block.get("auroc"), f"Aim3 {control} AUROC") - _finite(
            fine_block.get("auroc"), f"Aim3 {fine} AUROC"
        )
        _require(math.isclose(_finite(delta.get("estimate"), f"Aim3 {fine} delta"), expected_delta,
                              rel_tol=0.0, abs_tol=1e-12),
                 f"Aim3 {fine} delta algebra mismatch")
        _ci(fine_block.get("ci95"), f"Aim3 {fine} CI")
        _ci(control_block.get("ci95"), f"Aim3 {control} CI")
        _ci(delta.get("ci95"), f"Aim3 {fine} delta CI")
        expected_verdict = _expected_aim3_verdict(fine_block, control_block, delta)
        _require(pair.get("nominal_95_verdict") == expected_verdict,
                 f"Aim3 {fine} nominal verdict logic mismatch")
        gate = pair.get("familywise_ceiling_sensitivity", {})
        expected_gate = bool(
            _finite(gate.get("fine_99_upper"), f"Aim3 {fine} fine upper") < 0.60
            and _finite(gate.get("control_99_lower"), f"Aim3 {fine} control lower") > 0.50
            and _finite(gate.get("delta_99_lower"), f"Aim3 {fine} delta lower") > 0.0
        )
        _require(gate.get("pass") is expected_gate, f"Aim3 {fine} familywise gate mismatch")
        expected_status = "CEILING_FWER_PASS" if expected_gate else "CEILING_FWER_NOT_ESTABLISHED"
        _require(gate.get("status") == expected_status, f"Aim3 {fine} familywise status mismatch")
        if expected_gate:
            familywise_pass.append(fine)
        sensitivity = pair.get("subcohort_standardized_control_sensitivity", {})
        standardized_delta = _finite(sensitivity.get("delta_control_minus_fine"),
                                     f"Aim3 {fine} standardized delta")
        expected_standardized = _finite(sensitivity.get("control_auroc"),
                                        f"Aim3 {fine} standardized control") - fine_block["auroc"]
        _require(math.isclose(standardized_delta, expected_standardized, rel_tol=0.0, abs_tol=1e-12),
                 f"Aim3 {fine} standardized delta algebra mismatch")
        _ci(sensitivity.get("control_ci95"), f"Aim3 {fine} standardized control CI")
        _ci(sensitivity.get("delta_ci95"), f"Aim3 {fine} standardized delta CI")
        verdicts[fine] = {"nominal_95": expected_verdict, "familywise_ceiling": expected_gate}
    summary = report.get("familywise_summary", {})
    _require(summary.get("rungs_passing_ceiling_gate") == familywise_pass,
             "Aim3 familywise summary list mismatch")
    _require(summary.get("n_passing") == len(familywise_pass), "Aim3 familywise summary count mismatch")
    return {
        "status": "PASS",
        "result": _identity(result_path),
        "native_logit_only": True,
        "headline_verdicts": verdicts,
        "familywise_rungs_passing": familywise_pass,
    }


def _repeated_bounds(value: object, label: str) -> tuple[float, float]:
    _require(isinstance(value, dict), f"{label} is not a bounds object")
    low = _finite(value.get("lower"), f"{label}.lower")
    high = _finite(value.get("upper"), f"{label}.upper")
    confidence = _finite(value.get("one_sided_confidence"), f"{label}.confidence")
    _require(low <= high, f"{label} is reversed")
    _require(
        math.isclose(confidence, 0.99, rel_tol=0.0, abs_tol=1e-15),
        f"{label} is not the predeclared one-sided 99% bound",
    )
    return low, high


def _repeated_verdict(
    fine_bounds: tuple[float, float],
    control_bounds: tuple[float, float],
    delta_bounds: tuple[float, float],
) -> str:
    if control_bounds[0] <= 0.50:
        return "UNDERPOWERED"
    ceiling = fine_bounds[1] < 0.60 and delta_bounds[0] > 0.0
    signal = fine_bounds[0] > 0.50
    if ceiling and signal:
        return "CEILING_WITH_RESIDUAL_SIGNAL"
    if ceiling:
        return "CEILING"
    if signal:
        return "FINE_RESOLUTION_EVIDENCE"
    return "INCONCLUSIVE"


def _repeated_consensus(verdicts: list[str]) -> str:
    _require(
        len(verdicts) == len(REPEATED_DRAW_SEEDS),
        "Aim3 repeated-control consensus lacks exactly three draws",
    )
    ceiling = {"CEILING", "CEILING_WITH_RESIDUAL_SIGNAL"}
    if all(value in ceiling for value in verdicts):
        return "CONSENSUS_CEILING"
    if all(value == "UNDERPOWERED" for value in verdicts):
        return "CONSENSUS_UNDERPOWERED"
    return "NO_CEILING_CONSENSUS"


def _delegate_repeated_verifier(root: Path) -> str:
    """Run the campaign's exhaustive verifier without permitting output writes."""

    output = io.StringIO()
    with redirect_stdout(output):
        aim3_repeated_control_campaign.cmd_verify_output(
            argparse.Namespace(output_root=str(root))
        )
    message = output.getvalue().strip()
    _require(
        message.startswith("PASS — immutable Aim-3 campaign verified:"),
        "Aim3 repeated-control component verifier did not report PASS",
    )
    return message


def _verify_aim3_repeated(
    repeated_root: Path,
    *,
    expected_fine_root: Path,
) -> dict[str, Any]:
    """Verify the sealed three-draw sensitivity and its headline algebra."""

    required = (
        repeated_root / "lineage_start.json",
        repeated_root / "lineage_complete.json",
        repeated_root / "analysis" / "aim3_repeated_control_report.json",
        repeated_root / "analysis" / "bootstrap_distributions.npz",
        repeated_root / "analysis" / "analysis_audit.json",
    )
    missing = [str(path.relative_to(repeated_root)) for path in required if not path.is_file()]
    _require(
        not missing,
        "Aim3 repeated-control lineage is partial/incomplete; missing: " + ", ".join(missing),
    )

    # These constants are independently pinned here.  Refuse to delegate to a
    # live helper whose declared protocol no longer matches the sealed design.
    _require(aim3_repeated_control_campaign.CAP == CAP, "Repeated-control helper cap drifted")
    _require(
        tuple(aim3_repeated_control_campaign.MODEL_SEEDS) == REPEATED_MODEL_SEEDS,
        "Repeated-control helper model seeds drifted",
    )
    _require(
        tuple(aim3_repeated_control_campaign.WT_DRAW_SEEDS) == REPEATED_DRAW_SEEDS,
        "Repeated-control helper WT draw seeds drifted",
    )
    _require(
        aim3_repeated_control_campaign.N_BOOTSTRAP == REPEATED_BOOTSTRAP
        and aim3_repeated_control_campaign.BOOTSTRAP_SEED == REPEATED_BOOTSTRAP_SEED,
        "Repeated-control helper bootstrap protocol drifted",
    )

    start = _read_json(required[0])
    _require(start.get("schema_version") == REPEATED_SCHEMA_VERSION, "Aim3 repeated schema mismatch")
    _require(start.get("status") == "prepared", "Aim3 repeated lineage start is not prepared")
    _require(
        Path(str(start.get("output_root", ""))).resolve(strict=True) == repeated_root,
        "Aim3 repeated lineage records the wrong output root",
    )
    _require(
        Path(str(start.get("fine_input_root", ""))).resolve(strict=True) == expected_fine_root,
        "Aim3 repeated controls do not reuse the corrected Aim3 fine-task lineage",
    )
    start_protocol = start.get("protocol", {})
    exact_start_protocol = {
        "cap": CAP,
        "model_seeds": list(REPEATED_MODEL_SEEDS),
        "wt_draw_seeds": list(REPEATED_DRAW_SEEDS),
        "n_control_chains": REPEATED_CHAINS,
        "n_control_folds": REPEATED_CONTROL_FOLDS,
        "fine_folds_reused": REPEATED_FINE_FOLDS,
        "bootstrap_seed": REPEATED_BOOTSTRAP_SEED,
        "n_bootstrap": REPEATED_BOOTSTRAP,
    }
    for key, expected in exact_start_protocol.items():
        _require(
            start_protocol.get(key) == expected,
            f"Aim3 repeated lineage protocol mismatch: {key}",
        )
    _require(
        "five rung-level IUTs" in str(start_protocol.get("primary_fwer", ""))
        and "one-sided 99%" in str(start_protocol.get("primary_fwer", ""))
        and "three-draw consensus" in str(start_protocol.get("primary_fwer", "")),
        "Aim3 repeated lineage does not declare the preplanned familywise rule",
    )

    # The component verifier authenticates all frozen source/input bytes, the
    # packed store, 225 fold completions and OOF union, and all 20k arrays.
    delegate_message = _delegate_repeated_verifier(repeated_root)

    report_path = required[2]
    distribution_path = required[3]
    audit_path = required[4]
    report = _read_json(report_path)
    _require(
        report.get("schema_version") == REPEATED_SCHEMA_VERSION
        and report.get("status") == "complete",
        "Aim3 repeated report is not a completed schema-v2 result",
    )
    _require(
        Path(str(report.get("output_root", ""))).resolve(strict=True) == repeated_root,
        "Aim3 repeated report records the wrong output root",
    )
    protocol = report.get("protocol", {})
    _require(
        protocol.get("native_score")
        == "mean slide native logit per patient; mean across seeds 42/43/44",
        "Aim3 repeated result is not native-logit-only",
    )
    _require(
        protocol.get("n_bootstrap") == REPEATED_BOOTSTRAP
        and protocol.get("bootstrap_seed_base") == REPEATED_BOOTSTRAP_SEED,
        "Aim3 repeated report bootstrap budget/seed mismatch",
    )
    _require(
        "partial-paired" in str(protocol.get("bootstrap", ""))
        and "subcohort x frozen k_fold" in str(protocol.get("bootstrap", "")),
        "Aim3 repeated result lacks the predeclared partial-paired stratification",
    )
    _require(
        "positive patient IDs" in str(protocol.get("shared_component", ""))
        and "independently resampled" in str(protocol.get("distinct_component", "")),
        "Aim3 repeated result has the wrong pairing contract",
    )
    familywise = protocol.get("primary_fwer", {})
    _require(
        familywise.get("family") == "five rung-level intersection-union ceiling tests"
        and math.isclose(_finite(familywise.get("alpha"), "Aim3 repeated alpha"), 0.05)
        and math.isclose(
            _finite(familywise.get("per_rung_one_sided_alpha"), "Aim3 repeated rung alpha"),
            0.01,
        )
        and familywise.get("bounds") == "one-sided 99%",
        "Aim3 repeated primary familywise design mismatch",
    )
    _require(
        "all three predeclared WT draws" in str(protocol.get("ceiling_consensus", "")),
        "Aim3 repeated report does not require three-draw consensus",
    )
    _require(
        protocol.get("sensitivity_only")
        == "central 99.6667% intervals over 15 draw-by-rung comparisons",
        "Aim3 repeated 15-comparison interval is not labelled sensitivity-only",
    )
    _require(
        report.get("accounting")
        == {
            "fine_folds_reused": REPEATED_FINE_FOLDS,
            "control_chains": REPEATED_CHAINS,
            "control_folds": REPEATED_CONTROL_FOLDS,
        },
        "Aim3 repeated report fold/chain accounting mismatch",
    )
    _require(set(report.get("rungs", {})) == set(REPEATED_FINE_TASKS), "Aim3 repeated rung inventory mismatch")
    _require(not _walk_forbidden_roundtrip(report), "Aim3 repeated result uses a probability roundtrip")

    consensus: dict[str, str] = {}
    draw_verdicts: dict[str, dict[str, str]] = {}
    expected_draw_keys = {str(seed) for seed in REPEATED_DRAW_SEEDS}
    expected_model_keys = {str(seed) for seed in REPEATED_MODEL_SEEDS}
    for fine in REPEATED_FINE_TASKS:
        rung = report["rungs"][fine]
        _require(rung.get("control") == REPEATED_CONTROLS[fine], f"Aim3 repeated {fine} control mismatch")
        _require(
            set(rung.get("fine_per_model_seed_auroc", {})) == expected_model_keys,
            f"Aim3 repeated {fine} fine model-seed inventory mismatch",
        )
        draws = rung.get("draws", {})
        _require(set(draws) == expected_draw_keys, f"Aim3 repeated {fine} draw inventory mismatch")
        decisions: list[str] = []
        per_draw: dict[str, str] = {}
        frozen_fine: dict[str, Any] | None = None
        for draw_seed in REPEATED_DRAW_SEEDS:
            draw = draws[str(draw_seed)]
            _require(
                set(draw.get("control_per_model_seed_auroc", {})) == expected_model_keys,
                f"Aim3 repeated {fine}/wt{draw_seed} control model-seed inventory mismatch",
            )
            fine_result = draw.get("fine", {})
            control_result = draw.get("control", {})
            delta_result = draw.get("delta_control_minus_fine", {})
            fine_point = _finite(fine_result.get("auroc"), f"Aim3 repeated {fine} fine AUROC")
            control_point = _finite(
                control_result.get("auroc"), f"Aim3 repeated {fine}/wt{draw_seed} control AUROC"
            )
            delta_point = _finite(
                delta_result.get("estimate"), f"Aim3 repeated {fine}/wt{draw_seed} delta"
            )
            _require(
                math.isclose(delta_point, control_point - fine_point, rel_tol=0.0, abs_tol=1e-12),
                f"Aim3 repeated {fine}/wt{draw_seed} point-estimate algebra mismatch",
            )
            for label, block in (
                ("fine", fine_result),
                ("control", control_result),
                ("delta", delta_result),
            ):
                _ci(block.get("ci95_two_sided"), f"Aim3 repeated {fine}/wt{draw_seed} {label} CI95")
                _ci(
                    block.get("ultra_conservative_15_comparison_two_sided"),
                    f"Aim3 repeated {fine}/wt{draw_seed} {label} 15-comparison CI",
                )
            fine_bounds = _repeated_bounds(
                fine_result.get("primary_fwer_one_sided"),
                f"Aim3 repeated {fine}/wt{draw_seed} fine bounds",
            )
            control_bounds = _repeated_bounds(
                control_result.get("primary_fwer_one_sided"),
                f"Aim3 repeated {fine}/wt{draw_seed} control bounds",
            )
            delta_bounds = _repeated_bounds(
                delta_result.get("primary_fwer_one_sided"),
                f"Aim3 repeated {fine}/wt{draw_seed} delta bounds",
            )
            expected_conditions = {
                "fine_upper_99_lt_0.60": fine_bounds[1] < 0.60,
                "control_lower_99_gt_0.50": control_bounds[0] > 0.50,
                "delta_lower_99_gt_0": delta_bounds[0] > 0.0,
                "fine_lower_99_gt_0.50": fine_bounds[0] > 0.50,
            }
            _require(
                draw.get("primary_conditions") == expected_conditions,
                f"Aim3 repeated {fine}/wt{draw_seed} familywise-condition algebra mismatch",
            )
            expected_verdict = _repeated_verdict(fine_bounds, control_bounds, delta_bounds)
            _require(
                draw.get("primary_verdict") == expected_verdict,
                f"Aim3 repeated {fine}/wt{draw_seed} familywise verdict algebra mismatch",
            )
            if frozen_fine is None:
                frozen_fine = fine_result
            else:
                _require(
                    fine_result == frozen_fine,
                    f"Aim3 repeated {fine} frozen fine result differs across WT draws",
                )
            decisions.append(expected_verdict)
            per_draw[str(draw_seed)] = expected_verdict
        _require(rung.get("draw_verdicts") == decisions, f"Aim3 repeated {fine} draw-verdict list mismatch")
        expected_consensus = _repeated_consensus(decisions)
        _require(
            rung.get("consensus_verdict") == expected_consensus,
            f"Aim3 repeated {fine} three-draw consensus algebra mismatch",
        )
        consensus[fine] = expected_consensus
        draw_verdicts[fine] = per_draw

    audit = _read_json(audit_path)
    _require(
        audit.get("schema_version") == REPEATED_SCHEMA_VERSION and audit.get("status") == "PASS",
        "Aim3 repeated analysis audit is not PASS",
    )
    chains = audit.get("control_chains")
    _require(isinstance(chains, list) and len(chains) == REPEATED_CHAINS, "Aim3 repeated audit lacks 45 chains")
    observed_chains: set[tuple[str, int, int]] = set()
    fold_count = 0
    for chain in chains:
        _require(isinstance(chain, dict), "Aim3 repeated audit contains a malformed chain")
        key = (
            str(chain.get("control")),
            int(chain.get("draw_seed", -1)),
            int(chain.get("model_seed", -1)),
        )
        observed_chains.add(key)
        folds = chain.get("fold_completions")
        _require(isinstance(folds, list) and len(folds) == 5, f"Aim3 repeated chain {key} lacks five folds")
        fold_count += len(folds)
    expected_chains = {
        (control, draw_seed, model_seed)
        for control in REPEATED_CONTROLS.values()
        for draw_seed in REPEATED_DRAW_SEEDS
        for model_seed in REPEATED_MODEL_SEEDS
    }
    _require(observed_chains == expected_chains, "Aim3 repeated audit chain identities are not exact")
    _require(fold_count == REPEATED_CONTROL_FOLDS, "Aim3 repeated audit does not bind 225 folds")
    _require(audit.get("lineage_start") == _identity(required[0]), "Aim3 repeated audit start hash mismatch")
    _require(audit.get("report") == _identity(report_path), "Aim3 repeated audit result hash mismatch")
    _require(
        audit.get("bootstrap_distributions") == _identity(distribution_path),
        "Aim3 repeated audit bootstrap hash mismatch",
    )

    complete = _read_json(required[1])
    _require(
        complete.get("schema_version") == REPEATED_SCHEMA_VERSION
        and complete.get("status") == "completed",
        "Aim3 repeated completion receipt is invalid",
    )
    _require(
        Path(str(complete.get("output_root", ""))).resolve(strict=True) == repeated_root,
        "Aim3 repeated completion records the wrong root",
    )
    expected_artifacts = {
        "aim3_repeated_control_report.json": _identity(report_path),
        "bootstrap_distributions.npz": _identity(distribution_path),
        "analysis_audit.json": _identity(audit_path),
    }
    _require(complete.get("artifacts") == expected_artifacts, "Aim3 repeated completion hashes mismatch")
    return {
        "status": "PASS",
        "component_verifier": delegate_message,
        "native_logit_only": True,
        "control_chains": REPEATED_CHAINS,
        "control_folds": REPEATED_CONTROL_FOLDS,
        "n_bootstrap": REPEATED_BOOTSTRAP,
        "wt_draw_seeds": list(REPEATED_DRAW_SEEDS),
        "draw_verdicts": draw_verdicts,
        "consensus_verdicts": consensus,
        "lineage_start": _identity(required[0]),
        "result": _identity(report_path),
        "analysis_audit": _identity(audit_path),
        "lineage_complete": _identity(required[1]),
        "bootstrap_distributions": _identity(distribution_path),
    }


def _verify_refresh_binding(refresh_root: Path, core_root: Path) -> dict[str, Any]:
    receipt_path = refresh_root / aim2_exploratory_refresh.RECEIPT_NAME
    receipt = _read_json(receipt_path)
    _require(Path(str(receipt.get("input_root", ""))).resolve(strict=True) == core_root,
             "Aim2 E2d refresh is not bound to the explicit core root")
    _require(receipt.get("cap") == CAP, "Aim2 E2d refresh cap mismatch")
    _require(receipt.get("n_bootstrap") == FULL_BOOTSTRAP, "Aim2 E2d refresh is not full 10k")
    _require(receipt.get("bootstrap_seed") == BOOTSTRAP_SEED, "Aim2 E2d refresh seed mismatch")
    _require(receipt.get("analysis_grade") == "full_10000_bootstrap", "Aim2 refresh is smoke grade")
    verification_path = refresh_root / aim2_exploratory_refresh.VERIFICATION_NAME
    _require(verification_path.is_file(), "Aim2 refresh lacks its immutable verification receipt")
    verification = _read_json(verification_path)
    _require(verification.get("status") == "pass", "Aim2 refresh verification is not PASS")

    # A completed refresh is archival: executed source and protocol context are
    # preserved inside the immutable output root.  Later edits to the live
    # Markdown report (or to live source after the analysis) must not make the
    # sealed numerical result unverifiable.  Material data/model inputs remain
    # live-hash checked; source and the two documentation-context inputs are
    # checked against their frozen copies.
    context_names = {
        "current_protocol": "Experimental_Setup.md",
        "historical_design_archive": "Experimental_Setup_DESIGN_ARCHIVE.md",
    }
    external_inputs_checked = 0
    context_snapshots_checked = 0
    for recorded in receipt.get("input_inventory", []):
        label = str(recorded.get("label", ""))
        if label in context_names:
            actual = _identity(
                refresh_root / "context_snapshot" / "reports" / context_names[label]
            )
            context_snapshots_checked += 1
        else:
            actual = _identity(Path(str(recorded.get("path", ""))))
            external_inputs_checked += 1
        _require(
            actual["size_bytes"] == recorded.get("size_bytes")
            and actual["sha256"] == recorded.get("sha256"),
            f"Aim2 refresh input identity changed: {label}",
        )
    _require(
        context_snapshots_checked == len(context_names),
        "Aim2 refresh does not preserve both documentation-context snapshots",
    )

    source_snapshots_checked = 0
    for recorded in receipt.get("live_source_inventory", []):
        relative = Path(str(recorded.get("label", "")))
        _require(
            relative != Path("") and not relative.is_absolute() and ".." not in relative.parts,
            f"Aim2 refresh source label is unsafe: {relative}",
        )
        actual = _identity(refresh_root / "source_snapshot" / relative)
        _require(
            actual["size_bytes"] == recorded.get("size_bytes")
            and actual["sha256"] == recorded.get("sha256"),
            f"Aim2 refresh frozen source identity changed: {relative}",
        )
        source_snapshots_checked += 1
    _require(source_snapshots_checked > 0, "Aim2 refresh lacks frozen source provenance")

    active_reports = [
        _read_json(refresh_root / relative)
        for relative in aim2_exploratory_refresh.EXPECTED_RESULTS[:3]
    ]
    findings = [item for index, report in enumerate(active_reports)
                for item in _walk_forbidden_roundtrip(report, location=f"refresh[{index}]")]
    _require(not findings, f"Aim2 refresh contains active probability-roundtrip markers: {findings}")
    return {
        "status": "PASS",
        "refresh_receipt": _identity(receipt_path),
        "verification_receipt": _identity(verification_path),
        "supersedes_core_panels": ["E2d2", "E2d4", "E2d6"],
        "probability_roundtrip_rejected": True,
        "archival_provenance": {
            "external_material_inputs_live_checked": external_inputs_checked,
            "frozen_source_files_checked": source_snapshots_checked,
            "frozen_documentation_contexts_checked": context_snapshots_checked,
            "later_live_markdown_edits_are_not_analysis_inputs": True,
        },
    }


def verify(
    *,
    aim2_core_root: Path,
    e2c_root: Path,
    aim2_refresh_root: Path,
    aim3_root: Path,
    aim3_repeated_root: Path,
) -> dict[str, Any]:
    roots = {
        "aim2_core": _absolute_root(aim2_core_root, "--aim2-core-root"),
        "e2c_offset": _absolute_root(e2c_root, "--e2c-root"),
        "aim2_e2d_refresh": _absolute_root(aim2_refresh_root, "--aim2-refresh-root"),
        "aim3_corrected": _absolute_root(aim3_root, "--aim3-root"),
        "aim3_repeated_controls": _absolute_root(
            aim3_repeated_root, "--aim3-repeated-root"
        ),
    }
    _validate_distinct_roots(roots)

    aim2 = verify_aim2_corrected.verify(
        roots["aim2_core"],
        roots["e2c_offset"],
        CAP,
        allow_known_documentation_exception=True,
    )
    _require(aim2.get("scientific_and_numeric_checks") == "PASS", "Aim2 numeric verifier failed")
    _require(aim2.get("code_model_data_identity_checks") == "PASS", "Aim2 identity verifier failed")

    refresh_component = aim2_exploratory_refresh.verify_refresh(
        roots["aim2_e2d_refresh"], check_live_inputs=False
    )
    _require(refresh_component.get("status") == "PASS", "Aim2 refresh verifier failed")
    refresh = _verify_refresh_binding(roots["aim2_e2d_refresh"], roots["aim2_core"])
    e2c = _verify_e2c_native_contract(roots["e2c_offset"], roots["aim2_core"])

    aim3_start = _read_json(roots["aim3_corrected"] / "lineage_start.json")
    _require(aim3_start.get("n_bootstrap") == FULL_BOOTSTRAP, "Aim3 lineage is not full 10k")
    e3_root = _absolute_root(aim3_start.get("e3_root", ""), "Aim3 recorded E3 root")
    e0_root = _absolute_root(aim3_start.get("e0_root", ""), "Aim3 recorded E0 root")
    aim3_bundle = aim3_fixed_control_analysis.validate_inputs(e3_root, e0_root)
    aim3_component = aim3_fixed_control_analysis.verify_output(aim3_bundle, roots["aim3_corrected"])
    _require(aim3_component.get("status") == "PASS", "Aim3 component verifier failed")
    aim3 = _verify_aim3_headlines(roots["aim3_corrected"])
    aim3_repeated = _verify_aim3_repeated(
        roots["aim3_repeated_controls"], expected_fine_root=e3_root
    )

    component_receipts = {
        "aim2_core_lineage_start": _identity(roots["aim2_core"] / "lineage_start.json"),
        "e2c_lineage_complete": _identity(roots["e2c_offset"] / "lineage_complete.json"),
        "aim2_refresh": refresh["refresh_receipt"],
        "aim2_refresh_verification": refresh["verification_receipt"],
        "aim3_lineage_complete": _identity(roots["aim3_corrected"] / "lineage_complete.json"),
        "aim3_repeated_result": aim3_repeated["result"],
        "aim3_repeated_analysis_audit": aim3_repeated["analysis_audit"],
        "aim3_repeated_lineage_complete": aim3_repeated["lineage_complete"],
    }
    return {
        "schema_version": 2,
        "status": "PASS",
        "scope": "corrected Aim 2 and Aim 3, cap 8192",
        "roots": {name: str(path) for name, path in roots.items()},
        "component_receipts": component_receipts,
        "checks": {
            "aim2_core_and_e2c_component_verifier": aim2,
            "aim2_e2d_refresh_component_verifier": refresh_component,
            "aim2_e2d_refresh_binding": refresh,
            "e2c_native_logit_and_headline_algebra": e2c,
            "aim3_component_verifier": aim3_component,
            "aim3_native_logit_and_headline_algebra": aim3,
            "aim3_repeated_control_component_and_headline_verifier": aim3_repeated,
        },
        "canonical_result_map": {
            "Aim2_E2a_E2b_E2d1_E2d3_E2d5": str(roots["aim2_core"]),
            "Aim2_E2c": str(roots["e2c_offset"]),
            "Aim2_E2d2_E2d4_E2d6": str(roots["aim2_e2d_refresh"]),
            "Aim3_primary_fixed_control_draw": str(roots["aim3_corrected"]),
            "Aim3_repeated_control_robustness_sensitivity": str(
                roots["aim3_repeated_controls"]
            ),
        },
        "stale_artifact_policy": {
            "status": "PASS",
            "probability_roundtrip_results_are_canonical": False,
            "core_E2d2_E2d4_E2d6_are_canonical": False,
            "legacy_artifacts_may_remain_preserved_but_are_not_result_sources": True,
        },
        "interpretation_guardrail": (
            "A confidence interval crossing its null is not established; it is never proof of "
            "equivalence, invariance, absence of signal, absence of benefit, or absence of harm."
        ),
        "verifier": _identity(Path(__file__)),
    }


def _receipt_destination(value: str | Path, roots: tuple[Path, ...]) -> Path:
    path = Path(value).expanduser()
    _require(path.is_absolute(), f"--write-receipt must be an absolute path: {path}")
    resolved = path.resolve(strict=False)
    _require(resolved.parent.is_dir(), f"Receipt parent does not exist: {resolved.parent}")
    for root in roots:
        _require(resolved != root and root not in resolved.parents,
                 f"Receipt must be outside immutable root: {root}")
    return resolved


def _write_json_once_atomic(path: Path, value: dict[str, Any]) -> None:
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aim2-core-root", required=True)
    parser.add_argument("--e2c-root", required=True)
    parser.add_argument("--aim2-refresh-root", required=True)
    parser.add_argument("--aim3-root", required=True)
    parser.add_argument("--aim3-repeated-root", required=True)
    parser.add_argument("--write-receipt", metavar="ABS_JSON")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        roots = tuple(
            _absolute_root(value, label)
            for value, label in (
                (args.aim2_core_root, "--aim2-core-root"),
                (args.e2c_root, "--e2c-root"),
                (args.aim2_refresh_root, "--aim2-refresh-root"),
                (args.aim3_root, "--aim3-root"),
                (args.aim3_repeated_root, "--aim3-repeated-root"),
            )
        )
        receipt = verify(
            aim2_core_root=roots[0],
            e2c_root=roots[1],
            aim2_refresh_root=roots[2],
            aim3_root=roots[3],
            aim3_repeated_root=roots[4],
        )
        if args.write_receipt:
            destination = _receipt_destination(args.write_receipt, roots)
            _write_json_once_atomic(destination, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    except (
        ConsolidatedVerificationError,
        aim2_exploratory_refresh.RefreshError,
        verify_aim2_corrected.VerificationError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"FAIL — {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
