from __future__ import annotations

import hashlib
import json
import math
import sys
import types
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import final_v13_bundle_receipt as verifier  # noqa: E402


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path, display: str) -> dict[str, Any]:
    return {"path": display, "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _source_record(
    repo: Path,
    path: Path,
    *,
    source_id: str,
    aim: str,
    experiment: str,
    role: str,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "aims": [aim],
        "experiments": [experiment],
        "role": role,
        **_identity(path, str(path.relative_to(repo))),
    }


def _aim3_result(run_root: str) -> dict[str, Any]:
    def summary(
        estimate: float,
        *,
        fwer_lower: float,
        fwer_upper: float,
        ci_lower: float | None = None,
        ci_upper: float | None = None,
    ) -> dict[str, Any]:
        return {
            "estimate": estimate,
            "ci95_two_sided": [
                estimate - 0.05 if ci_lower is None else ci_lower,
                estimate + 0.05 if ci_upper is None else ci_upper,
            ],
            "primary_fwer_one_sided": {
                "lower": fwer_lower,
                "upper": fwer_upper,
                "confidence": 0.99,
            },
        }

    ceiling_gate = {
        "conditions": {
            "fine_upper_lt_0p60": True,
            "control_lower_gt_0p50": True,
            "delta_lower_gt_zero": True,
        },
        "ceiling": True,
        "verdict": "CEILING",
    }
    seed_map = {str(seed): 0.51 + index / 100 for index, seed in enumerate(verifier.MODEL_SEEDS)}
    fixed: dict[str, Any] = {"rungs": {}}
    repeated: dict[str, Any] = {"rungs": {}}
    bindings: list[dict[str, Any]] = []
    for rung_index, (rung, control) in enumerate(verifier.AIM3_RUNG_CONTROL.items()):
        base = 0.52 + rung_index / 100
        fixed["rungs"][rung] = {
            "control_task": control,
            "fine_per_seed": dict(seed_map),
            "control_per_seed": {str(seed): value + 0.1 for seed, value in seed_map.items()},
            "fine": summary(base, fwer_lower=0.49, fwer_upper=0.59),
            "control": summary(base + 0.1, fwer_lower=0.55, fwer_upper=0.75),
            "delta_control_minus_fine": summary(0.1, fwer_lower=0.01, fwer_upper=0.20),
            "gate": dict(ceiling_gate),
        }
        for metric, value in (
            ("fine_auroc", base),
            ("control_auroc", base + 0.1),
            ("delta_auroc", 0.1),
        ):
            bindings.append(
                {
                    "id": f"aim3_source.fixed.{rung}.{metric}",
                    "section": "fixed",
                    "rung": rung,
                    "draw_seed": None,
                    "metric": metric,
                    "value": value,
                }
            )
        draws: dict[str, Any] = {}
        for draw_index, draw_seed in enumerate(verifier.WT_DRAW_SEEDS):
            estimate = base + draw_index / 1000
            fine_summary = summary(estimate, fwer_lower=0.49, fwer_upper=0.59)
            control_summary = summary(estimate + 0.1, fwer_lower=0.55, fwer_upper=0.75)
            delta_summary = summary(0.1, fwer_lower=0.01, fwer_upper=0.20)
            gate = dict(ceiling_gate)
            if rung == "allele2" and draw_seed == 20260824:
                delta_summary = summary(
                    0.1,
                    fwer_lower=-0.01,
                    fwer_upper=0.20,
                    ci_lower=0.01,
                    ci_upper=0.19,
                )
                gate = {
                    "conditions": {
                        "fine_upper_lt_0p60": True,
                        "control_lower_gt_0p50": True,
                        "delta_lower_gt_zero": False,
                    },
                    "ceiling": False,
                    "verdict": "INCONCLUSIVE",
                }
            elif rung == "g12c" and draw_seed == 20260825:
                fine_summary = summary(estimate, fwer_lower=0.49, fwer_upper=0.61)
                control_summary = summary(
                    estimate + 0.1,
                    fwer_lower=0.49,
                    fwer_upper=0.75,
                    ci_lower=0.51,
                    ci_upper=0.71,
                )
                gate = {
                    "conditions": {
                        "fine_upper_lt_0p60": False,
                        "control_lower_gt_0p50": False,
                        "delta_lower_gt_zero": True,
                    },
                    "ceiling": False,
                    "verdict": "UNDERPOWERED",
                }
            draws[str(draw_seed)] = {
                "control_per_seed": {str(seed): value + 0.1 for seed, value in seed_map.items()},
                "fine": fine_summary,
                "control": control_summary,
                "delta_control_minus_fine": delta_summary,
                "gate": gate,
            }
            for metric, value in (
                ("fine_auroc", estimate),
                ("control_auroc", estimate + 0.1),
                ("delta_auroc", 0.1),
            ):
                bindings.append(
                    {
                        "id": f"aim3_source.repeated.{rung}.wt{draw_seed}.{metric}",
                        "section": "repeated",
                        "rung": rung,
                        "draw_seed": draw_seed,
                        "metric": metric,
                        "value": value,
                    }
                )
        consensus = "NO_CEILING_CONSENSUS" if rung in {"allele2", "g12c"} else "CONSENSUS_CEILING"
        repeated["rungs"][rung] = {
            "control_task": control,
            "fine_per_seed": dict(seed_map),
            "draws": draws,
            "consensus_verdict": consensus,
            "all_three_draws_required": True,
        }
        bindings.append(
            {
                "id": f"aim3_source.repeated.{rung}.consensus_verdict",
                "section": "repeated",
                "rung": rung,
                "draw_seed": None,
                "metric": "consensus_verdict",
                "value": consensus,
            }
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "campaign": verifier.EXPECTED_AIM3_EXPERIMENT,
        "population": "tcga_surgen_primary",
        "population_display": "TCGA + SurGen primaries",
        "encoder": "UNI-v1",
        "model_seeds": list(verifier.MODEL_SEEDS),
        "wt_draw_seeds": list(verifier.WT_DRAW_SEEDS),
        "external_development_cohorts": [],
        "aggregation": "patient native-logit five-seed ensemble",
        "fixed": fixed,
        "repeated": repeated,
        "report_bindings": sorted(bindings, key=lambda record: record["id"]),
    }


def _aim3_fine_result(full: dict[str, Any]) -> dict[str, Any]:
    rungs: dict[str, Any] = {}
    bindings: list[dict[str, Any]] = []
    for _index, rung in enumerate(verifier.AIM3_RUNG_CONTROL):
        summary = dict(full["fixed"]["rungs"][rung]["fine"])
        positive, negative = verifier.AIM3_FINE_EXTERNAL_EXPECTED_INTERNAL_CENSUS[rung]
        rungs[rung] = {
            "per_seed_auroc": dict(full["fixed"]["rungs"][rung]["fine_per_seed"]),
            "five_seed_ensemble": summary,
            "patients": positive + negative,
            "positive": positive,
            "negative": negative,
        }
        bindings.append(
            {
                "id": f"aim3_source.fine.{rung}.auroc",
                "section": "fine",
                "rung": rung,
                "draw_seed": None,
                "metric": "fine_auroc",
                "value": summary["estimate"],
            }
        )
    return {
        "schema_version": 1,
        "status": "fine_phase_complete_candidate_unsealed",
        "campaign": verifier.EXPECTED_AIM3_EXPERIMENT,
        "phase": "fine",
        "population": "tcga_surgen_primary",
        "population_display": "TCGA + SurGen primaries",
        "encoder": "UNI-v1",
        "model_seeds": list(verifier.MODEL_SEEDS),
        "wt_draw_seeds": list(verifier.WT_DRAW_SEEDS),
        "controls_status": "not_started",
        "external_development_cohorts": [],
        "aggregation": "patient native-logit five-seed ensemble",
        "fit_accounting": dict(verifier.EXPECTED_FINE_TRAINING_COUNTS),
        "fine": {"rungs": rungs},
        "report_bindings": sorted(bindings, key=lambda record: record["id"]),
        "v13_status": "candidate_unsealed_pending_fixed_and_repeated_controls",
    }


def _sample_summary(values: dict[str, float]) -> tuple[float, float]:
    array = list(values.values())
    mean = sum(array) / len(array)
    sd = math.sqrt(sum((value - mean) ** 2 for value in array) / (len(array) - 1))
    return mean, sd


def _external_metric_fixture(
    positive: int,
    negative: int,
    *,
    base: float,
    seed_key: str = "per_seed_auroc",
) -> dict[str, Any]:
    per_seed = {str(seed): base + index / 1000 for index, seed in enumerate(verifier.MODEL_SEEDS)}
    mean, sd = _sample_summary(per_seed)
    return {
        "status": "ESTIMABLE",
        "n_records": positive + negative,
        "positive": positive,
        "negative": negative,
        "support": "VERY_SPARSE_LT10"
        if min(positive, negative) < 10
        else ("EXPLORATORY_10_19" if min(positive, negative) < 20 else "BETTER_POWERED_GE20"),
        "sparse": min(positive, negative) < 20,
        seed_key: per_seed,
        "seed_auroc_mean": mean,
        "seed_auroc_sample_sd": sd,
        "five_seed_refit_ensemble": {
            "auroc": base + 0.002,
            "ci95": [max(0.0, base - 0.1), min(1.0, base + 0.1)],
        },
    }


def _aim3_fine_external_result(fine: dict[str, Any]) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    report_rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(verifier.AIM3_FINE_TASKS):
        fine_rung = fine["fine"]["rungs"][task]
        seed_values = dict(fine_rung["per_seed_auroc"])
        seed_mean, seed_sd = _sample_summary(seed_values)
        folds = {str(index): 0.45 + task_index / 100 + index / 1000 for index in range(5)}
        fold_mean, fold_sd = _sample_summary(folds)
        internal = {
            "design": "honest inherited five-fold OOF on TCGA+SurGen primaries",
            "patients": fine_rung["patients"],
            "positive": fine_rung["positive"],
            "negative": fine_rung["negative"],
            "per_seed_pooled_oof_auroc": seed_values,
            "seed_auroc_mean": seed_mean,
            "seed_auroc_sample_sd": seed_sd,
            "five_fold_ensemble_auroc": folds,
            "fold_auroc_mean": fold_mean,
            "fold_auroc_sample_sd": fold_sd,
            "five_seed_oof_ensemble": {
                "auroc": fine_rung["five_seed_ensemble"]["estimate"],
                "ci95": fine_rung["five_seed_ensemble"]["ci95_two_sided"],
            },
        }
        report_rows.append(
            {
                "task": task,
                "scope": "internal_oof",
                "population": "TCGA+SurGen-primary",
            }
        )
        per_cohort = {}
        external: dict[str, Any] = {"per_cohort": per_cohort}
        for scope_index, scope in enumerate(verifier.AIM3_FINE_EXTERNAL_REPORT_SCOPES):
            positive, negative = verifier.AIM3_FINE_EXTERNAL_EXPECTED_CENSUS[task][scope]
            block = _external_metric_fixture(
                positive,
                negative,
                base=0.4 + task_index / 100 + scope_index / 1000,
            )
            if scope == "strict_disjoint_combined":
                block.update(
                    {
                        "role": "headline patient-pooled AUROC",
                        "dual_role_exclusion": list(verifier.AIM3_FINE_EXTERNAL_DUAL_RIH_PATIENTS),
                        "all_eight_excluded_from_both_rih_roles": True,
                    }
                )
            if scope in verifier.AIM3_FINE_EXTERNAL_TARGETS:
                per_cohort[scope] = block
            else:
                external[scope] = block
            report_rows.append(
                {
                    "task": task,
                    "scope": "external_refit",
                    "population": scope,
                    "n": block["n_records"],
                    "positive": block["positive"],
                    "negative": block["negative"],
                    "seed_mean": block["seed_auroc_mean"],
                    "seed_sample_sd": block["seed_auroc_sample_sd"],
                    "ensemble_auroc": block["five_seed_refit_ensemble"]["auroc"],
                    "ci95": block["five_seed_refit_ensemble"]["ci95"],
                    "status": "ESTIMABLE",
                }
            )
        strict_positive, strict_negative = verifier.AIM3_FINE_EXTERNAL_EXPECTED_CENSUS[task][
            "strict_disjoint_combined"
        ]
        external["equal_cohort_macro_strict_disjoint"] = _external_metric_fixture(
            strict_positive,
            strict_negative,
            base=0.47 + task_index / 100,
            seed_key="per_seed_macro_auroc",
        )
        all_positive = sum(
            verifier.AIM3_FINE_EXTERNAL_EXPECTED_CENSUS[task][scope][0]
            for scope in verifier.AIM3_FINE_EXTERNAL_TARGETS
        )
        all_negative = sum(
            verifier.AIM3_FINE_EXTERNAL_EXPECTED_CENSUS[task][scope][1]
            for scope in verifier.AIM3_FINE_EXTERNAL_TARGETS
        )
        external["patient_role_pooled_clustered_sensitivity"] = _external_metric_fixture(
            all_positive,
            all_negative,
            base=0.48 + task_index / 100,
        )
        for scope in verifier.AIM3_FINE_EXTERNAL_SECONDARY_SCOPES:
            report_rows.append({"task": task, "scope": "external_refit", "population": scope})
        tasks[task] = {
            "display": task,
            "internal_oof": internal,
            "external_refit": external,
        }
    return {
        "schema_version": 1,
        "campaign": verifier.EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT,
        "status": "complete_after_sealed_zero_shot_inference",
        "encoder": "UNI-v1",
        "model_seeds": list(verifier.MODEL_SEEDS),
        "tasks": tasks,
        "report_rows": report_rows,
        "claim_boundary": {
            "external_target_use": "score-only after model freeze",
            "new_fits": 0,
            "target_calibration": False,
            "target_threshold_selection": False,
            "control_artifacts_absent_through_outcome_join": True,
            "controls_begin_only_after_external_analysis": True,
            "multiplicity_claim": False,
        },
    }


def _priority_sections() -> str:
    return "\n\n".join(f"{heading}\n\nComplete." for heading in verifier.PRIORITY_HEADINGS)


def _firewall_block() -> str:
    rows = "\n".join(
        f"| {cohort} | {role} | frozen_model_zero_shot_only |"
        for cohort, role in verifier.EXTERNAL_TARGET_ROLES.items()
    )
    return "\n".join((*verifier.FIREWALL_MARKERS, "", rows))


def _results(
    result: dict[str, Any],
    fine_external: dict[str, Any],
    k32: dict[str, str],
    curated_from: str,
) -> str:
    bindings = "\n".join(
        f"<!-- AIM3_SOURCE_VALUE {record['id']} "
        f"{json.dumps(record['value'], allow_nan=False, separators=(',', ':'))} -->"
        for record in result["report_bindings"]
        if record["metric"] != "consensus_verdict"
    )
    consensus = "\n".join(
        f"<!-- AIM3_CONSENSUS {rung} {result['repeated']['rungs'][rung]['consensus_verdict']} -->"
        for rung in sorted(verifier.AIM3_RUNG_CONTROL)
    )
    external_bindings = "\n".join(
        f"<!-- AIM3_SOURCE_VALUE {record['id']} "
        f"{json.dumps(record['value'], allow_nan=False, separators=(',', ':'))} -->"
        for record in verifier._fine_external_report_bindings(fine_external)
    )
    return f"""# FINAL-v13 results

{_firewall_block()}

{_priority_sections()}

### Controlling TCGA+SurGen-primary UNI-v1 five-seed ladder

{bindings}

### Frozen-refit fine-task external/test performance

{external_bindings}

### Repeated three-draw WT-control consensus

{consensus}

{verifier.AIM3_GATE_QUALIFICATION_MARKER}

{verifier.AIM3_CEILING_SCOPE_MARKER}

{verifier.AIM3_FIXED_NONOVERRIDE_MARKER}

### Legacy all-valid-patient Aim 3 — isolated secondary

{verifier.SECONDARY_MARKERS[0]}
{verifier.SOURCE_ONLY_RESULTS_EXCLUSION_MARKERS[0]}
{verifier.SECONDARY_MARKERS[1]}
{verifier.SOURCE_ONLY_RESULTS_EXCLUSION_MARKERS[1]}
{verifier.SECONDARY_MARKERS[2]}
{verifier.SOURCE_ONLY_RESULTS_EXCLUSION_MARKERS[2]}

{verifier.THEORETICAL_CEILING_NOT_RUN_MARKER}

{verifier.AIM4_BOUNDARY_MARKER}

reviews/k32/completed_review.md live SHA-256: `{k32["completed_review.md"]}`

reviews/k32/completed_review_structured.json curated-from SHA-256: `{curated_from}`
"""


def _side_document(title: str, *, include_firewall: bool) -> str:
    firewall = _firewall_block() + "\n\n" if include_firewall else ""
    return (
        f"# FINAL-v13 {title}\n\n{firewall}{_priority_sections()}\n\n"
        f"{verifier.THEORETICAL_CEILING_NOT_RUN_MARKER}\n\n"
        f"{verifier.AIM3_CEILING_SCOPE_MARKER}\n\n"
        f"{verifier.AIM3_FIXED_NONOVERRIDE_MARKER}\n"
    )


def _audit(
    sources: list[dict[str, Any]],
    pending: list[dict[str, Any]] | None = None,
    *,
    include_reconciliation_boundary: bool = True,
) -> str:
    rows = "\n".join(f"| `{row['id']}` | `{row['sha256']}` |" for row in sources)
    pending_rows = "\n".join(f"| `{row['id']}` | PENDING_PHASE2 |" for row in (pending or []))
    reconciliation_boundary = (
        f"\n{verifier.RECONCILIATION_PINS_BOUNDARY_MARKER}\n"
        f"\n{verifier.AIM3_TERMINAL_ANALYSIS_COMPLETION_MARKER}\n"
        if include_reconciliation_boundary
        else ""
    )
    return (
        _side_document("evidence audit", include_firewall=False)
        + f"\n{verifier.FAILED_V1_AUDIT_MARKER}\n"
        + reconciliation_boundary
        + f"\n## Exact source index\n\n| Source ID | SHA-256 |\n|---|---|\n{rows}\n"
        + f"\n## Pending Phase-2 source index\n\n{pending_rows}\n"
    )


def _phase1_results(
    result: dict[str, Any],
    fine_external: dict[str, Any],
    k32: dict[str, str],
    curated_from: str,
) -> str:
    bindings = "\n".join(
        f"<!-- AIM3_SOURCE_VALUE {record['id']} "
        f"{json.dumps(record['value'], allow_nan=False, separators=(',', ':'))} -->"
        for record in result["report_bindings"]
    )
    external_bindings = "\n".join(
        f"<!-- AIM3_SOURCE_VALUE {record['id']} "
        f"{json.dumps(record['value'], allow_nan=False, separators=(',', ':'))} -->"
        for record in verifier._fine_external_report_bindings(fine_external)
    )
    return f"""# FINAL-v13 results

{_firewall_block()}

{_priority_sections()}

### Phase-1 TCGA+SurGen-primary UNI-v1 five-seed fine ladders

{bindings}

### Frozen-refit fine-task external/test performance

{external_bindings}

### Repeated three-draw WT-control consensus — pending Phase 2

AIM3_PHASE1_STATUS: CANDIDATE_UNSEALED; FIXED_AND_REPEATED_CONTROLS_PENDING; DO_NOT_SEAL.

### Legacy all-valid-patient Aim 3 — isolated secondary

{verifier.SECONDARY_MARKERS[0]}
{verifier.SOURCE_ONLY_RESULTS_EXCLUSION_MARKERS[0]}
{verifier.SECONDARY_MARKERS[1]}
{verifier.SOURCE_ONLY_RESULTS_EXCLUSION_MARKERS[1]}
{verifier.SECONDARY_MARKERS[2]}
{verifier.SOURCE_ONLY_RESULTS_EXCLUSION_MARKERS[2]}

{verifier.AIM4_BOUNDARY_MARKER}

reviews/k32/completed_review.md live SHA-256: `{k32["completed_review.md"]}`

reviews/k32/completed_review_structured.json curated-from SHA-256: `{curated_from}`
"""


def _fixture(tmp_path: Path, *, parent_source_count: int = 2) -> verifier.BundlePaths:
    repo = tmp_path / "repo"
    parent_dir = repo / "reports/final_v12_1"
    final_dir = repo / "reports/final_v13"
    parent_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)

    parent_sources: list[dict[str, Any]] = []
    for index in range(parent_source_count):
        path = repo / f"evidence/parent-{index:03d}.json"
        _write_json(path, {"index": index})
        parent_sources.append(
            _source_record(
                repo,
                path,
                source_id=f"parent-source-{index:03d}",
                aim="Shared",
                experiment="parent fixture",
                role="parent_source",
            )
        )
    parent_manifest = {
        "schema_version": 2,
        "bundle": "final_v12_1",
        "status": verifier.PARENT_MANIFEST_STATUS,
        "artifacts": parent_sources,
        "pending_artifacts": [],
    }
    parent_manifest_path = parent_dir / verifier.SOURCE_MANIFEST_NAME
    _write_json(parent_manifest_path, parent_manifest)
    for name in verifier.REPORT_DOCUMENTS:
        (parent_dir / name).write_text(f"# fixture {name}\n", encoding="utf-8")
    parent_receipt = {
        "schema_version": 1,
        "bundle": "reports/final_v12_1",
        "status": verifier.PARENT_SEALED_STATUS,
        "source_count": parent_source_count,
        "checks": {"fixture": "PASS"},
        "source_manifest": _identity(
            parent_manifest_path, str(parent_manifest_path.relative_to(repo))
        ),
        "documents": {
            name: _identity(parent_dir / name, str((parent_dir / name).relative_to(repo)))
            for name in verifier.REPORT_DOCUMENTS
        },
        "authoritative_sources": parent_sources,
    }
    parent_receipt_path = parent_dir / verifier.FINAL_RECEIPT_NAME
    _write_json(parent_receipt_path, parent_receipt)

    run_root_path = repo / "campaign/aim3-run"
    run_root = str(run_root_path)
    aim3_path = run_root_path / "analysis/results.json"
    result = _aim3_result(run_root)
    _write_json(aim3_path, result)
    fine_result_path = run_root_path / "analysis/fine_results.json"
    fine_result = _aim3_fine_result(result)
    _write_json(fine_result_path, fine_result)
    fine_scheduler_path = run_root_path / "receipts/scheduler_fine.json"
    fine_job_keys = sorted(
        f"fine__{rung}__seed{seed}"
        for rung in verifier.AIM3_RUNG_CONTROL
        for seed in verifier.MODEL_SEEDS
    )
    _write_json(
        fine_scheduler_path,
        {
            "schema_version": 1,
            "status": "fine_phase_completed_rc0",
            "phase": "fine",
            "configured_max_parallel_chains": 6,
            "observed_max_parallel_chains": 6,
            "job_count": 25,
            "fit_accounting": dict(verifier.EXPECTED_FINE_TRAINING_COUNTS),
            "job_keys": fine_job_keys,
            "control_artifacts_absent": True,
            "events": [{"returncode": 0, "job_id": key} for key in fine_job_keys],
        },
    )
    fine_training_path = run_root_path / "receipts/training_complete_fine.json"
    _write_json(
        fine_training_path,
        {
            "schema_version": 1,
            "status": "fine_phase_complete_and_certified",
            "campaign": verifier.EXPECTED_AIM3_EXPERIMENT,
            "phase": "fine",
            "population": "tcga_surgen_primary",
            "encoder": "UNI-v1",
            "model_seeds": list(verifier.MODEL_SEEDS),
            "controls_status": "not_started",
            "fit_accounting": dict(verifier.EXPECTED_FINE_TRAINING_COUNTS),
            "scheduler": _identity(fine_scheduler_path, str(fine_scheduler_path.relative_to(repo))),
            "control_artifacts_absent": True,
            "external_development_cohorts": [],
            "job_receipts": [{"path": key} for key in fine_job_keys],
        },
    )

    fine_external_root = repo / "campaign/aim3-fine-external"
    external_controller = repo / "aim3_fine_external_controller.py"
    external_test = repo / "tests/test_aim3_fine_external_controller.py"
    external_controller.write_text("# governed external controller fixture\n", encoding="utf-8")
    external_test.parent.mkdir(parents=True, exist_ok=True)
    external_test.write_text("# governed external controller test fixture\n", encoding="utf-8")
    external_contract_path = fine_external_root / "contract.json"
    _write_json(
        external_contract_path,
        {
            "schema_version": 1,
            "campaign": verifier.EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT,
            "status": "prepared_label_blind_before_outcome_join",
            "output_root": str(fine_external_root),
            "training_root": run_root,
            "model_seeds": list(verifier.MODEL_SEEDS),
            "implementation": {
                "controller": _identity(external_controller, str(external_controller)),
                "controller_test": _identity(external_test, str(external_test)),
            },
            "source_target_overlap": {
                target: {"patients": 0, "slides": 0}
                for target in verifier.AIM3_FINE_EXTERNAL_TARGETS
            },
            "score_contract": {
                "jobs": 25,
                "score_artifacts": 25,
                "score_rows": 11_975,
                "new_fits": 0,
                "maximum_parallel_workers": 6,
            },
            "governance": {
                "external_test_data": [
                    "CPTAC-primary",
                    "Orion-primary",
                    "RIH-primary",
                    "RIH-metastatic",
                    "SurGen-metastatic",
                ],
                "phase_order": [
                    "fine_training_and_analysis",
                    "external_score_and_analysis",
                    "matched_and_repeated_controls",
                ],
                "inference_seal_exactly_once": True,
                "analysis_requires_inference_seal": True,
                **{
                    key: False
                    for key in (
                        "prepare_reads_target_outcomes",
                        "preflight_reads_target_outcomes",
                        "score_reads_target_outcomes",
                        "target_adaptation",
                        "target_calibration",
                        "target_encoder_or_construction_selection",
                        "target_fitting",
                        "target_model_or_checkpoint_selection",
                        "target_refitting",
                        "target_threshold_selection",
                    )
                },
            },
        },
    )
    external_preflight_path = fine_external_root / "receipts/deep_preflight.json"
    _write_json(
        external_preflight_path,
        {
            "schema_version": 1,
            "status": "ready_for_label_blind_inference",
            "maximum_parallel_workers": 6,
            "score_jobs": 25,
            "score_rows": 11_975,
            "authenticated_p75_checkpoints": 25,
            "target_outcomes_opened": False,
            "target_outcomes_present": False,
            "control_artifacts_absent": True,
            "checks": {"fixture_check": True},
            "contract": _identity(external_contract_path, str(external_contract_path)),
        },
    )
    score_events = [
        {
            "task": task,
            "seed": seed,
            "returncode": 0,
            "score_rows": 479,
            "execution_role": "label_blind_native_logit_inference",
            "started_unix_ns": index * 10 + 1,
            "completed_unix_ns": index * 10 + 9,
        }
        for index, (task, seed) in enumerate(
            (task, seed) for task in verifier.AIM3_FINE_TASKS for seed in verifier.MODEL_SEEDS
        )
    ]
    external_scoring_path = fine_external_root / "receipts/scoring_complete.json"
    _write_json(
        external_scoring_path,
        {
            "schema_version": 1,
            "status": "complete_before_outcome_join",
            "configured_max_workers": 6,
            "max_observed_parallel_inference_workers": 6,
            "score_jobs": 25,
            "score_rows": 11_975,
            "control_artifacts_absent": True,
            "inference_events": score_events,
            "contract": _identity(external_contract_path, str(external_contract_path)),
        },
    )
    external_inference_path = fine_external_root / "inference/inference_seal.json"
    _write_json(
        external_inference_path,
        {
            "schema_version": 1,
            "status": "sealed_before_outcome_join",
            "control_artifacts_absent_at_inference_seal": True,
            "target_outcomes_opened": False,
            "score_artifact_count": 25,
            "score_rows": 11_975,
            "score_artifacts": [
                {"task": task, "seed": seed, "rows": 479}
                for task in verifier.AIM3_FINE_TASKS
                for seed in verifier.MODEL_SEEDS
            ],
            "contract": _identity(external_contract_path, str(external_contract_path)),
            "preflight": _identity(external_preflight_path, str(external_preflight_path)),
        },
    )
    external_analysis_contract_path = fine_external_root / "analysis/contract.json"
    _write_json(
        external_analysis_contract_path,
        {
            "schema_version": 1,
            "status": "outcome_join_governed_after_inference_seal",
            "bootstrap_draws": 10_000,
            "bootstrap_seed": 20_260_828,
            "seal_status_observed_before_outcome_open": "sealed_before_outcome_join",
            "target_outcomes_opened_after_inference_seal": True,
            "control_artifacts_absent_at_outcome_join": True,
            "outcome_sources": {
                target: {"fixture": True} for target in verifier.AIM3_FINE_EXTERNAL_TARGETS
            },
            "contract": _identity(external_contract_path, str(external_contract_path)),
            "inference_seal": _identity(external_inference_path, str(external_inference_path)),
        },
    )
    external_patient_logits = fine_external_root / "analysis/patient_native_logits.parquet"
    external_bootstrap = fine_external_root / "analysis/bootstrap_distributions.npz"
    external_patient_logits.write_bytes(b"fixture patient logits\n")
    external_bootstrap.write_bytes(b"fixture bootstrap distributions\n")
    external_result_path = fine_external_root / "analysis/results.json"
    external_result = _aim3_fine_external_result(fine_result)
    external_result.update(
        {
            "analysis_contract": _identity(
                external_analysis_contract_path, str(external_analysis_contract_path)
            ),
            "inference_seal": _identity(external_inference_path, str(external_inference_path)),
            "patient_native_logits": _identity(
                external_patient_logits, str(external_patient_logits)
            ),
            "bootstrap_distributions": _identity(external_bootstrap, str(external_bootstrap)),
        }
    )
    _write_json(external_result_path, external_result)
    external_completion_path = fine_external_root / "analysis/analysis_completion.json"
    _write_json(
        external_completion_path,
        {
            "schema_version": 1,
            "status": "complete_after_sealed_zero_shot_inference",
            "bootstrap_array_count": 50,
            "report_row_count": 55,
            "analysis_contract": _identity(
                external_analysis_contract_path, str(external_analysis_contract_path)
            ),
            "patient_native_logits": _identity(
                external_patient_logits, str(external_patient_logits)
            ),
            "bootstrap_distributions": _identity(external_bootstrap, str(external_bootstrap)),
            "results": _identity(external_result_path, str(external_result_path)),
        },
    )
    controls_scheduler_path = run_root_path / "receipts/scheduler_controls.json"
    controls_events = [{"returncode": 0, "job_id": f"control-{index:03d}"} for index in range(100)]
    _write_json(
        controls_scheduler_path,
        {
            "schema_version": 1,
            "status": "control_phase_completed_rc0",
            "phase": "controls",
            "configured_max_parallel_chains": 6,
            "observed_max_parallel_chains": 6,
            "job_count": 100,
            "fit_accounting": dict(verifier.EXPECTED_CONTROL_TRAINING_COUNTS),
            "fixed_scheduled_before_repeated": True,
            "events": controls_events,
        },
    )
    scheduler_path = run_root_path / "receipts/scheduler.json"
    _write_json(
        scheduler_path,
        {
            "schema_version": 1,
            "status": "completed_phased_rc0",
            "configured_max_parallel_chains": 6,
            "observed_max_parallel_chains": 6,
            "job_count": 125,
            "fit_accounting": dict(verifier.EXPECTED_TRAINING_COUNTS),
            "fixed_ladder_scheduled_before_repeated": True,
            "phase_order": ["fine", "fine_analysis_candidate", "controls"],
            "fine_scheduler": _identity(
                fine_scheduler_path, str(fine_scheduler_path.relative_to(repo))
            ),
            "controls_scheduler": _identity(
                controls_scheduler_path, str(controls_scheduler_path.relative_to(repo))
            ),
            "events": [
                *[{"returncode": 0, "job_id": key} for key in fine_job_keys],
                *controls_events,
            ],
        },
    )
    training_path = run_root_path / "receipts/training_complete.json"
    _write_json(
        training_path,
        {
            "schema_version": 1,
            "status": "complete_and_certified",
            "campaign": verifier.EXPECTED_AIM3_EXPERIMENT,
            "population": "tcga_surgen_primary",
            "encoder": "UNI-v1",
            "model_seeds": list(verifier.MODEL_SEEDS),
            "wt_draw_seeds": list(verifier.WT_DRAW_SEEDS),
            "fit_accounting": dict(verifier.EXPECTED_TRAINING_COUNTS),
            "external_development_cohorts": [],
            "job_receipts": [{"path": f"job-{index:03d}"} for index in range(125)],
            "scheduler": _identity(scheduler_path, str(scheduler_path.relative_to(repo))),
        },
    )
    extension_sources = [
        _source_record(
            repo,
            fine_result_path,
            source_id="aim3-fine-results",
            aim="Aim 3",
            experiment="source-primary fine phase",
            role="aim3_fine_results",
        ),
        _source_record(
            repo,
            fine_scheduler_path,
            source_id="aim3-fine-scheduler",
            aim="Aim 3",
            experiment="source-primary fine phase",
            role="fine_scheduler_evidence",
        ),
        _source_record(
            repo,
            fine_training_path,
            source_id="aim3-fine-training",
            aim="Aim 3",
            experiment="source-primary fine phase",
            role="fine_training_completion",
        ),
        _source_record(
            repo,
            external_completion_path,
            source_id="aim3-fine-external-completion",
            aim="Aim 3",
            experiment="source-primary fine external",
            role="fine_external_analysis_completion",
        ),
        _source_record(
            repo,
            external_contract_path,
            source_id=verifier.EXPECTED_AIM3_FINE_EXTERNAL_CONTRACT_SOURCE_ID,
            aim="Aim 3",
            experiment="source-primary fine external",
            role="fine_external_campaign_contract",
        ),
        _source_record(
            repo,
            external_controller,
            source_id=verifier.EXPECTED_AIM3_FINE_EXTERNAL_CONTROLLER_SOURCE_ID,
            aim="Aim 3",
            experiment="source-primary fine external",
            role="fine_external_campaign_controller",
        ),
        _source_record(
            repo,
            external_inference_path,
            source_id="aim3-fine-external-inference",
            aim="Aim 3",
            experiment="source-primary fine external",
            role="fine_external_inference_seal",
        ),
        _source_record(
            repo,
            external_preflight_path,
            source_id=verifier.EXPECTED_AIM3_FINE_EXTERNAL_PREFLIGHT_SOURCE_ID,
            aim="Aim 3",
            experiment="source-primary fine external",
            role="fine_external_deep_preflight",
        ),
        _source_record(
            repo,
            external_result_path,
            source_id="aim3-fine-external-results",
            aim="Aim 3",
            experiment="source-primary fine external",
            role="fine_external_results",
        ),
        _source_record(
            repo,
            external_scoring_path,
            source_id="aim3-fine-external-scoring",
            aim="Aim 3",
            experiment="source-primary fine external",
            role="fine_external_scoring_completion",
        ),
        _source_record(
            repo,
            external_test,
            source_id=verifier.EXPECTED_AIM3_FINE_EXTERNAL_TEST_SOURCE_ID,
            aim="Aim 3",
            experiment="source-primary fine external",
            role="fine_external_campaign_contract_test",
        ),
        _source_record(
            repo,
            controls_scheduler_path,
            source_id="aim3-controls-scheduler",
            aim="Aim 3",
            experiment="source-primary control phase",
            role="controls_scheduler_evidence",
        ),
        _source_record(
            repo,
            aim3_path,
            source_id="aim3-new-results",
            aim="Aim 3",
            experiment="source-primary ladder",
            role="aim3_results",
        ),
        _source_record(
            repo,
            scheduler_path,
            source_id="aim3-new-scheduler",
            aim="Aim 3",
            experiment="source-primary ladder",
            role="scheduler_evidence",
        ),
        _source_record(
            repo,
            training_path,
            source_id="aim3-new-training",
            aim="Aim 3",
            experiment="source-primary ladder",
            role="training_completion",
        ),
    ]

    k32_root = repo / "reviews/k32"
    evidence_names = tuple(verifier.EXPECTED_AIM4_K32_SHA256)
    sidecar_names = tuple(verifier.EXPECTED_AIM4_K32_SIDECAR_SHA256)
    for name in (*evidence_names, *sidecar_names):
        path = k32_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name != "completed_review_structured.json":
            path.write_bytes(f"fixture bytes for {name}\n".encode())
    curated_from = "c" * 64
    montage_sha = {
        montage: _sha(k32_root / f"montages/{montage}.jpg")
        for montage in ("M04", "M05", "M07", "M11")
    }
    structured = {
        "schema_version": 1,
        "source_document": "completed_review.md",
        "source_sha256": curated_from,
        "extraction_status": "curated_from_completed_blinded_followup",
        "reviewer_metadata": {},
        "provenance": {
            "base_form_sha256": _sha(k32_root / "review_form.csv"),
            "montage_sha256": montage_sha,
        },
        "montages": {name: {} for name in ("M04", "M05", "M07", "M11")},
        "cross_montage": {},
        "global_quality_flags": ["fixture limitation"],
    }
    _write_json(k32_root / "completed_review_structured.json", structured)
    k32_sha = {name: _sha(k32_root / name) for name in evidence_names}
    sidecar_sha = {name: _sha(k32_root / name) for name in sidecar_names}
    for index, name in enumerate(sorted(evidence_names)):
        extension_sources.append(
            _source_record(
                repo,
                k32_root / name,
                source_id=f"aim4-k32-source-{index:02d}",
                aim="Aim 4",
                experiment="reviews k32",
                role="aim4_k32_review_provenance",
            )
        )

    for index in range(56):
        path = repo / f"evidence/full-extension-{index:02d}.json"
        _write_json(path, {"fixture_extension": index})
        extension_sources.append(
            _source_record(
                repo,
                path,
                source_id=f"fixture-full-extension-{index:02d}",
                aim="Shared",
                experiment="full reconciliation fixture",
                role="fixture_extension",
            )
        )

    all_sources = sorted([*parent_sources, *extension_sources], key=lambda row: row["id"])
    extension_ids = tuple(sorted(row["id"] for row in extension_sources))
    final_manifest = {
        "schema_version": 2,
        "bundle": "final_v13",
        "status": verifier.FINAL_MANIFEST_STATUS,
        "artifacts": all_sources,
        "pending_artifacts": [],
    }
    _write_json(final_dir / verifier.SOURCE_MANIFEST_NAME, final_manifest)
    (final_dir / "Experimental_Setup.md").write_text(
        _side_document("experimental setup", include_firewall=True), encoding="utf-8"
    )
    (final_dir / "Results.md").write_text(
        _results(result, external_result, k32_sha, curated_from), encoding="utf-8"
    )
    (final_dir / "Audit.md").write_text(_audit(all_sources), encoding="utf-8")

    parent_verifier = repo / "tools/final_v12_1_bundle_receipt.py"
    parent_test = repo / "tests/test_final_v12_1_bundle_receipt.py"
    current_verifier = repo / "tools/final_v13_bundle_receipt.py"
    current_test = repo / "tests/test_final_v13_bundle_receipt.py"
    for path, content in (
        (parent_verifier, "# parent verifier fixture\n"),
        (parent_test, "# parent test fixture\n"),
        (current_verifier, "# child verifier fixture\n"),
        (current_test, "# child test fixture\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    reconciliation_pins = final_dir / verifier.RECONCILIATION_PINS_NAME
    _write_json(
        reconciliation_pins,
        {
            "schema_version": 1,
            "bundle": "final_v13",
            "status": verifier.RECONCILIATION_PINS_STATUS,
            "parent_final_v12_1": {
                "status": verifier.PARENT_SEALED_STATUS,
                "receipt_sha256": _sha(parent_receipt_path),
                "source_manifest_sha256": _sha(parent_manifest_path),
            },
            "extension_source_count": 87,
            "extension_source_ids": list(extension_ids),
            "aim3_source_ids": {
                "results": "aim3-new-results",
                "scheduler": "aim3-new-scheduler",
                "training_completion": "aim3-new-training",
                "fine_results": "aim3-fine-results",
                "fine_scheduler": "aim3-fine-scheduler",
                "fine_training_completion": "aim3-fine-training",
                "controls_scheduler": "aim3-controls-scheduler",
                "fine_external_results": "aim3-fine-external-results",
                "fine_external_analysis_completion": "aim3-fine-external-completion",
                "fine_external_inference_seal": "aim3-fine-external-inference",
                "fine_external_scoring_completion": "aim3-fine-external-scoring",
            },
            "documents": {name: _sha(final_dir / name) for name in verifier.REPORT_DOCUMENTS},
            "verifier": {"sha256": _sha(current_verifier)},
            "verifier_test": {"sha256": _sha(current_test)},
            "scientific_source_manifest_membership": (verifier.RECONCILIATION_PINS_MEMBERSHIP),
        },
    )

    return verifier.BundlePaths(
        repo=repo,
        final_v13=final_dir,
        destination=final_dir / verifier.FINAL_RECEIPT_NAME,
        reconciliation_pins=reconciliation_pins,
        verifier_code=current_verifier,
        verifier_test=current_test,
        parent_dir=parent_dir,
        parent_receipt=parent_receipt_path,
        parent_manifest=parent_manifest_path,
        parent_verifier=parent_verifier,
        parent_test=parent_test,
        expected_parent_receipt_sha256=_sha(parent_receipt_path),
        expected_parent_manifest_sha256=_sha(parent_manifest_path),
        expected_parent_verifier_sha256=_sha(parent_verifier),
        expected_parent_test_sha256=_sha(parent_test),
        expected_parent_document_sha256={
            name: _sha(parent_dir / name) for name in verifier.REPORT_DOCUMENTS
        },
        expected_final_document_sha256={
            name: _sha(final_dir / name) for name in verifier.REPORT_DOCUMENTS
        },
        expected_extension_source_ids=extension_ids,
        aim3_results_source_id="aim3-new-results",
        aim3_scheduler_source_id="aim3-new-scheduler",
        aim3_training_source_id="aim3-new-training",
        aim3_fine_results_source_id="aim3-fine-results",
        aim3_fine_scheduler_source_id="aim3-fine-scheduler",
        aim3_fine_training_source_id="aim3-fine-training",
        aim3_controls_scheduler_source_id="aim3-controls-scheduler",
        aim3_fine_external_results_source_id="aim3-fine-external-results",
        aim3_fine_external_completion_source_id="aim3-fine-external-completion",
        aim3_fine_external_inference_seal_source_id="aim3-fine-external-inference",
        aim3_fine_external_scoring_source_id="aim3-fine-external-scoring",
        expected_aim3_experiment=verifier.EXPECTED_AIM3_EXPERIMENT,
        expected_aim3_run_root=run_root,
        expected_aim3_fine_external_experiment=(verifier.EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT),
        expected_aim3_fine_external_run_root=str(fine_external_root),
        aim4_k32_root=k32_root,
        expected_aim4_k32_sha256=k32_sha,
        expected_aim4_k32_sidecar_sha256=sidecar_sha,
        expected_aim4_curated_from_sha256=curated_from,
        expected_parent_source_count=parent_source_count,
        replay_parent_verifier=False,
    )


def _phase1_fixture(tmp_path: Path) -> verifier.BundlePaths:
    paths = _fixture(tmp_path)
    paths.reconciliation_pins.unlink()
    manifest_path = paths.final_v13 / verifier.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    pending_ids = {
        paths.aim3_results_source_id,
        paths.aim3_scheduler_source_id,
        paths.aim3_training_source_id,
        paths.aim3_controls_scheduler_source_id,
    }
    pending_full_records = sorted(
        [row for row in manifest["artifacts"] if row["id"] in pending_ids],
        key=lambda row: row["id"],
    )
    material = [row for row in manifest["artifacts"] if row["id"] not in pending_ids]
    for record in pending_full_records:
        path = paths.repo / record["path"]
        path.unlink()
    pending = [
        {key: value for key, value in record.items() if key not in {"size_bytes", "sha256"}}
        for record in pending_full_records
    ]
    manifest.update(
        {
            "status": verifier.PHASE1_MANIFEST_STATUS,
            "artifacts": material,
            "pending_artifacts": pending,
        }
    )
    _write_json(manifest_path, manifest)

    fine_path = Path(paths.expected_aim3_run_root) / "analysis/fine_results.json"
    fine_result = json.loads(fine_path.read_text())
    fine_external_path = Path(paths.expected_aim3_fine_external_run_root) / "analysis/results.json"
    fine_external = json.loads(fine_external_path.read_text())
    k32 = dict(paths.expected_aim4_k32_sha256)
    (paths.final_v13 / "Results.md").write_text(
        _phase1_results(
            fine_result,
            fine_external,
            k32,
            paths.expected_aim4_curated_from_sha256,
        ),
        encoding="utf-8",
    )
    (paths.final_v13 / "Audit.md").write_text(
        _audit(material, pending, include_reconciliation_boundary=False),
        encoding="utf-8",
    )
    return replace(
        paths,
        expected_final_document_sha256={
            name: verifier.UNFROZEN for name in verifier.REPORT_DOCUMENTS
        },
    )


def _rewrite_source_json(
    paths: verifier.BundlePaths,
    source_id: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    manifest_path = paths.final_v13 / verifier.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    record = next(row for row in manifest["artifacts"] if row["id"] == source_id)
    source_path = paths.repo / record["path"]
    value = json.loads(source_path.read_text())
    mutate(value)
    _write_json(source_path, value)
    record.update(_identity(source_path, record["path"]))
    _write_json(manifest_path, manifest)


def test_production_parent_pins_match_sealed_final_v12_1() -> None:
    paths = verifier.default_paths()
    assert verifier.sha256_file(paths.parent_receipt) == verifier.EXPECTED_PARENT_RECEIPT_SHA256
    assert verifier.sha256_file(paths.parent_manifest) == verifier.EXPECTED_PARENT_MANIFEST_SHA256
    assert verifier.sha256_file(paths.parent_verifier) == verifier.EXPECTED_PARENT_VERIFIER_SHA256
    assert verifier.sha256_file(paths.parent_test) == verifier.EXPECTED_PARENT_TEST_SHA256
    assert {
        name: verifier.sha256_file(paths.parent_dir / name) for name in verifier.REPORT_DOCUMENTS
    } == verifier.EXPECTED_PARENT_DOCUMENT_SHA256
    assert verifier.EXPECTED_PARENT_SOURCE_COUNT == 142


def test_production_child_membership_is_hard_coded_and_preseal_pins_are_frozen() -> None:
    paths = verifier.default_paths()
    assert set(paths.expected_final_document_sha256.values()) == {verifier.UNFROZEN}
    assert paths.expected_extension_source_ids == verifier.EXPECTED_EXTENSION_SOURCE_IDS
    assert len(paths.expected_extension_source_ids) == 87
    assert paths.aim3_results_source_id == "aim3-source-primary-results"
    assert paths.aim3_scheduler_source_id == "aim3-source-primary-scheduler"
    assert paths.aim3_training_source_id == "aim3-source-primary-training-completion"
    assert paths.aim3_fine_results_source_id == "aim3-source-primary-fine-results"
    assert paths.aim3_fine_scheduler_source_id == "aim3-source-primary-fine-scheduler"
    assert paths.aim3_fine_training_source_id == "aim3-source-primary-fine-training-completion"
    assert paths.aim3_controls_scheduler_source_id == "aim3-source-primary-controls-scheduler"
    assert paths.aim3_fine_external_results_source_id == "aim3-source-primary-fine-external-results"
    assert (
        paths.aim3_fine_external_completion_source_id
        == "aim3-source-primary-fine-external-analysis-completion"
    )
    assert "_v2_20260827" in paths.expected_aim3_run_root
    pins = json.loads(paths.reconciliation_pins.read_text(encoding="utf-8"))
    assert pins["status"] == verifier.RECONCILIATION_PINS_STATUS
    assert pins["extension_source_ids"] == list(verifier.EXPECTED_EXTENSION_SOURCE_IDS)
    assert set(pins["documents"]) == set(verifier.REPORT_DOCUMENTS)
    assert not paths.destination.exists()


def test_production_aim4_reviews_k32_pins_match_live_provenance() -> None:
    paths = verifier.default_paths()
    assert {
        name: verifier.sha256_file(paths.aim4_k32_root / name)
        for name in verifier.EXPECTED_AIM4_K32_SHA256
    } == verifier.EXPECTED_AIM4_K32_SHA256
    assert {
        name: verifier.sha256_file(paths.aim4_k32_root / name)
        for name in verifier.EXPECTED_AIM4_K32_SIDECAR_SHA256
    } == verifier.EXPECTED_AIM4_K32_SIDECAR_SHA256


def test_phase1_candidate_is_read_only_verified_and_never_seal_ready(tmp_path: Path) -> None:
    paths = _phase1_fixture(tmp_path)
    checked = verifier.check_bundle(paths)
    assert checked["status"] == "PHASE1_FINE_EXTERNAL_CANDIDATE_VERIFIED_UNSEALED"
    assert checked["seal_ready"] is False
    assert checked["document_pins"] == "UNFROZEN_PHASE1_CANDIDATE"
    assert checked["pending_phase2_source_count"] == 4
    assert checked["aim3_fine"]["numeric_binding_count"] == 5
    assert checked["aim3_fine_external"]["numeric_binding_count"] == 370
    assert checked["aim3_fine_external"]["new_fits"] == 0
    assert checked["aim3_fine"]["execution"]["training_counts"] == {
        "task_variants": 5,
        "chains": 25,
        "oof_folds": 125,
        "p75_refits": 25,
        "physical_mil_fits": 150,
    }
    assert "three_draw_consensus_pending" in checked["publication_blockers"]
    assert not paths.destination.exists()
    with pytest.raises(verifier.BundleVerificationError, match="reconciliation_pins.json"):
        verifier.build_receipt(paths)
    with pytest.raises(verifier.BundleVerificationError, match="reconciliation_pins.json"):
        verifier.seal(paths)
    assert not paths.destination.exists()


def test_phase1_candidate_rejects_materialized_control_or_full_artifact(tmp_path: Path) -> None:
    paths = _phase1_fixture(tmp_path)
    forbidden = Path(paths.expected_aim3_run_root) / "receipts/scheduler_controls.json"
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        verifier.BundleVerificationError, match="pending Phase-2 source already exists"
    ):
        verifier.check_bundle(paths)


def test_phase1_candidate_rejects_invented_consensus(tmp_path: Path) -> None:
    paths = _phase1_fixture(tmp_path)
    results = paths.final_v13 / "Results.md"
    results.write_text(
        results.read_text() + "\n<!-- AIM3_CONSENSUS codon CONSENSUS_CEILING -->\n",
        encoding="utf-8",
    )
    with pytest.raises(verifier.BundleVerificationError, match="consensus bindings drift"):
        verifier.check_bundle(paths)


def test_phase1_candidate_rejects_full_reconciliation_pins(tmp_path: Path) -> None:
    paths = _phase1_fixture(tmp_path)
    paths.reconciliation_pins.write_text("{}\n", encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="must not contain"):
        verifier.check_bundle(paths)


def test_full_and_incremental_source_authorities_remain_separate() -> None:
    assert len(verifier.EXPECTED_PHASE1_BASE_EXTENSION_SOURCE_IDS) == 47
    assert len(verifier.EXPECTED_INCREMENTAL_ADDITION_SOURCE_IDS) == 40
    assert len(verifier.EXPECTED_EXTENSION_SOURCE_IDS) == 87
    assert verifier.EXPECTED_FINAL_SOURCE_COUNT == 229
    assert len(verifier.EXPECTED_INCREMENTAL_EXTENSION_SOURCE_IDS) == 87
    assert (
        verifier.EXPECTED_INCREMENTAL_EXTENSION_SOURCE_IDS == verifier.EXPECTED_EXTENSION_SOURCE_IDS
    )
    assert set(verifier.EXPECTED_EXTENSION_SOURCE_IDS) == set(
        verifier.EXPECTED_PHASE1_BASE_EXTENSION_SOURCE_IDS
    ) | set(verifier.EXPECTED_INCREMENTAL_ADDITION_SOURCE_IDS)
    assert set(verifier.EXPECTED_PHASE1_BASE_EXTENSION_SOURCE_IDS).isdisjoint(
        verifier.EXPECTED_INCREMENTAL_ADDITION_SOURCE_IDS
    )


def test_production_full_candidate_is_installed_unsealed_and_ready() -> None:
    paths = verifier.default_paths()
    checked = verifier.check_bundle(paths)
    assert checked["status"] == "READY_TO_SEAL"
    assert checked["published_receipt_present"] is False
    assert checked["source_count"] == 229
    assert checked["aim2_target_internal_few_shot"]["numeric_binding_count"] == 348
    pooled = checked["aim2_conventional_external_primary_pool"]
    assert pooled["patients"] == 247
    assert pooled["auroc"] == 0.7445051240560949
    assert not paths.destination.exists()


def test_full_bundle_requires_regular_reconciliation_pins(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    original = paths.reconciliation_pins.read_bytes()
    paths.reconciliation_pins.unlink()
    with pytest.raises(verifier.BundleVerificationError, match="expected a regular file"):
        verifier.check_bundle(paths)
    decoy = paths.final_v13 / "pins-decoy.json"
    decoy.write_bytes(original)
    paths.reconciliation_pins.symlink_to(decoy)
    with pytest.raises(verifier.BundleVerificationError, match="symlink"):
        verifier.check_bundle(paths)


@pytest.mark.parametrize(
    "marker",
    (
        verifier.THEORETICAL_CEILING_NOT_RUN_MARKER,
        verifier.AIM3_CEILING_SCOPE_MARKER,
        verifier.AIM3_FIXED_NONOVERRIDE_MARKER,
    ),
)
def test_full_bundle_rejects_missing_ceiling_scope_disclosure(
    tmp_path: Path,
    marker: str,
) -> None:
    paths = _fixture(tmp_path)
    results = paths.final_v13 / "Results.md"
    results.write_text(
        results.read_text(encoding="utf-8").replace(marker, "", 1),
        encoding="utf-8",
    )
    pins = json.loads(paths.reconciliation_pins.read_text(encoding="utf-8"))
    pins["documents"]["Results.md"] = _sha(results)
    _write_json(paths.reconciliation_pins, pins)
    paths = replace(paths, expected_final_document_sha256=dict(pins["documents"]))
    with pytest.raises(verifier.BundleVerificationError, match="must contain exactly one"):
        verifier.check_bundle(paths)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda pins: pins["parent_final_v12_1"].__setitem__("status", "DRIFT"),
            "pin identity is invalid",
        ),
        (
            lambda pins: pins.__setitem__("extension_source_count", 38),
            "pin identity is invalid",
        ),
        (
            lambda pins: pins["documents"].pop("Audit.md"),
            "pin identity is invalid",
        ),
        (
            lambda pins: pins["verifier_test"].__setitem__("sha256", "0" * 64),
            "verifier-test SHA-256 drift",
        ),
    ),
)
def test_full_reconciliation_pins_fail_closed_on_identity_drift(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    paths = _fixture(tmp_path)
    pins = json.loads(paths.reconciliation_pins.read_text(encoding="utf-8"))
    mutate(pins)
    _write_json(paths.reconciliation_pins, pins)
    with pytest.raises(verifier.BundleVerificationError, match=message):
        verifier.check_bundle(paths)


def test_reconciliation_pins_cannot_choose_extension_or_aim3_ids(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    pins = json.loads(paths.reconciliation_pins.read_text(encoding="utf-8"))
    original = pins["extension_source_ids"][-1]
    pins["extension_source_ids"][-1] = "zz-arbitrary-extension"
    pins["extension_source_ids"].sort()
    pins["aim3_source_ids"]["results"] = "zz-arbitrary-extension"
    _write_json(paths.reconciliation_pins, pins)
    with pytest.raises(
        verifier.BundleVerificationError,
        match="cannot choose extension membership",
    ):
        verifier.check_bundle(paths)

    pins["extension_source_ids"].remove("zz-arbitrary-extension")
    pins["extension_source_ids"].append(original)
    pins["extension_source_ids"].sort()
    pins["aim3_source_ids"]["results"] = paths.aim3_fine_results_source_id
    _write_json(paths.reconciliation_pins, pins)
    with pytest.raises(
        verifier.BundleVerificationError,
        match="cannot choose Aim-3 semantic IDs",
    ):
        verifier.check_bundle(paths)


@pytest.mark.parametrize("mode", ("missing", "extra"))
def test_phase1_pending_roster_is_exactly_four(
    tmp_path: Path,
    mode: str,
) -> None:
    paths = _phase1_fixture(tmp_path)
    manifest_path = paths.final_v13 / verifier.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mode == "missing":
        manifest["pending_artifacts"].pop()
    else:
        record = next(
            row for row in manifest["artifacts"] if row["id"].startswith("fixture-full-extension-")
        )
        manifest["artifacts"].remove(record)
        manifest["pending_artifacts"].append(
            {key: value for key, value in record.items() if key not in {"size_bytes", "sha256"}}
        )
        manifest["pending_artifacts"].sort(key=lambda row: row["id"])
    _write_json(manifest_path, manifest)
    with pytest.raises(
        verifier.BundleVerificationError,
        match="pending roster is not exactly the four",
    ):
        verifier.check_bundle(paths)


@pytest.mark.parametrize(
    "pin_name",
    ("EXPECTED_FULL_RECONCILER_SHA256", "EXPECTED_PHASE1_CANDIDATE_HELPER_SHA256"),
)
def test_trusted_reconciliation_helper_drift_fails_before_import(
    monkeypatch: pytest.MonkeyPatch,
    pin_name: str,
) -> None:
    monkeypatch.setattr(verifier, pin_name, "0" * 64)
    with pytest.raises(verifier.BundleVerificationError, match="SHA-256 drift"):
        verifier._trusted_reconciliation_generator_identities()


def test_direct_seal_regenerates_bytes_after_check_and_rejects_repinned_false_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, parent_source_count=142)
    paths = replace(
        paths,
        expected_final_document_sha256={
            name: verifier.UNFROZEN for name in verifier.REPORT_DOCUMENTS
        },
    )
    original_payload = {
        "status": "FULL_RECONCILIATION_RENDERED_READ_ONLY",
        "scientific_source_count": 229,
        "extension_source_count": 87,
        "pending_source_count": 0,
        "retained_target_internal_binding_count": 348,
        "aim2_conventional_primary_pool_binding_count": 4,
        "source_manifest": json.loads(
            (paths.final_v13 / verifier.SOURCE_MANIFEST_NAME).read_text(encoding="utf-8")
        ),
        "documents": {
            name: (paths.final_v13 / name).read_text(encoding="utf-8")
            for name in verifier.REPORT_DOCUMENTS
        },
        "reconciliation_pins": json.loads(paths.reconciliation_pins.read_text(encoding="utf-8")),
    }
    identities = verifier._trusted_reconciliation_generator_identities()
    fake_reconciler = types.SimpleNamespace(build_full_payload=lambda selected: original_payload)
    monkeypatch.setattr(verifier, "_is_production_bundle", lambda selected: True)
    monkeypatch.setattr(
        verifier,
        "_validate_aim2_conventional_primary_pool",
        lambda selected, sources: {"bindings": []},
    )
    monkeypatch.setattr(
        verifier,
        "_load_trusted_full_reconciler",
        lambda: (fake_reconciler, identities),
    )
    monkeypatch.setattr(
        verifier,
        "EXPECTED_EXTENSION_SOURCE_IDS",
        tuple(paths.expected_extension_source_ids),
    )
    assert verifier.check_bundle(paths)["status"] == "READY_TO_SEAL"

    results = paths.final_v13 / "Results.md"
    results.write_text(
        results.read_text(encoding="utf-8")
        + "\nScientifically false inherited claim: AUROC was 0.9999.\n",
        encoding="utf-8",
    )
    pins = json.loads(paths.reconciliation_pins.read_text(encoding="utf-8"))
    pins["documents"]["Results.md"] = _sha(results)
    _write_json(paths.reconciliation_pins, pins)
    with pytest.raises(
        verifier.BundleVerificationError,
        match="differs from trusted deterministic reconciliation",
    ):
        verifier.seal(paths)
    assert not paths.destination.exists()


def test_custom_paths_cannot_target_production_receipt(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths = replace(
        paths,
        destination=verifier.FINAL_V13 / verifier.FINAL_RECEIPT_NAME,
    )
    with pytest.raises(
        verifier.BundleVerificationError,
        match="requires the exact trusted production paths",
    ):
        verifier.build_receipt(paths)


def test_receipt_is_deterministic_read_only_and_complete(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = verifier.build_receipt(paths)
    second = verifier.build_receipt(paths)
    assert first == second
    assert verifier._receipt_bytes(first) == verifier._receipt_bytes(second)
    assert first["source_count"] == paths.expected_parent_source_count + len(
        paths.expected_extension_source_ids
    )
    assert first["parent_final_v12_1"]["authoritative_source_count"] == 2
    assert first["aim3_tcga_surgen_primary"]["model_seeds"] == [42, 43, 44, 45, 46]
    assert first["aim3_tcga_surgen_primary"]["wt_draw_seeds"] == [
        20260823,
        20260824,
        20260825,
    ]
    assert first["aim3_tcga_surgen_primary"]["execution"]["observed_peak_parallel_chains"] == 6
    assert (
        first["reconciliation_generator"]["full_reconciler"]["sha256"]
        == verifier.EXPECTED_FULL_RECONCILER_SHA256
    )
    assert (
        first["reconciliation_generator"]["phase1_candidate_helper"]["sha256"]
        == verifier.EXPECTED_PHASE1_CANDIDATE_HELPER_SHA256
    )
    assert first["checks"]["retained_target_internal_bindings_source_replayed"] == ("NOT_PRESENT")
    assert (
        first["checks"]["aim2_conventional_primary_patient_pool_source_replayed"]
        == "NOT_APPLICABLE_SYNTHETIC_FIXTURE"
    )
    assert all(
        value == "PASS"
        for key, value in first["checks"].items()
        if key
        not in {
            "retained_target_internal_bindings_source_replayed",
            "aim2_conventional_primary_patient_pool_source_replayed",
        }
    )
    assert "created_utc" not in first
    assert not paths.destination.exists()


def test_final_manifest_uses_strict_json(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = paths.final_v13 / verifier.SOURCE_MANIFEST_NAME
    manifest.write_text('{"schema_version":2,"schema_version":2}\n', encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="duplicate key"):
        verifier.build_receipt(paths)


def test_every_inherited_and_extension_source_is_directly_rehashed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths.repo / "evidence/parent-001.json").write_text("drift\n", encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="source identity drift"):
        verifier.build_receipt(paths)


def test_source_symlink_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    source = Path(paths.expected_aim3_run_root) / "analysis/results.json"
    real = source.with_suffix(".real")
    source.rename(real)
    source.symlink_to(real)
    with pytest.raises(verifier.BundleVerificationError, match="symlink"):
        verifier.build_receipt(paths)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            verifier.PRIORITY_HEADINGS[0],
            "## Priority 1 — changed",
            "Priority 1",
        ),
        (
            verifier.FIREWALL_MARKERS[1],
            "External cohorts may tune the model.",
            "EXTERNAL_TEST_DATA",
        ),
        (
            "| Orion | external_cross_protocol_test | frozen_model_zero_shot_only |",
            "| Orion | development | model_selection |",
            "external_cross_protocol_test",
        ),
        (
            verifier.SECONDARY_MARKERS[2],
            "CONTROLLING_ALL_VALID",
            "SECONDARY_LEGACY_ALL_VALID",
        ),
    ),
)
def test_priority_firewall_roles_and_secondary_status_fail_closed(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    paths = _fixture(tmp_path)
    results = paths.final_v13 / "Results.md"
    text = results.read_text()
    assert old in text
    results.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match=message):
        verifier.build_receipt(paths)


@pytest.mark.parametrize(
    ("source_id", "mutate", "message"),
    (
        (
            "aim3-new-results",
            lambda value: value.__setitem__("encoder", "Virchow2-CLS"),
            "result identity",
        ),
        (
            "aim3-new-results",
            lambda value: value.__setitem__("model_seeds", [42, 43, 44]),
            "result identity",
        ),
        (
            "aim3-new-results",
            lambda value: value["repeated"]["rungs"]["codon"]["draws"].pop("20260825"),
            "exact three WT draws",
        ),
        (
            "aim3-new-results",
            lambda value: value["repeated"]["rungs"]["codon"].__setitem__(
                "consensus_verdict", "NO_CEILING_CONSENSUS"
            ),
            "consensus does not replay",
        ),
        (
            "aim3-new-scheduler",
            lambda value: value.__setitem__("observed_max_parallel_chains", 5),
            "configured and observed six-parallel",
        ),
        (
            "aim3-new-training",
            lambda value: value["fit_accounting"].__setitem__("physical_mil_fits", 749),
            "terminal training accounting",
        ),
    ),
)
def test_aim3_scope_seed_draw_consensus_scheduler_and_accounting_fail_closed(
    tmp_path: Path,
    source_id: str,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    paths = _fixture(tmp_path)
    _rewrite_source_json(paths, source_id, mutate)
    with pytest.raises(verifier.BundleVerificationError, match=message):
        verifier.build_receipt(paths)


def test_aim3_numeric_binding_must_resolve_to_primitive_source_value(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    def mutate(value: dict[str, Any]) -> None:
        value["report_bindings"][0]["value"] += 0.01

    _rewrite_source_json(paths, "aim3-new-results", mutate)
    with pytest.raises(verifier.BundleVerificationError, match="primitive result mapping"):
        verifier.build_receipt(paths)


def test_results_numeric_binding_is_exact_once_and_source_derived(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    results = paths.final_v13 / "Results.md"
    text = results.read_text()
    marker = next(line for line in text.splitlines() if line.startswith("<!-- AIM3_SOURCE_VALUE"))
    results.write_text(text.replace(marker, marker.replace(" -->", "1 -->"), 1), encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="numeric source bindings"):
        verifier.build_receipt(paths)


def test_aim4_reviews_k32_file_roster_and_pins_are_exact(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    unexpected = paths.aim4_k32_root / "untracked.txt"
    unexpected.write_text("not governed\n", encoding="utf-8")
    with pytest.raises(verifier.BundleVerificationError, match="file roster is not exact"):
        verifier.build_receipt(paths)


def test_aim4_structured_provenance_is_replayed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    structured_path = paths.aim4_k32_root / "completed_review_structured.json"
    value = json.loads(structured_path.read_text())
    value["provenance"]["base_form_sha256"] = "0" * 64
    _write_json(structured_path, value)

    manifest_path = paths.final_v13 / verifier.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    record = next(
        row
        for row in manifest["artifacts"]
        if row["path"].endswith("completed_review_structured.json")
    )
    record.update(_identity(structured_path, record["path"]))
    _write_json(manifest_path, manifest)
    pins = dict(paths.expected_aim4_k32_sha256)
    pins["completed_review_structured.json"] = _sha(structured_path)
    paths = replace(paths, expected_aim4_k32_sha256=pins)
    with pytest.raises(verifier.BundleVerificationError, match="structured provenance"):
        verifier.build_receipt(paths)


def test_seal_is_atomic_exactly_once_and_tool_identity_bound(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    expected = verifier.build_receipt(paths)
    assert verifier.seal(paths) == expected
    assert paths.destination.read_bytes() == verifier._receipt_bytes(expected)
    assert verifier.verify_published_receipt(paths) == expected

    original_tool = paths.verifier_code.read_bytes()
    paths.verifier_code.write_bytes(original_tool + b"# drift\n")
    with pytest.raises(verifier.BundleVerificationError, match="verifier SHA-256 drift"):
        verifier.verify_published_receipt(paths)
    paths.verifier_code.write_bytes(original_tool)
    assert verifier.verify_published_receipt(paths) == expected

    original_receipt = paths.destination.read_bytes()
    with pytest.raises(verifier.BundleVerificationError, match="overwrite"):
        verifier.seal(paths)
    assert paths.destination.read_bytes() == original_receipt


def test_existing_receipt_symlink_is_never_overwritten(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    decoy = paths.final_v13 / "decoy.json"
    decoy.write_text("do not touch\n", encoding="utf-8")
    paths.destination.symlink_to(decoy)
    with pytest.raises(verifier.BundleVerificationError, match="overwrite"):
        verifier.seal(paths)
    assert decoy.read_text() == "do not touch\n"


def test_cli_needs_explicit_seal_to_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(verifier, "default_paths", lambda: paths)
    assert verifier.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "READY_TO_SEAL"
    assert not paths.destination.exists()
    assert verifier.main(["--seal"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == verifier.FINAL_SEALED_STATUS
    assert paths.destination.is_file()
