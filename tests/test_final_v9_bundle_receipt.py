from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v9_bundle_receipt as bundle  # noqa: E402


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _valid_document(name: str) -> str:
    setup_only = ""
    if name == "Experimental_Setup.md":
        setup_only = (
            "The scheduler default is `max_concurrent_gpu_trainers=6`. "
            "Orion is excluded from canonical Aim 1.\n\n"
        )
    audit_only = ""
    if name == "Audit.md":
        audit_only = (
            "\nFive-seed scope: E0; E1v; all nine LOCO primary arms; E2-MET and "
            "Orion LOCO sensitivity; Aim 3 fixed, repeated, and E3v analyses.\n"
            "Three-seed unchanged scope: raw E2-CPHT; E2-CPHT-A; 15-fold Orion "
            "sensitivity; E2e; E2f-v3; between-slide; detailed E2-MET role/organ "
            "analyses; unaffected Aim 1 analyses.\n"
            "Study-wide MIL census: 1,225 = 735 adopted + 490 new fits in 102 new "
            "chains; model seeds 42, 43, 44, 45, 46; patient/slide folds unchanged; "
            "maximum concurrency 6; observed peak 6.\n"
        )
    return (
        f"# Final-v9 {name}\n\n"
        f"{setup_only}"
        "## Aim 1 — Primary colorectal discrimination\n\n"
        "The canonical conventional-H&E analysis is sealed.\n\n"
        "## Aim 2 — Transfer\n\n"
        f"{bundle.OFFICIAL_CPHT_NAME}.\n\n"
        "CPHT-R status: NOT RUN; it contributes no claim.\n\n"
        "## Aim 3 — Molecular hierarchy\n\n"
        "The completed molecular analyses are sealed.\n\n"
        "## Aim 4 — Morphologic atlas\n\n"
        "Aim 4 whole-section pathology validation status: GENERATED_UNREAD.\n"
        f"{audit_only}"
    )


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": bundle.sha256_file(path),
    }


def _refresh_manifest_identity(paths: bundle.BundlePaths, source_path: Path) -> None:
    manifest_path = paths.final_v9 / bundle.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        raw = Path(artifact["path"])
        resolved = raw.resolve() if raw.is_absolute() else (paths.repo / raw).resolve()
        if resolved == source_path.resolve():
            artifact["size_bytes"] = source_path.stat().st_size
            artifact["sha256"] = bundle.sha256_file(source_path)
            _write_json(manifest_path, manifest)
            return
    raise AssertionError(f"source is absent from fixture manifest: {source_path}")


def _mutate_json_source(
    paths: bundle.BundlePaths,
    source_path: Path,
    mutation: object,
) -> None:
    value = json.loads(source_path.read_text(encoding="utf-8"))
    assert callable(mutation)
    mutation(value)
    _write_json(source_path, value)
    _rebind_source_graph(paths)


def _mutate_json_sources(
    paths: bundle.BundlePaths,
    mutations: list[tuple[Path, object]],
) -> None:
    for source_path, mutation in mutations:
        value = json.loads(source_path.read_text(encoding="utf-8"))
        assert callable(mutation)
        mutation(value)
        _write_json(source_path, value)
    _rebind_source_graph(paths)


def _rebind_source_graph(paths: bundle.BundlePaths) -> None:
    """Let adversarial tests rehash an entire receipt DAG after semantic tampering."""

    manifest_path = paths.final_v9 / bundle.SOURCE_MANIFEST_NAME
    for _iteration in range(20):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_path: dict[Path, Path] = {}
        for artifact in manifest["artifacts"]:
            raw = Path(artifact["path"])
            resolved = raw.resolve() if raw.is_absolute() else (paths.repo / raw).resolve()
            by_path[resolved] = resolved
        for receipt in (paths.campaign_root / "campaign/receipts/jobs").glob("*.json"):
            by_path[receipt.resolve()] = receipt.resolve()

        changed = False

        def refresh(value: object, _by_path: dict[Path, Path] = by_path) -> None:
            nonlocal changed
            if isinstance(value, dict):
                if {"path", "size_bytes", "sha256"}.issubset(value):
                    raw = Path(str(value["path"]))
                    resolved = raw.resolve() if raw.is_absolute() else (paths.repo / raw).resolve()
                    target = _by_path.get(resolved)
                    if target is not None:
                        wanted_size = target.stat().st_size
                        wanted_hash = bundle.sha256_file(target)
                        if value["size_bytes"] != wanted_size or value["sha256"] != wanted_hash:
                            value["size_bytes"] = wanted_size
                            value["sha256"] = wanted_hash
                            changed = True
                for child in value.values():
                    refresh(child)
            elif isinstance(value, list):
                for child in value:
                    refresh(child)

        for source_path in by_path:
            if source_path.suffix != ".json":
                continue
            try:
                value = json.loads(source_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            before = json.dumps(value, sort_keys=True)
            refresh(value)
            if json.dumps(value, sort_keys=True) != before:
                _write_json(source_path, value)

        for artifact in manifest["artifacts"]:
            raw = Path(artifact["path"])
            resolved = raw.resolve() if raw.is_absolute() else (paths.repo / raw).resolve()
            artifact["size_bytes"] = resolved.stat().st_size
            artifact["sha256"] = bundle.sha256_file(resolved)
        _write_json(manifest_path, manifest)
        if not changed:
            return
    raise AssertionError("fixture receipt graph did not converge after semantic tampering")


def _direction(low: float) -> dict[str, object]:
    passed = low > 0.5
    return {
        "auroc": 0.65,
        "auroc_ci95": [low, 0.8],
        "directional_gate": {
            "rule": "patient-bootstrap AUROC lower 95% bound > 0.50",
            "lower_ci_above_0p5": passed,
            "passes": passed,
        },
    }


def _aim3_jobs(*, adopted: bool) -> list[dict[str, object]]:
    logical_jobs = [
        *(("fixed", task, None) for task in bundle.AIM3_FIXED_TASKS),
        *(
            ("repeated", task, draw_seed)
            for task in bundle.AIM3_REPEATED_TASKS
            for draw_seed in bundle.AIM3_REPEATED_DRAW_SEEDS
        ),
        *(("e3v", task, None) for task in bundle.AIM3_E3V_TASKS),
        *(("e1v", task, None) for task in bundle.AIM3_E1V_TASKS),
    ]
    seeds = bundle.ADOPTED_MODEL_SEEDS if adopted else bundle.NEW_MODEL_SEEDS
    jobs: list[dict[str, object]] = []
    for component, task, draw_seed in logical_jobs:
        for seed in seeds:
            key = bundle._aim3_job_key(component, task, draw_seed, seed)  # noqa: SLF001
            job = {
                "component": component,
                "model_seed": seed,
                "task": task,
                "draw_seed": draw_seed,
            }
            jobs.append(
                {"job": job, "job_key": key}
                if adopted
                else {
                    **job,
                    "key": key,
                    "actual_fits": 5 if component in {"e3v", "e1v"} else 6,
                }
            )
    return jobs


def _campaign_jobs() -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for seed in bundle.NEW_MODEL_SEEDS:
        component_job_key = f"aim1_e0_seed{seed}"
        jobs.append(
            {
                "job_id": f"aim1.{component_job_key}",
                "component_job_key": component_job_key,
                "component": "aim1_e0",
                "stage": "baseline",
                "seed": seed,
                "fit_count": 6,
            }
        )
    for arm in bundle.AIM2_ALL_ARMS:
        for seed in bundle.NEW_MODEL_SEEDS:
            for stage, fit_count in (("source_cv", 5), ("refit", 1)):
                component_job_key = f"aim2.{stage}.{arm}.seed{seed}"
                jobs.append(
                    {
                        "job_id": component_job_key,
                        "component_job_key": component_job_key,
                        "component": "aim2_loco",
                        "stage": stage,
                        "seed": seed,
                        "arm": arm,
                        "fit_count": fit_count,
                    }
                )
    for stage, task, draw_seed in bundle._aim3_logical_roster():  # noqa: SLF001
        for seed in bundle.NEW_MODEL_SEEDS:
            component_job_key = bundle._aim3_job_key(  # noqa: SLF001
                stage, task, draw_seed, seed
            )
            jobs.append(
                {
                    "job_id": f"aim3.{component_job_key}",
                    "component_job_key": component_job_key,
                    "component": "aim3_ladders",
                    "stage": stage,
                    "seed": seed,
                    "task": task,
                    "draw_seed": draw_seed,
                    "fit_count": 5 if stage in {"e3v", "e1v"} else 6,
                }
            )
    for job in jobs:
        job["command"] = ["synthetic-trainer", str(job["job_id"])]
    assert len(jobs) == 102
    assert sum(int(job["fit_count"]) for job in jobs) == 490
    return jobs


def _fixture(tmp_path: Path) -> tuple[bundle.BundlePaths, dict[str, Path]]:
    repo = tmp_path / "repo"
    final_v9 = repo / "reports" / "final_v9"
    for filename in bundle.REPORT_DOCUMENTS:
        _write(final_v9 / filename, _valid_document(filename))

    verifier = repo / "tools" / "final_v9_bundle_receipt.py"
    verifier_test = repo / "tests" / "test_final_v9_bundle_receipt.py"
    _write(verifier, "# sealed verifier\n")
    _write(verifier_test, "# sealed verifier tests\n")
    campaign = repo / "external/final_v9_mil_5seed_expansion_v1_20260823"
    adjudication = repo / "external/final_v9_adjudication/aim2_e2a_five_seed"
    paths = bundle.BundlePaths(
        repo=repo,
        final_v9=final_v9,
        destination=final_v9 / bundle.FINAL_RECEIPT_NAME,
        verifier_code=verifier,
        verifier_test=verifier_test,
        campaign_root=campaign,
        adjudication_root=adjudication,
    )

    required = bundle._required_five_seed_sources(paths)  # noqa: SLF001
    for source_id, (path, _aim) in required.items():
        if source_id.endswith(
            (
                "patient-logits",
                "source-oof",
                "primary-patients",
                "metastatic-patients",
                "orion-patients",
            )
        ):
            _write(path, "synthetic parquet bytes\n")
        elif source_id.endswith(("bootstrap", "table")):
            _write(path, "synthetic binary/table bytes\n")
        elif source_id.endswith(("implementation", "test")):
            _write(path, f"# {source_id}\n")
        elif source_id.endswith("calibrators"):
            _write_json(path, {"status": "synthetic calibrators"})

    legacy_specs = {
        "aim1-sealed-replay-results": ("Aim 1", "legacy_mixed_scope_continuity_results"),
        "aim1-encoder-and-control-results": ("Aim 1", "legacy_mixed_scope_continuity_results"),
        "aim1-worklist-results": ("Aim 1", "secondary_results"),
        "aim1-ras-composite-results": ("Aim 1", "secondary_results"),
        "aim1-decision-curve-results": ("Aim 1", "clinical_utility_results"),
        "aim2-between-slide-results": ("Aim 2", "unchanged_three_seed_results"),
        "aim2-e2met-results": ("Aim 2", "legacy_mixed_scope_continuity_results"),
        "aim2-e2e-results": ("Aim 2", "unchanged_three_seed_results"),
        "aim2-e2f-v3-results": ("Aim 2", "unchanged_three_seed_results"),
        "aim2-e2cpht-results": ("Aim 2", "controlling_three_seed_results"),
        "aim2-e2cpht-a-v2-results": ("Aim 2", "controlling_three_seed_results"),
        "aim4-results": ("Aim 4", "controlling_results"),
    }
    legacy_paths: dict[str, Path] = {}
    for source_id in legacy_specs:
        path = repo / "authoritative" / f"{source_id}.json"
        _write_json(path, {"status": "complete", "source": source_id})
        legacy_paths[source_id] = path

    aim1_contract_path = required["aim1-e0-five-seed-contract"][0]
    aim1_fit_census = {
        "inherited_folds": 15,
        "inherited_p75_refits": 3,
        "new_folds": 10,
        "new_p75_refits": 2,
        "final_folds": 25,
        "final_p75_refits": 5,
        "new_fits": 12,
        "final_fits": 30,
    }
    _write_json(
        aim1_contract_path,
        {
            "status": "sealed before extension training",
            "experiment": "Aim1 E0 canonical five-seed extension",
            "scientific_change": {
                "old_model_seeds": bundle.ADOPTED_MODEL_SEEDS,
                "new_model_seeds": bundle.MODEL_SEEDS,
                "added_model_seeds": bundle.NEW_MODEL_SEEDS,
                "split_layout_changed": False,
                "model_recipe_changed": False,
                "inherited_artifacts_modified": False,
            },
            "fit_census": aim1_fit_census,
            "analysis_contract": {
                "bootstrap": {"unit": "patient", "shared_indices_across_model_seeds": True}
            },
        },
    )
    aim2_contract_path = required["aim2-loco-five-seed-contract"][0]
    aim2_counts = {
        "arms": 9,
        "controlling_arms": 8,
        "adopted_fits": 162,
        "new_fits": 108,
        "complete_fits": 270,
        "training_chain_jobs": 36,
        "adopted_score_artifacts": 99,
        "new_score_artifacts": 66,
        "complete_score_artifacts": 165,
    }
    mixed_references = {
        "all_conventional_cpht_three_seed": _artifact(legacy_paths["aim2-e2cpht-results"]),
        "cpht_a_three_seed": _artifact(legacy_paths["aim2-e2cpht-a-v2-results"]),
    }
    _write_json(
        aim2_contract_path,
        {
            "status": "sealed_extension_contract",
            "experiment": "Aim 2 complete LOCO MIL five-seed extension",
            "seeds": {
                "adopted": bundle.ADOPTED_MODEL_SEEDS,
                "new": bundle.NEW_MODEL_SEEDS,
                "complete": bundle.MODEL_SEEDS,
            },
            "split_verdict": {
                "patient_and_slide_folds_change_across_model_seeds": False,
                "model_seed_changes_only_stochastic_training": True,
            },
            "statistics": {"model_seeds_are_not_inference_units": True},
            "scope": {
                "controlling_arms": list(bundle.AIM2_CONTROLLING_ARMS),
                "secondary_sensitivity": [bundle.AIM2_SENSITIVITY_ARM],
                "raw_all_conventional_cpht_seed_scope": bundle.ADOPTED_MODEL_SEEDS,
                "cpht_a_seed_scope": bundle.ADOPTED_MODEL_SEEDS,
            },
            "counts": aim2_counts,
            "arms": {arm: {"name": arm} for arm in bundle.AIM2_ALL_ARMS},
            "mixed_seed_scope_references": mixed_references,
        },
    )
    aim3_contract_path = required["aim3-ladders-five-seed-contract"][0]
    aim3_new_counts = {"chains": 64, "oof_folds": 320, "refits": 50, "actual_mil_fits": 370}
    _write_json(
        aim3_contract_path,
        {
            "status": "prepared",
            "protocol": {
                "old_seeds": bundle.ADOPTED_MODEL_SEEDS,
                "new_seeds": bundle.NEW_MODEL_SEEDS,
                "all_seeds": bundle.MODEL_SEEDS,
                "max_parallel_chains": 6,
            },
            "new_counts": aim3_new_counts,
            "adopted_old_chains": _aim3_jobs(adopted=True),
            "new_jobs": _aim3_jobs(adopted=False),
        },
    )

    campaign_contract_path = required["final-v9-five-seed-campaign-contract"][0]
    campaign_jobs = _campaign_jobs()
    _write_json(
        campaign_contract_path,
        {
            "status": "sealed_before_new_fit",
            "experiment": "final-v9 study-wide MIL five-seed expansion",
            "model_seeds": {
                "adopted": bundle.ADOPTED_MODEL_SEEDS,
                "new": bundle.NEW_MODEL_SEEDS,
                "complete": bundle.MODEL_SEEDS,
            },
            "split_policy": {
                "outer_and_inner_membership_changes_across_model_seeds": False,
                "rule": "reuse each experiment's exact frozen patient/slide fold manifest for seeds 42-46",
            },
            "execution": {"max_concurrent_gpu_trainers": 6, "fresh_process_per_chain": True},
            "counts": {
                "jobs": 102,
                "new_mil_fits": 490,
                "by_component": {
                    "aim1_e0": {"jobs": 2, "new_fits": 12},
                    "aim2_loco": {"jobs": 36, "new_fits": 108},
                    "aim3_ladders": {"jobs": 64, "new_fits": 370},
                },
            },
            "component_contracts": {
                "aim1_e0": _artifact(aim1_contract_path),
                "aim2_loco": _artifact(aim2_contract_path),
                "aim3_ladders": _artifact(aim3_contract_path),
            },
            "jobs": campaign_jobs,
        },
    )
    preflight_path = required["final-v9-five-seed-campaign-deep-preflight"][0]
    _write_json(
        preflight_path,
        {
            "status": "deep_preflight_passed",
            "maximum_concurrent_gpu_trainers": 6,
            "aim1_live_deep_contract_and_pack_authentication": True,
            "contract": _artifact(campaign_contract_path),
        },
    )

    aim1_training_path = required["aim1-e0-five-seed-training-validation"][0]
    _write_json(
        aim1_training_path,
        {
            "status": "complete",
            "campaign_contract": _artifact(aim1_contract_path),
            "model_seeds": bundle.MODEL_SEEDS,
            "fit_census": aim1_fit_census,
            "fold_layout_shared_across_all_seeds": True,
            "p75_refit_authenticated_for_all_seeds": True,
        },
    )
    aim1_results_path = required["aim1-e0-five-seed-results"][0]
    _write_json(
        aim1_results_path,
        {
            "experiment": "Aim1 E0 canonical five-seed extension",
            "model_seeds": bundle.MODEL_SEEDS,
            "inference": {
                "unit": "patient",
                "shared_indices_across_model_seeds": True,
                "model_seeds_are_inferential_units": False,
            },
            "inputs": {
                "campaign_contract": _artifact(aim1_contract_path),
                "training_validation": _artifact(aim1_training_path),
                "patient_native_logits": _artifact(required["aim1-e0-five-seed-patient-logits"][0]),
            },
        },
    )
    aim1_receipt_path = required["aim1-e0-five-seed-analysis-receipt"][0]
    _write_json(
        aim1_receipt_path,
        {
            "status": "complete",
            "probability_roundtrip_used": False,
            "results": _artifact(aim1_results_path),
            "patient_native_logits": _artifact(required["aim1-e0-five-seed-patient-logits"][0]),
            "training_validation": _artifact(aim1_training_path),
        },
    )

    aim2_seal_path = required["aim2-loco-five-seed-inference-seal"][0]
    _write_json(
        aim2_seal_path,
        {
            "status": "sealed_before_outcome_join",
            "target_outcomes_present": False,
            "contract": _artifact(aim2_contract_path),
            "adopted_score_count": 99,
            "new_score_count": 66,
            "complete_score_count": 165,
            "five_seed_loco_complete": True,
            "all_conventional_cpht_scope": bundle.ADOPTED_MODEL_SEEDS,
            "cpht_a_scope": bundle.ADOPTED_MODEL_SEEDS,
        },
    )
    primary_arms = {
        "family_cptac",
        "family_rih",
        "family_rih_sm",
        "family_surgen",
        "family_tcga",
        "sibling_sr386",
        "sibling_sr1482",
        "sibling_tcga_coad",
        "sibling_tcga_read",
    }
    aim2_results_path = required["aim2-loco-five-seed-results"][0]
    _write_json(
        aim2_results_path,
        {
            "experiment": "Aim 2 complete LOCO MIL five-seed results",
            "seeds": bundle.MODEL_SEEDS,
            "ensemble_rule": "mean native logits across seeds, then one sigmoid",
            "primary": {arm: {} for arm in sorted(primary_arms)},
            "mixed_seed_scope": {
                "five_seed": [
                    "all LOCO primary",
                    "E2-MET complete LOCO matrix",
                    "Orion LOCO sensitivity",
                ],
                "three_seed_unchanged": [
                    "raw all-conventional CPHT",
                    "CPHT-A residual adaptation",
                ],
                "references": mixed_references,
            },
        },
    )
    aim2_receipt_path = required["aim2-loco-five-seed-report-receipt"][0]
    aim2_artifact_ids = {
        "source_oof": "aim2-loco-five-seed-source-oof",
        "calibrators": "aim2-loco-five-seed-calibrators",
        "primary_patients": "aim2-loco-five-seed-primary-patients",
        "met_patients": "aim2-loco-five-seed-metastatic-patients",
        "orion_patients": "aim2-loco-five-seed-orion-patients",
        "results": "aim2-loco-five-seed-results",
        "table": "aim2-loco-five-seed-table",
    }
    _write_json(
        aim2_receipt_path,
        {
            "status": "sealed_five_seed_results",
            "contract": _artifact(aim2_contract_path),
            "inference_seal": _artifact(aim2_seal_path),
            "artifacts": {
                key: _artifact(required[source_id][0])
                for key, source_id in aim2_artifact_ids.items()
            },
            "target_outcomes_opened_only_after_inference_seal": True,
            "bootstrap_unit": "patient",
            "model_seeds_are_not_inference_units": True,
        },
    )

    aim3_results_path = required["aim3-ladders-five-seed-results"][0]
    _write_json(
        aim3_results_path,
        {
            "status": "complete",
            "estimand": "five-seed patient native-logit ensemble; seeds are not inference units",
            "three_seed_replay_before_extension": {},
            "e0_gene_reference": {},
            "fixed_univ1": {},
            "repeated_univ1": {},
            "e3v_virchow2_cls": {},
            "e1v_virchow2_cls_gene_reference": {},
        },
    )
    aim3_audit_path = required["aim3-ladders-five-seed-analysis-audit"][0]
    _write_json(
        aim3_audit_path,
        {
            "status": "PASS",
            "contract": _artifact(aim3_contract_path),
            "report": _artifact(aim3_results_path),
            "bootstrap_distributions": _artifact(required["aim3-ladders-five-seed-bootstrap"][0]),
        },
    )
    aim3_completion_path = required["aim3-ladders-five-seed-completion"][0]
    _write_json(
        aim3_completion_path,
        {
            "status": "completed",
            "counts": aim3_new_counts,
            "artifacts": {
                "five_seed_results.json": _artifact(aim3_results_path),
                "bootstrap_distributions.npz": _artifact(
                    required["aim3-ladders-five-seed-bootstrap"][0]
                ),
                "analysis_audit.json": _artifact(aim3_audit_path),
            },
        },
    )

    campaign_training_path = required["final-v9-five-seed-campaign-training-completion"][0]
    campaign_job_receipts: dict[str, dict[str, object]] = {}
    for job in campaign_jobs:
        job_id = str(job["job_id"])
        receipt_path = (
            campaign / "campaign/receipts/jobs" / f"{bundle._safe_campaign_job_name(job_id)}.json"  # noqa: SLF001
        )
        _write_json(
            receipt_path,
            {
                "status": "completed_rc0",
                "job_id": job_id,
                "returncode": 0,
                "contract": _artifact(campaign_contract_path),
                "command": job["command"],
            },
        )
        campaign_job_receipts[job_id] = _artifact(receipt_path)
    _write_json(
        campaign_training_path,
        {
            "status": "490 new MIL fits completed and certified",
            "contract": _artifact(campaign_contract_path),
            "preflight": _artifact(preflight_path),
            "job_receipts": campaign_job_receipts,
            "new_mil_fits": 490,
            "new_training_jobs": 102,
            "maximum_concurrent_gpu_trainers": 6,
            "observed_peak_concurrent_gpu_trainers": 6,
            "reconstructed_peak_from_job_exit_intervals": 6,
        },
    )
    campaign_final_path = required["final-v9-five-seed-campaign-results-completion"][0]
    _write_json(
        campaign_final_path,
        {
            "status": "five_seed_results_complete",
            "seeds": bundle.MODEL_SEEDS,
            "model_seeds_are_not_inference_units": True,
            "contract": _artifact(campaign_contract_path),
            "training": _artifact(campaign_training_path),
            "components": {
                "aim1_e0": {
                    "training": _artifact(aim1_training_path),
                    "results": _artifact(aim1_results_path),
                    "analysis_receipt": _artifact(aim1_receipt_path),
                },
                "aim2_loco": {
                    "inference_seal": _artifact(aim2_seal_path),
                    **{
                        f"analysis_{key}": _artifact(required[source_id][0])
                        for key, source_id in aim2_artifact_ids.items()
                    },
                    "analysis_receipt": _artifact(aim2_receipt_path),
                },
                "aim3_ladders": {
                    "results": _artifact(aim3_results_path),
                    "bootstrap_distributions": _artifact(
                        required["aim3-ladders-five-seed-bootstrap"][0]
                    ),
                    "analysis_audit": _artifact(aim3_audit_path),
                    "completion": _artifact(aim3_completion_path),
                },
            },
        },
    )

    family_names = [
        "CPTAC",
        "RIH",
        "SR386_given_whole_SurGen_holdout",
        "SR1482_given_whole_SurGen_holdout",
        "TCGA_pooled_COAD_READ",
    ]
    sibling_slugs = ["sr386", "sr1482", "tcga_coad", "tcga_read"]
    passed_family = family_names.copy()
    passed_siblings = sibling_slugs[:-1]
    precedence = {
        "source_report": _artifact(aim2_results_path),
        "superseded_source_report_json_pointers": list(bundle.SUPERSEDED_ADJUDICATION_POINTERS),
        "non_authoritative_source_paired_ci_json_pointers": list(
            bundle.NONAUTHORITATIVE_PAIRED_CI_POINTERS
        ),
        "retained_source_paired_delta_point_json_pointers": list(
            bundle.RETAINED_PAIRED_POINT_POINTERS
        ),
        "authoritative_replacements": {
            "E2a-F gates and claim": "/e2a_f/adjudication",
            "E2a-D gates and claim": "/e2a_d/adjudication",
            "sibling paired intervals": "/e2a_d/paired_sibling_minus_family",
            "RIH size-matched paired interval": "/e2a_d/size_matched_rih_sensitivity",
        },
        "superseded_sibling_primary_ci_replacements": {
            f"/primary/{arm}/auroc_ci95": f"/e2a_d/targets/{slug}/auroc_ci95"
            for arm, slug in zip(bundle._SIBLING_ARMS, bundle._SIBLING_SLUGS, strict=True)  # noqa: SLF001
        },
        "confirmed_unchanged_source_report_json_pointers": [
            "/family_loco_standardized_macro/directional_results",
            "/family_loco_standardized_macro/macro_auroc",
            "/family_loco_standardized_macro/macro_auroc_ci95",
            "/e2met_confirmatory",
        ],
        "supersession_is_field_scoped": True,
        "all_unlisted_source_report_fields_remain_authoritative": True,
        "macro_rescue_prohibited": True,
    }
    adjud_result_path = required["aim2-e2a-five-seed-adjudication-result"][0]
    _write_json(
        adjud_result_path,
        {
            "status": "governed_five_seed_adjudication",
            "experiment": "Aim 2 E2a five-seed governed adjudication",
            "model_seeds": bundle.MODEL_SEEDS,
            "ensemble_rule": "mean native logits across five model seeds",
            "inference": {
                "unit": "patient",
                "stratification": "target_x_KRAS",
                "n_bootstrap": 10_000,
                "bootstrap_seed": 20260817,
                "paired_scorer_indices_shared": True,
                "macro_recomputed_each_draw": True,
                "folds_and_model_seeds_are_not_resampling_units": True,
            },
            "precedence": precedence,
            "e2a_f": {
                "directions": {name: _direction(0.51) for name in family_names},
                "nested_four_family_macro": {
                    "auroc": 0.65,
                    "auroc_ci95": [0.55, 0.75],
                    "role": "panel summary; cannot rescue a failed direction",
                },
                "adjudication": {
                    "rule": (
                        "all five target-specific patient-bootstrap AUROC lower 95% bounds "
                        "must exceed 0.50; the macro cannot rescue a failed direction"
                    ),
                    "passed_directions": passed_family,
                    "required_directions": family_names,
                    "all_directional_lower_bounds_above_0p5": True,
                    "claim_family_loco_transport": True,
                },
            },
            "e2a_d": {
                "targets": {
                    name: _direction(0.51 if name in passed_siblings else 0.49)
                    for name in sibling_slugs
                },
                "macros": {
                    "secondary_five_acquisition_domain": {
                        "auroc": 0.65,
                        "auroc_ci95": [0.55, 0.75],
                        "role": "secondary panel summary; cannot rescue a failed direction",
                    },
                    "descriptive_equal_six_stratum": {
                        "auroc": 0.65,
                        "auroc_ci95": [0.55, 0.75],
                        "role": "descriptive; gives TCGA and SurGen two votes each",
                    },
                },
                "adjudication": {
                    "rule": (
                        "all four sibling-stratum patient-bootstrap AUROC lower 95% bounds "
                        "must exceed 0.50; neither macro can rescue a failed direction"
                    ),
                    "passed_directions": passed_siblings,
                    "required_directions": sibling_slugs,
                    "all_directional_lower_bounds_above_0p5": False,
                    "claim_sibling_stratum_transport": False,
                },
            },
            "scope_boundary": {
                "raw_cpht": {
                    "model_seeds": bundle.ADOPTED_MODEL_SEEDS,
                    "five_seed_adjudication_applied": False,
                    "result": _artifact(legacy_paths["aim2-e2cpht-results"]),
                },
                "cpht_a": {"model_seeds": bundle.ADOPTED_MODEL_SEEDS},
                "orion_loco_sensitivity": {"is_confirmatory_e2_cpht": False},
                "five_seed_e2met_role_and_organ_analyses": {
                    "required_to_confirm_existing_e2met_gate": False
                },
            },
        },
    )
    adjud_receipt_path = required["aim2-e2a-five-seed-adjudication-receipt"][0]
    _write_json(
        adjud_receipt_path,
        {
            "status": "sealed_governed_five_seed_adjudication",
            "experiment": "Aim 2 E2a five-seed governed adjudication",
            "outputs": {
                "result": _artifact(adjud_result_path),
                "bootstrap_distributions": _artifact(
                    required["aim2-e2a-five-seed-adjudication-bootstrap"][0]
                ),
            },
            "implementation": {
                "tool": _artifact(required["aim2-e2a-five-seed-adjudication-implementation"][0]),
                "focused_test": _artifact(required["aim2-e2a-five-seed-adjudication-test"][0]),
            },
            "bootstrap_inventory": {"draws_per_array": 10_000, "dtype": "float64"},
            "verification_contract": {
                "deterministic_full_recomputation": True,
                "verify_is_read_only": True,
                "receipt_written_last": True,
                "source_campaign_never_written": True,
            },
        },
    )

    artifacts: list[dict[str, object]] = []
    for source_id, (path, aim) in required.items():
        role = "integrity_receipt"
        if source_id in {
            "aim1-e0-five-seed-results",
            "aim2-loco-five-seed-results",
            "aim2-e2a-five-seed-adjudication-result",
            "aim3-ladders-five-seed-results",
        }:
            role = "controlling_five_seed_results"
        artifacts.append(
            {
                "id": source_id,
                "aims": [aim],
                "experiments": ["five-seed MIL expansion"],
                "role": role,
                "path": str(path.relative_to(repo)),
                "size_bytes": path.stat().st_size,
                "sha256": bundle.sha256_file(path),
            }
        )
    for source_id, (aim, role) in legacy_specs.items():
        path = legacy_paths[source_id]
        artifacts.append(
            {
                "id": source_id,
                "aims": [aim],
                "experiments": ["unchanged scope"],
                "role": role,
                "path": str(path.relative_to(repo)),
                "size_bytes": path.stat().st_size,
                "sha256": bundle.sha256_file(path),
            }
        )
    _write_json(
        final_v9 / bundle.SOURCE_MANIFEST_NAME,
        {"schema_version": 1, "bundle": "final_v9", "artifacts": artifacts},
    )
    sources = {
        "aim1": aim1_results_path,
        "aim2": aim2_results_path,
        "aim3": aim3_results_path,
        "aim4": legacy_paths["aim4-results"],
        "adjudication": adjud_result_path,
    }
    return paths, sources


def test_build_receipt_records_truthful_scope_and_complete_identities(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    receipt = bundle.build_receipt(paths)

    assert receipt["status"] == bundle.SEALED_STATUS
    assert receipt["organization"] == "aim_focused_clean_rewrite"
    assert receipt["post_outcome_official_specification"] is True
    assert receipt["not_preregistration"] is True
    assert receipt["preregistration"] is False
    assert receipt["append_only_inheritance_required"] is False
    assert receipt["execution_policy"]["max_concurrent_gpu_trainers"] == 6
    assert receipt["declared_scientific_states"] == {
        "canonical_aim1_orion_included": False,
        "cpht_r": "NOT_RUN",
        "aim4_whole_section_pathology_validation": "GENERATED_UNREAD",
    }
    assert receipt["model_seed_scope"]["complete"] == bundle.MODEL_SEEDS
    assert (
        receipt["model_seed_scope"]["patient_and_slide_fold_membership_changes_across_model_seeds"]
        is False
    )
    assert receipt["study_wide_mil_census"] == {
        "adopted_fits": 735,
        "new_fits": 490,
        "complete_fits": 1225,
        "new_training_chain_jobs": 102,
        "maximum_concurrent_gpu_trainers": 6,
        "observed_peak_concurrent_gpu_trainers": 6,
    }
    assert "raw E2-CPHT" in receipt["mixed_seed_scope"]["three_seed_unchanged"]
    assert receipt["checks"]["adjudication_pointer_and_lower_ci_gates"] == "PASS"
    assert list(receipt["documents"]) == list(bundle.REPORT_DOCUMENTS)
    assert len(receipt["authoritative_sources"]) == 41
    for item in [
        *receipt["documents"].values(),
        receipt["source_manifest"],
        *receipt["authoritative_sources"],
        *receipt["verification"].values(),
    ]:
        assert isinstance(item["size_bytes"], int)
        assert len(item["sha256"]) == 64


def test_seal_is_atomic_exactly_once_and_default_verify_is_read_only(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    sealed = bundle.seal(paths)
    receipt_bytes = paths.destination.read_bytes()

    assert bundle.verify_published_receipt(paths) == sealed
    assert paths.destination.read_bytes() == receipt_bytes
    with pytest.raises(bundle.BundleVerificationError, match="refusing to overwrite"):
        bundle.seal(paths)
    assert paths.destination.read_bytes() == receipt_bytes


@pytest.mark.parametrize("target", ["document", "source", "manifest", "verifier"])
def test_published_receipt_fails_closed_on_any_identity_drift(tmp_path: Path, target: str) -> None:
    paths, sources = _fixture(tmp_path)
    bundle.seal(paths)

    drift_targets = {
        "document": paths.final_v9 / "Results.md",
        "source": sources["aim2"],
        "manifest": paths.final_v9 / bundle.SOURCE_MANIFEST_NAME,
        "verifier": paths.verifier_code,
    }
    with drift_targets[target].open("a", encoding="utf-8") as handle:
        handle.write("drift\n")

    with pytest.raises(bundle.BundleVerificationError):
        bundle.verify_published_receipt(paths)


@pytest.mark.parametrize(
    ("filename", "old", "new", "message"),
    [
        ("Audit.md", "## Aim 3", "### Section 3", "ordered level-two heading"),
        ("Results.md", bundle.OFFICIAL_CPHT_NAME, "informal CPHT", "official E2-CPHT"),
        ("Audit.md", "NOT RUN", "deferred", "CPHT-R"),
        ("Results.md", "GENERATED_UNREAD", "pending", "GENERATED_UNREAD"),
        ("Experimental_Setup.md", "max_concurrent_gpu_trainers=6", "six jobs", "trainers=6"),
        (
            "Experimental_Setup.md",
            "Orion is excluded from canonical Aim 1",
            "Orion handling is described elsewhere",
            "Orion is excluded",
        ),
        ("Audit.md", "The completed", "TODO: The completed", "placeholder-like"),
        (
            "Audit.md",
            "Three-seed unchanged scope:",
            "Historical scope:",
            "Three-seed unchanged scope",
        ),
    ],
)
def test_candidate_report_contract_rejects_missing_required_fact(
    tmp_path: Path, filename: str, old: str, new: str, message: str
) -> None:
    paths, _ = _fixture(tmp_path)
    document = paths.final_v9 / filename
    document.write_text(
        document.read_text(encoding="utf-8").replace(old, new, 1),
        encoding="utf-8",
    )

    with pytest.raises(bundle.BundleVerificationError, match=message):
        bundle.build_receipt(paths)


def test_source_manifest_requires_every_aim_and_rejects_source_drift(
    tmp_path: Path,
) -> None:
    paths, sources = _fixture(tmp_path)
    manifest_path = paths.final_v9 / bundle.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = manifest["artifacts"][:-1]
    _write_json(manifest_path, manifest)
    with pytest.raises(bundle.BundleVerificationError, match="no authoritative artifact"):
        bundle.build_receipt(paths)

    paths, sources = _fixture(tmp_path / "second")
    sources["aim1"].write_text("changed source\n", encoding="utf-8")
    with pytest.raises(bundle.BundleVerificationError, match="identity drift"):
        bundle.build_receipt(paths)


def test_source_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    manifest_path = paths.final_v9 / bundle.SOURCE_MANIFEST_NAME
    text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        text.replace(
            '"bundle": "final_v9",',
            '"bundle": "final_v9",\n  "bundle": "final_v9",',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(bundle.BundleVerificationError, match="duplicate JSON key"):
        bundle.build_receipt(paths)


def test_source_manifest_rejects_stale_inventory_without_required_five_seed_id(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    manifest_path = paths.final_v9 / bundle.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        artifact
        for artifact in manifest["artifacts"]
        if artifact["id"] != "final-v9-five-seed-campaign-results-completion"
    ]
    _write_json(manifest_path, manifest)

    with pytest.raises(bundle.BundleVerificationError, match="lacks required five-seed IDs"):
        bundle.build_receipt(paths)


def test_required_five_seed_source_must_use_exact_governed_path(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    manifest_path = paths.final_v9 / bundle.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["id"] == "final-v9-five-seed-campaign-results-completion"
    )
    original = (paths.repo / source["path"]).resolve()
    wrong = paths.repo / "wrong/five_seed_results_complete.json"
    _write(wrong, original.read_text(encoding="utf-8"))
    source["path"] = str(wrong.relative_to(paths.repo))
    source["size_bytes"] = wrong.stat().st_size
    source["sha256"] = bundle.sha256_file(wrong)
    _write_json(manifest_path, manifest)

    with pytest.raises(bundle.BundleVerificationError, match="source path mismatch"):
        bundle.build_receipt(paths)


@pytest.mark.parametrize(
    ("source_id", "mutation", "message"),
    [
        (
            "final-v9-five-seed-campaign-contract",
            lambda value: value["model_seeds"].update(complete=[42, 43, 44, 45]),
            "model-seed roster",
        ),
        (
            "final-v9-five-seed-campaign-contract",
            lambda value: value["split_policy"].update(
                outer_and_inner_membership_changes_across_model_seeds=True
            ),
            "unchanged patient/slide folds",
        ),
        (
            "final-v9-five-seed-campaign-training-completion",
            lambda value: value.update(observed_peak_concurrent_gpu_trainers=5),
            "observed peak of six",
        ),
        (
            "aim2-loco-five-seed-contract",
            lambda value: value["scope"].update(
                raw_all_conventional_cpht_seed_scope=[42, 43, 44, 45]
            ),
            "mixed scope",
        ),
        (
            "aim3-ladders-five-seed-contract",
            lambda value: value["adopted_old_chains"].pop(),
            "component-by-seed distribution",
        ),
    ],
)
def test_five_seed_semantics_fail_closed_on_seed_split_scope_or_census_tampering(
    tmp_path: Path,
    source_id: str,
    mutation: object,
    message: str,
) -> None:
    paths, _ = _fixture(tmp_path)
    source_path = bundle._required_five_seed_sources(paths)[source_id][0]  # noqa: SLF001
    _mutate_json_source(paths, source_path, mutation)

    with pytest.raises(bundle.BundleVerificationError, match=message):
        bundle.build_receipt(paths)


def test_training_job_receipt_keys_must_exactly_match_contract_job_ids(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    source = bundle._required_five_seed_sources(paths)[  # noqa: SLF001
        "final-v9-five-seed-campaign-training-completion"
    ][0]

    def replace_one_key(value: dict[str, object]) -> None:
        records = value["job_receipts"]
        assert isinstance(records, dict)
        first = next(iter(records))
        record = records.pop(first)
        records["arbitrary-but-count-preserving-job"] = record

    _mutate_json_source(paths, source, replace_one_key)
    with pytest.raises(bundle.BundleVerificationError, match="exactly equal contract job IDs"):
        bundle.build_receipt(paths)


def test_training_job_receipt_identity_must_bind_live_bytes(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    source = bundle._required_five_seed_sources(paths)[  # noqa: SLF001
        "final-v9-five-seed-campaign-training-completion"
    ][0]

    def corrupt_hash(value: dict[str, object]) -> None:
        records = value["job_receipts"]
        assert isinstance(records, dict)
        record = records[next(iter(records))]
        assert isinstance(record, dict)
        record["sha256"] = "0" * 64

    value = json.loads(source.read_text(encoding="utf-8"))
    corrupt_hash(value)
    _write_json(source, value)
    _refresh_manifest_identity(paths, source)
    with pytest.raises(bundle.BundleVerificationError, match="not bound to live bytes"):
        bundle.build_receipt(paths)


def test_contract_component_census_is_derived_from_jobs_not_stale_declaration(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    source = bundle._required_five_seed_sources(paths)[  # noqa: SLF001
        "final-v9-five-seed-campaign-contract"
    ][0]

    def relabel_all_jobs(value: dict[str, object]) -> None:
        jobs = value["jobs"]
        assert isinstance(jobs, list)
        for ordinal, job in enumerate(jobs):
            assert isinstance(job, dict)
            job.update(
                component="aim3_ladders",
                stage="fixed",
                task=f"forged_task_{ordinal // 2}",
                draw_seed=None,
                fit_count=6,
            )

    _mutate_json_source(paths, source, relabel_all_jobs)
    with pytest.raises(bundle.BundleVerificationError, match="derived 2/36/64"):
        bundle.build_receipt(paths)


def test_aim3_new_job_inventory_requires_balanced_seed_pairs(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    source = bundle._required_five_seed_sources(paths)[  # noqa: SLF001
        "aim3-ladders-five-seed-contract"
    ][0]

    def skew_seeds(value: dict[str, object]) -> None:
        jobs = value["new_jobs"]
        assert isinstance(jobs, list) and len(jobs) == 64
        for index, job in enumerate(jobs):
            assert isinstance(job, dict)
            job["model_seed"] = 46 if index == len(jobs) - 1 else 45

    _mutate_json_source(paths, source, skew_seeds)
    with pytest.raises(bundle.BundleVerificationError, match="component-by-seed distribution"):
        bundle.build_receipt(paths)


def test_aim2_controlling_arm_roster_rejects_count_preserving_replacement(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    source = bundle._required_five_seed_sources(paths)[  # noqa: SLF001
        "aim2-loco-five-seed-contract"
    ][0]

    def replace_one_official_arm(value: dict[str, object]) -> None:
        scope = value["scope"]
        assert isinstance(scope, dict)
        arms = scope["controlling_arms"]
        assert isinstance(arms, list) and len(arms) == 8
        arms[0] = "bogus_arm"

    _mutate_json_source(paths, source, replace_one_official_arm)
    with pytest.raises(bundle.BundleVerificationError, match="exact eight-arm controlling roster"):
        bundle.build_receipt(paths)


def test_aim2_coupled_component_and_scheduler_relabel_fails_after_full_rehash(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    required = bundle._required_five_seed_sources(paths)  # noqa: SLF001
    component_source = required["aim2-loco-five-seed-contract"][0]
    scheduler_source = required["final-v9-five-seed-campaign-contract"][0]

    def relabel_component(value: dict[str, object]) -> None:
        arms = value["arms"]
        assert isinstance(arms, dict)
        arm = arms.pop("family_cptac")
        assert isinstance(arm, dict)
        arm["name"] = "bogus_arm"
        arms["bogus_arm"] = arm

    def relabel_scheduler(value: dict[str, object]) -> None:
        jobs = value["jobs"]
        assert isinstance(jobs, list)
        for job in jobs:
            assert isinstance(job, dict)
            if job.get("component") != "aim2_loco" or job.get("arm") != "family_cptac":
                continue
            job["arm"] = "bogus_arm"
            key = f"aim2.{job['stage']}.bogus_arm.seed{job['seed']}"
            job["job_id"] = key
            job["component_job_key"] = key

    _mutate_json_sources(
        paths,
        [
            (component_source, relabel_component),
            (scheduler_source, relabel_scheduler),
        ],
    )
    with pytest.raises(bundle.BundleVerificationError, match="Aim-2 exact scheduler job roster"):
        bundle.build_receipt(paths)


def test_aim3_adopted_component_roster_requires_the_e1v_gene_task(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    source = bundle._required_five_seed_sources(paths)[  # noqa: SLF001
        "aim3-ladders-five-seed-contract"
    ][0]

    def relabel_adopted_e1v(value: dict[str, object]) -> None:
        records = value["adopted_old_chains"]
        assert isinstance(records, list)
        for record in records:
            assert isinstance(record, dict)
            job = record["job"]
            assert isinstance(job, dict)
            if job.get("component") != "e1v":
                continue
            job["task"] = "bogus_gene"
            record["job_key"] = bundle._aim3_job_key(  # noqa: SLF001
                "e1v", "bogus_gene", None, int(job["model_seed"])
            )

    _mutate_json_source(paths, source, relabel_adopted_e1v)
    with pytest.raises(bundle.BundleVerificationError, match="exact fixed/repeated/E3v/E1v"):
        bundle.build_receipt(paths)


def test_aim3_coupled_component_and_scheduler_bogus_gene_fails_after_full_rehash(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    required = bundle._required_five_seed_sources(paths)  # noqa: SLF001
    component_source = required["aim3-ladders-five-seed-contract"][0]
    scheduler_source = required["final-v9-five-seed-campaign-contract"][0]

    def relabel_component(value: dict[str, object]) -> None:
        jobs = value["new_jobs"]
        assert isinstance(jobs, list)
        for job in jobs:
            assert isinstance(job, dict)
            if job.get("component") != "e1v":
                continue
            job["task"] = "bogus_gene"
            job["key"] = bundle._aim3_job_key(  # noqa: SLF001
                "e1v", "bogus_gene", None, int(job["model_seed"])
            )

    def relabel_scheduler(value: dict[str, object]) -> None:
        jobs = value["jobs"]
        assert isinstance(jobs, list)
        for job in jobs:
            assert isinstance(job, dict)
            if job.get("component") != "aim3_ladders" or job.get("stage") != "e1v":
                continue
            job["task"] = "bogus_gene"
            key = bundle._aim3_job_key(  # noqa: SLF001
                "e1v", "bogus_gene", None, int(job["seed"])
            )
            job["job_id"] = f"aim3.{key}"
            job["component_job_key"] = key

    _mutate_json_sources(
        paths,
        [
            (component_source, relabel_component),
            (scheduler_source, relabel_scheduler),
        ],
    )
    with pytest.raises(bundle.BundleVerificationError, match="Aim-3 exact scheduler task"):
        bundle.build_receipt(paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["precedence"]["superseded_source_report_json_pointers"].pop(),
            "pointer inventory",
        ),
        (
            lambda value: value["e2a_f"]["directions"]["CPTAC"]["auroc_ci95"].__setitem__(0, 0.49),
            "lower 95% AUROC bound",
        ),
        (
            lambda value: value["e2a_d"]["adjudication"].update(
                claim_sibling_stratum_transport=True
            ),
            "without macro rescue",
        ),
        (
            lambda value: value["e2a_d"]["macros"]["descriptive_equal_six_stratum"].update(
                role="descriptive; CAN rescue a failed direction"
            ),
            "without macro rescue",
        ),
        (
            lambda value: value["e2a_f"]["directions"]["CPTAC"].update(auroc_ci95=[0.51, 0.2]),
            "finite, ordered",
        ),
    ],
)
def test_adjudication_pointer_lower_ci_and_no_macro_rescue_contract_is_fail_closed(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    paths, _ = _fixture(tmp_path)
    result = bundle._required_five_seed_sources(paths)[  # noqa: SLF001
        "aim2-e2a-five-seed-adjudication-result"
    ][0]
    _mutate_json_source(paths, result, mutation)

    with pytest.raises(bundle.BundleVerificationError, match=message):
        bundle.build_receipt(paths)


def test_historical_upgraded_result_cannot_remain_unqualified_controlling_source(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    manifest_path = paths.final_v9 / bundle.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["id"] == "aim1-sealed-replay-results"
    )
    source["role"] = "controlling_results"
    _write_json(manifest_path, manifest)

    with pytest.raises(bundle.BundleVerificationError, match="still labeled controlling"):
        bundle.build_receipt(paths)


def test_source_manifest_does_not_recursively_follow_receipt_shaped_json(
    tmp_path: Path,
) -> None:
    paths, sources = _fixture(tmp_path)
    nested_missing = tmp_path / "does-not-exist.bin"
    sources["aim4"].write_text(
        json.dumps(
            {
                "status": "PASS",
                "outputs": [
                    {
                        "path": str(nested_missing),
                        "size_bytes": 7,
                        "sha256": "f" * 64,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = paths.final_v9 / bundle.SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = next(item for item in manifest["artifacts"] if item["id"] == "aim4-results")
    first["size_bytes"] = sources["aim4"].stat().st_size
    first["sha256"] = bundle.sha256_file(sources["aim4"])
    _write_json(manifest_path, manifest)

    receipt = bundle.build_receipt(paths)
    assert len(receipt["authoritative_sources"]) == 41
    assert not nested_missing.exists()


def test_default_verification_rejects_missing_or_tampered_receipt(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    with pytest.raises(bundle.BundleVerificationError, match="invalid final-v9 bundle receipt"):
        bundle.verify_published_receipt(paths)

    bundle.seal(paths)
    receipt = json.loads(paths.destination.read_text(encoding="utf-8"))
    receipt["status"] = "PASS"
    _write_json(paths.destination, receipt)
    with pytest.raises(bundle.BundleVerificationError, match="does not match"):
        bundle.verify_published_receipt(paths)
