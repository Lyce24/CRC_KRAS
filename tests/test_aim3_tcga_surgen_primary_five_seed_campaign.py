"""Focused CPU contracts for the governed Aim-3 source-only campaign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim3_tcga_surgen_primary_five_seed_campaign as campaign  # noqa: E402


def _source() -> pd.DataFrame:
    rows = []
    index = 0
    subcohorts = (
        ("TCGA", "TCGA-COAD"),
        ("TCGA", "TCGA-READ"),
        ("SurGen", "SR386"),
        ("SurGen", "SR1482"),
    )
    definitions = (
        ("wild_type", "", 20),
        ("mutant", "G12D", 3),
        ("mutant", "G12V", 3),
        ("mutant", "G12C", 3),
        ("mutant", "G12A", 3),
        ("mutant", "G13D", 3),
    )
    for cohort, subcohort in subcohorts:
        for fold in range(5):
            for kras, subvariant, count in definitions:
                for slot in range(count):
                    patient = f"P{index:05d}"
                    row = {
                        "slide_id": f"S{index:05d}",
                        "patient_id": patient,
                        "target_label": int(kras == "mutant"),
                        "kras": kras,
                        "kras_subvariant": subvariant,
                        "cohort": cohort,
                        "subcohort": subcohort,
                        "specimen_role": "primary",
                        "k_fold": fold,
                    }
                    for held_out in range(5):
                        row[f"val_fold_{held_out}"] = int(
                            fold != held_out and slot == 0
                        )
                    rows.append(row)
                    index += 1
    return pd.DataFrame(rows)


def test_inventory_and_physical_fit_accounting_are_exact():
    jobs = campaign.job_inventory()
    assert len(jobs) == 125
    assert len({job.key for job in jobs}) == 125
    assert {job.model_seed for job in jobs} == {42, 43, 44, 45, 46}
    assert sum(job.kind == "fine" for job in jobs) == 25
    assert sum(job.kind == "fixed" for job in jobs) == 25
    assert sum(job.kind == "repeated" for job in jobs) == 75
    assert campaign.fit_accounting() == {
        "task_variants": 25,
        "chains": 125,
        "oof_folds": 625,
        "p75_refits": 125,
        "physical_mil_fits": 750,
    }
    first_repeat = next(i for i, job in enumerate(jobs) if job.kind == "repeated")
    assert first_repeat == 50
    assert all(job.kind != "repeated" for job in jobs[:first_repeat])
    assert campaign.phase_accounting("fine") == {
        "task_variants": 5,
        "chains": 25,
        "oof_folds": 125,
        "p75_refits": 25,
        "physical_mil_fits": 150,
    }
    assert campaign.phase_accounting("controls") == {
        "task_variants": 20,
        "chains": 100,
        "oof_folds": 500,
        "p75_refits": 100,
        "physical_mil_fits": 600,
    }


def test_contract_is_source_only_univ1_and_six_wide(tmp_path: Path):
    contract = campaign.contract_semantics(tmp_path)
    assert contract["population"]["source_arm"] == "tcga_surgen_primary"
    assert contract["population"]["external_validation_cohorts_used_for_development"] == []
    assert contract["population"]["forbidden_development_cohorts"] == [
        "CPTAC",
        "Orion",
        "RIH",
        "SurGen-met",
    ]
    assert contract["protocol"]["encoder"] == "UNI-v1"
    assert contract["protocol"]["model_seeds"] == [42, 43, 44, 45, 46]
    assert contract["protocol"]["max_parallel_chains"] == 6
    assert contract["protocol"]["finalize"] == "required p75 full-data refit for every chain"
    assert campaign.DEFAULT_OUTPUT_ROOT.name == (
        "aim3_tcga_surgen_primary_univ1_5seed_v2_20260827"
    )


def test_material_snapshot_closes_root_and_tools_python_import_graph():
    files = set(campaign.material_source_files())
    expected_root_python = {
        path.name
        for path in campaign.REPO.glob("*.py")
        if path.is_file() and not path.is_symlink()
    }
    expected_tools_python = {
        path.relative_to(campaign.REPO).as_posix()
        for path in (campaign.REPO / "tools").rglob("*.py")
        if path.is_file() and not path.is_symlink()
    }
    assert expected_root_python <= files
    assert expected_tools_python <= files
    assert {
        "aim1_virchow2_encoder_arm.py",
        "aim3_virchow2_replication.py",
        "aim3_ladders_five_seed_extension.py",
        "tools/study_train.py",
        "src/oceanpath/workflows/training.py",
        "configs/train.yaml",
        "pyproject.toml",
        "uv.lock",
    } <= files
    assert not any(
        relative.startswith(("outputs/", ".git/"))
        or relative.endswith((".env", ".csv", ".parquet", ".ckpt"))
        for relative in files
    )


def test_frozen_controller_import_and_plan_smoke_from_prepared_root(tmp_path: Path):
    root = tmp_path / "prepared"
    root.mkdir()
    campaign._snapshot_sources(root)
    receipt = campaign.smoke_frozen_controller(root)
    assert receipt["status"] == "PASS"
    assert receipt["returncode"] == 0
    assert receipt["job_count"] == 125
    assert receipt["fit_accounting"] == campaign.fit_accounting()


def test_training_command_pins_univ1_oof_and_p75_refit(tmp_path: Path):
    job = campaign.Job("repeated", "allele1", 46, 20260825)
    command = campaign.train_command(tmp_path, job, num_workers=4, attempt_id="test")
    joined = " ".join(command)
    assert command[1] == str(tmp_path / "source_snapshot/tools/study_train.py")
    assert "encoder=univ1" in joined
    assert "training.skip_finalize=false" in joined
    assert "training.fixed_epoch_budget=12" in joined
    assert "training.dataset_max_instances=8192" in joined
    assert "training.eval_full_bags=true" in joined
    assert "training.train_sampling_strategy=patient_natural" in joined
    assert "training.num_workers=4" in joined
    assert "wt20260825" in joined
    assert not any(name.casefold() in joined.casefold() for name in ("CPTAC", "Orion", "RIH"))


def test_external_or_metastatic_rows_are_rejected():
    source = _source()
    source.loc[source.index[0], "cohort"] = "CPTAC"
    with pytest.raises(campaign.ContractError, match="exactly"):
        campaign.validate_source_population(source, production=False)
    source = _source()
    source.loc[source.index[0], "specimen_role"] = "metastatic"
    with pytest.raises(campaign.ContractError, match="primary"):
        campaign.validate_source_population(source, production=False)


def test_population_specific_fine_and_controls_inherit_source_folds():
    source = _source()
    fine = campaign.build_task_manifest(source, "fine", "allele1")
    fixed = campaign.build_task_manifest(source, "fixed", "allele1")
    labels = fine.drop_duplicates("patient_id")["target_label"].value_counts().to_dict()
    controls = fixed.drop_duplicates("patient_id")["target_label"].value_counts().to_dict()
    assert labels == controls
    fine_positive = set(
        fine.drop_duplicates("patient_id").loc[lambda x: x.target_label.eq(1), "patient_id"]
    )
    fixed_positive = set(
        fixed.drop_duplicates("patient_id").loc[lambda x: x.target_label.eq(1), "patient_id"]
    )
    assert fine_positive == fixed_positive
    fixed_negative = fixed.drop_duplicates("patient_id").loc[lambda x: x.target_label.eq(0)]
    assert fixed_negative["kras"].eq("wild_type").all()
    inherited = fixed.merge(
        source[["slide_id", "k_fold", *(f"val_fold_{i}" for i in range(5))]],
        on="slide_id",
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    for column in ["k_fold", *(f"val_fold_{i}" for i in range(5))]:
        assert inherited[column].equals(inherited[f"{column}_source"])


def test_repeated_controls_are_three_distinct_exact_stratum_matches():
    source = _source()
    fine = campaign.fine_patient_labels(source, "g12d_broad")
    expected = (
        fine.loc[fine.target_label.eq(0)]
        .groupby(["subcohort", "k_fold"], sort=True)
        .size()
    )
    rosters = []
    for draw in campaign.WT_DRAW_SEEDS:
        control = campaign.draw_control_patients(source, "g12d_broad", draw)
        negative = control.loc[control.target_label.eq(0)]
        observed = negative.groupby(["subcohort", "k_fold"], sort=True).size()
        pd.testing.assert_series_equal(observed, expected)
        rosters.append(frozenset(negative.patient_id))
    assert len(set(rosters)) == 3


def test_all_task_manifests_have_exact_variant_roster():
    manifests = campaign.build_all_task_manifests(_source())
    assert len(manifests) == 25
    assert sum(key[0] == "fine" for key in manifests) == 5
    assert sum(key[0] == "fixed" for key in manifests) == 5
    assert sum(key[0] == "repeated" for key in manifests) == 15
    assert all(set(frame.cohort) == {"TCGA", "SurGen"} for frame in manifests.values())
    assert all(set(frame.specimen_role) == {"primary"} for frame in manifests.values())


def test_default_parser_requires_six_parallel_chains():
    args = campaign.build_parser().parse_args(["train"])
    assert args.jobs == 6
    assert args.num_workers == 4
    assert args.apply is False
    fine = campaign.build_parser().parse_args(["train", "--kind", "fine"])
    controls = campaign.build_parser().parse_args(["train", "--kind", "controls"])
    assert fine.kind == "fine"
    assert controls.kind == "controls"


def test_fine_phase_inventory_contains_no_control_chain():
    fine = campaign.phase_jobs("fine")
    assert len(fine) == 25
    assert all(job.kind == "fine" and job.draw_seed is None for job in fine)
    assert {job.task for job in fine} == set(campaign.FINE_TASKS)
    controls = campaign.phase_jobs("controls")
    assert len(controls) == 100
    assert all(job.kind in {"fixed", "repeated"} for job in controls)


def test_fine_phase_fails_closed_if_any_wt_request_or_run_exists(tmp_path: Path):
    campaign.assert_no_control_artifacts(tmp_path)
    control = campaign.Job("repeated", "codon", 42, campaign.WT_DRAW_SEEDS[0])
    request = campaign.request_path(tmp_path, control)
    request.parent.mkdir(parents=True)
    request.write_text("{}")
    with pytest.raises(campaign.ContractError, match="fine-only phase forbids"):
        campaign.assert_no_control_artifacts(tmp_path)


def test_control_phase_resume_allows_certified_cached_and_untouched_pending(
    tmp_path: Path, monkeypatch
):
    completed = campaign.phase_jobs("controls")[0]
    receipt = campaign.job_receipt_path(tmp_path, completed)
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    campaign.run_dir(tmp_path, completed).mkdir(parents=True)
    calls = []
    monkeypatch.setattr(
        campaign,
        "validate_job",
        lambda root, job: calls.append((root, job.key)) or {},
    )
    state = campaign.audit_control_resume_state(tmp_path)
    assert state["completed"] == [completed.key]
    assert len(state["pending"]) == 99
    assert calls == [(tmp_path, completed.key)]

    partial = campaign.phase_jobs("controls")[1]
    request = campaign.request_path(tmp_path, partial)
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text("{}")
    with pytest.raises(campaign.ContractError, match="uncertified partial"):
        campaign.audit_control_resume_state(tmp_path)


def test_fine_scheduler_receipt_binds_exact_six_way_25_chain_phase(tmp_path: Path):
    events = [
        {
            "job_id": f"aim3.source.{job.key}",
            "returncode": 0,
            "started_monotonic": float(index),
            "finished_monotonic": float(index + 6),
        }
        for index, job in enumerate(campaign.phase_jobs("fine"))
    ]
    path = campaign.fine_scheduler_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        campaign.json.dumps(
            {
                "schema_version": campaign.SCHEMA_VERSION,
                "status": "fine_phase_completed_rc0",
                "phase": "fine",
                "configured_max_parallel_chains": 6,
                "observed_max_parallel_chains": 6,
                "job_count": 25,
                "fit_accounting": campaign.phase_accounting("fine"),
                "job_keys": sorted(job.key for job in campaign.phase_jobs("fine")),
                "control_artifacts_absent": True,
                "events": events,
            }
        )
    )
    receipt = campaign._validate_fine_scheduler(tmp_path, require_no_controls=True)
    assert receipt["configured_max_parallel_chains"] == 6
    assert receipt["observed_max_parallel_chains"] == 6


def test_composite_phased_scheduler_binds_fine_and_control_receipts(tmp_path: Path):
    fine_path = campaign.fine_scheduler_path(tmp_path)
    controls_path = campaign.controls_scheduler_path(tmp_path)
    fine_path.parent.mkdir(parents=True)
    fine_path.write_text("{}")
    controls_path.write_text("{}")
    events = [
        {"job_id": f"j{index}", "returncode": 0}
        for index in range(campaign.TOTAL_CHAINS)
    ]
    campaign.scheduler_path(tmp_path).write_text(
        campaign.json.dumps(
            {
                "status": "completed_phased_rc0",
                "configured_max_parallel_chains": 6,
                "observed_max_parallel_chains": 6,
                "job_count": 125,
                "fit_accounting": campaign.fit_accounting(),
                "fixed_ladder_scheduled_before_repeated": True,
                "phase_order": ["fine", "fine_analysis_candidate", "controls"],
                "fine_scheduler": campaign.identity(fine_path),
                "controls_scheduler": campaign.identity(controls_path),
                "events": events,
            }
        )
    )
    receipt = campaign._validate_scheduler(tmp_path)
    assert receipt["status"] == "completed_phased_rc0"


def test_plan_and_prepare_dry_run_are_read_only(tmp_path: Path, capsys):
    root = tmp_path / "new-campaign"
    campaign.cmd_plan(argparse.Namespace(output_root=root))
    assert not root.exists()
    campaign.cmd_prepare(
        argparse.Namespace(output_root=root, source_root=tmp_path / "unused", apply=False)
    )
    assert not root.exists()
    assert "DRY RUN" in capsys.readouterr().out


def test_output_root_rejects_source_tree_and_non_tmp_arbitrary_root():
    with pytest.raises(campaign.ContractError, match="production root"):
        campaign.validate_output_root(Path("/var/tmp/not-the-campaign"))
    with pytest.raises(campaign.ContractError, match="overlaps"):
        campaign.validate_output_root(campaign.SOURCE_ROOT)


def test_gate_and_three_draw_consensus_are_prespecified():
    fine = {"primary_fwer_one_sided": {"lower": 0.47, "upper": 0.58}}
    control = {"primary_fwer_one_sided": {"lower": 0.61, "upper": 0.72}}
    delta = {"primary_fwer_one_sided": {"lower": 0.03, "upper": 0.18}}
    assert campaign._gate(fine, control, delta)["verdict"] == "CEILING"
    assert campaign.repeated_stats.consensus_verdict(
        ["CEILING", "CEILING", "CEILING"]
    ) == "CONSENSUS_CEILING"


def test_source_fine_census_constants_match_requested_population():
    assert campaign.EXPECTED_FINE_PATIENTS == {
        "codon": {"positive": 354, "negative": 147},
        "g12d_broad": {"positive": 154, "negative": 347},
        "allele1": {"positive": 154, "negative": 200},
        "allele2": {"positive": 102, "negative": 252},
        "g12c": {"positive": 46, "negative": 308},
    }
    assert campaign.SOURCE_EXPECTED_SHA256["manifest"].startswith("d7087a23")
    assert campaign.SOURCE_EXPECTED_SHA256["splits"].startswith("31046858")
    assert campaign.SOURCE_EXPECTED_SHA256["training_seal"].startswith("31cc8005")


def test_peak_parallel_evidence_counts_overlapping_chains():
    events = [
        {"started_monotonic": float(index), "finished_monotonic": float(index + 6)}
        for index in range(6)
    ]
    assert campaign._peak_parallel(events) == 6


def test_patient_native_logit_ensemble_convention(monkeypatch, tmp_path: Path):
    manifest = pd.DataFrame(
        {
            "slide_id": ["S1", "S2"],
            "patient_id": ["P", "P"],
            "target_label": [1, 1],
            "cohort": ["TCGA", "TCGA"],
            "subcohort": ["TCGA-COAD", "TCGA-COAD"],
            "k_fold": [0, 0],
        }
    )
    monkeypatch.setattr(campaign.pd, "read_csv", lambda *_a, **_k: manifest)
    seed_values = iter(range(5))

    def fake_read(_path):
        value = float(next(seed_values))
        return pd.DataFrame(
            {
                "slide_id": ["S1", "S2"],
                "label": [1, 1],
                "logit": [value, value + 2.0],
                "fold": [0, 0],
            }
        )

    monkeypatch.setattr(campaign.pd, "read_parquet", fake_read)
    monkeypatch.setattr(campaign.fixed_stats, "_task_point", lambda _frame: 0.5)
    ensemble, per_seed = campaign._ensemble(tmp_path, "fine", "codon")
    # Each seed first averages its two slides: [1,2,3,4,5], then seeds average to 3.
    assert ensemble.loc[0, "mean_logit"] == pytest.approx(3.0)
    assert set(per_seed) == {"42", "43", "44", "45", "46"}
