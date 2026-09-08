"""Focused contracts for the narrow two-control Aim-3 continuation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import aim3_tcga_surgen_primary_two_fixed_controls_campaign as campaign  # noqa: E402


def _synthetic_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohorts = ["TCGA", "TCGA", "SurGen", "SurGen"]
    subcohorts = ["TCGA-COAD", "TCGA-READ", "SR386", "SR1482"]
    rows = []
    for index, (cohort, subcohort) in enumerate(zip(cohorts, subcohorts, strict=True)):
        rows.append(
            {
                "slide_id": f"pos-{index}",
                "patient_id": f"pos-{index}",
                "target_label": 1,
                "cohort": cohort,
                "subcohort": subcohort,
                "specimen_role": "primary",
                "k_fold": index,
                "kras": "mutant",
            }
        )
        rows.append(
            {
                "slide_id": f"fine-neg-{index}",
                "patient_id": f"fine-neg-{index}",
                "target_label": 0,
                "cohort": cohort,
                "subcohort": subcohort,
                "specimen_role": "primary",
                "k_fold": index,
                "kras": "mutant",
            }
        )
    fine = pd.DataFrame(rows)
    control = fine[fine["target_label"].eq(1)].copy()
    negatives = []
    for index, (cohort, subcohort) in enumerate(zip(cohorts, subcohorts, strict=True)):
        negatives.append(
            {
                "slide_id": f"ctrl-neg-{index}",
                "patient_id": f"ctrl-neg-{index}",
                "target_label": 0,
                "cohort": cohort,
                "subcohort": subcohort,
                "specimen_role": "primary",
                "k_fold": index,
                "kras": "wild_type",
            }
        )
    control = pd.concat([control, pd.DataFrame(negatives)], ignore_index=True)
    return fine, control


def test_exact_narrow_inventory_and_fit_accounting() -> None:
    jobs = campaign.job_inventory()
    assert len(jobs) == 10
    assert [job.key for job in jobs] == [
        *(f"fixed__ctrl_codon__seed{seed}" for seed in range(42, 47)),
        *(f"fixed__ctrl_g12d_broad__seed{seed}" for seed in range(42, 47)),
    ]
    assert campaign.fit_accounting() == {
        "task_variants": 2,
        "chains": 10,
        "oof_folds": 50,
        "p75_refits": 10,
        "physical_mil_fits": 60,
    }


def test_job_rejects_every_nonselected_task_or_seed() -> None:
    with pytest.raises(ValueError, match="unsupported fine task"):
        campaign.Job("g12c", 42)
    with pytest.raises(ValueError, match="unsupported model seed"):
        campaign.Job("codon", 47)


def test_real_frozen_plan_is_exact_train_one_and_never_broad_launcher() -> None:
    if not campaign.FROZEN_CONTROLLER.is_file():
        pytest.skip("certified Aim-3 source root is not mounted")
    plan = campaign.build_job_plan()
    assert len(plan) == 10
    expected_controller = str(campaign.FROZEN_CONTROLLER)
    for record in plan:
        command = record["command"]
        assert command[1:3] == [expected_controller, "_train-one"]
        assert command[3:5] == ["--output-root", str(campaign.SOURCE_ROOT)]
        assert command[5] == "--job-key"
        assert command[6] == record["job_key"]
        assert command[7:] == ["--num-workers", "4"]
        assert record["kind"] == "fixed"
        assert record["control_task"] in {"ctrl_codon", "ctrl_g12d_broad"}
        assert "repeated" not in record["job_key"]
        assert "--kind" not in command
        assert Path(record["source_output"]).is_relative_to(
            campaign.SOURCE_ROOT / "train/fixed"
        )


def test_real_source_authorities_and_population_are_still_pinned() -> None:
    if not campaign.SOURCE_ROOT.is_dir():
        pytest.skip("certified Aim-3 source root is not mounted")
    authorities = campaign._validate_source_authorities()
    assert {key: value["sha256"] for key, value in authorities.items()} == (
        campaign.PINNED_SOURCE_SHA256
    )
    census = campaign.validate_selected_manifests()
    assert set(census) == {"codon", "g12d_broad"}
    assert census["codon"]["patients"] == 501
    assert census["g12d_broad"]["patients"] == 501
    assert all(value["development_cohorts"] == ["SurGen", "TCGA"] for value in census.values())
    assert all(value["specimen_role"] == "primary" for value in census.values())


def test_pair_validator_requires_shared_positives_and_disjoint_matched_wt_negatives() -> None:
    fine, control = _synthetic_pair()
    observed = campaign.validate_pair_manifests(
        fine, control, "codon", production=False
    )
    assert observed["positive_roster_shared"] is True
    assert observed["negative_rosters_disjoint"] is True
    assert observed["control_negative_kras"] == "wild_type"
    assert observed["negative_matching"] == "subcohort x frozen outer fold"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda _fine, control: control.assign(specimen_role="metastatic"), "non-primary"),
        (lambda _fine, control: control.assign(cohort="CPTAC"), "TCGA\\+SurGen"),
        (lambda _fine, control: control.assign(kras="mutant"), "not all KRAS WT"),
    ],
)
def test_pair_validator_fails_closed_on_scope_drift(mutation, message: str) -> None:
    fine, control = _synthetic_pair()
    with pytest.raises(campaign.ContractError, match=message):
        campaign.validate_pair_manifests(
            fine, mutation(fine, control), "codon", production=False
        )


def test_pair_validator_rejects_overlap_and_cell_mismatch() -> None:
    fine, control = _synthetic_pair()
    overlap = control.copy()
    negative_index = overlap.index[overlap["target_label"].eq(0)][0]
    overlap.loc[negative_index, "patient_id"] = "fine-neg-0"
    with pytest.raises(campaign.ContractError, match="negative rosters overlap"):
        campaign.validate_pair_manifests(fine, overlap, "codon", production=False)

    mismatch = control.copy()
    negative_index = mismatch.index[mismatch["target_label"].eq(0)][0]
    mismatch.loc[negative_index, "k_fold"] = 4
    with pytest.raises(campaign.ContractError, match="subcohort/fold cells drifted"):
        campaign.validate_pair_manifests(fine, mismatch, "codon", production=False)


def test_output_root_isolated_and_fail_closed(tmp_path: Path) -> None:
    allowed = tmp_path / "continuation"
    assert campaign.validate_output_root(allowed) == allowed
    with pytest.raises(campaign.ContractError, match="absolute"):
        campaign.validate_output_root(Path("relative"))
    with pytest.raises(campaign.ContractError, match="overlap"):
        campaign.validate_output_root(campaign.SOURCE_ROOT / "nested")
    with pytest.raises(campaign.ContractError, match="exactly"):
        campaign.validate_output_root(Path("/home/yc_liu/not-governed"))


def test_output_root_rejects_symlink_ancestry(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(campaign.ContractError, match="symlink"):
        campaign.validate_output_root(alias / "continuation")


def test_strict_json_rejects_duplicate_keys_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status": "PASS", "status": "FAIL"}', encoding="utf-8")
    with pytest.raises(campaign.ContractError, match="invalid strict JSON"):
        campaign._read_json(duplicate)
    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(campaign.ContractError, match="invalid strict JSON"):
        campaign._read_json(nonfinite)


def test_identity_rejects_symlink_leaf(tmp_path: Path) -> None:
    material = tmp_path / "material.txt"
    material.write_text("evidence", encoding="utf-8")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(material)
    with pytest.raises(campaign.ContractError, match="symlinked"):
        campaign.identity(alias)


def test_peak_parallel_replays_overlap() -> None:
    events = [
        {"started_monotonic": 0.0, "finished_monotonic": 5.0},
        {"started_monotonic": 1.0, "finished_monotonic": 3.0},
        {"started_monotonic": 2.0, "finished_monotonic": 4.0},
        {"started_monotonic": 5.0, "finished_monotonic": 6.0},
    ]
    assert campaign._peak_parallel(events) == 3


def test_scheduler_validator_rejects_false_nonselected_absence_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "continuation"
    root.mkdir()
    records = [
        {"job_key": job.key, "command": ["frozen", job.key]}
        for job in campaign.job_inventory()
    ]
    events = [
        {
            "job_key": job.key,
            "returncode": 0,
            "command": ["frozen", job.key],
            "started_monotonic": float(index),
            "finished_monotonic": float(index + 1),
        }
        for index, job in enumerate(campaign.job_inventory())
    ]
    receipt = {
        "status": "completed_rc0",
        "contract": {"artifact": "contract"},
        "job_plan": {"artifact": "plan"},
        "preflight": {"artifact": "preflight"},
        "configured_max_parallel_chains": 6,
        "observed_max_parallel_chains": 2,
        "job_count": 10,
        "fit_accounting": campaign.fit_accounting(),
        "job_keys": [job.key for job in campaign.job_inventory()],
        "commands_exactly_frozen_train_one": True,
        "nonselected_fixed_or_repeated_jobs_dispatched": 0,
        "nonselected_source_control_artifacts_absent_at_completion": True,
        "events": events,
    }
    # The chosen closed intervals touch at their endpoints; the replay rule
    # counts finish/start ties concurrently and therefore produces peak=2.
    monkeypatch.setattr(campaign, "verify_contract", lambda *_args, **_kwargs: {"source_root": "/tmp/source"})
    monkeypatch.setattr(campaign, "_plan_records", lambda _root: records)
    monkeypatch.setattr(campaign, "identity", lambda path: {
        "artifact": "contract" if path.name == "contract.json" else (
            "plan" if path.name == "job_plan.json" else "preflight"
        )
    })
    monkeypatch.setattr(campaign, "_validate_continuation_job", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(campaign, "_read_json", lambda _path: receipt)
    campaign._validate_scheduler(root)
    receipt["nonselected_source_control_artifacts_absent_at_completion"] = False
    with pytest.raises(campaign.ContractError, match="scheduler receipt semantics"):
        campaign._validate_scheduler(root)


def test_training_seal_validator_rejects_false_nonselected_absence_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "continuation"
    root.mkdir()
    records = [
        {"job_key": job.key, "command": ["frozen", job.key]}
        for job in campaign.job_inventory()
    ]
    wrappers = [{"artifact": job.key} for job in campaign.job_inventory()]
    source_receipts = [{"source": job.key} for job in campaign.job_inventory()]
    receipt = {
        "status": "complete_and_certified",
        "campaign": campaign.CAMPAIGN,
        "population": "tcga_surgen_primary",
        "encoder": "UNI-v1",
        "model_seeds": list(campaign.MODEL_SEEDS),
        "fine_tasks": list(campaign.FINE_TASKS),
        "control_kind": "canonical_fixed_matched_wt",
        "repeated_wt_controls_scheduled": False,
        "external_development_cohorts": [],
        "fit_accounting": campaign.fit_accounting(),
        "contract": {"fixed": "contract.json"},
        "preflight": {"fixed": "preflight.json"},
        "scheduler": {"fixed": "scheduler.json"},
        "nonselected_source_control_artifacts_absent_at_seal": True,
        "continuation_job_receipts": wrappers,
        "source_job_receipts": source_receipts,
    }
    monkeypatch.setattr(campaign, "_read_json", lambda _path: receipt)
    monkeypatch.setattr(campaign, "identity", lambda path: (
        {"artifact": path.stem}
        if path.parent.name == "jobs"
        else {"fixed": path.name}
    ))
    monkeypatch.setattr(campaign, "verify_contract", lambda *_args, **_kwargs: {"source_root": "/tmp/source"})
    monkeypatch.setattr(campaign, "_plan_records", lambda _root: records)
    monkeypatch.setattr(
        campaign,
        "_validate_continuation_job",
        lambda _root, _source, job, _command: {"source_job_receipt": {"source": job.key}},
    )
    campaign._validate_training_seal(root)
    receipt["nonselected_source_control_artifacts_absent_at_seal"] = False
    with pytest.raises(campaign.ContractError, match="training seal drifted"):
        campaign._validate_training_seal(root)


def test_contract_semantics_exclude_external_and_repeated_controls(tmp_path: Path) -> None:
    if not campaign.SOURCE_ROOT.is_dir():
        pytest.skip("certified Aim-3 source root is not mounted")
    semantics = campaign._contract_semantics(tmp_path / "continuation")
    assert semantics["population"] == "tcga_surgen_primary"
    assert semantics["external_development_cohorts"] == []
    assert semantics["repeated_wt_controls_scheduled"] is False
    assert semantics["nonselected_fixed_controls_scheduled"] is False
    assert semantics["fit_accounting"]["physical_mil_fits"] == 60


def test_result_key_guard_detects_prohibited_claim_keys() -> None:
    keys = set(campaign._walk_keys({"rungs": [{"consensus_verdict": "x"}]}))
    assert "consensus_verdict" in keys
    assert "ceiling" not in keys


def test_cli_has_no_broad_or_internal_training_subcommand() -> None:
    parser = campaign.build_parser()
    for command in ("plan", "prepare", "preflight", "train", "validate", "analyze", "verify"):
        namespace = parser.parse_args([command, "--output-root", "/tmp/narrow-two-controls"])
        assert callable(namespace.func)
    with pytest.raises(SystemExit):
        parser.parse_args(["_train-one"])
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--kind", "controls"])


def test_controller_text_never_dispatches_broad_control_phase() -> None:
    text = Path(campaign.__file__).read_text(encoding="utf-8")
    assert '"--kind", "controls"' not in text
    assert "cmd_train_one" not in text
    assert "consensus_verdict(" not in text
    assert "CONTROL_FOR =" in text
    assert set(campaign.CONTROL_FOR.values()) == {"ctrl_codon", "ctrl_g12d_broad"}


def test_job_plan_json_shape_is_strict_object(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema_version": 1, "jobs": []}), encoding="utf-8")
    assert campaign._read_json(path) == {"schema_version": 1, "jobs": []}
