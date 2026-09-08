"""Focused contracts for the additive five-seed Aim-3 extension."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim3_ladders_five_seed_extension as extension  # noqa: E402
from oceanpath.datasets.packed import pack_features  # noqa: E402


def test_new_job_inventory_is_exact_and_trains_only_45_46():
    jobs = extension.job_inventory()
    assert len(jobs) == 64
    assert {job.model_seed for job in jobs} == {45, 46}
    assert sum(job.component == "fixed" for job in jobs) == 20
    assert sum(job.component == "repeated" for job in jobs) == 30
    assert sum(job.component == "e3v" for job in jobs) == 12
    assert sum(job.component == "e1v" for job in jobs) == 2
    assert len({job.key for job in jobs}) == len(jobs)


def test_new_fit_accounting_includes_real_refits():
    assert extension.job_counts() == {
        "chains": 64,
        "oof_folds": 320,
        "refits": 50,
        "actual_mil_fits": 370,
    }
    fixed = extension.job_counts(
        job for job in extension.job_inventory() if job.component == "fixed"
    )
    repeated = extension.job_counts(
        job for job in extension.job_inventory() if job.component == "repeated"
    )
    assert fixed == {"chains": 20, "oof_folds": 100, "refits": 20, "actual_mil_fits": 120}
    assert repeated == {"chains": 30, "oof_folds": 150, "refits": 30, "actual_mil_fits": 180}


def test_old_inventory_is_adoption_only():
    old = extension.job_inventory(seeds=extension.OLD_SEEDS)
    assert len(old) == 96
    with pytest.raises(ValueError, match="adoption-only"):
        extension.train_command(Path("/tmp/aim3"), old[0])


@pytest.mark.parametrize(
    ("job", "expected_fragment", "skip_finalize"),
    [
        (extension.Job("fixed", "codon", 45), "fixed_univ1/codon/seed45", False),
        (
            extension.Job("repeated", "ctrl_codon", 46, 20260823),
            "repeated_univ1/ctrl_codon/wt20260823/seed46",
            False,
        ),
        (extension.Job("e3v", "codon", 45), "e3v_virchow2_cls/codon/seed45", True),
        (extension.Job("e1v", "gene", 46), "e1v_virchow2_cls/seed46", True),
    ],
)
def test_train_commands_are_overlay_scoped_and_pin_finalize_contract(
    tmp_path: Path,
    job: extension.Job,
    expected_fragment: str,
    skip_finalize: bool,
):
    root = tmp_path / "aim3_ladders"
    command = extension.train_command(root, job, num_workers=4, attempt_id="test")
    joined = " ".join(command)
    assert expected_fragment in joined
    assert f"training.skip_finalize={str(skip_finalize).lower()}" in joined
    assert "training.dataset_max_instances=8192" in joined
    assert "training.eval_full_bags=true" in joined
    assert "training.train_sampling_strategy=patient_natural" in joined
    assert "training.num_workers=4" in joined
    assert "hydra.job.chdir=false" in joined
    assert str(root / "source_snapshot" / "tools" / "study_train.py") == command[1]
    assert str(extension.OLD_FIXED_ROOT / "train") not in joined
    assert str(extension.OLD_REPEATED_ROOT / "train") not in joined


def test_refit_and_oof_only_commands_preserve_the_declared_recipe(tmp_path: Path):
    fixed = " ".join(
        extension.train_command(tmp_path, extension.Job("fixed", "g12c", 45))
    )
    repeated = " ".join(
        extension.train_command(
            tmp_path,
            extension.Job("repeated", "ctrl_g12c", 45, 20260824),
        )
    )
    e3v = " ".join(
        extension.train_command(tmp_path, extension.Job("e3v", "g12d_broad", 45))
    )
    e1v = " ".join(extension.train_command(tmp_path, extension.Job("e1v", "gene", 45)))
    for command in (fixed, repeated):
        assert "training.skip_finalize=false" in command
        assert "training.fixed_epoch_budget=12" in command
        assert "encoder=univ1" in command
    for command in (e3v, e1v):
        assert "training.skip_finalize=true" in command
        assert "encoder=virchow2" in command
        assert "encoder.feature_dim=1280" in command
    assert "training.fixed_epoch_budget=12" in e3v
    assert "training.fixed_epoch_budget=12" not in e1v


def test_repeated_command_consumes_frozen_overlay_manifest_and_split(tmp_path: Path):
    job = extension.Job("repeated", "ctrl_allele1", 46, 20260825)
    joined = " ".join(extension.train_command(tmp_path, job))
    assert f"data.csv_path={extension.repeated_manifest(tmp_path, job.task, 20260825)}" in joined
    assert f"+splits.output_dir={extension.repeated_splits(tmp_path, job.task, 20260825)}" in joined
    assert str(extension.repeated_manifest_source(job.task, 20260825)) not in joined


def test_default_scheduler_is_globally_six_wide():
    args = extension.build_parser().parse_args(["train"])
    assert args.jobs == 6
    assert args.num_workers == 4
    assert args.model_seed is None


def test_external_job_inventory_is_one_slot_guarded_and_exact(tmp_path: Path):
    jobs = extension.build_training_jobs(tmp_path)
    assert len(jobs) == 64
    assert sum(job["fit_count"] for job in jobs) == 370
    assert all(job["scheduler_slots"] == 1 for job in jobs)
    assert all(job["external_worker_command"] == job["command"] for job in jobs)
    assert all("train-one" in job["external_worker_command"] for job in jobs)
    assert all("--external-scheduler" in job["external_worker_command"] for job in jobs)
    assert all("hydra-train" not in job["external_worker_command"] for job in jobs)
    assert [job["stage"] for job in jobs[:2]] == ["e1v", "e1v"]
    first = extension.job_from_key(jobs[0]["job_key"])
    assert first.component == "e1v"
    with pytest.raises(ValueError, match="unknown"):
        extension.job_from_key("not-a-job")


def test_manifest_dry_run_does_not_create_root(tmp_path: Path, capsys):
    root = tmp_path / "new_lineage"
    extension.cmd_manifest(
        argparse.Namespace(output_root=str(root), e0_root=str(tmp_path / "e0"), apply=False)
    )
    assert not root.exists()
    assert "DRY RUN" in capsys.readouterr().out


def test_training_requires_a_persisted_full_pack_preflight(tmp_path: Path):
    root = tmp_path / "lineage"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        extension._verify_preflight_receipt(root)


def test_output_root_rejects_a_sealed_source_tree():
    with pytest.raises(ValueError, match="root must be"):
        extension.validate_output_root(extension.OLD_FIXED_ROOT)


def test_p75_refit_validator_binds_summary_info_and_checkpoint(tmp_path: Path):
    refit = {
        "strategy": "refit",
        "refit_epochs": 5,
        "refit_epoch_rule": "p75",
        "fold_best_epochs": [2, 4, 5, 5, 9],
        "model_path": str(tmp_path / "final" / "refit" / "model.ckpt"),
    }
    final = tmp_path / "final"
    (final / "refit").mkdir(parents=True)
    (final / "finalize_summary.json").write_text(json.dumps({"refit": refit}))
    (final / "refit" / "info.json").write_text(json.dumps(refit))
    (final / "refit" / "model.ckpt").write_bytes(b"checkpoint")
    record = extension._validate_refit(tmp_path)
    assert record["refit_epochs"] == 5
    assert record["refit_checkpoint"]["sha256"] == extension.sha256_file(
        final / "refit" / "model.ckpt"
    )
    refit["refit_epochs"] = 4
    (final / "refit" / "info.json").write_text(json.dumps(refit))
    with pytest.raises(RuntimeError, match="summary/info mismatch"):
        extension._validate_refit(tmp_path)


def test_selected_pack_equivalence_detects_content_drift(tmp_path: Path):
    source = tmp_path / "features"
    source.mkdir()
    for index, slide in enumerate(("S1", "S2")):
        features = np.arange(24, dtype=np.float32).reshape(3, 8) + index
        coords = np.arange(6, dtype=np.int64).reshape(3, 2)
        with h5py.File(source / f"{slide}.h5", "w") as handle:
            handle.create_dataset("features", data=features)
            handle.create_dataset("coords", data=coords)
    pack = tmp_path / "packed"
    pack_features(source, pack)
    receipt = extension.verify_selected_pack_against_h5(pack, ["S1", "S2"])
    assert receipt["n_slides"] == 2
    assert len(receipt["selected_content_sha256"]) == 64
    with h5py.File(source / "S2.h5", "r+") as handle:
        handle["features"][0, 0] += 10
    with pytest.raises(RuntimeError, match="feature mismatch"):
        extension.verify_selected_pack_against_h5(pack, ["S2"])


def test_selected_pack_equivalence_replays_cls_prefix_transform(tmp_path: Path):
    source = tmp_path / "features_full"
    packed_source = tmp_path / "features_cls"
    source.mkdir()
    packed_source.mkdir()
    features = np.arange(48, dtype=np.float32).reshape(3, 16)
    coords = np.arange(6, dtype=np.int64).reshape(3, 2)
    for directory, values in ((source, features), (packed_source, features[:, :8])):
        with h5py.File(directory / "S1.h5", "w") as handle:
            handle.create_dataset("features", data=values)
            handle.create_dataset("coords", data=coords)
    pack = tmp_path / "packed_cls"
    pack_features(packed_source, pack)
    meta_path = pack / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["source_dir"] = str(source)
    meta_path.write_text(json.dumps(meta))

    receipt = extension.verify_selected_pack_against_h5(pack, ["S1"])
    assert receipt["n_slides"] == 1

    with h5py.File(source / "S1.h5", "r+") as handle:
        handle["features"][0, 0] += 10
    with pytest.raises(RuntimeError, match="feature mismatch"):
        extension.verify_selected_pack_against_h5(pack, ["S1"])


def test_gate_is_the_predeclared_one_sided_99_intersection():
    fine = {"primary_fwer_one_sided": {"lower": 0.48, "upper": 0.58}}
    control = {"primary_fwer_one_sided": {"lower": 0.61, "upper": 0.72}}
    delta = {"primary_fwer_one_sided": {"lower": 0.03, "upper": 0.19}}
    gate = extension._gate(fine, control, delta)
    assert gate["ceiling"] is True
    assert gate["verdict"] == "CEILING"


def test_bootstrap_streams_are_exactly_the_sealed_derivations():
    assert {
        fine: extension.fixed_stats._stable_seed(
            extension.BOOTSTRAP_SEED, "primary_pair", fine
        )
        for fine, _control in extension.FIXED_PAIRS
    } == extension.FIXED_PAIR_BOOTSTRAP_SEEDS
    assert extension.fixed_stats._stable_seed(
        extension.BOOTSTRAP_SEED, "standalone", "gene"
    ) == extension.GENE_BOOTSTRAP_SEED
    assert extension.E3V_PAIR_BOOTSTRAP_SEED == 20260817


def test_e0_is_an_explicit_fail_closed_sealing_dependency(tmp_path: Path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        extension._require_e0_dependency(tmp_path / "missing")
    root = tmp_path / "e0"
    (root / "receipts").mkdir(parents=True)
    expected_roots = {
        str(seed): str(
            extension.aim1_extension.resolve_run_dir(
                root, extension.aim1_extension.DEFAULT_LEGACY_ROOT, seed
            ).resolve()
        )
        for seed in extension.ALL_SEEDS
    }
    receipt = {
        "status": "complete",
        "model_seeds": list(extension.ALL_SEEDS),
        "resolved_roots": expected_roots,
        "runs": {str(seed): {} for seed in extension.ALL_SEEDS},
        "fold_layout_shared_across_all_seeds": True,
        "p75_refit_authenticated_for_all_seeds": True,
    }
    receipt_path = root / "receipts" / "five_seed_training_validation.json"
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(
        extension.aim1_extension,
        "_validate_five_seed_runs",
        lambda *_args: (receipt, "authenticated-patient-table"),
    )
    dependency = extension._require_e0_dependency(root)
    assert dependency["payload"]["status"] == "complete"
    assert dependency["patients"] == "authenticated-patient-table"


def test_e0_integration_uses_aim1_controller_paths_and_resolver():
    aim1 = extension.aim1_extension
    assert extension.DEFAULT_E0_ROOT == aim1.DEFAULT_CAMPAIGN_ROOT
    assert extension.OLD_E0_ROOT == aim1.DEFAULT_LEGACY_ROOT
    assert tuple(aim1.ALL_SEEDS) == extension.ALL_SEEDS
    assert aim1.resolve_run_dir(extension.DEFAULT_E0_ROOT, extension.OLD_E0_ROOT, 45) == (
        extension.DEFAULT_E0_ROOT / "train/1a_pb_cap8192/univ1/seed45"
    )
    assert (
        extension.DEFAULT_E0_ROOT / "receipts/five_seed_training_validation.json"
    ).name == "five_seed_training_validation.json"


def test_recompute_analysis_payload_routes_every_governed_analysis(tmp_path: Path, monkeypatch):
    calls = []
    manifest = object()
    monkeypatch.setattr(
        extension,
        "_three_seed_replay",
        lambda root, e0_root: calls.append(("replay", root, e0_root)) or {"replay": True},
    )
    monkeypatch.setattr(
        extension,
        "_fixed_analysis",
        lambda root: calls.append(("fixed", root))
        or ({"fixed": True}, {"fixed_draws": np.array([1.0, 2.0])}),
    )
    monkeypatch.setattr(
        extension,
        "_repeated_analysis",
        lambda root: calls.append(("repeated", root))
        or ({"repeated": True}, {"repeated_draws": np.array([3.0])}),
    )
    monkeypatch.setattr(
        extension,
        "_e3v_analysis",
        lambda root: calls.append(("e3v", root))
        or ({"e3v": True}, {"e3v_draws": np.array([4.0])}),
    )
    monkeypatch.setattr(
        extension,
        "_e1v_analysis",
        lambda root: calls.append(("e1v", root)) or {"e1v": True},
    )
    monkeypatch.setattr(extension.pd, "read_csv", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        extension,
        "_e0_analysis",
        lambda e0_root, frame: calls.append(("e0", e0_root, frame)) or {"e0": True},
    )

    report, arrays = extension._recompute_analysis_payload(tmp_path, tmp_path / "e0")

    assert [call[0] for call in calls] == ["replay", "fixed", "repeated", "e3v", "e1v", "e0"]
    assert report["three_seed_replay_before_extension"] == {"replay": True}
    assert report["e0_gene_reference"] == {"e0": True}
    assert set(arrays) == {"fixed_draws", "repeated_draws", "e3v_draws"}


def test_analysis_replay_compares_json_semantics_and_npz_arrays(tmp_path: Path):
    report_path = tmp_path / "five_seed_results.json"
    distributions_path = tmp_path / "bootstrap_distributions.npz"
    expected_report = {
        "schema_version": 1,
        "status": "complete",
        "nested": {"estimate": 0.625, "gate": True},
    }
    sealed_report = {**expected_report, "created_utc": "2026-08-23T12:00:00+00:00"}
    report_path.write_text(json.dumps(sealed_report))
    expected_arrays = {
        "fine": np.array([0.5, np.nan, 0.75], dtype=np.float64),
        "delta": np.array([0.1, 0.2], dtype=np.float32),
    }
    np.savez_compressed(distributions_path, **expected_arrays)

    extension._verify_analysis_replay(
        report_path,
        distributions_path,
        expected_report,
        expected_arrays,
    )

    report_path.write_text(json.dumps({**sealed_report, "status": "drifted"}))
    with pytest.raises(RuntimeError, match="semantic replay drifted"):
        extension._verify_analysis_replay(
            report_path,
            distributions_path,
            expected_report,
            expected_arrays,
        )

    report_path.write_text(json.dumps(sealed_report))
    np.savez_compressed(
        distributions_path,
        fine=expected_arrays["fine"],
        delta=np.array([0.1, 9.0], dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="array values drifted for delta"):
        extension._verify_analysis_replay(
            report_path,
            distributions_path,
            expected_report,
            expected_arrays,
        )


def test_verify_parser_exposes_explicit_aim1_e0_dependency():
    args = extension.build_parser().parse_args(["verify"])
    assert Path(args.e0_root) == extension.DEFAULT_E0_ROOT
