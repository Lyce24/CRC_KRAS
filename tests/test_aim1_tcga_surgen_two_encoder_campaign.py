"""Contract tests for the additive Aim-1 TCGA+SurGen two-encoder campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import aim1_tcga_surgen_two_encoder_campaign as campaign  # noqa: E402


def test_inventory_is_exactly_ten_jobs_and_35_new_fits(tmp_path: Path) -> None:
    jobs = campaign.build_job_inventory(tmp_path)

    assert len(jobs) == 10
    assert {(job["kind"], job["seed"]) for job in jobs} == {
        (kind, seed)
        for kind in ("univ1_refit", "virchow2_full")
        for seed in campaign.SEEDS
    }
    assert sum(job["oof_fits"] for job in jobs) == 25
    assert sum(job["refits"] for job in jobs) == 10
    assert sum(job["total_fits"] for job in jobs) == 35
    assert [job["kind"] for job in jobs[:5]] == ["virchow2_full"] * 5


def test_contract_has_exact_accounting_and_recipe(tmp_path: Path) -> None:
    contract = campaign.build_contract(tmp_path, created_utc="fixed")

    assert contract["fit_accounting"] == {
        "adopted_oof_fits": 25,
        "new_oof_fits": 25,
        "new_refits": 10,
        "new_fits": 35,
        "operational_lineage_fits": 60,
        "hidden_fits": 0,
    }
    assert contract["concurrency"] == {"maximum": 6}
    assert contract["seeds"] == [42, 43, 44, 45, 46]
    assert contract["folds"] == [0, 1, 2, 3, 4]
    assert contract["encoders"] == ["UNI-v1", "Virchow2-CLS"]
    assert contract["material_recipe"]["dataset_max_instances"] == 8192
    assert contract["material_recipe"]["train_sampling_strategy"] == "patient_natural"
    assert contract["material_recipe"]["sample_weight_column"] is None
    assert contract["material_recipe"]["refit_epoch_rule"] == "p75"


def test_stable_output_paths(tmp_path: Path) -> None:
    assert campaign.training_receipt_path(tmp_path) == tmp_path / "receipts/training_complete.json"
    assert campaign.run_dir(tmp_path, "univ1_refit", 42) == tmp_path / "train/e0/univ1/seed42"
    assert campaign.run_dir(tmp_path, "virchow2_full", 46) == tmp_path / "train/e0/virchow2_cls/seed46"
    assert campaign.job_receipt_path(tmp_path, "univ1_refit", 42) == tmp_path / "receipts/jobs/univ1_refit/seed42.json"
    assert campaign.job_receipt_path(tmp_path, "virchow2_full", 46) == tmp_path / "receipts/jobs/virchow2_full/seed46.json"


def test_worker_commands_are_internal_and_seed_specific(tmp_path: Path) -> None:
    jobs = campaign.build_training_jobs(tmp_path)

    assert len({tuple(job["command"]) for job in jobs}) == 10
    for job in jobs:
        command = job["command"]
        assert command[1].endswith("tools/aim1_tcga_surgen_two_encoder_campaign.py")
        assert command[2] == "_train-one"
        assert command[-4:] == ["--kind", job["kind"], "--seed", str(job["seed"])]
        if job["kind"] == "univ1_refit":
            assert job["oof_fits"] == 0
            assert job["refit_entrypoint"].endswith("._run_refit")
            assert "training_command" not in job
        else:
            assert job["oof_fits"] == 5
            assert job["training_command"][2] == "hydra-train"


def test_v2_hydra_overrides_compose_to_locked_recipe(tmp_path: Path) -> None:
    directory = campaign.run_dir(tmp_path, "virchow2_full", 42)
    with initialize_config_dir(
        config_dir=str(campaign.REPO / "configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="train",
            overrides=campaign._v2_overrides(tmp_path, 42, directory),
        )
    material = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(material, dict)
    campaign._validate_material_config(
        material,
        tmp_path,
        42,
        encoder="virchow2_cls",
        skip_finalize=False,
        context="test composition",
    )
    assert cfg.training.final_strategies == ["best_fold", "refit"]
    assert cfg.training.packed_dir == str(campaign.V2_PACK)
    assert cfg.extraction.patch_size == 224


def test_adopted_univ1_config_is_the_locked_refit_source() -> None:
    cfg = OmegaConf.load(campaign.adopted_run_dir(42) / "fold_0/config.yaml")
    material = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(material, dict)
    campaign._validate_material_config(
        material,
        campaign.ADOPTED_ROOT,
        42,
        encoder="univ1",
        skip_finalize=True,
        context="test adopted source",
    )
    assert cfg.training.final_strategies == ["best_fold", "refit"]
    assert cfg.training.refit_epoch_rule == "p75"


@pytest.mark.parametrize(
    "text",
    [
        '{"status":"ok","status":"duplicate"}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '[1, 2, 3]',
    ],
)
def test_strict_json_rejects_duplicate_nonfinite_and_nonobject(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(campaign.ContractError):
        campaign._read_json(path)


def test_json_writer_rejects_nonfinite(tmp_path: Path) -> None:
    with pytest.raises(campaign.ContractError):
        campaign._write_json_once(tmp_path / "bad.json", {"value": float("nan")})
    assert not (tmp_path / "bad.json").exists()


def test_write_once_is_idempotent_only_for_identical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    campaign._write_json_once(path, {"status": "ok"})
    campaign._write_json_once(path, {"status": "ok"})

    with pytest.raises(campaign.ContractError):
        campaign._write_json_once(path, {"status": "changed"})


def test_output_root_rejects_relative_and_symlink(tmp_path: Path) -> None:
    with pytest.raises(campaign.ContractError):
        campaign.assert_safe_output_root(Path("relative"))
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(campaign.ContractError):
        campaign.assert_safe_output_root(alias)


def test_train_dry_run_does_not_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(campaign, "validate_contract", lambda root, deep: {})
    args = argparse.Namespace(output_root=tmp_path, apply=False, max_workers=6)

    campaign.cmd_train(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DRY_RUN_NO_WRITES"
    assert len(payload["jobs"]) == 10
    assert list(tmp_path.iterdir()) == []


def test_apply_requires_exactly_six_workers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(campaign, "validate_contract", lambda root, deep: {})
    monkeypatch.setattr(campaign, "_validate_preflight", lambda root: {})
    args = argparse.Namespace(output_root=tmp_path, apply=True, max_workers=5)

    with pytest.raises(campaign.ContractError, match="exactly --max-workers 6"):
        campaign.cmd_train(args)


def test_peak_parallel_counts_overlaps() -> None:
    events = [
        {"started_monotonic": 0.0, "finished_monotonic": 4.0},
        {"started_monotonic": 1.0, "finished_monotonic": 3.0},
        {"started_monotonic": 2.0, "finished_monotonic": 5.0},
    ]
    assert campaign._peak_parallel(events) == 3


def test_terminal_schema_uses_verifier_field_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        campaign,
        "_adopted_chain",
        lambda seed, deep: {"seed": seed, "oof_fits": 5, "refits": 0},
    )
    monkeypatch.setattr(
        campaign,
        "_artifact",
        lambda path: {"path": str(path), "sha256": "0" * 64, "size_bytes": 1},
    )
    receipts = [
        {"path": f"receipt-{index}", "sha256": f"{index:064x}", "size_bytes": 1}
        for index in range(10)
    ]

    terminal = campaign._terminal_payload(tmp_path, receipts)

    assert terminal["fit_accounting"] == campaign._fit_accounting()
    assert terminal["concurrency"] == {"maximum": 6, "observed_peak": 6}
    assert len(terminal["adopted_univ1_oof_chains"]) == 5
    assert len(terminal["new_chains"]) == 10
    assert len(terminal["new_job_receipts"]) == 10


def test_parser_exposes_all_governed_stages() -> None:
    parser = campaign.build_parser()
    choices = next(
        action.choices
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert {"plan", "prepare", "preflight", "train", "validate", "verify", "status", "_train-one"}.issubset(choices)
