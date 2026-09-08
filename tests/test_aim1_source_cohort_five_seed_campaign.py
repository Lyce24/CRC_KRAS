from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import aim1_source_cohort_five_seed_campaign as campaign  # noqa: E402

EXPECTED_CENSUS = {
    "tcga_primary": (502, 508, 207, 295),
    "sr386_primary": (413, 413, 147, 266),
    "surgen_primary": (737, 881, 294, 443),
    "tcga_surgen_primary": (1239, 1389, 501, 738),
}

EXPECTED_SLIDES_BY_FOLD = {
    "tcga_primary": (102, 101, 102, 101, 102),
    "sr386_primary": (83, 82, 82, 83, 83),
    "surgen_primary": (176, 176, 175, 178, 176),
    "tcga_surgen_primary": (278, 277, 277, 279, 278),
}

EXPECTED_PATIENTS_BY_FOLD = {
    "tcga_primary": (101, 100, 101, 100, 100),
    "sr386_primary": (83, 82, 82, 83, 83),
    "surgen_primary": (147, 147, 147, 148, 148),
    "tcga_surgen_primary": (248, 247, 248, 248, 248),
}

EXPECTED_VALIDATION_SLIDES_BY_FOLD = {
    "tcga_primary": (61, 62, 61, 62, 61),
    "sr386_primary": (50, 50, 50, 50, 50),
    "surgen_primary": (106, 105, 106, 104, 105),
    "tcga_surgen_primary": (167, 167, 167, 166, 166),
}


def _normalized_census(value: Any) -> tuple[int, int, int, int]:
    """Normalize the public census without weakening its scientific fields."""
    if isinstance(value, Mapping):
        return (
            int(value["patients"]),
            int(value["slides"]),
            int(value["mutant"]),
            int(value["wild_type"]),
        )
    assert isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    assert len(value) == 4
    return tuple(int(item) for item in value)  # type: ignore[return-value]


def _command_text(job: Mapping[str, Any]) -> str:
    command = job.get("training_command", job.get("command"))
    assert isinstance(command, Sequence) and not isinstance(command, (str, bytes))
    return " ".join(str(part) for part in command)


def _json_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    value = json.loads(capsys.readouterr().out)
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def prepared_campaign(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not campaign.MASTER_MANIFEST.is_file():
        pytest.skip("frozen Aim-1 master manifest is not mounted")
    root = tmp_path_factory.mktemp("aim1-source-cohort") / "campaign"
    # Production preparation directly hashes the immutable ~50 GB UNI payload.
    # The focused unit fixture exercises contract semantics with a synthetic
    # feature-store record so routine tests do not repeatedly reread those bytes.
    original_packed_identity = campaign._packed_store_identity  # noqa: SLF001
    campaign._packed_store_identity = lambda _root: {  # type: ignore[method-assign]  # noqa: SLF001
        "path": str(campaign.paths.PACKED_FEATURE_DIR.resolve()),
        "encoder": "UNI-v1",
        "feature_dim": 1_024,
        "artifacts": {},
        "coverage": {
            arm: {"requested_slides": census[1], "missing_slides": 0}
            for arm, census in EXPECTED_CENSUS.items()
        },
    }
    try:
        campaign.cmd_prepare(argparse.Namespace(output_root=root, apply=True))
    finally:
        campaign._packed_store_identity = original_packed_identity  # type: ignore[method-assign]  # noqa: SLF001
    campaign.validate_contract(root, deep=True)
    return root


def test_frozen_public_study_contract_is_exact() -> None:
    assert tuple(campaign.SEEDS) == (42, 43, 44, 45, 46)
    assert tuple(campaign.FOLDS) == (0, 1, 2, 3, 4)
    assert set(campaign.ARM_SPECS) == set(EXPECTED_CENSUS)
    assert {
        arm: _normalized_census(census) for arm, census in campaign.EXPECTED_ARM_CENSUS.items()
    } == EXPECTED_CENSUS
    assert campaign.MAX_CONCURRENT_GPU_TRAINERS == 6


def test_arm_filters_are_exact_and_exclude_nonprimary_or_foreign_sources() -> None:
    frame = pd.DataFrame(
        {
            "slide_id": [f"S{index}" for index in range(8)],
            "specimen_role": [
                "primary",
                "primary",
                "primary",
                "primary",
                "metastatic",
                "primary",
                "primary",
                "primary",
            ],
            "cohort": ["TCGA", "TCGA", "SurGen", "SurGen", "TCGA", "RIH", "CPTAC", "SurGen"],
            "subcohort": [
                "TCGA-COAD",
                "TCGA-READ",
                "SR386",
                "SR1482",
                "TCGA-COAD",
                "RIH-Colon",
                "CPTAC-COAD",
                "foreign",
            ],
        }
    )
    observed = {
        arm: set(frame.loc[campaign._arm_mask(frame, arm), "slide_id"])  # noqa: SLF001
        for arm in campaign.ARM_SPECS
    }
    assert observed == {
        "tcga_primary": {"S0", "S1"},
        "sr386_primary": {"S2"},
        "surgen_primary": {"S2", "S3"},
        "tcga_surgen_primary": {"S0", "S1", "S2", "S3", "S7"},
    }


def test_inventory_is_exactly_four_arms_by_five_seeds_and_100_oof_fits(
    tmp_path: Path,
) -> None:
    jobs = campaign.build_job_inventory(tmp_path / "campaign")

    assert len(jobs) == 20
    assert {(job["arm"], int(job["seed"])) for job in jobs} == {
        (arm, seed) for arm in EXPECTED_CENSUS for seed in campaign.SEEDS
    }
    assert all(int(job["fit_count"]) == 5 for job in jobs)
    assert sum(int(job["fit_count"]) for job in jobs) == 100
    assert len({str(job["job_id"]) for job in jobs}) == 20
    assert len({str(job["output"]) for job in jobs}) == 20
    assert all(tuple(job.get("folds", campaign.FOLDS)) == campaign.FOLDS for job in jobs)


def test_every_training_recipe_is_oof_only_and_explicitly_skips_finalize(
    tmp_path: Path,
) -> None:
    jobs = campaign.build_job_inventory(tmp_path / "campaign")

    for job in jobs:
        command = _command_text(job)
        assert "training.skip_finalize=true" in command
        assert int(job.get("refit_count", 0)) == 0
        assert "refit_epoch_rule" not in command
        assert "final/refit" not in command


def test_contract_accounts_for_100_fits_zero_refits_and_six_trainers(
    tmp_path: Path,
) -> None:
    contract = campaign.build_contract(
        tmp_path / "campaign", created_utc="2026-08-24T12:00:00+00:00"
    )

    assert contract["execution"]["max_concurrent_gpu_trainers"] == 6
    assert contract["fit_accounting"] == {
        "jobs": 20,
        "fold_fits_per_job": 5,
        "oof_fits": 100,
        "refits": 0,
        "total_fits": 100,
    }
    assert contract["recipe"]["skip_finalize"] is True
    assert contract["model_seeds"] == list(campaign.SEEDS)


def test_cli_has_safe_training_default_and_all_public_lifecycle_commands() -> None:
    parser = campaign.build_parser()
    commands = {
        "plan",
        "prepare",
        "preflight",
        "train",
        "validate",
        "analyze",
        "verify",
        "status",
    }

    parsed = {command: parser.parse_args([command]) for command in commands}
    assert set(parsed) == commands
    assert parsed["train"].apply is False
    assert parser.parse_args(["train", "--apply"]).apply is True
    assert all(callable(args.func) for args in parsed.values())


def test_inventory_is_deterministic_and_output_scoped(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    first = campaign.build_job_inventory(root)
    second = campaign.build_job_inventory(root)

    assert first == second
    resolved_root = root.resolve()
    for job in first:
        assert Path(str(job["output"])).resolve().is_relative_to(resolved_root)


def test_duplicate_or_foreign_seed_roster_is_not_the_public_inventory(
    tmp_path: Path,
) -> None:
    jobs = campaign.build_job_inventory(tmp_path / "campaign")
    roster = [(str(job["arm"]), int(job["seed"])) for job in jobs]

    assert len(roster) == len(set(roster))
    assert not any(seed not in campaign.SEEDS for _arm, seed in roster)


def test_prepared_manifests_have_exact_censuses_nesting_and_frozen_folds(
    prepared_campaign: Path,
) -> None:
    master = pd.read_csv(campaign.MASTER_MANIFEST, low_memory=False)
    frames = {
        arm: pd.read_csv(campaign.manifest_path(prepared_campaign, arm), low_memory=False)
        for arm in campaign.ARM_SPECS
    }

    for arm, frame in frames.items():
        labels = frame.groupby("patient_id")["target_label"].first().astype(int)
        assert (
            frame["patient_id"].nunique(),
            len(frame),
            int(labels.eq(1).sum()),
            int(labels.eq(0).sum()),
        ) == EXPECTED_CENSUS[arm]
        assert (
            tuple(int(frame["k_fold"].eq(fold).sum()) for fold in campaign.FOLDS)
            == EXPECTED_SLIDES_BY_FOLD[arm]
        )
        assert (
            tuple(
                int(frame.loc[frame["k_fold"].eq(fold), "patient_id"].nunique())
                for fold in campaign.FOLDS
            )
            == EXPECTED_PATIENTS_BY_FOLD[arm]
        )
        assert (
            tuple(int(frame[f"val_fold_{fold}"].sum()) for fold in campaign.FOLDS)
            == EXPECTED_VALIDATION_SLIDES_BY_FOLD[arm]
        )

        expected = master.loc[campaign._arm_mask(master, arm)].reset_index(drop=True)  # noqa: SLF001
        identity_columns = [
            "slide_id",
            "patient_id",
            "target_label",
            "specimen_role",
            "cohort",
            "subcohort",
            "k_fold",
            *(f"val_fold_{fold}" for fold in campaign.FOLDS),
        ]
        pd.testing.assert_frame_equal(
            frame[identity_columns], expected[identity_columns], check_dtype=False
        )
        split = pd.read_parquet(
            campaign.split_dir(prepared_campaign, arm) / "splits.parquet"
        )[["slide_id", "fold", *(f"val_fold_{fold}" for fold in campaign.FOLDS)]]
        joined = split.merge(
            master[["slide_id", "k_fold", *(f"val_fold_{fold}" for fold in campaign.FOLDS)]],
            on="slide_id",
            validate="one_to_one",
            suffixes=("_derived", "_master"),
        )
        assert len(joined) == len(frame)
        assert joined["fold"].astype(int).equals(joined["k_fold"].astype(int))
        for fold in campaign.FOLDS:
            assert (
                joined[f"val_fold_{fold}_derived"]
                .astype(int)
                .equals(joined[f"val_fold_{fold}_master"].astype(int))
            )

    slide_ids = {arm: set(frame["slide_id"].astype(str)) for arm, frame in frames.items()}
    patient_ids = {arm: set(frame["patient_id"].astype(str)) for arm, frame in frames.items()}
    for rosters in (slide_ids, patient_ids):
        assert rosters["sr386_primary"] < rosters["surgen_primary"] < rosters["tcga_surgen_primary"]
        assert rosters["tcga_primary"] < rosters["tcga_surgen_primary"]
        assert rosters["tcga_primary"].isdisjoint(rosters["surgen_primary"])


def test_deep_contract_validation_rejects_derived_manifest_tamper(
    prepared_campaign: Path,
) -> None:
    path = campaign.manifest_path(prepared_campaign, "sr386_primary")
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(campaign.ContractError, match="drifted"):
            campaign.validate_contract(prepared_campaign, deep=True)
    finally:
        path.write_bytes(original)
    campaign.validate_contract(prepared_campaign, deep=True)


def test_manifest_validator_rejects_a_wrong_slide_roster(prepared_campaign: Path) -> None:
    frame = pd.read_csv(
        campaign.manifest_path(prepared_campaign, "tcga_primary"), low_memory=False
    ).iloc[:-1]
    with pytest.raises(campaign.ContractError, match="census mismatch"):
        campaign._validate_arm_frame(frame, "tcga_primary")  # noqa: SLF001


def test_artifacts_and_output_roots_reject_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(campaign.ContractError, match="symlinked"):
        campaign._artifact(link)  # noqa: SLF001

    directory = tmp_path / "real-root"
    directory.mkdir()
    root_link = tmp_path / "root-link"
    root_link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(campaign.ContractError, match="symlink"):
        campaign.assert_safe_output_root(root_link / "campaign")


def test_native_validation_rejects_any_final_or_refit_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oceanpath.workflows import training

    root = tmp_path / "campaign"
    directory = campaign.run_dir(root, "sr386_primary", 42)
    (directory / "final/refit").mkdir(parents=True)
    (directory / "final/refit/model.ckpt").write_bytes(b"forbidden")
    monkeypatch.setattr(
        training,
        "validate_training_run_dir",
        lambda *_args, **_kwargs: {
            "n_folds": 5,
            "skip_finalize": True,
            "fold_completions": [
                {"path": f"fold_{fold}/completion.json"} for fold in campaign.FOLDS
            ],
        },
    )
    with pytest.raises(campaign.ContractError, match="forbidden final/refit"):
        campaign._native_validation(root, "sr386_primary", 42)  # noqa: SLF001


def test_status_reports_pending_and_failed_without_certifying_fits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "campaign"
    campaign.cmd_status(argparse.Namespace(output_root=root))
    pending = _json_stdout(capsys)
    assert pending["jobs"] == {"pending": 20}
    assert pending["certified_logical_fits"] == 0
    assert pending["remaining_logical_fits"] == 100
    assert pending["training_sealed"] is False

    failed = campaign.failure_path(root, "tcga_primary", 42)
    failed.parent.mkdir(parents=True)
    failed.write_text('{"status":"failed"}\n', encoding="utf-8")
    campaign.cmd_status(argparse.Namespace(output_root=root))
    failure = _json_stdout(capsys)
    assert failure["jobs"] == {"failed": 1, "pending": 19}
    assert failure["certified_logical_fits"] == 0
    assert failure["remaining_logical_fits"] == 100


@pytest.mark.parametrize("command", ["plan", "status"])
def test_read_only_cli_commands_do_not_offer_apply(command: str) -> None:
    args = campaign.build_parser().parse_args([command])
    assert not hasattr(args, "apply")
