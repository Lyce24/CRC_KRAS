from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aim2_tcga_surgen_source_anchored_met_adaptation_campaign as campaign


def _blind_spec(*, key: str = "rih_metastatic") -> campaign.TargetSpec:
    return campaign.TargetSpec(
        key=key,
        cohort="RIH" if key == "rih_metastatic" else "SurGen",
        blind_source=Path("/does/not/matter.csv"),
        blind_sha256="0" * 64,
        outcome_source=Path("/does/not/matter-outcome.csv"),
        outcome_sha256="1" * 64,
        prior_score_key="unused",
        slides=2,
        patients=2,
        mutant=1,
        source_family_exposed=False,
    )


def _valid_blind(*, key: str = "rih_metastatic") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slide_id": ["s1", "s2"],
            "patient_id": ["p1", "p2"],
            "specimen_role": ["metastatic", "metastatic"],
            "subcohort": ["RIH-Colon", "RIH-Colon"]
            if key == "rih_metastatic"
            else ["SR1482", "SR1482"],
            "liver_class": ["liver", "non_liver"],
        }
    )


def test_exact_campaign_counts_and_two_method_matrix() -> None:
    accounting = campaign.fit_accounting()
    assert accounting["few_shot_final_residual_decisions"] == 22_500
    assert accounting["few_shot_final_pure_ridge_decisions"] == 22_500
    assert accounting["few_shot_solver_calls_per_method"] == 3_382_500
    assert accounting["few_shot_solver_calls_both_methods"] == 6_765_000
    assert accounting["few_shot_unique_support_procedures_by_fold"] == 4_500
    assert accounting["few_shot_final_head_decisions_both_methods"] == 45_000
    assert accounting["few_shot_fit_to_test_cohort_applications"] == 90_000
    assert accounting["full_label_final_residual_decisions"] == 250
    assert accounting["full_label_platt_fits"] == 50
    assert accounting["local_mil_oof_fits"] == 25
    assert len(campaign.score_jobs()) == 10
    assert len(campaign.adapt_jobs()) == 25
    assert len([job for job in campaign.adapt_jobs() if job.kind == "few"]) == 15
    assert len([job for job in campaign.adapt_jobs() if job.kind == "full"]) == 10
    assert len(campaign.local_jobs()) == 25
    assert campaign.SUPPORT_REGIMES == ("RIH_ONLY", "SURGEN_ONLY", "COMBINED")


def test_output_root_firewall(tmp_path: Path) -> None:
    assert campaign.validate_output_root(tmp_path / "fresh") == (tmp_path / "fresh").resolve()
    with pytest.raises(campaign.ContractError):
        campaign.validate_output_root(Path("relative"))
    with pytest.raises(campaign.ContractError):
        campaign.validate_output_root(campaign.SOURCE_ROOT)
    with pytest.raises(campaign.ContractError):
        campaign.validate_output_root(campaign.SOURCE_ROOT / "child")
    with pytest.raises(campaign.ContractError):
        campaign.validate_output_root(Path("/var/tmp/not-governed"))


def test_blind_roster_exact_allowlist_and_semantics() -> None:
    frame = _valid_blind()
    observed = campaign._validate_blind(frame, _blind_spec(), context="test")
    assert observed["slide_id"].tolist() == ["s1", "s2"]
    for mutation in (
        lambda value: value.assign(extra="forbidden"),
        lambda value: value.assign(specimen_role="primary"),
        lambda value: value.assign(subcohort="wrong"),
        lambda value: value.assign(liver_class="liver"),
    ):
        with pytest.raises(campaign.ContractError):
            campaign._validate_blind(mutation(frame.copy()), _blind_spec(), context="test")


def test_blind_null_rejected_before_string_coercion() -> None:
    frame = _valid_blind().astype(object)
    frame.loc[0, "patient_id"] = None
    with pytest.raises(campaign.ContractError, match="null blind identity"):
        campaign._validate_blind(frame, _blind_spec(), context="test")


def test_artifact_rejects_leaf_and_ancestor_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("bytes", encoding="utf-8")
    leaf = tmp_path / "leaf.txt"
    leaf.symlink_to(target)
    with pytest.raises(campaign.ContractError, match="symlink"):
        campaign._artifact(leaf)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    nested = real_dir / "nested.txt"
    nested.write_text("nested", encoding="utf-8")
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(campaign.ContractError, match="symlink"):
        campaign._artifact(linked_dir / "nested.txt")


@pytest.mark.parametrize("payload", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'])
def test_json_reader_rejects_duplicates_and_nonfinite(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(campaign.ContractError):
        campaign._read_json(path)


def test_jsonl_sorted_roundtrip_preserves_lambda_replay(tmp_path: Path) -> None:
    keys = [campaign.adapter.lambda_key(value) for value in campaign.LAMBDA_GRID]
    losses = {key: [float(index + 1)] * 4 for index, key in enumerate(keys)}
    means = {key: float(index + 1) for index, key in enumerate(keys)}
    record = {
        "support_total": 4,
        "layout_seed": campaign.OUTER_LAYOUT_SEEDS[0],
        "outer_fold": 0,
        "selected_lambda": keys[0],
        "inner_selection": {
            "scheme": "leave_one_out",
            "inner_seed": campaign.OUTER_LAYOUT_SEEDS[0],
            "n_pool": 4,
            "n_splits": 4,
            "losses_by_lambda": losses,
            "mean_loss_by_lambda": means,
            "solver_summary_by_lambda": {key: {} for key in keys},
            "selected_lambda": keys[0],
        },
        "coefficients": [0.0] * campaign.EMBED_DIM,
        "bias": 0.0,
        "solver_diagnostic": {"status": "native_exact"},
    }
    path = tmp_path / "record.jsonl"
    campaign._write_jsonl_once(path, [record])
    replayed = campaign._read_jsonl(path)[0]
    campaign._validate_selected_lambda(replayed)


def _synthetic_adapter_data() -> dict[str, dict[int, pd.DataFrame]]:
    result: dict[str, dict[int, pd.DataFrame]] = {}
    embedding_columns = [f"e{index}" for index in range(campaign.EMBED_DIM)]
    for cohort in ("RIH", "SurGen"):
        labels = np.asarray([0] * 10 + [1] * 10, dtype=int)
        base = pd.DataFrame(
            np.zeros((20, campaign.EMBED_DIM)),
            columns=embedding_columns,
            index=[f"{cohort}-p{index:02d}" for index in range(20)],
        )
        base["label"] = labels
        base["eta_native"] = np.linspace(-1, 1, 20)
        result[cohort] = {seed: base.copy() for seed in campaign.MODEL_SEEDS}
    return result


@pytest.mark.parametrize("regime", campaign.SUPPORT_REGIMES)
def test_support_generation_is_exact_balanced_and_deterministic(regime: str) -> None:
    data = _synthetic_adapter_data()
    kwargs = {
        "layout_seed": campaign.OUTER_LAYOUT_SEEDS[0],
        "fold": 0,
        "support_regime": regime,
        "support_per_class": 2,
        "draw": 3,
    }
    first = campaign._support_for_procedure(data, **kwargs)
    assert first == campaign._support_for_procedure(data, **kwargs)
    assert len(first) == 4
    assert [label for _cohort, _patient, label in first].count(0) == 2
    assert [label for _cohort, _patient, label in first].count(1) == 2
    if regime == "COMBINED":
        for cohort in ("RIH", "SurGen"):
            block = [label for source, _patient, label in first if source == cohort]
            assert sorted(block) == [0, 1]
    elif regime == "RIH_ONLY":
        assert {cohort for cohort, _patient, _label in first} == {"RIH"}
    else:
        assert {cohort for cohort, _patient, _label in first} == {"SurGen"}


def test_commands_are_fold_and_shard_scoped(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    local_commands = [campaign._local_command(root, job, num_workers=4) for job in campaign.local_jobs()]
    assert len({tuple(command) for command in local_commands}) == 25
    assert all("_train-local-one" in command for command in local_commands)
    assert all("--fold" in command for command in local_commands)
    adapt_commands = [campaign._adapt_command(root, job) for job in campaign.adapt_jobs()]
    assert len({tuple(command) for command in adapt_commands}) == 25
    assert all("_adapt-one" in command for command in adapt_commands)


def test_plan_is_nonmutating_and_explicit(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    plan = campaign.campaign_plan(root)
    assert not root.exists()
    assert plan["production_launch_authorized_by_this_command"] is False
    assert plan["adaptation"]["methods"] == [
        "source_anchored_residual_linear_probe",
        "pure_ridge_linear_probe",
    ]
    assert plan["adaptation"]["test_cohorts_per_head"] == ["RIH-M", "SurGen-M"]


def test_strict_jsonl_rejects_duplicate_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"ok": 1}) + "\n" + '{"x": 1, "x": 2}\n')
    with pytest.raises(campaign.ContractError):
        campaign._read_jsonl(path)


def test_missing_adaptation_completion_is_never_minted(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    job = campaign.AdaptJob(
        "few", campaign.OUTER_LAYOUT_SEEDS[0], support_regime="RIH_ONLY"
    )
    directory = campaign.adapt_shard_dir(root, job)
    directory.mkdir(parents=True)
    for name in ("oof.parquet", "fits.jsonl", "supports.jsonl", "solver.json", "execution.json"):
        (directory / name).write_bytes(b"partial")
    with pytest.raises(campaign.ContractError, match="partial adaptation shard"):
        campaign._validate_adapt_shard(root, job)
    assert not campaign.adapt_shard_receipt_path(root, job).exists()


def test_full_label_lower_lambda_boundary_fails_closed() -> None:
    record = {
        "phase": "full_label",
        "model_kind": "residual_adapter",
        "selected_lambda": campaign.adapter.lambda_key(campaign.LAMBDA_GRID[-1]),
    }
    with pytest.raises(RuntimeError, match="lambda search remains truncated"):
        campaign.adapter.assert_full_label_lambda_grid_closed([record])


def _synthetic_full_data() -> dict[str, dict[int, pd.DataFrame]]:
    embedding_columns = [f"e{index}" for index in range(campaign.EMBED_DIM)]
    labels = np.asarray([0] * 15 + [1] * 15, dtype=int)
    base = pd.DataFrame(
        np.zeros((30, campaign.EMBED_DIM)),
        columns=embedding_columns,
        index=[f"RIH-full-p{index:02d}" for index in range(30)],
    )
    base["label"] = labels
    base["eta_native"] = np.where(labels == 1, 1.0, -1.0) + np.linspace(-0.1, 0.1, 30)
    return {"RIH": {seed: base.copy() for seed in campaign.MODEL_SEEDS}}


def test_full_shard_runs_exact_2030_solver_calls_and_closes_grid() -> None:
    job = campaign.AdaptJob("full", campaign.OUTER_LAYOUT_SEEDS[0], cohort="RIH")
    frame, fits, supports, solver = campaign._run_full_shard(_synthetic_full_data(), job)
    assert len(frame) == 30
    assert len(fits) == 30
    assert supports == []
    assert solver["n_finite_calls"] + solver["n_native_exact_calls"] == 2_030
    campaign.adapter.assert_full_label_lambda_grid_closed(fits)
    residual = [row for row in fits if row["model_kind"] == "residual_adapter"]
    platt = [row for row in fits if row["model_kind"] == "platt"]
    assert len(residual) == 25
    assert len(platt) == 5
    mutated = dict(residual[0])
    mutated["selected_lambda"] = campaign.adapter.lambda_key(campaign.LAMBDA_GRID[-1])
    with pytest.raises(RuntimeError):
        campaign.adapter.assert_full_label_lambda_grid_closed([*residual[1:], mutated])


def test_all_local_configs_canonically_match_pinned_source_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack_path = (
        "/mnt/wsl/oceanpath-hot/features/colon_stream/"
        "20x_256px_0px_overlap_mpp0.5/packed_uni_v1"
    )
    monkeypatch.setattr(
        campaign,
        "load_contract",
        lambda _root, deep_pack: {"source": {"feature_store": {"path": pack_path}}},
    )
    for seed in campaign.MODEL_SEEDS:
        cfg = campaign._local_config(
            tmp_path, campaign.LocalJob(seed, 0), num_workers=campaign.DEFAULT_NUM_WORKERS
        )
        assert cfg.training.seed == seed
        assert cfg.training.num_workers == campaign.DEFAULT_NUM_WORKERS
        assert cfg.training.skip_finalize is True
        assert cfg.training.dataset_max_instances == campaign.CAP


def test_local_config_rejects_pinned_source_recipe_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = campaign.source_oof_config(42)
    mutated = tmp_path / "config.yaml"
    payload = source.read_bytes().replace(b"label_smoothing: 0.0", b"label_smoothing: 0.1")
    assert len(payload) == source.stat().st_size
    mutated.write_bytes(payload)
    monkeypatch.setattr(campaign, "source_oof_config", lambda _seed: mutated)
    monkeypatch.setattr(
        campaign,
        "load_contract",
        lambda _root, deep_pack: {
            "source": {
                "feature_store": {
                    "path": (
                        "/mnt/wsl/oceanpath-hot/features/colon_stream/"
                        "20x_256px_0px_overlap_mpp0.5/packed_uni_v1"
                    )
                }
            }
        },
    )
    with pytest.raises(campaign.ContractError, match="pinned artifact drifted"):
        campaign._local_config(tmp_path, campaign.LocalJob(42, 0), num_workers=4)


def _write_local_partition_fixture(root: Path) -> None:
    rows = []
    for fold in campaign.FOLDS:
        for offset in range(2):
            patient = f"p{fold}-{offset}"
            rows.append(
                {
                    "slide_id": f"s{fold}-{offset}",
                    "patient_id": patient,
                    "target_label": offset,
                    "k_fold": fold,
                    **{
                        f"val_fold_{index}": int(index == 0 and fold == 1 and offset == 0)
                        for index in campaign.FOLDS
                    },
                }
            )
    manifest = pd.DataFrame(rows)
    manifest_path = campaign.labeled_manifest_path(root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    splits = manifest.rename(
        columns={"patient_id": "group_id", "k_fold": "fold"}
    ).copy()
    split_path = campaign.split_dir(root) / "splits.parquet"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(split_path, index=False)


def test_local_partition_replays_exact_train_val_test_and_rejects_drift(
    tmp_path: Path,
) -> None:
    _write_local_partition_fixture(tmp_path)
    partition = campaign._expected_local_partition(tmp_path, campaign.LocalJob(42, 0))
    assert partition["test"] == {"s0-0", "s0-1"}
    assert partition["val"] == {"s1-0"}
    assert len(partition["train"]) == 7
    split_path = campaign.split_dir(tmp_path) / "splits.parquet"
    splits = pd.read_parquet(split_path)
    splits.loc[splits["slide_id"].eq("s1-0"), "val_fold_0"] = 0
    splits.to_parquet(split_path, index=False)
    with pytest.raises(campaign.ContractError, match="exact train/val/test membership"):
        campaign._expected_local_partition(tmp_path, campaign.LocalJob(42, 0))


def test_local_fingerprint_and_pack_gate_mismatches_fail_closed() -> None:
    campaign._assert_local_training_fingerprint("same", "same", context="test")
    with pytest.raises(campaign.ContractError, match="fingerprint"):
        campaign._assert_local_training_fingerprint("wrong", "right", context="test")
    pre = {"source_feature_store": {"sha": "a"}, "pack_stat_snapshot": {"x": 1}}
    campaign._assert_local_pack_gates_match(pre, dict(pre))
    with pytest.raises(campaign.ContractError, match="pre/post"):
        campaign._assert_local_pack_gates_match(
            pre, {"source_feature_store": {"sha": "a"}, "pack_stat_snapshot": {"x": 2}}
        )


def test_train_local_child_invokes_exactly_one_fold_without_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = campaign.LocalJob(42, 3)
    calls: list[tuple[int, Path]] = []
    monkeypatch.setattr(campaign, "validate_output_root", lambda root, must_exist=True: root)
    monkeypatch.setattr(campaign, "_validate_target_open", lambda _root: {})
    monkeypatch.setattr(campaign, "_validate_local_job", lambda _root, _job: None)
    monkeypatch.setattr(campaign, "_local_config", lambda _root, _job, num_workers: object())
    monkeypatch.setattr(campaign, "_local_job_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(campaign, "_write_json_once", lambda *args, **kwargs: None)

    @contextmanager
    def fake_context(_fold: int):
        yield

    import oceanpath.workflows.training as workflow

    monkeypatch.setattr(workflow, "fold_context", fake_context)
    monkeypatch.setattr(
        workflow,
        "run_fold",
        lambda cfg, fold_idx, output_dir: calls.append((fold_idx, output_dir)),
    )
    campaign._train_local_one(tmp_path, job, num_workers=4)
    assert calls == [(3, campaign.local_run_dir(tmp_path, job))]
    with pytest.raises(campaign.ContractError, match="num-workers 4"):
        campaign._train_local_one(tmp_path, job, num_workers=3)


def test_no_legacy_or_residual_only_15k_path_remains() -> None:
    source = Path(campaign.__file__).read_text(encoding="utf-8")
    assert "legacy_run_adaptation" not in source
    assert "run_support_curve" not in source
    assert "15_000" not in source
    assert "15000" not in source


def test_dry_analysis_and_status_do_not_mint_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    before = set(root.rglob("*"))
    result = campaign.analyze_campaign(root, apply=False)
    assert result["status"] == "DRY_RUN_ANALYSIS"
    assert set(root.rglob("*")) == before
    status = campaign.campaign_status(root)
    assert not any(status["stages"].values())
    assert set(root.rglob("*")) == before
