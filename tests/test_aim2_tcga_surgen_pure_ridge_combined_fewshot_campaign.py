from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aim2_tcga_surgen_pure_ridge_combined_fewshot_campaign as campaign


def _synthetic_data(*, patients_per_class: int = 20) -> dict[str, dict[int, pd.DataFrame]]:
    columns = [f"e{index}" for index in range(campaign.EMBED_DIM)]
    result: dict[str, dict[int, pd.DataFrame]] = {}
    for cohort_number, cohort in enumerate(("RIH", "SurGen")):
        labels = np.asarray(
            [0] * patients_per_class + [1] * patients_per_class, dtype=int
        )
        values = np.zeros((len(labels), campaign.EMBED_DIM), dtype=float)
        values[:, 0] = labels * 2 - 1
        values[:, 1] = np.linspace(-1, 1, len(labels)) + cohort_number
        base = pd.DataFrame(
            values,
            columns=columns,
            index=[f"{cohort}-p{index:03d}" for index in range(len(labels))],
        )
        base["label"] = labels
        base["eta_native"] = np.linspace(-0.8, 0.8, len(labels))
        result[cohort] = {
            seed: base.assign(e2=float(seed - 42)).copy()
            for seed in campaign.MODEL_SEEDS
        }
    return result


def _selection_record(total: int) -> dict[str, object]:
    keys = [campaign.broad.adapter.lambda_key(value) for value in campaign.LAMBDA_GRID]
    splits = total if total <= campaign.broad.adapter.LOO_POOL_MAX else len(campaign.FOLDS)
    losses = {
        key: [float(index + 1)] * splits for index, key in enumerate(keys)
    }
    means = {key: float(index + 1) for index, key in enumerate(keys)}
    summaries = {
        key: {
            "n_fits": splits,
            "n_native_exact": splits if key == "infinity" else 0,
            "n_scipy_success": 0 if key == "infinity" else splits,
            "n_accepted_small_gradient": 0,
            "max_gradient_inf_norm": None if key == "infinity" else 0.0,
            "min_objective_decrease": None if key == "infinity" else 0.0,
        }
        for key in keys
    }
    return {
        "support_total": total,
        "layout_seed": campaign.LAYOUT_SEEDS[0],
        "outer_fold": 0,
        "selected_lambda": keys[0],
        "inner_selection": {
            "scheme": "leave_one_out"
            if total <= campaign.broad.adapter.LOO_POOL_MAX
            else "five_fold",
            "inner_seed": campaign.LAYOUT_SEEDS[0],
            "n_pool": total,
            "n_splits": splits,
            "losses_by_lambda": losses,
            "mean_loss_by_lambda": means,
            "solver_summary_by_lambda": summaries,
            "selected_lambda": keys[0],
        },
    }


def test_exact_lean_matrix_and_accounting() -> None:
    observed = campaign.accounting()
    assert len(campaign.probe_jobs()) == 20
    assert len({job.key for job in campaign.probe_jobs()}) == 20
    assert observed == {
        "source_main_model_fits": 0,
        "label_blind_embedding_jobs": 10,
        "unique_support_procedures_by_fold": 2_000,
        "final_probe_head_decisions": 10_000,
        "fit_to_test_cohort_applications": 20_000,
        "inner_plus_final_solver_calls": 1_330_000,
        "residual_adapter_fits": 0,
        "local_mil_fits": 0,
        "full_label_fits": 0,
        "platt_fits": 0,
    }
    assert campaign.METHOD == "pure_ridge_linear_probe"
    assert campaign.SUPPORT_PER_CLASS == (2, 4, 8, 16)


@pytest.mark.parametrize(
    ("support", "calls_per_head", "calls_per_shard"),
    ((2, 65, 32_500), (4, 129, 64_500), (8, 257, 128_500), (16, 81, 40_500)),
)
def test_exact_nested_solver_scheme_and_census(
    support: int, calls_per_head: int, calls_per_shard: int
) -> None:
    job = campaign.ProbeJob(campaign.LAYOUT_SEEDS[0], support)
    assert campaign._solver_calls_per_head(support) == calls_per_head
    assert campaign._expected_shard_solver_calls(job) == calls_per_shard


def test_output_root_firewall(tmp_path: Path) -> None:
    assert campaign.validate_output_root(tmp_path / "new") == (tmp_path / "new").resolve()
    for invalid in (
        Path("relative"),
        campaign.SOURCE_ROOT,
        campaign.SOURCE_ROOT / "child",
        Path("/var/tmp/not-governed"),
    ):
        with pytest.raises(campaign.ContractError):
            campaign.validate_output_root(invalid)


def test_broad_context_is_scoped_and_restored() -> None:
    original = {
        name: getattr(campaign.broad, name)
        for name in (
            "DEFAULT_OUTPUT_ROOT",
            "CAMPAIGN",
            "validate_output_root",
            "_implementation_sources",
            "_contract_payload",
            "_source_score_command",
        )
    }
    with campaign._broad_context():
        assert campaign.broad.CAMPAIGN == campaign.CAMPAIGN
        assert campaign.broad.DEFAULT_OUTPUT_ROOT == campaign.DEFAULT_OUTPUT_ROOT
        assert campaign.broad._contract_payload is campaign._contract_payload
    assert all(getattr(campaign.broad, name) is value for name, value in original.items())


@pytest.mark.parametrize("support", campaign.SUPPORT_PER_CLASS)
def test_combined_support_is_exact_balanced_deterministic_and_leak_free(
    support: int,
) -> None:
    data = _synthetic_data()
    kwargs = {
        "layout_seed": campaign.LAYOUT_SEEDS[0],
        "support_per_class": support,
        "draw": 7,
        "fold": 3,
    }
    first = campaign._combined_support(data, **kwargs)
    assert first == campaign._combined_support(data, **kwargs)
    assert len(first) == 2 * support
    folds = campaign._folds_by_cohort(data, campaign.LAYOUT_SEEDS[0])
    for cohort in ("RIH", "SurGen"):
        block = [(patient, label) for source, patient, label in first if source == cohort]
        assert sum(label == 0 for _patient, label in block) == support // 2
        assert sum(label == 1 for _patient, label in block) == support // 2
        held_out = {
            str(data[cohort][campaign.MODEL_SEEDS[0]].index[index])
            for index in np.flatnonzero(folds[cohort] == kwargs["fold"])
        }
        assert not held_out & {patient for patient, _label in block}


@pytest.mark.parametrize("total", (4, 32))
def test_selection_replays_after_sorted_json_roundtrip(
    tmp_path: Path, total: int
) -> None:
    record = _selection_record(total)
    path = tmp_path / "fit.jsonl"
    campaign._write_jsonl_once(path, [record])
    replay = campaign._read_jsonl(path)[0]
    campaign._validate_selection(replay)
    replay["inner_selection"]["scheme"] = "five_fold" if total == 4 else "leave_one_out"
    with pytest.raises(campaign.ContractError):
        campaign._validate_selection(replay)


def test_finite_and_infinite_final_head_replay_detects_mutation() -> None:
    features = np.asarray(
        [[-1.0, 0.0], [-0.5, 1.0], [0.5, -1.0], [1.0, 0.0]], dtype=float
    )
    features = np.pad(features, ((0, 0), (0, campaign.EMBED_DIM - 2)))
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    ledger = campaign.broad.adapter.SolverLedger()
    weights, bias, diagnostic = campaign.broad.adapter.fit_residual(
        features,
        np.zeros(len(labels)),
        labels,
        1.0,
        ledger=ledger,
        context={"test": True},
    )
    finite = {
        "selected_lambda": "1",
        "coefficients": weights.tolist(),
        "bias": bias,
        "solver_diagnostic": diagnostic,
    }
    campaign._validate_final_head(finite, features=features, labels=labels)
    mutated = dict(finite)
    mutated["coefficients"] = finite["coefficients"].copy()
    mutated["coefficients"][0] += 0.1
    with pytest.raises(campaign.ContractError):
        campaign._validate_final_head(mutated, features=features, labels=labels)
    zero_weights, zero_bias, zero_diagnostic = campaign.broad.adapter.fit_residual(
        features,
        np.zeros(len(labels)),
        labels,
        math.inf,
        ledger=ledger,
        context={"test": True},
    )
    infinite = {
        "selected_lambda": "infinity",
        "coefficients": zero_weights.tolist(),
        "bias": zero_bias,
        "solver_diagnostic": zero_diagnostic,
    }
    campaign._validate_final_head(infinite, features=features, labels=labels)
    infinite["bias"] = 0.01
    with pytest.raises(campaign.ContractError):
        campaign._validate_final_head(infinite, features=features, labels=labels)


def test_missing_shard_completion_fails_without_minting(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    job = campaign.ProbeJob(campaign.LAYOUT_SEEDS[0], 2)
    directory = campaign.shard_dir(root, job)
    directory.mkdir(parents=True)
    for name in ("oof.parquet", "fits.jsonl", "supports.jsonl", "solver.json", "execution.json"):
        (directory / name).write_bytes(b"partial")
    with pytest.raises(campaign.ContractError, match="partial or noncanonical"):
        campaign._validate_probe_shard(root, job)
    assert not (directory / "completion.json").exists()


def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path: Path) -> None:
    for index, payload in enumerate(("{\"a\":1,\"a\":2}", "{\"a\":NaN}")):
        path = tmp_path / f"bad-{index}.json"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(campaign.ContractError):
            campaign._read_json(path)
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text(json.dumps({"ok": 1}) + "\n" + '{"x":1,"x":2}\n')
    with pytest.raises(campaign.ContractError):
        campaign._read_jsonl(jsonl)


def test_plan_and_status_are_nonmutating_and_scope_exact(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    plan = campaign.campaign_plan(root)
    status = campaign.campaign_status(root)
    assert not root.exists()
    assert status["production_root_absent"] is True
    assert plan["method"] == campaign.METHOD
    assert plan["support_regime"] == "COMBINED"
    assert plan["support_per_class_total"] == [2, 4, 8, 16]
    assert plan["shards"] == 20
    assert plan["max_workers"] == 6
    assert plan["production_launch_authorized_by_this_command"] is False
    assert "train-local" not in " ".join(plan["production_sequence"])


def test_run_probes_rejects_non_governed_worker_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    monkeypatch.setattr(campaign, "_validate_target_open", lambda _root: {})
    with pytest.raises(campaign.ContractError, match="exactly --max-workers 6"):
        campaign.run_probes(root, apply=False, max_workers=5)


def test_contract_declares_only_lean_method_and_combined_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    for target, spec in campaign.TARGETS.items():
        path = campaign.broad.blind_path(root, target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(spec.blind_source.read_bytes())
    monkeypatch.setattr(
        campaign,
        "_implementation_sources",
        lambda: {"test": {"path": "/tmp/test", "sha256": "0" * 64, "size": 1}},
    )
    payload = campaign._contract_payload(root, {"frozen": True}, created_utc="2026-08-28T00:00:00+00:00")
    assert payload["protocol"]["method"] == campaign.METHOD
    assert payload["protocol"]["support_per_class_total"] == [2, 4, 8, 16]
    assert payload["firewall"]["combined_support_only"] is True
    assert payload["firewall"]["residual_adapter_permitted"] is False
    assert payload["firewall"]["local_mil_permitted"] is False
    assert payload["firewall"]["full_label_or_platt_permitted"] is False
    assert payload["accounting"] == campaign.accounting()


def _authority_budget_frame() -> pd.DataFrame:
    base = Path(
        "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
        "aim1_tcga_surgen_two_encoder_v1_20260824/downstream_v2/"
        "continuation_v3/scores/univ1"
    )
    configs = (
        (
            "RIH",
            "rih_metastatic",
            Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_rih_metastatic.csv"),
        ),
        (
            "SurGen",
            "sr1482_metastatic",
            Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_surgen_metastatic.csv"),
        ),
    )
    patients: list[pd.DataFrame] = []
    for cohort, target, manifest_path in configs:
        manifest = pd.read_csv(manifest_path)[["slide_id", "patient_id", "target_label"]]
        per_seed: list[pd.Series] = []
        label: pd.Series | None = None
        for seed in campaign.MODEL_SEEDS:
            score = pd.read_parquet(base / target / f"seed{seed}.parquet")
            patient = (
                score.merge(manifest, on="slide_id", validate="one_to_one")
                .groupby("patient_id", sort=True)
                .agg(label=("target_label", "first"), logit=("logit", "mean"))
            )
            label = patient["label"]
            per_seed.append(patient["logit"].rename(f"eta_native_seed{seed}"))
        assert label is not None
        block = pd.concat(per_seed, axis=1)
        block["label"] = label
        block["eta_native"] = block[
            [f"eta_native_seed{seed}" for seed in campaign.MODEL_SEEDS]
        ].mean(axis=1)
        block["test_cohort"] = cohort
        block["patient_id"] = block.index.astype(str)
        patients.append(block.reset_index(drop=True))
    authority = pd.concat(patients, ignore_index=True)
    rows: list[pd.DataFrame] = []
    for layout in campaign.LAYOUT_SEEDS:
        for draw in range(campaign.DRAWS_PER_LAYOUT):
            block = authority.copy()
            block["layout_seed"] = layout
            block["draw"] = draw
            block["support_per_class"] = 2
            block["support_total"] = 4
            block["eta_pure_ridge"] = block["eta_native"]
            for seed in campaign.MODEL_SEEDS:
                block[f"eta_pure_ridge_seed{seed}"] = block[f"eta_native_seed{seed}"]
            rows.append(block)
    return pd.concat(rows, ignore_index=True)


def test_budget_summary_reports_exact_native_pooled_macro_and_paired_gain() -> None:
    frame = _authority_budget_frame()
    result = campaign._summarize_budget(
        frame, support_per_class=2, bootstrap_draws=12
    )
    assert result["procedures"] == 100
    scopes = result["scopes"]
    assert scopes["pooled_combined"]["census"] == {"patients": 159, "mutant": 67}
    assert scopes["pooled_combined"]["native"]["auroc"] == pytest.approx(
        campaign.EXPECTED_NATIVE_POOLED, abs=1e-15
    )
    assert scopes["equal_cohort_macro"]["native"]["auroc"] == pytest.approx(
        campaign.EXPECTED_NATIVE_EQUAL_MACRO, abs=1e-15
    )
    for scope in scopes.values():
        assert scope[f"{campaign.METHOD}_minus_native"]["auroc_gain"] == pytest.approx(0.0)


def test_budget_summary_rejects_missing_procedure_and_native_tamper() -> None:
    frame = _authority_budget_frame()
    with pytest.raises(campaign.ContractError, match="100x159"):
        campaign._summarize_budget(
            frame.iloc[:-1], support_per_class=2, bootstrap_draws=1
        )
    mutated = frame.copy()
    mutated.loc[mutated["test_cohort"].eq("RIH"), "eta_native"] *= -1
    with pytest.raises(campaign.ContractError, match="native"):
        campaign._summarize_budget(mutated, support_per_class=2, bootstrap_draws=1)


def test_production_root_is_absent_and_implementation_sources_bind_sibling_test() -> None:
    assert not campaign.DEFAULT_OUTPUT_ROOT.exists()
    sources = campaign._implementation_sources()
    assert set(sources) == {
        "lean_controller",
        "lean_test",
        "audited_source_controller",
        "embedding_runner",
        "ridge_solver",
    }
    assert sources["lean_test"]["path"].endswith(
        "test_aim2_tcga_surgen_pure_ridge_combined_fewshot_campaign.py"
    )
