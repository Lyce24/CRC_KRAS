from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aim2_tcga_surgen_residual_combined_fewshot_campaign as campaign


def _synthetic_data(*, patients_per_class: int = 20) -> dict[str, dict[int, pd.DataFrame]]:
    columns = [f"e{index}" for index in range(campaign.EMBED_DIM)]
    result: dict[str, dict[int, pd.DataFrame]] = {}
    for cohort_number, cohort in enumerate(("RIH", "SurGen")):
        labels = np.asarray([0] * patients_per_class + [1] * patients_per_class, dtype=int)
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
            seed: base.assign(e2=float(seed - 42)).copy() for seed in campaign.MODEL_SEEDS
        }
    return result


def _selection_record(total: int) -> dict[str, object]:
    keys = [campaign.broad.adapter.lambda_key(value) for value in campaign.LAMBDA_GRID]
    splits = total if total <= campaign.broad.adapter.LOO_POOL_MAX else len(campaign.FOLDS)
    losses = {key: [float(index + 1)] * splits for index, key in enumerate(keys)}
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
            "scheme": (
                "leave_one_out" if total <= campaign.broad.adapter.LOO_POOL_MAX else "five_fold"
            ),
            "inner_seed": campaign.LAYOUT_SEEDS[0],
            "n_pool": total,
            "n_splits": splits,
            "losses_by_lambda": losses,
            "mean_loss_by_lambda": means,
            "solver_summary_by_lambda": summaries,
            "selected_lambda": keys[0],
        },
    }


def test_exact_residual_matrix_and_accounting() -> None:
    observed = campaign.accounting()
    assert len(campaign.residual_jobs()) == 20
    assert len({job.key for job in campaign.residual_jobs()}) == 20
    assert observed == {
        "source_main_model_fits": 0,
        "new_label_blind_embedding_jobs": 0,
        "reused_sealed_embedding_artifacts": 10,
        "unique_support_procedures_by_fold": 2_000,
        "final_residual_head_decisions": 10_000,
        "fit_to_test_cohort_applications": 20_000,
        "inner_plus_final_solver_calls": 1_330_000,
        "residual_adapter_fits": 10_000,
        "pure_probe_fits": 0,
        "local_mil_fits": 0,
        "full_label_fits": 0,
        "platt_fits": 0,
    }
    assert campaign.METHOD == "source_anchored_residual_ridge"
    assert campaign.SUPPORT_PER_CLASS == (2, 4, 8, 16)


@pytest.mark.parametrize(
    ("support", "calls_per_head", "calls_per_shard"),
    ((2, 65, 32_500), (4, 129, 64_500), (8, 257, 128_500), (16, 81, 40_500)),
)
def test_exact_nested_solver_scheme_and_census(
    support: int, calls_per_head: int, calls_per_shard: int
) -> None:
    job = campaign.ResidualJob(campaign.LAYOUT_SEEDS[0], support)
    assert campaign._solver_calls_per_head(support) == calls_per_head
    assert campaign._expected_shard_solver_calls(job) == calls_per_shard


def test_output_root_firewall_rejects_source_and_upstream(tmp_path: Path) -> None:
    assert campaign.validate_output_root(tmp_path / "new") == (tmp_path / "new").resolve()
    for invalid in (
        Path("relative"),
        campaign.SOURCE_ROOT,
        campaign.SOURCE_ROOT / "child",
        campaign.UPSTREAM_ROOT,
        campaign.UPSTREAM_ROOT / "child",
        Path("/var/tmp/not-governed"),
    ):
        with pytest.raises(campaign.ContractError):
            campaign.validate_output_root(invalid)


@pytest.mark.parametrize("support", campaign.SUPPORT_PER_CLASS)
def test_supports_are_upstream_identical_balanced_and_leak_free(
    support: int,
) -> None:
    data = _synthetic_data()
    kwargs = {
        "layout_seed": campaign.LAYOUT_SEEDS[0],
        "support_per_class": support,
        "draw": 7,
        "fold": 3,
    }
    observed = campaign._combined_support(data, **kwargs)
    expected = campaign.pure._combined_support(data, **kwargs)
    assert observed == expected
    assert observed == campaign._combined_support(data, **kwargs)
    assert len(observed) == 2 * support
    folds = campaign._folds_by_cohort(data, campaign.LAYOUT_SEEDS[0])
    for cohort in ("RIH", "SurGen"):
        block = [(patient, label) for source, patient, label in observed if source == cohort]
        assert sum(label == 0 for _patient, label in block) == support // 2
        assert sum(label == 1 for _patient, label in block) == support // 2
        held_out = {
            str(data[cohort][campaign.MODEL_SEEDS[0]].index[index])
            for index in np.flatnonzero(folds[cohort] == kwargs["fold"])
        }
        assert not held_out & {patient for patient, _label in block}


@pytest.mark.parametrize("total", (4, 32))
def test_selection_replays_and_rejects_scheme_mutation(total: int) -> None:
    record = _selection_record(total)
    campaign._validate_selection(record)
    record["inner_selection"]["scheme"] = (  # type: ignore[index]
        "five_fold" if total == 4 else "leave_one_out"
    )
    with pytest.raises(campaign.ContractError):
        campaign._validate_selection(record)


def test_finite_residual_head_uses_native_offsets_and_detects_mutation() -> None:
    features = np.asarray([[-1.0, 0.0], [-0.5, 1.0], [0.5, -1.0], [1.0, 0.0]], dtype=float)
    features = np.pad(features, ((0, 0), (0, campaign.EMBED_DIM - 2)))
    offsets = np.asarray([-1.2, -0.7, 0.2, 0.9], dtype=float)
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    ledger = campaign.broad.adapter.SolverLedger()
    weights, bias, diagnostic = campaign.broad.adapter.fit_residual(
        features,
        offsets,
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
    campaign._validate_final_head(finite, features=features, offsets=offsets, labels=labels)
    with pytest.raises(campaign.ContractError):
        campaign._validate_final_head(
            finite,
            features=features,
            offsets=offsets + 0.25,
            labels=labels,
        )
    mutated = dict(finite)
    mutated["coefficients"] = finite["coefficients"].copy()
    mutated["coefficients"][0] += 0.1
    with pytest.raises(campaign.ContractError):
        campaign._validate_final_head(mutated, features=features, offsets=offsets, labels=labels)


def test_infinity_is_exact_native_no_correction_anchor() -> None:
    features = np.zeros((4, campaign.EMBED_DIM), dtype=float)
    features[:, 0] = [-1.0, -0.5, 0.5, 1.0]
    offsets = np.asarray([-2.0, -0.4, 0.3, 1.8], dtype=float)
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    ledger = campaign.broad.adapter.SolverLedger()
    weights, bias, diagnostic = campaign.broad.adapter.fit_residual(
        features,
        offsets,
        labels,
        math.inf,
        ledger=ledger,
        context={"test": True},
    )
    record = {
        "selected_lambda": "infinity",
        "coefficients": weights.tolist(),
        "bias": bias,
        "solver_diagnostic": diagnostic,
    }
    campaign._validate_final_head(record, features=features, offsets=offsets, labels=labels)
    assert np.array_equal(offsets + features @ weights + bias, offsets)
    record["bias"] = 0.01
    with pytest.raises(campaign.ContractError, match="native/no-correction"):
        campaign._validate_final_head(record, features=features, offsets=offsets, labels=labels)


def test_contract_declares_residual_only_and_direct_upstream_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        campaign,
        "_implementation_sources",
        lambda: {"test": {"path": "/tmp/test", "sha256": "0" * 64, "size_bytes": 1}},
    )
    upstream = {"sealed": {"path": "/tmp/sealed", "sha256": "1" * 64, "size_bytes": 1}}
    payload = campaign._contract_payload(
        tmp_path / "campaign",
        created_utc="2026-08-28T00:00:00+00:00",
        upstream_artifacts=upstream,
    )
    assert payload["protocol"]["formula"] == "eta_adapt = eta_native + H @ w + b"
    assert payload["protocol"]["infinity_semantics"].startswith("exact frozen")
    assert payload["upstream"]["reuse_mode"].startswith("direct read-only")
    assert payload["upstream"]["artifacts"] == upstream
    assert payload["firewall"]["residual_head_only"] is True
    assert payload["firewall"]["pure_probe_permitted"] is False
    assert payload["firewall"]["external_validation_claim_permitted"] is False
    assert payload["accounting"] == campaign.accounting()


def test_prepare_dry_run_is_nonmutating_and_binds_no_new_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    monkeypatch.setattr(
        campaign,
        "_upstream_artifacts",
        lambda: {"sealed": {"path": "/tmp/x", "sha256": "0" * 64, "size_bytes": 1}},
    )
    value = campaign.prepare(root, apply=False)
    assert value["new_embedding_jobs"] == 0
    assert value["bound_upstream_artifacts"] == 1
    assert not root.exists()


def test_plan_has_no_rescoring_or_target_open_stage(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    plan = campaign.campaign_plan(root)
    sequence = " ".join(plan["production_sequence"])
    assert "score-source" not in sequence
    assert "open-targets" not in sequence
    assert plan["method"] == campaign.METHOD
    assert plan["formula"] == "eta_adapt = eta_native + H @ w + b"
    assert plan["support_regime"] == "COMBINED"
    assert plan["support_per_class_total"] == [2, 4, 8, 16]
    assert plan["shards"] == 20
    assert plan["max_workers"] == 6
    assert plan["production_launch_authorized_by_this_command"] is False
    assert not root.exists()


def test_run_adaptation_rejects_non_governed_worker_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    monkeypatch.setattr(campaign, "_validate_preflight", lambda _root, deep: {})
    with pytest.raises(campaign.ContractError, match="--max-workers 6"):
        campaign.run_adaptation(root, apply=False, max_workers=5)


def test_summary_proxy_is_scoped_and_rejects_pure_columns() -> None:
    frame = pd.DataFrame(
        {
            "eta_residual_ridge": [0.1],
            **{f"eta_residual_ridge_seed{seed}": [float(seed)] for seed in campaign.MODEL_SEEDS},
        }
    )
    original = campaign.pure.METHOD
    proxy = campaign._summary_proxy(frame)
    assert "eta_pure_ridge" in proxy
    with campaign._pure_summary_context():
        assert campaign.pure.METHOD == campaign.METHOD
    assert original == campaign.pure.METHOD
    proxy["eta_residual_ridge"] = 0.0
    with pytest.raises(campaign.ContractError, match="forbidden pure-probe"):
        campaign._summary_proxy(proxy)


def test_expected_oof_columns_contain_native_and_residual_only() -> None:
    columns = campaign._expected_oof_columns()
    assert "eta_native" in columns
    assert "eta_residual_ridge" in columns
    assert "eta_pure_ridge" not in columns
    assert len(columns) == 21


def test_missing_shard_completion_fails_without_minting(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    job = campaign.ResidualJob(campaign.LAYOUT_SEEDS[0], 2)
    directory = campaign.shard_dir(root, job)
    directory.mkdir(parents=True)
    for name in (
        "oof.parquet",
        "fits.jsonl",
        "supports.jsonl",
        "solver.json",
        "execution.json",
    ):
        (directory / name).write_bytes(b"partial")
    with pytest.raises(campaign.ContractError, match="partial or noncanonical"):
        campaign._validate_shard(root, job, deep=False)
    assert not (directory / "completion.json").exists()


def test_reported_artifact_uses_canonical_path(tmp_path: Path) -> None:
    source = tmp_path / "stage.json"
    source.write_text("{}\n", encoding="utf-8")
    canonical = tmp_path / "final/completion.json"
    artifact = campaign._reported_artifact(source, canonical)
    assert artifact["path"] == str(canonical)
    assert artifact["size_bytes"] == 3


def test_crash_staging_reconciliation_is_scoped_and_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    staging = root / "adaptation/.staging"
    stale = staging / f"{campaign.residual_jobs()[0].key}.deadworker"
    stale.mkdir(parents=True)
    (stale / "oof.parquet").write_bytes(b"incomplete")
    campaign._reconcile_shard_staging(root)
    assert staging.is_dir()
    assert not any(staging.iterdir())
    unsafe = staging / "unexpected"
    unsafe.mkdir()
    with pytest.raises(campaign.ContractError, match="unsafe residual staging"):
        campaign._reconcile_shard_staging(root)


def _authority_budget_frame() -> pd.DataFrame:
    data = campaign._load_upstream_patient_data()
    patients: list[pd.DataFrame] = []
    for cohort in ("RIH", "SurGen"):
        reference = data[cohort][campaign.MODEL_SEEDS[0]]
        block = pd.DataFrame(
            {
                "label": reference["label"].to_numpy(int),
                "test_cohort": cohort,
                "patient_id": reference.index.astype(str),
                **{
                    f"eta_native_seed{seed}": data[cohort][seed]["eta_native"].to_numpy(float)
                    for seed in campaign.MODEL_SEEDS
                },
            }
        )
        block["eta_native"] = block[
            [f"eta_native_seed{seed}" for seed in campaign.MODEL_SEEDS]
        ].mean(axis=1)
        patients.append(block)
    authority = pd.concat(patients, ignore_index=True)
    rows: list[pd.DataFrame] = []
    for layout in campaign.LAYOUT_SEEDS:
        for draw in range(campaign.DRAWS_PER_LAYOUT):
            block = authority.copy()
            block["layout_seed"] = layout
            block["draw"] = draw
            block["support_per_class"] = 2
            block["support_total"] = 4
            block["eta_residual_ridge"] = block["eta_native"]
            for seed in campaign.MODEL_SEEDS:
                block[f"eta_residual_ridge_seed{seed}"] = block[f"eta_native_seed{seed}"]
            rows.append(block)
    return pd.concat(rows, ignore_index=True)


def test_budget_summary_replays_native_authority_and_paired_zero_gain() -> None:
    result = campaign._summarize_budget(
        _authority_budget_frame(), support_per_class=2, bootstrap_draws=8
    )
    assert result["method"] == campaign.METHOD
    assert result["native_offset_frozen"] is True
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


def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path: Path) -> None:
    for index, payload in enumerate(('{"a":1,"a":2}', '{"a":NaN}')):
        path = tmp_path / f"bad-{index}.json"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(campaign.ContractError):
            campaign._read_json(path)
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text(json.dumps({"ok": 1}) + "\n" + '{"x":1,"x":2}\n')
    with pytest.raises(campaign.ContractError):
        campaign._read_jsonl(jsonl)


def test_terminal_namespace_rejects_copied_upstream_data(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    campaign._validate_terminal_namespace(root)
    copied = root / "source_inference"
    copied.mkdir()
    with pytest.raises(campaign.ContractError, match="forbidden copied"):
        campaign._validate_terminal_namespace(root)


def test_production_root_absent_and_sources_bind_both_campaigns() -> None:
    assert not campaign.DEFAULT_OUTPUT_ROOT.exists()
    sources = campaign._implementation_sources()
    assert set(sources) == {
        "residual_controller",
        "residual_test",
        "sealed_upstream_controller",
        "sealed_upstream_test",
        "source_campaign_controller",
        "ridge_solver",
    }
    assert sources["residual_test"]["path"].endswith(
        "test_aim2_tcga_surgen_residual_combined_fewshot_campaign.py"
    )
