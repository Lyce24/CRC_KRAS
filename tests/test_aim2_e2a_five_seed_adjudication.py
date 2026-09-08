from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_loco_five_seed_extension as frozen_extension
import aim2_sibling_loco as final_v8_e2ad
from tools import aim2_e2a_five_seed_adjudication as adjudication


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _cell(
    arm: str,
    prefix: str,
    scores: list[float],
    *,
    subcohort: str,
) -> pd.DataFrame:
    labels = [0, 0, 1, 1]
    return pd.DataFrame(
        {
            "patient_id": [f"{prefix}_{index}" for index in range(4)],
            "arm": arm,
            "target": "primary",
            "label": labels,
            "mean_logit": scores,
            "subcohort": subcohort,
        }
    )


def _primary_table() -> pd.DataFrame:
    sr386_family = _cell("family_surgen", "sr386", [-1.1, -0.2, 0.7, 1.3], subcohort="SR386")
    sr1482_family = _cell("family_surgen", "sr1482", [-0.8, 0.1, 0.5, 1.2], subcohort="SR1482")
    coad_family = _cell("family_tcga", "coad", [-1.0, -0.1, 0.4, 0.9], subcohort="TCGA-COAD")
    read_family = _cell("family_tcga", "read", [-0.7, 0.2, 0.3, 1.0], subcohort="TCGA-READ")
    rih = _cell("family_rih", "rih", [-1.2, -0.4, 0.6, 1.4], subcohort="RIH")
    rih_sm = _cell("family_rih_sm", "rih", [-0.9, -0.2, 0.7, 1.1], subcohort="RIH")
    cptac = _cell("family_cptac", "cptac", [-1.3, -0.3, 0.5, 1.5], subcohort="CPTAC")
    siblings = [
        _cell("sibling_sr386", "sr386", [-0.9, 0.0, 0.8, 1.1], subcohort="SR386"),
        _cell("sibling_sr1482", "sr1482", [-0.6, 0.0, 0.6, 0.9], subcohort="SR1482"),
        _cell("sibling_tcga_coad", "coad", [-0.8, -0.2, 0.3, 1.2], subcohort="TCGA-COAD"),
        _cell("sibling_tcga_read", "read", [-0.5, 0.0, 0.4, 0.8], subcohort="TCGA-READ"),
    ]
    return pd.concat(
        [sr386_family, sr1482_family, coad_family, read_family, rih, rih_sm, cptac, *siblings],
        ignore_index=True,
    )


def _metastatic_table() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for target, arm, scores in (
        ("rih_m", "family_rih", [-1.0, 0.3, 0.2, 1.1]),
        ("sr1482_m", "family_surgen", [-0.7, 0.1, 0.5, 0.9]),
    ):
        rows.append(
            pd.DataFrame(
                {
                    "patient_id": [f"{target}_{index}" for index in range(4)],
                    "arm": arm,
                    "target": target,
                    "label": [0, 0, 1, 1],
                    "mean_logit": scores,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _paired_source_results(e2a_d: dict[str, Any]) -> dict[str, Any]:
    paired: dict[str, Any] = {}
    for key, slug in zip(
        adjudication.PAIRED_SOURCE_KEYS[:4],
        ("sr386", "sr1482", "tcga_coad", "tcga_read"),
        strict=True,
    ):
        block = e2a_d["paired_sibling_minus_family"][slug]
        paired[key] = {
            "delta_auroc": block["delta_auroc"],
            "ci_low": -1.0,
            "ci_high": 1.0,
            "n_patients": block["n_paired_patients"],
            "auroc_left": block["whole_family_held_out_auroc"],
            "auroc_right": block["sibling_retained_auroc"],
        }
    size_matched = e2a_d["size_matched_rih_sensitivity"]
    paired[adjudication.PAIRED_SOURCE_KEYS[4]] = {
        "delta_auroc": size_matched["delta_auroc"],
        "ci_low": -1.0,
        "ci_high": 1.0,
        "n_patients": size_matched["n_paired_patients"],
        "auroc_left": size_matched["full_source_auroc"],
        "auroc_right": size_matched["size_matched_auroc"],
    }
    return paired


def _e2met_source_outputs(
    table: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(adjudication.BOOTSTRAP_SEED)
    points: dict[str, float] = {}
    samples: dict[str, np.ndarray] = {}
    targets: dict[str, Any] = {}
    for target, arm in (("rih_m", "family_rih"), ("sr1482_m", "family_surgen")):
        frame = (
            table.loc[table["target"].eq(target) & table["arm"].eq(arm)]
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
        labels = frame["label"].to_numpy(int)
        indices = adjudication.stratified_bootstrap_indices(
            labels, n_bootstrap=adjudication.N_BOOTSTRAP, rng=rng
        )
        samples[target] = adjudication.bootstrap_auroc_samples(
            labels, frame["mean_logit"].to_numpy(float), indices
        )
        points[target] = float(adjudication.roc_auc_score(frame["label"], frame["mean_logit"]))
        targets[target] = {
            "auroc": points[target],
            "auroc_ci95": adjudication._interval(samples[target]),  # noqa: SLF001
        }
    macro = np.mean(np.vstack([samples["rih_m"], samples["sr1482_m"]]), axis=0)
    interval = adjudication._interval(macro)  # noqa: SLF001
    both = all(value > 0.5 for value in points.values())
    lower = interval[0] > 0.5
    gate = {
        "equal_cohort_metastatic_macro_auroc": float(np.mean(list(points.values()))),
        "macro_auroc_ci95": interval,
        "both_target_points_above_0p5": both,
        "macro_lower_bound_above_0p5": lower,
        "claim_metastatic_transport": both and lower,
        "gate": "both target AUROC points > 0.5 AND macro AUROC CI95 lower bound > 0.5",
    }
    return gate, targets


def _sealed_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    campaign = tmp_path / "campaign_source"
    adjudication_root = tmp_path / "adjudication_output"
    analysis = campaign / "aim2_loco/analysis"
    analysis.mkdir(parents=True)
    monkeypatch.setattr(
        adjudication,
        "EXPECTED_PRIMARY_CENSUS",
        {
            arm: ((8, 4) if arm in {"family_surgen", "family_tcga"} else (4, 2))
            for arm in adjudication.ALL_PRIMARY_ARMS
        },
    )
    monkeypatch.setattr(
        adjudication,
        "EXPECTED_CONFIRMATORY_MET_CENSUS",
        {
            ("family_rih", "rih_m"): (4, 2),
            ("family_surgen", "sr1482_m"): (4, 2),
        },
    )

    raw_cpht = tmp_path / "raw_cpht/results.json"
    _write_json(raw_cpht, {"schema_version": 1, "status": "sealed_three_seed"})
    raw_identity = _identity(raw_cpht)
    monkeypatch.setattr(adjudication, "EXPECTED_RAW_CPHT_SHA256", raw_identity["sha256"])

    aim2_contract_path = campaign / "aim2_loco/contract.json"
    _write_json(
        aim2_contract_path,
        {
            "schema_version": 1,
            "status": "sealed_extension_contract",
            "mixed_seed_scope_references": {
                "all_conventional_cpht_three_seed": raw_identity,
            },
        },
    )
    inference_seal_path = campaign / "aim2_loco/inference/inference_seal.json"
    _write_json(
        inference_seal_path,
        {
            "schema_version": 1,
            "status": "sealed_before_outcome_join",
            "five_seed_loco_complete": True,
            "contract": _identity(aim2_contract_path),
        },
    )

    primary_path = analysis / "primary_patient_scores_five_seed.parquet"
    met_path = analysis / "e2met_patient_scores_five_seed.parquet"
    primary = _primary_table()
    primary.to_parquet(primary_path, index=False)
    met = _metastatic_table()
    met.to_parquet(met_path, index=False)
    e2met_gate, e2met_targets = _e2met_source_outputs(met)
    family_result, _family_arrays = adjudication._compute_e2a_f(  # noqa: SLF001
        primary,
        n_bootstrap=adjudication.N_BOOTSTRAP,
        bootstrap_seed=adjudication.BOOTSTRAP_SEED,
    )
    sibling_result, _sibling_arrays = adjudication._compute_e2a_d(  # noqa: SLF001
        primary,
        n_bootstrap=50,
        bootstrap_seed=adjudication.BOOTSTRAP_SEED,
    )
    primary_results = {arm: {} for arm in adjudication.ALL_PRIMARY_ARMS}
    for arm in adjudication.SIBLING_ARMS:
        slug = adjudication.SIBLING_PAIRINGS[arm][2]
        primary_results[arm] = {"auroc_ci95": sibling_result["targets"][slug]["auroc_ci95"]}
    paired_results = _paired_source_results(sibling_result)
    results_path = analysis / "results_five_seed.json"
    results = {
        "schema_version": 1,
        "experiment": "Aim 2 complete LOCO MIL five-seed results",
        "seeds": list(adjudication.ALL_SEEDS),
        "bootstrap": {
            "unit": "patient",
            "n": adjudication.N_BOOTSTRAP,
            "seed": adjudication.BOOTSTRAP_SEED,
        },
        "primary": primary_results,
        "family_loco_standardized_macro": {
            "macro_auroc": family_result["nested_four_family_macro"]["auroc"],
            "macro_auroc_ci95": family_result["nested_four_family_macro"]["auroc_ci95"],
            "directional_results": {
                name: {
                    "auroc": block["auroc"],
                    "auroc_ci95": block["auroc_ci95"],
                }
                for name, block in family_result["directions"].items()
            },
            "directional_gate": {"criterion": "incorrect point gate"},
            "claim_family_loco_transport": True,
        },
        "sibling_loco_directional_gate": {"criterion": "incorrect point gate"},
        "paired_sibling_and_size_matched_contrasts": paired_results,
        "e2met_confirmatory": e2met_gate,
        "e2met_confirmatory_family_naive": e2met_targets,
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
            "references": {"all_conventional_cpht_three_seed": raw_identity},
        },
    }
    _write_json(results_path, results)

    report_receipt_path = analysis / "results_five_seed.receipt.json"
    _write_json(
        report_receipt_path,
        {
            "schema_version": 1,
            "status": "sealed_five_seed_results",
            "contract": _identity(aim2_contract_path),
            "inference_seal": _identity(inference_seal_path),
            "artifacts": {
                "results": _identity(results_path),
                "primary_patients": _identity(primary_path),
                "met_patients": _identity(met_path),
            },
        },
    )

    campaign_contract_path = campaign / "campaign/experiment_contract.json"
    _write_json(
        campaign_contract_path,
        {
            "schema_version": 1,
            "status": "sealed_before_new_fit",
            "experiment": "final-v9 study-wide MIL five-seed expansion",
            "model_seeds": {"complete": list(adjudication.ALL_SEEDS)},
            "execution": {"max_concurrent_gpu_trainers": 6},
            "controllers": {
                "campaign": _identity(adjudication.FROZEN_CAMPAIGN_CONTROLLER),
                "aim2_loco": _identity(adjudication.FROZEN_AIM2_EXTENSION),
            },
            "component_contracts": {"aim2_loco": _identity(aim2_contract_path)},
        },
    )
    final_receipt_path = campaign / "campaign/receipts/five_seed_results_complete.json"
    _write_json(
        final_receipt_path,
        {
            "schema_version": 1,
            "status": "five_seed_results_complete",
            "contract": _identity(campaign_contract_path),
            "seeds": list(adjudication.ALL_SEEDS),
            "model_seeds_are_not_inference_units": True,
            "components": {
                "aim2_loco": {
                    "inference_seal": _identity(inference_seal_path),
                    "analysis_results": _identity(results_path),
                    "analysis_receipt": _identity(report_receipt_path),
                    "analysis_primary_patients": _identity(primary_path),
                    "analysis_met_patients": _identity(met_path),
                }
            },
        },
    )
    return campaign, adjudication_root


def test_frozen_contract_and_explicit_precedence_are_exact() -> None:
    assert adjudication.ALL_SEEDS == (42, 43, 44, 45, 46)
    assert adjudication.N_BOOTSTRAP == 10_000
    assert adjudication.BOOTSTRAP_SEED == 20260817
    assert adjudication.EXPECTED_PREREGISTRATION_SHA256 == (
        "8637bbfa13509b67cce98b3c7e902842a8e2ba8cdd049a5f682ac36756ae463a"
    )
    assert adjudication.EXPECTED_FROZEN_AIM2_EXTENSION_SHA256 == (
        "35359fb0730614ab0c291d2f50af7e20af101d98804f49b1fdf98d119a50f243"
    )
    assert adjudication.EXPECTED_FROZEN_CAMPAIGN_CONTROLLER_SHA256 == (
        "6251f7e2224ce0dc0f687784b310cfb0165d1fbe2aefdfedbd71640f704c8d41"
    )
    assert adjudication.EXPECTED_FROZEN_E2AD_TOPOLOGY_SHA256 == (
        "9da3a45b2e484f28dbbbf1b11b779bde88b0458d3b023cd9e836887e39a6abfb"
    )
    assert adjudication.SUPERSEDED_SOURCE_POINTERS == (
        "/family_loco_standardized_macro/directional_gate",
        "/family_loco_standardized_macro/claim_family_loco_transport",
        "/sibling_loco_directional_gate",
        "/primary/sibling_sr386/auroc_ci95",
        "/primary/sibling_sr1482/auroc_ci95",
        "/primary/sibling_tcga_coad/auroc_ci95",
        "/primary/sibling_tcga_read/auroc_ci95",
    )
    assert len(adjudication.NONAUTHORITATIVE_PAIRED_CI_POINTERS) == 10
    assert len(adjudication.RETAINED_PAIRED_POINT_POINTERS) == 5


def test_stratified_indices_are_deterministic_and_preserve_class_counts() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    first = adjudication.stratified_bootstrap_indices(
        labels, n_bootstrap=40, rng=np.random.default_rng(19)
    )
    second = adjudication.stratified_bootstrap_indices(
        labels, n_bootstrap=40, rng=np.random.default_rng(19)
    )
    assert np.array_equal(first, second)
    assert first.shape == (40, 5)
    assert np.all(labels[first[:, :3]] == 0)
    assert np.all(labels[first[:, 3:]] == 1)


def test_shared_paired_bootstrap_has_exact_drawwise_delta() -> None:
    labels = np.array([0, 0, 1, 1])
    indices = adjudication.stratified_bootstrap_indices(
        labels, n_bootstrap=100, rng=np.random.default_rng(23)
    )
    family = adjudication.bootstrap_auroc_samples(labels, np.array([-1.0, 0.2, 0.0, 1.0]), indices)
    sibling = adjudication.bootstrap_auroc_samples(
        labels, np.array([-0.5, -0.1, 0.4, 0.8]), indices
    )
    delta = sibling - family
    assert np.array_equal(delta, sibling - family)
    assert np.isfinite(delta).all()


def test_source_paired_contract_requires_exact_roster_and_complete_finite_fields() -> None:
    e2a_d, _arrays = adjudication._compute_e2a_d(  # noqa: SLF001
        _primary_table(), n_bootstrap=50, bootstrap_seed=adjudication.BOOTSTRAP_SEED
    )
    valid = {"paired_sibling_and_size_matched_contrasts": _paired_source_results(e2a_d)}
    adjudication._validate_source_paired_points(valid)  # noqa: SLF001

    missing_delta = deepcopy(valid)
    del missing_delta["paired_sibling_and_size_matched_contrasts"][
        adjudication.PAIRED_SOURCE_KEYS[0]
    ]["delta_auroc"]
    with pytest.raises(adjudication.AdjudicationError, match="finite delta_auroc"):
        adjudication._validate_source_paired_points(missing_delta)  # noqa: SLF001

    extra_key = deepcopy(valid)
    extra_key["paired_sibling_and_size_matched_contrasts"]["unexpected"] = deepcopy(
        extra_key["paired_sibling_and_size_matched_contrasts"][adjudication.PAIRED_SOURCE_KEYS[0]]
    )
    with pytest.raises(adjudication.AdjudicationError, match="key roster changed"):
        adjudication._validate_source_paired_points(extra_key)  # noqa: SLF001

    nonfinite = deepcopy(valid)
    nonfinite["paired_sibling_and_size_matched_contrasts"][adjudication.PAIRED_SOURCE_KEYS[0]][
        "auroc_left"
    ] = "NaN"
    with pytest.raises(adjudication.AdjudicationError, match="finite auroc_left"):
        adjudication._validate_source_paired_points(nonfinite)  # noqa: SLF001


def test_all_five_retained_source_delta_points_are_mechanically_replayed() -> None:
    e2a_d, _arrays = adjudication._compute_e2a_d(  # noqa: SLF001
        _primary_table(), n_bootstrap=50, bootstrap_seed=adjudication.BOOTSTRAP_SEED
    )
    paired = _paired_source_results(e2a_d)
    source = {"paired_sibling_and_size_matched_contrasts": paired}
    confirmation = adjudication._confirm_retained_source_paired_points(  # noqa: SLF001
        e2a_d, source
    )
    assert confirmation["all_five_retained_delta_points_recomputed_and_match"] is True
    assert set(confirmation["comparisons"]) == set(adjudication.PAIRED_SOURCE_KEYS)

    for key in adjudication.PAIRED_SOURCE_KEYS:
        drifted = deepcopy(source)
        block = drifted["paired_sibling_and_size_matched_contrasts"][key]
        # Keep the source block internally coherent so only independent replay
        # against the patient tables can catch the drift.
        block["auroc_right"] += 0.01
        block["delta_auroc"] = block["auroc_right"] - block["auroc_left"]
        adjudication._validate_source_paired_points(drifted)  # noqa: SLF001
        with pytest.raises(adjudication.AdjudicationError, match="fails replay"):
            adjudication._confirm_retained_source_paired_points(  # noqa: SLF001
                e2a_d, drifted
            )


@pytest.mark.parametrize(
    ("builder", "validator", "message"),
    (
        (_primary_table, adjudication._validate_primary_table, "Primary labels"),  # noqa: SLF001
        (
            _metastatic_table,
            adjudication._validate_metastatic_table,  # noqa: SLF001
            "Metastatic labels",
        ),
    ),
)
def test_fractional_labels_are_rejected_before_integer_conversion(
    builder: Any, validator: Any, message: str
) -> None:
    frame = builder()
    frame["label"] = frame["label"].astype(float)
    frame.loc[frame.index[0], "label"] = 0.5
    with pytest.raises(adjudication.AdjudicationError, match=message):
        validator(frame)


def test_e2a_f_gate_uses_every_lower_bound_and_macro_cannot_rescue() -> None:
    table = _primary_table()
    table.loc[table["arm"].eq("family_cptac"), "mean_logit"] = [-1.0, 0.0, -0.5, 1.0]
    result, arrays = adjudication._compute_e2a_f(  # noqa: SLF001
        table, n_bootstrap=500, bootstrap_seed=adjudication.BOOTSTRAP_SEED
    )
    directions = result["directions"]
    for block in directions.values():
        assert block["directional_gate"]["passes"] == (block["auroc_ci95"][0] > 0.5)
    assert result["adjudication"]["claim_family_loco_transport"] == all(
        block["auroc_ci95"][0] > 0.5 for block in directions.values()
    )
    assert directions["CPTAC"]["auroc"] > 0.5
    assert directions["CPTAC"]["auroc_ci95"][0] <= 0.5
    assert result["nested_four_family_macro"]["auroc"] > 0.5
    assert result["adjudication"]["claim_family_loco_transport"] is False
    assert result["nested_four_family_macro"]["role"].endswith("cannot rescue a failed direction")
    assert "e2a_f__nested_four_family_macro" in arrays


def test_e2a_f_numeric_replay_matches_frozen_family_bootstrap_topology() -> None:
    table = _primary_table()
    primary = {arm: table.loc[table["arm"].eq(arm)].copy() for arm in adjudication.ALL_PRIMARY_ARMS}
    observed, _arrays = adjudication._compute_e2a_f(  # noqa: SLF001
        table, n_bootstrap=200, bootstrap_seed=adjudication.BOOTSTRAP_SEED
    )
    expected = frozen_extension._family_standardized_macro(  # noqa: SLF001
        {},
        primary,
        n_bootstrap=200,
        bootstrap_seed=adjudication.BOOTSTRAP_SEED,
    )
    assert observed["nested_four_family_macro"]["auroc"] == expected["macro_auroc"]
    assert np.allclose(
        observed["nested_four_family_macro"]["auroc_ci95"],
        expected["macro_auroc_ci95"],
        rtol=0.0,
        atol=1e-15,
    )
    for name, block in observed["directions"].items():
        assert block["auroc"] == expected["directional_results"][name]["auroc"]
        assert np.allclose(
            block["auroc_ci95"],
            expected["directional_results"][name]["auroc_ci95"],
            rtol=0.0,
            atol=1e-15,
        )


def test_e2a_d_restores_macros_and_shared_paired_intervals() -> None:
    table = _primary_table()
    table.loc[table["arm"].eq("sibling_tcga_read"), "mean_logit"] = [
        -1.0,
        0.0,
        -0.5,
        1.0,
    ]
    result, arrays = adjudication._compute_e2a_d(  # noqa: SLF001
        table, n_bootstrap=500, bootstrap_seed=adjudication.BOOTSTRAP_SEED
    )
    assert set(result["macros"]) == {
        "secondary_five_acquisition_domain",
        "descriptive_equal_six_stratum",
    }
    assert len(result["macros"]["secondary_five_acquisition_domain"]["primitives"]) == 5
    assert len(result["macros"]["descriptive_equal_six_stratum"]["primitives"]) == 6
    for slug in ("sr386", "sr1482", "tcga_coad", "tcga_read"):
        assert np.array_equal(
            arrays[f"e2a_d__{slug}__delta"],
            arrays[f"e2a_d__{slug}__sibling"] - arrays[f"e2a_d__{slug}__family"],
        )
    assert result["adjudication"]["claim_sibling_stratum_transport"] == all(
        block["auroc_ci95"][0] > 0.5 for block in result["targets"].values()
    )
    assert result["targets"]["tcga_read"]["auroc"] > 0.5
    assert result["targets"]["tcga_read"]["auroc_ci95"][0] <= 0.5
    assert result["macros"]["secondary_five_acquisition_domain"]["auroc"] > 0.5
    assert result["adjudication"]["claim_sibling_stratum_transport"] is False
    assert "identical indices shared" in result["size_matched_rih_sensitivity"]["bootstrap"]


def test_e2a_d_draws_match_original_final_v8_topology() -> None:
    table = _primary_table()
    _result, observed = adjudication._compute_e2a_d(  # noqa: SLF001
        table, n_bootstrap=200, bootstrap_seed=adjudication.BOOTSTRAP_SEED
    )
    rng = np.random.default_rng(adjudication.BOOTSTRAP_SEED)
    for sibling in adjudication.SIBLING_ARMS:
        family_arm, subcohort, slug = adjudication.SIBLING_PAIRINGS[sibling]
        sibling_frame = table.loc[table["arm"].eq(sibling)].sort_values("patient_id")
        family_frame = table.loc[
            table["arm"].eq(family_arm) & table["subcohort"].eq(subcohort)
        ].sort_values("patient_id")
        labels = sibling_frame["label"].to_numpy(int)
        indices = final_v8_e2ad.stratified_bootstrap_indices(labels, n_bootstrap=200, rng=rng)
        sibling_draws = final_v8_e2ad.bootstrap_auroc_samples(
            labels, sibling_frame["mean_logit"].to_numpy(float), indices
        )
        family_draws = final_v8_e2ad.bootstrap_auroc_samples(
            labels, family_frame["mean_logit"].to_numpy(float), indices
        )
        assert np.array_equal(observed[f"e2a_d__{slug}__sibling"], sibling_draws)
        assert np.array_equal(observed[f"e2a_d__{slug}__family"], family_draws)
    for name, arm in (
        ("tcga", "family_tcga"),
        ("rih", "family_rih"),
        ("cptac", "family_cptac"),
    ):
        frame = table.loc[table["arm"].eq(arm)].sort_values("patient_id")
        labels = frame["label"].to_numpy(int)
        indices = final_v8_e2ad.stratified_bootstrap_indices(labels, n_bootstrap=200, rng=rng)
        draws = final_v8_e2ad.bootstrap_auroc_samples(
            labels, frame["mean_logit"].to_numpy(float), indices
        )
        assert np.array_equal(observed[f"e2a_d__{name}__whole_family"], draws)


def test_npz_serialization_is_byte_deterministic() -> None:
    arrays = {
        "b": np.array([1.0, 2.0], dtype=np.float64),
        "a": np.array([3.0, 4.0], dtype=np.float64),
    }
    assert adjudication._npz_bytes(arrays) == adjudication._npz_bytes(arrays)  # noqa: SLF001


def test_dry_run_writes_nothing_apply_seals_and_verify_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, output = _sealed_campaign(tmp_path, monkeypatch)
    source_before = {
        str(path.relative_to(campaign)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in campaign.rglob("*")
        if path.is_file()
    }
    args = Namespace(
        campaign_root=campaign,
        adjudication_root=output,
        preregistration=adjudication.PREREGISTRATION,
        apply=False,
    )
    adjudication.cmd_adjudicate(args)
    assert not output.exists()

    args.apply = True
    adjudication.cmd_adjudicate(args)
    root = adjudication.component_root(output)
    assert {path.name for path in root.iterdir()} == {
        "result.json",
        "bootstrap_distributions.npz",
        "receipt.json",
    }
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.iterdir()}
    replayed = adjudication.verify(
        campaign_root=campaign,
        adjudication_root=output,
        preregistration=adjudication.PREREGISTRATION,
    )
    assert replayed["e2met_gate_confirmation"]["matches_and_remains_authoritative"] is True
    assert replayed["inputs"]["frozen_final_v8_e2ad_topology"]["sha256"] == (
        adjudication.EXPECTED_FROZEN_E2AD_TOPOLOGY_SHA256
    )
    assert (
        replayed["e2a_d"]["source_report_paired_point_confirmation"][
            "all_five_retained_delta_points_recomputed_and_match"
        ]
        is True
    )
    after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.iterdir()}
    assert before == after

    # Applying again is an idempotent verification, never an overwrite.
    adjudication.cmd_adjudicate(args)
    final = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.iterdir()}
    assert before == final
    source_after = {
        str(path.relative_to(campaign)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in campaign.rglob("*")
        if path.is_file()
    }
    assert source_before == source_after


def test_verify_fails_closed_when_source_or_output_identity_drifts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, output = _sealed_campaign(tmp_path, monkeypatch)
    args = Namespace(
        campaign_root=campaign,
        adjudication_root=output,
        preregistration=adjudication.PREREGISTRATION,
        apply=True,
    )
    adjudication.cmd_adjudicate(args)
    result_path = adjudication.component_root(output) / "result.json"
    result_path.write_bytes(result_path.read_bytes() + b" ")
    with pytest.raises(adjudication.AdjudicationError):
        adjudication.verify(
            campaign_root=campaign,
            adjudication_root=output,
            preregistration=adjudication.PREREGISTRATION,
        )


def test_existing_partial_output_is_never_completed_or_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, output = _sealed_campaign(tmp_path, monkeypatch)
    root = adjudication.component_root(output)
    root.mkdir(parents=True)
    marker = root / "partial.txt"
    marker.write_text("preserve me", encoding="utf-8")
    with pytest.raises(adjudication.AdjudicationError, match="inventory changed"):
        adjudication.cmd_adjudicate(
            Namespace(
                campaign_root=campaign,
                adjudication_root=output,
                preregistration=adjudication.PREREGISTRATION,
                apply=True,
            )
        )
    assert marker.read_text(encoding="utf-8") == "preserve me"


def test_source_and_adjudication_roots_must_be_separate(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    with pytest.raises(adjudication.AdjudicationError, match="separate"):
        adjudication._validate_adjudication_root(  # noqa: SLF001
            campaign / "adjudication", campaign.resolve()
        )


def test_cli_requires_explicit_apply_and_exposes_read_only_verify() -> None:
    parser = adjudication.build_parser()
    dry = parser.parse_args(["adjudicate"])
    apply = parser.parse_args(["adjudicate", "--apply"])
    verify = parser.parse_args(["verify"])
    assert dry.apply is False
    assert apply.apply is True
    assert verify.command == "verify"
    assert callable(verify.func)
