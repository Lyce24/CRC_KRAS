"""Synthetic tests of the locked Module III fitting and inferential boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import final_v14_modeling_v2 as modeling
from tools import final_v14_module3 as runner


def paired_frames() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(13)
    fine = pd.DataFrame({
        "patient_id": [f"P{i:02d}" for i in range(12)] + [f"F{i:02d}" for i in range(12)],
        "label": [1] * 12 + [0] * 12,
        "subcohort": (["A"] * 6 + ["B"] * 6) * 2,
        "k_fold": ([0, 0, 0, 1, 1, 1] * 2) * 2,
        "mean_logit": rng.normal(size=24),
    }).sort_values("patient_id").reset_index(drop=True)
    fine["mil_logit"] = fine.mean_logit - rng.normal(0, .2, len(fine))
    controls = {}
    for seed in runner.WT_SEEDS:
        frame = fine.drop(columns="mil_logit").copy()
        frame.loc[frame.label.eq(0), "patient_id"] = [f"W{seed}-{i:02d}" for i in range(12)]
        frame["mean_logit"] = rng.normal(size=24) + frame.label * .7
        controls[str(seed)] = frame
    return fine, controls


def inherited_canonical():
    """Load only the three sealed pure helpers, without training/WSI imports."""
    path = Path(__file__).resolve().parents[1] / "aim3_repeated_control_campaign.py"
    tree = ast.parse(path.read_text())
    names = {"_auc", "_blocks", "_draw_block_matrix", "partial_paired_bootstrap"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "pd": pd, "N_BOOTSTRAP": 20000, "BOOTSTRAP_SEED": 20260826, "Any": object}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["partial_paired_bootstrap"]


def metric(lower=.51, upper=.59, estimate=.55):
    return {"status": "ESTIMABLE", "estimate": estimate, "fwer_lower99": lower, "fwer_upper99": upper}


def draw_rows(control_lower=.6, verdict="CEILING_WITH_RESIDUAL_SIGNAL"):
    return {str(seed): {"control": metric(control_lower, .8, .7), "verdict": verdict} for seed in runner.WT_SEEDS}


def test_canonical_replays_inherited_sample_arrays_exactly():
    fine, controls = paired_frames()
    control = controls[str(runner.WT_SEEDS[0])].sort_values("patient_id")
    seed = runner.canonical_seed("codon")
    inherited = inherited_canonical()(fine, control, n_bootstrap=551, seed=seed)
    result = runner.partially_paired_bootstrap(fine, {"canonical": control}, n_bootstrap=551, seed=seed)
    np.testing.assert_array_equal(result["fine"], inherited["fine_values"])
    np.testing.assert_array_equal(result["control_canonical"], inherited["control_values"])
    np.testing.assert_array_equal(result["control_minus_fine_canonical"], inherited["delta_values"])


def test_three_draws_pair_fine_scores_and_use_independent_wt_indices():
    fine, controls = paired_frames()
    # Identical numerical control pools distinguish independent sampling from
    # restarting the RNG for every WT draw. Patient identities still differ.
    template = controls[str(runner.WT_SEEDS[0])].sort_values("patient_id")
    for frame in controls.values():
        for label in (0, 1):
            frame.loc[frame.label.eq(label), "mean_logit"] = template.loc[template.label.eq(label), "mean_logit"].to_numpy()
    fine["mil_logit"] = fine.mean_logit
    result = runner.partially_paired_bootstrap(fine, controls, n_bootstrap=77, seed=runner.repeated_seed("codon"))
    np.testing.assert_array_equal(result["concept_minus_mil"], np.zeros(77))
    assert not np.array_equal(result["control_20260823"], result["control_20260824"])
    for seed in runner.WT_SEEDS:
        np.testing.assert_array_equal(result[f"control_minus_fine_{seed}"], result[f"control_{seed}"] - result["fine"])
    replay = runner.partially_paired_bootstrap(fine, controls, n_bootstrap=77, seed=runner.repeated_seed("codon"))
    for key in result:
        np.testing.assert_array_equal(result[key], replay[key])


def test_bootstrap_rejects_unpaired_patients_and_unmatched_strata():
    fine, controls = paired_frames()
    controls["20260823"].loc[controls["20260823"].label.eq(1), "patient_id"] += "mismatch"
    with pytest.raises(runner.ContractError, match="positives"):
        runner.partially_paired_bootstrap(fine, controls, n_bootstrap=2, seed=1)
    fine, controls = paired_frames()
    controls["20260823"].loc[controls["20260823"].label.eq(0), "subcohort"] = "NEW"
    with pytest.raises(runner.ContractError, match="matched"):
        runner.partially_paired_bootstrap(fine, controls, n_bootstrap=2, seed=1)


def test_missing_control_keeps_fine_rng_and_fine_signal():
    fine, controls = paired_frames()
    complete = runner.partially_paired_bootstrap(fine, controls, n_bootstrap=9, seed=8)
    controls["20260824"]["mean_logit"] = np.nan
    missing = runner.partially_paired_bootstrap(fine, controls, n_bootstrap=9, seed=8)
    np.testing.assert_array_equal(missing["fine"], complete["fine"])
    assert np.isnan(missing["control_20260824"]).all()
    rows = draw_rows()
    rows["20260824"]["control"] = {"status": "NOT_ESTIMABLE"}
    statuses = runner.task_statuses(metric(), metric(.01, .06, .04), rows)
    assert statuses["control"] == "CONCEPT_CONTROL_NOT_EVALUABLE"
    assert statuses["boundary"] == "CONCEPT_BOUNDARY_NOT_EVALUABLE"
    assert statuses["superiority"] == "CONCEPT_FINE_RANKING_SUPERIORITY_NOT_EVALUABLE"
    assert statuses["fine_signal"] == "CONCEPT_FINE_SIGNAL_SUPPORTED"


def test_boundary_and_superiority_coexist_and_canonical_is_not_an_input():
    statuses = runner.task_statuses(metric(), metric(.01, .06, .04), draw_rows())
    assert statuses["boundary"] == "CONCEPT_NONRESOLUTION_CONSISTENT"
    assert statuses["superiority"] == "CONCEPT_FINE_RANKING_SUPERIOR_TO_MIL"
    assert not statuses["point_fine_auroc_at_least_060"]
    rows = draw_rows()
    rows["20260825"]["verdict"] = "FINE_RESOLUTION_EVIDENCE"
    assert runner.task_statuses(metric(), metric(.01, .06, .04), rows)["boundary"] == "CONCEPT_NONRESOLUTION_NOT_ESTABLISHED"


@pytest.mark.parametrize(("fine", "control", "delta", "expected"), [
    (metric(), metric(.5, .8, .7), metric(.01, .2, .1), "UNDERPOWERED"),
    (metric(), metric(.6, .8, .7), metric(.01, .2, .1), "CEILING_WITH_RESIDUAL_SIGNAL"),
    (metric(.5, .59, .55), metric(.6, .8, .7), metric(.01, .2, .1), "CEILING"),
    (metric(.51, .60, .55), metric(.6, .8, .7), metric(.01, .2, .1), "FINE_RESOLUTION_EVIDENCE"),
    (metric(.5, .60, .55), metric(.6, .8, .7), metric(0, .2, .1), "INCONCLUSIVE"),
])
def test_draw_gate_order_and_strict_thresholds(fine, control, delta, expected):
    assert runner.draw_verdict(fine, control, delta) == expected


def test_inadequate_control_blocks_boundary_but_retains_fine_estimate():
    statuses = runner.task_statuses(metric(), metric(.01, .06, .04), draw_rows(.5, "UNDERPOWERED"))
    assert statuses["control"] == "CONCEPT_CONTROL_INADEQUATE"
    assert statuses["boundary"] == "CONCEPT_BOUNDARY_NOT_EVALUABLE"
    assert statuses["fine_signal"] == "CONCEPT_FINE_SIGNAL_SUPPORTED"
    assert statuses["superiority"] == "CONCEPT_FINE_RANKING_SUPERIORITY_NOT_ESTABLISHED"


def test_seeds_and_minority_class_split_count_follow_task_draw_fold_indices():
    assert runner.inner_seed("g12c", "repeated", 20260825, 4) == 20264853
    assert runner.inner_seed("g12d_broad", "fine", None, 2) == runner.inner_seed("g12d_broad", "fixed", None, 2)
    assert len(runner.make_inner_splits(np.array([0] * 10 + [1] * 3), 123)) == 3
    assert runner.make_inner_splits(np.array([0] * 10 + [1]), 123) == []


def test_fit_outer_excludes_heldout_labels_and_sorts_training_patients(monkeypatch):
    count = 40
    roster = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(count)],
                           "label": np.arange(count) % 2, "subcohort": "A", "k_fold": np.arange(count) % 5})
    profiles = roster[["patient_id"]].copy()
    for index, column in enumerate(runner.PROTOTYPES):
        profiles[column] = np.arange(count) + index
    calls = []

    def capture(X, y, test, splits):
        calls.append((X.copy(), y.copy(), test.copy(), splits))
        return {"status": "ESTIMABLE", "predictions": np.zeros(len(test)).tolist(), "selected_penalty": 1.0}

    monkeypatch.setattr(modeling, "nested_logistic", capture)
    result, predictions = runner.fit_outer(roster.iloc[::-1], profiles, [2, 5], "codon", "fine", None, 0)
    expected = roster.loc[roster.k_fold.ne(0)].sort_values("patient_id")
    np.testing.assert_array_equal(calls[0][1], expected.label)
    assert result["audit"]["training_patient_ids"] == expected.patient_id.tolist()
    assert predictions.k_fold.eq(0).all()
    perturbed = roster.copy()
    perturbed.loc[perturbed.k_fold.eq(0), "label"] = 1 - perturbed.loc[perturbed.k_fold.eq(0), "label"]
    runner.fit_outer(perturbed, profiles, [2, 5], "codon", "fine", None, 0)
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    np.testing.assert_array_equal(calls[0][1], calls[1][1])
    for left, right in zip(calls[0][3], calls[1][3], strict=True):
        for a, b in zip(left, right, strict=True):
            np.testing.assert_array_equal(a, b)


def test_immutable_artifacts_reject_changes_and_require_naming_seal(tmp_path):
    path = tmp_path / "name.json"
    runner.write_json(path, {"status": "NAME_GATE_FAIL", "named_ref": [], "named_oof": {}})
    with pytest.raises(FileNotFoundError):
        runner.validate_name_freeze(path)
    runner.write_json(path.with_suffix(".json.seal.json"), {"artifact": runner.identity(path)})
    assert runner.validate_name_freeze(path)["status"] == "NAME_GATE_FAIL"
    with pytest.raises(runner.ContractError, match="immutable"):
        runner.write_json(path, {"status": "NAME_GATE_PASS"})
    path.write_text('{"status":"NAME_GATE_PASS"}')
    with pytest.raises(runner.ContractError, match="Identity"):
        runner.validate_name_freeze(path)


def test_post_reader_schema_does_not_authorize_named_fits_when_gate_fails(tmp_path):
    path = tmp_path / "name.json"
    runner.write_json(path, {"status": "NAMING_FROZEN_AFTER_AUTHORIZED_LITERAL_INTAKE",
                            "name_gate_status": "NAME_GATE_FAIL", "NAMED_REF": [6],
                            "NAMED_OOF": {str(i): [2] for i in range(5)}})
    runner.write_json(path.with_suffix(".json.seal.json"), {"artifact": runner.identity(path)})
    names = runner.validate_name_freeze(path)
    assert names["status"] == "NAME_GATE_FAIL"
    assert names["named_ref"] == [6]


def test_canonical_summaries_do_not_include_governed_bounds():
    result = runner.summarize(.6, np.linspace(.5, .7, 10000), governed=False)
    assert result["ci95"] == pytest.approx([.505, .695])
    assert "fwer_lower99" not in result
    assert runner.summarize(.6, np.linspace(.5, .7, 20000))["fwer_lower99"] == pytest.approx(.502)
