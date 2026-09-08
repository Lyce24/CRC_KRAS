from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_a_orion_residual_adaptation as cpht_a  # noqa: E402


def _toy_data(seed: int = 7) -> cpht_a.PatientData:
    rng = np.random.default_rng(seed)
    labels = np.array([0] * 25 + [1] * 15, dtype=np.int8)
    patient_ids = np.asarray([f"ORION:T{index:02d}" for index in range(40)])
    features = rng.normal(0, 0.05, size=(3, 40, 512))
    native = rng.normal(size=(3, 40))
    return cpht_a.PatientData(
        patient_ids=patient_ids,
        labels=labels,
        folds=cpht_a.stratified_folds(labels),
        features=features,
        native=native,
    )


def _toy_design(data: cpht_a.PatientData) -> cpht_a.RealizedDesign:
    folds = pd.DataFrame(
        {
            "patient_id": data.patient_ids,
            "label": data.labels,
            "outer_fold": data.folds,
        }
    )
    rows: list[dict[str, object]] = []
    for budget in cpht_a.BUDGETS:
        for draw in range(cpht_a.N_PROCEDURES):
            for outer_fold in range(cpht_a.N_FOLDS):
                support, seed = cpht_a._generate_support_for_seal(  # noqa: SLF001
                    data.labels, data.folds, budget, draw, outer_fold
                )
                for ordinal, patient_index in enumerate(support):
                    rows.append(
                        {
                            "budget_per_class": budget,
                            "procedure_draw": draw,
                            "outer_fold": outer_fold,
                            "support_seed": str(seed),
                            "model_seed_roster": "42|43|44",
                            "support_ordinal": ordinal,
                            "patient_id": str(data.patient_ids[patient_index]),
                            "label": int(data.labels[patient_index]),
                        }
                    )
    design = cpht_a.RealizedDesign(folds=folds, supports=pd.DataFrame(rows))
    cpht_a._validate_realized_design(design.folds, design.supports)  # noqa: SLF001
    return design


def test_fixed_stratified_layout_is_exact_and_reproducible() -> None:
    labels = np.array([0] * 25 + [1] * 15, dtype=np.int8)
    first = cpht_a.stratified_folds(labels)
    second = cpht_a.stratified_folds(labels)
    assert np.array_equal(first, second)
    assert set(first.tolist()) == set(range(5))
    for fold in range(5):
        test = labels[first == fold]
        assert len(test) == 8
        assert int(test.sum()) == 3
        assert int((test == 0).sum()) == 5


@pytest.mark.parametrize("budget", cpht_a.BUDGETS)
def test_seeded_support_is_exact_balanced_and_other_fold_only(budget: int) -> None:
    data = _toy_data()
    for fold in range(cpht_a.N_FOLDS):
        first, first_seed = cpht_a._generate_support_for_seal(  # noqa: SLF001
            data.labels, data.folds, budget, draw=17, outer_fold=fold
        )
        second, second_seed = cpht_a._generate_support_for_seal(  # noqa: SLF001
            data.labels, data.folds, budget, draw=17, outer_fold=fold
        )
        assert first_seed == second_seed
        assert np.array_equal(first, second)
        assert len(first) == len(set(first.tolist())) == 2 * budget
        assert int(data.labels[first].sum()) == budget
        assert np.all(data.folds[first] != fold)


def test_residual_solver_matches_exact_v7_e2c_kkt_and_infinity_is_exact() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(0, 0.1, size=(8, 512))
    native = rng.normal(size=8)
    labels = np.array([0] * 4 + [1] * 4, dtype=float)

    weights, bias, diagnostic = cpht_a.fit_residual(features, native, labels, lam=3.0)
    reference = cpht_a.e2c_offset.native_offset_fit_diagnostics(
        features, labels, native, weights, bias, 3.0
    )
    assert diagnostic["status"] == "finite_optimum"
    assert reference["kkt_gradient_inf_norm"] <= cpht_a.SOLVER_GRAD_TOL
    assert reference["objective_decrease"] >= -cpht_a.OBJECTIVE_TOL
    assert diagnostic["gradient_inf_norm"] == reference["kkt_gradient_inf_norm"]

    null_weights, null_bias, null_diagnostic = cpht_a.fit_residual(
        features, native, labels, lam=np.inf
    )
    assert np.array_equal(null_weights, np.zeros(512))
    assert null_bias == 0.0
    assert null_diagnostic["status"] == "native_exact"


def test_ordered_tie_prefers_infinity() -> None:
    losses: np.ndarray = np.ones(len(cpht_a.LAMBDA_GRID), dtype=float)
    assert cpht_a._tie_preferred_index(losses) == 0  # noqa: SLF001
    losses[-1] -= 1e-6
    assert cpht_a._tie_preferred_index(losses) == len(cpht_a.LAMBDA_GRID) - 1  # noqa: SLF001
    losses[0] = losses[-1]
    assert cpht_a._tie_preferred_index(losses) == 0  # noqa: SLF001


def test_loso_loss_and_selection_are_exact_v7_e2c_on_clipping_counterexample() -> None:
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    native = np.array(
        [8.159438018986767, -16.232357552032692, 10.687239335842143, 6.068997958472428]
    )
    features = np.array(
        [
            [-0.47740362773947176, 0.39084802949577685, 0.4894800956685554],
            [0.42088425820691994, -0.4098035487857844, 0.3578727866716222],
            [-0.4893586591169016, 0.3347773520286798, -0.28974212258198534],
            [0.05300461937078067, 0.40035692274213736, 0.1977049868135293],
        ]
    )
    selected, losses = cpht_a.select_lambda(features, native, labels)
    reference = np.asarray(
        [
            cpht_a.e2c_offset.loo_loss_native_offset(features, labels, native, lam)
            for lam in cpht_a.LAMBDA_GRID
        ],
        dtype=float,
    )
    expected = np.array(
        [
            8.166201886864451,
            8.166192976977339,
            8.16617218725091,
            8.166112788084803,
            8.165904891689667,
            8.165310907793037,
            8.163232032205608,
            8.157292977165636,
            8.136513424452932,
            8.077211833720659,
        ]
    )
    assert np.allclose(losses.mean(axis=1), reference, atol=1e-14, rtol=0)
    assert np.allclose(reference, expected, atol=1e-12, rtol=0)
    assert cpht_a.LAMBDA_GRID[selected] == 1.0
    assert cpht_a.e2c_offset.select_lambda_native_offset(features, labels, native) == 1.0


def test_complete_procedure_uses_native_exact_branch_without_ensemble_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _toy_data()
    design = _toy_design(data)
    monkeypatch.setattr(cpht_a, "LAMBDA_GRID", (np.inf,))
    monkeypatch.setattr(
        cpht_a,
        "_generate_support_for_seal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("run regenerated support instead of consuming its manifest")
        ),
    )
    arrays = cpht_a.compute_procedure(data, design, budget=2, draw=3)
    audit = cpht_a.audit_procedure(arrays, data, design, budget=2, draw=3)
    assert audit["outer_adapter_decisions"] == 15
    assert audit["finite_outer_optimizer_fits"] == 0
    assert audit["declined_exact_native"] == 15
    assert np.array_equal(arrays["adapted_by_seed"], data.native)
    assert np.array_equal(arrays["adapted_ensemble"], data.native.mean(axis=0))
    for fold in range(5):
        support = arrays["support_indices"][fold]
        assert np.all(data.folds[support] != fold)


def test_label_blind_feature_validator_rejects_outcome_column() -> None:
    rows: list[dict[str, object]] = []
    for seed in cpht_a.SEEDS:
        for patient in range(40):
            row: dict[str, object] = {
                "seed": seed,
                "patient_id": "ORION:C33" if patient == 33 else f"ORION:C{patient:02d}",
                "logit": float(patient),
                "n_slides": 2 if patient == 33 else 1,
                "exclude_neoadjuvant": False,
                "exclude_ambiguous_crc15": patient == 15,
            }
            row.update({f"e{index}": float(index) for index in range(512)})
            rows.append(row)
    frame = pd.DataFrame(rows)
    checked = cpht_a._validate_label_blind_features(frame)  # noqa: SLF001
    assert len(checked) == 120
    with pytest.raises(cpht_a.ContractError, match="leaks outcomes"):
        cpht_a._validate_label_blind_features(frame.assign(label=0))  # noqa: SLF001


def test_realized_design_roundtrip_and_resumable_parquet_publication(
    tmp_path: Path,
) -> None:
    data = _toy_data()
    design = _toy_design(data)
    cpht_a._publish_csv(cpht_a.fold_manifest_path(tmp_path), design.folds)  # noqa: SLF001
    cpht_a._publish_csv(  # noqa: SLF001
        cpht_a.support_manifest_path(tmp_path), design.supports
    )
    loaded = cpht_a._load_realized_design(tmp_path)  # noqa: SLF001
    assert len(loaded.folds) == 40
    assert len(loaded.supports) == 14_000
    assert (
        loaded.supports[["budget_per_class", "procedure_draw", "outer_fold"]]
        .drop_duplicates()
        .shape[0]
        == 1_500
    )

    artifact = tmp_path / "partial_report.parquet"
    frame = pd.DataFrame({"value": [1, 2, 3]})
    cpht_a._publish_parquet(artifact, frame)  # noqa: SLF001
    cpht_a._publish_parquet(artifact, frame)  # noqa: SLF001
    with pytest.raises(FileExistsError, match="non-identical"):
        cpht_a._publish_parquet(artifact, pd.DataFrame({"value": [9]}))  # noqa: SLF001


def test_paired_bootstrap_targets_mean_of_procedure_aurocs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _toy_data()
    rng = np.random.default_rng(19)
    adapted = rng.normal(size=(100, 40))
    native = data.native.mean(axis=0)
    monkeypatch.setattr(cpht_a, "N_BOOTSTRAP", 25)
    patient_indices, procedure_indices = cpht_a._bootstrap_layout(data.labels)  # noqa: SLF001
    adapted_boot, native_boot, delta_boot = cpht_a._bootstrap_expected_auc(  # noqa: SLF001
        data.labels,
        adapted,
        native,
        patient_indices,
        procedure_indices,
    )
    assert adapted_boot.shape == native_boot.shape == delta_boot.shape == (25,)
    assert np.allclose(delta_boot, adapted_boot - native_boot)
    point = float(cpht_a._auroc_rows(data.labels, adapted).mean())  # noqa: SLF001
    assert point == pytest.approx(
        np.mean([cpht_a._auroc_rows(data.labels, row)[0] for row in adapted])  # noqa: SLF001
    )

    # AUROC is nonlinear: this explicit construction proves the governed point
    # is not silently replaced by the AUROC of a 100-head prediction ensemble.
    small_labels = np.array([0, 0, 0, 1, 1, 1])
    two_procedures = np.array(
        [
            [1.8016, 1.3151, 0.3574, -1.2083, -0.0045, 0.6565],
            [-1.2884, 0.3951, 0.4299, 0.6960, -1.1841, -0.6617],
        ]
    )
    expected_procedure_auc = float(
        cpht_a._auroc_rows(small_labels, two_procedures).mean()  # noqa: SLF001
    )
    averaged_prediction_auc = float(
        cpht_a._auroc_rows(small_labels, two_procedures.mean(axis=0))[0]  # noqa: SLF001
    )
    assert expected_procedure_auc == pytest.approx(1 / 3)
    assert averaged_prediction_auc == 0.0


def test_compact_results_csv_uses_exact_float_roundtrip(tmp_path: Path) -> None:
    results = {
        "budgets": {
            "2": {
                "native_auroc": 0.7893333333333333,
                "expected_adapted_auroc": 0.778,
                "adapted_minus_native_auroc": -0.011333333333333306,
                "adapted_minus_native_auroc_ci95": [
                    -0.03442765944693073,
                    0.013875270562770598,
                ],
                "lambda_fraction_infinity": 0.6433333333333333,
            },
            "4": {
                "native_auroc": 0.7893333333333333,
                "expected_adapted_auroc": 0.7746933333333333,
                "adapted_minus_native_auroc": -0.014639999999999986,
                "adapted_minus_native_auroc_ci95": [
                    -0.040921355498721376,
                    0.013023761904761762,
                ],
                "lambda_fraction_infinity": 0.57,
            },
            "8": {
                "native_auroc": 0.7893333333333333,
                "expected_adapted_auroc": 0.7664266666666667,
                "adapted_minus_native_auroc": -0.02290666666666663,
                "adapted_minus_native_auroc_ci95": [
                    -0.0614422061128527,
                    0.01579955000505617,
                ],
                "lambda_fraction_infinity": 0.2673333333333333,
            },
        }
    }
    expected = cpht_a._compact_results(results)  # noqa: SLF001
    destination = tmp_path / "results.csv"
    cpht_a._publish_csv(destination, expected)  # noqa: SLF001

    assert destination.read_bytes() == cpht_a._csv_payload(expected)  # noqa: SLF001
    observed = pd.read_csv(destination, float_precision="round_trip")
    assert observed.equals(expected)


def test_component_name_routes_fresh_governed_replay(tmp_path: Path) -> None:
    try:
        cpht_a._activate_component_name("cpht_a_v2")  # noqa: SLF001
        assert cpht_a.component_root(tmp_path) == tmp_path / "cpht_a_v2"
        with pytest.raises(ValueError, match="component-name"):
            cpht_a._activate_component_name("../escape")  # noqa: SLF001
    finally:
        cpht_a._activate_component_name(cpht_a.DEFAULT_COMPONENT_NAME)  # noqa: SLF001


def test_cli_exposes_only_governed_commands() -> None:
    parser = cpht_a.build_parser()
    for command in ("preflight", "seal", "run", "report", "verify"):
        parsed = parser.parse_args([command, "--output-root", "/tmp/example"])
        assert parsed.command == command
        assert parsed.component_name == cpht_a.DEFAULT_COMPONENT_NAME
