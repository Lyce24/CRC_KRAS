from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

REPOSITORY = Path(__file__).resolve().parents[1]
ADDENDUM_NAME = "v5_pre_read_nuisance_addendum_20260821"


def _source(name: str) -> Path:
    final = REPOSITORY / "reviews" / ADDENDUM_NAME / name
    staging = REPOSITORY / "reviews" / f".{ADDENDUM_NAME}.building" / name
    return final if final.is_file() else staging


def _load(name: str, module_name: str) -> Any:
    specification = importlib.util.spec_from_file_location(module_name, _source(name))
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analyzer() -> Any:
    return _load("analyze_nuisance.py", "v5_nuisance_analyzer_test")


@pytest.fixture(scope="module")
def sealer() -> Any:
    return _load("seal_addendum.py", "v5_nuisance_sealer_test")


def _numeric_gradient(
    analyzer: Any, parameters: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    result = []
    for index, value in enumerate(parameters):
        step = 1e-6 * (1.0 + abs(float(value)))
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        upper_value = analyzer._ordinal_objective_and_gradient(upper, x, y, 4)[0]
        lower_value = analyzer._ordinal_objective_and_gradient(lower, x, y, 4)[0]
        result.append((upper_value - lower_value) / (2.0 * step))
    return np.asarray(result)


@pytest.mark.parametrize(
    "parameters,tolerance",
    [
        (np.array([0.3, -0.2, -1.0, 0.1, 0.2]), 1e-6),
        (np.array([8.0, -7.0, -10.0, -7.0, -6.0]), 2e-3),
    ],
)
def test_stable_ordinal_gradient_matches_finite_difference(
    analyzer: Any,
    parameters: np.ndarray,
    tolerance: float,
) -> None:
    rng = np.random.default_rng(41)
    matrix = rng.normal(size=(40, 2))
    outcome = np.tile(np.arange(4), 10)
    _, analytic = analyzer._ordinal_objective_and_gradient(parameters, matrix, outcome, 4)
    numeric = _numeric_gradient(analyzer, parameters, matrix, outcome)
    assert np.max(np.abs(analytic - numeric)) < tolerance


def test_missing_frozen_category_is_refused(analyzer: Any) -> None:
    frame = pd.DataFrame(
        {
            "x": np.linspace(-1, 1, 12),
            "mucin_score": np.tile([0, 2, 3], 4),
        }
    )
    with pytest.raises(analyzer.ModelFitError, match="every frozen mucin category"):
        analyzer.fit_proportional_odds(frame, ["x"])


def test_synthetic_positive_effect_has_correct_sign(analyzer: Any) -> None:
    rng = np.random.default_rng(20260821)
    x = rng.normal(size=600)
    eta = 1.1 * x
    thresholds = np.array([-1.1, 0.2, 1.2])
    cumulative = expit(thresholds[None, :] - eta[:, None])
    probabilities = np.column_stack(
        [
            cumulative[:, 0],
            cumulative[:, 1] - cumulative[:, 0],
            cumulative[:, 2] - cumulative[:, 1],
            1.0 - cumulative[:, 2],
        ]
    )
    uniform = rng.random(len(x))
    outcome = (uniform[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
    fit = analyzer.fit_proportional_odds(
        pd.DataFrame({"x": x, "mucin_score": outcome}),
        ["x"],
    )
    assert fit.coefficients["x"] > 0
    assert np.exp(fit.coefficients["x"]) > 1
    assert fit.minimum_predicted_probability > 0
    assert fit.maximum_probability_sum_error < 1e-10
    assert fit.information_minimum_eigenvalue > 0


def test_perfect_ordering_fails_closed(analyzer: Any) -> None:
    x = np.linspace(-3, 3, 80)
    outcome = np.repeat(np.arange(4), 20)
    with pytest.raises(analyzer.ModelFitError):
        analyzer.fit_proportional_odds(
            pd.DataFrame({"x": x, "mucin_score": outcome}),
            ["x"],
        )


def _dummy_fit(analyzer: Any, predictors: list[str]) -> Any:
    return analyzer.OrdinalFit(
        coefficients={name: 0.2 for name in predictors},
        thresholds=[-1.0, 0.0, 1.0],
        objective=10.0,
        iterations=4,
        max_abs_gradient=1e-8,
        category_counts={str(index): 4 for index in range(4)},
        minimum_predicted_probability=0.01,
        maximum_probability_sum_error=1e-16,
        information_minimum_eigenvalue=0.1,
        information_condition_number=10.0,
    )


def _model_frame() -> tuple[pd.DataFrame, list[str]]:
    blocks = [f"B{index}" for index in range(8)]
    groups = [("absent", 0.0), ("positive_low", 0.2), ("positive_high", 1.0)]
    rows = []
    case = 0
    for block in blocks:
        for group, p17 in groups:
            for replicate in range(2):
                rows.append(
                    {
                        "case_id": f"Q{case:05d}",
                        "analysis_block": block,
                        "sampling_cell": f"{block}|{group}",
                        "primary_complete": True,
                        "nonassessable": replicate,
                        "mucin_score": case % 4,
                        "p17_abundance": p17,
                        "p17_z": p17,
                        "log_positive_p17_z": np.log(p17) if p17 > 0 else np.nan,
                        "log_tissue_area_z": (case - 24) / 10,
                        "tissue_area_mm2": float(case + 2),
                        "area_rank_tertile": (
                            "smallest_20"
                            if case < 16
                            else "middle_20"
                            if case < 32
                            else "largest_20"
                        ),
                    }
                )
                case += 1
    return pd.DataFrame.from_records(rows), blocks


def test_positive_model_failure_is_independent(
    analyzer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, blocks = _model_frame()

    def fake_fit(data: pd.DataFrame, predictors: list[str], **_kwargs: Any) -> Any:
        if "log_positive_p17_z" in predictors:
            raise analyzer.ModelFitError("forced positive-only failure")
        return _dummy_fit(analyzer, predictors)

    monkeypatch.setattr(analyzer, "fit_proportional_odds", fake_fit)
    fits, failures = analyzer.fit_models_independently(
        frame,
        blocks,
        blocks[0],
        full_diagnostics=True,
    )
    assert set(fits) == {"all_case_unadjusted", "all_case_area_adjusted"}
    assert failures == {"positive_p17_area_adjusted": "forced positive-only failure"}


def test_bootstrap_failures_are_not_redrawn_or_shared(
    analyzer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, blocks = _model_frame()

    def fake_fit(data: pd.DataFrame, predictors: list[str], **_kwargs: Any) -> Any:
        if "log_positive_p17_z" in predictors:
            raise analyzer.ModelFitError("forced positive-only failure")
        return _dummy_fit(analyzer, predictors)

    monkeypatch.setattr(analyzer, "fit_proportional_odds", fake_fit)
    draws, summary, missingness = analyzer.stratified_bootstrap(
        frame,
        blocks,
        blocks[0],
        draws=3,
        seed=17,
        minimum_valid=2,
    )
    assert summary["models"]["all_case_unadjusted"]["valid_draws"] == 3
    assert summary["models"]["all_case_area_adjusted"]["valid_draws"] == 3
    assert summary["models"]["positive_p17_area_adjusted"]["valid_draws"] == 0
    assert summary["models"]["positive_p17_area_adjusted"]["status"] == "BOOTSTRAP_UNSTABLE"
    assert len(draws[draws["model"] == "all_case_unadjusted"]) == 3
    assert missingness["valid_draws"] == 3


def test_blank_v5_forms_are_refused(analyzer: Any) -> None:
    scores = REPOSITORY / "reviews/v5/FOR_PATHOLOGIST/scoring_form.csv"
    reviewer = REPOSITORY / "reviews/v5/FOR_PATHOLOGIST/reviewer_info.csv"
    case_ids = pd.read_csv(scores, dtype=str)["case_id"].tolist()
    with pytest.raises(analyzer.ValidationError, match="ANALYSIS REFUSED"):
        analyzer.validate_returned_forms(scores, reviewer, case_ids)


def test_receipt_validation_does_not_hash_key(
    analyzer: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write(name: str, content: str = "x\n") -> Path:
        path = tmp_path / name
        path.write_text(content)
        return path

    plan = write("plan.md")
    readme = write("readme.md")
    sealer_file = write("sealer.py")
    covariates = write("covariates.csv")
    constants = write("constants.json", "{}\n")
    test_file = write("test.py")
    key = write("key.csv", "secret,p17\n")
    development = write("development.csv")
    parent = write(
        "parent.json",
        json.dumps(
            {
                "status": "PASS",
                "problems": [],
                "scientific_status": "GENERATED_UNREAD",
                "analysis_executed": False,
                "unblinding_performed": False,
                "analysis_result": None,
            }
        ),
    )
    receipt = {
        "schema_version": 1,
        "addendum_id": ADDENDUM_NAME,
        "status": "PASS",
        "problems": [],
        "scientific_status": "GENERATED_UNREAD_SECONDARY_ADDENDUM",
        "analysis_executed": False,
        "unblinding_performed": False,
        "analysis_result": None,
        "confirmatory_role": "NONE_SECONDARY_ROBUSTNESS_ONLY",
        "analyzer": analyzer.file_identity(Path(analyzer.__file__)),
        "plan": analyzer.file_identity(plan),
        "readme": analyzer.file_identity(readme),
        "sealer": analyzer.file_identity(sealer_file),
        "covariate_manifest": analyzer.file_identity(covariates),
        "design_constants": analyzer.file_identity(constants),
        "tests": [analyzer.file_identity(test_file)],
        "v5_case_key": analyzer.file_identity(key),
        "parent_v5_receipt": analyzer.file_identity(parent),
        "frozen_inputs": [
            analyzer.file_identity(parent),
            analyzer.file_identity(key),
            analyzer.file_identity(development),
        ],
    }
    receipt_path = tmp_path / "ADDENDUM_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt))
    original = analyzer.sha256
    hashed: list[Path] = []

    def spy(path: Path) -> str:
        hashed.append(path.resolve())
        return original(path)

    monkeypatch.setattr(analyzer, "sha256", spy)
    analyzer.verify_receipts(receipt_path)
    assert key.resolve() not in hashed


def test_covariate_derivation_is_exact_and_patient_slide_joined(sealer: Any) -> None:
    covariates, constants = sealer.derive_covariates(
        REPOSITORY / "reviews/v5/KEYS_DO_NOT_DISTRIBUTE/case_key.csv",
        Path("/mnt/d/YC.Liu/manifests/colon/aim1_dev.csv"),
    )
    assert len(covariates) == 60
    assert not covariates.duplicated(["patient_id", "slide_id"]).any()
    assert covariates["area_rank_tertile"].value_counts().to_dict() == {
        "smallest_20": 20,
        "middle_20": 20,
        "largest_20": 20,
    }
    assert len(constants["analysis_blocks"]) == 8
    assert len(constants["sampling_cell_counts"]) == 24
    assert constants["bootstrap"]["draws"] == 2000


def test_sealer_external_test_mutation_leaves_no_pass_receipt(
    sealer: Any,
    tmp_path: Path,
) -> None:
    staging = tmp_path / "stage"
    final = tmp_path / "final"
    staging.mkdir()
    payload = staging / "payload.txt"
    payload.write_text("payload\n")
    external_test = tmp_path / "external_test.py"
    external_test.write_text("original\n")
    test_record = sealer.identity(external_test)
    external_test.write_text("mutated\n")
    receipt = {
        "outputs": [sealer.identity(payload, declared_path=final / payload.name)],
        "frozen_inputs": [],
        "tests": [test_record],
    }
    with pytest.raises(RuntimeError, match="post-rename"):
        sealer.publish_receipt_last(staging, final, receipt, {"payload.txt"})
    assert final.is_dir()
    assert (final / "DO_NOT_RELEASE_ADDENDUM_FAILED.txt").is_file()
    assert not (final / "ADDENDUM_RECEIPT.json").exists()


def test_analysis_receipt_failure_marks_final(
    analyzer: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "analysis_stage"
    final = tmp_path / "analysis_final"
    staging.mkdir()
    records = []
    for name in sorted(analyzer.ANALYSIS_OUTPUT_NAMES):
        result = staging / name
        result.write_text("{}\n")
        record = analyzer.file_identity(result)
        record["path"] = str(final / result.name)
        records.append(record)
    captured_file = tmp_path / "input.csv"
    captured_file.write_text("x\n")

    def fail_receipt(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("forced receipt failure")

    monkeypatch.setattr(analyzer, "atomic_write_json", fail_receipt)
    with pytest.raises(OSError, match="forced receipt failure"):
        analyzer.publish_analysis_receipt_last(
            staging,
            final,
            records,
            {"status": "PASS"},
            {"input": analyzer.file_identity(captured_file)},
        )
    assert (final / "DO_NOT_USE_FAILED.txt").is_file()
    assert not (final / "analysis_receipt.json").exists()


def test_analysis_extra_file_is_refused_before_receipt(
    analyzer: Any,
    tmp_path: Path,
) -> None:
    staging = tmp_path / "analysis_stage"
    final = tmp_path / "analysis_final"
    staging.mkdir()
    records = []
    for name in sorted(analyzer.ANALYSIS_OUTPUT_NAMES):
        path = staging / name
        path.write_text("{}\n")
        record = analyzer.file_identity(path)
        record["path"] = str(final / name)
        records.append(record)
    (staging / "injected.txt").write_text("unexpected\n")
    with pytest.raises(analyzer.ValidationError, match="file set differs"):
        analyzer.publish_analysis_receipt_last(staging, final, records, {}, {})
    assert (final / "DO_NOT_USE_FAILED.txt").is_file()
    assert not (final / "analysis_receipt.json").exists()


def test_sealer_atomic_receipt_failure_marks_final(
    sealer: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "seal_stage"
    final = tmp_path / "seal_final"
    staging.mkdir()
    payload = staging / "payload.txt"
    payload.write_text("payload\n")
    receipt = {
        "outputs": [sealer.identity(payload, declared_path=final / "payload.txt")],
        "frozen_inputs": [],
        "tests": [],
    }

    def fail_receipt(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("forced addendum receipt failure")

    monkeypatch.setattr(sealer, "atomic_receipt", fail_receipt)
    with pytest.raises(OSError, match="forced addendum receipt failure"):
        sealer.publish_receipt_last(staging, final, receipt, {"payload.txt"})
    assert (final / "DO_NOT_RELEASE_ADDENDUM_FAILED.txt").is_file()
    assert not (final / "ADDENDUM_RECEIPT.json").exists()


def test_addendum_receipt_swap_between_capture_and_parse_is_refused(
    analyzer: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"version": 1}\n')

    def swap_after_parse(_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        receipt.write_text('{"version": 2}\n')
        return {}, {}

    monkeypatch.setattr(analyzer, "verify_receipts", swap_after_parse)
    with pytest.raises(analyzer.ValidationError, match="after parsing"):
        analyzer.verify_addendum_receipt_stably(receipt)


def test_attenuation_and_reversal_labels_are_nonconfirmatory(analyzer: Any) -> None:
    attenuated = analyzer.robustness_interpretation(1.0, 0.4)
    reversed_result = analyzer.robustness_interpretation(1.0, -0.1)
    assert attenuated["status"] == "ROBUSTNESS_CONCERN_GE50_PERCENT_ATTENUATION"
    assert reversed_result["status"] == "ROBUSTNESS_CONCERN_DIRECTION_REVERSAL"
    assert attenuated["confirmatory_gate"] is False
    assert "non-collapsible" in attenuated["interpretation"]
