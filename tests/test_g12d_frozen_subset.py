"""Contract tests for frozen G12D Stage-B evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oceanpath.eval.g12d_frozen_subset import (
    G12DFrozenSubsetValidationError,
    evaluate_g12d_frozen_subset,
)
from oceanpath.eval.pooled_loco import RotationRecord

COHORTS = ("SurGen", "TCGA", "RIH")
CONTEXT_SCORES = {
    "G12D": (0.4, 1.2, 2.0),
    "other_KRAS_mutant": (1.3, 0.1, -0.4),
    "KRAS_wild_type": (-2.2, -1.8, -1.4, -1.0, -0.6, -0.2),
}


def _sigmoid(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exponent = np.exp(value)
    return float(exponent / (1.0 + exponent))


def _study(
    tmp_path: Path,
    *,
    seeds: tuple[str, ...] = ("seed_1", "seed_2"),
    extreme_logits: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[RotationRecord], dict[tuple[str, str], float]]:
    rows: list[dict[str, object]] = []
    base_logits: dict[str, float] = {}
    for cohort in COHORTS:
        for context, scores in CONTEXT_SCORES.items():
            label = int(context == "G12D")
            for patient_index, score in enumerate(scores):
                patient_id = f"{cohort}_{context}_{patient_index}"
                n_slides = 2 if context == "G12D" and patient_index == 0 else 1
                for slide_index in range(n_slides):
                    slide_id = f"{patient_id}_s{slide_index}"
                    rows.append(
                        {
                            "slide_id": slide_id,
                            "patient_id": patient_id,
                            "cohort": cohort,
                            "target_label": label,
                            "kras_context": context,
                        }
                    )
                    base_logits[slide_id] = score + 0.25 * slide_index
    stage_a = pd.DataFrame(rows)
    stage_c = stage_a.loc[
        stage_a["kras_context"].isin(("G12D", "other_KRAS_mutant")),
        ["slide_id", "patient_id", "cohort", "target_label"],
    ].reset_index(drop=True)

    records: list[RotationRecord] = []
    raw_logits: dict[tuple[str, str], float] = {}
    for seed_index, seed in enumerate(seeds):
        for cohort in COHORTS:
            selected = stage_a.loc[stage_a["cohort"] == cohort].copy()
            predictions = selected[["slide_id", "patient_id", "cohort"]].copy()
            predictions["label"] = selected["target_label"].to_numpy()
            logits = []
            probabilities = []
            for slide_id in selected["slide_id"]:
                logit = base_logits[str(slide_id)] + (-0.15 if seed_index == 0 else 0.25)
                if extreme_logits and slide_id == "SurGen_KRAS_wild_type_0_s0":
                    logit = -120.0 if seed_index == 0 else 100.0
                raw_logits[(seed, str(slide_id))] = logit
                logits.append(logit)
                probability = _sigmoid(logit)
                # One rounded score remains within the evaluator tolerance but
                # cannot reproduce the exact stored model logit.
                if slide_id == "SurGen_G12D_0_s0":
                    probability = round(probability, 7)
                probabilities.append(probability)
            predictions["logit"] = logits
            predictions["prob_1"] = probabilities
            path = tmp_path / "runs" / seed / cohort / "preds_test.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_parquet(path, index=False)
            records.append(
                RotationRecord(
                    heldout_cohort=cohort,
                    prediction_path=path,
                    seed=seed,
                    fold=0,
                    training_fingerprint=f"fingerprint-{seed}-{cohort}",
                )
            )
    return stage_a, stage_c, records, raw_logits


def _record(records: list[RotationRecord], *, seed: str, cohort: str) -> RotationRecord:
    return next(
        record
        for record in records
        if str(record.seed) == seed and record.heldout_cohort == cohort
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evaluates_exact_frozen_subset_and_writes_atomic_artifacts(tmp_path: Path) -> None:
    stage_a, stage_c, records, raw_logits = _study(tmp_path, extreme_logits=True)
    output_dir = tmp_path / "evaluation"

    result = evaluate_g12d_frozen_subset(
        records,
        stage_a,
        stage_c,
        output_dir=output_dir,
        n_bootstrap=30,
        bootstrap_seed=73,
        n_matched=80,
        matched_seed=91,
    )

    assert len(result.stage_a_patient_predictions) == 36
    assert len(result.stage_b_patient_predictions) == 18
    assert set(result.stage_b_slide_predictions["slide_id"]) == set(stage_c["slide_id"])
    assert set(result.stage_b_patient_predictions["kras_context"]) == {
        "G12D",
        "other_KRAS_mutant",
    }
    for block in result.metrics["stage_b"]["per_cohort"].values():
        assert block["n_patients"] == 6
        assert block["n_positive"] == 3
        assert block["n_negative"] == 3
        assert block["auroc"]["n_bootstrap_valid"] == 30
    assert result.metrics["stage_b"]["pooled_cohort_stratified"]["n_patients"] == 18
    assert result.metrics["delta_context_auc_a_minus_b"]["pooled_cohort_stratified"][
        "value"
    ] > 0

    # Exact raw logits survive saturated probabilities and are averaged seed -> slide.
    slide_id = "SurGen_KRAS_wild_type_0_s0"
    expected_logit = np.mean([raw_logits[(seed, slide_id)] for seed in ("seed_1", "seed_2")])
    observed = result.stage_a_slide_predictions.set_index("slide_id").loc[slide_id]
    assert observed["mean_logit"] == pytest.approx(expected_logit, rel=0, abs=1e-12)
    assert observed["mean_logit"] == -10.0

    expected_names = {
        "stage_a_slide_predictions.parquet",
        "stage_a_patient_predictions.parquet",
        "stage_b_slide_predictions.parquet",
        "stage_b_patient_predictions.parquet",
        "metrics.json",
        "report.md",
        "evaluation_completion.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_names
    completion = json.loads((output_dir / "evaluation_completion.json").read_text())
    assert completion["status"] == "complete"
    for filename, evidence in completion["artifacts"].items():
        assert evidence["sha256"] == _sha256(output_dir / filename)
        assert evidence["size_bytes"] == (output_dir / filename).stat().st_size
    assert "| Pooled |" in (output_dir / "report.md").read_text()


def test_requires_full_stage_a_coverage_before_conditional_filter(tmp_path: Path) -> None:
    stage_a, stage_c, records, _ = _study(tmp_path)
    target = Path(_record(records, seed="seed_1", cohort="SurGen").prediction_path)
    predictions = pd.read_parquet(target)
    # Remove a wild-type slide that would not be used in Stage B. The evaluator
    # must still reject the incomplete frozen Stage-A prediction artifact.
    predictions = predictions.loc[
        predictions["slide_id"] != "SurGen_KRAS_wild_type_0_s0"
    ]
    predictions.to_parquet(target, index=False)

    with pytest.raises(
        G12DFrozenSubsetValidationError,
        match="full Stage-A held-out slide coverage mismatch",
    ):
        evaluate_g12d_frozen_subset(
            records, stage_a, stage_c, n_bootstrap=10, n_matched=10
        )


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing_logit", "missing columns.*logit"),
        ("nonfinite_logit", "logit.*nonfinite"),
        ("inconsistent_probability", "inconsistent with sigmoid"),
    ],
)
def test_rejects_invalid_raw_prediction_contract(
    tmp_path: Path, defect: str, message: str
) -> None:
    stage_a, stage_c, records, _ = _study(tmp_path)
    target = Path(_record(records, seed="seed_1", cohort="TCGA").prediction_path)
    predictions = pd.read_parquet(target)
    if defect == "missing_logit":
        predictions = predictions.drop(columns="logit")
    elif defect == "nonfinite_logit":
        predictions.loc[0, "logit"] = np.inf
    else:
        predictions.loc[0, "prob_1"] = 0.5
    predictions.to_parquet(target, index=False)

    with pytest.raises(G12DFrozenSubsetValidationError, match=message):
        evaluate_g12d_frozen_subset(
            records, stage_a, stage_c, n_bootstrap=10, n_matched=10
        )


@pytest.mark.parametrize("defect", ["missing", "extra", "metadata", "label"])
def test_requires_stage_c_to_be_exact_conditional_subset(
    tmp_path: Path, defect: str
) -> None:
    stage_a, stage_c, records, _ = _study(tmp_path)
    if defect == "missing":
        stage_c = stage_c.iloc[1:].copy()
    elif defect == "extra":
        wild_type = stage_a.loc[
            stage_a["kras_context"] == "KRAS_wild_type",
            ["slide_id", "patient_id", "cohort", "target_label"],
        ].iloc[[0]]
        stage_c = pd.concat([stage_c, wild_type], ignore_index=True)
    elif defect == "metadata":
        stage_c.loc[0, "patient_id"] = "wrong_patient"
    else:
        row = stage_c.index[-1]
        stage_c.loc[row, "target_label"] = 1 - int(stage_c.loc[row, "target_label"])

    with pytest.raises(
        G12DFrozenSubsetValidationError,
        match="not the exact conditional subset",
    ):
        evaluate_g12d_frozen_subset(
            records, stage_a, stage_c, n_bootstrap=10, n_matched=10
        )


def test_nested_bootstrap_and_matched_marginal_are_deterministic(tmp_path: Path) -> None:
    stage_a, stage_c, records, _ = _study(tmp_path)
    kwargs = {
        "n_bootstrap": 40,
        "bootstrap_seed": 111,
        "n_matched": 60,
        "matched_seed": 222,
    }

    first = evaluate_g12d_frozen_subset(records, stage_a, stage_c, **kwargs)
    second = evaluate_g12d_frozen_subset(records, stage_a, stage_c, **kwargs)

    assert (
        first.metrics["delta_context_auc_a_minus_b"]
        == second.metrics["delta_context_auc_a_minus_b"]
    )
    assert first.metrics["matched_marginal"] == second.metrics["matched_marginal"]
    assert first.metrics["matched_marginal"]["n_repeats"] == 60
    for block in first.metrics["matched_marginal"]["per_cohort"].values():
        # Marginal negatives are 3 other-KRAS + 6 WT; matching the three
        # conditional negatives therefore fixes a 1/2 context quota.
        assert block["negative_context_population"] == {
            "other_KRAS_mutant": 3,
            "KRAS_wild_type": 6,
        }
        assert block["negative_context_quota"] == {
            "other_KRAS_mutant": 1,
            "KRAS_wild_type": 2,
        }
        assert block["n_positive"] == 3
        assert block["n_negative"] == 3
        assert block["auroc"]["n_repeats_valid"] == 60


def test_default_matched_control_uses_preregistered_one_thousand_draws(
    tmp_path: Path,
) -> None:
    stage_a, stage_c, records, _ = _study(tmp_path, seeds=("seed_1",))

    result = evaluate_g12d_frozen_subset(
        records,
        stage_a,
        stage_c,
        n_bootstrap=5,
    )

    assert result.metrics["matched_marginal"]["n_repeats"] == 1000
    assert result.metrics["matched_marginal"]["pooled_cohort_stratified"]["auroc"][
        "n_repeats_valid"
    ] == 1000
