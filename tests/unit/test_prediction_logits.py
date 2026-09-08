from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from oceanpath.aim1.evaluate import to_patient_level
from oceanpath.training.lightning import MILTrainModule


def _accumulate(logits: torch.Tensor, probabilities: torch.Tensor) -> list[dict]:
    rows: list[dict] = []
    output = SimpleNamespace(logits=logits)
    batch = {
        "slide_ids": [f"slide-{index}" for index in range(len(probabilities))],
        "labels": torch.tensor([0, 1][: len(probabilities)]),
    }
    MILTrainModule._accumulate_predictions(  # noqa: SLF001
        SimpleNamespace(), rows, output, batch, probabilities
    )
    return rows


def test_single_logit_predictions_persist_exact_raw_scores() -> None:
    logits = torch.tensor([-120.0, 120.0])
    probabilities = torch.sigmoid(logits)

    rows = _accumulate(logits, probabilities)

    assert [row["logit"] for row in rows] == [-120.0, 120.0]
    assert rows[0]["prob_1"] == pytest.approx(float(probabilities[0]))
    assert rows[1]["prob_1"] == pytest.approx(float(probabilities[1]))


def test_two_column_binary_predictions_store_exact_log_odds() -> None:
    logits = torch.tensor([[100.0, 97.0], [-50.0, -42.0]])
    probabilities = torch.softmax(logits, dim=1)[:, 1]

    rows = _accumulate(logits, probabilities)

    assert [row["logit"] for row in rows] == [-3.0, 8.0]


def test_multiclass_predictions_persist_each_raw_logit() -> None:
    logits = torch.tensor([[1.0, 2.0, 4.0], [-3.0, 0.0, 2.0]])
    probabilities = torch.softmax(logits, dim=1)

    rows = _accumulate(logits, probabilities)

    assert np.asarray(
        [[row[f"logit_{index}"] for index in range(3)] for row in rows]
    ) == pytest.approx(logits.numpy())


def test_patient_aggregation_uses_native_logits_not_saturated_probabilities() -> None:
    predictions = pd.DataFrame(
        {
            "slide_id": ["a", "b"],
            "label": [1, 1],
            "logit": [80.0, 120.0],
            # Both probabilities saturate to exactly one in float64.  A
            # probability round-trip would therefore erase the distinction.
            "prob_1": [1.0, 1.0],
        }
    )
    manifest = pd.DataFrame(
        {"slide_id": ["a", "b"], "patient_id": ["p", "p"]}
    )

    patients = to_patient_level(predictions, manifest)

    assert patients.loc[0, "mean_logit"] == pytest.approx(100.0)


def test_patient_aggregation_refuses_missing_native_logits() -> None:
    predictions = pd.DataFrame(
        {"slide_id": ["a"], "label": [0], "prob_1": [0.25]}
    )
    manifest = pd.DataFrame({"slide_id": ["a"], "patient_id": ["p"]})

    with pytest.raises(ValueError, match="native 'logit'"):
        to_patient_level(predictions, manifest)
