from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v3_aim1_worklist as worklist


def test_capacity_never_exceeds_nominal_fraction() -> None:
    assert worklist.selected_n(94, 0.30) == 28
    assert worklist.selected_n(153, 0.10) == 15
    assert worklist.selected_n(3, 0.10) == 1
    assert worklist.selected_n(94, 0.30) / 94 <= 0.30


def test_cutoff_ties_are_allocated_fractionally() -> None:
    y = np.array([1, 0, 1, 0])
    score = np.array([2.0, 1.0, 1.0, 0.0])

    observed = worklist._weighted_rank_metrics(
        y, score, np.ones(4), capacities=(0.50,)
    )[0]

    # One positive is strictly above the cutoff.  The remaining slot splits
    # equally across a positive and negative tied at the cutoff: 1.5 expected
    # positives in two slots.
    assert observed[0] == pytest.approx(0.75)  # capture
    assert observed[1] == pytest.approx(0.75)  # worklist PPV
    assert observed[2] == pytest.approx(1.50)  # enrichment over 50% prevalence
    assert observed[3] == pytest.approx(2.0 / 1.5)


def test_random_reference_is_analytic_diagonal_and_deterministic() -> None:
    y = np.array([0, 1] * 10)
    point_a, boot_a = worklist.random_point_and_bootstrap(y, n_bootstrap=100, seed=7)
    point_b, boot_b = worklist.random_point_and_bootstrap(y, n_bootstrap=100, seed=7)

    realized = np.array(
        [worklist.selected_n(len(y), q) / len(y) for q in worklist.CAPACITIES]
    )
    assert np.allclose(point_a[:, 0], realized)
    assert np.allclose(point_a[:, 2], 1.0)
    assert np.array_equal(point_a, point_b)
    assert np.array_equal(boot_a, boot_b)


def test_patient_bootstrap_is_deterministic_and_preserves_shape() -> None:
    y = np.array([0, 1] * 25)
    score = np.linspace(-2, 2, len(y))

    first = worklist.bootstrap_rank(y, score, n_bootstrap=100, seed=19)
    second = worklist.bootstrap_rank(y, score, n_bootstrap=100, seed=19)

    assert first.shape == (100, len(worklist.CAPACITIES), len(worklist.METRICS))
    assert np.allclose(first, second, equal_nan=True)
    assert np.nanmin(first[:, :, 0]) >= 0
    assert np.nanmax(first[:, :, 0]) <= 1


def test_append_only_writer_refuses_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "already_exists"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="append-only"):
        worklist.write_once(destination, {}, [])


def test_final_v2_documents_still_match_their_sealed_receipt() -> None:
    receipt = json.loads((worklist.FINAL_V2 / "report_bundle_receipt.json").read_text())

    for filename, expected in receipt["documents"].items():
        path = worklist.FINAL_V2 / filename
        assert path.stat().st_size == expected["size_bytes"]
        assert worklist.sha256_file(path) == expected["sha256"]


def test_frozen_e0_loader_reproduces_declared_population_and_metrics() -> None:
    frame, checks, _ = worklist.load_e0()

    assert (len(frame), int(frame["label"].sum())) == (1486, 604)
    assert checks
    assert all(check["pass"] for check in checks.values())
    assert np.isfinite(
        frame[
            [
                "wsi_seed42",
                "wsi_seed43",
                "wsi_seed44",
                "clinical_oof",
                "fusion_seed42",
                "fusion_seed43",
                "fusion_seed44",
            ]
        ].to_numpy()
    ).all()


def test_e2a_target_loader_is_label_blind_for_model_fitting() -> None:
    frame, check, _ = worklist.load_e2a_target("CPTAC")

    assert check["pass"]
    assert set(frame["cohort"]) == {"CPTAC"}
    assert np.isfinite(
        frame[["mean_logit", "clinical_source_only", "fusion_source_only"]].to_numpy()
    ).all()
