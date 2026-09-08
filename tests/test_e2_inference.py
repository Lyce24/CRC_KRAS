from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_loco_transport
import aim2_metastatic_transport
import aim2_setd_role_contrast
from oceanpath.aim1 import lineage, paths


def _arm(
    name: str,
    *,
    n_per_class: int,
    separation: float,
    noise: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = np.repeat([0, 1], n_per_class)
    score = separation * label + rng.normal(0, noise, len(label))
    return pd.DataFrame(
        {
            "patient_id": [f"{name}-{index}" for index in range(len(label))],
            "label": label,
            "mean_logit": score,
            "prob_raw": 1 / (1 + np.exp(-score)),
            "msi_dmmr": "MSS/pMMR",
            "braf": "wild_type",
        }
    )


def test_combined_decrement_uses_patient_uncertainty_and_detects_large_drop() -> None:
    arms = {
        "A": (
            _arm("ap", n_per_class=90, separation=2.0, noise=0.7, seed=1),
            _arm("am", n_per_class=90, separation=0.2, noise=1.2, seed=2),
        ),
        "B": (
            _arm("bp", n_per_class=90, separation=1.8, noise=0.7, seed=3),
            _arm("bm", n_per_class=90, separation=0.1, noise=1.2, seed=4),
        ),
    }

    result = aim2_metastatic_transport.combined_decrement_ci(arms, n_boot=500, seed=19)

    assert result["delta_auroc"] < 0
    assert result["delta_auroc_ci"][1] < 0
    assert result["evidence_of_overall_decrement"] is True
    assert result["n_bootstrap"] == 500
    assert "seed" not in result["estimand"].lower()


def test_combined_decrement_is_deterministic_and_does_not_claim_a_null_drop() -> None:
    arms = {}
    for cohort, seed in (("A", 10), ("B", 11)):
        primary = _arm(
            f"{cohort}p", n_per_class=100, separation=1.0, noise=1.0, seed=seed
        )
        metastatic = primary.copy()
        metastatic["patient_id"] = metastatic["patient_id"].str.replace("p-", "m-")
        arms[cohort] = (primary, metastatic)

    first = aim2_metastatic_transport.combined_decrement_ci(arms, n_boot=300, seed=23)
    second = aim2_metastatic_transport.combined_decrement_ci(arms, n_boot=300, seed=23)

    assert first == second
    assert first["delta_auroc"] == pytest.approx(0.0)
    assert first["delta_auroc_ci"][0] < 0 < first["delta_auroc_ci"][1]
    assert first["evidence_of_overall_decrement"] is False


def test_classification_does_not_promote_concordant_point_signs() -> None:
    target = {
        "primary_vs_metastatic": {
            "primary_auroc": 0.70,
            "metastatic_auroc": 0.65,
            "delta_auroc": -0.05,
            "delta_auroc_ci": [-0.15, 0.05],
        },
        "primary_contrast_block": {"brier": 0.2},
        "metastatic_overall": {"brier": 0.21},
    }
    combined = {
        "inference": "overall decrement not established",
        "delta_auroc": -0.05,
        "delta_auroc_ci": [-0.11, 0.01],
    }

    result = aim2_metastatic_transport.classify(
        {"targets": {"RIH": target, "SurGen": target}, "combined_decrement": combined}
    )

    assert result["outcome"] == "overall decrement not established"
    assert result["descriptive_point_pattern"] == "RIH: down, SurGen: down"
    assert result["seed_results_inferential_role"].startswith("none")


def _nested_arms() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(31)
    label = np.repeat([0, 1], 200)
    in_d = np.tile(np.repeat([False, True], 100), 2)
    primary_score = 2.0 * label + rng.normal(0, 0.7, len(label))
    metastatic_score = np.where(
        in_d,
        -1.5 * label + rng.normal(0, 0.8, len(label)),
        2.0 * label + rng.normal(0, 0.7, len(label)),
    )

    def frame(role: str, score: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "patient_id": [f"{role}-{index}" for index in range(len(label))],
                "label": label,
                "mean_logit": score,
                "prob_raw": 1 / (1 + np.exp(-score)),
                "msi_dmmr": np.where(in_d, "MSS/pMMR", "MSI/dMMR"),
                "braf": "wild_type",
            }
        )

    return frame("p", primary_score), frame("m", metastatic_score)


def test_e2d3_directly_bootstraps_nested_change_in_gap() -> None:
    primary, metastatic = _nested_arms()

    result = aim2_setd_role_contrast.restriction_change_ci(primary, metastatic, n_boot=500, seed=37)
    expected = (
        aim2_loco_transport._auroc(aim2_setd_role_contrast.set_d(metastatic))
        - aim2_loco_transport._auroc(aim2_setd_role_contrast.set_d(primary))
        - (aim2_loco_transport._auroc(metastatic) - aim2_loco_transport._auroc(primary))
    )

    assert result["change_delta_auroc"] == pytest.approx(expected)
    assert result["change_delta_auroc_ci"][1] < 0
    assert result["ci_excludes_zero"] is True
    assert "same arm resample" in result["bootstrap_method"]


def test_patient_bootstrap_fails_closed_on_duplicate_patient_rows() -> None:
    primary = _arm("p", n_per_class=10, separation=1.0, noise=1.0, seed=40)
    metastatic = _arm("m", n_per_class=10, separation=0.5, noise=1.0, seed=41)
    primary.loc[1, "patient_id"] = primary.loc[0, "patient_id"]

    with pytest.raises(ValueError, match="not one-row-per-patient"):
        aim2_metastatic_transport.contrast_ci(primary, metastatic, n_boot=10)


def test_lineage_output_is_new_namespaced_and_exclusive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OCEANPATH_AIM2_LINEAGE", "aim2_cap8192_v2_test")
    assert lineage.eval_root() == (
        paths.OUTPUT_ROOT / "reruns" / "aim2_cap8192_v2_test" / "eval"
    )

    artifact = tmp_path / "new" / "result.json"
    lineage.write_json_once(artifact, {"value": 1})
    with pytest.raises(FileExistsError, match="Refusing to overwrite|File exists"):
        lineage.write_json_once(artifact, {"value": 2})
