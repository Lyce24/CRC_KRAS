from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "tools" / "final_v3_aim2_operational.py"
SPEC = importlib.util.spec_from_file_location("final_v3_aim2_operational", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
aim2op = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = aim2op
SPEC.loader.exec_module(aim2op)


def test_icc_oneway_perfect_and_average_measure() -> None:
    groups = [np.asarray([value, value], dtype=float) for value in (1, 2, 4, 8)]
    single, average, k_eff = aim2op.icc_oneway(groups)
    assert single == 1.0
    assert average == 1.0
    assert k_eff == 2.0


def test_symmetric_pairs_are_orientation_invariant() -> None:
    first = [np.asarray([1.0, 2.0]), np.asarray([4.0, 8.0])]
    second = [group[::-1] for group in first]
    x1, y1 = aim2op.symmetric_pair_arrays(first)
    x2, y2 = aim2op.symmetric_pair_arrays(second)
    assert np.isclose(aim2op.correlation(x1, y1), aim2op.correlation(x2, y2))
    assert np.isclose(
        aim2op.correlation(x1, y1, rank=True), aim2op.correlation(x2, y2, rank=True)
    )


def test_patient_table_aggregates_native_logits() -> None:
    slides = pd.DataFrame(
        {
            "patient_id": ["a", "a", "b"],
            "slide_id": ["a1", "a2", "b1"],
            "target_label": [1, 1, 0],
            "subcohort": ["X", "X", "X"],
            "native_logit": [1.0, 3.0, -2.0],
        }
    )
    patients = aim2op.patient_table(slides).set_index("patient_id")
    assert patients.loc["a", "mean_logit"] == 2.0
    assert patients.loc["a", "lowest_logit"] == 1.0
    assert patients.loc["a", "highest_logit"] == 3.0
    assert patients.loc["a", "n_slides"] == 2


def test_canonical_patient_mean_uses_slides_then_seeds() -> None:
    group = pd.DataFrame(
        {
            "native_logit": [0.0, 0.0],
            "logit_seed42": [1.0, 3.0],
            "logit_seed43": [2.0, 8.0],
            "logit_seed44": [4.0, 6.0],
        }
    )
    assert aim2op.canonical_patient_mean(group) == 4.0


def test_capacity_membership_uses_floor_and_deterministic_ties() -> None:
    ids = np.asarray(["c", "a", "b"])
    scores = np.asarray([1.0, 1.0, 0.0])
    assert aim2op.top_capacity_membership(ids, scores, 0.30) == {"a"}
    assert aim2op.top_capacity_membership(ids, scores, 0.67) == {"a", "c"}


def test_deployment_matrix_preserves_central_boundaries() -> None:
    matrix = aim2op.deployment_matrix().set_index("deployment_setting")
    metastatic = matrix.loc["Metastatic tissue"]
    assert "fixed deployment gate failed" in metastatic["ranking_evidence"]
    assert "decrement remains unresolved" in metastatic["claim_boundary"]
    assert "Only the tested" in metastatic["local_adaptation"]
    primary = matrix.loc["Primary tumour, held-out cohort/data source"]
    assert "does not validate a target threshold" in primary["probability_scale"]
    allele = matrix.loc["Exact KRAS codon/substitution"]
    assert allele["recommended_status"] == "Direct molecular identification is required."


def test_design_does_not_claim_technical_reproducibility() -> None:
    design = aim2op.design_markdown(1000, 1000, 1000)
    assert "between-slide sampling agreement" in design
    assert "not technical\nreproducibility" in design


def test_bh_adjust_is_monotone_in_ranked_p_values() -> None:
    adjusted = aim2op.bh_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])


def test_receipt_output_paths_exist_after_atomic_publish(tmp_path: Path) -> None:
    staging = tmp_path / ".analysis.staging"
    published = tmp_path / "analysis"
    staging.mkdir()
    artifact = staging / "results.json"
    artifact.write_text('{"status":"PASS"}\n')
    receipt = {
        "outputs": {
            artifact.name: aim2op.published_identity(artifact, published / artifact.name)
        }
    }
    (staging / "receipt.json").write_text(json.dumps(receipt))
    staging.rename(published)

    checked = aim2op.validate_published_receipt(published / "receipt.json")
    assert set(checked) == {"results.json"}
    assert Path(checked["results.json"]["path"]).is_file()
