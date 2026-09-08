from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_cross_protocol_transfer as cpht


def _master() -> pd.DataFrame:
    rows: list[dict] = []
    treatment = {5, 12, 18, 23, 28, 32}
    mutant = set(range(1, 16))
    for patient in range(1, 41):
        slide_names = [f"CRC{patient:02d}"]
        if patient == 33:
            slide_names = ["CRC33_01", "CRC33_02"]
        for slide in slide_names:
            rows.append(
                {
                    "cohort": "Orion",
                    "subcohort": "Orion-CRC",
                    "specimen_role": "primary",
                    "output_id": slide,
                    "patient_uid": f"ORION:C{patient}",
                    "include": "yes",
                    "available": "yes",
                    "used_kras": "yes",
                    "qc_slides": "pass",
                    "qc_flags": "treatment"
                    if patient in treatment
                    else ("specimen" if patient == 15 else np.nan),
                    "mpp": 0.325,
                    "mpp_source": "ome_xml",
                    "kras": "mutant" if patient in mutant else "wild_type",
                }
            )
    return pd.DataFrame(rows)


def _pack_index() -> pd.DataFrame:
    slides = _master()["output_id"].tolist()
    return pd.DataFrame(
        {
            "slide_id": slides,
            "offset": np.arange(len(slides)) * 100,
            "n_patches": np.arange(len(slides)) + 5_000,
        }
    )


def _manifest() -> pd.DataFrame:
    return cpht.build_label_blind_manifest(_master(), _pack_index())


def _labels() -> pd.DataFrame:
    master = _master()
    return (
        master.assign(label=master["kras"].eq("mutant").astype(int))
        .drop_duplicates("patient_uid")[["patient_uid", "label"]]
        .rename(columns={"patient_uid": "patient_id"})
        .sort_values("patient_id", kind="stable")
        .reset_index(drop=True)
    )


def _scores(n_models: int, separation: float) -> pd.DataFrame:
    master = _master()
    rows = []
    for model in range(n_models):
        for _, slide in master.iterrows():
            label = int(slide["kras"] == "mutant")
            # A deterministic slide effect makes the C33 aggregation observable.
            slide_effect = 0.2 if str(slide["output_id"]).endswith("_02") else 0.0
            rows.append(
                {
                    "slide_id": slide["output_id"],
                    "seed": 42 + (model // 5 if n_models == 15 else model),
                    "fold": model % 5 if n_models == 15 else 0,
                    "logit": separation * label + 0.01 * model + slide_effect,
                }
            )
    return pd.DataFrame(rows)


def _contract() -> dict:
    return {
        "family_loco": {
            "models": {
                target: {"calibrator": {"a": -0.2, "b": 0.3}}
                for target in cpht.FAMILY_LOCO_TARGETS
            }
        }
    }


def test_label_blind_manifest_has_governed_census_and_no_molecular_columns() -> None:
    manifest = _manifest()

    assert len(manifest) == 41
    assert manifest["patient_id"].nunique() == 40
    assert not (cpht.FORBIDDEN_PREOUTCOME_COLUMNS & set(manifest.columns))
    assert manifest.loc[manifest["exclude_neoadjuvant"], "patient_id"].nunique() == 6
    assert manifest.loc[manifest["exclude_ambiguous_crc15"], "patient_id"].tolist() == [
        "ORION:C15"
    ]
    assert set(manifest.loc[manifest["patient_id"].eq("ORION:C33"), "slide_id"]) == {
        "CRC33_01",
        "CRC33_02",
    }
    assert manifest["patch_count"].min() == 5_000


def test_label_blind_manifest_rejects_missing_orion_slide_from_pack() -> None:
    with pytest.raises(cpht.ContractError, match="Packed store lacks Orion"):
        cpht.build_label_blind_manifest(_master(), _pack_index().iloc[:-1])


def test_patient_aggregation_averages_models_then_c33_slides_in_native_logit_space() -> None:
    scores = _scores(3, separation=1.0)
    patients = cpht.aggregate_patient_logits(scores, _manifest(), expected_models=3)

    c1 = patients.loc[patients["patient_id"].eq("ORION:C1")].iloc[0]
    c33 = patients.loc[patients["patient_id"].eq("ORION:C33")].iloc[0]
    assert c1["mean_logit"] == pytest.approx(np.mean([1.0, 1.01, 1.02]))
    assert c33["mean_logit"] == pytest.approx(np.mean([0.01, 0.11, 0.21]))
    assert c33["n_slides"] == 2


def test_stratified_bootstrap_is_deterministic_and_preserves_both_class_counts() -> None:
    labels = np.array([0] * 25 + [1] * 15)
    first = cpht.stratified_bootstrap_indices(labels, n_bootstrap=50, seed=19)
    second = cpht.stratified_bootstrap_indices(labels, n_bootstrap=50, seed=19)

    assert np.array_equal(first, second)
    assert first.shape == (50, 40)
    drawn = labels[first]
    assert np.all(drawn[:, :25] == 0)
    assert np.all(drawn[:, 25:] == 1)
    assert np.all(drawn.sum(axis=1) == 15)


def test_vectorized_bootstrap_matches_sklearn_with_resampling_ties() -> None:
    labels = np.array([0] * 8 + [1] * 6)
    scores = np.array([-2.0, -1.7, -1.1, -0.8, -0.2, 0.1, 0.5, 0.9,
                       -0.4, 0.0, 0.3, 0.7, 1.1, 1.8])
    indices = cpht.stratified_bootstrap_indices(labels, n_bootstrap=100, seed=23)
    samples = cpht.bootstrap_metric_samples(labels, scores, indices)

    expected_auc = np.array(
        [roc_auc_score(labels[index], scores[index]) for index in indices]
    )
    expected_ap = np.array(
        [average_precision_score(labels[index], scores[index]) for index in indices]
    )
    assert samples["auroc"] == pytest.approx(expected_auc)
    assert samples["auprc"] == pytest.approx(expected_ap)


def test_analysis_keeps_rank_only_aim1_and_reports_partial_loco_matrix() -> None:
    model_scores = {"aim1_outer15": _scores(15, separation=0.8)}
    for index, target in enumerate(cpht.FAMILY_LOCO_TARGETS):
        model_scores[f"loco_heldout_{target.lower()}"] = _scores(
            3, separation=0.5 + index * 0.1
        )

    patients, results = cpht.analyse_scores(
        model_scores, _manifest(), _labels(), _contract(), n_bootstrap=100
    )

    assert len(patients) == 200
    assert results["confirmatory_cpht_completed"] is False
    assert results["family_loco_matrix"]["status"] == "four-family partial"
    assert results["family_loco_matrix"]["complete_eight_model_matrix"] is False
    all_metrics = results["populations"]["all_40"]["metrics"]
    assert "source_calibrated" not in all_metrics["aim1_outer15"]
    assert all_metrics["aim1_outer15"]["probability_metrics"].startswith("not inferred")
    for target in cpht.FAMILY_LOCO_TARGETS:
        assert "source_calibrated" in all_metrics[f"loco_heldout_{target.lower()}"]
    assert results["populations"]["exclude_neoadjuvant"]["n"] == 34
    assert results["populations"]["exclude_ambiguous_crc15"]["n"] == 39
    assert len(results["populations"]["all_40"]["paired_auroc_contrasts"]) == 10


def test_cli_exposes_final_v8_command_boundary() -> None:
    parser = cpht.build_parser()
    for command in ("preflight", "manifest", "score", "report"):
        args = parser.parse_args([command])
        assert args.command == command


@pytest.mark.skipif(not cpht.LOCO_ROOT.is_dir(), reason="governed LOCO lineage unavailable")
def test_live_loco_contract_is_canonical_json_serializable() -> None:
    models = cpht._resolve_loco_models()  # noqa: SLF001

    for target in cpht.FAMILY_LOCO_TARGETS:
        assert set(models[target]) == {*(str(seed) for seed in cpht.SEEDS), "calibrator"}
    serialized = cpht._canonical_json({"family_loco": {"models": models}})  # noqa: SLF001
    assert '"calibrator"' in serialized
