from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v3_aim4_compressibility as compress  # noqa: E402


def test_patient_seed_logits_uses_native_equal_slide_mean() -> None:
    predictions = pd.DataFrame(
        {
            "slide_id": ["s1", "s2", "s3"],
            "label": [1, 1, 0],
            "logit": [2.0, 4.0, -1.0],
            # Deliberately incompatible probabilities: they must not be used.
            "prob_1": [0.01, 0.01, 0.99],
        }
    )
    manifest = pd.DataFrame(
        {
            "slide_id": ["s1", "s2", "s3"],
            "patient_id": ["p1", "p1", "p2"],
            "target_label": [1, 1, 0],
        }
    )

    observed = compress._patient_seed_logits(predictions, manifest).set_index("patient_id")

    assert observed.loc["p1", "patient_seed_logit"] == pytest.approx(3.0)
    assert observed.loc["p1", "n_slides"] == 2
    assert observed.loc["p2", "patient_seed_logit"] == pytest.approx(-1.0)


def test_patient_seed_logits_fails_closed_on_label_mismatch() -> None:
    predictions = pd.DataFrame({"slide_id": ["s1"], "label": [1], "logit": [2.0]})
    manifest = pd.DataFrame(
        {"slide_id": ["s1"], "patient_id": ["p1"], "target_label": [0]}
    )
    with pytest.raises(compress.CompressibilityError, match="labels differ"):
        compress._patient_seed_logits(predictions, manifest)


def test_prototype_features_selects_only_locked_numeric_ids_and_quantities() -> None:
    rows = []
    for patient in ("p1", "p2"):
        for prototype in range(32):
            rows.append(
                {
                    "arm": "e0",
                    "patient_id": patient,
                    "prototype": prototype,
                    "cohort": "CPTAC",
                    "label": int(patient == "p1"),
                    "abundance": prototype / 100,
                    "attn_mass_mean": (31 - prototype) / 100,
                    "n_attention_seeds": 3,
                }
            )
    observed = compress._prototype_features(pd.DataFrame(rows)).set_index("patient_id")

    assert list(observed.columns) == [
        "p17_abundance",
        "p28_abundance",
        "p28_attention_mass",
        "p5_attention_mass",
    ]
    assert observed.loc["p1", "p17_abundance"] == pytest.approx(0.17)
    assert observed.loc["p1", "p28_attention_mass"] == pytest.approx(0.03)


def _synthetic_frame(seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cohort_index, cohort in enumerate(compress.TARGETS):
        for patient in range(24):
            p17 = rng.uniform(0, 0.3)
            p28 = rng.uniform(0, 0.2)
            p28_attention = rng.uniform(0, 0.3)
            p5_attention = rng.uniform(0, 0.2)
            logit = (
                1.5 * p17
                + 2.5 * p28
                + 4.0 * p28_attention
                - 2.0 * p5_attention
                + rng.normal(0, 0.05)
            )
            rows.append(
                {
                    "patient_id": f"{cohort}-{patient}",
                    "cohort": cohort,
                    "label": patient % 2,
                    "target_label": patient % 2,
                    "native_ensemble_logit": logit,
                    "logit_seed42": logit - 0.1,
                    "logit_seed43": logit,
                    "logit_seed44": logit + 0.1,
                    "age_at_diagnosis": 50 + patient,
                    "sex": "female" if patient % 2 else "male",
                    "site_class": "Colon" if patient % 3 else "Rectum",
                    "stage_class": "I-II" if patient % 2 else "III-IV",
                    "p17_abundance": p17,
                    "p28_abundance": p28,
                    "p28_attention_mass": p28_attention,
                    "p5_attention_mass": p5_attention,
                    "cohort_index": cohort_index,
                }
            )
    return pd.DataFrame(rows)


def test_source_only_alpha_selection_is_deterministic() -> None:
    source = _synthetic_frame()
    source = source[source["cohort"] != "CPTAC"]

    alpha1, rows1 = compress.select_alpha_source_only(source, "abundance_plus_attention")
    alpha2, rows2 = compress.select_alpha_source_only(source, "abundance_plus_attention")

    assert alpha1 == alpha2
    assert rows1 == rows2
    assert alpha1 in compress.ALPHAS
    assert len(rows1) == len(compress.ALPHAS)
    assert all(len(row["fold_r2"]) == 3 for row in rows1)


def test_loco_predictions_cover_each_patient_once_and_do_not_train_on_kras() -> None:
    frame = _synthetic_frame()

    predictions, selections, contracts = compress.run_loco(frame)

    assert len(predictions) == len(frame)
    assert predictions["patient_id"].is_unique
    for model in compress.MODEL_FEATURES:
        assert np.isfinite(predictions[f"prediction_{model}"]).all()
        assert np.isfinite(predictions[f"residual_{model}"]).all()
        for target in compress.TARGETS:
            contract = contracts[model][target]
            assert target not in contract["outer_source_cohorts"]
            assert contract["outer_target_n"] == 24
    assert set(selections["selected"]) == {False, True}
    # Ridge is fit to the native logit; label is absent from every feature set.
    assert all("label" not in features for features in compress.MODEL_FEATURES.values())


def test_bootstrap_summary_is_deterministic_and_retains_negative_r2() -> None:
    frame = _synthetic_frame()
    predictions, _, _ = compress.run_loco(frame)
    # Force one held-out arm to be worse than its mean-only baseline.
    cptac = predictions["cohort"] == "CPTAC"
    predictions.loc[cptac, "prediction_clinical"] = 100.0

    result1, flat1 = compress.summarize_predictions(predictions, n_bootstrap=20, seed=123)
    result2, flat2 = compress.summarize_predictions(predictions, n_bootstrap=20, seed=123)

    assert result1 == result2
    pd.testing.assert_frame_equal(flat1, flat2)
    assert result1["models"]["clinical"]["targets"]["CPTAC"]["r2"]["point"] < 0
    increment = result1["comparisons"]["attention_increment_over_abundance"]["r2"]
    assert increment["ci95"][0] is not None
    assert increment["ci95"][1] is not None


def test_vectorized_bootstrap_metrics_match_scalar_estimand() -> None:
    truth = np.array([[0.1, 0.5, 0.5, 1.2], [-0.2, 0.3, 0.7, 1.1]])
    prediction = np.array([[0.2, 0.4, 0.4, 1.0], [0.0, 0.1, 0.8, 0.9]])
    label = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])

    vectorized = compress._metric_arrays(truth, prediction, label)

    for row in range(len(truth)):
        scalar = compress.metric_block(truth[row], prediction[row], label[row])
        for metric, value in scalar.items():
            assert vectorized[metric][row] == pytest.approx(value, abs=1e-12)


def test_bind_inputs_records_hashes_without_authoritative_override(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.parquet"
    manifest = tmp_path / "manifest.csv"
    prediction_root = tmp_path / "predictions"
    profiles.write_bytes(b"profiles")
    manifest.write_bytes(b"manifest")
    for seed in compress.SEEDS:
        path = prediction_root / f"seed{seed}/oof_predictions.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"seed-{seed}".encode())

    identities = compress.bind_inputs(
        profiles,
        manifest,
        prediction_root,
        enforce_authoritative_hashes=False,
    )

    assert set(identities) == {
        "patient_profiles",
        "manifest",
        "seed42_predictions",
        "seed43_predictions",
        "seed44_predictions",
    }
    assert all(len(identity["sha256"]) == 64 for identity in identities.values())


def test_staged_inventory_records_final_published_paths(tmp_path: Path) -> None:
    stage = tmp_path / ".staging"
    stage.mkdir()
    (stage / "result.json").write_text("{}\n", encoding="utf-8")
    final = tmp_path / "published"

    inventory = compress._output_inventory(stage, published_root=final)

    assert inventory[0]["relative_path"] == "result.json"
    assert inventory[0]["path"] == str(final / "result.json")
    assert inventory[0]["sha256"] == compress._sha256_file(stage / "result.json")
