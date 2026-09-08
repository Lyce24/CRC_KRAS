from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crc-v6-test-matplotlib")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import crc_final_v6_met_transfer_diagnostics as analysis  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def test_strict_json_rejects_duplicates_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}\n')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        analysis._strict_json_load(duplicate)

    for text in ('{"x": NaN}\n', '{"x": Infinity}\n', '{"x": 1e999}\n'):
        nonfinite = tmp_path / "nonfinite.json"
        nonfinite.write_text(text)
        with pytest.raises(ValueError, match="Non-finite"):
            analysis._strict_json_load(nonfinite)


def test_v6_is_exact_v5_extension_and_target_rosters_join() -> None:
    v6, validation = analysis.validate_v6()
    assert validation["v5_cells_preserved_exactly"] is True
    assert validation["v6_csv_xlsx_equal"] is True
    assert validation["rows"] == 2069
    assert validation["columns"] == 63

    metadata, _ = analysis.build_patient_metadata(v6)
    observed = {
        (cohort, role): (len(frame), int(frame["label"].sum()))
        for (cohort, role), frame in metadata.groupby(["cohort", "role"])
    }
    assert observed == {
        ("RIH", "metastatic"): (85, 37),
        ("RIH", "primary"): (153, 70),
        ("SR1482", "metastatic"): (74, 30),
        ("SR1482", "primary"): (324, 147),
    }


def test_governed_score_parent_chain_replays() -> None:
    result = analysis.validate_parent_score_receipts()
    assert result["score_artifacts_match_parent_receipts"] is True
    assert result["target_rosters_match_loco_contract"] is True
    assert result["final_v12_receipt_sha256"] == (
        "b5773af4fe5cb50269e391aaee04ec33f98f05548881c1aa042e4433d355fa38"
    )


def test_weighted_pairwise_auc_is_tie_correct() -> None:
    observed = analysis._weighted_pairwise_auc(
        np.array([0.8, 0.2]),
        np.array([0.2, 0.1]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
    )
    # Three wins plus one tie among four equally weighted pairs.
    assert observed == pytest.approx(0.875)


def test_fast_metric_bootstrap_matches_expanded_patient_resample() -> None:
    frame = pd.DataFrame({"label": [1, 1, 0, 0, 0], "score": [0.8, 0.2, 0.2, 0.1, 0.9]})
    rng = np.random.default_rng(47)
    auc, ap = analysis._bootstrap_metric_draws(frame, n_bootstrap=1, rng=rng, include_auprc=True)
    assert ap is not None

    replay = np.random.default_rng(47)
    positive = frame.loc[frame["label"].eq(1), "score"].to_numpy()
    negative = frame.loc[frame["label"].eq(0), "score"].to_numpy()
    positive_counts = replay.multinomial(2, np.full(2, 0.5), size=1)[0]
    negative_counts = replay.multinomial(3, np.full(3, 1 / 3), size=1)[0]
    scores = np.concatenate(
        [np.repeat(positive, positive_counts), np.repeat(negative, negative_counts)]
    )
    labels = np.concatenate([np.ones(positive_counts.sum()), np.zeros(negative_counts.sum())])
    assert auc[0] == pytest.approx(analysis.roc_auc_score(labels, scores))
    assert ap[0] == pytest.approx(analysis.average_precision_score(labels, scores))


def test_balanced_att_case_mix_fit_passes_on_common_support() -> None:
    rows: list[dict[str, Any]] = []
    for role in ("primary", "metastatic"):
        for label in (0, 1):
            for index in range(40):
                rows.append(
                    {
                        "role": role,
                        "label": label,
                        "score": float(label + index / 100),
                        "age": float(45 + index % 20),
                        "sex": "female" if index % 2 else "male",
                        "msi_dmmr": "MSS/pMMR" if index % 3 else "MSI/dMMR",
                        "braf": "wild_type" if index % 4 else "mutant",
                        "source_type": "biopsy" if index % 2 else "resection",
                    }
                )
    frame = pd.DataFrame(rows)
    fitted = analysis._case_mix_fit(
        frame[frame["role"].eq("primary")],
        frame[frame["role"].eq("metastatic")],
        analysis.CASE_MIX_BLOCKS["joint_source_clinical_molecular"],
    )
    assert fitted["estimable"] is True
    assert fitted["primary_att_to_metastatic_auroc"] == pytest.approx(1.0)
    assert fitted["att_standardized_difference"] == pytest.approx(0.0)


def _minimal_results(settings: dict[str, int]) -> dict[str, Any]:
    return {
        "schema_version": "crc-final-v6-met-transfer-diagnostics-1.0",
        "analysis_status": "SMOKE_TEST_ONLY_NOT_SCIENTIFIC_RESULT",
        "causal_status": "NONCAUSAL_OBSERVATIONAL_DECOMPOSITION",
        "headline": "test",
        "v6_validation": {},
        "settings": settings,
        "population": [],
        "performance_strata": [],
        "contrasts": [],
        "decomposition": [],
        "site_source_standardization": [],
        "case_mix_standardization": [],
        "score_structure": [],
        "source_alignment": [],
        "structural_cv": [],
        "paired_role_summary": [],
        "interpretation_boundary": [],
    }


def _minimal_verified_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any]]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    input_file = tmp_path / "fixed-input.txt"
    input_file.write_text("frozen\n")
    input_record = analysis._artifact(input_file)
    settings = {"n_bootstrap": 100, "cv_repeats": 2, "random_seed": analysis.RNG_SEED}
    results_settings = {
        **settings,
        "case_mix_bootstrap": 100,
        "bootstrap_unit": "patient",
    }
    _write_json(bundle / "results.json", _minimal_results(results_settings))
    output_record = analysis._artifact(bundle / "results.json")

    monkeypatch.setattr(analysis, "EXPECTED_INPUTS", {"fixed"})
    monkeypatch.setattr(analysis, "EXPECTED_OUTPUTS", {"results.json"})
    monkeypatch.setattr(analysis, "EXPECTED_CHECKS", {"closed_world"})
    monkeypatch.setattr(analysis, "input_artifacts", lambda: {"fixed": input_record})
    monkeypatch.setattr(analysis, "validate_parent_score_receipts", lambda: {})
    monkeypatch.setattr(analysis, "validate_v6", lambda: (pd.DataFrame(), {}))

    receipt = {
        "schema_version": "crc-final-v6-met-transfer-diagnostics-receipt-1.0",
        "status": "SMOKE_TEST_ONLY_NOT_SCIENTIFIC_RESULT",
        "analysis_command": (
            "python tools/crc_final_v6_met_transfer_diagnostics.py run "
            f"--output-dir {bundle} --n-bootstrap 100 --cv-repeats 2"
        ),
        "settings": settings,
        "inputs": {"fixed": input_record},
        "outputs": {"results.json": output_record},
        "checks": {"closed_world": "PASS"},
    }
    _write_json(bundle / "receipt.json", receipt)
    return bundle, receipt


def test_verifier_accepts_closed_world_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _ = _minimal_verified_bundle(tmp_path, monkeypatch)
    analysis.verify_published(bundle)


def test_verifier_rejects_extra_file_and_path_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, receipt = _minimal_verified_bundle(tmp_path, monkeypatch)
    (bundle / "extra.txt").write_text("unexpected\n")
    with pytest.raises(ValueError, match="entry roster drift"):
        analysis.verify_published(bundle)
    (bundle / "extra.txt").unlink()

    receipt["outputs"]["results.json"]["path"] = str(tmp_path / "results.json")
    _write_json(bundle / "receipt.json", receipt)
    with pytest.raises(ValueError, match="output path rejected"):
        analysis.verify_published(bundle)


def test_verifier_rejects_empty_maps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, receipt = _minimal_verified_bundle(tmp_path, monkeypatch)
    receipt["inputs"] = {}
    receipt["outputs"] = {}
    receipt["checks"] = {}
    _write_json(bundle / "receipt.json", receipt)
    with pytest.raises(ValueError, match="input roster drift"):
        analysis.verify_published(bundle)


def test_verifier_rejects_symlinked_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _ = _minimal_verified_bundle(tmp_path, monkeypatch)
    external = tmp_path / "external-receipt.json"
    (bundle / "receipt.json").replace(external)
    (bundle / "receipt.json").symlink_to(external)
    with pytest.raises(ValueError, match="Symlink"):
        analysis.verify_published(bundle)


def test_verifier_detects_output_replacement_during_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _ = _minimal_verified_bundle(tmp_path, monkeypatch)
    original_artifact = analysis._artifact
    result_calls = 0

    def replacing_artifact(path: Path) -> dict[str, Any]:
        nonlocal result_calls
        record = original_artifact(path)
        if path == bundle / "results.json":
            result_calls += 1
            if result_calls == 1:
                payload = analysis._strict_json_load(path)
                payload["headline"] = "coordinated replacement attempt"
                _write_json(path, payload)
        return record

    monkeypatch.setattr(analysis, "_artifact", replacing_artifact)
    with pytest.raises(RuntimeError, match="changed while it was being parsed"):
        analysis.verify_published(bundle)
