from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import aim1_source_cohort_five_seed_analysis as analysis  # noqa: E402
from tools import aim1_source_cohort_five_seed_campaign as training  # noqa: E402

SMALL_CENSUS = {
    "tcga_primary": {"patients": 8, "slides": 16, "mutant": 4, "wild_type": 4},
    "sr386_primary": {"patients": 8, "slides": 16, "mutant": 4, "wild_type": 4},
    "surgen_primary": {"patients": 16, "slides": 32, "mutant": 8, "wild_type": 8},
    "tcga_surgen_primary": {
        "patients": 24,
        "slides": 48,
        "mutant": 12,
        "wild_type": 12,
    },
}

SMALL_CONTRASTS = (
    analysis.ContrastSpec(
        "surgen_minus_sr386_on_sr386",
        "surgen_primary",
        "sr386_primary",
        "sr386_primary",
        8,
        4,
        4,
    ),
    analysis.ContrastSpec(
        "tcga_surgen_minus_surgen_on_surgen",
        "tcga_surgen_primary",
        "surgen_primary",
        "surgen_primary",
        16,
        8,
        8,
    ),
    analysis.ContrastSpec(
        "tcga_surgen_minus_tcga_on_tcga",
        "tcga_surgen_primary",
        "tcga_primary",
        "tcga_primary",
        8,
        4,
        4,
    ),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _patch_small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "EXPECTED_ARM_CENSUS", deepcopy(SMALL_CENSUS))
    monkeypatch.setattr(analysis, "CONTRASTS", SMALL_CONTRASTS)


def _population_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for prefix, cohort, subcohorts in (
        ("T", "TCGA", ("TCGA-COAD", "TCGA-READ")),
        ("R", "SurGen", ("SR386",)),
        ("G", "SurGen", ("SR1482",)),
    ):
        for index in range(8):
            patient_id = f"{prefix}{index:02d}"
            slides = [f"{patient_id}_slide0", f"{patient_id}_slide1"]
            metadata[patient_id] = {
                "patient_id": patient_id,
                "target_label": index % 2,
                "cohort": cohort,
                "subcohort": subcohorts[index % len(subcohorts)],
                "specimen_role": "primary",
                "k_fold": index % 5,
                "n_slides": 2,
                "slide_roster_sha256": hashlib.sha256("\n".join(slides).encode()).hexdigest(),
            }
    return metadata


def _arm_patient_ids() -> dict[str, list[str]]:
    tcga = [f"T{index:02d}" for index in range(8)]
    sr386 = [f"R{index:02d}" for index in range(8)]
    sr1482 = [f"G{index:02d}" for index in range(8)]
    return {
        "tcga_primary": tcga,
        "sr386_primary": sr386,
        "surgen_primary": [*sr386, *sr1482],
        "tcga_surgen_primary": [*tcga, *sr386, *sr1482],
    }


def _native_score(arm: str, patient_id: str, label: int, seed: int) -> float:
    strength = {
        "tcga_primary": 0.45,
        "sr386_primary": 0.35,
        "surgen_primary": 0.75,
        "tcga_surgen_primary": 0.95,
    }[arm]
    index = int(patient_id[1:])
    noise = (((index * 7) + ord(patient_id[0])) % 11 - 5) / 3.5
    seed_shift = (seed - 44) * (noise + 0.25) * 0.04
    return float((2 * label - 1) * strength + noise + seed_shift)


def _patient_scores() -> pd.DataFrame:
    metadata = _population_metadata()
    rows: list[dict[str, Any]] = []
    for arm, patients in _arm_patient_ids().items():
        for patient_id in patients:
            row = {"arm": arm, **metadata[patient_id]}
            for seed in analysis.SEEDS:
                row[f"logit_seed{seed}"] = _native_score(
                    arm, patient_id, int(row["target_label"]), seed
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    seed_columns = [f"logit_seed{seed}" for seed in analysis.SEEDS]
    frame["mean_logit_5seed"] = frame[seed_columns].mean(axis=1)
    return frame.sort_values(["arm", "patient_id"]).reset_index(drop=True)


def _source_bundle(tmp_path: Path) -> analysis.SourceBundle:
    patient_scores = _patient_scores()
    manifests: dict[str, pd.DataFrame] = {}
    oof: dict[tuple[str, int], pd.DataFrame] = {}
    for arm in analysis.ARMS:
        patients = patient_scores.loc[patient_scores["arm"].eq(arm)]
        manifest_rows: list[dict[str, Any]] = []
        for row in patients.to_dict("records"):
            for slide_index in range(2):
                manifest_rows.append(
                    {
                        "slide_id": f"{row['patient_id']}_slide{slide_index}",
                        "patient_id": row["patient_id"],
                        "target_label": row["target_label"],
                        "cohort": row["cohort"],
                        "subcohort": row["subcohort"],
                        "specimen_role": row["specimen_role"],
                        "k_fold": row["k_fold"],
                    }
                )
        manifest = pd.DataFrame(manifest_rows)
        manifests[arm] = manifest
        patient_lookup = patients.set_index("patient_id")
        for seed in analysis.SEEDS:
            rows = []
            for slide in manifest.to_dict("records"):
                center = float(patient_lookup.loc[slide["patient_id"], f"logit_seed{seed}"])
                offset = -0.2 if str(slide["slide_id"]).endswith("0") else 0.2
                rows.append(
                    {
                        "slide_id": slide["slide_id"],
                        "label": slide["target_label"],
                        "logit": center + offset,
                        "fold": slide["k_fold"],
                    }
                )
            oof[(arm, seed)] = pd.DataFrame(rows).sample(frac=1.0, random_state=seed)
    return analysis.SourceBundle(tmp_path, {}, manifests, oof)


def _sealed_synthetic_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str, list[tuple[str, int]]]:
    _patch_small_contract(monkeypatch)
    source = _source_bundle(tmp_path)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    for arm, manifest in source.manifests.items():
        path = training.manifest_path(campaign, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(path, index=False)
    for (arm, seed), frame in source.oof_by_arm_seed.items():
        path = training.run_dir(campaign, arm, seed) / "oof_predictions.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    contract_path = training.contract_path(campaign)
    preflight_path = training.preflight_path(campaign)
    scheduler_path = campaign / "receipts/scheduler.json"
    _write_json(contract_path, {"status": "sealed"})
    _write_json(preflight_path, {"status": "deep_preflight_passed"})
    _write_json(scheduler_path, {"status": "completed_rc0"})

    expected_pairs = [(arm, seed) for arm in analysis.ARMS for seed in analysis.SEEDS]
    job_identities = []
    for arm, seed in expected_pairs:
        directory = training.run_dir(campaign, arm, seed)
        oof_path = directory / "oof_predictions.parquet"
        completion_path = directory / "training_completion.json"
        identity_path = directory / "training_identity.json"
        summary_path = directory / "cv_summary.json"
        _write_json(completion_path, {"status": "complete"})
        _write_json(identity_path, {"fingerprint": f"{arm}-{seed}"})
        _write_json(summary_path, {"n_folds": 5})
        receipt = {
            "schema_version": training.SCHEMA_VERSION,
            "status": "completed",
            "arm": arm,
            "seed": seed,
            "fit_count": 5,
            "refit_count": 0,
            "artifacts": {
                "completion": analysis._artifact(completion_path),  # noqa: SLF001
                "identity": analysis._artifact(identity_path),  # noqa: SLF001
                "oof": analysis._artifact(oof_path),  # noqa: SLF001
                "cv_summary": analysis._artifact(summary_path),  # noqa: SLF001
                "oof_rows": len(source.manifests[arm]),
                "fold_count": 5,
                "refit_count": 0,
                "training_fingerprint": f"{arm}-{seed}",
            },
        }
        path = training.job_receipt_path(campaign, arm, seed)
        _write_json(path, receipt)
        job_identities.append(analysis._artifact(path))  # noqa: SLF001

    terminal = {
        "schema_version": training.SCHEMA_VERSION,
        "status": "complete_and_certified",
        "created_utc": "2026-08-24T18:00:00+00:00",
        "contract": analysis._artifact(contract_path),  # noqa: SLF001
        "preflight": analysis._artifact(preflight_path),  # noqa: SLF001
        "scheduler": analysis._artifact(scheduler_path),  # noqa: SLF001
        "arms": list(analysis.ARMS),
        "seeds": list(analysis.SEEDS),
        "job_count": 20,
        "folds_per_job": 5,
        "logical_fit_count": 100,
        "refit_count": 0,
        "job_receipts": job_identities,
    }
    _write_json(training.training_receipt_path(campaign), terminal)

    parent = tmp_path / "final_v10/report_bundle_receipt.json"
    _write_json(
        parent,
        {"status": analysis.EXPECTED_PARENT_FINAL_V10_STATUS, "schema_version": 1},
    )
    parent_sha = analysis._sha256(parent)  # noqa: SLF001
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        analysis,
        "_live_parent_verification",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "published_status": analysis.EXPECTED_PARENT_FINAL_V10_STATUS,
            "verifier": {"path": "synthetic", "sha256": "synthetic", "size_bytes": 0},
        },
    )
    monkeypatch.setattr(training, "validate_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(training, "_validate_arm_frame", lambda *_args, **_kwargs: {})

    def validate_job(root: Path, arm: str, seed: int) -> dict[str, Any]:
        calls.append((arm, seed))
        return analysis._read_json(training.job_receipt_path(root, arm, seed))  # noqa: SLF001

    monkeypatch.setattr(training, "_validate_job", validate_job)
    return campaign, parent, parent_sha, calls


def test_public_contract_and_exact_three_clean_contrasts() -> None:
    assert analysis.SEEDS == (42, 43, 44, 45, 46)
    assert analysis.N_BOOTSTRAP == 10_000
    assert analysis.ANALYSIS_FILES == (
        "contract.json",
        "patient_native_logits.parquet",
        "results.json",
        "bootstrap_distributions.npz",
        "analysis_completion_receipt.json",
    )
    assert [spec.key for spec in analysis.CONTRASTS] == [
        "surgen_minus_sr386_on_sr386",
        "tcga_surgen_minus_surgen_on_surgen",
        "tcga_surgen_minus_tcga_on_tcga",
    ]
    assert len(analysis.BOOTSTRAP_ARRAY_NAMES) == 26
    assert tuple(sorted(analysis.BOOTSTRAP_ARRAY_NAMES)) == analysis.BOOTSTRAP_ARRAY_NAMES


def test_live_sealed_final_v10_parent_replays_when_mounted() -> None:
    if not analysis.DEFAULT_PARENT_FINAL_V10_RECEIPT.is_file():
        pytest.skip("sealed FINAL-v10 parent receipt is not mounted")
    evidence = analysis._validate_parent_receipt(  # noqa: SLF001
        analysis.DEFAULT_PARENT_FINAL_V10_RECEIPT
    )
    assert evidence["receipt"]["sha256"] == (analysis.EXPECTED_PARENT_FINAL_V10_RECEIPT_SHA256)
    assert evidence["live_verification"]["status"] == "PASS"
    assert evidence["live_verification"]["published_status"] == (
        analysis.EXPECTED_PARENT_FINAL_V10_STATUS
    )


def test_native_slide_logits_are_averaged_within_patient_then_across_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_small_contract(monkeypatch)
    observed = analysis.aggregate_patient_scores(_source_bundle(tmp_path))
    expected = _patient_scores()
    columns = [
        "arm",
        "patient_id",
        "n_slides",
        "slide_roster_sha256",
        *(f"logit_seed{seed}" for seed in analysis.SEEDS),
        "mean_logit_5seed",
    ]
    pd.testing.assert_frame_equal(
        observed[columns], expected[columns], check_exact=False, rtol=0.0, atol=1e-15
    )
    assert observed["n_slides"].eq(2).all()


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_label", "wrong_fold", "nan"])
def test_oof_roster_label_fold_and_native_logit_faults_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    _patch_small_contract(monkeypatch)
    source = _source_bundle(tmp_path)
    tables = dict(source.oof_by_arm_seed)
    key = ("tcga_primary", 42)
    frame = tables[key].copy().reset_index(drop=True)
    if fault == "missing":
        frame = frame.iloc[:-1]
    elif fault == "duplicate":
        frame.loc[1, "slide_id"] = frame.loc[0, "slide_id"]
    elif fault == "wrong_label":
        frame.loc[0, "label"] = 1 - int(frame.loc[0, "label"])
    elif fault == "wrong_fold":
        frame.loc[0, "fold"] = (int(frame.loc[0, "fold"]) + 1) % 5
    else:
        frame.loc[0, "logit"] = np.nan
    tables[key] = frame
    broken = analysis.SourceBundle(source.campaign_root, {}, source.manifests, tables)
    with pytest.raises(analysis.AnalysisError):
        analysis.aggregate_patient_scores(broken)


def test_named_stratified_bootstrap_is_deterministic_and_preserves_cells() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    subcohorts = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    first = analysis.stratified_bootstrap_indices(
        labels, subcohorts, n_bootstrap=30, seed=17, stream="cell-test"
    )
    second = analysis.stratified_bootstrap_indices(
        labels, subcohorts, n_bootstrap=30, seed=17, stream="cell-test"
    )
    assert np.array_equal(first, second)
    expected = {
        (subcohort, label): int(np.sum((subcohorts == subcohort) & (labels == label)))
        for subcohort in set(subcohorts)
        for label in (0, 1)
    }
    for draw in first:
        observed = {
            (subcohort, label): int(
                np.sum((subcohorts[draw] == subcohort) & (labels[draw] == label))
            )
            for subcohort in set(subcohorts)
            for label in (0, 1)
        }
        assert observed == expected


def test_analysis_is_order_and_byte_deterministic_with_patient_only_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_contract(monkeypatch)
    patients = _patient_scores()
    first, arrays_first = analysis.build_analysis(patients, n_bootstrap=60, bootstrap_seed=123)
    second, arrays_second = analysis.build_analysis(
        patients.sample(frac=1.0, random_state=9),
        n_bootstrap=60,
        bootstrap_seed=123,
    )
    assert first == second
    assert arrays_first.keys() == arrays_second.keys()
    assert tuple(arrays_first) == analysis.BOOTSTRAP_ARRAY_NAMES
    assert len(arrays_first) == 26
    assert all(value.dtype == np.float64 for value in arrays_first.values())
    assert all(value.shape == (60,) for value in arrays_first.values())
    assert all(np.array_equal(arrays_first[key], arrays_second[key]) for key in arrays_first)
    assert analysis._npz_bytes(arrays_first) == analysis._npz_bytes(arrays_second)  # noqa: SLF001
    assert first["inference"]["unit"] == "patient"
    assert first["inference"]["model_seeds_are_inference_units"] is False
    assert first["inference"]["folds_are_inference_units"] is False
    assert first["cross_population_ranking"]["role"] == "descriptive_only"
    assert set(first["arm_performance"]) == set(analysis.ARMS)
    assert all(
        set(block["per_seed_descriptive"]) == {str(seed) for seed in analysis.SEEDS}
        for block in first["arm_performance"].values()
    )


def test_paired_draws_are_exact_shared_patient_differences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_contract(monkeypatch)
    results, arrays = analysis.build_analysis(_patient_scores(), n_bootstrap=50, bootstrap_seed=321)
    assert set(results["paired_common_patient_contrasts"]) == {spec.key for spec in SMALL_CONTRASTS}
    for spec in SMALL_CONTRASTS:
        block = results["paired_common_patient_contrasts"][spec.key]
        assert block["patients"] == spec.expected_patients
        assert block["paired_indices_shared"] is True
        prefix = f"contrast__{spec.key}"
        for metric in ("auroc", "auprc"):
            assert np.array_equal(
                arrays[f"{prefix}__delta__{metric}"],
                arrays[f"{prefix}__larger__{metric}"] - arrays[f"{prefix}__smaller__{metric}"],
            )


def test_pairing_rejects_metadata_drift_and_seed_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_contract(monkeypatch)
    patients = _patient_scores()
    mask = patients["arm"].eq("tcga_surgen_primary") & patients["patient_id"].eq("T00")
    patients.loc[mask, "k_fold"] = 4
    with pytest.raises(analysis.AnalysisError, match="paired patient metadata"):
        analysis.build_analysis(patients, n_bootstrap=5)
    with pytest.raises(analysis.AnalysisError, match="not permitted as inference units"):
        analysis.build_analysis(_patient_scores(), n_bootstrap=5, inference_unit="model_seed")


def test_source_loader_requires_exact_terminal_20_chain_100_fit_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, parent, parent_sha, calls = _sealed_synthetic_campaign(tmp_path, monkeypatch)
    source = analysis.load_training_source(
        campaign,
        parent_final_v10_receipt=parent,
        expected_parent_sha256=parent_sha,
    )
    assert set(source.oof_by_arm_seed) == {
        (arm, seed) for arm in analysis.ARMS for seed in analysis.SEEDS
    }
    assert calls == [(arm, seed) for arm in analysis.ARMS for seed in analysis.SEEDS]

    terminal_path = training.training_receipt_path(campaign)
    terminal = analysis._read_json(terminal_path)  # noqa: SLF001
    terminal["logical_fit_count"] = 99
    _write_json(terminal_path, terminal)
    with pytest.raises(analysis.AnalysisError, match="terminal semantics"):
        analysis.load_training_source(
            campaign,
            parent_final_v10_receipt=parent,
            expected_parent_sha256=parent_sha,
        )


def test_source_loader_rejects_coordinated_duplicate_key_job_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, parent, parent_sha, _calls = _sealed_synthetic_campaign(tmp_path, monkeypatch)
    receipt_path = training.job_receipt_path(campaign, "tcga_primary", 42)
    text = receipt_path.read_text(encoding="utf-8")
    receipt_path.write_text(
        text.replace('{\n', '{\n  "status": "adversarial-shadow-value",\n', 1),
        encoding="utf-8",
    )
    terminal_path = training.training_receipt_path(campaign)
    terminal = analysis._read_json(terminal_path)  # noqa: SLF001
    terminal["job_receipts"][0] = analysis._artifact(receipt_path)  # noqa: SLF001
    _write_json(terminal_path, terminal)
    monkeypatch.setattr(
        training,
        "_validate_job",
        lambda root, arm, seed: training._read_json(  # noqa: SLF001
            training.job_receipt_path(root, arm, seed)
        ),
    )
    with pytest.raises(analysis.AnalysisError, match="Duplicate JSON key"):
        analysis.load_training_source(
            campaign,
            parent_final_v10_receipt=parent,
            expected_parent_sha256=parent_sha,
        )


def test_incomplete_training_fails_before_analysis_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, parent, parent_sha, _calls = _sealed_synthetic_campaign(tmp_path, monkeypatch)
    training.training_receipt_path(campaign).unlink()
    with pytest.raises(analysis.AnalysisError):
        analysis.analyze(
            campaign,
            parent_final_v10_receipt=parent,
            expected_parent_sha256=parent_sha,
            n_bootstrap=10,
            _allow_test_parameters=True,
        )
    assert not analysis.analysis_root(campaign).exists()


@pytest.mark.parametrize(
    ("n_bootstrap", "bootstrap_seed"),
    ((9_999, analysis.BOOTSTRAP_SEED), (analysis.N_BOOTSTRAP, 1)),
)
def test_noncanonical_production_parameters_fail_before_any_write(
    tmp_path: Path, n_bootstrap: int, bootstrap_seed: int
) -> None:
    campaign = tmp_path / "campaign"
    with pytest.raises(analysis.AnalysisError, match="requires exactly"):
        analysis.analyze(
            campaign,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
    assert not analysis.analysis_root(campaign).exists()


def test_extra_job_receipt_and_source_oof_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, parent, parent_sha, _calls = _sealed_synthetic_campaign(tmp_path, monkeypatch)
    extra = campaign / "receipts/source_cv/extra.json"
    _write_json(extra, {"status": "foreign"})
    with pytest.raises(analysis.AnalysisError, match="missing or extra"):
        analysis.load_training_source(
            campaign,
            parent_final_v10_receipt=parent,
            expected_parent_sha256=parent_sha,
        )
    extra.unlink()
    oof = training.run_dir(campaign, "tcga_primary", 42) / "oof_predictions.parquet"
    oof.write_bytes(oof.read_bytes() + b"tamper")
    with pytest.raises(analysis.AnalysisError):
        analysis.load_training_source(
            campaign,
            parent_final_v10_receipt=parent,
            expected_parent_sha256=parent_sha,
        )


def test_publication_is_immutable_replayable_and_verify_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, parent, parent_sha, _calls = _sealed_synthetic_campaign(tmp_path, monkeypatch)
    result = analysis.analyze(
        campaign,
        parent_final_v10_receipt=parent,
        expected_parent_sha256=parent_sha,
        n_bootstrap=40,
        bootstrap_seed=19,
        _allow_test_parameters=True,
    )
    root = analysis.analysis_root(campaign)
    assert set(path.name for path in root.iterdir()) == set(analysis.ANALYSIS_FILES)
    assert result["inference"]["unit"] == "patient"
    receipt = analysis._read_json(root / "analysis_completion_receipt.json")  # noqa: SLF001
    assert receipt["status"] == "complete_and_certified"
    assert receipt["source_inputs"] == result["inputs"]
    assert receipt["inference"]["bootstrap_arrays"] == {
        "names": list(analysis.BOOTSTRAP_ARRAY_NAMES),
        "count": 26,
        "dtype": "float64",
        "length": 40,
    }
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.iterdir()}
    replay = analysis.verify(
        campaign,
        parent_final_v10_receipt=parent,
        expected_parent_sha256=parent_sha,
    )
    assert replay == result
    analysis.analyze(
        campaign,
        parent_final_v10_receipt=parent,
        expected_parent_sha256=parent_sha,
        n_bootstrap=40,
        bootstrap_seed=19,
        _allow_test_parameters=True,
    )
    after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.iterdir()}
    assert before == after


def test_coordinated_result_rehash_still_fails_canonical_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, parent, parent_sha, _calls = _sealed_synthetic_campaign(tmp_path, monkeypatch)
    analysis.analyze(
        campaign,
        parent_final_v10_receipt=parent,
        expected_parent_sha256=parent_sha,
        n_bootstrap=20,
        _allow_test_parameters=True,
    )
    root = analysis.analysis_root(campaign)
    result_path = root / "results.json"
    result_path.write_bytes(result_path.read_bytes() + b" ")
    receipt_path = root / "analysis_completion_receipt.json"
    receipt = analysis._read_json(receipt_path)  # noqa: SLF001
    receipt["artifacts"]["results"] = analysis._artifact(result_path)  # noqa: SLF001
    _write_json(receipt_path, receipt)
    with pytest.raises(analysis.AnalysisError, match="not canonical"):
        analysis.verify(
            campaign,
            parent_final_v10_receipt=parent,
            expected_parent_sha256=parent_sha,
        )


def test_symlink_extra_file_and_partial_component_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, parent, parent_sha, _calls = _sealed_synthetic_campaign(tmp_path, monkeypatch)
    analysis.analyze(
        campaign,
        parent_final_v10_receipt=parent,
        expected_parent_sha256=parent_sha,
        n_bootstrap=15,
        _allow_test_parameters=True,
    )
    root = analysis.analysis_root(campaign)
    (root / "extra.txt").write_text("foreign", encoding="utf-8")
    with pytest.raises(analysis.AnalysisError, match="missing or extra"):
        analysis.verify(
            campaign,
            parent_final_v10_receipt=parent,
            expected_parent_sha256=parent_sha,
        )

    partial_campaign = tmp_path / "partial-campaign"
    partial_campaign.mkdir()
    partial = analysis.analysis_root(partial_campaign)
    partial.mkdir()
    (partial / "contract.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        analysis,
        "load_training_source",
        lambda *_args, **_kwargs: analysis.SourceBundle(partial_campaign, {}, {}, {}),
    )
    with pytest.raises(analysis.AnalysisError):
        analysis.analyze(partial_campaign, n_bootstrap=5, _allow_test_parameters=True)


@pytest.mark.parametrize("concurrent_bytes", [b"foreign", b"sealed"])
def test_write_once_publication_never_overwrites_a_concurrent_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concurrent_bytes: bytes
) -> None:
    destination = tmp_path / "artifact.json"
    original_link = analysis.os.link

    def racing_link(source: Path, target: Path) -> None:
        Path(target).write_bytes(concurrent_bytes)
        original_link(source, target)

    monkeypatch.setattr(analysis.os, "link", racing_link)
    if concurrent_bytes == b"sealed":
        analysis._write_bytes_once(destination, b"sealed")  # noqa: SLF001
    else:
        with pytest.raises(analysis.AnalysisError, match="Refusing to replace"):
            analysis._write_bytes_once(destination, b"sealed")  # noqa: SLF001
    assert destination.read_bytes() == concurrent_bytes
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_analysis_artifact_identity_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(analysis.AnalysisError, match="symlinked"):
        analysis._artifact(link)  # noqa: SLF001


def test_cli_is_dry_by_default_and_exposes_read_only_verify_status() -> None:
    parser = analysis.build_parser()
    dry = parser.parse_args(["analyze"])
    apply = parser.parse_args(["analyze", "--apply"])
    verify = parser.parse_args(["verify"])
    status = parser.parse_args(["status"])
    assert dry.apply is False
    assert apply.apply is True
    assert not hasattr(apply, "n_bootstrap")
    assert not hasattr(apply, "bootstrap_seed")
    assert callable(verify.func)
    assert callable(status.func)


def test_status_waiting_does_not_require_or_create_analysis_output(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    payload = analysis.status(campaign)
    assert payload["status"] == "waiting_for_training"
    assert not analysis.analysis_root(campaign).exists()


def test_dry_run_validates_source_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    campaign, parent, parent_sha, _calls = _sealed_synthetic_campaign(tmp_path, monkeypatch)
    original_loader = analysis.load_training_source

    def load_with_synthetic_parent(
        campaign_root: Path, *, parent_final_v10_receipt: Path
    ) -> analysis.SourceBundle:
        return original_loader(
            campaign_root,
            parent_final_v10_receipt=parent_final_v10_receipt,
            expected_parent_sha256=parent_sha,
        )

    monkeypatch.setattr(analysis, "load_training_source", load_with_synthetic_parent)
    analysis.cmd_analyze(
        argparse.Namespace(
            campaign_root=campaign,
            parent_final_v10_receipt=parent,
            apply=False,
            n_bootstrap=10,
            bootstrap_seed=5,
        )
    )
    assert "DRY RUN PASS" in capsys.readouterr().out
    assert not analysis.analysis_root(campaign).exists()
