"""Focused contracts for the append-only Aim-3 repeated-control campaign."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim3_repeated_control_campaign as campaign  # noqa: E402


def _development() -> pd.DataFrame:
    rows = []
    index = 0
    for subcohort, cohort in (("A-COAD", "A"), ("B-COAD", "B")):
        for fold in range(5):
            definitions = [
                ("wild_type", "", 6),
                ("mutant", "G12D", 2),
                ("mutant", "G13D", 2),
            ]
            for kras, subvariant, count in definitions:
                for _ in range(count):
                    patient = f"P{index:04d}"
                    rows.append(
                        {
                            "slide_id": f"S{index:04d}",
                            "patient_id": patient,
                            "target_label": int(kras == "mutant"),
                            "kras": kras,
                            "kras_subvariant": subvariant,
                            "cohort": cohort,
                            "subcohort": subcohort,
                            "k_fold": fold,
                        }
                    )
                    index += 1
    return pd.DataFrame(rows)


def _arm(prefix: str, *, positive_shift: float, negative_shift: float) -> pd.DataFrame:
    rows = []
    for subcohort in ("A", "B"):
        for fold in range(2):
            for index in range(8):
                rows.append(
                    {
                        "patient_id": f"POS-{subcohort}-{fold}-{index}",
                        "label": 1,
                        "mean_logit": positive_shift + index / 50,
                        "subcohort": subcohort,
                        "k_fold": fold,
                    }
                )
            for index in range(8):
                rows.append(
                    {
                        "patient_id": f"{prefix}-NEG-{subcohort}-{fold}-{index}",
                        "label": 0,
                        "mean_logit": negative_shift + index / 50,
                        "subcohort": subcohort,
                        "k_fold": fold,
                    }
                )
    return pd.DataFrame(rows)


def test_draw_matches_replaced_negatives_in_every_subcohort_fold():
    development = _development()
    fine = campaign.fine_patient_labels(development, "codon")
    control = campaign.draw_control_patients(development, "codon", 20260823)
    expected = fine[fine.target_label.eq(0)].groupby(["subcohort", "k_fold"]).size()
    observed = control[control.target_label.eq(0)].groupby(["subcohort", "k_fold"]).size()
    pd.testing.assert_series_equal(observed, expected)
    assert control.target_label.value_counts().to_dict() == fine.target_label.value_counts().to_dict()


def test_three_predeclared_draws_are_deterministic_and_distinct():
    development = _development()
    draws = []
    for seed in campaign.WT_DRAW_SEEDS:
        first = campaign.draw_control_patients(development, "codon", seed)
        second = campaign.draw_control_patients(development, "codon", seed)
        pd.testing.assert_frame_equal(first, second)
        draws.append(frozenset(first.loc[first.target_label.eq(0), "patient_id"]))
    assert len(set(draws)) == 3


def test_unregistered_wt_seed_is_rejected():
    with pytest.raises(ValueError, match="unregistered"):
        campaign.draw_control_patients(_development(), "codon", 1)


def test_standalone_allele_parser_preserves_multi_substitution_membership():
    assert campaign.is_g12("G12V;G12C")
    assert campaign.has_allele("G12V;G12C", "G12C")
    assert campaign.is_g12d("G12D;G13D")
    assert not campaign.is_g12("A146T")


def test_partial_paired_bootstrap_detects_control_advantage():
    fine = _arm("F", positive_shift=0.0, negative_shift=0.0)
    control = _arm("C", positive_shift=1.0, negative_shift=0.0)
    result = campaign.partial_paired_bootstrap(fine, control, n_bootstrap=500, seed=9)
    assert result["point"]["control"] - result["point"]["fine"] > 0.45
    assert campaign.interval(result["delta_values"], alpha=0.05)[0] > 0


def test_shared_positive_indices_are_joint_not_two_independent_draws():
    fine = _arm("F", positive_shift=0.2, negative_shift=0.0)
    control = _arm("C", positive_shift=0.2, negative_shift=0.0)
    # Make every negative identical, so with identical arm scores and truly
    # joint positive indices every replicate's delta must be exactly zero.
    fine.loc[fine.label.eq(0), "mean_logit"] = 0.0
    control.loc[control.label.eq(0), "mean_logit"] = 0.0
    result = campaign.partial_paired_bootstrap(fine, control, n_bootstrap=200, seed=19)
    assert np.array_equal(result["fine_values"], result["control_values"])
    assert np.count_nonzero(result["delta_values"]) == 0


def test_partial_pairing_requires_exact_shared_positive_ids():
    fine = _arm("F", positive_shift=0.0, negative_shift=0.0)
    control = _arm("C", positive_shift=1.0, negative_shift=0.0)
    control.loc[control.label.eq(1).idxmax(), "patient_id"] = "WRONG"
    with pytest.raises(ValueError, match="positive patients"):
        campaign.partial_paired_bootstrap(fine, control, n_bootstrap=10)


def test_partial_pairing_requires_negative_stratum_counts():
    fine = _arm("F", positive_shift=0.0, negative_shift=0.0)
    control = _arm("C", positive_shift=1.0, negative_shift=0.0)
    control = control.drop(control[control.label.eq(0)].index[0])
    with pytest.raises(ValueError, match="counts"):
        campaign.partial_paired_bootstrap(fine, control, n_bootstrap=10)


def test_primary_bound_is_one_sided_99_percent_for_five_rungs():
    values = np.arange(10_001, dtype=float)
    bounds = campaign.one_sided_bounds(values)
    assert bounds["one_sided_confidence"] == pytest.approx(0.99)
    assert bounds["lower"] == pytest.approx(100.0)
    assert bounds["upper"] == pytest.approx(9900.0)


def test_ceiling_consensus_requires_every_draw():
    assert campaign.consensus_verdict(["CEILING", "CEILING", "CEILING"]) == "CONSENSUS_CEILING"
    assert (
        campaign.consensus_verdict(["CEILING", "INCONCLUSIVE", "CEILING"])
        == "NO_CEILING_CONSENSUS"
    )
    with pytest.raises(ValueError, match="all three"):
        campaign.consensus_verdict(["CEILING", "CEILING"])


def test_training_command_is_new_root_and_frozen_recipe(tmp_path: Path):
    root = tmp_path / "aim3_campaign"
    command = campaign.train_command(
        root,
        "ctrl_codon",
        20260823,
        42,
        4,
        packed_fingerprint="abc123",
        attempt_id="attempt-1",
    )
    joined = " ".join(command)
    assert "dataset_max_instances=8192" in joined
    assert "eval_full_bags=true" in joined
    assert "train_sampling_strategy=patient_natural" in joined
    assert "campaign_packed_store_sha256=abc123" in joined
    assert "hydra.job.chdir=false" in joined
    assert str(root / "state" / "hydra" / "ctrl_codon" / "wt20260823" / "seed42" / "attempt-1") in joined
    assert str(root / "train" / "ctrl_codon" / "wt20260823" / "seed42") in joined
    assert "/outputs/aim1/e3a/train/ctrl_codon" not in joined


def test_chain_accounting_is_exact():
    assert campaign.EXPECTED_FINE_FOLDS == 75
    assert campaign.EXPECTED_CONTROL_CHAINS == 45
    assert campaign.EXPECTED_CONTROL_FOLDS == 225
    assert len(campaign.all_chains()) == 45


def test_exclusive_writer_refuses_overwrite(tmp_path: Path):
    destination = tmp_path / "receipt.json"
    campaign._write_json_once(destination, {"first": True})
    with pytest.raises(FileExistsError, match="overwrite"):
        campaign._write_json_once(destination, {"second": True})
    assert destination.read_text().strip() == '{\n  "first": true\n}'


def test_material_snapshot_includes_transitive_source_configs_and_locks():
    files = set(campaign.material_source_files())
    assert {
        "src/oceanpath/models/abmil.py",
        "src/oceanpath/datasets/datamodule.py",
        "src/oceanpath/datasets/packed.py",
        "src/oceanpath/datasets/sampling.py",
        "src/oceanpath/training/lightning.py",
        "src/oceanpath/training/callbacks.py",
        "src/oceanpath/workflows/finalize.py",
        "configs/training/default.yaml",
        "configs/runtime/default.yaml",
        "configs/extraction/default.yaml",
        "configs/experiment/supervised.yaml",
        "pyproject.toml",
        "uv.lock",
    } <= files


def test_campaign_lock_rejects_a_second_launcher(tmp_path: Path):
    root = tmp_path / "aim3_campaign"
    with (
        campaign._campaign_train_lock(root),
        pytest.raises(RuntimeError, match="another process"),
        campaign._campaign_train_lock(root),
    ):
        pass


def test_exact_oof_must_equal_union_of_fold_test_predictions(tmp_path: Path):
    parts = []
    for fold in range(5):
        frame = pd.DataFrame(
            {
                "slide_id": [f"S{fold}"],
                "label": [fold % 2],
                "prob_1": [0.1 * (fold + 1)],
                "logit": [float(fold) - 2.0],
            }
        )
        directory = tmp_path / f"fold_{fold}"
        directory.mkdir()
        frame.to_parquet(directory / "preds_test.parquet", index=False)
        parts.append(frame.assign(fold=fold))
    oof_path = tmp_path / "oof_predictions.parquet"
    pd.concat(parts, ignore_index=True).to_parquet(oof_path, index=False)
    campaign._validate_exact_oof_union(tmp_path, oof_path)
    corrupted = pd.read_parquet(oof_path)
    corrupted.loc[0, "logit"] += 0.25
    corrupted.to_parquet(oof_path, index=False)
    with pytest.raises(RuntimeError, match="exact concatenated"):
        campaign._validate_exact_oof_union(tmp_path, oof_path)


def test_canonical_training_validator_is_bound_to_identity(monkeypatch, tmp_path: Path):
    payload = {"material_config": {"training": {"seed": 42}}}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:16]
    identity_path = tmp_path / "training_identity.json"
    identity_path.write_text(
        json.dumps({"schema_version": 2, "fingerprint": fingerprint, "payload": payload})
    )
    calls = []

    def fake_validate(directory, *, expected_fingerprint, require_test_predictions):
        calls.append((directory, expected_fingerprint, require_test_predictions))
        return {
            "n_folds": 5,
            "skip_finalize": False,
            "fold_completions": [
                {"path": f"fold_{fold}/completion.json"} for fold in range(5)
            ],
            "artifacts": {"cv_summary": {}, "best_fold_checkpoint": {}},
        }

    monkeypatch.setattr(campaign, "validate_training_run_dir", fake_validate)
    campaign._validated_training_completion(tmp_path, identity_path)
    assert calls == [(tmp_path, fingerprint, True)]


def test_packed_store_provenance_binds_payload_and_selected_coverage(
    monkeypatch, tmp_path: Path
):
    source = tmp_path / "features"
    source.mkdir()
    pack = tmp_path / "packed"
    pack.mkdir()
    index = pd.DataFrame(
        {"slide_id": ["S0", "S1"], "offset": [0, 1], "n_patches": [1, 1]}
    )
    index.to_parquet(pack / "index.parquet", index=False)
    (pack / "features.bin").write_bytes(np.zeros((2, 1024), dtype=np.float16).tobytes())
    (pack / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feat_dim": 1024,
                "feat_dtype": "float16",
                "coord_dim": 0,
                "n_slides": 2,
                "total_patches": 2,
                "source_dir": str(source),
                "source_inventory_sha256": "source-hash",
                "has_coords": False,
            }
        )
    )
    monkeypatch.setattr(campaign.paths, "PACKED_FEATURE_DIR", pack)
    monkeypatch.setattr(campaign.paths, "PINNED_FEATURE_DIR", source)
    manifests = {
        ("ctrl_codon", 20260823): pd.DataFrame({"slide_id": ["S0", "S1"]})
    }
    provenance = campaign.packed_store_provenance(manifests, hash_payload=True)
    assert provenance["selected_manifest_coverage"]["aim1_aim3rc_ctrl_codon_wt20260823"] == {
        "selected_slides": 2,
        "missing_slides": 0,
    }
    assert len(provenance["fingerprint_sha256"]) == 64
    with (pack / "features.bin").open("r+b") as stream:
        stream.write(b"changed!")
    with pytest.raises(RuntimeError, match="features.bin"):
        campaign.assert_packed_store_matches(provenance, manifests, full_hash=False)
