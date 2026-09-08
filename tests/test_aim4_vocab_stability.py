"""Focused contracts for the append-only Aim-4 vocabulary sensitivity."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import aim4_vocab_stability as stability  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_inputs(tmp_path: Path) -> stability.Inputs:
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    dimension = 64
    rows_by_slide = {
        "S-M04": np.vstack(
            [np.tile(np.eye(dimension, dtype=np.float32)[17], (12, 1)), np.eye(dimension, dtype=np.float32)[:32]]
        ),
        "S-M07": np.vstack(
            [np.tile(np.eye(dimension, dtype=np.float32)[28], (12, 1)), np.eye(dimension, dtype=np.float32)[:32]]
        ),
    }
    plan_rows = []
    for index, (slide, features) in enumerate(rows_by_slide.items()):
        with h5py.File(feature_root / f"{slide}.h5", "w") as handle:
            handle.create_dataset("features", data=features)
        plan_rows.append(
            {
                "slide_id": slide,
                "patient_id": f"P{index}",
                "atlas_group": "G",
                "n_tiles": len(features),
                "n_sample": len(features),
            }
        )
    plan = pd.DataFrame(plan_rows)
    sample_plan = tmp_path / "sample_plan_k32.parquet"
    plan.to_parquet(sample_plan, index=False)

    vocab = tmp_path / "vocab_k32.npz"
    centroids = np.eye(dimension, dtype=np.float32)[:32]
    np.savez_compressed(
        vocab,
        centroids=centroids,
        pca_mean=np.zeros(dimension, dtype=np.float32),
        pca_components=np.eye(dimension, dtype=np.float32),
    )
    _write_json(
        vocab.with_suffix(".json"),
        {
            "normalize": "l2",
            "n_prototypes": 32,
            "n_components": 64,
            "seed": 20260819,
            "n_sample_tiles": int(plan.n_sample.sum()),
            "cluster_sizes": [
                14 if prototype in (17, 28) else 2
                for prototype in range(32)
            ],
            "corpus_slides": 2,
            "corpus_patients": 2,
            "groups": ["G"],
            "label_blind": True,
        },
    )

    key_rows = []
    for montage, prototype, slide in (("M04", 17, "S-M04"), ("M07", 28, "S-M07")):
        for slot in range(12):
            key_rows.append(
                {
                    "montage_id": montage,
                    "slot": slot,
                    "prototype": prototype,
                    "slide_id": slide,
                    "tile_index": slot,
                }
            )
    review_key = tmp_path / "KEY_do_not_open_before_review.csv"
    pd.DataFrame(key_rows).to_csv(review_key, index=False)
    atlas_report = tmp_path / "e3b_atlas_k32.json"
    _write_json(
        atlas_report,
        {
            "k": 32,
            "pathology_review": {
                "status": "complete",
                "base_packet": {"key_sha256": stability.sha256_file(review_key)},
                "montage_by_prototype": {"17": "M04", "28": "M07"},
            },
        },
    )
    development_manifest = tmp_path / "aim1_dev.csv"
    pd.DataFrame(
        [
            {
                "slide_id": "S-M04",
                "patient_id": "P0",
                "target_label": 0,
                "kras": "wild_type",
                "msi_dmmr": "MSS/pMMR",
                "braf": "wild_type",
                "cohort": "A",
                "subcohort": "A1",
                "specimen_role": "primary",
            },
            {
                "slide_id": "S-M07",
                "patient_id": "P1",
                "target_label": 1,
                "kras": "mutant",
                "msi_dmmr": "MSS/pMMR",
                "braf": "wild_type",
                "cohort": "B",
                "subcohort": "B1",
                "specimen_role": "primary",
            },
        ]
    ).to_csv(development_manifest, index=False)

    canonical_rows = []
    for slide, patient in (("S-M04", "P0"), ("S-M07", "P1")):
        with h5py.File(feature_root / f"{slide}.h5", "r") as handle:
            features = handle["features"][:]
        labels = stability.assign_nearest(features, centroids)
        abundance = np.bincount(labels.astype(int), minlength=32) / len(labels)
        canonical_rows.extend(
            {
                "arm": "e0",
                "patient_id": patient,
                "prototype": prototype,
                "abundance": float(abundance[prototype]),
            }
            for prototype in range(32)
        )
    corrected_root = tmp_path / "corrected"
    (corrected_root / "profiles").mkdir(parents=True)
    (corrected_root / "analysis").mkdir()
    canonical_profiles = corrected_root / "profiles" / "patient_profiles_k32.parquet"
    pd.DataFrame(canonical_rows).to_parquet(canonical_profiles, index=False)
    structural = {
        "auc": None,
        "delta": None,
        "ci_low": None,
        "ci_high": None,
        "n": 2,
        "n_positive": 1,
        "p": 1.0,
        "estimable": False,
        "structural_p_policy": "p=1_in_fixed_32_family",
        "q": 1.0,
        "significant": False,
    }
    canonical_specificity = corrected_root / "analysis" / "specificity_k32.json"
    _write_json(
        canonical_specificity,
        {
            "component": "aim4_corrected_cap8192",
            "k": 32,
            "populations": {"A": 2, "D": 2},
            "protocol": {
                "bh_family": "fixed prototypes 0..31, including structural p=1 rows"
            },
            "prototypes": {
                str(prototype): {
                    "abundance": {"A": dict(structural), "D": dict(structural)}
                }
                for prototype in range(32)
            },
        },
    )
    artifacts = {}
    for path in (canonical_profiles, canonical_specificity):
        relative = path.relative_to(corrected_root).as_posix()
        record = stability.identity(path)
        record["path"] = relative
        artifacts[relative] = record
    completion = corrected_root / "numeric_complete.json"
    _write_json(
        completion,
        {
            "schema_version": 1,
            "component": "aim4_corrected_cap8192",
            "status": "numeric_completed_pathology_review_pending",
            "output_root": str(corrected_root),
            "artifacts": artifacts,
            "aggregate_sha256": hashlib.sha256(
                json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "validation": {
                "status": "PASS",
                "pathology_review": "PENDING_EXPLICIT_COMPLETED_FORMS",
            },
            "pathology_status": (
                "fresh corrected blinded packets sealed; completed review pending"
            ),
        },
    )
    verification = corrected_root / "numeric_verification.json"
    _write_json(
        verification,
        {
            "schema_version": 1,
            "component": "aim4_corrected_cap8192",
            "status": "PASS",
            "receipt_role": "immutable_numeric_verification_addendum",
            "output_root": str(corrected_root),
            "checks": {"deterministic_statistical_replay": "PASS"},
            "replay": {"status": "PASS"},
            "numeric_completion": stability.identity(completion),
            "numeric_aggregate_sha256": hashlib.sha256(
                json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )
    return stability.Inputs(
        vocab,
        sample_plan,
        feature_root,
        atlas_report,
        review_key,
        development_manifest,
        corrected_root,
        completion,
        verification,
        canonical_profiles,
        canonical_specificity,
    )


def test_partition_overlap_reports_split_and_merge_axes_separately():
    table = np.asarray([[5, 5, 0], [0, 0, 10]])
    result = stability.partition_overlap(table)
    assert result["anchor_recovery"] == pytest.approx(0.75)
    assert result["variant_purity"] == pytest.approx(1.0)
    assert result["symmetric_overlap"] == pytest.approx(6 / 7)


def test_seed_pair_matching_is_label_permutation_invariant():
    first = np.asarray([0, 0, 1, 1, 2, 2])
    second = np.asarray([2, 2, 0, 0, 1, 1])
    table = stability.contingency(first, second, 3, 3)
    assert stability.matched_accuracy(table) == 1.0


def test_anchor_mapping_records_overlap_and_prohibits_label_transfer():
    table = np.zeros((32, 4), dtype=int)
    table[17] = [6, 3, 1, 0]
    table[0, 0] = 4
    canonical = np.eye(32, 4)
    alternative = np.eye(4)
    result = stability.anchor_mapping(
        anchor=17,
        montage_id="M04",
        table=table,
        canonical_centroids=canonical,
        variant_centroids=alternative,
        montage_labels=np.asarray([0, 0, 0, 1]),
    )
    assert result["best_alternative_cluster"] == 0
    assert result["anchor_recall"] == pytest.approx(0.6)
    assert result["alternative_precision"] == pytest.approx(0.6)
    assert result["clusters_to_cover_80pct"] == [0, 1]
    assert result["reviewed_dominant_fraction"] == pytest.approx(0.75)
    assert "no_pathology_label_transfer" in result["interpretation"]


def test_exact_sample_projection_is_deterministic_and_content_hashed(tmp_path: Path):
    inputs = _fixture_inputs(tmp_path)
    vocab = stability.load_vocabulary(inputs)
    first, first_hash = stability.collect_projected_sample(inputs, vocab, progress=False)
    second, second_hash = stability.collect_projected_sample(inputs, vocab, progress=False)
    assert np.array_equal(first, second)
    assert first_hash == second_hash
    assert first.shape == (88, 64)


def test_actual_shaped_canonical_reprojection_accepts_only_boundary_mass_drift():
    # Counts observed on the immutable 397,263-tile atlas replay.  The training
    # counts came from PCA.fit_transform coordinates; the observed counts come
    # from the persisted PCA transform and frozen centroids.
    training = [
        5906, 12939, 19164, 14640, 20855, 7277, 12318, 5079,
        8430, 6878, 9268, 7047, 17965, 12241, 10565, 11139,
        10636, 9852, 12930, 10877, 11421, 21952, 11937, 14740,
        4715, 32166, 9892, 12881, 5354, 9579, 25351, 11269,
    ]
    reprojected = [
        5900, 12936, 19172, 14640, 20859, 7281, 12322, 5077,
        8431, 6875, 9269, 7047, 17964, 12238, 10555, 11131,
        10639, 9853, 12925, 10877, 11420, 21951, 11939, 14744,
        4716, 32169, 9885, 12888, 5353, 9584, 25355, 11268,
    ]
    labels = np.repeat(np.arange(32, dtype=np.int16), reprojected)
    same_seed_refit = {
        "variant": "k32_seed20260819",
        "ari_to_canonical_k32": 0.7135764973915966,
        "ami_to_canonical_k32": 0.8557251795614524,
    }
    result = stability.canonical_reconstruction_control(
        canonical_labels=labels,
        canonical_metadata={
            "n_sample_tiles": len(labels),
            "cluster_sizes": training,
        },
        sampled_feature_values_sha256=stability.CANONICAL_SAMPLE_VALUES_SHA256,
        same_seed_refit=same_seed_refit,
    )
    assert result["pass"] is True
    assert result["cluster_mass_total_variation"] == pytest.approx(
        0.00013089565350913628
    )
    assert result["maximum_cluster_mass_delta"] == pytest.approx(
        2.5172241059449283e-05
    )
    assert result["same_seed_refit"] == {
        "variant": "k32_seed20260819",
        "ari_to_frozen_canonical": pytest.approx(0.7135764973915966),
        "ami_to_frozen_canonical": pytest.approx(0.8557251795614524),
        "role": "sensitivity_variant_not_an_identity_control",
        "included_in_all_nine_biological_gate": True,
    }


def test_canonical_reprojection_rejects_hash_or_material_mass_drift():
    counts = np.full(32, 1000, dtype=int)
    labels = np.repeat(np.arange(32, dtype=np.int16), counts)
    refit = {
        "variant": "k32_seed20260819",
        "ari_to_canonical_k32": 1.0,
        "ami_to_canonical_k32": 1.0,
    }
    wrong_hash = stability.canonical_reconstruction_control(
        canonical_labels=labels,
        canonical_metadata={"n_sample_tiles": len(labels), "cluster_sizes": counts.tolist()},
        sampled_feature_values_sha256="0" * 64,
        same_seed_refit=refit,
    )
    assert wrong_hash["pass"] is False

    drifted = counts.copy()
    drifted[0] -= 100
    drifted[1] += 100
    wrong_mass = stability.canonical_reconstruction_control(
        canonical_labels=np.repeat(np.arange(32, dtype=np.int16), drifted),
        canonical_metadata={"n_sample_tiles": int(counts.sum()), "cluster_sizes": counts.tolist()},
        sampled_feature_values_sha256=stability.CANONICAL_SAMPLE_VALUES_SHA256,
        same_seed_refit=refit,
    )
    assert wrong_mass["cluster_mass_total_variation"] > (
        stability.CANONICAL_MAX_CLUSTER_MASS_TOTAL_VARIATION
    )
    assert wrong_mass["pass"] is False


def test_preflight_binds_review_key_and_all_feature_metadata(tmp_path: Path):
    inputs = _fixture_inputs(tmp_path)
    report = stability.inspect_inputs(inputs)
    assert report["status"] == "PASS"
    assert report["counts"] == {
        "sample_plan_slides": 2,
        "sample_plan_patients": 2,
        "sample_plan_groups": 1,
        "sampled_tiles": 88,
        "reviewed_anchor_tiles": 24,
        "variants": 9,
        "development_slides": 2,
        "development_patients_A": 2,
        "development_patients_D": 2,
        "matched_anchor_tests_per_family": 32,
    }
    payload = stability._json_bytes(report["feature_inventory"])
    assert hashlib.sha256(payload).hexdigest() == report["feature_inventory_sha256"]
    assert "evaluated separately" in report["design"]["claim_gate"]


def test_numeric_seal_requires_full_replay_addendum(tmp_path: Path):
    inputs = _fixture_inputs(tmp_path)
    receipt = json.loads(inputs.corrected_numeric_verification.read_text())
    receipt["replay"]["status"] = "SKIPPED_BY_FLAG"
    _write_json(inputs.corrected_numeric_verification, receipt)
    with pytest.raises(stability.StabilityError, match="full statistical replay"):
        stability.validate_corrected_aim4_numeric_seal(inputs)


def test_preflight_rejects_a_review_key_not_bound_by_final_atlas(tmp_path: Path):
    inputs = _fixture_inputs(tmp_path)
    frame = pd.read_csv(inputs.review_key)
    frame.loc[0, "tile_index"] = 13
    frame.to_csv(inputs.review_key, index=False)
    with pytest.raises(stability.StabilityError, match="does not match"):
        stability.inspect_inputs(inputs)


def test_new_output_root_is_absolute_exclusive_and_outside_inputs(tmp_path: Path):
    inputs = _fixture_inputs(tmp_path)
    output = tmp_path / "new-results"
    assert stability.new_output_root(output, inputs) == output
    output.mkdir()
    with pytest.raises(stability.StabilityError, match="already exists"):
        stability.new_output_root(output, inputs)
    with pytest.raises(stability.StabilityError, match="absolute"):
        stability.new_output_root(Path("relative"), inputs)


def test_exclusive_writer_refuses_overwrite(tmp_path: Path):
    path = tmp_path / "receipt.json"
    stability._write_json_once(path, {"first": True})
    with pytest.raises(FileExistsError, match="overwrite"):
        stability._write_json_once(path, {"second": True})


def test_atomic_publication_refuses_even_an_empty_existing_root(tmp_path: Path):
    stage = tmp_path / "stage"
    output = tmp_path / "output"
    stage.mkdir()
    output.mkdir()
    (stage / "evidence.txt").write_text("sealed")
    with pytest.raises(FileExistsError, match="overwrite"):
        stability._rename_noreplace(stage, output)
    assert (stage / "evidence.txt").read_text() == "sealed"
    assert output.is_dir()


@pytest.mark.parametrize("threads", [1, 18])
def test_real_kmeans_helper_is_deterministic_on_a_small_matrix(threads: int):
    rng = np.random.default_rng(7)
    z = rng.normal(size=(80, 4)).astype(np.float32)
    first = stability.fit_variants(
        z, k_values=(2, 3), seeds=(5, 6), n_init=2, threads=threads
    )
    second = stability.fit_variants(
        z, k_values=(2, 3), seeds=(5, 6), n_init=2, threads=threads
    )
    for key in first[0]:
        assert np.array_equal(first[0][key], second[0][key])
        assert np.array_equal(first[1][key], second[1][key])
        assert first[2][key] == second[2][key]
        assert set(first[2][key]) == {
            "k",
            "seed",
            "model_inertia",
            "replay_inertia_float64",
            "n_iter",
        }
        assert first[2][key]["model_inertia"] > 0.0
        assert first[2][key]["replay_inertia_float64"] == (
            stability.deterministic_replay_inertia(
                z, first[1][key], first[0][key]
            )
        )


def test_replay_inertia_is_a_separate_exact_float64_sum():
    z = np.asarray(
        [[1.0, 0.25], [0.5, -0.75], [-0.125, 1.5]], dtype=np.float32
    )
    centers = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1], dtype=np.int16)
    expected = float(
        np.sum((z - centers[labels.astype(int)]) ** 2, dtype=np.float64)
    )

    with stability.threadpool_limits(limits=1):
        first = stability.deterministic_replay_inertia(z, centers, labels)
    with stability.threadpool_limits(limits=18):
        second = stability.deterministic_replay_inertia(z, centers, labels)

    assert first == expected
    assert second == expected
    assert first == second


def test_structural_effect_rows_remain_in_fixed_32_family_with_null_json():
    rows = []
    for patient in range(6):
        for prototype in range(32):
            rows.append(
                {
                    "patient_id": f"P{patient}",
                    "prototype": prototype,
                    "is_mutant": 0,
                    "abundance": prototype / 100,
                }
            )
    table = stability.fixed_effect_family(pd.DataFrame(rows), n_bootstrap=5)
    assert table.prototype.tolist() == list(range(32))
    assert not table.estimable.any()
    assert (table.p == 1.0).all() and (table.q == 1.0).all()
    assert table.auc.isna().all() and table.ci_low.isna().all()
    assert not table.significant.any()
    sanitized = stability._sanitize_json(table.to_dict("records"))
    assert all(row["auc"] is None for row in sanitized)


def test_structural_p_one_rows_produce_the_expected_m32_q_shift():
    fixed = np.ones(32, dtype=float)
    fixed[0] = 0.001
    fixed_q = stability.benjamini_hochberg(fixed)[0]

    legacy_finite_only = fixed[:27]
    order = np.argsort(legacy_finite_only)
    ranked = legacy_finite_only[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    legacy_q = restored[0]

    assert fixed_q == pytest.approx(0.032)
    assert legacy_q == pytest.approx(0.027)
    assert fixed_q > legacy_q


def test_association_analysis_requires_estimability_for_claim_gate(tmp_path: Path):
    inputs = _fixture_inputs(tmp_path)
    canonical = pd.read_parquet(inputs.canonical_patient_profiles)
    blocks = []
    for variant in [
        "canonical_k32",
        *(f"k{k}_seed{seed}" for k in stability.K_VALUES for seed in stability.CLUSTER_SEEDS),
    ]:
        block = canonical.copy()
        block.insert(0, "variant", variant)
        blocks.append(block)
    result = stability.association_analysis(
        inputs, pd.concat(blocks, ignore_index=True), n_bootstrap=5
    )
    assert result["canonical_effect_reconstruction"]["pass"] is True
    assert result["canonical_effect_reconstruction"]["structural_rows_checked"] == 64
    for anchor in ("17", "28"):
        assert result["claim_stability"][anchor]["all_variants_pass"] is False
        assert len(result["claim_stability"][anchor]["failing_variants"]) == 9
    assert all(
        row["auc_A"] is None and not row["estimable_A"]
        for row in result["association_effects"]
    )


def test_tiny_append_only_campaign_and_structural_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    inputs = _fixture_inputs(tmp_path)

    def fake_fit(z, **kwargs):
        labels = {}
        centers = {}
        info = {}
        for k in stability.K_VALUES:
            for seed in stability.CLUSTER_SEEDS:
                key = f"k{k}_seed{seed}"
                if k <= 32:
                    center = stability.load_vocabulary(inputs).centroids[:k].copy()
                else:
                    center = np.vstack(
                        [
                            stability.load_vocabulary(inputs).centroids,
                            np.zeros((k - 32, 64), dtype=np.float32),
                        ]
                    )
                centers[key] = center
                labels[key] = stability.assign_nearest(z, center).astype(np.int16)
                replay_inertia = stability.deterministic_replay_inertia(
                    z, center, labels[key]
                )
                info[key] = {
                    "k": k,
                    "seed": seed,
                    # Deliberately different: the model-native diagnostic is
                    # not the replay identity.
                    "model_inertia": replay_inertia + float(k),
                    "replay_inertia_float64": replay_inertia,
                    "n_iter": 1,
                }
        return labels, centers, info

    def fake_association(inputs, profiles, **kwargs):
        return {
            "multiplicity": {"family_size": 32},
            "predeclared_gate": {"grid": "nine"},
            "canonical_effect_reconstruction": {"pass": True},
            "claim_stability": {},
            "association_effects": [],
        }

    monkeypatch.setattr(stability, "fit_variants", fake_fit)
    monkeypatch.setattr(stability, "association_analysis", fake_association)
    _, fixture_sample_hash = stability.collect_projected_sample(
        inputs, stability.load_vocabulary(inputs), progress=False
    )
    monkeypatch.setattr(
        stability, "CANONICAL_SAMPLE_VALUES_SHA256", fixture_sample_hash
    )
    output = tmp_path / "aim4-stability-output"
    stability.run_campaign(inputs, output, threads=1)
    receipt = stability.verify_output(output, replay_input=True)
    assert receipt["status"] == "PASS"
    assert receipt["checks"]["statistical_recomputation"] == "PASS"
    assert receipt["checks"]["exact_feature_and_assignment_replay"] == "PASS"
    assert receipt["checks"]["deterministic_inertia_replay"] == "PASS"
    assert (output / stability.COMPLETION_NAME).is_file()
    assert not any(tmp_path.glob(".aim4-stability-output.staging-*"))

    results_path = output / stability.RESULTS_NAME
    results = json.loads(results_path.read_text())
    assert results["schema_version"] == 2
    results["variant_metrics"][0]["inertia"] = results["variant_metrics"][0][
        "model_inertia"
    ]
    _write_json(results_path, results)
    completion_path = output / stability.COMPLETION_NAME
    completion = json.loads(completion_path.read_text())
    completion["payload_inventory"] = stability.relative_inventory(
        output,
        excluded={stability.COMPLETION_NAME, stability.VERIFICATION_NAME},
    )
    completion["results_sha256"] = stability.sha256_file(results_path)
    _write_json(completion_path, completion)
    with pytest.raises(stability.StabilityError, match="ambiguous legacy inertia"):
        stability.verify_output(output, replay_input=False)


def test_material_snapshot_includes_canonical_atlas_implementation():
    relative = {path.relative_to(stability.REPO).as_posix() for path in stability.material_source_files()}
    assert {
        "tools/aim4_vocab_stability.py",
        "aim4_morphologic_atlas.py",
        "aim4_morphologic_atlas_base.py",
        "src/oceanpath/aim1/atlas.py",
        "pyproject.toml",
        "uv.lock",
    } == relative
