"""KRAS allele-study contracts: frozen label rules, shared splits, and the
OOF-CV early-stopping protocol (Experimental_Setup.md §5, §6.4).

The synthetic frames reproduce every tricky pre-registered case by name:
multi-token slides (G12V;G12C, G13D;R164Q, A146T;A59T;G13D), the variant-less
mutant (RIH SL-102), and the B2 multi-assignment exclusions.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── §5 label rules ────────────────────────────────────────────────────────────


def test_subvariant_token_parsing_never_uses_substrings():
    from oceanpath.kras.labels import has_token, subvariant_tokens

    assert subvariant_tokens("G12V;G12C") == ["G12V", "G12C"]
    assert subvariant_tokens(" G13D ; R164Q ") == ["G13D", "R164Q"]
    assert subvariant_tokens(None) == []
    assert subvariant_tokens(float("nan")) == []
    assert has_token("G12V;G12C", "G12C")
    assert has_token("A146T;A59T;G13D", "G13D")
    # Membership, not substring: G12 is not a token of G12D.
    assert not has_token("G12D", "G12")


def test_codon12_and_a146_are_exact_codon_matches():
    from oceanpath.kras.labels import has_a146, is_codon12

    assert is_codon12("G12S")
    assert is_codon12("G13D;G12V")
    assert not is_codon12("G13D")
    assert not is_codon12("G128X")  # codon 128 is not codon 12
    assert has_a146("A146T;A59T;G13D")
    assert not has_a146("A59T")


def test_kras_class_priority_and_b2_membership():
    from oceanpath.kras.labels import b2_class_set, kras_class

    assert kras_class("wild_type", None) == "wild_type"
    assert kras_class("mutant", "Q61H") == "other_mutant"
    assert kras_class("mutant", "G13D;G12D") == "G12D"  # fixed priority for strata
    assert b2_class_set("G12D") == {"G12D"}
    assert b2_class_set("Q61H") == {"other"}
    assert b2_class_set("G12D;G13D") == {"G12D", "G13D"}  # multi → excluded
    assert b2_class_set("G12C;G12V") == {"G12V", "other"}  # G12C lives in "other"


def _dev_frame() -> pd.DataFrame:
    rows = [
        # slide, patient, kras, subvariant
        ("s01", "p01", "mutant", "G12D"),
        ("s02", "p02", "mutant", "G12V"),
        ("s03", "p03", "mutant", "G12C;G12V"),  # positive for BOTH S1 and X1
        ("s04", "p04", "mutant", "G13D;R164Q"),
        ("s05", "p05", "mutant", "A146T;A59T;G13D"),
        ("s06", "p06", "mutant", "Q61H"),
        ("s07", "p07", "mutant", "G12D;G13D"),  # B2 multi-assignment
        ("s08", "p08", "mutant", None),  # SL-102: no variant recorded
        ("s09", "p09", "wild_type", None),
        ("s10", "p10", "wild_type", None),
    ]
    frame = pd.DataFrame(rows, columns=["slide_id", "patient_id", "kras", "kras_subvariant"])
    frame["cohort_group"] = ["SR386", "TCGA"] * 5
    frame["subcohort"] = frame["cohort_group"]
    frame["specimen_role"] = "primary"
    frame["msi_dmmr"] = "MSS/pMMR"
    frame["tumor_site_group"] = "Colon"
    frame["metastatic_site_group"] = None
    return frame


def test_registry_p1_partition_matches_the_frozen_rules():
    from oceanpath.kras.labels import add_kras_class_columns
    from oceanpath.kras.registry import EXPERIMENTS

    dev = add_kras_class_columns(_dev_frame())
    p1 = EXPERIMENTS["p1"].select_and_label(dev)

    # Variant-known mutants only: WT out, the variant-less mutant (SL-102) out.
    assert set(p1["slide_id"]) == {"s01", "s02", "s03", "s04", "s05", "s06", "s07"}
    positives = set(p1.loc[p1["target_label"] == 1, "slide_id"])
    assert positives == {"s01", "s07"}  # token presence, incl. G12D;G13D


def test_registry_multi_token_slides_are_positive_in_multiple_models():
    from oceanpath.kras.labels import add_kras_class_columns
    from oceanpath.kras.registry import EXPERIMENTS

    dev = add_kras_class_columns(_dev_frame())
    s1 = EXPERIMENTS["s1"].select_and_label(dev)
    x1 = EXPERIMENTS["x1"].select_and_label(dev)
    s2 = EXPERIMENTS["s2"].select_and_label(dev)

    assert s1.loc[s1["slide_id"] == "s03", "target_label"].item() == 1
    assert x1.loc[x1["slide_id"] == "s03", "target_label"].item() == 1
    assert s2.loc[s2["slide_id"] == "s04", "target_label"].item() == 1
    assert s2.loc[s2["slide_id"] == "s05", "target_label"].item() == 1


def test_registry_b1_keeps_the_variantless_mutant_and_wild_types():
    from oceanpath.kras.labels import add_kras_class_columns
    from oceanpath.kras.registry import EXPERIMENTS

    dev = add_kras_class_columns(_dev_frame())
    b1 = EXPERIMENTS["b1"].select_and_label(dev)
    assert "s08" in set(b1["slide_id"])
    assert len(b1) == 10
    assert b1.loc[b1["slide_id"] == "s08", "target_label"].item() == 1
    assert b1.loc[b1["slide_id"] == "s09", "target_label"].item() == 0


def test_registry_b2_excludes_multi_assignment_and_orders_classes():
    from oceanpath.kras.labels import B2_CLASS_NAMES, add_kras_class_columns
    from oceanpath.kras.registry import EXPERIMENTS

    dev = add_kras_class_columns(_dev_frame())
    b2 = EXPERIMENTS["b2"].select_and_label(dev)

    # Multi-assignment across the 4 B2 classes excludes: s07 (G12D;G13D, two
    # named), s03 (G12C;G12V — the AG-4008 pattern that the pre-registered
    # count 361 requires excluding, since G12C lives in "other"), and by the
    # same rule s04/s05 (named token + non-named token). They all REMAIN
    # positives in their allele models — only the 4-way benchmark drops them.
    assert set(b2["slide_id"]) == {"s01", "s02", "s06"}
    label_of = dict(zip(b2["slide_id"], b2["target_label"], strict=True))
    assert B2_CLASS_NAMES[label_of["s01"]] == "G12D"
    assert B2_CLASS_NAMES[label_of["s02"]] == "G12V"
    assert B2_CLASS_NAMES[label_of["s06"]] == "other"


def test_registry_codon_controls_and_head_to_head():
    from oceanpath.kras.labels import add_kras_class_columns
    from oceanpath.kras.registry import EXPERIMENTS

    dev = add_kras_class_columns(_dev_frame())
    m1 = EXPERIMENTS["m1"].select_and_label(dev)
    assert set(m1["slide_id"]) == {"s01", "s02", "s03", "s07"}  # any G12x token
    m3 = EXPERIMENTS["m3"].select_and_label(dev)
    assert set(m3["slide_id"]) == {"s01", "s02", "s03", "s07"}
    assert set(m3.loc[m3["target_label"] == 1, "slide_id"]) == {"s01", "s07"}

    ambiguous = dev.copy()
    ambiguous.loc[ambiguous["slide_id"] == "s01", "kras_subvariant"] = "G12D;G12V"
    m3_guarded = EXPERIMENTS["m3"].select_and_label(ambiguous)
    assert "s01" not in set(m3_guarded["slide_id"])


def test_registry_r4_emits_per_head_labels():
    from oceanpath.kras.labels import add_kras_class_columns
    from oceanpath.kras.registry import EXPERIMENTS

    dev = add_kras_class_columns(_dev_frame())
    r4 = EXPERIMENTS["r4"].select_and_label(dev)
    row = r4[r4["slide_id"] == "s03"].iloc[0]
    assert (row["label_g12v"], row["label_g12c"], row["label_g12d"]) == (1, 1, 0)


def test_manifest_invariants_reject_conflicting_patient_labels():
    from oceanpath.kras.study import assert_manifest_invariants

    good = pd.DataFrame(
        {
            "slide_id": ["a", "b"],
            "patient_id": ["p1", "p2"],
            "target_label": [0, 1],
        }
    )
    assert_manifest_invariants(good, "good")

    conflicted = pd.DataFrame(
        {
            "slide_id": ["a", "b"],
            "patient_id": ["p1", "p1"],
            "target_label": [0, 1],
        }
    )
    with pytest.raises(AssertionError, match="conflicting labels"):
        assert_manifest_invariants(conflicted, "conflicted")


# ── Shared splits: OOF layout + subset derivation ─────────────────────────────


def _master_split_fixture(tmp_path: Path, n_patients: int = 40):
    from oceanpath.splitting import SplitConfig, generate_splits

    rows = []
    for index in range(n_patients):
        # Two slides for a few patients; composite stratification key.
        strata = f"C{index % 2}|K{index % 4}"
        rows.append(
            {
                "slide_id": f"s{index:03d}",
                "patient_id": f"p{index // 2:03d}" if index < 8 else f"p{index:03d}",
                "strat": strata,
                "target_label": index % 2,
            }
        )
    manifest = tmp_path / "master.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    spec = SplitConfig(
        scheme="oof_kfold",
        name="master",
        csv_path=str(manifest),
        output_dir=str(tmp_path / "master_splits"),
        filename_column="slide_id",
        label_column="strat",
        group_column="patient_id",
        n_folds=4,
        seed=7,
        val_ratio=0.2,
    )
    generate_splits(spec)
    return manifest, tmp_path / "master_splits", pd.DataFrame(rows)


def test_oof_kfold_layout_has_no_leakage(tmp_path):
    _, splits_dir, rows = _master_split_fixture(tmp_path)
    splits = pd.read_parquet(splits_dir / "splits.parquet")

    assert set(splits["fold"].unique()) == set(range(4))
    for fold in range(4):
        val_col = f"val_fold_{fold}"
        # Test fold never carries a val flag; roles are patient-pure.
        assert not ((splits["fold"] == fold) & (splits[val_col] == 1)).any()
        role = pd.Series("train", index=splits.index)
        role[splits[val_col] == 1] = "val"
        role[splits["fold"] == fold] = "test"
        assert (role.groupby(splits["group_id"]).nunique() == 1).all()
        assert (role == "val").sum() > 0


def test_derive_subset_splits_preserves_the_assignment(tmp_path):
    from oceanpath.splitting.core import derive_subset_splits

    manifest, splits_dir, rows = _master_split_fixture(tmp_path)
    subset_rows = rows.iloc[::2]
    subset_csv = tmp_path / "exp.csv"
    subset_rows.to_csv(subset_csv, index=False)

    out = derive_subset_splits(splits_dir, subset_csv, tmp_path / "derived")
    derived = pd.read_parquet(out)
    master = pd.read_parquet(splits_dir / "splits.parquet")

    assert set(derived["slide_id"]) == set(subset_rows["slide_id"])
    merged = derived.merge(master, on="slide_id", suffixes=("_d", "_m"))
    assert (merged["fold_d"] == merged["fold_m"]).all()  # folds copied, not re-drawn
    for fold in range(4):
        assert (merged[f"val_fold_{fold}_d"] == merged[f"val_fold_{fold}_m"]).all()
    assert (tmp_path / "derived" / ".integrity_hash").is_file()

    # Idempotent when current; a non-subset manifest must fail.
    derive_subset_splits(splits_dir, subset_csv, tmp_path / "derived")
    alien = pd.concat(
        [subset_rows, pd.DataFrame([{"slide_id": "zzz", "patient_id": "px", "strat": "a"}])]
    )
    alien_csv = tmp_path / "alien.csv"
    alien.to_csv(alien_csv, index=False)
    with pytest.raises(ValueError, match="not a subset"):
        derive_subset_splits(splits_dir, alien_csv, tmp_path / "derived2")


# ── Patient-level aggregation + seed averaging ────────────────────────────────


def test_to_patient_logits_averages_slide_logits():
    from oceanpath.kras.study import to_patient_logits

    manifest = pd.DataFrame({"slide_id": ["a", "b", "c"], "patient_id": ["p1", "p1", "p2"]})
    slides = pd.DataFrame(
        {
            "slide_id": ["a", "b", "c"],
            "label": [1, 1, 0],
            "prob_1": [0.8, 0.6, 0.3],
        }
    )
    patients = to_patient_logits(slides, manifest)
    assert len(patients) == 2
    expected = np.mean([np.log(0.8 / 0.2), np.log(0.6 / 0.4)])
    got = patients.loc[patients["patient_id"] == "p1", "logit"].item()
    assert got == pytest.approx(expected, rel=1e-6)


def test_seed_averaging_requires_identical_patient_sets():
    from oceanpath.kras.study import seed_averaged_patients

    base = pd.DataFrame(
        {"patient_id": ["p1", "p2"], "label": [1, 0], "logit": [1.0, -1.0], "n_slides": [1, 1]}
    )
    shifted = base.assign(logit=[3.0, 1.0])
    pooled = seed_averaged_patients({42: base, 43: shifted})
    assert pooled.loc[pooled["patient_id"] == "p1", "logit"].item() == pytest.approx(2.0)

    with pytest.raises(SystemExit, match="different patient set"):
        seed_averaged_patients({42: base, 43: shifted.iloc[:1]})


def test_bootstrap_patient_auroc_bounds_are_ordered():
    from oceanpath.kras.study import bootstrap_patient_auroc

    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 30)
    patients = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(60)],
            "label": labels,
            "logit": labels * 1.5 + rng.normal(0, 1, 60),
            "n_slides": 1,
        }
    )
    result = bootstrap_patient_auroc(patients, n_bootstrap=100, seed=1)
    assert result["n_patients"] == 60
    assert result["ci_low"] <= result["auroc"] <= result["ci_high"]


# ── The early-stopping small-class rule ───────────────────────────────────────


class _FakeValDataset:
    def __init__(self, slide_ids, labels):
        self.slide_ids = slide_ids
        self.labels_list = labels


class _FakeDataModule:
    def __init__(self, slide_ids, labels, patient_map):
        self.val_dataset = _FakeValDataset(slide_ids, labels)
        self.patient_of_slide = patient_map


def test_small_class_rule_counts_positive_patients_not_slides():
    from oceanpath.workflows.training import _min_val_positive_patients

    # Two positive SLIDES that belong to one positive PATIENT.
    datamodule = _FakeDataModule(
        ["a", "b", "c", "d"],
        [1, 1, 0, 0],
        {"a": "p1", "b": "p1", "c": "p2", "d": "p3"},
    )
    assert _min_val_positive_patients(datamodule, num_classes=2) == 1

    multiclass = _FakeDataModule(
        ["a", "b", "c", "d"],
        [0, 1, 2, 3],
        {"a": "p1", "b": "p2", "c": "p3", "d": "p4"},
    )
    assert _min_val_positive_patients(multiclass, num_classes=4) == 1


def test_inverse_prevalence_weights_come_from_the_training_fold():
    from oceanpath.workflows.training import _inverse_prevalence_weights

    class _TrainDataset:
        @staticmethod
        def get_label_counts():
            return {0: 30, 1: 10}

    weights = _inverse_prevalence_weights(_TrainDataset(), num_classes=2)
    assert weights == pytest.approx([40 / 60, 40 / 20])
    # BCE pos_weight derived downstream: w1/w0 = N_neg/N_pos.
    assert weights[1] / weights[0] == pytest.approx(3.0)

    class _Degenerate:
        @staticmethod
        def get_label_counts():
            return {0: 30}

    with pytest.raises(ValueError, match="no examples"):
        _inverse_prevalence_weights(_Degenerate(), num_classes=2)


# ── Patient AUROC monitor in the Lightning module ─────────────────────────────


def _tiny_module(num_classes: int):
    from oceanpath.training.lightning import MILTrainModule

    return MILTrainModule(
        arch="abmil",
        in_dim=8,
        num_classes=num_classes,
        model_cfg={"embed_dim": 8, "attn_dim": 8},
        loss_type="bce" if num_classes == 1 else "ce",
    )


def test_patient_auroc_monitor_aggregates_by_mean_logit():
    module = _tiny_module(num_classes=1)
    module.patient_map = {"a": "p1", "b": "p1", "c": "p2", "d": "p3"}
    rows = [
        {"slide_id": "a", "label": 1, "prob_1": 0.9},
        {"slide_id": "b", "label": 1, "prob_1": 0.7},
        {"slide_id": "c", "label": 0, "prob_1": 0.2},
        {"slide_id": "d", "label": 1, "prob_1": 0.6},
    ]
    assert module._patient_level_auroc(rows) == pytest.approx(1.0)

    # Degenerate epoch (single class) must not crash the monitor.
    assert module._patient_level_auroc(rows[:2]) == 0.5
    assert module._patient_level_auroc([]) == 0.5


def test_patient_auroc_monitor_supports_multiclass_macro():
    module = _tiny_module(num_classes=4)
    module.patient_map = None
    rows = []
    for index, label in enumerate([0, 1, 2, 3] * 3):
        probs = {f"prob_{c}": 0.1 for c in range(4)}
        probs[f"prob_{label}"] = 0.7
        rows.append({"slide_id": f"s{index}", "label": label, **probs})
    assert module._patient_level_auroc(rows) == pytest.approx(1.0)


def test_bce_loss_flattens_single_logit_heads():
    import torch

    module = _tiny_module(num_classes=1)
    logits = torch.tensor([[2.0], [-1.0], [0.5]])
    labels = torch.tensor([1, 0, 1])
    loss = module._compute_loss(logits, labels)
    assert loss.ndim == 0
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(-1), labels.float()
    )
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


# ── End-to-end: the study protocol through the real CLI ───────────────────────


def test_oof_cv_with_patient_monitor_and_es_fallback_end_to_end(tmp_path):
    """One fold-complete oof_kfold run exercising the §6.4 protocol pieces:
    patient column, val/patient_auroc monitor, single-logit BCE with
    per-fold inverse-prevalence weights, and the small-class fallback
    (forced via an unreachable positive threshold)."""
    import h5py

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    n_slides = 24
    for index in range(n_slides):
        slide_id = f"slide_{index}"
        with h5py.File(feature_dir / f"{slide_id}.h5", "w") as handle:
            handle.create_dataset("features", data=rng.standard_normal((6, 16)).astype(np.float32))
            handle.create_dataset("coords", data=rng.integers(0, 99, (6, 2)).astype(np.int64))
        rows.append(
            {
                "slide_id": slide_id,
                "patient_id": f"pat_{index // 2}",  # two slides per patient
                "label": (index // 2) % 2,
                "strat": f"S{(index // 2) % 2}",
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    from oceanpath.splitting import SplitConfig, generate_splits

    splits_dir = tmp_path / "splits_out"
    generate_splits(
        SplitConfig(
            scheme="oof_kfold",
            name="tiny_oof",
            csv_path=str(manifest),
            output_dir=str(splits_dir),
            filename_column="slide_id",
            label_column="strat",
            group_column="patient_id",
            n_folds=2,
            seed=3,
            val_ratio=0.25,
        )
    )

    train_dir = tmp_path / "train-output"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "runtime.verify_environment=false",
        "platform.num_workers=0",
        "platform.accelerator=cpu",
        f"data.csv_path={manifest}",
        f"data.feature_h5_dir={feature_dir}",
        "data.filename_column=slide_id",
        "data.label_columns=[label]",
        "data.patient_id_column=patient_id",
        "encoder.feature_dim=16",
        f"+splits.output_dir={splits_dir}",
        "splits.scheme=oof_kfold",
        "splits.n_folds=2",
        "model.embed_dim=8",
        "model.attn_dim=8",
        "training.verify_splits=false",
        "training.max_epochs=3",
        "training.warmup_epochs=0",
        "training.loss_type=bce",
        "training.auto_class_weights=inverse_prevalence",
        "training.monitor_metric=val/patient_auroc",
        "training.monitor_mode=max",
        # Unreachable threshold → the small-class fallback fires on every fold.
        "training.es_min_val_positives=999",
        "training.fixed_epoch_budget=1",
        "training.skip_finalize=true",
        "training.collect_embeddings=false",
        f"train_dir={train_dir}",
    ]
    environment = {
        **os.environ,
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        # The child must import THIS repo's source, whatever venv is active.
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
    }
    completed = subprocess.run(
        command, cwd=REPO_ROOT, env=environment, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    assert "EARLY STOPPING DISABLED" in completed.stdout + completed.stderr

    oof = pd.read_parquet(train_dir / "oof_predictions.parquet")
    # Outer test folds partition the cohort exactly once.
    assert sorted(oof["slide_id"]) == sorted(r["slide_id"] for r in rows)
    assert "prob_1" in oof.columns  # single-logit sigmoid path

    for fold in range(2):
        metrics = json.loads((train_dir / f"fold_{fold}" / "fold_metrics.json").read_text())
        assert metrics["early_stopping_disabled"].startswith("inner_val_positive_patients")
        assert metrics["fixed_epoch_budget"] == 1
        assert metrics["best_epoch"] == 1
        assert metrics["best_checkpoint"].endswith("last.ckpt")
        assert "val/patient_auroc" in metrics
