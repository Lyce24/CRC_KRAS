"""Exploratory task builders preserve labels and the shared fold assignment."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from oceanpath.kras import study
from oceanpath.kras.manifests import (
    inner_validation_patient_support,
    write_task_manifest_and_splits,
)


@pytest.fixture
def master_manifest(tmp_path, monkeypatch):
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    monkeypatch.setattr(study, "MANIFEST_ROOT", manifest_root)
    variants = [
        "G12D",
        "G12D",
        "G12V",
        "G12C",
        "G13D",
        "Q61H",
        "G12D;G13D",
        "G12C;G12V",
        "G13D;R164Q",
        "G128X",
        None,
        None,
        "G12D",
    ]
    rows = []
    for index, variant in enumerate(variants):
        row = dict.fromkeys(study.MANIFEST_COLUMNS, "")
        row.update(
            slide_id=f"s{index:02d}",
            patient_id=f"p{max(0, index - 1):02d}",
            kras={11: "wild_type", 12: "unknown"}.get(index, "mutant"),
            kras_subvariant=variant,
            target_label=0,
            cohort_group="TCGA",
        )
        rows.append(row)
    frame = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
    frame.to_csv(study.master_manifest_path(), index=False)

    splits = frame[["slide_id", "patient_id"]].rename(columns={"patient_id": "group_id"})
    splits["fold"] = splits["group_id"].map(lambda patient: int(patient[1:]) % study.N_FOLDS)
    for fold in range(study.N_FOLDS):
        splits[f"val_fold_{fold}"] = (splits["fold"] != fold).astype(int)
    destination = tmp_path / "outputs/splits/colon_kras_master" / study.split_name(study.SEEDS[0])
    destination.mkdir(parents=True)
    splits.to_parquet(destination / "splits.parquet", index=False)
    return splits


@pytest.mark.parametrize(
    ("builder", "expected"),
    [
        (
            "build_codon_task_manifests",
            {
                "e1": {"s00": 0, "s01": 0, "s02": 0, "s03": 1, "s04": 2, "s08": 2},
                "e2": dict(
                    zip(
                        [f"s{i:02d}" for i in range(10)],
                        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                        strict=True,
                    )
                ),
                "e3": dict(
                    zip(
                        [f"s{i:02d}" for i in range(10)],
                        [1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
                        strict=True,
                    )
                ),
            },
        ),
        (
            "build_granularity_manifests",
            {
                "e4": dict(
                    zip(
                        [f"s{i:02d}" for i in range(10)],
                        [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                        strict=True,
                    )
                ),
                "e5": dict(
                    zip(
                        [f"s{i:02d}" for i in range(10)],
                        [1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
                        strict=True,
                    )
                ),
            },
        ),
    ],
)
def test_task_builders_preserve_labels_and_master_splits(
    tmp_path, monkeypatch, master_manifest, builder, expected
):
    source = Path(__file__).resolve().parents[2] / "tools" / f"{builder}.py"
    spec = importlib.util.spec_from_file_location(builder, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO", tmp_path)
    module.build()

    for task, labels in expected.items():
        manifest = pd.read_csv(study.dev_manifest_path(task))
        assert list(manifest.columns) == study.MANIFEST_COLUMNS
        assert manifest["slide_id"].tolist() == sorted(labels)
        assert dict(zip(manifest["slide_id"], manifest["target_label"], strict=True)) == labels
        split_path = (
            tmp_path
            / "outputs/splits"
            / study.data_name(task)
            / study.split_name(study.SEEDS[0])
            / "splits.parquet"
        )
        derived = pd.read_parquet(split_path)
        preserved = master_manifest[master_manifest["slide_id"].isin(labels)].reset_index(drop=True)
        pd.testing.assert_frame_equal(derived, preserved)


def test_validation_support_counts_patients_once():
    splits = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3"],
            "target_label": [1, 1, 0, 1],
            "val_fold_0": [1, 1, 1, 0],
            "val_fold_1": [0, 0, 1, 0],
            "val_fold_2": [0, 0, 0, 1],
            "val_fold_3": [0, 0, 0, 0],
            "val_fold_4": [0, 0, 0, 0],
        }
    )
    assert inner_validation_patient_support(splits, 1) == [1, 0, 1, 0, 0]


def test_conflicting_patient_labels_fail_before_writing(tmp_path, master_manifest):
    frame = pd.read_csv(study.master_manifest_path())
    frame = frame[frame["slide_id"].isin(["s00", "s01"])].copy()
    frame["target_label"] = [0, 1]
    with pytest.raises(AssertionError, match="conflicting labels"):
        write_task_manifest_and_splits("invalid", frame, split_root=tmp_path / "outputs/splits")
    assert not study.dev_manifest_path("invalid").exists()
    assert not (tmp_path / "outputs/splits/colon_kras_invalid").exists()
