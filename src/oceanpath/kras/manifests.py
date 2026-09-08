"""Manifest writing and shared-split summaries for exploratory KRAS tasks.

Task builders own their population and label rules. This module handles the
common output contract: canonical columns, slide ordering, manifest validation,
and subsets of the first seed's existing fold assignment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oceanpath.kras import study
from oceanpath.splitting.core import derive_subset_splits


def write_task_manifest_and_splits(
    task: str, frame: pd.DataFrame, *, split_root: Path
) -> tuple[Path, pd.DataFrame]:
    """Write a task manifest and return its path and labeled validation flags.

    Split derivation preserves the master fold assignment and its existing
    stale-output checks. It never draws a new assignment.
    """
    out = frame[study.MANIFEST_COLUMNS].sort_values("slide_id").reset_index(drop=True)
    study.assert_manifest_invariants(out, task)
    path = study.dev_manifest_path(task)
    out.to_csv(path, index=False)

    assignment = study.split_name(study.SEEDS[0])
    derived = derive_subset_splits(
        master_splits_dir=split_root / "colon_kras_master" / assignment,
        manifest_csv=path,
        output_dir=split_root / study.data_name(task) / assignment,
        filename_column="slide_id",
    )
    splits = pd.read_parquet(derived)
    validation_columns = [f"val_fold_{fold}" for fold in range(study.N_FOLDS)]
    labeled_splits = splits[["slide_id", *validation_columns]].merge(
        out[["slide_id", "patient_id", "target_label"]], on="slide_id"
    )
    return path, labeled_splits


def inner_validation_patient_support(splits: pd.DataFrame, label: int) -> list[int]:
    """Count distinct patients of one class in each inner-validation fold."""
    return [
        int(
            splits.loc[
                (splits[f"val_fold_{fold}"] == 1) & (splits["target_label"] == label),
                "patient_id",
            ].nunique()
        )
        for fold in range(study.N_FOLDS)
    ]
