#!/usr/bin/env python3
"""Granularity-ladder additions (2026-08-15, study owner directive).

    e4  Exon level:       exon-2 (codon 12 or 13 token, G12*/G13*) vs
                          non-exon-2 mutants (Q61*, K117*, A146*, ...)
    e5  Functional group: G12D-or-G12V-or-G13D vs every other mutant

Population: variant-known DEV mutants; token rules; shared-assignment splits.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.kras import study  # noqa: E402
from oceanpath.kras.labels import mutant_with_variant_mask, subvariant_tokens  # noqa: E402
from oceanpath.kras.manifests import (  # noqa: E402
    inner_validation_patient_support,
    write_task_manifest_and_splits,
)

EXON2 = re.compile(r"G1[23][A-Z]")  # codons 12 and 13 = KRAS exon 2


def build() -> None:
    master = pd.read_csv(study.master_manifest_path())
    mutants = master[mutant_with_variant_mask(master)].copy()

    tasks = {}
    e4 = mutants.copy()
    e4["target_label"] = e4["kras_subvariant"].map(
        lambda v: int(any(EXON2.fullmatch(t) for t in subvariant_tokens(v)))
    )
    tasks["e4"] = ("exon-2 vs non-exon-2", e4)

    e5 = mutants.copy()
    e5["target_label"] = e5["kras_subvariant"].map(
        lambda v: int(any(t in ("G12D", "G12V", "G13D") for t in subvariant_tokens(v)))
    )
    tasks["e5"] = ("G12D/V/G13D vs other mutants", e5)

    for name, (title, df) in tasks.items():
        print(f"== {name}: {title}")
        print(
            df.groupby("target_label").agg(
                slides=("slide_id", "size"), patients=("patient_id", "nunique")
            )
        )
        print(
            f"   per-cohort: {pd.crosstab(df['cohort_group'], df['target_label']).to_dict('index')}"
        )
        neg_tokens = sorted(
            {
                t
                for v in df.loc[df.target_label == 0, "kras_subvariant"]
                for t in subvariant_tokens(v)
            }
        )
        print(f"   negative-class tokens: {neg_tokens}")

        path, splits = write_task_manifest_and_splits(name, df, split_root=REPO / "outputs/splits")
        pos_support = inner_validation_patient_support(splits, 1)
        print(f"   wrote {path.name}; inner-val POSITIVE patients per fold: {pos_support}\n")


if __name__ == "__main__":
    build()
