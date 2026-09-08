#!/usr/bin/env python3
"""Exploratory codon-granularity tasks (2026-08-15, study owner directive).

Three DEV tasks over the feature-backed master population, sharing the frozen
fold assignment (derived subsets — paired with every registry experiment):

    e1  3-class: G12D/G12V vs other-codon-12 vs G13D
        population = variant-known mutants carrying a codon-12 or G13D token;
        slides spanning >1 class (e.g. G12C;G12V, G12D;G13D) are excluded.
        Extra non-covered tokens (e.g. R164Q beside G13D) do not exclude.
    e2  binary: G12D-or-G12V vs every other variant-known mutant
    e3  binary: codon-12 vs non-codon-12 (variant-known mutants)

Prints the class distributions, writes crc_kras_dev_e{1,2,3}.csv, and derives
outputs/splits/colon_kras_e{1,2,3}/<seed42 assignment>.
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

CODON12 = re.compile(r"G12[A-Z]")

E1_CLASSES = ("g12d_v", "other_codon12", "g13d")


def token_classes(sub: object) -> set[int]:
    tokens = subvariant_tokens(sub)
    classes: set[int] = set()
    if any(t in ("G12D", "G12V") for t in tokens):
        classes.add(0)
    if any(CODON12.fullmatch(t) and t not in ("G12D", "G12V") for t in tokens):
        classes.add(1)
    if "G13D" in tokens:
        classes.add(2)
    return classes


def build() -> None:
    master = pd.read_csv(study.master_manifest_path())
    mutants = master[mutant_with_variant_mask(master)].copy()
    print(
        f"variant-known DEV mutants: {len(mutants)} slides / "
        f"{mutants['patient_id'].nunique()} patients\n"
    )

    tasks: dict[str, pd.DataFrame] = {}

    # e1 — 3-class granularity
    classes = mutants["kras_subvariant"].map(token_classes)
    covered = mutants[classes.map(len) == 1].copy()
    ambiguous = mutants[classes.map(len) > 1]
    covered["target_label"] = [next(iter(c)) for c in classes[classes.map(len) == 1]]
    tasks["e1"] = covered
    print("== e1: G12D/V vs other-codon-12 vs G13D (3-class)")
    for idx, name in enumerate(E1_CLASSES):
        sub = covered[covered["target_label"] == idx]
        print(
            f"   class {idx} {name:14s}: {len(sub):3d} slides / {sub['patient_id'].nunique():3d} patients"
        )
    print(f"   excluded (multi-class): {len(ambiguous)} -> {ambiguous['kras_subvariant'].tolist()}")
    print(f"   excluded (not codon12/G13D): {len(mutants) - len(covered) - len(ambiguous)}")

    # e2 — G12D/V vs other KRAS mutants
    e2 = mutants.copy()
    e2["target_label"] = e2["kras_subvariant"].map(
        lambda v: int(any(t in ("G12D", "G12V") for t in subvariant_tokens(v)))
    )
    tasks["e2"] = e2
    print("\n== e2: G12D/V vs other KRAS-mutant (binary)")
    print(
        e2.groupby("target_label").agg(
            slides=("slide_id", "size"), patients=("patient_id", "nunique")
        )
    )

    # e3 — codon12 vs non-codon12
    e3 = mutants.copy()
    e3["target_label"] = e3["kras_subvariant"].map(
        lambda v: int(any(CODON12.fullmatch(t) for t in subvariant_tokens(v)))
    )
    tasks["e3"] = e3
    print("\n== e3: codon-12 vs non-codon-12 (binary)")
    print(
        e3.groupby("target_label").agg(
            slides=("slide_id", "size"), patients=("patient_id", "nunique")
        )
    )

    print("\nper-cohort class balance:")
    for name, df in tasks.items():
        ct = pd.crosstab(df["cohort_group"], df["target_label"])
        print(f"  {name}: {ct.to_dict('index')}")

    for name, df in tasks.items():
        path, splits = write_task_manifest_and_splits(name, df, split_root=REPO / "outputs/splits")
        # inner-val positive support per fold (the early-stopping rule input)
        rare = splits["target_label"].value_counts().idxmin()
        support = inner_validation_patient_support(splits, rare)
        print(
            f"  {name}: wrote {path.name} ({len(df)} slides); derived splits OK; "
            f"rarest-class inner-val patients per fold: {support}"
        )


if __name__ == "__main__":
    build()
