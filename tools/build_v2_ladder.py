#!/usr/bin/env python3
"""V2 ladder dataset (2026-08-15, study owner directive).

DEV v2 (OOF 5-fold CV): SR386-P + SR1482-P + TCGA-P, KRAS known, features
present. LEAKAGE RULE: SR1482-P patients who also appear in SR1482-M are
excluded from DEV (their metastases are an external test set).

External test sets (per task): rih_all, rih_primary, rih_metastatic,
sr1482_metastatic. rih_all uses PATIENT x SPECIMEN-ROLE units (§7) because
RIH patients can carry conflicting labels across specimens.

Tasks (same six ladder rungs, ids suffixed _v2):
    b1_v2 gene | e4_v2 exon2 | e3_v2 codon12 | e2_v2 G12D/V |
    e5_v2 G12D/V/G13D | p1_v2 G12D

Writes crc_kras_dev_<task>_v2.csv + crc_kras_<task>_v2_<set>.csv, generates
the v2 shared assignment (seed 42, patient-level, 3-way cohort x 6-class
strata), and derives per-task split subsets.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.kras import study  # noqa: E402
from oceanpath.kras.labels import add_kras_class_columns, subvariant_tokens  # noqa: E402
from oceanpath.splitting.core import (  # noqa: E402
    SplitConfig,
    derive_subset_splits,
    generate_splits,
)

CODON12 = re.compile(r"G12[A-Z]")
EXON2 = re.compile(r"G1[23][A-Z]")

SPLIT_NAME = study.split_name(42)  # oofk5_seed42_es15_strat6c (v2 lives under _v2 data names)


def load_source() -> pd.DataFrame:
    df = pd.read_csv(study.KRAS_FINAL_CSV, dtype=str)
    df = df[(df["include"] == "yes") & df["kras"].isin(["mutant", "wild_type"])].copy()
    df["cohort_group"] = df["subcohort"].map(
        lambda s: "TCGA" if str(s).startswith("TCGA") else ("RIH" if s == "RIH-Colon" else s)
    )
    df = add_kras_class_columns(df)
    stems = {p.stem for p in study.PINNED_FEATURE_DIR.glob("*.h5")}
    by_barcode = {s.split(".")[0]: s for s in stems if s.startswith("TCGA")}

    def to_stem(uid: str) -> str | None:
        cohort, local = uid.split(":", 1)
        if cohort == "TCGA":
            return by_barcode.get(local)
        return local if local in stems else None

    df["slide_id"] = df["slide_uid"].map(to_stem)
    df["patient_id"] = df["patient_uid"]
    return df.dropna(subset=["slide_id"]).copy()


def variant_known(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["kras_subvariant"].map(lambda v: len(subvariant_tokens(v)) > 0)]


TASKS = {
    "b1_v2": ("gene: mutant vs WT", lambda df: df.assign(
        target_label=(df["kras"] == "mutant").astype(int))),
    "e4_v2": ("exon: exon-2 vs non-exon-2", lambda df: variant_known(df[df["kras"] == "mutant"]).pipe(
        lambda m: m.assign(target_label=m["kras_subvariant"].map(
            lambda v: int(any(EXON2.fullmatch(t) for t in subvariant_tokens(v))))))),
    "e3_v2": ("codon: G12* vs non-G12", lambda df: variant_known(df[df["kras"] == "mutant"]).pipe(
        lambda m: m.assign(target_label=m["kras_subvariant"].map(
            lambda v: int(any(CODON12.fullmatch(t) for t in subvariant_tokens(v))))))),
    "e2_v2": ("functional: G12D/V vs other mutants", lambda df: variant_known(df[df["kras"] == "mutant"]).pipe(
        lambda m: m.assign(target_label=m["kras_subvariant"].map(
            lambda v: int(any(t in ("G12D", "G12V") for t in subvariant_tokens(v))))))),
    "e5_v2": ("functional-wide: G12D/V/G13D vs other mutants", lambda df: variant_known(df[df["kras"] == "mutant"]).pipe(
        lambda m: m.assign(target_label=m["kras_subvariant"].map(
            lambda v: int(any(t in ("G12D", "G12V", "G13D") for t in subvariant_tokens(v))))))),
    "p1_v2": ("allele: G12D vs other mutants", lambda df: variant_known(df[df["kras"] == "mutant"]).pipe(
        lambda m: m.assign(target_label=m["kras_subvariant"].map(
            lambda v: int("G12D" in subvariant_tokens(v)))))),
}

EXTERNAL_SETS = {
    "rih_all": lambda df: df[df["cohort_group"] == "RIH"].assign(
        patient_id=lambda d: d["patient_uid"] + "|" + d["specimen_role"]),
    "rih_primary": lambda df: df[(df["cohort_group"] == "RIH") & (df["specimen_role"] == "primary")],
    "rih_metastatic": lambda df: df[(df["cohort_group"] == "RIH") & (df["specimen_role"] == "metastatic")],
    "sr1482_metastatic": lambda df: df[(df["cohort_group"] == "SR1482") & (df["specimen_role"] == "metastatic")],
}


def main() -> None:
    source = load_source()

    sr1482_m_patients = set(
        source[(source["cohort_group"] == "SR1482") & (source["specimen_role"] == "metastatic")][
            "patient_id"]
    )
    dev = source[
        (source["specimen_role"] == "primary")
        & source["cohort_group"].isin(["SR386", "SR1482", "TCGA"])
    ].copy()
    n_before = len(dev)
    leak = dev[(dev["cohort_group"] == "SR1482") & dev["patient_id"].isin(sr1482_m_patients)]
    dev = dev[~dev.index.isin(leak.index)].copy()
    print(f"DEV v2: {n_before} primary slides -> {len(dev)} after excluding "
          f"{len(leak)} SR1482-P slides from {leak['patient_id'].nunique()} patients "
          f"with SR1482-M specimens (leakage rule)")
    print(f"DEV v2 patients: {dev['patient_id'].nunique()}; per cohort: "
          f"{dev.groupby('cohort_group')['patient_id'].nunique().to_dict()}")
    print(f"DEV v2 mutant slides: {(dev['kras'] == 'mutant').sum()} "
          f"(variant-known {len(variant_known(dev[dev['kras'] == 'mutant']))})\n")

    # Master v2 manifest + shared assignment
    master = dev.copy()
    master["target_label"] = (master["kras"] == "mutant").astype(int)
    master_out = master[study.MANIFEST_COLUMNS].sort_values("slide_id").reset_index(drop=True)
    study.assert_manifest_invariants(master_out, "master_v2")
    master_csv = study.MANIFEST_ROOT / "crc_kras_master_dev_v2.csv"
    master_out.to_csv(master_csv, index=False)

    master_splits_dir = REPO / "outputs/splits/colon_kras_master_v2" / SPLIT_NAME
    generate_splits(SplitConfig(
        scheme="oof_kfold",
        name=SPLIT_NAME,
        csv_path=str(master_csv),
        output_dir=str(master_splits_dir),
        filename_column="slide_id",
        label_column="strat_kras_class_cohort",
        group_column="patient_id",
        n_folds=5,
        seed=42,
        val_ratio=0.15,
    ))

    # Task manifests (DEV) + derived splits + external test manifests
    for task, (title, build_fn) in TASKS.items():
        rows = build_fn(dev)
        out = rows[study.MANIFEST_COLUMNS].sort_values("slide_id").reset_index(drop=True)
        study.assert_manifest_invariants(out, task)
        path = study.MANIFEST_ROOT / f"crc_kras_dev_{task}.csv"
        out.to_csv(path, index=False)
        derive_subset_splits(
            master_splits_dir=master_splits_dir,
            manifest_csv=path,
            output_dir=REPO / f"outputs/splits/colon_kras_{task}" / SPLIT_NAME,
            filename_column="slide_id",
        )
        dist = out.groupby("target_label")["patient_id"].nunique().to_dict()
        print(f"== {task} ({title}): {len(out)} slides, patients per class {dist}")

        for set_name, select in EXTERNAL_SETS.items():
            ext = select(source)
            ext_rows = build_fn(ext)
            if ext_rows["target_label"].nunique() < 2:
                print(f"   {set_name}: SKIPPED (single class)")
                continue
            ext_out = ext_rows[study.MANIFEST_COLUMNS].sort_values("slide_id").reset_index(drop=True)
            study.assert_manifest_invariants(ext_out, f"{task}/{set_name}")
            ext_path = study.MANIFEST_ROOT / f"crc_kras_{task}_{set_name}.csv"
            ext_out.to_csv(ext_path, index=False)
            units = ext_out["patient_id"].nunique()
            pos = int((ext_out.groupby("patient_id")["target_label"].max() == 1).sum())
            print(f"   {set_name}: {len(ext_out)} slides / {units} units ({pos} pos)")

    # Safety: DEV v2 must share no patients with any external set.
    dev_patients = set(master_out["patient_id"])
    for set_name, select in EXTERNAL_SETS.items():
        ext = select(source)
        overlap = dev_patients & set(ext["patient_uid"])
        assert not overlap, f"DEV v2 overlaps {set_name}: {sorted(overlap)[:5]}"
    print("\nDEV v2 x external patient overlap: none (SR1482 dual-arm patients excluded)")


if __name__ == "__main__":
    main()
