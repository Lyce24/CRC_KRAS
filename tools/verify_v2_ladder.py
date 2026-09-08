#!/usr/bin/env python3
"""Independent audit of the v2 ladder: labels re-derived from the raw source,
splits checked through the datamodule's own code paths. Exits non-zero on any
violation.

Label rules are RE-IMPLEMENTED here (not imported from the builder) so a
builder bug cannot silently verify itself.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.kras import study  # noqa: E402
from oceanpath.splitting.core import get_slide_ids_for_fold, verify_split_integrity  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL  {message}")


# ── Fresh, independent label rules ────────────────────────────────────────────

_C12 = re.compile(r"^G12[A-Z]$")
_EX2 = re.compile(r"^G1[23][A-Z]$")


def toks(v: object) -> list[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return []
    return [t.strip() for t in str(v).split(";") if t.strip()]


RULES = {
    "b1_v2": lambda kras, sub: 1 if kras == "mutant" else 0,
    "e4_v2": lambda kras, sub: int(any(_EX2.match(t) for t in toks(sub))),
    "e3_v2": lambda kras, sub: int(any(_C12.match(t) for t in toks(sub))),
    "e2_v2": lambda kras, sub: int(bool({"G12D", "G12V"} & set(toks(sub)))),
    "e5_v2": lambda kras, sub: int(bool({"G12D", "G12V", "G13D"} & set(toks(sub)))),
    "p1_v2": lambda kras, sub: int("G12D" in toks(sub)),
}
MUTANT_ONLY = {"e4_v2", "e3_v2", "e2_v2", "e5_v2", "p1_v2"}
EXTERNAL_SETS = ("rih_all", "rih_primary", "rih_metastatic", "sr1482_metastatic")


def raw_source() -> pd.DataFrame:
    df = pd.read_csv(study.KRAS_FINAL_CSV, dtype=str)
    df = df[(df["include"] == "yes") & df["kras"].isin(["mutant", "wild_type"])].copy()
    df["local_id"] = df["slide_uid"].str.split(":", n=1).str[1]
    return df


def audit_labels() -> None:
    src = raw_source()
    by_local: dict[str, tuple[str, object, str, str, str]] = {}
    for _, r in src.iterrows():
        by_local[r["local_id"]] = (
            r["kras"], r["kras_subvariant"], r["patient_uid"], r["specimen_role"], r["subcohort"]
        )

    def src_row(slide_id: str):
        # TCGA manifests carry the barcode + UUID stem; raw uses the barcode.
        if slide_id in by_local:
            return by_local[slide_id]
        return by_local.get(slide_id.split(".")[0])

    print("== label re-derivation (fresh rules) vs every manifest on disk")
    n_checked = 0
    for task, rule in RULES.items():
        manifests = [("dev", study.MANIFEST_ROOT / f"crc_kras_dev_{task}.csv")] + [
            (s, study.MANIFEST_ROOT / f"crc_kras_{task}_{s}.csv") for s in EXTERNAL_SETS
        ]
        for set_name, path in manifests:
            if not path.is_file():
                continue
            man = pd.read_csv(path)
            bad = 0
            for _, row in man.iterrows():
                raw = src_row(row["slide_id"])
                if raw is None:
                    bad += 1
                    continue
                kras, sub, puid, role, subc = raw
                expected = rule(kras, sub)
                if int(row["target_label"]) != expected:
                    bad += 1
                # mutant-only tasks must never contain WT or variant-less rows
                if task in MUTANT_ONLY and (kras != "mutant" or not toks(sub)):
                    bad += 1
                # unit identity: rih_all -> uid|role, else raw patient uid
                expected_pid = f"{puid}|{role}" if set_name == "rih_all" else puid
                if str(row["patient_id"]) != expected_pid:
                    bad += 1
                n_checked += 1
            check(bad == 0, f"{task}/{set_name}: {bad} label/population/unit mismatches")
        # population completeness for DEV: every eligible source slide present
        dev_man = pd.read_csv(study.MANIFEST_ROOT / f"crc_kras_dev_{task}.csv")
        master = pd.read_csv(study.MANIFEST_ROOT / "crc_kras_master_dev_v2.csv")
        if task in MUTANT_ONLY:
            eligible = master[(master["kras"] == "mutant")
                              & master["kras_subvariant"].map(lambda v: len(toks(v)) > 0)]
        else:
            eligible = master
        check(set(dev_man["slide_id"]) == set(eligible["slide_id"]),
              f"{task}/dev: population != eligible master rows")
    print(f"   {n_checked} manifest rows re-derived and matched")

    print("== named edge cases")
    cases = [
        # T264 (G12V;G12C) is an SR1482 PRIMARY slide — part of DEV in v2.
        ("e2_v2", "dev", "SR1482_40X_HE_T264", 1),
        ("p1_v2", "dev", "SR1482_40X_HE_T264", 0),
        ("e3_v2", "rih_metastatic", "SL-258", 0),   # G13D;R164Q: codon 13, not 12
        ("e4_v2", "rih_metastatic", "SL-258", 1),   # ...but exon 2
        ("e5_v2", "rih_metastatic", "SL-258", 1),
        ("p1_v2", "dev", "TCGA-G4-6320", 1),        # G12D;G13D
        ("e5_v2", "dev", "TCGA-G4-6320", 1),
        ("p1_v2", "dev", "TCGA-AG-4008", 0),        # G12C;G12V
        ("e2_v2", "dev", "TCGA-AG-4008", 1),
    ]
    for task, set_name, needle, want in cases:
        path = (study.MANIFEST_ROOT / f"crc_kras_dev_{task}.csv" if set_name == "dev"
                else study.MANIFEST_ROOT / f"crc_kras_{task}_{set_name}.csv")
        man = pd.read_csv(path)
        hit = man[man["slide_id"].str.contains(needle, regex=False)]
        check(not hit.empty and int(hit["target_label"].iloc[0]) == want,
              f"edge case {needle} in {task}: expected {want}")

    # SL-102 (mutant, no variant): in b1_v2 externals as positive, absent from allele tasks
    b1_rih = pd.read_csv(study.MANIFEST_ROOT / "crc_kras_b1_v2_rih_metastatic.csv")
    check(b1_rih[b1_rih.slide_id == "SL-102"]["target_label"].tolist() == [1],
          "SL-102 must be a b1_v2 positive")
    p1_rih = pd.read_csv(study.MANIFEST_ROOT / "crc_kras_p1_v2_rih_metastatic.csv")
    check("SL-102" not in set(p1_rih["slide_id"]), "SL-102 must be absent from p1_v2")

    # RIH cross-specimen conflict: one patient, two units, opposite b1 labels
    b1_all = pd.read_csv(study.MANIFEST_ROOT / "crc_kras_b1_v2_rih_all.csv")
    conflicted = src[src["subcohort"] == "RIH-Colon"].groupby("patient_uid")["kras"].nunique()
    conflict_uids = set(conflicted[conflicted > 1].index)
    check(len(conflict_uids) > 0, "expected >=1 RIH cross-specimen kras conflict")
    for uid in sorted(conflict_uids):
        units = b1_all[b1_all["patient_id"].str.startswith(uid + "|")]
        check(units["patient_id"].nunique() == len(units.groupby("patient_id")),
              f"{uid}: units malformed")
        per_unit = units.groupby("patient_id")["target_label"].nunique()
        check((per_unit == 1).all(), f"{uid}: label conflict WITHIN a patient-role unit")
    print(f"   {len(conflict_uids)} conflicted RIH patients correctly split into role units")


def audit_splits() -> None:
    print("== v2 split audit (datamodule code paths)")
    master_csv = study.MANIFEST_ROOT / "crc_kras_master_dev_v2.csv"
    master = pd.read_csv(master_csv)
    split_name = study.split_name(42)
    master_dir = REPO / "outputs/splits/colon_kras_master_v2" / split_name

    def audit_dir(name: str, splits_dir: Path, manifest_csv: Path) -> None:
        manifest = pd.read_csv(manifest_csv)
        pmap = dict(zip(manifest["slide_id"], manifest["patient_id"], strict=True))
        try:
            verify_split_integrity(splits_dir, manifest_csv)
        except Exception as error:  # noqa: BLE001
            check(False, f"{name}: integrity ({error})")
            return
        splits = pd.read_parquet(splits_dir / "splits.parquet")
        check(set(splits["slide_id"]) == set(manifest["slide_id"]), f"{name}: rows != manifest")
        spanning = splits.groupby(splits["slide_id"].map(pmap))["fold"].nunique()
        check((spanning == 1).all(), f"{name}: patient spans folds")
        seen = []
        for fold in range(5):
            roles = get_slide_ids_for_fold(splits, fold, scheme="oof_kfold")
            train, val, test = (set(roles[r]) for r in ("train", "val", "test"))
            check(not (train & val) and not (val & test) and not (train & test),
                  f"{name} f{fold}: role overlap")
            check(train | val | test == set(splits["slide_id"]), f"{name} f{fold}: no cover")
            p = [{pmap[s] for s in ids} for ids in (train, val, test)]
            check(not (p[0] & p[1]) and not (p[0] & p[2]) and not (p[1] & p[2]),
                  f"{name} f{fold}: PATIENT crosses roles")
            seen.extend(roles["test"])
        check(sorted(seen) == sorted(splits["slide_id"]), f"{name}: OOF not a partition")

    audit_dir("master_v2", master_dir, master_csv)
    # stratification balance (patient level, 18 strata)
    splits = pd.read_parquet(master_dir / "splits.parquet")[["slide_id", "fold"]]
    splits["stratum"] = splits["slide_id"].map(
        dict(zip(master["slide_id"], master["strat_kras_class_cohort"], strict=True)))
    splits["pid"] = splits["slide_id"].map(
        dict(zip(master["slide_id"], master["patient_id"], strict=True)))
    patients = splits.groupby("pid").agg(fold=("fold", "first"), stratum=("stratum", "first"))
    overall = patients["stratum"].value_counts(normalize=True)
    deviation = max(
        abs(patients[patients["fold"] == f]["stratum"].value_counts(normalize=True).get(s, 0.0)
            - overall[s])
        for f in range(5) for s in overall.index)
    check(deviation < 0.06, f"master_v2 stratum deviation {deviation:.3f} >= 0.06")
    print(f"   master_v2 OK (max stratum deviation {deviation:.3%})")

    master_folds = pd.read_parquet(master_dir / "splits.parquet").set_index("slide_id")
    for task in RULES:
        d = REPO / f"outputs/splits/colon_kras_{task}" / split_name
        audit_dir(task, d, study.MANIFEST_ROOT / f"crc_kras_dev_{task}.csv")
        derived = pd.read_parquet(d / "splits.parquet").set_index("slide_id")
        joined = derived.join(master_folds, rsuffix="_m")
        check(bool((joined["fold"] == joined["fold_m"]).all()),
              f"{task}: folds differ from master_v2")
        print(f"   {task} OK")

    print("== DEV v2 / external patient disjointness (raw UIDs)")
    dev_uids = set(master["patient_id"])
    for task in RULES:
        for set_name in EXTERNAL_SETS:
            path = study.MANIFEST_ROOT / f"crc_kras_{task}_{set_name}.csv"
            if not path.is_file():
                continue
            ext = pd.read_csv(path)
            raw_uids = set(ext["patient_id"].str.split("|").str[0])
            overlap = dev_uids & raw_uids
            check(not overlap, f"{task}/{set_name}: DEV patients in external: "
                  f"{sorted(overlap)[:3]}")
    print("   no DEV patient appears in any external set")


def main() -> None:
    audit_labels()
    audit_splits()
    print()
    if FAILURES:
        raise SystemExit(f"V2 AUDIT FAILED — {len(FAILURES)} violation(s)")
    print("V2 AUDIT PASSED — labels re-derive exactly, populations correct, units "
          "sound, splits patient-pure with exact OOF partitions, zero leakage.")


if __name__ == "__main__":
    main()
