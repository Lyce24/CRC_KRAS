#!/usr/bin/env python3
"""Derive AJCC stage for rows that have TNM but no stage group.

WHY. `crc_final_v4.csv` carries t_stage / n_stage / m_stage for most of the
cohort but `stage_group` / `stage_group_major` were never computed for SR1482 —
593 rows, of which 155 development patients have usable T and 149 usable N. The
study had been treating SR1482 as stage-unknown and declaring the SurGen
subcohort gap un-adjustable for stage. That was an artefact of the derived
column, not of the data.

RULE (AJCC 7th and 8th edition; identical for the MAJOR group, which is all this
fills in for the study):

    M1 / M1a / M1b / M1c              -> IV
    N1* or N2*  (M0)                  -> III
    N0, T3 / T4 / T4a / T4b (M0)      -> II
    N0, T1 / T2 / Tis (M0)            -> I
    anything else (TX, NX, missing)   -> left unknown

MISSING M IS TREATED AS M0. That is not an assumption invented here: of the
1,367 rows that already carry a stage group, 397 have no m_stage, and 396 of
them were staged as I-III by whoever derived the original column. The rule
therefore reproduces the existing convention rather than introducing a new one.

VALIDATION IS THE POINT. The rule is applied to the rows that ALREADY have a
stage group and the result compared against it. Only if that agreement is high
is the rule used to fill the blanks. Derived values are written to NEW columns
and flagged, so nothing original is overwritten and every filled value is
traceable.

Usage:
    python tools/derive_stage_crc_final_v4.py            # validate only
    python tools/derive_stage_crc_final_v4.py --apply    # validate, then write
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v4.csv")


def _clean(v: object) -> str:
    s = str(v).strip().upper()
    return "" if s in {"", "NAN", "NONE", "NA", "N/A", "UNKNOWN"} else s


def stage_major(t: object, n: object, m: object) -> str | None:
    """AJCC major group, or None when the components cannot decide it."""
    T, N, M = _clean(t), _clean(n), _clean(m)
    if M.startswith("M1"):
        return "IV"
    # MX and missing M both fall through to the M0 branch, per the file's own
    # existing convention (see module docstring).
    if N.startswith("N1") or N.startswith("N2"):
        return "III"
    if N == "N0":
        if T in {"T3", "T4", "T4A", "T4B"}:
            return "II"
        if T in {"T1", "T2", "TIS"}:
            return "I"
        return None          # N0 but TX/missing -> cannot separate I from II
    return None              # NX or missing N, and not M1


def stage_group_detail(t: object, n: object, m: object) -> str | None:
    """Sub-stage where the components support it, else the major group."""
    T, N, M = _clean(t), _clean(n), _clean(m)
    major = stage_major(t, n, m)
    if major is None:
        return None
    if major == "IV":
        return {"M1A": "Stage IVA", "M1B": "Stage IVB", "M1C": "Stage IVC"}.get(M, "Stage IV")
    if major == "II":
        return {"T3": "Stage IIA", "T4A": "Stage IIB", "T4B": "Stage IIC"}.get(T, "Stage II")
    if major == "I":
        return "Stage I"
    # III — sub-staging needs N1a/N1b/N2a/N2b granularity that is often absent
    if N in {"N1", "N1A", "N1B", "N1C"} and T in {"T1", "T2"}:
        return "Stage IIIA"
    if N in {"N1", "N1A", "N1B", "N1C"} and T in {"T3", "T4A"}:
        return "Stage IIIB"
    if N in {"N2", "N2A", "N2B"} and T in {"T4A", "T4B"}:
        return "Stage IIIC"
    if T == "T4B":
        return "Stage IIIC"
    return "Stage III"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    d = pd.read_csv(SRC, low_memory=False)
    derived_major = [stage_major(t, n, m) for t, n, m in
                     zip(d["t_stage"], d["n_stage"], d["m_stage"])]
    derived_group = [stage_group_detail(t, n, m) for t, n, m in
                     zip(d["t_stage"], d["n_stage"], d["m_stage"])]
    d["_derived_major"] = derived_major
    d["_derived_group"] = derived_group

    # ── VALIDATION against rows that already carry a stage ──────────────────
    have = d["stage_group_major"].notna()
    both = have & d["_derived_major"].notna()
    agree = (d.loc[both, "stage_group_major"].astype(str).str.strip()
             == d.loc[both, "_derived_major"]).sum()
    print(f"{'=' * 78}\nVALIDATION — rule applied to rows that ALREADY have a stage\n{'=' * 78}")
    print(f"  rows with an existing stage group : {int(have.sum())}")
    print(f"  of those, rule also decides       : {int(both.sum())}")
    print(f"  agreement                         : {agree}/{int(both.sum())} "
          f"({agree / max(int(both.sum()), 1):.2%})")
    dis = d.loc[both & (d["stage_group_major"].astype(str).str.strip() != d["_derived_major"])]
    if len(dis):
        print(f"\n  disagreements ({len(dis)}):")
        print(dis.groupby(["stage_group_major", "_derived_major"]).size().to_string())
        print("\n  sample:")
        print(dis[["subcohort", "t_stage", "n_stage", "m_stage",
                   "stage_group_major", "_derived_major"]].head(8).to_string(index=False))
    miss = have & d["_derived_major"].isna()
    print(f"\n  existing stage the rule CANNOT reproduce (TX/NX): {int(miss.sum())}")

    # ── what would be filled ────────────────────────────────────────────────
    fillable = d["stage_group_major"].isna() & d["_derived_major"].notna()
    print(f"\n{'=' * 78}\nFILL — rows with no stage group that the rule can decide\n{'=' * 78}")
    print(f"  fillable rows: {int(fillable.sum())}")
    print(d.loc[fillable].groupby("subcohort")["_derived_major"]
          .value_counts().unstack(fill_value=0).to_string())

    if not a.apply:
        print("\nDry run — pass --apply to write.")
        return

    backup = SRC.with_name("crc_final_v4_PRE_STAGE_DERIVATION.csv")
    if not backup.exists():
        shutil.copy2(SRC, backup)
        print(f"\nbacked up original -> {backup.name}")

    d["stage_group_major_derived"] = d["_derived_major"]
    d["stage_group_derived"] = d["_derived_group"]
    d["stage_source"] = np.where(
        d["stage_group_major"].notna(), "original",
        np.where(fillable, "derived_ajcc_from_tnm", "unknown"))
    d["stage_group_major_filled"] = d["stage_group_major"].where(
        d["stage_group_major"].notna(), d["_derived_major"])
    d["stage_group_filled"] = d["stage_group"].where(
        d["stage_group"].notna(), d["_derived_group"])
    d = d.drop(columns=["_derived_major", "_derived_group"])
    d.to_csv(SRC, index=False)
    print(f"wrote {SRC}")
    print("  new columns: stage_group_major_derived, stage_group_derived, stage_source,")
    print("               stage_group_major_filled, stage_group_filled")
    print(f"  stage_source: {d.stage_source.value_counts().to_dict()}")
    print(f"  stage_group_major_filled populated: {int(d.stage_group_major_filled.notna().sum())}"
          f" / {len(d)} (was {int(have.sum())})")


if __name__ == "__main__":
    main()
