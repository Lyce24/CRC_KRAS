#!/usr/bin/env python3
"""Derive tumour sidedness into crc_final_v4.csv.

Right/proximal (embryologic midgut): caecum, ascending colon, hepatic flexure.
Left/distal (hindgut): splenic flexure, descending colon, sigmoid, rectosigmoid,
rectum.

TRANSVERSE COLON IS ITS OWN LEVEL, not forced into "right". It spans the
midgut-hindgut watershed and the literature splits on it; collapsing it would
bury a real ambiguity inside a binary the reader cannot audit. 53 development
patients are affected.

WHAT CANNOT BE DERIVED. 372 of the 434 unresolved development patients are
TCGA-COAD coded literally as "Colon" with no sublocation. That is a TCGA data
limitation, not a parsing gap, and it caps development coverage at ~71%.

SPECIMEN-SITE CAVEAT. `tumor_site_meaning` is `specimen_site` for RIH and
SurGen, so for a METASTATIC specimen the recorded site is the metastasis, not
the primary. Sidedness is therefore written ONLY for primary specimens; every
metastatic row is left null regardless of what its site string says.

The raw strings contain typos carried through from source
("Hepatix Flexure", "Tranverse Colon", "decending colon"), matched here by
substring. The design document asks for pathologist review of `tumor_site_raw`
before this variable is used in a publication; this derivation does not replace
that.

Usage:
    python tools/derive_sidedness_crc_final_v4.py [--apply]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v4.csv")

RIGHT = ("caecum", "cecum", "caecal", "ascend", "hepatic flex", "hepatix flex",
         "right hemicol", "right (ascend", "right colon", "proximal")
LEFT = ("splenic flex", "descend", "decending", "sigmoid", "rectosig", "rectal",
        "rectum", "left hemicol", "anterior resection", "distal")
TRANSVERSE = ("transverse", "tranverse")


def sidedness(raw: object) -> str | None:
    s = str(raw).strip().lower()
    if s in {"", "nan", "none", "unknown"}:
        return None
    if any(k in s for k in TRANSVERSE):
        return "transverse"
    if any(k in s for k in RIGHT):
        return "right"
    if any(k in s for k in LEFT):
        return "left"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    d = pd.read_csv(SRC, low_memory=False)
    raw_side = d["tumor_site_raw"].map(sidedness)
    is_primary = d["specimen_role"].astype(str).str.strip().str.lower().eq("primary")
    side = raw_side.where(is_primary)          # metastatic rows -> null, see docstring

    print(f"{'=' * 78}\nSIDEDNESS — all {len(d)} rows\n{'=' * 78}")
    print(f"  primary rows           : {int(is_primary.sum())}")
    print(f"  resolved (primary only): {int(side.notna().sum())}")
    print(f"  distribution           : {side.value_counts(dropna=False).to_dict()}")
    print(f"\n  suppressed on metastatic rows: "
          f"{int((raw_side.notna() & ~is_primary).sum())} would have parsed but are not primary")

    print(f"\n  by subcohort (primary rows):")
    pr = d[is_primary].assign(side=side[is_primary])
    for sub, b in pr.groupby("subcohort"):
        vc = b.side.value_counts(dropna=False).to_dict()
        cov = b.side.notna().mean()
        print(f"    {sub:12s} n={len(b):5d} cov={cov:6.1%}  {vc}")

    print(f"\n  unresolved primary rows, top raw values:")
    print("   ", pr.loc[pr.side.isna(), "tumor_site_raw"].astype(str)
          .value_counts().head(6).to_dict())

    if not a.apply:
        print("\nDry run — pass --apply to write.")
        return

    backup = SRC.with_name("crc_final_v4_PRE_SIDEDNESS.csv")
    if not backup.exists():
        shutil.copy2(SRC, backup)
        print(f"\nbacked up -> {backup.name}")
    d["sidedness"] = side
    d["sidedness_source"] = np.where(
        side.notna(), "derived_from_tumor_site_raw",
        np.where(is_primary, "unresolved", "not_applicable_metastatic"))
    d.to_csv(SRC, index=False)
    print(f"wrote {SRC}")
    print(f"  new columns: sidedness, sidedness_source")
    print(f"  sidedness_source: {d.sidedness_source.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
