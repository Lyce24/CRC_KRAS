#!/usr/bin/env python3
"""Build crc_final_v5.csv = crc_final_v4.csv + the 41 CRC-Orion H&E slides.

v5 is purely additive. Every one of v4's 2,028 rows is carried through
unchanged -- the build asserts this record-by-record and refuses to write if a
single field moved -- and 41 Orion rows are appended in v4's exact 62-column
schema, giving 2,069 slides / 1,864 patients across five cohorts.

Orion values are not re-derived by hand. The study's own rules are imported and
applied to the new rows so v5 stays internally consistent:

* stage      -- `derive_stage_crc_final_v4.stage_major` / `stage_group_detail`
* sidedness  -- `derive_sidedness_crc_final_v4.sidedness`

Two columns are legitimately empty for Orion and one carries a new value:

* ``ajcc_edition``          cBioPortal publishes stage and TNM but not the
                            edition. SurGen and CPTAC are 0% here too.
* ``metastatic_site_group`` not applicable: every Orion specimen is a primary.
* ``mpp_source = ome_xml``  these OME-TIFFs leave the TIFF resolution tags
                            empty, so the pixel size is read per file from the
                            OME-XML block rather than from TIFF metadata. It is
                            still read from the file, never a cohort default.

Sidedness deliberately uses the study's derivation rather than Orion's own
``SIDE`` field: the source forces transverse colon into "right", while the
study keeps ``transverse`` as its own level. Using the source field would
import a second convention for 4 patients.

Usage::

    python tools/build_crc_final_v5.py            # dry run: build + verify
    python tools/build_crc_final_v5.py --apply    # write the outputs
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from derive_sidedness_crc_final_v4 import sidedness  # noqa: E402
from derive_stage_crc_final_v4 import stage_group_detail, stage_major  # noqa: E402

MANIFEST_ROOT = Path("/mnt/d/YC.Liu/manifests/colon")
V4_CSV = MANIFEST_ROOT / "crc_final_v4.csv"
V5_CSV = MANIFEST_ROOT / "crc_final_v5.csv"
V5_XLSX = MANIFEST_ROOT / "crc_final_v5.xlsx"
ORION_CSV = MANIFEST_ROOT / "crc_orion" / "crc_orion_by_slide.csv"
INVENTORY_CSV = MANIFEST_ROOT / "colon_ready_inventory.csv"

EXPECTED_V4_ROWS = 2_028
EXPECTED_ORION_ROWS = 41


def _blank(value: object) -> bool:
    s = str(value).strip()
    return s in {"", "nan", "None", "NaN"}


def build_orion_rows(columns: list[str]) -> pd.DataFrame:
    """Render the Orion cohort into v4's schema, one row per slide."""
    orion = pd.read_csv(ORION_CSV, dtype=str, keep_default_na=False)
    if len(orion) != EXPECTED_ORION_ROWS:
        raise SystemExit(f"{ORION_CSV}: expected {EXPECTED_ORION_ROWS} rows, got {len(orion)}")

    inventory = pd.read_csv(INVENTORY_CSV, dtype=str, keep_default_na=False)
    inventory = inventory[inventory["cohort"] == "Orion"]
    if len(inventory) != EXPECTED_ORION_ROWS:
        raise SystemExit(
            f"{INVENTORY_CSV}: {len(inventory)} Orion rows, expected {EXPECTED_ORION_ROWS}. "
            "Run `scripts/watch_colon_encoding.py inventory` first."
        )
    inv = inventory.set_index("output_id")
    if not_ready := sorted(inv.index[inv["status"] != "ready"]):
        raise SystemExit(f"Orion slides not ready in the inventory: {not_ready}")

    rows = []
    for record in orion.to_dict("records"):
        slide_id = record["slide_id"]
        if slide_id not in inv.index:
            raise SystemExit(f"{slide_id} is absent from {INVENTORY_CSV}")
        i = inv.loc[slide_id]
        wsi = i["wsi"]

        row = {c: "" for c in columns}
        # 1. the 36 label columns shared with kras_final / v4, by name.
        for c in columns:
            if c in record:
                row[c] = record[c]
        # v4's name for the field the Orion table calls braf_variant.
        row["braf_subvariant"] = record.get("braf_variant", "")

        # 2. slide identity and inventory decision, straight from the ledger
        #    the watcher wrote -- never recomputed here.
        endpoint_known = record["kras"] in {"mutant", "wild_type"}
        ras_known = record["ras"] in {"mutant", "wild_type"}
        row.update({
            "wsi": wsi,
            "output_id": slide_id,
            "image_filename": Path(wsi).name,
            "image_format": Path(wsi).suffix,
            "available": "yes",
            "used": "yes",
            "used_kras": "yes" if endpoint_known else "no",
            "used_ras": "yes" if ras_known else "no",
            "mpp": i["mpp"],
            "mpp_valid": "yes",
            # Read per file from the OME-XML block, not from TIFF tags.
            "mpp_source": "ome_xml",
            "qc_slides": "pass",
            "inventory_status": i["status"],
            "inventory_source_status": i["source_status"],
            "inventory_reason": i["reason"],
            "inventory_evidence": i["evidence"],
            "slide_size_bytes": i["size_bytes"],
            "slide_mtime_ns": i["mtime_ns"],
        })

        # 3. stage, via the study's own AJCC rule. Orion publishes a stage
        #    group directly, so stage_source is 'original' and the derived
        #    columns stand beside it as an independent cross-check.
        t, n, m = record["t_stage"], record["n_stage"], record["m_stage"]
        row["stage_group_major_derived"] = stage_major(t, n, m) or ""
        row["stage_group_derived"] = stage_group_detail(t, n, m) or ""
        has_original = not _blank(record["stage_group_major"])
        row["stage_source"] = (
            "original" if has_original
            else "derived_ajcc_from_tnm" if row["stage_group_major_derived"]
            else "unknown"
        )
        row["stage_group_major_filled"] = (
            record["stage_group_major"] if has_original else row["stage_group_major_derived"]
        )
        row["stage_group_filled"] = (
            record["stage_group"] if not _blank(record["stage_group"])
            else row["stage_group_derived"]
        )

        # 3b. A published stage that contradicts its own TNM is flagged, not
        #     silently kept: v4 carries the identical defect for SurGen
        #     SR386_T084/T231/T305 (M1 published as III) under qc_flags='stage',
        #     and §10.3 lists those disagreements as needing adjudication
        #     before publication. Flagging puts Orion's into the same net.
        if has_original and row["stage_group_major_derived"] and (
            record["stage_group_major"] != row["stage_group_major_derived"]
        ):
            flags = [f for f in row["qc_flags"].split(";") if f]
            if "stage" not in flags:
                flags.append("stage")
            row["qc_flags"] = ";".join(flags)
            note = (f"published stage {record['stage_group']} contradicts TNM "
                    f"{record['t_stage']}/{record['n_stage']}/{record['m_stage']} "
                    f"-> AJCC {row['stage_group_derived']}")
            row["qc_note"] = f"{row['qc_note']}; {note}" if row["qc_note"] else note

        # 4. sidedness, via the study's own rule (see module docstring).
        side = sidedness(record["tumor_site_raw"])
        row["sidedness"] = side or ""
        row["sidedness_source"] = (
            "derived_from_tumor_site_raw" if side
            else "not_applicable_metastatic" if record["specimen_role"] == "metastatic"
            else "unresolved"
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    v4 = pd.read_csv(V4_CSV, dtype=str, keep_default_na=False)
    if len(v4) != EXPECTED_V4_ROWS:
        raise SystemExit(f"{V4_CSV}: expected {EXPECTED_V4_ROWS} rows, got {len(v4)}")
    orion = build_orion_rows(list(v4.columns))
    return pd.concat([v4, orion], ignore_index=True), v4


def verify(v5: pd.DataFrame, v4: pd.DataFrame) -> None:
    """Fail closed unless v5 is exactly v4 plus 41 Orion rows."""
    problems: list[str] = []
    if list(v5.columns) != list(v4.columns):
        problems.append("column set or order changed")
    if len(v5) != len(v4) + EXPECTED_ORION_ROWS:
        problems.append(f"{len(v5)} rows, expected {len(v4) + EXPECTED_ORION_ROWS}")

    # every original row must survive byte-identically, in place
    head = v5.iloc[: len(v4)].reset_index(drop=True)
    if not head.equals(v4.reset_index(drop=True)):
        diff = (head != v4.reset_index(drop=True)).any(axis=1)
        problems.append(f"{int(diff.sum())} pre-existing v4 rows changed, "
                        f"first at index {list(diff[diff].index[:3])}")

    tail = v5.iloc[len(v4):]
    if set(tail["cohort"]) != {"Orion"}:
        problems.append(f"appended rows are not all Orion: {sorted(set(tail['cohort']))}")
    if v5["slide_uid"].duplicated().any():
        dupes = sorted(v5.loc[v5["slide_uid"].duplicated(), "slide_uid"])[:5]
        problems.append(f"duplicate slide_uid: {dupes}")
    for col in ("mpp", "wsi", "output_id", "kras", "msi_dmmr"):
        if (tail[col].map(_blank)).any():
            problems.append(f"Orion rows have a blank {col}")
    if problems:
        raise SystemExit("VERIFICATION FAILED:\n  - " + "\n  - ".join(problems))


def atomic_write(table: pd.DataFrame, csv_path: Path, xlsx_path: Path) -> None:
    for path, writer in ((csv_path, "csv"), (xlsx_path, "xlsx")):
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        if writer == "csv":
            table.to_csv(tmp, index=False)
        else:
            table.to_excel(tmp, index=False, sheet_name="crc_final_v5")
        os.replace(tmp, path)


def summarize(v5: pd.DataFrame, v4: pd.DataFrame) -> None:
    print(f"v4: {len(v4):5d} slides, {v4['patient_uid'].nunique():5d} patients, "
          f"{v4['cohort'].nunique()} cohorts")
    print(f"v5: {len(v5):5d} slides, {v5['patient_uid'].nunique():5d} patients, "
          f"{v5['cohort'].nunique()} cohorts")
    print("\nper-cohort slides:", v5["cohort"].value_counts().to_dict())
    orion = v5[v5["cohort"] == "Orion"]
    print(f"\nOrion: {len(orion)} slides / {orion['patient_uid'].nunique()} patients")
    for col in ("kras", "ras", "braf", "msi_dmmr", "stage_group_major",
                "sidedness", "used_kras", "stage_source"):
        vals = orion[col].replace("", "(blank)").value_counts().to_dict()
        print(f"  {col:20s} {vals}")
    empty = [c for c in v5.columns if orion[c].map(_blank).all()]
    print(f"\ncolumns empty for every Orion row: {empty}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write crc_final_v5.csv/.xlsx")
    args = ap.parse_args()

    v5, v4 = build_table()
    verify(v5, v4)
    summarize(v5, v4)

    if not args.apply:
        with tempfile.TemporaryDirectory(prefix="crc-final-v5-") as d:
            atomic_write(v5, Path(d) / V5_CSV.name, Path(d) / V5_XLSX.name)
            back = pd.read_csv(Path(d) / V5_CSV.name, dtype=str, keep_default_na=False)
            if not back.equals(v5):
                raise SystemExit("round-trip through CSV changed the table")
        print("\nDry run verified (including CSV round-trip). Use --apply to write.")
        return 0

    if V5_CSV.exists() or V5_XLSX.exists():
        raise SystemExit("Refusing to overwrite an existing crc_final_v5 output")
    atomic_write(v5, V5_CSV, V5_XLSX)
    print(f"\nWrote {V5_CSV}\nWrote {V5_XLSX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
