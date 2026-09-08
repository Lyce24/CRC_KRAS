#!/usr/bin/env python3
"""Build the CRC-Orion (DFCI/Orion 2024) clinical + genomic label tables.

Two linked public sources, joined on the ``C<n>`` patient id:

* **Images** — anonymous S3 bucket ``s3://lin-2023-orion-crc/data``. Only the
  ``*-registered.ome.tif`` H&E files are used (41 files / 40 patients,
  ~40 GB); the 19-channel Orion IF ``*-zlib.ome.tiff`` files (70-150 GB each)
  are deliberately not downloaded. See ``tools/download_crc_orion_slides.py``.
* **Clinical + genomics** — cBioPortal study ``crc_orion_2024`` (74 patients,
  DFCI OncoPanel targeted sequencing). Fetched from the live REST API because
  this study is absent from the Datahub git repository.

The imaged subset is not assumed from the release README: it is *derived* and
cross-checked three ways (see ``build_slide_map``).

Outputs under ``<out>/``::

    raw/                          verbatim cBioPortal exports (tsv)
    crc_orion_slide_map.csv       image folder <-> patient, with the evidence
    crc_orion_clinical.csv        tidy per-patient table, all 74 patients
    crc_orion_by_slide.csv        per-slide rows in the kras_final.csv schema

Usage::

    python tools/build_crc_orion_labels.py \
        --out /mnt/d/YC.Liu/manifests/colon/crc_orion
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.cbioportal.org/api"
STUDY = "crc_orion_2024"
BUCKET_HOST = "https://lin-2023-orion-crc.s3.amazonaws.com"
MUT_PROFILE = f"{STUDY}_mutations"
CNA_PROFILE = f"{STUDY}_cna"
SV_PROFILE = f"{STUDY}_structural_variants"

# Genes carried as explicit label columns. Every one is on all four OncoPanel
# versions present in this study (asserted in `check_panel_coverage`).
LABEL_GENES = ("KRAS", "NRAS", "BRAF", "PIK3CA", "APC", "TP53", "SMAD4")

# Days per month, used only to carry cBioPortal's OS/PFS months onto the
# `os_time_days` column that the other cohorts use. Recorded in
# `os_time_basis` so the derivation is never mistaken for recorded days.
DAYS_PER_MONTH = 30.4375

# kras_final.csv column order — the Orion by-slide table leads with exactly
# these so it concatenates onto the existing manifest without a reindex.
KRAS_FINAL_COLUMNS = [
    "slide_uid", "patient_uid", "specimen_uid", "cohort", "subcohort",
    "specimen_role", "tumor_site_raw", "tumor_site_group", "tumor_site_meaning",
    "metastatic_site_group", "sex", "age_at_diagnosis", "msi_dmmr", "braf",
    "kras", "kras_subvariant", "nras", "ras", "t_stage", "n_stage", "m_stage",
    "stage_group", "stage_group_major", "ajcc_edition", "vital_status",
    "os_event", "os_time_days", "os_time_basis", "survival_analysis_eligible",
    "mortality_1y", "mortality_1y_basis", "mortality_2y", "mortality_2y_basis",
    "include", "qc_flags", "qc_note",
]

# Orion `LOCATION` -> the tumor_site_group vocabulary already in kras_final.
# Rectosigmoid groups to Rectum, matching the existing 'rectosigmoid' rows.
SITE_GROUP = {
    "Sigmoid": "Colon", "Ascending": "Colon", "Cecum": "Colon",
    "Descending": "Colon", "Transverse": "Colon",
    "Rectum": "Rectum", "Rectosigmoid": "Rectum",
    "Appendix": "Appendix",
}

_TNM_RE = re.compile(
    r"^(?P<tpre>y?p?|c)?T(?P<t>4a|4b|4|3|2|1|is|X)"
    r"_?N(?P<n>0|1a|1b|1c|1|2a|2b|2|X)"
    r"_(?P<mpre>[cp])?M(?P<m>0|1a|1b|1c|1|X)$"
)


# --------------------------------------------------------------------------
# fetch


def _get(url: str, cache: Path, name: str) -> object:
    f = cache / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    with urllib.request.urlopen(url, timeout=300) as r:
        payload = r.read()
    f.write_bytes(payload)
    return json.loads(payload)


def _post(url: str, body: dict, cache: Path, name: str) -> object:
    f = cache / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        payload = r.read()
    f.write_bytes(payload)
    return json.loads(payload)


def fetch_all(cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    big = "pageSize=100000"
    d = {
        "study": _get(f"{API}/studies/{STUDY}", cache, "study"),
        "patients": _get(f"{API}/studies/{STUDY}/patients?{big}", cache, "patients"),
        "samples": _get(f"{API}/studies/{STUDY}/samples?projection=DETAILED&{big}",
                        cache, "samples"),
        "attrs": _get(f"{API}/studies/{STUDY}/clinical-attributes?projection=DETAILED&{big}",
                      cache, "clin_attrs"),
        "clin_patient": _get(
            f"{API}/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT"
            f"&projection=DETAILED&{big}", cache, "clin_patient"),
        "clin_sample": _get(
            f"{API}/studies/{STUDY}/clinical-data?clinicalDataType=SAMPLE"
            f"&projection=DETAILED&{big}", cache, "clin_sample"),
        "mutations": _post(
            f"{API}/molecular-profiles/{MUT_PROFILE}/mutations/fetch"
            f"?projection=DETAILED&{big}",
            {"sampleListId": f"{STUDY}_all"}, cache, "mutations"),
        "panel_data": _post(
            f"{API}/molecular-profiles/{MUT_PROFILE}/gene-panel-data/fetch",
            {"sampleListId": f"{STUDY}_all"}, cache, "gpd_mut"),
        "cna": _post(
            f"{API}/molecular-profiles/{CNA_PROFILE}/discrete-copy-number/fetch"
            "?discreteCopyNumberEventType=ALL&projection=DETAILED",
            {"sampleListId": f"{STUDY}_all"}, cache, "cna"),
        "sv": _post(f"{API}/structural-variant/fetch",
                    {"molecularProfileIds": [SV_PROFILE]}, cache, "sv"),
    }
    panels = sorted({r["genePanelId"] for r in d["panel_data"]})
    d["panels"] = {
        p: _get(f"{API}/gene-panels/{urllib.parse.quote(p)}", cache, f"panel_{p}")
        for p in panels
    }
    return d


def fetch_s3_listing(cache: Path) -> list[dict]:
    """List the bucket over anonymous HTTPS (no AWS credentials needed)."""
    f = cache / "s3_listing.json"
    if f.exists():
        return json.loads(f.read_text())
    out, token = [], ""
    while True:
        url = f"{BUCKET_HOST}/?list-type=2&prefix=data/&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token, safe="")
        with urllib.request.urlopen(url, timeout=300) as r:
            xml = r.read().decode()
        for m in re.finditer(r"<Contents>(.*?)</Contents>", xml, re.S):
            b = m.group(1)
            out.append({
                "key": re.search(r"<Key>(.*?)</Key>", b).group(1),
                "etag": re.search(r"<ETag>&quot;(.*?)&quot;</ETag>", b).group(1),
                "size": int(re.search(r"<Size>(\d+)</Size>", b).group(1)),
            })
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            break
        token = m.group(1)
    f.write_text(json.dumps(out, indent=1))
    return out


def _attr(xml: str, name: str) -> str:
    """First value of an XML attribute, or '' if the attribute is absent."""
    m = re.search(rf'{name}="([^"]*)"', xml)
    return m.group(1) if m else ""


def fetch_ome_headers(listing: list[dict], cache: Path) -> dict[str, dict]:
    """Read each H&E file's OME-XML header with a ranged request.

    The pixel size lives only in the OME-XML, which `palom` writes as the very
    last ~700 bytes of the file, so a suffix Range fetch reads it without
    pulling the gigabytes in between. This matters because the TIFF resolution
    tags are empty (`ResolutionUnit = none`), so OpenSlide reports no mpp and
    the mpp-driven patcher has nothing to rescale from — the value has to be
    supplied externally via `data.custom_list_of_wsis`.
    """
    f = cache / "ome_headers.json"
    if f.exists():
        return json.loads(f.read_text())
    out = {}
    for rec in listing:
        if not rec["key"].endswith("-registered.ome.tif"):
            continue
        slide = rec["key"].split("/")[1]
        req = urllib.request.Request(f"{BUCKET_HOST}/{rec['key']}",
                                     headers={"Range": "bytes=-16384"})
        with urllib.request.urlopen(req, timeout=300) as r:
            buf = r.read()
        m = re.search(rb"<OME\b.*?</OME>", buf, re.S)
        if not m:
            raise SystemExit(f"{slide}: no OME-XML in the last 16 KB")
        xml = m.group(0).decode("utf-8", "replace")
        out[slide] = {
            "native_mpp": _attr(xml, "PhysicalSizeX"),
            "native_mpp_y": _attr(xml, "PhysicalSizeY"),
            "mpp_unit": _attr(xml, "PhysicalSizeXUnit"),
            "size_x": _attr(xml, "SizeX"),
            "size_y": _attr(xml, "SizeY"),
            "size_c": _attr(xml, "SizeC"),
            "pixel_type": _attr(xml, "Type"),
            "ome_creator": _attr(xml, "Creator"),
        }
    f.write_text(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------
# reshape


def pivot_clinical(rows: list[dict], id_key: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for r in rows:
        out[r[id_key]][r["clinicalAttributeId"]] = r["value"]
    return dict(out)


def check_panel_coverage(data: dict) -> dict[str, str]:
    """Assert every label gene is on every panel actually used in the study.

    Without this, "no mutation row" cannot be read as wild-type: an absent
    call and an unassayed gene look identical in a targeted-panel MAF.
    """
    used = {r["genePanelId"] for r in data["panel_data"]}
    for pid in sorted(used):
        genes = {g["hugoGeneSymbol"] for g in data["panels"][pid]["genes"]}
        missing = [g for g in LABEL_GENES if g not in genes]
        if missing:
            raise SystemExit(
                f"panel {pid} does not cover {missing}; wild-type calls for "
                "those genes would be unsafe — narrow LABEL_GENES or add a "
                "per-gene 'unknown' rule before continuing")
    profiled = {r["sampleId"] for r in data["panel_data"] if r.get("profiled")}
    all_samples = {s["sampleId"] for s in data["samples"]}
    if profiled != all_samples:
        raise SystemExit(f"unprofiled samples: {sorted(all_samples - profiled)}")
    return {r["sampleId"]: r["genePanelId"] for r in data["panel_data"]}


def gene_calls(mutations: list[dict]) -> dict[str, dict[str, list[str]]]:
    """sample -> gene -> sorted protein changes (all rows here are non-silent)."""
    out: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for m in mutations:
        out[m["sampleId"]][m["gene"]["hugoGeneSymbol"]].append(
            m.get("proteinChange") or "")
    return {s: {g: sorted(v) for g, v in d.items()} for s, d in out.items()}


def build_slide_map(listing: list[dict], clin_p: dict,
                    headers: dict[str, dict]) -> list[dict]:
    """Map each S3 image folder to a cBioPortal patient, and prove it.

    Three independent checks, all of which must agree:

    1. **folder number** — ``CRC07`` / ``CRC33_01`` -> ``C7`` / ``C33``.
    2. **filename identifier** — the scan filenames embed either the LSP
       specimen id (``18459_LSP10353_...``) or the case id
       (``19510_C11_...``, ``19510_P37-S83_C40_...``).
    3. **SLIDE_ID** — the cBioPortal patient attribute holding the LSP id.

    Check 2 resolves to a patient via check 3 for the LSP-named files and
    directly for the C-named ones, so the release README is corroborated by
    the data rather than trusted.
    """
    slide_by_lsp = {v: p for p, a in clin_p.items() if (v := a.get("SLIDE_ID"))}
    rows = []
    for rec in sorted(listing, key=lambda r: r["key"]):
        if not rec["key"].endswith("-registered.ome.tif"):
            continue
        folder = rec["key"].split("/")[1]          # CRC01 .. CRC33_01 .. CRC40
        basename = rec["key"].split("/")[-1]
        from_folder = "C" + str(int(folder[3:5]))

        lsp = re.search(r"_(LSP\d+)_", basename)
        cid = re.search(r"_C(\d+)_US_SCAN", basename)
        if lsp:
            from_name = slide_by_lsp.get(lsp.group(1))
            evidence = f"filename LSP={lsp.group(1)} -> SLIDE_ID"
        elif cid:
            from_name = f"C{int(cid.group(1))}"
            evidence = f"filename case id C{int(cid.group(1))}"
        else:
            from_name, evidence = None, "no identifier in filename"

        agree = from_name is not None and from_name == from_folder
        rows.append({
            "slide_id": folder,
            "patient_id": from_folder,
            "specimen_id": folder,
            "lsp_slide_id": clin_p[from_folder].get("SLIDE_ID", ""),
            "htan_participant_id": clin_p[from_folder].get("HTAN_PARTICIPANT_ID", ""),
            "s3_key": rec["key"],
            "s3_basename": basename,
            "size_bytes": rec["size"],
            "etag": rec["etag"],
            **headers.get(folder, {}),
            "patient_from_folder": from_folder,
            "patient_from_filename": from_name or "",
            "mapping_evidence": evidence,
            "mapping_verified": "yes" if agree else "NO",
        })
    bad = [r["slide_id"] for r in rows if r["mapping_verified"] != "yes"]
    if bad:
        raise SystemExit(f"slide->patient mapping unverified for {bad}")
    return rows


def parse_tnm(value: str) -> dict[str, str]:
    """Split an Orion TNM string into the kras_final t/n/m vocabulary.

    Stage prefixes are stripped from the stage tokens but the neoadjuvant
    ``y``/``yp`` marker is kept separately: those resections are post-therapy
    and their morphology is not comparable to treatment-naive tissue.
    """
    m = _TNM_RE.match((value or "").strip())
    if not m:
        return {"t_stage": "", "n_stage": "", "m_stage": "", "neoadjuvant": ""}
    return {
        "t_stage": "T" + m.group("t"),
        "n_stage": "N" + m.group("n"),
        "m_stage": "M" + m.group("m"),
        "neoadjuvant": "yes" if (m.group("tpre") or "").startswith("y") else "no",
    }


def survival(months: str, status: str, horizon_days: float) -> tuple[str, str]:
    """kras_final mortality_<h> / mortality_<h>_basis for one horizon."""
    if not months:
        return "", "invalid_time_or_status"
    days = float(months) * DAYS_PER_MONTH
    dead = status.startswith("1")
    if dead:
        return ("1", "death_on_or_before_horizon") if days <= horizon_days \
            else ("0", "death_after_horizon")
    return ("0", "followed_alive_through_horizon") if days >= horizon_days \
        else ("", "followup_ended_before_horizon")


# --------------------------------------------------------------------------
# write


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows x {len(cols)} cols)")


def write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def export_raw(out: Path, data: dict, panel_of: dict[str, str]) -> None:
    raw = out / "raw"
    cp, cs = data["clin_patient"], data["clin_sample"]
    write_tsv(raw / "clinical_patient.tsv",
              [{"patientId": r["patientId"],
                "attribute": r["clinicalAttributeId"], "value": r["value"]} for r in cp],
              ["patientId", "attribute", "value"])
    write_tsv(raw / "clinical_sample.tsv",
              [{"sampleId": r["sampleId"], "patientId": r["patientId"],
                "attribute": r["clinicalAttributeId"], "value": r["value"]} for r in cs],
              ["sampleId", "patientId", "attribute", "value"])

    mut_cols = ["sampleId", "patientId", "Hugo_Symbol", "entrezGeneId", "chr",
                "startPosition", "endPosition", "referenceAllele", "variantAllele",
                "proteinChange", "mutationType", "variantType", "ncbiBuild",
                "refseqMrnaId", "proteinPosStart", "proteinPosEnd", "tumorAltCount",
                "tumorRefCount", "mutationStatus", "validationStatus", "center",
                "genePanelId"]
    write_tsv(raw / "mutations.tsv",
              [{**m, "Hugo_Symbol": m["gene"]["hugoGeneSymbol"],
                "genePanelId": panel_of.get(m["sampleId"], "")}
               for m in data["mutations"]], mut_cols)

    entrez = {}
    for m in data["mutations"]:
        entrez[m["entrezGeneId"]] = m["gene"]["hugoGeneSymbol"]
    write_tsv(raw / "cna.tsv",
              [{"sampleId": c["sampleId"], "patientId": c["patientId"],
                "entrezGeneId": c["entrezGeneId"],
                "Hugo_Symbol": (c.get("gene") or {}).get("hugoGeneSymbol")
                               or entrez.get(c["entrezGeneId"], ""),
                "alteration": c["alteration"]}
               for c in data["cna"] if c["alteration"] != 0],
              ["sampleId", "patientId", "entrezGeneId", "Hugo_Symbol", "alteration"])

    write_tsv(raw / "structural_variants.tsv", data["sv"],
              ["sampleId", "patientId", "site1HugoSymbol", "site2HugoSymbol",
               "eventInfo", "variantClass", "svStatus"])

    write_tsv(raw / "gene_panel_coverage.tsv",
              [{"sampleId": r["sampleId"], "genePanelId": r["genePanelId"],
                "profiled": r.get("profiled")} for r in data["panel_data"]],
              ["sampleId", "genePanelId", "profiled"])
    write_tsv(raw / "gene_panel_genes.tsv",
              [{"genePanelId": pid, "Hugo_Symbol": g["hugoGeneSymbol"],
                "entrezGeneId": g["entrezGeneId"]}
               for pid, p in sorted(data["panels"].items()) for g in p["genes"]],
              ["genePanelId", "Hugo_Symbol", "entrezGeneId"])


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/d/YC.Liu/manifests/colon/crc_orion"))
    ap.add_argument("--cache", type=Path, default=None,
                    help="directory for raw API/S3 payloads (default <out>/.cache)")
    args = ap.parse_args()
    out = args.out
    cache = args.cache or out / ".cache"

    print("fetching cBioPortal study", STUDY)
    data = fetch_all(cache)
    print("listing s3://lin-2023-orion-crc/data")
    listing = fetch_s3_listing(cache)
    print("reading OME-XML headers (ranged)")
    headers = fetch_ome_headers(listing, cache)

    clin_p = pivot_clinical(data["clin_patient"], "patientId")
    clin_s = pivot_clinical(data["clin_sample"], "sampleId")
    sample_of = {s["patientId"]: s["sampleId"] for s in data["samples"]}
    panel_of = check_panel_coverage(data)
    calls = gene_calls(data["mutations"])

    print("exporting raw tables")
    export_raw(out, data, panel_of)

    print("building slide map")
    slide_map = build_slide_map(listing, clin_p, headers)
    write_csv(out / "crc_orion_slide_map.csv", slide_map)
    imaged = {r["patient_id"] for r in slide_map}

    # cBioPortal's own COHORT attribute independently marks the imaged set.
    cohort1 = {p for p, a in clin_p.items() if a.get("COHORT") == "1"}
    if cohort1 != imaged:
        print(f"  WARNING: COHORT==1 {sorted(cohort1)} != imaged {sorted(imaged)}",
              file=sys.stderr)
    else:
        print(f"  imaged set ({len(imaged)}) matches cBioPortal COHORT==1")

    print("building per-patient clinical table")
    patients = []
    for pid in sorted(clin_p, key=lambda p: int(p[1:])):
        a, s = clin_p[pid], clin_s.get(sample_of.get(pid, ""), {})
        c = calls.get(sample_of.get(pid, ""), {})
        tnm = parse_tnm(a.get("TNM_AT_DIAGNOSIS", ""))
        row = {
            "patient_id": pid,
            "sample_id": sample_of.get(pid, ""),
            "htan_participant_id": a.get("HTAN_PARTICIPANT_ID", ""),
            "lsp_slide_id": a.get("SLIDE_ID", ""),
            "has_he_wsi": "yes" if pid in imaged else "no",
            "orion_cohort": a.get("COHORT", ""),
            "sex": a.get("SEX", "").lower(),
            "age_at_diagnosis": a.get("AGE_DX", ""),
            "age_at_sequencing": s.get("AGE_AT_SEQ", ""),
            "primary_status": a.get("PRIMARY_STATUS", ""),
            "location": a.get("LOCATION", ""),
            "side": s.get("SIDE", ""),
            "oncotree_code": s.get("ONCOTREE_CODE", ""),
            "histology": a.get("HISTOLOGY", ""),
            "grade": a.get("GRADE", ""),
            "mucinous": s.get("MUCINOUS", ""),
            "stage_at_diagnosis": a.get("STAGE_AT_DIAGNOSIS", ""),
            "stage_number": s.get("STAGE NUMBER", ""),
            "tnm_at_diagnosis": a.get("TNM_AT_DIAGNOSIS", ""),
            **tnm,
            "distant_mets_at_diagnosis": a.get("DISTANT_METS_AT_DIAGNOSIS", ""),
            "lvi": a.get("LVI", ""), "pni": a.get("PNI", ""),
            "tumor_deposits": a.get("DEPOSITS", ""),
            "tumor_border": a.get("BORDER", ""),
            "til": a.get("TIL", ""), "til_num": s.get("TIL_NUM", ""),
            "mmr_status": a.get("MMR_STATUS", ""),
            "mmr_ihc": a.get("MMR_IHC", ""),
            "tipmmr": a.get("TIPMMR", ""),
            "hypermutant": a.get("HYPERMUTANT", ""),
            "tmb": s.get("TMB", ""),
            "mutation_count": s.get("MUTATION_COUNT", ""),
            "tumor_purity": s.get("TUMOR_PURITY", ""),
            "panel_version": s.get("PANEL_VERSION", ""),
            "gene_panel_id": panel_of.get(sample_of.get(pid, ""), ""),
            "somatic_status": s.get("SOMATIC_STATUS", ""),
            "recurrence": a.get("RECURRENCE", ""),
            "location_of_recurrence": a.get("LOCATION_OF_RECURRENCE", ""),
            "os_status": a.get("OS_STATUS", ""), "os_months": a.get("OS_MONTHS", ""),
            "pfs_status": a.get("PFS_STATUS", ""), "pfs_months": a.get("PFS_MONTHS", ""),
        }
        for g in LABEL_GENES:
            variants = c.get(g, [])
            row[f"{g.lower()}"] = "mutant" if variants else "wild_type"
            row[f"{g.lower()}_variant"] = ";".join(variants)
        row["braf_v600e"] = "yes" if "V600E" in c.get("BRAF", []) else "no"
        row["ras"] = "mutant" if (row["kras"] == "mutant" or row["nras"] == "mutant") \
            else "wild_type"
        patients.append(row)
    write_csv(out / "crc_orion_clinical.csv", patients)

    print("building per-slide table (kras_final schema)")
    by_pid = {p["patient_id"]: p for p in patients}
    slides = []
    for sm in slide_map:
        p = by_pid[sm["patient_id"]]
        months, status = p["os_months"], p["os_status"]
        m1, m1b = survival(months, status, 365.25)
        m2, m2b = survival(months, status, 730.5)
        flags, notes = [], []
        if p["primary_status"] != "Primary Tumor":
            # Record the flag WITH the evidence that contradicts it. cBioPortal's
            # "Metastasis or Otherwise" is a catch-all, and for a resection whose
            # own TNM is cM0 at a primary colorectal site it cannot mean the
            # section is metastatic: an M0 stage and a pathological T category
            # both require the resected primary. specimen_role therefore stays
            # `primary` and the flag is carried for review, not acted on.
            flags.append("specimen")
            site, m = p["location"], p["m_stage"]
            note = f"PRIMARY_STATUS={p['primary_status']}"
            if m == "M0" and SITE_GROUP.get(site):
                note += (f", but TNM {p['tnm_at_diagnosis']} is {m} at a primary "
                         f"site ({site}) — specimen_role kept primary; flag unexplained")
            notes.append(note)
        if p["neoadjuvant"] == "yes":
            flags.append("treatment")
            notes.append(f"neoadjuvant-treated resection ({p['tnm_at_diagnosis']})")
        if sm["slide_id"].startswith("CRC33"):
            notes.append("one of two image specimens for patient C33")
        slides.append({
            "slide_uid": f"ORION:{sm['slide_id']}",
            "patient_uid": f"ORION:{sm['patient_id']}",
            "specimen_uid": f"ORION:{sm['specimen_id']}",
            "cohort": "Orion", "subcohort": "Orion-CRC",
            "specimen_role": "primary",
            "tumor_site_raw": p["location"],
            "tumor_site_group": SITE_GROUP.get(p["location"], "Other or unknown"),
            "tumor_site_meaning": "specimen_site",
            "metastatic_site_group": "",
            "sex": p["sex"], "age_at_diagnosis": p["age_at_diagnosis"],
            "msi_dmmr": {"pMMR": "MSS/pMMR", "dMMR": "MSI/dMMR"}.get(
                p["mmr_status"], "unknown"),
            "braf": p["braf"], "kras": p["kras"],
            "kras_subvariant": p["kras_variant"], "nras": p["nras"], "ras": p["ras"],
            "t_stage": p["t_stage"], "n_stage": p["n_stage"], "m_stage": p["m_stage"],
            "stage_group": f"Stage {p['stage_at_diagnosis']}"
                           if p["stage_at_diagnosis"] else "",
            "stage_group_major": re.match(r"[IV]+", p["stage_at_diagnosis"] or "").group(0)
                                 if p["stage_at_diagnosis"] else "",
            "ajcc_edition": "",
            "vital_status": "dead" if status.startswith("1") else "alive",
            "os_event": "1" if status.startswith("1") else "0",
            "os_time_days": round(float(months) * DAYS_PER_MONTH, 1) if months else "",
            "os_time_basis": "event_and_censoring_from_months",
            "survival_analysis_eligible": "True" if months else "False",
            "mortality_1y": m1, "mortality_1y_basis": m1b,
            "mortality_2y": m2, "mortality_2y_basis": m2b,
            "include": "yes",
            "qc_flags": ";".join(flags), "qc_note": "; ".join(notes),
            # --- extras beyond the kras_final schema ---
            "slide_id": sm["slide_id"], "patient_id": sm["patient_id"],
            "s3_key": sm["s3_key"], "s3_basename": sm["s3_basename"],
            "wsi_filename": f"{sm['slide_id']}.tif",
            "native_mpp": sm.get("native_mpp", ""),
            "size_x": sm.get("size_x", ""), "size_y": sm.get("size_y", ""),
            "braf_variant": p["braf_variant"], "braf_v600e": p["braf_v600e"],
            "nras_variant": p["nras_variant"],
            "pik3ca": p["pik3ca"], "pik3ca_variant": p["pik3ca_variant"],
            "apc": p["apc"], "tp53": p["tp53"], "smad4": p["smad4"],
            "mmr_ihc": p["mmr_ihc"], "tipmmr": p["tipmmr"],
            "hypermutant": p["hypermutant"], "tmb": p["tmb"],
            "tumor_purity": p["tumor_purity"], "panel_version": p["panel_version"],
            "gene_panel_id": p["gene_panel_id"], "somatic_status": p["somatic_status"],
            "neoadjuvant": p["neoadjuvant"], "primary_status": p["primary_status"],
            "grade": p["grade"], "side": p["side"], "histology": p["histology"],
            "lvi": p["lvi"], "pni": p["pni"], "til": p["til"],
            "os_months": months, "pfs_months": p["pfs_months"],
            "pfs_status": p["pfs_status"],
        })
    extras = [c for c in slides[0] if c not in KRAS_FINAL_COLUMNS]
    write_csv(out / "crc_orion_by_slide.csv", slides, KRAS_FINAL_COLUMNS + extras)

    # `data.custom_list_of_wsis` input: OpenSlide cannot see the mpp in these
    # files (see fetch_ome_headers), so extraction must be told it explicitly.
    wsi_list = [{"wsi": f"crc_orion/{s['slide_id']}.tif", "mpp": s["native_mpp"]}
                for s in slides]
    write_csv(out / "crc_orion_wsi_mpp.csv", wsi_list, ["wsi", "mpp"])

    n_pat = len({s["patient_uid"] for s in slides})
    print(f"\nsummary: {len(slides)} slides / {n_pat} patients with H&E + genomics")
    for col in ("kras", "nras", "braf", "ras", "msi_dmmr"):
        print(f"  {col:10s} {dict(collections.Counter(s[col] for s in slides))}")
    print("  kras alleles",
          dict(collections.Counter(
              s["kras_subvariant"] for s in slides if s["kras"] == "mutant")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
