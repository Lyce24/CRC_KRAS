#!/usr/bin/env python3
"""Build reviews/v2 — the single-reader Aim 4 pathology study packets.

Produces two independent, blinded study parts for ONE pathologist. A second
reader could not be recruited, so between-reader agreement is unobtainable
and the design substitutes what a single reader can still support.

PART A — montage replication (~30 min). The reader fills the UNREAD
corrected-v2 blank forms on the sealed corrected-v2 montages. This retires
the final-v5 packet-generation bridge, because the corrected-v2 material
stops being unread, and it adds a third independent description of the same
prototypes alongside the existing original-render human read and the
non-authoritative machine read.

PART B — case-level morphology study (~10-14 h). A frozen, stratified sample
of primary cases is scored on whole slides for the two frozen prototype
correlates (extracellular mucin; gland formation and differentiation) plus
the routine features Aim 1's clinical comparator lacks (grade proxy,
histotype, budding, TILs). Stratification is KRAS x cohort x p17-abundance
tertile, so the sample supports a dose-response test of pathologist-scored
mucin against frozen p17 abundance, and a mediation test of whether
pathologist-recognisable morphology accounts for the WSI score's KRAS
association. Both endpoints compare reader against model, so neither needs a
second reader; case count, not reader count, drives their power.

RELIABILITY. Because between-reader agreement is unavailable, a KRAS-balanced
subset of cases is scored twice under different case identifiers, shuffled
among the originals, yielding an INTRA-reader (test-retest) estimate. It must
be reported as intra-reader; between-reader reproducibility remains an
explicit limitation of the study.

BLINDING. Readers receive case identifiers only. Cohort, patient identity,
KRAS/MSI/BRAF/NRAS status, p17/p28 values and every model score are held in
`KEYS_DO_NOT_DISTRIBUTE/`, which is never copied into a reader folder. Slide
filenames leak cohort (for example TCGA barcodes), so the tool emits a link
plan that renames every slide to its blinded case identifier; the coordinator
must run it and hand over only the renamed copies.

Nothing here is analysed. Read-only over frozen inputs; writes only under
reviews/v2.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import paths  # noqa: E402

AIM4 = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim4_cap8192_corrected_v2_20260820")
PACKETS = AIM4 / "review_bundles" / "k32"
PROFILES = AIM4 / "profiles" / "patient_profiles_k32.parquet"
E2A = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim2_cap8192_v4_20260819/e2a")
SLIDE_ROOT = Path("/mnt/d/YC.Liu/slides/colon")
# Canonical slide directories only, per slides/colon/README.md. The R/B-swapped
# SurGen originals ("never extract from these"), the CZI references and the RIH
# quarantine are deliberately excluded: a colour-swapped slide would silently
# corrupt a mucin assessment, which is the whole point of Part B.
CANONICAL_SLIDE_DIRS = ("CPTAC_COAD", "TCGA", "SURGEN", "rih")
EXCLUDED_SLIDE_DIRS = ("SURGEN_rb_swapped_original", "SurGen_ORI", "rih_quarantine", "rih_fixed")
OUTPUT = REPO / "reviews" / "v2"
READERS = ("reader_1",)
SELECTION_SEED = 20260821
PER_CELL = 10  # KRAS(2) x cohort(4) x p17 tertile(3) = 24 cells -> ~240 cases
# Embedded intra-reader replicates. With a single reader, between-reader
# agreement is unobtainable, so a blinded test-retest subset is scored twice
# under different case identifiers to preserve a reliability estimate.
N_REPLICATES = 45
TARGETS = ("CPTAC", "RIH", "SurGen", "TCGA")

CASE_FORM_COLUMNS = [
    "case_id", "assessable", "tumour_present",
    "extracellular_mucin_extent", "mucin_pools_with_floating_clusters", "signet_ring_cells",
    "gland_formation", "differentiation", "predominant_architecture", "histotype",
    "dirty_necrosis", "tumour_budding", "tils_or_crohn_like", "desmoplasia",
    "confidence", "reviewer_id", "review_date", "blinding_attestation", "free_text",
]
CASE_VOCAB = {
    "assessable": "yes | partial | no",
    "tumour_present": "yes | no",
    "extracellular_mucin_extent": "none | focal_lt10 | moderate_10_50 | extensive_gt50",
    "mucin_pools_with_floating_clusters": "absent | present | not_assessable",
    "signet_ring_cells": "absent | present_lt50 | present_ge50 | not_assessable",
    "gland_formation": "gt95 | pct50_95 | lt50 | not_assessable",
    "differentiation": "well | moderate | poor | not_assessable",
    "predominant_architecture": "glandular | cribriform | solid | papillary | mucinous | mixed | not_assessable",
    "histotype": "adenocarcinoma_nos | mucinous | signet_ring | medullary | serrated | other | not_assessable",
    "dirty_necrosis": "absent | focal | extensive | not_assessable",
    "tumour_budding": "bd1_low | bd2_intermediate | bd3_high | not_assessable",
    "tils_or_crohn_like": "absent | present | not_assessable",
    "desmoplasia": "absent | mild | marked | not_assessable",
    "confidence": "low | moderate | high",
    "reviewer_id": "free text",
    "review_date": "ISO date, YYYY-MM-DD",
    "blinding_attestation": "confirmed_no_key_access",
    "free_text": "free text; write 'none' if nothing to add",
}


def load_case_frame() -> pd.DataFrame:
    """One row per development primary patient with p17/p28 and a held-out score."""
    profiles = pd.read_parquet(PROFILES)
    e0 = profiles[profiles["arm"] == "e0"]
    wide = e0.pivot_table(index="patient_id", columns="prototype", values="abundance", aggfunc="first")
    wide = wide[[17, 28]].rename(columns={17: "p17_abundance", 28: "p28_abundance"})

    manifest = pd.read_csv(paths.DEV_MANIFEST, low_memory=False)
    patients = manifest.drop_duplicates("patient_id")[
        ["patient_id", "cohort", "subcohort", "kras", "msi_dmmr", "braf", "nras",
         "tumor_site_group", "stage_group_major", "age_at_diagnosis", "sex"]
    ]
    slides = (
        manifest.groupby("patient_id")["slide_id"]
        .apply(lambda s: sorted(s.astype(str)))
        .rename("slide_ids")
        .reset_index()
    )
    frame = patients.merge(wide.reset_index(), on="patient_id", validate="one_to_one")
    frame = frame.merge(slides, on="patient_id", validate="one_to_one")

    scores = []
    for target in TARGETS:
        path = E2A / "calibrated" / f"cap8192_{target.lower()}_primary.parquet"
        block = pd.read_parquet(path)[["patient_id", "label", "mean_logit"]]
        block = block.rename(columns={"mean_logit": "wsi_heldout_logit"})
        scores.append(block)
    scores = pd.concat(scores, ignore_index=True)
    frame = frame.merge(scores, on="patient_id", how="left", validate="one_to_one")
    if frame["wsi_heldout_logit"].isna().any():
        missing = int(frame["wsi_heldout_logit"].isna().sum())
        raise AssertionError(f"{missing} patients lack a held-out E2a score")
    return frame


def stratified_sample(frame: pd.DataFrame, per_cell: int) -> pd.DataFrame:
    """KRAS x cohort x p17-tertile deterministic sample; cells smaller than the
    quota contribute every patient they have."""
    frame = frame.copy()
    # Rank-based tertiles: p17 abundance has many exact ties (patients with no
    # p17 tiles at all), which collapse quantile edges; ranking with a stable
    # tie-break always yields exactly three within-cohort strata.
    def tertile(series: pd.Series) -> pd.Series:
        ranks = series.rank(method="first")
        edges = np.floor((ranks - 1) * 3 / len(series)).astype(int).clip(0, 2)
        return edges.map({0: "T1", 1: "T2", 2: "T3"})

    frame["p17_tertile"] = (
        frame.sort_values("patient_id")
        .groupby("cohort")["p17_abundance"]
        .transform(tertile)
    )
    rng = np.random.default_rng(SELECTION_SEED)
    picks: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for (cohort, kras, tertile), block in frame.groupby(["cohort", "kras", "p17_tertile"], sort=True):
        take = min(per_cell, len(block))
        order = block.sort_values("patient_id").index.to_numpy()
        rng.shuffle(order)
        picks.append(frame.loc[order[:take]])
        audit.append(
            {
                "cohort": cohort, "kras": kras, "p17_tertile": tertile,
                "available": int(len(block)), "selected": int(take),
            }
        )
    sample = pd.concat(picks).sort_values("patient_id").reset_index(drop=True)
    audit_frame = pd.DataFrame(audit)
    sample["replicate_of_patient"] = ""
    return _assign_case_ids(sample, audit_frame, rng)


def _assign_case_ids(
    sample: pd.DataFrame, audit_frame: pd.DataFrame, rng: np.random.Generator, n_replicates: int = N_REPLICATES
) -> pd.DataFrame:
    """Shuffle, add blinded replicate copies, then stamp opaque case ids.

    Replicates are drawn balanced on KRAS so the reliability estimate is not
    computed on a one-sided subset, receive their own case identifiers, and are
    shuffled among the originals so the reader cannot pair them by position.
    """
    replicates = pd.DataFrame(columns=sample.columns)
    if n_replicates:
        per_class = n_replicates // 2
        chosen: list[np.ndarray] = []
        for value in ("mutant", "wild_type"):
            pool = sample.index[sample["kras"] == value].to_numpy()
            take = min(per_class, len(pool))
            chosen.append(rng.choice(pool, take, replace=False))
        replicates = sample.loc[np.concatenate(chosen)].copy()
        replicates["replicate_of_patient"] = replicates["patient_id"].to_numpy()

    combined = pd.concat([sample, replicates], ignore_index=True)
    order = np.arange(len(combined))
    rng.shuffle(order)  # case ids carry no ordering information
    combined = combined.iloc[order].reset_index(drop=True)
    combined.insert(0, "case_id", [f"C{i + 1:03d}" for i in range(len(combined))])
    # map each replicate to the case id of its original
    original_id = (
        combined.loc[combined["replicate_of_patient"] == "", ["patient_id", "case_id"]]
        .set_index("patient_id")["case_id"]
    )
    combined["replicate_of_case"] = combined["replicate_of_patient"].map(original_id).fillna("")
    combined.attrs["audit"] = audit_frame.to_dict(orient="records")
    combined.attrs["n_unique"] = int(len(sample))
    combined.attrs["n_replicates"] = int(len(replicates))
    return combined


def write_part_a(root: Path) -> dict[str, Any]:
    part = root / "part_a_montage_replication"
    inventory: dict[str, Any] = {}
    for reader in READERS:
        for bundle in ("base", "attention_addendum"):
            source = PACKETS / bundle / "packet"
            destination = part / reader / bundle
            shutil.copytree(source, destination)
            # the blank master stays blank; the reader edits their own copy
            (destination / "review_form.csv").rename(destination / "review_form_TEMPLATE.csv")
            shutil.copy(
                destination / "review_form_TEMPLATE.csv",
                destination / f"review_form_{reader}.csv",
            )
        inventory[reader] = sorted(str(p.relative_to(part)) for p in (part / reader).rglob("*.csv"))
    return inventory


def write_part_b(root: Path, sample: pd.DataFrame) -> None:
    part = root / "part_b_case_level"
    part.mkdir(parents=True)

    # blinded case list for readers: identifiers and slide counts only
    reader_list = sample[["case_id"]].copy()
    reader_list["n_slides"] = sample["slide_ids"].apply(len)
    reader_list.to_csv(part / "case_list.csv", index=False)

    for reader in READERS:
        with open(part / f"case_scoring_form_{reader}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CASE_FORM_COLUMNS, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for _, row in reader_list.iterrows():
                writer.writerow(
                    {c: ("pending" if c != "case_id" else row["case_id"]) for c in CASE_FORM_COLUMNS}
                )

    index_by_stem: dict[str, Path] = {}
    for directory in CANONICAL_SLIDE_DIRS:
        for candidate in sorted((SLIDE_ROOT / directory).glob("*")):
            if candidate.is_file() and candidate.suffix.lower() in {".svs", ".tif", ".tiff", ".scn"}:
                index_by_stem.setdefault(candidate.name.rsplit(".", 1)[0], candidate)

    plan_rows = []
    for _, row in sample.iterrows():
        for index, slide in enumerate(row["slide_ids"], start=1):
            source = index_by_stem.get(slide)
            plan_rows.append(
                {
                    "case_id": row["case_id"],
                    "slide_index": index,
                    "blinded_filename": f"{row['case_id']}_s{index}{source.suffix if source else ''}",
                    "source_path": str(source) if source else "",
                    "source_dir": source.parent.name if source else "",
                    "source_found": source is not None,
                }
            )
    plan = pd.DataFrame(plan_rows)
    bad = plan.loc[plan["source_dir"].isin(EXCLUDED_SLIDE_DIRS), "source_dir"]
    if len(bad):
        raise AssertionError(f"resolved {len(bad)} slides to non-canonical directories: {set(bad)}")
    if not plan["source_found"].all():
        missing = plan.loc[~plan["source_found"], "case_id"].tolist()
        raise AssertionError(
            f"{len(missing)} slides did not resolve in canonical directories, e.g. {missing[:5]}"
        )
    plan.to_csv(part / "slide_link_plan.csv", index=False)

    script = part / "make_blinded_slide_folder.sh"
    lines = [
        "#!/usr/bin/env bash",
        "# Creates a blinded slide folder for Part B. Run once, then hand readers",
        "# ONLY the resulting directory. Symlinks keep it cheap; use cp -L if the",
        "# readers' viewer cannot follow links across filesystems.",
        "set -euo pipefail",
        'OUT="${1:?usage: make_blinded_slide_folder.sh /path/to/blinded_slides}"',
        'mkdir -p "$OUT"',
    ]
    for _, row in plan[plan["source_found"]].iterrows():
        lines.append(f'ln -sf "{row["source_path"]}" "$OUT/{row["blinded_filename"]}"')
    lines.append('echo "linked $(ls "$OUT" | wc -l) slides into $OUT"')
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)

    (part / "VOCABULARY.md").write_text(
        "# Part B field vocabulary\n\nEvery field must be filled with exactly one listed "
        "value. Never leave a cell blank: `not_assessable` is an answer, a blank is an "
        "omission.\n\n| Field | Allowed values |\n| --- | --- |\n"
        + "\n".join(f"| `{k}` | {v} |" for k, v in CASE_VOCAB.items())
        + "\n"
    )


def write_keys(root: Path, sample: pd.DataFrame) -> None:
    keys = root / "KEYS_DO_NOT_DISTRIBUTE"
    keys.mkdir(parents=True)
    key = sample.copy()
    key["slide_ids"] = key["slide_ids"].apply(lambda v: ";".join(v))
    key.to_csv(keys / "part_b_case_key.csv", index=False)
    for bundle in ("base", "attention_addendum"):
        shutil.copy(
            PACKETS / bundle / "unblinding_key_DO_NOT_SHARE.csv",
            keys / f"part_a_{bundle}_key.csv",
        )
    audit = pd.DataFrame(sample.attrs["audit"])
    audit.to_csv(keys / "part_b_selection_audit.csv", index=False)
    (keys / "README.md").write_text(
        "# DO NOT DISTRIBUTE\n\nThis directory maps blinded identifiers to patients, "
        "cohorts, molecular status, prototype values and model scores for BOTH study "
        "parts. It must never be copied into a reader folder, and must not be opened "
        "by anyone who will read cases. Open it only after both readers have returned "
        "completed forms.\n"
    )


def write_readme(root: Path, sample: pd.DataFrame) -> None:
    audit = pd.DataFrame(sample.attrs["audit"])
    unique = sample[sample["replicate_of_patient"] == ""]
    by_cohort = unique.groupby("cohort").size().to_dict()
    by_kras = unique.groupby("kras").size().to_dict()
    n_unique = sample.attrs["n_unique"]
    n_rep = sample.attrs["n_replicates"]
    (root / "README_COORDINATOR.md").write_text(
        f"""# reviews/v2 — coordinator instructions

ONE blinded reader, two parts. Hand out only what each section names.
A second reader could not be recruited, so the design substitutes embedded
intra-reader replicates for between-reader agreement (Part B) and reports
Part A descriptively against the existing independent reads.

## What each reader receives

| Part | Give the reader | Do NOT give |
| --- | --- | --- |
| A | their `part_a_montage_replication/reader_1/` folder | any key, any prior read, the machine-read annex |
| B | `part_b_case_level/case_list.csv`, `case_scoring_form_reader_1.csv`, `VOCABULARY.md`, and the blinded slide folder | `KEYS_DO_NOT_DISTRIBUTE/`, `slide_link_plan.csv`, patient or cohort identifiers, which items repeat |

Before Part B, run `part_b_case_level/make_blinded_slide_folder.sh <dest>` once.
Slide filenames encode cohort (TCGA barcodes, RIH `SL-` numbers), so readers
must receive only the renamed copies.

## Part A — montage replication (~30 min)

Eleven montages: ten in `base`, one in `attention_addendum`. The reader fills
`review_form_reader_1.csv` in their folder, following the packet README's
completion rules. Work through all eleven before opening anything else.

This part also closes an open caveat: the corrected-v2 forms have never been
filled, so completing them removes the packet-generation bridge that the
final-v5 report had to document.

## Part B — case-level morphology ({len(sample)} scoring items, ~10-14 h)

Cases are whole primary colorectal resections, viewed as whole slides. For
each case the reader records extracellular mucin extent, whether mucin forms
pools containing floating tumour clusters, signet-ring morphology, gland
formation, differentiation, architecture, histotype, dirty necrosis, budding,
TILs/Crohn-like reaction and desmoplasia — then a confidence rating and free
text. `VOCABULARY.md` lists the allowed values; every field must be filled,
and `not_assessable` is an answer while a blank is an omission.

Readers describe morphology only. They are not told, and must not be told,
what any feature is expected to predict.

### Frozen sample

Selected {n_unique} of 1,486 development primaries, stratified
KRAS x cohort x within-cohort p17 tertile with a quota of {PER_CELL} per cell
(cells smaller than the quota contribute everything they have), drawn with
seed {SELECTION_SEED}. Case identifiers are randomised and carry no ordering.

* by cohort: {by_cohort}
* by KRAS: {by_kras}
* cells at quota: {int((audit['selected'] >= PER_CELL).sum())} of {len(audit)}

Balance on KRAS and on p17 tertile is what powers the two planned endpoints;
it does not make the sample representative of prevalence, and no prevalence
statistic should be computed from it.

### Embedded reliability replicates ({n_rep} items)

With one reader, between-reader agreement cannot be measured, so {n_rep} of
the selected cases appear a second time under different case identifiers,
KRAS-balanced and shuffled among the originals. The reader is not told which
items repeat and must not try to identify them. Scoring the full list
therefore yields an intra-reader (test-retest) reliability estimate — weaker
than between-reader agreement, and it must be reported as such, never as
inter-observer agreement.

Total scoring items: {len(sample)} = {n_unique} unique + {n_rep} replicates.

**Order matters.** Work the list top to bottom in one pass. If time runs
short, the primary endpoints need the unique cases, so do not stop
mid-list — tell the coordinator, who can identify which items remain.
A gap of a day or more between the first and second half strengthens the
test-retest estimate by reducing recall.

## After the reader returns

1. Store the completed forms outside the sealed packets.
2. Only then open `KEYS_DO_NOT_DISTRIBUTE/`.
3. Part A: report concordance descriptively against the two existing
   independent reads (Dr. Lu on the original renders; the non-authoritative
   machine read on these exact renders). Eleven montages cannot support a
   stable kappa under any design, so do not compute one. Completing these
   forms also retires the packet-generation bridge, because the corrected-v2
   material stops being unread.
4. Part B primary endpoints: (i) dose-response of pathologist-scored mucin
   extent against frozen p17 abundance; (ii) mediation — whether
   pathologist-scored morphology attenuates the association between the
   held-out WSI logit and KRAS status, adjusted for BRAF, MSI, NRAS, site,
   stage, histotype/grade and cohort.
5. Part B secondary: routine pathology features versus prototype scores
   versus the frozen WSI worklist ranking, as an Aim 1 comparator upgrade.
6. Part B reliability: intra-reader (test-retest) agreement from the embedded
   replicate pairs — weighted kappa for ordinal fields, ICC for the mucin
   extent score. Report it as INTRA-reader; it does not establish
   between-reader reproducibility, which remains a stated limitation.

Readers must not see Part B keys, the final-v5 machine-read annex, or any
prior unblinded root before their own forms are complete.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--per-cell", type=int, default=PER_CELL)
    args = parser.parse_args()
    root: Path = args.output
    if root.exists():
        raise SystemExit(f"append-only: {root} already exists")

    frame = load_case_frame()
    sample = stratified_sample(frame, args.per_cell)
    root.mkdir(parents=True)
    inventory = write_part_a(root)
    write_part_b(root, sample)
    write_keys(root, sample)
    write_readme(root, sample)

    plan = pd.read_csv(root / "part_b_case_level" / "slide_link_plan.csv")
    summary = {
        "created": True,
        "part_a_forms": inventory,
        "part_b_cases": int(len(sample)),
        "part_b_slides": int(len(plan)),
        "part_b_slides_found": int(plan["source_found"].sum()),
        "by_cohort": sample.groupby("cohort").size().to_dict(),
        "by_kras": sample.groupby("kras").size().to_dict(),
    }
    (root / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
