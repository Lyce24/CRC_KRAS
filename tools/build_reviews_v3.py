#!/usr/bin/env python3
"""Build reviews/v3 — a one-hour, single-reader case-level validation.

SCOPE DECISION. One pathologist, 30-60 minutes total. That budget buys one
task, so it is spent on the question that produces a NEW result rather than
re-confirming an existing one: does the frozen p17 prototype correspond to
extracellular mucin at the CASE level, and does p28 track gland formation?
Montage re-reading is dropped - two independent reads (Dr. Lu on the original
renders, the non-authoritative machine read on corrected-v2) already agree at
high confidence on p17/p28/p5, so a third description adds little per minute.

THE ASK. 120 whole primary CRC cases, two ordinal fields each, scored at low
power. Nothing free-text is required. Estimated 20-30 seconds per case once
in rhythm: roughly 40-60 minutes.

    extracellular_mucin_extent   none | focal_lt10 | moderate_10_50 | extensive_gt50
    gland_formation              gt95 | pct50_95 | lt50

Those two fields are the p17 and p28 correlates respectively, and
`gland_formation` doubles as the WHO grade proxy that the study manifests do
not contain - so it also upgrades Aim 1's clinical comparator.

STOP-ANYTIME DESIGN. Cases are ordered at random with respect to every
stratum, so ANY PREFIX of the list is an unbiased subsample. The reader may
stop at any point and the completed prefix remains analysable; power simply
scales with how far they got. 120 is the target, ~90 is still adequate for
the primary endpoint.

PRIMARY ENDPOINT (declared before reading). Monotone trend of
pathologist-scored mucin extent across frozen within-cohort p17 tertiles
(Jonckheere-Terpstra). Sampling is balanced 40/40/40 across p17 tertiles and
KRAS-balanced within tertile, which maximises trend power per case but makes
the sample non-representative: no prevalence statistic may be computed from
it, and the effect size is not transportable to an unselected series.

SECONDARY. p28 abundance versus gland formation; and whether either scored
feature attenuates the association between the held-out WSI logit and KRAS.

NOT ESTIMATED. Between-reader and intra-reader reliability. A single reader
without replicates cannot support either, and twelve replicate pairs would
not yield a stable estimate - so reliability is left as an explicit stated
limitation rather than an underpowered number.

Read-only over frozen inputs; writes only under reviews/v3.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools.build_reviews_v2 import (  # noqa: E402
    CANONICAL_SLIDE_DIRS,
    EXCLUDED_SLIDE_DIRS,
    SLIDE_ROOT,
    load_case_frame,
)

OUTPUT = REPO / "reviews" / "v3"
SELECTION_SEED = 20260821
PER_TERTILE = 40  # 3 tertiles -> 120 cases
FORM_COLUMNS = [
    "case_id",
    "assessable",
    "extracellular_mucin_extent",
    "gland_formation",
    "note_if_unusual",
]
VOCAB = {
    "assessable": "yes | no  (no = slide unreadable or no tumour present)",
    "extracellular_mucin_extent": (
        "none | focal_lt10 | moderate_10_50 | extensive_gt50  "
        "(share of tumour area occupied by EXTRACELLULAR mucin)"
    ),
    "gland_formation": (
        "gt95 | pct50_95 | lt50  (share of tumour forming glands; the WHO grade proxy)"
    ),
    "note_if_unusual": "optional free text; leave blank unless something needs flagging",
}


def select_cases(frame: pd.DataFrame, per_tertile: int) -> pd.DataFrame:
    """Balanced 40/40/40 over within-cohort p17 tertiles, KRAS-balanced inside
    each tertile and spread over cohorts, then randomly ordered."""
    work = frame.copy()

    def tertile(series: pd.Series) -> pd.Series:
        ranks = series.rank(method="first")
        edges = np.floor((ranks - 1) * 3 / len(series)).astype(int).clip(0, 2)
        return edges.map({0: "T1", 1: "T2", 2: "T3"})

    work["p17_tertile"] = (
        work.sort_values("patient_id").groupby("cohort")["p17_abundance"].transform(tertile)
    )
    rng = np.random.default_rng(SELECTION_SEED)
    picks: list[int] = []
    audit: list[dict[str, Any]] = []
    for tier in ("T1", "T2", "T3"):
        per_class = per_tertile // 2
        for kras in ("mutant", "wild_type"):
            block = work[(work["p17_tertile"] == tier) & (work["kras"] == kras)]
            # spread across cohorts: round-robin cohorts, shuffled within each
            by_cohort = {
                cohort: rng.permutation(sub.sort_values("patient_id").index.to_numpy())
                for cohort, sub in block.groupby("cohort", sort=True)
            }
            chosen: list[int] = []
            position = 0
            while len(chosen) < per_class and any(
                position < len(v) for v in by_cohort.values()
            ):
                for cohort in sorted(by_cohort):
                    if len(chosen) >= per_class:
                        break
                    if position < len(by_cohort[cohort]):
                        chosen.append(int(by_cohort[cohort][position]))
                position += 1
            picks.extend(chosen)
            audit.append(
                {
                    "p17_tertile": tier,
                    "kras": kras,
                    "available": int(len(block)),
                    "selected": int(len(chosen)),
                }
            )
    sample = work.loc[picks].copy()
    order = rng.permutation(len(sample))  # prefix-unbiased ordering
    sample = sample.iloc[order].reset_index(drop=True)
    sample.insert(0, "case_id", [f"V{i + 1:03d}" for i in range(len(sample))])
    sample.attrs["audit"] = audit
    return sample


def resolve_slides(sample: pd.DataFrame) -> pd.DataFrame:
    index_by_stem: dict[str, Path] = {}
    for directory in CANONICAL_SLIDE_DIRS:
        for candidate in sorted((SLIDE_ROOT / directory).glob("*")):
            if candidate.is_file() and candidate.suffix.lower() in {".svs", ".tif", ".tiff", ".scn"}:
                index_by_stem.setdefault(candidate.name.rsplit(".", 1)[0], candidate)
    rows = []
    for _, row in sample.iterrows():
        for index, slide in enumerate(row["slide_ids"], start=1):
            source = index_by_stem.get(slide)
            rows.append(
                {
                    "case_id": row["case_id"],
                    "slide_index": index,
                    "blinded_filename": f"{row['case_id']}_s{index}{source.suffix if source else ''}",
                    "source_path": str(source) if source else "",
                    "source_dir": source.parent.name if source else "",
                    "source_found": source is not None,
                }
            )
    plan = pd.DataFrame(rows)
    if plan["source_dir"].isin(EXCLUDED_SLIDE_DIRS).any():
        raise AssertionError("a slide resolved to a non-canonical directory")
    if not plan["source_found"].all():
        raise AssertionError(f"{int((~plan['source_found']).sum())} slides did not resolve")
    return plan


def write_packet(root: Path, sample: pd.DataFrame, plan: pd.DataFrame) -> None:
    reader = root / "FOR_PATHOLOGIST"
    reader.mkdir(parents=True)

    with open(reader / "scoring_form.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORM_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for case_id in sample["case_id"]:
            writer.writerow(
                {c: (case_id if c == "case_id" else "") for c in FORM_COLUMNS}
            )

    (reader / "INSTRUCTIONS.md").write_text(
        f"""# Case scoring — {len(sample)} cases, about 40-60 minutes

Thank you. This is a single pass with **two questions per case**. Please do
not spend long on any one case: a low-power impression is exactly what is
wanted.

## What to do

Open each slide from the slide folder by its case name (`{sample['case_id'].iloc[0]}_s1`, and
`_s2` where a case has two slides), then fill one row in `scoring_form.csv`.

| Column | Answer with exactly one of |
| --- | --- |
| `assessable` | {VOCAB['assessable']} |
| `extracellular_mucin_extent` | {VOCAB['extracellular_mucin_extent']} |
| `gland_formation` | {VOCAB['gland_formation']} |
| `note_if_unusual` | {VOCAB['note_if_unusual']} |

If `assessable` is `no`, leave the two scoring columns blank and move on.

**Extracellular mucin** means mucin lying outside cells - pools, lakes, or
stromal mucin - not intracytoplasmic mucin or goblet cells. Judge it as a
share of tumour area at low power.

**Gland formation** is the usual grading estimate: the proportion of the
tumour forming recognisable glands.

## Please note

* You may **stop at any point.** The case order is randomised, so whatever
  you complete is a valid sample. Finishing more cases adds precision, but
  there is no threshold you must reach.
* Cases are ordered randomly and case names carry no information.
* Please do not look up any case, and do not ask what these features are
  expected to predict - the study depends on that being unknown to you.

When finished, return `scoring_form.csv` and note roughly how long it took.
"""
    )

    keys = root / "KEYS_DO_NOT_DISTRIBUTE"
    keys.mkdir(parents=True)
    key = sample.copy()
    key["slide_ids"] = key["slide_ids"].apply(lambda v: ";".join(v))
    key.to_csv(keys / "case_key.csv", index=False)
    pd.DataFrame(sample.attrs["audit"]).to_csv(keys / "selection_audit.csv", index=False)
    plan.to_csv(keys / "slide_link_plan.csv", index=False)

    script = keys / "make_blinded_slide_folder.sh"
    lines = [
        "#!/usr/bin/env bash",
        "# Build the blinded slide folder, then hand the pathologist ONLY that folder",
        "# plus FOR_PATHOLOGIST/. Raw filenames encode cohort, so never share them.",
        "set -euo pipefail",
        'OUT="${1:?usage: make_blinded_slide_folder.sh /path/to/blinded_slides}"',
        'mkdir -p "$OUT"',
    ]
    for _, row in plan.iterrows():
        lines.append(f'ln -sf "{row["source_path"]}" "$OUT/{row["blinded_filename"]}"')
    lines.append('echo "linked $(ls "$OUT" | wc -l) slides into $OUT"')
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)

    (keys / "README.md").write_text(
        "# DO NOT DISTRIBUTE\n\nMaps blinded case ids to patients, cohorts, molecular "
        "status, p17/p28 abundance and the held-out WSI logit. Open only after the "
        "completed form is returned. `slide_link_plan.csv` also leaks cohort through "
        "source paths.\n"
    )

    audit = pd.DataFrame(sample.attrs["audit"])
    (root / "README_COORDINATOR.md").write_text(
        f"""# reviews/v3 — one-hour case-level validation

One pathologist, one sitting, {len(sample)} cases, two ordinal fields.
Estimated 40-60 minutes. Replaces the larger reviews/v2 design, which asked
for 10-14 hours and could not be staffed.

## Hand over exactly two things

1. `FOR_PATHOLOGIST/` (instructions + `scoring_form.csv`)
2. the blinded slide folder produced by
   `KEYS_DO_NOT_DISTRIBUTE/make_blinded_slide_folder.sh <dest>`

Never hand over `KEYS_DO_NOT_DISTRIBUTE/` or raw slide paths: TCGA barcodes
and RIH `SL-` numbers identify the cohort. Slides are resolved only from the
colour-corrected canonical directories, so the R/B-swapped SurGen originals
can never reach a mucin assessment.

## Sample

{len(sample)} of 1,486 development primaries: {PER_TERTILE} per within-cohort
p17 tertile, KRAS-balanced inside each tertile, spread round-robin across
cohorts, seed {SELECTION_SEED}. Order is randomised with respect to every
stratum, so **any prefix is an unbiased subsample** and an early stop costs
precision but not validity.

* by cohort: {sample.groupby('cohort').size().to_dict()}
* by KRAS: {sample.groupby('kras').size().to_dict()}
* by p17 tertile: {sample.groupby('p17_tertile').size().to_dict()}
* slides to link: {len(plan)}

Balanced tertile sampling maximises trend power per case and makes the sample
deliberately non-representative: compute no prevalence statistic from it, and
do not transport the effect size to an unselected series.

## Endpoints, declared before reading

1. **Primary** — monotone trend of scored `extracellular_mucin_extent` across
   frozen p17 tertiles (Jonckheere-Terpstra), with the tertile-1 versus
   tertile-3 contrast reported alongside.
2. **Secondary** — `gland_formation` against frozen p28 abundance; and
   whether either scored feature attenuates the association between the
   held-out WSI logit and KRAS status, adjusted for BRAF, MSI, NRAS, site,
   stage and cohort.
3. **Comparator upgrade** — `gland_formation` is the grade proxy absent from
   the manifests, so it can enter Aim 1's clinical baseline.

Reliability is **not** estimated: one reader without replicates supports
neither between- nor intra-reader agreement, and that remains a stated
limitation rather than an underpowered number.

## Selection audit

{audit.to_string(index=False)}
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--per-tertile", type=int, default=PER_TERTILE)
    args = parser.parse_args()
    root: Path = args.output
    if root.exists():
        raise SystemExit(f"append-only: {root} already exists")
    frame = load_case_frame()
    sample = select_cases(frame, args.per_tertile)
    plan = resolve_slides(sample)
    root.mkdir(parents=True)
    write_packet(root, sample, plan)
    summary = {
        "cases": int(len(sample)),
        "slides": int(len(plan)),
        "by_cohort": sample.groupby("cohort").size().to_dict(),
        "by_kras": sample.groupby("kras").size().to_dict(),
        "by_p17_tertile": sample.groupby("p17_tertile").size().to_dict(),
        "estimated_minutes": [int(len(sample) * 20 / 60), int(len(sample) * 30 / 60)],
    }
    (root / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
