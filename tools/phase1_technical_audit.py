#!/usr/bin/env python3
"""Aim 1 phase 1 — label-independent technical variables for every cohort.

The frozen island audit (``univ1_umap_separation_audit``) covered SurGen and
RIH only. Aim 1 develops on TCGA + SR386 and validates on CPTAC, so the same
descriptors are recomputed here for every feature-backed eligible slide from
artifacts that already exist (feature H5 attributes + HEST thumbnails). No
KRAS label is read; the output is a technical *context* table used for fold
balancing (§4.4), subgroup evaluation (§5.3), the technical-only negative
control (§5.5), and 1C-B covariate balancing (§6) — never as a model input.

It also runs the §4.5 physical-resolution audit: the cross-cohort biological
claim is only admissible if every cohort was tiled at one common physical
resolution, which this verifies from the H5 attributes themselves.

Usage:
    python tools/phase1_technical_audit.py build            # dry run
    python tools/phase1_technical_audit.py build --apply
    python tools/phase1_technical_audit.py report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import paths, population, technical  # noqa: E402


def build(args: argparse.Namespace) -> None:
    rows = population.eligible()
    print(f"Eligible slides: {len(rows):,} across {rows['subcohort'].nunique()} subcohorts")
    print(rows.groupby("subcohort")["slide_id"].size().to_string())

    frame, summary = technical.build(rows["slide_id"], rows["subcohort"])

    audit = summary["extraction_policy_audit"]
    print("\nPhysical-resolution audit (§4.5)")
    print(f"  sampling_mode           {audit['sampling_mode']}")
    print(f"  target_mpp              {audit['target_mpp']}")
    print(f"  max |effective - 0.5|   {audit['max_abs_effective_mpp_deviation']:.6f} um/px")
    print(f"  native MPP values       {audit['native_mpp_values']}")

    fit = summary["derived_classes"]
    section = fit["section_size"]
    print("\nSection-size class — one frozen physical threshold")
    print(
        f"  {section['threshold_mm2']:.1f} mm2 tissue; balanced accuracy "
        f"{section['balanced_accuracy']:.3f} on {section['n_labelled']} slides with "
        f"recorded procedure (sens {section['sensitivity']:.3f}, "
        f"spec {section['specificity']:.3f})"
    )
    color = fit["color"]
    print("\nColour class — within-subcohort robust low-hue tail")
    print(
        f"  k={color['k']:.2f} flags {color['sr386_flagged']} SR386 slides "
        f"(frozen island: {color['sr386_island_target']})"
    )
    print("\nClass counts")
    for column, counts in fit["class_counts"].items():
        print(f"  {column}: {counts}")

    if not args.apply:
        print("\nDry run — pass --apply to write the frozen table.")
        return

    paths.TECHNICAL_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(paths.TECHNICAL_TABLE, index=False)
    paths.TECHNICAL_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote {paths.TECHNICAL_TABLE} ({len(frame):,} slides)")
    print(f"Wrote {paths.TECHNICAL_SUMMARY}")


def report(_: argparse.Namespace) -> None:
    frame = technical.load()
    summary = technical.load_summary()
    print(json.dumps(summary["extraction_policy_audit"], indent=2))
    rows = population.eligible()[["slide_id", "subcohort", "kras"]]
    merged = frame.merge(rows, on="slide_id", suffixes=("", "_label"))
    for column in ("mpp_bin", "section_size_class", "color_class"):
        print(f"\n{column} by subcohort")
        print(pd.crosstab(merged["subcohort"], merged[column]).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="compute and freeze technical variables")
    build_parser.add_argument("--apply", action="store_true", help="write the table")
    build_parser.set_defaults(func=build)

    report_parser = sub.add_parser("report", help="print the frozen audit")
    report_parser.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
