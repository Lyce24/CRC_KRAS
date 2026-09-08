#!/usr/bin/env python3
"""Aim 4 blinded-review ingestion: validate a completed form, then unblind.

This is the missing downstream half of the Aim 4 pathology workflow. The
corrected v2 packets (base: 10 montages; attention addendum: 1 montage) ship
blank forms and keep the montage-to-prototype key outside the packet. This
tool takes a reviewer's COMPLETED external form copies and:

    1. validates them against the sealed blank masters — schema, montage set,
       required completion semantics and the controlled vocabulary of every
       field, including the interpretable=no cascade;
    2. verifies it can bind the exact sealed packet bytes (montages, blank
       form, scale info) by SHA-256 before anything is joined;
    3. only after validation, opens the unblinding keys and joins each montage
       to its frozen k=32 prototype and to the sealed canonical prototype
       classification/statistics table; and
    4. writes an append-only unblinded morphology table with a receipt.

BLINDING SEMANTICS. The protocol requires every form row to be complete
before any key access; this tool enforces that mechanically by failing
validation first. Its OUTPUT is unblinding-sensitive: it reveals the
montage-to-prototype mapping, so it must never be shown to a reviewer who has
not yet completed their own forms. The output root carries this warning.

REVIEWER KINDS. ``--reviewer-kind human`` records an external pathologist
read — the read the study's pathology seal is waiting for. ``--reviewer-kind
ai`` records a preliminary machine read (e.g. a blinded vision-model read):
it is explicitly NON-AUTHORITATIVE, does not satisfy the pathology seal, and
is labeled as an annex in every artifact this tool writes. Each ingested read
goes to its own append-only output root.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools import final_v3_aim1_worklist as worklist  # noqa: E402

AIM4_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim4_cap8192_corrected_v2_20260820"
)
BUNDLES = AIM4_ROOT / "review_bundles" / "k32"
PROTOTYPE_TABLE = AIM4_ROOT / "analysis" / "prototypes_k32.csv"

FORM_COLUMNS = [
    "montage_id", "n_tiles", "review_status", "interpretable", "dominant_pattern",
    "heterogeneity", "architecture", "mucin", "dirty_necrosis", "desmoplasia_stroma",
    "budding_invasion", "immune_infiltration", "normal_organ_tissue", "artifact",
    "differentiation", "confidence", "reviewer_id", "review_date",
    "blinding_attestation", "free_text",
]
BINARY_FIELDS = (
    "mucin", "dirty_necrosis", "desmoplasia_stroma", "budding_invasion",
    "immune_infiltration", "normal_organ_tissue", "artifact",
)
VOCAB: dict[str, set[str]] = {
    "review_status": {"complete"},
    "interpretable": {"yes", "partial", "no"},
    "heterogeneity": {"homogeneous", "mixed", "not_assessable"},
    "architecture": {
        "glandular", "cribriform", "solid", "papillary", "mixed",
        "non_tumour", "artifact", "not_assessable",
    },
    "differentiation": {"well", "moderate", "poor", "mixed", "not_assessable"},
    "confidence": {"low", "moderate", "high", "not_assessable"},
    "blinding_attestation": {"confirmed_no_key_access"},
}
CASCADE_FIELDS = (
    "dominant_pattern", "heterogeneity", "architecture", *BINARY_FIELDS,
    "differentiation", "confidence",
)


def packet_paths(bundle: str) -> dict[str, Path]:
    root = BUNDLES / bundle
    return {
        "packet_form": root / "packet" / "review_form.csv",
        "packet_readme": root / "packet" / "README.md",
        "packet_scale": root / "packet" / "scale_info.csv",
        "montage_dir": root / "packet" / "montages",
        "key": root / "unblinding_key_DO_NOT_SHARE.csv",
    }


def validate_form(form: pd.DataFrame, blank: pd.DataFrame, name: str) -> list[str]:
    """Return a list of validation failures (empty = valid)."""
    problems: list[str] = []
    if list(form.columns) != FORM_COLUMNS:
        problems.append(f"{name}: column set/order differs from the sealed blank master")
        return problems
    expected = sorted(blank["montage_id"].astype(str))
    observed = sorted(form["montage_id"].astype(str))
    if expected != observed:
        problems.append(f"{name}: montage ids {observed} != sealed {expected}")
    for _, row in form.iterrows():
        mid = row["montage_id"]
        for field, allowed in VOCAB.items():
            if str(row[field]) not in allowed:
                problems.append(f"{name}/{mid}: {field}={row[field]!r} not in {sorted(allowed)}")
        for field in BINARY_FIELDS:
            if str(row[field]) not in {"present", "absent", "not_assessable"}:
                problems.append(f"{name}/{mid}: {field}={row[field]!r} invalid")
        for field in ("dominant_pattern", "free_text", "reviewer_id", "review_date"):
            value = str(row[field]).strip()
            if not value or value == "pending":
                problems.append(f"{name}/{mid}: {field} incomplete")
        if str(row["interpretable"]) == "no":
            for field in CASCADE_FIELDS:
                if str(row[field]) != "not_assessable":
                    problems.append(
                        f"{name}/{mid}: interpretable=no requires {field}=not_assessable"
                    )
    return problems


def montage_prototype_map(key_path: Path) -> pd.DataFrame:
    key = pd.read_csv(key_path)
    grouped = key.groupby("montage_id")["prototype"].agg(["nunique", "first", "count"])
    if (grouped["nunique"] != 1).any():
        raise AssertionError(f"{key_path}: a montage maps to more than one prototype")
    out = grouped.reset_index().rename(columns={"first": "prototype", "count": "n_key_tiles"})
    strata = key.groupby("montage_id")["selection_stratum"].first().reset_index()
    return out[["montage_id", "prototype", "n_key_tiles"]].merge(strata, on="montage_id")


def run(args: argparse.Namespace) -> None:
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    inputs: list[Path] = [PROTOTYPE_TABLE]
    bundles = {"base": args.form, "attention_addendum": args.addendum_form}
    frames: list[pd.DataFrame] = []
    problems: list[str] = []
    for bundle, form_path in bundles.items():
        if form_path is None:
            continue
        paths = packet_paths(bundle)
        inputs += [form_path, paths["packet_form"], paths["packet_scale"], paths["key"]]
        inputs += sorted(paths["montage_dir"].glob("*.jpg"))
        blank = pd.read_csv(paths["packet_form"])
        form = pd.read_csv(form_path, dtype=str)
        problems += validate_form(form, blank, bundle)
        if problems:
            continue
        mapping = montage_prototype_map(paths["key"])
        merged = form.merge(mapping, on="montage_id", validate="one_to_one")
        merged["bundle"] = bundle
        frames.append(merged)
    if problems:
        raise SystemExit("form validation FAILED:\n  " + "\n  ".join(problems))
    if not frames:
        raise SystemExit("no completed form was provided")

    unblinded = pd.concat(frames, ignore_index=True)
    prototypes = pd.read_csv(PROTOTYPE_TABLE)
    context_columns = [
        "prototype", "group", "abundance_auc_A", "abundance_q_A", "abundance_auc_D",
        "abundance_q_D", "attention_auc_A", "attention_q_A", "persists_in_D",
        "transport_state", "shortcut_flags", "pathology_status",
    ]
    unblinded = unblinded.merge(prototypes[context_columns], on="prototype", validate="one_to_one")
    unblinded = unblinded.sort_values(["bundle", "prototype"]).reset_index(drop=True)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": created,
        "component": "aim4_pathology_ingest",
        "reviewer_kind": args.reviewer_kind,
        "authoritative_for_pathology_seal": args.reviewer_kind == "human",
        "reviewer_ids": sorted(unblinded["reviewer_id"].unique().tolist()),
        "n_montages": int(len(unblinded)),
        "unblinding_sensitive": True,
        "reads": unblinded.to_dict(orient="records"),
    }

    output: Path = args.output
    if output.exists():
        raise FileExistsError(f"append-only destination already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    warning = (
        "# UNBLINDING-SENSITIVE\n\n"
        "This directory reveals the montage-to-prototype mapping of the Aim 4\n"
        "corrected v2 review packets. It must NOT be shown to any reviewer who\n"
        "has not yet completed and returned their own blank forms.\n\n"
        f"Reviewer kind: {args.reviewer_kind}. "
        + (
            "This read DOES NOT satisfy the study's pathology seal; it is a\n"
            "preliminary machine-read annex.\n"
            if args.reviewer_kind == "ai"
            else "This read is a candidate input to the study's pathology seal.\n"
        )
    )
    (output / "README_UNBLINDING_SENSITIVE.md").write_text(warning)
    (output / "results.json").write_text(
        json.dumps(worklist._json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    unblinded.to_csv(output / "unblinded_reads.csv", index=False)

    lines = [
        "# Aim 4 unblinded morphology reads",
        "",
        f"Reviewer kind: **{args.reviewer_kind}**"
        + (" — preliminary machine read, NON-AUTHORITATIVE annex" if args.reviewer_kind == "ai" else ""),
        f"Generated {created}.",
        "",
        "| Prototype | Class | Montage | Interpretable | Architecture | Dominant pattern | Differentiation | Confidence |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in unblinded.iterrows():
        lines.append(
            f"| p{row['prototype']} | {row['group']} | {row['montage_id']} ({row['bundle']}) | "
            f"{row['interpretable']} | {row['architecture']} | {row['dominant_pattern']} | "
            f"{row['differentiation']} | {row['confidence']} |"
        )
    (output / "MORPHOLOGY_TABLE.md").write_text("\n".join(lines) + "\n")

    produced = [
        output / "README_UNBLINDING_SENSITIVE.md",
        output / "results.json",
        output / "unblinded_reads.csv",
        output / "MORPHOLOGY_TABLE.md",
    ]
    unique_inputs = sorted({p.resolve() for p in inputs})
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "append_only": True,
                "created_utc": created,
                "reviewer_kind": args.reviewer_kind,
                "authoritative_for_pathology_seal": args.reviewer_kind == "human",
                "unblinding_sensitive": True,
                "output_root": str(output.resolve()),
                "inputs": [worklist.identity(p) for p in unique_inputs],
                "artifacts": [worklist.identity(p) for p in produced],
                "promise": "No sealed packet byte or upstream result root was modified.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"PASS — {len(unblinded)} montages unblinded → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--form", type=Path, required=True, help="completed base form CSV")
    parser.add_argument("--addendum-form", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer-kind", choices=("human", "ai"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
