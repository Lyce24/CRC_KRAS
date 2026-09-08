#!/usr/bin/env python3
"""Build the final-v3 Aim 3 molecular-resolution/actionability ladder.

This is a provenance-checked evidence synthesis, not a new predictive fit.  It
joins the sealed fixed-control and repeated-control Aim 3 results to a frozen
clinical-use interpretation.  The output directory is write-once so that this
add-on cannot mutate the final_v2 bundle or an earlier v3 run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_FIXED = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim3_cap8192_corrected_v1_20260819/analysis/aim3_corrected.json"
)
DEFAULT_REPEATED = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim3_controls_repeated_v1_20260819/analysis/aim3_repeated_control_report.json"
)
DEFAULT_OUTPUT = Path("reports/reruns/final_v3_additions_20260820/aim3_actionability")

EXPECTED_INPUT_SHA256 = {
    "fixed": "2622b71dddde795112df65026c30ca226ca9d8afdbe1d052f6be4372a4bbb731",
    "repeated": "809cff9a752319e7924f321f3990dfcc2b2f4a76503099ab83bcdbece8232ee3",
}

RUNG_CONFIG = {
    "codon": {
        "label": "Codon 12 versus other KRAS",
        "control": "ctrl_codon",
        "decision": "No",
        "interpretation": (
            "Consensus empirical ceiling under the prespecified model/control design; "
            "direct molecular confirmation is required."
        ),
    },
    "g12d_broad": {
        "label": "G12D versus all other KRAS",
        "control": "ctrl_g12d_broad",
        "decision": "No",
        "interpretation": (
            "Consensus empirical ceiling under the prespecified model/control design; "
            "direct molecular confirmation is required."
        ),
    },
    "allele1": {
        "label": "G12D versus other G12",
        "control": "ctrl_allele1",
        "decision": "No",
        "interpretation": (
            "Consensus empirical ceiling within codon 12; direct molecular confirmation "
            "is required."
        ),
    },
    "allele2": {
        "label": "G12V versus other G12",
        "control": "ctrl_allele2",
        "decision": "No",
        "interpretation": (
            "No ceiling consensus; the experiment is inconclusive and does not establish "
            "either detectability or absence."
        ),
    },
    "g12c": {
        "label": "G12C versus other G12",
        "control": "ctrl_g12c",
        "decision": "No",
        "interpretation": (
            "No ceiling consensus (one inconclusive and two underpowered repeated draws); "
            "an exact approved molecular test remains required."
        ),
    },
}

CLINICAL_REFERENCES = {
    "panitumumab_label": (
        "https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/125147s213lbl.pdf"
    ),
    "sotorasib_panitumumab_approval": (
        "https://www.fda.gov/drugs/resources-information-approved-drugs/"
        "fda-approves-sotorasib-panitumumab-kras-g12c-mutated-colorectal-cancer"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def validate_inputs(fixed_path: Path, repeated_path: Path) -> dict[str, dict[str, Any]]:
    paths = {"fixed": fixed_path.resolve(), "repeated": repeated_path.resolve()}
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        observed = sha256_file(path)
        expected = EXPECTED_INPUT_SHA256[name]
        if observed != expected:
            raise ValueError(
                f"{name} input hash mismatch: observed={observed}, expected={expected}"
            )
        result[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": observed,
        }
    return result


def build_rows(fixed: dict[str, Any], repeated: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = fixed["tasks"]
    pairs = fixed["pairs"]
    repeated_rungs = repeated["rungs"]
    rows: list[dict[str, Any]] = [
        {
            "molecular_level": "KRAS gene status, primary CRC",
            "n": tasks["gene"]["n"],
            "fine_auroc": tasks["gene"]["auroc"],
            "matched_control_auroc": None,
            "control_minus_fine": None,
            "fixed_verdict": "GENE_REFERENCE",
            "repeated_control_consensus": "NOT_APPLICABLE",
            "direct_clinical_decision_supported": "No",
            "appropriate_interpretation": (
                "Modest gene-level ranking supports retrospective molecular-testing "
                "prioritization or quality assurance only; it is not a replacement assay."
            ),
            "evidence_type": "empirical",
        },
        {
            "molecular_level": "Extended RAS status",
            "n": None,
            "fine_auroc": None,
            "matched_control_auroc": None,
            "control_minus_fine": None,
            "fixed_verdict": "NOT_EVALUATED",
            "repeated_control_consensus": "NOT_EVALUATED",
            "direct_clinical_decision_supported": "No",
            "appropriate_interpretation": (
                "KRAS-only inference cannot establish anti-EGFR eligibility because both "
                "KRAS and NRAS context are required."
            ),
            "evidence_type": "clinical_context",
        },
    ]

    for rung, config in RUNG_CONFIG.items():
        task = tasks[rung]
        control = tasks[config["control"]]
        pair = pairs[rung]
        consensus = repeated_rungs[rung]["consensus_verdict"]
        rows.append(
            {
                "molecular_level": config["label"],
                "n": task["n"],
                "fine_auroc": task["auroc"],
                "matched_control_auroc": control["auroc"],
                "control_minus_fine": pair["primary_delta_control_minus_fine"]["estimate"],
                "fixed_verdict": pair["nominal_95_verdict"],
                "repeated_control_consensus": consensus,
                "direct_clinical_decision_supported": config["decision"],
                "appropriate_interpretation": config["interpretation"],
                "evidence_type": "empirical",
            }
        )

    rows.append(
        {
            "molecular_level": "Metastatic exact allele",
            "n": None,
            "fine_auroc": None,
            "matched_control_auroc": None,
            "control_minus_fine": None,
            "fixed_verdict": "OUTSIDE_EVIDENCE_BASE",
            "repeated_control_consensus": "OUTSIDE_EVIDENCE_BASE",
            "direct_clinical_decision_supported": "No",
            "appropriate_interpretation": (
                "Exact-allele inference from metastatic tissue was not evaluated and is "
                "outside the evidence base."
            ),
            "evidence_type": "scope_boundary",
        }
    )
    return rows


def design_markdown() -> str:
    return """# Aim 3 final-v3 actionability synthesis

## Status

This component performs no model fitting and changes no Aim 3 estimate. It joins the
sealed fixed-control and repeated-control outputs to a frozen clinical-actionability
interpretation.

## Rules

- Every empirical value is imported from a SHA-256-validated Aim 3 source.
- A ceiling is an empirical result for the prespecified architecture, population and
  matched-control gate; it is not biological absence of morphology.
- No WSI row supports an autonomous treatment decision.
- Extended RAS and metastatic exact-allele rows are explicit scope boundaries, not
  newly evaluated experiments.
- G12C is reported as no consensus: one repeated draw was inconclusive and two were
  underpowered.
"""


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    atomic_write(path, payload)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty actionability table")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def run(fixed_path: Path, repeated_path: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_root}")
    inputs = validate_inputs(fixed_path, repeated_path)
    fixed = read_json(fixed_path)
    repeated = read_json(repeated_path)
    rows = build_rows(fixed, repeated)

    output_root.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "component": "final_v3_aim3_actionability",
        "status": "PASS",
        "analysis_type": "provenance_checked_evidence_synthesis_no_model_fit",
        "central_interpretation": (
            "Scientific detectability at KRAS gene level is not equivalent to direct "
            "clinical actionability at extended-RAS, codon, or exact-allele resolution."
        ),
        "rows": rows,
        "clinical_references": CLINICAL_REFERENCES,
        "inputs": inputs,
    }
    result_path = output_root / "actionability_ladder.json"
    csv_path = output_root / "actionability_ladder.csv"
    design_path = output_root / "design.md"
    write_json(result_path, result)
    write_csv(csv_path, rows)
    atomic_write(design_path, design_markdown().encode())

    outputs = {}
    for path in (result_path, csv_path, design_path):
        outputs[path.name] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    receipt = {
        "schema_version": 1,
        "component": "final_v3_aim3_actionability",
        "status": "PASS",
        "append_only": True,
        "inputs": inputs,
        "outputs": outputs,
    }
    write_json(output_root / "receipt.json", receipt)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--repeated", type=Path, default=DEFAULT_REPEATED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.fixed, args.repeated, args.output_root)
    print(json.dumps({"status": result["status"], "rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
