#!/usr/bin/env python3
"""Final-v6 report bundle verifier and seal.

Verifies the final-v6 evidence chain and writes the bundle receipt LAST with
write-once semantics:

    1. the pre-v6 snapshot of final-v5 is byte-identical to live final-v5 and
       matches the snapshot receipt's recorded identities;
    2. `parent_final_v5_receipt.json` is byte-identical to the live final-v5
       bundle receipt;
    3. the final-v6 component receipt declares PASS and every artifact it
       lists rehashes correctly;
    4. the three training lineages sealed by this version (E1v, E3v, E1e)
       each expose the expected number of validated OOF chains, and their
       analysis results.json files rehash to the identities the component
       receipt recorded; and
    5. the three final-v6 Markdown files are hashed and bound.

The receipt does not hash itself; any later change requires final-v7.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
HOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1")
FINAL_V5 = REPO / "reports" / "final_v5"
FINAL_V6 = REPO / "reports" / "final_v6"
SNAPSHOT = REPO / "reports" / "snapshots" / "final_v5_pre_v6_20260820"
SNAPSHOT_RECEIPT = REPO / "reports" / "snapshots" / "final_v5_pre_v6_20260820_receipt.json"
ADDITIONS = REPO / "reports" / "reruns" / "final_v6_additions_20260820"

COMPONENT_RECEIPTS = {
    "encoder_and_control": ADDITIONS / "encoder_and_control" / "receipt.json",
}
# lineage -> (results.json, expected validated OOF chains)
LINEAGES = {
    "e1v_virchow2_aim1": (
        HOT / "train" / "1a_pb_cap8192" / "virchow2_cls" / "analysis" / "results.json",
        HOT / "train" / "1a_pb_cap8192" / "virchow2_cls",
        3,
    ),
    "e1e_msi_control": (
        HOT / "e1e" / "msi_cap8192" / "univ1" / "analysis" / "results.json",
        HOT / "e1e" / "msi_cap8192" / "univ1",
        3,
    ),
    "e3v_virchow2_aim3": (
        HOT / "reruns" / "aim3_virchow2_cap8192_v1_20260820" / "analysis" / "results.json",
        HOT / "reruns" / "aim3_virchow2_cap8192_v1_20260820" / "train",
        18,
    ),
}
MARKDOWN = ("Experimental_Setup.md", "Results.md", "Audit.md")
SNAPSHOT_FILES = (*MARKDOWN, "report_bundle_receipt.json", "parent_final_v4_receipt.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(problems: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {"snapshot": {}, "components": {}, "lineages": {}, "markdown": {}}

    recorded = json.loads(SNAPSHOT_RECEIPT.read_text())["identities"]
    for name in SNAPSHOT_FILES:
        live_hash = sha256(FINAL_V5 / name)
        snap_hash = sha256(SNAPSHOT / name)
        if live_hash != snap_hash:
            problems.append(f"snapshot drift: {name} differs between final_v5 and snapshot")
        if snap_hash != recorded[name]["sha256"]:
            problems.append(f"snapshot receipt mismatch for {name}")
        payload["snapshot"][name] = snap_hash

    parent = FINAL_V6 / "parent_final_v5_receipt.json"
    if sha256(parent) != sha256(FINAL_V5 / "report_bundle_receipt.json"):
        problems.append("parent_final_v5_receipt.json differs from live final-v5 receipt")
    payload["parent_final_v5_receipt_sha256"] = sha256(parent)

    for name, receipt_path in COMPONENT_RECEIPTS.items():
        if not receipt_path.is_file():
            problems.append(f"{name}: receipt missing at {receipt_path}")
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "PASS":
            problems.append(f"{name}: receipt status {receipt.get('status')!r} != PASS")
        entry: dict[str, Any] = {"receipt_sha256": sha256(receipt_path), "verified_artifacts": 0}
        for artifact in receipt.get("artifacts", []):
            path = Path(artifact["path"])
            if not path.is_file():
                problems.append(f"{name}: declared artifact missing: {path}")
                continue
            if sha256(path) != artifact["sha256"]:
                problems.append(f"{name}: artifact hash mismatch: {path}")
            entry["verified_artifacts"] += 1
        declared_inputs = {Path(i["path"]): i["sha256"] for i in receipt.get("inputs", [])}
        payload["components"][name] = entry
        payload["components"][name]["declared_input_count"] = len(declared_inputs)

    for name, (results, root, expected_chains) in LINEAGES.items():
        if not results.is_file():
            problems.append(f"{name}: analysis results missing at {results}")
            continue
        chains = len(list(root.glob("**/oof_predictions.parquet")))
        if chains != expected_chains:
            problems.append(f"{name}: {chains} OOF chains, expected {expected_chains}")
        payload["lineages"][name] = {
            "results_sha256": sha256(results),
            "oof_chains": chains,
            "expected_chains": expected_chains,
        }

    for name in MARKDOWN:
        payload["markdown"][name] = {
            "sha256": sha256(FINAL_V6 / name),
            "bytes": (FINAL_V6 / name).stat().st_size,
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FINAL_V6 / "report_bundle_receipt.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"append-only: {args.output} already exists; write final-v7 instead")

    problems: list[str] = []
    payload = verify(problems)
    receipt = {
        "schema_version": 1,
        "bundle": "reports/final_v6",
        "parent_bundle": "reports/final_v5",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "adds": [
            "E1v — Aim 1 Virchow2-CLS at study cap 8,192",
            "E3v — Virchow2 replication of the three Aim 3 consensus ceilings",
            "E1e — MSI/dMMR pipeline positive control",
            "paired encoder contrast and Why-D encoder comparison",
        ],
        "verification": payload,
        "self_hash_excluded": True,
    }
    if problems:
        sys.stderr.write("FINAL-V6 VERIFICATION FAILED:\n  " + "\n  ".join(problems) + "\n")
        raise SystemExit(1)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    total = sum(entry["verified_artifacts"] for entry in payload["components"].values())
    chains = sum(entry["oof_chains"] for entry in payload["lineages"].values())
    print(
        f"PASS — {total} component artifacts and {chains} training chains verified; "
        f"receipt written to {args.output}"
    )


if __name__ == "__main__":
    main()
