#!/usr/bin/env python3
"""Final-v4 report bundle verifier and seal.

Independently verifies the final-v4 evidence chain and writes the bundle
receipt LAST, with write-once semantics:

    1. the pre-v4 snapshot of final-v3 is byte-identical to live final-v3,
       and both match the snapshot receipt's recorded identities;
    2. `parent_final_v3_receipt.json` is byte-identical to the live final-v3
       bundle receipt;
    3. every final-v4 component receipt exists, declares PASS, and every
       artifact it lists rehashes to its recorded identity;
    4. the E2e lineage receipts (build and analysis_v2) exist and their
       recorded artifact identities rehash correctly; and
    5. the three final-v4 Markdown files are hashed and bound.

The receipt does not hash itself. Any later report change requires a new
append-only version (final-v5), never an overwrite.
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
FINAL_V3 = REPO / "reports" / "final_v3"
FINAL_V4 = REPO / "reports" / "final_v4"
SNAPSHOT = REPO / "reports" / "snapshots" / "final_v3_pre_v4_20260820"
SNAPSHOT_RECEIPT = REPO / "reports" / "snapshots" / "final_v3_pre_v4_20260820_receipt.json"
ADDITIONS = REPO / "reports" / "reruns" / "final_v4_additions_20260820"
E2E_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim2_e2e_met_local_v1_20260820")

COMPONENT_RECEIPTS = {
    "aim1_ras_composite": ADDITIONS / "aim1_ras_composite" / "receipt.json",
    "decision_curve": ADDITIONS / "decision_curve" / "receipt.json",
    "aim4_ai_blinded_read": ADDITIONS / "aim4_ai_blinded_read" / "unblinded_ai_read" / "receipt.json",
}
E2E_RECEIPTS = {
    "e2e_build": E2E_ROOT / "build_receipt.json",
    "e2e_analysis_v2": E2E_ROOT / "analysis_v2" / "receipt.json",
}
MARKDOWN = ("Experimental_Setup.md", "Results.md", "Audit.md")
SNAPSHOT_FILES = (*MARKDOWN, "report_bundle_receipt.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(problems: list[str], message: str) -> None:
    problems.append(message)


def verify(problems: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {"snapshot": {}, "components": {}, "e2e": {}, "markdown": {}}

    # 1. snapshot byte identity against live final-v3 and the snapshot receipt.
    recorded = json.loads(SNAPSHOT_RECEIPT.read_text())["identities"]
    for name in SNAPSHOT_FILES:
        live_hash = sha256(FINAL_V3 / name)
        snap_hash = sha256(SNAPSHOT / name)
        if live_hash != snap_hash:
            fail(problems, f"snapshot drift: {name} differs between final_v3 and snapshot")
        if snap_hash != recorded[name]["sha256"]:
            fail(problems, f"snapshot receipt mismatch for {name}")
        payload["snapshot"][name] = snap_hash

    # 2. parent receipt byte identity.
    parent = FINAL_V4 / "parent_final_v3_receipt.json"
    if sha256(parent) != sha256(FINAL_V3 / "report_bundle_receipt.json"):
        fail(problems, "parent_final_v3_receipt.json differs from live final-v3 receipt")
    payload["parent_final_v3_receipt_sha256"] = sha256(parent)

    # 3. component receipts and their declared artifacts.
    for name, receipt_path in {**COMPONENT_RECEIPTS, **E2E_RECEIPTS}.items():
        bucket = "e2e" if name.startswith("e2e") else "components"
        if not receipt_path.is_file():
            fail(problems, f"{name}: receipt missing at {receipt_path}")
            continue
        receipt = json.loads(receipt_path.read_text())
        entry: dict[str, Any] = {"receipt_sha256": sha256(receipt_path), "verified_artifacts": 0}
        declared = receipt.get("artifacts", [])
        for artifact in declared:
            path = Path(artifact["path"])
            if not path.is_file():
                fail(problems, f"{name}: declared artifact missing: {path}")
                continue
            if sha256(path) != artifact["sha256"]:
                fail(problems, f"{name}: artifact hash mismatch: {path}")
            entry["verified_artifacts"] += 1
        if name == "e2e_build":
            for key in ("manifest_sha256", "splits_sha256"):
                recorded_hash = receipt[key]
                path = (
                    Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2e_met.csv")
                    if key == "manifest_sha256"
                    else E2E_ROOT / "inputs" / "splits" / "aim1_e2e_met" / "aim1_balanced5" / "splits.parquet"
                )
                if sha256(path) != recorded_hash:
                    fail(problems, f"e2e_build: {key} mismatch for {path}")
                entry["verified_artifacts"] += 1
        if name == "e2e_analysis_v2":
            for seed, recorded_hash in receipt["oof_sha256"].items():
                path = E2E_ROOT / "train" / f"e2e_met_cap8192_seed{seed}" / "oof_predictions.parquet"
                if sha256(path) != recorded_hash:
                    fail(problems, f"e2e_analysis_v2: OOF hash mismatch seed {seed}")
                entry["verified_artifacts"] += 1
            results = E2E_ROOT / "analysis_v2" / "results.json"
            if sha256(results) != receipt["results_sha256"]:
                fail(problems, "e2e_analysis_v2: results.json hash mismatch")
            entry["verified_artifacts"] += 1
        payload[bucket][name] = entry

    # component receipts of read-only additions must also declare PASS status.
    for name, receipt_path in COMPONENT_RECEIPTS.items():
        if receipt_path.is_file():
            status = json.loads(receipt_path.read_text()).get("status")
            if status != "PASS":
                fail(problems, f"{name}: receipt status {status!r} != PASS")

    # 5. final-v4 markdown identities.
    for name in MARKDOWN:
        payload["markdown"][name] = {
            "sha256": sha256(FINAL_V4 / name),
            "bytes": (FINAL_V4 / name).stat().st_size,
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FINAL_V4 / "report_bundle_receipt.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"append-only: {args.output} already exists; write final-v5 instead")

    problems: list[str] = []
    payload = verify(problems)
    receipt = {
        "schema_version": 1,
        "bundle": "reports/final_v4",
        "parent_bundle": "reports/final_v3",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "pathology_status": "PENDING_HUMAN_READ_AI_ANNEX_RECORDED",
        "verification": payload,
        "self_hash_excluded": True,
    }
    if problems:
        sys.stderr.write("FINAL-V4 VERIFICATION FAILED:\n  " + "\n  ".join(problems) + "\n")
        raise SystemExit(1)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    total = sum(
        entry["verified_artifacts"]
        for bucket in ("components", "e2e")
        for entry in payload[bucket].values()
    )
    print(f"PASS — {total} artifacts verified; receipt written to {args.output}")


if __name__ == "__main__":
    main()
