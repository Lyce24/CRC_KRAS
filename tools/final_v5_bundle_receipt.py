#!/usr/bin/env python3
"""Final-v5 report bundle verifier and seal.

Verifies the final-v5 evidence chain and writes the bundle receipt LAST with
write-once semantics:

    1. the pre-v5 snapshot of final-v4 is byte-identical to live final-v4 and
       matches the snapshot receipt's recorded identities;
    2. `parent_final_v4_receipt.json` is byte-identical to the live final-v4
       bundle receipt;
    3. every final-v5 component receipt (human-read v2, context weld, Aim 1
       sealed replay) declares PASS and every artifact it lists rehashes
       correctly; the Aim 1 root must additionally carry a PASS
       `replay_receipt.json`;
    4. the three final-v5 Markdown files are hashed and bound.

The in-flight E1v/E3v/E1e campaigns are deliberately OUTSIDE this seal: their
designs are pre-registered in the final-v5 report, and their results belong
to final-v6. The receipt does not hash itself; any later change requires a
new append-only version.
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
FINAL_V4 = REPO / "reports" / "final_v4"
FINAL_V5 = REPO / "reports" / "final_v5"
SNAPSHOT = REPO / "reports" / "snapshots" / "final_v4_pre_v5_20260820"
SNAPSHOT_RECEIPT = REPO / "reports" / "snapshots" / "final_v4_pre_v5_20260820_receipt.json"
ADDITIONS = REPO / "reports" / "reruns" / "final_v5_additions_20260820"

COMPONENT_RECEIPTS = {
    "aim4_human_read_legacy_v2": ADDITIONS / "aim4_human_read_legacy_v2" / "receipt.json",
    "aim4_context_weld": ADDITIONS / "aim4_context_weld" / "receipt.json",
    "aim1_sealed_replay": ADDITIONS / "aim1_sealed_replay" / "receipt.json",
}
AIM1_REPLAY_RECEIPT = ADDITIONS / "aim1_sealed_replay" / "replay_receipt.json"
MARKDOWN = ("Experimental_Setup.md", "Results.md", "Audit.md")
SNAPSHOT_FILES = (*MARKDOWN, "report_bundle_receipt.json", "parent_final_v3_receipt.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(problems: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {"snapshot": {}, "components": {}, "markdown": {}}

    recorded = json.loads(SNAPSHOT_RECEIPT.read_text())["identities"]
    for name in SNAPSHOT_FILES:
        live_hash = sha256(FINAL_V4 / name)
        snap_hash = sha256(SNAPSHOT / name)
        if live_hash != snap_hash:
            problems.append(f"snapshot drift: {name} differs between final_v4 and snapshot")
        if snap_hash != recorded[name]["sha256"]:
            problems.append(f"snapshot receipt mismatch for {name}")
        payload["snapshot"][name] = snap_hash

    parent = FINAL_V5 / "parent_final_v4_receipt.json"
    if sha256(parent) != sha256(FINAL_V4 / "report_bundle_receipt.json"):
        problems.append("parent_final_v4_receipt.json differs from live final-v4 receipt")
    payload["parent_final_v4_receipt_sha256"] = sha256(parent)

    for name, receipt_path in COMPONENT_RECEIPTS.items():
        if not receipt_path.is_file():
            problems.append(f"{name}: receipt missing at {receipt_path}")
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "PASS":
            problems.append(f"{name}: receipt status {receipt.get('status')!r} != PASS")
        entry = {"receipt_sha256": sha256(receipt_path), "verified_artifacts": 0}
        for artifact in receipt.get("artifacts", []):
            path = Path(artifact["path"])
            if not path.is_file():
                problems.append(f"{name}: declared artifact missing: {path}")
                continue
            if sha256(path) != artifact["sha256"]:
                problems.append(f"{name}: artifact hash mismatch: {path}")
            entry["verified_artifacts"] += 1
        payload["components"][name] = entry

    if not AIM1_REPLAY_RECEIPT.is_file():
        problems.append("aim1_sealed_replay: replay_receipt.json missing (run --verify first)")
    else:
        replay = json.loads(AIM1_REPLAY_RECEIPT.read_text())
        if replay.get("status") != "PASS":
            problems.append(f"aim1_sealed_replay: replay status {replay.get('status')!r}")
        results = ADDITIONS / "aim1_sealed_replay" / "results.json"
        if replay.get("replayed_against") != sha256(results):
            problems.append("aim1_sealed_replay: replay receipt binds a different results.json")
        payload["aim1_replay_receipt_sha256"] = sha256(AIM1_REPLAY_RECEIPT)

    for name in MARKDOWN:
        payload["markdown"][name] = {
            "sha256": sha256(FINAL_V5 / name),
            "bytes": (FINAL_V5 / name).stat().st_size,
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FINAL_V5 / "report_bundle_receipt.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"append-only: {args.output} already exists; write final-v6 instead")

    problems: list[str] = []
    payload = verify(problems)
    receipt = {
        "schema_version": 1,
        "bundle": "reports/final_v5",
        "parent_bundle": "reports/final_v4",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "pathology_status": "HUMAN_READ_ADOPTED_VIA_GENERATION_BRIDGE",
        "aim1_status": "SEALED_REPLAY_PASS",
        "in_flight_excluded": ["E1v (Virchow2 cap-8192 Aim 1)", "E3v (Virchow2 Aim 3 ceilings)", "E1e (MSI positive control)"],
        "verification": payload,
        "self_hash_excluded": True,
    }
    if problems:
        sys.stderr.write("FINAL-V5 VERIFICATION FAILED:\n  " + "\n  ".join(problems) + "\n")
        raise SystemExit(1)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    total = sum(entry["verified_artifacts"] for entry in payload["components"].values())
    print(f"PASS — {total} artifacts verified; receipt written to {args.output}")


if __name__ == "__main__":
    main()
