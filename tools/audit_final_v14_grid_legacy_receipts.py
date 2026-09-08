"""Inventory historical source-only receipt schemas without choosing grid patients."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import final_v14_module2 as common  # noqa: E402

GRID = common.RUN / "grid_offset"
CHECKS = GRID / "operational_preflight_20260905/candidate_checks.csv"
OUTPUT = GRID / "historical_receipt_schema_20260905/receipt_inventory.json"
OLD_ROOT = Path("/mnt/d/YC.Liu/features/colon_stream/_stream_receipts")
HEST = Path("/home/yc_liu/.cache/trident/deeplabv3_seg_v4.ckpt")
HEST_SHA = "4ddf8be82384544ae9ef9af7ad0cc1e8190c781b945eea9826c778f93d08fd5a"
UNI_SHA = "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"


def audit() -> dict:
    if OUTPUT.exists():
        result = common.read_sealed(OUTPUT)
        for item in [result["source_candidate_checks"], result["local_hest_checkpoint"],
                     result["auditor"], *result["source_segmentation_receipts"]]:
            common.verify_identity(item)
        return result
    weight = common.identity(HEST)
    if weight["sha256"] != HEST_SHA:
        raise ValueError("Local frozen HEST checkpoint differs from the known checkpoint")
    frame = pd.read_csv(CHECKS)
    candidates = frame.loc[frame.candidate_header_geometry_pass].copy()
    if len(candidates) != 64 or candidates.slide_id.duplicated().any():
        raise ValueError("Historical source-candidate inventory differs from the completed preflight")
    lineages: Counter = Counter()
    receipts = []
    for row in candidates.itertuples(index=False):
        path = OLD_ROOT / str(row.slide_id) / "seg.json"
        data = json.loads(path.read_text())
        checkpoints = data.get("checkpoint_hashes", {})
        if (data.get("stage") != "seg" or data.get("source", {}).get("output_id") != row.slide_id
                or checkpoints.get("hest") != HEST_SHA or checkpoints.get("uni_v1") != UNI_SHA):
            raise ValueError(f"Historical source receipt/checkpoint mismatch: {path}")
        segmenter = data.get("segmentation_policy", {}).get("segmenter")
        if segmenter not in (None, "hest"):
            raise ValueError(f"Historical receipt explicitly names another segmenter: {path}")
        lineages[(data["implementation_hash"], segmenter)] += 1
        receipts.append(common.identity(path))
    result = {
        "status": "HISTORICAL_SOURCE_RECEIPT_SCHEMA_INVENTORY_COMPLETE",
        "created_utc": common.now(), "auditor": common.identity(Path(__file__)),
        "source_candidate_checks": common.identity(CHECKS), "local_hest_checkpoint": weight,
        "source_segmentation_receipts": receipts,
        "lineages": [{"recorded_implementation_hash": key[0], "segmenter_field": key[1],
                      "source_receipts": count, "hest_sha256": HEST_SHA, "uni_sha256": UNI_SHA}
                     for key, count in sorted(lineages.items())],
        "scope": "Schema compatibility evidence only. Recorded implementation hashes were inventoried; historical source code was not recovered or independently replayed.",
        "canonical_copy_substitution": False, "grid_patients_sampled": 0,
        "offset_coordinates_generated": False, "features_or_outcomes_read": False,
        "claim_limit": "These older source candidates cannot establish final grid eligibility. Every actual selected input still requires the restored canonical archive and its full read-only preflight.",
    }
    common.write_json(OUTPUT, result)
    common.seal(OUTPUT)
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps({"status": result["status"], "source_receipts": len(result["source_segmentation_receipts"]),
                      "lineages": result["lineages"], "output": str(OUTPUT)}, indent=2))
