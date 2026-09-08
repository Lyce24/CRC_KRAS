#!/usr/bin/env python3
"""Read-only report-order verification erratum for frozen FINAL-v11 analysis.

The governed analysis and all 10,000 bootstrap draws completed successfully.
Its completion-time verifier then compared report rows reconstructed from
JSON-loaded nested dictionaries against rows stored before ``sort_keys=True``
serialization.  JSON key sorting changed only the replay order of 12/78
performance positions and 3/48 contrast positions.  The unique row-ID sets and
every complete ID-to-row mapping are identical; both Why-D records replay
exactly.

This additive verifier authenticates the frozen inference graph, mean-validator
erratum, and exact five published analysis artifacts; reproduces the precise
order-only defect; and temporarily reorders only the two replayed lists in
memory before invoking the full frozen verifier.  It exposes no write, analysis,
bootstrap, scoring, training, or fitting command.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools import aim1_tcga_surgen_full_pipeline_analysis_mean_erratum as mean  # noqa: E402

legacy = mean.legacy
v3 = mean.v3
GovernanceError = mean.GovernanceError

SCHEMA_VERSION = 1
ERRATUM = "final_v11_report_order_verification_erratum_v5"
DEFAULT_CAMPAIGN_ROOT = mean.DEFAULT_CAMPAIGN_ROOT
ORDER_ERRATUM_TEST = REPO / (
    "tests/test_aim1_tcga_surgen_full_pipeline_analysis_report_order_erratum.py"
)

MEAN_ERRATUM_SHA256 = "675c190421a057949f9f22dc6a3710d38970625113af182a8120b44307542192"
MEAN_ERRATUM_SIZE = 24_274
MEAN_ERRATUM_TEST_SHA256 = "b7e49678bdc1c8cde5a880c7586ea1df476a863bb27fdf04b500fb82d33da92f"
MEAN_ERRATUM_TEST_SIZE = 15_194

ANALYSIS_PINS: dict[str, tuple[str, str, int]] = {
    "analysis_contract": (
        "contract.json",
        "3b1655fbf421d05b862615cff24dd2526c1cdf6f4112830bc158456d6c36b455",
        36_974,
    ),
    "patient_native_logits": (
        "patient_native_logits.parquet",
        "02f4a5bf0c2cd909c5ecb3f9edc7c05ae725649199a015cf0a1d0cb544df8d09",
        311_419,
    ),
    "bootstrap_distributions": (
        "bootstrap_distributions.npz",
        "7d47405fe4ac58e4134e58ff81bf88869d0ccf9e6258844ebc0760d8c08f7aaf",
        24_163_117,
    ),
    "results": (
        "results.json",
        "a46faeffcbb30e436776e4853602cccf801a31228ef490ab9caabac48f72f2b6",
        299_495,
    ),
    "analysis_completion": (
        "analysis_completion_receipt.json",
        "639696aaa4bdfafa887fc67cc541b976d6c3bf6113fc3909e2f133823df1b125",
        2_070,
    ),
}

STORED_CLAIM_IDS = (
    "aim1.canonical_e0.pooled_all_primary.univ1",
    "aim1.canonical_e0.pooled_all_primary.virchow2_cls",
    "aim1.canonical_e0.tcga.univ1",
    "aim1.canonical_e0.tcga.virchow2_cls",
    "aim1.canonical_e0.sr386.univ1",
    "aim1.canonical_e0.sr386.virchow2_cls",
    "aim1.canonical_e0.sr1482.univ1",
    "aim1.canonical_e0.sr1482.virchow2_cls",
    "aim1.canonical_e0.surgen.univ1",
    "aim1.canonical_e0.surgen.virchow2_cls",
    "aim1.canonical_e0.tcga_surgen.univ1",
    "aim1.canonical_e0.tcga_surgen.virchow2_cls",
    "aim1.source_restricted_e0.pooled_source.univ1",
    "aim1.source_restricted_e0.pooled_source.virchow2_cls",
    "aim1.source_restricted_e0.tcga.univ1",
    "aim1.source_restricted_e0.tcga.virchow2_cls",
    "aim1.source_restricted_e0.sr386.univ1",
    "aim1.source_restricted_e0.sr386.virchow2_cls",
    "aim1.source_restricted_e0.sr1482.univ1",
    "aim1.source_restricted_e0.sr1482.virchow2_cls",
    "aim1.source_restricted_e0.surgen.univ1",
    "aim1.source_restricted_e0.surgen.virchow2_cls",
    "aim1.source_restricted_e0.tcga_surgen.univ1",
    "aim1.source_restricted_e0.tcga_surgen.virchow2_cls",
    "aim2.cptac_primary.univ1",
    "aim2.cptac_primary.virchow2_cls",
    "aim2.rih_primary.univ1",
    "aim2.rih_primary.virchow2_cls",
    "aim2.rih_metastatic.univ1",
    "aim2.rih_metastatic.virchow2_cls",
    "aim2.sr1482_metastatic.univ1",
    "aim2.sr1482_metastatic.virchow2_cls",
    "aim2.orion_cpht.univ1",
    "aim2.orion_cpht.virchow2_cls",
    "aim1.e1a.A_all_primary.univ1",
    "aim1.e1a.A_all_primary.virchow2_cls",
    "aim1.e1a.A_complete.univ1",
    "aim1.e1a.A_complete.virchow2_cls",
    "aim1.e1a.B_mss.univ1",
    "aim1.e1a.B_mss.virchow2_cls",
    "aim1.e1a.C_braf_wt.univ1",
    "aim1.e1a.C_braf_wt.virchow2_cls",
    "aim1.e1a.D_mss_braf_wt.univ1",
    "aim1.e1a.D_mss_braf_wt.virchow2_cls",
    "aim1.e1a.E_colon.univ1",
    "aim1.e1a.E_colon.virchow2_cls",
    "aim1.e1a.F_rectum.univ1",
    "aim1.e1a.F_rectum.virchow2_cls",
    "aim1.e1a.G_stage_known_derived.univ1",
    "aim1.e1a.G_stage_known_derived.virchow2_cls",
    "aim1.e1a.G_stage_known_frozen.univ1",
    "aim1.e1a.G_stage_known_frozen.virchow2_cls",
    "aim1.e1a.H_stage_iv.univ1",
    "aim1.e1a.H_stage_iv.virchow2_cls",
    "aim1.e1a.I_right_proximal.univ1",
    "aim1.e1a.I_right_proximal.virchow2_cls",
    "aim1.e1a.J_left_distal.univ1",
    "aim1.e1a.J_left_distal.virchow2_cls",
    "aim1.e1a.K_transverse.univ1",
    "aim1.e1a.K_transverse.virchow2_cls",
    "aim1.e1d.A_all_primary.clinical",
    "aim1.e1d.A_all_primary.univ1.wsi",
    "aim1.e1d.A_all_primary.univ1.fusion",
    "aim1.e1d.A_all_primary.virchow2_cls.wsi",
    "aim1.e1d.A_all_primary.virchow2_cls.fusion",
    "aim1.e1d.G_stage_known_derived.clinical",
    "aim1.e1d.G_stage_known_derived.univ1.wsi",
    "aim1.e1d.G_stage_known_derived.univ1.fusion",
    "aim1.e1d.G_stage_known_derived.virchow2_cls.wsi",
    "aim1.e1d.G_stage_known_derived.virchow2_cls.fusion",
    "aim2.rih_disjoint_role.primary.univ1",
    "aim2.rih_disjoint_role.primary.virchow2_cls",
    "aim2.rih_disjoint_role.metastatic.univ1",
    "aim2.rih_disjoint_role.metastatic.virchow2_cls",
    "aim2.cpht_raw.exclude_neoadjuvant.univ1",
    "aim2.cpht_raw.exclude_neoadjuvant.virchow2_cls",
    "aim2.cpht_raw.exclude_ambiguous_crc15.univ1",
    "aim2.cpht_raw.exclude_ambiguous_crc15.virchow2_cls",
)

STORED_CONTRAST_IDS = (
    "aim1.source_restricted_e0_encoder_delta.pooled_source.virchow2_cls_minus_univ1",
    "aim1.source_restricted_e0_encoder_delta.tcga.virchow2_cls_minus_univ1",
    "aim1.source_restricted_e0_encoder_delta.sr386.virchow2_cls_minus_univ1",
    "aim1.source_restricted_e0_encoder_delta.sr1482.virchow2_cls_minus_univ1",
    "aim1.source_restricted_e0_encoder_delta.surgen.virchow2_cls_minus_univ1",
    "aim1.source_restricted_e0_encoder_delta.tcga_surgen.virchow2_cls_minus_univ1",
    "aim1.e1a_delta_A_minus_set.A_complete.univ1",
    "aim1.e1a_delta_A_minus_set.A_complete.virchow2_cls",
    "aim1.e1a_delta_A_minus_set.B_mss.univ1",
    "aim1.e1a_delta_A_minus_set.B_mss.virchow2_cls",
    "aim1.e1a_delta_A_minus_set.C_braf_wt.univ1",
    "aim1.e1a_delta_A_minus_set.C_braf_wt.virchow2_cls",
    "aim1.e1a_delta_A_minus_set.D_mss_braf_wt.univ1",
    "aim1.e1a_delta_A_minus_set.D_mss_braf_wt.virchow2_cls",
    "aim1.e1a_delta_A_minus_set.E_colon.univ1",
    "aim1.e1a_delta_A_minus_set.E_colon.virchow2_cls",
    "aim1.e1a_delta_A_minus_set.F_rectum.univ1",
    "aim1.e1a_delta_A_minus_set.F_rectum.virchow2_cls",
    "aim1.e1a_delta_A_minus_set.G_stage_known_derived.univ1",
    "aim1.e1a_delta_A_minus_set.G_stage_known_derived.virchow2_cls",
    "aim1.e1a_delta_A_minus_set.G_stage_known_frozen.univ1",
    "aim1.e1a_delta_A_minus_set.G_stage_known_frozen.virchow2_cls",
    "aim1.e1a_delta_A_minus_set.H_stage_iv.univ1",
    "aim1.e1a_delta_A_minus_set.H_stage_iv.virchow2_cls",
    "aim1.e1a_delta_A_minus_set.I_right_proximal.univ1",
    "aim1.e1a_delta_A_minus_set.I_right_proximal.virchow2_cls",
    "aim1.e1a_delta_A_minus_set.J_left_distal.univ1",
    "aim1.e1a_delta_A_minus_set.J_left_distal.virchow2_cls",
    "aim1.e1a_delta_A_minus_set.K_transverse.univ1",
    "aim1.e1a_delta_A_minus_set.K_transverse.virchow2_cls",
    "aim1.e1a_s_delta_D_minus_reference.A_all_primary.univ1",
    "aim1.e1a_s_delta_D_minus_reference.A_all_primary.virchow2_cls",
    "aim1.e1a_s_delta_D_minus_reference.A_complete.univ1",
    "aim1.e1a_s_delta_D_minus_reference.A_complete.virchow2_cls",
    "aim1.e1d_delta.A_all_primary.univ1.wsi_minus_clinical",
    "aim1.e1d_delta.A_all_primary.univ1.fusion_minus_clinical",
    "aim1.e1d_delta.A_all_primary.univ1.fusion_minus_wsi",
    "aim1.e1d_delta.A_all_primary.virchow2_cls.wsi_minus_clinical",
    "aim1.e1d_delta.A_all_primary.virchow2_cls.fusion_minus_clinical",
    "aim1.e1d_delta.A_all_primary.virchow2_cls.fusion_minus_wsi",
    "aim1.e1d_delta.G_stage_known_derived.univ1.wsi_minus_clinical",
    "aim1.e1d_delta.G_stage_known_derived.univ1.fusion_minus_clinical",
    "aim1.e1d_delta.G_stage_known_derived.univ1.fusion_minus_wsi",
    "aim1.e1d_delta.G_stage_known_derived.virchow2_cls.wsi_minus_clinical",
    "aim1.e1d_delta.G_stage_known_derived.virchow2_cls.fusion_minus_clinical",
    "aim1.e1d_delta.G_stage_known_derived.virchow2_cls.fusion_minus_wsi",
    "aim2.rih_disjoint_role.delta_metastatic_minus_primary.univ1",
    "aim2.rih_disjoint_role.delta_metastatic_minus_primary.virchow2_cls",
)

STORED_CLAIM_ID_SHA256 = "6c6587bfabf1a56c874e8db48c2ab78582c1e1b826baae3c809af78b4939e237"
REPLAY_CLAIM_ID_SHA256 = "02eba4d7650d38bf6a540979984bd38d304b92855ddbad6902938f1e5569cbe9"
STORED_CONTRAST_ID_SHA256 = "72abd95f8fb022faa3b95a974d91bfc452bb79a8ac861f45e507337ee0347a8f"
REPLAY_CONTRAST_ID_SHA256 = "19703b1cd59fa477e78ddb93e57cca9d2bcafd5879b9826f41cf360861a4ec3a"
STORED_CLAIM_ROWS_SHA256 = "f02e4ed21f5332d7ee1c49063365fae6e420be411169add2ad6315f1be9dc9ed"
REPLAY_CLAIM_ROWS_SHA256 = "a6aa054a6c1595dea368e570d8e18d677ba744736357c4d674d2d16e1b5d860e"
STORED_CONTRAST_ROWS_SHA256 = "2db064d65d0d336e4bdab9c8da67df53b530898fd78c3e588ca46ba388804d87"
REPLAY_CONTRAST_ROWS_SHA256 = "729ef14b437ef94c0b43aa8ebc0309ff43e1e025c3f9078c0f33de975ae8e30f"
WHY_D_ROWS_SHA256 = "c4c7f66f4da377d9e1af79dd0fdcc882935546977e2994b6303754b9c8cf7e5c"
POSITIONAL_PROFILE_SHA256 = "80a533bf327bea346c5a4f2ad94e9ae59c786d827b7564a9976dc1dbb66936ee"

EXPECTED_CLAIM_POSITIONAL_MISMATCHES = (
    (2, "aim1.canonical_e0.tcga.univ1", "aim1.canonical_e0.sr1482.univ1"),
    (
        3,
        "aim1.canonical_e0.tcga.virchow2_cls",
        "aim1.canonical_e0.sr1482.virchow2_cls",
    ),
    (6, "aim1.canonical_e0.sr1482.univ1", "aim1.canonical_e0.surgen.univ1"),
    (
        7,
        "aim1.canonical_e0.sr1482.virchow2_cls",
        "aim1.canonical_e0.surgen.virchow2_cls",
    ),
    (8, "aim1.canonical_e0.surgen.univ1", "aim1.canonical_e0.tcga.univ1"),
    (
        9,
        "aim1.canonical_e0.surgen.virchow2_cls",
        "aim1.canonical_e0.tcga.virchow2_cls",
    ),
    (
        14,
        "aim1.source_restricted_e0.tcga.univ1",
        "aim1.source_restricted_e0.sr1482.univ1",
    ),
    (
        15,
        "aim1.source_restricted_e0.tcga.virchow2_cls",
        "aim1.source_restricted_e0.sr1482.virchow2_cls",
    ),
    (
        18,
        "aim1.source_restricted_e0.sr1482.univ1",
        "aim1.source_restricted_e0.surgen.univ1",
    ),
    (
        19,
        "aim1.source_restricted_e0.sr1482.virchow2_cls",
        "aim1.source_restricted_e0.surgen.virchow2_cls",
    ),
    (
        20,
        "aim1.source_restricted_e0.surgen.univ1",
        "aim1.source_restricted_e0.tcga.univ1",
    ),
    (
        21,
        "aim1.source_restricted_e0.surgen.virchow2_cls",
        "aim1.source_restricted_e0.tcga.virchow2_cls",
    ),
)

EXPECTED_CONTRAST_POSITIONAL_MISMATCHES = (
    (
        1,
        "aim1.source_restricted_e0_encoder_delta.tcga.virchow2_cls_minus_univ1",
        "aim1.source_restricted_e0_encoder_delta.sr1482.virchow2_cls_minus_univ1",
    ),
    (
        3,
        "aim1.source_restricted_e0_encoder_delta.sr1482.virchow2_cls_minus_univ1",
        "aim1.source_restricted_e0_encoder_delta.surgen.virchow2_cls_minus_univ1",
    ),
    (
        4,
        "aim1.source_restricted_e0_encoder_delta.surgen.virchow2_cls_minus_univ1",
        "aim1.source_restricted_e0_encoder_delta.tcga.virchow2_cls_minus_univ1",
    ),
)

_ORIGINAL_REPORT_CLAIM_ROWS = legacy._report_claim_rows
_ORIGINAL_REPORT_CONTRAST_ROWS = legacy._report_contrast_rows
_ORIGINAL_WHY_D_RECORDS = legacy._why_d_evidence_records
_RUNTIME_LOCK = threading.RLock()


def _artifact_record(path: Path, sha256: str, size_bytes: int) -> dict[str, Any]:
    return {"path": str(path.resolve(strict=False)), "sha256": sha256, "size_bytes": size_bytes}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _mean_erratum_source_identities() -> dict[str, dict[str, Any]]:
    expected = {
        "mean_erratum_controller": _artifact_record(
            Path(mean.__file__), MEAN_ERRATUM_SHA256, MEAN_ERRATUM_SIZE
        ),
        "mean_erratum_controller_test": _artifact_record(
            mean.ERRATUM_TEST, MEAN_ERRATUM_TEST_SHA256, MEAN_ERRATUM_TEST_SIZE
        ),
    }
    for name, identity in expected.items():
        if legacy._artifact(Path(str(identity["path"]))) != identity:
            raise GovernanceError(f"Frozen {name} bytes drifted")
    return expected


def _analysis_artifact_identities(root: Path) -> dict[str, dict[str, Any]]:
    directory = v3.continuation_analysis_root(root)
    expected = {
        key: _artifact_record(directory / relative, sha256, size_bytes)
        for key, (relative, sha256, size_bytes) in ANALYSIS_PINS.items()
    }
    for name, identity in expected.items():
        if legacy._artifact(Path(str(identity["path"]))) != identity:
            raise GovernanceError(f"Frozen analysis artifact {name} bytes drifted")
    return expected


def _replay_rows(results: Mapping[str, Any]) -> tuple[list[dict[str, Any]], ...]:
    aim1 = results["aim1"]
    aim2 = results["aim2"]
    claims = _ORIGINAL_REPORT_CLAIM_ROWS(
        aim1["canonical_all_primary_e0"]["performance"],
        aim1["source_restricted_tcga_surgen_e0"]["performance"],
        aim2["targets"],
        aim1["e1a"],
        aim1["e1d"],
        aim2["rih_disjoint_role_contrast"],
        aim2["cpht_raw"],
    )
    contrasts = _ORIGINAL_REPORT_CONTRAST_ROWS(
        aim1["source_restricted_tcga_surgen_e0"]["paired_encoder_contrasts"],
        aim1["e1a"],
        aim1["e1a_s"],
        aim1["e1d"],
        aim2["rih_disjoint_role_contrast"],
    )
    why_d = _ORIGINAL_WHY_D_RECORDS(aim1["why_d"])
    return claims, contrasts, why_d


def _row_map(rows: Any, *, key: str, expected_count: int) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise GovernanceError(f"Report-order {key} row census drifted")
    if any(not isinstance(row, dict) or not isinstance(row.get(key), str) for row in rows):
        raise GovernanceError(f"Report-order {key} row schema drifted")
    result = {str(row[key]): row for row in rows}
    if len(result) != expected_count:
        raise GovernanceError(f"Report-order {key} values are not unique")
    return result


def _position_mismatches(
    stored: Sequence[Mapping[str, Any]], replayed: Sequence[Mapping[str, Any]], *, key: str
) -> tuple[tuple[int, str, str], ...]:
    if len(stored) != len(replayed):
        raise GovernanceError("Report-order compared-list lengths drifted")
    return tuple(
        (index, str(left[key]), str(right[key]))
        for index, (left, right) in enumerate(zip(stored, replayed, strict=True))
        if left != right
    )


def _validate_order_only_defect(results: Mapping[str, Any]) -> dict[str, Any]:
    stored_claims = results.get("report_claim_rows")
    stored_contrasts = results.get("report_contrast_rows")
    stored_why_d = results.get("why_d_evidence_records")
    claims, contrasts, why_d = _replay_rows(results)
    stored_claim_map = _row_map(stored_claims, key="row_id", expected_count=78)
    replay_claim_map = _row_map(claims, key="row_id", expected_count=78)
    stored_contrast_map = _row_map(stored_contrasts, key="row_id", expected_count=48)
    replay_contrast_map = _row_map(contrasts, key="row_id", expected_count=48)
    stored_why_map = _row_map(stored_why_d, key="record_id", expected_count=2)
    replay_why_map = _row_map(why_d, key="record_id", expected_count=2)
    stored_claim_ids = tuple(row["row_id"] for row in stored_claims)
    stored_contrast_ids = tuple(row["row_id"] for row in stored_contrasts)
    replay_claim_ids = [row["row_id"] for row in claims]
    replay_contrast_ids = [row["row_id"] for row in contrasts]
    claim_mismatches = _position_mismatches(stored_claims, claims, key="row_id")
    contrast_mismatches = _position_mismatches(stored_contrasts, contrasts, key="row_id")
    positional_profile = {
        "claim_mismatches": [
            {"index": index, "stored_row_id": stored, "replayed_row_id": replayed}
            for index, stored, replayed in claim_mismatches
        ],
        "contrast_mismatches": [
            {"index": index, "stored_row_id": stored, "replayed_row_id": replayed}
            for index, stored, replayed in contrast_mismatches
        ],
    }
    if (
        stored_claim_ids != STORED_CLAIM_IDS
        or stored_contrast_ids != STORED_CONTRAST_IDS
        or _canonical_sha256(list(stored_claim_ids)) != STORED_CLAIM_ID_SHA256
        or _canonical_sha256(list(stored_contrast_ids)) != STORED_CONTRAST_ID_SHA256
        or _canonical_sha256(replay_claim_ids) != REPLAY_CLAIM_ID_SHA256
        or _canonical_sha256(replay_contrast_ids) != REPLAY_CONTRAST_ID_SHA256
        or _canonical_sha256(stored_claims) != STORED_CLAIM_ROWS_SHA256
        or _canonical_sha256(claims) != REPLAY_CLAIM_ROWS_SHA256
        or _canonical_sha256(stored_contrasts) != STORED_CONTRAST_ROWS_SHA256
        or _canonical_sha256(contrasts) != REPLAY_CONTRAST_ROWS_SHA256
        or _canonical_sha256(stored_why_d) != WHY_D_ROWS_SHA256
        or _canonical_sha256(why_d) != WHY_D_ROWS_SHA256
        or stored_claim_map != replay_claim_map
        or stored_contrast_map != replay_contrast_map
        or stored_why_map != replay_why_map
        or claim_mismatches != EXPECTED_CLAIM_POSITIONAL_MISMATCHES
        or contrast_mismatches != EXPECTED_CONTRAST_POSITIONAL_MISMATCHES
        or _canonical_sha256(positional_profile) != POSITIONAL_PROFILE_SHA256
    ):
        raise GovernanceError("Published report rows do not match the exact order-only defect")
    return {
        "status": "exact_id_keyed_rows_with_serialization_order_only_drift",
        "performance_rows": 78,
        "performance_positional_mismatches": 12,
        "contrast_rows": 48,
        "contrast_positional_mismatches": 3,
        "why_d_rows": 2,
        "why_d_positional_mismatches": 0,
        "stored_claim_id_sequence_sha256": STORED_CLAIM_ID_SHA256,
        "stored_contrast_id_sequence_sha256": STORED_CONTRAST_ID_SHA256,
        "positional_profile_sha256": POSITIONAL_PROFILE_SHA256,
        "id_keyed_claim_rows_equal": True,
        "id_keyed_contrast_rows_equal": True,
        "why_d_rows_equal": True,
    }


def _reorder_replayed_rows(
    rows: list[dict[str, Any]], *, expected_ids: tuple[str, ...], replay_sha256: str
) -> list[dict[str, Any]]:
    mapping = _row_map(rows, key="row_id", expected_count=len(expected_ids))
    observed_ids = [row["row_id"] for row in rows]
    if set(mapping) != set(expected_ids) or _canonical_sha256(observed_ids) != replay_sha256:
        raise GovernanceError("Report-order bridge received an unexpected replay roster/order")
    return [mapping[row_id] for row_id in expected_ids]


def _ordered_claim_replay(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return _reorder_replayed_rows(
        _ORIGINAL_REPORT_CLAIM_ROWS(*args, **kwargs),
        expected_ids=STORED_CLAIM_IDS,
        replay_sha256=REPLAY_CLAIM_ID_SHA256,
    )


def _ordered_contrast_replay(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return _reorder_replayed_rows(
        _ORIGINAL_REPORT_CONTRAST_ROWS(*args, **kwargs),
        expected_ids=STORED_CONTRAST_IDS,
        replay_sha256=REPLAY_CONTRAST_ID_SHA256,
    )


@contextlib.contextmanager
def _scoped_order_verification_runtime() -> Iterator[None]:
    with _RUNTIME_LOCK, mean._scoped_erratum_runtime():
        originals = {
            "_report_claim_rows": legacy._report_claim_rows,
            "_report_contrast_rows": legacy._report_contrast_rows,
        }
        try:
            legacy._report_claim_rows = _ordered_claim_replay
            legacy._report_contrast_rows = _ordered_contrast_replay
            yield
        finally:
            for name, value in originals.items():
                setattr(legacy, name, value)


def _validate_exact_published_bundle(root: Path) -> dict[str, Any]:
    mean_sources = _mean_erratum_source_identities()
    v3_sources = mean._v3_source_identities()
    controls = mean._v3_control_identities(root)
    analysis = _analysis_artifact_identities(root)
    expected_inventory = v3._expected_final_inventory(root)
    observed_inventory = {path for path in v3.continuation_root(root).rglob("*") if path.is_file()}
    if observed_inventory != expected_inventory or any(
        path.is_symlink() for path in v3.continuation_root(root).rglob("*")
    ):
        raise GovernanceError("Frozen final 111-file continuation inventory drifted")
    with mean._scoped_erratum_runtime():
        mean._validate_sealed_v3(root, deep=True)
        contract = legacy._read_json(v3.continuation_analysis_root(root) / "contract.json")
        mean._validate_analysis_contract_erratum(contract)
    receipt = legacy._read_json(
        v3.continuation_analysis_root(root) / "analysis_completion_receipt.json"
    )
    expected_receipt_artifacts = {
        "contract": analysis["analysis_contract"],
        "patient_native_logits": analysis["patient_native_logits"],
        "bootstrap_distributions": analysis["bootstrap_distributions"],
        "results": analysis["results"],
    }
    if (
        receipt.get("status") != "complete_and_verified"
        or receipt.get("artifacts") != expected_receipt_artifacts
        or receipt.get("analysis_artifact_count") != 5
        or receipt.get("patient_rows") != 6_342
        or receipt.get("bootstrap_draws") != 10_000
        or receipt.get("report_claim_row_count") != 78
        or receipt.get("report_contrast_row_count") != 48
        or receipt.get("why_d_evidence_record_count") != 2
    ):
        raise GovernanceError("Frozen analysis completion receipt semantics drifted")
    return {
        "v3_sources": v3_sources,
        "mean_erratum_sources": mean_sources,
        "controls": controls,
        "analysis": analysis,
    }


def _verification_source_graph(
    root: Path, bundle: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inference, analysis = mean._source_graph(root)
    if inference != {
        "continuation_v3_controller": bundle["v3_sources"]["controller"],
        "continuation_v3_controller_test": bundle["v3_sources"]["controller_test"],
        "continuation_v3_contract": bundle["controls"]["contract"],
        "continuation_v3_score_job_plan": bundle["controls"]["score_job_plan"],
        "continuation_v3_deep_preflight": bundle["controls"]["deep_preflight"],
        "continuation_v3_scoring_completion": bundle["controls"]["scoring_completion"],
        "continuation_v3_inference_environment": bundle["controls"]["inference_environment"],
        "continuation_v3_inference_seal": bundle["controls"]["inference_seal"],
    } or analysis != {
        "mean_erratum_controller": bundle["mean_erratum_sources"]["mean_erratum_controller"],
        "mean_erratum_controller_test": bundle["mean_erratum_sources"][
            "mean_erratum_controller_test"
        ],
        **bundle["analysis"],
    }:
        raise GovernanceError("Mean-erratum 15-record source graph drifted")
    verification = {
        "report_order_erratum_controller": legacy._artifact(Path(__file__).resolve()),
        "report_order_erratum_controller_test": legacy._artifact(ORDER_ERRATUM_TEST),
    }
    if len(set(inference) | set(analysis) | set(verification)) != 17:
        raise GovernanceError("Report-order verification source graph is not exactly 17 records")
    return inference, analysis, verification


def verify(campaign_root: Path) -> dict[str, Any]:
    root = legacy._safe_campaign_root(campaign_root)
    bundle = _validate_exact_published_bundle(root)
    results_path = v3.continuation_analysis_root(root) / "results.json"
    before = legacy._artifact(results_path)
    profile = _validate_order_only_defect(legacy._read_json(results_path))
    with _scoped_order_verification_runtime():
        verified = mean.verify(root)
    refreshed_bundle = _validate_exact_published_bundle(root)
    if refreshed_bundle != bundle or legacy._artifact(results_path) != before:
        raise GovernanceError("Frozen analysis evidence changed during report-order verification")
    if _validate_order_only_defect(legacy._read_json(results_path)) != profile:
        raise GovernanceError("Report-order defect profile changed during verification")
    inference, analysis, verification = _verification_source_graph(root, bundle)
    return {
        **verified,
        "status": "PASS_WITH_BOUNDED_REPORT_ORDER_VERIFICATION_ERRATUM",
        "report_order_erratum": profile,
        "campaign_artifact_count": 111,
        "direct_source_graph_count": 17,
        "inference_source_graph": inference,
        "analysis_source_graph": analysis,
        "verification_erratum_source_graph": verification,
    }


def status(campaign_root: Path) -> dict[str, Any]:
    root = legacy._safe_campaign_root(campaign_root)
    directory = v3.continuation_analysis_root(root)
    present = sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []
    return {
        "campaign_root": str(root),
        "analysis_root": str(directory),
        "analysis_files_present": present,
        "exact_five_present": present == sorted(v3.ANALYSIS_FILES),
        "write_surface": False,
        "verification_erratum": ERRATUM,
    }


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "status"):
        child = subparsers.add_parser(command)
        child.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "verify":
        _print(verify(args.campaign_root))
    elif args.command == "status":
        _print(status(args.campaign_root))
    else:  # pragma: no cover
        raise GovernanceError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
