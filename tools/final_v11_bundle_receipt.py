#!/usr/bin/env python3
"""Stage, verify, and exactly-once seal the comprehensive FINAL-v11 bundle.

FINAL-v11 is an additive, reorganized report over the sealed FINAL-v10.5
evidence graph.  The draft verifier recursively replays FINAL-v10.5 and
directly rehashes each of its 94 authoritative sources.  The incident-aware
extension reserves 45 direct source records for the TCGA+SurGen
two-encoder campaign, its two bounded training recoveries, two immutable
downstream preparation incidents, and the continuation-v3 label-blind
analysis, including frozen additive mean-validator and report-order
verification errata.  The exact five analysis outputs are pinned and replayed.

This module deliberately ships with unfrozen document and publication pins.
Consequently the production bundle cannot become a
candidate and cannot be sealed merely because this scaffold exists.  Only
``--refresh-manifest`` mutates the draft source manifest, and only ``--seal``
can publish ``report_bundle_receipt.json``.  Publication is non-overwriting,
atomic, and followed by byte-exact replay.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import final_v10_5_bundle_receipt as parent  # noqa: E402

FINAL_V11 = REPO / "reports" / "final_v11"
PARENT_DIR = REPO / "reports" / "final_v10_5"
PARENT_RECEIPT = PARENT_DIR / "report_bundle_receipt.json"
PARENT_MANIFEST = PARENT_DIR / "source_manifest.json"
PARENT_VERIFIER = REPO / "tools" / "final_v10_5_bundle_receipt.py"
PARENT_TEST = REPO / "tests" / "test_final_v10_5_bundle_receipt.py"

CAMPAIGN_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim1_tcga_surgen_two_encoder_v1_20260824"
)
CAMPAIGN_CONTROLLER = REPO / "tools" / "aim1_tcga_surgen_two_encoder_campaign.py"
CAMPAIGN_TEST = REPO / "tests" / "test_aim1_tcga_surgen_two_encoder_campaign.py"
RECOVERY_V1_CONTROLLER = REPO / "tools" / "aim1_tcga_surgen_two_encoder_recovery.py"
RECOVERY_V1_TEST = REPO / "tests" / "test_aim1_tcga_surgen_two_encoder_recovery.py"
RECOVERY_V2_CONTROLLER = REPO / "tools" / "aim1_tcga_surgen_two_encoder_recovery_v2.py"
RECOVERY_V2_TEST = REPO / "tests" / "test_aim1_tcga_surgen_two_encoder_recovery_v2.py"
LEGACY_ANALYSIS_CONTROLLER = REPO / "tools" / "aim1_tcga_surgen_full_pipeline_analysis.py"
LEGACY_ANALYSIS_TEST = REPO / "tests" / "test_aim1_tcga_surgen_full_pipeline_analysis.py"
INCIDENT_V2_CONTROLLER = REPO / "tools" / "aim1_tcga_surgen_full_pipeline_analysis_v2.py"
INCIDENT_V2_TEST = REPO / "tests" / "test_aim1_tcga_surgen_full_pipeline_analysis_v2.py"
ANALYSIS_CONTROLLER = REPO / "tools" / "aim1_tcga_surgen_full_pipeline_analysis_v3.py"
ANALYSIS_TEST = REPO / "tests" / "test_aim1_tcga_surgen_full_pipeline_analysis_v3.py"
ANALYSIS_ERRATUM_CONTROLLER = (
    REPO / "tools" / "aim1_tcga_surgen_full_pipeline_analysis_mean_erratum.py"
)
ANALYSIS_ERRATUM_TEST = (
    REPO / "tests" / "test_aim1_tcga_surgen_full_pipeline_analysis_mean_erratum.py"
)
REPORT_ORDER_ERRATUM_CONTROLLER = (
    REPO / "tools" / "aim1_tcga_surgen_full_pipeline_analysis_report_order_erratum.py"
)
REPORT_ORDER_ERRATUM_TEST = (
    REPO / "tests" / "test_aim1_tcga_surgen_full_pipeline_analysis_report_order_erratum.py"
)

REPORT_DOCUMENTS = ("Experimental_Setup.md", "Results.md", "Audit.md")
SOURCE_MANIFEST_NAME = "source_manifest.json"
FINAL_RECEIPT_NAME = "report_bundle_receipt.json"

SEALED_STATUS = parent.SEALED_STATUS
DRAFT_STATUS = "draft_awaiting_final_v11_independent_review_and_flattening"
CANDIDATE_STATUS = "candidate_ready_for_final_v11_verification"
EXPECTED_PARENT_SOURCE_COUNT = 94
EXPECTED_NEW_SOURCE_COUNT = 45
EXPECTED_COMPLETE_SOURCE_COUNT = 139
EXPECTED_PARENT_FIT_CENSUS = 1325
EXPECTED_NEW_FITS = 35
EXPECTED_CAMPAIGN_LINEAGE_FITS = 60
EXPECTED_COMPLETE_FIT_CENSUS = 1360
EXPECTED_MAX_CONCURRENCY = 6
EXPECTED_SCORE_JOBS = 50
EXPECTED_SCORE_SLIDE_ROWS = 4790
EXPECTED_ANALYSIS_ARTIFACTS = 5
EXPECTED_ANALYSIS_PATIENT_ROWS = 6_342
EXPECTED_BOOTSTRAP_DRAWS = 10_000
EXPECTED_BOOTSTRAP_SEED = 20_260_824
EXPECTED_ANALYSIS_EXPERIMENT = "final_v11_tcga_surgen_analysis_mean_validator_erratum_v4"
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_ENCODERS = ("univ1", "virchow2_cls")
EXPECTED_TRAINING_ENCODERS = ("UNI-v1", "Virchow2-CLS")
EXPECTED_TARGETS = (
    "cptac_primary",
    "rih_primary",
    "rih_metastatic",
    "sr1482_metastatic",
    "orion_cpht",
)
EXPECTED_CANONICAL_E0_POPULATIONS = (
    "pooled_all_primary",
    "tcga",
    "sr386",
    "sr1482",
    "surgen",
    "tcga_surgen",
)
EXPECTED_SOURCE_E0_POPULATIONS = (
    "pooled_source",
    "tcga",
    "sr386",
    "sr1482",
    "surgen",
    "tcga_surgen",
)
EXPECTED_E1A_POPULATIONS = (
    "A_all_primary",
    "A_complete",
    "B_mss",
    "C_braf_wt",
    "D_mss_braf_wt",
    "E_colon",
    "F_rectum",
    "G_stage_known_derived",
    "G_stage_known_frozen",
    "H_stage_iv",
    "I_right_proximal",
    "J_left_distal",
    "K_transverse",
)
EXPECTED_E1D_POPULATIONS = ("A_all_primary", "G_stage_known_derived")
EXPECTED_PERFORMANCE_CLAIM_ROW_IDS = (
    *(
        f"aim1.canonical_e0.{population}.{encoder}"
        for population in EXPECTED_CANONICAL_E0_POPULATIONS
        for encoder in EXPECTED_ENCODERS
    ),
    *(
        f"aim1.source_restricted_e0.{population}.{encoder}"
        for population in EXPECTED_SOURCE_E0_POPULATIONS
        for encoder in EXPECTED_ENCODERS
    ),
    *(f"aim2.{target}.{encoder}" for target in EXPECTED_TARGETS for encoder in EXPECTED_ENCODERS),
    *(
        f"aim1.e1a.{population}.{encoder}"
        for population in EXPECTED_E1A_POPULATIONS
        for encoder in EXPECTED_ENCODERS
    ),
    *(f"aim1.e1d.{population}.clinical" for population in EXPECTED_E1D_POPULATIONS),
    *(
        f"aim1.e1d.{population}.{encoder}.{model}"
        for population in EXPECTED_E1D_POPULATIONS
        for encoder in EXPECTED_ENCODERS
        for model in ("wsi", "fusion")
    ),
    *(
        f"aim2.rih_disjoint_role.{role}.{encoder}"
        for role in ("primary", "metastatic")
        for encoder in EXPECTED_ENCODERS
    ),
    *(
        f"aim2.cpht_raw.{population}.{encoder}"
        for population in ("exclude_neoadjuvant", "exclude_ambiguous_crc15")
        for encoder in EXPECTED_ENCODERS
    ),
)
EXPECTED_CONTRAST_CLAIM_ROW_IDS = (
    *(
        f"aim1.source_restricted_e0_encoder_delta.{population}.virchow2_cls_minus_univ1"
        for population in EXPECTED_SOURCE_E0_POPULATIONS
    ),
    *(
        f"aim1.e1a_delta_A_minus_set.{population}.{encoder}"
        for population in EXPECTED_E1A_POPULATIONS
        if population != "A_all_primary"
        for encoder in EXPECTED_ENCODERS
    ),
    *(
        f"aim1.e1a_s_delta_D_minus_reference.{reference}.{encoder}"
        for reference in ("A_all_primary", "A_complete")
        for encoder in EXPECTED_ENCODERS
    ),
    *(
        f"aim1.e1d_delta.{population}.{encoder}.{comparison}"
        for population in EXPECTED_E1D_POPULATIONS
        for encoder in EXPECTED_ENCODERS
        for comparison in (
            "wsi_minus_clinical",
            "fusion_minus_clinical",
            "fusion_minus_wsi",
        )
    ),
    *(
        f"aim2.rih_disjoint_role.delta_metastatic_minus_primary.{encoder}"
        for encoder in EXPECTED_ENCODERS
    ),
)
EXPECTED_WHY_D_RECORD_IDS = tuple(f"aim1.why_d.{encoder}" for encoder in EXPECTED_ENCODERS)
EXPECTED_REPORT_CLAIM_ROW_IDS = EXPECTED_PERFORMANCE_CLAIM_ROW_IDS
EXPECTED_DERIVED_COVARIATE_SOURCE = {
    "artifact": {
        "path": "/mnt/d/YC.Liu/manifests/colon/crc_final_v4.csv",
        "size_bytes": 1_431_625,
        "sha256": "312438c12aaa25b4376cc55b70ebc3a23763f149c7413f1fde7c3ce71c21596c",
    },
    "source_patient_census": {
        "G_stage_known_derived": [1_060, 422],
        "G_stage_known_frozen": [895, 344],
        "H_stage_iv": [120, 54],
        "I_right_proximal": [212, 99],
        "J_left_distal": [573, 212],
        "K_transverse": [42, 10],
    },
}
EXPECTED_ANALYSIS_MEAN_ERRATUM = {
    "status": "bounded_floating_point_mean_validator_erratum",
    "returned_patient_table_mutated": False,
    "scores_or_seal_mutated": False,
    "mismatch_rows": 18,
    "exact_rows": 6_324,
    "block_counts": {
        "univ1|cptac_primary": 3,
        "univ1|sr1482_metastatic": 10,
        "virchow2_cls|cptac_primary": 1,
        "virchow2_cls|sr1482_metastatic": 4,
    },
    "max_abs": 8.881784197001252e-16,
    "rtol": 0.0,
    "atol": 1.0e-15,
    "records_sha256": "ea5ef308db921565024fb1fa37598ab1d771d886d2b5a98bc6ba23fcb6b9874f",
    "validation_bridge": (
        "run frozen patient validator on a copy whose redundant mean is recomputed "
        "from the five patient seed columns"
    ),
}
EXPECTED_REPORT_ORDER_ERRATUM = {
    "status": "exact_id_keyed_rows_with_serialization_order_only_drift",
    "performance_rows": 78,
    "performance_positional_mismatches": 12,
    "contrast_rows": 48,
    "contrast_positional_mismatches": 3,
    "why_d_rows": 2,
    "why_d_positional_mismatches": 0,
    "stored_claim_id_sequence_sha256": (
        "6c6587bfabf1a56c874e8db48c2ab78582c1e1b826baae3c809af78b4939e237"
    ),
    "stored_contrast_id_sequence_sha256": (
        "72abd95f8fb022faa3b95a974d91bfc452bb79a8ac861f45e507337ee0347a8f"
    ),
    "positional_profile_sha256": (
        "80a533bf327bea346c5a4f2ad94e9ae59c786d827b7564a9976dc1dbb66936ee"
    ),
    "id_keyed_claim_rows_equal": True,
    "id_keyed_contrast_rows_equal": True,
    "why_d_rows_equal": True,
}
EXPECTED_ANALYSIS_OUTPUT_INVENTORY = (
    "contract.json",
    "patient_native_logits.parquet",
    "bootstrap_distributions.npz",
    "results.json",
    "analysis_completion_receipt.json",
)

EXPECTED_FROZEN_PARENT_SHA256 = {
    "parent_receipt": "3e82b528962500a11b7a8fd956851fe013c61adc5539515078a9c12ac81b6934",
    "parent_manifest": "bf0112fb5e7136900593ba7af5a3912389e5d5da1af3586679ebdce135bb3df6",
    "parent_verifier": "bb7e0e49553bf4299481cde2523fbaa67ac065962ddcc935a34f5959164ca83f",
    "parent_test": "4417bbc32af6d28f825a4bfa55076a9aab6e5109bc4ce40f8fec18d4e1ea3006",
}
EXPECTED_FROZEN_DOWNSTREAM_CODE_SHA256 = {
    "campaign_controller": "4635042e76c3ad3f8fc9aea46e478881681d4a1dfeaef046e0968424d7e2bcc7",
    "campaign_test": "9609411e1bff2feb7517a23931a1e3907eb4984bee90da5d7cdd95c750ae3a16",
    "recovery_v1_controller": "d47cbd54b584fb3d0dd5160ae0bdd1fe2aca79b53207f8c1b63e366957b66ae5",
    "recovery_v1_test": "2cf182aab9cb8a92d4416402024251a7b450adc11e0be7da4f24d0bfac6c2f24",
    "recovery_v2_controller": "e474781f955a932cef575e6f8452f7eb824e178e1fbbe9254432ef7008b4687d",
    "recovery_v2_test": "2dc7b5362edbb55bfdd84d53cb3ff61ce5b30b0fc12c0a1a1fa5b9f763d52891",
    "legacy_analysis_controller": "9f86156da96a8fb2620d45a692c222f46cceef9fa546afd8dd4ca35e09178292",
    "legacy_analysis_test": "ac971cca54cac60bda914e7729bb817f47e9513f74e6a5cce55ee4db5dfd0219",
    "incident_v2_controller": "5ff574f3143c1a2c695334e6b9afb6fa126e7a7e814f787d156f63511fe92345",
    "incident_v2_test": "db2c668c8d1be483e0f8aeecd06fd21b099a5eae03efa9011e632618a51fb08e",
    "analysis_controller": "fadffae63348b2d664ea9f49b2d8126cc0e7869a2eb9bed1245153800b974767",
    "analysis_test": "f0329f2f7bd6cf9f549ae6f8763480a8766819c9ad3cf6920f4d364bb61aa32e",
    "analysis_erratum_controller": (
        "675c190421a057949f9f22dc6a3710d38970625113af182a8120b44307542192"
    ),
    "analysis_erratum_test": ("b7e49678bdc1c8cde5a880c7586ea1df476a863bb27fdf04b500fb82d33da92f"),
    "report_order_erratum_controller": (
        "7e87acf801c6dff2ca4c23792c8f4d9c0f3727977bf4c3b38035e6d19f7919d1"
    ),
    "report_order_erratum_test": (
        "cce7a74b5bf8b104198a133cc4f22fb21b6ad0b6010b54515ad303d020ae1d2b"
    ),
}

EXPECTED_FINAL_ANALYSIS_SOURCE_IDENTITIES = {
    "aim1-tcga-surgen-two-encoder-downstream-v3-analysis-contract": {
        "size_bytes": 36_974,
        "sha256": "3b1655fbf421d05b862615cff24dd2526c1cdf6f4112830bc158456d6c36b455",
    },
    "aim1-tcga-surgen-two-encoder-downstream-v3-patient-native-logits": {
        "size_bytes": 311_419,
        "sha256": "02f4a5bf0c2cd909c5ecb3f9edc7c05ae725649199a015cf0a1d0cb544df8d09",
    },
    "aim1-tcga-surgen-two-encoder-downstream-v3-bootstrap": {
        "size_bytes": 24_163_117,
        "sha256": "7d47405fe4ac58e4134e58ff81bf88869d0ccf9e6258844ebc0760d8c08f7aaf",
    },
    "aim1-tcga-surgen-two-encoder-downstream-v3-results": {
        "size_bytes": 299_495,
        "sha256": "a46faeffcbb30e436776e4853602cccf801a31228ef490ab9caabac48f72f2b6",
    },
    "aim1-tcga-surgen-two-encoder-downstream-v3-analysis-completion": {
        "size_bytes": 2_070,
        "sha256": "639696aaa4bdfafa887fc67cc541b976d6c3bf6113fc3909e2f133823df1b125",
    },
}

_UNFROZEN = "REPLACE_AFTER_FINAL_V11_RECONCILIATION"
EXPECTED_FINAL_DOCUMENT_SHA256 = {
    "Experimental_Setup.md": "85a50335db2eb0546f4bbffae8302cfb4cadf7ef86d84debebb52e469f6d05e5",
    "Results.md": "9c737cdb52be2bdcf518a81c758b241e018ff9621db5505ac299abafbe713d1a",
    "Audit.md": "db7f4e733067a4e36c84521b5156bfd6046f5e4ebc8c1ab17f57dbe754a37b1a",
}
EXPECTED_TERMINAL_RECEIPT_SHA256 = {
    "aim1-tcga-surgen-two-encoder-recovery-v1-training-completion": (
        "4c3a2f4626b8a66e37b37afd20b211bb2de8fa32901db49a86d123a572b13c86"
    ),
    "aim1-tcga-surgen-two-encoder-recovery-v2-training-completion": (
        "fa41798f4266ebcb6bfde7ccae7ad4990e8de965c2cdacca4be1276989b43d08"
    ),
    "aim1-tcga-surgen-two-encoder-downstream-v3-inference-seal": (
        "d7a435ddae080fd8fed814a1d9e659844a83cf140565ed1e57972a3be4b24f45"
    ),
    "aim1-tcga-surgen-two-encoder-downstream-v3-analysis-completion": (
        "639696aaa4bdfafa887fc67cc541b976d6c3bf6113fc3909e2f133823df1b125"
    ),
}
EXPECTED_RECEIPT_CREATED_UTC = "2026-08-25T07:10:26+00:00"

FINAL_STATE_STATUS_PARAGRAPH = (
    "FINAL-v11 evidence is complete: sealed FINAL-v10.5 recursively replayed; all 94 "
    "inherited sources and all 45 new sources directly rehashed; the exact 35-new-fit, "
    "60-lineage-fit campaign and label-blind 50-job/4,790-row score seal passed; and "
    "the comprehensive report was verified before exactly-once publication."
)

_DRAFT_KEYS = {
    "schema_version",
    "bundle",
    "status",
    "base_bundle_receipt",
    "base_source_manifest",
    "artifacts",
    "pending_artifacts",
}
_FLAT_KEYS = {"schema_version", "bundle", "status", "artifacts", "pending_artifacts"}
_ARTIFACT_KEYS = {
    "id",
    "aims",
    "experiments",
    "role",
    "path",
    "size_bytes",
    "sha256",
}
_PENDING_KEYS = {"id", "aims", "experiments", "role", "path"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SOURCE_ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_FORBIDDEN_SCORE_COLUMNS = {
    "label",
    "target_label",
    "kras",
    "outcome",
    "outcome_label",
    "mutation_status",
}


class BundleVerificationError(RuntimeError):
    """A fail-closed FINAL-v11 staging, source, report, or seal error."""


ParentValidator = Callable[["BundlePaths"], dict[str, Any]]
StageValidator = Callable[["BundlePaths"], dict[str, Any] | None]


@dataclass(frozen=True)
class BundlePaths:
    """Filesystem locations and frozen candidate handoff values."""

    repo: Path
    final_v11: Path
    destination: Path
    verifier_code: Path
    verifier_test: Path
    parent_dir: Path
    parent_receipt: Path
    parent_manifest: Path
    parent_verifier: Path
    parent_test: Path
    campaign_root: Path
    campaign_controller: Path
    campaign_test: Path
    analysis_controller: Path
    analysis_test: Path
    expected_parent_sha256: dict[str, str]
    expected_downstream_code_sha256: dict[str, str] | None = None
    expected_document_sha256: dict[str, str] | None = None
    expected_terminal_receipt_sha256: dict[str, str] | None = None
    expected_created_utc: str | None = None
    expected_parent_source_count: int = EXPECTED_PARENT_SOURCE_COUNT
    parent_validator: ParentValidator | None = None
    campaign_validator: StageValidator | None = None
    analysis_validator: StageValidator | None = None


def default_paths() -> BundlePaths:
    """Return production FINAL-v11 staging paths."""

    return BundlePaths(
        repo=REPO,
        final_v11=FINAL_V11,
        destination=FINAL_V11 / FINAL_RECEIPT_NAME,
        verifier_code=Path(__file__).resolve(),
        verifier_test=REPO / "tests" / "test_final_v11_bundle_receipt.py",
        parent_dir=PARENT_DIR,
        parent_receipt=PARENT_RECEIPT,
        parent_manifest=PARENT_MANIFEST,
        parent_verifier=PARENT_VERIFIER,
        parent_test=PARENT_TEST,
        campaign_root=CAMPAIGN_ROOT,
        campaign_controller=CAMPAIGN_CONTROLLER,
        campaign_test=CAMPAIGN_TEST,
        analysis_controller=ANALYSIS_CONTROLLER,
        analysis_test=ANALYSIS_TEST,
        expected_parent_sha256=dict(EXPECTED_FROZEN_PARENT_SHA256),
        expected_downstream_code_sha256=dict(EXPECTED_FROZEN_DOWNSTREAM_CODE_SHA256),
        expected_document_sha256=dict(EXPECTED_FINAL_DOCUMENT_SHA256),
        expected_terminal_receipt_sha256=dict(EXPECTED_TERMINAL_RECEIPT_SHA256),
        expected_created_utc=EXPECTED_RECEIPT_CREATED_UTC,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_symlink_chain(path: Path, *, context: str) -> None:
    candidate = path if path.is_absolute() else path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise BundleVerificationError(f"{context} contains a symlink component: {component}")


def identity(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    _reject_symlink_chain(path, context="artifact path")
    if not path.is_file():
        raise BundleVerificationError(f"expected a regular file: {path}")
    return {
        "path": str(path if display_path is None else display_path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise BundleVerificationError(f"JSON contains duplicate key {key!r}")
        value[key] = child
    return value


def _reject_nonfinite_json_constant(value: str) -> None:
    raise BundleVerificationError(f"JSON contains non-finite constant {value}")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _reject_symlink_chain(path, context=label)
    if not path.is_file():
        raise BundleVerificationError(f"{label} is missing: {path}")
    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleVerificationError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise BundleVerificationError(f"{label} must be a JSON object")
    return parsed


def _display(path: Path, repo: Path) -> str:
    try:
        return str(path.absolute().relative_to(repo.absolute()))
    except ValueError:
        return str(path.absolute())


def _lexical_path(raw_path: str, paths: BundlePaths) -> Path:
    path = Path(raw_path)
    if any(part == ".." for part in path.parts):
        raise BundleVerificationError(f"source path contains parent traversal: {raw_path}")
    return path if path.is_absolute() else paths.repo / path


def _record_identity(record: Mapping[str, Any], paths: BundlePaths) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise BundleVerificationError("source record path is invalid")
    path = _lexical_path(raw_path, paths)
    actual = identity(path, display_path=raw_path)
    if actual["size_bytes"] != record.get("size_bytes") or actual["sha256"] != record.get("sha256"):
        raise BundleVerificationError(f"source identity drift: {record.get('id', raw_path)}")
    return path


def _source_specs(paths: BundlePaths) -> dict[str, dict[str, Any]]:
    train_common = {
        "aims": ["Aim 1", "Aim 2"],
        "experiments": ["TCGA+SurGen two-encoder E0 and deployment refits"],
    }
    recovery_common = {
        "aims": ["Aim 1", "Aim 2"],
        "experiments": ["TCGA+SurGen training certification and bounded errata"],
    }
    analysis_common = {
        "aims": ["Aim 1", "Aim 2"],
        "experiments": ["Aim 1 full pipeline and Aim 2 label-blind zero-shot"],
    }
    root = paths.campaign_root
    continuation = root / "downstream_v2/continuation_v3"
    return {
        "aim1-tcga-surgen-two-encoder-campaign-controller": {
            **train_common,
            "role": "governed_two_encoder_campaign_controller",
            "path": _display(paths.campaign_controller, paths.repo),
        },
        "aim1-tcga-surgen-two-encoder-campaign-test": {
            **train_common,
            "role": "focused_two_encoder_campaign_validation",
            "path": _display(paths.campaign_test, paths.repo),
        },
        "aim1-tcga-surgen-two-encoder-campaign-contract": {
            **train_common,
            "role": "immutable_tcga_surgen_two_encoder_contract",
            "path": str(root / "contract.json"),
        },
        "aim1-tcga-surgen-two-encoder-deep-preflight": {
            **train_common,
            "role": "roster_split_feature_pack_resource_preflight",
            "path": str(root / "receipts/deep_preflight.json"),
        },
        "aim1-tcga-surgen-two-encoder-recovery-v1-controller": {
            **recovery_common,
            "role": "bounded_postfit_validator_recovery_controller",
            "path": _display(
                paths.repo / "tools/aim1_tcga_surgen_two_encoder_recovery.py",
                paths.repo,
            ),
        },
        "aim1-tcga-surgen-two-encoder-recovery-v1-test": {
            **recovery_common,
            "role": "bounded_postfit_validator_recovery_validation",
            "path": _display(
                paths.repo / "tests/test_aim1_tcga_surgen_two_encoder_recovery.py",
                paths.repo,
            ),
        },
        "aim1-tcga-surgen-two-encoder-recovery-v1-erratum-contract": {
            **recovery_common,
            "role": "exact_three_key_training_identity_schema_erratum",
            "path": str(root / "recovery_v1/contract_erratum.json"),
        },
        "aim1-tcga-surgen-two-encoder-recovery-v1-adjudication": {
            **recovery_common,
            "role": "raw_training_artifact_and_validator_adjudication",
            "path": str(root / "recovery_v1/receipts/validator_adjudication.json"),
        },
        "aim1-tcga-surgen-two-encoder-recovery-v1-scheduler": {
            **recovery_common,
            "role": "transparent_native_rc0_wrapper_rc1_scheduler_recovery",
            "path": str(root / "recovery_v1/receipts/scheduler_recovery.json"),
        },
        "aim1-tcga-surgen-two-encoder-recovery-v1-training-completion": {
            **recovery_common,
            "role": "terminal_35_physical_60_lineage_fit_bounded_erratum_receipt",
            "path": str(root / "recovery_v1/receipts/training_complete_recovered.json"),
        },
        "aim1-tcga-surgen-two-encoder-recovery-v2-controller": {
            **recovery_common,
            "role": "scoped_immutability_recovery_controller",
            "path": _display(
                paths.repo / "tools/aim1_tcga_surgen_two_encoder_recovery_v2.py",
                paths.repo,
            ),
        },
        "aim1-tcga-surgen-two-encoder-recovery-v2-test": {
            **recovery_common,
            "role": "scoped_immutability_recovery_validation",
            "path": _display(
                paths.repo / "tests/test_aim1_tcga_surgen_two_encoder_recovery_v2.py",
                paths.repo,
            ),
        },
        "aim1-tcga-surgen-two-encoder-recovery-v2-scope-contract": {
            **recovery_common,
            "role": "immutable_441_file_training_scope_erratum",
            "path": str(root / "recovery_v2/contract_scope_erratum.json"),
        },
        "aim1-tcga-surgen-two-encoder-recovery-v2-scope-adjudication": {
            **recovery_common,
            "role": "scoped_training_and_delegated_namespace_adjudication",
            "path": str(root / "recovery_v2/receipts/scope_adjudication.json"),
        },
        "aim1-tcga-surgen-two-encoder-recovery-v2-training-completion": {
            **recovery_common,
            "role": "terminal_scoped_training_certification_receipt",
            "path": str(root / "recovery_v2/receipts/training_complete_scoped.json"),
        },
        "aim1-tcga-surgen-two-encoder-legacy-prepare-controller": {
            **analysis_common,
            "role": "immutable_legacy_prepare_controller_incident_provenance",
            "path": _display(
                paths.repo / "tools/aim1_tcga_surgen_full_pipeline_analysis.py",
                paths.repo,
            ),
        },
        "aim1-tcga-surgen-two-encoder-legacy-prepare-test": {
            **analysis_common,
            "role": "immutable_legacy_prepare_controller_validation",
            "path": _display(
                paths.repo / "tests/test_aim1_tcga_surgen_full_pipeline_analysis.py",
                paths.repo,
            ),
        },
        "aim1-tcga-surgen-two-encoder-legacy-prepare-contract": {
            **analysis_common,
            "role": "immutable_label_blind_legacy_prepare_contract",
            "path": str(root / "downstream/contract.json"),
        },
        "aim1-tcga-surgen-two-encoder-legacy-prepare-cptac-primary": {
            **analysis_common,
            "role": "label_blind_cptac_primary_scoring_roster",
            "path": str(root / "downstream/inputs/label_blind/cptac_primary.csv"),
        },
        "aim1-tcga-surgen-two-encoder-legacy-prepare-orion-cpht": {
            **analysis_common,
            "role": "label_blind_orion_cpht_scoring_roster",
            "path": str(root / "downstream/inputs/label_blind/orion_cpht.csv"),
        },
        "aim1-tcga-surgen-two-encoder-legacy-prepare-rih-metastatic": {
            **analysis_common,
            "role": "label_blind_rih_metastatic_scoring_roster",
            "path": str(root / "downstream/inputs/label_blind/rih_metastatic.csv"),
        },
        "aim1-tcga-surgen-two-encoder-legacy-prepare-rih-primary": {
            **analysis_common,
            "role": "label_blind_rih_primary_scoring_roster",
            "path": str(root / "downstream/inputs/label_blind/rih_primary.csv"),
        },
        "aim1-tcga-surgen-two-encoder-legacy-prepare-sr1482-metastatic": {
            **analysis_common,
            "role": "label_blind_sr1482_metastatic_scoring_roster",
            "path": str(root / "downstream/inputs/label_blind/sr1482_metastatic.csv"),
        },
        "aim1-tcga-surgen-two-encoder-legacy-prepare-score-job-plan": {
            **analysis_common,
            "role": "immutable_legacy_prepare_50_job_intent",
            "path": str(root / "downstream/jobs/score_jobs.json"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v2-controller": {
            **analysis_common,
            "role": "immutable_downstream_v2_prepare_incident_controller",
            "path": _display(
                paths.repo / "tools/aim1_tcga_surgen_full_pipeline_analysis_v2.py",
                paths.repo,
            ),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v2-test": {
            **analysis_common,
            "role": "immutable_downstream_v2_prepare_incident_validation",
            "path": _display(
                paths.repo / "tests/test_aim1_tcga_surgen_full_pipeline_analysis_v2.py",
                paths.repo,
            ),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v2-contract": {
            **analysis_common,
            "role": "immutable_downstream_v2_prepare_contract_incident_evidence",
            "path": str(root / "downstream_v2/contract.json"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v2-score-job-plan": {
            **analysis_common,
            "role": "immutable_downstream_v2_prepare_job_plan_incident_evidence",
            "path": str(root / "downstream_v2/jobs/score_jobs.json"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-controller": {
            **analysis_common,
            "role": "governed_continuation_v3_scoring_and_analysis_controller",
            "path": _display(paths.analysis_controller, paths.repo),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-test": {
            **analysis_common,
            "role": "focused_continuation_v3_scoring_and_analysis_validation",
            "path": _display(paths.analysis_test, paths.repo),
        },
        "aim1-tcga-surgen-two-encoder-analysis-mean-erratum-controller": {
            **analysis_common,
            "role": "bounded_analysis_only_floating_mean_validator_erratum",
            "path": _display(
                paths.repo / "tools/aim1_tcga_surgen_full_pipeline_analysis_mean_erratum.py",
                paths.repo,
            ),
        },
        "aim1-tcga-surgen-two-encoder-analysis-mean-erratum-test": {
            **analysis_common,
            "role": "focused_bounded_mean_validator_erratum_validation",
            "path": _display(
                paths.repo / "tests/test_aim1_tcga_surgen_full_pipeline_analysis_mean_erratum.py",
                paths.repo,
            ),
        },
        "aim1-tcga-surgen-two-encoder-analysis-report-order-erratum-controller": {
            **analysis_common,
            "role": "bounded_read_only_report_order_verification_erratum",
            "path": _display(
                paths.repo
                / "tools/aim1_tcga_surgen_full_pipeline_analysis_report_order_erratum.py",
                paths.repo,
            ),
        },
        "aim1-tcga-surgen-two-encoder-analysis-report-order-erratum-test": {
            **analysis_common,
            "role": "focused_id_keyed_report_order_erratum_validation",
            "path": _display(
                paths.repo
                / "tests/test_aim1_tcga_surgen_full_pipeline_analysis_report_order_erratum.py",
                paths.repo,
            ),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-contract": {
            **analysis_common,
            "role": "immutable_continuation_v3_estimand_and_score_contract",
            "path": str(continuation / "contract.json"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-score-job-plan": {
            **analysis_common,
            "role": "continuation_v3_50_job_label_blind_score_plan",
            "path": str(continuation / "jobs/score_jobs.json"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-deep-preflight": {
            **analysis_common,
            "role": "continuation_v3_deep_scoring_preflight",
            "path": str(continuation / "receipts/deep_preflight.json"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-scoring-completion": {
            **analysis_common,
            "role": "continuation_v3_zero_fit_scoring_completion_receipt",
            "path": str(continuation / "receipts/scoring_complete.json"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-inference-environment": {
            **analysis_common,
            "role": "continuation_v3_inference_environment_receipt",
            "path": str(continuation / "inference/environment.json"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-inference-seal": {
            **analysis_common,
            "role": "label_blind_score_seal_before_outcome_join",
            "path": str(continuation / "inference/inference_seal.json"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-analysis-contract": {
            **analysis_common,
            "role": "immutable_estimand_and_outcome_join_contract",
            "path": str(continuation / "analysis/contract.json"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-patient-native-logits": {
            **analysis_common,
            "role": "governed_patient_native_logit_table",
            "path": str(continuation / "analysis/patient_native_logits.parquet"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-bootstrap": {
            **analysis_common,
            "role": "patient_bootstrap_distributions",
            "path": str(continuation / "analysis/bootstrap_distributions.npz"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-results": {
            **analysis_common,
            "role": "aim1_challenge_clinical_and_aim2_zero_shot_results",
            "path": str(continuation / "analysis/results.json"),
        },
        "aim1-tcga-surgen-two-encoder-downstream-v3-analysis-completion": {
            **analysis_common,
            "role": "terminal_deterministic_analysis_replay_receipt",
            "path": str(continuation / "analysis/analysis_completion_receipt.json"),
        },
    }


def _validate_parent_dependency_pins(paths: BundlePaths) -> None:
    expected_paths = {
        "parent_receipt": paths.parent_receipt,
        "parent_manifest": paths.parent_manifest,
        "parent_verifier": paths.parent_verifier,
        "parent_test": paths.parent_test,
    }
    if set(paths.expected_parent_sha256) != set(expected_paths):
        raise BundleVerificationError("frozen parent dependency roster changed")
    if paths.repo.resolve() == REPO.resolve() and (
        paths.expected_parent_sha256 != EXPECTED_FROZEN_PARENT_SHA256
        or paths.parent_validator is not None
        or paths.campaign_validator is not None
        or paths.analysis_validator is not None
    ):
        raise BundleVerificationError("production callbacks or parent pins were overridden")
    for key, path in expected_paths.items():
        if identity(path)["sha256"] != paths.expected_parent_sha256[key]:
            raise BundleVerificationError(f"frozen parent identity drift: {key}")


def _validate_final_downstream_code_pins(paths: BundlePaths) -> None:
    """Authenticate the reconciled downstream implementation before long replay."""

    if paths.repo.resolve() != REPO.resolve():
        return
    expected = paths.expected_downstream_code_sha256
    if expected != EXPECTED_FROZEN_DOWNSTREAM_CODE_SHA256:
        raise BundleVerificationError("production downstream code pins were overridden")
    observed_paths = {
        "campaign_controller": paths.campaign_controller,
        "campaign_test": paths.campaign_test,
        "recovery_v1_controller": RECOVERY_V1_CONTROLLER,
        "recovery_v1_test": RECOVERY_V1_TEST,
        "recovery_v2_controller": RECOVERY_V2_CONTROLLER,
        "recovery_v2_test": RECOVERY_V2_TEST,
        "legacy_analysis_controller": LEGACY_ANALYSIS_CONTROLLER,
        "legacy_analysis_test": LEGACY_ANALYSIS_TEST,
        "incident_v2_controller": INCIDENT_V2_CONTROLLER,
        "incident_v2_test": INCIDENT_V2_TEST,
        "analysis_controller": paths.analysis_controller,
        "analysis_test": paths.analysis_test,
        "analysis_erratum_controller": ANALYSIS_ERRATUM_CONTROLLER,
        "analysis_erratum_test": ANALYSIS_ERRATUM_TEST,
        "report_order_erratum_controller": REPORT_ORDER_ERRATUM_CONTROLLER,
        "report_order_erratum_test": REPORT_ORDER_ERRATUM_TEST,
    }
    if set(expected or {}) != set(observed_paths):
        raise BundleVerificationError("frozen downstream code pin roster changed")
    for key, path in observed_paths.items():
        if identity(path)["sha256"] != expected[key]:
            raise BundleVerificationError(f"frozen downstream code identity drift: {key}")


def _verify_parent(paths: BundlePaths) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_parent_dependency_pins(paths)
    if paths.parent_validator is None:
        canonical_python = paths.repo / ".venv/bin/python"
        if not canonical_python.is_file():
            raise BundleVerificationError(
                f"canonical parent replay interpreter is missing: {canonical_python}"
            )
        original_executable = sys.executable
        try:
            # The sealed parent contract records the lexical `.venv/bin/python`
            # interpreter path.  `uv run python` may expose the same interpreter
            # as `.venv/bin/python3`; normalizing the lexical identity here keeps
            # recursive replay independent of the caller's equivalent symlink.
            sys.executable = str(canonical_python)
            receipt = parent.verify_published_receipt()
        except parent.BundleVerificationError as exc:
            raise BundleVerificationError(f"sealed FINAL-v10.5 replay failed: {exc}") from exc
        finally:
            sys.executable = original_executable
    else:
        receipt = paths.parent_validator(paths)
    if not isinstance(receipt, dict) or receipt.get("status") != SEALED_STATUS:
        raise BundleVerificationError(
            "parent validator did not return a sealed FINAL-v10.5 receipt"
        )
    sources = receipt.get("authoritative_sources")
    if not isinstance(sources, list) or len(sources) != paths.expected_parent_source_count:
        raise BundleVerificationError("sealed FINAL-v10.5 source census changed")
    source_ids: set[str] = set()
    source_paths: set[Path] = set()
    source_hashes: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or set(source) != _ARTIFACT_KEYS:
            raise BundleVerificationError(f"parent source {index} shape is invalid")
        source_id = source.get("id")
        digest = source.get("sha256")
        if not isinstance(source_id, str) or _SOURCE_ID_RE.fullmatch(source_id) is None:
            raise BundleVerificationError(f"parent source {index} ID is invalid")
        if source_id in source_ids:
            raise BundleVerificationError(f"duplicate parent source ID: {source_id}")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise BundleVerificationError(f"parent source {source_id} hash is invalid")
        if digest in source_hashes:
            raise BundleVerificationError(f"duplicate parent source hash: {digest}")
        source_ids.add(source_id)
        source_hashes.add(digest)
        source_paths.add(_record_identity(source, paths).resolve())
    if len(source_paths) != len(sources):
        raise BundleVerificationError("parent source ledger contains duplicate paths")
    manifest = _load_json(paths.parent_manifest, label="sealed FINAL-v10.5 source manifest")
    if (
        manifest.get("status") != parent.CANDIDATE_STATUS
        or manifest.get("pending_artifacts") != []
        or manifest.get("artifacts") != sources
    ):
        raise BundleVerificationError("FINAL-v10.5 receipt and live manifest disagree")
    if receipt.get("source_manifest") != identity(
        paths.parent_manifest, display_path=_display(paths.parent_manifest, paths.repo)
    ):
        raise BundleVerificationError("FINAL-v10.5 receipt does not bind its live manifest")
    return receipt, sources


def _validate_source_record(
    record: Any,
    *,
    specs: Mapping[str, Mapping[str, Any]],
    pending: bool,
    location: str,
) -> dict[str, Any]:
    expected_keys = _PENDING_KEYS if pending else _ARTIFACT_KEYS
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise BundleVerificationError(f"{location} keys are not exact")
    source_id = record.get("id")
    if source_id not in specs:
        raise BundleVerificationError(f"{location} has unexpected source ID: {source_id}")
    expected = {"id": source_id, **specs[str(source_id)]}
    for key, value in expected.items():
        if record.get(key) != value:
            raise BundleVerificationError(f"{location}.{key} differs from frozen metadata")
    if not pending and (
        isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or int(record["size_bytes"]) < 0
        or not isinstance(record.get("sha256"), str)
        or _SHA256_RE.fullmatch(str(record["sha256"])) is None
    ):
        raise BundleVerificationError(f"{location} byte identity is invalid")
    return record


def _manifest_identity_record(path: Path, paths: BundlePaths) -> dict[str, Any]:
    return identity(path, display_path=_display(path, paths.repo))


def _validate_manifest(
    paths: BundlePaths, *, require_candidate: bool
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_final_downstream_code_pins(paths)
    parent_receipt, parent_sources = _verify_parent(paths)
    manifest_path = paths.final_v11 / SOURCE_MANIFEST_NAME
    manifest = _load_json(manifest_path, label="FINAL-v11 source manifest")
    flat = set(manifest) == _FLAT_KEYS
    if not flat and set(manifest) != _DRAFT_KEYS:
        raise BundleVerificationError("source manifest shape is neither exact draft nor exact flat")
    if manifest.get("bundle") != "final_v11" or manifest.get("schema_version") not in {1, 2}:
        raise BundleVerificationError("source manifest header is invalid")
    if flat and manifest.get("schema_version") != 2:
        raise BundleVerificationError("flat candidate manifest must use schema version 2")
    if not flat:
        if manifest["base_bundle_receipt"] != _manifest_identity_record(
            paths.parent_receipt, paths
        ):
            raise BundleVerificationError("draft parent receipt identity drift")
        if manifest["base_source_manifest"] != _manifest_identity_record(
            paths.parent_manifest, paths
        ):
            raise BundleVerificationError("draft parent manifest identity drift")
        if parent_receipt.get("authoritative_sources") != parent_sources:
            raise BundleVerificationError("parent replay returned inconsistent sources")

    artifacts = manifest.get("artifacts")
    pending = manifest.get("pending_artifacts")
    if not isinstance(artifacts, list) or not isinstance(pending, list):
        raise BundleVerificationError("manifest artifact ledgers must be lists")
    specs = _source_specs(paths)
    if len(specs) != EXPECTED_NEW_SOURCE_COUNT:
        raise BundleVerificationError("internal FINAL-v11 source roster is not exact")

    parent_by_id = {str(item["id"]): item for item in parent_sources}
    if flat:
        by_id = {str(item.get("id")): item for item in artifacts if isinstance(item, dict)}
        if len(by_id) != len(artifacts):
            raise BundleVerificationError("flat manifest has duplicate or invalid source IDs")
        if set(by_id) != set(parent_by_id) | set(specs):
            raise BundleVerificationError("flat manifest source ID roster is not exact")
        for source_id, inherited in parent_by_id.items():
            if by_id[source_id] != inherited:
                raise BundleVerificationError(f"inherited source record drift: {source_id}")
        new_sources = [
            _validate_source_record(
                by_id[source_id], specs=specs, pending=False, location=f"artifacts[{source_id}]"
            )
            for source_id in specs
        ]
    else:
        new_sources = [
            _validate_source_record(
                item, specs=specs, pending=False, location=f"artifacts[{index}]"
            )
            for index, item in enumerate(artifacts)
        ]
        pending = [
            _validate_source_record(
                item, specs=specs, pending=True, location=f"pending_artifacts[{index}]"
            )
            for index, item in enumerate(pending)
        ]
        observed = [str(item["id"]) for item in [*new_sources, *pending]]
        if len(observed) != len(set(observed)) or set(observed) != set(specs):
            raise BundleVerificationError("draft extension source roster is not exact")

    for source in new_sources:
        _record_identity(source, paths)
    all_records = [*parent_sources, *new_sources, *pending]
    all_paths = [_lexical_path(str(item["path"]), paths).absolute() for item in all_records]
    if len(all_paths) != len(set(all_paths)):
        raise BundleVerificationError("source graph declares a path more than once")
    complete_records = [*parent_sources, *new_sources]
    complete_hashes = [str(item["sha256"]) for item in complete_records]
    if len(complete_hashes) != len(set(complete_hashes)):
        raise BundleVerificationError("source graph SHA table is not one-to-one")

    if flat:
        if pending or len(artifacts) != EXPECTED_COMPLETE_SOURCE_COUNT:
            raise BundleVerificationError(
                f"flat candidate must have {EXPECTED_COMPLETE_SOURCE_COUNT} sources and zero pending"
            )
        if manifest.get("status") != CANDIDATE_STATUS:
            raise BundleVerificationError("flat complete manifest is not candidate-ready")
    else:
        if manifest.get("status") != DRAFT_STATUS:
            raise BundleVerificationError("layered manifest must remain draft")
        if not pending:
            raise BundleVerificationError("complete layered manifest must flatten atomically")
        if require_candidate:
            raise BundleVerificationError(
                f"FINAL-v11 candidate is blocked by {len(pending)} pending sources"
            )
    return manifest, parent_sources, new_sources, pending


def _source_map(sources: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(source["id"]): source for source in sources}


def _source_json(
    source_id: str, sources: Sequence[Mapping[str, Any]], paths: BundlePaths
) -> dict[str, Any]:
    by_id = _source_map(sources)
    if source_id not in by_id:
        raise BundleVerificationError(f"required source is absent: {source_id}")
    path = _record_identity(by_id[source_id], paths)
    return _load_json(path, label=source_id)


def _nested_find(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys:
                return child
        for child in value.values():
            found = _nested_find(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _nested_find(child, keys)
            if found is not None:
                return found
    return None


def _require_int(value: Any, expected: int, *, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise BundleVerificationError(f"{context} must equal {expected}, observed {value!r}")


def _validate_identity_reference(
    observed: Any,
    source_id: str,
    sources: Sequence[Mapping[str, Any]],
    *,
    context: str,
    paths: BundlePaths | None = None,
) -> None:
    selected = default_paths() if paths is None else paths
    expected = _source_map(sources).get(source_id)
    if expected is None:
        raise BundleVerificationError(f"{context} refers to absent source {source_id}")
    if not isinstance(observed, dict) or set(observed) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise BundleVerificationError(f"{context} identity schema is not exact")
    if observed["size_bytes"] != expected["size_bytes"] or observed["sha256"] != expected["sha256"]:
        raise BundleVerificationError(f"{context} identity does not bind {source_id}")
    observed_raw = observed["path"]
    expected_raw = expected["path"]
    if not isinstance(observed_raw, str) or not observed_raw:
        raise BundleVerificationError(f"{context} identity path is invalid")
    if not isinstance(expected_raw, str) or not expected_raw:
        raise BundleVerificationError(f"{context} source path is invalid")
    observed_path = _lexical_path(observed_raw, selected).absolute()
    expected_path = _lexical_path(expected_raw, selected).absolute()
    _reject_symlink_chain(observed_path, context=f"{context} identity path")
    if observed_path != expected_path:
        raise BundleVerificationError(f"{context} identity path does not bind {source_id}")


def _validate_campaign(
    paths: BundlePaths,
    new_sources: list[dict[str, Any]],
    *,
    deep_replay: bool = True,
) -> None:
    if paths.campaign_validator is not None:
        result = paths.campaign_validator(paths)
        if not isinstance(result, dict) or result.get("status") != "PASS":
            raise BundleVerificationError("campaign callback did not return PASS")
        return
    terminal_id = "aim1-tcga-surgen-two-encoder-recovery-v2-training-completion"
    terminal = _source_json(terminal_id, new_sources, paths)
    if terminal.get("status") != "complete_and_certified_via_scoped_census_erratum":
        raise BundleVerificationError("scoped training terminal is not complete")
    expected_accounting = {
        "adopted_oof_fits": 25,
        "hidden_fits": 0,
        "new_fits": EXPECTED_NEW_FITS,
        "new_oof_fits": 25,
        "new_refits": 10,
        "operational_lineage_fits": EXPECTED_CAMPAIGN_LINEAGE_FITS,
        "physical_new_fits": EXPECTED_NEW_FITS,
        "recovery_new_fits": 0,
    }
    if terminal.get("fit_accounting") != expected_accounting:
        raise BundleVerificationError(
            f"training fit accounting drift: {terminal.get('fit_accounting')!r}"
        )
    expected_execution = {
        "attempts_per_job": 1,
        "job_count": 10,
        "retries": 0,
        "total_attempts": 10,
    }
    if terminal.get("execution_accounting") != expected_execution:
        raise BundleVerificationError("training execution accounting drift")
    concurrency = terminal.get("concurrency")
    if (
        not isinstance(concurrency, dict)
        or set(concurrency) != {"maximum", "observed_peak", "witness_utc"}
        or concurrency.get("maximum") != EXPECTED_MAX_CONCURRENCY
        or concurrency.get("observed_peak") != EXPECTED_MAX_CONCURRENCY
    ):
        raise BundleVerificationError("training concurrency must be exact maximum/peak six")
    _parse_utc(concurrency["witness_utc"], context="training concurrency witness")
    _validate_identity_reference(
        terminal.get("predecessor_v1_terminal"),
        "aim1-tcga-surgen-two-encoder-recovery-v1-training-completion",
        new_sources,
        context="training_v2.predecessor_v1_terminal",
        paths=paths,
    )
    _validate_identity_reference(
        terminal.get("scope_contract"),
        "aim1-tcga-surgen-two-encoder-recovery-v2-scope-contract",
        new_sources,
        context="training_v2.scope_contract",
        paths=paths,
    )
    _validate_identity_reference(
        terminal.get("scope_adjudication"),
        "aim1-tcga-surgen-two-encoder-recovery-v2-scope-adjudication",
        new_sources,
        context="training_v2.scope_adjudication",
        paths=paths,
    )
    implementation = terminal.get("recovery_implementation")
    if not isinstance(implementation, dict) or set(implementation) != {
        "controller",
        "controller_test",
    }:
        raise BundleVerificationError("training_v2 implementation binding is invalid")
    for key, source_id in (
        ("controller", "aim1-tcga-surgen-two-encoder-recovery-v2-controller"),
        ("controller_test", "aim1-tcga-surgen-two-encoder-recovery-v2-test"),
    ):
        _validate_identity_reference(
            implementation[key],
            source_id,
            new_sources,
            context=f"training_v2.{key}",
            paths=paths,
        )
    scoped = terminal.get("training_scoped_census")
    expected_scoped = {
        "artifact_count": 441,
        "total_size_bytes": 989_858_771,
        "tree_sha256": "a2ffd5b61eaf65dc8d0c5df6a1b29867ffabc57f0af869120603ea37591ac261",
    }
    if not isinstance(scoped, dict) or any(
        scoped.get(key) != value for key, value in expected_scoped.items()
    ):
        raise BundleVerificationError("immutable 441-file training census drift")
    _validate_identity_reference(
        scoped.get("original_census_source"),
        "aim1-tcga-surgen-two-encoder-recovery-v1-adjudication",
        new_sources,
        context="training_v2.original_census_source",
        paths=paths,
    )
    if (
        scoped.get("all_original_records_rehashed_at_certification") is not True
        or scoped.get("closed_roster_outside_exclusions") is not True
    ):
        raise BundleVerificationError("immutable training census was not fail-closed")
    policy = terminal.get("namespace_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("delegated_growth_namespaces") != ["downstream_v2/"]
        or policy.get("root_analysis_namespace_authorized") is not False
        or policy.get("prepared_baseline_namespace")
        != "downstream/ (seven named bytes remain immutable)"
    ):
        raise BundleVerificationError("training-v2 delegated namespace policy drift")
    baseline = terminal.get("prepared_downstream_baseline")
    legacy_ids = (
        "aim1-tcga-surgen-two-encoder-legacy-prepare-contract",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-cptac-primary",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-orion-cpht",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-rih-metastatic",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-rih-primary",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-sr1482-metastatic",
        "aim1-tcga-surgen-two-encoder-legacy-prepare-score-job-plan",
    )
    if (
        not isinstance(baseline, dict)
        or baseline.get("artifact_count") != 7
        or baseline.get("total_size_bytes") != 95_802
        or baseline.get("tree_sha256")
        != "d6400d6004ad2c1a850c022db1cb694f08b0fac445cdbbc3d94400f1fc995621"
        or baseline.get("artifacts")
        != [
            {
                key: _source_map(new_sources)[source_id][key]
                for key in ("path", "size_bytes", "sha256")
            }
            for source_id in legacy_ids
        ]
    ):
        raise BundleVerificationError("immutable seven-file downstream baseline drift")

    predecessor = _source_json(
        "aim1-tcga-surgen-two-encoder-recovery-v1-training-completion",
        new_sources,
        paths,
    )
    if predecessor.get("status") != "complete_and_certified_via_bounded_erratum":
        raise BundleVerificationError("bounded recovery-v1 terminal is not complete")
    if predecessor.get("fit_accounting") != expected_accounting:
        raise BundleVerificationError("recovery-v1 fit accounting drift")
    if predecessor.get("execution_accounting") != expected_execution:
        raise BundleVerificationError("recovery-v1 execution accounting drift")
    if predecessor.get("seeds") != list(EXPECTED_SEEDS) or predecessor.get("encoders") != list(
        EXPECTED_TRAINING_ENCODERS
    ):
        raise BundleVerificationError("training seed or encoder roster drift")
    _require_int(predecessor.get("job_count"), 10, context="training.job_count")
    population = predecessor.get("source_population")
    if not isinstance(population, dict):
        raise BundleVerificationError("training source population is invalid")
    expected_population = {
        "slides": 1389,
        "patients": 1239,
        "mutant_patients": 501,
        "wildtype_patients": 738,
    }
    for key, expected in expected_population.items():
        _require_int(population.get(key), expected, context=f"source_population.{key}")
    _validate_identity_reference(
        predecessor.get("base_contract"),
        "aim1-tcga-surgen-two-encoder-campaign-contract",
        new_sources,
        context="training_v1.base_contract",
        paths=paths,
    )
    _validate_identity_reference(
        predecessor.get("base_preflight"),
        "aim1-tcga-surgen-two-encoder-deep-preflight",
        new_sources,
        context="training_v1.base_preflight",
        paths=paths,
    )
    for terminal_key, source_id in (
        ("erratum_contract", "aim1-tcga-surgen-two-encoder-recovery-v1-erratum-contract"),
        ("validator_adjudication", "aim1-tcga-surgen-two-encoder-recovery-v1-adjudication"),
        ("scheduler_recovery", "aim1-tcga-surgen-two-encoder-recovery-v1-scheduler"),
    ):
        _validate_identity_reference(
            predecessor.get(terminal_key),
            source_id,
            new_sources,
            context=f"training_v1.{terminal_key}",
            paths=paths,
        )
    v1_implementation = predecessor.get("recovery_implementation")
    if not isinstance(v1_implementation, dict):
        raise BundleVerificationError("training_v1 implementation binding is invalid")
    for key, source_id in (
        ("controller", "aim1-tcga-surgen-two-encoder-recovery-v1-controller"),
        ("controller_test", "aim1-tcga-surgen-two-encoder-recovery-v1-test"),
    ):
        _validate_identity_reference(
            v1_implementation.get(key),
            source_id,
            new_sources,
            context=f"training_v1.{key}",
            paths=paths,
        )
    if not deep_replay:
        return
    controller = _record_identity(
        _source_map(new_sources)["aim1-tcga-surgen-two-encoder-recovery-v2-controller"],
        paths,
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("_final_v11_training_replay", controller)
    if spec is None or spec.loader is None:
        raise BundleVerificationError("could not load governed training controller")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        replayed = module.validate_scoped_terminal(paths.campaign_root, deep_scope=True)
        if replayed != terminal:
            raise BundleVerificationError("scoped recovery-v2 replay returned terminal drift")
    except Exception as exc:
        raise BundleVerificationError("governed training deep replay failed") from exc


def _parse_utc(value: Any, *, context: str) -> dt.datetime:
    if not isinstance(value, str):
        raise BundleVerificationError(f"{context} is not a timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise BundleVerificationError(f"{context} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise BundleVerificationError(f"{context} must be UTC")
    return parsed


def _validate_score_artifacts(seal: dict[str, Any], paths: BundlePaths) -> None:
    records = seal.get("score_artifacts")
    if not isinstance(records, list) or len(records) != EXPECTED_SCORE_JOBS:
        raise BundleVerificationError("inference seal must bind exactly 50 score artifacts")
    expected_jobs = [
        (
            f"final_v11.continuation_v3.score.{encoder}.{target}.seed{seed}",
            encoder,
            target,
            seed,
        )
        for encoder in EXPECTED_ENCODERS
        for target in EXPECTED_TARGETS
        for seed in EXPECTED_SEEDS
    ]
    observed_jobs: list[tuple[str, str, str, int]] = []
    total_rows = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise BundleVerificationError(f"score artifact {index} is invalid")
        required = {"job_id", "score", "receipt", "rows"}
        if set(record) != required:
            raise BundleVerificationError(f"score artifact {index} schema is not exact")
        job_id = record["job_id"]
        if not isinstance(job_id, str):
            raise BundleVerificationError(f"score artifact {index} has invalid job_id")
        match = re.fullmatch(
            r"final_v11\.continuation_v3\.score\.(univ1|virchow2_cls)\."
            r"(cptac_primary|rih_primary|rih_metastatic|sr1482_metastatic|orion_cpht)\."
            r"seed(42|43|44|45|46)",
            job_id,
        )
        if match is None:
            raise BundleVerificationError(f"score artifact {job_id} job identity is invalid")
        encoder, target, seed_text = match.groups()
        seed = int(seed_text)
        observed_jobs.append((job_id, encoder, target, seed))
        rows = record["rows"]
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise BundleVerificationError(f"score artifact {job_id} row count is invalid")
        total_rows += rows
        score_identity = record["score"]
        receipt_identity = record["receipt"]
        if not isinstance(score_identity, dict) or set(score_identity) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise BundleVerificationError(f"score artifact identity schema drift: {job_id}")
        if not isinstance(receipt_identity, dict) or set(receipt_identity) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise BundleVerificationError(f"score receipt identity schema drift: {job_id}")
        path = _lexical_path(str(score_identity["path"]), paths)
        actual = identity(path, display_path=str(score_identity["path"]))
        if actual != score_identity:
            raise BundleVerificationError(f"score artifact identity drift: {job_id}")
        if path.suffix.lower() not in {".parquet", ".pq"}:
            raise BundleVerificationError(f"score artifact is not Parquet: {path}")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - production dependency
            raise BundleVerificationError("pyarrow is required to inspect score Parquet") from exc
        parquet = pq.ParquetFile(path)
        columns = list(parquet.schema_arrow.names)
        observed_rows = int(parquet.metadata.num_rows)
        if observed_rows != rows:
            raise BundleVerificationError(f"score artifact row-count drift: {job_id}")
        lowered = {str(column).strip().lower() for column in columns}
        if lowered & _FORBIDDEN_SCORE_COLUMNS:
            raise BundleVerificationError(f"target outcomes leaked into score artifact: {job_id}")
        if columns != ["slide_id", "seed", "fold", "logit"]:
            raise BundleVerificationError(f"score artifact schema drift: {job_id}")
        receipt_path = _lexical_path(str(receipt_identity["path"]), paths)
        receipt_actual = identity(receipt_path, display_path=str(receipt_identity["path"]))
        if receipt_actual != receipt_identity:
            raise BundleVerificationError(f"score receipt identity drift: {job_id}")
        receipt_payload = _load_json(receipt_path, label=f"score receipt {job_id}")
        receipt_expected = {
            "status": "complete",
            "contains_target_outcomes": False,
            "encoder": encoder,
            "target": target,
            "seed": seed,
            "artifact": score_identity,
            "n_rows": rows,
        }
        for key, expected in receipt_expected.items():
            if receipt_payload.get(key) != expected:
                raise BundleVerificationError(f"score receipt {job_id}.{key} drift")
    if observed_jobs != expected_jobs:
        raise BundleVerificationError("score artifacts are not in exact contracted order")
    if total_rows != EXPECTED_SCORE_SLIDE_ROWS:
        raise BundleVerificationError(
            f"score artifacts contain {total_rows} rows, expected {EXPECTED_SCORE_SLIDE_ROWS}"
        )


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleVerificationError(f"{context} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BundleVerificationError(f"{context} is non-finite")
    return result


def _metric_cell(point: Any, interval: Any, *, context: str) -> str:
    value = _finite_float(point, context=f"{context}.point")
    if not isinstance(interval, list) or len(interval) != 2:
        raise BundleVerificationError(f"{context}.interval is invalid")
    low = _finite_float(interval[0], context=f"{context}.low")
    high = _finite_float(interval[1], context=f"{context}.high")
    if low > value or value > high:
        raise BundleVerificationError(f"{context} point is outside interval")
    return f"{value:.4f} [{low:.4f}, {high:.4f}]"


def _validate_claim_rows(results: dict[str, Any], results_text: str) -> None:
    rows = results.get("report_claim_rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_PERFORMANCE_CLAIM_ROW_IDS):
        raise BundleVerificationError(
            "analysis results must contain exactly 78 performance report_claim_rows"
        )
    seen: set[str] = set()
    for index, row in enumerate(rows):
        required = {
            "row_id",
            "section",
            "training_population",
            "evaluation_population",
            "encoder",
            "n_patients",
            "n_mutant",
            "auroc",
            "auroc_ci95",
            "auprc",
            "auprc_ci95",
            "evidence_state",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise BundleVerificationError(f"report_claim_rows[{index}] schema is not exact")
        row_id = row["row_id"]
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            raise BundleVerificationError(f"report claim row ID is invalid: {row_id!r}")
        seen.add(row_id)
        if results_text.count(row_id) != 1:
            raise BundleVerificationError(
                f"claim row ID {row_id} must occur exactly once in Results.md"
            )
        expected_cells = [
            str(row_id),
            str(row["training_population"]),
            str(row["evaluation_population"]),
            str(row["encoder"]),
            str(row["n_patients"]),
            str(row["n_mutant"]),
            _metric_cell(row["auroc"], row["auroc_ci95"], context=f"claim {row_id} AUROC"),
            _metric_cell(row["auprc"], row["auprc_ci95"], context=f"claim {row_id} AUPRC"),
            str(row["evidence_state"]),
        ]
        expected_line = "| " + " | ".join(expected_cells) + " |"
        if results_text.count(expected_line) != 1:
            raise BundleVerificationError(
                f"claim row {row_id} is not bound exactly once to governed results"
            )
    if seen != set(EXPECTED_PERFORMANCE_CLAIM_ROW_IDS):
        raise BundleVerificationError("report claim-row ID roster is not exact")


def _contrast_metric_summary(metrics: Any, *, context: str) -> str:
    if not isinstance(metrics, dict) or not metrics:
        raise BundleVerificationError(f"{context} metrics are missing")
    required = {
        "reference",
        "reference_ci95",
        "comparison",
        "comparison_ci95",
        "delta_comparison_minus_reference",
        "delta_ci95",
        "lower_is_better",
    }
    cells: list[str] = []
    for metric_name in sorted(metrics):
        metric = metrics[metric_name]
        if not isinstance(metric, dict) or set(metric) != required:
            raise BundleVerificationError(f"{context}/{metric_name} schema is not exact")
        if not isinstance(metric["lower_is_better"], bool):
            raise BundleVerificationError(f"{context}/{metric_name}.lower_is_better is not boolean")
        reference = _metric_cell(
            metric["reference"],
            metric["reference_ci95"],
            context=f"{context}/{metric_name}.reference",
        )
        comparison = _metric_cell(
            metric["comparison"],
            metric["comparison_ci95"],
            context=f"{context}/{metric_name}.comparison",
        )
        delta = _metric_cell(
            metric["delta_comparison_minus_reference"],
            metric["delta_ci95"],
            context=f"{context}/{metric_name}.delta",
        )
        lower = str(metric["lower_is_better"]).lower()
        cells.append(
            f"{metric_name}: reference {reference}; comparison {comparison}; "
            f"delta(comparison-reference) {delta}; lower_is_better={lower}"
        )
    return "; ".join(cells)


def _validate_contrast_rows(results: dict[str, Any], results_text: str) -> None:
    rows = results.get("report_contrast_rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_CONTRAST_CLAIM_ROW_IDS):
        raise BundleVerificationError(
            "analysis results must contain exactly 48 report_contrast_rows"
        )
    required = {
        "row_id",
        "section",
        "training_population",
        "encoder",
        "reference",
        "comparison",
        "contrast_definition",
        "metrics",
        "evidence_state",
    }
    population_required = {"name", "n_patients", "n_mutant"}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise BundleVerificationError(f"report_contrast_rows[{index}] schema is not exact")
        row_id = row["row_id"]
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            raise BundleVerificationError(f"report contrast row ID is invalid: {row_id!r}")
        seen.add(row_id)
        if results_text.count(row_id) != 1:
            raise BundleVerificationError(
                f"contrast row ID {row_id} must occur exactly once in Results.md"
            )
        if row["contrast_definition"] != "comparison minus reference":
            raise BundleVerificationError(f"contrast definition drifted: {row_id}")
        reference = row["reference"]
        comparison = row["comparison"]
        if (
            not isinstance(reference, dict)
            or set(reference) != population_required
            or not isinstance(comparison, dict)
            or set(comparison) != population_required
        ):
            raise BundleVerificationError(f"contrast population schema drifted: {row_id}")
        for arm_name, arm in (("reference", reference), ("comparison", comparison)):
            if (
                not isinstance(arm["name"], str)
                or not arm["name"]
                or isinstance(arm["n_patients"], bool)
                or not isinstance(arm["n_patients"], int)
                or arm["n_patients"] < 1
                or isinstance(arm["n_mutant"], bool)
                or not isinstance(arm["n_mutant"], int)
                or not 0 < arm["n_mutant"] < arm["n_patients"]
            ):
                raise BundleVerificationError(f"contrast {row_id} {arm_name} census is invalid")
        metric_summary = _contrast_metric_summary(row["metrics"], context=f"contrast {row_id}")
        expected_line = (
            "| "
            + " | ".join(
                [
                    row_id,
                    str(row["training_population"]),
                    str(row["encoder"]),
                    str(reference["name"]),
                    str(reference["n_patients"]),
                    str(reference["n_mutant"]),
                    str(comparison["name"]),
                    str(comparison["n_patients"]),
                    str(comparison["n_mutant"]),
                    metric_summary,
                    str(row["evidence_state"]),
                ]
            )
            + " |"
        )
        if results_text.count(expected_line) != 1:
            raise BundleVerificationError(
                f"contrast row {row_id} is not bound exactly once to governed results"
            )
    if seen != set(EXPECTED_CONTRAST_CLAIM_ROW_IDS):
        raise BundleVerificationError("report contrast-row ID roster is not exact")


def _canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BundleVerificationError("claim evidence is not canonical finite JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _md_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _why_d_human_lines(record: dict[str, Any]) -> list[str]:
    """Render the material Why-D A/B/C/D estimates in deterministic table rows."""

    encoder = _md_cell(record["encoder"])
    lines: list[str] = []
    analysis_a = record["analysis_a_random_restriction"]
    required_a = {
        "auc_A_complete",
        "auc_D",
        "observed_delta",
        "n_pos_D",
        "n_neg_D",
        "random",
        "stratified",
    }
    if not isinstance(analysis_a, dict) or set(analysis_a) != required_a:
        raise BundleVerificationError(f"Why-D {encoder} analysis A schema is not exact")
    for key in ("n_pos_D", "n_neg_D"):
        value = analysis_a[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BundleVerificationError(f"Why-D {encoder} analysis A {key} is invalid")
    a_summary = [
        f"A_complete AUROC {_finite_float(analysis_a['auc_A_complete'], context='Why-D A AUROC'):.4f}",
        f"D AUROC {_finite_float(analysis_a['auc_D'], context='Why-D D AUROC'):.4f}",
        "observed D-minus-A "
        f"{_finite_float(analysis_a['observed_delta'], context='Why-D observed delta'):+.4f}",
    ]
    random_required = {"mean", "sd", "p2.5", "p97.5", "p_ge_observed", "p_note"}
    for restriction in ("random", "stratified"):
        block = analysis_a[restriction]
        if not isinstance(block, dict) or set(block) != random_required:
            raise BundleVerificationError(
                f"Why-D {encoder} analysis A {restriction} schema is not exact"
            )
        mean = _finite_float(block["mean"], context=f"Why-D A {restriction}.mean")
        low = _finite_float(block["p2.5"], context=f"Why-D A {restriction}.low")
        high = _finite_float(block["p97.5"], context=f"Why-D A {restriction}.high")
        p_value = _finite_float(block["p_ge_observed"], context=f"Why-D A {restriction}.p")
        sd = _finite_float(block["sd"], context=f"Why-D A {restriction}.sd")
        if low > mean or mean > high or sd < 0 or not 0 <= p_value <= 1:
            raise BundleVerificationError(
                f"Why-D {encoder} analysis A {restriction} values are invalid"
            )
        a_summary.append(
            f"{restriction} mean {mean:+.4f} [{low:+.4f}, {high:+.4f}], "
            f"SD {sd:.4f}, p>={p_value:.6g}"
        )
    lines.append(
        "| "
        + " | ".join(
            [
                "Why-D A",
                encoder,
                str(record["n_patients"]),
                str(record["n_mutant"]),
                str(analysis_a["n_pos_D"]),
                str(analysis_a["n_neg_D"]),
                "; ".join(a_summary),
            ]
        )
        + " |"
    )

    analysis_b = record["analysis_b_pairwise_auc_decomposition"]
    expected_b = {"D+ vs D-", "D+ vs C-", "C+ vs D-", "C+ vs C-"}
    if not isinstance(analysis_b, dict) or set(analysis_b) != expected_b:
        raise BundleVerificationError(f"Why-D {encoder} analysis B roster is not exact")
    required_b = {"auc", "n_pos", "n_neg", "pair_share", "ci"}
    for comparison in sorted(analysis_b):
        block = analysis_b[comparison]
        if not isinstance(block, dict) or set(block) != required_b:
            raise BundleVerificationError(
                f"Why-D {encoder} analysis B/{comparison} schema is not exact"
            )
        n_pos = block["n_pos"]
        n_neg = block["n_neg"]
        if (
            isinstance(n_pos, bool)
            or not isinstance(n_pos, int)
            or n_pos < 1
            or isinstance(n_neg, bool)
            or not isinstance(n_neg, int)
            or n_neg < 1
        ):
            raise BundleVerificationError(
                f"Why-D {encoder} analysis B/{comparison} census is invalid"
            )
        share = _finite_float(block["pair_share"], context=f"Why-D B/{comparison}.pair_share")
        if not 0 <= share <= 1:
            raise BundleVerificationError(
                f"Why-D {encoder} analysis B/{comparison} pair share is invalid"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    "Why-D B",
                    encoder,
                    _md_cell(comparison),
                    str(n_pos),
                    str(n_neg),
                    _metric_cell(block["auc"], block["ci"], context=f"Why-D B/{comparison} AUROC"),
                    f"{share:.6f}",
                ]
            )
            + " |"
        )

    analysis_c = record["analysis_c_molecular_score_distributions"]
    if not isinstance(analysis_c, dict) or set(analysis_c) != {"KRAS-WT", "KRAS-mutant"}:
        raise BundleVerificationError(f"Why-D {encoder} analysis C roster is not exact")
    required_c = {
        "n",
        "mean_logit",
        "mean_ci",
        "median_logit",
        "median_ci",
        "mean_prob",
    }
    for kras_group in ("KRAS-WT", "KRAS-mutant"):
        contexts = analysis_c[kras_group]
        if not isinstance(contexts, dict) or not contexts:
            raise BundleVerificationError(f"Why-D {encoder} analysis C/{kras_group} is empty")
        for context in sorted(contexts):
            block = contexts[context]
            if not isinstance(block, dict) or set(block) != required_c:
                raise BundleVerificationError(
                    f"Why-D {encoder} analysis C/{kras_group}/{context} schema is not exact"
                )
            n_patients = block["n"]
            if isinstance(n_patients, bool) or not isinstance(n_patients, int) or n_patients < 1:
                raise BundleVerificationError(
                    f"Why-D {encoder} analysis C/{kras_group}/{context} census is invalid"
                )
            probability = _finite_float(
                block["mean_prob"], context=f"Why-D C/{kras_group}/{context}.mean_prob"
            )
            if not 0 <= probability <= 1:
                raise BundleVerificationError(
                    f"Why-D {encoder} analysis C/{kras_group}/{context} probability is invalid"
                )
            lines.append(
                "| "
                + " | ".join(
                    [
                        "Why-D C",
                        encoder,
                        kras_group,
                        _md_cell(context),
                        str(n_patients),
                        _metric_cell(
                            block["mean_logit"],
                            block["mean_ci"],
                            context=f"Why-D C/{kras_group}/{context}.mean",
                        ),
                        _metric_cell(
                            block["median_logit"],
                            block["median_ci"],
                            context=f"Why-D C/{kras_group}/{context}.median",
                        ),
                        f"{probability:.4f}",
                    ]
                )
                + " |"
            )

    analysis_d = record["analysis_d_adjusted_molecular_association"]
    material_terms = ("BRAF_mut", "MSI", "BRAF_mut x MSI")
    required_d = {"beta", "ci", "excludes_zero"}
    if not isinstance(analysis_d, dict) or not set(material_terms) <= set(analysis_d):
        raise BundleVerificationError(f"Why-D {encoder} analysis D roster is incomplete")
    for term in material_terms:
        block = analysis_d[term]
        if (
            not isinstance(block, dict)
            or set(block) != required_d
            or not isinstance(block["excludes_zero"], bool)
        ):
            raise BundleVerificationError(f"Why-D {encoder} analysis D/{term} schema is not exact")
        lines.append(
            "| "
            + " | ".join(
                [
                    "Why-D D",
                    encoder,
                    _md_cell(term),
                    _metric_cell(block["beta"], block["ci"], context=f"Why-D D/{term}.beta"),
                    str(block["excludes_zero"]).lower(),
                ]
            )
            + " |"
        )
    return lines


def _validate_why_d_evidence_records(results: dict[str, Any], results_text: str) -> None:
    records = results.get("why_d_evidence_records")
    if not isinstance(records, list) or len(records) != len(EXPECTED_WHY_D_RECORD_IDS):
        raise BundleVerificationError("analysis results must contain two Why-D evidence records")
    required = {
        "record_id",
        "section",
        "training_population",
        "evaluation_population",
        "encoder",
        "n_patients",
        "n_mutant",
        "random_restriction_draws",
        "patient_bootstrap_draws",
        "analysis_a_random_restriction",
        "analysis_b_pairwise_auc_decomposition",
        "analysis_c_molecular_score_distributions",
        "analysis_d_adjusted_molecular_association",
        "result_path",
        "evidence_state",
    }
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != required:
            raise BundleVerificationError(f"why_d_evidence_records[{index}] schema is not exact")
        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise BundleVerificationError(f"Why-D record ID is invalid: {record_id!r}")
        seen.add(record_id)
        if results_text.count(record_id) != 1:
            raise BundleVerificationError(
                f"Why-D record ID {record_id} must occur exactly once in Results.md"
            )
        for key in (
            "n_patients",
            "n_mutant",
            "random_restriction_draws",
            "patient_bootstrap_draws",
        ):
            if isinstance(record[key], bool) or not isinstance(record[key], int):
                raise BundleVerificationError(f"Why-D {record_id}/{key} is not integer")
        expected_line = (
            "| "
            + " | ".join(
                [
                    record_id,
                    str(record["training_population"]),
                    str(record["evaluation_population"]),
                    str(record["encoder"]),
                    str(record["n_patients"]),
                    str(record["n_mutant"]),
                    str(record["random_restriction_draws"]),
                    str(record["patient_bootstrap_draws"]),
                    str(record["result_path"]),
                    _canonical_json_sha256(record),
                    str(record["evidence_state"]),
                ]
            )
            + " |"
        )
        if results_text.count(expected_line) != 1:
            raise BundleVerificationError(
                f"Why-D record {record_id} is not hash-bound exactly once"
            )
        for line in _why_d_human_lines(record):
            if results_text.count(line) != 1:
                raise BundleVerificationError(
                    f"Why-D record {record_id} lacks an exact human-rendered material row"
                )
    if seen != set(EXPECTED_WHY_D_RECORD_IDS):
        raise BundleVerificationError("Why-D evidence-record ID roster is not exact")


def _validate_estimand_boundaries(results: dict[str, Any]) -> None:
    expected = {
        "canonical_e0_evaluation_is_not_source_restricted_training": True,
        "source_restricted_oof_is_not_external_transport": True,
        "zero_shot_targets_absent_from_training_selection": True,
        "model_seeds_are_not_inference_units": True,
        "folds_are_not_inference_units": True,
        "cpht_raw_role": "retrospective_sensitivity",
        "cpht_a_status": "INHERITED_NO_NEW_RUN",
        "cpht_r_status": "NOT_RUN",
        "whole_section_pathology_status": "GENERATED_UNREAD",
    }
    if results.get("estimand_boundaries") != expected:
        raise BundleVerificationError("analysis estimand boundaries are not exact")


def _validate_analysis_contract(contract: dict[str, Any]) -> int:
    required = {
        "schema_version",
        "status",
        "created_utc",
        "experiment",
        "campaign_root",
        "downstream_contract",
        "inference_seal",
        "seal_status_observed_before_outcome_open",
        "outcomes_opened_only_after_seal",
        "outcome_sources",
        "derived_covariate_source",
        "inherited_canonical_e0",
        "analysis",
        "patient_table_rows",
        "report_claim_row_count",
        "report_contrast_row_count",
        "why_d_evidence_record_count",
        "output_inventory",
        "implementation",
    }
    if set(contract) != required:
        raise BundleVerificationError("analysis contract top-level schema is not exact")
    expected_values = {
        "schema_version": 1,
        "status": "analysis_governed_after_inference_seal",
        "experiment": EXPECTED_ANALYSIS_EXPERIMENT,
        "seal_status_observed_before_outcome_open": "sealed_before_outcome_join",
        "outcomes_opened_only_after_seal": True,
        "derived_covariate_source": EXPECTED_DERIVED_COVARIATE_SOURCE,
        "patient_table_rows": EXPECTED_ANALYSIS_PATIENT_ROWS,
        "report_claim_row_count": len(EXPECTED_PERFORMANCE_CLAIM_ROW_IDS),
        "report_contrast_row_count": len(EXPECTED_CONTRAST_CLAIM_ROW_IDS),
        "why_d_evidence_record_count": len(EXPECTED_WHY_D_RECORD_IDS),
        "output_inventory": list(EXPECTED_ANALYSIS_OUTPUT_INVENTORY),
    }
    mismatch = {
        key: {"expected": expected, "observed": contract.get(key)}
        for key, expected in expected_values.items()
        if contract.get(key) != expected
    }
    if mismatch:
        raise BundleVerificationError(f"analysis contract governed metadata drift: {mismatch}")
    analysis = contract.get("analysis")
    required_analysis = {
        "inference_unit",
        "bootstrap_draws",
        "bootstrap_seed",
        "shared_encoder_draws",
        "target_calibration",
        "target_model_selection",
        "test_only_noncanonical_parameters",
        "bootstrap_arrays",
        "mean_validator_erratum",
    }
    if not isinstance(analysis, dict) or set(analysis) != required_analysis:
        raise BundleVerificationError("analysis contract analysis schema is not exact")
    expected_analysis = {
        "inference_unit": "patient",
        "bootstrap_draws": EXPECTED_BOOTSTRAP_DRAWS,
        "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
        "shared_encoder_draws": True,
        "target_calibration": False,
        "target_model_selection": False,
        "test_only_noncanonical_parameters": False,
    }
    if any(analysis.get(key) != value for key, value in expected_analysis.items()):
        raise BundleVerificationError("analysis contract estimand/bootstrap semantics drift")
    if analysis.get("mean_validator_erratum") != EXPECTED_ANALYSIS_MEAN_ERRATUM:
        raise BundleVerificationError("analysis mean-validator erratum contract drift")
    arrays = analysis.get("bootstrap_arrays")
    if not isinstance(arrays, dict) or set(arrays) != {"names", "count", "dtype", "length"}:
        raise BundleVerificationError("analysis bootstrap-array contract schema is not exact")
    names = arrays.get("names")
    count = arrays.get("count")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(names)
        or arrays.get("dtype") != "float64"
        or arrays.get("length") != EXPECTED_BOOTSTRAP_DRAWS
    ):
        raise BundleVerificationError("analysis bootstrap-array contract drift")
    return count


def _validate_analysis_completion(
    completion: dict[str, Any],
    contract: dict[str, Any],
    new_sources: list[dict[str, Any]],
    *,
    paths: BundlePaths | None = None,
) -> None:
    selected = default_paths() if paths is None else paths
    required = {
        "schema_version",
        "status",
        "created_utc",
        "experiment",
        "seal_before_outcomes",
        "artifacts",
        "analysis_artifact_count",
        "patient_rows",
        "bootstrap_array_count",
        "bootstrap_draws",
        "report_claim_row_count",
        "report_contrast_row_count",
        "why_d_evidence_record_count",
        "outcomes_opened_after_inference_seal",
        "target_refits",
        "target_calibrations",
        "target_model_selections",
    }
    if set(completion) != required:
        raise BundleVerificationError("analysis terminal schema is not exact")
    arrays = contract["analysis"]["bootstrap_arrays"]
    expected_values = {
        "schema_version": 1,
        "status": "complete_and_verified",
        "experiment": EXPECTED_ANALYSIS_EXPERIMENT,
        "analysis_artifact_count": EXPECTED_ANALYSIS_ARTIFACTS,
        "patient_rows": EXPECTED_ANALYSIS_PATIENT_ROWS,
        "bootstrap_array_count": arrays["count"],
        "bootstrap_draws": EXPECTED_BOOTSTRAP_DRAWS,
        "report_claim_row_count": len(EXPECTED_PERFORMANCE_CLAIM_ROW_IDS),
        "report_contrast_row_count": len(EXPECTED_CONTRAST_CLAIM_ROW_IDS),
        "why_d_evidence_record_count": len(EXPECTED_WHY_D_RECORD_IDS),
        "outcomes_opened_after_inference_seal": True,
        "target_refits": 0,
        "target_calibrations": 0,
        "target_model_selections": 0,
    }
    mismatch = {
        key: {"expected": expected, "observed": completion.get(key)}
        for key, expected in expected_values.items()
        if completion.get(key) != expected
    }
    if mismatch:
        raise BundleVerificationError(f"analysis terminal census/semantics drift: {mismatch}")
    artifacts = completion.get("artifacts")
    source_ids = {
        "contract": "aim1-tcga-surgen-two-encoder-downstream-v3-analysis-contract",
        "patient_native_logits": (
            "aim1-tcga-surgen-two-encoder-downstream-v3-patient-native-logits"
        ),
        "bootstrap_distributions": "aim1-tcga-surgen-two-encoder-downstream-v3-bootstrap",
        "results": "aim1-tcga-surgen-two-encoder-downstream-v3-results",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != set(source_ids):
        raise BundleVerificationError("analysis terminal artifact roster is not exact")
    for key, source_id in source_ids.items():
        _validate_identity_reference(
            artifacts[key],
            source_id,
            new_sources,
            context=f"analysis.artifacts.{key}",
            paths=selected,
        )


def _validate_analysis(paths: BundlePaths, new_sources: list[dict[str, Any]]) -> None:
    if paths.analysis_validator is not None:
        result = paths.analysis_validator(paths)
        if not isinstance(result, dict) or result.get("status") != "PASS":
            raise BundleVerificationError("analysis callback did not return PASS")
        return
    by_id = _source_map(new_sources)
    for source_id, expected in EXPECTED_FINAL_ANALYSIS_SOURCE_IDENTITIES.items():
        source = by_id.get(source_id)
        if not isinstance(source, Mapping) or any(
            source.get(key) != value for key, value in expected.items()
        ):
            raise BundleVerificationError(f"frozen final analysis byte identity drift: {source_id}")
    seal_id = "aim1-tcga-surgen-two-encoder-downstream-v3-inference-seal"
    seal = _source_json(seal_id, new_sources, paths)
    required_seal_values = {
        "status": "sealed_before_outcome_join",
        "target_outcomes_present": False,
        "target_outcome_files_opened": False,
        "score_artifact_count": EXPECTED_SCORE_JOBS,
        "score_slide_rows": EXPECTED_SCORE_SLIDE_ROWS,
        "encoders": list(EXPECTED_ENCODERS),
        "seeds": list(EXPECTED_SEEDS),
        "targets": list(EXPECTED_TARGETS),
        "max_scoring_concurrency": 6,
    }
    for key, expected in required_seal_values.items():
        if seal.get(key) != expected:
            raise BundleVerificationError(f"inference seal {key} drift: {seal.get(key)!r}")
    sealed_at = _parse_utc(seal.get("created_utc"), context="inference seal created_utc")
    _validate_score_artifacts(seal, paths)

    contract = _source_json(
        "aim1-tcga-surgen-two-encoder-downstream-v3-analysis-contract",
        new_sources,
        paths,
    )
    _validate_analysis_contract(contract)
    _validate_identity_reference(
        contract.get("inference_seal"),
        seal_id,
        new_sources,
        context="analysis contract inference_seal",
        paths=paths,
    )
    analysis_started = _parse_utc(
        contract.get("created_utc"), context="analysis contract created_utc"
    )
    if analysis_started < sealed_at:
        raise BundleVerificationError("analysis contract predates the score seal")

    completion = _source_json(
        "aim1-tcga-surgen-two-encoder-downstream-v3-analysis-completion",
        new_sources,
        paths,
    )
    _validate_analysis_completion(completion, contract, new_sources, paths=paths)
    _validate_identity_reference(
        completion.get("seal_before_outcomes"),
        seal_id,
        new_sources,
        context="analysis.inference_seal",
        paths=paths,
    )
    completion_at = _parse_utc(
        completion.get("created_utc"), context="analysis completion timestamp"
    )
    if completion_at < analysis_started:
        raise BundleVerificationError(
            "target outcomes were not structurally gated after score sealing"
        )
    results = _source_json("aim1-tcga-surgen-two-encoder-downstream-v3-results", new_sources, paths)
    required_results = {
        "schema_version",
        "status",
        "experiment",
        "score_contract",
        "inference",
        "calibration",
        "aim1",
        "aim2",
        "report_claim_rows",
        "report_contrast_rows",
        "why_d_evidence_records",
        "estimand_boundaries",
        "analysis_contract",
    }
    if (
        set(results) != required_results
        or results.get("schema_version") != 1
        or results.get("status") != "complete"
        or results.get("experiment") != EXPECTED_ANALYSIS_EXPERIMENT
    ):
        raise BundleVerificationError("analysis results top-level schema/status drift")
    _validate_identity_reference(
        results.get("analysis_contract"),
        "aim1-tcga-surgen-two-encoder-downstream-v3-analysis-contract",
        new_sources,
        context="analysis results contract",
        paths=paths,
    )
    _validate_estimand_boundaries(results)
    results_text = (paths.final_v11 / "Results.md").read_text(encoding="utf-8")
    _validate_claim_rows(results, results_text)
    _validate_contrast_rows(results, results_text)
    _validate_why_d_evidence_records(results, results_text)
    implementation = _record_identity(
        _source_map(new_sources)[
            "aim1-tcga-surgen-two-encoder-analysis-report-order-erratum-controller"
        ],
        paths,
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("_final_v11_analysis_replay", implementation)
    if spec is None or spec.loader is None:
        raise BundleVerificationError("could not load governed analysis implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise BundleVerificationError("governed analysis implementation import failed") from exc
    replay = module.verify(paths.campaign_root)
    if (
        not isinstance(replay, dict)
        or replay.get("status") != "PASS_WITH_BOUNDED_REPORT_ORDER_VERIFICATION_ERRATUM"
        or replay.get("mean_validator_erratum") != EXPECTED_ANALYSIS_MEAN_ERRATUM
        or replay.get("report_order_erratum") != EXPECTED_REPORT_ORDER_ERRATUM
        or replay.get("campaign_artifact_count") != 111
    ):
        raise BundleVerificationError("governed analysis deep replay did not return PASS")
    inference_ids = {
        "continuation_v3_controller": ("aim1-tcga-surgen-two-encoder-downstream-v3-controller"),
        "continuation_v3_controller_test": ("aim1-tcga-surgen-two-encoder-downstream-v3-test"),
        "continuation_v3_contract": ("aim1-tcga-surgen-two-encoder-downstream-v3-contract"),
        "continuation_v3_score_job_plan": (
            "aim1-tcga-surgen-two-encoder-downstream-v3-score-job-plan"
        ),
        "continuation_v3_deep_preflight": (
            "aim1-tcga-surgen-two-encoder-downstream-v3-deep-preflight"
        ),
        "continuation_v3_scoring_completion": (
            "aim1-tcga-surgen-two-encoder-downstream-v3-scoring-completion"
        ),
        "continuation_v3_inference_environment": (
            "aim1-tcga-surgen-two-encoder-downstream-v3-inference-environment"
        ),
        "continuation_v3_inference_seal": seal_id,
    }
    analysis_ids = {
        "mean_erratum_controller": (
            "aim1-tcga-surgen-two-encoder-analysis-mean-erratum-controller"
        ),
        "mean_erratum_controller_test": ("aim1-tcga-surgen-two-encoder-analysis-mean-erratum-test"),
        "analysis_contract": ("aim1-tcga-surgen-two-encoder-downstream-v3-analysis-contract"),
        "patient_native_logits": (
            "aim1-tcga-surgen-two-encoder-downstream-v3-patient-native-logits"
        ),
        "bootstrap_distributions": ("aim1-tcga-surgen-two-encoder-downstream-v3-bootstrap"),
        "results": ("aim1-tcga-surgen-two-encoder-downstream-v3-results"),
        "analysis_completion": ("aim1-tcga-surgen-two-encoder-downstream-v3-analysis-completion"),
    }
    verification_ids = {
        "report_order_erratum_controller": (
            "aim1-tcga-surgen-two-encoder-analysis-report-order-erratum-controller"
        ),
        "report_order_erratum_controller_test": (
            "aim1-tcga-surgen-two-encoder-analysis-report-order-erratum-test"
        ),
    }
    if replay.get("direct_source_graph_count") != (
        len(inference_ids) + len(analysis_ids) + len(verification_ids)
    ):
        raise BundleVerificationError("mean-erratum direct source graph count drift")
    for graph_name, graph_ids in (
        ("inference_source_graph", inference_ids),
        ("analysis_source_graph", analysis_ids),
        ("verification_erratum_source_graph", verification_ids),
    ):
        source_graph = replay.get(graph_name)
        if not isinstance(source_graph, dict) or set(source_graph) != set(graph_ids):
            raise BundleVerificationError(f"mean-erratum {graph_name} roster drift")
        for graph_key, source_id in graph_ids.items():
            _validate_identity_reference(
                source_graph[graph_key],
                source_id,
                new_sources,
                context=f"mean_erratum.{graph_name}.{graph_key}",
                paths=paths,
            )


def _document_text(paths: BundlePaths, name: str) -> str:
    path = paths.final_v11 / name
    _reject_symlink_chain(path, context=f"FINAL-v11 {name}")
    if not path.is_file():
        raise BundleVerificationError(f"FINAL-v11 document is missing: {name}")
    return path.read_text(encoding="utf-8")


_REQUIRED_RESULT_HEADINGS = (
    "## Status, terminology, and reading rules",
    "## Integrated conclusions",
    "## Complete experiment-state ledger",
    "## Aim 1",
    "### E0 — performance by training/evaluation population",
    "#### Canonical all-primary OOF model: evaluation-population breakdown",
    "#### Source-restricted OOF training",
    "#### Encoder-paired contrasts",
    "### E1a — controlled challenge populations",
    "### E1a-S — acquisition/composition standardization",
    "### Why-D",
    "### E1d — Clinical Improvements over routine clinical variables",
    "## Aim 2",
    "## Aim 3",
    "## Aim 4",
    "## Cross-aim synthesis and claim boundaries",
    "## Historical and superseded-result ledger",
    "## Governed source index",
)

_REQUIRED_INHERITED_RESULT_MARKERS = (
    "Inherited E0 estimand",
    "inherited five-seed Virchow2-CLS gene-level reference",
    "Inherited FINAL-v10.5 separately trained within-source arms",
    "Inherited FINAL-v10.5 paired common-patient contrasts",
    "random/composition-restricted tests",
    "Cross-fitted routine-clinical comparator",
    "At 30% development worklist capacity",
    "E2a-F direction",
    "E2a-D sibling target",
    "Inherited paired contrast",
    "Five-seed family-naive target",
    "historical three-seed family-naive points",
    "E2-CPHT-A support/class",
    "Fixed UNI-v1 molecular contrast",
    "Three-draw repeated-control consensus",
    "Virchow2-CLS molecular contrast",
    "A abundance AUROC, q",
    "Held-out macro R-squared",
    "Human-machine concordance was 5 agree",
)
_RESULT_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:\+|-)?\d+(?:\.\d+)?(?:e[+-]?\d+)?%?",
    re.IGNORECASE,
)


def _result_number_tokens(text: str) -> set[str]:
    return set(_RESULT_NUMBER_RE.findall(text))


def _validate_documents(paths: BundlePaths, *, require_candidate: bool) -> dict[str, Any]:
    texts = {name: _document_text(paths, name) for name in REPORT_DOCUMENTS}
    results = texts["Results.md"]
    if not results.startswith("# FINAL-v11 comprehensive results"):
        raise BundleVerificationError("Results.md title is not exact")
    result_lines = results.splitlines()
    positions = []
    for heading in _REQUIRED_RESULT_HEADINGS:
        count = result_lines.count(heading)
        if count != 1:
            raise BundleVerificationError(f"Results.md must contain heading once: {heading}")
        positions.append(result_lines.index(heading))
    if positions != sorted(positions):
        raise BundleVerificationError("Results.md heading topology is out of order")
    required_literals = (
        "`e2a_s` is interpreted as **E1a-S**",
        "no governed E2a-S experiment exists",
        "E2-CPHT-R",
        "NOT_RUN",
        "GENERATED_UNREAD",
        "training_population",
        "evaluation_population",
        "Source-restricted OOF training",
    )
    for literal in required_literals:
        if literal not in results:
            raise BundleVerificationError(f"Results.md lacks required boundary: {literal}")
    for marker in _REQUIRED_INHERITED_RESULT_MARKERS:
        if results.count(marker) != 1:
            raise BundleVerificationError(
                f"Results.md inherited-result coverage marker is not exact: {marker}"
            )
    for row_id in (
        *EXPECTED_PERFORMANCE_CLAIM_ROW_IDS,
        *EXPECTED_CONTRAST_CLAIM_ROW_IDS,
        *EXPECTED_WHY_D_RECORD_IDS,
    ):
        if results.count(row_id) != 1:
            raise BundleVerificationError(
                f"Results.md claim-row placeholder/binding is not exact: {row_id}"
            )
    parent_results_path = paths.parent_dir / "Results.md"
    _reject_symlink_chain(parent_results_path, context="sealed FINAL-v10.5 Results.md")
    if not parent_results_path.is_file():
        raise BundleVerificationError("sealed FINAL-v10.5 Results.md is missing")
    parent_results = parent_results_path.read_text(encoding="utf-8")
    missing_parent_numbers = sorted(
        _result_number_tokens(parent_results) - _result_number_tokens(results)
    )
    if missing_parent_numbers:
        raise BundleVerificationError(
            "Results.md omits sealed FINAL-v10.5 numeric tokens: "
            + ", ".join(missing_parent_numbers)
        )
    setup = texts["Experimental_Setup.md"]
    audit = texts["Audit.md"]
    for name, text in (("Experimental_Setup.md", setup), ("Audit.md", audit)):
        for heading in ("## Aim 1", "## Aim 2", "## Aim 3", "## Aim 4"):
            if heading not in text:
                raise BundleVerificationError(f"{name} lacks {heading}")
    if "50" not in setup or "4,790" not in setup or "label-blind" not in setup:
        raise BundleVerificationError("Experimental_Setup.md lacks the zero-shot score contract")
    if (
        str(EXPECTED_PARENT_SOURCE_COUNT) not in audit
        or str(EXPECTED_NEW_SOURCE_COUNT) not in audit
        or str(EXPECTED_COMPLETE_SOURCE_COUNT) not in audit
    ):
        raise BundleVerificationError("Audit.md lacks the exact source census")
    if "35" not in audit or "60" not in audit or "six" not in audit.lower():
        raise BundleVerificationError("Audit.md lacks fit/concurrency accounting")

    if require_candidate:
        blockers = re.compile(r"\b(?:DRAFT|PENDING|AWAITING|UNSEALED)\b", re.IGNORECASE)
        for name, text in texts.items():
            if blockers.search(text):
                raise BundleVerificationError(
                    f"candidate document retains staging language: {name}"
                )
            if FINAL_STATE_STATUS_PARAGRAPH not in text:
                raise BundleVerificationError(
                    f"candidate document lacks final-state paragraph: {name}"
                )
        expected = paths.expected_document_sha256
        if not isinstance(expected, dict) or set(expected) != set(REPORT_DOCUMENTS):
            raise BundleVerificationError("final document pins are not exact")
        for name in REPORT_DOCUMENTS:
            pin = expected[name]
            if pin == _UNFROZEN or _SHA256_RE.fullmatch(str(pin)) is None:
                raise BundleVerificationError(f"FINAL-v11 document pin remains unfrozen: {name}")
            if sha256_file(paths.final_v11 / name) != pin:
                raise BundleVerificationError(f"FINAL-v11 document identity drift: {name}")
    return {
        name: identity(
            paths.final_v11 / name, display_path=_display(paths.final_v11 / name, paths.repo)
        )
        for name in REPORT_DOCUMENTS
    }


def _validate_terminal_pins(paths: BundlePaths, new_sources: list[dict[str, Any]]) -> None:
    expected = paths.expected_terminal_receipt_sha256
    if not isinstance(expected, dict) or set(expected) != set(EXPECTED_TERMINAL_RECEIPT_SHA256):
        raise BundleVerificationError("terminal receipt pin roster is not exact")
    by_id = _source_map(new_sources)
    for source_id, pin in expected.items():
        if pin == _UNFROZEN or _SHA256_RE.fullmatch(str(pin)) is None:
            raise BundleVerificationError(f"terminal receipt pin remains unfrozen: {source_id}")
        if by_id.get(source_id, {}).get("sha256") != pin:
            raise BundleVerificationError(f"terminal receipt byte pin drift: {source_id}")


def _validate_source_index(paths: BundlePaths, sources: list[dict[str, Any]]) -> None:
    audit = _document_text(paths, "Audit.md")
    rows = re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", audit, flags=re.MULTILINE)
    observed = {source_id: digest for source_id, digest in rows}
    expected = {str(source["id"]): str(source["sha256"]) for source in sources}
    if len(rows) != len(observed) or observed != expected:
        raise BundleVerificationError("Audit governed source index is not one-to-one and exact")


def draft_status(paths: BundlePaths | None = None) -> dict[str, Any]:
    selected = default_paths() if paths is None else paths
    manifest, parent_sources, new_sources, pending = _validate_manifest(
        selected, require_candidate=False
    )
    flat = set(manifest) == _FLAT_KEYS
    _validate_documents(selected, require_candidate=flat)
    if flat:
        _validate_terminal_pins(selected, new_sources)
        _validate_campaign(selected, new_sources)
        _validate_analysis(selected, new_sources)
        _validate_source_index(selected, [*parent_sources, *new_sources])
    ready: list[str] = []
    missing: list[str] = []
    for source in pending:
        path = _lexical_path(str(source["path"]), selected)
        (ready if path.is_file() and not path.is_symlink() else missing).append(str(source["path"]))
    return {
        "bundle": "reports/final_v11",
        "status": manifest["status"],
        "published_receipt_present": selected.destination.exists()
        or selected.destination.is_symlink(),
        "sealed_parent_source_count": len(parent_sources),
        "materialized_new_source_count": len(new_sources),
        "pending_new_source_count": len(pending),
        "pending_now_materialized_count": len(ready),
        "missing_source_count": len(missing),
        "pending_now_materialized": ready,
        "missing_sources": missing,
        "sealed_parent_fit_census": EXPECTED_PARENT_FIT_CENSUS,
        "pending_complete_fit_census": EXPECTED_COMPLETE_FIT_CENSUS,
        "campaign_lineage_fit_census": EXPECTED_CAMPAIGN_LINEAGE_FITS,
        "parent_recursive_verification": "PASS",
        "parent_direct_94_source_rehash": "PASS",
    }


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def refresh_manifest(paths: BundlePaths | None = None) -> dict[str, Any]:
    selected = default_paths() if paths is None else paths
    if selected.destination.exists() or selected.destination.is_symlink():
        raise BundleVerificationError("refusing manifest refresh after receipt publication")
    manifest, parent_sources, new_sources, pending = _validate_manifest(
        selected, require_candidate=False
    )
    if set(manifest) == _FLAT_KEYS:
        return draft_status(selected)
    promoted: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for source in pending:
        path = _lexical_path(str(source["path"]), selected)
        if path.is_file() and not path.is_symlink():
            promoted.append({**source, **identity(path, display_path=str(source["path"]))})
        else:
            remaining.append(source)
    if not promoted:
        return draft_status(selected)
    new_sources = [*new_sources, *promoted]
    if remaining:
        updated = {
            **manifest,
            "artifacts": sorted(new_sources, key=lambda item: str(item["id"])),
            "pending_artifacts": sorted(remaining, key=lambda item: str(item["id"])),
            "status": DRAFT_STATUS,
        }
    else:
        updated = {
            "schema_version": 2,
            "bundle": "final_v11",
            "status": CANDIDATE_STATUS,
            "artifacts": sorted([*parent_sources, *new_sources], key=lambda item: str(item["id"])),
            "pending_artifacts": [],
        }
        _validate_documents(selected, require_candidate=True)
        _validate_terminal_pins(selected, new_sources)
        _validate_campaign(selected, new_sources)
        _validate_analysis(selected, new_sources)
        _validate_source_index(selected, [*parent_sources, *new_sources])
    _atomic_replace(selected.final_v11 / SOURCE_MANIFEST_NAME, _manifest_bytes(updated))
    return draft_status(selected)


def _validated_created_utc(paths: BundlePaths) -> str:
    value = paths.expected_created_utc
    if paths.repo.resolve() == REPO.resolve() and value != EXPECTED_RECEIPT_CREATED_UTC:
        raise BundleVerificationError("production receipt timestamp was overridden")
    if not isinstance(value, str) or value == _UNFROZEN:
        raise BundleVerificationError("FINAL-v11 receipt timestamp remains unfrozen")
    parsed = _parse_utc(value, context="FINAL-v11 receipt timestamp")
    if parsed.isoformat() != value:
        raise BundleVerificationError("receipt timestamp must use canonical +00:00 form")
    return value


def build_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    selected = default_paths() if paths is None else paths
    manifest, parent_sources, new_sources, pending = _validate_manifest(
        selected, require_candidate=True
    )
    if pending:
        raise BundleVerificationError("candidate retained pending sources")
    documents = _validate_documents(selected, require_candidate=True)
    _validate_terminal_pins(selected, new_sources)
    _validate_campaign(selected, new_sources)
    _validate_analysis(selected, new_sources)
    all_sources = sorted([*parent_sources, *new_sources], key=lambda item: str(item["id"]))
    _validate_source_index(selected, all_sources)
    if len(all_sources) != EXPECTED_COMPLETE_SOURCE_COUNT:
        raise BundleVerificationError("complete FINAL-v11 source census is not exact")
    return {
        "schema_version": 1,
        "bundle": "reports/final_v11",
        "status": SEALED_STATUS,
        "created_utc": _validated_created_utc(selected),
        "organization": "additive_comprehensive_reorganization_over_final_v10_5",
        "base_final_v10_5": {
            "receipt": _manifest_identity_record(selected.parent_receipt, selected),
            "source_manifest": _manifest_identity_record(selected.parent_manifest, selected),
            "source_count": EXPECTED_PARENT_SOURCE_COUNT,
            "recursive_verification": "PASS",
            "direct_source_rehash": "PASS",
        },
        "fit_census": {
            "sealed_final_v10_5": EXPECTED_PARENT_FIT_CENSUS,
            "adopted_tcga_surgen_univ1_oof": 25,
            "new_virchow2_oof": 25,
            "new_full_source_refits": 10,
            "new_fits": EXPECTED_NEW_FITS,
            "tcga_surgen_campaign_lineage": EXPECTED_CAMPAIGN_LINEAGE_FITS,
            "complete_study_wide": EXPECTED_COMPLETE_FIT_CENSUS,
            "maximum_concurrent_trainers": 6,
            "observed_peak_concurrent_trainers": 6,
        },
        "score_census": {
            "label_blind_jobs": EXPECTED_SCORE_JOBS,
            "label_blind_slide_rows": EXPECTED_SCORE_SLIDE_ROWS,
            "encoders": list(EXPECTED_ENCODERS),
            "seeds": list(EXPECTED_SEEDS),
            "targets": list(EXPECTED_TARGETS),
            "outcome_join_after_seal": True,
        },
        "declared_nonresults": {
            "source_restricted_missing_cells": "NOT_RUN",
            "e2_cpht_r": "NOT_RUN",
            "whole_section_pathology": "GENERATED_UNREAD",
        },
        "documents": documents,
        "source_manifest": identity(
            selected.final_v11 / SOURCE_MANIFEST_NAME,
            display_path=_display(selected.final_v11 / SOURCE_MANIFEST_NAME, selected.repo),
        ),
        "authoritative_sources": all_sources,
        "verification": {
            "verifier": identity(
                selected.verifier_code, display_path=_display(selected.verifier_code, selected.repo)
            ),
            "tests": identity(
                selected.verifier_test, display_path=_display(selected.verifier_test, selected.repo)
            ),
            "frozen_parent_verifier": identity(
                selected.parent_verifier,
                display_path=_display(selected.parent_verifier, selected.repo),
            ),
            "training_controller": identity(
                selected.campaign_controller,
                display_path=_display(selected.campaign_controller, selected.repo),
            ),
            "analysis_controller": identity(
                selected.analysis_controller,
                display_path=_display(selected.analysis_controller, selected.repo),
            ),
        },
        "checks": {
            "sealed_final_v10_5_recursive_verification": "PASS",
            "direct_94_parent_source_rehash": "PASS",
            "exact_new_source_extension_inventory": "PASS",
            "strict_json_and_no_symlinks": "PASS",
            "one_to_one_source_sha_table": "PASS",
            "exact_35_new_60_lineage_fit_census": "PASS",
            "maximum_and_observed_concurrency_six": "PASS",
            "label_blind_score_seal_before_outcomes": "PASS",
            "exact_50_score_job_4790_row_census": "PASS",
            "estimand_boundaries": "PASS",
            "claim_row_binding": "PASS",
            "comprehensive_nonprefix_report_topology": "PASS",
            "final_document_byte_pins": "PASS",
            "terminal_receipt_byte_pins": "PASS",
        },
    }


def _receipt_bytes(receipt: dict[str, Any]) -> bytes:
    return (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def verify_published_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    selected = default_paths() if paths is None else paths
    _reject_symlink_chain(selected.destination, context="published receipt")
    if not selected.destination.is_file():
        raise BundleVerificationError("published FINAL-v11 receipt is absent")
    published = _load_json(selected.destination, label="published FINAL-v11 receipt")
    expected = build_receipt(selected)
    if published != expected or selected.destination.read_bytes() != _receipt_bytes(expected):
        raise BundleVerificationError("published FINAL-v11 receipt byte identity drift")
    return published


def seal(paths: BundlePaths | None = None) -> dict[str, Any]:
    selected = default_paths() if paths is None else paths
    if selected.destination.exists() or selected.destination.is_symlink():
        raise BundleVerificationError("refusing to overwrite FINAL-v11 receipt")
    receipt = build_receipt(selected)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{selected.destination.name}.", suffix=".tmp", dir=selected.destination.parent
    )
    temporary = Path(temporary_name)
    temporary_inode: tuple[int, int] | None = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_receipt_bytes(receipt))
            handle.flush()
            os.fsync(handle.fileno())
            stat = os.fstat(handle.fileno())
            temporary_inode = (stat.st_dev, stat.st_ino)
        try:
            os.link(temporary, selected.destination)
        except FileExistsError as exc:
            raise BundleVerificationError("receipt was concurrently published") from exc
        _fsync_directory(selected.destination.parent)
        temporary.unlink(missing_ok=True)
        return verify_published_receipt(selected)
    except BaseException:
        try:
            observed = selected.destination.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if temporary_inode == (observed.st_dev, observed.st_ino):
                with contextlib.suppress(FileNotFoundError):
                    selected.destination.unlink()
                with contextlib.suppress(OSError):
                    _fsync_directory(selected.destination.parent)
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--refresh-manifest", action="store_true")
    actions.add_argument(
        "--build-candidate", "--check-candidate", dest="build_candidate", action="store_true"
    )
    actions.add_argument("--verify-published", action="store_true")
    actions.add_argument("--seal", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.refresh_manifest:
            value = refresh_manifest()
        elif args.build_candidate:
            value = build_receipt()
        elif args.verify_published:
            value = verify_published_receipt()
        elif args.seal:
            value = seal()
        else:
            value = draft_status()
    except BundleVerificationError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
